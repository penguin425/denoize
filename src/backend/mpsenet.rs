//! MP-SENet adapter matching the official VoiceBank+DEMAND inference frontend.
//!
//! The converted ONNX graph receives compressed magnitude and phase tensors
//! shaped `[1, 201, frames]` and returns enhanced tensors with the same shapes.
//! STFT, normalization, phase reconstruction, and duration preservation happen
//! in Rust so the runtime does not depend on Python or PyTorch.

use super::{onnx::resample_linear, OnnxModelConfig};
use rustfft::{num_complex::Complex32, FftPlanner};
use tract_onnx::prelude::*;

const MODEL_RATE: u32 = 16_000;
const FFT_SIZE: usize = 400;
const HOP_SIZE: usize = 100;
const BINS: usize = FFT_SIZE / 2 + 1;
const COMPRESS_FACTOR: f32 = 0.3;
const MODEL_SAMPLES: usize = 32_000;
const MODEL_FRAMES: usize = MODEL_SAMPLES / HOP_SIZE + 1;
const CHUNK_HOP: usize = MODEL_SAMPLES / 2;

pub fn process(
    channels: &[Vec<f64>],
    input_sample_rate: u32,
    config: &OnnxModelConfig,
) -> Result<Vec<Vec<f64>>, String> {
    if config.sample_rate != MODEL_RATE {
        return Err(format!(
            "MP-SENet expects a {MODEL_RATE} Hz model, got {} Hz",
            config.sample_rate
        ));
    }
    if !config.path.is_file() {
        return Err(format!(
            "MP-SENet ONNX model does not exist or is not a file: {}",
            config.path.display()
        ));
    }
    let model = load_model(config)?;
    channels
        .iter()
        .map(|channel| process_channel(channel, input_sample_rate, &model))
        .collect()
}

fn process_channel(
    input: &[f64],
    input_sample_rate: u32,
    model: &TypedRunnableModel<TypedModel>,
) -> Result<Vec<f64>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let at_model_rate = resample_linear(input, input_sample_rate, MODEL_RATE);
    let energy: f64 = at_model_rate.iter().map(|sample| sample * sample).sum();
    if energy <= f64::EPSILON {
        return Ok(vec![0.0; input.len()]);
    }
    let normalization = (at_model_rate.len() as f64 / energy).sqrt();
    let normalized: Vec<f32> = at_model_rate
        .iter()
        .map(|sample| (sample * normalization) as f32)
        .collect();
    let reconstructed = enhance_chunks(&normalized, model)?;
    let denormalized: Vec<f64> = reconstructed
        .iter()
        .map(|sample| *sample as f64 / normalization)
        .collect();
    let at_input_rate = resample_linear(&denormalized, MODEL_RATE, input_sample_rate);
    let mut output = at_input_rate;
    output.truncate(input.len());
    output.resize(input.len(), 0.0);
    Ok(output)
}

fn enhance_chunks(
    input: &[f32],
    model: &TypedRunnableModel<TypedModel>,
) -> Result<Vec<f32>, String> {
    let mut output = vec![0.0f32; input.len()];
    let mut envelope = vec![0.0f32; input.len()];
    let mut start = 0;
    while start < input.len() {
        let copy_length = (input.len() - start).min(MODEL_SAMPLES);
        let mut chunk = vec![0.0f32; MODEL_SAMPLES];
        chunk[..copy_length].copy_from_slice(&input[start..start + copy_length]);
        let spectrum = stft(&chunk);
        debug_assert_eq!(spectrum.frames, MODEL_FRAMES);
        let (enhanced_magnitude, enhanced_phase) =
            run_model(&spectrum.magnitude, &spectrum.phase, spectrum.frames, model)?;
        let enhanced = istft(
            &enhanced_magnitude,
            &enhanced_phase,
            spectrum.frames,
            MODEL_SAMPLES,
        )?;

        let has_previous = start > 0;
        let has_next = start + MODEL_SAMPLES < input.len();
        for index in 0..copy_length {
            let mut weight = 1.0;
            if has_previous && index < CHUNK_HOP {
                let angle = std::f32::consts::FRAC_PI_2 * index as f32 / CHUNK_HOP as f32;
                weight *= angle.sin().powi(2);
            }
            if has_next && index >= MODEL_SAMPLES - CHUNK_HOP {
                let offset = index - (MODEL_SAMPLES - CHUNK_HOP);
                let angle = std::f32::consts::FRAC_PI_2 * offset as f32 / CHUNK_HOP as f32;
                weight *= angle.cos().powi(2);
            }
            output[start + index] += enhanced[index] * weight;
            envelope[start + index] += weight;
        }
        start += CHUNK_HOP;
    }
    for (sample, weight) in output.iter_mut().zip(envelope) {
        if weight > 1e-8 {
            *sample /= weight;
        }
    }
    Ok(output)
}

struct Spectrum {
    magnitude: Vec<f32>,
    phase: Vec<f32>,
    frames: usize,
}

fn stft(input: &[f32]) -> Spectrum {
    let pad = FFT_SIZE / 2;
    let padded: Vec<f32> = (0..input.len() + 2 * pad)
        .map(|index| input[reflect_index(index as isize - pad as isize, input.len())])
        .collect();
    let frames = 1 + (padded.len() - FFT_SIZE) / HOP_SIZE;
    let window = periodic_hann();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut magnitude = vec![0.0; BINS * frames];
    let mut phase = vec![0.0; BINS * frames];
    let mut buffer = vec![Complex32::default(); FFT_SIZE];

    for frame in 0..frames {
        let start = frame * HOP_SIZE;
        for index in 0..FFT_SIZE {
            buffer[index] = Complex32::new(padded[start + index] * window[index], 0.0);
        }
        fft.process(&mut buffer);
        for bin in 0..BINS {
            let value = buffer[bin];
            let offset = bin * frames + frame;
            magnitude[offset] = (value.norm_sqr() + 1e-9).sqrt().powf(COMPRESS_FACTOR);
            phase[offset] = (value.im + 1e-10).atan2(value.re + 1e-5);
        }
    }
    Spectrum {
        magnitude,
        phase,
        frames,
    }
}

fn istft(
    magnitude: &[f32],
    phase: &[f32],
    frames: usize,
    output_length: usize,
) -> Result<Vec<f32>, String> {
    let expected = BINS * frames;
    if magnitude.len() != expected || phase.len() != expected {
        return Err("MP-SENet output tensor has an unexpected size".into());
    }
    let window = periodic_hann();
    let padded_length = (frames - 1) * HOP_SIZE + FFT_SIZE;
    let mut signal = vec![0.0f32; padded_length];
    let mut envelope = vec![0.0f32; padded_length];
    let mut planner = FftPlanner::new();
    let inverse = planner.plan_fft_inverse(FFT_SIZE);
    let mut buffer = vec![Complex32::default(); FFT_SIZE];

    for frame in 0..frames {
        for bin in 0..BINS {
            let offset = bin * frames + frame;
            let amplitude = magnitude[offset].max(0.0).powf(1.0 / COMPRESS_FACTOR);
            buffer[bin] = Complex32::from_polar(amplitude, phase[offset]);
        }
        for bin in BINS..FFT_SIZE {
            buffer[bin] = buffer[FFT_SIZE - bin].conj();
        }
        inverse.process(&mut buffer);
        let start = frame * HOP_SIZE;
        for index in 0..FFT_SIZE {
            let windowed = buffer[index].re / FFT_SIZE as f32 * window[index];
            signal[start + index] += windowed;
            envelope[start + index] += window[index] * window[index];
        }
    }
    for (sample, weight) in signal.iter_mut().zip(envelope) {
        if weight > 1e-8 {
            *sample /= weight;
        }
    }

    let pad = FFT_SIZE / 2;
    let available = signal.len().saturating_sub(2 * pad);
    let copy_length = output_length.min(available);
    let mut output = signal[pad..pad + copy_length].to_vec();
    output.resize(output_length, 0.0);
    if output.iter().any(|sample| !sample.is_finite()) {
        return Err("MP-SENet reconstruction produced a non-finite sample".into());
    }
    Ok(output)
}

fn load_model(config: &OnnxModelConfig) -> Result<TypedRunnableModel<TypedModel>, String> {
    let mut model = tract_onnx::onnx()
        .model_for_path(&config.path)
        .map_err(|error| model_error("load", error))?;
    if model
        .input_outlets()
        .map_err(|e| model_error("inspect", e))?
        .len()
        != 2
        || model
            .output_outlets()
            .map_err(|e| model_error("inspect", e))?
            .len()
            != 2
    {
        return Err("MP-SENet ONNX model must have two inputs and two outputs".into());
    }
    let shape = tvec!(1, BINS, MODEL_FRAMES);
    for input_index in 0..2 {
        model
            .set_input_fact(input_index, f32::fact(shape.clone()).into())
            .map_err(|error| model_error("configure input", error))?;
    }
    for output_index in 0..2 {
        model
            .set_output_fact(output_index, f32::fact(shape.clone()).into())
            .map_err(|error| model_error("configure output", error))?;
    }
    model
        .into_optimized()
        .and_then(|model| model.into_runnable())
        .map_err(|error| model_error("optimize", error))
}

fn run_model(
    magnitude: &[f32],
    phase: &[f32],
    frames: usize,
    model: &TypedRunnableModel<TypedModel>,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if frames != MODEL_FRAMES {
        return Err(format!(
            "MP-SENet requires {MODEL_FRAMES} spectral frames per chunk, got {frames}"
        ));
    }
    let shape = tvec!(1, BINS, frames);
    let magnitude_tensor = Tensor::from_shape(&shape, magnitude)
        .map_err(|error| model_error("create magnitude tensor", error))?;
    let phase_tensor = Tensor::from_shape(&shape, phase)
        .map_err(|error| model_error("create phase tensor", error))?;
    let outputs = model
        .run(tvec!(
            magnitude_tensor.into_tvalue(),
            phase_tensor.into_tvalue()
        ))
        .map_err(|error| model_error("run", error))?;
    let enhanced_magnitude = tensor_samples(&outputs[0], magnitude.len(), "magnitude")?;
    let enhanced_phase = tensor_samples(&outputs[1], phase.len(), "phase")?;
    Ok((enhanced_magnitude, enhanced_phase))
}

fn tensor_samples(tensor: &TValue, expected: usize, name: &str) -> Result<Vec<f32>, String> {
    let view = tensor
        .to_array_view::<f32>()
        .map_err(|error| model_error(&format!("read {name} output"), error))?;
    if view.len() != expected {
        return Err(format!(
            "MP-SENet {name} output has {} values; expected {expected}",
            view.len()
        ));
    }
    let values: Vec<f32> = view.iter().copied().collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "MP-SENet {name} output contains a non-finite value"
        ));
    }
    Ok(values)
}

fn model_error(stage: &str, error: impl std::fmt::Display) -> String {
    format!("MP-SENet ONNX {stage} failed: {error:#}")
}

fn periodic_hann() -> Vec<f32> {
    (0..FFT_SIZE)
        .map(|index| {
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / FFT_SIZE as f32).cos()
        })
        .collect()
}

fn reflect_index(mut index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let last = length as isize - 1;
    while index < 0 || index > last {
        if index < 0 {
            index = -index;
        }
        if index > last {
            index = 2 * last - index;
        }
    }
    index as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use tract_onnx::pb::{
        tensor_proto, tensor_shape_proto, type_proto, GraphProto, ModelProto, NodeProto,
        OperatorSetIdProto, TensorShapeProto, TypeProto, ValueInfoProto,
    };

    #[test]
    fn stft_identity_reconstruction_is_transparent() {
        let input: Vec<f32> = (0..16_000)
            .map(|index| {
                (2.0 * std::f32::consts::PI * 440.0 * index as f32 / MODEL_RATE as f32).sin() * 0.25
            })
            .collect();
        let spectrum = stft(&input);
        let output = istft(
            &spectrum.magnitude,
            &spectrum.phase,
            spectrum.frames,
            input.len(),
        )
        .unwrap();
        let mse = input
            .iter()
            .zip(output)
            .map(|(expected, actual)| (expected - actual).powi(2))
            .sum::<f32>()
            / input.len() as f32;
        assert!(mse < 1e-8, "identity STFT MSE was {mse}");
    }

    #[test]
    fn reflection_matches_torch_style_edges() {
        let mapped: Vec<usize> = (-3..8).map(|index| reflect_index(index, 5)).collect();
        assert_eq!(mapped, vec![3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1]);
    }

    #[test]
    fn spectral_identity_model_runs_end_to_end() {
        let mut bytes = Vec::new();
        spectral_identity_model().encode(&mut bytes).unwrap();
        let path = std::env::temp_dir().join(format!(
            "denoize-mpsenet-identity-{}-{}.onnx",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, bytes).unwrap();
        let config = OnnxModelConfig {
            path: path.clone(),
            sample_rate: MODEL_RATE,
        };
        let input: Vec<f64> = (0..64_000)
            .map(|index| {
                (2.0 * std::f64::consts::PI * 440.0 * index as f64 / MODEL_RATE as f64).sin() * 0.1
            })
            .collect();
        let output = process(&[input.clone()], MODEL_RATE, &config).unwrap();
        std::fs::remove_file(path).unwrap();

        let mse = input
            .iter()
            .zip(&output[0])
            .map(|(expected, actual)| (expected - actual).powi(2))
            .sum::<f64>()
            / input.len() as f64;
        assert_eq!(output[0].len(), input.len());
        assert!(mse < 1e-9, "spectral identity model MSE was {mse}");
    }

    fn spectral_identity_model() -> ModelProto {
        let value_info = |name: &str| ValueInfoProto {
            name: name.into(),
            r#type: Some(TypeProto {
                denotation: String::new(),
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: tensor_proto::DataType::Float as i32,
                    shape: Some(TensorShapeProto {
                        dim: vec![
                            dimension_value(1),
                            dimension_value(BINS as i64),
                            dimension_parameter("frames"),
                        ],
                    }),
                })),
            }),
            doc_string: String::new(),
        };
        ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            producer_name: "denoize-test".into(),
            graph: Some(GraphProto {
                name: "mp-senet-spectral-identity".into(),
                node: vec![
                    identity_node("magnitude", "enhanced_magnitude", "magnitude_identity"),
                    identity_node("phase", "enhanced_phase", "phase_identity"),
                ],
                input: vec![value_info("magnitude"), value_info("phase")],
                output: vec![
                    value_info("enhanced_magnitude"),
                    value_info("enhanced_phase"),
                ],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn identity_node(input: &str, output: &str, name: &str) -> NodeProto {
        NodeProto {
            input: vec![input.into()],
            output: vec![output.into()],
            name: name.into(),
            op_type: "Identity".into(),
            ..Default::default()
        }
    }

    fn dimension_value(value: i64) -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(value)),
            denotation: String::new(),
        }
    }

    fn dimension_parameter(name: &str) -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(name.into())),
            denotation: String::new(),
        }
    }
}
