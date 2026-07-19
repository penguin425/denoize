//! ESPnet BSRNN adapter for the pinned VCTK+DEMAND xtiny checkpoint.
//!
//! The converted ONNX graph receives a real/imaginary spectrum shaped
//! `[1, frames, 481, 2]`. Rust reproduces ESPnet's variance normalization,
//! centered periodic-Hann 960-point STFT, whole-utterance inference, inverse
//! STFT, sample-rate conversion, and exact duration restoration.

use super::{onnx::resample_linear, OnnxModelConfig};
use rustfft::{num_complex::Complex32, FftPlanner};
use tract_onnx::prelude::*;

const MODEL_RATE: u32 = 48_000;
const FFT_SIZE: usize = 960;
const HOP_SIZE: usize = 480;
const BINS: usize = FFT_SIZE / 2 + 1;

pub fn process(
    channels: &[Vec<f64>],
    input_sample_rate: u32,
    config: &OnnxModelConfig,
) -> Result<Vec<Vec<f64>>, String> {
    if config.sample_rate != MODEL_RATE {
        return Err(format!(
            "BSRNN expects a {MODEL_RATE} Hz model, got {} Hz",
            config.sample_rate
        ));
    }
    if !config.path.is_file() {
        return Err(format!(
            "BSRNN ONNX model does not exist or is not a file: {}",
            config.path.display()
        ));
    }
    if channels.is_empty() {
        return Ok(Vec::new());
    }

    let model_samples = resample_linear(&channels[0], input_sample_rate, MODEL_RATE).len();
    if model_samples == 0 {
        return Ok(channels.iter().map(|_| Vec::new()).collect());
    }
    let frames = model_samples / HOP_SIZE + 1;
    let model = load_model(config, frames)?;
    channels
        .iter()
        .map(|channel| process_channel(channel, input_sample_rate, frames, &model))
        .collect()
}

fn process_channel(
    input: &[f64],
    input_sample_rate: u32,
    expected_frames: usize,
    model: &TypedRunnableModel<TypedModel>,
) -> Result<Vec<f64>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let at_model_rate = resample_linear(input, input_sample_rate, MODEL_RATE);
    let mean = at_model_rate.iter().sum::<f64>() / at_model_rate.len() as f64;
    let variance = if at_model_rate.len() > 1 {
        at_model_rate
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / (at_model_rate.len() - 1) as f64
    } else {
        0.0
    };
    let standard_deviation = variance.sqrt();
    if standard_deviation <= f64::EPSILON {
        return Ok(vec![0.0; input.len()]);
    }
    let normalized: Vec<f32> = at_model_rate
        .iter()
        .map(|sample| (*sample / standard_deviation) as f32)
        .collect();
    let spectrum = stft(&normalized);
    if spectrum.frames != expected_frames {
        return Err(format!(
            "BSRNN channel produced {} frames; expected {expected_frames}",
            spectrum.frames
        ));
    }
    let enhanced_spectrum = run_model(&spectrum.values, spectrum.frames, model)?;
    let reconstructed = istft(&enhanced_spectrum, spectrum.frames, normalized.len())?;
    let denormalized: Vec<f64> = reconstructed
        .iter()
        .map(|sample| *sample as f64 * standard_deviation)
        .collect();
    let mut output = resample_linear(&denormalized, MODEL_RATE, input_sample_rate);
    output.truncate(input.len());
    output.resize(input.len(), 0.0);
    Ok(output)
}

struct Spectrum {
    values: Vec<f32>,
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
    let mut values = vec![0.0; frames * BINS * 2];
    let mut buffer = vec![Complex32::default(); FFT_SIZE];
    for frame in 0..frames {
        let start = frame * HOP_SIZE;
        for index in 0..FFT_SIZE {
            buffer[index] = Complex32::new(padded[start + index] * window[index], 0.0);
        }
        fft.process(&mut buffer);
        for bin in 0..BINS {
            let offset = (frame * BINS + bin) * 2;
            values[offset] = buffer[bin].re;
            values[offset + 1] = buffer[bin].im;
        }
    }
    Spectrum { values, frames }
}

fn istft(spectrum: &[f32], frames: usize, output_length: usize) -> Result<Vec<f32>, String> {
    if spectrum.len() != frames * BINS * 2 {
        return Err("BSRNN output tensor has an unexpected size".into());
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
            let offset = (frame * BINS + bin) * 2;
            buffer[bin] = Complex32::new(spectrum[offset], spectrum[offset + 1]);
        }
        for bin in BINS..FFT_SIZE {
            buffer[bin] = buffer[FFT_SIZE - bin].conj();
        }
        inverse.process(&mut buffer);
        let start = frame * HOP_SIZE;
        for index in 0..FFT_SIZE {
            signal[start + index] += buffer[index].re / FFT_SIZE as f32 * window[index];
            envelope[start + index] += window[index] * window[index];
        }
    }
    for (sample, weight) in signal.iter_mut().zip(envelope) {
        if weight > 1e-8 {
            *sample /= weight;
        }
    }
    let pad = FFT_SIZE / 2;
    // ESPnet/PyTorch iSTFT receives the original signal length explicitly. A
    // final partial hop is reconstructed from the last centered frame, so only
    // the leading center pad limits this crop.
    let available = signal.len().saturating_sub(pad);
    let copy_length = output_length.min(available);
    let mut output = signal[pad..pad + copy_length].to_vec();
    output.resize(output_length, 0.0);
    if output.iter().any(|sample| !sample.is_finite()) {
        return Err("BSRNN reconstruction produced a non-finite sample".into());
    }
    Ok(output)
}

fn load_model(
    config: &OnnxModelConfig,
    frames: usize,
) -> Result<TypedRunnableModel<TypedModel>, String> {
    let shape = tvec!(1, frames, BINS, 2);
    let mut model = tract_onnx::onnx()
        .model_for_path(&config.path)
        .map_err(|error| model_error("load", error))?;
    if model
        .input_outlets()
        .map_err(|e| model_error("inspect", e))?
        .len()
        != 1
        || model
            .output_outlets()
            .map_err(|e| model_error("inspect", e))?
            .len()
            != 1
    {
        return Err("BSRNN ONNX model must have one input and one output".into());
    }
    model
        .set_input_fact(0, f32::fact(shape.clone()).into())
        .map_err(|error| model_error("configure input", error))?;
    model
        .set_output_fact(0, f32::fact(shape).into())
        .map_err(|error| model_error("configure output", error))?;
    model
        .into_optimized()
        .and_then(|model| model.into_runnable())
        .map_err(|error| model_error("optimize", error))
}

fn run_model(
    spectrum: &[f32],
    frames: usize,
    model: &TypedRunnableModel<TypedModel>,
) -> Result<Vec<f32>, String> {
    let shape = tvec!(1, frames, BINS, 2);
    let tensor = Tensor::from_shape(&shape, spectrum)
        .map_err(|error| model_error("create spectrum tensor", error))?;
    let outputs = model
        .run(tvec!(tensor.into_tvalue()))
        .map_err(|error| model_error("run", error))?;
    let view = outputs[0]
        .to_array_view::<f32>()
        .map_err(|error| model_error("read output", error))?;
    if view.len() != spectrum.len() {
        return Err(format!(
            "BSRNN output has {} values; expected {}",
            view.len(),
            spectrum.len()
        ));
    }
    let values: Vec<f32> = view.iter().copied().collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("BSRNN output contains a non-finite value".into());
    }
    Ok(values)
}

fn model_error(stage: &str, error: impl std::fmt::Display) -> String {
    format!("BSRNN ONNX {stage} failed: {error:#}")
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
        let input: Vec<f32> = (0..32_000)
            .map(|index| {
                (2.0 * std::f32::consts::PI * 440.0 * index as f32 / MODEL_RATE as f32).sin() * 0.25
            })
            .collect();
        let spectrum = stft(&input);
        assert_eq!(spectrum.frames, 67);
        let output = istft(&spectrum.values, spectrum.frames, input.len()).unwrap();
        let mse = input
            .iter()
            .zip(&output)
            .map(|(expected, actual)| (expected - actual).powi(2))
            .sum::<f32>()
            / input.len() as f32;
        assert!(mse < 1e-8, "identity STFT MSE was {mse}");
    }

    #[test]
    fn torch_style_variance_uses_bessel_correction() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mean = input.iter().sum::<f64>() / input.len() as f64;
        let variance =
            input.iter().map(|x| (*x - mean).powi(2)).sum::<f64>() / (input.len() - 1) as f64;
        assert!((variance.sqrt() - 1.290_994_448_735_805_6).abs() < 1e-12);
    }

    #[test]
    fn spectral_identity_model_runs_end_to_end() {
        let mut bytes = Vec::new();
        spectral_identity_model().encode(&mut bytes).unwrap();
        let path = std::env::temp_dir().join(format!(
            "denoize-bsrnn-identity-{}-{}.onnx",
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
        let input: Vec<f64> = (0..32_000)
            .map(|index| {
                0.1 * (2.0 * std::f64::consts::PI * 440.0 * index as f64 / MODEL_RATE as f64).sin()
                    + 0.02
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
        assert!(mse < 1e-10, "spectral identity model MSE was {mse}");
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
                            dimension_parameter("frames"),
                            dimension_value(BINS as i64),
                            dimension_value(2),
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
                name: "bsrnn-spectral-identity".into(),
                node: vec![NodeProto {
                    input: vec!["spectrum".into()],
                    output: vec!["enhanced_spectrum".into()],
                    name: "identity".into(),
                    op_type: "Identity".into(),
                    ..Default::default()
                }],
                input: vec![value_info("spectrum")],
                output: vec![value_info("enhanced_spectrum")],
                ..Default::default()
            }),
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
