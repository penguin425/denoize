//! ClearerVoice MossFormer2 48 kHz speech-enhancement adapter.
//!
//! The converted graph maps `[1, 496, 180]` Kaldi fbank/delta features to a
//! real-valued `[1, 496, 961]` spectral mask.  This module reproduces the
//! official four-second segmentation, 40 ms/8 ms frontend, non-centred
//! symmetric-Hamming STFT, mask application, and edge-discard stitching.

use super::OnnxModelConfig;
use kaldi_native_fbank::{
    mel::MelOptions, FbankComputer, FbankOptions, FrameOptions, OnlineFeature,
};
use rustfft::{num_complex::Complex32, FftPlanner};
use tract_onnx::prelude::*;

const MODEL_RATE: u32 = 48_000;
const WINDOW_SAMPLES: usize = 192_000;
const STRIDE_SAMPLES: usize = 144_000;
const GIVE_UP_SAMPLES: usize = 24_000;
const FFT_SIZE: usize = 1_920;
const HOP_SIZE: usize = 384;
const FRAMES: usize = 496;
const BINS: usize = FFT_SIZE / 2 + 1;
const MEL_BINS: usize = 60;
const FEATURES: usize = MEL_BINS * 3;

pub fn process(
    channels: &[Vec<f64>],
    input_sample_rate: u32,
    config: &OnnxModelConfig,
) -> Result<Vec<Vec<f64>>, String> {
    if config.sample_rate != MODEL_RATE {
        return Err(format!(
            "MossFormer2 expects a {MODEL_RATE} Hz model, got {} Hz",
            config.sample_rate
        ));
    }
    if !config.path.is_file() {
        return Err(format!(
            "MossFormer2 ONNX model does not exist or is not a file: {}",
            config.path.display()
        ));
    }
    if channels.is_empty() {
        return Ok(Vec::new());
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
    let at_model_rate = crate::resample::resample(input, input_sample_rate, MODEL_RATE)?;
    let original_model_length = at_model_rate.len();
    let padded_length = segmentation_length(original_model_length);
    let mut padded = at_model_rate;
    padded.resize(padded_length, 0.0);
    let mut enhanced = vec![0.0f32; padded_length];

    let mut start = 0;
    while start + WINDOW_SAMPLES <= padded_length {
        let segment: Vec<f32> = padded[start..start + WINDOW_SAMPLES]
            .iter()
            .map(|sample| (*sample * 32_768.0) as f32)
            .collect();
        let output = enhance_segment(&segment, model)?;
        let (source_start, source_end, target_start) = if start == 0 {
            (0, WINDOW_SAMPLES - GIVE_UP_SAMPLES, start)
        } else {
            (
                GIVE_UP_SAMPLES,
                WINDOW_SAMPLES - GIVE_UP_SAMPLES,
                start + GIVE_UP_SAMPLES,
            )
        };
        enhanced[target_start..target_start + source_end - source_start]
            .copy_from_slice(&output[source_start..source_end]);
        start += STRIDE_SAMPLES;
    }

    let enhanced: Vec<f64> = enhanced[..original_model_length]
        .iter()
        .map(|sample| *sample as f64 / 32_768.0)
        .collect();
    let mut output = crate::resample::resample(&enhanced, MODEL_RATE, input_sample_rate)?;
    output.truncate(input.len());
    output.resize(input.len(), 0.0);
    Ok(output)
}

fn segmentation_length(input_length: usize) -> usize {
    if input_length <= WINDOW_SAMPLES {
        return WINDOW_SAMPLES;
    }
    let extra = input_length - WINDOW_SAMPLES;
    WINDOW_SAMPLES + extra.div_ceil(STRIDE_SAMPLES) * STRIDE_SAMPLES
}

fn enhance_segment(
    samples: &[f32],
    model: &TypedRunnableModel<TypedModel>,
) -> Result<Vec<f32>, String> {
    let features = fbank_with_deltas(samples)?;
    let mask = run_model(&features, model)?;
    let spectrum = stft(samples);
    let masked: Vec<Complex32> = spectrum
        .into_iter()
        .zip(mask)
        .map(|(value, gain)| value * gain)
        .collect();
    istft(&masked, samples.len())
}

fn fbank_with_deltas(samples: &[f32]) -> Result<Vec<f32>, String> {
    let frame_opts = FrameOptions {
        samp_freq: MODEL_RATE as f32,
        frame_shift_ms: 8.0,
        frame_length_ms: 40.0,
        // Upstream requests one PCM-unit of random dither. Deployment uses
        // zero dither so identical audio has deterministic model features.
        dither: 0.0,
        preemph_coeff: 0.97,
        remove_dc_offset: true,
        window_type: "hamming".into(),
        round_to_power_of_two: true,
        blackman_coeff: 0.42,
        snip_edges: true,
    };
    let mut mel_opts = MelOptions::default();
    mel_opts.num_bins = MEL_BINS;
    let options = FbankOptions {
        frame_opts,
        mel_opts,
        use_energy: false,
        raw_energy: true,
        htk_compat: false,
        energy_floor: 1.0,
        use_log_fbank: true,
        use_power: true,
    };
    let computer = FbankComputer::new(options)
        .map_err(|error| format!("MossFormer2 fbank setup failed: {error}"))?;
    let mut online =
        OnlineFeature::new(kaldi_native_fbank::online::FeatureComputer::Fbank(computer));
    online.accept_waveform(MODEL_RATE as f32, samples);
    online.input_finished();
    if online.num_frames_ready() != FRAMES {
        return Err(format!(
            "MossFormer2 frontend produced {} frames; expected {FRAMES}",
            online.num_frames_ready()
        ));
    }
    let base: Vec<f32> = online.features.into_iter().flatten().collect();
    let delta = deltas(&base, FRAMES, MEL_BINS);
    let delta_delta = deltas(&delta, FRAMES, MEL_BINS);
    let mut result = vec![0.0; FRAMES * FEATURES];
    for frame in 0..FRAMES {
        let output = &mut result[frame * FEATURES..(frame + 1) * FEATURES];
        output[..MEL_BINS].copy_from_slice(&base[frame * MEL_BINS..(frame + 1) * MEL_BINS]);
        output[MEL_BINS..2 * MEL_BINS]
            .copy_from_slice(&delta[frame * MEL_BINS..(frame + 1) * MEL_BINS]);
        output[2 * MEL_BINS..]
            .copy_from_slice(&delta_delta[frame * MEL_BINS..(frame + 1) * MEL_BINS]);
    }
    Ok(result)
}

fn deltas(input: &[f32], frames: usize, bins: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for frame in 0..frames {
        for bin in 0..bins {
            let mut numerator = 0.0;
            for distance in 1..=2 {
                let before = frame.saturating_sub(distance);
                let after = (frame + distance).min(frames - 1);
                numerator +=
                    distance as f32 * (input[after * bins + bin] - input[before * bins + bin]);
            }
            output[frame * bins + bin] = numerator / 10.0;
        }
    }
    output
}

fn symmetric_hamming() -> Vec<f32> {
    (0..FFT_SIZE)
        .map(|index| {
            0.54 - 0.46 * (2.0 * std::f32::consts::PI * index as f32 / (FFT_SIZE - 1) as f32).cos()
        })
        .collect()
}

fn stft(input: &[f32]) -> Vec<Complex32> {
    let window = symmetric_hamming();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut output = vec![Complex32::default(); FRAMES * BINS];
    let mut buffer = vec![Complex32::default(); FFT_SIZE];
    for frame in 0..FRAMES {
        let start = frame * HOP_SIZE;
        for index in 0..FFT_SIZE {
            buffer[index] = Complex32::new(input[start + index] * window[index], 0.0);
        }
        fft.process(&mut buffer);
        output[frame * BINS..(frame + 1) * BINS].copy_from_slice(&buffer[..BINS]);
    }
    output
}

fn istft(spectrum: &[Complex32], output_length: usize) -> Result<Vec<f32>, String> {
    if spectrum.len() != FRAMES * BINS {
        return Err("MossFormer2 mask has an unexpected spectrum size".into());
    }
    let window = symmetric_hamming();
    let reconstructed_length = (FRAMES - 1) * HOP_SIZE + FFT_SIZE;
    let mut signal = vec![0.0f32; reconstructed_length];
    let mut envelope = vec![0.0f32; reconstructed_length];
    let mut planner = FftPlanner::new();
    let inverse = planner.plan_fft_inverse(FFT_SIZE);
    let mut buffer = vec![Complex32::default(); FFT_SIZE];
    for frame in 0..FRAMES {
        buffer[..BINS].copy_from_slice(&spectrum[frame * BINS..(frame + 1) * BINS]);
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
    signal.resize(output_length, 0.0);
    if signal.iter().any(|sample| !sample.is_finite()) {
        return Err("MossFormer2 reconstruction produced a non-finite sample".into());
    }
    Ok(signal)
}

fn load_model(config: &OnnxModelConfig) -> Result<TypedRunnableModel<TypedModel>, String> {
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
        return Err("MossFormer2 ONNX model must have one input and one output".into());
    }
    model
        .set_input_fact(0, f32::fact(tvec!(1, FRAMES, FEATURES)).into())
        .map_err(|error| model_error("configure input", error))?;
    model
        .set_output_fact(0, f32::fact(tvec!(1, FRAMES, BINS)).into())
        .map_err(|error| model_error("configure output", error))?;
    model
        .into_optimized()
        .and_then(|model| model.into_runnable())
        .map_err(|error| model_error("optimize", error))
}

fn run_model(features: &[f32], model: &TypedRunnableModel<TypedModel>) -> Result<Vec<f32>, String> {
    let input = Tensor::from_shape(&[1, FRAMES, FEATURES], features)
        .map_err(|error| model_error("create feature tensor", error))?;
    let outputs = model
        .run(tvec!(input.into_tvalue()))
        .map_err(|error| model_error("run", error))?;
    let output = outputs[0]
        .as_slice::<f32>()
        .map_err(|error| model_error("read output", error))?;
    if output.len() != FRAMES * BINS || output.iter().any(|value| !value.is_finite()) {
        return Err("MossFormer2 model returned an invalid mask".into());
    }
    Ok(output.to_vec())
}

fn model_error(stage: &str, error: impl std::fmt::Display) -> String {
    format!("MossFormer2 ONNX {stage} failed: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_window_has_496_frames() {
        assert_eq!(1 + (WINDOW_SAMPLES - FFT_SIZE) / HOP_SIZE, FRAMES);
    }

    #[test]
    fn segmentation_covers_the_requested_duration() {
        for length in [1, WINDOW_SAMPLES, WINDOW_SAMPLES + 1, 1_000_000] {
            let padded = segmentation_length(length);
            assert!(padded >= length);
            assert_eq!((padded - WINDOW_SAMPLES) % STRIDE_SAMPLES, 0);
        }
    }

    #[test]
    fn deltas_replicate_boundary_frames() {
        let input = vec![0.0, 1.0, 2.0, 3.0];
        let actual = deltas(&input, 4, 1);
        assert_eq!(actual, vec![0.5, 0.8, 0.8, 0.5]);
    }

    #[test]
    fn stft_identity_reconstruction_is_transparent() {
        let input: Vec<f32> = (0..WINDOW_SAMPLES)
            .map(|index| (index as f32 * 0.013).sin())
            .collect();
        let output = istft(&stft(&input), input.len()).unwrap();
        let maximum = input
            .iter()
            .zip(output)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum < 2e-4, "maximum reconstruction error: {maximum}");
    }
}
