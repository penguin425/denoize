//! SGMSE+ adapter for the official VoiceBank+DEMAND checkpoint.
//!
//! The converted score graph receives `[state.real, state.imag, noisy.real,
//! noisy.imag]` in `[1, 4, 256, frames]` layout. This module implements the
//! official 16 kHz complex-STFT frontend and the quality-oriented 30-step
//! OUVE predictor/corrector sampler without a Python runtime.

use super::{OnnxModelConfig, SgmseProfile};
use rustfft::{num_complex::Complex32, FftPlanner};
use tract_onnx::prelude::*;

const MODEL_RATE: u32 = 16_000;
const FFT_SIZE: usize = 510;
const HOP_SIZE: usize = 128;
const BINS: usize = FFT_SIZE / 2 + 1;
const FRAME_MULTIPLE: usize = 64;
const SPEC_FACTOR: f32 = 0.15;
const THETA: f32 = 1.5;
const SIGMA_MIN: f32 = 0.05;
const SIGMA_MAX: f32 = 0.5;
const EPSILON: f32 = 0.03;
const CORRECTOR_SNR: f32 = 0.5;
const DEFAULT_SEED: u64 = 0x5347_4d53_452b_0030;

pub fn process(
    channels: &[Vec<f64>],
    input_sample_rate: u32,
    config: &OnnxModelConfig,
    profile: SgmseProfile,
) -> Result<Vec<Vec<f64>>, String> {
    if config.sample_rate != MODEL_RATE {
        return Err(format!(
            "SGMSE+ expects a {MODEL_RATE} Hz model, got {} Hz",
            config.sample_rate
        ));
    }
    if !config.path.is_file() {
        return Err(format!(
            "SGMSE+ ONNX model does not exist or is not a file: {}",
            config.path.display()
        ));
    }
    if channels.is_empty() {
        return Ok(Vec::new());
    }

    let model_samples =
        crate::resample::resample(&channels[0], input_sample_rate, MODEL_RATE)?.len();
    if model_samples == 0 {
        return Ok(channels.iter().map(|_| Vec::new()).collect());
    }
    let frames = padded_frame_count(model_samples);
    let model = load_model(config, frames)?;
    channels
        .iter()
        .enumerate()
        .map(|(index, channel)| {
            process_channel(
                channel,
                input_sample_rate,
                frames,
                DEFAULT_SEED.wrapping_add(index as u64),
                profile.steps(),
                &model,
            )
        })
        .collect()
}

fn process_channel(
    input: &[f64],
    input_sample_rate: u32,
    expected_frames: usize,
    seed: u64,
    steps: usize,
    model: &TypedRunnableModel<TypedModel>,
) -> Result<Vec<f64>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let at_model_rate = crate::resample::resample(input, input_sample_rate, MODEL_RATE)?;
    let normalization = at_model_rate
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0f64, f64::max);
    if normalization <= f64::EPSILON {
        return Ok(vec![0.0; input.len()]);
    }
    let normalized: Vec<f32> = at_model_rate
        .iter()
        .map(|sample| (*sample / normalization) as f32)
        .collect();
    let noisy = stft(&normalized);
    if noisy.frames != expected_frames {
        return Err(format!(
            "SGMSE+ channel produced {} padded frames; expected {expected_frames}",
            noisy.frames
        ));
    }
    let enhanced = sample(&noisy.values, noisy.frames, seed, steps, |state, time| {
        run_score_model(state, &noisy.values, noisy.frames, time, model)
    })?;
    let reconstructed = istft(&enhanced, noisy.frames, normalized.len())?;
    let denormalized: Vec<f64> = reconstructed
        .into_iter()
        .map(|sample| sample as f64 * normalization)
        .collect();
    let mut output = crate::resample::resample(&denormalized, MODEL_RATE, input_sample_rate)?;
    output.truncate(input.len());
    output.resize(input.len(), 0.0);
    Ok(output)
}

struct Spectrum {
    values: Vec<Complex32>,
    frames: usize,
}

fn padded_frame_count(samples: usize) -> usize {
    let frames = samples / HOP_SIZE + 1;
    frames.div_ceil(FRAME_MULTIPLE) * FRAME_MULTIPLE
}

fn stft(input: &[f32]) -> Spectrum {
    let natural_frames = input.len() / HOP_SIZE + 1;
    let frames = natural_frames.div_ceil(FRAME_MULTIPLE) * FRAME_MULTIPLE;
    let pad = FFT_SIZE / 2;
    let window = periodic_hann();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut values = vec![Complex32::default(); BINS * frames];
    let mut buffer = vec![Complex32::default(); FFT_SIZE];
    for frame in 0..natural_frames {
        let start = frame * HOP_SIZE;
        for index in 0..FFT_SIZE {
            let source = reflect_index(start as isize + index as isize - pad as isize, input.len());
            buffer[index] = Complex32::new(input[source] * window[index], 0.0);
        }
        fft.process(&mut buffer);
        for bin in 0..BINS {
            let raw = buffer[bin];
            let magnitude = raw.norm().sqrt() * SPEC_FACTOR;
            values[bin * frames + frame] = if magnitude == 0.0 {
                Complex32::default()
            } else {
                Complex32::from_polar(magnitude, raw.arg())
            };
        }
    }
    Spectrum { values, frames }
}

fn istft(spectrum: &[Complex32], frames: usize, output_length: usize) -> Result<Vec<f32>, String> {
    if spectrum.len() != BINS * frames {
        return Err("SGMSE+ spectrum has an unexpected size".into());
    }
    let natural_frames = output_length / HOP_SIZE + 1;
    if natural_frames > frames {
        return Err("SGMSE+ spectrum has too few frames".into());
    }
    let window = periodic_hann();
    let padded_length = (natural_frames - 1) * HOP_SIZE + FFT_SIZE;
    let mut signal = vec![0.0f32; padded_length];
    let mut envelope = vec![0.0f32; padded_length];
    let mut planner = FftPlanner::new();
    let inverse = planner.plan_fft_inverse(FFT_SIZE);
    let mut buffer = vec![Complex32::default(); FFT_SIZE];
    for frame in 0..natural_frames {
        for bin in 0..BINS {
            let transformed = spectrum[bin * frames + frame] / SPEC_FACTOR;
            let magnitude = transformed.norm().powi(2);
            buffer[bin] = Complex32::from_polar(magnitude, transformed.arg());
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
    let available = signal.len().saturating_sub(pad);
    let copy_length = output_length.min(available);
    let mut output = signal[pad..pad + copy_length].to_vec();
    output.resize(output_length, 0.0);
    if output.iter().any(|sample| !sample.is_finite()) {
        return Err("SGMSE+ reconstruction produced a non-finite sample".into());
    }
    Ok(output)
}

fn sample<F>(
    noisy: &[Complex32],
    frames: usize,
    seed: u64,
    steps: usize,
    mut score: F,
) -> Result<Vec<Complex32>, String>
where
    F: FnMut(&[Complex32], f32) -> Result<Vec<Complex32>, String>,
{
    let mut rng = NormalRng::new(seed);
    let prior_std = marginal_std(1.0);
    let mut state: Vec<Complex32> = noisy
        .iter()
        .map(|value| *value + rng.complex_normal() * prior_std)
        .collect();
    let mut final_mean = state.clone();
    if steps < 2 {
        return Err("SGMSE+ requires at least two diffusion steps".into());
    }
    for index in 0..steps {
        let time = 1.0 - (1.0 - EPSILON) * index as f32 / (steps - 1) as f32;
        let next_time = if index + 1 < steps {
            1.0 - (1.0 - EPSILON) * (index + 1) as f32 / (steps - 1) as f32
        } else {
            0.0
        };
        let step = time - next_time;

        let correction = score(&state, time)?;
        validate_score(&correction, noisy.len(), frames)?;
        let corrector_step = 2.0 * (CORRECTOR_SNR * marginal_std(time)).powi(2);
        for (value, gradient) in state.iter_mut().zip(correction) {
            *value += gradient * corrector_step;
            *value += rng.complex_normal() * (2.0 * corrector_step).sqrt();
        }

        let prediction = score(&state, time)?;
        validate_score(&prediction, noisy.len(), frames)?;
        let diffusion = SIGMA_MIN
            * (SIGMA_MAX / SIGMA_MIN).powf(time)
            * (2.0 * (SIGMA_MAX / SIGMA_MIN).ln()).sqrt();
        let variance = diffusion * diffusion * step;
        final_mean = Vec::with_capacity(state.len());
        for ((value, noisy_value), gradient) in state.iter_mut().zip(noisy).zip(prediction) {
            let mean = *value - THETA * (*noisy_value - *value) * step + gradient * variance;
            final_mean.push(mean);
            *value = mean + rng.complex_normal() * variance.sqrt();
        }
    }
    if final_mean
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("SGMSE+ sampler produced a non-finite value".into());
    }
    Ok(final_mean)
}

fn validate_score(score: &[Complex32], expected: usize, frames: usize) -> Result<(), String> {
    if score.len() != expected {
        return Err(format!(
            "SGMSE+ score has {} values; expected {} for {frames} frames",
            score.len(),
            expected
        ));
    }
    if score
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("SGMSE+ score contains a non-finite value".into());
    }
    Ok(())
}

fn marginal_std(time: f32) -> f32 {
    let log_sigma = (SIGMA_MAX / SIGMA_MIN).ln();
    let numerator = SIGMA_MIN.powi(2)
        * (-2.0 * THETA * time).exp()
        * ((2.0 * (THETA + log_sigma) * time).exp() - 1.0)
        * log_sigma;
    (numerator / (THETA + log_sigma)).sqrt()
}

fn load_model(
    config: &OnnxModelConfig,
    frames: usize,
) -> Result<TypedRunnableModel<TypedModel>, String> {
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
            != 1
    {
        return Err("SGMSE+ ONNX model must have two inputs and one output".into());
    }
    model
        .set_input_fact(0, f32::fact(tvec!(1, 4, BINS, frames)).into())
        .map_err(|error| model_error("configure features", error))?;
    model
        .set_input_fact(1, f32::fact(tvec!(1)).into())
        .map_err(|error| model_error("configure time", error))?;
    model
        .set_output_fact(0, f32::fact(tvec!(1, 2, BINS, frames)).into())
        .map_err(|error| model_error("configure score", error))?;
    model
        .into_optimized()
        .and_then(|model| model.into_runnable())
        .map_err(|error| model_error("optimize", error))
}

fn run_score_model(
    state: &[Complex32],
    noisy: &[Complex32],
    frames: usize,
    time: f32,
    model: &TypedRunnableModel<TypedModel>,
) -> Result<Vec<Complex32>, String> {
    let plane = BINS * frames;
    let mut features = vec![0.0f32; 4 * plane];
    for index in 0..plane {
        features[index] = state[index].re;
        features[plane + index] = state[index].im;
        features[2 * plane + index] = noisy[index].re;
        features[3 * plane + index] = noisy[index].im;
    }
    let feature_tensor = Tensor::from_shape(&tvec!(1, 4, BINS, frames), &features)
        .map_err(|error| model_error("create feature tensor", error))?;
    let time_tensor = Tensor::from_shape(&tvec!(1), &[time])
        .map_err(|error| model_error("create time tensor", error))?;
    let outputs = model
        .run(tvec!(
            feature_tensor.into_tvalue(),
            time_tensor.into_tvalue()
        ))
        .map_err(|error| model_error("run", error))?;
    let view = outputs[0]
        .to_array_view::<f32>()
        .map_err(|error| model_error("read score", error))?;
    if view.len() != 2 * plane {
        return Err(format!(
            "SGMSE+ score tensor has {} values; expected {}",
            view.len(),
            2 * plane
        ));
    }
    let flat: Vec<f32> = view.iter().copied().collect();
    Ok((0..plane)
        .map(|index| Complex32::new(flat[index], flat[plane + index]))
        .collect())
}

fn model_error(stage: &str, error: impl std::fmt::Display) -> String {
    format!("SGMSE+ ONNX {stage} failed: {error:#}")
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

struct NormalRng {
    state: u64,
    spare: Option<f32>,
}

impl NormalRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn uniform(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        let mantissa = (value >> 40) as u32;
        (mantissa as f32 + 0.5) / 16_777_216.0
    }

    fn normal(&mut self) -> f32 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = 2.0 * std::f32::consts::PI * self.uniform();
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }

    fn complex_normal(&mut self) -> Complex32 {
        let scale = std::f32::consts::FRAC_1_SQRT_2;
        Complex32::new(self.normal() * scale, self.normal() * scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_round_trip_matches_input() {
        let input: Vec<f32> = (0..8_000)
            .map(|index| {
                (2.0 * std::f32::consts::PI * 440.0 * index as f32 / MODEL_RATE as f32).sin() * 0.5
            })
            .collect();
        let spectrum = stft(&input);
        assert_eq!(spectrum.frames % FRAME_MULTIPLE, 0);
        let output = istft(&spectrum.values, spectrum.frames, input.len()).unwrap();
        let max_error = input
            .iter()
            .zip(output)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 2e-4, "maximum error: {max_error}");
    }

    #[test]
    fn sampler_is_deterministic_for_a_fixed_seed() {
        let noisy = vec![Complex32::new(0.1, -0.2); 17];
        let run = || {
            sample(&noisy, 64, 42, 30, |state, _| {
                Ok(vec![Complex32::default(); state.len()])
            })
            .unwrap()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn sampler_uses_sixty_score_evaluations() {
        let noisy = vec![Complex32::default(); 3];
        let mut evaluations = 0;
        sample(&noisy, 64, 7, 30, |state, _| {
            evaluations += 1;
            Ok(vec![Complex32::default(); state.len()])
        })
        .unwrap();
        assert_eq!(evaluations, 60);
    }

    #[test]
    fn marginal_std_matches_official_ouve_reference() {
        assert!((marginal_std(1.0) - 0.388_982_65).abs() < 1e-6);
        assert!((marginal_std(0.03) - 0.018_830_1).abs() < 1e-6);
    }
}
