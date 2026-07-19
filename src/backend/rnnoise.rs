//! RNNoise backend via `nnnoiseless` (pure-Rust port of Xiph RNNoise).
//!
//! Operates at 48 kHz, 480-sample frames. Other sample rates are linearly
//! resampled.

use nnnoiseless::DenoiseState;

const RN_SR: u32 = 48_000;
const FRAME: usize = DenoiseState::FRAME_SIZE;

/// Denoise channels using RNNoise.
pub fn process(channels: &[Vec<f64>], sample_rate: u32) -> Result<Vec<Vec<f64>>, String> {
    let mut out = Vec::with_capacity(channels.len());
    for ch in channels {
        out.push(process_channel(ch, sample_rate)?);
    }
    Ok(out)
}

fn process_channel(input: &[f64], sample_rate: u32) -> Result<Vec<f64>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    // Resample to 48 kHz if needed.
    let at_48k: Vec<f32> = if sample_rate == RN_SR {
        input.iter().map(|&x| (x as f32) * 32768.0).collect()
    } else {
        resample_linear(input, sample_rate, RN_SR)
            .into_iter()
            .map(|x| (x as f32) * 32768.0)
            .collect()
    };

    let mut denoise = DenoiseState::new();
    let mut out_buf = [0.0f32; FRAME];
    let mut output = Vec::with_capacity(at_48k.len());
    let mut i = 0;
    while i < at_48k.len() {
        let end = (i + FRAME).min(at_48k.len());
        let mut frame = [0.0f32; FRAME];
        frame[..end - i].copy_from_slice(&at_48k[i..end]);
        denoise.process_frame(&mut out_buf, &frame);
        // Keep every frame aligned with its input. Discarding the first frame
        // shortens the stream by 10 ms, shifts all remaining audio earlier,
        // and turns inputs <= FRAME into silence.
        let n = if end - i == FRAME { FRAME } else { end - i };
        output.extend_from_slice(&out_buf[..n]);
        i += FRAME;
    }

    // Resample back to original rate.
    let normalized: Vec<f64> = output.iter().map(|&x| (x as f64) / 32768.0).collect();
    let result = if sample_rate == RN_SR {
        normalized
    } else {
        resample_linear_f64(&normalized, RN_SR, sample_rate)
    };

    // Match input length.
    let mut trimmed = result;
    trimmed.truncate(input.len());
    if trimmed.len() < input.len() {
        trimmed.resize(input.len(), 0.0);
    }
    Ok(trimmed)
}

fn resample_linear(input: &[f64], from_sr: u32, to_sr: u32) -> Vec<f64> {
    if from_sr == to_sr {
        return input.to_vec();
    }
    let ratio = to_sr as f64 / from_sr as f64;
    let out_len = (input.len() as f64 * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len.max(1));
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = src - idx as f64;
        let a = input.get(idx).copied().unwrap_or(0.0);
        let b = input.get(idx + 1).copied().unwrap_or(a);
        out.push(a + frac * (b - a));
    }
    out
}

fn resample_linear_f64(input: &[f64], from_sr: u32, to_sr: u32) -> Vec<f64> {
    resample_linear(input, from_sr, to_sr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        assert!(process_channel(&[], 48_000).unwrap().is_empty());
    }

    #[test]
    fn short_input_keeps_length_and_audio() {
        let input: Vec<f64> = (0..FRAME)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / RN_SR as f64).sin() * 0.5)
            .collect();
        let output = process_channel(&input, RN_SR).unwrap();
        assert_eq!(output.len(), input.len());
        assert!(output.iter().any(|x| x.abs() > 1e-6));
    }
}
