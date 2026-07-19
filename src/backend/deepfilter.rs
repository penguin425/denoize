//! DeepFilterNet v3 backend via the official `deep_filter` crate (tract ONNX).
//!
//! Requires `--features deepfilter` at build time. Uses the embedded DFN3 model.

use df::tract::{DfParams, DfTract, RuntimeParams};
use df::transforms::resample;
use ndarray::{Array2, Axis};

/// Target sample rate for DeepFilterNet (48 kHz).
const DF_SR: usize = 48_000;

/// Denoise channels using DeepFilterNet v3.
pub fn process(channels: &[Vec<f64>], sample_rate: u32) -> Result<Vec<Vec<f64>>, String> {
    let n_ch = channels.len().max(1);
    let max_len = channels.iter().map(|c| c.len()).max().unwrap_or(0);
    if max_len == 0 {
        return Ok(channels.to_vec());
    }

    // Build f32 array [channels, samples] at 48 kHz.
    let mut ch_data: Vec<Vec<f32>> = Vec::with_capacity(n_ch);
    for ch in channels {
        let f32_in: Vec<f32> = ch.iter().map(|&x| x as f32).collect();
        let at_48k = if sample_rate as usize == DF_SR {
            f32_in
        } else {
            resample_to_48k(&f32_in, sample_rate as usize)?
        };
        ch_data.push(at_48k);
    }

    let r_params = RuntimeParams::default_with_ch(n_ch)
        .with_atten_lim(100.0)
        .with_thresholds(-15.0, 35.0, 35.0);
    let df_params = DfParams::default();
    let mut model = DfTract::new(df_params, &r_params)
        .map_err(|e| format!("DeepFilterNet init failed: {e}"))?;

    // Pad both unequal channels and the final partial model hop. The result is
    // trimmed back to each original channel length below.
    let source_len_48k = ch_data.iter().map(|c| c.len()).max().unwrap_or(0);
    let len_48k = padded_hop_len(source_len_48k, model.hop_size);
    for c in &mut ch_data {
        c.resize(len_48k, 0.0);
    }

    let noisy = Array2::from_shape_fn((n_ch, len_48k), |(ch, i)| ch_data[ch][i]);
    let mut enh = Array2::zeros((n_ch, len_48k));

    for (ns_chunk, enh_chunk) in noisy
        .view()
        .axis_chunks_iter(Axis(1), model.hop_size)
        .zip(enh.view_mut().axis_chunks_iter_mut(Axis(1), model.hop_size))
    {
        debug_assert_eq!(ns_chunk.len_of(Axis(1)), model.hop_size);
        model
            .process(ns_chunk, enh_chunk)
            .map_err(|e| format!("DeepFilterNet process failed: {e}"))?;
    }

    // Extract per-channel output and resample back.
    let mut result = Vec::with_capacity(n_ch);
    for ch in 0..n_ch {
        let row: Vec<f32> = enh.row(ch).iter().copied().collect();
        let f64_out: Vec<f64> = if sample_rate as usize == DF_SR {
            row.iter().map(|&x| x as f64).collect()
        } else {
            resample_from_48k(&row, sample_rate as usize)?
                .iter()
                .map(|&x| x as f64)
                .collect()
        };
        let orig_len = channels.get(ch).map(|c| c.len()).unwrap_or(len_48k);
        let mut trimmed = f64_out;
        trimmed.truncate(orig_len);
        if trimmed.len() < orig_len {
            trimmed.resize(orig_len, 0.0);
        }
        result.push(trimmed);
    }
    Ok(result)
}

fn padded_hop_len(input_len: usize, hop_size: usize) -> usize {
    input_len.div_ceil(hop_size) * hop_size
}

fn resample_to_48k(input: &[f32], from_sr: usize) -> Result<Vec<f32>, String> {
    if from_sr == DF_SR {
        return Ok(input.to_vec());
    }
    let arr = ndarray::Array2::from_shape_fn((1, input.len()), |(_, i)| input[i]);
    let out = resample(arr.view(), from_sr, DF_SR, None)
        .map_err(|e| format!("resample to 48k failed: {e}"))?;
    Ok(out.row(0).iter().copied().collect())
}

fn resample_from_48k(input: &[f32], to_sr: usize) -> Result<Vec<f32>, String> {
    if to_sr == DF_SR {
        return Ok(input.to_vec());
    }
    let arr = ndarray::Array2::from_shape_fn((1, input.len()), |(_, i)| input[i]);
    let out = resample(arr.view(), DF_SR, to_sr, None)
        .map_err(|e| format!("resample from 48k failed: {e}"))?;
    Ok(out.row(0).iter().copied().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_partial_hop_is_padded() {
        assert_eq!(padded_hop_len(1, 480), 480);
        assert_eq!(padded_hop_len(480, 480), 480);
        assert_eq!(padded_hop_len(481, 480), 960);
    }
}
