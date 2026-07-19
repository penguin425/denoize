//! Official GTCRN streaming ONNX adapter.
//!
//! The model consumes one 512-point STFT frame at a time at 16 kHz and carries
//! three recurrent state tensors. The tensor layout follows the upstream MIT
//! implementation in `Xiaobin-Rong/gtcrn`.

use super::OnnxModelConfig;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::sync::Arc;
use tract_onnx::prelude::*;

pub const SAMPLE_RATE: u32 = 16_000;
pub const FFT_SIZE: usize = 512;
pub const HOP_SIZE: usize = 256;
pub const BINS: usize = 257;
const CONV_SIZE: usize = 2 * 1 * 16 * 16 * 33;
const TRA_SIZE: usize = 2 * 3 * 1 * 1 * 16;
const INTER_SIZE: usize = 2 * 1 * 33 * 16;

pub fn process(
    channels: &[Vec<f64>],
    input_sample_rate: u32,
    config: &OnnxModelConfig,
) -> Result<Vec<Vec<f64>>, String> {
    validate_config(config)?;
    channels
        .iter()
        .map(|channel| process_channel(channel, input_sample_rate, config))
        .collect()
}

fn process_channel(
    input: &[f64],
    input_sample_rate: u32,
    config: &OnnxModelConfig,
) -> Result<Vec<f64>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let at_model_rate = crate::resample::resample(input, input_sample_rate, SAMPLE_RATE)?;
    let mut stream = GtcrnStream::open(&config.path)?;
    let mut enhanced = Vec::with_capacity(at_model_rate.len() + FFT_SIZE);
    for chunk in at_model_rate.chunks(HOP_SIZE) {
        let mut hop = [0.0; HOP_SIZE];
        for (output, input) in hop.iter_mut().zip(chunk) {
            *output = *input as f32;
        }
        enhanced.extend(stream.process_hop(&hop)?);
    }
    enhanced.extend(stream.flush()?);
    // The causal WOLA frontend has one hop of algorithmic latency.
    let enhanced = enhanced
        .into_iter()
        .skip(HOP_SIZE)
        .take(at_model_rate.len());
    let model_output: Vec<f64> = enhanced.map(|sample| sample as f64).collect();
    let mut output = crate::resample::resample(&model_output, SAMPLE_RATE, input_sample_rate)?;
    output.truncate(input.len());
    output.resize(input.len(), 0.0);
    Ok(output)
}

fn validate_config(config: &OnnxModelConfig) -> Result<(), String> {
    if config.sample_rate != SAMPLE_RATE {
        return Err(format!(
            "GTCRN expects a {SAMPLE_RATE} Hz model, got {} Hz",
            config.sample_rate
        ));
    }
    if !config.path.is_file() {
        return Err(format!(
            "GTCRN ONNX model does not exist or is not a file: {}",
            config.path.display()
        ));
    }
    Ok(())
}

/// Stateful 16 kHz GTCRN processor. Each call consumes and returns exactly
/// 256 mono samples, making it suitable for realtime hosts and pipes.
pub struct GtcrnStream {
    model: TypedRunnableModel<TypedModel>,
    conv: Vec<f32>,
    tra: Vec<f32>,
    inter: Vec<f32>,
    analysis: [f32; FFT_SIZE],
    overlap: [f32; FFT_SIZE],
    window: [f32; FFT_SIZE],
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
}

impl GtcrnStream {
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let model = load_model(path)?;
        let window = std::array::from_fn(|index| {
            let phase = std::f32::consts::TAU * index as f32 / FFT_SIZE as f32;
            (0.5 * (1.0 - phase.cos())).sqrt()
        });
        let mut planner = FftPlanner::new();
        Ok(Self {
            model,
            conv: vec![0.0; CONV_SIZE],
            tra: vec![0.0; TRA_SIZE],
            inter: vec![0.0; INTER_SIZE],
            analysis: [0.0; FFT_SIZE],
            overlap: [0.0; FFT_SIZE],
            window,
            fft: planner.plan_fft_forward(FFT_SIZE),
            ifft: planner.plan_fft_inverse(FFT_SIZE),
        })
    }

    pub fn reset(&mut self) {
        self.conv.fill(0.0);
        self.tra.fill(0.0);
        self.inter.fill(0.0);
        self.analysis.fill(0.0);
        self.overlap.fill(0.0);
    }

    pub fn process_hop(&mut self, input: &[f32; HOP_SIZE]) -> Result<[f32; HOP_SIZE], String> {
        self.analysis.copy_within(HOP_SIZE.., 0);
        self.analysis[FFT_SIZE - HOP_SIZE..].copy_from_slice(input);
        let mut spectrum: Vec<Complex32> = self
            .analysis
            .iter()
            .zip(self.window)
            .map(|(sample, window)| Complex32::new(sample * window, 0.0))
            .collect();
        self.fft.process(&mut spectrum);
        let mut model_input = Vec::with_capacity(BINS * 2);
        for value in spectrum.iter().take(BINS) {
            model_input.extend([value.re, value.im]);
        }
        let enhanced = self.infer(&model_input)?;
        for bin in 0..BINS {
            spectrum[bin] = Complex32::new(enhanced[bin * 2], enhanced[bin * 2 + 1]);
        }
        for bin in BINS..FFT_SIZE {
            spectrum[bin] = spectrum[FFT_SIZE - bin].conj();
        }
        spectrum[0].im = 0.0;
        spectrum[BINS - 1].im = 0.0;
        self.ifft.process(&mut spectrum);
        for (index, value) in spectrum.iter().enumerate() {
            self.overlap[index] += value.re * self.window[index] / FFT_SIZE as f32;
        }
        let output = std::array::from_fn(|index| self.overlap[index]);
        self.overlap.copy_within(HOP_SIZE.., 0);
        self.overlap[FFT_SIZE - HOP_SIZE..].fill(0.0);
        Ok(output)
    }

    pub fn flush(&mut self) -> Result<[f32; HOP_SIZE], String> {
        self.process_hop(&[0.0; HOP_SIZE])
    }

    fn infer(&mut self, spectrum: &[f32]) -> Result<Vec<f32>, String> {
        let inputs = tvec!(
            Tensor::from_shape(&[1, BINS, 1, 2], spectrum)
                .map_err(tract_error)?
                .into_tvalue(),
            Tensor::from_shape(&[2, 1, 16, 16, 33], &self.conv)
                .map_err(tract_error)?
                .into_tvalue(),
            Tensor::from_shape(&[2, 3, 1, 1, 16], &self.tra)
                .map_err(tract_error)?
                .into_tvalue(),
            Tensor::from_shape(&[2, 1, 33, 16], &self.inter)
                .map_err(tract_error)?
                .into_tvalue(),
        );
        let outputs = self.model.run(inputs).map_err(tract_error)?;
        let copy = |tensor: &TValue| -> Result<Vec<f32>, String> {
            Ok(tensor
                .to_array_view::<f32>()
                .map_err(tract_error)?
                .iter()
                .copied()
                .collect())
        };
        let enhanced = copy(&outputs[0])?;
        self.conv = copy(&outputs[1])?;
        self.tra = copy(&outputs[2])?;
        self.inter = copy(&outputs[3])?;
        Ok(enhanced)
    }
}

fn load_model(path: &std::path::Path) -> Result<TypedRunnableModel<TypedModel>, String> {
    let mut model = tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|error| format!("failed to load GTCRN model {}: {error}", path.display()))?;
    let input_shapes: [&[usize]; 4] = [
        &[1, BINS, 1, 2],
        &[2, 1, 16, 16, 33],
        &[2, 3, 1, 1, 16],
        &[2, 1, 33, 16],
    ];
    let output_shapes = input_shapes;
    for (index, shape) in input_shapes.iter().enumerate() {
        model
            .set_input_fact(index, f32::fact(*shape).into())
            .map_err(tract_error)?;
    }
    for (index, shape) in output_shapes.iter().enumerate() {
        model
            .set_output_fact(index, f32::fact(*shape).into())
            .map_err(tract_error)?;
    }
    model
        .into_optimized()
        .and_then(|model| model.into_runnable())
        .map_err(tract_error)
}

fn tract_error(error: impl std::fmt::Display) -> String {
    format!("GTCRN inference failed: {error:#}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_rate_before_loading() {
        let config = OnnxModelConfig {
            path: "missing.onnx".into(),
            sample_rate: 48_000,
        };
        assert!(validate_config(&config).unwrap_err().contains("16000 Hz"));
    }

    #[test]
    fn published_state_sizes_match_shapes() {
        assert_eq!(CONV_SIZE, 16_896);
        assert_eq!(TRA_SIZE, 96);
        assert_eq!(INTER_SIZE, 1_056);
    }
}
