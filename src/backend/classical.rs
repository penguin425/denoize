use crate::denoiser::{Denoiser, DenoiserConfig};

/// Run the classical STFT/IMCRA pipeline.
pub fn process_classical(channels: &[Vec<f64>], cfg: &DenoiserConfig) -> Vec<Vec<f64>> {
    let mut den = Denoiser::new(cfg.clone());
    den.process(channels)
}
