//! Optional AI denoising backends (feature-gated).

mod classical;

pub use classical::process_classical;

/// Denoising backend selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Enhanced classical DSP pipeline (default).
    Classical,
    /// RNNoise (nnnoiseless — pure-Rust port of Xiph RNNoise).
    #[cfg(feature = "rnnoise")]
    Rnnoise,
    /// DeepFilterNet v3 (tract ONNX, embedded default model).
    #[cfg(feature = "deepfilter")]
    DeepFilter,
}

impl Backend {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "classical" | "dsp" | "stft" => Backend::Classical,
            #[cfg(feature = "rnnoise")]
            "rnnoise" | "rnn" => Backend::Rnnoise,
            #[cfg(feature = "deepfilter")]
            "deepfilter" | "deepfilternet" | "dfn" | "dfn3" => Backend::DeepFilter,
            #[cfg(not(feature = "rnnoise"))]
            "rnnoise" | "rnn" => return None,
            #[cfg(not(feature = "deepfilter"))]
            "deepfilter" | "deepfilternet" | "dfn" | "dfn3" => return None,
            _ => return None,
        })
    }

    pub fn available_names() -> &'static [&'static str] {
        #[cfg(all(feature = "rnnoise", feature = "deepfilter"))]
        return &["classical", "rnnoise", "deepfilter"];
        #[cfg(all(feature = "rnnoise", not(feature = "deepfilter")))]
        return &["classical", "rnnoise"];
        #[cfg(all(not(feature = "rnnoise"), feature = "deepfilter"))]
        return &["classical", "deepfilter"];
        #[cfg(not(any(feature = "rnnoise", feature = "deepfilter")))]
        return &["classical"];
    }
}

/// Process all channels with the selected backend.
pub fn process_channels(
    backend: Backend,
    channels: &[Vec<f64>],
    sample_rate: u32,
    classical_cfg: &crate::denoiser::DenoiserConfig,
) -> Result<Vec<Vec<f64>>, String> {
    let _ = sample_rate; // used by AI backends; classical reads from cfg
    match backend {
        Backend::Classical => Ok(process_classical(channels, classical_cfg)),
        #[cfg(feature = "rnnoise")]
        Backend::Rnnoise => rnnoise::process(channels, sample_rate),
        #[cfg(feature = "deepfilter")]
        Backend::DeepFilter => deepfilter::process(channels, sample_rate),
    }
}

#[cfg(feature = "rnnoise")]
pub mod rnnoise;

#[cfg(feature = "deepfilter")]
pub mod deepfilter;
