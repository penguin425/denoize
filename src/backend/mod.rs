//! Optional AI denoising backends (feature-gated).

mod classical;

use std::path::PathBuf;

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
    /// User-supplied waveform-to-waveform ONNX model (pure-Rust tract runtime).
    #[cfg(feature = "onnx")]
    Onnx,
    /// MP-SENet magnitude/phase speech enhancement model.
    #[cfg(feature = "mpsenet")]
    MpSenet,
    /// ESPnet band-split recurrent speech enhancement model.
    #[cfg(feature = "bsrnn")]
    Bsrnn,
}

/// Configuration for a waveform-to-waveform ONNX enhancement model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnxModelConfig {
    /// Path to the ONNX model file.
    pub path: PathBuf,
    /// Sample rate expected and produced by the model.
    pub sample_rate: u32,
}

/// Options used by backends that require external model configuration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BackendOptions {
    /// Model configuration used by the `onnx` backend when that feature is enabled.
    pub onnx: Option<OnnxModelConfig>,
}

impl Backend {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "classical" | "dsp" | "stft" => Backend::Classical,
            #[cfg(feature = "rnnoise")]
            "rnnoise" | "rnn" => Backend::Rnnoise,
            #[cfg(feature = "deepfilter")]
            "deepfilter" | "deepfilternet" | "dfn" | "dfn3" => Backend::DeepFilter,
            #[cfg(feature = "onnx")]
            "onnx" | "model" => Backend::Onnx,
            #[cfg(feature = "mpsenet")]
            "mpsenet" | "mp-senet" | "mp_senet" => Backend::MpSenet,
            #[cfg(feature = "bsrnn")]
            "bsrnn" | "bs-rnn" | "bs_rnn" => Backend::Bsrnn,
            #[cfg(not(feature = "rnnoise"))]
            "rnnoise" | "rnn" => return None,
            #[cfg(not(feature = "deepfilter"))]
            "deepfilter" | "deepfilternet" | "dfn" | "dfn3" => return None,
            #[cfg(not(feature = "onnx"))]
            "onnx" | "model" => return None,
            #[cfg(not(feature = "mpsenet"))]
            "mpsenet" | "mp-senet" | "mp_senet" => return None,
            #[cfg(not(feature = "bsrnn"))]
            "bsrnn" | "bs-rnn" | "bs_rnn" => return None,
            _ => return None,
        })
    }

    pub fn available_names() -> &'static [&'static str] {
        &[
            "classical",
            #[cfg(feature = "rnnoise")]
            "rnnoise",
            #[cfg(feature = "deepfilter")]
            "deepfilter",
            #[cfg(feature = "onnx")]
            "onnx",
            #[cfg(feature = "mpsenet")]
            "mpsenet",
            #[cfg(feature = "bsrnn")]
            "bsrnn",
        ]
    }
}

/// Process all channels with the selected backend.
pub fn process_channels(
    backend: Backend,
    channels: &[Vec<f64>],
    sample_rate: u32,
    classical_cfg: &crate::denoiser::DenoiserConfig,
    backend_options: &BackendOptions,
) -> Result<Vec<Vec<f64>>, String> {
    let _ = sample_rate; // used by AI backends; classical reads from cfg
    let _ = backend_options; // used by configured model backends when enabled
    match backend {
        Backend::Classical => Ok(process_classical(channels, classical_cfg)),
        #[cfg(feature = "rnnoise")]
        Backend::Rnnoise => rnnoise::process(channels, sample_rate),
        #[cfg(feature = "deepfilter")]
        Backend::DeepFilter => deepfilter::process(channels, sample_rate),
        #[cfg(feature = "onnx")]
        Backend::Onnx => {
            let config = backend_options.onnx.as_ref().ok_or_else(|| {
                "ONNX backend requires a model path (CLI: --onnx-model <PATH>)".to_string()
            })?;
            onnx::process(channels, sample_rate, config)
        }
        #[cfg(feature = "mpsenet")]
        Backend::MpSenet => {
            let config = backend_options.onnx.as_ref().ok_or_else(|| {
                "MP-SENet backend requires a converted model (CLI: --onnx-model <PATH>)".to_string()
            })?;
            mpsenet::process(channels, sample_rate, config)
        }
        #[cfg(feature = "bsrnn")]
        Backend::Bsrnn => {
            let config = backend_options.onnx.as_ref().ok_or_else(|| {
                "BSRNN backend requires a converted model (CLI: --onnx-model <PATH>)".to_string()
            })?;
            bsrnn::process(channels, sample_rate, config)
        }
    }
}

#[cfg(feature = "rnnoise")]
pub mod rnnoise;

#[cfg(feature = "deepfilter")]
pub mod deepfilter;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "mpsenet")]
pub mod mpsenet;

#[cfg(feature = "bsrnn")]
pub mod bsrnn;

#[cfg(all(test, any(feature = "mpsenet", feature = "bsrnn")))]
mod tests {
    use super::*;

    #[cfg(feature = "mpsenet")]
    #[test]
    fn parses_mp_senet_aliases() {
        assert_eq!(Backend::parse("mpsenet"), Some(Backend::MpSenet));
        assert_eq!(Backend::parse("mp-senet"), Some(Backend::MpSenet));
        assert!(Backend::available_names().contains(&"mpsenet"));
    }

    #[cfg(feature = "bsrnn")]
    #[test]
    fn parses_bsrnn_aliases() {
        assert_eq!(Backend::parse("bsrnn"), Some(Backend::Bsrnn));
        assert_eq!(Backend::parse("bs-rnn"), Some(Backend::Bsrnn));
        assert!(Backend::available_names().contains(&"bsrnn"));
    }
}
