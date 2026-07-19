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
    /// ClearerVoice MossFormer2 48 kHz speech enhancement model.
    #[cfg(feature = "mossformer2")]
    Mossformer2,
    /// SGMSE+ diffusion speech enhancement model.
    #[cfg(feature = "sgmse")]
    Sgmse,
    /// Official streaming GTCRN speech enhancement model.
    #[cfg(feature = "gtcrn")]
    Gtcrn,
}

/// Configuration for a waveform-to-waveform ONNX enhancement model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnxModelConfig {
    /// Path to the ONNX model file.
    pub path: PathBuf,
    /// Sample rate expected and produced by the model.
    pub sample_rate: u32,
}

/// How a stereo pair is presented to a denoising backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChannelMode {
    /// Process channels separately (legacy behavior).
    #[default]
    Independent,
    /// Estimate one common correction from the stereo mid signal and apply it
    /// equally to left and right. This preserves the side signal exactly.
    StereoLinked,
    /// Transform left/right to mid/side, denoise both, then reconstruct.
    MidSide,
}

impl ChannelMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "independent" | "separate" => Some(Self::Independent),
            "linked" | "stereo-linked" | "stereo_linked" => Some(Self::StereoLinked),
            "mid-side" | "midside" | "ms" => Some(Self::MidSide),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SgmseProfile {
    Fast,
    #[default]
    Balanced,
    Quality,
}

impl SgmseProfile {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "fast" => Some(Self::Fast),
            "balanced" | "default" => Some(Self::Balanced),
            "quality" | "high" => Some(Self::Quality),
            _ => None,
        }
    }
    #[cfg(feature = "sgmse")]
    pub(crate) fn steps(self) -> usize {
        match self {
            Self::Fast => 8,
            Self::Balanced => 20,
            Self::Quality => 30,
        }
    }
}

/// Options used by backends that require external model configuration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BackendOptions {
    /// Model configuration used by the `onnx` backend when that feature is enabled.
    pub onnx: Option<OnnxModelConfig>,
    /// Stereo channel coupling strategy.
    pub channel_mode: ChannelMode,
    /// SGMSE+ diffusion budget.
    pub sgmse_profile: SgmseProfile,
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
            #[cfg(feature = "mossformer2")]
            "mossformer2" | "moss-former2" | "mossformer" => Backend::Mossformer2,
            #[cfg(feature = "sgmse")]
            "sgmse" | "sgmse+" | "sgmse-plus" => Backend::Sgmse,
            #[cfg(feature = "gtcrn")]
            "gtcrn" => Backend::Gtcrn,
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
            #[cfg(not(feature = "mossformer2"))]
            "mossformer2" | "moss-former2" | "mossformer" => return None,
            #[cfg(not(feature = "sgmse"))]
            "sgmse" | "sgmse+" | "sgmse-plus" => return None,
            #[cfg(not(feature = "gtcrn"))]
            "gtcrn" => return None,
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
            #[cfg(feature = "mossformer2")]
            "mossformer2",
            #[cfg(feature = "sgmse")]
            "sgmse",
            #[cfg(feature = "gtcrn")]
            "gtcrn",
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
    if channels.len() == 2 && backend_options.channel_mode != ChannelMode::Independent {
        return process_stereo(
            backend,
            channels,
            sample_rate,
            classical_cfg,
            backend_options,
        );
    }
    process_channels_independent(
        backend,
        channels,
        sample_rate,
        classical_cfg,
        backend_options,
    )
}

fn process_stereo(
    backend: Backend,
    channels: &[Vec<f64>],
    sample_rate: u32,
    classical_cfg: &crate::denoiser::DenoiserConfig,
    backend_options: &BackendOptions,
) -> Result<Vec<Vec<f64>>, String> {
    if channels[0].len() != channels[1].len() {
        return Err("stereo channels must contain the same number of frames".into());
    }
    let mid: Vec<f64> = channels[0]
        .iter()
        .zip(&channels[1])
        .map(|(left, right)| (left + right) * 0.5)
        .collect();
    match backend_options.channel_mode {
        ChannelMode::StereoLinked => {
            let enhanced = process_channels_independent(
                backend,
                std::slice::from_ref(&mid),
                sample_rate,
                classical_cfg,
                backend_options,
            )?
            .pop()
            .unwrap_or_default();
            let mut result = channels.to_vec();
            let (left_channels, right_channels) = result.split_at_mut(1);
            for ((left, right), (original, clean)) in left_channels[0]
                .iter_mut()
                .zip(&mut right_channels[0])
                .zip(mid.iter().zip(enhanced.iter()))
            {
                let correction = clean - original;
                *left += correction;
                *right += correction;
            }
            Ok(result)
        }
        ChannelMode::MidSide => {
            let side: Vec<f64> = channels[0]
                .iter()
                .zip(&channels[1])
                .map(|(left, right)| (left - right) * 0.5)
                .collect();
            let processed = process_channels_independent(
                backend,
                &[mid, side],
                sample_rate,
                classical_cfg,
                backend_options,
            )?;
            Ok(vec![
                processed[0]
                    .iter()
                    .zip(&processed[1])
                    .map(|(mid, side)| mid + side)
                    .collect(),
                processed[0]
                    .iter()
                    .zip(&processed[1])
                    .map(|(mid, side)| mid - side)
                    .collect(),
            ])
        }
        ChannelMode::Independent => unreachable!(),
    }
}

fn process_channels_independent(
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
        #[cfg(feature = "mossformer2")]
        Backend::Mossformer2 => {
            let config = backend_options.onnx.as_ref().ok_or_else(|| {
                "MossFormer2 backend requires a converted model (CLI: --onnx-model <PATH>)"
                    .to_string()
            })?;
            mossformer2::process(channels, sample_rate, config)
        }
        #[cfg(feature = "sgmse")]
        Backend::Sgmse => {
            let config = backend_options.onnx.as_ref().ok_or_else(|| {
                "SGMSE+ backend requires a converted model (CLI: --onnx-model <PATH>)".to_string()
            })?;
            sgmse::process(channels, sample_rate, config, backend_options.sgmse_profile)
        }
        #[cfg(feature = "gtcrn")]
        Backend::Gtcrn => {
            let config = backend_options.onnx.as_ref().ok_or_else(|| {
                "GTCRN backend requires the official streaming model (CLI: --onnx-model <PATH>)"
                    .to_string()
            })?;
            gtcrn::process(channels, sample_rate, config)
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

#[cfg(feature = "mossformer2")]
pub mod mossformer2;

#[cfg(feature = "sgmse")]
pub mod sgmse;

#[cfg(feature = "gtcrn")]
pub mod gtcrn;

#[cfg(test)]
mod channel_tests {
    use super::*;

    #[test]
    fn parses_channel_modes() {
        assert_eq!(
            ChannelMode::parse("linked"),
            Some(ChannelMode::StereoLinked)
        );
        assert_eq!(ChannelMode::parse("mid-side"), Some(ChannelMode::MidSide));
        assert_eq!(
            ChannelMode::parse("independent"),
            Some(ChannelMode::Independent)
        );
    }

    #[test]
    fn stereo_linked_preserves_the_side_signal() {
        let frames = 4_096;
        let left: Vec<f64> = (0..frames)
            .map(|i| (i as f64 * 0.013).sin() * 0.4)
            .collect();
        let right: Vec<f64> = (0..frames)
            .map(|i| (i as f64 * 0.017).sin() * 0.3)
            .collect();
        let input = vec![left, right];
        let options = BackendOptions {
            channel_mode: ChannelMode::StereoLinked,
            ..BackendOptions::default()
        };
        let output = process_channels(
            Backend::Classical,
            &input,
            48_000,
            &crate::denoiser::DenoiserConfig::default(48_000),
            &options,
        )
        .unwrap();
        for index in 0..frames {
            let before = input[0][index] - input[1][index];
            let after = output[0][index] - output[1][index];
            assert!((before - after).abs() < 1e-12);
        }
    }
}

#[cfg(all(
    test,
    any(
        feature = "mpsenet",
        feature = "bsrnn",
        feature = "mossformer2",
        feature = "sgmse"
    )
))]
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

    #[cfg(feature = "mossformer2")]
    #[test]
    fn parses_mossformer2_aliases() {
        assert_eq!(Backend::parse("mossformer2"), Some(Backend::Mossformer2));
        assert_eq!(Backend::parse("moss-former2"), Some(Backend::Mossformer2));
        assert!(Backend::available_names().contains(&"mossformer2"));
    }

    #[cfg(feature = "sgmse")]
    #[test]
    fn parses_sgmse_aliases() {
        assert_eq!(Backend::parse("sgmse"), Some(Backend::Sgmse));
        assert_eq!(Backend::parse("sgmse+"), Some(Backend::Sgmse));
        assert!(Backend::available_names().contains(&"sgmse"));
    }
}
