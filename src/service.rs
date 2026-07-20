//! Shared application service used by the CLI and graphical frontends.

use crate::loudness::LoudnessReport;
#[cfg(feature = "gtcrn")]
use crate::OnnxModelConfig;
use crate::{denoise_audio_with_backend_config, Audio, Backend, BackendOptions, DenoiserConfig};
use std::time::Duration;

/// User-facing backend choice shared by every application frontend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    Auto,
    Explicit(Backend),
}

/// Options for processing decoded audio.
#[derive(Clone, Debug)]
pub struct ProcessingOptions {
    pub backend: BackendChoice,
    pub quality: Option<String>,
    pub denoiser: DenoiserConfig,
    pub backend_options: BackendOptions,
    pub loudness_lufs: Option<f64>,
    pub true_peak_dbtp: f64,
}

/// Information produced by a completed processing operation.
#[derive(Clone, Copy, Debug)]
pub struct ProcessingResult {
    pub backend: Backend,
    pub elapsed: Duration,
    pub loudness: Option<LoudnessReport>,
}

/// Stable display/configuration name for a compiled backend.
pub fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Classical => "classical",
        #[cfg(feature = "rnnoise")]
        Backend::Rnnoise => "rnnoise",
        #[cfg(feature = "deepfilter")]
        Backend::DeepFilter => "deepfilter",
        #[cfg(feature = "onnx")]
        Backend::Onnx => "onnx",
        #[cfg(feature = "mpsenet")]
        Backend::MpSenet => "mpsenet",
        #[cfg(feature = "bsrnn")]
        Backend::Bsrnn => "bsrnn",
        #[cfg(feature = "mossformer2")]
        Backend::Mossformer2 => "mossformer2",
        #[cfg(feature = "sgmse")]
        Backend::Sgmse => "sgmse",
        #[cfg(feature = "gtcrn")]
        Backend::Gtcrn => "gtcrn",
    }
}

/// Whether a backend needs a user-selected ONNX file rather than embedded or
/// managed weights.
pub fn requires_external_model(backend: Backend) -> bool {
    match backend {
        #[cfg(feature = "onnx")]
        Backend::Onnx => true,
        #[cfg(feature = "mpsenet")]
        Backend::MpSenet => true,
        #[cfg(feature = "bsrnn")]
        Backend::Bsrnn => true,
        #[cfg(feature = "mossformer2")]
        Backend::Mossformer2 => true,
        #[cfg(feature = "sgmse")]
        Backend::Sgmse => true,
        _ => false,
    }
}

/// Select a backend consistently for CLI and graphical processing.
pub fn select_backend(
    choice: BackendChoice,
    _duration_seconds: f64,
    _quality: Option<&str>,
) -> Backend {
    if let BackendChoice::Explicit(backend) = choice {
        return backend;
    }
    #[cfg(feature = "deepfilter")]
    {
        let high_quality = matches!(_quality, Some("high" | "ultra" | "max" | "highest"));
        if high_quality || _duration_seconds <= 10.0 * 60.0 {
            return Backend::DeepFilter;
        }
    }
    #[cfg(feature = "rnnoise")]
    {
        return Backend::Rnnoise;
    }
    #[allow(unreachable_code)]
    Backend::Classical
}

/// Fill backend options that can be resolved from the managed model library.
pub fn resolve_backend_options(
    _backend: Backend,
    #[allow(unused_mut)] mut options: BackendOptions,
) -> Result<BackendOptions, String> {
    #[cfg(feature = "gtcrn")]
    if _backend == Backend::Gtcrn && options.onnx.is_none() {
        let model = crate::models::find("gtcrn").expect("built-in GTCRN manifest entry");
        options.onnx = Some(OnnxModelConfig {
            path: crate::models::verify(model).map_err(|_| {
                "GTCRN model is not installed; run `denoize models install gtcrn`".to_string()
            })?,
            sample_rate: model.sample_rate,
        });
    }
    Ok(options)
}

/// Process already-decoded audio with common backend and delivery behavior.
pub fn process_audio(
    audio: &mut Audio,
    options: ProcessingOptions,
) -> Result<ProcessingResult, String> {
    let duration = audio.frames() as f64 / audio.sample_rate.max(1) as f64;
    let backend = select_backend(options.backend, duration, options.quality.as_deref());
    let backend_options = resolve_backend_options(backend, options.backend_options)?;
    let elapsed =
        denoise_audio_with_backend_config(audio, options.denoiser, backend, &backend_options)?;
    let loudness = options
        .loudness_lufs
        .map(|target| crate::loudness::normalize(audio, target, options.true_peak_dbtp))
        .transpose()?;
    Ok(ProcessingResult {
        backend,
        elapsed,
        loudness,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_backend_is_preserved() {
        assert_eq!(
            select_backend(
                BackendChoice::Explicit(Backend::Classical),
                10.0,
                Some("ultra")
            ),
            Backend::Classical
        );
    }

    #[test]
    fn automatic_backend_is_compiled() {
        let selected = select_backend(BackendChoice::Auto, 10.0, None);
        assert!(Backend::available_names().contains(&backend_name(selected)));
    }

    #[test]
    fn classical_does_not_require_external_weights() {
        assert!(!requires_external_model(Backend::Classical));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn generic_onnx_requires_external_weights() {
        assert!(requires_external_model(Backend::Onnx));
    }
}
