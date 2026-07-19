//! `denoize` — pure-Rust audio denoiser built for the world's highest fidelity.
//!
//! Goal: transparent, artifact-free restoration that preserves timbre,
//! transients, dynamics, and "air" better than any classical offline tool.
//!
//! ## Implemented technologies
//!
//! ### Classical DSP (always available)
//! - STFT/ISTFT + Perfect Reconstruction OLA（高オーバーラップ対応）
//! - IMCRA/MCRA ノイズ推定 + Spectral Flatness プロファイル + Anchoring
//! - Ephraim-Malah Decision-Directed SNR
//! - 8種類のゲイン推定器（OMLSA, LogMMSE, MMSE-STSA, Wiener, SpecSub + 非線形/幾何学的）
//! - Attack/Release + Cepstral Smoothing + Transient Protection
//! - 高度窓関数: Kaiser / Flat-top / DPSS
//! - マルチバンドスペクトルサブトラクション
//! - 知覚重み付け（Bark帯域）+ 音楽ノイズ抑制ポストフィルタ
//!
//! ### Input / output codecs (built-in, no ffmpeg)
//! - **Decode**: WAV / MP3 (`nanomp3`) / M4A (Pure Rust AAC-LC)
//! - **Encode**: WAV / MP3 (`shine-rs`) / M4A (`oxideav-aac` Pure-Rust AAC-LC)
//! - Decoded to `f64` PCM at native sample rate (no extra quantisation)
//!
//! ### Optional AI backends (feature-gated)
//! - `rnnoise` feature: RNNoise via nnnoiseless (pure-Rust)
//! - `deepfilter` feature: DeepFilterNet v3 via tract ONNX
//! - `onnx` feature: user-supplied waveform ONNX models via tract
//! - `mpsenet` feature: MP-SENet compressed-magnitude/phase ONNX adapter
//! - `bsrnn` feature: ESPnet BSRNN spectral ONNX adapter
//! - `mossformer2` feature: ClearerVoice MossFormer2 48 kHz ONNX adapter
//! - `sgmse` feature: SGMSE+ iterative diffusion ONNX adapter
//!
//! Build with all backends: `cargo build --release --features full`

pub mod audio;
pub mod backend;
pub mod bessel;
pub mod decode;
pub mod denoiser;
pub mod encode;
pub mod fft;
pub mod gain;
pub mod noise;
pub mod perceptual;
pub mod postfilter;
pub mod stft;
pub mod window;

pub use audio::{read_audio, read_wav, write_audio, write_wav, Audio};
pub use backend::{Backend, BackendOptions, OnnxModelConfig};
pub use decode::{decode_file, AudioFormat, DecodedPcm};
pub use denoiser::{Denoiser, DenoiserConfig, Preset};
pub use encode::{EncodeOptions, OutputFormat};
pub use gain::{Algorithm, SpecSubLaw};
pub use window::{WindowParams, WindowType};

/// Denoise a WAV file end-to-end, writing the result to `output`.
pub fn denoise_file<P1, P2>(input: P1, output: P2, config: DenoiserConfig) -> Result<Audio, String>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    denoise_file_with_backend(input, output, config, Backend::Classical)
}

/// Denoise with an explicit backend (classical / rnnoise / deepfilter).
pub fn denoise_file_with_backend<P1, P2>(
    input: P1,
    output: P2,
    config: DenoiserConfig,
    backend: Backend,
) -> Result<Audio, String>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    denoise_file_with_backend_opts(input, output, config, backend, EncodeOptions::default())
}

/// Denoise with explicit backend and output encode options.
pub fn denoise_file_with_backend_opts<P1, P2>(
    input: P1,
    output: P2,
    config: DenoiserConfig,
    backend: Backend,
    encode_opts: EncodeOptions,
) -> Result<Audio, String>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    denoise_file_with_backend_config(
        input,
        output,
        config,
        backend,
        encode_opts,
        BackendOptions::default(),
    )
}

/// Denoise with explicit backend, encoder, and backend-specific model options.
pub fn denoise_file_with_backend_config<P1, P2>(
    input: P1,
    output: P2,
    mut config: DenoiserConfig,
    backend: Backend,
    encode_opts: EncodeOptions,
    backend_options: BackendOptions,
) -> Result<Audio, String>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    let mut audio = read_audio(input)?;
    config.sample_rate = audio.sample_rate;
    let t0 = std::time::Instant::now();
    audio.channels = backend::process_channels(
        backend,
        &audio.channels,
        audio.sample_rate,
        &config,
        &backend_options,
    )?;
    let elapsed = t0.elapsed();
    write_audio(output, &audio, encode_opts)?;
    eprintln!(
        "denoize: {:?} | {}ch x {} frames ({:.2}s) in {:.2?} ({:.1}x realtime)",
        backend,
        audio.channels(),
        audio.frames(),
        audio.frames() as f64 / audio.sample_rate as f64,
        elapsed,
        (audio.frames() as f64 / audio.sample_rate as f64) / elapsed.as_secs_f64().max(1e-9),
    );
    Ok(audio)
}
