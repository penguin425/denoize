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
pub mod benchmark;
pub mod bessel;
pub mod decode;
pub mod denoiser;
pub mod encode;
pub mod fft;
pub mod gain;
#[cfg(feature = "live")]
pub mod live;
pub mod loudness;
pub mod metadata;
pub mod models;
pub mod noise;
pub mod perceptual;
pub mod postfilter;
pub mod resample;
pub mod stft;
pub mod stream;
pub mod vad;
pub mod window;

pub use audio::{
    read_audio, read_wav, read_wav_bytes, write_audio, write_wav, write_wav_bytes, Audio,
};
pub use backend::{Backend, BackendOptions, ChannelMode, OnnxModelConfig, SgmseProfile};
pub use decode::{decode_file, AudioFormat, DecodedPcm};
pub use denoiser::{Denoiser, DenoiserConfig, Preset, ProcessingMode};
pub use encode::{AacEncoder, EncodeOptions, OutputFormat};
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
    config: DenoiserConfig,
    backend: Backend,
    encode_opts: EncodeOptions,
    backend_options: BackendOptions,
) -> Result<Audio, String>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    let input = input.as_ref();
    let output = output.as_ref();
    let tag = metadata::read(input)?;
    let mut audio = read_audio(input)?;
    denoise_audio_with_backend_config(&mut audio, config, backend, &backend_options)?;
    write_audio(output, &audio, encode_opts)?;
    if let Some(tag) = tag {
        metadata::write(tag, output)?;
    }
    Ok(audio)
}

/// Process already-decoded audio in place. This is the path used by stdin and
/// embedders that do not have filesystem-backed input.
pub fn denoise_audio_with_backend_config(
    audio: &mut Audio,
    mut config: DenoiserConfig,
    backend: Backend,
    backend_options: &BackendOptions,
) -> Result<std::time::Duration, String> {
    config.sample_rate = audio.sample_rate;
    let t0 = std::time::Instant::now();
    audio.channels = if config.vad {
        process_with_vad(
            backend,
            &audio.channels,
            audio.sample_rate,
            &config,
            backend_options,
        )?
    } else {
        backend::process_channels(
            backend,
            &audio.channels,
            audio.sample_rate,
            &config,
            backend_options,
        )?
    };
    let elapsed = t0.elapsed();
    eprintln!(
        "denoize: {:?} | {}ch x {} frames ({:.2}s) in {:.2?} ({:.1}x realtime)",
        backend,
        audio.channels(),
        audio.frames(),
        audio.frames() as f64 / audio.sample_rate as f64,
        elapsed,
        (audio.frames() as f64 / audio.sample_rate as f64) / elapsed.as_secs_f64().max(1e-9),
    );
    Ok(elapsed)
}

fn process_with_vad(
    backend: Backend,
    channels: &[Vec<f64>],
    sample_rate: u32,
    config: &DenoiserConfig,
    backend_options: &BackendOptions,
) -> Result<Vec<Vec<f64>>, String> {
    let regions = vad::speech_regions(channels, sample_rate);
    let fade_frames = (sample_rate as usize / 50).max(1); // 20 ms
    let mut output: Vec<Vec<f64>> = channels
        .iter()
        .map(|channel| channel.iter().map(|sample| sample * 0.08).collect())
        .collect();
    for region in regions {
        let input: Vec<Vec<f64>> = channels
            .iter()
            .map(|channel| {
                channel[region.start.min(channel.len())..region.end.min(channel.len())].to_vec()
            })
            .collect();
        let enhanced =
            backend::process_channels(backend, &input, sample_rate, config, backend_options)?;
        for (channel_index, enhanced_channel) in enhanced.iter().enumerate() {
            let Some(destination) = output.get_mut(channel_index) else {
                continue;
            };
            let original = &channels[channel_index];
            for (offset, sample) in enhanced_channel.iter().enumerate() {
                let index = region.start + offset;
                if index >= destination.len() || index >= original.len() || index >= region.end {
                    break;
                }
                let target = sample * 0.85 + original[index] * 0.15;
                let weight = vad_mix_weight(offset, region.end - region.start, fade_frames);
                destination[index] = destination[index] * (1.0 - weight) + target * weight;
            }
        }
    }
    Ok(output)
}

fn vad_mix_weight(offset: usize, length: usize, fade_frames: usize) -> f64 {
    let from_start = offset.min(fade_frames) as f64 / fade_frames.max(1) as f64;
    let from_end =
        length.saturating_sub(offset + 1).min(fade_frames) as f64 / fade_frames.max(1) as f64;
    from_start.min(from_end).clamp(0.0, 1.0)
}

#[cfg(test)]
mod vad_mix_tests {
    use super::vad_mix_weight;

    #[test]
    fn fades_vad_region_edges_without_exceeding_unity() {
        assert_eq!(vad_mix_weight(0, 100, 10), 0.0);
        assert_eq!(vad_mix_weight(99, 100, 10), 0.0);
        assert_eq!(vad_mix_weight(50, 100, 10), 1.0);
        assert!((vad_mix_weight(5, 100, 10) - 0.5).abs() < f64::EPSILON);
    }
}
