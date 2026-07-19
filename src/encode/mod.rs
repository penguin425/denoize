//! Audio encode layer — WAV / MP3 / M4A output.
//!
//! | Format | Backend |
//! |--------|---------|
//! | WAV | `hound` (lossless, preserves bit depth) |
//! | MP3 | `shine-rs` (Pure Rust) |
//! | M4A | `oxideav-aac` + `mp4` mux (Pure-Rust AAC-LC) |

#[cfg(feature = "m4a-encode")]
mod m4a;
mod mp3;
mod opus;
mod pcm;

#[cfg(feature = "m4a-encode")]
pub use m4a::write_m4a;
pub use mp3::{write_mp3, DEFAULT_MP3_BITRATE};

/// Default AAC bitrate (bps, not kbps).
pub const DEFAULT_M4A_BITRATE: u32 = 192_000;

use std::path::Path;

use crate::audio::{write_wav, Audio};

/// Output container inferred from file extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Wav,
    Flac,
    OggOpus,
    Mp3,
    M4a,
}

impl OutputFormat {
    pub fn from_path(path: &Path) -> Result<Self, String> {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref()
        {
            Some("wav") => Ok(OutputFormat::Wav),
            Some("flac") => Ok(OutputFormat::Flac),
            Some("opus" | "ogg") => Ok(OutputFormat::OggOpus),
            Some("mp3") => Ok(OutputFormat::Mp3),
            Some("m4a" | "m4b" | "mp4") => Ok(OutputFormat::M4a),
            Some(ext) => Err(format!(
                "unsupported output format '.{ext}'; use .wav, .mp3, or .m4a"
            )),
            None => Err("output path has no extension; use .wav, .mp3, or .m4a".into()),
        }
    }
}

/// Encoding options for lossy outputs.
#[derive(Clone, Copy, Debug)]
pub struct EncodeOptions {
    /// MP3 constant bitrate in kbps.
    pub mp3_bitrate_kbps: u32,
    /// AAC constant bitrate in bps.
    pub m4a_bitrate_bps: u32,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            mp3_bitrate_kbps: DEFAULT_MP3_BITRATE,
            m4a_bitrate_bps: DEFAULT_M4A_BITRATE,
        }
    }
}

/// Write audio to a file; format is chosen from the path extension.
pub fn write_audio<P: AsRef<Path>>(
    path: P,
    audio: &Audio,
    options: EncodeOptions,
) -> Result<(), String> {
    let path = path.as_ref();
    match OutputFormat::from_path(path)? {
        OutputFormat::Wav => write_wav(path, audio),
        OutputFormat::Flac => write_flac(path, audio),
        OutputFormat::OggOpus => opus::write_ogg_opus(path, audio, 128_000),
        OutputFormat::Mp3 => write_mp3(path, audio, options.mp3_bitrate_kbps),
        OutputFormat::M4a => {
            #[cfg(feature = "m4a-encode")]
            {
                write_m4a(path, audio, options.m4a_bitrate_bps)
            }
            #[cfg(not(feature = "m4a-encode"))]
            {
                let _ = options;
                Err("M4A output is unavailable in the crates.io build; use WAV/MP3 or a GitHub release binary".into())
            }
        }
    }
}

fn write_flac(path: &Path, audio: &Audio) -> Result<(), String> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;
    let bits = audio.bits_per_sample.clamp(8, 24) as usize;
    let scale = (1_i64 << (bits - 1)) as f64;
    let mut samples = Vec::with_capacity(audio.frames() * audio.channels());
    for frame in 0..audio.frames() {
        for channel in &audio.channels {
            samples.push(
                (channel[frame].clamp(-1.0, 1.0) * scale)
                    .round()
                    .clamp(-scale, scale - 1.0) as i32,
            );
        }
    }
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|e| format!("FLAC config: {:?}", e.1))?;
    let source = flacenc::source::MemSource::from_samples(
        &samples,
        audio.channels(),
        bits,
        audio.sample_rate as usize,
    );
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| format!("FLAC encode: {e}"))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| format!("FLAC serialize: {e}"))?;
    std::fs::write(path, sink.as_slice()).map_err(|e| format!("FLAC write: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_output_formats() {
        assert_eq!(
            OutputFormat::from_path(Path::new("out.mp3")).unwrap(),
            OutputFormat::Mp3
        );
        assert_eq!(
            OutputFormat::from_path(Path::new("out.m4a")).unwrap(),
            OutputFormat::M4a
        );
        assert_eq!(
            OutputFormat::from_path(Path::new("out.flac")).unwrap(),
            OutputFormat::Flac
        );
        assert_eq!(
            OutputFormat::from_path(Path::new("out.opus")).unwrap(),
            OutputFormat::OggOpus
        );
    }
}
