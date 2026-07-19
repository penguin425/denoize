//! Audio encode layer — WAV / MP3 / M4A output.
//!
//! | Format | Backend |
//! |--------|---------|
//! | WAV | `hound` (lossless, preserves bit depth) |
//! | MP3 | `shine-rs` (Pure Rust) |
//! | M4A | `oxideav-aac` + `mp4` mux (Pure-Rust AAC-LC) |

mod m4a;
mod mp3;
mod pcm;

pub use m4a::{write_m4a, DEFAULT_M4A_BITRATE};
pub use mp3::{write_mp3, DEFAULT_MP3_BITRATE};

use std::path::Path;

use crate::audio::{write_wav, Audio};

/// Output container inferred from file extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Wav,
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
        OutputFormat::Mp3 => write_mp3(path, audio, options.mp3_bitrate_kbps),
        OutputFormat::M4a => write_m4a(path, audio, options.m4a_bitrate_bps),
    }
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
        assert!(OutputFormat::from_path(Path::new("out.flac")).is_err());
    }
}
