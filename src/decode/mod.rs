//! denoize 自作デコード層 — MP3 / M4A / WAV を高品質 PCM (`f64`) へ。
//!
//! # 設計方針（劣化最小）
//! - デコード出力は `f32` → `f64` へ拡張のみ（再量子化なし）
//! - サンプルレート変換なし（ソースレートを維持）
//! - 内部パイプラインは 32-bit float 相当精度で denoise へ渡す
//!
//! # バックエンド
//! | 形式 | 実装 |
//! |------|------|
//! | WAV  | `hound`（既存） |
//! | MP3  | `nanomp3`（Pure Rust / minimp3 移植） |
//! | M4A  | `mp4` demux + `oxideav-aac` Pure-Rust AAC-LC decode |

mod m4a;
mod mp3;
mod opus;
mod pcm;

pub use pcm::DecodedPcm;

use std::path::Path;

/// Detected container / codec family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioFormat {
    Wav,
    Flac,
    OggOpus,
    Mp3,
    M4a,
    /// AAC in ADTS (.aac) — not yet supported.
    AacAdts,
    Unknown,
}

impl AudioFormat {
    /// Sniff from file content and extension.
    pub fn detect(path: &Path, header: &[u8]) -> Self {
        if header.len() >= 12 {
            if &header[0..4] == b"RIFF" && header.len() >= 12 && &header[8..12] == b"WAVE" {
                return AudioFormat::Wav;
            }
            if &header[0..4] == b"fLaC" {
                return AudioFormat::Flac;
            }
            if &header[0..4] == b"OggS" {
                return AudioFormat::OggOpus;
            }
            if &header[4..8] == b"ftyp" {
                return AudioFormat::M4a;
            }
            // ADTS has a 12-bit sync word and its two layer bits are always 0.
            // Check it before the broader 11-bit MPEG audio sync test.
            if header[0] == 0xFF && (header[1] & 0xF6) == 0xF0 {
                return AudioFormat::AacAdts;
            }
            if &header[0..3] == b"ID3" {
                return AudioFormat::Mp3;
            }
            if header[0] == 0xFF && (header[1] & 0xE0) == 0xE0 {
                return AudioFormat::Mp3;
            }
        }

        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref()
        {
            Some("wav") => AudioFormat::Wav,
            Some("flac") => AudioFormat::Flac,
            Some("opus" | "ogg") => AudioFormat::OggOpus,
            Some("mp3") => AudioFormat::Mp3,
            Some("m4a" | "m4b" | "m4p" | "mp4") => AudioFormat::M4a,
            Some("aac") => AudioFormat::AacAdts,
            _ => AudioFormat::Unknown,
        }
    }

    pub fn extensions(self) -> &'static [&'static str] {
        match self {
            AudioFormat::Wav => &["wav"],
            AudioFormat::Flac => &["flac"],
            AudioFormat::OggOpus => &["opus", "ogg"],
            AudioFormat::Mp3 => &["mp3"],
            AudioFormat::M4a => &["m4a", "m4b", "mp4", "aac"],
            AudioFormat::AacAdts => &["aac"],
            AudioFormat::Unknown => &[],
        }
    }
}

/// Decode any supported audio file to high-fidelity planar PCM.
pub fn decode_file(path: &Path) -> Result<DecodedPcm, String> {
    let header = read_header(path, 4096)?;
    let fmt = AudioFormat::detect(path, &header);

    match fmt {
        AudioFormat::Wav => decode_wav(path),
        AudioFormat::Flac => decode_flac(path),
        AudioFormat::OggOpus => opus::decode_ogg_opus(path),
        AudioFormat::Mp3 => mp3::decode_mp3_file(path),
        AudioFormat::M4a => m4a::decode_m4a(path),
        AudioFormat::AacAdts => Err("ADTS .aac not yet supported; convert to M4A or WAV".into()),
        AudioFormat::Unknown => Err(format!(
            "unsupported audio format ({}); supported input: wav, mp3, m4a",
            path.display()
        )),
    }
}

fn decode_flac(path: &Path) -> Result<DecodedPcm, String> {
    let mut reader = claxon::FlacReader::open(path).map_err(|e| format!("FLAC open: {e}"))?;
    let info = reader.streaminfo();
    let channels = info.channels as usize;
    let scale = 1.0 / (1_u64 << (info.bits_per_sample - 1)) as f64;
    let mut output = vec![Vec::new(); channels];
    for (index, sample) in reader.samples().enumerate() {
        output[index % channels]
            .push(sample.map_err(|e| format!("FLAC decode: {e}"))? as f64 * scale);
    }
    Ok(DecodedPcm {
        sample_rate: info.sample_rate,
        channels: output,
    })
}

fn read_header(path: &Path, n: usize) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
    let mut buf = vec![0u8; n];
    let got = f.read(&mut buf).map_err(|e| format!("read: {e}"))?;
    buf.truncate(got);
    Ok(buf)
}

fn decode_wav(path: &Path) -> Result<DecodedPcm, String> {
    let audio = crate::audio::read_wav(path)?;
    Ok(DecodedPcm {
        sample_rate: audio.sample_rate,
        channels: audio.channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_wav() {
        let h = b"RIFF\x00\x00\x00\x00WAVE";
        assert_eq!(AudioFormat::detect(Path::new("x.wav"), h), AudioFormat::Wav);
    }

    #[test]
    fn detect_mp3_id3() {
        assert_eq!(
            AudioFormat::detect(Path::new("x.mp3"), b"ID3"),
            AudioFormat::Mp3
        );
    }

    #[test]
    fn detect_m4a_ftyp() {
        let h = b"\x00\x00\x00\x20ftypM4A ";
        assert_eq!(AudioFormat::detect(Path::new("x.m4a"), h), AudioFormat::M4a);
    }

    #[test]
    fn detect_adts_before_mp3() {
        let h = b"\xff\xf1\x50\x80\x00\x1f\xfc\x00\x00\x00\x00\x00";
        assert_eq!(
            AudioFormat::detect(Path::new("x.aac"), h),
            AudioFormat::AacAdts
        );
        assert_eq!(
            AudioFormat::detect(Path::new("x.aac"), b""),
            AudioFormat::AacAdts
        );
    }
}
