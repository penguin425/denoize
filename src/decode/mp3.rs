//! MP3 decoder — Pure Rust via `nanomp3` (minimp3 algorithm, vendored translation).
//!
//! Decodes to 32-bit float internally, accumulated as `f64` for the denoise pipeline.
//! No resampling; original sample rate is preserved.

use nanomp3::{Channels, Decoder, FrameInfo, MAX_SAMPLES_PER_FRAME};

use super::pcm::DecodedPcm;

/// Skip ID3v2 tag if present at the start of an MP3 file.
fn skip_id3(data: &[u8]) -> usize {
    if data.len() >= 10 && &data[0..3] == b"ID3" {
        let size = ((data[6] as usize & 0x7f) << 21)
            | ((data[7] as usize & 0x7f) << 14)
            | ((data[8] as usize & 0x7f) << 7)
            | (data[9] as usize & 0x7f);
        return (10 + size).min(data.len());
    }
    0
}

/// Find next MP3 sync word (11 bits set: 0xFFE).
fn find_sync(data: &[u8]) -> Option<usize> {
    data.windows(2)
        .position(|w| w[0] == 0xFF && (w[1] & 0xE0) == 0xE0)
}

/// Decode MP3 bytes to high-fidelity planar PCM.
pub fn decode_mp3(data: &[u8]) -> Result<DecodedPcm, String> {
    if data.is_empty() {
        return Err("empty MP3 data".into());
    }

    let start = skip_id3(data);
    let data = &data[start..];

    let mut decoder = Decoder::new();
    let mut pcm_buf = vec![0.0f32; MAX_SAMPLES_PER_FRAME];

    let mut out_l: Vec<f64> = Vec::new();
    let mut out_r: Vec<f64> = Vec::new();
    let mut sample_rate: Option<u32> = None;
    let mut is_stereo = false;

    let mut pos = find_sync(data).unwrap_or(0);

    while pos < data.len() {
        let slice = &data[pos..];
        if slice.len() < 4 {
            break;
        }

        let (consumed, info) = decoder.decode(slice, &mut pcm_buf);

        if consumed == 0 {
            // Lost sync — scan forward.
            if let Some(off) = find_sync(&data[pos + 1..]) {
                pos += 1 + off;
            } else {
                break;
            }
            continue;
        }

        pos += consumed;

        if let Some(info) = info {
            append_frame(&info, &pcm_buf, &mut out_l, &mut out_r);
            if sample_rate.is_none() {
                sample_rate = Some(info.sample_rate);
                is_stereo = info.channels == Channels::Stereo;
            }
        }
    }

    let sample_rate = sample_rate.ok_or("no valid MP3 frames found")?;

    let channels = if is_stereo {
        vec![out_l, out_r]
    } else {
        vec![out_l]
    };

    Ok(DecodedPcm {
        sample_rate,
        channels,
    })
}

fn append_frame(info: &FrameInfo, pcm: &[f32], out_l: &mut Vec<f64>, out_r: &mut Vec<f64>) {
    let n_ch = info.channels.num() as usize;
    let n_frames = info.samples_produced / n_ch;
    if n_frames == 0 {
        return;
    }
    let samples = &pcm[..info.samples_produced];
    if n_ch == 1 {
        for &s in samples.iter().take(n_frames) {
            out_l.push(super::pcm::f32_to_f64(s));
        }
    } else {
        for i in 0..n_frames {
            out_l.push(super::pcm::f32_to_f64(samples[i * 2]));
            out_r.push(super::pcm::f32_to_f64(samples[i * 2 + 1]));
        }
    }
}

/// Decode MP3 from file (reads entire file — MP3 frame parsing requires seeking context).
pub fn decode_mp3_file(path: &std::path::Path) -> Result<DecodedPcm, String> {
    let data = std::fs::read(path).map_err(|e| format!("read mp3: {e}"))?;
    decode_mp3(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty() {
        assert!(decode_mp3(&[]).is_err());
    }

    #[test]
    fn rejects_garbage() {
        assert!(decode_mp3(b"not mp3 data at all").is_err());
    }
}
