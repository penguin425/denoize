//! MP3 encode via `shine-rs` (Pure Rust, LGPL-2.0).

use std::path::Path;

use shine_rs::{Mp3Encoder, Mp3EncoderConfig, StereoMode, SUPPORTED_BITRATES};

use crate::audio::Audio;

use super::pcm::{lossy_channel_layout, planar_f64_to_interleaved_i16};

/// Default MP3 bitrate (kbps).
pub const DEFAULT_MP3_BITRATE: u32 = 192;

/// Write planar `f64` audio to an MP3 file.
pub fn write_mp3<P: AsRef<Path>>(path: P, audio: &Audio, bitrate_kbps: u32) -> Result<(), String> {
    let path = path.as_ref();
    if !shine_rs::SUPPORTED_SAMPLE_RATES.contains(&audio.sample_rate) {
        return Err(format!(
            "MP3 encode: unsupported sample rate {} Hz (supported: {:?})",
            audio.sample_rate,
            shine_rs::SUPPORTED_SAMPLE_RATES,
        ));
    }

    let layout = lossy_channel_layout(audio)?;
    let bitrate = pick_mp3_bitrate(bitrate_kbps, audio.sample_rate);
    let stereo_mode = if layout.is_stereo {
        StereoMode::JointStereo
    } else {
        StereoMode::Mono
    };

    let config = Mp3EncoderConfig {
        sample_rate: audio.sample_rate,
        bitrate,
        channels: layout.count,
        stereo_mode,
        copyright: false,
        original: true,
    };

    let pcm = planar_f64_to_interleaved_i16(audio, layout);
    let mut encoder = Mp3Encoder::new(config).map_err(|e| format!("mp3 encoder: {e}"))?;

    let mut mp3 = Vec::new();
    for frame in encoder
        .encode_interleaved(&pcm)
        .map_err(|e| format!("mp3 encode: {e}"))?
    {
        mp3.extend(frame);
    }
    mp3.extend(encoder.finish().map_err(|e| format!("mp3 finish: {e}"))?);

    std::fs::write(path, &mp3).map_err(|e| format!("write mp3: {e}"))?;
    Ok(())
}

fn pick_mp3_bitrate(requested: u32, sample_rate: u32) -> u32 {
    let candidates: Vec<u32> = SUPPORTED_BITRATES
        .iter()
        .copied()
        .filter(|b| *b <= requested)
        .collect();
    let fallback = *candidates.last().unwrap_or(&SUPPORTED_BITRATES[0]);
    for &b in candidates.iter().rev() {
        let cfg = Mp3EncoderConfig {
            sample_rate,
            bitrate: b,
            channels: 2,
            stereo_mode: StereoMode::Stereo,
            ..Default::default()
        };
        if cfg.validate().is_ok() {
            return b;
        }
    }
    fallback
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::SampleFormat;

    fn sine_stereo(sr: u32, secs: f32) -> Audio {
        let frames = (sr as f32 * secs) as usize;
        let mut l = Vec::with_capacity(frames);
        let mut r = Vec::with_capacity(frames);
        for i in 0..frames {
            let t = i as f64 / sr as f64;
            let v = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.25;
            l.push(v);
            r.push(v * 0.8);
        }
        Audio {
            sample_rate: sr,
            channels: vec![l, r],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("denoize_mp3_{}_{}", std::process::id(), name));
        p
    }

    #[test]
    fn mp3_roundtrip_decode() {
        let path = tmp("rt.mp3");
        let audio = sine_stereo(44100, 0.5);
        write_mp3(&path, &audio, 128).unwrap();
        assert!(path.metadata().unwrap().len() > 100);

        let decoded = crate::decode::decode_file(&path).unwrap();
        assert_eq!(decoded.sample_rate, 44100);
        assert_eq!(decoded.n_channels(), 2);
        assert!(decoded.frames() > 10000);

        // Lossy but should retain energy
        let rms_in: f64 =
            audio.channels[0].iter().map(|s| s * s).sum::<f64>() / audio.frames() as f64;
        let rms_out: f64 =
            decoded.channels[0].iter().map(|s| s * s).sum::<f64>() / decoded.frames() as f64;
        assert!(rms_out > 0.01);
        assert!(rms_out < rms_in * 2.0);

        let _ = std::fs::remove_file(&path);
    }
}
