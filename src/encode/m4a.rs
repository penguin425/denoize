//! M4A (AAC-LC in MP4) encode — Pure-Rust `oxideav-aac` + `mp4` muxer.

use std::io::BufWriter;
use std::path::Path;

use mp4::{
    AacConfig, AudioObjectType as Mp4Aot, ChannelConfig, FourCC, MediaConfig, Mp4Config, Mp4Sample,
    Mp4Writer, SampleFreqIndex, TrackConfig, TrackType,
};
use oxideav_aac::adts::ADTS_HEADER_BYTES_NO_CRC;
use oxideav_aac::encoder::{EncoderConfig, StreamEncoder, FRAME_LEN};

use crate::audio::Audio;

use super::pcm::{lossy_channel_layout, planar_f64_to_interleaved_i16};

/// Default AAC bitrate (bps, not kbps).
pub const DEFAULT_M4A_BITRATE: u32 = 192_000;

/// Write planar `f64` audio to an M4A file.
pub fn write_m4a<P: AsRef<Path>>(path: P, audio: &Audio, bitrate_bps: u32) -> Result<(), String> {
    let path = path.as_ref();
    let sample_rate = audio.sample_rate;
    let freq_index = sample_rate_to_index(sample_rate)?;
    let layout = lossy_channel_layout(audio)?;

    let chan_conf = if layout.is_stereo {
        ChannelConfig::Stereo
    } else {
        ChannelConfig::Mono
    };

    let enc_config = EncoderConfig {
        sample_rate,
        channels: layout.count,
        bitrate: bitrate_bps,
    };
    let mut encoder =
        StreamEncoder::new(enc_config).map_err(|e| format!("aac encoder init: {e}"))?;

    let pcm = planar_f64_to_interleaved_i16(audio, layout);
    let out_ch = layout.count as usize;
    let hop = FRAME_LEN * out_ch;

    let file = std::fs::File::create(path).map_err(|e| format!("create m4a: {e}"))?;
    let writer = BufWriter::new(file);

    let parse_brand = |s: &str| -> Result<FourCC, String> {
        s.parse::<FourCC>()
            .map_err(|e| format!("mp4 brand '{s}': {e}"))
    };
    let mp4_config = Mp4Config {
        major_brand: parse_brand("M4A ")?,
        minor_version: 0,
        compatible_brands: vec![
            parse_brand("M4A ")?,
            parse_brand("mp42")?,
            parse_brand("isom")?,
        ],
        timescale: sample_rate,
    };

    let mut mp4_writer =
        Mp4Writer::write_start(writer, &mp4_config).map_err(|e| format!("mp4 start: {e}"))?;

    let aac_config = AacConfig {
        bitrate: bitrate_bps,
        profile: Mp4Aot::AacLowComplexity,
        freq_index,
        chan_conf,
    };
    let track_config = TrackConfig {
        track_type: TrackType::Audio,
        timescale: sample_rate,
        language: "und".into(),
        media_conf: MediaConfig::AacConfig(aac_config),
    };

    mp4_writer
        .add_track(&track_config)
        .map_err(|e| format!("mp4 add track: {e}"))?;

    let mut pts = 0u64;
    let mut off = 0usize;

    while off < pcm.len() {
        let end = (off + hop).min(pcm.len());
        let chunk = &pcm[off..end];
        off = end;
        write_aac_sample(&mut mp4_writer, &mut encoder, chunk, &mut pts)?;
    }

    let flush = encoder.finish().map_err(|e| format!("aac finish: {e}"))?;
    write_raw_aac_frame(&mut mp4_writer, &flush, &mut pts)?;

    mp4_writer
        .write_end()
        .map_err(|e| format!("mp4 finalize: {e}"))?;

    Ok(())
}

fn write_aac_sample<W: std::io::Write + std::io::Seek>(
    mp4_writer: &mut Mp4Writer<W>,
    encoder: &mut StreamEncoder,
    pcm_chunk: &[i16],
    pts: &mut u64,
) -> Result<(), String> {
    let adts = encoder
        .encode_frame(pcm_chunk)
        .map_err(|e| format!("aac encode: {e}"))?;
    write_raw_aac_frame(mp4_writer, &adts, pts)
}

fn write_raw_aac_frame<W: std::io::Write + std::io::Seek>(
    mp4_writer: &mut Mp4Writer<W>,
    adts_frame: &[u8],
    pts: &mut u64,
) -> Result<(), String> {
    if adts_frame.len() <= ADTS_HEADER_BYTES_NO_CRC {
        return Ok(());
    }
    let raw = &adts_frame[ADTS_HEADER_BYTES_NO_CRC..];
    if raw.is_empty() {
        return Ok(());
    }
    let sample = Mp4Sample {
        start_time: *pts,
        duration: FRAME_LEN as u32,
        rendering_offset: 0,
        is_sync: true,
        bytes: mp4::Bytes::from(raw.to_vec()),
    };
    mp4_writer
        .write_sample(1, &sample)
        .map_err(|e| format!("mp4 sample: {e}"))?;
    *pts += FRAME_LEN as u64;
    Ok(())
}

fn sample_rate_to_index(sr: u32) -> Result<SampleFreqIndex, String> {
    match sr {
        96000 => Ok(SampleFreqIndex::Freq96000),
        88200 => Ok(SampleFreqIndex::Freq88200),
        64000 => Ok(SampleFreqIndex::Freq64000),
        48000 => Ok(SampleFreqIndex::Freq48000),
        44100 => Ok(SampleFreqIndex::Freq44100),
        32000 => Ok(SampleFreqIndex::Freq32000),
        24000 => Ok(SampleFreqIndex::Freq24000),
        22050 => Ok(SampleFreqIndex::Freq22050),
        16000 => Ok(SampleFreqIndex::Freq16000),
        12000 => Ok(SampleFreqIndex::Freq12000),
        11025 => Ok(SampleFreqIndex::Freq11025),
        8000 => Ok(SampleFreqIndex::Freq8000),
        7350 => Ok(SampleFreqIndex::Freq7350),
        _ => Err(format!(
            "M4A encode: unsupported sample rate {sr} Hz (AAC standard rates only)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::SampleFormat;

    fn sine_mono(sr: u32, secs: f32) -> Audio {
        let frames = (sr as f32 * secs) as usize;
        let mut ch = Vec::with_capacity(frames);
        for i in 0..frames {
            let t = i as f64 / sr as f64;
            ch.push((2.0 * std::f64::consts::PI * 330.0 * t).sin() * 0.3);
        }
        Audio {
            sample_rate: sr,
            channels: vec![ch],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("denoize_m4a_{}_{}", std::process::id(), name));
        p
    }

    #[test]
    fn m4a_roundtrip_decode() {
        let path = tmp("rt.m4a");
        let audio = sine_mono(44100, 0.5);
        write_m4a(&path, &audio, 128_000).unwrap();
        assert!(path.metadata().unwrap().len() > 100);

        let decoded = crate::decode::decode_file(&path).unwrap();
        assert_eq!(decoded.sample_rate, 44100);
        assert!(decoded.frames() > 10000);

        let rms_out: f64 =
            decoded.channels[0].iter().map(|s| s * s).sum::<f64>() / decoded.frames() as f64;
        assert!(rms_out > 0.005);

        let _ = std::fs::remove_file(&path);
    }
}
