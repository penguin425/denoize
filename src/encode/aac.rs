//! Raw AAC-LC in ADTS encoding.

use super::pcm::{lossy_channel_layout, planar_f64_to_interleaved_i16};
use crate::Audio;
use oxideav_aac_encoder::encoder::{EncoderConfig, StreamEncoder, FRAME_LEN};
use std::io::Write;
use std::path::Path;

pub fn write_adts_aac<P: AsRef<Path>>(
    path: P,
    audio: &Audio,
    bitrate_bps: u32,
) -> Result<(), String> {
    let layout = lossy_channel_layout(audio)?;
    let mut encoder = StreamEncoder::new(EncoderConfig {
        sample_rate: audio.sample_rate,
        channels: layout.count,
        bitrate: bitrate_bps,
    })
    .map_err(|error| format!("AAC encoder init: {error}"))?;
    let pcm = planar_f64_to_interleaved_i16(audio, layout);
    let frame_samples = FRAME_LEN * layout.count as usize;
    let mut output = std::io::BufWriter::new(
        std::fs::File::create(path).map_err(|error| format!("create AAC: {error}"))?,
    );
    for input in pcm.chunks(frame_samples) {
        let frame = encoder
            .encode_frame(input)
            .map_err(|error| format!("AAC encode: {error}"))?;
        output
            .write_all(&frame)
            .map_err(|error| format!("write AAC: {error}"))?;
    }
    let final_frame = encoder
        .finish()
        .map_err(|error| format!("AAC finish: {error}"))?;
    output
        .write_all(&final_frame)
        .map_err(|error| format!("write AAC: {error}"))?;
    output
        .flush()
        .map_err(|error| format!("flush AAC: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adts_roundtrip() {
        let sample_rate = 44_100;
        let audio = Audio {
            sample_rate,
            channels: vec![(0..sample_rate / 2)
                .map(|index| {
                    let time = index as f64 / sample_rate as f64;
                    0.2 * (2.0 * std::f64::consts::PI * 330.0 * time).sin()
                })
                .collect()],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let path = std::env::temp_dir().join(format!("denoize-adts-{}.aac", std::process::id()));
        write_adts_aac(&path, &audio, 128_000).unwrap();
        let decoded = crate::decode::decode_file(&path).unwrap();
        assert_eq!(decoded.sample_rate, sample_rate);
        assert!(decoded.frames() > 10_000);
        std::fs::remove_file(path).unwrap();
    }
}
