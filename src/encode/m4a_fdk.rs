//! Optional Fraunhofer FDK-AAC encoder with MP4/M4A muxing.

use std::io::BufWriter;
use std::path::Path;

use fdk_aac_rust::encoder::{
    ConfiguredPureRustEncoder, EncoderParameter, PureRustEncoderParameters,
};
use mp4::{
    AacConfig, AudioObjectType as Mp4Aot, ChannelConfig, FourCC, MediaConfig, Mp4Config, Mp4Sample,
    Mp4Writer, TrackConfig, TrackType,
};

use crate::audio::Audio;

use super::m4a::sample_rate_to_index;
use super::pcm::{lossy_channel_layout, planar_f64_to_interleaved_i16};

pub fn write_m4a_fdk<P: AsRef<Path>>(
    path: P,
    audio: &Audio,
    bitrate_bps: u32,
) -> Result<(), String> {
    let layout = lossy_channel_layout(audio)?;
    let mut parameters = PureRustEncoderParameters::new(layout.count as usize);
    for (parameter, value) in [
        (EncoderParameter::AudioObjectType, 2),
        (EncoderParameter::SampleRate, audio.sample_rate),
        (EncoderParameter::Bitrate, bitrate_bps),
        (EncoderParameter::BitrateMode, 0),
        (
            EncoderParameter::ChannelMode,
            if layout.is_stereo { 2 } else { 1 },
        ),
        (EncoderParameter::ChannelOrder, 1),
        (EncoderParameter::Afterburner, 1),
        (EncoderParameter::TransportMux, 0),
    ] {
        parameters
            .set_parameter(parameter, value)
            .map_err(|error| format!("FDK-AAC parameter: {error}"))?;
    }
    let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters)
        .map_err(|error| format!("FDK-AAC encoder init: {error}"))?;
    let frame_length = encoder.input_samples_per_channel();
    let channel_count = layout.count as usize;
    let input_length = frame_length * channel_count;
    let mut pcm: Vec<f32> = planar_f64_to_interleaved_i16(audio, layout)
        .into_iter()
        .map(|sample| sample as f32 / 32768.0)
        .collect();
    pcm.resize(pcm.len().div_ceil(input_length) * input_length, 0.0);

    let file = std::fs::File::create(path).map_err(|error| format!("create m4a: {error}"))?;
    let writer = BufWriter::new(file);
    let brand = |value: &str| {
        value
            .parse::<FourCC>()
            .map_err(|error| format!("mp4 brand '{value}': {error}"))
    };
    let mut mp4 = Mp4Writer::write_start(
        writer,
        &Mp4Config {
            major_brand: brand("M4A ")?,
            minor_version: 0,
            compatible_brands: vec![brand("M4A ")?, brand("mp42")?, brand("isom")?],
            timescale: audio.sample_rate,
        },
    )
    .map_err(|error| format!("mp4 start: {error}"))?;
    mp4.add_track(&TrackConfig {
        track_type: TrackType::Audio,
        timescale: audio.sample_rate,
        language: "und".into(),
        media_conf: MediaConfig::AacConfig(AacConfig {
            bitrate: bitrate_bps,
            profile: Mp4Aot::AacLowComplexity,
            freq_index: sample_rate_to_index(audio.sample_rate)?,
            chan_conf: if layout.is_stereo {
                ChannelConfig::Stereo
            } else {
                ChannelConfig::Mono
            },
        }),
    })
    .map_err(|error| format!("mp4 add track: {error}"))?;

    let mut pts = 0u64;
    for input in pcm.chunks(input_length) {
        let encoded = encoder
            .encode_interleaved_f32(input)
            .map_err(|error| format!("FDK-AAC encode: {error}"))?;
        if encoded.is_empty() {
            continue;
        }
        mp4.write_sample(
            1,
            &Mp4Sample {
                start_time: pts,
                duration: frame_length as u32,
                rendering_offset: 0,
                is_sync: true,
                bytes: mp4::Bytes::from(encoded),
            },
        )
        .map_err(|error| format!("mp4 sample: {error}"))?;
        pts += frame_length as u64;
    }
    mp4.write_end()
        .map_err(|error| format!("mp4 finalize: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fdk_m4a_roundtrip() {
        let sample_rate = 44_100;
        let samples = (0..sample_rate / 2)
            .map(|index| {
                let time = index as f64 / sample_rate as f64;
                0.2 * (2.0 * std::f64::consts::PI * 440.0 * time).sin()
            })
            .collect();
        let audio = Audio {
            sample_rate,
            channels: vec![samples],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let path = std::env::temp_dir().join(format!("denoize-fdk-{}.m4a", std::process::id()));
        write_m4a_fdk(&path, &audio, 128_000).unwrap();
        let decoded = crate::decode::decode_file(&path).unwrap();
        assert_eq!(decoded.sample_rate, sample_rate);
        assert!(decoded.frames() >= sample_rate as usize / 3);
        std::fs::remove_file(path).unwrap();
    }
}
