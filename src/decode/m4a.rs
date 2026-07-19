//! M4A / MP4-AAC decoder — `mp4` demux + Pure-Rust `oxideav-aac` AAC-LC decode.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use mp4::{ChannelConfig, MediaType, Mp4Reader};
use oxideav_aac::decode::StreamDecoder;

use super::pcm::DecodedPcm;

/// Decode M4A/MP4-AAC from path.
pub fn decode_m4a(path: &Path) -> Result<DecodedPcm, String> {
    let file = File::open(path).map_err(|e| format!("open m4a: {e}"))?;
    let size = file.metadata().map_err(|e| format!("stat m4a: {e}"))?.len();
    let reader = BufReader::new(file);
    let mut mp4 = Mp4Reader::read_header(reader, size).map_err(|e| format!("mp4 parse: {e}"))?;

    let track_id = mp4
        .tracks()
        .values()
        .find(|t| t.media_type().ok() == Some(MediaType::AAC))
        .map(|t| t.track_id())
        .ok_or("no AAC audio track found in M4A/MP4")?;

    let track = mp4
        .tracks()
        .get(&track_id)
        .ok_or("AAC track metadata missing")?;

    let profile = track
        .audio_profile()
        .map_err(|e| format!("aac profile: {e}"))?;
    let freq_index = track
        .sample_freq_index()
        .map_err(|e| format!("aac sample rate: {e}"))?;
    let channel_config = track
        .channel_config()
        .map_err(|e| format!("aac channels: {e}"))?;

    let sample_rate = freq_index.freq();
    let aot = profile as u8;
    let fs_index = freq_index as u8;
    let chan_conf = channel_config as u8;
    let n_ch = channel_config_to_count(channel_config);

    let mut decoder = StreamDecoder::new();
    let mut channels: Vec<Vec<f64>> = (0..n_ch).map(|_| Vec::new()).collect();

    let mut sample_id = 1u32;
    loop {
        let sample = match mp4.read_sample(track_id, sample_id) {
            Ok(Some(s)) => s,
            Ok(None) => break,
            Err(e) => return Err(format!("mp4 read sample: {e}")),
        };
        sample_id += 1;

        if sample.bytes.is_empty() {
            continue;
        }

        let frame = decoder
            .decode_raw_data_block(aot, fs_index, sample_rate, chan_conf, 1, &sample.bytes)
            .map_err(|e| format!("aac decode: {e}"))?;

        if frame.channels == 0 || frame.pcm.is_empty() {
            continue;
        }

        append_interleaved_i16(&mut channels, &frame.pcm, frame.channels);
    }

    if channels.first().map(|c| c.is_empty()).unwrap_or(true) {
        return Err("M4A decode produced no samples".into());
    }

    Ok(DecodedPcm {
        sample_rate,
        channels,
    })
}

fn channel_config_to_count(cfg: ChannelConfig) -> usize {
    match cfg {
        ChannelConfig::Mono => 1,
        ChannelConfig::Stereo => 2,
        ChannelConfig::Three => 3,
        ChannelConfig::Four => 4,
        ChannelConfig::Five => 5,
        ChannelConfig::FiveOne => 6,
        ChannelConfig::SevenOne => 8,
    }
}

fn append_interleaved_i16(channels: &mut [Vec<f64>], interleaved: &[i16], n_ch: usize) {
    let frames = interleaved.len() / n_ch.max(1);
    for f in 0..frames {
        for ch in 0..n_ch.min(channels.len()) {
            let v = interleaved[f * n_ch + ch] as f64 / 32768.0;
            channels[ch].push(v.clamp(-1.0, 1.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_file() {
        assert!(decode_m4a(Path::new("/nonexistent/file.m4a")).is_err());
    }
}
