//! Raw ADTS AAC decoding.

use super::pcm::DecodedPcm;
use oxideav_aac::decode::StreamDecoder;
use std::path::Path;

pub fn decode_adts(path: &Path) -> Result<DecodedPcm, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read AAC: {error}"))?;
    let frames = StreamDecoder::new()
        .decode_all(&bytes)
        .map_err(|error| format!("decode ADTS AAC: {error}"))?;
    let first = frames.first().ok_or("ADTS AAC contains no frames")?;
    let sample_rate = first.sample_rate;
    let channel_count = first.channels;
    let mut channels = vec![Vec::new(); channel_count];
    for frame in frames {
        if frame.sample_rate != sample_rate || frame.channels != channel_count {
            return Err("ADTS AAC changes sample rate or channel count mid-stream".into());
        }
        for samples in frame.pcm.chunks_exact(channel_count) {
            for (channel, sample) in channels.iter_mut().zip(samples) {
                channel.push(*sample as f64 / 32768.0);
            }
        }
    }
    Ok(DecodedPcm {
        sample_rate,
        channels,
    })
}
