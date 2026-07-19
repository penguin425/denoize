use super::DecodedPcm;
use opus::{Channels, Decoder};
use std::path::Path;

pub fn decode_ogg_opus(path: &Path) -> Result<DecodedPcm, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("Opus open: {e}"))?;
    let mut packets = ogg::PacketReader::new(std::io::BufReader::new(file));
    let head = packets
        .read_packet()
        .map_err(|e| format!("Ogg read: {e}"))?
        .ok_or("missing OpusHead")?;
    if head.data.len() < 19 || &head.data[..8] != b"OpusHead" {
        return Err("Ogg stream is not Opus".into());
    }
    let count = head.data[9] as usize;
    if !(1..=2).contains(&count) {
        return Err("only mono/stereo Opus is supported".into());
    }
    let pre_skip = u16::from_le_bytes([head.data[10], head.data[11]]) as usize;
    let channels = if count == 1 {
        Channels::Mono
    } else {
        Channels::Stereo
    };
    let mut decoder = Decoder::new(48_000, channels).map_err(|e| format!("Opus decoder: {e}"))?;
    let _tags = packets
        .read_packet()
        .map_err(|e| format!("Ogg tags: {e}"))?;
    let mut decoded = Vec::<f32>::new();
    let mut final_granule = 0_u64;
    let mut buffer = vec![0.0f32; 5_760 * count];
    while let Some(packet) = packets
        .read_packet()
        .map_err(|e| format!("Ogg read: {e}"))?
    {
        final_granule = final_granule.max(packet.absgp_page());
        let frames = decoder
            .decode_float(&packet.data, &mut buffer, false)
            .map_err(|e| format!("Opus decode: {e}"))?;
        decoded.extend_from_slice(&buffer[..frames * count]);
    }
    let skip = (pre_skip * count).min(decoded.len());
    let wanted = final_granule.saturating_sub(pre_skip as u64) as usize * count;
    let end = (skip + wanted).min(decoded.len());
    let mut output = vec![Vec::new(); count];
    for (index, sample) in decoded[skip..end].iter().enumerate() {
        output[index % count].push(*sample as f64);
    }
    Ok(DecodedPcm {
        sample_rate: 48_000,
        channels: output,
    })
}
