//! Lightweight energy VAD and speech-region segmentation.

/// Inclusive-exclusive sample range containing speech plus context padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeechRegion {
    pub start: usize,
    pub end: usize,
}

/// Detect speech regions from planar audio using 20 ms RMS frames.
pub fn speech_regions(channels: &[Vec<f64>], sample_rate: u32) -> Vec<SpeechRegion> {
    let frames = channels.iter().map(Vec::len).max().unwrap_or(0);
    if frames == 0 || channels.is_empty() {
        return Vec::new();
    }
    let window = (sample_rate as usize / 50).max(1);
    let mut levels = Vec::with_capacity(frames.div_ceil(window));
    for start in (0..frames).step_by(window) {
        let end = (start + window).min(frames);
        let mut energy = 0.0;
        let mut count = 0usize;
        for channel in channels {
            for sample in &channel[start.min(channel.len())..end.min(channel.len())] {
                energy += sample * sample;
                count += 1;
            }
        }
        let rms = (energy / count.max(1) as f64).sqrt();
        levels.push(20.0 * rms.max(1e-10).log10());
    }
    let mut sorted = levels.clone();
    sorted.sort_by(f64::total_cmp);
    let floor = sorted[sorted.len() / 5];
    let peak = sorted.last().copied().unwrap_or(-200.0);
    let threshold = if peak - floor < 6.0 {
        if peak > -50.0 {
            floor - 1.0
        } else {
            -50.0
        }
    } else {
        (floor + 6.0).clamp(-55.0, -25.0)
    };
    let hangover_frames = 10; // 200 ms
    let mut active = vec![false; levels.len()];
    let mut hangover = 0usize;
    for (index, level) in levels.iter().enumerate() {
        if *level >= threshold {
            hangover = hangover_frames;
            active[index] = true;
        } else if hangover > 0 {
            active[index] = true;
            hangover -= 1;
        }
    }

    let padding = sample_rate as usize / 10; // 100 ms
    let merge_gap = sample_rate as usize * 3 / 10; // 300 ms
    let mut regions: Vec<SpeechRegion> = Vec::new();
    let mut index = 0usize;
    while index < active.len() {
        if !active[index] {
            index += 1;
            continue;
        }
        let first = index;
        while index < active.len() && active[index] {
            index += 1;
        }
        let mut region = SpeechRegion {
            start: (first * window).saturating_sub(padding),
            end: (index * window + padding).min(frames),
        };
        if let Some(previous) = regions.last_mut() {
            if region.start.saturating_sub(previous.end) <= merge_gap {
                previous.end = region.end;
                continue;
            }
        }
        region.end = region.end.max(region.start);
        regions.push(region);
    }
    regions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_separated_speech_and_skips_long_silence() {
        let mut audio = vec![0.0; 48_000 * 3];
        for sample in &mut audio[48_000..48_000 + 9_600] {
            *sample = 0.2;
        }
        let regions = speech_regions(&[audio], 48_000);
        assert_eq!(regions.len(), 1);
        assert!(regions[0].start < 48_000);
        assert!(regions[0].end < 48_000 * 2);
    }

    #[test]
    fn silence_has_no_regions() {
        assert!(speech_regions(&[vec![0.0; 16_000]], 16_000).is_empty());
    }
}
