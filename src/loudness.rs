//! EBU R128 integrated-loudness normalization with a true-peak ceiling.

use crate::Audio;
use ebur128::{EbuR128, Mode};

#[derive(Clone, Copy, Debug)]
pub struct LoudnessReport {
    pub input_lufs: f64,
    pub output_lufs: f64,
    pub true_peak_dbtp: f64,
    pub gain_db: f64,
}

/// Apply a constant gain toward `target_lufs`, constrained by `peak_limit_dbtp`.
pub fn normalize(
    audio: &mut Audio,
    target_lufs: f64,
    peak_limit_dbtp: f64,
) -> Result<LoudnessReport, String> {
    if !target_lufs.is_finite() || !(-70.0..=0.0).contains(&target_lufs) {
        return Err("loudness target must be between -70 and 0 LUFS".into());
    }
    if !peak_limit_dbtp.is_finite() || !(-20.0..=0.0).contains(&peak_limit_dbtp) {
        return Err("true-peak limit must be between -20 and 0 dBTP".into());
    }
    let (input_lufs, input_peak) = measure(audio)?;
    let loudness_gain = target_lufs - input_lufs;
    let peak_gain = peak_limit_dbtp - input_peak;
    let gain_db = loudness_gain.min(peak_gain);
    let gain = 10f64.powf(gain_db / 20.0);
    for channel in &mut audio.channels {
        for sample in channel {
            *sample *= gain;
        }
    }
    let (output_lufs, true_peak_dbtp) = measure(audio)?;
    Ok(LoudnessReport {
        input_lufs,
        output_lufs,
        true_peak_dbtp,
        gain_db,
    })
}

pub fn measure(audio: &Audio) -> Result<(f64, f64), String> {
    let channels = audio.channels();
    if channels == 0 || audio.frames() == 0 {
        return Err("cannot measure empty audio".into());
    }
    let mut analyzer = EbuR128::new(
        channels as u32,
        audio.sample_rate,
        Mode::I | Mode::TRUE_PEAK,
    )
    .map_err(|error| format!("initialize loudness analyzer: {error}"))?;
    let mut interleaved = Vec::with_capacity(audio.frames() * channels);
    for frame in 0..audio.frames() {
        for channel in &audio.channels {
            interleaved.push(channel.get(frame).copied().unwrap_or(0.0));
        }
    }
    analyzer
        .add_frames_f64(&interleaved)
        .map_err(|error| format!("analyze loudness: {error}"))?;
    let loudness = analyzer
        .loudness_global()
        .map_err(|error| format!("measure integrated loudness: {error}"))?;
    if !loudness.is_finite() {
        return Err("integrated loudness is undefined (audio may be silent or too short)".into());
    }
    let mut peak = 0.0f64;
    for channel in 0..channels {
        peak = peak.max(
            analyzer
                .true_peak(channel as u32)
                .map_err(|error| format!("measure true peak: {error}"))?,
        );
    }
    Ok((loudness, 20.0 * peak.max(1e-10).log10()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reaches_loudness_target_without_exceeding_true_peak() {
        let sample_rate = 48_000;
        let channel = (0..sample_rate * 2)
            .map(|index| {
                let time = index as f64 / sample_rate as f64;
                0.08 * (2.0 * std::f64::consts::PI * 440.0 * time).sin()
            })
            .collect();
        let mut audio = Audio {
            sample_rate,
            channels: vec![channel],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let report = normalize(&mut audio, -20.0, -1.0).unwrap();
        assert!((report.output_lufs + 20.0).abs() < 0.1);
        assert!(report.true_peak_dbtp <= -1.0 + 1e-6);
    }
}
