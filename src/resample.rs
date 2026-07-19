//! Band-limited, channel-synchronous sample-rate conversion.

use rubato::{FftFixedIn, Resampler};

const CHUNK_FRAMES: usize = 1024;
const SUB_CHUNKS: usize = 2;

pub fn resample(input: &[f64], from_rate: u32, to_rate: u32) -> Result<Vec<f64>, String> {
    let channels = resample_channels(&[input.to_vec()], from_rate, to_rate)?;
    Ok(channels.into_iter().next().unwrap_or_default())
}

/// Resample every channel through one shared clock so stereo phase and timing
/// cannot drift. The FFT resampler includes a band-limiting filter, unlike the
/// linear interpolation previously used here.
pub fn resample_channels(
    input: &[Vec<f64>],
    from_rate: u32,
    to_rate: u32,
) -> Result<Vec<Vec<f64>>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    if from_rate == 0 || to_rate == 0 {
        return Err("sample rates must be greater than zero".into());
    }
    let frames = input[0].len();
    if input.iter().any(|channel| channel.len() != frames) {
        return Err("all channels must contain the same number of frames".into());
    }
    if frames == 0 || from_rate == to_rate {
        return Ok(input.to_vec());
    }

    let expected =
        ((frames as u128 * to_rate as u128 + from_rate as u128 / 2) / from_rate as u128) as usize;
    let mut converter = FftFixedIn::<f64>::new(
        from_rate as usize,
        to_rate as usize,
        CHUNK_FRAMES,
        SUB_CHUNKS,
        input.len(),
    )
    .map_err(|error| format!("failed to create sample-rate converter: {error}"))?;
    let delay = converter.output_delay();
    let mut output = vec![Vec::with_capacity(expected + delay + CHUNK_FRAMES); input.len()];
    let mut position = 0;

    while frames - position >= converter.input_frames_next() {
        let count = converter.input_frames_next();
        let chunk: Vec<&[f64]> = input
            .iter()
            .map(|channel| &channel[position..position + count])
            .collect();
        let converted = converter
            .process(&chunk, None)
            .map_err(|error| format!("sample-rate conversion failed: {error}"))?;
        append(&mut output, &converted);
        position += count;
    }
    if position < frames {
        let tail: Vec<&[f64]> = input.iter().map(|channel| &channel[position..]).collect();
        let converted = converter
            .process_partial(Some(&tail), None)
            .map_err(|error| format!("sample-rate conversion failed: {error}"))?;
        append(&mut output, &converted);
    }
    while output.first().map_or(0, Vec::len) < delay + expected {
        let converted = converter
            .process_partial::<&[f64]>(None, None)
            .map_err(|error| format!("sample-rate conversion flush failed: {error}"))?;
        append(&mut output, &converted);
    }

    for channel in &mut output {
        channel.drain(..delay.min(channel.len()));
        channel.truncate(expected);
        channel.resize(expected, 0.0);
    }
    Ok(output)
}

fn append(output: &mut [Vec<f64>], chunk: &[Vec<f64>]) {
    for (output, chunk) in output.iter_mut().zip(chunk) {
        output.extend_from_slice(chunk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn same_rate_is_an_exact_identity() {
        let input = vec![0.25, -0.5, 1.0];
        assert_eq!(resample(&input, 48_000, 48_000).unwrap(), input);
    }

    #[test]
    fn preserves_requested_duration() {
        let input = vec![0.0; 44_100];
        assert_eq!(resample(&input, 44_100, 16_000).unwrap().len(), 16_000);
        assert_eq!(resample(&input, 44_100, 48_000).unwrap().len(), 48_000);
    }

    #[test]
    fn downsampling_rejects_content_above_nyquist() {
        let tone = |frequency: f64| {
            (0..48_000)
                .map(|i| (TAU * frequency * i as f64 / 48_000.0).sin())
                .collect::<Vec<_>>()
        };
        let passband = resample(&tone(1_000.0), 48_000, 16_000).unwrap();
        let stopband = resample(&tone(12_000.0), 48_000, 16_000).unwrap();
        let rms = |samples: &[f64]| {
            (samples.iter().map(|x| x * x).sum::<f64>() / samples.len() as f64).sqrt()
        };
        assert!(rms(&stopband) < rms(&passband) * 0.01);
    }

    #[test]
    fn linked_channels_remain_sample_identical() {
        let channel: Vec<f64> = (0..4_410)
            .map(|i| (TAU * 997.0 * i as f64 / 44_100.0).sin())
            .collect();
        let output = resample_channels(&[channel.clone(), channel], 44_100, 48_000).unwrap();
        assert_eq!(output[0], output[1]);
    }
}
