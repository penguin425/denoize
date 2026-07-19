//! PCM conversion helpers for lossy encoders (MP3 / M4A).

use crate::audio::Audio;

/// Stereo/mono layout for lossy encoders (max 2 channels).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeChannels {
    pub count: u8,
    pub is_stereo: bool,
}

/// Reduce channel count to 1 or 2 for MP3/AAC encoders.
pub fn lossy_channel_layout(audio: &Audio) -> Result<EncodeChannels, String> {
    let n = audio.channels();
    match n {
        0 => Err("no audio channels".into()),
        1 => Ok(EncodeChannels {
            count: 1,
            is_stereo: false,
        }),
        2 => Ok(EncodeChannels {
            count: 2,
            is_stereo: true,
        }),
        _ => {
            eprintln!("denoize: warning: {n} channels — downmixing to stereo for lossy output");
            Ok(EncodeChannels {
                count: 2,
                is_stereo: true,
            })
        }
    }
}

/// Planar `f64` [-1, 1] → interleaved `i16` for shine / fdk-aac.
pub fn planar_f64_to_interleaved_i16(audio: &Audio, layout: EncodeChannels) -> Vec<i16> {
    let frames = audio.frames();
    let n_in = audio.channels();
    let out_ch = layout.count as usize;
    let mut out = Vec::with_capacity(frames * out_ch);

    for f in 0..frames {
        if layout.is_stereo {
            let l = sample_at(audio, f, 0, n_in);
            let r = sample_at(audio, f, 1, n_in);
            out.push(f64_to_i16(l));
            out.push(f64_to_i16(r));
        } else {
            let m = sample_at(audio, f, 0, n_in);
            out.push(f64_to_i16(m));
        }
    }
    out
}

#[inline]
fn sample_at(audio: &Audio, frame: usize, ch: usize, n_in: usize) -> f64 {
    if n_in == 1 {
        return audio.channels[0].get(frame).copied().unwrap_or(0.0);
    }
    if n_in == 2 {
        return audio.channels[ch].get(frame).copied().unwrap_or(0.0);
    }
    // >2: simple stereo downmix — even → L, odd → R
    let mut l = 0.0;
    let mut r = 0.0;
    let mut lc = 0usize;
    let mut rc = 0usize;
    for (i, ch_data) in audio.channels.iter().enumerate() {
        let v = ch_data.get(frame).copied().unwrap_or(0.0);
        if i % 2 == 0 {
            l += v;
            lc += 1;
        } else {
            r += v;
            rc += 1;
        }
    }
    let v = if ch == 0 {
        if lc > 0 {
            l / lc as f64
        } else {
            0.0
        }
    } else if rc > 0 {
        r / rc as f64
    } else if lc > 0 {
        l / lc as f64
    } else {
        0.0
    };
    v.clamp(-1.0, 1.0)
}

#[inline]
fn f64_to_i16(v: f64) -> i16 {
    let q = (v.clamp(-1.0, 1.0) * 32767.0).round() as i32;
    q.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::SampleFormat;

    fn mono_audio(vals: &[f64]) -> Audio {
        Audio {
            sample_rate: 44100,
            channels: vec![vals.to_vec()],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        }
    }

    #[test]
    fn mono_to_i16() {
        let a = mono_audio(&[0.0, 0.5, -0.5]);
        let pcm = planar_f64_to_interleaved_i16(
            &a,
            EncodeChannels {
                count: 1,
                is_stereo: false,
            },
        );
        assert_eq!(pcm.len(), 3);
        assert_eq!(pcm[1], 16384);
    }

    #[test]
    fn quad_downmix_stereo() {
        let a = Audio {
            sample_rate: 44100,
            channels: vec![vec![1.0], vec![0.0], vec![0.0], vec![1.0]],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let pcm = planar_f64_to_interleaved_i16(
            &a,
            EncodeChannels {
                count: 2,
                is_stereo: true,
            },
        );
        assert_eq!(pcm.len(), 2);
        assert!(pcm[0] > 0);
        assert!(pcm[1] > 0);
    }
}
