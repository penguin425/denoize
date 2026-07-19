//! High-fidelity PCM utilities for the decode pipeline.
//!
//! Decoded audio is accumulated in `f64` planar buffers in `[-1, 1]` with no
//! premature quantisation — preserving decoder output precision for denoising.

/// Decoded PCM prior to wrapping in [`crate::audio::Audio`].
#[derive(Clone, Debug)]
pub struct DecodedPcm {
    pub sample_rate: u32,
    pub channels: Vec<Vec<f64>>,
}

impl DecodedPcm {
    /// Wrap as [`crate::audio::Audio`] using 32-bit float metadata (full decode precision).
    pub fn into_audio(self) -> crate::audio::Audio {
        crate::audio::Audio {
            sample_rate: self.sample_rate,
            channels: self.channels,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        }
    }

    pub fn n_channels(&self) -> usize {
        self.channels.len()
    }

    pub fn frames(&self) -> usize {
        self.channels.first().map(|c| c.len()).unwrap_or(0)
    }

    /// Ensure all channels have equal length (pad shorter with silence).
    pub fn normalize_lengths(&mut self) {
        let max = self.channels.iter().map(Vec::len).max().unwrap_or(0);
        for ch in &mut self.channels {
            if ch.len() < max {
                ch.resize(max, 0.0);
            }
        }
    }
}

/// Convert `f32` decoder samples to `f64` without clamping (preserves headroom).
#[inline]
pub fn f32_to_f64(v: f32) -> f64 {
    v as f64
}

/// Append interleaved `f32` PCM from a decoder frame into planar `f64` buffers.
#[cfg(test)]
pub fn append_interleaved_f32(
    channels: &mut [Vec<f64>],
    interleaved: &[f32],
    n_frames: usize,
    n_ch: usize,
) {
    debug_assert!(interleaved.len() >= n_frames * n_ch);
    for i in 0..n_frames {
        for ch in 0..n_ch {
            channels[ch].push(f32_to_f64(interleaved[i * n_ch + ch]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_interleaved_stereo() {
        let mut ch = vec![vec![], vec![]];
        append_interleaved_f32(&mut ch, &[0.5, -0.5, 1.0, -1.0], 2, 2);
        assert_eq!(ch[0], vec![0.5, 1.0]);
        assert_eq!(ch[1], vec![-0.5, -1.0]);
    }

    #[test]
    fn normalize_lengths_uses_longest_channel() {
        let mut pcm = DecodedPcm {
            sample_rate: 48000,
            channels: vec![vec![1.0], vec![2.0, 3.0, 4.0]],
        };
        pcm.normalize_lengths();
        assert_eq!(pcm.channels[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(pcm.channels[1].len(), 3);
    }
}
