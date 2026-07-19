//! Short-time Fourier transform engine with perfect-reconstruction
//! overlap-add (OLA) synthesis.
//!
//! The analysis window `w_a` and synthesis window `w_s` are applied on the way
//! in and out. Perfect reconstruction of an *unmodified* spectrum is achieved
//! by normalizing each output sample by the accumulated
//! `sum_k w_a[n-kH] * w_s[n-kH]`, which is tracked in a parallel buffer. This
//! makes the OLA exact for *any* window and any overlap ratio, including the
//! 75%-overlap Hann configuration used by default (where the sum is not 1.0).

use crate::fft::{Complex, Fft};
use crate::window::{make_with_params, WindowParams, WindowType};

/// Configuration for the STFT engine.
#[derive(Clone, Copy, Debug)]
pub struct StftConfig {
    pub frame_size: usize,
    pub hop: usize,
    pub window: WindowType,
    pub window_params: WindowParams,
}

pub struct Stft {
    cfg: StftConfig,
    analysis: Vec<f64>,
    synthesis: Vec<f64>,
    fft: Fft,
}

impl Stft {
    pub fn new(cfg: StftConfig) -> Self {
        assert!(cfg.frame_size.is_power_of_two());
        assert!(cfg.hop > 0 && cfg.hop <= cfg.frame_size);
        // Identical analysis and synthesis windows: the normalization buffer
        // then holds sum_k w[n-kH]^2 which is smooth and strictly positive.
        let w = make_with_params(cfg.window, cfg.frame_size, &cfg.window_params);
        Stft {
            cfg,
            analysis: w.clone(),
            synthesis: w,
            fft: Fft::new(cfg.frame_size),
        }
    }

    #[inline]
    pub fn frame_size(&self) -> usize {
        self.cfg.frame_size
    }

    #[inline]
    pub fn hop(&self) -> usize {
        self.cfg.hop
    }

    #[inline]
    pub fn nbins(&self) -> usize {
        self.fft.nbins()
    }

    #[inline]
    pub fn fft(&self) -> &Fft {
        &self.fft
    }

    /// Window a time-domain frame (`len == frame_size`) and forward-transform
    /// it into `spec` (`len == frame_size`), with the imaginary part set up.
    pub fn analyze(&self, time: &[f64], spec: &mut [Complex]) {
        debug_assert_eq!(time.len(), self.cfg.frame_size);
        debug_assert_eq!(spec.len(), self.cfg.frame_size);
        for i in 0..self.cfg.frame_size {
            spec[i] = Complex::new(time[i] * self.analysis[i], 0.0);
        }
        self.fft.forward(spec);
    }

    /// Inverse-transform `spec`, apply the synthesis window, and overlap-add
    /// into `out` while accumulating the normalization weight into `norm`, at
    /// sample offset `start`.
    pub fn synthesize(
        &self,
        spec: &mut [Complex],
        out: &mut [f64],
        norm: &mut [f64],
        start: usize,
    ) {
        debug_assert_eq!(spec.len(), self.cfg.frame_size);
        self.fft.inverse(spec);
        let n = self.cfg.frame_size;
        // Guard against writing past the end of the output buffers.
        let end = (start + n).min(out.len());
        let lim = end - start;
        for i in 0..lim {
            let s = spec[i].re * self.synthesis[i];
            out[start + i] += s;
            norm[start + i] += self.analysis[i] * self.synthesis[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_reconstruction_hann_75pct() {
        let n = 1024;
        let hop = n / 4; // 75% overlap
        let stft = Stft::new(StftConfig {
            frame_size: n,
            hop,
            window: WindowType::Hann,
            window_params: WindowParams::default(),
        });

        // Build a test signal longer than one frame.
        let total = 8 * n;
        let signal: Vec<f64> = (0..total)
            .map(|i| (0.017 * i as f64).sin() + 0.5 * (0.003 * i as f64).cos())
            .collect();

        let mut out = vec![0.0; total];
        let mut norm = vec![0.0; total];
        let mut spec = vec![Complex::default(); n];
        let mut frame = vec![0.0; n];

        let mut start = 0;
        while start + n <= total {
            frame.copy_from_slice(&signal[start..start + n]);
            stft.analyze(&frame, &mut spec);
            // No modification -> must reconstruct exactly.
            stft.synthesize(&mut spec, &mut out, &mut norm, start);
            start += hop;
        }

        // Normalize by the OLA weight and compare in the fully-covered interior.
        let interior = n..total - n;
        let mut max_err: f64 = 0.0;
        for i in interior {
            let r = out[i] / norm[i];
            max_err = max_err.max((r - signal[i]).abs());
        }
        assert!(max_err < 1e-6, "reconstruction error too high: {max_err}");
    }
}
