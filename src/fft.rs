//! A self-contained, allocation-light, iterative radix-2 Cooley-Tukey FFT.
//!
//! Operates in-place on a slice of [`Complex`] values and only supports
//! power-of-two sizes — fast enough for the short-time Fourier transforms used
//! by the de-noizer (frame sizes 512..8192) while keeping the project free of
//! external numeric dependencies.

use std::f64::consts::PI;

/// A complex number stored as separate real / imaginary parts.
#[derive(Clone, Copy, Debug, Default)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }

    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Complex {
            re: self.re * s,
            im: self.im * s,
        }
    }

    /// Multiply by a *real* scalar (used heavily in the gain stage).
    #[inline]
    pub fn mul_real(self, s: f64) -> Self {
        self.scale(s)
    }
}

#[inline]
fn mul(a: Complex, b: Complex) -> Complex {
    Complex {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    }
}

/// A precomputed FFT plan for a single (power-of-two) size.
pub struct Fft {
    n: usize,
    rev: Vec<usize>,
    /// Twiddle factors `exp(-2*pi*i*k/N)` for `k = 0..N/2` (forward transform).
    twiddle: Vec<Complex>,
}

impl Fft {
    /// Create a plan for size `n`. Panics if `n` is not a power of two.
    pub fn new(n: usize) -> Self {
        assert!(
            n.is_power_of_two() && n >= 2,
            "fft size must be a power of two"
        );
        let bits = n.trailing_zeros() as usize;

        let mut rev = vec![0usize; n];
        for i in 0..n {
            let mut x = i;
            let mut r = 0usize;
            for _ in 0..bits {
                r = (r << 1) | (x & 1);
                x >>= 1;
            }
            rev[i] = r;
        }

        let half = n / 2;
        let twiddle = (0..half)
            .map(|k| {
                let theta = -2.0 * PI * k as f64 / n as f64;
                Complex::new(theta.cos(), theta.sin())
            })
            .collect();

        Fft { n, rev, twiddle }
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.n
    }

    /// Number of *unique* (non-negative) frequency bins: `N/2 + 1`.
    #[inline]
    pub fn nbins(&self) -> usize {
        self.n / 2 + 1
    }

    /// In-place forward FFT. `a` must have length `size()`.
    pub fn forward(&self, a: &mut [Complex]) {
        debug_assert_eq!(a.len(), self.n);
        let rev = &self.rev;
        for i in 0..self.n {
            let j = rev[i];
            if i < j {
                a.swap(i, j);
            }
        }

        let mut size = 2usize;
        while size <= self.n {
            let half = size / 2;
            let twstride = self.n / size;
            let mut k = 0;
            while k < self.n {
                let mut t = 0;
                for j in 0..half {
                    let w = self.twiddle[t];
                    let idx = k + j + half;
                    let x = a[idx];
                    let tw = mul(w, x);
                    let u = a[k + j];
                    a[k + j] = Complex::new(u.re + tw.re, u.im + tw.im);
                    a[k + j + half] = Complex::new(u.re - tw.re, u.im - tw.im);
                    t += twstride;
                }
                k += size;
            }
            size <<= 1;
        }
    }

    /// In-place inverse FFT (normalized by `1/N`). `a` must have length `size()`.
    pub fn inverse(&self, a: &mut [Complex]) {
        debug_assert_eq!(a.len(), self.n);
        for x in a.iter_mut() {
            x.im = -x.im;
        }
        self.forward(a);
        let inv = 1.0 / self.n as f64;
        for x in a.iter_mut() {
            x.re *= inv;
            x.im = -x.im * inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft(a: &[Complex]) -> Vec<Complex> {
        let n = a.len();
        let mut out = vec![Complex::default(); n];
        for k in 0..n {
            let mut re = 0.0;
            let mut im = 0.0;
            for t in 0..n {
                let ang = -2.0 * PI * (k as f64) * (t as f64) / n as f64;
                re += a[t].re * ang.cos() - a[t].im * ang.sin();
                im += a[t].re * ang.sin() + a[t].im * ang.cos();
            }
            out[k] = Complex::new(re, im);
        }
        out
    }

    #[test]
    fn fft_matches_dft() {
        let n = 64;
        let fft = Fft::new(n);
        let mut input = vec![Complex::default(); n];
        for i in 0..n {
            input[i] = Complex::new(
                (0.7 * (i as f64)).sin() + 0.3 * (0.13 * i as f64).cos(),
                0.0,
            );
        }
        let expected = naive_dft(&input);
        let mut got = input.clone();
        fft.forward(&mut got);
        for k in 0..n {
            assert!((got[k].re - expected[k].re).abs() < 1e-8, "re @ {k}");
            assert!((got[k].im - expected[k].im).abs() < 1e-8, "im @ {k}");
        }
    }

    #[test]
    fn fft_inverse_roundtrip() {
        let n = 256;
        let fft = Fft::new(n);
        let mut input = vec![Complex::default(); n];
        for i in 0..n {
            input[i] = Complex::new((0.05 * i as f64).sin(), (0.02 * i as f64).cos());
        }
        let original = input.clone();
        fft.forward(&mut input);
        fft.inverse(&mut input);
        for i in 0..n {
            assert!((input[i].re - original[i].re).abs() < 1e-9);
            assert!((input[i].im - original[i].im).abs() < 1e-9);
        }
    }
}
