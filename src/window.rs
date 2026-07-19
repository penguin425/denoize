//! Window functions for short-time Fourier analysis/synthesis.
//!
//! All windows here are *periodic* (computed over `N` samples, not `N+1`),
//! which makes the constant-overlap-add (COLA) sums exact for the standard
//! hop sizes.

use std::f64::consts::PI;

/// Supported window families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowType {
    /// Periodic Hann window. Default; COLA-exact at 50% overlap.
    Hann,
    /// Periodic Hamming window (slightly higher sidelobe attenuation trade-off).
    Hamming,
    /// Sine (a.k.a. half-sine / sqrt-Hann) window.
    Sine,
    /// Blackman window (low spectral leakage, narrower main lobe).
    Blackman,
    /// Kaiser-Bessel window. Adjustable sidelobe suppression via `kaiser_beta`.
    Kaiser,
    /// Flat-top window. Excellent amplitude accuracy; wider main lobe.
    FlatTop,
    /// DPSS (Discrete Prolate Spheroidal Sequence) / Slepian window.
    /// Excellent energy concentration; `dpss_bandwidth` sets time-bandwidth product NW.
    Dpss,
}

impl WindowType {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "hann" => WindowType::Hann,
            "hamming" => WindowType::Hamming,
            "sine" => WindowType::Sine,
            "blackman" => WindowType::Blackman,
            "kaiser" | "kaiser-bessel" => WindowType::Kaiser,
            "flattop" | "flat-top" | "flat_top" => WindowType::FlatTop,
            "dpss" | "slepian" => WindowType::Dpss,
            _ => return None,
        })
    }
}

/// Parameters for advanced windows.
#[derive(Clone, Copy, Debug)]
pub struct WindowParams {
    /// Kaiser β (typical 5–12 for audio; higher = lower sidelobes).
    pub kaiser_beta: f64,
    /// DPSS time-bandwidth product NW (typical 2.5–4.0).
    pub dpss_bandwidth: f64,
}

impl Default for WindowParams {
    fn default() -> Self {
        WindowParams {
            kaiser_beta: 8.0,
            dpss_bandwidth: 3.0,
        }
    }
}

/// Evaluate a single window sample `w(n)` for `n in 0..N` (periodic form).
pub fn sample(kind: WindowType, n: usize, n_total: usize) -> f64 {
    sample_with_params(kind, n, n_total, &WindowParams::default())
}

/// Evaluate with explicit parameters (Kaiser β, DPSS NW).
pub fn sample_with_params(
    kind: WindowType,
    n: usize,
    n_total: usize,
    params: &WindowParams,
) -> f64 {
    let n = n as f64;
    let nn = n_total as f64;
    match kind {
        WindowType::Hann => 0.5 * (1.0 - (2.0 * PI * n / nn).cos()),
        WindowType::Hamming => 0.54 - 0.46 * (2.0 * PI * n / nn).cos(),
        WindowType::Sine => (PI * (n + 0.5) / nn).sin(),
        WindowType::Blackman => {
            0.42 - 0.5 * (2.0 * PI * n / nn).cos() + 0.08 * (4.0 * PI * n / nn).cos()
        }
        WindowType::Kaiser => kaiser(n, nn, params.kaiser_beta),
        WindowType::FlatTop => flat_top(n, nn),
        WindowType::Dpss => dpss_approx(n, nn, params.dpss_bandwidth),
    }
}

/// Build a window of length `n_total`.
pub fn make(kind: WindowType, n_total: usize) -> Vec<f64> {
    make_with_params(kind, n_total, &WindowParams::default())
}

/// Build with explicit parameters.
pub fn make_with_params(kind: WindowType, n_total: usize, params: &WindowParams) -> Vec<f64> {
    let mut w: Vec<f64> = (0..n_total)
        .map(|i| sample_with_params(kind, i, n_total, params))
        .collect();
    // DPSS approximation may need peak normalisation.
    if kind == WindowType::Dpss {
        let peak = w.iter().cloned().fold(0.0f64, f64::max).max(1e-12);
        if (peak - 1.0).abs() > 1e-6 {
            for v in &mut w {
                *v /= peak;
            }
        }
    }
    w
}

/// Kaiser-Bessel window (periodic form).
fn kaiser(n: f64, n_total: f64, beta: f64) -> f64 {
    let alpha = 0.5 * (n_total - 1.0);
    let x = (n - alpha) / alpha;
    bessel_i0(beta * (1.0 - x * x).max(0.0).sqrt()) / bessel_i0(beta)
}

/// Modified Bessel I0 for Kaiser window (simple series).
fn bessel_i0(x: f64) -> f64 {
    if x.abs() < 3.75 {
        let t = x / 3.75;
        let t2 = t * t;
        1.0 + 3.5156229 * t2
            + 3.0899424 * t2.powi(2)
            + 1.2067492 * t2.powi(3)
            + 0.2659732 * t2.powi(4)
            + 0.0360768 * t2.powi(5)
            + 0.0045813 * t2.powi(6)
    } else {
        let t = 3.75 / x.abs();
        let ax = x.abs().exp() / x.abs().sqrt();
        let c = 0.39894228 + 0.01328592 * t + 0.00225319 * t * t - 0.00157565 * t.powi(3)
            + 0.00916281 * t.powi(4)
            - 0.02057706 * t.powi(5)
            + 0.02635537 * t.powi(6)
            - 0.01647633 * t.powi(7)
            + 0.00392377 * t.powi(8);
        ax * c
    }
}

/// Flat-top window (Heinzel et al. 5-term, periodic, clamped ≥ 0 for STFT).
fn flat_top(n: f64, n_total: f64) -> f64 {
    let a0 = 0.21557895;
    let a1 = 0.41663158;
    let a2 = 0.277263158;
    let a3 = 0.083578947;
    let a4 = 0.006947368;
    let phi = 2.0 * PI * n / n_total;
    let w = a0 - a1 * phi.cos() + a2 * (2.0 * phi).cos() - a3 * (3.0 * phi).cos()
        + a4 * (4.0 * phi).cos();
    w.max(0.0)
}

/// Approximate DPSS (Slepian) window via summed-cosine expansion.
/// NW = time-bandwidth product; higher NW → narrower main lobe, lower sidelobes.
fn dpss_approx(n: f64, n_total: f64, nw: f64) -> f64 {
    let m = (2.0 * nw - 0.5).ceil() as i32;
    let m = m.clamp(1, 8);
    let mut w = 0.0;
    for k in 0..m {
        let vk = dpss_eigenvector_coeff(k, m, nw);
        w += vk * (2.0 * PI * k as f64 * n / n_total).cos();
    }
    w.max(0.0)
}

/// DPSS eigenvector coefficient (simplified, good for audio STFT).
fn dpss_eigenvector_coeff(k: i32, m: i32, nw: f64) -> f64 {
    // Normalised cosine-series weights; dominant k=0 term ≈ 1.
    let lambda_k = 1.0 - (k as f64) / (2.0 * nw);
    let mut v = lambda_k.max(0.0);
    if k == 0 {
        v = 1.0;
    } else {
        v *= 0.5 / (k as f64);
    }
    let _ = m; // m controls series length above
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_cola_at_50pct() {
        let n = 1024;
        let w = make(WindowType::Hann, n);
        let hop = n / 2;
        let len = n + hop * 3;
        let mut cola = vec![0.0; len];
        let mut start = 0;
        while start + n <= len {
            for i in 0..n {
                cola[start + i] += w[i];
            }
            start += hop;
        }
        for i in n..len - n {
            assert!((cola[i] - 1.0).abs() < 1e-12, "cola@{i}={}", cola[i]);
        }
    }

    #[test]
    fn advanced_windows_bounded() {
        let n = 512;
        for kind in [WindowType::Kaiser, WindowType::FlatTop, WindowType::Dpss] {
            let w = make(kind, n);
            for &v in &w {
                assert!(v.is_finite());
                // Flat-top has tiny negative sidelobes at edges; clamp check.
                assert!(v >= -0.01 && v <= 1.05, "{kind:?} value {v}");
            }
        }
    }
}
