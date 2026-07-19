//! Spectral gain estimators.
//!
//! Each estimator maps a per-bin a-priori SNR `xi` and a-posteriori SNR `gamma`
//! to a real-valued gain `G in [0, 1]` applied to the noisy STFT magnitude
//! (phase is preserved). The estimators implemented here are, in increasing
//! sophistication:
//!
//! * **Spectral subtraction** (Berouti et al.) — a classic baseline.
//! * **Wiener filter** — `G = xi / (1 + xi)`.
//! * **MMSE-STSA** (Ephraim & Malah, 1984) — minimum mean-square error of the
//!   *short-time spectral amplitude*.
//! * **LogMMSE** (Ephraim & Malah, 1985) — minimizes log-spectral distortion;
//!   subjectively the best of the "pure" classical estimators.
//! * **OMLSA** (Cohen, 2001, simplified) — LogMMSE gain weighted by the
//!   speech-presence probability, which softly pulls gains toward a noise
//!   floor when speech is unlikely. This is the default and recommended
//!   algorithm.

use crate::bessel::{bessel_i0_scaled, bessel_i1_scaled, exp_int_e1};
use std::f64::consts::PI;

/// Available gain-estimation algorithms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    SpectralSubtraction,
    /// Nonlinear (power-law) spectral subtraction.
    SpecSubNonlinear,
    /// Geometric-mean spectral subtraction.
    SpecSubGeometric,
    Wiener,
    MmseStsa,
    LogMmse,
    Omlsa,
}

impl Algorithm {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "specsub" | "spectral-subtraction" | "spectralsub" => Algorithm::SpectralSubtraction,
            "specsub-nl" | "specsub-nonlinear" | "nonlinear-specsub" => Algorithm::SpecSubNonlinear,
            "specsub-geo" | "specsub-geometric" | "geometric-specsub" => {
                Algorithm::SpecSubGeometric
            }
            "wiener" => Algorithm::Wiener,
            "mmse" | "mmse-stsa" | "stsa" => Algorithm::MmseStsa,
            "logmmse" | "log-mmse" => Algorithm::LogMmse,
            "omlsa" => Algorithm::Omlsa,
            _ => return None,
        })
    }
}

/// Spectral-subtraction oversubtraction law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpecSubLaw {
    /// Classic magnitude subtraction: `G = 1 - α/√γ`.
    Linear,
    /// Power-law: `G = 1 - α/γ^p` (p=0.5 → linear).
    PowerLaw(f64),
    /// Geometric: operate on log-magnitude SNR.
    Geometric,
}

/// Parameters shared by all estimators for a given frame.
#[derive(Clone, Copy, Debug)]
pub struct GainParams {
    /// A-priori SNR floor `xi_min` (e.g. -25 dB).
    pub xi_min: f64,
    /// Minimum gain floor `G_min` (suppresses musical noise).
    pub g_min: f64,
    /// Spectral-subtraction oversubtraction factor `alpha_os`.
    pub alpha_os: f64,
    /// Spectral-subtraction noise-floor factor `beta`.
    pub beta_floor: f64,
}

const GAMMA_EPS: f64 = 1e-12;
const NU_EPS: f64 = 1e-12;

/// Wiener filter gain `G = xi / (1 + xi)`.
pub fn wiener(xi: f64, _gamma: f64, _p: GainParams) -> f64 {
    let xi = xi.max(0.0);
    xi / (1.0 + xi)
}

/// Spectral-subtraction gain, expressed in terms of `gamma = |Y|^2 / lambda_d`:
/// `G = max(1 - alpha_os / sqrt(gamma), sqrt(beta / gamma))`.
pub fn spectral_subtraction(_xi: f64, gamma: f64, p: GainParams) -> f64 {
    spectral_subtraction_with_law(_xi, gamma, p, SpecSubLaw::Linear)
}

/// Spectral subtraction with selectable nonlinear law.
pub fn spectral_subtraction_with_law(_xi: f64, gamma: f64, p: GainParams, law: SpecSubLaw) -> f64 {
    let gamma = gamma.max(GAMMA_EPS);
    let g1 = match law {
        SpecSubLaw::Linear => 1.0 - p.alpha_os / gamma.sqrt(),
        SpecSubLaw::PowerLaw(exp) => {
            let exp = exp.clamp(0.25, 1.0);
            1.0 - p.alpha_os / gamma.powf(exp)
        }
        SpecSubLaw::Geometric => {
            // Geometric mean in log-domain: subtract noise in log-power.
            let log_gamma = gamma.ln();
            let log_sub = log_gamma - 2.0 * p.alpha_os.ln();
            log_sub.exp().sqrt().min(1.0)
        }
    };
    let g2 = (p.beta_floor / gamma).sqrt();
    g1.max(g2).clamp(0.0, 1.0)
}

/// Multiband spectral subtraction: compute per-band SNR, derive band gains,
/// then interpolate back to per-bin gains.
pub fn multiband_specsub_gains(
    gamma: &[f64],
    bands: &[usize],
    n_bands: usize,
    p: GainParams,
    law: SpecSubLaw,
) -> Vec<f64> {
    let m = gamma.len();
    let mut band_gamma = vec![0.0f64; n_bands];
    let mut band_count = vec![0usize; n_bands];
    for (k, &g) in gamma.iter().enumerate() {
        let b = bands[k].min(n_bands - 1);
        band_gamma[b] += g;
        band_count[b] += 1;
    }
    let mut band_gain = vec![1.0f64; n_bands];
    for b in 0..n_bands {
        if band_count[b] > 0 {
            let avg = band_gamma[b] / band_count[b] as f64;
            band_gain[b] = spectral_subtraction_with_law(0.0, avg, p, law);
        }
    }
    let mut g = vec![1.0f64; m];
    for k in 0..m {
        let b = bands[k].min(n_bands - 1);
        g[k] = band_gain[b];
    }
    g
}

/// MMSE-STSA gain (Ephraim & Malah, 1984). Uses *exponentially-scaled* Bessel
/// functions so the formula is numerically stable for arbitrarily large a
/// priori SNR (no `0 * Inf` cancellation).
pub fn mmse_stsa(xi: f64, gamma: f64, _p: GainParams) -> f64 {
    let gamma = gamma.max(GAMMA_EPS);
    let xi = xi.max(0.0);
    let nu = (xi * gamma / (1.0 + xi)).max(NU_EPS);
    let half = 0.5 * nu;
    // g = 0.5*sqrt(pi) * sqrt(nu)/gamma * e^{-half} * [(1+nu) I0(half) + nu I1(half)]
    //   = 0.5*sqrt(pi) * sqrt(nu)/gamma * [(1+nu) I0(half)e^{-half} + nu I1(half)e^{-half}]
    let term = (1.0 + nu) * bessel_i0_scaled(half) + nu * bessel_i1_scaled(half);
    let g = 0.5 * PI.sqrt() * (nu.sqrt() / gamma) * term;
    g.clamp(0.0, 1.0)
}

/// LogMMSE gain (Ephraim & Malah, 1985): `G = xi/(1+xi) * exp(0.5 * E1(nu))`,
/// with `nu = xi*gamma/(1+xi)`. Returns the *raw* gain in `[0, 1]` (no floor).
pub fn logmmse_raw(xi: f64, gamma: f64) -> f64 {
    let gamma = gamma.max(GAMMA_EPS);
    let xi = xi.max(0.0);
    let nu = (xi * gamma / (1.0 + xi)).max(NU_EPS);
    let g = (xi / (1.0 + xi)) * (0.5 * exp_int_e1(nu)).exp();
    g.clamp(0.0, 1.0)
}

pub fn logmmse(xi: f64, gamma: f64, _p: GainParams) -> f64 {
    logmmse_raw(xi, gamma)
}

/// Simplified OMLSA gain: the LogMMSE gain raised to the speech-presence
/// probability `p`, blended with the noise floor `G_min`:
/// `G = G_log^p * G_min^(1-p)`.
pub fn omlsa(xi: f64, gamma: f64, p: GainParams, spp: f64) -> f64 {
    let g_log = logmmse_raw(xi, gamma).max(1e-6);
    let spp = spp.clamp(0.0, 1.0);
    let g = g_log.powf(spp) * p.g_min.powf(1.0 - spp);
    g.clamp(p.g_min, 1.0)
}

/// Dispatch to the selected algorithm. `spp` is the speech-presence
/// probability (only used by [`Algorithm::Omlsa`]).
pub fn compute_gain(algo: Algorithm, xi: f64, gamma: f64, spp: f64, p: GainParams) -> f64 {
    match algo {
        Algorithm::SpectralSubtraction => spectral_subtraction(xi, gamma, p),
        Algorithm::SpecSubNonlinear => {
            spectral_subtraction_with_law(xi, gamma, p, SpecSubLaw::PowerLaw(0.75))
        }
        Algorithm::SpecSubGeometric => {
            spectral_subtraction_with_law(xi, gamma, p, SpecSubLaw::Geometric)
        }
        Algorithm::Wiener => wiener(xi, gamma, p),
        Algorithm::MmseStsa => mmse_stsa(xi, gamma, p),
        Algorithm::LogMmse => logmmse(xi, gamma, p),
        Algorithm::Omlsa => omlsa(xi, gamma, p, spp),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> GainParams {
        GainParams {
            xi_min: 10f64.powf(-25.0 / 10.0),
            g_min: 10f64.powf(-25.0 / 10.0),
            alpha_os: 2.0,
            beta_floor: 0.01,
        }
    }

    #[test]
    fn wiener_bounds() {
        let p = params();
        assert!((wiener(0.0, 1.0, p)).abs() < 1e-12);
        let g = wiener(10.0, 1.0, p);
        assert!(g > 0.9 && g < 1.0);
    }

    #[test]
    fn high_snr_passes_signal() {
        let p = params();
        // +20 dB a-priori SNR, +20 dB a-posteriori SNR -> gain near 1.
        let xi = 100.0;
        let gamma = 100.0;
        for algo in [
            Algorithm::Wiener,
            Algorithm::MmseStsa,
            Algorithm::LogMmse,
            Algorithm::Omlsa,
        ] {
            let g = compute_gain(algo, xi, gamma, 1.0, p);
            assert!(g > 0.9, "{algo:?} high-snr gain {g}");
        }
    }

    #[test]
    fn low_snr_attenuates() {
        let p = params();
        // Very low SNR -> gain should be small (and never below G_min).
        let xi = 10f64.powf(-20.0 / 10.0);
        let gamma = 1.0;
        for algo in [
            Algorithm::Wiener,
            Algorithm::MmseStsa,
            Algorithm::LogMmse,
            Algorithm::Omlsa,
        ] {
            let g = compute_gain(algo, xi, gamma, 0.0, p);
            assert!(g < 0.3, "{algo:?} low-snr gain {g}");
            assert!(g >= 0.0);
        }
    }

    #[test]
    fn no_nan_at_extreme_snr() {
        // Extreme a-priori / a-posteriori SNR must never yield NaN/Inf.
        let p = params();
        for &(xi, gamma) in &[
            (1e6, 1e6),
            (1e9, 1e9),
            (1e3, 1e6),
            (1e-12, 1e-12),
            (1e6, 1e-3),
        ] {
            for algo in [
                Algorithm::Wiener,
                Algorithm::MmseStsa,
                Algorithm::LogMmse,
                Algorithm::Omlsa,
                Algorithm::SpectralSubtraction,
            ] {
                let g = compute_gain(algo, xi, gamma, 1.0, p);
                assert!(
                    g.is_finite(),
                    "{algo:?} not finite at xi={xi} gamma={gamma}: {g}"
                );
                assert!((0.0..=1.0).contains(&g), "{algo:?} out of range: {g}");
            }
        }
    }
}
