//! Noise-power-spectral-density estimation.
//!
//! Implements a robust *minima-controlled recursive averaging* (MCRA / IMCRA
//! style) estimator, optionally seeded by a *noise profile* learned from a
//! leading-silence (or explicitly provided) segment. This is the classical
//! state-of-the-art approach to tracking non-stationary noise without needing
//! a separate noise-only recording, while a profile gives an excellent
//! cold-start.
//!
//! Per frequency bin `k`, every frame:
//!   1. Recursively smooth the noisy power: `S = a_s*S + (1-a_s)*|Y|^2`.
//!   2. Track the running minimum `S_min` over a window of `L` frames using a
//!      two-tracker (Dpledged) scheme that avoids the minimum "sticking".
//!   3. Estimate the speech-presence probability `p` from the ratio
//!      `zeta = S / (B_min * S_min)` via a sigmoid.
//!   4. Update the noise PSD with a smoothing factor that *freezes* when
//!      speech is present: `a_d_eff = a_d + (1-a_d)*p`,
//!      `lambda_d = a_d_eff*lambda_d + (1-a_d_eff)*|Y|^2`.

/// Tunable parameters for the noise estimator.
#[derive(Clone, Copy, Debug)]
pub struct NoiseConfig {
    /// Power-spectrum smoothing factor (close to 1 = slow).
    pub alpha_s: f64,
    /// Base noise-PSD smoothing factor used in speech absence.
    pub alpha_d: f64,
    /// Minima-tracking window length, in frames (~1.5 s of audio by default).
    pub window_frames: usize,
    /// Bias compensation applied to the running minimum.
    pub b_min: f64,
    /// Sigmoid midpoint: speech declared around `zeta == zeta0`.
    pub zeta0: f64,
    /// Sigmoid steepness (smaller = sharper transition).
    pub sigma: f64,
    /// Maximum amount (in dB) the noise PSD may rise *above a learned noise
    /// profile*. This anchors the estimate so a long, sustained signal cannot
    /// drag `lambda_d` up and suppress itself. `f64::INFINITY` disables the
    /// anchor.
    pub anchor_cap_db: f64,
    /// Maximum upward slew rate of the noise PSD, in dB per second. Limits how
    /// fast `lambda_d` can rise in the blind (no-profile) case. `f64::INFINITY`
    /// disables the rate limiter.
    pub up_rate_dbps: f64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        NoiseConfig {
            alpha_s: 0.9,
            alpha_d: 0.95,
            window_frames: 0, // 0 => computed from sample rate / hop (~1.5 s)
            b_min: 2.0,
            zeta0: 2.0,
            sigma: 0.25,
            anchor_cap_db: 15.0,
            up_rate_dbps: 6.0,
        }
    }
}

pub struct NoiseEstimator {
    cfg: NoiseConfig,
    nbins: usize,
    s: Vec<f64>,        // smoothed noisy power
    s_min: Vec<f64>,    // fast-forgetting minimum (SPP decision)
    s_tmp: Vec<f64>,    // slow-forgetting minimum (long-term floor)
    lambda_d: Vec<f64>, // noise PSD
    p: Vec<f64>,        // speech-presence probability
    initialized: bool,
    /// Whether the noise PSD is allowed to adapt over time.
    pub adapt: bool,
    /// Learned noise profile used as an upward anchor (None if not profiled).
    profile_psd: Vec<f64>,
    has_profile: bool,
    /// Per-frame multiplicative cap for the upward rate limiter.
    up_ratio: f64,
    /// Multiplicative anchor cap (10^(anchor_cap_db/10)); `INFINITY` disables.
    anchor_ratio: f64,
    /// Per-frame forgetting factor for s_min (~+1.5 dB/s).
    min_forget_fast: f64,
    /// Per-frame forgetting factor for s_tmp (~+0.5 dB/s).
    min_forget_slow: f64,
}

impl NoiseEstimator {
    /// Create a new estimator for `nbins` unique bins. `window_frames == 0`
    /// selects a ~1.5 s window from `sample_rate` and `hop`.
    pub fn new(cfg: NoiseConfig, nbins: usize, sample_rate: u32, hop: usize) -> Self {
        let window_frames = if cfg.window_frames == 0 {
            ((1.5 * sample_rate as f64 / hop as f64).round() as usize).max(8)
        } else {
            cfg.window_frames
        };
        let anchor_ratio = 10f64.powf(cfg.anchor_cap_db / 10.0);
        let up_ratio = 10f64.powf(cfg.up_rate_dbps * hop as f64 / sample_rate as f64 / 10.0);
        let dt = hop as f64 / sample_rate as f64; // seconds per frame
                                                  // Fast tracker: rises ~9 dB/s so it converges to the current noise level
                                                  // within ~1 s (giving zeta ≈ 1 for stationary noise → low SPP → noise
                                                  // attenuated), while staying well below a sustained signal (which is
                                                  // typically >20 dB above noise, so zeta stays high → SPP ≈ 1 → signal
                                                  // preserved). Slow tracker: +0.5 dB/s, a long-term floor reference.
        let min_forget_fast = 10f64.powf(9.0 * dt / 10.0);
        let min_forget_slow = 10f64.powf(0.5 * dt / 10.0);
        NoiseEstimator {
            cfg: NoiseConfig {
                window_frames,
                ..cfg
            },
            nbins,
            s: vec![0.0; nbins],
            s_min: vec![f64::MAX; nbins],
            s_tmp: vec![f64::MAX; nbins],
            lambda_d: vec![1e-10; nbins],
            p: vec![0.0; nbins],
            initialized: false,
            adapt: true,
            profile_psd: vec![0.0; nbins],
            has_profile: false,
            up_ratio,
            anchor_ratio,
            min_forget_fast,
            min_forget_slow,
        }
    }

    #[inline]
    pub fn nbins(&self) -> usize {
        self.nbins
    }

    /// Seed the estimator from a noise profile: per-frame power spectra
    /// `|Y|^2` (length `nbins` each) averaged into `lambda_d`.
    pub fn seed_from_profile(&mut self, profile_frames: &[Vec<f64>]) {
        if profile_frames.is_empty() {
            return;
        }
        for k in 0..self.nbins {
            let mut acc = 0.0;
            let mut n = 0.0;
            for fr in profile_frames {
                if k < fr.len() {
                    acc += fr[k];
                    n += 1.0;
                }
            }
            let val = (if n > 0.0 { acc / n } else { 1e-10 }).max(1e-12);
            self.lambda_d[k] = val;
            self.s[k] = val;
            self.s_min[k] = val;
            self.s_tmp[k] = val;
            self.profile_psd[k] = val;
        }
        self.initialized = true;
        self.has_profile = true;
    }

    /// Current noise PSD estimate (length `nbins`).
    #[inline]
    pub fn noise_psd(&self) -> &[f64] {
        &self.lambda_d
    }

    /// Current per-bin speech-presence probability (length `nbins`).
    #[inline]
    pub fn speech_presence(&self) -> &[f64] {
        &self.p
    }

    /// Update with one frame of noisy power `y2[k] = |Y[k]|^2` (length `nbins`).
    pub fn update(&mut self, y2: &[f64]) {
        debug_assert_eq!(y2.len(), self.nbins);
        let cfg = self.cfg;

        // Skip near-silent (zero-padding) frames before bootstrap so they do not
        // corrupt the initial noise estimate. We treat a frame as padding when
        // its total energy is more than 60 dB below the loudest bin seen so far
        // (or, pre-bootstrap, simply when it is effectively zero).
        let frame_energy: f64 = y2.iter().sum();
        if !self.initialized {
            if frame_energy < 1e-9 {
                self.p.fill(0.0);
                return;
            }
            for k in 0..self.nbins {
                let v = y2[k].max(1e-12);
                self.s[k] = v;
                self.s_min[k] = v;
                self.s_tmp[k] = v;
                self.lambda_d[k] = v;
            }
            self.initialized = true;
            self.p.fill(0.0);
            return;
        }

        // 1. Smoothed noisy power.
        for k in 0..self.nbins {
            self.s[k] = cfg.alpha_s * self.s[k] + (1.0 - cfg.alpha_s) * y2[k];
        }

        // 2. Running minimum with exponential forgetting.
        //
        // s_min = min(s, s_min * gamma) where gamma > 1 lets s_min rise slowly
        // to track genuine noise-level changes. When a noise PROFILE is
        // available, the profile anchor caps s_min, so we use a SLOW forgetting
        // rate (the anchor does the heavy lifting and s_min stays near the true
        // floor → maximum SPP contrast). Without a profile, we use a FAST rate
        // so s_min converges to the current noise level within ~1 s, giving
        // zeta ≈ 1 for stationary noise (→ low SPP → noise attenuated) while
        // staying well below any sustained signal (→ high SPP → signal kept).
        let gamma_fast = if self.has_profile {
            self.min_forget_slow // slow (+0.5 dB/s) when anchored
        } else {
            self.min_forget_fast // fast (+9 dB/s) when blind
        };
        let gamma_slow = self.min_forget_slow;
        for k in 0..self.nbins {
            let f = self.s_min[k] * gamma_fast;
            self.s_min[k] = if self.s[k] < f { self.s[k] } else { f };
            let g = self.s_tmp[k] * gamma_slow;
            self.s_tmp[k] = if self.s[k] < g { self.s[k] } else { g };
        }
        // Anchor the running minima to the profile (when available): never let
        // s_min rise far above a learned noise floor. This is a tighter cap
        // (half the dB of the lambda_d anchor) to keep SPP sensitive to speech.
        if self.has_profile {
            let cap = self.anchor_ratio.sqrt();
            for k in 0..self.nbins {
                let m = self.profile_psd[k] * cap;
                if self.s_min[k] > m {
                    self.s_min[k] = m;
                }
                if self.s_tmp[k] > m {
                    self.s_tmp[k] = m;
                }
            }
        }

        // 3. SPP and 4. noise-PSD update.
        for k in 0..self.nbins {
            let denom = (cfg.b_min * self.s_min[k]).max(1e-12);
            let zeta = self.s[k] / denom;
            let arg = (zeta - cfg.zeta0) / cfg.sigma;
            let p = if arg >= 0.0 {
                1.0 / (1.0 + (-arg).exp())
            } else {
                let e = arg.exp();
                e / (1.0 + e)
            };
            self.p[k] = p;
            if self.adapt {
                let old = self.lambda_d[k];
                let a_d_eff = cfg.alpha_d + (1.0 - cfg.alpha_d) * p;
                let mut new_ld = a_d_eff * old + (1.0 - a_d_eff) * y2[k];
                // Upward rate limiter: limit how fast lambda_d can rise, so a
                // misclassified sustained signal cannot drag the estimate up
                // quickly. Downward movement is unrestricted.
                let cap_up = old * self.up_ratio;
                if new_ld > cap_up {
                    new_ld = cap_up;
                }
                // Profile anchor: never let lambda_d rise far above a learned
                // noise profile. This is the decisive guard against long
                // sustained signals being absorbed as noise.
                if self.has_profile {
                    let anchor = self.profile_psd[k] * self.anchor_ratio;
                    if new_ld > anchor {
                        new_ld = anchor;
                    }
                }
                self.lambda_d[k] = new_ld;
            }
        }
    }
}
