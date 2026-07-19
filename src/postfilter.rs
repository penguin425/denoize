//! Musical-noise suppression post-filter.
//!
//! After the main gain estimation, this module detects isolated spectral
//! peaks (typical "birdies" / musical noise) and smooths them using
//! time–frequency median filtering and perceptual masking thresholds.

/// Post-filter configuration.
#[derive(Clone, Copy, Debug)]
pub struct PostFilterConfig {
    /// Enable the post-filter.
    pub enabled: bool,
    /// Strength in `[0, 1]` — higher = more aggressive artifact removal.
    pub strength: f64,
    /// Temporal smoothing of the detection mask.
    pub time_smooth: f64,
    /// Minimum local SNR (dB) below which a bin is considered artifact-prone.
    pub artifact_snr_db: f64,
}

impl Default for PostFilterConfig {
    fn default() -> Self {
        PostFilterConfig {
            enabled: true,
            strength: 0.5,
            time_smooth: 0.7,
            artifact_snr_db: 6.0,
        }
    }
}

/// Stateful post-filter (per channel).
pub struct MusicalNoisePostFilter {
    cfg: PostFilterConfig,
    prev_mask: Vec<f64>,
    nbins: usize,
}

impl MusicalNoisePostFilter {
    pub fn new(nbins: usize, cfg: PostFilterConfig) -> Self {
        MusicalNoisePostFilter {
            cfg,
            prev_mask: vec![0.0; nbins],
            nbins,
        }
    }

    pub fn reset(&mut self) {
        self.prev_mask.fill(0.0);
    }

    /// Refine gains `g[k]` using spectral peakiness detection.
    ///
    /// `y2` = noisy power, `lambda_d` = noise PSD, `g` = current gains (in/out).
    pub fn apply(&mut self, y2: &[f64], lambda_d: &[f64], g: &mut [f64]) {
        if !self.cfg.enabled {
            return;
        }
        debug_assert_eq!(y2.len(), self.nbins);
        debug_assert_eq!(lambda_d.len(), self.nbins);
        debug_assert_eq!(g.len(), self.nbins);

        let m = self.nbins;
        let strength = self.cfg.strength.clamp(0.0, 1.0);
        let snr_thresh = 10f64.powf(self.cfg.artifact_snr_db / 10.0);
        let ts = self.cfg.time_smooth.clamp(0.0, 0.95);

        for k in 1..m.saturating_sub(1) {
            let snr = y2[k] / lambda_d[k].max(1e-12);
            if snr < snr_thresh {
                continue;
            }

            // Local peakiness: bin much louder than neighbours → musical noise candidate.
            let local_mean = 0.5 * (y2[k - 1] + y2[k + 1]).max(1e-12);
            let peakiness = (y2[k] / local_mean).ln().max(0.0);

            // Isolated deep gain notch (gain valley surrounded by higher gains).
            let g_local = 0.5 * (g[k - 1] + g[k + 1]);
            let gain_dip = (g_local - g[k]).max(0.0);

            let raw_mask = (peakiness * 0.35 + gain_dip * 2.5).min(1.0);
            let mask = ts * self.prev_mask[k] + (1.0 - ts) * raw_mask;
            self.prev_mask[k] = mask;

            if mask > 0.15 {
                // Blend gain toward local neighbourhood average.
                let target = g_local;
                let blend = strength * mask * 0.6;
                g[k] = g[k] * (1.0 - blend) + target * blend;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooths_isolated_dip() {
        let mut pf = MusicalNoisePostFilter::new(5, PostFilterConfig::default());
        let y2 = vec![1.0, 1.0, 100.0, 1.0, 1.0];
        let lambda_d = vec![1.0; 5];
        let mut g = vec![0.8, 0.8, 0.1, 0.8, 0.8];
        pf.apply(&y2, &lambda_d, &mut g);
        assert!(g[2] > 0.1, "isolated dip should be raised: {}", g[2]);
    }
}
