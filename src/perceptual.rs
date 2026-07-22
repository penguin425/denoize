//! Perceptual weighting for spectral gain control.
//!
//! Maps FFT bin indices to approximate Bark-scale bands and applies
//! frequency-dependent gain shaping so that suppression is stronger where
//! the ear is less sensitive and gentler in speech-critical bands.

/// Number of Bark bands used for perceptual weighting.
pub const N_BARK_BANDS: usize = 24;

/// Convert Hz to Bark (Traunmüller 1990).
#[inline]
pub fn hz_to_bark(hz: f64) -> f64 {
    26.81 / (1.0 + 1960.0 / hz.max(1.0)) - 0.53
}

/// Convert Bark to Hz (Traunmüller inverse).
#[inline]
pub fn bark_to_hz(bark: f64) -> f64 {
    1960.0 / (26.81 / (bark + 0.53) - 1.0).max(1e-6)
}

/// Bark band edges in Hz (25 bands → 24 intervals).
pub fn bark_edges_hz() -> [f64; N_BARK_BANDS + 1] {
    let mut edges = [0.0; N_BARK_BANDS + 1];
    for (i, e) in edges.iter_mut().enumerate() {
        let bark = i as f64 * 24.0 / N_BARK_BANDS as f64;
        *e = bark_to_hz(bark);
    }
    edges[N_BARK_BANDS] = edges[N_BARK_BANDS].max(20_000.0);
    edges
}

/// Build a per-bin Bark band index table (length `nbins`).
pub fn bin_to_bark_band(nbins: usize, sample_rate: u32) -> Vec<usize> {
    let nyq = sample_rate as f64 / 2.0;
    let edges = bark_edges_hz();
    let mut table = vec![0usize; nbins];
    for (k, slot) in table.iter_mut().enumerate().take(nbins) {
        let hz = k as f64 * sample_rate as f64 / (2 * (nbins - 1)) as f64;
        let hz = hz.min(nyq);
        let mut band = 0usize;
        for b in 0..N_BARK_BANDS {
            if hz >= edges[b] && hz < edges[b + 1] {
                band = b;
                break;
            }
            if hz >= edges[b + 1] {
                band = b;
            }
        }
        *slot = band.min(N_BARK_BANDS - 1);
    }
    table
}

/// Perceptual weight for a Bark band in `[w_min, 1.0]`.
/// Lower weight → more aggressive suppression allowed in that band.
pub fn bark_weight(band: usize, strength: f64) -> f64 {
    // Speech formant region (roughly 300–3500 Hz, bands 2–12): preserve.
    // Very low and very high: can suppress more.
    let base = match band {
        0..=1 => 0.55,   // sub-200 Hz
        2..=12 => 1.0,   // speech core
        13..=17 => 0.85, // upper speech / presence
        _ => 0.65,       // air / hiss region
    };
    let w_min = 0.4 + 0.2 * (1.0 - strength);
    w_min + (1.0 - w_min) * base
}

/// Apply perceptual weighting to a gain vector: `g[k] = g_min + (g[k]-g_min)*w[k]`.
pub fn apply_perceptual_weights(g: &mut [f64], bands: &[usize], strength: f64, g_min: f64) {
    debug_assert_eq!(g.len(), bands.len());
    for (gk, &band) in g.iter_mut().zip(bands.iter()) {
        let w = bark_weight(band, strength);
        let raw = *gk;
        *gk = g_min + (raw - g_min) * w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bark_edges_monotonic() {
        let e = bark_edges_hz();
        for i in 1..e.len() {
            assert!(e[i] >= e[i - 1], "band {i}");
        }
    }

    #[test]
    fn speech_bands_get_full_weight() {
        assert!((bark_weight(5, 0.5) - 1.0).abs() < 1e-9);
    }
}
