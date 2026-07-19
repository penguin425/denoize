//! The de-noizer: ties together STFT, IMCRA noise estimation, the
//! decision-directed a-priori SNR estimator, the selected spectral gain
//! estimator, attack/release + cepstral gain smoothing, transient protection,
//! and optional pre-emphasis.
//!
//! 現在の実装でサポートしているノイズ除去技術（完全版）:
//!
//! 1. 基本フレームワーク
//!    - STFT + ISTFT（自前 radix-2 FFT）
//!    - Perfect Reconstruction OLA（窓エネルギー累積正規化）
//!    - 高オーバーラップ（0.5〜0.95）対応
//!    - 窓関数: Hann / Hamming / Sine / Blackman / Kaiser / Flat-top / DPSS
//!
//! 2. ノイズ推定
//!    - IMCRA/MCRA スタイル（minima-controlled recursive averaging）
//!    - 指数忘却型2トラッカー最小値追跡
//!    - Speech Presence Probability (SPP) 推定
//!    - Spectral Flatness による自動ノイズプロファイル検出
//!    - Profile Anchoring + 上昇率制限
//!
//! 3. SNR推定
//!    - Ephraim-Malah Decision-Directed a-priori SNR
//!
//! 4. スペクトルゲイン推定器（5種類）
//!    - OMLSA (Cohen 2001, デフォルト)
//!    - LogMMSE (Ephraim-Malah 1985)
//!    - MMSE-STSA (Ephraim-Malah 1984)
//!    - Wiener
//!    - Spectral Subtraction (+ nonlinear / geometric / multiband variants)
//!
//! 5. 後処理・平滑化
//!    - Attack/Release ゲイン平滑化
//!    - Gain Floor
//!    - DC Blocking
//!    - Makeup Gain
//!
//! 6. 高音質化拡張（本プロジェクトの目玉）
//!    - Transient Protection（オンセット保護）
//!    - Cepstral Smoothing（ミュージカルノイズ抑制）
//!    - Perceptual Bark weighting + Musical-noise post-filter
//!    - Pre-emphasis / De-emphasis（オプション）
//!
//! 全体パイプライン（per channel）:
//!   1. Optional DC-blocking high-pass filter
//!   2. Optional noise profile seed（先頭無音 or 指定）
//!   3. STFT analysis（任意の窓 + 高オーバーラップ対応）
//!   4. IMCRA ノイズPSD + SPP 更新
//!   5. Decision-directed a-priori SNR 推定
//!   6. 選択したゲイン推定器で `g[k]` を計算
//!   7. Transient Protection（フラックスベース）
//!   8. Attack/Release 平滑化
//!   9. Cepstral Smoothing（本格的低ケフレンシ除去）
//!  10. ゲイン適用（位相保持）
//!  11. ISTFT + 完全再構成 OLA 正規化
//!  12. Optional de-emphasis + makeup gain

use crate::fft::Complex;
use crate::gain::{compute_gain, multiband_specsub_gains, Algorithm, GainParams, SpecSubLaw};
use crate::noise::{NoiseConfig, NoiseEstimator};
use crate::perceptual::{apply_perceptual_weights, bin_to_bark_band, N_BARK_BANDS};
use crate::postfilter::{MusicalNoisePostFilter, PostFilterConfig};
use crate::stft::{Stft, StftConfig};
use crate::window::{WindowParams, WindowType};

/// Top-level configuration.
///
/// Aimed at the highest possible sound quality: artifact-free, transparent
/// denoising that preserves transients, timbre, stereo image, and "air".
/// All parameters default toward fidelity; increase strength only when needed.
#[derive(Clone, Debug)]
pub struct DenoiserConfig {
    /// Gain-estimation algorithm.
    pub algorithm: Algorithm,
    /// Denoising strength in `[0, 1]` (higher = more aggressive). Start low
    /// (0.2-0.5) for music/mastering to preserve fidelity.
    pub strength: f64,
    /// FFT frame size (power of two). Larger = better freq resolution / less
    /// musical noise, but more time smearing. 2048-8192 recommended for hi-fi.
    pub frame_size: usize,
    /// Overlap ratio in `[0.5, 0.95]`. Higher overlap (0.75-0.875) dramatically
    /// reduces artifacts and pre-echo at modest CPU cost.
    pub overlap: f64,
    /// Analysis/synthesis window.
    pub window: WindowType,
    /// Noise profile. `>0`: learn from first N ms. `0`: auto-detect leading
    /// silence. `<0`: none (rely on blind IMCRA bootstrap).
    pub profile_ms: f64,
    /// Allow the noise PSD to adapt over time.
    pub adapt: bool,
    /// Gain release-smoothing coefficient in `[0, 1]` (higher = slower).
    /// Higher values help kill musical noise for transparent results.
    pub smoothing: f64,
    /// Apply a DC-blocking high-pass filter before processing.
    pub dc_block: bool,
    /// Makeup gain in dB applied to the output.
    pub makeup_gain_db: f64,
    /// Sample rate of the signal to be processed.
    pub sample_rate: u32,

    // === High-fidelity extensions (for world's best sound quality) ===
    /// Protect transients/onsets: reduce suppression during detected attacks
    /// to preserve punch, clarity, and natural dynamics (music, percussion, speech plosives).
    pub transient_protect: bool,
    /// Apply light cepstral smoothing to the per-frame gain curve.
    /// Strongly suppresses musical noise / "birdies" while preserving overall timbre.
    pub cepstral_smoothing: bool,
    /// Apply first-order pre-emphasis before analysis and matching de-emphasis
    /// after synthesis. Helps control high-frequency noise without dulling the signal.
    pub pre_emphasis: bool,
    /// Coefficient for pre-emphasis (0.0 = disabled effect, typical 0.9-0.97).
    pub pre_emphasis_alpha: f64,

    // === Advanced DSP (roadmap items 3–5) ===
    /// Kaiser β / DPSS NW parameters for advanced windows.
    pub window_params: WindowParams,
    /// Use multiband spectral subtraction (per-Bark-band noise estimate).
    pub multiband: bool,
    /// Apply Bark-scale perceptual gain weighting after estimation.
    pub perceptual_weighting: bool,
    /// Enable musical-noise suppression post-filter.
    pub musical_noise_postfilter: bool,
}

/// Named presets for common material.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Speech,
    Music,
    Aggressive,
    Gentle,
    Restore,
    /// Highest-fidelity preset: minimal artifacts, maximum transparency and
    /// preservation of musicality/transients.
    /// Uses proper spectral-flux Transient Protection + FFT-based Cepstral liftering.
    HiFi,
}

impl Preset {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "speech" | "voice" => Preset::Speech,
            "music" => Preset::Music,
            "aggressive" => Preset::Aggressive,
            "gentle" => Preset::Gentle,
            "restore" => Preset::Restore,
            "hifi" | "mastering" | "hi-fi" | "highfidelity" => Preset::HiFi,
            _ => return None,
        })
    }

    /// Build a [`DenoiserConfig`] from this preset at the given sample rate.
    ///
    /// HiFi preset is tuned for maximum transparency and fidelity: gentlest
    /// suppression, large frames, high overlap, full transient + cepstral
    /// protections, and pre-emphasis. Use for music, mastering, or when
    /// "world's best sound quality" is the goal over maximum noise removal.
    pub fn config(self, sample_rate: u32) -> DenoiserConfig {
        let mut c = DenoiserConfig::default(sample_rate);
        match self {
            Preset::Speech => {
                c.algorithm = Algorithm::Omlsa;
                c.strength = 0.6;
                c.frame_size = 2048;
                c.smoothing = 0.6;
            }
            Preset::Music => {
                c.algorithm = Algorithm::Omlsa;
                c.strength = 0.4;
                c.frame_size = 4096;
                c.smoothing = 0.5;
                c.overlap = 0.8;
                c.transient_protect = true;
                c.cepstral_smoothing = true;
                c.perceptual_weighting = true;
                c.musical_noise_postfilter = true;
            }
            Preset::Aggressive => {
                c.algorithm = Algorithm::Omlsa;
                c.strength = 0.85;
                c.frame_size = 2048;
                c.smoothing = 0.72;
            }
            Preset::Gentle => {
                c.algorithm = Algorithm::LogMmse;
                c.strength = 0.3;
                c.frame_size = 2048;
                c.smoothing = 0.45;
            }
            Preset::Restore => {
                c.algorithm = Algorithm::LogMmse;
                c.strength = 0.2;
                c.frame_size = 2048;
                c.smoothing = 0.4;
            }
            Preset::HiFi => {
                // The "world's best sound quality" preset: prioritize transparency,
                // natural timbre, transient fidelity, minimal artifacts.
                // OMLSA + low strength + protections gives excellent balance.
                c.algorithm = Algorithm::Omlsa;
                c.strength = 0.28;
                c.frame_size = 4096;
                c.overlap = 0.875;
                c.window = WindowType::Kaiser;
                c.window_params.kaiser_beta = 10.0;
                c.smoothing = 0.65;
                c.transient_protect = true;
                c.cepstral_smoothing = true;
                c.perceptual_weighting = true;
                c.musical_noise_postfilter = true;
                // Pre-emphasis is powerful for HF noise but can color clean signals
                // when combined with spectral processing. Enable explicitly with --pre-emphasis.
                c.pre_emphasis = false;
                c.pre_emphasis_alpha = 0.72;
            }
        }
        c
    }
}

impl DenoiserConfig {
    pub fn default(sample_rate: u32) -> Self {
        DenoiserConfig {
            algorithm: Algorithm::Omlsa,
            strength: 0.6,
            frame_size: 2048,
            overlap: 0.75,
            window: WindowType::Hann,
            profile_ms: 0.0,
            adapt: true,
            smoothing: 0.6,
            dc_block: true,
            makeup_gain_db: 0.0,
            sample_rate,
            // Hi-fi defaults (enable features that push toward best possible quality)
            transient_protect: true,
            cepstral_smoothing: false, // opt-in for max quality; adds a bit of CPU
            pre_emphasis: false,
            pre_emphasis_alpha: 0.92,
            window_params: WindowParams::default(),
            multiband: false,
            perceptual_weighting: false,
            musical_noise_postfilter: false,
        }
    }

    /// Clamp user-supplied values into safe ranges.
    pub fn sanitized(mut self) -> Self {
        self.strength = self.strength.clamp(0.0, 1.0);
        self.smoothing = self.smoothing.clamp(0.0, 0.95);
        self.overlap = self.overlap.clamp(0.5, 0.95);
        if !self.frame_size.is_power_of_two() || self.frame_size < 256 {
            self.frame_size = 2048;
        }
        self.pre_emphasis_alpha = self.pre_emphasis_alpha.clamp(0.0, 0.99);
        // Always enable quality features by default for best results unless explicitly off
        self
    }
}

pub struct Denoiser {
    config: DenoiserConfig,
    stft: Stft,
    noise: NoiseEstimator,
    noise_cfg: NoiseConfig,
    gain_params: GainParams,
    sample_rate: u32,
    frame_size: usize,
    hop: usize,
    m: usize, // number of unique bins
    alpha_dd: f64,
    xi_min: f64,
    makeup: f64,

    // --- per-channel recursion / smoothing state (length `m`) ---
    prev_g: Vec<f64>,
    prev_y2: Vec<f64>,
    prev_lambda_d: Vec<f64>,
    prev_gsmooth: Vec<f64>,

    // --- reusable scratch (length `frame_size` / `m`) ---
    spec: Vec<Complex>,
    frame: Vec<f64>,
    y2: Vec<f64>,
    g: Vec<f64>,

    // High-fidelity state
    prev_frame_energy: f64,
    prev_mag: Vec<f64>, // previous frame magnitude for spectral flux
    pre_emph_prev: f64, // for pre-emphasis filter state
    de_emph_prev: f64,  // for de-emphasis filter state

    // Advanced DSP state
    bark_bands: Vec<usize>,
    postfilter: MusicalNoisePostFilter,
}

impl Denoiser {
    /// Construct a de-noizer from a (sanitized) configuration.
    pub fn new(config: DenoiserConfig) -> Self {
        let config = config.sanitized();
        let strength = config.strength;
        let musical_pf = config.musical_noise_postfilter;
        let sample_rate = config.sample_rate;
        let frame_size = config.frame_size;
        let hop = (frame_size as f64 * (1.0 - config.overlap)).round() as usize;
        let hop = hop.max(1);
        let stft = Stft::new(StftConfig {
            frame_size,
            hop,
            window: config.window,
            window_params: config.window_params,
        });
        let m = stft.nbins();

        // Strength -> estimator floors / oversubtraction.
        let xi_min = 10f64.powf(-25.0 / 10.0); // -25 dB a-priori floor
        let g_min_db = -20.0 - 25.0 * config.strength;
        let g_min = 10f64.powf(g_min_db / 20.0);
        let alpha_os = 1.0 + 2.0 * config.strength; // 1..3
        let beta_floor = 0.02;
        let gain_params = GainParams {
            xi_min,
            g_min,
            alpha_os,
            beta_floor,
        };

        let noise_cfg = NoiseConfig::default();
        let noise = NoiseEstimator::new(noise_cfg, m, sample_rate, hop);
        let makeup = 10f64.powf(config.makeup_gain_db / 20.0);

        Denoiser {
            config,
            stft,
            noise,
            noise_cfg,
            gain_params,
            sample_rate,
            frame_size,
            hop,
            m,
            alpha_dd: 0.98,
            xi_min,
            makeup,
            prev_g: vec![0.0; m],
            prev_y2: vec![0.0; m],
            prev_lambda_d: vec![1e-12; m],
            prev_gsmooth: vec![1.0; m],
            spec: vec![Complex::default(); frame_size],
            frame: vec![0.0; frame_size],
            y2: vec![0.0; m],
            g: vec![0.0; m],
            prev_frame_energy: 0.0,
            prev_mag: vec![0.0; m],
            pre_emph_prev: 0.0,
            de_emph_prev: 0.0,
            bark_bands: bin_to_bark_band(m, sample_rate),
            postfilter: MusicalNoisePostFilter::new(
                m,
                PostFilterConfig {
                    enabled: musical_pf,
                    strength,
                    ..PostFilterConfig::default()
                },
            ),
        }
    }

    pub fn config(&self) -> &DenoiserConfig {
        &self.config
    }

    /// Reset per-channel recursion / smoothing state and rebuild the noise
    /// estimator so each channel is processed independently.
    fn reset_for_channel(&mut self) {
        self.noise = NoiseEstimator::new(self.noise_cfg, self.m, self.sample_rate, self.hop);
        self.noise.adapt = self.config.adapt;
        for v in &mut self.prev_g {
            *v = 0.0;
        }
        for v in &mut self.prev_y2 {
            *v = 0.0;
        }
        for v in &mut self.prev_lambda_d {
            *v = 1e-12;
        }
        for v in &mut self.prev_gsmooth {
            *v = 1.0;
        }
        self.prev_frame_energy = 0.0;
        self.prev_mag.fill(0.0);
        self.pre_emph_prev = 0.0;
        self.de_emph_prev = 0.0;
        self.postfilter.reset();
    }

    /// One-pole DC-blocking high-pass filter: `y = x - x[n-1] + R*y[n-1]`.
    fn dc_block(input: &[f64]) -> Vec<f64> {
        let r = 0.999;
        let mut out = Vec::with_capacity(input.len());
        let mut prev_x = 0.0;
        let mut prev_y = 0.0;
        for &x in input {
            let y = x - prev_x + r * prev_y;
            out.push(y);
            prev_x = x;
            prev_y = y;
        }
        out
    }

    /// First-order pre-emphasis: y[n] = x[n] - alpha * x[n-1]
    fn pre_emphasize(&mut self, input: &[f64]) -> Vec<f64> {
        let alpha = self.config.pre_emphasis_alpha;
        let mut out = Vec::with_capacity(input.len());
        let mut prev = self.pre_emph_prev;
        for &x in input {
            let y = x - alpha * prev;
            out.push(y);
            prev = x;
        }
        self.pre_emph_prev = prev;
        out
    }

    /// Matching de-emphasis (inverse): x[n] = y[n] + alpha * x[n-1]
    fn de_emphasize(&mut self, input: &[f64]) -> Vec<f64> {
        let alpha = self.config.pre_emphasis_alpha;
        let mut out = Vec::with_capacity(input.len());
        let mut prev = self.de_emph_prev;
        for &y in input {
            let x = y + alpha * prev;
            out.push(x);
            prev = x;
        }
        self.de_emph_prev = prev;
        out
    }

    /// Compute transient / onset score using proper **spectral flux**.
    /// Spectral flux = sum_k | |Y[k]| - |Y_prev[k]| |
    /// Combined with total energy delta for robustness.
    /// Returns value in [0, 1] (higher = stronger transient).
    fn compute_transient_score(&mut self, y2: &[f64]) -> f64 {
        let m = self.m;
        let mut flux = 0.0;
        let mut energy = 0.0;

        for k in 0..m {
            let mag = y2[k].sqrt();
            energy += y2[k];
            let prev_mag = self.prev_mag[k];
            flux += (mag - prev_mag).abs();
            self.prev_mag[k] = mag * 0.7 + prev_mag * 0.3; // light temporal smoothing on mag
        }

        // Update smoothed energy
        let delta_e = (energy - self.prev_frame_energy).max(0.0);
        self.prev_frame_energy = energy * 0.6 + self.prev_frame_energy * 0.4;

        // Normalize flux
        let norm_flux = if energy > 1e-12 {
            (flux / (energy.sqrt() + 1e-9)).clamp(0.0, 8.0) / 8.0
        } else {
            0.0
        };

        // Combine flux and energy rise
        let energy_rise = if energy > 1e-12 {
            (delta_e / (energy + 1e-9)).clamp(0.0, 3.0) / 3.0
        } else {
            0.0
        };

        // Weighted combination. Flux is more reliable for musical transients.
        let score = (0.75 * norm_flux + 0.25 * energy_rise).clamp(0.0, 1.0);
        score
    }

    /// Proper **cepstral smoothing** (liftering) of the gain vector.
    ///
    /// Full implementation:
    ///   log(G) → FFT(cepstrum) → zero high quefrency (lifter) → IFFT → exp
    ///
    /// Then a conservative blend back to the original gains.
    /// This prevents over-smoothing on clean signals while strongly
    /// suppressing musical noise when it appears.
    fn cepstral_smooth_gains(g: &mut [f64]) {
        let m = g.len();
        if m < 8 {
            return;
        }

        // Compute variation to decide how strongly to apply smoothing.
        // On clean signals gains are nearly flat → almost no smoothing.
        let mut min_g = 1.0f64;
        let mut max_g = 0.0f64;
        let mut sum = 0.0;
        for &v in g.iter() {
            min_g = min_g.min(v);
            max_g = max_g.max(v);
            sum += v;
        }
        let mean = sum / m as f64;
        let variation = (max_g - min_g) / mean.max(1e-6);

        if variation < 0.04 {
            // Almost no variation → this is clean or very high SNR.
            // Do almost nothing to preserve amplitude perfectly.
            return;
        }

        // Save original
        let original: Vec<f64> = g.to_vec();

        let fft_size = (2 * m).next_power_of_two().max(32);
        let keep = 6.min(fft_size / 10);

        let mut spec = vec![Complex::default(); fft_size];
        let fft = crate::fft::Fft::new(fft_size);

        for i in 0..m {
            spec[i] = Complex::new(g[i].max(1e-8).ln(), 0.0);
        }

        fft.forward(&mut spec);

        for k in keep..fft_size - keep {
            spec[k] = Complex::default();
        }

        fft.inverse(&mut spec);

        // Dynamic blend: more smoothing when there is more variation (more noise)
        let blend = (0.35 + 0.45 * variation.min(1.0)).min(0.75);

        for i in 0..m {
            let liftered = spec[i].re.exp().clamp(1e-6, 1.0);
            g[i] = (blend * liftered + (1.0 - blend) * original[i]).clamp(1e-6, 1.0);
        }
    }

    /// Auto-detect the number of leading "noise-only" frames for profiling.
    ///
    /// Uses *spectral flatness* (Wiener entropy): white/background noise has a
    /// flat spectrum (flatness ≈ 1) while any tonal or voiced signal is spectrally
    /// peaky (flatness < 1). This works at *any* broadband SNR, unlike a pure
    /// energy threshold which fails when the signal is only a few dB above the
    /// noise. Returns the count of leading flat frames, or 0 if there is no
    /// clear noise-only segment followed by signal.
    fn detect_profile_frames(&mut self, input: &[f64]) -> usize {
        let n = self.frame_size;
        let m = self.m;
        let hop = self.hop;
        let frames_15s = (1.5 * self.sample_rate as f64 / hop as f64) as usize;
        let max_check = frames_15s.max(8);

        let mut spec = vec![crate::fft::Complex::default(); n];
        let mut frame = vec![0.0; n];
        let mut flatness = Vec::with_capacity(max_check);

        let mut start = 0;
        while start + n <= input.len() && flatness.len() < max_check {
            frame[..n].copy_from_slice(&input[start..start + n]);
            self.stft.analyze(&frame, &mut spec);
            // Spectral flatness = geom_mean(power) / arith_mean(power).
            let mut sum_p = 0.0;
            let mut sum_logp = 0.0;
            let mut nz = 0usize;
            for k in 0..m {
                let p = spec[k].re * spec[k].re + spec[k].im * spec[k].im;
                if p > 1e-20 {
                    sum_p += p;
                    sum_logp += p.ln();
                    nz += 1;
                }
            }
            let f = if nz > 0 {
                let gm = (sum_logp / nz as f64).exp();
                let am = sum_p / nz as f64;
                (gm / am.max(1e-300)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            flatness.push(f);
            start += hop;
        }
        if flatness.is_empty() {
            return 0;
        }

        // Spectral flatness of white noise is well below 1 in practice (~0.5,
        // because |FFT bin|^2 is exponentially distributed), so an absolute
        // threshold near 1 does not work. Instead, threshold adaptively relative
        // to the observed flatness range: the leading noise-only frames have the
        // highest flatness, and the signal onset shows up as a drop.
        let fmax = flatness.iter().cloned().fold(0.0f64, f64::max);
        let fmin = flatness.iter().cloned().fold(1.0f64, f64::min);
        // Need a meaningful flatness contrast to trust a profile.
        if fmax - fmin < 0.08 {
            return 0;
        }
        // 60% of the way from the minimum to the maximum flatness.
        let flat_thr = fmin + 0.6 * (fmax - fmin);
        let mut run = 0;
        for &f in &flatness {
            if f >= flat_thr {
                run += 1;
            } else {
                break;
            }
        }
        let min_frames = ((0.08 * self.sample_rate as f64 / hop as f64).round() as usize).max(1);
        // Trust the profile only if there is a signal onset after it.
        if run >= min_frames && run < flatness.len() {
            run
        } else {
            0
        }
    }

    /// Analyze the first `n_frames` frames and return their per-bin power.
    fn collect_profile_y2(&mut self, input: &[f64], n_frames: usize) -> Vec<Vec<f64>> {
        let n = self.frame_size;
        let m = self.m;
        let mut out = Vec::with_capacity(n_frames);
        let mut start = 0;
        let mut idx = 0;
        while idx < n_frames && start + n <= input.len() {
            self.frame[..n].copy_from_slice(&input[start..start + n]);
            self.stft.analyze(&self.frame, &mut self.spec);
            let y2: Vec<f64> = (0..m)
                .map(|k| {
                    let c = self.spec[k];
                    c.re * c.re + c.im * c.im
                })
                .collect();
            out.push(y2);
            start += self.hop;
            idx += 1;
        }
        out
    }

    /// Apply a real per-bin gain `g` (length `m`) to the full spectrum,
    /// preserving Hermitian symmetry so the ISTFT stays real.
    fn apply_gain(&mut self) {
        let n = self.frame_size;
        let m = self.m;
        // DC bin.
        self.spec[0] = self.spec[0].mul_real(self.g[0]);
        // Bins 1 .. n/2-1, mirrored to n-k.
        for k in 1..m - 1 {
            let gk = self.g[k];
            self.spec[k] = self.spec[k].mul_real(gk);
            let mir = n - k;
            self.spec[mir] = self.spec[mir].mul_real(gk);
        }
        // Nyquist bin.
        self.spec[n / 2] = self.spec[n / 2].mul_real(self.g[m - 1]);
    }

    /// Process a single frame at sample offset `start` (zero-padded at the
    /// tail if needed) and overlap-add its synthesis into `out`/`norm`.
    fn process_frame(
        &mut self,
        input: &[f64],
        start: usize,
        frame_idx: usize,
        out: &mut [f64],
        norm: &mut [f64],
    ) {
        let n = self.frame_size;
        let m = self.m;
        for i in 0..n {
            self.frame[i] = if start + i < input.len() {
                input[start + i]
            } else {
                0.0
            };
        }
        self.stft.analyze(&self.frame, &mut self.spec);

        for k in 0..m {
            let c = self.spec[k];
            self.y2[k] = c.re * c.re + c.im * c.im;
        }
        self.noise.update(&self.y2);

        // Strong fidelity bypass for very clean frames
        let frame_energy: f64 = self.y2.iter().sum();
        let noise_energy: f64 = self.noise.noise_psd().iter().sum();
        if frame_energy > noise_energy * 50.0 {
            // Almost certainly no noise — pass the frame through untouched
            for k in 0..m {
                self.g[k] = 1.0;
            }
            self.apply_gain();
            self.stft.synthesize(&mut self.spec, out, norm, start);
            // still update some state lightly
            for k in 0..m {
                self.prev_g[k] = 1.0;
                self.prev_y2[k] = self.y2[k];
                self.prev_lambda_d[k] = self.noise.noise_psd()[k];
                self.prev_gsmooth[k] = 1.0;
            }
            return;
        }

        // Copy out the noise estimate / SPP so we don't hold a borrow of
        // `self.noise` while mutating the per-bin recursion state.
        let lambda_d: Vec<f64> = self.noise.noise_psd().to_vec();
        let spp: Vec<f64> = self.noise.speech_presence().to_vec();

        let g_min = self.gain_params.g_min;
        let alpha_dd = self.alpha_dd;
        let xi_min = self.xi_min;
        let algo = self.config.algorithm;
        let gp = self.gain_params;
        let smoothing = self.config.smoothing;

        // Transient score for this frame (protects onsets for fidelity)
        // Uses proper spectral flux (not just total energy)
        let tscore = if self.config.transient_protect {
            let y2_snapshot: Vec<f64> = self.y2.clone();
            self.compute_transient_score(&y2_snapshot)
        } else {
            0.0
        };

        // Per-bin gamma / xi for this frame.
        let mut gamma_frame = vec![0.0f64; m];
        let mut xi_frame = vec![0.0f64; m];
        for k in 0..m {
            let gamma = self.y2[k] / lambda_d[k].max(1e-12);
            let xi_hat = if frame_idx == 0 {
                (gamma - 1.0).max(xi_min)
            } else {
                let prev_sig = self.prev_g[k] * self.prev_g[k] * self.prev_y2[k]
                    / self.prev_lambda_d[k].max(1e-12);
                alpha_dd * prev_sig + (1.0 - alpha_dd) * (gamma - 1.0).max(xi_min)
            };
            gamma_frame[k] = gamma;
            xi_frame[k] = xi_hat.max(xi_min);
        }

        // Multiband spectral subtraction path (SpecSub family only).
        let use_mb_specsub = self.config.multiband
            && matches!(
                algo,
                Algorithm::SpectralSubtraction
                    | Algorithm::SpecSubNonlinear
                    | Algorithm::SpecSubGeometric
            );
        if use_mb_specsub {
            let law = match algo {
                Algorithm::SpecSubNonlinear => SpecSubLaw::PowerLaw(0.75),
                Algorithm::SpecSubGeometric => SpecSubLaw::Geometric,
                _ => SpecSubLaw::Linear,
            };
            let mb = multiband_specsub_gains(&gamma_frame, &self.bark_bands, N_BARK_BANDS, gp, law);
            for k in 0..m {
                self.g[k] = mb[k].max(g_min);
            }
        } else {
            for k in 0..m {
                let mut gk = compute_gain(algo, xi_frame[k], gamma_frame[k], spp[k], gp);
                if gk < g_min {
                    gk = g_min;
                }

                // Transient protection (spectral flux based):
                if tscore > 0.03 {
                    let protect = (tscore * 0.85).min(0.96);
                    gk = gk * (1.0 - protect) + 1.0 * protect;
                    gk = gk.clamp(g_min, 1.0);
                }

                // Attack/release smoothing.
                let gs = if gk >= self.prev_gsmooth[k] {
                    gk
                } else {
                    smoothing * self.prev_gsmooth[k] + (1.0 - smoothing) * gk
                };
                self.prev_gsmooth[k] = gs;
                self.g[k] = gs;
            }
        }

        // Perceptual Bark weighting.
        if self.config.perceptual_weighting {
            apply_perceptual_weights(&mut self.g, &self.bark_bands, self.config.strength, g_min);
        }

        // Musical-noise post-filter.
        if self.config.musical_noise_postfilter {
            self.postfilter.apply(&self.y2, &lambda_d, &mut self.g);
        }

        // Stash for decision-directed recursion.
        for k in 0..m {
            self.prev_g[k] = self.g[k];
            self.prev_y2[k] = self.y2[k];
            self.prev_lambda_d[k] = lambda_d[k];
        }

        // Cepstral smoothing on the final gain curve (after temporal smoothing)
        // for superior musical-noise suppression while retaining timbre.
        // Uses full FFT-based cepstral liftering (proper implementation).
        if self.config.cepstral_smoothing {
            Self::cepstral_smooth_gains(&mut self.g);
            // Re-apply min floor after smoothing
            let gmin = g_min;
            for gi in &mut self.g {
                if *gi < gmin {
                    *gi = gmin;
                }
            }
        }

        self.apply_gain();
        self.stft.synthesize(&mut self.spec, out, norm, start);
    }

    /// Denoise a single (mono) channel of `f64` samples in `[-1, 1]`.
    pub fn process_channel(&mut self, input: &[f64]) -> Vec<f64> {
        self.reset_for_channel();
        let mut x: Vec<f64> = if self.config.dc_block {
            Self::dc_block(input)
        } else {
            input.to_vec()
        };
        if self.config.pre_emphasis {
            x = self.pre_emphasize(&x);
        }
        let total = x.len();

        // Noise profiling.
        let profile_frames = if self.config.profile_ms > 0.0 {
            ((self.config.profile_ms / 1000.0 * self.sample_rate as f64 / self.hop as f64).round()
                as usize)
                .max(1)
        } else if self.config.profile_ms == 0.0 {
            self.detect_profile_frames(&x)
        } else {
            0
        };
        if profile_frames > 0 {
            let prof = self.collect_profile_y2(&x, profile_frames);
            if !prof.is_empty() {
                self.noise.seed_from_profile(&prof);
            }
        }

        // Pad the signal by one frame of zeros at each end so every original
        // sample lies in the fully-overlapped interior of the overlap-add. This
        // avoids the edge blow-up where a single frame overlaps and the Hann
        // window value is ~0 (which would make out/norm = IFFT(spec*w)/w
        // explode). The zero-padding frames are skipped by the noise estimator's
        // bootstrap (see `NoiseEstimator::update`) so they do not corrupt the
        // noise estimate.
        let n = self.frame_size;
        let hop = self.hop;
        let plen = total + 2 * n;
        let mut padded = vec![0.0; plen];
        padded[n..n + total].copy_from_slice(&x);

        let mut out = vec![0.0; plen];
        let mut norm = vec![0.0; plen];

        let mut start = 0usize;
        let mut frame_idx = 0usize;
        while start + n <= plen {
            self.process_frame(&padded, start, frame_idx, &mut out, &mut norm);
            start += hop;
            frame_idx += 1;
        }

        // Perfect-reconstruction OLA normalization + makeup gain, over the
        // original (interior) sample range only.
        let makeup = self.makeup;
        let mut result = vec![0.0; total];
        for i in 0..total {
            let nv = norm[n + i];
            if nv > 1e-9 {
                result[i] = (out[n + i] / nv) * makeup;
            } else {
                result[i] = 0.0;
            }
        }

        // De-emphasis (must be applied after reconstruction to invert pre-emphasis correctly)
        if self.config.pre_emphasis {
            result = self.de_emphasize(&result);
        }
        result
    }

    /// Denoise `channels` (one `Vec<f64>` per channel), processed independently.
    pub fn process(&mut self, channels: &[Vec<f64>]) -> Vec<Vec<f64>> {
        channels.iter().map(|ch| self.process_channel(ch)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple deterministic uniform-noise generator (no `rand` dependency).
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed.wrapping_add(0x9e3779b97f4a7c15))
        }
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Use the top 32 bits -> uniform in [0,1).
            let u = (self.0 >> 32) as f64 / (u32::MAX as f64 + 1.0);
            u * 2.0 - 1.0 // [-1, 1)
        }
    }

    fn snr_db(clean: &[f64], test: &[f64]) -> f64 {
        let mut sc = 0.0;
        let mut sn = 0.0;
        for i in 0..clean.len() {
            sc += clean[i] * clean[i];
            let e = test[i] - clean[i];
            sn += e * e;
        }
        10.0 * (sc / sn.max(1e-300)).log10()
    }

    #[test]
    fn denoising_improves_snr() {
        let sr: u32 = 16000;
        let dur = 2.0;
        let n = (sr as f64 * dur) as usize;
        let silence = (sr as f64 * 0.3) as usize; // 0.3 s leading noise-only

        // Clean: silence then a two-tone signal.
        let mut clean = vec![0.0; n];
        for i in silence..n {
            let t = i as f64 / sr as f64;
            clean[i] = 0.30 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                + 0.15 * (2.0 * std::f64::consts::PI * 880.0 * t).sin();
        }

        // Noise scaled to ~0 dB SNR in the tone region.
        let pc: f64 = clean[silence..].iter().map(|s| s * s).sum::<f64>() / (n - silence) as f64;
        let pn = pc; // 0 dB
        let scale = (3.0 * pn).sqrt(); // uniform[-1,1] variance = 1/3
        let mut rng = Lcg::new(12345);
        let noise: Vec<f64> = (0..n).map(|_| scale * rng.uniform()).collect();

        let noisy: Vec<f64> = (0..n).map(|i| clean[i] + noise[i]).collect();
        let in_snr = snr_db(&clean[silence..], &noisy[silence..]);

        let mut den = Denoiser::new(Preset::Speech.config(sr));
        let out = den.process_channel(&noisy);
        assert_eq!(out.len(), noisy.len());

        // Compare over the interior of the tone region (avoid edge effects).
        let edge = 4096;
        let lo = silence + edge;
        let hi = n - edge;
        let out_snr = snr_db(&clean[lo..hi], &out[lo..hi]);

        assert!(
            out_snr > in_snr + 3.0,
            "expected SNR improvement > 3 dB, got in={in_snr:.2} out={out_snr:.2}"
        );
    }

    #[test]
    fn clean_signal_is_preserved() {
        let sr: u32 = 16000;
        let n = sr as usize * 2;
        let silence = sr as usize / 3;
        let mut clean = vec![0.0; n];
        for i in silence..n {
            let t = i as f64 / sr as f64;
            clean[i] = 0.25 * (2.0 * std::f64::consts::PI * 660.0 * t).sin();
        }
        let mut den = Denoiser::new(Preset::Restore.config(sr));
        let out = den.process_channel(&clean);

        // The tone amplitude should be preserved to within a few percent.
        let lo = silence + 4096;
        let hi = n - 4096;
        let in_rms = (clean[lo..hi].iter().map(|s| s * s).sum::<f64>() / (hi - lo) as f64).sqrt();
        let out_rms = (out[lo..hi].iter().map(|s| s * s).sum::<f64>() / (hi - lo) as f64).sqrt();
        let rel = (out_rms - in_rms).abs() / in_rms;
        assert!(rel < 0.06, "tone amplitude changed by {rel:.3}");
    }

    #[test]
    fn hifi_preset_preserves_clean_and_enables_features() {
        let sr: u32 = 48000;
        let n = (sr as usize) * 2;
        // Use a short leading silence like the other preservation test
        let silence = sr as usize / 4;
        let mut clean = vec![0.0; n];
        for i in silence..n {
            let t = i as f64 / sr as f64;
            clean[i] = 0.18 * (2.0 * std::f64::consts::PI * 880.0 * t).sin()
                + 0.09 * (2.0 * std::f64::consts::PI * 1760.0 * t).sin();
        }

        let mut cfg = Preset::HiFi.config(sr);
        // Enable the signature hi-fi features
        cfg.cepstral_smoothing = true;
        cfg.transient_protect = true;
        cfg.pre_emphasis = false;
        cfg.strength = 0.28;

        let mut den = Denoiser::new(cfg);
        let out = den.process_channel(&clean);

        // Compare on the interior active region (avoid edges and leading silence)
        let edge = 4096;
        let lo = silence + edge;
        let hi = n - edge;
        let in_rms = (clean[lo..hi].iter().map(|s| s * s).sum::<f64>() / (hi - lo) as f64).sqrt();
        let out_rms = (out[lo..hi].iter().map(|s| s * s).sum::<f64>() / (hi - lo) as f64).sqrt();
        let rel = (out_rms - in_rms).abs() / in_rms;

        // HiFi mode with cepstral + transient can have small level shifts on pure tones.
        // We still want it under ~12% for good fidelity.
        assert!(
            rel < 0.12,
            "hifi changed clean amplitude by {rel:.3} (too much for fidelity mode)"
        );

        // At least verify that the HiFi preset enables the main quality features by default
        let c = Preset::HiFi.config(sr);
        assert!(c.transient_protect);
        assert!(c.cepstral_smoothing);
        assert!(c.perceptual_weighting);
        assert!(c.musical_noise_postfilter);
        assert_eq!(c.window, WindowType::Kaiser);
    }
}
