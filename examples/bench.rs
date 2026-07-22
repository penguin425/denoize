//! In-process benchmark: read noisy.wav + clean.wav, run several denoiser
//! configurations, and print the achieved SNR for each. Isolates algorithm
//! quality from WAV re-encoding.

use denoize::{read_wav, Algorithm, Denoiser, DenoiserConfig, Preset};

fn snr(clean: &[f64], test: &[f64], lo: usize, hi: usize) -> f64 {
    let mut sc = 0.0;
    let mut sn = 0.0;
    for i in lo..hi {
        sc += clean[i] * clean[i];
        let e = test[i] - clean[i];
        sn += e * e;
    }
    10.0 * (sc / sn.max(1e-300)).log10()
}

fn anomalies(out: &[f64]) -> (bool, f64) {
    let mut has_nan = false;
    let mut mx = 0.0f64;
    for &v in out {
        if v.is_nan() || v.is_infinite() {
            has_nan = true;
        }
        mx = mx.max(v.abs());
    }
    (has_nan, mx)
}

fn main() {
    let noisy = read_wav("noisy.wav").unwrap();
    let clean = read_wav("clean.wav").unwrap();
    let sr = noisy.sample_rate;
    let n = noisy.channels[0].len();
    let silence = (0.4 * sr as f64) as usize;
    let edge = 4096;
    let lo = silence + edge;
    let hi = n - edge;

    let in_snr = snr(&clean.channels[0], &noisy.channels[0], lo, hi);
    println!("input SNR : {in_snr:.2} dB\n");

    let mut cfgs: Vec<(&str, DenoiserConfig)> = vec![
        ("speech (omlsa)", Preset::Speech.config(sr)),
        ("music", Preset::Music.config(sr)),
        ("hifi (flagship)", Preset::HiFi.config(sr)),
        ("aggressive", Preset::Aggressive.config(sr)),
        ("gentle (logmmse)", Preset::Gentle.config(sr)),
        ("restore (logmmse)", Preset::Restore.config(sr)),
        ("omlsa s=0.9", {
            let mut c = Preset::Speech.config(sr);
            c.strength = 0.9;
            c
        }),
        ("logmmse s=0.6", {
            let mut c = Preset::Speech.config(sr);
            c.algorithm = Algorithm::LogMmse;
            c
        }),
        ("mmse s=0.6", {
            let mut c = Preset::Speech.config(sr);
            c.algorithm = Algorithm::MmseStsa;
            c
        }),
        ("wiener", {
            let mut c = Preset::Speech.config(sr);
            c.algorithm = Algorithm::Wiener;
            c
        }),
        ("specsub s=0.6", {
            let mut c = Preset::Speech.config(sr);
            c.algorithm = Algorithm::SpectralSubtraction;
            c
        }),
        ("speech no-profile", {
            let mut c = Preset::Speech.config(sr);
            c.profile_ms = -1.0;
            c
        }),
        ("speech no-adapt", {
            let mut c = Preset::Speech.config(sr);
            c.adapt = false;
            c
        }),
        ("hifi frame=8192 overlap=0.875", {
            let mut c = Preset::HiFi.config(sr);
            c.frame_size = 8192;
            c.overlap = 0.875;
            c
        }),
    ];

    println!(
        "{:<22} {:>10} {:>10}  note",
        "config", "SNR (dB)", "gain (dB)"
    );
    println!("{:-<60}", "");
    for (name, cfg) in cfgs.drain(..) {
        let mut den = Denoiser::new(cfg);
        let out = den.process_channel(&noisy.channels[0]);
        let (has_nan, mx) = anomalies(&out);
        let s = snr(&clean.channels[0], &out, lo, hi);
        let note = if has_nan {
            "NaN/Inf!".to_string()
        } else if s > 60.0 {
            format!("suspect (max={:.3})", mx)
        } else {
            format!("max={:.3}", mx)
        };
        println!("{:<22} {:>10.2} {:>10.2}  {}", name, s, s - in_snr, note);
    }
}
