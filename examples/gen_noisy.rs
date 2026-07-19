//! Generate a realistic test signal: 0.4 s of noise-only "silence" followed by
//! a harmonic tone, plus Gaussian noise at ~0 dB SNR. Writes `clean.wav` and
//! `noisy.wav` (16-bit mono, 16 kHz).

use hound::{SampleFormat, WavSpec, WavWriter};
use std::f64::consts::PI;

struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Lcg(s.wrapping_add(0x9e3779b97f4a7c15))
    }
    fn u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn uniform(&mut self) -> f64 {
        self.u32() as f64 / (u32::MAX as f64)
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

fn main() {
    let sr = 16000u32;
    let silence = (0.4 * sr as f64) as usize;
    let dur = 4.0;
    let n = (sr as f64 * dur) as usize;

    let mut clean = vec![0.0f64; n];
    for i in silence..n {
        let t = i as f64 / sr as f64;
        clean[i] = 0.30 * (2.0 * PI * 440.0 * t).sin()
            + 0.12 * (2.0 * PI * 880.0 * t).sin()
            + 0.06 * (2.0 * PI * 1320.0 * t).sin();
    }

    // Scale noise to ~0 dB SNR in the tone region.
    let pc: f64 = clean[silence..].iter().map(|s| s * s).sum::<f64>() / (n - silence) as f64;
    let sigma = pc.sqrt();
    let mut rng = Lcg::new(98765);
    let noisy: Vec<f64> = (0..n)
        .map(|i| (clean[i] + rng.gauss() * sigma).clamp(-1.0, 1.0))
        .collect();

    let spec = WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let write = |path: &str, sig: &[f64]| {
        let mut w = WavWriter::create(path, spec).unwrap();
        for &v in sig {
            w.write_sample((v * 32767.0).round().clamp(-32768.0, 32767.0) as i16)
                .unwrap();
        }
        w.finalize().unwrap();
    };
    write("clean.wav", &clean);
    write("noisy.wav", &noisy);
    println!("wrote clean.wav and noisy.wav ({dur}s, 16-bit mono, ~0 dB SNR in tone region)");
}
