//! Measure global SNR of a test WAV against a clean reference WAV.
//! Usage: `measure <clean.wav> <test.wav>` — prints SNR over the tone region.

use hound::WavReader;

fn read(path: &str) -> (u32, Vec<f64>) {
    let mut r = WavReader::open(path).unwrap();
    let spec = r.spec();
    let inv = 1.0 / ((1u64 << (spec.bits_per_sample - 1)) as f64);
    let samples: Vec<f64> = if spec.bits_per_sample <= 16 {
        r.samples::<i16>()
            .map(|s| s.unwrap() as f64 * inv)
            .collect()
    } else {
        r.samples::<i32>()
            .map(|s| s.unwrap() as f64 * inv)
            .collect()
    };
    (spec.sample_rate, samples)
}

fn snr(clean: &[f64], test: &[f64]) -> f64 {
    let mut sc = 0.0;
    let mut sn = 0.0;
    for i in 0..clean.len() {
        sc += clean[i] * clean[i];
        let e = test[i] - clean[i];
        sn += e * e;
    }
    10.0 * (sc / sn.max(1e-300)).log10()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: measure <clean.wav> <test.wav>");
        std::process::exit(1);
    }
    let (sr, clean) = read(&args[1]);
    let (_sr2, test) = read(&args[2]);
    assert_eq!(clean.len(), test.len(), "length mismatch");

    // Compare over the tone region interior (skip leading silence + edges).
    let silence = (0.4 * sr as f64) as usize;
    let edge = 4096;
    let lo = silence + edge;
    let hi = clean.len() - edge;
    let g = snr(&clean[lo..hi], &test[lo..hi]);
    println!("global SNR (tone region): {g:.2} dB");
}
