//! Objective quality and stereo-imaging reports.

use crate::Audio;

#[derive(Clone, Debug)]
pub struct BenchmarkReport {
    pub frames: usize,
    pub sample_rate: u32,
    pub channels: usize,
    pub si_sdr_db: f64,
    pub si_snr_db: f64,
    pub snr_db: f64,
    pub segmental_snr_db: f64,
    pub stereo_side_sdr_db: Option<f64>,
    pub correlation_error: Option<f64>,
    pub stoi: Option<f64>,
    pub pesq: Option<f64>,
    pub elapsed_ms: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
}

impl BenchmarkReport {
    pub fn compare(reference: &Audio, test: &Audio) -> Result<Self, String> {
        if reference.sample_rate != test.sample_rate {
            return Err("benchmark sample rates differ".into());
        }
        if reference.channels.len() != test.channels.len() || reference.channels.is_empty() {
            return Err("benchmark channel counts differ or are empty".into());
        }
        let frames = reference.frames().min(test.frames());
        if frames == 0 {
            return Err("benchmark inputs are empty".into());
        }
        let r = downmix(reference, frames);
        let t = downmix(test, frames);
        let (side_sdr, correlation_error) = if reference.channels.len() == 2 {
            let rs = side(reference, frames);
            let ts = side(test, frames);
            (
                Some(si_sdr(&rs, &ts)),
                Some(
                    (correlation(
                        &reference.channels[0][..frames],
                        &reference.channels[1][..frames],
                    ) - correlation(&test.channels[0][..frames], &test.channels[1][..frames]))
                    .abs(),
                ),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            frames,
            sample_rate: reference.sample_rate,
            channels: reference.channels.len(),
            si_sdr_db: si_sdr(&r, &t),
            si_snr_db: si_snr(&r, &t),
            snr_db: snr(&r, &t),
            segmental_snr_db: segmental_snr(&r, &t, reference.sample_rate),
            stereo_side_sdr_db: side_sdr,
            correlation_error,
            stoi: None,
            pesq: None,
            elapsed_ms: None,
            peak_rss_bytes: None,
        })
    }

    pub fn json(&self) -> String {
        format!("{{\"frames\":{},\"sample_rate\":{},\"channels\":{},\"si_sdr_db\":{:.6},\"si_snr_db\":{:.6},\"snr_db\":{:.6},\"segmental_snr_db\":{:.6},\"stereo_side_sdr_db\":{},\"correlation_error\":{},\"stoi\":{},\"pesq\":{},\"elapsed_ms\":{},\"peak_rss_bytes\":{}}}", self.frames, self.sample_rate, self.channels, self.si_sdr_db, self.si_snr_db, self.snr_db, self.segmental_snr_db, optional(self.stereo_side_sdr_db), optional(self.correlation_error), optional(self.stoi), optional(self.pesq), optional(self.elapsed_ms), self.peak_rss_bytes.map_or_else(|| "null".into(), |v| v.to_string()))
    }

    pub fn markdown(&self) -> String {
        format!("| Metric | Value |\n|---|---:|\n| SI-SDR | {:.3} dB |\n| SI-SNR | {:.3} dB |\n| SNR | {:.3} dB |\n| Segmental SNR | {:.3} dB |\n| Stereo side SDR | {} |\n| Correlation error | {} |\n| STOI | {} |\n| PESQ | {} |", self.si_sdr_db, self.si_snr_db, self.snr_db, self.segmental_snr_db, db(self.stereo_side_sdr_db), display(self.correlation_error, 6), display(self.stoi, 4), display(self.pesq, 3))
    }
}

#[derive(Clone, Debug)]
pub struct ComparisonReport {
    pub noisy: BenchmarkReport,
    pub enhanced: BenchmarkReport,
}

impl ComparisonReport {
    pub fn compare(clean: &Audio, noisy: &Audio, enhanced: &Audio) -> Result<Self, String> {
        Ok(Self {
            noisy: BenchmarkReport::compare(clean, noisy)?,
            enhanced: BenchmarkReport::compare(clean, enhanced)?,
        })
    }

    pub fn json(&self) -> String {
        format!(
            "{{\"noisy\":{},\"enhanced\":{},\"improvement\":{{\"si_sdr_db\":{:.6},\"si_snr_db\":{:.6},\"snr_db\":{:.6},\"segmental_snr_db\":{:.6}}}}}",
            self.noisy.json(), self.enhanced.json(),
            self.enhanced.si_sdr_db - self.noisy.si_sdr_db,
            self.enhanced.si_snr_db - self.noisy.si_snr_db,
            self.enhanced.snr_db - self.noisy.snr_db,
            self.enhanced.segmental_snr_db - self.noisy.segmental_snr_db,
        )
    }

    pub fn markdown(&self) -> String {
        format!(
            "| Metric | Noisy | Enhanced | Improvement |\n|---|---:|---:|---:|\n| SI-SDR | {:.3} dB | {:.3} dB | {:+.3} dB |\n| SI-SNR | {:.3} dB | {:.3} dB | {:+.3} dB |\n| SNR | {:.3} dB | {:.3} dB | {:+.3} dB |\n| Segmental SNR | {:.3} dB | {:.3} dB | {:+.3} dB |\n\nSTOI, PESQ, and DNSMOS: not measured (no external model or licensed reference implementation configured).",
            self.noisy.si_sdr_db, self.enhanced.si_sdr_db, self.enhanced.si_sdr_db - self.noisy.si_sdr_db,
            self.noisy.si_snr_db, self.enhanced.si_snr_db, self.enhanced.si_snr_db - self.noisy.si_snr_db,
            self.noisy.snr_db, self.enhanced.snr_db, self.enhanced.snr_db - self.noisy.snr_db,
            self.noisy.segmental_snr_db, self.enhanced.segmental_snr_db, self.enhanced.segmental_snr_db - self.noisy.segmental_snr_db,
        )
    }

    pub fn html(&self) -> String {
        let rows = self
            .markdown()
            .lines()
            .skip(2)
            .take(4)
            .map(|line| {
                let cells = line
                    .trim_matches('|')
                    .split('|')
                    .map(str::trim)
                    .map(|cell| format!("<td>{cell}</td>"))
                    .collect::<String>();
                format!("<tr>{cells}</tr>")
            })
            .collect::<String>();
        format!("<!doctype html><meta charset=\"utf-8\"><title>denoize comparison</title><style>body{{font-family:system-ui;max-width:900px;margin:3rem auto}}table{{border-collapse:collapse}}td,th{{padding:.5rem 1rem;border:1px solid #ccc;text-align:right}}td:first-child{{text-align:left}}</style><h1>denoize quality comparison</h1><table><thead><tr><th>Metric</th><th>Noisy</th><th>Enhanced</th><th>Improvement</th></tr></thead><tbody>{rows}</tbody></table><p>STOI, PESQ, and DNSMOS were not measured.</p>")
    }
}

fn optional(v: Option<f64>) -> String {
    v.map_or_else(|| "null".into(), |v| format!("{v:.6}"))
}
fn display(v: Option<f64>, precision: usize) -> String {
    v.map_or_else(|| "n/a".into(), |v| format!("{v:.precision$}"))
}
fn db(v: Option<f64>) -> String {
    v.map_or_else(|| "n/a".into(), |v| format!("{v:.3} dB"))
}
fn downmix(a: &Audio, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| a.channels.iter().map(|c| c[i]).sum::<f64>() / a.channels.len() as f64)
        .collect()
}
fn side(a: &Audio, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (a.channels[0][i] - a.channels[1][i]) * 0.5)
        .collect()
}

pub fn si_sdr(reference: &[f64], estimate: &[f64]) -> f64 {
    let dot = reference
        .iter()
        .zip(estimate)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let scale = dot / reference.iter().map(|x| x * x).sum::<f64>().max(1e-30);
    let target_energy = reference.iter().map(|x| (x * scale).powi(2)).sum::<f64>();
    let noise_energy = reference
        .iter()
        .zip(estimate)
        .map(|(a, b)| (a * scale - b).powi(2))
        .sum::<f64>();
    10.0 * (target_energy / noise_energy.max(1e-30)).log10()
}

pub fn si_snr(reference: &[f64], estimate: &[f64]) -> f64 {
    let rm = reference.iter().sum::<f64>() / reference.len() as f64;
    let em = estimate.iter().sum::<f64>() / estimate.len() as f64;
    si_sdr(
        &reference.iter().map(|x| x - rm).collect::<Vec<_>>(),
        &estimate.iter().map(|x| x - em).collect::<Vec<_>>(),
    )
}

pub fn snr(reference: &[f64], estimate: &[f64]) -> f64 {
    let signal = reference.iter().map(|sample| sample * sample).sum::<f64>();
    let noise = reference
        .iter()
        .zip(estimate)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>();
    10.0 * (signal / noise.max(1e-30)).log10()
}

pub fn segmental_snr(reference: &[f64], estimate: &[f64], sample_rate: u32) -> f64 {
    let window = (sample_rate as usize / 50).max(1);
    let mut values = Vec::new();
    for (r, e) in reference.chunks(window).zip(estimate.chunks(window)) {
        let signal = r.iter().map(|sample| sample * sample).sum::<f64>();
        if signal > 1e-12 {
            values.push(snr(r, e).clamp(-10.0, 35.0));
        }
    }
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let dot = a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    dot / (a.iter().map(|x| x * x).sum::<f64>() * b.iter().map(|x| x * x).sum::<f64>())
        .sqrt()
        .max(1e-30)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_ignore_gain() {
        let reference = [1.0, -1.0, 0.5, -0.5];
        let estimate = [0.5, -0.5, 0.25, -0.25];
        assert!(si_sdr(&reference, &estimate) > 250.0);
        assert!(si_snr(&reference, &estimate) > 250.0);
    }

    #[test]
    fn comparison_reports_quality_improvement_in_all_formats() {
        let clean = Audio {
            sample_rate: 16_000,
            channels: vec![(0..1600)
                .map(|index| (index as f64 * 0.031).sin() * 0.5)
                .collect()],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let noisy = Audio {
            sample_rate: clean.sample_rate,
            channels: vec![clean.channels[0]
                .iter()
                .enumerate()
                .map(|(index, sample)| sample + if index % 2 == 0 { 0.1 } else { -0.1 })
                .collect()],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let enhanced = Audio {
            sample_rate: clean.sample_rate,
            channels: vec![clean.channels[0]
                .iter()
                .enumerate()
                .map(|(index, sample)| sample + if index % 2 == 0 { 0.02 } else { -0.02 })
                .collect()],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let report = ComparisonReport::compare(&clean, &noisy, &enhanced).unwrap();
        assert!(report.enhanced.snr_db > report.noisy.snr_db);
        assert!(report.json().contains("\"improvement\""));
        assert!(report.markdown().contains("Segmental SNR"));
        assert!(report.html().starts_with("<!doctype html>"));
    }
}
