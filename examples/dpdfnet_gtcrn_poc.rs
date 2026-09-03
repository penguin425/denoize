//! Reproducible model-level comparison of DPDFNet-2 48 kHz HR and GTCRN.

use denoize::backend::dpdfnet::{MODEL_LOOKAHEAD_SAMPLES, SAMPLE_RATE as DPDFNET_RATE};
use denoize::{
    read_audio, write_wav, Audio, ComparisonReport, DpdfnetModel, GtcrnModel, OnnxModelConfig,
};
use serde_json::{json, Value};
use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const GTCRN_RATE: u32 = 16_000;
const DEFAULT_RUNS: usize = 3;

#[derive(Debug)]
struct Args {
    clean: PathBuf,
    noisy: PathBuf,
    dpdfnet_model: PathBuf,
    gtcrn_model: PathBuf,
    output_dir: Option<PathBuf>,
    json: Option<PathBuf>,
    runs: usize,
}

fn main() {
    if env::args()
        .skip(1)
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!("{}", usage());
        return;
    }
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let clean = read_audio(&args.clean)?;
    let noisy = read_audio(&args.noisy)?;
    validate_fixture(&clean, &noisy)?;
    let audio_seconds = clean.frames() as f64 / clean.sample_rate as f64;

    let started = Instant::now();
    let dpdfnet = DpdfnetModel::load(&OnnxModelConfig {
        path: args.dpdfnet_model.clone(),
        sample_rate: DPDFNET_RATE,
    })?;
    let dpdfnet_load = started.elapsed();

    let started = Instant::now();
    let gtcrn = GtcrnModel::load(&OnnxModelConfig {
        path: args.gtcrn_model.clone(),
        sample_rate: GTCRN_RATE,
    })?;
    let gtcrn_load = started.elapsed();

    let warmup_frames = clean
        .sample_rate
        .min(u32::try_from(noisy.frames()).unwrap_or(u32::MAX)) as usize;
    let warmup = vec![noisy.channels[0][..warmup_frames].to_vec()];
    dpdfnet.process(&warmup, clean.sample_rate)?;
    gtcrn.process(&warmup, clean.sample_rate)?;

    let (dpdfnet_channels, dpdfnet_times) = benchmark(args.runs, || {
        dpdfnet.process(&noisy.channels, noisy.sample_rate)
    })?;
    let (gtcrn_channels, gtcrn_times) = benchmark(args.runs, || {
        gtcrn.process(&noisy.channels, noisy.sample_rate)
    })?;

    let dpdfnet_raw = with_channels(&noisy, dpdfnet_channels)?;
    let dpdfnet_aligned = with_channels(
        &noisy,
        dpdfnet.process_aligned(&noisy.channels, noisy.sample_rate)?,
    )?;
    let gtcrn_output = with_channels(&noisy, gtcrn_channels)?;
    let alignment_samples = ((MODEL_LOOKAHEAD_SAMPLES as u64 * noisy.sample_rate as u64
        + (DPDFNET_RATE as u64 / 2))
        / DPDFNET_RATE as u64) as usize;

    let dpdfnet_raw_quality = ComparisonReport::compare(&clean, &noisy, &dpdfnet_raw)?;
    let dpdfnet_aligned_quality = ComparisonReport::compare(&clean, &noisy, &dpdfnet_aligned)?;
    let gtcrn_quality = ComparisonReport::compare(&clean, &noisy, &gtcrn_output)?;

    let noisy_mono = downmix(&noisy);
    let dpdfnet_delay_ms =
        estimate_nonnegative_delay_ms(&noisy_mono, &downmix(&dpdfnet_raw), noisy.sample_rate, 100);
    let gtcrn_delay_ms =
        estimate_nonnegative_delay_ms(&noisy_mono, &downmix(&gtcrn_output), noisy.sample_rate, 100);

    let dpdfnet_median = median(&dpdfnet_times);
    let gtcrn_median = median(&gtcrn_times);
    let result = json!({
        "schema": "denoize-dpdfnet-gtcrn-poc-v1",
        "fixture": {
            "clean": args.clean,
            "noisy": args.noisy,
            "sample_rate": clean.sample_rate,
            "channels": clean.channels(),
            "frames": clean.frames(),
            "duration_seconds": audio_seconds,
        },
        "models": {
            "dpdfnet2_48khz_hr": {
                "path": args.dpdfnet_model,
                "native_sample_rate": DPDFNET_RATE,
                "load_ms": milliseconds(dpdfnet_load),
                "process_ms_median": milliseconds(dpdfnet_median),
                "process_ms_runs": dpdfnet_times.iter().map(|value| milliseconds(*value)).collect::<Vec<_>>(),
                "rtf": dpdfnet_median.as_secs_f64() / audio_seconds,
                "throughput_realtime_x": audio_seconds / dpdfnet_median.as_secs_f64(),
                "first_output_after_ms": 20.0,
                "implementation_model_lookahead_ms": MODEL_LOOKAHEAD_SAMPLES as f64 * 1000.0 / DPDFNET_RATE as f64,
                "cross_correlation_delay_ms": dpdfnet_delay_ms,
                "alignment_samples_at_fixture_rate": alignment_samples,
                "quality_causal_unaligned": report_json(&dpdfnet_raw_quality)?,
                "quality_lookahead_aligned": report_json(&dpdfnet_aligned_quality)?,
            },
            "gtcrn": {
                "path": args.gtcrn_model,
                "native_sample_rate": GTCRN_RATE,
                "load_ms": milliseconds(gtcrn_load),
                "process_ms_median": milliseconds(gtcrn_median),
                "process_ms_runs": gtcrn_times.iter().map(|value| milliseconds(*value)).collect::<Vec<_>>(),
                "rtf": gtcrn_median.as_secs_f64() / audio_seconds,
                "throughput_realtime_x": audio_seconds / gtcrn_median.as_secs_f64(),
                "first_output_after_ms": 16.0,
                "finite_wrapper_latency_compensation_ms": 16.0,
                "cross_correlation_delay_ms": gtcrn_delay_ms,
                "quality": report_json(&gtcrn_quality)?,
            },
        }
    });

    print_summary(
        &result,
        &dpdfnet_aligned_quality,
        &gtcrn_quality,
        dpdfnet_median,
        gtcrn_median,
        audio_seconds,
    );

    if let Some(output_dir) = &args.output_dir {
        std::fs::create_dir_all(output_dir).map_err(|error| {
            format!("create output directory {}: {error}", output_dir.display())
        })?;
        write_wav(output_dir.join("dpdfnet-causal.wav"), &dpdfnet_raw)?;
        write_wav(output_dir.join("dpdfnet-aligned.wav"), &dpdfnet_aligned)?;
        write_wav(output_dir.join("gtcrn.wav"), &gtcrn_output)?;
    }
    if let Some(path) = &args.json {
        let bytes = serde_json::to_vec_pretty(&result)
            .map_err(|error| format!("encode PoC JSON: {error}"))?;
        std::fs::write(path, bytes)
            .map_err(|error| format!("write PoC JSON {}: {error}", path.display()))?;
    }
    Ok(())
}

fn benchmark<F>(runs: usize, mut process: F) -> Result<(Vec<Vec<f64>>, Vec<Duration>), String>
where
    F: FnMut() -> Result<Vec<Vec<f64>>, String>,
{
    let mut output = None;
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let started = Instant::now();
        let current = process()?;
        times.push(started.elapsed());
        output.get_or_insert(current);
    }
    Ok((output.unwrap_or_default(), times))
}

fn validate_fixture(clean: &Audio, noisy: &Audio) -> Result<(), String> {
    if clean.sample_rate != noisy.sample_rate
        || clean.channels() != noisy.channels()
        || clean.frames() != noisy.frames()
    {
        return Err(format!(
            "clean/noisy geometry differs: clean={} Hz/{} ch/{} frames, noisy={} Hz/{} ch/{} frames",
            clean.sample_rate,
            clean.channels(),
            clean.frames(),
            noisy.sample_rate,
            noisy.channels(),
            noisy.frames()
        ));
    }
    if clean.channels() == 0 || clean.frames() == 0 {
        return Err("fixture must contain at least one non-empty channel".into());
    }
    Ok(())
}

fn with_channels(template: &Audio, channels: Vec<Vec<f64>>) -> Result<Audio, String> {
    if channels.len() != template.channels()
        || channels.iter().any(|c| c.len() != template.frames())
    {
        return Err("model output geometry differs from the fixture".into());
    }
    let mut output = template.clone();
    output.channels = channels;
    Ok(output)
}

fn downmix(audio: &Audio) -> Vec<f64> {
    let scale = 1.0 / audio.channels() as f64;
    (0..audio.frames())
        .map(|frame| {
            audio
                .channels
                .iter()
                .map(|channel| channel[frame])
                .sum::<f64>()
                * scale
        })
        .collect()
}

fn estimate_nonnegative_delay_ms(
    reference: &[f64],
    test: &[f64],
    sample_rate: u32,
    maximum_ms: usize,
) -> usize {
    let stride = (sample_rate as usize / 1_000).max(1);
    (0..=maximum_ms)
        .max_by(|left, right| {
            let left_score = lag_correlation(reference, test, left * stride, stride);
            let right_score = lag_correlation(reference, test, right * stride, stride);
            left_score.total_cmp(&right_score)
        })
        .unwrap_or(0)
}

fn lag_correlation(reference: &[f64], test: &[f64], lag: usize, stride: usize) -> f64 {
    if lag >= reference.len().min(test.len()) {
        return f64::NEG_INFINITY;
    }
    let pairs: Vec<(f64, f64)> = (0..reference.len() - lag)
        .step_by(stride)
        .map(|index| (reference[index], test[index + lag]))
        .collect();
    if pairs.len() < 2 {
        return f64::NEG_INFINITY;
    }
    let scale = 1.0 / pairs.len() as f64;
    let reference_mean = pairs.iter().map(|pair| pair.0).sum::<f64>() * scale;
    let test_mean = pairs.iter().map(|pair| pair.1).sum::<f64>() * scale;
    let mut covariance = 0.0;
    let mut reference_energy = 0.0;
    let mut test_energy = 0.0;
    for (reference, test) in pairs {
        let reference = reference - reference_mean;
        let test = test - test_mean;
        covariance += reference * test;
        reference_energy += reference * reference;
        test_energy += test * test;
    }
    covariance / (reference_energy * test_energy).sqrt().max(1.0e-20)
}

fn median(values: &[Duration]) -> Duration {
    let mut values = values.to_vec();
    values.sort_unstable();
    values[values.len() / 2]
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn report_json(report: &ComparisonReport) -> Result<Value, String> {
    serde_json::from_str(&report.json()).map_err(|error| format!("decode quality report: {error}"))
}

fn print_summary(
    result: &Value,
    dpdfnet: &ComparisonReport,
    gtcrn: &ComparisonReport,
    dpdfnet_time: Duration,
    gtcrn_time: Duration,
    audio_seconds: f64,
) {
    println!("# DPDFNet-2 48 kHz HR vs GTCRN PoC\n");
    println!("| Model | SI-SDR improvement | STOI | median RTF | realtime throughput | estimated content delay |");
    println!("|---|---:|---:|---:|---:|---:|");
    println!(
        "| DPDFNet-2 48 kHz HR (4-hop aligned) | {:+.3} dB | {} | {:.4} | {:.2}x | {} ms |",
        dpdfnet.enhanced.si_sdr_db - dpdfnet.noisy.si_sdr_db,
        display_optional(dpdfnet.enhanced.stoi),
        dpdfnet_time.as_secs_f64() / audio_seconds,
        audio_seconds / dpdfnet_time.as_secs_f64(),
        result["models"]["dpdfnet2_48khz_hr"]["cross_correlation_delay_ms"]
    );
    println!(
        "| GTCRN (finite-wrapper aligned) | {:+.3} dB | {} | {:.4} | {:.2}x | {} ms |",
        gtcrn.enhanced.si_sdr_db - gtcrn.noisy.si_sdr_db,
        display_optional(gtcrn.enhanced.stoi),
        gtcrn_time.as_secs_f64() / audio_seconds,
        audio_seconds / gtcrn_time.as_secs_f64(),
        result["models"]["gtcrn"]["cross_correlation_delay_ms"]
    );
}

fn display_optional(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".into(), |value| format!("{value:.4}"))
}

fn parse_args() -> Result<Args, String> {
    let mut args = env::args().skip(1);
    let mut clean = None;
    let mut noisy = None;
    let mut dpdfnet_model = None;
    let mut gtcrn_model = None;
    let mut output_dir = None;
    let mut json = None;
    let mut runs = DEFAULT_RUNS;
    while let Some(argument) = args.next() {
        let value = |args: &mut std::iter::Skip<std::env::Args>| {
            args.next()
                .ok_or_else(|| format!("missing value after `{argument}`"))
        };
        match argument.as_str() {
            "--clean" => clean = Some(PathBuf::from(value(&mut args)?)),
            "--noisy" => noisy = Some(PathBuf::from(value(&mut args)?)),
            "--dpdfnet-model" => dpdfnet_model = Some(PathBuf::from(value(&mut args)?)),
            "--gtcrn-model" => gtcrn_model = Some(PathBuf::from(value(&mut args)?)),
            "--output-dir" => output_dir = Some(PathBuf::from(value(&mut args)?)),
            "--json" => json = Some(PathBuf::from(value(&mut args)?)),
            "--runs" => {
                runs = value(&mut args)?
                    .parse()
                    .map_err(|_| "--runs must be a positive integer".to_string())?;
                if runs == 0 {
                    return Err("--runs must be a positive integer".into());
                }
            }
            "--help" | "-h" => return Err(usage()),
            _ => return Err(format!("unknown argument `{argument}`\n{}", usage())),
        }
    }
    Ok(Args {
        clean: required_path(clean, "--clean")?,
        noisy: required_path(noisy, "--noisy")?,
        dpdfnet_model: required_path(dpdfnet_model, "--dpdfnet-model")?,
        gtcrn_model: required_path(gtcrn_model, "--gtcrn-model")?,
        output_dir,
        json,
        runs,
    })
}

fn required_path(value: Option<PathBuf>, flag: &str) -> Result<PathBuf, String> {
    let path = value.ok_or_else(|| format!("missing required {flag}\n{}", usage()))?;
    if !path.is_file() {
        return Err(format!("{flag} is not a file: {}", path.display()));
    }
    Ok(path)
}

fn usage() -> String {
    format!(
        "usage: dpdfnet_gtcrn_poc --clean CLEAN.wav --noisy NOISY.wav \\\n  --dpdfnet-model dpdfnet2_48khz_hr.onnx --gtcrn-model gtcrn_simple.onnx \\\n  [--runs {DEFAULT_RUNS}] [--output-dir DIR] [--json RESULT.json]"
    )
}
