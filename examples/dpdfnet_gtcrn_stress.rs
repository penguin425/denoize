//! Sustained and concurrent deadline probe for the DPDFNet issue PoC.

use denoize::backend::{dpdfnet, gtcrn};
use denoize::{
    BackendOptions, ChannelMode, DenoiserConfig, DpdfnetModel, GtcrnModel, OnnxModelConfig,
    StreamingBackendSession,
};
use serde_json::{json, Value};
use sha2::{Digest as _, Sha256};
use std::env;
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

const DAW_SAMPLE_RATE: u32 = 48_000;
const DAW_BLOCK_FRAMES: usize = 480;
const WARMUP_CALLS: usize = 100;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelKind {
    Dpdfnet2,
    Dpdfnet8,
    DpdfnetDaw,
    Gtcrn,
    GtcrnDaw,
}

impl ModelKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "dpdfnet2" => Ok(Self::Dpdfnet2),
            "dpdfnet8" => Ok(Self::Dpdfnet8),
            "dpdfnet-daw" => Ok(Self::DpdfnetDaw),
            "gtcrn" => Ok(Self::Gtcrn),
            "gtcrn-daw" => Ok(Self::GtcrnDaw),
            _ => Err(format!("unknown model `{value}`")),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Dpdfnet2 => "dpdfnet2_48khz_hr",
            Self::Dpdfnet8 => "dpdfnet8_48khz_hr",
            Self::DpdfnetDaw => "dpdfnet2_48khz_stereo_linked_daw_path",
            Self::Gtcrn => "gtcrn_native_hop",
            Self::GtcrnDaw => "gtcrn_48khz_stereo_linked_daw_path",
        }
    }
}

#[derive(Debug)]
struct Args {
    kind: ModelKind,
    model_path: PathBuf,
    seconds: usize,
    parallel: usize,
    realtime_paced: bool,
    json: PathBuf,
}

#[derive(Debug)]
struct ThreadResult {
    durations_ms: Vec<f64>,
    audio_seconds: f64,
    wall_seconds: f64,
    checksum: f64,
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
    let rss_before_load = current_rss_bytes();
    let load_started = Instant::now();
    match args.kind {
        ModelKind::Dpdfnet2 | ModelKind::Dpdfnet8 | ModelKind::DpdfnetDaw => {
            let model = DpdfnetModel::load(&OnnxModelConfig {
                path: args.model_path.clone(),
                sample_rate: dpdfnet::SAMPLE_RATE,
            })?;
            let expected = if args.kind == ModelKind::Dpdfnet8 {
                dpdfnet::DPDFNET8_STATE_SIZE
            } else {
                dpdfnet::DPDFNET2_STATE_SIZE
            };
            if model.metadata().state_size != expected {
                return Err(format!(
                    "{} has {} state scalars, expected {expected}",
                    args.kind.name(),
                    model.metadata().state_size
                ));
            }
            let load_ms = milliseconds(load_started.elapsed());
            let production_contract = args.kind != ModelKind::Dpdfnet8;
            let robustness = dpdfnet_robustness(&model, production_contract)?;
            let (threads, deadline_semantics) = if args.kind == ModelKind::DpdfnetDaw {
                (
                    run_dpdfnet_daw_threads(
                        model,
                        args.model_path.clone(),
                        args.seconds,
                        args.parallel,
                        args.realtime_paced,
                    )?,
                    if args.realtime_paced {
                        "one real-time-paced 48-kHz stereo-linked 480-frame host block through the production arbitrary-block adapter"
                    } else {
                        "one unpaced 48-kHz stereo-linked 480-frame host block through the production arbitrary-block adapter"
                    },
                )
            } else {
                if args.realtime_paced {
                    return Err("--realtime-paced requires --model dpdfnet-daw".into());
                }
                (
                    run_dpdfnet_threads(model, args.seconds, args.parallel)?,
                    "one native 480-sample/10-ms inference hop",
                )
            };
            return finish(
                &args,
                threads,
                robustness,
                Some(expected),
                10.0,
                deadline_semantics,
                load_ms,
                rss_before_load,
            );
        }
        ModelKind::Gtcrn | ModelKind::GtcrnDaw => {
            let model = GtcrnModel::load(&OnnxModelConfig {
                path: args.model_path.clone(),
                sample_rate: gtcrn::SAMPLE_RATE,
            })?;
            let load_ms = milliseconds(load_started.elapsed());
            let robustness = gtcrn_robustness(&model)?;
            if args.kind == ModelKind::Gtcrn {
                let threads = run_gtcrn_threads(model, args.seconds, args.parallel)?;
                return finish(
                    &args,
                    threads,
                    robustness,
                    None,
                    16.0,
                    "one native 256-sample/16-ms inference hop",
                    load_ms,
                    rss_before_load,
                );
            }
            let threads =
                run_gtcrn_daw_threads(model, args.model_path.clone(), args.seconds, args.parallel)?;
            return finish(
                &args,
                threads,
                robustness,
                None,
                10.0,
                "one 48-kHz stereo-linked host block; individual calls alternate between buffering and inference, while the released worker has a separate 240-ms queue deadline",
                load_ms,
                rss_before_load,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn finish(
    args: &Args,
    threads: Vec<ThreadResult>,
    robustness: Value,
    state_size: Option<usize>,
    budget_ms: f64,
    deadline_semantics: &str,
    load_ms: f64,
    rss_before_load: Option<u64>,
) -> Result<(), String> {
    let rss_after_run = current_rss_bytes();
    let peak_rss = peak_rss_bytes();
    let mut durations: Vec<f64> = threads
        .iter()
        .flat_map(|thread| thread.durations_ms.iter().copied())
        .collect();
    durations.sort_by(f64::total_cmp);
    if durations.is_empty() {
        return Err("stress run produced no timing samples".into());
    }
    let sum = durations.iter().sum::<f64>();
    let mean = sum / durations.len() as f64;
    let variance = durations
        .iter()
        .map(|duration| (duration - mean).powi(2))
        .sum::<f64>()
        / durations.len() as f64;
    let total_audio_seconds = threads
        .iter()
        .map(|thread| thread.audio_seconds)
        .sum::<f64>();
    let total_processing_seconds = sum / 1_000.0;
    let concurrent_wall_seconds = threads
        .iter()
        .map(|thread| thread.wall_seconds)
        .fold(0.0, f64::max);
    let result = json!({
        "schema": "denoize-dpdfnet-gtcrn-stress-v1",
        "model": args.kind.name(),
        "model_path": args.model_path,
        "model_file_bytes": std::fs::metadata(&args.model_path).map(|value| value.len()).ok(),
        "model_file_sha256": std::fs::read(&args.model_path)
            .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
            .map_err(|error| format!("hash model {}: {error}", args.model_path.display()))?,
        "state_size": state_size,
        "parallel_streams": args.parallel,
        "requested_seconds_per_stream": args.seconds,
        "realtime_paced": args.realtime_paced,
        "measured_audio_seconds": total_audio_seconds,
        "calls": durations.len(),
        "load_ms": load_ms,
        "timing": {
            "mean_ms": mean,
            "standard_deviation_ms": variance.sqrt(),
            "p50_ms": percentile(&durations, 0.50),
            "p95_ms": percentile(&durations, 0.95),
            "p99_ms": percentile(&durations, 0.99),
            "p99_9_ms": percentile(&durations, 0.999),
            "maximum_ms": durations[durations.len() - 1],
            "budget_ms": budget_ms,
            "deadline_semantics": deadline_semantics,
            "calls_over_budget": durations.iter().filter(|duration| **duration > budget_ms).count(),
            "calls_over_budget_fraction": durations.iter().filter(|duration| **duration > budget_ms).count() as f64 / durations.len() as f64,
            "summed_compute_rtf": total_processing_seconds / total_audio_seconds,
            "concurrent_wall_seconds": concurrent_wall_seconds,
            "aggregate_realtime_throughput_x": total_audio_seconds / concurrent_wall_seconds.max(1.0e-20),
        },
        "memory": {
            "rss_before_model_load_bytes": rss_before_load,
            "rss_after_run_bytes": rss_after_run,
            "peak_rss_bytes": peak_rss,
            "rss_growth_bytes": optional_difference(rss_after_run, rss_before_load),
        },
        "output_checksum": threads.iter().map(|thread| thread.checksum).sum::<f64>(),
        "robustness": robustness,
        "environment": {
            "os": env::consts::OS,
            "arch": env::consts::ARCH,
            "logical_parallelism": std::thread::available_parallelism().map(|value| value.get()).ok(),
            "source_commit": env::var("DENOIZE_EVIDENCE_SOURCE_COMMIT").ok(),
            "target": env::var("DENOIZE_EVIDENCE_TARGET").ok(),
            "os_version": env::var("DENOIZE_EVIDENCE_OS_VERSION").ok(),
            "cpu_model": env::var("DENOIZE_EVIDENCE_CPU_MODEL").ok(),
            "hardware_tier": env::var("DENOIZE_EVIDENCE_HARDWARE_TIER").ok(),
            "runner_label": env::var("DENOIZE_EVIDENCE_RUNNER_LABEL").ok(),
        },
    });
    let bytes = serde_json::to_vec_pretty(&result)
        .map_err(|error| format!("encode stress result: {error}"))?;
    std::fs::write(&args.json, bytes)
        .map_err(|error| format!("write stress result {}: {error}", args.json.display()))?;
    println!("stress JSON result: {}", args.json.display());
    Ok(())
}

fn run_dpdfnet_threads(
    model: DpdfnetModel,
    seconds: usize,
    parallel: usize,
) -> Result<Vec<ThreadResult>, String> {
    let calls = seconds
        .checked_mul(100)
        .ok_or_else(|| "DPDFNet stress duration overflow".to_string())?;
    run_parallel(parallel, move |thread_index, barrier| {
        let mut stream = model.stream()?;
        let inputs = dpdfnet_inputs(thread_index);
        for index in 0..WARMUP_CALLS + 1 {
            stream.process_hop(&inputs[index % inputs.len()])?;
        }
        stream.reset();
        stream.process_hop(&inputs[0])?;
        barrier.wait();
        let wall_started = Instant::now();
        let mut durations_ms = Vec::with_capacity(calls);
        let mut checksum = 0.0;
        for index in 0..calls {
            let started = Instant::now();
            let output = stream
                .process_hop(&inputs[(index + 1) % inputs.len()])?
                .ok_or_else(|| "primed DPDFNet stream withheld a hop".to_string())?;
            durations_ms.push(milliseconds(started.elapsed()));
            checksum += output[index % output.len()] as f64;
            if output.iter().any(|sample| !sample.is_finite()) {
                return Err("DPDFNet produced a non-finite stress sample".into());
            }
        }
        Ok(ThreadResult {
            durations_ms,
            audio_seconds: calls as f64 * dpdfnet::HOP_SIZE as f64 / dpdfnet::SAMPLE_RATE as f64,
            wall_seconds: wall_started.elapsed().as_secs_f64(),
            checksum,
        })
    })
}

fn run_gtcrn_threads(
    model: GtcrnModel,
    seconds: usize,
    parallel: usize,
) -> Result<Vec<ThreadResult>, String> {
    let calls = seconds
        .checked_mul(gtcrn::SAMPLE_RATE as usize)
        .and_then(|samples| samples.checked_div(gtcrn::HOP_SIZE))
        .ok_or_else(|| "GTCRN stress duration overflow".to_string())?;
    run_parallel(parallel, move |thread_index, barrier| {
        let mut stream = model.stream()?;
        let inputs = gtcrn_inputs(thread_index);
        for index in 0..WARMUP_CALLS {
            stream.process_hop(&inputs[index % inputs.len()])?;
        }
        stream.reset();
        barrier.wait();
        let wall_started = Instant::now();
        let mut durations_ms = Vec::with_capacity(calls);
        let mut checksum = 0.0;
        for index in 0..calls {
            let started = Instant::now();
            let output = stream.process_hop(&inputs[index % inputs.len()])?;
            durations_ms.push(milliseconds(started.elapsed()));
            checksum += output[index % output.len()] as f64;
            if output.iter().any(|sample| !sample.is_finite()) {
                return Err("GTCRN produced a non-finite stress sample".into());
            }
        }
        Ok(ThreadResult {
            durations_ms,
            audio_seconds: calls as f64 * gtcrn::HOP_SIZE as f64 / gtcrn::SAMPLE_RATE as f64,
            wall_seconds: wall_started.elapsed().as_secs_f64(),
            checksum,
        })
    })
}

fn run_gtcrn_daw_threads(
    model: GtcrnModel,
    model_path: PathBuf,
    seconds: usize,
    parallel: usize,
) -> Result<Vec<ThreadResult>, String> {
    let calls = seconds
        .checked_mul(100)
        .ok_or_else(|| "GTCRN DAW stress duration overflow".to_string())?;
    run_parallel(parallel, move |thread_index, barrier| {
        let options = BackendOptions {
            onnx: Some(OnnxModelConfig {
                path: model_path.clone(),
                sample_rate: gtcrn::SAMPLE_RATE,
            }),
            deterministic: true,
            channel_mode: ChannelMode::StereoLinked,
            ..BackendOptions::default()
        };
        let mut denoiser = DenoiserConfig::default(DAW_SAMPLE_RATE);
        denoiser.vad = false;
        let mut stream = StreamingBackendSession::new_gtcrn_for_daw_with_prepared_model(
            DAW_SAMPLE_RATE,
            2,
            denoiser,
            options,
            &model,
        )?;
        let input = daw_input(thread_index);
        for _ in 0..WARMUP_CALLS {
            stream.process_block(&input)?;
        }
        stream.reset()?;
        barrier.wait();
        let wall_started = Instant::now();
        let mut durations_ms = Vec::with_capacity(calls);
        let mut checksum = 0.0;
        for index in 0..calls {
            let started = Instant::now();
            let output = stream.process_block(&input)?;
            durations_ms.push(milliseconds(started.elapsed()));
            checksum += output
                .first()
                .and_then(|channel| channel.get(index % channel.len().max(1)))
                .copied()
                .unwrap_or(0.0);
            if output.iter().flatten().any(|sample| !sample.is_finite()) {
                return Err("GTCRN DAW path produced a non-finite stress sample".into());
            }
        }
        Ok(ThreadResult {
            durations_ms,
            audio_seconds: calls as f64 * DAW_BLOCK_FRAMES as f64 / DAW_SAMPLE_RATE as f64,
            wall_seconds: wall_started.elapsed().as_secs_f64(),
            checksum,
        })
    })
}

fn run_dpdfnet_daw_threads(
    model: DpdfnetModel,
    model_path: PathBuf,
    seconds: usize,
    parallel: usize,
    realtime_paced: bool,
) -> Result<Vec<ThreadResult>, String> {
    let calls = seconds
        .checked_mul(100)
        .ok_or_else(|| "DPDFNet DAW stress duration overflow".to_string())?;
    run_parallel(parallel, move |thread_index, barrier| {
        let options = BackendOptions {
            onnx: Some(OnnxModelConfig {
                path: model_path.clone(),
                sample_rate: dpdfnet::SAMPLE_RATE,
            }),
            deterministic: true,
            channel_mode: ChannelMode::StereoLinked,
            ..BackendOptions::default()
        };
        let mut denoiser = DenoiserConfig::default(DAW_SAMPLE_RATE);
        denoiser.vad = false;
        let mut stream = StreamingBackendSession::new_dpdfnet_for_daw_with_prepared_model(
            DAW_SAMPLE_RATE,
            2,
            denoiser,
            options,
            &model,
        )?;
        let input = daw_input(thread_index);
        for _ in 0..WARMUP_CALLS {
            stream.process_block(&input)?;
        }
        stream.reset()?;
        // Exercise direct production calls under the same platform scheduling
        // class as the released inference worker. The guard is thread-bound
        // and remains live for the complete measurement.
        let mut priority_guard = denoize::neural_daw::NeuralDawWorkerPriorityGuard::acquire()?;
        barrier.wait();
        let wall_started = Instant::now();
        let mut durations_ms = Vec::with_capacity(calls);
        let mut checksum = 0.0;
        for index in 0..calls {
            if realtime_paced {
                let due =
                    wall_started + Duration::from_micros((index as u64).saturating_mul(10_000));
                if let Some(delay) = due.checked_duration_since(Instant::now()) {
                    std::thread::sleep(delay);
                }
            }
            let started = Instant::now();
            let output = priority_guard.run_inference_cycle(|| stream.process_block(&input))?;
            durations_ms.push(milliseconds(started.elapsed()));
            checksum += output
                .first()
                .and_then(|channel| channel.get(index % channel.len().max(1)))
                .copied()
                .unwrap_or(0.0);
            if output.iter().flatten().any(|sample| !sample.is_finite()) {
                return Err("DPDFNet DAW path produced a non-finite stress sample".into());
            }
        }
        Ok(ThreadResult {
            durations_ms,
            audio_seconds: calls as f64 * DAW_BLOCK_FRAMES as f64 / DAW_SAMPLE_RATE as f64,
            wall_seconds: wall_started.elapsed().as_secs_f64(),
            checksum,
        })
    })
}

fn run_parallel<F>(parallel: usize, task: F) -> Result<Vec<ThreadResult>, String>
where
    F: Fn(usize, Arc<Barrier>) -> Result<ThreadResult, String> + Send + Sync + 'static,
{
    let barrier = Arc::new(Barrier::new(parallel));
    let task = Arc::new(task);
    let mut handles = Vec::with_capacity(parallel);
    for thread_index in 0..parallel {
        let barrier = Arc::clone(&barrier);
        let task = Arc::clone(&task);
        handles.push(std::thread::spawn(move || task(thread_index, barrier)));
    }
    handles
        .into_iter()
        .map(|handle| {
            handle
                .join()
                .map_err(|_| "stress worker panicked".to_string())?
        })
        .collect()
}

fn dpdfnet_robustness(model: &DpdfnetModel, production_contract: bool) -> Result<Value, String> {
    let inputs = dpdfnet_inputs(0);
    let first = dpdfnet_sequence(model, &inputs)?;
    let second = dpdfnet_sequence(model, &inputs)?;
    let reset_max_error = maximum_absolute_error(&first, &second);
    let process = |samples, rate| {
        if production_contract {
            model.process_aligned(&[samples], rate)
        } else {
            model.process(&[samples], rate)
        }
    };
    let geometry = finite_geometry_checks(process)?;
    let empty = if production_contract {
        model.process_aligned(&[Vec::new()], dpdfnet::SAMPLE_RATE)?
    } else {
        model.process(&[Vec::new()], dpdfnet::SAMPLE_RATE)?
    };
    Ok(json!({
        "independent_stream_max_abs_error": reset_max_error,
        "independent_stream_bit_exact": reset_max_error == 0.0,
        "finite_geometry": geometry,
        "empty_input_exact": empty == vec![Vec::<f64>::new()],
        "stream_contract": if production_contract {
            "production arbitrary-block, arbitrary-rate, exact-length stream with four-hop content alignment"
        } else {
            "fixed 480-sample native-rate comparison stream retaining the model delay"
        },
    }))
}

fn gtcrn_robustness(model: &GtcrnModel) -> Result<Value, String> {
    let inputs = gtcrn_inputs(0);
    let first = gtcrn_sequence(model, &inputs)?;
    let second = gtcrn_sequence(model, &inputs)?;
    let reset_max_error = maximum_absolute_error(&first, &second);
    let geometry = finite_geometry_checks(|samples, rate| model.process(&[samples], rate))?;
    Ok(json!({
        "independent_stream_max_abs_error": reset_max_error,
        "independent_stream_bit_exact": reset_max_error == 0.0,
        "finite_geometry": geometry,
        "empty_input_exact": model.process(&[Vec::new()], gtcrn::SAMPLE_RATE)? == vec![Vec::<f64>::new()],
    }))
}

fn finite_geometry_checks<F>(mut process: F) -> Result<Value, String>
where
    F: FnMut(Vec<f64>, u32) -> Result<Vec<Vec<f64>>, String>,
{
    let mut rates = Vec::new();
    for rate in [8_000_u32, 16_000, 44_100, 48_000, 96_000] {
        let frames = (rate as usize / 4).max(1);
        let mut samples = signal(frames, rate, 0);
        if samples.len() >= 5 {
            samples[0] = f64::NAN;
            samples[1] = f64::INFINITY;
            samples[2] = f64::NEG_INFINITY;
            samples[3] = 2.0;
            samples[4] = -2.0;
        }
        let output = process(samples, rate)?;
        let channel = output
            .first()
            .ok_or_else(|| "robustness output has no channel".to_string())?;
        rates.push(json!({
            "sample_rate": rate,
            "expected_frames": frames,
            "actual_frames": channel.len(),
            "exact_length": channel.len() == frames,
            "all_finite": channel.iter().all(|sample| sample.is_finite()),
            "peak_abs": channel.iter().fold(0.0_f64, |peak, sample| peak.max(sample.abs())),
        }));
    }
    Ok(Value::Array(rates))
}

fn dpdfnet_sequence(
    model: &DpdfnetModel,
    inputs: &[[f32; dpdfnet::HOP_SIZE]],
) -> Result<Vec<f32>, String> {
    let mut stream = model.stream()?;
    let mut output = Vec::new();
    for input in inputs.iter().take(20) {
        if let Some(hop) = stream.process_hop(input)? {
            output.extend(hop);
        }
    }
    if let Some(hop) = stream.flush()? {
        output.extend(hop);
    }
    Ok(output)
}

fn gtcrn_sequence(
    model: &GtcrnModel,
    inputs: &[[f32; gtcrn::HOP_SIZE]],
) -> Result<Vec<f32>, String> {
    let mut stream = model.stream()?;
    let mut output = Vec::new();
    for input in inputs.iter().take(20) {
        output.extend(stream.process_hop(input)?);
    }
    output.extend(stream.flush()?);
    Ok(output)
}

fn maximum_absolute_error(left: &[f32], right: &[f32]) -> f64 {
    if left.len() != right.len() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| f64::from((left - right).abs()))
        .fold(0.0, f64::max)
}

fn dpdfnet_inputs(thread_index: usize) -> Vec<[f32; dpdfnet::HOP_SIZE]> {
    (0..101)
        .map(|block| {
            let samples = signal(
                dpdfnet::HOP_SIZE,
                dpdfnet::SAMPLE_RATE,
                thread_index * 101 + block,
            );
            std::array::from_fn(|index| samples[index] as f32)
        })
        .collect()
}

fn gtcrn_inputs(thread_index: usize) -> Vec<[f32; gtcrn::HOP_SIZE]> {
    (0..101)
        .map(|block| {
            let samples = signal(
                gtcrn::HOP_SIZE,
                gtcrn::SAMPLE_RATE,
                thread_index * 101 + block,
            );
            std::array::from_fn(|index| samples[index] as f32)
        })
        .collect()
}

fn daw_input(thread_index: usize) -> Vec<Vec<f64>> {
    let left = signal(DAW_BLOCK_FRAMES, DAW_SAMPLE_RATE, thread_index);
    let right = left
        .iter()
        .enumerate()
        .map(|(index, sample)| sample * 0.9 + (index as f64 * 0.017).sin() * 0.01)
        .collect();
    vec![left, right]
}

fn signal(frames: usize, sample_rate: u32, seed: usize) -> Vec<f64> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ seed as u64;
    (0..frames)
        .map(|frame| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let noise = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f64
                / (1_u64 << 24) as f64
                - 0.5;
            let time = (frame + seed * frames) as f64 / sample_rate as f64;
            (std::f64::consts::TAU * 173.0 * time).sin() * 0.08 + noise * 0.04
        })
        .collect()
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted[index]
}

fn optional_difference(after: Option<u64>, before: Option<u64>) -> Option<u64> {
    Some(after?.saturating_sub(before?))
}

fn current_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
        let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
        kib.checked_mul(1_024)
    }
    #[cfg(target_os = "windows")]
    {
        windows_memory_counters().map(|counters| counters.WorkingSetSize as u64)
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

fn peak_rss_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: getrusage initializes the provided rusage when it succeeds.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
            return None;
        }
        // SAFETY: the successful call above initialized the value.
        let usage = unsafe { usage.assume_init() };
        #[cfg(target_os = "macos")]
        return u64::try_from(usage.ru_maxrss).ok();
        #[cfg(not(target_os = "macos"))]
        return u64::try_from(usage.ru_maxrss)
            .ok()
            .and_then(|value| value.checked_mul(1_024));
    }
    #[cfg(target_os = "windows")]
    {
        windows_memory_counters().map(|counters| counters.PeakWorkingSetSize as u64)
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        None
    }
}

#[cfg(target_os = "windows")]
fn windows_memory_counters(
) -> Option<windows_sys::Win32::System::ProcessStatus::PROCESS_MEMORY_COUNTERS> {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: u32::try_from(std::mem::size_of::<PROCESS_MEMORY_COUNTERS>()).ok()?,
        ..PROCESS_MEMORY_COUNTERS::default()
    };
    // SAFETY: the pseudo-handle is valid for the current process and the
    // writable counter buffer has the exact size supplied in `cb`.
    let succeeded =
        unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    (succeeded != 0).then_some(counters)
}

fn milliseconds(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn parse_args() -> Result<Args, String> {
    let mut arguments = env::args().skip(1);
    let mut kind = None;
    let mut model_path = None;
    let mut seconds = 60usize;
    let mut parallel = 1usize;
    let mut realtime_paced = false;
    let mut json = None;
    while let Some(argument) = arguments.next() {
        let value = |arguments: &mut std::iter::Skip<std::env::Args>| {
            arguments
                .next()
                .ok_or_else(|| format!("{argument} requires a value"))
        };
        match argument.as_str() {
            "--model" => kind = Some(ModelKind::parse(&value(&mut arguments)?)?),
            "--model-path" => model_path = Some(PathBuf::from(value(&mut arguments)?)),
            "--seconds" => {
                seconds = value(&mut arguments)?
                    .parse()
                    .map_err(|_| "--seconds must be an integer".to_string())?;
            }
            "--parallel" => {
                parallel = value(&mut arguments)?
                    .parse()
                    .map_err(|_| "--parallel must be an integer".to_string())?;
            }
            "--realtime-paced" => realtime_paced = true,
            "--json" => json = Some(PathBuf::from(value(&mut arguments)?)),
            _ => return Err(format!("unknown argument `{argument}`\n{}", usage())),
        }
    }
    if !(1..=3_600).contains(&seconds) {
        return Err("--seconds must be between 1 and 3600".into());
    }
    if !(1..=64).contains(&parallel) {
        return Err("--parallel must be between 1 and 64".into());
    }
    Ok(Args {
        kind: kind.ok_or_else(|| format!("missing --model\n{}", usage()))?,
        model_path: model_path.ok_or_else(|| format!("missing --model-path\n{}", usage()))?,
        seconds,
        parallel,
        realtime_paced,
        json: json.ok_or_else(|| format!("missing --json\n{}", usage()))?,
    })
}

fn usage() -> &'static str {
    "usage: dpdfnet_gtcrn_stress --model dpdfnet2|dpdfnet8|dpdfnet-daw|gtcrn|gtcrn-daw \\\n  --model-path MODEL.onnx [--seconds 60] [--parallel 1] [--realtime-paced] --json RESULT.json"
}
