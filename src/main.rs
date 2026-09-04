//! `denoize` command-line interface.

use denoize::audio::{
    ensure_memory_limit, estimate_audio_working_set_bytes, estimate_session_memory_bytes,
    estimate_stream_memory_bytes_checked, read_audio, read_audio_from_session_with_limits,
    read_wav_bytes_with_limits, write_wav_bytes, write_wav_channel_mask_to_file,
};
use denoize::batch_resume::{
    self, BatchSession, ConsumedModel, Digest, FileFingerprint, MetadataPolicy, ResumeDecision,
    ResumeExpectation, LEGACY_DESKTOP_STATE_FILE_NAME, LOCK_FILE_NAME, RECIPE_DOMAIN,
    RECIPE_OUTPUT_ABI_VERSION, RECIPE_VERSION, STATE_FILE_NAME,
};
use denoize::config::{MAX_SAMPLE_RATE, MAX_STREAM_BLOCK_FRAMES};
use denoize::decode::{
    inspect_audio_stream_session,
    probe_file_from_session_with_limits as probe_audio_session_with_limits, AudioCodec,
    AudioFormat, AudioProbe, AudioStreamReader, DecodeLimits,
};
use denoize::denoiser::{DenoiserConfig, Preset, ProcessingMode};
use denoize::ipc::{
    initialize_ipc_state, run_ipc_server, IpcClient, IpcGrantDocument, IpcGrantPolicy, IpcJobKind,
    IpcJobSpec, IpcLimits, IpcOperation, IpcResponseResult, IpcServerConfig,
};
use denoize::metadata::MetadataLimits;
use denoize::service::{self, BackendChoice, ProcessingOptions};
use denoize::window::MAX_DENOISER_DPSS_NW;
use denoize::AudioInputSession;
use denoize::{
    neural_daw_chunk_frames, neural_daw_latency_frames, neural_daw_latency_millis, read_daw_preset,
    read_daw_session, read_neural_daw_session, verify_stream_output_file, write_daw_preset,
    write_daw_session, write_neural_daw_session, AacEncoder, AcceleratorPreference,
    AcceleratorSelection, Algorithm, AtomicOutput, AudioStreamWriter, Backend, BackendOptions,
    BackendSession, ChannelMode, CommitMode, DawParameters, DawPortConfiguration, DawPreset,
    DawRealtimeProcessor, DawSessionState, DownmixMode, EncodeOptions, ExecutionKind,
    ExecutionPlan, ExecutionPlanItem, ExecutionReceiptPayload, NeuralDawModel,
    NeuralDawOverloadFallback, NeuralDawParameters, NeuralDawPortConfiguration,
    NeuralDawSessionState, OnnxModelConfig, OutputFormat, PlannedArtifact, PlannedOutput,
    PlannedResources, ReceiptItem, ReceiptPublicKey, ReceiptSecretKey, ReceiptTrustPolicy,
    RecommendationGoal, RecommendationOptions, ResourceGovernor, ResourceLimits, ResourcePermit,
    ResourceRequest, RuntimeModelPackage, SgmseProfile, SignedExecutionReceipt,
    SpooledAudioStreamWriter, StreamEncodeLimits, StreamEncodeSpec, StreamPcmSpool,
    StreamSpoolLimits, StreamingBackendSession, WatchCycleReport, WatchFolder, WatchFolderConfig,
    WatchFolderJob, WatchProcessError, WindowType, DAW_FIXED_LATENCY_MILLIS, DAW_LATENCY_POLICY,
    DAW_PLUGIN_ID, NEURAL_DAW_LATENCY_POLICY, NEURAL_DAW_MODEL_ID, NEURAL_DAW_MODEL_SHA256,
    NEURAL_DAW_PLUGIN_ID, NEURAL_DAW_QUEUE_BLOCKS, WATCH_CYCLE_SCHEMA,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read as _, Write as _};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use zeroize::Zeroizing;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const STREAM_BLOCK_FRAMES: usize = 8192;
const STREAM_CHECKPOINT_FRAMES: u64 = 1_048_576;
const MIN_STREAM_BLOCK_FRAMES: usize = 1;
const MIN_LIVE_CHUNK_MS: u32 = 10;
const MAX_LIVE_CHUNK_MS: u32 = 2_000;
const MIN_LIVE_TARGET_LATENCY_MS: u32 = 20;
const MAX_LIVE_TARGET_LATENCY_MS: u32 = 5_000;
const MAX_LIVE_DRIFT_PPM: u32 = 10_000;
const MAX_LIVE_RECONNECT_TIMEOUT_MS: u32 = 300_000;
const MAX_BATCH_JOBS: usize = 32;
const VALIDATION_SAMPLE_RATE: u32 = 48_000;
const BYTES_PER_MIB: u64 = 1024 * 1024;
const INPUT_MEMORY_EXPANSION_FACTOR: u64 = 8;
const STDIN_READ_CHUNK_BYTES: usize = 64 * 1024;
const CLI_JSON_SCHEMA: &str = "denoize-cli-output-v1";
const CLI_JSON_SCHEMA_VERSION: u32 = 1;
const ISOLATED_CHILD_ENV: &str = "DENOIZE_INTERNAL_ISOLATED_CHILD";
#[cfg(windows)]
const ISOLATION_GATE_ENV: &str = "DENOIZE_INTERNAL_ISOLATION_GATE";
static CANCELLED: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
thread_local! {
    static TEST_STREAM_CHECKPOINT_FRAMES: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    static TEST_STOP_AFTER_STREAM_CHECKPOINT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static TEST_STOP_AFTER_STREAM_COMMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static TEST_CORRUPT_STREAM_OUTPUT_BEFORE_VERIFY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn stream_checkpoint_frames() -> u64 {
    #[cfg(test)]
    if let Some(frames) = TEST_STREAM_CHECKPOINT_FRAMES.with(std::cell::Cell::get) {
        return frames;
    }
    STREAM_CHECKPOINT_FRAMES
}

fn injected_stop_after_stream_checkpoint() -> bool {
    #[cfg(test)]
    {
        return TEST_STOP_AFTER_STREAM_CHECKPOINT.with(|value| value.replace(false));
    }
    #[cfg(not(test))]
    false
}

fn injected_stop_after_stream_commit() -> bool {
    #[cfg(test)]
    {
        return TEST_STOP_AFTER_STREAM_COMMIT.with(|value| value.replace(false));
    }
    #[cfg(not(test))]
    false
}

fn inject_stream_output_corruption(file: &mut std::fs::File) -> Result<(), String> {
    #[cfg(test)]
    if TEST_CORRUPT_STREAM_OUTPUT_BEFORE_VERIFY.with(|value| value.replace(false)) {
        file.set_len(16)
            .map_err(|error| format!("inject staged stream output corruption: {error}"))?;
    }
    let _ = file;
    Ok(())
}
static CANCEL_HANDLER: OnceLock<Result<(), String>> = OnceLock::new();

fn with_batch_publication_fence<T>(
    fence: &Mutex<()>,
    cancelled: &AtomicBool,
    publish: impl FnOnce() -> Result<T, String>,
) -> Result<Option<T>, String> {
    let _guard = fence
        .lock()
        .map_err(|_| "batch publication fence is poisoned".to_string())?;
    if cancelled.load(Ordering::SeqCst) {
        Ok(None)
    } else {
        publish().map(Some)
    }
}

#[derive(Serialize)]
struct RecipeJson {
    domain: &'static str,
    version: u32,
    output_abi_version: u32,
    digest: Option<String>,
}

#[derive(Serialize)]
struct AcceleratorJson {
    requested: &'static str,
    effective: &'static str,
    fallback: Option<&'static str>,
}

#[cfg(feature = "live")]
#[derive(Serialize)]
struct LiveStatusJson {
    schema: &'static str,
    schema_version: u32,
    event: &'static str,
    mode: &'static str,
    state: &'static str,
    input_sample_rate: u32,
    output_sample_rate: u32,
    input_channels: usize,
    output_channels: usize,
    chunk_frames: usize,
    input_level: f32,
    output_level: f32,
    processed_chunks: u64,
    dropped_chunks: u64,
    underrun_frames: u64,
    overflow_frames: u64,
    queued_frames: usize,
    target_queue_frames: usize,
    queue_latency_ms: f64,
    processing_latency_ms: f64,
    input_device_latency_ms: f64,
    output_device_latency_ms: f64,
    estimated_total_latency_ms: f64,
    drift_correction_ppm: f64,
    reconnect_attempts: u64,
    device_generation: u64,
    accelerator: AcceleratorJson,
}

#[derive(Serialize)]
struct ProcessResultJson<'a> {
    schema: &'static str,
    schema_version: u32,
    event: &'static str,
    mode: &'static str,
    recipe: RecipeJson,
    input: &'a str,
    output: &'a str,
    backend: &'a str,
    accelerator: AcceleratorJson,
    channels: usize,
    frames: usize,
    sample_rate: u32,
    elapsed_ms: f64,
}

#[derive(Serialize)]
struct StreamResultJson<'a> {
    schema: &'static str,
    schema_version: u32,
    event: &'static str,
    mode: &'static str,
    recipe: RecipeJson,
    input: &'a str,
    output: &'a str,
    backend: &'a str,
    accelerator: AcceleratorJson,
    channels: u16,
    frames: usize,
    sample_rate: u32,
    stream: bool,
}

#[derive(Serialize)]
#[serde(tag = "event", rename_all = "lowercase")]
enum BatchJson<'a> {
    Progress {
        schema: &'static str,
        schema_version: u32,
        recipe: RecipeJson,
        status: &'a str,
        completed: usize,
        total: usize,
        elapsed_seconds: f64,
        eta_seconds: f64,
        input: &'a str,
    },
    Summary {
        schema: &'static str,
        schema_version: u32,
        recipe: RecipeJson,
        total: usize,
        succeeded: usize,
        skipped: usize,
        failed: usize,
        cancelled_count: usize,
        cancelled: bool,
        output: &'a str,
    },
}

fn serialize_json_line<T: Serialize + ?Sized>(payload: &T) -> String {
    serde_json::to_string(payload).expect("fixed CLI JSON payload must serialize")
}

fn recipe_json(digest: Option<Digest>) -> RecipeJson {
    RecipeJson {
        domain: RECIPE_DOMAIN,
        version: RECIPE_VERSION,
        output_abi_version: RECIPE_OUTPUT_ABI_VERSION,
        digest: digest.map(|value| value.as_hex()),
    }
}

fn accelerator_json(selection: AcceleratorSelection) -> AcceleratorJson {
    AcceleratorJson {
        requested: selection.requested().name(),
        effective: selection.effective().name(),
        fallback: selection.fallback().map(|fallback| fallback.name()),
    }
}

fn accelerator_description(selection: AcceleratorSelection) -> String {
    let mut description = format!(
        "{} -> {}",
        selection.requested().name(),
        selection.effective().name()
    );
    if let Some(fallback) = selection.fallback() {
        description.push_str(" (");
        description.push_str(fallback.name());
        description.push(')');
    }
    description
}

fn round_to_three_decimals(value: f64) -> f64 {
    format!("{value:.3}")
        .parse()
        .expect("formatted JSON number must parse")
}

fn process_result_json_line(
    input: &str,
    output: &str,
    backend: &str,
    accelerator: AcceleratorSelection,
    channels: usize,
    frames: usize,
    sample_rate: u32,
    elapsed_ms: f64,
    recipe: Option<Digest>,
) -> String {
    serialize_json_line(&ProcessResultJson {
        schema: CLI_JSON_SCHEMA,
        schema_version: CLI_JSON_SCHEMA_VERSION,
        event: "result",
        mode: "file",
        recipe: recipe_json(recipe),
        input,
        output,
        backend,
        accelerator: accelerator_json(accelerator),
        channels,
        frames,
        sample_rate,
        elapsed_ms: round_to_three_decimals(elapsed_ms),
    })
}

fn stream_result_json_line(
    input: &str,
    output: &str,
    backend: &str,
    accelerator: AcceleratorSelection,
    channels: u16,
    frames: usize,
    sample_rate: u32,
) -> String {
    serialize_json_line(&StreamResultJson {
        schema: CLI_JSON_SCHEMA,
        schema_version: CLI_JSON_SCHEMA_VERSION,
        event: "result",
        mode: "stream",
        recipe: recipe_json(None),
        input,
        output,
        backend,
        accelerator: accelerator_json(accelerator),
        channels,
        frames,
        sample_rate,
        stream: true,
    })
}

fn batch_progress_json_line(
    status: &str,
    completed: usize,
    total: usize,
    elapsed_seconds: f64,
    eta_seconds: f64,
    input: &str,
    recipe: Digest,
) -> String {
    serialize_json_line(&BatchJson::Progress {
        schema: CLI_JSON_SCHEMA,
        schema_version: CLI_JSON_SCHEMA_VERSION,
        recipe: recipe_json(Some(recipe)),
        status,
        completed,
        total,
        elapsed_seconds: round_to_three_decimals(elapsed_seconds),
        eta_seconds: round_to_three_decimals(eta_seconds),
        input,
    })
}

fn batch_summary_json_line(
    total: usize,
    succeeded: usize,
    skipped: usize,
    failed: usize,
    cancelled_count: usize,
    cancelled: bool,
    output: &str,
) -> String {
    serialize_json_line(&BatchJson::Summary {
        schema: CLI_JSON_SCHEMA,
        schema_version: CLI_JSON_SCHEMA_VERSION,
        recipe: recipe_json(None),
        total,
        succeeded,
        skipped,
        failed,
        cancelled_count,
        cancelled,
        output,
    })
}

fn install_cancel_handler() -> Result<(), String> {
    CANCEL_HANDLER
        .get_or_init(|| {
            ctrlc::set_handler(|| CANCELLED.store(true, Ordering::SeqCst))
                .map_err(|error| format!("install Ctrl+C handler: {error}"))
        })
        .clone()
}

fn usage() -> String {
    let backends = Backend::available_names().join("|");
    format!(
        "\
denoize {VERSION} — pure-Rust audio denoiser engineered for the world's highest sound quality

Classical DSP + optional local AI backends for files, streams, and realtime audio.
Input: WAV/BWF/RF64, AIFF, CAF, FLAC, Ogg Opus/Vorbis, MP3, M4A/ALAC, AAC (built in; no ffmpeg).
Output: WAV, FLAC, Ogg Opus, MP3, M4A, AAC.

USAGE:
    denoize <INPUT> <OUTPUT.wav|flac|opus|ogg|mp3|m4a|aac> [OPTIONS]
    denoize live [--input-device NAME] [--output-device NAME] [OPTIONS]
    denoize live --list-devices
    denoize hardware [--json|--pretty]
    denoize recommend <INPUT> [--goal balanced|quality|speed|low-memory] [OPTIONS]
    denoize diagnose <INPUT> [--analysis-seconds N] [--json|--pretty]
    denoize assess <INPUT> [--analysis-seconds N] [--json|--pretty]
    denoize assess <BEFORE> <AFTER> [--analysis-seconds N] [--json|--pretty]
    denoize restore <INPUT> [OUTPUT] [OPTIONS]
    denoize universal <INPUT> <OUTPUT> --model-package PACKAGE --model-package-key KEY [OPTIONS]
    denoize target-speaker <MIXTURE> <ENROLLMENT> <OUTPUT> --model-package PACKAGE --model-package-key KEY --promotion-evidence EVIDENCE --promotion-evidence-key KEY [OPTIONS]
    denoize target-sound <INPUT> --query QUERY.json --target TARGET.wav --residual RESIDUAL.wav --output OUTPUT.wav --report REPORT.json --mode preserve|remove --model-package PACKAGE --model-package-key KEY --promotion-evidence EVIDENCE --promotion-evidence-key KEY [OPTIONS]
    denoize target-sound causal <INPUT> --query QUERY.json --target TARGET.wav --residual RESIDUAL.wav --output OUTPUT.wav --report REPORT.json --mode preserve|remove --model-package PACKAGE --model-package-key KEY --offline-promotion-evidence EVIDENCE --offline-promotion-evidence-key KEY --causal-promotion-evidence EVIDENCE --causal-promotion-evidence-key KEY [OPTIONS]
    denoize meeting-speakers <MEETING> <OUTPUT.wav> --model-package PACKAGE --model-package-key KEY --promotion-evidence EVIDENCE --promotion-evidence-key KEY [OPTIONS]
    denoize music-restore <PROGRAM> <CANDIDATE.wav> --correction CORRECTION.wav --report REPORT.json --task TASK --model-package PACKAGE --model-package-key KEY --promotion-evidence EVIDENCE --promotion-evidence-key KEY [OPTIONS]
    denoize aec <MICROPHONE> <FAR_END_REFERENCE> <OUTPUT> --promotion-evidence EVIDENCE --promotion-evidence-key KEY [OPTIONS]
    denoize array <MICROPHONE_ARRAY> <OUTPUT> --array-config CONFIG --promotion-evidence EVIDENCE --promotion-evidence-key KEY [OPTIONS]
    denoize plan <INPUT> <OUTPUT> [OPTIONS] [--pretty]
    denoize watch <INPUT_DIR> <OUTPUT_DIR> [OPTIONS]  (run `denoize watch --help`)
    denoize receipts <COMMAND> [OPTIONS]  (run `denoize receipts --help`)
    denoize models <COMMAND> [MODEL|all] [OPTIONS]  (run `denoize models --help`)
    denoize evaluate <COMMAND> [OPTIONS]  (run `denoize evaluate --help`)
    denoize metrics <REFERENCE> <TEST> [--json|--markdown]
    denoize compare <CLEAN> <NOISY> <ENHANCED> [--json|--html]
    denoize plugin <COMMAND> [OPTIONS]  (run `denoize plugin --help`)
    denoize ipc <COMMAND> [OPTIONS]  (run `denoize ipc --help`)
    denoize update <COMMAND> [OPTIONS]  (run `denoize update --help`)
    denoize project <COMMAND> [OPTIONS]  (run `denoize project --help`)
    denoize sdk <COMMAND> [OPTIONS]  (run `denoize sdk --help`)

LIVE:
    Low-latency live processing supports classical, rnnoise, gtcrn, and
    dpdfnet when compiled; other backends are rejected before capture or
    playback starts.

OPTIONS:
        --config <PATH>      load TOML defaults (CLI options take precedence)
    -b, --backend <NAME>     auto|{backends}  (default: classical)
    -a, --algorithm <NAME>   omlsa|logmmse|mmse|wiener|specsub|specsub-nl|specsub-geo
    -p, --preset <NAME>      speech|music|aggressive|gentle|restore|hifi
        --mode <NAME>        speech|music|ambient processing intent
    -s, --strength <0..1>    denoising strength (default: 0.6)
        --profile <MS>       finite duration: <0 off, 0 auto, >0 up to 60000
        --no-profile         no profiling; rely on blind IMCRA bootstrap
        --no-adapt           freeze the noise estimate
        --adaptive-noise     learn noise from noise-only regions throughout the file
        --vad                speech-aware segmentation and silence suppression
        --frame <N>          FFT size: power of two in 256..65536 (default: 2048)
        --overlap <F>        overlap ratio 0.5..0.95 (default: 0.75)
        --window <NAME>      hann|hamming|sine|blackman|kaiser|flattop|dpss
        --kaiser-beta <B>    finite Kaiser beta in 0..50 (default: 8.0)
        --dpss-nw <NW>       classical DPSS time-bandwidth product in (0, {MAX_DENOISER_DPSS_NW}] (default: 3.0)
        --multiband          enable multiband spectral subtraction
        --perceptual         enable Bark-scale perceptual gain weighting
        --postfilter         enable musical-noise suppression post-filter
        --smoothing <0..1>   gain release smoothing (default: 0.6)
        --makeup <DB>        makeup gain in -120..120 dB (default: 0.0)
        --no-dc-block        disable DC-blocking pre-filter
        --quality <LEVEL>    high|ultra
        --no-transient       disable transient/onset protection
        --cepstral           enable cepstral gain smoothing
        --no-cepstral        disable cepstral smoothing
        --pre-emphasis       enable pre/de-emphasis
        --no-pre-emphasis    disable pre-emphasis
        --report             print settings report and exit
        --mp3-bitrate <KBPS> MP3 CBR bitrate (default: 192)
        --m4a-bitrate <KBPS> positive M4A/AAC CBR bitrate (default: 192)
        --aac-encoder <NAME> oxide|fdk (default: oxide)
        --downmix <MODE>     preserve|stereo (default: preserve; lossy outputs reject surround unless explicit)
        --loudness <LUFS>     finite normalization target in -70..0 LUFS
        --true-peak <DBTP>    finite ceiling in -20..0 dBTP with --loudness (default: -1)
        --onnx-model <PATH>   waveform ONNX model (required for -b onnx)
        --onnx-rate <HZ>      model sample rate in 1..768000 Hz (default: 16000)
        --model-package <PATH> signed runtime package (.dmp; -b onnx or bsrnn)
        --model-package-key <PATH> trusted Minisign public key for --model-package
        --channels <MODE>     independent|linked|mid-side (default: independent)
        --sgmse-profile <P>   fast|balanced|quality (default: balanced)
        --accelerator <NAME>  cpu|auto|gpu|metal|cuda (default: cpu)
        --deterministic       serialize processing for reproducible audio output
        --seed <N>            SGMSE sampler seed (implies --deterministic)
        --batch               process files in INPUT directory into OUTPUT directory
        --stream              bounded WAV/FLAC/Vorbis/Opus/MP3/ADTS-AAC/M4A-to-WAV processing
        --stream-frames <N>   block size in 1..1048576 frames (default: 8192)
        --max-memory <MB>     per-input denoize allocation/metadata cap in MiB (regular files; min: 1)
        --max-process-memory <MB> aggregate denoize RAM reservations across workers (min: 1)
        --max-temp-space <MB> aggregate staged-output reservation in MiB (min: 1)
        --max-gpu-memory <MB> aggregate conservative GPU reservation in MiB (min: 1)
        --max-gpu-jobs <N>    concurrent GPU workers in 1..32 (default: 1)
        --isolate             run processing in a resource-isolated child process
        --recursive           include subdirectories in batch mode
        --jobs <N>            workers in 1..32 (default: min(CPU count, 32))
        --output-format <EXT> convert all batch outputs (required when source codec cannot be preserved)
        --force               allow replacing existing output files
        --resume              resume a stream checkpoint or verify exact v3 batch outputs
        --receipt <PATH>      publish a signed execution receipt after finite output succeeds
        --receipt-key <PATH>  owner-only Ed25519 key used with --receipt
        --plan <PATH>         require exact correspondence to a read-only execution plan
        --no-progress         suppress batch progress and ETA output
        --json                emit a machine-readable result
        --no-metadata         do not copy input tags/artwork/chapters to the output
        --input-device <NAME> live capture device (default: system default)
        --output-device <NAME> live playback device (default: system default)
        --chunk-ms <MS>       live chunk duration in 10..2000 ms (default: 100)
        --live-latency <MS>   playback target: 0 auto or 20..5000 ms (default: auto)
        --max-drift-ppm <N>   clock correction in 0..10000 ppm (default: 2500)
        --reconnect-timeout <MS> hotplug recovery window in 0..300000 ms (default: 30000)
    -h, --help               show this help
    -V, --version            show version

BACKENDS (build with --features full for all):
    classical   Enhanced STFT/IMCRA/OMLSA pipeline (default)
    rnnoise     RNNoise via nnnoiseless (requires --features rnnoise)
    deepfilter  DeepFilterNet v3 for files and --stream (requires --features deepfilter)
    onnx        External waveform ONNX model (requires --features onnx)
    mpsenet     MP-SENet magnitude/phase model (requires --features mpsenet)
    bsrnn       ESPnet BSRNN spectral model (requires --features bsrnn)
    mossformer2 ClearerVoice MossFormer2 for files and --stream (requires --features mossformer2)
    sgmse       SGMSE+ diffusion model (requires --features sgmse)
    gtcrn       Official causal GTCRN for files, --stream, and live processing
    dpdfnet     Official DPDFNet-2 48 kHz HR for files, --stream, and live

PRESETS:
    hifi        Flagship transparency: OMLSA + protections + advanced DSP
    speech      Voice-optimised balance
    music       Instruments; enables perceptual + postfilter

CONFIGURATION:
    TOML syntax and enum names are checked when loaded. CLI values then override
    TOML numeric defaults, and the final effective configuration is validated
    before audio decoding, output staging, or batch worker creation.
"
    )
}

#[derive(Clone, Debug, Default)]
struct Overrides {
    backend: Option<Backend>,
    auto_backend: bool,
    algorithm: Option<Algorithm>,
    preset: Option<Preset>,
    mode: Option<ProcessingMode>,
    strength: Option<f64>,
    profile_ms: Option<f64>,
    no_profile: bool,
    no_adapt: bool,
    adaptive_noise: Option<bool>,
    vad: Option<bool>,
    frame_size: Option<usize>,
    overlap: Option<f64>,
    window: Option<WindowType>,
    kaiser_beta: Option<f64>,
    dpss_nw: Option<f64>,
    multiband: bool,
    perceptual: bool,
    postfilter: bool,
    smoothing: Option<f64>,
    makeup: Option<f64>,
    no_dc_block: bool,
    report: bool,
    quality: Option<String>,
    no_transient: bool,
    cepstral: bool,
    no_cepstral: bool,
    pre_emphasis: bool,
    no_pre_emphasis: bool,
    mp3_bitrate_kbps: Option<u32>,
    m4a_bitrate_kbps: Option<u32>,
    aac_encoder: Option<AacEncoder>,
    downmix: Option<DownmixMode>,
    loudness_lufs: Option<f64>,
    true_peak_dbtp: Option<f64>,
    onnx_model: Option<String>,
    onnx_sample_rate: Option<u32>,
    model_package: Option<String>,
    model_package_key: Option<String>,
    channel_mode: Option<ChannelMode>,
    sgmse_profile: Option<SgmseProfile>,
    accelerator: Option<AcceleratorPreference>,
    deterministic: bool,
    seed: Option<u64>,
    batch: bool,
    stream: bool,
    stream_frames: Option<usize>,
    max_memory_mb: Option<usize>,
    max_process_memory_mb: Option<usize>,
    max_temporary_mb: Option<usize>,
    max_gpu_memory_mb: Option<usize>,
    max_gpu_jobs: Option<usize>,
    isolate: bool,
    recursive: bool,
    jobs: Option<usize>,
    output_format: Option<String>,
    force: bool,
    resume: bool,
    receipt: Option<String>,
    receipt_key: Option<String>,
    execution_plan: Option<String>,
    no_progress: bool,
    json: bool,
    no_metadata: bool,
    input_device: Option<String>,
    output_device: Option<String>,
    chunk_ms: Option<u32>,
    live_latency_ms: Option<u32>,
    max_drift_ppm: Option<u32>,
    reconnect_timeout_ms: Option<u32>,
    list_devices: bool,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileConfig {
    backend: Option<String>,
    algorithm: Option<String>,
    preset: Option<String>,
    mode: Option<String>,
    strength: Option<f64>,
    profile_ms: Option<f64>,
    adaptive_noise: Option<bool>,
    vad: Option<bool>,
    frame_size: Option<usize>,
    overlap: Option<f64>,
    window: Option<String>,
    kaiser_beta: Option<f64>,
    dpss_nw: Option<f64>,
    smoothing: Option<f64>,
    makeup_db: Option<f64>,
    quality: Option<String>,
    mp3_bitrate_kbps: Option<u32>,
    m4a_bitrate_kbps: Option<u32>,
    aac_encoder: Option<String>,
    loudness_lufs: Option<f64>,
    true_peak_dbtp: Option<f64>,
    onnx_model: Option<String>,
    onnx_rate: Option<u32>,
    model_package: Option<String>,
    model_package_key: Option<String>,
    channels: Option<String>,
    sgmse_profile: Option<String>,
    accelerator: Option<String>,
    downmix: Option<String>,
    deterministic: bool,
    seed: Option<u64>,
    batch: bool,
    stream: bool,
    stream_frames: Option<usize>,
    max_memory_mb: Option<usize>,
    max_process_memory_mb: Option<usize>,
    max_temporary_mb: Option<usize>,
    max_gpu_memory_mb: Option<usize>,
    max_gpu_jobs: Option<usize>,
    isolate: bool,
    recursive: bool,
    jobs: Option<usize>,
    output_format: Option<String>,
    force: bool,
    resume: bool,
    progress: Option<bool>,
    preserve_metadata: Option<bool>,
    chunk_ms: Option<u32>,
    live_latency_ms: Option<u32>,
    max_drift_ppm: Option<u32>,
    reconnect_timeout_ms: Option<u32>,
}

fn load_config(path: &str) -> Result<Overrides, String> {
    let source = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read config {path}: {error}"))?;
    parse_config(&source, path)
}

fn parse_quality(value: &str, source: &str) -> Result<String, String> {
    match value.to_ascii_lowercase().as_str() {
        "high" => Ok("high".into()),
        // Preserve the long-standing aliases while exposing one canonical
        // effective value to backend selection and the quality preset logic.
        "ultra" | "max" | "highest" => Ok("ultra".into()),
        _ => Err(format!(
            "unknown quality{source}: {value} (expected high or ultra)"
        )),
    }
}

fn parse_config(source: &str, path: &str) -> Result<Overrides, String> {
    let config: FileConfig =
        toml::from_str(source).map_err(|error| format!("invalid config {path}: {error}"))?;
    let mut ov = Overrides::default();
    if let Some(name) = config.backend {
        if name.eq_ignore_ascii_case("auto") {
            ov.auto_backend = true;
        } else {
            ov.backend = Some(
                Backend::parse(&name)
                    .ok_or_else(|| format!("unknown backend in config: {name}"))?,
            );
        }
    }
    if let Some(name) = config.algorithm {
        ov.algorithm = Some(
            Algorithm::parse(&name)
                .ok_or_else(|| format!("unknown algorithm in config: {name}"))?,
        );
    }
    if let Some(name) = config.preset {
        ov.preset =
            Some(Preset::parse(&name).ok_or_else(|| format!("unknown preset in config: {name}"))?);
    }
    if let Some(name) = config.mode {
        ov.mode = Some(
            ProcessingMode::parse(&name)
                .ok_or_else(|| format!("unknown mode in config: {name}"))?,
        );
    }
    if let Some(name) = config.window {
        ov.window = Some(
            WindowType::parse(&name).ok_or_else(|| format!("unknown window in config: {name}"))?,
        );
    }
    if let Some(name) = config.channels {
        ov.channel_mode = Some(
            ChannelMode::parse(&name)
                .ok_or_else(|| format!("unknown channel mode in config: {name}"))?,
        );
    }
    if let Some(name) = config.downmix {
        ov.downmix = Some(DownmixMode::parse(&name).ok_or_else(|| {
            format!("unknown downmix mode in config: {name} (expected preserve or stereo)")
        })?);
    }
    if let Some(name) = config.aac_encoder {
        ov.aac_encoder = Some(AacEncoder::parse(&name).ok_or_else(|| {
            format!("unknown AAC encoder in config: {name} (expected oxide or fdk)")
        })?);
    }
    if let Some(profile) = config.sgmse_profile {
        ov.sgmse_profile = Some(SgmseProfile::parse(&profile).ok_or_else(|| {
            format!(
                "unknown SGMSE profile in config: {profile} (expected fast, balanced, or quality)"
            )
        })?);
    }
    if let Some(accelerator) = config.accelerator {
        ov.accelerator = Some(AcceleratorPreference::parse(&accelerator).ok_or_else(|| {
            format!(
                "unknown accelerator in config: {accelerator} (expected cpu, auto, gpu, metal, or cuda)"
            )
        })?);
    }
    ov.strength = config.strength;
    ov.profile_ms = config.profile_ms;
    ov.adaptive_noise = config.adaptive_noise;
    ov.vad = config.vad;
    ov.frame_size = config.frame_size;
    ov.overlap = config.overlap;
    ov.kaiser_beta = config.kaiser_beta;
    ov.dpss_nw = config.dpss_nw;
    ov.smoothing = config.smoothing;
    ov.makeup = config.makeup_db;
    ov.quality = config
        .quality
        .map(|value| parse_quality(&value, " in config"))
        .transpose()?;
    ov.mp3_bitrate_kbps = config.mp3_bitrate_kbps;
    ov.m4a_bitrate_kbps = config.m4a_bitrate_kbps;
    ov.loudness_lufs = config.loudness_lufs;
    ov.true_peak_dbtp = if config.loudness_lufs.is_none() && config.true_peak_dbtp == Some(-1.0) {
        None
    } else {
        config.true_peak_dbtp
    };
    ov.onnx_model = config.onnx_model;
    ov.onnx_sample_rate = config.onnx_rate;
    ov.model_package = config.model_package;
    ov.model_package_key = config.model_package_key;
    ov.deterministic = config.deterministic;
    ov.seed = config.seed;
    if ov.seed.is_some() {
        ov.deterministic = true;
    }
    ov.batch = config.batch;
    ov.stream = config.stream;
    ov.stream_frames = config.stream_frames;
    ov.max_memory_mb = config.max_memory_mb;
    ov.max_process_memory_mb = config.max_process_memory_mb;
    ov.max_temporary_mb = config.max_temporary_mb;
    ov.max_gpu_memory_mb = config.max_gpu_memory_mb;
    ov.max_gpu_jobs = config.max_gpu_jobs;
    ov.isolate = config.isolate;
    ov.recursive = config.recursive;
    ov.jobs = config.jobs;
    ov.output_format = config
        .output_format
        .map(|value| {
            normalize_output_extension(&value)
                .map(|extension| extension.to_ascii_lowercase())
                .map_err(|error| format!("{error} in config"))
        })
        .transpose()?;
    ov.force = config.force;
    ov.resume = config.resume;
    ov.no_progress = config.progress == Some(false);
    ov.no_metadata = config.preserve_metadata == Some(false);
    ov.chunk_ms = config.chunk_ms;
    ov.live_latency_ms = config.live_latency_ms;
    ov.max_drift_ppm = config.max_drift_ppm;
    ov.reconnect_timeout_ms = config.reconnect_timeout_ms;
    Ok(ov)
}

fn parse_value<T>(args: &[String], i: &mut usize, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    *i += 1;
    if *i >= args.len() {
        return Err(format!("missing value for {flag}"));
    }
    args[*i]
        .parse::<T>()
        .map_err(|e| format!("invalid value for {flag}: {e}"))
}

fn parse_args(args: &[String]) -> Result<(String, String, Overrides), String> {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let config_path = args
        .windows(2)
        .find(|pair| pair[0] == "--config")
        .map(|pair| pair[1].as_str());
    if args.last().map(String::as_str) == Some("--config") {
        return Err("missing value for --config".into());
    }
    let mut ov = match config_path {
        Some(path) => load_config(path)?,
        None => Overrides::default(),
    };
    let mut cli_raw_model = false;
    let mut cli_runtime_package = false;

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--config" => {
                let _: String = parse_value(args, &mut i, a)?;
            }
            "-h" | "--help" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!("denoize {VERSION}");
                std::process::exit(0);
            }
            "-b" | "--backend" => {
                let name: String = parse_value(args, &mut i, a)?;
                if name.eq_ignore_ascii_case("auto") {
                    ov.auto_backend = true;
                    ov.backend = None;
                    i += 1;
                    continue;
                }
                ov.auto_backend = false;
                ov.backend = Some(Backend::parse(&name).ok_or_else(|| {
                    format!(
                        "unknown backend: {name} (available: {:?})",
                        Backend::available_names()
                    )
                })?);
            }
            "-a" | "--algorithm" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.algorithm = Some(
                    Algorithm::parse(&name).ok_or_else(|| format!("unknown algorithm: {name}"))?,
                );
            }
            "-p" | "--preset" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.preset =
                    Some(Preset::parse(&name).ok_or_else(|| format!("unknown preset: {name}"))?);
            }
            "--mode" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.mode = Some(ProcessingMode::parse(&name).ok_or_else(|| {
                    format!("unknown mode: {name} (expected speech, music, or ambient)")
                })?);
            }
            "-s" | "--strength" => ov.strength = Some(parse_value(args, &mut i, a)?),
            "--profile" => ov.profile_ms = Some(parse_value(args, &mut i, a)?),
            "--no-profile" => ov.no_profile = true,
            "--no-adapt" => ov.no_adapt = true,
            "--adaptive-noise" => ov.adaptive_noise = Some(true),
            "--vad" => ov.vad = Some(true),
            "--frame" => ov.frame_size = Some(parse_value(args, &mut i, a)?),
            "--overlap" => ov.overlap = Some(parse_value(args, &mut i, a)?),
            "--window" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.window = Some(
                    WindowType::parse(&name).ok_or_else(|| format!("unknown window: {name}"))?,
                );
            }
            "--kaiser-beta" => ov.kaiser_beta = Some(parse_value(args, &mut i, a)?),
            "--dpss-nw" => ov.dpss_nw = Some(parse_value(args, &mut i, a)?),
            "--multiband" => ov.multiband = true,
            "--perceptual" => ov.perceptual = true,
            "--postfilter" => ov.postfilter = true,
            "--smoothing" => ov.smoothing = Some(parse_value(args, &mut i, a)?),
            "--makeup" => ov.makeup = Some(parse_value(args, &mut i, a)?),
            "--no-dc-block" => ov.no_dc_block = true,
            "--report" => ov.report = true,
            "--quality" => {
                let q: String = parse_value(args, &mut i, a)?;
                ov.quality = Some(parse_quality(&q, "")?);
            }
            "--no-transient" => ov.no_transient = true,
            "--cepstral" => ov.cepstral = true,
            "--no-cepstral" => ov.no_cepstral = true,
            "--pre-emphasis" => ov.pre_emphasis = true,
            "--no-pre-emphasis" => ov.no_pre_emphasis = true,
            "--mp3-bitrate" => ov.mp3_bitrate_kbps = Some(parse_value(args, &mut i, a)?),
            "--m4a-bitrate" => ov.m4a_bitrate_kbps = Some(parse_value(args, &mut i, a)?),
            "--aac-encoder" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.aac_encoder = Some(AacEncoder::parse(&name).ok_or_else(|| {
                    format!("unknown AAC encoder: {name} (expected oxide or fdk)")
                })?);
            }
            "--downmix" => {
                let mode: String = parse_value(args, &mut i, a)?;
                ov.downmix = Some(DownmixMode::parse(&mode).ok_or_else(|| {
                    format!("unknown downmix mode: {mode} (expected preserve or stereo)")
                })?);
            }
            "--loudness" => ov.loudness_lufs = Some(parse_value(args, &mut i, a)?),
            "--true-peak" => ov.true_peak_dbtp = Some(parse_value(args, &mut i, a)?),
            "--onnx-model" => {
                if !cli_runtime_package {
                    ov.model_package = None;
                    ov.model_package_key = None;
                }
                cli_raw_model = true;
                ov.onnx_model = Some(parse_value(args, &mut i, a)?);
            }
            "--onnx-rate" => {
                if !cli_runtime_package {
                    ov.model_package = None;
                    ov.model_package_key = None;
                }
                cli_raw_model = true;
                ov.onnx_sample_rate = Some(parse_value(args, &mut i, a)?);
            }
            "--model-package" => {
                if !cli_raw_model {
                    ov.onnx_model = None;
                    ov.onnx_sample_rate = None;
                }
                cli_runtime_package = true;
                ov.model_package = Some(parse_value(args, &mut i, a)?);
            }
            "--model-package-key" => {
                if !cli_raw_model {
                    ov.onnx_model = None;
                    ov.onnx_sample_rate = None;
                }
                cli_runtime_package = true;
                ov.model_package_key = Some(parse_value(args, &mut i, a)?);
            }
            "--channels" => {
                let mode: String = parse_value(args, &mut i, a)?;
                ov.channel_mode = Some(ChannelMode::parse(&mode).ok_or_else(|| {
                    format!(
                        "unknown channel mode: {mode} (expected independent, linked, or mid-side)"
                    )
                })?);
            }
            "--sgmse-profile" => {
                let profile: String = parse_value(args, &mut i, a)?;
                ov.sgmse_profile = Some(SgmseProfile::parse(&profile).ok_or_else(|| {
                    format!(
                        "unknown SGMSE profile: {profile} (expected fast, balanced, or quality)"
                    )
                })?);
            }
            "--accelerator" => {
                let accelerator: String = parse_value(args, &mut i, a)?;
                ov.accelerator = Some(AcceleratorPreference::parse(&accelerator).ok_or_else(
                    || {
                        format!(
                            "unknown accelerator: {accelerator} (expected cpu, auto, gpu, metal, or cuda)"
                        )
                    },
                )?);
            }
            "--deterministic" => ov.deterministic = true,
            "--seed" => {
                ov.seed = Some(parse_value(args, &mut i, a)?);
                ov.deterministic = true;
            }
            "--batch" => ov.batch = true,
            "--stream" => ov.stream = true,
            "--stream-frames" => ov.stream_frames = Some(parse_value(args, &mut i, a)?),
            "--max-memory" => ov.max_memory_mb = Some(parse_value(args, &mut i, a)?),
            "--max-process-memory" => {
                ov.max_process_memory_mb = Some(parse_value(args, &mut i, a)?)
            }
            "--max-temp-space" => ov.max_temporary_mb = Some(parse_value(args, &mut i, a)?),
            "--max-gpu-memory" => ov.max_gpu_memory_mb = Some(parse_value(args, &mut i, a)?),
            "--max-gpu-jobs" => ov.max_gpu_jobs = Some(parse_value(args, &mut i, a)?),
            "--isolate" => ov.isolate = true,
            "--recursive" => ov.recursive = true,
            "--jobs" => ov.jobs = Some(parse_value(args, &mut i, a)?),
            "--output-format" => {
                let value: String = parse_value(args, &mut i, a)?;
                ov.output_format = Some(normalize_output_extension(&value)?.to_ascii_lowercase());
            }
            "--force" => ov.force = true,
            "--resume" => ov.resume = true,
            "--receipt" => ov.receipt = Some(parse_value(args, &mut i, a)?),
            "--receipt-key" => ov.receipt_key = Some(parse_value(args, &mut i, a)?),
            "--plan" => ov.execution_plan = Some(parse_value(args, &mut i, a)?),
            "--no-progress" => ov.no_progress = true,
            "--json" => ov.json = true,
            "--no-metadata" => ov.no_metadata = true,
            "--input-device" => ov.input_device = Some(parse_value(args, &mut i, a)?),
            "--output-device" => ov.output_device = Some(parse_value(args, &mut i, a)?),
            "--chunk-ms" => ov.chunk_ms = Some(parse_value(args, &mut i, a)?),
            "--live-latency" => ov.live_latency_ms = Some(parse_value(args, &mut i, a)?),
            "--max-drift-ppm" => ov.max_drift_ppm = Some(parse_value(args, &mut i, a)?),
            "--reconnect-timeout" => ov.reconnect_timeout_ms = Some(parse_value(args, &mut i, a)?),
            "--list-devices" => ov.list_devices = true,
            "-" => {
                if input.is_none() {
                    input = Some(a.clone());
                } else if output.is_none() {
                    output = Some(a.clone());
                } else {
                    return Err("unexpected extra argument: -".into());
                }
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option: {other}"));
            }
            _ => {
                if input.is_none() {
                    input = Some(a.clone());
                } else if output.is_none() {
                    output = Some(a.clone());
                } else {
                    return Err(format!("unexpected extra argument: {a}"));
                }
            }
        }
        i += 1;
    }

    // Validate the fully merged, effective configuration before looking at
    // positional paths. This keeps configuration errors deterministic and
    // guarantees that invalid values cannot trigger input/output I/O.
    validate_effective_options(&ov, VALIDATION_SAMPLE_RATE)?;
    let input = input.ok_or("missing INPUT")?;
    let output = output.ok_or("missing OUTPUT audio path")?;
    Ok((input, output, ov))
}

fn checked_mib_limit_bytes(value_mb: Option<usize>, option: &str) -> Result<Option<u64>, String> {
    let Some(value_mb) = value_mb else {
        return Ok(None);
    };
    if value_mb == 0 {
        return Err(format!("{option} must be at least 1 MiB"));
    }
    let value_mb = u64::try_from(value_mb)
        .map_err(|_| format!("{option} is too large to represent safely"))?;
    value_mb
        .checked_mul(BYTES_PER_MIB)
        .map(Some)
        .ok_or_else(|| format!("{option} is too large to represent safely"))
}

fn checked_memory_limit_bytes(max_memory_mb: Option<usize>) -> Result<Option<u64>, String> {
    checked_mib_limit_bytes(max_memory_mb, "--max-memory")
}

fn resource_governor(ov: &Overrides, cpu_jobs: usize) -> Result<ResourceGovernor, String> {
    ResourceGovernor::new(
        ResourceLimits::new()
            .with_max_memory_bytes(checked_mib_limit_bytes(
                ov.max_process_memory_mb,
                "--max-process-memory",
            )?)
            .with_max_temporary_bytes(checked_mib_limit_bytes(
                ov.max_temporary_mb,
                "--max-temp-space",
            )?)
            .with_max_cpu_jobs(Some(cpu_jobs))
            .with_max_gpu_jobs(Some(ov.max_gpu_jobs.unwrap_or(1)))
            .with_max_gpu_memory_bytes(checked_mib_limit_bytes(
                ov.max_gpu_memory_mb,
                "--max-gpu-memory",
            )?),
    )
}

fn minimum_limit(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(limit), None) | (None, Some(limit)) => Some(limit),
        (None, None) => None,
    }
}

fn effective_input_memory_mb(ov: &Overrides) -> Option<usize> {
    match (ov.max_memory_mb, ov.max_process_memory_mb) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(limit), None) | (None, Some(limit)) => Some(limit),
        (None, None) => None,
    }
}

fn effective_input_memory_limit_bytes(ov: &Overrides) -> Result<Option<u64>, String> {
    checked_mib_limit_bytes(
        effective_input_memory_mb(ov),
        "effective input memory limit",
    )
}

fn decode_limits_for_bytes(max_working_set_bytes: Option<u64>) -> DecodeLimits {
    DecodeLimits::new(
        metadata_limits_for_available_bytes(max_working_set_bytes),
        max_working_set_bytes,
    )
}

fn decode_limits_for_options(ov: &Overrides) -> Result<DecodeLimits, String> {
    Ok(decode_limits_for_bytes(effective_input_memory_limit_bytes(
        ov,
    )?))
}

fn backend_session_request(
    options: &service::ResolvedProcessingOptions,
) -> Result<ResourceRequest, String> {
    backend_resource_request(
        options.backend,
        &options.backend_options,
        options.accelerator,
    )
}

fn backend_resource_request(
    backend: Backend,
    options: &BackendOptions,
    accelerator: AcceleratorSelection,
) -> Result<ResourceRequest, String> {
    denoize::estimate_backend_session_request(backend, options, accelerator)
}

fn worker_resource_request(
    input_bytes: u64,
    audio: &denoize::Audio,
    metadata_bytes: u64,
    decode_reservation_bytes: Option<u64>,
    processing: &service::ResolvedProcessingOptions,
    writes_staged_output: bool,
) -> Result<ResourceRequest, String> {
    let memory_bytes = estimate_audio_working_set_bytes(audio)
        .checked_add(metadata_bytes)
        .and_then(|bytes| {
            bytes.checked_add(denoize::estimate_backend_worker_memory_bytes(
                &processing.backend_options,
            ))
        })
        .ok_or_else(|| "worker memory reservation overflow".to_string())?
        .max(decode_reservation_bytes.unwrap_or(0));
    let mut request = ResourceRequest::worker(
        memory_bytes,
        if writes_staged_output {
            denoize::estimate_temporary_bytes(input_bytes, audio)?
        } else {
            0
        },
    );
    if processing.accelerator.effective() != denoize::AcceleratorRuntime::Cpu {
        let gpu_bytes = denoize::estimate_gpu_worker_bytes(audio)?
            .checked_add(denoize::estimate_backend_worker_gpu_memory_bytes(
                &processing.backend_options,
            ))
            .ok_or_else(|| "worker GPU memory reservation overflow".to_string())?;
        request = request.with_gpu_jobs(1).with_gpu_memory_bytes(gpu_bytes);
    }
    Ok(request)
}

/// Derive parser limits from bytes which are still available to metadata.
///
/// Metadata is represented more than once while it is translated between a
/// native container and Lofty's generic model. Reserving only one sixteenth
/// of the available working-set budget for payload keeps those copies and
/// allocator overhead conservative. Descriptor counts receive their own
/// finite bound so a stream of empty fields, pages, or blocks cannot evade the
/// byte limits.
fn metadata_limits_for_available_bytes(available: Option<u64>) -> MetadataLimits {
    denoize::metadata_limits_for_available_memory(available)
}

fn retained_metadata_limits(
    max_memory_mb: Option<usize>,
    retained_working_set_bytes: u64,
) -> Result<MetadataLimits, String> {
    Ok(retained_metadata_limits_for_bytes(
        checked_memory_limit_bytes(max_memory_mb)?,
        retained_working_set_bytes,
    ))
}

fn retained_metadata_limits_for_bytes(
    maximum: Option<u64>,
    retained_working_set_bytes: u64,
) -> MetadataLimits {
    denoize::metadata_limits_after_retained_memory(maximum, retained_working_set_bytes)
}

fn checked_m4a_bitrate_bps(kbps: u32) -> Result<u32, String> {
    kbps.checked_mul(1000).ok_or_else(|| {
        format!(
            "invalid --m4a-bitrate/m4a_bitrate_kbps value {kbps}: converting from kbps to bps exceeds the supported u32 representation (maximum {} kbps)",
            u32::MAX / 1000
        )
    })
}

fn build_encode_options(ov: &Overrides) -> Result<EncodeOptions, String> {
    let mut options = EncodeOptions::default();
    if let Some(kbps) = ov.mp3_bitrate_kbps {
        options.mp3_bitrate_kbps = kbps;
    }
    if let Some(kbps) = ov.m4a_bitrate_kbps {
        options.m4a_bitrate_bps = checked_m4a_bitrate_bps(kbps)?;
    }
    if let Some(encoder) = ov.aac_encoder {
        options.aac_encoder = encoder;
    }
    if let Some(downmix) = ov.downmix {
        options.downmix = downmix;
    }
    Ok(options)
}

fn validate_encode_preflight(
    options: EncodeOptions,
    formats: impl IntoIterator<Item = OutputFormat>,
) -> Result<(), String> {
    for format in formats {
        options.validate_options(format)?;
    }
    Ok(())
}

fn batch_preflight_decode_admission(
    ov: &Overrides,
    governor: &ResourceGovernor,
) -> Result<(DecodeLimits, Option<ResourcePermit>), String> {
    let per_input = checked_memory_limit_bytes(ov.max_memory_mb)?;
    let Some(process_limit) =
        checked_mib_limit_bytes(ov.max_process_memory_mb, "--max-process-memory")?
    else {
        return Ok((decode_limits_for_bytes(per_input), None));
    };
    let usage = governor.usage()?;
    let available = process_limit
        .checked_sub(usage.memory_bytes())
        .ok_or_else(|| "cached model sessions exceed --max-process-memory".to_string())?;
    let decode_limit = minimum_limit(per_input, Some(available)).unwrap_or(available);
    if decode_limit < BYTES_PER_MIB {
        return Err(format!(
            "less than 1 MiB remains under --max-process-memory after cached model sessions"
        ));
    }
    let request = ResourceRequest::new().with_memory_bytes(decode_limit);
    let permit = governor.try_acquire(request)?.ok_or_else(|| {
        "batch preflight could not reserve the available process memory".to_string()
    })?;
    Ok((decode_limits_for_bytes(Some(decode_limit)), Some(permit)))
}

fn batch_worker_decode_limit(
    ov: &Overrides,
    governor: &ResourceGovernor,
    transient_audio_bytes: u64,
) -> Result<Option<u64>, String> {
    let per_input = checked_memory_limit_bytes(ov.max_memory_mb)?;
    let process_remaining =
        match checked_mib_limit_bytes(ov.max_process_memory_mb, "--max-process-memory")? {
            Some(limit) => Some(
                limit
                    .checked_sub(
                        governor
                            .usage()?
                            .memory_bytes()
                            .saturating_sub(transient_audio_bytes),
                    )
                    .ok_or_else(|| {
                        "cached model sessions exceed --max-process-memory".to_string()
                    })?,
            ),
            None => None,
        };
    let limit = minimum_limit(per_input, process_remaining);
    if limit.is_some_and(|limit| limit < BYTES_PER_MIB) {
        return Err(
            "less than 1 MiB remains for a decoder under the process resource limits".into(),
        );
    }
    Ok(limit)
}

#[derive(Clone)]
struct GovernedBackendSession {
    session: Arc<BackendSession>,
    _permit: Arc<ResourcePermit>,
}

fn preflight_batch_items(
    items: &[BatchItem],
    ov: &Overrides,
    options: EncodeOptions,
    pre_resolved_backend_options: Option<&BackendOptions>,
    governor: &ResourceGovernor,
    read_only: bool,
) -> Result<Vec<PreparedBatchItem>, String> {
    let effective_memory_mb = effective_input_memory_mb(ov);
    let metadata_policy = if ov.no_metadata {
        MetadataPolicy::Drop
    } else {
        MetadataPolicy::Preserve
    };
    let mut model_fingerprints =
        std::collections::HashMap::<(std::path::PathBuf, u32), ConsumedModel>::new();
    let mut backend_sessions = Vec::<(
        Backend,
        BackendOptions,
        AcceleratorSelection,
        GovernedBackendSession,
    )>::new();
    let mut prepared = Vec::with_capacity(items.len());
    for item in items {
        let (decode_limits, preflight_decode_permit) =
            batch_preflight_decode_admission(ov, governor)?;
        let mut input_session = AudioInputSession::open(&item.input).map_err(|error| {
            format!(
                "open batch input {} during preflight: {error}",
                item.input.display()
            )
        })?;
        let current_probe = probe_audio_session_with_limits(&mut input_session, decode_limits)
            .map_err(|error| {
                format!(
                    "probe batch input {} during preflight: {error}",
                    item.input.display()
                )
            })?;
        if current_probe != item.probe {
            return Err(format!(
                "batch input codec/container changed after planning: {}",
                item.input.display()
            ));
        }
        let input_fingerprint = batch_resume::fingerprint_input_session(&mut input_session)
            .map_err(|error| {
                format!(
                    "fingerprint batch input {} during preflight: {error}",
                    item.input.display()
                )
            })?;
        let estimate = estimate_session_memory_bytes(&input_session);
        ensure_memory_limit(estimate, effective_memory_mb, "batch input preflight")?;
        let mut audio = read_audio_from_session_with_limits(&mut input_session, decode_limits)
            .map_err(|error| {
                format!(
                    "decode batch input {} during preflight: {error}",
                    item.input.display()
                )
            })?;
        let mut decoded_working_set = estimate_audio_working_set_bytes(&audio);
        ensure_memory_limit(
            decoded_working_set,
            effective_memory_mb,
            "batch decoded audio working set",
        )?;
        drop(preflight_decode_permit);
        let mut audio_permit = Some(governor
            .try_acquire(ResourceRequest::new().with_memory_bytes(decoded_working_set))?
            .ok_or_else(|| {
                format!(
                    "batch input {} cannot fit beside cached model sessions under --max-process-memory",
                    item.input.display()
                )
            })?);
        item.output_format
            .validate_config(&audio, &options)
            .map_err(|error| {
                format!(
                    "batch output preflight failed for {}: {error}",
                    item.input.display()
                )
            })?;
        let processing_options = build_processing_options(
            ov,
            audio.sample_rate,
            match pre_resolved_backend_options {
                Some(options) => options.clone(),
                None => build_backend_options(ov)?,
            },
        );
        let resolved_processing = if read_only {
            service::resolve_processing_options_read_only(&audio, processing_options)
        } else {
            service::resolve_processing_options(&audio, processing_options)
        }
        .map_err(|error| {
            format!(
                "batch processing preflight failed for {}: {error}",
                item.input.display()
            )
        })?;
        let model = match batch_resume::consumed_model_config(&resolved_processing)? {
            Some(config) => {
                let key = (config.path.clone(), config.sample_rate);
                let model = match model_fingerprints.get(&key) {
                    Some(model) => model.clone(),
                    None => {
                        let model = (if ov.resume {
                            batch_resume::resumable_consumed_model(&resolved_processing)
                        } else {
                            batch_resume::consumed_model(&resolved_processing)
                        })
                        .map_err(|error| {
                            format!(
                                "fingerprint selected backend model {}: {error}",
                                config.path.display()
                            )
                        })?
                        .ok_or_else(|| {
                            "resolved backend lost its consumed model during fingerprinting"
                                .to_string()
                        })?;
                        model_fingerprints.insert(key, model.clone());
                        model
                    }
                };
                Some(model)
            }
            None => None,
        };
        // Hash the selected model before preparing its graph. The whole-plan
        // source fence below then re-hashes it after preparation, so a
        // persistent pathname replacement cannot bind model A's graph to
        // model B's resume fingerprint.
        let backend_session = cached_backend_session(
            &mut backend_sessions,
            &resolved_processing,
            ov.report,
            governor,
        )
        .map_err(|error| {
            format!(
                "prepare batch backend {} for {}: {error}",
                service::backend_name(resolved_processing.backend),
                item.input.display()
            )
        })?;
        let final_decode_limit = batch_worker_decode_limit(ov, governor, decoded_working_set)?;
        let must_redecode = match (decode_limits.max_working_set_bytes, final_decode_limit) {
            (Some(initial), Some(final_limit)) => final_limit < initial,
            (None, Some(_)) => true,
            _ => false,
        };
        if must_redecode {
            drop(audio_permit.take());
            drop(audio);
            let final_limit = final_decode_limit.expect("redecode requires a finite limit");
            let decode_permit = governor
                .try_acquire(ResourceRequest::new().with_memory_bytes(final_limit))?
                .ok_or_else(|| {
                    format!(
                        "batch input {} cannot reserve its final decode budget",
                        item.input.display()
                    )
                })?;
            audio = read_audio_from_session_with_limits(
                &mut input_session,
                decode_limits_for_bytes(final_decode_limit),
            )
            .map_err(|error| {
                format!(
                    "decode batch input {} beside cached model sessions: {error}",
                    item.input.display()
                )
            })?;
            drop(decode_permit);
            decoded_working_set = estimate_audio_working_set_bytes(&audio);
            audio_permit = Some(
                governor
                    .try_acquire(
                        ResourceRequest::new().with_memory_bytes(decoded_working_set),
                    )?
                    .ok_or_else(|| {
                        format!(
                            "batch input {} cannot retain decoded audio beside cached model sessions",
                            item.input.display()
                        )
                    })?,
            );
        }
        let metadata_bytes = if !ov.no_metadata {
            let metadata_limits =
                retained_metadata_limits_for_bytes(final_decode_limit, decoded_working_set);
            input_session
                .read_metadata_with_limits(metadata_limits)
                .map_err(|error| {
                    format!(
                        "read batch input metadata {} during preflight: {error}",
                        item.input.display()
                    )
                })?
                .as_ref()
                .map(denoize::metadata::Metadata::estimated_memory_bytes)
                .unwrap_or(0)
        } else {
            0
        };
        let resource_request = worker_resource_request(
            input_session.len(),
            &audio,
            metadata_bytes,
            final_decode_limit,
            &resolved_processing,
            !ov.report,
        )?;
        let recipe = batch_resume::recipe_digest(
            &resolved_processing,
            audio.channels(),
            item.output_format,
            options,
            metadata_policy,
            model
                .as_ref()
                .map(|model| (&model.fingerprint, model.sample_rate)),
        )?;
        let input_identity = normalize_batch_path(&item.input)?;
        let item_id = batch_resume::item_identity(
            &input_identity,
            &item.input_relative,
            &item.destination_relative,
            item.output_format,
        );
        let expectation = ResumeExpectation::new(
            item_id,
            item.destination.clone(),
            item.input.clone(),
            input_fingerprint,
            model,
            recipe,
        );
        drop(audio_permit);
        drop(governor.try_acquire(resource_request)?.ok_or_else(|| {
            format!(
                "batch input {} cannot be admitted under the configured process resource limits",
                item.input.display()
            )
        })?);
        prepared.push(PreparedBatchItem {
            item: item.clone(),
            resolved_processing,
            backend_session,
            resource_request,
            expectation,
            recipe,
            channels: audio.channels(),
            frames: u64::try_from(audio.frames()).map_err(|_| {
                format!(
                    "batch input frame count is too large to represent: {}",
                    item.input.display()
                )
            })?,
            sample_rate: audio.sample_rate,
        });
    }
    // A cached model fingerprint is safe only if the model still matches once
    // the complete plan has been built. Inputs receive the same whole-plan
    // fence before the output directory or state can be touched.
    for item in &prepared {
        item.expectation.verify_sources()?;
        drop(
            governor
                .try_acquire(item.resource_request)?
                .ok_or_else(|| {
                    format!(
                        "batch input {} no longer fits after all backend sessions were prepared; increase --max-process-memory or lower --max-memory",
                        item.item.input.display()
                    )
                })?,
        );
    }
    debug_assert_eq!(prepared.len(), items.len());
    Ok(prepared)
}

fn cached_backend_session(
    cache: &mut Vec<(
        Backend,
        BackendOptions,
        AcceleratorSelection,
        GovernedBackendSession,
    )>,
    options: &service::ResolvedProcessingOptions,
    report_only: bool,
    governor: &ResourceGovernor,
) -> Result<Option<GovernedBackendSession>, String> {
    if report_only {
        return Ok(None);
    }
    if let Some((_, _, _, session)) =
        cache
            .iter()
            .find(|(backend, backend_options, accelerator, _)| {
                *backend == options.backend
                    && backend_options == &options.backend_options
                    && *accelerator == options.accelerator
            })
    {
        return Ok(Some(session.clone()));
    }
    let request = backend_session_request(options)?;
    let permit = Arc::new(governor.try_acquire(request)?.ok_or_else(|| {
        format!(
            "backend session {} cannot fit under the configured process resource limits",
            service::backend_name(options.backend)
        )
    })?);
    let session = Arc::new(BackendSession::prepare_with_accelerator(
        options.backend,
        options.backend_options.clone(),
        options.accelerator,
    )?);
    let governed = GovernedBackendSession {
        session,
        _permit: permit,
    };
    cache.push((
        options.backend,
        options.backend_options.clone(),
        options.accelerator,
        governed.clone(),
    ));
    Ok(Some(governed))
}

fn effective_batch_jobs(ov: &Overrides) -> usize {
    ov.jobs.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(MAX_BATCH_JOBS)
    })
}

fn build_backend_options(ov: &Overrides) -> Result<BackendOptions, String> {
    let runtime_package = match (&ov.model_package, &ov.model_package_key) {
        (Some(package), Some(key)) => Some(RuntimeModelPackage::open(package, key)?),
        (None, None) => None,
        _ => {
            return Err("--model-package and --model-package-key must be supplied together".into())
        }
    };
    let mut options = BackendOptions {
        onnx: ov.onnx_model.as_ref().map(|path| OnnxModelConfig {
            path: path.into(),
            sample_rate: ov.onnx_sample_rate.unwrap_or(16_000),
        }),
        runtime_package: None,
        channel_mode: ov.channel_mode.unwrap_or_default(),
        sgmse_profile: ov.sgmse_profile.unwrap_or_default(),
        deterministic: ov.deterministic,
        accelerator: ov.accelerator.unwrap_or_default(),
        seed: ov.seed,
    };
    if let Some(package) = runtime_package {
        options = options.with_runtime_model_package(package);
    }
    Ok(options)
}

fn processing_backend_choice(ov: &Overrides) -> BackendChoice {
    if ov.auto_backend {
        BackendChoice::Auto
    } else {
        BackendChoice::Explicit(ov.backend.unwrap_or(Backend::Classical))
    }
}

fn runtime_package_backend_selected(ov: &Overrides) -> bool {
    #[cfg(feature = "onnx")]
    {
        if ov.auto_backend {
            return false;
        }
        if ov.backend == Some(Backend::Onnx) {
            return true;
        }
        #[cfg(feature = "bsrnn")]
        if ov.backend == Some(Backend::Bsrnn) {
            return true;
        }
        false
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = ov;
        false
    }
}

fn build_processing_options(
    ov: &Overrides,
    sample_rate: u32,
    backend_options: BackendOptions,
) -> ProcessingOptions {
    ProcessingOptions {
        backend: processing_backend_choice(ov),
        quality: ov.quality.clone(),
        denoiser: build_config(ov, sample_rate),
        backend_options,
        loudness_lufs: ov.loudness_lufs,
        true_peak_dbtp: ov.true_peak_dbtp.unwrap_or(-1.0),
    }
}

fn resolve_explicit_backend_options(ov: &Overrides) -> Result<Option<BackendOptions>, String> {
    if ov.auto_backend {
        return Ok(None);
    }
    let backend = ov.backend.unwrap_or(Backend::Classical);
    let options = service::resolve_backend_options(backend, build_backend_options(ov)?)?;
    denoize::select_accelerator_for_options(backend, &options)?;
    Ok(Some(options))
}

fn resolve_explicit_backend_options_read_only(
    ov: &Overrides,
) -> Result<Option<BackendOptions>, String> {
    if ov.auto_backend {
        return Ok(None);
    }
    let backend = ov.backend.unwrap_or(Backend::Classical);
    let options = service::resolve_backend_options_read_only(backend, build_backend_options(ov)?)?;
    denoize::select_accelerator_for_options(backend, &options)?;
    Ok(Some(options))
}

fn validate_effective_options(ov: &Overrides, sample_rate: u32) -> Result<(), String> {
    // `parse_config` deliberately postpones numeric checks until after CLI
    // overrides have been applied. Validate only this final effective config.
    build_config(ov, sample_rate)
        .validate_config()
        .map_err(|error| error.to_string())?;

    if let Some(loudness) = ov.loudness_lufs {
        if !loudness.is_finite() || !(-70.0..=0.0).contains(&loudness) {
            return Err(format!(
                "invalid --loudness/loudness_lufs value {loudness}: expected a finite value in [-70, 0] LUFS"
            ));
        }
    }
    if let Some(true_peak) = ov.true_peak_dbtp {
        if !true_peak.is_finite() || !(-20.0..=0.0).contains(&true_peak) {
            return Err(format!(
                "invalid --true-peak/true_peak_dbtp value {true_peak}: expected a finite value in [-20, 0] dBTP"
            ));
        }
        if ov.loudness_lufs.is_none() {
            return Err("--true-peak requires --loudness".into());
        }
    }
    if let Some(sample_rate) = ov.onnx_sample_rate {
        if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
            return Err(format!(
                "--onnx-rate/onnx_rate must be in 1..={MAX_SAMPLE_RATE} Hz"
            ));
        }
    }
    match (&ov.model_package, &ov.model_package_key) {
        (Some(_), Some(_)) => {
            if ov.onnx_model.is_some() || ov.onnx_sample_rate.is_some() {
                return Err(
                    "--model-package cannot be combined with --onnx-model or --onnx-rate".into(),
                );
            }
            if !runtime_package_backend_selected(ov) {
                return Err("--model-package requires --backend onnx or bsrnn".into());
            }
        }
        (None, None) => {
            if !ov.auto_backend {
                build_backend_options(ov)?
                    .validate_config(ov.backend.unwrap_or(Backend::Classical))
                    .map_err(|error| error.to_string())?;
            }
        }
        _ => {
            return Err("--model-package and --model-package-key must be supplied together".into())
        }
    }
    if let Some(stream_frames) = ov.stream_frames {
        if !(MIN_STREAM_BLOCK_FRAMES..=MAX_STREAM_BLOCK_FRAMES).contains(&stream_frames) {
            return Err(format!(
                "--stream-frames/stream_frames must be in {MIN_STREAM_BLOCK_FRAMES}..={MAX_STREAM_BLOCK_FRAMES}"
            ));
        }
    }
    if let Some(chunk_ms) = ov.chunk_ms {
        if !(MIN_LIVE_CHUNK_MS..=MAX_LIVE_CHUNK_MS).contains(&chunk_ms) {
            return Err(format!(
                "--chunk-ms must be in {MIN_LIVE_CHUNK_MS}..={MAX_LIVE_CHUNK_MS}"
            ));
        }
    }
    if let Some(latency_ms) = ov.live_latency_ms {
        if latency_ms != 0
            && !(MIN_LIVE_TARGET_LATENCY_MS..=MAX_LIVE_TARGET_LATENCY_MS).contains(&latency_ms)
        {
            return Err(format!(
                "--live-latency must be 0 or in {MIN_LIVE_TARGET_LATENCY_MS}..={MAX_LIVE_TARGET_LATENCY_MS}"
            ));
        }
    }
    if let Some(max_drift_ppm) = ov.max_drift_ppm {
        if max_drift_ppm > MAX_LIVE_DRIFT_PPM {
            return Err(format!(
                "--max-drift-ppm must be in 0..={MAX_LIVE_DRIFT_PPM}"
            ));
        }
    }
    if let Some(timeout_ms) = ov.reconnect_timeout_ms {
        if timeout_ms > MAX_LIVE_RECONNECT_TIMEOUT_MS {
            return Err(format!(
                "--reconnect-timeout must be in 0..={MAX_LIVE_RECONNECT_TIMEOUT_MS}"
            ));
        }
    }
    if let Some(jobs) = ov.jobs {
        if !(1..=MAX_BATCH_JOBS).contains(&jobs) {
            return Err(format!("--jobs/jobs must be in 1..={MAX_BATCH_JOBS}"));
        }
    }
    if let Some(max_gpu_jobs) = ov.max_gpu_jobs {
        if !(1..=MAX_BATCH_JOBS).contains(&max_gpu_jobs) {
            return Err(format!(
                "--max-gpu-jobs/max_gpu_jobs must be in 1..={MAX_BATCH_JOBS}"
            ));
        }
    }
    let encode_options = build_encode_options(ov)?;
    if let Some(extension) = ov.output_format.as_deref() {
        let path = std::path::PathBuf::from(format!("output.{extension}"));
        validate_encode_preflight(encode_options, [OutputFormat::from_path(&path)?])?;
    }
    checked_memory_limit_bytes(ov.max_memory_mb)?;
    checked_mib_limit_bytes(ov.max_process_memory_mb, "--max-process-memory")?;
    checked_mib_limit_bytes(ov.max_temporary_mb, "--max-temp-space")?;
    checked_mib_limit_bytes(ov.max_gpu_memory_mb, "--max-gpu-memory")?;
    Ok(())
}

fn build_config(ov: &Overrides, sample_rate: u32) -> DenoiserConfig {
    let mut cfg = match ov.preset {
        Some(p) => p.config(sample_rate),
        None => DenoiserConfig::default(sample_rate),
    };
    if let Some(mode) = ov.mode {
        mode.apply(&mut cfg);
    }
    if let Some(a) = ov.algorithm {
        cfg.algorithm = a;
    }
    if let Some(s) = ov.strength {
        cfg.strength = s;
    }
    if ov.no_profile {
        cfg.profile_ms = -1.0;
    } else if let Some(ms) = ov.profile_ms {
        cfg.profile_ms = ms;
    }
    if ov.no_adapt {
        cfg.adapt = false;
    }
    if let Some(adaptive_noise) = ov.adaptive_noise {
        cfg.adaptive_noise = adaptive_noise;
    }
    if let Some(vad) = ov.vad {
        cfg.vad = vad;
    }
    if let Some(f) = ov.frame_size {
        cfg.frame_size = f;
    }
    if let Some(o) = ov.overlap {
        cfg.overlap = o;
    }
    if let Some(w) = ov.window {
        cfg.window = w;
    }
    if let Some(b) = ov.kaiser_beta {
        cfg.window_params.kaiser_beta = b;
    }
    if let Some(nw) = ov.dpss_nw {
        cfg.window_params.dpss_bandwidth = nw;
    }
    if ov.multiband {
        cfg.multiband = true;
    }
    if ov.perceptual {
        cfg.perceptual_weighting = true;
    }
    if ov.postfilter {
        cfg.musical_noise_postfilter = true;
    }
    if let Some(s) = ov.smoothing {
        cfg.smoothing = s;
    }
    if let Some(m) = ov.makeup {
        cfg.makeup_gain_db = m;
    }
    if ov.no_dc_block {
        cfg.dc_block = false;
    }

    if let Some(ref q) = ov.quality {
        match q.as_str() {
            "high" => {
                if cfg.frame_size < 2048 {
                    cfg.frame_size = 2048;
                }
                if cfg.overlap < 0.8 {
                    cfg.overlap = 0.8;
                }
                cfg.transient_protect = true;
                cfg.cepstral_smoothing = true;
                cfg.perceptual_weighting = true;
                cfg.musical_noise_postfilter = true;
                if !ov.no_pre_emphasis {
                    cfg.pre_emphasis = true;
                }
            }
            "ultra" | "max" | "highest" => {
                cfg.frame_size = cfg.frame_size.max(4096);
                cfg.overlap = 0.875;
                if ov.window.is_none() {
                    cfg.window = WindowType::Kaiser;
                }
                if ov.kaiser_beta.is_none() {
                    cfg.window_params.kaiser_beta = 10.0;
                }
                cfg.transient_protect = true;
                cfg.cepstral_smoothing = true;
                cfg.perceptual_weighting = true;
                cfg.musical_noise_postfilter = true;
                cfg.pre_emphasis = true;
                if ov.strength.is_none() && cfg.strength > 0.4 {
                    cfg.strength = 0.32;
                }
            }
            _ => {}
        }
    }

    if ov.no_transient {
        cfg.transient_protect = false;
    }
    if ov.cepstral {
        cfg.cepstral_smoothing = true;
    }
    if ov.no_cepstral {
        cfg.cepstral_smoothing = false;
    }
    if ov.pre_emphasis {
        cfg.pre_emphasis = true;
    }
    if ov.no_pre_emphasis {
        cfg.pre_emphasis = false;
    }

    cfg
}

fn print_report(
    input: &std::path::Path,
    audio: &denoize::Audio,
    cfg: &DenoiserConfig,
    backend: Backend,
    accelerator: AcceleratorSelection,
) {
    let hop = (cfg.frame_size as f64 * (1.0 - cfg.overlap)).round() as usize;
    let g_min_db = -20.0 - 25.0 * cfg.strength;
    let dur = audio.frames() as f64 / audio.sample_rate as f64;
    println!("input      : {}", input.display());
    println!(
        "format     : {}ch, {:.2}s ({} frames), {} Hz, {}-bit {:?}",
        audio.channels(),
        dur,
        audio.frames(),
        audio.sample_rate,
        audio.bits_per_sample,
        audio.sample_format,
    );
    println!("layout     : {}", audio.channel_layout());
    if let Some(mask) = audio.channel_mask {
        println!("mask       : {mask}");
    }
    if let Some(pan) = audio.pan_info() {
        let positions = pan
            .iter()
            .enumerate()
            .map(|(index, info)| {
                format!(
                    "ch{}={:.0}°/{:.0}°",
                    index + 1,
                    info.azimuth_degrees,
                    info.elevation_degrees
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!("pan        : {positions}");
    }
    println!("backend    : {backend:?}");
    println!("accelerator: {}", accelerator_description(accelerator));
    println!("algorithm  : {:?}", cfg.algorithm);
    println!(
        "strength   : {:.2}  (gain floor ~{:.0} dB)",
        cfg.strength, g_min_db
    );
    println!(
        "STFT       : frame={}, hop={}, overlap={:.0}%, window={:?}",
        cfg.frame_size,
        hop,
        cfg.overlap * 100.0,
        cfg.window,
    );
    println!(
        "advanced   : multiband={}, perceptual={}, postfilter={}",
        cfg.multiband, cfg.perceptual_weighting, cfg.musical_noise_postfilter
    );
    println!("smoothing  : {:.2}", cfg.smoothing);
    println!(
        "profile    : {}",
        if cfg.profile_ms < 0.0 {
            "disabled".to_string()
        } else if cfg.profile_ms == 0.0 {
            "auto (leading silence)".to_string()
        } else {
            format!("{:.0} ms", cfg.profile_ms)
        }
    );
    println!("adapt      : {}", cfg.adapt);
    println!("adaptive-profile: {}", cfg.adaptive_noise);
    println!("dc-block   : {}", cfg.dc_block);
    println!("makeup     : {:.1} dB", cfg.makeup_gain_db);
    println!(
        "hi-fi      : transient={}, cepstral={}, pre-emphasis={}",
        cfg.transient_protect, cfg.cepstral_smoothing, cfg.pre_emphasis
    );
}

fn watch_usage() -> String {
    format!(
        "\
denoize {VERSION} watch-folder automation

USAGE:
    denoize watch <INPUT_DIR> <OUTPUT_DIR> --receipt-key <SECRET_KEY.json> [OPTIONS]

WATCH OPTIONS:
        --once                    settle and scan once, then exit
        --settle-ms <MS>          unchanged-content interval in 0..2592000000 (default: 2000)
        --poll-ms <MS>            daemon polling interval in 1..2592000000 (default: 500)
        --retry-initial-ms <MS>   initial retry delay (default: 1000)
        --retry-max-ms <MS>       maximum exponential delay (default: 60000)
        --max-attempts <N>        attempts before quarantine in 1..100 (default: 5)
        --max-watch-files <N>     bounded directory entries in 1..100000 (default: 10000)
        --quarantine <DIR>        failed-input root (default: OUTPUT/.denoize-quarantine)
        --receipt-dir <DIR>       per-item signed receipts (default: OUTPUT/.denoize-receipts)
        --watch-state <PATH>      durable state (default: OUTPUT/.denoize-watch-state.json)

PROCESSING OPTIONS:
    File-processing options from `denoize --help` are accepted. `--output-format`
    defaults to wav. `--recursive` includes subdirectories. Watch mode is
    sequential and forbids --batch, --stream, --resume, --force, --report,
    --isolate, --receipt, and --jobs. A receipt key is mandatory; every
    successful output is atomically paired with a signed receipt.

SETTLE AND FAILURE CONTRACT:
    A candidate must retain the same regular-file length, modification stamp,
    and SHA-256 content for the full settle interval. Processing failures use
    bounded exponential retry. Exhausted or permanent failures are copied to
    quarantine with a v1 JSON explanation before the source is removed. The
    durable state and output roots must be outside the input tree. State is
    bound to the processing, output, signing-key, and explicit-model template;
    choose a new state path after an intentional template change.
"
    )
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct WatchCommandOptions {
    once: bool,
    settle_millis: Option<u64>,
    poll_millis: Option<u64>,
    retry_initial_millis: Option<u64>,
    retry_max_millis: Option<u64>,
    max_attempts: Option<u32>,
    max_files: Option<usize>,
    quarantine_root: Option<String>,
    receipt_root: Option<String>,
    state_path: Option<String>,
}

#[derive(Serialize)]
struct WatchCycleJson<'a> {
    schema: &'static str,
    schema_version: u32,
    input: &'a str,
    output: &'a str,
    cancelled: bool,
    #[serde(flatten)]
    report: &'a WatchCycleReport,
}

fn parse_watch_args(
    args: &[String],
) -> Result<(String, String, Overrides, WatchCommandOptions), String> {
    if args.is_empty() || (args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help")) {
        print!("{}", watch_usage());
        return Err(String::new());
    }
    let mut watch = WatchCommandOptions::default();
    let mut processing = Vec::with_capacity(args.len());
    let mut index = 0_usize;
    while index < args.len() {
        let value = &args[index];
        match value.as_str() {
            "--once" => watch.once = true,
            "--settle-ms" => watch.settle_millis = Some(parse_value(args, &mut index, value)?),
            "--poll-ms" => watch.poll_millis = Some(parse_value(args, &mut index, value)?),
            "--retry-initial-ms" => {
                watch.retry_initial_millis = Some(parse_value(args, &mut index, value)?)
            }
            "--retry-max-ms" => {
                watch.retry_max_millis = Some(parse_value(args, &mut index, value)?)
            }
            "--max-attempts" => watch.max_attempts = Some(parse_value(args, &mut index, value)?),
            "--max-watch-files" => watch.max_files = Some(parse_value(args, &mut index, value)?),
            "--quarantine" => watch.quarantine_root = Some(parse_value(args, &mut index, value)?),
            "--receipt-dir" => watch.receipt_root = Some(parse_value(args, &mut index, value)?),
            "--watch-state" => watch.state_path = Some(parse_value(args, &mut index, value)?),
            _ => processing.push(value.clone()),
        }
        index += 1;
    }
    let (input, output, options) = parse_args(&processing)?;
    Ok((input, output, options, watch))
}

fn validate_watch_processing_options(options: &Overrides) -> Result<(), String> {
    if options.batch
        || options.stream
        || options.resume
        || options.force
        || options.report
        || options.isolate
    {
        return Err(
            "watch cannot use --batch, --stream, --resume, --force, --report, or --isolate".into(),
        );
    }
    if options.receipt.is_some() {
        return Err(
            "watch creates one receipt per item; use --receipt-dir instead of --receipt".into(),
        );
    }
    if options.receipt_key.is_none() {
        return Err("watch requires --receipt-key for per-item signed receipts".into());
    }
    if options.jobs.is_some() {
        return Err("watch is sequential; --jobs is not supported".into());
    }
    if options.input_device.is_some()
        || options.output_device.is_some()
        || options.list_devices
        || options.chunk_ms.is_some()
        || options.live_latency_ms.is_some()
        || options.max_drift_ppm.is_some()
        || options.reconnect_timeout_ms.is_some()
    {
        return Err("live-device options cannot be used with watch".into());
    }
    Ok(())
}

fn watch_config(
    input: &str,
    output: &str,
    processing: &Overrides,
    command: &WatchCommandOptions,
    processor_identity: Digest,
) -> WatchFolderConfig {
    let mut config = WatchFolderConfig::new(input, output, processor_identity.as_bytes())
        .with_recursive(processing.recursive)
        .with_output_extension(processing.output_format.as_deref().unwrap_or("wav"));
    if let Some(path) = &command.quarantine_root {
        config = config.with_quarantine_root(path);
    }
    if let Some(path) = &command.receipt_root {
        config = config.with_receipt_root(path);
    }
    if let Some(path) = &command.state_path {
        config = config.with_state_path(path);
    }
    if let Some(value) = command.settle_millis {
        config = config.with_settle_duration(Duration::from_millis(value));
    }
    if let Some(value) = command.poll_millis {
        config = config.with_poll_interval(Duration::from_millis(value));
    }
    config = config.with_retry_delays(
        Duration::from_millis(command.retry_initial_millis.unwrap_or(1_000)),
        Duration::from_millis(command.retry_max_millis.unwrap_or(60_000)),
    );
    if let Some(value) = command.max_attempts {
        config = config.with_max_attempts(value);
    }
    if let Some(value) = command.max_files {
        config = config.with_max_files(value);
    }
    config
}

fn update_watch_identity(hasher: &mut Sha256, label: &str, value: &[u8]) {
    hasher.update((label.len() as u64).to_le_bytes());
    hasher.update(label.as_bytes());
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn watch_processor_identity(
    processing: &Overrides,
    public_key: &ReceiptPublicKey,
) -> Result<Digest, String> {
    let mut material = processing.clone();
    material.max_memory_mb = None;
    material.max_process_memory_mb = None;
    material.max_temporary_mb = None;
    material.max_gpu_memory_mb = None;
    material.max_gpu_jobs = None;
    material.no_progress = false;
    material.json = false;
    material.receipt_key = None;
    let mut hasher = Sha256::new();
    update_watch_identity(&mut hasher, "domain", b"denoize-watch-processor-v1");
    update_watch_identity(&mut hasher, "denoize-version", VERSION.as_bytes());
    update_watch_identity(
        &mut hasher,
        "processing-options",
        format!("{material:#?}").as_bytes(),
    );
    update_watch_identity(
        &mut hasher,
        "receipt-public-key-id",
        public_key.key_id.as_bytes(),
    );
    for (label, path) in [
        ("onnx-model", processing.onnx_model.as_deref()),
        ("model-package", processing.model_package.as_deref()),
        ("model-package-key", processing.model_package_key.as_deref()),
    ] {
        update_watch_identity(
            &mut hasher,
            &format!("{label}-present"),
            &[u8::from(path.is_some())],
        );
        if let Some(path) = path {
            let fingerprint = batch_resume::fingerprint_file(std::path::Path::new(path))
                .map_err(|error| format!("fingerprint watch {label} {path}: {error}"))?;
            let mut encoded = [0_u8; 40];
            encoded[..8].copy_from_slice(&fingerprint.len.to_le_bytes());
            encoded[8..].copy_from_slice(fingerprint.digest.as_bytes());
            update_watch_identity(&mut hasher, label, &encoded);
        }
    }
    Ok(Digest::from_bytes(hasher.finalize().into()))
}

fn classify_watch_process_error(error: String) -> WatchProcessError {
    let lowercase = error.to_ascii_lowercase();
    if lowercase.contains("cancelled") || error.contains("キャンセル") {
        return WatchProcessError::deferred(error);
    }
    if [
        "unsupported",
        "unknown or ambiguous",
        "malformed",
        "truncated",
        "cannot preserve",
        "must contain exactly one supported audio track",
        "invalid audio",
    ]
    .iter()
    .any(|needle| lowercase.contains(needle))
    {
        WatchProcessError::permanent(error)
    } else {
        WatchProcessError::retryable(error)
    }
}

fn path_exists_for_watch(path: &std::path::Path) -> Result<bool, String> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!(
            "inspect watch artifact {}: {error}",
            path.display()
        )),
    }
}

fn recover_watch_job(job: &WatchFolderJob, public_key: &ReceiptPublicKey) -> Result<bool, String> {
    let output_exists = path_exists_for_watch(&job.output_path)?;
    let receipt_exists = path_exists_for_watch(&job.receipt_path)?;
    match (output_exists, receipt_exists) {
        (false, false) => return Ok(false),
        (true, false) => {
            return Err(format!(
                "watch output exists without its signed receipt: {}",
                job.output_path.display()
            ))
        }
        (false, true) => {
            return Err(format!(
                "watch receipt exists without its output: {}",
                job.receipt_path.display()
            ))
        }
        (true, true) => {}
    }
    let receipt = SignedExecutionReceipt::from_file(&job.receipt_path)?;
    let output_root = job
        .output_path
        .parent()
        .ok_or("watch output path has no parent directory")?;
    receipt.verify_with_key(public_key, None, &job.receipt_path, Some(output_root))?;
    if receipt.payload.items.len() != 1 {
        return Err("watch receipt must contain exactly one item".into());
    }
    let item = &receipt.payload.items[0];
    if item.input.fingerprint != job.input_fingerprint {
        return Err("watch receipt input fingerprint does not match its settled job".into());
    }
    let expected_output = denoize::portable_file_locator(&job.output_path)?;
    if item.output.path != expected_output {
        return Err("watch receipt output locator does not match its scheduled job".into());
    }
    Ok(true)
}

fn process_watch_job(
    job: &WatchFolderJob,
    base_options: &Overrides,
    receipt_key_path: &std::path::Path,
    expected_key_fingerprint: FileFingerprint,
    public_key: &ReceiptPublicKey,
    expected_processor_identity: Digest,
) -> Result<(), WatchProcessError> {
    let current_key = batch_resume::fingerprint_file(receipt_key_path).map_err(|error| {
        WatchProcessError::deferred(format!(
            "watch receipt key is temporarily unavailable: {error}"
        ))
    })?;
    if current_key != expected_key_fingerprint {
        return Err(WatchProcessError::deferred(
            "watch receipt key changed; restart the watcher to adopt the new key",
        ));
    }
    let current_processor_identity =
        watch_processor_identity(base_options, public_key).map_err(|error| {
            WatchProcessError::deferred(format!(
                "watch processor template is temporarily unavailable: {error}"
            ))
        })?;
    if current_processor_identity != expected_processor_identity {
        return Err(WatchProcessError::deferred(
            "watch processor template changed; restart with a fresh state path to adopt it",
        ));
    }
    match recover_watch_job(job, public_key) {
        Ok(true) => return Ok(()),
        Ok(false) => {}
        Err(error) => return Err(WatchProcessError::permanent(error)),
    }
    let mut options = base_options.clone();
    options.batch = false;
    options.stream = false;
    options.recursive = false;
    options.resume = false;
    options.force = false;
    options.report = false;
    options.json = false;
    options.no_progress = true;
    options.receipt = Some(job.receipt_path.to_string_lossy().into_owned());
    options.receipt_key = Some(receipt_key_path.to_string_lossy().into_owned());
    run_one_with_output_format(
        &job.input_path,
        &job.output_path,
        options,
        None,
        None,
        Some(job.input_fingerprint),
    )
    .map_err(classify_watch_process_error)?;
    match recover_watch_job(job, public_key) {
        Ok(true) => Ok(()),
        Ok(false) => Err(WatchProcessError::permanent(
            "watch processing returned without publishing output and receipt",
        )),
        Err(error) => Err(WatchProcessError::permanent(error)),
    }
}

fn print_watch_cycle(
    input: &str,
    output: &str,
    options: &Overrides,
    report: &WatchCycleReport,
    cancelled: bool,
) {
    if options.json {
        println!(
            "{}",
            serialize_json_line(&WatchCycleJson {
                schema: WATCH_CYCLE_SCHEMA,
                schema_version: 1,
                input,
                output,
                cancelled,
                report,
            })
        );
    } else if !options.no_progress
        && (report.observed != 0
            || report.attempted != 0
            || report.quarantined != 0
            || report.scan_errors != 0
            || cancelled)
    {
        eprintln!(
            "denoize: watch observed={} attempted={} succeeded={} retrying={} quarantined={} superseded={} scan_errors={} pending={}{}",
            report.observed,
            report.attempted,
            report.succeeded,
            report.retrying,
            report.quarantined,
            report.superseded,
            report.scan_errors,
            report.pending,
            if cancelled { " cancelled" } else { "" }
        );
    }
}

fn wait_watch_interval(duration: Duration) {
    let started = Instant::now();
    while !CANCELLED.load(Ordering::SeqCst) {
        let elapsed = started.elapsed();
        if elapsed >= duration {
            break;
        }
        std::thread::sleep((duration - elapsed).min(Duration::from_millis(100)));
    }
}

fn run_watch(args: &[String]) -> Result<(), String> {
    let (input, output, mut processing, command) = match parse_watch_args(args) {
        Ok(values) => values,
        Err(error) if error.is_empty() => return Ok(()),
        Err(error) => return Err(error),
    };
    validate_watch_processing_options(&processing)?;
    let key_path = std::path::PathBuf::from(
        processing
            .receipt_key
            .as_deref()
            .ok_or("watch requires --receipt-key")?,
    );
    let key = ReceiptSecretKey::from_file(&key_path)?;
    let public_key = key.public_key()?;
    drop(key);
    let key_path = std::fs::canonicalize(&key_path)
        .map_err(|error| format!("resolve watch receipt key {}: {error}", key_path.display()))?;
    let normalized_input = normalize_batch_path(std::path::Path::new(&input))?;
    let normalized_output = normalize_batch_path(std::path::Path::new(&output))?;
    if key_path.starts_with(&normalized_input) || key_path.starts_with(&normalized_output) {
        return Err("watch receipt key must be outside the input and output trees".into());
    }
    let key_fingerprint = batch_resume::fingerprint_file(&key_path)?;
    processing.receipt_key = Some(key_path.to_string_lossy().into_owned());
    let processor_identity = watch_processor_identity(&processing, &public_key)?;
    let config = watch_config(&input, &output, &processing, &command, processor_identity);
    let settle_duration = config.settle_duration();
    let poll_interval = config.poll_interval();
    let mut watch = WatchFolder::open(config)?;
    CANCELLED.store(false, Ordering::SeqCst);
    install_cancel_handler()?;
    let run_cycle = |watch: &mut WatchFolder| {
        watch.cycle(|job| {
            process_watch_job(
                job,
                &processing,
                &key_path,
                key_fingerprint,
                &public_key,
                processor_identity,
            )
        })
    };

    if command.once {
        let first = run_cycle(&mut watch)?;
        print_watch_cycle(&input, &output, &processing, &first, false);
        if first.observed != 0 && settle_duration != Duration::ZERO {
            wait_watch_interval(settle_duration);
            if !CANCELLED.load(Ordering::SeqCst) {
                let second = run_cycle(&mut watch)?;
                print_watch_cycle(&input, &output, &processing, &second, false);
            }
        }
        return Ok(());
    }

    while !CANCELLED.load(Ordering::SeqCst) {
        match run_cycle(&mut watch) {
            Ok(report) => print_watch_cycle(&input, &output, &processing, &report, false),
            Err(error) => {
                if processing.json {
                    eprintln!("denoize: watch scan failed: {error}");
                } else {
                    eprintln!("denoize: watch scan failed; retrying: {error}");
                }
            }
        }
        wait_watch_interval(poll_interval);
    }
    print_watch_cycle(
        &input,
        &output,
        &processing,
        &WatchCycleReport::default(),
        true,
    );
    Ok(())
}

fn ipc_usage() -> &'static str {
    "\
USAGE:
    denoize ipc init --state-dir <DIR> --admin-grant <GRANT.json> [LIMITS]
    denoize ipc serve --state-dir <DIR> [--discovery <DISCOVERY.json>]
    denoize ipc ping --discovery <DISCOVERY.json> --grant <GRANT.json>
    denoize ipc dry-run <file|batch|stream> <INPUT> <OUTPUT> [CLIENT OPTIONS] [-- PROCESSING OPTIONS]
    denoize ipc submit <file|batch|stream> <INPUT> <OUTPUT> [CLIENT OPTIONS] [-- PROCESSING OPTIONS]
    denoize ipc status <JOB_ID> [CLIENT OPTIONS]
    denoize ipc list|history [--limit <N>] [CLIENT OPTIONS]
    denoize ipc cancel|pause|resume <JOB_ID> [CLIENT OPTIONS]
    denoize ipc grant create <POLICY.json> <GRANT.json> [CLIENT OPTIONS]
    denoize ipc grant revoke <GRANT_ID> [CLIENT OPTIONS]
    denoize ipc grant list [--limit <N>] [CLIENT OPTIONS]
    denoize ipc shutdown [--force] [CLIENT OPTIONS]

CLIENT OPTIONS:
    --discovery <PATH>        owner-private server discovery document
    --grant <PATH>            owner-private bearer capability document
    --priority <-100..100>    durable queue priority for dry-run/submit (default: 0)
    --pretty                  emit indented JSON instead of compact JSON

INIT LIMITS:
    --max-request-bytes <N>   framed request limit (default: 1048576)
    --max-response-bytes <N>  framed response limit (default: 16777216)
    --request-timeout-ms <N>  connection/request timeout (default: 900000)
    --planning-timeout-ms <N> bounded plan child timeout (default: 900000)
    --job-timeout-ms <N>      finite execution timeout (default: 86400000)
    --max-connections <N>     concurrent loopback connections (default: 8)
    --max-queue <N>           durable nonterminal jobs (default: 1024)
    --max-history <N>         terminal history records (default: 1024)
    --max-memory <MiB>        optional per-input denoize working-set limit
    --max-temp-space <MiB>    optional aggregate temporary-space limit
    --max-gpu-memory <MiB>    optional GPU-memory limit

The v1 service binds only 127.0.0.1, executes one finite job at a time, and
requires a capability for every request. Processing options begin after `--`;
server-controlled publication, receipt, isolation, model-path, and resource
options are rejected. File jobs are cancel-and-retry only; batch and stream
pause at verified durable checkpoint/publication boundaries.
"
}

fn run_ipc(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        if args.len() != 1 {
            return Err("ipc --help accepts no other arguments".into());
        }
        print!("{}", ipc_usage());
        return Ok(());
    }
    match args.first().map(String::as_str) {
        Some("init") => run_ipc_init(&args[1..]),
        Some("serve") => run_ipc_serve(&args[1..]),
        Some("ping") => run_ipc_simple(&args[1..], IpcOperation::Ping),
        Some("dry-run") => run_ipc_job(&args[1..], false),
        Some("submit") => run_ipc_job(&args[1..], true),
        Some("status") => run_ipc_job_id_command(&args[1..], "status"),
        Some("cancel") => run_ipc_job_id_command(&args[1..], "cancel"),
        Some("pause") => run_ipc_job_id_command(&args[1..], "pause"),
        Some("resume") => run_ipc_job_id_command(&args[1..], "resume"),
        Some("list") => run_ipc_list_command(&args[1..], false),
        Some("history") => run_ipc_list_command(&args[1..], true),
        Some("grant") => run_ipc_grant(&args[1..]),
        Some("shutdown") => run_ipc_shutdown(&args[1..]),
        Some(command) => Err(format!("unknown ipc command: {command}")),
        None => Err("ipc requires a command (run `denoize ipc --help`)".into()),
    }
}

fn run_ipc_init(args: &[String]) -> Result<(), String> {
    let mut state = None;
    let mut admin = None;
    let mut limits = IpcLimits::default();
    let mut index = 0;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--state-dir" => state = Some(ipc_parse_string(args, &mut index, option)?),
            "--admin-grant" => admin = Some(ipc_parse_string(args, &mut index, option)?),
            "--max-request-bytes" => {
                limits.max_request_bytes = ipc_parse_number(args, &mut index, option)?
            }
            "--max-response-bytes" => {
                limits.max_response_bytes = ipc_parse_number(args, &mut index, option)?
            }
            "--request-timeout-ms" => {
                limits.request_timeout_millis = ipc_parse_number(args, &mut index, option)?
            }
            "--planning-timeout-ms" => {
                limits.planning_timeout_millis = ipc_parse_number(args, &mut index, option)?
            }
            "--job-timeout-ms" => {
                limits.job_timeout_millis = ipc_parse_number(args, &mut index, option)?
            }
            "--max-connections" => {
                limits.max_connections = ipc_parse_number(args, &mut index, option)?
            }
            "--max-queue" => limits.max_queue_entries = ipc_parse_number(args, &mut index, option)?,
            "--max-history" => {
                limits.max_history_entries = ipc_parse_number(args, &mut index, option)?
            }
            "--max-memory" => {
                limits.max_memory_bytes = Some(ipc_parse_mib(args, &mut index, option)?)
            }
            "--max-temp-space" => {
                limits.max_temporary_bytes = Some(ipc_parse_mib(args, &mut index, option)?)
            }
            "--max-gpu-memory" => {
                limits.max_gpu_memory_bytes = Some(ipc_parse_mib(args, &mut index, option)?)
            }
            value => return Err(format!("unknown ipc init option: {value}")),
        }
        index += 1;
    }
    let state = state.ok_or("ipc init requires --state-dir")?;
    let admin = admin.ok_or("ipc init requires --admin-grant")?;
    limits.validate()?;
    let document = initialize_ipc_state(&state, &admin, limits)?;
    println!("initialized IPC state for server {}", document.server_id);
    println!("administrator grant: {admin}");
    Ok(())
}

fn run_ipc_serve(args: &[String]) -> Result<(), String> {
    let mut state = None;
    let mut discovery = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--state-dir" => state = Some(ipc_parse_string(args, &mut index, "--state-dir")?),
            "--discovery" => discovery = Some(ipc_parse_string(args, &mut index, "--discovery")?),
            value => return Err(format!("unknown ipc serve option: {value}")),
        }
        index += 1;
    }
    let state = std::path::PathBuf::from(state.ok_or("ipc serve requires --state-dir")?);
    let mut config = IpcServerConfig::new(state)?;
    if let Some(discovery) = discovery {
        config = config.with_discovery_file(discovery);
    }
    run_ipc_server(config)
}

fn run_ipc_job(args: &[String], submit: bool) -> Result<(), String> {
    let separator = args.iter().position(|argument| argument == "--");
    let (control, processing) = match separator {
        Some(position) => (&args[..position], args[position + 1..].to_vec()),
        None => (args, Vec::new()),
    };
    if control.len() < 3 {
        return Err("ipc dry-run/submit requires KIND INPUT OUTPUT".into());
    }
    let kind = match control[0].as_str() {
        "file" => IpcJobKind::File,
        "batch" => IpcJobKind::Batch,
        "stream" => IpcJobKind::Stream,
        value => return Err(format!("unknown IPC job kind: {value}")),
    };
    let input = canonical_cli_path(&control[1], "IPC input")?;
    let output = absolute_cli_path(&control[2], "IPC output")?;
    let mut common = IpcClientOptions::default();
    let mut priority = 0_i16;
    parse_ipc_client_options(&control[3..], &mut common, Some(&mut priority), None)?;
    let job = IpcJobSpec::new(kind, input, output)
        .with_arguments(processing)
        .with_priority(priority);
    job.validate()?;
    let operation = if submit {
        IpcOperation::Submit { job }
    } else {
        IpcOperation::DryRun { job }
    };
    let result = ipc_client(&common)?.request(operation)?;
    print_ipc_result(&result, common.pretty)
}

#[derive(Default)]
struct IpcClientOptions {
    discovery: Option<String>,
    grant: Option<String>,
    pretty: bool,
}

fn ipc_client(options: &IpcClientOptions) -> Result<IpcClient, String> {
    IpcClient::from_files(
        options
            .discovery
            .as_deref()
            .ok_or("IPC client command requires --discovery")?,
        options
            .grant
            .as_deref()
            .ok_or("IPC client command requires --grant")?,
    )
}

fn parse_ipc_client_options(
    args: &[String],
    options: &mut IpcClientOptions,
    mut priority: Option<&mut i16>,
    mut limit: Option<&mut u32>,
) -> Result<(), String> {
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--discovery" => {
                options.discovery = Some(ipc_parse_string(args, &mut index, "--discovery")?)
            }
            "--grant" => options.grant = Some(ipc_parse_string(args, &mut index, "--grant")?),
            "--pretty" => {
                if options.pretty {
                    return Err("--pretty may be supplied only once".into());
                }
                options.pretty = true;
            }
            "--priority" => {
                let target = priority
                    .as_deref_mut()
                    .ok_or("--priority is only valid for dry-run/submit")?;
                *target = ipc_parse_number(args, &mut index, "--priority")?;
            }
            "--limit" => {
                let target = limit
                    .as_deref_mut()
                    .ok_or("--limit is not valid for this IPC command")?;
                *target = ipc_parse_number(args, &mut index, "--limit")?;
            }
            value => return Err(format!("unknown IPC client option: {value}")),
        }
        index += 1;
    }
    Ok(())
}

fn run_ipc_simple(args: &[String], operation: IpcOperation) -> Result<(), String> {
    let mut options = IpcClientOptions::default();
    parse_ipc_client_options(args, &mut options, None, None)?;
    let result = ipc_client(&options)?.request(operation)?;
    print_ipc_result(&result, options.pretty)
}

fn run_ipc_job_id_command(args: &[String], command: &str) -> Result<(), String> {
    let job_id = args
        .first()
        .filter(|value| !value.starts_with('-'))
        .ok_or_else(|| format!("ipc {command} requires JOB_ID"))?
        .clone();
    let mut options = IpcClientOptions::default();
    parse_ipc_client_options(&args[1..], &mut options, None, None)?;
    let operation = match command {
        "status" => IpcOperation::Status { job_id },
        "cancel" => IpcOperation::Cancel { job_id },
        "pause" => IpcOperation::Pause { job_id },
        "resume" => IpcOperation::Resume { job_id },
        _ => return Err(format!("unsupported IPC job command: {command}")),
    };
    let result = ipc_client(&options)?.request(operation)?;
    print_ipc_result(&result, options.pretty)
}

fn run_ipc_list_command(args: &[String], history: bool) -> Result<(), String> {
    let mut options = IpcClientOptions::default();
    let mut limit = 100_u32;
    parse_ipc_client_options(args, &mut options, None, Some(&mut limit))?;
    if !(1..=10_000).contains(&limit) {
        return Err("IPC --limit must be in 1..=10000".into());
    }
    let operation = if history {
        IpcOperation::History { limit }
    } else {
        IpcOperation::List { limit }
    };
    let result = ipc_client(&options)?.request(operation)?;
    print_ipc_result(&result, options.pretty)
}

fn run_ipc_grant(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("create") => {
            let policy_path = args
                .get(1)
                .filter(|value| !value.starts_with('-'))
                .ok_or("ipc grant create requires POLICY.json")?;
            let output_path = args
                .get(2)
                .filter(|value| !value.starts_with('-'))
                .ok_or("ipc grant create requires output GRANT.json")?;
            let policy = read_ipc_policy(policy_path)?;
            let mut options = IpcClientOptions::default();
            parse_ipc_client_options(&args[3..], &mut options, None, None)?;
            let result = ipc_client(&options)?.request(IpcOperation::CreateGrant { policy })?;
            let IpcResponseResult::Grant(document) = result else {
                return Err("IPC server returned the wrong grant-create response".into());
            };
            write_ipc_grant(output_path, &document)?;
            println!("created IPC grant {}: {output_path}", document.grant_id);
            Ok(())
        }
        Some("revoke") => {
            let grant_id = args
                .get(1)
                .filter(|value| !value.starts_with('-'))
                .ok_or("ipc grant revoke requires GRANT_ID")?
                .clone();
            let mut options = IpcClientOptions::default();
            parse_ipc_client_options(&args[2..], &mut options, None, None)?;
            let result = ipc_client(&options)?.request(IpcOperation::RevokeGrant { grant_id })?;
            print_ipc_result(&result, options.pretty)
        }
        Some("list") => {
            let mut options = IpcClientOptions::default();
            let mut limit = 100_u32;
            parse_ipc_client_options(
                args.get(1..).unwrap_or_default(),
                &mut options,
                None,
                Some(&mut limit),
            )?;
            if !(1..=10_000).contains(&limit) {
                return Err("IPC --limit must be in 1..=10000".into());
            }
            let result = ipc_client(&options)?.request(IpcOperation::ListGrants { limit })?;
            print_ipc_result(&result, options.pretty)
        }
        Some(command) => Err(format!("unknown ipc grant command: {command}")),
        None => Err("ipc grant requires create, revoke, or list".into()),
    }
}

fn run_ipc_shutdown(args: &[String]) -> Result<(), String> {
    let mut force = false;
    let mut filtered = Vec::new();
    for argument in args {
        if argument == "--force" {
            if force {
                return Err("ipc shutdown accepts --force only once".into());
            }
            force = true;
        } else {
            filtered.push(argument.clone());
        }
    }
    let mut options = IpcClientOptions::default();
    parse_ipc_client_options(&filtered, &mut options, None, None)?;
    let result = ipc_client(&options)?.request(IpcOperation::Shutdown { force })?;
    print_ipc_result(&result, options.pretty)
}

fn read_ipc_policy(path: &str) -> Result<IpcGrantPolicy, String> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOCTTY);
    }
    let file = options
        .open(path)
        .map_err(|error| format!("open IPC grant policy {path}: {error}"))?;
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle as _;
        use windows_sys::Win32::Storage::FileSystem::{GetFileType, FILE_TYPE_DISK};
        if unsafe { GetFileType(file.as_raw_handle()) } != FILE_TYPE_DISK {
            return Err("IPC grant policy must be a regular disk file".into());
        }
    }
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect IPC grant policy {path}: {error}"))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > 1024 * 1024 {
        return Err("IPC grant policy must be a nonempty regular file of at most 1 MiB".into());
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(metadata.len() as usize)
        .map_err(|error| format!("reserve IPC grant policy: {error}"))?;
    file.take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read IPC grant policy {path}: {error}"))?;
    if bytes.len() > 1024 * 1024 {
        return Err("IPC grant policy exceeds its 1 MiB limit".into());
    }
    let policy: IpcGrantPolicy = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse IPC grant policy {path}: {error}"))?;
    policy.validate()?;
    Ok(policy)
}

#[cfg(test)]
mod ipc_policy_tests {
    use super::*;

    #[test]
    fn regular_ipc_policy_is_read_from_one_bounded_handle() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("policy.json");
        std::fs::write(
            &path,
            br#"{
                "label":"worker",
                "capabilities":["plan"],
                "input_roots":["/input"],
                "output_roots":["/output"],
                "max_priority":0,
                "expires_at_unix_millis":null
            }"#,
        )
        .unwrap();
        let policy = read_ipc_policy(path.to_str().unwrap()).unwrap();
        assert_eq!(policy.label, "worker");
        assert_eq!(policy.capabilities, vec![denoize::ipc::IpcCapability::Plan]);
    }

    #[cfg(unix)]
    #[test]
    fn fifo_ipc_policy_is_rejected_without_waiting_for_a_writer() {
        use std::ffi::CString;
        use std::time::{Duration, Instant};

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("policy.fifo");
        let encoded = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(encoded.as_ptr(), 0o600) }, 0);
        let started = Instant::now();
        let error = read_ipc_policy(path.to_str().unwrap()).unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(error.contains("regular file"), "{error}");
    }
}

fn write_ipc_grant(path: &str, document: &IpcGrantDocument) -> Result<(), String> {
    let mut bytes = Zeroizing::new(
        serde_json::to_vec_pretty(document)
            .map_err(|error| format!("serialize IPC capability grant: {error}"))?,
    );
    bytes.push(b'\n');
    let mut output = AtomicOutput::new_private(path)?;
    output
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| format!("write IPC capability grant {path}: {error}"))?;
    output.commit(CommitMode::NoClobber)
}

fn print_ipc_result(result: &IpcResponseResult, pretty: bool) -> Result<(), String> {
    let encoded = if pretty {
        serde_json::to_string_pretty(result)
    } else {
        serde_json::to_string(result)
    }
    .map_err(|error| format!("serialize IPC result: {error}"))?;
    println!("{encoded}");
    Ok(())
}

fn ipc_parse_string(args: &[String], index: &mut usize, option: &str) -> Result<String, String> {
    *index = index.checked_add(1).ok_or("IPC argument index overflow")?;
    args.get(*index)
        .filter(|value| !value.is_empty())
        .cloned()
        .ok_or_else(|| format!("missing value for {option}"))
}

fn ipc_parse_number<T>(args: &[String], index: &mut usize, option: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = ipc_parse_string(args, index, option)?;
    value
        .parse::<T>()
        .map_err(|error| format!("invalid value for {option}: {error}"))
}

fn ipc_parse_mib(args: &[String], index: &mut usize, option: &str) -> Result<u64, String> {
    let mib: u64 = ipc_parse_number(args, index, option)?;
    mib.checked_mul(BYTES_PER_MIB)
        .ok_or_else(|| format!("{option} byte conversion overflows"))
}

fn canonical_cli_path(value: &str, label: &str) -> Result<String, String> {
    let resolved = std::fs::canonicalize(value)
        .map_err(|error| format!("resolve {label} {value}: {error}"))?;
    Ok(resolved.to_string_lossy().into_owned())
}

fn absolute_cli_path(value: &str, label: &str) -> Result<String, String> {
    let path = std::path::PathBuf::from(value);
    let absolute = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .map_err(|error| format!("resolve current directory for {label}: {error}"))?
            .join(path)
    };
    Ok(absolute.to_string_lossy().into_owned())
}

fn plugin_usage() -> &'static str {
    "\
USAGE:
    denoize plugin info [--json|--pretty]
    denoize plugin latency [--sample-rate <HZ>] [--json|--pretty]
    denoize plugin neural info [--sample-rate <HZ>] [--json|--pretty]
    denoize plugin neural latency [--sample-rate <HZ>] [--json|--pretty]
    denoize plugin neural session create <OUTPUT.json> [OPTIONS]
    denoize plugin neural session inspect|validate <SESSION.json> [--json|--pretty]
    denoize plugin preset create <speech|gentle|music> <OUTPUT.json> [OPTIONS]
    denoize plugin preset inspect|validate <PRESET.json> [--json|--pretty]
    denoize plugin session create <PRESET.json> <OUTPUT.json> [--mono|--stereo] [OPTIONS]
    denoize plugin session inspect|validate <SESSION.json> [--json|--pretty]

PRESET CREATE OPTIONS:
    --name <NAME>             portable preset display name
    --amount <0..1>           suppression amount
    --threshold-dbfs <-96..-18>
    --release-ms <20..1000>
    --mix <0..1>
    --output-gain-db <-24..24>
    --bypass|--no-bypass
    --stereo-link|--no-stereo-link
    --replace                 atomically replace an existing output
    --json|--pretty           print the created contract as JSON

SESSION CREATE OPTIONS:
    --mono|--stereo           restored port layout (default: stereo)
    --replace                 atomically replace an existing output
    --json|--pretty           print the created contract as JSON

NEURAL SESSION CREATE OPTIONS:
    --mono|--stereo           main and reserved-reference layout (default: stereo)
    --mix <0..1>
    --output-gain-db <-24..24>
    --fallback <delayed-dry|last-safe-gain|silence>
    --bypass|--no-bypass
    --replace                 atomically replace an existing output
    --json|--pretty           print the created contract as JSON

CLAP state and these JSON contracts use the same stable parameter IDs, fixed
latency policies, and deterministic compact serialization."
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PluginOutputMode {
    Human,
    Json,
    Pretty,
}

fn parse_plugin_output_mode(args: &[String], command: &str) -> Result<PluginOutputMode, String> {
    match args {
        [] => Ok(PluginOutputMode::Human),
        [flag] if flag == "--json" => Ok(PluginOutputMode::Json),
        [flag] if flag == "--pretty" => Ok(PluginOutputMode::Pretty),
        _ => Err(format!("{command} accepts only one of --json or --pretty")),
    }
}

fn print_plugin_json<T: Serialize>(value: &T, mode: PluginOutputMode) -> Result<(), String> {
    let encoded = match mode {
        PluginOutputMode::Pretty => serde_json::to_string_pretty(value),
        PluginOutputMode::Human | PluginOutputMode::Json => serde_json::to_string(value),
    }
    .map_err(|error| format!("serialize DAW plug-in output: {error}"))?;
    println!("{encoded}");
    Ok(())
}

fn run_plugin(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        if args.len() != 1 {
            return Err("plugin --help accepts no other arguments".into());
        }
        println!("{}", plugin_usage());
        return Ok(());
    }
    match args.first().map(String::as_str) {
        Some("info") => run_plugin_info(&args[1..]),
        Some("latency") => run_plugin_latency(&args[1..]),
        Some("neural") => run_neural_plugin(&args[1..]),
        Some("preset") => run_plugin_preset(&args[1..]),
        Some("session") => run_plugin_session(&args[1..]),
        Some(command) => Err(format!("unknown plugin command: {command}")),
        None => Err("plugin requires a command (run `denoize plugin --help`)".into()),
    }
}

fn run_neural_plugin(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("info") => run_neural_plugin_info(&args[1..]),
        Some("latency") => run_neural_plugin_latency(&args[1..]),
        Some("session") => run_neural_plugin_session(&args[1..]),
        Some(command) => Err(format!("unknown plugin neural command: {command}")),
        None => Err("plugin neural requires info, latency, or session".into()),
    }
}

fn parse_neural_sample_rate_output(
    args: &[String],
    command: &str,
) -> Result<(f64, PluginOutputMode), String> {
    let mut sample_rate = 48_000.0_f64;
    let mut sample_rate_seen = false;
    let mut output = PluginOutputMode::Human;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--sample-rate" if !sample_rate_seen => {
                sample_rate_seen = true;
                index += 1;
                sample_rate = args
                    .get(index)
                    .ok_or_else(|| format!("{command} requires a value for --sample-rate"))?
                    .parse::<f64>()
                    .map_err(|error| format!("invalid --sample-rate: {error}"))?;
            }
            "--sample-rate" => {
                return Err(format!("{command} accepts --sample-rate only once"));
            }
            "--json" if output == PluginOutputMode::Human => output = PluginOutputMode::Json,
            "--pretty" if output == PluginOutputMode::Human => output = PluginOutputMode::Pretty,
            "--json" | "--pretty" => {
                return Err(format!("{command} accepts only one of --json or --pretty"));
            }
            option => return Err(format!("unknown {command} option: {option}")),
        }
        index += 1;
    }
    // Validate before any managed-model cache inspection.
    neural_daw_chunk_frames(sample_rate)?;
    Ok((sample_rate, output))
}

fn run_neural_plugin_info(args: &[String]) -> Result<(), String> {
    let (sample_rate, output) = parse_neural_sample_rate_output(args, "plugin neural info")?;
    let chunk_frames = neural_daw_chunk_frames(sample_rate)?;
    let latency_frames = neural_daw_latency_frames(sample_rate)?;
    let latency_millis = neural_daw_latency_millis(sample_rate)?;
    let model_installed = denoize::models::MODELS
        .iter()
        .find(|model| {
            model.name == NEURAL_DAW_MODEL_ID
                && model.backend == "gtcrn"
                && model.sha256 == NEURAL_DAW_MODEL_SHA256
        })
        .is_some_and(|model| denoize::models::verify(model).is_ok());
    let report = serde_json::json!({
        "schema": CLI_JSON_SCHEMA,
        "schema_version": CLI_JSON_SCHEMA_VERSION,
        "event": "plugin-neural-info",
        "plugin_id": NEURAL_DAW_PLUGIN_ID,
        "name": "denoize Neural",
        "version": VERSION,
        "format": "CLAP",
        "backend": "gtcrn",
        "model_id": NEURAL_DAW_MODEL_ID,
        "model_sha256": NEURAL_DAW_MODEL_SHA256,
        "model_installed": model_installed,
        "port_configurations": ["mono", "stereo"],
        "reference_port": "reserved-independent-input",
        "sample_formats": ["f32", "f64"],
        "sample_rate": sample_rate,
        "chunk_frames": chunk_frames,
        "latency_policy": NEURAL_DAW_LATENCY_POLICY,
        "latency_frames": latency_frames,
        "latency_millis": latency_millis,
        "queue_blocks": NEURAL_DAW_QUEUE_BLOCKS,
        "overload_fallbacks": ["delayed-dry", "last-safe-gain", "silence"],
        "realtime_contract": {
            "allocations": 0,
            "locks": 0,
            "file_io": false,
            "network_io": false,
            "logging": false,
            "worker_waits": false,
            "inference_on_callback": false
        }
    });
    if output != PluginOutputMode::Human {
        return print_plugin_json(&report, output);
    }
    println!("denoize Neural {VERSION} CLAP ({NEURAL_DAW_PLUGIN_ID})");
    println!(
        "model: {NEURAL_DAW_MODEL_ID} ({})",
        if model_installed {
            "verified"
        } else {
            "not installed; run `denoize models install gtcrn`"
        }
    );
    println!("ports: mono/stereo main + reserved reference; samples: f32, f64");
    println!(
        "latency: {latency_frames} frames ({latency_millis:.6} ms at {sample_rate:.0} Hz; {NEURAL_DAW_LATENCY_POLICY})"
    );
    println!("audio callback: bounded lock-free queues; zero allocation, locks, I/O, logging, inference, or worker waits");
    Ok(())
}

fn run_neural_plugin_latency(args: &[String]) -> Result<(), String> {
    let (sample_rate, output) = parse_neural_sample_rate_output(args, "plugin neural latency")?;
    let reported_frames = neural_daw_latency_frames(sample_rate)?;
    let mut delay = vec![0.0_f64; reported_frames as usize];
    let mut cursor = 0usize;
    let mut measured_frames = None;
    for frame in 0..=reported_frames.saturating_add(1) {
        let input = if frame == 0 { 1.0 } else { 0.0 };
        let delayed = delay[cursor];
        delay[cursor] = input;
        cursor += 1;
        if cursor == delay.len() {
            cursor = 0;
        }
        if delayed != 0.0 {
            measured_frames = Some(frame);
            break;
        }
    }
    let measured_frames =
        measured_frames.ok_or("neural DAW delayed-dry impulse measurement produced no output")?;
    if measured_frames != reported_frames {
        return Err(format!(
            "neural DAW measured latency {measured_frames} differs from reported latency {reported_frames} frames"
        ));
    }
    let report = serde_json::json!({
        "schema": CLI_JSON_SCHEMA,
        "schema_version": CLI_JSON_SCHEMA_VERSION,
        "event": "plugin-neural-latency",
        "plugin_id": NEURAL_DAW_PLUGIN_ID,
        "latency_policy": NEURAL_DAW_LATENCY_POLICY,
        "sample_rate": sample_rate,
        "chunk_frames": neural_daw_chunk_frames(sample_rate)?,
        "latency_frames": reported_frames,
        "latency_millis": neural_daw_latency_millis(sample_rate)?,
        "measured_latency_frames": measured_frames,
        "matches_reported": true,
        "measurement": "f64-delayed-dry-impulse-v1"
    });
    if output != PluginOutputMode::Human {
        return print_plugin_json(&report, output);
    }
    println!(
        "{sample_rate:.0} Hz: {reported_frames} frames ({:.6} ms; {NEURAL_DAW_LATENCY_POLICY})",
        neural_daw_latency_millis(sample_rate)?
    );
    println!("measured delayed-dry impulse: {measured_frames} frames (matches reported)");
    Ok(())
}

fn run_neural_plugin_session(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("create") => run_neural_plugin_session_create(&args[1..]),
        Some("inspect") => run_neural_plugin_session_read(&args[1..], false),
        Some("validate") => run_neural_plugin_session_read(&args[1..], true),
        Some(command) => Err(format!("unknown plugin neural session command: {command}")),
        None => Err("plugin neural session requires create, inspect, or validate".into()),
    }
}

fn run_neural_plugin_session_create(args: &[String]) -> Result<(), String> {
    let path = args
        .first()
        .ok_or("plugin neural session create requires OUTPUT.json")?;
    let mut port_configuration = NeuralDawPortConfiguration::Stereo;
    let mut parameters = NeuralDawParameters::default();
    let mut mode = CommitMode::NoClobber;
    let mut output = PluginOutputMode::Human;
    let mut seen = BTreeSet::<&str>::new();
    let mut index = 1usize;
    while index < args.len() {
        let option = args[index].as_str();
        let unique = matches!(
            option,
            "--mono"
                | "--stereo"
                | "--mix"
                | "--output-gain-db"
                | "--fallback"
                | "--bypass"
                | "--no-bypass"
                | "--replace"
        );
        if unique && !seen.insert(option) {
            return Err(format!("{option} may be supplied only once"));
        }
        match option {
            "--mono" if !seen.contains("--stereo") => {
                port_configuration = NeuralDawPortConfiguration::Mono
            }
            "--stereo" if !seen.contains("--mono") => {
                port_configuration = NeuralDawPortConfiguration::Stereo
            }
            "--mono" | "--stereo" => {
                return Err("neural session accepts only one of --mono or --stereo".into());
            }
            "--mix" => parameters.mix = plugin_number_value(args, &mut index, option)?,
            "--output-gain-db" => {
                parameters.output_gain_db = plugin_number_value(args, &mut index, option)?
            }
            "--fallback" => {
                let value = plugin_option_value(args, &mut index, option)?;
                parameters.overload_fallback = NeuralDawOverloadFallback::parse(&value)
                    .ok_or("--fallback must be delayed-dry, last-safe-gain, or silence")?;
            }
            "--bypass" if !seen.contains("--no-bypass") => parameters.bypass = true,
            "--no-bypass" if !seen.contains("--bypass") => parameters.bypass = false,
            "--bypass" | "--no-bypass" => {
                return Err("neural session accepts only one of --bypass or --no-bypass".into());
            }
            "--replace" => mode = CommitMode::Replace,
            "--json" if output == PluginOutputMode::Human => output = PluginOutputMode::Json,
            "--pretty" if output == PluginOutputMode::Human => output = PluginOutputMode::Pretty,
            "--json" | "--pretty" => {
                return Err("neural session accepts only one of --json or --pretty".into());
            }
            value => return Err(format!("unknown plugin neural session option: {value}")),
        }
        index += 1;
    }
    parameters.validate()?;
    let state = NeuralDawSessionState::new(port_configuration, parameters)?;
    write_neural_daw_session(path, &state, mode)?;
    if output != PluginOutputMode::Human {
        print_plugin_json(&state, output)
    } else {
        println!(
            "created {:?} neural DAW session: {path}",
            port_configuration
        );
        Ok(())
    }
}

fn run_neural_plugin_session_read(args: &[String], validate: bool) -> Result<(), String> {
    let path = args
        .first()
        .ok_or("plugin neural session inspect/validate requires SESSION.json")?;
    let output = parse_plugin_output_mode(
        &args[1..],
        if validate {
            "plugin neural session validate"
        } else {
            "plugin neural session inspect"
        },
    )?;
    let state = read_neural_daw_session(path)?;
    state.validate_for_model(NeuralDawModel::Gtcrn)?;
    if output != PluginOutputMode::Human {
        if validate {
            return print_plugin_json(
                &serde_json::json!({
                    "schema": CLI_JSON_SCHEMA,
                    "schema_version": CLI_JSON_SCHEMA_VERSION,
                    "event": "plugin-neural-session-validation",
                    "valid": true,
                    "path": path,
                    "plugin_id": state.plugin_id,
                    "model_id": state.model_id,
                    "model_sha256": state.model_sha256,
                    "port_configuration": state.port_configuration,
                    "latency_policy": state.latency_policy
                }),
                output,
            );
        }
        return print_plugin_json(&state, output);
    }
    if validate {
        println!("valid neural DAW session: {path}");
    } else {
        println!("plugin: {}", state.plugin_id);
        println!("model: {} ({})", state.model_id, state.model_sha256);
        println!(
            "ports: {:?}; latency: {}",
            state.port_configuration, state.latency_policy
        );
        println!(
            "bypass={} mix={:.3} output={:.1} dB fallback={:?}",
            state.parameters.bypass,
            state.parameters.mix,
            state.parameters.output_gain_db,
            state.parameters.overload_fallback
        );
    }
    Ok(())
}

fn run_plugin_info(args: &[String]) -> Result<(), String> {
    let mode = parse_plugin_output_mode(args, "plugin info")?;
    let report = serde_json::json!({
        "schema": CLI_JSON_SCHEMA,
        "schema_version": CLI_JSON_SCHEMA_VERSION,
        "event": "plugin-info",
        "plugin_id": DAW_PLUGIN_ID,
        "name": "denoize",
        "version": VERSION,
        "format": "CLAP",
        "port_configurations": ["mono", "stereo"],
        "sample_formats": ["f32", "f64"],
        "factory_presets": ["speech", "gentle", "music"],
        "latency_policy": DAW_LATENCY_POLICY,
        "latency_millis": DAW_FIXED_LATENCY_MILLIS,
        "realtime_contract": {
            "allocations": 0,
            "locks": 0,
            "file_io": false,
            "system_calls": false
        }
    });
    if mode != PluginOutputMode::Human {
        return print_plugin_json(&report, mode);
    }
    println!("denoize {VERSION} CLAP ({DAW_PLUGIN_ID})");
    println!("ports: mono, stereo; samples: f32, f64");
    println!("latency: fixed {DAW_FIXED_LATENCY_MILLIS:.1} ms ({DAW_LATENCY_POLICY})");
    println!("factory presets: speech, gentle, music");
    println!("audio callback: zero allocations, locks, file I/O, or system calls");
    Ok(())
}

fn run_plugin_latency(args: &[String]) -> Result<(), String> {
    let mut sample_rate = 48_000.0_f64;
    let mut output = PluginOutputMode::Human;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--sample-rate" => {
                index += 1;
                sample_rate = args
                    .get(index)
                    .ok_or("plugin latency requires a value for --sample-rate")?
                    .parse::<f64>()
                    .map_err(|error| format!("invalid --sample-rate: {error}"))?;
            }
            "--json" if output == PluginOutputMode::Human => output = PluginOutputMode::Json,
            "--pretty" if output == PluginOutputMode::Human => output = PluginOutputMode::Pretty,
            "--json" | "--pretty" => {
                return Err("plugin latency accepts only one of --json or --pretty".into())
            }
            option => return Err(format!("unknown plugin latency option: {option}")),
        }
        index += 1;
    }
    let mut processor = DawRealtimeProcessor::new(sample_rate, 1)?;
    let reported_frames = processor.latency_frames();
    let runtime = processor.prepare_parameters(&DawParameters {
        bypass: true,
        ..DawParameters::default()
    })?;
    let mut measured_frames = None;
    for frame in 0..=reported_frames.saturating_add(1) {
        let input = if frame == 0 { 1.0 } else { 0.0 };
        if processor.process_frame_f64([input, 0.0], &runtime)[0] != 0.0 {
            measured_frames = Some(frame);
            break;
        }
    }
    let measured_frames =
        measured_frames.ok_or("DAW impulse latency measurement produced no output")?;
    let matches_reported = measured_frames == reported_frames;
    if !matches_reported {
        return Err(format!(
            "DAW measured latency {measured_frames} frames differs from reported latency {reported_frames} frames"
        ));
    }
    let report = serde_json::json!({
        "schema": CLI_JSON_SCHEMA,
        "schema_version": CLI_JSON_SCHEMA_VERSION,
        "event": "plugin-latency",
        "plugin_id": DAW_PLUGIN_ID,
        "latency_policy": DAW_LATENCY_POLICY,
        "sample_rate": sample_rate,
        "latency_frames": reported_frames,
        "latency_millis": processor.latency_millis(),
        "measured_latency_frames": measured_frames,
        "matches_reported": matches_reported,
        "measurement": "f64-bypass-impulse-v1"
    });
    if output != PluginOutputMode::Human {
        return print_plugin_json(&report, output);
    }
    println!(
        "{:.6} Hz: {} frames ({:.6} ms; {})",
        sample_rate,
        reported_frames,
        processor.latency_millis(),
        DAW_LATENCY_POLICY
    );
    println!("measured impulse: {measured_frames} frames (matches reported)");
    Ok(())
}

fn run_plugin_preset(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("create") => run_plugin_preset_create(&args[1..]),
        Some("inspect") => run_plugin_preset_read(&args[1..], false),
        Some("validate") => run_plugin_preset_read(&args[1..], true),
        Some(command) => Err(format!("unknown plugin preset command: {command}")),
        None => Err("plugin preset requires create, inspect, or validate".into()),
    }
}

fn run_plugin_preset_create(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("plugin preset create requires FACTORY and OUTPUT.json".into());
    }
    let factory = &args[0];
    let path = &args[1];
    let mut preset = DawPreset::factory(factory).ok_or_else(|| {
        format!("unknown DAW factory preset {factory}; expected speech, gentle, or music")
    })?;
    let mut mode = CommitMode::NoClobber;
    let mut output = PluginOutputMode::Human;
    let mut index = 2;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--name" => preset.name = plugin_option_value(args, &mut index, option)?,
            "--amount" => preset.parameters.amount = plugin_number_value(args, &mut index, option)?,
            "--threshold-dbfs" => {
                preset.parameters.threshold_dbfs = plugin_number_value(args, &mut index, option)?
            }
            "--release-ms" => {
                preset.parameters.release_ms = plugin_number_value(args, &mut index, option)?
            }
            "--mix" => preset.parameters.mix = plugin_number_value(args, &mut index, option)?,
            "--output-gain-db" => {
                preset.parameters.output_gain_db = plugin_number_value(args, &mut index, option)?
            }
            "--bypass" => preset.parameters.bypass = true,
            "--no-bypass" => preset.parameters.bypass = false,
            "--stereo-link" => preset.parameters.stereo_link = true,
            "--no-stereo-link" => preset.parameters.stereo_link = false,
            "--replace" if mode == CommitMode::NoClobber => mode = CommitMode::Replace,
            "--replace" => return Err("--replace may be supplied only once".into()),
            "--json" if output == PluginOutputMode::Human => output = PluginOutputMode::Json,
            "--pretty" if output == PluginOutputMode::Human => output = PluginOutputMode::Pretty,
            "--json" | "--pretty" => {
                return Err("preset create accepts only one of --json or --pretty".into())
            }
            value => return Err(format!("unknown plugin preset create option: {value}")),
        }
        index += 1;
    }
    preset.validate()?;
    write_daw_preset(path, &preset, mode)?;
    if output != PluginOutputMode::Human {
        print_plugin_json(&preset, output)
    } else {
        println!("created DAW preset {}: {path}", preset.name);
        Ok(())
    }
}

fn run_plugin_preset_read(args: &[String], validate: bool) -> Result<(), String> {
    let path = args
        .first()
        .ok_or("plugin preset inspect/validate requires PRESET.json")?;
    let mode = parse_plugin_output_mode(
        &args[1..],
        if validate {
            "plugin preset validate"
        } else {
            "plugin preset inspect"
        },
    )?;
    let preset = read_daw_preset(path)?;
    if mode != PluginOutputMode::Human {
        if validate {
            return print_plugin_json(
                &serde_json::json!({
                    "schema": CLI_JSON_SCHEMA,
                    "schema_version": CLI_JSON_SCHEMA_VERSION,
                    "event": "plugin-preset-validation",
                    "valid": true,
                    "path": path,
                    "plugin_id": preset.plugin_id,
                    "name": preset.name
                }),
                mode,
            );
        }
        return print_plugin_json(&preset, mode);
    }
    if validate {
        println!("valid DAW preset: {path}");
    } else {
        print_preset_summary(&preset);
    }
    Ok(())
}

fn print_preset_summary(preset: &DawPreset) {
    let parameters = preset.parameters;
    println!("preset: {}", preset.name);
    println!("plugin: {}", preset.plugin_id);
    println!(
        "amount={:.3} threshold={:.1} dBFS release={:.1} ms mix={:.3} output={:.1} dB",
        parameters.amount,
        parameters.threshold_dbfs,
        parameters.release_ms,
        parameters.mix,
        parameters.output_gain_db
    );
    println!(
        "bypass={} stereo-link={}",
        parameters.bypass, parameters.stereo_link
    );
}

fn run_plugin_session(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("create") => run_plugin_session_create(&args[1..]),
        Some("inspect") => run_plugin_session_read(&args[1..], false),
        Some("validate") => run_plugin_session_read(&args[1..], true),
        Some(command) => Err(format!("unknown plugin session command: {command}")),
        None => Err("plugin session requires create, inspect, or validate".into()),
    }
}

fn run_plugin_session_create(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("plugin session create requires PRESET.json and OUTPUT.json".into());
    }
    let preset_path = &args[0];
    let output_path = &args[1];
    let mut configuration = DawPortConfiguration::Stereo;
    let mut configuration_selected = false;
    let mut mode = CommitMode::NoClobber;
    let mut output = PluginOutputMode::Human;
    for option in &args[2..] {
        match option.as_str() {
            "--mono" if !configuration_selected => {
                configuration = DawPortConfiguration::Mono;
                configuration_selected = true;
            }
            "--stereo" if !configuration_selected => {
                configuration = DawPortConfiguration::Stereo;
                configuration_selected = true;
            }
            "--mono" | "--stereo" => {
                return Err("session create accepts only one of --mono or --stereo".into())
            }
            "--replace" if mode == CommitMode::NoClobber => mode = CommitMode::Replace,
            "--replace" => return Err("--replace may be supplied only once".into()),
            "--json" if output == PluginOutputMode::Human => output = PluginOutputMode::Json,
            "--pretty" if output == PluginOutputMode::Human => output = PluginOutputMode::Pretty,
            "--json" | "--pretty" => {
                return Err("session create accepts only one of --json or --pretty".into())
            }
            value => return Err(format!("unknown plugin session create option: {value}")),
        }
    }
    let state = DawSessionState::new(read_daw_preset(preset_path)?, configuration)?;
    write_daw_session(output_path, &state, mode)?;
    if output != PluginOutputMode::Human {
        print_plugin_json(&state, output)
    } else {
        println!(
            "created deterministic {:?} DAW session: {output_path}",
            configuration
        );
        Ok(())
    }
}

fn run_plugin_session_read(args: &[String], validate: bool) -> Result<(), String> {
    let path = args
        .first()
        .ok_or("plugin session inspect/validate requires SESSION.json")?;
    let mode = parse_plugin_output_mode(
        &args[1..],
        if validate {
            "plugin session validate"
        } else {
            "plugin session inspect"
        },
    )?;
    let state = read_daw_session(path)?;
    if mode != PluginOutputMode::Human {
        if validate {
            return print_plugin_json(
                &serde_json::json!({
                    "schema": CLI_JSON_SCHEMA,
                    "schema_version": CLI_JSON_SCHEMA_VERSION,
                    "event": "plugin-session-validation",
                    "valid": true,
                    "path": path,
                    "plugin_id": state.plugin_id,
                    "port_configuration": state.port_configuration,
                    "latency_policy": state.latency_policy
                }),
                mode,
            );
        }
        return print_plugin_json(&state, mode);
    }
    if validate {
        println!("valid deterministic DAW session: {path}");
    } else {
        println!("session: {:?}", state.port_configuration);
        println!("latency: {}", state.latency_policy);
        print_preset_summary(&state.preset);
    }
    Ok(())
}

fn plugin_option_value(args: &[String], index: &mut usize, option: &str) -> Result<String, String> {
    *index = index
        .checked_add(1)
        .ok_or("plugin argument index overflow")?;
    args.get(*index)
        .filter(|value| !value.is_empty())
        .cloned()
        .ok_or_else(|| format!("missing value for {option}"))
}

fn plugin_number_value<T>(args: &[String], index: &mut usize, option: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    plugin_option_value(args, index, option)?
        .parse()
        .map_err(|error| format!("invalid value for {option}: {error}"))
}

fn update_usage() -> &'static str {
    "\
Recoverable signed application updates

USAGE:
    denoize update manifest verify <MANIFEST.json> <MANIFEST.sig> [--public-key PATH] [--pretty]
    denoize update bundle inspect <BUNDLE.dub> [--public-key PATH] [--pretty]
    denoize update bundle download <OUTPUT.dub> --platform ID --from-version VERSION \\
        [--manifest-url URL --signature-url URL] [--public-key PATH] [--pretty]
    denoize update bundle build <OUTPUT.dub> --manifest PATH --signature PATH \\
        --platform ID --from-version VERSION --candidate-artifact PATH \\
        --candidate-sbom PATH --candidate-provenance PATH --rollback-artifact PATH \\
        --rollback-sbom PATH --rollback-provenance PATH [--public-key PATH] [--pretty]
    denoize update check <MANIFEST.json> <MANIFEST.sig> --state-dir DIR \\
        --channel CHANNEL --platform ID --current-version VERSION [--public-key PATH] [--pretty]
    denoize update check-online --state-dir DIR --channel CHANNEL --platform ID \\
        --current-version VERSION [--manifest-url URL --signature-url URL] \\
        [--public-key PATH] [--pretty]
    denoize update dry-run <BUNDLE.dub> --state-dir DIR --current-version VERSION \\
        [--max-staging-bytes N] [--public-key PATH] [--pretty]
    denoize update apply <BUNDLE.dub> --state-dir DIR --current-version VERSION \\
        [--max-staging-bytes N] [--public-key PATH] [--pretty]
    denoize update status --state-dir DIR [--pretty]
    denoize update health begin --state-dir DIR --running-version VERSION [--pretty]
    denoize update health confirm --state-dir DIR --running-version VERSION --token TOKEN [--pretty]
    denoize update recover --state-dir DIR [--reason CODE] [--pretty]

All successful commands emit one versioned JSON document. `check` and `dry-run`
are read-only. `apply` stages the authenticated candidate and an offline
last-known-good installation, then waits for explicit startup health confirmation.
Recovery never lowers the accepted-version floor and never requires a network.
"
}

fn update_option_value(args: &[String], index: &mut usize, option: &str) -> Result<String, String> {
    *index = index
        .checked_add(1)
        .ok_or("update argument index overflow")?;
    args.get(*index)
        .filter(|value| !value.is_empty())
        .cloned()
        .ok_or_else(|| format!("missing value for {option}"))
}

fn set_update_option<T>(slot: &mut Option<T>, value: T, option: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("{option} specified more than once"));
    }
    *slot = Some(value);
    Ok(())
}

fn mark_update_output(flag: &str, pretty: &mut bool, json: &mut bool) -> Result<(), String> {
    match flag {
        "--pretty" if !*pretty => *pretty = true,
        "--json" if !*json => *json = true,
        "--pretty" | "--json" => return Err(format!("{flag} specified more than once")),
        _ => return Err(format!("unknown update output option: {flag}")),
    }
    Ok(())
}

fn required_update_option<T>(value: Option<T>, option: &str) -> Result<T, String> {
    value.ok_or_else(|| format!("missing required update option {option}"))
}

fn absolute_update_state_path(raw: String) -> Result<std::path::PathBuf, String> {
    let path = std::path::PathBuf::from(raw);
    if path.is_absolute() {
        Ok(path)
    } else {
        std::env::current_dir()
            .map(|directory| directory.join(path))
            .map_err(|error| format!("resolve update state directory: {error}"))
    }
}

fn print_update_document<T: Serialize>(value: &T, pretty: bool) -> Result<(), String> {
    let mut document = if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .map_err(|error| format!("serialize update result: {error}"))?;
    document.push('\n');
    std::io::stdout()
        .lock()
        .write_all(document.as_bytes())
        .map_err(|error| format!("write update result: {error}"))
}

fn run_update(args: &[String]) -> Result<(), String> {
    if args.is_empty()
        || args.first().map(String::as_str) == Some("help")
        || args
            .iter()
            .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        print!("{}", update_usage());
        return Ok(());
    }
    match args[0].as_str() {
        "manifest" => run_update_manifest(&args[1..]),
        "bundle" => run_update_bundle(&args[1..]),
        "check" => run_update_check(&args[1..]),
        "check-online" => run_update_check_online(&args[1..]),
        "dry-run" => run_update_bundle_action(&args[1..], false),
        "apply" => run_update_bundle_action(&args[1..], true),
        "status" => run_update_status(&args[1..]),
        "health" => run_update_health(&args[1..]),
        "recover" => run_update_recover(&args[1..]),
        command => Err(format!("unknown update command: {command}")),
    }
}

fn run_update_manifest(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) != Some("verify") || args.len() < 3 {
        return Err("update manifest requires `verify MANIFEST.json MANIFEST.sig`".into());
    }
    let manifest = &args[1];
    let signature = &args[2];
    let mut public_key = None;
    let mut pretty = false;
    let mut json = false;
    let mut index = 3;
    while index < args.len() {
        match args[index].as_str() {
            "--public-key" => {
                let value = update_option_value(args, &mut index, "--public-key")?;
                set_update_option(
                    &mut public_key,
                    std::path::PathBuf::from(value),
                    "--public-key",
                )?;
            }
            "--pretty" | "--json" => mark_update_output(&args[index], &mut pretty, &mut json)?,
            value => return Err(format!("unknown update manifest verify option: {value}")),
        }
        index += 1;
    }
    let verified =
        denoize::update::UpdateManifest::from_file(manifest, signature, public_key.as_deref())?;
    print_update_document(&verified.verification, pretty)
}

fn run_update_bundle(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("inspect") => run_update_bundle_inspect(&args[1..]),
        Some("build") => run_update_bundle_build(&args[1..]),
        Some("download") => run_update_bundle_download(&args[1..]),
        _ => Err("update bundle requires `inspect`, `download`, or `build`".into()),
    }
}

fn run_update_bundle_inspect(args: &[String]) -> Result<(), String> {
    let bundle = args
        .first()
        .filter(|value| !value.starts_with('-'))
        .ok_or("update bundle inspect requires BUNDLE.dub")?;
    let mut public_key = None;
    let mut pretty = false;
    let mut json = false;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--public-key" => {
                let value = update_option_value(args, &mut index, "--public-key")?;
                set_update_option(
                    &mut public_key,
                    std::path::PathBuf::from(value),
                    "--public-key",
                )?;
            }
            "--pretty" | "--json" => mark_update_output(&args[index], &mut pretty, &mut json)?,
            value => return Err(format!("unknown update bundle inspect option: {value}")),
        }
        index += 1;
    }
    let info = denoize::update::inspect_update_bundle(bundle, public_key.as_deref())?;
    print_update_document(&info, pretty)
}

fn run_update_bundle_download(args: &[String]) -> Result<(), String> {
    let output = args
        .first()
        .filter(|value| !value.starts_with('-'))
        .ok_or("update bundle download requires OUTPUT.dub")?;
    let mut platform = None;
    let mut from_version = None;
    let mut manifest_url = None;
    let mut signature_url = None;
    let mut public_key = None;
    let mut pretty = false;
    let mut json = false;
    let mut index = 1;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--platform" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut platform, value, option)?;
            }
            "--from-version" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut from_version, value, option)?;
            }
            "--manifest-url" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut manifest_url, value, option)?;
            }
            "--signature-url" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut signature_url, value, option)?;
            }
            "--public-key" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut public_key, std::path::PathBuf::from(value), option)?;
            }
            "--pretty" | "--json" => mark_update_output(option, &mut pretty, &mut json)?,
            value => return Err(format!("unknown update bundle download option: {value}")),
        }
        index += 1;
    }
    let verified = denoize::update::fetch_update_manifest(
        manifest_url
            .as_deref()
            .unwrap_or(denoize::update::DEFAULT_UPDATE_MANIFEST_URL),
        signature_url
            .as_deref()
            .unwrap_or(denoize::update::DEFAULT_UPDATE_MANIFEST_SIGNATURE_URL),
        public_key.as_deref(),
    )?;
    let report = denoize::update::download_update_bundle(
        &verified,
        &required_update_option(platform, "--platform")?,
        &required_update_option(from_version, "--from-version")?,
        output,
        public_key.as_deref(),
    )?;
    print_update_document(&report, pretty)
}

fn run_update_bundle_build(args: &[String]) -> Result<(), String> {
    let output = args
        .first()
        .filter(|value| !value.starts_with('-'))
        .ok_or("update bundle build requires OUTPUT.dub")?;
    let mut platform = None;
    let mut from_version = None;
    let mut manifest = None;
    let mut signature = None;
    let mut candidate_artifact = None;
    let mut candidate_sbom = None;
    let mut candidate_provenance = None;
    let mut rollback_artifact = None;
    let mut rollback_sbom = None;
    let mut rollback_provenance = None;
    let mut public_key = None;
    let mut pretty = false;
    let mut json = false;
    let mut index = 1;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--platform" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut platform, value, option)?;
            }
            "--from-version" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut from_version, value, option)?;
            }
            "--manifest" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut manifest, value.into(), option)?;
            }
            "--signature" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut signature, value.into(), option)?;
            }
            "--candidate-artifact" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut candidate_artifact, value.into(), option)?;
            }
            "--candidate-sbom" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut candidate_sbom, value.into(), option)?;
            }
            "--candidate-provenance" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut candidate_provenance, value.into(), option)?;
            }
            "--rollback-artifact" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut rollback_artifact, value.into(), option)?;
            }
            "--rollback-sbom" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut rollback_sbom, value.into(), option)?;
            }
            "--rollback-provenance" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut rollback_provenance, value.into(), option)?;
            }
            "--public-key" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut public_key, value.into(), option)?;
            }
            "--pretty" | "--json" => mark_update_output(option, &mut pretty, &mut json)?,
            value => return Err(format!("unknown update bundle build option: {value}")),
        }
        index += 1;
    }
    let request = denoize::update::UpdateBundleBuildRequest {
        platform: required_update_option(platform, "--platform")?,
        from_version: required_update_option(from_version, "--from-version")?,
        manifest_path: required_update_option(manifest, "--manifest")?,
        signature_path: required_update_option(signature, "--signature")?,
        candidate_artifact_path: required_update_option(
            candidate_artifact,
            "--candidate-artifact",
        )?,
        candidate_sbom_path: required_update_option(candidate_sbom, "--candidate-sbom")?,
        candidate_provenance_path: required_update_option(
            candidate_provenance,
            "--candidate-provenance",
        )?,
        rollback_artifact_path: required_update_option(rollback_artifact, "--rollback-artifact")?,
        rollback_sbom_path: required_update_option(rollback_sbom, "--rollback-sbom")?,
        rollback_provenance_path: required_update_option(
            rollback_provenance,
            "--rollback-provenance",
        )?,
        public_key_path: public_key,
    };
    let info = denoize::update::build_update_bundle(output, &request)?;
    print_update_document(&info, pretty)
}

fn run_update_check(args: &[String]) -> Result<(), String> {
    if args.len() < 2 || args[0].starts_with('-') || args[1].starts_with('-') {
        return Err("update check requires MANIFEST.json MANIFEST.sig".into());
    }
    let manifest = &args[0];
    let signature = &args[1];
    let mut state_dir = None;
    let mut channel = None;
    let mut platform = None;
    let mut current_version = None;
    let mut public_key = None;
    let mut pretty = false;
    let mut json = false;
    let mut index = 2;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--state-dir" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut state_dir, absolute_update_state_path(value)?, option)?;
            }
            "--channel" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut channel, value, option)?;
            }
            "--platform" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut platform, value, option)?;
            }
            "--current-version" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut current_version, value, option)?;
            }
            "--public-key" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut public_key, std::path::PathBuf::from(value), option)?;
            }
            "--pretty" | "--json" => mark_update_output(option, &mut pretty, &mut json)?,
            value => return Err(format!("unknown update check option: {value}")),
        }
        index += 1;
    }
    let verified =
        denoize::update::UpdateManifest::from_file(manifest, signature, public_key.as_deref())?;
    let report = denoize::update::check_update_manifest(
        &verified,
        required_update_option(state_dir, "--state-dir")?,
        &required_update_option(channel, "--channel")?,
        &required_update_option(platform, "--platform")?,
        &required_update_option(current_version, "--current-version")?,
    )?;
    print_update_document(&report, pretty)
}

fn run_update_check_online(args: &[String]) -> Result<(), String> {
    let mut state_dir = None;
    let mut channel = None;
    let mut platform = None;
    let mut current_version = None;
    let mut manifest_url = None;
    let mut signature_url = None;
    let mut public_key = None;
    let mut pretty = false;
    let mut json = false;
    let mut index = 0;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--state-dir" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut state_dir, absolute_update_state_path(value)?, option)?;
            }
            "--channel" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut channel, value, option)?;
            }
            "--platform" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut platform, value, option)?;
            }
            "--current-version" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut current_version, value, option)?;
            }
            "--manifest-url" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut manifest_url, value, option)?;
            }
            "--signature-url" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut signature_url, value, option)?;
            }
            "--public-key" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut public_key, std::path::PathBuf::from(value), option)?;
            }
            "--pretty" | "--json" => mark_update_output(option, &mut pretty, &mut json)?,
            value => return Err(format!("unknown update check-online option: {value}")),
        }
        index += 1;
    }
    let verified = denoize::update::fetch_update_manifest(
        manifest_url
            .as_deref()
            .unwrap_or(denoize::update::DEFAULT_UPDATE_MANIFEST_URL),
        signature_url
            .as_deref()
            .unwrap_or(denoize::update::DEFAULT_UPDATE_MANIFEST_SIGNATURE_URL),
        public_key.as_deref(),
    )?;
    let report = denoize::update::check_update_manifest(
        &verified,
        required_update_option(state_dir, "--state-dir")?,
        &required_update_option(channel, "--channel")?,
        &required_update_option(platform, "--platform")?,
        &required_update_option(current_version, "--current-version")?,
    )?;
    print_update_document(&report, pretty)
}

fn run_update_bundle_action(args: &[String], apply: bool) -> Result<(), String> {
    let command = if apply { "apply" } else { "dry-run" };
    let bundle = args
        .first()
        .filter(|value| !value.starts_with('-'))
        .ok_or_else(|| format!("update {command} requires BUNDLE.dub"))?;
    let mut state_dir = None;
    let mut current_version = None;
    let mut maximum_staging_bytes = None;
    let mut public_key = None;
    let mut pretty = false;
    let mut json = false;
    let mut index = 1;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--state-dir" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut state_dir, absolute_update_state_path(value)?, option)?;
            }
            "--current-version" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut current_version, value, option)?;
            }
            "--max-staging-bytes" => {
                let raw = update_option_value(args, &mut index, option)?;
                let value = raw
                    .parse::<u64>()
                    .map_err(|error| format!("invalid {option}: {error}"))?;
                if value == 0 {
                    return Err("--max-staging-bytes must be positive".into());
                }
                set_update_option(&mut maximum_staging_bytes, value, option)?;
            }
            "--public-key" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut public_key, std::path::PathBuf::from(value), option)?;
            }
            "--pretty" | "--json" => mark_update_output(option, &mut pretty, &mut json)?,
            value => return Err(format!("unknown update {command} option: {value}")),
        }
        index += 1;
    }
    let state_dir = required_update_option(state_dir, "--state-dir")?;
    let current_version = required_update_option(current_version, "--current-version")?;
    if apply {
        let report = denoize::update::apply_update_bundle(
            bundle,
            state_dir,
            &current_version,
            maximum_staging_bytes,
            public_key.as_deref(),
        )?;
        print_update_document(&report, pretty)
    } else {
        let report = denoize::update::dry_run_update_bundle(
            bundle,
            state_dir,
            &current_version,
            maximum_staging_bytes,
            public_key.as_deref(),
        )?;
        print_update_document(&report, pretty)
    }
}

fn run_update_status(args: &[String]) -> Result<(), String> {
    let mut state_dir = None;
    let mut pretty = false;
    let mut json = false;
    let mut index = 0;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--state-dir" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut state_dir, absolute_update_state_path(value)?, option)?;
            }
            "--pretty" | "--json" => mark_update_output(option, &mut pretty, &mut json)?,
            value => return Err(format!("unknown update status option: {value}")),
        }
        index += 1;
    }
    let report = denoize::update::update_status(required_update_option(state_dir, "--state-dir")?)?;
    print_update_document(&report, pretty)
}

fn run_update_health(args: &[String]) -> Result<(), String> {
    let action = args
        .first()
        .map(String::as_str)
        .ok_or("update health requires `begin` or `confirm`")?;
    if !matches!(action, "begin" | "confirm") {
        return Err(format!("unknown update health command: {action}"));
    }
    let mut state_dir = None;
    let mut running_version = None;
    let mut token = None;
    let mut pretty = false;
    let mut json = false;
    let mut index = 1;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--state-dir" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut state_dir, absolute_update_state_path(value)?, option)?;
            }
            "--running-version" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut running_version, value, option)?;
            }
            "--token" if action == "confirm" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut token, value, option)?;
            }
            "--pretty" | "--json" => mark_update_output(option, &mut pretty, &mut json)?,
            value => return Err(format!("unknown update health {action} option: {value}")),
        }
        index += 1;
    }
    let state_dir = required_update_option(state_dir, "--state-dir")?;
    let running_version = required_update_option(running_version, "--running-version")?;
    let report = if action == "begin" {
        denoize::update::begin_update_startup_health(state_dir, &running_version, None)?
    } else {
        denoize::update::confirm_update_health(
            state_dir,
            &running_version,
            &required_update_option(token, "--token")?,
            None,
        )?
    };
    print_update_document(&report, pretty)
}

fn run_update_recover(args: &[String]) -> Result<(), String> {
    let mut state_dir = None;
    let mut reason = None;
    let mut pretty = false;
    let mut json = false;
    let mut index = 0;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--state-dir" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut state_dir, absolute_update_state_path(value)?, option)?;
            }
            "--reason" => {
                let value = update_option_value(args, &mut index, option)?;
                set_update_option(&mut reason, value, option)?;
            }
            "--pretty" | "--json" => mark_update_output(option, &mut pretty, &mut json)?,
            value => return Err(format!("unknown update recover option: {value}")),
        }
        index += 1;
    }
    let report = denoize::update::recover_update(
        required_update_option(state_dir, "--state-dir")?,
        reason.as_deref().unwrap_or("manual-recovery"),
        None,
    )?;
    print_update_document(&report, pretty)
}

fn project_usage() -> &'static str {
    "\
Portable project and deterministic partial-file timeline commands:

    denoize project create <PROJECT.json> --root DIR --project-id ID \\
        --source ID=PATH [--source ID=PATH ...] \\
        --selection ID=SOURCE,START_SECONDS,DURATION_SECONDS[,CHANNEL_MAP[,PAD_BEFORE[,PAD_AFTER[,CROSSFADE]]]] \\
        [--source-license SOURCE=ID=PATH] [--setting ID=PATH] [--preset ID=PATH] \\
        [--model ID=PACKAGE.dmp,PUBLIC_KEY] [--plan ID=PATH] [--receipt ID=PATH] \\
        [--timeline ID] [--pretty] [--force]
    denoize project inspect <PROJECT.json> [--pretty]
    denoize project validate <PROJECT.json> --root DIR [--pretty]
    denoize project assemble <PROJECT.json> <OUTPUT.wav> --root DIR \\
        [--timeline ID] [--plan PLAN.json] \\
        [--receipt RECEIPT.json --receipt-key SECRET.json] [--pretty] [--force]
    denoize project relocate <PROJECT.json> <SOURCE_ID> <CANDIDATE> \\
        --root DIR --output PROJECT.json [--pretty] [--force]
    denoize project bundle create <PROJECT.json> <OUTPUT.dpb> --root DIR \\
        [--include-sources --max-source-bytes N] \\
        [--include-models --max-model-bytes N] [--pretty] [--force]
    denoize project bundle inspect <BUNDLE.dpb> [--pretty]
    denoize project bundle import <BUNDLE.dpb> <NEW_PROJECT_DIR> [--pretty]
    denoize project plan create <PROJECT.json> <OUTPUT.wav> --root DIR \\
        --output PLAN.json [--timeline ID] [--pretty] [--force]
    denoize project receipt verify <RECEIPT.json> --root DIR \\
        (--public-key KEY.json | --trust-policy POLICY.json) [--plan PLAN.json] [--pretty]
    denoize project batch <PROJECT.json>... --root DIR --output-dir DIR \\
        [--timeline ID] [--pretty] [--force]
    denoize project watch <INPUT_DIR> <OUTPUT_DIR> --root DIR \\
        --receipt-key SECRET.json [--timeline ID] [--once] [--settle-ms N] \\
        [--poll-ms N] [--recursive] [--pretty]
    denoize project v2 <COMMAND>  (run `denoize project v2 help`)

CHANNEL_MAP is a '+'-separated list of zero-based source channels, for example
`0+1` or `0+0`. Times are quantized exactly once onto the source presentation
timebase. Crossfades are supported only between adjacent unpadded selections.
All commands reject unknown/future records and changed fingerprints before any
project or audio output is published. Bundles always carry settings, presets,
plans, receipts, source licenses, model public keys, and verification evidence.
Source audio and model packages require explicit aggregate byte limits. Import
publishes only to a new directory and never replaces an existing project.
"
}

fn run_project(args: &[String]) -> Result<(), String> {
    if args.is_empty()
        || args.first().map(String::as_str) == Some("help")
        || args
            .iter()
            .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        print!("{}", project_usage());
        return Ok(());
    }
    match args[0].as_str() {
        "create" => run_project_create(&args[1..]),
        "inspect" => run_project_inspect(&args[1..]),
        "validate" => run_project_validate(&args[1..]),
        "assemble" => run_project_assemble(&args[1..]),
        "relocate" => run_project_relocate(&args[1..]),
        "bundle" => run_project_bundle(&args[1..]),
        "plan" => run_project_plan(&args[1..]),
        "receipt" => run_project_receipt(&args[1..]),
        "batch" => run_project_batch_command(&args[1..]),
        "watch" => run_project_watch(&args[1..]),
        "v2" => run_project_v2(&args[1..]),
        command => Err(format!("unknown project command: {command}")),
    }
}

fn project_v2_usage() -> &'static str {
    "\
Durable non-destructive project graph v2 commands:

    denoize project v2 migrate <V1.json> <V2.json> --root DIR [--pretty] [--force]
    denoize project v2 inspect <V2.json> [--pretty]
    denoize project v2 validate <V2.json> --root DIR [--pretty]
    denoize project v2 render <V2.json> <OUTPUT> --root DIR \\
        [--graph ID] [--jobs N] [--max-memory-mib N] \\
        [--max-output-frames N] [--pretty] [--force]
    denoize project v2 journal inspect <JOURNAL.ndjson> [--pretty]
    denoize project v2 cache key <V2.json> [--graph ID] \\
        [--format wav-f32|wav-pcm24|flac24|opus|mp3|m4a] \\
        [--bitrate BPS] [--jobs N] [--pretty]
    denoize project v2 interchange assess <V2.json> \\
        --format otio|otioz|otiod|adm-bw64 \\
        [--graph ID] [--direction import|export] [--pretty]
    denoize project v2 otio export <V2.json> <OUTPUT.otio> --root DIR \\
        [--graph ID] [--accept-losses] [--pretty] [--force]
    denoize project v2 otio inspect <INPUT.otio> [--pretty]
    denoize project v2 provenance sign <V2.json> <OUTPUT_AUDIO> <PROVENANCE.json> \\
        --root DIR --secret-key SECRET_KEY.json --format wav-f32|wav-pcm24|flac24|opus|mp3|m4a \\
        [--graph ID] [--pretty] [--force]
    denoize project v2 provenance verify <PROVENANCE.json> <OUTPUT_AUDIO> \\
        --public-key PUBLIC_KEY.json [--pretty]

The v2 manifest is a closed executable graph: unknown fields and nodes fail
closed. Cache keys bind source bytes, graph topology, immutable effects,
automation, models, runtime choice, and output settings. OTIO/ADM are explicit
loss-reporting interchange boundaries and never import free-form executable
effects. Provenance output is a detached C2PA 2.4-targeted, Ed25519-signed
assertion; this release does not embed a C2PA manifest store. Ogg/Opus always
uses its explicit detached carrier.
"
}

fn run_project_v2(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        None | Some("help") => {
            print!("{}", project_v2_usage());
            Ok(())
        }
        Some("migrate") => run_project_v2_migrate(&args[1..]),
        Some("inspect") => run_project_v2_inspect(&args[1..], false),
        Some("validate") => run_project_v2_inspect(&args[1..], true),
        Some("render") => run_project_v2_render(&args[1..]),
        Some("journal") => run_project_v2_journal(&args[1..]),
        Some("cache") => run_project_v2_cache(&args[1..]),
        Some("interchange") => run_project_v2_interchange(&args[1..]),
        Some("otio") => run_project_v2_otio(&args[1..]),
        Some("provenance") => run_project_v2_provenance(&args[1..]),
        Some(command) => Err(format!("unknown project v2 command: {command}")),
    }
}

fn run_project_v2_migrate(args: &[String]) -> Result<(), String> {
    if args.len() < 2 || args[0].starts_with('-') || args[1].starts_with('-') {
        return Err("project v2 migrate requires V1.json V2.json".into());
    }
    let mut root = None;
    let mut pretty = false;
    let mut force = false;
    let mut index = 2;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--root" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut root, value, option)?;
            }
            "--pretty" => pretty = true,
            "--force" => force = true,
            value => return Err(format!("unknown project v2 migrate option: {value}")),
        }
        index += 1;
    }
    let root = canonical_cli_project_root(
        root.as_deref()
            .ok_or("project v2 migrate requires --root DIR")?,
    )?;
    let legacy = denoize::ProjectManifest::from_file(&args[0])?;
    denoize::validate_project_files(&legacy, &root, DecodeLimits::default())?;
    reject_project_v2_cli_file_aliases(
        std::path::Path::new(&args[1]),
        &[(std::path::Path::new(&args[0]), "v1 manifest")],
        "project v2 migration output",
    )?;
    reject_project_v1_migration_reference_aliases(&legacy, &root, std::path::Path::new(&args[1]))?;
    let migrated = denoize::project_v2::migrate_project_v1_to_v2(&legacy)?;
    let report =
        denoize::project_v2::verify_project_v2_files(&migrated, &root, DecodeLimits::default())?;
    denoize::project_v2::write_project_v2_manifest(
        &args[1],
        &migrated,
        project_commit_mode(force),
        pretty,
    )?;
    print_project_document(&report, pretty)
}

fn run_project_v2_inspect(args: &[String], validation_only: bool) -> Result<(), String> {
    let command = if validation_only {
        "validate"
    } else {
        "inspect"
    };
    let input = args
        .first()
        .filter(|value| !value.starts_with('-'))
        .ok_or("project v2 inspect/validate requires V2.json")?;
    let mut pretty = false;
    let mut root = None;
    let mut index = 1;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--root" if validation_only => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut root, value, option)?;
            }
            "--pretty" => pretty = true,
            value => return Err(format!("unknown project v2 {command} option: {value}")),
        }
        index += 1;
    }
    let manifest = denoize::project_v2::ProjectV2Manifest::from_file(input)?;
    if validation_only {
        let root = canonical_cli_project_root(
            root.as_deref()
                .ok_or("project v2 validate requires --root DIR")?,
        )?;
        print_project_document(
            &denoize::project_v2::verify_project_v2_files(
                &manifest,
                root,
                DecodeLimits::default(),
            )?,
            pretty,
        )
    } else {
        print_project_document(&manifest, pretty)
    }
}

fn run_project_v2_render(args: &[String]) -> Result<(), String> {
    if args.len() < 2 || args[0].starts_with('-') || args[1].starts_with('-') {
        return Err("project v2 render requires V2.json OUTPUT".into());
    }
    let mut root = None;
    let mut graph = None;
    let mut jobs = 1_u16;
    let mut max_memory_bytes = 1024_u64 * 1024 * 1024;
    let mut max_output_frames = 48_000_u64 * 60 * 60 * 8;
    let mut pretty = false;
    let mut force = false;
    let mut index = 2;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--root" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut root, value, option)?;
            }
            "--graph" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut graph, value, option)?;
            }
            "--jobs" => {
                let value = project_option_value(args, &mut index, option)?;
                jobs = value
                    .parse()
                    .map_err(|_| format!("invalid project v2 jobs: {value}"))?;
            }
            "--max-memory-mib" => {
                let value = project_option_value(args, &mut index, option)?;
                max_memory_bytes = value
                    .parse::<u64>()
                    .map_err(|_| format!("invalid project v2 memory limit: {value}"))?
                    .checked_mul(1024 * 1024)
                    .ok_or("project v2 memory limit overflows")?;
            }
            "--max-output-frames" => {
                let value = project_option_value(args, &mut index, option)?;
                max_output_frames = value
                    .parse()
                    .map_err(|_| format!("invalid project v2 frame limit: {value}"))?;
            }
            "--pretty" => pretty = true,
            "--force" => force = true,
            value => return Err(format!("unknown project v2 render option: {value}")),
        }
        index += 1;
    }
    let root = canonical_cli_project_root(
        root.as_deref()
            .ok_or("project v2 render requires --root DIR")?,
    )?;
    let manifest = denoize::project_v2::ProjectV2Manifest::from_file(&args[0])?;
    let graph = graph.unwrap_or_else(|| manifest.root_graph_id.clone());
    reject_project_v2_cli_file_aliases(
        std::path::Path::new(&args[1]),
        &[(std::path::Path::new(&args[0]), "v2 manifest")],
        "project v2 render output",
    )?;
    let options = denoize::project_v2::ProjectV2RenderOptions {
        deterministic: true,
        jobs,
        max_memory_bytes,
        max_output_frames,
    };
    let limits = DecodeLimits::default().with_max_working_set_bytes(Some(max_memory_bytes));
    let report = denoize::project_v2::publish_project_v2_graph(
        &manifest,
        &graph,
        root,
        &args[1],
        options,
        limits,
        EncodeOptions::default(),
        project_commit_mode(force),
    )?;
    print_project_document(&report, pretty)
}

fn run_project_v2_journal(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) != Some("inspect") {
        return Err("project v2 journal requires `inspect`".into());
    }
    let input = args
        .get(1)
        .filter(|value| !value.starts_with('-'))
        .ok_or("project v2 journal inspect requires JOURNAL.ndjson")?;
    let mut pretty = false;
    for option in &args[2..] {
        match option.as_str() {
            "--pretty" => pretty = true,
            value => return Err(format!("unknown project v2 journal option: {value}")),
        }
    }
    let report = denoize::project_v2::read_project_v2_journal(input)?;
    print_project_document(&report, pretty)
}

fn run_project_v2_cache(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) != Some("key") {
        return Err("project v2 cache requires `key`".into());
    }
    let input = args
        .get(1)
        .filter(|value| !value.starts_with('-'))
        .ok_or("project v2 cache key requires V2.json")?;
    let mut graph = None;
    let mut format = denoize::project_v2::ProjectV2OutputFormat::WavFloat32;
    let mut bitrate = None;
    let mut jobs = 1_u16;
    let mut pretty = false;
    let mut index = 2;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--graph" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut graph, value, option)?;
            }
            "--format" => {
                let value = project_option_value(args, &mut index, option)?;
                format = parse_project_v2_output_format(&value)?;
            }
            "--bitrate" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(
                    &mut bitrate,
                    value
                        .parse::<u32>()
                        .map_err(|_| format!("invalid project v2 bitrate: {value}"))?,
                    option,
                )?;
            }
            "--jobs" => {
                let value = project_option_value(args, &mut index, option)?;
                jobs = value
                    .parse()
                    .map_err(|_| format!("invalid project v2 jobs: {value}"))?;
            }
            "--pretty" => pretty = true,
            value => return Err(format!("unknown project v2 cache option: {value}")),
        }
        index += 1;
    }
    let manifest = denoize::project_v2::ProjectV2Manifest::from_file(input)?;
    let graph_id = graph.unwrap_or_else(|| manifest.root_graph_id.clone());
    let graph = manifest.graph(&graph_id)?;
    let lossy = matches!(
        format,
        denoize::project_v2::ProjectV2OutputFormat::OggOpus
            | denoize::project_v2::ProjectV2OutputFormat::Mp3
            | denoize::project_v2::ProjectV2OutputFormat::M4a
    );
    let output = denoize::project_v2::ProjectV2OutputSettings {
        format,
        sample_rate: graph.sample_rate,
        channels: graph.channels,
        bitrate_bps: if lossy {
            Some(bitrate.unwrap_or(192_000))
        } else {
            bitrate
        },
        metadata_policy: "drop".into(),
        provenance_policy_digest: None,
    };
    let request = denoize::project_v2::ProjectV2CacheRequest::from_manifest(
        &manifest,
        &graph_id,
        denoize::project_v2::ProjectV2RuntimeIdentity::deterministic_scalar(jobs),
        output,
    )?;
    let document = denoize::project_v2::ProjectV2CacheKeyReport::new(request)?;
    print_project_document(&document, pretty)
}

fn parse_project_v2_output_format(
    value: &str,
) -> Result<denoize::project_v2::ProjectV2OutputFormat, String> {
    match value {
        "wav-f32" | "wav-float32" => Ok(denoize::project_v2::ProjectV2OutputFormat::WavFloat32),
        "wav-pcm24" | "wav24" => Ok(denoize::project_v2::ProjectV2OutputFormat::WavPcm24),
        "flac24" | "flac" => Ok(denoize::project_v2::ProjectV2OutputFormat::Flac24),
        "opus" | "ogg-opus" => Ok(denoize::project_v2::ProjectV2OutputFormat::OggOpus),
        "mp3" => Ok(denoize::project_v2::ProjectV2OutputFormat::Mp3),
        "m4a" => Ok(denoize::project_v2::ProjectV2OutputFormat::M4a),
        _ => Err(format!("unknown project v2 output format: {value}")),
    }
}

fn run_project_v2_interchange(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) != Some("assess") {
        return Err("project v2 interchange requires `assess`".into());
    }
    let input = args
        .get(1)
        .filter(|value| !value.starts_with('-'))
        .ok_or("project v2 interchange assess requires V2.json")?;
    let mut graph = None;
    let mut format = None;
    let mut direction = denoize::project_v2::ProjectV2InterchangeDirection::Export;
    let mut pretty = false;
    let mut index = 2;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--graph" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut graph, value, option)?;
            }
            "--format" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(
                    &mut format,
                    parse_project_v2_interchange_format(&value)?,
                    option,
                )?;
            }
            "--direction" => {
                let value = project_option_value(args, &mut index, option)?;
                direction = match value.as_str() {
                    "import" => denoize::project_v2::ProjectV2InterchangeDirection::Import,
                    "export" => denoize::project_v2::ProjectV2InterchangeDirection::Export,
                    _ => return Err(format!("unknown project v2 interchange direction: {value}")),
                };
            }
            "--pretty" => pretty = true,
            value => return Err(format!("unknown project v2 interchange option: {value}")),
        }
        index += 1;
    }
    let manifest = denoize::project_v2::ProjectV2Manifest::from_file(input)?;
    let graph = graph.unwrap_or_else(|| manifest.root_graph_id.clone());
    let report = denoize::project_v2::assess_project_v2_interchange(
        &manifest,
        &graph,
        format.ok_or("project v2 interchange requires --format")?,
        direction,
    )?;
    print_project_document(&report, pretty)
}

fn parse_project_v2_interchange_format(
    value: &str,
) -> Result<denoize::project_v2::ProjectV2InterchangeFormat, String> {
    match value {
        "otio" => Ok(denoize::project_v2::ProjectV2InterchangeFormat::Otio),
        "otioz" => Ok(denoize::project_v2::ProjectV2InterchangeFormat::Otioz),
        "otiod" => Ok(denoize::project_v2::ProjectV2InterchangeFormat::Otiod),
        "adm-bw64" | "adm" | "bw64" => Ok(denoize::project_v2::ProjectV2InterchangeFormat::AdmBw64),
        _ => Err(format!("unknown project v2 interchange format: {value}")),
    }
}

fn run_project_v2_otio(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("inspect") => {
            let input = args
                .get(1)
                .filter(|value| !value.starts_with('-'))
                .ok_or("project v2 otio inspect requires INPUT.otio")?;
            let mut pretty = false;
            for option in &args[2..] {
                match option.as_str() {
                    "--pretty" => pretty = true,
                    value => {
                        return Err(format!("unknown project v2 otio inspect option: {value}"))
                    }
                }
            }
            print_project_document(
                &denoize::project_v2::inspect_project_v2_otio(input)?,
                pretty,
            )
        }
        Some("export") => {
            if args.len() < 3 || args[1].starts_with('-') || args[2].starts_with('-') {
                return Err("project v2 otio export requires V2.json OUTPUT.otio".into());
            }
            let mut graph = None;
            let mut root = None;
            let mut accept_losses = false;
            let mut pretty = false;
            let mut force = false;
            let mut index = 3;
            while index < args.len() {
                let option = args[index].as_str();
                match option {
                    "--root" => {
                        let value = project_option_value(args, &mut index, option)?;
                        set_project_option(&mut root, value, option)?;
                    }
                    "--graph" => {
                        let value = project_option_value(args, &mut index, option)?;
                        set_project_option(&mut graph, value, option)?;
                    }
                    "--accept-losses" => accept_losses = true,
                    "--pretty" => pretty = true,
                    "--force" => force = true,
                    value => return Err(format!("unknown project v2 otio export option: {value}")),
                }
                index += 1;
            }
            let manifest = denoize::project_v2::ProjectV2Manifest::from_file(&args[1])?;
            let graph = graph.unwrap_or_else(|| manifest.root_graph_id.clone());
            let root = canonical_cli_project_root(
                root.as_deref()
                    .ok_or("project v2 otio export requires --root DIR")?,
            )?;
            reject_project_v2_cli_file_aliases(
                std::path::Path::new(&args[2]),
                &[(std::path::Path::new(&args[1]), "v2 manifest")],
                "project v2 OTIO output",
            )?;
            let report = denoize::project_v2::export_project_v2_otio(
                &manifest,
                &graph,
                &root,
                &args[2],
                accept_losses,
                project_commit_mode(force),
            )?;
            print_project_document(&report, pretty)
        }
        _ => Err("project v2 otio requires `export` or `inspect`".into()),
    }
}

fn run_project_v2_provenance(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("sign") => run_project_v2_provenance_sign(&args[1..]),
        Some("verify") => run_project_v2_provenance_verify(&args[1..]),
        _ => Err("project v2 provenance requires `sign` or `verify`".into()),
    }
}

fn run_project_v2_provenance_sign(args: &[String]) -> Result<(), String> {
    if args.len() < 3
        || args[0].starts_with('-')
        || args[1].starts_with('-')
        || args[2].starts_with('-')
    {
        return Err(
            "project v2 provenance sign requires V2.json OUTPUT_AUDIO PROVENANCE.json".into(),
        );
    }
    let mut root = None;
    let mut secret_key = None;
    let mut graph = None;
    let mut format = None;
    let mut pretty = false;
    let mut force = false;
    let mut index = 3;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--root" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut root, value, option)?;
            }
            "--secret-key" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut secret_key, value, option)?;
            }
            "--graph" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut graph, value, option)?;
            }
            "--format" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut format, parse_project_v2_output_format(&value)?, option)?;
            }
            "--pretty" => pretty = true,
            "--force" => force = true,
            value => {
                return Err(format!(
                    "unknown project v2 provenance sign option: {value}"
                ))
            }
        }
        index += 1;
    }
    let root = canonical_cli_project_root(
        root.as_deref()
            .ok_or("project v2 provenance sign requires --root DIR")?,
    )?;
    let secret_key = secret_key
        .as_deref()
        .ok_or("project v2 provenance sign requires --secret-key")?;
    let format = format.ok_or("project v2 provenance sign requires --format")?;
    let manifest = denoize::project_v2::ProjectV2Manifest::from_file(&args[0])?;
    let graph = graph.unwrap_or_else(|| manifest.root_graph_id.clone());
    reject_project_v2_cli_file_aliases(
        std::path::Path::new(&args[2]),
        &[
            (std::path::Path::new(&args[0]), "v2 manifest"),
            (std::path::Path::new(&args[1]), "published audio"),
            (std::path::Path::new(secret_key), "provenance secret key"),
        ],
        "project v2 provenance output",
    )?;
    denoize::project_v2::validate_project_v2_publication_destination(
        &manifest,
        &root,
        std::path::Path::new(&args[2]),
    )?;
    // No output container is mutated in v0.85.0. The signed assertion is an
    // explicit detached handoff; Ogg Opus receives a distinct carrier label so
    // callers cannot accidentally claim embedded support.
    let carrier = if format == denoize::project_v2::ProjectV2OutputFormat::OggOpus {
        denoize::project_v2::ProjectV2ProvenanceCarrier::DetachedOggOpus
    } else {
        denoize::project_v2::ProjectV2ProvenanceCarrier::DetachedGeneric
    };
    let payload = denoize::project_v2::build_project_v2_provenance_payload(
        &manifest,
        &graph,
        root,
        &args[1],
        format,
        carrier,
        DecodeLimits::default(),
    )?;
    let signed = denoize::project_v2::sign_project_v2_provenance(
        payload,
        &ReceiptSecretKey::from_file(secret_key)?,
    )?;
    denoize::project_v2::write_signed_project_v2_provenance(
        &args[2],
        &signed,
        project_commit_mode(force),
        pretty,
    )?;
    print_project_document(&signed, pretty)
}

fn run_project_v2_provenance_verify(args: &[String]) -> Result<(), String> {
    if args.len() < 2 || args[0].starts_with('-') || args[1].starts_with('-') {
        return Err("project v2 provenance verify requires PROVENANCE.json OUTPUT_AUDIO".into());
    }
    let mut public_key = None;
    let mut pretty = false;
    let mut index = 2;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--public-key" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut public_key, value, option)?;
            }
            "--pretty" => pretty = true,
            value => {
                return Err(format!(
                    "unknown project v2 provenance verify option: {value}"
                ))
            }
        }
        index += 1;
    }
    let public_key = public_key
        .as_deref()
        .ok_or("project v2 provenance verify requires --public-key")?;
    let signed = denoize::project_v2::SignedProjectV2Provenance::from_file(&args[0])?;
    denoize::project_v2::verify_project_v2_provenance_output(
        &signed,
        &ReceiptPublicKey::from_file(public_key)?,
        &args[1],
        DecodeLimits::default(),
    )?;
    print_project_document(&signed, pretty)
}

fn reject_project_v2_cli_file_aliases(
    output: &std::path::Path,
    protected: &[(&std::path::Path, &str)],
    context: &str,
) -> Result<(), String> {
    let destination = normalized_project_destination(output, context)?;
    let existing_target = std::fs::canonicalize(&destination).ok();
    for (path, label) in protected {
        let protected = std::fs::canonicalize(path)
            .map_err(|error| format!("re-resolve {label} {}: {error}", path.display()))?;
        if destination == protected || existing_target.as_ref() == Some(&protected) {
            return Err(format!("{context} must not replace or alias its {label}"));
        }
    }
    Ok(())
}

fn reject_project_v1_migration_reference_aliases(
    manifest: &denoize::ProjectManifest,
    root: &std::path::Path,
    output: &std::path::Path,
) -> Result<(), String> {
    let context = "project v2 migration output";
    let destination = normalized_project_destination(output, context)?;
    let existing_target = std::fs::canonicalize(&destination).ok();
    let mut locators = Vec::new();
    for source in &manifest.sources {
        locators.push(source.locator.as_str());
        if let Some(license) = &source.license {
            locators.push(license.locator.as_str());
        }
    }
    for reference in manifest
        .settings
        .iter()
        .chain(&manifest.presets)
        .chain(&manifest.plans)
        .chain(&manifest.receipts)
    {
        locators.push(reference.locator.as_str());
    }
    for model in &manifest.models {
        locators.push(model.package.locator.as_str());
        locators.push(model.public_key.locator.as_str());
    }
    for locator in locators {
        let requested = root.join(locator);
        let artifact = std::fs::canonicalize(&requested)
            .ok()
            .or_else(|| normalized_project_destination(&requested, "v1 project artifact").ok());
        if let Some(artifact) = artifact {
            if destination == artifact || existing_target.as_ref() == Some(&artifact) {
                return Err(format!(
                    "{context} collides with referenced v1 artifact {locator}"
                ));
            }
        }
    }
    Ok(())
}

fn run_project_bundle(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("create") => run_project_bundle_create(&args[1..]),
        Some("inspect") => run_project_bundle_inspect(&args[1..]),
        Some("import") => run_project_bundle_import(&args[1..]),
        _ => Err("project bundle requires `create`, `inspect`, or `import`".into()),
    }
}

fn run_project_bundle_create(args: &[String]) -> Result<(), String> {
    if args.len() < 2 || args[0].starts_with('-') || args[1].starts_with('-') {
        return Err("project bundle create requires PROJECT.json OUTPUT.dpb".into());
    }
    let project = &args[0];
    let output = &args[1];
    let mut root = None;
    let mut include_sources = false;
    let mut source_limit = None;
    let mut include_models = false;
    let mut model_limit = None;
    let mut pretty = false;
    let mut force = false;
    let mut index = 2;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--root" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut root, value, option)?;
            }
            "--include-sources" if !include_sources => include_sources = true,
            "--include-models" if !include_models => include_models = true,
            "--max-source-bytes" => {
                let raw = project_option_value(args, &mut index, option)?;
                let value = raw
                    .parse::<u64>()
                    .map_err(|_| format!("invalid value for {option}: {raw}"))?;
                set_project_option(&mut source_limit, value, option)?;
            }
            "--max-model-bytes" => {
                let raw = project_option_value(args, &mut index, option)?;
                let value = raw
                    .parse::<u64>()
                    .map_err(|_| format!("invalid value for {option}: {raw}"))?;
                set_project_option(&mut model_limit, value, option)?;
            }
            "--pretty" if !pretty => pretty = true,
            "--force" if !force => force = true,
            "--include-sources" | "--include-models" | "--pretty" | "--force" => {
                return Err(format!("{option} specified more than once"));
            }
            value => return Err(format!("unknown project bundle create option: {value}")),
        }
        index += 1;
    }
    let root = canonical_cli_project_root(
        root.as_deref()
            .ok_or("project bundle create requires --root DIR")?,
    )?;
    let options = denoize::ProjectBundleBuildOptions {
        include_sources,
        source_payload_limit_bytes: source_limit.unwrap_or(0),
        include_models,
        model_payload_limit_bytes: model_limit.unwrap_or(0),
        commit_mode: project_commit_mode(force),
    };
    let report =
        denoize::build_project_bundle(project, root, output, &options, DecodeLimits::default())?;
    print_project_document(&report, pretty)
}

fn run_project_bundle_inspect(args: &[String]) -> Result<(), String> {
    let bundle = args
        .first()
        .filter(|value| !value.starts_with('-'))
        .ok_or("project bundle inspect requires BUNDLE.dpb")?;
    let mut pretty = false;
    for option in &args[1..] {
        match option.as_str() {
            "--pretty" if !pretty => pretty = true,
            "--pretty" => return Err("--pretty specified more than once".into()),
            value => return Err(format!("unknown project bundle inspect option: {value}")),
        }
    }
    let report = denoize::inspect_project_bundle(bundle)?;
    print_project_document(&report, pretty)
}

fn run_project_bundle_import(args: &[String]) -> Result<(), String> {
    if args.len() < 2 || args[0].starts_with('-') || args[1].starts_with('-') {
        return Err("project bundle import requires BUNDLE.dpb NEW_PROJECT_DIR".into());
    }
    let mut pretty = false;
    for option in &args[2..] {
        match option.as_str() {
            "--pretty" if !pretty => pretty = true,
            "--pretty" => return Err("--pretty specified more than once".into()),
            value => return Err(format!("unknown project bundle import option: {value}")),
        }
    }
    let report = denoize::import_project_bundle(&args[0], &args[1])?;
    print_project_document(&report, pretty)
}

fn run_project_plan(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("create") => run_project_plan_create(&args[1..]),
        _ => Err("project plan requires `create`".into()),
    }
}

fn run_project_plan_create(args: &[String]) -> Result<(), String> {
    if args.len() < 2 || args[0].starts_with('-') || args[1].starts_with('-') {
        return Err("project plan create requires PROJECT.json OUTPUT.wav".into());
    }
    let project_raw = &args[0];
    let audio_output_raw = &args[1];
    let mut root = None;
    let mut plan_output = None;
    let mut timeline = None;
    let mut pretty = false;
    let mut force = false;
    let mut index = 2;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--root" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut root, value, option)?;
            }
            "--output" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut plan_output, value, option)?;
            }
            "--timeline" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut timeline, value, option)?;
            }
            "--pretty" if !pretty => pretty = true,
            "--force" if !force => force = true,
            "--pretty" | "--force" => return Err(format!("{option} specified more than once")),
            value => return Err(format!("unknown project plan create option: {value}")),
        }
        index += 1;
    }
    let root = canonical_cli_project_root(
        root.as_deref()
            .ok_or("project plan create requires --root DIR")?,
    )?;
    let project_path = contained_project_input(&root, project_raw, "project manifest")?;
    let audio_output = contained_project_output(&root, audio_output_raw, "project output")?;
    let manifest = denoize::ProjectManifest::from_file(&project_path)?;
    denoize::validate_project_files(&manifest, &root, DecodeLimits::default())?;
    let timeline = timeline
        .or_else(|| {
            manifest
                .timelines
                .first()
                .map(|timeline| timeline.id.clone())
        })
        .ok_or("project has no timeline")?;
    let manifest_reference = denoize::project_artifact_reference("manifest", &project_path, &root)?;
    let output_locator = denoize::portable_locator(&audio_output, &root)?;
    let plan = denoize::ProjectExecutionPlan::new(
        &manifest,
        &timeline,
        manifest_reference,
        output_locator,
        project_commit_mode(force),
    )?;
    let plan_output = plan_output.ok_or("project plan create requires --output PLAN.json")?;
    reject_cli_project_publication_collision(
        &manifest,
        &root,
        Some(&project_path),
        std::path::Path::new(&plan_output),
        "project plan output",
    )?;
    let plan_destination =
        normalized_project_destination(std::path::Path::new(&plan_output), "project plan output")?;
    let audio_destination = normalized_project_destination(&audio_output, "project audio output")?;
    let plan_target = std::fs::canonicalize(&plan_destination).ok();
    let audio_target = std::fs::canonicalize(&audio_destination).ok();
    if plan_destination == audio_destination || plan_target.is_some() && plan_target == audio_target
    {
        return Err("project plan and audio output paths must differ".into());
    }
    denoize::write_project_execution_plan(plan_output, &plan, project_commit_mode(force), pretty)?;
    print_project_document(&plan, pretty)
}

fn run_project_receipt(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("verify") => run_project_receipt_verify(&args[1..]),
        _ => Err("project receipt requires `verify`".into()),
    }
}

fn run_project_receipt_verify(args: &[String]) -> Result<(), String> {
    let receipt_path = args
        .first()
        .filter(|value| !value.starts_with('-'))
        .ok_or("project receipt verify requires RECEIPT.json")?;
    let mut root = None;
    let mut public_key = None;
    let mut trust_policy = None;
    let mut plan = None;
    let mut pretty = false;
    let mut index = 1;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--root" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut root, value, option)?;
            }
            "--public-key" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut public_key, value, option)?;
            }
            "--trust-policy" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut trust_policy, value, option)?;
            }
            "--plan" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut plan, value, option)?;
            }
            "--pretty" if !pretty => pretty = true,
            "--pretty" => return Err("--pretty specified more than once".into()),
            value => return Err(format!("unknown project receipt verify option: {value}")),
        }
        index += 1;
    }
    if public_key.is_some() == trust_policy.is_some() {
        return Err(
            "project receipt verify requires exactly one of --public-key or --trust-policy".into(),
        );
    }
    let root = canonical_cli_project_root(
        root.as_deref()
            .ok_or("project receipt verify requires --root DIR")?,
    )?;
    let receipt = denoize::SignedProjectExecutionReceipt::from_file(receipt_path)?;
    let plan = plan
        .as_deref()
        .map(denoize::ProjectExecutionPlan::from_file)
        .transpose()?;
    let report = if let Some(path) = public_key {
        let key = denoize::ReceiptPublicKey::from_file(path)?;
        receipt.verify_with_key(&key, plan.as_ref(), &root)?
    } else {
        let policy_path =
            trust_policy.ok_or("project receipt trust source disappeared after validation")?;
        let policy = denoize::ReceiptTrustPolicy::from_file(policy_path)?;
        receipt.verify_with_policy(&policy, plan.as_ref(), &root)?
    };
    print_project_document(&report, pretty)
}

fn run_project_batch_command(args: &[String]) -> Result<(), String> {
    let mut manifests = Vec::new();
    let mut root = None;
    let mut output_dir = None;
    let mut timeline = None;
    let mut pretty = false;
    let mut force = false;
    let mut index = 0;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--root" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut root, value, option)?;
            }
            "--output-dir" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut output_dir, value, option)?;
            }
            "--timeline" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut timeline, value, option)?;
            }
            "--pretty" if !pretty => pretty = true,
            "--force" if !force => force = true,
            "--pretty" | "--force" => return Err(format!("{option} specified more than once")),
            value if value.starts_with('-') => {
                return Err(format!("unknown project batch option: {value}"));
            }
            value => manifests.push(value.to_string()),
        }
        index += 1;
    }
    if manifests.is_empty() {
        return Err("project batch requires at least one PROJECT.json".into());
    }
    let root =
        canonical_cli_project_root(root.as_deref().ok_or("project batch requires --root DIR")?)?;
    let output_dir = contained_project_directory(
        &root,
        output_dir
            .as_deref()
            .ok_or("project batch requires --output-dir DIR")?,
        "project batch output directory",
        true,
    )?;
    let mut requests = Vec::new();
    for raw in manifests {
        let manifest_path = contained_project_input(&root, &raw, "project batch manifest")?;
        let manifest = denoize::ProjectManifest::from_file(&manifest_path)?;
        let selected_timeline = timeline
            .clone()
            .or_else(|| manifest.timelines.first().map(|value| value.id.clone()))
            .ok_or("project batch manifest has no timeline")?;
        manifest.timeline(&selected_timeline)?;
        let output = output_dir.join(format!("{}.{}.wav", manifest.project_id, selected_timeline));
        requests.push(denoize::ProjectBatchRequest {
            manifest_path,
            timeline_id: Some(selected_timeline),
            output_path: output,
        });
    }
    let report = denoize::run_project_batch(
        &requests,
        &root,
        project_commit_mode(force),
        DecodeLimits::default(),
    )?;
    print_project_document(&report, pretty)
}

#[derive(Serialize)]
struct ProjectWatchCycleJson<'a> {
    schema: &'static str,
    schema_version: u32,
    root: &'a str,
    input: &'a str,
    output: &'a str,
    timeline: Option<&'a str>,
    cancelled: bool,
    #[serde(flatten)]
    report: &'a WatchCycleReport,
}

fn print_project_watch_cycle(
    root: &str,
    input: &str,
    output: &str,
    timeline: Option<&str>,
    report: &WatchCycleReport,
    cancelled: bool,
    pretty: bool,
) -> Result<(), String> {
    print_project_document(
        &ProjectWatchCycleJson {
            schema: denoize::PROJECT_WATCH_CYCLE_SCHEMA,
            schema_version: 1,
            root,
            input,
            output,
            timeline,
            cancelled,
            report,
        },
        pretty,
    )
}

fn run_project_watch(args: &[String]) -> Result<(), String> {
    if args.len() < 2 || args[0].starts_with('-') || args[1].starts_with('-') {
        return Err("project watch requires INPUT_DIR OUTPUT_DIR".into());
    }
    let input_raw = &args[0];
    let output_raw = &args[1];
    let mut root = None;
    let mut receipt_key = None;
    let mut timeline = None;
    let mut once = false;
    let mut recursive = false;
    let mut settle_millis = 2_000_u64;
    let mut poll_millis = 500_u64;
    let mut retry_initial_millis = 1_000_u64;
    let mut retry_max_millis = 60_000_u64;
    let mut max_attempts = 5_u32;
    let mut max_files = 10_000_usize;
    let mut pretty = false;
    let mut seen_settle = false;
    let mut seen_poll = false;
    let mut seen_retry_initial = false;
    let mut seen_retry_max = false;
    let mut seen_max_attempts = false;
    let mut seen_max_files = false;
    let mut index = 2;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--root" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut root, value, option)?;
            }
            "--receipt-key" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut receipt_key, value, option)?;
            }
            "--timeline" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut timeline, value, option)?;
            }
            "--settle-ms" if !seen_settle => {
                settle_millis = parse_project_watch_number(args, &mut index, option)?;
                seen_settle = true;
            }
            "--poll-ms" if !seen_poll => {
                poll_millis = parse_project_watch_number(args, &mut index, option)?;
                seen_poll = true;
            }
            "--retry-initial-ms" if !seen_retry_initial => {
                retry_initial_millis = parse_project_watch_number(args, &mut index, option)?;
                seen_retry_initial = true;
            }
            "--retry-max-ms" if !seen_retry_max => {
                retry_max_millis = parse_project_watch_number(args, &mut index, option)?;
                seen_retry_max = true;
            }
            "--max-attempts" if !seen_max_attempts => {
                max_attempts = parse_project_watch_number(args, &mut index, option)?;
                seen_max_attempts = true;
            }
            "--max-watch-files" if !seen_max_files => {
                max_files = parse_project_watch_number(args, &mut index, option)?;
                seen_max_files = true;
            }
            "--once" if !once => once = true,
            "--recursive" if !recursive => recursive = true,
            "--pretty" if !pretty => pretty = true,
            "--once" | "--recursive" | "--pretty" | "--settle-ms" | "--poll-ms"
            | "--retry-initial-ms" | "--retry-max-ms" | "--max-attempts" | "--max-watch-files" => {
                return Err(format!("{option} specified more than once"))
            }
            value => return Err(format!("unknown project watch option: {value}")),
        }
        index += 1;
    }
    let root =
        canonical_cli_project_root(root.as_deref().ok_or("project watch requires --root DIR")?)?;
    let input =
        contained_project_directory(&root, input_raw, "project watch input directory", true)?;
    let output =
        contained_project_directory(&root, output_raw, "project watch output directory", false)?;
    let key_raw = receipt_key.ok_or("project watch requires --receipt-key SECRET.json")?;
    let key_path = std::fs::canonicalize(&key_raw)
        .map_err(|error| format!("resolve project watch receipt key {key_raw}: {error}"))?;
    if key_path.starts_with(&input) || key_path.starts_with(&output) {
        return Err("project watch receipt key must be outside input and output trees".into());
    }
    let secret = denoize::ReceiptSecretKey::from_file(&key_path)?;
    let public = secret.public_key()?;
    let key_fingerprint = batch_resume::fingerprint_file(&key_path)?;
    let identity = format!(
        "denoize-project-watch-v1\nroot={}\ntimeline={}\nkey-id={}\nkey-fingerprint={:?}",
        root.display(),
        timeline.as_deref().unwrap_or("<first>"),
        public.key_id,
        key_fingerprint
    );
    let config = WatchFolderConfig::new(&input, &output, identity.as_bytes())
        .with_input_extensions(["json"])
        .with_output_extension("wav")
        .with_recursive(recursive)
        .with_settle_duration(Duration::from_millis(settle_millis))
        .with_poll_interval(Duration::from_millis(poll_millis))
        .with_retry_delays(
            Duration::from_millis(retry_initial_millis),
            Duration::from_millis(retry_max_millis),
        )
        .with_max_attempts(max_attempts)
        .with_max_files(max_files);
    let settle_duration = config.settle_duration();
    let poll_interval = config.poll_interval();
    let mut watch = WatchFolder::open(config)?;
    CANCELLED.store(false, Ordering::SeqCst);
    install_cancel_handler()?;
    let run_cycle = |watch: &mut WatchFolder| {
        watch.cycle(|job| {
            process_project_watch_job(
                job,
                &root,
                timeline.as_deref(),
                &key_path,
                key_fingerprint,
                &secret,
                &public,
            )
        })
    };
    let root_text = root.to_string_lossy();
    let input_text = input.to_string_lossy();
    let output_text = output.to_string_lossy();
    if once {
        let first = run_cycle(&mut watch)?;
        print_project_watch_cycle(
            &root_text,
            &input_text,
            &output_text,
            timeline.as_deref(),
            &first,
            false,
            pretty,
        )?;
        if first.observed != 0 && settle_duration != Duration::ZERO {
            wait_watch_interval(settle_duration);
            if !CANCELLED.load(Ordering::SeqCst) {
                let second = run_cycle(&mut watch)?;
                print_project_watch_cycle(
                    &root_text,
                    &input_text,
                    &output_text,
                    timeline.as_deref(),
                    &second,
                    false,
                    pretty,
                )?;
            }
        }
        return Ok(());
    }
    while !CANCELLED.load(Ordering::SeqCst) {
        match run_cycle(&mut watch) {
            Ok(report) => print_project_watch_cycle(
                &root_text,
                &input_text,
                &output_text,
                timeline.as_deref(),
                &report,
                false,
                pretty,
            )?,
            Err(error) => eprintln!("denoize: project watch scan failed; retrying: {error}"),
        }
        wait_watch_interval(poll_interval);
    }
    print_project_watch_cycle(
        &root_text,
        &input_text,
        &output_text,
        timeline.as_deref(),
        &WatchCycleReport::default(),
        true,
        pretty,
    )
}

fn parse_project_watch_number<T>(
    args: &[String],
    index: &mut usize,
    option: &str,
) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let raw = project_option_value(args, index, option)?;
    raw.parse()
        .map_err(|error| format!("invalid value for {option}: {error}"))
}

fn prepare_project_watch_plan(
    job: &WatchFolderJob,
    root: &std::path::Path,
    requested_timeline: Option<&str>,
) -> Result<
    (
        denoize::ProjectManifest,
        denoize::ProjectExecutionPlan,
        String,
    ),
    String,
> {
    if batch_resume::fingerprint_file(&job.input_path)? != job.input_fingerprint {
        return Err("project watch manifest changed after settling".into());
    }
    let manifest = denoize::ProjectManifest::from_file(&job.input_path)?;
    if batch_resume::fingerprint_file(&job.input_path)? != job.input_fingerprint {
        return Err("project watch manifest changed while it was parsed".into());
    }
    let timeline = requested_timeline
        .map(str::to_string)
        .or_else(|| manifest.timelines.first().map(|value| value.id.clone()))
        .ok_or("project watch manifest has no timeline")?;
    manifest.timeline(&timeline)?;
    let manifest_locator = denoize::portable_locator(&job.input_path, root)?;
    let reference = denoize::ProjectArtifactReference::new(
        "manifest",
        manifest_locator,
        job.input_fingerprint,
    )?;
    let output_locator = denoize::portable_locator(&job.output_path, root)?;
    let plan = denoize::ProjectExecutionPlan::new(
        &manifest,
        &timeline,
        reference,
        output_locator,
        CommitMode::NoClobber,
    )?;
    Ok((manifest, plan, timeline))
}

fn recover_project_watch_job(
    job: &WatchFolderJob,
    root: &std::path::Path,
    plan: &denoize::ProjectExecutionPlan,
    public: &ReceiptPublicKey,
) -> Result<bool, String> {
    let output_exists = path_exists_for_watch(&job.output_path)?;
    let receipt_exists = path_exists_for_watch(&job.receipt_path)?;
    match (output_exists, receipt_exists) {
        (false, false) => return Ok(false),
        (true, false) => {
            return Err(format!(
                "project watch output exists without its receipt: {}",
                job.output_path.display()
            ));
        }
        (false, true) => {
            return Err(format!(
                "project watch receipt exists without its output: {}",
                job.receipt_path.display()
            ));
        }
        (true, true) => {}
    }
    let receipt = denoize::SignedProjectExecutionReceipt::from_file(&job.receipt_path)?;
    receipt.verify_with_key(public, Some(plan), root)?;
    Ok(true)
}

fn classify_project_watch_error(error: String) -> WatchProcessError {
    let lowercase = error.to_ascii_lowercase();
    if lowercase.contains("changed after settling") || lowercase.contains("changed while") {
        return WatchProcessError::deferred(error);
    }
    if [
        "unsupported",
        "unknown field",
        "invalid",
        "differs from",
        "collides",
        "must",
        "cannot",
    ]
    .iter()
    .any(|needle| lowercase.contains(needle))
    {
        WatchProcessError::permanent(error)
    } else {
        WatchProcessError::retryable(error)
    }
}

#[allow(clippy::too_many_arguments)]
fn process_project_watch_job(
    job: &WatchFolderJob,
    root: &std::path::Path,
    requested_timeline: Option<&str>,
    key_path: &std::path::Path,
    expected_key_fingerprint: FileFingerprint,
    secret: &ReceiptSecretKey,
    public: &ReceiptPublicKey,
) -> Result<(), WatchProcessError> {
    let current_key = batch_resume::fingerprint_file(key_path).map_err(|error| {
        WatchProcessError::deferred(format!(
            "project watch receipt key is temporarily unavailable: {error}"
        ))
    })?;
    if current_key != expected_key_fingerprint {
        return Err(WatchProcessError::deferred(
            "project watch receipt key changed; restart with a fresh state path",
        ));
    }
    let (manifest, plan, timeline) = prepare_project_watch_plan(job, root, requested_timeline)
        .map_err(classify_project_watch_error)?;
    match recover_project_watch_job(job, root, &plan, public) {
        Ok(true) => return Ok(()),
        Ok(false) => {}
        Err(error) => return Err(WatchProcessError::permanent(error)),
    }
    let render = denoize::assemble_project_timeline(
        &manifest,
        &timeline,
        root,
        &job.output_path,
        CommitMode::NoClobber,
        DecodeLimits::default(),
    )
    .map_err(classify_project_watch_error)?;
    let receipt = denoize::SignedProjectExecutionReceipt::sign(&plan, render.output, secret)
        .map_err(WatchProcessError::permanent)?;
    denoize::write_signed_project_execution_receipt(
        &job.receipt_path,
        &receipt,
        CommitMode::NoClobber,
        false,
    )
    .map_err(WatchProcessError::permanent)?;
    match recover_project_watch_job(job, root, &plan, public) {
        Ok(true) => Ok(()),
        Ok(false) => Err(WatchProcessError::permanent(
            "project watch returned without publishing an output/receipt pair",
        )),
        Err(error) => Err(WatchProcessError::permanent(error)),
    }
}

fn project_option_value(
    args: &[String],
    index: &mut usize,
    option: &str,
) -> Result<String, String> {
    *index = index
        .checked_add(1)
        .ok_or("project argument index overflow")?;
    args.get(*index)
        .filter(|value| !value.is_empty())
        .cloned()
        .ok_or_else(|| format!("missing value for {option}"))
}

fn set_project_option<T>(slot: &mut Option<T>, value: T, option: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("{option} specified more than once"));
    }
    *slot = Some(value);
    Ok(())
}

fn print_project_document<T: Serialize>(value: &T, pretty: bool) -> Result<(), String> {
    let mut document = if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .map_err(|error| format!("serialize project result: {error}"))?;
    document.push('\n');
    std::io::stdout()
        .lock()
        .write_all(document.as_bytes())
        .map_err(|error| format!("write project result: {error}"))
}

fn project_commit_mode(force: bool) -> CommitMode {
    if force {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    }
}

fn canonical_cli_project_root(raw: &str) -> Result<std::path::PathBuf, String> {
    let root = std::fs::canonicalize(raw)
        .map_err(|error| format!("resolve project root {raw}: {error}"))?;
    if !root.is_dir() {
        return Err(format!(
            "project root is not a directory: {}",
            root.display()
        ));
    }
    Ok(root)
}

fn project_input_path(root: &std::path::Path, raw: &str) -> std::path::PathBuf {
    let path = std::path::PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}

fn contained_project_input(
    root: &std::path::Path,
    raw: &str,
    context: &str,
) -> Result<std::path::PathBuf, String> {
    let path = project_input_path(root, raw);
    let path = std::fs::canonicalize(&path)
        .map_err(|error| format!("resolve {context} {}: {error}", path.display()))?;
    if !path.starts_with(root) {
        return Err(format!(
            "{context} is outside project root {}",
            root.display()
        ));
    }
    Ok(path)
}

fn contained_project_output(
    root: &std::path::Path,
    raw: &str,
    context: &str,
) -> Result<std::path::PathBuf, String> {
    let requested = project_input_path(root, raw);
    let name = requested
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| format!("{context} must name a file"))?;
    let parent = requested
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(root);
    let parent = std::fs::canonicalize(parent)
        .map_err(|error| format!("resolve {context} parent {}: {error}", parent.display()))?;
    if !parent.starts_with(root) {
        return Err(format!(
            "{context} is outside project root {}",
            root.display()
        ));
    }
    let output = parent.join(name);
    if output == root {
        return Err(format!("{context} must name a file"));
    }
    Ok(output)
}

fn normalized_project_destination(
    path: &std::path::Path,
    context: &str,
) -> Result<std::path::PathBuf, String> {
    let requested = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("resolve current directory for {context}: {error}"))?
            .join(path)
    };
    let name = requested
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| format!("{context} must name a file"))?;
    let parent = requested
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let parent = std::fs::canonicalize(parent)
        .map_err(|error| format!("resolve {context} parent {}: {error}", parent.display()))?;
    Ok(parent.join(name))
}

fn reject_cli_project_publication_collision(
    manifest: &denoize::ProjectManifest,
    root: &std::path::Path,
    manifest_path: Option<&std::path::Path>,
    output: &std::path::Path,
    context: &str,
) -> Result<(), String> {
    let destination = normalized_project_destination(output, context)?;
    let existing_target = std::fs::canonicalize(&destination).ok();
    if let Some(manifest_path) = manifest_path {
        let manifest_path = std::fs::canonicalize(manifest_path).map_err(|error| {
            format!(
                "re-resolve project manifest {}: {error}",
                manifest_path.display()
            )
        })?;
        if destination == manifest_path || existing_target.as_ref() == Some(&manifest_path) {
            return Err(format!("{context} must not replace its project manifest"));
        }
    }

    let mut locators = Vec::new();
    for source in &manifest.sources {
        locators.push(source.locator.as_str());
        if let Some(license) = &source.license {
            locators.push(license.locator.as_str());
        }
    }
    for reference in manifest
        .settings
        .iter()
        .chain(&manifest.presets)
        .chain(&manifest.plans)
        .chain(&manifest.receipts)
    {
        locators.push(reference.locator.as_str());
    }
    for model in &manifest.models {
        locators.push(model.package.locator.as_str());
        locators.push(model.public_key.locator.as_str());
    }
    for locator in locators {
        let artifact = contained_project_input(root, locator, "project artifact")?;
        if destination == artifact || existing_target.as_ref() == Some(&artifact) {
            return Err(format!(
                "{context} collides with referenced project artifact {locator}"
            ));
        }
    }
    Ok(())
}

fn contained_project_directory(
    root: &std::path::Path,
    raw: &str,
    context: &str,
    must_exist: bool,
) -> Result<std::path::PathBuf, String> {
    let requested = project_input_path(root, raw);
    if must_exist || requested.exists() {
        let directory = std::fs::canonicalize(&requested)
            .map_err(|error| format!("resolve {context} {}: {error}", requested.display()))?;
        if !directory.is_dir() || !directory.starts_with(root) {
            return Err(format!(
                "{context} must be a directory below {}",
                root.display()
            ));
        }
        return Ok(directory);
    }
    let name = requested
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| format!("{context} must name a directory"))?;
    let parent = requested
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(root);
    let parent = std::fs::canonicalize(parent)
        .map_err(|error| format!("resolve {context} parent {}: {error}", parent.display()))?;
    if !parent.is_dir() || !parent.starts_with(root) {
        return Err(format!(
            "{context} is outside project root {}",
            root.display()
        ));
    }
    Ok(parent.join(name))
}

fn parse_project_binding(value: &str, option: &str) -> Result<(String, String), String> {
    let (id, path) = value
        .split_once('=')
        .ok_or_else(|| format!("{option} must use ID=PATH"))?;
    if id.is_empty() || path.is_empty() {
        return Err(format!("{option} must use non-empty ID=PATH"));
    }
    Ok((id.to_string(), path.to_string()))
}

fn parse_project_seconds(value: &str, field: &str) -> Result<f64, String> {
    let value = value
        .parse::<f64>()
        .map_err(|_| format!("invalid {field}: {value}"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{field} must be a finite non-negative value"));
    }
    Ok(value)
}

fn project_seconds_to_ticks(value: f64, timescale: u32, field: &str) -> Result<u64, String> {
    let ticks = value * f64::from(timescale);
    if !ticks.is_finite() || ticks > 9_007_199_254_740_991_f64 {
        return Err(format!("{field} does not fit the project timebase"));
    }
    Ok(ticks.round() as u64)
}

fn parse_project_channel_map(raw: &str, source_channels: u16) -> Result<Vec<u16>, String> {
    if raw.is_empty() {
        return Ok((0..source_channels).collect());
    }
    let channels = raw
        .split('+')
        .map(|value| {
            value
                .parse::<u16>()
                .map_err(|_| format!("invalid project channel index: {value}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if channels.is_empty() || channels.iter().any(|channel| *channel >= source_channels) {
        return Err(format!(
            "project channel map must reference channels below {source_channels}"
        ));
    }
    Ok(channels)
}

fn build_project_references(
    values: &[String],
    option: &str,
    root: &std::path::Path,
) -> Result<Vec<denoize::ProjectArtifactReference>, String> {
    values
        .iter()
        .map(|value| {
            let (id, raw) = parse_project_binding(value, option)?;
            let path = contained_project_input(root, &raw, option)?;
            denoize::project_artifact_reference(id, path, root)
        })
        .collect()
}

#[derive(Default)]
struct ProjectCreateOptions {
    root: Option<String>,
    project_id: Option<String>,
    timeline_id: Option<String>,
    sources: Vec<String>,
    source_licenses: Vec<String>,
    selections: Vec<String>,
    settings: Vec<String>,
    presets: Vec<String>,
    models: Vec<String>,
    plans: Vec<String>,
    receipts: Vec<String>,
    pretty: bool,
    force: bool,
}

fn run_project_create(args: &[String]) -> Result<(), String> {
    let output = args
        .first()
        .filter(|value| !value.starts_with('-'))
        .ok_or("project create requires PROJECT.json")?;
    let mut options = ProjectCreateOptions::default();
    let mut index = 1;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--root" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut options.root, value, option)?;
            }
            "--project-id" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut options.project_id, value, option)?;
            }
            "--timeline" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut options.timeline_id, value, option)?;
            }
            "--source" => options
                .sources
                .push(project_option_value(args, &mut index, option)?),
            "--source-license" => options
                .source_licenses
                .push(project_option_value(args, &mut index, option)?),
            "--selection" => options
                .selections
                .push(project_option_value(args, &mut index, option)?),
            "--setting" => options
                .settings
                .push(project_option_value(args, &mut index, option)?),
            "--preset" => options
                .presets
                .push(project_option_value(args, &mut index, option)?),
            "--model" => options
                .models
                .push(project_option_value(args, &mut index, option)?),
            "--plan" => options
                .plans
                .push(project_option_value(args, &mut index, option)?),
            "--receipt" => options
                .receipts
                .push(project_option_value(args, &mut index, option)?),
            "--pretty" if !options.pretty => options.pretty = true,
            "--force" if !options.force => options.force = true,
            "--pretty" | "--force" => return Err(format!("{option} specified more than once")),
            value => return Err(format!("unknown project create option: {value}")),
        }
        index += 1;
    }
    let root = canonical_cli_project_root(
        options
            .root
            .as_deref()
            .ok_or("project create requires --root DIR")?,
    )?;
    let project_id = options
        .project_id
        .ok_or("project create requires --project-id ID")?;
    if options.sources.is_empty() || options.selections.is_empty() {
        return Err("project create requires at least one --source and --selection".into());
    }

    let mut license_specs = BTreeMap::new();
    for value in &options.source_licenses {
        let (source_id, nested) = parse_project_binding(value, "--source-license")?;
        let (license_id, raw) = parse_project_binding(&nested, "--source-license")?;
        if license_specs
            .insert(source_id.clone(), (license_id, raw))
            .is_some()
        {
            return Err(format!("duplicate --source-license for {source_id}"));
        }
    }
    let mut seen_sources = BTreeSet::new();
    let mut sources = Vec::new();
    for value in &options.sources {
        let (id, raw) = parse_project_binding(value, "--source")?;
        if !seen_sources.insert(id.clone()) {
            return Err(format!("duplicate project source ID: {id}"));
        }
        let path = contained_project_input(&root, &raw, "project source")?;
        let inspection = denoize::inspect_project_source(&path, DecodeLimits::default())?;
        let locator = denoize::portable_locator(&path, &root)?;
        let license = license_specs
            .remove(&id)
            .map(|(license_id, raw)| {
                let path = contained_project_input(&root, &raw, "project source license")?;
                denoize::project_artifact_reference(license_id, path, &root)
            })
            .transpose()?;
        sources.push(denoize::ProjectSource::new(
            id, locator, inspection, license,
        )?);
    }
    if let Some((unknown, _)) = license_specs.into_iter().next() {
        return Err(format!(
            "--source-license references unknown project source {unknown}"
        ));
    }
    sources.sort_by(|left, right| left.id.cmp(&right.id));
    let source_map = sources
        .iter()
        .map(|source| (source.id.as_str(), source))
        .collect::<BTreeMap<_, _>>();
    let mut selections = Vec::new();
    for value in &options.selections {
        let (selection_id, specification) = value
            .split_once('=')
            .ok_or("--selection must use ID=SOURCE,START,DURATION[,CHANNEL_MAP[,PAD_BEFORE[,PAD_AFTER[,CROSSFADE]]]]")?;
        let fields = specification.split(',').collect::<Vec<_>>();
        if !(3..=7).contains(&fields.len()) {
            return Err("--selection must contain 3..=7 comma-separated fields".into());
        }
        let source = source_map.get(fields[0]).ok_or_else(|| {
            format!(
                "selection {selection_id} references unknown source {}",
                fields[0]
            )
        })?;
        let start = parse_project_seconds(fields[1], "selection start")?;
        let duration = parse_project_seconds(fields[2], "selection duration")?;
        if duration <= 0.0 {
            return Err("selection duration must be positive".into());
        }
        let channel_map =
            parse_project_channel_map(fields.get(3).copied().unwrap_or(""), source.channels)?;
        let padding_before = parse_project_seconds(
            fields.get(4).copied().unwrap_or("0"),
            "selection padding before",
        )?;
        let padding_after = parse_project_seconds(
            fields.get(5).copied().unwrap_or("0"),
            "selection padding after",
        )?;
        let crossfade =
            parse_project_seconds(fields.get(6).copied().unwrap_or("0"), "selection crossfade")?;
        selections.push(denoize::ProjectSelection::new(
            selection_id,
            source.id.clone(),
            denoize::PresentationRegion::from_seconds(
                source.fingerprint,
                source.timescale,
                start,
                duration,
            )?,
            channel_map,
            project_seconds_to_ticks(padding_before, source.timescale, "selection padding before")?,
            project_seconds_to_ticks(padding_after, source.timescale, "selection padding after")?,
            project_seconds_to_ticks(crossfade, source.timescale, "selection crossfade")?,
        )?);
    }
    let first = selections
        .first()
        .ok_or("project create requires at least one selection")?;
    let first_source = source_map
        .get(first.source_id.as_str())
        .ok_or("first project selection source disappeared")?;
    let channels = u16::try_from(first.channel_map.len())
        .map_err(|_| "project output channel count does not fit u16".to_string())?;
    let timeline = denoize::ProjectTimeline::new(
        options.timeline_id.unwrap_or_else(|| "main".into()),
        first_source.timescale,
        channels,
        selections,
    )?;

    let settings = build_project_references(&options.settings, "--setting", &root)?;
    let presets = build_project_references(&options.presets, "--preset", &root)?;
    let plans = build_project_references(&options.plans, "--plan", &root)?;
    let receipts = build_project_references(&options.receipts, "--receipt", &root)?;
    let mut models = Vec::new();
    for value in &options.models {
        let (id, paths) = value
            .split_once('=')
            .ok_or("--model must use ID=PACKAGE.dmp,PUBLIC_KEY")?;
        let (package_raw, key_raw) = paths
            .split_once(',')
            .ok_or("--model must use ID=PACKAGE.dmp,PUBLIC_KEY")?;
        let package_path = contained_project_input(&root, package_raw, "project model package")?;
        let key_path = contained_project_input(&root, key_raw, "project model public key")?;
        let package =
            denoize::project_artifact_reference(format!("{id}.package"), package_path, &root)?;
        let public_key =
            denoize::project_artifact_reference(format!("{id}.public-key"), key_path, &root)?;
        models.push(denoize::ProjectModelReference::open(
            id, package, public_key, &root,
        )?);
    }
    let manifest = denoize::ProjectManifest::new(
        project_id,
        sources,
        vec![timeline],
        settings,
        presets,
        models,
        plans,
        receipts,
    )?;
    reject_cli_project_publication_collision(
        &manifest,
        &root,
        None,
        std::path::Path::new(output),
        "project manifest output",
    )?;
    denoize::write_project_manifest(
        output,
        &manifest,
        project_commit_mode(options.force),
        options.pretty,
    )?;
    print_project_document(&manifest, options.pretty)
}

fn run_project_inspect(args: &[String]) -> Result<(), String> {
    let project = args
        .first()
        .filter(|value| !value.starts_with('-'))
        .ok_or("project inspect requires PROJECT.json")?;
    let mut pretty = false;
    for option in &args[1..] {
        match option.as_str() {
            "--pretty" if !pretty => pretty = true,
            "--pretty" => return Err("--pretty specified more than once".into()),
            value => return Err(format!("unknown project inspect option: {value}")),
        }
    }
    let manifest = denoize::ProjectManifest::from_file(project)?;
    print_project_document(&manifest, pretty)
}

struct ProjectPathOptions {
    root: std::path::PathBuf,
    output: Option<String>,
    timeline: Option<String>,
    pretty: bool,
    force: bool,
}

fn parse_project_root_and_output_options(
    args: &[String],
    start: usize,
    allow_output: bool,
) -> Result<ProjectPathOptions, String> {
    let mut root = None;
    let mut output = None;
    let mut timeline = None;
    let mut pretty = false;
    let mut force = false;
    let mut index = start;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--root" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut root, value, option)?;
            }
            "--output" if allow_output => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut output, value, option)?;
            }
            "--timeline" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut timeline, value, option)?;
            }
            "--pretty" if !pretty => pretty = true,
            "--force" if !force => force = true,
            "--pretty" | "--force" => return Err(format!("{option} specified more than once")),
            value => return Err(format!("unknown project option: {value}")),
        }
        index += 1;
    }
    let root = canonical_cli_project_root(
        root.as_deref()
            .ok_or("project command requires --root DIR")?,
    )?;
    Ok(ProjectPathOptions {
        root,
        output,
        timeline,
        pretty,
        force,
    })
}

fn run_project_validate(args: &[String]) -> Result<(), String> {
    let project = args
        .first()
        .filter(|value| !value.starts_with('-'))
        .ok_or("project validate requires PROJECT.json")?;
    let ProjectPathOptions {
        root,
        output,
        timeline,
        pretty,
        force,
    } = parse_project_root_and_output_options(args, 1, false)?;
    if output.is_some() || timeline.is_some() || force {
        return Err("project validate accepts only --root and --pretty".into());
    }
    let manifest = denoize::ProjectManifest::from_file(project)?;
    let report = denoize::validate_project_files(&manifest, root, DecodeLimits::default())?;
    print_project_document(&report, pretty)
}

fn run_project_assemble(args: &[String]) -> Result<(), String> {
    if args.len() < 2 || args[0].starts_with('-') || args[1].starts_with('-') {
        return Err("project assemble requires PROJECT.json OUTPUT.wav".into());
    }
    let project_raw = &args[0];
    let output_raw = &args[1];
    let mut root = None;
    let mut timeline = None;
    let mut supplied_plan = None;
    let mut receipt = None;
    let mut receipt_key = None;
    let mut pretty = false;
    let mut force = false;
    let mut index = 2;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--root" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut root, value, option)?;
            }
            "--timeline" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut timeline, value, option)?;
            }
            "--plan" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut supplied_plan, value, option)?;
            }
            "--receipt" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut receipt, value, option)?;
            }
            "--receipt-key" => {
                let value = project_option_value(args, &mut index, option)?;
                set_project_option(&mut receipt_key, value, option)?;
            }
            "--pretty" if !pretty => pretty = true,
            "--force" if !force => force = true,
            "--pretty" | "--force" => return Err(format!("{option} specified more than once")),
            value => return Err(format!("unknown project assemble option: {value}")),
        }
        index += 1;
    }
    if receipt.is_some() != receipt_key.is_some() {
        return Err("project assemble requires --receipt and --receipt-key together".into());
    }
    let root = canonical_cli_project_root(
        root.as_deref()
            .ok_or("project assemble requires --root DIR")?,
    )?;
    let project_path = if supplied_plan.is_some() || receipt.is_some() {
        contained_project_input(&root, project_raw, "project manifest")?
    } else {
        std::path::PathBuf::from(project_raw)
    };
    let manifest = denoize::ProjectManifest::from_file(&project_path)?;
    let timeline = timeline
        .or_else(|| {
            manifest
                .timelines
                .first()
                .map(|timeline| timeline.id.clone())
        })
        .ok_or("project has no timeline")?;
    let mode = project_commit_mode(force);
    let mut planned = None;
    let output_path = if supplied_plan.is_some() || receipt.is_some() {
        let output = contained_project_output(&root, output_raw, "project output")?;
        let manifest_reference =
            denoize::project_artifact_reference("manifest", &project_path, &root)?;
        let output_locator = denoize::portable_locator(&output, &root)?;
        let expected = denoize::ProjectExecutionPlan::new(
            &manifest,
            &timeline,
            manifest_reference,
            output_locator,
            mode,
        )?;
        if let Some(path) = supplied_plan.as_deref() {
            let supplied = denoize::ProjectExecutionPlan::from_file(path)?;
            if supplied != expected {
                return Err(format!(
                    "project assembly no longer matches supplied plan: supplied={} current={}",
                    supplied.digest()?,
                    expected.digest()?
                ));
            }
        }
        planned = Some(expected);
        output
    } else {
        std::path::PathBuf::from(output_raw)
    };

    let receipt_state = if let (Some(path), Some(key)) = (receipt, receipt_key) {
        let path = contained_project_output(&root, &path, "project receipt")?;
        if path == output_path {
            return Err("project receipt and audio output paths must differ".into());
        }
        match std::fs::symlink_metadata(&path) {
            Ok(_) => {
                return Err(format!(
                    "project receipt destination already exists: {}",
                    path.display()
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "inspect project receipt destination {}: {error}",
                    path.display()
                ));
            }
        }
        Some((path, denoize::ReceiptSecretKey::from_file(key)?))
    } else {
        None
    };
    reject_cli_project_publication_collision(
        &manifest,
        &root,
        Some(&project_path),
        &output_path,
        "project output",
    )?;
    let report = denoize::assemble_project_timeline(
        &manifest,
        &timeline,
        &root,
        &output_path,
        mode,
        DecodeLimits::default(),
    )?;
    if let Some((receipt_path, key)) = receipt_state {
        let plan = planned
            .as_ref()
            .ok_or("project receipt plan state disappeared after assembly")?;
        let signed = denoize::SignedProjectExecutionReceipt::sign(plan, report.output, &key)?;
        denoize::write_signed_project_execution_receipt(
            &receipt_path,
            &signed,
            CommitMode::NoClobber,
            pretty,
        )
        .map_err(|error| {
            format!(
                "project audio was published to {}, but its signed receipt could not be published to {}: {error}",
                output_path.display(),
                receipt_path.display()
            )
        })?;
    }
    print_project_document(&report, pretty)
}

fn run_project_relocate(args: &[String]) -> Result<(), String> {
    if args.len() < 3
        || args[0].starts_with('-')
        || args[1].starts_with('-')
        || args[2].starts_with('-')
    {
        return Err("project relocate requires PROJECT.json SOURCE_ID CANDIDATE".into());
    }
    let project = &args[0];
    let source_id = &args[1];
    let candidate_raw = &args[2];
    let ProjectPathOptions {
        root,
        output,
        timeline,
        pretty,
        force,
    } = parse_project_root_and_output_options(args, 3, true)?;
    if timeline.is_some() {
        return Err("project relocate does not accept --timeline".into());
    }
    let output = output.ok_or("project relocate requires --output PROJECT.json")?;
    let candidate = project_input_path(&root, candidate_raw);
    let manifest = denoize::ProjectManifest::from_file(project)?;
    let relocated = denoize::relocate_project_source(
        &manifest,
        source_id,
        candidate,
        &root,
        DecodeLimits::default(),
    )?;
    reject_cli_project_publication_collision(
        &manifest,
        &root,
        Some(std::path::Path::new(project)),
        std::path::Path::new(&output),
        "relocated project manifest output",
    )?;
    reject_cli_project_publication_collision(
        &relocated,
        &root,
        Some(std::path::Path::new(project)),
        std::path::Path::new(&output),
        "relocated project manifest output",
    )?;
    denoize::write_project_manifest(output, &relocated, project_commit_mode(force), pretty)?;
    print_project_document(&relocated, pretty)
}

fn sdk_usage() -> &'static str {
    "\
USAGE:
    denoize sdk capabilities [--json|--pretty]
    denoize sdk lifecycle [--json|--pretty]

COMMANDS:
    capabilities    print the frozen Stage 33 SDK feature matrix
    lifecycle       print the mobile route/lifecycle state-machine contract

The SDK never downloads a model implicitly. Unsupported backends and host
profiles remain explicit in the capability matrix."
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SdkOutputMode {
    Human,
    Json,
    PrettyJson,
}

fn parse_sdk_output_mode(args: &[String], command: &str) -> Result<SdkOutputMode, String> {
    match args {
        [] => Ok(SdkOutputMode::Human),
        [flag] if flag == "--json" => Ok(SdkOutputMode::Json),
        [flag] if flag == "--pretty" => Ok(SdkOutputMode::PrettyJson),
        _ => Err(format!(
            "sdk {command} accepts at most one of --json or --pretty"
        )),
    }
}

fn parse_sdk_document(source: &str, name: &str) -> Result<serde_json::Value, String> {
    serde_json::from_str(source).map_err(|error| format!("parse embedded {name}: {error}"))
}

fn print_sdk_document(document: &serde_json::Value, mode: SdkOutputMode) -> Result<(), String> {
    match mode {
        SdkOutputMode::Human => Ok(()),
        SdkOutputMode::Json => {
            println!(
                "{}",
                serde_json::to_string(document)
                    .map_err(|error| format!("serialize SDK document: {error}"))?
            );
            Ok(())
        }
        SdkOutputMode::PrettyJson => {
            println!(
                "{}",
                serde_json::to_string_pretty(document)
                    .map_err(|error| format!("serialize SDK document: {error}"))?
            );
            Ok(())
        }
    }
}

fn run_sdk(args: &[String]) -> Result<(), String> {
    if args.is_empty()
        || args
            .iter()
            .any(|argument| matches!(argument.as_str(), "-h" | "--help" | "help"))
    {
        print!("{}", sdk_usage());
        return Ok(());
    }
    let command = args[0].as_str();
    let mode = parse_sdk_output_mode(&args[1..], command)?;
    match command {
        "capabilities" => {
            let source = denoize::sdk::sdk_capabilities_json();
            let document = parse_sdk_document(source, "SDK capability matrix")?;
            let library_version = document
                .get("library_version")
                .and_then(serde_json::Value::as_str)
                .ok_or("embedded SDK capability matrix has no library_version")?;
            if library_version != VERSION {
                return Err(format!(
                    "SDK capability version {library_version} does not match binary version {VERSION}"
                ));
            }
            if mode != SdkOutputMode::Human {
                return print_sdk_document(&document, mode);
            }
            println!("SDK capabilities: v{library_version} (Stage 33)");
            println!("C ABI: stable ABI v1, classical scalar backend");
            println!("WASM: finite and incremental scalar processing");
            println!("Web Audio: Worker DSP with a non-blocking shared ring");
            println!("Android/iOS: worker wrappers with route-generation rebuilds");
            println!("WAM: optional and host-matrix gated");
            println!("No SDK call downloads or installs a model implicitly.");
            Ok(())
        }
        "lifecycle" => {
            let source = denoize::sdk::mobile_lifecycle_json();
            let document = parse_sdk_document(source, "mobile lifecycle contract")?;
            if mode != SdkOutputMode::Human {
                return print_sdk_document(&document, mode);
            }
            let state_count = document
                .get("states")
                .and_then(serde_json::Value::as_array)
                .map(Vec::len)
                .ok_or("embedded mobile lifecycle contract has no states")?;
            let transition_count = document
                .get("transitions")
                .and_then(serde_json::Value::as_array)
                .map(Vec::len)
                .ok_or("embedded mobile lifecycle contract has no transitions")?;
            println!("Mobile lifecycle: denoize-mobile-lifecycle-v1");
            println!("States: {state_count}; transitions: {transition_count}");
            println!(
                "Route, interruption, background, and memory events invalidate stale processors."
            );
            println!("Resume requires an explicit current route and creates a new generation.");
            Ok(())
        }
        value => Err(format!("unknown sdk command: {value}")),
    }
}

#[cfg(test)]
mod sdk_cli_tests {
    use super::*;

    #[test]
    fn sdk_output_mode_is_closed_and_unambiguous() {
        assert_eq!(
            parse_sdk_output_mode(&[], "capabilities").unwrap(),
            SdkOutputMode::Human
        );
        assert_eq!(
            parse_sdk_output_mode(&["--json".into()], "capabilities").unwrap(),
            SdkOutputMode::Json
        );
        assert!(
            parse_sdk_output_mode(&["--json".into(), "--pretty".into()], "capabilities").is_err()
        );
        assert!(parse_sdk_output_mode(&["--yaml".into()], "capabilities").is_err());
    }

    #[test]
    fn embedded_sdk_documents_are_valid_and_version_bound() {
        let capabilities = parse_sdk_document(
            denoize::sdk::sdk_capabilities_json(),
            "SDK capability matrix",
        )
        .unwrap();
        assert_eq!(capabilities["library_version"].as_str(), Some(VERSION));
        let lifecycle = parse_sdk_document(
            denoize::sdk::mobile_lifecycle_json(),
            "mobile lifecycle contract",
        )
        .unwrap();
        assert_eq!(lifecycle["transitions"].as_array().unwrap().len(), 8);
    }
}

fn run(args: &[String]) -> Result<(), String> {
    #[cfg(windows)]
    wait_for_isolation_gate()?;
    if args.first().map(String::as_str) == Some("hardware") {
        return run_hardware(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("ipc") {
        return run_ipc(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("recommend") {
        return run_recommend(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("diagnose") {
        return run_diagnose(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("assess") {
        return run_assess(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("restore") {
        return run_restore(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("universal") {
        return run_universal(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("target-speaker") {
        return run_target_speaker(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("target-sound") {
        return run_target_sound(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("meeting-speakers") {
        return run_meeting_speakers(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("music-restore") {
        return run_music_restoration(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("aec") {
        return run_aec(&args[1..]);
    }
    if matches!(
        args.first().map(String::as_str),
        Some("array" | "array-enhance")
    ) {
        return run_microphone_array(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("receipts") {
        return run_receipts(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("plan") {
        return run_execution_plan(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("watch") {
        return run_watch(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("live") {
        return run_live(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("models") {
        return run_models(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("evaluate") {
        return run_evaluate(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("metrics") {
        return run_metrics(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("compare") {
        return run_compare(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("plugin") {
        return run_plugin(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("update") {
        return run_update(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("project") {
        return run_project(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("sdk") {
        return run_sdk(&args[1..]);
    }
    let (input, output, ov) = parse_args(args)?;
    if ov.resume && !ov.batch && !ov.stream {
        return Err("--resume requires --batch or --stream".into());
    }
    validate_receipt_cli_options(&input, &output, &ov)?;
    validate_requested_execution_plan(&input, &output, &ov)?;
    if ov.isolate && std::env::var_os(ISOLATED_CHILD_ENV).is_none() {
        return run_isolated(args, &ov);
    }
    if ov.batch {
        if ov.stream {
            return Err("--stream cannot be combined with --batch".into());
        }
        return run_batch(&input, &output, &ov);
    }
    if ov.stream {
        return run_streaming_wav(&input, &output, ov);
    }
    run_one(&input, &output, ov)
}

fn validate_requested_execution_plan(
    input: &str,
    output: &str,
    options: &Overrides,
) -> Result<(), String> {
    let Some(path) = options.execution_plan.as_deref() else {
        return Ok(());
    };
    if input == "-" || output == "-" {
        return Err("--plan execution requires durable regular-file input and output".into());
    }
    if options.isolate {
        return Err("--plan cannot be combined with --isolate".into());
    }
    let supplied = ExecutionPlan::from_file(path)?;
    let expected = if options.batch {
        build_batch_execution_plan(
            std::path::Path::new(input),
            std::path::Path::new(output),
            options,
        )?
    } else if options.stream {
        build_stream_execution_plan(input, output, options)?
    } else {
        build_single_execution_plan(
            std::path::Path::new(input),
            std::path::Path::new(output),
            options,
        )?
    };
    if supplied != expected {
        return Err(format!(
            "execution no longer matches supplied plan: supplied={} current={}",
            supplied.digest()?,
            expected.digest()?
        ));
    }
    Ok(())
}

fn validate_receipt_cli_options(
    input: &str,
    output: &str,
    options: &Overrides,
) -> Result<(), String> {
    match (&options.receipt, &options.receipt_key) {
        (None, None) => return Ok(()),
        (Some(_), Some(_)) => {}
        _ => return Err("--receipt and --receipt-key must be supplied together".into()),
    }
    if (input == "-" || output == "-") && !options.stream {
        return Err("signed stdin/stdout receipts require --stream".into());
    }
    if options.report {
        return Err(
            "--receipt cannot be combined with --report because no output is published".into(),
        );
    }
    if options
        .receipt
        .as_deref()
        .is_some_and(|path| path.is_empty())
        || options
            .receipt_key
            .as_deref()
            .is_some_and(|path| path.is_empty())
    {
        return Err("receipt and receipt-key paths must not be empty".into());
    }
    if options.receipt.as_deref() == Some("-") || options.receipt_key.as_deref() == Some("-") {
        return Err("receipt and receipt-key must use durable regular-file paths".into());
    }
    Ok(())
}

#[cfg(unix)]
fn run_isolated(args: &[String], ov: &Overrides) -> Result<(), String> {
    use std::os::unix::process::CommandExt;

    let memory_limit = checked_mib_limit_bytes(ov.max_process_memory_mb, "--max-process-memory")?;
    let executable = std::env::current_exe()
        .map_err(|error| format!("locate denoize executable for --isolate: {error}"))?;
    let mut command = std::process::Command::new(executable);
    command.args(args).env(ISOLATED_CHILD_ENV, "1");
    if let Some(memory_limit) = memory_limit {
        let memory_limit = libc::rlim_t::try_from(memory_limit)
            .map_err(|_| "--max-process-memory exceeds this platform's RLIMIT_AS range")?;
        // SAFETY: `pre_exec` runs after fork and before exec. The closure only
        // performs async-signal-safe resource-limit syscalls and constructs an
        // `io::Error` from the captured errno on failure.
        unsafe {
            command.pre_exec(move || {
                let mut current = libc::rlimit {
                    rlim_cur: 0,
                    rlim_max: 0,
                };
                if libc::getrlimit(libc::RLIMIT_AS, &mut current) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                let limit = libc::rlimit {
                    rlim_cur: current.rlim_cur.min(memory_limit),
                    rlim_max: current.rlim_max,
                };
                if libc::setrlimit(libc::RLIMIT_AS, &limit) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    isolated_child_status(
        command
            .status()
            .map_err(|error| format!("start isolated denoize child: {error}"))?,
    )
}

#[cfg(windows)]
fn run_isolated(args: &[String], ov: &Overrides) -> Result<(), String> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
    };

    struct JobHandle(HANDLE);
    impl Drop for JobHandle {
        fn drop(&mut self) {
            // SAFETY: this wrapper uniquely owns the valid job handle.
            unsafe {
                CloseHandle(self.0);
            }
        }
    }

    let memory_limit = checked_mib_limit_bytes(ov.max_process_memory_mb, "--max-process-memory")?;
    // SAFETY: null security/name pointers request an unnamed job with default
    // security. The returned handle is checked and then uniquely owned.
    let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
    if job.is_null() {
        return Err(format!(
            "create Windows isolation job: {}",
            std::io::Error::last_os_error()
        ));
    }
    let job = JobHandle(job);
    let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if let Some(memory_limit) = memory_limit {
        limits.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        limits.ProcessMemoryLimit = usize::try_from(memory_limit)
            .map_err(|_| "--max-process-memory exceeds this platform's job-object range")?;
    }
    // SAFETY: the pointer and byte count describe `limits` for the documented
    // extended-limit information class, and the job handle remains live.
    if unsafe {
        SetInformationJobObject(
            job.0,
            JobObjectExtendedLimitInformation,
            std::ptr::from_ref(&limits).cast(),
            std::mem::size_of_val(&limits) as u32,
        )
    } == 0
    {
        return Err(format!(
            "configure Windows isolation job: {}",
            std::io::Error::last_os_error()
        ));
    }

    // The child waits on this private marker before parsing input. That closes
    // the normal-spawn race between `CreateProcess` and job assignment without
    // replacing stdin, which may carry a WAV stream.
    let gate = tempfile::NamedTempFile::new()
        .map_err(|error| format!("create Windows isolation gate: {error}"))?;
    let executable = std::env::current_exe()
        .map_err(|error| format!("locate denoize executable for --isolate: {error}"))?;
    let mut child = std::process::Command::new(executable)
        .args(args)
        .env(ISOLATED_CHILD_ENV, "1")
        .env(ISOLATION_GATE_ENV, gate.path())
        .spawn()
        .map_err(|error| format!("start isolated denoize child: {error}"))?;
    let process = child.as_raw_handle() as HANDLE;
    // SAFETY: `process` remains owned by `child`; assignment only associates
    // it with the live job and does not transfer or close either handle.
    if unsafe { AssignProcessToJobObject(job.0, process) } == 0 {
        let error = std::io::Error::last_os_error();
        let _ = child.kill();
        return Err(format!("assign isolated child to Windows job: {error}"));
    }
    drop(gate);
    let status = child
        .wait()
        .map_err(|error| format!("wait for isolated denoize child: {error}"))?;
    drop(job);
    isolated_child_status(status)
}

#[cfg(not(any(unix, windows)))]
fn run_isolated(_args: &[String], _ov: &Overrides) -> Result<(), String> {
    Err("--isolate is unavailable on this platform".into())
}

fn isolated_child_status(status: std::process::ExitStatus) -> Result<(), String> {
    if status.success() {
        Ok(())
    } else {
        Err(format!("isolated denoize child exited with {status}"))
    }
}

#[cfg(windows)]
fn wait_for_isolation_gate() -> Result<(), String> {
    let Some(path) = std::env::var_os(ISOLATION_GATE_ENV) else {
        return Ok(());
    };
    let path = std::path::PathBuf::from(path);
    while path.exists() {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    std::env::remove_var(ISOLATION_GATE_ENV);
    Ok(())
}

fn run_hardware(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        println!("USAGE:\n    denoize hardware [--json|--pretty]");
        return Ok(());
    }
    let mode = match args {
        [] => None,
        [flag] if flag == "--json" => Some(false),
        [flag] if flag == "--pretty" => Some(true),
        _ => return Err("hardware accepts only --json or --pretty".into()),
    };
    let report = denoize::hardware_capabilities();
    if let Some(pretty) = mode {
        let json = if pretty {
            report.to_pretty_json()?
        } else {
            report.to_json()?
        };
        println!("{json}");
        return Ok(());
    }
    println!(
        "host: {} {} ({} logical CPUs)",
        report.os(),
        report.architecture(),
        report.logical_cpus()
    );
    println!(
        "cpu-features: {}",
        if report.cpu_features().is_empty() {
            "none".into()
        } else {
            report.cpu_features().join(",")
        }
    );
    for runtime in report.runtimes() {
        let status = if runtime.available() {
            "available"
        } else if runtime.compiled() {
            "unavailable"
        } else {
            "not-compiled"
        };
        let mut details = Vec::new();
        if let Some(device) = runtime.device() {
            details.push(device.to_string());
        }
        if let Some(memory_bytes) = runtime.memory_bytes() {
            details.push(format_device_memory(memory_bytes));
        }
        if let Some(compute_capability) = runtime.compute_capability() {
            details.push(format!("compute capability {compute_capability}"));
        }
        if let Some(detail) = runtime.detail() {
            details.push(detail.to_string());
        }
        if details.is_empty() {
            println!("runtime {}: {status}", runtime.runtime().name());
        } else {
            println!(
                "runtime {}: {status} ({})",
                runtime.runtime().name(),
                details.join(", ")
            );
        }
    }
    println!("accelerated-backends:");
    for backend in report
        .backends()
        .iter()
        .filter(|backend| backend.accelerated())
    {
        println!("  {}", backend.backend());
    }
    Ok(())
}

fn receipts_usage() -> &'static str {
    "\
USAGE:
    denoize receipts keygen <SECRET_KEY.json> <PUBLIC_KEY.json>
    denoize receipts public-key <SECRET_KEY.json> <PUBLIC_KEY.json>
    denoize receipts policy create <POLICY.json> <PUBLIC_KEY.json>... [--revoke KEY_ID]...
    denoize receipts verify <RECEIPT.json> (--key PUBLIC_KEY.json | --policy POLICY.json) [OPTIONS]

VERIFY OPTIONS:
        --plan <PLAN.json>   require exact correspondence to a read-only plan
        --output-root <DIR> anchor portable output locators below DIR
        --output <FILE>      exact file that captured a v2 stdout stream
        --json               emit compact verification JSON
        --pretty             emit indented verification JSON

Secret keys are unencrypted and generated owner-only. Receipts never embed a
trust key: verification requires an explicit public key or a rotation/revocation
policy. Without --output-root, file locators are anchored beside the receipt.
Stdout stream receipts use the `-` locator and require --output during verification.
"
}

fn run_receipts(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        if args.len() != 1 {
            return Err("receipts --help accepts no other arguments".into());
        }
        print!("{}", receipts_usage());
        return Ok(());
    }
    match args.first().map(String::as_str) {
        Some("keygen") => match &args[1..] {
            [secret, public] => {
                let key_id = denoize::write_new_receipt_keypair(secret, public)?;
                println!("generated receipt signing key {key_id}");
                println!("secret: {secret}");
                println!("public: {public}");
                Ok(())
            }
            _ => Err("receipts keygen requires SECRET_KEY.json and PUBLIC_KEY.json".into()),
        },
        Some("public-key") => match &args[1..] {
            [secret, public] => {
                let key_id = denoize::export_receipt_public_key(secret, public)?;
                println!("exported receipt public key {key_id} to {public}");
                Ok(())
            }
            _ => Err("receipts public-key requires SECRET_KEY.json and PUBLIC_KEY.json".into()),
        },
        Some("policy") => run_receipt_policy(&args[1..]),
        Some("verify") => run_receipt_verify(&args[1..]),
        Some(command) => Err(format!("unknown receipts command: {command}")),
        None => Err("receipts requires a command (run `denoize receipts --help`)".into()),
    }
}

fn run_receipt_policy(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) != Some("create") {
        return Err("receipts policy supports only `create`".into());
    }
    let destination = args
        .get(1)
        .ok_or("receipts policy create requires POLICY.json")?;
    let mut public_paths = Vec::new();
    let mut revoked = Vec::new();
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--revoke" => {
                index += 1;
                let key_id = args
                    .get(index)
                    .ok_or("missing key ID for --revoke")?
                    .clone();
                revoked.push(key_id);
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown receipts policy option: {value}"));
            }
            value => public_paths.push(value.to_string()),
        }
        index += 1;
    }
    if public_paths.is_empty() {
        return Err("receipts policy create requires at least one public key".into());
    }
    let keys = public_paths
        .iter()
        .map(ReceiptPublicKey::from_file)
        .collect::<Result<Vec<_>, String>>()?;
    let policy = ReceiptTrustPolicy::new(keys, revoked)?;
    denoize::write_receipt_trust_policy(destination, &policy)?;
    println!(
        "created receipt trust policy with {} trusted key(s) and {} revocation(s): {}",
        policy.trusted_keys.len(),
        policy.revoked_key_ids.len(),
        destination
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReceiptVerificationOutput {
    Human,
    Json,
    PrettyJson,
}

fn run_receipt_verify(args: &[String]) -> Result<(), String> {
    let receipt_path = args
        .first()
        .filter(|value| !value.starts_with('-'))
        .ok_or("receipts verify requires RECEIPT.json")?;
    let mut public_key = None;
    let mut policy = None;
    let mut plan_path = None;
    let mut output_root = None;
    let mut stream_output = None;
    let mut output = ReceiptVerificationOutput::Human;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--key" => {
                index += 1;
                public_key = Some(
                    args.get(index)
                        .ok_or("missing public key path for --key")?
                        .clone(),
                );
            }
            "--policy" => {
                index += 1;
                policy = Some(
                    args.get(index)
                        .ok_or("missing trust policy path for --policy")?
                        .clone(),
                );
            }
            "--plan" => {
                index += 1;
                plan_path = Some(
                    args.get(index)
                        .ok_or("missing execution plan path for --plan")?
                        .clone(),
                );
            }
            "--output-root" => {
                index += 1;
                output_root = Some(
                    args.get(index)
                        .ok_or("missing directory path for --output-root")?
                        .clone(),
                );
            }
            "--output" => {
                index += 1;
                stream_output = Some(
                    args.get(index)
                        .ok_or("missing captured stream path for --output")?
                        .clone(),
                );
            }
            "--json" => {
                if output != ReceiptVerificationOutput::Human {
                    return Err("receipts verify accepts only one of --json or --pretty".into());
                }
                output = ReceiptVerificationOutput::Json;
            }
            "--pretty" => {
                if output != ReceiptVerificationOutput::Human {
                    return Err("receipts verify accepts only one of --json or --pretty".into());
                }
                output = ReceiptVerificationOutput::PrettyJson;
            }
            value => return Err(format!("unknown receipts verify option: {value}")),
        }
        index += 1;
    }
    match (&public_key, &policy) {
        (Some(_), Some(_)) | (None, None) => {
            return Err("receipts verify requires exactly one of --key or --policy".into());
        }
        _ => {}
    }

    let receipt = SignedExecutionReceipt::from_file(receipt_path)?;
    let root = output_root.as_deref().map(std::path::Path::new);
    let stream_output = stream_output.as_deref().map(std::path::Path::new);
    let report = match (public_key, policy) {
        (Some(path), None) => {
            let key = ReceiptPublicKey::from_file(path)?;
            receipt.verify_signature(&key)?;
            let plan = plan_path
                .as_deref()
                .map(ExecutionPlan::from_file)
                .transpose()?;
            receipt.verify_with_key_at_stream_output(
                &key,
                plan.as_ref(),
                std::path::Path::new(receipt_path),
                root,
                stream_output,
            )?
        }
        (None, Some(path)) => {
            let policy = ReceiptTrustPolicy::from_file(path)?;
            receipt.verify_policy(&policy)?;
            let plan = plan_path
                .as_deref()
                .map(ExecutionPlan::from_file)
                .transpose()?;
            receipt.verify_with_policy_at_stream_output(
                &policy,
                plan.as_ref(),
                std::path::Path::new(receipt_path),
                root,
                stream_output,
            )?
        }
        _ => return Err("receipt verification authority changed after validation".into()),
    };
    match output {
        ReceiptVerificationOutput::Json => println!("{}", report.to_json()?),
        ReceiptVerificationOutput::PrettyJson => println!("{}", report.to_pretty_json()?),
        ReceiptVerificationOutput::Human => {
            println!(
                "verified receipt: key={} plan={} outputs={}",
                report.key_id,
                report.plan_digest,
                report.verified_items.len()
            );
            for item in &report.verified_items {
                println!(
                    "  {}  {}  {} bytes  {}",
                    item.item_id, item.output_path, item.output.len, item.output.digest
                );
            }
        }
    }
    Ok(())
}

fn execution_plan_usage() -> &'static str {
    "\
USAGE:
    denoize plan <INPUT> <OUTPUT> [PROCESSING OPTIONS] [--pretty]
    denoize plan <INPUT|-> <OUTPUT|-> --stream [STREAM OPTIONS] [--pretty]
    denoize plan <INPUT_DIR> <OUTPUT_DIR> --batch [BATCH OPTIONS] [--pretty]

The command performs the same bounded decode, model verification, backend
preparation, resource admission, recipe hashing, and destination validation as
execution, but never creates output, lock, journal, checkpoint, or model state.
It emits v1 file/batch or v2 bounded-stream execution-plan JSON to stdout. Paths
are portable relative locators, never absolute paths; `-` identifies stdin or
stdout only in a v2 stream plan. Planning stdin consumes it into a bounded spool.
"
}

fn run_execution_plan(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        if args.len() != 1 {
            return Err("plan --help accepts no other arguments".into());
        }
        print!("{}", execution_plan_usage());
        return Ok(());
    }
    let mut parseable = Vec::with_capacity(args.len());
    let mut pretty = false;
    for argument in args {
        if argument == "--pretty" {
            if pretty {
                return Err("plan accepts --pretty only once".into());
            }
            pretty = true;
        } else {
            parseable.push(argument.clone());
        }
    }
    let (input, output, options) = parse_args(&parseable)?;
    if options.report {
        return Err("plan cannot be combined with --report".into());
    }
    if options.isolate {
        return Err("plan is already read-only and cannot be combined with --isolate".into());
    }
    if options.receipt.is_some() || options.receipt_key.is_some() {
        return Err(
            "plan cannot publish a receipt; pass --receipt and --receipt-key to execution".into(),
        );
    }
    if options.execution_plan.is_some() {
        return Err("plan cannot be combined with execution --plan".into());
    }
    if options.batch && options.stream {
        return Err("plan cannot combine --batch and --stream".into());
    }
    let plan = if options.batch {
        build_batch_execution_plan(
            std::path::Path::new(&input),
            std::path::Path::new(&output),
            &options,
        )?
    } else if options.stream {
        build_stream_execution_plan(&input, &output, &options)?
    } else {
        if input == "-" || output == "-" {
            return Err("stdin/stdout execution plans require --stream".into());
        }
        build_single_execution_plan(
            std::path::Path::new(&input),
            std::path::Path::new(&output),
            &options,
        )?
    };
    if pretty {
        println!("{}", plan.to_pretty_json()?);
    } else {
        println!("{}", plan.to_json()?);
    }
    Ok(())
}

fn build_batch_execution_plan(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
    options: &Overrides,
) -> Result<ExecutionPlan, String> {
    validate_effective_options(options, VALIDATION_SAMPLE_RATE)?;
    if !input_dir.is_dir() {
        return Err(format!(
            "batch input is not a directory: {}",
            input_dir.display()
        ));
    }
    validate_batch_directories(input_dir, output_dir)?;
    let encode_options = build_encode_options(options)?;
    let resolved_backend_options = resolve_explicit_backend_options_read_only(options)?;
    let jobs = effective_batch_jobs(options);
    let governor = resource_governor(options, jobs)?;
    let output_extension = options
        .output_format
        .as_deref()
        .map(normalize_output_extension)
        .transpose()?;
    let files = collect_batch_files(input_dir, options.recursive)?;
    if files.is_empty() {
        return Err("batch input contains no supported audio files".into());
    }
    let items = plan_batch_files_with_limits(
        input_dir,
        output_dir,
        files,
        output_extension,
        decode_limits_for_options(options)?,
    )?;
    for (path, name) in [
        (output_dir.join(STATE_FILE_NAME), STATE_FILE_NAME),
        (
            output_dir.join(LEGACY_DESKTOP_STATE_FILE_NAME),
            LEGACY_DESKTOP_STATE_FILE_NAME,
        ),
        (output_dir.join(LOCK_FILE_NAME), LOCK_FILE_NAME),
    ] {
        validate_batch_reserved_path(&items, &path, name)?;
    }
    validate_encode_preflight(encode_options, items.iter().map(|item| item.output_format))?;
    let prepared = preflight_batch_items(
        &items,
        options,
        encode_options,
        resolved_backend_options.as_ref(),
        &governor,
        true,
    )?;
    let expectations = prepared
        .iter()
        .map(|item| item.expectation.clone())
        .collect::<Vec<_>>();
    let decisions = batch_resume::inspect_batch_decisions_with_evidence(
        output_dir,
        options.resume,
        &expectations,
        options.force,
    )?;
    if decisions.len() != prepared.len() {
        return Err("read-only batch decision count does not match the plan".into());
    }
    let planned = prepared
        .into_iter()
        .zip(decisions)
        .map(|(prepared, evidence)| PlannedBatchItem {
            prepared,
            decision: evidence.decision(),
            existing_output: evidence.existing_output(),
        })
        .collect::<Vec<_>>();
    for item in &planned {
        item.prepared.expectation.verify_sources()?;
    }
    build_batch_execution_plan_from_planned(input_dir, output_dir, options, &planned)
}

fn build_batch_execution_plan_from_planned(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
    options: &Overrides,
    planned: &[PlannedBatchItem],
) -> Result<ExecutionPlan, String> {
    let metadata_policy = if options.no_metadata {
        MetadataPolicy::Drop
    } else {
        MetadataPolicy::Preserve
    };
    let mut plan_items = Vec::with_capacity(planned.len());
    for planned in planned {
        let prepared = &planned.prepared;
        let decision = planned.decision;
        prepared.expectation.verify_sources()?;
        let input_locator = denoize::portable_locator(&prepared.item.input, input_dir)?;
        let output_locator = denoize::portable_locator(&prepared.item.destination, output_dir)?;
        let input_fingerprint = prepared.expectation.input_fingerprint();
        let item_id =
            denoize::execution_item_id(input_fingerprint, &output_locator, prepared.recipe)?;
        let (publication, action, resources) = match decision {
            ResumeDecision::Skip { .. } => {
                ("none", "skip", planned_resources(ResourceRequest::new()))
            }
            ResumeDecision::Process { commit_mode, .. } => {
                let request = prepared
                    .resource_request
                    .checked_add(backend_session_request(&prepared.resolved_processing)?)?;
                let publication = match commit_mode {
                    CommitMode::Replace => "replace",
                    CommitMode::NoClobber => "no-clobber",
                };
                (publication, "process", planned_resources(request))
            }
        };
        let model = match prepared.expectation.model() {
            Some(model) => Some(PlannedArtifact {
                path: denoize::portable_file_locator(&model.path)?,
                fingerprint: model.fingerprint,
            }),
            None => None,
        };
        plan_items.push(ExecutionPlanItem {
            item_id,
            input: PlannedArtifact {
                path: input_locator,
                fingerprint: input_fingerprint,
            },
            output: PlannedOutput {
                path: output_locator,
                format: output_format_name(prepared.item.output_format).into(),
                publication: publication.into(),
                action: action.into(),
                reason: decision.reason().as_str().into(),
                existing_fingerprint: planned.existing_output,
            },
            model,
            recipe: prepared.recipe,
            backend: service::backend_name(prepared.resolved_processing.backend).into(),
            accelerator: prepared
                .resolved_processing
                .accelerator
                .effective()
                .name()
                .into(),
            input_format: audio_format_name(prepared.item.probe.format).into(),
            input_codec: audio_codec_name(prepared.item.probe.codec).into(),
            channels: prepared.channels as u64,
            frames: prepared.frames,
            sample_rate: prepared.sample_rate,
            resources,
        });
    }
    ExecutionPlan::new(
        ExecutionKind::Batch,
        options.deterministic,
        metadata_policy_name(metadata_policy),
        plan_items,
    )
}

fn build_stream_execution_plan(
    input: &str,
    output: &str,
    options: &Overrides,
) -> Result<ExecutionPlan, String> {
    validate_effective_options(options, VALIDATION_SAMPLE_RATE)?;
    let standard_input = input == "-";
    let standard_output = output == "-";
    if options.resume && (standard_input || standard_output) {
        return Err(
            "--resume requires durable regular-file stream input and output; stdin/stdout spools are intentionally ephemeral"
                .into(),
        );
    }
    let input_path = std::path::Path::new(input);
    let output_path = std::path::Path::new(output);
    let output_format = if standard_output {
        match options.output_format.as_deref() {
            Some(extension) => {
                OutputFormat::from_path(&std::path::PathBuf::from(format!("output.{extension}")))?
            }
            None => OutputFormat::Wav,
        }
    } else {
        OutputFormat::from_path(output_path)?
    };
    let encode_options = build_encode_options(options)?;
    validate_encode_preflight(encode_options, [output_format])?;
    let governor = resource_governor(options, 1)?;
    let configured_temporary_limit = governor.limits().max_temporary_bytes();
    let stdio_spool_limits = StreamSpoolLimits::new(
        configured_temporary_limit.unwrap_or_else(|| StreamSpoolLimits::default().max_bytes()),
    );
    let stdio_request = if standard_input || standard_output {
        ResourceRequest::new().with_temporary_bytes(stdio_spool_limits.max_bytes())
    } else {
        ResourceRequest::new()
    };
    let _stdio_permit = if standard_input || standard_output {
        Some(governor.acquire(stdio_request)?)
    } else {
        None
    };

    let backend = if options.auto_backend {
        service::select_live_backend()
    } else {
        options.backend.unwrap_or(Backend::Classical)
    };
    if !StreamingBackendSession::supports(backend) {
        return Err(format!(
            "backend {} does not support --stream",
            service::backend_name(backend)
        ));
    }
    let backend_options =
        service::resolve_backend_options_read_only(backend, build_backend_options(options)?)?;
    let accelerator = denoize::select_accelerator_for_options(backend, &backend_options)?;
    let initial_publication = if options.resume {
        None
    } else if standard_output {
        Some(("stdout", "non-seekable"))
    } else {
        Some(planned_publication(output_path, options.force)?)
    };

    let mut input_session = if standard_input {
        let stdin = std::io::stdin();
        AudioInputSession::from_reader_with_limits(stdin.lock(), stdio_spool_limits)?
    } else {
        AudioInputSession::open(input_path)?
    };
    let input_spool_bytes = if standard_input {
        input_session.len()
    } else {
        0
    };
    let remaining_stdio_spool_bytes = stdio_spool_limits
        .max_bytes()
        .checked_sub(input_spool_bytes)
        .ok_or_else(|| "stdin spool exceeded the shared non-seekable spool limit".to_string())?;
    let effective_memory_mb = effective_input_memory_mb(options);
    let effective_memory_bytes = effective_input_memory_limit_bytes(options)?;
    let initial_metadata_limits = metadata_limits_for_available_bytes(effective_memory_bytes);
    let initial_decode_limits = DecodeLimits::new(initial_metadata_limits, effective_memory_bytes);
    let stream_info = inspect_audio_stream_session(&mut input_session, initial_decode_limits)?;
    let spec = stream_info.output_spec;
    let encode_spec =
        StreamEncodeSpec::new(spec, stream_info.channel_mask, stream_info.total_frames);
    output_format.validate_stream_config(encode_spec, encode_options)?;
    let effective_temporary_limit = if standard_input || standard_output {
        Some(remaining_stdio_spool_bytes)
    } else {
        configured_temporary_limit
    };
    let auxiliary_limit = match (stream_info.total_frames, effective_temporary_limit) {
        (None, Some(limit)) => limit / 3,
        (_, Some(limit)) => limit,
        (_, None) => StreamEncodeLimits::default().max_auxiliary_temporary_bytes(),
    };
    let encode_limits = StreamEncodeLimits::new(auxiliary_limit);
    validate_effective_options(options, spec.sample_rate)?;
    let cfg = build_config(options, spec.sample_rate);
    let block_frames = options.stream_frames.unwrap_or(STREAM_BLOCK_FRAMES);
    let base_stream_working_set = estimate_stream_memory_bytes_checked(
        spec.channels as usize,
        block_frames,
        cfg.frame_size,
        spec.sample_rate,
        cfg.profile_ms,
    )
    .map_err(|error| error.to_string())?;
    let backend_stream_state = StreamingBackendSession::estimate_additional_bytes(
        backend,
        spec.sample_rate,
        spec.channels as usize,
        backend_options.channel_mode,
    )
    .map_err(|error| error.to_string())?;
    let vad_stream_state = if cfg.vad {
        StreamingBackendSession::estimate_vad_additional_bytes(
            spec.sample_rate,
            spec.channels as usize,
            block_frames,
            cfg.frame_size,
            cfg.profile_ms,
        )
        .map_err(|error| error.to_string())?
    } else {
        0
    };
    let loudness_stream_state = if options.loudness_lufs.is_some() {
        denoize::loudness::estimate_streaming_loudness_bytes(
            spec.channels as usize,
            spec.sample_rate,
            block_frames,
        )
        .map_err(|error| error.to_string())?
    } else {
        0
    };
    let encoder_stream_state = denoize::estimate_stream_encode_additional_bytes(
        output_format,
        encode_spec,
        block_frames,
        encode_options,
    )?;
    let stream_working_set = base_stream_working_set
        .checked_add(backend_stream_state)
        .and_then(|bytes| bytes.checked_add(vad_stream_state))
        .and_then(|bytes| bytes.checked_add(loudness_stream_state))
        .and_then(|bytes| bytes.checked_add(stream_info.decoder_additional_bytes))
        .and_then(|bytes| bytes.checked_add(encoder_stream_state))
        .and_then(|bytes| {
            bytes.checked_add(denoize::estimate_backend_worker_memory_bytes(
                &backend_options,
            ))
        })
        .and_then(|bytes| {
            bytes.checked_add(if options.resume {
                batch_resume::STREAM_CHECKPOINT_SCRATCH_BYTES
            } else {
                0
            })
        })
        .ok_or_else(|| "stream plan working-set estimate overflow".to_string())?;
    let verification_block_frames = block_frames.min(STREAM_BLOCK_FRAMES);
    let initial_verification_working_set = denoize::estimate_stream_output_verification_bytes(
        output_format,
        encode_spec,
        verification_block_frames,
        encode_options,
        encode_limits,
        initial_decode_limits,
    )?;
    let initial_worker_memory = stream_working_set.max(initial_verification_working_set);
    ensure_memory_limit(
        initial_worker_memory,
        effective_memory_mb,
        "stream plan working set",
    )?;
    let metadata_limits = retained_metadata_limits(effective_memory_mb, initial_worker_memory)?;
    let decode_limits = DecodeLimits::new(metadata_limits, effective_memory_bytes);
    let final_info = inspect_audio_stream_session(&mut input_session, decode_limits)?;
    if final_info.format != stream_info.format
        || final_info.codec != stream_info.codec
        || final_info.output_spec != stream_info.output_spec
        || final_info.channel_mask != stream_info.channel_mask
        || final_info.total_frames != stream_info.total_frames
        || final_info.max_decoder_frames != stream_info.max_decoder_frames
    {
        return Err("stream input geometry changed during read-only planning".into());
    }
    let verification_working_set = denoize::estimate_stream_output_verification_bytes(
        output_format,
        encode_spec,
        verification_block_frames,
        encode_options,
        encode_limits,
        decode_limits,
    )?;
    ensure_memory_limit(
        stream_working_set.max(verification_working_set),
        effective_memory_mb,
        "stream plan working set",
    )?;
    let metadata_policy = if options.no_metadata {
        MetadataPolicy::Drop
    } else {
        MetadataPolicy::Preserve
    };
    let metadata_bytes = if metadata_policy == MetadataPolicy::Preserve {
        input_session
            .read_metadata_with_limits(metadata_limits)?
            .as_ref()
            .map(denoize::metadata::Metadata::estimated_memory_bytes)
            .unwrap_or(0)
    } else {
        0
    };
    let worker_memory_bytes = stream_working_set
        .checked_add(metadata_bytes)
        .ok_or_else(|| "stream plan memory reservation overflow".to_string())?
        .max(verification_working_set);
    let temporary_reservation = if standard_output {
        let auxiliary = denoize::estimate_stream_encode_temporary_bytes(
            output_format,
            encode_spec,
            encode_options,
            encode_limits,
        )?;
        let total = denoize::estimate_spooled_stream_output_bytes(
            output_format,
            encode_spec,
            encode_options,
            encode_limits,
        )?;
        let total = match total {
            Some(bytes) => bytes
                .checked_add(metadata_bytes)
                .ok_or_else(|| "non-seekable output metadata size overflows".to_string())?,
            None => remaining_stdio_spool_bytes,
        };
        if total > remaining_stdio_spool_bytes {
            return Err(format!(
                "non-seekable output requires {total} bytes, but stdin and output share only {remaining_stdio_spool_bytes} remaining spool bytes"
            ));
        }
        StreamTemporaryReservation {
            total_bytes: total,
            encoder_auxiliary_bytes: auxiliary,
            checkpoint_limit: None,
        }
    } else {
        stream_temporary_reservation_bytes(
            final_info,
            output_format,
            encode_spec,
            encode_options,
            encode_limits,
            effective_temporary_limit,
            options.resume,
            options.loudness_lufs.is_some(),
            metadata_bytes,
        )?
    };
    let admitted_temporary_bytes = if standard_input || standard_output {
        0
    } else {
        temporary_reservation.total_bytes
    };
    let mut worker_request = ResourceRequest::worker(worker_memory_bytes, admitted_temporary_bytes);
    if accelerator.effective() != denoize::AcceleratorRuntime::Cpu {
        let gpu_memory = stream_working_set
            .checked_mul(2)
            .and_then(|bytes| {
                bytes.checked_add(denoize::estimate_backend_worker_gpu_memory_bytes(
                    &backend_options,
                ))
            })
            .ok_or_else(|| "stream plan GPU reservation overflow".to_string())?;
        worker_request = worker_request
            .with_gpu_jobs(1)
            .with_gpu_memory_bytes(gpu_memory);
    }
    let admission_request = worker_request.checked_add(backend_resource_request(
        backend,
        &backend_options,
        accelerator,
    )?)?;
    drop(governor.try_acquire(admission_request)?.ok_or_else(|| {
        "stream execution plan cannot be admitted under the configured resource limits".to_string()
    })?);
    let reported_request = admission_request.checked_add(stdio_request)?;

    let resolved = service::ResolvedProcessingOptions {
        backend,
        denoiser: cfg.clone(),
        backend_options: backend_options.clone(),
        accelerator,
        loudness_lufs: options.loudness_lufs,
        true_peak_dbtp: options.true_peak_dbtp.unwrap_or(-1.0),
    };
    resolved.validate_config()?;
    let model = if options.resume {
        batch_resume::resumable_consumed_model(&resolved)?
    } else {
        batch_resume::consumed_model(&resolved)?
    };
    let _processor = StreamingBackendSession::new_with_accelerator(
        backend,
        spec.sample_rate,
        spec.channels as usize,
        cfg,
        backend_options,
        accelerator,
    )?;
    if let Some(model) = &model {
        if batch_resume::fingerprint_file(&model.path)? != model.fingerprint {
            return Err(format!(
                "selected streaming model changed during read-only planning: {}",
                model.path.display()
            ));
        }
    }
    let base_recipe = batch_resume::recipe_digest(
        &resolved,
        spec.channels as usize,
        output_format,
        encode_options,
        metadata_policy,
        model
            .as_ref()
            .map(|model| (&model.fingerprint, model.sample_rate)),
    )?;
    let recipe = batch_resume::stream_recipe_digest(base_recipe, block_frames, final_info)?;

    let mut reader = AudioStreamReader::from_session(input_session, decode_limits)?;
    let input_fingerprint = reader.fingerprint_input()?;
    let mut frames = 0_u64;
    while let Some(block) = reader.next_block(block_frames)? {
        frames = frames
            .checked_add(block.first().map(Vec::len).unwrap_or(0) as u64)
            .ok_or_else(|| "stream plan frame count overflows".to_string())?;
    }
    if reader.fingerprint_input()? != input_fingerprint {
        return Err("stream input changed while it was planned".into());
    }
    if !standard_input && batch_resume::fingerprint_file(input_path)? != input_fingerprint {
        return Err(format!(
            "stream input path changed during read-only planning: {}",
            input_path.display()
        ));
    }
    if frames == 0 {
        return Err("stream execution plan input contains no presentation frames".into());
    }
    let input_locator = if standard_input {
        "-".to_string()
    } else {
        denoize::portable_file_locator(input_path)?
    };
    let output_locator = if standard_output {
        "-".to_string()
    } else {
        denoize::portable_file_locator(output_path)?
    };
    let (publication, action, reason, existing_fingerprint, planned_request) = if options.resume {
        let decision = batch_resume::inspect_stream_checkpoint_decision(
            output_path,
            input_fingerprint,
            recipe,
            spec,
            block_frames,
            temporary_reservation.checkpoint_limit,
            options.force,
        )?;
        if decision
            .checkpoint()
            .is_some_and(|checkpoint| checkpoint.input_frames() > frames)
        {
            return Err("stream checkpoint exceeds the current input presentation length".into());
        }
        match decision {
            batch_resume::StreamCheckpointDecision::Skip { checkpoint, output } => {
                if checkpoint.input_frames() != frames || checkpoint.output_frames() != frames {
                    return Err(
                        "completed stream checkpoint length differs from the current input".into(),
                    );
                }
                (
                    "none",
                    "skip",
                    "completed",
                    Some(output),
                    ResourceRequest::new(),
                )
            }
            batch_resume::StreamCheckpointDecision::Process { checkpoint, reset } => {
                let (publication, publication_reason) =
                    planned_publication(output_path, options.force)?;
                let reason = if reset {
                    "forced"
                } else if checkpoint.is_some() {
                    "checkpoint"
                } else {
                    publication_reason
                };
                (publication, "process", reason, None, reported_request)
            }
        }
    } else {
        let (publication, reason) =
            initial_publication.ok_or("stream publication decision is missing after preflight")?;
        (publication, "process", reason, None, reported_request)
    };
    let item_id = denoize::execution_item_id(input_fingerprint, &output_locator, recipe)?;
    let planned_model = model
        .as_ref()
        .map(|model| {
            Ok::<PlannedArtifact, String>(PlannedArtifact {
                path: denoize::portable_file_locator(&model.path)?,
                fingerprint: model.fingerprint,
            })
        })
        .transpose()?;
    ExecutionPlan::new_stream(
        resolved.backend_options.deterministic,
        metadata_policy_name(metadata_policy),
        vec![ExecutionPlanItem {
            item_id,
            input: PlannedArtifact {
                path: input_locator,
                fingerprint: input_fingerprint,
            },
            output: PlannedOutput {
                path: output_locator,
                format: output_format_name(output_format).into(),
                publication: publication.into(),
                action: action.into(),
                reason: reason.into(),
                existing_fingerprint,
            },
            model: planned_model,
            recipe,
            backend: service::backend_name(backend).into(),
            accelerator: accelerator.effective().name().into(),
            input_format: audio_format_name(final_info.format).into(),
            input_codec: audio_codec_name(final_info.codec).into(),
            channels: u64::from(spec.channels),
            frames,
            sample_rate: spec.sample_rate,
            resources: planned_resources(planned_request),
        }],
    )
}

fn build_single_execution_plan(
    input: &std::path::Path,
    output: &std::path::Path,
    options: &Overrides,
) -> Result<ExecutionPlan, String> {
    if options.resume {
        return Err("--resume requires --batch in a read-only execution plan".into());
    }
    validate_effective_options(options, VALIDATION_SAMPLE_RATE)?;
    let encode_options = build_encode_options(options)?;
    let output_format = OutputFormat::from_path(output)?;
    validate_encode_preflight(encode_options, Some(output_format))?;
    let (publication, reason) = planned_publication(output, options.force)?;
    let governor = resource_governor(options, 1)?;
    let decode_limits = decode_limits_for_options(options)?;
    let effective_memory_mb = effective_input_memory_mb(options);
    let mut input_session = AudioInputSession::open(input)?;
    let probe = probe_audio_session_with_limits(&mut input_session, decode_limits)?;
    if probe.audio_tracks != 1 || probe.codec == AudioCodec::Unknown {
        return Err(format!(
            "plan input must contain exactly one supported audio track: {}",
            input.display()
        ));
    }
    let input_fingerprint = batch_resume::fingerprint_input_session(&mut input_session)?;
    ensure_memory_limit(
        estimate_session_memory_bytes(&input_session),
        effective_memory_mb,
        "plan input preflight",
    )?;
    let audio = read_audio_from_session_with_limits(&mut input_session, decode_limits)?;
    let decoded_working_set = estimate_audio_working_set_bytes(&audio);
    ensure_memory_limit(
        decoded_working_set,
        effective_memory_mb,
        "plan decoded audio working set",
    )?;
    output_format.validate_config(&audio, &encode_options)?;
    let metadata_bytes = if options.no_metadata {
        0
    } else {
        let limits = retained_metadata_limits(effective_memory_mb, decoded_working_set)?;
        input_session
            .read_metadata_with_limits(limits)?
            .as_ref()
            .map(denoize::metadata::Metadata::estimated_memory_bytes)
            .unwrap_or(0)
    };
    let resolved = service::resolve_processing_options_read_only(
        &audio,
        build_processing_options(options, audio.sample_rate, build_backend_options(options)?),
    )?;
    let model = batch_resume::consumed_model(&resolved)?;
    let session_request = backend_session_request(&resolved)?;
    let worker_request = worker_resource_request(
        input_session.len(),
        &audio,
        metadata_bytes,
        decode_limits.max_working_set_bytes,
        &resolved,
        true,
    )?;
    let request = worker_request.checked_add(session_request)?;
    drop(governor.try_acquire(request)?.ok_or_else(|| {
        "execution plan cannot be admitted under the configured process resource limits".to_string()
    })?);
    let _backend_session = BackendSession::prepare_with_accelerator(
        resolved.backend,
        resolved.backend_options.clone(),
        resolved.accelerator,
    )?;
    if let Some(model) = &model {
        let current = batch_resume::fingerprint_file(&model.path)?;
        if current != model.fingerprint {
            return Err(format!(
                "selected backend model changed during read-only planning: {}",
                model.path.display()
            ));
        }
    }
    let current_input = batch_resume::fingerprint_input_session(&mut input_session)?;
    if current_input != input_fingerprint {
        return Err(format!(
            "input changed during read-only planning: {}",
            input.display()
        ));
    }
    let current_input_path = batch_resume::fingerprint_file(input)?;
    if current_input_path != input_fingerprint {
        return Err(format!(
            "input path changed during read-only planning: {}",
            input.display()
        ));
    }
    let metadata_policy = if options.no_metadata {
        MetadataPolicy::Drop
    } else {
        MetadataPolicy::Preserve
    };
    let recipe = batch_resume::recipe_digest(
        &resolved,
        audio.channels(),
        output_format,
        encode_options,
        metadata_policy,
        model
            .as_ref()
            .map(|model| (&model.fingerprint, model.sample_rate)),
    )?;
    let output_locator = denoize::portable_file_locator(output)?;
    let item_id = denoize::execution_item_id(input_fingerprint, &output_locator, recipe)?;
    let frames = u64::try_from(audio.frames())
        .map_err(|_| "plan frame count is too large to represent".to_string())?;
    let item = ExecutionPlanItem {
        item_id,
        input: PlannedArtifact {
            path: denoize::portable_file_locator(input)?,
            fingerprint: input_fingerprint,
        },
        output: PlannedOutput {
            path: output_locator,
            format: output_format_name(output_format).into(),
            publication: publication.into(),
            action: "process".into(),
            reason: reason.into(),
            existing_fingerprint: None,
        },
        model: match model.as_ref() {
            Some(model) => Some(PlannedArtifact {
                path: denoize::portable_file_locator(&model.path)?,
                fingerprint: model.fingerprint,
            }),
            None => None,
        },
        recipe,
        backend: service::backend_name(resolved.backend).into(),
        accelerator: resolved.accelerator.effective().name().into(),
        input_format: audio_format_name(probe.format).into(),
        input_codec: audio_codec_name(probe.codec).into(),
        channels: audio.channels() as u64,
        frames,
        sample_rate: audio.sample_rate,
        resources: planned_resources(request),
    };
    ExecutionPlan::new(
        ExecutionKind::File,
        resolved.backend_options.deterministic,
        metadata_policy_name(metadata_policy),
        vec![item],
    )
}

fn planned_publication(
    path: &std::path::Path,
    force: bool,
) -> Result<(&'static str, &'static str), String> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok((if force { "replace" } else { "no-clobber" }, "missing"))
        }
        Ok(metadata) if force && (metadata.is_file() || metadata.file_type().is_symlink()) => {
            Ok(("replace", "untracked"))
        }
        Ok(_) if force => Err(format!(
            "output exists but is not a replaceable file or symlink: {}",
            path.display()
        )),
        Ok(_) => Err(format!(
            "output already exists: {} (use --force to replace it)",
            path.display()
        )),
        Err(error) => Err(format!(
            "inspect output destination {}: {error}",
            path.display()
        )),
    }
}

fn planned_resources(request: ResourceRequest) -> PlannedResources {
    PlannedResources {
        memory_bytes: request.memory_bytes(),
        temporary_bytes: request.temporary_bytes(),
        cpu_jobs: request.cpu_jobs() as u64,
        gpu_jobs: request.gpu_jobs() as u64,
        gpu_memory_bytes: request.gpu_memory_bytes(),
    }
}

fn metadata_policy_name(policy: MetadataPolicy) -> &'static str {
    match policy {
        MetadataPolicy::Preserve => "preserve",
        MetadataPolicy::Drop => "drop",
    }
}

fn output_format_name(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Wav => "wav",
        OutputFormat::Flac => "flac",
        OutputFormat::OggOpus => "ogg-opus",
        OutputFormat::Mp3 => "mp3",
        OutputFormat::M4a => "m4a",
        OutputFormat::AacAdts => "aac-adts",
    }
}

fn audio_format_name(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Wav => "wav",
        AudioFormat::Rf64 => "rf64",
        AudioFormat::Aiff => "aiff",
        AudioFormat::Caf => "caf",
        AudioFormat::Flac => "flac",
        AudioFormat::OggOpus => "ogg-opus",
        AudioFormat::OggVorbis => "ogg-vorbis",
        AudioFormat::Mp3 => "mp3",
        AudioFormat::M4a => "m4a",
        AudioFormat::AacAdts => "aac-adts",
        AudioFormat::Unknown => "unknown",
    }
}

fn audio_codec_name(codec: AudioCodec) -> &'static str {
    match codec {
        AudioCodec::Pcm => "pcm",
        AudioCodec::Flac => "flac",
        AudioCodec::Opus => "opus",
        AudioCodec::Vorbis => "vorbis",
        AudioCodec::Mp3 => "mp3",
        AudioCodec::Aac => "aac",
        AudioCodec::Alac => "alac",
        AudioCodec::Unknown => "unknown",
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RecommendationOutput {
    Human,
    Json,
    PrettyJson,
}

fn recommendation_usage() -> &'static str {
    "\
USAGE:
    denoize recommend <INPUT> [OPTIONS]

Analyze a bounded input prefix and rank only locally runnable backends. This
command never updates the model catalog/cache or downloads a model.

OPTIONS:
        --goal <NAME>          balanced|quality|speed|low-memory (default: balanced)
        --analysis-seconds <N> analyze 1..60 seconds (default: 12)
        --calibrate            run the fixed on-device Classical Hi-Fi benchmark
        --calibration-runs <N> measured calibration runs in 1..9 (default: 3)
        --accelerator <NAME>   cpu|auto|gpu|metal|cuda (default: auto)
        --max-memory <MB>      decode/model reservation ceiling (minimum: 1)
        --max-gpu-memory <MB>  GPU session reservation ceiling (minimum: 1)
        --deterministic        keep the recommended execution path reproducible
        --json                 emit compact denoize-recommendation-v1 JSON
        --pretty               emit indented denoize-recommendation-v1 JSON
    -h, --help                 show this help
"
}

fn parse_recommendation_args(
    args: &[String],
) -> Result<(String, RecommendationOptions, RecommendationOutput), String> {
    let mut input = None;
    let mut goal = RecommendationGoal::Balanced;
    let mut analysis_seconds = 12_u32;
    let mut calibration_runs = None;
    let mut accelerator = AcceleratorPreference::Auto;
    let mut max_memory_mb = None;
    let mut max_gpu_memory_mb = None;
    let mut deterministic = false;
    let mut output = RecommendationOutput::Human;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--goal" => {
                let value: String = parse_value(args, &mut index, "--goal")?;
                goal = RecommendationGoal::parse(&value).ok_or_else(|| {
                    format!(
                        "unknown recommendation goal: {value} (expected balanced, quality, speed, or low-memory)"
                    )
                })?;
            }
            "--analysis-seconds" => {
                analysis_seconds = parse_value(args, &mut index, "--analysis-seconds")?;
            }
            "--calibrate" => {
                if calibration_runs.is_none() {
                    calibration_runs = Some(3);
                }
            }
            "--calibration-runs" => {
                calibration_runs = Some(parse_value(args, &mut index, "--calibration-runs")?);
            }
            "--accelerator" => {
                let value: String = parse_value(args, &mut index, "--accelerator")?;
                accelerator = AcceleratorPreference::parse(&value)
                    .ok_or_else(|| format!("unknown accelerator: {value}"))?;
            }
            "--max-memory" => {
                max_memory_mb = Some(parse_value(args, &mut index, "--max-memory")?);
            }
            "--max-gpu-memory" => {
                max_gpu_memory_mb = Some(parse_value(args, &mut index, "--max-gpu-memory")?);
            }
            "--deterministic" => deterministic = true,
            "--json" => {
                if output != RecommendationOutput::Human {
                    return Err("recommend accepts only one of --json or --pretty".into());
                }
                output = RecommendationOutput::Json;
            }
            "--pretty" => {
                if output != RecommendationOutput::Human {
                    return Err("recommend accepts only one of --json or --pretty".into());
                }
                output = RecommendationOutput::PrettyJson;
            }
            "-h" | "--help" => return Err("recommendation help requested".into()),
            "-" => {
                if input.replace("-".into()).is_some() {
                    return Err("unexpected extra recommend argument: -".into());
                }
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown recommend option: {value}"));
            }
            value => {
                if input.replace(value.to_string()).is_some() {
                    return Err(format!("unexpected extra recommend argument: {value}"));
                }
            }
        }
        index += 1;
    }
    let input = input.ok_or("recommend requires INPUT")?;
    if input == "-" {
        return Err("recommend requires a regular-file INPUT; stdin is supported only by --stream processing".into());
    }
    let maximum = checked_mib_limit_bytes(max_memory_mb, "--max-memory")?;
    let maximum_gpu = checked_mib_limit_bytes(max_gpu_memory_mb, "--max-gpu-memory")?;
    let limits = DecodeLimits::new(metadata_limits_for_available_bytes(maximum), maximum);
    let options = RecommendationOptions::new()
        .with_goal(goal)
        .with_analysis_seconds(analysis_seconds)
        .with_calibration_runs(calibration_runs)
        .with_decode_limits(limits)
        .with_max_gpu_memory_bytes(maximum_gpu)
        .with_accelerator(accelerator)
        .with_deterministic(deterministic);
    // Validate option-only errors before opening the positional input.
    options.validate()?;
    Ok((input, options, output))
}

fn run_recommend(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        if args.len() != 1 {
            return Err("recommend --help accepts no other arguments".into());
        }
        print!("{}", recommendation_usage());
        return Ok(());
    }
    let (input, options, output) = parse_recommendation_args(args)?;
    let report = denoize::recommend_file_with_options(&input, options)?;
    match output {
        RecommendationOutput::Json => println!("{}", report.to_json()?),
        RecommendationOutput::PrettyJson => println!("{}", report.to_pretty_json()?),
        RecommendationOutput::Human => {
            println!(
                "recommendation: backend={} preset={} mode={} strength={:.2} adaptive={} vad={} accelerator={}",
                report.decision.backend,
                report.decision.preset,
                report.decision.processing_mode,
                report.decision.strength,
                report.decision.adaptive_noise,
                report.decision.vad,
                report.decision.accelerator
            );
            println!(
                "input: {} {} Hz, {} channel(s), material={} confidence={:.3}, analyzed={} frames ({})",
                report.input.format,
                report.input.sample_rate,
                report.input.channels,
                report.input.material.name(),
                report.input.material_confidence,
                report.input.analyzed_frames,
                report.input.analysis_mode
            );
            println!(
                "signal: rms={:.2} dBFS peak={:.2} dBFS crest={:.2} dB active={:.3}",
                report.input.rms_dbfs,
                report.input.peak_dbfs,
                report.input.crest_db,
                report.input.active_ratio
            );
            println!(
                "device: {} {} ({} logical CPUs; runtimes={})",
                report.device.os,
                report.device.architecture,
                report.device.logical_cpus,
                report.device.available_runtimes.join(",")
            );
            if let Some(calibration) = &report.calibration {
                println!(
                    "calibration: {} runs, median {:.3} ms, baseline headroom {:.3}x, fixture {}",
                    calibration.measured_runs,
                    calibration.median_elapsed_ms,
                    calibration.baseline_realtime_headroom,
                    calibration.fixture_sha256
                );
            } else {
                println!("calibration: not requested (use --calibrate)");
            }
            println!("arguments: {}", report.decision.arguments.join(" "));
            println!("candidates:");
            for candidate in &report.candidates {
                println!(
                    "  {} score={} eligible={} runtime={} ram={} gpu={}{}",
                    candidate.backend,
                    candidate.score,
                    candidate.eligible,
                    candidate.effective_accelerator.as_deref().unwrap_or("none"),
                    candidate
                        .estimated_memory_bytes
                        .map(format_device_memory)
                        .unwrap_or_else(|| "n/a".into()),
                    candidate
                        .estimated_gpu_memory_bytes
                        .map(format_device_memory)
                        .unwrap_or_else(|| "n/a".into()),
                    candidate
                        .model
                        .as_ref()
                        .map(|model| format!(" model={model}"))
                        .unwrap_or_default()
                );
                for reason in &candidate.reasons {
                    println!(
                        "    {} ({:+}): {}",
                        reason.code, reason.impact, reason.detail
                    );
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DiagnosticPrintMode {
    Human,
    Json,
    PrettyJson,
}

fn diagnose_usage() -> &'static str {
    "\
USAGE:
    denoize diagnose <INPUT> [OPTIONS]

Analyze a bounded input prefix for noise, clipping, hum, clicks, reverberation,
bandwidth limitation, dropouts, wind/plosives, and codec risk. The native
estimator is network-free and reports confidence and uncertainty; it is not a
human-MOS or semantic-fidelity release gate.

OPTIONS:
        --analysis-seconds <N> analyze 1..60 seconds (default: 12)
        --max-memory <MB>      bound denoize-owned decode and analysis memory
        --json                 emit compact denoize-diagnostic-v1 JSON
        --pretty               emit indented denoize-diagnostic-v1 JSON
    -h, --help                 show this help
"
}

fn assess_usage() -> &'static str {
    "\
USAGE:
    denoize assess <INPUT> [OPTIONS]
    denoize assess <BEFORE> <AFTER> [OPTIONS]

Produce a single-input no-reference quality report or compare the same bounded
metrics before and after processing. Before/after mode also verifies sample
rate, channel count, and presentation duration. It never treats a proxy score
as proof of semantic or speaker-identity fidelity.

OPTIONS:
        --analysis-seconds <N> analyze 1..60 seconds from each input (default: 12)
        --max-memory <MB>      bound denoize-owned decode and analysis memory
        --json                 emit compact denoize-assessment-v1 JSON
        --pretty               emit indented denoize-assessment-v1 JSON
    -h, --help                 show this help
"
}

fn parse_diagnostic_args(
    command: &str,
    args: &[String],
    maximum_inputs: usize,
) -> Result<(Vec<String>, denoize::DiagnosticOptions, DiagnosticPrintMode), String> {
    let mut inputs = Vec::new();
    let mut analysis_seconds = 12_u32;
    let mut max_memory_mb = None;
    let mut output = DiagnosticPrintMode::Human;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--analysis-seconds" => {
                analysis_seconds = parse_value(args, &mut index, "--analysis-seconds")?;
            }
            "--max-memory" => {
                max_memory_mb = Some(parse_value(args, &mut index, "--max-memory")?);
            }
            "--json" => {
                if output != DiagnosticPrintMode::Human {
                    return Err(format!("{command} accepts only one of --json or --pretty"));
                }
                output = DiagnosticPrintMode::Json;
            }
            "--pretty" => {
                if output != DiagnosticPrintMode::Human {
                    return Err(format!("{command} accepts only one of --json or --pretty"));
                }
                output = DiagnosticPrintMode::PrettyJson;
            }
            "-h" | "--help" => return Err(format!("{command} help requested")),
            "-" => {
                return Err(format!(
                    "{command} requires regular-file input; stdin is supported only by --stream processing"
                ));
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown {command} option: {value}"));
            }
            value => {
                if inputs.len() == maximum_inputs {
                    return Err(format!("unexpected extra {command} argument: {value}"));
                }
                inputs.push(value.to_string());
            }
        }
        index += 1;
    }
    if inputs.is_empty() {
        return Err(format!("{command} requires INPUT"));
    }
    let maximum = checked_mib_limit_bytes(max_memory_mb, "--max-memory")?;
    let limits = DecodeLimits::new(metadata_limits_for_available_bytes(maximum), maximum);
    let options = denoize::DiagnosticOptions::new()
        .with_analysis_seconds(analysis_seconds)
        .with_decode_limits(limits);
    options.validate()?;
    Ok((inputs, options, output))
}

fn run_diagnose(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        if args.len() != 1 {
            return Err("diagnose --help accepts no other arguments".into());
        }
        print!("{}", diagnose_usage());
        return Ok(());
    }
    let (inputs, options, mode) = parse_diagnostic_args("diagnose", args, 1)?;
    let report = denoize::diagnose_file_with_options(&inputs[0], options)?;
    match mode {
        DiagnosticPrintMode::Json => println!("{}", report.to_json()?),
        DiagnosticPrintMode::PrettyJson => println!("{}", report.to_pretty_json()?),
        DiagnosticPrintMode::Human => print_diagnostic_report(&report),
    }
    Ok(())
}

fn run_assess(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        if args.len() != 1 {
            return Err("assess --help accepts no other arguments".into());
        }
        print!("{}", assess_usage());
        return Ok(());
    }
    let (inputs, options, mode) = parse_diagnostic_args("assess", args, 2)?;
    let report = if inputs.len() == 1 {
        denoize::assess_file_with_options(&inputs[0], options)?
    } else {
        denoize::compare_files_with_options(&inputs[0], &inputs[1], options)?
    };
    match mode {
        DiagnosticPrintMode::Json => println!("{}", report.to_json()?),
        DiagnosticPrintMode::PrettyJson => println!("{}", report.to_pretty_json()?),
        DiagnosticPrintMode::Human => print_assessment_report(&report),
    }
    Ok(())
}

fn restore_usage() -> &'static str {
    "\
USAGE:
    denoize restore <INPUT> <OUTPUT> [OPTIONS]
    denoize restore <INPUT> --detect-only [OPTIONS]

Run deterministic de-clipping, de-clicking, harmonic de-hum, finite WPE
de-reverberation, and conservative wind/plosive repair. Audio geometry is
preserved. Every run can export a closed report and a complete same-length RLE
mask; uncertain damage is reported or bypassed instead of being invented.

OPTIONS:
        --operations <LIST>             comma-separated declip,declick,dehum,dereverb,wind-plosive
        --detect-only                   detect and export evidence without modifying PCM
        --report <PATH.json>            atomically write denoize-restoration-report-v1
        --mask <PATH.json>              atomically write denoize-restoration-mask-v1
        --max-memory <MB>               bound decode and restoration working memory
        --no-metadata                   do not copy input metadata to an audio output
        --replace                       atomically replace output/report/mask destinations
        --dehum-attenuation-db <DB>     maximum harmonic subtraction, 0..80 (default: 30)
        --declick-threshold-mad <N>     robust residual threshold, 4..40 (default: 10)
        --declip-iterations <N>         sparse projection iterations, 1..128 (default: 24)
        --wpe-channel-mode <MODE>       independent|multichannel (default: independent)
        --wpe-delay <FRAMES>            late-prediction delay, 1..20 (default: 3)
        --wpe-taps <N>                  prediction taps, 1..24 (default: 8)
        --wpe-iterations <N>            WPE iterations, 1..10 (default: 3)
        --wpe-regularization <F>        finite solver regularization, 1e-12..1
        --wpe-max-attenuation-db <DB>   WPE attenuation ceiling, 0..40 (default: 12)
        --wind-max-attenuation-db <DB>  burst attenuation ceiling, 0..40 (default: 18)
        --json                          emit compact report JSON to stdout
        --pretty                        emit indented report JSON to stdout
    -h, --help                          show this help
"
}

struct RestorationCliOptions {
    input: String,
    output: Option<String>,
    report: Option<String>,
    mask: Option<String>,
    config: denoize::RestorationConfig,
    max_memory_mb: Option<usize>,
    preserve_metadata: bool,
    commit_mode: CommitMode,
    print_mode: DiagnosticPrintMode,
}

fn parse_restoration_operations(value: &str) -> Result<Vec<denoize::RestorationOperation>, String> {
    if value.is_empty() {
        return Err("--operations requires at least one operation".into());
    }
    value
        .split(',')
        .map(|operation| match operation {
            "declip" => Ok(denoize::RestorationOperation::Declip),
            "declick" => Ok(denoize::RestorationOperation::Declick),
            "dehum" => Ok(denoize::RestorationOperation::Dehum),
            "dereverb" => Ok(denoize::RestorationOperation::Dereverb),
            "wind-plosive" => Ok(denoize::RestorationOperation::WindPlosive),
            _ => Err(format!(
                "unknown restoration operation: {operation} (expected declip, declick, dehum, dereverb, or wind-plosive)"
            )),
        })
        .collect()
}

fn parse_restore_args(args: &[String]) -> Result<RestorationCliOptions, String> {
    let mut positional = Vec::new();
    let mut report = None;
    let mut mask = None;
    let mut config = denoize::RestorationConfig::default();
    let mut max_memory_mb = None;
    let mut preserve_metadata = true;
    let mut commit_mode = CommitMode::NoClobber;
    let mut print_mode = DiagnosticPrintMode::Human;
    let mut detect_only = false;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--operations" => {
                let value: String = parse_value(args, &mut index, "--operations")?;
                config.operations = parse_restoration_operations(&value)?;
            }
            "--detect-only" if !detect_only => {
                detect_only = true;
                config.mode = denoize::RestorationMode::DetectOnly;
            }
            "--detect-only" => return Err("--detect-only may be supplied only once".into()),
            "--report" if report.is_none() => {
                report = Some(parse_value(args, &mut index, "--report")?);
            }
            "--report" => return Err("--report may be supplied only once".into()),
            "--mask" if mask.is_none() => {
                mask = Some(parse_value(args, &mut index, "--mask")?);
            }
            "--mask" => return Err("--mask may be supplied only once".into()),
            "--max-memory" if max_memory_mb.is_none() => {
                max_memory_mb = Some(parse_value(args, &mut index, "--max-memory")?);
            }
            "--max-memory" => return Err("--max-memory may be supplied only once".into()),
            "--no-metadata" if preserve_metadata => preserve_metadata = false,
            "--no-metadata" => return Err("--no-metadata may be supplied only once".into()),
            "--replace" if commit_mode == CommitMode::NoClobber => {
                commit_mode = CommitMode::Replace;
            }
            "--replace" => return Err("--replace may be supplied only once".into()),
            "--dehum-attenuation-db" => {
                config.dehum.attenuation_db =
                    parse_value(args, &mut index, "--dehum-attenuation-db")?;
            }
            "--declick-threshold-mad" => {
                config.declick.residual_threshold_mad =
                    parse_value(args, &mut index, "--declick-threshold-mad")?;
            }
            "--declip-iterations" => {
                config.declip.iterations = parse_value(args, &mut index, "--declip-iterations")?;
            }
            "--wpe-channel-mode" => {
                let value: String = parse_value(args, &mut index, "--wpe-channel-mode")?;
                config.dereverb.channel_mode = match value.as_str() {
                    "independent" => denoize::WpeChannelMode::Independent,
                    "multichannel" => denoize::WpeChannelMode::Multichannel,
                    _ => {
                        return Err(format!(
                            "unknown --wpe-channel-mode value: {value} (expected independent or multichannel)"
                        ))
                    }
                };
            }
            "--wpe-delay" => {
                config.dereverb.prediction_delay_frames =
                    parse_value(args, &mut index, "--wpe-delay")?;
            }
            "--wpe-taps" => {
                config.dereverb.prediction_taps = parse_value(args, &mut index, "--wpe-taps")?;
            }
            "--wpe-iterations" => {
                config.dereverb.iterations = parse_value(args, &mut index, "--wpe-iterations")?;
            }
            "--wpe-regularization" => {
                config.dereverb.regularization =
                    parse_value(args, &mut index, "--wpe-regularization")?;
            }
            "--wpe-max-attenuation-db" => {
                config.dereverb.maximum_attenuation_db =
                    parse_value(args, &mut index, "--wpe-max-attenuation-db")?;
            }
            "--wind-max-attenuation-db" => {
                config.wind_plosive.maximum_attenuation_db =
                    parse_value(args, &mut index, "--wind-max-attenuation-db")?;
            }
            "--json" => {
                if print_mode != DiagnosticPrintMode::Human {
                    return Err("restore accepts only one of --json or --pretty".into());
                }
                print_mode = DiagnosticPrintMode::Json;
            }
            "--pretty" => {
                if print_mode != DiagnosticPrintMode::Human {
                    return Err("restore accepts only one of --json or --pretty".into());
                }
                print_mode = DiagnosticPrintMode::PrettyJson;
            }
            "-h" | "--help" => return Err("restore help requested".into()),
            "-" => {
                return Err(
                    "restore requires regular-file paths; stdin/stdout are unsupported".into(),
                )
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown restore option: {value}"));
            }
            value => {
                if positional.len() == 2 {
                    return Err(format!("unexpected extra restore argument: {value}"));
                }
                positional.push(value.to_string());
            }
        }
        index += 1;
    }
    let input = positional
        .first()
        .cloned()
        .ok_or("restore requires INPUT")?;
    let output = positional.get(1).cloned();
    if config.mode == denoize::RestorationMode::Apply && output.is_none() {
        return Err(
            "restore apply mode requires OUTPUT; use --detect-only for report-only analysis".into(),
        );
    }
    if output.is_none()
        && report.is_none()
        && mask.is_none()
        && print_mode == DiagnosticPrintMode::Human
    {
        return Err("detect-only restore requires --report, --mask, --json, or --pretty".into());
    }
    checked_mib_limit_bytes(max_memory_mb, "--max-memory")?;
    config.validate()?;
    Ok(RestorationCliOptions {
        input,
        output,
        report,
        mask,
        config,
        max_memory_mb,
        preserve_metadata,
        commit_mode,
        print_mode,
    })
}

fn validate_restoration_publication_paths(options: &RestorationCliOptions) -> Result<(), String> {
    let input = std::fs::canonicalize(&options.input)
        .map_err(|error| format!("resolve restoration input {}: {error}", options.input))?;
    let mut destinations = Vec::new();
    for (path, context) in [
        (options.output.as_deref(), "restoration audio output"),
        (options.report.as_deref(), "restoration report"),
        (options.mask.as_deref(), "restoration mask"),
    ] {
        let Some(path) = path else {
            continue;
        };
        let normalized = normalized_project_destination(std::path::Path::new(path), context)?;
        let existing = std::fs::canonicalize(&normalized).ok();
        if normalized == input || existing.as_ref() == Some(&input) {
            return Err(format!("{context} must not replace the restoration input"));
        }
        ensure_restoration_destination_available(&normalized, options.commit_mode)?;
        destinations.push((batch_collision_key(&normalized), context));
    }
    destinations.sort_by(|left, right| left.0.cmp(&right.0));
    if let Some(pair) = destinations.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        return Err(format!(
            "{} and {} must use distinct destinations",
            pair[0].1, pair[1].1
        ));
    }
    Ok(())
}

fn ensure_restoration_destination_available(
    path: &std::path::Path,
    mode: CommitMode,
) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata)
            if mode == CommitMode::Replace
                && (metadata.is_file() || metadata.file_type().is_symlink()) =>
        {
            Ok(())
        }
        Ok(_) if mode == CommitMode::Replace => Err(format!(
            "restoration destination exists but is not a replaceable file or symlink: {}",
            path.display()
        )),
        Ok(_) => Err(format!(
            "restoration destination already exists: {} (use --replace to replace it)",
            path.display()
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "inspect restoration destination {}: {error}",
            path.display()
        )),
    }
}

fn stage_restoration_json<T: Serialize>(path: &str, document: &T) -> Result<AtomicOutput, String> {
    let mut bytes = serde_json::to_vec_pretty(document)
        .map_err(|error| format!("serialize restoration document: {error}"))?;
    bytes.push(b'\n');
    let mut output = AtomicOutput::new(path)?;
    output
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| format!("write staged restoration document {path}: {error}"))?;
    output
        .file_mut()
        .sync_data()
        .map_err(|error| format!("sync staged restoration document {path}: {error}"))?;
    Ok(output)
}

fn run_restore(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        if args.len() != 1 {
            return Err("restore --help accepts no other arguments".into());
        }
        print!("{}", restore_usage());
        return Ok(());
    }
    let options = parse_restore_args(args)?;
    validate_restoration_publication_paths(&options)?;
    let maximum = checked_mib_limit_bytes(options.max_memory_mb, "--max-memory")?;
    let decode_limits = DecodeLimits::new(metadata_limits_for_available_bytes(maximum), maximum);
    let mut input_session = AudioInputSession::open(&options.input)?;
    ensure_memory_limit(
        estimate_session_memory_bytes(&input_session),
        options.max_memory_mb,
        "restoration input preflight",
    )?;
    let audio = read_audio_from_session_with_limits(&mut input_session, decode_limits)?;
    let restoration_memory = denoize::estimate_restoration_memory_bytes(&audio, &options.config);
    ensure_memory_limit(
        restoration_memory,
        options.max_memory_mb,
        "restoration decoded working set",
    )?;
    let metadata = if options.output.is_some() && options.preserve_metadata {
        input_session.read_metadata_with_limits(retained_metadata_limits(
            options.max_memory_mb,
            restoration_memory,
        )?)?
    } else {
        None
    };
    let result = denoize::restore_audio(&audio, &options.config)?;
    let mut staged_report = options
        .report
        .as_deref()
        .map(|path| stage_restoration_json(path, &result.report))
        .transpose()?;
    let mut staged_mask = options
        .mask
        .as_deref()
        .map(|path| stage_restoration_json(path, &result.mask))
        .transpose()?;
    if let Some(output) = &options.output {
        let format = OutputFormat::from_path(std::path::Path::new(output))?;
        let encode_options = EncodeOptions::default();
        encode_options.validate_options(format)?;
        format.validate_config(&result.audio, &encode_options)?;
        denoize::write_audio_transactional(
            output,
            &result.audio,
            encode_options,
            metadata,
            options.commit_mode,
        )?;
    }
    if let Some(report) = staged_report.take() {
        report.commit(options.commit_mode)?;
    }
    if let Some(mask) = staged_mask.take() {
        mask.commit(options.commit_mode)?;
    }
    match options.print_mode {
        DiagnosticPrintMode::Json => println!("{}", result.report.to_json()?),
        DiagnosticPrintMode::PrettyJson => println!("{}", result.report.to_pretty_json()?),
        DiagnosticPrintMode::Human => {
            println!(
                "restoration: mode={:?} channels={} frames={} detected={} changed={} confidence={:.3} energy={:+.3} dB",
                result.report.mode,
                result.report.channels,
                result.report.frames,
                result.report.detected_samples,
                result.report.changed_samples,
                result.report.confidence,
                result.report.energy_delta_db
            );
            for operation in &result.report.operations {
                println!(
                    "  {}: {:?}, detected={}, changed={}, confidence={:.3}, energy={:+.3} dB",
                    operation.operation.name(),
                    operation.status,
                    operation.detected_samples,
                    operation.changed_samples,
                    operation.confidence,
                    operation.energy_delta_db
                );
            }
            for warning in &result.report.warnings {
                println!("warning: {warning}");
            }
        }
    }
    Ok(())
}

fn universal_usage() -> &'static str {
    "\
USAGE:
    denoize universal <INPUT> <OUTPUT> --model-package <PACKAGE.dmp> --model-package-key <KEY> [OPTIONS]
    denoize universal evidence verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]

Run fail-closed universal speech restoration through an authenticated BSRNN
spectral package v2. The safe default is discriminative and primary. Clean
input bypasses inference. A candidate is published only after geometry,
finite-sample, energy, peak, clipping, silence-injection, and native-quality
gates pass; otherwise OUTPUT contains the bit-exact decoded input.

OPTIONS:
        --model-package <PATH>            required signed runtime package v2
        --model-package-key <PATH>        trusted Minisign public key
        --family <FAMILY>                 discriminative|hybrid|generative
        --render-role <ROLE>              primary|alternate
        --experimental                    required for hybrid/generative alternate renders
        --analysis-seconds <N>            bounded diagnosis prefix, 1..60 (default: 12)
        --minimum-degradation-score <F>   inference threshold, 0..1 (default: 0.08)
        --maximum-energy-gain-db <DB>     fail-closed candidate ceiling, 0..24 (default: 6)
        --maximum-peak-gain-db <DB>       fail-closed peak-rise ceiling, 0..24 (default: 6)
        --maximum-new-clipping-ratio <F>  added clipping ceiling, 0..0.1 (default: 0.0001)
        --maximum-quality-regression <F>  native proxy regression ceiling, 0..25 (default: 5)
        --accelerator <NAME>              cpu|auto|gpu|metal|cuda (default: cpu)
        --report <PATH.json>              atomically write the closed report
        --mask <PATH.json>                atomically write the complete RLE change mask
        --max-memory <MB>                 bound decode, model, candidate, and mask memory
        --no-metadata                     do not copy input metadata
        --replace                         atomically replace output/report/mask destinations
        --json                            emit compact report JSON
        --pretty                          emit indented report JSON
    -h, --help                            show this help
"
}

#[cfg_attr(not(feature = "bsrnn"), allow(dead_code))]
#[derive(Debug)]
struct UniversalCliOptions {
    input: String,
    output: String,
    package: String,
    package_key: String,
    report: Option<String>,
    mask: Option<String>,
    config: denoize::UniversalRestorationConfig,
    accelerator: AcceleratorPreference,
    max_memory_mb: Option<usize>,
    preserve_metadata: bool,
    commit_mode: CommitMode,
    print_mode: DiagnosticPrintMode,
}

fn parse_universal_args(args: &[String]) -> Result<UniversalCliOptions, String> {
    let mut positional = Vec::new();
    let mut package = None;
    let mut package_key = None;
    let mut report = None;
    let mut mask = None;
    let mut config = denoize::UniversalRestorationConfig::default();
    let mut accelerator = AcceleratorPreference::Cpu;
    let mut accelerator_seen = false;
    let mut max_memory_mb = None;
    let mut preserve_metadata = true;
    let mut commit_mode = CommitMode::NoClobber;
    let mut print_mode = DiagnosticPrintMode::Human;
    let mut experimental_seen = false;
    let mut scalar_options = std::collections::HashSet::new();
    let mut index = 0usize;
    while index < args.len() {
        let argument = args[index].as_str();
        if matches!(
            argument,
            "--family"
                | "--render-role"
                | "--analysis-seconds"
                | "--minimum-degradation-score"
                | "--maximum-energy-gain-db"
                | "--maximum-peak-gain-db"
                | "--maximum-new-clipping-ratio"
                | "--maximum-quality-regression"
        ) && !scalar_options.insert(argument)
        {
            return Err(format!("{argument} may be supplied only once"));
        }
        match argument {
            "--model-package" if package.is_none() => {
                package = Some(parse_value(args, &mut index, "--model-package")?);
            }
            "--model-package" => return Err("--model-package may be supplied only once".into()),
            "--model-package-key" if package_key.is_none() => {
                package_key = Some(parse_value(args, &mut index, "--model-package-key")?);
            }
            "--model-package-key" => {
                return Err("--model-package-key may be supplied only once".into())
            }
            "--family" => {
                let value: String = parse_value(args, &mut index, "--family")?;
                config.model_family = denoize::UniversalModelFamily::parse(&value).ok_or_else(|| {
                    format!(
                        "unknown universal model family: {value} (expected discriminative, hybrid, or generative)"
                    )
                })?;
            }
            "--render-role" => {
                let value: String = parse_value(args, &mut index, "--render-role")?;
                config.render_role =
                    denoize::UniversalRenderRole::parse(&value).ok_or_else(|| {
                        format!(
                        "unknown universal render role: {value} (expected primary or alternate)"
                    )
                    })?;
            }
            "--experimental" if !experimental_seen => {
                experimental_seen = true;
                config.allow_experimental = true;
            }
            "--experimental" => return Err("--experimental may be supplied only once".into()),
            "--analysis-seconds" => {
                config.analysis_seconds = parse_value(args, &mut index, "--analysis-seconds")?;
            }
            "--minimum-degradation-score" => {
                config.minimum_degradation_score =
                    parse_value(args, &mut index, "--minimum-degradation-score")?;
            }
            "--maximum-energy-gain-db" => {
                config.maximum_energy_gain_db =
                    parse_value(args, &mut index, "--maximum-energy-gain-db")?;
            }
            "--maximum-peak-gain-db" => {
                config.maximum_peak_gain_db =
                    parse_value(args, &mut index, "--maximum-peak-gain-db")?;
            }
            "--maximum-new-clipping-ratio" => {
                config.maximum_new_clipping_ratio =
                    parse_value(args, &mut index, "--maximum-new-clipping-ratio")?;
            }
            "--maximum-quality-regression" => {
                config.maximum_quality_score_regression =
                    parse_value(args, &mut index, "--maximum-quality-regression")?;
            }
            "--accelerator" if !accelerator_seen => {
                accelerator_seen = true;
                let value: String = parse_value(args, &mut index, "--accelerator")?;
                accelerator = AcceleratorPreference::parse(&value).ok_or_else(|| {
                    format!(
                        "unknown universal accelerator: {value} (expected cpu, auto, gpu, metal, or cuda)"
                    )
                })?;
            }
            "--accelerator" => return Err("--accelerator may be supplied only once".into()),
            "--report" if report.is_none() => {
                report = Some(parse_value(args, &mut index, "--report")?);
            }
            "--report" => return Err("--report may be supplied only once".into()),
            "--mask" if mask.is_none() => {
                mask = Some(parse_value(args, &mut index, "--mask")?);
            }
            "--mask" => return Err("--mask may be supplied only once".into()),
            "--max-memory" if max_memory_mb.is_none() => {
                max_memory_mb = Some(parse_value(args, &mut index, "--max-memory")?);
            }
            "--max-memory" => return Err("--max-memory may be supplied only once".into()),
            "--no-metadata" if preserve_metadata => preserve_metadata = false,
            "--no-metadata" => return Err("--no-metadata may be supplied only once".into()),
            "--replace" if commit_mode == CommitMode::NoClobber => {
                commit_mode = CommitMode::Replace;
            }
            "--replace" => return Err("--replace may be supplied only once".into()),
            "--json" => {
                if print_mode != DiagnosticPrintMode::Human {
                    return Err("universal accepts only one of --json or --pretty".into());
                }
                print_mode = DiagnosticPrintMode::Json;
            }
            "--pretty" => {
                if print_mode != DiagnosticPrintMode::Human {
                    return Err("universal accepts only one of --json or --pretty".into());
                }
                print_mode = DiagnosticPrintMode::PrettyJson;
            }
            "-h" | "--help" => return Err("universal help requested".into()),
            "-" => return Err(
                "universal restoration requires regular-file paths; stdin/stdout are unsupported"
                    .into(),
            ),
            value if value.starts_with('-') => {
                return Err(format!("unknown universal option: {value}"));
            }
            value => {
                if positional.len() == 2 {
                    return Err(format!("unexpected extra universal argument: {value}"));
                }
                positional.push(value.to_string());
            }
        }
        index += 1;
    }
    let input = positional
        .first()
        .cloned()
        .ok_or("universal requires INPUT")?;
    let output = positional
        .get(1)
        .cloned()
        .ok_or("universal requires OUTPUT")?;
    let package = package.ok_or("universal requires --model-package")?;
    let package_key = package_key.ok_or("universal requires --model-package-key")?;
    checked_mib_limit_bytes(max_memory_mb, "--max-memory")?;
    config.validate()?;
    Ok(UniversalCliOptions {
        input,
        output,
        package,
        package_key,
        report,
        mask,
        config,
        accelerator,
        max_memory_mb,
        preserve_metadata,
        commit_mode,
        print_mode,
    })
}

#[cfg(feature = "bsrnn")]
fn validate_universal_publication_paths(options: &UniversalCliOptions) -> Result<(), String> {
    let mut sources = Vec::new();
    for (path, context) in [
        (&options.input, "universal input"),
        (&options.package, "universal model package"),
        (&options.package_key, "universal model package key"),
    ] {
        sources.push((
            std::fs::canonicalize(path)
                .map_err(|error| format!("resolve {context} {path}: {error}"))?,
            context,
        ));
    }
    for left in 0..sources.len() {
        for right in left + 1..sources.len() {
            if sources[left].0 == sources[right].0 {
                return Err(format!(
                    "{} and {} must use distinct source files",
                    sources[left].1, sources[right].1
                ));
            }
        }
    }
    let mut destinations = Vec::new();
    for (path, context) in [
        (Some(options.output.as_str()), "universal audio output"),
        (options.report.as_deref(), "universal report"),
        (options.mask.as_deref(), "universal mask"),
    ] {
        let Some(path) = path else {
            continue;
        };
        let normalized = normalized_project_destination(std::path::Path::new(path), context)?;
        let existing = std::fs::canonicalize(&normalized).ok();
        if sources.iter().any(|(source, _)| {
            normalized == *source || existing.as_ref().is_some_and(|path| path == source)
        }) {
            return Err(format!(
                "{context} must not replace an input, model package, or key"
            ));
        }
        ensure_restoration_destination_available(&normalized, options.commit_mode)?;
        destinations.push((batch_collision_key(&normalized), context));
    }
    destinations.sort_by(|left, right| left.0.cmp(&right.0));
    if let Some(pair) = destinations.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        return Err(format!(
            "{} and {} must use distinct destinations",
            pair[0].1, pair[1].1
        ));
    }
    Ok(())
}

fn run_universal(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) == Some("evidence") {
        return run_universal_evidence(&args[1..]);
    }
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        if args.len() != 1 {
            return Err("universal --help accepts no other arguments".into());
        }
        print!("{}", universal_usage());
        return Ok(());
    }
    let options = parse_universal_args(args)?;
    #[cfg(feature = "bsrnn")]
    {
        validate_universal_publication_paths(&options)?;
        run_universal_audio(options)
    }
    #[cfg(not(feature = "bsrnn"))]
    {
        run_universal_audio(options)
    }
}

fn run_universal_evidence(args: &[String]) -> Result<(), String> {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help") {
        print!("{}", universal_usage());
        return Ok(());
    }
    if args.first().map(String::as_str) != Some("verify") {
        return Err("universal evidence requires: verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]".into());
    }
    let mut positional = Vec::new();
    let mut mode = DiagnosticPrintMode::Human;
    for argument in &args[1..] {
        match argument.as_str() {
            "--json" if mode == DiagnosticPrintMode::Human => mode = DiagnosticPrintMode::Json,
            "--pretty" if mode == DiagnosticPrintMode::Human => {
                mode = DiagnosticPrintMode::PrettyJson
            }
            "--json" | "--pretty" => {
                return Err("universal evidence verify accepts only one output mode".into())
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown universal evidence option: {value}"))
            }
            value => positional.push(value.to_string()),
        }
    }
    if positional.len() != 2 {
        return Err("universal evidence verify requires EVIDENCE.json and PUBLIC-KEY.json".into());
    }
    let evidence = denoize::SignedUniversalPromotionEvidence::from_file(&positional[0])?;
    let key = ReceiptPublicKey::from_file(&positional[1])?;
    evidence.verify_signature(&key)?;
    match mode {
        DiagnosticPrintMode::Json => println!(
            "{}",
            serde_json::to_string(&evidence)
                .map_err(|error| format!("serialize universal promotion evidence: {error}"))?
        ),
        DiagnosticPrintMode::PrettyJson => println!("{}", evidence.to_pretty_json()?),
        DiagnosticPrintMode::Human => println!(
            "verified universal promotion evidence: family={:?}, package={}, strata={}, listeners={}, accepted={}",
            evidence.payload.model_family,
            evidence.payload.model_package_sha256,
            evidence.payload.strata.len(),
            evidence.payload.listener_count,
            evidence.payload.accepted
        ),
    }
    if !evidence.payload.accepted {
        return Err(
            "universal promotion evidence is authentic but does not pass promotion gates".into(),
        );
    }
    Ok(())
}

#[cfg(feature = "bsrnn")]
fn run_universal_audio(options: UniversalCliOptions) -> Result<(), String> {
    let maximum = checked_mib_limit_bytes(options.max_memory_mb, "--max-memory")?;
    let package = RuntimeModelPackage::open(&options.package, &options.package_key)?;
    if package.manifest_v2().is_none() {
        return Err("universal restoration requires runtime model package v2".into());
    }
    let mut backend_options = BackendOptions::default().with_runtime_model_package(package);
    backend_options.deterministic = true;
    backend_options.accelerator = options.accelerator;
    let accelerator = denoize::select_accelerator_for_options(Backend::Bsrnn, &backend_options)?;
    let package = backend_options
        .runtime_package
        .as_ref()
        .expect("universal backend options retain their package");
    let profile = package
        .precision_profile_for(accelerator.effective())?
        .expect("universal package v2 selects a precision profile");
    let model_working_set = profile
        .resources
        .max_session_memory_bytes
        .saturating_add(profile.resources.max_worker_memory_bytes);
    ensure_memory_limit(
        model_working_set,
        options.max_memory_mb,
        "universal model working set",
    )?;
    // Preparing the session authenticates the selected model component,
    // validates its graph contract, and executes every signed numerical vector
    // before user-controlled audio is opened or decoded.
    let session =
        BackendSession::prepare_with_accelerator(Backend::Bsrnn, backend_options, accelerator)?;
    let decode_maximum = maximum.map(|limit| limit.saturating_sub(model_working_set));
    let decode_limits = DecodeLimits::new(
        metadata_limits_for_available_bytes(decode_maximum),
        decode_maximum,
    );
    let mut input_session = AudioInputSession::open(&options.input)?;
    ensure_memory_limit(
        model_working_set.saturating_add(estimate_session_memory_bytes(&input_session)),
        options.max_memory_mb,
        "universal input/model preflight",
    )?;
    let audio = read_audio_from_session_with_limits(&mut input_session, decode_limits)?;
    let working_set = denoize::estimate_universal_restoration_memory_bytes(&audio)
        .saturating_add(model_working_set);
    ensure_memory_limit(
        working_set,
        options.max_memory_mb,
        "universal decoded/model working set",
    )?;
    let metadata = if options.preserve_metadata {
        input_session.read_metadata_with_limits(retained_metadata_limits(
            options.max_memory_mb,
            working_set,
        )?)?
    } else {
        None
    };
    let result = denoize::restore_universal_audio(&audio, &session, &options.config)?;
    let mut staged_report = options
        .report
        .as_deref()
        .map(|path| stage_restoration_json(path, &result.report))
        .transpose()?;
    let mut staged_mask = options
        .mask
        .as_deref()
        .map(|path| stage_restoration_json(path, &result.mask))
        .transpose()?;
    let format = OutputFormat::from_path(std::path::Path::new(&options.output))?;
    let encode_options = EncodeOptions::default();
    encode_options.validate_options(format)?;
    format.validate_config(&result.audio, &encode_options)?;
    denoize::write_audio_transactional(
        &options.output,
        &result.audio,
        encode_options,
        metadata,
        options.commit_mode,
    )?;
    if let Some(report) = staged_report.take() {
        report.commit(options.commit_mode)?;
    }
    if let Some(mask) = staged_mask.take() {
        mask.commit(options.commit_mode)?;
    }
    match options.print_mode {
        DiagnosticPrintMode::Json => println!("{}", result.report.to_json()?),
        DiagnosticPrintMode::PrettyJson => println!("{}", result.report.to_pretty_json()?),
        DiagnosticPrintMode::Human => {
            println!(
                "universal restoration: decision={:?} family={:?} role={:?} channels={} frames={} changed={} package={}",
                result.report.decision,
                result.report.model_family,
                result.report.render_role,
                result.report.channels,
                result.report.frames,
                result.report.changed_samples,
                result.report.model.package_sha256
            );
            for warning in &result.report.warnings {
                println!("warning: {warning}");
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "bsrnn"))]
fn run_universal_audio(_options: UniversalCliOptions) -> Result<(), String> {
    Err("universal audio restoration requires a build with the bsrnn feature".into())
}

fn meeting_speaker_usage() -> &'static str {
    "\
USAGE:
    denoize meeting-speakers <MEETING> <OUTPUT.wav> --model-package <PACKAGE.dmp> --model-package-key <KEY> --promotion-evidence <EVIDENCE.json> --promotion-evidence-key <PUBLIC-KEY.json> [OPTIONS]
    denoize meeting-speakers evidence verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]

Separate a mono meeting or an explicitly fixed microphone array into at most
eight anonymous speaker channels plus one final unassigned-residual channel.
The signed package must expose fixed-window separated audio, per-track
inactive/uncertain/active probabilities, and global no-speech/assigned/unknown
probabilities. The residual is required and exactly recombines with every
published track to the arithmetic-mean reference. Unknown speech is never
forced into an identity. The output is WAV only; the report maps its first N
channels to speaker-NNN and its final channel to unassigned.

OPTIONS:
        --model-package <PATH>                    required signed runtime package v2
        --model-package-key <PATH>                trusted Minisign public key
        --promotion-evidence <PATH>               accepted signed meeting evaluation evidence
        --promotion-evidence-key <PATH>           trusted Ed25519 evidence public key
        --track-labels <PATH.json>                optional Stage 29 consent-bound labels
        --minimum-active-probability <F>          active threshold, 0.5..1 (default: 0.8)
        --minimum-inactive-probability <F>        inactive threshold, 0.5..1 (default: 0.8)
        --minimum-unknown-probability <F>         unknown threshold, 0.5..1 (default: 0.8)
        --minimum-active-frames <N>               consecutive frames needed to publish, 1..100 (default: 2)
        --permutation-minimum-correlation <F>     window stitch threshold, 0..1 (default: 0.2)
        --permutation-minimum-margin <F>          best/runner-up stitch margin, 0..1 (default: 0.05)
        --maximum-track-peak <F>                  absolute track peak, 0.5..1 (default: 1)
        --maximum-residual-peak <F>               absolute residual peak, 0.5..1 (default: 1)
        --accelerator <NAME>                      cpu|auto|gpu|metal|cuda (default: cpu)
        --report <PATH.json>                      atomically write the closed path-free report
        --max-memory <MB>                         bound decode, model, tracks, and residual memory
        --replace                                 atomically replace output/report destinations
        --json                                    emit compact report JSON
        --pretty                                  emit indented report JSON
    -h, --help                                    show this help
"
}

#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
#[derive(Debug)]
struct MeetingSpeakerCliOptions {
    input: String,
    output: String,
    package: String,
    package_key: String,
    promotion_evidence: String,
    promotion_evidence_key: String,
    track_labels: Option<String>,
    report: Option<String>,
    config: denoize::MeetingSpeakerConfig,
    accelerator: AcceleratorPreference,
    max_memory_mb: Option<usize>,
    commit_mode: CommitMode,
    print_mode: DiagnosticPrintMode,
}

fn parse_meeting_speaker_args(args: &[String]) -> Result<MeetingSpeakerCliOptions, String> {
    let mut positional = Vec::new();
    let mut package = None;
    let mut package_key = None;
    let mut promotion_evidence = None;
    let mut promotion_evidence_key = None;
    let mut track_labels = None;
    let mut report = None;
    let mut config = denoize::MeetingSpeakerConfig::default();
    let mut accelerator = AcceleratorPreference::Cpu;
    let mut accelerator_seen = false;
    let mut max_memory_mb = None;
    let mut commit_mode = CommitMode::NoClobber;
    let mut print_mode = DiagnosticPrintMode::Human;
    let mut scalar_options = std::collections::HashSet::new();
    let mut index = 0usize;
    while index < args.len() {
        let argument = args[index].as_str();
        if matches!(
            argument,
            "--minimum-active-probability"
                | "--minimum-inactive-probability"
                | "--minimum-unknown-probability"
                | "--minimum-active-frames"
                | "--permutation-minimum-correlation"
                | "--permutation-minimum-margin"
                | "--maximum-track-peak"
                | "--maximum-residual-peak"
        ) && !scalar_options.insert(argument)
        {
            return Err(format!("{argument} may be supplied only once"));
        }
        match argument {
            "--model-package" if package.is_none() => {
                package = Some(parse_value(args, &mut index, "--model-package")?);
            }
            "--model-package" => return Err("--model-package may be supplied only once".into()),
            "--model-package-key" if package_key.is_none() => {
                package_key = Some(parse_value(args, &mut index, "--model-package-key")?);
            }
            "--model-package-key" => {
                return Err("--model-package-key may be supplied only once".into());
            }
            "--promotion-evidence" if promotion_evidence.is_none() => {
                promotion_evidence = Some(parse_value(args, &mut index, "--promotion-evidence")?);
            }
            "--promotion-evidence" => {
                return Err("--promotion-evidence may be supplied only once".into());
            }
            "--promotion-evidence-key" if promotion_evidence_key.is_none() => {
                promotion_evidence_key =
                    Some(parse_value(args, &mut index, "--promotion-evidence-key")?);
            }
            "--promotion-evidence-key" => {
                return Err("--promotion-evidence-key may be supplied only once".into());
            }
            "--track-labels" if track_labels.is_none() => {
                track_labels = Some(parse_value(args, &mut index, "--track-labels")?);
            }
            "--track-labels" => return Err("--track-labels may be supplied only once".into()),
            "--minimum-active-probability" => {
                config.minimum_active_probability =
                    parse_value(args, &mut index, "--minimum-active-probability")?;
            }
            "--minimum-inactive-probability" => {
                config.minimum_inactive_probability =
                    parse_value(args, &mut index, "--minimum-inactive-probability")?;
            }
            "--minimum-unknown-probability" => {
                config.minimum_unknown_probability =
                    parse_value(args, &mut index, "--minimum-unknown-probability")?;
            }
            "--minimum-active-frames" => {
                config.minimum_active_frames =
                    parse_value(args, &mut index, "--minimum-active-frames")?;
            }
            "--permutation-minimum-correlation" => {
                config.permutation_minimum_correlation =
                    parse_value(args, &mut index, "--permutation-minimum-correlation")?;
            }
            "--permutation-minimum-margin" => {
                config.permutation_minimum_margin =
                    parse_value(args, &mut index, "--permutation-minimum-margin")?;
            }
            "--maximum-track-peak" => {
                config.maximum_track_peak = parse_value(args, &mut index, "--maximum-track-peak")?;
            }
            "--maximum-residual-peak" => {
                config.maximum_residual_peak =
                    parse_value(args, &mut index, "--maximum-residual-peak")?;
            }
            "--accelerator" if !accelerator_seen => {
                accelerator_seen = true;
                let value: String = parse_value(args, &mut index, "--accelerator")?;
                accelerator = AcceleratorPreference::parse(&value).ok_or_else(|| {
                    format!(
                        "unknown meeting-speaker accelerator: {value} (expected cpu, auto, gpu, metal, or cuda)"
                    )
                })?;
            }
            "--accelerator" => return Err("--accelerator may be supplied only once".into()),
            "--report" if report.is_none() => {
                report = Some(parse_value(args, &mut index, "--report")?);
            }
            "--report" => return Err("--report may be supplied only once".into()),
            "--max-memory" if max_memory_mb.is_none() => {
                max_memory_mb = Some(parse_value(args, &mut index, "--max-memory")?);
            }
            "--max-memory" => return Err("--max-memory may be supplied only once".into()),
            "--replace" if commit_mode == CommitMode::NoClobber => {
                commit_mode = CommitMode::Replace;
            }
            "--replace" => return Err("--replace may be supplied only once".into()),
            "--json" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::Json;
            }
            "--pretty" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err("meeting-speakers accepts only one of --json or --pretty".into());
            }
            "-h" | "--help" => return Err("meeting-speakers help requested".into()),
            "-" => {
                return Err(
                    "meeting-speakers requires regular-file paths; stdin/stdout are unsupported"
                        .into(),
                );
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown meeting-speakers option: {value}"));
            }
            value => {
                if positional.len() == 2 {
                    return Err(format!(
                        "unexpected extra meeting-speakers argument: {value}"
                    ));
                }
                positional.push(value.to_string());
            }
        }
        index += 1;
    }
    let input = positional
        .first()
        .cloned()
        .ok_or("meeting-speakers requires MEETING")?;
    let output = positional
        .get(1)
        .cloned()
        .ok_or("meeting-speakers requires OUTPUT.wav")?;
    if OutputFormat::from_path(std::path::Path::new(&output))? != OutputFormat::Wav {
        return Err(
            "meeting-speakers output must be WAV so every track and residual is lossless".into(),
        );
    }
    let package = package.ok_or("meeting-speakers requires --model-package")?;
    let package_key = package_key.ok_or("meeting-speakers requires --model-package-key")?;
    let promotion_evidence =
        promotion_evidence.ok_or("meeting-speakers requires --promotion-evidence")?;
    let promotion_evidence_key =
        promotion_evidence_key.ok_or("meeting-speakers requires --promotion-evidence-key")?;
    checked_mib_limit_bytes(max_memory_mb, "--max-memory")?;
    config.validate()?;
    Ok(MeetingSpeakerCliOptions {
        input,
        output,
        package,
        package_key,
        promotion_evidence,
        promotion_evidence_key,
        track_labels,
        report,
        config,
        accelerator,
        max_memory_mb,
        commit_mode,
        print_mode,
    })
}

fn run_meeting_speakers(args: &[String]) -> Result<(), String> {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help") {
        print!("{}", meeting_speaker_usage());
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("evidence") {
        return run_meeting_speaker_evidence(&args[1..]);
    }
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        return Err("meeting-speakers --help accepts no other arguments".into());
    }
    let options = parse_meeting_speaker_args(args)?;
    validate_meeting_speaker_publication_paths(&options)?;
    run_meeting_speaker_audio(options)
}

fn run_meeting_speaker_evidence(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) != Some("verify") {
        return Err(
            "meeting-speaker evidence requires: verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]"
                .into(),
        );
    }
    let mut positional = Vec::new();
    let mut mode = DiagnosticPrintMode::Human;
    for argument in &args[1..] {
        match argument.as_str() {
            "--json" if mode == DiagnosticPrintMode::Human => mode = DiagnosticPrintMode::Json,
            "--pretty" if mode == DiagnosticPrintMode::Human => {
                mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err("meeting-speaker evidence verify accepts only one output mode".into());
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown meeting-speaker evidence option: {value}"));
            }
            value => positional.push(value.to_string()),
        }
    }
    if positional.len() != 2 {
        return Err(
            "meeting-speaker evidence verify requires EVIDENCE.json and PUBLIC-KEY.json".into(),
        );
    }
    let evidence = denoize::SignedMeetingSpeakerPromotionEvidence::from_file(&positional[0])?;
    let key = ReceiptPublicKey::from_file(&positional[1])?;
    evidence.verify_signature(&key)?;
    match mode {
        DiagnosticPrintMode::Json => println!(
            "{}",
            serde_json::to_string(&evidence)
                .map_err(|error| format!("serialize meeting-speaker evidence: {error}"))?
        ),
        DiagnosticPrintMode::PrettyJson => println!("{}", evidence.to_pretty_json()?),
        DiagnosticPrintMode::Human => println!(
            "verified meeting-speaker evidence: strata={} real_meetings={} speakers={} languages={} retained_embeddings={} accepted={}",
            evidence.payload.strata.len(),
            evidence.payload.real_meeting_cases,
            evidence.payload.distinct_speakers,
            evidence.payload.language_count,
            evidence.payload.retained_speaker_embeddings,
            evidence.payload.accepted,
        ),
    }
    if !evidence.payload.accepted {
        return Err(
            "meeting-speaker evidence is authentic but does not pass promotion gates".into(),
        );
    }
    Ok(())
}

fn validate_meeting_speaker_publication_paths(
    options: &MeetingSpeakerCliOptions,
) -> Result<(), String> {
    let mut sources = Vec::new();
    for (path, context) in [
        (Some(options.input.as_str()), "meeting-speaker input"),
        (Some(options.package.as_str()), "meeting-speaker package"),
        (
            Some(options.package_key.as_str()),
            "meeting-speaker package key",
        ),
        (
            Some(options.promotion_evidence.as_str()),
            "meeting-speaker evidence",
        ),
        (
            Some(options.promotion_evidence_key.as_str()),
            "meeting-speaker evidence key",
        ),
        (
            options.track_labels.as_deref(),
            "meeting-speaker track labels",
        ),
    ] {
        let Some(path) = path else {
            continue;
        };
        sources.push((
            std::fs::canonicalize(path)
                .map_err(|error| format!("resolve {context} {path}: {error}"))?,
            context,
        ));
    }
    for left in 0..sources.len() {
        for right in left + 1..sources.len() {
            if sources[left].0 == sources[right].0 {
                return Err(format!(
                    "{} and {} must use distinct source files",
                    sources[left].1, sources[right].1
                ));
            }
        }
    }
    let mut destinations = Vec::new();
    for (path, context) in [
        (Some(options.output.as_str()), "meeting-speaker WAV output"),
        (options.report.as_deref(), "meeting-speaker report"),
    ] {
        let Some(path) = path else {
            continue;
        };
        let normalized = normalized_project_destination(std::path::Path::new(path), context)?;
        let existing = std::fs::canonicalize(&normalized).ok();
        if sources.iter().any(|(source, _)| {
            normalized == *source || existing.as_ref().is_some_and(|path| path == source)
        }) {
            return Err(format!(
                "{context} must not replace an input, package, key, evidence, or label document"
            ));
        }
        ensure_restoration_destination_available(&normalized, options.commit_mode)?;
        destinations.push((batch_collision_key(&normalized), context));
    }
    destinations.sort_by(|left, right| left.0.cmp(&right.0));
    if let Some(pair) = destinations.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        return Err(format!(
            "{} and {} must use distinct destinations",
            pair[0].1, pair[1].1
        ));
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn run_meeting_speaker_audio(options: MeetingSpeakerCliOptions) -> Result<(), String> {
    let maximum = checked_mib_limit_bytes(options.max_memory_mb, "--max-memory")?;
    let evidence =
        denoize::SignedMeetingSpeakerPromotionEvidence::from_file(&options.promotion_evidence)?;
    let evidence_key = ReceiptPublicKey::from_file(&options.promotion_evidence_key)?;
    let package = RuntimeModelPackage::open(&options.package, &options.package_key)?;
    let labels = options
        .track_labels
        .as_deref()
        .map(denoize::MeetingTrackLabelsDocument::from_file)
        .transpose()?
        .map_or_else(Vec::new, |document| document.labels);
    // Authenticate the graph, vectors, evidence, privacy record, and selected
    // accelerator before opening user-controlled meeting audio.
    let session = denoize::MeetingSpeakerSession::prepare(
        package,
        &evidence,
        &evidence_key,
        &options.config,
        options.accelerator,
    )?;
    let model_working_set = session.model_working_set_bytes()?;
    ensure_memory_limit(
        model_working_set,
        options.max_memory_mb,
        "meeting-speaker model working set",
    )?;
    let mut input_session = AudioInputSession::open(&options.input)?;
    let session_memory = estimate_session_memory_bytes(&input_session);
    ensure_memory_limit(
        model_working_set.saturating_add(session_memory),
        options.max_memory_mb,
        "meeting-speaker input/model preflight",
    )?;
    let decode_maximum = maximum.map(|limit| {
        limit
            .saturating_sub(model_working_set)
            .saturating_sub(session_memory)
    });
    let input = read_audio_from_session_with_limits(
        &mut input_session,
        DecodeLimits::new(
            metadata_limits_for_available_bytes(decode_maximum),
            decode_maximum,
        ),
    )?;
    let working_set = session
        .processing_working_set_bytes(&input)?
        .saturating_add(model_working_set)
        .saturating_add(session_memory);
    ensure_memory_limit(
        working_set,
        options.max_memory_mb,
        "meeting-speaker decoded/model/tracks/residual working set",
    )?;
    let result = session.separate(&input, &options.config, &labels)?;
    let mut channels = Vec::new();
    channels
        .try_reserve_exact(result.tracks.len() + 1)
        .map_err(|_| "unable to reserve meeting-speaker WAV channels".to_string())?;
    for track in result.tracks {
        channels.push(
            track
                .channels
                .into_iter()
                .next()
                .ok_or("meeting-speaker track lost its mono channel")?,
        );
    }
    channels.push(
        result
            .unassigned
            .channels
            .into_iter()
            .next()
            .ok_or("meeting-speaker residual lost its mono channel")?,
    );
    let output = denoize::Audio {
        sample_rate: input.sample_rate,
        channels,
        bits_per_sample: input.bits_per_sample,
        sample_format: input.sample_format,
        channel_mask: None,
    };
    let mut staged_report = options
        .report
        .as_deref()
        .map(|path| stage_restoration_json(path, &result.report))
        .transpose()?;
    let encode_options = EncodeOptions::default();
    OutputFormat::Wav.validate_config(&output, &encode_options)?;
    denoize::write_audio_transactional(
        &options.output,
        &output,
        encode_options,
        None,
        options.commit_mode,
    )?;
    if let Some(report) = staged_report.take() {
        report.commit(options.commit_mode)?;
    }
    match options.print_mode {
        DiagnosticPrintMode::Json => println!("{}", result.report.to_json()?),
        DiagnosticPrintMode::PrettyJson => println!("{}", result.report.to_pretty_json()?),
        DiagnosticPrintMode::Human => println!(
            "meeting speakers: tracks={} residual_channel={} unknown_regions={} overlap_regions={} ambiguous_windows={} frames={} package={}",
            result.report.published_tracks,
            result.report.published_tracks + 1,
            result.report.unknown_regions.len(),
            result.report.overlap_regions.len(),
            result.report.permutation_ambiguous_windows,
            result.report.source_frames,
            result.report.model.package_sha256,
        ),
    }
    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn run_meeting_speaker_audio(_options: MeetingSpeakerCliOptions) -> Result<(), String> {
    Err("meeting speaker tracks require a build with the onnx feature".into())
}

#[cfg(test)]
mod meeting_speaker_cli_tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn parser_requires_authenticated_package_and_evidence() {
        let error =
            parse_meeting_speaker_args(&arguments(&["meeting.wav", "tracks.wav"])).unwrap_err();
        assert_eq!(error, "meeting-speakers requires --model-package");
        let parsed = parse_meeting_speaker_args(&arguments(&[
            "meeting.wav",
            "tracks.wav",
            "--model-package",
            "model.dmp",
            "--model-package-key",
            "model.pub",
            "--promotion-evidence",
            "evidence.json",
            "--promotion-evidence-key",
            "evidence.pub.json",
        ]))
        .unwrap();
        assert_eq!(parsed.config.minimum_active_probability, 0.8);
        assert_eq!(parsed.commit_mode, CommitMode::NoClobber);
    }

    #[test]
    fn parser_rejects_lossy_output_and_ambiguous_thresholds() {
        let base = [
            "meeting.wav",
            "tracks.wav",
            "--model-package",
            "model.dmp",
            "--model-package-key",
            "model.pub",
            "--promotion-evidence",
            "evidence.json",
            "--promotion-evidence-key",
            "evidence.pub.json",
        ];
        let mut duplicate = base.to_vec();
        duplicate.extend([
            "--minimum-active-probability",
            "0.8",
            "--minimum-active-probability",
            "0.9",
        ]);
        assert!(parse_meeting_speaker_args(&arguments(&duplicate)).is_err());
        let mut lossy = base.to_vec();
        lossy[1] = "tracks.mp3";
        assert!(parse_meeting_speaker_args(&arguments(&lossy)).is_err());
    }
}

fn music_restoration_usage() -> &'static str {
    "\
USAGE:
    denoize music-restore <PROGRAM> <CANDIDATE.wav> --correction <CORRECTION.wav> --report <REPORT.json> --task <codec-repair|bandwidth-extension> --model-package <PACKAGE.dmp> --model-package-key <KEY> --promotion-evidence <EVIDENCE.json> --promotion-evidence-key <PUBLIC-KEY.json> [OPTIONS]
    denoize music-restore evidence verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]

Render a bounded restoration candidate for one complete mono or stereo music
mixture. This operation repairs codec damage or extends bandwidth; it never
claims recovered ground truth, estimates dry stems, or applies creative
mastering. Confidently clean and uncertain frames remain unchanged. The
candidate, exact in-memory correction residual, and path-free audit report are
all required publication artifacts. Only authenticated package-v2 models with
accepted, task-matched promotion evidence may process audio.

OPTIONS:
        --correction <PATH.wav>                    required float32 correction residual
        --report <PATH.json>                       required closed path-free audit report
        --task <NAME>                              codec-repair|bandwidth-extension (required)
        --model-package <PATH>                     required signed runtime package v2
        --model-package-key <PATH>                 trusted Minisign public key
        --promotion-evidence <PATH>                accepted signed restoration evidence
        --promotion-evidence-key <PATH>            trusted Ed25519 evidence public key
        --minimum-apply-probability <F>            apply threshold, 0.5..1 (default: 0.8)
        --minimum-bypass-probability <F>           clean-bypass threshold, 0.5..1 (default: 0.8)
        --minimum-apply-frames <N>                 consecutive frames needed to apply, 1..100 (default: 2)
        --maximum-output-peak <F>                  candidate absolute peak, 0.5..1 (default: 1)
        --maximum-absolute-correction <F>          per-sample correction limit, 0.01..1 (default: 0.5)
        --maximum-stereo-correlation-delta <F>     stereo correlation change, 0..0.25 (default: 0.05)
        --maximum-mid-side-ratio-delta-db <F>      mid/side energy change, 0..6 dB (default: 1.5)
        --accelerator <NAME>                       cpu|auto|gpu|metal|cuda (default: cpu)
        --max-memory <MB>                          bound decode, model, candidate, and residual memory
        --replace                                  atomically replace all three destinations
        --json                                     emit compact report JSON
        --pretty                                   emit indented report JSON
    -h, --help                                     show this help
"
}

#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
#[derive(Debug)]
struct MusicRestorationCliOptions {
    input: String,
    output: String,
    correction: String,
    report: String,
    package: String,
    package_key: String,
    promotion_evidence: String,
    promotion_evidence_key: String,
    config: denoize::MusicRestorationConfig,
    accelerator: AcceleratorPreference,
    max_memory_mb: Option<usize>,
    commit_mode: CommitMode,
    print_mode: DiagnosticPrintMode,
}

fn parse_music_restoration_task(value: &str) -> Result<denoize::MusicRestorationTask, String> {
    match value {
        "codec-repair" => Ok(denoize::MusicRestorationTask::CodecRepair),
        "bandwidth-extension" => Ok(denoize::MusicRestorationTask::BandwidthExtension),
        _ => Err(format!(
            "unknown music-restoration task: {value} (expected codec-repair or bandwidth-extension)"
        )),
    }
}

fn parse_music_restoration_args(args: &[String]) -> Result<MusicRestorationCliOptions, String> {
    let mut positional = Vec::new();
    let mut correction = None;
    let mut report = None;
    let mut task = None;
    let mut package = None;
    let mut package_key = None;
    let mut promotion_evidence = None;
    let mut promotion_evidence_key = None;
    let mut config = denoize::MusicRestorationConfig::default();
    let mut accelerator = AcceleratorPreference::Cpu;
    let mut accelerator_seen = false;
    let mut max_memory_mb = None;
    let mut commit_mode = CommitMode::NoClobber;
    let mut print_mode = DiagnosticPrintMode::Human;
    let mut scalar_options = std::collections::HashSet::new();
    let mut index = 0usize;
    while index < args.len() {
        let argument = args[index].as_str();
        if matches!(
            argument,
            "--minimum-apply-probability"
                | "--minimum-bypass-probability"
                | "--minimum-apply-frames"
                | "--maximum-output-peak"
                | "--maximum-absolute-correction"
                | "--maximum-stereo-correlation-delta"
                | "--maximum-mid-side-ratio-delta-db"
        ) && !scalar_options.insert(argument)
        {
            return Err(format!("{argument} may be supplied only once"));
        }
        match argument {
            "--correction" if correction.is_none() => {
                correction = Some(parse_value(args, &mut index, "--correction")?);
            }
            "--correction" => return Err("--correction may be supplied only once".into()),
            "--report" if report.is_none() => {
                report = Some(parse_value(args, &mut index, "--report")?);
            }
            "--report" => return Err("--report may be supplied only once".into()),
            "--task" if task.is_none() => {
                let value: String = parse_value(args, &mut index, "--task")?;
                task = Some(parse_music_restoration_task(&value)?);
            }
            "--task" => return Err("--task may be supplied only once".into()),
            "--model-package" if package.is_none() => {
                package = Some(parse_value(args, &mut index, "--model-package")?);
            }
            "--model-package" => return Err("--model-package may be supplied only once".into()),
            "--model-package-key" if package_key.is_none() => {
                package_key = Some(parse_value(args, &mut index, "--model-package-key")?);
            }
            "--model-package-key" => {
                return Err("--model-package-key may be supplied only once".into());
            }
            "--promotion-evidence" if promotion_evidence.is_none() => {
                promotion_evidence = Some(parse_value(args, &mut index, "--promotion-evidence")?);
            }
            "--promotion-evidence" => {
                return Err("--promotion-evidence may be supplied only once".into());
            }
            "--promotion-evidence-key" if promotion_evidence_key.is_none() => {
                promotion_evidence_key =
                    Some(parse_value(args, &mut index, "--promotion-evidence-key")?);
            }
            "--promotion-evidence-key" => {
                return Err("--promotion-evidence-key may be supplied only once".into());
            }
            "--minimum-apply-probability" => {
                config.minimum_apply_probability =
                    parse_value(args, &mut index, "--minimum-apply-probability")?;
            }
            "--minimum-bypass-probability" => {
                config.minimum_bypass_probability =
                    parse_value(args, &mut index, "--minimum-bypass-probability")?;
            }
            "--minimum-apply-frames" => {
                config.minimum_apply_frames =
                    parse_value(args, &mut index, "--minimum-apply-frames")?;
            }
            "--maximum-output-peak" => {
                config.maximum_output_peak =
                    parse_value(args, &mut index, "--maximum-output-peak")?;
            }
            "--maximum-absolute-correction" => {
                config.maximum_absolute_correction =
                    parse_value(args, &mut index, "--maximum-absolute-correction")?;
            }
            "--maximum-stereo-correlation-delta" => {
                config.maximum_stereo_correlation_delta =
                    parse_value(args, &mut index, "--maximum-stereo-correlation-delta")?;
            }
            "--maximum-mid-side-ratio-delta-db" => {
                config.maximum_mid_side_energy_ratio_delta_db =
                    parse_value(args, &mut index, "--maximum-mid-side-ratio-delta-db")?;
            }
            "--accelerator" if !accelerator_seen => {
                accelerator_seen = true;
                let value: String = parse_value(args, &mut index, "--accelerator")?;
                accelerator = AcceleratorPreference::parse(&value).ok_or_else(|| {
                    format!(
                        "unknown music-restoration accelerator: {value} (expected cpu, auto, gpu, metal, or cuda)"
                    )
                })?;
            }
            "--accelerator" => return Err("--accelerator may be supplied only once".into()),
            "--max-memory" if max_memory_mb.is_none() => {
                max_memory_mb = Some(parse_value(args, &mut index, "--max-memory")?);
            }
            "--max-memory" => return Err("--max-memory may be supplied only once".into()),
            "--replace" if commit_mode == CommitMode::NoClobber => {
                commit_mode = CommitMode::Replace;
            }
            "--replace" => return Err("--replace may be supplied only once".into()),
            "--json" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::Json;
            }
            "--pretty" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err("music-restore accepts only one of --json or --pretty".into());
            }
            "-h" | "--help" => return Err("music-restore help requested".into()),
            "-" => {
                return Err(
                    "music-restore requires regular-file paths; stdin/stdout are unsupported"
                        .into(),
                );
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown music-restore option: {value}"));
            }
            value => {
                if positional.len() == 2 {
                    return Err(format!("unexpected extra music-restore argument: {value}"));
                }
                positional.push(value.to_string());
            }
        }
        index += 1;
    }
    let input = positional
        .first()
        .cloned()
        .ok_or("music-restore requires PROGRAM")?;
    let output = positional
        .get(1)
        .cloned()
        .ok_or("music-restore requires CANDIDATE.wav")?;
    let correction = correction.ok_or("music-restore requires --correction")?;
    let report = report.ok_or("music-restore requires --report")?;
    let task = task.ok_or("music-restore requires --task")?;
    for (path, context) in [(&output, "candidate"), (&correction, "correction residual")] {
        if OutputFormat::from_path(std::path::Path::new(path))? != OutputFormat::Wav {
            return Err(format!(
                "music-restore {context} output must be WAV to avoid lossy encoding"
            ));
        }
    }
    if std::path::Path::new(&report)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        != Some("json")
    {
        return Err("music-restore report must use a .json extension".into());
    }
    config.task = task;
    let package = package.ok_or("music-restore requires --model-package")?;
    let package_key = package_key.ok_or("music-restore requires --model-package-key")?;
    let promotion_evidence =
        promotion_evidence.ok_or("music-restore requires --promotion-evidence")?;
    let promotion_evidence_key =
        promotion_evidence_key.ok_or("music-restore requires --promotion-evidence-key")?;
    checked_mib_limit_bytes(max_memory_mb, "--max-memory")?;
    config.validate()?;
    Ok(MusicRestorationCliOptions {
        input,
        output,
        correction,
        report,
        package,
        package_key,
        promotion_evidence,
        promotion_evidence_key,
        config,
        accelerator,
        max_memory_mb,
        commit_mode,
        print_mode,
    })
}

fn run_music_restoration(args: &[String]) -> Result<(), String> {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help") {
        print!("{}", music_restoration_usage());
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("evidence") {
        return run_music_restoration_evidence(&args[1..]);
    }
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        return Err("music-restore --help accepts no other arguments".into());
    }
    let options = parse_music_restoration_args(args)?;
    validate_music_restoration_publication_paths(&options)?;
    run_music_restoration_audio(options)
}

fn run_music_restoration_evidence(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) != Some("verify") {
        return Err(
            "music-restoration evidence requires: verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]"
                .into(),
        );
    }
    let mut positional = Vec::new();
    let mut mode = DiagnosticPrintMode::Human;
    for argument in &args[1..] {
        match argument.as_str() {
            "--json" if mode == DiagnosticPrintMode::Human => mode = DiagnosticPrintMode::Json,
            "--pretty" if mode == DiagnosticPrintMode::Human => {
                mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err(
                    "music-restoration evidence verify accepts only one output mode".into(),
                );
            }
            value if value.starts_with('-') => {
                return Err(format!(
                    "unknown music-restoration evidence option: {value}"
                ));
            }
            value => positional.push(value.to_string()),
        }
    }
    if positional.len() != 2 {
        return Err(
            "music-restoration evidence verify requires EVIDENCE.json and PUBLIC-KEY.json".into(),
        );
    }
    let evidence = denoize::SignedMusicRestorationPromotionEvidence::from_file(&positional[0])?;
    let key = ReceiptPublicKey::from_file(&positional[1])?;
    evidence.verify_signature(&key)?;
    match mode {
        DiagnosticPrintMode::Json => println!(
            "{}",
            serde_json::to_string(&evidence)
                .map_err(|error| format!("serialize music-restoration evidence: {error}"))?
        ),
        DiagnosticPrintMode::PrettyJson => println!("{}", evidence.to_pretty_json()?),
        DiagnosticPrintMode::Human => println!(
            "verified music-restoration evidence: task={:?} strata={} paired_clips={} full_tracks={} listeners={} restricted_artifacts={} accepted={}",
            evidence.payload.task,
            evidence.payload.strata.len(),
            evidence.payload.paired_clips,
            evidence.payload.full_length_tracks,
            evidence.payload.listener_count,
            evidence.payload.redistributed_restricted_artifacts,
            evidence.payload.accepted,
        ),
    }
    if !evidence.payload.accepted {
        return Err(
            "music-restoration evidence is authentic but does not pass promotion gates".into(),
        );
    }
    Ok(())
}

fn validate_music_restoration_publication_paths(
    options: &MusicRestorationCliOptions,
) -> Result<(), String> {
    let mut sources = Vec::new();
    for (path, context) in [
        (options.input.as_str(), "music-restoration input"),
        (options.package.as_str(), "music-restoration package"),
        (
            options.package_key.as_str(),
            "music-restoration package key",
        ),
        (
            options.promotion_evidence.as_str(),
            "music-restoration evidence",
        ),
        (
            options.promotion_evidence_key.as_str(),
            "music-restoration evidence key",
        ),
    ] {
        sources.push((
            std::fs::canonicalize(path)
                .map_err(|error| format!("resolve {context} {path}: {error}"))?,
            context,
        ));
    }
    for left in 0..sources.len() {
        for right in left + 1..sources.len() {
            if sources[left].0 == sources[right].0 {
                return Err(format!(
                    "{} and {} must use distinct source files",
                    sources[left].1, sources[right].1
                ));
            }
        }
    }
    let mut destinations = Vec::new();
    for (path, context) in [
        (options.output.as_str(), "music-restoration candidate"),
        (
            options.correction.as_str(),
            "music-restoration correction residual",
        ),
        (options.report.as_str(), "music-restoration report"),
    ] {
        let normalized = normalized_project_destination(std::path::Path::new(path), context)?;
        let existing = std::fs::canonicalize(&normalized).ok();
        if sources.iter().any(|(source, _)| {
            normalized == *source || existing.as_ref().is_some_and(|path| path == source)
        }) {
            return Err(format!(
                "{context} must not replace an input, package, key, or evidence document"
            ));
        }
        ensure_restoration_destination_available(&normalized, options.commit_mode)?;
        destinations.push((batch_collision_key(&normalized), context));
    }
    destinations.sort_by(|left, right| left.0.cmp(&right.0));
    if let Some(pair) = destinations.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        return Err(format!(
            "{} and {} must use distinct destinations",
            pair[0].1, pair[1].1
        ));
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn run_music_restoration_audio(options: MusicRestorationCliOptions) -> Result<(), String> {
    let maximum = checked_mib_limit_bytes(options.max_memory_mb, "--max-memory")?;
    let evidence =
        denoize::SignedMusicRestorationPromotionEvidence::from_file(&options.promotion_evidence)?;
    let evidence_key = ReceiptPublicKey::from_file(&options.promotion_evidence_key)?;
    let package = RuntimeModelPackage::open(&options.package, &options.package_key)?;
    // Authenticate the package, graph, numerical vectors, evaluation evidence,
    // task, configuration, and accelerator before opening user-controlled audio.
    let session = denoize::MusicRestorationSession::prepare(
        package,
        &evidence,
        &evidence_key,
        &options.config,
        options.accelerator,
    )?;
    let model_working_set = session.model_working_set_bytes()?;
    ensure_memory_limit(
        model_working_set,
        options.max_memory_mb,
        "music-restoration model working set",
    )?;
    let mut input_session = AudioInputSession::open(&options.input)?;
    let session_memory = estimate_session_memory_bytes(&input_session);
    ensure_memory_limit(
        model_working_set.saturating_add(session_memory),
        options.max_memory_mb,
        "music-restoration input/model preflight",
    )?;
    let decode_maximum = maximum.map(|limit| {
        limit
            .saturating_sub(model_working_set)
            .saturating_sub(session_memory)
    });
    let input = read_audio_from_session_with_limits(
        &mut input_session,
        DecodeLimits::new(
            metadata_limits_for_available_bytes(decode_maximum),
            decode_maximum,
        ),
    )?;
    let working_set = session
        .processing_working_set_bytes(&input)?
        .saturating_add(model_working_set)
        .saturating_add(session_memory);
    ensure_memory_limit(
        working_set,
        options.max_memory_mb,
        "music-restoration decoded/model/candidate/residual working set",
    )?;
    let result = session.restore(&input, &options.config)?;
    let encode_options = EncodeOptions::default();
    OutputFormat::Wav.validate_config(&result.output, &encode_options)?;
    OutputFormat::Wav.validate_config(&result.correction, &encode_options)?;

    // Finish every potentially fallible encode before publishing any artifact.
    let mut staged_candidate = AtomicOutput::new(&options.output)?;
    denoize::encode::write_audio_to_file(
        staged_candidate.file_mut(),
        OutputFormat::Wav,
        &result.output,
        encode_options,
    )?;
    let mut staged_correction = AtomicOutput::new(&options.correction)?;
    denoize::encode::write_audio_to_file(
        staged_correction.file_mut(),
        OutputFormat::Wav,
        &result.correction,
        encode_options,
    )?;
    let staged_report = stage_restoration_json(&options.report, &result.report)?;

    // Publish the candidate last: a partial multi-file commit may leave audit
    // artifacts, but never an unaudited candidate without its residual/report.
    staged_correction.commit(options.commit_mode)?;
    staged_report.commit(options.commit_mode)?;
    staged_candidate.commit(options.commit_mode)?;

    match options.print_mode {
        DiagnosticPrintMode::Json => println!("{}", result.report.to_json()?),
        DiagnosticPrintMode::PrettyJson => println!("{}", result.report.to_pretty_json()?),
        DiagnosticPrintMode::Human => println!(
            "music restoration: task={:?} channels={} frames={} applied_frames={} uncertain_frames={} changed_samples={} maximum_correction={:.6} package={}",
            result.report.task,
            result.report.source_channels,
            result.report.source_frames,
            result.report.applied_decision_frames,
            result.report.uncertain_decision_frames,
            result.report.changed_samples,
            result.report.maximum_absolute_correction,
            result.report.model.package_sha256,
        ),
    }
    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn run_music_restoration_audio(_options: MusicRestorationCliOptions) -> Result<(), String> {
    Err("music restoration requires a build with the onnx feature".into())
}

#[cfg(test)]
mod music_restoration_cli_tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    fn valid_arguments() -> Vec<String> {
        arguments(&[
            "program.wav",
            "candidate.wav",
            "--correction",
            "correction.wav",
            "--report",
            "report.json",
            "--task",
            "codec-repair",
            "--model-package",
            "model.dmp",
            "--model-package-key",
            "model.pub",
            "--promotion-evidence",
            "evidence.json",
            "--promotion-evidence-key",
            "evidence.pub.json",
        ])
    }

    #[test]
    fn parser_requires_every_audit_artifact_and_explicit_task() {
        let error = parse_music_restoration_args(&arguments(&["program.wav", "candidate.wav"]))
            .unwrap_err();
        assert_eq!(error, "music-restore requires --correction");
        let parsed = parse_music_restoration_args(&valid_arguments()).unwrap();
        assert_eq!(
            parsed.config.task,
            denoize::MusicRestorationTask::CodecRepair
        );
        assert_eq!(parsed.config.minimum_apply_probability, 0.8);
        assert_eq!(parsed.commit_mode, CommitMode::NoClobber);
    }

    #[test]
    fn parser_rejects_lossy_residual_and_duplicate_thresholds() {
        let mut lossy = valid_arguments();
        let correction = lossy
            .iter()
            .position(|argument| argument == "correction.wav")
            .unwrap();
        lossy[correction] = "correction.mp3".into();
        assert!(parse_music_restoration_args(&lossy).is_err());

        let mut duplicate = valid_arguments();
        duplicate.extend([
            "--minimum-apply-probability".into(),
            "0.8".into(),
            "--minimum-apply-probability".into(),
            "0.9".into(),
        ]);
        assert!(parse_music_restoration_args(&duplicate).is_err());
    }
}

fn target_sound_usage() -> &'static str {
    "\
USAGE:
    denoize target-sound <INPUT> --query <QUERY.json> --target <TARGET.wav> --residual <RESIDUAL.wav> --output <OUTPUT.wav> --report <REPORT.json> --mode <preserve|remove> --model-package <PACKAGE.dmp> --model-package-key <KEY> --promotion-evidence <EVIDENCE.json> --promotion-evidence-key <PUBLIC-KEY.json> [OPTIONS]
    denoize target-sound causal <INPUT> --query <QUERY.json> --target <TARGET.wav> --residual <RESIDUAL.wav> --output <OUTPUT.wav> --report <REPORT.json> --mode <preserve|remove> --model-package <CAUSAL-PACKAGE.dmp> --model-package-key <KEY> --offline-promotion-evidence <OFFLINE.json> --offline-promotion-evidence-key <KEY.json> --causal-promotion-evidence <CAUSAL.json> --causal-promotion-evidence-key <KEY.json> [OPTIONS]
    denoize target-sound evidence verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]
    denoize target-sound causal evidence verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]

Extract or remove one sound selected from an authenticated finite class
catalog. Open text is never accepted or sent to the model. The graph must
produce target, residual, and calibrated absent/uncertain/present values; all
audio publication is withheld unless presence and every conservation, signal,
geometry, spatial, package, license, and evaluation gate passes. The three WAV
artifacts and path-free report are published together when accepted. No model,
checkpoint, catalog, or dataset is bundled.

The causal form preserves continuous timing and always publishes a complete
decomposition. Unsafe, absent, uncertain, warm-up, late, or overloaded blocks
use target silence plus untouched residual. Its separately signed evidence
binds the accepted offline baseline, recurrent package, snapshot/reset tests,
named-device end-to-end latency, callback audit, and transition audit.

OPTIONS:
        --query <PATH.json>                         complete ordered finite catalog and selected ID
        --target <PATH.wav>                         required float WAV target estimate
        --residual <PATH.wav>                       required float WAV exact mixture residual
        --output <PATH.wav>                         target for preserve; residual for remove
        --report <PATH.json>                        required closed path-free audit report
        --mode <NAME>                               preserve|remove (required)
        --model-package <PATH>                      required signed runtime package v2
        --model-package-key <PATH>                  trusted Minisign public key
        --promotion-evidence <PATH>                 accepted signed target-sound evidence
        --promotion-evidence-key <PATH>             trusted Ed25519 evidence public key
        --offline-promotion-evidence <PATH>         accepted offline baseline evidence (causal form)
        --offline-promotion-evidence-key <PATH>     trusted offline Ed25519 key (causal form)
        --causal-promotion-evidence <PATH>          accepted causal non-inferiority evidence
        --causal-promotion-evidence-key <PATH>      trusted causal Ed25519 key
        --minimum-present-probability <F>           present threshold, 0.5..1 (default: 0.9)
        --minimum-absent-probability <F>            absent threshold, 0.5..1 (default: 0.9)
        --present-hold-blocks <N>                   consecutive present blocks, 1..100 (causal default: 3)
        --maximum-model-recombination-error <F>     graph target+residual error, 0..0.1 (default: 0.01)
        --maximum-publication-recombination-error <F> source-rate error (offline 0..1e-6, default 1e-12; causal 0..1e-5, default 1e-6)
        --maximum-target-peak <F>                   target absolute peak, 0.5..1 (default: 1)
        --maximum-residual-peak <F>                 residual absolute peak, 0.5..1 (default: 1)
        --maximum-energy-gain-db <F>                target/residual energy gain, 0..12 dB (default: 3)
        --maximum-stereo-correlation-delta <F>      resampling spatial drift, 0..0.25 (default: 0.05)
        --maximum-mid-side-ratio-delta-db <F>       resampling spatial drift, 0..6 dB (default: 1.5)
        --accelerator <NAME>                        cpu|auto|gpu|metal|cuda (default: cpu)
        --max-memory <MB>                           bound decode, model, target, residual, and report memory
        --replace                                   atomically replace each published destination
        --json                                      emit compact report JSON
        --pretty                                    emit indented report JSON
    -h, --help                                      show this help
"
}

#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
#[derive(Debug)]
struct TargetSoundCliOptions {
    input: String,
    query: String,
    target: String,
    residual: String,
    output: String,
    report: String,
    package: String,
    package_key: String,
    promotion_evidence: String,
    promotion_evidence_key: String,
    config: denoize::TargetSoundConfig,
    accelerator: AcceleratorPreference,
    max_memory_mb: Option<usize>,
    commit_mode: CommitMode,
    print_mode: DiagnosticPrintMode,
}

fn parse_target_sound_mode(value: &str) -> Result<denoize::TargetSoundMode, String> {
    match value {
        "preserve" => Ok(denoize::TargetSoundMode::Preserve),
        "remove" => Ok(denoize::TargetSoundMode::Remove),
        _ => Err(format!(
            "unknown target-sound mode: {value} (expected preserve or remove)"
        )),
    }
}

fn parse_target_sound_args(args: &[String]) -> Result<TargetSoundCliOptions, String> {
    let mut positional = Vec::new();
    let mut query = None;
    let mut target = None;
    let mut residual = None;
    let mut output = None;
    let mut report = None;
    let mut mode = None;
    let mut package = None;
    let mut package_key = None;
    let mut promotion_evidence = None;
    let mut promotion_evidence_key = None;
    let mut config = denoize::TargetSoundConfig::default();
    let mut accelerator = AcceleratorPreference::Cpu;
    let mut accelerator_seen = false;
    let mut max_memory_mb = None;
    let mut commit_mode = CommitMode::NoClobber;
    let mut print_mode = DiagnosticPrintMode::Human;
    let mut scalar_options = std::collections::HashSet::new();
    let mut index = 0usize;
    while index < args.len() {
        let argument = args[index].as_str();
        if matches!(
            argument,
            "--minimum-present-probability"
                | "--minimum-absent-probability"
                | "--maximum-model-recombination-error"
                | "--maximum-publication-recombination-error"
                | "--maximum-target-peak"
                | "--maximum-residual-peak"
                | "--maximum-energy-gain-db"
                | "--maximum-stereo-correlation-delta"
                | "--maximum-mid-side-ratio-delta-db"
        ) && !scalar_options.insert(argument)
        {
            return Err(format!("{argument} may be supplied only once"));
        }
        match argument {
            "--query" if query.is_none() => {
                query = Some(parse_value(args, &mut index, "--query")?);
            }
            "--query" => return Err("--query may be supplied only once".into()),
            "--target" if target.is_none() => {
                target = Some(parse_value(args, &mut index, "--target")?);
            }
            "--target" => return Err("--target may be supplied only once".into()),
            "--residual" if residual.is_none() => {
                residual = Some(parse_value(args, &mut index, "--residual")?);
            }
            "--residual" => return Err("--residual may be supplied only once".into()),
            "--output" if output.is_none() => {
                output = Some(parse_value(args, &mut index, "--output")?);
            }
            "--output" => return Err("--output may be supplied only once".into()),
            "--report" if report.is_none() => {
                report = Some(parse_value(args, &mut index, "--report")?);
            }
            "--report" => return Err("--report may be supplied only once".into()),
            "--mode" if mode.is_none() => {
                let value: String = parse_value(args, &mut index, "--mode")?;
                mode = Some(parse_target_sound_mode(&value)?);
            }
            "--mode" => return Err("--mode may be supplied only once".into()),
            "--model-package" if package.is_none() => {
                package = Some(parse_value(args, &mut index, "--model-package")?);
            }
            "--model-package" => return Err("--model-package may be supplied only once".into()),
            "--model-package-key" if package_key.is_none() => {
                package_key = Some(parse_value(args, &mut index, "--model-package-key")?);
            }
            "--model-package-key" => {
                return Err("--model-package-key may be supplied only once".into());
            }
            "--promotion-evidence" if promotion_evidence.is_none() => {
                promotion_evidence = Some(parse_value(args, &mut index, "--promotion-evidence")?);
            }
            "--promotion-evidence" => {
                return Err("--promotion-evidence may be supplied only once".into());
            }
            "--promotion-evidence-key" if promotion_evidence_key.is_none() => {
                promotion_evidence_key =
                    Some(parse_value(args, &mut index, "--promotion-evidence-key")?);
            }
            "--promotion-evidence-key" => {
                return Err("--promotion-evidence-key may be supplied only once".into());
            }
            "--minimum-present-probability" => {
                config.minimum_present_probability =
                    parse_value(args, &mut index, "--minimum-present-probability")?;
            }
            "--minimum-absent-probability" => {
                config.minimum_absent_probability =
                    parse_value(args, &mut index, "--minimum-absent-probability")?;
            }
            "--maximum-model-recombination-error" => {
                config.maximum_model_recombination_error =
                    parse_value(args, &mut index, "--maximum-model-recombination-error")?;
            }
            "--maximum-publication-recombination-error" => {
                config.maximum_publication_recombination_error = parse_value(
                    args,
                    &mut index,
                    "--maximum-publication-recombination-error",
                )?;
            }
            "--maximum-target-peak" => {
                config.maximum_target_peak =
                    parse_value(args, &mut index, "--maximum-target-peak")?;
            }
            "--maximum-residual-peak" => {
                config.maximum_residual_peak =
                    parse_value(args, &mut index, "--maximum-residual-peak")?;
            }
            "--maximum-energy-gain-db" => {
                config.maximum_energy_gain_db =
                    parse_value(args, &mut index, "--maximum-energy-gain-db")?;
            }
            "--maximum-stereo-correlation-delta" => {
                config.maximum_stereo_correlation_delta =
                    parse_value(args, &mut index, "--maximum-stereo-correlation-delta")?;
            }
            "--maximum-mid-side-ratio-delta-db" => {
                config.maximum_mid_side_energy_ratio_delta_db =
                    parse_value(args, &mut index, "--maximum-mid-side-ratio-delta-db")?;
            }
            "--accelerator" if !accelerator_seen => {
                accelerator_seen = true;
                let value: String = parse_value(args, &mut index, "--accelerator")?;
                accelerator = AcceleratorPreference::parse(&value).ok_or_else(|| {
                    format!(
                        "unknown target-sound accelerator: {value} (expected cpu, auto, gpu, metal, or cuda)"
                    )
                })?;
            }
            "--accelerator" => return Err("--accelerator may be supplied only once".into()),
            "--max-memory" if max_memory_mb.is_none() => {
                max_memory_mb = Some(parse_value(args, &mut index, "--max-memory")?);
            }
            "--max-memory" => return Err("--max-memory may be supplied only once".into()),
            "--replace" if commit_mode == CommitMode::NoClobber => {
                commit_mode = CommitMode::Replace;
            }
            "--replace" => return Err("--replace may be supplied only once".into()),
            "--json" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::Json;
            }
            "--pretty" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err("target-sound accepts only one of --json or --pretty".into());
            }
            "-h" | "--help" => return Err("target-sound help requested".into()),
            "-" => {
                return Err(
                    "target-sound requires regular-file paths; stdin/stdout are unsupported".into(),
                );
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown target-sound option: {value}"));
            }
            value => {
                if !positional.is_empty() {
                    return Err(format!("unexpected extra target-sound argument: {value}"));
                }
                positional.push(value.to_string());
            }
        }
        index += 1;
    }
    let input = positional
        .first()
        .cloned()
        .ok_or("target-sound requires INPUT")?;
    let query = query.ok_or("target-sound requires --query")?;
    let target = target.ok_or("target-sound requires --target")?;
    let residual = residual.ok_or("target-sound requires --residual")?;
    let output = output.ok_or("target-sound requires --output")?;
    let report = report.ok_or("target-sound requires --report")?;
    config.mode = mode.ok_or("target-sound requires --mode")?;
    for (path, context) in [
        (&target, "target"),
        (&residual, "residual"),
        (&output, "selected output"),
    ] {
        if OutputFormat::from_path(std::path::Path::new(path))? != OutputFormat::Wav {
            return Err(format!(
                "target-sound {context} must be WAV to avoid lossy encoding"
            ));
        }
    }
    if std::path::Path::new(&report)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        != Some("json")
    {
        return Err("target-sound report must use a .json extension".into());
    }
    let package = package.ok_or("target-sound requires --model-package")?;
    let package_key = package_key.ok_or("target-sound requires --model-package-key")?;
    let promotion_evidence =
        promotion_evidence.ok_or("target-sound requires --promotion-evidence")?;
    let promotion_evidence_key =
        promotion_evidence_key.ok_or("target-sound requires --promotion-evidence-key")?;
    checked_mib_limit_bytes(max_memory_mb, "--max-memory")?;
    config.validate()?;
    Ok(TargetSoundCliOptions {
        input,
        query,
        target,
        residual,
        output,
        report,
        package,
        package_key,
        promotion_evidence,
        promotion_evidence_key,
        config,
        accelerator,
        max_memory_mb,
        commit_mode,
        print_mode,
    })
}

fn run_target_sound(args: &[String]) -> Result<(), String> {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help") {
        print!("{}", target_sound_usage());
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("evidence") {
        return run_target_sound_evidence(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("causal") {
        return run_causal_target_sound(&args[1..]);
    }
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        return Err("target-sound --help accepts no other arguments".into());
    }
    let options = parse_target_sound_args(args)?;
    validate_target_sound_publication_paths(&options)?;
    run_target_sound_audio(options)
}

fn run_target_sound_evidence(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) != Some("verify") {
        return Err(
            "target-sound evidence requires: verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]"
                .into(),
        );
    }
    let mut positional = Vec::new();
    let mut mode = DiagnosticPrintMode::Human;
    for argument in &args[1..] {
        match argument.as_str() {
            "--json" if mode == DiagnosticPrintMode::Human => mode = DiagnosticPrintMode::Json,
            "--pretty" if mode == DiagnosticPrintMode::Human => {
                mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err("target-sound evidence verify accepts only one output mode".into());
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown target-sound evidence option: {value}"));
            }
            value => positional.push(value.to_string()),
        }
    }
    if positional.len() != 2 {
        return Err(
            "target-sound evidence verify requires EVIDENCE.json and PUBLIC-KEY.json".into(),
        );
    }
    let evidence = denoize::SignedTargetSoundPromotionEvidence::from_file(&positional[0])?;
    let key = ReceiptPublicKey::from_file(&positional[1])?;
    evidence.verify_signature(&key)?;
    match mode {
        DiagnosticPrintMode::Json => println!(
            "{}",
            serde_json::to_string(&evidence)
                .map_err(|error| format!("serialize target-sound evidence: {error}"))?
        ),
        DiagnosticPrintMode::PrettyJson => println!("{}", evidence.to_pretty_json()?),
        DiagnosticPrintMode::Human => println!(
            "verified target-sound evidence: package={} catalog={} classes={}/{} per_class_present={} per_class_absent={} worst_fp={} worst_fn={} strata={} paired_cases={} absent={} protected={} binaural={} listeners={} accepted={}",
            evidence.payload.model_package_sha256,
            evidence.payload.query_catalog_sha256,
            evidence.payload.evaluated_class_count,
            evidence.payload.query_class_count,
            evidence.payload.minimum_present_cases_per_class,
            evidence.payload.minimum_absent_cases_per_class,
            evidence.payload.worst_class_false_positive_rate,
            evidence.payload.worst_class_false_negative_rate,
            evidence.payload.strata.len(),
            evidence.payload.paired_cases,
            evidence.payload.target_absent_cases,
            evidence.payload.protected_foreground_cases,
            evidence.payload.binaural_cases,
            evidence.payload.listener_count,
            evidence.payload.accepted,
        ),
    }
    if !evidence.payload.accepted {
        return Err("target-sound evidence is authentic but does not pass promotion gates".into());
    }
    Ok(())
}

fn validate_target_sound_publication_paths(options: &TargetSoundCliOptions) -> Result<(), String> {
    let mut sources = Vec::new();
    for (path, context) in [
        (options.input.as_str(), "target-sound input"),
        (options.query.as_str(), "target-sound query"),
        (options.package.as_str(), "target-sound package"),
        (options.package_key.as_str(), "target-sound package key"),
        (
            options.promotion_evidence.as_str(),
            "target-sound promotion evidence",
        ),
        (
            options.promotion_evidence_key.as_str(),
            "target-sound promotion evidence key",
        ),
    ] {
        sources.push((
            std::fs::canonicalize(path)
                .map_err(|error| format!("resolve {context} {path}: {error}"))?,
            context,
        ));
    }
    for left in 0..sources.len() {
        for right in left + 1..sources.len() {
            if sources[left].0 == sources[right].0 {
                return Err(format!(
                    "{} and {} must use distinct source files",
                    sources[left].1, sources[right].1
                ));
            }
        }
    }
    let mut destinations = Vec::new();
    for (path, context) in [
        (options.target.as_str(), "target-sound target"),
        (options.residual.as_str(), "target-sound residual"),
        (options.output.as_str(), "target-sound selected output"),
        (options.report.as_str(), "target-sound report"),
    ] {
        let normalized = normalized_project_destination(std::path::Path::new(path), context)?;
        let existing = std::fs::canonicalize(&normalized).ok();
        if sources.iter().any(|(source, _)| {
            normalized == *source || existing.as_ref().is_some_and(|path| path == source)
        }) {
            return Err(format!(
                "{context} must not replace an input, query, package, key, or evidence document"
            ));
        }
        ensure_restoration_destination_available(&normalized, options.commit_mode)?;
        destinations.push((batch_collision_key(&normalized), context));
    }
    destinations.sort_by(|left, right| left.0.cmp(&right.0));
    if let Some(pair) = destinations.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        return Err(format!(
            "{} and {} must use distinct destinations",
            pair[0].1, pair[1].1
        ));
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn run_target_sound_audio(options: TargetSoundCliOptions) -> Result<(), String> {
    let maximum = checked_mib_limit_bytes(options.max_memory_mb, "--max-memory")?;
    let query = denoize::TargetSoundQuery::from_file(&options.query)?;
    let evidence =
        denoize::SignedTargetSoundPromotionEvidence::from_file(&options.promotion_evidence)?;
    let evidence_key = ReceiptPublicKey::from_file(&options.promotion_evidence_key)?;
    let package = RuntimeModelPackage::open(&options.package, &options.package_key)?;
    // Authenticate package bytes, tensor semantics, numerical vectors,
    // configuration, finite catalog, licenses, and promotion evidence before
    // opening user-controlled audio.
    let session = denoize::TargetSoundSession::prepare(
        package,
        &evidence,
        &evidence_key,
        &query,
        &options.config,
        options.accelerator,
    )?;
    let model_working_set = session.model_working_set_bytes()?;
    ensure_memory_limit(
        model_working_set,
        options.max_memory_mb,
        "target-sound model working set",
    )?;
    let mut input_session = AudioInputSession::open(&options.input)?;
    let session_memory = estimate_session_memory_bytes(&input_session);
    ensure_memory_limit(
        model_working_set.saturating_add(session_memory),
        options.max_memory_mb,
        "target-sound input/model preflight",
    )?;
    let decode_maximum = maximum.map(|limit| {
        limit
            .saturating_sub(model_working_set)
            .saturating_sub(session_memory)
    });
    let input = read_audio_from_session_with_limits(
        &mut input_session,
        DecodeLimits::new(
            metadata_limits_for_available_bytes(decode_maximum),
            decode_maximum,
        ),
    )?;
    let working_set = session
        .processing_working_set_bytes(&input)?
        .saturating_add(model_working_set)
        .saturating_add(session_memory);
    ensure_memory_limit(
        working_set,
        options.max_memory_mb,
        "target-sound decoded/model/target/residual working set",
    )?;
    let result = session.extract(&input, &query, &options.config)?;
    let staged_report = stage_restoration_json(&options.report, &result.report)?;
    if let (Some(target), Some(residual), Some(output)) = (
        result.target.as_ref(),
        result.residual.as_ref(),
        result.output.as_ref(),
    ) {
        let encode_options = EncodeOptions::default();
        OutputFormat::Wav.validate_config(target, &encode_options)?;
        OutputFormat::Wav.validate_config(residual, &encode_options)?;
        OutputFormat::Wav.validate_config(output, &encode_options)?;
        let mut staged_target = AtomicOutput::new(&options.target)?;
        denoize::encode::write_audio_to_file(
            staged_target.file_mut(),
            OutputFormat::Wav,
            target,
            encode_options,
        )?;
        let mut staged_residual = AtomicOutput::new(&options.residual)?;
        denoize::encode::write_audio_to_file(
            staged_residual.file_mut(),
            OutputFormat::Wav,
            residual,
            encode_options,
        )?;
        let mut staged_output = AtomicOutput::new(&options.output)?;
        denoize::encode::write_audio_to_file(
            staged_output.file_mut(),
            OutputFormat::Wav,
            output,
            encode_options,
        )?;
        // Publish the report first and the mode-selected output last. A partial
        // multi-file commit can never leave a selected output without its audit.
        staged_report.commit(options.commit_mode)?;
        staged_target.commit(options.commit_mode)?;
        staged_residual.commit(options.commit_mode)?;
        staged_output.commit(options.commit_mode)?;
    } else {
        // Withheld decisions intentionally publish only the audit report.
        staged_report.commit(options.commit_mode)?;
    }
    match options.print_mode {
        DiagnosticPrintMode::Json => println!("{}", result.report.to_json()?),
        DiagnosticPrintMode::PrettyJson => println!("{}", result.report.to_pretty_json()?),
        DiagnosticPrintMode::Human => {
            println!(
                "target-sound extraction: mode={:?} class={} index={} decision={:?} presence={:?} published={} frames={} package={}",
                result.report.mode,
                result.report.query.class_id,
                result.report.query.class_index,
                result.report.decision,
                result.report.presence.state,
                result.report.output_published,
                result.report.source_frames,
                result.report.model.package_sha256,
            );
            for warning in &result.report.warnings {
                println!("warning: {warning}");
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn run_target_sound_audio(_options: TargetSoundCliOptions) -> Result<(), String> {
    Err("target-sound extraction requires a build with the onnx feature".into())
}

#[cfg(feature = "onnx")]
#[derive(Debug)]
struct CausalTargetSoundCliOptions {
    input: String,
    query: String,
    target: String,
    residual: String,
    output: String,
    report: String,
    package: String,
    package_key: String,
    offline_evidence: String,
    offline_evidence_key: String,
    causal_evidence: String,
    causal_evidence_key: String,
    config: denoize::CausalTargetSoundConfig,
    accelerator: AcceleratorPreference,
    max_memory_mb: Option<usize>,
    commit_mode: CommitMode,
    print_mode: DiagnosticPrintMode,
}

#[cfg(feature = "onnx")]
fn parse_causal_target_sound_args(args: &[String]) -> Result<CausalTargetSoundCliOptions, String> {
    let mut positional = Vec::new();
    let mut query = None;
    let mut target = None;
    let mut residual = None;
    let mut output = None;
    let mut report = None;
    let mut mode = None;
    let mut package = None;
    let mut package_key = None;
    let mut offline_evidence = None;
    let mut offline_evidence_key = None;
    let mut causal_evidence = None;
    let mut causal_evidence_key = None;
    let mut config = denoize::CausalTargetSoundConfig::default();
    let mut accelerator = AcceleratorPreference::Cpu;
    let mut accelerator_seen = false;
    let mut max_memory_mb = None;
    let mut commit_mode = CommitMode::NoClobber;
    let mut print_mode = DiagnosticPrintMode::Human;
    let mut seen = std::collections::HashSet::new();
    let mut index = 0_usize;
    while index < args.len() {
        let argument = args[index].as_str();
        if matches!(
            argument,
            "--minimum-present-probability"
                | "--minimum-absent-probability"
                | "--present-hold-blocks"
                | "--maximum-model-recombination-error"
                | "--maximum-publication-recombination-error"
                | "--maximum-target-peak"
                | "--maximum-residual-peak"
                | "--maximum-energy-gain-db"
                | "--maximum-stereo-correlation-delta"
                | "--maximum-mid-side-ratio-delta-db"
        ) && !seen.insert(argument)
        {
            return Err(format!("{argument} may be supplied only once"));
        }
        match argument {
            "--query" if query.is_none() => {
                query = Some(parse_value(args, &mut index, "--query")?);
            }
            "--query" => return Err("--query may be supplied only once".into()),
            "--target" if target.is_none() => {
                target = Some(parse_value(args, &mut index, "--target")?);
            }
            "--target" => return Err("--target may be supplied only once".into()),
            "--residual" if residual.is_none() => {
                residual = Some(parse_value(args, &mut index, "--residual")?);
            }
            "--residual" => return Err("--residual may be supplied only once".into()),
            "--output" if output.is_none() => {
                output = Some(parse_value(args, &mut index, "--output")?);
            }
            "--output" => return Err("--output may be supplied only once".into()),
            "--report" if report.is_none() => {
                report = Some(parse_value(args, &mut index, "--report")?);
            }
            "--report" => return Err("--report may be supplied only once".into()),
            "--mode" if mode.is_none() => {
                let value: String = parse_value(args, &mut index, "--mode")?;
                mode = Some(parse_target_sound_mode(&value)?);
            }
            "--mode" => return Err("--mode may be supplied only once".into()),
            "--model-package" if package.is_none() => {
                package = Some(parse_value(args, &mut index, "--model-package")?);
            }
            "--model-package" => return Err("--model-package may be supplied only once".into()),
            "--model-package-key" if package_key.is_none() => {
                package_key = Some(parse_value(args, &mut index, "--model-package-key")?);
            }
            "--model-package-key" => {
                return Err("--model-package-key may be supplied only once".into());
            }
            "--offline-promotion-evidence" if offline_evidence.is_none() => {
                offline_evidence = Some(parse_value(
                    args,
                    &mut index,
                    "--offline-promotion-evidence",
                )?);
            }
            "--offline-promotion-evidence" => {
                return Err("--offline-promotion-evidence may be supplied only once".into());
            }
            "--offline-promotion-evidence-key" if offline_evidence_key.is_none() => {
                offline_evidence_key = Some(parse_value(
                    args,
                    &mut index,
                    "--offline-promotion-evidence-key",
                )?);
            }
            "--offline-promotion-evidence-key" => {
                return Err("--offline-promotion-evidence-key may be supplied only once".into());
            }
            "--causal-promotion-evidence" if causal_evidence.is_none() => {
                causal_evidence = Some(parse_value(
                    args,
                    &mut index,
                    "--causal-promotion-evidence",
                )?);
            }
            "--causal-promotion-evidence" => {
                return Err("--causal-promotion-evidence may be supplied only once".into());
            }
            "--causal-promotion-evidence-key" if causal_evidence_key.is_none() => {
                causal_evidence_key = Some(parse_value(
                    args,
                    &mut index,
                    "--causal-promotion-evidence-key",
                )?);
            }
            "--causal-promotion-evidence-key" => {
                return Err("--causal-promotion-evidence-key may be supplied only once".into());
            }
            "--minimum-present-probability" => {
                config.minimum_present_probability =
                    parse_value(args, &mut index, "--minimum-present-probability")?;
            }
            "--minimum-absent-probability" => {
                config.minimum_absent_probability =
                    parse_value(args, &mut index, "--minimum-absent-probability")?;
            }
            "--present-hold-blocks" => {
                config.present_hold_blocks =
                    parse_value(args, &mut index, "--present-hold-blocks")?;
            }
            "--maximum-model-recombination-error" => {
                config.maximum_model_recombination_error =
                    parse_value(args, &mut index, "--maximum-model-recombination-error")?;
            }
            "--maximum-publication-recombination-error" => {
                config.maximum_publication_recombination_error = parse_value(
                    args,
                    &mut index,
                    "--maximum-publication-recombination-error",
                )?;
            }
            "--maximum-target-peak" => {
                config.maximum_target_peak =
                    parse_value(args, &mut index, "--maximum-target-peak")?;
            }
            "--maximum-residual-peak" => {
                config.maximum_residual_peak =
                    parse_value(args, &mut index, "--maximum-residual-peak")?;
            }
            "--maximum-energy-gain-db" => {
                config.maximum_energy_gain_db =
                    parse_value(args, &mut index, "--maximum-energy-gain-db")?;
            }
            "--maximum-stereo-correlation-delta" => {
                config.maximum_stereo_correlation_delta =
                    parse_value(args, &mut index, "--maximum-stereo-correlation-delta")?;
            }
            "--maximum-mid-side-ratio-delta-db" => {
                config.maximum_mid_side_energy_ratio_delta_db =
                    parse_value(args, &mut index, "--maximum-mid-side-ratio-delta-db")?;
            }
            "--accelerator" if !accelerator_seen => {
                accelerator_seen = true;
                let value: String = parse_value(args, &mut index, "--accelerator")?;
                accelerator = AcceleratorPreference::parse(&value).ok_or_else(|| {
                    format!(
                        "unknown causal target-sound accelerator: {value} (expected cpu, auto, gpu, metal, or cuda)"
                    )
                })?;
            }
            "--accelerator" => return Err("--accelerator may be supplied only once".into()),
            "--max-memory" if max_memory_mb.is_none() => {
                max_memory_mb = Some(parse_value(args, &mut index, "--max-memory")?);
            }
            "--max-memory" => return Err("--max-memory may be supplied only once".into()),
            "--replace" if commit_mode == CommitMode::NoClobber => {
                commit_mode = CommitMode::Replace;
            }
            "--replace" => return Err("--replace may be supplied only once".into()),
            "--json" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::Json;
            }
            "--pretty" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err("causal target-sound accepts only one of --json or --pretty".into());
            }
            "-h" | "--help" => return Err("causal target-sound help requested".into()),
            "-" => {
                return Err(
                    "causal target-sound requires regular-file paths; stdin/stdout are unsupported"
                        .into(),
                );
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown causal target-sound option: {value}"));
            }
            value => {
                if !positional.is_empty() {
                    return Err(format!(
                        "unexpected extra causal target-sound argument: {value}"
                    ));
                }
                positional.push(value.to_string());
            }
        }
        index += 1;
    }
    let input = positional
        .first()
        .cloned()
        .ok_or("causal target-sound requires INPUT")?;
    let query = query.ok_or("causal target-sound requires --query")?;
    let target = target.ok_or("causal target-sound requires --target")?;
    let residual = residual.ok_or("causal target-sound requires --residual")?;
    let output = output.ok_or("causal target-sound requires --output")?;
    let report = report.ok_or("causal target-sound requires --report")?;
    config.mode = mode.ok_or("causal target-sound requires --mode")?;
    for (path, context) in [
        (&target, "target"),
        (&residual, "residual"),
        (&output, "selected output"),
    ] {
        if OutputFormat::from_path(std::path::Path::new(path))? != OutputFormat::Wav {
            return Err(format!(
                "causal target-sound {context} must be WAV to avoid lossy encoding"
            ));
        }
    }
    if std::path::Path::new(&report)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        != Some("json")
    {
        return Err("causal target-sound report must use a .json extension".into());
    }
    let package = package.ok_or("causal target-sound requires --model-package")?;
    let package_key = package_key.ok_or("causal target-sound requires --model-package-key")?;
    let offline_evidence =
        offline_evidence.ok_or("causal target-sound requires --offline-promotion-evidence")?;
    let offline_evidence_key = offline_evidence_key
        .ok_or("causal target-sound requires --offline-promotion-evidence-key")?;
    let causal_evidence =
        causal_evidence.ok_or("causal target-sound requires --causal-promotion-evidence")?;
    let causal_evidence_key = causal_evidence_key
        .ok_or("causal target-sound requires --causal-promotion-evidence-key")?;
    checked_mib_limit_bytes(max_memory_mb, "--max-memory")?;
    config.validate()?;
    Ok(CausalTargetSoundCliOptions {
        input,
        query,
        target,
        residual,
        output,
        report,
        package,
        package_key,
        offline_evidence,
        offline_evidence_key,
        causal_evidence,
        causal_evidence_key,
        config,
        accelerator,
        max_memory_mb,
        commit_mode,
        print_mode,
    })
}

fn run_causal_target_sound(args: &[String]) -> Result<(), String> {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help") {
        print!("{}", target_sound_usage());
        return Ok(());
    }
    #[cfg(feature = "onnx")]
    {
        run_causal_target_sound_with_onnx(args)
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = args;
        Err("causal target-sound extraction requires a build with the onnx feature".into())
    }
}

#[cfg(feature = "onnx")]
fn run_causal_target_sound_with_onnx(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) == Some("evidence") {
        return run_causal_target_sound_evidence(&args[1..]);
    }
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        return Err("causal target-sound --help accepts no other arguments".into());
    }
    let options = parse_causal_target_sound_args(args)?;
    validate_causal_target_sound_publication_paths(&options)?;
    run_causal_target_sound_audio(options)
}

#[cfg(feature = "onnx")]
fn run_causal_target_sound_evidence(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) != Some("verify") {
        return Err(
            "causal target-sound evidence requires: verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]"
                .into(),
        );
    }
    let mut positional = Vec::new();
    let mut mode = DiagnosticPrintMode::Human;
    for argument in &args[1..] {
        match argument.as_str() {
            "--json" if mode == DiagnosticPrintMode::Human => mode = DiagnosticPrintMode::Json,
            "--pretty" if mode == DiagnosticPrintMode::Human => {
                mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err("causal target-sound evidence accepts one output mode".into());
            }
            value if value.starts_with('-') => {
                return Err(format!(
                    "unknown causal target-sound evidence option: {value}"
                ));
            }
            value => positional.push(value.to_string()),
        }
    }
    if positional.len() != 2 {
        return Err(
            "causal target-sound evidence verify requires EVIDENCE.json and PUBLIC-KEY.json".into(),
        );
    }
    let evidence = denoize::SignedCausalTargetSoundPromotionEvidence::from_file(&positional[0])?;
    let key = ReceiptPublicKey::from_file(&positional[1])?;
    evidence.verify_signature(&key)?;
    match mode {
        DiagnosticPrintMode::Json => println!(
            "{}",
            serde_json::to_string(&evidence)
                .map_err(|error| format!("serialize causal target-sound evidence: {error}"))?
        ),
        DiagnosticPrintMode::PrettyJson => println!("{}", evidence.to_pretty_json()?),
        DiagnosticPrintMode::Human => println!(
            "verified causal target-sound evidence: offline_package={} causal_package={} catalog={} classes={} strata={} devices={} worst_latency_ms={} accepted={}",
            evidence.payload.offline_model_package_sha256,
            evidence.payload.causal_model_package_sha256,
            evidence.payload.query_catalog_sha256,
            evidence.payload.query_class_count,
            evidence.payload.strata.len(),
            evidence.payload.device_measurements.len(),
            evidence.payload.worst_effective_latency_milliseconds,
            evidence.payload.accepted,
        ),
    }
    if !evidence.payload.accepted {
        return Err("causal target-sound evidence is authentic but fails promotion gates".into());
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn validate_causal_target_sound_publication_paths(
    options: &CausalTargetSoundCliOptions,
) -> Result<(), String> {
    let mut sources = Vec::new();
    for (path, context) in [
        (options.input.as_str(), "causal target-sound input"),
        (options.query.as_str(), "causal target-sound query"),
        (options.package.as_str(), "causal target-sound package"),
        (
            options.package_key.as_str(),
            "causal target-sound package key",
        ),
        (
            options.offline_evidence.as_str(),
            "causal target-sound offline evidence",
        ),
        (
            options.offline_evidence_key.as_str(),
            "causal target-sound offline evidence key",
        ),
        (
            options.causal_evidence.as_str(),
            "causal target-sound causal evidence",
        ),
        (
            options.causal_evidence_key.as_str(),
            "causal target-sound causal evidence key",
        ),
    ] {
        sources.push((
            std::fs::canonicalize(path)
                .map_err(|error| format!("resolve {context} {path}: {error}"))?,
            context,
        ));
    }
    for left in 0..sources.len() {
        for right in left + 1..sources.len() {
            if sources[left].0 == sources[right].0 {
                return Err(format!(
                    "{} and {} must use distinct source files",
                    sources[left].1, sources[right].1
                ));
            }
        }
    }
    let mut destinations = Vec::new();
    for (path, context) in [
        (options.target.as_str(), "causal target-sound target"),
        (options.residual.as_str(), "causal target-sound residual"),
        (
            options.output.as_str(),
            "causal target-sound selected output",
        ),
        (options.report.as_str(), "causal target-sound report"),
    ] {
        let normalized = normalized_project_destination(std::path::Path::new(path), context)?;
        let existing = std::fs::canonicalize(&normalized).ok();
        if sources.iter().any(|(source, _)| {
            normalized == *source || existing.as_ref().is_some_and(|path| path == source)
        }) {
            return Err(format!(
                "{context} must not replace an input, query, package, key, or evidence document"
            ));
        }
        ensure_restoration_destination_available(&normalized, options.commit_mode)?;
        destinations.push((batch_collision_key(&normalized), context));
    }
    destinations.sort_by(|left, right| left.0.cmp(&right.0));
    if let Some(pair) = destinations.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        return Err(format!(
            "{} and {} must use distinct destinations",
            pair[0].1, pair[1].1
        ));
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn run_causal_target_sound_audio(options: CausalTargetSoundCliOptions) -> Result<(), String> {
    let maximum = checked_mib_limit_bytes(options.max_memory_mb, "--max-memory")?;
    let query = denoize::TargetSoundQuery::from_file(&options.query)?;
    let offline_evidence =
        denoize::SignedTargetSoundPromotionEvidence::from_file(&options.offline_evidence)?;
    let offline_key = ReceiptPublicKey::from_file(&options.offline_evidence_key)?;
    let causal_evidence =
        denoize::SignedCausalTargetSoundPromotionEvidence::from_file(&options.causal_evidence)?;
    let causal_key = ReceiptPublicKey::from_file(&options.causal_evidence_key)?;
    let package = RuntimeModelPackage::open(&options.package, &options.package_key)?;
    // Authenticate both evidence layers, catalog/config binding, graph, states,
    // numerical sequences, licenses, and resource contract before user audio.
    let session = denoize::CausalTargetSoundSession::prepare(
        package,
        &offline_evidence,
        &offline_key,
        &causal_evidence,
        &causal_key,
        &query,
        &options.config,
        options.accelerator,
    )?;
    let model_working_set = session.model_working_set_bytes()?;
    ensure_memory_limit(
        model_working_set,
        options.max_memory_mb,
        "causal target-sound model working set",
    )?;
    let mut input_session = AudioInputSession::open(&options.input)?;
    let session_memory = estimate_session_memory_bytes(&input_session);
    ensure_memory_limit(
        model_working_set.saturating_add(session_memory),
        options.max_memory_mb,
        "causal target-sound input/model preflight",
    )?;
    let decode_maximum = maximum.map(|limit| {
        limit
            .saturating_sub(model_working_set)
            .saturating_sub(session_memory)
    });
    let input = read_audio_from_session_with_limits(
        &mut input_session,
        DecodeLimits::new(
            metadata_limits_for_available_bytes(decode_maximum),
            decode_maximum,
        ),
    )?;
    let working_set = denoize::estimate_causal_target_sound_memory_bytes(
        &input,
        session.sample_rate_hz(),
        session.channels(),
        session.frame_samples(),
        session.flush_samples(),
    )?
    .saturating_add(model_working_set)
    .saturating_add(session_memory);
    ensure_memory_limit(
        working_set,
        options.max_memory_mb,
        "causal target-sound decoded/model/target/residual working set",
    )?;
    let result = session.render(&input)?;
    let encode_options = EncodeOptions::default();
    OutputFormat::Wav.validate_config(&result.target, &encode_options)?;
    OutputFormat::Wav.validate_config(&result.residual, &encode_options)?;
    OutputFormat::Wav.validate_config(&result.output, &encode_options)?;

    // Finish all fallible encoding before any artifact becomes visible.
    let mut staged_target = AtomicOutput::new(&options.target)?;
    denoize::encode::write_audio_to_file(
        staged_target.file_mut(),
        OutputFormat::Wav,
        &result.target,
        encode_options,
    )?;
    let mut staged_residual = AtomicOutput::new(&options.residual)?;
    denoize::encode::write_audio_to_file(
        staged_residual.file_mut(),
        OutputFormat::Wav,
        &result.residual,
        encode_options,
    )?;
    let mut staged_output = AtomicOutput::new(&options.output)?;
    denoize::encode::write_audio_to_file(
        staged_output.file_mut(),
        OutputFormat::Wav,
        &result.output,
        encode_options,
    )?;
    let staged_report = stage_restoration_json(&options.report, &result.report)?;
    // Selected output remains last, so it cannot be visible without its audit
    // report and both halves of the decomposition.
    staged_report.commit(options.commit_mode)?;
    staged_target.commit(options.commit_mode)?;
    staged_residual.commit(options.commit_mode)?;
    staged_output.commit(options.commit_mode)?;

    match options.print_mode {
        DiagnosticPrintMode::Json => println!("{}", result.report.to_json()?),
        DiagnosticPrintMode::PrettyJson => println!("{}", result.report.to_pretty_json()?),
        DiagnosticPrintMode::Human => {
            let fallback_blocks = result.report.decision_counts.fallback_blocks();
            println!(
                "causal target-sound extraction: mode={:?} class={} published_blocks={} fallback_blocks={} transitions={} frames={} latency_samples={} package={}",
                result.report.mode,
                result.report.query.class_id,
                result.report.decision_counts.published_present_blocks,
                fallback_blocks,
                result.report.presence_transitions,
                result.report.source_frames,
                result.report.algorithmic_latency_samples,
                result.report.model.package_sha256,
            );
            for warning in &result.report.warnings {
                println!("warning: {warning}");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod target_sound_cli_tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    fn valid_arguments() -> Vec<String> {
        arguments(&[
            "program.wav",
            "--query",
            "query.json",
            "--target",
            "target.wav",
            "--residual",
            "residual.wav",
            "--output",
            "output.wav",
            "--report",
            "report.json",
            "--mode",
            "preserve",
            "--model-package",
            "model.dmp",
            "--model-package-key",
            "model.pub",
            "--promotion-evidence",
            "evidence.json",
            "--promotion-evidence-key",
            "evidence.pub.json",
        ])
    }

    #[cfg(feature = "onnx")]
    fn valid_causal_arguments() -> Vec<String> {
        arguments(&[
            "program.wav",
            "--query",
            "query.json",
            "--target",
            "target.wav",
            "--residual",
            "residual.wav",
            "--output",
            "output.wav",
            "--report",
            "report.json",
            "--mode",
            "remove",
            "--model-package",
            "causal.dmp",
            "--model-package-key",
            "model.pub",
            "--offline-promotion-evidence",
            "offline.json",
            "--offline-promotion-evidence-key",
            "offline.pub.json",
            "--causal-promotion-evidence",
            "causal.json",
            "--causal-promotion-evidence-key",
            "causal.pub.json",
        ])
    }

    #[test]
    fn parser_requires_query_all_outputs_and_explicit_mode() {
        let error = parse_target_sound_args(&arguments(&["program.wav"])).unwrap_err();
        assert_eq!(error, "target-sound requires --query");
        let parsed = parse_target_sound_args(&valid_arguments()).unwrap();
        assert_eq!(parsed.config.mode, denoize::TargetSoundMode::Preserve);
        assert_eq!(parsed.config.minimum_present_probability, 0.9);
        assert_eq!(parsed.commit_mode, CommitMode::NoClobber);
    }

    #[test]
    fn parser_rejects_lossy_artifacts_and_duplicate_thresholds() {
        let mut lossy = valid_arguments();
        let residual = lossy
            .iter()
            .position(|argument| argument == "residual.wav")
            .unwrap();
        lossy[residual] = "residual.mp3".into();
        assert!(parse_target_sound_args(&lossy).is_err());

        let mut duplicate = valid_arguments();
        duplicate.extend([
            "--minimum-present-probability".into(),
            "0.9".into(),
            "--minimum-present-probability".into(),
            "0.95".into(),
        ]);
        assert!(parse_target_sound_args(&duplicate).is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn causal_parser_requires_both_evidence_layers_and_complete_outputs() {
        assert_eq!(
            parse_causal_target_sound_args(&arguments(&["program.wav"])).unwrap_err(),
            "causal target-sound requires --query"
        );
        let parsed = parse_causal_target_sound_args(&valid_causal_arguments()).unwrap();
        assert_eq!(parsed.config.mode, denoize::TargetSoundMode::Remove);
        assert_eq!(parsed.config.present_hold_blocks, 3);
        assert_eq!(
            parsed.config.maximum_publication_recombination_error,
            1.0e-6
        );
        assert_eq!(parsed.commit_mode, CommitMode::NoClobber);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn causal_parser_rejects_duplicate_hold_and_lossy_residual() {
        let mut duplicate = valid_causal_arguments();
        duplicate.extend([
            "--present-hold-blocks".into(),
            "2".into(),
            "--present-hold-blocks".into(),
            "3".into(),
        ]);
        assert!(parse_causal_target_sound_args(&duplicate).is_err());

        let mut lossy = valid_causal_arguments();
        let residual = lossy
            .iter()
            .position(|argument| argument == "residual.wav")
            .unwrap();
        lossy[residual] = "residual.mp3".into();
        assert!(parse_causal_target_sound_args(&lossy).is_err());
    }
}

fn microphone_array_usage() -> &'static str {
    "\
USAGE:
    denoize array <MICROPHONE_ARRAY> <OUTPUT> --array-config <CONFIG.json> --promotion-evidence <EVIDENCE.json> --promotion-evidence-key <PUBLIC-KEY.json> [OPTIONS]
    denoize array evidence verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]

Enhance a declared two-to-four-channel microphone array to a mono reference
image. The input is accepted only when the closed configuration explicitly
binds every channel ID, right-handed x-forward/y-left/z-up coordinate,
sample-skew and gain/phase calibration, and the reference microphone. Ordinary
program stereo or surround is never inferred to be an array. Authenticated
promotion evidence must bind the exact WPE plus conditioned mask-MVDR
configuration before the input audio is opened.

OPTIONS:
        --array-config <PATH.json>         required closed geometry and DSP configuration
        --promotion-evidence <PATH>        accepted signed array evaluation evidence
        --promotion-evidence-key <PATH>    trusted Ed25519 evidence public key
        --report <PATH.json>               atomically write the closed path-free report
        --max-memory <MB>                  bound decode, WPE, STFT, covariance, and output memory
        --no-metadata                      do not copy input metadata to output
        --replace                          atomically replace output/report destinations
        --json                             emit compact report JSON
        --pretty                           emit indented report JSON
    -h, --help                             show this help
"
}

#[derive(Debug)]
struct MicrophoneArrayCliOptions {
    input: String,
    output: String,
    config_path: String,
    promotion_evidence: String,
    promotion_evidence_key: String,
    report: Option<String>,
    max_memory_mb: Option<usize>,
    preserve_metadata: bool,
    commit_mode: CommitMode,
    print_mode: DiagnosticPrintMode,
}

fn parse_microphone_array_args(args: &[String]) -> Result<MicrophoneArrayCliOptions, String> {
    let mut positional = Vec::new();
    let mut config_path = None;
    let mut promotion_evidence = None;
    let mut promotion_evidence_key = None;
    let mut report = None;
    let mut max_memory_mb = None;
    let mut preserve_metadata = true;
    let mut commit_mode = CommitMode::NoClobber;
    let mut print_mode = DiagnosticPrintMode::Human;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--array-config" if config_path.is_none() => {
                config_path = Some(parse_value(args, &mut index, "--array-config")?);
            }
            "--array-config" => return Err("--array-config may be supplied only once".into()),
            "--promotion-evidence" if promotion_evidence.is_none() => {
                promotion_evidence = Some(parse_value(args, &mut index, "--promotion-evidence")?);
            }
            "--promotion-evidence" => {
                return Err("--promotion-evidence may be supplied only once".into());
            }
            "--promotion-evidence-key" if promotion_evidence_key.is_none() => {
                promotion_evidence_key =
                    Some(parse_value(args, &mut index, "--promotion-evidence-key")?);
            }
            "--promotion-evidence-key" => {
                return Err("--promotion-evidence-key may be supplied only once".into());
            }
            "--report" if report.is_none() => {
                report = Some(parse_value(args, &mut index, "--report")?);
            }
            "--report" => return Err("--report may be supplied only once".into()),
            "--max-memory" if max_memory_mb.is_none() => {
                max_memory_mb = Some(parse_value(args, &mut index, "--max-memory")?);
            }
            "--max-memory" => return Err("--max-memory may be supplied only once".into()),
            "--no-metadata" if preserve_metadata => preserve_metadata = false,
            "--no-metadata" => return Err("--no-metadata may be supplied only once".into()),
            "--replace" if commit_mode == CommitMode::NoClobber => {
                commit_mode = CommitMode::Replace;
            }
            "--replace" => return Err("--replace may be supplied only once".into()),
            "--json" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::Json;
            }
            "--pretty" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err(
                    "microphone-array enhancement accepts only one of --json or --pretty".into(),
                );
            }
            "-h" | "--help" => return Err("microphone-array help requested".into()),
            "-" => {
                return Err(
                    "microphone-array enhancement requires regular-file paths; stdin/stdout are unsupported"
                        .into(),
                );
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown microphone-array option: {value}"));
            }
            value => {
                if positional.len() == 2 {
                    return Err(format!(
                        "unexpected extra microphone-array argument: {value}"
                    ));
                }
                positional.push(value.to_string());
            }
        }
        index += 1;
    }
    let input = positional
        .first()
        .cloned()
        .ok_or("microphone-array enhancement requires MICROPHONE_ARRAY")?;
    let output = positional
        .get(1)
        .cloned()
        .ok_or("microphone-array enhancement requires OUTPUT")?;
    let config_path = config_path.ok_or("microphone-array enhancement requires --array-config")?;
    let promotion_evidence =
        promotion_evidence.ok_or("microphone-array enhancement requires --promotion-evidence")?;
    let promotion_evidence_key = promotion_evidence_key
        .ok_or("microphone-array enhancement requires --promotion-evidence-key")?;
    checked_mib_limit_bytes(max_memory_mb, "--max-memory")?;
    Ok(MicrophoneArrayCliOptions {
        input,
        output,
        config_path,
        promotion_evidence,
        promotion_evidence_key,
        report,
        max_memory_mb,
        preserve_metadata,
        commit_mode,
        print_mode,
    })
}

fn run_microphone_array(args: &[String]) -> Result<(), String> {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help") {
        print!("{}", microphone_array_usage());
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("evidence") {
        return run_microphone_array_evidence(&args[1..]);
    }
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        return Err("microphone-array --help accepts no other arguments".into());
    }
    let options = parse_microphone_array_args(args)?;
    validate_microphone_array_publication_paths(&options)?;
    run_microphone_array_audio(options)
}

fn run_microphone_array_evidence(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) != Some("verify") {
        return Err(
            "microphone-array evidence requires: verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]"
                .into(),
        );
    }
    let mut positional = Vec::new();
    let mut mode = DiagnosticPrintMode::Human;
    for argument in &args[1..] {
        match argument.as_str() {
            "--json" if mode == DiagnosticPrintMode::Human => mode = DiagnosticPrintMode::Json,
            "--pretty" if mode == DiagnosticPrintMode::Human => {
                mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err("microphone-array evidence verify accepts only one output mode".into());
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown microphone-array evidence option: {value}"));
            }
            value => positional.push(value.to_string()),
        }
    }
    if positional.len() != 2 {
        return Err(
            "microphone-array evidence verify requires EVIDENCE.json and PUBLIC-KEY.json".into(),
        );
    }
    let evidence = denoize::SignedMicrophoneArrayPromotionEvidence::from_file(&positional[0])?;
    let key = ReceiptPublicKey::from_file(&positional[1])?;
    evidence.verify_signature(&key)?;
    match mode {
        DiagnosticPrintMode::Json => println!(
            "{}",
            serde_json::to_string(&evidence).map_err(|error| {
                format!("serialize microphone-array promotion evidence: {error}")
            })?
        ),
        DiagnosticPrintMode::PrettyJson => println!("{}", evidence.to_pretty_json()?),
        DiagnosticPrintMode::Human => println!(
            "verified microphone-array promotion evidence: implementation={}, strata={}, real_meetings={}, unseen_geometries={}, permutations={}, accepted={}",
            evidence.payload.implementation,
            evidence.payload.strata.len(),
            evidence.payload.real_meeting_cases,
            evidence.payload.unseen_geometry_cases,
            evidence.payload.permutation_cases,
            evidence.payload.accepted,
        ),
    }
    if !evidence.payload.accepted {
        return Err(
            "microphone-array promotion evidence is authentic but does not pass promotion gates"
                .into(),
        );
    }
    Ok(())
}

fn validate_microphone_array_publication_paths(
    options: &MicrophoneArrayCliOptions,
) -> Result<(), String> {
    let mut sources = Vec::new();
    for (path, context) in [
        (options.input.as_str(), "microphone-array input"),
        (
            options.config_path.as_str(),
            "microphone-array configuration",
        ),
        (
            options.promotion_evidence.as_str(),
            "microphone-array promotion evidence",
        ),
        (
            options.promotion_evidence_key.as_str(),
            "microphone-array promotion evidence key",
        ),
    ] {
        sources.push((
            std::fs::canonicalize(path)
                .map_err(|error| format!("resolve {context} {path}: {error}"))?,
            context,
        ));
    }
    for left in 0..sources.len() {
        for right in left + 1..sources.len() {
            if sources[left].0 == sources[right].0 {
                return Err(format!(
                    "{} and {} must use distinct source files",
                    sources[left].1, sources[right].1
                ));
            }
        }
    }
    let mut destinations = Vec::new();
    for (path, context) in [
        (
            Some(options.output.as_str()),
            "microphone-array audio output",
        ),
        (options.report.as_deref(), "microphone-array report"),
    ] {
        let Some(path) = path else {
            continue;
        };
        let normalized = normalized_project_destination(std::path::Path::new(path), context)?;
        let existing = std::fs::canonicalize(&normalized).ok();
        if sources.iter().any(|(source, _)| {
            normalized == *source || existing.as_ref().is_some_and(|path| path == source)
        }) {
            return Err(format!(
                "{context} must not replace an input, configuration, evidence, or key"
            ));
        }
        ensure_restoration_destination_available(&normalized, options.commit_mode)?;
        destinations.push((batch_collision_key(&normalized), context));
    }
    destinations.sort_by(|left, right| left.0.cmp(&right.0));
    if let Some(pair) = destinations.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        return Err(format!(
            "{} and {} must use distinct destinations",
            pair[0].1, pair[1].1
        ));
    }
    Ok(())
}

fn run_microphone_array_audio(options: MicrophoneArrayCliOptions) -> Result<(), String> {
    let maximum = checked_mib_limit_bytes(options.max_memory_mb, "--max-memory")?;
    let config = denoize::MicrophoneArrayConfig::from_file(&options.config_path)?;
    let evidence =
        denoize::SignedMicrophoneArrayPromotionEvidence::from_file(&options.promotion_evidence)?;
    let evidence_key = ReceiptPublicKey::from_file(&options.promotion_evidence_key)?;
    // Authenticate the exact geometry and DSP configuration before opening
    // user-controlled audio.
    let session =
        denoize::MicrophoneArraySession::prepare(&evidence, &evidence_key, config.clone())?;
    let fixed_memory = denoize::estimate_microphone_array_memory_bytes(&config, 0)?;
    ensure_memory_limit(
        fixed_memory,
        options.max_memory_mb,
        "microphone-array fixed working set",
    )?;

    let mut input_session = AudioInputSession::open(&options.input)?;
    let session_memory = estimate_session_memory_bytes(&input_session);
    ensure_memory_limit(
        fixed_memory.saturating_add(session_memory),
        options.max_memory_mb,
        "microphone-array input preflight",
    )?;
    let decode_maximum = maximum.map(|limit| {
        limit
            .saturating_sub(fixed_memory)
            .saturating_sub(session_memory)
    });
    let input = read_audio_from_session_with_limits(
        &mut input_session,
        DecodeLimits::new(
            metadata_limits_for_available_bytes(decode_maximum),
            decode_maximum,
        ),
    )?;
    let array_memory =
        denoize::estimate_microphone_array_memory_bytes(session.config(), input.frames())?;
    let working_set = array_memory.saturating_add(session_memory);
    ensure_memory_limit(
        working_set,
        options.max_memory_mb,
        "microphone-array decoded/WPE/MVDR/output working set",
    )?;
    let result = session.enhance(&input)?;
    let mut staged_report = options
        .report
        .as_deref()
        .map(|path| stage_restoration_json(path, &result.report))
        .transpose()?;
    let format = OutputFormat::from_path(std::path::Path::new(&options.output))?;
    let encode_options = EncodeOptions::default();
    encode_options.validate_options(format)?;
    format.validate_config(&result.audio, &encode_options)?;
    let metadata = if options.preserve_metadata {
        input_session.read_metadata_with_limits(retained_metadata_limits(
            options.max_memory_mb,
            working_set,
        )?)?
    } else {
        None
    };
    denoize::write_audio_transactional(
        &options.output,
        &result.audio,
        encode_options,
        metadata,
        options.commit_mode,
    )?;
    if let Some(report) = staged_report.take() {
        report.commit(options.commit_mode)?;
    }
    match options.print_mode {
        DiagnosticPrintMode::Json => println!("{}", result.report.to_json()?),
        DiagnosticPrintMode::PrettyJson => println!("{}", result.report.to_pretty_json()?),
        DiagnosticPrintMode::Human => println!(
            "microphone array: input_channels={} active={} solved_bins={} fallback_bins={} frames={} latency_ms={:.3}",
            result.report.input_channels,
            result.report.active_microphones,
            result.report.solved_frequency_bins,
            result.report.fallback_frequency_bins,
            result.report.output_frames,
            result.report.algorithmic_latency_milliseconds,
        ),
    }
    Ok(())
}

fn aec_usage() -> &'static str {
    "\
USAGE:
    denoize aec <MICROPHONE> <FAR_END_REFERENCE> <OUTPUT> --promotion-evidence <EVIDENCE.json> --promotion-evidence-key <PUBLIC-KEY.json> [OPTIONS]
    denoize aec evidence verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]

Cancel a typed mono far-end reference from a mono microphone recording. The
safe native path uses explicit signed delay (including negative delay), an
explicit reference clock mapping, partitioned frequency-domain NLMS, frozen
adaptation during double talk, and conservative residual suppression. The
exact configuration must be accepted by separately signed promotion evidence.
A missing or low-confidence reference preserves the microphone; it is never
treated as evidence that near-end speech should be suppressed.

OPTIONS:
        --promotion-evidence <PATH>        accepted signed AEC evaluation evidence
        --promotion-evidence-key <PATH>    trusted Ed25519 evidence public key
        --aec-config <PATH.json>           closed AEC configuration (default: promoted 48 kHz baseline)
        --reference-clock-ppm <F>          explicit reference clock offset, -2000..2000 (default: 0)
        --initial-delay-samples <N>        signed alignment hint within the promoted search range
        --route-generation <N>             non-negative route identity, reset on every change
        --report <PATH.json>               atomically write the closed path-free report
        --max-memory <MB>                  bound decode, alignment, FFT, and output memory
        --no-metadata                      do not copy microphone metadata to output
        --replace                          atomically replace output/report destinations
        --json                             emit compact report JSON
        --pretty                           emit indented report JSON
    -h, --help                             show this help
"
}

#[cfg(feature = "aec")]
#[derive(Debug)]
struct AecCliOptions {
    microphone: String,
    reference: String,
    output: String,
    promotion_evidence: String,
    promotion_evidence_key: String,
    config_path: Option<String>,
    report: Option<String>,
    reference_clock_ppm: f64,
    initial_delay_samples: i32,
    route_generation: u64,
    max_memory_mb: Option<usize>,
    preserve_metadata: bool,
    commit_mode: CommitMode,
    print_mode: DiagnosticPrintMode,
}

#[cfg(feature = "aec")]
fn parse_aec_args(args: &[String]) -> Result<AecCliOptions, String> {
    let mut positional = Vec::new();
    let mut promotion_evidence = None;
    let mut promotion_evidence_key = None;
    let mut config_path = None;
    let mut report = None;
    let mut reference_clock_ppm = 0.0_f64;
    let mut initial_delay_samples = 0_i32;
    let mut route_generation = 0_u64;
    let mut max_memory_mb = None;
    let mut preserve_metadata = true;
    let mut commit_mode = CommitMode::NoClobber;
    let mut print_mode = DiagnosticPrintMode::Human;
    let mut scalar_options = std::collections::HashSet::new();
    let mut index = 0usize;
    while index < args.len() {
        let argument = args[index].as_str();
        if matches!(
            argument,
            "--reference-clock-ppm" | "--initial-delay-samples" | "--route-generation"
        ) && !scalar_options.insert(argument)
        {
            return Err(format!("{argument} may be supplied only once"));
        }
        match argument {
            "--promotion-evidence" if promotion_evidence.is_none() => {
                promotion_evidence = Some(parse_value(args, &mut index, "--promotion-evidence")?);
            }
            "--promotion-evidence" => {
                return Err("--promotion-evidence may be supplied only once".into());
            }
            "--promotion-evidence-key" if promotion_evidence_key.is_none() => {
                promotion_evidence_key =
                    Some(parse_value(args, &mut index, "--promotion-evidence-key")?);
            }
            "--promotion-evidence-key" => {
                return Err("--promotion-evidence-key may be supplied only once".into());
            }
            "--aec-config" if config_path.is_none() => {
                config_path = Some(parse_value(args, &mut index, "--aec-config")?);
            }
            "--aec-config" => return Err("--aec-config may be supplied only once".into()),
            "--reference-clock-ppm" => {
                reference_clock_ppm = parse_value(args, &mut index, "--reference-clock-ppm")?;
            }
            "--initial-delay-samples" => {
                initial_delay_samples = parse_value(args, &mut index, "--initial-delay-samples")?;
            }
            "--route-generation" => {
                route_generation = parse_value(args, &mut index, "--route-generation")?;
            }
            "--report" if report.is_none() => {
                report = Some(parse_value(args, &mut index, "--report")?);
            }
            "--report" => return Err("--report may be supplied only once".into()),
            "--max-memory" if max_memory_mb.is_none() => {
                max_memory_mb = Some(parse_value(args, &mut index, "--max-memory")?);
            }
            "--max-memory" => return Err("--max-memory may be supplied only once".into()),
            "--no-metadata" if preserve_metadata => preserve_metadata = false,
            "--no-metadata" => return Err("--no-metadata may be supplied only once".into()),
            "--replace" if commit_mode == CommitMode::NoClobber => {
                commit_mode = CommitMode::Replace;
            }
            "--replace" => return Err("--replace may be supplied only once".into()),
            "--json" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::Json;
            }
            "--pretty" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err("AEC accepts only one of --json or --pretty".into());
            }
            "-h" | "--help" => return Err("AEC help requested".into()),
            "-" => {
                return Err("AEC requires regular-file paths; stdin/stdout are unsupported".into());
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown AEC option: {value}"));
            }
            value => {
                if positional.len() == 3 {
                    return Err(format!("unexpected extra AEC argument: {value}"));
                }
                positional.push(value.to_string());
            }
        }
        index += 1;
    }
    let microphone = positional
        .first()
        .cloned()
        .ok_or("AEC requires MICROPHONE")?;
    let reference = positional
        .get(1)
        .cloned()
        .ok_or("AEC requires FAR_END_REFERENCE")?;
    let output = positional.get(2).cloned().ok_or("AEC requires OUTPUT")?;
    let promotion_evidence = promotion_evidence.ok_or("AEC requires --promotion-evidence")?;
    let promotion_evidence_key =
        promotion_evidence_key.ok_or("AEC requires --promotion-evidence-key")?;
    if !reference_clock_ppm.is_finite() || !(-2_000.0..=2_000.0).contains(&reference_clock_ppm) {
        return Err("AEC --reference-clock-ppm must be finite and in -2000..=2000".into());
    }
    if route_generation > (1_u64 << 53) - 1 {
        return Err("AEC --route-generation exceeds the JSON safe-integer limit".into());
    }
    checked_mib_limit_bytes(max_memory_mb, "--max-memory")?;
    Ok(AecCliOptions {
        microphone,
        reference,
        output,
        promotion_evidence,
        promotion_evidence_key,
        config_path,
        report,
        reference_clock_ppm,
        initial_delay_samples,
        route_generation,
        max_memory_mb,
        preserve_metadata,
        commit_mode,
        print_mode,
    })
}

fn target_speaker_usage() -> &'static str {
    "\
USAGE:
    denoize target-speaker <MIXTURE> <ENROLLMENT> <OUTPUT> --model-package <PACKAGE.dmp> --model-package-key <KEY> --promotion-evidence <EVIDENCE.json> --promotion-evidence-key <PUBLIC-KEY.json> [OPTIONS]
    denoize target-speaker evidence verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]
    denoize target-speaker causal <MIXTURE> <ENROLLMENT> <OUTPUT> --model-package <PACKAGE.dmp> --model-package-key <KEY> --offline-promotion-evidence <EVIDENCE.json> --offline-promotion-evidence-key <PUBLIC-KEY.json> --causal-promotion-evidence <EVIDENCE.json> --causal-promotion-evidence-key <PUBLIC-KEY.json> [OPTIONS]
    denoize target-speaker causal evidence verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]

Run offline target-speaker extraction through a signed package v2 graph with
mixture and enrollment inputs, extracted-audio output, and calibrated
absent/uncertain/present probabilities. The exact package must also have
accepted, signed promotion evidence covering REAL-T, TS-SUPERB, target absence,
similar voices, enrollment mismatch, ASR, identity, leakage, and listening
gates. Audio is published only for a confidently present target whose candidate
passes every runtime gate. Absent, uncertain, and unsafe candidates publish no
audio; they never fall back to the mixture or an unverified voice.

OPTIONS:
        --model-package <PATH>             required signed runtime package v2
        --model-package-key <PATH>         trusted Minisign public key
        --promotion-evidence <PATH>        accepted signed evaluation evidence
        --promotion-evidence-key <PATH>    trusted Ed25519 evidence public key
        --minimum-present-probability <F>  present threshold, 0.5..1 (default: 0.9)
        --minimum-absent-probability <F>   absent threshold, 0.5..1 (default: 0.9)
        --maximum-energy-gain-db <DB>      candidate energy-rise ceiling, 0..12 (default: 3)
        --maximum-peak-gain-db <DB>        candidate peak-rise ceiling, 0..12 (default: 3)
        --maximum-new-clipping-ratio <F>   added clipping ceiling, 0..0.01 (default: 0.0001)
        --accelerator <NAME>               cpu|auto|gpu|metal|cuda (deterministic v1 uses CPU)
        --report <PATH.json>               atomically write the closed path-free report
        --max-memory <MB>                  bound decode, model, enrollment, and candidate memory
        --no-metadata                      do not copy mixture metadata to accepted output
        --replace                          atomically replace output/report destinations
        --json                             emit compact report JSON
        --pretty                           emit indented report JSON
    -h, --help                             show this help

CAUSAL OPTIONS:
        --offline-promotion-evidence <PATH>      accepted signed offline evidence
        --offline-promotion-evidence-key <PATH> trusted offline evidence public key
        --causal-promotion-evidence <PATH>       accepted signed causal evidence
        --causal-promotion-evidence-key <PATH>  trusted causal evidence public key
        --present-hold-blocks <N>                consecutive present blocks, 1..100 (default: 3)
        --maximum-peak <F>                       absolute candidate peak, 0.5..1 (default: 1)
    The remaining model, probability, energy, accelerator, report, memory,
    metadata, replacement, and JSON options above also apply to causal mode.
"
}

#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
#[derive(Debug)]
struct TargetSpeakerCliOptions {
    mixture: String,
    enrollment: String,
    output: String,
    package: String,
    package_key: String,
    promotion_evidence: String,
    promotion_evidence_key: String,
    report: Option<String>,
    config: denoize::TargetSpeakerExtractionConfig,
    accelerator: AcceleratorPreference,
    max_memory_mb: Option<usize>,
    preserve_metadata: bool,
    commit_mode: CommitMode,
    print_mode: DiagnosticPrintMode,
}

fn parse_target_speaker_args(args: &[String]) -> Result<TargetSpeakerCliOptions, String> {
    let mut positional = Vec::new();
    let mut package = None;
    let mut package_key = None;
    let mut promotion_evidence = None;
    let mut promotion_evidence_key = None;
    let mut report = None;
    let mut config = denoize::TargetSpeakerExtractionConfig::default();
    let mut accelerator = AcceleratorPreference::Cpu;
    let mut accelerator_seen = false;
    let mut max_memory_mb = None;
    let mut preserve_metadata = true;
    let mut commit_mode = CommitMode::NoClobber;
    let mut print_mode = DiagnosticPrintMode::Human;
    let mut scalar_options = std::collections::HashSet::new();
    let mut index = 0usize;
    while index < args.len() {
        let argument = args[index].as_str();
        if matches!(
            argument,
            "--minimum-present-probability"
                | "--minimum-absent-probability"
                | "--maximum-energy-gain-db"
                | "--maximum-peak-gain-db"
                | "--maximum-new-clipping-ratio"
        ) && !scalar_options.insert(argument)
        {
            return Err(format!("{argument} may be supplied only once"));
        }
        match argument {
            "--model-package" if package.is_none() => {
                package = Some(parse_value(args, &mut index, "--model-package")?);
            }
            "--model-package" => return Err("--model-package may be supplied only once".into()),
            "--model-package-key" if package_key.is_none() => {
                package_key = Some(parse_value(args, &mut index, "--model-package-key")?);
            }
            "--model-package-key" => {
                return Err("--model-package-key may be supplied only once".into());
            }
            "--promotion-evidence" if promotion_evidence.is_none() => {
                promotion_evidence = Some(parse_value(args, &mut index, "--promotion-evidence")?);
            }
            "--promotion-evidence" => {
                return Err("--promotion-evidence may be supplied only once".into());
            }
            "--promotion-evidence-key" if promotion_evidence_key.is_none() => {
                promotion_evidence_key =
                    Some(parse_value(args, &mut index, "--promotion-evidence-key")?);
            }
            "--promotion-evidence-key" => {
                return Err("--promotion-evidence-key may be supplied only once".into());
            }
            "--minimum-present-probability" => {
                config.minimum_present_probability =
                    parse_value(args, &mut index, "--minimum-present-probability")?;
            }
            "--minimum-absent-probability" => {
                config.minimum_absent_probability =
                    parse_value(args, &mut index, "--minimum-absent-probability")?;
            }
            "--maximum-energy-gain-db" => {
                config.maximum_energy_gain_db =
                    parse_value(args, &mut index, "--maximum-energy-gain-db")?;
            }
            "--maximum-peak-gain-db" => {
                config.maximum_peak_gain_db =
                    parse_value(args, &mut index, "--maximum-peak-gain-db")?;
            }
            "--maximum-new-clipping-ratio" => {
                config.maximum_new_clipping_ratio =
                    parse_value(args, &mut index, "--maximum-new-clipping-ratio")?;
            }
            "--accelerator" if !accelerator_seen => {
                accelerator_seen = true;
                let value: String = parse_value(args, &mut index, "--accelerator")?;
                accelerator = AcceleratorPreference::parse(&value).ok_or_else(|| {
                    format!(
                        "unknown target-speaker accelerator: {value} (expected cpu, auto, gpu, metal, or cuda)"
                    )
                })?;
            }
            "--accelerator" => return Err("--accelerator may be supplied only once".into()),
            "--report" if report.is_none() => {
                report = Some(parse_value(args, &mut index, "--report")?);
            }
            "--report" => return Err("--report may be supplied only once".into()),
            "--max-memory" if max_memory_mb.is_none() => {
                max_memory_mb = Some(parse_value(args, &mut index, "--max-memory")?);
            }
            "--max-memory" => return Err("--max-memory may be supplied only once".into()),
            "--no-metadata" if preserve_metadata => preserve_metadata = false,
            "--no-metadata" => return Err("--no-metadata may be supplied only once".into()),
            "--replace" if commit_mode == CommitMode::NoClobber => {
                commit_mode = CommitMode::Replace;
            }
            "--replace" => return Err("--replace may be supplied only once".into()),
            "--json" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::Json;
            }
            "--pretty" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err("target-speaker accepts only one of --json or --pretty".into());
            }
            "-h" | "--help" => return Err("target-speaker help requested".into()),
            "-" => {
                return Err(
                    "target-speaker extraction requires regular-file paths; stdin/stdout are unsupported"
                        .into(),
                );
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown target-speaker option: {value}"));
            }
            value => {
                if positional.len() == 3 {
                    return Err(format!("unexpected extra target-speaker argument: {value}"));
                }
                positional.push(value.to_string());
            }
        }
        index += 1;
    }
    let mixture = positional
        .first()
        .cloned()
        .ok_or("target-speaker requires MIXTURE")?;
    let enrollment = positional
        .get(1)
        .cloned()
        .ok_or("target-speaker requires ENROLLMENT")?;
    let output = positional
        .get(2)
        .cloned()
        .ok_or("target-speaker requires OUTPUT")?;
    let package = package.ok_or("target-speaker requires --model-package")?;
    let package_key = package_key.ok_or("target-speaker requires --model-package-key")?;
    let promotion_evidence =
        promotion_evidence.ok_or("target-speaker requires --promotion-evidence")?;
    let promotion_evidence_key =
        promotion_evidence_key.ok_or("target-speaker requires --promotion-evidence-key")?;
    checked_mib_limit_bytes(max_memory_mb, "--max-memory")?;
    config.validate()?;
    Ok(TargetSpeakerCliOptions {
        mixture,
        enrollment,
        output,
        package,
        package_key,
        promotion_evidence,
        promotion_evidence_key,
        report,
        config,
        accelerator,
        max_memory_mb,
        preserve_metadata,
        commit_mode,
        print_mode,
    })
}

#[cfg(feature = "onnx")]
#[derive(Debug)]
struct CausalTargetSpeakerCliOptions {
    mixture: String,
    enrollment: String,
    output: String,
    package: String,
    package_key: String,
    offline_evidence: String,
    offline_evidence_key: String,
    causal_evidence: String,
    causal_evidence_key: String,
    report: Option<String>,
    config: denoize::CausalTargetSpeakerConfig,
    accelerator: AcceleratorPreference,
    max_memory_mb: Option<usize>,
    preserve_metadata: bool,
    commit_mode: CommitMode,
    print_mode: DiagnosticPrintMode,
}

#[cfg(feature = "onnx")]
fn parse_causal_target_speaker_args(
    args: &[String],
) -> Result<CausalTargetSpeakerCliOptions, String> {
    let mut positional = Vec::new();
    let mut package = None;
    let mut package_key = None;
    let mut offline_evidence = None;
    let mut offline_evidence_key = None;
    let mut causal_evidence = None;
    let mut causal_evidence_key = None;
    let mut report = None;
    let mut config = denoize::CausalTargetSpeakerConfig::default();
    let mut accelerator = AcceleratorPreference::Cpu;
    let mut accelerator_seen = false;
    let mut max_memory_mb = None;
    let mut preserve_metadata = true;
    let mut commit_mode = CommitMode::NoClobber;
    let mut print_mode = DiagnosticPrintMode::Human;
    let mut scalar_options = std::collections::HashSet::new();
    let mut index = 0usize;
    while index < args.len() {
        let argument = args[index].as_str();
        if matches!(
            argument,
            "--minimum-present-probability"
                | "--minimum-absent-probability"
                | "--present-hold-blocks"
                | "--maximum-energy-gain-db"
                | "--maximum-peak"
        ) && !scalar_options.insert(argument)
        {
            return Err(format!("{argument} may be supplied only once"));
        }
        match argument {
            "--model-package" if package.is_none() => {
                package = Some(parse_value(args, &mut index, "--model-package")?);
            }
            "--model-package" => return Err("--model-package may be supplied only once".into()),
            "--model-package-key" if package_key.is_none() => {
                package_key = Some(parse_value(args, &mut index, "--model-package-key")?);
            }
            "--model-package-key" => {
                return Err("--model-package-key may be supplied only once".into());
            }
            "--offline-promotion-evidence" if offline_evidence.is_none() => {
                offline_evidence = Some(parse_value(
                    args,
                    &mut index,
                    "--offline-promotion-evidence",
                )?);
            }
            "--offline-promotion-evidence" => {
                return Err("--offline-promotion-evidence may be supplied only once".into());
            }
            "--offline-promotion-evidence-key" if offline_evidence_key.is_none() => {
                offline_evidence_key = Some(parse_value(
                    args,
                    &mut index,
                    "--offline-promotion-evidence-key",
                )?);
            }
            "--offline-promotion-evidence-key" => {
                return Err("--offline-promotion-evidence-key may be supplied only once".into());
            }
            "--causal-promotion-evidence" if causal_evidence.is_none() => {
                causal_evidence = Some(parse_value(
                    args,
                    &mut index,
                    "--causal-promotion-evidence",
                )?);
            }
            "--causal-promotion-evidence" => {
                return Err("--causal-promotion-evidence may be supplied only once".into());
            }
            "--causal-promotion-evidence-key" if causal_evidence_key.is_none() => {
                causal_evidence_key = Some(parse_value(
                    args,
                    &mut index,
                    "--causal-promotion-evidence-key",
                )?);
            }
            "--causal-promotion-evidence-key" => {
                return Err("--causal-promotion-evidence-key may be supplied only once".into());
            }
            "--minimum-present-probability" => {
                config.minimum_present_probability =
                    parse_value(args, &mut index, "--minimum-present-probability")?;
            }
            "--minimum-absent-probability" => {
                config.minimum_absent_probability =
                    parse_value(args, &mut index, "--minimum-absent-probability")?;
            }
            "--present-hold-blocks" => {
                config.present_hold_blocks =
                    parse_value(args, &mut index, "--present-hold-blocks")?;
            }
            "--maximum-energy-gain-db" => {
                config.maximum_energy_gain_db =
                    parse_value(args, &mut index, "--maximum-energy-gain-db")?;
            }
            "--maximum-peak" => {
                config.maximum_peak = parse_value(args, &mut index, "--maximum-peak")?;
            }
            "--accelerator" if !accelerator_seen => {
                accelerator_seen = true;
                let value: String = parse_value(args, &mut index, "--accelerator")?;
                accelerator = AcceleratorPreference::parse(&value).ok_or_else(|| {
                    format!(
                        "unknown causal target-speaker accelerator: {value} (expected cpu, auto, gpu, metal, or cuda)"
                    )
                })?;
            }
            "--accelerator" => return Err("--accelerator may be supplied only once".into()),
            "--report" if report.is_none() => {
                report = Some(parse_value(args, &mut index, "--report")?);
            }
            "--report" => return Err("--report may be supplied only once".into()),
            "--max-memory" if max_memory_mb.is_none() => {
                max_memory_mb = Some(parse_value(args, &mut index, "--max-memory")?);
            }
            "--max-memory" => return Err("--max-memory may be supplied only once".into()),
            "--no-metadata" if preserve_metadata => preserve_metadata = false,
            "--no-metadata" => return Err("--no-metadata may be supplied only once".into()),
            "--replace" if commit_mode == CommitMode::NoClobber => {
                commit_mode = CommitMode::Replace;
            }
            "--replace" => return Err("--replace may be supplied only once".into()),
            "--json" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::Json;
            }
            "--pretty" if print_mode == DiagnosticPrintMode::Human => {
                print_mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err("causal target-speaker accepts only one of --json or --pretty".into());
            }
            "-h" | "--help" => return Err("causal target-speaker help requested".into()),
            "-" => {
                return Err(
                    "causal target-speaker extraction requires regular-file paths; stdin/stdout are unsupported"
                        .into(),
                );
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown causal target-speaker option: {value}"));
            }
            value => {
                if positional.len() == 3 {
                    return Err(format!(
                        "unexpected extra causal target-speaker argument: {value}"
                    ));
                }
                positional.push(value.to_string());
            }
        }
        index += 1;
    }
    let mixture = positional
        .first()
        .cloned()
        .ok_or("causal target-speaker requires MIXTURE")?;
    let enrollment = positional
        .get(1)
        .cloned()
        .ok_or("causal target-speaker requires ENROLLMENT")?;
    let output = positional
        .get(2)
        .cloned()
        .ok_or("causal target-speaker requires OUTPUT")?;
    let package = package.ok_or("causal target-speaker requires --model-package")?;
    let package_key = package_key.ok_or("causal target-speaker requires --model-package-key")?;
    let offline_evidence =
        offline_evidence.ok_or("causal target-speaker requires --offline-promotion-evidence")?;
    let offline_evidence_key = offline_evidence_key
        .ok_or("causal target-speaker requires --offline-promotion-evidence-key")?;
    let causal_evidence =
        causal_evidence.ok_or("causal target-speaker requires --causal-promotion-evidence")?;
    let causal_evidence_key = causal_evidence_key
        .ok_or("causal target-speaker requires --causal-promotion-evidence-key")?;
    checked_mib_limit_bytes(max_memory_mb, "--max-memory")?;
    config.validate()?;
    Ok(CausalTargetSpeakerCliOptions {
        mixture,
        enrollment,
        output,
        package,
        package_key,
        offline_evidence,
        offline_evidence_key,
        causal_evidence,
        causal_evidence_key,
        report,
        config,
        accelerator,
        max_memory_mb,
        preserve_metadata,
        commit_mode,
        print_mode,
    })
}

#[cfg(feature = "onnx")]
fn validate_target_speaker_publication_paths(
    options: &TargetSpeakerCliOptions,
) -> Result<(), String> {
    let mut sources = Vec::new();
    for (path, context) in [
        (&options.mixture, "target-speaker mixture"),
        (&options.enrollment, "target-speaker enrollment"),
        (&options.package, "target-speaker model package"),
        (&options.package_key, "target-speaker model package key"),
        (
            &options.promotion_evidence,
            "target-speaker promotion evidence",
        ),
        (
            &options.promotion_evidence_key,
            "target-speaker promotion evidence key",
        ),
    ] {
        sources.push((
            std::fs::canonicalize(path)
                .map_err(|error| format!("resolve {context} {path}: {error}"))?,
            context,
        ));
    }
    for left in 0..sources.len() {
        for right in left + 1..sources.len() {
            if sources[left].0 == sources[right].0 {
                return Err(format!(
                    "{} and {} must use distinct source files",
                    sources[left].1, sources[right].1
                ));
            }
        }
    }
    let mut destinations = Vec::new();
    for (path, context) in [
        (Some(options.output.as_str()), "target-speaker audio output"),
        (options.report.as_deref(), "target-speaker report"),
    ] {
        let Some(path) = path else {
            continue;
        };
        let normalized = normalized_project_destination(std::path::Path::new(path), context)?;
        let existing = std::fs::canonicalize(&normalized).ok();
        if sources.iter().any(|(source, _)| {
            normalized == *source || existing.as_ref().is_some_and(|path| path == source)
        }) {
            return Err(format!(
                "{context} must not replace a mixture, enrollment, package, evidence, or key"
            ));
        }
        ensure_restoration_destination_available(&normalized, options.commit_mode)?;
        destinations.push((batch_collision_key(&normalized), context));
    }
    destinations.sort_by(|left, right| left.0.cmp(&right.0));
    if let Some(pair) = destinations.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        return Err(format!(
            "{} and {} must use distinct destinations",
            pair[0].1, pair[1].1
        ));
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn validate_causal_target_speaker_publication_paths(
    options: &CausalTargetSpeakerCliOptions,
) -> Result<(), String> {
    let mut sources = Vec::new();
    for (path, context) in [
        (&options.mixture, "causal target-speaker mixture"),
        (&options.enrollment, "causal target-speaker enrollment"),
        (&options.package, "causal target-speaker model package"),
        (
            &options.package_key,
            "causal target-speaker model package key",
        ),
        (
            &options.offline_evidence,
            "causal target-speaker offline promotion evidence",
        ),
        (
            &options.offline_evidence_key,
            "causal target-speaker offline promotion evidence key",
        ),
        (
            &options.causal_evidence,
            "causal target-speaker promotion evidence",
        ),
        (
            &options.causal_evidence_key,
            "causal target-speaker promotion evidence key",
        ),
    ] {
        sources.push((
            std::fs::canonicalize(path)
                .map_err(|error| format!("resolve {context} {path}: {error}"))?,
            context,
        ));
    }
    for left in 0..sources.len() {
        for right in left + 1..sources.len() {
            if sources[left].0 == sources[right].0 {
                return Err(format!(
                    "{} and {} must use distinct source files",
                    sources[left].1, sources[right].1
                ));
            }
        }
    }
    let mut destinations = Vec::new();
    for (path, context) in [
        (
            Some(options.output.as_str()),
            "causal target-speaker audio output",
        ),
        (options.report.as_deref(), "causal target-speaker report"),
    ] {
        let Some(path) = path else {
            continue;
        };
        let normalized = normalized_project_destination(std::path::Path::new(path), context)?;
        let existing = std::fs::canonicalize(&normalized).ok();
        if sources.iter().any(|(source, _)| {
            normalized == *source || existing.as_ref().is_some_and(|path| path == source)
        }) {
            return Err(format!(
                "{context} must not replace a mixture, enrollment, package, evidence, or key"
            ));
        }
        ensure_restoration_destination_available(&normalized, options.commit_mode)?;
        destinations.push((batch_collision_key(&normalized), context));
    }
    destinations.sort_by(|left, right| left.0.cmp(&right.0));
    if let Some(pair) = destinations.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        return Err(format!(
            "{} and {} must use distinct destinations",
            pair[0].1, pair[1].1
        ));
    }
    Ok(())
}

fn run_aec(args: &[String]) -> Result<(), String> {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help") {
        print!("{}", aec_usage());
        return Ok(());
    }
    #[cfg(feature = "aec")]
    {
        if args.first().map(String::as_str) == Some("evidence") {
            return run_aec_evidence(&args[1..]);
        }
        if args
            .iter()
            .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
        {
            return Err("AEC --help accepts no other arguments".into());
        }
        let options = parse_aec_args(args)?;
        validate_aec_publication_paths(&options)?;
        run_aec_audio(options)
    }
    #[cfg(not(feature = "aec"))]
    {
        let _ = args;
        Err("acoustic echo cancellation requires a build with the aec feature".into())
    }
}

#[cfg(feature = "aec")]
fn run_aec_evidence(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) != Some("verify") {
        return Err(
            "AEC evidence requires: verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]"
                .into(),
        );
    }
    let mut positional = Vec::new();
    let mut mode = DiagnosticPrintMode::Human;
    for argument in &args[1..] {
        match argument.as_str() {
            "--json" if mode == DiagnosticPrintMode::Human => mode = DiagnosticPrintMode::Json,
            "--pretty" if mode == DiagnosticPrintMode::Human => {
                mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err("AEC evidence verify accepts only one output mode".into());
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown AEC evidence option: {value}"));
            }
            value => positional.push(value.to_string()),
        }
    }
    if positional.len() != 2 {
        return Err("AEC evidence verify requires EVIDENCE.json and PUBLIC-KEY.json".into());
    }
    let evidence = denoize::SignedAecPromotionEvidence::from_file(&positional[0])?;
    let key = ReceiptPublicKey::from_file(&positional[1])?;
    evidence.verify_signature(&key)?;
    match mode {
        DiagnosticPrintMode::Json => println!(
            "{}",
            serde_json::to_string(&evidence)
                .map_err(|error| format!("serialize AEC promotion evidence: {error}"))?
        ),
        DiagnosticPrintMode::PrettyJson => println!("{}", evidence.to_pretty_json()?),
        DiagnosticPrintMode::Human => println!(
            "verified AEC promotion evidence: implementation={}, strata={}, real_devices={}, nonlinear_devices={}, paced_blocks={}, accepted={}",
            evidence.payload.implementation,
            evidence.payload.strata.len(),
            evidence.payload.real_device_cases,
            evidence.payload.nonlinear_device_cases,
            evidence.payload.paced_realtime_blocks,
            evidence.payload.accepted,
        ),
    }
    if !evidence.payload.accepted {
        return Err("AEC promotion evidence is authentic but does not pass promotion gates".into());
    }
    Ok(())
}

#[cfg(feature = "aec")]
fn validate_aec_publication_paths(options: &AecCliOptions) -> Result<(), String> {
    let mut sources = Vec::new();
    for (path, context) in [
        (options.microphone.as_str(), "AEC microphone"),
        (options.reference.as_str(), "AEC far-end reference"),
        (
            options.promotion_evidence.as_str(),
            "AEC promotion evidence",
        ),
        (
            options.promotion_evidence_key.as_str(),
            "AEC promotion evidence key",
        ),
    ] {
        sources.push((
            std::fs::canonicalize(path)
                .map_err(|error| format!("resolve {context} {path}: {error}"))?,
            context,
        ));
    }
    if let Some(path) = options.config_path.as_deref() {
        sources.push((
            std::fs::canonicalize(path)
                .map_err(|error| format!("resolve AEC configuration {path}: {error}"))?,
            "AEC configuration",
        ));
    }
    for left in 0..sources.len() {
        for right in left + 1..sources.len() {
            if sources[left].0 == sources[right].0 {
                return Err(format!(
                    "{} and {} must use distinct source files",
                    sources[left].1, sources[right].1
                ));
            }
        }
    }
    let mut destinations = Vec::new();
    for (path, context) in [
        (Some(options.output.as_str()), "AEC audio output"),
        (options.report.as_deref(), "AEC report"),
    ] {
        let Some(path) = path else {
            continue;
        };
        let normalized = normalized_project_destination(std::path::Path::new(path), context)?;
        let existing = std::fs::canonicalize(&normalized).ok();
        if sources.iter().any(|(source, _)| {
            normalized == *source || existing.as_ref().is_some_and(|path| path == source)
        }) {
            return Err(format!(
                "{context} must not replace a microphone, reference, configuration, evidence, or key"
            ));
        }
        ensure_restoration_destination_available(&normalized, options.commit_mode)?;
        destinations.push((batch_collision_key(&normalized), context));
    }
    destinations.sort_by(|left, right| left.0.cmp(&right.0));
    if let Some(pair) = destinations.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        return Err(format!(
            "{} and {} must use distinct destinations",
            pair[0].1, pair[1].1
        ));
    }
    Ok(())
}

#[cfg(feature = "aec")]
fn run_aec_audio(options: AecCliOptions) -> Result<(), String> {
    let maximum = checked_mib_limit_bytes(options.max_memory_mb, "--max-memory")?;
    let config = options
        .config_path
        .as_deref()
        .map(denoize::AecConfig::from_file)
        .transpose()?
        .unwrap_or_default();
    let evidence = denoize::SignedAecPromotionEvidence::from_file(&options.promotion_evidence)?;
    let evidence_key = ReceiptPublicKey::from_file(&options.promotion_evidence_key)?;
    // Authenticate promotion and bind every DSP parameter before either
    // user-controlled audio source is opened.
    let session = denoize::AecSession::prepare(&evidence, &evidence_key, config.clone())?;
    let aec_memory = denoize::estimate_aec_memory_bytes(&config)?;
    ensure_memory_limit(aec_memory, options.max_memory_mb, "AEC filter working set")?;

    let mut microphone_session = AudioInputSession::open(&options.microphone)?;
    let mut reference_session = AudioInputSession::open(&options.reference)?;
    let session_memory = estimate_session_memory_bytes(&microphone_session)
        .saturating_add(estimate_session_memory_bytes(&reference_session));
    ensure_memory_limit(
        aec_memory.saturating_add(session_memory),
        options.max_memory_mb,
        "AEC input/filter preflight",
    )?;
    let decode_maximum = maximum.map(|limit| {
        limit
            .saturating_sub(aec_memory)
            .saturating_sub(session_memory)
    });
    let microphone = read_audio_from_session_with_limits(
        &mut microphone_session,
        DecodeLimits::new(
            metadata_limits_for_available_bytes(decode_maximum),
            decode_maximum,
        ),
    )?;
    let retained_microphone = denoize::estimate_audio_memory_bytes(&microphone);
    let reference_maximum = decode_maximum.map(|limit| limit.saturating_sub(retained_microphone));
    let reference = read_audio_from_session_with_limits(
        &mut reference_session,
        DecodeLimits::new(
            metadata_limits_for_available_bytes(reference_maximum),
            reference_maximum,
        ),
    )?;
    let retained_reference = denoize::estimate_audio_memory_bytes(&reference);
    let working_set = aec_memory
        .saturating_add(session_memory)
        .saturating_add(retained_reference)
        .saturating_add(retained_microphone.saturating_mul(4));
    ensure_memory_limit(
        working_set,
        options.max_memory_mb,
        "AEC decoded/alignment/output working set",
    )?;
    let mapping = denoize::AecClockMapping {
        microphone_sample_rate: microphone.sample_rate,
        reference_sample_rate: reference.sample_rate,
        reference_clock_ppm: options.reference_clock_ppm,
        initial_delay_samples: options.initial_delay_samples,
        route_generation: options.route_generation,
    };
    let result = session.render(&microphone, &reference, &mapping)?;
    let mut staged_report = options
        .report
        .as_deref()
        .map(|path| stage_restoration_json(path, &result.report))
        .transpose()?;
    let format = OutputFormat::from_path(std::path::Path::new(&options.output))?;
    let encode_options = EncodeOptions::default();
    encode_options.validate_options(format)?;
    format.validate_config(&result.audio, &encode_options)?;
    let metadata = if options.preserve_metadata {
        microphone_session.read_metadata_with_limits(retained_metadata_limits(
            options.max_memory_mb,
            working_set,
        )?)?
    } else {
        None
    };
    denoize::write_audio_transactional(
        &options.output,
        &result.audio,
        encode_options,
        metadata,
        options.commit_mode,
    )?;
    if let Some(report) = staged_report.take() {
        report.commit(options.commit_mode)?;
    }
    match options.print_mode {
        DiagnosticPrintMode::Json => println!("{}", result.report.to_json()?),
        DiagnosticPrintMode::PrettyJson => println!("{}", result.report.to_pretty_json()?),
        DiagnosticPrintMode::Human => println!(
            "AEC: delay={} samples confidence={:.3} far-only={} double-talk={} uncertain={} frames={} latency_ms={:.3}",
            result.report.delay.signed_delay_samples,
            result.report.delay.confidence,
            result.report.far_end_only_blocks,
            result.report.double_talk_blocks,
            result.report.reference_uncertain_blocks,
            result.report.output_frames,
            result.report.algorithmic_plus_buffering_milliseconds,
        ),
    }
    Ok(())
}

fn run_target_speaker(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) == Some("causal") {
        return run_causal_target_speaker(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("evidence") {
        return run_target_speaker_evidence(&args[1..]);
    }
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        if args.len() != 1 {
            return Err("target-speaker --help accepts no other arguments".into());
        }
        print!("{}", target_speaker_usage());
        return Ok(());
    }
    let options = parse_target_speaker_args(args)?;
    #[cfg(feature = "onnx")]
    {
        validate_target_speaker_publication_paths(&options)?;
        run_target_speaker_audio(options)
    }
    #[cfg(not(feature = "onnx"))]
    {
        run_target_speaker_audio(options)
    }
}

fn run_causal_target_speaker(args: &[String]) -> Result<(), String> {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help") {
        print!("{}", target_speaker_usage());
        return Ok(());
    }
    #[cfg(feature = "onnx")]
    {
        run_causal_target_speaker_with_onnx(args)
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = args;
        Err("causal target-speaker extraction requires a build with the onnx feature".into())
    }
}

#[cfg(feature = "onnx")]
fn run_causal_target_speaker_with_onnx(args: &[String]) -> Result<(), String> {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help") {
        print!("{}", target_speaker_usage());
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("evidence") {
        return run_causal_target_speaker_evidence(&args[1..]);
    }
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        return Err("target-speaker causal --help accepts no other arguments".into());
    }
    let options = parse_causal_target_speaker_args(args)?;
    validate_causal_target_speaker_publication_paths(&options)?;
    run_causal_target_speaker_audio(options)
}

#[cfg(feature = "onnx")]
fn run_causal_target_speaker_evidence(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) != Some("verify") {
        return Err("target-speaker causal evidence requires: verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]".into());
    }
    let mut positional = Vec::new();
    let mut mode = DiagnosticPrintMode::Human;
    for argument in &args[1..] {
        match argument.as_str() {
            "--json" if mode == DiagnosticPrintMode::Human => mode = DiagnosticPrintMode::Json,
            "--pretty" if mode == DiagnosticPrintMode::Human => {
                mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err(
                    "causal target-speaker evidence verify accepts only one output mode".into(),
                );
            }
            value if value.starts_with('-') => {
                return Err(format!(
                    "unknown causal target-speaker evidence option: {value}"
                ));
            }
            value => positional.push(value.to_string()),
        }
    }
    if positional.len() != 2 {
        return Err(
            "causal target-speaker evidence verify requires EVIDENCE.json and PUBLIC-KEY.json"
                .into(),
        );
    }
    let evidence = denoize::SignedCausalTargetSpeakerPromotionEvidence::from_file(&positional[0])?;
    let key = ReceiptPublicKey::from_file(&positional[1])?;
    evidence.verify_signature(&key)?;
    match mode {
        DiagnosticPrintMode::Json => println!(
            "{}",
            serde_json::to_string(&evidence).map_err(|error| format!(
                "serialize causal target-speaker promotion evidence: {error}"
            ))?
        ),
        DiagnosticPrintMode::PrettyJson => println!("{}", evidence.to_pretty_json()?),
        DiagnosticPrintMode::Human => println!(
            "verified causal target-speaker promotion evidence: package={}, strata={}, latency_ms={}, paced_blocks={}, transitions={}, accepted={}",
            evidence.payload.model_package_sha256,
            evidence.payload.strata.len(),
            evidence.payload.effective_latency_milliseconds,
            evidence.payload.realtime.paced_blocks,
            evidence.payload.transitions.absent_to_present_cases
                + evidence.payload.transitions.present_to_absent_cases
                + evidence.payload.transitions.uncertain_transition_cases,
            evidence.payload.accepted
        ),
    }
    if !evidence.payload.accepted {
        return Err(
            "causal target-speaker promotion evidence is authentic but does not pass promotion gates"
                .into(),
        );
    }
    Ok(())
}

fn run_target_speaker_evidence(args: &[String]) -> Result<(), String> {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help") {
        print!("{}", target_speaker_usage());
        return Ok(());
    }
    if args.first().map(String::as_str) != Some("verify") {
        return Err("target-speaker evidence requires: verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]".into());
    }
    let mut positional = Vec::new();
    let mut mode = DiagnosticPrintMode::Human;
    for argument in &args[1..] {
        match argument.as_str() {
            "--json" if mode == DiagnosticPrintMode::Human => mode = DiagnosticPrintMode::Json,
            "--pretty" if mode == DiagnosticPrintMode::Human => {
                mode = DiagnosticPrintMode::PrettyJson;
            }
            "--json" | "--pretty" => {
                return Err("target-speaker evidence verify accepts only one output mode".into());
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown target-speaker evidence option: {value}"));
            }
            value => positional.push(value.to_string()),
        }
    }
    if positional.len() != 2 {
        return Err(
            "target-speaker evidence verify requires EVIDENCE.json and PUBLIC-KEY.json".into(),
        );
    }
    let evidence = denoize::SignedTargetSpeakerPromotionEvidence::from_file(&positional[0])?;
    let key = ReceiptPublicKey::from_file(&positional[1])?;
    evidence.verify_signature(&key)?;
    match mode {
        DiagnosticPrintMode::Json => println!(
            "{}",
            serde_json::to_string(&evidence).map_err(|error| format!(
                "serialize target-speaker promotion evidence: {error}"
            ))?
        ),
        DiagnosticPrintMode::PrettyJson => println!("{}", evidence.to_pretty_json()?),
        DiagnosticPrintMode::Human => println!(
            "verified target-speaker promotion evidence: package={}, strata={}, targets={}, interferers={}, languages={}, accepted={}",
            evidence.payload.model_package_sha256,
            evidence.payload.strata.len(),
            evidence.payload.target_speaker_count,
            evidence.payload.interferer_speaker_count,
            evidence.payload.language_count,
            evidence.payload.accepted
        ),
    }
    if !evidence.payload.accepted {
        return Err(
            "target-speaker promotion evidence is authentic but does not pass promotion gates"
                .into(),
        );
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn run_target_speaker_audio(options: TargetSpeakerCliOptions) -> Result<(), String> {
    use zeroize::Zeroize as _;

    let maximum = checked_mib_limit_bytes(options.max_memory_mb, "--max-memory")?;
    let evidence =
        denoize::SignedTargetSpeakerPromotionEvidence::from_file(&options.promotion_evidence)?;
    let evidence_key = ReceiptPublicKey::from_file(&options.promotion_evidence_key)?;
    let package = RuntimeModelPackage::open(&options.package, &options.package_key)?;
    // Session preparation authenticates the selected model bytes, graph tensor
    // names/shapes, signed numerical vectors, and promotion evidence before
    // either user-controlled audio source is opened.
    let session = denoize::TargetSpeakerSession::prepare(
        package,
        &evidence,
        &evidence_key,
        options.accelerator,
    )?;
    let model_working_set = session.model_working_set_bytes()?;
    ensure_memory_limit(
        model_working_set,
        options.max_memory_mb,
        "target-speaker model working set",
    )?;
    let mut mixture_session = AudioInputSession::open(&options.mixture)?;
    let mut enrollment_session = AudioInputSession::open(&options.enrollment)?;
    let session_memory = estimate_session_memory_bytes(&mixture_session)
        .saturating_add(estimate_session_memory_bytes(&enrollment_session));
    ensure_memory_limit(
        model_working_set.saturating_add(session_memory),
        options.max_memory_mb,
        "target-speaker input/model preflight",
    )?;
    let decode_maximum = maximum.map(|limit| {
        limit
            .saturating_sub(model_working_set)
            .saturating_sub(session_memory)
    });
    let mixture = read_audio_from_session_with_limits(
        &mut mixture_session,
        DecodeLimits::new(
            metadata_limits_for_available_bytes(decode_maximum),
            decode_maximum,
        ),
    )?;
    let retained_mixture = denoize::estimate_audio_memory_bytes(&mixture);
    let enrollment_maximum = decode_maximum.map(|limit| limit.saturating_sub(retained_mixture));
    let mut enrollment = read_audio_from_session_with_limits(
        &mut enrollment_session,
        DecodeLimits::new(
            metadata_limits_for_available_bytes(enrollment_maximum),
            enrollment_maximum,
        ),
    )?;
    let working_set = denoize::estimate_target_speaker_memory_bytes(&mixture, &enrollment)
        .saturating_add(model_working_set)
        .saturating_add(session_memory);
    if let Err(error) = ensure_memory_limit(
        working_set,
        options.max_memory_mb,
        "target-speaker decoded/model working set",
    ) {
        for channel in &mut enrollment.channels {
            channel.zeroize();
        }
        return Err(error);
    }
    let result = session.extract(&mixture, enrollment, &options.config)?;
    let mut staged_report = options
        .report
        .as_deref()
        .map(|path| stage_restoration_json(path, &result.report))
        .transpose()?;
    if let Some(audio) = result.audio.as_ref() {
        let format = OutputFormat::from_path(std::path::Path::new(&options.output))?;
        let encode_options = EncodeOptions::default();
        encode_options.validate_options(format)?;
        format.validate_config(audio, &encode_options)?;
        let metadata = if options.preserve_metadata {
            mixture_session.read_metadata_with_limits(retained_metadata_limits(
                options.max_memory_mb,
                working_set,
            )?)?
        } else {
            None
        };
        denoize::write_audio_transactional(
            &options.output,
            audio,
            encode_options,
            metadata,
            options.commit_mode,
        )?;
    }
    if let Some(report) = staged_report.take() {
        report.commit(options.commit_mode)?;
    }
    match options.print_mode {
        DiagnosticPrintMode::Json => println!("{}", result.report.to_json()?),
        DiagnosticPrintMode::PrettyJson => println!("{}", result.report.to_pretty_json()?),
        DiagnosticPrintMode::Human => {
            println!(
                "target-speaker extraction: decision={:?} presence={:?} published={} frames={} package={}",
                result.report.decision,
                result.report.presence.state,
                result.report.output_published,
                result.report.source_frames,
                result.report.model.package_sha256
            );
            for warning in &result.report.warnings {
                println!("warning: {warning}");
            }
        }
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn run_causal_target_speaker_audio(options: CausalTargetSpeakerCliOptions) -> Result<(), String> {
    use zeroize::Zeroize as _;

    let maximum = checked_mib_limit_bytes(options.max_memory_mb, "--max-memory")?;
    let offline_evidence =
        denoize::SignedTargetSpeakerPromotionEvidence::from_file(&options.offline_evidence)?;
    let offline_evidence_key = ReceiptPublicKey::from_file(&options.offline_evidence_key)?;
    let causal_evidence =
        denoize::SignedCausalTargetSpeakerPromotionEvidence::from_file(&options.causal_evidence)?;
    let causal_evidence_key = ReceiptPublicKey::from_file(&options.causal_evidence_key)?;
    let package = RuntimeModelPackage::open(&options.package, &options.package_key)?;
    // Authenticate both promotion layers, recurrent vectors, stream geometry,
    // and the selected graph before opening either user-controlled audio file.
    let session = denoize::CausalTargetSpeakerSession::prepare(
        package,
        &offline_evidence,
        &offline_evidence_key,
        &causal_evidence,
        &causal_evidence_key,
        options.accelerator,
    )?;
    let model_working_set = session.model_working_set_bytes()?;
    ensure_memory_limit(
        model_working_set,
        options.max_memory_mb,
        "causal target-speaker model working set",
    )?;
    let mut mixture_session = AudioInputSession::open(&options.mixture)?;
    let mut enrollment_session = AudioInputSession::open(&options.enrollment)?;
    let session_memory = estimate_session_memory_bytes(&mixture_session)
        .saturating_add(estimate_session_memory_bytes(&enrollment_session));
    ensure_memory_limit(
        model_working_set.saturating_add(session_memory),
        options.max_memory_mb,
        "causal target-speaker input/model preflight",
    )?;
    let decode_maximum = maximum.map(|limit| {
        limit
            .saturating_sub(model_working_set)
            .saturating_sub(session_memory)
    });
    let mixture = read_audio_from_session_with_limits(
        &mut mixture_session,
        DecodeLimits::new(
            metadata_limits_for_available_bytes(decode_maximum),
            decode_maximum,
        ),
    )?;
    let retained_mixture = denoize::estimate_audio_memory_bytes(&mixture);
    let enrollment_maximum = decode_maximum.map(|limit| limit.saturating_sub(retained_mixture));
    let mut enrollment = read_audio_from_session_with_limits(
        &mut enrollment_session,
        DecodeLimits::new(
            metadata_limits_for_available_bytes(enrollment_maximum),
            enrollment_maximum,
        ),
    )?;
    let working_set = denoize::estimate_causal_target_speaker_memory_bytes(
        &mixture,
        &enrollment,
        session.sample_rate_hz(),
        session.frame_samples(),
        session.flush_samples(),
    )
    .saturating_add(model_working_set)
    .saturating_add(session_memory);
    if let Err(error) = ensure_memory_limit(
        working_set,
        options.max_memory_mb,
        "causal target-speaker decoded/model working set",
    ) {
        for channel in &mut enrollment.channels {
            channel.zeroize();
        }
        return Err(error);
    }
    let result = session.render(&mixture, enrollment, options.config)?;
    let mut staged_report = options
        .report
        .as_deref()
        .map(|path| stage_restoration_json(path, &result.report))
        .transpose()?;
    let format = OutputFormat::from_path(std::path::Path::new(&options.output))?;
    let encode_options = EncodeOptions::default();
    encode_options.validate_options(format)?;
    format.validate_config(&result.audio, &encode_options)?;
    let metadata = if options.preserve_metadata {
        mixture_session.read_metadata_with_limits(retained_metadata_limits(
            options.max_memory_mb,
            working_set,
        )?)?
    } else {
        None
    };
    denoize::write_audio_transactional(
        &options.output,
        &result.audio,
        encode_options,
        metadata,
        options.commit_mode,
    )?;
    if let Some(report) = staged_report.take() {
        report.commit(options.commit_mode)?;
    }
    match options.print_mode {
        DiagnosticPrintMode::Json => println!("{}", result.report.to_json()?),
        DiagnosticPrintMode::PrettyJson => println!("{}", result.report.to_pretty_json()?),
        DiagnosticPrintMode::Human => {
            println!(
                "causal target-speaker extraction: published_blocks={} muted_blocks={} transitions={} frames={} latency_samples={} package={}",
                result.report.decision_counts.published_present_blocks,
                result.report.decision_counts.muted_absent_blocks
                    + result.report.decision_counts.muted_uncertain_blocks
                    + result.report.decision_counts.muted_present_warmup_blocks
                    + result.report.decision_counts.muted_safety_gate_blocks
                    + result.report.decision_counts.muted_flush_blocks,
                result.report.presence_transitions,
                result.report.output_frames,
                result.report.algorithmic_latency_samples,
                result.report.model.package_sha256,
            );
            for warning in &result.report.warnings {
                println!("warning: {warning}");
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn run_target_speaker_audio(_options: TargetSpeakerCliOptions) -> Result<(), String> {
    Err("target-speaker extraction requires a build with the onnx feature".into())
}

#[cfg(test)]
mod microphone_array_cli_tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn array_parser_requires_explicit_geometry_and_evidence() {
        let error = parse_microphone_array_args(&arguments(&["array.wav", "out.wav"])).unwrap_err();
        assert_eq!(
            error,
            "microphone-array enhancement requires --array-config"
        );
    }

    #[test]
    fn array_parser_accepts_closed_publication_options() {
        let parsed = parse_microphone_array_args(&arguments(&[
            "array.wav",
            "out.flac",
            "--array-config",
            "geometry.json",
            "--promotion-evidence",
            "evidence.json",
            "--promotion-evidence-key",
            "key.json",
            "--report",
            "report.json",
            "--max-memory",
            "512",
            "--no-metadata",
            "--replace",
            "--pretty",
        ]))
        .unwrap();
        assert_eq!(parsed.input, "array.wav");
        assert_eq!(parsed.output, "out.flac");
        assert_eq!(parsed.config_path, "geometry.json");
        assert_eq!(parsed.max_memory_mb, Some(512));
        assert!(!parsed.preserve_metadata);
        assert_eq!(parsed.commit_mode, CommitMode::Replace);
        assert_eq!(parsed.print_mode, DiagnosticPrintMode::PrettyJson);
    }

    #[test]
    fn array_parser_rejects_ambiguous_or_streaming_inputs() {
        let base = [
            "array.wav",
            "out.wav",
            "--array-config",
            "geometry.json",
            "--promotion-evidence",
            "evidence.json",
            "--promotion-evidence-key",
            "key.json",
        ];
        let mut duplicate = base.to_vec();
        duplicate.extend(["--array-config", "other.json"]);
        assert!(parse_microphone_array_args(&arguments(&duplicate)).is_err());

        let mut modes = base.to_vec();
        modes.extend(["--json", "--pretty"]);
        assert!(parse_microphone_array_args(&arguments(&modes)).is_err());

        let mut streaming = base.to_vec();
        streaming[0] = "-";
        assert!(parse_microphone_array_args(&arguments(&streaming)).is_err());
    }
}

#[cfg(all(test, feature = "aec"))]
mod aec_cli_tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn parser_requires_both_authenticated_evidence_inputs() {
        let error = parse_aec_args(&arguments(&["mic.wav", "ref.wav", "out.wav"])).unwrap_err();
        assert_eq!(error, "AEC requires --promotion-evidence");
        let parsed = parse_aec_args(&arguments(&[
            "mic.wav",
            "ref.wav",
            "out.wav",
            "--promotion-evidence",
            "evidence.json",
            "--promotion-evidence-key",
            "public.json",
        ]))
        .unwrap();
        assert_eq!(parsed.initial_delay_samples, 0);
        assert_eq!(parsed.reference_clock_ppm, 0.0);
        assert_eq!(parsed.commit_mode, CommitMode::NoClobber);
    }

    #[test]
    fn parser_accepts_negative_delay_and_clock_mapping() {
        let parsed = parse_aec_args(&arguments(&[
            "mic.wav",
            "ref.wav",
            "out.wav",
            "--promotion-evidence",
            "evidence.json",
            "--promotion-evidence-key",
            "public.json",
            "--aec-config",
            "config.json",
            "--reference-clock-ppm",
            "-125.5",
            "--initial-delay-samples",
            "-240",
            "--route-generation",
            "9",
            "--replace",
            "--pretty",
        ]))
        .unwrap();
        assert_eq!(parsed.reference_clock_ppm, -125.5);
        assert_eq!(parsed.initial_delay_samples, -240);
        assert_eq!(parsed.route_generation, 9);
        assert_eq!(parsed.config_path.as_deref(), Some("config.json"));
        assert_eq!(parsed.commit_mode, CommitMode::Replace);
        assert_eq!(parsed.print_mode, DiagnosticPrintMode::PrettyJson);
    }

    #[test]
    fn parser_rejects_duplicate_and_unsafe_scalar_options() {
        let base = [
            "mic.wav",
            "ref.wav",
            "out.wav",
            "--promotion-evidence",
            "evidence.json",
            "--promotion-evidence-key",
            "public.json",
        ];
        let mut duplicate = base.to_vec();
        duplicate.extend(["--route-generation", "1", "--route-generation", "2"]);
        assert!(parse_aec_args(&arguments(&duplicate)).is_err());
        let mut unsafe_clock = base.to_vec();
        unsafe_clock.extend(["--reference-clock-ppm", "2000.1"]);
        assert!(parse_aec_args(&arguments(&unsafe_clock)).is_err());
    }
}

#[cfg(test)]
mod target_speaker_cli_tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn parser_requires_all_authenticated_inputs_without_opening_them() {
        let base = ["mixture.wav", "enrollment.wav", "output.wav"];
        let error = parse_target_speaker_args(&arguments(&base)).unwrap_err();
        assert_eq!(error, "target-speaker requires --model-package");
        let parsed = parse_target_speaker_args(&arguments(&[
            "mixture.wav",
            "enrollment.wav",
            "output.wav",
            "--model-package",
            "model.dmp",
            "--model-package-key",
            "model.pub",
            "--promotion-evidence",
            "evidence.json",
            "--promotion-evidence-key",
            "evidence.pub.json",
        ]))
        .unwrap();
        assert_eq!(parsed.config.minimum_present_probability, 0.9);
        assert_eq!(parsed.commit_mode, CommitMode::NoClobber);
    }

    #[test]
    fn parser_rejects_weak_or_ambiguous_options() {
        let common = [
            "mixture.wav",
            "enrollment.wav",
            "output.wav",
            "--model-package",
            "model.dmp",
            "--model-package-key",
            "model.pub",
            "--promotion-evidence",
            "evidence.json",
            "--promotion-evidence-key",
            "evidence.pub.json",
        ];
        let mut weak = common
            .iter()
            .map(|value| (*value).into())
            .collect::<Vec<String>>();
        weak.extend(["--minimum-present-probability".into(), "0.2".into()]);
        assert!(parse_target_speaker_args(&weak)
            .unwrap_err()
            .contains("0.5..=1"));
        let mut modes = common
            .iter()
            .map(|value| (*value).into())
            .collect::<Vec<String>>();
        modes.extend(["--json".into(), "--pretty".into()]);
        assert!(parse_target_speaker_args(&modes)
            .unwrap_err()
            .contains("only one"));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn causal_parser_requires_both_promotion_layers() {
        let base = ["mixture.wav", "enrollment.wav", "output.wav"];
        assert_eq!(
            parse_causal_target_speaker_args(&arguments(&base)).unwrap_err(),
            "causal target-speaker requires --model-package"
        );
        let parsed = parse_causal_target_speaker_args(&arguments(&[
            "mixture.wav",
            "enrollment.wav",
            "output.wav",
            "--model-package",
            "model.dmp",
            "--model-package-key",
            "model.pub",
            "--offline-promotion-evidence",
            "offline.json",
            "--offline-promotion-evidence-key",
            "offline.pub.json",
            "--causal-promotion-evidence",
            "causal.json",
            "--causal-promotion-evidence-key",
            "causal.pub.json",
        ]))
        .unwrap();
        assert_eq!(parsed.config.present_hold_blocks, 3);
        assert_eq!(parsed.config.maximum_peak, 1.0);
        assert_eq!(parsed.commit_mode, CommitMode::NoClobber);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn causal_parser_rejects_weak_or_duplicate_options() {
        let common = [
            "mixture.wav",
            "enrollment.wav",
            "output.wav",
            "--model-package",
            "model.dmp",
            "--model-package-key",
            "model.pub",
            "--offline-promotion-evidence",
            "offline.json",
            "--offline-promotion-evidence-key",
            "offline.pub.json",
            "--causal-promotion-evidence",
            "causal.json",
            "--causal-promotion-evidence-key",
            "causal.pub.json",
        ];
        let mut weak = arguments(&common);
        weak.extend(["--present-hold-blocks".into(), "0".into()]);
        assert!(parse_causal_target_speaker_args(&weak)
            .unwrap_err()
            .contains("1..=100"));
        let mut duplicate = arguments(&common);
        duplicate.extend([
            "--maximum-peak".into(),
            "1".into(),
            "--maximum-peak".into(),
            "1".into(),
        ]);
        assert!(parse_causal_target_speaker_args(&duplicate)
            .unwrap_err()
            .contains("only once"));
        let mut unnormalized = arguments(&common);
        unnormalized.extend(["--maximum-peak".into(), "1.1".into()]);
        assert!(parse_causal_target_speaker_args(&unnormalized)
            .unwrap_err()
            .contains("0.5..=1"));
    }
}

#[cfg(test)]
mod universal_cli_tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn universal_cli_defaults_to_discriminative_primary_and_no_clobber() {
        let parsed = parse_universal_args(&arguments(&[
            "input.wav",
            "output.wav",
            "--model-package",
            "model.dmp",
            "--model-package-key",
            "model.pub",
        ]))
        .unwrap();
        assert_eq!(
            parsed.config.model_family,
            denoize::UniversalModelFamily::Discriminative
        );
        assert_eq!(
            parsed.config.render_role,
            denoize::UniversalRenderRole::Primary
        );
        assert!(!parsed.config.allow_experimental);
        assert_eq!(parsed.accelerator, AcceleratorPreference::Cpu);
        assert_eq!(parsed.commit_mode, CommitMode::NoClobber);
        assert!(parsed.preserve_metadata);
    }

    #[test]
    fn universal_cli_requires_explicit_alternate_for_experimental_models() {
        let base = [
            "input.wav",
            "output.wav",
            "--model-package",
            "model.dmp",
            "--model-package-key",
            "model.pub",
            "--family",
            "generative",
        ];
        assert!(parse_universal_args(&arguments(&base))
            .unwrap_err()
            .contains("allow_experimental=true"));

        let mut allowed = base.to_vec();
        allowed.extend(["--render-role", "alternate", "--experimental"]);
        let parsed = parse_universal_args(&arguments(&allowed)).unwrap();
        assert_eq!(
            parsed.config.model_family,
            denoize::UniversalModelFamily::Generative
        );
        assert_eq!(
            parsed.config.render_role,
            denoize::UniversalRenderRole::Alternate
        );
        assert!(parsed.config.allow_experimental);
    }

    #[test]
    fn universal_cli_rejects_ambiguous_and_unbounded_values_before_io() {
        let required = [
            "input.wav",
            "output.wav",
            "--model-package",
            "model.dmp",
            "--model-package-key",
            "model.pub",
        ];
        for extra in [
            vec!["--json", "--pretty"],
            vec!["--max-memory", "0"],
            vec!["--analysis-seconds", "61"],
            vec!["--maximum-new-clipping-ratio", "NaN"],
            vec!["--family", "safe", "--family", "generative"],
            vec![
                "--minimum-degradation-score",
                "0.1",
                "--minimum-degradation-score",
                "0.2",
            ],
        ] {
            let mut values = required.to_vec();
            values.extend(extra);
            assert!(parse_universal_args(&arguments(&values)).is_err());
        }

        let error = run_universal(&arguments(&[
            "missing.wav",
            "output.wav",
            "--model-package",
            "missing.dmp",
            "--model-package-key",
            "missing.pub",
            "--family",
            "hybrid",
        ]))
        .unwrap_err();
        assert!(error.contains("allow_experimental=true"));
        assert!(!error.contains("resolve universal input"));
    }

    #[test]
    fn universal_evidence_parser_is_closed_before_file_io() {
        assert!(
            run_universal_evidence(&arguments(&["verify", "evidence.json"]))
                .unwrap_err()
                .contains("requires EVIDENCE.json and PUBLIC-KEY.json")
        );
        assert!(run_universal_evidence(&arguments(&[
            "verify",
            "evidence.json",
            "public.json",
            "--json",
            "--pretty",
        ]))
        .unwrap_err()
        .contains("only one output mode"));
        assert!(run_universal_evidence(&arguments(&["unknown"]))
            .unwrap_err()
            .contains("requires: verify"));
    }
}

fn print_diagnostic_report(report: &denoize::DiagnosticReport) {
    println!(
        "quality: {:.1}/100 (MOS proxy {:.2}, uncertainty ±{:.2})",
        report.quality.score, report.quality.estimated_mos_proxy, report.quality.uncertainty
    );
    println!(
        "input: {} / {}, {} Hz, {} channel(s), {:.2} seconds analyzed at {} Hz ({})",
        report.input.format,
        report.input.codec,
        report.input.sample_rate,
        report.input.channels,
        report.input.analyzed_seconds,
        report.input.analysis_sample_rate,
        report.input.analysis_mode
    );
    println!(
        "signal: rms={:.1} dBFS peak={:.1} dBFS noise-floor={:.1} dBFS SNR-proxy={:.1} dB bandwidth={:.0} Hz",
        report.metrics.rms_dbfs,
        report.metrics.peak_dbfs,
        report.metrics.noise_floor_dbfs,
        report.metrics.estimated_snr_db,
        report.metrics.estimated_bandwidth_hz
    );
    println!("findings:");
    for finding in &report.findings {
        println!(
            "  {} detected={} severity={:.3} confidence={:.3}: {}",
            finding.kind, finding.detected, finding.severity, finding.confidence, finding.evidence
        );
    }
    println!(
        "recommended pipeline: {}",
        report.recommended_pipeline.join(" -> ")
    );
    println!(
        "warning: this native proxy does not assess words, phonemes, speaker identity, or generative hallucination"
    );
}

fn print_assessment_report(report: &denoize::AssessmentReport) {
    println!("assessment: {}", report.verdict);
    if let (Some(baseline), Some(comparison)) = (&report.baseline, &report.comparison) {
        println!(
            "quality: {:.1} -> {:.1} ({:+.1}); MOS proxy {:.2} -> {:.2} ({:+.2})",
            baseline.quality.score,
            report.candidate.quality.score,
            comparison.quality_score_delta,
            baseline.quality.estimated_mos_proxy,
            report.candidate.quality.estimated_mos_proxy,
            comparison.estimated_mos_proxy_delta
        );
        println!(
            "presentation: rate={} channels={} duration={} semantic-fidelity-assessed={}",
            comparison.sample_rate_equal,
            comparison.channel_count_equal,
            comparison.presentation_preserved,
            comparison.semantic_fidelity_assessed
        );
    } else {
        print_diagnostic_report(&report.candidate);
    }
    for warning in &report.warnings {
        println!("warning: {warning}");
    }
}

fn format_device_memory(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GiB device memory", bytes as f64 / GIB)
    } else {
        format!("{:.1} MiB device memory", bytes as f64 / MIB)
    }
}

#[cfg(feature = "live")]
fn run_live(args: &[String]) -> Result<(), String> {
    let mut parseable = vec!["-".to_string(), "-".to_string()];
    parseable.extend_from_slice(args);
    let (_, _, ov) = parse_args(&parseable)?;
    if ov.isolate && std::env::var_os(ISOLATED_CHILD_ENV).is_none() {
        let mut child_args = vec!["live".to_string()];
        child_args.extend_from_slice(args);
        return run_isolated(&child_args, &ov);
    }
    validate_effective_options(&ov, 48_000)?;
    if ov.list_devices {
        let (inputs, outputs) = denoize::live::device_names()?;
        println!("Input devices:");
        for device in inputs {
            println!("  {device}");
        }
        println!("Output devices:");
        for device in outputs {
            println!("  {device}");
        }
        return Ok(());
    }
    let backend = if ov.auto_backend {
        service::select_live_backend()
    } else {
        ov.backend.unwrap_or(Backend::Classical)
    };
    let sample_rate = 48_000;
    let denoiser = build_config(&ov, sample_rate);
    let backend_options = build_backend_options(&ov)?;
    let governor = resource_governor(&ov, 1)?;
    let resilience = denoize::live::LiveResilienceConfig::new()
        .with_target_latency_ms(ov.live_latency_ms.unwrap_or(0))
        .with_max_drift_ppm(ov.max_drift_ppm.unwrap_or(2_500))
        .with_reconnect_timeout_ms(ov.reconnect_timeout_ms.unwrap_or(30_000));
    let prepared = denoize::live::PreparedLiveConfig::new(denoize::live::LiveConfig {
        input_device: ov.input_device,
        output_device: ov.output_device,
        chunk_ms: ov.chunk_ms.unwrap_or(100),
        backend,
        backend_options,
        denoiser,
    })?
    .with_resilience(resilience)?;
    let json = ov.json;
    let mut previous_state = None;
    let mut last_running_report = None::<Instant>;
    denoize::live::run_prepared_with_governor_and_status(prepared, &governor, move |status| {
        let now = Instant::now();
        let state_changed = previous_state != Some(status.connection_state);
        let running_due = last_running_report
            .is_none_or(|last| now.saturating_duration_since(last).as_secs_f64() >= 1.0);
        if !state_changed
            && (status.connection_state != denoize::live::LiveConnectionState::Running
                || !running_due)
        {
            return;
        }
        previous_state = Some(status.connection_state);
        if status.connection_state == denoize::live::LiveConnectionState::Running {
            last_running_report = Some(now);
        }
        let state = live_connection_state_name(status.connection_state);
        if json {
            println!(
                "{}",
                serialize_json_line(&LiveStatusJson {
                    schema: CLI_JSON_SCHEMA,
                    schema_version: CLI_JSON_SCHEMA_VERSION,
                    event: "status",
                    mode: "live",
                    state,
                    input_sample_rate: status.input_sample_rate,
                    output_sample_rate: status.output_sample_rate,
                    input_channels: status.input_channels,
                    output_channels: status.output_channels,
                    chunk_frames: status.chunk_frames,
                    input_level: status.input_level,
                    output_level: status.output_level,
                    processed_chunks: status.processed_chunks,
                    dropped_chunks: status.dropped_chunks,
                    underrun_frames: status.underrun_frames,
                    overflow_frames: status.overflow_frames,
                    queued_frames: status.queued_frames,
                    target_queue_frames: status.target_queue_frames,
                    queue_latency_ms: status.queue_latency_ms,
                    processing_latency_ms: status.processing_latency_ms,
                    input_device_latency_ms: status.input_device_latency_ms,
                    output_device_latency_ms: status.output_device_latency_ms,
                    estimated_total_latency_ms: status.estimated_total_latency_ms,
                    drift_correction_ppm: status.drift_correction_ppm,
                    reconnect_attempts: status.reconnect_attempts,
                    device_generation: status.device_generation,
                    accelerator: accelerator_json(status.accelerator),
                })
            );
            // A pipe is block-buffered rather than terminal-line-buffered.
            // Flush every sparse status record so automation observes state
            // transitions and periodic samples when they happen.
            let _ = std::io::Write::flush(&mut std::io::stdout());
        } else if status.sample_rate == 0 {
            eprintln!(
                "live: state={state} reconnects={} generation={}",
                status.reconnect_attempts, status.device_generation
            );
        } else {
            eprintln!(
                    "live: state={state} rate={}->{} Hz queue={}/{} frames latency={:.1} ms (device={:.1}/{:.1}, processing={:.1}) drift={:+.1} ppm underrun={} overflow={} dropped={} reconnects={}",
                    status.input_sample_rate,
                    status.output_sample_rate,
                    status.queued_frames,
                    status.target_queue_frames,
                    status.estimated_total_latency_ms,
                    status.input_device_latency_ms,
                    status.output_device_latency_ms,
                    status.processing_latency_ms,
                    status.drift_correction_ppm,
                    status.underrun_frames,
                    status.overflow_frames,
                    status.dropped_chunks,
                    status.reconnect_attempts,
                );
        }
    })
}

#[cfg(feature = "live")]
fn live_connection_state_name(state: denoize::live::LiveConnectionState) -> &'static str {
    match state {
        denoize::live::LiveConnectionState::Connecting => "connecting",
        denoize::live::LiveConnectionState::Priming => "priming",
        denoize::live::LiveConnectionState::Running => "running",
        denoize::live::LiveConnectionState::Recovering => "recovering",
        _ => "unknown",
    }
}

#[cfg(not(feature = "live"))]
fn run_live(_args: &[String]) -> Result<(), String> {
    Err("live audio is unavailable in this build; rebuild with --features live".into())
}

fn ensure_output_available(path: &std::path::Path, force: bool) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if force && (metadata.is_file() || metadata.file_type().is_symlink()) => {
            Ok(())
        }
        Ok(_) if force => Err(format!(
            "output exists but is not a replaceable file or symlink: {}",
            path.display()
        )),
        Ok(_) => Err(format!(
            "output already exists: {} (use --force to replace it)",
            path.display()
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "inspect output destination {}: {error}",
            path.display()
        )),
    }
}

fn read_stdin_bytes(
    mut input: impl std::io::Read,
    max_memory_mb: Option<usize>,
) -> Result<Vec<u8>, String> {
    let max_encoded_bytes = checked_memory_limit_bytes(max_memory_mb)?
        .map(|limit| limit / INPUT_MEMORY_EXPANSION_FACTOR);
    let bounded_read_len = max_encoded_bytes
        .map(|limit| {
            limit
                .checked_add(1)
                .ok_or_else(|| "--max-memory stdin byte limit overflow".to_string())
                .and_then(|limit| {
                    usize::try_from(limit)
                        .map_err(|_| "--max-memory stdin byte limit is too large".to_string())
                })
        })
        .transpose()?;
    let initial_capacity = bounded_read_len
        .unwrap_or(STDIN_READ_CHUNK_BYTES)
        .min(STDIN_READ_CHUNK_BYTES);
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(initial_capacity)
        .map_err(|_| "unable to reserve memory for stdin input".to_string())?;

    let mut chunk = [0u8; STDIN_READ_CHUNK_BYTES];
    loop {
        let read_len = match bounded_read_len {
            Some(limit) => {
                let remaining = limit
                    .checked_sub(bytes.len())
                    .ok_or_else(|| "stdin byte limit accounting overflow".to_string())?;
                if remaining == 0 {
                    break;
                }
                remaining.min(chunk.len())
            }
            None => chunk.len(),
        };
        let read = input
            .read(&mut chunk[..read_len])
            .map_err(|error| format!("failed to read stdin: {error}"))?;
        if read == 0 {
            break;
        }
        bytes
            .try_reserve_exact(read)
            .map_err(|_| "unable to reserve memory for stdin input".to_string())?;
        bytes.extend_from_slice(&chunk[..read]);
    }

    let encoded_len = u64::try_from(bytes.len())
        .map_err(|_| "stdin input length is too large to represent safely".to_string())?;
    let estimate = encoded_len
        .checked_mul(INPUT_MEMORY_EXPANSION_FACTOR)
        .ok_or_else(|| "stdin input memory estimate overflow".to_string())?
        .max(BYTES_PER_MIB);
    ensure_memory_limit(estimate, max_memory_mb, "stdin input preflight")?;
    Ok(bytes)
}

fn run_one(input: &str, output: &str, ov: Overrides) -> Result<(), String> {
    run_one_with_output_format(
        std::path::Path::new(input),
        std::path::Path::new(output),
        ov,
        None,
        None,
        None,
    )
}

struct StagedProcessOutput {
    transaction: AtomicOutput,
    _resource_permit: Option<ResourcePermit>,
    effective_recipe: Option<Digest>,
    backend: Backend,
    accelerator: AcceleratorSelection,
    channels: usize,
    frames: usize,
    sample_rate: u32,
    elapsed_ms: f64,
    execution_evidence: Option<StagedExecutionEvidence>,
}

struct StagedExecutionEvidence {
    input_fingerprint: FileFingerprint,
    input_probe: AudioProbe,
    model: Option<ConsumedModel>,
    recipe: Digest,
    output_format: OutputFormat,
    resources: ResourceRequest,
}

struct PreparedStreamReceipt {
    path: std::path::PathBuf,
    key: ReceiptSecretKey,
}

#[derive(Clone)]
struct StreamExecutionEvidence {
    input_fingerprint: FileFingerprint,
    stream_info: denoize::AudioStreamInfo,
    model: Option<ConsumedModel>,
    recipe: Digest,
    output_format: OutputFormat,
    resources: ResourceRequest,
    backend: Backend,
    accelerator: AcceleratorSelection,
    deterministic: bool,
    metadata_policy: MetadataPolicy,
}

fn run_one_with_output_format(
    input: &std::path::Path,
    output: &std::path::Path,
    ov: Overrides,
    planned_output_format: Option<OutputFormat>,
    pre_resolved_backend_options: Option<BackendOptions>,
    expected_input_fingerprint: Option<FileFingerprint>,
) -> Result<(), String> {
    let governor = resource_governor(&ov, 1)?;
    let commit_mode = if ov.force {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    };
    let json = ov.json;
    let metadata_policy = if ov.no_metadata {
        MetadataPolicy::Drop
    } else {
        MetadataPolicy::Preserve
    };
    let receipt_paths = match (&ov.receipt, &ov.receipt_key) {
        (Some(receipt), Some(key)) => Some((
            std::path::PathBuf::from(receipt),
            std::path::PathBuf::from(key),
        )),
        (None, None) => None,
        _ => return Err("--receipt and --receipt-key must be supplied together".into()),
    };
    let receipt_requested = receipt_paths.is_some();
    let deterministic = ov.deterministic;
    let (signing_key, planned_publication_mode, planned_reason) =
        if let Some((receipt_path, key_path)) = &receipt_paths {
            preflight_receipt_paths(input, output, receipt_path, key_path)?;
            let (publication, reason) = planned_publication(output, ov.force)?;
            (
                Some(ReceiptSecretKey::from_file(key_path)?),
                publication,
                reason,
            )
        } else {
            (None, "none", "not-requested")
        };
    let recipe_metadata_policy = (json || receipt_requested).then_some(metadata_policy);
    let staged = process_one_to_staged_output(
        input,
        output,
        ov,
        planned_output_format,
        pre_resolved_backend_options,
        None,
        recipe_metadata_policy,
        None,
        expected_input_fingerprint,
        None,
        true,
        Some(&governor),
        receipt_requested,
    )?;
    let Some(staged) = staged else {
        return Ok(());
    };
    let mut staged = staged;
    let staged_receipt = if let Some((receipt_path, _)) = &receipt_paths {
        let plan = build_single_execution_plan_from_staged(
            input,
            output,
            deterministic,
            metadata_policy,
            planned_publication_mode,
            planned_reason,
            &staged,
        )?;
        let output_fingerprint =
            batch_resume::fingerprint_open_file_at(staged.transaction.file_mut(), output)?;
        let plan_item = plan
            .items
            .first()
            .ok_or("single execution plan unexpectedly contains no items")?;
        let item = ReceiptItem::from_plan_item(plan_item, output_fingerprint, "succeeded")?;
        let payload = ExecutionReceiptPayload::new(&plan, vec![item])?;
        let receipt = signing_key
            .as_ref()
            .ok_or("receipt signing key is missing after successful preflight")?
            .sign(payload)?;
        Some(stage_signed_receipt(receipt_path, &receipt)?)
    } else {
        None
    };
    staged.transaction.commit(commit_mode)?;
    if let (Some((receipt_path, _)), Some(receipt)) = (&receipt_paths, staged_receipt) {
        receipt.commit(CommitMode::NoClobber).map_err(|error| {
            format!(
                "audio output was committed to {}, but its signed receipt could not be published to {}: {error}",
                output.display(),
                receipt_path.display()
            )
        })?;
    }
    if json {
        let input = input.to_string_lossy();
        let output = output.to_string_lossy();
        println!(
            "{}",
            process_result_json_line(
                input.as_ref(),
                output.as_ref(),
                service::backend_name(staged.backend),
                staged.accelerator,
                staged.channels,
                staged.frames,
                staged.sample_rate,
                staged.elapsed_ms,
                staged.effective_recipe,
            )
        );
    }
    Ok(())
}

fn preflight_receipt_paths(
    input: &std::path::Path,
    output: &std::path::Path,
    receipt: &std::path::Path,
    secret_key: &std::path::Path,
) -> Result<(), String> {
    match std::fs::symlink_metadata(receipt) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Ok(_) => {
            return Err(format!(
                "execution receipt already exists: {} (refusing to replace it)",
                receipt.display()
            ))
        }
        Err(error) => {
            return Err(format!(
                "inspect execution receipt destination {}: {error}",
                receipt.display()
            ))
        }
    }
    let paths = [
        ("input", input),
        ("output", output),
        ("receipt", receipt),
        ("receipt key", secret_key),
    ];
    let mut normalized = Vec::with_capacity(paths.len());
    for (label, path) in paths {
        normalized.push((label, normalize_batch_path(path)?));
    }
    for left in 0..normalized.len() {
        for right in left + 1..normalized.len() {
            if batch_collision_key(&normalized[left].1) == batch_collision_key(&normalized[right].1)
            {
                return Err(format!(
                    "{} and {} must use distinct paths: {}",
                    normalized[left].0,
                    normalized[right].0,
                    normalized[left].1.display()
                ));
            }
        }
    }
    Ok(())
}

fn prepare_stream_receipt(
    input: &str,
    output: &str,
    options: &Overrides,
) -> Result<Option<PreparedStreamReceipt>, String> {
    let (Some(receipt), Some(secret_key)) = (&options.receipt, &options.receipt_key) else {
        return Ok(None);
    };
    let receipt = std::path::PathBuf::from(receipt);
    let secret_key = std::path::PathBuf::from(secret_key);
    match std::fs::symlink_metadata(&receipt) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Ok(_) => {
            return Err(format!(
                "execution receipt already exists: {} (refusing to replace it)",
                receipt.display()
            ));
        }
        Err(error) => {
            return Err(format!(
                "inspect execution receipt destination {}: {error}",
                receipt.display()
            ));
        }
    }
    let mut paths = Vec::with_capacity(4);
    if input != "-" {
        paths.push(("input", std::path::Path::new(input)));
    }
    if output != "-" {
        paths.push(("output", std::path::Path::new(output)));
    }
    paths.push(("receipt", receipt.as_path()));
    paths.push(("receipt key", secret_key.as_path()));
    let mut normalized = Vec::with_capacity(paths.len());
    for (label, path) in paths {
        normalized.push((label, normalize_batch_path(path)?));
    }
    for left in 0..normalized.len() {
        for right in left + 1..normalized.len() {
            if batch_collision_key(&normalized[left].1) == batch_collision_key(&normalized[right].1)
            {
                return Err(format!(
                    "{} and {} must use distinct paths: {}",
                    normalized[left].0,
                    normalized[right].0,
                    normalized[left].1.display()
                ));
            }
        }
    }
    Ok(Some(PreparedStreamReceipt {
        path: receipt,
        key: ReceiptSecretKey::from_file(secret_key)?,
    }))
}

fn preflight_batch_receipt_paths(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
    receipt: &std::path::Path,
    secret_key: &std::path::Path,
) -> Result<(), String> {
    preflight_receipt_paths(input_dir, output_dir, receipt, secret_key)?;
    let input_root = normalize_batch_path(input_dir)?;
    let receipt_path = normalize_batch_path(receipt)?;
    if receipt_path.starts_with(&input_root) {
        return Err(format!(
            "batch execution receipt must be outside the input directory: {}",
            receipt.display()
        ));
    }
    Ok(())
}

fn build_single_execution_plan_from_staged(
    input: &std::path::Path,
    output: &std::path::Path,
    deterministic: bool,
    metadata_policy: MetadataPolicy,
    publication: &str,
    reason: &str,
    staged: &StagedProcessOutput,
) -> Result<ExecutionPlan, String> {
    let evidence = staged
        .execution_evidence
        .as_ref()
        .ok_or("receipt-bound output is missing execution evidence")?;
    let output_locator = denoize::portable_file_locator(output)?;
    let item_id =
        denoize::execution_item_id(evidence.input_fingerprint, &output_locator, evidence.recipe)?;
    let model = match &evidence.model {
        Some(model) => Some(PlannedArtifact {
            path: denoize::portable_file_locator(&model.path)?,
            fingerprint: model.fingerprint,
        }),
        None => None,
    };
    let frames = u64::try_from(staged.frames)
        .map_err(|_| "receipt frame count is too large to represent".to_string())?;
    ExecutionPlan::new(
        ExecutionKind::File,
        deterministic,
        metadata_policy_name(metadata_policy),
        vec![ExecutionPlanItem {
            item_id,
            input: PlannedArtifact {
                path: denoize::portable_file_locator(input)?,
                fingerprint: evidence.input_fingerprint,
            },
            output: PlannedOutput {
                path: output_locator,
                format: output_format_name(evidence.output_format).into(),
                publication: publication.into(),
                action: "process".into(),
                reason: reason.into(),
                existing_fingerprint: None,
            },
            model,
            recipe: evidence.recipe,
            backend: service::backend_name(staged.backend).into(),
            accelerator: staged.accelerator.effective().name().into(),
            input_format: audio_format_name(evidence.input_probe.format).into(),
            input_codec: audio_codec_name(evidence.input_probe.codec).into(),
            channels: staged.channels as u64,
            frames,
            sample_rate: staged.sample_rate,
            resources: planned_resources(evidence.resources),
        }],
    )
}

fn build_stream_execution_plan_from_evidence(
    input: &str,
    output: &str,
    evidence: &StreamExecutionEvidence,
    frames: u64,
    publication: &str,
    action: &str,
    reason: &str,
    existing_fingerprint: Option<FileFingerprint>,
) -> Result<ExecutionPlan, String> {
    let input_locator = if input == "-" {
        "-".to_string()
    } else {
        denoize::portable_file_locator(std::path::Path::new(input))?
    };
    let output_locator = if output == "-" {
        "-".to_string()
    } else {
        denoize::portable_file_locator(std::path::Path::new(output))?
    };
    let item_id =
        denoize::execution_item_id(evidence.input_fingerprint, &output_locator, evidence.recipe)?;
    let model = evidence
        .model
        .as_ref()
        .map(|model| {
            Ok::<PlannedArtifact, String>(PlannedArtifact {
                path: denoize::portable_file_locator(&model.path)?,
                fingerprint: model.fingerprint,
            })
        })
        .transpose()?;
    ExecutionPlan::new_stream(
        evidence.deterministic,
        metadata_policy_name(evidence.metadata_policy),
        vec![ExecutionPlanItem {
            item_id,
            input: PlannedArtifact {
                path: input_locator,
                fingerprint: evidence.input_fingerprint,
            },
            output: PlannedOutput {
                path: output_locator,
                format: output_format_name(evidence.output_format).into(),
                publication: publication.into(),
                action: action.into(),
                reason: reason.into(),
                existing_fingerprint,
            },
            model,
            recipe: evidence.recipe,
            backend: service::backend_name(evidence.backend).into(),
            accelerator: evidence.accelerator.effective().name().into(),
            input_format: audio_format_name(evidence.stream_info.format).into(),
            input_codec: audio_codec_name(evidence.stream_info.codec).into(),
            channels: u64::from(evidence.stream_info.output_spec.channels),
            frames,
            sample_rate: evidence.stream_info.output_spec.sample_rate,
            resources: planned_resources(evidence.resources),
        }],
    )
}

fn stage_stream_signed_receipt(
    input: &str,
    output: &str,
    receipt: &PreparedStreamReceipt,
    evidence: &StreamExecutionEvidence,
    frames: u64,
    output_fingerprint: FileFingerprint,
    publication: &str,
    action: &str,
    reason: &str,
    existing_fingerprint: Option<FileFingerprint>,
) -> Result<AtomicOutput, String> {
    let plan = build_stream_execution_plan_from_evidence(
        input,
        output,
        evidence,
        frames,
        publication,
        action,
        reason,
        existing_fingerprint,
    )?;
    let plan_item = plan
        .items
        .first()
        .ok_or("stream execution plan unexpectedly contains no items")?;
    let outcome = match action {
        "process" => "succeeded",
        "skip" => "skipped",
        value => return Err(format!("unsupported stream receipt action: {value}")),
    };
    let item = ReceiptItem::from_plan_item(plan_item, output_fingerprint, outcome)?;
    let payload = ExecutionReceiptPayload::new(&plan, vec![item])?;
    let signed = receipt.key.sign(payload)?;
    stage_signed_receipt(&receipt.path, &signed)
}

fn commit_stream_receipt_after_output(
    receipt: &PreparedStreamReceipt,
    staged: AtomicOutput,
    output: &str,
) -> Result<(), String> {
    staged.commit(CommitMode::NoClobber).map_err(|error| {
        format!(
            "stream output was published to {output}, but its signed receipt could not be published to {}: {error}",
            receipt.path.display()
        )
    })
}

fn verify_stream_receipt_sources(
    reader: &AudioStreamReader,
    input: &str,
    evidence: Option<&StreamExecutionEvidence>,
) -> Result<(), String> {
    let Some(evidence) = evidence else {
        return Ok(());
    };
    if reader.fingerprint_input()? != evidence.input_fingerprint {
        return Err("stream receipt input changed while it was processed".into());
    }
    if input != "-"
        && batch_resume::fingerprint_file(std::path::Path::new(input))?
            != evidence.input_fingerprint
    {
        return Err(format!(
            "stream receipt input path changed while it was processed: {input}"
        ));
    }
    if let Some(model) = &evidence.model {
        if batch_resume::fingerprint_file(&model.path)? != model.fingerprint {
            return Err(format!(
                "stream receipt model changed while it was processed: {}",
                model.path.display()
            ));
        }
    }
    Ok(())
}

fn stage_signed_receipt(
    path: &std::path::Path,
    receipt: &SignedExecutionReceipt,
) -> Result<AtomicOutput, String> {
    let mut output = AtomicOutput::new(path)?;
    write_signed_receipt_to_stage(&mut output, path, receipt)?;
    Ok(output)
}

fn write_signed_receipt_to_stage(
    output: &mut AtomicOutput,
    path: &std::path::Path,
    receipt: &SignedExecutionReceipt,
) -> Result<(), String> {
    use std::io::Write as _;

    let mut bytes = receipt.to_pretty_json()?.into_bytes();
    bytes.push(b'\n');
    output
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| format!("write staged execution receipt {}: {error}", path.display()))?;
    output
        .file_mut()
        .sync_data()
        .map_err(|error| format!("sync staged execution receipt {}: {error}", path.display()))?;
    Ok(())
}

fn process_one_to_staged_output(
    input: &std::path::Path,
    output: &std::path::Path,
    ov: Overrides,
    planned_output_format: Option<OutputFormat>,
    pre_resolved_backend_options: Option<BackendOptions>,
    pre_resolved_processing: Option<service::ResolvedProcessingOptions>,
    recipe_metadata_policy: Option<MetadataPolicy>,
    expected_input_probe: Option<AudioProbe>,
    expected_input_fingerprint: Option<FileFingerprint>,
    pre_prepared_backend_session: Option<Arc<BackendSession>>,
    inspect_destination: bool,
    governor: Option<&ResourceGovernor>,
    capture_execution_evidence: bool,
) -> Result<Option<StagedProcessOutput>, String> {
    validate_effective_options(&ov, VALIDATION_SAMPLE_RATE)?;
    let effective_memory_mb = effective_input_memory_mb(&ov);
    let standard_input = input == std::path::Path::new("-");
    let standard_output = output == std::path::Path::new("-");
    if capture_execution_evidence && (standard_input || standard_output || ov.report) {
        return Err("execution receipts require finite regular-file input and output".into());
    }
    if capture_execution_evidence && recipe_metadata_policy.is_none() {
        return Err("receipt-bound processing requires an effective recipe policy".into());
    }
    let encode_options = build_encode_options(&ov)?;
    let output_format = if !ov.report && !standard_output {
        Some(match planned_output_format {
            Some(format) => format,
            None => OutputFormat::from_path(output)?,
        })
    } else {
        None
    };
    validate_encode_preflight(encode_options, output_format)?;

    let resolved_backend_options = match (
        pre_resolved_processing.as_ref(),
        pre_resolved_backend_options,
    ) {
        (Some(_), _) => None,
        (None, Some(options)) => Some(options),
        (None, None) => resolve_explicit_backend_options(&ov)?,
    };
    if inspect_destination && output_format.is_some() {
        ensure_output_available(output, ov.force)?;
    }
    let mut input_session = if standard_input {
        None
    } else {
        Some(AudioInputSession::open(input)?)
    };
    if let (Some(session), Some(expected)) = (&mut input_session, expected_input_probe) {
        let current = probe_audio_session_with_limits(session, decode_limits_for_options(&ov)?)?;
        if current != expected {
            return Err(format!(
                "input codec/container changed after batch preflight: {}",
                input.display()
            ));
        }
    }
    if let (Some(session), Some(expected)) = (&mut input_session, expected_input_fingerprint) {
        let current = batch_resume::fingerprint_input_session(session)?;
        if current != expected {
            return Err(format!(
                "input bytes changed after batch preflight: {}",
                input.display()
            ));
        }
    }
    let captured_input_probe = if capture_execution_evidence {
        let session = input_session
            .as_mut()
            .ok_or("receipt input session is missing after preflight")?;
        let probe = probe_audio_session_with_limits(session, decode_limits_for_options(&ov)?)?;
        if probe.audio_tracks != 1 || probe.codec == AudioCodec::Unknown {
            return Err(format!(
                "receipt input must contain exactly one supported audio track: {}",
                input.display()
            ));
        }
        Some(probe)
    } else {
        None
    };
    let captured_input_fingerprint = if capture_execution_evidence {
        Some(batch_resume::fingerprint_input_session(
            input_session
                .as_mut()
                .ok_or("receipt input session is missing before fingerprinting")?,
        )?)
    } else {
        None
    };
    if let Some(session) = &input_session {
        let estimate = estimate_session_memory_bytes(session);
        ensure_memory_limit(estimate, effective_memory_mb, "input preflight")?;
    }
    let decode_limits = decode_limits_for_options(&ov)?;
    let (mut audio, input_bytes) = if standard_input {
        let stdin = std::io::stdin();
        let bytes = read_stdin_bytes(stdin.lock(), effective_memory_mb)?;
        let input_bytes = u64::try_from(bytes.len())
            .map_err(|_| "stdin input length is too large to represent safely".to_string())?;
        (
            read_wav_bytes_with_limits(bytes, decode_limits)?,
            input_bytes,
        )
    } else {
        let session = input_session
            .as_mut()
            .ok_or("filesystem input session is missing before decode")?;
        let input_bytes = session.len();
        (
            read_audio_from_session_with_limits(session, decode_limits)?,
            input_bytes,
        )
    };
    let decoded_working_set = estimate_audio_working_set_bytes(&audio);
    ensure_memory_limit(
        decoded_working_set,
        effective_memory_mb,
        "decoded audio working set",
    )?;
    let metadata_limits = retained_metadata_limits(effective_memory_mb, decoded_working_set)?;
    let metadata = if !standard_input && !ov.no_metadata {
        input_session
            .as_mut()
            .ok_or("filesystem input session is missing before metadata read")?
            .read_metadata_with_limits(metadata_limits)?
    } else {
        None
    };
    validate_effective_options(&ov, audio.sample_rate)?;
    let resolved_processing = match pre_resolved_processing {
        Some(options) => options,
        None => service::resolve_processing_options(
            &audio,
            build_processing_options(
                &ov,
                audio.sample_rate,
                match resolved_backend_options {
                    Some(options) => options,
                    None => build_backend_options(&ov)?,
                },
            ),
        )?,
    };
    let backend = resolved_processing.backend;
    if ov.auto_backend && !ov.json {
        eprintln!(
            "denoize: auto-selected backend {}",
            service::backend_name(backend)
        );
    }
    if !ov.json && resolved_processing.accelerator.requested() != AcceleratorPreference::Cpu {
        eprintln!(
            "denoize: accelerator {}",
            accelerator_description(resolved_processing.accelerator)
        );
    }

    if ov.report {
        print_report(
            input,
            &audio,
            &resolved_processing.denoiser,
            backend,
            resolved_processing.accelerator,
        );
        return Ok(None);
    }

    if let Some(format) = output_format {
        format.validate_config(&audio, &encode_options)?;
    }

    let needs_session_reservation = pre_prepared_backend_session.is_none();
    let metadata_bytes = metadata
        .as_ref()
        .map(denoize::metadata::Metadata::estimated_memory_bytes)
        .unwrap_or(0);
    let worker_request = worker_resource_request(
        input_bytes,
        &audio,
        metadata_bytes,
        if capture_execution_evidence {
            decode_limits.max_working_set_bytes
        } else {
            None
        },
        &resolved_processing,
        output_format.is_some(),
    )?;
    let request = if needs_session_reservation {
        worker_request.checked_add(backend_session_request(&resolved_processing)?)?
    } else {
        worker_request
    };
    let resource_permit = governor
        .map(|governor| governor.acquire(request))
        .transpose()?;

    let captured_model = if capture_execution_evidence {
        batch_resume::consumed_model(&resolved_processing)?
    } else {
        None
    };
    let backend_session = match pre_prepared_backend_session {
        Some(session) => session,
        None => Arc::new(BackendSession::prepare_with_accelerator(
            resolved_processing.backend,
            resolved_processing.backend_options.clone(),
            resolved_processing.accelerator,
        )?),
    };
    let result = service::process_audio_resolved_with_session(
        &mut audio,
        &resolved_processing,
        &backend_session,
    )?;
    if let Some(report) = result.loudness {
        if !ov.json {
            eprintln!(
                "denoize: loudness {:.2} -> {:.2} LUFS, true peak {:.2} dBTP, gain {:+.2} dB",
                report.input_lufs, report.output_lufs, report.true_peak_dbtp, report.gain_db
            );
        }
    } else if ov.true_peak_dbtp.is_some() {
        return Err("--true-peak requires --loudness".into());
    }
    if let Some(expected) = captured_input_fingerprint {
        let observed = batch_resume::fingerprint_input_session(
            input_session
                .as_mut()
                .ok_or("receipt input session closed before the final source fence")?,
        )?;
        if observed != expected {
            return Err(format!(
                "input changed while receipt-bound processing was running: {}",
                input.display()
            ));
        }
        let observed_path = batch_resume::fingerprint_file(input)?;
        if observed_path != expected {
            return Err(format!(
                "input path changed while receipt-bound processing was running: {}",
                input.display()
            ));
        }
    }
    if let Some(model) = &captured_model {
        let observed = batch_resume::fingerprint_file(&model.path)?;
        if observed != model.fingerprint {
            return Err(format!(
                "selected backend model changed while receipt-bound processing was running: {}",
                model.path.display()
            ));
        }
    }
    if standard_output {
        let bytes = write_wav_bytes(&audio)?;
        std::io::Write::write_all(&mut std::io::stdout(), &bytes)
            .map_err(|error| format!("failed to write stdout: {error}"))?;
        Ok(None)
    } else {
        let output_format = output_format.ok_or("filesystem output format is missing")?;
        let recipe_model = if recipe_metadata_policy.is_some() {
            if capture_execution_evidence {
                captured_model.clone()
            } else {
                batch_resume::consumed_model(&resolved_processing)?
            }
        } else {
            None
        };
        let effective_recipe = recipe_metadata_policy
            .map(|metadata_policy| {
                batch_resume::recipe_digest(
                    &resolved_processing,
                    audio.channels(),
                    output_format,
                    encode_options,
                    metadata_policy,
                    recipe_model
                        .as_ref()
                        .map(|model| (&model.fingerprint, model.sample_rate)),
                )
            })
            .transpose()?;
        let mut transaction = AtomicOutput::new(output)?;
        denoize::encode::write_audio_to_file(
            transaction.file_mut(),
            output_format,
            &audio,
            encode_options,
        )?;
        if let Some(metadata) = metadata {
            denoize::metadata::write_extended_to_file_with_limits(
                metadata,
                transaction.file_mut(),
                metadata_limits,
            )?;
        }
        let staged_bytes = transaction
            .file_mut()
            .metadata()
            .map_err(|error| format!("inspect staged output: {error}"))?
            .len();
        if staged_bytes > worker_request.temporary_bytes() {
            return Err(format!(
                "staged output requires {staged_bytes} bytes, exceeding its {}-byte temporary reservation",
                worker_request.temporary_bytes()
            ));
        }
        Ok(Some(StagedProcessOutput {
            transaction,
            _resource_permit: resource_permit,
            effective_recipe,
            backend: result.backend,
            accelerator: result.accelerator,
            channels: audio.channels(),
            frames: audio.frames(),
            sample_rate: audio.sample_rate,
            elapsed_ms: result.elapsed.as_secs_f64() * 1_000.0,
            execution_evidence: if capture_execution_evidence {
                Some(StagedExecutionEvidence {
                    input_fingerprint: captured_input_fingerprint
                        .ok_or("receipt input fingerprint is missing")?,
                    input_probe: captured_input_probe.ok_or("receipt input probe is missing")?,
                    model: captured_model,
                    recipe: effective_recipe.ok_or("receipt recipe is missing")?,
                    output_format,
                    resources: request,
                })
            } else {
                None
            },
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StreamTemporaryReservation {
    total_bytes: u64,
    encoder_auxiliary_bytes: u64,
    checkpoint_limit: Option<u64>,
}

fn virtual_wav_bytes(info: denoize::AudioStreamInfo, frames: u64) -> Result<u64, String> {
    frames
        .checked_mul(u64::from(info.output_spec.channels))
        .and_then(|samples| samples.checked_mul(u64::from(info.output_spec.bits_per_sample / 8)))
        .and_then(|bytes| bytes.checked_add(68))
        .ok_or_else(|| "stream virtual WAV byte count overflows".to_string())
}

fn stream_temporary_reservation_bytes(
    info: denoize::AudioStreamInfo,
    output_format: OutputFormat,
    encode_spec: StreamEncodeSpec,
    encode_options: EncodeOptions,
    encode_limits: StreamEncodeLimits,
    configured_limit: Option<u64>,
    checkpointed: bool,
    two_pass_loudness: bool,
    metadata_allowance_bytes: u64,
) -> Result<StreamTemporaryReservation, String> {
    const MAX_WAV_FILE_BYTES: u64 = u32::MAX as u64 + 8;
    let encoder_auxiliary_bytes = denoize::estimate_stream_encode_temporary_bytes(
        output_format,
        encode_spec,
        encode_options,
        encode_limits,
    )?;
    let staged_output_bytes = denoize::estimate_stream_encode_output_bytes(
        output_format,
        encode_spec,
        encode_options,
        encode_limits,
    )?;

    let Some(frames) = info.total_frames else {
        // Ogg and raw ADTS do not expose their presentation length before the
        // terminal packet has been consumed. A configured cap is therefore
        // the complete transaction budget. Without one, preserve the previous
        // finite 4-GiB staged-file ceiling and additionally reserve the M4A
        // sample-table ceiling plus the checkpoint PCM spool when requested.
        if let Some(limit) = configured_limit {
            let unavailable = encoder_auxiliary_bytes
                .checked_add(metadata_allowance_bytes)
                .ok_or_else(|| "stream temporary reservation overflows".to_string())?;
            if unavailable >= limit {
                return Err(format!(
                    "stream encoder auxiliary data and metadata require {unavailable} bytes, leaving no staged-output capacity under --max-temp-space ({limit} bytes)"
                ));
            }
            return Ok(StreamTemporaryReservation {
                total_bytes: limit,
                encoder_auxiliary_bytes,
                checkpoint_limit: checkpointed.then_some(limit - unavailable),
            });
        }
        if !checkpointed {
            let mut total_bytes = MAX_WAV_FILE_BYTES
                .checked_add(encoder_auxiliary_bytes)
                .and_then(|bytes| bytes.checked_add(metadata_allowance_bytes))
                .ok_or_else(|| "stream temporary reservation overflows".to_string())?;
            if two_pass_loudness {
                let data_limit = MAX_WAV_FILE_BYTES.saturating_sub(68);
                let output_sample_bytes = u64::from(info.output_spec.bits_per_sample / 8);
                let max_samples = data_limit / output_sample_bytes;
                let spool_bytes = max_samples
                    .checked_mul(std::mem::size_of::<f64>() as u64)
                    .ok_or_else(|| "stream loudness spool reservation overflows".to_string())?;
                total_bytes = total_bytes
                    .checked_add(spool_bytes)
                    .ok_or_else(|| "stream loudness temporary reservation overflows".to_string())?;
            }
            return Ok(StreamTemporaryReservation {
                total_bytes,
                encoder_auxiliary_bytes,
                checkpoint_limit: None,
            });
        }
        let data_limit = MAX_WAV_FILE_BYTES.saturating_sub(68);
        let output_sample_bytes = u64::from(info.output_spec.bits_per_sample / 8);
        let max_samples = data_limit / output_sample_bytes;
        let spool_bytes = max_samples
            .checked_mul(std::mem::size_of::<f64>() as u64)
            .ok_or_else(|| "stream checkpoint spool reservation overflows".to_string())?;
        let checkpoint_limit = MAX_WAV_FILE_BYTES
            .checked_add(spool_bytes)
            .ok_or_else(|| "stream checkpoint temporary reservation overflows".to_string());
        let checkpoint_limit = checkpoint_limit?;
        let total_bytes = checkpoint_limit
            .checked_add(encoder_auxiliary_bytes)
            .and_then(|bytes| bytes.checked_add(metadata_allowance_bytes))
            .ok_or_else(|| "stream checkpoint temporary reservation overflows".to_string())?;
        return Ok(StreamTemporaryReservation {
            total_bytes,
            encoder_auxiliary_bytes,
            checkpoint_limit: Some(checkpoint_limit),
        });
    };
    let staged_output_bytes = staged_output_bytes
        .ok_or_else(|| "known stream duration has no staged-output estimate".to_string())?;
    let base_bytes = staged_output_bytes
        .checked_add(encoder_auxiliary_bytes)
        .and_then(|bytes| bytes.checked_add(metadata_allowance_bytes))
        .ok_or_else(|| "stream output file size overflows".to_string())?;
    if !checkpointed {
        let total_bytes = if two_pass_loudness {
            let spool_bytes = frames
                .checked_mul(u64::from(info.output_spec.channels))
                .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>() as u64))
                .ok_or_else(|| "stream loudness spool reservation overflows".to_string())?;
            base_bytes
                .checked_add(spool_bytes)
                .ok_or_else(|| "stream loudness temporary reservation overflows".to_string())?
        } else {
            base_bytes
        };
        if configured_limit.is_some_and(|limit| total_bytes > limit) {
            return Err(format!(
                "staged stream output, encoder auxiliary data, metadata, and multi-pass PCM require {total_bytes} bytes, exceeding --max-temp-space ({} bytes)",
                configured_limit.unwrap_or(0)
            ));
        }
        return Ok(StreamTemporaryReservation {
            total_bytes,
            encoder_auxiliary_bytes,
            checkpoint_limit: None,
        });
    }
    let spool_bytes = frames
        .checked_mul(u64::from(info.output_spec.channels))
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>() as u64))
        .ok_or_else(|| "stream checkpoint spool reservation overflows".to_string())?;
    let total_bytes = base_bytes
        .checked_add(spool_bytes)
        .ok_or_else(|| "stream checkpoint temporary reservation overflows".to_string())?;
    if configured_limit.is_some_and(|limit| total_bytes > limit) {
        return Err(format!(
            "stream checkpoint, staged output, encoder auxiliary data, and metadata require {total_bytes} bytes, exceeding --max-temp-space ({} bytes)",
            configured_limit.unwrap_or(0)
        ));
    }
    // The checkpoint implementation historically models its staged peer as a
    // WAV of the same PCM geometry. Give it that virtual bound plus the exact
    // spool bound; the process-wide permit separately reserves the selected
    // encoded format's bound above.
    let checkpoint_limit = virtual_wav_bytes(info, frames)?
        .checked_add(spool_bytes)
        .ok_or_else(|| "stream checkpoint temporary reservation overflows".to_string())?;
    Ok(StreamTemporaryReservation {
        total_bytes,
        encoder_auxiliary_bytes,
        checkpoint_limit: Some(checkpoint_limit),
    })
}

fn replay_stream_checkpoint(
    reader: &mut AudioStreamReader,
    processor: &mut StreamingBackendSession,
    block_frames: usize,
    checkpoint: batch_resume::StreamCheckpoint,
    channels: usize,
) -> Result<u64, String> {
    let mut digest = batch_resume::StreamPcmDigest::new(channels)?;
    let mut input_frames = 0_u64;
    while input_frames < checkpoint.input_frames() {
        if CANCELLED.load(Ordering::Relaxed) {
            return Err(
                "streaming cancelled during checkpoint replay; checkpoint preserved".into(),
            );
        }
        let block = reader
            .next_block(block_frames)?
            .ok_or_else(|| "stream checkpoint extends beyond the input".to_string())?;
        let frames = block.first().map(Vec::len).unwrap_or(0) as u64;
        let next = input_frames
            .checked_add(frames)
            .ok_or_else(|| "stream replay frame count overflows".to_string())?;
        if next > checkpoint.input_frames() {
            return Err(
                "stream checkpoint is not aligned to the configured decoder block boundary".into(),
            );
        }
        let enhanced = processor.process_block(&block)?;
        digest.update(&enhanced)?;
        input_frames = next;
    }
    if digest.frames() != checkpoint.output_frames()
        || digest.len() != checkpoint.spool_len()
        || digest.digest() != checkpoint.spool_digest()
    {
        return Err(
            "replayed stream state does not match the durable checkpoint; use --force to restart"
                .into(),
        );
    }
    Ok(input_frames)
}

fn process_stream_blocks(
    reader: &mut AudioStreamReader,
    processor: &mut StreamingBackendSession,
    block_frames: usize,
    mut write_block: impl FnMut(&[Vec<f64>]) -> Result<(), String>,
) -> Result<usize, String> {
    let mut input_frames = 0usize;
    let mut output_frames = 0usize;
    while let Some(block) = reader.next_block(block_frames)? {
        if CANCELLED.load(Ordering::Relaxed) {
            return Err("streaming cancelled".into());
        }
        let block_frames = block.first().map(Vec::len).unwrap_or(0);
        let enhanced = processor.process_block(&block)?;
        let enhanced_frames = enhanced.first().map(Vec::len).unwrap_or(0);
        write_block(&enhanced)?;
        input_frames = input_frames
            .checked_add(block_frames)
            .ok_or_else(|| "streaming frame count overflows".to_string())?;
        output_frames = output_frames
            .checked_add(enhanced_frames)
            .ok_or_else(|| "streaming output frame count overflows".to_string())?;
    }
    let tail = processor.finish()?;
    let tail_frames = tail.first().map(Vec::len).unwrap_or(0);
    write_block(&tail)?;
    output_frames = output_frames
        .checked_add(tail_frames)
        .ok_or_else(|| "streaming output frame count overflows".to_string())?;
    if output_frames != input_frames {
        return Err(format!(
            "streaming backend produced {output_frames} frames from {input_frames} input frames"
        ));
    }
    Ok(input_frames)
}

fn stream_pcm_spool_limit(
    info: denoize::AudioStreamInfo,
    total_temporary_bytes: u64,
    encoder_auxiliary_bytes: u64,
    metadata_bytes: u64,
) -> Result<u64, String> {
    if let Some(frames) = info.total_frames {
        return frames
            .checked_mul(u64::from(info.output_spec.channels))
            .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>() as u64))
            .ok_or_else(|| "stream loudness PCM spool size overflows".to_string());
    }
    let unavailable = encoder_auxiliary_bytes
        .checked_add(metadata_bytes)
        .ok_or_else(|| "stream loudness temporary allowance overflows".to_string())?;
    let shared = total_temporary_bytes
        .checked_sub(unavailable)
        .ok_or_else(|| "stream loudness has no temporary space for PCM".to_string())?;
    // Reserve at least half for the encoded stage. WAV is at most half the
    // interleaved-f64 PCM size; compressed formats normally need less.
    Ok(shared / 2)
}

fn analyze_stream_pcm_spool(
    spool: &mut StreamPcmSpool,
    spec: hound::WavSpec,
    channel_mask: Option<denoize::ChannelMask>,
    block_frames: usize,
    target_lufs: f64,
    true_peak_dbtp: f64,
) -> Result<denoize::loudness::StreamingLoudnessGain, String> {
    spool.prepare_read()?;
    let mut analyzer = denoize::loudness::StreamingLoudnessAnalyzer::new(
        spec.channels as usize,
        spec.sample_rate,
        channel_mask,
    )?;
    while let Some(block) = spool.next_block(block_frames)? {
        analyzer.add_block(&block)?;
    }
    let gain = analyzer.finish(target_lufs, true_peak_dbtp)?;
    spool.prepare_read()?;
    Ok(gain)
}

fn run_streaming_wav(input: &str, output: &str, ov: Overrides) -> Result<(), String> {
    validate_effective_options(&ov, VALIDATION_SAMPLE_RATE)?;
    let standard_input = input == "-";
    let standard_output = output == "-";
    if (standard_input || standard_output) && ov.resume {
        return Err(
            "--resume requires durable regular-file stream input and output; stdin/stdout spools are intentionally ephemeral"
                .into(),
        );
    }
    if standard_output && ov.json && !ov.report {
        return Err("--json cannot share stdout with encoded audio output".into());
    }
    let stream_receipt = prepare_stream_receipt(input, output, &ov)?;
    let input_path = std::path::Path::new(input);
    let output_path = std::path::Path::new(output);
    let output_format = if standard_output {
        match ov.output_format.as_deref() {
            Some(extension) => {
                let path = std::path::PathBuf::from(format!("output.{extension}"));
                OutputFormat::from_path(&path)?
            }
            None => OutputFormat::Wav,
        }
    } else {
        OutputFormat::from_path(output_path)?
    };
    let receipt_publication = if stream_receipt.is_some() && !ov.resume {
        Some(if standard_output {
            ("stdout", "non-seekable")
        } else {
            planned_publication(output_path, ov.force)?
        })
    } else {
        None
    };
    let encode_options = build_encode_options(&ov)?;
    validate_encode_preflight(encode_options, [output_format])?;
    let resource_governor = resource_governor(&ov, 1)?;
    let configured_temporary_limit = resource_governor.limits().max_temporary_bytes();
    let stdio_spool_limits = StreamSpoolLimits::new(
        configured_temporary_limit.unwrap_or_else(|| StreamSpoolLimits::default().max_bytes()),
    );
    let stdio_request = if standard_input || standard_output {
        ResourceRequest::new().with_temporary_bytes(stdio_spool_limits.max_bytes())
    } else {
        ResourceRequest::new()
    };
    let _stdio_temporary_permit = if standard_input || standard_output {
        Some(resource_governor.acquire(stdio_request)?)
    } else {
        None
    };
    let effective_memory_mb = effective_input_memory_mb(&ov);
    let backend = if ov.auto_backend {
        service::select_live_backend()
    } else {
        ov.backend.unwrap_or(Backend::Classical)
    };
    if !StreamingBackendSession::supports(backend) {
        return Err(format!(
            "backend {} does not support --stream",
            service::backend_name(backend)
        ));
    }
    let backend_options = service::resolve_backend_options(backend, build_backend_options(&ov)?)?;
    let accelerator = denoize::select_accelerator_for_options(backend, &backend_options)?;
    if !ov.resume && !standard_output {
        ensure_output_available(output_path, ov.force)?;
    }

    let mut input_session = if standard_input {
        let stdin = std::io::stdin();
        AudioInputSession::from_reader_with_limits(stdin.lock(), stdio_spool_limits)?
    } else {
        AudioInputSession::open(input_path)?
    };
    let input_spool_bytes = if standard_input {
        input_session.len()
    } else {
        0
    };
    let remaining_stdio_spool_bytes = stdio_spool_limits
        .max_bytes()
        .checked_sub(input_spool_bytes)
        .ok_or_else(|| "stdin spool exceeded the shared non-seekable spool limit".to_string())?;
    let effective_memory_bytes = effective_input_memory_limit_bytes(&ov)?;
    let initial_metadata_limits = metadata_limits_for_available_bytes(effective_memory_bytes);
    let initial_decode_limits = DecodeLimits::new(initial_metadata_limits, effective_memory_bytes);
    let stream_info = inspect_audio_stream_session(&mut input_session, initial_decode_limits)?;
    let spec = stream_info.output_spec;
    let channel_mask = stream_info.channel_mask;
    let encode_spec = StreamEncodeSpec::new(spec, channel_mask, stream_info.total_frames);
    output_format.validate_stream_config(encode_spec, encode_options)?;
    let effective_temporary_limit = if standard_input || standard_output {
        Some(remaining_stdio_spool_bytes)
    } else {
        configured_temporary_limit
    };
    let auxiliary_limit = match (stream_info.total_frames, effective_temporary_limit) {
        (None, Some(limit)) => limit / 3,
        (_, Some(limit)) => limit,
        (_, None) => StreamEncodeLimits::default().max_auxiliary_temporary_bytes(),
    };
    let encode_limits = StreamEncodeLimits::new(auxiliary_limit);
    validate_effective_options(&ov, spec.sample_rate)?;
    let cfg = build_config(&ov, spec.sample_rate);
    let block_frames = ov.stream_frames.unwrap_or(STREAM_BLOCK_FRAMES);
    let base_stream_working_set = estimate_stream_memory_bytes_checked(
        spec.channels as usize,
        block_frames,
        cfg.frame_size,
        spec.sample_rate,
        cfg.profile_ms,
    )
    .map_err(|error| error.to_string())?;
    let backend_stream_state = StreamingBackendSession::estimate_additional_bytes(
        backend,
        spec.sample_rate,
        spec.channels as usize,
        backend_options.channel_mode,
    )
    .map_err(|error| error.to_string())?;
    let vad_stream_state = if cfg.vad {
        StreamingBackendSession::estimate_vad_additional_bytes(
            spec.sample_rate,
            spec.channels as usize,
            block_frames,
            cfg.frame_size,
            cfg.profile_ms,
        )
        .map_err(|error| error.to_string())?
    } else {
        0
    };
    let loudness_stream_state = if ov.loudness_lufs.is_some() {
        denoize::loudness::estimate_streaming_loudness_bytes(
            spec.channels as usize,
            spec.sample_rate,
            block_frames,
        )
        .map_err(|error| error.to_string())?
    } else {
        0
    };
    let encoder_stream_state = denoize::estimate_stream_encode_additional_bytes(
        output_format,
        encode_spec,
        block_frames,
        encode_options,
    )?;
    let checkpoint_scratch = if ov.resume {
        batch_resume::STREAM_CHECKPOINT_SCRATCH_BYTES
    } else {
        0
    };
    let initial_stream_working_set = base_stream_working_set
        .checked_add(backend_stream_state)
        .and_then(|bytes| bytes.checked_add(vad_stream_state))
        .and_then(|bytes| bytes.checked_add(loudness_stream_state))
        .and_then(|bytes| bytes.checked_add(stream_info.decoder_additional_bytes))
        .and_then(|bytes| bytes.checked_add(encoder_stream_state))
        .and_then(|bytes| bytes.checked_add(checkpoint_scratch))
        .and_then(|bytes| {
            bytes.checked_add(denoize::estimate_backend_worker_memory_bytes(
                &backend_options,
            ))
        })
        .ok_or_else(|| "streaming working-set estimate overflow".to_string())?;
    let verification_block_frames = block_frames.min(STREAM_BLOCK_FRAMES);
    let initial_verification_working_set = denoize::estimate_stream_output_verification_bytes(
        output_format,
        encode_spec,
        verification_block_frames,
        encode_options,
        encode_limits,
        initial_decode_limits,
    )?;
    let initial_worker_memory = initial_stream_working_set.max(initial_verification_working_set);
    ensure_memory_limit(
        initial_worker_memory,
        effective_memory_mb,
        "streaming working set",
    )?;
    let metadata_limits = retained_metadata_limits(effective_memory_mb, initial_worker_memory)?;
    let decode_limits = DecodeLimits::new(metadata_limits, effective_memory_bytes);
    let final_stream_info = inspect_audio_stream_session(&mut input_session, decode_limits)?;
    if final_stream_info.format != stream_info.format
        || final_stream_info.codec != stream_info.codec
        || final_stream_info.output_spec != stream_info.output_spec
        || final_stream_info.channel_mask != stream_info.channel_mask
        || final_stream_info.total_frames != stream_info.total_frames
        || final_stream_info.max_decoder_frames != stream_info.max_decoder_frames
    {
        return Err("stream input geometry changed during preflight".into());
    }
    let stream_info = final_stream_info;
    let stream_working_set = base_stream_working_set
        .checked_add(backend_stream_state)
        .and_then(|bytes| bytes.checked_add(vad_stream_state))
        .and_then(|bytes| bytes.checked_add(loudness_stream_state))
        .and_then(|bytes| bytes.checked_add(stream_info.decoder_additional_bytes))
        .and_then(|bytes| bytes.checked_add(encoder_stream_state))
        .and_then(|bytes| bytes.checked_add(checkpoint_scratch))
        .and_then(|bytes| {
            bytes.checked_add(denoize::estimate_backend_worker_memory_bytes(
                &backend_options,
            ))
        })
        .ok_or_else(|| "streaming working-set estimate overflow".to_string())?;
    let verification_working_set = denoize::estimate_stream_output_verification_bytes(
        output_format,
        encode_spec,
        verification_block_frames,
        encode_options,
        encode_limits,
        decode_limits,
    )?;
    ensure_memory_limit(
        stream_working_set.max(verification_working_set),
        effective_memory_mb,
        "streaming working set",
    )?;
    if ov.report {
        if ov.resume {
            ensure_output_available(output_path, ov.force)?;
        }
        println!(
            "input      : {input}\ncontainer  : {:?} / {:?}\nformat     : {}ch, {} Hz, {}-bit {:?}\noutput     : {}\nbackend    : {}\naccelerator: {}\nstream     : enabled ({} frames/block)",
            stream_info.format,
            stream_info.codec,
            spec.channels,
            spec.sample_rate,
            spec.bits_per_sample,
            spec.sample_format,
            output_format_name(output_format),
            service::backend_name(backend),
            accelerator_description(accelerator),
            block_frames
        );
        return Ok(());
    }

    let stream_metadata_policy = if ov.no_metadata {
        MetadataPolicy::Drop
    } else {
        MetadataPolicy::Preserve
    };
    let execution_identity = if ov.resume || stream_receipt.is_some() {
        let input_fingerprint = batch_resume::fingerprint_input_session(&mut input_session)?;
        let resolved = service::ResolvedProcessingOptions {
            backend,
            denoiser: cfg.clone(),
            backend_options: backend_options.clone(),
            accelerator,
            loudness_lufs: ov.loudness_lufs,
            true_peak_dbtp: ov.true_peak_dbtp.unwrap_or(-1.0),
        };
        resolved.validate_config()?;
        let model = if ov.resume {
            batch_resume::resumable_consumed_model(&resolved)?
        } else {
            batch_resume::consumed_model(&resolved)?
        };
        let base_recipe = batch_resume::recipe_digest(
            &resolved,
            spec.channels as usize,
            output_format,
            encode_options,
            stream_metadata_policy,
            model
                .as_ref()
                .map(|model| (&model.fingerprint, model.sample_rate)),
        )?;
        let recipe = batch_resume::stream_recipe_digest(base_recipe, block_frames, stream_info)?;
        Some((input_fingerprint, recipe, model))
    } else {
        None
    };

    let metadata = if stream_metadata_policy == MetadataPolicy::Preserve {
        input_session.read_metadata_with_limits(metadata_limits)?
    } else {
        None
    };
    let metadata_bytes = metadata
        .as_ref()
        .map(denoize::metadata::Metadata::estimated_memory_bytes)
        .unwrap_or(0);
    let encode_phase_memory = stream_working_set
        .checked_add(metadata_bytes)
        .ok_or_else(|| "streaming memory reservation overflow".to_string())?;
    let worker_memory_bytes = encode_phase_memory.max(verification_working_set);
    let temporary_reservation = if standard_output {
        let encoder_auxiliary_bytes = denoize::estimate_stream_encode_temporary_bytes(
            output_format,
            encode_spec,
            encode_options,
            encode_limits,
        )?;
        let total_bytes = denoize::estimate_spooled_stream_output_bytes(
            output_format,
            encode_spec,
            encode_options,
            encode_limits,
        )?;
        let total_bytes = match total_bytes {
            Some(bytes) => bytes
                .checked_add(metadata_bytes)
                .ok_or_else(|| "non-seekable output metadata size overflows".to_string())?,
            None => remaining_stdio_spool_bytes,
        };
        if total_bytes > remaining_stdio_spool_bytes {
            return Err(format!(
                "non-seekable output requires {total_bytes} bytes, but stdin and output share only {} remaining spool bytes",
                remaining_stdio_spool_bytes
            ));
        }
        StreamTemporaryReservation {
            total_bytes,
            encoder_auxiliary_bytes,
            checkpoint_limit: None,
        }
    } else {
        stream_temporary_reservation_bytes(
            stream_info,
            output_format,
            encode_spec,
            encode_options,
            encode_limits,
            effective_temporary_limit,
            ov.resume,
            ov.loudness_lufs.is_some(),
            metadata_bytes,
        )?
    };
    let temporary_bytes = temporary_reservation.total_bytes;
    // Stdio acquired the complete shared spool allowance before reading a byte,
    // so the worker must not reserve the same temporary bytes a second time.
    let admitted_worker_temporary_bytes = if standard_input || standard_output {
        0
    } else {
        temporary_bytes
    };
    let mut worker_request =
        ResourceRequest::worker(worker_memory_bytes, admitted_worker_temporary_bytes);
    if accelerator.effective() != denoize::AcceleratorRuntime::Cpu {
        let gpu_memory = stream_working_set
            .checked_mul(2)
            .and_then(|bytes| {
                bytes.checked_add(denoize::estimate_backend_worker_gpu_memory_bytes(
                    &backend_options,
                ))
            })
            .ok_or_else(|| "streaming GPU reservation overflow".to_string())?;
        worker_request = worker_request
            .with_gpu_jobs(1)
            .with_gpu_memory_bytes(gpu_memory);
    }
    let request = worker_request.checked_add(backend_resource_request(
        backend,
        &backend_options,
        accelerator,
    )?)?;
    let reported_request = request.checked_add(stdio_request)?;
    let _resource_permit = resource_governor.acquire(request)?;
    let stream_evidence = match (&stream_receipt, &execution_identity) {
        (Some(_), Some((input_fingerprint, recipe, model))) => Some(StreamExecutionEvidence {
            input_fingerprint: *input_fingerprint,
            stream_info,
            model: model.clone(),
            recipe: *recipe,
            output_format,
            resources: reported_request,
            backend,
            accelerator,
            deterministic: backend_options.deterministic,
            metadata_policy: stream_metadata_policy,
        }),
        (Some(_), None) => {
            return Err("stream receipt evidence is missing after successful preflight".into());
        }
        (None, _) => None,
    };
    let inspected_resume = if ov.resume && stream_receipt.is_some() {
        let (input_fingerprint, recipe, _) = execution_identity
            .as_ref()
            .ok_or("stream receipt resume identity is missing after preflight")?;
        Some(batch_resume::inspect_stream_checkpoint_decision(
            output_path,
            *input_fingerprint,
            *recipe,
            spec,
            block_frames,
            temporary_reservation.checkpoint_limit,
            ov.force,
        )?)
    } else {
        None
    };

    // Construct every allocation-sensitive processor before opening the
    // transactional output. Invalid or hostile resource plans therefore leave
    // neither a destination nor a temporary `.part` file behind.
    let mut processor = StreamingBackendSession::new_with_accelerator(
        backend,
        spec.sample_rate,
        spec.channels as usize,
        cfg,
        backend_options,
        accelerator,
    )?;
    debug_assert_eq!(processor.accelerator(), accelerator);
    let mut reader = AudioStreamReader::from_session(input_session, decode_limits)?;
    let commit_mode = if ov.force {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    };
    let mut streaming_loudness = None;
    let frames = if ov.resume {
        let (input_fingerprint, recipe, model) = execution_identity
            .clone()
            .ok_or("stream resume identity is missing after successful preflight")?;
        if let Some(model) = model.as_ref() {
            let current = batch_resume::fingerprint_file(&model.path)?;
            if current != model.fingerprint {
                return Err(format!(
                    "selected streaming model changed while it was prepared: {}",
                    model.path.display()
                ));
            }
        }
        let acquired = batch_resume::StreamCheckpointSession::acquire(
            output_path,
            input_fingerprint,
            recipe,
            spec,
            block_frames,
            temporary_reservation.checkpoint_limit,
            ov.force,
        )?;
        match acquired {
            batch_resume::StreamCheckpointAcquire::Completed(completed) => {
                if completed.input_frames() != completed.output_frames() {
                    return Err(
                        "completed stream checkpoint has mismatched input/output length".into(),
                    );
                }
                verify_stream_receipt_sources(&reader, input, stream_evidence.as_ref())?;
                if let (Some(receipt), Some(evidence)) = (&stream_receipt, &stream_evidence) {
                    let output_fingerprint = batch_resume::fingerprint_file(output_path)?;
                    let mut skipped_evidence = evidence.clone();
                    skipped_evidence.resources = ResourceRequest::new();
                    let staged = stage_stream_signed_receipt(
                        input,
                        output,
                        receipt,
                        &skipped_evidence,
                        completed.input_frames(),
                        output_fingerprint,
                        "none",
                        "skip",
                        "completed",
                        Some(output_fingerprint),
                    )?;
                    commit_stream_receipt_after_output(receipt, staged, output)?;
                }
                usize::try_from(completed.input_frames())
                    .map_err(|_| "streaming frame count does not fit this platform".to_string())?
            }
            batch_resume::StreamCheckpointAcquire::Active(mut checkpoint, loaded) => {
                let loaded_checkpoint = loaded;
                let mut input_frames = match loaded {
                    Some(saved) => replay_stream_checkpoint(
                        &mut reader,
                        &mut processor,
                        block_frames,
                        saved,
                        spec.channels as usize,
                    )?,
                    None => 0,
                };
                let checkpoint_frames = stream_checkpoint_frames();
                let mut next_checkpoint = input_frames
                    .checked_div(checkpoint_frames)
                    .and_then(|multiple| multiple.checked_add(1))
                    .and_then(|multiple| multiple.checked_mul(checkpoint_frames))
                    .unwrap_or(u64::MAX);
                while let Some(block) = reader.next_block(block_frames)? {
                    if CANCELLED.load(Ordering::Relaxed) {
                        return Err("streaming cancelled; checkpoint preserved".into());
                    }
                    let decoded_frames = block.first().map(Vec::len).unwrap_or(0) as u64;
                    let enhanced = processor.process_block(&block)?;
                    checkpoint.append_block(&enhanced)?;
                    input_frames = input_frames
                        .checked_add(decoded_frames)
                        .ok_or_else(|| "streaming input frame count overflows".to_string())?;
                    if input_frames >= next_checkpoint {
                        checkpoint.checkpoint(input_frames)?;
                        denoize::fault_injection::hit("stream-checkpoint.after-periodic-sync")?;
                        denoize::ipc::check_process_control_boundary()?;
                        if injected_stop_after_stream_checkpoint() {
                            return Err("injected stop after durable stream checkpoint".into());
                        }
                        next_checkpoint = input_frames
                            .checked_div(checkpoint_frames)
                            .and_then(|multiple| multiple.checked_add(1))
                            .and_then(|multiple| multiple.checked_mul(checkpoint_frames))
                            .unwrap_or(u64::MAX);
                    }
                }
                let tail = processor.finish()?;
                checkpoint.append_block(&tail)?;
                let final_fingerprint = reader.fingerprint_input()?;
                if final_fingerprint != input_fingerprint {
                    return Err(
                        "stream input changed while it was being processed; checkpoint preserved"
                            .into(),
                    );
                }

                let output_frames = checkpoint.output_frames();
                if output_frames != input_frames {
                    return Err(format!(
                        "streaming backend produced {output_frames} frames from {input_frames} input frames"
                    ));
                }
                verify_stream_receipt_sources(&reader, input, stream_evidence.as_ref())?;
                drop(reader);
                drop(processor);

                checkpoint.prepare_spool_read()?;
                let loudness_gain = if let Some(target_lufs) = ov.loudness_lufs {
                    let mut analyzer = denoize::loudness::StreamingLoudnessAnalyzer::new(
                        spec.channels as usize,
                        spec.sample_rate,
                        channel_mask,
                    )?;
                    while let Some(block) = checkpoint.next_spool_block(block_frames)? {
                        analyzer.add_block(&block)?;
                    }
                    checkpoint.prepare_spool_read()?;
                    Some(analyzer.finish(target_lufs, ov.true_peak_dbtp.unwrap_or(-1.0))?)
                } else {
                    None
                };
                let mut final_encode_spec = encode_spec;
                final_encode_spec.total_frames = Some(output_frames);
                let mut transaction = AtomicOutput::new(output_path)?;
                {
                    let mut writer = AudioStreamWriter::new_with_limits(
                        transaction.file_mut(),
                        output_format,
                        final_encode_spec,
                        encode_options,
                        encode_limits,
                    )?;
                    while let Some(mut block) = checkpoint.next_spool_block(block_frames)? {
                        if let Some(gain) = loudness_gain {
                            gain.apply(&mut block);
                        }
                        writer.write_block(&block)?;
                    }
                    writer.finalize()?;
                }
                if output_format == OutputFormat::Wav {
                    write_wav_channel_mask_to_file(
                        transaction.file_mut(),
                        spec.channels as usize,
                        channel_mask,
                    )?;
                }
                if let Some(metadata) = metadata {
                    denoize::metadata::write_extended_to_file_with_limits(
                        metadata,
                        transaction.file_mut(),
                        metadata_limits,
                    )?;
                }
                streaming_loudness = loudness_gain.map(|gain| gain.report());
                let staged_bytes = transaction
                    .file_mut()
                    .metadata()
                    .map_err(|error| format!("inspect staged stream output: {error}"))?
                    .len();
                let combined_bytes = staged_bytes
                    .checked_add(checkpoint.spool_len())
                    .and_then(|bytes| {
                        bytes.checked_add(temporary_reservation.encoder_auxiliary_bytes)
                    })
                    .ok_or_else(|| {
                        "stream checkpoint temporary byte count overflows".to_string()
                    })?;
                if combined_bytes > temporary_bytes {
                    return Err(format!(
                        "checkpoint spool and staged output require {combined_bytes} bytes, exceeding their {temporary_bytes}-byte temporary reservation"
                    ));
                }
                inject_stream_output_corruption(transaction.file_mut())?;
                verify_stream_output_file(
                    transaction.file_mut(),
                    output_path,
                    output_format,
                    final_encode_spec,
                    output_frames,
                    encode_options,
                    decode_limits,
                    verification_block_frames,
                )?;
                let output_fingerprint =
                    batch_resume::fingerprint_open_file_at(transaction.file_mut(), output_path)?;
                let staged_receipt = match (&stream_receipt, &stream_evidence) {
                    (Some(receipt), Some(evidence)) => {
                        let (publication, publication_reason) =
                            planned_publication(output_path, ov.force)?;
                        let reset = inspected_resume.is_some_and(|decision| decision.reset());
                        let reason = if reset {
                            "forced"
                        } else if loaded_checkpoint.is_some() {
                            "checkpoint"
                        } else {
                            publication_reason
                        };
                        Some(stage_stream_signed_receipt(
                            input,
                            output,
                            receipt,
                            evidence,
                            input_frames,
                            output_fingerprint,
                            publication,
                            "process",
                            reason,
                            None,
                        )?)
                    }
                    (None, None) => None,
                    _ => return Err("stream receipt state changed after preflight".into()),
                };
                checkpoint.prepare_publish(input_frames, output_fingerprint)?;
                denoize::fault_injection::hit("stream-checkpoint.after-prepare-publish-sync")?;
                transaction.commit(commit_mode)?;
                if let Err(error) =
                    denoize::fault_injection::hit("stream-checkpoint.after-output-publish")
                {
                    return Err(format!(
                        "stream output was committed before fault injection: {error}"
                    ));
                }
                if injected_stop_after_stream_commit() {
                    return Err("injected stop after committed stream output".into());
                }
                if let (Some(receipt), Some(staged)) = (&stream_receipt, staged_receipt) {
                    commit_stream_receipt_after_output(receipt, staged, output)?;
                    if let Err(error) =
                        denoize::fault_injection::hit("stream-checkpoint.after-receipt-publish")
                    {
                        return Err(format!(
                            "stream output and receipt were committed before fault injection: {error}"
                        ));
                    }
                }
                denoize::fault_injection::hit("stream-checkpoint.before-cleanup")?;
                if let Err(error) = checkpoint.cleanup() {
                    eprintln!(
                        "denoize: warning: output committed but checkpoint cleanup failed: {error}"
                    );
                }
                usize::try_from(input_frames)
                    .map_err(|_| "streaming frame count does not fit this platform".to_string())?
            }
        }
    } else if standard_output {
        let stdout = std::io::stdout();
        let mut writer = SpooledAudioStreamWriter::new_with_limits(
            stdout.lock(),
            output_format,
            encode_spec,
            encode_options,
            encode_limits,
            decode_limits,
            StreamSpoolLimits::new(temporary_bytes),
            verification_block_frames,
        )?;
        let frames = process_stream_blocks(&mut reader, &mut processor, block_frames, |block| {
            writer.write_block(block)
        })?;
        verify_stream_receipt_sources(&reader, input, stream_evidence.as_ref())?;
        drop(reader);
        drop(processor);
        // `process_stream_blocks` validates the backend presentation length
        // before finalize publishes the first caller-visible byte.
        let (_stdout, output_fingerprint, loudness_report) = writer
            .finalize_with_metadata_and_loudness(
                metadata,
                metadata_limits,
                ov.loudness_lufs
                    .map(|target| (target, ov.true_peak_dbtp.unwrap_or(-1.0))),
            )?;
        streaming_loudness = loudness_report;
        if let (Some(receipt), Some(evidence)) = (&stream_receipt, &stream_evidence) {
            let (publication, reason) = receipt_publication
                .ok_or("stdout stream receipt publication is missing after preflight")?;
            let staged = stage_stream_signed_receipt(
                input,
                output,
                receipt,
                evidence,
                frames as u64,
                output_fingerprint,
                publication,
                "process",
                reason,
                None,
            )
            .map_err(|error| {
                format!(
                    "stream output was published to stdout, but its signed receipt could not be staged: {error}"
                )
            })?;
            commit_stream_receipt_after_output(receipt, staged, "stdout")?;
        }
        frames
    } else if let Some(target_lufs) = ov.loudness_lufs {
        let spool_limit = stream_pcm_spool_limit(
            stream_info,
            temporary_bytes,
            temporary_reservation.encoder_auxiliary_bytes,
            metadata_bytes,
        )?;
        let mut spool = StreamPcmSpool::new(spec.channels as usize, spool_limit)?;
        let frames = process_stream_blocks(&mut reader, &mut processor, block_frames, |block| {
            spool.write_block(block)
        })?;
        verify_stream_receipt_sources(&reader, input, stream_evidence.as_ref())?;
        drop(reader);
        drop(processor);
        let loudness_gain = analyze_stream_pcm_spool(
            &mut spool,
            spec,
            channel_mask,
            block_frames,
            target_lufs,
            ov.true_peak_dbtp.unwrap_or(-1.0),
        )?;
        let mut final_encode_spec = encode_spec;
        final_encode_spec.total_frames = Some(frames as u64);
        let mut transaction = AtomicOutput::new(output_path)?;
        {
            let mut writer = AudioStreamWriter::new_with_limits(
                transaction.file_mut(),
                output_format,
                final_encode_spec,
                encode_options,
                encode_limits,
            )?;
            while let Some(mut block) = spool.next_block(block_frames)? {
                loudness_gain.apply(&mut block);
                writer.write_block(&block)?;
            }
            writer.finalize()?;
        }
        if output_format == OutputFormat::Wav {
            write_wav_channel_mask_to_file(
                transaction.file_mut(),
                spec.channels as usize,
                channel_mask,
            )?;
        }
        if let Some(metadata) = metadata {
            denoize::metadata::write_extended_to_file_with_limits(
                metadata,
                transaction.file_mut(),
                metadata_limits,
            )?;
        }
        let staged_bytes = transaction
            .file_mut()
            .metadata()
            .map_err(|error| format!("inspect staged stream output: {error}"))?
            .len();
        let combined_bytes = staged_bytes
            .checked_add(spool.len())
            .and_then(|bytes| bytes.checked_add(temporary_reservation.encoder_auxiliary_bytes))
            .ok_or_else(|| "stream loudness temporary byte count overflows".to_string())?;
        if combined_bytes > temporary_bytes {
            return Err(format!(
                "loudness PCM spool, staged output, and encoder auxiliary data require {combined_bytes} bytes, exceeding their {temporary_bytes}-byte temporary reservation"
            ));
        }
        inject_stream_output_corruption(transaction.file_mut())?;
        verify_stream_output_file(
            transaction.file_mut(),
            output_path,
            output_format,
            final_encode_spec,
            frames as u64,
            encode_options,
            decode_limits,
            verification_block_frames,
        )?;
        let staged_receipt = match (&stream_receipt, &stream_evidence) {
            (Some(receipt), Some(evidence)) => {
                let (publication, reason) = receipt_publication
                    .ok_or("stream receipt publication is missing after preflight")?;
                let output_fingerprint =
                    batch_resume::fingerprint_open_file_at(transaction.file_mut(), output_path)?;
                Some(stage_stream_signed_receipt(
                    input,
                    output,
                    receipt,
                    evidence,
                    frames as u64,
                    output_fingerprint,
                    publication,
                    "process",
                    reason,
                    None,
                )?)
            }
            (None, None) => None,
            _ => return Err("stream receipt state changed after preflight".into()),
        };
        transaction.commit(commit_mode)?;
        if let (Some(receipt), Some(staged)) = (&stream_receipt, staged_receipt) {
            commit_stream_receipt_after_output(receipt, staged, output)?;
        }
        streaming_loudness = Some(loudness_gain.report());
        frames
    } else {
        let mut transaction = AtomicOutput::new(output_path)?;
        let frames = (|| -> Result<usize, String> {
            let mut writer = AudioStreamWriter::new_with_limits(
                transaction.file_mut(),
                output_format,
                encode_spec,
                encode_options,
                encode_limits,
            )?;
            let frames =
                process_stream_blocks(&mut reader, &mut processor, block_frames, |block| {
                    writer.write_block(block)
                })?;
            writer.finalize()?;
            Ok(frames)
        })()?;
        verify_stream_receipt_sources(&reader, input, stream_evidence.as_ref())?;
        drop(reader);
        drop(processor);
        if output_format == OutputFormat::Wav {
            write_wav_channel_mask_to_file(
                transaction.file_mut(),
                spec.channels as usize,
                channel_mask,
            )?;
        }
        if let Some(metadata) = metadata {
            denoize::metadata::write_extended_to_file_with_limits(
                metadata,
                transaction.file_mut(),
                metadata_limits,
            )?;
        }
        let staged_bytes = transaction
            .file_mut()
            .metadata()
            .map_err(|error| format!("inspect staged stream output: {error}"))?
            .len();
        let combined_bytes = staged_bytes
            .checked_add(temporary_reservation.encoder_auxiliary_bytes)
            .ok_or_else(|| "stream temporary byte count overflows".to_string())?;
        if combined_bytes > temporary_bytes {
            return Err(format!(
                "staged stream output and encoder auxiliary data require {combined_bytes} bytes, exceeding their {temporary_bytes}-byte temporary reservation"
            ));
        }
        inject_stream_output_corruption(transaction.file_mut())?;
        verify_stream_output_file(
            transaction.file_mut(),
            output_path,
            output_format,
            encode_spec,
            frames as u64,
            encode_options,
            decode_limits,
            verification_block_frames,
        )?;
        let staged_receipt = match (&stream_receipt, &stream_evidence) {
            (Some(receipt), Some(evidence)) => {
                let (publication, reason) = receipt_publication
                    .ok_or("stream receipt publication is missing after preflight")?;
                let output_fingerprint =
                    batch_resume::fingerprint_open_file_at(transaction.file_mut(), output_path)?;
                Some(stage_stream_signed_receipt(
                    input,
                    output,
                    receipt,
                    evidence,
                    frames as u64,
                    output_fingerprint,
                    publication,
                    "process",
                    reason,
                    None,
                )?)
            }
            (None, None) => None,
            _ => return Err("stream receipt state changed after preflight".into()),
        };
        transaction.commit(commit_mode)?;
        if let (Some(receipt), Some(staged)) = (&stream_receipt, staged_receipt) {
            commit_stream_receipt_after_output(receipt, staged, output)?;
        }
        frames
    };
    if let Some(report) = streaming_loudness {
        eprintln!(
            "denoize: loudness {:.2} -> {:.2} LUFS, true peak {:.2} dBTP, gain {:+.2} dB",
            report.input_lufs, report.output_lufs, report.true_peak_dbtp, report.gain_db
        );
    }
    if ov.json {
        println!(
            "{}",
            stream_result_json_line(
                input,
                output,
                service::backend_name(backend),
                accelerator,
                spec.channels,
                frames,
                spec.sample_rate,
            )
        );
    } else {
        if accelerator.requested() != AcceleratorPreference::Cpu {
            eprintln!(
                "denoize: accelerator {}",
                accelerator_description(accelerator)
            );
        }
        eprintln!(
            "denoize: streaming {} {} complete: {}ch x {} frames",
            service::backend_name(backend),
            output_format_name(output_format),
            spec.channels,
            frames
        );
    }
    Ok(())
}

enum BatchFileOutcome {
    Completed(Option<FileFingerprint>),
    Skipped(FileFingerprint),
    Failed(String),
    Cancelled,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct BatchCounts {
    succeeded: usize,
    skipped: usize,
    failed: usize,
    cancelled: usize,
}

fn count_batch_results(results: &[BatchFileOutcome]) -> BatchCounts {
    let mut counts = BatchCounts::default();
    for result in results {
        match result {
            BatchFileOutcome::Completed(_) => counts.succeeded += 1,
            BatchFileOutcome::Skipped(_) => counts.skipped += 1,
            BatchFileOutcome::Failed(_) => counts.failed += 1,
            BatchFileOutcome::Cancelled => counts.cancelled += 1,
        }
    }
    counts
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BatchItem {
    input: std::path::PathBuf,
    input_relative: std::path::PathBuf,
    destination: std::path::PathBuf,
    destination_relative: std::path::PathBuf,
    output_format: OutputFormat,
    probe: AudioProbe,
}

#[derive(Clone)]
struct PreparedBatchItem {
    item: BatchItem,
    resolved_processing: service::ResolvedProcessingOptions,
    backend_session: Option<GovernedBackendSession>,
    resource_request: ResourceRequest,
    expectation: ResumeExpectation,
    recipe: Digest,
    channels: usize,
    frames: u64,
    sample_rate: u32,
}

#[derive(Clone)]
struct PlannedBatchItem {
    prepared: PreparedBatchItem,
    decision: ResumeDecision,
    existing_output: Option<batch_resume::FileFingerprint>,
}

fn batch_probe_description(probe: &AudioProbe) -> &'static str {
    if probe.is_broadcast_wave {
        return "Broadcast Wave (BWF) PCM";
    }
    match (probe.format, probe.codec) {
        (AudioFormat::Wav, AudioCodec::Pcm) => "WAV PCM",
        (AudioFormat::Rf64, AudioCodec::Pcm) => "RF64 PCM",
        (AudioFormat::Aiff, AudioCodec::Pcm) => "AIFF/AIFC",
        (AudioFormat::Caf, AudioCodec::Pcm) => "CAF",
        (AudioFormat::Flac, AudioCodec::Flac) => "FLAC",
        (AudioFormat::OggOpus, AudioCodec::Opus) => "Ogg Opus",
        (AudioFormat::OggVorbis, AudioCodec::Vorbis) => "Ogg Vorbis",
        (AudioFormat::Mp3, AudioCodec::Mp3) => "MP3",
        (AudioFormat::M4a, AudioCodec::Aac) => "AAC-in-MP4",
        (AudioFormat::M4a, AudioCodec::Alac) => "ALAC-in-MP4",
        (AudioFormat::AacAdts, AudioCodec::Aac) => "ADTS AAC",
        _ => "unknown or ambiguous audio encoding",
    }
}

fn batch_can_preserve(probe: &AudioProbe, output_format: OutputFormat) -> bool {
    probe.audio_tracks == 1
        && !probe.has_non_audio_tracks
        && !probe.is_broadcast_wave
        && matches!(
            (probe.format, probe.codec, output_format),
            (AudioFormat::Wav, AudioCodec::Pcm, OutputFormat::Wav)
                | (AudioFormat::Flac, AudioCodec::Flac, OutputFormat::Flac)
                | (
                    AudioFormat::OggOpus,
                    AudioCodec::Opus,
                    OutputFormat::OggOpus
                )
                | (AudioFormat::Mp3, AudioCodec::Mp3, OutputFormat::Mp3)
                | (AudioFormat::M4a, AudioCodec::Aac, OutputFormat::M4a)
                | (AudioFormat::AacAdts, AudioCodec::Aac, OutputFormat::AacAdts)
        )
}

#[cfg(test)]
fn plan_batch_files(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
    files: Vec<std::path::PathBuf>,
    output_extension: Option<&str>,
) -> Result<Vec<BatchItem>, String> {
    plan_batch_files_with_limits(
        input_dir,
        output_dir,
        files,
        output_extension,
        DecodeLimits::default(),
    )
}

fn plan_batch_files_with_limits(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
    files: Vec<std::path::PathBuf>,
    output_extension: Option<&str>,
    decode_limits: DecodeLimits,
) -> Result<Vec<BatchItem>, String> {
    let mut items = Vec::with_capacity(files.len());
    for input in files {
        let relative = input
            .strip_prefix(input_dir)
            .map_err(|error| {
                format!(
                    "batch input {} is outside {}: {error}",
                    input.display(),
                    input_dir.display()
                )
            })?
            .to_path_buf();
        let mut destination = output_dir.join(&relative);
        if let Some(extension) = output_extension {
            destination.set_extension(extension);
        }

        let mut input_session = AudioInputSession::open(&input)
            .map_err(|error| format!("open batch input {}: {error}", input.display()))?;
        let probe = probe_audio_session_with_limits(&mut input_session, decode_limits)
            .map_err(|error| format!("probe batch input {}: {error}", input.display()))?;
        if probe.audio_tracks != 1 {
            return Err(format!(
                "batch input {} must contain exactly one supported audio track; found {}",
                input.display(),
                probe.audio_tracks
            ));
        }
        if probe.codec == AudioCodec::Unknown {
            return Err(format!(
                "batch input {} has no supported, unambiguous audio track",
                input.display()
            ));
        }
        let output_format = OutputFormat::from_path(&destination).map_err(|error| {
            if output_extension.is_none() {
                format!(
                    "batch cannot preserve {} ({}): {error}; specify --output-format wav, flac, opus, ogg, oga, mp3, m4a, or aac",
                    input.display(),
                    batch_probe_description(&probe)
                )
            } else {
                error
            }
        })?;
        if output_extension.is_none() && !batch_can_preserve(&probe, output_format) {
            let track_detail = if probe.audio_tracks != 1 || probe.has_non_audio_tracks {
                format!(
                    "; source contains {} audio track(s){}",
                    probe.audio_tracks,
                    if probe.has_non_audio_tracks {
                        " and non-audio tracks"
                    } else {
                        ""
                    }
                )
            } else {
                String::new()
            };
            return Err(format!(
                "batch cannot preserve {} ({}) without an explicit conversion{track_detail}; specify --output-format wav, flac, opus, ogg, oga, mp3, m4a, or aac",
                input.display(),
                batch_probe_description(&probe)
            ));
        }
        let destination_relative = destination
            .strip_prefix(output_dir)
            .map_err(|error| {
                format!(
                    "batch output {} is outside {}: {error}",
                    destination.display(),
                    output_dir.display()
                )
            })?
            .to_path_buf();
        items.push(BatchItem {
            input,
            input_relative: relative,
            destination,
            destination_relative,
            output_format,
            probe,
        });
    }
    validate_batch_destinations(input_dir, &items)?;
    Ok(items)
}

fn batch_collision_key(path: &std::path::Path) -> std::path::PathBuf {
    #[cfg(any(windows, target_os = "macos"))]
    {
        std::path::PathBuf::from(path.to_string_lossy().to_lowercase())
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        path.to_path_buf()
    }
}

fn validate_batch_destinations(
    input_dir: &std::path::Path,
    items: &[BatchItem],
) -> Result<(), String> {
    let input_root = normalize_batch_path(input_dir)?;
    let mut destinations = Vec::with_capacity(items.len());
    for item in items {
        let resolved = normalize_batch_path(&item.destination)?;
        if resolved.starts_with(&input_root) {
            return Err(format!(
                "batch output {} resolves inside the input directory; remove output symlinks or choose a separate output directory",
                item.destination.display()
            ));
        }
        destinations.push((batch_collision_key(&resolved), item));
    }
    destinations.sort_by(|left, right| left.0.cmp(&right.0));

    for pair in destinations.windows(2) {
        let (left_path, left) = &pair[0];
        let (right_path, right) = &pair[1];
        if right_path == left_path {
            return Err(format!(
                "multiple inputs map to the same batch output: {} and {} -> {}",
                left.input.display(),
                right.input.display(),
                right.destination.display()
            ));
        }
        if right_path.starts_with(left_path) {
            return Err(format!(
                "batch outputs conflict as a file and directory: {} -> {} and {} -> {}",
                left.input.display(),
                left.destination.display(),
                right.input.display(),
                right.destination.display()
            ));
        }
    }
    Ok(())
}

fn validate_batch_reserved_path(
    items: &[BatchItem],
    reserved: &std::path::Path,
    reserved_name: &str,
) -> Result<(), String> {
    let reserved = batch_collision_key(&normalize_batch_path(reserved)?);
    for item in items {
        let destination = batch_collision_key(&normalize_batch_path(&item.destination)?);
        if destination == reserved
            || destination.starts_with(&reserved)
            || reserved.starts_with(&destination)
        {
            return Err(format!(
                "batch output {} conflicts with reserved batch control path {reserved_name}",
                item.destination.display(),
            ));
        }
    }
    Ok(())
}

fn normalize_batch_path(path: &std::path::Path) -> Result<std::path::PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("resolve current directory: {error}"))?
            .join(path)
    };
    #[derive(Debug)]
    enum MissingComponent {
        Normal(std::ffi::OsString),
        Parent,
    }

    let mut ancestor = absolute.clone();
    let mut missing = Vec::new();
    loop {
        match std::fs::symlink_metadata(&ancestor) {
            Ok(_) => break,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "inspect batch path {}: {error}",
                    ancestor.display()
                ));
            }
        }
        let component = ancestor
            .components()
            .next_back()
            .ok_or_else(|| format!("cannot resolve batch path {}", absolute.display()))?;
        match component {
            std::path::Component::Normal(name) => {
                missing.push(MissingComponent::Normal(name.to_os_string()))
            }
            std::path::Component::ParentDir => missing.push(MissingComponent::Parent),
            std::path::Component::CurDir => {}
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                return Err(format!("cannot resolve batch path {}", absolute.display()));
            }
        }
        if !ancestor.pop() {
            return Err(format!("cannot resolve batch path {}", absolute.display()));
        }
    }
    let mut resolved = std::fs::canonicalize(&ancestor)
        .map_err(|error| format!("resolve {}: {error}", ancestor.display()))?;
    for component in missing.into_iter().rev() {
        match component {
            MissingComponent::Normal(name) => resolved.push(name),
            MissingComponent::Parent => {
                resolved.pop();
            }
        }
    }
    Ok(resolved)
}

fn validate_batch_directories(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
) -> Result<(), String> {
    let input = normalize_batch_path(input_dir)?;
    let output = normalize_batch_path(output_dir)?;
    if input.starts_with(&output) || output.starts_with(&input) {
        return Err(format!(
            "batch input and output directories must not overlap: {} and {}",
            input_dir.display(),
            output_dir.display()
        ));
    }
    Ok(())
}

fn run_batch(input: &str, output: &str, ov: &Overrides) -> Result<(), String> {
    use rayon::prelude::*;

    validate_effective_options(ov, VALIDATION_SAMPLE_RATE)?;
    if ov.report && ov.receipt.is_some() {
        return Err(
            "--receipt cannot be combined with --report because no output is published".into(),
        );
    }
    let encode_options = build_encode_options(ov)?;
    let resolved_backend_options = resolve_explicit_backend_options(ov)?;
    let jobs = effective_batch_jobs(ov);
    let resource_governor = resource_governor(ov, jobs)?;
    let input_dir = std::path::Path::new(input);
    let output_dir = std::path::Path::new(output);
    let receipt_paths = match (&ov.receipt, &ov.receipt_key) {
        (Some(receipt), Some(key)) => Some((
            std::path::PathBuf::from(receipt),
            std::path::PathBuf::from(key),
        )),
        (None, None) => None,
        _ => return Err("--receipt and --receipt-key must be supplied together".into()),
    };
    let signing_key = if let Some((receipt, key)) = &receipt_paths {
        preflight_batch_receipt_paths(input_dir, output_dir, receipt, key)?;
        Some(ReceiptSecretKey::from_file(key)?)
    } else {
        None
    };
    if !input_dir.is_dir() {
        return Err(format!("batch input is not a directory: {input}"));
    }
    validate_batch_directories(input_dir, output_dir)?;
    let output_extension = ov
        .output_format
        .as_deref()
        .map(normalize_output_extension)
        .transpose()?;
    let files = collect_batch_files(input_dir, ov.recursive)?;
    if files.is_empty() {
        return Err("batch input contains no supported audio files".into());
    }
    let items = plan_batch_files_with_limits(
        input_dir,
        output_dir,
        files,
        output_extension,
        decode_limits_for_options(ov)?,
    )?;
    let state_path = output_dir.join(STATE_FILE_NAME);
    let legacy_state_path = output_dir.join(LEGACY_DESKTOP_STATE_FILE_NAME);
    let lock_path = output_dir.join(LOCK_FILE_NAME);
    validate_batch_reserved_path(&items, &state_path, STATE_FILE_NAME)?;
    validate_batch_reserved_path(&items, &legacy_state_path, LEGACY_DESKTOP_STATE_FILE_NAME)?;
    validate_batch_reserved_path(&items, &lock_path, LOCK_FILE_NAME)?;
    if let Some((receipt, key)) = &receipt_paths {
        validate_batch_reserved_path(&items, receipt, "execution receipt")?;
        validate_batch_reserved_path(&items, key, "execution receipt key")?;
    }
    validate_encode_preflight(encode_options, items.iter().map(|item| item.output_format))?;
    let prepared = preflight_batch_items(
        &items,
        ov,
        encode_options,
        resolved_backend_options.as_ref(),
        &resource_governor,
        false,
    )?;

    std::fs::create_dir_all(output_dir).map_err(|e| format!("create batch output: {e}"))?;
    let session = Arc::new(BatchSession::acquire(output_dir, ov.resume)?);
    let planned = prepared
        .into_iter()
        .map(|prepared| {
            let evidence = session.plan_with_evidence(&prepared.expectation, ov.force)?;
            Ok(PlannedBatchItem {
                prepared,
                decision: evidence.decision(),
                existing_output: evidence.existing_output(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    // Planning can be long for a large output set. Recheck every source after
    // the final decision and before activate performs the first state change.
    for item in &planned {
        item.prepared.expectation.verify_sources()?;
    }
    let receipt_plan = receipt_paths
        .as_ref()
        .map(|_| build_batch_execution_plan_from_planned(input_dir, output_dir, ov, &planned))
        .transpose()?;
    let mut staged_receipt = receipt_paths
        .as_ref()
        .map(|(path, _)| AtomicOutput::new(path))
        .transpose()?;
    CANCELLED.store(false, Ordering::SeqCst);
    install_cancel_handler()?;
    let finished = AtomicUsize::new(0);
    let publication_fence = Mutex::new(());
    let started = Instant::now();
    let metadata_policy = if ov.no_metadata {
        MetadataPolicy::Drop
    } else {
        MetadataPolicy::Preserve
    };
    let process_item = |planned: &PlannedBatchItem| -> BatchFileOutcome {
        let item = &planned.prepared.item;
        let finish = |outcome, status| {
            report_batch_progress(
                &finished,
                items.len(),
                started,
                &item.input,
                status,
                planned.prepared.recipe,
                ov,
            );
            outcome
        };
        if let Err(error) = denoize::ipc::check_process_control_boundary() {
            CANCELLED.store(true, Ordering::SeqCst);
            return finish(BatchFileOutcome::Failed(error), "paused");
        }
        let commit_mode = match planned.decision {
            ResumeDecision::Skip { .. } => {
                let Some(fingerprint) = planned.existing_output else {
                    return finish(
                        BatchFileOutcome::Failed(
                            "resume skip is missing its planned output fingerprint".into(),
                        ),
                        "failed",
                    );
                };
                return finish(BatchFileOutcome::Skipped(fingerprint), "skipped");
            }
            ResumeDecision::Process { commit_mode, .. } => commit_mode,
        };
        if CANCELLED.load(Ordering::SeqCst) {
            return finish(BatchFileOutcome::Cancelled, "cancelled");
        }
        let worker_permit = match resource_governor
            .acquire_with_cancel(planned.prepared.resource_request, || {
                CANCELLED.load(Ordering::SeqCst)
            }) {
            Ok(permit) => permit,
            Err(_error) if CANCELLED.load(Ordering::SeqCst) => {
                return finish(BatchFileOutcome::Cancelled, "cancelled");
            }
            Err(error) => return finish(BatchFileOutcome::Failed(error), "failed"),
        };
        if let Some(parent) = item.destination.parent() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                return finish(
                    BatchFileOutcome::Failed(format!("create {}: {error}", parent.display())),
                    "failed",
                );
            }
        }
        let mut options = ov.clone();
        options.batch = false;
        options.json = false;
        let staged = match process_one_to_staged_output(
            &item.input,
            &item.destination,
            options,
            Some(item.output_format),
            None,
            Some(planned.prepared.resolved_processing.clone()),
            Some(metadata_policy),
            Some(item.probe),
            Some(planned.prepared.expectation.input_fingerprint()),
            planned
                .prepared
                .backend_session
                .as_ref()
                .map(|session| Arc::clone(&session.session)),
            false,
            None,
            false,
        ) {
            Ok(staged) => staged,
            Err(error) => return finish(BatchFileOutcome::Failed(error), "failed"),
        };
        let Some(staged) = staged else {
            // Batch --report intentionally retains its existing report-only
            // behavior and has no filesystem output to publish.
            return finish(BatchFileOutcome::Completed(None), "completed");
        };
        if staged.effective_recipe != Some(planned.prepared.recipe) {
            return finish(
                BatchFileOutcome::Failed(format!(
                    "effective batch recipe changed after preflight for {}",
                    item.input.display()
                )),
                "failed",
            );
        }
        if let Err(error) = denoize::ipc::check_process_control_boundary() {
            CANCELLED.store(true, Ordering::SeqCst);
            return finish(BatchFileOutcome::Failed(error), "paused");
        }
        match with_batch_publication_fence(&publication_fence, &CANCELLED, || {
            session.publish(
                &planned.prepared.expectation,
                staged.transaction,
                commit_mode,
            )
        }) {
            Ok(Some(fingerprint)) => {
                drop(worker_permit);
                finish(BatchFileOutcome::Completed(Some(fingerprint)), "completed")
            }
            Ok(None) => finish(BatchFileOutcome::Cancelled, "cancelled"),
            Err(error) => finish(BatchFileOutcome::Failed(error), "failed"),
        }
    };
    let results = if CANCELLED.load(Ordering::SeqCst) {
        // Do not activate (and therefore do not repair or create state) when
        // cancellation was observed before any item could publish. Running
        // the closure still gives every exact skip or cancelled item one
        // stable progress outcome.
        planned.iter().map(process_item).collect::<Vec<_>>()
    } else {
        session.activate()?;
        if ov.deterministic {
            planned.iter().map(process_item).collect::<Vec<_>>()
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(jobs)
                .build()
                .map_err(|e| format!("create batch worker pool: {e}"))?
                .install(|| planned.par_iter().map(process_item).collect::<Vec<_>>())
        }
    };
    let counts = count_batch_results(&results);
    let failures: Vec<_> = results
        .iter()
        .filter_map(|result| match result {
            BatchFileOutcome::Failed(error) => Some(error),
            _ => None,
        })
        .collect();
    debug_assert_eq!(counts.failed, failures.len());
    debug_assert_eq!(
        counts.succeeded + counts.skipped + counts.failed + counts.cancelled,
        items.len()
    );
    if failures.is_empty() && counts.cancelled == 0 {
        if let (Some(plan), Some(key), Some((receipt_path, _)), Some(receipt_stage)) = (
            receipt_plan.as_ref(),
            signing_key.as_ref(),
            receipt_paths.as_ref(),
            staged_receipt.as_mut(),
        ) {
            let receipt_items =
                build_batch_receipt_items(plan, &planned, &results, input_dir, output_dir)?;
            let payload = ExecutionReceiptPayload::new(plan, receipt_items)?;
            let receipt = key.sign(payload)?;
            write_signed_receipt_to_stage(receipt_stage, receipt_path, &receipt)?;
        }
        if let (Some((receipt_path, _)), Some(receipt_stage)) =
            (receipt_paths.as_ref(), staged_receipt.take())
        {
            receipt_stage
                .commit(CommitMode::NoClobber)
                .map_err(|error| {
                    format!(
                        "batch outputs were committed, but their signed receipt could not be published to {}: {error}",
                        receipt_path.display()
                    )
                })?;
        }
    }
    if ov.json {
        println!(
            "{}",
            batch_summary_json_line(
                items.len(),
                counts.succeeded,
                counts.skipped,
                counts.failed,
                counts.cancelled,
                counts.cancelled != 0,
                output,
            )
        );
    } else {
        eprintln!(
            "denoize: batch complete: {} succeeded, {} skipped, {} failed, {} cancelled",
            counts.succeeded, counts.skipped, counts.failed, counts.cancelled
        );
        for error in &failures {
            eprintln!("denoize: batch error: {error}");
        }
    }
    if failures.is_empty() && counts.cancelled == 0 {
        Ok(())
    } else {
        Err(format!(
            "{} batch file(s) failed and {} cancelled",
            failures.len(),
            counts.cancelled
        ))
    }
}

fn build_batch_receipt_items(
    plan: &ExecutionPlan,
    planned: &[PlannedBatchItem],
    results: &[BatchFileOutcome],
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
) -> Result<Vec<ReceiptItem>, String> {
    if results.len() != planned.len() {
        return Err("batch result count does not match its execution plan".into());
    }
    let mut items = Vec::with_capacity(planned.len());
    for (planned_item, result) in planned.iter().zip(results) {
        planned_item.prepared.expectation.verify_sources()?;
        let prepared = &planned_item.prepared;
        let output_locator = denoize::portable_locator(&prepared.item.destination, output_dir)?;
        let item_id = denoize::execution_item_id(
            prepared.expectation.input_fingerprint(),
            &output_locator,
            prepared.recipe,
        )?;
        let index = plan
            .items
            .binary_search_by_key(&item_id, |item| item.item_id)
            .map_err(|_| {
                format!(
                    "completed batch item is absent from its execution plan: {}",
                    prepared.item.input.display()
                )
            })?;
        let plan_item = &plan.items[index];
        let expected_input_locator = denoize::portable_locator(&prepared.item.input, input_dir)?;
        if plan_item.input.path != expected_input_locator || plan_item.output.path != output_locator
        {
            return Err(format!(
                "completed batch path differs from its execution plan: {}",
                prepared.item.input.display()
            ));
        }
        let (fingerprint, outcome) = match result {
            BatchFileOutcome::Completed(Some(fingerprint)) => (*fingerprint, "succeeded"),
            BatchFileOutcome::Completed(None) => {
                return Err("an output-free batch result cannot enter a receipt".into());
            }
            BatchFileOutcome::Skipped(fingerprint) => (*fingerprint, "skipped"),
            BatchFileOutcome::Failed(_) | BatchFileOutcome::Cancelled => {
                return Err("an unsuccessful batch result cannot enter a receipt".into());
            }
        };
        let current = batch_resume::fingerprint_file(&prepared.item.destination)?;
        if current != fingerprint {
            return Err(format!(
                "batch output changed after publication and cannot enter a receipt: {}",
                prepared.item.destination.display()
            ));
        }
        items.push(ReceiptItem::from_plan_item(
            plan_item,
            fingerprint,
            outcome,
        )?);
    }
    Ok(items)
}

fn report_batch_progress(
    finished: &AtomicUsize,
    total: usize,
    started: Instant,
    path: &std::path::Path,
    status: &str,
    recipe: Digest,
    ov: &Overrides,
) {
    let count = finished.fetch_add(1, Ordering::Relaxed) + 1;
    let elapsed = started.elapsed().as_secs_f64();
    let eta = if count == 0 {
        0.0
    } else {
        elapsed / count as f64 * total.saturating_sub(count) as f64
    };
    if ov.json {
        let input = path.to_string_lossy();
        println!(
            "{}",
            batch_progress_json_line(status, count, total, elapsed, eta, input.as_ref(), recipe,)
        );
    } else if !ov.no_progress {
        eprintln!(
            "denoize: batch {count}/{total} {status} {} ({elapsed:.1}s elapsed, ETA {eta:.1}s)",
            path.display()
        );
    }
}

fn collect_batch_files(
    root: &std::path::Path,
    recursive: bool,
) -> Result<Vec<std::path::PathBuf>, String> {
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory)
            .map_err(|e| format!("read batch input {}: {e}", directory.display()))?
        {
            let entry = entry.map_err(|e| format!("read batch entry: {e}"))?;
            let file_type = entry
                .file_type()
                .map_err(|e| format!("read batch entry type: {e}"))?;
            let path = entry.path();
            if file_type.is_dir() && recursive {
                pending.push(path);
            } else if file_type.is_file() && is_supported_audio_path(&path) {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn is_supported_audio_path(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "wav"
                    | "rf64"
                    | "bwf"
                    | "aif"
                    | "aiff"
                    | "aifc"
                    | "caf"
                    | "mp3"
                    | "m4a"
                    | "mp4"
                    | "aac"
                    | "flac"
                    | "opus"
                    | "ogg"
                    | "oga"
                    | "vorbis"
            )
        })
        .unwrap_or(false)
}

fn normalize_output_extension(value: &str) -> Result<&str, String> {
    let extension = value.trim_start_matches('.');
    if matches!(
        extension.to_ascii_lowercase().as_str(),
        "wav" | "mp3" | "m4a" | "aac" | "flac" | "opus" | "ogg" | "oga"
    ) {
        Ok(extension)
    } else {
        Err(format!("unsupported --output-format: {value}"))
    }
}

#[cfg(test)]
mod json_output_tests {
    use super::*;
    use serde_json::Value;

    const SPECIAL_INPUT: &str = "input-cafe\u{301}-quote\"-slash\\-line\n-control\u{1}.wav";
    const SPECIAL_OUTPUT: &str = "output-cafe\u{301}-quote\"-slash\\-line\n-control\u{2}.wav";

    fn parse_json_line(line: &str) -> Value {
        assert!(
            !line.contains("\\u{"),
            "Rust escape leaked into JSON: {line}"
        );
        assert!(
            !line.contains('\n'),
            "serialized JSON line contains a physical newline"
        );
        serde_json::from_str(line).expect("CLI output must be valid JSON")
    }

    #[test]
    fn process_result_json_round_trips_special_paths() {
        let value = parse_json_line(&process_result_json_line(
            SPECIAL_INPUT,
            SPECIAL_OUTPUT,
            "classical",
            AcceleratorSelection::default(),
            2,
            48_001,
            48_000,
            1.2345,
            Some(Digest::from_bytes([7; 32])),
        ));

        assert_eq!(value.as_object().unwrap().len(), 13);
        assert_eq!(value["schema"], CLI_JSON_SCHEMA);
        assert_eq!(value["schema_version"], CLI_JSON_SCHEMA_VERSION);
        assert_eq!(value["event"], "result");
        assert_eq!(value["mode"], "file");
        assert_eq!(value["recipe"]["domain"], RECIPE_DOMAIN);
        assert_eq!(value["recipe"]["version"], RECIPE_VERSION);
        assert_eq!(
            value["recipe"]["output_abi_version"],
            RECIPE_OUTPUT_ABI_VERSION
        );
        assert_eq!(value["recipe"]["digest"], "07".repeat(32));
        assert_eq!(value["input"].as_str(), Some(SPECIAL_INPUT));
        assert_eq!(value["output"].as_str(), Some(SPECIAL_OUTPUT));
        assert_eq!(value["backend"].as_str(), Some("classical"));
        assert_eq!(value["accelerator"]["requested"], "cpu");
        assert_eq!(value["accelerator"]["effective"], "cpu");
        assert!(value["accelerator"]["fallback"].is_null());
        assert_eq!(value["channels"].as_u64(), Some(2));
        assert_eq!(value["frames"].as_u64(), Some(48_001));
        assert_eq!(value["sample_rate"].as_u64(), Some(48_000));
        assert_eq!(value["elapsed_ms"].as_f64(), Some(1.234));
    }

    #[test]
    fn stream_result_json_round_trips_special_paths() {
        let value = parse_json_line(&stream_result_json_line(
            SPECIAL_INPUT,
            SPECIAL_OUTPUT,
            "gtcrn",
            AcceleratorSelection::default(),
            2,
            8_193,
            44_100,
        ));

        assert_eq!(value.as_object().unwrap().len(), 13);
        assert_eq!(value["schema"], CLI_JSON_SCHEMA);
        assert_eq!(value["schema_version"], CLI_JSON_SCHEMA_VERSION);
        assert_eq!(value["event"], "result");
        assert_eq!(value["mode"], "stream");
        assert!(value["recipe"]["digest"].is_null());
        assert_eq!(value["input"].as_str(), Some(SPECIAL_INPUT));
        assert_eq!(value["output"].as_str(), Some(SPECIAL_OUTPUT));
        assert_eq!(value["backend"].as_str(), Some("gtcrn"));
        assert_eq!(value["channels"].as_u64(), Some(2));
        assert_eq!(value["frames"].as_u64(), Some(8_193));
        assert_eq!(value["sample_rate"].as_u64(), Some(44_100));
        assert_eq!(value["stream"].as_bool(), Some(true));
    }

    #[test]
    fn batch_progress_json_round_trips_special_paths() {
        let value = parse_json_line(&batch_progress_json_line(
            "completed",
            3,
            5,
            1.23456,
            0.45678,
            SPECIAL_INPUT,
            Digest::from_bytes([9; 32]),
        ));

        assert_eq!(value.as_object().unwrap().len(), 10);
        assert_eq!(value["schema"], CLI_JSON_SCHEMA);
        assert_eq!(value["schema_version"], CLI_JSON_SCHEMA_VERSION);
        assert_eq!(value["event"].as_str(), Some("progress"));
        assert_eq!(value["recipe"]["digest"], "09".repeat(32));
        assert_eq!(value["status"].as_str(), Some("completed"));
        assert_eq!(value["completed"].as_u64(), Some(3));
        assert_eq!(value["total"].as_u64(), Some(5));
        assert_eq!(value["elapsed_seconds"].as_f64(), Some(1.235));
        assert_eq!(value["eta_seconds"].as_f64(), Some(0.457));
        assert_eq!(value["input"].as_str(), Some(SPECIAL_INPUT));
    }

    #[test]
    fn batch_summary_json_round_trips_special_paths() {
        let value = parse_json_line(&batch_summary_json_line(
            8,
            4,
            2,
            1,
            1,
            true,
            SPECIAL_OUTPUT,
        ));

        assert_eq!(value.as_object().unwrap().len(), 11);
        assert_eq!(value["schema"], CLI_JSON_SCHEMA);
        assert_eq!(value["schema_version"], CLI_JSON_SCHEMA_VERSION);
        assert_eq!(value["event"].as_str(), Some("summary"));
        assert!(value["recipe"]["digest"].is_null());
        assert_eq!(value["total"].as_u64(), Some(8));
        assert_eq!(value["succeeded"].as_u64(), Some(4));
        assert_eq!(value["skipped"].as_u64(), Some(2));
        assert_eq!(value["failed"].as_u64(), Some(1));
        assert_eq!(value["cancelled_count"].as_u64(), Some(1));
        assert_eq!(value["cancelled"].as_bool(), Some(true));
        assert_eq!(value["output"].as_str(), Some(SPECIAL_OUTPUT));
    }

    #[test]
    fn cli_schema_includes_live_diagnostics_without_a_recipe() {
        let schema: Value =
            serde_json::from_str(include_str!("../schemas/denoize-cli-output-v1.schema.json"))
                .unwrap();
        assert!(schema["oneOf"]
            .as_array()
            .unwrap()
            .iter()
            .any(|variant| variant["$ref"] == "#/$defs/liveStatus"));
        let live = &schema["$defs"]["liveStatus"];
        assert_eq!(live["properties"]["event"]["const"], "status");
        assert_eq!(live["properties"]["mode"]["const"], "live");
        assert!(live["properties"].get("recipe").is_none());
        for required in [
            "state",
            "estimated_total_latency_ms",
            "drift_correction_ppm",
            "underrun_frames",
            "overflow_frames",
            "reconnect_attempts",
            "device_generation",
        ] {
            assert!(live["required"]
                .as_array()
                .unwrap()
                .iter()
                .any(|field| field == required));
        }
    }

    #[cfg(feature = "live")]
    #[test]
    fn live_connection_states_have_stable_json_names() {
        use denoize::live::LiveConnectionState;

        assert_eq!(
            live_connection_state_name(LiveConnectionState::Connecting),
            "connecting"
        );
        assert_eq!(
            live_connection_state_name(LiveConnectionState::Priming),
            "priming"
        );
        assert_eq!(
            live_connection_state_name(LiveConnectionState::Running),
            "running"
        );
        assert_eq!(
            live_connection_state_name(LiveConnectionState::Recovering),
            "recovering"
        );
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    // Item identities preserve each platform's raw OS path representation:
    // UTF-8 bytes on Unix-like targets and UTF-16LE code units on Windows.
    #[cfg(not(windows))]
    const FRONTEND_PARITY_ITEM_ID_HEX: &str =
        "795ada4ccf8186cdaa1d64cec4f53165bc5ca003d68e0964aee9a33a5f8105e8";
    #[cfg(windows)]
    const FRONTEND_PARITY_ITEM_ID_HEX: &str =
        "28a3a5bc0a5112777268b438a5357badea3c055ea91a1472a9cdba3c1a8522f0";
    // The package version is intentionally part of the v3 recipe ABI. Update
    // this value in both frontend tests when an intentional release bump lands.
    const FRONTEND_PARITY_RECIPE_HEX: &str =
        "ce2c6726706896f76a1ea74e8d5576b4bbfd7184aefd6f14cee99c1d62417e90";

    #[test]
    fn batch_reuses_one_prepared_backend_for_equal_resolved_options() {
        let options = service::ResolvedProcessingOptions {
            backend: Backend::Classical,
            denoiser: DenoiserConfig::default(48_000),
            backend_options: BackendOptions::default(),
            accelerator: denoize::AcceleratorSelection::default(),
            loudness_lufs: None,
            true_peak_dbtp: -1.0,
        };
        let mut cache = Vec::new();
        let governor = resource_governor(&Overrides::default(), 1).unwrap();
        let first = cached_backend_session(&mut cache, &options, false, &governor)
            .unwrap()
            .unwrap();
        let second = cached_backend_session(&mut cache, &options, false, &governor)
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&first.session, &second.session));
        assert_eq!(cache.len(), 1);
        assert!(
            cached_backend_session(&mut cache, &options, true, &governor)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn cancellation_while_waiting_for_publication_fence_never_publishes() {
        let fence = Arc::new(Mutex::new(()));
        let cancelled = Arc::new(AtomicBool::new(false));
        let published = Arc::new(AtomicBool::new(false));
        let held = fence.lock().unwrap();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let worker_fence = Arc::clone(&fence);
        let worker_cancelled = Arc::clone(&cancelled);
        let worker_published = Arc::clone(&published);
        let worker = std::thread::spawn(move || {
            ready_tx.send(()).unwrap();
            with_batch_publication_fence(&worker_fence, &worker_cancelled, || {
                worker_published.store(true, Ordering::SeqCst);
                Ok(())
            })
        });

        ready_rx.recv().unwrap();
        cancelled.store(true, Ordering::SeqCst);
        drop(held);

        assert_eq!(worker.join().unwrap().unwrap(), None);
        assert!(!published.load(Ordering::SeqCst));
    }

    fn temporary_directory() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "denoize-batch-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn write_stereo_batch_wav(path: &std::path::Path) {
        let audio = denoize::Audio {
            sample_rate: 48_000,
            channels: vec![vec![0.0; 960], vec![0.0; 960]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        denoize::write_audio(path, &audio, EncodeOptions::default()).unwrap();
    }

    #[test]
    fn cli_batch_recipe_matches_the_frontend_parity_golden_vector() {
        let root = temporary_directory().join("frontend-parity-golden");
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(&input).unwrap();
        let source = input.join("stereo.wav");
        write_stereo_batch_wav(&source);
        let options = parse_config(
            r#"
backend = "classical"
preset = "hifi"
mode = "speech"
strength = 0.37
adaptive_noise = false
vad = false
channels = "linked"
downmix = "preserve"
loudness_lufs = -16.0
true_peak_dbtp = -1.0
preserve_metadata = false
mp3_bitrate_kbps = 256
m4a_bitrate_kbps = 224
aac_encoder = "oxide"
output_format = "mp3"
batch = true
resume = true
max_memory_mb = 64
"#,
            "frontend-parity.toml",
        )
        .unwrap();
        let encode = build_encode_options(&options).unwrap();
        let items = plan_batch_files(
            &input,
            &output,
            vec![source.clone()],
            options.output_format.as_deref(),
        )
        .unwrap();
        let prepared = preflight_batch_items(
            &items,
            &options,
            encode,
            resolve_explicit_backend_options(&options).unwrap().as_ref(),
            &resource_governor(&options, 1).unwrap(),
            false,
        )
        .unwrap();
        let prepared = &prepared[0];

        assert_eq!(prepared.resolved_processing.backend, Backend::Classical);
        assert!(!prepared.resolved_processing.denoiser.adaptive_noise);
        assert!(!prepared.resolved_processing.denoiser.vad);
        assert_eq!(
            prepared.resolved_processing.backend_options.channel_mode,
            ChannelMode::StereoLinked
        );
        assert_eq!(prepared.resolved_processing.loudness_lufs, Some(-16.0));
        assert_eq!(prepared.resolved_processing.true_peak_dbtp, -1.0);
        assert_eq!(prepared.item.output_format, OutputFormat::Mp3);
        assert_eq!(encode.mp3_bitrate_kbps, 256);
        assert!(options.no_metadata);
        assert_eq!(options.max_memory_mb, Some(64));
        assert!(prepared.expectation.model().is_none());
        assert_eq!(prepared.expectation.recipe(), prepared.recipe);
        assert_eq!(prepared.recipe.as_hex(), FRONTEND_PARITY_RECIPE_HEX);
        assert_eq!(
            prepared.expectation.item_id(),
            batch_resume::item_identity(
                &normalize_batch_path(&source).unwrap(),
                &prepared.item.input_relative,
                &prepared.item.destination_relative,
                OutputFormat::Mp3,
            )
        );

        let fixed_item_id = batch_resume::item_identity(
            std::path::Path::new("/denoize/frontend-parity/input/stereo.wav"),
            std::path::Path::new("stereo.wav"),
            std::path::Path::new("stereo.mp3"),
            OutputFormat::Mp3,
        );
        assert_eq!(fixed_item_id.as_hex(), FRONTEND_PARITY_ITEM_ID_HEX);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cli_treats_legacy_desktop_state_as_untrusted_then_preserves_it() {
        for (index, legacy) in [
            b"sample.wav\n".to_vec(),
            format!("v2:{}\n", "41".repeat(32)).into_bytes(),
        ]
        .into_iter()
        .enumerate()
        {
            let root = temporary_directory().join(format!("legacy-gui-state-{index}"));
            let input = root.join("input");
            let output = root.join("output");
            std::fs::create_dir_all(&input).unwrap();
            std::fs::create_dir_all(&output).unwrap();
            write_stereo_batch_wav(&input.join("sample.wav"));
            let destination = output.join("sample.wav");
            let original_output = b"legacy desktop output";
            std::fs::write(&destination, original_output).unwrap();
            let legacy_path = output.join(LEGACY_DESKTOP_STATE_FILE_NAME);
            std::fs::write(&legacy_path, &legacy).unwrap();
            let mut options = Overrides {
                batch: true,
                resume: true,
                no_progress: true,
                ..Overrides::default()
            };

            let error =
                run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap_err();
            assert!(error.contains("legacy"), "{error}");
            assert!(error.contains("--force"), "{error}");
            assert_eq!(std::fs::read(&destination).unwrap(), original_output);
            assert_eq!(std::fs::read(&legacy_path).unwrap(), legacy);
            assert!(!output.join(STATE_FILE_NAME).exists());

            options.force = true;
            run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();
            let migrated_state = std::fs::read(output.join(STATE_FILE_NAME)).unwrap();
            let migrated_output = std::fs::read(&destination).unwrap();
            assert!(String::from_utf8_lossy(&migrated_state).contains("\"version\":3"));
            assert_eq!(std::fs::read(&legacy_path).unwrap(), legacy);

            options.force = false;
            run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();
            assert_eq!(
                std::fs::read(output.join(STATE_FILE_NAME)).unwrap(),
                migrated_state
            );
            assert_eq!(std::fs::read(destination).unwrap(), migrated_output);
            assert_eq!(std::fs::read(&legacy_path).unwrap(), legacy);
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn batch_collection_is_recursive_and_sorted() {
        let root = temporary_directory();
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(root.join("b.wav"), []).unwrap();
        std::fs::write(root.join("ignore.txt"), []).unwrap();
        std::fs::write(nested.join("a.FLAC"), []).unwrap();

        assert_eq!(
            collect_batch_files(&root, false).unwrap(),
            vec![root.join("b.wav")]
        );
        assert_eq!(
            collect_batch_files(&root, true).unwrap(),
            vec![root.join("b.wav"), nested.join("a.FLAC")]
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn validates_batch_output_format() {
        assert_eq!(normalize_output_extension(".flac").unwrap(), "flac");
        assert_eq!(normalize_output_extension("aac").unwrap(), "aac");
        assert_eq!(normalize_output_extension("oga").unwrap(), "oga");
        assert!(normalize_output_extension("wma").is_err());
    }

    fn probe(format: AudioFormat, codec: AudioCodec) -> AudioProbe {
        AudioProbe {
            format,
            codec,
            audio_tracks: 1,
            has_non_audio_tracks: false,
            is_broadcast_wave: false,
        }
    }

    #[test]
    fn batch_preserve_policy_is_codec_and_container_exact() {
        for (source, output) in [
            (probe(AudioFormat::Wav, AudioCodec::Pcm), OutputFormat::Wav),
            (
                probe(AudioFormat::Flac, AudioCodec::Flac),
                OutputFormat::Flac,
            ),
            (
                probe(AudioFormat::OggOpus, AudioCodec::Opus),
                OutputFormat::OggOpus,
            ),
            (probe(AudioFormat::Mp3, AudioCodec::Mp3), OutputFormat::Mp3),
            (probe(AudioFormat::M4a, AudioCodec::Aac), OutputFormat::M4a),
            (
                probe(AudioFormat::AacAdts, AudioCodec::Aac),
                OutputFormat::AacAdts,
            ),
        ] {
            assert!(batch_can_preserve(&source, output));
        }

        for (source, output) in [
            (probe(AudioFormat::Rf64, AudioCodec::Pcm), OutputFormat::Wav),
            (probe(AudioFormat::Aiff, AudioCodec::Pcm), OutputFormat::Wav),
            (probe(AudioFormat::Caf, AudioCodec::Pcm), OutputFormat::Wav),
            (
                probe(AudioFormat::OggVorbis, AudioCodec::Vorbis),
                OutputFormat::OggOpus,
            ),
            (probe(AudioFormat::M4a, AudioCodec::Alac), OutputFormat::M4a),
        ] {
            assert!(!batch_can_preserve(&source, output));
        }

        let mut multi_track = probe(AudioFormat::M4a, AudioCodec::Aac);
        multi_track.audio_tracks = 2;
        assert!(!batch_can_preserve(&multi_track, OutputFormat::M4a));
        multi_track.audio_tracks = 1;
        multi_track.has_non_audio_tracks = true;
        assert!(!batch_can_preserve(&multi_track, OutputFormat::M4a));

        let mut broadcast_wave = probe(AudioFormat::Wav, AudioCodec::Pcm);
        broadcast_wave.is_broadcast_wave = true;
        assert!(!batch_can_preserve(&broadcast_wave, OutputFormat::Wav));
    }

    #[test]
    fn batch_resume_identity_includes_destination_and_codec() {
        let identity = std::path::Path::new("/input-a/voice.aiff");
        let input = std::path::Path::new("voice.aiff");
        let wav = batch_resume::item_identity(
            identity,
            input,
            std::path::Path::new("voice.wav"),
            OutputFormat::Wav,
        );
        let flac = batch_resume::item_identity(
            identity,
            input,
            std::path::Path::new("voice.flac"),
            OutputFormat::Flac,
        );
        let renamed = batch_resume::item_identity(
            identity,
            input,
            std::path::Path::new("nested/voice.wav"),
            OutputFormat::Wav,
        );

        assert_ne!(wav, flac);
        assert_ne!(wav, renamed);
        assert_ne!(
            wav,
            batch_resume::item_identity(
                std::path::Path::new("/input-b/voice.aiff"),
                input,
                std::path::Path::new("voice.wav"),
                OutputFormat::Wav,
            )
        );
        assert_eq!(wav.as_hex().len(), 64);
    }

    #[test]
    fn batch_plan_rejects_collisions_before_processing() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(&input).unwrap();
        let aiff = input.join("clip.aiff");
        let caf = input.join("clip.caf");
        std::fs::write(&aiff, b"FORM\0\0\0\0AIFF").unwrap();
        std::fs::write(&caf, b"caff\0\x01\0\0\0\0\0\0").unwrap();

        let error = plan_batch_files(&input, &output, vec![aiff, caf], Some("flac")).unwrap_err();

        assert!(error.contains("multiple inputs map to the same batch output"));
        assert!(!output.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_plan_rejects_file_directory_prefix_collisions() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        let nested = input.join("clip.wav");
        std::fs::create_dir_all(&nested).unwrap();
        let aiff = input.join("clip.aiff");
        let caf = nested.join("child.caf");
        std::fs::write(&aiff, b"FORM\0\0\0\0AIFF").unwrap();
        std::fs::write(&caf, b"caff\0\x01\0\0\0\0\0\0").unwrap();

        let error = plan_batch_files(&input, &output, vec![aiff, caf], Some("wav")).unwrap_err();

        assert!(error.contains("conflict as a file and directory"));
        assert!(!output.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn batch_collection_does_not_follow_directory_symlinks() {
        use std::os::unix::fs::symlink;

        let root = temporary_directory();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("voice.wav"), []).unwrap();
        symlink(&root, root.join("loop")).unwrap();

        assert_eq!(
            collect_batch_files(&root, true).unwrap(),
            vec![root.join("voice.wav")]
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn batch_plan_rejects_output_symlinks_into_the_input_tree() {
        use std::os::unix::fs::symlink;

        let root = temporary_directory();
        let input = root.join("input");
        let input_nested = input.join("nested");
        let output = root.join("output");
        std::fs::create_dir_all(&input_nested).unwrap();
        std::fs::create_dir_all(&output).unwrap();
        symlink(&input_nested, output.join("nested")).unwrap();
        let aiff = input_nested.join("voice.aiff");
        std::fs::write(&aiff, b"FORM\0\0\0\0AIFF").unwrap();

        let error = plan_batch_files(&input, &output, vec![aiff], Some("wav")).unwrap_err();

        assert!(error.contains("resolves inside the input directory"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_plan_accepts_explicit_conversion_for_decode_only_input() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(&input).unwrap();
        let aiff = input.join("voice.aiff");
        std::fs::write(&aiff, b"FORM\0\0\0\0AIFF").unwrap();

        let items = plan_batch_files(&input, &output, vec![aiff.clone()], Some("wav")).unwrap();

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].input, aiff);
        assert_eq!(items[0].destination, output.join("voice.wav"));
        assert_eq!(items[0].input_relative, std::path::Path::new("voice.aiff"));
        assert_eq!(
            items[0].destination_relative,
            std::path::Path::new("voice.wav")
        );
        assert_eq!(items[0].output_format, OutputFormat::Wav);
        assert!(!output.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_preflight_has_no_output_side_effects() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(&input).unwrap();
        let audio = denoize::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 1_600]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        denoize::write_audio(input.join("a-valid.wav"), &audio, EncodeOptions::default()).unwrap();
        std::fs::write(input.join("b-decode-only.aiff"), b"FORM\0\0\0\0AIFF").unwrap();
        let options = Overrides {
            batch: true,
            no_progress: true,
            ..Overrides::default()
        };

        let error =
            run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap_err();

        assert!(error.contains("AIFF/AIFC"));
        assert!(error.contains("--output-format"));
        assert!(!output.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_temporary_limit_fails_before_creating_the_output_directory() {
        let root = temporary_directory().join("temporary-limit");
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(&input).unwrap();
        write_stereo_batch_wav(&input.join("voice.wav"));
        let options = Overrides {
            batch: true,
            no_progress: true,
            max_temporary_mb: Some(1),
            ..Overrides::default()
        };

        let error =
            run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap_err();

        assert!(error.contains("temporary"), "unexpected error: {error}");
        assert!(!output.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_process_memory_limit_serializes_two_full_weight_workers() {
        let root = temporary_directory().join("process-limit");
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(&input).unwrap();
        write_stereo_batch_wav(&input.join("first.wav"));
        write_stereo_batch_wav(&input.join("second.wav"));
        let options = Overrides {
            batch: true,
            no_progress: true,
            jobs: Some(2),
            max_process_memory_mb: Some(1),
            ..Overrides::default()
        };

        run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();

        assert!(output.join("first.wav").is_file());
        assert!(output.join("second.wav").is_file());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_rejects_overlapping_input_and_output_directories() {
        let root = temporary_directory();
        let nested = root.join("nested");
        std::fs::create_dir_all(&root).unwrap();

        assert!(validate_batch_directories(&root, &root).is_err());
        assert!(validate_batch_directories(&root, &nested).is_err());
        assert!(validate_batch_directories(&nested, &root).is_err());

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resume_state_path_is_reserved_before_processing() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        let reserved = input.join(".denoize-state");
        std::fs::create_dir_all(&reserved).unwrap();
        let audio = denoize::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 1_600]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        denoize::write_audio(reserved.join("voice.wav"), &audio, EncodeOptions::default()).unwrap();
        let options = Overrides {
            batch: true,
            recursive: true,
            resume: true,
            no_progress: true,
            ..Overrides::default()
        };

        let error =
            run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap_err();

        assert!(error.contains(STATE_FILE_NAME));
        assert!(!output.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_counts_distinguish_completed_skipped_and_failed_results() {
        let results = [
            BatchFileOutcome::Completed(Some(FileFingerprint {
                len: 1,
                digest: Digest::from_bytes([1; 32]),
            })),
            BatchFileOutcome::Skipped(FileFingerprint {
                len: 1,
                digest: Digest::from_bytes([2; 32]),
            }),
            BatchFileOutcome::Failed("processing failed".into()),
            BatchFileOutcome::Cancelled,
        ];

        assert_eq!(
            count_batch_results(&results),
            BatchCounts {
                succeeded: 1,
                skipped: 1,
                failed: 1,
                cancelled: 1,
            }
        );
    }

    #[test]
    fn batch_processes_nested_audio_and_converts_format() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(input.join("nested")).unwrap();
        let audio = denoize::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 3_200]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        denoize::write_audio(
            input.join("nested/sample.wav"),
            &audio,
            EncodeOptions::default(),
        )
        .unwrap();
        let options = Overrides {
            batch: true,
            recursive: true,
            jobs: Some(2),
            output_format: Some("flac".into()),
            ..Overrides::default()
        };

        run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();
        assert!(std::fs::symlink_metadata(output.join("nested/sample.flac"))
            .unwrap()
            .file_type()
            .is_file());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn deterministic_batch_is_byte_stable_even_with_multiple_requested_jobs() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(&input).unwrap();
        for (name, frequency) in [("a.wav", 220.0), ("b.wav", 440.0)] {
            let audio = denoize::Audio {
                sample_rate: 16_000,
                channels: vec![(0..3_200)
                    .map(|index| {
                        (2.0 * std::f64::consts::PI * frequency * index as f64 / 16_000.0).sin()
                            * 0.2
                    })
                    .collect()],
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
                channel_mask: None,
            };
            denoize::write_audio(input.join(name), &audio, EncodeOptions::default()).unwrap();
        }
        let options = Overrides {
            batch: true,
            deterministic: true,
            force: true,
            jobs: Some(8),
            no_progress: true,
            ..Overrides::default()
        };

        run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();
        let first_a = std::fs::read(output.join("a.wav")).unwrap();
        let first_b = std::fs::read(output.join("b.wav")).unwrap();
        run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();
        assert_eq!(first_a, std::fs::read(output.join("a.wav")).unwrap());
        assert_eq!(first_b, std::fs::read(output.join("b.wav")).unwrap());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resume_skips_outputs_recorded_as_complete() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(&input).unwrap();
        let audio = denoize::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 1_600]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        denoize::write_audio(input.join("sample.wav"), &audio, EncodeOptions::default()).unwrap();
        let options = Overrides {
            batch: true,
            resume: true,
            no_progress: true,
            ..Overrides::default()
        };

        run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();
        let first_modified = std::fs::metadata(output.join("sample.wav"))
            .unwrap()
            .modified()
            .unwrap();
        run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();
        let second_modified = std::fs::metadata(output.join("sample.wav"))
            .unwrap()
            .modified()
            .unwrap();
        assert_eq!(first_modified, second_modified);
        let state = std::fs::read_to_string(output.join(STATE_FILE_NAME)).unwrap();
        assert!(state.contains("\"version\":3"));
        assert!(state.contains("\"kind\":\"prepare\""));
        assert!(state.contains("\"kind\":\"complete\""));
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod auto_backend_tests {
    use super::*;

    #[test]
    fn parses_auto_backend() {
        let (_, _, options) = parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--backend".into(),
            "auto".into(),
        ])
        .unwrap();
        assert!(options.auto_backend);
        assert!(options.backend.is_none());
    }

    #[test]
    fn automatic_selection_uses_an_available_backend() {
        let selected = service::select_backend(BackendChoice::Auto, 30.0, None);
        assert!(Backend::available_names().contains(&service::backend_name(selected)));
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use base64::Engine as _;
    #[cfg(feature = "gtcrn")]
    use prost::Message;
    #[cfg(feature = "gtcrn")]
    use tract_onnx::pb::{
        tensor_proto, tensor_shape_proto, type_proto, GraphProto, ModelProto, NodeProto,
        OperatorSetIdProto, TensorShapeProto, TypeProto, ValueInfoProto,
    };

    const SILENT_STEREO_ADTS: [u8; 13] = [
        0xff, 0xf1, 0x50, 0x80, 0x01, 0xbf, 0xfc, 0x21, 0x00, 0x00, 0x00, 0x00, 0x1c,
    ];

    struct ResetCheckpointHooks;

    impl Drop for ResetCheckpointHooks {
        fn drop(&mut self) {
            TEST_STREAM_CHECKPOINT_FRAMES.with(|value| value.set(None));
            TEST_STOP_AFTER_STREAM_CHECKPOINT.with(|value| value.set(false));
            TEST_STOP_AFTER_STREAM_COMMIT.with(|value| value.set(false));
            TEST_CORRUPT_STREAM_OUTPUT_BEFORE_VERIFY.with(|value| value.set(false));
        }
    }

    fn stop_after_checkpoint(interval_frames: u64) -> ResetCheckpointHooks {
        TEST_STREAM_CHECKPOINT_FRAMES.with(|value| value.set(Some(interval_frames)));
        TEST_STOP_AFTER_STREAM_CHECKPOINT.with(|value| value.set(true));
        ResetCheckpointHooks
    }

    fn stop_after_stream_commit() -> ResetCheckpointHooks {
        TEST_STOP_AFTER_STREAM_COMMIT.with(|value| value.set(true));
        ResetCheckpointHooks
    }

    fn corrupt_stream_output_before_verify() -> ResetCheckpointHooks {
        TEST_CORRUPT_STREAM_OUTPUT_BEFORE_VERIFY.with(|value| value.set(true));
        ResetCheckpointHooks
    }

    #[test]
    fn parses_stream_option() {
        let (_, _, options) = parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--stream".into(),
            "--stream-frames".into(),
            "4096".into(),
            "--max-memory".into(),
            "64".into(),
        ])
        .unwrap();
        assert!(options.stream);
        assert_eq!(options.stream_frames, Some(4096));
        assert_eq!(options.max_memory_mb, Some(64));
    }

    #[test]
    fn resume_requires_batch_or_stream_before_input_io() {
        let error = run(&[
            "missing-input.wav".into(),
            "unused-output.wav".into(),
            "--resume".into(),
            "--isolate".into(),
        ])
        .expect_err("standalone resume must be rejected before isolation or input I/O");
        assert_eq!(error, "--resume requires --batch or --stream");
    }

    #[test]
    fn rejects_out_of_range_resource_limits() {
        let error = validate_effective_options(
            &Overrides {
                max_memory_mb: Some(0),
                ..Overrides::default()
            },
            VALIDATION_SAMPLE_RATE,
        )
        .unwrap_err();
        assert!(error.contains("--max-memory"));
        let error = validate_effective_options(
            &Overrides {
                stream_frames: Some(MAX_STREAM_BLOCK_FRAMES + 1),
                ..Overrides::default()
            },
            VALIDATION_SAMPLE_RATE,
        )
        .unwrap_err();
        assert!(error.contains("--stream-frames"));
    }

    #[test]
    fn metadata_limits_reserve_payload_and_descriptor_overhead() {
        let limits = metadata_limits_for_available_bytes(Some(BYTES_PER_MIB));
        assert_eq!(limits.max_total_bytes, 64 * 1024);
        assert_eq!(limits.max_item_bytes, 64 * 1024);
        assert_eq!(limits.max_flac_block_bytes, 64 * 1024);
        assert_eq!(limits.max_ogg_packet_bytes, 64 * 1024);
        assert_eq!(limits.max_items, 256);
        assert_eq!(limits.max_flac_blocks, 256);
        assert_eq!(limits.max_ogg_pages, 256);
        assert_eq!(
            limits.max_ogg_streams,
            MetadataLimits::DEFAULT_MAX_OGG_STREAMS
        );

        let defaults = MetadataLimits::default();
        let uncapped = metadata_limits_for_available_bytes(None);
        assert_eq!(uncapped, defaults);
        let large = metadata_limits_for_available_bytes(Some(u64::MAX));
        assert_eq!(large, defaults);

        let exhausted = retained_metadata_limits(Some(1), BYTES_PER_MIB).unwrap();
        assert_eq!(exhausted.max_total_bytes, 0);
        assert_eq!(exhausted.max_items, 0);
        assert_eq!(exhausted.max_flac_block_bytes, 64 * 1024);
        assert_eq!(exhausted.max_flac_blocks, 256);
        assert_eq!(exhausted.max_ogg_packet_bytes, 64 * 1024);
        assert_eq!(exhausted.max_ogg_pages, 256);
    }

    fn write_stream_output_fixture(path: &std::path::Path, frames: usize) -> denoize::Audio {
        let sample_rate = 48_000;
        let left = (0..frames)
            .map(|frame| {
                let phase = std::f64::consts::TAU * 431.0 * frame as f64 / sample_rate as f64;
                phase.sin() * 0.23 + (phase * 0.37).cos() * 0.04
            })
            .collect::<Vec<_>>();
        let audio = denoize::Audio {
            sample_rate,
            channels: vec![left],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: denoize::ChannelLayout::Mono.mask(),
        };
        denoize::write_audio(path, &audio, EncodeOptions::default()).unwrap();
        audio
    }

    #[test]
    fn streams_wav_to_each_builtin_encoded_output() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.wav");
        let input_frames = 5_001usize;
        let audio = write_stream_output_fixture(&input, input_frames);

        for (extension, expected_frames) in [
            ("wav", input_frames),
            ("flac", input_frames),
            ("opus", input_frames),
            ("mp3", input_frames.div_ceil(1_152) * 1_152),
        ] {
            let output = root.path().join(format!("output.{extension}"));
            run_streaming_wav(
                input.to_str().unwrap(),
                output.to_str().unwrap(),
                Overrides {
                    stream: true,
                    no_metadata: true,
                    stream_frames: Some(317),
                    max_memory_mb: Some(128),
                    ..Overrides::default()
                },
            )
            .unwrap_or_else(|error| panic!("stream WAV to {extension}: {error}"));

            let decoded = read_audio(&output)
                .unwrap_or_else(|error| panic!("decode streamed {extension}: {error}"));
            assert_eq!(decoded.sample_rate, audio.sample_rate, "{extension}");
            assert_eq!(decoded.channels(), audio.channels(), "{extension}");
            assert_eq!(decoded.frames(), expected_frames, "{extension}");
        }
    }

    #[test]
    fn corrupt_stream_output_is_rejected_before_atomic_publication() {
        let _reset = corrupt_stream_output_before_verify();
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.wav");
        let output = root.path().join("output.flac");
        write_stream_output_fixture(&input, 5_001);

        let error = run_streaming_wav(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            Overrides {
                stream: true,
                no_metadata: true,
                stream_frames: Some(317),
                max_memory_mb: Some(128),
                ..Overrides::default()
            },
        )
        .expect_err("corrupt staged FLAC must fail verification");

        assert!(
            error.contains("FLAC"),
            "unexpected verification error: {error}"
        );
        assert!(!output.exists());
        let remaining = std::fs::read_dir(root.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>();
        assert_eq!(remaining, vec![input.file_name().unwrap()]);
    }

    #[cfg(feature = "m4a-encode")]
    #[test]
    fn streams_wav_to_native_aac_outputs() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.wav");
        let input_frames = 5_001usize;
        let audio = write_stream_output_fixture(&input, input_frames);

        for (extension, expected_frames) in [
            ("m4a", input_frames),
            // Raw ADTS has no edit-list or end-granule field. The Oxide AAC
            // encoder emits one delayed access unit in addition to the padded
            // source access units, so its physical decode timeline is explicit.
            ("aac", (input_frames.div_ceil(1_024) + 1) * 1_024),
        ] {
            let output = root.path().join(format!("output.{extension}"));
            run_streaming_wav(
                input.to_str().unwrap(),
                output.to_str().unwrap(),
                Overrides {
                    stream: true,
                    no_metadata: true,
                    stream_frames: Some(317),
                    max_memory_mb: Some(256),
                    ..Overrides::default()
                },
            )
            .unwrap_or_else(|error| panic!("stream WAV to {extension}: {error}"));

            let decoded = read_audio(&output)
                .unwrap_or_else(|error| panic!("decode streamed {extension}: {error}"));
            assert_eq!(decoded.sample_rate, audio.sample_rate, "{extension}");
            assert_eq!(decoded.channels(), audio.channels(), "{extension}");
            assert_eq!(decoded.frames(), expected_frames, "{extension}");
        }
    }

    #[cfg(feature = "fdk-aac-encoder")]
    #[test]
    fn streams_wav_to_gapless_fdk_m4a() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.wav");
        let output = root.path().join("output.m4a");
        let input_frames = 5_001usize;
        let audio = write_stream_output_fixture(&input, input_frames);

        run_streaming_wav(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            Overrides {
                stream: true,
                no_metadata: true,
                stream_frames: Some(317),
                max_memory_mb: Some(256),
                aac_encoder: Some(AacEncoder::Fdk),
                ..Overrides::default()
            },
        )
        .unwrap();

        let decoded = read_audio(&output).unwrap();
        assert_eq!(decoded.sample_rate, audio.sample_rate);
        assert_eq!(decoded.channels(), audio.channels());
        assert_eq!(decoded.frames(), input_frames);
    }

    #[test]
    fn streams_wav_without_loading_the_complete_audio() {
        let root = std::env::temp_dir().join(format!(
            "denoize-stream-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&root).unwrap();
        let input = root.join("input.wav");
        let output = root.join("output.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&input, spec).unwrap();
        for frame in 0..20_000 {
            let sample = (0.2
                * (2.0 * std::f64::consts::PI * 440.0 * frame as f64 / spec.sample_rate as f64)
                    .sin()
                * 32_767.0) as i16;
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();

        run_streaming_wav(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            Overrides {
                stream: true,
                stream_frames: Some(257),
                ..Overrides::default()
            },
        )
        .unwrap();
        let result = read_audio(&output).unwrap();
        assert_eq!(result.sample_rate, spec.sample_rate);
        assert_eq!(result.channels(), 1);
        assert_eq!(result.frames(), 20_000);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stream_vad_preserves_duration_and_attenuates_quiet_regions() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.wav");
        let plain = root.path().join("plain.wav");
        let gated = root.path().join("gated.wav");
        let audio = denoize::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.001; 16_000]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        denoize::write_audio(&input, &audio, EncodeOptions::default()).unwrap();
        for (output, vad) in [(&plain, false), (&gated, true)] {
            run_streaming_wav(
                input.to_str().unwrap(),
                output.to_str().unwrap(),
                Overrides {
                    stream: true,
                    vad: Some(vad),
                    no_profile: true,
                    stream_frames: Some(113),
                    ..Overrides::default()
                },
            )
            .unwrap();
        }
        let gated = read_audio(&gated).unwrap();
        assert_eq!(gated.frames(), audio.frames());
        let energy = |audio: &denoize::Audio| {
            audio.channels[0]
                .iter()
                .map(|sample| sample * sample)
                .sum::<f64>()
        };
        assert!(energy(&gated) < energy(&audio) * 0.01);
    }

    #[test]
    fn stream_loudness_runs_bounded_two_pass_normalization() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.wav");
        let output = root.path().join("output.wav");
        write_stream_output_fixture(&input, 96_000);
        run_streaming_wav(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            Overrides {
                stream: true,
                no_profile: true,
                loudness_lufs: Some(-24.0),
                true_peak_dbtp: Some(-1.0),
                stream_frames: Some(317),
                max_temporary_mb: Some(64),
                ..Overrides::default()
            },
        )
        .unwrap();
        let output = read_audio(&output).unwrap();
        assert_eq!(output.frames(), 96_000);
        let (lufs, peak) = denoize::loudness::measure(&output).unwrap();
        assert!((lufs + 24.0).abs() < 0.15, "measured {lufs} LUFS");
        assert!(peak <= -1.0 + 0.1, "measured {peak} dBTP");
    }

    #[test]
    fn streams_flac_to_atomic_wav() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.flac");
        let output = root.path().join("output.wav");
        let frames = 12_345;
        let audio = denoize::Audio {
            sample_rate: 24_000,
            channels: vec![(0..frames)
                .map(|frame| (frame as f64 * 0.03).sin() * 0.4)
                .collect()],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        denoize::write_audio(&input, &audio, EncodeOptions::default()).unwrap();

        run_streaming_wav(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            Overrides {
                stream: true,
                stream_frames: Some(131),
                max_memory_mb: Some(32),
                ..Overrides::default()
            },
        )
        .unwrap();
        let result = read_audio(&output).unwrap();
        assert_eq!(result.sample_rate, audio.sample_rate);
        assert_eq!(result.channels(), 1);
        assert_eq!(result.frames(), frames);
    }

    #[test]
    fn resumes_flac_stream_from_a_durable_checkpoint_byte_exactly() {
        let _reset = stop_after_checkpoint(300);
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.flac");
        let resumed_output = root.path().join("resumed.flac");
        let uninterrupted_output = root.path().join("uninterrupted.flac");
        let frames = 2_000;
        let audio = denoize::Audio {
            sample_rate: 24_000,
            channels: vec![(0..frames)
                .map(|frame| {
                    let phase = frame as f64 * 0.041;
                    phase.sin() * 0.35 + (phase * 0.37).cos() * 0.08
                })
                .collect()],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        denoize::write_audio(&input, &audio, EncodeOptions::default()).unwrap();

        let options = Overrides {
            stream: true,
            resume: true,
            no_metadata: true,
            stream_frames: Some(73),
            max_memory_mb: Some(64),
            ..Overrides::default()
        };
        let error = run_streaming_wav(
            input.to_str().unwrap(),
            resumed_output.to_str().unwrap(),
            options.clone(),
        )
        .unwrap_err();
        assert!(
            error.contains("injected stop after durable stream checkpoint"),
            "unexpected checkpoint error: {error}"
        );
        assert!(!resumed_output.exists());
        let (state, spool, _) = batch_resume::stream_checkpoint_sidecar_paths(&resumed_output)
            .expect("resolve checkpoint sidecars");
        assert!(state.exists());
        assert!(spool.exists());

        run_streaming_wav(
            input.to_str().unwrap(),
            resumed_output.to_str().unwrap(),
            options,
        )
        .unwrap();
        assert!(!state.exists());
        assert!(!spool.exists());

        run_streaming_wav(
            input.to_str().unwrap(),
            uninterrupted_output.to_str().unwrap(),
            Overrides {
                stream: true,
                no_metadata: true,
                stream_frames: Some(73),
                max_memory_mb: Some(64),
                ..Overrides::default()
            },
        )
        .unwrap();
        assert_eq!(
            std::fs::read(&resumed_output).unwrap(),
            std::fs::read(&uninterrupted_output).unwrap()
        );
        assert_eq!(read_audio(&resumed_output).unwrap().frames(), frames);
    }

    #[test]
    fn resumed_stream_plan_is_read_only_and_matches_its_signed_receipt() {
        let _reset = stop_after_checkpoint(300);
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.flac");
        let output = root.path().join("output.flac");
        let receipt = root.path().join("output.receipt.json");
        let secret = root.path().join("receipt-secret.json");
        let public = root.path().join("receipt-public.json");
        write_stream_output_fixture(&input, 2_000);
        denoize::write_new_receipt_keypair(&secret, &public).unwrap();
        let mut options = Overrides {
            stream: true,
            resume: true,
            no_metadata: true,
            stream_frames: Some(73),
            max_memory_mb: Some(64),
            ..Overrides::default()
        };

        let error = run_streaming_wav(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            options.clone(),
        )
        .unwrap_err();
        assert!(error.contains("injected stop after durable stream checkpoint"));
        let (state, spool, _) = batch_resume::stream_checkpoint_sidecar_paths(&output).unwrap();
        let state_before = std::fs::read(&state).unwrap();
        let spool_before = std::fs::read(&spool).unwrap();

        let plan = build_stream_execution_plan(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            &options,
        )
        .unwrap();
        assert_eq!(plan.kind, ExecutionKind::Stream);
        assert_eq!(plan.items[0].output.action, "process");
        assert_eq!(plan.items[0].output.reason, "checkpoint");
        assert_eq!(std::fs::read(&state).unwrap(), state_before);
        assert_eq!(std::fs::read(&spool).unwrap(), spool_before);
        assert!(!output.exists());

        options.receipt = Some(receipt.to_string_lossy().into_owned());
        options.receipt_key = Some(secret.to_string_lossy().into_owned());
        run_streaming_wav(input.to_str().unwrap(), output.to_str().unwrap(), options).unwrap();

        let signed = SignedExecutionReceipt::from_file(&receipt).unwrap();
        let key = ReceiptPublicKey::from_file(&public).unwrap();
        let report = signed
            .verify_with_key(&key, Some(&plan), &receipt, Some(root.path()))
            .unwrap();
        assert_eq!(report.kind, ExecutionKind::Stream);
        assert_eq!(report.verified_items[0].outcome, "succeeded");
        assert!(output.is_file());
        assert!(!state.exists());
        assert!(!spool.exists());
    }

    #[test]
    fn reconciles_a_committed_stream_after_cleanup_was_interrupted() {
        let _reset = stop_after_stream_commit();
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.flac");
        let output = root.path().join("output.wav");
        let frames = 1_111;
        let audio = denoize::Audio {
            sample_rate: 24_000,
            channels: vec![(0..frames)
                .map(|frame| (frame as f64 * 0.029).sin() * 0.3)
                .collect()],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        denoize::write_audio(&input, &audio, EncodeOptions::default()).unwrap();
        let options = Overrides {
            stream: true,
            resume: true,
            no_metadata: true,
            stream_frames: Some(97),
            max_memory_mb: Some(32),
            ..Overrides::default()
        };

        let error = run_streaming_wav(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            options.clone(),
        )
        .unwrap_err();
        assert!(error.contains("injected stop after committed stream output"));
        let published = std::fs::read(&output).unwrap();
        let (state, spool, _) = batch_resume::stream_checkpoint_sidecar_paths(&output).unwrap();
        assert!(state.exists());
        assert!(spool.exists());

        run_streaming_wav(input.to_str().unwrap(), output.to_str().unwrap(), options).unwrap();
        assert_eq!(std::fs::read(&output).unwrap(), published);
        assert!(!state.exists());
        assert!(!spool.exists());
        assert_eq!(read_audio(&output).unwrap().frames(), frames);
    }

    #[test]
    fn committed_stream_plan_skips_and_receipt_reconciles_after_cleanup_crash() {
        let _reset = stop_after_stream_commit();
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.flac");
        let output = root.path().join("output.wav");
        let receipt = root.path().join("output.receipt.json");
        let secret = root.path().join("receipt-secret.json");
        let public = root.path().join("receipt-public.json");
        write_stream_output_fixture(&input, 1_111);
        denoize::write_new_receipt_keypair(&secret, &public).unwrap();
        let mut options = Overrides {
            stream: true,
            resume: true,
            no_metadata: true,
            stream_frames: Some(97),
            max_memory_mb: Some(32),
            receipt: Some(receipt.to_string_lossy().into_owned()),
            receipt_key: Some(secret.to_string_lossy().into_owned()),
            ..Overrides::default()
        };

        let error = run_streaming_wav(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            options.clone(),
        )
        .unwrap_err();
        assert!(error.contains("injected stop after committed stream output"));
        assert!(output.is_file());
        assert!(!receipt.exists());
        let (state, spool, _) = batch_resume::stream_checkpoint_sidecar_paths(&output).unwrap();
        let state_before = std::fs::read(&state).unwrap();
        let spool_before = std::fs::read(&spool).unwrap();
        options.receipt = None;
        options.receipt_key = None;

        let plan = build_stream_execution_plan(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            &options,
        )
        .unwrap();
        assert_eq!(plan.items[0].output.action, "skip");
        assert_eq!(plan.items[0].output.publication, "none");
        assert_eq!(plan.items[0].output.reason, "completed");
        assert!(plan.items[0].output.existing_fingerprint.is_some());
        assert_eq!(std::fs::read(&state).unwrap(), state_before);
        assert_eq!(std::fs::read(&spool).unwrap(), spool_before);

        options.receipt = Some(receipt.to_string_lossy().into_owned());
        options.receipt_key = Some(secret.to_string_lossy().into_owned());
        run_streaming_wav(input.to_str().unwrap(), output.to_str().unwrap(), options).unwrap();

        let signed = SignedExecutionReceipt::from_file(&receipt).unwrap();
        let key = ReceiptPublicKey::from_file(&public).unwrap();
        let report = signed
            .verify_with_key(&key, Some(&plan), &receipt, Some(root.path()))
            .unwrap();
        assert_eq!(report.kind, ExecutionKind::Stream);
        assert_eq!(report.verified_items[0].outcome, "skipped");
        assert!(!state.exists());
        assert!(!spool.exists());
    }

    #[test]
    fn streams_ogg_vorbis_to_atomic_wav() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.ogg");
        let output = root.path().join("output.wav");
        let encoded = base64::engine::general_purpose::STANDARD
            .decode(include_str!("decode/testdata/tiny-vorbis.ogg.b64").trim())
            .unwrap();
        std::fs::write(&input, encoded).unwrap();
        let expected = read_audio(&input).unwrap();

        run_streaming_wav(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            Overrides {
                stream: true,
                stream_frames: Some(73),
                max_memory_mb: Some(32),
                ..Overrides::default()
            },
        )
        .unwrap();
        let result = read_audio(&output).unwrap();
        assert_eq!(result.sample_rate, expected.sample_rate);
        assert_eq!(result.channels(), expected.channels());
        assert_eq!(result.frames(), expected.frames());
    }

    #[test]
    fn resumes_ogg_vorbis_stream_from_a_durable_checkpoint_byte_exactly() {
        let _reset = stop_after_checkpoint(300);
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.ogg");
        let resumed_output = root.path().join("resumed.wav");
        let uninterrupted_output = root.path().join("uninterrupted.wav");
        let encoded = base64::engine::general_purpose::STANDARD
            .decode(include_str!("decode/testdata/tiny-vorbis.ogg.b64").trim())
            .unwrap();
        std::fs::write(&input, encoded).unwrap();

        let options = Overrides {
            stream: true,
            resume: true,
            no_metadata: true,
            stream_frames: Some(73),
            max_memory_mb: Some(32),
            ..Overrides::default()
        };
        let error = run_streaming_wav(
            input.to_str().unwrap(),
            resumed_output.to_str().unwrap(),
            options.clone(),
        )
        .unwrap_err();
        assert!(
            error.contains("injected stop after durable stream checkpoint"),
            "unexpected Ogg Vorbis checkpoint error: {error}"
        );
        assert!(!resumed_output.exists());

        run_streaming_wav(
            input.to_str().unwrap(),
            resumed_output.to_str().unwrap(),
            options,
        )
        .unwrap();
        run_streaming_wav(
            input.to_str().unwrap(),
            uninterrupted_output.to_str().unwrap(),
            Overrides {
                stream: true,
                no_metadata: true,
                stream_frames: Some(73),
                max_memory_mb: Some(32),
                ..Overrides::default()
            },
        )
        .unwrap();
        assert_eq!(
            std::fs::read(&resumed_output).unwrap(),
            std::fs::read(&uninterrupted_output).unwrap()
        );
        let (state, spool, _) = batch_resume::stream_checkpoint_sidecar_paths(&resumed_output)
            .expect("resolve checkpoint sidecars");
        assert!(!state.exists());
        assert!(!spool.exists());
    }

    #[test]
    fn resumes_mp3_stream_from_a_durable_checkpoint_byte_exactly() {
        let _reset = stop_after_checkpoint(300);
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.mp3");
        let resumed_output = root.path().join("resumed.wav");
        let uninterrupted_output = root.path().join("uninterrupted.wav");
        let input_frames = 5_001;
        let audio = denoize::Audio {
            sample_rate: 44_100,
            channels: vec![(0..input_frames)
                .map(|frame| {
                    let phase = std::f64::consts::TAU * 330.0 * frame as f64 / 44_100.0;
                    phase.sin() * 0.27
                })
                .collect()],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: denoize::ChannelLayout::Mono.mask(),
        };
        denoize::write_audio(&input, &audio, EncodeOptions::default()).unwrap();
        let expected_frames = read_audio(&input).unwrap().frames();

        let options = Overrides {
            stream: true,
            resume: true,
            no_metadata: true,
            stream_frames: Some(127),
            max_memory_mb: Some(32),
            ..Overrides::default()
        };
        let error = run_streaming_wav(
            input.to_str().unwrap(),
            resumed_output.to_str().unwrap(),
            options.clone(),
        )
        .unwrap_err();
        assert!(
            error.contains("injected stop after durable stream checkpoint"),
            "unexpected MP3 checkpoint error: {error}"
        );
        assert!(!resumed_output.exists());

        run_streaming_wav(
            input.to_str().unwrap(),
            resumed_output.to_str().unwrap(),
            options,
        )
        .unwrap();
        run_streaming_wav(
            input.to_str().unwrap(),
            uninterrupted_output.to_str().unwrap(),
            Overrides {
                stream: true,
                no_metadata: true,
                stream_frames: Some(127),
                max_memory_mb: Some(32),
                ..Overrides::default()
            },
        )
        .unwrap();
        assert_eq!(
            std::fs::read(&resumed_output).unwrap(),
            std::fs::read(&uninterrupted_output).unwrap()
        );
        assert_eq!(
            read_audio(&resumed_output).unwrap().frames(),
            expected_frames
        );
        let (state, spool, _) = batch_resume::stream_checkpoint_sidecar_paths(&resumed_output)
            .expect("resolve checkpoint sidecars");
        assert!(!state.exists());
        assert!(!spool.exists());
    }

    #[test]
    fn resumes_granule_aware_opus_stream_from_a_durable_checkpoint_byte_exactly() {
        let _reset = stop_after_checkpoint(300);
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.opus");
        let resumed_output = root.path().join("resumed.wav");
        let uninterrupted_output = root.path().join("uninterrupted.wav");
        let input_frames = 5_001;
        let audio = denoize::Audio {
            sample_rate: 48_000,
            channels: vec![(0..input_frames)
                .map(|frame| {
                    let phase = std::f64::consts::TAU * 510.0 * frame as f64 / 48_000.0;
                    phase.sin() * 0.23
                })
                .collect()],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: denoize::ChannelLayout::Mono.mask(),
        };
        denoize::write_audio(&input, &audio, EncodeOptions::default()).unwrap();
        let expected_frames = read_audio(&input).unwrap().frames();

        let options = Overrides {
            stream: true,
            resume: true,
            no_metadata: true,
            stream_frames: Some(127),
            max_memory_mb: Some(64),
            ..Overrides::default()
        };
        let error = run_streaming_wav(
            input.to_str().unwrap(),
            resumed_output.to_str().unwrap(),
            options.clone(),
        )
        .unwrap_err();
        assert!(
            error.contains("injected stop after durable stream checkpoint"),
            "unexpected Ogg Opus checkpoint error: {error}"
        );
        assert!(!resumed_output.exists());

        run_streaming_wav(
            input.to_str().unwrap(),
            resumed_output.to_str().unwrap(),
            options,
        )
        .unwrap();
        run_streaming_wav(
            input.to_str().unwrap(),
            uninterrupted_output.to_str().unwrap(),
            Overrides {
                stream: true,
                no_metadata: true,
                stream_frames: Some(127),
                max_memory_mb: Some(64),
                ..Overrides::default()
            },
        )
        .unwrap();
        assert_eq!(
            std::fs::read(&resumed_output).unwrap(),
            std::fs::read(&uninterrupted_output).unwrap()
        );
        assert_eq!(
            read_audio(&resumed_output).unwrap().frames(),
            expected_frames
        );
        let (state, spool, _) = batch_resume::stream_checkpoint_sidecar_paths(&resumed_output)
            .expect("resolve checkpoint sidecars");
        assert!(!state.exists());
        assert!(!spool.exists());
    }

    #[test]
    fn resumes_frame_aware_adts_aac_stream_from_a_durable_checkpoint_byte_exactly() {
        let _reset = stop_after_checkpoint(300);
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.aac");
        let resumed_output = root.path().join("resumed.wav");
        let uninterrupted_output = root.path().join("uninterrupted.wav");
        std::fs::write(&input, SILENT_STEREO_ADTS.repeat(3)).unwrap();
        let expected_frames = read_audio(&input).unwrap().frames();

        let options = Overrides {
            stream: true,
            resume: true,
            no_metadata: true,
            stream_frames: Some(127),
            max_memory_mb: Some(256),
            ..Overrides::default()
        };
        let error = run_streaming_wav(
            input.to_str().unwrap(),
            resumed_output.to_str().unwrap(),
            options.clone(),
        )
        .unwrap_err();
        assert!(
            error.contains("injected stop after durable stream checkpoint"),
            "unexpected ADTS AAC checkpoint error: {error}"
        );
        assert!(!resumed_output.exists());

        run_streaming_wav(
            input.to_str().unwrap(),
            resumed_output.to_str().unwrap(),
            options,
        )
        .unwrap();
        run_streaming_wav(
            input.to_str().unwrap(),
            uninterrupted_output.to_str().unwrap(),
            Overrides {
                stream: true,
                no_metadata: true,
                stream_frames: Some(127),
                max_memory_mb: Some(256),
                ..Overrides::default()
            },
        )
        .unwrap();
        assert_eq!(
            std::fs::read(&resumed_output).unwrap(),
            std::fs::read(&uninterrupted_output).unwrap()
        );
        assert_eq!(
            read_audio(&resumed_output).unwrap().frames(),
            expected_frames
        );
        let (state, spool, _) = batch_resume::stream_checkpoint_sidecar_paths(&resumed_output)
            .expect("resolve checkpoint sidecars");
        assert!(!state.exists());
        assert!(!spool.exists());
    }

    #[cfg(feature = "m4a-encode")]
    #[test]
    fn resumes_m4a_aac_stream_from_a_durable_checkpoint_byte_exactly() {
        let _reset = stop_after_checkpoint(300);
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.m4a");
        let resumed_output = root.path().join("resumed.wav");
        let uninterrupted_output = root.path().join("uninterrupted.wav");
        let input_frames = 5_001;
        let audio = denoize::Audio {
            sample_rate: 48_000,
            channels: vec![(0..input_frames)
                .map(|frame| {
                    let phase = std::f64::consts::TAU * 470.0 * frame as f64 / 48_000.0;
                    phase.sin() * 0.21
                })
                .collect()],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: denoize::ChannelLayout::Mono.mask(),
        };
        denoize::write_audio(&input, &audio, EncodeOptions::default()).unwrap();
        let expected_frames = read_audio(&input).unwrap().frames();

        let options = Overrides {
            stream: true,
            resume: true,
            no_metadata: true,
            stream_frames: Some(127),
            max_memory_mb: Some(256),
            ..Overrides::default()
        };
        let error = run_streaming_wav(
            input.to_str().unwrap(),
            resumed_output.to_str().unwrap(),
            options.clone(),
        )
        .unwrap_err();
        assert!(
            error.contains("injected stop after durable stream checkpoint"),
            "unexpected M4A AAC checkpoint error: {error}"
        );
        assert!(!resumed_output.exists());

        run_streaming_wav(
            input.to_str().unwrap(),
            resumed_output.to_str().unwrap(),
            options,
        )
        .unwrap();
        run_streaming_wav(
            input.to_str().unwrap(),
            uninterrupted_output.to_str().unwrap(),
            Overrides {
                stream: true,
                no_metadata: true,
                stream_frames: Some(127),
                max_memory_mb: Some(256),
                ..Overrides::default()
            },
        )
        .unwrap();
        assert_eq!(
            std::fs::read(&resumed_output).unwrap(),
            std::fs::read(&uninterrupted_output).unwrap()
        );
        assert_eq!(
            read_audio(&resumed_output).unwrap().frames(),
            expected_frames
        );
        let (state, spool, _) = batch_resume::stream_checkpoint_sidecar_paths(&resumed_output)
            .expect("resolve checkpoint sidecars");
        assert!(!state.exists());
        assert!(!spool.exists());
    }

    #[cfg(feature = "gtcrn")]
    #[test]
    fn streams_gtcrn_wav_through_the_common_session() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.wav");
        let output = root.path().join("output.wav");
        let model_path = root.path().join("gtcrn.onnx");
        let mut model_bytes = Vec::new();
        gtcrn_identity_model().encode(&mut model_bytes).unwrap();
        std::fs::write(&model_path, model_bytes).unwrap();

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&input, spec).unwrap();
        for frame in 0..1_201 {
            let sample = ((frame as f64 * 0.031).sin() * 8_000.0) as i16;
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();

        run_streaming_wav(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            Overrides {
                stream: true,
                stream_frames: Some(37),
                backend: Some(Backend::Gtcrn),
                onnx_model: Some(model_path.to_string_lossy().into_owned()),
                onnx_sample_rate: Some(16_000),
                ..Overrides::default()
            },
        )
        .unwrap();
        let result = read_audio(&output).unwrap();
        assert_eq!(result.sample_rate, spec.sample_rate);
        assert_eq!(result.channels(), 1);
        assert_eq!(result.frames(), 1_201);
        assert!(result.channels[0].iter().all(|sample| sample.is_finite()));
    }

    #[cfg(feature = "gtcrn")]
    fn gtcrn_identity_model() -> ModelProto {
        let bins = denoize::backend::gtcrn::BINS as i64;
        let shapes: [(&str, &str, &[i64]); 4] = [
            ("mixture", "enhanced", &[1, bins, 1, 2]),
            ("conv", "conv_out", &[2, 1, 16, 16, 33]),
            ("tra", "tra_out", &[2, 3, 1, 1, 16]),
            ("inter", "inter_out", &[2, 1, 33, 16]),
        ];
        ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            graph: Some(GraphProto {
                name: "gtcrn-cli-identity".into(),
                node: shapes
                    .iter()
                    .map(|(input, output, _)| NodeProto {
                        input: vec![(*input).into()],
                        output: vec![(*output).into()],
                        name: format!("{input}_identity"),
                        op_type: "Identity".into(),
                        ..Default::default()
                    })
                    .collect(),
                input: shapes
                    .iter()
                    .map(|(input, _, shape)| gtcrn_value_info(input, shape))
                    .collect(),
                output: shapes
                    .iter()
                    .map(|(_, output, shape)| gtcrn_value_info(output, shape))
                    .collect(),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[cfg(feature = "gtcrn")]
    fn gtcrn_value_info(name: &str, shape: &[i64]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.into(),
            r#type: Some(TypeProto {
                denotation: String::new(),
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: tensor_proto::DataType::Float as i32,
                    shape: Some(TensorShapeProto {
                        dim: shape
                            .iter()
                            .map(|value| tensor_shape_proto::Dimension {
                                value: Some(tensor_shape_proto::dimension::Value::DimValue(*value)),
                                denotation: String::new(),
                            })
                            .collect(),
                    }),
                })),
            }),
            doc_string: String::new(),
        }
    }
}

#[cfg(test)]
mod config_file_tests {
    use super::*;

    fn write_test_config(source: &str, label: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "denoize-{label}-{}-{}.toml",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, source).unwrap();
        path
    }

    fn cli_error(extra: &[&str]) -> String {
        let mut args = vec!["input.wav".into(), "output.wav".into()];
        args.extend(extra.iter().map(|value| (*value).to_string()));
        parse_args(&args).unwrap_err()
    }

    fn cli_ok(extra: &[&str]) {
        let mut args = vec!["input.wav".into(), "output.wav".into()];
        args.extend(extra.iter().map(|value| (*value).to_string()));
        parse_args(&args).unwrap();
    }

    #[test]
    fn parses_toml_defaults() {
        let options = parse_config(
            r#"
backend = "auto"
preset = "hifi"
mode = "speech"
strength = 0.42
dpss_nw = 2.5
kaiser_beta = 9.0
adaptive_noise = true
vad = true
preserve_metadata = false
downmix = "stereo"
accelerator = "auto"
deterministic = true
seed = 12345
stream_frames = 4096
max_memory_mb = 64
max_process_memory_mb = 256
max_temporary_mb = 128
max_gpu_memory_mb = 512
max_gpu_jobs = 2
isolate = true
chunk_ms = 100
live_latency_ms = 80
max_drift_ppm = 1500
reconnect_timeout_ms = 45000
"#,
            "test.toml",
        )
        .unwrap();
        assert!(options.auto_backend);
        assert!(options.deterministic);
        assert_eq!(options.accelerator, Some(AcceleratorPreference::Auto));
        assert_eq!(options.seed, Some(12345));
        assert_eq!(options.downmix, Some(DownmixMode::Stereo));
        assert_eq!(options.preset, Some(Preset::HiFi));
        assert_eq!(options.mode, Some(ProcessingMode::Speech));
        assert_eq!(options.strength, Some(0.42));
        assert_eq!(options.dpss_nw, Some(2.5));
        assert_eq!(options.kaiser_beta, Some(9.0));
        assert_eq!(options.adaptive_noise, Some(true));
        assert_eq!(options.vad, Some(true));
        assert!(options.no_metadata);
        assert_eq!(options.stream_frames, Some(4096));
        assert_eq!(options.max_memory_mb, Some(64));
        assert_eq!(options.max_process_memory_mb, Some(256));
        assert_eq!(options.max_temporary_mb, Some(128));
        assert_eq!(options.max_gpu_memory_mb, Some(512));
        assert_eq!(options.max_gpu_jobs, Some(2));
        assert!(options.isolate);
        assert_eq!(options.chunk_ms, Some(100));
        assert_eq!(options.live_latency_ms, Some(80));
        assert_eq!(options.max_drift_ppm, Some(1_500));
        assert_eq!(options.reconnect_timeout_ms, Some(45_000));
    }

    #[test]
    fn parses_desktop_exported_config() {
        let options = parse_config(
            r#"
backend = "auto"
preset = "hifi"
mode = "speech"
strength = 0.42
adaptive_noise = true
vad = true
channels = "linked"
downmix = "stereo"
loudness_lufs = -16.0
true_peak_dbtp = -1.0
preserve_metadata = false
force = true
mp3_bitrate_kbps = 256
m4a_bitrate_kbps = 224
aac_encoder = "oxide"
onnx_model = "model.onnx"
onnx_rate = 48000
sgmse_profile = "quality"
accelerator = "auto"
deterministic = true
"#,
            "desktop.toml",
        )
        .unwrap();

        assert!(options.auto_backend);
        assert_eq!(options.preset, Some(Preset::HiFi));
        assert_eq!(options.mode, Some(ProcessingMode::Speech));
        assert_eq!(options.strength, Some(0.42));
        assert_eq!(options.adaptive_noise, Some(true));
        assert_eq!(options.vad, Some(true));
        assert_eq!(options.channel_mode, Some(ChannelMode::StereoLinked));
        assert_eq!(options.downmix, Some(DownmixMode::Stereo));
        assert_eq!(options.loudness_lufs, Some(-16.0));
        assert_eq!(options.true_peak_dbtp, Some(-1.0));
        assert!(options.no_metadata);
        assert!(options.force);
        assert_eq!(options.mp3_bitrate_kbps, Some(256));
        assert_eq!(options.m4a_bitrate_kbps, Some(224));
        assert_eq!(options.aac_encoder, Some(AacEncoder::Oxide));
        assert_eq!(options.onnx_model.as_deref(), Some("model.onnx"));
        assert_eq!(options.onnx_sample_rate, Some(48_000));
        assert_eq!(options.sgmse_profile, Some(SgmseProfile::Quality));
        assert_eq!(options.accelerator, Some(AcceleratorPreference::Auto));
        assert!(options.deterministic);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn parses_runtime_model_package_config_paths() {
        let options = parse_config(
            "backend = \"onnx\"\nmodel_package = \"voice.dmp\"\nmodel_package_key = \"vendor.pub\"\n",
            "runtime-package.toml",
        )
        .unwrap();
        assert_eq!(options.backend, Some(Backend::Onnx));
        assert_eq!(options.model_package.as_deref(), Some("voice.dmp"));
        assert_eq!(options.model_package_key.as_deref(), Some("vendor.pub"));
    }

    #[test]
    fn explicit_false_config_overrides_mode_boolean_defaults() {
        let options = parse_config(
            r#"
mode = "speech"
adaptive_noise = false
vad = false
"#,
            "explicit-false.toml",
        )
        .unwrap();

        let config = build_config(&options, 48_000);
        assert!(!config.adaptive_noise);
        assert!(!config.vad);
    }

    #[test]
    fn rejects_invalid_desktop_enum_values() {
        let error = parse_config("aac_encoder = \"invalid\"", "desktop.toml").unwrap_err();
        assert!(error.contains("unknown AAC encoder in config: invalid"));

        let error = parse_config("sgmse_profile = \"invalid\"", "desktop.toml").unwrap_err();
        assert!(error.contains("unknown SGMSE profile in config: invalid"));

        let error = parse_config("quality = \"impossible\"", "desktop.toml").unwrap_err();
        assert!(error.contains("unknown quality in config: impossible"));

        let error = parse_config("accelerator = \"vulkan\"", "desktop.toml").unwrap_err();
        assert!(error.contains("unknown accelerator in config: vulkan"));
    }

    #[test]
    fn accepts_legacy_desktop_true_peak_without_loudness() {
        let options = parse_config(
            "true_peak_dbtp = -1.0\nmp3_bitrate_kbps = 192\n",
            "legacy-desktop.toml",
        )
        .unwrap();
        assert_eq!(options.loudness_lufs, None);
        assert_eq!(options.true_peak_dbtp, None);

        let explicit = parse_config("true_peak_dbtp = -0.5", "manual.toml").unwrap();
        assert_eq!(explicit.true_peak_dbtp, Some(-0.5));
    }

    #[test]
    fn rejects_unknown_config_keys() {
        let error = parse_config("strenth = 0.5", "test.toml").unwrap_err();
        assert!(error.contains("unknown field"));
    }

    #[test]
    fn validates_configured_dpss_time_bandwidth_product() {
        for invalid in ["nan", "inf", "+inf", "-inf", "0.0", "-0.5", "8.000001"] {
            let options = parse_config(
                &format!("window = \"dpss\"\ndpss_nw = {invalid}"),
                "test.toml",
            )
            .unwrap();
            let error = validate_effective_options(&options, VALIDATION_SAMPLE_RATE).unwrap_err();
            assert!(
                error.contains("DPSS") || error.contains("dpss"),
                "unexpected error for {invalid}: {error}"
            );
        }

        let options = parse_config("window = \"dpss\"\ndpss_nw = 8.0", "test.toml").unwrap();
        validate_effective_options(&options, VALIDATION_SAMPLE_RATE).unwrap();
        assert_eq!(options.dpss_nw, Some(8.0));
    }

    #[test]
    fn invalid_active_dpss_nw_is_rejected_before_input_or_output_io() {
        let root = std::env::temp_dir().join(format!(
            "denoize-dpss-preflight-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let input = root.join("missing.wav");
        let output = root.join("output.wav");
        let error = run(&[
            input.to_string_lossy().into_owned(),
            output.to_string_lossy().into_owned(),
            "--window".into(),
            "dpss".into(),
            "--dpss-nw".into(),
            "9".into(),
        ])
        .unwrap_err();
        assert!(error.contains("dpss") || error.contains("DPSS"));
        assert!(!output.exists());
    }

    #[test]
    fn explicit_dpss_window_takes_precedence_over_ultra_quality() {
        let explicit = Overrides {
            window: Some(WindowType::Dpss),
            dpss_nw: Some(4.0),
            quality: Some("ultra".into()),
            ..Overrides::default()
        };
        let config = build_config(&explicit, 48_000);
        assert_eq!(config.window, WindowType::Dpss);
        assert_eq!(config.window_params.dpss_bandwidth, 4.0);

        let implicit = Overrides {
            quality: Some("ultra".into()),
            ..Overrides::default()
        };
        let config = build_config(&implicit, 48_000);
        assert_eq!(config.window, WindowType::Kaiser);
        assert_eq!(config.window_params.kaiser_beta, 10.0);
    }

    #[test]
    fn command_line_overrides_config_defaults() {
        let path = write_test_config(
            "backend = \"auto\"\nstrength = 0.25\ndpss_nw = 2.5\n",
            "config",
        );
        let args = vec![
            "input.wav".into(),
            "output.wav".into(),
            "--config".into(),
            path.to_string_lossy().into_owned(),
            "--backend".into(),
            "classical".into(),
            "--strength".into(),
            "0.75".into(),
            "--dpss-nw".into(),
            "4.0".into(),
        ];
        let (_, _, options) = parse_args(&args).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(options.backend, Some(Backend::Classical));
        assert!(!options.auto_backend);
        assert_eq!(options.strength, Some(0.75));
        assert_eq!(options.dpss_nw, Some(4.0));
    }

    #[test]
    fn command_line_model_selection_replaces_the_other_config_representation() {
        if Backend::parse("onnx").is_none() {
            return;
        }
        let raw = write_test_config(
            "backend = \"onnx\"\nonnx_model = \"raw.onnx\"\nonnx_rate = 48000\n",
            "raw-model-config",
        );
        let (_, _, package) = parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--config".into(),
            raw.to_string_lossy().into_owned(),
            "--model-package".into(),
            "voice.dmp".into(),
            "--model-package-key".into(),
            "vendor.pub".into(),
        ])
        .unwrap();
        std::fs::remove_file(raw).unwrap();
        assert!(package.onnx_model.is_none());
        assert!(package.onnx_sample_rate.is_none());
        assert_eq!(package.model_package.as_deref(), Some("voice.dmp"));
        assert_eq!(package.model_package_key.as_deref(), Some("vendor.pub"));

        let packaged = write_test_config(
            "backend = \"onnx\"\nmodel_package = \"old.dmp\"\nmodel_package_key = \"old.pub\"\n",
            "package-model-config",
        );
        let (_, _, raw) = parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--config".into(),
            packaged.to_string_lossy().into_owned(),
            "--onnx-model".into(),
            "replacement.onnx".into(),
            "--onnx-rate".into(),
            "16000".into(),
        ])
        .unwrap();
        std::fs::remove_file(packaged).unwrap();
        assert_eq!(raw.onnx_model.as_deref(), Some("replacement.onnx"));
        assert_eq!(raw.onnx_sample_rate, Some(16_000));
        assert!(raw.model_package.is_none());
        assert!(raw.model_package_key.is_none());
    }

    #[test]
    fn numeric_cli_values_override_invalid_toml_defaults_before_validation() {
        let path = write_test_config(
            r#"
strength = nan
profile_ms = inf
frame_size = 131072
window = "kaiser"
kaiser_beta = nan
dpss_nw = 9.0
loudness_lufs = nan
true_peak_dbtp = -30.0
onnx_rate = 0
stream_frames = 0
max_memory_mb = 0
jobs = 33
chunk_ms = 2001
live_latency_ms = 5001
max_drift_ppm = 10001
reconnect_timeout_ms = 300001
"#,
            "numeric-precedence",
        );
        let args = vec![
            "input.wav".into(),
            "output.wav".into(),
            "--config".into(),
            path.to_string_lossy().into_owned(),
            "--strength".into(),
            "0.5".into(),
            "--profile".into(),
            "-1".into(),
            "--frame".into(),
            "256".into(),
            "--dpss-nw".into(),
            "4".into(),
            "--kaiser-beta".into(),
            "8".into(),
            "--loudness".into(),
            "-16".into(),
            "--true-peak".into(),
            "-1".into(),
            "--onnx-rate".into(),
            "16000".into(),
            "--stream-frames".into(),
            "1".into(),
            "--max-memory".into(),
            "1".into(),
            "--jobs".into(),
            "1".into(),
            "--chunk-ms".into(),
            "100".into(),
            "--live-latency".into(),
            "80".into(),
            "--max-drift-ppm".into(),
            "1500".into(),
            "--reconnect-timeout".into(),
            "45000".into(),
        ];
        let (_, _, options) = parse_args(&args).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(options.strength, Some(0.5));
        assert_eq!(options.profile_ms, Some(-1.0));
        assert_eq!(options.frame_size, Some(256));
        assert_eq!(options.stream_frames, Some(1));
        assert_eq!(options.jobs, Some(1));
        assert_eq!(options.live_latency_ms, Some(80));
        assert_eq!(options.max_drift_ppm, Some(1_500));
        assert_eq!(options.reconnect_timeout_ms, Some(45_000));
    }

    #[test]
    fn invalid_toml_enum_is_not_hidden_by_a_cli_override() {
        let path = write_test_config("quality = \"impossible\"\n", "enum-precedence");
        let error = parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--config".into(),
            path.to_string_lossy().into_owned(),
            "--quality".into(),
            "high".into(),
        ])
        .unwrap_err();
        std::fs::remove_file(path).unwrap();
        assert!(error.contains("unknown quality in config"));

        let path = write_test_config("output_format = \"wma\"\n", "format-precedence");
        let error = parse_args(&[
            "input".into(),
            "output".into(),
            "--config".into(),
            path.to_string_lossy().into_owned(),
            "--output-format".into(),
            "wav".into(),
        ])
        .unwrap_err();
        std::fs::remove_file(path).unwrap();
        assert!(error.contains("unsupported --output-format"));
    }

    #[test]
    fn rejects_unknown_cli_quality_and_normalizes_legacy_aliases() {
        assert!(cli_error(&["--quality", "impossible"]).contains("unknown quality"));
        assert!(cli_error(&["--accelerator", "vulkan"]).contains("unknown accelerator"));
        let (_, _, options) = parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--quality".into(),
            "highest".into(),
        ])
        .unwrap();
        assert_eq!(options.quality.as_deref(), Some("ultra"));
    }

    #[test]
    fn rejects_non_finite_external_float_values() {
        for value in ["NaN", "inf", "-inf"] {
            for (flag, prefix) in [
                ("--strength", &[][..]),
                ("--profile", &[][..]),
                ("--overlap", &[][..]),
                ("--kaiser-beta", &["--window", "kaiser"][..]),
                ("--dpss-nw", &["--window", "dpss"][..]),
                ("--smoothing", &[][..]),
                ("--makeup", &[][..]),
                ("--loudness", &[][..]),
                ("--true-peak", &["--loudness", "-16"][..]),
            ] {
                let mut extra = prefix.to_vec();
                extra.extend([flag, value]);
                let error = cli_error(&extra);
                assert!(
                    error.contains("finite"),
                    "{flag}={value} produced unexpected error: {error}"
                );
            }
        }
    }

    #[test]
    fn validates_external_float_and_rate_boundaries() {
        for (flag, minimum, maximum, below, above, prefix) in [
            ("--strength", "0", "1", "-0.001", "1.001", &[][..]),
            ("--overlap", "0.5", "0.95", "0.499", "0.951", &[][..]),
            ("--smoothing", "0", "1", "-0.001", "1.001", &[][..]),
            ("--makeup", "-120", "120", "-120.001", "120.001", &[][..]),
            (
                "--kaiser-beta",
                "0",
                "50",
                "-0.001",
                "50.001",
                &["--window", "kaiser"][..],
            ),
            (
                "--dpss-nw",
                "0.001",
                "8",
                "0",
                "8.001",
                &["--window", "dpss"][..],
            ),
            ("--loudness", "-70", "0", "-70.001", "0.001", &[][..]),
            (
                "--true-peak",
                "-20",
                "0",
                "-20.001",
                "0.001",
                &["--loudness", "-16"][..],
            ),
        ] {
            for value in [minimum, maximum] {
                let mut extra = prefix.to_vec();
                extra.extend([flag, value]);
                cli_ok(&extra);
            }
            for value in [below, above] {
                let mut extra = prefix.to_vec();
                extra.extend([flag, value]);
                assert!(!cli_error(&extra).is_empty());
            }
        }

        cli_ok(&["--profile", "60000"]);
        assert!(cli_error(&["--profile", "60000.001"]).contains("profile"));
        for value in ["1", "768000"] {
            cli_ok(&["--onnx-rate", value]);
        }
        for value in ["0", "768001"] {
            assert!(cli_error(&["--onnx-rate", value]).contains("onnx-rate"));
        }
    }

    #[test]
    fn validates_frame_resource_and_live_boundaries() {
        for value in ["0", "255", "257", "65537", "131072"] {
            assert!(cli_error(&["--frame", value]).contains("frame"));
        }
        for value in ["256", "65536"] {
            parse_args(&[
                "input.wav".into(),
                "output.wav".into(),
                "--frame".into(),
                value.into(),
            ])
            .unwrap();
        }

        for value in ["0", "1048577"] {
            assert!(cli_error(&["--stream-frames", value]).contains("stream-frames"));
        }
        for value in ["1", "1048576"] {
            parse_args(&[
                "input.wav".into(),
                "output.wav".into(),
                "--stream-frames".into(),
                value.into(),
            ])
            .unwrap();
        }

        for value in ["0", "33"] {
            assert!(cli_error(&["--jobs", value]).contains("--jobs"));
            assert!(cli_error(&["--max-gpu-jobs", value]).contains("--max-gpu-jobs"));
        }
        for value in ["1", "32"] {
            parse_args(&[
                "input.wav".into(),
                "output.wav".into(),
                "--jobs".into(),
                value.into(),
            ])
            .unwrap();
            parse_args(&[
                "input.wav".into(),
                "output.wav".into(),
                "--max-gpu-jobs".into(),
                value.into(),
            ])
            .unwrap();
        }

        for value in ["9", "2001"] {
            assert!(cli_error(&["--chunk-ms", value]).contains("--chunk-ms"));
        }
        for value in ["10", "2000"] {
            parse_args(&[
                "input.wav".into(),
                "output.wav".into(),
                "--chunk-ms".into(),
                value.into(),
            ])
            .unwrap();
        }

        for value in ["0", "20", "5000"] {
            cli_ok(&["--live-latency", value]);
        }
        for value in ["1", "19", "5001"] {
            assert!(cli_error(&["--live-latency", value]).contains("--live-latency"));
        }
        for value in ["0", "10000"] {
            cli_ok(&["--max-drift-ppm", value]);
        }
        assert!(cli_error(&["--max-drift-ppm", "10001"]).contains("--max-drift-ppm"));
        for value in ["0", "300000"] {
            cli_ok(&["--reconnect-timeout", value]);
        }
        assert!(cli_error(&["--reconnect-timeout", "300001"]).contains("--reconnect-timeout"));
    }

    #[test]
    fn rejects_hostile_integer_values_without_arithmetic_overflow() {
        let usize_max = usize::MAX.to_string();
        for (flag, field) in [
            ("--frame", "frame_size"),
            ("--stream-frames", "stream-frames"),
            ("--jobs", "--jobs"),
            ("--max-memory", "--max-memory"),
            ("--max-process-memory", "--max-process-memory"),
            ("--max-temp-space", "--max-temp-space"),
            ("--max-gpu-memory", "--max-gpu-memory"),
            ("--max-gpu-jobs", "--max-gpu-jobs"),
        ] {
            let error = cli_error(&[flag, &usize_max]);
            assert!(error.contains(field), "{flag} produced: {error}");
        }
        assert!(cli_error(&["--chunk-ms", &u32::MAX.to_string()]).contains("--chunk-ms"));
        assert!(cli_error(&["--live-latency", &u32::MAX.to_string()]).contains("--live-latency"));
        assert!(cli_error(&["--max-drift-ppm", &u32::MAX.to_string()]).contains("--max-drift-ppm"));
        assert!(cli_error(&["--reconnect-timeout", &u32::MAX.to_string()])
            .contains("--reconnect-timeout"));
    }

    #[test]
    fn invalid_configuration_precedes_missing_input() {
        let error = parse_args(&["--strength".into(), "NaN".into()]).unwrap_err();
        assert!(error.contains("strength"));
        assert!(!error.contains("missing INPUT"));

        let error = parse_args(&["--jobs".into(), "33".into()]).unwrap_err();
        assert!(error.contains("--jobs"));
        assert!(!error.contains("missing INPUT"));

        let error =
            parse_args(&["--batch".into(), "--output-format".into(), "wma".into()]).unwrap_err();
        assert!(error.contains("unsupported --output-format"));
        assert!(!error.contains("missing INPUT"));
    }

    #[test]
    fn process_resource_flags_are_merged_and_validated_before_io() {
        let (_, _, options) = parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--max-memory".into(),
            "96".into(),
            "--max-process-memory".into(),
            "64".into(),
            "--max-temp-space".into(),
            "32".into(),
            "--max-gpu-memory".into(),
            "256".into(),
            "--max-gpu-jobs".into(),
            "3".into(),
            "--isolate".into(),
        ])
        .unwrap();
        assert_eq!(effective_input_memory_mb(&options), Some(64));
        let governor = resource_governor(&options, 4).unwrap();
        assert_eq!(
            governor.limits().max_memory_bytes(),
            Some(64 * BYTES_PER_MIB)
        );
        assert_eq!(
            governor.limits().max_temporary_bytes(),
            Some(32 * BYTES_PER_MIB)
        );
        assert_eq!(
            governor.limits().max_gpu_memory_bytes(),
            Some(256 * BYTES_PER_MIB)
        );
        assert_eq!(governor.limits().max_cpu_jobs(), Some(4));
        assert_eq!(governor.limits().max_gpu_jobs(), Some(3));
        assert!(options.isolate);

        for (flag, expected) in [
            ("--max-process-memory", "--max-process-memory"),
            ("--max-temp-space", "--max-temp-space"),
            ("--max-gpu-memory", "--max-gpu-memory"),
        ] {
            let error = parse_args(&[flag.into(), "0".into()]).unwrap_err();
            assert!(error.contains(expected));
            assert!(!error.contains("missing INPUT"));
        }
    }

    #[test]
    fn preserves_profile_and_true_peak_sentinel_semantics() {
        for profile in ["-1000000", "-1", "0"] {
            parse_args(&[
                "input.wav".into(),
                "output.wav".into(),
                "--profile".into(),
                profile.into(),
            ])
            .unwrap();
        }
        parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--loudness".into(),
            "-16".into(),
            "--true-peak".into(),
            "-1".into(),
        ])
        .unwrap();
    }

    #[test]
    fn default_batch_worker_count_is_bounded() {
        let jobs = effective_batch_jobs(&Overrides::default());
        assert!((1..=MAX_BATCH_JOBS).contains(&jobs));
    }

    #[test]
    fn parses_explicit_downmix_mode() {
        let (_, _, options) = parse_args(&[
            "input.wav".into(),
            "output.mp3".into(),
            "--downmix".into(),
            "stereo".into(),
        ])
        .unwrap();
        assert_eq!(options.downmix, Some(DownmixMode::Stereo));
    }

    #[test]
    fn parses_deterministic_seed_and_implies_mode() {
        let (_, _, options) = parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--seed".into(),
            "42".into(),
        ])
        .unwrap();
        assert!(options.deterministic);
        assert_eq!(options.seed, Some(42));
    }

    #[test]
    fn evaluation_cli_rejects_ambiguous_or_incomplete_contracts_before_io() {
        let arguments = |values: &[&str]| {
            values
                .iter()
                .map(|value| (*value).to_string())
                .collect::<Vec<_>>()
        };

        let error = run_evaluate(&arguments(&["validate", "manifest.json"])).unwrap_err();
        assert_eq!(error, "evaluate validate requires --corpus-root");

        let error = run_evaluate(&arguments(&[
            "run",
            "manifest.json",
            "--corpus-root",
            "corpus",
            "--key",
            "secret.json",
        ]))
        .unwrap_err();
        assert_eq!(error, "evaluate run requires --output");

        let error = run_evaluate(&arguments(&[
            "compare",
            "baseline.json",
            "candidate.json",
            "--key",
            "shared.json",
            "--baseline-key",
            "baseline-key.json",
        ]))
        .unwrap_err();
        assert!(error.contains("either --key or separate"));

        let error = run_evaluate(&arguments(&[
            "verify",
            "result.json",
            "--key",
            "public.json",
            "--json",
            "--pretty",
        ]))
        .unwrap_err();
        assert_eq!(error, "evaluate accepts only one of --json or --pretty");

        let error =
            run_evaluate(&arguments(&["validate", "manifest.json", "--corpus-root"])).unwrap_err();
        assert_eq!(error, "missing value for --corpus-root");
    }
}

fn evaluation_usage() -> &'static str {
    "\
Run reproducible licensed-corpus release evaluation.

USAGE:
    denoize evaluate validate <MANIFEST.json> --corpus-root <DIR> [--json|--pretty]
    denoize evaluate run <MANIFEST.json> --corpus-root <DIR> --key <SECRET_KEY.json> --output <RESULT.json> [--listening-result <RESULT.json>] [--json|--pretty]
    denoize evaluate verify <RESULT.json> --key <PUBLIC_KEY.json> [--manifest <MANIFEST.json>] [--json|--pretty]
    denoize evaluate compare <BASELINE.json> <CANDIDATE.json> (--key <PUBLIC_KEY.json> | --baseline-key <PUBLIC_KEY.json> --candidate-key <PUBLIC_KEY.json>) [--json|--pretty]

The manifest pins every corpus/model artifact by license, immutable source
revision, preparation digest, byte length, and SHA-256. Artifact paths must be
portable regular files below --corpus-root and may not traverse symlinks.

`run` always writes a signed result before returning a non-zero status for a
missed threshold or rejected listening test. `compare` authenticates both
results and rejects incomparable hardware/runtime/recipe contexts.
"
}

#[derive(Clone, Copy, Default)]
struct EvaluationPrintMode {
    json: bool,
    pretty: bool,
}

fn run_evaluate(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        if args.len() != 1 {
            return Err("evaluate --help accepts no other arguments".into());
        }
        print!("{}", evaluation_usage());
        return Ok(());
    }
    match args.first().map(String::as_str) {
        Some("validate") => run_evaluate_validate(&args[1..]),
        Some("run") => run_evaluate_run(&args[1..]),
        Some("verify") => run_evaluate_verify(&args[1..]),
        Some("compare") => run_evaluate_compare(&args[1..]),
        Some(command) => Err(format!("unknown evaluate command: {command}")),
        None => Err("evaluate requires a command (run `denoize evaluate --help`)".into()),
    }
}

fn run_evaluate_validate(args: &[String]) -> Result<(), String> {
    let manifest_path = evaluation_positional(args, 0, "evaluate validate requires MANIFEST.json")?;
    let mut corpus_root = None;
    let mut mode = EvaluationPrintMode::default();
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--corpus-root" => {
                set_evaluation_option(
                    &mut corpus_root,
                    evaluation_value(args, &mut index, "--corpus-root")?,
                    "--corpus-root",
                )?;
            }
            "--json" => set_evaluation_print_mode(&mut mode, false)?,
            "--pretty" => set_evaluation_print_mode(&mut mode, true)?,
            value => return Err(format!("unknown evaluate validate option: {value}")),
        }
        index += 1;
    }
    let corpus_root = corpus_root.ok_or("evaluate validate requires --corpus-root")?;
    let manifest = denoize::EvaluationManifest::from_file(manifest_path)?;
    let report = denoize::validate_evaluation_corpus(&manifest, corpus_root)?;
    if mode.json || mode.pretty {
        print_evaluation_json(&report, mode)?;
    } else {
        println!(
            "validated corpus {} {}: {} cases, {:.3}s audio, manifest {}",
            report.corpus_id,
            report.corpus_version,
            report.cases,
            report.total_audio_seconds,
            report.manifest_digest
        );
    }
    Ok(())
}

fn run_evaluate_run(args: &[String]) -> Result<(), String> {
    let manifest_path = evaluation_positional(args, 0, "evaluate run requires MANIFEST.json")?;
    let mut corpus_root = None;
    let mut secret_key = None;
    let mut output = None;
    let mut listening_result = None;
    let mut mode = EvaluationPrintMode::default();
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--corpus-root" => set_evaluation_option(
                &mut corpus_root,
                evaluation_value(args, &mut index, "--corpus-root")?,
                "--corpus-root",
            )?,
            "--key" => set_evaluation_option(
                &mut secret_key,
                evaluation_value(args, &mut index, "--key")?,
                "--key",
            )?,
            "--output" => set_evaluation_option(
                &mut output,
                evaluation_value(args, &mut index, "--output")?,
                "--output",
            )?,
            "--listening-result" => set_evaluation_option(
                &mut listening_result,
                evaluation_value(args, &mut index, "--listening-result")?,
                "--listening-result",
            )?,
            "--json" => set_evaluation_print_mode(&mut mode, false)?,
            "--pretty" => set_evaluation_print_mode(&mut mode, true)?,
            value => return Err(format!("unknown evaluate run option: {value}")),
        }
        index += 1;
    }
    let corpus_root = corpus_root.ok_or("evaluate run requires --corpus-root")?;
    let secret_key = secret_key.ok_or("evaluate run requires --key")?;
    let output = output.ok_or("evaluate run requires --output")?;
    let manifest = denoize::EvaluationManifest::from_file(manifest_path)?;
    let key = ReceiptSecretKey::from_file(secret_key)?;
    let result = denoize::run_evaluation(
        &manifest,
        corpus_root,
        &key,
        listening_result.as_deref().map(std::path::Path::new),
    )?;
    denoize::write_signed_evaluation_result(&output, &result)?;
    if mode.json || mode.pretty {
        print_evaluation_json(&result, mode)?;
    } else {
        println!(
            "wrote {} evaluation evidence for {} cases: {}",
            if result.payload.accepted {
                "accepted"
            } else {
                "rejected"
            },
            result.payload.cases.len(),
            output
        );
        for outcome in result
            .payload
            .threshold_outcomes
            .iter()
            .filter(|outcome| !outcome.passed)
        {
            println!(
                "failed threshold {}: observed {}, limit {}",
                outcome.metric, outcome.observed, outcome.limit
            );
        }
    }
    if !result.payload.accepted {
        return Err(format!(
            "evaluation evidence was published to {output}, but its acceptance policy failed"
        ));
    }
    Ok(())
}

fn run_evaluate_verify(args: &[String]) -> Result<(), String> {
    let result_path = evaluation_positional(args, 0, "evaluate verify requires RESULT.json")?;
    let mut public_key = None;
    let mut manifest_path = None;
    let mut mode = EvaluationPrintMode::default();
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--key" => set_evaluation_option(
                &mut public_key,
                evaluation_value(args, &mut index, "--key")?,
                "--key",
            )?,
            "--manifest" => set_evaluation_option(
                &mut manifest_path,
                evaluation_value(args, &mut index, "--manifest")?,
                "--manifest",
            )?,
            "--json" => set_evaluation_print_mode(&mut mode, false)?,
            "--pretty" => set_evaluation_print_mode(&mut mode, true)?,
            value => return Err(format!("unknown evaluate verify option: {value}")),
        }
        index += 1;
    }
    let public_key = public_key.ok_or("evaluate verify requires --key")?;
    let result = denoize::SignedEvaluationResult::from_file(result_path)?;
    let key = ReceiptPublicKey::from_file(public_key)?;
    let manifest = manifest_path
        .as_deref()
        .map(denoize::EvaluationManifest::from_file)
        .transpose()?;
    let report = denoize::verify_evaluation_result(&result, &key, manifest.as_ref())?;
    if mode.json || mode.pretty {
        print_evaluation_json(&report, mode)?;
    } else {
        println!(
            "verified {} evaluation result for {} cases with key {}",
            if report.accepted {
                "accepted"
            } else {
                "rejected"
            },
            report.cases,
            report.key_id
        );
    }
    if !report.accepted {
        return Err("verified evaluation result is not accepted".into());
    }
    Ok(())
}

fn run_evaluate_compare(args: &[String]) -> Result<(), String> {
    let baseline_path = evaluation_positional(
        args,
        0,
        "evaluate compare requires BASELINE.json and CANDIDATE.json",
    )?;
    let candidate_path = evaluation_positional(
        args,
        1,
        "evaluate compare requires BASELINE.json and CANDIDATE.json",
    )?;
    let mut shared_key = None;
    let mut baseline_key = None;
    let mut candidate_key = None;
    let mut mode = EvaluationPrintMode::default();
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--key" => set_evaluation_option(
                &mut shared_key,
                evaluation_value(args, &mut index, "--key")?,
                "--key",
            )?,
            "--baseline-key" => set_evaluation_option(
                &mut baseline_key,
                evaluation_value(args, &mut index, "--baseline-key")?,
                "--baseline-key",
            )?,
            "--candidate-key" => set_evaluation_option(
                &mut candidate_key,
                evaluation_value(args, &mut index, "--candidate-key")?,
                "--candidate-key",
            )?,
            "--json" => set_evaluation_print_mode(&mut mode, false)?,
            "--pretty" => set_evaluation_print_mode(&mut mode, true)?,
            value => return Err(format!("unknown evaluate compare option: {value}")),
        }
        index += 1;
    }
    if shared_key.is_some() && (baseline_key.is_some() || candidate_key.is_some()) {
        return Err(
            "evaluate compare accepts either --key or separate baseline/candidate keys".into(),
        );
    }
    let baseline_key_path = shared_key
        .as_deref()
        .or(baseline_key.as_deref())
        .ok_or("evaluate compare requires --key or --baseline-key")?;
    let candidate_key_path = shared_key
        .as_deref()
        .or(candidate_key.as_deref())
        .ok_or("evaluate compare requires --key or --candidate-key")?;
    let baseline = denoize::SignedEvaluationResult::from_file(baseline_path)?;
    let candidate = denoize::SignedEvaluationResult::from_file(candidate_path)?;
    let baseline_key = ReceiptPublicKey::from_file(baseline_key_path)?;
    let candidate_key = ReceiptPublicKey::from_file(candidate_key_path)?;
    let report =
        denoize::compare_evaluation_results(&baseline, &baseline_key, &candidate, &candidate_key)?;
    if mode.json || mode.pretty {
        print_evaluation_json(&report, mode)?;
    } else {
        println!(
            "evaluation regression comparison: {} ({} -> {})",
            if report.passed { "passed" } else { "failed" },
            report.baseline_version,
            report.candidate_version
        );
        for regression in &report.regressions {
            println!(
                "{}: baseline {}, candidate {}, regression {} <= {} ({})",
                regression.metric,
                regression.baseline,
                regression.candidate,
                regression.regression,
                regression.limit,
                if regression.passed { "pass" } else { "fail" }
            );
        }
    }
    if !report.passed {
        return Err("evaluation regression policy failed".into());
    }
    Ok(())
}

fn evaluation_positional<'a>(
    args: &'a [String],
    index: usize,
    error: &str,
) -> Result<&'a str, String> {
    args.get(index)
        .filter(|value| !value.is_empty() && !value.starts_with('-'))
        .map(String::as_str)
        .ok_or_else(|| error.to_string())
}

fn evaluation_value(args: &[String], index: &mut usize, option: &str) -> Result<String, String> {
    *index = index
        .checked_add(1)
        .ok_or("evaluation argument index overflow")?;
    args.get(*index)
        .filter(|value| !value.is_empty() && !value.starts_with('-'))
        .cloned()
        .ok_or_else(|| format!("missing value for {option}"))
}

fn set_evaluation_option(
    target: &mut Option<String>,
    value: String,
    option: &str,
) -> Result<(), String> {
    if target.replace(value).is_some() {
        Err(format!("{option} may be supplied only once"))
    } else {
        Ok(())
    }
}

fn set_evaluation_print_mode(mode: &mut EvaluationPrintMode, pretty: bool) -> Result<(), String> {
    if mode.json || mode.pretty {
        return Err("evaluate accepts only one of --json or --pretty".into());
    }
    mode.json = !pretty;
    mode.pretty = pretty;
    Ok(())
}

fn print_evaluation_json<T: Serialize>(value: &T, mode: EvaluationPrintMode) -> Result<(), String> {
    let json = if mode.pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .map_err(|error| format!("serialize evaluation CLI result: {error}"))?;
    println!("{json}");
    Ok(())
}

fn run_metrics(args: &[String]) -> Result<(), String> {
    let reference = args.first().ok_or("metrics requires REFERENCE and TEST")?;
    let test = args.get(1).ok_or("metrics requires REFERENCE and TEST")?;
    let report =
        denoize::benchmark::BenchmarkReport::compare(&read_audio(reference)?, &read_audio(test)?)?;
    if args.iter().any(|argument| argument == "--json") {
        println!("{}", report.json());
    } else {
        println!("{}", report.markdown());
    }
    Ok(())
}

fn run_compare(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("compare requires CLEAN NOISY ENHANCED".into());
    }
    if args[3..]
        .iter()
        .any(|argument| argument != "--json" && argument != "--html")
    {
        return Err("compare accepts only --json or --html after the input files".into());
    }
    if args.iter().any(|argument| argument == "--json")
        && args.iter().any(|argument| argument == "--html")
    {
        return Err("compare accepts only one output format".into());
    }
    let clean = args
        .first()
        .ok_or("compare requires CLEAN NOISY ENHANCED")?;
    let noisy = args.get(1).ok_or("compare requires CLEAN NOISY ENHANCED")?;
    let enhanced = args.get(2).ok_or("compare requires CLEAN NOISY ENHANCED")?;
    let report = denoize::benchmark::ComparisonReport::compare(
        &read_audio(clean)?,
        &read_audio(noisy)?,
        &read_audio(enhanced)?,
    )?;
    if args.iter().any(|argument| argument == "--json") {
        println!("{}", report.json());
    } else if args.iter().any(|argument| argument == "--html") {
        println!("{}", report.html());
    } else {
        println!("{}", report.markdown());
    }
    Ok(())
}

fn models_usage() -> &'static str {
    "\
Manage verified external models.

USAGE:
    denoize models list
    denoize models info <MODEL|all>
    denoize models install <MODEL|all> [DOWNLOAD OPTIONS]
    denoize models install <MODEL> --from <PATH>
    denoize models update <MODEL|all> [DOWNLOAD OPTIONS]
    denoize models verify <MODEL|all>
    denoize models doctor
    denoize models repair <MODEL|all> [DOWNLOAD OPTIONS]
    denoize models prune [--dry-run]
    denoize models remove <MODEL|all>
    denoize models path <MODEL|all>
    denoize models catalog status
    denoize models catalog update [DOWNLOAD OPTIONS]
    denoize models catalog import <CATALOG.json> <CATALOG.json.sig>
    denoize models catalog trust status
    denoize models catalog trust import <TRUST-ROOT.json> <SIGNATURES.json>
    denoize models catalog trust recover
    denoize models catalog trust reset-time-floor
    denoize models bundle inspect <BUNDLE.dmb>
    denoize models bundle import <BUNDLE.dmb>
    denoize models bundle create <OUTPUT.dmb> <CATALOG.json> <CATALOG.json.sig> <TRUST-ROOT.json> <COMPONENTS-DIR>
    denoize models package inspect <PACKAGE.dmp> <MINISIGN.pub>
    denoize models package license <PACKAGE.dmp> <MINISIGN.pub>
    denoize models package create <OUTPUT.dmp> <MANIFEST.json> <MANIFEST.json.sig> <MINISIGN.pub> <MODEL.onnx> <LICENSE>
    denoize models package create-v2 <OUTPUT.dmp> <MANIFEST.json> <MANIFEST.json.sig> <MINISIGN.pub> <COMPONENTS-DIR>
    denoize models snapshot [--json] [--pretty]
    denoize models cache-dir

DOWNLOAD OPTIONS:
        --offline                  never access the network; use only verified cached data
        --proxy <URL>              use this proxy instead of proxy environment variables
        --no-proxy                 connect directly and ignore proxy environment variables
        --url <URL>                alternate model URL; catalog update requires HTTPS JSON
        --bearer-token-env <VAR>   read a bearer token from environment variable VAR
        --basic-user <USER>        username for HTTP Basic authentication
        --basic-password-env <VAR> read the Basic password from environment variable VAR
        --from <PATH>              install one MODEL from a local file (install only)

Bearer tokens and Basic passwords are read from environment variables instead
of literal secret flags. Basic authentication requires both --basic-user and
--basic-password-env. Signed --url values and proxy credentials can still be
visible in process arguments. Alternate sources, origin authentication, and
--from accept one model, not `all`; --url rejects userinfo credentials.

ENVIRONMENT:
    DENOIZE_MODEL_OFFLINE, DENOIZE_MODEL_URL, DENOIZE_MODEL_CATALOG_URL,
    DENOIZE_MODEL_PROXY,
    DENOIZE_MODEL_BEARER_TOKEN, DENOIZE_MODEL_USERNAME, DENOIZE_MODEL_PASSWORD
    HTTPS_PROXY, HTTP_PROXY, ALL_PROXY, NO_PROXY (and lowercase variants)
"
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelCommand {
    Info,
    Install,
    Update,
    Verify,
    Repair,
    Remove,
    Path,
}

#[derive(Debug)]
enum ParsedModelsCommand {
    Help,
    List,
    CacheDir,
    Doctor,
    Snapshot {
        pretty: bool,
    },
    Prune {
        dry_run: bool,
    },
    Run {
        command: ModelCommand,
        target: String,
        download_options: Option<Box<denoize::models::ModelDownloadOptions>>,
        source_file: Option<std::path::PathBuf>,
    },
}

fn models_option_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    *index += 1;
    let value = args
        .get(*index)
        .ok_or_else(|| format!("missing value for {flag}"))?;
    if value.is_empty() {
        return Err(format!("empty value for {flag}"));
    }
    Ok(value.clone())
}

fn validate_model_source_url(value: &str) -> Result<(), String> {
    let source = url::Url::parse(value)
        .map_err(|_| "invalid value for --url: expected an HTTP(S) URL".to_string())?;
    if !matches!(source.scheme(), "http" | "https") || source.host_str().is_none() {
        return Err("invalid value for --url: expected an HTTP(S) URL".into());
    }
    if !source.username().is_empty() || source.password().is_some() {
        return Err(
            "--url must not contain credentials; use --bearer-token-env or Basic authentication options"
                .into(),
        );
    }
    Ok(())
}

fn read_model_secret<F>(
    flag: &str,
    variable: &str,
    read_environment: &mut F,
) -> Result<String, String>
where
    F: FnMut(&str) -> Result<String, String>,
{
    if variable.trim().is_empty() {
        return Err(format!("empty environment variable name for {flag}"));
    }
    let secret = read_environment(variable).map_err(|error| {
        format!("failed to read environment variable {variable} for {flag}: {error}")
    })?;
    if secret.is_empty() {
        return Err(format!(
            "environment variable {variable} referenced by {flag} is empty"
        ));
    }
    Ok(secret)
}

fn parse_models_command<F>(
    args: &[String],
    mut download_options: denoize::models::ModelDownloadOptions,
    mut read_environment: F,
) -> Result<ParsedModelsCommand, String>
where
    F: FnMut(&str) -> Result<String, String>,
{
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
        || args.first().map(String::as_str) == Some("help")
    {
        return Ok(ParsedModelsCommand::Help);
    }

    let command_name = args.first().map(String::as_str).unwrap_or("list");
    if matches!(command_name, "list" | "cache-dir") {
        if args.len() > 1 {
            return Err(format!("models {command_name} accepts no arguments"));
        }
        return Ok(if command_name == "list" {
            ParsedModelsCommand::List
        } else {
            ParsedModelsCommand::CacheDir
        });
    }

    if command_name == "doctor" {
        if args.len() > 1 {
            return Err("models doctor accepts no arguments".into());
        }
        return Ok(ParsedModelsCommand::Doctor);
    }

    if command_name == "snapshot" {
        let mut pretty = false;
        let mut json_seen = false;
        for option in &args[1..] {
            match option.as_str() {
                "--pretty" if !pretty => pretty = true,
                "--json" if !json_seen => json_seen = true,
                "--pretty" | "--json" => {
                    return Err(format!("models snapshot option repeated: {option}"))
                }
                value => return Err(format!("unknown models snapshot option: {value}")),
            }
        }
        return Ok(ParsedModelsCommand::Snapshot { pretty });
    }

    if command_name == "prune" {
        let dry_run = match args.get(1).map(String::as_str) {
            None => false,
            Some("--dry-run") if args.len() == 2 => true,
            Some(value) => return Err(format!("unknown models prune option: {value}")),
        };
        return Ok(ParsedModelsCommand::Prune { dry_run });
    }

    let command = match command_name {
        "info" => ModelCommand::Info,
        "install" => ModelCommand::Install,
        "update" => ModelCommand::Update,
        "verify" => ModelCommand::Verify,
        "repair" => ModelCommand::Repair,
        "remove" => ModelCommand::Remove,
        "path" => ModelCommand::Path,
        _ => return Err(format!("unknown models command: {command_name}")),
    };
    let target = args
        .get(1)
        .filter(|target| !target.starts_with('-'))
        .ok_or_else(|| format!("models {command_name} requires MODEL|all"))?
        .clone();

    if !matches!(
        command,
        ModelCommand::Install | ModelCommand::Update | ModelCommand::Repair
    ) {
        if args.len() > 2 {
            return Err(format!(
                "models {command_name} does not accept options or extra arguments"
            ));
        }
        return Ok(ParsedModelsCommand::Run {
            command,
            target,
            download_options: None,
            source_file: None,
        });
    }

    let mut offline_seen = false;
    let mut proxy_flag: Option<&str> = None;
    let mut source_url_seen = false;
    let mut bearer_variable: Option<String> = None;
    let mut basic_user: Option<String> = None;
    let mut basic_password_variable: Option<String> = None;
    let mut source_file: Option<std::path::PathBuf> = None;
    let mut index = 2;
    while index < args.len() {
        let flag = args[index].as_str();
        match flag {
            "--offline" => {
                if offline_seen {
                    return Err("--offline specified more than once".into());
                }
                offline_seen = true;
                download_options.offline = true;
            }
            "--proxy" => {
                if let Some(previous) = proxy_flag {
                    return Err(format!("--proxy cannot be combined with {previous}"));
                }
                let value = models_option_value(args, &mut index, flag)?;
                proxy_flag = Some("--proxy");
                download_options.proxy = denoize::models::ModelProxy::Url(value);
            }
            "--no-proxy" => {
                if let Some(previous) = proxy_flag {
                    return Err(format!("--no-proxy cannot be combined with {previous}"));
                }
                proxy_flag = Some("--no-proxy");
                download_options.proxy = denoize::models::ModelProxy::Disabled;
            }
            "--url" => {
                if source_url_seen {
                    return Err("--url specified more than once".into());
                }
                let value = models_option_value(args, &mut index, flag)?;
                validate_model_source_url(&value)?;
                source_url_seen = true;
                download_options.source_url = Some(value);
            }
            "--bearer-token-env" => {
                if bearer_variable.is_some() {
                    return Err("--bearer-token-env specified more than once".into());
                }
                bearer_variable = Some(models_option_value(args, &mut index, flag)?);
            }
            "--basic-user" => {
                if basic_user.is_some() {
                    return Err("--basic-user specified more than once".into());
                }
                basic_user = Some(models_option_value(args, &mut index, flag)?);
            }
            "--basic-password-env" => {
                if basic_password_variable.is_some() {
                    return Err("--basic-password-env specified more than once".into());
                }
                basic_password_variable = Some(models_option_value(args, &mut index, flag)?);
            }
            "--from" => {
                if source_file.is_some() {
                    return Err("--from specified more than once".into());
                }
                source_file = Some(models_option_value(args, &mut index, flag)?.into());
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown models {command_name} option: {value}"));
            }
            value => {
                return Err(format!(
                    "unexpected argument for models {command_name}: {value}"
                ));
            }
        }
        index += 1;
    }

    if source_file.is_some() {
        if command != ModelCommand::Install {
            return Err("--from is supported only by `models install`".into());
        }
        if target == "all" {
            return Err("--from requires one MODEL and cannot be used with `all`".into());
        }
        if source_url_seen
            || proxy_flag.is_some()
            || bearer_variable.is_some()
            || basic_user.is_some()
            || basic_password_variable.is_some()
        {
            return Err("--from cannot be combined with network download options".into());
        }
        download_options = denoize::models::ModelDownloadOptions::default();
        download_options.offline = offline_seen;
    }

    if bearer_variable.is_some() && (basic_user.is_some() || basic_password_variable.is_some()) {
        return Err(
            "--bearer-token-env cannot be combined with Basic authentication options".into(),
        );
    }
    download_options.authentication = if let Some(variable) = bearer_variable {
        Some(denoize::models::ModelAuthentication::Bearer(
            read_model_secret("--bearer-token-env", &variable, &mut read_environment)?,
        ))
    } else {
        match (basic_user, basic_password_variable) {
            (Some(username), Some(variable)) => {
                let password =
                    read_model_secret("--basic-password-env", &variable, &mut read_environment)?;
                Some(denoize::models::ModelAuthentication::Basic { username, password })
            }
            (None, None) => download_options.authentication,
            _ => {
                return Err(
                    "--basic-user and --basic-password-env must be specified together".into(),
                )
            }
        }
    };

    if target == "all" && download_options.source_url.is_some() {
        return Err(
            "an alternate model URL requires one MODEL and cannot be used with `all`".into(),
        );
    }
    if target == "all" && download_options.authentication.is_some() {
        return Err("model authentication requires one MODEL and cannot be used with `all`".into());
    }

    Ok(ParsedModelsCommand::Run {
        command,
        target,
        download_options: Some(Box::new(download_options)),
        source_file,
    })
}

fn model_download_options_from_environment_with<F>(
    args: &[String],
    mut read_environment: F,
) -> Result<denoize::models::ModelDownloadOptions, String>
where
    F: FnMut(&str) -> Option<String>,
{
    if args.iter().any(|argument| argument == "--from") {
        return Ok(denoize::models::ModelDownloadOptions::default());
    }
    let overrides_offline = args.iter().any(|argument| argument == "--offline");
    let overrides_source = args.iter().any(|argument| argument == "--url");
    let overrides_proxy = args
        .iter()
        .any(|argument| matches!(argument.as_str(), "--proxy" | "--no-proxy"));
    let overrides_authentication = args.iter().any(|argument| {
        matches!(
            argument.as_str(),
            "--bearer-token-env" | "--basic-user" | "--basic-password-env"
        )
    });
    denoize::models::ModelDownloadOptions::from_env_with(|name| {
        let overridden = match name {
            "DENOIZE_MODEL_OFFLINE" => overrides_offline,
            "DENOIZE_MODEL_URL" => overrides_source,
            "DENOIZE_MODEL_PROXY" => overrides_proxy,
            "DENOIZE_MODEL_BEARER_TOKEN" | "DENOIZE_MODEL_USERNAME" | "DENOIZE_MODEL_PASSWORD" => {
                overrides_authentication
            }
            _ => false,
        };
        (!overridden).then(|| read_environment(name)).flatten()
    })
}

fn model_catalog_download_options_from_environment_with<F>(
    args: &[String],
    mut read_environment: F,
) -> Result<denoize::models::ModelDownloadOptions, String>
where
    F: FnMut(&str) -> Option<String>,
{
    let overrides_offline = args.iter().any(|argument| argument == "--offline");
    let overrides_source = args.iter().any(|argument| argument == "--url");
    let overrides_proxy = args
        .iter()
        .any(|argument| matches!(argument.as_str(), "--proxy" | "--no-proxy"));
    let overrides_authentication = args.iter().any(|argument| {
        matches!(
            argument.as_str(),
            "--bearer-token-env" | "--basic-user" | "--basic-password-env"
        )
    });
    denoize::models::ModelDownloadOptions::from_env_with(|name| {
        let overridden = match name {
            "DENOIZE_MODEL_OFFLINE" => overrides_offline,
            "DENOIZE_MODEL_URL" => overrides_source,
            "DENOIZE_MODEL_PROXY" => overrides_proxy,
            "DENOIZE_MODEL_BEARER_TOKEN" | "DENOIZE_MODEL_USERNAME" | "DENOIZE_MODEL_PASSWORD" => {
                overrides_authentication
            }
            _ => false,
        };
        let environment_name = if name == "DENOIZE_MODEL_URL" {
            "DENOIZE_MODEL_CATALOG_URL"
        } else {
            name
        };
        (!overridden)
            .then(|| read_environment(environment_name))
            .flatten()
    })
}

#[cfg(test)]
fn model_info_output(model: &denoize::models::ModelInfo, path: &std::path::Path) -> String {
    format!(
        "name: {}\nbackend: {}\nsample-rate: {}\nlicense: {}\nrevision: {}\nsize-bytes: {}\nsha256: {}\nurl: {}\npath: {}\n",
        model.name,
        model.backend,
        model.sample_rate,
        model.license,
        model.revision,
        model.size_bytes,
        model.sha256,
        denoize::models::redact_url(model.url),
        path.display(),
    )
}

fn catalog_origin_output(origin: &denoize::models::CatalogOrigin) -> String {
    match origin {
        denoize::models::CatalogOrigin::Embedded => "embedded".into(),
        denoize::models::CatalogOrigin::Signed { source } if source == "local-import" => {
            "signed:local-import".into()
        }
        denoize::models::CatalogOrigin::Signed { source } => {
            format!("signed:{}", denoize::models::redact_url(source))
        }
        _ => "unknown".into(),
    }
}

fn installation_source_output(source: &denoize::models::ModelInstallationSource) -> String {
    match source {
        denoize::models::ModelInstallationSource::CatalogUrl { url } => {
            format!("catalog-url:{}", denoize::models::redact_url(url))
        }
        denoize::models::ModelInstallationSource::AlternateUrl { url } => {
            format!("alternate-url:{}", denoize::models::redact_url(url))
        }
        denoize::models::ModelInstallationSource::LocalFile => "local-file".into(),
        denoize::models::ModelInstallationSource::CompletedPartial => "completed-partial".into(),
        denoize::models::ModelInstallationSource::ExistingCacheMigration => {
            "existing-cache-migration".into()
        }
        denoize::models::ModelInstallationSource::OfflineBundle { bundle_sha256 } => {
            format!("offline-bundle:{bundle_sha256}")
        }
        _ => "unknown".into(),
    }
}

fn catalog_model_info_output(
    model: &denoize::models::CatalogModel,
    path: &std::path::Path,
) -> String {
    let mut output = format!(
        "name: {}\nbackend: {}\nsample-rate: {}\nlicense: {}\nrevision: {}\nsize-bytes: {}\nsha256: {}\nurl: {}\npath: {}\ncatalog-sequence: {}\ncatalog-sha256: {}\ncatalog-signing-key: {}\ncatalog-issued-at-unix-seconds: {}\ncatalog-expires-at-unix-seconds: {}\ncatalog-trust-root-version: {}\ncatalog-origin: {}\n",
        model.name(),
        model.backend(),
        model.sample_rate(),
        model.license(),
        model.revision(),
        model.size_bytes(),
        model.sha256(),
        denoize::models::redact_url(model.url()),
        path.display(),
        model.catalog_sequence(),
        model.catalog_sha256(),
        model.catalog_signing_key_id(),
        model
            .catalog_issued_at_unix_seconds()
            .map_or_else(|| "legacy-none".into(), |value| value.to_string()),
        model
            .catalog_expires_at_unix_seconds()
            .map_or_else(|| "legacy-none".into(), |value| value.to_string()),
        model.catalog_trust_root_version(),
        catalog_origin_output(model.catalog_origin()),
    );
    if let Some(bundle) = model.offline_bundle() {
        output.push_str(&format!(
            "bundle-license: {}\t{}\t{}\nbundle-provenance: {}\t{}\t{}\n",
            bundle.license().filename(),
            bundle.license().size_bytes(),
            bundle.license().sha256(),
            bundle.provenance().filename(),
            bundle.provenance().size_bytes(),
            bundle.provenance().sha256(),
        ));
    }
    match denoize::models::catalog_model_provenance(model) {
        Ok(provenance) => {
            output.push_str(&format!(
                "installed: true\ninstalled-source: {}\ninstalled-at-unix-seconds: {}\ninstalled-catalog-sequence: {}\ninstalled-catalog-sha256: {}\ninstalled-catalog-signing-key: {}\n",
                installation_source_output(&provenance.installation_source),
                provenance.installed_at_unix_seconds,
                provenance.catalog_sequence,
                provenance.catalog_sha256,
                provenance.catalog_signing_key_id,
            ));
        }
        Err(_) => output.push_str("installed: false\n"),
    }
    output
}

fn print_catalog_status(status: &denoize::models::CatalogStatus) {
    println!("sequence: {}", status.sequence);
    println!("sha256: {}", status.sha256);
    println!("signing-key: {}", status.signing_key_id);
    println!("origin: {}", catalog_origin_output(&status.origin));
    println!("models: {}", status.model_count);
    println!(
        "highest-accepted-sequence: {}",
        status.highest_accepted_sequence
    );
    println!("cached-path: {}", status.cached_catalog_path.display());
    println!(
        "issued-at-unix-seconds: {}",
        status
            .issued_at_unix_seconds
            .map_or_else(|| "legacy-none".into(), |value| value.to_string())
    );
    println!(
        "expires-at-unix-seconds: {}",
        status
            .expires_at_unix_seconds
            .map_or_else(|| "legacy-none".into(), |value| value.to_string())
    );
    println!("trust-root-version: {}", status.trust_root_version);
    println!("trust-root-sha256: {}", status.trust_root_sha256);
    println!(
        "trust-root-expires-at-unix-seconds: {}",
        status.trust_root_expires_at_unix_seconds
    );
    println!(
        "trust-root-highest-observed-unix-seconds: {}",
        status
            .trust_root_highest_observed_unix_seconds
            .map_or_else(|| "unrecorded".into(), |value| value.to_string())
    );
    println!("acquisition-allowed: {}", status.acquisition_allowed);
}

fn trust_root_origin_output(origin: &denoize::models::TrustRootOrigin) -> String {
    match origin {
        denoize::models::TrustRootOrigin::Embedded => "embedded".into(),
        denoize::models::TrustRootOrigin::Signed { source } if source == "local-import" => {
            "signed:local-import".into()
        }
        denoize::models::TrustRootOrigin::Signed { source } => {
            format!("signed:{}", denoize::models::redact_url(source))
        }
        _ => "unknown".into(),
    }
}

fn print_trust_root_status(status: &denoize::models::TrustRootStatus) {
    println!("version: {}", status.version);
    println!("sha256: {}", status.sha256);
    println!("issued-at-unix-seconds: {}", status.issued_at_unix_seconds);
    println!(
        "expires-at-unix-seconds: {}",
        status.expires_at_unix_seconds
    );
    println!("expired: {}", status.expired);
    println!("signature-threshold: {}", status.signature_threshold);
    println!("root-keys: {}", status.root_key_ids.join(","));
    println!(
        "catalog-signing-keys: {}",
        status.catalog_signing_key_ids.join(",")
    );
    println!("origin: {}", trust_root_origin_output(&status.origin));
    println!(
        "highest-accepted-version: {}",
        status.highest_accepted_version
    );
    println!(
        "highest-observed-unix-seconds: {}",
        status
            .highest_observed_unix_seconds
            .map_or_else(|| "unrecorded".into(), |value| value.to_string())
    );
    println!(
        "cached-chain-path: {}",
        status.cached_trust_chain_path.display()
    );
}

fn print_offline_bundle_info(info: &denoize::models::OfflineBundleInfo) {
    println!("format-version: {}", info.format_version);
    println!("bundle-sha256: {}", info.bundle_sha256);
    println!("size-bytes: {}", info.size_bytes);
    println!("catalog-sequence: {}", info.catalog_sequence);
    println!("catalog-sha256: {}", info.catalog_sha256);
    println!("catalog-signing-key: {}", info.catalog_signing_key_id);
    println!(
        "catalog-issued-at-unix-seconds: {}",
        info.catalog_issued_at_unix_seconds
            .map_or_else(|| "unrecorded".into(), |value| value.to_string())
    );
    println!(
        "catalog-expires-at-unix-seconds: {}",
        info.catalog_expires_at_unix_seconds
            .map_or_else(|| "unrecorded".into(), |value| value.to_string())
    );
    println!("trust-root-version: {}", info.trust_root_version);
    println!("trust-root-sha256: {}", info.trust_root_sha256);
    println!("models: {}", info.models.len());
    for model in &info.models {
        println!(
            "model: {}\t{}\t{}\t{}\t{}",
            model.name,
            model.backend,
            model.artifact_filename,
            model.artifact_size_bytes,
            model.artifact_sha256
        );
        println!(
            "license: {}\t{}\t{}\t{}",
            model.name, model.license_filename, model.license_size_bytes, model.license_sha256
        );
        println!(
            "provenance: {}\t{}\t{}\t{}",
            model.name,
            model.provenance_filename,
            model.provenance_size_bytes,
            model.provenance_sha256
        );
    }
}

fn run_model_bundle(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .skip(1)
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
        || args.get(1).map(String::as_str) == Some("help")
    {
        print!("{}", models_usage());
        return Ok(());
    }
    match args.get(1).map(String::as_str).unwrap_or("inspect") {
        "inspect" => {
            if args.len() != 3 {
                return Err("models bundle inspect requires BUNDLE.dmb".into());
            }
            print_offline_bundle_info(&denoize::models::inspect_offline_bundle(&args[2])?);
        }
        "import" => {
            if args.len() != 3 {
                return Err("models bundle import requires BUNDLE.dmb".into());
            }
            let report = denoize::models::import_offline_bundle(&args[2])?;
            print_offline_bundle_info(&report.bundle);
            for path in &report.installed {
                println!("installed: {}", path.display());
            }
            for path in &report.already_present {
                println!("already-present: {}", path.display());
            }
        }
        "create" => {
            if args.len() != 7 {
                return Err(
                    "models bundle create requires OUTPUT.dmb CATALOG.json CATALOG.json.sig TRUST-ROOT.json COMPONENTS-DIR"
                        .into(),
                );
            }
            let info = denoize::models::build_offline_bundle(
                &args[2], &args[3], &args[4], &args[5], &args[6],
            )?;
            print_offline_bundle_info(&info);
            eprintln!(
                "created authenticated offline bundle {} ({})",
                args[2], info.bundle_sha256
            );
        }
        value => return Err(format!("unknown models bundle command: {value}")),
    }
    Ok(())
}

fn print_runtime_model_package_info(info: &denoize::RuntimeModelPackageInfo) {
    println!("format-version: {}", info.format_version);
    println!("package-sha256: {}", info.package_sha256);
    println!("size-bytes: {}", info.size_bytes);
    println!("package-id: {}", info.package_id);
    println!("package-revision: {}", info.package_revision);
    println!("signing-key: {}", info.signing_key_id);
    println!("sample-rate-hz: {}", info.sample_rate_hz);
    println!("tensor-layout: {}", info.tensor_layout);
    println!(
        "fixed-input-samples: {}",
        info.fixed_input_samples
            .map_or_else(|| "dynamic".into(), |value| value.to_string())
    );
    println!(
        "fixed-output-samples: {}",
        info.fixed_output_samples
            .map_or_else(|| "dynamic".into(), |value| value.to_string())
    );
    println!(
        "model: {}\t{}\t{}",
        info.model_filename, info.model_size_bytes, info.model_sha256
    );
    println!(
        "license: {}\t{}\t{}\t{}",
        info.license_spdx, info.license_filename, info.license_size_bytes, info.license_sha256
    );
    println!("accelerators: {}", info.accelerators.join(","));
    println!(
        "max-session-memory-bytes: {}",
        info.max_session_memory_bytes
    );
    println!("max-worker-memory-bytes: {}", info.max_worker_memory_bytes);
    println!(
        "max-gpu-session-memory-bytes: {}",
        info.max_gpu_session_memory_bytes
    );
    println!(
        "max-gpu-worker-memory-bytes: {}",
        info.max_gpu_worker_memory_bytes
    );
    if let Some(v2) = &info.v2 {
        println!("runtime-kind: {}", v2.runtime_kind);
        println!("runtime-mode: {}", v2.runtime_mode);
        println!("channel-policy: {}", v2.channel_policy);
        println!("component-count: {}", v2.component_count);
        println!("numerical-vector-cases: {}", v2.numerical_vector_cases);
        println!(
            "latency: frame={} hop={} left={} right={} lookahead={} algorithmic={} flush={}",
            v2.latency.frame_samples,
            v2.latency.hop_samples,
            v2.latency.left_context_samples,
            v2.latency.right_context_samples,
            v2.latency.lookahead_samples,
            v2.latency.algorithmic_latency_samples,
            v2.latency.flush_samples
        );
        for tensor in &v2.inputs {
            println!(
                "input-tensor: {}\t{}\t{}\t{}",
                tensor.name,
                tensor.role,
                tensor.element_type,
                tensor
                    .axes
                    .iter()
                    .map(|axis| format!(
                        "{}:{}:{}",
                        axis.name,
                        axis.kind,
                        axis.fixed
                            .map_or_else(|| "dynamic".into(), |value| value.to_string())
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
        for tensor in &v2.outputs {
            println!(
                "output-tensor: {}\t{}\t{}\t{}",
                tensor.name,
                tensor.role,
                tensor.element_type,
                tensor
                    .axes
                    .iter()
                    .map(|axis| format!(
                        "{}:{}:{}",
                        axis.name,
                        axis.kind,
                        axis.fixed
                            .map_or_else(|| "dynamic".into(), |value| value.to_string())
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
        for state in &v2.state_pairs {
            println!(
                "state-pair: {}\t{}\t{}\t{}",
                state.id, state.input, state.output, state.initialization
            );
        }
        for profile in &v2.precision_profiles {
            println!(
                "precision-profile: {}\t{}\t{}\t{}\t{}",
                profile.id,
                profile.element_type,
                profile.model_component,
                profile.numerical_vectors_component,
                profile.resources.accelerators.join(",")
            );
        }
        println!(
            "default-precision-profile: {}",
            v2.default_precision_profile
        );
        println!(
            "provenance: source={}@{} checkpoint={} conversion={}@{} datasets={}",
            v2.provenance.source_repository,
            v2.provenance.source_revision,
            v2.provenance.checkpoint_sha256,
            v2.provenance.conversion_tool,
            v2.provenance.conversion_revision,
            v2.provenance.training_datasets.len()
        );
    }
}

fn run_runtime_model_package(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .skip(1)
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
        || args.get(1).map(String::as_str) == Some("help")
    {
        print!("{}", models_usage());
        return Ok(());
    }
    match args.get(1).map(String::as_str).unwrap_or("inspect") {
        "inspect" => {
            if args.len() != 4 {
                return Err("models package inspect requires PACKAGE.dmp and MINISIGN.pub".into());
            }
            let info = denoize::inspect_runtime_model_package(&args[2], &args[3])?;
            print_runtime_model_package_info(&info);
        }
        "license" => {
            if args.len() != 4 {
                return Err("models package license requires PACKAGE.dmp and MINISIGN.pub".into());
            }
            let package = RuntimeModelPackage::open(&args[2], &args[3])?;
            let mut license = package.open_license_reader()?;
            let stdout = std::io::stdout();
            let mut output = stdout.lock();
            std::io::copy(&mut license, &mut output)
                .map_err(|error| format!("write runtime model license to stdout: {error}"))?;
            std::io::Write::flush(&mut output)
                .map_err(|error| format!("flush runtime model license to stdout: {error}"))?;
        }
        "create" => {
            if args.len() != 8 {
                return Err("models package create requires OUTPUT.dmp MANIFEST.json MANIFEST.json.sig MINISIGN.pub MODEL.onnx LICENSE".into());
            }
            let info = denoize::build_runtime_model_package(
                &args[2], &args[3], &args[4], &args[5], &args[6], &args[7],
            )?;
            print_runtime_model_package_info(&info);
            eprintln!(
                "created authenticated runtime model package {} ({})",
                args[2], info.package_sha256
            );
        }
        "create-v2" => {
            if args.len() != 7 {
                return Err("models package create-v2 requires OUTPUT.dmp MANIFEST.json MANIFEST.json.sig MINISIGN.pub COMPONENTS-DIR".into());
            }
            let info = denoize::build_runtime_model_package_v2(
                &args[2], &args[3], &args[4], &args[5], &args[6],
            )?;
            print_runtime_model_package_info(&info);
            eprintln!(
                "created authenticated runtime model package v2 {} ({})",
                args[2], info.package_sha256
            );
        }
        value => return Err(format!("unknown models package command: {value}")),
    }
    Ok(())
}

fn run_model_catalog_trust(args: &[String]) -> Result<(), String> {
    let command = args.get(2).map(String::as_str).unwrap_or("status");
    match command {
        "status" => {
            if args.len() != 3 {
                return Err("models catalog trust status accepts no arguments".into());
            }
            print_trust_root_status(&denoize::models::trust_root_status()?);
        }
        "import" => {
            if args.len() != 5 {
                return Err(
                    "models catalog trust import requires TRUST-ROOT.json and SIGNATURES.json"
                        .into(),
                );
            }
            let status = denoize::models::import_trust_root(&args[3], &args[4])?;
            print_trust_root_status(&status);
            eprintln!(
                "verified model trust-root version {} ({})",
                status.version, status.sha256
            );
        }
        "recover" => {
            if args.len() != 3 {
                return Err("models catalog trust recover accepts no arguments".into());
            }
            let status = denoize::models::recover_embedded_trust_root()?;
            print_trust_root_status(&status);
            eprintln!(
                "recovered embedded model trust-root version {} ({})",
                status.version, status.sha256
            );
        }
        "reset-time-floor" => {
            if args.len() != 3 {
                return Err("models catalog trust reset-time-floor accepts no arguments".into());
            }
            let status = denoize::models::reset_trust_time_floor()?;
            print_trust_root_status(&status);
            eprintln!(
                "reset model trusted-time floor under trust-root version {} ({})",
                status.version, status.sha256
            );
        }
        value => return Err(format!("unknown models catalog trust command: {value}")),
    }
    Ok(())
}

fn run_model_catalog(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .skip(1)
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
        || args.get(1).map(String::as_str) == Some("help")
    {
        print!("{}", models_usage());
        return Ok(());
    }
    let command = args.get(1).map(String::as_str).unwrap_or("status");
    match command {
        "trust" => return run_model_catalog_trust(args),
        "status" => {
            if args.len() > 2 {
                return Err("models catalog status accepts no arguments".into());
            }
            print_catalog_status(&denoize::models::catalog_status()?);
        }
        "import" => {
            if args.len() != 4 {
                return Err(
                    "models catalog import requires CATALOG.json and CATALOG.json.sig".into(),
                );
            }
            let catalog = denoize::models::import_catalog(&args[2], &args[3])?;
            print_catalog_status(&denoize::models::catalog_status()?);
            eprintln!(
                "verified model catalog sequence {} ({})",
                catalog.sequence(),
                catalog.sha256()
            );
        }
        "update" => {
            let mut options = model_catalog_download_options_from_environment_with(args, |name| {
                std::env::var(name).ok()
            })?;
            let mut synthetic = vec!["update".to_string(), "catalog".to_string()];
            synthetic.extend_from_slice(&args[2..]);
            let parsed = parse_models_command(&synthetic, options.clone(), |name| {
                std::env::var(name).map_err(|error| error.to_string())
            })?;
            let ParsedModelsCommand::Run {
                download_options,
                source_file,
                ..
            } = parsed
            else {
                return Err("invalid models catalog update arguments".into());
            };
            if source_file.is_some() {
                return Err("use models catalog import for local catalog files".into());
            }
            options = *download_options.expect("catalog update has download options");
            let catalog = denoize::models::update_catalog(&options)?;
            print_catalog_status(&denoize::models::catalog_status()?);
            eprintln!(
                "verified model catalog sequence {} ({})",
                catalog.sequence(),
                catalog.sha256()
            );
        }
        value => return Err(format!("unknown models catalog command: {value}")),
    }
    Ok(())
}

fn model_cache_status_output(status: denoize::models::ModelCacheModelStatus) -> &'static str {
    match status {
        denoize::models::ModelCacheModelStatus::Missing => "missing",
        denoize::models::ModelCacheModelStatus::Healthy => "healthy",
        denoize::models::ModelCacheModelStatus::Corrupt => "corrupt",
        denoize::models::ModelCacheModelStatus::ProvenanceMissing => "provenance-missing",
        denoize::models::ModelCacheModelStatus::ProvenanceInvalid => "provenance-invalid",
        denoize::models::ModelCacheModelStatus::Unsafe => "unsafe",
        _ => "unknown",
    }
}

fn model_cache_issue_output(kind: denoize::models::ModelCacheIssueKind) -> &'static str {
    match kind {
        denoize::models::ModelCacheIssueKind::MissingArtifact => "missing-artifact",
        denoize::models::ModelCacheIssueKind::CorruptArtifact => "corrupt-artifact",
        denoize::models::ModelCacheIssueKind::MissingProvenance => "missing-provenance",
        denoize::models::ModelCacheIssueKind::InvalidProvenance => "invalid-provenance",
        denoize::models::ModelCacheIssueKind::IncompleteDownload => "incomplete-download",
        denoize::models::ModelCacheIssueKind::StaleDownloadState => "stale-download-state",
        denoize::models::ModelCacheIssueKind::OrphanedEntry => "orphaned-entry",
        denoize::models::ModelCacheIssueKind::UnsafeEntry => "unsafe-entry",
        _ => "unknown",
    }
}

fn print_model_cache_issue(issue: &denoize::models::ModelCacheIssue) {
    println!(
        "issue: {}\t{}\t{}{}",
        model_cache_issue_output(issue.kind),
        issue.path.display(),
        issue.detail,
        if issue.prunable { "\tprunable" } else { "" }
    );
}

fn print_model_cache_report(report: &denoize::models::ModelCacheReport) {
    println!("cache: {}", report.cache_dir.display());
    println!("catalog-sequence: {}", report.catalog_sequence);
    println!("catalog-sha256: {}", report.catalog_sha256);
    println!("NAME\tSTATUS\tPATH");
    for model in &report.models {
        println!(
            "{}\t{}\t{}",
            model.name,
            model_cache_status_output(model.status),
            model.path.display()
        );
        for issue in &model.issues {
            if issue.kind != denoize::models::ModelCacheIssueKind::MissingArtifact {
                print_model_cache_issue(issue);
            }
        }
    }
    for issue in &report.issues {
        print_model_cache_issue(issue);
    }
    let healthy = report
        .models
        .iter()
        .filter(|model| model.status == denoize::models::ModelCacheModelStatus::Healthy)
        .count();
    let missing = report
        .models
        .iter()
        .filter(|model| model.status == denoize::models::ModelCacheModelStatus::Missing)
        .count();
    println!(
        "doctor-summary: {healthy} healthy, {missing} missing, {} attention, {} cache issues",
        report.models.len() - healthy - missing,
        report.issues.len()
    );
}

fn run_models(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) == Some("catalog") {
        return run_model_catalog(args);
    }
    if args.first().map(String::as_str) == Some("bundle") {
        return run_model_bundle(args);
    }
    if args.first().map(String::as_str) == Some("package") {
        return run_runtime_model_package(args);
    }
    let help_requested = args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
        || args.first().map(String::as_str) == Some("help");
    let download_command = matches!(
        args.first().map(String::as_str),
        Some("install" | "update" | "repair")
    );
    let download_options = if download_command && !help_requested {
        model_download_options_from_environment_with(args, |name| std::env::var(name).ok())?
    } else {
        denoize::models::ModelDownloadOptions::default()
    };
    let parsed = parse_models_command(args, download_options, |name| {
        std::env::var(name).map_err(|error| error.to_string())
    })?;

    let (command, target, download_options, source_file) = match parsed {
        ParsedModelsCommand::Help => {
            print!("{}", models_usage());
            return Ok(());
        }
        ParsedModelsCommand::List => {
            let catalog = denoize::models::active_catalog()?;
            println!("NAME\tBACKEND\tRATE\tLICENSE\tSTATUS");
            for model in catalog.models() {
                let status = if denoize::models::verify_catalog_model(model).is_ok() {
                    "installed"
                } else {
                    "not-installed"
                };
                println!(
                    "{}\t{}\t{}\t{}\t{}",
                    model.name(),
                    model.backend(),
                    model.sample_rate(),
                    model.license(),
                    status
                );
            }
            return Ok(());
        }
        ParsedModelsCommand::CacheDir => {
            println!("{}", denoize::models::cache_dir()?.display());
            return Ok(());
        }
        ParsedModelsCommand::Doctor => {
            let report = denoize::models::doctor_model_cache()?;
            print_model_cache_report(&report);
            if !report.is_clean() {
                return Err(
                    "model cache needs attention; run `denoize models repair all` and `denoize models prune --dry-run`"
                        .into(),
                );
            }
            return Ok(());
        }
        ParsedModelsCommand::Snapshot { pretty } => {
            let snapshot = denoize::automation::capture_automation_snapshot()?;
            let mut json = if pretty {
                snapshot.to_pretty_json()?
            } else {
                snapshot.to_json()?
            };
            json.push('\n');
            std::io::Write::write_all(&mut std::io::stdout().lock(), json.as_bytes())
                .map_err(|error| format!("write automation snapshot: {error}"))?;
            return Ok(());
        }
        ParsedModelsCommand::Prune { dry_run } => {
            let report = denoize::models::prune_model_cache(dry_run)?;
            for path in &report.would_remove {
                println!("would-remove {}", path.display());
            }
            for path in &report.removed {
                println!("removed {}", path.display());
            }
            for issue in &report.retained {
                eprintln!("retained {}: {}", issue.path.display(), issue.detail);
            }
            println!(
                "prune-summary: {} removed, {} would-remove, {} retained",
                report.removed.len(),
                report.would_remove.len(),
                report.retained.len()
            );
            return Ok(());
        }
        ParsedModelsCommand::Run {
            command,
            target,
            download_options,
            source_file,
        } => (command, target, download_options, source_file),
    };

    let catalog = denoize::models::active_catalog()?;
    let models: Vec<_> = if target == "all" {
        catalog.models().iter().collect()
    } else {
        vec![catalog
            .find(&target)
            .ok_or_else(|| format!("unknown model: {target} (run `denoize models list`)"))?]
    };
    for model in models {
        match command {
            ModelCommand::Info => {
                let path = denoize::models::path_for_catalog_model(model)?;
                print!("{}", catalog_model_info_output(model, &path));
            }
            ModelCommand::Install => {
                let installed = if let Some(source) = source_file.as_ref() {
                    denoize::models::install_catalog_model_from_file(model, source)?
                } else {
                    denoize::models::install_catalog_model_with_options(
                        model,
                        download_options
                            .as_ref()
                            .expect("download options exist for install"),
                    )?
                };
                println!("{}", installed.display());
            }
            ModelCommand::Update => println!(
                "{}",
                denoize::models::update_catalog_model_with_options(
                    model,
                    download_options
                        .as_ref()
                        .expect("download options exist for update"),
                )?
                .display()
            ),
            ModelCommand::Verify => {
                println!(
                    "verified {}",
                    denoize::models::verify_catalog_model(model)?.display()
                )
            }
            ModelCommand::Repair => {
                let outcome = denoize::models::repair_catalog_model_with_options(
                    model,
                    download_options
                        .as_ref()
                        .expect("download options exist for repair"),
                )?;
                let action = match outcome {
                    denoize::models::ModelRepairOutcome::AlreadyHealthy => "healthy",
                    denoize::models::ModelRepairOutcome::ProvenanceRebuilt => "provenance-rebuilt",
                    denoize::models::ModelRepairOutcome::ArtifactInstalled => "artifact-installed",
                    _ => "repaired",
                };
                println!("{action} {}", model.name());
            }
            ModelCommand::Remove => println!(
                "{} {}",
                if denoize::models::remove_catalog_model(model)? {
                    "removed"
                } else {
                    "not-installed"
                },
                model.name()
            ),
            ModelCommand::Path => println!(
                "{}",
                denoize::models::path_for_catalog_model(model)?.display()
            ),
        }
    }
    Ok(())
}

#[cfg(test)]
mod model_command_tests {
    use super::*;

    fn missing_secret(name: &str) -> Result<String, String> {
        Err(format!("{name} is not set"))
    }

    #[test]
    fn model_info_reports_exact_manifest_size_in_bytes() {
        let model = denoize::models::ModelInfo {
            name: "test-model",
            backend: "test-backend",
            filename: "model.onnx",
            url: "https://models.example/model.onnx",
            revision: "test-revision",
            size_bytes: 12_345_678,
            sha256: "0123456789abcdef",
            license: "MIT",
            sample_rate: 16_000,
        };

        let output = model_info_output(&model, std::path::Path::new("model.onnx"));

        assert_eq!(
            output,
            "name: test-model\nbackend: test-backend\nsample-rate: 16000\nlicense: MIT\nrevision: test-revision\nsize-bytes: 12345678\nsha256: 0123456789abcdef\nurl: https://models.example/model.onnx\npath: model.onnx\n"
        );
    }

    #[test]
    fn local_catalog_origin_has_a_stable_non_url_label() {
        assert_eq!(
            catalog_origin_output(&denoize::models::CatalogOrigin::Signed {
                source: "local-import".into(),
            }),
            "signed:local-import"
        );
    }

    #[test]
    fn explicit_model_flags_override_invalid_environment_defaults() {
        let args = vec![
            "install".into(),
            "gtcrn-dns3".into(),
            "--offline".into(),
            "--url".into(),
            "https://models.example/model.onnx".into(),
            "--no-proxy".into(),
            "--bearer-token-env".into(),
            "MODEL_TOKEN".into(),
        ];
        let options = model_download_options_from_environment_with(&args, |name| {
            Some(
                match name {
                    "DENOIZE_MODEL_OFFLINE" => "not-a-boolean",
                    "DENOIZE_MODEL_URL" => "environment-url",
                    "DENOIZE_MODEL_PROXY" => "environment-proxy",
                    "DENOIZE_MODEL_BEARER_TOKEN" => "environment-bearer",
                    "DENOIZE_MODEL_USERNAME" => "environment-user",
                    "DENOIZE_MODEL_PASSWORD" => "environment-password",
                    _ => return None,
                }
                .into(),
            )
        })
        .unwrap();
        assert!(!options.offline);
        assert!(options.source_url.is_none());
        assert!(matches!(
            options.proxy,
            denoize::models::ModelProxy::Environment
        ));
        assert!(options.authentication.is_none());
    }

    #[test]
    fn explicit_catalog_flags_override_invalid_environment_defaults() {
        let args = vec![
            "catalog".into(),
            "update".into(),
            "--offline".into(),
            "--url".into(),
            "https://catalog.example.test/catalog.json".into(),
            "--no-proxy".into(),
            "--bearer-token-env".into(),
            "CATALOG_TOKEN".into(),
        ];
        let options = model_catalog_download_options_from_environment_with(&args, |name| {
            Some(
                match name {
                    "DENOIZE_MODEL_OFFLINE" => "not-a-boolean",
                    "DENOIZE_MODEL_CATALOG_URL" => "not-a-url",
                    "DENOIZE_MODEL_PROXY" => "not-a-proxy",
                    "DENOIZE_MODEL_BEARER_TOKEN" => "environment-bearer",
                    "DENOIZE_MODEL_USERNAME" => "environment-user",
                    "DENOIZE_MODEL_PASSWORD" => "environment-password",
                    _ => return None,
                }
                .into(),
            )
        })
        .unwrap();
        assert!(!options.offline);
        assert!(options.source_url.is_none());
        assert!(matches!(
            options.proxy,
            denoize::models::ModelProxy::Environment
        ));
        assert!(options.authentication.is_none());
    }

    #[test]
    fn local_model_install_does_not_validate_unrelated_environment_defaults() {
        let args = vec![
            "install".into(),
            "gtcrn-dns3".into(),
            "--from".into(),
            "model.onnx".into(),
        ];
        let options = model_download_options_from_environment_with(&args, |_| {
            panic!("local installs must not read model download environment variables")
        })
        .unwrap();
        assert!(!options.offline);
        assert!(options.source_url.is_none());
        assert!(options.authentication.is_none());
    }

    #[test]
    fn parses_model_download_overrides_without_reading_process_environment() {
        let mut base = denoize::models::ModelDownloadOptions::default();
        base.source_url = Some("https://environment.invalid/model".into());
        base.authentication = Some(denoize::models::ModelAuthentication::Basic {
            username: "environment-user".into(),
            password: "environment-secret".into(),
        });
        let args = vec![
            "update".into(),
            "gtcrn-dns3".into(),
            "--url".into(),
            "https://models.example/model.onnx".into(),
            "--no-proxy".into(),
            "--bearer-token-env".into(),
            "MODEL_TOKEN".into(),
        ];
        let parsed = parse_models_command(&args, base, |name| {
            assert_eq!(name, "MODEL_TOKEN");
            Ok("secret-token".into())
        })
        .unwrap();

        let ParsedModelsCommand::Run {
            command,
            target,
            download_options: Some(options),
            source_file,
        } = parsed
        else {
            panic!("expected an executable model command");
        };
        assert_eq!(command, ModelCommand::Update);
        assert_eq!(target, "gtcrn-dns3");
        assert!(source_file.is_none());
        assert_eq!(
            options.source_url.as_deref(),
            Some("https://models.example/model.onnx")
        );
        assert!(matches!(
            options.proxy,
            denoize::models::ModelProxy::Disabled
        ));
        assert!(matches!(
            options.authentication,
            Some(denoize::models::ModelAuthentication::Bearer(ref token)) if token == "secret-token"
        ));
    }

    #[test]
    fn parses_basic_authentication_and_local_install() {
        let basic = vec![
            "install".into(),
            "gtcrn-dns3".into(),
            "--basic-user".into(),
            "release-bot".into(),
            "--basic-password-env".into(),
            "MODEL_PASSWORD".into(),
        ];
        let parsed = parse_models_command(
            &basic,
            denoize::models::ModelDownloadOptions::default(),
            |_| Ok("password-from-environment".into()),
        )
        .unwrap();
        let ParsedModelsCommand::Run {
            download_options: Some(options),
            ..
        } = parsed
        else {
            panic!("expected download options");
        };
        assert!(matches!(
            options.authentication,
            Some(denoize::models::ModelAuthentication::Basic {
                ref username,
                ref password,
            }) if username == "release-bot" && password == "password-from-environment"
        ));

        let local = vec![
            "install".into(),
            "gtcrn-dns3".into(),
            "--offline".into(),
            "--from".into(),
            "model.onnx".into(),
        ];
        let parsed = parse_models_command(
            &local,
            denoize::models::ModelDownloadOptions::default(),
            missing_secret,
        )
        .unwrap();
        let ParsedModelsCommand::Run {
            command,
            source_file: Some(source),
            download_options: Some(options),
            ..
        } = parsed
        else {
            panic!("expected a local install");
        };
        assert_eq!(command, ModelCommand::Install);
        assert_eq!(source, std::path::PathBuf::from("model.onnx"));
        assert!(options.offline);
    }

    #[test]
    fn rejects_conflicting_or_incomplete_model_options() {
        let cases = [
            (
                vec![
                    "install".into(),
                    "gtcrn-dns3".into(),
                    "--proxy".into(),
                    "http://proxy.example".into(),
                    "--no-proxy".into(),
                ],
                "cannot be combined",
            ),
            (
                vec![
                    "install".into(),
                    "gtcrn-dns3".into(),
                    "--basic-user".into(),
                    "release-bot".into(),
                ],
                "must be specified together",
            ),
            (
                vec![
                    "install".into(),
                    "gtcrn-dns3".into(),
                    "--bearer-token-env".into(),
                    "TOKEN".into(),
                    "--basic-user".into(),
                    "release-bot".into(),
                    "--basic-password-env".into(),
                    "PASSWORD".into(),
                ],
                "cannot be combined",
            ),
            (
                vec![
                    "install".into(),
                    "gtcrn-dns3".into(),
                    "--from".into(),
                    "model.onnx".into(),
                    "--proxy".into(),
                    "http://proxy.example".into(),
                ],
                "network download options",
            ),
        ];
        for (args, expected) in cases {
            let error = parse_models_command(
                &args,
                denoize::models::ModelDownloadOptions::default(),
                missing_secret,
            )
            .unwrap_err();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    #[test]
    fn rejects_options_outside_their_supported_target_or_command() {
        let cases = [
            (
                vec!["info".into(), "gtcrn-dns3".into(), "--offline".into()],
                "does not accept options",
            ),
            (
                vec![
                    "update".into(),
                    "gtcrn-dns3".into(),
                    "--from".into(),
                    "model.onnx".into(),
                ],
                "install",
            ),
            (
                vec![
                    "install".into(),
                    "all".into(),
                    "--from".into(),
                    "model.onnx".into(),
                ],
                "cannot be used with `all`",
            ),
            (
                vec![
                    "update".into(),
                    "all".into(),
                    "--url".into(),
                    "https://models.example/model.onnx".into(),
                ],
                "cannot be used with `all`",
            ),
            (
                vec![
                    "install".into(),
                    "gtcrn-dns3".into(),
                    "--url".into(),
                    "https://user:secret@models.example/model.onnx".into(),
                ],
                "must not contain credentials",
            ),
        ];
        for (args, expected) in cases {
            let error = parse_models_command(
                &args,
                denoize::models::ModelDownloadOptions::default(),
                missing_secret,
            )
            .unwrap_err();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    #[test]
    fn rejects_environment_source_or_authentication_for_all_models() {
        let args = vec!["update".into(), "all".into()];
        let mut source = denoize::models::ModelDownloadOptions::default();
        source.source_url = Some("https://mirror.example/model.onnx".into());
        let source_error = parse_models_command(&args, source, missing_secret).unwrap_err();
        assert!(source_error.contains("cannot be used with `all`"));

        let mut authenticated = denoize::models::ModelDownloadOptions::default();
        authenticated.authentication = Some(denoize::models::ModelAuthentication::Bearer(
            "environment-token".into(),
        ));
        let authentication_error =
            parse_models_command(&args, authenticated, missing_secret).unwrap_err();
        assert!(authentication_error.contains("requires one MODEL"));
    }

    #[test]
    fn reports_missing_secret_environment_variables_without_exposing_values() {
        let args = vec![
            "install".into(),
            "gtcrn-dns3".into(),
            "--bearer-token-env".into(),
            "MISSING_TOKEN".into(),
        ];
        let error = parse_models_command(
            &args,
            denoize::models::ModelDownloadOptions::default(),
            missing_secret,
        )
        .unwrap_err();
        assert!(error.contains("MISSING_TOKEN"));
        assert!(error.contains("not set"));
    }

    #[test]
    fn exposes_dedicated_models_help() {
        let parsed = parse_models_command(
            &["--help".into()],
            denoize::models::ModelDownloadOptions::default(),
            missing_secret,
        )
        .unwrap();
        assert!(matches!(parsed, ParsedModelsCommand::Help));
        for flag in [
            "--offline",
            "--proxy",
            "--no-proxy",
            "--url",
            "--bearer-token-env",
            "--basic-user",
            "--basic-password-env",
            "--from",
            "bundle inspect",
            "bundle import",
            "bundle create",
            "package inspect",
            "package license",
            "package create",
            "package create-v2",
            "models snapshot",
        ] {
            assert!(models_usage().contains(flag));
        }
    }

    #[test]
    fn parses_snapshot_format_without_reading_download_secrets() {
        let compact = parse_models_command(
            &["snapshot".into(), "--json".into()],
            denoize::models::ModelDownloadOptions::default(),
            |_| panic!("snapshot must not read a secret"),
        )
        .unwrap();
        assert!(matches!(
            compact,
            ParsedModelsCommand::Snapshot { pretty: false }
        ));

        let pretty = parse_models_command(
            &["snapshot".into(), "--pretty".into()],
            denoize::models::ModelDownloadOptions::default(),
            |_| panic!("snapshot must not read a secret"),
        )
        .unwrap();
        assert!(matches!(
            pretty,
            ParsedModelsCommand::Snapshot { pretty: true }
        ));

        let error = parse_models_command(
            &["snapshot".into(), "--pretty".into(), "--pretty".into()],
            denoize::models::ModelDownloadOptions::default(),
            missing_secret,
        )
        .unwrap_err();
        assert!(error.contains("option repeated"));
    }

    #[test]
    fn model_bundle_commands_reject_bad_arity_before_file_io() {
        let cases = [
            (
                vec!["bundle".into(), "inspect".into()],
                "models bundle inspect requires BUNDLE.dmb",
            ),
            (
                vec![
                    "bundle".into(),
                    "import".into(),
                    "a.dmb".into(),
                    "extra".into(),
                ],
                "models bundle import requires BUNDLE.dmb",
            ),
            (
                vec!["bundle".into(), "create".into(), "output.dmb".into()],
                "models bundle create requires OUTPUT.dmb",
            ),
        ];
        for (args, expected) in cases {
            let error = run_models(&args).unwrap_err();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn runtime_model_package_commands_reject_bad_arity_before_file_io() {
        for (args, expected) in [
            (
                vec!["package".into(), "inspect".into()],
                "models package inspect requires PACKAGE.dmp and MINISIGN.pub",
            ),
            (
                vec!["package".into(), "create".into(), "output.dmp".into()],
                "models package create requires OUTPUT.dmp",
            ),
            (
                vec!["package".into(), "create-v2".into(), "output.dmp".into()],
                "models package create-v2 requires OUTPUT.dmp",
            ),
            (
                vec!["package".into(), "license".into()],
                "models package license requires PACKAGE.dmp and MINISIGN.pub",
            ),
        ] {
            let error = run_models(&args).unwrap_err();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn parses_recommendation_options_without_input_io() {
        let (input, options, output) = parse_recommendation_args(&[
            "missing.wav".into(),
            "--goal".into(),
            "quality".into(),
            "--analysis-seconds".into(),
            "7".into(),
            "--calibration-runs".into(),
            "2".into(),
            "--accelerator".into(),
            "cpu".into(),
            "--max-memory".into(),
            "64".into(),
            "--max-gpu-memory".into(),
            "128".into(),
            "--deterministic".into(),
            "--pretty".into(),
        ])
        .unwrap();
        assert_eq!(input, "missing.wav");
        assert_eq!(options.goal(), RecommendationGoal::Quality);
        assert_eq!(options.analysis_seconds(), 7);
        assert_eq!(options.calibration_runs(), Some(2));
        assert_eq!(options.accelerator(), AcceleratorPreference::Cpu);
        assert!(options.deterministic());
        assert_eq!(
            options.decode_limits().max_working_set_bytes,
            Some(64 * BYTES_PER_MIB)
        );
        assert_eq!(options.max_gpu_memory_bytes(), Some(128 * BYTES_PER_MIB));
        assert_eq!(output, RecommendationOutput::PrettyJson);
    }

    #[test]
    fn recommendation_rejects_invalid_options_before_input_io() {
        for (args, expected) in [
            (
                vec![
                    "missing.wav".into(),
                    "--analysis-seconds".into(),
                    "0".into(),
                ],
                "analysis duration",
            ),
            (
                vec![
                    "missing.wav".into(),
                    "--calibration-runs".into(),
                    "10".into(),
                ],
                "calibration runs",
            ),
            (
                vec!["missing.wav".into(), "--max-memory".into(), "0".into()],
                "at least 1 MiB",
            ),
            (
                vec!["missing.wav".into(), "--max-gpu-memory".into(), "0".into()],
                "at least 1 MiB",
            ),
            (
                vec!["missing.wav".into(), "--json".into(), "--pretty".into()],
                "only one",
            ),
        ] {
            let error = parse_recommendation_args(&args).unwrap_err();
            assert!(error.contains(expected), "{error}");
            assert!(!error.contains("read input"), "{error}");
        }
    }

    #[test]
    fn recommendation_rejects_nonseekable_input_explicitly() {
        let error = parse_recommendation_args(&["-".into()]).unwrap_err();
        assert!(error.contains("regular-file INPUT"));
        assert!(error.contains("--stream"));
    }
}

fn main() {
    let _ipc_control = match denoize::ipc::install_process_control() {
        Ok(guard) => guard,
        Err(error) => {
            eprintln!("denoize: error: {error}");
            std::process::exit(1);
        }
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("denoize: error: {e}");
        eprintln!("run 'denoize --help' for usage.");
        std::process::exit(1);
    }
}

#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::*;

    #[test]
    fn watch_cancellation_does_not_consume_an_attempt() {
        let error = classify_watch_process_error("cancelled".into());
        assert!(error.is_retryable());
        assert!(!error.counts_attempt());
    }

    #[test]
    fn parses_onnx_model_options() {
        let args = vec![
            "input.wav".into(),
            "output.wav".into(),
            "--backend".into(),
            "onnx".into(),
            "--onnx-model".into(),
            "model.onnx".into(),
            "--onnx-rate".into(),
            "48000".into(),
        ];
        let (_, _, options) = parse_args(&args).unwrap();
        assert_eq!(options.backend, Some(Backend::Onnx));
        assert_eq!(options.onnx_model.as_deref(), Some("model.onnx"));
        assert_eq!(options.onnx_sample_rate, Some(48_000));
    }

    #[test]
    fn parses_signed_runtime_model_package_options_without_opening_paths() {
        let args = vec![
            "input.wav".into(),
            "output.wav".into(),
            "--backend".into(),
            "onnx".into(),
            "--model-package".into(),
            "missing.dmp".into(),
            "--model-package-key".into(),
            "missing.pub".into(),
        ];
        let (_, _, options) = parse_args(&args).unwrap();
        assert_eq!(options.backend, Some(Backend::Onnx));
        assert_eq!(options.model_package.as_deref(), Some("missing.dmp"));
        assert_eq!(options.model_package_key.as_deref(), Some("missing.pub"));
    }

    #[test]
    fn runtime_model_package_options_reject_partial_conflicting_or_auto_selection() {
        for (args, expected) in [
            (
                vec![
                    "--backend".into(),
                    "onnx".into(),
                    "--model-package".into(),
                    "missing.dmp".into(),
                ],
                "must be supplied together",
            ),
            (
                vec![
                    "--backend".into(),
                    "onnx".into(),
                    "--model-package".into(),
                    "missing.dmp".into(),
                    "--model-package-key".into(),
                    "missing.pub".into(),
                    "--onnx-model".into(),
                    "raw.onnx".into(),
                ],
                "cannot be combined",
            ),
            (
                vec![
                    "--backend".into(),
                    "auto".into(),
                    "--model-package".into(),
                    "missing.dmp".into(),
                    "--model-package-key".into(),
                    "missing.pub".into(),
                ],
                "requires --backend onnx or bsrnn",
            ),
        ] {
            let error = parse_args(&args).unwrap_err();
            assert!(error.contains(expected), "{error}");
            assert!(!error.contains("missing INPUT"), "{error}");
        }
    }

    #[test]
    fn selected_external_backend_requires_a_model_before_input_io() {
        let error = parse_args(&[
            "--backend".into(),
            "onnx".into(),
            "--onnx-rate".into(),
            "16000".into(),
        ])
        .unwrap_err();
        assert!(error.contains("backend_options.onnx"));
        assert!(!error.contains("missing INPUT"));
    }

    #[test]
    fn parses_live_device_options() {
        let args = vec![
            "-".into(),
            "-".into(),
            "--input-device".into(),
            "Mic".into(),
            "--output-device".into(),
            "Cable".into(),
            "--chunk-ms".into(),
            "40".into(),
            "--live-latency".into(),
            "80".into(),
            "--max-drift-ppm".into(),
            "1500".into(),
            "--reconnect-timeout".into(),
            "45000".into(),
        ];
        let (_, _, options) = parse_args(&args).unwrap();
        assert_eq!(options.input_device.as_deref(), Some("Mic"));
        assert_eq!(options.output_device.as_deref(), Some("Cable"));
        assert_eq!(options.chunk_ms, Some(40));
        assert_eq!(options.live_latency_ms, Some(80));
        assert_eq!(options.max_drift_ppm, Some(1_500));
        assert_eq!(options.reconnect_timeout_ms, Some(45_000));
    }
}
