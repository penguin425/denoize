use denoize::audio::{read_audio, write_audio};
use denoize::benchmark::ComparisonReport;
use denoize::denoiser::{DenoiserConfig, Preset, ProcessingMode};
use denoize::service::{self, BackendChoice, ProcessingOptions};
use denoize::{
    AacEncoder, Backend, BackendOptions, ChannelMode, EncodeOptions, OnnxModelConfig,
    OutputFormat, SgmseProfile,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Manager, State};
use rayon::prelude::*;

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Default)]
struct AppState {
    jobs: Arc<Mutex<HashMap<u64, Arc<AtomicBool>>>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProcessOptions {
    backend: String,
    preset: Option<String>,
    mode: Option<String>,
    strength: f64,
    adaptive_noise: bool,
    vad: bool,
    channel_mode: String,
    loudness_lufs: Option<f64>,
    true_peak_dbtp: f64,
    preserve_metadata: bool,
    force: bool,
    mp3_bitrate_kbps: u32,
    aac_bitrate_kbps: u32,
    aac_encoder: String,
    onnx_model: Option<String>,
    onnx_sample_rate: u32,
    sgmse_profile: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProcessRequest {
    input: String,
    output: String,
    options: ProcessOptions,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchRequest {
    inputs: Vec<String>,
    input_dir: Option<String>,
    output_dir: String,
    output_format: String,
    recursive: bool,
    jobs: usize,
    resume: bool,
    options: ProcessOptions,
}

#[derive(Clone, Debug)]
struct BatchItem {
    input: PathBuf,
    output: PathBuf,
    state_key: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JobProgress {
    job_id: u64,
    kind: &'static str,
    status: &'static str,
    message: String,
    current: usize,
    total: usize,
    fraction: f64,
    elapsed_seconds: f64,
    output: Option<String>,
    error: Option<String>,
    eta_seconds: Option<f64>,
    item: Option<String>,
    item_status: Option<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AppInfo {
    version: &'static str,
    backends: Vec<BackendInfo>,
    formats: Vec<&'static str>,
    fdk_available: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BackendInfo {
    name: &'static str,
    external_model: bool,
    managed_model: Option<&'static str>,
    sample_rate: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ComparisonOutput {
    markdown: String,
    json: String,
    html: String,
    noisy_snr_db: f64,
    enhanced_snr_db: f64,
    improvement_db: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelRow {
    name: &'static str,
    backend: &'static str,
    license: &'static str,
    sample_rate: u32,
    revision: &'static str,
    installed: bool,
    path: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PreviewData {
    source: String,
    playable_path: String,
    duration_seconds: f64,
    rms_db: f64,
    waveform: Vec<f64>,
}

#[tauri::command]
fn app_info() -> AppInfo {
    AppInfo {
        version: env!("CARGO_PKG_VERSION"),
        backends: Backend::available_names()
            .iter()
            .filter_map(|name| Backend::parse(name))
            .map(|backend| BackendInfo {
                name: service::backend_name(backend),
                external_model: service::requires_external_model(backend),
                managed_model: (service::backend_name(backend) == "gtcrn").then_some("gtcrn"),
                sample_rate: match service::backend_name(backend) {
                    "bsrnn" | "mossformer2" | "gtcrn" => Some(48_000),
                    "onnx" | "mpsenet" | "sgmse" => Some(16_000),
                    _ => None,
                },
            })
            .collect(),
        formats: vec!["wav", "flac", "opus", "mp3", "m4a", "aac"],
        fdk_available: cfg!(feature = "fdk-aac-encoder"),
    }
}

#[tauri::command]
fn start_process(
    app: AppHandle,
    state: State<'_, AppState>,
    request: ProcessRequest,
) -> Result<u64, String> {
    validate_request(&request.input, &request.output, &request.options)?;
    let (job_id, cancelled) = register_job(&state)?;
    let jobs = Arc::clone(&state.jobs);
    std::thread::spawn(move || {
        let started = Instant::now();
        emit_progress(
            &app,
            job_id,
            "file",
            "running",
            "音声を読み込んでいます",
            0,
            4,
            started,
            None,
            None,
        );
        let result = process_file(&request, &cancelled, |stage, message| {
            emit_progress(
                &app, job_id, "file", "running", message, stage, 4, started, None, None,
            );
        });
        match result {
            Ok(output) => emit_progress(
                &app,
                job_id,
                "file",
                "completed",
                "処理が完了しました",
                4,
                4,
                started,
                Some(output),
                None,
            ),
            Err(error) if error == "cancelled" => emit_progress(
                &app,
                job_id,
                "file",
                "cancelled",
                "処理をキャンセルしました",
                0,
                4,
                started,
                None,
                None,
            ),
            Err(error) => emit_progress(
                &app,
                job_id,
                "file",
                "failed",
                "処理に失敗しました",
                0,
                4,
                started,
                None,
                Some(error),
            ),
        }
        if let Ok(mut jobs) = jobs.lock() {
            jobs.remove(&job_id);
        }
    });
    Ok(job_id)
}

#[tauri::command]
fn start_batch(
    app: AppHandle,
    state: State<'_, AppState>,
    request: BatchRequest,
) -> Result<u64, String> {
    if !Path::new(&request.output_dir).is_dir() {
        return Err("出力フォルダが存在しません".into());
    }
    if !(1..=32).contains(&request.jobs) {
        return Err("並列数は1〜32にしてください".into());
    }
    let extension = request
        .output_format
        .trim_start_matches('.')
        .to_ascii_lowercase();
    let probe = PathBuf::from(format!("output.{extension}"));
    OutputFormat::from_path(&probe)?;
    let items = collect_batch_items(&request, &extension)?;
    if items.is_empty() {
        return Err("対応する音声ファイルがありません".into());
    }
    let (job_id, cancelled) = register_job(&state)?;
    let jobs = Arc::clone(&state.jobs);
    std::thread::spawn(move || {
        let started = Instant::now();
        let total = items.len();
        let state_path = Path::new(&request.output_dir).join(".denoize-gui-state");
        let completed = if request.resume {
            read_batch_state(&state_path).unwrap_or_default()
        } else {
            HashSet::new()
        };
        let state_file = request.resume.then(|| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&state_path)
                .map(Mutex::new)
        });
        let state_file = match state_file.transpose() {
            Ok(file) => file.map(Arc::new),
            Err(error) => {
                emit_progress(&app, job_id, "batch", "failed", "再開状態を開けません", 0, total, started, None, Some(error.to_string()));
                if let Ok(mut jobs) = jobs.lock() { jobs.remove(&job_id); }
                return;
            }
        };
        let finished = AtomicUsize::new(0);
        let succeeded = AtomicUsize::new(0);
        let skipped = AtomicUsize::new(0);
        let failures = Mutex::new(Vec::<String>::new());
        let pool = rayon::ThreadPoolBuilder::new().num_threads(request.jobs).build();
        let run = || {
            items.par_iter().for_each(|batch_item| {
                if cancelled.load(Ordering::SeqCst) { return; }
                if request.resume && completed.contains(&batch_item.state_key) && batch_item.output.is_file() {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    let current = finished.fetch_add(1, Ordering::SeqCst) + 1;
                    emit_batch_item(&app, job_id, "skipped", batch_item, current, total, started, None);
                    return;
                }
                let process_request = ProcessRequest {
                    input: batch_item.input.to_string_lossy().into_owned(),
                    output: batch_item.output.to_string_lossy().into_owned(),
                    options: request.options.clone(),
                };
                let result = validate_request(&process_request.input, &process_request.output, &process_request.options)
                    .and_then(|_| process_file(&process_request, &cancelled, |_, _| {}));
                let current = finished.fetch_add(1, Ordering::SeqCst) + 1;
                match result {
                    Ok(_) => {
                        succeeded.fetch_add(1, Ordering::Relaxed);
                        if let Some(file) = &state_file {
                            if let Ok(mut file) = file.lock() { let _ = writeln!(file, "{}", batch_item.state_key); }
                        }
                        emit_batch_item(&app, job_id, "completed", batch_item, current, total, started, None);
                    }
                    Err(error) if error == "cancelled" => {}
                    Err(error) => {
                        if let Ok(mut list) = failures.lock() { list.push(format!("{}: {error}", batch_item.input.display())); }
                        emit_batch_item(&app, job_id, "failed", batch_item, current, total, started, Some(error));
                    }
                }
            });
        };
        match pool { Ok(pool) => pool.install(run), Err(error) => failures.lock().unwrap().push(error.to_string()) }
        if cancelled.load(Ordering::SeqCst) {
            emit_progress(
                &app,
                job_id,
                "batch",
                "cancelled",
                "バッチをキャンセルしました",
                0,
                total,
                started,
                None,
                None,
            );
        } else {
            let failure_count = failures.lock().map(|list| list.len()).unwrap_or(0);
            let success_count = succeeded.load(Ordering::Relaxed);
            let skipped_count = skipped.load(Ordering::Relaxed);
            emit_progress(
                &app,
                job_id,
                "batch",
                "completed",
                &format!("完了 {success_count} · スキップ {skipped_count} · 失敗 {failure_count}"),
                total,
                total,
                started,
                Some(request.output_dir),
                None,
            );
        }
        if let Ok(mut jobs) = jobs.lock() {
            jobs.remove(&job_id);
        }
    });
    Ok(job_id)
}

fn collect_batch_items(request: &BatchRequest, extension: &str) -> Result<Vec<BatchItem>, String> {
    let output_root = Path::new(&request.output_dir);
    let mut sources = request.inputs.iter().map(PathBuf::from).collect::<Vec<_>>();
    let input_root = request.input_dir.as_deref().map(Path::new);
    if let Some(root) = input_root {
        if !root.is_dir() {
            return Err("入力フォルダが存在しません".into());
        }
        if root == output_root {
            return Err("入力フォルダと出力フォルダは分けてください".into());
        }
        collect_audio_files(root, request.recursive, &mut sources)?;
        if output_root.starts_with(root) {
            sources.retain(|path| !path.starts_with(output_root));
        }
    }
    sources.sort();
    sources.dedup();
    let mut destinations = HashSet::new();
    sources.into_iter().map(|input| {
        if !input.is_file() { return Err(format!("入力ファイルが存在しません: {}", input.display())); }
        let relative = input_root.and_then(|root| input.strip_prefix(root).ok())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(input.file_name().unwrap_or_default()));
        let mut output = output_root.join(&relative);
        output.set_extension(extension);
        if !destinations.insert(output.clone()) { return Err(format!("同じ出力先になるファイルがあります: {}", output.display())); }
        Ok(BatchItem { input, output, state_key: relative.to_string_lossy().replace('\\', "/") })
    }).collect()
}

fn collect_audio_files(dir: &Path, recursive: bool, files: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in std::fs::read_dir(dir)
        .map_err(|error| format!("入力フォルダを読めません: {error}"))?
    {
        let entry = entry.map_err(|error| error.to_string())?;
        let file_type = entry.file_type().map_err(|error| error.to_string())?;
        let path = entry.path();
        if file_type.is_dir() && recursive {
            collect_audio_files(&path, true, files)?;
        } else if file_type.is_file() && is_audio_path(&path) {
            files.push(path);
        }
    }
    Ok(())
}

fn is_audio_path(path: &Path) -> bool {
    path.extension().and_then(|value| value.to_str()).is_some_and(|value| {
        matches!(value.to_ascii_lowercase().as_str(), "wav" | "flac" | "opus" | "ogg" | "mp3" | "m4a" | "aac")
    })
}

fn read_batch_state(path: &Path) -> Result<HashSet<String>, String> {
    match std::fs::read_to_string(path) {
        Ok(source) => Ok(source.lines().map(str::to_owned).collect()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(HashSet::new()),
        Err(error) => Err(format!("再開状態を読めません: {error}")),
    }
}

#[tauri::command]
fn cancel_job(state: State<'_, AppState>, job_id: u64) -> Result<(), String> {
    let jobs = state
        .jobs
        .lock()
        .map_err(|_| "ジョブ状態を取得できません")?;
    let flag = jobs.get(&job_id).ok_or("実行中のジョブが見つかりません")?;
    flag.store(true, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
async fn compare_audio(
    clean: String,
    noisy: String,
    enhanced: String,
) -> Result<ComparisonOutput, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let report = ComparisonReport::compare(
            &read_audio(clean)?,
            &read_audio(noisy)?,
            &read_audio(enhanced)?,
        )?;
        Ok(ComparisonOutput {
            markdown: report.markdown(),
            json: report.json(),
            html: report.html(),
            noisy_snr_db: report.noisy.snr_db,
            enhanced_snr_db: report.enhanced.snr_db,
            improvement_db: report.enhanced.snr_db - report.noisy.snr_db,
        })
    })
    .await
    .map_err(|error| format!("比較タスクに失敗しました: {error}"))?
}

#[tauri::command]
fn list_models() -> Result<Vec<ModelRow>, String> {
    denoize::models::MODELS
        .iter()
        .map(|model| {
            let path = denoize::models::path(model)?;
            Ok(ModelRow {
                name: model.name,
                backend: model.backend,
                license: model.license,
                sample_rate: model.sample_rate,
                revision: model.revision,
                installed: denoize::models::verify(model).is_ok(),
                path: path.to_string_lossy().into_owned(),
            })
        })
        .collect()
}

#[tauri::command]
async fn model_action(name: String, action: String) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let model = denoize::models::find(&name).ok_or_else(|| format!("不明なモデル: {name}"))?;
        match action.as_str() {
            "install" => Ok(denoize::models::install(model)?.display().to_string()),
            "update" => Ok(denoize::models::update(model)?.display().to_string()),
            "verify" => Ok(denoize::models::verify(model)?.display().to_string()),
            "remove" => {
                denoize::models::remove(model)?;
                Ok("削除しました".into())
            }
            _ => Err(format!("不明な操作: {action}")),
        }
    })
    .await
    .map_err(|error| format!("モデル操作に失敗しました: {error}"))?
}

#[tauri::command]
async fn prepare_preview(path: String, points: Option<usize>) -> Result<PreviewData, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let source = Path::new(&path);
        if !source.is_file() {
            return Err("プレビューする音声ファイルが存在しません".into());
        }
        let audio = read_audio(source)?;
        let frames = audio.frames();
        let point_count = points.unwrap_or(180).clamp(32, 512);
        let mut waveform = vec![0.0f64; point_count];
        let mut sum_squares = 0.0;
        let mut sample_count = 0usize;
        for channel in &audio.channels {
            for (index, sample) in channel.iter().enumerate() {
                let bucket = index.saturating_mul(point_count) / frames.max(1);
                if let Some(peak) = waveform.get_mut(bucket.min(point_count - 1)) {
                    *peak = peak.max(sample.abs());
                }
                sum_squares += sample * sample;
                sample_count += 1;
            }
        }
        let peak = waveform.iter().copied().fold(0.0f64, f64::max).max(1e-9);
        for value in &mut waveform { *value /= peak; }
        let rms = (sum_squares / sample_count.max(1) as f64).sqrt();
        let mut hasher = DefaultHasher::new();
        path.hash(&mut hasher);
        source.metadata().and_then(|metadata| metadata.modified()).ok().hash(&mut hasher);
        let preview_dir = std::env::temp_dir().join("denoize-previews");
        std::fs::create_dir_all(&preview_dir).map_err(|error| format!("プレビューフォルダを作成できません: {error}"))?;
        let playable = preview_dir.join(format!("{:016x}.wav", hasher.finish()));
        if !playable.is_file() {
            write_audio(&playable, &audio, EncodeOptions::default())?;
        }
        Ok(PreviewData {
            source: path,
            playable_path: playable.to_string_lossy().into_owned(),
            duration_seconds: frames as f64 / audio.sample_rate.max(1) as f64,
            rms_db: 20.0 * rms.max(1e-10).log10(),
            waveform,
        })
    }).await.map_err(|error| format!("プレビュー処理に失敗しました: {error}"))?
}

#[tauri::command]
fn save_text_file(path: String, contents: String) -> Result<(), String> {
    std::fs::write(&path, contents).map_err(|error| format!("{path} を保存できません: {error}"))
}

fn register_job(state: &State<'_, AppState>) -> Result<(u64, Arc<AtomicBool>), String> {
    let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    let cancelled = Arc::new(AtomicBool::new(false));
    let mut jobs = state
        .jobs
        .lock()
        .map_err(|_| "ジョブ状態を更新できません")?;
    if !jobs.is_empty() {
        return Err("別の処理が実行中です。完了またはキャンセル後に再試行してください".into());
    }
    jobs.insert(job_id, Arc::clone(&cancelled));
    Ok((job_id, cancelled))
}

fn validate_request(input: &str, output: &str, options: &ProcessOptions) -> Result<(), String> {
    if !Path::new(input).is_file() {
        return Err("入力ファイルが存在しません".into());
    }
    OutputFormat::from_path(Path::new(output))?;
    if Path::new(output).exists() && !options.force {
        return Err("出力ファイルが既に存在します。「上書きを許可」を有効にしてください".into());
    }
    if !(0.0..=1.0).contains(&options.strength) {
        return Err("強度は0〜1で指定してください".into());
    }
    if options.mp3_bitrate_kbps < 32 || options.aac_bitrate_kbps < 32 {
        return Err("ビットレートは32kbps以上にしてください".into());
    }
    let backend = if options.backend == "auto" {
        None
    } else {
        Some(Backend::parse(&options.backend).ok_or_else(|| {
            format!(
                "このビルドでは利用できないバックエンドです: {}",
                options.backend
            )
        })?)
    };
    if backend.is_some_and(service::requires_external_model) {
        let model = options.onnx_model.as_deref().unwrap_or_default();
        if !Path::new(model).is_file() {
            return Err("選択したバックエンドのONNXモデルファイルを指定してください".into());
        }
    }
    if options.onnx_sample_rate == 0 {
        return Err("モデルのサンプルレートは1Hz以上にしてください".into());
    }
    Ok(())
}

fn process_file(
    request: &ProcessRequest,
    cancelled: &AtomicBool,
    progress: impl Fn(usize, &'static str),
) -> Result<String, String> {
    check_cancelled(cancelled)?;
    let input = Path::new(&request.input);
    let output = Path::new(&request.output);
    let metadata = if request.options.preserve_metadata {
        denoize::metadata::read(input)?
    } else {
        None
    };
    let mut audio = read_audio(input)?;
    progress(1, "ノイズ除去を実行しています");
    check_cancelled(cancelled)?;
    let config = processing_config(&request.options, audio.sample_rate)?;
    let backend = if request.options.backend == "auto" {
        BackendChoice::Auto
    } else {
        BackendChoice::Explicit(Backend::parse(&request.options.backend).ok_or_else(|| {
            format!(
                "このビルドでは利用できないバックエンドです: {}",
                request.options.backend
            )
        })?)
    };
    let backend_options = BackendOptions {
        onnx: request.options.onnx_model.as_ref().map(|path| OnnxModelConfig {
            path: path.into(),
            sample_rate: request.options.onnx_sample_rate,
        }),
        channel_mode: ChannelMode::parse(&request.options.channel_mode)
            .ok_or_else(|| format!("不明なチャンネルモード: {}", request.options.channel_mode))?,
        sgmse_profile: SgmseProfile::parse(&request.options.sgmse_profile).ok_or_else(|| {
            format!(
                "不明なSGMSEプロファイル: {}",
                request.options.sgmse_profile
            )
        })?,
    };
    progress(2, "ラウドネスと出力を準備しています");
    service::process_audio(
        &mut audio,
        ProcessingOptions {
            backend,
            quality: None,
            denoiser: config,
            backend_options,
            loudness_lufs: request.options.loudness_lufs,
            true_peak_dbtp: request.options.true_peak_dbtp,
        },
    )?;
    check_cancelled(cancelled)?;
    let encode = EncodeOptions {
        mp3_bitrate_kbps: request.options.mp3_bitrate_kbps,
        m4a_bitrate_bps: request.options.aac_bitrate_kbps.saturating_mul(1000),
        aac_encoder: match request.options.aac_encoder.as_str() {
            "oxide" => AacEncoder::Oxide,
            "fdk" => AacEncoder::Fdk,
            other => return Err(format!("不明なAACエンコーダー: {other}")),
        },
    };
    progress(3, "ファイルを書き出しています");
    let filename = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("無効な出力ファイル名です")?;
    let temporary = output
        .with_file_name(format!(".denoize-gui-{filename}.part"))
        .with_extension(
            output
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("wav"),
        );
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("出力フォルダを作成できません: {error}"))?;
    }
    if let Err(error) = write_audio(&temporary, &audio, encode) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    if cancelled.load(Ordering::SeqCst) {
        let _ = std::fs::remove_file(&temporary);
        return Err("cancelled".into());
    }
    if output.exists() {
        std::fs::remove_file(output)
            .map_err(|error| format!("既存の出力を置換できません: {error}"))?;
    }
    std::fs::rename(&temporary, output)
        .map_err(|error| format!("出力を確定できません: {error}"))?;
    if let Some(metadata) = metadata {
        denoize::metadata::write(metadata, output)?;
    }
    Ok(output.to_string_lossy().into_owned())
}

fn processing_config(options: &ProcessOptions, sample_rate: u32) -> Result<DenoiserConfig, String> {
    let mut config = match options.preset.as_deref() {
        Some("") | None => DenoiserConfig::default(sample_rate),
        Some(value) => Preset::parse(value)
            .ok_or_else(|| format!("不明なプリセット: {value}"))?
            .config(sample_rate),
    };
    if let Some(mode) = options.mode.as_deref().filter(|value| !value.is_empty()) {
        ProcessingMode::parse(mode)
            .ok_or_else(|| format!("不明な処理モード: {mode}"))?
            .apply(&mut config);
    }
    config.strength = options.strength;
    config.adaptive_noise = options.adaptive_noise;
    config.vad = options.vad;
    Ok(config)
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<(), String> {
    if cancelled.load(Ordering::SeqCst) {
        Err("cancelled".into())
    } else {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_progress(
    app: &AppHandle,
    job_id: u64,
    kind: &'static str,
    status: &'static str,
    message: &str,
    current: usize,
    total: usize,
    started: Instant,
    output: Option<String>,
    error: Option<String>,
) {
    let _ = app.emit(
        "job-progress",
        JobProgress {
            job_id,
            kind,
            status,
            message: message.into(),
            current,
            total,
            fraction: current as f64 / total.max(1) as f64,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            output,
            error,
            eta_seconds: None,
            item: None,
            item_status: None,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_batch_item(
    app: &AppHandle,
    job_id: u64,
    item_status: &'static str,
    item: &BatchItem,
    current: usize,
    total: usize,
    started: Instant,
    error: Option<String>,
) {
    let elapsed = started.elapsed().as_secs_f64();
    let eta = (current > 0).then(|| elapsed / current as f64 * total.saturating_sub(current) as f64);
    let name = item.input.file_name().and_then(|value| value.to_str()).unwrap_or("audio");
    let _ = app.emit("job-progress", JobProgress {
        job_id,
        kind: "batch",
        status: "running",
        message: format!("{name}: {item_status}"),
        current,
        total,
        fraction: current as f64 / total.max(1) as f64,
        elapsed_seconds: elapsed,
        output: Some(item.output.to_string_lossy().into_owned()),
        error,
        eta_seconds: eta,
        item: Some(item.input.to_string_lossy().into_owned()),
        item_status: Some(item_status),
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            app_info,
            start_process,
            start_batch,
            cancel_job,
            compare_audio,
            list_models,
            model_action,
            prepare_preview,
            save_text_file
        ])
        .setup(|app| {
            let preview_dir = std::env::temp_dir().join("denoize-previews");
            let _ = std::fs::remove_dir_all(&preview_dir);
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_title(&format!("denoize {}", env!("CARGO_PKG_VERSION")));
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("failed to run denoize desktop");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> ProcessOptions {
        ProcessOptions {
            backend: "auto".into(),
            preset: Some("hifi".into()),
            mode: Some("music".into()),
            strength: 0.4,
            adaptive_noise: false,
            vad: false,
            channel_mode: "linked".into(),
            loudness_lufs: None,
            true_peak_dbtp: -1.0,
            preserve_metadata: true,
            force: false,
            mp3_bitrate_kbps: 192,
            aac_bitrate_kbps: 192,
            aac_encoder: "oxide".into(),
            onnx_model: None,
            onnx_sample_rate: 16_000,
            sgmse_profile: "balanced".into(),
        }
    }

    #[test]
    fn gui_options_build_a_valid_processing_configuration() {
        let config = processing_config(&options(), 48_000).unwrap();
        assert_eq!(config.strength, 0.4);
        assert!(config.transient_protect);
        let selected = service::select_backend(BackendChoice::Auto, 30.0, None);
        assert_eq!(Backend::parse(service::backend_name(selected)), Some(selected));
    }

    #[test]
    fn invalid_backend_is_rejected() {
        assert!(Backend::parse("missing").is_none());
    }

    #[test]
    fn batch_folder_preserves_relative_paths() {
        let root = std::env::temp_dir().join(format!(
            "denoize-gui-batch-{}-{}",
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let input = root.join("input");
        let nested = input.join("nested");
        let output = root.join("output");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::create_dir_all(&output).unwrap();
        std::fs::write(input.join("one.wav"), []).unwrap();
        std::fs::write(nested.join("two.flac"), []).unwrap();
        std::fs::write(nested.join("ignored.txt"), []).unwrap();
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_string_lossy().into_owned()),
            output_dir: output.to_string_lossy().into_owned(),
            output_format: "opus".into(),
            recursive: true,
            jobs: 2,
            resume: true,
            options: options(),
        };
        let items = collect_batch_items(&request, "opus").unwrap();
        assert_eq!(items.len(), 2);
        assert!(items.iter().any(|item| item.output == output.join("one.opus")));
        assert!(items
            .iter()
            .any(|item| item.output == output.join("nested/two.opus")));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_state_missing_file_is_empty() {
        let path = std::env::temp_dir().join(format!(
            "denoize-missing-state-{}-{}",
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(read_batch_state(&path).unwrap().is_empty());
    }
}
