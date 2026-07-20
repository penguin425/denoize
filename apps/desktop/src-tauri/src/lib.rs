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
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Manager, State};

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
    output_dir: String,
    output_format: String,
    options: ProcessOptions,
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
    if request.inputs.is_empty() {
        return Err("音声ファイルを1つ以上選択してください".into());
    }
    if !Path::new(&request.output_dir).is_dir() {
        return Err("出力フォルダが存在しません".into());
    }
    let extension = request
        .output_format
        .trim_start_matches('.')
        .to_ascii_lowercase();
    let probe = PathBuf::from(format!("output.{extension}"));
    OutputFormat::from_path(&probe)?;
    let mut outputs = HashSet::new();
    for input in &request.inputs {
        if !Path::new(input).is_file() {
            return Err(format!("入力ファイルが存在しません: {input}"));
        }
        let stem = Path::new(input)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("output");
        let output = Path::new(&request.output_dir).join(format!("{stem}.{extension}"));
        if !outputs.insert(output) {
            return Err(format!(
                "同じ出力名になるファイルがあります: {stem}.{extension}"
            ));
        }
    }
    let (job_id, cancelled) = register_job(&state)?;
    let jobs = Arc::clone(&state.jobs);
    std::thread::spawn(move || {
        let started = Instant::now();
        let total = request.inputs.len();
        let mut failure = None;
        for (index, input) in request.inputs.iter().enumerate() {
            if cancelled.load(Ordering::SeqCst) {
                break;
            }
            let stem = Path::new(input)
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("output");
            let output = Path::new(&request.output_dir).join(format!("{stem}.{extension}"));
            emit_progress(
                &app,
                job_id,
                "batch",
                "running",
                &format!(
                    "{} を処理しています",
                    Path::new(input)
                        .file_name()
                        .and_then(|v| v.to_str())
                        .unwrap_or(input)
                ),
                index,
                total,
                started,
                None,
                None,
            );
            let item = ProcessRequest {
                input: input.clone(),
                output: output.to_string_lossy().into_owned(),
                options: request.options.clone(),
            };
            if let Err(error) = validate_request(&item.input, &item.output, &item.options)
                .and_then(|_| process_file(&item, &cancelled, |_, _| {}))
            {
                failure = Some(format!("{}: {error}", Path::new(input).display()));
                break;
            }
        }
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
        } else if let Some(error) = failure {
            emit_progress(
                &app,
                job_id,
                "batch",
                "failed",
                "バッチ処理に失敗しました",
                0,
                total,
                started,
                None,
                Some(error),
            );
        } else {
            emit_progress(
                &app,
                job_id,
                "batch",
                "completed",
                "すべてのファイルを処理しました",
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
        },
    );
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
            save_text_file
        ])
        .setup(|app| {
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
}
