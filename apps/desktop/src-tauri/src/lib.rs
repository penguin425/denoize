use denoize::audio::{
    estimate_audio_working_set_bytes, estimate_stream_memory_bytes_checked, read_audio,
    read_audio_from_session_with_limits,
};
use denoize::batch_resume::{
    self, BatchSession, Digest, MetadataPolicy, ResumeDecision, ResumeExpectation,
};
use denoize::benchmark::{BenchmarkReport, ComparisonReport};
use denoize::denoiser::{DenoiserConfig, Preset, ProcessingMode};
use denoize::encode::write_audio_to_file;
use denoize::ipc::{IpcClient, IpcOperation, IpcResponseResult};
use denoize::metadata::MetadataLimits;
use denoize::models::{ModelAuthentication, ModelDownloadOptions, ModelProxy};
use denoize::service::{self, BackendChoice, ProcessingOptions};
use denoize::{
    neural_daw_chunk_frames, neural_daw_latency_frames, neural_daw_latency_millis, AacEncoder,
    AcceleratorPreference, AtomicOutput, AudioStreamInfo, AudioStreamReader, AudioStreamWriter,
    Backend, BackendOptions, BackendSession, ChannelMode, CommitMode, DawPortConfiguration,
    DawPreset, DawRealtimeProcessor, DawSessionState, DecodeLimits, DownmixMode, EncodeOptions,
    ExecutionKind, ExecutionPlan, ExecutionPlanItem, ExecutionReceiptPayload, OnnxModelConfig,
    OutputFormat, PlannedArtifact, PlannedOutput, PlannedResources, ReceiptItem, ReceiptPublicKey,
    ReceiptSecretKey, ReceiptTrustPolicy, ReceiptVerificationReport, ResourceGovernor,
    ResourceLimits, ResourcePermit, ResourceRequest, SgmseProfile, SignedExecutionReceipt,
    StreamEncodeLimits, StreamEncodeSpec, StreamPcmSpool, StreamingBackendSession,
    WatchCycleReport, WatchFolder, WatchFolderConfig, WatchFolderJob, WatchProcessError,
    DAW_LATENCY_POLICY, DAW_PLUGIN_ID, NEURAL_DAW_BLOCK_POOL_SIZE, NEURAL_DAW_LATENCY_POLICY,
    NEURAL_DAW_MODEL_ID, NEURAL_DAW_MODEL_SHA256, NEURAL_DAW_PLUGIN_ID,
    NEURAL_DAW_QUEUE_BLOCKS,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::utils::config::BundleType;
use tauri::{AppHandle, Emitter, Manager, State};

mod diagnostics;
mod job_worker;
mod preview;
mod recovery;

pub use job_worker::{job_worker_request_from_args, run_job_worker};
pub use preview::{preview_worker_request_from_args, run_preview_worker};

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);
static ACCESSIBILITY_E2E_ACTIVE: AtomicBool = AtomicBool::new(false);
static EVALUATION_RUNNING: AtomicBool = AtomicBool::new(false);
static PROJECT_OPERATION_RUNNING: AtomicBool = AtomicBool::new(false);
const ACCESSIBILITY_E2E_ARGUMENT: &str = "--denoize-desktop-a11y-e2e";
const MAX_DESKTOP_ERROR_DETAIL_BYTES: usize = 4 * 1024;
const MAX_DESKTOP_ERROR_PARAMETERS: usize = 16;
const MAX_DESKTOP_ERROR_PARAMETER_KEY_BYTES: usize = 64;
const MAX_DESKTOP_ERROR_PARAMETER_VALUE_BYTES: usize = 256;
const MAX_ACCESSIBILITY_E2E_FAILURE_BYTES: usize = 2 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DesktopError {
    code: String,
    parameters: BTreeMap<String, String>,
    technical_detail: String,
}

type DesktopResult<T> = Result<T, DesktopError>;

fn desktop_ipc_operation_allowed(operation: &IpcOperation) -> bool {
    !matches!(
        operation,
        IpcOperation::CreateGrant { .. }
            | IpcOperation::RevokeGrant { .. }
            | IpcOperation::ListGrants { .. }
            | IpcOperation::Shutdown { .. }
    )
}

impl DesktopError {
    fn new(code: impl Into<String>, technical_detail: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            parameters: BTreeMap::new(),
            technical_detail: bounded_desktop_error_detail(technical_detail.into()),
        }
    }

    fn classify(technical_detail: String) -> Self {
        let technical_detail = bounded_desktop_error_detail(technical_detail);
        let lower = technical_detail.to_ascii_lowercase();
        let code = if lower == "cancelled" || lower.contains("キャンセル") {
            "job.cancelled"
        } else if (lower.contains("job") || lower.contains("ジョブ"))
            && (lower.contains("not found") || lower.contains("見つかりません"))
        {
            "job.not-found"
        } else if lower.contains("別の処理")
            || lower.contains("already running")
            || lower.contains("実行中です")
        {
            "job.busy"
        } else if lower.contains("gpu") || lower.contains("accelerator") {
            "resource.accelerator"
        } else if lower.contains("memory")
            || lower.contains("メモリ")
            || lower.contains("working set")
        {
            "resource.memory"
        } else if lower.contains("temporary") || lower.contains("一時領域") {
            "resource.temporary"
        } else if lower.contains("worker") || lower.contains("隔離") {
            "worker.failed"
        } else if lower.contains("evaluation")
            || lower.contains("corpus")
            || lower.contains("評価証跡")
        {
            "evaluation.failed"
        } else if lower.contains("receipt") || lower.contains("実行証明") {
            "receipt.failed"
        } else if lower.contains("ipc") || lower.contains("capability") {
            "ipc.failed"
        } else if lower.contains("model") || lower.contains("モデル") {
            "model.failed"
        } else if lower.contains("update") || lower.contains("アプリ更新") {
            "update.failed"
        } else if lower.contains("復旧") || lower.contains("recovery") {
            "recovery.failed"
        } else if lower.contains("regular file") || lower.contains("regular-file") {
            "input.not-regular"
        } else if lower.contains("no such file")
            || lower.contains("not found")
            || lower.contains("見つかりません")
            || lower.contains("存在しません")
        {
            "input.not-found"
        } else if lower.contains("invalid")
            || lower.contains("unsupported")
            || lower.contains("不正")
            || lower.contains("指定してください")
            || lower.contains("対応していません")
        {
            "validation.invalid"
        } else if lower.contains("read")
            || lower.contains("write")
            || lower.contains("open")
            || lower.contains("保存")
            || lower.contains("読み")
            || lower.contains("書き")
        {
            "io.failed"
        } else {
            "operation.failed"
        };
        Self::new(code, technical_detail)
    }

    fn is_valid(&self) -> bool {
        matches!(
            self.code.as_str(),
            "job.cancelled"
                | "job.busy"
                | "job.not-found"
                | "input.not-regular"
                | "input.not-found"
                | "resource.memory"
                | "resource.temporary"
                | "resource.accelerator"
                | "worker.failed"
                | "evaluation.failed"
                | "receipt.failed"
                | "ipc.failed"
                | "model.failed"
                | "update.failed"
                | "recovery.failed"
                | "validation.invalid"
                | "io.failed"
                | "operation.failed"
        ) && !self.technical_detail.is_empty()
            && self.technical_detail.len() <= MAX_DESKTOP_ERROR_DETAIL_BYTES
            && self.parameters.len() <= MAX_DESKTOP_ERROR_PARAMETERS
            && self.parameters.iter().all(|(key, value)| {
                !key.is_empty()
                    && key.len() <= MAX_DESKTOP_ERROR_PARAMETER_KEY_BYTES
                    && key.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
                    })
                    && value.len() <= MAX_DESKTOP_ERROR_PARAMETER_VALUE_BYTES
            })
    }
}

fn bounded_desktop_error_detail(mut detail: String) -> String {
    if detail.len() <= MAX_DESKTOP_ERROR_DETAIL_BYTES {
        return detail;
    }
    let mut end = MAX_DESKTOP_ERROR_DETAIL_BYTES.saturating_sub('…'.len_utf8());
    while !detail.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    detail.truncate(end);
    detail.push('…');
    detail
}

#[tauri::command]
async fn ipc_request(
    discovery: String,
    grant: String,
    operation: IpcOperation,
) -> DesktopResult<IpcResponseResult> {
    if !desktop_ipc_operation_allowed(&operation) {
        return Err(DesktopError::new(
            "ipc.failed",
            "privileged IPC capability-management and shutdown operations are not exposed to the WebView",
        ));
    }
    tauri::async_runtime::spawn_blocking(move || {
        let client = IpcClient::from_files(discovery, grant)?;
        client.request(operation)
    })
    .await
    .map_err(|error| format!("IPC request task failed: {error}"))?
    .map_err(DesktopError::from)
}

impl From<String> for DesktopError {
    fn from(value: String) -> Self {
        Self::classify(value)
    }
}

impl From<&str> for DesktopError {
    fn from(value: &str) -> Self {
        Self::classify(value.into())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AccessibilityE2eReport {
    schema: String,
    schema_version: u32,
    assertions: Vec<String>,
    failures: Vec<String>,
}

fn validate_accessibility_e2e_report(report: &AccessibilityE2eReport) -> Result<(), String> {
    if report.schema != "denoize-desktop-a11y-e2e-v1" || report.schema_version != 1 {
        return Err("invalid desktop accessibility E2E report schema".into());
    }
    if report.assertions.is_empty() || report.assertions.len() > 64 || report.failures.len() > 32 {
        return Err("desktop accessibility E2E report has invalid bounds".into());
    }
    let mut unique = HashSet::new();
    for assertion in &report.assertions {
        if assertion.is_empty()
            || assertion.len() > 128
            || !assertion
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
            || !unique.insert(assertion)
        {
            return Err("desktop accessibility E2E assertion is invalid".into());
        }
    }
    if report
        .failures
        .iter()
        .any(|failure| failure.is_empty() || failure.len() > MAX_ACCESSIBILITY_E2E_FAILURE_BYTES)
    {
        return Err("desktop accessibility E2E failure is invalid".into());
    }
    Ok(())
}

#[tauri::command]
fn accessibility_e2e_active() -> bool {
    ACCESSIBILITY_E2E_ACTIVE.load(Ordering::SeqCst)
}

#[tauri::command]
fn finish_accessibility_e2e(app: AppHandle, report: AccessibilityE2eReport) -> DesktopResult<()> {
    if !ACCESSIBILITY_E2E_ACTIVE.load(Ordering::SeqCst) {
        return Err(DesktopError::new(
            "validation.invalid",
            "desktop accessibility E2E mode is not active",
        ));
    }
    validate_accessibility_e2e_report(&report)?;
    let status = if report.failures.is_empty() {
        "PASS"
    } else {
        "FAIL"
    };
    let payload = serde_json::to_string(&report)
        .map_err(|error| format!("serialize desktop accessibility E2E report: {error}"))?;
    println!("DENOIZE_DESKTOP_A11Y_E2E:{status}:{payload}");
    let _ = std::io::stdout().flush();
    ACCESSIBILITY_E2E_ACTIVE.store(false, Ordering::SeqCst);
    let exit_code = if report.failures.is_empty() { 0 } else { 1 };
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(50));
        app.exit(exit_code);
    });
    Ok(())
}

pub fn accessibility_e2e_requested_from_args() -> Result<bool, String> {
    let mut arguments = std::env::args_os().skip(1);
    if arguments.next().as_deref() != Some(std::ffi::OsStr::new(ACCESSIBILITY_E2E_ARGUMENT)) {
        return Ok(false);
    }
    if arguments.next().is_some() {
        return Err("desktop accessibility E2E mode accepts no additional arguments".into());
    }
    Ok(true)
}

pub fn run_accessibility_e2e() {
    ACCESSIBILITY_E2E_ACTIVE.store(true, Ordering::SeqCst);
    run();
}

#[cfg(test)]
thread_local! {
    static TEST_STOP_AFTER_DESKTOP_STREAM_COMMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn injected_stop_after_desktop_stream_commit() -> bool {
    #[cfg(test)]
    {
        return TEST_STOP_AFTER_DESKTOP_STREAM_COMMIT.with(|value| value.replace(false));
    }
    #[cfg(not(test))]
    false
}

const VALIDATION_SAMPLE_RATE_HZ: u32 = 48_000;
const MAX_MODEL_SAMPLE_RATE_HZ: u32 = 768_000;
const MIN_LOUDNESS_LUFS: f64 = -70.0;
const MAX_LOUDNESS_LUFS: f64 = 0.0;
const MIN_TRUE_PEAK_DBTP: f64 = -20.0;
const MAX_TRUE_PEAK_DBTP: f64 = 0.0;
const DEFAULT_MODEL_SAMPLE_RATE_HZ: u32 = 16_000;
const BYTES_PER_MIB: u64 = 1024 * 1024;
const DEFAULT_STREAM_BLOCK_FRAMES: usize = 8_192;
const STREAM_CHECKPOINT_FRAMES: u64 = 1_048_576;

const fn default_stream_block_frames() -> usize {
    DEFAULT_STREAM_BLOCK_FRAMES
}

fn default_accelerator() -> String {
    "cpu".into()
}

const fn default_max_gpu_jobs() -> usize {
    1
}

#[derive(Clone)]
struct DesktopWatchProcessorTemplate {
    output_format: String,
    recursive: bool,
    options: ProcessOptions,
}

struct DesktopWatchSession {
    watch: WatchFolder,
    processor_template: DesktopWatchProcessorTemplate,
    processor_identity: Digest,
    key_path: PathBuf,
    key_fingerprint: batch_resume::FileFingerprint,
    public_key: ReceiptPublicKey,
}

#[derive(Clone, Default)]
struct AppState {
    jobs: Arc<Mutex<HashMap<u64, Arc<JobControl>>>>,
    live: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    watch: Arc<Mutex<Option<DesktopWatchSession>>>,
    watch_active: Arc<AtomicBool>,
    diagnostics: Arc<diagnostics::DiagnosticLog>,
    startup_update_health: Arc<Mutex<Option<denoize::update::UpdateHealthReport>>>,
}

struct IsolatedChild {
    child: std::process::Child,
    #[cfg(windows)]
    _job: std::os::windows::io::OwnedHandle,
}

impl IsolatedChild {
    #[cfg(not(windows))]
    fn new(child: std::process::Child, _memory_limit: Option<u64>) -> Result<Self, String> {
        Ok(Self { child })
    }

    #[cfg(windows)]
    fn new(mut child: std::process::Child, memory_limit: Option<u64>) -> Result<Self, String> {
        use std::os::windows::io::{AsRawHandle as _, FromRawHandle as _};
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
        };

        let process_memory_limit = match memory_limit.map(usize::try_from).transpose() {
            Ok(limit) => limit,
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("隔離workerのmemory上限がこのplatformの範囲を超えます".into());
            }
        };
        let raw_job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if raw_job.is_null() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "隔離worker Job Objectを作成できません: {}",
                std::io::Error::last_os_error()
            ));
        }
        let job = unsafe { std::os::windows::io::OwnedHandle::from_raw_handle(raw_job) };
        let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if let Some(memory_limit) = process_memory_limit {
            limits.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY;
            limits.ProcessMemoryLimit = memory_limit;
        }
        let configured = unsafe {
            SetInformationJobObject(
                job.as_raw_handle(),
                JobObjectExtendedLimitInformation,
                std::ptr::from_ref(&limits).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if configured == 0
            || unsafe { AssignProcessToJobObject(job.as_raw_handle(), child.as_raw_handle()) } == 0
        {
            let error = std::io::Error::last_os_error();
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("隔離workerをJob Objectへ登録できません: {error}"));
        }
        Ok(Self { child, _job: job })
    }

    fn kill(&mut self) -> std::io::Result<()> {
        self.child.kill()
    }

    fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.child.try_wait()
    }

    fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        self.child.wait()
    }
}

impl Drop for IsolatedChild {
    fn drop(&mut self) {
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

struct SharedCancellation {
    marker: PathBuf,
    fence: File,
}

#[derive(Default)]
struct JobControl {
    cancelled: AtomicBool,
    commit_gate: Mutex<()>,
    child: Mutex<Option<IsolatedChild>>,
    shared_cancellation: Mutex<Option<SharedCancellation>>,
    recovery: Mutex<Option<Arc<recovery::RecoveryTracker>>>,
}

impl JobControl {
    fn is_cancelled(&self) -> bool {
        if self.cancelled.load(Ordering::SeqCst) {
            return true;
        }
        let Ok(boundary) = self.shared_cancellation.lock() else {
            return true;
        };
        let Some(boundary) = boundary.as_ref() else {
            return false;
        };
        cancellation_marker_exists(&boundary.marker)
    }

    fn cancel(&self) -> Result<(), String> {
        let boundary = self
            .shared_cancellation
            .lock()
            .map_err(|_| "隔離workerの取消境界を取得できません")?;
        if let Some(boundary) = boundary.as_ref() {
            fs2::FileExt::lock_exclusive(&boundary.fence)
                .map_err(|error| format!("隔離workerの公開境界をlockできません: {error}"))?;
            self.cancelled.store(true, Ordering::SeqCst);
            let result = write_private_cancel_marker(&boundary.marker);
            let unlock = fs2::FileExt::unlock(&boundary.fence)
                .map_err(|error| format!("隔離workerの公開境界をunlockできません: {error}"));
            result?;
            unlock?;
            return Ok(());
        }
        drop(boundary);
        // Signal first so no waiter can enter a later commit fence while this
        // call waits for the publication that already owns the gate.
        self.cancelled.store(true, Ordering::SeqCst);
        if let Ok(mut child) = self.child.lock() {
            if let Some(child) = child.as_mut() {
                let _ = child.kill();
            }
        }
        let _commit_guard = self
            .commit_gate
            .lock()
            .map_err(|_| "出力確定状態を取得できません")?;
        Ok(())
    }

    fn request_cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    fn install_shared_cancellation(&self, marker: PathBuf, fence: File) -> Result<(), String> {
        let mut boundary = self
            .shared_cancellation
            .lock()
            .map_err(|_| "隔離workerの取消境界を更新できません")?;
        if boundary.is_some() {
            return Err("隔離workerの取消境界は既に登録されています".into());
        }
        *boundary = Some(SharedCancellation { marker, fence });
        Ok(())
    }

    fn install_child(&self, child: IsolatedChild) -> Result<(), String> {
        let mut slot = self
            .child
            .lock()
            .map_err(|_| "隔離ワーカー状態を更新できません")?;
        if slot.is_some() {
            return Err("隔離ワーカーは既に実行中です".into());
        }
        *slot = Some(child);
        if self.is_cancelled() {
            if let Some(child) = slot.as_mut() {
                let _ = child.kill();
            }
        }
        Ok(())
    }

    fn install_recovery(&self, tracker: Arc<recovery::RecoveryTracker>) -> Result<(), String> {
        let mut slot = self
            .recovery
            .lock()
            .map_err(|_| "復旧状態を更新できません")?;
        if slot.is_some() {
            return Err("復旧状態は既に登録されています".into());
        }
        *slot = Some(tracker);
        Ok(())
    }

    fn recovery_attachment(&self) -> Result<recovery::RecoveryAttachment, String> {
        let tracker = self
            .recovery
            .lock()
            .map_err(|_| "復旧状態を取得できません")?
            .clone()
            .ok_or_else(|| "復旧状態が登録されていません".to_string())?;
        tracker.attachment()
    }

    fn cleanup_isolated_recovery(&self) -> Result<usize, String> {
        let tracker = self
            .recovery
            .lock()
            .map_err(|_| "復旧状態を取得できません")?
            .clone()
            .ok_or_else(|| "復旧状態が登録されていません".to_string())?;
        tracker.cleanup_isolated_stages()
    }

    fn track_stage(&self, output: &AtomicOutput) -> Result<recovery::RecoveryStageGuard, String> {
        let tracker = self
            .recovery
            .lock()
            .map_err(|_| "復旧状態を取得できません")?
            .clone();
        match tracker {
            Some(tracker) => tracker.track(output),
            None => Ok(recovery::RecoveryStageGuard::untracked()),
        }
    }

    fn finish_recovery(&self, status: &'static str) {
        let tracker = self
            .recovery
            .lock()
            .ok()
            .and_then(|tracker| tracker.clone());
        if let Some(tracker) = tracker {
            if let Err(error) = tracker.finish(status) {
                eprintln!("denoize desktop: recovery record cleanup failed: {error}");
            }
        }
    }

    fn wait_for_child(
        &self,
        timeout: std::time::Duration,
    ) -> Result<std::process::ExitStatus, String> {
        let started = Instant::now();
        loop {
            let status = {
                let mut slot = self
                    .child
                    .lock()
                    .map_err(|_| "隔離ワーカー状態を取得できません")?;
                let child = slot
                    .as_mut()
                    .ok_or_else(|| "隔離ワーカーが登録されていません".to_string())?;
                child
                    .try_wait()
                    .map_err(|error| format!("隔離ワーカー状態を確認できません: {error}"))?
            };
            if let Some(status) = status {
                let mut slot = self
                    .child
                    .lock()
                    .map_err(|_| "隔離ワーカー状態を更新できません")?;
                let mut child = slot
                    .take()
                    .ok_or_else(|| "隔離ワーカーが途中で失われました".to_string())?;
                let _ = child.wait();
                return Ok(status);
            }
            if started.elapsed() >= timeout {
                let mut slot = self
                    .child
                    .lock()
                    .map_err(|_| "隔離ワーカー状態を更新できません")?;
                let mut child = slot
                    .take()
                    .ok_or_else(|| "隔離ワーカーが途中で失われました".to_string())?;
                let _ = child.kill();
                let _ = child.wait();
                return Err("隔離プレビューワーカーが制限時間を超えました".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
    }

    fn wait_for_job_child(
        &self,
        cancellation_grace: std::time::Duration,
    ) -> Result<std::process::ExitStatus, String> {
        let mut cancellation_started = None;
        loop {
            let status = {
                let mut slot = self
                    .child
                    .lock()
                    .map_err(|_| "隔離worker状態を取得できません")?;
                let child = slot
                    .as_mut()
                    .ok_or_else(|| "隔離workerが登録されていません".to_string())?;
                child
                    .try_wait()
                    .map_err(|error| format!("隔離worker状態を確認できません: {error}"))?
            };
            if let Some(status) = status {
                let mut slot = self
                    .child
                    .lock()
                    .map_err(|_| "隔離worker状態を更新できません")?;
                let mut child = slot
                    .take()
                    .ok_or_else(|| "隔離workerが途中で失われました".to_string())?;
                let _ = child.wait();
                return Ok(status);
            }
            if self.is_cancelled() {
                let started = cancellation_started.get_or_insert_with(Instant::now);
                if started.elapsed() >= cancellation_grace {
                    let mut slot = self
                        .child
                        .lock()
                        .map_err(|_| "隔離worker状態を更新できません")?;
                    let mut child = slot
                        .take()
                        .ok_or_else(|| "隔離workerが途中で失われました".to_string())?;
                    let _ = child.kill();
                    return child
                        .wait()
                        .map_err(|error| format!("隔離workerの終了を待機できません: {error}"));
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
    }

    fn commit(&self, transaction: AtomicOutput, mode: CommitMode) -> Result<(), String> {
        self.commit_fence(|| transaction.commit(mode))
    }

    fn commit_fence<T>(&self, publish: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
        let _commit_guard = self
            .commit_gate
            .lock()
            .map_err(|_| "出力確定状態を取得できません")?;
        let boundary = self
            .shared_cancellation
            .lock()
            .map_err(|_| "隔離workerの取消境界を取得できません")?;
        if let Some(boundary) = boundary.as_ref() {
            fs2::FileExt::lock_exclusive(&boundary.fence)
                .map_err(|error| format!("隔離workerの公開境界をlockできません: {error}"))?;
            let cancelled = self.cancelled.load(Ordering::SeqCst)
                || cancellation_marker_exists(&boundary.marker);
            let result = if cancelled {
                Err("cancelled".into())
            } else {
                publish()
            };
            let unlock = fs2::FileExt::unlock(&boundary.fence)
                .map_err(|error| format!("隔離workerの公開境界をunlockできません: {error}"));
            return match (result, unlock) {
                (Err(error), _) => Err(error),
                (Ok(_), Err(error)) => Err(error),
                (Ok(value), Ok(())) => Ok(value),
            };
        }
        drop(boundary);
        check_cancelled(self)?;
        publish()
    }
}

fn cancellation_marker_exists(path: &Path) -> bool {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata.file_type().is_file(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(_) => true,
    }
}

fn write_private_cancel_marker(path: &Path) -> Result<(), String> {
    use std::fs::OpenOptions;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    match options.open(path) {
        Ok(file) => file
            .sync_all()
            .map_err(|error| format!("隔離workerの取消markerを同期できません: {error}")),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(format!("隔離workerの取消markerを作成できません: {error}")),
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ProcessOptions {
    backend: String,
    preset: Option<String>,
    mode: Option<String>,
    strength: f64,
    adaptive_noise: bool,
    vad: bool,
    channel_mode: String,
    downmix: String,
    loudness_lufs: Option<f64>,
    true_peak_dbtp: f64,
    preserve_metadata: bool,
    force: bool,
    mp3_bitrate_kbps: u32,
    aac_bitrate_kbps: u32,
    aac_encoder: String,
    onnx_model: Option<String>,
    onnx_sample_rate: u32,
    #[serde(default)]
    model_package: Option<String>,
    #[serde(default)]
    model_package_key: Option<String>,
    sgmse_profile: String,
    #[serde(default = "default_accelerator")]
    accelerator: String,
    #[serde(default)]
    deterministic: bool,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    max_process_memory_mb: Option<usize>,
    #[serde(default)]
    max_temporary_mb: Option<usize>,
    #[serde(default)]
    max_gpu_memory_mb: Option<usize>,
    #[serde(default = "default_max_gpu_jobs")]
    max_gpu_jobs: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GuiConfig {
    backend: String,
    preset: String,
    mode: String,
    strength: f64,
    adaptive_noise: bool,
    vad: bool,
    channels: String,
    downmix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    loudness_lufs: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    true_peak_dbtp: Option<f64>,
    preserve_metadata: bool,
    force: bool,
    mp3_bitrate_kbps: u32,
    m4a_bitrate_kbps: u32,
    aac_encoder: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    onnx_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    onnx_rate: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_package: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_package_key: Option<String>,
    sgmse_profile: String,
    #[serde(default = "default_accelerator")]
    accelerator: String,
    deterministic: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_process_memory_mb: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_temporary_mb: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_gpu_memory_mb: Option<usize>,
    #[serde(default = "default_max_gpu_jobs")]
    max_gpu_jobs: usize,
}

/// A typed, partial desktop/CLI configuration import.
///
/// Every field is optional so existing reusable TOML snippets continue to
/// overlay the settings currently shown in the UI. Serde still rejects unknown
/// fields and wrong value types before anything is applied.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GuiConfigPatch {
    backend: Option<String>,
    preset: Option<String>,
    mode: Option<String>,
    strength: Option<f64>,
    adaptive_noise: Option<bool>,
    vad: Option<bool>,
    channels: Option<String>,
    downmix: Option<String>,
    loudness_lufs: Option<f64>,
    true_peak_dbtp: Option<f64>,
    preserve_metadata: Option<bool>,
    force: Option<bool>,
    mp3_bitrate_kbps: Option<u32>,
    m4a_bitrate_kbps: Option<u32>,
    aac_encoder: Option<String>,
    onnx_model: Option<String>,
    onnx_rate: Option<u32>,
    model_package: Option<String>,
    model_package_key: Option<String>,
    sgmse_profile: Option<String>,
    accelerator: Option<String>,
    deterministic: Option<bool>,
    max_process_memory_mb: Option<usize>,
    max_temporary_mb: Option<usize>,
    max_gpu_memory_mb: Option<usize>,
    max_gpu_jobs: Option<usize>,
}

impl GuiConfig {
    fn process_options(&self) -> ProcessOptions {
        ProcessOptions {
            backend: self.backend.clone(),
            preset: Some(self.preset.clone()),
            mode: Some(self.mode.clone()),
            strength: self.strength,
            adaptive_noise: self.adaptive_noise,
            vad: self.vad,
            channel_mode: self.channels.clone(),
            downmix: self.downmix.clone(),
            loudness_lufs: self.loudness_lufs,
            true_peak_dbtp: self.true_peak_dbtp.unwrap_or(-1.0),
            preserve_metadata: self.preserve_metadata,
            force: self.force,
            mp3_bitrate_kbps: self.mp3_bitrate_kbps,
            aac_bitrate_kbps: self.m4a_bitrate_kbps,
            aac_encoder: self.aac_encoder.clone(),
            onnx_model: self.onnx_model.clone(),
            onnx_sample_rate: self.onnx_rate.unwrap_or(DEFAULT_MODEL_SAMPLE_RATE_HZ),
            model_package: self.model_package.clone(),
            model_package_key: self.model_package_key.clone(),
            sgmse_profile: self.sgmse_profile.clone(),
            accelerator: self.accelerator.clone(),
            deterministic: self.deterministic,
            seed: None,
            max_process_memory_mb: self.max_process_memory_mb,
            max_temporary_mb: self.max_temporary_mb,
            max_gpu_memory_mb: self.max_gpu_memory_mb,
            max_gpu_jobs: self.max_gpu_jobs,
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.loudness_lufs.is_none()
            && self
                .true_peak_dbtp
                .is_some_and(|true_peak| true_peak != -1.0)
        {
            return Err("true_peak_dbtp は loudness_lufs と一緒に指定してください".into());
        }
        validate_process_options(&self.process_options())
    }

    fn normalized(mut self) -> Result<Self, String> {
        let backend = configured_backend(&self.backend)?;
        if !backend.is_some_and(service::requires_external_model) {
            self.onnx_model = None;
            self.onnx_rate = None;
            self.model_package = None;
            self.model_package_key = None;
        } else if self.model_package.is_some() && self.model_package_key.is_some() {
            // The authenticated manifest is the only model-rate authority.
            // Omitting this raw-ONNX field also keeps exported TOML directly
            // consumable by the CLI package flags.
            self.onnx_rate = None;
        }
        self.validate()?;
        Ok(self)
    }
}

impl GuiConfigPatch {
    fn merge(self, mut current: GuiConfig) -> Result<GuiConfig, String> {
        let explicitly_discards_model_fields = match self.backend.as_deref() {
            Some(backend) => {
                !configured_backend(backend)?.is_some_and(service::requires_external_model)
            }
            None => false,
        };
        macro_rules! replace_present {
            ($field:ident) => {
                if let Some(value) = self.$field {
                    current.$field = value;
                }
            };
        }

        replace_present!(backend);
        replace_present!(preset);
        replace_present!(mode);
        replace_present!(strength);
        replace_present!(adaptive_noise);
        replace_present!(vad);
        replace_present!(channels);
        replace_present!(downmix);
        let explicit_loudness_clear = self.loudness_lufs.is_none()
            && self
                .true_peak_dbtp
                .is_some_and(|true_peak| true_peak == -1.0);
        if explicit_loudness_clear {
            current.loudness_lufs = None;
            current.true_peak_dbtp = None;
        } else if let Some(value) = self.loudness_lufs {
            current.loudness_lufs = Some(value);
            if let Some(true_peak) = self.true_peak_dbtp {
                current.true_peak_dbtp = Some(true_peak);
            }
        } else if let Some(value) = self.true_peak_dbtp {
            current.true_peak_dbtp = Some(value);
        }
        replace_present!(preserve_metadata);
        replace_present!(force);
        replace_present!(mp3_bitrate_kbps);
        replace_present!(m4a_bitrate_kbps);
        replace_present!(aac_encoder);
        let raw_model_supplied = self.onnx_model.is_some();
        let package_supplied = self.model_package.is_some() || self.model_package_key.is_some();
        if !explicitly_discards_model_fields && raw_model_supplied && package_supplied {
            return Err(
                "onnx_model と model_package/model_package_key は同時に指定できません".into(),
            );
        }
        if let Some(value) = self.onnx_model {
            current.onnx_model = Some(value);
            current.model_package = None;
            current.model_package_key = None;
        }
        if let Some(value) = self.onnx_rate {
            current.onnx_rate = Some(value);
        }
        if package_supplied {
            current.onnx_model = None;
        }
        if let Some(value) = self.model_package {
            current.model_package = Some(value);
        }
        if let Some(value) = self.model_package_key {
            current.model_package_key = Some(value);
        }
        replace_present!(sgmse_profile);
        replace_present!(accelerator);
        replace_present!(deterministic);
        if let Some(value) = self.max_process_memory_mb {
            current.max_process_memory_mb = Some(value);
        }
        if let Some(value) = self.max_temporary_mb {
            current.max_temporary_mb = Some(value);
        }
        if let Some(value) = self.max_gpu_memory_mb {
            current.max_gpu_memory_mb = Some(value);
        }
        replace_present!(max_gpu_jobs);
        current.normalized()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProcessRequest {
    input: String,
    output: String,
    #[serde(default)]
    expected_input_fingerprint: Option<batch_resume::FileFingerprint>,
    #[serde(default)]
    expected_recipe: Option<Digest>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    resume: bool,
    #[serde(default = "default_stream_block_frames")]
    stream_frames: usize,
    #[serde(default)]
    receipt: Option<String>,
    #[serde(default)]
    receipt_key: Option<String>,
    options: ProcessOptions,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RecommendationRequest {
    input: String,
    goal: String,
    calibrate: bool,
    analysis_seconds: u32,
    max_memory_mb: Option<usize>,
    max_gpu_memory_mb: Option<usize>,
    accelerator: String,
    deterministic: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DiagnosticRequest {
    input: String,
    analysis_seconds: u32,
    max_memory_mb: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AssessmentRequest {
    baseline: Option<String>,
    candidate: String,
    analysis_seconds: u32,
    max_memory_mb: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RestorationRequest {
    input: String,
    output: Option<String>,
    operations: Vec<String>,
    detect_only: bool,
    max_memory_mb: Option<usize>,
    preserve_metadata: bool,
    replace: bool,
    wpe_channel_mode: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DesktopRestorationResult {
    output: Option<String>,
    report: denoize::RestorationReport,
    mask: denoize::RestorationMask,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[cfg_attr(not(feature = "full"), allow(dead_code))]
struct UniversalRestorationRequest {
    input: String,
    output: String,
    model_package: String,
    model_package_key: String,
    model_family: String,
    render_role: String,
    allow_experimental: bool,
    analysis_seconds: u32,
    minimum_degradation_score: f64,
    maximum_energy_gain_db: f64,
    maximum_peak_gain_db: f64,
    maximum_new_clipping_ratio: f64,
    maximum_quality_regression: f64,
    accelerator: String,
    max_memory_mb: Option<usize>,
    preserve_metadata: bool,
    replace: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DesktopUniversalRestorationResult {
    output: String,
    report: denoize::UniversalRestorationReport,
    mask: denoize::UniversalRestorationMask,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[cfg_attr(not(feature = "full"), allow(dead_code))]
struct TargetSpeakerRequest {
    mixture: String,
    enrollment: String,
    output: String,
    model_package: String,
    model_package_key: String,
    promotion_evidence: String,
    promotion_evidence_key: String,
    minimum_present_probability: f64,
    minimum_absent_probability: f64,
    maximum_energy_gain_db: f64,
    maximum_peak_gain_db: f64,
    maximum_new_clipping_ratio: f64,
    accelerator: String,
    max_memory_mb: Option<usize>,
    preserve_metadata: bool,
    replace: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DesktopTargetSpeakerResult {
    output: Option<String>,
    report: denoize::TargetSpeakerExtractionReport,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct EvaluationValidationRequest {
    manifest: String,
    corpus_root: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct EvaluationRunRequest {
    manifest: String,
    corpus_root: String,
    secret_key: String,
    output: String,
    listening_result: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct EvaluationVerificationRequest {
    result: String,
    public_key: String,
    manifest: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct EvaluationComparisonRequest {
    baseline: String,
    candidate: String,
    baseline_key: String,
    candidate_key: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ProjectDocumentRequest {
    manifest: String,
    root: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ProjectTimelineRequest {
    manifest: String,
    root: String,
    timeline: String,
    output: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ProjectAssemblyRequest {
    manifest: String,
    root: String,
    timeline: String,
    output: String,
    plan: denoize::ProjectExecutionPlan,
    receipt: Option<String>,
    receipt_key: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ProjectBundleBuildRequest {
    manifest: String,
    root: String,
    output: String,
    include_sources: bool,
    source_payload_limit_mb: Option<usize>,
    include_models: bool,
    model_payload_limit_mb: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchRequest {
    inputs: Vec<String>,
    input_dir: Option<String>,
    output_dir: String,
    output_format: String,
    recursive: bool,
    jobs: usize,
    resume: bool,
    #[serde(default)]
    receipt: Option<String>,
    #[serde(default)]
    receipt_key: Option<String>,
    options: ProcessOptions,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WatchRequest {
    input_dir: String,
    output_dir: String,
    receipt_key: String,
    output_format: String,
    recursive: bool,
    settle_millis: u64,
    retry_initial_millis: u64,
    retry_max_millis: u64,
    max_attempts: u32,
    max_files: usize,
    #[serde(default)]
    quarantine_dir: Option<String>,
    #[serde(default)]
    receipt_dir: Option<String>,
    #[serde(default)]
    state_path: Option<String>,
    options: ProcessOptions,
}

#[derive(Clone, Debug)]
struct BatchItem {
    input: PathBuf,
    output: PathBuf,
    output_format: OutputFormat,
    item_id: Digest,
}

#[derive(Clone, Debug)]
struct PreparedBatchItem {
    item: BatchItem,
    input_probe: denoize::AudioProbe,
    input_channels: usize,
    input_frames: u64,
    sample_rate: u32,
    encode: EncodeOptions,
    metadata_policy: MetadataPolicy,
    processing: service::ResolvedProcessingOptions,
    backend_session: Arc<BackendSession>,
    _backend_session_permit: Arc<ResourcePermit>,
    governor: ResourceGovernor,
    resource_request: ResourceRequest,
    decode_limits: DecodeLimits,
    metadata_limits: MetadataLimits,
    expectation: ResumeExpectation,
}

#[derive(Clone, Debug)]
struct PlannedBatchItem {
    prepared: PreparedBatchItem,
    decision: ResumeDecision,
    existing_output: Option<batch_resume::FileFingerprint>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(not(feature = "live"), allow(dead_code))]
struct LiveRequest {
    input_device: Option<String>,
    output_device: Option<String>,
    chunk_ms: u32,
    target_latency_ms: Option<u32>,
    max_drift_ppm: Option<u32>,
    reconnect_timeout_ms: Option<u32>,
    backend: String,
    options: ProcessOptions,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LiveDevices {
    inputs: Vec<String>,
    outputs: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
#[cfg(feature = "live")]
struct LiveEvent {
    status: &'static str,
    connection_state: &'static str,
    message: String,
    #[serde(flatten)]
    metrics: LiveEventMetrics,
    accelerator: Option<AcceleratorResult>,
    error: Option<DesktopError>,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
#[cfg(feature = "live")]
struct LiveEventMetrics {
    sample_rate: u32,
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
}

#[cfg(feature = "live")]
impl From<denoize::live::LiveStatus> for LiveEventMetrics {
    fn from(status: denoize::live::LiveStatus) -> Self {
        Self {
            sample_rate: status.sample_rate,
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
        }
    }
}

#[cfg(feature = "live")]
fn live_connection_event(
    state: denoize::live::LiveConnectionState,
) -> (&'static str, &'static str) {
    match state {
        denoize::live::LiveConnectionState::Connecting => ("connecting", "デバイスへ接続中"),
        denoize::live::LiveConnectionState::Priming => ("priming", "再生キューを準備中"),
        denoize::live::LiveConnectionState::Running => ("running", "ライブ処理中"),
        denoize::live::LiveConnectionState::Recovering => ("recovering", "デバイス接続を復旧中"),
        _ => ("unknown", "ライブ状態を更新中"),
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct AcceleratorResult {
    requested: String,
    effective: String,
    fallback: Option<String>,
}

fn accelerator_result(selection: denoize::AcceleratorSelection) -> AcceleratorResult {
    AcceleratorResult {
        requested: selection.requested().name().into(),
        effective: selection.effective().name().into(),
        fallback: selection.fallback().map(|fallback| fallback.name().into()),
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct JobProgress {
    job_id: u64,
    kind: String,
    status: String,
    message: String,
    current: usize,
    total: usize,
    fraction: f64,
    elapsed_seconds: f64,
    output: Option<String>,
    error: Option<DesktopError>,
    eta_seconds: Option<f64>,
    item: Option<String>,
    item_status: Option<String>,
    item_id: Option<String>,
    resume_reason: Option<String>,
    accelerator: Option<AcceleratorResult>,
}

#[derive(Debug)]
struct ProcessFileResult {
    output: String,
    accelerator: denoize::AcceleratorSelection,
}

struct DesktopReceiptContext {
    path: PathBuf,
    key: ReceiptSecretKey,
    stage: AtomicOutput,
    publication: &'static str,
    reason: &'static str,
    _recovery_stage: Option<recovery::RecoveryStageGuard>,
}

struct DesktopBatchReceiptContext {
    path: PathBuf,
    key: ReceiptSecretKey,
    stage: AtomicOutput,
    plan: ExecutionPlan,
    _recovery_stage: Option<recovery::RecoveryStageGuard>,
}

struct UnplannedDesktopBatchReceipt {
    path: PathBuf,
    key_path: PathBuf,
    key: ReceiptSecretKey,
    stage: AtomicOutput,
    _recovery_stage: Option<recovery::RecoveryStageGuard>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReceiptVerificationRequest {
    receipt: String,
    key: Option<String>,
    policy: Option<String>,
    plan: Option<String>,
    output_root: Option<String>,
}

fn write_desktop_receipt_stage(
    stage: &mut AtomicOutput,
    path: &Path,
    receipt: &SignedExecutionReceipt,
) -> Result<(), String> {
    let mut bytes = receipt.to_pretty_json()?.into_bytes();
    bytes.push(b'\n');
    stage
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| format!("実行証明 {} を書き込めません: {error}", path.display()))?;
    stage
        .file_mut()
        .sync_data()
        .map_err(|error| format!("実行証明 {} を同期できません: {error}", path.display()))
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelProgress {
    job_id: u64,
    name: String,
    status: &'static str,
    message: String,
    downloaded: u64,
    total: Option<u64>,
    fraction: Option<f64>,
    error: Option<DesktopError>,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelActionOptions {
    #[serde(default)]
    offline: bool,
    source_url: Option<String>,
    proxy_url: Option<String>,
    #[serde(default)]
    direct: bool,
    bearer_token: Option<String>,
    basic_username: Option<String>,
    basic_password: Option<String>,
    source_path: Option<String>,
}

fn model_action_options(
    input: Option<ModelActionOptions>,
) -> Result<(ModelDownloadOptions, Option<PathBuf>), String> {
    model_action_options_with_environment(input, |name| std::env::var(name).ok())
}

fn catalog_action_options(
    input: Option<ModelActionOptions>,
) -> Result<ModelDownloadOptions, String> {
    catalog_action_options_with_environment(input, |name| std::env::var(name).ok())
}

fn catalog_action_options_with_environment<F>(
    input: Option<ModelActionOptions>,
    mut read_environment: F,
) -> Result<ModelDownloadOptions, String>
where
    F: FnMut(&str) -> Option<String>,
{
    let (options, source) = model_action_options_with_environment(input, |name| {
        let name = if name == "DENOIZE_MODEL_URL" {
            "DENOIZE_MODEL_CATALOG_URL"
        } else {
            name
        };
        read_environment(name)
    })?;
    if source.is_some() {
        return Err("ローカルカタログはCLIの models catalog import で導入してください".into());
    }
    Ok(options)
}

fn model_action_options_with_environment<F>(
    input: Option<ModelActionOptions>,
    mut read_environment: F,
) -> Result<(ModelDownloadOptions, Option<PathBuf>), String>
where
    F: FnMut(&str) -> Option<String>,
{
    let input = input.unwrap_or_default();
    let source_url = trimmed_value(input.source_url);
    let proxy_url = trimmed_value(input.proxy_url);
    let bearer_token = trimmed_value(input.bearer_token);
    let basic_username = trimmed_value(input.basic_username);
    let basic_password = input.basic_password.filter(|value| !value.is_empty());
    let source_path = input
        .source_path
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);

    if input.direct && proxy_url.is_some() {
        return Err("プロキシURLと直接接続は同時に指定できません".into());
    }
    if bearer_token.is_some() && (basic_username.is_some() || basic_password.is_some()) {
        return Err("Bearer認証とBasic認証は同時に指定できません".into());
    }
    let authentication = if let Some(token) = bearer_token {
        Some(ModelAuthentication::Bearer(token))
    } else {
        match (basic_username, basic_password) {
            (Some(username), Some(password)) => {
                Some(ModelAuthentication::Basic { username, password })
            }
            (None, None) => None,
            _ => return Err("Basic認証のユーザー名とパスワードは両方指定してください".into()),
        }
    };

    if source_path.is_some() {
        if input.offline
            || source_url.is_some()
            || proxy_url.is_some()
            || input.direct
            || authentication.is_some()
        {
            return Err(
                "ローカルファイルはネットワーク・認証オプションと同時に指定できません".into(),
            );
        }
        return Ok((ModelDownloadOptions::default(), source_path));
    }

    let overrides_authentication = authentication.is_some();
    let mut options = ModelDownloadOptions::from_env_with(|name| {
        let overridden = match name {
            "DENOIZE_MODEL_OFFLINE" => input.offline,
            "DENOIZE_MODEL_URL" => source_url.is_some(),
            "DENOIZE_MODEL_PROXY" => input.direct || proxy_url.is_some(),
            "DENOIZE_MODEL_BEARER_TOKEN" | "DENOIZE_MODEL_USERNAME" | "DENOIZE_MODEL_PASSWORD" => {
                overrides_authentication
            }
            _ => false,
        };
        (!overridden).then(|| read_environment(name)).flatten()
    })?;
    if input.offline {
        options.offline = true;
    }
    if source_url.is_some() {
        options.source_url = source_url;
    }
    if input.direct {
        options.proxy = ModelProxy::Disabled;
    } else if let Some(url) = proxy_url {
        options.proxy = ModelProxy::Url(url);
    }
    if authentication.is_some() {
        options.authentication = authentication;
    }
    Ok((options, source_path))
}

fn trimmed_value(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let value = value.trim();
        (!value.is_empty()).then(|| value.to_owned())
    })
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AppInfo {
    version: &'static str,
    backends: Vec<BackendInfo>,
    formats: Vec<&'static str>,
    fdk_available: bool,
    accelerators: Vec<AcceleratorInfo>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BackendInfo {
    name: &'static str,
    external_model: bool,
    managed_model: Option<&'static str>,
    sample_rate: Option<u32>,
    accelerated: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AcceleratorInfo {
    name: &'static str,
    compiled: bool,
    available: bool,
    device: Option<String>,
    memory_bytes: Option<u64>,
    compute_capability: Option<String>,
    detail: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ArtifactMetrics {
    musical_noise_score: f64,
    pumping_score: f64,
    transient_loss_score: f64,
    phase_distortion_score: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ComparisonMetrics {
    si_sdr_db: f64,
    si_snr_db: f64,
    snr_db: f64,
    segmental_snr_db: f64,
    stereo_side_sdr_db: Option<f64>,
    correlation_error: Option<f64>,
    stoi: Option<f64>,
    pesq: Option<f64>,
    visqol: Option<f64>,
    artifact_scores: ArtifactMetrics,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ComparisonMetricSet {
    noisy: ComparisonMetrics,
    enhanced: ComparisonMetrics,
    improvement: ComparisonMetrics,
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
    metrics: ComparisonMetricSet,
}

fn comparison_metrics(report: &BenchmarkReport) -> ComparisonMetrics {
    ComparisonMetrics {
        si_sdr_db: report.si_sdr_db,
        si_snr_db: report.si_snr_db,
        snr_db: report.snr_db,
        segmental_snr_db: report.segmental_snr_db,
        stereo_side_sdr_db: report.stereo_side_sdr_db,
        correlation_error: report.correlation_error,
        stoi: report.stoi,
        pesq: report.pesq,
        visqol: report.visqol,
        artifact_scores: ArtifactMetrics {
            musical_noise_score: report.artifact_scores.musical_noise_score,
            pumping_score: report.artifact_scores.pumping_score,
            transient_loss_score: report.artifact_scores.transient_loss_score,
            phase_distortion_score: report.artifact_scores.phase_distortion_score,
        },
    }
}

fn optional_metric_difference(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a - b),
        _ => None,
    }
}

fn comparison_metric_set(report: &ComparisonReport) -> ComparisonMetricSet {
    let noisy = comparison_metrics(&report.noisy);
    let enhanced = comparison_metrics(&report.enhanced);
    let improvement = ComparisonMetrics {
        si_sdr_db: report.enhanced.si_sdr_db - report.noisy.si_sdr_db,
        si_snr_db: report.enhanced.si_snr_db - report.noisy.si_snr_db,
        snr_db: report.enhanced.snr_db - report.noisy.snr_db,
        segmental_snr_db: report.enhanced.segmental_snr_db - report.noisy.segmental_snr_db,
        stereo_side_sdr_db: optional_metric_difference(
            report.enhanced.stereo_side_sdr_db,
            report.noisy.stereo_side_sdr_db,
        ),
        correlation_error: optional_metric_difference(
            report.noisy.correlation_error,
            report.enhanced.correlation_error,
        ),
        stoi: optional_metric_difference(report.enhanced.stoi, report.noisy.stoi),
        pesq: optional_metric_difference(report.enhanced.pesq, report.noisy.pesq),
        visqol: optional_metric_difference(report.enhanced.visqol, report.noisy.visqol),
        artifact_scores: ArtifactMetrics {
            musical_noise_score: report.noisy.artifact_scores.musical_noise_score
                - report.enhanced.artifact_scores.musical_noise_score,
            pumping_score: report.noisy.artifact_scores.pumping_score
                - report.enhanced.artifact_scores.pumping_score,
            transient_loss_score: report.noisy.artifact_scores.transient_loss_score
                - report.enhanced.artifact_scores.transient_loss_score,
            phase_distortion_score: optional_metric_difference(
                report.noisy.artifact_scores.phase_distortion_score,
                report.enhanced.artifact_scores.phase_distortion_score,
            ),
        },
    };
    ComparisonMetricSet {
        noisy,
        enhanced,
        improvement,
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelRow {
    name: String,
    backend: String,
    license: String,
    sample_rate: u32,
    revision: String,
    installed: bool,
    path: String,
    catalog_sequence: u64,
    catalog_sha256: String,
    catalog_signing_key: String,
    provenance_source: Option<String>,
    installed_at_unix_seconds: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelCatalogRow {
    sequence: u64,
    sha256: String,
    signing_key: String,
    origin: String,
    model_count: usize,
    highest_accepted_sequence: u64,
    cached_path: String,
    issued_at_unix_seconds: Option<u64>,
    expires_at_unix_seconds: Option<u64>,
    trust_root_version: u64,
    trust_root_sha256: String,
    trust_root_expires_at_unix_seconds: u64,
    trust_root_highest_observed_unix_seconds: Option<u64>,
    acquisition_allowed: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelCacheIssueRow {
    kind: String,
    path: String,
    model: Option<String>,
    detail: String,
    prunable: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelCacheHealthRow {
    name: String,
    path: String,
    status: String,
    issues: Vec<ModelCacheIssueRow>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelCacheReportRow {
    cache_dir: String,
    catalog_sequence: u64,
    catalog_sha256: String,
    clean: bool,
    models: Vec<ModelCacheHealthRow>,
    issues: Vec<ModelCacheIssueRow>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelLibraryRow {
    models: Vec<ModelRow>,
    health: ModelCacheReportRow,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelPruneReportRow {
    dry_run: bool,
    would_remove: Vec<String>,
    removed: Vec<String>,
    retained: Vec<ModelCacheIssueRow>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct OfflineBundleModelRow {
    name: String,
    backend: String,
    artifact_filename: String,
    artifact_sha256: String,
    artifact_size_bytes: u64,
    license_filename: String,
    license_sha256: String,
    license_size_bytes: u64,
    provenance_filename: String,
    provenance_sha256: String,
    provenance_size_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct OfflineBundleRow {
    format_version: u32,
    bundle_sha256: String,
    size_bytes: u64,
    catalog_sequence: u64,
    catalog_sha256: String,
    catalog_signing_key_id: String,
    catalog_issued_at_unix_seconds: Option<u64>,
    catalog_expires_at_unix_seconds: Option<u64>,
    trust_root_version: u64,
    trust_root_sha256: String,
    models: Vec<OfflineBundleModelRow>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct OfflineBundleImportRow {
    bundle: OfflineBundleRow,
    installed: Vec<String>,
    already_present: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PreviewProgress {
    job_id: u64,
    status: &'static str,
    message: String,
    result: Option<preview::PreviewResult>,
    error: Option<DesktopError>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DropSelection {
    audio_files: Vec<String>,
    directories: Vec<String>,
    ignored: Vec<String>,
}

#[tauri::command]
fn app_info() -> AppInfo {
    let hardware = denoize::hardware_capabilities();
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
                    "bsrnn" | "mossformer2" => Some(48_000),
                    "onnx" | "mpsenet" | "sgmse" | "gtcrn" => Some(16_000),
                    _ => None,
                },
                accelerated: denoize::backend_supports_acceleration(backend),
            })
            .collect(),
        formats: vec!["wav", "flac", "opus", "mp3", "m4a", "aac"],
        fdk_available: cfg!(feature = "fdk-aac-encoder"),
        accelerators: hardware
            .runtimes()
            .iter()
            .map(|runtime| AcceleratorInfo {
                name: runtime.runtime().name(),
                compiled: runtime.compiled(),
                available: runtime.available(),
                device: runtime.device().map(str::to_owned),
                memory_bytes: runtime.memory_bytes(),
                compute_capability: runtime.compute_capability().map(str::to_owned),
                detail: runtime.detail().map(str::to_owned),
            })
            .collect(),
    }
}

fn desktop_recommendation_options(
    request: &RecommendationRequest,
) -> Result<denoize::RecommendationOptions, String> {
    let goal = denoize::RecommendationGoal::parse(&request.goal)
        .ok_or_else(|| format!("不明な推奨目標です: {}", request.goal))?;
    let accelerator = AcceleratorPreference::parse(&request.accelerator)
        .ok_or_else(|| format!("不明なアクセラレータです: {}", request.accelerator))?;
    let maximum = checked_desktop_mib(request.max_memory_mb, "プロセスメモリ上限")?;
    let maximum_gpu = checked_desktop_mib(request.max_gpu_memory_mb, "GPUメモリ上限")?;
    let limits = DecodeLimits::new(
        denoize::metadata_limits_for_available_memory(maximum),
        maximum,
    );
    let options = denoize::RecommendationOptions::new()
        .with_goal(goal)
        .with_analysis_seconds(request.analysis_seconds)
        .with_calibration(request.calibrate)
        .with_decode_limits(limits)
        .with_max_gpu_memory_bytes(maximum_gpu)
        .with_accelerator(accelerator)
        .with_deterministic(request.deterministic);
    options.validate()?;
    Ok(options)
}

#[tauri::command]
async fn recommend_settings(
    request: RecommendationRequest,
) -> DesktopResult<denoize::RecommendationReport> {
    let options = desktop_recommendation_options(&request)?;
    if request.input.trim().is_empty() {
        return Err("推奨を分析する入力ファイルを選択してください".into());
    }
    let input = request.input;
    Ok(tauri::async_runtime::spawn_blocking(move || {
        denoize::recommend_file_with_options(input, options)
    })
    .await
    .map_err(|error| format!("推奨分析タスクに失敗しました: {error}"))??)
}

fn desktop_diagnostic_options(
    analysis_seconds: u32,
    max_memory_mb: Option<usize>,
) -> Result<denoize::DiagnosticOptions, String> {
    let maximum = checked_desktop_mib(max_memory_mb, "プロセスメモリ上限")?;
    let limits = DecodeLimits::new(
        denoize::metadata_limits_for_available_memory(maximum),
        maximum,
    );
    let options = denoize::DiagnosticOptions::new()
        .with_analysis_seconds(analysis_seconds)
        .with_decode_limits(limits);
    options.validate()?;
    Ok(options)
}

#[tauri::command]
async fn diagnose_audio_input(
    request: DiagnosticRequest,
) -> DesktopResult<denoize::DiagnosticReport> {
    let options = desktop_diagnostic_options(request.analysis_seconds, request.max_memory_mb)?;
    if request.input.trim().is_empty() {
        return Err("劣化診断する入力ファイルを選択してください".into());
    }
    let input = request.input;
    Ok(tauri::async_runtime::spawn_blocking(move || {
        denoize::diagnose_file_with_options(input, options)
    })
    .await
    .map_err(|error| format!("劣化診断タスクに失敗しました: {error}"))??)
}

#[tauri::command]
async fn assess_audio_inputs(
    request: AssessmentRequest,
) -> DesktopResult<denoize::AssessmentReport> {
    let options = desktop_diagnostic_options(request.analysis_seconds, request.max_memory_mb)?;
    if request.candidate.trim().is_empty() {
        return Err("品質評価する入力ファイルを選択してください".into());
    }
    if request
        .baseline
        .as_deref()
        .is_some_and(|baseline| baseline.trim().is_empty())
    {
        return Err("比較元ファイルは空でないパスにしてください".into());
    }
    let baseline = request.baseline;
    let candidate = request.candidate;
    Ok(
        tauri::async_runtime::spawn_blocking(move || match baseline {
            Some(baseline) => denoize::compare_files_with_options(baseline, candidate, options),
            None => denoize::assess_file_with_options(candidate, options),
        })
        .await
        .map_err(|error| format!("品質評価タスクに失敗しました: {error}"))??,
    )
}

fn desktop_restoration_config(
    request: &RestorationRequest,
) -> Result<denoize::RestorationConfig, String> {
    let mut config = denoize::RestorationConfig::default();
    config.mode = if request.detect_only {
        denoize::RestorationMode::DetectOnly
    } else {
        denoize::RestorationMode::Apply
    };
    config.operations = request
        .operations
        .iter()
        .map(|operation| match operation.as_str() {
            "declip" => Ok(denoize::RestorationOperation::Declip),
            "declick" => Ok(denoize::RestorationOperation::Declick),
            "dehum" => Ok(denoize::RestorationOperation::Dehum),
            "dereverb" => Ok(denoize::RestorationOperation::Dereverb),
            "wind-plosive" => Ok(denoize::RestorationOperation::WindPlosive),
            _ => Err(format!("不明な復元処理です: {operation}")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    config.dereverb.channel_mode = match request.wpe_channel_mode.as_str() {
        "independent" => denoize::WpeChannelMode::Independent,
        "multichannel" => denoize::WpeChannelMode::Multichannel,
        value => return Err(format!("不明なWPEチャンネルモードです: {value}")),
    };
    config.validate()?;
    Ok(config)
}

fn run_desktop_restoration(
    request: RestorationRequest,
) -> Result<DesktopRestorationResult, String> {
    const MAX_DESKTOP_MASK_RUNS: usize = 200_000;
    if request.input.trim().is_empty() {
        return Err("復元する入力ファイルを選択してください".into());
    }
    if !request.detect_only && request.output.as_deref().is_none_or(str::is_empty) {
        return Err("適用モードでは音声の保存先を選択してください".into());
    }
    let config = desktop_restoration_config(&request)?;
    let input = Path::new(&request.input);
    let output_path = request.output.as_deref().map(Path::new);
    let mut paths = vec![("入力", input)];
    if let Some(output) = output_path {
        paths.push(("出力", output));
        ensure_output_available(output, request.replace)?;
    }
    require_distinct_execution_paths(&paths)?;
    let maximum = checked_desktop_mib(request.max_memory_mb, "プロセスメモリ上限")?;
    let limits = DecodeLimits::new(
        denoize::metadata_limits_for_available_memory(maximum),
        maximum,
    );
    let mut input_session = denoize::AudioInputSession::open(input)?;
    denoize::ensure_memory_limit(
        denoize::estimate_session_memory_bytes(&input_session),
        request.max_memory_mb,
        "desktop restoration input preflight",
    )?;
    let audio = read_audio_from_session_with_limits(&mut input_session, limits)?;
    let working_set = denoize::estimate_restoration_memory_bytes(&audio, &config);
    denoize::ensure_memory_limit(
        working_set,
        request.max_memory_mb,
        "desktop restoration working set",
    )?;
    let metadata = if output_path.is_some() && request.preserve_metadata {
        input_session
            .read_metadata_with_limits(desktop_retained_metadata_limits(maximum, working_set))?
    } else {
        None
    };
    let result = denoize::restore_audio(&audio, &config)?;
    if result.mask.runs.len() > MAX_DESKTOP_MASK_RUNS {
        return Err(format!(
            "復元マスクがDesktop表示上限の{MAX_DESKTOP_MASK_RUNS} runsを超えました。CLIの--maskを使用してください"
        ));
    }
    if let Some(output) = output_path {
        let format = OutputFormat::from_path(output)?;
        let encode = EncodeOptions::default();
        encode.validate_options(format)?;
        format.validate_config(&result.audio, &encode)?;
        denoize::write_audio_transactional(
            output,
            &result.audio,
            encode,
            metadata,
            if request.replace {
                CommitMode::Replace
            } else {
                CommitMode::NoClobber
            },
        )?;
    }
    Ok(DesktopRestorationResult {
        output: request.output,
        report: result.report,
        mask: result.mask,
    })
}

#[tauri::command]
async fn restore_audio_input(
    request: RestorationRequest,
) -> DesktopResult<DesktopRestorationResult> {
    Ok(
        tauri::async_runtime::spawn_blocking(move || run_desktop_restoration(request))
            .await
            .map_err(|error| format!("決定的復元タスクに失敗しました: {error}"))??,
    )
}

#[cfg_attr(not(feature = "full"), allow(dead_code))]
fn desktop_universal_restoration_config(
    request: &UniversalRestorationRequest,
) -> Result<denoize::UniversalRestorationConfig, String> {
    let config = denoize::UniversalRestorationConfig {
        model_family: denoize::UniversalModelFamily::parse(&request.model_family)
            .ok_or_else(|| format!("不明な汎用復元モデル種別です: {}", request.model_family))?,
        render_role: denoize::UniversalRenderRole::parse(&request.render_role)
            .ok_or_else(|| format!("不明な汎用復元レンダー種別です: {}", request.render_role))?,
        allow_experimental: request.allow_experimental,
        analysis_seconds: request.analysis_seconds,
        minimum_degradation_score: request.minimum_degradation_score,
        maximum_energy_gain_db: request.maximum_energy_gain_db,
        maximum_peak_gain_db: request.maximum_peak_gain_db,
        maximum_new_clipping_ratio: request.maximum_new_clipping_ratio,
        maximum_quality_score_regression: request.maximum_quality_regression,
    };
    config.validate()?;
    Ok(config)
}

#[cfg(feature = "full")]
fn run_desktop_universal_restoration(
    request: UniversalRestorationRequest,
) -> Result<DesktopUniversalRestorationResult, String> {
    const MAX_DESKTOP_UNIVERSAL_MASK_RUNS: usize = 200_000;
    for (value, label) in [
        (&request.input, "汎用復元入力"),
        (&request.output, "汎用復元出力"),
        (&request.model_package, "署名付きモデルパッケージ"),
        (&request.model_package_key, "モデルパッケージ公開鍵"),
    ] {
        if value.trim().is_empty() {
            return Err(format!("{label}を選択してください"));
        }
    }
    let config = desktop_universal_restoration_config(&request)?;
    let accelerator = AcceleratorPreference::parse(&request.accelerator)
        .ok_or_else(|| format!("不明なアクセラレータです: {}", request.accelerator))?;
    checked_desktop_mib(request.max_memory_mb, "プロセスメモリ上限")?;

    let input = Path::new(&request.input);
    let output = Path::new(&request.output);
    let package_path = Path::new(&request.model_package);
    let package_key_path = Path::new(&request.model_package_key);
    require_distinct_execution_paths(&[
        ("入力", input),
        ("出力", output),
        ("モデルパッケージ", package_path),
        ("モデル公開鍵", package_key_path),
    ])?;
    ensure_output_available(output, request.replace)?;

    let package = denoize::RuntimeModelPackage::open(package_path, package_key_path)?;
    if package.manifest_v2().is_none() {
        return Err("汎用復元には署名済みruntime model package v2が必要です".into());
    }
    let mut backend_options = BackendOptions::default().with_runtime_model_package(package);
    backend_options.deterministic = true;
    backend_options.accelerator = accelerator;
    let accelerator = denoize::select_accelerator_for_options(Backend::Bsrnn, &backend_options)?;
    let profile = backend_options
        .runtime_package
        .as_ref()
        .expect("universal restoration retains its authenticated package")
        .precision_profile_for(accelerator.effective())?
        .expect("runtime model package v2 selects a precision profile");
    let model_working_set = profile
        .resources
        .max_session_memory_bytes
        .saturating_add(profile.resources.max_worker_memory_bytes);
    denoize::ensure_memory_limit(
        model_working_set,
        request.max_memory_mb,
        "desktop universal restoration model working set",
    )?;
    // Fail closed on model bytes, graph semantics, and numerical vectors
    // before the user's audio file is opened.
    let session =
        BackendSession::prepare_with_accelerator(Backend::Bsrnn, backend_options, accelerator)?;

    let maximum = checked_desktop_mib(request.max_memory_mb, "プロセスメモリ上限")?;
    let decode_maximum = maximum.map(|limit| limit.saturating_sub(model_working_set));
    let limits = DecodeLimits::new(
        denoize::metadata_limits_for_available_memory(decode_maximum),
        decode_maximum,
    );
    let mut input_session = denoize::AudioInputSession::open(input)?;
    denoize::ensure_memory_limit(
        model_working_set.saturating_add(denoize::estimate_session_memory_bytes(&input_session)),
        request.max_memory_mb,
        "desktop universal restoration input/model preflight",
    )?;
    let audio = read_audio_from_session_with_limits(&mut input_session, limits)?;
    let working_set = denoize::estimate_universal_restoration_memory_bytes(&audio)
        .saturating_add(model_working_set);
    denoize::ensure_memory_limit(
        working_set,
        request.max_memory_mb,
        "desktop universal restoration working set",
    )?;
    let metadata = if request.preserve_metadata {
        input_session
            .read_metadata_with_limits(desktop_retained_metadata_limits(maximum, working_set))?
    } else {
        None
    };
    let result = denoize::restore_universal_audio(&audio, &session, &config)?;
    if result.mask.runs.len() > MAX_DESKTOP_UNIVERSAL_MASK_RUNS {
        return Err(format!(
            "汎用復元マスクがDesktop表示上限の{MAX_DESKTOP_UNIVERSAL_MASK_RUNS} runsを超えました。CLIの--maskを使用してください"
        ));
    }
    let format = OutputFormat::from_path(output)?;
    let encode = EncodeOptions::default();
    encode.validate_options(format)?;
    format.validate_config(&result.audio, &encode)?;
    denoize::write_audio_transactional(
        output,
        &result.audio,
        encode,
        metadata,
        if request.replace {
            CommitMode::Replace
        } else {
            CommitMode::NoClobber
        },
    )?;
    Ok(DesktopUniversalRestorationResult {
        output: request.output,
        report: result.report,
        mask: result.mask,
    })
}

#[cfg(not(feature = "full"))]
fn run_desktop_universal_restoration(
    _request: UniversalRestorationRequest,
) -> Result<DesktopUniversalRestorationResult, String> {
    Err("このDesktopビルドでは汎用BSRNN復元を利用できません".into())
}

#[tauri::command]
async fn restore_universal_audio_input(
    request: UniversalRestorationRequest,
) -> DesktopResult<DesktopUniversalRestorationResult> {
    Ok(
        tauri::async_runtime::spawn_blocking(move || run_desktop_universal_restoration(request))
            .await
            .map_err(|error| format!("汎用音声復元タスクに失敗しました: {error}"))??,
    )
}

#[cfg_attr(not(feature = "full"), allow(dead_code))]
fn desktop_target_speaker_config(
    request: &TargetSpeakerRequest,
) -> Result<denoize::TargetSpeakerExtractionConfig, String> {
    let config = denoize::TargetSpeakerExtractionConfig {
        minimum_present_probability: request.minimum_present_probability,
        minimum_absent_probability: request.minimum_absent_probability,
        maximum_energy_gain_db: request.maximum_energy_gain_db,
        maximum_peak_gain_db: request.maximum_peak_gain_db,
        maximum_new_clipping_ratio: request.maximum_new_clipping_ratio,
    };
    config.validate()?;
    Ok(config)
}

#[cfg_attr(not(feature = "full"), allow(dead_code))]
fn desktop_target_speaker_preflight(
    request: &TargetSpeakerRequest,
) -> Result<
    (
        denoize::TargetSpeakerExtractionConfig,
        AcceleratorPreference,
        Option<u64>,
    ),
    String,
> {
    for (value, label) in [
        (&request.mixture, "対象話者抽出の混合音声"),
        (&request.enrollment, "対象話者の登録音声"),
        (&request.output, "対象話者音声の保存先"),
        (&request.model_package, "署名付き対象話者モデルパッケージ"),
        (&request.model_package_key, "モデルパッケージ公開鍵"),
        (&request.promotion_evidence, "対象話者promotion evidence"),
        (&request.promotion_evidence_key, "promotion evidence公開鍵"),
    ] {
        if value.trim().is_empty() {
            return Err(format!("{label}を選択してください"));
        }
    }
    let config = desktop_target_speaker_config(request)?;
    let accelerator = AcceleratorPreference::parse(&request.accelerator)
        .ok_or_else(|| format!("不明なアクセラレータです: {}", request.accelerator))?;
    let maximum = checked_desktop_mib(request.max_memory_mb, "プロセスメモリ上限")?;
    require_distinct_execution_paths(&[
        ("混合音声", Path::new(&request.mixture)),
        ("登録音声", Path::new(&request.enrollment)),
        ("出力", Path::new(&request.output)),
        ("モデルパッケージ", Path::new(&request.model_package)),
        ("モデル公開鍵", Path::new(&request.model_package_key)),
        ("promotion evidence", Path::new(&request.promotion_evidence)),
        (
            "promotion evidence公開鍵",
            Path::new(&request.promotion_evidence_key),
        ),
    ])?;
    Ok((config, accelerator, maximum))
}

#[cfg(feature = "full")]
fn run_desktop_target_speaker(
    request: TargetSpeakerRequest,
) -> Result<DesktopTargetSpeakerResult, String> {
    use zeroize::Zeroize as _;

    let (config, accelerator, maximum) = desktop_target_speaker_preflight(&request)?;

    let mixture_path = Path::new(&request.mixture);
    let enrollment_path = Path::new(&request.enrollment);
    let output_path = Path::new(&request.output);
    let package_path = Path::new(&request.model_package);
    let package_key_path = Path::new(&request.model_package_key);
    let evidence_path = Path::new(&request.promotion_evidence);
    let evidence_key_path = Path::new(&request.promotion_evidence_key);
    ensure_output_available(output_path, request.replace)?;

    // Authenticate the complete package, graph contract, numerical vectors,
    // and promotion claim before either biometric audio source is opened.
    let evidence = denoize::SignedTargetSpeakerPromotionEvidence::from_file(evidence_path)?;
    let evidence_key = ReceiptPublicKey::from_file(evidence_key_path)?;
    let package = denoize::RuntimeModelPackage::open(package_path, package_key_path)?;
    let session = denoize::TargetSpeakerSession::prepare(
        package,
        &evidence,
        &evidence_key,
        accelerator,
    )?;
    let model_working_set = session.model_working_set_bytes()?;
    denoize::ensure_memory_limit(
        model_working_set,
        request.max_memory_mb,
        "desktop target-speaker model working set",
    )?;

    let mut mixture_session = denoize::AudioInputSession::open(mixture_path)?;
    let mut enrollment_session = denoize::AudioInputSession::open(enrollment_path)?;
    let session_memory = denoize::estimate_session_memory_bytes(&mixture_session)
        .saturating_add(denoize::estimate_session_memory_bytes(&enrollment_session));
    denoize::ensure_memory_limit(
        model_working_set.saturating_add(session_memory),
        request.max_memory_mb,
        "desktop target-speaker input/model preflight",
    )?;
    let decode_maximum = maximum.map(|limit| {
        limit
            .saturating_sub(model_working_set)
            .saturating_sub(session_memory)
    });
    let mixture = read_audio_from_session_with_limits(
        &mut mixture_session,
        DecodeLimits::new(
            denoize::metadata_limits_for_available_memory(decode_maximum),
            decode_maximum,
        ),
    )?;
    let retained_mixture = denoize::estimate_audio_memory_bytes(&mixture);
    let enrollment_maximum = decode_maximum.map(|limit| limit.saturating_sub(retained_mixture));
    let mut enrollment = read_audio_from_session_with_limits(
        &mut enrollment_session,
        DecodeLimits::new(
            denoize::metadata_limits_for_available_memory(enrollment_maximum),
            enrollment_maximum,
        ),
    )?;
    let working_set = denoize::estimate_target_speaker_memory_bytes(&mixture, &enrollment)
        .saturating_add(model_working_set)
        .saturating_add(session_memory);
    if let Err(error) = denoize::ensure_memory_limit(
        working_set,
        request.max_memory_mb,
        "desktop target-speaker decoded/model working set",
    ) {
        for channel in &mut enrollment.channels {
            channel.zeroize();
        }
        return Err(error);
    }
    let result = session.extract(&mixture, enrollment, &config)?;
    let output = if let Some(audio) = result.audio.as_ref() {
        let format = OutputFormat::from_path(output_path)?;
        let encode = EncodeOptions::default();
        encode.validate_options(format)?;
        format.validate_config(audio, &encode)?;
        let metadata = if request.preserve_metadata {
            mixture_session.read_metadata_with_limits(desktop_retained_metadata_limits(
                maximum,
                working_set,
            ))?
        } else {
            None
        };
        denoize::write_audio_transactional(
            output_path,
            audio,
            encode,
            metadata,
            if request.replace {
                CommitMode::Replace
            } else {
                CommitMode::NoClobber
            },
        )?;
        Some(request.output)
    } else {
        None
    };
    Ok(DesktopTargetSpeakerResult {
        output,
        report: result.report,
    })
}

#[cfg(not(feature = "full"))]
fn run_desktop_target_speaker(
    request: TargetSpeakerRequest,
) -> Result<DesktopTargetSpeakerResult, String> {
    let _ = desktop_target_speaker_preflight(&request)?;
    Err("このDesktopビルドでは対象話者抽出を利用できません".into())
}

#[tauri::command]
async fn extract_target_speaker_audio(
    request: TargetSpeakerRequest,
) -> DesktopResult<DesktopTargetSpeakerResult> {
    Ok(
        tauri::async_runtime::spawn_blocking(move || run_desktop_target_speaker(request))
            .await
            .map_err(|error| format!("対象話者抽出タスクに失敗しました: {error}"))??,
    )
}

#[tauri::command]
async fn plan_process(request: ProcessRequest) -> DesktopResult<ExecutionPlan> {
    Ok(
        tauri::async_runtime::spawn_blocking(move || build_process_execution_plan(&request))
            .await
            .map_err(|error| format!("実行計画タスクに失敗しました: {error}"))??,
    )
}

#[tauri::command]
async fn plan_batch(request: BatchRequest) -> DesktopResult<ExecutionPlan> {
    Ok(
        tauri::async_runtime::spawn_blocking(move || build_batch_execution_plan(&request))
            .await
            .map_err(|error| format!("バッチ実行計画タスクに失敗しました: {error}"))??,
    )
}

struct ProjectOperationGuard;

impl ProjectOperationGuard {
    fn acquire() -> Result<Self, String> {
        PROJECT_OPERATION_RUNNING
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| "another project operation is already running".to_string())?;
        Ok(Self)
    }
}

impl Drop for ProjectOperationGuard {
    fn drop(&mut self) {
        PROJECT_OPERATION_RUNNING.store(false, Ordering::SeqCst);
    }
}

fn canonical_desktop_project_root(raw: &str) -> Result<PathBuf, String> {
    if raw.trim().is_empty() {
        return Err("project root must not be empty".into());
    }
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

fn desktop_project_path(root: &Path, raw: &str) -> Result<PathBuf, String> {
    if raw.trim().is_empty() {
        return Err("project path must not be empty".into());
    }
    let path = PathBuf::from(raw);
    Ok(if path.is_absolute() {
        path
    } else {
        root.join(path)
    })
}

fn contained_desktop_project_input(
    root: &Path,
    raw: &str,
    context: &str,
) -> Result<PathBuf, String> {
    let requested = desktop_project_path(root, raw)?;
    let resolved = std::fs::canonicalize(&requested)
        .map_err(|error| format!("resolve {context} {}: {error}", requested.display()))?;
    if !resolved.starts_with(root) {
        return Err(format!(
            "{context} is outside project root {}",
            root.display()
        ));
    }
    Ok(resolved)
}

fn contained_desktop_project_output(
    root: &Path,
    raw: &str,
    context: &str,
) -> Result<PathBuf, String> {
    let requested = desktop_project_path(root, raw)?;
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
    Ok(parent.join(name))
}

fn prepare_desktop_project_plan(
    request: &ProjectTimelineRequest,
) -> Result<
    (
        PathBuf,
        PathBuf,
        PathBuf,
        denoize::ProjectManifest,
        denoize::ProjectExecutionPlan,
    ),
    String,
> {
    let root = canonical_desktop_project_root(&request.root)?;
    let manifest_path =
        contained_desktop_project_input(&root, &request.manifest, "project manifest")?;
    let output = contained_desktop_project_output(&root, &request.output, "project output")?;
    if output
        .extension()
        .and_then(|extension| extension.to_str())
        .is_none_or(|extension| !extension.eq_ignore_ascii_case("wav"))
    {
        return Err("project timeline output must use the .wav extension".into());
    }
    let manifest = denoize::ProjectManifest::from_file(&manifest_path)?;
    denoize::validate_project_files(&manifest, &root, DecodeLimits::default())?;
    let timeline = if request.timeline.trim().is_empty() {
        manifest
            .timelines
            .first()
            .map(|timeline| timeline.id.as_str())
            .ok_or("project has no timeline")?
    } else {
        request.timeline.as_str()
    };
    manifest.timeline(timeline)?;
    let manifest_reference =
        denoize::project_artifact_reference("manifest", &manifest_path, &root)?;
    let output_locator = denoize::portable_locator(&output, &root)?;
    let plan = denoize::ProjectExecutionPlan::new(
        &manifest,
        timeline,
        manifest_reference,
        output_locator,
        CommitMode::NoClobber,
    )?;
    Ok((root, manifest_path, output, manifest, plan))
}

struct PreparedDesktopProjectReceipt {
    path: PathBuf,
    key: ReceiptSecretKey,
    stage: AtomicOutput,
}

fn prepare_desktop_project_receipt(
    request: &ProjectAssemblyRequest,
    root: &Path,
    manifest: &Path,
    output: &Path,
) -> Result<Option<PreparedDesktopProjectReceipt>, String> {
    validate_receipt_pair(request.receipt.as_deref(), request.receipt_key.as_deref())?;
    let (Some(receipt), Some(key)) = (&request.receipt, &request.receipt_key) else {
        return Ok(None);
    };
    let receipt = contained_desktop_project_output(root, receipt, "project receipt")?;
    let key = PathBuf::from(key);
    require_missing_receipt(&receipt)?;
    require_distinct_execution_paths(&[
        ("project manifest", manifest),
        ("project output", output),
        ("project receipt", &receipt),
        ("project receipt key", &key),
    ])?;
    let secret = ReceiptSecretKey::from_file(&key)?;
    let stage = AtomicOutput::new(&receipt)?;
    Ok(Some(PreparedDesktopProjectReceipt {
        path: receipt,
        key: secret,
        stage,
    }))
}

fn write_desktop_project_receipt_stage(
    receipt: &denoize::SignedProjectExecutionReceipt,
    prepared: &mut PreparedDesktopProjectReceipt,
) -> Result<(), String> {
    let mut bytes = receipt.to_pretty_json()?.into_bytes();
    bytes.push(b'\n');
    prepared
        .stage
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| {
            format!(
                "write staged project receipt {}: {error}",
                prepared.path.display()
            )
        })?;
    prepared.stage.file_mut().sync_data().map_err(|error| {
        format!(
            "sync staged project receipt {}: {error}",
            prepared.path.display()
        )
    })
}

fn project_bundle_payload_limit(
    included: bool,
    limit_mb: Option<usize>,
    label: &str,
) -> Result<u64, String> {
    match (included, limit_mb) {
        (false, None) => Ok(0),
        (false, Some(_)) => Err(format!(
            "project {label} payload limit requires its include option"
        )),
        (true, None) => Err(format!(
            "included project {label} payloads require a positive MiB limit"
        )),
        (true, Some(limit_mb)) => checked_desktop_mib(Some(limit_mb), label)?
            .ok_or_else(|| format!("included project {label} payload limit is missing")),
    }
}

#[tauri::command]
async fn inspect_project_manifest(path: String) -> DesktopResult<denoize::ProjectManifest> {
    Ok(tauri::async_runtime::spawn_blocking(move || {
        if path.trim().is_empty() {
            return Err("project manifest path must not be empty".into());
        }
        denoize::ProjectManifest::from_file(path)
    })
    .await
    .map_err(|error| format!("project manifest inspection task failed: {error}"))??)
}

#[tauri::command]
async fn validate_project_manifest(
    request: ProjectDocumentRequest,
) -> DesktopResult<denoize::ProjectValidationReport> {
    Ok(tauri::async_runtime::spawn_blocking(move || {
        let _guard = ProjectOperationGuard::acquire()?;
        let root = canonical_desktop_project_root(&request.root)?;
        let path = contained_desktop_project_input(&root, &request.manifest, "project manifest")?;
        let manifest = denoize::ProjectManifest::from_file(path)?;
        denoize::validate_project_files(&manifest, root, DecodeLimits::default())
    })
    .await
    .map_err(|error| format!("project validation task failed: {error}"))??)
}

#[tauri::command]
async fn plan_project_timeline(
    request: ProjectTimelineRequest,
) -> DesktopResult<denoize::ProjectExecutionPlan> {
    Ok(tauri::async_runtime::spawn_blocking(move || {
        let _guard = ProjectOperationGuard::acquire()?;
        prepare_desktop_project_plan(&request).map(|(_, _, _, _, plan)| plan)
    })
    .await
    .map_err(|error| format!("project planning task failed: {error}"))??)
}

#[tauri::command]
fn save_project_execution_plan(
    path: String,
    plan: denoize::ProjectExecutionPlan,
) -> DesktopResult<()> {
    if path.trim().is_empty() {
        return Err("project plan destination must not be empty".into());
    }
    denoize::write_project_execution_plan(path, &plan, CommitMode::NoClobber, true)
        .map_err(DesktopError::from)
}

#[tauri::command]
async fn assemble_project_timeline(
    request: ProjectAssemblyRequest,
) -> DesktopResult<denoize::ProjectRenderReport> {
    Ok(tauri::async_runtime::spawn_blocking(move || {
        let _guard = ProjectOperationGuard::acquire()?;
        let timeline_request = ProjectTimelineRequest {
            manifest: request.manifest.clone(),
            root: request.root.clone(),
            timeline: request.timeline.clone(),
            output: request.output.clone(),
        };
        let (root, manifest_path, output, manifest, expected_plan) =
            prepare_desktop_project_plan(&timeline_request)?;
        if request.plan != expected_plan {
            return Err(format!(
                "project assembly no longer matches its reviewed plan: reviewed={} current={}",
                request.plan.digest()?,
                expected_plan.digest()?
            ));
        }
        let prepared_receipt = prepare_desktop_project_receipt(
            &request,
            &root,
            &manifest_path,
            &output,
        )?;
        let report = denoize::assemble_project_timeline(
            &manifest,
            &expected_plan.timeline_id,
            &root,
            &output,
            CommitMode::NoClobber,
            DecodeLimits::default(),
        )?;
        if let Some(mut prepared) = prepared_receipt {
            let receipt = denoize::SignedProjectExecutionReceipt::sign(
                &expected_plan,
                report.output,
                &prepared.key,
            )?;
            write_desktop_project_receipt_stage(&receipt, &mut prepared)?;
            let receipt_path = prepared.path.clone();
            prepared
                .stage
                .commit(CommitMode::NoClobber)
                .map_err(|error| {
                    format!(
                        "project audio was published to {}, but its signed receipt could not be published to {}: {error}",
                        output.display(),
                        receipt_path.display()
                    )
                })?;
        }
        Ok(report)
    })
    .await
    .map_err(|error| format!("project assembly task failed: {error}"))??)
}

#[tauri::command]
async fn create_project_bundle(
    request: ProjectBundleBuildRequest,
) -> DesktopResult<denoize::ProjectBundleInfo> {
    Ok(tauri::async_runtime::spawn_blocking(move || {
        let _guard = ProjectOperationGuard::acquire()?;
        if request.output.trim().is_empty() {
            return Err("project bundle destination must not be empty".into());
        }
        let root = canonical_desktop_project_root(&request.root)?;
        let manifest =
            contained_desktop_project_input(&root, &request.manifest, "project manifest")?;
        let options = denoize::ProjectBundleBuildOptions {
            include_sources: request.include_sources,
            source_payload_limit_bytes: project_bundle_payload_limit(
                request.include_sources,
                request.source_payload_limit_mb,
                "source",
            )?,
            include_models: request.include_models,
            model_payload_limit_bytes: project_bundle_payload_limit(
                request.include_models,
                request.model_payload_limit_mb,
                "model",
            )?,
            commit_mode: CommitMode::NoClobber,
        };
        denoize::build_project_bundle(
            manifest,
            root,
            request.output,
            &options,
            DecodeLimits::default(),
        )
    })
    .await
    .map_err(|error| format!("project bundle creation task failed: {error}"))??)
}

#[tauri::command]
async fn inspect_project_bundle(path: String) -> DesktopResult<denoize::ProjectBundleInfo> {
    Ok(tauri::async_runtime::spawn_blocking(move || {
        let _guard = ProjectOperationGuard::acquire()?;
        if path.trim().is_empty() {
            return Err("project bundle path must not be empty".into());
        }
        denoize::inspect_project_bundle(path)
    })
    .await
    .map_err(|error| format!("project bundle inspection task failed: {error}"))??)
}

#[tauri::command]
async fn import_project_bundle(
    path: String,
    destination: String,
) -> DesktopResult<denoize::ProjectBundleImportReport> {
    Ok(tauri::async_runtime::spawn_blocking(move || {
        let _guard = ProjectOperationGuard::acquire()?;
        if path.trim().is_empty() || destination.trim().is_empty() {
            return Err("project bundle and new destination must not be empty".into());
        }
        denoize::import_project_bundle(path, destination)
    })
    .await
    .map_err(|error| format!("project bundle import task failed: {error}"))??)
}

#[tauri::command]
fn save_execution_plan(path: String, plan: ExecutionPlan) -> DesktopResult<()> {
    denoize::write_execution_plan(path, &plan).map_err(DesktopError::from)
}

struct EvaluationRunGuard;

impl EvaluationRunGuard {
    fn acquire() -> Result<Self, String> {
        EVALUATION_RUNNING
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| "another evaluation is already running".to_string())?;
        Ok(Self)
    }
}

impl Drop for EvaluationRunGuard {
    fn drop(&mut self) {
        EVALUATION_RUNNING.store(false, Ordering::SeqCst);
    }
}

#[tauri::command]
async fn validate_evaluation_corpus(
    request: EvaluationValidationRequest,
) -> DesktopResult<denoize::EvaluationCorpusValidation> {
    Ok(tauri::async_runtime::spawn_blocking(move || {
        if request.manifest.trim().is_empty() || request.corpus_root.trim().is_empty() {
            return Err("evaluation manifest and corpus root must not be empty".into());
        }
        let manifest = denoize::EvaluationManifest::from_file(&request.manifest)?;
        denoize::validate_evaluation_corpus(&manifest, &request.corpus_root)
    })
    .await
    .map_err(|error| format!("evaluation validation task failed: {error}"))??)
}

#[tauri::command]
async fn run_release_evaluation(
    request: EvaluationRunRequest,
) -> DesktopResult<denoize::SignedEvaluationResult> {
    Ok(tauri::async_runtime::spawn_blocking(move || {
        let _guard = EvaluationRunGuard::acquire()?;
        if request.manifest.trim().is_empty()
            || request.corpus_root.trim().is_empty()
            || request.secret_key.trim().is_empty()
            || request.output.trim().is_empty()
        {
            return Err(
                "evaluation manifest, corpus root, secret key, and output must not be empty"
                    .to_string(),
            );
        }
        let manifest = denoize::EvaluationManifest::from_file(&request.manifest)?;
        let key = ReceiptSecretKey::from_file(&request.secret_key)?;
        let result = denoize::run_evaluation(
            &manifest,
            &request.corpus_root,
            &key,
            request.listening_result.as_deref().map(Path::new),
        )?;
        denoize::write_signed_evaluation_result(&request.output, &result)?;
        Ok(result)
    })
    .await
    .map_err(|error| format!("evaluation runner task failed: {error}"))??)
}

#[tauri::command]
async fn verify_evaluation_evidence(
    request: EvaluationVerificationRequest,
) -> DesktopResult<denoize::EvaluationVerificationReport> {
    Ok(tauri::async_runtime::spawn_blocking(move || {
        if request.result.trim().is_empty() || request.public_key.trim().is_empty() {
            return Err("evaluation result and public key must not be empty".into());
        }
        let result = denoize::SignedEvaluationResult::from_file(&request.result)?;
        let key = ReceiptPublicKey::from_file(&request.public_key)?;
        let manifest = request
            .manifest
            .as_deref()
            .map(denoize::EvaluationManifest::from_file)
            .transpose()?;
        denoize::verify_evaluation_result(&result, &key, manifest.as_ref())
    })
    .await
    .map_err(|error| format!("evaluation verification task failed: {error}"))??)
}

#[tauri::command]
async fn compare_evaluation_evidence(
    request: EvaluationComparisonRequest,
) -> DesktopResult<denoize::EvaluationComparisonReport> {
    Ok(tauri::async_runtime::spawn_blocking(move || {
        if request.baseline.trim().is_empty()
            || request.candidate.trim().is_empty()
            || request.baseline_key.trim().is_empty()
            || request.candidate_key.trim().is_empty()
        {
            return Err(
                "evaluation baseline, candidate, and both public keys must not be empty".into(),
            );
        }
        let baseline = denoize::SignedEvaluationResult::from_file(&request.baseline)?;
        let candidate = denoize::SignedEvaluationResult::from_file(&request.candidate)?;
        let baseline_key = ReceiptPublicKey::from_file(&request.baseline_key)?;
        let candidate_key = ReceiptPublicKey::from_file(&request.candidate_key)?;
        denoize::compare_evaluation_results(&baseline, &baseline_key, &candidate, &candidate_key)
    })
    .await
    .map_err(|error| format!("evaluation comparison task failed: {error}"))??)
}

#[tauri::command]
fn generate_receipt_key(secret: String, public: String) -> DesktopResult<String> {
    denoize::write_new_receipt_keypair(secret, public).map_err(DesktopError::from)
}

#[tauri::command]
fn export_receipt_public_key(secret: String, public: String) -> DesktopResult<String> {
    denoize::export_receipt_public_key(secret, public).map_err(DesktopError::from)
}

#[tauri::command]
fn create_receipt_policy(
    path: String,
    public_keys: Vec<String>,
    revoked_key_ids: Vec<String>,
) -> DesktopResult<()> {
    let keys = public_keys
        .iter()
        .map(denoize::ReceiptPublicKey::from_file)
        .collect::<Result<Vec<_>, _>>()?;
    let policy = ReceiptTrustPolicy::new(keys, revoked_key_ids)?;
    denoize::write_receipt_trust_policy(path, &policy).map_err(DesktopError::from)
}

#[tauri::command]
async fn verify_execution_receipt(
    request: ReceiptVerificationRequest,
) -> DesktopResult<ReceiptVerificationReport> {
    Ok(tauri::async_runtime::spawn_blocking(move || {
        let receipt = SignedExecutionReceipt::from_file(&request.receipt)?;
        let output_root = request.output_root.as_deref().map(Path::new);
        match (request.key.as_deref(), request.policy.as_deref()) {
            (Some(key), None) => {
                let key = denoize::ReceiptPublicKey::from_file(key)?;
                receipt.verify_signature(&key)?;
                let plan = request
                    .plan
                    .as_deref()
                    .map(ExecutionPlan::from_file)
                    .transpose()?;
                receipt.verify_with_key(
                    &key,
                    plan.as_ref(),
                    Path::new(&request.receipt),
                    output_root,
                )
            }
            (None, Some(policy)) => {
                let policy = ReceiptTrustPolicy::from_file(policy)?;
                receipt.verify_policy(&policy)?;
                let plan = request
                    .plan
                    .as_deref()
                    .map(ExecutionPlan::from_file)
                    .transpose()?;
                receipt.verify_with_policy(
                    &policy,
                    plan.as_ref(),
                    Path::new(&request.receipt),
                    output_root,
                )
            }
            _ => Err("公開鍵または信頼ポリシーのどちらか一方を指定してください".into()),
        }
    })
    .await
    .map_err(|error| format!("実行証明の検証タスクに失敗しました: {error}"))??)
}

#[tauri::command]
fn start_process(
    app: AppHandle,
    state: State<'_, AppState>,
    request: ProcessRequest,
) -> DesktopResult<u64> {
    start_process_inner(app, &state, request).map_err(DesktopError::from)
}

fn start_process_inner(
    app: AppHandle,
    state: &AppState,
    request: ProcessRequest,
) -> Result<u64, String> {
    validate_request(&request)?;
    let (job_id, control) = register_job(state)?;
    let tracker = match recovery::RecoveryTracker::create(
        &app,
        job_id,
        recovery::RecoveryOperation::File(request.clone()),
    ) {
        Ok(tracker) => tracker,
        Err(error) => {
            unregister_job(state, job_id);
            return Err(error);
        }
    };
    if let Err(error) = control.install_recovery(tracker) {
        unregister_job(state, job_id);
        return Err(error);
    }
    state
        .diagnostics
        .record(diagnostics::DiagnosticCode::FileJobStarted);
    let diagnostic_log = Arc::clone(&state.diagnostics);
    let jobs = Arc::clone(&state.jobs);
    std::thread::spawn(move || {
        let started = Instant::now();
        let result = job_worker::run_isolated(
            &app,
            job_id,
            recovery::RecoveryOperation::File(request),
            &control,
        );
        let terminal = match result {
            Ok(progress) => match progress.status.as_str() {
                "completed" => "completed",
                "cancelled" => "cancelled",
                _ => "failed",
            },
            Err(error) if error == "cancelled" || control.is_cancelled() => {
                emit_progress(
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
                );
                "cancelled"
            }
            Err(error) => {
                emit_progress(
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
                );
                "failed"
            }
        };
        diagnostic_log.record(match terminal {
            "completed" => diagnostics::DiagnosticCode::FileJobCompleted,
            "cancelled" => diagnostics::DiagnosticCode::FileJobCancelled,
            _ => diagnostics::DiagnosticCode::FileJobFailed,
        });
        match control.cleanup_isolated_recovery() {
            Ok(_) => control.finish_recovery(terminal),
            Err(error) => {
                eprintln!("denoize desktop: isolated file stage cleanup failed: {error}")
            }
        }
        if let Ok(mut jobs) = jobs.lock() {
            jobs.remove(&job_id);
        }
    });
    Ok(job_id)
}

fn validate_watch_request(request: &WatchRequest) -> Result<(), String> {
    validate_process_options(&request.options)?;
    if request.options.force {
        return Err("watch-folder automation never replaces an existing output".into());
    }
    if request.input_dir.trim().is_empty()
        || request.output_dir.trim().is_empty()
        || request.receipt_key.trim().is_empty()
        || request.output_format.trim().is_empty()
    {
        return Err("watch input, output, receipt key, and output format must not be empty".into());
    }
    let probe_name = format!("output.{}", request.output_format.trim_start_matches('.'));
    OutputFormat::from_path(Path::new(&probe_name))?
        .validate_encoder(parse_aac_encoder(&request.options.aac_encoder)?)?;
    Ok(())
}

fn desktop_watch_config(request: &WatchRequest, processor_identity: Digest) -> WatchFolderConfig {
    let mut config = WatchFolderConfig::new(
        &request.input_dir,
        &request.output_dir,
        processor_identity.as_bytes(),
    )
    .with_output_extension(request.output_format.trim_start_matches('.'))
    .with_recursive(request.recursive)
    .with_settle_duration(Duration::from_millis(request.settle_millis))
    .with_retry_delays(
        Duration::from_millis(request.retry_initial_millis),
        Duration::from_millis(request.retry_max_millis),
    )
    .with_max_attempts(request.max_attempts)
    .with_max_files(request.max_files);
    if let Some(path) = request.quarantine_dir.as_deref() {
        config = config.with_quarantine_root(path);
    }
    if let Some(path) = request.receipt_dir.as_deref() {
        config = config.with_receipt_root(path);
    }
    if let Some(path) = request.state_path.as_deref() {
        config = config.with_state_path(path);
    }
    config
}

fn update_desktop_watch_identity(hasher: &mut Sha256, label: &str, value: &[u8]) {
    hasher.update((label.len() as u64).to_le_bytes());
    hasher.update(label.as_bytes());
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn desktop_watch_processor_identity(
    template: &DesktopWatchProcessorTemplate,
    public_key: &ReceiptPublicKey,
) -> Result<Digest, String> {
    let mut material = template.options.clone();
    material.force = false;
    material.max_process_memory_mb = None;
    material.max_temporary_mb = None;
    material.max_gpu_memory_mb = None;
    material.max_gpu_jobs = default_max_gpu_jobs();
    let processing = serde_json::to_vec(&(
        env!("CARGO_PKG_VERSION"),
        &template.output_format,
        template.recursive,
        &material,
    ))
    .map_err(|error| format!("serialize watch processing template: {error}"))?;
    let mut hasher = Sha256::new();
    update_desktop_watch_identity(&mut hasher, "domain", b"denoize-watch-processor-v1");
    update_desktop_watch_identity(&mut hasher, "processing-options", &processing);
    update_desktop_watch_identity(
        &mut hasher,
        "receipt-public-key-id",
        public_key.key_id.as_bytes(),
    );
    for (label, path) in [
        ("onnx-model", template.options.onnx_model.as_deref()),
        ("model-package", template.options.model_package.as_deref()),
        (
            "model-package-key",
            template.options.model_package_key.as_deref(),
        ),
    ] {
        update_desktop_watch_identity(
            &mut hasher,
            &format!("{label}-present"),
            &[u8::from(path.is_some())],
        );
        if let Some(path) = path {
            let fingerprint = batch_resume::fingerprint_file(Path::new(path))
                .map_err(|error| format!("fingerprint watch {label} {path}: {error}"))?;
            let mut encoded = [0_u8; 40];
            encoded[..8].copy_from_slice(&fingerprint.len.to_le_bytes());
            encoded[8..].copy_from_slice(fingerprint.digest.as_bytes());
            update_desktop_watch_identity(&mut hasher, label, &encoded);
        }
    }
    Ok(Digest::from_bytes(hasher.finalize().into()))
}

fn desktop_watch_processor_template(request: &WatchRequest) -> DesktopWatchProcessorTemplate {
    DesktopWatchProcessorTemplate {
        output_format: request
            .output_format
            .trim_start_matches('.')
            .to_ascii_lowercase(),
        recursive: request.recursive,
        options: request.options.clone(),
    }
}

fn desktop_watch_path_exists(path: &Path) -> Result<bool, String> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!(
            "inspect watch artifact {}: {error}",
            path.display()
        )),
    }
}

fn recover_desktop_watch_job(
    job: &WatchFolderJob,
    public_key: &ReceiptPublicKey,
) -> Result<bool, String> {
    let output_exists = desktop_watch_path_exists(&job.output_path)?;
    let receipt_exists = desktop_watch_path_exists(&job.receipt_path)?;
    match (output_exists, receipt_exists) {
        (false, false) => return Ok(false),
        (true, false) => {
            return Err(format!(
                "watch output exists without its signed receipt: {}",
                job.output_path.display()
            ));
        }
        (false, true) => {
            return Err(format!(
                "watch receipt exists without its output: {}",
                job.receipt_path.display()
            ));
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
    if item.output.path != denoize::portable_file_locator(&job.output_path)? {
        return Err("watch receipt output locator does not match its scheduled job".into());
    }
    Ok(true)
}

fn classify_desktop_watch_error(error: String) -> WatchProcessError {
    let lower = error.to_ascii_lowercase();
    if lower.contains("cancelled") || error.contains("キャンセル") {
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
        "not a regular file",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
        || ["不明", "不正", "非対応", "壊れ", "切り詰め"]
            .iter()
            .any(|needle| error.contains(needle))
    {
        WatchProcessError::permanent(error)
    } else {
        WatchProcessError::retryable(error)
    }
}

fn run_desktop_watch_job_isolated(
    app: &AppHandle,
    state: &AppState,
    request: ProcessRequest,
) -> Result<(), String> {
    validate_request(&request)?;
    let (job_id, control) = register_watch_job(state)?;
    let operation = recovery::RecoveryOperation::File(request);
    let tracker = match recovery::RecoveryTracker::create(app, job_id, operation.clone()) {
        Ok(tracker) => tracker,
        Err(error) => {
            unregister_job(state, job_id);
            return Err(error);
        }
    };
    if let Err(error) = control.install_recovery(tracker) {
        unregister_job(state, job_id);
        return Err(error);
    }
    state
        .diagnostics
        .record(diagnostics::DiagnosticCode::FileJobStarted);
    let result = job_worker::run_isolated(app, job_id, operation, &control);
    let (terminal, outcome) = match result {
        Ok(progress) if progress.status == "completed" => ("completed", Ok(())),
        Ok(progress) if progress.status == "cancelled" => {
            ("cancelled", Err("cancelled".to_string()))
        }
        Ok(progress) => (
            "failed",
            Err(progress
                .error
                .map(|error| error.technical_detail)
                .unwrap_or_else(|| "isolated watch worker failed".into())),
        ),
        Err(error) if error == "cancelled" || control.is_cancelled() => {
            ("cancelled", Err("cancelled".to_string()))
        }
        Err(error) => ("failed", Err(error)),
    };
    state.diagnostics.record(match terminal {
        "completed" => diagnostics::DiagnosticCode::FileJobCompleted,
        "cancelled" => diagnostics::DiagnosticCode::FileJobCancelled,
        _ => diagnostics::DiagnosticCode::FileJobFailed,
    });
    let cleanup = control.cleanup_isolated_recovery();
    if cleanup.is_ok() {
        control.finish_recovery(terminal);
    }
    unregister_job(state, job_id);
    cleanup?;
    outcome
}

fn process_desktop_watch_job(
    app: &AppHandle,
    state: &AppState,
    job: &WatchFolderJob,
    processor_template: &DesktopWatchProcessorTemplate,
    expected_processor_identity: Digest,
    key_path: &Path,
    expected_key_fingerprint: batch_resume::FileFingerprint,
    public_key: &ReceiptPublicKey,
) -> Result<(), WatchProcessError> {
    let current_key = batch_resume::fingerprint_file(key_path).map_err(|error| {
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
        desktop_watch_processor_identity(processor_template, public_key).map_err(|error| {
            WatchProcessError::deferred(format!(
                "watch processor template is temporarily unavailable: {error}"
            ))
        })?;
    if current_processor_identity != expected_processor_identity {
        return Err(WatchProcessError::deferred(
            "watch processor template changed; restart with a fresh state path to adopt it",
        ));
    }
    match recover_desktop_watch_job(job, public_key) {
        Ok(true) => return Ok(()),
        Ok(false) => {}
        Err(error) => return Err(WatchProcessError::permanent(error)),
    }
    let mut options = processor_template.options.clone();
    options.force = false;
    let request = ProcessRequest {
        input: job.input_path.to_string_lossy().into_owned(),
        output: job.output_path.to_string_lossy().into_owned(),
        expected_input_fingerprint: Some(job.input_fingerprint),
        expected_recipe: None,
        stream: false,
        resume: false,
        stream_frames: DEFAULT_STREAM_BLOCK_FRAMES,
        receipt: Some(job.receipt_path.to_string_lossy().into_owned()),
        receipt_key: Some(key_path.to_string_lossy().into_owned()),
        options,
    };
    run_desktop_watch_job_isolated(app, state, request).map_err(classify_desktop_watch_error)?;
    match recover_desktop_watch_job(job, public_key) {
        Ok(true) => Ok(()),
        Ok(false) => Err(WatchProcessError::permanent(
            "watch processing returned without publishing output and receipt",
        )),
        Err(error) => Err(WatchProcessError::permanent(error)),
    }
}

fn create_desktop_watch_session(request: WatchRequest) -> Result<DesktopWatchSession, String> {
    validate_watch_request(&request)?;
    let key_path = PathBuf::from(&request.receipt_key);
    let key = ReceiptSecretKey::from_file(&key_path)?;
    let public_key = key.public_key()?;
    drop(key);
    let key_path = std::fs::canonicalize(&key_path)
        .map_err(|error| format!("resolve watch receipt key {}: {error}", key_path.display()))?;
    let normalized_input = normalize_batch_path(Path::new(&request.input_dir))?;
    let normalized_output = normalize_batch_path(Path::new(&request.output_dir))?;
    if key_path.starts_with(&normalized_input) || key_path.starts_with(&normalized_output) {
        return Err("watch receipt key must be outside the input and output trees".into());
    }
    let key_fingerprint = batch_resume::fingerprint_file(&key_path)?;
    let processor_template = desktop_watch_processor_template(&request);
    let processor_identity = desktop_watch_processor_identity(&processor_template, &public_key)?;
    let watch = WatchFolder::open(desktop_watch_config(&request, processor_identity))?;
    Ok(DesktopWatchSession {
        watch,
        processor_template,
        processor_identity,
        key_path,
        key_fingerprint,
        public_key,
    })
}

fn cycle_desktop_watch_session(
    app: &AppHandle,
    state: &AppState,
    session: &mut DesktopWatchSession,
) -> Result<WatchCycleReport, String> {
    let DesktopWatchSession {
        watch,
        processor_template,
        processor_identity,
        key_path,
        key_fingerprint,
        public_key,
    } = session;
    watch.cycle(|job| {
        process_desktop_watch_job(
            app,
            state,
            job,
            processor_template,
            *processor_identity,
            key_path,
            *key_fingerprint,
            public_key,
        )
    })
}

fn install_desktop_watch_session(state: &AppState, request: WatchRequest) -> Result<(), String> {
    let mut watch = state
        .watch
        .lock()
        .map_err(|_| "watch-folder state could not be updated")?;
    if watch.is_some() || state.watch_active.load(Ordering::Acquire) {
        return Err("watch-folder automation is already running".into());
    }
    state.watch_active.store(true, Ordering::Release);
    let result = (|| {
        let jobs = state
            .jobs
            .lock()
            .map_err(|_| "job state could not be inspected")?;
        let live = state
            .live
            .lock()
            .map_err(|_| "live state could not be inspected")?;
        if !jobs.is_empty() || live.is_some() {
            return Err("stop the active operation before starting watch-folder automation".into());
        }
        drop(live);
        drop(jobs);
        *watch = Some(create_desktop_watch_session(request)?);
        Ok(())
    })();
    if result.is_err() {
        state.watch_active.store(false, Ordering::Release);
    }
    result
}

fn poll_watch_folder_inner(app: &AppHandle, state: &AppState) -> Result<WatchCycleReport, String> {
    let mut watch = state
        .watch
        .lock()
        .map_err(|_| "watch-folder state could not be updated")?;
    let session = watch
        .as_mut()
        .ok_or("watch-folder automation is not running")?;
    cycle_desktop_watch_session(app, state, session)
}

fn stop_watch_folder_inner(state: &AppState) -> Result<(), String> {
    let mut watch = state
        .watch
        .lock()
        .map_err(|_| "watch-folder state could not be updated")?;
    *watch = None;
    state.watch_active.store(false, Ordering::Release);
    Ok(())
}

fn start_watch_folder_inner(
    app: AppHandle,
    state: AppState,
    request: WatchRequest,
) -> Result<WatchCycleReport, String> {
    install_desktop_watch_session(&state, request)?;
    match poll_watch_folder_inner(&app, &state) {
        Ok(report) => Ok(report),
        Err(error) => {
            let _ = stop_watch_folder_inner(&state);
            Err(error)
        }
    }
}

#[tauri::command]
async fn start_watch_folder(
    app: AppHandle,
    state: State<'_, AppState>,
    request: WatchRequest,
) -> DesktopResult<WatchCycleReport> {
    let state = state.inner().clone();
    Ok(
        tauri::async_runtime::spawn_blocking(move || start_watch_folder_inner(app, state, request))
            .await
            .map_err(|error| format!("watch-folder task failed: {error}"))??,
    )
}

#[tauri::command]
async fn poll_watch_folder(
    app: AppHandle,
    state: State<'_, AppState>,
) -> DesktopResult<WatchCycleReport> {
    let state = state.inner().clone();
    Ok(
        tauri::async_runtime::spawn_blocking(move || poll_watch_folder_inner(&app, &state))
            .await
            .map_err(|error| format!("watch-folder task failed: {error}"))??,
    )
}

#[tauri::command]
async fn stop_watch_folder(state: State<'_, AppState>) -> DesktopResult<()> {
    let state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || stop_watch_folder_inner(&state))
        .await
        .map_err(|error| format!("watch-folder stop task failed: {error}"))??;
    Ok(())
}

#[tauri::command]
fn start_batch(
    app: AppHandle,
    state: State<'_, AppState>,
    request: BatchRequest,
) -> DesktopResult<u64> {
    start_batch_inner(app, &state, request).map_err(DesktopError::from)
}

fn start_batch_inner(
    app: AppHandle,
    state: &AppState,
    request: BatchRequest,
) -> Result<u64, String> {
    let (job_id, control) = register_batch_job(state, &request)?;
    let tracker = match recovery::RecoveryTracker::create(
        &app,
        job_id,
        recovery::RecoveryOperation::Batch(request.clone()),
    ) {
        Ok(tracker) => tracker,
        Err(error) => {
            unregister_job(state, job_id);
            return Err(error);
        }
    };
    if let Err(error) = control.install_recovery(tracker) {
        unregister_job(state, job_id);
        return Err(error);
    }
    state
        .diagnostics
        .record(diagnostics::DiagnosticCode::BatchJobStarted);
    let diagnostic_log = Arc::clone(&state.diagnostics);
    let jobs = Arc::clone(&state.jobs);
    std::thread::spawn(move || {
        let started = Instant::now();
        let output = request.output_dir.clone();
        let result = job_worker::run_isolated(
            &app,
            job_id,
            recovery::RecoveryOperation::Batch(request),
            &control,
        );
        let terminal = match result {
            Ok(progress) => match progress.status.as_str() {
                "completed" => "completed",
                "cancelled" => "cancelled",
                _ => "failed",
            },
            Err(error) if error == "cancelled" || control.is_cancelled() => {
                emit_progress(
                    &app,
                    job_id,
                    "batch",
                    "cancelled",
                    "バッチをキャンセルしました",
                    0,
                    1,
                    started,
                    Some(output),
                    None,
                );
                "cancelled"
            }
            Err(error) => {
                emit_progress(
                    &app,
                    job_id,
                    "batch",
                    "failed",
                    "バッチ処理に失敗しました",
                    0,
                    1,
                    started,
                    Some(output),
                    Some(error),
                );
                "failed"
            }
        };
        diagnostic_log.record(match terminal {
            "completed" => diagnostics::DiagnosticCode::BatchJobCompleted,
            "cancelled" => diagnostics::DiagnosticCode::BatchJobCancelled,
            _ => diagnostics::DiagnosticCode::BatchJobFailed,
        });
        match control.cleanup_isolated_recovery() {
            Ok(_) => control.finish_recovery(terminal),
            Err(error) => {
                eprintln!("denoize desktop: isolated batch stage cleanup failed: {error}")
            }
        }
        if let Ok(mut jobs) = jobs.lock() {
            jobs.remove(&job_id);
        }
    });
    Ok(job_id)
}

fn register_batch_job(
    state: &AppState,
    request: &BatchRequest,
) -> Result<(u64, Arc<JobControl>), String> {
    validate_batch_request(request)?;
    register_job(state)
}

struct PreparedBatchExecution {
    pool: Option<rayon::ThreadPool>,
    session: BatchSession,
    items: Vec<PlannedBatchItem>,
    receipt: Option<DesktopBatchReceiptContext>,
}

fn prepare_batch_execution(
    request: &BatchRequest,
    control: &Arc<JobControl>,
) -> Result<PreparedBatchExecution, String> {
    let mut unplanned_receipt = prepare_batch_receipt(request)?;
    let receipt_stage = unplanned_receipt
        .as_ref()
        .map(|receipt| control.track_stage(&receipt.stage))
        .transpose()?;
    if let Some(guard) = receipt_stage {
        unplanned_receipt.as_mut().unwrap()._recovery_stage = Some(guard);
    }
    let prepared = prepare_batch_request(request)?;
    if let Some(receipt) = &unplanned_receipt {
        let batch_items = prepared
            .iter()
            .map(|prepared| prepared.item.clone())
            .collect::<Vec<_>>();
        validate_batch_receipt_output_paths(&batch_items, receipt)?;
    }
    let pool = if request.options.deterministic {
        None
    } else {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(request.jobs)
                .build()
                .map_err(|error| format!("並列処理を準備できませんでした: {error}"))?,
        )
    };
    let session = BatchSession::acquire(Path::new(&request.output_dir), request.resume)?;
    let items = plan_batch_items(&session, prepared, request.options.force)?;
    let receipt = unplanned_receipt
        .map(|receipt| {
            Ok::<DesktopBatchReceiptContext, String>(DesktopBatchReceiptContext {
                path: receipt.path,
                key: receipt.key,
                stage: receipt.stage,
                plan: build_desktop_batch_plan(request, &items)?,
                _recovery_stage: receipt._recovery_stage,
            })
        })
        .transpose()?;
    session.activate()?;
    Ok(PreparedBatchExecution {
        pool,
        session,
        items,
        receipt,
    })
}

#[derive(Debug, PartialEq, Eq)]
struct BatchTerminalOutcome {
    status: &'static str,
    message: String,
    error: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BatchOutcomeCounts {
    completed: usize,
    skipped: usize,
    failed: usize,
    cancelled: usize,
}

impl BatchOutcomeCounts {
    fn total(self) -> usize {
        self.completed + self.skipped + self.failed + self.cancelled
    }
}

#[derive(Debug, PartialEq, Eq)]
enum BatchItemOutcome {
    Completed(batch_resume::FileFingerprint),
    Skipped(batch_resume::FileFingerprint),
    Failed(String),
    Cancelled,
}

fn batch_item_commit_mode(
    decision: ResumeDecision,
    existing_output: Option<batch_resume::FileFingerprint>,
    cancelled: bool,
) -> Result<CommitMode, BatchItemOutcome> {
    match decision {
        ResumeDecision::Skip { .. } => Err(match existing_output {
            Some(fingerprint) => BatchItemOutcome::Skipped(fingerprint),
            None => BatchItemOutcome::Failed(
                "resume skip is missing its planned output fingerprint".into(),
            ),
        }),
        ResumeDecision::Process { .. } if cancelled => Err(BatchItemOutcome::Cancelled),
        ResumeDecision::Process { commit_mode, .. } => Ok(commit_mode),
    }
}

impl BatchItemOutcome {
    fn status(&self) -> &'static str {
        match self {
            Self::Completed(_) => "completed",
            Self::Skipped(_) => "skipped",
            Self::Failed(_) => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    fn error(&self) -> Option<String> {
        match self {
            Self::Failed(error) => Some(error.clone()),
            _ => None,
        }
    }
}

fn count_batch_outcomes(outcomes: &[BatchItemOutcome]) -> BatchOutcomeCounts {
    let mut counts = BatchOutcomeCounts::default();
    for outcome in outcomes {
        match outcome {
            BatchItemOutcome::Completed(_) => counts.completed += 1,
            BatchItemOutcome::Skipped(_) => counts.skipped += 1,
            BatchItemOutcome::Failed(_) => counts.failed += 1,
            BatchItemOutcome::Cancelled => counts.cancelled += 1,
        }
    }
    counts
}

fn batch_terminal_outcome(counts: BatchOutcomeCounts) -> BatchTerminalOutcome {
    let message = format!(
        "完了 {} · スキップ {} · 失敗 {} · キャンセル {}",
        counts.completed, counts.skipped, counts.failed, counts.cancelled
    );
    if counts.cancelled > 0 {
        BatchTerminalOutcome {
            status: "cancelled",
            message,
            error: None,
        }
    } else if counts.failed == 0 {
        BatchTerminalOutcome {
            status: "completed",
            message,
            error: None,
        }
    } else {
        BatchTerminalOutcome {
            status: "failed",
            message,
            error: Some(format!(
                "{}件のファイルを処理できませんでした",
                counts.failed
            )),
        }
    }
}

fn execute_prepared_batch(
    request: &BatchRequest,
    job_id: u64,
    control: &Arc<JobControl>,
    prepared: PreparedBatchExecution,
    emit: &(dyn Fn(JobProgress) + Sync),
) -> BatchTerminalOutcome {
    let PreparedBatchExecution {
        pool,
        session,
        items,
        mut receipt,
    } = prepared;
    let started = Instant::now();
    let total = items.len();
    let finished = AtomicUsize::new(0);
    let run_item = |batch_item: &PlannedBatchItem| -> BatchItemOutcome {
        let commit_mode = match batch_item_commit_mode(
            batch_item.decision,
            batch_item.existing_output,
            control.is_cancelled(),
        ) {
            Ok(commit_mode) => commit_mode,
            Err(outcome) => return outcome,
        };
        let prepared = &batch_item.prepared;
        let worker_permit = match prepared
            .governor
            .acquire_with_cancel(prepared.resource_request, || control.is_cancelled())
        {
            Ok(permit) => permit,
            Err(_) if control.is_cancelled() => return BatchItemOutcome::Cancelled,
            Err(error) => return BatchItemOutcome::Failed(error),
        };
        let result = stage_batch_output(
            &prepared.item.input,
            &prepared.item.output,
            prepared.item.output_format,
            prepared.encode,
            prepared.metadata_policy,
            &prepared.processing,
            &prepared.backend_session,
            prepared.decode_limits,
            prepared.metadata_limits,
            prepared.resource_request.temporary_bytes(),
            control,
        )
        .and_then(|transaction| {
            let _recovery_stage = control.track_stage(&transaction)?;
            verify_prepared_batch_recipe(prepared)?;
            control
                .commit_fence(|| session.publish(&prepared.expectation, transaction, commit_mode))
        });
        drop(worker_permit);
        match result {
            Ok(fingerprint) => BatchItemOutcome::Completed(fingerprint),
            Err(error) if error == "cancelled" => BatchItemOutcome::Cancelled,
            Err(error) => BatchItemOutcome::Failed(error),
        }
    };
    let process_item = |batch_item: &PlannedBatchItem| {
        let outcome = run_item(batch_item);
        let current = finished.fetch_add(1, Ordering::SeqCst) + 1;
        emit(batch_item_progress(
            job_id,
            outcome.status(),
            &batch_item.prepared.item,
            Some(batch_item.decision.reason()),
            current,
            total,
            started,
            outcome.error(),
            batch_item.prepared.processing.accelerator,
        ));
        outcome
    };
    let outcomes = if let Some(pool) = pool {
        pool.install(|| items.par_iter().map(process_item).collect::<Vec<_>>())
    } else {
        items.iter().map(process_item).collect::<Vec<_>>()
    };
    let counts = count_batch_outcomes(&outcomes);
    debug_assert_eq!(counts.total(), total);
    debug_assert_eq!(finished.load(Ordering::SeqCst), total);
    let receipt_result = if counts.failed == 0 && counts.cancelled == 0 {
        match receipt.take() {
            Some(receipt) => {
                publish_desktop_batch_receipt(receipt, &items, &outcomes, request, control)
            }
            None => Ok(()),
        }
    } else {
        Ok(())
    };
    let mut terminal = batch_terminal_outcome(counts);
    if let Err(receipt_error) = receipt_result {
        if receipt_error == "cancelled" {
            terminal.status = "cancelled";
            terminal.message = format!(
                "{} · 実行証明はキャンセルにより公開されませんでした",
                terminal.message
            );
            terminal.error = None;
        } else {
            terminal.status = "failed";
            terminal.message = format!(
                "{} · 出力は確定しましたが実行証明を公開できませんでした",
                terminal.message
            );
            terminal.error = Some(receipt_error);
        }
    }
    drop(receipt);
    emit(job_progress(
        job_id,
        "batch",
        terminal.status,
        &terminal.message,
        total,
        total,
        started,
        Some(request.output_dir.clone()),
        terminal.error.clone(),
        None,
    ));
    terminal
}

fn plan_batch_items(
    session: &BatchSession,
    prepared: Vec<PreparedBatchItem>,
    force: bool,
) -> Result<Vec<PlannedBatchItem>, String> {
    let mut planned = Vec::with_capacity(prepared.len());
    for prepared in prepared {
        let evidence = session
            .plan_with_evidence(&prepared.expectation, force)
            .map_err(actionable_batch_resume_error)?;
        planned.push(PlannedBatchItem {
            prepared,
            decision: evidence.decision(),
            existing_output: evidence.existing_output(),
        });
    }
    for item in &planned {
        item.prepared.expectation.verify_sources()?;
    }
    Ok(planned)
}

fn actionable_batch_resume_error(error: String) -> String {
    if error.contains("without --force") {
        format!(
            "{error}\n既存出力が古い・未追跡・旧形式・安全でない状態です。「既存を上書き」を有効にして再処理してください"
        )
    } else {
        error
    }
}

fn collect_batch_items(request: &BatchRequest, extension: &str) -> Result<Vec<BatchItem>, String> {
    let output_root = Path::new(&request.output_dir);
    let mut sources = request.inputs.iter().map(PathBuf::from).collect::<Vec<_>>();
    let input_root = request.input_dir.as_deref().map(Path::new);
    if let Some(root) = input_root {
        if !root.is_dir() {
            return Err("入力フォルダが存在しません".into());
        }
        validate_batch_directories(root, output_root)?;
        collect_audio_files(root, request.recursive, &mut sources)?;
        if output_root.starts_with(root) {
            sources.retain(|path| !path.starts_with(output_root));
        }
    }
    sources.sort();
    sources.dedup();
    let mut destinations = HashSet::new();
    let items = sources
        .into_iter()
        .map(|input| {
            if !input.is_file() {
                return Err(format!("入力ファイルが存在しません: {}", input.display()));
            }
            let relative = input_root
                .and_then(|root| input.strip_prefix(root).ok())
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(input.file_name().unwrap_or_default()));
            let mut output = output_root.join(&relative);
            output.set_extension(extension);
            if input.to_str().is_none() || output.to_str().is_none() {
                return Err(format!(
                    "GUIバッチではUTF-8で表現できないパスを処理できません: {}",
                    input.display()
                ));
            }
            if !destinations.insert(output.clone()) {
                return Err(format!(
                    "同じ出力先になるファイルがあります: {}",
                    output.display()
                ));
            }
            let output_relative = output.strip_prefix(output_root).map_err(|error| {
                format!(
                    "バッチ出力 {} が出力フォルダ外です: {error}",
                    output.display()
                )
            })?;
            let output_format = OutputFormat::from_path(&output)?;
            let input_identity = std::fs::canonicalize(&input).map_err(|error| {
                format!("バッチ入力 {} を解決できません: {error}", input.display())
            })?;
            let item_id = batch_resume::item_identity(
                &input_identity,
                &relative,
                output_relative,
                output_format,
            );
            Ok(BatchItem {
                input,
                output,
                output_format,
                item_id,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    validate_batch_destinations(input_root, &items)?;
    Ok(items)
}

fn collect_audio_files(
    dir: &Path,
    recursive: bool,
    files: &mut Vec<PathBuf>,
) -> Result<(), String> {
    for entry in
        std::fs::read_dir(dir).map_err(|error| format!("入力フォルダを読めません: {error}"))?
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
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "wav" | "flac" | "opus" | "ogg" | "mp3" | "m4a" | "aac"
            )
        })
}

fn batch_collision_key_with_case(path: &Path, case_insensitive: bool) -> PathBuf {
    if case_insensitive {
        PathBuf::from(path.to_string_lossy().to_lowercase())
    } else {
        path.to_path_buf()
    }
}

fn batch_collision_key(path: &Path) -> PathBuf {
    batch_collision_key_with_case(path, cfg!(any(windows, target_os = "macos")))
}

fn validate_batch_destinations(
    input_root: Option<&Path>,
    items: &[BatchItem],
) -> Result<(), String> {
    let input_root = input_root.map(normalize_batch_path).transpose()?;
    let input_paths = items
        .iter()
        .map(|item| normalize_batch_path(&item.input).map(|path| batch_collision_key(&path)))
        .collect::<Result<HashSet<_>, _>>()?;
    let mut destinations = Vec::with_capacity(items.len());
    for item in items {
        let resolved = normalize_batch_path(&item.output)?;
        if input_root
            .as_deref()
            .is_some_and(|root| resolved.starts_with(root))
        {
            return Err(format!(
                "バッチ出力 {} が入力フォルダ内へ解決されます。出力先のシンボリックリンクを除くか、別の出力フォルダを選択してください",
                item.output.display()
            ));
        }
        let collision_key = batch_collision_key(&resolved);
        if input_paths.contains(&collision_key) {
            return Err(format!(
                "バッチ出力 {} が入力ファイルを上書きします",
                item.output.display()
            ));
        }
        destinations.push((collision_key, item));
    }
    destinations.sort_by(|left, right| left.0.cmp(&right.0));

    for pair in destinations.windows(2) {
        let (left_path, left) = &pair[0];
        let (right_path, right) = &pair[1];
        if right_path == left_path {
            return Err(format!(
                "複数の入力が同じバッチ出力になります: {} と {} -> {}",
                left.input.display(),
                right.input.display(),
                right.output.display()
            ));
        }
        if right_path.starts_with(left_path) {
            return Err(format!(
                "バッチ出力がファイルとディレクトリとして競合します: {} -> {} / {} -> {}",
                left.input.display(),
                left.output.display(),
                right.input.display(),
                right.output.display()
            ));
        }
    }
    Ok(())
}

fn normalize_batch_path(path: &Path) -> Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("現在のフォルダを解決できません: {error}"))?
            .join(path)
    };
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
                    "バッチパス {} を確認できません: {error}",
                    ancestor.display()
                ));
            }
        }
        let component = ancestor
            .components()
            .next_back()
            .ok_or_else(|| format!("バッチパス {} を解決できません", absolute.display()))?;
        match component {
            std::path::Component::Normal(name) => {
                missing.push(MissingComponent::Normal(name.to_os_string()))
            }
            std::path::Component::ParentDir => missing.push(MissingComponent::Parent),
            std::path::Component::CurDir => {}
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                return Err(format!(
                    "バッチパス {} を解決できません",
                    absolute.display()
                ));
            }
        }
        if !ancestor.pop() {
            return Err(format!(
                "バッチパス {} を解決できません",
                absolute.display()
            ));
        }
    }
    let mut resolved = std::fs::canonicalize(&ancestor)
        .map_err(|error| format!("{} を解決できません: {error}", ancestor.display()))?;
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

fn validate_batch_directories(input_dir: &Path, output_dir: &Path) -> Result<(), String> {
    let input = normalize_batch_path(input_dir)?;
    let output = normalize_batch_path(output_dir)?;
    if input.starts_with(&output) || output.starts_with(&input) {
        return Err(format!(
            "入力フォルダと出力フォルダは重ならない場所を選択してください: {} / {}",
            input_dir.display(),
            output_dir.display()
        ));
    }
    Ok(())
}

fn validate_batch_control_paths(items: &[BatchItem], output_root: &Path) -> Result<(), String> {
    for name in [
        batch_resume::STATE_FILE_NAME,
        batch_resume::LEGACY_DESKTOP_STATE_FILE_NAME,
        batch_resume::LOCK_FILE_NAME,
    ] {
        validate_batch_reserved_path(items, &output_root.join(name), name)?;
    }
    Ok(())
}

fn validate_batch_reserved_path(
    items: &[BatchItem],
    reserved: &Path,
    reserved_name: &str,
) -> Result<(), String> {
    let reserved_key = batch_collision_key(&normalize_batch_path(reserved)?);
    for item in items {
        let output = batch_collision_key(&normalize_batch_path(&item.output)?);
        if output == reserved_key
            || output.starts_with(&reserved_key)
            || reserved_key.starts_with(&output)
        {
            return Err(format!(
                "バッチ出力 {} は予約済みのパス {reserved_name} と競合します",
                item.output.display()
            ));
        }
    }
    Ok(())
}

#[tauri::command]
fn cancel_job(state: State<'_, AppState>, job_id: u64) -> DesktopResult<()> {
    let jobs = state
        .jobs
        .lock()
        .map_err(|_| "ジョブ状態を取得できません")?;
    let control = jobs.get(&job_id).ok_or("実行中のジョブが見つかりません")?;
    control.cancel().map_err(DesktopError::from)
}

#[tauri::command]
#[cfg(feature = "live")]
async fn live_devices() -> DesktopResult<LiveDevices> {
    tauri::async_runtime::spawn_blocking(|| {
        let (inputs, outputs) = denoize::live::device_names()?;
        Ok(LiveDevices { inputs, outputs })
    })
    .await
    .map_err(|error| format!("デバイス一覧の取得に失敗しました: {error}"))?
}

#[tauri::command]
#[cfg(not(feature = "live"))]
async fn live_devices() -> DesktopResult<LiveDevices> {
    Err("このビルドではライブ処理を利用できません".into())
}

#[tauri::command]
#[cfg(feature = "live")]
fn start_live(
    app: AppHandle,
    state: State<'_, AppState>,
    request: LiveRequest,
) -> DesktopResult<()> {
    let backend = validate_live_request(&request)?;
    let backend_options = parsed_backend_options_for(backend, &request.options)?;
    let denoiser = processing_config(&request.options, 48_000)?;
    let governor = desktop_resource_governor(&request.options, 1)?;
    let resilience = denoize::live::LiveResilienceConfig::new()
        .with_target_latency_ms(request.target_latency_ms.unwrap_or(0))
        .with_max_drift_ppm(request.max_drift_ppm.unwrap_or(2_500))
        .with_reconnect_timeout_ms(request.reconnect_timeout_ms.unwrap_or(30_000));
    let prepared = denoize::live::PreparedLiveConfig::new(denoize::live::LiveConfig {
        input_device: request.input_device,
        output_device: request.output_device,
        chunk_ms: request.chunk_ms,
        backend,
        backend_options,
        denoiser,
    })?
    .with_resilience(resilience)?;
    let accelerator = prepared.accelerator();
    let running = Arc::new(AtomicBool::new(true));
    register_live_session(&state, Arc::clone(&running))?;
    let live_state = Arc::clone(&state.live);
    std::thread::spawn(move || {
        let mut last_status = None;
        let result = denoize::live::run_prepared_with_status_and_governor(
            prepared,
            running,
            &governor,
            |status| {
                last_status = Some(status);
                let (connection_state, message) = live_connection_event(status.connection_state);
                let _ = app.emit(
                    "live-status",
                    LiveEvent {
                        status: "running",
                        connection_state,
                        message: message.into(),
                        metrics: status.into(),
                        accelerator: Some(accelerator_result(status.accelerator)),
                        error: None,
                    },
                );
            },
        );
        let (status, message, error) = match result {
            Ok(()) => ("stopped", "ライブ処理を停止しました".into(), None),
            Err(error) => {
                let structured = DesktopError::from(error.clone());
                ("failed", error, Some(structured))
            }
        };
        let _ = app.emit(
            "live-status",
            LiveEvent {
                status,
                connection_state: status,
                message,
                metrics: last_status.map(Into::into).unwrap_or_default(),
                accelerator: Some(accelerator_result(accelerator)),
                error,
            },
        );
        if let Ok(mut live) = live_state.lock() {
            *live = None;
        }
    });
    Ok(())
}

#[tauri::command]
#[cfg(not(feature = "live"))]
fn start_live(
    _app: AppHandle,
    _state: State<'_, AppState>,
    _request: LiveRequest,
) -> DesktopResult<()> {
    Err("このビルドではライブ処理を利用できません".into())
}

#[tauri::command]
#[cfg(feature = "live")]
fn stop_live(state: State<'_, AppState>) -> DesktopResult<()> {
    let live = state
        .live
        .lock()
        .map_err(|_| "ライブ状態を取得できません")?;
    let running = live.as_ref().ok_or("ライブ処理は実行されていません")?;
    running.store(false, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
#[cfg(not(feature = "live"))]
fn stop_live(_state: State<'_, AppState>) -> DesktopResult<()> {
    Err("このビルドではライブ処理を利用できません".into())
}

#[tauri::command]
async fn compare_audio(
    clean: String,
    noisy: String,
    enhanced: String,
) -> DesktopResult<ComparisonOutput> {
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
            metrics: comparison_metric_set(&report),
        })
    })
    .await
    .map_err(|error| format!("比較タスクに失敗しました: {error}"))?
}

#[tauri::command]
fn list_models() -> DesktopResult<ModelLibraryRow> {
    let catalog = denoize::models::active_catalog()?;
    let health = denoize::models::doctor_model_cache_for_catalog(&catalog)?;
    let models = catalog
        .models()
        .iter()
        .map(|model| {
            let path = denoize::models::path_for_catalog_model(model)?;
            let cache_model = health
                .models
                .iter()
                .find(|candidate| candidate.name == model.name())
                .ok_or_else(|| format!("モデル診断結果がありません: {}", model.name()))?;
            let installed = cache_model.status == denoize::models::ModelCacheModelStatus::Healthy;
            let provenance = cache_model.provenance.as_ref();
            Ok(ModelRow {
                name: model.name().to_string(),
                backend: model.backend().to_string(),
                license: model.license().to_string(),
                sample_rate: model.sample_rate(),
                revision: model.revision().to_string(),
                installed,
                path: path.to_string_lossy().into_owned(),
                catalog_sequence: model.catalog_sequence(),
                catalog_sha256: model.catalog_sha256().to_string(),
                catalog_signing_key: model.catalog_signing_key_id().to_string(),
                provenance_source: provenance
                    .as_ref()
                    .map(|provenance| model_installation_source(&provenance.installation_source)),
                installed_at_unix_seconds: provenance
                    .map(|provenance| provenance.installed_at_unix_seconds),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(ModelLibraryRow {
        models,
        health: model_cache_report_row(health),
    })
}

fn model_catalog_origin(origin: &denoize::models::CatalogOrigin) -> String {
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

fn model_installation_source(source: &denoize::models::ModelInstallationSource) -> String {
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

fn model_cache_status(status: denoize::models::ModelCacheModelStatus) -> String {
    match status {
        denoize::models::ModelCacheModelStatus::Missing => "missing",
        denoize::models::ModelCacheModelStatus::Healthy => "healthy",
        denoize::models::ModelCacheModelStatus::Corrupt => "corrupt",
        denoize::models::ModelCacheModelStatus::ProvenanceMissing => "provenance-missing",
        denoize::models::ModelCacheModelStatus::ProvenanceInvalid => "provenance-invalid",
        denoize::models::ModelCacheModelStatus::Unsafe => "unsafe",
        _ => "unknown",
    }
    .into()
}

fn model_cache_issue_kind(kind: denoize::models::ModelCacheIssueKind) -> String {
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
    .into()
}

fn model_cache_issue_row(issue: denoize::models::ModelCacheIssue) -> ModelCacheIssueRow {
    ModelCacheIssueRow {
        kind: model_cache_issue_kind(issue.kind),
        path: issue.path.to_string_lossy().into_owned(),
        model: issue.model,
        detail: issue.detail,
        prunable: issue.prunable,
    }
}

fn model_cache_report_row(report: denoize::models::ModelCacheReport) -> ModelCacheReportRow {
    let clean = report.is_clean();
    ModelCacheReportRow {
        cache_dir: report.cache_dir.to_string_lossy().into_owned(),
        catalog_sequence: report.catalog_sequence,
        catalog_sha256: report.catalog_sha256,
        clean,
        models: report
            .models
            .into_iter()
            .map(|model| ModelCacheHealthRow {
                name: model.name,
                path: model.path.to_string_lossy().into_owned(),
                status: model_cache_status(model.status),
                issues: model
                    .issues
                    .into_iter()
                    .map(model_cache_issue_row)
                    .collect(),
            })
            .collect(),
        issues: report
            .issues
            .into_iter()
            .map(model_cache_issue_row)
            .collect(),
    }
}

fn current_model_catalog_row() -> Result<ModelCatalogRow, String> {
    let status = denoize::models::catalog_status()?;
    Ok(ModelCatalogRow {
        sequence: status.sequence,
        sha256: status.sha256,
        signing_key: status.signing_key_id,
        origin: model_catalog_origin(&status.origin),
        model_count: status.model_count,
        highest_accepted_sequence: status.highest_accepted_sequence,
        cached_path: status.cached_catalog_path.to_string_lossy().into_owned(),
        issued_at_unix_seconds: status.issued_at_unix_seconds,
        expires_at_unix_seconds: status.expires_at_unix_seconds,
        trust_root_version: status.trust_root_version,
        trust_root_sha256: status.trust_root_sha256,
        trust_root_expires_at_unix_seconds: status.trust_root_expires_at_unix_seconds,
        trust_root_highest_observed_unix_seconds: status.trust_root_highest_observed_unix_seconds,
        acquisition_allowed: status.acquisition_allowed,
    })
}

fn offline_bundle_row(info: denoize::models::OfflineBundleInfo) -> OfflineBundleRow {
    OfflineBundleRow {
        format_version: info.format_version,
        bundle_sha256: info.bundle_sha256,
        size_bytes: info.size_bytes,
        catalog_sequence: info.catalog_sequence,
        catalog_sha256: info.catalog_sha256,
        catalog_signing_key_id: info.catalog_signing_key_id,
        catalog_issued_at_unix_seconds: info.catalog_issued_at_unix_seconds,
        catalog_expires_at_unix_seconds: info.catalog_expires_at_unix_seconds,
        trust_root_version: info.trust_root_version,
        trust_root_sha256: info.trust_root_sha256,
        models: info
            .models
            .into_iter()
            .map(|model| OfflineBundleModelRow {
                name: model.name,
                backend: model.backend,
                artifact_filename: model.artifact_filename,
                artifact_sha256: model.artifact_sha256,
                artifact_size_bytes: model.artifact_size_bytes,
                license_filename: model.license_filename,
                license_sha256: model.license_sha256,
                license_size_bytes: model.license_size_bytes,
                provenance_filename: model.provenance_filename,
                provenance_sha256: model.provenance_sha256,
                provenance_size_bytes: model.provenance_size_bytes,
            })
            .collect(),
    }
}

#[tauri::command]
fn model_catalog_status() -> DesktopResult<ModelCatalogRow> {
    current_model_catalog_row().map_err(DesktopError::from)
}

#[tauri::command]
async fn update_model_catalog(
    options: Option<ModelActionOptions>,
) -> DesktopResult<ModelCatalogRow> {
    let options = catalog_action_options(options)?;
    Ok(tauri::async_runtime::spawn_blocking(move || {
        denoize::models::update_catalog(&options)?;
        current_model_catalog_row()
    })
    .await
    .map_err(|error| format!("モデルカタログ更新タスクに失敗しました: {error}"))??)
}

#[tauri::command]
async fn inspect_model_bundle(path: String) -> DesktopResult<OfflineBundleRow> {
    Ok(tauri::async_runtime::spawn_blocking(move || {
        denoize::models::inspect_offline_bundle(path).map(offline_bundle_row)
    })
    .await
    .map_err(|error| format!("オフラインモデルバンドル検証タスクに失敗しました: {error}"))??)
}

#[tauri::command]
async fn inspect_runtime_model_package(
    path: String,
    public_key: String,
) -> DesktopResult<denoize::RuntimeModelPackageInfo> {
    Ok(tauri::async_runtime::spawn_blocking(move || {
        denoize::inspect_runtime_model_package(path, public_key)
    })
    .await
    .map_err(|error| format!("ランタイムモデルパッケージ検証タスクに失敗しました: {error}"))??)
}

#[tauri::command]
async fn import_model_bundle(
    path: String,
    expected_bundle_sha256: String,
) -> DesktopResult<OfflineBundleImportRow> {
    tauri::async_runtime::spawn_blocking(move || {
        let report =
            denoize::models::import_offline_bundle_if_sha256(path, &expected_bundle_sha256)?;
        Ok(OfflineBundleImportRow {
            bundle: offline_bundle_row(report.bundle),
            installed: report
                .installed
                .into_iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect(),
            already_present: report
                .already_present
                .into_iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect(),
        })
    })
    .await
    .map_err(|error| format!("オフラインモデルバンドル導入タスクに失敗しました: {error}"))?
}

#[tauri::command]
async fn recover_model_trust_root() -> DesktopResult<ModelCatalogRow> {
    Ok(tauri::async_runtime::spawn_blocking(|| {
        denoize::models::recover_embedded_trust_root()?;
        current_model_catalog_row()
    })
    .await
    .map_err(|error| format!("モデル信頼ルート復旧タスクに失敗しました: {error}"))??)
}

#[tauri::command]
async fn reset_model_trust_time_floor() -> DesktopResult<ModelCatalogRow> {
    Ok(tauri::async_runtime::spawn_blocking(|| {
        denoize::models::reset_trust_time_floor()?;
        current_model_catalog_row()
    })
    .await
    .map_err(|error| format!("モデル信頼時刻リセットタスクに失敗しました: {error}"))??)
}

#[tauri::command]
async fn model_cache_doctor() -> DesktopResult<ModelCacheReportRow> {
    Ok(tauri::async_runtime::spawn_blocking(|| {
        denoize::models::doctor_model_cache().map(model_cache_report_row)
    })
    .await
    .map_err(|error| format!("モデルキャッシュ診断タスクに失敗しました: {error}"))??)
}

fn application_update_state_root(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_local_data_dir()
        .map(|path| path.join("application-update-v1"))
        .map_err(|error| format!("resolve application update state directory: {error}"))
}

fn application_update_platform() -> Result<&'static str, String> {
    let bundle_type = tauri::utils::platform::bundle_type();
    let legacy_appimage = std::env::var_os("APPIMAGE").is_some();
    application_update_platform_for(
        std::env::consts::OS,
        std::env::consts::ARCH,
        bundle_type,
        legacy_appimage,
    )
}

fn application_update_platform_for(
    os: &str,
    architecture: &str,
    bundle_type: Option<BundleType>,
    legacy_appimage: bool,
) -> Result<&'static str, String> {
    match (os, architecture, bundle_type) {
        ("macos", "aarch64", Some(BundleType::App | BundleType::Dmg) | None) => {
            Ok("darwin-aarch64-app")
        }
        ("macos", "x86_64", Some(BundleType::App | BundleType::Dmg) | None) => {
            Ok("darwin-x86_64-app")
        }
        ("windows", "x86_64", Some(BundleType::Msi)) => Ok("windows-x86_64-msi"),
        ("windows", "x86_64", Some(BundleType::Nsis)) => Ok("windows-x86_64-nsis"),
        ("linux", "x86_64", Some(BundleType::AppImage)) => Ok("linux-x86_64-appimage"),
        ("linux", "x86_64", Some(BundleType::Deb)) => Ok("linux-x86_64-deb"),
        ("linux", "x86_64", None) if legacy_appimage => Ok("linux-x86_64-appimage"),
        (_, _, Some(bundle_type)) => Err(format!(
            "application update bundle type is unsupported on {os}-{architecture}: {bundle_type}"
        )),
        _ => Err(format!(
            "application update requires a packaged Desktop build: {os}-{architecture}"
        )),
    }
}

fn application_update_activation_for_platform(
    platform: &str,
) -> Result<denoize::update::UpdateActivationKind, String> {
    match platform {
        "darwin-aarch64-app" | "darwin-x86_64-app" => {
            Ok(denoize::update::UpdateActivationKind::MacosAppArchive)
        }
        "linux-x86_64-appimage" => Ok(denoize::update::UpdateActivationKind::AppImage),
        "linux-x86_64-deb" => Ok(denoize::update::UpdateActivationKind::DebPackage),
        "windows-x86_64-msi" => Ok(denoize::update::UpdateActivationKind::MsiInstaller),
        "windows-x86_64-nsis" => Ok(denoize::update::UpdateActivationKind::NsisInstaller),
        _ => Err(format!(
            "application update platform has no activation contract: {platform}"
        )),
    }
}

fn activate_application_update_target(
    target: denoize::update::UpdateActivationTarget,
    recovery: bool,
) -> Result<denoize::update::UpdateActivationTarget, String> {
    let expected_platform = application_update_platform()?;
    if target.platform != expected_platform {
        return Err(format!(
            "staged update platform {} does not match {expected_platform}",
            target.platform
        ));
    }
    let expected_activation = application_update_activation_for_platform(expected_platform)?;
    if target.activation != expected_activation {
        return Err(format!(
            "staged update activation {:?} does not match {expected_platform}",
            target.activation
        ));
    }
    match target.activation {
        #[cfg(target_os = "linux")]
        denoize::update::UpdateActivationKind::AppImage => {
            activate_appimage_update(&target)?;
        }
        #[cfg(target_os = "linux")]
        denoize::update::UpdateActivationKind::DebPackage => {
            activate_deb_update(&target, recovery)?;
        }
        #[cfg(target_os = "macos")]
        denoize::update::UpdateActivationKind::MacosAppArchive => {
            activate_macos_update(&target)?;
        }
        #[cfg(windows)]
        denoize::update::UpdateActivationKind::NsisInstaller => {
            activate_windows_nsis_update(&target)?;
        }
        #[cfg(windows)]
        denoize::update::UpdateActivationKind::MsiInstaller => {
            activate_windows_msi_update(&target)?;
        }
        activation => {
            return Err(format!(
                "staged activation {activation:?} is unsupported by this Desktop package"
            ));
        }
    }
    Ok(target)
}

fn begin_application_update_startup(
    state_root: &Path,
) -> Result<denoize::update::UpdateHealthReport, String> {
    let report =
        denoize::update::begin_update_startup_health(state_root, env!("CARGO_PKG_VERSION"), None)?;
    if matches!(
        report.action.as_str(),
        "recovered-last-known-good" | "reactivate-managed-version"
    ) {
        let target = denoize::update::active_update_target(state_root)?;
        activate_application_update_target(target, true)?;
    }
    Ok(report)
}

#[cfg(target_os = "linux")]
fn activate_appimage_update(
    target: &denoize::update::UpdateActivationTarget,
) -> Result<(), String> {
    let current = std::env::var_os("APPIMAGE")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "AppImage activation requires the APPIMAGE environment path".to_string())?;
    let current = std::fs::canonicalize(PathBuf::from(current))
        .map_err(|error| format!("resolve current AppImage: {error}"))?;
    let mut source = File::open(&target.artifact_path)
        .map_err(|error| format!("open authenticated staged AppImage: {error}"))?;
    let source_len = source
        .metadata()
        .map_err(|error| format!("inspect authenticated staged AppImage: {error}"))?
        .len();
    if source_len != target.artifact.len {
        return Err("authenticated staged AppImage length changed before activation".into());
    }
    let mut output = AtomicOutput::new(&current)?;
    let copied = std::io::copy(&mut source, output.file_mut())
        .map_err(|error| format!("stage replacement AppImage: {error}"))?;
    if copied != target.artifact.len {
        return Err("replacement AppImage copy ended at the wrong length".into());
    }
    output.commit(CommitMode::Replace)
}

#[cfg(target_os = "linux")]
fn activate_deb_update(
    target: &denoize::update::UpdateActivationTarget,
    recovery: bool,
) -> Result<(), String> {
    let mut command = std::process::Command::new("pkexec");
    command.arg("dpkg");
    if recovery {
        command.arg("--force-downgrade");
    }
    let status = command
        .arg("-i")
        .arg(&target.artifact_path)
        .status()
        .map_err(|error| format!("start authenticated deb installer with pkexec: {error}"))?;
    if !status.success() {
        return Err(format!(
            "authenticated deb installer exited with status {status}"
        ));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn activate_macos_update(target: &denoize::update::UpdateActivationTarget) -> Result<(), String> {
    let executable = std::env::current_exe()
        .and_then(std::fs::canonicalize)
        .map_err(|error| format!("resolve current application executable: {error}"))?;
    let current_app = executable
        .ancestors()
        .find(|path| {
            path.extension()
                .is_some_and(|extension| extension == std::ffi::OsStr::new("app"))
        })
        .ok_or_else(|| "current executable is not contained in a macOS app bundle".to_string())?;
    let parent = current_app
        .parent()
        .ok_or_else(|| "current macOS app bundle has no parent directory".to_string())?;
    let transaction = tempfile::Builder::new()
        .prefix(".denoize-update-")
        .tempdir_in(parent)
        .map_err(|error| format!("create macOS update transaction directory: {error}"))?;
    let extracted = transaction.path().join("extracted");
    std::fs::create_dir(&extracted)
        .map_err(|error| format!("create macOS update extraction directory: {error}"))?;
    let status = std::process::Command::new("/usr/bin/tar")
        .arg("-xzf")
        .arg(&target.artifact_path)
        .arg("-C")
        .arg(&extracted)
        .status()
        .map_err(|error| format!("extract authenticated macOS app archive: {error}"))?;
    if !status.success() {
        return Err(format!(
            "macOS app archive extraction exited with status {status}"
        ));
    }
    let candidate = extracted.join("denoize.app");
    let candidate_metadata = std::fs::symlink_metadata(&candidate)
        .map_err(|error| format!("inspect extracted macOS app bundle: {error}"))?;
    if !candidate_metadata.is_dir() || candidate_metadata.file_type().is_symlink() {
        return Err("authenticated macOS archive did not contain one regular denoize.app".into());
    }
    let backup = transaction.path().join("last-known-good.app");
    std::fs::rename(current_app, &backup)
        .map_err(|error| format!("stage current macOS app for rollback: {error}"))?;
    if let Err(error) = std::fs::rename(&candidate, current_app) {
        let restore = std::fs::rename(&backup, current_app);
        return Err(match restore {
            Ok(()) => format!("activate authenticated macOS app: {error}; restored current app"),
            Err(restore) => format!(
                "activate authenticated macOS app: {error}; restoring current app failed: {restore}"
            ),
        });
    }
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("sync macOS application directory: {error}"))?;
    Ok(())
}

#[cfg(windows)]
fn activate_windows_nsis_update(
    target: &denoize::update::UpdateActivationTarget,
) -> Result<(), String> {
    std::process::Command::new(&target.artifact_path)
        // Match Tauri's passive NSIS updater contract. No current-process
        // arguments are forwarded, so /ARGS deliberately terminates the list.
        .args(["/P", "/R", "/UPDATE", "/ARGS"])
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("start authenticated NSIS updater: {error}"))
}

#[cfg(windows)]
fn activate_windows_msi_update(
    target: &denoize::update::UpdateActivationTarget,
) -> Result<(), String> {
    std::process::Command::new("msiexec.exe")
        .arg("/i")
        .arg(&target.artifact_path)
        .args(["/passive", "/promptrestart", "AUTOLAUNCHAPP=True"])
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("start authenticated MSI updater: {error}"))
}

#[tauri::command]
async fn application_update_status(
    app: AppHandle,
) -> DesktopResult<denoize::update::UpdateStatusReport> {
    let state_root = application_update_state_root(&app)?;
    Ok(
        tauri::async_runtime::spawn_blocking(move || denoize::update::update_status(state_root))
            .await
            .map_err(|error| format!("application update status task failed: {error}"))??,
    )
}

#[tauri::command]
async fn inspect_application_update_bundle(
    path: String,
) -> DesktopResult<denoize::update::UpdateBundleInfo> {
    Ok(tauri::async_runtime::spawn_blocking(move || {
        denoize::update::inspect_update_bundle(path, None)
    })
    .await
    .map_err(|error| format!("application update bundle inspection task failed: {error}"))??)
}

#[tauri::command]
async fn check_application_update(
    app: AppHandle,
    manifest: String,
    signature: String,
) -> DesktopResult<denoize::update::UpdateCheckReport> {
    let state_root = application_update_state_root(&app)?;
    let platform = application_update_platform()?.to_string();
    Ok(tauri::async_runtime::spawn_blocking(move || {
        let verified = denoize::update::UpdateManifest::from_file(manifest, signature, None)?;
        denoize::update::check_update_manifest(
            &verified,
            state_root,
            "stable",
            &platform,
            env!("CARGO_PKG_VERSION"),
        )
    })
    .await
    .map_err(|error| format!("application update check task failed: {error}"))??)
}

#[tauri::command]
async fn check_application_update_online(
    app: AppHandle,
) -> DesktopResult<denoize::update::UpdateCheckReport> {
    let state_root = application_update_state_root(&app)?;
    let platform = application_update_platform()?.to_string();
    Ok(tauri::async_runtime::spawn_blocking(move || {
        let verified = denoize::update::fetch_update_manifest(
            denoize::update::DEFAULT_UPDATE_MANIFEST_URL,
            denoize::update::DEFAULT_UPDATE_MANIFEST_SIGNATURE_URL,
            None,
        )?;
        denoize::update::check_update_manifest(
            &verified,
            state_root,
            "stable",
            &platform,
            env!("CARGO_PKG_VERSION"),
        )
    })
    .await
    .map_err(|error| format!("online application update check task failed: {error}"))??)
}

#[tauri::command]
async fn download_application_update_bundle(
    path: String,
) -> DesktopResult<denoize::update::UpdateDownloadReport> {
    let platform = application_update_platform()?.to_string();
    Ok(tauri::async_runtime::spawn_blocking(move || {
        let verified = denoize::update::fetch_update_manifest(
            denoize::update::DEFAULT_UPDATE_MANIFEST_URL,
            denoize::update::DEFAULT_UPDATE_MANIFEST_SIGNATURE_URL,
            None,
        )?;
        denoize::update::download_update_bundle(
            &verified,
            &platform,
            env!("CARGO_PKG_VERSION"),
            path,
            None,
        )
    })
    .await
    .map_err(|error| format!("application update download task failed: {error}"))??)
}

#[tauri::command]
async fn dry_run_application_update_bundle(
    app: AppHandle,
    path: String,
) -> DesktopResult<denoize::update::UpdateDryRunReport> {
    let state_root = application_update_state_root(&app)?;
    Ok(tauri::async_runtime::spawn_blocking(move || {
        denoize::update::dry_run_update_bundle(
            path,
            state_root,
            env!("CARGO_PKG_VERSION"),
            None,
            None,
        )
    })
    .await
    .map_err(|error| format!("application update dry-run task failed: {error}"))??)
}

#[tauri::command]
async fn apply_application_update_bundle(
    app: AppHandle,
    state: State<'_, AppState>,
    path: String,
) -> DesktopResult<denoize::update::UpdateApplyReport> {
    let state_root = application_update_state_root(&app)?;
    if let Some(parent) = state_root.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            format!(
                "create application update data directory {}: {error}",
                parent.display()
            )
        })?;
    }
    let result = tauri::async_runtime::spawn_blocking(move || {
        let report = denoize::update::apply_update_bundle(
            path,
            &state_root,
            env!("CARGO_PKG_VERSION"),
            None,
            None,
        )?;
        let target = denoize::update::active_update_target(&state_root)?;
        if let Err(activation_error) = activate_application_update_target(target, false) {
            let recovery = denoize::update::recover_update(
                &state_root,
                "activation-failed",
                None,
            );
            return Err(match recovery {
                Ok(_) => format!(
                    "application update activation failed and state recovered last-known-good: {activation_error}"
                ),
                Err(recovery_error) => format!(
                    "application update activation failed: {activation_error}; state recovery also failed: {recovery_error}"
                ),
            });
        }
        Ok(report)
    })
    .await
    .map_err(|error| format!("application update apply task failed: {error}"))?;
    match result {
        Ok(report) => {
            state
                .diagnostics
                .record(diagnostics::DiagnosticCode::UpdateStaged);
            Ok(report)
        }
        Err(error) => {
            state
                .diagnostics
                .record(diagnostics::DiagnosticCode::UpdateFailed);
            Err(DesktopError::from(error))
        }
    }
}

#[tauri::command]
async fn recover_application_update(
    app: AppHandle,
    state: State<'_, AppState>,
) -> DesktopResult<denoize::update::UpdateHealthReport> {
    let state_root = application_update_state_root(&app)?;
    let result = tauri::async_runtime::spawn_blocking(move || {
        let target = denoize::update::last_known_good_update_target(&state_root)?;
        activate_application_update_target(target, true)?;
        denoize::update::recover_update(state_root, "desktop-manual-recovery", None)
    })
    .await
    .map_err(|error| format!("application update recovery task failed: {error}"))?;
    match result {
        Ok(report) => {
            state
                .diagnostics
                .record(diagnostics::DiagnosticCode::UpdateRecovered);
            Ok(report)
        }
        Err(error) => {
            state
                .diagnostics
                .record(diagnostics::DiagnosticCode::UpdateFailed);
            Err(DesktopError::from(error))
        }
    }
}

#[tauri::command]
async fn confirm_application_update_startup(
    app: AppHandle,
    state: State<'_, AppState>,
) -> DesktopResult<denoize::update::UpdateHealthReport> {
    let startup = state
        .startup_update_health
        .lock()
        .map_err(|_| {
            DesktopError::new(
                "update.failed",
                "application update health state is poisoned",
            )
        })?
        .take();
    let state_root = application_update_state_root(&app)?;
    let report = match startup {
        Some(report) if report.action == "confirm-required" => {
            let token = report.health_token.ok_or_else(|| {
                DesktopError::new(
                    "update.failed",
                    "pending application update has no startup health token",
                )
            })?;
            tauri::async_runtime::spawn_blocking(move || {
                denoize::update::confirm_update_health(
                    state_root,
                    env!("CARGO_PKG_VERSION"),
                    &token,
                    None,
                )
            })
            .await
            .map_err(|error| format!("application update health task failed: {error}"))??
        }
        Some(report) => report,
        None => tauri::async_runtime::spawn_blocking(move || {
            begin_application_update_startup(&state_root)
        })
        .await
        .map_err(|error| format!("application update health task failed: {error}"))??,
    };
    if report.action == "confirmed" {
        state
            .diagnostics
            .record(diagnostics::DiagnosticCode::UpdateConfirmed);
    } else if matches!(
        report.action.as_str(),
        "recovered-last-known-good" | "reactivate-managed-version"
    ) {
        state
            .diagnostics
            .record(diagnostics::DiagnosticCode::UpdateRecovered);
    }
    Ok(report)
}

fn write_automation_json(path: &Path, json: &str) -> Result<(), String> {
    let mut transaction = AtomicOutput::new(path)?;
    std::io::Write::write_all(transaction.file_mut(), json.as_bytes())
        .map_err(|error| format!("自動化JSONを書き込めません: {error}"))?;
    transaction.commit(CommitMode::Replace)
}

fn write_automation_snapshot(path: &Path) -> Result<(), String> {
    let snapshot = denoize::automation::capture_automation_snapshot()?;
    let mut json = snapshot.to_pretty_json()?;
    json.push('\n');
    write_automation_json(path, &json)
}

#[tauri::command]
async fn save_automation_snapshot(path: String) -> DesktopResult<()> {
    Ok(
        tauri::async_runtime::spawn_blocking(move || write_automation_snapshot(Path::new(&path)))
            .await
            .map_err(|error| format!("自動化JSONの書出タスクに失敗しました: {error}"))??,
    )
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DawPluginInfo {
    plugin_id: &'static str,
    version: &'static str,
    format: &'static str,
    latency_policy: &'static str,
    sample_rate: f64,
    latency_frames: u32,
    measured_latency_frames: u32,
    matches_reported: bool,
    latency_millis: f64,
    port_configurations: [&'static str; 2],
    sample_formats: [&'static str; 2],
    realtime_allocations: u32,
}

#[tauri::command]
fn daw_plugin_info(sample_rate: f64) -> DesktopResult<DawPluginInfo> {
    let mut processor = DawRealtimeProcessor::new(sample_rate, 2)?;
    let latency_frames = processor.latency_frames();
    let runtime = processor.prepare_parameters(&denoize::DawParameters {
        bypass: true,
        ..denoize::DawParameters::default()
    })?;
    let measured_latency_frames = (0..=latency_frames.saturating_add(1))
        .find(|&frame| {
            let input = if frame == 0 { 1.0 } else { 0.0 };
            processor.process_frame_f64([input, input], &runtime)[0] != 0.0
        })
        .ok_or_else(|| {
            DesktopError::new(
                "operation.failed",
                "DAW impulse latency measurement produced no output",
            )
        })?;
    let matches_reported = measured_latency_frames == latency_frames;
    if !matches_reported {
        return Err(DesktopError::new(
            "operation.failed",
            format!(
                "DAW measured latency {measured_latency_frames} frames differs from reported latency {latency_frames} frames"
            ),
        ));
    }
    Ok(DawPluginInfo {
        plugin_id: DAW_PLUGIN_ID,
        version: env!("CARGO_PKG_VERSION"),
        format: "CLAP",
        latency_policy: DAW_LATENCY_POLICY,
        sample_rate,
        latency_frames,
        measured_latency_frames,
        matches_reported,
        latency_millis: processor.latency_millis(),
        port_configurations: ["mono", "stereo"],
        sample_formats: ["f32", "f64"],
        realtime_allocations: 0,
    })
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct NeuralDawPluginInfo {
    plugin_id: &'static str,
    version: &'static str,
    format: &'static str,
    backend: &'static str,
    model_id: &'static str,
    model_sha256: &'static str,
    model_installed: bool,
    latency_policy: &'static str,
    sample_rate: f64,
    chunk_frames: u32,
    latency_frames: u32,
    measured_latency_frames: u32,
    matches_reported: bool,
    latency_millis: f64,
    port_configurations: [&'static str; 2],
    reference_port: &'static str,
    sample_formats: [&'static str; 2],
    queue_blocks: usize,
    block_pool: usize,
    overload_fallbacks: [&'static str; 3],
    realtime_allocations: u32,
}

#[tauri::command]
fn neural_daw_plugin_info(sample_rate: f64) -> DesktopResult<NeuralDawPluginInfo> {
    let chunk_frames = neural_daw_chunk_frames(sample_rate)?;
    let latency_frames = neural_daw_latency_frames(sample_rate)?;
    let mut delay = vec![0.0_f64; latency_frames as usize];
    let mut cursor = 0usize;
    let measured_latency_frames = (0..=latency_frames.saturating_add(1))
        .find(|&frame| {
            let input = if frame == 0 { 1.0 } else { 0.0 };
            let delayed = delay[cursor];
            delay[cursor] = input;
            cursor += 1;
            if cursor == delay.len() {
                cursor = 0;
            }
            delayed != 0.0
        })
        .ok_or_else(|| {
            DesktopError::new(
                "operation.failed",
                "neural DAW delayed-dry latency measurement produced no output",
            )
        })?;
    if measured_latency_frames != latency_frames {
        return Err(DesktopError::new(
            "operation.failed",
            format!(
                "neural DAW measured latency {measured_latency_frames} frames differs from reported latency {latency_frames} frames"
            ),
        ));
    }
    let model_installed = denoize::models::MODELS
        .iter()
        .find(|model| {
            model.name == NEURAL_DAW_MODEL_ID
                && model.backend == "gtcrn"
                && model.sha256 == NEURAL_DAW_MODEL_SHA256
        })
        .is_some_and(|model| denoize::models::verify(model).is_ok());
    Ok(NeuralDawPluginInfo {
        plugin_id: NEURAL_DAW_PLUGIN_ID,
        version: env!("CARGO_PKG_VERSION"),
        format: "CLAP",
        backend: "gtcrn",
        model_id: NEURAL_DAW_MODEL_ID,
        model_sha256: NEURAL_DAW_MODEL_SHA256,
        model_installed,
        latency_policy: NEURAL_DAW_LATENCY_POLICY,
        sample_rate,
        chunk_frames,
        latency_frames,
        measured_latency_frames,
        matches_reported: true,
        latency_millis: neural_daw_latency_millis(sample_rate)?,
        port_configurations: ["mono", "stereo"],
        reference_port: "reserved-independent-input",
        sample_formats: ["f32", "f64"],
        queue_blocks: NEURAL_DAW_QUEUE_BLOCKS,
        block_pool: NEURAL_DAW_BLOCK_POOL_SIZE,
        overload_fallbacks: ["delayed-dry", "last-safe-gain", "silence"],
        realtime_allocations: 0,
    })
}

#[tauri::command]
fn daw_factory_preset(factory: String) -> DesktopResult<DawPreset> {
    DawPreset::factory(&factory).ok_or_else(|| {
        DesktopError::new(
            "validation.invalid",
            format!("unknown DAW factory preset {factory}; expected speech, gentle, or music"),
        )
    })
}

#[tauri::command]
fn import_daw_preset(path: String) -> DesktopResult<DawPreset> {
    denoize::read_daw_preset(path).map_err(DesktopError::from)
}

#[tauri::command]
fn export_daw_preset(path: String, preset: DawPreset, replace: bool) -> DesktopResult<DawPreset> {
    preset.validate()?;
    denoize::write_daw_preset(
        path,
        &preset,
        if replace {
            CommitMode::Replace
        } else {
            CommitMode::NoClobber
        },
    )?;
    Ok(preset)
}

#[tauri::command]
fn import_daw_session(path: String) -> DesktopResult<DawSessionState> {
    denoize::read_daw_session(path).map_err(DesktopError::from)
}

#[tauri::command]
fn export_daw_session(
    path: String,
    preset: DawPreset,
    port_configuration: DawPortConfiguration,
    replace: bool,
) -> DesktopResult<DawSessionState> {
    let state = DawSessionState::new(preset, port_configuration)?;
    denoize::write_daw_session(
        path,
        &state,
        if replace {
            CommitMode::Replace
        } else {
            CommitMode::NoClobber
        },
    )?;
    Ok(state)
}

#[tauri::command]
async fn prune_model_cache(dry_run: bool) -> DesktopResult<ModelPruneReportRow> {
    tauri::async_runtime::spawn_blocking(move || {
        let report = denoize::models::prune_model_cache(dry_run)?;
        Ok(ModelPruneReportRow {
            dry_run: report.dry_run,
            would_remove: report
                .would_remove
                .into_iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect(),
            removed: report
                .removed
                .into_iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect(),
            retained: report
                .retained
                .into_iter()
                .map(model_cache_issue_row)
                .collect(),
        })
    })
    .await
    .map_err(|error| format!("モデルキャッシュ整理タスクに失敗しました: {error}"))?
}

#[tauri::command]
fn model_action(
    app: AppHandle,
    state: State<'_, AppState>,
    name: String,
    action: String,
    options: Option<ModelActionOptions>,
) -> DesktopResult<u64> {
    let catalog = denoize::models::active_catalog()?;
    let model = catalog
        .find(&name)
        .cloned()
        .ok_or_else(|| format!("不明なモデル: {name}"))?;
    if !matches!(
        action.as_str(),
        "install" | "update" | "verify" | "repair" | "remove"
    ) {
        return Err(format!("不明な操作: {action}").into());
    }
    let (download_options, source_path) =
        if matches!(action.as_str(), "install" | "update" | "repair") {
            model_action_options(options)?
        } else {
            (ModelDownloadOptions::default(), None)
        };
    if source_path.is_some() && action != "install" {
        return Err("ローカルファイルは導入操作でのみ使用できます".into());
    }
    let (job_id, cancelled) = register_job(&state)?;
    let jobs = Arc::clone(&state.jobs);
    std::thread::spawn(move || {
        emit_model_progress(&app, job_id, &name, "running", "準備しています", 0, None);
        let progress_message = if source_path.is_some() {
            "ローカルモデルを検証しています"
        } else {
            "モデルをダウンロードしています"
        };
        let progress = |downloaded, total| {
            emit_model_progress(
                &app,
                job_id,
                &name,
                "running",
                progress_message,
                downloaded,
                total,
            );
        };
        let result = match action.as_str() {
            "install" => match source_path {
                Some(source) => denoize::models::install_catalog_model_from_file_with_progress(
                    &model,
                    source,
                    || cancelled.is_cancelled(),
                    progress,
                ),
                None => denoize::models::install_catalog_model_with_options_and_progress(
                    &model,
                    &download_options,
                    || cancelled.is_cancelled(),
                    progress,
                ),
            }
            .map(|path| path.display().to_string()),
            "update" => denoize::models::update_catalog_model_with_options_and_progress(
                &model,
                &download_options,
                || cancelled.is_cancelled(),
                progress,
            )
            .map(|path| path.display().to_string()),
            "verify" => {
                denoize::models::verify_catalog_model(&model).map(|path| path.display().to_string())
            }
            "repair" => denoize::models::repair_catalog_model_with_options_and_progress(
                &model,
                &download_options,
                || cancelled.is_cancelled(),
                progress,
            )
            .map(|outcome| match outcome {
                denoize::models::ModelRepairOutcome::AlreadyHealthy => {
                    "正常なため修復は不要です".into()
                }
                denoize::models::ModelRepairOutcome::ProvenanceRebuilt => {
                    "provenanceを再構築しました".into()
                }
                denoize::models::ModelRepairOutcome::ArtifactInstalled => {
                    "モデルを再取得して修復しました".into()
                }
                _ => "モデルを修復しました".into(),
            }),
            "remove" => {
                denoize::models::remove_catalog_model(&model).map(|_| "削除しました".into())
            }
            _ => unreachable!(),
        };
        match result {
            Ok(message) => {
                emit_model_progress(&app, job_id, &name, "completed", &message, 1, Some(1))
            }
            Err(error) if error == "cancelled" => emit_model_progress(
                &app,
                job_id,
                &name,
                "cancelled",
                "モデル操作を中断しました",
                0,
                None,
            ),
            Err(error) => emit_model_progress(&app, job_id, &name, "failed", &error, 0, None),
        }
        if let Ok(mut jobs) = jobs.lock() {
            jobs.remove(&job_id);
        }
    });
    Ok(job_id)
}

fn emit_model_progress(
    app: &AppHandle,
    job_id: u64,
    name: &str,
    status: &'static str,
    message: &str,
    downloaded: u64,
    total: Option<u64>,
) {
    let _ = app.emit(
        "model-progress",
        ModelProgress {
            job_id,
            name: name.into(),
            status,
            message: message.into(),
            downloaded,
            total,
            fraction: total
                .filter(|total| *total > 0)
                .map(|total| downloaded as f64 / total as f64),
            error: (status == "failed").then(|| DesktopError::from(message)),
        },
    );
}

fn emit_preview_progress(
    app: &AppHandle,
    job_id: u64,
    status: &'static str,
    message: impl Into<String>,
    result: Option<preview::PreviewResult>,
    error: Option<String>,
) {
    let _ = app.emit(
        "preview-progress",
        PreviewProgress {
            job_id,
            status,
            message: message.into(),
            result,
            error: error.map(DesktopError::from),
        },
    );
}

#[tauri::command]
fn start_preview(
    app: AppHandle,
    state: State<'_, AppState>,
    request: preview::PreviewRequest,
) -> DesktopResult<u64> {
    preview::validate_preview_request(&request)?;
    let (job_id, control) = register_job(&state)?;
    state
        .diagnostics
        .record(diagnostics::DiagnosticCode::PreviewStarted);
    let diagnostic_log = Arc::clone(&state.diagnostics);
    let jobs = Arc::clone(&state.jobs);
    std::thread::spawn(move || {
        emit_preview_progress(
            &app,
            job_id,
            "running",
            "隔離ワーカーでプレビューを作成しています",
            None,
            None,
        );
        let code = match preview::render_isolated(request, job_id, &control) {
            Ok(result) => {
                emit_preview_progress(
                    &app,
                    job_id,
                    "completed",
                    "プレビューを作成しました",
                    Some(result),
                    None,
                );
                diagnostics::DiagnosticCode::PreviewCompleted
            }
            Err(error) if error == "cancelled" => {
                emit_preview_progress(
                    &app,
                    job_id,
                    "cancelled",
                    "プレビューをキャンセルしました",
                    None,
                    None,
                );
                diagnostics::DiagnosticCode::PreviewCancelled
            }
            Err(error) => {
                emit_preview_progress(
                    &app,
                    job_id,
                    "failed",
                    "プレビューを作成できませんでした",
                    None,
                    Some(error),
                );
                diagnostics::DiagnosticCode::PreviewFailed
            }
        };
        diagnostic_log.record(code);
        if let Ok(mut jobs) = jobs.lock() {
            jobs.remove(&job_id);
        }
    });
    Ok(job_id)
}

#[tauri::command]
async fn release_preview_artifacts(preview_id: String) -> DesktopResult<()> {
    Ok(
        tauri::async_runtime::spawn_blocking(move || preview::release_preview(&preview_id))
            .await
            .map_err(|error| format!("プレビュー消去タスクに失敗しました: {error}"))??,
    )
}

#[tauri::command]
fn load_gui_config(path: String, current: GuiConfig) -> DesktopResult<GuiConfig> {
    let source =
        std::fs::read_to_string(&path).map_err(|error| format!("{path} を読めません: {error}"))?;
    Ok(parse_gui_config(&source, current)?)
}

#[tauri::command]
fn save_gui_config(path: String, config: GuiConfig) -> DesktopResult<()> {
    let mut config = config.normalized()?;
    // `-1` is the CLI-compatible legacy sentinel for explicitly disabling
    // loudness/true-peak processing. Keeping it in exported TOML distinguishes
    // a full disabled config from an omitted field in a partial overlay.
    if config.loudness_lufs.is_none() {
        config.true_peak_dbtp = Some(-1.0);
    }
    let source = toml::to_string_pretty(&config)
        .map_err(|error| format!("設定をTOMLへ変換できません: {error}"))?;
    Ok(std::fs::write(&path, source)
        .map_err(|error| format!("{path} を保存できません: {error}"))?)
}

fn parse_gui_config(source: &str, current: GuiConfig) -> Result<GuiConfig, String> {
    let patch: GuiConfigPatch =
        toml::from_str(source).map_err(|error| format!("TOML設定が不正です: {error}"))?;
    patch.merge(current)
}

#[tauri::command]
fn classify_dropped_paths(paths: Vec<String>) -> DropSelection {
    let mut selection = DropSelection {
        audio_files: Vec::new(),
        directories: Vec::new(),
        ignored: Vec::new(),
    };
    for value in paths {
        let path = Path::new(&value);
        if path.is_dir() {
            selection.directories.push(value);
        } else if path.is_file() && is_audio_path(path) {
            selection.audio_files.push(value);
        } else {
            selection.ignored.push(value);
        }
    }
    selection
}

#[tauri::command]
fn save_text_file(path: String, contents: String) -> DesktopResult<()> {
    Ok(std::fs::write(&path, contents)
        .map_err(|error| format!("{path} を保存できません: {error}"))?)
}

#[tauri::command]
fn list_recoveries(app: AppHandle) -> DesktopResult<Vec<recovery::RecoverySummary>> {
    recovery::RecoveryStore::for_app(&app)?
        .list()
        .map_err(DesktopError::from)
}

#[tauri::command]
fn discard_recovery(
    app: AppHandle,
    state: State<'_, AppState>,
    recovery_id: String,
) -> DesktopResult<usize> {
    let removed = recovery::RecoveryStore::for_app(&app)?.discard(&recovery_id)?;
    state
        .diagnostics
        .record(diagnostics::DiagnosticCode::RecoveryDiscarded);
    Ok(removed)
}

#[tauri::command]
fn retry_recovery(
    app: AppHandle,
    state: State<'_, AppState>,
    recovery_id: String,
) -> DesktopResult<u64> {
    let store = recovery::RecoveryStore::for_app(&app)?;
    let operation = store.operation_for_retry(&recovery_id)?;
    store.cleanup_stages(&recovery_id)?;
    let result = match operation {
        recovery::RecoveryOperation::File(request) => start_process_inner(app, &state, request),
        recovery::RecoveryOperation::Batch(request) => start_batch_inner(app, &state, request),
    };
    if result.is_ok() {
        state
            .diagnostics
            .record(diagnostics::DiagnosticCode::RecoveryRetried);
        if let Err(error) = store.remove_record(&recovery_id) {
            eprintln!("denoize desktop: superseded recovery record cleanup failed: {error}");
        }
    }
    result.map_err(DesktopError::from)
}

#[tauri::command]
fn export_redacted_diagnostics(
    app: AppHandle,
    state: State<'_, AppState>,
    path: String,
) -> DesktopResult<()> {
    let recoveries = recovery::RecoveryStore::for_app(&app)?.list()?;
    let recovery_counts = diagnostics::DiagnosticRecoveryCounts {
        pending: recoveries.iter().filter(|summary| !summary.corrupt).count(),
        corrupt: recoveries.iter().filter(|summary| summary.corrupt).count(),
        staged_artifacts: recoveries
            .iter()
            .map(|summary| summary.staged_artifacts)
            .sum(),
    };
    let active_jobs = state
        .jobs
        .lock()
        .map_err(|_| "診断用のジョブ状態を取得できません")?
        .len();
    let live_session_active = state
        .live
        .lock()
        .map_err(|_| "診断用のlive状態を取得できません")?
        .is_some();
    diagnostics::DiagnosticReport::build(
        active_jobs,
        live_session_active,
        recovery_counts,
        state.diagnostics.snapshot(),
    )
    .write_new(Path::new(&path))
    .map_err(DesktopError::from)
}

fn register_job_impl(
    state: &AppState,
    allow_watch: bool,
) -> Result<(u64, Arc<JobControl>), String> {
    let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    let control = Arc::new(JobControl::default());
    let mut jobs = state
        .jobs
        .lock()
        .map_err(|_| "ジョブ状態を更新できません")?;
    let live = state
        .live
        .lock()
        .map_err(|_| "ライブ状態を取得できません")?;
    if !allow_watch && state.watch_active.load(Ordering::Acquire) {
        return Err(
            "watch-folder automation is running; stop it before starting another job".into(),
        );
    }
    if live.is_some() {
        return Err("ライブ処理を停止してから開始してください".into());
    }
    if !jobs.is_empty() {
        return Err("別の処理が実行中です。完了またはキャンセル後に再試行してください".into());
    }
    jobs.insert(job_id, Arc::clone(&control));
    Ok((job_id, control))
}

fn register_job(state: &AppState) -> Result<(u64, Arc<JobControl>), String> {
    register_job_impl(state, false)
}

fn register_watch_job(state: &AppState) -> Result<(u64, Arc<JobControl>), String> {
    register_job_impl(state, true)
}

fn unregister_job(state: &AppState, job_id: u64) {
    if let Ok(mut jobs) = state.jobs.lock() {
        jobs.remove(&job_id);
    }
}

#[cfg(any(feature = "live", test))]
fn register_live_session(state: &AppState, running: Arc<AtomicBool>) -> Result<(), String> {
    // Use the same jobs-then-live lock order as `register_job`. Holding both
    // guards across validation and insertion makes the desktop's one-active-
    // operation contract atomic instead of allowing simultaneous file/live
    // registration to pass two independent observations.
    let jobs = state
        .jobs
        .lock()
        .map_err(|_| "ジョブ状態を取得できません")?;
    let mut live = state
        .live
        .lock()
        .map_err(|_| "ライブ状態を更新できません")?;
    if state.watch_active.load(Ordering::Acquire) {
        return Err(
            "watch-folder automation is running; stop it before starting live processing".into(),
        );
    }
    if !jobs.is_empty() {
        return Err("ファイル処理の完了後に開始してください".into());
    }
    if live.is_some() {
        return Err("ライブ処理は既に実行中です".into());
    }
    *live = Some(running);
    Ok(())
}

fn checked_desktop_mib(value: Option<usize>, name: &str) -> Result<Option<u64>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value == 0 {
        return Err(format!("{name}は1 MiB以上にしてください"));
    }
    u64::try_from(value)
        .ok()
        .and_then(|value| value.checked_mul(BYTES_PER_MIB))
        .map(Some)
        .ok_or_else(|| format!("{name}が大きすぎます"))
}

fn desktop_resource_governor(
    options: &ProcessOptions,
    cpu_jobs: usize,
) -> Result<ResourceGovernor, String> {
    ResourceGovernor::new(
        ResourceLimits::new()
            .with_max_memory_bytes(checked_desktop_mib(
                options.max_process_memory_mb,
                "プロセスメモリ上限",
            )?)
            .with_max_temporary_bytes(checked_desktop_mib(
                options.max_temporary_mb,
                "一時領域上限",
            )?)
            .with_max_cpu_jobs(Some(cpu_jobs))
            .with_max_gpu_jobs(Some(options.max_gpu_jobs))
            .with_max_gpu_memory_bytes(checked_desktop_mib(
                options.max_gpu_memory_mb,
                "GPUメモリ上限",
            )?),
    )
}

fn desktop_decode_limits(options: &ProcessOptions) -> Result<DecodeLimits, String> {
    let maximum = checked_desktop_mib(options.max_process_memory_mb, "プロセスメモリ上限")?;
    Ok(DecodeLimits::new(
        denoize::metadata_limits_for_available_memory(maximum),
        maximum,
    ))
}

fn desktop_retained_metadata_limits(
    maximum: Option<u64>,
    retained_working_set_bytes: u64,
) -> MetadataLimits {
    denoize::metadata_limits_after_retained_memory(maximum, retained_working_set_bytes)
}

fn validate_process_options(options: &ProcessOptions) -> Result<(), String> {
    if !options.strength.is_finite() || !(0.0..=1.0).contains(&options.strength) {
        return Err("強度は0〜1の有限値で指定してください".into());
    }
    if let Some(target) = options.loudness_lufs {
        if !target.is_finite() || !(MIN_LOUDNESS_LUFS..=MAX_LOUDNESS_LUFS).contains(&target) {
            return Err("ラウドネスは-70〜0 LUFSの有限値で指定してください".into());
        }
    }
    if !options.true_peak_dbtp.is_finite()
        || !(MIN_TRUE_PEAK_DBTP..=MAX_TRUE_PEAK_DBTP).contains(&options.true_peak_dbtp)
    {
        return Err("True Peakは-20〜0 dBTPの有限値で指定してください".into());
    }
    if options.loudness_lufs.is_none() && options.true_peak_dbtp != -1.0 {
        return Err("True Peakはラウドネス正規化と一緒に指定してください".into());
    }
    checked_desktop_mib(options.max_process_memory_mb, "プロセスメモリ上限")?;
    checked_desktop_mib(options.max_temporary_mb, "一時領域上限")?;
    checked_desktop_mib(options.max_gpu_memory_mb, "GPUメモリ上限")?;
    if !(1..=32).contains(&options.max_gpu_jobs) {
        return Err("GPU並列数は1〜32にしてください".into());
    }
    if options.mp3_bitrate_kbps < 32 || options.aac_bitrate_kbps < 32 {
        return Err("ビットレートは32kbps以上にしてください".into());
    }
    checked_aac_bitrate_bps(options.aac_bitrate_kbps)?;
    let backend = configured_backend(&options.backend)?;
    let runtime_package_requested =
        options.model_package.is_some() || options.model_package_key.is_some();
    if runtime_package_requested && backend.is_none() {
        return Err("モデルパッケージには明示的なONNXバックエンドを指定してください".into());
    }
    if runtime_package_requested
        && backend.is_some_and(|selected| {
            service::requires_external_model(selected) && Backend::parse("onnx") != Some(selected)
        })
    {
        return Err("モデルパッケージはONNXバックエンドでのみ利用できます".into());
    }
    if DownmixMode::parse(&options.downmix).is_none() {
        return Err("ダウンミックスは preserve または stereo を指定してください".into());
    }
    parse_aac_encoder(&options.aac_encoder)?;
    if !runtime_package_requested
        && backend.is_some_and(service::requires_external_model)
        && !(1..=MAX_MODEL_SAMPLE_RATE_HZ).contains(&options.onnx_sample_rate)
    {
        return Err(format!(
            "モデルのサンプルレートは1〜{MAX_MODEL_SAMPLE_RATE_HZ}Hzにしてください"
        ));
    }
    let backend_options = match backend {
        Some(backend) => parsed_backend_options_for(backend, options)?,
        None => parsed_backend_options(options)?,
    };
    if let Some(backend) = backend {
        backend_options
            .validate_config(backend)
            .map_err(|error| error.to_string())?;
    }
    processing_config(options, VALIDATION_SAMPLE_RATE_HZ)?;
    Ok(())
}

fn validate_batch_request(request: &BatchRequest) -> Result<String, String> {
    validate_process_options(&request.options)?;
    validate_receipt_pair(request.receipt.as_deref(), request.receipt_key.as_deref())?;
    if !(1..=32).contains(&request.jobs) {
        return Err("並列数は1〜32にしてください".into());
    }
    let extension = request
        .output_format
        .trim_start_matches('.')
        .to_ascii_lowercase();
    let probe = PathBuf::from(format!("output.{extension}"));
    let format = OutputFormat::from_path(&probe)?;
    parsed_encode_options(&request.options)?.validate_options(format)?;
    Ok(extension)
}

fn prepare_batch_request(request: &BatchRequest) -> Result<Vec<PreparedBatchItem>, String> {
    let extension = validate_batch_request(request)?;
    preflight_explicit_backend_resources(&request.options)?;
    if !Path::new(&request.output_dir).is_dir() {
        return Err("出力フォルダが存在しません".into());
    }
    let items = collect_batch_items(request, &extension)?;
    if items.is_empty() {
        return Err("対応する音声ファイルがありません".into());
    }
    validate_batch_control_paths(&items, Path::new(&request.output_dir))?;
    let governor = desktop_resource_governor(&request.options, request.jobs)?;
    preflight_batch_items_with_mode(&request.options, request.resume, items, &governor, false)
}

fn build_batch_execution_plan(request: &BatchRequest) -> Result<ExecutionPlan, String> {
    let extension = validate_batch_request(request)?;
    preflight_explicit_backend_resources_read_only(&request.options)?;
    let output_root = Path::new(&request.output_dir);
    match std::fs::symlink_metadata(output_root) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => return Err("バッチ出力はフォルダでなければなりません".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("バッチ出力フォルダを確認できません: {error}")),
    }
    let items = collect_batch_items(request, &extension)?;
    if items.is_empty() {
        return Err("対応する音声ファイルがありません".into());
    }
    validate_batch_control_paths(&items, output_root)?;
    let governor = desktop_resource_governor(&request.options, request.jobs)?;
    let prepared =
        preflight_batch_items_with_mode(&request.options, request.resume, items, &governor, true)?;
    let expectations = prepared
        .iter()
        .map(|item| item.expectation.clone())
        .collect::<Vec<_>>();
    let decisions = batch_resume::inspect_batch_decisions_with_evidence(
        output_root,
        request.resume,
        &expectations,
        request.options.force,
    )?;
    if decisions.len() != prepared.len() {
        return Err("バッチ実行計画の項目数が一致しません".into());
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
    build_desktop_batch_plan(request, &planned)
}

fn build_desktop_batch_plan(
    request: &BatchRequest,
    planned: &[PlannedBatchItem],
) -> Result<ExecutionPlan, String> {
    let input_root = request.input_dir.as_deref().map(Path::new);
    let output_root = Path::new(&request.output_dir);
    let metadata_policy = if request.options.preserve_metadata {
        MetadataPolicy::Preserve
    } else {
        MetadataPolicy::Drop
    };
    let mut items = Vec::with_capacity(planned.len());
    for planned_item in planned {
        let prepared = &planned_item.prepared;
        prepared.expectation.verify_sources()?;
        let input_locator = match input_root {
            Some(root) => denoize::portable_locator(&prepared.item.input, root)?,
            None => denoize::portable_file_locator(&prepared.item.input)?,
        };
        let output_locator = denoize::portable_locator(&prepared.item.output, output_root)?;
        let input_fingerprint = prepared.expectation.input_fingerprint();
        let item_id = denoize::execution_item_id(
            input_fingerprint,
            &output_locator,
            prepared.expectation.recipe(),
        )?;
        let (publication, action, resources) = match planned_item.decision {
            ResumeDecision::Skip { .. } => (
                "none",
                "skip",
                desktop_planned_resources(ResourceRequest::new()),
            ),
            ResumeDecision::Process { commit_mode, .. } => {
                let session = denoize::estimate_backend_session_request(
                    prepared.processing.backend,
                    &prepared.processing.backend_options,
                    prepared.processing.accelerator,
                )?;
                let request = prepared.resource_request.checked_add(session)?;
                let publication = match commit_mode {
                    CommitMode::Replace => "replace",
                    CommitMode::NoClobber => "no-clobber",
                };
                (publication, "process", desktop_planned_resources(request))
            }
        };
        let model = prepared
            .expectation
            .model()
            .map(|model| {
                Ok::<PlannedArtifact, String>(PlannedArtifact {
                    path: denoize::portable_file_locator(&model.path)?,
                    fingerprint: model.fingerprint,
                })
            })
            .transpose()?;
        items.push(ExecutionPlanItem {
            item_id,
            input: PlannedArtifact {
                path: input_locator,
                fingerprint: input_fingerprint,
            },
            output: PlannedOutput {
                path: output_locator,
                format: desktop_output_format_name(prepared.item.output_format).into(),
                publication: publication.into(),
                action: action.into(),
                reason: planned_item.decision.reason().as_str().into(),
                existing_fingerprint: planned_item.existing_output,
            },
            model,
            recipe: prepared.expectation.recipe(),
            backend: service::backend_name(prepared.processing.backend).into(),
            accelerator: prepared.processing.accelerator.effective().name().into(),
            input_format: desktop_audio_format_name(prepared.input_probe.format).into(),
            input_codec: desktop_audio_codec_name(prepared.input_probe.codec).into(),
            channels: prepared.input_channels as u64,
            frames: prepared.input_frames,
            sample_rate: prepared.sample_rate,
            resources,
        });
    }
    ExecutionPlan::new(
        ExecutionKind::Batch,
        request.options.deterministic,
        desktop_metadata_policy_name(metadata_policy),
        items,
    )
}

fn build_desktop_batch_receipt_items(
    plan: &ExecutionPlan,
    planned: &[PlannedBatchItem],
    outcomes: &[BatchItemOutcome],
    request: &BatchRequest,
) -> Result<Vec<ReceiptItem>, String> {
    if outcomes.len() != planned.len() {
        return Err("バッチ結果件数が実行計画と一致しません".into());
    }
    let input_root = request.input_dir.as_deref().map(Path::new);
    let output_root = Path::new(&request.output_dir);
    let mut items = Vec::with_capacity(planned.len());
    for (planned_item, outcome) in planned.iter().zip(outcomes) {
        let prepared = &planned_item.prepared;
        prepared.expectation.verify_sources()?;
        let output_locator = denoize::portable_locator(&prepared.item.output, output_root)?;
        let item_id = denoize::execution_item_id(
            prepared.expectation.input_fingerprint(),
            &output_locator,
            prepared.expectation.recipe(),
        )?;
        let index = plan
            .items
            .binary_search_by_key(&item_id, |item| item.item_id)
            .map_err(|_| {
                format!(
                    "完了したバッチ項目が実行計画にありません: {}",
                    prepared.item.input.display()
                )
            })?;
        let plan_item = &plan.items[index];
        let input_locator = match input_root {
            Some(root) => denoize::portable_locator(&prepared.item.input, root)?,
            None => denoize::portable_file_locator(&prepared.item.input)?,
        };
        if plan_item.input.path != input_locator || plan_item.output.path != output_locator {
            return Err(format!(
                "完了したバッチ項目のパスが実行計画と一致しません: {}",
                prepared.item.input.display()
            ));
        }
        let (output_fingerprint, receipt_outcome) = match outcome {
            BatchItemOutcome::Completed(fingerprint) => (*fingerprint, "succeeded"),
            BatchItemOutcome::Skipped(fingerprint) => (*fingerprint, "skipped"),
            BatchItemOutcome::Failed(_) | BatchItemOutcome::Cancelled => {
                return Err("失敗またはキャンセルされた項目は実行証明に含められません".into());
            }
        };
        let current = batch_resume::fingerprint_file(&prepared.item.output)?;
        if current != output_fingerprint {
            return Err(format!(
                "公開後にバッチ出力が変更されたため実行証明を作成できません: {}",
                prepared.item.output.display()
            ));
        }
        items.push(ReceiptItem::from_plan_item(
            plan_item,
            output_fingerprint,
            receipt_outcome,
        )?);
    }
    Ok(items)
}

fn publish_desktop_batch_receipt(
    mut receipt: DesktopBatchReceiptContext,
    planned: &[PlannedBatchItem],
    outcomes: &[BatchItemOutcome],
    request: &BatchRequest,
    control: &JobControl,
) -> Result<(), String> {
    let items = build_desktop_batch_receipt_items(&receipt.plan, planned, outcomes, request)?;
    let payload = ExecutionReceiptPayload::new(&receipt.plan, items)?;
    let signed = receipt.key.sign(payload)?;
    write_desktop_receipt_stage(&mut receipt.stage, &receipt.path, &signed)?;
    let receipt_path = receipt.path;
    control
        .commit_fence(|| receipt.stage.commit(CommitMode::NoClobber))
        .map_err(|error| {
            if error == "cancelled" {
                error
            } else {
                format!(
                    "実行証明 {} を公開できませんでした: {error}",
                    receipt_path.display()
                )
            }
        })
}

fn desktop_preflight_decode_admission(
    options: &ProcessOptions,
    governor: &ResourceGovernor,
) -> Result<(DecodeLimits, Option<ResourcePermit>), String> {
    let Some(process_limit) =
        checked_desktop_mib(options.max_process_memory_mb, "プロセスメモリ上限")?
    else {
        return Ok((DecodeLimits::default(), None));
    };
    let available = process_limit
        .checked_sub(governor.usage()?.memory_bytes())
        .ok_or_else(|| "モデルセッションがプロセスメモリ上限を超えています".to_string())?;
    if available < BYTES_PER_MIB {
        return Err("モデル読込後の利用可能メモリが1 MiB未満です".into());
    }
    let permit = governor
        .try_acquire(ResourceRequest::new().with_memory_bytes(available))?
        .ok_or_else(|| "バッチ事前検査用メモリを予約できません".to_string())?;
    Ok((
        DecodeLimits::new(
            denoize::metadata_limits_for_available_memory(Some(available)),
            Some(available),
        ),
        Some(permit),
    ))
}

fn desktop_worker_decode_limit(
    options: &ProcessOptions,
    governor: &ResourceGovernor,
    transient_audio_bytes: u64,
) -> Result<Option<u64>, String> {
    let Some(process_limit) =
        checked_desktop_mib(options.max_process_memory_mb, "プロセスメモリ上限")?
    else {
        return Ok(None);
    };
    let available = process_limit
        .checked_sub(
            governor
                .usage()?
                .memory_bytes()
                .saturating_sub(transient_audio_bytes),
        )
        .ok_or_else(|| "モデルセッションがプロセスメモリ上限を超えています".to_string())?;
    if available < BYTES_PER_MIB {
        return Err("デコード用の利用可能メモリが1 MiB未満です".into());
    }
    Ok(Some(available))
}

fn preflight_batch_items_with_mode(
    options: &ProcessOptions,
    resume_enabled: bool,
    items: Vec<BatchItem>,
    governor: &ResourceGovernor,
    read_only: bool,
) -> Result<Vec<PreparedBatchItem>, String> {
    let encode = parsed_encode_options(options)?;
    let metadata_policy = if options.preserve_metadata {
        MetadataPolicy::Preserve
    } else {
        MetadataPolicy::Drop
    };
    let mut model_fingerprints = HashMap::<(PathBuf, u32), batch_resume::ConsumedModel>::new();
    let mut backend_sessions = Vec::<(
        Backend,
        BackendOptions,
        denoize::AcceleratorSelection,
        Arc<BackendSession>,
        Arc<ResourcePermit>,
    )>::new();
    let mut prepared = Vec::with_capacity(items.len());
    for item in items {
        let (decode_limits, preflight_decode_permit) =
            desktop_preflight_decode_admission(options, governor)?;
        let mut input_session = denoize::AudioInputSession::open(&item.input).map_err(|error| {
            format!("バッチ入力 {} を開けません: {error}", item.input.display())
        })?;
        let input_bytes = input_session.len();
        let input_fingerprint = batch_resume::fingerprint_input_session(&mut input_session)
            .map_err(|error| {
                format!(
                    "バッチ入力 {} の内容を確認できません: {error}",
                    item.input.display()
                )
            })?;
        let input_probe =
            denoize::probe_file_from_session_with_limits(&mut input_session, decode_limits)
                .map_err(|error| {
                    format!(
                        "バッチ入力 {} のcodecを確認できません: {error}",
                        item.input.display()
                    )
                })?;
        if input_probe.audio_tracks != 1 || input_probe.codec == denoize::AudioCodec::Unknown {
            return Err(format!(
                "バッチ入力 {} には対応する音声トラックが1つ必要です",
                item.input.display()
            ));
        }
        let mut audio = read_audio_from_session_with_limits(&mut input_session, decode_limits)
            .map_err(|error| {
                format!(
                    "バッチ入力 {} を事前検査できません: {error}",
                    item.input.display()
                )
            })?;
        drop(preflight_decode_permit);
        let mut decoded_working_set = estimate_audio_working_set_bytes(&audio);
        let mut audio_permit = Some(
            governor
                .try_acquire(ResourceRequest::new().with_memory_bytes(decoded_working_set))?
                .ok_or_else(|| {
                    format!(
                        "バッチ入力 {} がモデルと同時にプロセスメモリ上限へ収まりません",
                        item.input.display()
                    )
                })?,
        );
        item.output_format
            .validate_config(&audio, &encode)
            .map_err(|error| {
                format!(
                    "バッチ出力 {} のcodec設定が不正です: {error}",
                    item.output.display()
                )
            })?;
        let processing = if read_only {
            resolved_processing_options_read_only(options, &audio)
        } else {
            resolved_processing_options(options, &audio)
        }
        .map_err(|error| {
            format!(
                "バッチ入力 {} の処理設定が不正です: {error}",
                item.input.display()
            )
        })?;
        let model = match batch_resume::consumed_model_config(&processing)
            .map_err(|error| format!("使用するモデル設定を確認できません: {error}"))?
        {
            Some(config) => {
                let key = (config.path.clone(), config.sample_rate);
                let model = match model_fingerprints.get(&key) {
                    Some(model) => model.clone(),
                    None => {
                        let model = (if resume_enabled {
                            batch_resume::resumable_consumed_model(&processing)
                        } else {
                            batch_resume::consumed_model(&processing)
                        })
                        .map_err(|error| format!("使用するモデルの内容を確認できません: {error}"))?
                        .ok_or_else(|| {
                            "選択済みバックエンドのモデル識別が失われました".to_string()
                        })?;
                        model_fingerprints.insert(key, model.clone());
                        model
                    }
                };
                Some(model)
            }
            None => None,
        };
        // Bind the prepared graph to the model bytes already captured by the
        // resume recipe. The whole-plan source fence below re-hashes the path
        // after graph preparation and rejects a persistent replacement.
        let (backend_session, backend_session_permit) = if let Some((_, _, _, session, permit)) =
            backend_sessions
                .iter()
                .find(|(backend, backend_options, accelerator, _, _)| {
                    *backend == processing.backend
                        && backend_options == &processing.backend_options
                        && *accelerator == processing.accelerator
                }) {
            (Arc::clone(session), Arc::clone(permit))
        } else {
            let permit = Arc::new(
                governor
                    .try_acquire(denoize::estimate_backend_session_request(
                        processing.backend,
                        &processing.backend_options,
                        processing.accelerator,
                    )?)?
                    .ok_or_else(|| {
                        format!(
                            "バッチ入力 {} のモデルがプロセス資源上限へ収まりません",
                            item.input.display()
                        )
                    })?,
            );
            let session = Arc::new(
                BackendSession::prepare_with_accelerator(
                    processing.backend,
                    processing.backend_options.clone(),
                    processing.accelerator,
                )
                .map_err(|error| {
                    format!(
                        "バッチ入力 {} のバックエンドを準備できません: {error}",
                        item.input.display()
                    )
                })?,
            );
            backend_sessions.push((
                processing.backend,
                processing.backend_options.clone(),
                processing.accelerator,
                Arc::clone(&session),
                Arc::clone(&permit),
            ));
            (session, permit)
        };
        let final_decode_limit =
            desktop_worker_decode_limit(options, governor, decoded_working_set)?;
        let must_redecode = match (decode_limits.max_working_set_bytes, final_decode_limit) {
            (Some(initial), Some(final_limit)) => final_limit < initial,
            (None, Some(_)) => true,
            _ => false,
        };
        if must_redecode {
            drop(audio_permit.take());
            drop(audio);
            let final_limit = final_decode_limit.expect("再デコードには有限の上限が必要です");
            let decode_permit = governor
                .try_acquire(ResourceRequest::new().with_memory_bytes(final_limit))?
                .ok_or_else(|| {
                    format!(
                        "バッチ入力 {} の最終デコード予算を予約できません",
                        item.input.display()
                    )
                })?;
            audio = read_audio_from_session_with_limits(
                &mut input_session,
                DecodeLimits::new(
                    denoize::metadata_limits_for_available_memory(final_decode_limit),
                    final_decode_limit,
                ),
            )
            .map_err(|error| {
                format!(
                    "モデル読込後にバッチ入力 {} をデコードできません: {error}",
                    item.input.display()
                )
            })?;
            drop(decode_permit);
            decoded_working_set = estimate_audio_working_set_bytes(&audio);
            audio_permit = Some(
                governor
                    .try_acquire(ResourceRequest::new().with_memory_bytes(decoded_working_set))?
                    .ok_or_else(|| {
                        format!(
                            "バッチ入力 {} のデコード済み音声を保持できません",
                            item.input.display()
                        )
                    })?,
            );
        }
        let final_decode_limits = DecodeLimits::new(
            denoize::metadata_limits_for_available_memory(final_decode_limit),
            final_decode_limit,
        );
        let metadata_limits =
            desktop_retained_metadata_limits(final_decode_limit, decoded_working_set);
        let metadata_bytes = if metadata_policy == MetadataPolicy::Preserve {
            input_session
                .read_metadata_with_limits(metadata_limits)?
                .as_ref()
                .map(denoize::metadata::Metadata::estimated_memory_bytes)
                .unwrap_or(0)
        } else {
            0
        };
        let resource_request = desktop_worker_request(
            input_bytes,
            &audio,
            metadata_bytes,
            final_decode_limit,
            &processing,
            true,
        )?;
        let recipe = batch_resume::recipe_digest(
            &processing,
            audio.channels(),
            item.output_format,
            encode,
            metadata_policy,
            model
                .as_ref()
                .map(|model| (&model.fingerprint, model.sample_rate)),
        )?;
        let expectation = ResumeExpectation::new(
            item.item_id,
            item.output.clone(),
            item.input.clone(),
            input_fingerprint,
            model,
            recipe,
        );
        drop(audio_permit);
        drop(governor.try_acquire(resource_request)?.ok_or_else(|| {
            format!(
                "バッチ入力 {} を設定された資源上限で実行できません",
                item.input.display()
            )
        })?);
        prepared.push(PreparedBatchItem {
            item,
            input_probe,
            input_channels: audio.channels(),
            input_frames: u64::try_from(audio.frames())
                .map_err(|_| "バッチ入力のフレーム数が大きすぎます".to_string())?,
            sample_rate: audio.sample_rate,
            encode,
            metadata_policy,
            processing,
            backend_session,
            _backend_session_permit: backend_session_permit,
            governor: governor.clone(),
            resource_request,
            decode_limits: final_decode_limits,
            metadata_limits,
            expectation,
        });
    }
    for item in &prepared {
        item.expectation.verify_sources()?;
        drop(
            governor
                .try_acquire(item.resource_request)?
                .ok_or_else(|| {
                    format!(
                        "バッチ入力 {} は全モデル読込後のプロセス資源上限へ収まりません",
                        item.item.input.display()
                    )
                })?,
        );
    }
    Ok(prepared)
}

#[cfg(feature = "live")]
fn validate_live_request(request: &LiveRequest) -> Result<Backend, String> {
    if !(10..=2_000).contains(&request.chunk_ms) {
        return Err("チャンク長は10〜2000msにしてください".into());
    }
    if request
        .target_latency_ms
        .is_some_and(|value| value != 0 && !(20..=5_000).contains(&value))
    {
        return Err("目標レイテンシは0（自動）または20〜5000msにしてください".into());
    }
    if request.max_drift_ppm.is_some_and(|value| value > 10_000) {
        return Err("ドリフト補正は0〜10000ppmにしてください".into());
    }
    if request
        .reconnect_timeout_ms
        .is_some_and(|value| value > 300_000)
    {
        return Err("再接続時間は0〜300000msにしてください".into());
    }
    validate_process_options(&request.options)?;
    let backend = if request.backend == "auto" {
        service::select_live_backend()
    } else {
        Backend::parse(&request.backend)
            .ok_or_else(|| format!("利用できないバックエンドです: {}", request.backend))?
    };
    if !denoize::live::backend_is_live_capable(backend) {
        return Err(format!(
            "ライブ処理に対応していないバックエンドです: {}",
            service::backend_name(backend)
        ));
    }
    parsed_backend_options_for(backend, &request.options)?
        .validate_config(backend)
        .map_err(|error| error.to_string())?;
    Ok(backend)
}

#[cfg(not(feature = "live"))]
#[allow(dead_code)]
fn validate_live_request(_request: &LiveRequest) -> Result<Backend, String> {
    Err("このビルドではライブ処理を利用できません".into())
}

fn parse_aac_encoder(value: &str) -> Result<AacEncoder, String> {
    match value {
        "oxide" => Ok(AacEncoder::Oxide),
        "fdk" => Ok(AacEncoder::Fdk),
        other => Err(format!("不明なAACエンコーダー: {other}")),
    }
}

fn checked_aac_bitrate_bps(bitrate_kbps: u32) -> Result<u32, String> {
    bitrate_kbps
        .checked_mul(1_000)
        .ok_or_else(|| "AACビットレートが大きすぎます".to_string())
}

fn parsed_encode_options(options: &ProcessOptions) -> Result<EncodeOptions, String> {
    Ok(EncodeOptions {
        mp3_bitrate_kbps: options.mp3_bitrate_kbps,
        m4a_bitrate_bps: checked_aac_bitrate_bps(options.aac_bitrate_kbps)?,
        aac_encoder: parse_aac_encoder(&options.aac_encoder)?,
        downmix: DownmixMode::parse(&options.downmix).ok_or_else(|| {
            "ダウンミックスは preserve または stereo を指定してください".to_string()
        })?,
    })
}

fn parsed_backend_options(options: &ProcessOptions) -> Result<BackendOptions, String> {
    let package_paths = match (&options.model_package, &options.model_package_key) {
        (Some(package), Some(key)) => Some((package, key)),
        (None, None) => None,
        _ => return Err("モデルパッケージと信頼済み公開鍵を両方指定してください".into()),
    };
    if package_paths.is_some() && options.onnx_model.is_some() {
        return Err("モデルパッケージと生のONNXモデルは同時に指定できません".into());
    }
    let runtime_package = package_paths
        .map(|(package, key)| denoize::RuntimeModelPackage::open(package, key))
        .transpose()?;
    let mut parsed = BackendOptions {
        onnx: options.onnx_model.as_ref().map(|path| OnnxModelConfig {
            path: path.into(),
            sample_rate: options.onnx_sample_rate,
        }),
        runtime_package: None,
        channel_mode: ChannelMode::parse(&options.channel_mode)
            .ok_or_else(|| format!("不明なチャンネルモード: {}", options.channel_mode))?,
        sgmse_profile: SgmseProfile::parse(&options.sgmse_profile)
            .ok_or_else(|| format!("不明なSGMSEプロファイル: {}", options.sgmse_profile))?,
        accelerator: AcceleratorPreference::parse(&options.accelerator)
            .ok_or_else(|| format!("不明なアクセラレータ: {}", options.accelerator))?,
        deterministic: options.deterministic,
        seed: options.seed,
    };
    if let Some(package) = runtime_package {
        parsed = parsed.with_runtime_model_package(package);
    }
    Ok(parsed)
}

fn configured_backend(value: &str) -> Result<Option<Backend>, String> {
    if value == "auto" {
        Ok(None)
    } else {
        Backend::parse(value)
            .map(Some)
            .ok_or_else(|| format!("このビルドでは利用できないバックエンドです: {value}"))
    }
}

fn parsed_backend_options_for(
    backend: Backend,
    options: &ProcessOptions,
) -> Result<BackendOptions, String> {
    if !service::requires_external_model(backend) {
        let mut without_model = options.clone();
        without_model.onnx_model = None;
        without_model.model_package = None;
        without_model.model_package_key = None;
        return parsed_backend_options(&without_model);
    }
    parsed_backend_options(options)
}

fn resolve_gui_backend_options(
    backend: Backend,
    options: &ProcessOptions,
) -> Result<BackendOptions, String> {
    service::resolve_backend_options(backend, parsed_backend_options_for(backend, options)?)
}

fn resolve_gui_backend_options_read_only(
    backend: Backend,
    options: &ProcessOptions,
) -> Result<BackendOptions, String> {
    service::resolve_backend_options_read_only(
        backend,
        parsed_backend_options_for(backend, options)?,
    )
}

fn preflight_explicit_backend_resources(options: &ProcessOptions) -> Result<(), String> {
    preflight_explicit_backend_resources_with_mode(options, false)
}

fn preflight_explicit_backend_resources_read_only(options: &ProcessOptions) -> Result<(), String> {
    preflight_explicit_backend_resources_with_mode(options, true)
}

fn preflight_explicit_backend_resources_with_mode(
    options: &ProcessOptions,
    read_only: bool,
) -> Result<(), String> {
    if let Some(backend) = configured_backend(&options.backend)? {
        let parsed = parsed_backend_options_for(backend, options)?;
        let backend_options = if read_only {
            service::resolve_backend_options_read_only(backend, parsed)?
        } else {
            service::resolve_backend_options(backend, parsed)?
        };
        denoize::select_accelerator_for_options(backend, &backend_options)?;
    }
    Ok(())
}

fn ensure_output_available(path: &Path, force: bool) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if force && (metadata.is_file() || metadata.file_type().is_symlink()) => {
            Ok(())
        }
        Ok(_) if force => Err(format!(
            "出力先は置換可能なファイルまたはシンボリックリンクではありません: {}",
            path.display()
        )),
        Ok(_) => Err("出力ファイルが既に存在します。「上書きを許可」を有効にしてください".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("出力先を確認できません: {error}")),
    }
}

fn desktop_planned_publication(
    path: &Path,
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
            "出力先は置換可能なファイルまたはシンボリックリンクではありません: {}",
            path.display()
        )),
        Ok(_) => Err("出力ファイルが既に存在します。「上書きを許可」を有効にしてください".into()),
        Err(error) => Err(format!("出力先を確認できません: {error}")),
    }
}

fn validate_receipt_pair(receipt: Option<&str>, receipt_key: Option<&str>) -> Result<(), String> {
    match (receipt, receipt_key) {
        (None, None) => return Ok(()),
        (Some(receipt), Some(key)) if !receipt.trim().is_empty() && !key.trim().is_empty() => {}
        (Some(_), Some(_)) => return Err("証明書と署名鍵のパスを空にはできません".into()),
        _ => return Err("証明書と署名鍵は両方を指定してください".into()),
    }
    Ok(())
}

fn require_missing_receipt(path: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Ok(_) => Err(format!(
            "実行証明が既に存在します（置換しません）: {}",
            path.display()
        )),
        Err(error) => Err(format!("実行証明の出力先を確認できません: {error}")),
    }
}

fn require_distinct_execution_paths(paths: &[(&str, &Path)]) -> Result<(), String> {
    let mut normalized = Vec::with_capacity(paths.len());
    for (label, path) in paths {
        normalized.push((*label, normalize_batch_path(path)?));
    }
    for left in 0..normalized.len() {
        for right in left + 1..normalized.len() {
            if batch_collision_key(&normalized[left].1) == batch_collision_key(&normalized[right].1)
            {
                return Err(format!(
                    "{}と{}は別のパスにしてください: {}",
                    normalized[left].0,
                    normalized[right].0,
                    paths[left].1.display()
                ));
            }
        }
    }
    Ok(())
}

fn prepare_process_receipt(
    request: &ProcessRequest,
) -> Result<Option<DesktopReceiptContext>, String> {
    let (Some(receipt), Some(key_path)) = (&request.receipt, &request.receipt_key) else {
        return Ok(None);
    };
    let receipt = PathBuf::from(receipt);
    let key_path = PathBuf::from(key_path);
    require_missing_receipt(&receipt)?;
    require_distinct_execution_paths(&[
        ("入力", Path::new(&request.input)),
        ("出力", Path::new(&request.output)),
        ("実行証明", &receipt),
        ("署名鍵", &key_path),
    ])?;
    let key = ReceiptSecretKey::from_file(&key_path)?;
    let (publication, reason) = if request.stream && request.resume {
        ("pending", "pending")
    } else {
        desktop_planned_publication(Path::new(&request.output), request.options.force)?
    };
    let stage = AtomicOutput::new(&receipt)?;
    Ok(Some(DesktopReceiptContext {
        path: receipt,
        key,
        stage,
        publication,
        reason,
        _recovery_stage: None,
    }))
}

fn prepare_batch_receipt(
    request: &BatchRequest,
) -> Result<Option<UnplannedDesktopBatchReceipt>, String> {
    validate_receipt_pair(request.receipt.as_deref(), request.receipt_key.as_deref())?;
    let (Some(receipt), Some(key_path)) = (&request.receipt, &request.receipt_key) else {
        return Ok(None);
    };
    let receipt = PathBuf::from(receipt);
    let key_path = PathBuf::from(key_path);
    require_missing_receipt(&receipt)?;

    let mut paths = vec![
        ("出力フォルダ", Path::new(&request.output_dir)),
        ("実行証明", receipt.as_path()),
        ("署名鍵", key_path.as_path()),
    ];
    if let Some(input_dir) = request.input_dir.as_deref() {
        paths.push(("入力フォルダ", Path::new(input_dir)));
    }
    for input in &request.inputs {
        paths.push(("入力", Path::new(input)));
    }
    require_distinct_execution_paths(&paths)?;

    if let Some(input_dir) = request.input_dir.as_deref() {
        let input_root = normalize_batch_path(Path::new(input_dir))?;
        let receipt_path = normalize_batch_path(&receipt)?;
        if receipt_path.starts_with(&input_root) {
            return Err(format!(
                "バッチ実行証明は入力フォルダの外へ保存してください: {}",
                receipt.display()
            ));
        }
    }

    let key = ReceiptSecretKey::from_file(&key_path)?;
    let stage = AtomicOutput::new(&receipt)?;
    Ok(Some(UnplannedDesktopBatchReceipt {
        path: receipt,
        key_path,
        key,
        stage,
        _recovery_stage: None,
    }))
}

fn validate_batch_receipt_output_paths(
    items: &[BatchItem],
    receipt: &UnplannedDesktopBatchReceipt,
) -> Result<(), String> {
    validate_batch_reserved_path(items, &receipt.path, "実行証明")?;
    validate_batch_reserved_path(items, &receipt.key_path, "実行証明の署名鍵")
}

fn validate_request(request: &ProcessRequest) -> Result<(), String> {
    validate_process_options(&request.options)?;
    validate_receipt_pair(request.receipt.as_deref(), request.receipt_key.as_deref())?;
    if request.expected_recipe.is_some() && request.expected_input_fingerprint.is_none() {
        return Err("期待recipeには期待入力fingerprintも指定してください".into());
    }
    let format = OutputFormat::from_path(Path::new(&request.output))?;
    if request.stream {
        if !(1..=denoize::config::MAX_STREAM_BLOCK_FRAMES).contains(&request.stream_frames) {
            return Err(format!(
                "ストリームブロックは1〜{} framesにしてください",
                denoize::config::MAX_STREAM_BLOCK_FRAMES
            ));
        }
        format.validate_encoder(parse_aac_encoder(&request.options.aac_encoder)?)?;
        if let Some(backend) = configured_backend(&request.options.backend)? {
            if !StreamingBackendSession::supports(backend) {
                return Err(format!(
                    "バックエンド {} は長時間ストリームに対応していません",
                    service::backend_name(backend)
                ));
            }
        }
    } else {
        if request.resume {
            return Err("単一ファイルの再開には長時間ストリームを有効にしてください".into());
        }
        format.validate_encoder(parse_aac_encoder(&request.options.aac_encoder)?)?;
    }
    preflight_explicit_backend_resources(&request.options)?;
    if !Path::new(&request.input).is_file() {
        return Err("入力ファイルが存在しません".into());
    }
    if request.resume {
        Ok(())
    } else {
        ensure_output_available(Path::new(&request.output), request.options.force)
    }
}

fn validate_expected_preview_binding(
    request: &ProcessRequest,
    input_fingerprint: batch_resume::FileFingerprint,
    recipe: Digest,
) -> Result<(), String> {
    if let Some(expected_input) = request.expected_input_fingerprint {
        if input_fingerprint != expected_input {
            return Err("期待した入力fingerprintと最終処理の入力fingerprintが一致しません".into());
        }
    }
    if let Some(expected_recipe) = request.expected_recipe {
        if recipe != expected_recipe {
            return Err("採用したプレビューと最終処理のrecipeが一致しません".into());
        }
    }
    Ok(())
}

fn build_process_execution_plan(request: &ProcessRequest) -> Result<ExecutionPlan, String> {
    if request.stream {
        return build_stream_process_execution_plan(request);
    }
    if request.resume {
        return Err("単一ファイルの再開には長時間ストリームを有効にしてください".into());
    }
    validate_process_options(&request.options)?;
    preflight_explicit_backend_resources_read_only(&request.options)?;
    let input = Path::new(&request.input);
    let output = Path::new(&request.output);
    require_distinct_execution_paths(&[("入力", input), ("出力", output)])?;
    let format = OutputFormat::from_path(output)?;
    let encode = parsed_encode_options(&request.options)?;
    format.validate_encoder(encode.aac_encoder)?;
    let (publication, reason) = desktop_planned_publication(output, request.options.force)?;
    let governor = desktop_resource_governor(&request.options, 1)?;
    let decode_limits = desktop_decode_limits(&request.options)?;
    let mut input_session = denoize::AudioInputSession::open(input)?;
    let input_bytes = input_session.len();
    let probe = denoize::probe_file_from_session_with_limits(&mut input_session, decode_limits)?;
    if probe.audio_tracks != 1 || probe.codec == denoize::AudioCodec::Unknown {
        return Err("実行計画の入力には対応する音声トラックが1つ必要です".into());
    }
    let input_fingerprint = batch_resume::fingerprint_input_session(&mut input_session)?;
    let audio = read_audio_from_session_with_limits(&mut input_session, decode_limits)?;
    format.validate_config(&audio, &encode)?;
    let decoded_working_set = estimate_audio_working_set_bytes(&audio);
    let metadata_limits =
        desktop_retained_metadata_limits(decode_limits.max_working_set_bytes, decoded_working_set);
    let metadata_bytes = if request.options.preserve_metadata {
        input_session
            .read_metadata_with_limits(metadata_limits)?
            .as_ref()
            .map(denoize::metadata::Metadata::estimated_memory_bytes)
            .unwrap_or(0)
    } else {
        0
    };
    let processing = resolved_processing_options_read_only(&request.options, &audio)?;
    let model = batch_resume::consumed_model(&processing)?;
    let worker_request = desktop_worker_request(
        input_bytes,
        &audio,
        metadata_bytes,
        decode_limits.max_working_set_bytes,
        &processing,
        true,
    )?;
    let resource_request =
        worker_request.checked_add(denoize::estimate_backend_session_request(
            processing.backend,
            &processing.backend_options,
            processing.accelerator,
        )?)?;
    drop(
        governor
            .try_acquire(resource_request)?
            .ok_or_else(|| "設定された資源上限では実行計画を許可できません".to_string())?,
    );
    let _backend = BackendSession::prepare_with_accelerator(
        processing.backend,
        processing.backend_options.clone(),
        processing.accelerator,
    )?;
    if let Some(model) = &model {
        if batch_resume::fingerprint_file(&model.path)? != model.fingerprint {
            return Err(format!(
                "実行計画中にモデルが変更されました: {}",
                model.path.display()
            ));
        }
    }
    if batch_resume::fingerprint_input_session(&mut input_session)? != input_fingerprint
        || batch_resume::fingerprint_file(input)? != input_fingerprint
    {
        return Err(format!(
            "実行計画中に入力が変更されました: {}",
            input.display()
        ));
    }
    let metadata_policy = if request.options.preserve_metadata {
        MetadataPolicy::Preserve
    } else {
        MetadataPolicy::Drop
    };
    let recipe = batch_resume::recipe_digest(
        &processing,
        audio.channels(),
        format,
        encode,
        metadata_policy,
        model
            .as_ref()
            .map(|model| (&model.fingerprint, model.sample_rate)),
    )?;
    validate_expected_preview_binding(request, input_fingerprint, recipe)?;
    let output_locator = denoize::portable_file_locator(output)?;
    let item_id = denoize::execution_item_id(input_fingerprint, &output_locator, recipe)?;
    let frames = u64::try_from(audio.frames())
        .map_err(|_| "実行計画のフレーム数が大きすぎます".to_string())?;
    let model = model
        .as_ref()
        .map(|model| {
            Ok::<PlannedArtifact, String>(PlannedArtifact {
                path: denoize::portable_file_locator(&model.path)?,
                fingerprint: model.fingerprint,
            })
        })
        .transpose()?;
    ExecutionPlan::new(
        ExecutionKind::File,
        processing.backend_options.deterministic,
        desktop_metadata_policy_name(metadata_policy),
        vec![ExecutionPlanItem {
            item_id,
            input: PlannedArtifact {
                path: denoize::portable_file_locator(input)?,
                fingerprint: input_fingerprint,
            },
            output: PlannedOutput {
                path: output_locator,
                format: desktop_output_format_name(format).into(),
                publication: publication.into(),
                action: "process".into(),
                reason: reason.into(),
                existing_fingerprint: None,
            },
            model,
            recipe,
            backend: service::backend_name(processing.backend).into(),
            accelerator: processing.accelerator.effective().name().into(),
            input_format: desktop_audio_format_name(probe.format).into(),
            input_codec: desktop_audio_codec_name(probe.codec).into(),
            channels: audio.channels() as u64,
            frames,
            sample_rate: audio.sample_rate,
            resources: desktop_planned_resources(resource_request),
        }],
    )
}

fn build_stream_process_execution_plan(request: &ProcessRequest) -> Result<ExecutionPlan, String> {
    validate_process_options(&request.options)?;
    if !(1..=denoize::config::MAX_STREAM_BLOCK_FRAMES).contains(&request.stream_frames) {
        return Err(format!(
            "ストリームブロックは1〜{} framesにしてください",
            denoize::config::MAX_STREAM_BLOCK_FRAMES
        ));
    }
    preflight_explicit_backend_resources_read_only(&request.options)?;
    let input = Path::new(&request.input);
    let output = Path::new(&request.output);
    require_distinct_execution_paths(&[("入力", input), ("出力", output)])?;
    let maximum = checked_desktop_mib(request.options.max_process_memory_mb, "プロセスメモリ上限")?;
    let configured_temporary =
        checked_desktop_mib(request.options.max_temporary_mb, "一時領域上限")?;
    let output_format = OutputFormat::from_path(output)?;
    let encode_options = parsed_encode_options(&request.options)?;
    output_format.validate_encoder(encode_options.aac_encoder)?;
    let initial_publication = if request.resume {
        None
    } else {
        Some(desktop_planned_publication(output, request.options.force)?)
    };
    let governor = desktop_resource_governor(&request.options, 1)?;
    let mut input_session = denoize::AudioInputSession::open(input)?;
    let initial_limits = desktop_decode_limits(&request.options)?;
    let initial_info = denoize::inspect_audio_stream_session(&mut input_session, initial_limits)?;
    let spec = initial_info.output_spec;
    let channel_mask = initial_info.channel_mask;
    let encode_spec = StreamEncodeSpec::new(spec, channel_mask, initial_info.total_frames);
    output_format.validate_stream_config(encode_spec, encode_options)?;
    let auxiliary_limit = match (initial_info.total_frames, configured_temporary) {
        (None, Some(limit)) => limit / 3,
        (_, Some(limit)) => limit,
        (_, None) => StreamEncodeLimits::default().max_auxiliary_temporary_bytes(),
    };
    let encode_limits = StreamEncodeLimits::new(auxiliary_limit);
    let config = processing_config(&request.options, spec.sample_rate)?;
    let backend =
        configured_backend(&request.options.backend)?.unwrap_or_else(service::select_live_backend);
    if !StreamingBackendSession::supports(backend) {
        return Err(format!(
            "バックエンド {} は長時間ストリームに対応していません",
            service::backend_name(backend)
        ));
    }
    let backend_options = resolve_gui_backend_options_read_only(backend, &request.options)?;
    let accelerator = denoize::select_accelerator_for_options(backend, &backend_options)?;
    let base_working_set = estimate_stream_memory_bytes_checked(
        spec.channels as usize,
        request.stream_frames,
        config.frame_size,
        spec.sample_rate,
        config.profile_ms,
    )
    .map_err(|error| error.to_string())?;
    let backend_state = StreamingBackendSession::estimate_additional_bytes(
        backend,
        spec.sample_rate,
        spec.channels as usize,
        backend_options.channel_mode,
    )
    .map_err(|error| error.to_string())?;
    let vad_state = if config.vad {
        StreamingBackendSession::estimate_vad_additional_bytes(
            spec.sample_rate,
            spec.channels as usize,
            request.stream_frames,
            config.frame_size,
            config.profile_ms,
        )
        .map_err(|error| error.to_string())?
    } else {
        0
    };
    let loudness_state = if request.options.loudness_lufs.is_some() {
        denoize::loudness::estimate_streaming_loudness_bytes(
            spec.channels as usize,
            spec.sample_rate,
            request.stream_frames,
        )
        .map_err(|error| error.to_string())?
    } else {
        0
    };
    let encoder_state = denoize::estimate_stream_encode_additional_bytes(
        output_format,
        encode_spec,
        request.stream_frames,
        encode_options,
    )?;
    let initial_working_set = base_working_set
        .checked_add(backend_state)
        .and_then(|bytes| bytes.checked_add(vad_state))
        .and_then(|bytes| bytes.checked_add(loudness_state))
        .and_then(|bytes| bytes.checked_add(initial_info.decoder_additional_bytes))
        .and_then(|bytes| bytes.checked_add(encoder_state))
        .and_then(|bytes| {
            bytes.checked_add(denoize::estimate_backend_worker_memory_bytes(
                &backend_options,
            ))
        })
        .and_then(|bytes| {
            bytes.checked_add(if request.resume {
                batch_resume::STREAM_CHECKPOINT_SCRATCH_BYTES
            } else {
                0
            })
        })
        .ok_or_else(|| "ストリームのメモリ予約量が大きすぎます".to_string())?;
    let verification_block_frames = request.stream_frames.min(DEFAULT_STREAM_BLOCK_FRAMES);
    let initial_verification_working_set = denoize::estimate_stream_output_verification_bytes(
        output_format,
        encode_spec,
        verification_block_frames,
        encode_options,
        encode_limits,
        initial_limits,
    )?;
    let initial_required_memory = initial_working_set.max(initial_verification_working_set);
    if maximum.is_some_and(|limit| initial_required_memory > limit) {
        return Err(format!(
            "ストリームには{initial_required_memory} bytes必要ですが、プロセスメモリ上限を超えます"
        ));
    }
    let metadata_limits = desktop_retained_metadata_limits(maximum, initial_required_memory);
    let decode_limits = DecodeLimits::new(metadata_limits, maximum);
    let info = denoize::inspect_audio_stream_session(&mut input_session, decode_limits)?;
    if info.format != initial_info.format
        || info.codec != initial_info.codec
        || info.output_spec != initial_info.output_spec
        || info.channel_mask != initial_info.channel_mask
        || info.total_frames != initial_info.total_frames
        || info.max_decoder_frames != initial_info.max_decoder_frames
    {
        return Err("事前検査中にストリーム入力形状が変化しました".into());
    }
    let working_set = base_working_set
        .checked_add(backend_state)
        .and_then(|bytes| bytes.checked_add(vad_state))
        .and_then(|bytes| bytes.checked_add(loudness_state))
        .and_then(|bytes| bytes.checked_add(info.decoder_additional_bytes))
        .and_then(|bytes| bytes.checked_add(encoder_state))
        .and_then(|bytes| {
            bytes.checked_add(denoize::estimate_backend_worker_memory_bytes(
                &backend_options,
            ))
        })
        .and_then(|bytes| {
            bytes.checked_add(if request.resume {
                batch_resume::STREAM_CHECKPOINT_SCRATCH_BYTES
            } else {
                0
            })
        })
        .ok_or_else(|| "ストリームのメモリ予約量が大きすぎます".to_string())?;
    let verification_working_set = denoize::estimate_stream_output_verification_bytes(
        output_format,
        encode_spec,
        verification_block_frames,
        encode_options,
        encode_limits,
        decode_limits,
    )?;
    let required_memory = working_set.max(verification_working_set);
    if maximum.is_some_and(|limit| required_memory > limit) {
        return Err(format!(
            "ストリームには{required_memory} bytes必要ですが、プロセスメモリ上限を超えます"
        ));
    }

    let resolved = service::ResolvedProcessingOptions {
        backend,
        denoiser: config.clone(),
        backend_options: backend_options.clone(),
        accelerator,
        loudness_lufs: request.options.loudness_lufs,
        true_peak_dbtp: request.options.true_peak_dbtp,
    };
    resolved.validate_config()?;
    let input_fingerprint = batch_resume::fingerprint_input_session(&mut input_session)?;
    let model = if request.resume {
        batch_resume::resumable_consumed_model(&resolved)?
    } else {
        batch_resume::consumed_model(&resolved)?
    };
    let metadata_policy = if request.options.preserve_metadata {
        MetadataPolicy::Preserve
    } else {
        MetadataPolicy::Drop
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
    let temporary_reservation = desktop_stream_temporary_bytes(
        info,
        output_format,
        encode_spec,
        encode_options,
        encode_limits,
        configured_temporary,
        request.resume,
        request.options.loudness_lufs.is_some(),
        metadata_bytes,
    )?;
    let mut worker_request = ResourceRequest::worker(
        working_set
            .checked_add(metadata_bytes)
            .ok_or_else(|| "ストリームのメモリ予約量が大きすぎます".to_string())?
            .max(verification_working_set),
        temporary_reservation.total_bytes,
    );
    if accelerator.effective() != denoize::AcceleratorRuntime::Cpu {
        let gpu_memory = working_set
            .checked_mul(2)
            .and_then(|bytes| {
                bytes.checked_add(denoize::estimate_backend_worker_gpu_memory_bytes(
                    &backend_options,
                ))
            })
            .ok_or_else(|| "ストリームのGPU予約量が大きすぎます".to_string())?;
        worker_request = worker_request
            .with_gpu_jobs(1)
            .with_gpu_memory_bytes(gpu_memory);
    }
    let resources = worker_request.checked_add(denoize::estimate_backend_session_request(
        backend,
        &backend_options,
        accelerator,
    )?)?;
    drop(
        governor.try_acquire(resources)?.ok_or_else(|| {
            "設定された資源上限ではストリーム実行計画を許可できません".to_string()
        })?,
    );
    let _processor = StreamingBackendSession::new_with_accelerator(
        backend,
        spec.sample_rate,
        spec.channels as usize,
        config,
        backend_options.clone(),
        accelerator,
    )?;
    if let Some(model) = &model {
        if batch_resume::fingerprint_file(&model.path)? != model.fingerprint {
            return Err(format!(
                "実行計画中にモデルが変更されました: {}",
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
    validate_expected_preview_binding(request, input_fingerprint, base_recipe)?;
    let recipe = batch_resume::stream_recipe_digest(base_recipe, request.stream_frames, info)?;
    let mut reader = AudioStreamReader::from_session(input_session, decode_limits)?;
    let mut frames = 0_u64;
    while let Some(block) = reader.next_block(request.stream_frames)? {
        frames = frames
            .checked_add(block.first().map(Vec::len).unwrap_or(0) as u64)
            .ok_or_else(|| "ストリーム実行計画のフレーム数が大きすぎます".to_string())?;
    }
    if frames == 0 {
        return Err("ストリーム実行計画の入力にpresentation frameがありません".into());
    }
    if reader.fingerprint_input()? != input_fingerprint
        || batch_resume::fingerprint_file(input)? != input_fingerprint
    {
        return Err(format!(
            "実行計画中にストリーム入力が変更されました: {}",
            input.display()
        ));
    }
    let (publication, action, reason, existing_fingerprint, planned_resources) = if request.resume {
        let decision = batch_resume::inspect_stream_checkpoint_decision(
            output,
            input_fingerprint,
            recipe,
            spec,
            request.stream_frames,
            temporary_reservation.checkpoint_limit,
            request.options.force,
        )?;
        if decision
            .checkpoint()
            .is_some_and(|checkpoint| checkpoint.input_frames() > frames)
        {
            return Err("ストリームcheckpointが現在の入力長を超えています".into());
        }
        match decision {
            batch_resume::StreamCheckpointDecision::Skip { checkpoint, output } => {
                if checkpoint.input_frames() != frames || checkpoint.output_frames() != frames {
                    return Err("完了済みストリームcheckpointの長さが現在の入力と異なります".into());
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
                    desktop_planned_publication(output, request.options.force)?;
                let reason = if reset {
                    "forced"
                } else if checkpoint.is_some() {
                    "checkpoint"
                } else {
                    publication_reason
                };
                (publication, "process", reason, None, resources)
            }
        }
    } else {
        let (publication, reason) =
            initial_publication.ok_or("ストリーム実行計画のpublication判定がありません")?;
        (publication, "process", reason, None, resources)
    };
    let evidence = DesktopStreamExecutionEvidence {
        input_fingerprint,
        stream_info: info,
        model,
        recipe,
        output_format,
        resources: planned_resources,
        backend,
        accelerator,
        deterministic: backend_options.deterministic,
        metadata_policy,
    };
    build_desktop_stream_plan_from_evidence(
        input,
        output,
        publication,
        action,
        reason,
        existing_fingerprint,
        &evidence,
        frames,
    )
}

fn validated_processing_options(
    options: &ProcessOptions,
    audio: &denoize::Audio,
) -> Result<ProcessingOptions, String> {
    let denoiser = processing_config(options, audio.sample_rate)?;
    let backend = match configured_backend(&options.backend)? {
        Some(backend) => BackendChoice::Explicit(backend),
        None => BackendChoice::Auto,
    };
    let selected_backend = service::select_backend(
        backend,
        audio.frames() as f64 / audio.sample_rate.max(1) as f64,
        None,
    );
    let processing = ProcessingOptions {
        backend,
        quality: None,
        denoiser,
        backend_options: parsed_backend_options_for(selected_backend, options)?,
        loudness_lufs: options.loudness_lufs,
        true_peak_dbtp: options.true_peak_dbtp,
    };
    processing
        .validate_config(audio)
        .map_err(|error| error.to_string())?;
    Ok(processing)
}

fn resolved_processing_options(
    options: &ProcessOptions,
    audio: &denoize::Audio,
) -> Result<service::ResolvedProcessingOptions, String> {
    service::resolve_processing_options(audio, validated_processing_options(options, audio)?)
}

fn resolved_processing_options_read_only(
    options: &ProcessOptions,
    audio: &denoize::Audio,
) -> Result<service::ResolvedProcessingOptions, String> {
    service::resolve_processing_options_read_only(
        audio,
        validated_processing_options(options, audio)?,
    )
}

fn desktop_worker_request(
    input_bytes: u64,
    audio: &denoize::Audio,
    metadata_bytes: u64,
    decode_reservation_bytes: Option<u64>,
    processing: &service::ResolvedProcessingOptions,
    writes_output: bool,
) -> Result<ResourceRequest, String> {
    let memory_bytes = estimate_audio_working_set_bytes(audio)
        .checked_add(metadata_bytes)
        .and_then(|bytes| {
            bytes.checked_add(denoize::estimate_backend_worker_memory_bytes(
                &processing.backend_options,
            ))
        })
        .ok_or_else(|| "ワーカーメモリ予約量が大きすぎます".to_string())?
        .max(decode_reservation_bytes.unwrap_or(0));
    let mut request = ResourceRequest::worker(
        memory_bytes,
        if writes_output {
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
            .ok_or_else(|| "ワーカーGPUメモリ予約量が大きすぎます".to_string())?;
        request = request.with_gpu_jobs(1).with_gpu_memory_bytes(gpu_bytes);
    }
    Ok(request)
}

fn desktop_planned_resources(request: ResourceRequest) -> PlannedResources {
    PlannedResources {
        memory_bytes: request.memory_bytes(),
        temporary_bytes: request.temporary_bytes(),
        cpu_jobs: request.cpu_jobs() as u64,
        gpu_jobs: request.gpu_jobs() as u64,
        gpu_memory_bytes: request.gpu_memory_bytes(),
    }
}

fn desktop_metadata_policy_name(policy: MetadataPolicy) -> &'static str {
    match policy {
        MetadataPolicy::Preserve => "preserve",
        MetadataPolicy::Drop => "drop",
    }
}

fn desktop_output_format_name(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Wav => "wav",
        OutputFormat::Flac => "flac",
        OutputFormat::OggOpus => "ogg-opus",
        OutputFormat::Mp3 => "mp3",
        OutputFormat::M4a => "m4a",
        OutputFormat::AacAdts => "aac-adts",
    }
}

fn desktop_audio_format_name(format: denoize::AudioFormat) -> &'static str {
    match format {
        denoize::AudioFormat::Wav => "wav",
        denoize::AudioFormat::Rf64 => "rf64",
        denoize::AudioFormat::Aiff => "aiff",
        denoize::AudioFormat::Caf => "caf",
        denoize::AudioFormat::Flac => "flac",
        denoize::AudioFormat::OggOpus => "ogg-opus",
        denoize::AudioFormat::OggVorbis => "ogg-vorbis",
        denoize::AudioFormat::Mp3 => "mp3",
        denoize::AudioFormat::M4a => "m4a",
        denoize::AudioFormat::AacAdts => "aac-adts",
        denoize::AudioFormat::Unknown => "unknown",
    }
}

fn desktop_audio_codec_name(codec: denoize::AudioCodec) -> &'static str {
    match codec {
        denoize::AudioCodec::Pcm => "pcm",
        denoize::AudioCodec::Flac => "flac",
        denoize::AudioCodec::Opus => "opus",
        denoize::AudioCodec::Vorbis => "vorbis",
        denoize::AudioCodec::Mp3 => "mp3",
        denoize::AudioCodec::Aac => "aac",
        denoize::AudioCodec::Alac => "alac",
        denoize::AudioCodec::Unknown => "unknown",
    }
}

#[derive(Clone)]
struct DesktopStreamExecutionEvidence {
    input_fingerprint: batch_resume::FileFingerprint,
    stream_info: AudioStreamInfo,
    model: Option<batch_resume::ConsumedModel>,
    recipe: Digest,
    output_format: OutputFormat,
    resources: ResourceRequest,
    backend: Backend,
    accelerator: denoize::AcceleratorSelection,
    deterministic: bool,
    metadata_policy: MetadataPolicy,
}

fn build_desktop_stream_plan_from_evidence(
    input: &Path,
    output: &Path,
    publication: &str,
    action: &str,
    reason: &str,
    existing_fingerprint: Option<batch_resume::FileFingerprint>,
    evidence: &DesktopStreamExecutionEvidence,
    frames: u64,
) -> Result<ExecutionPlan, String> {
    let input_locator = denoize::portable_file_locator(input)?;
    let output_locator = denoize::portable_file_locator(output)?;
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
        desktop_metadata_policy_name(evidence.metadata_policy),
        vec![ExecutionPlanItem {
            item_id,
            input: PlannedArtifact {
                path: input_locator,
                fingerprint: evidence.input_fingerprint,
            },
            output: PlannedOutput {
                path: output_locator,
                format: desktop_output_format_name(evidence.output_format).into(),
                publication: publication.into(),
                action: action.into(),
                reason: reason.into(),
                existing_fingerprint,
            },
            model,
            recipe: evidence.recipe,
            backend: service::backend_name(evidence.backend).into(),
            accelerator: evidence.accelerator.effective().name().into(),
            input_format: desktop_audio_format_name(evidence.stream_info.format).into(),
            input_codec: desktop_audio_codec_name(evidence.stream_info.codec).into(),
            channels: u64::from(evidence.stream_info.output_spec.channels),
            frames,
            sample_rate: evidence.stream_info.output_spec.sample_rate,
            resources: desktop_planned_resources(evidence.resources),
        }],
    )
}

fn verify_desktop_stream_bound_sources(
    reader: &AudioStreamReader,
    input: &Path,
    bound_input: Option<batch_resume::FileFingerprint>,
    bound_model: Option<&batch_resume::ConsumedModel>,
    evidence: Option<&DesktopStreamExecutionEvidence>,
) -> Result<(), String> {
    let expected_input = match (bound_input, evidence) {
        (Some(bound), Some(evidence)) if bound != evidence.input_fingerprint => {
            return Err("ストリーム入力の拘束情報が一致しません".into());
        }
        (Some(bound), _) => Some(bound),
        (None, Some(evidence)) => Some(evidence.input_fingerprint),
        (None, None) => None,
    };
    if let Some(expected_input) = expected_input {
        if reader.fingerprint_input()? != expected_input
            || batch_resume::fingerprint_file(input)? != expected_input
        {
            return Err(format!(
                "プレビューまたは実行証明に拘束されたストリーム入力が処理中に変更されました: {}",
                input.display()
            ));
        }
    }
    if let (Some(bound), Some(evidence_model)) = (
        bound_model,
        evidence.and_then(|evidence| evidence.model.as_ref()),
    ) {
        if bound != evidence_model {
            return Err("ストリームモデルの拘束情報が一致しません".into());
        }
    }
    let expected_model = bound_model.or_else(|| evidence.and_then(|item| item.model.as_ref()));
    if let Some(model) = expected_model {
        if batch_resume::fingerprint_file(&model.path)? != model.fingerprint {
            return Err(format!(
                "プレビューまたは実行証明に拘束されたストリームモデルが処理中に変更されました: {}",
                model.path.display()
            ));
        }
    }
    Ok(())
}

fn write_desktop_stream_receipt(
    receipt: &mut DesktopReceiptContext,
    input: &Path,
    output: &Path,
    evidence: &DesktopStreamExecutionEvidence,
    frames: u64,
    output_fingerprint: batch_resume::FileFingerprint,
    publication: &str,
    action: &str,
    reason: &str,
    existing_fingerprint: Option<batch_resume::FileFingerprint>,
) -> Result<(), String> {
    let plan = build_desktop_stream_plan_from_evidence(
        input,
        output,
        publication,
        action,
        reason,
        existing_fingerprint,
        evidence,
        frames,
    )?;
    let plan_item = plan
        .items
        .first()
        .ok_or("ストリーム実行計画に項目がありません")?;
    let outcome = match action {
        "process" => "succeeded",
        "skip" => "skipped",
        value => return Err(format!("不明なストリーム実行証明actionです: {value}")),
    };
    let item = ReceiptItem::from_plan_item(plan_item, output_fingerprint, outcome)?;
    let payload = ExecutionReceiptPayload::new(&plan, vec![item])?;
    let signed = receipt.key.sign(payload)?;
    write_desktop_receipt_stage(&mut receipt.stage, &receipt.path, &signed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DesktopStreamTemporaryReservation {
    total_bytes: u64,
    encoder_auxiliary_bytes: u64,
    checkpoint_limit: Option<u64>,
}

fn desktop_virtual_wav_bytes(info: AudioStreamInfo, frames: u64) -> Result<u64, String> {
    frames
        .checked_mul(u64::from(info.output_spec.channels))
        .and_then(|samples| samples.checked_mul(u64::from(info.output_spec.bits_per_sample / 8)))
        .and_then(|bytes| bytes.checked_add(68))
        .ok_or_else(|| "仮想WAV出力サイズが大きすぎます".to_string())
}

#[allow(clippy::too_many_arguments)]
fn desktop_stream_temporary_bytes(
    info: AudioStreamInfo,
    output_format: OutputFormat,
    encode_spec: StreamEncodeSpec,
    encode_options: EncodeOptions,
    encode_limits: StreamEncodeLimits,
    configured_limit: Option<u64>,
    checkpointed: bool,
    two_pass_loudness: bool,
    metadata_allowance_bytes: u64,
) -> Result<DesktopStreamTemporaryReservation, String> {
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
        if let Some(limit) = configured_limit {
            let unavailable = encoder_auxiliary_bytes
                .checked_add(metadata_allowance_bytes)
                .ok_or_else(|| "ストリーム一時領域の予約量が大きすぎます".to_string())?;
            if unavailable >= limit {
                return Err(format!(
                    "エンコーダー補助データとメタデータに{unavailable} bytes必要なため、一時領域上限{limit} bytes内に出力容量が残りません"
                ));
            }
            return Ok(DesktopStreamTemporaryReservation {
                total_bytes: limit,
                encoder_auxiliary_bytes,
                checkpoint_limit: checkpointed.then_some(limit - unavailable),
            });
        }
        if !checkpointed {
            let mut total_bytes = MAX_WAV_FILE_BYTES
                .checked_add(encoder_auxiliary_bytes)
                .and_then(|bytes| bytes.checked_add(metadata_allowance_bytes))
                .ok_or_else(|| "ストリーム一時領域の予約量が大きすぎます".to_string())?;
            if two_pass_loudness {
                let data_limit = MAX_WAV_FILE_BYTES.saturating_sub(68);
                let output_sample_bytes = u64::from(info.output_spec.bits_per_sample / 8);
                let max_samples = data_limit / output_sample_bytes;
                let spool_bytes = max_samples
                    .checked_mul(std::mem::size_of::<f64>() as u64)
                    .ok_or_else(|| "ストリームラウドネスPCMの一時領域が大きすぎます".to_string())?;
                total_bytes = total_bytes
                    .checked_add(spool_bytes)
                    .ok_or_else(|| "ストリームラウドネスの一時領域が大きすぎます".to_string())?;
            }
            return Ok(DesktopStreamTemporaryReservation {
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
            .ok_or_else(|| "ストリームチェックポイントの一時領域が大きすぎます".to_string())?;
        let checkpoint_limit = MAX_WAV_FILE_BYTES
            .checked_add(spool_bytes)
            .ok_or_else(|| "ストリームチェックポイントの一時領域が大きすぎます".to_string())?;
        let total_bytes = checkpoint_limit
            .checked_add(encoder_auxiliary_bytes)
            .and_then(|bytes| bytes.checked_add(metadata_allowance_bytes))
            .ok_or_else(|| "ストリームチェックポイントの一時領域が大きすぎます".to_string())?;
        return Ok(DesktopStreamTemporaryReservation {
            total_bytes,
            encoder_auxiliary_bytes,
            checkpoint_limit: Some(checkpoint_limit),
        });
    };
    let staged_output_bytes = staged_output_bytes
        .ok_or_else(|| "既知のストリーム長に出力サイズ上限がありません".to_string())?;
    let base_bytes = staged_output_bytes
        .checked_add(encoder_auxiliary_bytes)
        .and_then(|bytes| bytes.checked_add(metadata_allowance_bytes))
        .ok_or_else(|| "ストリーム出力サイズが大きすぎます".to_string())?;
    if !checkpointed {
        let total_bytes = if two_pass_loudness {
            let spool_bytes = frames
                .checked_mul(u64::from(info.output_spec.channels))
                .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>() as u64))
                .ok_or_else(|| "ストリームラウドネスPCMの一時領域が大きすぎます".to_string())?;
            base_bytes
                .checked_add(spool_bytes)
                .ok_or_else(|| "ストリームラウドネスの一時領域が大きすぎます".to_string())?
        } else {
            base_bytes
        };
        if configured_limit.is_some_and(|limit| total_bytes > limit) {
            return Err(format!(
                "一時出力、エンコーダー補助データ、メタデータ、multi-pass PCMに{total_bytes} bytes必要ですが、一時領域上限を超えます"
            ));
        }
        return Ok(DesktopStreamTemporaryReservation {
            total_bytes,
            encoder_auxiliary_bytes,
            checkpoint_limit: None,
        });
    }
    let spool_bytes = frames
        .checked_mul(u64::from(info.output_spec.channels))
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>() as u64))
        .ok_or_else(|| "ストリームチェックポイントのPCMサイズが大きすぎます".to_string())?;
    let total_bytes = base_bytes
        .checked_add(spool_bytes)
        .ok_or_else(|| "ストリームチェックポイントの一時領域が大きすぎます".to_string())?;
    if configured_limit.is_some_and(|limit| total_bytes > limit) {
        return Err(format!(
            "チェックポイント、一時出力、エンコーダー補助データ、メタデータに{total_bytes} bytes必要ですが、一時領域上限を超えます"
        ));
    }
    let checkpoint_limit = desktop_virtual_wav_bytes(info, frames)?
        .checked_add(spool_bytes)
        .ok_or_else(|| "ストリームチェックポイントの一時領域が大きすぎます".to_string())?;
    Ok(DesktopStreamTemporaryReservation {
        total_bytes,
        encoder_auxiliary_bytes,
        checkpoint_limit: Some(checkpoint_limit),
    })
}

fn desktop_stream_pcm_spool_limit(
    info: AudioStreamInfo,
    total_temporary_bytes: u64,
    encoder_auxiliary_bytes: u64,
    metadata_bytes: u64,
) -> Result<u64, String> {
    if let Some(frames) = info.total_frames {
        return frames
            .checked_mul(u64::from(info.output_spec.channels))
            .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>() as u64))
            .ok_or_else(|| "ストリームラウドネスPCMサイズが大きすぎます".to_string());
    }
    let unavailable = encoder_auxiliary_bytes
        .checked_add(metadata_bytes)
        .ok_or_else(|| "ストリームラウドネス一時領域が大きすぎます".to_string())?;
    let shared = total_temporary_bytes
        .checked_sub(unavailable)
        .ok_or_else(|| "ストリームラウドネスPCM用の一時領域がありません".to_string())?;
    // Encoded WAV is at most half the interleaved-f64 PCM size. Keep at least
    // the other half for the staged output when the decoder cannot declare a
    // presentation length during preflight.
    Ok(shared / 2)
}

fn analyze_desktop_stream_pcm_spool(
    spool: &mut StreamPcmSpool,
    channels: usize,
    sample_rate: u32,
    channel_mask: Option<denoize::ChannelMask>,
    block_frames: usize,
    target_lufs: f64,
    true_peak_dbtp: f64,
) -> Result<denoize::loudness::StreamingLoudnessGain, String> {
    spool.prepare_read()?;
    let mut analyzer =
        denoize::loudness::StreamingLoudnessAnalyzer::new(channels, sample_rate, channel_mask)?;
    while let Some(block) = spool.next_block(block_frames)? {
        analyzer.add_block(&block)?;
    }
    let gain = analyzer.finish(target_lufs, true_peak_dbtp)?;
    spool.prepare_read()?;
    Ok(gain)
}

fn replay_desktop_stream_checkpoint(
    reader: &mut AudioStreamReader,
    processor: &mut StreamingBackendSession,
    block_frames: usize,
    checkpoint: batch_resume::StreamCheckpoint,
    channels: usize,
    control: &JobControl,
) -> Result<u64, String> {
    let mut digest = batch_resume::StreamPcmDigest::new(channels)?;
    let mut input_frames = 0_u64;
    while input_frames < checkpoint.input_frames() {
        check_cancelled(control)?;
        let block = reader
            .next_block(block_frames)?
            .ok_or_else(|| "チェックポイントが入力音声の終端を超えています".to_string())?;
        let frames = block.first().map(Vec::len).unwrap_or(0) as u64;
        let next = input_frames
            .checked_add(frames)
            .ok_or_else(|| "ストリーム再生フレーム数が大きすぎます".to_string())?;
        if next > checkpoint.input_frames() {
            return Err("チェックポイントとデコーダーブロック境界が一致しません".into());
        }
        digest.update(&processor.process_block(&block)?)?;
        input_frames = next;
    }
    if digest.frames() != checkpoint.output_frames()
        || digest.len() != checkpoint.spool_len()
        || digest.digest() != checkpoint.spool_digest()
    {
        return Err(
            "再生したストリーム状態がチェックポイントと一致しません。上書きで再作成してください"
                .into(),
        );
    }
    Ok(input_frames)
}

fn process_desktop_stream_blocks(
    reader: &mut AudioStreamReader,
    processor: &mut StreamingBackendSession,
    block_frames: usize,
    control: &JobControl,
    mut write_block: impl FnMut(&[Vec<f64>]) -> Result<(), String>,
) -> Result<u64, String> {
    let mut input_frames = 0_u64;
    let mut output_frames = 0_u64;
    while let Some(block) = reader.next_block(block_frames)? {
        check_cancelled(control)?;
        let decoded_frames = block.first().map(Vec::len).unwrap_or(0) as u64;
        let enhanced = processor.process_block(&block)?;
        let enhanced_frames = enhanced.first().map(Vec::len).unwrap_or(0) as u64;
        write_block(&enhanced)?;
        input_frames = input_frames
            .checked_add(decoded_frames)
            .ok_or_else(|| "ストリーム入力フレーム数が大きすぎます".to_string())?;
        output_frames = output_frames
            .checked_add(enhanced_frames)
            .ok_or_else(|| "ストリーム出力フレーム数が大きすぎます".to_string())?;
    }
    let tail = processor.finish()?;
    output_frames = output_frames
        .checked_add(tail.first().map(Vec::len).unwrap_or(0) as u64)
        .ok_or_else(|| "ストリーム出力フレーム数が大きすぎます".to_string())?;
    write_block(&tail)?;
    if output_frames != input_frames {
        return Err(format!(
            "ストリームバックエンドが{input_frames}入力framesから{output_frames}出力framesを生成しました"
        ));
    }
    Ok(input_frames)
}

fn process_stream_file(
    request: &ProcessRequest,
    mut receipt: Option<DesktopReceiptContext>,
    control: &JobControl,
    progress: impl Fn(usize, &'static str),
) -> Result<ProcessFileResult, String> {
    check_cancelled(control)?;
    let input = Path::new(&request.input);
    let output = Path::new(&request.output);
    let maximum = checked_desktop_mib(request.options.max_process_memory_mb, "プロセスメモリ上限")?;
    let configured_temporary =
        checked_desktop_mib(request.options.max_temporary_mb, "一時領域上限")?;
    let output_format = OutputFormat::from_path(output)?;
    let encode_options = parsed_encode_options(&request.options)?;
    let mut input_session = denoize::AudioInputSession::open(input)?;
    let initial_limits = desktop_decode_limits(&request.options)?;
    let initial_info = denoize::inspect_audio_stream_session(&mut input_session, initial_limits)?;
    let spec = initial_info.output_spec;
    let channel_mask = initial_info.channel_mask;
    let encode_spec = StreamEncodeSpec::new(spec, channel_mask, initial_info.total_frames);
    output_format.validate_stream_config(encode_spec, encode_options)?;
    let auxiliary_limit = match (initial_info.total_frames, configured_temporary) {
        (None, Some(limit)) => limit / 3,
        (_, Some(limit)) => limit,
        (_, None) => StreamEncodeLimits::default().max_auxiliary_temporary_bytes(),
    };
    let encode_limits = StreamEncodeLimits::new(auxiliary_limit);
    let config = processing_config(&request.options, spec.sample_rate)?;
    let backend =
        configured_backend(&request.options.backend)?.unwrap_or_else(service::select_live_backend);
    if !StreamingBackendSession::supports(backend) {
        return Err(format!(
            "バックエンド {} は長時間ストリームに対応していません",
            service::backend_name(backend)
        ));
    }
    let backend_options = resolve_gui_backend_options(backend, &request.options)?;
    let accelerator = denoize::select_accelerator_for_options(backend, &backend_options)?;
    let base_working_set = estimate_stream_memory_bytes_checked(
        spec.channels as usize,
        request.stream_frames,
        config.frame_size,
        spec.sample_rate,
        config.profile_ms,
    )
    .map_err(|error| error.to_string())?;
    let backend_state = StreamingBackendSession::estimate_additional_bytes(
        backend,
        spec.sample_rate,
        spec.channels as usize,
        backend_options.channel_mode,
    )
    .map_err(|error| error.to_string())?;
    let vad_state = if config.vad {
        StreamingBackendSession::estimate_vad_additional_bytes(
            spec.sample_rate,
            spec.channels as usize,
            request.stream_frames,
            config.frame_size,
            config.profile_ms,
        )
        .map_err(|error| error.to_string())?
    } else {
        0
    };
    let loudness_state = if request.options.loudness_lufs.is_some() {
        denoize::loudness::estimate_streaming_loudness_bytes(
            spec.channels as usize,
            spec.sample_rate,
            request.stream_frames,
        )
        .map_err(|error| error.to_string())?
    } else {
        0
    };
    let encoder_state = denoize::estimate_stream_encode_additional_bytes(
        output_format,
        encode_spec,
        request.stream_frames,
        encode_options,
    )?;
    let checkpoint_scratch = if request.resume {
        batch_resume::STREAM_CHECKPOINT_SCRATCH_BYTES
    } else {
        0
    };
    let initial_working_set = base_working_set
        .checked_add(backend_state)
        .and_then(|bytes| bytes.checked_add(vad_state))
        .and_then(|bytes| bytes.checked_add(loudness_state))
        .and_then(|bytes| bytes.checked_add(initial_info.decoder_additional_bytes))
        .and_then(|bytes| bytes.checked_add(encoder_state))
        .and_then(|bytes| bytes.checked_add(checkpoint_scratch))
        .and_then(|bytes| {
            bytes.checked_add(denoize::estimate_backend_worker_memory_bytes(
                &backend_options,
            ))
        })
        .ok_or_else(|| "ストリームのメモリ予約量が大きすぎます".to_string())?;
    let verification_block_frames = request.stream_frames.min(DEFAULT_STREAM_BLOCK_FRAMES);
    let initial_verification_working_set = denoize::estimate_stream_output_verification_bytes(
        output_format,
        encode_spec,
        verification_block_frames,
        encode_options,
        encode_limits,
        initial_limits,
    )?;
    let initial_required_memory = initial_working_set.max(initial_verification_working_set);
    if maximum.is_some_and(|limit| initial_required_memory > limit) {
        return Err(format!(
            "ストリームには{initial_required_memory} bytes必要ですが、プロセスメモリ上限を超えます"
        ));
    }
    let metadata_limits = desktop_retained_metadata_limits(maximum, initial_required_memory);
    let decode_limits = DecodeLimits::new(metadata_limits, maximum);
    let info = denoize::inspect_audio_stream_session(&mut input_session, decode_limits)?;
    if info.format != initial_info.format
        || info.codec != initial_info.codec
        || info.output_spec != initial_info.output_spec
        || info.channel_mask != initial_info.channel_mask
        || info.total_frames != initial_info.total_frames
        || info.max_decoder_frames != initial_info.max_decoder_frames
    {
        return Err("事前検査中にストリーム入力形状が変化しました".into());
    }
    let working_set = base_working_set
        .checked_add(backend_state)
        .and_then(|bytes| bytes.checked_add(vad_state))
        .and_then(|bytes| bytes.checked_add(loudness_state))
        .and_then(|bytes| bytes.checked_add(info.decoder_additional_bytes))
        .and_then(|bytes| bytes.checked_add(encoder_state))
        .and_then(|bytes| bytes.checked_add(checkpoint_scratch))
        .and_then(|bytes| {
            bytes.checked_add(denoize::estimate_backend_worker_memory_bytes(
                &backend_options,
            ))
        })
        .ok_or_else(|| "ストリームのメモリ予約量が大きすぎます".to_string())?;
    let verification_working_set = denoize::estimate_stream_output_verification_bytes(
        output_format,
        encode_spec,
        verification_block_frames,
        encode_options,
        encode_limits,
        decode_limits,
    )?;
    if maximum.is_some_and(|limit| working_set.max(verification_working_set) > limit) {
        return Err(format!(
            "ストリームには{} bytes必要ですが、プロセスメモリ上限を超えます",
            working_set.max(verification_working_set)
        ));
    }

    let resolved = service::ResolvedProcessingOptions {
        backend,
        denoiser: config.clone(),
        backend_options: backend_options.clone(),
        accelerator,
        loudness_lufs: request.options.loudness_lufs,
        true_peak_dbtp: request.options.true_peak_dbtp,
    };
    resolved.validate_config()?;
    let metadata_policy = if request.options.preserve_metadata {
        MetadataPolicy::Preserve
    } else {
        MetadataPolicy::Drop
    };
    let execution_identity =
        if request.resume || receipt.is_some() || request.expected_input_fingerprint.is_some() {
            let input_fingerprint = batch_resume::fingerprint_input_session(&mut input_session)?;
            let model = if request.resume {
                batch_resume::resumable_consumed_model(&resolved)?
            } else {
                batch_resume::consumed_model(&resolved)?
            };
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
            validate_expected_preview_binding(request, input_fingerprint, base_recipe)?;
            let recipe =
                batch_resume::stream_recipe_digest(base_recipe, request.stream_frames, info)?;
            Some((input_fingerprint, recipe, model))
        } else {
            None
        };
    let metadata = if request.options.preserve_metadata {
        input_session.read_metadata_with_limits(metadata_limits)?
    } else {
        None
    };
    let metadata_bytes = metadata
        .as_ref()
        .map(denoize::metadata::Metadata::estimated_memory_bytes)
        .unwrap_or(0);
    let temporary_reservation = desktop_stream_temporary_bytes(
        info,
        output_format,
        encode_spec,
        encode_options,
        encode_limits,
        configured_temporary,
        request.resume,
        request.options.loudness_lufs.is_some(),
        metadata_bytes,
    )?;
    let temporary_bytes = temporary_reservation.total_bytes;
    let governor = desktop_resource_governor(&request.options, 1)?;
    let mut worker_request = ResourceRequest::worker(
        working_set
            .checked_add(metadata_bytes)
            .ok_or_else(|| "ストリームのメモリ予約量が大きすぎます".to_string())?
            .max(verification_working_set),
        temporary_bytes,
    );
    if accelerator.effective() != denoize::AcceleratorRuntime::Cpu {
        let gpu_memory = working_set
            .checked_mul(2)
            .and_then(|bytes| {
                bytes.checked_add(denoize::estimate_backend_worker_gpu_memory_bytes(
                    &backend_options,
                ))
            })
            .ok_or_else(|| "ストリームのGPU予約量が大きすぎます".to_string())?;
        worker_request = worker_request
            .with_gpu_jobs(1)
            .with_gpu_memory_bytes(gpu_memory);
    }
    let request_resources = worker_request.checked_add(
        denoize::estimate_backend_session_request(backend, &backend_options, accelerator)?,
    )?;
    let _permit = governor
        .acquire_with_cancel(request_resources, || control.is_cancelled())
        .map_err(|error| {
            if control.is_cancelled() {
                "cancelled".to_string()
            } else {
                error
            }
        })?;
    let stream_evidence = match (&receipt, &execution_identity) {
        (Some(_), Some((input_fingerprint, recipe, model))) => {
            Some(DesktopStreamExecutionEvidence {
                input_fingerprint: *input_fingerprint,
                stream_info: info,
                model: model.clone(),
                recipe: *recipe,
                output_format,
                resources: request_resources,
                backend,
                accelerator,
                deterministic: backend_options.deterministic,
                metadata_policy,
            })
        }
        (Some(_), None) => {
            return Err("ストリーム実行証明の事前検査情報がありません".into());
        }
        (None, _) => None,
    };
    let inspected_resume = if request.resume && receipt.is_some() {
        let (input_fingerprint, recipe, _) = execution_identity
            .as_ref()
            .ok_or("再開ストリーム実行証明のidentityがありません")?;
        Some(batch_resume::inspect_stream_checkpoint_decision(
            output,
            *input_fingerprint,
            *recipe,
            spec,
            request.stream_frames,
            temporary_reservation.checkpoint_limit,
            request.options.force,
        )?)
    } else {
        None
    };
    if let Some(model) = execution_identity
        .as_ref()
        .and_then(|(_, _, model)| model.as_ref())
    {
        if batch_resume::fingerprint_file(&model.path)? != model.fingerprint {
            return Err(format!(
                "署名対象のストリームモデルが準備中に変更されました: {}",
                model.path.display()
            ));
        }
    }
    let mut processor = StreamingBackendSession::new_with_accelerator(
        backend,
        spec.sample_rate,
        spec.channels as usize,
        config,
        backend_options,
        accelerator,
    )?;
    let mut reader = AudioStreamReader::from_session(input_session, decode_limits)?;
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("出力フォルダを作成できません: {error}"))?;
    }
    progress(1, "ストリームを処理しています");
    let commit_mode = if request.options.force {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    };

    if request.resume {
        let (input_fingerprint, recipe, model) = execution_identity
            .clone()
            .ok_or("再開ストリームのidentityがありません")?;
        if let Some(model) = model.as_ref() {
            if batch_resume::fingerprint_file(&model.path)? != model.fingerprint {
                return Err(format!(
                    "準備中にストリームモデルが変更されました: {}",
                    model.path.display()
                ));
            }
        }
        let acquired = batch_resume::StreamCheckpointSession::acquire(
            output,
            input_fingerprint,
            recipe,
            spec,
            request.stream_frames,
            temporary_reservation.checkpoint_limit,
            request.options.force,
        )?;
        let (mut checkpoint, loaded) = match acquired {
            batch_resume::StreamCheckpointAcquire::Completed(completed) => {
                if completed.input_frames() != completed.output_frames() {
                    return Err("完了済みストリームcheckpointの入出力長が一致しません".into());
                }
                verify_desktop_stream_bound_sources(
                    &reader,
                    input,
                    execution_identity
                        .as_ref()
                        .map(|(fingerprint, _, _)| *fingerprint),
                    execution_identity
                        .as_ref()
                        .and_then(|(_, _, model)| model.as_ref()),
                    stream_evidence.as_ref(),
                )?;
                if let (Some(receipt_context), Some(evidence)) =
                    (receipt.as_mut(), stream_evidence.as_ref())
                {
                    let output_fingerprint = batch_resume::fingerprint_file(output)?;
                    let mut skipped_evidence = evidence.clone();
                    skipped_evidence.resources = ResourceRequest::new();
                    write_desktop_stream_receipt(
                        receipt_context,
                        input,
                        output,
                        &skipped_evidence,
                        completed.input_frames(),
                        output_fingerprint,
                        "none",
                        "skip",
                        "completed",
                        Some(output_fingerprint),
                    )?;
                }
                progress(4, "既存の完了済み出力を確認しました");
                if let Some(receipt_context) = receipt.take() {
                    let receipt_path = receipt_context.path;
                    control
                        .commit_fence(|| receipt_context.stage.commit(CommitMode::NoClobber))
                        .map_err(|error| {
                            format!(
                                "完了済みストリーム出力の実行証明 {} を公開できませんでした: {error}",
                                receipt_path.display()
                            )
                        })?;
                }
                return Ok(ProcessFileResult {
                    output: output.to_string_lossy().into_owned(),
                    accelerator,
                });
            }
            batch_resume::StreamCheckpointAcquire::Active(checkpoint, loaded) => {
                (checkpoint, loaded)
            }
        };
        let loaded_checkpoint = loaded;
        let mut input_frames = match loaded {
            Some(saved) => replay_desktop_stream_checkpoint(
                &mut reader,
                &mut processor,
                request.stream_frames,
                saved,
                spec.channels as usize,
                control,
            )?,
            None => 0,
        };
        let mut next_checkpoint = input_frames
            .checked_div(STREAM_CHECKPOINT_FRAMES)
            .and_then(|multiple| multiple.checked_add(1))
            .and_then(|multiple| multiple.checked_mul(STREAM_CHECKPOINT_FRAMES))
            .unwrap_or(u64::MAX);
        while let Some(block) = reader.next_block(request.stream_frames)? {
            check_cancelled(control)?;
            let frames = block.first().map(Vec::len).unwrap_or(0) as u64;
            checkpoint.append_block(&processor.process_block(&block)?)?;
            input_frames = input_frames
                .checked_add(frames)
                .ok_or_else(|| "ストリーム入力フレーム数が大きすぎます".to_string())?;
            if input_frames >= next_checkpoint {
                checkpoint.checkpoint(input_frames)?;
                denoize::fault_injection::hit("stream-checkpoint.after-periodic-sync")?;
                next_checkpoint = input_frames
                    .checked_div(STREAM_CHECKPOINT_FRAMES)
                    .and_then(|multiple| multiple.checked_add(1))
                    .and_then(|multiple| multiple.checked_mul(STREAM_CHECKPOINT_FRAMES))
                    .unwrap_or(u64::MAX);
            }
        }
        checkpoint.append_block(&processor.finish()?)?;
        verify_desktop_stream_bound_sources(
            &reader,
            input,
            Some(input_fingerprint),
            execution_identity
                .as_ref()
                .and_then(|(_, _, model)| model.as_ref()),
            stream_evidence.as_ref(),
        )?;
        let output_frames = checkpoint.output_frames();
        if output_frames != input_frames {
            return Err(format!(
                "ストリームバックエンドが{input_frames}入力framesから{output_frames}出力framesを生成しました"
            ));
        }
        drop(reader);
        drop(processor);
        check_cancelled(control)?;
        progress(3, "エンコード出力を準備しています");
        checkpoint.prepare_spool_read()?;
        let loudness_gain = if let Some(target_lufs) = request.options.loudness_lufs {
            let mut analyzer = denoize::loudness::StreamingLoudnessAnalyzer::new(
                spec.channels as usize,
                spec.sample_rate,
                channel_mask,
            )?;
            while let Some(block) = checkpoint.next_spool_block(request.stream_frames)? {
                check_cancelled(control)?;
                analyzer.add_block(&block)?;
            }
            checkpoint.prepare_spool_read()?;
            Some(analyzer.finish(target_lufs, request.options.true_peak_dbtp)?)
        } else {
            None
        };
        let mut final_encode_spec = encode_spec;
        final_encode_spec.total_frames = Some(output_frames);
        let mut transaction = AtomicOutput::new(output)?;
        let _recovery_stage = control.track_stage(&transaction)?;
        {
            let mut writer = AudioStreamWriter::new_with_limits(
                transaction.file_mut(),
                output_format,
                final_encode_spec,
                encode_options,
                encode_limits,
            )?;
            while let Some(mut block) = checkpoint.next_spool_block(request.stream_frames)? {
                check_cancelled(control)?;
                if let Some(gain) = loudness_gain {
                    gain.apply(&mut block);
                }
                writer.write_block(&block)?;
            }
            writer.finalize()?;
        }
        if output_format == OutputFormat::Wav {
            denoize::audio::write_wav_channel_mask_to_file(
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
            .map_err(|error| format!("一時出力のサイズを確認できません: {error}"))?
            .len();
        let combined = staged_bytes
            .checked_add(checkpoint.spool_len())
            .and_then(|bytes| bytes.checked_add(temporary_reservation.encoder_auxiliary_bytes))
            .ok_or_else(|| "ストリームの一時領域が大きすぎます".to_string())?;
        if combined > temporary_bytes {
            return Err(format!(
                "チェックポイントと一時出力が予約量を超えました: {combined} > {temporary_bytes} bytes"
            ));
        }
        denoize::verify_stream_output_file(
            transaction.file_mut(),
            output,
            output_format,
            final_encode_spec,
            output_frames,
            encode_options,
            decode_limits,
            verification_block_frames,
        )?;
        let output_fingerprint =
            batch_resume::fingerprint_open_file_at(transaction.file_mut(), output)?;
        match (receipt.as_mut(), stream_evidence.as_ref()) {
            (Some(receipt_context), Some(evidence)) => {
                let (publication, publication_reason) =
                    desktop_planned_publication(output, request.options.force)?;
                let reset = inspected_resume.is_some_and(|decision| decision.reset());
                let reason = if reset {
                    "forced"
                } else if loaded_checkpoint.is_some() {
                    "checkpoint"
                } else {
                    publication_reason
                };
                write_desktop_stream_receipt(
                    receipt_context,
                    input,
                    output,
                    evidence,
                    input_frames,
                    output_fingerprint,
                    publication,
                    "process",
                    reason,
                    None,
                )?;
            }
            (None, None) => {}
            _ => return Err("再開ストリーム実行証明の状態が変化しました".into()),
        }
        checkpoint.prepare_publish(input_frames, output_fingerprint)?;
        denoize::fault_injection::hit("stream-checkpoint.after-prepare-publish-sync")?;
        progress(4, "出力を確定しています");
        if let Some(receipt_context) = receipt.take() {
            let receipt_path = receipt_context.path;
            control.commit_fence(|| {
                transaction.commit(commit_mode)?;
                if let Err(error) =
                    denoize::fault_injection::hit("stream-checkpoint.after-output-publish")
                {
                    return Err(format!(
                        "ストリーム音声は確定しましたがfault injectionで停止しました: {error}"
                    ));
                }
                if injected_stop_after_desktop_stream_commit() {
                    return Err("injected stop after committed desktop stream output".into());
                }
                receipt_context
                    .stage
                    .commit(CommitMode::NoClobber)
                    .map_err(|error| {
                        format!(
                            "ストリーム音声は確定しましたが、実行証明 {} を公開できませんでした: {error}",
                            receipt_path.display()
                        )
                    })?;
                denoize::fault_injection::hit("stream-checkpoint.after-receipt-publish")
            })?;
        } else {
            control.commit(transaction, commit_mode)?;
            if let Err(error) =
                denoize::fault_injection::hit("stream-checkpoint.after-output-publish")
            {
                return Err(format!(
                    "ストリーム音声は確定しましたがfault injectionで停止しました: {error}"
                ));
            }
        }
        denoize::fault_injection::hit("stream-checkpoint.before-cleanup")?;
        if let Err(error) = checkpoint.cleanup() {
            eprintln!("denoize desktop: checkpoint cleanup failed after commit: {error}");
        }
    } else if let Some(target_lufs) = request.options.loudness_lufs {
        let spool_limit = desktop_stream_pcm_spool_limit(
            info,
            temporary_bytes,
            temporary_reservation.encoder_auxiliary_bytes,
            metadata_bytes,
        )?;
        let mut spool = StreamPcmSpool::new(spec.channels as usize, spool_limit)?;
        let input_frames = process_desktop_stream_blocks(
            &mut reader,
            &mut processor,
            request.stream_frames,
            control,
            |block| spool.write_block(block),
        )?;
        if spool.frames() != input_frames {
            return Err("ストリームラウドネスPCMのフレーム数が一致しません".into());
        }
        verify_desktop_stream_bound_sources(
            &reader,
            input,
            execution_identity
                .as_ref()
                .map(|(fingerprint, _, _)| *fingerprint),
            execution_identity
                .as_ref()
                .and_then(|(_, _, model)| model.as_ref()),
            stream_evidence.as_ref(),
        )?;
        drop(reader);
        drop(processor);
        check_cancelled(control)?;
        progress(2, "ラウドネスを測定しています");
        let loudness_gain = analyze_desktop_stream_pcm_spool(
            &mut spool,
            spec.channels as usize,
            spec.sample_rate,
            channel_mask,
            request.stream_frames,
            target_lufs,
            request.options.true_peak_dbtp,
        )?;
        let mut final_encode_spec = encode_spec;
        final_encode_spec.total_frames = Some(input_frames);
        progress(3, "エンコード出力を準備しています");
        let mut transaction = AtomicOutput::new(output)?;
        let _recovery_stage = control.track_stage(&transaction)?;
        {
            let mut writer = AudioStreamWriter::new_with_limits(
                transaction.file_mut(),
                output_format,
                final_encode_spec,
                encode_options,
                encode_limits,
            )?;
            while let Some(mut block) = spool.next_block(request.stream_frames)? {
                check_cancelled(control)?;
                loudness_gain.apply(&mut block);
                writer.write_block(&block)?;
            }
            writer.finalize()?;
        }
        if output_format == OutputFormat::Wav {
            denoize::audio::write_wav_channel_mask_to_file(
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
            .map_err(|error| format!("一時出力のサイズを確認できません: {error}"))?
            .len();
        let combined = staged_bytes
            .checked_add(spool.len())
            .and_then(|bytes| bytes.checked_add(temporary_reservation.encoder_auxiliary_bytes))
            .ok_or_else(|| "ストリームラウドネスの一時領域が大きすぎます".to_string())?;
        if combined > temporary_bytes {
            return Err(format!(
                "ラウドネスPCM、一時出力、エンコーダー補助データが予約量を超えました: {combined} > {temporary_bytes} bytes"
            ));
        }
        denoize::verify_stream_output_file(
            transaction.file_mut(),
            output,
            output_format,
            final_encode_spec,
            input_frames,
            encode_options,
            decode_limits,
            verification_block_frames,
        )?;
        match (receipt.as_mut(), stream_evidence.as_ref()) {
            (Some(receipt), Some(evidence)) => {
                let output_fingerprint =
                    batch_resume::fingerprint_open_file_at(transaction.file_mut(), output)?;
                let publication = receipt.publication;
                let reason = receipt.reason;
                write_desktop_stream_receipt(
                    receipt,
                    input,
                    output,
                    evidence,
                    input_frames,
                    output_fingerprint,
                    publication,
                    "process",
                    reason,
                    None,
                )?;
            }
            (None, None) => {}
            _ => return Err("ストリーム実行証明の状態が事前検査後に変化しました".into()),
        }
        progress(4, "出力を確定しています");
        if let Some(receipt_context) = receipt.take() {
            let receipt_path = receipt_context.path;
            control.commit_fence(|| {
                transaction.commit(commit_mode)?;
                receipt_context
                    .stage
                    .commit(CommitMode::NoClobber)
                    .map_err(|error| {
                        format!(
                            "ストリーム音声は確定しましたが、実行証明 {} を公開できませんでした: {error}",
                            receipt_path.display()
                        )
                    })
            })?;
        } else {
            control.commit(transaction, commit_mode)?;
        }
    } else {
        let mut transaction = AtomicOutput::new(output)?;
        let _recovery_stage = control.track_stage(&transaction)?;
        let mut input_frames = 0_u64;
        let mut output_frames = 0_u64;
        {
            let mut writer = AudioStreamWriter::new_with_limits(
                transaction.file_mut(),
                output_format,
                encode_spec,
                encode_options,
                encode_limits,
            )?;
            while let Some(block) = reader.next_block(request.stream_frames)? {
                check_cancelled(control)?;
                let decoded_frames = block.first().map(Vec::len).unwrap_or(0) as u64;
                let enhanced = processor.process_block(&block)?;
                let enhanced_frames = enhanced.first().map(Vec::len).unwrap_or(0) as u64;
                writer.write_block(&enhanced)?;
                input_frames = input_frames
                    .checked_add(decoded_frames)
                    .ok_or_else(|| "ストリーム入力フレーム数が大きすぎます".to_string())?;
                output_frames = output_frames
                    .checked_add(enhanced_frames)
                    .ok_or_else(|| "ストリーム出力フレーム数が大きすぎます".to_string())?;
            }
            let tail = processor.finish()?;
            output_frames = output_frames
                .checked_add(tail.first().map(Vec::len).unwrap_or(0) as u64)
                .ok_or_else(|| "ストリーム出力フレーム数が大きすぎます".to_string())?;
            writer.write_block(&tail)?;
            if output_frames != input_frames {
                return Err(format!(
                    "ストリームバックエンドが{input_frames}入力framesから{output_frames}出力framesを生成しました"
                ));
            }
            writer.finalize()?;
        }
        if output_format == OutputFormat::Wav {
            denoize::audio::write_wav_channel_mask_to_file(
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
            .map_err(|error| format!("一時出力のサイズを確認できません: {error}"))?
            .len();
        let combined = staged_bytes
            .checked_add(temporary_reservation.encoder_auxiliary_bytes)
            .ok_or_else(|| "ストリームの一時領域が大きすぎます".to_string())?;
        if combined > temporary_bytes {
            return Err(format!(
                "一時出力とエンコーダー補助データが予約量を超えました: {combined} > {temporary_bytes} bytes"
            ));
        }
        denoize::verify_stream_output_file(
            transaction.file_mut(),
            output,
            output_format,
            encode_spec,
            output_frames,
            encode_options,
            decode_limits,
            verification_block_frames,
        )?;
        verify_desktop_stream_bound_sources(
            &reader,
            input,
            execution_identity
                .as_ref()
                .map(|(fingerprint, _, _)| *fingerprint),
            execution_identity
                .as_ref()
                .and_then(|(_, _, model)| model.as_ref()),
            stream_evidence.as_ref(),
        )?;
        drop(reader);
        drop(processor);
        match (receipt.as_mut(), stream_evidence.as_ref()) {
            (Some(receipt), Some(evidence)) => {
                let output_fingerprint =
                    batch_resume::fingerprint_open_file_at(transaction.file_mut(), output)?;
                let publication = receipt.publication;
                let reason = receipt.reason;
                write_desktop_stream_receipt(
                    receipt,
                    input,
                    output,
                    evidence,
                    input_frames,
                    output_fingerprint,
                    publication,
                    "process",
                    reason,
                    None,
                )?;
            }
            (None, None) => {}
            _ => return Err("ストリーム実行証明の状態が事前検査後に変化しました".into()),
        }
        progress(4, "出力を確定しています");
        if let Some(receipt_context) = receipt.take() {
            let receipt_path = receipt_context.path;
            control.commit_fence(|| {
                transaction.commit(commit_mode)?;
                receipt_context
                    .stage
                    .commit(CommitMode::NoClobber)
                    .map_err(|error| {
                        format!(
                            "ストリーム音声は確定しましたが、実行証明 {} を公開できませんでした: {error}",
                            receipt_path.display()
                        )
                    })
            })?;
        } else {
            control.commit(transaction, commit_mode)?;
        }
    }
    Ok(ProcessFileResult {
        output: output.to_string_lossy().into_owned(),
        accelerator,
    })
}

fn process_file(
    request: &ProcessRequest,
    mut receipt: Option<DesktopReceiptContext>,
    control: &JobControl,
    progress: impl Fn(usize, &'static str),
) -> Result<ProcessFileResult, String> {
    if request.stream {
        return process_stream_file(request, receipt, control, progress);
    }
    check_cancelled(control)?;
    let input = Path::new(&request.input);
    let output = Path::new(&request.output);
    let mut input_session = denoize::AudioInputSession::open(input)?;
    let input_bytes = input_session.len();
    let decode_limits = desktop_decode_limits(&request.options)?;
    let receipt_probe = if receipt.is_some() {
        let probe =
            denoize::probe_file_from_session_with_limits(&mut input_session, decode_limits)?;
        if probe.audio_tracks != 1 || probe.codec == denoize::AudioCodec::Unknown {
            return Err("実行証明の入力には対応する音声トラックが1つ必要です".into());
        }
        Some(probe)
    } else {
        None
    };
    let bound_input_fingerprint =
        if receipt.is_some() || request.expected_input_fingerprint.is_some() {
            Some(batch_resume::fingerprint_input_session(&mut input_session)?)
        } else {
            None
        };
    if let (Some(expected), Some(actual)) =
        (request.expected_input_fingerprint, bound_input_fingerprint)
    {
        if expected != actual {
            return Err("採用したプレビューと最終処理の入力fingerprintが一致しません".into());
        }
    }
    let mut audio = read_audio_from_session_with_limits(&mut input_session, decode_limits)?;
    let decoded_working_set = estimate_audio_working_set_bytes(&audio);
    let metadata_limits =
        desktop_retained_metadata_limits(decode_limits.max_working_set_bytes, decoded_working_set);
    let metadata = if request.options.preserve_metadata {
        input_session.read_metadata_with_limits(metadata_limits)?
    } else {
        None
    };
    let metadata_bytes = metadata
        .as_ref()
        .map(denoize::metadata::Metadata::estimated_memory_bytes)
        .unwrap_or(0);
    let encode = parsed_encode_options(&request.options)?;
    let format = OutputFormat::from_path(output)?;
    format.validate_config(&audio, &encode)?;
    progress(1, "ノイズ除去を実行しています");
    check_cancelled(control)?;
    let processing = resolved_processing_options(&request.options, &audio)?;
    let bound_model = if receipt.is_some() || request.expected_recipe.is_some() {
        batch_resume::consumed_model(&processing)?
    } else {
        None
    };
    let metadata_policy = if request.options.preserve_metadata {
        MetadataPolicy::Preserve
    } else {
        MetadataPolicy::Drop
    };
    let bound_recipe = if receipt.is_some() || request.expected_recipe.is_some() {
        Some(batch_resume::recipe_digest(
            &processing,
            audio.channels(),
            format,
            encode,
            metadata_policy,
            bound_model
                .as_ref()
                .map(|model| (&model.fingerprint, model.sample_rate)),
        )?)
    } else {
        None
    };
    if let (Some(input_fingerprint), Some(recipe)) = (bound_input_fingerprint, bound_recipe) {
        validate_expected_preview_binding(request, input_fingerprint, recipe)?;
    }
    let governor = desktop_resource_governor(&request.options, 1)?;
    let worker_request = desktop_worker_request(
        input_bytes,
        &audio,
        metadata_bytes,
        decode_limits.max_working_set_bytes,
        &processing,
        true,
    )?;
    let resource_request =
        worker_request.checked_add(denoize::estimate_backend_session_request(
            processing.backend,
            &processing.backend_options,
            processing.accelerator,
        )?)?;
    let _permit = governor
        .acquire_with_cancel(resource_request, || control.is_cancelled())
        .map_err(|error| {
            if control.is_cancelled() {
                "cancelled".to_string()
            } else {
                error
            }
        })?;
    let backend_session = BackendSession::prepare_with_accelerator(
        processing.backend,
        processing.backend_options.clone(),
        processing.accelerator,
    )?;
    if let Some(model) = &bound_model {
        if batch_resume::fingerprint_file(&model.path)? != model.fingerprint {
            return Err(format!(
                "署名対象のモデルが準備中に変更されました: {}",
                model.path.display()
            ));
        }
    }
    progress(2, "ラウドネスと出力を準備しています");
    let processing_result =
        service::process_audio_resolved_with_session(&mut audio, &processing, &backend_session)?;
    check_cancelled(control)?;
    progress(3, "ファイルを書き出しています");
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("出力フォルダを作成できません: {error}"))?;
    }
    let mut transaction = AtomicOutput::new(output)?;
    let _recovery_stage = control.track_stage(&transaction)?;
    write_audio_to_file(transaction.file_mut(), format, &audio, encode)?;
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
        .map_err(|error| format!("一時出力のサイズを確認できません: {error}"))?
        .len();
    if staged_bytes > worker_request.temporary_bytes() {
        return Err(format!(
            "一時出力が予約量を超えました: {staged_bytes} > {} bytes",
            worker_request.temporary_bytes()
        ));
    }
    if let Some(input_fingerprint) = bound_input_fingerprint {
        if batch_resume::fingerprint_input_session(&mut input_session)? != input_fingerprint
            || batch_resume::fingerprint_file(input)? != input_fingerprint
        {
            return Err(format!(
                "プレビューまたは実行証明に拘束された入力が処理中に変更されました: {}",
                input.display()
            ));
        }
    }
    if let Some(model) = &bound_model {
        if batch_resume::fingerprint_file(&model.path)? != model.fingerprint {
            return Err(format!(
                "プレビューまたは実行証明に拘束されたモデルが処理中に変更されました: {}",
                model.path.display()
            ));
        }
    }
    if let Some(receipt_context) = receipt.as_mut() {
        let input_fingerprint =
            bound_input_fingerprint.ok_or("実行証明の入力fingerprintが取得されていません")?;
        let receipt_probe = receipt_probe
            .as_ref()
            .ok_or("実行証明の入力codecが確認されていません")?;
        let recipe = bound_recipe.ok_or("実行証明のrecipeが取得されていません")?;
        let output_locator = denoize::portable_file_locator(output)?;
        let item_id = denoize::execution_item_id(input_fingerprint, &output_locator, recipe)?;
        let frames = u64::try_from(audio.frames())
            .map_err(|_| "実行証明のフレーム数が大きすぎます".to_string())?;
        let planned_model = bound_model
            .as_ref()
            .map(|model| {
                Ok::<PlannedArtifact, String>(PlannedArtifact {
                    path: denoize::portable_file_locator(&model.path)?,
                    fingerprint: model.fingerprint,
                })
            })
            .transpose()?;
        let plan = ExecutionPlan::new(
            ExecutionKind::File,
            processing.backend_options.deterministic,
            desktop_metadata_policy_name(metadata_policy),
            vec![ExecutionPlanItem {
                item_id,
                input: PlannedArtifact {
                    path: denoize::portable_file_locator(input)?,
                    fingerprint: input_fingerprint,
                },
                output: PlannedOutput {
                    path: output_locator,
                    format: desktop_output_format_name(format).into(),
                    publication: receipt_context.publication.into(),
                    action: "process".into(),
                    reason: receipt_context.reason.into(),
                    existing_fingerprint: None,
                },
                model: planned_model,
                recipe,
                backend: service::backend_name(processing.backend).into(),
                accelerator: processing.accelerator.effective().name().into(),
                input_format: desktop_audio_format_name(receipt_probe.format).into(),
                input_codec: desktop_audio_codec_name(receipt_probe.codec).into(),
                channels: audio.channels() as u64,
                frames,
                sample_rate: audio.sample_rate,
                resources: desktop_planned_resources(resource_request),
            }],
        )?;
        let output_fingerprint =
            batch_resume::fingerprint_open_file_at(transaction.file_mut(), output)?;
        let plan_item = plan
            .items
            .first()
            .ok_or("単一ファイル実行計画に項目がありません")?;
        let item = ReceiptItem::from_plan_item(plan_item, output_fingerprint, "succeeded")?;
        let payload = ExecutionReceiptPayload::new(&plan, vec![item])?;
        let signed = receipt_context.key.sign(payload)?;
        write_desktop_receipt_stage(&mut receipt_context.stage, &receipt_context.path, &signed)?;
    }
    progress(4, "出力を確定しています");
    let commit_mode = if request.options.force {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    };
    if let Some(receipt_context) = receipt.take() {
        let receipt_path = receipt_context.path;
        control.commit_fence(|| {
            transaction.commit(commit_mode)?;
            receipt_context
                .stage
                .commit(CommitMode::NoClobber)
                .map_err(|error| {
                    format!(
                        "音声出力は確定しましたが、実行証明 {} を公開できませんでした: {error}",
                        receipt_path.display()
                    )
                })
        })?;
    } else {
        control.commit(transaction, commit_mode)?;
    }
    Ok(ProcessFileResult {
        output: output.to_string_lossy().into_owned(),
        accelerator: processing_result.accelerator,
    })
}

fn stage_batch_output(
    input: &Path,
    output: &Path,
    format: OutputFormat,
    encode: EncodeOptions,
    metadata_policy: MetadataPolicy,
    processing: &service::ResolvedProcessingOptions,
    backend_session: &BackendSession,
    decode_limits: DecodeLimits,
    metadata_limits: MetadataLimits,
    temporary_reservation_bytes: u64,
    control: &JobControl,
) -> Result<AtomicOutput, String> {
    check_cancelled(control)?;
    let mut input_session = denoize::AudioInputSession::open(input)?;
    let mut audio = read_audio_from_session_with_limits(&mut input_session, decode_limits)?;
    let metadata = if metadata_policy == MetadataPolicy::Preserve {
        input_session.read_metadata_with_limits(metadata_limits)?
    } else {
        None
    };
    format.validate_config(&audio, &encode)?;
    check_cancelled(control)?;
    service::process_audio_resolved_with_session(&mut audio, processing, backend_session)?;
    check_cancelled(control)?;
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("出力フォルダを作成できません: {error}"))?;
    }
    let mut transaction = AtomicOutput::new(output)?;
    write_audio_to_file(transaction.file_mut(), format, &audio, encode)?;
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
        .map_err(|error| format!("一時出力のサイズを確認できません: {error}"))?
        .len();
    if staged_bytes > temporary_reservation_bytes {
        return Err(format!(
            "一時出力が予約量を超えました: {staged_bytes} > {temporary_reservation_bytes} bytes"
        ));
    }
    Ok(transaction)
}

fn verify_prepared_batch_recipe(item: &PreparedBatchItem) -> Result<(), String> {
    let recipe = batch_resume::recipe_digest(
        &item.processing,
        item.input_channels,
        item.item.output_format,
        item.encode,
        item.metadata_policy,
        item.expectation
            .model()
            .map(|model| (&model.fingerprint, model.sample_rate)),
    )?;
    if recipe != item.expectation.recipe() {
        return Err(format!(
            "バッチ出力 {} の有効な処理設定が事前検査後に変化しました",
            item.item.output.display()
        ));
    }
    Ok(())
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
    config
        .validate_config()
        .map_err(|error| error.to_string())?;
    Ok(config)
}

fn check_cancelled(control: &JobControl) -> Result<(), String> {
    if control.is_cancelled() {
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
    emit_progress_with_accelerator(
        app, job_id, kind, status, message, current, total, started, output, error, None,
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_progress_with_accelerator(
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
    accelerator: Option<denoize::AcceleratorSelection>,
) {
    let _ = app.emit(
        "job-progress",
        job_progress(
            job_id,
            kind,
            status,
            message,
            current,
            total,
            started,
            output,
            error,
            accelerator,
        ),
    );
}

#[allow(clippy::too_many_arguments)]
fn job_progress(
    job_id: u64,
    kind: &str,
    status: &str,
    message: &str,
    current: usize,
    total: usize,
    started: Instant,
    output: Option<String>,
    error: Option<String>,
    accelerator: Option<denoize::AcceleratorSelection>,
) -> JobProgress {
    JobProgress {
        job_id,
        kind: kind.into(),
        status: status.into(),
        message: message.into(),
        current,
        total,
        fraction: current as f64 / total.max(1) as f64,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        output,
        error: error.map(DesktopError::from),
        eta_seconds: None,
        item: None,
        item_status: None,
        item_id: None,
        resume_reason: None,
        accelerator: accelerator.map(accelerator_result),
    }
}

#[allow(clippy::too_many_arguments)]
fn batch_item_progress(
    job_id: u64,
    item_status: &str,
    item: &BatchItem,
    resume_reason: Option<batch_resume::ResumeReason>,
    current: usize,
    total: usize,
    started: Instant,
    error: Option<String>,
    accelerator: denoize::AcceleratorSelection,
) -> JobProgress {
    let elapsed = started.elapsed().as_secs_f64();
    let eta =
        (current > 0).then(|| elapsed / current as f64 * total.saturating_sub(current) as f64);
    let name = item
        .input
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("audio");
    JobProgress {
        job_id,
        kind: "batch".into(),
        status: "running".into(),
        message: format!("{name}: {item_status}"),
        current,
        total,
        fraction: current as f64 / total.max(1) as f64,
        elapsed_seconds: elapsed,
        output: Some(item.output.to_string_lossy().into_owned()),
        error: error.map(DesktopError::from),
        eta_seconds: eta,
        item: Some(item.input.to_string_lossy().into_owned()),
        item_status: Some(item_status.into()),
        item_id: Some(item.item_id.as_hex()),
        resume_reason: resume_reason
            .map(batch_resume::ResumeReason::as_str)
            .map(Into::into),
        accelerator: Some(accelerator_result(accelerator)),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            accessibility_e2e_active,
            finish_accessibility_e2e,
            app_info,
            recommend_settings,
            diagnose_audio_input,
            assess_audio_inputs,
            restore_audio_input,
            restore_universal_audio_input,
            extract_target_speaker_audio,
            plan_process,
            plan_batch,
            inspect_project_manifest,
            validate_project_manifest,
            plan_project_timeline,
            save_project_execution_plan,
            assemble_project_timeline,
            create_project_bundle,
            inspect_project_bundle,
            import_project_bundle,
            save_execution_plan,
            validate_evaluation_corpus,
            run_release_evaluation,
            verify_evaluation_evidence,
            compare_evaluation_evidence,
            generate_receipt_key,
            export_receipt_public_key,
            create_receipt_policy,
            verify_execution_receipt,
            ipc_request,
            start_process,
            start_watch_folder,
            poll_watch_folder,
            stop_watch_folder,
            start_batch,
            list_recoveries,
            retry_recovery,
            discard_recovery,
            export_redacted_diagnostics,
            cancel_job,
            live_devices,
            start_live,
            stop_live,
            compare_audio,
            list_models,
            model_catalog_status,
            update_model_catalog,
            inspect_model_bundle,
            import_model_bundle,
            inspect_runtime_model_package,
            recover_model_trust_root,
            reset_model_trust_time_floor,
            model_cache_doctor,
            application_update_status,
            inspect_application_update_bundle,
            check_application_update,
            check_application_update_online,
            download_application_update_bundle,
            dry_run_application_update_bundle,
            apply_application_update_bundle,
            recover_application_update,
            confirm_application_update_startup,
            save_automation_snapshot,
            daw_plugin_info,
            neural_daw_plugin_info,
            daw_factory_preset,
            import_daw_preset,
            export_daw_preset,
            import_daw_session,
            export_daw_session,
            prune_model_cache,
            model_action,
            start_preview,
            release_preview_artifacts,
            load_gui_config,
            save_gui_config,
            classify_dropped_paths,
            save_text_file
        ])
        .setup(|app| {
            let app_state = app.state::<AppState>();
            app_state
                .diagnostics
                .record(diagnostics::DiagnosticCode::ApplicationStarted);
            match application_update_state_root(app.handle())
                .and_then(|state_root| begin_application_update_startup(&state_root))
            {
                Ok(report) => {
                    if matches!(
                        report.action.as_str(),
                        "recovered-last-known-good" | "reactivate-managed-version"
                    ) {
                        app_state
                            .diagnostics
                            .record(diagnostics::DiagnosticCode::UpdateRecovered);
                    }
                    if let Ok(mut startup) = app_state.startup_update_health.lock() {
                        *startup = Some(report);
                    }
                }
                Err(error) => {
                    app_state
                        .diagnostics
                        .record(diagnostics::DiagnosticCode::UpdateFailed);
                    eprintln!("denoize desktop: application update health check failed: {error}");
                }
            }
            if let Err(error) = preview::cleanup_preview_root() {
                eprintln!("denoize desktop: stale preview cleanup failed: {error}");
            }
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_title(&format!("denoize {}", env!("CARGO_PKG_VERSION")));
            }
            if ACCESSIBILITY_E2E_ACTIVE.load(Ordering::SeqCst) {
                let handle = app.handle().clone();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_secs(45));
                    if ACCESSIBILITY_E2E_ACTIVE.swap(false, Ordering::SeqCst) {
                        eprintln!("DENOIZE_DESKTOP_A11Y_E2E:FAIL:timeout");
                        handle.exit(124);
                    }
                });
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("failed to run denoize desktop");
}

#[cfg(test)]
mod tests {
    use super::*;

    static PROJECT_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn desktop_diagnostic_contract_is_bounded_and_closed() {
        let options = desktop_diagnostic_options(12, Some(64)).unwrap();
        assert_eq!(options.analysis_seconds(), 12);
        assert_eq!(
            options.decode_limits().max_working_set_bytes,
            Some(64 * 1024 * 1024)
        );
        assert!(desktop_diagnostic_options(0, None).is_err());
        assert!(desktop_diagnostic_options(61, None).is_err());

        let request: DiagnosticRequest = serde_json::from_value(serde_json::json!({
            "input": "recording.wav",
            "analysisSeconds": 7,
            "maxMemoryMb": 32
        }))
        .unwrap();
        assert_eq!(request.input, "recording.wav");
        assert_eq!(request.analysis_seconds, 7);
        assert!(
            serde_json::from_value::<DiagnosticRequest>(serde_json::json!({
                "input": "recording.wav",
                "analysisSeconds": 7,
                "maxMemoryMb": null,
                "unexpected": true
            }))
            .is_err()
        );
    }

    #[test]
    fn desktop_restoration_contract_is_closed_and_maps_explicit_operations() {
        let request: RestorationRequest = serde_json::from_value(serde_json::json!({
            "input": "recording.wav",
            "output": null,
            "operations": ["declip", "dereverb", "wind-plosive"],
            "detectOnly": true,
            "maxMemoryMb": 64,
            "preserveMetadata": false,
            "replace": false,
            "wpeChannelMode": "multichannel"
        }))
        .unwrap();
        let config = desktop_restoration_config(&request).unwrap();
        assert_eq!(config.mode, denoize::RestorationMode::DetectOnly);
        assert_eq!(
            config.operations,
            vec![
                denoize::RestorationOperation::Declip,
                denoize::RestorationOperation::Dereverb,
                denoize::RestorationOperation::WindPlosive,
            ]
        );
        assert_eq!(
            config.dereverb.channel_mode,
            denoize::WpeChannelMode::Multichannel
        );

        assert!(
            serde_json::from_value::<RestorationRequest>(serde_json::json!({
                "input": "recording.wav",
                "output": null,
                "operations": ["declick"],
                "detectOnly": true,
                "maxMemoryMb": null,
                "preserveMetadata": false,
                "replace": false,
                "wpeChannelMode": "independent",
                "unexpected": true
            }))
            .is_err()
        );
    }

    #[test]
    fn desktop_restoration_rejects_invalid_configuration_before_file_io() {
        let mut request = RestorationRequest {
            input: "missing.wav".into(),
            output: None,
            operations: vec!["declick".into()],
            detect_only: false,
            max_memory_mb: None,
            preserve_metadata: false,
            replace: false,
            wpe_channel_mode: "independent".into(),
        };
        assert!(run_desktop_restoration(request.clone())
            .unwrap_err()
            .contains("適用モードでは音声の保存先"));
        request.detect_only = true;
        request.operations.clear();
        assert!(run_desktop_restoration(request.clone())
            .unwrap_err()
            .contains("between 1 and 5"));
        request.operations = vec!["unknown".into()];
        assert!(run_desktop_restoration(request)
            .unwrap_err()
            .contains("不明な復元処理"));
    }

    fn universal_desktop_request() -> UniversalRestorationRequest {
        UniversalRestorationRequest {
            input: "missing.wav".into(),
            output: "restored.wav".into(),
            model_package: "model.dmp".into(),
            model_package_key: "model.pub".into(),
            model_family: "discriminative".into(),
            render_role: "primary".into(),
            allow_experimental: false,
            analysis_seconds: 12,
            minimum_degradation_score: 0.08,
            maximum_energy_gain_db: 6.0,
            maximum_peak_gain_db: 6.0,
            maximum_new_clipping_ratio: 0.0001,
            maximum_quality_regression: 5.0,
            accelerator: "cpu".into(),
            max_memory_mb: Some(256),
            preserve_metadata: true,
            replace: false,
        }
    }

    #[test]
    fn desktop_universal_restoration_contract_is_closed_and_safe_by_default() {
        let request: UniversalRestorationRequest = serde_json::from_value(serde_json::json!({
            "input": "recording.wav",
            "output": "restored.wav",
            "modelPackage": "model.dmp",
            "modelPackageKey": "model.pub",
            "modelFamily": "discriminative",
            "renderRole": "primary",
            "allowExperimental": false,
            "analysisSeconds": 12,
            "minimumDegradationScore": 0.08,
            "maximumEnergyGainDb": 6.0,
            "maximumPeakGainDb": 6.0,
            "maximumNewClippingRatio": 0.0001,
            "maximumQualityRegression": 5.0,
            "accelerator": "cpu",
            "maxMemoryMb": 256,
            "preserveMetadata": true,
            "replace": false
        }))
        .unwrap();
        let config = desktop_universal_restoration_config(&request).unwrap();
        assert_eq!(
            config.model_family,
            denoize::UniversalModelFamily::Discriminative
        );
        assert_eq!(config.render_role, denoize::UniversalRenderRole::Primary);
        assert!(!config.allow_experimental);

        assert!(
            serde_json::from_value::<UniversalRestorationRequest>(serde_json::json!({
                "input": "recording.wav",
                "output": "restored.wav",
                "modelPackage": "model.dmp",
                "modelPackageKey": "model.pub",
                "modelFamily": "discriminative",
                "renderRole": "primary",
                "allowExperimental": false,
                "analysisSeconds": 12,
                "minimumDegradationScore": 0.08,
                "maximumEnergyGainDb": 6.0,
                "maximumPeakGainDb": 6.0,
                "maximumNewClippingRatio": 0.0001,
                "maximumQualityRegression": 5.0,
                "accelerator": "cpu",
                "maxMemoryMb": null,
                "preserveMetadata": true,
                "replace": false,
                "unexpected": true
            }))
            .is_err()
        );
    }

    #[test]
    fn desktop_universal_restoration_rejects_unsafe_modes_before_file_io() {
        let mut request = universal_desktop_request();
        request.model_family = "generative".into();
        assert!(desktop_universal_restoration_config(&request)
            .unwrap_err()
            .contains("allow_experimental=true"));

        request.allow_experimental = true;
        request.render_role = "alternate".into();
        assert!(desktop_universal_restoration_config(&request).is_ok());

        request.analysis_seconds = 0;
        assert!(desktop_universal_restoration_config(&request)
            .unwrap_err()
            .contains("analysis_seconds"));
    }

    fn target_speaker_desktop_request() -> TargetSpeakerRequest {
        TargetSpeakerRequest {
            mixture: "missing-mixture.wav".into(),
            enrollment: "missing-enrollment.wav".into(),
            output: "target.wav".into(),
            model_package: "target-speaker.dmp".into(),
            model_package_key: "package.pub".into(),
            promotion_evidence: "promotion.json".into(),
            promotion_evidence_key: "evaluator.json".into(),
            minimum_present_probability: 0.9,
            minimum_absent_probability: 0.9,
            maximum_energy_gain_db: 3.0,
            maximum_peak_gain_db: 3.0,
            maximum_new_clipping_ratio: 0.0001,
            accelerator: "cpu".into(),
            max_memory_mb: Some(256),
            preserve_metadata: true,
            replace: false,
        }
    }

    #[test]
    fn desktop_target_speaker_contract_is_closed_and_safe_by_default() {
        let request: TargetSpeakerRequest = serde_json::from_value(serde_json::json!({
            "mixture": "meeting.wav",
            "enrollment": "enrollment.wav",
            "output": "target.wav",
            "modelPackage": "target-speaker.dmp",
            "modelPackageKey": "package.pub",
            "promotionEvidence": "promotion.json",
            "promotionEvidenceKey": "evaluator.json",
            "minimumPresentProbability": 0.9,
            "minimumAbsentProbability": 0.9,
            "maximumEnergyGainDb": 3.0,
            "maximumPeakGainDb": 3.0,
            "maximumNewClippingRatio": 0.0001,
            "accelerator": "cpu",
            "maxMemoryMb": 256,
            "preserveMetadata": true,
            "replace": false
        }))
        .unwrap();
        let config = desktop_target_speaker_config(&request).unwrap();
        assert_eq!(config.minimum_present_probability, 0.9);
        assert_eq!(config.minimum_absent_probability, 0.9);
        assert_eq!(config.maximum_energy_gain_db, 3.0);
        assert_eq!(config.maximum_peak_gain_db, 3.0);
        assert_eq!(config.maximum_new_clipping_ratio, 0.0001);

        assert!(serde_json::from_value::<TargetSpeakerRequest>(serde_json::json!({
            "mixture": "meeting.wav",
            "enrollment": "enrollment.wav",
            "output": "target.wav",
            "modelPackage": "target-speaker.dmp",
            "modelPackageKey": "package.pub",
            "promotionEvidence": "promotion.json",
            "promotionEvidenceKey": "evaluator.json",
            "minimumPresentProbability": 0.9,
            "minimumAbsentProbability": 0.9,
            "maximumEnergyGainDb": 3.0,
            "maximumPeakGainDb": 3.0,
            "maximumNewClippingRatio": 0.0001,
            "accelerator": "cpu",
            "maxMemoryMb": null,
            "preserveMetadata": true,
            "replace": false,
            "unexpected": true
        }))
        .is_err());
    }

    #[test]
    fn desktop_target_speaker_rejects_invalid_config_before_file_io() {
        let mut request = target_speaker_desktop_request();
        request.minimum_present_probability = 0.49;
        assert!(run_desktop_target_speaker(request)
            .unwrap_err()
            .contains("minimum_present_probability"));

        let mut request = target_speaker_desktop_request();
        request.enrollment = request.mixture.clone();
        assert!(run_desktop_target_speaker(request)
            .unwrap_err()
            .contains("別のパス"));
    }

    #[test]
    fn application_update_platform_tracks_the_packaged_bundle_type() {
        assert_eq!(
            application_update_platform_for("linux", "x86_64", Some(BundleType::AppImage), false,)
                .unwrap(),
            "linux-x86_64-appimage"
        );
        assert_eq!(
            application_update_platform_for("linux", "x86_64", Some(BundleType::Deb), false,)
                .unwrap(),
            "linux-x86_64-deb"
        );
        assert_eq!(
            application_update_platform_for("windows", "x86_64", Some(BundleType::Msi), false,)
                .unwrap(),
            "windows-x86_64-msi"
        );
        assert_eq!(
            application_update_platform_for("windows", "x86_64", Some(BundleType::Nsis), false,)
                .unwrap(),
            "windows-x86_64-nsis"
        );
        assert_eq!(
            application_update_platform_for("macos", "aarch64", Some(BundleType::Dmg), false,)
                .unwrap(),
            "darwin-aarch64-app"
        );
        assert_eq!(
            application_update_activation_for_platform("linux-x86_64-appimage").unwrap(),
            denoize::update::UpdateActivationKind::AppImage
        );
        assert_eq!(
            application_update_activation_for_platform("linux-x86_64-deb").unwrap(),
            denoize::update::UpdateActivationKind::DebPackage
        );
        assert_eq!(
            application_update_activation_for_platform("windows-x86_64-msi").unwrap(),
            denoize::update::UpdateActivationKind::MsiInstaller
        );
        assert_eq!(
            application_update_activation_for_platform("windows-x86_64-nsis").unwrap(),
            denoize::update::UpdateActivationKind::NsisInstaller
        );
        assert!(application_update_platform_for("linux", "x86_64", None, false).is_err());
        assert!(application_update_activation_for_platform("portable-test").is_err());
    }

    #[test]
    fn daw_desktop_contract_round_trips_portable_state() {
        let info = daw_plugin_info(44_100.0).unwrap();
        assert_eq!(info.plugin_id, DAW_PLUGIN_ID);
        assert_eq!(info.latency_frames, 441);
        assert_eq!(info.measured_latency_frames, 441);
        assert!(info.matches_reported);
        assert_eq!(info.realtime_allocations, 0);

        let neural = neural_daw_plugin_info(44_100.5).unwrap();
        assert_eq!(neural.plugin_id, NEURAL_DAW_PLUGIN_ID);
        assert_eq!(neural.model_id, NEURAL_DAW_MODEL_ID);
        assert_eq!(neural.chunk_frames, 442);
        assert_eq!(neural.latency_frames, 10_608);
        assert_eq!(neural.measured_latency_frames, 10_608);
        assert!(neural.matches_reported);
        assert_eq!(neural.realtime_allocations, 0);

        let directory = tempfile::tempdir().unwrap();
        let preset_path = directory.path().join("studio.json");
        let session_path = directory.path().join("session.json");
        let mut preset = daw_factory_preset("speech".into()).unwrap();
        preset.name = "Studio".into();
        preset.parameters.amount = 0.8;
        let exported = export_daw_preset(
            preset_path.to_string_lossy().into_owned(),
            preset.clone(),
            false,
        )
        .unwrap();
        assert_eq!(exported, preset);
        assert_eq!(
            import_daw_preset(preset_path.to_string_lossy().into_owned()).unwrap(),
            preset
        );

        let state = export_daw_session(
            session_path.to_string_lossy().into_owned(),
            preset,
            DawPortConfiguration::Mono,
            false,
        )
        .unwrap();
        assert_eq!(state.port_configuration, DawPortConfiguration::Mono);
        assert_eq!(
            import_daw_session(session_path.to_string_lossy().into_owned()).unwrap(),
            state
        );
    }

    struct ResetDesktopStreamHooks;

    impl Drop for ResetDesktopStreamHooks {
        fn drop(&mut self) {
            TEST_STOP_AFTER_DESKTOP_STREAM_COMMIT.with(|value| value.set(false));
        }
    }

    fn stop_after_desktop_stream_commit() -> ResetDesktopStreamHooks {
        TEST_STOP_AFTER_DESKTOP_STREAM_COMMIT.with(|value| value.set(true));
        ResetDesktopStreamHooks
    }

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
    fn desktop_errors_have_stable_codes_and_camel_case_wire_fields() {
        let error = DesktopError::from("GPU memory budget exceeded".to_string());
        assert_eq!(error.code, "resource.accelerator");
        let value = serde_json::to_value(error).unwrap();
        assert_eq!(value["code"], "resource.accelerator");
        assert_eq!(value["parameters"], serde_json::json!({}));
        assert_eq!(value["technicalDetail"], "GPU memory budget exceeded");
        assert!(value.get("technical_detail").is_none());

        let bounded = DesktopError::from("界".repeat(2_000));
        assert!(bounded.technical_detail.len() <= MAX_DESKTOP_ERROR_DETAIL_BYTES);
        assert!(bounded.technical_detail.ends_with('…'));
        assert!(bounded.is_valid());

        let mut invalid = bounded.clone();
        invalid.code = "unknown.failure".into();
        assert!(!invalid.is_valid());
        invalid.code = "operation.failed".into();
        invalid.parameters.insert("bad key".into(), "value".into());
        assert!(!invalid.is_valid());
    }

    #[test]
    fn webview_ipc_bridge_rejects_privileged_capability_and_shutdown_operations() {
        assert!(desktop_ipc_operation_allowed(&IpcOperation::Ping));
        assert!(desktop_ipc_operation_allowed(&IpcOperation::History {
            limit: 10
        }));
        assert!(!desktop_ipc_operation_allowed(&IpcOperation::Shutdown {
            force: true
        }));
        assert!(!desktop_ipc_operation_allowed(&IpcOperation::ListGrants {
            limit: 10
        }));
        assert!(!desktop_ipc_operation_allowed(&IpcOperation::RevokeGrant {
            grant_id: "grant-1".into()
        }));
    }

    #[test]
    fn accessibility_e2e_reports_are_bounded_and_unique() {
        let mut report = AccessibilityE2eReport {
            schema: "denoize-desktop-a11y-e2e-v1".into(),
            schema_version: 1,
            assertions: vec!["runtime.started".into(), "runtime.keyboard".into()],
            failures: Vec::new(),
        };
        validate_accessibility_e2e_report(&report).unwrap();
        report.assertions.push("runtime.keyboard".into());
        assert!(validate_accessibility_e2e_report(&report).is_err());
        report.assertions.pop();
        report
            .failures
            .push("x".repeat(MAX_ACCESSIBILITY_E2E_FAILURE_BYTES + 1));
        assert!(validate_accessibility_e2e_report(&report).is_err());
    }

    #[test]
    fn recommendation_request_maps_to_bounded_library_options() {
        let request = RecommendationRequest {
            input: "missing.wav".into(),
            goal: "quality".into(),
            calibrate: true,
            analysis_seconds: 7,
            max_memory_mb: Some(64),
            max_gpu_memory_mb: Some(128),
            accelerator: "cpu".into(),
            deterministic: true,
        };
        let options = desktop_recommendation_options(&request).unwrap();
        assert_eq!(options.goal(), denoize::RecommendationGoal::Quality);
        assert_eq!(options.analysis_seconds(), 7);
        assert_eq!(options.calibration_runs(), Some(3));
        assert_eq!(options.accelerator(), AcceleratorPreference::Cpu);
        assert!(options.deterministic());
        assert_eq!(
            options.decode_limits().max_working_set_bytes,
            Some(64 * BYTES_PER_MIB)
        );
        assert_eq!(options.max_gpu_memory_bytes(), Some(128 * BYTES_PER_MIB));
    }

    #[test]
    fn recommendation_request_rejects_invalid_options_before_input() {
        let request = RecommendationRequest {
            input: "missing.wav".into(),
            goal: "unknown".into(),
            calibrate: false,
            analysis_seconds: 12,
            max_memory_mb: None,
            max_gpu_memory_mb: None,
            accelerator: "cpu".into(),
            deterministic: false,
        };
        assert!(desktop_recommendation_options(&request)
            .unwrap_err()
            .contains("不明な推奨目標"));

        let mut request = request;
        request.goal = "balanced".into();
        request.max_gpu_memory_mb = Some(0);
        assert!(desktop_recommendation_options(&request)
            .unwrap_err()
            .contains("GPUメモリ上限は1 MiB以上"));
    }

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn create(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "denoize-gui-{label}-{}-{}",
                std::process::id(),
                NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self { path }
        }

        fn join(&self, name: &str) -> PathBuf {
            self.path.join(name)
        }

        fn assert_no_staged_outputs(&self) {
            let staged: Vec<_> = std::fs::read_dir(&self.path)
                .unwrap()
                .filter_map(Result::ok)
                .map(|entry| entry.file_name())
                .filter(|name| name.to_string_lossy().starts_with(".denoize-"))
                .collect();
            assert!(staged.is_empty(), "staged outputs remain: {staged:?}");
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn desktop_automation_json_export_is_one_atomic_replacement() {
        let directory = TestDirectory::create("automation-json");
        let output = directory.join("denoize-automation.json");
        std::fs::write(&output, b"old contents").unwrap();
        let json = "{\"schema\":\"denoize-automation-v1\",\"schema_version\":1}\n";

        write_automation_json(&output, json).unwrap();

        assert_eq!(std::fs::read_to_string(&output).unwrap(), json);
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn desktop_offline_bundle_commands_fail_closed_on_invalid_inputs() {
        let directory = TestDirectory::create("invalid-offline-bundle");
        let missing = directory.join("missing.dmb");
        let inspect_error = tauri::async_runtime::block_on(inspect_model_bundle(
            missing.to_string_lossy().into_owned(),
        ))
        .unwrap_err();
        assert!(
            inspect_error.technical_detail.contains("file not found"),
            "{inspect_error:?}"
        );

        let import_error = tauri::async_runtime::block_on(import_model_bundle(
            missing.to_string_lossy().into_owned(),
            "not-a-sha256".into(),
        ))
        .unwrap_err();
        assert!(
            import_error
                .technical_detail
                .contains("expected offline bundle SHA-256"),
            "{import_error:?}"
        );
    }

    fn write_test_wav(path: &Path) {
        let mut samples = Vec::with_capacity(3_200);
        for index in 0..1_600 {
            let sample = if index % 80 < 40 {
                1_000_i16
            } else {
                -1_000_i16
            };
            samples.extend(sample.to_le_bytes());
        }
        let mut wav = Vec::with_capacity(44 + samples.len());
        wav.extend(b"RIFF");
        wav.extend((36_u32 + samples.len() as u32).to_le_bytes());
        wav.extend(b"WAVEfmt ");
        wav.extend(16_u32.to_le_bytes());
        wav.extend(1_u16.to_le_bytes());
        wav.extend(1_u16.to_le_bytes());
        wav.extend(16_000_u32.to_le_bytes());
        wav.extend(32_000_u32.to_le_bytes());
        wav.extend(2_u16.to_le_bytes());
        wav.extend(16_u16.to_le_bytes());
        wav.extend(b"data");
        wav.extend((samples.len() as u32).to_le_bytes());
        wav.extend(samples);
        std::fs::write(path, wav).unwrap();
    }

    fn desktop_timeline_project_fixture(
        directory: &TestDirectory,
    ) -> (PathBuf, denoize::ProjectManifest) {
        let source_path = directory.join("source.wav");
        let manifest_path = directory.join("project.json");
        write_test_wav(&source_path);
        let inspection =
            denoize::inspect_project_source(&source_path, DecodeLimits::default()).unwrap();
        let source =
            denoize::ProjectSource::new("source", "source.wav", inspection.clone(), None).unwrap();
        let selection = denoize::ProjectSelection::new(
            "selection",
            "source",
            denoize::PresentationRegion::new(
                inspection.fingerprint,
                inspection.timescale,
                0,
                inspection.presentation_frames,
            )
            .unwrap(),
            vec![0],
            0,
            0,
            0,
        )
        .unwrap();
        let timeline = denoize::ProjectTimeline::new(
            "main",
            inspection.timescale,
            inspection.channels,
            vec![selection],
        )
        .unwrap();
        let manifest = denoize::ProjectManifest::new(
            "desktop-project",
            vec![source],
            vec![timeline],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        denoize::write_project_manifest(&manifest_path, &manifest, CommitMode::NoClobber, true)
            .unwrap();
        (manifest_path, manifest)
    }

    #[test]
    fn desktop_project_assembly_requires_the_reviewed_plan_and_signs_the_exact_output() {
        let _project_lock = PROJECT_TEST_LOCK.lock().unwrap();
        let directory = TestDirectory::create("timeline-project");
        let (manifest_path, _) = desktop_timeline_project_fixture(&directory);
        let output = directory.join("assembled.wav");
        let receipt = directory.join("assembled.receipt.json");
        let secret = directory.join("receipt-secret.json");
        let public = directory.join("receipt-public.json");
        denoize::write_new_receipt_keypair(&secret, &public).unwrap();
        let timeline_request = ProjectTimelineRequest {
            manifest: manifest_path.to_string_lossy().into_owned(),
            root: directory.path.to_string_lossy().into_owned(),
            timeline: "main".into(),
            output: output.to_string_lossy().into_owned(),
        };
        let (_, _, _, _, plan) = prepare_desktop_project_plan(&timeline_request).unwrap();

        let report =
            tauri::async_runtime::block_on(assemble_project_timeline(ProjectAssemblyRequest {
                manifest: timeline_request.manifest.clone(),
                root: timeline_request.root.clone(),
                timeline: timeline_request.timeline.clone(),
                output: timeline_request.output.clone(),
                plan: plan.clone(),
                receipt: Some(receipt.to_string_lossy().into_owned()),
                receipt_key: Some(secret.to_string_lossy().into_owned()),
            }))
            .unwrap();
        assert_eq!(report.schema, denoize::PROJECT_RENDER_SCHEMA);
        let signed = denoize::SignedProjectExecutionReceipt::from_file(&receipt).unwrap();
        let key = ReceiptPublicKey::from_file(public).unwrap();
        signed
            .verify_with_key(&key, Some(&plan), &directory.path)
            .unwrap();

        let rejected_output = directory.join("rejected.wav");
        let rejected_request = ProjectTimelineRequest {
            output: rejected_output.to_string_lossy().into_owned(),
            ..timeline_request
        };
        let (_, _, _, _, expected) = prepare_desktop_project_plan(&rejected_request).unwrap();
        let mut stale = expected.clone();
        stale.output.path = "different.wav".into();
        let error =
            tauri::async_runtime::block_on(assemble_project_timeline(ProjectAssemblyRequest {
                manifest: rejected_request.manifest,
                root: rejected_request.root,
                timeline: rejected_request.timeline,
                output: rejected_request.output,
                plan: stale,
                receipt: None,
                receipt_key: None,
            }))
            .unwrap_err();
        assert!(error.technical_detail.contains("reviewed plan"));
        assert!(!rejected_output.exists());
    }

    #[test]
    fn desktop_project_bundle_defaults_to_references_and_imports_no_clobber() {
        let _project_lock = PROJECT_TEST_LOCK.lock().unwrap();
        let directory = TestDirectory::create("timeline-project-bundle");
        let (manifest, _) = desktop_timeline_project_fixture(&directory);
        let bundle = directory.join("project.dpb");
        let destination = directory.join("imported-project");
        let info =
            tauri::async_runtime::block_on(create_project_bundle(ProjectBundleBuildRequest {
                manifest: manifest.to_string_lossy().into_owned(),
                root: directory.path.to_string_lossy().into_owned(),
                output: bundle.to_string_lossy().into_owned(),
                include_sources: false,
                source_payload_limit_mb: None,
                include_models: false,
                model_payload_limit_mb: None,
            }))
            .unwrap();
        assert!(!info.source_payloads_included);
        assert_eq!(info.source_payload_bytes, 0);
        assert!(bundle.exists());

        let inspected =
            tauri::async_runtime::block_on(inspect_project_bundle(bundle.to_string_lossy().into()))
                .unwrap();
        assert_eq!(inspected, info);
        let imported = tauri::async_runtime::block_on(import_project_bundle(
            bundle.to_string_lossy().into_owned(),
            destination.to_string_lossy().into_owned(),
        ))
        .unwrap();
        assert_eq!(imported.omitted_sources, vec!["source"]);
        assert!(destination.join("project.denoize.json").exists());
        assert!(!destination.join("source.wav").exists());

        let error = tauri::async_runtime::block_on(import_project_bundle(
            bundle.to_string_lossy().into_owned(),
            destination.to_string_lossy().into_owned(),
        ))
        .unwrap_err();
        assert!(error.technical_detail.contains("already exists"));
    }

    fn write_test_stereo_wav(path: &Path) {
        let frames = 960_u32;
        let samples = vec![0_u8; frames as usize * 2 * 2];
        let mut wav = Vec::with_capacity(44 + samples.len());
        wav.extend(b"RIFF");
        wav.extend((36_u32 + samples.len() as u32).to_le_bytes());
        wav.extend(b"WAVEfmt ");
        wav.extend(16_u32.to_le_bytes());
        wav.extend(1_u16.to_le_bytes());
        wav.extend(2_u16.to_le_bytes());
        wav.extend(48_000_u32.to_le_bytes());
        wav.extend(192_000_u32.to_le_bytes());
        wav.extend(4_u16.to_le_bytes());
        wav.extend(16_u16.to_le_bytes());
        wav.extend(b"data");
        wav.extend((samples.len() as u32).to_le_bytes());
        wav.extend(samples);
        std::fs::write(path, wav).unwrap();
    }

    fn write_loudness_test_wav(path: &Path) {
        let sample_rate = 16_000_u32;
        let frames = sample_rate as usize * 2;
        let mut samples = Vec::with_capacity(frames * 2);
        for frame in 0..frames {
            let phase = frame as f64 * std::f64::consts::TAU * 440.0 / sample_rate as f64;
            let sample = (phase.sin() * 0.25 * i16::MAX as f64).round() as i16;
            samples.extend(sample.to_le_bytes());
        }
        let mut wav = Vec::with_capacity(44 + samples.len());
        wav.extend(b"RIFF");
        wav.extend((36_u32 + samples.len() as u32).to_le_bytes());
        wav.extend(b"WAVEfmt ");
        wav.extend(16_u32.to_le_bytes());
        wav.extend(1_u16.to_le_bytes());
        wav.extend(1_u16.to_le_bytes());
        wav.extend(sample_rate.to_le_bytes());
        wav.extend((sample_rate * 2).to_le_bytes());
        wav.extend(2_u16.to_le_bytes());
        wav.extend(16_u16.to_le_bytes());
        wav.extend(b"data");
        wav.extend((samples.len() as u32).to_le_bytes());
        wav.extend(samples);
        std::fs::write(path, wav).unwrap();
    }

    fn classical_options(force: bool) -> ProcessOptions {
        let mut options = options();
        options.backend = "classical".into();
        options.preserve_metadata = false;
        options.force = force;
        options
    }

    fn process_request(input: &Path, output: &Path, options: ProcessOptions) -> ProcessRequest {
        ProcessRequest {
            input: input.to_string_lossy().into_owned(),
            output: output.to_string_lossy().into_owned(),
            expected_input_fingerprint: None,
            expected_recipe: None,
            stream: false,
            resume: false,
            stream_frames: DEFAULT_STREAM_BLOCK_FRAMES,
            receipt: None,
            receipt_key: None,
            options,
        }
    }

    fn options() -> ProcessOptions {
        ProcessOptions {
            backend: "auto".into(),
            preset: Some("hifi".into()),
            mode: Some("music".into()),
            strength: 0.4,
            adaptive_noise: false,
            vad: false,
            channel_mode: "linked".into(),
            downmix: "preserve".into(),
            loudness_lufs: None,
            true_peak_dbtp: -1.0,
            preserve_metadata: true,
            force: false,
            mp3_bitrate_kbps: 192,
            aac_bitrate_kbps: 192,
            aac_encoder: "oxide".into(),
            onnx_model: None,
            onnx_sample_rate: 16_000,
            model_package: None,
            model_package_key: None,
            sgmse_profile: "balanced".into(),
            accelerator: "cpu".into(),
            deterministic: false,
            seed: None,
            max_process_memory_mb: None,
            max_temporary_mb: None,
            max_gpu_memory_mb: None,
            max_gpu_jobs: 1,
        }
    }

    fn watch_request() -> WatchRequest {
        WatchRequest {
            input_dir: "input".into(),
            output_dir: "output".into(),
            receipt_key: "receipt-secret.json".into(),
            output_format: "flac".into(),
            recursive: true,
            settle_millis: 2_500,
            retry_initial_millis: 750,
            retry_max_millis: 30_000,
            max_attempts: 7,
            max_files: 123,
            quarantine_dir: Some("output/quarantine".into()),
            receipt_dir: Some("output/receipts".into()),
            state_path: Some("output/state.json".into()),
            options: classical_options(false),
        }
    }

    #[test]
    fn desktop_watch_request_maps_to_the_bounded_library_engine() {
        let request = watch_request();

        validate_watch_request(&request).unwrap();
        let config = desktop_watch_config(&request, Digest::from_bytes([0x57; 32]));

        assert_eq!(config.input_root(), Path::new("input"));
        assert_eq!(config.output_root(), Path::new("output"));
        assert_eq!(config.quarantine_root(), Path::new("output/quarantine"));
        assert_eq!(config.receipt_root(), Path::new("output/receipts"));
        assert_eq!(config.state_path(), Path::new("output/state.json"));
        assert_eq!(config.output_extension(), "flac");
        assert!(config.recursive());
        assert_eq!(config.settle_duration(), Duration::from_millis(2_500));
        assert_eq!(config.max_attempts(), 7);
    }

    #[test]
    fn desktop_watch_identity_tracks_audio_settings_but_not_resource_caps() {
        let request = watch_request();
        let public_key = ReceiptPublicKey {
            schema: "denoize-receipt-public-key-v1".into(),
            schema_version: 1,
            algorithm: "Ed25519".into(),
            key_id: "11".repeat(32),
            public_key_base64: "unused-by-template-hash".into(),
        };
        let base = desktop_watch_processor_template(&request);
        let base_identity = desktop_watch_processor_identity(&base, &public_key).unwrap();

        let mut resource_change = base.clone();
        resource_change.options.max_process_memory_mb = Some(512);
        resource_change.options.max_temporary_mb = Some(1_024);
        resource_change.options.max_gpu_jobs = 4;
        assert_eq!(
            desktop_watch_processor_identity(&resource_change, &public_key).unwrap(),
            base_identity
        );

        let mut audio_change = base;
        audio_change.options.strength += 0.1;
        assert_ne!(
            desktop_watch_processor_identity(&audio_change, &public_key).unwrap(),
            base_identity
        );
    }

    #[test]
    fn desktop_watch_rejects_overwrite_and_invalid_output_before_scanning() {
        let mut request = watch_request();
        request.options.force = true;
        assert!(validate_watch_request(&request)
            .unwrap_err()
            .contains("never replaces"));

        request.options.force = false;
        request.output_format = "missing".into();
        assert!(validate_watch_request(&request)
            .unwrap_err()
            .contains("unsupported output format"));
    }

    #[test]
    fn desktop_watch_cancellation_does_not_consume_an_attempt() {
        let error = classify_desktop_watch_error("cancelled".into());
        assert!(error.is_retryable());
        assert!(!error.counts_attempt());
    }

    fn gui_config() -> GuiConfig {
        GuiConfig {
            backend: "auto".into(),
            preset: "hifi".into(),
            mode: "music".into(),
            strength: 0.4,
            adaptive_noise: false,
            vad: false,
            channels: "linked".into(),
            downmix: "preserve".into(),
            loudness_lufs: None,
            true_peak_dbtp: None,
            preserve_metadata: true,
            force: false,
            mp3_bitrate_kbps: 192,
            m4a_bitrate_kbps: 192,
            aac_encoder: "oxide".into(),
            onnx_model: None,
            onnx_rate: Some(16_000),
            model_package: None,
            model_package_key: None,
            sgmse_profile: "balanced".into(),
            accelerator: "cpu".into(),
            deterministic: false,
            max_process_memory_mb: None,
            max_temporary_mb: None,
            max_gpu_memory_mb: None,
            max_gpu_jobs: 1,
        }
    }

    fn gui_config_source() -> String {
        toml::to_string_pretty(&gui_config()).unwrap()
    }

    fn batch_request() -> BatchRequest {
        BatchRequest {
            inputs: Vec::new(),
            input_dir: None,
            output_dir: "missing-output-directory".into(),
            output_format: "wav".into(),
            recursive: false,
            jobs: 1,
            resume: false,
            receipt: None,
            receipt_key: None,
            options: options(),
        }
    }

    fn test_batch_item(input: PathBuf, output: PathBuf, id: u8) -> BatchItem {
        BatchItem {
            input,
            output_format: OutputFormat::from_path(&output).unwrap_or(OutputFormat::Wav),
            output,
            item_id: Digest::from_bytes([id; 32]),
        }
    }

    fn desktop_batch_fixture(directory: &TestDirectory, resume: bool, force: bool) -> BatchRequest {
        let input = directory.join("input");
        let output = directory.join("output");
        std::fs::create_dir_all(&input).unwrap();
        std::fs::create_dir_all(&output).unwrap();
        let source = input.join("sample.wav");
        if !source.exists() {
            write_test_wav(&source);
        }
        BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_string_lossy().into_owned()),
            output_dir: output.to_string_lossy().into_owned(),
            output_format: "wav".into(),
            recursive: false,
            jobs: 1,
            resume,
            receipt: None,
            receipt_key: None,
            options: classical_options(force),
        }
    }

    #[test]
    fn desktop_batch_recipe_matches_the_frontend_parity_golden_vector() {
        let directory = TestDirectory::create("frontend-parity-golden");
        let input = directory.join("input");
        let output = directory.join("output");
        std::fs::create_dir(&input).unwrap();
        std::fs::create_dir(&output).unwrap();
        let source = input.join("stereo.wav");
        write_test_stereo_wav(&source);
        let config = parse_gui_config(
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
force = false
mp3_bitrate_kbps = 256
m4a_bitrate_kbps = 224
aac_encoder = "oxide"
sgmse_profile = "balanced"
deterministic = false
"#,
            gui_config(),
        )
        .unwrap();
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_string_lossy().into_owned()),
            output_dir: output.to_string_lossy().into_owned(),
            output_format: "mp3".into(),
            recursive: false,
            jobs: 1,
            resume: true,
            receipt: None,
            receipt_key: None,
            options: config.process_options(),
        };
        let prepared = prepare_batch_request(&request).unwrap();
        let prepared = &prepared[0];

        assert_eq!(prepared.processing.backend, Backend::Classical);
        assert!(!prepared.processing.denoiser.adaptive_noise);
        assert!(!prepared.processing.denoiser.vad);
        assert_eq!(
            prepared.processing.backend_options.channel_mode,
            ChannelMode::StereoLinked
        );
        assert_eq!(prepared.processing.loudness_lufs, Some(-16.0));
        assert_eq!(prepared.processing.true_peak_dbtp, -1.0);
        assert_eq!(prepared.input_channels, 2);
        assert_eq!(prepared.item.output_format, OutputFormat::Mp3);
        assert_eq!(prepared.encode.mp3_bitrate_kbps, 256);
        assert_eq!(prepared.metadata_policy, MetadataPolicy::Drop);
        assert!(prepared.expectation.model().is_none());
        assert_eq!(
            prepared.expectation.recipe().as_hex(),
            FRONTEND_PARITY_RECIPE_HEX
        );
        assert_eq!(prepared.expectation.item_id(), prepared.item.item_id);

        let fixed_item_id = batch_resume::item_identity(
            Path::new("/denoize/frontend-parity/input/stereo.wav"),
            Path::new("stereo.wav"),
            Path::new("stereo.mp3"),
            OutputFormat::Mp3,
        );
        assert_eq!(fixed_item_id.as_hex(), FRONTEND_PARITY_ITEM_ID_HEX);
    }

    fn publish_planned_item(
        session: &BatchSession,
        item: &PlannedBatchItem,
    ) -> Result<batch_resume::FileFingerprint, String> {
        let ResumeDecision::Process { commit_mode, .. } = item.decision else {
            return Err("test item unexpectedly planned as a skip".into());
        };
        let control = JobControl::default();
        let transaction = stage_batch_output(
            &item.prepared.item.input,
            &item.prepared.item.output,
            item.prepared.item.output_format,
            item.prepared.encode,
            item.prepared.metadata_policy,
            &item.prepared.processing,
            &item.prepared.backend_session,
            item.prepared.decode_limits,
            item.prepared.metadata_limits,
            item.prepared.resource_request.temporary_bytes(),
            &control,
        )?;
        verify_prepared_batch_recipe(&item.prepared)?;
        control
            .commit_fence(|| session.publish(&item.prepared.expectation, transaction, commit_mode))
    }

    fn complete_desktop_batch(request: &BatchRequest) {
        let prepared = prepare_batch_request(request).unwrap();
        let session =
            BatchSession::acquire(Path::new(&request.output_dir), request.resume).unwrap();
        let planned = plan_batch_items(&session, prepared, request.options.force).unwrap();
        session.activate().unwrap();
        for item in &planned {
            if matches!(item.decision, ResumeDecision::Process { .. }) {
                publish_planned_item(&session, item).unwrap();
            }
        }
    }

    #[test]
    fn desktop_file_plan_is_read_only_and_portable() {
        let directory = TestDirectory::create("execution-file-plan");
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        write_test_wav(&input);
        let request = process_request(&input, &output, classical_options(false));

        let plan = build_process_execution_plan(&request).unwrap();

        assert_eq!(plan.kind, ExecutionKind::File);
        assert_eq!(plan.items.len(), 1);
        assert_eq!(plan.items[0].input.path, "input.wav");
        assert_eq!(plan.items[0].output.path, "output.wav");
        assert!(!output.exists());
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn adopted_preview_binds_the_final_input_and_recipe() {
        let directory = TestDirectory::create("adopted-preview-binding");
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        write_test_wav(&input);
        let mut request = process_request(&input, &output, classical_options(false));
        let plan = build_process_execution_plan(&request).unwrap();
        let item = plan.items.first().unwrap();
        request.expected_input_fingerprint = Some(item.input.fingerprint);
        request.expected_recipe = Some(item.recipe);

        validate_request(&request).unwrap();
        process_file(&request, None, &JobControl::default(), |_, _| {}).unwrap();

        assert!(output.is_file());
        directory.assert_no_staged_outputs();

        let mismatched_output = directory.join("mismatched.wav");
        let mut mismatched = process_request(&input, &mismatched_output, classical_options(false));
        mismatched.expected_input_fingerprint = Some(item.input.fingerprint);
        mismatched.expected_recipe = Some("00".repeat(32).parse().unwrap());
        let error = process_file(&mismatched, None, &JobControl::default(), |_, _| {}).unwrap_err();
        assert!(error.contains("recipe"), "{error}");
        assert!(!mismatched_output.exists());
        directory.assert_no_staged_outputs();

        mismatched.expected_recipe = None;
        validate_request(&mismatched).unwrap();
        mismatched.expected_input_fingerprint = None;
        mismatched.expected_recipe = Some(item.recipe);
        assert!(validate_request(&mismatched)
            .unwrap_err()
            .contains("期待入力fingerprint"));
    }

    #[test]
    fn adopted_preview_recipe_also_binds_a_streaming_final_job() {
        let directory = TestDirectory::create("adopted-preview-stream-binding");
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        write_test_wav(&input);
        let mut request = process_request(&input, &output, classical_options(false));
        let preview_plan = build_process_execution_plan(&request).unwrap();
        let preview_item = preview_plan.items.first().unwrap();
        request.expected_input_fingerprint = Some(preview_item.input.fingerprint);
        request.expected_recipe = Some(preview_item.recipe);
        request.stream = true;
        request.stream_frames = 113;

        validate_request(&request).unwrap();
        let stream_plan = build_process_execution_plan(&request).unwrap();
        assert_eq!(stream_plan.kind, ExecutionKind::Stream);
        process_file(&request, None, &JobControl::default(), |_, _| {}).unwrap();

        assert!(output.is_file());
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn desktop_stream_plan_and_receipt_authenticate_the_same_encoded_output() {
        let directory = TestDirectory::create("execution-stream-plan-receipt");
        let input = directory.join("input.wav");
        let output = directory.join("output.flac");
        let secret = directory.join("receipt-secret.json");
        let public = directory.join("receipt-public.json");
        let receipt_path = directory.join("output.receipt.json");
        write_test_wav(&input);
        denoize::write_new_receipt_keypair(&secret, &public).unwrap();
        let mut request = process_request(&input, &output, classical_options(false));
        request.stream = true;
        request.stream_frames = 113;
        request.receipt = Some(receipt_path.to_string_lossy().into_owned());
        request.receipt_key = Some(secret.to_string_lossy().into_owned());

        let plan = build_process_execution_plan(&request).unwrap();
        assert_eq!(plan.kind, ExecutionKind::Stream);
        assert_eq!(plan.schema_version, 2);
        assert_eq!(plan.items.len(), 1);
        assert_eq!(plan.items[0].input.path, "input.wav");
        assert_eq!(plan.items[0].output.path, "output.flac");
        assert!(!output.exists());
        assert!(!receipt_path.exists());
        directory.assert_no_staged_outputs();

        validate_request(&request).unwrap();
        let receipt = prepare_process_receipt(&request).unwrap();
        process_file(&request, receipt, &JobControl::default(), |_, _| {}).unwrap();

        let signed = SignedExecutionReceipt::from_file(&receipt_path).unwrap();
        let key = denoize::ReceiptPublicKey::from_file(&public).unwrap();
        let report = signed
            .verify_with_key(
                &key,
                Some(&plan),
                &receipt_path,
                Some(directory.path.as_path()),
            )
            .unwrap();
        assert_eq!(report.kind, ExecutionKind::Stream);
        assert_eq!(report.schema_version, 2);
        assert_eq!(report.verified_items.len(), 1);
        assert!(output.is_file());
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn desktop_resumed_stream_plan_is_read_only_and_matches_its_receipt() {
        let directory = TestDirectory::create("execution-resumed-stream-plan-receipt");
        let input = directory.join("input.wav");
        let output = directory.join("output.flac");
        let secret = directory.join("receipt-secret.json");
        let public = directory.join("receipt-public.json");
        let receipt_path = directory.join("output.receipt.json");
        write_test_wav(&input);
        denoize::write_new_receipt_keypair(&secret, &public).unwrap();
        let mut request = process_request(&input, &output, classical_options(false));
        request.stream = true;
        request.resume = true;
        request.stream_frames = 113;

        let cancelled = JobControl::default();
        let error = process_file(&request, None, &cancelled, |stage, _| {
            if stage == 1 {
                cancelled.cancel().unwrap();
            }
        })
        .unwrap_err();
        assert_eq!(error, "cancelled");
        let (state, spool, _) = batch_resume::stream_checkpoint_sidecar_paths(&output).unwrap();
        let state_before = std::fs::read(&state).unwrap();
        let spool_before = std::fs::read(&spool).unwrap();

        let plan = build_process_execution_plan(&request).unwrap();
        assert_eq!(plan.kind, ExecutionKind::Stream);
        assert_eq!(plan.items[0].output.action, "process");
        assert_eq!(plan.items[0].output.reason, "checkpoint");
        assert_eq!(std::fs::read(&state).unwrap(), state_before);
        assert_eq!(std::fs::read(&spool).unwrap(), spool_before);
        assert!(!output.exists());

        request.receipt = Some(receipt_path.to_string_lossy().into_owned());
        request.receipt_key = Some(secret.to_string_lossy().into_owned());
        validate_request(&request).unwrap();
        let receipt = prepare_process_receipt(&request).unwrap();
        process_file(&request, receipt, &JobControl::default(), |_, _| {}).unwrap();

        let signed = SignedExecutionReceipt::from_file(&receipt_path).unwrap();
        let key = denoize::ReceiptPublicKey::from_file(&public).unwrap();
        let report = signed
            .verify_with_key(
                &key,
                Some(&plan),
                &receipt_path,
                Some(directory.path.as_path()),
            )
            .unwrap();
        assert_eq!(report.kind, ExecutionKind::Stream);
        assert_eq!(report.verified_items[0].outcome, "succeeded");
        assert!(output.is_file());
        assert!(!state.exists());
        assert!(!spool.exists());
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn desktop_committed_stream_plan_skips_and_receipt_reconciles_after_cleanup_crash() {
        let _reset = stop_after_desktop_stream_commit();
        let directory = TestDirectory::create("execution-committed-stream-plan-receipt");
        let input = directory.join("input.wav");
        let output = directory.join("output.flac");
        let secret = directory.join("receipt-secret.json");
        let public = directory.join("receipt-public.json");
        let receipt_path = directory.join("output.receipt.json");
        write_test_wav(&input);
        denoize::write_new_receipt_keypair(&secret, &public).unwrap();
        let mut request = process_request(&input, &output, classical_options(false));
        request.stream = true;
        request.resume = true;
        request.stream_frames = 113;
        request.receipt = Some(receipt_path.to_string_lossy().into_owned());
        request.receipt_key = Some(secret.to_string_lossy().into_owned());

        validate_request(&request).unwrap();
        let receipt = prepare_process_receipt(&request).unwrap();
        let error = process_file(&request, receipt, &JobControl::default(), |_, _| {}).unwrap_err();
        assert!(error.contains("injected stop after committed desktop stream output"));
        assert!(output.is_file());
        assert!(!receipt_path.exists());
        let (state, spool, _) = batch_resume::stream_checkpoint_sidecar_paths(&output).unwrap();
        let state_before = std::fs::read(&state).unwrap();
        let spool_before = std::fs::read(&spool).unwrap();

        let plan = build_process_execution_plan(&request).unwrap();
        assert_eq!(plan.kind, ExecutionKind::Stream);
        assert_eq!(plan.items[0].output.action, "skip");
        assert_eq!(plan.items[0].output.publication, "none");
        assert_eq!(plan.items[0].output.reason, "completed");
        assert!(plan.items[0].output.existing_fingerprint.is_some());
        assert_eq!(std::fs::read(&state).unwrap(), state_before);
        assert_eq!(std::fs::read(&spool).unwrap(), spool_before);

        let receipt = prepare_process_receipt(&request).unwrap();
        process_file(&request, receipt, &JobControl::default(), |_, _| {}).unwrap();

        let signed = SignedExecutionReceipt::from_file(&receipt_path).unwrap();
        let key = denoize::ReceiptPublicKey::from_file(&public).unwrap();
        let report = signed
            .verify_with_key(
                &key,
                Some(&plan),
                &receipt_path,
                Some(directory.path.as_path()),
            )
            .unwrap();
        assert_eq!(report.kind, ExecutionKind::Stream);
        assert_eq!(report.verified_items[0].outcome, "skipped");
        assert!(!state.exists());
        assert!(!spool.exists());
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn desktop_batch_plan_does_not_create_output_or_resume_state() {
        let directory = TestDirectory::create("execution-batch-plan");
        let input = directory.join("input");
        let output = directory.join("missing-output");
        std::fs::create_dir(&input).unwrap();
        write_test_wav(&input.join("one.wav"));
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_string_lossy().into_owned()),
            output_dir: output.to_string_lossy().into_owned(),
            output_format: "wav".into(),
            recursive: false,
            jobs: 1,
            resume: true,
            receipt: None,
            receipt_key: None,
            options: classical_options(false),
        };

        let plan = build_batch_execution_plan(&request).unwrap();

        assert_eq!(plan.kind, ExecutionKind::Batch);
        assert_eq!(plan.items.len(), 1);
        assert_eq!(plan.items[0].input.path, "one.wav");
        assert_eq!(plan.items[0].output.path, "one.wav");
        assert!(!output.exists());
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn desktop_file_receipt_authenticates_the_committed_output() {
        let directory = TestDirectory::create("execution-file-receipt");
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        let secret = directory.join("receipt-secret.json");
        let public = directory.join("receipt-public.json");
        let receipt_path = directory.join("output.receipt.json");
        write_test_wav(&input);
        denoize::write_new_receipt_keypair(&secret, &public).unwrap();
        let mut request = process_request(&input, &output, classical_options(false));
        request.receipt = Some(receipt_path.to_string_lossy().into_owned());
        request.receipt_key = Some(secret.to_string_lossy().into_owned());
        validate_request(&request).unwrap();
        let receipt = prepare_process_receipt(&request).unwrap();

        process_file(&request, receipt, &JobControl::default(), |_, _| {}).unwrap();

        let signed = SignedExecutionReceipt::from_file(&receipt_path).unwrap();
        let key = denoize::ReceiptPublicKey::from_file(&public).unwrap();
        let report = signed
            .verify_with_key(&key, None, &receipt_path, Some(directory.path.as_path()))
            .unwrap();
        assert_eq!(report.kind, ExecutionKind::File);
        assert_eq!(report.verified_items.len(), 1);
        assert!(output.is_file());
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn desktop_batch_receipt_authenticates_all_committed_outputs() {
        let directory = TestDirectory::create("execution-batch-receipt");
        let mut request = desktop_batch_fixture(&directory, true, false);
        let secret = directory.join("receipt-secret.json");
        let public = directory.join("receipt-public.json");
        let receipt_path = directory.join("batch.receipt.json");
        denoize::write_new_receipt_keypair(&secret, &public).unwrap();
        request.receipt = Some(receipt_path.to_string_lossy().into_owned());
        request.receipt_key = Some(secret.to_string_lossy().into_owned());
        let unplanned = prepare_batch_receipt(&request).unwrap().unwrap();
        let prepared = prepare_batch_request(&request).unwrap();
        let batch_items = prepared
            .iter()
            .map(|item| item.item.clone())
            .collect::<Vec<_>>();
        validate_batch_receipt_output_paths(&batch_items, &unplanned).unwrap();
        let session = BatchSession::acquire(Path::new(&request.output_dir), true).unwrap();
        let planned = plan_batch_items(&session, prepared, false).unwrap();
        let plan = build_desktop_batch_plan(&request, &planned).unwrap();
        let receipt = DesktopBatchReceiptContext {
            path: unplanned.path,
            key: unplanned.key,
            stage: unplanned.stage,
            plan,
            _recovery_stage: unplanned._recovery_stage,
        };
        session.activate().unwrap();
        let outcomes = planned
            .iter()
            .map(|item| BatchItemOutcome::Completed(publish_planned_item(&session, item).unwrap()))
            .collect::<Vec<_>>();

        publish_desktop_batch_receipt(
            receipt,
            &planned,
            &outcomes,
            &request,
            &JobControl::default(),
        )
        .unwrap();

        let signed = SignedExecutionReceipt::from_file(&receipt_path).unwrap();
        let key = denoize::ReceiptPublicKey::from_file(&public).unwrap();
        let report = signed
            .verify_with_key(
                &key,
                None,
                &receipt_path,
                Some(Path::new(&request.output_dir)),
            )
            .unwrap();
        assert_eq!(report.kind, ExecutionKind::Batch);
        assert_eq!(report.verified_items.len(), planned.len());
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn desktop_batch_receipt_rejects_output_changed_after_publication() {
        let directory = TestDirectory::create("execution-batch-receipt-race");
        let mut request = desktop_batch_fixture(&directory, true, false);
        let secret = directory.join("receipt-secret.json");
        let public = directory.join("receipt-public.json");
        let receipt_path = directory.join("batch.receipt.json");
        denoize::write_new_receipt_keypair(&secret, &public).unwrap();
        request.receipt = Some(receipt_path.to_string_lossy().into_owned());
        request.receipt_key = Some(secret.to_string_lossy().into_owned());
        let unplanned = prepare_batch_receipt(&request).unwrap().unwrap();
        let prepared = prepare_batch_request(&request).unwrap();
        let session = BatchSession::acquire(Path::new(&request.output_dir), true).unwrap();
        let planned = plan_batch_items(&session, prepared, false).unwrap();
        let receipt = DesktopBatchReceiptContext {
            path: unplanned.path,
            key: unplanned.key,
            stage: unplanned.stage,
            plan: build_desktop_batch_plan(&request, &planned).unwrap(),
            _recovery_stage: unplanned._recovery_stage,
        };
        session.activate().unwrap();
        let outcomes = planned
            .iter()
            .map(|item| BatchItemOutcome::Completed(publish_planned_item(&session, item).unwrap()))
            .collect::<Vec<_>>();
        std::fs::write(&planned[0].prepared.item.output, b"externally replaced").unwrap();

        let error = publish_desktop_batch_receipt(
            receipt,
            &planned,
            &outcomes,
            &request,
            &JobControl::default(),
        )
        .unwrap_err();

        assert!(error.contains("公開後にバッチ出力が変更"), "{error}");
        assert!(!receipt_path.exists());
        directory.assert_no_staged_outputs();
    }

    fn live_request() -> LiveRequest {
        LiveRequest {
            input_device: None,
            output_device: None,
            chunk_ms: 20,
            target_latency_ms: Some(0),
            max_drift_ppm: Some(2_500),
            reconnect_timeout_ms: Some(30_000),
            backend: "auto".into(),
            options: options(),
        }
    }

    #[test]
    fn gui_options_build_a_valid_processing_configuration() {
        let config = processing_config(&options(), 48_000).unwrap();
        assert_eq!(config.strength, 0.4);
        assert!(config.transient_protect);
        let selected = service::select_backend(BackendChoice::Auto, 30.0, None);
        assert_eq!(
            Backend::parse(service::backend_name(selected)),
            Some(selected)
        );
    }

    #[test]
    fn invalid_backend_is_rejected() {
        assert!(Backend::parse("missing").is_none());
    }

    #[test]
    fn app_info_reports_named_backend_model_rates() {
        let info = app_info();
        assert_eq!(info.accelerators.len(), 3);
        assert_eq!(info.accelerators[0].name, "cpu");
        assert!(info.accelerators[0].compiled);
        assert!(info.accelerators[0].available);
        assert!(info.accelerators[0].device.is_none());
        assert!(info.accelerators[0].memory_bytes.is_none());
        assert!(info.accelerators[0].compute_capability.is_none());
        assert_eq!(info.accelerators[1].name, "metal");
        assert_eq!(info.accelerators[2].name, "cuda");
        for (name, expected_rate) in [
            ("mpsenet", 16_000),
            ("sgmse", 16_000),
            ("gtcrn", 16_000),
            ("bsrnn", 48_000),
            ("mossformer2", 48_000),
        ] {
            if let Some(backend) = info.backends.iter().find(|backend| backend.name == name) {
                assert_eq!(backend.sample_rate, Some(expected_rate), "{name}");
            }
        }
    }

    #[test]
    fn process_options_reject_non_finite_numbers() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut process = options();
            process.strength = value;
            assert!(validate_process_options(&process)
                .unwrap_err()
                .contains("強度"));

            let mut process = options();
            process.loudness_lufs = Some(value);
            assert!(validate_process_options(&process)
                .unwrap_err()
                .contains("ラウドネス"));

            let mut process = options();
            process.true_peak_dbtp = value;
            assert!(validate_process_options(&process)
                .unwrap_err()
                .contains("True Peak"));
        }
    }

    #[test]
    fn process_numeric_bounds_are_inclusive() {
        for strength in [0.0, 1.0] {
            let mut process = options();
            process.strength = strength;
            validate_process_options(&process).unwrap();
        }
        for target in [MIN_LOUDNESS_LUFS, MAX_LOUDNESS_LUFS] {
            let mut process = options();
            process.loudness_lufs = Some(target);
            validate_process_options(&process).unwrap();
        }
        for peak in [MIN_TRUE_PEAK_DBTP, MAX_TRUE_PEAK_DBTP] {
            let mut process = options();
            process.loudness_lufs = Some(-23.0);
            process.true_peak_dbtp = peak;
            validate_process_options(&process).unwrap();
        }
        if Backend::parse("onnx").is_some() {
            for sample_rate in [1, MAX_MODEL_SAMPLE_RATE_HZ] {
                let mut process = options();
                process.backend = "onnx".into();
                process.onnx_model = Some("model.onnx".into());
                process.onnx_sample_rate = sample_rate;
                validate_process_options(&process).unwrap();
            }
        }
    }

    #[test]
    fn process_numeric_values_outside_bounds_are_rejected() {
        for strength in [-f64::EPSILON, 1.0 + f64::EPSILON] {
            let mut process = options();
            process.strength = strength;
            assert!(validate_process_options(&process).is_err());
        }
        for target in [MIN_LOUDNESS_LUFS - 0.1, MAX_LOUDNESS_LUFS + 0.1] {
            let mut process = options();
            process.loudness_lufs = Some(target);
            assert!(validate_process_options(&process).is_err());
        }
        for peak in [MIN_TRUE_PEAK_DBTP - 0.1, MAX_TRUE_PEAK_DBTP + 0.1] {
            let mut process = options();
            process.loudness_lufs = Some(-23.0);
            process.true_peak_dbtp = peak;
            assert!(validate_process_options(&process).is_err());
        }
        if Backend::parse("onnx").is_some() {
            for sample_rate in [0, MAX_MODEL_SAMPLE_RATE_HZ + 1] {
                let mut process = options();
                process.backend = "onnx".into();
                process.onnx_model = Some("model.onnx".into());
                process.onnx_sample_rate = sample_rate;
                assert!(validate_process_options(&process)
                    .unwrap_err()
                    .contains("サンプルレート"));
            }
        }
        let mut process = options();
        process.aac_bitrate_kbps = u32::MAX;
        assert!(validate_process_options(&process)
            .unwrap_err()
            .contains("ビットレートが大きすぎます"));
    }

    #[test]
    fn true_peak_requires_loudness_normalization() {
        let mut process = options();
        process.true_peak_dbtp = -2.0;
        let error = validate_process_options(&process).unwrap_err();
        assert!(error.contains("ラウドネス"), "unexpected error: {error}");

        let mut config = gui_config();
        config.true_peak_dbtp = Some(-2.0);
        let error = config.validate().unwrap_err();
        assert!(error.contains("loudness_lufs"), "unexpected error: {error}");

        config.true_peak_dbtp = Some(-1.0);
        config.validate().unwrap();
    }

    #[test]
    fn selected_backend_contract_and_package_validation_are_ordered_before_processing() {
        if Backend::parse("mpsenet").is_some() {
            let mut process = options();
            process.backend = "mpsenet".into();
            process.onnx_model = Some("model-that-must-not-be-opened.onnx".into());
            process.onnx_sample_rate = 48_000;
            assert!(validate_process_options(&process)
                .unwrap_err()
                .contains("backend_options.onnx.sample_rate"));

            process.onnx_sample_rate = 16_000;
            validate_process_options(&process).unwrap();
        }

        if Backend::parse("onnx").is_some() {
            let mut process = options();
            process.backend = "onnx".into();
            process.onnx_model = None;
            assert!(validate_process_options(&process)
                .unwrap_err()
                .contains("backend_options.onnx"));

            process.onnx_model = Some("raw.onnx".into());
            process.model_package = Some("missing.dmp".into());
            process.model_package_key = Some("missing.pub".into());
            let error = validate_process_options(&process).unwrap_err();
            assert!(error.contains("同時に指定できません"), "{error}");
            assert!(!error.contains("missing.dmp"), "{error}");

            process.onnx_model = None;
            process.onnx_sample_rate = 0;
            let error = validate_process_options(&process).unwrap_err();
            assert!(error.contains("missing.pub"), "{error}");
            assert!(!error.contains("サンプルレート"), "{error}");
        }

        let mut automatic = options();
        automatic.backend = "auto".into();
        automatic.model_package = Some("missing.dmp".into());
        automatic.model_package_key = Some("missing.pub".into());
        let error = validate_process_options(&automatic).unwrap_err();
        assert!(error.contains("明示的なONNX"), "{error}");
        assert!(!error.contains("missing.dmp"), "{error}");
    }

    #[test]
    fn managed_gtcrn_ignores_caller_model_configuration() {
        let Some(backend) = Backend::parse("gtcrn") else {
            return;
        };
        let mut process = options();
        process.backend = "gtcrn".into();
        process.onnx_model = Some("caller-model-must-not-be-used.onnx".into());
        process.onnx_sample_rate = 0;

        validate_process_options(&process).unwrap();
        assert!(parsed_backend_options_for(backend, &process)
            .unwrap()
            .onnx
            .is_none());
    }

    #[test]
    fn non_external_backends_ignore_hidden_model_configuration() {
        for name in Backend::available_names().iter().copied().filter(|name| {
            Backend::parse(name).is_some_and(|backend| !service::requires_external_model(backend))
        }) {
            let backend = Backend::parse(name).unwrap();
            let mut process = options();
            process.backend = name.into();
            process.onnx_model = Some("hidden-model-must-not-be-used.onnx".into());
            process.onnx_sample_rate = 0;
            process.model_package = Some("hidden-package-must-not-be-opened.dmp".into());
            process.model_package_key = Some("hidden-key-must-not-be-opened.pub".into());

            validate_process_options(&process).unwrap();
            let parsed = parsed_backend_options_for(backend, &process).unwrap();
            assert!(parsed.onnx.is_none());
            assert!(parsed.runtime_package.is_none());
        }
    }

    #[test]
    fn unknown_ipc_option_strings_are_rejected() {
        let mutations: &[fn(&mut ProcessOptions)] = &[
            |process| process.backend = "missing".into(),
            |process| process.preset = Some("missing".into()),
            |process| process.mode = Some("missing".into()),
            |process| process.channel_mode = "missing".into(),
            |process| process.downmix = "missing".into(),
            |process| process.aac_encoder = "missing".into(),
            |process| process.sgmse_profile = "missing".into(),
            |process| process.accelerator = "missing".into(),
        ];
        for mutate in mutations {
            let mut process = options();
            mutate(&mut process);
            assert!(validate_process_options(&process).is_err());
        }

        let mut batch = batch_request();
        batch.output_format = "missing".into();
        assert!(validate_batch_request(&batch).is_err());

        let mut live = live_request();
        live.backend = "missing".into();
        assert!(validate_live_request(&live).is_err());
    }

    #[test]
    fn batch_jobs_and_live_chunk_bounds_are_enforced() {
        for jobs in [1, 32] {
            let mut batch = batch_request();
            batch.jobs = jobs;
            validate_batch_request(&batch).unwrap();
        }
        for jobs in [0, 33] {
            let mut batch = batch_request();
            batch.jobs = jobs;
            assert!(validate_batch_request(&batch)
                .unwrap_err()
                .contains("並列数"));
        }

        #[cfg(feature = "live")]
        {
            for chunk_ms in [10, 2_000] {
                let mut live = live_request();
                live.chunk_ms = chunk_ms;
                validate_live_request(&live).unwrap();
            }
            for chunk_ms in [9, 2_001] {
                let mut live = live_request();
                live.chunk_ms = chunk_ms;
                assert!(validate_live_request(&live)
                    .unwrap_err()
                    .contains("チャンク長"));
            }
            for target_latency_ms in [0, 20, 5_000] {
                let mut live = live_request();
                live.target_latency_ms = Some(target_latency_ms);
                validate_live_request(&live).unwrap();
            }
            for target_latency_ms in [1, 19, 5_001] {
                let mut live = live_request();
                live.target_latency_ms = Some(target_latency_ms);
                assert!(validate_live_request(&live)
                    .unwrap_err()
                    .contains("レイテンシ"));
            }
            for max_drift_ppm in [0, 10_000] {
                let mut live = live_request();
                live.max_drift_ppm = Some(max_drift_ppm);
                validate_live_request(&live).unwrap();
            }
            let mut live = live_request();
            live.max_drift_ppm = Some(10_001);
            assert!(validate_live_request(&live)
                .unwrap_err()
                .contains("ドリフト"));
            for reconnect_timeout_ms in [0, 300_000] {
                let mut live = live_request();
                live.reconnect_timeout_ms = Some(reconnect_timeout_ms);
                validate_live_request(&live).unwrap();
            }
            let mut live = live_request();
            live.reconnect_timeout_ms = Some(300_001);
            assert!(validate_live_request(&live).unwrap_err().contains("再接続"));
        }
    }

    #[cfg(feature = "live")]
    #[test]
    fn live_event_metrics_keep_the_flat_camel_case_ipc_contract() {
        let event = LiveEvent {
            status: "running",
            connection_state: "priming",
            message: "再生キューを準備中".into(),
            metrics: LiveEventMetrics {
                input_sample_rate: 44_100,
                output_sample_rate: 48_000,
                target_queue_frames: 3_840,
                estimated_total_latency_ms: 91.25,
                drift_correction_ppm: 125.0,
                device_generation: 2,
                ..LiveEventMetrics::default()
            },
            accelerator: None,
            error: None,
        };
        let value = serde_json::to_value(event).unwrap();
        assert_eq!(value["connectionState"], "priming");
        assert_eq!(value["inputSampleRate"], 44_100);
        assert_eq!(value["outputSampleRate"], 48_000);
        assert_eq!(value["targetQueueFrames"], 3_840);
        assert_eq!(value["estimatedTotalLatencyMs"], 91.25);
        assert_eq!(value["driftCorrectionPpm"], 125.0);
        assert_eq!(value["deviceGeneration"], 2);
        assert!(value.get("metrics").is_none());
    }

    #[cfg(feature = "live")]
    #[test]
    fn non_live_backends_are_rejected_before_starting_a_session() {
        let Some(name) = Backend::available_names()
            .iter()
            .copied()
            .find(|name| !matches!(*name, "classical" | "rnnoise" | "gtcrn"))
        else {
            return;
        };
        let mut live = live_request();
        live.backend = name.into();
        let error = validate_live_request(&live).unwrap_err();
        assert!(error.contains("ライブ処理"), "unexpected error: {error}");
    }

    #[test]
    fn invalid_ipc_options_precede_io_and_preserve_state_and_output() {
        let directory = TestDirectory::create("invalid-ipc");
        let missing_input = directory.join("missing.wav");
        let output = directory.join("output.wav");
        std::fs::write(&output, b"existing output").unwrap();
        let state = AppState::default();
        let mut process = classical_options(false);
        process.strength = f64::NAN;

        let request = process_request(&missing_input, &output, process);
        let error = validate_request(&request).unwrap_err();

        assert!(error.contains("強度"));
        assert_eq!(std::fs::read(&output).unwrap(), b"existing output");
        assert!(state.jobs.lock().unwrap().is_empty());
        assert!(state.live.lock().unwrap().is_none());
        directory.assert_no_staged_outputs();

        let mut batch = batch_request();
        batch.output_dir = directory
            .join("missing-output")
            .to_string_lossy()
            .into_owned();
        batch.jobs = 0;
        assert!(prepare_batch_request(&batch)
            .unwrap_err()
            .contains("並列数"));
        assert!(!Path::new(&batch.output_dir).exists());
        assert!(state.jobs.lock().unwrap().is_empty());
    }

    #[test]
    fn stream_request_validation_rejects_incompatible_options() {
        let directory = TestDirectory::create("stream-validation");
        let input = directory.join("input.wav");
        write_test_wav(&input);

        let mut request = process_request(
            &input,
            &directory.join("output.flac"),
            classical_options(false),
        );
        request.stream = true;
        validate_request(&request).unwrap();

        request.stream_frames = 0;
        assert!(validate_request(&request)
            .unwrap_err()
            .contains("ストリームブロック"));

        request.stream_frames = DEFAULT_STREAM_BLOCK_FRAMES;
        request.options.vad = true;
        validate_request(&request).unwrap();

        request.options.loudness_lufs = Some(-24.0);
        request.options.true_peak_dbtp = -1.0;
        validate_request(&request).unwrap();

        request.stream = false;
        request.resume = true;
        assert!(validate_request(&request)
            .unwrap_err()
            .contains("長時間ストリーム"));
    }

    #[test]
    fn desktop_stream_vad_preserves_presentation_length() {
        let directory = TestDirectory::create("stream-vad");
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        write_test_wav(&input);
        let original = read_audio(&input).unwrap();
        let mut options = classical_options(false);
        options.vad = true;
        let mut request = process_request(&input, &output, options);
        request.stream = true;
        request.stream_frames = 113;

        validate_request(&request).unwrap();
        process_file(&request, None, &JobControl::default(), |_, _| {}).unwrap();

        let enhanced = read_audio(&output).unwrap();
        assert_eq!(enhanced.sample_rate, original.sample_rate);
        assert_eq!(enhanced.channels(), original.channels());
        assert_eq!(enhanced.frames(), original.frames());
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn desktop_stream_loudness_normalizes_non_resume_and_resume_outputs() {
        let directory = TestDirectory::create("stream-loudness");
        let input = directory.join("input.wav");
        write_loudness_test_wav(&input);
        let original = read_audio(&input).unwrap();

        for (name, resume) in [("plain.wav", false), ("resume.wav", true)] {
            let output = directory.join(name);
            let mut options = classical_options(false);
            options.loudness_lufs = Some(-24.0);
            options.true_peak_dbtp = -1.0;
            let mut request = process_request(&input, &output, options);
            request.stream = true;
            request.resume = resume;
            request.stream_frames = 257;

            validate_request(&request).unwrap();
            process_file(&request, None, &JobControl::default(), |_, _| {}).unwrap();

            let enhanced = read_audio(&output).unwrap();
            assert_eq!(enhanced.sample_rate, original.sample_rate);
            assert_eq!(enhanced.channels(), original.channels());
            assert_eq!(enhanced.frames(), original.frames());
            let (measured_lufs, measured_peak) = denoize::loudness::measure(&enhanced).unwrap();
            assert!(
                (measured_lufs - -24.0).abs() < 0.2,
                "{name}: {measured_lufs}"
            );
            assert!(measured_peak <= -0.8, "{name}: {measured_peak}");

            if resume {
                let (state, spool, lock) =
                    batch_resume::stream_checkpoint_sidecar_paths(&output).unwrap();
                assert!(!state.exists());
                assert!(!spool.exists());
                assert!(lock.is_file());
            }
        }
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn desktop_flac_stream_resume_writes_encoded_output_and_cleans_data_sidecars() {
        let directory = TestDirectory::create("stream-flac-resume");
        let wav = directory.join("source.wav");
        let input = directory.join("input.flac");
        let output = directory.join("output.flac");
        write_test_wav(&wav);
        let original = read_audio(&wav).unwrap();
        denoize::audio::write_audio(&input, &original, EncodeOptions::default()).unwrap();

        let mut request = process_request(&input, &output, classical_options(false));
        request.stream = true;
        request.resume = true;
        request.stream_frames = 113;
        validate_request(&request).unwrap();
        process_file(&request, None, &JobControl::default(), |_, _| {}).unwrap();

        let enhanced = read_audio(&output).unwrap();
        assert_eq!(enhanced.sample_rate, original.sample_rate);
        assert_eq!(enhanced.channels(), original.channels());
        assert_eq!(enhanced.frames(), original.frames());
        let (state, spool, lock) = batch_resume::stream_checkpoint_sidecar_paths(&output).unwrap();
        assert!(!state.exists());
        assert!(!spool.exists());
        assert!(lock.is_file());
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn file_and_live_registration_share_one_atomic_operation_slot() {
        let state = AppState::default();
        let running = Arc::new(AtomicBool::new(true));
        register_live_session(&state, Arc::clone(&running)).unwrap();
        let file_error = register_job(&state)
            .err()
            .expect("a live session must exclude a file job");
        assert!(file_error.contains("ライブ処理を停止"));

        *state.live.lock().unwrap() = None;
        let (job_id, _) = register_job(&state).unwrap();
        assert!(register_live_session(&state, running)
            .unwrap_err()
            .contains("ファイル処理の完了後"));
        state.jobs.lock().unwrap().remove(&job_id);
    }

    #[test]
    fn watch_registration_excludes_manual_file_and_live_operations() {
        let state = AppState::default();
        state.watch_active.store(true, Ordering::Release);

        assert!(register_job(&state)
            .err()
            .expect("watch automation must exclude a manual file job")
            .contains("watch-folder automation is running"));
        assert!(
            register_live_session(&state, Arc::new(AtomicBool::new(true)))
                .unwrap_err()
                .contains("watch-folder automation is running")
        );

        let (job_id, _) = register_watch_job(&state).unwrap();
        unregister_job(&state, job_id);
        state.watch_active.store(false, Ordering::Release);
        assert!(register_job(&state).is_ok());
    }

    #[test]
    fn active_job_rejection_precedes_resume_journal_mutation() {
        let directory = TestDirectory::create("active-job-resume-order");
        let state_path = directory.join(batch_resume::STATE_FILE_NAME);
        std::fs::write(&state_path, b"legacy/item.wav\n{\"version\":3").unwrap();
        let before = std::fs::read(&state_path).unwrap();
        let state = AppState::default();
        state
            .jobs
            .lock()
            .unwrap()
            .insert(999, Arc::new(JobControl::default()));
        let mut request = batch_request();
        request.output_dir = directory.path.to_string_lossy().into_owned();

        let error = register_batch_job(&state, &request)
            .err()
            .expect("an active job must reject the batch");

        assert!(error.contains("別の処理が実行中"));
        assert_eq!(std::fs::read(&state_path).unwrap(), before);
        assert_eq!(state.jobs.lock().unwrap().len(), 1);
    }

    #[test]
    fn model_action_options_deserialize_camel_case_and_build_policy() {
        let input: ModelActionOptions = serde_json::from_value(serde_json::json!({
            "offline": true,
            "sourceUrl": " https://models.example.test/model.onnx ",
            "proxyUrl": "http://proxy.example.test:8080",
            "basicUsername": "alice",
            "basicPassword": " secret "
        }))
        .unwrap();
        let (options, source) = model_action_options(Some(input)).unwrap();
        assert!(options.offline);
        assert_eq!(
            options.source_url.as_deref(),
            Some("https://models.example.test/model.onnx")
        );
        assert_eq!(
            options.proxy,
            ModelProxy::Url("http://proxy.example.test:8080".into())
        );
        match options.authentication {
            Some(ModelAuthentication::Basic { username, password }) => {
                assert_eq!(username, "alice");
                assert_eq!(password, " secret ");
            }
            _ => panic!("expected basic authentication"),
        }
        assert!(source.is_none());
    }

    #[test]
    fn model_action_options_inherit_download_environment_defaults() {
        let (options, source) = model_action_options_with_environment(None, |name| {
            Some(
                match name {
                    "DENOIZE_MODEL_OFFLINE" => "true",
                    "DENOIZE_MODEL_URL" => "https://mirror.example.test/model.onnx",
                    "DENOIZE_MODEL_PROXY" => "http://proxy.example.test:8080",
                    "DENOIZE_MODEL_BEARER_TOKEN" => "environment-token",
                    _ => return None,
                }
                .into(),
            )
        })
        .unwrap();
        assert!(options.offline);
        assert_eq!(
            options.source_url.as_deref(),
            Some("https://mirror.example.test/model.onnx")
        );
        assert_eq!(
            options.proxy,
            ModelProxy::Url("http://proxy.example.test:8080".into())
        );
        assert!(matches!(
            options.authentication,
            Some(ModelAuthentication::Bearer(ref token)) if token == "environment-token"
        ));
        assert!(source.is_none());
    }

    #[test]
    fn catalog_options_use_the_dedicated_catalog_environment_source() {
        let options = catalog_action_options_with_environment(None, |name| match name {
            "DENOIZE_MODEL_CATALOG_URL" => Some("https://catalog.example.test/catalog.json".into()),
            "DENOIZE_MODEL_URL" => Some("https://wrong.example.test/model.onnx".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(
            options.source_url.as_deref(),
            Some("https://catalog.example.test/catalog.json")
        );

        let input = ModelActionOptions {
            source_path: Some("catalog.json".into()),
            ..ModelActionOptions::default()
        };
        let error = catalog_action_options_with_environment(Some(input), |_| None).unwrap_err();
        assert!(error.contains("models catalog import"), "{error}");
    }

    #[test]
    fn model_action_options_reject_local_and_network_controls() {
        let input = ModelActionOptions {
            source_path: Some("/tmp/model.onnx".into()),
            proxy_url: Some("http://proxy.example.test:8080".into()),
            ..Default::default()
        };
        assert!(model_action_options_with_environment(Some(input), |_| {
            panic!("a local install must not read download environment variables")
        })
        .unwrap_err()
        .contains("同時に指定できません"));
    }

    #[test]
    fn model_action_options_reject_conflicting_authentication() {
        let input = ModelActionOptions {
            bearer_token: Some("token".into()),
            basic_username: Some("alice".into()),
            basic_password: Some("secret".into()),
            ..Default::default()
        };
        assert_eq!(
            model_action_options(Some(input)).unwrap_err(),
            "Bearer認証とBasic認証は同時に指定できません"
        );
    }

    #[test]
    fn model_action_options_reject_partial_basic_authentication() {
        let input = ModelActionOptions {
            basic_username: Some("alice".into()),
            ..Default::default()
        };
        assert_eq!(
            model_action_options(Some(input)).unwrap_err(),
            "Basic認証のユーザー名とパスワードは両方指定してください"
        );
    }

    #[test]
    fn model_action_options_support_direct_connections() {
        let input = ModelActionOptions {
            direct: true,
            ..Default::default()
        };
        let (options, _) = model_action_options(Some(input)).unwrap();
        assert_eq!(options.proxy, ModelProxy::Disabled);
    }

    #[test]
    fn model_action_options_reject_proxy_with_direct_connection() {
        let input = ModelActionOptions {
            proxy_url: Some("http://proxy.example.test:8080".into()),
            direct: true,
            ..Default::default()
        };
        assert_eq!(
            model_action_options(Some(input)).unwrap_err(),
            "プロキシURLと直接接続は同時に指定できません"
        );
    }

    #[test]
    fn model_cache_health_labels_are_stable() {
        assert_eq!(
            model_cache_status(denoize::models::ModelCacheModelStatus::ProvenanceInvalid),
            "provenance-invalid"
        );
        assert_eq!(
            model_cache_issue_kind(denoize::models::ModelCacheIssueKind::InvalidProvenance),
            "invalid-provenance"
        );
        assert_eq!(
            model_cache_issue_kind(denoize::models::ModelCacheIssueKind::OrphanedEntry),
            "orphaned-entry"
        );
    }

    #[test]
    fn no_force_rechecks_destination_when_committing() {
        let directory = TestDirectory::create("commit-race");
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        write_test_wav(&input);
        let request = process_request(&input, &output, classical_options(false));
        validate_request(&request).unwrap();

        let result = process_file(&request, None, &JobControl::default(), |stage, _| {
            if stage == 3 {
                std::fs::write(&output, b"racing writer").unwrap();
            }
        });

        assert!(result.unwrap_err().contains("output already exists"));
        assert_eq!(std::fs::read(&output).unwrap(), b"racing writer");
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn cancellation_before_commit_preserves_existing_output() {
        let directory = TestDirectory::create("cancel-commit");
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        write_test_wav(&input);
        std::fs::write(&output, b"existing output").unwrap();
        let request = process_request(&input, &output, classical_options(true));
        let control = Arc::new(JobControl::default());
        let worker_control = Arc::clone(&control);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            process_file(&request, None, &worker_control, |stage, _| {
                if stage == 4 {
                    ready_tx.send(()).unwrap();
                    resume_rx.recv().unwrap();
                }
            })
        });

        ready_rx.recv().unwrap();
        control.cancel().unwrap();
        resume_tx.send(()).unwrap();
        let result = worker.join().unwrap();

        assert_eq!(result.unwrap_err(), "cancelled");
        assert_eq!(std::fs::read(&output).unwrap(), b"existing output");
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn cancellation_waits_for_an_active_commit_fence() {
        let control = Arc::new(JobControl::default());
        let worker_control = Arc::clone(&control);
        let (inside_tx, inside_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let published = Arc::new(AtomicBool::new(false));
        let worker_published = Arc::clone(&published);
        let worker = std::thread::spawn(move || {
            worker_control.commit_fence(|| {
                inside_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                worker_published.store(true, Ordering::SeqCst);
                Ok(())
            })
        });

        inside_rx.recv().unwrap();
        let cancel_control = Arc::clone(&control);
        let (cancelled_tx, cancelled_rx) = std::sync::mpsc::channel();
        let canceller = std::thread::spawn(move || {
            cancel_control.cancel().unwrap();
            cancelled_tx.send(()).unwrap();
        });
        assert!(cancelled_rx.try_recv().is_err());
        let wait_started = Instant::now();
        while !control.is_cancelled() && wait_started.elapsed() < std::time::Duration::from_secs(1)
        {
            std::thread::yield_now();
        }
        assert!(
            control.is_cancelled(),
            "cancellation must be visible while the active publication finishes"
        );

        release_tx.send(()).unwrap();
        worker.join().unwrap().unwrap();
        cancelled_rx.recv().unwrap();
        canceller.join().unwrap();

        assert!(published.load(Ordering::SeqCst));
        assert!(control.is_cancelled());
    }

    #[test]
    fn shared_worker_cancel_fence_prevents_later_publication() {
        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("cancel");
        let fence_path = directory.path().join("commit.lock");
        std::fs::write(&fence_path, b"").unwrap();
        let parent_fence = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&fence_path)
            .unwrap();
        let worker_fence = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&fence_path)
            .unwrap();
        let parent = JobControl::default();
        let worker = JobControl::default();
        parent
            .install_shared_cancellation(marker.clone(), parent_fence)
            .unwrap();
        worker
            .install_shared_cancellation(marker.clone(), worker_fence)
            .unwrap();

        parent.cancel().unwrap();
        let published = AtomicBool::new(false);
        let error = worker
            .commit_fence(|| {
                published.store(true, Ordering::SeqCst);
                Ok(())
            })
            .unwrap_err();

        assert_eq!(error, "cancelled");
        assert!(!published.load(Ordering::SeqCst));
        assert!(marker.is_file());
    }

    #[test]
    fn commit_fence_propagates_shared_publisher_failures() {
        let control = JobControl::default();
        let error = control
            .commit_fence::<()>(|| Err("injected journal failure".into()))
            .unwrap_err();
        assert_eq!(error, "injected journal failure");
        assert!(!control.is_cancelled());
    }

    #[test]
    fn invalid_codec_config_precedes_processing_and_output_staging() {
        let directory = TestDirectory::create("codec-preflight");
        let input = directory.join("input.wav");
        write_test_wav(&input);
        let mut wav = std::fs::read(&input).unwrap();
        wav[24..28].copy_from_slice(&12_345_u32.to_le_bytes());
        wav[28..32].copy_from_slice(&(12_345_u32 * 2).to_le_bytes());
        std::fs::write(&input, wav).unwrap();
        let output_dir = directory.join("new-output-directory");
        let output = output_dir.join("output.mp3");
        let request = process_request(&input, &output, classical_options(false));
        let stages = Mutex::new(Vec::new());

        let error = process_file(&request, None, &JobControl::default(), |stage, _| {
            stages.lock().unwrap().push(stage);
        })
        .unwrap_err();

        assert!(
            error.contains("unsupported sample rate"),
            "unexpected error: {error}"
        );
        assert!(stages.lock().unwrap().is_empty());
        assert!(!output_dir.exists());
        directory.assert_no_staged_outputs();
    }

    #[cfg(unix)]
    #[test]
    fn legacy_gui_stage_symlink_does_not_clobber_its_target() {
        let directory = TestDirectory::create("stage-symlink");
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        let victim = directory.join("victim.bin");
        let legacy_stage = directory.join(".denoize-gui-output.wav.wav");
        write_test_wav(&input);
        std::fs::write(&victim, b"victim").unwrap();
        std::os::unix::fs::symlink(&victim, &legacy_stage).unwrap();
        let request = process_request(&input, &output, classical_options(false));

        process_file(&request, None, &JobControl::default(), |_, _| {}).unwrap();

        assert_eq!(std::fs::read(&victim).unwrap(), b"victim");
        assert!(std::fs::symlink_metadata(&legacy_stage)
            .unwrap()
            .file_type()
            .is_symlink());
        assert!(read_audio(&output).is_ok());
    }

    #[test]
    fn comparison_metrics_include_quality_and_artifact_improvements() {
        let report = |snr_db: f64, stoi: f64, musical_noise_score: f64| BenchmarkReport {
            frames: 1,
            sample_rate: 48_000,
            channels: 1,
            si_sdr_db: snr_db,
            si_snr_db: snr_db + 1.0,
            snr_db,
            segmental_snr_db: snr_db - 1.0,
            stereo_side_sdr_db: None,
            correlation_error: None,
            artifact_scores: denoize::benchmark::ArtifactReport {
                musical_noise_score,
                pumping_score: musical_noise_score + 0.1,
                transient_loss_score: musical_noise_score + 0.2,
                phase_distortion_score: None,
            },
            stoi: Some(stoi),
            pesq: None,
            visqol: Some(stoi + 1.0),
            elapsed_ms: None,
            peak_rss_bytes: None,
        };
        let comparison = ComparisonReport {
            noisy: report(2.0, 0.5, 0.4),
            enhanced: report(5.0, 0.8, 0.1),
        };
        let metrics = comparison_metric_set(&comparison);
        assert_eq!(metrics.noisy.snr_db, 2.0);
        assert_eq!(metrics.enhanced.stoi, Some(0.8));
        assert_eq!(metrics.improvement.snr_db, 3.0);
        assert!((metrics.improvement.stoi.unwrap() - 0.3).abs() < 1e-10);
        assert!((metrics.improvement.artifact_scores.musical_noise_score - 0.3).abs() < 1e-10);
        assert!((metrics.improvement.visqol.unwrap() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn batch_preflights_every_codec_before_state_or_output_changes() {
        let directory = TestDirectory::create("batch-codec-preflight");
        let input = directory.join("input");
        let output = directory.join("output");
        std::fs::create_dir(&input).unwrap();
        std::fs::create_dir(&output).unwrap();
        write_test_wav(&input.join("a-valid.wav"));
        write_test_wav(&input.join("b-invalid-rate.wav"));
        let invalid = input.join("b-invalid-rate.wav");
        let mut wav = std::fs::read(&invalid).unwrap();
        wav[24..28].copy_from_slice(&12_345_u32.to_le_bytes());
        wav[28..32].copy_from_slice(&(12_345_u32 * 2).to_le_bytes());
        std::fs::write(&invalid, wav).unwrap();
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_string_lossy().into_owned()),
            output_dir: output.to_string_lossy().into_owned(),
            output_format: "mp3".into(),
            recursive: false,
            jobs: 2,
            resume: true,
            receipt: None,
            receipt_key: None,
            options: classical_options(false),
        };

        let error = prepare_batch_request(&request).unwrap_err();

        assert!(error.contains("unsupported sample rate"), "{error}");
        assert!(!output.join("a-valid.mp3").exists());
        assert!(!output.join("b-invalid-rate.mp3").exists());
        assert!(!output.join(".denoize-state").exists());
        assert!(!output.join(".denoize-batch.lock").exists());
    }

    #[test]
    fn batch_preflights_actual_sample_rate_processing_before_outputs() {
        let directory = TestDirectory::create("batch-processing-preflight");
        let input = directory.join("input");
        let output = directory.join("output");
        std::fs::create_dir(&input).unwrap();
        std::fs::create_dir(&output).unwrap();
        write_test_wav(&input.join("a-valid.wav"));
        write_test_wav(&input.join("b-invalid-processing-rate.wav"));
        let invalid = input.join("b-invalid-processing-rate.wav");
        let mut wav = std::fs::read(&invalid).unwrap();
        let sample_rate = MAX_MODEL_SAMPLE_RATE_HZ + 1;
        wav[24..28].copy_from_slice(&sample_rate.to_le_bytes());
        wav[28..32].copy_from_slice(&(sample_rate * 2).to_le_bytes());
        std::fs::write(&invalid, wav).unwrap();
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_string_lossy().into_owned()),
            output_dir: output.to_string_lossy().into_owned(),
            output_format: "wav".into(),
            recursive: false,
            jobs: 2,
            resume: true,
            receipt: None,
            receipt_key: None,
            options: classical_options(false),
        };

        let error = prepare_batch_request(&request).unwrap_err();

        assert!(error.contains("sample_rate"), "{error}");
        assert!(!output.join("a-valid.wav").exists());
        assert!(!output.join("b-invalid-processing-rate.wav").exists());
        assert!(!output.join(".denoize-state").exists());
        assert!(!output.join(".denoize-batch.lock").exists());
    }

    #[test]
    fn batch_preflights_all_destinations_and_replacement_types() {
        let directory = TestDirectory::create("batch-output-preflight");
        let input = directory.join("input");
        let output = directory.join("output");
        std::fs::create_dir(&input).unwrap();
        std::fs::create_dir(&output).unwrap();
        write_test_wav(&input.join("a.wav"));
        write_test_wav(&input.join("b.wav"));
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_string_lossy().into_owned()),
            output_dir: output.to_string_lossy().into_owned(),
            output_format: "wav".into(),
            recursive: false,
            jobs: 2,
            resume: false,
            receipt: None,
            receipt_key: None,
            options: classical_options(false),
        };
        let items = prepare_batch_request(&request).unwrap();
        std::fs::write(output.join("b.wav"), b"existing").unwrap();

        let session = BatchSession::acquire(&output, false).unwrap();
        let error = plan_batch_items(&session, items, false).unwrap_err();
        assert!(error.contains("--force"), "{error}");
        assert!(!output.join("a.wav").exists());
        assert_eq!(std::fs::read(output.join("b.wav")).unwrap(), b"existing");
        assert!(!output.join(batch_resume::STATE_FILE_NAME).exists());
        drop(session);

        std::fs::remove_file(output.join("b.wav")).unwrap();
        std::fs::create_dir(output.join("b.wav")).unwrap();
        let items = prepare_batch_request(&request).unwrap();
        let session = BatchSession::acquire(&output, false).unwrap();
        let error = plan_batch_items(&session, items, true).unwrap_err();
        assert!(error.contains("cannot be replaced"), "{error}");
        assert!(!output.join("a.wav").exists());
    }

    #[cfg(unix)]
    #[test]
    fn linked_batch_outputs_are_never_treated_as_resumable_files() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::create("batch-resume-output-symlink");
        let input_dir = directory.join("input");
        let output_dir = directory.join("output");
        std::fs::create_dir(&input_dir).unwrap();
        std::fs::create_dir(&output_dir).unwrap();
        write_test_wav(&input_dir.join("sample.wav"));
        let victim = directory.join("victim.wav");
        let output = output_dir.join("sample.wav");
        write_test_wav(&victim);
        symlink(&victim, &output).unwrap();
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input_dir.to_string_lossy().into_owned()),
            output_dir: output_dir.to_string_lossy().into_owned(),
            output_format: "wav".into(),
            recursive: false,
            jobs: 1,
            resume: true,
            receipt: None,
            receipt_key: None,
            options: classical_options(false),
        };
        let prepared = prepare_batch_request(&request).unwrap();
        let session = BatchSession::acquire(&output_dir, true).unwrap();

        assert!(
            output.is_file(),
            "the follow-link check would incorrectly skip"
        );
        let error = plan_batch_items(&session, prepared.clone(), false).unwrap_err();
        assert!(error.contains("unsafe"), "{error}");
        let planned = plan_batch_items(&session, prepared, true).unwrap();
        assert!(matches!(
            planned[0].decision,
            ResumeDecision::Process {
                reason: batch_resume::ResumeReason::Unsafe,
                ..
            }
        ));
        assert!(std::fs::symlink_metadata(&output)
            .unwrap()
            .file_type()
            .is_symlink());
    }

    #[cfg(unix)]
    #[test]
    fn hard_linked_batch_outputs_are_never_treated_as_resumable_files() {
        use std::os::unix::fs::MetadataExt as _;

        let directory = TestDirectory::create("batch-resume-output-hardlink");
        let input_dir = directory.join("input");
        let output_dir = directory.join("output");
        std::fs::create_dir(&input_dir).unwrap();
        std::fs::create_dir(&output_dir).unwrap();
        write_test_wav(&input_dir.join("sample.wav"));
        let victim = directory.join("victim.wav");
        let output = output_dir.join("sample.wav");
        write_test_wav(&victim);
        std::fs::hard_link(&victim, &output).unwrap();
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input_dir.to_string_lossy().into_owned()),
            output_dir: output_dir.to_string_lossy().into_owned(),
            output_format: "wav".into(),
            recursive: false,
            jobs: 1,
            resume: true,
            receipt: None,
            receipt_key: None,
            options: classical_options(false),
        };
        let prepared = prepare_batch_request(&request).unwrap();
        let session = BatchSession::acquire(&output_dir, true).unwrap();

        let error = plan_batch_items(&session, prepared.clone(), false).unwrap_err();
        assert!(error.contains("unsafe"), "{error}");
        let planned = plan_batch_items(&session, prepared, true).unwrap();
        assert!(matches!(
            planned[0].decision,
            ResumeDecision::Process {
                reason: batch_resume::ResumeReason::Unsafe,
                ..
            }
        ));
        let output_metadata = std::fs::metadata(&output).unwrap();
        let victim_metadata = std::fs::metadata(&victim).unwrap();
        assert_eq!(output_metadata.dev(), victim_metadata.dev());
        assert_eq!(output_metadata.ino(), victim_metadata.ino());
        assert!(output_metadata.nlink() > 1);
    }

    #[test]
    fn batch_destination_preflight_rejects_exact_and_file_directory_collisions() {
        let directory = TestDirectory::create("batch-destination-collisions");
        let input_a = directory.join("input-a.wav");
        let input_b = directory.join("input-b.wav");
        let output = directory.join("output");
        write_test_wav(&input_a);
        write_test_wav(&input_b);
        std::fs::create_dir(&output).unwrap();

        let exact = vec![
            test_batch_item(input_a.clone(), output.join("same.wav"), 1),
            test_batch_item(input_b.clone(), output.join("same.wav"), 2),
        ];
        assert!(validate_batch_destinations(None, &exact)
            .unwrap_err()
            .contains("同じバッチ出力"));

        let file_and_directory = vec![
            test_batch_item(input_a, output.join("foo.flac"), 1),
            test_batch_item(input_b, output.join("foo.flac/bar.flac"), 2),
        ];
        assert!(validate_batch_destinations(None, &file_and_directory)
            .unwrap_err()
            .contains("ファイルとディレクトリ"));
        assert!(std::fs::read_dir(&output).unwrap().next().is_none());
    }

    #[test]
    fn case_insensitive_batch_collision_keys_are_normalized() {
        assert_eq!(
            batch_collision_key_with_case(Path::new("Output/Voice.WAV"), true),
            batch_collision_key_with_case(Path::new("output/voice.wav"), true)
        );
        assert_ne!(
            batch_collision_key_with_case(Path::new("Output/Voice.WAV"), false),
            batch_collision_key_with_case(Path::new("output/voice.wav"), false)
        );
    }

    #[cfg(unix)]
    #[test]
    fn batch_destination_preflight_rejects_symlinks_back_into_input() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::create("batch-destination-input-symlink");
        let input = directory.join("input");
        let nested = input.join("nested");
        let output = directory.join("output");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::create_dir(&output).unwrap();
        write_test_wav(&nested.join("voice.wav"));
        symlink(&nested, output.join("nested")).unwrap();
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_str().unwrap().into()),
            output_dir: output.to_str().unwrap().into(),
            output_format: "flac".into(),
            recursive: true,
            jobs: 1,
            resume: false,
            receipt: None,
            receipt_key: None,
            options: classical_options(false),
        };

        let error = collect_batch_items(&request, "flac").unwrap_err();
        assert!(error.contains("入力フォルダ内"), "{error}");
        assert!(!nested.join("voice.flac").exists());
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
            receipt: None,
            receipt_key: None,
            options: options(),
        };
        let items = collect_batch_items(&request, "opus").unwrap();
        assert_eq!(items.len(), 2);
        assert!(items
            .iter()
            .any(|item| item.output == output.join("one.opus")));
        assert!(items
            .iter()
            .any(|item| item.output == output.join("nested/two.opus")));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_item_ids_include_input_identity_destination_and_format() {
        let relative = Path::new("voice.wav");
        let output = Path::new("nested/voice.output");
        let wav = batch_resume::item_identity(
            Path::new("/input-a/voice.wav"),
            relative,
            output,
            OutputFormat::Wav,
        );
        assert_ne!(
            wav,
            batch_resume::item_identity(
                Path::new("/input-b/voice.wav"),
                relative,
                output,
                OutputFormat::Wav,
            )
        );
        assert_ne!(
            wav,
            batch_resume::item_identity(
                Path::new("/input-a/voice.wav"),
                relative,
                output,
                OutputFormat::Flac,
            )
        );
        assert_ne!(
            wav,
            batch_resume::item_identity(
                Path::new("/input-a/voice.wav"),
                relative,
                Path::new("other/voice.output"),
                OutputFormat::Wav,
            )
        );
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_batch_paths_do_not_collide_and_are_rejected_before_side_effects() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt as _;

        let first_relative = PathBuf::from(OsString::from_vec(b"voice-\x80.wav".to_vec()));
        let second_relative = PathBuf::from(OsString::from_vec(b"voice-\x81.wav".to_vec()));
        assert_eq!(
            first_relative.to_string_lossy(),
            second_relative.to_string_lossy()
        );
        assert_ne!(
            batch_resume::item_identity(
                Path::new("/input/voice.wav"),
                &first_relative,
                &first_relative,
                OutputFormat::Wav,
            ),
            batch_resume::item_identity(
                Path::new("/input/voice.wav"),
                &second_relative,
                &second_relative,
                OutputFormat::Wav,
            )
        );

        let directory = TestDirectory::create("batch-non-utf8");
        let input = directory.join("input");
        let output = directory.join("output");
        std::fs::create_dir(&input).unwrap();
        std::fs::create_dir(&output).unwrap();
        write_test_wav(&input.join(first_relative));
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_str().unwrap().into()),
            output_dir: output.to_str().unwrap().into(),
            output_format: "wav".into(),
            recursive: false,
            jobs: 1,
            resume: true,
            receipt: None,
            receipt_key: None,
            options: classical_options(false),
        };

        let error = collect_batch_items(&request, "wav").unwrap_err();
        assert!(error.contains("UTF-8"), "{error}");
        assert!(std::fs::read_dir(&output).unwrap().next().is_none());
        assert!(!output.join(".denoize-state").exists());
        assert!(!output.join(".denoize-batch.lock").exists());
    }

    #[test]
    fn desktop_resume_uses_canonical_state_and_exact_force_still_skips() {
        let directory = TestDirectory::create("batch-v3-exact");
        let request = desktop_batch_fixture(&directory, true, false);
        complete_desktop_batch(&request);
        let output = Path::new(&request.output_dir);
        assert!(output.join(batch_resume::STATE_FILE_NAME).is_file());
        assert!(output.join(batch_resume::LOCK_FILE_NAME).is_file());
        assert!(!output
            .join(batch_resume::LEGACY_DESKTOP_STATE_FILE_NAME)
            .exists());

        let prepared = prepare_batch_request(&request).unwrap();
        let session = BatchSession::acquire(output, true).unwrap();
        let planned = plan_batch_items(&session, prepared, true).unwrap();
        assert!(matches!(
            planned[0].decision,
            ResumeDecision::Skip {
                reason: batch_resume::ResumeReason::Exact
            }
        ));
    }

    #[test]
    fn desktop_resume_detects_input_recipe_and_output_changes() {
        for change in ["input", "recipe", "output"] {
            let directory = TestDirectory::create(&format!("batch-v3-{change}"));
            let mut request = desktop_batch_fixture(&directory, true, false);
            complete_desktop_batch(&request);
            let output = PathBuf::from(&request.output_dir);
            let expected_reason = match change {
                "input" => {
                    let input = Path::new(request.input_dir.as_deref().unwrap()).join("sample.wav");
                    let mut bytes = std::fs::read(&input).unwrap();
                    *bytes.last_mut().unwrap() ^= 0x7f;
                    std::fs::write(input, bytes).unwrap();
                    batch_resume::ResumeReason::InputChanged
                }
                "recipe" => {
                    request.options.strength = 0.8;
                    batch_resume::ResumeReason::RecipeChanged
                }
                "output" => {
                    std::fs::write(output.join("sample.wav"), b"changed output").unwrap();
                    batch_resume::ResumeReason::OutputChanged
                }
                _ => unreachable!(),
            };
            let prepared = prepare_batch_request(&request).unwrap();
            let session = BatchSession::acquire(&output, true).unwrap();
            let error = plan_batch_items(&session, prepared.clone(), false).unwrap_err();
            assert!(error.contains("上書き"), "{change}: {error}");
            let planned = plan_batch_items(&session, prepared, true).unwrap();
            assert!(matches!(
                planned[0].decision,
                ResumeDecision::Process { reason, .. } if reason == expected_reason
            ));
        }
    }

    #[test]
    fn desktop_resume_tracks_the_consumed_model_fingerprint() {
        use std::io::Write as _;

        let directory = TestDirectory::create("batch-v3-model");
        let input = directory.join("input.wav");
        let model = directory.join("model.onnx");
        let output_root = directory.join("output");
        let output = output_root.join("output.wav");
        std::fs::create_dir(&output_root).unwrap();
        write_test_wav(&input);
        std::fs::write(&model, b"model one").unwrap();
        let input_fingerprint = batch_resume::fingerprint_file(&input).unwrap();
        let recipe = Digest::from_bytes([9; 32]);
        let expectation = ResumeExpectation::new(
            Digest::from_bytes([7; 32]),
            output.clone(),
            input.clone(),
            input_fingerprint,
            Some(batch_resume::ConsumedModel {
                path: model.clone(),
                fingerprint: batch_resume::fingerprint_file(&model).unwrap(),
                sample_rate: 16_000,
            }),
            recipe,
        );
        let session = BatchSession::acquire(&output_root, true).unwrap();
        let ResumeDecision::Process { commit_mode, .. } =
            session.plan(&expectation, false).unwrap()
        else {
            panic!("missing output must be processed");
        };
        session.activate().unwrap();
        let mut transaction = AtomicOutput::new(&output).unwrap();
        transaction.file_mut().write_all(b"encoded output").unwrap();
        session
            .publish(&expectation, transaction, commit_mode)
            .unwrap();
        drop(session);

        std::fs::write(&model, b"model two").unwrap();
        let changed = ResumeExpectation::new(
            expectation.item_id(),
            output,
            input,
            input_fingerprint,
            Some(batch_resume::ConsumedModel {
                path: model.clone(),
                fingerprint: batch_resume::fingerprint_file(&model).unwrap(),
                sample_rate: 16_000,
            }),
            recipe,
        );
        let session = BatchSession::acquire(&output_root, true).unwrap();
        let decision = session.plan(&changed, true).unwrap();
        assert!(matches!(
            decision,
            ResumeDecision::Process {
                reason: batch_resume::ResumeReason::ModelChanged,
                ..
            }
        ));
    }

    #[test]
    fn legacy_desktop_state_is_untrusted_migrated_and_never_modified() {
        let directory = TestDirectory::create("batch-v3-legacy");
        let request = desktop_batch_fixture(&directory, true, false);
        let output = PathBuf::from(&request.output_dir);
        let destination = output.join("sample.wav");
        std::fs::write(&destination, b"legacy output").unwrap();
        let legacy_path = output.join(batch_resume::LEGACY_DESKTOP_STATE_FILE_NAME);
        let legacy = format!("v2:{}\n", "11".repeat(32));
        std::fs::write(&legacy_path, &legacy).unwrap();

        let prepared = prepare_batch_request(&request).unwrap();
        let session = BatchSession::acquire(&output, true).unwrap();
        let error = plan_batch_items(&session, prepared.clone(), false).unwrap_err();
        assert!(error.contains("上書き"), "{error}");
        assert_eq!(std::fs::read_to_string(&legacy_path).unwrap(), legacy);
        let planned = plan_batch_items(&session, prepared, true).unwrap();
        assert!(matches!(
            planned[0].decision,
            ResumeDecision::Process {
                reason: batch_resume::ResumeReason::Legacy,
                ..
            }
        ));
        session.activate().unwrap();
        publish_planned_item(&session, &planned[0]).unwrap();
        drop(session);
        assert_eq!(std::fs::read_to_string(&legacy_path).unwrap(), legacy);

        let prepared = prepare_batch_request(&request).unwrap();
        let session = BatchSession::acquire(&output, true).unwrap();
        let planned = plan_batch_items(&session, prepared, false).unwrap();
        assert!(matches!(planned[0].decision, ResumeDecision::Skip { .. }));
        assert_eq!(std::fs::read_to_string(legacy_path).unwrap(), legacy);
    }

    #[test]
    fn desktop_migrates_canonical_v1_and_v2_without_touching_legacy_gui_state() {
        for (index, canonical_legacy) in [
            b"sample.wav\n".to_vec(),
            format!("v2:{}\n", "52".repeat(32)).into_bytes(),
        ]
        .into_iter()
        .enumerate()
        {
            let directory = TestDirectory::create(&format!("canonical-legacy-{index}"));
            let mut request = desktop_batch_fixture(&directory, true, false);
            let output = PathBuf::from(&request.output_dir);
            let destination = output.join("sample.wav");
            let original_output = b"canonical legacy output";
            std::fs::write(&destination, original_output).unwrap();
            let canonical_path = output.join(batch_resume::STATE_FILE_NAME);
            std::fs::write(&canonical_path, &canonical_legacy).unwrap();

            let prepared = prepare_batch_request(&request).unwrap();
            let session = BatchSession::acquire(&output, true).unwrap();
            let error = plan_batch_items(&session, prepared, false).unwrap_err();
            assert!(error.contains("legacy"), "{error}");
            assert_eq!(std::fs::read(&destination).unwrap(), original_output);
            assert_eq!(std::fs::read(&canonical_path).unwrap(), canonical_legacy);
            drop(session);

            let legacy_gui_path = output.join(batch_resume::LEGACY_DESKTOP_STATE_FILE_NAME);
            let legacy_gui = format!("v2:{}\n", "63".repeat(32)).into_bytes();
            std::fs::write(&legacy_gui_path, &legacy_gui).unwrap();
            request.options.force = true;
            let prepared = prepare_batch_request(&request).unwrap();
            let session = BatchSession::acquire(&output, true).unwrap();
            let planned = plan_batch_items(&session, prepared, true).unwrap();
            assert!(matches!(
                planned[0].decision,
                ResumeDecision::Process {
                    reason: batch_resume::ResumeReason::Legacy,
                    ..
                }
            ));
            session.activate().unwrap();
            publish_planned_item(&session, &planned[0]).unwrap();
            drop(session);
            let migrated_state = std::fs::read(&canonical_path).unwrap();
            assert!(migrated_state.starts_with(&canonical_legacy));
            assert!(String::from_utf8_lossy(&migrated_state).contains("\"version\":3"));
            assert_eq!(std::fs::read(&legacy_gui_path).unwrap(), legacy_gui);

            request.options.force = false;
            let prepared = prepare_batch_request(&request).unwrap();
            let session = BatchSession::acquire(&output, true).unwrap();
            let planned = plan_batch_items(&session, prepared, false).unwrap();
            assert!(matches!(
                planned[0].decision,
                ResumeDecision::Skip {
                    reason: batch_resume::ResumeReason::Exact
                }
            ));
            session.activate().unwrap();
            assert_eq!(std::fs::read(&canonical_path).unwrap(), migrated_state);
            assert_eq!(std::fs::read(&legacy_gui_path).unwrap(), legacy_gui);
        }
    }

    #[test]
    fn desktop_batch_session_guard_lives_in_the_worker() {
        let directory = TestDirectory::create("batch-v3-lock-lifetime");
        let output = directory.join("output");
        std::fs::create_dir(&output).unwrap();
        let session = BatchSession::acquire(&output, false).unwrap();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            ready_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            drop(session);
        });
        ready_rx.recv().unwrap();
        assert!(BatchSession::acquire(&output, false).is_err());
        release_tx.send(()).unwrap();
        worker.join().unwrap();
        assert!(BatchSession::acquire(&output, false).is_ok());
    }

    #[test]
    fn batch_outputs_cannot_claim_control_paths() {
        let directory = TestDirectory::create("batch-state-reserved");
        for name in [
            batch_resume::STATE_FILE_NAME,
            batch_resume::LEGACY_DESKTOP_STATE_FILE_NAME,
            batch_resume::LOCK_FILE_NAME,
        ] {
            let items = vec![test_batch_item(
                directory.join("input.wav"),
                directory.join(name).join("nested.wav"),
                1,
            )];
            let error = validate_batch_control_paths(&items, &directory.path).unwrap_err();
            assert!(error.contains(name), "{error}");
        }
    }

    #[test]
    fn successful_batch_has_completed_terminal_outcome() {
        let counts = BatchOutcomeCounts {
            completed: 3,
            skipped: 1,
            ..Default::default()
        };
        let outcome = batch_terminal_outcome(counts);
        assert_eq!(outcome.status, "completed");
        assert_eq!(
            outcome.message,
            "完了 3 · スキップ 1 · 失敗 0 · キャンセル 0"
        );
        assert_eq!(outcome.error, None);
        assert_eq!(counts.total(), 4);
    }

    #[test]
    fn mixed_batch_has_failed_terminal_outcome() {
        let counts = BatchOutcomeCounts {
            completed: 2,
            skipped: 1,
            failed: 1,
            cancelled: 0,
        };
        let outcome = batch_terminal_outcome(counts);
        assert_eq!(outcome.status, "failed");
        assert_eq!(
            outcome.message,
            "完了 2 · スキップ 1 · 失敗 1 · キャンセル 0"
        );
        assert_eq!(
            outcome.error.as_deref(),
            Some("1件のファイルを処理できませんでした")
        );
        assert_eq!(counts.total(), 4);
    }

    #[test]
    fn all_failed_batch_has_failed_terminal_outcome() {
        let counts = BatchOutcomeCounts {
            failed: 3,
            ..Default::default()
        };
        let outcome = batch_terminal_outcome(counts);
        assert_eq!(outcome.status, "failed");
        assert_eq!(
            outcome.message,
            "完了 0 · スキップ 0 · 失敗 3 · キャンセル 0"
        );
        assert_eq!(
            outcome.error.as_deref(),
            Some("3件のファイルを処理できませんでした")
        );
        assert_eq!(counts.total(), 3);
    }

    #[test]
    fn cancelled_batch_has_one_total_partition() {
        let fingerprint = batch_resume::FileFingerprint {
            len: 1,
            digest: batch_resume::Digest::from_bytes([42; 32]),
        };
        assert_eq!(
            batch_item_commit_mode(
                ResumeDecision::Skip {
                    reason: batch_resume::ResumeReason::Exact,
                },
                Some(fingerprint),
                true,
            ),
            Err(BatchItemOutcome::Skipped(fingerprint)),
            "an exact resume skip remains skipped after cancellation"
        );
        assert_eq!(
            batch_item_commit_mode(
                ResumeDecision::Process {
                    commit_mode: CommitMode::NoClobber,
                    reason: batch_resume::ResumeReason::Missing,
                },
                None,
                true,
            ),
            Err(BatchItemOutcome::Cancelled),
            "only work that still needs processing is cancelled"
        );
        let outcomes = vec![
            BatchItemOutcome::Completed(fingerprint),
            BatchItemOutcome::Skipped(fingerprint),
            BatchItemOutcome::Failed("injected".into()),
            BatchItemOutcome::Cancelled,
            BatchItemOutcome::Cancelled,
        ];
        assert_eq!(
            outcomes
                .iter()
                .map(BatchItemOutcome::status)
                .collect::<Vec<_>>(),
            vec!["completed", "skipped", "failed", "cancelled", "cancelled"]
        );
        assert_eq!(outcomes[2].error().as_deref(), Some("injected"));
        let counts = count_batch_outcomes(&outcomes);
        let outcome = batch_terminal_outcome(counts);
        assert_eq!(outcome.status, "cancelled");
        assert_eq!(
            outcome.message,
            "完了 1 · スキップ 1 · 失敗 1 · キャンセル 2"
        );
        assert_eq!(outcome.error, None);
        assert_eq!(counts.total(), 5);
    }

    #[test]
    fn valid_gui_toml_config_round_trips_without_nulls() {
        let path = std::env::temp_dir().join(format!(
            "denoize-gui-config-{}-{}.toml",
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let mut expected = gui_config();
        expected.strength = 0.42;
        let expected = expected.normalized().unwrap();
        save_gui_config(path.to_string_lossy().into_owned(), expected.clone()).unwrap();
        let loaded = load_gui_config(path.to_string_lossy().into_owned(), gui_config()).unwrap();
        assert_eq!(loaded, expected);
        let source = std::fs::read_to_string(&path).unwrap();
        assert!(!source.contains("onnx_model"));
        assert!(!source.contains("loudness_lufs"));
        assert!(source.contains("true_peak_dbtp = -1.0"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn signed_package_toml_omits_the_unsigned_raw_model_rate() {
        let mut config = gui_config();
        config.backend = "onnx".into();
        config.model_package = Some("voice.dmp".into());
        config.model_package_key = Some("vendor.pub".into());
        config.onnx_rate = None;

        let source = toml::to_string_pretty(&config).unwrap();
        assert!(source.contains("model_package = \"voice.dmp\""));
        assert!(source.contains("model_package_key = \"vendor.pub\""));
        assert!(!source.contains("onnx_rate"));
    }

    #[test]
    fn exported_loudness_sentinel_clears_an_enabled_current_config() {
        let directory = TestDirectory::create("gui-loudness-clear");
        let path = directory.join("config.toml");
        save_gui_config(path.to_str().unwrap().into(), gui_config()).unwrap();
        let mut current = gui_config();
        current.loudness_lufs = Some(-16.0);
        current.true_peak_dbtp = Some(-1.0);

        let loaded = load_gui_config(path.to_str().unwrap().into(), current).unwrap();

        assert!(loaded.loudness_lufs.is_none());
        assert!(loaded.true_peak_dbtp.is_none());
    }

    #[test]
    fn gui_toml_partial_patch_preserves_current_settings() {
        let mut current = gui_config();
        current.mode = "ambient".into();
        current.force = true;
        let loaded = parse_gui_config(
            "backend = \"classical\"\nstrength = 0.73\n",
            current.clone(),
        )
        .unwrap();

        assert_eq!(loaded.backend, "classical");
        assert_eq!(loaded.strength, 0.73);
        assert_eq!(loaded.mode, current.mode);
        assert_eq!(loaded.force, current.force);
        assert_eq!(loaded.preset, current.preset);
    }

    #[test]
    fn raw_model_patch_replaces_a_dormant_package_selection() {
        if Backend::parse("onnx").is_none() {
            return;
        }
        let mut current = gui_config();
        current.backend = "onnx".into();
        current.onnx_model = None;
        current.model_package = Some("old.dmp".into());
        current.model_package_key = Some("old.pub".into());

        let loaded = parse_gui_config(
            "onnx_model = \"replacement.onnx\"\nonnx_rate = 48000\n",
            current,
        )
        .unwrap();
        assert_eq!(loaded.onnx_model.as_deref(), Some("replacement.onnx"));
        assert_eq!(loaded.onnx_rate, Some(48_000));
        assert!(loaded.model_package.is_none());
        assert!(loaded.model_package_key.is_none());

        let error = parse_gui_config(
            "onnx_model = \"raw.onnx\"\nmodel_package = \"voice.dmp\"\nmodel_package_key = \"vendor.pub\"\n",
            gui_config(),
        )
        .unwrap_err();
        assert!(error.contains("同時に指定できません"), "{error}");
    }

    #[test]
    fn gui_toml_discards_hidden_models_for_non_external_backends() {
        let loaded = parse_gui_config(
            "backend = \"classical\"\nonnx_model = \"stale.onnx\"\nonnx_rate = 0\nmodel_package = \"stale.dmp\"\nmodel_package_key = \"stale.pub\"\n",
            gui_config(),
        )
        .unwrap();

        assert!(loaded.onnx_model.is_none());
        assert!(loaded.model_package.is_none());
        assert!(loaded.model_package_key.is_none());
        assert_eq!(loaded.onnx_rate, None);
    }

    #[test]
    fn gui_toml_config_rejects_boolean_strings() {
        for field in ["force", "preserve_metadata"] {
            let source = gui_config_source()
                .replace(&format!("{field} = false"), &format!("{field} = \"false\""))
                .replace(&format!("{field} = true"), &format!("{field} = \"true\""));
            assert!(parse_gui_config(&source, gui_config()).is_err(), "{field}");
        }
    }

    #[test]
    fn gui_toml_config_rejects_unknown_fields() {
        let source = format!("{}unknown_option = true\n", gui_config_source());
        let error = parse_gui_config(&source, gui_config()).unwrap_err();
        assert!(error.contains("unknown field"), "unexpected error: {error}");
    }

    #[test]
    fn gui_toml_config_rejects_unknown_enums() {
        let mut config = gui_config();
        config.channels = "surround".into();
        let source = toml::to_string_pretty(&config).unwrap();
        let error = parse_gui_config(&source, gui_config()).unwrap_err();
        assert!(
            error.contains("チャンネルモード"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn gui_toml_config_rejects_out_of_range_values() {
        let mut config = gui_config();
        config.strength = 1.01;
        let source = toml::to_string_pretty(&config).unwrap();
        let error = parse_gui_config(&source, gui_config()).unwrap_err();
        assert!(error.contains("強度"), "unexpected error: {error}");
    }

    #[test]
    fn dropped_paths_are_classified_without_reading_contents() {
        let root = std::env::temp_dir().join(format!(
            "denoize-gui-drop-{}-{}",
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let audio = root.join("voice.wav");
        let ignored = root.join("notes.txt");
        std::fs::write(&audio, []).unwrap();
        std::fs::write(&ignored, []).unwrap();
        let result = classify_dropped_paths(vec![
            root.to_string_lossy().into_owned(),
            audio.to_string_lossy().into_owned(),
            ignored.to_string_lossy().into_owned(),
        ]);
        assert_eq!(result.directories.len(), 1);
        assert_eq!(result.audio_files.len(), 1);
        assert_eq!(result.ignored.len(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }
}
