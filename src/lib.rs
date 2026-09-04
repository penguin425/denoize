//! `denoize` — pure-Rust audio denoiser built for the world's highest fidelity.
//!
//! Goal: transparent, artifact-free restoration that preserves timbre,
//! transients, dynamics, and "air" better than any classical offline tool.
//!
//! ## Implemented technologies
//!
//! ### Classical DSP (always available)
//! - STFT/ISTFT + Perfect Reconstruction OLA（高オーバーラップ対応）
//! - IMCRA/MCRA ノイズ推定 + Spectral Flatness プロファイル + Anchoring
//! - Ephraim-Malah Decision-Directed SNR
//! - 8種類のゲイン推定器（OMLSA, LogMMSE, MMSE-STSA, Wiener, SpecSub + 非線形/幾何学的）
//! - Attack/Release + Cepstral Smoothing + Transient Protection
//! - 高度窓関数: Kaiser / Flat-top / DPSS
//! - マルチバンドスペクトルサブトラクション
//! - 知覚重み付け（Bark帯域）+ 音楽ノイズ抑制ポストフィルタ
//!
//! ### Input / output codecs (built-in, no ffmpeg)
//! - **Decode**: WAV / MP3 (`symphonia`, including Xing/LAME gapless timing,
//!   with a bounded raw-stream compatibility fallback) / M4A (Pure-Rust AAC-LC
//!   and ALAC with v0/v1 unity-rate edit-list presentation timing)
//! - **Encode**: WAV / MP3 (`shine-rs`) / M4A (`oxideav-aac` Pure-Rust AAC-LC)
//! - Decoded to `f64` PCM at native sample rate (no extra quantisation)
//!
//! ### Optional AI backends (feature-gated)
//! - `rnnoise` feature: RNNoise via nnnoiseless (pure-Rust)
//! - `deepfilter` feature: DeepFilterNet v3 via tract ONNX
//! - `onnx` feature: user-supplied waveform ONNX models via tract
//!   (including a reusable, contract-checked loaded-model API)
//! - `aec` feature: typed far-end-reference partitioned frequency-domain AEC
//! - `mpsenet` feature: MP-SENet compressed-magnitude/phase ONNX adapter
//! - `bsrnn` feature: ESPnet BSRNN spectral ONNX adapter
//! - `mossformer2` feature: ClearerVoice MossFormer2 48 kHz ONNX adapter
//! - `sgmse` feature: SGMSE+ iterative diffusion ONNX adapter
//!
//! Build with all backends: `cargo build --release --features full`

#[cfg(feature = "aec")]
pub mod acoustic_echo;
pub mod atomic_output;
pub mod audio;
pub mod automation;
pub mod backend;
#[doc(hidden)]
pub mod batch_resume;
pub mod benchmark;
pub mod bessel;
#[cfg(feature = "onnx")]
pub mod causal_target_sound;
#[cfg(feature = "onnx")]
pub mod causal_target_speaker;
pub mod channel_layout;
pub mod config;
pub mod daw;
pub mod decode;
pub mod denoiser;
pub mod diagnostics;
pub mod encode;
pub mod evaluation;
pub mod execution;
#[doc(hidden)]
pub mod fault_injection;
pub mod fft;
pub mod gain;
pub mod hardware;
pub mod input;
pub mod ipc;
#[cfg(feature = "live")]
pub mod live;
pub mod loudness;
pub mod meeting_speaker;
pub mod metadata;
pub mod microphone_array;
pub mod model_package;
pub mod models;
pub mod music_restoration;
pub mod neural_daw;
pub mod noise;
pub mod perceptual;
pub mod postfilter;
pub mod project;
pub mod project_v2;
pub mod quality;
pub mod recommendation;
pub mod region;
pub mod resample;
pub mod resource;
pub mod restoration;
pub mod sdk;
pub mod service;
pub mod stft;
mod stoi_resample;
pub mod stream;
pub mod target_sound;
pub mod target_speaker;
pub mod universal_restoration;
pub mod update;
pub mod vad;
pub mod watch;
pub mod window;

#[cfg(feature = "aec")]
pub use acoustic_echo::{
    estimate_aec_memory_bytes, sign_aec_promotion_evidence, AecBlockDiagnostics, AecClockMapping,
    AecConfig, AecDelayEstimate, AecEvidenceMetric, AecEvidenceMetricOperator, AecEvidenceStratum,
    AecEvidenceStratumKind, AecPromotionEvidencePayload, AecRealtimeAdapter,
    AecRealtimeDiagnostics, AecRenderReport, AecRenderResult, AecResetCounts, AecResetReason,
    AecSession, AecStream, AecTalkState, SignedAecPromotionEvidence, AEC_PROMOTION_EVIDENCE_SCHEMA,
    AEC_REPORT_SCHEMA, AEC_SCHEMA_VERSION,
};
pub use atomic_output::{AtomicOutput, CommitMode};
pub use audio::{
    ensure_memory_limit, estimate_audio_memory_bytes, estimate_audio_working_set_bytes,
    estimate_file_memory_bytes, estimate_session_memory_bytes, estimate_stream_memory_bytes,
    estimate_stream_memory_bytes_checked, inspect_wav_session, read_audio, read_audio_from_session,
    read_audio_from_session_with_limits, read_audio_with_limits, read_audio_with_metadata_limits,
    read_wav, read_wav_bytes, read_wav_bytes_with_limits, read_wav_from_session,
    read_wav_from_session_with_limits, read_wav_with_limits, sanitize_sample, write_audio,
    write_wav, write_wav_bytes, write_wav_channel_mask, Audio, WavStreamInfo, WavStreamReader,
    WavStreamWriter,
};
#[cfg(feature = "dpdfnet")]
pub use backend::dpdfnet::{DpdfnetMetadata, DpdfnetModel, DpdfnetStream};
#[cfg(feature = "gtcrn")]
pub use backend::gtcrn::{GtcrnModel, GtcrnStream};
#[cfg(feature = "onnx")]
pub use backend::onnx::{OnnxWaveformContract, OnnxWaveformLayout, OnnxWaveformModel};
pub use backend::{
    decode_mid_side, encode_mid_side, Backend, BackendOptions, BackendSession, ChannelMode,
    OnnxModelConfig, SgmseProfile, StreamingBackendSession,
};
pub use benchmark::{ArtifactReport, BenchmarkReport, ComparisonReport};
#[cfg(feature = "onnx")]
pub use causal_target_sound::{
    estimate_causal_target_sound_memory_bytes, sign_causal_target_sound_promotion_evidence,
    write_causal_target_sound_conservative_fallback, CausalTargetSoundBlock,
    CausalTargetSoundBlockDecision, CausalTargetSoundConfig, CausalTargetSoundDecisionCounts,
    CausalTargetSoundDeviceLatencyMeasurement, CausalTargetSoundEvidenceIdentity,
    CausalTargetSoundMetricEvidence, CausalTargetSoundModelIdentity,
    CausalTargetSoundPromotionEvidencePayload, CausalTargetSoundQueryIdentity,
    CausalTargetSoundRealtimeAudit, CausalTargetSoundRealtimeMetrics,
    CausalTargetSoundRealtimeReceiveError, CausalTargetSoundRealtimeResult,
    CausalTargetSoundRealtimeScheduler, CausalTargetSoundRealtimeSubmitError,
    CausalTargetSoundRealtimeToken, CausalTargetSoundRenderReport, CausalTargetSoundRenderResult,
    CausalTargetSoundSession, CausalTargetSoundSnapshot, CausalTargetSoundSnapshotState,
    CausalTargetSoundStratumEvidence, CausalTargetSoundStream, CausalTargetSoundTransitionAudit,
    SignedCausalTargetSoundPromotionEvidence, CAUSAL_TARGET_SOUND_EVIDENCE_SCHEMA,
    CAUSAL_TARGET_SOUND_REPORT_SCHEMA, CAUSAL_TARGET_SOUND_SCHEMA_VERSION,
    CAUSAL_TARGET_SOUND_SNAPSHOT_SCHEMA,
};
#[cfg(feature = "onnx")]
pub use causal_target_speaker::{
    estimate_causal_target_speaker_memory_bytes, sign_causal_target_speaker_promotion_evidence,
    CausalTargetSpeakerBlock, CausalTargetSpeakerBlockDecision, CausalTargetSpeakerConfig,
    CausalTargetSpeakerDecisionCounts, CausalTargetSpeakerEnrollmentSummary,
    CausalTargetSpeakerEvidenceIdentity, CausalTargetSpeakerMetricEvidence,
    CausalTargetSpeakerModelIdentity, CausalTargetSpeakerPromotionEvidencePayload,
    CausalTargetSpeakerRealtimeAudit, CausalTargetSpeakerRealtimeMetrics,
    CausalTargetSpeakerRealtimeReceiveError, CausalTargetSpeakerRealtimeResult,
    CausalTargetSpeakerRealtimeScheduler, CausalTargetSpeakerRealtimeSubmitError,
    CausalTargetSpeakerRealtimeToken, CausalTargetSpeakerRenderReport,
    CausalTargetSpeakerRenderResult, CausalTargetSpeakerSession,
    CausalTargetSpeakerStratumEvidence, CausalTargetSpeakerStream,
    CausalTargetSpeakerTransitionAudit, SignedCausalTargetSpeakerPromotionEvidence,
    CAUSAL_TARGET_SPEAKER_PROMOTION_EVIDENCE_SCHEMA, CAUSAL_TARGET_SPEAKER_REPORT_SCHEMA,
    CAUSAL_TARGET_SPEAKER_SCHEMA_VERSION,
};
pub use channel_layout::{ChannelLayout, ChannelMask, ChannelPosition, PanInfo};
pub use config::{ConfigError, ResourcePlan};
pub use daw::{
    read_daw_preset, read_daw_session, write_daw_preset, write_daw_session, DawParameters,
    DawPortConfiguration, DawPreset, DawRealtimeParameters, DawRealtimeProcessor, DawSessionState,
    DAW_FIXED_LATENCY_MILLIS, DAW_LATENCY_POLICY, DAW_MAX_SAMPLE_RATE, DAW_PLUGIN_ID,
    DAW_PRESET_SCHEMA, DAW_PRESET_SCHEMA_VERSION, DAW_SESSION_SCHEMA, DAW_SESSION_SCHEMA_VERSION,
};
pub use decode::{
    decode_file, decode_file_from_session_with_limits, decode_file_with_limits,
    decode_file_with_metadata_limits, inspect_audio_stream_session, probe_file,
    probe_file_from_session_with_limits, probe_file_with_limits, probe_file_with_metadata_limits,
    AudioCodec, AudioFormat, AudioProbe, AudioStreamInfo, AudioStreamReader, DecodeLimits,
    DecodedPcm,
};
pub use denoiser::{Denoiser, DenoiserConfig, Preset, ProcessingMode, StreamingDenoiser};
pub use diagnostics::{
    assess_file_with_options, compare_files_with_options, diagnose_audio, diagnose_file,
    diagnose_file_with_options, AssessmentComparison, AssessmentReport, DiagnosticFinding,
    DiagnosticInput, DiagnosticMetrics, DiagnosticOptions, DiagnosticReport, NoReferenceQuality,
    ASSESSMENT_SCHEMA, ASSESSMENT_SCHEMA_VERSION, DIAGNOSTIC_SCHEMA, DIAGNOSTIC_SCHEMA_VERSION,
};
pub use encode::{
    estimate_spooled_stream_output_bytes, estimate_stream_encode_additional_bytes,
    estimate_stream_encode_output_bytes, estimate_stream_encode_temporary_bytes,
    estimate_stream_output_verification_bytes, verify_stream_output_file, AacEncoder,
    AudioStreamWriter, DownmixMode, EncodeOptions, OutputFormat, SpooledAudioStreamWriter,
    StreamEncodeLimits, StreamEncodeSpec, StreamOutputVerification, StreamPcmSpool,
};
pub use evaluation::{
    compare_evaluation_results, run_evaluation, validate_evaluation_corpus,
    verify_evaluation_result, write_signed_evaluation_result, EvaluationArtifact, EvaluationCase,
    EvaluationCaseResult, EvaluationComparisonReport, EvaluationCorpusValidation,
    EvaluationEnvironment, EvaluationLicense, EvaluationManifest, EvaluationPolicy,
    EvaluationRecipe, EvaluationResultPayload, EvaluationSource, EvaluationThreshold,
    EvaluationVerificationReport, ListeningEvidence, ListeningPolicy, ListeningProtocol,
    ListeningTestResult, ObjectiveEvaluationMetrics, OutputQualityMetrics, PerformanceMetrics,
    RegressionDirection, RegressionOutcome, RegressionTolerance, SignalPreparation,
    SignedEvaluationResult, ThresholdAggregation, ThresholdOperator, ThresholdOutcome,
    EVALUATION_COMPARISON_SCHEMA, EVALUATION_CORPUS_SCHEMA, EVALUATION_CORPUS_VERIFICATION_SCHEMA,
    EVALUATION_RESULT_SCHEMA, EVALUATION_SCHEMA_VERSION, EVALUATION_VERIFICATION_SCHEMA,
    LISTENING_RESULT_SCHEMA,
};
pub use execution::{
    execution_item_id, export_receipt_public_key, generate_receipt_keypair, portable_file_locator,
    portable_locator, write_execution_plan, write_new_receipt_keypair, write_receipt_trust_policy,
    write_signed_receipt, ExecutionKind, ExecutionPlan, ExecutionPlanItem, ExecutionReceiptPayload,
    PlannedArtifact, PlannedOutput, PlannedResources, ReceiptItem, ReceiptOutput, ReceiptPublicKey,
    ReceiptSecretKey, ReceiptSignature, ReceiptTrustPolicy, ReceiptVerificationReport,
    SignedExecutionReceipt, VerifiedReceiptItem, EXECUTION_PLAN_SCHEMA, EXECUTION_RECEIPT_SCHEMA,
    EXECUTION_SCHEMA_VERSION, RECEIPT_PUBLIC_KEY_SCHEMA, RECEIPT_SECRET_KEY_SCHEMA,
    RECEIPT_TRUST_POLICY_SCHEMA, RECEIPT_VERIFICATION_SCHEMA, STREAM_EXECUTION_PLAN_SCHEMA,
    STREAM_EXECUTION_RECEIPT_SCHEMA, STREAM_EXECUTION_SCHEMA_VERSION,
    STREAM_RECEIPT_VERIFICATION_SCHEMA,
};
pub use gain::{Algorithm, SpecSubLaw};
pub use hardware::{
    backend_supports_acceleration, hardware_capabilities, select_accelerator,
    select_accelerator_for_options, AcceleratorFallback, AcceleratorPreference, AcceleratorRuntime,
    AcceleratorSelection, BackendCapability, HardwareCapabilities, RuntimeCapability,
    HARDWARE_SCHEMA, HARDWARE_SCHEMA_VERSION,
};
pub use input::AudioInputSession;
pub use input::StreamSpoolLimits;
#[cfg(feature = "onnx")]
pub use meeting_speaker::MeetingSpeakerSession;
pub use meeting_speaker::{
    estimate_meeting_speaker_memory_bytes, sign_meeting_speaker_promotion_evidence,
    MeetingActivityState, MeetingRegion, MeetingSpeakerConfig, MeetingSpeakerEvidenceIdentity,
    MeetingSpeakerEvidenceStratum, MeetingSpeakerModelIdentity,
    MeetingSpeakerPromotionEvidencePayload, MeetingSpeakerReport, MeetingSpeakerResult,
    MeetingSpeakerSegment, MeetingSpeakerTrackSummary, MeetingTrackLabel,
    MeetingTrackLabelsDocument, SignedMeetingSpeakerPromotionEvidence, MAX_MEETING_SPEAKER_TRACKS,
    MEETING_SPEAKER_EVIDENCE_SCHEMA, MEETING_SPEAKER_REPORT_SCHEMA, MEETING_SPEAKER_SCHEMA_VERSION,
    MEETING_TRACK_LABELS_SCHEMA,
};
pub use microphone_array::{
    estimate_microphone_array_memory_bytes, sign_microphone_array_promotion_evidence,
    ArrayCoordinateUnit, ArrayHandedness, ArrayInputSemantics, MicrophoneArrayConfig,
    MicrophoneArrayEvidenceStratum, MicrophoneArrayGeometry,
    MicrophoneArrayPromotionEvidencePayload, MicrophoneArrayReport, MicrophoneArrayResult,
    MicrophoneArraySession, MicrophonePosition, SignedMicrophoneArrayPromotionEvidence,
    MICROPHONE_ARRAY_EVIDENCE_SCHEMA, MICROPHONE_ARRAY_REPORT_SCHEMA,
    MICROPHONE_ARRAY_SCHEMA_VERSION,
};
pub use model_package::{
    build_runtime_model_package, build_runtime_model_package_v2, inspect_runtime_model_package,
    RuntimeModelAxisContractV2, RuntimeModelChannelContractV2, RuntimeModelChannelRoleContractV2,
    RuntimeModelComponentContractV2, RuntimeModelFileContract, RuntimeModelFrontendContract,
    RuntimeModelFrontendContractV2, RuntimeModelGeometryContractV2, RuntimeModelLatencyContractV2,
    RuntimeModelLicenseContract, RuntimeModelLicenseContractV2, RuntimeModelMicrophonePositionV2,
    RuntimeModelNumericalCaseV1, RuntimeModelNumericalTensorV1, RuntimeModelNumericalToleranceV1,
    RuntimeModelNumericalVectorsV1, RuntimeModelPackage, RuntimeModelPackageInfo,
    RuntimeModelPackageManifest, RuntimeModelPackageManifestV2, RuntimeModelPackageReader,
    RuntimeModelPackageV2Info, RuntimeModelPrecisionProfileContractV2,
    RuntimeModelProvenanceContractV2, RuntimeModelResourceContract, RuntimeModelRuntimeContract,
    RuntimeModelRuntimeContractV2, RuntimeModelStatePairContractV2, RuntimeModelTensorContract,
    RuntimeModelTensorContractV2, RuntimeModelTensorSetContractV2,
    RuntimeModelTrainingDatasetContractV2, RUNTIME_MODEL_PACKAGE_SCHEMA,
    RUNTIME_MODEL_PACKAGE_SCHEMA_V2, RUNTIME_MODEL_PACKAGE_VERSION,
    RUNTIME_MODEL_PACKAGE_VERSION_V2,
};
#[cfg(feature = "onnx")]
pub use music_restoration::MusicRestorationSession;
pub use music_restoration::{
    estimate_music_restoration_memory_bytes, sign_music_restoration_promotion_evidence,
    MusicRestorationConfig, MusicRestorationDecision, MusicRestorationEvidenceIdentity,
    MusicRestorationEvidenceStratum, MusicRestorationModelIdentity,
    MusicRestorationPromotionEvidencePayload, MusicRestorationRegion, MusicRestorationReport,
    MusicRestorationResult, MusicRestorationTask, MusicRestorationTrainingDatasetIdentity,
    SignedMusicRestorationPromotionEvidence, MUSIC_RESTORATION_EVIDENCE_SCHEMA,
    MUSIC_RESTORATION_REPORT_SCHEMA, MUSIC_RESTORATION_SCHEMA_VERSION,
};
pub use neural_daw::{
    neural_daw_chunk_frames, neural_daw_latency_frames, neural_daw_latency_millis,
    read_neural_daw_session, write_neural_daw_session, NeuralDawModel, NeuralDawOverloadFallback,
    NeuralDawParameters, NeuralDawPortConfiguration, NeuralDawSessionState,
    NEURAL_DAW_BLOCK_POOL_SIZE, NEURAL_DAW_CHUNK_MILLIS, NEURAL_DAW_LATENCY_CHUNKS,
    NEURAL_DAW_LATENCY_POLICY, NEURAL_DAW_MAX_SAMPLE_RATE, NEURAL_DAW_MODEL_ID,
    NEURAL_DAW_MODEL_SHA256, NEURAL_DAW_PLUGIN_ID, NEURAL_DAW_QUEUE_BLOCKS,
    NEURAL_DAW_SESSION_SCHEMA, NEURAL_DAW_SESSION_SCHEMA_VERSION, NEURAL_HQ_DAW_MODEL_ID,
    NEURAL_HQ_DAW_MODEL_SHA256, NEURAL_HQ_DAW_PLUGIN_ID,
};
pub use project::{
    assemble_project_timeline, build_project_bundle, import_project_bundle, inspect_project_bundle,
    inspect_project_source, project_artifact_reference, relocate_project_source, run_project_batch,
    validate_project_files, write_project_execution_plan, write_project_manifest,
    write_signed_project_execution_receipt, ProjectArtifactReference, ProjectBatchItemReport,
    ProjectBatchReport, ProjectBatchRequest, ProjectBundleBinding, ProjectBundleBindingKind,
    ProjectBundleBuildOptions, ProjectBundleFileInfo, ProjectBundleImportReport, ProjectBundleInfo,
    ProjectExecutionPlan, ProjectExecutionReceiptPayload, ProjectManifest, ProjectModelReference,
    ProjectReceiptVerificationReport, ProjectRenderReport, ProjectSelection, ProjectSource,
    ProjectSourceInspection, ProjectTimeline, ProjectValidationReport,
    SignedProjectExecutionReceipt, PROJECT_BATCH_SCHEMA, PROJECT_BUNDLE_IMPORT_SCHEMA,
    PROJECT_BUNDLE_SCHEMA, PROJECT_EXECUTION_PLAN_SCHEMA, PROJECT_EXECUTION_RECEIPT_SCHEMA,
    PROJECT_MANIFEST_SCHEMA, PROJECT_MANIFEST_SCHEMA_VERSION, PROJECT_RECEIPT_VERIFICATION_SCHEMA,
    PROJECT_RENDER_SCHEMA, PROJECT_VALIDATION_SCHEMA, PROJECT_WATCH_CYCLE_SCHEMA,
};
pub use quality::QualityMetrics;
pub use recommendation::{
    recommend_audio, recommend_file, recommend_file_with_options, run_device_calibration,
    CalibrationEvidence, RecommendationCandidate, RecommendationDecision, RecommendationDevice,
    RecommendationGoal, RecommendationInput, RecommendationMaterial, RecommendationOptions,
    RecommendationReason, RecommendationReport, RECOMMENDATION_SCHEMA,
    RECOMMENDATION_SCHEMA_VERSION,
};
pub use region::{
    PresentationRegion, PRESENTATION_REGION_SCHEMA, PRESENTATION_REGION_SCHEMA_VERSION,
};
pub use resource::{
    estimate_backend_session_request, estimate_backend_worker_gpu_memory_bytes,
    estimate_backend_worker_memory_bytes, estimate_gpu_session_bytes, estimate_gpu_worker_bytes,
    estimate_model_session_bytes, estimate_temporary_bytes, metadata_limits_after_retained_memory,
    metadata_limits_for_available_memory, ResourceGovernor, ResourceLimits, ResourcePermit,
    ResourceRequest, ResourceUsage,
};
pub use restoration::{
    estimate_restoration_memory_bytes, restore_audio, DeclickConfig, DeclipConfig, DehumConfig,
    RestorationConfig, RestorationMask, RestorationMaskRun, RestorationMaskState, RestorationMode,
    RestorationOperation, RestorationOperationDetails, RestorationOperationReport,
    RestorationReport, RestorationResult, RestorationStatus, WindPlosiveConfig, WpeChannelMode,
    WpeConfig, MAX_RESTORATION_CHANNELS, MAX_RESTORATION_MASK_RUNS, MAX_RESTORATION_OPERATIONS,
    RESTORATION_MASK_SCHEMA, RESTORATION_REPORT_SCHEMA, RESTORATION_SCHEMA_VERSION,
};
#[cfg(feature = "onnx")]
pub use target_sound::TargetSoundSession;
pub use target_sound::{
    estimate_target_sound_memory_bytes, sign_target_sound_promotion_evidence,
    SignedTargetSoundPromotionEvidence, TargetSoundCatalogClass, TargetSoundConfig,
    TargetSoundDecision, TargetSoundEvidenceIdentity, TargetSoundEvidenceStratum,
    TargetSoundMetricOperator, TargetSoundMetricOutcome, TargetSoundMode, TargetSoundModelIdentity,
    TargetSoundPresence, TargetSoundPresenceAssessment, TargetSoundPromotionEvidencePayload,
    TargetSoundQuery, TargetSoundQueryIdentity, TargetSoundReport, TargetSoundResult,
    TargetSoundSafetyGate, TargetSoundSafetyGateKind, TargetSoundSafetyMeasurements,
    TargetSoundStratumKind, TargetSoundTrainingDatasetIdentity, MAX_TARGET_SOUND_AUDIO_SECONDS,
    MAX_TARGET_SOUND_CLASSES, MAX_TARGET_SOUND_WINDOWS, TARGET_SOUND_EVIDENCE_SCHEMA,
    TARGET_SOUND_QUERY_SCHEMA, TARGET_SOUND_REPORT_SCHEMA, TARGET_SOUND_SCHEMA_VERSION,
};
#[cfg(feature = "onnx")]
pub use target_speaker::TargetSpeakerSession;
pub use target_speaker::{
    estimate_target_speaker_memory_bytes, sign_target_speaker_promotion_evidence,
    SignedTargetSpeakerPromotionEvidence, TargetSpeakerDecision, TargetSpeakerEnrollmentSummary,
    TargetSpeakerEvidenceIdentity, TargetSpeakerExtractionConfig, TargetSpeakerExtractionReport,
    TargetSpeakerExtractionResult, TargetSpeakerMetricOperator, TargetSpeakerMetricOutcome,
    TargetSpeakerModelIdentity, TargetSpeakerPresence, TargetSpeakerPresenceAssessment,
    TargetSpeakerPromotionEvidencePayload, TargetSpeakerSafetyGate, TargetSpeakerSafetyGateKind,
    TargetSpeakerSafetyMeasurements, TargetSpeakerStratumEvidence, TargetSpeakerStratumKind,
    MAX_TARGET_SPEAKER_ENROLLMENT_MILLIS, MAX_TARGET_SPEAKER_EVIDENCE_METRICS,
    MAX_TARGET_SPEAKER_EVIDENCE_STRATA, MAX_TARGET_SPEAKER_MIXTURE_SECONDS,
    MIN_TARGET_SPEAKER_ENROLLMENT_MILLIS, TARGET_SPEAKER_PROMOTION_EVIDENCE_SCHEMA,
    TARGET_SPEAKER_REPORT_SCHEMA, TARGET_SPEAKER_SCHEMA_VERSION,
};
#[cfg(feature = "bsrnn")]
pub use universal_restoration::restore_universal_audio;
pub use universal_restoration::{
    estimate_universal_restoration_memory_bytes, sign_universal_promotion_evidence,
    SignedUniversalPromotionEvidence, UniversalDegradation, UniversalDegradationEvidence,
    UniversalMaskRun, UniversalMaskState, UniversalMetricOperator, UniversalMetricOutcome,
    UniversalModelFamily, UniversalModelIdentity, UniversalPromotionEvidencePayload,
    UniversalRenderRole, UniversalRestorationConfig, UniversalRestorationDecision,
    UniversalRestorationMask, UniversalRestorationReport, UniversalRestorationResult,
    UniversalSafetyGate, UniversalSafetyGateKind, UniversalSafetyMeasurements,
    UniversalStratumEvidence, MAX_UNIVERSAL_EVIDENCE_METRICS, MAX_UNIVERSAL_EVIDENCE_STRATA,
    MAX_UNIVERSAL_MASK_RUNS, UNIVERSAL_PROMOTION_EVIDENCE_SCHEMA,
    UNIVERSAL_RESTORATION_MASK_SCHEMA, UNIVERSAL_RESTORATION_REPORT_SCHEMA,
    UNIVERSAL_RESTORATION_SCHEMA_VERSION,
};
pub use watch::{
    WatchCycleReport, WatchFolder, WatchFolderConfig, WatchFolderJob, WatchProcessError,
    WATCH_CYCLE_SCHEMA, WATCH_QUARANTINE_SCHEMA, WATCH_SCHEMA_VERSION, WATCH_STATE_SCHEMA,
};
pub use window::{WindowParams, WindowType};

/// Encode audio and optional metadata into a staged file, then publish it in
/// one filesystem commit.
pub fn write_audio_transactional(
    output: impl AsRef<std::path::Path>,
    audio: &Audio,
    encode_options: EncodeOptions,
    metadata_snapshot: Option<metadata::Metadata>,
    commit_mode: CommitMode,
) -> Result<(), String> {
    let output = output.as_ref();
    let format = OutputFormat::from_path(output)?;
    write_audio_transactional_as(
        output,
        format,
        audio,
        encode_options,
        metadata_snapshot,
        commit_mode,
    )
}

/// Encode audio using a format selected during preflight, then publish it in
/// one filesystem commit without re-inferring the codec from the path.
pub fn write_audio_transactional_as(
    output: impl AsRef<std::path::Path>,
    format: OutputFormat,
    audio: &Audio,
    encode_options: EncodeOptions,
    metadata_snapshot: Option<metadata::Metadata>,
    commit_mode: CommitMode,
) -> Result<(), String> {
    let output = output.as_ref();
    format.validate_config(audio, &encode_options)?;
    let mut transaction = AtomicOutput::new(output)?;
    encode::write_audio_to_file(transaction.file_mut(), format, audio, encode_options)?;
    if let Some(metadata_snapshot) = metadata_snapshot {
        metadata::write_extended_to_file(metadata_snapshot, transaction.file_mut())?;
    }
    transaction.commit(commit_mode)
}

/// Denoise a WAV file end-to-end, writing the result to `output`.
pub fn denoise_file<P1, P2>(input: P1, output: P2, config: DenoiserConfig) -> Result<Audio, String>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    denoise_file_with_backend(input, output, config, Backend::Classical)
}

/// Denoise with an explicit backend (classical / rnnoise / deepfilter).
pub fn denoise_file_with_backend<P1, P2>(
    input: P1,
    output: P2,
    config: DenoiserConfig,
    backend: Backend,
) -> Result<Audio, String>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    denoise_file_with_backend_opts(input, output, config, backend, EncodeOptions::default())
}

/// Denoise with explicit backend and output encode options.
pub fn denoise_file_with_backend_opts<P1, P2>(
    input: P1,
    output: P2,
    config: DenoiserConfig,
    backend: Backend,
    encode_opts: EncodeOptions,
) -> Result<Audio, String>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    denoise_file_with_backend_config(
        input,
        output,
        config,
        backend,
        encode_opts,
        BackendOptions::default(),
    )
}

/// Denoise with explicit backend, encoder, and backend-specific model options.
pub fn denoise_file_with_backend_config<P1, P2>(
    input: P1,
    output: P2,
    config: DenoiserConfig,
    backend: Backend,
    encode_opts: EncodeOptions,
    backend_options: BackendOptions,
) -> Result<Audio, String>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    let input = input.as_ref();
    let output = output.as_ref();
    // The file's decoded rate replaces the caller's placeholder rate. Validate
    // every rate-independent field before touching the filesystem while using
    // a harmless valid rate solely for this preflight pass.
    let mut preflight_config = config.clone();
    preflight_config.sample_rate = 1;
    preflight_config
        .validate_config()
        .map_err(|error| error.to_string())?;
    backend_options
        .validate_resolved_config(backend)
        .map_err(|error| error.to_string())?;
    let format = OutputFormat::from_path(output)?;
    encode_opts.validate_options(format)?;
    backend_options.validate_resolved_resources(backend)?;
    let (metadata, mut audio) = read_file_input_snapshot(input, || {})?;
    format.validate_config(&audio, &encode_opts)?;
    denoise_audio_with_backend_config(&mut audio, config, backend, &backend_options)?;
    write_audio_transactional_as(
        output,
        format,
        &audio,
        encode_opts,
        metadata,
        CommitMode::Replace,
    )?;
    Ok(audio)
}

/// Read metadata and decoded audio from one validated filesystem object.
///
/// The callback exists so tests can deterministically replace the pathname at
/// the exact boundary which used to separate two independent opens.
fn read_file_input_snapshot(
    input: &std::path::Path,
    after_metadata: impl FnOnce(),
) -> Result<(Option<metadata::Metadata>, Audio), String> {
    let mut session = AudioInputSession::open(input)?;
    let metadata = session.read_metadata()?;
    after_metadata();
    let audio = read_audio_from_session(&mut session)?;
    Ok((metadata, audio))
}

/// Process already-decoded audio in place. This is the path used by stdin and
/// embedders that do not have filesystem-backed input.
pub fn denoise_audio_with_backend_config(
    audio: &mut Audio,
    config: DenoiserConfig,
    backend: Backend,
    backend_options: &BackendOptions,
) -> Result<std::time::Duration, String> {
    let session = BackendSession::prepare(backend, backend_options.clone())?;
    denoise_audio_with_backend_session(audio, config, &session)
}

/// Process decoded audio with a prepared backend graph.
///
/// Reuse the same session across files or VAD regions to avoid reparsing and
/// reoptimizing model weights. Per-call recurrent state remains isolated.
pub fn denoise_audio_with_backend_session(
    audio: &mut Audio,
    config: DenoiserConfig,
    session: &BackendSession,
) -> Result<std::time::Duration, String> {
    let (processed, elapsed) = process_audio_copy_with_backend_session(audio, config, session)?;
    *audio = processed;
    Ok(elapsed)
}

pub(crate) fn process_audio_copy_with_backend_session(
    audio: &Audio,
    mut config: DenoiserConfig,
    session: &BackendSession,
) -> Result<(Audio, std::time::Duration), String> {
    config.sample_rate = audio.sample_rate;
    config
        .validate_config()
        .map_err(|error| error.to_string())?;
    let mut input = audio.try_clone_fallible("denoising input")?;
    input.sanitize_samples();
    let t0 = std::time::Instant::now();
    let channels = if config.vad {
        process_with_vad(session, &input.channels, input.sample_rate, &config)?
    } else {
        session.process(&input.channels, input.sample_rate, &config)?
    };
    let mut processed = Audio {
        sample_rate: audio.sample_rate,
        channels,
        bits_per_sample: audio.bits_per_sample,
        sample_format: audio.sample_format,
        channel_mask: audio.channel_mask,
    };
    processed.sanitize_samples();
    let elapsed = t0.elapsed();
    eprintln!(
        "denoize: {:?} | {}ch x {} frames ({:.2}s) in {:.2?} ({:.1}x realtime)",
        session.backend(),
        processed.channels(),
        processed.frames(),
        processed.frames() as f64 / processed.sample_rate as f64,
        elapsed,
        (processed.frames() as f64 / processed.sample_rate as f64)
            / elapsed.as_secs_f64().max(1e-9),
    );
    Ok((processed, elapsed))
}

fn process_with_vad(
    session: &BackendSession,
    channels: &[Vec<f64>],
    sample_rate: u32,
    config: &DenoiserConfig,
) -> Result<Vec<Vec<f64>>, String> {
    let regions = vad::speech_regions(channels, sample_rate);
    let fade_frames = (sample_rate as usize / 50).max(1); // 20 ms
    let silence_gain = config.vad_silence_gain;
    let speech_mix = config.vad_speech_mix;
    let mut output: Vec<Vec<f64>> = channels
        .iter()
        .map(|channel| channel.iter().map(|sample| sample * silence_gain).collect())
        .collect();
    for region in regions {
        let input: Vec<Vec<f64>> = channels
            .iter()
            .map(|channel| {
                channel[region.start.min(channel.len())..region.end.min(channel.len())].to_vec()
            })
            .collect();
        let enhanced = session.process(&input, sample_rate, config)?;
        for (channel_index, enhanced_channel) in enhanced.iter().enumerate() {
            let Some(destination) = output.get_mut(channel_index) else {
                continue;
            };
            let original = &channels[channel_index];
            for (offset, sample) in enhanced_channel.iter().enumerate() {
                let index = region.start + offset;
                if index >= destination.len() || index >= original.len() || index >= region.end {
                    break;
                }
                let target = sample * speech_mix + original[index] * (1.0 - speech_mix);
                let weight = vad_mix_weight(offset, region.end - region.start, fade_frames);
                destination[index] = destination[index] * (1.0 - weight) + target * weight;
            }
        }
    }
    Ok(output)
}

fn vad_mix_weight(offset: usize, length: usize, fade_frames: usize) -> f64 {
    // Start and end at the attenuated signal so a processed region cannot
    // introduce a discontinuity at either handoff.
    let from_start = offset.min(fade_frames) as f64 / fade_frames.max(1) as f64;
    let from_end =
        length.saturating_sub(offset + 1).min(fade_frames) as f64 / fade_frames.max(1) as f64;
    from_start.min(from_end).clamp(0.0, 1.0)
}

#[cfg(test)]
mod vad_mix_tests {
    use super::{
        process_with_vad, vad, vad_mix_weight, Backend, BackendOptions, BackendSession,
        DenoiserConfig,
    };

    #[test]
    fn fades_vad_region_edges_without_exceeding_unity() {
        assert_eq!(vad_mix_weight(0, 100, 10), 0.0);
        assert_eq!(vad_mix_weight(99, 100, 10), 0.0);
        assert_eq!(vad_mix_weight(50, 100, 10), 1.0);
        assert!((vad_mix_weight(5, 100, 10) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn fade_weights_are_bounded_monotonic_and_slope_limited() {
        let fade_frames = 10;
        let weights: Vec<_> = (0..100)
            .map(|offset| vad_mix_weight(offset, 100, fade_frames))
            .collect();

        assert!(weights.iter().all(|weight| (0.0..=1.0).contains(weight)));
        assert!(weights
            .windows(2)
            .all(|pair| { (pair[1] - pair[0]).abs() <= 1.0 / fade_frames as f64 + f64::EPSILON }));
        assert!(weights[..=fade_frames]
            .windows(2)
            .all(|pair| pair[1] >= pair[0]));
        assert!(weights[fade_frames..]
            .windows(2)
            .all(|pair| pair[1] <= pair[0]));
        assert_eq!(weights.first().copied(), Some(0.0));
        assert_eq!(weights.last().copied(), Some(0.0));
    }

    fn test_config(sample_rate: u32) -> DenoiserConfig {
        let mut config = DenoiserConfig::default(sample_rate);
        config.vad = true;
        config.vad_silence_gain = 0.2;
        config.vad_speech_mix = 0.0;
        config.sanitized()
    }

    fn test_session() -> BackendSession {
        BackendSession::prepare(Backend::Classical, BackendOptions::default()).unwrap()
    }

    #[test]
    fn vad_applies_configured_gain_to_non_speech_audio() {
        let sample_rate = 16_000;
        let input: Vec<f64> = (0..sample_rate)
            .map(|index| {
                1.0e-5
                    * (2.0 * std::f64::consts::PI * 37.0 * index as f64 / sample_rate as f64).sin()
            })
            .collect();
        assert!(vad::speech_regions(std::slice::from_ref(&input), sample_rate).is_empty());

        let output = process_with_vad(
            &test_session(),
            std::slice::from_ref(&input),
            sample_rate,
            &test_config(sample_rate),
        )
        .unwrap();

        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), input.len());
        for (actual, original) in output[0].iter().zip(&input) {
            assert!((actual - original * 0.2).abs() < 1e-20);
        }
    }

    #[test]
    fn vad_crossfade_matches_expected_edges_without_clicks() {
        let sample_rate = 16_000;
        let frames = sample_rate as usize * 2;
        let active_start = sample_rate as usize / 2;
        let active_end = sample_rate as usize * 3 / 2;
        let transition = sample_rate as usize / 20;
        let input: Vec<f64> = (0..frames)
            .map(|index| {
                let envelope = if index < active_start.saturating_sub(transition) {
                    0.0
                } else if index < active_start {
                    let position = (index - (active_start - transition)) as f64 / transition as f64;
                    let smooth = position * position * (3.0 - 2.0 * position);
                    0.3 * smooth
                } else if index < active_end {
                    0.3
                } else if index < active_end + transition {
                    let position = (index - active_end) as f64 / transition as f64;
                    let smooth = position * position * (3.0 - 2.0 * position);
                    0.3 * (1.0 - smooth)
                } else {
                    0.0
                };
                envelope
                    * (2.0 * std::f64::consts::PI * 80.0 * index as f64 / sample_rate as f64).sin()
            })
            .collect();
        let regions = vad::speech_regions(std::slice::from_ref(&input), sample_rate);
        assert!(!regions.is_empty());

        let output = process_with_vad(
            &test_session(),
            std::slice::from_ref(&input),
            sample_rate,
            &test_config(sample_rate),
        )
        .unwrap();
        assert_eq!(output[0].len(), input.len());
        assert!(output[0].iter().all(|sample| sample.is_finite()));

        let silence_gain = 0.2;
        let mut expected: Vec<f64> = input.iter().map(|sample| sample * silence_gain).collect();
        for region in &regions {
            for offset in 0..region.end.saturating_sub(region.start) {
                let index = region.start + offset;
                if index >= expected.len() {
                    break;
                }
                let weight =
                    vad_mix_weight(offset, region.end - region.start, sample_rate as usize / 50);
                expected[index] = expected[index] * (1.0 - weight) + input[index] * weight;
            }
        }
        for (actual, expected) in output[0].iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }

        for region in regions {
            if region.start > 0 {
                let jump = (output[0][region.start] - output[0][region.start - 1]).abs();
                assert!(jump < 0.02, "VAD start boundary jump: {jump}");
            }
            if region.end < output[0].len() {
                let jump = (output[0][region.end] - output[0][region.end - 1]).abs();
                assert!(jump < 0.02, "VAD end boundary jump: {jump}");
            }
        }
    }
}

#[cfg(test)]
mod input_safety_tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn file_input_snapshot_stays_on_one_inode_when_path_changes_after_metadata() {
        use lofty::tag::{Accessor as _, Tag, TagType};

        fn tagged_wav(path: &std::path::Path, title: &str, samples: Vec<f64>) {
            let audio = Audio {
                sample_rate: 16_000,
                channels: vec![samples],
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
                channel_mask: None,
            };
            write_wav(path, &audio).unwrap();
            let mut tag = Tag::new(TagType::RiffInfo);
            tag.set_title(title.into());
            metadata::write(tag, path).unwrap();
        }

        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.wav");
        let replacement = root.path().join("replacement.wav");
        tagged_wav(&input, "original inode", vec![0.25]);
        tagged_wav(&replacement, "replacement inode", vec![-0.5, -0.25]);

        let (metadata, audio) = read_file_input_snapshot(&input, || {
            std::fs::rename(&replacement, &input).unwrap();
        })
        .unwrap();

        let metadata = metadata.expect("original input has metadata");
        assert_eq!(metadata.tag().title().as_deref(), Some("original inode"));
        assert_eq!(audio.frames(), 1);
        assert!((audio.channels[0][0] - 0.25).abs() < 1e-3);

        let replacement_audio = read_audio(&input).unwrap();
        assert_eq!(replacement_audio.frames(), 2);
        assert_eq!(
            metadata::read_extended(&input)
                .unwrap()
                .expect("replacement input has metadata")
                .tag()
                .title()
                .as_deref(),
            Some("replacement inode")
        );
    }

    #[test]
    fn high_level_processing_sanitizes_nonfinite_samples_and_keeps_empty_audio_safe() {
        let mut audio = Audio {
            sample_rate: 16_000,
            channels: vec![vec![f64::NAN, f64::INFINITY, -f64::INFINITY, 2.0, -2.0]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        denoise_audio_with_backend_config(
            &mut audio,
            DenoiserConfig::default(16_000),
            Backend::Classical,
            &BackendOptions::default(),
        )
        .unwrap();
        assert!(audio.channels[0].iter().all(|sample| sample.is_finite()));
        assert!(audio.channels[0].iter().all(|sample| sample.abs() <= 1.0));

        let mut empty = Audio {
            sample_rate: 16_000,
            channels: vec![Vec::new()],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        denoise_audio_with_backend_config(
            &mut empty,
            DenoiserConfig::default(16_000),
            Backend::Classical,
            &BackendOptions::default(),
        )
        .unwrap();
        assert_eq!(empty.frames(), 0);
    }

    #[test]
    fn classical_high_level_processing_rejects_invalid_dpss_bandwidth() {
        let mut audio = Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.25; 512]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let original = audio.channels.clone();
        let mut config = DenoiserConfig::default(audio.sample_rate);
        config.window = WindowType::Dpss;
        config.window_params.dpss_bandwidth = crate::window::MAX_DENOISER_DPSS_NW + 0.5;

        let error = denoise_audio_with_backend_config(
            &mut audio,
            config,
            Backend::Classical,
            &BackendOptions::default(),
        )
        .unwrap_err();

        assert!(
            error.contains("DPSS bandwidth"),
            "unexpected error: {error}"
        );
        assert_eq!(audio.channels, original);
    }

    #[test]
    fn invalid_decoded_config_leaves_audio_completely_unchanged() {
        let mut audio = Audio {
            sample_rate: 16_000,
            channels: vec![vec![2.0, -2.0, 0.25]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let original = audio.clone();
        let mut config = DenoiserConfig::default(48_000);
        config.vad_speech_mix = f64::NAN;

        let error = denoise_audio_with_backend_config(
            &mut audio,
            config,
            Backend::Classical,
            &BackendOptions::default(),
        )
        .unwrap_err();

        assert!(
            error.contains("vad_speech_mix"),
            "unexpected error: {error}"
        );
        assert_eq!(audio.sample_rate, original.sample_rate);
        assert_eq!(audio.channels, original.channels);
        assert_eq!(audio.bits_per_sample, original.bits_per_sample);
        assert_eq!(audio.sample_format, original.sample_format);
        assert_eq!(audio.channel_mask, original.channel_mask);
    }

    #[test]
    fn decoded_effective_sample_rate_is_validated_before_mutation() {
        let mut audio = Audio {
            sample_rate: 0,
            channels: vec![vec![2.0]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let original = audio.channels.clone();
        let error = denoise_audio_with_backend_config(
            &mut audio,
            DenoiserConfig::default(48_000),
            Backend::Classical,
            &BackendOptions::default(),
        )
        .unwrap_err();
        assert!(error.contains("sample_rate"), "unexpected error: {error}");
        assert_eq!(audio.channels, original);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn decoded_model_resource_is_validated_before_mutation() {
        let mut audio = Audio {
            sample_rate: 48_000,
            channels: vec![vec![2.0]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let original = audio.channels.clone();
        let backend_options = BackendOptions {
            onnx: Some(OnnxModelConfig {
                path: "model-that-does-not-exist.onnx".into(),
                sample_rate: 48_000,
            }),
            ..BackendOptions::default()
        };

        let error = denoise_audio_with_backend_config(
            &mut audio,
            DenoiserConfig::default(48_000),
            Backend::Onnx,
            &backend_options,
        )
        .unwrap_err();

        assert!(
            error.contains("model does not exist"),
            "unexpected error: {error}"
        );
        assert_eq!(audio.channels, original);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn malformed_model_failure_does_not_mutate_decoded_audio() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("malformed.onnx");
        std::fs::write(&model, b"not an ONNX graph").unwrap();
        let mut audio = Audio {
            sample_rate: 48_000,
            channels: vec![vec![2.0, -2.0, 0.25]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let original = audio.clone();
        let backend_options = BackendOptions {
            onnx: Some(OnnxModelConfig {
                path: model,
                sample_rate: 48_000,
            }),
            ..BackendOptions::default()
        };

        let error = denoise_audio_with_backend_config(
            &mut audio,
            DenoiserConfig::default(48_000),
            Backend::Onnx,
            &backend_options,
        )
        .unwrap_err();

        assert!(!error.is_empty());
        assert_eq!(audio.sample_rate, original.sample_rate);
        assert_eq!(audio.channels, original.channels);
        assert_eq!(audio.bits_per_sample, original.bits_per_sample);
        assert_eq!(audio.sample_format, original.sample_format);
        assert_eq!(audio.channel_mask, original.channel_mask);
    }

    #[test]
    fn file_processing_validates_dpss_before_reading_input() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("missing.wav");
        let output = root.path().join("output.wav");
        let mut config = DenoiserConfig::default(16_000);
        config.window = WindowType::Dpss;
        config.window_params.dpss_bandwidth = crate::window::MAX_DENOISER_DPSS_NW + 0.5;

        let error = denoise_file_with_backend_config(
            &input,
            &output,
            config,
            Backend::Classical,
            EncodeOptions::default(),
            BackendOptions::default(),
        )
        .unwrap_err();

        assert!(
            error.contains("DPSS bandwidth"),
            "unexpected error: {error}"
        );
        assert!(!output.exists());
    }

    #[test]
    fn file_processing_rejects_rate_independent_config_before_missing_input() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("missing.wav");
        let output = root.path().join("output.wav");
        let mut config = DenoiserConfig::default(0);
        config.strength = f64::NAN;

        let error = denoise_file_with_backend_config(
            &input,
            &output,
            config,
            Backend::Classical,
            EncodeOptions::default(),
            BackendOptions::default(),
        )
        .unwrap_err();

        assert!(error.contains("strength"), "unexpected error: {error}");
        assert!(!error.contains("sample_rate"));
        assert!(!output.exists());
    }

    #[test]
    fn file_preflight_does_not_reject_placeholder_sample_rate() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("missing.wav");
        let output = root.path().join("output.wav");
        let config = DenoiserConfig::default(0);

        let error = denoise_file_with_backend_config(
            &input,
            &output,
            config,
            Backend::Classical,
            EncodeOptions::default(),
            BackendOptions::default(),
        )
        .unwrap_err();

        assert!(!error.contains("sample_rate"), "unexpected error: {error}");
        assert!(!output.exists());
    }

    #[test]
    fn file_processing_validates_backend_options_before_missing_input() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("missing.wav");
        let output = root.path().join("output.wav");
        let backend_options = BackendOptions {
            onnx: Some(OnnxModelConfig {
                path: root.path().join("model-that-must-not-be-opened.onnx"),
                sample_rate: 0,
            }),
            ..BackendOptions::default()
        };

        let error = denoise_file_with_backend_config(
            &input,
            &output,
            DenoiserConfig::default(0),
            Backend::Classical,
            EncodeOptions::default(),
            backend_options,
        )
        .unwrap_err();

        assert!(
            error.contains("backend_options.onnx.sample_rate"),
            "unexpected error: {error}"
        );
        assert!(!output.exists());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn file_processing_validates_output_format_before_model_path_io() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("missing.wav");
        let output = root.path().join("output.invalid");
        let backend_options = BackendOptions {
            onnx: Some(OnnxModelConfig {
                path: root.path().join("model-that-must-not-be-opened.onnx"),
                sample_rate: 48_000,
            }),
            ..BackendOptions::default()
        };

        let error = denoise_file_with_backend_config(
            &input,
            &output,
            DenoiserConfig::default(0),
            Backend::Onnx,
            EncodeOptions::default(),
            backend_options,
        )
        .unwrap_err();

        assert!(
            error.contains("unsupported output format"),
            "unexpected error: {error}"
        );
        assert!(!error.contains("model does not exist"));
        assert!(!output.exists());
    }

    #[cfg(feature = "m4a-encode")]
    #[test]
    fn file_processing_validates_encode_options_before_missing_input() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("missing.wav");
        let output = root.path().join("output.aac");
        let mut encode_options = EncodeOptions::default();
        encode_options.m4a_bitrate_bps = 0;

        let error = denoise_file_with_backend_config(
            &input,
            &output,
            DenoiserConfig::default(0),
            Backend::Classical,
            encode_options,
            BackendOptions::default(),
        )
        .unwrap_err();

        assert!(error.contains("bitrate must be greater than zero"));
        assert!(!error.contains("missing.wav"));
        assert!(!output.exists());
    }

    #[test]
    fn transactional_encode_validation_precedes_output_staging() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("output.wav");
        std::fs::write(&output, b"existing output").unwrap();
        let invalid_audio = Audio {
            sample_rate: 48_000,
            channels: Vec::new(),
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };

        let error = write_audio_transactional_as(
            &output,
            OutputFormat::Wav,
            &invalid_audio,
            EncodeOptions::default(),
            None,
            CommitMode::Replace,
        )
        .unwrap_err();

        assert!(error.contains("at least one channel"));
        assert_eq!(std::fs::read(&output).unwrap(), b"existing output");
        assert!(std::fs::read_dir(root.path())
            .unwrap()
            .filter_map(Result::ok)
            .all(|entry| !entry.file_name().to_string_lossy().starts_with(".denoize-")));
    }
}
