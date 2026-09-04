//! Off-callback neural CLAP processor.
//!
//! The host audio thread owns only fixed-size buffers and bounded lock-free
//! queues. Model verification, graph preparation, resampling, recurrent state,
//! and inference remain on one permanent worker thread.

use clack_extensions::audio_ports::*;
use clack_extensions::audio_ports_config::*;
use clack_extensions::gui::*;
use clack_extensions::latency::{PluginLatency, PluginLatencyImpl};
use clack_extensions::params::*;
use clack_extensions::state::{PluginState, PluginStateImpl};
use clack_plugin::events::spaces::CoreEventSpace;
use clack_plugin::prelude::*;
use clack_plugin::process::audio::{ChannelPair, PairedChannels, SampleType};
use clack_plugin::stream::{InputStream, OutputStream};
use crossbeam_queue::ArrayQueue;
use denoize::{
    AcceleratorRuntime, Backend, BackendOptions, ChannelMode, DenoiserConfig, GtcrnModel,
    NEURAL_DAW_BLOCK_POOL_SIZE, NEURAL_DAW_LATENCY_CHUNKS, NEURAL_DAW_MAX_SAMPLE_RATE,
    NEURAL_DAW_MODEL_ID, NEURAL_DAW_MODEL_SHA256, NEURAL_DAW_PLUGIN_ID, NEURAL_DAW_QUEUE_BLOCKS,
    NeuralDawModel, NeuralDawOverloadFallback as OverloadFallback,
    NeuralDawParameters as NeuralParameters, NeuralDawPortConfiguration as NeuralPortConfiguration,
    NeuralDawSessionState as NeuralSessionState, OnnxModelConfig, StreamingBackendSession,
    neural_daw_chunk_frames, select_accelerator_for_options,
};
#[cfg(feature = "experimental-dpdfnet-hq")]
use denoize::{
    DpdfnetModel, NEURAL_HQ_DAW_MODEL_ID, NEURAL_HQ_DAW_MODEL_SHA256, NEURAL_HQ_DAW_PLUGIN_ID,
};
use denoize_plugin_editor::{ControlKind, DisplayUnit, EditorModel, ParameterSpec, PluginEditor};
use std::collections::VecDeque;
use std::ffi::CStr;
use std::fmt::Write as _;
use std::io::{Read, Write as _};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub(crate) const NEURAL_PLUGIN_ID: &str = NEURAL_DAW_PLUGIN_ID;
#[cfg(feature = "experimental-dpdfnet-hq")]
pub(crate) const NEURAL_HQ_PLUGIN_ID: &str = NEURAL_HQ_DAW_PLUGIN_ID;
const STATE_LIMIT_BYTES: u64 = 64 * 1024;
const MODEL_ID: &str = NEURAL_DAW_MODEL_ID;
const LATENCY_CHUNKS: u32 = NEURAL_DAW_LATENCY_CHUNKS;
// Keep a complete declared-latency window available to the worker. A shorter
// input queue can turn a recoverable host scheduling pause into a dropped
// block and recurrent-state discontinuity before the 240 ms deadline expires.
const QUEUE_BLOCKS: usize = NEURAL_DAW_QUEUE_BLOCKS;
const BLOCK_POOL_SIZE: usize = NEURAL_DAW_BLOCK_POOL_SIZE;
const WORKER_WARMUP_EXTRA_BLOCKS: usize = QUEUE_BLOCKS + 8;
const WORKER_POLL: Duration = Duration::from_micros(100);
const MAX_SAMPLE_RATE: u32 = NEURAL_DAW_MAX_SAMPLE_RATE;
const MAX_OUTPUT_PEAK: f64 = 4.0;
const HOST_EVIDENCE_WARMUP_LATENCIES: u64 = 4;

const MONO_CONFIG_ID: ClapId = ClapId::new(101);
const STEREO_CONFIG_ID: ClapId = ClapId::new(102);
const INPUT_PORT_ID: ClapId = ClapId::new(110);
const SIDECHAIN_PORT_ID: ClapId = ClapId::new(111);
const OUTPUT_PORT_ID: ClapId = ClapId::new(112);

const PARAM_BYPASS: ClapId = ClapId::new(0);
const PARAM_MIX: ClapId = ClapId::new(1);
const PARAM_OUTPUT_GAIN: ClapId = ClapId::new(2);
const PARAM_FALLBACK: ClapId = ClapId::new(3);
const PARAMETER_COUNT: u32 = 4;

const FALLBACK_LABELS: &[&str] = &["Delayed Dry", "Last Safe Gain", "Silence"];
const EDITOR_PARAMETERS: &[ParameterSpec] = &[
    ParameterSpec {
        id: 0,
        name: "Bypass",
        minimum: 0.0,
        maximum: 1.0,
        default: 0.0,
        step: 1.0,
        page_step: 1.0,
        kind: ControlKind::Toggle,
        unit: DisplayUnit::Plain,
    },
    ParameterSpec {
        id: 1,
        name: "Mix",
        minimum: 0.0,
        maximum: 1.0,
        default: 1.0,
        step: 0.01,
        page_step: 0.1,
        kind: ControlKind::Continuous,
        unit: DisplayUnit::Percent,
    },
    ParameterSpec {
        id: 2,
        name: "Output Gain",
        minimum: -24.0,
        maximum: 24.0,
        default: 0.0,
        step: 0.5,
        page_step: 3.0,
        kind: ControlKind::Continuous,
        unit: DisplayUnit::Decibels,
    },
    ParameterSpec {
        id: 3,
        name: "Overload Fallback",
        minimum: 0.0,
        maximum: 2.0,
        default: 0.0,
        step: 1.0,
        page_step: 1.0,
        kind: ControlKind::Choice(FALLBACK_LABELS),
        unit: DisplayUnit::Plain,
    },
];

pub(crate) struct NeuralPlugin;
#[cfg(feature = "experimental-dpdfnet-hq")]
pub(crate) struct NeuralHqPlugin;

impl Plugin for NeuralPlugin {
    type AudioProcessor<'a> = NeuralAudioProcessor<'a>;
    type Shared<'a> = NeuralShared;
    type MainThread<'a> = NeuralMainThread<'a>;

    fn declare_extensions(
        builder: &mut PluginExtensions<Self>,
        _shared: Option<&Self::Shared<'_>>,
    ) {
        builder
            .register::<PluginAudioPorts>()
            .register::<PluginAudioPortsConfig>()
            .register::<PluginAudioPortsConfigInfo>()
            .register::<super::gui_contract::DenoizePluginGui>()
            .register::<PluginParams>()
            .register::<PluginState>()
            .register::<PluginLatency>();
    }
}

impl DefaultPluginFactory for NeuralPlugin {
    fn get_descriptor() -> PluginDescriptor {
        use clack_plugin::plugin::features::*;

        PluginDescriptor::new(NEURAL_PLUGIN_ID, "denoize Neural")
            .with_vendor("denoize")
            .with_url("https://github.com/penguin425/denoize")
            .with_version(env!("CARGO_PKG_VERSION"))
            .with_description("Off-callback causal GTCRN speech restoration")
            .with_features([AUDIO_EFFECT, RESTORATION, MONO, STEREO])
    }

    fn new_shared(_host: HostSharedHandle<'_>) -> Result<Self::Shared<'_>, PluginError> {
        NeuralShared::new_for_model(NeuralDawModel::Gtcrn)
    }

    fn new_main_thread<'a>(
        host: HostMainThreadHandle<'a>,
        shared: &'a Self::Shared<'a>,
    ) -> Result<Self::MainThread<'a>, PluginError> {
        let host_gui = host.get_extension::<HostGui>();
        Ok(NeuralMainThread {
            host,
            shared,
            host_gui,
            editor: None,
            pending_automation: None,
            port_configuration: NeuralPortConfiguration::Stereo,
            latency_frames: 0,
        })
    }
}

#[cfg(feature = "experimental-dpdfnet-hq")]
impl Plugin for NeuralHqPlugin {
    type AudioProcessor<'a> = NeuralAudioProcessor<'a>;
    type Shared<'a> = NeuralShared;
    type MainThread<'a> = NeuralMainThread<'a>;

    fn declare_extensions(
        builder: &mut PluginExtensions<Self>,
        _shared: Option<&Self::Shared<'_>>,
    ) {
        builder
            .register::<PluginAudioPorts>()
            .register::<PluginAudioPortsConfig>()
            .register::<PluginAudioPortsConfigInfo>()
            .register::<super::gui_contract::DenoizePluginGui>()
            .register::<PluginParams>()
            .register::<PluginState>()
            .register::<PluginLatency>();
    }
}

#[cfg(feature = "experimental-dpdfnet-hq")]
impl DefaultPluginFactory for NeuralHqPlugin {
    fn get_descriptor() -> PluginDescriptor {
        use clack_plugin::plugin::features::*;

        PluginDescriptor::new(NEURAL_HQ_PLUGIN_ID, "denoize Neural HQ")
            .with_vendor("denoize")
            .with_url("https://github.com/penguin425/denoize")
            .with_version(env!("CARGO_PKG_VERSION"))
            .with_description("Experimental off-callback DPDFNet-2 fullband speech restoration")
            .with_features([AUDIO_EFFECT, RESTORATION, MONO, STEREO])
    }

    fn new_shared(_host: HostSharedHandle<'_>) -> Result<Self::Shared<'_>, PluginError> {
        NeuralShared::new_for_model(NeuralDawModel::Dpdfnet2)
    }

    fn new_main_thread<'a>(
        host: HostMainThreadHandle<'a>,
        shared: &'a Self::Shared<'a>,
    ) -> Result<Self::MainThread<'a>, PluginError> {
        let host_gui = host.get_extension::<HostGui>();
        Ok(NeuralMainThread {
            host,
            shared,
            host_gui,
            editor: None,
            pending_automation: None,
            port_configuration: NeuralPortConfiguration::Stereo,
            latency_frames: 0,
        })
    }
}

pub(crate) struct NeuralShared {
    model: NeuralDawModel,
    parameters: SharedParameters,
    reset_generation: AtomicU64,
    overload_blocks: AtomicU64,
    late_blocks: AtomicU64,
    invalid_blocks: AtomicU64,
    worker_errors: Arc<AtomicU64>,
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct WorkerMetrics {
    overload_blocks: u64,
    late_blocks: u64,
    invalid_blocks: u64,
    worker_errors: u64,
}

impl WorkerMetrics {
    fn saturating_since(self, baseline: Self) -> Self {
        Self {
            overload_blocks: self
                .overload_blocks
                .saturating_sub(baseline.overload_blocks),
            late_blocks: self.late_blocks.saturating_sub(baseline.late_blocks),
            invalid_blocks: self.invalid_blocks.saturating_sub(baseline.invalid_blocks),
            worker_errors: self.worker_errors.saturating_sub(baseline.worker_errors),
        }
    }
}

impl NeuralShared {
    #[cfg(test)]
    fn new() -> Result<Self, PluginError> {
        Self::new_for_model(NeuralDawModel::Gtcrn)
    }

    fn new_for_model(model: NeuralDawModel) -> Result<Self, PluginError> {
        Ok(Self {
            model,
            parameters: SharedParameters::new(NeuralParameters::default(), model.display_name())?,
            reset_generation: AtomicU64::new(1),
            overload_blocks: AtomicU64::new(0),
            late_blocks: AtomicU64::new(0),
            invalid_blocks: AtomicU64::new(0),
            worker_errors: Arc::new(AtomicU64::new(0)),
        })
    }

    fn restore(&self, parameters: NeuralParameters) {
        self.parameters.store(parameters);
        self.reset_generation.fetch_add(1, Ordering::Release);
    }

    fn worker_metrics(&self) -> WorkerMetrics {
        WorkerMetrics {
            overload_blocks: self.overload_blocks.load(Ordering::Relaxed),
            late_blocks: self.late_blocks.load(Ordering::Relaxed),
            invalid_blocks: self.invalid_blocks.load(Ordering::Relaxed),
            worker_errors: self.worker_errors.load(Ordering::Relaxed),
        }
    }
}

impl PluginShared<'_> for NeuralShared {}

pub(crate) struct NeuralMainThread<'a> {
    host: HostMainThreadHandle<'a>,
    shared: &'a NeuralShared,
    host_gui: Option<HostGui>,
    editor: Option<PluginEditor>,
    pending_automation: Option<super::PendingAutomation>,
    port_configuration: NeuralPortConfiguration,
    latency_frames: u32,
}

impl<'a> PluginMainThread<'a, NeuralShared> for NeuralMainThread<'a> {
    fn on_main_thread(&mut self) {
        if let Some(editor) = &self.editor {
            editor.host_main_thread_callback();
        }
    }
}

impl PluginGuiImpl for NeuralMainThread<'_> {
    fn is_api_supported(&mut self, configuration: GuiConfiguration<'_>) -> bool {
        PluginEditor::supports(configuration)
    }

    fn get_preferred_api(&mut self) -> Option<GuiConfiguration<'_>> {
        PluginEditor::preferred_configuration()
    }

    fn create(&mut self, configuration: GuiConfiguration<'_>) -> Result<(), PluginError> {
        if self.editor.is_some() {
            return Err(PluginError::Message(
                "denoize Neural editor is already created",
            ));
        }
        self.editor = Some(PluginEditor::create(
            &self.host,
            self.host_gui,
            Arc::clone(&self.shared.parameters.editor),
            configuration,
        )?);
        Ok(())
    }

    fn destroy(&mut self) {
        self.editor.take();
    }

    fn set_scale(&mut self, scale: f64) -> Result<(), PluginError> {
        self.editor
            .as_ref()
            .ok_or(PluginError::Message("denoize Neural editor is not created"))?
            .set_scale(scale)
    }

    fn get_size(&mut self) -> Option<GuiSize> {
        self.editor.as_ref().map(PluginEditor::size)
    }

    fn can_resize(&mut self) -> bool {
        self.editor.as_ref().is_some_and(PluginEditor::can_resize)
    }

    fn get_resize_hints(&mut self) -> Option<GuiResizeHints> {
        self.editor.as_ref().map(PluginEditor::resize_hints)
    }

    fn adjust_size(&mut self, size: GuiSize) -> Option<GuiSize> {
        self.editor
            .as_ref()
            .and_then(|editor| editor.adjust_size(size))
    }

    fn set_size(&mut self, size: GuiSize) -> Result<(), PluginError> {
        self.editor
            .as_ref()
            .ok_or(PluginError::Message("denoize Neural editor is not created"))?
            .set_size(size)
    }

    fn set_parent(&mut self, window: clack_extensions::gui::Window<'_>) -> Result<(), PluginError> {
        self.editor
            .as_ref()
            .ok_or(PluginError::Message("denoize Neural editor is not created"))?
            .set_parent(window)
    }

    fn set_transient(
        &mut self,
        _window: clack_extensions::gui::Window<'_>,
    ) -> Result<(), PluginError> {
        Err(PluginError::Message(
            "denoize Neural editor does not support floating windows",
        ))
    }

    fn show(&mut self) -> Result<(), PluginError> {
        self.editor
            .as_ref()
            .ok_or(PluginError::Message("denoize Neural editor is not created"))?
            .show()
    }

    fn hide(&mut self) -> Result<(), PluginError> {
        self.editor
            .as_ref()
            .ok_or(PluginError::Message("denoize Neural editor is not created"))?
            .hide()
    }
}

impl PluginAudioPortsImpl for NeuralMainThread<'_> {
    fn count(&mut self, is_input: bool) -> u32 {
        if is_input { 2 } else { 1 }
    }

    fn get(&mut self, index: u32, is_input: bool, writer: &mut AudioPortInfoWriter) {
        let (channel_count, port_type) = port_shape(self.port_configuration);
        let info = match (is_input, index) {
            (true, 0) => AudioPortInfo {
                id: INPUT_PORT_ID,
                name: b"Input",
                channel_count,
                flags: AudioPortFlags::IS_MAIN | AudioPortFlags::SUPPORTS_64BITS,
                port_type: Some(port_type),
                in_place_pair: Some(OUTPUT_PORT_ID),
            },
            (true, 1) => AudioPortInfo {
                id: SIDECHAIN_PORT_ID,
                name: b"Reference (reserved)",
                channel_count,
                flags: AudioPortFlags::SUPPORTS_64BITS,
                port_type: Some(port_type),
                in_place_pair: None,
            },
            (false, 0) => AudioPortInfo {
                id: OUTPUT_PORT_ID,
                name: b"Output",
                channel_count,
                flags: AudioPortFlags::IS_MAIN | AudioPortFlags::SUPPORTS_64BITS,
                port_type: Some(port_type),
                in_place_pair: Some(INPUT_PORT_ID),
            },
            _ => return,
        };
        writer.set(&info);
    }
}

impl PluginAudioPortsConfigImpl for NeuralMainThread<'_> {
    fn count(&mut self) -> u32 {
        2
    }

    fn get(&mut self, index: u32, writer: &mut AudioPortConfigWriter) {
        let configuration = match index {
            0 => NeuralPortConfiguration::Mono,
            1 => NeuralPortConfiguration::Stereo,
            _ => return,
        };
        let (name, channel_count, port_type) = match configuration {
            NeuralPortConfiguration::Mono => {
                (b"Mono + reference".as_slice(), 1, AudioPortType::MONO)
            }
            NeuralPortConfiguration::Stereo => {
                (b"Stereo + reference".as_slice(), 2, AudioPortType::STEREO)
            }
        };
        let main = MainPortInfo {
            channel_count,
            port_type: Some(port_type),
        };
        writer.write(&AudioPortsConfiguration {
            id: port_configuration_id(configuration),
            name,
            input_port_count: 2,
            output_port_count: 1,
            main_input: Some(main),
            main_output: Some(main),
        });
    }

    fn select(&mut self, config_id: ClapId) -> Result<(), PluginError> {
        self.port_configuration = port_configuration_from_id(config_id).ok_or(
            PluginError::Message("Unknown denoize Neural audio port configuration"),
        )?;
        Ok(())
    }
}

impl PluginAudioPortsConfigInfoImpl for NeuralMainThread<'_> {
    fn current_config(&mut self) -> Option<ClapId> {
        Some(port_configuration_id(self.port_configuration))
    }

    fn get(
        &mut self,
        config_id: ClapId,
        index: u32,
        is_input: bool,
        writer: &mut AudioPortInfoWriter,
    ) {
        let previous = self.port_configuration;
        if let Some(configuration) = port_configuration_from_id(config_id) {
            self.port_configuration = configuration;
            PluginAudioPortsImpl::get(self, index, is_input, writer);
            self.port_configuration = previous;
        }
    }
}

fn port_shape(configuration: NeuralPortConfiguration) -> (u32, AudioPortType<'static>) {
    match configuration {
        NeuralPortConfiguration::Mono => (1, AudioPortType::MONO),
        NeuralPortConfiguration::Stereo => (2, AudioPortType::STEREO),
    }
}

const fn port_configuration_id(configuration: NeuralPortConfiguration) -> ClapId {
    match configuration {
        NeuralPortConfiguration::Mono => MONO_CONFIG_ID,
        NeuralPortConfiguration::Stereo => STEREO_CONFIG_ID,
    }
}

fn port_configuration_from_id(id: ClapId) -> Option<NeuralPortConfiguration> {
    if id == MONO_CONFIG_ID {
        Some(NeuralPortConfiguration::Mono)
    } else if id == STEREO_CONFIG_ID {
        Some(NeuralPortConfiguration::Stereo)
    } else {
        None
    }
}

impl PluginLatencyImpl for NeuralMainThread<'_> {
    fn get(&mut self) -> u32 {
        self.latency_frames
    }
}

impl PluginStateImpl for NeuralMainThread<'_> {
    fn save(&mut self, output: &mut OutputStream) -> Result<(), PluginError> {
        let state = NeuralSessionState::new_for_model(
            self.shared.model,
            self.port_configuration,
            self.shared.parameters.snapshot(),
        )
        .map_err(invalid_state)?;
        let bytes = state.to_canonical_bytes().map_err(invalid_state)?;
        if bytes.len() as u64 > STATE_LIMIT_BYTES {
            return Err(PluginError::Message("denoize Neural state exceeds 64 KiB"));
        }
        output.write_all(&bytes)?;
        Ok(())
    }

    fn load(&mut self, input: &mut InputStream) -> Result<(), PluginError> {
        let mut bytes = Vec::new();
        input.take(STATE_LIMIT_BYTES + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > STATE_LIMIT_BYTES {
            return Err(PluginError::Message("denoize Neural state exceeds 64 KiB"));
        }
        let state = NeuralSessionState::from_bytes(&bytes).map_err(invalid_state)?;
        state
            .validate_for_model(self.shared.model)
            .map_err(invalid_state)?;
        let changed_ports = state.port_configuration != self.port_configuration;
        self.port_configuration = state.port_configuration;
        self.shared.restore(state.parameters);
        if let Some(params) = self.host.get_extension::<HostParams>() {
            params.rescan(&mut self.host, ParamRescanFlags::VALUES);
        }
        if changed_ports
            && let Some(audio_ports) = self.host.get_extension::<HostAudioPortsConfig>()
        {
            audio_ports.rescan(&mut self.host);
        }
        Ok(())
    }
}

fn invalid_state(message: String) -> PluginError {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message).into()
}

struct SharedParameters {
    editor: Arc<EditorModel>,
    bypass: AtomicU32,
    mix: AtomicU32,
    output_gain_db: AtomicU32,
    overload_fallback: AtomicU32,
}

impl SharedParameters {
    fn new(parameters: NeuralParameters, display_name: &'static str) -> Result<Self, PluginError> {
        let editor = EditorModel::new(
            display_name,
            EDITOR_PARAMETERS,
            &[
                f64::from(bool_value(parameters.bypass)),
                f64::from(parameters.mix),
                f64::from(parameters.output_gain_db),
                f64::from(parameters.overload_fallback.index()),
            ],
        )
        .map_err(PluginError::from)?;
        Ok(Self {
            editor,
            bypass: AtomicU32::new(bool_value(parameters.bypass).to_bits()),
            mix: AtomicU32::new(parameters.mix.to_bits()),
            output_gain_db: AtomicU32::new(parameters.output_gain_db.to_bits()),
            overload_fallback: AtomicU32::new(parameters.overload_fallback.index()),
        })
    }

    fn snapshot(&self) -> NeuralParameters {
        NeuralParameters {
            bypass: f32::from_bits(self.bypass.load(Ordering::Relaxed)) >= 0.5,
            mix: f32::from_bits(self.mix.load(Ordering::Relaxed)),
            output_gain_db: f32::from_bits(self.output_gain_db.load(Ordering::Relaxed)),
            overload_fallback: OverloadFallback::from_index(
                self.overload_fallback.load(Ordering::Relaxed),
            ),
        }
    }

    fn store(&self, parameters: NeuralParameters) {
        self.bypass
            .store(bool_value(parameters.bypass).to_bits(), Ordering::Relaxed);
        self.mix.store(parameters.mix.to_bits(), Ordering::Relaxed);
        self.output_gain_db
            .store(parameters.output_gain_db.to_bits(), Ordering::Relaxed);
        self.overload_fallback
            .store(parameters.overload_fallback.index(), Ordering::Relaxed);
        self.editor
            .set_host_value(PARAM_BYPASS.get(), f64::from(bool_value(parameters.bypass)));
        self.editor
            .set_host_value(PARAM_MIX.get(), f64::from(parameters.mix));
        self.editor.set_host_value(
            PARAM_OUTPUT_GAIN.get(),
            f64::from(parameters.output_gain_db),
        );
        self.editor.set_host_value(
            PARAM_FALLBACK.get(),
            f64::from(parameters.overload_fallback.index()),
        );
    }

    fn value(&self, id: ClapId) -> Option<f64> {
        if id == PARAM_BYPASS {
            Some(f64::from(f32::from_bits(
                self.bypass.load(Ordering::Relaxed),
            )))
        } else if id == PARAM_MIX {
            Some(f64::from(f32::from_bits(self.mix.load(Ordering::Relaxed))))
        } else if id == PARAM_OUTPUT_GAIN {
            Some(f64::from(f32::from_bits(
                self.output_gain_db.load(Ordering::Relaxed),
            )))
        } else if id == PARAM_FALLBACK {
            Some(f64::from(self.overload_fallback.load(Ordering::Relaxed)))
        } else {
            None
        }
    }

    fn set_value(&self, id: ClapId, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        if id == PARAM_BYPASS {
            self.bypass
                .store(bool_value(value >= 0.5).to_bits(), Ordering::Relaxed);
        } else if id == PARAM_MIX {
            self.mix
                .store((value as f32).clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        } else if id == PARAM_OUTPUT_GAIN {
            self.output_gain_db.store(
                (value as f32).clamp(-24.0, 24.0).to_bits(),
                Ordering::Relaxed,
            );
        } else if id == PARAM_FALLBACK {
            self.overload_fallback.store(
                OverloadFallback::from_index(value.round().clamp(0.0, 2.0) as u32).index(),
                Ordering::Relaxed,
            );
        } else {
            return false;
        }
        self.editor.set_host_value(id.get(), value);
        true
    }

    fn handle_event(&self, event: &UnknownEvent) {
        if let Some(CoreEventSpace::ParamValue(value)) = event.as_core_event()
            && let Some(param_id) = value.param_id()
        {
            self.set_value(param_id, value.value());
        }
    }
}

const fn bool_value(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}

impl PluginMainThreadParams for NeuralMainThread<'_> {
    fn count(&mut self) -> u32 {
        PARAMETER_COUNT
    }

    fn get_info(&mut self, param_index: u32, writer: &mut ParamInfoWriter) {
        if let Some(info) = parameter_info(param_index) {
            writer.set(&info);
        }
    }

    fn get_value(&mut self, param_id: ClapId) -> Option<f64> {
        self.shared.parameters.value(param_id)
    }

    fn value_to_text(
        &mut self,
        param_id: ClapId,
        value: f64,
        writer: &mut ParamDisplayWriter,
    ) -> std::fmt::Result {
        if param_id == PARAM_BYPASS {
            writer.write_str(if value >= 0.5 { "On" } else { "Off" })
        } else if param_id == PARAM_MIX {
            write!(writer, "{:.1} %", value * 100.0)
        } else if param_id == PARAM_OUTPUT_GAIN {
            write!(writer, "{value:.1} dB")
        } else if param_id == PARAM_FALLBACK {
            writer.write_str(OverloadFallback::from_index(value.round() as u32).label())
        } else {
            Err(std::fmt::Error)
        }
    }

    fn text_to_value(&mut self, param_id: ClapId, text: &CStr) -> Option<f64> {
        let text = text.to_str().ok()?.trim();
        if param_id == PARAM_BYPASS {
            return match text.to_ascii_lowercase().as_str() {
                "on" | "true" | "yes" | "1" => Some(1.0),
                "off" | "false" | "no" | "0" => Some(0.0),
                _ => None,
            };
        }
        if param_id == PARAM_FALLBACK {
            return match text.to_ascii_lowercase().replace([' ', '_'], "-").as_str() {
                "delayed-dry" | "dry" | "0" => Some(0.0),
                "last-safe-gain" | "gain" | "1" => Some(1.0),
                "silence" | "2" => Some(2.0),
                _ => None,
            };
        }
        let number = text
            .strip_suffix('%')
            .or_else(|| text.strip_suffix("dB"))
            .or_else(|| text.strip_suffix("db"))
            .unwrap_or(text)
            .trim()
            .parse::<f64>()
            .ok()?;
        if param_id == PARAM_MIX {
            Some(number / 100.0)
        } else if param_id == PARAM_OUTPUT_GAIN {
            Some(number)
        } else {
            None
        }
    }

    fn flush(&mut self, input: &InputEvents, output: &mut OutputEvents) {
        for event in input {
            self.shared.parameters.handle_event(event);
        }
        let retry = super::drain_editor_automation(
            &self.shared.parameters.editor,
            output,
            &mut self.pending_automation,
            |parameter_id, value| {
                self.shared.parameters.set_value(parameter_id, value);
            },
        );
        if retry && let Some(params) = self.host.get_extension::<HostParams>() {
            params.request_flush(&self.host.shared());
        }
    }
}

fn parameter_info(index: u32) -> Option<ParamInfo<'static>> {
    let defaults = NeuralParameters::default();
    let automatable = ParamInfoFlags::IS_AUTOMATABLE;
    let stepped = automatable | ParamInfoFlags::IS_STEPPED;
    let (id, flags, name, module, minimum, maximum, default) = match index {
        0 => (
            PARAM_BYPASS,
            // This is denoize's persistent, latency-aligned DSP control, not
            // the host's process-level bypass button. Advertising IS_BYPASS
            // lets some hosts merge their host-managed bypass state back into
            // this parameter and overwrite repeated user changes.
            stepped,
            b"Bypass".as_slice(),
            b"Neural".as_slice(),
            0.0,
            1.0,
            f64::from(bool_value(defaults.bypass)),
        ),
        1 => (
            PARAM_MIX,
            automatable,
            b"Mix".as_slice(),
            b"Neural".as_slice(),
            0.0,
            1.0,
            f64::from(defaults.mix),
        ),
        2 => (
            PARAM_OUTPUT_GAIN,
            automatable,
            b"Output Gain".as_slice(),
            b"Neural".as_slice(),
            -24.0,
            24.0,
            f64::from(defaults.output_gain_db),
        ),
        3 => (
            PARAM_FALLBACK,
            stepped,
            b"Overload Fallback".as_slice(),
            b"Safety".as_slice(),
            0.0,
            2.0,
            f64::from(defaults.overload_fallback.index()),
        ),
        _ => return None,
    };
    Some(ParamInfo {
        id,
        flags,
        cookie: Default::default(),
        name,
        module,
        min_value: minimum,
        max_value: maximum,
        default_value: default,
    })
}

#[derive(Clone, Copy)]
struct RuntimeParameters {
    bypass: bool,
    mix: f64,
    output_gain: f64,
    fallback: OverloadFallback,
}

impl From<NeuralParameters> for RuntimeParameters {
    fn from(parameters: NeuralParameters) -> Self {
        Self {
            bypass: parameters.bypass,
            mix: f64::from(parameters.mix),
            output_gain: 10.0_f64.powf(f64::from(parameters.output_gain_db) / 20.0),
            fallback: parameters.overload_fallback,
        }
    }
}

pub(crate) struct NeuralAudioProcessor<'a> {
    shared: &'a NeuralShared,
    engine: NeuralEngine<'a>,
    observed_reset_generation: u64,
}

impl<'a> PluginAudioProcessor<'a, NeuralShared, NeuralMainThread<'a>> for NeuralAudioProcessor<'a> {
    fn activate(
        _host: HostAudioProcessorHandle<'a>,
        main_thread: &mut NeuralMainThread<'a>,
        shared: &'a NeuralShared,
        audio_config: PluginAudioConfiguration,
    ) -> Result<Self, PluginError> {
        let sample_rate = validated_sample_rate(audio_config.sample_rate).map_err(invalid_state)?;
        let channels = main_thread.port_configuration.channels();
        let engine = NeuralEngine::new_model_for_host(
            shared.model,
            sample_rate,
            channels,
            shared,
            audio_config,
        )
        .map_err(invalid_state)?;
        main_thread.latency_frames = engine.latency_frames;
        Ok(Self {
            shared,
            engine,
            observed_reset_generation: shared.reset_generation.load(Ordering::Acquire),
        })
    }

    fn process(
        &mut self,
        _process: Process,
        mut audio: Audio,
        events: Events,
    ) -> Result<ProcessStatus, PluginError> {
        self.apply_pending_reset();
        let mut port = audio.port_pair(0).ok_or(PluginError::Message(
            "denoize Neural requires one main audio port pair",
        ))?;
        if port.channel_pair_count() != self.engine.channels {
            return Err(PluginError::Message(
                "denoize Neural host channel count does not match the selected configuration",
            ));
        }
        match port.channels()? {
            SampleType::F32(channels) => self.process_channels(channels, events.input)?,
            SampleType::F64(channels) => self.process_channels(channels, events.input)?,
            SampleType::Both(channels, _) => self.process_channels(channels, events.input)?,
        }
        Ok(ProcessStatus::ContinueIfNotQuiet)
    }

    fn deactivate(mut self, _main_thread: &mut NeuralMainThread<'a>) {
        self.engine.stop();
    }

    fn reset(&mut self) {
        self.engine.reset();
        self.observed_reset_generation = self.shared.reset_generation.load(Ordering::Acquire);
    }
}

impl NeuralAudioProcessor<'_> {
    fn apply_pending_reset(&mut self) {
        let generation = self.shared.reset_generation.load(Ordering::Acquire);
        if generation != self.observed_reset_generation {
            self.engine.reset();
            self.observed_reset_generation = generation;
        }
    }

    fn process_channels<S: AudioSample>(
        &mut self,
        mut channels: PairedChannels<'_, S>,
        events: &InputEvents,
    ) -> Result<(), PluginError> {
        if channels.input_channel_count() != self.engine.channels
            || channels.output_channel_count() != self.engine.channels
        {
            return Err(PluginError::Message(
                "denoize Neural requires matching main input and output channel counts",
            ));
        }
        let frames = channels.frames_count() as usize;
        self.engine.record_callback(frames);
        let mut left = channels.channel_pair(0).ok_or(PluginError::Message(
            "denoize Neural left channel is missing",
        ))?;
        let mut right = if self.engine.channels == 2 {
            Some(channels.channel_pair(1).ok_or(PluginError::Message(
                "denoize Neural right channel is missing",
            ))?)
        } else {
            None
        };

        for batch in events.batch() {
            for event in batch.events() {
                self.shared.parameters.handle_event(event);
            }
            let parameters = RuntimeParameters::from(self.shared.parameters.snapshot());
            let start = batch.first_sample().min(frames);
            let end = batch
                .next_batch_first_sample()
                .unwrap_or(frames)
                .min(frames);
            for frame in start..end {
                let input = [
                    read_channel(&left, frame).to_f64(),
                    right
                        .as_ref()
                        .map_or(0.0, |pair| read_channel(pair, frame).to_f64()),
                ];
                let output = self.engine.process_frame(input, parameters);
                write_channel(&mut left, frame, S::from_f64(output[0]));
                if let Some(pair) = right.as_mut() {
                    write_channel(pair, frame, S::from_f64(output[1]));
                }
            }
        }
        Ok(())
    }
}

impl PluginAudioProcessorParams for NeuralAudioProcessor<'_> {
    fn flush(&mut self, input: &InputEvents, _output: &mut OutputEvents) {
        for event in input {
            self.shared.parameters.handle_event(event);
        }
    }
}

trait AudioSample: Copy {
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
}

impl AudioSample for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl AudioSample for f64 {
    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }
}

fn read_channel<S: AudioSample>(channel: &ChannelPair<'_, S>, frame: usize) -> S {
    match channel {
        ChannelPair::InputOnly(input) | ChannelPair::InputOutput(input, _) => input[frame],
        ChannelPair::InPlace(buffer) => buffer[frame],
        ChannelPair::OutputOnly(_) => S::from_f64(0.0),
    }
}

fn write_channel<S: AudioSample>(channel: &mut ChannelPair<'_, S>, frame: usize, value: S) {
    match channel {
        ChannelPair::OutputOnly(output)
        | ChannelPair::InputOutput(_, output)
        | ChannelPair::InPlace(output) => output[frame] = value,
        ChannelPair::InputOnly(_) => {}
    }
}

struct AudioBlock {
    generation: u64,
    start_frame: u64,
    frames: usize,
    samples: Box<[f32]>,
}

struct ProcessedBlock {
    block: AudioBlock,
    valid: bool,
    invalid_output: bool,
}

trait BlockProcessor: Send {
    fn process(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String>;
    fn reset(&mut self) -> Result<(), String>;
}

fn warm_up_block_processor(
    processor: &mut dyn BlockProcessor,
    channels: usize,
    frames: usize,
) -> Result<(), String> {
    let input = vec![vec![0.0; frames]; channels];
    let output = processor
        .process(&input)
        .map_err(|error| format!("warm up neural inference worker: {error}"))?;
    // Streaming backends may retain their look-ahead tail until `finish`, so a
    // successful warm-up can legitimately return fewer frames than it accepts.
    let output_frames = output.first().map_or(0, Vec::len);
    if output.len() != channels
        || output_frames == 0
        || output_frames > frames
        || output.iter().any(|channel| {
            channel.len() != output_frames || channel.iter().any(|sample| !sample.is_finite())
        })
    {
        return Err("neural inference worker returned invalid warm-up output".to_owned());
    }
    processor
        .reset()
        .map_err(|error| format!("reset neural inference worker after warm-up: {error}"))
}

struct GtcrnProcessor(StreamingBackendSession);

static GTCRN_MODEL_CACHE: OnceLock<Mutex<Option<GtcrnModel>>> = OnceLock::new();

impl GtcrnProcessor {
    fn new(sample_rate: u32, channels: usize) -> Result<Self, String> {
        let model = gtcrn_model()?;
        let path = denoize::models::verify(model).map_err(|error| {
            format!(
                "GTCRN model is unavailable ({error}); run `denoize models install gtcrn` before activating denoize Neural"
            )
        })?;
        let mut options = BackendOptions {
            onnx: Some(OnnxModelConfig {
                path,
                sample_rate: model.sample_rate,
            }),
            deterministic: true,
            ..BackendOptions::default()
        };
        if channels == 2 {
            options.channel_mode = ChannelMode::StereoLinked;
        }
        let accelerator = select_accelerator_for_options(Backend::Gtcrn, &options)?;
        let Some(model_config) = options.onnx.as_ref() else {
            return Err("internal GTCRN model options are unavailable".to_owned());
        };
        let prepared = prepared_gtcrn_model(model_config, accelerator.effective())?;
        let mut denoiser = DenoiserConfig::default(sample_rate);
        denoiser.vad = false;
        Ok(Self(
            StreamingBackendSession::new_gtcrn_for_daw_with_prepared_model(
                sample_rate,
                channels,
                denoiser,
                options,
                &prepared,
            )?,
        ))
    }
}

fn prepared_gtcrn_model(
    config: &OnnxModelConfig,
    runtime: AcceleratorRuntime,
) -> Result<GtcrnModel, String> {
    let cache = GTCRN_MODEL_CACHE.get_or_init(|| Mutex::new(None));
    let mut cached = cache
        .lock()
        .map_err(|_| "denoize Neural compiled-model cache lock was poisoned".to_owned())?;
    if let Some(cached) = cached.as_ref()
        && cached.runtime() == runtime
    {
        return Ok(cached.clone());
    }
    let model = GtcrnModel::load_with_accelerator(config, runtime)?;
    *cached = Some(model.clone());
    Ok(model)
}

impl BlockProcessor for GtcrnProcessor {
    fn process(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        self.0.process_block(channels)
    }

    fn reset(&mut self) -> Result<(), String> {
        self.0.reset()
    }
}

fn gtcrn_model() -> Result<&'static denoize::models::ModelInfo, String> {
    denoize::models::MODELS
        .iter()
        .find(|model| {
            model.name == MODEL_ID
                && model.backend == "gtcrn"
                && model.sha256 == NEURAL_DAW_MODEL_SHA256
        })
        .ok_or_else(|| "this build does not contain the pinned GTCRN model identity".to_owned())
}

#[cfg(feature = "experimental-dpdfnet-hq")]
struct DpdfnetProcessor(StreamingBackendSession);

#[cfg(feature = "experimental-dpdfnet-hq")]
static DPDFNET_MODEL_CACHE: OnceLock<Mutex<Option<DpdfnetModel>>> = OnceLock::new();

#[cfg(feature = "experimental-dpdfnet-hq")]
impl DpdfnetProcessor {
    fn new(sample_rate: u32, channels: usize) -> Result<Self, String> {
        let model = dpdfnet_model()?;
        let path = denoize::models::verify(model).map_err(|error| {
            format!(
                "DPDFNet model is unavailable ({error}); run `denoize models install dpdfnet` before activating denoize Neural HQ"
            )
        })?;
        let mut options = BackendOptions {
            onnx: Some(OnnxModelConfig {
                path,
                sample_rate: model.sample_rate,
            }),
            deterministic: true,
            ..BackendOptions::default()
        };
        if channels == 2 {
            options.channel_mode = ChannelMode::StereoLinked;
        }
        let accelerator = select_accelerator_for_options(Backend::Dpdfnet, &options)?;
        let Some(model_config) = options.onnx.as_ref() else {
            return Err("internal DPDFNet model options are unavailable".to_owned());
        };
        let prepared = prepared_dpdfnet_model(model_config, accelerator.effective())?;
        let mut denoiser = DenoiserConfig::default(sample_rate);
        denoiser.vad = false;
        Ok(Self(
            StreamingBackendSession::new_dpdfnet_for_daw_with_prepared_model(
                sample_rate,
                channels,
                denoiser,
                options,
                &prepared,
            )?,
        ))
    }
}

#[cfg(feature = "experimental-dpdfnet-hq")]
fn prepared_dpdfnet_model(
    config: &OnnxModelConfig,
    runtime: AcceleratorRuntime,
) -> Result<DpdfnetModel, String> {
    let cache = DPDFNET_MODEL_CACHE.get_or_init(|| Mutex::new(None));
    let mut cached = cache
        .lock()
        .map_err(|_| "denoize Neural HQ compiled-model cache lock was poisoned".to_owned())?;
    if let Some(cached) = cached.as_ref()
        && cached.runtime() == runtime
    {
        return Ok(cached.clone());
    }
    let model = DpdfnetModel::load_dpdfnet2_with_accelerator(config, runtime)?;
    *cached = Some(model.clone());
    Ok(model)
}

#[cfg(feature = "experimental-dpdfnet-hq")]
impl BlockProcessor for DpdfnetProcessor {
    fn process(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        self.0.process_block(channels)
    }

    fn reset(&mut self) -> Result<(), String> {
        self.0.reset()
    }
}

#[cfg(feature = "experimental-dpdfnet-hq")]
fn dpdfnet_model() -> Result<&'static denoize::models::ModelInfo, String> {
    denoize::models::MODELS
        .iter()
        .find(|model| {
            model.name == NEURAL_HQ_DAW_MODEL_ID
                && model.backend == "dpdfnet"
                && model.sha256 == NEURAL_HQ_DAW_MODEL_SHA256
        })
        .ok_or_else(|| "this build does not contain the pinned DPDFNet model identity".to_owned())
}

struct NeuralEngine<'a> {
    channels: usize,
    chunk_frames: usize,
    latency_frames: u32,
    input_queue: Arc<ArrayQueue<AudioBlock>>,
    output_queue: Arc<ArrayQueue<ProcessedBlock>>,
    free_blocks: Vec<AudioBlock>,
    capture: Option<AudioBlock>,
    capture_frames: usize,
    ready: VecDeque<ProcessedBlock>,
    playback: Option<ProcessedBlock>,
    dry_delay: Vec<f64>,
    dry_cursor: usize,
    input_frame: u64,
    processed_frames: u64,
    activated_at: std::time::Instant,
    host_evidence_warmup_frames: u64,
    host_evidence_baseline: Option<WorkerMetrics>,
    host_min_frames_count: u32,
    host_max_frames_count: u32,
    callback_calls: u64,
    callback_min_frames: u32,
    callback_max_frames: u32,
    generation: u64,
    last_safe_gain: [f64; 2],
    running: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    worker_started: bool,
    finished_gracefully: bool,
    metrics: &'a NeuralShared,
}

impl<'a> NeuralEngine<'a> {
    fn new_model_for_host(
        model: NeuralDawModel,
        sample_rate: f64,
        channels: usize,
        metrics: &'a NeuralShared,
        audio_config: PluginAudioConfiguration,
    ) -> Result<Self, String> {
        let mut engine = Self::new_model(model, sample_rate, channels, metrics)?;
        engine.host_min_frames_count = audio_config.min_frames_count;
        engine.host_max_frames_count = audio_config.max_frames_count;
        Ok(engine)
    }

    fn new_model(
        model: NeuralDawModel,
        sample_rate: f64,
        channels: usize,
        metrics: &'a NeuralShared,
    ) -> Result<Self, String> {
        match model {
            NeuralDawModel::Gtcrn => Self::new_gtcrn(sample_rate, channels, metrics),
            NeuralDawModel::Dpdfnet2 => Self::new_dpdfnet(sample_rate, channels, metrics),
        }
    }

    fn new_gtcrn(
        sample_rate: f64,
        channels: usize,
        metrics: &'a NeuralShared,
    ) -> Result<Self, String> {
        let backend_sample_rate = sample_rate.round() as u32;
        Self::new_with_factory(sample_rate, channels, metrics, move || {
            GtcrnProcessor::new(backend_sample_rate, channels)
                .map(|processor| Box::new(processor) as Box<dyn BlockProcessor>)
        })
    }

    #[cfg(feature = "experimental-dpdfnet-hq")]
    fn new_dpdfnet(
        sample_rate: f64,
        channels: usize,
        metrics: &'a NeuralShared,
    ) -> Result<Self, String> {
        let backend_sample_rate = sample_rate.round() as u32;
        Self::new_with_factory(sample_rate, channels, metrics, move || {
            DpdfnetProcessor::new(backend_sample_rate, channels)
                .map(|processor| Box::new(processor) as Box<dyn BlockProcessor>)
        })
    }

    #[cfg(not(feature = "experimental-dpdfnet-hq"))]
    fn new_dpdfnet(
        _sample_rate: f64,
        _channels: usize,
        _metrics: &'a NeuralShared,
    ) -> Result<Self, String> {
        Err("this CLAP build does not enable the experimental DPDFNet HQ descriptor".to_owned())
    }

    fn new_with_factory<F>(
        sample_rate: f64,
        channels: usize,
        metrics: &'a NeuralShared,
        factory: F,
    ) -> Result<Self, String>
    where
        F: FnOnce() -> Result<Box<dyn BlockProcessor>, String> + Send + 'static,
    {
        if !(1..=2).contains(&channels) {
            return Err("neural plug-in supports one or two channels".into());
        }
        let chunk_frames = usize::try_from(neural_daw_chunk_frames(sample_rate)?)
            .map_err(|_| "neural plug-in chunk geometry does not fit memory".to_owned())?;
        let latency_frames_usize = chunk_frames
            .checked_mul(LATENCY_CHUNKS as usize)
            .ok_or_else(|| "neural plug-in latency geometry overflow".to_owned())?;
        let latency_frames = u32::try_from(latency_frames_usize)
            .map_err(|_| "neural plug-in latency exceeds the CLAP contract".to_owned())?;
        let warmup_frames = chunk_frames
            .checked_mul(WORKER_WARMUP_EXTRA_BLOCKS)
            .and_then(|extra| latency_frames_usize.checked_add(extra))
            .ok_or_else(|| "neural worker warm-up geometry overflow".to_owned())?;
        let samples_per_block = chunk_frames
            .checked_mul(channels)
            .ok_or_else(|| "neural plug-in block geometry overflow".to_owned())?;
        let input_queue = Arc::new(ArrayQueue::new(QUEUE_BLOCKS));
        let output_queue = Arc::new(ArrayQueue::new(QUEUE_BLOCKS));
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(BLOCK_POOL_SIZE)
            .map_err(|_| "unable to reserve neural plug-in block pool".to_owned())?;
        for _ in 0..BLOCK_POOL_SIZE {
            let mut samples = Vec::new();
            samples
                .try_reserve_exact(samples_per_block)
                .map_err(|_| "unable to reserve neural plug-in audio block".to_owned())?;
            samples.resize(samples_per_block, 0.0);
            blocks.push(AudioBlock {
                generation: 1,
                start_frame: 0,
                frames: 0,
                samples: samples.into_boxed_slice(),
            });
        }
        let capture = blocks
            .pop()
            .ok_or_else(|| "neural plug-in block pool is empty".to_owned())?;
        let mut dry_delay = Vec::new();
        dry_delay
            .try_reserve_exact(
                latency_frames_usize
                    .checked_mul(channels)
                    .ok_or_else(|| "neural plug-in dry-delay geometry overflow".to_owned())?,
            )
            .map_err(|_| "unable to reserve neural plug-in dry delay".to_owned())?;
        dry_delay.resize(latency_frames_usize * channels, 0.0);
        let mut ready = VecDeque::new();
        ready
            .try_reserve_exact(BLOCK_POOL_SIZE)
            .map_err(|_| "unable to reserve neural plug-in result queue".to_owned())?;

        let running = Arc::new(AtomicBool::new(true));
        let worker_running = Arc::clone(&running);
        let worker_input = Arc::clone(&input_queue);
        let worker_output = Arc::clone(&output_queue);
        let worker_errors = Arc::clone(&metrics.worker_errors);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("denoize-neural".to_owned())
            .spawn(move || {
                let (mut processor, mut priority_guard) =
                    match factory().and_then(|mut processor| {
                        warm_up_block_processor(&mut *processor, channels, warmup_frames)?;
                        let priority_guard =
                            denoize::neural_daw::NeuralDawWorkerPriorityGuard::acquire()?;
                        Ok((processor, priority_guard))
                    }) {
                        Ok(initialized) => {
                            let _ = ready_tx.send(Ok(()));
                            initialized
                        }
                        Err(error) => {
                            let _ = ready_tx.send(Err(error));
                            return;
                        }
                    };
                worker_loop(
                    &mut *processor,
                    &mut priority_guard,
                    channels,
                    worker_input,
                    worker_output,
                    worker_running,
                    worker_errors,
                );
                drop(priority_guard);
            })
            .map_err(|error| format!("start neural inference worker: {error}"))?;
        let mut finished_gracefully = true;
        let worker = match ready_rx.recv() {
            Ok(Ok(())) => Some(worker),
            Ok(Err(error)) => {
                // Some hosts only deliver parameter changes while the audio
                // processor is active. Keep the fixed-latency fallback path
                // alive when the authenticated model cannot be prepared so
                // their generic and accessibility parameter surfaces remain
                // usable; no neural inference runs in this state.
                eprintln!("denoize Neural worker startup error: {error}");
                metrics.worker_errors.fetch_add(1, Ordering::Relaxed);
                running.store(false, Ordering::Release);
                if worker.join().is_err() {
                    finished_gracefully = false;
                    metrics.worker_errors.fetch_add(1, Ordering::Relaxed);
                }
                None
            }
            Err(error) => {
                eprintln!("denoize Neural worker startup handshake error: {error}");
                metrics.worker_errors.fetch_add(1, Ordering::Relaxed);
                running.store(false, Ordering::Release);
                if worker.join().is_err() {
                    finished_gracefully = false;
                    metrics.worker_errors.fetch_add(1, Ordering::Relaxed);
                }
                None
            }
        };
        let worker_started = worker.is_some();
        let host_evidence_warmup_frames =
            if std::env::var_os("DENOIZE_NEURAL_HOST_EVIDENCE").is_some() {
                u64::from(latency_frames).saturating_mul(HOST_EVIDENCE_WARMUP_LATENCIES)
            } else {
                0
            };
        let host_evidence_baseline =
            (host_evidence_warmup_frames == 0).then(WorkerMetrics::default);

        Ok(Self {
            channels,
            chunk_frames,
            latency_frames,
            input_queue,
            output_queue,
            free_blocks: blocks,
            capture: Some(capture),
            capture_frames: 0,
            ready,
            playback: None,
            dry_delay,
            dry_cursor: 0,
            input_frame: 0,
            processed_frames: 0,
            activated_at: std::time::Instant::now(),
            host_evidence_warmup_frames,
            host_evidence_baseline,
            host_min_frames_count: 1,
            host_max_frames_count: chunk_frames as u32,
            callback_calls: 0,
            callback_min_frames: u32::MAX,
            callback_max_frames: 0,
            generation: 1,
            last_safe_gain: [1.0; 2],
            running,
            worker,
            worker_started,
            finished_gracefully,
            metrics,
        })
    }

    fn record_callback(&mut self, frames: usize) {
        let Ok(frames) = u32::try_from(frames) else {
            return;
        };
        if frames == 0 {
            return;
        }
        self.callback_calls = self.callback_calls.saturating_add(1);
        self.callback_min_frames = self.callback_min_frames.min(frames);
        self.callback_max_frames = self.callback_max_frames.max(frames);
    }

    #[inline]
    fn process_frame(&mut self, mut input: [f64; 2], parameters: RuntimeParameters) -> [f64; 2] {
        for sample in input.iter_mut().take(self.channels) {
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
        if self.input_frame.is_multiple_of(self.chunk_frames as u64) {
            self.begin_output_chunk();
        }

        let offset = self.capture_frames;
        if let Some(capture) = self.capture.as_mut() {
            for (channel, sample) in input.iter().take(self.channels).enumerate() {
                capture.samples[offset * self.channels + channel] = *sample as f32;
            }
        }

        let latency_frames = self.latency_frames as usize;
        let mut delayed = [0.0; 2];
        for channel in 0..self.channels {
            let delay_index = self.dry_cursor * self.channels + channel;
            delayed[channel] = self.dry_delay[delay_index];
            self.dry_delay[delay_index] = input[channel];
        }
        self.dry_cursor += 1;
        if self.dry_cursor == latency_frames {
            self.dry_cursor = 0;
        }

        let mut output = [0.0; 2];
        let valid_playback = self.playback.as_ref().is_some_and(|result| result.valid);
        for channel in 0..self.channels {
            let wet = if valid_playback {
                let value = self.playback.as_ref().map_or(0.0, |result| {
                    f64::from(result.block.samples[offset * self.channels + channel])
                });
                if delayed[channel].abs() > 1.0e-7 {
                    let ratio = (value.abs() / delayed[channel].abs()).clamp(0.0, 2.0);
                    self.last_safe_gain[channel] =
                        0.995 * self.last_safe_gain[channel] + 0.005 * ratio;
                }
                value
            } else {
                match parameters.fallback {
                    OverloadFallback::DelayedDry => delayed[channel],
                    OverloadFallback::LastSafeGain => {
                        delayed[channel] * self.last_safe_gain[channel]
                    }
                    OverloadFallback::Silence => 0.0,
                }
            };
            let mixed = if parameters.bypass {
                delayed[channel]
            } else {
                delayed[channel] * (1.0 - parameters.mix) + wet * parameters.mix
            };
            let gained = mixed * parameters.output_gain;
            output[channel] = if gained.is_finite() { gained } else { 0.0 };
        }

        self.capture_frames += 1;
        self.input_frame = self.input_frame.wrapping_add(1);
        self.processed_frames = self.processed_frames.saturating_add(1);
        if self.host_evidence_baseline.is_none()
            && self.processed_frames >= self.host_evidence_warmup_frames
        {
            self.host_evidence_baseline = Some(self.metrics.worker_metrics());
        }
        if self.capture_frames == self.chunk_frames {
            self.submit_capture();
        }
        output
    }

    #[inline]
    fn begin_output_chunk(&mut self) {
        if self.worker.is_none() {
            return;
        }
        if let Some(result) = self.playback.take() {
            self.recycle(result.block);
        }
        while let Some(result) = self.output_queue.pop() {
            if result.block.generation != self.generation {
                self.recycle(result.block);
            } else if self.ready.len() < BLOCK_POOL_SIZE {
                self.ready.push_back(result);
            } else {
                self.metrics.overload_blocks.fetch_add(1, Ordering::Relaxed);
                self.recycle(result.block);
            }
        }
        if self.input_frame < u64::from(self.latency_frames) {
            return;
        }
        let due = self.input_frame - u64::from(self.latency_frames);
        while self
            .ready
            .front()
            .is_some_and(|result| result.block.start_frame < due)
        {
            if let Some(late) = self.ready.pop_front() {
                self.metrics.late_blocks.fetch_add(1, Ordering::Relaxed);
                self.recycle(late.block);
            }
        }
        if self
            .ready
            .front()
            .is_some_and(|result| result.block.start_frame == due)
        {
            self.playback = self.ready.pop_front();
            if self
                .playback
                .as_ref()
                .is_some_and(|result| result.invalid_output)
            {
                self.metrics.invalid_blocks.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            self.metrics.overload_blocks.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    fn submit_capture(&mut self) {
        let Some(mut completed) = self.capture.take() else {
            return;
        };
        if self.worker.is_none() {
            completed.frames = 0;
            self.capture = Some(completed);
            self.capture_frames = 0;
            return;
        }
        completed.generation = self.generation;
        completed.start_frame = self.input_frame.saturating_sub(self.chunk_frames as u64);
        completed.frames = self.chunk_frames;
        let Some(replacement) = self.free_blocks.pop() else {
            self.metrics.overload_blocks.fetch_add(1, Ordering::Relaxed);
            completed.frames = 0;
            self.capture = Some(completed);
            self.capture_frames = 0;
            return;
        };
        match self.input_queue.push(completed) {
            Ok(()) => self.capture = Some(replacement),
            Err(mut returned) => {
                self.metrics.overload_blocks.fetch_add(1, Ordering::Relaxed);
                returned.frames = 0;
                self.capture = Some(returned);
                self.free_blocks.push(replacement);
            }
        }
        self.capture_frames = 0;
    }

    #[inline]
    fn recycle(&mut self, mut block: AudioBlock) {
        block.frames = 0;
        self.free_blocks.push(block);
    }

    fn reset(&mut self) {
        self.generation = self.generation.wrapping_add(1).max(1);
        self.capture_frames = 0;
        self.input_frame = 0;
        self.dry_delay.fill(0.0);
        self.dry_cursor = 0;
        self.last_safe_gain = [1.0; 2];
        if let Some(result) = self.playback.take() {
            self.recycle(result.block);
        }
        while let Some(result) = self.ready.pop_front() {
            self.recycle(result.block);
        }
        // A transport reset starts a new stream generation. Reclaim queued
        // input from the previous generation immediately so the worker does
        // not spend the new stream's latency budget processing stale audio.
        // The queue is fixed-size and CLAP only calls reset while processing
        // is stopped, so this remains bounded and cannot race a new callback.
        while let Some(block) = self.input_queue.pop() {
            self.recycle(block);
        }
        while let Some(result) = self.output_queue.pop() {
            self.recycle(result.block);
        }
    }

    fn stop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            self.finished_gracefully = false;
            self.metrics.worker_errors.fetch_add(1, Ordering::Relaxed);
            eprintln!("denoize Neural worker panicked during shutdown");
        }
    }
}

impl Drop for NeuralEngine<'_> {
    fn drop(&mut self) {
        self.stop();
        // REAPER may activate and immediately deactivate the processor while
        // changing audio devices. That zero-frame probe must not claim the
        // single evidence path before the activation that actually ran audio.
        if self.should_write_host_evidence() {
            self.write_host_evidence();
        }
    }
}

impl NeuralEngine<'_> {
    fn should_write_host_evidence(&self) -> bool {
        self.processed_frames > self.host_evidence_warmup_frames
    }

    fn write_host_evidence(&self) {
        let Ok(path) = std::env::var("DENOIZE_NEURAL_HOST_EVIDENCE") else {
            return;
        };
        let lifetime_metrics = self.metrics.worker_metrics();
        let baseline = self.host_evidence_baseline.unwrap_or(lifetime_metrics);
        let measured_metrics = lifetime_metrics.saturating_since(baseline);
        let measured_frames = self
            .processed_frames
            .saturating_sub(self.host_evidence_warmup_frames);
        let document = serde_json::json!({
            "schema": "denoize-dpdfnet-clap-host-run-v2",
            "schema_version": 2,
            "source_commit": std::env::var("DENOIZE_EVIDENCE_SOURCE_COMMIT").unwrap_or_default(),
            "model_id": self.metrics.model.model_id(),
            "model_sha256": self.metrics.model.model_sha256(),
            "plugin_id": self.metrics.model.plugin_id(),
            "sample_rate_hz": self.chunk_frames * 100,
            "channels": self.channels,
            "chunk_frames": self.chunk_frames,
            "latency_frames": self.latency_frames,
            "processed_frames": self.processed_frames,
            "active_seconds": self.activated_at.elapsed().as_secs_f64(),
            "measurement": {
                "warmup_frames": self.host_evidence_warmup_frames,
                "measured_frames": measured_frames,
            },
            "host_audio_configuration": {
                "min_frames_count": self.host_min_frames_count,
                "max_frames_count": self.host_max_frames_count,
            },
            "callback_frames": {
                "calls": self.callback_calls,
                "minimum": self.callback_min_frames,
                "maximum": self.callback_max_frames,
            },
            // `deactivate()` stops and joins the worker before `Drop`. Keep
            // the successful startup fact independently of the live handle.
            "worker_started": self.worker_started,
            "finished_gracefully": self.finished_gracefully,
            "metrics": {
                "overload_blocks": measured_metrics.overload_blocks,
                "late_blocks": measured_metrics.late_blocks,
                "invalid_blocks": measured_metrics.invalid_blocks,
                "worker_errors": measured_metrics.worker_errors,
            },
            "lifetime_metrics": {
                "overload_blocks": lifetime_metrics.overload_blocks,
                "late_blocks": lifetime_metrics.late_blocks,
                "invalid_blocks": lifetime_metrics.invalid_blocks,
                "worker_errors": lifetime_metrics.worker_errors,
            },
            "environment": {
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
            },
        });
        let result = (|| -> Result<(), String> {
            let bytes = serde_json::to_vec_pretty(&document)
                .map_err(|error| format!("encode host evidence: {error}"))?;
            let mut destination = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|error| format!("create host evidence {path}: {error}"))?;
            destination
                .write_all(&bytes)
                .and_then(|()| destination.write_all(b"\n"))
                .map_err(|error| format!("write host evidence {path}: {error}"))
        })();
        if let Err(error) = result {
            eprintln!("denoize Neural host evidence error: {error}");
        }
    }
}

fn worker_loop(
    processor: &mut dyn BlockProcessor,
    priority_guard: &mut denoize::neural_daw::NeuralDawWorkerPriorityGuard,
    channels: usize,
    input: Arc<ArrayQueue<AudioBlock>>,
    output: Arc<ArrayQueue<ProcessedBlock>>,
    running: Arc<AtomicBool>,
    worker_errors: Arc<AtomicU64>,
) {
    let mut generation = 0u64;
    let mut next_start = 0u64;
    let mut pending = VecDeque::<AudioBlock>::new();
    let mut completed = VecDeque::<ProcessedBlock>::new();
    let mut ready = (0..channels)
        .map(|_| VecDeque::<f64>::new())
        .collect::<Vec<_>>();
    let mut failed = false;

    while running.load(Ordering::Acquire) {
        if let Some(result) = completed.pop_front()
            && let Err(result) = output.push(result)
        {
            completed.push_front(result);
            thread::park_timeout(WORKER_POLL);
            continue;
        }
        let Some(block) = input.pop() else {
            thread::park_timeout(WORKER_POLL);
            continue;
        };
        // Once a block has been dequeued, all deadline-bound preparation,
        // inference, and output assembly belongs to the same Audio Work
        // Interval. Queue exchange and diagnostics stay outside the interval.
        let cycle_failure = priority_guard.run_inference_cycle(|| {
            let mut cycle_failure = None;
            let discontinuity = block.generation != generation || block.start_frame != next_start;
            if discontinuity {
                generation = block.generation;
                ready.iter_mut().for_each(VecDeque::clear);
                while let Some(pending_block) = pending.pop_front() {
                    completed.push_back(ProcessedBlock {
                        block: pending_block,
                        valid: false,
                        invalid_output: false,
                    });
                }
                failed = if let Err(error) = processor.reset() {
                    worker_errors.fetch_add(1, Ordering::Relaxed);
                    cycle_failure = Some(("reset", error));
                    true
                } else {
                    false
                };
            }
            next_start = block.start_frame.saturating_add(block.frames as u64);
            if failed {
                completed.push_back(ProcessedBlock {
                    block,
                    valid: false,
                    invalid_output: false,
                });
                return cycle_failure;
            }

            let planar = block_to_planar(&block, channels);
            let processed = processor.process(&planar).and_then(|processed| {
                append_ready(&mut ready, &processed, channels)
                    .map_err(|()| "neural worker returned invalid channel geometry".to_owned())
            });
            match processed {
                Ok(()) => {
                    pending.push_back(block);
                    complete_ready_blocks(&mut pending, &mut completed, &mut ready, channels);
                }
                Err(error) => {
                    failed = true;
                    worker_errors.fetch_add(1, Ordering::Relaxed);
                    cycle_failure = Some(("processing", error));
                    completed.push_back(ProcessedBlock {
                        block,
                        valid: false,
                        invalid_output: false,
                    });
                    while let Some(pending_block) = pending.pop_front() {
                        completed.push_back(ProcessedBlock {
                            block: pending_block,
                            valid: false,
                            invalid_output: false,
                        });
                    }
                    ready.iter_mut().for_each(VecDeque::clear);
                }
            }
            cycle_failure
        });
        if let Some((operation, error)) = cycle_failure {
            eprintln!("denoize Neural worker {operation} error: {error}");
        }
    }
}

fn block_to_planar(block: &AudioBlock, channels: usize) -> Vec<Vec<f64>> {
    let mut planar = (0..channels)
        .map(|_| Vec::with_capacity(block.frames))
        .collect::<Vec<_>>();
    for frame in 0..block.frames {
        for (channel, destination) in planar.iter_mut().enumerate() {
            destination.push(f64::from(block.samples[frame * channels + channel]));
        }
    }
    planar
}

fn append_ready(
    ready: &mut [VecDeque<f64>],
    processed: &[Vec<f64>],
    channels: usize,
) -> Result<(), ()> {
    if processed.len() != channels {
        return Err(());
    }
    let frames = processed.first().map_or(0, Vec::len);
    if processed.iter().any(|channel| channel.len() != frames) {
        return Err(());
    }
    for (destination, source) in ready.iter_mut().zip(processed) {
        destination.extend(source.iter().copied());
    }
    Ok(())
}

fn complete_ready_blocks(
    pending: &mut VecDeque<AudioBlock>,
    completed: &mut VecDeque<ProcessedBlock>,
    ready: &mut [VecDeque<f64>],
    channels: usize,
) {
    while pending.front().is_some_and(|block| {
        ready
            .first()
            .is_some_and(|queue| queue.len() >= block.frames)
    }) {
        let Some(mut block) = pending.pop_front() else {
            break;
        };
        let mut valid = true;
        for frame in 0..block.frames {
            for (channel, channel_ready) in ready.iter_mut().take(channels).enumerate() {
                let sample = channel_ready.pop_front().unwrap_or(0.0);
                if !sample.is_finite() || sample.abs() > MAX_OUTPUT_PEAK {
                    valid = false;
                }
                block.samples[frame * channels + channel] = if sample.is_finite() {
                    sample as f32
                } else {
                    0.0
                };
            }
        }
        completed.push_back(ProcessedBlock {
            block,
            valid,
            invalid_output: !valid,
        });
    }
}

fn validated_sample_rate(sample_rate: f64) -> Result<f64, String> {
    if !sample_rate.is_finite() || sample_rate < 1.0 || sample_rate > f64::from(MAX_SAMPLE_RATE) {
        return Err(format!(
            "denoize Neural requires a finite sample rate within [1, {MAX_SAMPLE_RATE}], got {sample_rate}"
        ));
    }
    Ok(sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[allow(unsafe_code)]
    mod allocation_counter {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;
        use std::sync::atomic::{AtomicUsize, Ordering};

        pub(super) struct CountingAllocator;

        thread_local! {
            static RECORDING: Cell<bool> = const { Cell::new(false) };
        }
        static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

        unsafe impl GlobalAlloc for CountingAllocator {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                record();
                // SAFETY: the allocation request is forwarded unchanged.
                unsafe { System.alloc(layout) }
            }

            unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
                // SAFETY: the allocation originated from the system allocator.
                unsafe { System.dealloc(pointer, layout) }
            }

            unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
                record();
                // SAFETY: the allocation request is forwarded unchanged.
                unsafe { System.alloc_zeroed(layout) }
            }

            unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                record();
                // SAFETY: the allocation originated from the system allocator.
                unsafe { System.realloc(pointer, layout, new_size) }
            }
        }

        #[global_allocator]
        static ALLOCATOR: CountingAllocator = CountingAllocator;

        fn record() {
            if RECORDING
                .try_with(|recording| recording.get())
                .unwrap_or(false)
            {
                ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            }
        }

        pub(super) fn start() {
            RECORDING.with(|recording| recording.set(false));
            ALLOCATIONS.store(0, Ordering::Relaxed);
            RECORDING.with(|recording| recording.set(true));
        }

        pub(super) fn stop() -> usize {
            RECORDING.with(|recording| recording.set(false));
            ALLOCATIONS.load(Ordering::Relaxed)
        }
    }

    struct IdentityProcessor;

    impl BlockProcessor for IdentityProcessor {
        fn process(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
            Ok(channels.to_vec())
        }

        fn reset(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    struct StalledProcessor;

    impl BlockProcessor for StalledProcessor {
        fn process(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
            thread::sleep(Duration::from_millis(250));
            Ok(channels.to_vec())
        }

        fn reset(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    fn test_engine<F>(channels: usize, factory: F) -> NeuralEngine<'static>
    where
        F: FnOnce() -> Result<Box<dyn BlockProcessor>, String> + Send + 'static,
    {
        let shared = Box::leak(Box::new(NeuralShared::new().unwrap()));
        NeuralEngine::new_with_factory(48_000.0, channels, shared, factory).unwrap()
    }

    #[test]
    fn state_is_closed_and_binds_model_and_latency() {
        let state =
            NeuralSessionState::new(NeuralPortConfiguration::Stereo, NeuralParameters::default())
                .unwrap();
        state.validate().unwrap();
        let bytes = serde_json::to_vec(&state).unwrap();
        let parsed: NeuralSessionState = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed, state);

        let mut object = serde_json::to_value(state).unwrap();
        object["future"] = serde_json::json!(true);
        assert!(serde_json::from_value::<NeuralSessionState>(object).is_err());

        let hq = NeuralSessionState::new_for_model(
            NeuralDawModel::Dpdfnet2,
            NeuralPortConfiguration::Mono,
            NeuralParameters::default(),
        )
        .unwrap();
        hq.validate_for_model(NeuralDawModel::Dpdfnet2).unwrap();
        assert!(hq.validate_for_model(NeuralDawModel::Gtcrn).is_err());
    }

    #[test]
    fn worker_queues_cover_the_declared_latency_window() {
        assert_eq!(QUEUE_BLOCKS, LATENCY_CHUNKS as usize);
        assert_eq!(BLOCK_POOL_SIZE, QUEUE_BLOCKS * 2 + 8);
    }

    #[test]
    fn lowest_tier_worker_gate_relaxes_only_scheduling_counters() {
        let scheduling_only = WorkerMetrics {
            overload_blocks: 3,
            late_blocks: 2,
            ..WorkerMetrics::default()
        };
        assert!(worker_metrics_pass(scheduling_only, false));
        assert!(!worker_metrics_pass(scheduling_only, true));
        assert!(!worker_metrics_pass(
            WorkerMetrics {
                invalid_blocks: 1,
                ..WorkerMetrics::default()
            },
            false,
        ));
        assert!(!worker_metrics_pass(
            WorkerMetrics {
                worker_errors: 1,
                ..WorkerMetrics::default()
            },
            false,
        ));
    }

    #[test]
    fn unsafe_worker_output_is_distinct_from_a_scheduling_fallback() {
        let mut pending = VecDeque::from([AudioBlock {
            generation: 1,
            start_frame: 0,
            frames: 1,
            samples: vec![0.0].into_boxed_slice(),
        }]);
        let mut completed = VecDeque::new();
        let mut ready = [VecDeque::from([MAX_OUTPUT_PEAK + 1.0])];
        complete_ready_blocks(&mut pending, &mut completed, &mut ready, 1);

        let result = completed.pop_front().unwrap();
        assert!(!result.valid);
        assert!(result.invalid_output);
    }

    #[test]
    fn identity_worker_is_aligned_to_the_declared_fixed_latency() {
        let mut engine = test_engine(1, || Ok(Box::new(IdentityProcessor)));
        let parameters = RuntimeParameters::from(NeuralParameters::default());
        let latency = engine.latency_frames as usize;
        let mut output = Vec::new();
        for frame in 0..latency + engine.chunk_frames * 4 {
            let sample = if frame == 0 { 0.75 } else { 0.0 };
            output.push(engine.process_frame([sample, 0.0], parameters)[0]);
            if frame % engine.chunk_frames == 0 {
                thread::yield_now();
            }
        }
        assert_eq!(
            output[..latency]
                .iter()
                .filter(|sample| **sample != 0.0)
                .count(),
            0
        );
        assert_eq!(output[latency], 0.75);
    }

    #[test]
    fn stalled_worker_never_blocks_the_callback_and_uses_delayed_dry() {
        let mut engine = test_engine(1, || Ok(Box::new(StalledProcessor)));
        let parameters = RuntimeParameters::from(NeuralParameters::default());
        let frames = engine.latency_frames as usize + engine.chunk_frames * (QUEUE_BLOCKS + 4);
        let started = Instant::now();
        let mut delayed_impulse = 0.0;
        for frame in 0..frames {
            let sample = if frame == 0 { 0.5 } else { 0.0 };
            let output = engine.process_frame([sample, 0.0], parameters)[0];
            if frame == engine.latency_frames as usize {
                delayed_impulse = output;
            }
        }
        assert_eq!(delayed_impulse, 0.5);
        assert!(started.elapsed() < Duration::from_millis(100));
        assert!(engine.metrics.overload_blocks.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn unavailable_model_keeps_the_processor_active_with_delayed_dry() {
        let shared = Box::leak(Box::new(NeuralShared::new().unwrap()));
        let mut engine = NeuralEngine::new_with_factory(48_000.0, 1, shared, || {
            Err("pinned model is unavailable".to_owned())
        })
        .unwrap();
        assert!(engine.worker.is_none());
        assert_eq!(shared.worker_errors.load(Ordering::Relaxed), 1);

        let latency = engine.latency_frames as usize;
        let frames = latency + engine.chunk_frames * (QUEUE_BLOCKS + 4);
        for frame in 0..frames {
            let input = match frame {
                0 => 0.5,
                1 => 0.25,
                2 => 0.125,
                3 => 0.0625,
                _ => 0.0,
            };
            let parameters = match frame.checked_sub(latency) {
                Some(1) => RuntimeParameters {
                    bypass: false,
                    mix: 1.0,
                    output_gain: 1.0,
                    fallback: OverloadFallback::Silence,
                },
                Some(2) => RuntimeParameters {
                    bypass: false,
                    mix: 0.5,
                    output_gain: 2.0,
                    fallback: OverloadFallback::Silence,
                },
                Some(3) => RuntimeParameters {
                    bypass: true,
                    mix: 1.0,
                    output_gain: 2.0,
                    fallback: OverloadFallback::Silence,
                },
                _ => RuntimeParameters::from(NeuralParameters::default()),
            };
            let output = engine.process_frame([input, 0.0], parameters)[0];
            if frame < latency {
                assert_eq!(output, 0.0);
            } else {
                match frame - latency {
                    0 => assert_eq!(output, 0.5),
                    1 => assert_eq!(output, 0.0),
                    2 => assert_eq!(output, 0.125),
                    3 => assert_eq!(output, 0.125),
                    _ => {}
                }
            }
        }
        assert_eq!(engine.input_queue.len(), 0);
        assert_eq!(engine.output_queue.len(), 0);
        assert_eq!(shared.overload_blocks.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn host_evidence_retains_worker_start_after_deactivation() {
        let mut engine = test_engine(1, || Ok(Box::new(IdentityProcessor)));
        assert!(engine.worker_started);
        assert_eq!(engine.processed_frames, 0);
        assert!(!engine.should_write_host_evidence());

        engine.process_frame(
            [0.1, 0.0],
            RuntimeParameters::from(NeuralParameters::default()),
        );
        engine.stop();

        assert!(engine.worker.is_none());
        assert!(engine.worker_started);
        assert_eq!(engine.processed_frames, 1);
        assert!(engine.should_write_host_evidence());
    }

    #[test]
    fn host_evidence_separates_priming_from_the_measured_window() {
        let mut engine = test_engine(1, || Ok(Box::new(IdentityProcessor)));
        engine.host_evidence_warmup_frames = 1;
        engine.host_evidence_baseline = None;
        engine.metrics.overload_blocks.store(3, Ordering::Relaxed);

        engine.process_frame(
            [0.1, 0.0],
            RuntimeParameters::from(NeuralParameters::default()),
        );
        assert_eq!(
            engine.host_evidence_baseline,
            Some(WorkerMetrics {
                overload_blocks: 3,
                ..WorkerMetrics::default()
            })
        );

        engine.metrics.overload_blocks.store(5, Ordering::Relaxed);
        let lifetime = engine.metrics.worker_metrics();
        let measured = lifetime.saturating_since(engine.host_evidence_baseline.unwrap());
        assert_eq!(lifetime.overload_blocks, 5);
        assert_eq!(measured.overload_blocks, 2);
    }

    #[test]
    fn host_callback_geometry_records_activation_and_observed_bounds() {
        let mut engine = test_engine(1, || Ok(Box::new(IdentityProcessor)));
        engine.host_min_frames_count = 64;
        engine.host_max_frames_count = 1_024;
        engine.record_callback(128);
        engine.record_callback(1_024);
        engine.record_callback(480);

        assert_eq!(engine.host_min_frames_count, 64);
        assert_eq!(engine.host_max_frames_count, 1_024);
        assert_eq!(engine.callback_calls, 3);
        assert_eq!(engine.callback_min_frames, 128);
        assert_eq!(engine.callback_max_frames, 1_024);
    }

    #[test]
    fn worker_panic_during_shutdown_is_recorded() {
        struct PanicOnDrop;

        impl BlockProcessor for PanicOnDrop {
            fn process(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
                Ok(channels.to_vec())
            }

            fn reset(&mut self) -> Result<(), String> {
                Ok(())
            }
        }

        impl Drop for PanicOnDrop {
            fn drop(&mut self) {
                panic!("injected worker shutdown panic");
            }
        }

        let mut engine = test_engine(1, || Ok(Box::new(PanicOnDrop)));
        assert!(engine.finished_gracefully);
        engine.stop();
        assert!(!engine.finished_gracefully);
        assert_eq!(engine.metrics.worker_errors.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn repeated_bypass_updates_remain_latched() {
        let shared = NeuralShared::new().unwrap();
        for expected in [true, false, true] {
            assert!(
                shared
                    .parameters
                    .set_value(PARAM_BYPASS, f64::from(bool_value(expected)))
            );
            assert_eq!(shared.parameters.snapshot().bypass, expected);
        }
    }

    #[test]
    fn activation_warms_and_resets_the_worker_before_returning() {
        struct WarmupProbe {
            process_calls: Arc<AtomicU64>,
            reset_calls: Arc<AtomicU64>,
        }

        impl BlockProcessor for WarmupProbe {
            fn process(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
                self.process_calls.fetch_add(1, Ordering::Relaxed);
                Ok(channels
                    .iter()
                    .map(|channel| channel[..channel.len() - 1].to_vec())
                    .collect())
            }

            fn reset(&mut self) -> Result<(), String> {
                self.reset_calls.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
        }

        let process_calls = Arc::new(AtomicU64::new(0));
        let reset_calls = Arc::new(AtomicU64::new(0));
        let observed_process_calls = Arc::clone(&process_calls);
        let observed_reset_calls = Arc::clone(&reset_calls);
        let mut engine = test_engine(2, move || {
            Ok(Box::new(WarmupProbe {
                process_calls,
                reset_calls,
            }))
        });

        assert!(engine.worker_started);
        assert_eq!(observed_process_calls.load(Ordering::Relaxed), 1);
        assert_eq!(observed_reset_calls.load(Ordering::Relaxed), 1);
        assert_eq!(engine.processed_frames, 0);
        assert_eq!(engine.input_queue.len(), 0);
        assert_eq!(engine.output_queue.len(), 0);
        engine.stop();
    }

    #[test]
    fn callback_path_allocates_zero_bytes_after_activation() {
        let mut engine = test_engine(2, || Ok(Box::new(IdentityProcessor)));
        let parameters = RuntimeParameters::from(NeuralParameters::default());
        for _ in 0..engine.chunk_frames * 2 {
            engine.process_frame([0.1, -0.1], parameters);
        }

        allocation_counter::start();
        for _ in 0..engine.chunk_frames * 1_000 {
            engine.process_frame([0.1, -0.1], parameters);
        }
        let allocations = allocation_counter::stop();
        assert_eq!(allocations, 0, "callback allocated {allocations} times");
    }

    #[test]
    fn reset_rejects_stale_results_from_the_previous_generation() {
        let mut engine = test_engine(1, || Ok(Box::new(IdentityProcessor)));
        let parameters = RuntimeParameters::from(NeuralParameters::default());
        for _ in 0..engine.chunk_frames * 2 {
            engine.process_frame([0.8, 0.0], parameters);
        }
        engine.reset();
        for _ in 0..engine.latency_frames as usize + engine.chunk_frames * 2 {
            assert_eq!(engine.process_frame([0.0, 0.0], parameters)[0], 0.0);
        }
    }

    #[test]
    fn reset_reclaims_queued_input_from_the_previous_generation() {
        struct ResetBlockingProcessor {
            warmed: bool,
            started: mpsc::SyncSender<()>,
            release: mpsc::Receiver<()>,
        }

        impl BlockProcessor for ResetBlockingProcessor {
            fn process(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
                if self.warmed {
                    let _ = self.started.send(());
                    let _ = self.release.recv();
                }
                Ok(channels.to_vec())
            }

            fn reset(&mut self) -> Result<(), String> {
                self.warmed = true;
                Ok(())
            }
        }

        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let mut engine = test_engine(1, move || {
            Ok(Box::new(ResetBlockingProcessor {
                warmed: false,
                started: started_tx,
                release: release_rx,
            }))
        });
        let parameters = RuntimeParameters::from(NeuralParameters::default());

        for _ in 0..engine.chunk_frames {
            engine.process_frame([0.1, 0.0], parameters);
        }
        started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("worker did not begin processing the first generation");
        for _ in 0..engine.chunk_frames * 4 {
            engine.process_frame([0.2, 0.0], parameters);
        }

        let queued = engine.input_queue.len();
        let free_before_reset = engine.free_blocks.len();
        assert!(queued >= 4);
        engine.reset();
        assert!(engine.input_queue.is_empty());
        assert_eq!(engine.free_blocks.len(), free_before_reset + queued);

        release_tx.send(()).unwrap();
        engine.stop();
    }

    #[test]
    fn parameters_and_port_contracts_are_stable() {
        assert_eq!(EDITOR_PARAMETERS.len(), PARAMETER_COUNT as usize);
        for index in 0..PARAMETER_COUNT {
            let info = parameter_info(index).unwrap();
            let editor = &EDITOR_PARAMETERS[index as usize];
            assert_eq!(info.id.get(), index);
            assert_eq!(editor.id, info.id.get());
            assert_eq!(editor.name.as_bytes(), info.name);
            assert_eq!(editor.minimum, info.min_value);
            assert_eq!(editor.maximum, info.max_value);
            assert_eq!(editor.default, info.default_value);
            assert!(info.min_value <= info.default_value);
            assert!(info.default_value <= info.max_value);
        }
        let bypass = parameter_info(PARAM_BYPASS.get()).unwrap();
        assert!(bypass.flags.contains(ParamInfoFlags::IS_STEPPED));
        assert!(bypass.flags.contains(ParamInfoFlags::IS_AUTOMATABLE));
        assert!(!bypass.flags.contains(ParamInfoFlags::IS_BYPASS));
        assert!(parameter_info(PARAMETER_COUNT).is_none());
        assert_eq!(
            port_configuration_from_id(MONO_CONFIG_ID),
            Some(NeuralPortConfiguration::Mono)
        );
        assert_eq!(
            port_configuration_from_id(STEREO_CONFIG_ID),
            Some(NeuralPortConfiguration::Stereo)
        );
    }

    #[test]
    #[ignore = "requires the pinned managed GTCRN model and cargo test --release"]
    fn pinned_gtcrn_release_worker_meets_sustained_deadlines() {
        assert_pinned_release_worker(NeuralDawModel::Gtcrn, true);
    }

    #[test]
    #[ignore = "requires the pinned managed DPDFNet model and cargo test --release"]
    #[cfg(feature = "experimental-dpdfnet-hq")]
    fn pinned_dpdfnet2_release_worker_meets_sustained_deadlines() {
        assert_pinned_release_worker(NeuralDawModel::Dpdfnet2, true);
    }

    #[test]
    #[ignore = "requires the pinned managed DPDFNet model and cargo test --release"]
    #[cfg(feature = "experimental-dpdfnet-hq")]
    fn pinned_dpdfnet2_release_worker_measures_lowest_tier_capacity() {
        assert_pinned_release_worker(NeuralDawModel::Dpdfnet2, false);
    }

    fn assert_pinned_release_worker(model: NeuralDawModel, require_zero_scheduling_counters: bool) {
        assert!(
            !cfg!(debug_assertions),
            "the sustained neural deadline gate must exercise the release profile"
        );
        let shared = Box::leak(Box::new(NeuralShared::new_for_model(model).unwrap()));
        let mut engine = NeuralEngine::new_model(model, 48_000.0, 1, shared).unwrap();
        let parameters = RuntimeParameters::from(NeuralParameters::default());

        // Activation does the one-time tract kernel selection off the audio
        // callback and resets recurrent state before the worker is published.
        assert_eq!(shared.worker_errors.load(Ordering::Relaxed), 0);
        assert!(engine.worker_started);
        assert_eq!(shared.overload_blocks.load(Ordering::Relaxed), 0);
        assert_eq!(shared.late_blocks.load(Ordering::Relaxed), 0);
        assert_eq!(shared.invalid_blocks.load(Ordering::Relaxed), 0);

        let paced_seconds = std::env::var("DENOIZE_NEURAL_WORKER_SECONDS")
            .map(|value| {
                value
                    .parse::<usize>()
                    .expect("DENOIZE_NEURAL_WORKER_SECONDS must be an integer")
            })
            .unwrap_or(1);
        assert!(
            (1..=3_600).contains(&paced_seconds),
            "DENOIZE_NEURAL_WORKER_SECONDS must be between 1 and 3600"
        );
        let paced_blocks = paced_seconds * 100;
        let latency = engine.latency_frames as usize;
        assert_eq!(
            latency % engine.chunk_frames,
            0,
            "worker latency must contain whole callback blocks"
        );
        let latency_blocks = latency / engine.chunk_frames;
        let total_blocks = latency_blocks + paced_blocks;
        let frames = engine.chunk_frames * total_blocks;
        let mut finite = 0usize;
        let mut neural_frames = 0usize;
        let mut inputs = Vec::with_capacity(frames);
        let mut noise_state = 0x5eed_1234_9876_abcd_u64;
        // Fixture generation is not part of the simulated callback cadence.
        // Build it before starting the wall-clock measurement so every block
        // can be presented on one absolute 10 ms schedule.
        for frame in 0..frames {
            let phase = frame as f64 * 440.0 * std::f64::consts::TAU / 48_000.0;
            noise_state = noise_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let noise = f64::from((noise_state >> 32) as u32) / f64::from(u32::MAX) - 0.5;
            let input = phase.sin() * 0.03 + noise * 0.08;
            inputs.push(input);
        }

        let measurement_started = Instant::now();
        for block in 0..total_blocks {
            let due = measurement_started
                + Duration::from_micros(
                    (block as u64)
                        .saturating_mul(u64::from(denoize::NEURAL_DAW_CHUNK_MILLIS))
                        .saturating_mul(1_000),
                );
            if let Some(delay) = due.checked_duration_since(Instant::now()) {
                thread::sleep(delay);
            }
            let start = block * engine.chunk_frames;
            for frame in start..start + engine.chunk_frames {
                let output = engine.process_frame([inputs[frame], 0.0], parameters)[0];
                finite += usize::from(output.is_finite());
                if frame >= latency && (output - inputs[frame - latency]).abs() > 1.0e-6 {
                    neural_frames += 1;
                }
            }
        }
        let measurement_wall_seconds = measurement_started.elapsed().as_secs_f64();
        let worker_metrics = shared.worker_metrics();
        // Preserve the complete run even when a following gate assertion
        // fails, so rejected CI artifacts identify the actual counter rather
        // than stopping at the first assertion without worker evidence.
        write_worker_evidence(
            model,
            &engine,
            frames,
            finite,
            neural_frames,
            paced_blocks,
            measurement_wall_seconds,
            shared,
        );
        assert_eq!(finite, frames);
        assert!(
            neural_frames >= engine.chunk_frames,
            "the pinned worker never produced one complete non-fallback block: neural_frames={neural_frames}, overload={}, late={}, invalid={}, worker_errors={}, input_queue={}, output_queue={}, ready={}",
            shared.overload_blocks.load(Ordering::Relaxed),
            shared.late_blocks.load(Ordering::Relaxed),
            shared.invalid_blocks.load(Ordering::Relaxed),
            shared.worker_errors.load(Ordering::Relaxed),
            engine.input_queue.len(),
            engine.output_queue.len(),
            engine.ready.len(),
        );
        assert!(
            worker_metrics_pass(worker_metrics, require_zero_scheduling_counters),
            "worker metrics did not pass this tier: {worker_metrics:?}"
        );
    }

    fn worker_metrics_pass(metrics: WorkerMetrics, require_zero_scheduling_counters: bool) -> bool {
        metrics.worker_errors == 0
            && metrics.invalid_blocks == 0
            && (!require_zero_scheduling_counters || metrics == WorkerMetrics::default())
    }

    fn write_worker_evidence(
        model: NeuralDawModel,
        engine: &NeuralEngine<'_>,
        measured_frames: usize,
        finite_frames: usize,
        neural_frames: usize,
        paced_blocks: usize,
        measurement_wall_seconds: f64,
        shared: &NeuralShared,
    ) {
        let Ok(path) = std::env::var("DENOIZE_NEURAL_WORKER_EVIDENCE") else {
            return;
        };
        let document = serde_json::json!({
            "schema": "denoize-dpdfnet-worker-run-v1",
            "schema_version": 1,
            "source_commit": std::env::var("DENOIZE_EVIDENCE_SOURCE_COMMIT").unwrap_or_default(),
            "model_id": model.model_id(),
            "model_sha256": model.model_sha256(),
            "plugin_id": model.plugin_id(),
            "sample_rate_hz": 48_000,
            "channels": 1,
            "chunk_frames": engine.chunk_frames,
            "latency_frames": engine.latency_frames,
            "paced_blocks": paced_blocks,
            "measured_frames": measured_frames,
            "finite_frames": finite_frames,
            "neural_frames": neural_frames,
            "measurement_wall_seconds": measurement_wall_seconds,
            "metrics": {
                "overload_blocks": shared.overload_blocks.load(Ordering::Relaxed),
                "late_blocks": shared.late_blocks.load(Ordering::Relaxed),
                "invalid_blocks": shared.invalid_blocks.load(Ordering::Relaxed),
                "worker_errors": shared.worker_errors.load(Ordering::Relaxed),
            },
            "queues_after_run": {
                "input": engine.input_queue.len(),
                "output": engine.output_queue.len(),
                "ready": engine.ready.len(),
            },
            "environment": {
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
                "logical_parallelism": std::thread::available_parallelism()
                    .map(|value| value.get())
                    .unwrap_or(0),
                "target": std::env::var("DENOIZE_EVIDENCE_TARGET").unwrap_or_default(),
                "cpu_model": std::env::var("DENOIZE_EVIDENCE_CPU_MODEL").unwrap_or_default(),
                "hardware_tier": std::env::var("DENOIZE_EVIDENCE_HARDWARE_TIER").unwrap_or_default(),
                "runner_label": std::env::var("DENOIZE_EVIDENCE_RUNNER_LABEL").unwrap_or_default(),
            },
        });
        let bytes = serde_json::to_vec_pretty(&document).expect("encode worker evidence");
        let mut destination = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap_or_else(|error| panic!("create worker evidence {path}: {error}"));
        destination
            .write_all(&bytes)
            .unwrap_or_else(|error| panic!("write worker evidence {path}: {error}"));
        destination
            .write_all(b"\n")
            .unwrap_or_else(|error| panic!("finish worker evidence {path}: {error}"));
    }
}
