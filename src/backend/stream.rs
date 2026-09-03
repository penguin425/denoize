//! Stateful backend session shared by bounded file and realtime processing.

use std::collections::VecDeque;
use std::marker::PhantomData;

use super::{Backend, BackendOptions, ChannelMode};
use crate::config::{ConfigError, MAX_STREAM_BLOCK_FRAMES, MAX_STREAM_CHANNELS};
use crate::{
    select_accelerator_for_options, AcceleratorSelection, DenoiserConfig, StreamingDenoiser,
};

#[cfg(feature = "rnnoise")]
const RNNOISE_STATE_ALLOWANCE_PER_CHANNEL: u64 = 2 * 1024 * 1024;

/// A reusable stateful denoising session for continuous planar audio.
///
/// Supported backends retain their overlap, recurrent model, and resampler
/// state between calls. Model-backed sessions load and optimize their graph at
/// construction and never reopen it while processing or resetting the stream.
pub struct StreamingBackendSession {
    backend: Backend,
    accelerator: AcceleratorSelection,
    input_channels: usize,
    processor_channels: usize,
    channel_mode: ChannelMode,
    denoiser: DenoiserConfig,
    processor: StreamingBackend,
    vad: Option<crate::vad::StreamingVad>,
    linked_original: VecDeque<(f64, f64)>,
    finished: bool,
}

enum StreamingBackend {
    Classical(StreamingDenoiser),
    #[cfg(feature = "rnnoise")]
    Rnnoise(Box<super::rnnoise::StreamingProcessor>),
    #[cfg(feature = "deepfilter")]
    DeepFilter(Box<super::deepfilter::StreamingProcessor>),
    #[cfg(feature = "mossformer2")]
    Mossformer2(Box<super::mossformer2::StreamingProcessor>),
    #[cfg(feature = "gtcrn")]
    Gtcrn(Box<super::gtcrn::StreamingProcessor>),
    #[cfg(feature = "dpdfnet")]
    Dpdfnet(Box<super::dpdfnet::StreamingProcessor>),
}

impl StreamingBackend {
    fn process_block(&mut self, input: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        match self {
            Self::Classical(processor) => processor.process_block(input),
            #[cfg(feature = "rnnoise")]
            Self::Rnnoise(processor) => processor.process_block(input),
            #[cfg(feature = "deepfilter")]
            Self::DeepFilter(processor) => processor.process_block(input),
            #[cfg(feature = "mossformer2")]
            Self::Mossformer2(processor) => processor.process_block(input),
            #[cfg(feature = "gtcrn")]
            Self::Gtcrn(processor) => processor.process_block(input),
            #[cfg(feature = "dpdfnet")]
            Self::Dpdfnet(processor) => processor.process_block(input),
        }
    }

    fn process_owned_block(&mut self, input: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
        match self {
            #[cfg(feature = "dpdfnet")]
            Self::Dpdfnet(processor) => processor.process_owned_block(input),
            Self::Classical(processor) => processor.process_block(&input),
            #[cfg(feature = "rnnoise")]
            Self::Rnnoise(processor) => processor.process_block(&input),
            #[cfg(feature = "deepfilter")]
            Self::DeepFilter(processor) => processor.process_block(&input),
            #[cfg(feature = "mossformer2")]
            Self::Mossformer2(processor) => processor.process_block(&input),
            #[cfg(feature = "gtcrn")]
            Self::Gtcrn(processor) => processor.process_block(&input),
        }
    }
}

#[derive(Default)]
struct StreamConstruction<'a> {
    daw_host_rate: bool,
    #[cfg(feature = "gtcrn")]
    gtcrn: Option<&'a super::gtcrn::GtcrnModel>,
    #[cfg(feature = "dpdfnet")]
    dpdfnet: Option<&'a super::dpdfnet::DpdfnetModel>,
    marker: PhantomData<&'a ()>,
}

impl StreamingBackendSession {
    /// Return whether a compiled backend has a continuous stateful adapter.
    #[allow(unreachable_patterns)]
    pub fn supports(backend: Backend) -> bool {
        match backend {
            Backend::Classical => true,
            #[cfg(feature = "rnnoise")]
            Backend::Rnnoise => true,
            #[cfg(feature = "deepfilter")]
            Backend::DeepFilter => true,
            #[cfg(feature = "mossformer2")]
            Backend::Mossformer2 => true,
            #[cfg(feature = "gtcrn")]
            Backend::Gtcrn => true,
            #[cfg(feature = "dpdfnet")]
            Backend::Dpdfnet => true,
            _ => false,
        }
    }

    /// Construct a stream and allocate every backend state before accepting
    /// audio. `backend_options` must already contain any managed model path.
    pub fn new(
        backend: Backend,
        sample_rate: u32,
        channels: usize,
        denoiser: DenoiserConfig,
        backend_options: BackendOptions,
    ) -> Result<Self, String> {
        let accelerator = select_accelerator_for_options(backend, &backend_options)?;
        Self::new_with_accelerator(
            backend,
            sample_rate,
            channels,
            denoiser,
            backend_options,
            accelerator,
        )
    }

    /// Construct a stream using an already-resolved accelerator snapshot.
    ///
    /// This keeps capability reporting, recipe identity, and model preparation
    /// bound to the same decision when a frontend preflights the runtime before
    /// opening an input or audio device.
    pub fn new_with_accelerator(
        backend: Backend,
        sample_rate: u32,
        channels: usize,
        denoiser: DenoiserConfig,
        backend_options: BackendOptions,
        accelerator: AcceleratorSelection,
    ) -> Result<Self, String> {
        Self::new_with_accelerator_inner(
            backend,
            sample_rate,
            channels,
            denoiser,
            backend_options,
            accelerator,
            StreamConstruction::default(),
        )
    }

    /// Create a GTCRN stream from an already authenticated and optimized graph.
    ///
    /// The options still undergo the ordinary deterministic/runtime validation;
    /// callers must prepare `model` for the resulting effective runtime.
    #[cfg(feature = "gtcrn")]
    pub fn new_gtcrn_with_prepared_model(
        sample_rate: u32,
        channels: usize,
        denoiser: DenoiserConfig,
        backend_options: BackendOptions,
        model: &super::gtcrn::GtcrnModel,
    ) -> Result<Self, String> {
        Self::new_gtcrn_with_prepared_model_contract(
            sample_rate,
            channels,
            denoiser,
            backend_options,
            model,
            false,
        )
    }

    /// Construct the GTCRN stream at the wider, bounded DAW host-rate limit.
    ///
    /// This is format-neutral and retains the ordinary backend, resource, and
    /// accelerator checks. Only file/offline sample-rate validation is
    /// replaced by the plug-in host contract.
    #[cfg(feature = "gtcrn")]
    pub fn new_gtcrn_for_daw(
        sample_rate: u32,
        channels: usize,
        denoiser: DenoiserConfig,
        backend_options: BackendOptions,
    ) -> Result<Self, String> {
        let backend = Backend::Gtcrn;
        let accelerator = select_accelerator_for_options(backend, &backend_options)?;
        Self::new_with_accelerator_inner(
            backend,
            sample_rate,
            channels,
            denoiser,
            backend_options,
            accelerator,
            StreamConstruction {
                daw_host_rate: true,
                ..StreamConstruction::default()
            },
        )
    }

    /// Create a prepared GTCRN stream at the wider, bounded DAW host-rate limit.
    ///
    /// This is format-neutral and retains the ordinary backend, model,
    /// resource, and accelerator checks. Only file/offline sample-rate
    /// validation is replaced by the plug-in host contract.
    #[cfg(feature = "gtcrn")]
    pub fn new_gtcrn_for_daw_with_prepared_model(
        sample_rate: u32,
        channels: usize,
        denoiser: DenoiserConfig,
        backend_options: BackendOptions,
        model: &super::gtcrn::GtcrnModel,
    ) -> Result<Self, String> {
        Self::new_gtcrn_with_prepared_model_contract(
            sample_rate,
            channels,
            denoiser,
            backend_options,
            model,
            true,
        )
    }

    #[cfg(feature = "gtcrn")]
    fn new_gtcrn_with_prepared_model_contract(
        sample_rate: u32,
        channels: usize,
        denoiser: DenoiserConfig,
        backend_options: BackendOptions,
        model: &super::gtcrn::GtcrnModel,
        daw_host_rate: bool,
    ) -> Result<Self, String> {
        let backend = Backend::Gtcrn;
        let accelerator = select_accelerator_for_options(backend, &backend_options)?;
        if model.runtime() != accelerator.effective() {
            return Err(format!(
                "prepared GTCRN graph uses {}, but the effective stream runtime is {}",
                model.runtime().name(),
                accelerator.effective().name()
            ));
        }
        Self::new_with_accelerator_inner(
            backend,
            sample_rate,
            channels,
            denoiser,
            backend_options,
            accelerator,
            StreamConstruction {
                daw_host_rate,
                gtcrn: Some(model),
                ..StreamConstruction::default()
            },
        )
    }

    /// Construct a DPDFNet-2 stream at the bounded DAW host-rate limit.
    #[cfg(feature = "dpdfnet")]
    pub fn new_dpdfnet_for_daw(
        sample_rate: u32,
        channels: usize,
        denoiser: DenoiserConfig,
        backend_options: BackendOptions,
    ) -> Result<Self, String> {
        let backend = Backend::Dpdfnet;
        let accelerator = select_accelerator_for_options(backend, &backend_options)?;
        Self::new_with_accelerator_inner(
            backend,
            sample_rate,
            channels,
            denoiser,
            backend_options,
            accelerator,
            StreamConstruction {
                daw_host_rate: true,
                ..StreamConstruction::default()
            },
        )
    }

    /// Create a prepared DPDFNet-2 stream at the bounded DAW host-rate limit.
    #[cfg(feature = "dpdfnet")]
    pub fn new_dpdfnet_for_daw_with_prepared_model(
        sample_rate: u32,
        channels: usize,
        denoiser: DenoiserConfig,
        backend_options: BackendOptions,
        model: &super::dpdfnet::DpdfnetModel,
    ) -> Result<Self, String> {
        let backend = Backend::Dpdfnet;
        let accelerator = select_accelerator_for_options(backend, &backend_options)?;
        if model.runtime() != accelerator.effective() {
            return Err(format!(
                "prepared DPDFNet graph uses {}, but the effective stream runtime is {}",
                model.runtime().name(),
                accelerator.effective().name()
            ));
        }
        Self::new_with_accelerator_inner(
            backend,
            sample_rate,
            channels,
            denoiser,
            backend_options,
            accelerator,
            StreamConstruction {
                daw_host_rate: true,
                dpdfnet: Some(model),
                ..StreamConstruction::default()
            },
        )
    }

    fn new_with_accelerator_inner(
        backend: Backend,
        sample_rate: u32,
        channels: usize,
        mut denoiser: DenoiserConfig,
        backend_options: BackendOptions,
        accelerator: AcceleratorSelection,
        construction: StreamConstruction<'_>,
    ) -> Result<Self, String> {
        if channels == 0 || channels > MAX_STREAM_CHANNELS {
            return Err(format!(
                "streaming backend channels must be between 1 and {MAX_STREAM_CHANNELS}"
            ));
        }
        if !Self::supports(backend) {
            return Err("selected backend does not support stateful streaming".into());
        }
        denoiser.sample_rate = sample_rate;
        if construction.daw_host_rate {
            denoiser
                .validate_daw_config()
                .map_err(|error| error.to_string())?;
        } else {
            denoiser
                .validate_config()
                .map_err(|error| error.to_string())?;
        }
        backend_options.validate_resolved_resources(backend)?;
        crate::hardware::validate_accelerator_selection(
            backend,
            backend_options.accelerator,
            backend_options.deterministic,
            accelerator,
        )?;
        let stereo_mode = channels == 2 && backend_options.channel_mode != ChannelMode::Independent;
        let processor_channels =
            if stereo_mode && backend_options.channel_mode == ChannelMode::StereoLinked {
                1
            } else {
                channels
            };
        let _ = (sample_rate, processor_channels);
        let vad = denoiser
            .vad
            .then(|| {
                crate::vad::StreamingVad::new(
                    sample_rate,
                    channels,
                    denoiser.vad_silence_gain,
                    denoiser.vad_speech_mix,
                )
            })
            .transpose()?;
        let mut processor_denoiser = denoiser.clone();
        processor_denoiser.vad = false;
        let processor = Self::build_processor(
            backend,
            sample_rate,
            processor_channels,
            &processor_denoiser,
            &backend_options,
            accelerator,
            &construction,
        )?;
        Ok(Self {
            backend,
            accelerator,
            input_channels: channels,
            processor_channels,
            channel_mode: if stereo_mode {
                backend_options.channel_mode
            } else {
                ChannelMode::Independent
            },
            denoiser,
            processor,
            vad,
            linked_original: VecDeque::new(),
            finished: false,
        })
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    #[must_use]
    pub const fn accelerator(&self) -> AcceleratorSelection {
        self.accelerator
    }

    /// Conservative backend-specific state beyond the classical stream and
    /// caller-owned input/output blocks.
    pub fn estimate_additional_bytes(
        backend: Backend,
        sample_rate: u32,
        channels: usize,
        channel_mode: ChannelMode,
    ) -> Result<u64, ConfigError> {
        if channels == 0 || channels > MAX_STREAM_CHANNELS {
            return Err(ConfigError::invalid("channels", "an integer in 1..=64"));
        }
        let processor_channels = if channels == 2 && channel_mode == ChannelMode::StereoLinked {
            1
        } else {
            channels
        };
        let _ = (sample_rate, processor_channels);
        match backend {
            Backend::Classical => Ok(0),
            #[cfg(feature = "rnnoise")]
            Backend::Rnnoise => {
                let resamplers = resampler_pair_bytes(
                    processor_channels,
                    sample_rate,
                    48_000,
                    "RNNoise stream resamplers",
                )?;
                let state = u64::try_from(processor_channels)
                    .ok()
                    .and_then(|channels| channels.checked_mul(RNNOISE_STATE_ALLOWANCE_PER_CHANNEL))
                    .ok_or(ConfigError::ResourceOverflow {
                        resource: "RNNoise stream state",
                    })?;
                resamplers
                    .checked_add(state)
                    .ok_or(ConfigError::ResourceOverflow {
                        resource: "RNNoise stream state",
                    })
            }
            #[cfg(feature = "deepfilter")]
            Backend::DeepFilter => {
                let resamplers = resampler_pair_bytes(
                    processor_channels,
                    sample_rate,
                    48_000,
                    "DeepFilterNet stream resamplers",
                )?;
                resamplers
                    .checked_add(super::deepfilter::streaming_state_bytes(
                        processor_channels,
                        sample_rate,
                        channels,
                    )?)
                    .ok_or(ConfigError::ResourceOverflow {
                        resource: "DeepFilterNet stream state",
                    })
            }
            #[cfg(feature = "mossformer2")]
            Backend::Mossformer2 => {
                let resamplers = resampler_pair_bytes(
                    processor_channels,
                    sample_rate,
                    48_000,
                    "MossFormer2 stream resamplers",
                )?;
                resamplers
                    .checked_add(super::mossformer2::streaming_state_bytes(
                        processor_channels,
                        sample_rate,
                        channels,
                    )?)
                    .ok_or(ConfigError::ResourceOverflow {
                        resource: "MossFormer2 stream state",
                    })
            }
            #[cfg(feature = "gtcrn")]
            Backend::Gtcrn => {
                let resamplers = resampler_pair_bytes(
                    processor_channels,
                    sample_rate,
                    super::gtcrn::SAMPLE_RATE,
                    "GTCRN stream resamplers",
                )?;
                resamplers
                    .checked_add(super::gtcrn::streaming_state_bytes(processor_channels)?)
                    .ok_or(ConfigError::ResourceOverflow {
                        resource: "GTCRN stream state",
                    })
            }
            #[cfg(feature = "dpdfnet")]
            Backend::Dpdfnet => {
                let resamplers = resampler_pair_bytes(
                    processor_channels,
                    sample_rate,
                    super::dpdfnet::SAMPLE_RATE,
                    "DPDFNet stream resamplers",
                )?;
                resamplers
                    .checked_add(super::dpdfnet::streaming_state_bytes(processor_channels)?)
                    .ok_or(ConfigError::ResourceOverflow {
                        resource: "DPDFNet stream state",
                    })
            }
            #[allow(unreachable_patterns)]
            _ => Err(ConfigError::invalid(
                "backend",
                "a compiled backend with stateful streaming support",
            )),
        }
    }

    /// Conservative VAD alignment state beyond the backend and ordinary
    /// caller-owned stream blocks.
    pub fn estimate_vad_additional_bytes(
        sample_rate: u32,
        channels: usize,
        block_frames: usize,
        frame_size: usize,
        profile_ms: f64,
    ) -> Result<u64, ConfigError> {
        crate::vad::estimate_streaming_bytes(
            sample_rate,
            channels,
            block_frames,
            frame_size,
            profile_ms,
        )
    }

    /// Process a block. The returned block can be empty while bounded model or
    /// sample-rate-converter latency is retained.
    pub fn process_block(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if self.finished {
            return Err("streaming backend session has already been finished".into());
        }
        self.process_block_with_limit(channels, MAX_STREAM_BLOCK_FRAMES)
    }

    /// Flush every pending model frame and converter delay exactly once.
    pub fn finish(&mut self) -> Result<Vec<Vec<f64>>, String> {
        if self.finished {
            return Err("streaming backend session has already been finished".into());
        }
        let processed = match &mut self.processor {
            StreamingBackend::Classical(processor) => processor.finish(),
            #[cfg(feature = "rnnoise")]
            StreamingBackend::Rnnoise(processor) => processor.finish(),
            #[cfg(feature = "deepfilter")]
            StreamingBackend::DeepFilter(processor) => processor.finish(),
            #[cfg(feature = "mossformer2")]
            StreamingBackend::Mossformer2(processor) => processor.finish(),
            #[cfg(feature = "gtcrn")]
            StreamingBackend::Gtcrn(processor) => processor.finish(),
            #[cfg(feature = "dpdfnet")]
            StreamingBackend::Dpdfnet(processor) => processor.finish(),
        }?;
        let output = self.restore_channel_mode(processed)?;
        let output = if let Some(vad) = &mut self.vad {
            vad.finish_input()?;
            vad.push_processed(&output)?;
            let output = vad.drain_ready()?;
            vad.finish_output()?;
            output
        } else {
            output
        };
        if self.channel_mode == ChannelMode::StereoLinked && !self.linked_original.is_empty() {
            return Err("linked streaming backend did not flush every input frame".into());
        }
        self.finished = true;
        Ok(output)
    }

    /// Start an independent stream while retaining any already-loaded model.
    pub fn reset(&mut self) -> Result<(), String> {
        match &mut self.processor {
            StreamingBackend::Classical(processor) => {
                let replacement =
                    StreamingDenoiser::new(self.denoiser.clone(), self.processor_channels)?;
                *processor = replacement;
            }
            #[cfg(feature = "rnnoise")]
            StreamingBackend::Rnnoise(processor) => processor.reset(),
            #[cfg(feature = "deepfilter")]
            StreamingBackend::DeepFilter(processor) => processor.reset()?,
            #[cfg(feature = "mossformer2")]
            StreamingBackend::Mossformer2(processor) => processor.reset(),
            #[cfg(feature = "gtcrn")]
            StreamingBackend::Gtcrn(processor) => processor.reset(),
            #[cfg(feature = "dpdfnet")]
            StreamingBackend::Dpdfnet(processor) => processor.reset(),
        }
        self.linked_original.clear();
        if let Some(vad) = &mut self.vad {
            vad.reset();
        }
        self.finished = false;
        Ok(())
    }

    fn build_processor(
        backend: Backend,
        sample_rate: u32,
        channels: usize,
        denoiser: &DenoiserConfig,
        backend_options: &BackendOptions,
        accelerator: AcceleratorSelection,
        construction: &StreamConstruction<'_>,
    ) -> Result<StreamingBackend, String> {
        let _ = (sample_rate, backend_options, accelerator);
        match backend {
            Backend::Classical => Ok(StreamingBackend::Classical(StreamingDenoiser::new(
                denoiser.clone(),
                channels,
            )?)),
            #[cfg(feature = "rnnoise")]
            Backend::Rnnoise => Ok(StreamingBackend::Rnnoise(Box::new(
                super::rnnoise::StreamingProcessor::new(sample_rate, channels)?,
            ))),
            #[cfg(feature = "deepfilter")]
            Backend::DeepFilter => Ok(StreamingBackend::DeepFilter(Box::new(
                super::deepfilter::StreamingProcessor::new(sample_rate, channels)?,
            ))),
            #[cfg(feature = "mossformer2")]
            Backend::Mossformer2 => {
                let model = backend_options.onnx.as_ref().ok_or_else(|| {
                    "MossFormer2 streaming requires the configured ONNX model".to_string()
                })?;
                Ok(StreamingBackend::Mossformer2(Box::new(
                    super::mossformer2::StreamingProcessor::new_with_accelerator(
                        model,
                        sample_rate,
                        channels,
                        accelerator.effective(),
                    )?,
                )))
            }
            #[cfg(feature = "gtcrn")]
            Backend::Gtcrn => {
                let model = backend_options
                    .onnx
                    .as_ref()
                    .ok_or_else(|| "GTCRN streaming requires the managed ONNX model".to_string())?;
                let processor = if let Some(prepared) = construction.gtcrn {
                    super::gtcrn::StreamingProcessor::new_with_model(
                        prepared,
                        sample_rate,
                        channels,
                    )?
                } else {
                    super::gtcrn::StreamingProcessor::new_with_accelerator(
                        model,
                        sample_rate,
                        channels,
                        accelerator.effective(),
                    )?
                };
                Ok(StreamingBackend::Gtcrn(Box::new(processor)))
            }
            #[cfg(feature = "dpdfnet")]
            Backend::Dpdfnet => {
                let model = backend_options.onnx.as_ref().ok_or_else(|| {
                    "DPDFNet streaming requires the managed ONNX model".to_string()
                })?;
                let processor = if let Some(prepared) = construction.dpdfnet {
                    super::dpdfnet::StreamingProcessor::new_with_model(
                        prepared,
                        sample_rate,
                        channels,
                    )?
                } else {
                    super::dpdfnet::StreamingProcessor::new_with_accelerator(
                        model,
                        sample_rate,
                        channels,
                        accelerator.effective(),
                    )?
                };
                Ok(StreamingBackend::Dpdfnet(Box::new(processor)))
            }
            #[allow(unreachable_patterns)]
            _ => Err("selected backend does not support stateful streaming".into()),
        }
    }

    pub(crate) fn process_block_with_limit(
        &mut self,
        channels: &[Vec<f64>],
        block_limit: usize,
    ) -> Result<Vec<Vec<f64>>, String> {
        validate_block(channels, self.input_channels)?;
        if block_limit == 0 {
            return Err("streaming backend block limit must be positive".into());
        }
        let frames = channels.first().map(Vec::len).unwrap_or(0);
        if frames <= block_limit {
            return self.process_bounded_block(channels);
        }

        let mut output = empty_channels(self.input_channels)?;
        let mut position = 0usize;
        while position < frames {
            let end = position.saturating_add(block_limit).min(frames);
            let block = clone_range(channels, position, end)?;
            let ready = self.process_bounded_block(&block)?;
            append_channels(&mut output, &ready, self.input_channels)?;
            position = end;
        }
        Ok(output)
    }

    fn process_bounded_block(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if let Some(vad) = &mut self.vad {
            vad.push_input(channels)?;
        }
        let processed = match self.channel_mode {
            ChannelMode::Independent => self.processor.process_block(channels),
            ChannelMode::StereoLinked => {
                let frames = channels[0].len();
                self.linked_original.try_reserve(frames).map_err(|_| {
                    ConfigError::allocation_failed("linked stream alignment").to_string()
                })?;
                let mut mid = Vec::new();
                mid.try_reserve_exact(frames)
                    .map_err(|_| ConfigError::allocation_failed("linked stream mid").to_string())?;
                for (&left, &right) in channels[0].iter().zip(&channels[1]) {
                    let left = crate::audio::sanitize_sample(left);
                    let right = crate::audio::sanitize_sample(right);
                    mid.push((left + right) * 0.5);
                    self.linked_original.push_back((left, right));
                }
                let mut input = Vec::new();
                input.try_reserve_exact(2).map_err(|_| {
                    ConfigError::allocation_failed("linked stream input channels").to_string()
                })?;
                input.push(mid);
                self.processor.process_owned_block(input)
            }
            ChannelMode::MidSide => {
                let (mid, side) = super::encode_mid_side(&channels[0], &channels[1])?;
                self.processor.process_owned_block(vec![mid, side])
            }
        }?;
        let processed = self.restore_channel_mode(processed)?;
        if let Some(vad) = &mut self.vad {
            vad.push_processed(&processed)?;
            vad.drain_ready()
        } else {
            Ok(processed)
        }
    }

    fn restore_channel_mode(
        &mut self,
        mut processed: Vec<Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>, String> {
        match self.channel_mode {
            ChannelMode::Independent => {
                validate_block(&processed, self.input_channels)?;
                Ok(processed)
            }
            ChannelMode::StereoLinked => {
                if processed.len() != 1 {
                    return Err("linked streaming backend must return one channel".into());
                }
                let left = processed.first_mut().ok_or_else(|| {
                    "linked streaming backend must return one channel".to_string()
                })?;
                if left.len() > self.linked_original.len() {
                    return Err("linked streaming backend returned unaligned frames".into());
                }
                let mut right = Vec::new();
                right.try_reserve_exact(left.len()).map_err(|_| {
                    ConfigError::allocation_failed("linked stream output").to_string()
                })?;
                for clean in left {
                    let (original_left, original_right) = self
                        .linked_original
                        .pop_front()
                        .ok_or_else(|| "linked streaming alignment queue underflow".to_string())?;
                    let original_mid = (original_left + original_right) * 0.5;
                    let correction = *clean - original_mid;
                    *clean = crate::audio::sanitize_sample(original_left + correction);
                    right.push(crate::audio::sanitize_sample(original_right + correction));
                }
                processed.try_reserve(1).map_err(|_| {
                    ConfigError::allocation_failed("linked stream output channels").to_string()
                })?;
                processed.push(right);
                Ok(processed)
            }
            ChannelMode::MidSide => {
                if processed.len() != 2 {
                    return Err("mid-side streaming backend must return two channels".into());
                }
                let (left, right) = super::decode_mid_side(&processed[0], &processed[1])?;
                Ok(vec![left, right])
            }
        }
    }
}

#[cfg(any(
    feature = "rnnoise",
    feature = "deepfilter",
    feature = "mossformer2",
    feature = "gtcrn",
    feature = "dpdfnet"
))]
fn resampler_pair_bytes(
    channels: usize,
    source_rate: u32,
    model_rate: u32,
    resource: &'static str,
) -> Result<u64, ConfigError> {
    let forward = crate::resample::resampler_plan_bytes(channels, source_rate, model_rate)
        .map_err(|_| ConfigError::invalid("sample_rate", "a bounded resampler plan"))?;
    let reverse = crate::resample::resampler_plan_bytes(channels, model_rate, source_rate)
        .map_err(|_| ConfigError::invalid("sample_rate", "a bounded resampler plan"))?;
    forward
        .checked_add(reverse)
        .ok_or(ConfigError::ResourceOverflow { resource })
}

fn validate_block(channels: &[Vec<f64>], expected_channels: usize) -> Result<usize, String> {
    if channels.len() != expected_channels {
        return Err(format!(
            "expected {expected_channels} streaming channels, got {}",
            channels.len()
        ));
    }
    let frames = channels.first().map(Vec::len).unwrap_or(0);
    if channels.iter().any(|channel| channel.len() != frames) {
        return Err("streaming blocks must have equal channel lengths".into());
    }
    Ok(frames)
}

fn empty_channels(channels: usize) -> Result<Vec<Vec<f64>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(channels)
        .map_err(|_| ConfigError::allocation_failed("stream output channels").to_string())?;
    for _ in 0..channels {
        output.push(Vec::new());
    }
    Ok(output)
}

fn clone_range(channels: &[Vec<f64>], start: usize, end: usize) -> Result<Vec<Vec<f64>>, String> {
    let frames = end.checked_sub(start).ok_or_else(|| {
        ConfigError::ResourceOverflow {
            resource: "stream split block",
        }
        .to_string()
    })?;
    let mut block = Vec::new();
    block
        .try_reserve_exact(channels.len())
        .map_err(|_| ConfigError::allocation_failed("stream split channels").to_string())?;
    for channel in channels {
        let mut split = Vec::new();
        split
            .try_reserve_exact(frames)
            .map_err(|_| ConfigError::allocation_failed("stream split samples").to_string())?;
        split.extend_from_slice(&channel[start..end]);
        block.push(split);
    }
    Ok(block)
}

fn append_channels(
    output: &mut [Vec<f64>],
    block: &[Vec<f64>],
    expected_channels: usize,
) -> Result<(), String> {
    validate_block(block, expected_channels)?;
    if output.len() != expected_channels {
        return Err("stream split output channel count changed".into());
    }
    for (output, block) in output.iter_mut().zip(block) {
        output
            .try_reserve_exact(block.len())
            .map_err(|_| ConfigError::allocation_failed("stream split output").to_string())?;
    }
    for (output, block) in output.iter_mut().zip(block) {
        output.extend_from_slice(block);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classical_stream_preserves_channel_geometry() {
        let mut config = DenoiserConfig::default(48_000);
        config.profile_ms = -1.0;
        let mut session = StreamingBackendSession::new(
            Backend::Classical,
            48_000,
            2,
            config,
            BackendOptions::default(),
        )
        .unwrap();
        let input = vec![vec![0.1; 2048], vec![-0.1; 2048]];
        let mut output = session.process_block(&input).unwrap();
        let tail = session.finish().unwrap();
        for (channel, tail) in output.iter_mut().zip(tail) {
            channel.extend(tail);
        }
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|channel| channel.len() == 2048));
    }

    #[test]
    fn stereo_linked_restore_reuses_enhanced_output_and_preserves_side() {
        let mut config = DenoiserConfig::default(48_000);
        config.profile_ms = -1.0;
        let options = BackendOptions {
            channel_mode: ChannelMode::StereoLinked,
            ..BackendOptions::default()
        };
        let mut session =
            StreamingBackendSession::new(Backend::Classical, 48_000, 2, config, options).unwrap();
        let original = [(0.25, -0.15), (-0.4, 0.2), (0.8, 0.3)];
        session.linked_original.extend(original);

        let enhanced = vec![0.2, -0.05, 0.4];
        let enhanced_ptr = enhanced.as_ptr();
        let mut enhanced_channels = Vec::with_capacity(2);
        enhanced_channels.push(enhanced);
        let channels_ptr = enhanced_channels.as_ptr();
        let output = session.restore_channel_mode(enhanced_channels).unwrap();

        assert_eq!(output.len(), 2);
        assert_eq!(output.as_ptr(), channels_ptr);
        assert_eq!(output[0].as_ptr(), enhanced_ptr);
        assert!(session.linked_original.is_empty());
        let expected = [[0.4, -0.35, 0.65], [0.0, 0.25, 0.15]];
        for (actual, expected) in output.iter().zip(expected) {
            assert_eq!(actual.len(), expected.len());
            for (&actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
        for frame in 0..original.len() {
            let original_side = original[frame].0 - original[frame].1;
            let restored_side = output[0][frame] - output[1][frame];
            assert!((restored_side - original_side).abs() < 1e-12);
        }
    }

    #[test]
    fn classical_streaming_vad_preserves_delayed_presentation_length() {
        let mut config = DenoiserConfig::default(48_000);
        config.profile_ms = -1.0;
        config.vad = true;
        let mut session = StreamingBackendSession::new(
            Backend::Classical,
            48_000,
            1,
            config,
            BackendOptions::default(),
        )
        .unwrap();
        let input = vec![vec![0.001; 4_321]];
        let mut output = Vec::new();
        for block in input[0].chunks(113) {
            output.extend(session.process_block(&[block.to_vec()]).unwrap().remove(0));
        }
        output.extend(session.finish().unwrap().remove(0));
        assert_eq!(output.len(), input[0].len());
        assert!(output.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn streaming_vad_resource_estimate_is_finite_and_profile_aware() {
        let without_profile =
            StreamingBackendSession::estimate_vad_additional_bytes(48_000, 2, 8_192, 2_048, -1.0)
                .unwrap();
        let with_profile = StreamingBackendSession::estimate_vad_additional_bytes(
            48_000, 2, 8_192, 2_048, 1_000.0,
        )
        .unwrap();
        assert!(without_profile > 0);
        assert!(with_profile > without_profile);
    }

    #[cfg(feature = "deepfilter")]
    #[test]
    fn deepfilter_build_keeps_public_streaming_sessions_sendable() {
        fn assert_send<T: Send>() {}
        assert_send::<StreamingBackendSession>();
    }

    #[cfg(feature = "dpdfnet")]
    #[test]
    fn dpdfnet_build_keeps_public_streaming_sessions_sendable() {
        fn assert_send<T: Send>() {}
        assert_send::<StreamingBackendSession>();
    }

    #[test]
    fn unsupported_compiled_backend_is_rejected() {
        #[cfg(feature = "onnx")]
        assert!(!StreamingBackendSession::supports(Backend::Onnx));
    }
}
