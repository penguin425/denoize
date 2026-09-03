//! Official stateful adapter for the 48 kHz DPDFNet HR graph family.
//!
//! The production backend accepts the authenticated DPDFNet-2 profile. The
//! model-level API also recognizes the upstream DPDFNet-8 state geometry so
//! the pinned comparison harness can continue to evaluate both profiles.

use super::tract_runtime::SharedRunnable;
use super::OnnxModelConfig;
use crate::AcceleratorRuntime;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::Arc;
use tract_onnx::prelude::*;

pub const SAMPLE_RATE: u32 = 48_000;
pub const FFT_SIZE: usize = 960;
pub const HOP_SIZE: usize = 480;
pub const BINS: usize = 481;
pub const DPDFNET2_STATE_SIZE: usize = 56_436;
pub const DPDFNET8_STATE_SIZE: usize = 90_228;
/// Backwards-compatible name for the DPDFNet-2 state size.
pub const STATE_SIZE: usize = DPDFNET2_STATE_SIZE;
pub const ERB_NORM_STATE_SIZE: usize = 481;
pub const SPEC_NORM_STATE_SIZE: usize = 96;
const COMPILED_MODEL_ALLOWANCE_BYTES: u64 = 128 * 1024 * 1024;

/// The official 48 kHz exporter defaults to two convolution-lookahead and two
/// deep-filter-lookahead frames. Its offline path advances the reconstruction
/// by four hops. The causal stream itself deliberately leaves this delay in.
pub const MODEL_LOOKAHEAD_HOPS: usize = 4;
pub const MODEL_LOOKAHEAD_SAMPLES: usize = MODEL_LOOKAHEAD_HOPS * HOP_SIZE;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DpdfnetMetadata {
    pub profile: String,
    pub sample_rate: u32,
    pub fft_size: usize,
    pub hop_size: usize,
    pub bins: usize,
    pub state_size: usize,
    pub erb_norm_state_size: usize,
    pub spec_norm_state_size: usize,
}

#[derive(Clone)]
pub struct DpdfnetModel {
    model: SharedRunnable,
    initial_state: Arc<Tensor>,
    metadata: DpdfnetMetadata,
    runtime: AcceleratorRuntime,
}

impl DpdfnetModel {
    pub fn load(config: &OnnxModelConfig) -> Result<Self, String> {
        Self::load_with_accelerator(config, AcceleratorRuntime::Cpu)
    }

    pub fn load_with_accelerator(
        config: &OnnxModelConfig,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        validate_config(config)?;
        let (metadata, initial_state) = read_metadata(&config.path)?;
        let model = load_model(&config.path, runtime, metadata.state_size)?;
        let initial_state = Tensor::from_shape(&[initial_state.len()], &initial_state)
            .map_err(tract_error)?
            .into_arc_tensor();
        Ok(Self {
            model,
            initial_state,
            metadata,
            runtime,
        })
    }

    /// Load only the DPDFNet-2 48 kHz HR geometry used by the managed backend.
    pub fn load_dpdfnet2_with_accelerator(
        config: &OnnxModelConfig,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        let model = Self::load_with_accelerator(config, runtime)?;
        model.require_dpdfnet2()?;
        Ok(model)
    }

    fn require_dpdfnet2(&self) -> Result<(), String> {
        if self.metadata.state_size != DPDFNET2_STATE_SIZE {
            return Err(format!(
                "managed DPDFNet backend requires the DPDFNet-2 {DPDFNET2_STATE_SIZE}-scalar state, got {} scalars",
                self.metadata.state_size
            ));
        }
        Ok(())
    }

    pub const fn runtime(&self) -> AcceleratorRuntime {
        self.runtime
    }

    pub fn metadata(&self) -> &DpdfnetMetadata {
        &self.metadata
    }

    pub fn stream(&self) -> Result<DpdfnetStream, String> {
        DpdfnetStream::from_model(Arc::clone(&self.model), Arc::clone(&self.initial_state))
    }

    /// Process finite planar audio through the causal stream. The result has
    /// the input geometry, but retains DPDFNet's model lookahead delay.
    pub fn process(
        &self,
        channels: &[Vec<f64>],
        input_sample_rate: u32,
    ) -> Result<Vec<Vec<f64>>, String> {
        channels
            .iter()
            .map(|channel| process_channel(channel, input_sample_rate, self))
            .collect()
    }

    /// Process finite planar audio with the production stream contract.
    ///
    /// Unlike [`process`](Self::process), this compensates the authenticated
    /// model's four-hop content offset and returns exactly the input geometry.
    pub fn process_aligned(
        &self,
        channels: &[Vec<f64>],
        input_sample_rate: u32,
    ) -> Result<Vec<Vec<f64>>, String> {
        self.require_dpdfnet2()?;
        let mut stream =
            StreamingProcessor::new_with_model(self, input_sample_rate, channels.len())?;
        let mut output = stream.process_block(channels)?;
        let tail = stream.finish()?;
        for (channel, tail) in output.iter_mut().zip(tail) {
            channel.extend(tail);
        }
        Ok(output)
    }
}

/// Stateful native-rate DPDFNet processor.
///
/// Calls consume one 10 ms hop. The first call buffers half a window and
/// returns `None`; every later call returns one enhanced hop. [`flush`](Self::flush)
/// emits the final overlap-add hop once.
pub struct DpdfnetStream {
    model: SharedRunnable,
    reuse_runtime_state: bool,
    initial_state: Arc<Tensor>,
    state: Arc<Tensor>,
    analysis: [f32; FFT_SIZE],
    overlap: [f32; FFT_SIZE],
    window: [f32; FFT_SIZE],
    spectrum: Vec<Complex32>,
    model_input: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    primed: bool,
}

impl DpdfnetStream {
    pub fn open(path: &Path) -> Result<Self, String> {
        let model = DpdfnetModel::load(&OnnxModelConfig {
            path: path.to_path_buf(),
            sample_rate: SAMPLE_RATE,
        })?;
        model.stream()
    }

    fn from_model(model: SharedRunnable, initial_state: Arc<Tensor>) -> Result<Self, String> {
        let mut planner = FftPlanner::new();
        let state = initial_state.as_ref().clone().into_arc_tensor();
        let reuse_runtime_state = super::tract_runtime::supports_state_reuse(&model);
        Ok(Self {
            model,
            reuse_runtime_state,
            initial_state,
            state,
            analysis: [0.0; FFT_SIZE],
            overlap: [0.0; FFT_SIZE],
            window: vorbis_window(),
            spectrum: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
            model_input: vec![0.0; BINS * 2],
            fft: planner.plan_fft_forward(FFT_SIZE),
            ifft: planner.plan_fft_inverse(FFT_SIZE),
            primed: false,
        })
    }

    pub fn reset(&mut self) {
        self.state = self.initial_state.as_ref().clone().into_arc_tensor();
        self.analysis.fill(0.0);
        self.overlap.fill(0.0);
        self.primed = false;
    }

    pub fn process_hop(
        &mut self,
        input: &[f32; HOP_SIZE],
    ) -> Result<Option<[f32; HOP_SIZE]>, String> {
        if !self.primed {
            self.analysis[..HOP_SIZE].copy_from_slice(input);
            self.primed = true;
            return Ok(None);
        }
        self.analysis[HOP_SIZE..].copy_from_slice(input);
        let output = self.process_frame()?;
        self.analysis.copy_within(HOP_SIZE.., 0);
        Ok(Some(output))
    }

    pub fn flush(&mut self) -> Result<Option<[f32; HOP_SIZE]>, String> {
        if !self.primed {
            return Ok(None);
        }
        self.analysis[HOP_SIZE..].fill(0.0);
        let output = self.process_frame()?;
        self.analysis.fill(0.0);
        self.primed = false;
        Ok(Some(output))
    }

    fn process_frame(&mut self) -> Result<[f32; HOP_SIZE], String> {
        for (index, (sample, window)) in self.analysis.iter().zip(&self.window).enumerate() {
            self.spectrum[index] = Complex32::new(sample * window, 0.0);
        }
        self.fft.process(&mut self.spectrum);

        for (bin, value) in self.spectrum.iter().take(BINS).enumerate() {
            self.model_input[bin * 2] = value.re;
            self.model_input[bin * 2 + 1] = value.im;
        }
        let enhanced = self.infer()?;
        let enhanced_plain = enhanced.try_as_plain().map_err(tract_error)?;
        let enhanced = enhanced_plain.as_slice::<f32>().map_err(tract_error)?;
        if enhanced.len() != BINS * 2 {
            return Err(format!(
                "DPDFNet returned {} spectrum scalars, expected {}",
                enhanced.len(),
                BINS * 2
            ));
        }
        for bin in 0..BINS {
            self.spectrum[bin] = Complex32::new(enhanced[bin * 2], enhanced[bin * 2 + 1]);
        }
        for bin in BINS..FFT_SIZE {
            self.spectrum[bin] = self.spectrum[FFT_SIZE - bin].conj();
        }
        self.spectrum[0].im = 0.0;
        self.spectrum[BINS - 1].im = 0.0;

        self.ifft.process(&mut self.spectrum);
        for (index, value) in self.spectrum.iter().enumerate() {
            self.overlap[index] += value.re * self.window[index] / FFT_SIZE as f32;
        }
        let output = std::array::from_fn(|index| self.overlap[index]);
        self.overlap.copy_within(HOP_SIZE.., 0);
        self.overlap[FFT_SIZE - HOP_SIZE..].fill(0.0);
        Ok(output)
    }

    fn infer(&mut self) -> Result<TValue, String> {
        let state = std::mem::replace(&mut self.state, Arc::clone(&self.initial_state));
        let inputs = tvec!(
            Tensor::from_shape(&[1, 1, BINS, 2], &self.model_input)
                .map_err(tract_error)?
                .into_tvalue(),
            state.into_tvalue(),
        );
        let mut outputs = if self.reuse_runtime_state {
            super::tract_runtime::run_reusing_state(&self.model, inputs)
        } else {
            self.model.run(inputs)
        }
        .map_err(tract_error)?;
        if outputs.len() != 2 {
            return Err(format!(
                "DPDFNet returned {} outputs, expected 2",
                outputs.len()
            ));
        }
        let next_state = outputs
            .pop()
            .expect("DPDFNet output count was validated above");
        {
            let next_state_plain = next_state.try_as_plain().map_err(tract_error)?;
            let next_state = next_state_plain.as_slice::<f32>().map_err(tract_error)?;
            if next_state.len() != self.state.len() {
                return Err(format!(
                    "DPDFNet returned {} state scalars, expected {}",
                    next_state.len(),
                    self.state.len()
                ));
            }
        }
        self.state = next_state.into_arc_tensor();
        Ok(outputs.remove(0))
    }
}

fn process_channel(
    input: &[f64],
    input_sample_rate: u32,
    model: &DpdfnetModel,
) -> Result<Vec<f64>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let at_model_rate = crate::resample::resample(input, input_sample_rate, SAMPLE_RATE)?;
    let mut stream = model.stream()?;
    let mut enhanced = Vec::with_capacity(at_model_rate.len() + HOP_SIZE);
    for chunk in at_model_rate.chunks(HOP_SIZE) {
        let mut hop = [0.0; HOP_SIZE];
        for (output, input) in hop.iter_mut().zip(chunk) {
            *output = crate::audio::sanitize_sample(*input) as f32;
        }
        if let Some(output) = stream.process_hop(&hop)? {
            enhanced.extend(output);
        }
    }
    if let Some(output) = stream.flush()? {
        enhanced.extend(output);
    }
    enhanced.truncate(at_model_rate.len());
    enhanced.resize(at_model_rate.len(), 0.0);
    let model_output: Vec<f64> = enhanced.into_iter().map(f64::from).collect();
    let mut output = crate::resample::resample(&model_output, SAMPLE_RATE, input_sample_rate)?;
    output.truncate(input.len());
    output.resize(input.len(), 0.0);
    Ok(output)
}

/// Continuous, channel-planar DPDFNet-2 processing at an arbitrary input rate.
///
/// The optimized graph is shared while recurrent, WOLA, converter, and partial
/// hop state remain independent per channel. The four-hop model offset is
/// removed before samples leave this adapter, and [`finish`](Self::finish)
/// returns the exact remaining number of input-rate frames.
pub(crate) struct StreamingProcessor {
    channels: usize,
    native_rate: bool,
    to_model_rate: crate::resample::StreamingResampler,
    from_model_rate: crate::resample::StreamingResampler,
    streams: Vec<DpdfnetStream>,
    pending_model_rate: Vec<VecDeque<f64>>,
    hop_scratch: Vec<[f32; HOP_SIZE]>,
    enhanced_scratch: Vec<Option<[f32; HOP_SIZE]>>,
    discard_model_frames: usize,
    model_input_frames: usize,
    model_output_frames: usize,
    input_frames: usize,
    output_frames: usize,
    finished: bool,
}

impl StreamingProcessor {
    #[cfg(test)]
    pub(crate) fn new(
        config: &OnnxModelConfig,
        sample_rate: u32,
        channels: usize,
    ) -> Result<Self, String> {
        let model = DpdfnetModel::load_dpdfnet2_with_accelerator(config, AcceleratorRuntime::Cpu)?;
        Self::new_with_model(&model, sample_rate, channels)
    }

    pub(crate) fn new_with_accelerator(
        config: &OnnxModelConfig,
        sample_rate: u32,
        channels: usize,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        let model = DpdfnetModel::load_dpdfnet2_with_accelerator(config, runtime)?;
        Self::new_with_model(&model, sample_rate, channels)
    }

    pub(crate) fn new_with_model(
        model: &DpdfnetModel,
        sample_rate: u32,
        channels: usize,
    ) -> Result<Self, String> {
        model.require_dpdfnet2()?;
        if channels == 0 || channels > crate::config::MAX_STREAM_CHANNELS {
            return Err(format!(
                "DPDFNet streaming channels must be between 1 and {}",
                crate::config::MAX_STREAM_CHANNELS
            ));
        }
        let to_model_rate =
            crate::resample::StreamingResampler::new(channels, sample_rate, SAMPLE_RATE)?;
        let from_model_rate =
            crate::resample::StreamingResampler::new(channels, SAMPLE_RATE, sample_rate)?;
        let mut streams = Vec::new();
        streams
            .try_reserve_exact(channels)
            .map_err(|_| "unable to reserve DPDFNet channel streams".to_string())?;
        let mut pending_model_rate = Vec::new();
        pending_model_rate
            .try_reserve_exact(channels)
            .map_err(|_| "unable to reserve DPDFNet pending channels".to_string())?;
        for _ in 0..channels {
            streams.push(model.stream()?);
            let mut pending = VecDeque::new();
            pending
                .try_reserve(HOP_SIZE)
                .map_err(|_| "unable to reserve DPDFNet pending samples".to_string())?;
            pending_model_rate.push(pending);
        }
        let mut hop_scratch = Vec::new();
        hop_scratch
            .try_reserve_exact(channels)
            .map_err(|_| "unable to reserve DPDFNet input-hop scratch".to_string())?;
        hop_scratch.resize(channels, [0.0; HOP_SIZE]);
        let mut enhanced_scratch = Vec::new();
        enhanced_scratch
            .try_reserve_exact(channels)
            .map_err(|_| "unable to reserve DPDFNet output-hop scratch".to_string())?;
        enhanced_scratch.resize(channels, None);
        Ok(Self {
            channels,
            native_rate: sample_rate == SAMPLE_RATE,
            to_model_rate,
            from_model_rate,
            streams,
            pending_model_rate,
            hop_scratch,
            enhanced_scratch,
            discard_model_frames: MODEL_LOOKAHEAD_SAMPLES,
            model_input_frames: 0,
            model_output_frames: 0,
            input_frames: 0,
            output_frames: 0,
            finished: false,
        })
    }

    pub(crate) fn process_block(&mut self, input: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if self.finished {
            return Err("DPDFNet stream is finished; reset it before processing more input".into());
        }
        let frames = validate_stream_block(input, self.channels)?;
        let input_frames = self
            .input_frames
            .checked_add(frames)
            .ok_or_else(|| "DPDFNet streaming input length overflow".to_string())?;
        let converted;
        let at_model_rate = if self.native_rate {
            input
        } else {
            converted = self.to_model_rate.process(input)?;
            &converted
        };
        let enhanced_model_rate = self.process_model_rate(at_model_rate)?;
        let output = if self.native_rate {
            enhanced_model_rate
        } else {
            self.from_model_rate.process(&enhanced_model_rate)?
        };
        let produced = validate_stream_block(&output, self.channels)?;
        let output_frames = self
            .output_frames
            .checked_add(produced)
            .ok_or_else(|| "DPDFNet streaming output length overflow".to_string())?;
        if output_frames > input_frames {
            return Err("DPDFNet stream produced samples ahead of its input clock".into());
        }
        self.input_frames = input_frames;
        self.output_frames = output_frames;
        Ok(output)
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<Vec<f64>>, String> {
        let remaining = self
            .input_frames
            .checked_sub(self.output_frames)
            .ok_or_else(|| "DPDFNet stream exceeded its input clock".to_string())?;
        let mut output = empty_output(self.channels, remaining)?;
        if self.finished {
            return Ok(output);
        }

        if self.native_rate {
            let enhanced = self.finish_model_rate()?;
            append_limited(&mut output, &enhanced, remaining)?;
        } else {
            let model_input_tail = self.to_model_rate.finish()?;
            let enhanced = self.process_model_rate(&model_input_tail)?;
            let converted = self.from_model_rate.process(&enhanced)?;
            append_limited(&mut output, &converted, remaining)?;

            let enhanced = self.finish_model_rate()?;
            let converted = self.from_model_rate.process(&enhanced)?;
            append_limited(&mut output, &converted, remaining)?;

            let converted = self.from_model_rate.finish()?;
            append_limited(&mut output, &converted, remaining)?;
        }
        if output.first().map_or(0, Vec::len) < remaining {
            for channel in &mut output {
                channel.resize(remaining, 0.0);
            }
        }
        self.output_frames = self.input_frames;
        self.finished = true;
        Ok(output)
    }

    pub(crate) fn reset(&mut self) {
        self.to_model_rate.reset();
        self.from_model_rate.reset();
        for stream in &mut self.streams {
            stream.reset();
        }
        for pending in &mut self.pending_model_rate {
            pending.clear();
        }
        self.discard_model_frames = MODEL_LOOKAHEAD_SAMPLES;
        self.model_input_frames = 0;
        self.model_output_frames = 0;
        self.input_frames = 0;
        self.output_frames = 0;
        self.finished = false;
    }

    fn process_model_rate(&mut self, input: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let frames = validate_stream_block(input, self.channels)?;
        self.model_input_frames = self
            .model_input_frames
            .checked_add(frames)
            .ok_or_else(|| "DPDFNet model input length overflow".to_string())?;
        for (pending, channel) in self.pending_model_rate.iter_mut().zip(input) {
            pending
                .try_reserve(channel.len())
                .map_err(|_| "unable to grow DPDFNet pending input".to_string())?;
            pending.extend(channel.iter().copied());
        }
        let reserve = self
            .model_input_frames
            .checked_sub(self.model_output_frames)
            .ok_or_else(|| "DPDFNet model output exceeded its input clock".to_string())?;
        let mut output = empty_output(self.channels, reserve)?;
        while self
            .pending_model_rate
            .first()
            .is_some_and(|pending| pending.len() >= HOP_SIZE)
        {
            self.take_pending_hop()?;
            self.process_scratch_hop(&mut output)?;
        }
        Ok(output)
    }

    fn finish_model_rate(&mut self) -> Result<Vec<Vec<f64>>, String> {
        let remaining = self
            .model_input_frames
            .checked_sub(self.model_output_frames)
            .ok_or_else(|| "DPDFNet model exceeded its input clock".to_string())?;
        let mut output = empty_output(self.channels, remaining)?;
        let pending = self.pending_model_rate.first().map_or(0, VecDeque::len);
        if self
            .pending_model_rate
            .iter()
            .any(|channel| channel.len() != pending)
        {
            return Err("DPDFNet pending channels became unaligned".into());
        }
        if pending > 0 {
            for channel in &mut self.pending_model_rate {
                channel.resize(HOP_SIZE, 0.0);
            }
            self.take_pending_hop()?;
            self.process_scratch_hop(&mut output)?;
        }
        self.hop_scratch.fill([0.0; HOP_SIZE]);
        for _ in 0..MODEL_LOOKAHEAD_HOPS {
            self.process_scratch_hop(&mut output)?;
        }
        self.flush_streams(&mut output)?;
        if output.first().map_or(0, Vec::len) < remaining {
            for channel in &mut output {
                channel.resize(remaining, 0.0);
            }
        }
        self.model_output_frames = self.model_input_frames;
        Ok(output)
    }

    fn take_pending_hop(&mut self) -> Result<(), String> {
        for (pending, hop) in self
            .pending_model_rate
            .iter_mut()
            .zip(&mut self.hop_scratch)
        {
            if pending.len() < HOP_SIZE {
                return Err("DPDFNet pending input underflow".into());
            }
            for destination in hop {
                let source = pending
                    .pop_front()
                    .ok_or_else(|| "DPDFNet pending input underflow".to_string())?;
                *destination = crate::audio::sanitize_sample(source) as f32;
            }
        }
        Ok(())
    }

    fn process_scratch_hop(&mut self, output: &mut [Vec<f64>]) -> Result<(), String> {
        for ((stream, hop), enhanced) in self
            .streams
            .iter_mut()
            .zip(&self.hop_scratch)
            .zip(&mut self.enhanced_scratch)
        {
            *enhanced = stream.process_hop(hop)?;
        }
        self.append_scratch_hop(output)
    }

    fn flush_streams(&mut self, output: &mut [Vec<f64>]) -> Result<(), String> {
        for (stream, enhanced) in self.streams.iter_mut().zip(&mut self.enhanced_scratch) {
            *enhanced = stream.flush()?;
        }
        self.append_scratch_hop(output)
    }

    fn append_scratch_hop(&mut self, output: &mut [Vec<f64>]) -> Result<(), String> {
        let produced = self.enhanced_scratch.first().is_some_and(Option::is_some);
        if self
            .enhanced_scratch
            .iter()
            .any(|channel| channel.is_some() != produced)
        {
            return Err("DPDFNet channel streams became misaligned".into());
        }
        if produced {
            self.append_model_hop(output)?;
        }
        Ok(())
    }

    fn append_model_hop(&mut self, output: &mut [Vec<f64>]) -> Result<(), String> {
        let skip = self.discard_model_frames.min(HOP_SIZE);
        let available = HOP_SIZE - skip;
        let remaining = self
            .model_input_frames
            .checked_sub(self.model_output_frames)
            .ok_or_else(|| "DPDFNet model output exceeded its input clock".to_string())?;
        let retained = available.min(remaining);
        for (destination, channel) in output.iter_mut().zip(&self.enhanced_scratch) {
            let channel = channel
                .as_ref()
                .ok_or_else(|| "DPDFNet enhanced hop is missing".to_string())?;
            destination.extend(
                channel[skip..skip + retained]
                    .iter()
                    .map(|sample| f64::from(*sample)),
            );
        }
        self.discard_model_frames = self
            .discard_model_frames
            .checked_sub(skip)
            .ok_or_else(|| "DPDFNet latency accounting underflow".to_string())?;
        self.model_output_frames = self
            .model_output_frames
            .checked_add(retained)
            .ok_or_else(|| "DPDFNet model output length overflow".to_string())?;
        Ok(())
    }
}

fn validate_stream_block(input: &[Vec<f64>], channels: usize) -> Result<usize, String> {
    if input.len() != channels {
        return Err(format!(
            "DPDFNet stream expected {channels} channels, received {}",
            input.len()
        ));
    }
    let frames = input.first().map_or(0, Vec::len);
    if input.iter().any(|channel| channel.len() != frames) {
        return Err("DPDFNet stream channels must contain the same number of frames".into());
    }
    Ok(frames)
}

fn empty_output(channels: usize, capacity: usize) -> Result<Vec<Vec<f64>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(channels)
        .map_err(|_| "unable to reserve DPDFNet output channels".to_string())?;
    for _ in 0..channels {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(capacity)
            .map_err(|_| "unable to reserve DPDFNet output samples".to_string())?;
        output.push(channel);
    }
    Ok(output)
}

fn append_limited(
    destination: &mut [Vec<f64>],
    source: &[Vec<f64>],
    frame_limit: usize,
) -> Result<(), String> {
    if source.len() != destination.len() {
        return Err("DPDFNet stream produced an invalid channel count".into());
    }
    let destination_frames = destination.first().map_or(0, Vec::len);
    let source_frames = validate_stream_block(source, destination.len())?;
    if destination
        .iter()
        .any(|channel| channel.len() != destination_frames)
    {
        return Err("DPDFNet destination channels became unaligned".into());
    }
    let retained = frame_limit
        .checked_sub(destination_frames)
        .ok_or_else(|| "DPDFNet streaming output exceeded its target".to_string())?
        .min(source_frames);
    for (destination, source) in destination.iter_mut().zip(source) {
        destination
            .try_reserve_exact(retained)
            .map_err(|_| "unable to grow DPDFNet output".to_string())?;
        destination.extend_from_slice(&source[..retained]);
    }
    Ok(())
}

pub(crate) fn streaming_state_bytes(channels: usize) -> Result<u64, crate::ConfigError> {
    let per_channel_scalars = DPDFNET2_STATE_SIZE
        .checked_add(FFT_SIZE * 5)
        .and_then(|value| value.checked_add(BINS * 2))
        .and_then(|value| value.checked_add(HOP_SIZE * 2 + 1))
        .ok_or(crate::ConfigError::ResourceOverflow {
            resource: "DPDFNet stream state",
        })?;
    let per_channel_bytes = u64::try_from(per_channel_scalars)
        .ok()
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
        .and_then(|value| value.checked_add((std::mem::size_of::<Vec<f32>>() * 7) as u64))
        .ok_or(crate::ConfigError::ResourceOverflow {
            resource: "DPDFNet stream state",
        })?;
    let channel_bytes = per_channel_bytes
        .checked_mul(
            u64::try_from(channels).map_err(|_| crate::ConfigError::ResourceOverflow {
                resource: "DPDFNet stream state",
            })?,
        )
        .ok_or(crate::ConfigError::ResourceOverflow {
            resource: "DPDFNet stream state",
        })?;
    COMPILED_MODEL_ALLOWANCE_BYTES
        .checked_add(channel_bytes)
        .ok_or(crate::ConfigError::ResourceOverflow {
            resource: "DPDFNet stream state",
        })
}

fn validate_config(config: &OnnxModelConfig) -> Result<(), String> {
    if config.sample_rate != SAMPLE_RATE {
        return Err(format!(
            "DPDFNet 48 kHz expects a {SAMPLE_RATE} Hz model, got {} Hz",
            config.sample_rate
        ));
    }
    if !config.path.is_file() {
        return Err(format!(
            "DPDFNet ONNX model does not exist or is not a file: {}",
            config.path.display()
        ));
    }
    Ok(())
}

fn read_metadata(path: &Path) -> Result<(DpdfnetMetadata, Vec<f32>), String> {
    let proto = tract_onnx::onnx()
        .proto_model_for_path(path)
        .map_err(|error| {
            format!(
                "failed to read DPDFNet metadata {}: {error}",
                path.display()
            )
        })?;
    let properties: BTreeMap<String, String> = proto
        .metadata_props
        .into_iter()
        .map(|entry| (entry.key, entry.value))
        .collect();
    parse_metadata(&properties)
}

fn parse_metadata(
    properties: &BTreeMap<String, String>,
) -> Result<(DpdfnetMetadata, Vec<f32>), String> {
    let required = |key: &str| {
        properties
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("DPDFNet model is missing required metadata key `{key}`"))
    };
    let parse_usize = |key: &str| -> Result<usize, String> {
        required(key)?
            .parse()
            .map_err(|_| format!("DPDFNet metadata `{key}` is not an unsigned integer"))
    };
    let model_type = required("model_type")?;
    if model_type != "dpdfnet" {
        return Err(format!(
            "DPDFNet metadata `model_type` must be `dpdfnet`, got `{model_type}`"
        ));
    }
    let profile = required("profile")?.to_owned();
    if profile != "dpdfnet2_48khz_hr" {
        return Err(format!(
            "DPDFNet requires profile `dpdfnet2_48khz_hr`, got `{profile}`"
        ));
    }
    if required("window_type")? != "vorbis" {
        return Err("DPDFNet requires the `vorbis` analysis window".into());
    }
    for (key, expected) in [
        ("version", "1"),
        ("normalized", "0"),
        ("window_length", "960"),
        ("erb_bins", "32"),
        ("spec_bins", "96"),
    ] {
        let actual = required(key)?;
        if actual != expected {
            return Err(format!(
                "DPDFNet metadata `{key}` must be `{expected}`, got `{actual}`"
            ));
        }
    }

    let sample_rate = u32::try_from(parse_usize("sample_rate")?)
        .map_err(|_| "DPDFNet metadata `sample_rate` exceeds u32".to_string())?;

    let metadata = DpdfnetMetadata {
        profile,
        sample_rate,
        fft_size: parse_usize("n_fft")?,
        hop_size: parse_usize("hop_length")?,
        bins: parse_usize("freq_bins")?,
        state_size: parse_usize("state_size")?,
        erb_norm_state_size: parse_usize("erb_norm_state_size")?,
        spec_norm_state_size: parse_usize("spec_norm_state_size")?,
    };
    // The pinned upstream DPDFNet-8 HR export currently carries the same
    // `dpdfnet2_48khz_hr` profile string as DPDFNet-2.  The state geometry is
    // therefore part of the authenticated model contract and is the only
    // reliable way to distinguish these two official exports here.
    let common_contract_matches = metadata.profile == "dpdfnet2_48khz_hr"
        && metadata.sample_rate == SAMPLE_RATE
        && metadata.fft_size == FFT_SIZE
        && metadata.hop_size == HOP_SIZE
        && metadata.bins == BINS
        && metadata.erb_norm_state_size == ERB_NORM_STATE_SIZE
        && metadata.spec_norm_state_size == SPEC_NORM_STATE_SIZE;
    if !common_contract_matches
        || !matches!(
            metadata.state_size,
            DPDFNET2_STATE_SIZE | DPDFNET8_STATE_SIZE
        )
    {
        return Err(format!(
            "DPDFNet model contract mismatch: got {metadata:?}; expected the official 48 kHz HR geometry with a {DPDFNET2_STATE_SIZE}-scalar DPDFNet-2 or {DPDFNET8_STATE_SIZE}-scalar DPDFNet-8 state"
        ));
    }

    let erb = parse_float_list(required("erb_norm_init")?, "erb_norm_init")?;
    let spec = parse_float_list(required("spec_norm_init")?, "spec_norm_init")?;
    if erb.len() != ERB_NORM_STATE_SIZE {
        return Err(format!(
            "DPDFNet `erb_norm_init` has {} values, expected {ERB_NORM_STATE_SIZE}",
            erb.len()
        ));
    }
    if spec.len() != SPEC_NORM_STATE_SIZE {
        return Err(format!(
            "DPDFNet `spec_norm_init` has {} values, expected {SPEC_NORM_STATE_SIZE}",
            spec.len()
        ));
    }
    let mut initial_state = vec![0.0; metadata.state_size];
    initial_state[..ERB_NORM_STATE_SIZE].copy_from_slice(&erb);
    initial_state[ERB_NORM_STATE_SIZE..ERB_NORM_STATE_SIZE + SPEC_NORM_STATE_SIZE]
        .copy_from_slice(&spec);
    Ok((metadata, initial_state))
}

fn parse_float_list(value: &str, key: &str) -> Result<Vec<f32>, String> {
    value
        .split(',')
        .map(|item| {
            let value = item
                .parse::<f32>()
                .map_err(|_| format!("DPDFNet metadata `{key}` contains an invalid float"))?;
            if !value.is_finite() {
                return Err(format!(
                    "DPDFNet metadata `{key}` contains a non-finite float"
                ));
            }
            Ok(value)
        })
        .collect()
}

fn vorbis_window() -> [f32; FFT_SIZE] {
    std::array::from_fn(|index| {
        let phase = std::f32::consts::PI * (index as f32 + 0.5) / FFT_SIZE as f32;
        (0.5 * std::f32::consts::PI * phase.sin().powi(2)).sin()
    })
}

fn load_model(
    path: &Path,
    runtime: AcceleratorRuntime,
    state_size: usize,
) -> Result<SharedRunnable, String> {
    let mut model = tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|error| format!("failed to load DPDFNet model {}: {error}", path.display()))?;
    let state_shape = [state_size];
    let input_shapes: [&[usize]; 2] = [&[1, 1, BINS, 2], &state_shape];
    let output_shapes = input_shapes;
    for (index, shape) in input_shapes.iter().enumerate() {
        model
            .set_input_fact(index, f32::fact(*shape).into())
            .map_err(tract_error)?;
    }
    for (index, shape) in output_shapes.iter().enumerate() {
        model
            .set_output_fact(index, f32::fact(*shape).into())
            .map_err(tract_error)?;
    }
    let model = model.into_typed().map_err(tract_error)?;
    super::tract_runtime::prepare(model, runtime, "DPDFNet 48 kHz model")
}

fn tract_error(error: impl std::fmt::Display) -> String {
    format!("DPDFNet inference failed: {error:#}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use tract_onnx::pb::{
        tensor_proto, tensor_shape_proto, type_proto, GraphProto, ModelProto, NodeProto,
        OperatorSetIdProto, StringStringEntryProto, TensorShapeProto, TypeProto, ValueInfoProto,
    };

    fn valid_metadata() -> BTreeMap<String, String> {
        let mut metadata: BTreeMap<String, String> = [
            ("model_type", "dpdfnet"),
            ("version", "1"),
            ("profile", "dpdfnet2_48khz_hr"),
            ("sample_rate", "48000"),
            ("n_fft", "960"),
            ("hop_length", "480"),
            ("window_length", "960"),
            ("window_type", "vorbis"),
            ("normalized", "0"),
            ("freq_bins", "481"),
            ("erb_bins", "32"),
            ("spec_bins", "96"),
            ("state_size", "56436"),
            ("erb_norm_state_size", "481"),
            ("spec_norm_state_size", "96"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
        metadata.insert(
            "erb_norm_init".into(),
            std::iter::repeat_n("-1", ERB_NORM_STATE_SIZE)
                .collect::<Vec<_>>()
                .join(","),
        );
        metadata.insert(
            "spec_norm_init".into(),
            std::iter::repeat_n("-2", SPEC_NORM_STATE_SIZE)
                .collect::<Vec<_>>()
                .join(","),
        );
        metadata
    }

    #[test]
    fn vorbis_window_is_squared_cola_at_half_overlap() {
        let window = vorbis_window();
        for index in 0..HOP_SIZE {
            let sum = window[index].powi(2) + window[index + HOP_SIZE].powi(2);
            assert!((sum - 1.0).abs() < 2.0e-6, "index={index}, sum={sum}");
        }
    }

    #[test]
    fn rejects_wrong_rate_before_loading() {
        let config = OnnxModelConfig {
            path: "missing.onnx".into(),
            sample_rate: 16_000,
        };
        assert!(validate_config(&config).unwrap_err().contains("48000 Hz"));
    }

    #[test]
    fn published_contract_sizes_are_consistent() {
        assert_eq!(BINS, FFT_SIZE / 2 + 1);
        assert_eq!(MODEL_LOOKAHEAD_SAMPLES, 1_920);
        assert_eq!(STATE_SIZE, DPDFNET2_STATE_SIZE);
        assert!(DPDFNET8_STATE_SIZE > DPDFNET2_STATE_SIZE);
        assert!(ERB_NORM_STATE_SIZE + SPEC_NORM_STATE_SIZE < STATE_SIZE);
    }

    #[test]
    fn metadata_builds_the_official_normalization_prefix() {
        let (metadata, state) = parse_metadata(&valid_metadata()).unwrap();
        assert_eq!(metadata.profile, "dpdfnet2_48khz_hr");
        assert_eq!(state.len(), STATE_SIZE);
        assert!(state[..ERB_NORM_STATE_SIZE]
            .iter()
            .all(|value| *value == -1.0));
        assert!(
            state[ERB_NORM_STATE_SIZE..ERB_NORM_STATE_SIZE + SPEC_NORM_STATE_SIZE]
                .iter()
                .all(|value| *value == -2.0)
        );
        assert!(state[ERB_NORM_STATE_SIZE + SPEC_NORM_STATE_SIZE..]
            .iter()
            .all(|value| *value == 0.0));
    }

    #[test]
    fn metadata_rejects_non_finite_normalization_state() {
        let mut metadata = valid_metadata();
        metadata.insert(
            "spec_norm_init".into(),
            std::iter::once("NaN")
                .chain(std::iter::repeat_n("-2", SPEC_NORM_STATE_SIZE - 1))
                .collect::<Vec<_>>()
                .join(","),
        );
        assert!(parse_metadata(&metadata)
            .unwrap_err()
            .contains("non-finite"));
    }

    #[test]
    fn metadata_accepts_the_official_dpdfnet8_state_geometry() {
        let mut metadata = valid_metadata();
        metadata.insert("state_size".into(), DPDFNET8_STATE_SIZE.to_string());
        let (metadata, state) = parse_metadata(&metadata).unwrap();
        assert_eq!(metadata.state_size, DPDFNET8_STATE_SIZE);
        assert_eq!(state.len(), DPDFNET8_STATE_SIZE);
    }

    #[test]
    fn production_stream_is_partition_invariant_resettable_and_exact_length() {
        let (_directory, path) = write_identity_model();
        let config = OnnxModelConfig {
            path,
            sample_rate: SAMPLE_RATE,
        };
        let input: Vec<f64> = (0..5_123)
            .map(|index| match index {
                17 => f64::NAN,
                2_401 => f64::INFINITY,
                _ => (std::f64::consts::TAU * 437.0 * index as f64 / 44_100.0).sin() * 0.2,
            })
            .collect();
        let mut stream = StreamingProcessor::new(&config, 44_100, 1).unwrap();
        assert!(stream
            .streams
            .iter()
            .all(|stream| stream.reuse_runtime_state));
        let mut expected = stream.process_block(&[input.clone()]).unwrap();
        expected[0].extend(stream.finish().unwrap().remove(0));

        stream.reset();
        let mut actual = vec![Vec::new()];
        let mut position = 0usize;
        for size in [1usize, 17, 503, 2, 997, 31, 2_048] {
            if position == input.len() {
                break;
            }
            let end = position.saturating_add(size).min(input.len());
            let ready = stream
                .process_block(&[input[position..end].to_vec()])
                .unwrap();
            actual[0].extend_from_slice(&ready[0]);
            position = end;
        }
        if position < input.len() {
            let ready = stream.process_block(&[input[position..].to_vec()]).unwrap();
            actual[0].extend_from_slice(&ready[0]);
        }
        actual[0].extend(stream.finish().unwrap().remove(0));

        assert_eq!(expected[0].len(), input.len());
        assert_eq!(actual[0].len(), input.len());
        assert!(actual[0].iter().all(|sample| sample.is_finite()));
        let maximum = expected[0]
            .iter()
            .zip(&actual[0])
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f64, f64::max);
        assert!(maximum < 1e-6, "partition maximum difference was {maximum}");
        assert!(stream.process_block(&[vec![0.0]]).is_err());
    }

    #[test]
    fn native_rate_stream_sanitizes_and_finishes_exactly() {
        let (_directory, path) = write_identity_model();
        let config = OnnxModelConfig {
            path,
            sample_rate: SAMPLE_RATE,
        };
        let input: Vec<f64> = (0..5_123)
            .map(|index| match index {
                31 => f64::NAN,
                2_777 => f64::NEG_INFINITY,
                _ => {
                    (std::f64::consts::TAU * 613.0 * index as f64 / SAMPLE_RATE as f64).sin() * 0.2
                }
            })
            .collect();
        let mut stream = StreamingProcessor::new(&config, SAMPLE_RATE, 1).unwrap();
        assert!(stream.native_rate);
        let mut output = stream.process_block(&[input.clone()]).unwrap().remove(0);
        output.extend(stream.finish().unwrap().remove(0));

        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|sample| sample.is_finite()));
    }

    fn write_identity_model() -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dpdfnet-identity.onnx");
        let mut bytes = Vec::new();
        identity_model().encode(&mut bytes).unwrap();
        std::fs::write(&path, bytes).unwrap();
        (directory, path)
    }

    fn identity_model() -> ModelProto {
        let shapes: [(&str, &str, &[i64]); 2] = [
            ("spectrum", "enhanced", &[1, 1, BINS as i64, 2]),
            ("state", "state_out", &[DPDFNET2_STATE_SIZE as i64]),
        ];
        ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            producer_name: "denoize-test".into(),
            graph: Some(GraphProto {
                name: "dpdfnet-identity".into(),
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
                    .map(|(input, _, shape)| value_info(input, shape))
                    .collect(),
                output: shapes
                    .iter()
                    .map(|(_, output, shape)| value_info(output, shape))
                    .collect(),
                ..Default::default()
            }),
            metadata_props: valid_metadata()
                .into_iter()
                .map(|(key, value)| StringStringEntryProto { key, value })
                .collect(),
            ..Default::default()
        }
    }

    fn value_info(name: &str, shape: &[i64]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.into(),
            r#type: Some(TypeProto {
                denotation: String::new(),
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: tensor_proto::DataType::Float as i32,
                    shape: Some(TensorShapeProto {
                        dim: shape.iter().copied().map(dimension).collect(),
                    }),
                })),
            }),
            doc_string: String::new(),
        }
    }

    fn dimension(value: i64) -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(value)),
            denotation: String::new(),
        }
    }
}
