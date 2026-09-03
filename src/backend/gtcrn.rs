//! Official GTCRN streaming ONNX adapter.
//!
//! The model consumes one 512-point STFT frame at a time at 16 kHz and carries
//! three recurrent state tensors. The tensor layout follows the upstream MIT
//! implementation in `Xiaobin-Rong/gtcrn`.

use super::tract_runtime::SharedRunnable;
use super::OnnxModelConfig;
use crate::AcceleratorRuntime;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::sync::Arc;
use tract_onnx::prelude::*;

pub const SAMPLE_RATE: u32 = 16_000;
pub const FFT_SIZE: usize = 512;
pub const HOP_SIZE: usize = 256;
pub const BINS: usize = 257;
const SPECTRUM_SHAPE: [usize; 4] = [1, BINS, 1, 2];
const CONV_SHAPE: [usize; 5] = [2, 1, 16, 16, 33];
const TRA_SHAPE: [usize; 5] = [2, 3, 1, 1, 16];
const INTER_SHAPE: [usize; 4] = [2, 1, 33, 16];
const CONV_SIZE: usize = 2 * 16 * 16 * 33;
const TRA_SIZE: usize = 2 * 3 * 16;
const INTER_SIZE: usize = 2 * 33 * 16;
const RECURRENT_STATE_SIZE: usize = CONV_SIZE + TRA_SIZE + INTER_SIZE;
const COMPILED_MODEL_ALLOWANCE_BYTES: u64 = 64 * 1024 * 1024;
const FFT_SCRATCH_COMPLEX_ALLOWANCE: usize = FFT_SIZE;

pub fn process(
    channels: &[Vec<f64>],
    input_sample_rate: u32,
    config: &OnnxModelConfig,
) -> Result<Vec<Vec<f64>>, String> {
    GtcrnModel::load(config)?.process(channels, input_sample_rate)
}

fn process_channel(
    input: &[f64],
    input_sample_rate: u32,
    model: &GtcrnModel,
) -> Result<Vec<f64>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let at_model_rate = crate::resample::resample(input, input_sample_rate, SAMPLE_RATE)?;
    let mut stream = model.stream()?;
    let mut enhanced = Vec::with_capacity(at_model_rate.len() + FFT_SIZE);
    for chunk in at_model_rate.chunks(HOP_SIZE) {
        let mut hop = [0.0; HOP_SIZE];
        for (output, input) in hop.iter_mut().zip(chunk) {
            *output = *input as f32;
        }
        enhanced.extend(stream.process_hop(&hop)?);
    }
    enhanced.extend(stream.flush()?);
    // The causal WOLA frontend has one hop of algorithmic latency.
    let enhanced = enhanced
        .into_iter()
        .skip(HOP_SIZE)
        .take(at_model_rate.len());
    let model_output: Vec<f64> = enhanced.map(|sample| sample as f64).collect();
    let mut output = crate::resample::resample(&model_output, SAMPLE_RATE, input_sample_rate)?;
    output.truncate(input.len());
    output.resize(input.len(), 0.0);
    Ok(output)
}

fn validate_config(config: &OnnxModelConfig) -> Result<(), String> {
    if config.sample_rate != SAMPLE_RATE {
        return Err(format!(
            "GTCRN expects a {SAMPLE_RATE} Hz model, got {} Hz",
            config.sample_rate
        ));
    }
    if !config.path.is_file() {
        return Err(format!(
            "GTCRN ONNX model does not exist or is not a file: {}",
            config.path.display()
        ));
    }
    Ok(())
}

/// Parsed and optimized GTCRN graph that can create independent recurrent
/// streams without reopening or recompiling the model pathname.
#[derive(Clone)]
pub struct GtcrnModel {
    model: SharedRunnable,
    initial_conv: Arc<Tensor>,
    initial_tra: Arc<Tensor>,
    initial_inter: Arc<Tensor>,
    runtime: AcceleratorRuntime,
}

impl GtcrnModel {
    pub fn load(config: &OnnxModelConfig) -> Result<Self, String> {
        Self::load_with_accelerator(config, AcceleratorRuntime::Cpu)
    }

    /// Load the graph for a concrete already-selected runtime.
    pub fn load_with_accelerator(
        config: &OnnxModelConfig,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        validate_config(config)?;
        Ok(Self {
            model: load_model(&config.path, runtime)?,
            initial_conv: zeroed_state(&CONV_SHAPE, "GTCRN convolution state")?,
            initial_tra: zeroed_state(&TRA_SHAPE, "GTCRN recurrent state")?,
            initial_inter: zeroed_state(&INTER_SHAPE, "GTCRN inter-frame state")?,
            runtime,
        })
    }

    /// Concrete runtime used to optimize this reusable graph.
    pub const fn runtime(&self) -> AcceleratorRuntime {
        self.runtime
    }

    pub fn stream(&self) -> Result<GtcrnStream, String> {
        GtcrnStream::from_model(
            Arc::clone(&self.model),
            Arc::clone(&self.initial_conv),
            Arc::clone(&self.initial_tra),
            Arc::clone(&self.initial_inter),
        )
    }

    /// Process finite planar audio with independent recurrent state per
    /// channel while reusing this parsed graph.
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
}

/// Stateful 16 kHz GTCRN processor. Each call consumes and returns exactly
/// 256 mono samples, making it suitable for realtime hosts and pipes.
pub struct GtcrnStream {
    model: SharedRunnable,
    initial_conv: Arc<Tensor>,
    initial_tra: Arc<Tensor>,
    initial_inter: Arc<Tensor>,
    conv: Arc<Tensor>,
    tra: Arc<Tensor>,
    inter: Arc<Tensor>,
    analysis: [f32; FFT_SIZE],
    overlap: [f32; FFT_SIZE],
    window: [f32; FFT_SIZE],
    spectrum: Vec<Complex32>,
    model_input: Option<Tensor>,
    fft_scratch: Vec<Complex32>,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
}

impl GtcrnStream {
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let model = GtcrnModel::load(&OnnxModelConfig {
            path: path.to_path_buf(),
            sample_rate: SAMPLE_RATE,
        })?;
        model.stream()
    }

    fn from_model(
        model: SharedRunnable,
        initial_conv: Arc<Tensor>,
        initial_tra: Arc<Tensor>,
        initial_inter: Arc<Tensor>,
    ) -> Result<Self, String> {
        let window = std::array::from_fn(|index| {
            let phase = std::f32::consts::TAU * index as f32 / FFT_SIZE as f32;
            (0.5 * (1.0 - phase.cos())).sqrt()
        });
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let ifft = planner.plan_fft_inverse(FFT_SIZE);
        let fft_scratch_len = fft
            .get_inplace_scratch_len()
            .max(ifft.get_inplace_scratch_len());
        if fft_scratch_len > FFT_SCRATCH_COMPLEX_ALLOWANCE {
            return Err(format!(
                "GTCRN FFT requires {fft_scratch_len} scratch values, exceeding the {FFT_SCRATCH_COMPLEX_ALLOWANCE}-value resource allowance"
            ));
        }
        Ok(Self {
            model,
            conv: initial_conv.as_ref().clone().into_arc_tensor(),
            tra: initial_tra.as_ref().clone().into_arc_tensor(),
            inter: initial_inter.as_ref().clone().into_arc_tensor(),
            initial_conv,
            initial_tra,
            initial_inter,
            analysis: [0.0; FFT_SIZE],
            overlap: [0.0; FFT_SIZE],
            window,
            spectrum: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
            model_input: Some(Tensor::zero::<f32>(&SPECTRUM_SHAPE).map_err(tract_error)?),
            fft_scratch: vec![Complex32::new(0.0, 0.0); fft_scratch_len],
            fft,
            ifft,
        })
    }

    pub fn reset(&mut self) {
        self.conv = self.initial_conv.as_ref().clone().into_arc_tensor();
        self.tra = self.initial_tra.as_ref().clone().into_arc_tensor();
        self.inter = self.initial_inter.as_ref().clone().into_arc_tensor();
        self.analysis.fill(0.0);
        self.overlap.fill(0.0);
    }

    pub fn process_hop(&mut self, input: &[f32; HOP_SIZE]) -> Result<[f32; HOP_SIZE], String> {
        self.analysis.copy_within(HOP_SIZE.., 0);
        self.analysis[FFT_SIZE - HOP_SIZE..].copy_from_slice(input);
        for (index, (sample, window)) in self.analysis.iter().zip(&self.window).enumerate() {
            self.spectrum[index] = Complex32::new(sample * window, 0.0);
        }
        self.fft
            .process_with_scratch(&mut self.spectrum, &mut self.fft_scratch);

        if self.model_input.is_none() {
            // Inference owns its input even on failure. Reallocate only for a
            // subsequent recovery attempt; successful hops recycle the
            // enhanced output tensor below.
            self.model_input = Some(Tensor::zero::<f32>(&SPECTRUM_SHAPE).map_err(tract_error)?);
        }
        {
            let model_input = self
                .model_input
                .as_mut()
                .ok_or_else(|| "GTCRN model input is unavailable".to_string())?;
            validate_model_tensor(model_input, &SPECTRUM_SHAPE, BINS * 2, "spectrum input")?;
            let mut model_input_plain = model_input.try_as_plain_mut().map_err(tract_error)?;
            let model_input = model_input_plain
                .as_slice_mut::<f32>()
                .map_err(tract_error)?;
            for (bin, value) in self.spectrum.iter().take(BINS).enumerate() {
                model_input[bin * 2] = value.re;
                model_input[bin * 2 + 1] = value.im;
            }
        }
        let model_input = self
            .model_input
            .take()
            .expect("GTCRN model input was validated above");
        // The enhanced spectrum has the same plain F32 geometry as the next
        // model input, so retain its owned tensor and overwrite it next hop.
        let enhanced = self.infer(model_input)?;
        self.model_input = Some(enhanced);
        let enhanced = self
            .model_input
            .as_ref()
            .expect("successful GTCRN inference restores the model input tensor");
        let enhanced_plain = enhanced.try_as_plain().map_err(tract_error)?;
        let enhanced = enhanced_plain.as_slice::<f32>().map_err(tract_error)?;
        for bin in 0..BINS {
            self.spectrum[bin] = Complex32::new(enhanced[bin * 2], enhanced[bin * 2 + 1]);
        }
        for bin in BINS..FFT_SIZE {
            self.spectrum[bin] = self.spectrum[FFT_SIZE - bin].conj();
        }
        self.spectrum[0].im = 0.0;
        self.spectrum[BINS - 1].im = 0.0;
        self.ifft
            .process_with_scratch(&mut self.spectrum, &mut self.fft_scratch);
        for (index, value) in self.spectrum.iter().enumerate() {
            self.overlap[index] += value.re * self.window[index] / FFT_SIZE as f32;
        }
        let output = std::array::from_fn(|index| self.overlap[index]);
        self.overlap.copy_within(HOP_SIZE.., 0);
        self.overlap[FFT_SIZE - HOP_SIZE..].fill(0.0);
        Ok(output)
    }

    pub fn flush(&mut self) -> Result<[f32; HOP_SIZE], String> {
        self.process_hop(&[0.0; HOP_SIZE])
    }

    fn infer(&mut self, spectrum: Tensor) -> Result<Tensor, String> {
        let conv = std::mem::replace(&mut self.conv, Arc::clone(&self.initial_conv));
        let tra = std::mem::replace(&mut self.tra, Arc::clone(&self.initial_tra));
        let inter = std::mem::replace(&mut self.inter, Arc::clone(&self.initial_inter));
        let inputs = tvec!(
            spectrum.into_tvalue(),
            conv.into_tvalue(),
            tra.into_tvalue(),
            inter.into_tvalue(),
        );
        // GTCRN contains stateful Scan operators. A fresh Runnable::run state
        // is required for every hop; only its three explicit recurrent
        // payloads are moved between calls here.
        let mut outputs = self.model.run(inputs).map_err(tract_error)?;
        if outputs.len() != 4 {
            return Err(format!(
                "GTCRN returned {} outputs, expected 4",
                outputs.len()
            ));
        }
        validate_model_tensor(&outputs[0], &SPECTRUM_SHAPE, BINS * 2, "enhanced spectrum")?;
        validate_model_tensor(&outputs[1], &CONV_SHAPE, CONV_SIZE, "convolution state")?;
        validate_model_tensor(&outputs[2], &TRA_SHAPE, TRA_SIZE, "recurrent state")?;
        validate_model_tensor(&outputs[3], &INTER_SHAPE, INTER_SIZE, "inter-frame state")?;

        let next_inter = outputs
            .pop()
            .expect("GTCRN output count was validated above");
        let next_tra = outputs
            .pop()
            .expect("GTCRN output count was validated above");
        let next_conv = outputs
            .pop()
            .expect("GTCRN output count was validated above");
        let enhanced = outputs
            .pop()
            .expect("GTCRN output count was validated above");
        self.conv = next_conv.into_arc_tensor();
        self.tra = next_tra.into_arc_tensor();
        self.inter = next_inter.into_arc_tensor();
        Ok(enhanced.into_tensor())
    }
}

fn validate_model_tensor(
    tensor: &Tensor,
    expected_shape: &[usize],
    expected_len: usize,
    context: &str,
) -> Result<(), String> {
    if tensor.shape() != expected_shape {
        return Err(format!(
            "GTCRN returned {context} shape {:?}, expected {expected_shape:?}",
            tensor.shape()
        ));
    }
    if tensor.datum_type() != DatumType::F32 {
        return Err(format!(
            "GTCRN returned {context} type {:?}, expected F32",
            tensor.datum_type()
        ));
    }
    if tensor.len() != expected_len {
        return Err(format!(
            "GTCRN returned {context} with {} scalars, expected {expected_len}",
            tensor.len()
        ));
    }
    tensor
        .try_as_plain()
        .map_err(tract_error)?
        .as_slice::<f32>()
        .map_err(tract_error)?;
    Ok(())
}

/// Continuous, channel-planar GTCRN processing at an arbitrary input rate.
///
/// One optimized graph is shared by every channel while recurrent tensors,
/// WOLA state, sample-rate-converter clocks, and incomplete model hops remain
/// independent. Output can be empty until the bounded conversion and model
/// latency has been satisfied. [`finish`](Self::finish) returns the exact
/// remaining number of input-rate frames.
pub(crate) struct StreamingProcessor {
    channels: usize,
    to_model_rate: crate::resample::StreamingResampler,
    from_model_rate: crate::resample::StreamingResampler,
    streams: Vec<GtcrnStream>,
    pending_model_rate: Vec<Vec<f64>>,
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
        Self::new_with_accelerator(config, sample_rate, channels, AcceleratorRuntime::Cpu)
    }

    pub(crate) fn new_with_accelerator(
        config: &OnnxModelConfig,
        sample_rate: u32,
        channels: usize,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        let model = GtcrnModel::load_with_accelerator(config, runtime)?;
        Self::new_with_model(&model, sample_rate, channels)
    }

    pub(crate) fn new_with_model(
        model: &GtcrnModel,
        sample_rate: u32,
        channels: usize,
    ) -> Result<Self, String> {
        if channels == 0 || channels > crate::config::MAX_STREAM_CHANNELS {
            return Err(format!(
                "GTCRN streaming channels must be between 1 and {}",
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
            .map_err(|_| "unable to reserve GTCRN channel streams".to_string())?;
        let mut pending_model_rate = Vec::new();
        pending_model_rate
            .try_reserve_exact(channels)
            .map_err(|_| "unable to reserve GTCRN pending channels".to_string())?;
        for _ in 0..channels {
            streams.push(model.stream()?);
            let mut pending = Vec::new();
            pending
                .try_reserve_exact(HOP_SIZE)
                .map_err(|_| "unable to reserve GTCRN pending samples".to_string())?;
            pending_model_rate.push(pending);
        }
        Ok(Self {
            channels,
            to_model_rate,
            from_model_rate,
            streams,
            pending_model_rate,
            discard_model_frames: HOP_SIZE,
            model_input_frames: 0,
            model_output_frames: 0,
            input_frames: 0,
            output_frames: 0,
            finished: false,
        })
    }

    pub(crate) fn process_block(&mut self, input: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if self.finished {
            return Err("GTCRN stream is finished; reset it before processing more input".into());
        }
        let frames = validate_stream_block(input, self.channels)?;
        let input_frames = self
            .input_frames
            .checked_add(frames)
            .ok_or_else(|| "GTCRN streaming input length overflow".to_string())?;
        let at_model_rate = self.to_model_rate.process(input)?;
        let enhanced_model_rate = self.process_model_rate(&at_model_rate)?;
        let output = self.from_model_rate.process(&enhanced_model_rate)?;
        let produced = validate_stream_block(&output, self.channels)?;
        let output_frames = self
            .output_frames
            .checked_add(produced)
            .ok_or_else(|| "GTCRN streaming output length overflow".to_string())?;
        if output_frames > input_frames {
            return Err("GTCRN stream produced samples ahead of its input clock".into());
        }
        self.input_frames = input_frames;
        self.output_frames = output_frames;
        Ok(output)
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<Vec<f64>>, String> {
        let remaining = self
            .input_frames
            .checked_sub(self.output_frames)
            .ok_or_else(|| "GTCRN stream exceeded its input clock".to_string())?;
        let mut output = empty_output(self.channels, remaining)?;
        if self.finished {
            return Ok(output);
        }

        let model_input_tail = self.to_model_rate.finish()?;
        let enhanced = self.process_model_rate(&model_input_tail)?;
        let converted = self.from_model_rate.process(&enhanced)?;
        append_limited(&mut output, &converted, remaining)?;

        let enhanced = self.finish_model_rate()?;
        let converted = self.from_model_rate.process(&enhanced)?;
        append_limited(&mut output, &converted, remaining)?;

        let converted = self.from_model_rate.finish()?;
        append_limited(&mut output, &converted, remaining)?;
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
        self.discard_model_frames = HOP_SIZE;
        self.model_input_frames = 0;
        self.model_output_frames = 0;
        self.input_frames = 0;
        self.output_frames = 0;
        self.finished = false;
    }

    fn process_model_rate(&mut self, input: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let frames = validate_stream_block(input, self.channels)?;
        let model_input_frames = self
            .model_input_frames
            .checked_add(frames)
            .ok_or_else(|| "GTCRN model input length overflow".to_string())?;
        let reserve = model_input_frames
            .checked_sub(self.model_output_frames)
            .ok_or_else(|| "GTCRN model output exceeded its input clock".to_string())?;
        let mut output = empty_output(self.channels, reserve)?;
        self.model_input_frames = model_input_frames;
        if frames == 0 {
            return Ok(output);
        }

        let mut position = 0usize;
        let pending_frames = self.pending_model_rate.first().map_or(0, Vec::len);
        if pending_frames > 0 {
            let copied = (HOP_SIZE - pending_frames).min(frames);
            for (pending, source) in self.pending_model_rate.iter_mut().zip(input) {
                pending.extend_from_slice(&source[..copied]);
            }
            position = copied;
            if pending_frames + copied == HOP_SIZE {
                self.process_pending_hop(&mut output)?;
                for pending in &mut self.pending_model_rate {
                    pending.clear();
                }
            }
        }

        while frames - position >= HOP_SIZE {
            self.process_slice_hop(input, position, &mut output)?;
            position += HOP_SIZE;
        }
        if position < frames {
            for (pending, source) in self.pending_model_rate.iter_mut().zip(input) {
                pending.extend_from_slice(&source[position..]);
            }
        }
        Ok(output)
    }

    fn finish_model_rate(&mut self) -> Result<Vec<Vec<f64>>, String> {
        let remaining = self
            .model_input_frames
            .checked_sub(self.model_output_frames)
            .ok_or_else(|| "GTCRN model exceeded its input clock".to_string())?;
        let mut output = empty_output(self.channels, remaining)?;
        let pending = self.pending_model_rate.first().map_or(0, Vec::len);
        if pending > 0 {
            if self
                .pending_model_rate
                .iter()
                .any(|channel| channel.len() != pending)
            {
                return Err("GTCRN pending channels became unaligned".into());
            }
            for channel in &mut self.pending_model_rate {
                channel.resize(HOP_SIZE, 0.0);
            }
            self.process_pending_hop(&mut output)?;
            for channel in &mut self.pending_model_rate {
                channel.clear();
            }
        }
        self.flush_model_hop(&mut output)?;
        if output.first().map_or(0, Vec::len) < remaining {
            for channel in &mut output {
                channel.resize(remaining, 0.0);
            }
        }
        self.model_output_frames = self.model_input_frames;
        Ok(output)
    }

    fn process_pending_hop(&mut self, output: &mut [Vec<f64>]) -> Result<(), String> {
        let (skip, retained) = self.model_hop_window()?;
        for channel in 0..self.channels {
            let mut input = [0.0f32; HOP_SIZE];
            for (destination, source) in input.iter_mut().zip(&self.pending_model_rate[channel]) {
                *destination = crate::audio::sanitize_sample(*source) as f32;
            }
            let enhanced = self.streams[channel].process_hop(&input)?;
            output[channel].extend(
                enhanced[skip..skip + retained]
                    .iter()
                    .map(|sample| *sample as f64),
            );
        }
        self.commit_model_hop(skip, retained)
    }

    fn process_slice_hop(
        &mut self,
        input: &[Vec<f64>],
        position: usize,
        output: &mut [Vec<f64>],
    ) -> Result<(), String> {
        let (skip, retained) = self.model_hop_window()?;
        for channel in 0..self.channels {
            let mut hop = [0.0f32; HOP_SIZE];
            for (destination, source) in hop
                .iter_mut()
                .zip(&input[channel][position..position + HOP_SIZE])
            {
                *destination = crate::audio::sanitize_sample(*source) as f32;
            }
            let enhanced = self.streams[channel].process_hop(&hop)?;
            output[channel].extend(
                enhanced[skip..skip + retained]
                    .iter()
                    .map(|sample| *sample as f64),
            );
        }
        self.commit_model_hop(skip, retained)
    }

    fn flush_model_hop(&mut self, output: &mut [Vec<f64>]) -> Result<(), String> {
        let (skip, retained) = self.model_hop_window()?;
        for channel in 0..self.channels {
            let enhanced = self.streams[channel].flush()?;
            output[channel].extend(
                enhanced[skip..skip + retained]
                    .iter()
                    .map(|sample| *sample as f64),
            );
        }
        self.commit_model_hop(skip, retained)
    }

    fn model_hop_window(&self) -> Result<(usize, usize), String> {
        let skip = self.discard_model_frames.min(HOP_SIZE);
        let available = HOP_SIZE - skip;
        let remaining = self
            .model_input_frames
            .checked_sub(self.model_output_frames)
            .ok_or_else(|| "GTCRN model output exceeded its input clock".to_string())?;
        Ok((skip, available.min(remaining)))
    }

    fn commit_model_hop(&mut self, skipped: usize, retained: usize) -> Result<(), String> {
        self.discard_model_frames = self
            .discard_model_frames
            .checked_sub(skipped)
            .ok_or_else(|| "GTCRN latency accounting underflow".to_string())?;
        self.model_output_frames = self
            .model_output_frames
            .checked_add(retained)
            .ok_or_else(|| "GTCRN model output length overflow".to_string())?;
        Ok(())
    }
}

fn validate_stream_block(input: &[Vec<f64>], channels: usize) -> Result<usize, String> {
    if input.len() != channels {
        return Err(format!(
            "GTCRN stream expected {channels} channels, received {}",
            input.len()
        ));
    }
    let frames = input.first().map_or(0, Vec::len);
    if input.iter().any(|channel| channel.len() != frames) {
        return Err("GTCRN stream channels must contain the same number of frames".into());
    }
    Ok(frames)
}

fn empty_output(channels: usize, capacity: usize) -> Result<Vec<Vec<f64>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(channels)
        .map_err(|_| "unable to reserve GTCRN output channels".to_string())?;
    for _ in 0..channels {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(capacity)
            .map_err(|_| "unable to reserve GTCRN output samples".to_string())?;
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
        return Err("GTCRN stream produced an invalid channel count".into());
    }
    let destination_frames = destination.first().map_or(0, Vec::len);
    let source_frames = validate_stream_block(source, destination.len())?;
    if destination
        .iter()
        .any(|channel| channel.len() != destination_frames)
    {
        return Err("GTCRN destination channels became unaligned".into());
    }
    let retained = frame_limit
        .checked_sub(destination_frames)
        .ok_or_else(|| "GTCRN streaming output exceeded its target".to_string())?
        .min(source_frames);
    for (destination, source) in destination.iter_mut().zip(source) {
        destination
            .try_reserve_exact(retained)
            .map_err(|_| "unable to grow GTCRN output".to_string())?;
        destination.extend_from_slice(&source[..retained]);
    }
    Ok(())
}

fn zeroed_state(shape: &[usize], context: &str) -> Result<Arc<Tensor>, String> {
    Tensor::zero::<f32>(shape)
        .map(|tensor| tensor.into_arc_tensor())
        .map_err(|error| format!("unable to allocate {context}: {error:#}"))
}

pub(crate) fn streaming_state_bytes(channels: usize) -> Result<u64, crate::ConfigError> {
    // Count both the active recurrent tensors and a full per-channel copy of
    // their reset templates even though the templates are shared by streams.
    // This keeps the estimate conservative while accounting for every retained
    // FFT/model-input buffer and the maximum permitted rustfft scratch space.
    let per_channel_f32 = RECURRENT_STATE_SIZE
        .checked_mul(2)
        .and_then(|value| value.checked_add(FFT_SIZE * 5))
        .and_then(|value| value.checked_add(FFT_SCRATCH_COMPLEX_ALLOWANCE * 2))
        .and_then(|value| value.checked_add(BINS * 2))
        .ok_or(crate::ConfigError::ResourceOverflow {
            resource: "GTCRN stream state",
        })?;
    let per_channel_bytes = u64::try_from(per_channel_f32)
        .ok()
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
        .and_then(|value| {
            u64::try_from(HOP_SIZE)
                .ok()
                .and_then(|pending| pending.checked_mul(std::mem::size_of::<f64>() as u64))
                .and_then(|pending| value.checked_add(pending))
        })
        .and_then(|value| {
            value.checked_add(
                (std::mem::size_of::<GtcrnStream>()
                    + std::mem::size_of::<Vec<f64>>()
                    + std::mem::size_of::<Tensor>() * 7) as u64,
            )
        })
        .ok_or(crate::ConfigError::ResourceOverflow {
            resource: "GTCRN stream state",
        })?;
    let channel_bytes = per_channel_bytes
        .checked_mul(
            u64::try_from(channels).map_err(|_| crate::ConfigError::ResourceOverflow {
                resource: "GTCRN stream state",
            })?,
        )
        .ok_or(crate::ConfigError::ResourceOverflow {
            resource: "GTCRN stream state",
        })?;
    COMPILED_MODEL_ALLOWANCE_BYTES
        .checked_add(channel_bytes)
        .ok_or(crate::ConfigError::ResourceOverflow {
            resource: "GTCRN stream state",
        })
}

fn load_model(
    path: &std::path::Path,
    runtime: AcceleratorRuntime,
) -> Result<SharedRunnable, String> {
    let mut model = tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|error| format!("failed to load GTCRN model {}: {error}", path.display()))?;
    let input_shapes: [&[usize]; 4] = [&SPECTRUM_SHAPE, &CONV_SHAPE, &TRA_SHAPE, &INTER_SHAPE];
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
    super::tract_runtime::prepare(model, runtime, "GTCRN model")
}

fn tract_error(error: impl std::fmt::Display) -> String {
    format!("GTCRN inference failed: {error:#}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use tract_onnx::pb::{
        tensor_proto, tensor_shape_proto, type_proto, GraphProto, ModelProto, NodeProto,
        OperatorSetIdProto, TensorShapeProto, TypeProto, ValueInfoProto,
    };

    #[test]
    fn rejects_wrong_rate_before_loading() {
        let config = OnnxModelConfig {
            path: "missing.onnx".into(),
            sample_rate: 48_000,
        };
        assert!(validate_config(&config).unwrap_err().contains("16000 Hz"));
    }

    #[test]
    fn published_state_sizes_match_shapes() {
        assert_eq!(CONV_SIZE, 16_896);
        assert_eq!(TRA_SIZE, 96);
        assert_eq!(INTER_SIZE, 1_056);
        assert_eq!(CONV_SHAPE.iter().product::<usize>(), CONV_SIZE);
        assert_eq!(TRA_SHAPE.iter().product::<usize>(), TRA_SIZE);
        assert_eq!(INTER_SHAPE.iter().product::<usize>(), INTER_SIZE);
        assert_eq!(SPECTRUM_SHAPE.iter().product::<usize>(), BINS * 2);
    }

    #[test]
    fn stream_with_owned_tensors_remains_send() {
        fn assert_send<T: Send>() {}

        assert_send::<GtcrnStream>();
    }

    #[test]
    fn identity_model_moves_recurrent_payloads_and_reset_restores_zeroes() {
        let (_directory, path) = write_identity_model();
        let model = GtcrnModel::load(&OnnxModelConfig {
            path,
            sample_rate: SAMPLE_RATE,
        })
        .unwrap();
        let mut stream = model.stream().unwrap();
        stream.conv = filled_state(&CONV_SHAPE, CONV_SIZE, 0.25);
        stream.tra = filled_state(&TRA_SHAPE, TRA_SIZE, -0.5);
        stream.inter = filled_state(&INTER_SHAPE, INTER_SIZE, 0.75);

        stream.process_hop(&[0.0; HOP_SIZE]).unwrap();
        assert_state_value(&stream.conv, 0.25);
        assert_state_value(&stream.tra, -0.5);
        assert_state_value(&stream.inter, 0.75);

        stream.reset();
        assert_state_value(&stream.conv, 0.0);
        assert_state_value(&stream.tra, 0.0);
        assert_state_value(&stream.inter, 0.0);
    }

    #[test]
    fn reset_owned_tensor_stream_matches_fresh_identity_stream() {
        let (_directory, path) = write_identity_model();
        let model = GtcrnModel::load(&OnnxModelConfig {
            path,
            sample_rate: SAMPLE_RATE,
        })
        .unwrap();
        let warmup = std::array::from_fn(|index| (index as f32 * 0.017).sin() * 0.3);
        let first = std::array::from_fn(|index| (index as f32 * 0.031).cos() * 0.2);
        let second = std::array::from_fn(|index| (index as f32 * 0.047).sin() * 0.1);

        let mut reset = model.stream().unwrap();
        reset.process_hop(&warmup).unwrap();
        reset.process_hop(&first).unwrap();
        reset.reset();
        let mut fresh = model.stream().unwrap();

        assert_eq!(
            reset.process_hop(&first).unwrap(),
            fresh.process_hop(&first).unwrap()
        );
        assert_eq!(
            reset.process_hop(&second).unwrap(),
            fresh.process_hop(&second).unwrap()
        );
        assert_eq!(reset.flush().unwrap(), fresh.flush().unwrap());
    }

    #[test]
    fn loaded_model_survives_path_replacement_and_creates_independent_streams() {
        let (_directory, path) = write_identity_model();
        let model = GtcrnModel::load(&OnnxModelConfig {
            path: path.clone(),
            sample_rate: SAMPLE_RATE,
        })
        .unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, b"replaced after GTCRN load").unwrap();

        let input = std::array::from_fn(|index| (index as f32 * 0.031).sin() * 0.2);
        let mut first = model.stream().unwrap();
        let mut second = model.stream().unwrap();
        assert_eq!(
            first.process_hop(&input).unwrap(),
            second.process_hop(&input).unwrap()
        );

        let mut streaming = StreamingProcessor::new_with_model(&model, 48_000, 1).unwrap();
        let ready = streaming.process_block(&[vec![0.1; 480]]).unwrap();
        assert_eq!(ready.len(), 1);
    }

    #[test]
    fn arbitrary_rate_stream_is_partition_invariant_and_exact_length() {
        let (_directory, path) = write_identity_model();
        let config = OnnxModelConfig {
            path,
            sample_rate: SAMPLE_RATE,
        };
        let input: Vec<f64> = (0..5_123)
            .map(|index| (std::f64::consts::TAU * 437.0 * index as f64 / 44_100.0).sin() * 0.2)
            .collect();

        let mut contiguous = StreamingProcessor::new(&config, 44_100, 1).unwrap();
        let mut expected = contiguous.process_block(&[input.clone()]).unwrap();
        let tail = contiguous.finish().unwrap();
        expected[0].extend_from_slice(&tail[0]);

        let mut partitioned = StreamingProcessor::new(&config, 44_100, 1).unwrap();
        let mut actual = vec![Vec::new()];
        let mut position = 0;
        for size in [1usize, 17, 503, 2, 997, 31, 2048] {
            if position == input.len() {
                break;
            }
            let end = position.saturating_add(size).min(input.len());
            let ready = partitioned
                .process_block(&[input[position..end].to_vec()])
                .unwrap();
            actual[0].extend_from_slice(&ready[0]);
            position = end;
        }
        if position < input.len() {
            let ready = partitioned
                .process_block(&[input[position..].to_vec()])
                .unwrap();
            actual[0].extend_from_slice(&ready[0]);
        }
        let tail = partitioned.finish().unwrap();
        actual[0].extend_from_slice(&tail[0]);

        assert_eq!(expected[0].len(), input.len());
        assert_eq!(actual[0].len(), input.len());
        assert!(actual[0].iter().all(|sample| sample.is_finite()));
        let maximum = expected[0]
            .iter()
            .zip(&actual[0])
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f64, f64::max);
        assert!(maximum < 1e-6, "partition maximum difference was {maximum}");
    }

    #[test]
    fn common_streaming_session_reuses_gtcrn_for_stereo_linked_audio() {
        let (_directory, path) = write_identity_model();
        let options = crate::BackendOptions {
            onnx: Some(OnnxModelConfig {
                path,
                sample_rate: SAMPLE_RATE,
            }),
            channel_mode: crate::ChannelMode::StereoLinked,
            ..Default::default()
        };
        let mut session = crate::StreamingBackendSession::new(
            crate::Backend::Gtcrn,
            SAMPLE_RATE,
            2,
            crate::DenoiserConfig::default(SAMPLE_RATE),
            options,
        )
        .unwrap();
        let input = vec![vec![0.1; 777], vec![-0.05; 777]];
        let mut output = session.process_block(&input).unwrap();
        let tail = session.finish().unwrap();
        for (channel, tail) in output.iter_mut().zip(tail) {
            channel.extend(tail);
        }
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|channel| channel.len() == 777));
    }

    pub(crate) fn write_identity_model() -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("gtcrn-identity.onnx");
        let mut bytes = Vec::new();
        identity_model().encode(&mut bytes).unwrap();
        std::fs::write(&path, bytes).unwrap();
        (directory, path)
    }

    fn identity_model() -> ModelProto {
        let shapes: [(&str, &str, &[i64]); 4] = [
            ("mixture", "enhanced", &[1, BINS as i64, 1, 2]),
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
            producer_name: "denoize-test".into(),
            graph: Some(GraphProto {
                name: "gtcrn-identity".into(),
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

    fn filled_state(shape: &[usize], length: usize, value: f32) -> Arc<Tensor> {
        let values = vec![value; length];
        Tensor::from_shape(shape, &values)
            .unwrap()
            .into_arc_tensor()
    }

    fn assert_state_value(state: &Tensor, expected: f32) {
        let state_plain = state.try_as_plain().unwrap();
        let state = state_plain.as_slice::<f32>().unwrap();
        assert!(state.iter().all(|value| *value == expected));
    }
}
