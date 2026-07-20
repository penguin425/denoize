//! Realtime system-audio capture, denoising, and playback.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};

use crate::audio::Audio;
use crate::denoiser::DenoiserConfig;
use crate::{denoise_audio_with_backend_config, Backend, BackendOptions};

/// Settings for a realtime capture-to-playback session.
#[derive(Clone, Debug)]
pub struct LiveConfig {
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub chunk_ms: u32,
    pub backend: Backend,
    pub backend_options: BackendOptions,
    pub denoiser: DenoiserConfig,
}

/// Return the input and output device names exposed by the default host.
pub fn device_names() -> Result<(Vec<String>, Vec<String>), String> {
    let host = cpal::default_host();
    let inputs = host
        .input_devices()
        .map_err(|e| format!("enumerate input devices: {e}"))?
        .map(|d| d.name().unwrap_or_else(|_| "<unknown>".into()))
        .collect();
    let outputs = host
        .output_devices()
        .map_err(|e| format!("enumerate output devices: {e}"))?
        .map(|d| d.name().unwrap_or_else(|_| "<unknown>".into()))
        .collect();
    Ok((inputs, outputs))
}

/// Run until Ctrl-C, processing bounded chunks away from the audio callbacks.
pub fn run(mut config: LiveConfig) -> Result<(), String> {
    let host = cpal::default_host();
    let input = select_device(&host, true, config.input_device.as_deref())?;
    let output = select_device(&host, false, config.output_device.as_deref())?;
    let input_supported = input
        .default_input_config()
        .map_err(|e| format!("input config: {e}"))?;
    let output_supported = output
        .default_output_config()
        .map_err(|e| format!("output config: {e}"))?;
    let input_cfg: StreamConfig = input_supported.clone().into();
    let output_cfg: StreamConfig = output_supported.clone().into();
    if input_cfg.sample_rate != output_cfg.sample_rate {
        return Err(format!(
            "input/output sample rates differ ({} vs {} Hz); select devices with a common default rate",
            input_cfg.sample_rate.0, output_cfg.sample_rate.0
        ));
    }

    let rate = input_cfg.sample_rate.0;
    config.denoiser.sample_rate = rate;
    let in_channels = input_cfg.channels as usize;
    let out_channels = output_cfg.channels as usize;
    let chunk_frames = ((rate as u64 * config.chunk_ms.max(10) as u64) / 1000).max(1) as usize;
    let queue_capacity = chunk_frames * out_channels * 8;
    let playback = Arc::new(Mutex::new(VecDeque::<f32>::with_capacity(queue_capacity)));
    let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(4);
    let running = Arc::new(AtomicBool::new(true));
    let worker_running = Arc::clone(&running);
    let worker_playback = Arc::clone(&playback);
    let worker = std::thread::spawn(move || {
        while worker_running.load(Ordering::Relaxed) {
            let samples = match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(samples) => samples,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => break,
            };
            let mut channels = vec![Vec::with_capacity(chunk_frames); in_channels];
            for frame in samples.chunks_exact(in_channels) {
                for (channel, sample) in channels.iter_mut().zip(frame) {
                    channel.push(*sample as f64);
                }
            }
            let mut audio = Audio {
                sample_rate: rate,
                channels,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            if let Err(error) = denoise_audio_with_backend_config(
                &mut audio,
                config.denoiser.clone(),
                config.backend,
                &config.backend_options,
            ) {
                eprintln!("denoize: live processing error: {error}");
                continue;
            }
            let frames = audio.frames();
            if let Ok(mut queue) = worker_playback.lock() {
                for frame in 0..frames {
                    for out_ch in 0..out_channels {
                        let source = out_ch.min(audio.channels().saturating_sub(1));
                        if queue.len() == queue_capacity {
                            queue.pop_front();
                        }
                        queue.push_back(audio.channels[source][frame] as f32);
                    }
                }
            }
        }
    });

    let input_stream = build_input(
        &input,
        &input_cfg,
        input_supported.sample_format(),
        tx,
        chunk_frames,
    )?;
    let output_stream = build_output(
        &output,
        &output_cfg,
        output_supported.sample_format(),
        playback,
    )?;
    let signal_running = Arc::clone(&running);
    ctrlc::set_handler(move || signal_running.store(false, Ordering::SeqCst))
        .map_err(|e| format!("install Ctrl-C handler: {e}"))?;
    output_stream
        .play()
        .map_err(|e| format!("start output: {e}"))?;
    input_stream
        .play()
        .map_err(|e| format!("start input: {e}"))?;
    eprintln!("denoize: live at {rate} Hz, {in_channels} input channel(s), {chunk_frames} frames/chunk; press Ctrl-C to stop");
    while running.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(50));
    }
    drop(input_stream);
    drop(output_stream);
    worker
        .join()
        .map_err(|_| "live worker panicked".to_string())?;
    Ok(())
}

fn select_device(
    host: &cpal::Host,
    input: bool,
    requested: Option<&str>,
) -> Result<Device, String> {
    if let Some(name) = requested {
        let devices = if input {
            host.input_devices()
        } else {
            host.output_devices()
        }
        .map_err(|e| format!("enumerate devices: {e}"))?;
        return devices
            .filter_map(|device| device.name().ok().map(|n| (n, device)))
            .find(|(n, _)| n == name)
            .map(|(_, device)| device)
            .ok_or_else(|| {
                format!(
                    "{} device not found: {name}",
                    if input { "input" } else { "output" }
                )
            });
    }
    if input {
        host.default_input_device()
    } else {
        host.default_output_device()
    }
    .ok_or_else(|| {
        format!(
            "no default {} device",
            if input { "input" } else { "output" }
        )
    })
}

fn build_input(
    device: &Device,
    cfg: &StreamConfig,
    format: SampleFormat,
    tx: mpsc::SyncSender<Vec<f32>>,
    chunk_frames: usize,
) -> Result<Stream, String> {
    let channels = cfg.channels as usize;
    let capacity = chunk_frames * channels;
    macro_rules! stream {
        ($ty:ty, $convert:expr) => {{
            let mut pending = Vec::with_capacity(capacity);
            device.build_input_stream(
                cfg,
                move |data: &[$ty], _| {
                    pending.extend(data.iter().map($convert));
                    while pending.len() >= capacity {
                        let tail = pending.split_off(capacity);
                        let chunk = std::mem::replace(&mut pending, tail);
                        let _ = tx.try_send(chunk);
                    }
                },
                |e| eprintln!("denoize: input stream error: {e}"),
                None,
            )
        }};
    }
    let result = match format {
        SampleFormat::F32 => stream!(f32, |x: &f32| *x),
        SampleFormat::I16 => stream!(i16, |x: &i16| *x as f32 / 32768.0),
        SampleFormat::U16 => stream!(u16, |x: &u16| *x as f32 / 32767.5 - 1.0),
        other => return Err(format!("unsupported live input sample format: {other:?}")),
    };
    result.map_err(|e| format!("build input stream: {e}"))
}

fn build_output(
    device: &Device,
    cfg: &StreamConfig,
    format: SampleFormat,
    queue: Arc<Mutex<VecDeque<f32>>>,
) -> Result<Stream, String> {
    macro_rules! stream {
        ($ty:ty, $convert:expr) => {{
            let queue = Arc::clone(&queue);
            device.build_output_stream(
                cfg,
                move |data: &mut [$ty], _| {
                    if let Ok(mut queue) = queue.lock() {
                        for sample in data {
                            *sample = $convert(queue.pop_front().unwrap_or(0.0));
                        }
                    }
                },
                |e| eprintln!("denoize: output stream error: {e}"),
                None,
            )
        }};
    }
    let result = match format {
        SampleFormat::F32 => stream!(f32, |x: f32| x),
        SampleFormat::I16 => stream!(i16, |x: f32| (x.clamp(-1.0, 1.0) * 32767.0) as i16),
        SampleFormat::U16 => stream!(u16, |x: f32| ((x.clamp(-1.0, 1.0) + 1.0) * 32767.5) as u16),
        other => return Err(format!("unsupported live output sample format: {other:?}")),
    };
    result.map_err(|e| format!("build output stream: {e}"))
}
