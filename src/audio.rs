//! Audio file I/O: WAV read/write (`hound`) + unified decode for MP3/M4A/WAV.
//!
//! Decoded compressed audio is promoted to `f64` planar PCM at native sample rate
//! (see [`crate::decode`]) before denoising. WAV write preserves bit depth.

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::io::{BufReader, BufWriter, Read, Seek, Write};

/// In-memory audio: one `Vec<f64>` per channel, plus format metadata.
#[derive(Clone, Debug)]
pub struct Audio {
    pub sample_rate: u32,
    pub channels: Vec<Vec<f64>>,
    pub bits_per_sample: u16,
    pub sample_format: SampleFormat,
}

impl Audio {
    pub fn channels(&self) -> usize {
        self.channels.len()
    }

    pub fn frames(&self) -> usize {
        self.channels.first().map(|c| c.len()).unwrap_or(0)
    }

    /// A `WavSpec` matching this audio for writing.
    fn wav_spec(&self) -> WavSpec {
        WavSpec {
            channels: self.channels() as u16,
            sample_rate: self.sample_rate,
            bits_per_sample: self.bits_per_sample,
            sample_format: self.sample_format,
        }
    }
}

/// Read any supported audio file (WAV, MP3, M4A) into de-interleaved `f64` channels.
///
/// Compressed formats are decoded losslessly to float precision (no rate conversion).
pub fn read_audio<P: AsRef<std::path::Path>>(path: P) -> Result<Audio, String> {
    let path = path.as_ref();
    // Keep the original WAV representation so WAV -> WAV processing preserves
    // integer/float sample format and bit depth. Compressed decoders do not
    // have equivalent PCM container metadata and are promoted to f32 PCM.
    let header = {
        use std::io::Read;
        let mut file = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
        let mut header = [0u8; 12];
        let n = file.read(&mut header).map_err(|e| format!("read: {e}"))?;
        header[..n].to_vec()
    };
    if crate::decode::AudioFormat::detect(path, &header) == crate::decode::AudioFormat::Wav {
        return read_wav(path);
    }
    let pcm = crate::decode::decode_file(path)?;
    Ok(pcm.into_audio())
}

/// Read a WAV file into de-interleaved `f64` channels.
pub fn read_wav<P: AsRef<std::path::Path>>(path: P) -> Result<Audio, String> {
    let reader = WavReader::open(&path).map_err(|e| format!("open: {e}"))?;
    read_wav_reader(reader)
}

/// Read WAV data supplied by a pipe or another in-memory source.
pub fn read_wav_bytes(bytes: Vec<u8>) -> Result<Audio, String> {
    let reader = WavReader::new(std::io::Cursor::new(bytes)).map_err(|e| format!("open: {e}"))?;
    read_wav_reader(reader)
}

fn read_wav_reader<R: std::io::Read>(mut reader: WavReader<R>) -> Result<Audio, String> {
    let spec = reader.spec();
    let nchan = spec.channels as usize;
    if nchan == 0 {
        return Err("0 channels".into());
    }

    let max = (1u64 << (spec.bits_per_sample - 1)) as f64; // 2^(bits-1)
    let inv = 1.0 / max;

    let mut channels: Vec<Vec<f64>> = (0..nchan).map(|_| Vec::new()).collect();

    match spec.sample_format {
        SampleFormat::Float => {
            let samples: Result<Vec<f32>, String> = reader
                .samples::<f32>()
                .map(|s| s.map_err(|e| format!("read: {e}")))
                .collect();
            for (i, v) in samples?.iter().enumerate() {
                channels[i % nchan].push((*v as f64).clamp(-1.0, 1.0));
            }
        }
        SampleFormat::Int => {
            if spec.bits_per_sample <= 16 {
                let samples: Result<Vec<i16>, String> = reader
                    .samples::<i16>()
                    .map(|s| s.map_err(|e| format!("read: {e}")))
                    .collect();
                for (i, v) in samples?.iter().enumerate() {
                    channels[i % nchan].push((*v as f64 * inv).clamp(-1.0, 1.0));
                }
            } else {
                let samples: Result<Vec<i32>, String> = reader
                    .samples::<i32>()
                    .map(|s| s.map_err(|e| format!("read: {e}")))
                    .collect();
                for (i, v) in samples?.iter().enumerate() {
                    channels[i % nchan].push((*v as f64 * inv).clamp(-1.0, 1.0));
                }
            }
        }
    }

    Ok(Audio {
        sample_rate: spec.sample_rate,
        channels,
        bits_per_sample: spec.bits_per_sample,
        sample_format: spec.sample_format,
    })
}

/// Write an [`Audio`] to a file; format is inferred from the extension (`.wav`, `.mp3`, `.m4a`).
pub fn write_audio<P: AsRef<std::path::Path>>(
    path: P,
    audio: &Audio,
    options: crate::encode::EncodeOptions,
) -> Result<(), String> {
    crate::encode::write_audio(path, audio, options)
}

/// Write an [`Audio`] to a WAV file, preserving its bit depth / format.
pub fn write_wav<P: AsRef<std::path::Path>>(path: P, audio: &Audio) -> Result<(), String> {
    let spec = audio.wav_spec();
    let writer = WavWriter::create(path, spec).map_err(|e| format!("create: {e}"))?;
    write_wav_writer(writer, audio)
}

/// Encode a complete WAV into memory for stdout and network transports.
pub fn write_wav_bytes(audio: &Audio) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut bytes);
        let writer =
            WavWriter::new(cursor, audio.wav_spec()).map_err(|e| format!("create: {e}"))?;
        write_wav_writer(writer, audio)?;
    }
    Ok(bytes)
}

fn write_wav_writer<W: std::io::Write + std::io::Seek>(
    mut writer: WavWriter<W>,
    audio: &Audio,
) -> Result<(), String> {
    let nchan = audio.channels();
    let frames = audio.frames();

    match audio.sample_format {
        SampleFormat::Float => {
            for f in 0..frames {
                for ch in 0..nchan {
                    let v = audio.channels[ch]
                        .get(f)
                        .copied()
                        .unwrap_or(0.0)
                        .clamp(-1.0, 1.0);
                    writer
                        .write_sample(v as f32)
                        .map_err(|e| format!("write: {e}"))?;
                }
            }
        }
        SampleFormat::Int => {
            let max = (1i64 << (audio.bits_per_sample - 1)) as f64;
            let hi = (max - 1.0) as i64;
            let lo = -max as i64;
            if audio.bits_per_sample <= 16 {
                for f in 0..frames {
                    for ch in 0..nchan {
                        let v = audio.channels[ch]
                            .get(f)
                            .copied()
                            .unwrap_or(0.0)
                            .clamp(-1.0, 1.0);
                        let q = ((v * max).round() as i64).min(hi).max(lo);
                        writer
                            .write_sample(q as i16)
                            .map_err(|e| format!("write: {e}"))?;
                    }
                }
            } else {
                for f in 0..frames {
                    for ch in 0..nchan {
                        let v = audio.channels[ch]
                            .get(f)
                            .copied()
                            .unwrap_or(0.0)
                            .clamp(-1.0, 1.0);
                        let q = ((v * max).round() as i64).min(hi).max(lo);
                        writer
                            .write_sample(q as i32)
                            .map_err(|e| format!("write: {e}"))?;
                    }
                }
            }
        }
    }
    writer.finalize().map_err(|e| format!("finalize: {e}"))?;
    Ok(())
}

/// Block-oriented WAV reader. Samples are returned as planar `f64` channels,
/// keeping at most `max_frames` frames in memory per call.
pub struct WavStreamReader<R: Read + Seek> {
    reader: WavReader<R>,
    spec: WavSpec,
}

impl WavStreamReader<BufReader<std::fs::File>> {
    /// Open a filesystem WAV for bounded-memory reading.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
        let reader = WavReader::new(BufReader::new(file)).map_err(|e| format!("open: {e}"))?;
        Self::from_reader(reader)
    }
}

impl<R: Read + Seek> WavStreamReader<R> {
    /// Wrap an existing seekable WAV source.
    pub fn from_reader(reader: WavReader<R>) -> Result<Self, String> {
        let spec = reader.spec();
        if spec.channels == 0 {
            return Err("0 channels".into());
        }
        if spec.sample_format == SampleFormat::Int && !(1..=32).contains(&spec.bits_per_sample) {
            return Err(format!(
                "unsupported integer WAV bit depth: {}",
                spec.bits_per_sample
            ));
        }
        Ok(Self { reader, spec })
    }

    pub fn spec(&self) -> WavSpec {
        self.spec
    }

    /// Read up to `max_frames`, returning `None` only at clean end-of-file.
    pub fn next_block(&mut self, max_frames: usize) -> Result<Option<Vec<Vec<f64>>>, String> {
        if max_frames == 0 {
            return Err("stream block size must be at least one frame".into());
        }
        let nchan = self.spec.channels as usize;
        let max_samples = max_frames.saturating_mul(nchan);
        let mut interleaved = Vec::with_capacity(max_samples);
        match self.spec.sample_format {
            SampleFormat::Float => {
                for sample in self.reader.samples::<f32>().take(max_samples) {
                    interleaved
                        .push(sample.map_err(|e| format!("read: {e}"))?.clamp(-1.0, 1.0) as f64);
                }
            }
            SampleFormat::Int if self.spec.bits_per_sample <= 16 => {
                let max = (1u64 << (self.spec.bits_per_sample - 1)) as f64;
                let inv = 1.0 / max;
                for sample in self.reader.samples::<i16>().take(max_samples) {
                    interleaved.push(
                        (sample.map_err(|e| format!("read: {e}"))? as f64 * inv).clamp(-1.0, 1.0),
                    );
                }
            }
            SampleFormat::Int => {
                let max = (1u64 << (self.spec.bits_per_sample - 1)) as f64;
                let inv = 1.0 / max;
                for sample in self.reader.samples::<i32>().take(max_samples) {
                    interleaved.push(
                        (sample.map_err(|e| format!("read: {e}"))? as f64 * inv).clamp(-1.0, 1.0),
                    );
                }
            }
        }
        if interleaved.is_empty() {
            return Ok(None);
        }
        if interleaved.len() % nchan != 0 {
            return Err("truncated WAV frame at end of input".into());
        }
        let frames = interleaved.len() / nchan;
        let mut channels: Vec<Vec<f64>> = (0..nchan).map(|_| Vec::with_capacity(frames)).collect();
        for (index, sample) in interleaved.into_iter().enumerate() {
            channels[index % nchan].push(sample);
        }
        Ok(Some(channels))
    }
}

/// Block-oriented WAV writer. The output format is fixed by the supplied
/// [`WavSpec`], so integer bit depth and floating-point WAVs are preserved.
pub struct WavStreamWriter<W: Write + Seek> {
    writer: WavWriter<W>,
    spec: WavSpec,
}

impl WavStreamWriter<BufWriter<std::fs::File>> {
    /// Create a filesystem WAV for bounded-memory writing.
    pub fn create<P: AsRef<std::path::Path>>(path: P, spec: WavSpec) -> Result<Self, String> {
        let file = std::fs::File::create(path).map_err(|e| format!("create: {e}"))?;
        let writer =
            WavWriter::new(BufWriter::new(file), spec).map_err(|e| format!("create: {e}"))?;
        Self::from_writer(writer, spec)
    }
}

impl<W: Write + Seek> WavStreamWriter<W> {
    /// Wrap an existing WAV sink.
    pub fn from_writer(writer: WavWriter<W>, spec: WavSpec) -> Result<Self, String> {
        if spec.channels == 0 {
            return Err("0 channels".into());
        }
        if spec.sample_format == SampleFormat::Int && !(1..=32).contains(&spec.bits_per_sample) {
            return Err(format!(
                "unsupported integer WAV bit depth: {}",
                spec.bits_per_sample
            ));
        }
        Ok(Self { writer, spec })
    }

    /// Write a planar block, interleaving it in the WAV container.
    pub fn write_block(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        let nchan = self.spec.channels as usize;
        if channels.len() != nchan {
            return Err(format!("expected {nchan} channels, got {}", channels.len()));
        }
        let frames = channels.first().map(Vec::len).unwrap_or(0);
        if channels.iter().any(|channel| channel.len() != frames) {
            return Err("stream blocks must have equal channel lengths".into());
        }
        match self.spec.sample_format {
            SampleFormat::Float => {
                for frame in 0..frames {
                    for channel in channels {
                        self.writer
                            .write_sample(channel[frame].clamp(-1.0, 1.0) as f32)
                            .map_err(|e| format!("write: {e}"))?;
                    }
                }
            }
            SampleFormat::Int if self.spec.bits_per_sample <= 16 => {
                let max = (1i64 << (self.spec.bits_per_sample - 1)) as f64;
                let hi = (max - 1.0) as i64;
                let lo = -max as i64;
                for frame in 0..frames {
                    for channel in channels {
                        let value = channel[frame].clamp(-1.0, 1.0);
                        let quantized = ((value * max).round() as i64).min(hi).max(lo);
                        self.writer
                            .write_sample(quantized as i16)
                            .map_err(|e| format!("write: {e}"))?;
                    }
                }
            }
            SampleFormat::Int => {
                let max = (1i64 << (self.spec.bits_per_sample - 1)) as f64;
                let hi = (max - 1.0) as i64;
                let lo = -max as i64;
                for frame in 0..frames {
                    for channel in channels {
                        let value = channel[frame].clamp(-1.0, 1.0);
                        let quantized = ((value * max).round() as i64).min(hi).max(lo);
                        self.writer
                            .write_sample(quantized as i32)
                            .map_err(|e| format!("write: {e}"))?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Finalize the WAV header and flush the underlying sink.
    pub fn finalize(self) -> Result<(), String> {
        self.writer.finalize().map_err(|e| format!("finalize: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("denoize_audio_{}_{}", std::process::id(), name));
        p
    }

    #[test]
    fn wav_16bit_roundtrip() {
        let path = tmp("rt16.wav");
        let sr = 16000u32;
        let spec = WavSpec {
            channels: 1,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        let mut signal = Vec::new();
        for i in 0..sr as usize {
            let v = (2.0 * std::f64::consts::PI * 220.0 * i as f64 / sr as f64).sin() * 0.5;
            signal.push(v);
            w.write_sample((v * 32767.0) as i16).unwrap();
        }
        w.finalize().unwrap();

        let audio = read_wav(&path).unwrap();
        assert_eq!(audio.sample_rate, sr);
        assert_eq!(audio.channels(), 1);
        assert_eq!(audio.frames(), sr as usize);
        for (i, &sig) in signal.iter().enumerate() {
            assert!((audio.channels[0][i] - sig).abs() < 1e-3, "@{i}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_audio_preserves_wav_format() {
        let path = tmp("preserve16.wav");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        w.write_sample(123i16).unwrap();
        w.finalize().unwrap();

        let audio = read_audio(&path).unwrap();
        assert_eq!(audio.bits_per_sample, 16);
        assert_eq!(audio.sample_format, SampleFormat::Int);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn wav_stream_reader_and_writer_roundtrip_blocks() {
        let input = tmp("stream_in.wav");
        let output = tmp("stream_out.wav");
        let spec = WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(&input, spec).unwrap();
        for frame in 0..257 {
            writer.write_sample((frame as i16).wrapping_mul(3)).unwrap();
            writer
                .write_sample(-(frame as i16).wrapping_mul(2))
                .unwrap();
        }
        writer.finalize().unwrap();

        let mut reader = WavStreamReader::open(&input).unwrap();
        assert_eq!(reader.spec(), spec);
        let first = reader.next_block(100).unwrap().unwrap();
        assert_eq!(
            first.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![100, 100]
        );
        let second = reader.next_block(100).unwrap().unwrap();
        assert_eq!(
            second.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![100, 100]
        );
        let third = reader.next_block(100).unwrap().unwrap();
        assert_eq!(third.iter().map(Vec::len).collect::<Vec<_>>(), vec![57, 57]);
        assert!(reader.next_block(100).unwrap().is_none());

        let mut stream_writer = WavStreamWriter::create(&output, spec).unwrap();
        stream_writer.write_block(&first).unwrap();
        stream_writer.write_block(&second).unwrap();
        stream_writer.write_block(&third).unwrap();
        stream_writer.finalize().unwrap();
        let roundtrip = read_wav(&output).unwrap();
        assert_eq!(roundtrip.frames(), 257);
        assert_eq!(roundtrip.channels(), 2);
        assert!((roundtrip.channels[0][120] - (360.0 / 32_768.0)).abs() < 1e-5);
        assert!((roundtrip.channels[1][120] + (240.0 / 32_768.0)).abs() < 1e-5);

        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }
}
