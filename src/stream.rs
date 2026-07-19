//! Stateful block-processing API for realtime and pipe integrations.

/// A resettable audio processor with a fixed input/output block size.
pub trait StreamProcessor {
    fn sample_rate(&self) -> u32;
    fn channels(&self) -> usize;
    fn block_size(&self) -> usize;
    fn process_block(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String>;
    fn flush(&mut self) -> Result<Vec<Vec<f64>>, String>;
    fn reset(&mut self);
}

#[cfg(feature = "gtcrn")]
pub struct GtcrnProcessor {
    streams: Vec<crate::backend::gtcrn::GtcrnStream>,
}

#[cfg(feature = "gtcrn")]
impl GtcrnProcessor {
    pub fn open(path: &std::path::Path, channels: usize) -> Result<Self, String> {
        if channels == 0 {
            return Err("stream must have at least one channel".into());
        }
        let mut streams = Vec::with_capacity(channels);
        for _ in 0..channels {
            streams.push(crate::backend::gtcrn::GtcrnStream::open(path)?);
        }
        Ok(Self { streams })
    }
}

#[cfg(feature = "gtcrn")]
impl StreamProcessor for GtcrnProcessor {
    fn sample_rate(&self) -> u32 {
        crate::backend::gtcrn::SAMPLE_RATE
    }

    fn channels(&self) -> usize {
        self.streams.len()
    }

    fn block_size(&self) -> usize {
        crate::backend::gtcrn::HOP_SIZE
    }

    fn process_block(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if channels.len() != self.streams.len() {
            return Err(format!(
                "expected {} channels, got {}",
                self.streams.len(),
                channels.len()
            ));
        }
        self.streams
            .iter_mut()
            .zip(channels)
            .map(|(stream, channel)| {
                let hop: [f32; crate::backend::gtcrn::HOP_SIZE] = channel
                    .iter()
                    .map(|sample| *sample as f32)
                    .collect::<Vec<_>>()
                    .try_into()
                    .map_err(|samples: Vec<f32>| {
                        format!(
                            "expected {} frames, got {}",
                            crate::backend::gtcrn::HOP_SIZE,
                            samples.len()
                        )
                    })?;
                Ok(stream
                    .process_hop(&hop)?
                    .into_iter()
                    .map(|sample| sample as f64)
                    .collect())
            })
            .collect()
    }

    fn flush(&mut self) -> Result<Vec<Vec<f64>>, String> {
        self.streams
            .iter_mut()
            .map(|stream| {
                Ok(stream
                    .flush()?
                    .into_iter()
                    .map(|sample| sample as f64)
                    .collect())
            })
            .collect()
    }

    fn reset(&mut self) {
        for stream in &mut self.streams {
            stream.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Gain(f64);

    impl StreamProcessor for Gain {
        fn sample_rate(&self) -> u32 {
            48_000
        }
        fn channels(&self) -> usize {
            1
        }
        fn block_size(&self) -> usize {
            2
        }
        fn process_block(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
            Ok(channels
                .iter()
                .map(|channel| channel.iter().map(|x| x * self.0).collect())
                .collect())
        }
        fn flush(&mut self) -> Result<Vec<Vec<f64>>, String> {
            Ok(vec![Vec::new()])
        }
        fn reset(&mut self) {}
    }

    #[test]
    fn trait_supports_stateful_block_processors() {
        let mut processor: Box<dyn StreamProcessor> = Box::new(Gain(0.5));
        assert_eq!(
            processor.process_block(&[vec![1.0, -1.0]]).unwrap(),
            vec![vec![0.5, -0.5]]
        );
    }
}
