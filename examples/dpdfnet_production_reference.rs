//! Deterministic export of the compensated production DPDFNet path.

use denoize::{
    write_wav, Audio, BackendOptions, ChannelMode, DenoiserConfig, DpdfnetModel, OnnxModelConfig,
    StreamingBackendSession,
};
use hound::SampleFormat;
use std::env;
use std::path::PathBuf;

const SAMPLE_RATE: u32 = 48_000;
const FIXTURE_FRAMES: usize = 96_137;
const BLOCK_PATTERN: &[usize] = &[1, 127, 480, 1_024, 31, 511, 97];

#[derive(Debug)]
struct Args {
    model: PathBuf,
    input: PathBuf,
    actual: PathBuf,
}

fn main() {
    if env::args()
        .skip(1)
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!("{}", usage());
        return;
    }
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let samples = fixture();
    let input = mono_audio(samples.clone());
    write_wav(&args.input, &input)?;

    let model_config = OnnxModelConfig {
        path: args.model.clone(),
        sample_rate: SAMPLE_RATE,
    };
    let model = DpdfnetModel::load(&model_config)?;
    let options = BackendOptions {
        onnx: Some(model_config),
        deterministic: true,
        channel_mode: ChannelMode::Independent,
        ..BackendOptions::default()
    };
    let mut denoiser = DenoiserConfig::default(SAMPLE_RATE);
    denoiser.vad = false;
    let mut stream = StreamingBackendSession::new_dpdfnet_for_daw_with_prepared_model(
        SAMPLE_RATE,
        1,
        denoiser,
        options,
        &model,
    )?;

    let mut enhanced = Vec::new();
    let mut position = 0usize;
    let mut block = 0usize;
    while position < samples.len() {
        let end = position
            .saturating_add(BLOCK_PATTERN[block % BLOCK_PATTERN.len()])
            .min(samples.len());
        let ready = stream.process_block(&[samples[position..end].to_vec()])?;
        let channel = ready
            .first()
            .ok_or_else(|| "production DPDFNet output has no channel".to_string())?;
        enhanced.extend_from_slice(channel);
        position = end;
        block += 1;
    }
    let tail = stream.finish()?;
    enhanced.extend_from_slice(
        tail.first()
            .ok_or_else(|| "production DPDFNet tail has no channel".to_string())?,
    );
    if enhanced.len() != samples.len() {
        return Err(format!(
            "production DPDFNet output has {} frames, expected {}",
            enhanced.len(),
            samples.len()
        ));
    }
    if enhanced.iter().any(|sample| !sample.is_finite()) {
        return Err("production DPDFNet output contains a non-finite sample".into());
    }
    write_wav(&args.actual, &mono_audio(enhanced))?;
    println!("reference input: {}", args.input.display());
    println!("production output: {}", args.actual.display());
    Ok(())
}

fn fixture() -> Vec<f64> {
    let mut state = 0x4d59_5df4_d0f3_3173u64;
    (0..FIXTURE_FRAMES)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state >> 40) as f64 / ((1u64 << 24) - 1) as f64 * 2.0 - 1.0;
            let time = index as f64 / SAMPLE_RATE as f64;
            let value = 0.24 * (std::f64::consts::TAU * 173.0 * time).sin()
                + 0.11 * (std::f64::consts::TAU * 947.0 * time).sin()
                + 0.07 * noise;
            let pcm = (value.clamp(-1.0, 1.0) * 32_768.0)
                .round()
                .clamp(-32_768.0, 32_767.0) as i16;
            f64::from(pcm) / 32_768.0
        })
        .collect()
}

fn mono_audio(samples: Vec<f64>) -> Audio {
    Audio {
        sample_rate: SAMPLE_RATE,
        channels: vec![samples],
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
        channel_mask: None,
    }
}

fn parse_args() -> Result<Args, String> {
    let mut arguments = env::args().skip(1);
    let mut model = None;
    let mut input = None;
    let mut actual = None;
    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after `{argument}`"))?;
        match argument.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--input" => input = Some(PathBuf::from(value)),
            "--actual" => actual = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument `{argument}`\n{}", usage())),
        }
    }
    let model = model.ok_or_else(|| format!("missing --model\n{}", usage()))?;
    if !model.is_file() {
        return Err(format!("--model is not a file: {}", model.display()));
    }
    Ok(Args {
        model,
        input: input.ok_or_else(|| format!("missing --input\n{}", usage()))?,
        actual: actual.ok_or_else(|| format!("missing --actual\n{}", usage()))?,
    })
}

fn usage() -> &'static str {
    "usage: dpdfnet_production_reference --model MODEL.onnx --input INPUT.wav --actual ACTUAL.wav"
}
