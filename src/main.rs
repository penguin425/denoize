//! `denoize` command-line interface.

use denoize::audio::read_audio;
use denoize::denoiser::{DenoiserConfig, Preset};
use denoize::{
    denoise_file_with_backend_config, Algorithm, Backend, BackendOptions, EncodeOptions,
    OnnxModelConfig, WindowType,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage() -> String {
    let backends = Backend::available_names().join("|");
    format!(
        "\
denoize {VERSION} — pure-Rust audio denoiser engineered for the world's highest sound quality

Classical DSP + optional AI backends (RNNoise, DeepFilterNet v3, MP-SENet, BSRNN).
Input: WAV, MP3, M4A (built-in Pure Rust decode).
Output: WAV, MP3 (shine-rs), M4A (oxideav-aac Pure-Rust AAC-LC).

USAGE:
    denoize <INPUT> <OUTPUT.wav|mp3|m4a> [OPTIONS]

OPTIONS:
    -b, --backend <NAME>     {backends}  (default: classical)
    -a, --algorithm <NAME>   omlsa|logmmse|mmse|wiener|specsub|specsub-nl|specsub-geo
    -p, --preset <NAME>      speech|music|aggressive|gentle|restore|hifi
    -s, --strength <0..1>    denoising strength (default: 0.6)
        --profile <MS>       learn noise from first MS ms (default: auto-detect)
        --no-profile         no profiling; rely on blind IMCRA bootstrap
        --no-adapt           freeze the noise estimate
        --frame <N>          FFT size: 512|1024|2048|4096|8192 (default: 2048)
        --overlap <F>        overlap ratio 0.5..0.95 (default: 0.75)
        --window <NAME>      hann|hamming|sine|blackman|kaiser|flattop|dpss
        --kaiser-beta <B>    Kaiser window beta (default: 8.0)
        --dpss-nw <NW>       DPSS time-bandwidth product (default: 3.0)
        --multiband          enable multiband spectral subtraction
        --perceptual         enable Bark-scale perceptual gain weighting
        --postfilter         enable musical-noise suppression post-filter
        --smoothing <0..1>   gain release smoothing (default: 0.6)
        --makeup <DB>        makeup gain in dB (default: 0.0)
        --no-dc-block        disable DC-blocking pre-filter
        --quality <LEVEL>    high|ultra
        --no-transient       disable transient/onset protection
        --cepstral           enable cepstral gain smoothing
        --no-cepstral        disable cepstral smoothing
        --pre-emphasis       enable pre/de-emphasis
        --no-pre-emphasis    disable pre-emphasis
        --report             print settings report and exit
        --mp3-bitrate <KBPS> MP3 CBR bitrate (default: 192)
        --m4a-bitrate <KBPS> M4A/AAC CBR bitrate (default: 192)
        --onnx-model <PATH>   waveform ONNX model (required for -b onnx)
        --onnx-rate <HZ>      ONNX model sample rate (default: 16000)
    -h, --help               show this help
    -V, --version            show version

BACKENDS (build with --features full for all):
    classical   Enhanced STFT/IMCRA/OMLSA pipeline (default)
    rnnoise     RNNoise via nnnoiseless (requires --features rnnoise)
    deepfilter  DeepFilterNet v3 (requires --features deepfilter)
    onnx        External waveform ONNX model (requires --features onnx)
    mpsenet     MP-SENet magnitude/phase model (requires --features mpsenet)
    bsrnn       ESPnet BSRNN spectral model (requires --features bsrnn)

PRESETS:
    hifi        Flagship transparency: OMLSA + protections + advanced DSP
    speech      Voice-optimised balance
    music       Instruments; enables perceptual + postfilter
"
    )
}

#[derive(Default)]
struct Overrides {
    backend: Option<Backend>,
    algorithm: Option<Algorithm>,
    preset: Option<Preset>,
    strength: Option<f64>,
    profile_ms: Option<f64>,
    no_profile: bool,
    no_adapt: bool,
    frame_size: Option<usize>,
    overlap: Option<f64>,
    window: Option<WindowType>,
    kaiser_beta: Option<f64>,
    dpss_nw: Option<f64>,
    multiband: bool,
    perceptual: bool,
    postfilter: bool,
    smoothing: Option<f64>,
    makeup: Option<f64>,
    no_dc_block: bool,
    report: bool,
    quality: Option<String>,
    no_transient: bool,
    cepstral: bool,
    no_cepstral: bool,
    pre_emphasis: bool,
    no_pre_emphasis: bool,
    mp3_bitrate_kbps: Option<u32>,
    m4a_bitrate_kbps: Option<u32>,
    onnx_model: Option<String>,
    onnx_sample_rate: Option<u32>,
}

fn parse_value<T>(args: &[String], i: &mut usize, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    *i += 1;
    if *i >= args.len() {
        return Err(format!("missing value for {flag}"));
    }
    args[*i]
        .parse::<T>()
        .map_err(|e| format!("invalid value for {flag}: {e}"))
}

fn parse_args(args: &[String]) -> Result<(String, String, Overrides), String> {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut ov = Overrides::default();

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "-h" | "--help" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!("denoize {VERSION}");
                std::process::exit(0);
            }
            "-b" | "--backend" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.backend = Some(Backend::parse(&name).ok_or_else(|| {
                    format!(
                        "unknown backend: {name} (available: {:?})",
                        Backend::available_names()
                    )
                })?);
            }
            "-a" | "--algorithm" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.algorithm = Some(
                    Algorithm::parse(&name).ok_or_else(|| format!("unknown algorithm: {name}"))?,
                );
            }
            "-p" | "--preset" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.preset =
                    Some(Preset::parse(&name).ok_or_else(|| format!("unknown preset: {name}"))?);
            }
            "-s" | "--strength" => ov.strength = Some(parse_value(args, &mut i, a)?),
            "--profile" => ov.profile_ms = Some(parse_value(args, &mut i, a)?),
            "--no-profile" => ov.no_profile = true,
            "--no-adapt" => ov.no_adapt = true,
            "--frame" => ov.frame_size = Some(parse_value(args, &mut i, a)?),
            "--overlap" => ov.overlap = Some(parse_value(args, &mut i, a)?),
            "--window" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.window = Some(
                    WindowType::parse(&name).ok_or_else(|| format!("unknown window: {name}"))?,
                );
            }
            "--kaiser-beta" => ov.kaiser_beta = Some(parse_value(args, &mut i, a)?),
            "--dpss-nw" => ov.dpss_nw = Some(parse_value(args, &mut i, a)?),
            "--multiband" => ov.multiband = true,
            "--perceptual" => ov.perceptual = true,
            "--postfilter" => ov.postfilter = true,
            "--smoothing" => ov.smoothing = Some(parse_value(args, &mut i, a)?),
            "--makeup" => ov.makeup = Some(parse_value(args, &mut i, a)?),
            "--no-dc-block" => ov.no_dc_block = true,
            "--report" => ov.report = true,
            "--quality" => {
                let q: String = parse_value(args, &mut i, a)?;
                ov.quality = Some(q.to_ascii_lowercase());
            }
            "--no-transient" => ov.no_transient = true,
            "--cepstral" => ov.cepstral = true,
            "--no-cepstral" => ov.no_cepstral = true,
            "--pre-emphasis" => ov.pre_emphasis = true,
            "--no-pre-emphasis" => ov.no_pre_emphasis = true,
            "--mp3-bitrate" => ov.mp3_bitrate_kbps = Some(parse_value(args, &mut i, a)?),
            "--m4a-bitrate" => ov.m4a_bitrate_kbps = Some(parse_value(args, &mut i, a)?),
            "--onnx-model" => ov.onnx_model = Some(parse_value(args, &mut i, a)?),
            "--onnx-rate" => ov.onnx_sample_rate = Some(parse_value(args, &mut i, a)?),
            other if other.starts_with('-') => {
                return Err(format!("unknown option: {other}"));
            }
            _ => {
                if input.is_none() {
                    input = Some(a.clone());
                } else if output.is_none() {
                    output = Some(a.clone());
                } else {
                    return Err(format!("unexpected extra argument: {a}"));
                }
            }
        }
        i += 1;
    }

    let input = input.ok_or("missing INPUT")?;
    let output = output.ok_or("missing OUTPUT (.wav|.mp3|.m4a)")?;
    Ok((input, output, ov))
}

fn build_config(ov: &Overrides, sample_rate: u32) -> DenoiserConfig {
    let mut cfg = match ov.preset {
        Some(p) => p.config(sample_rate),
        None => DenoiserConfig::default(sample_rate),
    };
    if let Some(a) = ov.algorithm {
        cfg.algorithm = a;
    }
    if let Some(s) = ov.strength {
        cfg.strength = s;
    }
    if ov.no_profile {
        cfg.profile_ms = -1.0;
    } else if let Some(ms) = ov.profile_ms {
        cfg.profile_ms = ms;
    }
    if ov.no_adapt {
        cfg.adapt = false;
    }
    if let Some(f) = ov.frame_size {
        cfg.frame_size = f;
    }
    if let Some(o) = ov.overlap {
        cfg.overlap = o;
    }
    if let Some(w) = ov.window {
        cfg.window = w;
    }
    if let Some(b) = ov.kaiser_beta {
        cfg.window_params.kaiser_beta = b;
    }
    if let Some(nw) = ov.dpss_nw {
        cfg.window_params.dpss_bandwidth = nw;
    }
    if ov.multiband {
        cfg.multiband = true;
    }
    if ov.perceptual {
        cfg.perceptual_weighting = true;
    }
    if ov.postfilter {
        cfg.musical_noise_postfilter = true;
    }
    if let Some(s) = ov.smoothing {
        cfg.smoothing = s;
    }
    if let Some(m) = ov.makeup {
        cfg.makeup_gain_db = m;
    }
    if ov.no_dc_block {
        cfg.dc_block = false;
    }

    if let Some(ref q) = ov.quality {
        match q.as_str() {
            "high" => {
                if cfg.frame_size < 2048 {
                    cfg.frame_size = 2048;
                }
                if cfg.overlap < 0.8 {
                    cfg.overlap = 0.8;
                }
                cfg.transient_protect = true;
                cfg.cepstral_smoothing = true;
                cfg.perceptual_weighting = true;
                cfg.musical_noise_postfilter = true;
                if !ov.no_pre_emphasis {
                    cfg.pre_emphasis = true;
                }
            }
            "ultra" | "max" | "highest" => {
                cfg.frame_size = cfg.frame_size.max(4096);
                cfg.overlap = 0.875;
                cfg.window = WindowType::Kaiser;
                cfg.window_params.kaiser_beta = 10.0;
                cfg.transient_protect = true;
                cfg.cepstral_smoothing = true;
                cfg.perceptual_weighting = true;
                cfg.musical_noise_postfilter = true;
                cfg.pre_emphasis = true;
                if ov.strength.is_none() && cfg.strength > 0.4 {
                    cfg.strength = 0.32;
                }
            }
            _ => {}
        }
    }

    if ov.no_transient {
        cfg.transient_protect = false;
    }
    if ov.cepstral {
        cfg.cepstral_smoothing = true;
    }
    if ov.no_cepstral {
        cfg.cepstral_smoothing = false;
    }
    if ov.pre_emphasis {
        cfg.pre_emphasis = true;
    }
    if ov.no_pre_emphasis {
        cfg.pre_emphasis = false;
    }

    cfg
}

fn print_report(input: &str, audio: &denoize::Audio, cfg: &DenoiserConfig, backend: Backend) {
    let hop = (cfg.frame_size as f64 * (1.0 - cfg.overlap)).round() as usize;
    let g_min_db = -20.0 - 25.0 * cfg.strength;
    let dur = audio.frames() as f64 / audio.sample_rate as f64;
    println!("input      : {input}");
    println!(
        "format     : {}ch, {:.2}s ({} frames), {} Hz, {}-bit {:?}",
        audio.channels(),
        dur,
        audio.frames(),
        audio.sample_rate,
        audio.bits_per_sample,
        audio.sample_format,
    );
    println!("backend    : {backend:?}");
    println!("algorithm  : {:?}", cfg.algorithm);
    println!(
        "strength   : {:.2}  (gain floor ~{:.0} dB)",
        cfg.strength, g_min_db
    );
    println!(
        "STFT       : frame={}, hop={}, overlap={:.0}%, window={:?}",
        cfg.frame_size,
        hop,
        cfg.overlap * 100.0,
        cfg.window,
    );
    println!(
        "advanced   : multiband={}, perceptual={}, postfilter={}",
        cfg.multiband, cfg.perceptual_weighting, cfg.musical_noise_postfilter
    );
    println!("smoothing  : {:.2}", cfg.smoothing);
    println!(
        "profile    : {}",
        if cfg.profile_ms < 0.0 {
            "disabled".to_string()
        } else if cfg.profile_ms == 0.0 {
            "auto (leading silence)".to_string()
        } else {
            format!("{:.0} ms", cfg.profile_ms)
        }
    );
    println!("adapt      : {}", cfg.adapt);
    println!("dc-block   : {}", cfg.dc_block);
    println!("makeup     : {:.1} dB", cfg.makeup_gain_db);
    println!(
        "hi-fi      : transient={}, cepstral={}, pre-emphasis={}",
        cfg.transient_protect, cfg.cepstral_smoothing, cfg.pre_emphasis
    );
}

fn run(args: &[String]) -> Result<(), String> {
    let (input, output, ov) = parse_args(args)?;
    let audio = read_audio(&input)?;
    let cfg = build_config(&ov, audio.sample_rate);
    let backend = ov.backend.unwrap_or(Backend::Classical);

    if ov.report {
        print_report(&input, &audio, &cfg, backend);
        return Ok(());
    }

    let mut enc = EncodeOptions::default();
    if let Some(kbps) = ov.mp3_bitrate_kbps {
        enc.mp3_bitrate_kbps = kbps;
    }
    if let Some(kbps) = ov.m4a_bitrate_kbps {
        enc.m4a_bitrate_bps = kbps.saturating_mul(1000);
    }

    let backend_options = BackendOptions {
        onnx: ov.onnx_model.map(|path| OnnxModelConfig {
            path: path.into(),
            sample_rate: ov.onnx_sample_rate.unwrap_or(16_000),
        }),
    };
    denoise_file_with_backend_config(&input, &output, cfg, backend, enc, backend_options)?;
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("denoize: error: {e}");
        eprintln!("run 'denoize --help' for usage.");
        std::process::exit(1);
    }
}

#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::*;

    #[test]
    fn parses_onnx_model_options() {
        let args = vec![
            "input.wav".into(),
            "output.wav".into(),
            "--backend".into(),
            "onnx".into(),
            "--onnx-model".into(),
            "model.onnx".into(),
            "--onnx-rate".into(),
            "48000".into(),
        ];
        let (_, _, options) = parse_args(&args).unwrap();
        assert_eq!(options.backend, Some(Backend::Onnx));
        assert_eq!(options.onnx_model.as_deref(), Some("model.onnx"));
        assert_eq!(options.onnx_sample_rate, Some(48_000));
    }
}
