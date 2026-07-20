//! `denoize` command-line interface.

use denoize::audio::{read_audio, read_wav_bytes, write_audio, write_wav_bytes};
use denoize::denoiser::{DenoiserConfig, Preset, ProcessingMode};
use denoize::{
    denoise_audio_with_backend_config, AacEncoder, Algorithm, Backend, BackendOptions, ChannelMode,
    EncodeOptions, OnnxModelConfig, SgmseProfile, WindowType,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage() -> String {
    let backends = Backend::available_names().join("|");
    format!(
        "\
denoize {VERSION} — pure-Rust audio denoiser engineered for the world's highest sound quality

Classical DSP + optional AI backends (RNNoise, DeepFilterNet v3, MP-SENet, BSRNN).
Input/output: WAV, FLAC, Ogg Opus, MP3, M4A (built in; no ffmpeg).

USAGE:
    denoize <INPUT> <OUTPUT.wav|flac|opus|ogg|mp3|m4a|aac> [OPTIONS]
    denoize live [--input-device NAME] [--output-device NAME] [OPTIONS]
    denoize live --list-devices
    denoize models <list|info|install|update|verify|remove|path|cache-dir> [MODEL|all]
    denoize metrics <REFERENCE> <TEST> [--json|--markdown]

OPTIONS:
    -b, --backend <NAME>     auto|{backends}  (default: classical)
    -a, --algorithm <NAME>   omlsa|logmmse|mmse|wiener|specsub|specsub-nl|specsub-geo
    -p, --preset <NAME>      speech|music|aggressive|gentle|restore|hifi
        --mode <NAME>        speech|music|ambient processing intent
    -s, --strength <0..1>    denoising strength (default: 0.6)
        --profile <MS>       learn noise from first MS ms (default: auto-detect)
        --no-profile         no profiling; rely on blind IMCRA bootstrap
        --no-adapt           freeze the noise estimate
        --adaptive-noise     learn noise from noise-only regions throughout the file
        --vad                speech-aware segmentation and silence suppression
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
        --aac-encoder <NAME> oxide|fdk (default: oxide)
        --loudness <LUFS>     normalize integrated loudness after denoising
        --true-peak <DBTP>    true-peak ceiling with --loudness (default: -1)
        --onnx-model <PATH>   waveform ONNX model (required for -b onnx)
        --onnx-rate <HZ>      ONNX model sample rate (default: 16000)
        --channels <MODE>     independent|linked|mid-side (default: independent)
        --sgmse-profile <P>   fast|balanced|quality (default: balanced)
        --batch               process files in INPUT directory into OUTPUT directory
        --recursive           include subdirectories in batch mode
        --jobs <N>            concurrent batch workers (default: CPU count)
        --output-format <EXT> convert every batch output to this format
        --force               allow replacing existing output files
        --json                emit a machine-readable result
        --no-metadata         do not copy input tags/artwork to the output
        --input-device <NAME> live capture device (default: system default)
        --output-device <NAME> live playback device (default: system default)
        --chunk-ms <MS>       live processing chunk duration (default: 100)
    -h, --help               show this help
    -V, --version            show version

BACKENDS (build with --features full for all):
    classical   Enhanced STFT/IMCRA/OMLSA pipeline (default)
    rnnoise     RNNoise via nnnoiseless (requires --features rnnoise)
    deepfilter  DeepFilterNet v3 (requires --features deepfilter)
    onnx        External waveform ONNX model (requires --features onnx)
    mpsenet     MP-SENet magnitude/phase model (requires --features mpsenet)
    bsrnn       ESPnet BSRNN spectral model (requires --features bsrnn)
    mossformer2 ClearerVoice MossFormer2 model (requires --features mossformer2)
    sgmse       SGMSE+ diffusion model (requires --features sgmse)
    gtcrn       Official low-complexity streaming GTCRN (requires --features gtcrn)

PRESETS:
    hifi        Flagship transparency: OMLSA + protections + advanced DSP
    speech      Voice-optimised balance
    music       Instruments; enables perceptual + postfilter
"
    )
}

#[derive(Clone, Default)]
struct Overrides {
    backend: Option<Backend>,
    auto_backend: bool,
    algorithm: Option<Algorithm>,
    preset: Option<Preset>,
    mode: Option<ProcessingMode>,
    strength: Option<f64>,
    profile_ms: Option<f64>,
    no_profile: bool,
    no_adapt: bool,
    adaptive_noise: bool,
    vad: bool,
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
    aac_encoder: Option<AacEncoder>,
    loudness_lufs: Option<f64>,
    true_peak_dbtp: Option<f64>,
    onnx_model: Option<String>,
    onnx_sample_rate: Option<u32>,
    channel_mode: Option<ChannelMode>,
    sgmse_profile: Option<SgmseProfile>,
    batch: bool,
    recursive: bool,
    jobs: Option<usize>,
    output_format: Option<String>,
    force: bool,
    json: bool,
    no_metadata: bool,
    input_device: Option<String>,
    output_device: Option<String>,
    chunk_ms: Option<u32>,
    list_devices: bool,
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
                if name.eq_ignore_ascii_case("auto") {
                    ov.auto_backend = true;
                    i += 1;
                    continue;
                }
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
            "--mode" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.mode = Some(ProcessingMode::parse(&name).ok_or_else(|| {
                    format!("unknown mode: {name} (expected speech, music, or ambient)")
                })?);
            }
            "-s" | "--strength" => ov.strength = Some(parse_value(args, &mut i, a)?),
            "--profile" => ov.profile_ms = Some(parse_value(args, &mut i, a)?),
            "--no-profile" => ov.no_profile = true,
            "--no-adapt" => ov.no_adapt = true,
            "--adaptive-noise" => ov.adaptive_noise = true,
            "--vad" => ov.vad = true,
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
            "--aac-encoder" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.aac_encoder = Some(AacEncoder::parse(&name).ok_or_else(|| {
                    format!("unknown AAC encoder: {name} (expected oxide or fdk)")
                })?);
            }
            "--loudness" => ov.loudness_lufs = Some(parse_value(args, &mut i, a)?),
            "--true-peak" => ov.true_peak_dbtp = Some(parse_value(args, &mut i, a)?),
            "--onnx-model" => ov.onnx_model = Some(parse_value(args, &mut i, a)?),
            "--onnx-rate" => ov.onnx_sample_rate = Some(parse_value(args, &mut i, a)?),
            "--channels" => {
                let mode: String = parse_value(args, &mut i, a)?;
                ov.channel_mode = Some(ChannelMode::parse(&mode).ok_or_else(|| {
                    format!(
                        "unknown channel mode: {mode} (expected independent, linked, or mid-side)"
                    )
                })?);
            }
            "--sgmse-profile" => {
                let profile: String = parse_value(args, &mut i, a)?;
                ov.sgmse_profile = Some(SgmseProfile::parse(&profile).ok_or_else(|| {
                    format!(
                        "unknown SGMSE profile: {profile} (expected fast, balanced, or quality)"
                    )
                })?);
            }
            "--batch" => ov.batch = true,
            "--recursive" => ov.recursive = true,
            "--jobs" => ov.jobs = Some(parse_value(args, &mut i, a)?),
            "--output-format" => ov.output_format = Some(parse_value(args, &mut i, a)?),
            "--force" => ov.force = true,
            "--json" => ov.json = true,
            "--no-metadata" => ov.no_metadata = true,
            "--input-device" => ov.input_device = Some(parse_value(args, &mut i, a)?),
            "--output-device" => ov.output_device = Some(parse_value(args, &mut i, a)?),
            "--chunk-ms" => ov.chunk_ms = Some(parse_value(args, &mut i, a)?),
            "--list-devices" => ov.list_devices = true,
            "-" => {
                if input.is_none() {
                    input = Some(a.clone());
                } else if output.is_none() {
                    output = Some(a.clone());
                } else {
                    return Err("unexpected extra argument: -".into());
                }
            }
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
    let output = output.ok_or("missing OUTPUT audio path")?;
    Ok((input, output, ov))
}

fn build_config(ov: &Overrides, sample_rate: u32) -> DenoiserConfig {
    let mut cfg = match ov.preset {
        Some(p) => p.config(sample_rate),
        None => DenoiserConfig::default(sample_rate),
    };
    if let Some(mode) = ov.mode {
        mode.apply(&mut cfg);
    }
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
    if ov.adaptive_noise {
        cfg.adaptive_noise = true;
    }
    if ov.vad {
        cfg.vad = true;
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
    println!("adaptive-profile: {}", cfg.adaptive_noise);
    println!("dc-block   : {}", cfg.dc_block);
    println!("makeup     : {:.1} dB", cfg.makeup_gain_db);
    println!(
        "hi-fi      : transient={}, cepstral={}, pre-emphasis={}",
        cfg.transient_protect, cfg.cepstral_smoothing, cfg.pre_emphasis
    );
}

fn run(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) == Some("live") {
        return run_live(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("models") {
        return run_models(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("metrics") {
        return run_metrics(&args[1..]);
    }
    let (input, output, ov) = parse_args(args)?;
    if ov.batch {
        return run_batch(&input, &output, &ov);
    }
    run_one(&input, &output, ov)
}

#[cfg(feature = "live")]
fn run_live(args: &[String]) -> Result<(), String> {
    let mut parseable = vec!["-".to_string(), "-".to_string()];
    parseable.extend_from_slice(args);
    let (_, _, ov) = parse_args(&parseable)?;
    if ov.list_devices {
        let (inputs, outputs) = denoize::live::device_names()?;
        println!("Input devices:");
        for device in inputs {
            println!("  {device}");
        }
        println!("Output devices:");
        for device in outputs {
            println!("  {device}");
        }
        return Ok(());
    }
    let backend = if ov.auto_backend {
        select_live_backend()
    } else {
        ov.backend.unwrap_or(Backend::Classical)
    };
    let sample_rate = 48_000;
    let denoiser = build_config(&ov, sample_rate);
    let backend_options = BackendOptions {
        onnx: ov.onnx_model.map(|path| OnnxModelConfig {
            path: path.into(),
            sample_rate: ov.onnx_sample_rate.unwrap_or(16_000),
        }),
        channel_mode: ov.channel_mode.unwrap_or_default(),
        sgmse_profile: ov.sgmse_profile.unwrap_or_default(),
    };
    denoize::live::run(denoize::live::LiveConfig {
        input_device: ov.input_device,
        output_device: ov.output_device,
        chunk_ms: ov.chunk_ms.unwrap_or(100),
        backend,
        backend_options,
        denoiser,
    })
}

#[cfg(not(feature = "live"))]
fn run_live(_args: &[String]) -> Result<(), String> {
    Err("live audio is unavailable in this build; rebuild with --features live".into())
}

fn run_one(input: &str, output: &str, ov: Overrides) -> Result<(), String> {
    let metadata = if input != "-" && !ov.no_metadata {
        denoize::metadata::read(std::path::Path::new(input))?
    } else {
        None
    };
    let mut audio = if input == "-" {
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut std::io::stdin(), &mut bytes)
            .map_err(|error| format!("failed to read stdin: {error}"))?;
        read_wav_bytes(bytes)?
    } else {
        read_audio(input)?
    };
    let cfg = build_config(&ov, audio.sample_rate);
    let backend = if ov.auto_backend {
        select_auto_backend(
            audio.frames() as f64 / audio.sample_rate as f64,
            ov.quality.as_deref(),
        )
    } else {
        ov.backend.unwrap_or(Backend::Classical)
    };
    if ov.auto_backend && !ov.json {
        eprintln!("denoize: auto-selected backend {}", backend_name(backend));
    }

    if ov.report {
        print_report(input, &audio, &cfg, backend);
        return Ok(());
    }
    if output != "-" && std::path::Path::new(output).exists() && !ov.force {
        return Err(format!(
            "output already exists: {output} (use --force to replace it)"
        ));
    }

    let mut enc = EncodeOptions::default();
    if let Some(kbps) = ov.mp3_bitrate_kbps {
        enc.mp3_bitrate_kbps = kbps;
    }
    if let Some(kbps) = ov.m4a_bitrate_kbps {
        enc.m4a_bitrate_bps = kbps.saturating_mul(1000);
    }
    if let Some(encoder) = ov.aac_encoder {
        enc.aac_encoder = encoder;
    }

    #[allow(unused_mut)]
    let mut backend_options = BackendOptions {
        onnx: ov.onnx_model.map(|path| OnnxModelConfig {
            path: path.into(),
            sample_rate: ov.onnx_sample_rate.unwrap_or(16_000),
        }),
        channel_mode: ov.channel_mode.unwrap_or_default(),
        sgmse_profile: ov.sgmse_profile.unwrap_or_default(),
    };
    #[cfg(feature = "gtcrn")]
    if backend == Backend::Gtcrn && backend_options.onnx.is_none() {
        let model = denoize::models::find("gtcrn").expect("built-in GTCRN manifest entry");
        backend_options.onnx = Some(OnnxModelConfig {
            path: denoize::models::verify(model).map_err(|_| {
                "GTCRN model is not installed; run `denoize models install gtcrn`".to_string()
            })?,
            sample_rate: model.sample_rate,
        });
    }
    let elapsed = denoise_audio_with_backend_config(&mut audio, cfg, backend, &backend_options)?;
    if let Some(target) = ov.loudness_lufs {
        let report =
            denoize::loudness::normalize(&mut audio, target, ov.true_peak_dbtp.unwrap_or(-1.0))?;
        if !ov.json {
            eprintln!(
                "denoize: loudness {:.2} -> {:.2} LUFS, true peak {:.2} dBTP, gain {:+.2} dB",
                report.input_lufs, report.output_lufs, report.true_peak_dbtp, report.gain_db
            );
        }
    } else if ov.true_peak_dbtp.is_some() {
        return Err("--true-peak requires --loudness".into());
    }
    if output == "-" {
        let bytes = write_wav_bytes(&audio)?;
        std::io::Write::write_all(&mut std::io::stdout(), &bytes)
            .map_err(|error| format!("failed to write stdout: {error}"))?;
    } else {
        let output_path = std::path::Path::new(output);
        let filename = output_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("invalid output filename")?;
        let temporary = output_path.with_file_name(format!(".denoize-{filename}.part"));
        // Preserve the real codec extension on the temporary file.
        let temporary = temporary.with_extension(
            output_path
                .extension()
                .and_then(|x| x.to_str())
                .unwrap_or("wav"),
        );
        if let Err(error) = write_audio(&temporary, &audio, enc) {
            let _ = std::fs::remove_file(&temporary);
            return Err(error);
        }
        if output_path.exists() {
            std::fs::remove_file(output_path).map_err(|e| format!("replace output: {e}"))?;
        }
        std::fs::rename(&temporary, output_path).map_err(|e| format!("commit output: {e}"))?;
        if let Some(metadata) = metadata {
            denoize::metadata::write(metadata, output_path)?;
        }
        if ov.json {
            println!("{{\"input\":{:?},\"output\":{:?},\"backend\":{:?},\"channels\":{},\"frames\":{},\"sample_rate\":{},\"elapsed_ms\":{:.3}}}", input, output, format!("{backend:?}").to_ascii_lowercase(), audio.channels(), audio.frames(), audio.sample_rate, elapsed.as_secs_f64() * 1000.0);
        }
    }
    Ok(())
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Classical => "classical",
        #[cfg(feature = "rnnoise")]
        Backend::Rnnoise => "rnnoise",
        #[cfg(feature = "deepfilter")]
        Backend::DeepFilter => "deepfilter",
        #[cfg(feature = "onnx")]
        Backend::Onnx => "onnx",
        #[cfg(feature = "mpsenet")]
        Backend::MpSenet => "mpsenet",
        #[cfg(feature = "bsrnn")]
        Backend::Bsrnn => "bsrnn",
        #[cfg(feature = "mossformer2")]
        Backend::Mossformer2 => "mossformer2",
        #[cfg(feature = "sgmse")]
        Backend::Sgmse => "sgmse",
        #[cfg(feature = "gtcrn")]
        Backend::Gtcrn => "gtcrn",
    }
}

/// Choose the strongest built-in backend whose expected cost fits the request.
fn select_auto_backend(_duration_seconds: f64, _quality: Option<&str>) -> Backend {
    #[cfg(feature = "deepfilter")]
    {
        let high_quality = matches!(_quality, Some("high" | "ultra" | "max" | "highest"));
        if high_quality || _duration_seconds <= 10.0 * 60.0 {
            return Backend::DeepFilter;
        }
    }
    #[cfg(feature = "rnnoise")]
    {
        return Backend::Rnnoise;
    }
    #[allow(unreachable_code)]
    Backend::Classical
}

#[cfg(feature = "live")]
fn select_live_backend() -> Backend {
    #[cfg(feature = "rnnoise")]
    {
        return Backend::Rnnoise;
    }
    #[allow(unreachable_code)]
    Backend::Classical
}

fn run_batch(input: &str, output: &str, ov: &Overrides) -> Result<(), String> {
    use rayon::prelude::*;

    let input_dir = std::path::Path::new(input);
    let output_dir = std::path::Path::new(output);
    if !input_dir.is_dir() {
        return Err(format!("batch input is not a directory: {input}"));
    }
    if let Some(jobs) = ov.jobs {
        if jobs == 0 {
            return Err("--jobs must be at least 1".into());
        }
    }
    let output_extension = ov
        .output_format
        .as_deref()
        .map(normalize_output_extension)
        .transpose()?;
    std::fs::create_dir_all(output_dir).map_err(|e| format!("create batch output: {e}"))?;
    let files = collect_batch_files(input_dir, ov.recursive)?;
    if files.is_empty() {
        return Err("batch input contains no supported audio files".into());
    }
    if let Some(extension) = output_extension {
        let mut destinations = std::collections::HashSet::new();
        for path in &files {
            let relative = path.strip_prefix(input_dir).map_err(|e| e.to_string())?;
            let mut destination = output_dir.join(relative);
            destination.set_extension(extension);
            if !destinations.insert(destination.clone()) {
                return Err(format!(
                    "multiple inputs map to the same batch output: {}",
                    destination.display()
                ));
            }
        }
    }

    let process = || {
        files
            .par_iter()
            .enumerate()
            .map(|(index, path)| {
                let relative = path.strip_prefix(input_dir).map_err(|e| e.to_string())?;
                let mut destination = output_dir.join(relative);
                if let Some(extension) = output_extension {
                    destination.set_extension(extension);
                }
                if let Some(parent) = destination.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("create {}: {e}", parent.display()))?;
                }
                eprintln!(
                    "denoize: batch {}/{} {}",
                    index + 1,
                    files.len(),
                    path.display()
                );
                let mut options = ov.clone();
                options.batch = false;
                options.json = false;
                run_one(
                    &path.to_string_lossy(),
                    &destination.to_string_lossy(),
                    options,
                )?;
                Ok::<_, String>((path.clone(), destination))
            })
            .collect::<Vec<_>>()
    };
    let results = if let Some(jobs) = ov.jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build()
            .map_err(|e| format!("create batch worker pool: {e}"))?
            .install(process)
    } else {
        process()
    };
    let succeeded = results.iter().filter(|result| result.is_ok()).count();
    let failures: Vec<_> = results
        .iter()
        .filter_map(|result| result.as_ref().err())
        .collect();
    if ov.json {
        println!(
            "{{\"total\":{},\"succeeded\":{},\"failed\":{},\"output\":{:?}}}",
            files.len(),
            succeeded,
            failures.len(),
            output
        );
    } else {
        eprintln!(
            "denoize: batch complete: {succeeded} succeeded, {} failed",
            failures.len()
        );
        for error in &failures {
            eprintln!("denoize: batch error: {error}");
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!("{} batch file(s) failed", failures.len()))
    }
}

fn collect_batch_files(
    root: &std::path::Path,
    recursive: bool,
) -> Result<Vec<std::path::PathBuf>, String> {
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory)
            .map_err(|e| format!("read batch input {}: {e}", directory.display()))?
        {
            let path = entry.map_err(|e| format!("read batch entry: {e}"))?.path();
            if path.is_dir() && recursive {
                pending.push(path);
            } else if path.is_file() && is_supported_audio_path(&path) {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn is_supported_audio_path(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "wav" | "mp3" | "m4a" | "aac" | "flac" | "opus" | "ogg"
            )
        })
        .unwrap_or(false)
}

fn normalize_output_extension(value: &str) -> Result<&str, String> {
    let extension = value.trim_start_matches('.');
    if matches!(
        extension.to_ascii_lowercase().as_str(),
        "wav" | "mp3" | "m4a" | "aac" | "flac" | "opus" | "ogg"
    ) {
        Ok(extension)
    } else {
        Err(format!("unsupported --output-format: {value}"))
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    fn temporary_directory() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "denoize-batch-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn batch_collection_is_recursive_and_sorted() {
        let root = temporary_directory();
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(root.join("b.wav"), []).unwrap();
        std::fs::write(root.join("ignore.txt"), []).unwrap();
        std::fs::write(nested.join("a.FLAC"), []).unwrap();

        assert_eq!(
            collect_batch_files(&root, false).unwrap(),
            vec![root.join("b.wav")]
        );
        assert_eq!(
            collect_batch_files(&root, true).unwrap(),
            vec![root.join("b.wav"), nested.join("a.FLAC")]
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn validates_batch_output_format() {
        assert_eq!(normalize_output_extension(".flac").unwrap(), "flac");
        assert_eq!(normalize_output_extension("aac").unwrap(), "aac");
        assert!(normalize_output_extension("wma").is_err());
    }

    #[test]
    fn batch_processes_nested_audio_and_converts_format() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(input.join("nested")).unwrap();
        let audio = denoize::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 3_200]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        denoize::write_audio(
            input.join("nested/sample.wav"),
            &audio,
            EncodeOptions::default(),
        )
        .unwrap();
        let options = Overrides {
            batch: true,
            recursive: true,
            jobs: Some(2),
            output_format: Some("flac".into()),
            ..Overrides::default()
        };

        run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();
        assert!(output.join("nested/sample.flac").is_file());
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod auto_backend_tests {
    use super::*;

    #[test]
    fn parses_auto_backend() {
        let (_, _, options) = parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--backend".into(),
            "auto".into(),
        ])
        .unwrap();
        assert!(options.auto_backend);
        assert!(options.backend.is_none());
    }

    #[test]
    fn automatic_selection_uses_an_available_backend() {
        let selected = select_auto_backend(30.0, None);
        assert!(Backend::available_names().contains(&backend_name(selected)));
    }
}

fn run_metrics(args: &[String]) -> Result<(), String> {
    let reference = args.first().ok_or("metrics requires REFERENCE and TEST")?;
    let test = args.get(1).ok_or("metrics requires REFERENCE and TEST")?;
    let report =
        denoize::benchmark::BenchmarkReport::compare(&read_audio(reference)?, &read_audio(test)?)?;
    if args.iter().any(|argument| argument == "--json") {
        println!("{}", report.json());
    } else {
        println!("{}", report.markdown());
    }
    Ok(())
}

fn run_models(args: &[String]) -> Result<(), String> {
    let command = args.first().map(String::as_str).unwrap_or("list");
    if command == "list" {
        println!("NAME\tBACKEND\tRATE\tLICENSE\tSTATUS");
        for model in denoize::models::MODELS {
            let status = if denoize::models::verify(model).is_ok() {
                "installed"
            } else {
                "not-installed"
            };
            println!(
                "{}\t{}\t{}\t{}\t{}",
                model.name, model.backend, model.sample_rate, model.license, status
            );
        }
        return Ok(());
    }
    if command == "cache-dir" {
        println!("{}", denoize::models::cache_dir()?.display());
        return Ok(());
    }
    let name = args
        .get(1)
        .ok_or_else(|| format!("models {command} requires MODEL"))?;
    let models: Vec<_> = if name == "all" {
        denoize::models::MODELS.iter().collect()
    } else {
        vec![denoize::models::find(name)
            .ok_or_else(|| format!("unknown model: {name} (run `denoize models list`)"))?]
    };
    for model in models {
        match command {
            "info" => {
                println!("name: {}", model.name);
                println!("backend: {}", model.backend);
                println!("sample-rate: {}", model.sample_rate);
                println!("license: {}", model.license);
                println!("revision: {}", model.revision);
                println!("sha256: {}", model.sha256);
                println!("url: {}", model.url);
                println!("path: {}", denoize::models::path(model)?.display());
            }
            "install" => println!("{}", denoize::models::install(model)?.display()),
            "update" => println!("{}", denoize::models::update(model)?.display()),
            "verify" => println!("verified {}", denoize::models::verify(model)?.display()),
            "remove" => println!(
                "{} {}",
                if denoize::models::remove(model)? {
                    "removed"
                } else {
                    "not-installed"
                },
                model.name
            ),
            "path" => println!("{}", denoize::models::path(model)?.display()),
            _ => return Err(format!("unknown models command: {command}")),
        }
    }
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

    #[test]
    fn parses_live_device_options() {
        let args = vec![
            "-".into(),
            "-".into(),
            "--input-device".into(),
            "Mic".into(),
            "--output-device".into(),
            "Cable".into(),
            "--chunk-ms".into(),
            "40".into(),
        ];
        let (_, _, options) = parse_args(&args).unwrap();
        assert_eq!(options.input_device.as_deref(), Some("Mic"));
        assert_eq!(options.output_device.as_deref(), Some("Cable"));
        assert_eq!(options.chunk_ms, Some(40));
    }
}
