# denoize CLI reference

```text
denoize 0.4.0 — pure-Rust audio denoiser engineered for the world's highest sound quality

Classical DSP + optional AI backends (RNNoise, DeepFilterNet v3, MP-SENet, BSRNN).
Input/output: WAV, FLAC, Ogg Opus, MP3, M4A (built in; no ffmpeg).

USAGE:
    denoize <INPUT> <OUTPUT.wav|flac|opus|ogg|mp3|m4a> [OPTIONS]
    denoize models <list|install|verify|path> [MODEL]
    denoize metrics <REFERENCE> <TEST> [--json|--markdown]

OPTIONS:
    -b, --backend <NAME>     classical|rnnoise|deepfilter|onnx|mpsenet|bsrnn|mossformer2|sgmse|gtcrn  (default: classical)
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
        --channels <MODE>     independent|linked|mid-side (default: independent)
        --sgmse-profile <P>   fast|balanced|quality (default: balanced)
        --batch               process files in INPUT directory into OUTPUT directory
        --force               allow replacing existing output files
        --json                emit a machine-readable result
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
```
