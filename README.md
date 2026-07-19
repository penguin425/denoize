# denoize

**The pursuit of the world's highest-fidelity audio denoising — in pure Rust.**

`denoize` removes background noise from WAV recordings with **maximum transparency**:
preserving timbre, transients, dynamics, stereo imaging, and natural "air".

## Implemented technology stack

### Classical DSP (always available)
- STFT/ISTFT + Perfect Reconstruction OLA + high overlap
- IMCRA/MCRA noise estimation + SPP + spectral-flatness profiling
- Ephraim-Malah Decision-Directed SNR
- **8 gain estimators**: OMLSA, LogMMSE, MMSE-STSA, Wiener, SpecSub, SpecSub-NL, SpecSub-Geo
- Transient protection, cepstral smoothing, pre-emphasis
- **Advanced windows**: Kaiser, Flat-top, DPSS (+ Hann/Hamming/Sine/Blackman)
- **Multiband spectral subtraction** (Bark bands)
- **Perceptual weighting** (Bark-scale gain shaping)
- **Musical-noise post-filter**

### Optional AI backends (feature-gated)
| Backend | Feature | Description |
|---------|---------|-------------|
| `rnnoise` | `--features rnnoise` | RNNoise via nnnoiseless (pure-Rust) |
| `deepfilter` | `--features deepfilter` | DeepFilterNet v3 (tract ONNX, embedded model) |

Build everything: `cargo build --release --features full`

## Supported input formats

| Format | Decoder | Notes |
|--------|---------|-------|
| WAV | `hound` | 8–32 bit int / float |
| MP3 | `nanomp3` (Pure Rust) | ID3 skip, no resampling |
| M4A/AAC | `oxideav-aac` (Pure Rust) | MP4 demux + AAC-LC decode |

### Output formats

| Format | Encoder | Notes |
|--------|---------|-------|
| WAV | `hound` | Lossless; preserves bit depth |
| MP3 | `shine-rs` (Pure Rust) | `--mp3-bitrate` (default 192 kbps) |
| M4A | `oxideav-aac` + MP4 mux | `--m4a-bitrate` (default 192 kbps); Pure-Rust AAC-LC |

```sh
# MP3 / M4A input and output — no manual ffmpeg conversion
denoize noisy.mp3 clean.mp3 -p hifi
denoize noisy.m4a clean.m4a -b deepfilter
denoize noisy.wav clean.wav --mp3-bitrate 320
```

## Quick start

```sh
cargo build --release --features full

# Best classical quality
./target/release/denoize noisy.wav clean.wav -p hifi

# RNNoise AI backend
./target/release/denoize noisy.wav clean.wav -b rnnoise

# DeepFilterNet v3 AI backend
./target/release/denoize noisy.wav clean.wav -b deepfilter

# Advanced DSP options
./target/release/denoize noisy.wav clean.wav \
  --window kaiser --kaiser-beta 10 \
  --multiband --perceptual --postfilter \
  -a specsub-nl -s 0.5
```

## CLI highlights

```
-b, --backend <NAME>     classical|rnnoise|deepfilter
-a, --algorithm <NAME>    omlsa|logmmse|mmse|wiener|specsub|specsub-nl|specsub-geo
--window <NAME>          hann|hamming|sine|blackman|kaiser|flattop|dpss
--kaiser-beta <B>        Kaiser β (default 8.0)
--dpss-nw <NW>           DPSS bandwidth (default 3.0)
--multiband              Multiband spectral subtraction
--perceptual             Bark perceptual gain weighting
--postfilter             Musical-noise suppression post-filter
-p hifi                   Flagship preset (Kaiser + perceptual + postfilter)
--quality ultra           Maximum fidelity settings
```

## Library API

```rust
use denoize::{denoise_file_with_backend, Backend, DenoiserConfig, Preset};

let cfg = Preset::HiFi.config(48000);
denoise_file_with_backend("noisy.wav", "clean.wav", cfg, Backend::Classical)?;

// With AI (requires --features full at build time)
denoise_file_with_backend("noisy.wav", "clean.wav", cfg, Backend::DeepFilter)?;
```

## Roadmap status

| Priority | Technology | Status |
|----------|-----------|--------|
| 1 | DeepFilterNet v3 | ✅ `--features deepfilter` |
| 2 | RNNoise | ✅ `--features rnnoise` |
| 3 | Kaiser/Flat-top/DPSS windows | ✅ |
| 4 | Multiband / nonlinear SpecSub | ✅ |
| 5 | Perceptual weighting + musical-noise PF | ✅ |
| 6–8 | BSRNN / MP-SENet / MossFormer2 / SGMSE+ | 🔲 Future |

## License

MIT OR Apache-2.0.