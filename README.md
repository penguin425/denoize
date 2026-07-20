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
| `onnx` | `--features onnx` | External waveform-to-waveform ONNX model (tract, Pure Rust) |
| `mpsenet` | `--features mpsenet` | MP-SENet magnitude/phase enhancement adapter (external converted model) |
| `bsrnn` | `--features bsrnn` | ESPnet BSRNN spectral enhancement adapter (external converted model) |
| `mossformer2` | `--features mossformer2` | ClearerVoice MossFormer2 48 kHz mask adapter (external converted model) |
| `sgmse` | `--features sgmse` | SGMSE+ iterative diffusion adapter (external converted model) |
| `gtcrn` | `--features gtcrn` | Official 48K-parameter causal GTCRN; offline and stateful streaming |

Build everything: `cargo build --release --features full`

The generic ONNX backend is the deployment foundation for future neural
models. It intentionally accepts only single-input/single-output waveform
models; spectral models and diffusion samplers require dedicated adapters.

> The prebuilt GitHub binaries include every backend. Because DeepFilterNet
> 0.5.6 is not available from crates.io, the crates.io package's `full` feature
> currently includes RNNoise, generic ONNX, MP-SENet, BSRNN, MossFormer2, and
> SGMSE+, but not DeepFilterNet.

## Supported input formats

| Format | Decoder | Notes |
|--------|---------|-------|
| WAV | `hound` | 8–32 bit int / float |
| MP3 | `nanomp3` (Pure Rust) | ID3 skip, no resampling |
| M4A/AAC | `oxideav-aac` (Pure Rust) | MP4 demux + AAC-LC decode |
| FLAC | `claxon` | Lossless FLAC |
| Ogg Opus | `opus` + `ogg` | Mono/stereo; native 48 kHz decode |

### Output formats

| Format | Encoder | Notes |
|--------|---------|-------|
| WAV | `hound` | Lossless; preserves bit depth |
| MP3 | `shine-rs` (Pure Rust) | `--mp3-bitrate` (default 192 kbps) |
| M4A | `oxideav-aac` + MP4 mux | GitHub/source builds; `--m4a-bitrate` (default 192 kbps) |
| FLAC | `flacenc` | Lossless, pure Rust |
| Ogg Opus | `opus` + `ogg` | 128 kbps, mono/stereo |

```sh
# MP3 / M4A input and output — no manual ffmpeg conversion
denoize noisy.mp3 clean.mp3 -p hifi
denoize noisy.m4a clean.m4a -b deepfilter
denoize noisy.wav clean.wav --mp3-bitrate 320

# User-supplied waveform model: [1, samples] or [1, 1, samples]
denoize noisy.wav clean.wav -b onnx \
  --onnx-model model.onnx --onnx-rate 16000

# Official MP-SENet checkpoint converted with scripts/export-mpsenet.py
denoize noisy.wav clean.wav -b mpsenet \
  --onnx-model mp-senet-vb.onnx --onnx-rate 16000

# ESPnet BSRNN xtiny checkpoint converted with scripts/export-bsrnn.py
denoize noisy.wav clean.wav -b bsrnn \
  --onnx-model bsrnn-xtiny.onnx --onnx-rate 48000

# ClearerVoice MossFormer2 48 kHz model
denoize noisy.wav clean.wav -b mossformer2 \
  --onnx-model mossformer2-se-48k.onnx --onnx-rate 48000

# Official SGMSE+ VoiceBank model (30-step quality sampler)
denoize noisy.wav clean.wav -b sgmse \
  --onnx-model sgmse-vb.onnx --onnx-rate 16000 --sgmse-profile quality

# Verified official GTCRN model (manual model path is unnecessary afterwards)
denoize models install gtcrn
denoize models verify all
denoize models update gtcrn
denoize models remove gtcrn
denoize noisy.wav clean.wav -b gtcrn

# Stereo coupling, pipes, metrics, and directory batches
denoize stereo.wav clean.flac --channels linked
cat noisy.wav | denoize - - > clean.wav
denoize metrics reference.wav clean.wav --json
denoize recordings/ cleaned/ --batch
```

To prepare the pinned official MP-SENet VoiceBank model:

```sh
git clone https://github.com/yxlu-0102/MP-SENet.git
git -C MP-SENet checkout 89932cfe90d1dacb8e170e4a331d762462c21792
python3 -m pip install torch onnx onnxscript pesq joblib matplotlib
python3 scripts/export-mpsenet.py \
  --repo MP-SENet \
  --checkpoint MP-SENet/best_ckpt/g_best_vb \
  --output mp-senet-vb.onnx
```

The VoiceBank graph is about 9 MiB and expects 16 kHz audio. On the reference
x86-64 Linux host, a two-second mono speech fixture took 43.67 seconds after
model loading and the complete process used 410,048 KiB maximum RSS. Run the
pinned real-speech quality gate after conversion:

```sh
python3 scripts/validate-mpsenet.py \
  --denoize target/release/denoize \
  --model mp-senet-vb.onnx
```

To prepare the pinned ESPnet BSRNN xtiny model (CC-BY-4.0):

```sh
curl -L \
  'https://huggingface.co/wyz/vctk_bsrnn_xtiny_causal/resolve/59e1f2263b7946b1970a222d1beef9adc5a67eaa/exp_vctk/enh_train_enh_bsrnn_xtiny_raw/58epoch.pth' \
  -o 58epoch.pth
echo 'e3cb771a452e0503144af74720b476e81b57f518b789b37ba2c253c6cc22d70b  58epoch.pth' \
  | sha256sum -c -
python3 -m pip install torch onnx onnxruntime
python3 scripts/export-bsrnn.py \
  --checkpoint 58epoch.pth \
  --output bsrnn-xtiny.onnx \
  --verify
```

The adapter resamples to 48 kHz and reproduces the published model's
variance normalization, centered 960-point Hann STFT with a 480-sample hop,
whole-utterance recurrent inference, and inverse STFT. The converted model is
about 2.4 MiB. On a release build on the project reference x86-64 Linux host,
the fixed two-second regression fixture took 1.58 seconds (1.3x realtime) and
used 44,628 KiB maximum RSS. Runtime and memory grow with utterance length.

Run the reproducible real-speech quality gate after conversion:

```sh
python3 scripts/validate-bsrnn.py \
  --denoize target/release/denoize \
  --model bsrnn-xtiny.onnx
```

To prepare the pinned Apache-2.0 MossFormer2 SE 48 kHz model:

```sh
git clone https://github.com/modelscope/ClearerVoice-Studio.git
git -C ClearerVoice-Studio checkout 6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61
curl -L \
  'https://huggingface.co/alibabasglab/MossFormer2_SE_48K/resolve/eff8c97925c8bec812af707814b3e5d777fd4503/last_best_checkpoint.pt' \
  -o last_best_checkpoint.pt
echo '03692b9f773bbd6bb43b9c5a41f96b1e28affd66e13796b7bec66ad3d8b227c6  last_best_checkpoint.pt' \
  | sha256sum -c -
python3 -m pip install torch onnx onnxruntime numpy einops rotary-embedding-torch
python3 scripts/export-mossformer2.py \
  --repo ClearerVoice-Studio \
  --checkpoint last_best_checkpoint.pt \
  --output mossformer2-se-48k.onnx \
  --verify
```

The adapter uses 48 kHz audio, 40 ms Kaldi fbank frames at an 8 ms shift,
first- and second-order deltas, a non-centred 1,920-point symmetric-Hamming
STFT, and the official four-second/three-second-stride edge-discard
reconstruction. The converted graph is about 217 MiB. On the reference
x86-64 Linux host, a four-second mono fixture took 7.74 seconds and used
483,400 KiB maximum RSS in a release build. Model weights are not bundled.

Run the pinned real-speech quality gate after conversion:

```sh
python3 scripts/validate-mossformer2.py \
  --denoize target/release/denoize \
  --model mossformer2-se-48k.onnx
```

To prepare the pinned MIT-licensed SGMSE+ VoiceBank+DEMAND model:

```sh
git clone https://github.com/sp-uhh/sgmse.git
git -C sgmse checkout 1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e
curl -L \
  'https://huggingface.co/sp-uhh/speech-enhancement-sgmse/resolve/b6485214b3662a7f90309f397cacf1384046783c/train_vb_29nqe0uh_epoch%3D115.ckpt?download=true' \
  -o 'train_vb_29nqe0uh_epoch=115.ckpt'
echo 'e3875747b5646092d5c556bae68e5af639e2c1f45f009c669f379cd4d415cbd8  train_vb_29nqe0uh_epoch=115.ckpt' \
  | sha256sum -c -
python3 -m pip install torch onnx onnxruntime numpy
python3 scripts/export-sgmse.py \
  --source sgmse \
  --checkpoint 'train_vb_29nqe0uh_epoch=115.ckpt' \
  --output sgmse-vb.onnx \
  --verify
```

The adapter reproduces the official noisy-peak normalization, centered
510-point periodic-Hann STFT with a 128-sample hop, complex square-root
spectrum transform, and OUVE predictor/corrector sampler. The explicit
quality/speed choice is the upstream 30 reverse steps with one ALD corrector
step (`snr=0.5`), or 60 score-network evaluations. The graph is about 252 MiB
and weights are not bundled. On the reference x86-64 Linux host, the pinned
two-second mono fixture took 737.92 seconds and used 1,204,648 KiB maximum RSS
in a release build. This backend prioritizes generative quality rather than
interactive speed.

Run the pinned quality gate after conversion (expect a long CPU run):

```sh
python3 scripts/validate-sgmse.py \
  --denoize target/release/denoize \
  --model sgmse-vb.onnx
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

## Desktop app

The Tauri desktop app exposes single-file denoising, batch conversion, quality
comparison, and model management without sending audio off the computer. Its
default build includes every backend in the repository's `full` feature set;
FDK-AAC remains an explicit opt-in because of its separate licensing terms.
ONNX-based backends expose model-file, model-rate, and SGMSE quality controls
when selected; managed GTCRN weights are resolved automatically after install.
Desktop batches accept files or folders, preserve relative paths, run with a
configurable worker count, continue after individual failures, and can resume
from the `.denoize-gui-state` journal in the output directory.
Single-file processing also provides local waveform previews, RMS-matched
before/after switching, click-to-seek, and configurable section looping.
Desktop settings are restored automatically, can be stored as named presets,
and can be imported or exported as CLI-compatible TOML. Recent input files are
kept locally for quick reuse.
Audio files and folders can be dropped onto the single-file or batch input
zones; output folders have dedicated drop targets. Multiple audio files switch
the app to batch mode automatically.
The realtime page routes a selected capture device through a low-latency
backend to a playback device, with input/output meters, dropped-chunk counters,
and explicit start/stop controls. Headphones help prevent acoustic feedback.

```sh
cd apps/desktop
npm ci
npm run tauri -- dev

# Build a platform-native installer/package
npm run tauri -- build

# Optional FDK-AAC selector
npm run tauri -- build --features fdk-aac-encoder
```

Linux development requires the WebKitGTK 4.1 and GTK 3 development packages.
For Ubuntu 24.04 or later:

```sh
sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev patchelf
```

## Prebuilt binaries

Each [GitHub Release](https://github.com/penguin425/denoize/releases) contains
prebuilt `full`-feature binaries for:

- Linux x86-64
- macOS Intel and Apple Silicon
- Windows x86-64

Every archive has a matching `.sha256` checksum file.

## Install with Cargo

The crates.io package provides the CLI and library with the classical DSP and
optional RNNoise backends:

```sh
cargo install denoize --features full
```

For the embedded DeepFilterNet backend, use a prebuilt GitHub binary or build
this repository with its primary `Cargo.toml`.

### Publishing a release

1. Set the same version in `Cargo.toml` and `Cargo.crates-io.toml`, then update
   `Cargo.lock`.
2. Commit and push the version change.
3. Create and push a matching tag:

```sh
git tag -a v0.1.0 -m "denoize v0.1.0"
git push origin v0.1.0
```

The `GitHub Release` workflow validates that the tag matches `Cargo.toml`, runs
the full test suite, builds all supported platforms, attaches archives and
checksums, signs desktop updater artifacts, and publishes generated release
notes. Installed desktop apps check the signed `latest.json` feed on startup;
updates are only installed after user confirmation. The updater private key is
kept in the `TAURI_SIGNING_PRIVATE_KEY` repository secret. A failed build leaves
the release as a draft so it cannot expose an incomplete asset set.

## CLI highlights

### Realtime audio

Build with the optional system-audio integration, list devices, then route a
microphone through a denoising backend to an output or virtual-audio device:

```sh
cargo build --release --features live,rnnoise
denoize live --list-devices
denoize live --backend rnnoise --input-device "Microphone" --output-device "Virtual Cable"
```

Realtime processing runs outside the device callbacks and uses bounded queues,
so an overloaded backend drops stale capture chunks instead of blocking the
audio thread. `--chunk-ms` controls the latency/throughput trade-off and defaults
to 100 ms. Input and output devices must currently share a default sample rate.

### Batch processing

Process a directory tree concurrently while preserving its relative layout:

```sh
denoize recordings cleaned --batch --recursive --jobs 4 --output-format flac
```

Batch mode continues after per-file failures and reports a final success/failure
summary. Existing outputs remain protected unless `--force` is supplied. Omit
`--output-format` to retain each input file's format.

### Automatic backend selection

Use `--backend auto` when the build contains multiple denoisers. Short and
quality-prioritized files use DeepFilterNet when available; long files use
RNNoise to bound processing cost. Realtime sessions prefer RNNoise. The
classical backend is the dependency-free fallback, and the selected backend is
reported before processing.

### Adaptive noise profiling

`--adaptive-noise` detects spectrally noise-like, low-speech-probability regions
throughout a recording and slowly refreshes the classical estimator's anchored
noise profile. This handles changing fans, air conditioning, and room tone
without assuming that the recording begins with silence. Tonal frames are
rejected to reduce the risk of learning sustained notes as noise.

### Voice activity detection

`--vad` detects speech with 20 ms energy frames, hangover, context padding, and
region merging. Long silent spans bypass expensive backend inference and are
strongly attenuated; enhanced speech retains a small dry-signal blend to protect
consonants and attacks. Output channel count and duration remain unchanged.

### Loudness delivery

Normalize denoised output to an EBU R128 integrated-loudness target while
respecting an oversampled true-peak ceiling:

```sh
denoize input.wav output.flac --loudness -16 --true-peak -1
```

The applied gain is reduced when necessary to satisfy the peak ceiling, so
peak safety takes precedence over reaching the requested LUFS exactly.

### Content modes

`--mode speech`, `--mode music`, and `--mode ambient` coordinate related DSP
controls instead of changing only one strength value. Speech mode enables VAD
and adaptive profiling; music mode prioritizes transients, stereo content, and
low suppression; ambient mode preserves environmental texture while tracking
slowly changing noise. Explicit options such as `--strength` still override the
mode defaults.

### Optional FDK-AAC encoder

Pure-Rust `oxideav-aac` remains the default. Source builders can opt into the
Fraunhofer encoder and select it per invocation:

```sh
cargo build --release --features fdk-aac-encoder
denoize input.wav output.m4a --aac-encoder fdk --m4a-bitrate 192
```

The FDK feature uses the third-party Rust port and is intentionally excluded
from `full` and official release binaries. Fraunhofer's codec source has its own
license and MPEG-AAC patent language; downstream distributors are responsible
for reviewing both. Enabling it raises the minimum Rust version to 1.87.

### Raw ADTS AAC

`.aac` files are decoded and encoded directly as ADTS streams without an MP4
container or an ffmpeg conversion step. M4A and raw AAC share
`--m4a-bitrate`; raw ADTS output currently uses the default oxideav encoder.

### Metadata preservation

File processing preserves the input's primary metadata tag after encoding.
Native tags are retained for same-format output; conversions remap common
fields such as title, artist, album, date, ReplayGain, comments, and artwork to
the destination container's tag type. Use `--no-metadata` for a clean output.

### Quality comparison

```sh
denoize compare clean.wav noisy.wav enhanced.wav
denoize compare clean.wav noisy.wav enhanced.wav --json
denoize compare clean.wav noisy.wav enhanced.wav --html > report.html
```

The report shows noisy and enhanced SI-SDR, SI-SNR, SNR, segmental SNR, and
improvement deltas. Metrics requiring external models or licensed reference
implementations are explicitly marked as unmeasured.

### Configuration file

Reusable defaults can be stored in TOML and loaded with `--config`. Explicit
command-line options override the file.

```toml
backend = "auto"
preset = "hifi"
mode = "speech"
strength = 0.45
adaptive_noise = true
vad = true
loudness_lufs = -16.0
true_peak_dbtp = -1.0
```

```sh
denoize input.wav output.flac --config denoize.toml --strength 0.55
```

### Batch progress and recovery

Batch runs show completed files, elapsed time, and ETA. `--resume` records
successful outputs in `.denoize-state` under the output directory and skips
them on the next run. Ctrl+C stops scheduling new files; each output is first
written to a temporary file so an interrupted encode cannot replace a valid
destination. Use `--no-progress` for quiet operation or `--json` for NDJSON
progress and summary records.

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
--onnx-model <PATH>       Waveform ONNX model used by the onnx backend
--onnx-rate <HZ>          Model sample rate (default: 16000)
```

## Library API

```rust
use denoize::{denoise_file_with_backend, Backend, DenoiserConfig, Preset};

let cfg = Preset::HiFi.config(48000);
denoise_file_with_backend("noisy.wav", "clean.wav", cfg, Backend::Classical)?;

// With DeepFilterNet (GitHub/source build with --features full)
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
| 6 | Pure-Rust external ONNX inference foundation | 🟨 waveform contract implemented |
| 7 | BSRNN / MP-SENet / MossFormer2 adapters | ✅ implemented and quality-gated |
| 8 | SGMSE+ | ✅ 30-step PC sampler + score-model adapter |

See [ROADMAP.md](ROADMAP.md) for the implementation audit and the acceptance
criteria and numerical evidence for each named model.

## License

The Rust project is MIT licensed. See [THIRD_PARTY.md](THIRD_PARTY.md) for the
Apache-2.0 BSRNN conversion code and CC-BY-4.0 model attribution.
