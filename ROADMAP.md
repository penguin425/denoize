# Neural backend roadmap

This document separates deployable implementations from architectural names.
A backend is complete only when denoize can load a documented pretrained model,
run it without Python, preserve the input channel count and duration, and pass
an end-to-end audio fixture test.

## Investigation status

| Model | Upstream artifact | Native integration gap | Status |
|---|---|---|---|
| BSRNN | [ESPnet VCTK+DEMAND xtiny checkpoint](https://huggingface.co/wyz/vctk_bsrnn_xtiny_causal) (CC-BY-4.0) | External conversion is required because upstream publishes PyTorch only | Implemented |
| MP-SENet | [Official MIT repository](https://github.com/yxlu-0102/MP-SENet) with PyTorch checkpoints | Numerical parity and quality fixture for the converted graph | Adapter implemented |
| MossFormer2 | [Official MIT repository](https://github.com/alibabasglab/MossFormer2), now directing users to ClearerVoice-Studio | Speech-enhancement checkpoint export, segmentation, and overlap reconstruction | Researching |
| SGMSE+ | [Official MIT repository](https://github.com/sp-uhh/sgmse) with PyTorch Lightning checkpoints | Complex STFT transforms plus an iterative predictor/corrector or ODE sampler; it is not a one-pass waveform graph | Researching |

None of these upstream projects currently publishes a model artifact with a
documented ONNX contract that can be embedded directly in this Rust CLI. Their
PyTorch checkpoints are not treated as implemented support.

## Implemented foundation

The `onnx` feature provides a Pure-Rust tract backend for one-input,
one-output `float32` waveform models:

- input layout `[batch, samples]` or `[batch, channels, samples]`;
- batch and model channel dimension are fixed to one;
- file channels are processed independently;
- audio is resampled to and from the configured model rate;
- output duration and original channel count are preserved;
- missing files, unsupported ranks, short outputs, and non-finite samples are
  rejected with explicit errors.

This contract can host exported waveform models, but it does not make any of
the named roadmap models complete by itself.

## MP-SENet adapter

The `mpsenet` feature implements the official 16 kHz frontend in Rust: RMS
normalization, centered 400-point periodic-Hann STFT with 100-sample hop,
0.3-power magnitude compression, parallel magnitude/phase inference, inverse
STFT, 50%-overlapped reconstruction of the official 32,000-sample training
segments, and exact input-duration restoration. `scripts/export-mpsenet.py`
converts an official `g_best_vb` or `g_best_dns` checkpoint into the adapter's
two-input/two-output ONNX contract. The adapter remains partial until a pinned
converted model is covered by an automated denoising-quality fixture.

The converter pins upstream revision
`89932cfe90d1dacb8e170e4a331d762462c21792` and verifies the official checkpoint
SHA-256 before export. On a fixed two-second 16 kHz fixture, the converted graph
matched upstream PyTorch through ONNX Runtime with magnitude correlation above
`0.9999999999` and phase correlation above `0.9999999999`; tract matched ONNX
Runtime at the same correlation threshold. End-to-end Rust/PyTorch waveform
correlation was `0.9900` (MSE `8.56e-6`), with the remaining difference dominated
by phase wrapping in low-energy FFT bins across the two FFT implementations.
On the repository's synthetic tone-plus-noise fixture, the converted VoiceBank
model improved global SNR from `-0.01 dB` to `0.24 dB`; this manual result is not
yet the automated speech-quality gate required for completion.

## BSRNN adapter

The `bsrnn` feature implements the causal ESPnet BSRNN frontend and inference
contract at 48 kHz: per-channel sample-standard-deviation normalization,
centered 960-point periodic-Hann STFT with a 480-sample hop, whole-utterance
recurrent inference, inverse STFT, de-normalization, and exact input channel
count/rate/duration restoration. `scripts/export-bsrnn.py` converts the pinned
`wyz/vctk_bsrnn_xtiny_causal` checkpoint into a dynamic-frame
`[1, frames, 481, 2]` ONNX graph and can verify it against PyTorch using ONNX
Runtime.

The model revision is `59e1f2263b7946b1970a222d1beef9adc5a67eaa`, the
checkpoint SHA-256 is
`e3cb771a452e0503144af74720b476e81b57f518b789b37ba2c253c6cc22d70b`,
and the reference architecture is pinned to Apache-2.0 ESPnet revision
`5208894ceaa534732164212357b63d83dd137eab`. The model is CC-BY-4.0 and the
adapted reference implementation is Apache-2.0; denoize does not bundle its
weights.

On the fixed 67-frame numerical fixture, PyTorch and ONNX Runtime correlation
was `0.999999999998` (MSE `1.88e-11`, maximum absolute error `2.34e-4`). On the
same fixture's PyTorch and Rust waveforms, after the CLI's documented PCM
clipping and quantization, correlated at `0.99999999958` (MSE `2.18e-10`,
maximum absolute error `1.85e-4`). On the pinned two-second Apache-2.0 ESPnet
speech fixture, the Rust end-to-end quality gate improved SI-SNR from
`2.719 dB` to `9.612 dB` (`+6.892 dB`). A release build on the reference x86-64
Linux host processed it in 1.58 seconds (1.3x realtime) with 44,628 KiB maximum
RSS. The model is about 2.4 MiB; memory and latency grow with utterance length
because upstream inference is recurrent and whole-utterance.

## Completion gates

For each named backend:

1. Pin the upstream architecture and checkpoint revision and record its license.
2. Supply a reproducible conversion or a native safe-tensors loader.
3. Implement the exact normalization, STFT, chunking, and reconstruction used
   by upstream inference.
4. Verify numerical parity against upstream inference on a fixed fixture.
5. Add a denoising quality regression fixture, not only shape tests.
6. Document model download, checksum, sample rate, latency, and memory use.
7. Include the backend in `full` only when release binaries can actually run it.

SGMSE+ additionally requires deterministic sampler tests and an explicit
quality/speed choice because its iterative inference cost differs substantially
from one-pass enhancement networks.
