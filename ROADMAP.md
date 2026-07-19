# Neural backend roadmap

This document separates deployable implementations from architectural names.
A backend is complete only when denoize can load a documented pretrained model,
run it without Python, preserve the input channel count and duration, and pass
an end-to-end audio fixture test.

## Investigation status

| Model | Upstream artifact | Native integration gap | Status |
|---|---|---|---|
| BSRNN | [ESPnet VCTK+DEMAND xtiny checkpoint](https://huggingface.co/wyz/vctk_bsrnn_xtiny_causal) (CC-BY-4.0) | External conversion is required because upstream publishes PyTorch only | Implemented |
| MP-SENet | [Official MIT repository](https://github.com/yxlu-0102/MP-SENet) with PyTorch checkpoints | External conversion is required because upstream publishes PyTorch only | Implemented |
| MossFormer2 | [Apache-2.0 ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) and the official 48 kHz checkpoint | External conversion is required because upstream publishes PyTorch only | Implemented |
| SGMSE+ | [Official MIT repository](https://github.com/sp-uhh/sgmse) with PyTorch Lightning checkpoints | External conversion plus a native iterative predictor/corrector sampler | Implemented |

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
two-input/two-output ONNX contract. The converted model is covered by a pinned
automated real-speech quality fixture.

The converter pins upstream revision
`89932cfe90d1dacb8e170e4a331d762462c21792` and verifies the official checkpoint
SHA-256 before export. On a fixed two-second 16 kHz fixture, the converted graph
matched upstream PyTorch through ONNX Runtime with magnitude correlation above
`0.9999999999` and phase correlation above `0.9999999999`; tract matched ONNX
Runtime at the same correlation threshold. End-to-end Rust/PyTorch waveform
correlation was `0.9900` (MSE `8.56e-6`), with the remaining difference dominated
by phase wrapping in low-energy FFT bins across the two FFT implementations.
On the pinned two-second Apache-2.0 ESPnet speech fixture, the Rust end-to-end
quality gate improved SI-SNR from `2.719 dB` to `10.282 dB` (`+7.563 dB`). The
converted graph is about 9 MiB. On the reference x86-64 Linux host, inference
for the fixture took 43.67 seconds and the complete process used 410,048 KiB
maximum RSS.

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

## MossFormer2 adapter

The `mossformer2` feature implements the ClearerVoice 48 kHz frontend and its
four-second deployment contract: 60-bin Kaldi fbank features with first- and
second-order deltas, a non-centred 1,920-point symmetric-Hamming STFT with a
384-sample hop, real spectral-mask application, three-second-stride segmented
inference, 0.5-second edge discard, resampling, and exact input-duration and
channel restoration. `scripts/export-mossformer2.py` pins and verifies the
official checkpoint and rewrites the fixed 496-frame graph to tract-supported
primitive ONNX operations.

The architecture revision is `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`,
the model revision is `eff8c97925c8bec812af707814b3e5d777fd4503`, and the
checkpoint SHA-256 is
`03692b9f773bbd6bb43b9c5a41f96b1e28affd66e13796b7bec66ad3d8b227c6`.
Both architecture and model are Apache-2.0; weights are external. On a fixed
496-frame numerical fixture, the compatibility rewrite matched its source
graph exactly, while tract and ONNX Runtime correlated at
`0.999999999997` (MSE `4.93e-12`, maximum absolute error `4.49e-5`). The graph
is about 217 MiB. A four-second release-build CLI run on the reference x86-64
Linux host took 7.74 seconds and used 483,400 KiB maximum RSS. On the pinned
four-second Apache-2.0 ESPnet speech fixture, the Rust end-to-end quality gate
improved SI-SNR from `2.683 dB` to `13.928 dB` (`+11.246 dB`).

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

## SGMSE+ adapter

The `sgmse` feature implements the official 16 kHz VoiceBank+DEMAND inference
path: noisy-peak normalization, centered 510-point periodic-Hann STFT with a
128-sample hop, magnitude-square-root complex transform scaled by 0.15,
multiple-of-64 spectral padding, inverse transform, and exact duration/channel
restoration. `scripts/export-sgmse.py` loads the official EMA parameters and
exports the dynamic-frame NCSN++ score network with explicit real/imaginary
channels for tract.

The architecture revision is `1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e`,
the model revision is `b6485214b3662a7f90309f397cacf1384046783c`, and the
checkpoint SHA-256 is
`e3875747b5646092d5c556bae68e5af639e2c1f45f009c669f379cd4d415cbd8`.
Both code and model are MIT licensed; weights are external. The explicit
quality/speed choice is the upstream quality configuration: 30 OUVE reverse
steps, one ALD corrector step per reverse step, `snr=0.5`, and therefore 60
score-network evaluations. Sampling uses a documented fixed SplitMix64 and
Box-Muller normal stream so repeated runs are deterministic.

On a fixed 64-frame score fixture, PyTorch and ONNX Runtime correlated above
`0.999999999999` (MSE `4.66e-12`, maximum absolute error `1.53e-5`). On the
pinned two-second Apache-2.0 ESPnet speech fixture, the Rust end-to-end output
correlated with the same deterministic Python/ONNX sampler at
`0.9999999972` (MSE `2.35e-11`, maximum PCM difference `3.05e-5`). The quality
gate improved SI-SNR from `2.719 dB` to `11.471 dB` (`+8.752 dB`). The graph is
about 252 MiB. A release build on the reference x86-64 Linux host took 737.92
seconds for the two-second fixture and used 1,204,648 KiB maximum RSS.
