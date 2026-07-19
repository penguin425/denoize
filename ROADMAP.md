# Neural backend roadmap

This document separates deployable implementations from architectural names.
A backend is complete only when denoize can load a documented pretrained model,
run it without Python, preserve the input channel count and duration, and pass
an end-to-end audio fixture test.

## Investigation status

| Model | Upstream artifact | Native integration gap | Status |
|---|---|---|---|
| BSRNN | PyTorch implementations and checkpoints | Complex STFT band splitting, recurrent model port, and a stable redistributable speech-enhancement checkpoint | Researching |
| MP-SENet | [Official MIT repository](https://github.com/yxlu-0102/MP-SENet) with PyTorch checkpoints | Compressed magnitude/phase STFT adapter and compatible exported graph | Researching |
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
