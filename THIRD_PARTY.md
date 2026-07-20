# Third-party notices

## ESPnet BSRNN reference

`scripts/export-bsrnn.py` contains an adapted transcription of the BSRNN
reference implementation from [`espnet/espnet`](https://github.com/espnet/espnet)
revision `5208894ceaa534732164212357b63d83dd137eab`, authored by ESPnet's
contributors and distributed under the Apache License 2.0.

The converter supports the `wyz/vctk_bsrnn_xtiny_causal` model revision
`59e1f2263b7946b1970a222d1beef9adc5a67eaa`, also published under
CC-BY-4.0. Model weights are not included in denoize.

The original implementation and model are used for speech enhancement. The
converter removes the ESPnet framework dependency, fixes the supported
architecture to the xtiny causal configuration, expresses complex arithmetic
as real-valued ONNX operations, and exports a dynamic-frame graph.

Apache-2.0 license text: <https://www.apache.org/licenses/LICENSE-2.0>

CC-BY-4.0 license text: <https://creativecommons.org/licenses/by/4.0/legalcode>

## ClearerVoice MossFormer2

`scripts/export-mossformer2.py` loads the MossFormer2 speech-enhancement
architecture from [`modelscope/ClearerVoice-Studio`](https://github.com/modelscope/ClearerVoice-Studio)
revision `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`. The upstream code and the
`alibabasglab/MossFormer2_SE_48K` model revision
`eff8c97925c8bec812af707814b3e5d777fd4503` are distributed under the Apache
License 2.0. Model weights are not included in denoize.

The converter fixes the deployment graph to the official four-second feature
window and rewrites ONNX operations to numerically equivalent tract-supported
primitives. No upstream source code is copied into the Rust adapter.

## SGMSE+

`scripts/export-sgmse.py` loads the NCSN++ speech-enhancement architecture
from [`sp-uhh/sgmse`](https://github.com/sp-uhh/sgmse) revision
`1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e` and the official VoiceBank model
revision `b6485214b3662a7f90309f397cacf1384046783c`. The upstream code and model
are distributed under the MIT License. Model weights are not included in
denoize.

The converter loads the published EMA parameters and replaces only the
PyTorch complex tensor boundary with explicit real and imaginary ONNX
channels. The Rust adapter independently implements the documented OUVE
predictor/corrector sampler and signal-processing frontend.
# Optional FDK-AAC

The `fdk-aac-encoder` feature uses the third-party `fdk-aac-rust` port of the
Fraunhofer FDK AAC Codec Library for Android. It is not enabled by default, by
`full`, or in official binaries. The upstream Fraunhofer license and patent
notice apply when this feature is built or distributed.
