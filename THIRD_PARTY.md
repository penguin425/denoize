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
