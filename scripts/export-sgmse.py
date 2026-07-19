#!/usr/bin/env python3
"""Export the pinned official SGMSE+ VoiceBank score model to ONNX.

The source checkout is required because the 65.6M-parameter NCSN++
architecture is maintained upstream. The converter verifies both source and
checkpoint revisions, loads the checkpoint's EMA weights, and replaces only
the complex tensor boundary with explicit real/imaginary channels.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import pathlib
import subprocess
import sys
import textwrap
import types

import torch


MODEL_REVISION = "b6485214b3662a7f90309f397cacf1384046783c"
ARCHITECTURE_REVISION = "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e"
CHECKPOINT_NAME = "train_vb_29nqe0uh_epoch=115.ckpt"
CHECKPOINT_SHA256 = "e3875747b5646092d5c556bae68e5af639e2c1f45f009c669f379cd4d415cbd8"
FREQUENCY_BINS = 256
EXPORT_FRAMES = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def sha256(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def patch_real_boundary(model: torch.nn.Module, module: types.ModuleType) -> None:
    source = textwrap.dedent(inspect.getsource(type(model).forward))
    complex_input = """x = torch.cat((x[:,[0],:,:].real, x[:,[0],:,:].imag,
            x[:,[1],:,:].real, x[:,[1],:,:].imag), dim=1)"""
    complex_output = """# Convert back to complex number
    h = self.output_layer(h)
    h = torch.permute(h, (0, 2, 3, 1)).contiguous()
    h = torch.view_as_complex(h)[:,None, :, :]
    return h"""
    if source.count(complex_input) != 1 or source.count(complex_output) != 1:
        raise SystemExit("pinned NCSN++ complex boundary no longer matches converter")
    source = source.replace(complex_input, "x = x")
    source = source.replace(
        complex_output,
        "# ScoreModel.forward negates the legacy NCSN++ output.\n"
        "    return -self.output_layer(h)",
    )
    namespace: dict[str, object] = {}
    exec(source, module.__dict__, namespace)
    model.forward = types.MethodType(namespace["forward"], model)


def make_pad_tract_compatible(path: pathlib.Path) -> None:
    import numpy as np
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(path)
    for index, node in enumerate(graph.graph.node):
        if node.op_type != "Pad":
            continue
        while len(node.input) < 3:
            node.input.append("")
        if not node.input[2]:
            zero = f"__denoize_pad_zero_{index}"
            graph.graph.initializer.append(
                numpy_helper.from_array(np.array(0, dtype=np.float32), zero)
            )
            node.input[2] = zero
    onnx.checker.check_model(graph)
    onnx.save(graph, path)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    checkpoint = args.checkpoint.resolve()
    if not (source / "sgmse" / "backbones" / "ncsnpp.py").is_file():
        raise SystemExit(f"SGMSE source checkout not found: {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != ARCHITECTURE_REVISION:
        raise SystemExit(f"expected architecture revision {ARCHITECTURE_REVISION}, got {revision}")
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    digest = sha256(checkpoint)
    if checkpoint.name != CHECKPOINT_NAME or digest != CHECKPOINT_SHA256:
        raise SystemExit(f"unrecognized checkpoint {checkpoint.name} with sha256 {digest}")

    sys.path.insert(0, str(source))
    import sgmse.backbones.ncsnpp as ncsnpp_module

    # Lightning stored the data-module class in hyperparameters. A minimal
    # class with the same qualified name permits trusted official checkpoint
    # loading without installing the training-only Lightning dependency.
    data_module = types.ModuleType("sgmse.data_module")
    specs_data_module = type("SpecsDataModule", (), {})
    specs_data_module.__module__ = "sgmse.data_module"
    data_module.SpecsDataModule = specs_data_module
    sys.modules["sgmse.data_module"] = data_module
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hyperparameters = saved["hyper_parameters"]
    required = {
        "backbone": "ncsnpp",
        "sde": "ouve",
        "n_fft": 510,
        "hop_length": 128,
        "spec_factor": 0.15,
        "spec_abs_exponent": 0.5,
        "normalize": "noisy",
    }
    for key, expected in required.items():
        if hyperparameters.get(key) != expected:
            raise SystemExit(
                f"checkpoint {key}={hyperparameters.get(key)!r}; expected {expected!r}"
            )

    model = ncsnpp_module.NCSNpp(**hyperparameters).cpu()
    state = {
        key.removeprefix("dnn."): value
        for key, value in saved["state_dict"].items()
        if key.startswith("dnn.")
    }
    model.load_state_dict(state, strict=True)
    parameters = list(model.parameters())
    ema = saved["ema"]["shadow_params"]
    if len(parameters) != len(ema):
        raise SystemExit("checkpoint EMA parameter count does not match NCSN++")
    for parameter, average in zip(parameters, ema):
        parameter.data.copy_(average)
    patch_real_boundary(model, ncsnpp_module)
    model.eval()

    features = torch.zeros((1, 4, FREQUENCY_BINS, EXPORT_FRAMES), dtype=torch.float32)
    time = torch.ones((1,), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (features, time),
        args.output,
        input_names=["features", "time"],
        output_names=["score"],
        dynamic_axes={"features": {3: "frames"}, "score": {3: "frames"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    make_pad_tract_compatible(args.output)
    print(f"wrote {args.output}")
    print(f"sha256 {sha256(args.output)}")
    print(f"model revision {MODEL_REVISION}")
    print(f"architecture revision {ARCHITECTURE_REVISION}")
    if args.verify:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as error:
            raise SystemExit("--verify requires numpy and onnxruntime") from error
        generator = torch.Generator().manual_seed(425)
        fixture = torch.randn(features.shape, generator=generator)
        fixture_time = torch.tensor([0.7], dtype=torch.float32)
        with torch.no_grad():
            reference = model(fixture, fixture_time).numpy()
        actual = ort.InferenceSession(
            str(args.output), providers=["CPUExecutionProvider"]
        ).run(None, {"features": fixture.numpy(), "time": fixture_time.numpy()})[0]
        difference = reference - actual
        correlation = float(np.corrcoef(reference.ravel(), actual.ravel())[0, 1])
        mse = float(np.mean(difference * difference))
        maximum = float(np.max(np.abs(difference)))
        print(f"PyTorch/ONNX correlation {correlation:.12f}")
        print(f"PyTorch/ONNX mse {mse:.12g}")
        print(f"PyTorch/ONNX max_abs {maximum:.12g}")
        if correlation < 0.999999:
            raise SystemExit("exported SGMSE+ graph failed numerical parity")


if __name__ == "__main__":
    main()
