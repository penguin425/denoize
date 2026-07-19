#!/usr/bin/env python3
"""Export the pinned ClearerVoice MossFormer2 48 kHz SE checkpoint.

The official graph contains a few legal ONNX patterns that tract cannot
currently lower.  The compatibility pass below preserves their numerical
meaning while rewriting them to primitive operations supported by tract.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import torch


ARCHITECTURE_REVISION = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"
MODEL_REVISION = "eff8c97925c8bec812af707814b3e5d777fd4503"
CHECKPOINT_NAME = "last_best_checkpoint.pt"
CHECKPOINT_SHA256 = "03692b9f773bbd6bb43b9c5a41f96b1e28affd66e13796b7bec66ad3d8b227c6"
FRAMES = 496
FEATURES = 180
FREQUENCY_BINS = 961


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", required=True, type=pathlib.Path,
        help="checkout of https://github.com/modelscope/ClearerVoice-Studio",
    )
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def constant_values(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    values = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            value = next((a.t for a in node.attribute if a.name == "value"), None)
            if value is not None:
                values[node.output[0]] = numpy_helper.to_array(value)
    return values


def make_pad_tract_compatible(model: onnx.ModelProto) -> None:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in list(inferred.graph.input) + list(inferred.graph.value_info)
        if value.type.HasField("tensor_type")
    }
    rewritten = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Pad":
            rewritten.append(node)
            continue
        while len(node.input) < 3:
            node.input.append("")
        if not node.input[2]:
            zero = f"__denoize_pad_zero_{index}"
            model.graph.initializer.append(
                numpy_helper.from_array(np.array(0, dtype=np.float32), zero)
            )
            node.input[2] = zero
        if types.get(node.input[0]) != TensorProto.BOOL:
            rewritten.append(node)
            continue
        float_input = f"{node.input[0]}__float"
        float_output = f"{node.output[0]}__float"
        zero = f"__denoize_bool_pad_zero_{index}"
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(0, dtype=np.float32), zero)
        )
        rewritten.append(
            helper.make_node(
                "Cast", [node.input[0]], [float_input],
                name=f"{node.name}/CastIn", to=TensorProto.FLOAT,
            )
        )
        node.input[0] = float_input
        node.input[2] = zero
        original_output = node.output[0]
        node.output[0] = float_output
        rewritten.append(node)
        rewritten.append(
            helper.make_node(
                "Cast", [float_output], [original_output],
                name=f"{node.name}/CastOut", to=TensorProto.BOOL,
            )
        )
    del model.graph.node[:]
    model.graph.node.extend(rewritten)


def add_legacy_squeeze_axes(model: onnx.ModelProto) -> None:
    for node in model.graph.node:
        if node.op_type == "Squeeze" and not any(
            attribute.name == "axes" for attribute in node.attribute
        ):
            # These are the 24 FSMN tensors shaped [1, 1, frames, 256].
            node.attribute.append(helper.make_attribute("axes", [0, 1]))


def optimize_with_ort(source: pathlib.Path, output: pathlib.Path) -> None:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.optimized_model_filepath = str(output)
    ort.InferenceSession(
        str(source), options, providers=["CPUExecutionProvider"],
        disabled_optimizers=[
            "QuickGeluFusion", "SkipLayerNormFusion", "LayerNormFusion",
            "LayerNormFusionL1", "LayerNormFusionL2",
            "SimplifiedLayerNormFusion",
            "ConvActivationFusion",
        ],
    )


def rewrite_einsum(model: onnx.ModelProto) -> None:
    rewritten = []
    for node in model.graph.node:
        equation = next(
            (a.s.decode("utf-8") for a in node.attribute if a.name == "equation"),
            None,
        )
        if node.op_type == "Einsum" and equation == "... d, h d -> ... h d":
            expanded = f"{node.output[0]}__expanded"
            rewritten.append(
                helper.make_node(
                    "Unsqueeze", [node.input[0]], [expanded],
                    name=f"{node.name}/Unsqueeze", axes=[-2],
                )
            )
            rewritten.append(
                helper.make_node(
                    "Mul", [expanded, node.input[1]], list(node.output),
                    name=f"{node.name}/Mul",
                )
            )
        else:
            rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)


def remove_empty_concat_inputs(model: onnx.ModelProto) -> None:
    values = constant_values(model)
    empty_outputs = set()
    for node in model.graph.node:
        if node.op_type != "Slice" or any(name not in values for name in node.input[1:]):
            continue
        starts = np.asarray(values[node.input[1]]).ravel()
        ends = np.asarray(values[node.input[2]]).ravel()
        steps = (
            np.asarray(values[node.input[4]]).ravel()
            if len(node.input) > 4 and node.input[4]
            else np.ones_like(starts)
        )
        if any(start == end and step > 0 for start, end, step in zip(starts, ends, steps)):
            empty_outputs.add(node.output[0])
    for node in model.graph.node:
        if node.op_type == "Concat":
            retained = [name for name in node.input if name not in empty_outputs]
            if retained and len(retained) != len(node.input):
                del node.input[:]
                node.input.extend(retained)


def split_negative_padding(model: onnx.ModelProto) -> None:
    values = constant_values(model)
    rewritten = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Pad" or node.input[1] not in values:
            rewritten.append(node)
            continue
        pads = np.asarray(values[node.input[1]], dtype=np.int64).copy()
        if not np.any(pads < 0):
            rewritten.append(node)
            continue
        rank = len(pads) // 2
        axes, starts, ends = [], [], []
        for axis in range(rank):
            if pads[axis] < 0 or pads[rank + axis] < 0:
                axes.append(axis)
                starts.append(max(0, -int(pads[axis])))
                ends.append(
                    int(pads[rank + axis])
                    if pads[rank + axis] < 0 else np.iinfo(np.int64).max
                )
        parameters = []
        for label, array in (
            ("starts", starts), ("ends", ends), ("axes", axes),
            ("steps", [1] * len(axes)),
        ):
            name = f"__denoize_pad_crop_{index}_{label}"
            model.graph.initializer.append(
                numpy_helper.from_array(np.asarray(array, dtype=np.int64), name)
            )
            parameters.append(name)
        cropped = f"{node.input[0]}__cropped"
        rewritten.append(
            helper.make_node(
                "Slice", [node.input[0], *parameters], [cropped],
                name=f"{node.name}/Crop",
            )
        )
        node.input[0] = cropped
        pads = np.maximum(pads, 0)
        pad_name = f"__denoize_positive_pad_{index}"
        model.graph.initializer.append(numpy_helper.from_array(pads, pad_name))
        node.input[1] = pad_name
        rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)


def prune(model: onnx.ModelProto) -> None:
    needed = {value.name for value in model.graph.output}
    retained = []
    for node in reversed(model.graph.node):
        if any(output in needed for output in node.output):
            retained.append(node)
            needed.update(name for name in node.input if name)
    retained.reverse()
    del model.graph.node[:]
    model.graph.node.extend(retained)
    used = {name for node in retained for name in node.input}
    initializers = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)


def run_ort(path: pathlib.Path, features: np.ndarray) -> np.ndarray:
    return ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    ).run(None, {"features": features})[0]


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if revision != ARCHITECTURE_REVISION:
        raise SystemExit(f"ClearerVoice checkout is {revision}; expected {ARCHITECTURE_REVISION}")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint.name != CHECKPOINT_NAME or digest != CHECKPOINT_SHA256:
        raise SystemExit(f"unrecognized checkpoint {checkpoint.name} with sha256 {digest}")

    sys.path.insert(0, str(repo / "clearvoice"))
    from clearvoice.models.mossformer2_se.mossformer2_se_wrapper import TestNet

    model = TestNet().cpu().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    fixture = torch.zeros((1, FRAMES, FEATURES), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="denoize-mossformer2-") as directory:
        temporary = pathlib.Path(directory)
        raw = temporary / "raw.onnx"
        compatible = temporary / "compatible.onnx"
        optimized = temporary / "optimized.onnx"
        torch.onnx.export(
            model, fixture, raw, input_names=["features"], output_names=["mask"],
            opset_version=12, do_constant_folding=True, dynamo=False,
        )
        graph = onnx.load(raw)
        make_pad_tract_compatible(graph)
        add_legacy_squeeze_axes(graph)
        onnx.save(graph, compatible)
        optimize_with_ort(compatible, optimized)
        graph = onnx.load(optimized)
        rewrite_einsum(graph)
        remove_empty_concat_inputs(graph)
        split_negative_padding(graph)
        prune(graph)
        onnx.checker.check_model(graph)
        onnx.save(graph, args.output)

        if args.verify:
            generator = np.random.default_rng(425)
            features = generator.standard_normal(
                (1, FRAMES, FEATURES), dtype=np.float32
            )
            reference = run_ort(compatible, features)
            actual = run_ort(args.output, features)
            difference = reference - actual
            correlation = float(np.corrcoef(reference.ravel(), actual.ravel())[0, 1])
            mse = float(np.mean(difference * difference))
            maximum = float(np.max(np.abs(difference)))
            print(f"compatibility/tract graph correlation {correlation:.12f}")
            print(f"compatibility/tract graph mse {mse:.12g}")
            print(f"compatibility/tract graph max_abs {maximum:.12g}")
            if correlation < 0.999999 or maximum > 1e-4:
                raise SystemExit("tract compatibility rewrite failed numerical parity")

    print(f"wrote {args.output}")
    print(f"sha256 {hashlib.sha256(args.output.read_bytes()).hexdigest()}")
    print(f"model revision {MODEL_REVISION}")
    print(f"architecture revision {ARCHITECTURE_REVISION}")


if __name__ == "__main__":
    main()
