#!/usr/bin/env python3
"""Export the official MP-SENet generator checkpoint to denoize's ONNX contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


PINNED_REVISION = "89932cfe90d1dacb8e170e4a331d762462c21792"
PINNED_CHECKPOINTS = {
    "g_best_vb": "aedfb1aa549159f71b39613d94db831dfa983b3a68b0deea02e84e0ad563f4f9",
    "g_best_dns": "97a77ba67c5c484c65363bb703ea85962f773ca0819e22ce81b4ec33db5e7206",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=pathlib.Path,
                        help="checkout of https://github.com/yxlu-0102/MP-SENet")
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path,
                        help="official g_best_vb or g_best_dns checkpoint")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    # tract 0.19 does not implement the opset-17 LayerNormalization operator.
    # Opset 13 makes PyTorch decompose it into supported primitive operations.
    parser.add_argument("--opset", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    if not (repo / "models" / "model.py").is_file():
        raise SystemExit(f"not an MP-SENet checkout: {repo}")
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != PINNED_REVISION:
        raise SystemExit(
            f"MP-SENet checkout is {revision}; expected pinned revision {PINNED_REVISION}"
        )
    expected_checkpoint = PINNED_CHECKPOINTS.get(checkpoint.name)
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if expected_checkpoint is None or checkpoint_digest != expected_checkpoint:
        raise SystemExit(
            f"unrecognized checkpoint {checkpoint.name} with sha256 {checkpoint_digest}"
        )

    sys.path.insert(0, str(repo))
    import torch  # pylint: disable=import-error,import-outside-toplevel
    from env import AttrDict  # pylint: disable=import-error,import-outside-toplevel
    from models.model import MPNet  # pylint: disable=import-error,import-outside-toplevel

    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file():
        config_path = repo / "config.json"
    config = AttrDict(json.loads(config_path.read_text(encoding="utf-8")))
    if (config.sampling_rate, config.n_fft, config.hop_size, config.win_size,
            config.compress_factor) != (16000, 400, 100, 400, 0.3):
        raise SystemExit("checkpoint frontend does not match denoize's MP-SENet adapter")

    model = MPNet(config).cpu().eval()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["generator"])

    class ExportWrapper(torch.nn.Module):
        def __init__(self, generator: torch.nn.Module) -> None:
            super().__init__()
            self.generator = generator

        def forward(self, magnitude, phase):
            enhanced_magnitude, enhanced_phase, _ = self.generator(magnitude, phase)
            return enhanced_magnitude, enhanced_phase

    wrapper = ExportWrapper(model).eval()
    # The official training segment is 32,000 samples. With centered
    # n_fft=400/hop=100 STFT this is 321 frames. denoize performs overlapped
    # chunk reconstruction around this fixed graph shape.
    frames = 321
    magnitude = torch.ones((1, 201, frames), dtype=torch.float32)
    phase = torch.zeros((1, 201, frames), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (magnitude, phase),
        args.output,
        input_names=["magnitude", "phase"],
        output_names=["enhanced_magnitude", "enhanced_phase"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
