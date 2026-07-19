#!/usr/bin/env python3
"""Export the pinned ESPnet BSRNN xtiny checkpoint to denoize's ONNX contract.

The model implementation below is a dependency-free transcription of the
Apache-2.0 ESPnet reference implementation. Keeping the small separator graph
here avoids requiring a complete ESPnet installation merely to convert a
checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib

import torch
import torch.nn as nn


MODEL_REVISION = "59e1f2263b7946b1970a222d1beef9adc5a67eaa"
ARCHITECTURE_REVISION = "5208894ceaa534732164212357b63d83dd137eab"
CHECKPOINT_NAME = "58epoch.pth"
CHECKPOINT_SHA256 = "e3cb771a452e0503144af74720b476e81b57f518b789b37ba2c253c6cc22d70b"
SUBBANDS = (5,) + (4,) * 19 + (10,) * 6 + (40,) * 7 + (60,)
FREQUENCY_BINS = 481


class BandSplit(nn.Module):
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.norm = nn.ModuleList(
            nn.GroupNorm(1, subband * 2) for subband in SUBBANDS
        )
        self.fc = nn.ModuleList(
            nn.Conv1d(subband * 2, channels, 1) for subband in SUBBANDS
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        bands = []
        offset = 0
        for subband, norm, projection in zip(SUBBANDS, self.norm, self.fc):
            band = spectrum[:, :, offset : offset + subband, :]
            band = band.reshape(band.shape[0], band.shape[1], subband * 2)
            bands.append(projection(norm(band.transpose(1, 2))).unsqueeze(-1))
            offset += subband
        return torch.cat(bands, dim=-1)


class MaskDecoder(nn.Module):
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.mlp_mask = nn.ModuleList(self._head(channels, band) for band in SUBBANDS)
        self.mlp_residual = nn.ModuleList(
            self._head(channels, band) for band in SUBBANDS
        )

    @staticmethod
    def _head(channels: int, subband: int) -> nn.Sequential:
        return nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.Conv1d(channels, 4 * channels, 1),
            nn.Tanh(),
            nn.Conv1d(4 * channels, subband * 4, 1),
            nn.GLU(dim=1),
        )

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = []
        residuals = []
        for index, (mask_head, residual_head) in enumerate(
            zip(self.mlp_mask, self.mlp_residual)
        ):
            band = embedding[:, :, :, index]
            mask = mask_head(band).transpose(1, 2).contiguous()
            residual = residual_head(band).transpose(1, 2).contiguous()
            masks.append(mask.reshape(mask.shape[0], mask.shape[1], -1, 2))
            residuals.append(
                residual.reshape(residual.shape[0], residual.shape[1], -1, 2)
            )
        return torch.cat(masks, dim=2), torch.cat(residuals, dim=2)


class BSRNN(nn.Module):
    def __init__(self, channels: int = 16, layers: int = 6) -> None:
        super().__init__()
        self.layers = layers
        self.band_split = BandSplit(channels)
        self.norm_time = nn.ModuleList()
        self.rnn_time = nn.ModuleList()
        self.fc_time = nn.ModuleList()
        self.norm_freq = nn.ModuleList()
        self.rnn_freq = nn.ModuleList()
        self.fc_freq = nn.ModuleList()
        hidden = 2 * channels
        for _ in range(layers):
            self.norm_time.append(nn.GroupNorm(1, channels))
            self.rnn_time.append(
                nn.LSTM(channels, hidden, batch_first=True, bidirectional=False)
            )
            self.fc_time.append(nn.Linear(hidden, channels))
            self.norm_freq.append(nn.GroupNorm(1, channels))
            self.rnn_freq.append(
                nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
            )
            self.fc_freq.append(nn.Linear(4 * channels, channels))
        self.mask_decoder = MaskDecoder(channels)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        embedding = self.band_split(spectrum)
        batch, channels, frames, bands = embedding.shape
        skip = embedding
        for index in range(self.layers):
            temporal = self.norm_time[index](skip)
            temporal = temporal.transpose(1, 3).reshape(batch * bands, frames, channels)
            temporal, _ = self.rnn_time[index](temporal)
            temporal = self.fc_time[index](temporal)
            temporal = temporal.reshape(batch, bands, frames, channels).transpose(1, 3)
            skip = skip + temporal

            frequency = self.norm_freq[index](skip)
            frequency = frequency.permute(0, 2, 3, 1).contiguous()
            frequency = frequency.reshape(batch * frames, bands, channels)
            frequency, _ = self.rnn_freq[index](frequency)
            frequency = self.fc_freq[index](frequency)
            frequency = frequency.reshape(batch, frames, bands, channels)
            frequency = frequency.permute(0, 3, 1, 2).contiguous()
            skip = skip + frequency

        mask, residual = self.mask_decoder(skip)
        real = mask[..., 0] * spectrum[..., 0] - mask[..., 1] * spectrum[..., 1]
        imag = mask[..., 0] * spectrum[..., 1] + mask[..., 1] * spectrum[..., 0]
        return torch.stack(
            (real + residual[..., 0], imag + residual[..., 1]), dim=-1
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="compare the exported graph with PyTorch using ONNX Runtime",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint.name != CHECKPOINT_NAME or digest != CHECKPOINT_SHA256:
        raise SystemExit(
            f"unrecognized checkpoint {checkpoint.name} with sha256 {digest}"
        )

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefix = "separator.bsrnn."
    separator_state = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    model = BSRNN().cpu().eval()
    model.load_state_dict(separator_state, strict=True)

    # 32,000 samples is the official training chunk and produces 67 centered
    # STFT frames. The recurrent separator itself accepts arbitrary frame
    # counts, so retain that axis as dynamic for whole-utterance inference.
    spectrum = torch.zeros((1, 67, FREQUENCY_BINS, 2), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        spectrum,
        args.output,
        input_names=["spectrum"],
        output_names=["enhanced_spectrum"],
        dynamic_axes={
            "spectrum": {1: "frames"},
            "enhanced_spectrum": {1: "frames"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    output_digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output}")
    print(f"sha256 {output_digest}")
    print(f"model revision {MODEL_REVISION}")
    print(f"architecture revision {ARCHITECTURE_REVISION}")
    if args.verify:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as error:
            raise SystemExit("--verify requires numpy and onnxruntime") from error
        generator = torch.Generator().manual_seed(425)
        fixture = torch.randn(
            (1, 67, FREQUENCY_BINS, 2), generator=generator, dtype=torch.float32
        )
        with torch.no_grad():
            reference = model(fixture).numpy()
        actual = ort.InferenceSession(
            str(args.output), providers=["CPUExecutionProvider"]
        ).run(None, {"spectrum": fixture.numpy()})[0]
        difference = reference - actual
        correlation = float(np.corrcoef(reference.ravel(), actual.ravel())[0, 1])
        mse = float(np.mean(difference * difference))
        maximum = float(np.max(np.abs(difference)))
        print(f"PyTorch/ONNX correlation {correlation:.12f}")
        print(f"PyTorch/ONNX mse {mse:.12g}")
        print(f"PyTorch/ONNX max_abs {maximum:.12g}")
        if correlation < 0.999999:
            raise SystemExit("exported BSRNN graph failed numerical parity")


if __name__ == "__main__":
    main()
