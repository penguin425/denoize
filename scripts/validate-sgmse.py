#!/usr/bin/env python3
"""Run SGMSE+'s pinned end-to-end real-speech quality regression."""

from __future__ import annotations

import argparse
import hashlib
import math
import pathlib
import random
import struct
import subprocess
import tempfile
import urllib.request
import wave


FIXTURE_URL = (
    "https://raw.githubusercontent.com/espnet/espnet/"
    "443028662106472c60fe8bd892cb277e5b488651/test_utils/st_test.wav"
)
FIXTURE_SHA256 = "55441b4929df3806be67cb9dfca28a8554c2f7fc111b742baff3fe90a490ae1c"
NOISE_SEED = 425
NOISE_AMPLITUDE = 0.05
MINIMUM_SI_SNR_IMPROVEMENT_DB = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--denoize", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    return parser.parse_args()


def read_wav(path: pathlib.Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
            raise SystemExit("SGMSE+ fixture must be 16-bit mono PCM")
        sample_rate = reader.getframerate()
        count = reader.getnframes()
        samples = struct.unpack(f"<{count}h", reader.readframes(count))
    return [sample / 32768.0 for sample in samples], sample_rate


def write_wav(path: pathlib.Path, samples: list[float], sample_rate: int) -> None:
    quantized = [
        max(-32768, min(32767, round(sample * 32768.0))) for sample in samples
    ]
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack(f"<{len(quantized)}h", *quantized))


def si_snr(reference: list[float], estimate: list[float]) -> float:
    if len(reference) != len(estimate):
        raise SystemExit("enhanced fixture duration changed")
    ref_mean = sum(reference) / len(reference)
    estimate_mean = sum(estimate) / len(estimate)
    ref = [sample - ref_mean for sample in reference]
    est = [sample - estimate_mean for sample in estimate]
    ref_energy = sum(sample * sample for sample in ref)
    scale = sum(left * right for left, right in zip(est, ref)) / ref_energy
    target = [scale * sample for sample in ref]
    noise = [sample - projected for sample, projected in zip(est, target)]
    return 10.0 * math.log10(
        (sum(sample * sample for sample in target) + 1e-12)
        / (sum(sample * sample for sample in noise) + 1e-12)
    )


def main() -> None:
    args = parse_args()
    denoize = args.denoize.resolve()
    model = args.model.resolve()
    if not denoize.is_file():
        raise SystemExit(f"denoize executable not found: {denoize}")
    if not model.is_file():
        raise SystemExit(f"SGMSE+ model not found: {model}")

    with tempfile.TemporaryDirectory(prefix="denoize-sgmse-") as directory:
        root = pathlib.Path(directory)
        clean_path = root / "clean.wav"
        noisy_path = root / "noisy.wav"
        enhanced_path = root / "enhanced.wav"
        urllib.request.urlretrieve(FIXTURE_URL, clean_path)
        digest = hashlib.sha256(clean_path.read_bytes()).hexdigest()
        if digest != FIXTURE_SHA256:
            raise SystemExit(f"speech fixture sha256 mismatch: {digest}")
        clean, sample_rate = read_wav(clean_path)
        generator = random.Random(NOISE_SEED)
        noisy = [
            max(
                -1.0,
                min(
                    32767.0 / 32768.0,
                    sample + generator.gauss(0.0, NOISE_AMPLITUDE),
                ),
            )
            for sample in clean
        ]
        write_wav(noisy_path, noisy, sample_rate)
        subprocess.run(
            [
                str(denoize),
                str(noisy_path),
                str(enhanced_path),
                "--backend",
                "sgmse",
                "--onnx-model",
                str(model),
                "--onnx-rate",
                "16000",
            ],
            check=True,
        )
        actual_noisy, _ = read_wav(noisy_path)
        enhanced, enhanced_rate = read_wav(enhanced_path)
        if enhanced_rate != sample_rate:
            raise SystemExit("enhanced fixture sample rate changed")
        noisy_score = si_snr(clean, actual_noisy)
        enhanced_score = si_snr(clean, enhanced)
        improvement = enhanced_score - noisy_score
        print(f"noisy SI-SNR: {noisy_score:.3f} dB")
        print(f"enhanced SI-SNR: {enhanced_score:.3f} dB")
        print(f"improvement: {improvement:.3f} dB")
        if improvement < MINIMUM_SI_SNR_IMPROVEMENT_DB:
            raise SystemExit(
                "SGMSE+ speech quality regression: "
                f"expected >= {MINIMUM_SI_SNR_IMPROVEMENT_DB:.1f} dB improvement"
            )


if __name__ == "__main__":
    main()
