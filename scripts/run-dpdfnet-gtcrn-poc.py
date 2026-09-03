#!/usr/bin/env python3
"""Download pinned PoC inputs and run the DPDFNet-vs-GTCRN comparison."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import random
import struct
import subprocess
import urllib.request
import wave


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
NARROWBAND_NOISY_SHA256 = "d6d98ad426a120c6af2131a22ba8a19f36cb29df04bb757272c6348faca534ea"
ASSETS = {
    "dpdfnet2_48khz_hr.onnx": (
        "https://huggingface.co/Ceva-IP/DPDFNet/resolve/"
        "dd6818d00f50c836fed43a6243ebe49116de5964/onnx/"
        "dpdfnet2_48khz_hr.onnx",
        "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b",
        10_493_337,
    ),
    "gtcrn_simple.onnx": (
        "https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/"
        "3862c44808dca492ea5a8a145d2dc2a1028d08c8/stream/onnx_models/"
        "gtcrn_simple.onnx",
        "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87",
        535_190,
    ),
    "clean_freesound_33711.wav": (
        "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/"
        "d375b2d8309e0935d165700c91da9de862a99c31/assets/"
        "clean_freesound_33711.wav",
        "2d885e9f45c0f9381aee09a397ecd160aff1875ffb75ebe2217d21c3511b0d5d",
        1_017_226,
    ),
    "noisy_snr0.wav": (
        "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/"
        "d375b2d8309e0935d165700c91da9de862a99c31/assets/noisy_snr0.wav",
        "e1e08601f3b7ceb2f36d45c86343e38cd8927a73d7ad5526d6c4687c33aa7186",
        1_017_226,
    ),
    "espnet_st_test.wav": (
        "https://raw.githubusercontent.com/espnet/espnet/"
        "443028662106472c60fe8bd892cb277e5b488651/test_utils/st_test.wav",
        "55441b4929df3806be67cb9dfca28a8554c2f7fc111b742baff3fe90a490ae1c",
        64_078,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/denoize-dpdfnet-gtcrn-poc"),
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--validate-onnxruntime",
        action="store_true",
        help="also run scripts/validate-dpdfnet-poc.py (requires numpy and onnxruntime)",
    )
    return parser.parse_args()


def digest(path: pathlib.Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def acquire(root: pathlib.Path, name: str) -> pathlib.Path:
    url, expected_hash, expected_size = ASSETS[name]
    destination = root / name
    if destination.is_file():
        if destination.stat().st_size == expected_size and digest(destination) == expected_hash:
            return destination
        raise SystemExit(
            f"existing asset does not match the pinned {name}: {destination}"
        )
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, partial)
        actual_size = partial.stat().st_size
        actual_hash = digest(partial)
        if actual_size != expected_size or actual_hash != expected_hash:
            raise SystemExit(
                f"asset verification failed for {name}: "
                f"size={actual_size}, sha256={actual_hash}"
            )
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def make_white_noise_fixture(clean: pathlib.Path, noisy: pathlib.Path) -> None:
    with wave.open(str(clean), "rb") as reader:
        parameters = reader.getparams()
        if parameters.nchannels != 1 or parameters.sampwidth != 2:
            raise SystemExit("ESPnet PoC fixture must be mono PCM16")
        raw = reader.readframes(parameters.nframes)
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    generator = random.Random(425)
    mixed = [
        max(
            -32768,
            min(
                32767,
                round((sample / 32768.0 + generator.gauss(0.0, 0.05)) * 32768.0),
            ),
        )
        for sample in samples
    ]
    with wave.open(str(noisy), "wb") as writer:
        writer.setparams(parameters)
        writer.writeframes(struct.pack(f"<{len(mixed)}h", *mixed))
    actual_hash = digest(noisy)
    if actual_hash != NARROWBAND_NOISY_SHA256:
        raise SystemExit(
            "generated white-noise fixture is not reproducible: "
            f"sha256={actual_hash}"
        )


def run_case(
    *,
    clean: pathlib.Path,
    noisy: pathlib.Path,
    dpdfnet: pathlib.Path,
    gtcrn: pathlib.Path,
    output: pathlib.Path,
    result: pathlib.Path,
    runs: int,
) -> None:
    output.mkdir(exist_ok=True)
    subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--features",
            "dpdfnet,gtcrn",
            "--example",
            "dpdfnet_gtcrn_poc",
            "--",
            "--clean",
            str(clean),
            "--noisy",
            str(noisy),
            "--dpdfnet-model",
            str(dpdfnet),
            "--gtcrn-model",
            str(gtcrn),
            "--runs",
            str(runs),
            "--output-dir",
            str(output),
            "--json",
            str(result),
        ],
        check=True,
        cwd=REPOSITORY_ROOT,
    )


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {name: acquire(root, name) for name in ASSETS}
    fullband_output = root / "fullband-output"
    fullband_result = root / "fullband-result.json"
    run_case(
        clean=paths["clean_freesound_33711.wav"],
        noisy=paths["noisy_snr0.wav"],
        dpdfnet=paths["dpdfnet2_48khz_hr.onnx"],
        gtcrn=paths["gtcrn_simple.onnx"],
        output=fullband_output,
        result=fullband_result,
        runs=args.runs,
    )
    clean_output = root / "clean-preservation-output"
    clean_result = root / "clean-preservation-result.json"
    run_case(
        clean=paths["clean_freesound_33711.wav"],
        noisy=paths["clean_freesound_33711.wav"],
        dpdfnet=paths["dpdfnet2_48khz_hr.onnx"],
        gtcrn=paths["gtcrn_simple.onnx"],
        output=clean_output,
        result=clean_result,
        runs=args.runs,
    )
    narrowband_noisy = root / "espnet_st_test_white_noise.wav"
    make_white_noise_fixture(paths["espnet_st_test.wav"], narrowband_noisy)
    narrowband_output = root / "narrowband-output"
    narrowband_result = root / "narrowband-result.json"
    run_case(
        clean=paths["espnet_st_test.wav"],
        noisy=narrowband_noisy,
        dpdfnet=paths["dpdfnet2_48khz_hr.onnx"],
        gtcrn=paths["gtcrn_simple.onnx"],
        output=narrowband_output,
        result=narrowband_result,
        runs=args.runs,
    )
    if args.validate_onnxruntime:
        validator = str(REPOSITORY_ROOT / "scripts/validate-dpdfnet-poc.py")
        common = [
            "python3",
            validator,
            "--model",
            str(paths["dpdfnet2_48khz_hr.onnx"]),
            "--input",
            str(paths["noisy_snr0.wav"]),
        ]
        subprocess.run(
            common
            + ["--actual", str(fullband_output / "dpdfnet-causal.wav")],
            check=True,
            cwd=REPOSITORY_ROOT,
        )
        subprocess.run(
            common
            + [
                "--actual",
                str(fullband_output / "dpdfnet-aligned.wav"),
                "--alignment-samples",
                "1920",
            ],
            check=True,
            cwd=REPOSITORY_ROOT,
        )
    print(f"full-band JSON result: {fullband_result}")
    print(f"clean-preservation JSON result: {clean_result}")
    print(f"narrow-band JSON result: {narrowband_result}")
    print(f"enhanced WAVs: {fullband_output}, {clean_output}, {narrowband_output}")


if __name__ == "__main__":
    main()
