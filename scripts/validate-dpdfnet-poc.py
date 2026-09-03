#!/usr/bin/env python3
"""Cross-check the Rust DPDFNet PoC stream against official ONNX Runtime."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort


SAMPLE_RATE = 48_000
FFT_SIZE = 960
HOP_SIZE = 480
MAX_ABS_ERROR = 2.5e-4
MAX_RMS_ERROR = 3.0e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument(
        "--alignment-samples",
        type=int,
        default=0,
        help="advance the causal reference by this many samples before comparison",
    )
    return parser.parse_args()


def read_pcm16_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getframerate() != SAMPLE_RATE
            or reader.getnchannels() != 1
            or reader.getsampwidth() != 2
        ):
            raise SystemExit(f"expected 48 kHz mono PCM16 WAV: {path}")
        data = reader.readframes(reader.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def initial_state(session: ort.InferenceSession) -> np.ndarray:
    metadata = session.get_modelmeta().custom_metadata_map
    required = (
        "state_size",
        "erb_norm_state_size",
        "spec_norm_state_size",
        "erb_norm_init",
        "spec_norm_init",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise SystemExit(f"model metadata is missing: {', '.join(missing)}")
    state_size = int(metadata["state_size"])
    erb_size = int(metadata["erb_norm_state_size"])
    spec_size = int(metadata["spec_norm_state_size"])
    erb = np.fromstring(metadata["erb_norm_init"], sep=",", dtype=np.float32)
    spec = np.fromstring(metadata["spec_norm_init"], sep=",", dtype=np.float32)
    if erb.size != erb_size or spec.size != spec_size:
        raise SystemExit("model normalization-state metadata has inconsistent lengths")
    state = np.zeros(state_size, dtype=np.float32)
    state[:erb_size] = erb
    state[erb_size : erb_size + spec_size] = spec
    return state


def enhance(session: ort.InferenceSession, audio: np.ndarray) -> np.ndarray:
    indices = np.arange(FFT_SIZE, dtype=np.float32)
    phase = np.pi * (indices + 0.5) / FFT_SIZE
    window = np.sin(0.5 * np.pi * np.sin(phase) ** 2).astype(np.float32)
    state = initial_state(session)
    input_name, state_name = (value.name for value in session.get_inputs())
    output_name, state_output_name = (value.name for value in session.get_outputs())
    pending = np.asarray(audio, dtype=np.float32).copy()
    overlap = np.zeros(FFT_SIZE, dtype=np.float32)
    output: list[np.ndarray] = []

    def process_frame(frame: np.ndarray) -> None:
        nonlocal state, overlap
        spectrum = np.fft.rfft(frame * window)
        model_input = np.stack(
            [spectrum.real.astype(np.float32), spectrum.imag.astype(np.float32)],
            axis=-1,
        )[None, None, :, :]
        enhanced, state = session.run(
            [output_name, state_output_name],
            {input_name: model_input, state_name: state},
        )
        complex_frame = enhanced[0, 0, :, 0] + 1j * enhanced[0, 0, :, 1]
        overlap += (np.fft.irfft(complex_frame, n=FFT_SIZE) * window).astype(
            np.float32
        )
        output.append(overlap[:HOP_SIZE].copy())
        overlap[:HOP_SIZE] = overlap[HOP_SIZE:]
        overlap[HOP_SIZE:] = 0.0

    while pending.size >= FFT_SIZE:
        process_frame(pending[:FFT_SIZE])
        pending = pending[HOP_SIZE:]
    if pending.size:
        process_frame(np.pad(pending, (0, FFT_SIZE - pending.size)))
    if not output:
        return np.zeros_like(audio)
    return np.concatenate(output)[: audio.size]


def main() -> None:
    args = parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(args.model), options, providers=["CPUExecutionProvider"]
    )
    source = read_pcm16_mono(args.input)
    actual = read_pcm16_mono(args.actual)
    if args.alignment_samples < 0:
        raise SystemExit("--alignment-samples must be non-negative")
    if actual.size != source.size:
        raise SystemExit(
            "Rust output length differs from the production exact-length contract: "
            f"actual={actual.size}, expected={source.size}"
        )
    # The Rust finite wrapper presents a zero-padded final 10 ms host hop and
    # trims it afterward. Mirror that call contract around the official causal
    # frame loop so non-hop-aligned WAV lengths remain exactly preserved.
    padded_input_size = ((source.size + HOP_SIZE - 1) // HOP_SIZE) * HOP_SIZE
    reference_size = padded_input_size + args.alignment_samples
    causal_reference = enhance(
        session,
        np.pad(source, (0, reference_size - source.size)),
    )
    expected = causal_reference[
        args.alignment_samples : args.alignment_samples + source.size
    ]
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(difference * difference)))
    correlation = float(np.corrcoef(actual, expected)[0, 1])
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"alignment samples: {args.alignment_samples}")
    print(f"compared samples: {actual.size}")
    print(f"maximum absolute error: {maximum:.9g}")
    print(f"RMS error: {rms:.9g}")
    print(f"correlation: {correlation:.12f}")
    if maximum > MAX_ABS_ERROR or rms > MAX_RMS_ERROR:
        raise SystemExit(
            "Rust DPDFNet stream differs materially from the ONNX Runtime reference"
        )


if __name__ == "__main__":
    main()
