#!/usr/bin/env python3
"""Recover a private DPDFNet listening answer key from named audio hashes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import wave


MODEL_DPDFNET = "dpdfnet2-48khz-hr"
MODEL_GTCRN = "gtcrn-dns3"
CORE_SELECTION = "lexicographically-smallest-trial-id-v1"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_AUDIO_BYTES = 64 * 1024 * 1024
CASE_ID_RE = re.compile(r"^[A-Za-z0-9._+-]{1,128}$")
EXPECTED_AUDIO_NAMES = {
    "clean.wav",
    "noisy.wav",
    "dpdfnet2.wav",
    "dpdfnet8.wav",
    "gtcrn.wav",
}
REQUIRED_AUDIO_NAMES = EXPECTED_AUDIO_NAMES - {"dpdfnet8.wav"}


class RecoveryError(RuntimeError):
    pass


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical(document: object) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path, label: str) -> tuple[dict, bytes]:
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"{label} is not a regular file: {path}")
    size = path.stat().st_size
    if not 1 <= size <= MAX_JSON_BYTES:
        raise RecoveryError(f"{label} size is outside 1..={MAX_JSON_BYTES}: {path}")
    payload = path.read_bytes()
    try:
        document = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryError(f"invalid {label} JSON {path}: {error}") from error
    if not isinstance(document, dict):
        raise RecoveryError(f"{label} must be a JSON object: {path}")
    return document, payload


def load_protocol(path: Path) -> tuple[dict, bytes, dict[str, dict]]:
    protocol, payload = load_json(path, "public protocol")
    scorer_path = Path(__file__).resolve().with_name("score-dpdfnet-blind-listening.py")
    specification = importlib.util.spec_from_file_location(
        "denoize_dpdfnet_blind_scorer", scorer_path
    )
    if specification is None or specification.loader is None:
        raise RecoveryError("load blinded-listening protocol validator")
    scorer = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(scorer)
        trials = scorer.validate_protocol(protocol, payload, path.resolve())
    except Exception as error:
        raise RecoveryError(f"invalid public protocol: {error}") from error
    return protocol, payload, trials


def audio_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"candidate audio is not a regular file: {path}")
    size = path.stat().st_size
    if not 44 <= size <= MAX_AUDIO_BYTES:
        raise RecoveryError(f"candidate audio size is outside 44..={MAX_AUDIO_BYTES}: {path}")
    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            frames = source.getnframes()
            sample_width = source.getsampwidth()
            compression = source.getcomptype()
    except (EOFError, wave.Error) as error:
        raise RecoveryError(f"invalid candidate WAV {path}: {error}") from error
    if channels != 1 or sample_rate != 48_000 or frames <= 0:
        raise RecoveryError(f"candidate WAV must be non-empty mono 48 kHz: {path}")
    if sample_width not in {2, 3, 4} or compression != "NONE":
        raise RecoveryError(f"candidate WAV must be uncompressed PCM: {path}")
    payload = path.read_bytes()
    return {
        "size_bytes": len(payload),
        "sha256": sha256(payload),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "frames": frames,
    }


def classify(case: dict) -> str:
    if case.get("kind") == "clean-preservation":
        return "source-preservation"
    if case.get("kind") != "noise-matrix":
        raise RecoveryError(
            f"unsupported listening case kind for {case.get('id')}"
        )
    noise = case.get("noise")
    if noise == "three-talker-babble":
        return "babble"
    if isinstance(noise, str) and noise.startswith("freesound-"):
        return "recorded-noise"
    return "synthetic-noise"


def matrix_cases(matrix: dict) -> dict[str, dict]:
    cases = matrix.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RecoveryError("source matrix must contain a non-empty cases array")
    by_id: dict[str, dict] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise RecoveryError(f"source matrix case {index} must be an object")
        identifier = case.get("id")
        if not isinstance(identifier, str) or not CASE_ID_RE.fullmatch(identifier):
            raise RecoveryError(f"source matrix case {index} has an invalid id")
        if identifier in by_id:
            raise RecoveryError(f"source matrix contains duplicate case id: {identifier}")
        by_id[identifier] = case
    return by_id


def candidate_records(
    audio_root: Path, cases: dict[str, dict]
) -> tuple[dict[tuple[str, str], tuple[str, dict[str, dict[str, object]]]], str]:
    if audio_root.is_symlink() or not audio_root.is_dir():
        raise RecoveryError(f"candidate audio directory is unavailable: {audio_root}")
    resolved_root = audio_root.resolve()
    by_inputs: dict[
        tuple[str, str], tuple[str, dict[str, dict[str, object]]]
    ] = {}
    manifest: list[dict[str, object]] = []
    for directory in sorted(resolved_root.iterdir(), key=lambda value: value.name):
        if directory.is_symlink() or not directory.is_dir():
            raise RecoveryError(f"candidate audio root contains an unexpected entry: {directory}")
        identifier = directory.name
        if identifier not in cases:
            raise RecoveryError(f"candidate audio case is absent from source matrix: {identifier}")
        names = {entry.name for entry in directory.iterdir()}
        if not REQUIRED_AUDIO_NAMES <= names or not names <= EXPECTED_AUDIO_NAMES:
            raise RecoveryError(
                f"candidate audio case {identifier} must contain the closed named-output set"
            )
        records = {
            name.removesuffix(".wav"): audio_record(directory / name)
            for name in sorted(names)
        }
        geometries = {
            (
                record["sample_rate_hz"],
                record["channels"],
                record["frames"],
            )
            for record in records.values()
        }
        if len(geometries) != 1:
            raise RecoveryError(
                f"candidate audio geometry differs within case: {identifier}"
            )
        input_key = (records["clean"]["sha256"], records["noisy"]["sha256"])
        if input_key in by_inputs:
            raise RecoveryError("candidate cases have an ambiguous clean/noisy fingerprint")
        if records["dpdfnet2"]["sha256"] == records["gtcrn"]["sha256"]:
            raise RecoveryError(f"candidate model outputs are indistinguishable: {identifier}")
        by_inputs[input_key] = (identifier, records)
        manifest.append({"source_case_id": identifier, "audio": records})
    if len(manifest) != 12:
        raise RecoveryError(f"candidate audio must contain exactly 12 cases, observed {len(manifest)}")
    return by_inputs, sha256(canonical(manifest))


def recover(args: argparse.Namespace) -> None:
    protocol, protocol_payload, protocol_trials = load_protocol(args.protocol)
    protocol_path = args.protocol.resolve()
    if len(protocol_trials) != 16:
        raise RecoveryError("public protocol must contain exactly 16 trials")
    matrix, matrix_payload = load_json(args.matrix_result, "source matrix")
    observed_matrix_sha256 = sha256(matrix_payload)
    if observed_matrix_sha256 != protocol["source_matrix_sha256"]:
        raise RecoveryError("source matrix digest differs from the public protocol")
    cases = matrix_cases(matrix)
    candidates, candidate_manifest_sha256 = candidate_records(args.audio_dir, cases)

    recovered: list[dict[str, object]] = []
    by_case: dict[str, list[dict[str, object]]] = {}
    for trial in protocol["trials"]:
        trial_id = trial["trial_id"]
        public_audio = trial["audio"]
        input_key = (
            public_audio["reference"]["sha256"],
            public_audio["input"]["sha256"],
        )
        candidate = candidates.get(input_key)
        if candidate is None:
            raise RecoveryError(f"public trial inputs have no named candidate match: {trial_id}")
        source_case_id, records = candidate
        expected_stratum = classify(cases[source_case_id])
        if trial["stratum"] != expected_stratum:
            raise RecoveryError(
                f"public trial stratum differs from the source matrix: {trial_id}"
            )
        named_outputs = {
            records["dpdfnet2"]["sha256"]: MODEL_DPDFNET,
            records["gtcrn"]["sha256"]: MODEL_GTCRN,
        }
        a_sha256 = public_audio["a"]["sha256"]
        b_sha256 = public_audio["b"]["sha256"]
        if {a_sha256, b_sha256} != set(named_outputs):
            raise RecoveryError(f"public trial sides do not match both named outputs: {trial_id}")
        record: dict[str, object] = {
            "trial_id": trial_id,
            "source_case_id": source_case_id,
            "stratum": expected_stratum,
            "role": None,
            "duplicate_of": None,
            "a_model": named_outputs[a_sha256],
            "b_model": named_outputs[b_sha256],
            "a_sha256": a_sha256,
            "b_sha256": b_sha256,
        }
        recovered.append(record)
        by_case.setdefault(source_case_id, []).append(record)

    if set(by_case) != {identifier for identifier, _ in candidates.values()}:
        raise RecoveryError("public protocol and candidate audio do not cover the same cases")
    single_count = sum(len(records) == 1 for records in by_case.values())
    pair_count = sum(len(records) == 2 for records in by_case.values())
    if single_count != 8 or pair_count != 4 or any(len(records) not in {1, 2} for records in by_case.values()):
        raise RecoveryError("public protocol must contain eight single cases and four duplicate pairs")
    for records in by_case.values():
        records.sort(key=lambda record: str(record["trial_id"]))
        records[0]["role"] = "core"
        if len(records) == 2:
            records[1]["role"] = "repeat"
            records[1]["duplicate_of"] = records[0]["trial_id"]

    answer_key = {
        "schema": "denoize-dpdfnet-blind-answer-key-v2",
        "schema_version": 2,
        "bundle_id": protocol["bundle_id"],
        "protocol_sha256": sha256(protocol_payload),
        "source_matrix_sha256": observed_matrix_sha256,
        "recovery": {
            "method": "named-output-sha256-v1",
            "core_selection": CORE_SELECTION,
            "candidate_audio_manifest_sha256": candidate_manifest_sha256,
            "source_case_count": len(by_case),
        },
        "trials": recovered,
    }
    output = args.output.absolute()
    resolved_output = output.resolve()
    if resolved_output == protocol_path or resolved_output.is_relative_to(
        protocol_path.parent
    ):
        raise RecoveryError("refusing to write a private answer key inside the public bundle")
    write_exclusive(output, canonical(answer_key))
    print(f"private recovered answer key: {output}")
    print(f"protocol SHA-256: {answer_key['protocol_sha256']}")
    print(f"candidate audio manifest SHA-256: {candidate_manifest_sha256}")
    print("recovered trials: 16 (12 core, 4 repeat)")


def write_exclusive(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RecoveryError(f"refusing to replace existing answer key: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as destination:
            descriptor = -1
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--protocol", type=Path, required=True)
    result.add_argument("--matrix-result", type=Path, required=True)
    result.add_argument("--audio-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    try:
        recover(parser().parse_args())
    except (RecoveryError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
