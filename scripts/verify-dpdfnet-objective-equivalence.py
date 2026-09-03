#!/usr/bin/env python3
"""Bind two DPDFNet objective matrices whose deterministic payloads match."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_JSON_BYTES = 32 * 1024 * 1024
MATRIX_MODELS = (
    "dpdfnet2_48khz_hr",
    "dpdfnet8_48khz_hr",
    "gtcrn",
)
RESERVED_FIELDS = {
    "fixture_manifest",
    "logical_parallelism",
    "path",
    "load_ms",
    "process_ms",
    "rtf",
}
EXCLUDED_FIELDS = [
    "fixture_manifest",
    "environment.logical_parallelism",
    "models.*.path",
    "models.*.load_ms",
    "cases.*.{dpdfnet2_48khz_hr,dpdfnet8_48khz_hr,gtcrn}.process_ms",
    "cases.*.{dpdfnet2_48khz_hr,dpdfnet8_48khz_hr,gtcrn}.rtf",
]
SUMMARY_MODELS = {
    "dpdfnet2-48khz-hr": "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b",
    "dpdfnet8-48khz-hr": "7b3afbb260a08fe9af3d16e3bda992971be1e7e951d1dee7c2d235f5c43f5631",
    "gtcrn-dns3": "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87",
}


class EquivalenceError(RuntimeError):
    pass


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise EquivalenceError(f"{label} is not a regular file: {path}")
    size = path.stat().st_size
    if not 1 <= size <= MAX_JSON_BYTES:
        raise EquivalenceError(f"{label} size must be in 1..={MAX_JSON_BYTES}")
    payload = path.read_bytes()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EquivalenceError(f"invalid {label} JSON: {error}") from error
    if not isinstance(document, dict):
        raise EquivalenceError(f"{label} must be a JSON object")
    return document, payload


def numeric(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EquivalenceError(f"{label} must be numeric")
    if not math.isfinite(float(value)):
        raise EquivalenceError(f"{label} must be finite")


def file_record(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "name": path.name,
        "size_bytes": len(payload),
        "sha256": sha256(payload),
    }


def reserved_paths(value: Any, prefix: tuple[Any, ...] = ()) -> set[tuple[Any, ...]]:
    found: set[tuple[Any, ...]] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = (*prefix, key)
            if key in RESERVED_FIELDS:
                found.add(path)
            found.update(reserved_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(reserved_paths(child, (*prefix, index)))
    return found


def normalize_matrix(matrix: dict[str, Any], label: str) -> dict[str, Any]:
    expected_top = {
        "schema",
        "fixture_manifest",
        "fixture_fingerprint",
        "models",
        "environment",
        "cases",
    }
    if set(matrix) != expected_top or matrix.get("schema") != "denoize-dpdfnet-gtcrn-matrix-v1":
        raise EquivalenceError(f"{label} has an unsupported matrix contract")
    if not isinstance(matrix["fixture_manifest"], str) or not matrix["fixture_manifest"]:
        raise EquivalenceError(f"{label} fixture_manifest must be a non-empty path")
    fingerprint = matrix["fixture_fingerprint"]
    if not isinstance(fingerprint, str) or not SHA256_RE.fullmatch(fingerprint):
        raise EquivalenceError(f"{label} has an invalid fixture_fingerprint")

    environment = matrix["environment"]
    if not isinstance(environment, dict) or set(environment) != {
        "arch",
        "logical_parallelism",
        "os",
        "visqol_enabled",
    }:
        raise EquivalenceError(f"{label} has an unsupported environment contract")
    logical_parallelism = environment["logical_parallelism"]
    if (
        isinstance(logical_parallelism, bool)
        or not isinstance(logical_parallelism, int)
        or logical_parallelism < 1
    ):
        raise EquivalenceError(f"{label} logical_parallelism must be a positive integer")

    models = matrix["models"]
    if not isinstance(models, dict) or set(models) != set(MATRIX_MODELS):
        raise EquivalenceError(f"{label} has an unsupported model set")
    for model_name in MATRIX_MODELS:
        model = models[model_name]
        if not isinstance(model, dict):
            raise EquivalenceError(f"{label} model {model_name} must be an object")
        if not isinstance(model.get("path"), str) or not model["path"]:
            raise EquivalenceError(f"{label} model {model_name} path is invalid")
        numeric(model.get("load_ms"), f"{label} model {model_name} load_ms")

    cases = matrix["cases"]
    if not isinstance(cases, list) or not cases:
        raise EquivalenceError(f"{label} cases must be a non-empty array")
    case_ids: set[str] = set()
    allowed_reserved = {
        ("fixture_manifest",),
        ("environment", "logical_parallelism"),
    }
    for model_name in MATRIX_MODELS:
        allowed_reserved.update(
            {
                ("models", model_name, "path"),
                ("models", model_name, "load_ms"),
            }
        )
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise EquivalenceError(f"{label} case {index} must be an object")
        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in case_ids:
            raise EquivalenceError(f"{label} case {index} has an invalid or duplicate id")
        case_ids.add(identifier)
        for model_name in MATRIX_MODELS:
            model = case.get(model_name)
            if not isinstance(model, dict):
                raise EquivalenceError(
                    f"{label} case {identifier} is missing model {model_name}"
                )
            numeric(
                model.get("process_ms"),
                f"{label} case {identifier} {model_name} process_ms",
            )
            numeric(model.get("rtf"), f"{label} case {identifier} {model_name} rtf")
            allowed_reserved.update(
                {
                    ("cases", index, model_name, "process_ms"),
                    ("cases", index, model_name, "rtf"),
                }
            )

    observed_reserved = reserved_paths(matrix)
    if observed_reserved != allowed_reserved:
        unexpected = sorted(
            (".".join(map(str, path)) for path in observed_reserved - allowed_reserved)
        )
        missing = sorted(
            (".".join(map(str, path)) for path in allowed_reserved - observed_reserved)
        )
        raise EquivalenceError(
            f"{label} nondeterministic-field contract differs; "
            f"unexpected={unexpected}, missing={missing}"
        )

    normalized = copy.deepcopy(matrix)
    del normalized["fixture_manifest"]
    del normalized["environment"]["logical_parallelism"]
    for model_name in MATRIX_MODELS:
        del normalized["models"][model_name]["path"]
        del normalized["models"][model_name]["load_ms"]
    for case in normalized["cases"]:
        for model_name in MATRIX_MODELS:
            del case[model_name]["process_ms"]
            del case[model_name]["rtf"]
    return normalized


def canonical(document: dict[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise EquivalenceError(f"matrix is not canonical JSON data: {error}") from error
    return (encoded + "\n").encode("utf-8")


def validate_summary(
    summary: dict[str, Any],
    summary_payload: bytes,
    matrix: dict[str, Any],
    matrix_payload: bytes,
    label: str,
) -> str:
    if summary.get("schema") != "denoize-dpdfnet-gtcrn-evaluation-summary-v1":
        raise EquivalenceError(f"{label} has an unsupported summary contract")
    source_commit = summary.get("source_commit")
    if not isinstance(source_commit, str) or not COMMIT_RE.fullmatch(source_commit):
        raise EquivalenceError(f"{label} summary has an invalid source_commit")
    if summary.get("matrix_result_sha256") != sha256(matrix_payload):
        raise EquivalenceError(f"{label} summary does not bind its matrix")
    if summary.get("fixture_fingerprint") != matrix.get("fixture_fingerprint"):
        raise EquivalenceError(f"{label} summary and matrix fixture fingerprints differ")
    if summary.get("models") != SUMMARY_MODELS:
        raise EquivalenceError(f"{label} summary has an unsupported model identity set")
    # Ensure the payload is consumed so callers cannot accidentally bind a
    # re-encoded document instead of the exact summary input.
    if not summary_payload:
        raise EquivalenceError(f"{label} summary payload is empty")
    return source_commit


def write_exclusive(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise EquivalenceError(f"refusing to replace objective equivalence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as destination:
            descriptor = -1
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    reference_matrix, reference_matrix_payload = load(
        args.reference_matrix, "reference matrix"
    )
    candidate_matrix, candidate_matrix_payload = load(
        args.candidate_matrix, "candidate matrix"
    )
    reference_summary, reference_summary_payload = load(
        args.reference_summary, "reference summary"
    )
    candidate_summary, candidate_summary_payload = load(
        args.candidate_summary, "candidate summary"
    )
    reference_commit = validate_summary(
        reference_summary,
        reference_summary_payload,
        reference_matrix,
        reference_matrix_payload,
        "reference",
    )
    candidate_commit = validate_summary(
        candidate_summary,
        candidate_summary_payload,
        candidate_matrix,
        candidate_matrix_payload,
        "candidate",
    )
    normalized_reference = canonical(
        normalize_matrix(reference_matrix, "reference matrix")
    )
    normalized_candidate = canonical(
        normalize_matrix(candidate_matrix, "candidate matrix")
    )
    if normalized_reference != normalized_candidate:
        raise EquivalenceError("objective matrices differ outside allowed measurements")
    if sha256(reference_matrix_payload) == sha256(candidate_matrix_payload):
        raise EquivalenceError("objective matrices are already byte-identical")

    document = {
        "schema": "denoize-dpdfnet-objective-equivalence-v1",
        "schema_version": 1,
        "reference": {
            "source_commit": reference_commit,
            "matrix": file_record(args.reference_matrix, reference_matrix_payload),
            "summary": file_record(args.reference_summary, reference_summary_payload),
        },
        "candidate": {
            "source_commit": candidate_commit,
            "matrix": file_record(args.candidate_matrix, candidate_matrix_payload),
            "summary": file_record(args.candidate_summary, candidate_summary_payload),
        },
        "fixture_fingerprint": reference_matrix["fixture_fingerprint"],
        "models": SUMMARY_MODELS,
        "case_count": len(reference_matrix["cases"]),
        "canonicalization": {
            "algorithm": "denoize-dpdfnet-objective-deterministic-v1",
            "excluded_fields": EXCLUDED_FIELDS,
            "sha256": sha256(normalized_reference),
        },
        "equivalent": True,
    }
    write_exclusive(
        args.output,
        (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        ),
    )
    return document


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--reference-matrix", type=Path, required=True)
    result.add_argument("--reference-summary", type=Path, required=True)
    result.add_argument("--candidate-matrix", type=Path, required=True)
    result.add_argument("--candidate-summary", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        document = generate(args)
    except (EquivalenceError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"canonical matrix SHA-256: {document['canonicalization']['sha256']}")
    print(f"objective equivalence: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
