#!/usr/bin/env python3
"""Assemble the closed, attestable DPDFNet promotion decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import zipfile
from typing import Any


MODEL_SHA256 = "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b"
DPDFNET8_SHA256 = "7b3afbb260a08fe9af3d16e3bda992971be1e7e951d1dee7c2d235f5c43f5631"
GTCRN_SHA256 = "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
SIGNER_WORKFLOW = "penguin425/denoize/.github/workflows/dpdfnet-promotion.yml"
REPORTER_V2_CHECKS = {
    "reaper-version",
    "run-duration-rate",
    "requested-buffer-coverage",
    "effective-buffer-observed",
    "realtime-worker",
    "audible-continuity",
    "nvda-osara",
    "no-host-plugin-crashes",
}
OBJECTIVE_EQUIVALENCE_FIELDS = [
    "fixture_manifest",
    "environment.logical_parallelism",
    "models.*.path",
    "models.*.load_ms",
    "cases.*.{dpdfnet2_48khz_hr,dpdfnet8_48khz_hr,gtcrn}.process_ms",
    "cases.*.{dpdfnet2_48khz_hr,dpdfnet8_48khz_hr,gtcrn}.rtf",
]
OBJECTIVE_MODELS = {
    "dpdfnet2-48khz-hr": MODEL_SHA256,
    "dpdfnet8-48khz-hr": DPDFNET8_SHA256,
    "gtcrn-dns3": GTCRN_SHA256,
}


class PromotionError(RuntimeError):
    pass


def load(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise PromotionError(f"{label} is not a regular file: {path}")
    size = path.stat().st_size
    if not 1 <= size <= MAX_JSON_BYTES:
        raise PromotionError(f"{label} size must be in 1..={MAX_JSON_BYTES}")
    payload = path.read_bytes()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"invalid {label} JSON: {error}") from error
    if not isinstance(document, dict):
        raise PromotionError(f"{label} must be a JSON object")
    return document, payload


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path, payload: bytes) -> dict[str, Any]:
    return {"name": path.name, "size_bytes": len(payload), "sha256": sha256(payload)}


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromotionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PromotionError(f"{label} must be finite")
    return result


def nested(document: dict[str, Any], path: str) -> Any:
    value: Any = document
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise PromotionError(f"evidence is missing {path}")
        value = value[component]
    return value


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise PromotionError(f"{label} has an unsupported field set")
    return value


def validate_json_file_record(value: Any, label: str) -> dict[str, Any]:
    record = exact_keys(value, {"name", "size_bytes", "sha256"}, label)
    name = record["name"]
    size = record["size_bytes"]
    digest = record["sha256"]
    if (
        not isinstance(name, str)
        or not name
        or "/" in name
        or "\\" in name
        or not name.endswith(".json")
    ):
        raise PromotionError(f"{label} has an invalid file name")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or not 1 <= size <= MAX_JSON_BYTES
    ):
        raise PromotionError(f"{label} has an invalid file size")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise PromotionError(f"{label} has an invalid SHA-256")
    return record


def objective_equivalence_record(
    summary: dict[str, Any],
    summary_path: Path,
    summary_payload: bytes,
    listening: dict[str, Any],
    equivalence_path: Path | None,
    source_commit: str,
) -> dict[str, Any] | None:
    candidate_matrix_sha256 = summary.get("matrix_result_sha256")
    reference_matrix_sha256 = listening.get("source_matrix_sha256")
    if (
        not isinstance(candidate_matrix_sha256, str)
        or not SHA256_RE.fullmatch(candidate_matrix_sha256)
        or not isinstance(reference_matrix_sha256, str)
        or not SHA256_RE.fullmatch(reference_matrix_sha256)
    ):
        raise PromotionError("objective and listening matrix digests must be SHA-256")
    if candidate_matrix_sha256 == reference_matrix_sha256:
        if equivalence_path is not None:
            raise PromotionError(
                "objective equivalence must be omitted for byte-identical matrices"
            )
        return None
    if equivalence_path is None:
        raise PromotionError(
            "objective evaluation and listening protocol bind different matrices without equivalence evidence"
        )

    document, payload = load(equivalence_path, "objective equivalence")
    expected_top = {
        "schema",
        "schema_version",
        "reference",
        "candidate",
        "fixture_fingerprint",
        "models",
        "case_count",
        "canonicalization",
        "equivalent",
    }
    exact_keys(document, expected_top, "objective equivalence")
    if (
        document.get("schema") != "denoize-dpdfnet-objective-equivalence-v1"
        or document.get("schema_version") != 1
        or document.get("equivalent") is not True
    ):
        raise PromotionError("unsupported objective equivalence contract")
    if document.get("models") != OBJECTIVE_MODELS:
        raise PromotionError("objective equivalence binds the wrong model identities")
    if summary.get("models") != OBJECTIVE_MODELS:
        raise PromotionError("objective evaluation has the wrong equivalence model set")
    fixture_fingerprint = summary.get("fixture_fingerprint")
    if (
        not isinstance(fixture_fingerprint, str)
        or not SHA256_RE.fullmatch(fixture_fingerprint)
        or document.get("fixture_fingerprint") != fixture_fingerprint
    ):
        raise PromotionError("objective equivalence binds the wrong fixture fingerprint")
    case_count = nested(summary, "quality.case_counts.total")
    if (
        isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or document.get("case_count") != case_count
    ):
        raise PromotionError("objective equivalence binds the wrong case count")

    canonicalization = exact_keys(
        document.get("canonicalization"),
        {"algorithm", "excluded_fields", "sha256"},
        "objective equivalence canonicalization",
    )
    if (
        canonicalization.get("algorithm")
        != "denoize-dpdfnet-objective-deterministic-v1"
        or canonicalization.get("excluded_fields") != OBJECTIVE_EQUIVALENCE_FIELDS
        or not isinstance(canonicalization.get("sha256"), str)
        or not SHA256_RE.fullmatch(canonicalization["sha256"])
    ):
        raise PromotionError("objective equivalence canonicalization differs")

    reference = exact_keys(
        document.get("reference"),
        {"source_commit", "matrix", "summary"},
        "objective equivalence reference",
    )
    candidate = exact_keys(
        document.get("candidate"),
        {"source_commit", "matrix", "summary"},
        "objective equivalence candidate",
    )
    reference_commit = reference.get("source_commit")
    if not isinstance(reference_commit, str) or not COMMIT_RE.fullmatch(reference_commit):
        raise PromotionError("objective equivalence has an invalid reference commit")
    if candidate.get("source_commit") != source_commit:
        raise PromotionError("objective equivalence binds a different candidate commit")
    reference_matrix = validate_json_file_record(
        reference.get("matrix"), "objective equivalence reference matrix"
    )
    validate_json_file_record(
        reference.get("summary"), "objective equivalence reference summary"
    )
    candidate_matrix = validate_json_file_record(
        candidate.get("matrix"), "objective equivalence candidate matrix"
    )
    candidate_summary = validate_json_file_record(
        candidate.get("summary"), "objective equivalence candidate summary"
    )
    if reference_matrix["sha256"] != reference_matrix_sha256:
        raise PromotionError("objective equivalence does not bind the listening matrix")
    if candidate_matrix["sha256"] != candidate_matrix_sha256:
        raise PromotionError("objective equivalence does not bind the candidate matrix")
    if candidate_summary != file_record(summary_path, summary_payload):
        raise PromotionError("objective equivalence does not bind the candidate summary")
    return file_record(equivalence_path, payload)


def verify_attestation(subject: Path, bundle: Path, source_commit: str) -> dict[str, Any]:
    if subject.is_symlink() or not subject.is_file():
        raise PromotionError(f"attestation subject is not a regular file: {subject}")
    if bundle.is_symlink() or not bundle.is_file():
        raise PromotionError(f"Sigstore bundle is not a regular file: {bundle}")
    command = [
        "gh", "attestation", "verify", str(subject),
        "--repo", "penguin425/denoize",
        "--bundle", str(bundle),
        "--signer-workflow", SIGNER_WORKFLOW,
        "--source-digest", source_commit,
        "--format", "json",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise PromotionError(f"attestation verification failed for {subject}: {result.stderr.strip()}")
    try:
        verification = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise PromotionError(f"gh returned invalid attestation JSON: {error}") from error
    if not isinstance(verification, list) or not verification:
        raise PromotionError(f"gh returned no verified attestations for {subject}")
    return {
        "subject_sha256": sha256(subject.read_bytes()),
        "bundle_sha256": sha256(bundle.read_bytes()),
        "verified_attestations": len(verification),
        "repository": "penguin425/denoize",
        "signer_workflow": SIGNER_WORKFLOW,
        "source_commit": source_commit,
    }


def inspect_windows_archive(path: Path, source_commit: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PromotionError(f"experimental archive is not a regular file: {path}")
    size = path.stat().st_size
    if not 1 <= size <= MAX_ARCHIVE_BYTES or path.suffix.lower() != ".zip":
        raise PromotionError("reporter artifact must be a bounded Windows ZIP archive")
    with zipfile.ZipFile(path) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        for item in files:
            name = PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts:
                raise PromotionError(f"unsafe archive member: {item.filename}")
            mode = item.external_attr >> 16
            if mode and stat.S_ISLNK(mode):
                raise PromotionError(f"archive contains a symbolic link: {item.filename}")
        manifests = [item for item in files if PurePosixPath(item.filename).name == "manifest.json"]
        if len(manifests) != 1:
            raise PromotionError("experimental archive must contain one manifest.json")
        manifest_payload = archive.read(manifests[0])
        if len(manifest_payload) > MAX_JSON_BYTES:
            raise PromotionError("experimental package manifest is too large")
        try:
            manifest = json.loads(manifest_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PromotionError(f"invalid experimental package manifest: {error}") from error
        if not isinstance(manifest, dict):
            raise PromotionError("experimental package manifest must be an object")
        if manifest.get("schema") != "denoize-dpdfnet-experimental-clap-package-v1" or manifest.get("schema_version") != 1:
            raise PromotionError("unsupported experimental package manifest")
        if manifest.get("source_commit") != source_commit or manifest.get("target") != "x86_64-pc-windows-msvc":
            raise PromotionError("experimental package binds the wrong source or target")
        expected_scope = {
            "format": "clap",
            "descriptor_count": 3,
            "experimental_descriptor_id": "org.penguin425.denoize.neural-hq",
            "vst3_extended": False,
            "auv3_extended": False,
            "lv2_extended": False,
        }
        if manifest.get("scope") != expected_scope:
            raise PromotionError("experimental package format scope differs from policy")
        model = manifest.get("model")
        if not isinstance(model, dict) or model.get("id") != "dpdfnet2-48khz-hr" or model.get("sha256") != MODEL_SHA256:
            raise PromotionError("experimental package manifest binds the wrong model")
        root = PurePosixPath(manifests[0].filename).parent
        expected_model = root / "models/dpdfnet2-48khz-hr/dpdfnet2_48khz_hr.onnx"
        expected_plugin = root / "denoize.clap"
        by_name = {PurePosixPath(item.filename): item for item in files}
        if expected_model not in by_name or expected_plugin not in by_name:
            raise PromotionError("experimental archive lacks the Windows CLAP or pinned model")
        model_payload = archive.read(by_name[expected_model])
        plugin_payload = archive.read(by_name[expected_plugin])
        if sha256(model_payload) != MODEL_SHA256 or sha256(plugin_payload) != manifest.get("plugin_sha256"):
            raise PromotionError("experimental archive component fingerprint mismatch")
    payload = path.read_bytes()
    return {
        "archive": file_record(path, payload),
        "package_manifest_sha256": sha256(manifest_payload),
        "target": "x86_64-pc-windows-msvc",
        "plugin_sha256": manifest["plugin_sha256"],
        "model_sha256": MODEL_SHA256,
        "descriptor_count": 3,
    }


def reporter_passed(document: dict[str, Any]) -> bool:
    schema = document.get("schema")
    if schema == "denoize-dpdfnet-reporter-evidence-v1":
        return document.get("accepted") is True
    if schema != "denoize-dpdfnet-reporter-evidence-v2":
        raise PromotionError("unsupported issue-reporter evidence")
    checks = document.get("checks")
    if not isinstance(checks, list):
        raise PromotionError("issue-reporter v2 checks are missing")
    by_id = {
        item.get("id"): item
        for item in checks
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if set(by_id) != REPORTER_V2_CHECKS or len(by_id) != len(checks):
        raise PromotionError("issue-reporter v2 checks are incomplete or duplicated")
    passed = all(
        item.get("passed") is True
        and item.get("observed") == 1
        and item.get("operator") == "greater-or-equal"
        and item.get("limit") == 1
        for item in by_id.values()
    )
    if document.get("accepted") is not passed:
        raise PromotionError("issue-reporter v2 accepted flag differs from its checks")
    return passed


def objective_checks(summary: dict[str, Any], source_commit: str) -> list[dict[str, Any]]:
    if summary.get("schema") != "denoize-dpdfnet-gtcrn-evaluation-summary-v1":
        raise PromotionError("unsupported objective-evaluation summary")
    if summary.get("source_commit") != source_commit:
        raise PromotionError("objective evaluation binds a different source commit")
    models = summary.get("models")
    if not isinstance(models, dict) or models.get("dpdfnet2-48khz-hr") != MODEL_SHA256 or models.get("gtcrn-dns3") != GTCRN_SHA256:
        raise PromotionError("objective evaluation binds the wrong model identities")
    values = [
        ("objective-noise-cases", nested(summary, "quality.case_counts.noise_matrix"), ">=", 280),
        ("objective-clean-cases", nested(summary, "quality.case_counts.clean_preservation"), ">=", 10),
        ("objective-speakers", nested(summary, "quality.case_counts.speakers"), ">=", 10),
        ("objective-noises", nested(summary, "quality.case_counts.noises"), ">=", 7),
        ("objective-si-sdr-ci-lower", nested(summary, "quality.dpdfnet2_vs_gtcrn.si_sdr_improvement_db.speaker_cluster_bootstrap_95ci")[0], ">=", 0.0),
        ("objective-stoi-ci-lower", nested(summary, "quality.dpdfnet2_vs_gtcrn.stoi_improvement.speaker_cluster_bootstrap_95ci")[0], ">=", 0.0),
        ("objective-musical-noise-ci-lower", nested(summary, "quality.dpdfnet2_vs_gtcrn.musical_noise.speaker_cluster_bootstrap_95ci")[0], ">=", 0.0),
        ("objective-babble-ci-lower", nested(summary, "quality.strata.noise.three-talker-babble.speaker_cluster_bootstrap_95ci")[0], ">=", 0.0),
    ]
    checks = []
    for identifier, observed_value, operator, limit in values:
        observed = finite(observed_value, identifier)
        passed = observed >= limit
        checks.append({"id": identifier, "observed": observed, "operator": "greater-or-equal", "limit": limit, "passed": passed})
    return checks


def generate(args: argparse.Namespace) -> bool:
    source_commit = args.source_commit
    if not COMMIT_RE.fullmatch(source_commit):
        raise PromotionError("source commit must be a lowercase 40-character SHA-1")
    summary, summary_payload = load(args.evaluation_summary, "objective evaluation")
    listening, listening_payload = load(args.listening_result, "blinded-listening result")
    automated, automated_payload = load(args.reaper_automated, "automated REAPER evidence")
    reporter, reporter_payload = load(args.reporter_evidence, "issue-reporter evidence")
    if listening.get("schema") != "denoize-dpdfnet-blind-listening-result-v1":
        raise PromotionError("unsupported blinded-listening result")
    objective_equivalence = objective_equivalence_record(
        summary,
        args.evaluation_summary,
        summary_payload,
        listening,
        getattr(args, "objective_equivalence", None),
        source_commit,
    )
    if automated.get("schema") != "denoize-dpdfnet-reaper-automated-evidence-v1" or automated.get("source_commit") != source_commit:
        raise PromotionError("automated REAPER evidence binds the wrong schema or source")
    if nested(reporter, "payload.source_commit") != source_commit:
        raise PromotionError("issue-reporter evidence binds the wrong schema or source")
    reporter_is_accepted = reporter_passed(reporter)

    archive = inspect_windows_archive(args.artifact, source_commit)
    if nested(reporter, "payload.artifact_sha256") != archive["archive"]["sha256"]:
        raise PromotionError("issue reporter tested a different experimental archive")
    artifact_verification = verify_attestation(args.artifact, args.artifact_attestation, source_commit)
    automated_verification = verify_attestation(args.reaper_automated, args.artifact_attestation, source_commit)

    if len(args.platform_evidence) != len(args.platform_attestation):
        raise PromotionError("each platform evidence file requires its matching Sigstore bundle")
    platforms: list[dict[str, Any]] = []
    operating_systems: set[str] = set()
    platform_slots: set[tuple[str, str]] = set()
    lowest_tier = 0
    for path, bundle in zip(args.platform_evidence, args.platform_attestation, strict=True):
        document, payload = load(path, "platform evidence")
        platform_schema = document.get("schema")
        platform_versions = {
            "denoize-dpdfnet-platform-evidence-v1": 1,
            "denoize-dpdfnet-platform-evidence-v2": 2,
        }
        if (
            platform_schema not in platform_versions
            or document.get("schema_version") != platform_versions[platform_schema]
            or document.get("source_commit") != source_commit
        ):
            raise PromotionError(f"platform evidence binds the wrong schema or source: {path}")
        operating_system = nested(document, "platform.os")
        hardware_tier = nested(document, "platform.hardware_tier")
        if (
            platform_schema == "denoize-dpdfnet-platform-evidence-v2"
            and nested(document, "measurement.stress_realtime_paced") is not True
        ):
            raise PromotionError(
                "v2 platform promotion evidence must record real-time pacing"
            )
        if platform_schema == "denoize-dpdfnet-platform-evidence-v2":
            deadline_clock = nested(document, "measurement.deadline_clock")
            compute_rtf_clock = nested(document, "measurement.compute_rtf_clock")
            direct_call_deadline_gate_eligible = nested(
                document, "measurement.direct_call_deadline_gate_eligible"
            )
            wall_clock_worker_gate_eligible = nested(
                document, "measurement.wall_clock_worker_gate_eligible"
            )
            expected_deadline_clock = (
                "process-cpu" if operating_system == "macos" else "monotonic-wall"
            )
            expected_compute_rtf_clock = (
                "process-cpu"
                if operating_system in {"macos", "windows"}
                else "monotonic-wall"
            )
            if deadline_clock != expected_deadline_clock:
                raise PromotionError(
                    f"v2 {operating_system} platform evidence must use "
                    f"{expected_deadline_clock} deadlines"
                )
            if compute_rtf_clock != expected_compute_rtf_clock:
                raise PromotionError(
                    f"v2 {operating_system} platform evidence must use "
                    f"{expected_compute_rtf_clock} compute RTF"
                )
            expected_direct_call_gate = False
            expected_worker_gate = hardware_tier == "portable-ci"
            if direct_call_deadline_gate_eligible is not expected_direct_call_gate:
                raise PromotionError(
                    f"v2 {hardware_tier} platform evidence has invalid "
                    "direct-call deadline eligibility"
                )
            if wall_clock_worker_gate_eligible is not expected_worker_gate:
                raise PromotionError(
                    f"v2 {hardware_tier} platform evidence has invalid "
                    "wall-clock worker eligibility"
                )
            direct_call_check_ids = {
                "stress-p99-9-ms",
                "stress-maximum-ms",
                "stress-deadline-misses",
            }
            observed_check_ids = {
                item.get("id")
                for item in document.get("checks", [])
                if isinstance(item, dict)
            }
            has_any_direct_call_gate = bool(
                direct_call_check_ids & observed_check_ids
            )
            if has_any_direct_call_gate:
                raise PromotionError(
                    f"v2 {hardware_tier} platform evidence unexpectedly applies "
                    "the direct-call gate"
                )
            worker_gate_id = (
                "worker-error-counters"
                if wall_clock_worker_gate_eligible
                else "worker-processing-errors"
            )
            unexpected_worker_gate_id = (
                "worker-processing-errors"
                if wall_clock_worker_gate_eligible
                else "worker-error-counters"
            )
            if (
                worker_gate_id not in observed_check_ids
                or unexpected_worker_gate_id in observed_check_ids
            ):
                raise PromotionError(
                    f"v2 {hardware_tier} platform evidence applies the wrong worker gate"
                )
            wall_p99_9_ms = nested(document, "measurement.wall_p99_9_ms")
        else:
            deadline_clock = "monotonic-wall"
            compute_rtf_clock = "monotonic-wall"
            direct_call_deadline_gate_eligible = True
            wall_clock_worker_gate_eligible = True
            wall_p99_9_ms = nested(document, "measurement.p99_9_ms")
        if (
            hardware_tier == "lowest-supported"
            and platform_schema != "denoize-dpdfnet-platform-evidence-v2"
        ):
            raise PromotionError(
                "lowest-supported promotion evidence must use the real-time-paced v2 schema"
            )
        slot = (operating_system, hardware_tier)
        if slot in platform_slots:
            raise PromotionError(
                f"duplicate {hardware_tier} platform evidence for {operating_system}"
            )
        platform_slots.add(slot)
        operating_systems.add(operating_system)
        lowest_tier += int(hardware_tier == "lowest-supported")
        platforms.append({
            "file": file_record(path, payload),
            "attestation": verify_attestation(path, bundle, source_commit),
            "os": operating_system,
            "target": nested(document, "platform.target"),
            "cpu_model": nested(document, "platform.cpu_model"),
            "hardware_tier": hardware_tier,
            "runner_label": nested(document, "platform.runner_label"),
            "deadline_clock": deadline_clock,
            "compute_rtf_clock": compute_rtf_clock,
            "direct_call_deadline_gate_eligible": direct_call_deadline_gate_eligible,
            "wall_clock_worker_gate_eligible": wall_clock_worker_gate_eligible,
            "p99_9_ms": nested(document, "measurement.p99_9_ms"),
            "wall_p99_9_ms": wall_p99_9_ms,
            "peak_rss_bytes": nested(document, "measurement.peak_rss_bytes"),
            "accepted": document.get("accepted") is True and all(item.get("passed") is True for item in document.get("checks", [])),
        })

    objective = objective_checks(summary, source_commit)
    checks = [
        *objective,
        {
            "id": "blinded-listening",
            "observed": int(listening.get("accepted") is True and all(item.get("passed") is True for item in listening.get("checks", []))),
            "operator": "greater-or-equal", "limit": 1, "passed": listening.get("accepted") is True and all(item.get("passed") is True for item in listening.get("checks", [])),
        },
        {
            "id": "portable-operating-systems",
            "observed": len(operating_systems), "operator": "greater-or-equal", "limit": 3,
            "passed": operating_systems == {"linux", "macos", "windows"} and all(item["accepted"] for item in platforms),
        },
        {
            "id": "lowest-supported-hardware",
            "observed": lowest_tier, "operator": "greater-or-equal", "limit": 1, "passed": lowest_tier >= 1,
        },
        {
            "id": "automated-reaper",
            "observed": int(automated.get("accepted_automated") is True), "operator": "greater-or-equal", "limit": 1,
            "passed": automated.get("accepted_automated") is True,
        },
        {
            "id": "issue-reporter-nvda-osara",
            "observed": int(reporter_is_accepted), "operator": "greater-or-equal", "limit": 1,
            "passed": reporter_is_accepted,
        },
        {
            "id": "attested-windows-experimental-clap",
            "observed": artifact_verification["verified_attestations"], "operator": "greater-or-equal", "limit": 1,
            "passed": artifact_verification["verified_attestations"] >= 1 and automated_verification["verified_attestations"] >= 1,
        },
    ]
    accepted = all(item["passed"] for item in checks)
    document = {
        "schema": "denoize-dpdfnet-promotion-evidence-v1",
        "schema_version": 1,
        "source": {"repository": "penguin425/denoize", "commit": source_commit, "issue": 221},
        "candidate": {
            "model_id": "dpdfnet2-48khz-hr",
            "model_sha256": MODEL_SHA256,
            "plugin_id": "org.penguin425.denoize.neural-hq",
        },
        "baseline": {"model_id": "gtcrn-dns3", "model_sha256": GTCRN_SHA256},
        "inputs": {
            "objective_evaluation": file_record(args.evaluation_summary, summary_payload),
            "blinded_listening": file_record(args.listening_result, listening_payload),
            "automated_reaper": file_record(args.reaper_automated, automated_payload),
            "issue_reporter": file_record(args.reporter_evidence, reporter_payload),
            **(
                {"objective_equivalence": objective_equivalence}
                if objective_equivalence is not None
                else {}
            ),
        },
        "artifact": {
            **archive,
            "attestation": artifact_verification,
            "automated_reaper_attestation": automated_verification,
        },
        "platforms": sorted(
            platforms, key=lambda item: (item["os"], item["hardware_tier"])
        ),
        "checks": checks,
        "decision": {
            "dpdfnet2_selectable_hq": accepted,
            "eligible_for_default_review": accepted,
            "keep_gtcrn_selectable": True,
            "include_dpdfnet8": False,
            "normal_clap_descriptor_count_before_decision": 2,
            "experimental_clap_descriptor_count": 3,
            "expand_vst3": False,
            "expand_auv3": False,
            "expand_lv2": False,
        },
        "accepted": accepted,
    }
    output = args.output
    if output.exists() or output.is_symlink():
        raise PromotionError(f"refusing to replace existing promotion evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as destination:
            descriptor = -1
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return accepted


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-commit", required=True)
    result.add_argument("--evaluation-summary", type=Path, required=True)
    result.add_argument("--objective-equivalence", type=Path)
    result.add_argument("--listening-result", type=Path, required=True)
    result.add_argument("--platform-evidence", type=Path, nargs="+", required=True)
    result.add_argument("--platform-attestation", type=Path, nargs="+", required=True)
    result.add_argument("--reaper-automated", type=Path, required=True)
    result.add_argument("--reporter-evidence", type=Path, required=True)
    result.add_argument("--artifact", type=Path, required=True)
    result.add_argument("--artifact-attestation", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--allow-rejected", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        accepted = generate(args)
    except (PromotionError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not accepted and not args.allow_rejected:
        print("error: one or more DPDFNet promotion gates remain open", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
