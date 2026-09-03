#!/usr/bin/env python3
"""Verify and score blinded DPDFNet/GTCRN paired-listening responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any


MODEL_DPDFNET = "dpdfnet2-48khz-hr"
MODEL_GTCRN = "gtcrn-dns3"
RECOVERED_CORE_SELECTION = "lexicographically-smallest-trial-id-v1"
LISTENER_RE = re.compile(r"^[A-Za-z0-9._-]{3,64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STRATA = ("recorded-noise", "babble", "source-preservation", "synthetic-noise")
MAX_JSON_BYTES = 8 * 1024 * 1024


class ScoreError(RuntimeError):
    pass


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScoreError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ScoreError(f"{label} is not a regular file: {path}")
    path = path.resolve()
    size = path.stat().st_size
    if not 1 <= size <= MAX_JSON_BYTES:
        raise ScoreError(f"{label} size must be in 1..={MAX_JSON_BYTES} bytes")
    payload = path.read_bytes()
    try:
        document = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScoreError(f"invalid {label} JSON {path}: {error}") from error
    if not isinstance(document, dict):
        raise ScoreError(f"{label} must be a JSON object: {path}")
    return document, payload


def exact_keys(document: dict, expected: set[str], label: str) -> None:
    actual = set(document)
    if actual != expected:
        raise ScoreError(f"{label} keys differ: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoreError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ScoreError(f"{label} must be finite")
    return result


def validate_protocol(protocol: dict, payload: bytes, path: Path) -> dict[str, dict]:
    exact_keys(protocol, {"schema", "schema_version", "bundle_id", "source_matrix_sha256", "policy", "trials"}, "protocol")
    if protocol["schema"] != "denoize-dpdfnet-blind-protocol-v1" or protocol["schema_version"] != 1:
        raise ScoreError("unsupported blinded-listening protocol")
    if not re.fullmatch(r"[0-9a-f]{24}", protocol["bundle_id"]):
        raise ScoreError("invalid protocol bundle_id")
    if not SHA256_RE.fullmatch(protocol["source_matrix_sha256"]):
        raise ScoreError("invalid source_matrix_sha256")
    policy = protocol["policy"]
    exact_keys(
        policy,
        {
            "core_trials", "repeat_trials", "minimum_retained_listeners",
            "maximum_listener_duplicate_inconsistency", "maximum_aggregate_duplicate_inconsistency",
            "minimum_overall_dpdfnet_preference", "minimum_overall_bootstrap_95ci_lower",
            "minimum_stratum_dpdfnet_preference", "tie_score", "listener_cluster_bootstrap_resamples",
        },
        "protocol policy",
    )
    expected_policy = {
        "core_trials": 12,
        "repeat_trials": 4,
        "minimum_retained_listeners": 20,
        "maximum_listener_duplicate_inconsistency": 0.5,
        "maximum_aggregate_duplicate_inconsistency": 0.25,
        "minimum_overall_dpdfnet_preference": 0.55,
        "minimum_overall_bootstrap_95ci_lower": 0.5,
        "minimum_stratum_dpdfnet_preference": {
            "recorded-noise": 0.5,
            "babble": 0.4,
            "source-preservation": 0.45,
            "synthetic-noise": 0.5,
        },
        "tie_score": 0.5,
        "listener_cluster_bootstrap_resamples": 20_000,
    }
    if policy != expected_policy:
        raise ScoreError("protocol policy differs from the predeclared promotion policy")
    trials = protocol["trials"]
    if not isinstance(trials, list) or len(trials) != 16:
        raise ScoreError("protocol must contain exactly 16 trials")
    by_id: dict[str, dict] = {}
    protocol_root = path.resolve().parent
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ScoreError(f"protocol trial {index} must be an object")
        exact_keys(trial, {"trial_id", "stratum", "question", "audio"}, f"protocol trial {index}")
        trial_id = trial["trial_id"]
        if not isinstance(trial_id, str) or not re.fullmatch(r"[0-9a-f]{24}", trial_id) or trial_id in by_id:
            raise ScoreError(f"invalid or duplicate trial_id: {trial_id}")
        if trial["stratum"] not in STRATA:
            raise ScoreError(f"invalid trial stratum: {trial['stratum']}")
        expected_question = "source-preservation" if trial["stratum"] == "source-preservation" else "noise-reduction"
        if trial["question"] != expected_question:
            raise ScoreError(f"question does not match stratum for {trial_id}")
        audio = trial["audio"]
        if not isinstance(audio, dict):
            raise ScoreError(f"audio record must be an object for {trial_id}")
        exact_keys(audio, {"reference", "input", "a", "b"}, f"audio record {trial_id}")
        for label, record in audio.items():
            if not isinstance(record, dict):
                raise ScoreError(f"audio {trial_id}/{label} must be an object")
            exact_keys(record, {"path", "size_bytes", "sha256", "sample_rate_hz", "channels", "frames"}, f"audio {trial_id}/{label}")
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ScoreError(f"unsafe audio path: {relative}")
            audio_path = protocol_root / relative
            if audio_path.is_symlink() or not audio_path.is_file():
                raise ScoreError(f"audio is not a regular file: {audio_path}")
            resolved = audio_path.resolve()
            if not resolved.is_relative_to(protocol_root):
                raise ScoreError(f"audio escapes bundle root: {audio_path}")
            audio_payload = resolved.read_bytes()
            if record["size_bytes"] != len(audio_payload) or record["sha256"] != sha256(audio_payload):
                raise ScoreError(f"audio fingerprint mismatch: {audio_path}")
            if record["sample_rate_hz"] != 48_000 or record["channels"] != 1 or not isinstance(record["frames"], int) or record["frames"] <= 0:
                raise ScoreError(f"audio geometry mismatch: {audio_path}")
        by_id[trial_id] = trial
    if sha256(payload) == "0" * 64:
        raise ScoreError("unreachable protocol digest")
    return by_id


def validate_answer_key(
    answer: dict,
    protocol: dict,
    protocol_payload: bytes,
    protocol_trials: dict[str, dict],
) -> dict[str, dict]:
    schema = answer.get("schema")
    version = answer.get("schema_version")
    recovered = schema == "denoize-dpdfnet-blind-answer-key-v2" and version == 2
    if schema == "denoize-dpdfnet-blind-answer-key-v1" and version == 1:
        exact_keys(
            answer,
            {
                "schema",
                "schema_version",
                "bundle_id",
                "protocol_sha256",
                "source_matrix_sha256",
                "randomization_key_sha256",
                "trials",
            },
            "answer key",
        )
        if not is_sha256(answer["randomization_key_sha256"]):
            raise ScoreError("invalid randomization-key digest")
    elif recovered:
        exact_keys(
            answer,
            {
                "schema",
                "schema_version",
                "bundle_id",
                "protocol_sha256",
                "source_matrix_sha256",
                "recovery",
                "trials",
            },
            "answer key",
        )
        recovery = answer["recovery"]
        if not isinstance(recovery, dict):
            raise ScoreError("answer-key recovery must be an object")
        exact_keys(
            recovery,
            {
                "method",
                "core_selection",
                "candidate_audio_manifest_sha256",
                "source_case_count",
            },
            "answer-key recovery",
        )
        if recovery["method"] != "named-output-sha256-v1":
            raise ScoreError("unsupported answer-key recovery method")
        if recovery["core_selection"] != RECOVERED_CORE_SELECTION:
            raise ScoreError("unsupported recovered core-selection rule")
        if not is_sha256(recovery["candidate_audio_manifest_sha256"]):
            raise ScoreError("invalid candidate-audio manifest digest")
        if recovery["source_case_count"] != 12:
            raise ScoreError("recovered answer key must bind exactly 12 source cases")
    else:
        raise ScoreError("unsupported blinded-listening answer key")
    if answer["bundle_id"] != protocol["bundle_id"] or answer["source_matrix_sha256"] != protocol["source_matrix_sha256"]:
        raise ScoreError("answer key does not bind the protocol source")
    if answer["protocol_sha256"] != sha256(protocol_payload):
        raise ScoreError("answer key protocol digest mismatch")
    trials = answer["trials"]
    if not isinstance(trials, list) or len(trials) != len(protocol_trials):
        raise ScoreError("answer key trial count mismatch")
    by_id: dict[str, dict] = {}
    core = 0
    repeat = 0
    strata: dict[str, int] = {name: 0 for name in STRATA}
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ScoreError(f"answer trial {index} must be an object")
        trial_keys = {
            "trial_id",
            "source_case_id",
            "stratum",
            "role",
            "duplicate_of",
            "a_model",
            "b_model",
        }
        if recovered:
            trial_keys |= {"a_sha256", "b_sha256"}
        exact_keys(trial, trial_keys, f"answer trial {index}")
        trial_id = trial["trial_id"]
        if (
            not isinstance(trial_id, str)
            or trial_id not in protocol_trials
            or trial_id in by_id
        ):
            raise ScoreError(f"answer key has unknown or duplicate trial: {trial_id}")
        if trial["stratum"] != protocol_trials[trial_id]["stratum"]:
            raise ScoreError(f"answer stratum mismatch: {trial_id}")
        source_case_id = trial["source_case_id"]
        if (
            not isinstance(source_case_id, str)
            or not re.fullmatch(r"[A-Za-z0-9._+-]{1,128}", source_case_id)
        ):
            raise ScoreError(f"invalid answer source case ID: {trial_id}")
        a_model = trial["a_model"]
        b_model = trial["b_model"]
        if (
            not isinstance(a_model, str)
            or not isinstance(b_model, str)
            or {a_model, b_model} != {MODEL_DPDFNET, MODEL_GTCRN}
        ):
            raise ScoreError(f"answer sides do not contain both models: {trial_id}")
        if recovered:
            public_audio = protocol_trials[trial_id]["audio"]
            if trial["a_sha256"] != public_audio["a"]["sha256"]:
                raise ScoreError(f"answer side A digest mismatch: {trial_id}")
            if trial["b_sha256"] != public_audio["b"]["sha256"]:
                raise ScoreError(f"answer side B digest mismatch: {trial_id}")
        if trial["role"] == "core":
            if trial["duplicate_of"] is not None:
                raise ScoreError(f"core trial has duplicate_of: {trial_id}")
            core += 1
            strata[trial["stratum"]] += 1
        elif trial["role"] == "repeat":
            if not isinstance(trial["duplicate_of"], str):
                raise ScoreError(f"repeat trial lacks duplicate_of: {trial_id}")
            repeat += 1
        else:
            raise ScoreError(f"invalid answer role: {trial['role']}")
        by_id[trial_id] = trial
    if core != 12 or repeat != 4 or strata != {"recorded-noise": 4, "babble": 3, "source-preservation": 3, "synthetic-noise": 2}:
        raise ScoreError(f"answer-key trial geometry mismatch: core={core}, repeat={repeat}, strata={strata}")
    core_by_id = {trial_id: trial for trial_id, trial in by_id.items() if trial["role"] == "core"}
    duplicate_targets: set[str] = set()
    for trial in by_id.values():
        if trial["role"] != "repeat":
            continue
        target = core_by_id.get(trial["duplicate_of"])
        if target is None or target["source_case_id"] != trial["source_case_id"] or target["stratum"] != trial["stratum"]:
            raise ScoreError(f"invalid repeat mapping: {trial['trial_id']}")
        if trial["duplicate_of"] in duplicate_targets:
            raise ScoreError(f"core trial repeated more than once: {trial['duplicate_of']}")
        duplicate_targets.add(trial["duplicate_of"])
    if len(duplicate_targets) != 4:
        raise ScoreError("answer key must repeat four distinct core trials")
    if recovered:
        by_case: dict[str, list[dict]] = {}
        for trial in by_id.values():
            by_case.setdefault(trial["source_case_id"], []).append(trial)
        single_count = sum(len(records) == 1 for records in by_case.values())
        pair_count = sum(len(records) == 2 for records in by_case.values())
        if len(by_case) != 12 or single_count != 8 or pair_count != 4:
            raise ScoreError("recovered answer key has invalid source-case geometry")
        for records in by_case.values():
            if len(records) not in {1, 2}:
                raise ScoreError("recovered source case appears outside one or two trials")
            ordered = sorted(records, key=lambda record: record["trial_id"])
            if ordered[0]["role"] != "core" or ordered[0]["duplicate_of"] is not None:
                raise ScoreError("recovered core selection differs from the public rule")
            if len(ordered) == 2 and (
                ordered[1]["role"] != "repeat"
                or ordered[1]["duplicate_of"] != ordered[0]["trial_id"]
            ):
                raise ScoreError("recovered repeat mapping differs from the public rule")
    return by_id


def response_paths(values: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        if value.is_dir() and not value.is_symlink():
            paths.extend(sorted(value.glob("*.json")))
        else:
            paths.append(value)
    resolved = sorted({path.resolve() for path in paths})
    if not resolved:
        raise ScoreError("no listener response JSON files were found")
    return resolved


def canonical_choice(preference: str, answer: dict) -> str:
    if preference == "tie":
        return "tie"
    return answer[f"{preference}_model"]


def validate_response(path: Path, protocol_sha256: str, trial_ids: set[str]) -> tuple[str, dict[str, str], bytes]:
    response, payload = load_json(path, "listener response")
    exact_keys(response, {"schema", "schema_version", "protocol_sha256", "listener_id", "consent", "trials"}, f"response {path.name}")
    if response["schema"] != "denoize-dpdfnet-blind-listener-response-v1" or response["schema_version"] != 1:
        raise ScoreError(f"unsupported response schema: {path}")
    if response["protocol_sha256"] != protocol_sha256:
        raise ScoreError(f"response protocol digest mismatch: {path}")
    listener = response["listener_id"]
    if not isinstance(listener, str) or not LISTENER_RE.fullmatch(listener):
        raise ScoreError(f"invalid pseudonymous listener ID: {path}")
    if response["consent"] is not True:
        raise ScoreError(f"response does not record consent: {path}")
    trials = response["trials"]
    if not isinstance(trials, list) or len(trials) != len(trial_ids):
        raise ScoreError(f"response must answer every trial exactly once: {path}")
    answers: dict[str, str] = {}
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ScoreError(f"response trial {index} must be an object: {path}")
        exact_keys(trial, {"trial_id", "preference"}, f"response trial {index}")
        trial_id = trial["trial_id"]
        preference = trial["preference"]
        if trial_id not in trial_ids or trial_id in answers or preference not in {"a", "b", "tie"}:
            raise ScoreError(f"unknown, duplicate, or invalid response trial: {path}")
        answers[trial_id] = preference
    if set(answers) != trial_ids:
        raise ScoreError(f"response trial set mismatch: {path}")
    return listener, answers, payload


def percentile(sorted_values: list[float], probability: float) -> float:
    index = max(0, min(len(sorted_values) - 1, math.ceil(probability * len(sorted_values)) - 1))
    return sorted_values[index]


def bootstrap_interval(listener_scores: list[float], resamples: int, seed: int) -> list[float]:
    generator = random.Random(seed)
    count = len(listener_scores)
    values = []
    for _ in range(resamples):
        values.append(sum(listener_scores[generator.randrange(count)] for _ in range(count)) / count)
    values.sort()
    return [percentile(values, 0.025), percentile(values, 0.975)]


def write_exclusive(path: Path, document: dict) -> None:
    if path.exists() or path.is_symlink():
        raise ScoreError(f"refusing to replace existing listening result: {path}")
    payload = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
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


def score(args: argparse.Namespace) -> bool:
    protocol_path = args.protocol
    protocol, protocol_payload = load_json(protocol_path, "protocol")
    protocol_path = protocol_path.resolve()
    protocol_trials = validate_protocol(protocol, protocol_payload, protocol_path)
    answer, answer_payload = load_json(args.answer_key, "answer key")
    answers = validate_answer_key(answer, protocol, protocol_payload, protocol_trials)
    protocol_sha256 = sha256(protocol_payload)

    listeners: list[dict[str, Any]] = []
    seen: set[str] = set()
    response_fingerprints: list[str] = []
    for path in response_paths(args.responses):
        listener_id, selections, payload = validate_response(path, protocol_sha256, set(protocol_trials))
        if listener_id in seen:
            raise ScoreError(f"duplicate pseudonymous listener ID: {listener_id}")
        seen.add(listener_id)
        response_fingerprints.append(sha256(payload))
        canonical = {trial_id: canonical_choice(preference, answers[trial_id]) for trial_id, preference in selections.items()}
        inconsistencies = 0
        repeats = 0
        for trial_id, answer_trial in answers.items():
            if answer_trial["role"] == "repeat":
                repeats += 1
                inconsistencies += int(canonical[trial_id] != canonical[answer_trial["duplicate_of"]])
        fraction = inconsistencies / repeats
        core_scores: dict[str, float] = {}
        for trial_id, answer_trial in answers.items():
            if answer_trial["role"] != "core":
                continue
            choice = canonical[trial_id]
            core_scores[trial_id] = 1.0 if choice == MODEL_DPDFNET else 0.5 if choice == "tie" else 0.0
        listeners.append({
            "duplicate_inconsistencies": inconsistencies,
            "duplicate_inconsistency_fraction": fraction,
            "retained": fraction <= protocol["policy"]["maximum_listener_duplicate_inconsistency"],
            "scores": core_scores,
        })
    retained = [listener for listener in listeners if listener["retained"]]
    if not retained:
        raise ScoreError("no listener passed the predeclared duplicate-consistency screen")

    core_answers = {trial_id: trial for trial_id, trial in answers.items() if trial["role"] == "core"}
    listener_scores = [sum(listener["scores"].values()) / len(core_answers) for listener in retained]
    overall = sum(listener_scores) / len(listener_scores)
    interval = bootstrap_interval(
        listener_scores,
        protocol["policy"]["listener_cluster_bootstrap_resamples"],
        int(protocol_sha256[:16], 16),
    )
    stratum_results: dict[str, dict[str, Any]] = {}
    for stratum in STRATA:
        trial_ids = [trial_id for trial_id, trial in core_answers.items() if trial["stratum"] == stratum]
        values = [listener["scores"][trial_id] for listener in retained for trial_id in trial_ids]
        score_value = sum(values) / len(values)
        limit = protocol["policy"]["minimum_stratum_dpdfnet_preference"][stratum]
        stratum_results[stratum] = {
            "core_trials": len(trial_ids),
            "ratings": len(values),
            "dpdfnet_preference_score": score_value,
            "minimum": limit,
            "passed": score_value >= limit,
        }
    duplicate_inconsistencies = sum(listener["duplicate_inconsistencies"] for listener in listeners)
    duplicate_comparisons = len(listeners) * protocol["policy"]["repeat_trials"]
    duplicate_fraction = duplicate_inconsistencies / duplicate_comparisons

    checks = [
        {
            "id": "minimum-retained-listeners",
            "observed": len(retained),
            "operator": "greater-or-equal",
            "limit": protocol["policy"]["minimum_retained_listeners"],
            "passed": len(retained) >= protocol["policy"]["minimum_retained_listeners"],
        },
        {
            "id": "aggregate-duplicate-inconsistency",
            "observed": duplicate_fraction,
            "operator": "less-or-equal",
            "limit": protocol["policy"]["maximum_aggregate_duplicate_inconsistency"],
            "passed": duplicate_fraction <= protocol["policy"]["maximum_aggregate_duplicate_inconsistency"],
        },
        {
            "id": "overall-dpdfnet-preference",
            "observed": overall,
            "operator": "greater-or-equal",
            "limit": protocol["policy"]["minimum_overall_dpdfnet_preference"],
            "passed": overall >= protocol["policy"]["minimum_overall_dpdfnet_preference"],
        },
        {
            "id": "overall-bootstrap-95ci-lower",
            "observed": interval[0],
            "operator": "greater-or-equal",
            "limit": protocol["policy"]["minimum_overall_bootstrap_95ci_lower"],
            "passed": interval[0] >= protocol["policy"]["minimum_overall_bootstrap_95ci_lower"],
        },
    ]
    checks.extend(
        {
            "id": f"stratum-{stratum}",
            "observed": result["dpdfnet_preference_score"],
            "operator": "greater-or-equal",
            "limit": result["minimum"],
            "passed": result["passed"],
        }
        for stratum, result in stratum_results.items()
    )
    accepted = all(check["passed"] for check in checks)
    fingerprint_payload = "\n".join(sorted(response_fingerprints)).encode("ascii")
    result = {
        "schema": "denoize-dpdfnet-blind-listening-result-v1",
        "schema_version": 1,
        "bundle_id": protocol["bundle_id"],
        "source_matrix_sha256": protocol["source_matrix_sha256"],
        "protocol_sha256": protocol_sha256,
        "answer_key_sha256": sha256(answer_payload),
        "response_set_sha256": sha256(fingerprint_payload),
        "models": {"candidate": MODEL_DPDFNET, "baseline": MODEL_GTCRN},
        "listeners": {
            "submitted": len(listeners),
            "retained": len(retained),
            "excluded_by_predeclared_consistency_screen": len(listeners) - len(retained),
        },
        "duplicate_consistency": {
            "comparisons": duplicate_comparisons,
            "inconsistencies": duplicate_inconsistencies,
            "inconsistency_fraction": duplicate_fraction,
            "maximum": protocol["policy"]["maximum_aggregate_duplicate_inconsistency"],
            "passed": duplicate_fraction <= protocol["policy"]["maximum_aggregate_duplicate_inconsistency"],
        },
        "overall": {
            "ratings": len(retained) * len(core_answers),
            "dpdfnet_preference_score": overall,
            "listener_cluster_bootstrap_95ci": interval,
            "tie_score": protocol["policy"]["tie_score"],
        },
        "strata": stratum_results,
        "checks": checks,
        "accepted": accepted,
    }
    write_exclusive(args.output, result)
    return accepted


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--protocol", type=Path, required=True)
    result.add_argument("--answer-key", type=Path, required=True)
    result.add_argument("--responses", type=Path, nargs="+", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--allow-rejected", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        accepted = score(args)
    except (ScoreError, OSError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 1
    if not accepted and not args.allow_rejected:
        print("error: blinded-listening promotion thresholds were not met", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
