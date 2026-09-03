#!/usr/bin/env python3
"""Fetch and validate issue #221 reporter evidence from a GitHub comment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable


COMMENT_RE = re.compile(
    r"^https://github\.com/penguin425/denoize/issues/221#issuecomment-([1-9][0-9]*)$"
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ATTACHMENT_RE = re.compile(
    r"^https://github\.com/user-attachments/files/[1-9][0-9]*/"
    r"[A-Za-z0-9._-]{1,160}\.json$"
)
PARAMETERS = ["Bypass", "Mix", "Output Gain", "Overload Fallback"]
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MODEL_ID = "dpdfnet2-48khz-hr"
MODEL_SHA256 = "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b"
PLUGIN_ID = "org.penguin425.denoize.neural-hq"


class ReporterError(RuntimeError):
    pass


def api(path: str) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "denoize-dpdfnet-promotion-evidence-v2",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"https://api.github.com{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.URLError as error:
        raise ReporterError(f"GitHub API request failed: {error}") from error
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ReporterError("GitHub API response exceeds the size limit")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReporterError(f"GitHub API returned invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise ReporterError("GitHub API response must be an object")
    return document


def attachment(url: str) -> bytes:
    if not ATTACHMENT_RE.fullmatch(url):
        raise ReporterError("host evidence URL must be a bounded GitHub user attachment JSON")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "denoize-dpdfnet-promotion-evidence-v2"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.URLError as error:
        raise ReporterError(f"host evidence download failed: {error}") from error
    if not 1 <= len(payload) <= MAX_RESPONSE_BYTES:
        raise ReporterError("host evidence attachment size is outside the allowed range")
    return payload


def exact(document: dict, keys: set[str], label: str) -> None:
    if set(document) != keys:
        raise ReporterError(f"{label} has missing or unknown fields")


def bounded(value: Any, label: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum or "\x00" in value:
        raise ReporterError(f"{label} must be a bounded non-empty string")
    return value


def integer(value: Any, label: str, minimum: int = 0, maximum: int = 9_007_199_254_740_991) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ReporterError(f"{label} must be an integer in {minimum}..={maximum}")
    return value


def boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ReporterError(f"{label} must be a boolean")
    return value


def parse_json(payload: bytes | str, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReporterError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReporterError(f"invalid {label} JSON: {error}") from error
    if not isinstance(document, dict):
        raise ReporterError(f"{label} must be an object")
    return document


def validate_payload_v1(payload: dict[str, Any]) -> None:
    exact(
        payload,
        {
            "schema", "schema_version", "source_commit", "artifact_sha256", "environment",
            "runs", "accessibility", "quality_observation", "consent_to_publish",
        },
        "reporter payload",
    )
    if payload["schema"] != "denoize-dpdfnet-reporter-submission-v1" or payload["schema_version"] != 1:
        raise ReporterError("unsupported reporter-submission schema")
    if not isinstance(payload["source_commit"], str) or not COMMIT_RE.fullmatch(payload["source_commit"]):
        raise ReporterError("reporter submission has an invalid source commit")
    if not isinstance(payload["artifact_sha256"], str) or not SHA256_RE.fullmatch(payload["artifact_sha256"]):
        raise ReporterError("reporter submission has an invalid artifact digest")
    if payload["consent_to_publish"] is not True:
        raise ReporterError("reporter did not consent to publish the evidence")
    if payload["quality_observation"] not in {"dpdfnet-better", "equivalent", "gtcrn-better"}:
        raise ReporterError("quality_observation is invalid")
    environment = payload["environment"]
    if not isinstance(environment, dict):
        raise ReporterError("environment must be an object")
    exact(environment, {"windows_version", "cpu_model", "audio_device", "audio_driver", "reaper_version", "nvda_version", "osara_version"}, "reporter environment")
    for name, value in environment.items():
        bounded(value, name)
    version_match = re.fullmatch(r"([0-9]+)\.([0-9]+)", environment["reaper_version"])
    if version_match is None or tuple(map(int, version_match.groups())) < (7, 79):
        raise ReporterError("REAPER must be version 7.79 or newer")

    runs = payload["runs"]
    if not isinstance(runs, list) or not 3 <= len(runs) <= 16:
        raise ReporterError("reporter submission must contain 3..=16 buffer runs")
    buffers: set[int] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ReporterError(f"run {index} must be an object")
        exact(run, {"buffer_frames", "sample_rate_hz", "duration_seconds", "overload_events", "late_events", "audible_xruns", "continuous_audio"}, f"run {index}")
        for name in ("buffer_frames", "sample_rate_hz", "duration_seconds", "overload_events", "late_events", "audible_xruns"):
            if isinstance(run[name], bool) or not isinstance(run[name], int):
                raise ReporterError(f"run {index} {name} must be an integer")
        if run["sample_rate_hz"] != 48_000 or run["duration_seconds"] < 300:
            raise ReporterError(f"run {index} must cover at least 300 seconds at 48 kHz")
        if run["overload_events"] != 0 or run["late_events"] != 0 or run["audible_xruns"] != 0 or run["continuous_audio"] is not True:
            raise ReporterError(f"run {index} did not pass the realtime gate")
        if not 16 <= run["buffer_frames"] <= 8192:
            raise ReporterError(f"run {index} buffer size is outside 16..=8192")
        if run["buffer_frames"] in buffers:
            raise ReporterError("buffer runs must be unique")
        buffers.add(run["buffer_frames"])
    if min(buffers) > 128 or 480 not in buffers or max(buffers) < 1024:
        raise ReporterError("buffer coverage must include <=128, exactly 480, and >=1024 frames")

    accessibility = payload["accessibility"]
    if not isinstance(accessibility, dict):
        raise ReporterError("accessibility must be an object")
    exact(accessibility, {"nvda_active", "osara_active", "parameters_announced", "values_announced", "all_adjustable", "focus_stable", "host_or_plugin_crashes"}, "accessibility")
    if accessibility["parameters_announced"] != PARAMETERS:
        raise ReporterError("NVDA/OSARA must announce the four closed parameter names in order")
    if any(accessibility[name] is not True for name in ("nvda_active", "osara_active", "values_announced", "all_adjustable", "focus_stable")):
        raise ReporterError("one or more NVDA/OSARA checks failed")
    if accessibility["host_or_plugin_crashes"] != 0:
        raise ReporterError("the human test recorded a host or plug-in crash")


def validate_payload_v2(payload: dict[str, Any]) -> None:
    exact(
        payload,
        {
            "schema", "schema_version", "source_commit", "artifact_sha256", "environment",
            "runs", "accessibility", "quality_observation", "consent_to_publish",
        },
        "reporter payload",
    )
    if payload["schema"] != "denoize-dpdfnet-reporter-submission-v2" or payload["schema_version"] != 2:
        raise ReporterError("unsupported reporter-submission schema")
    if not isinstance(payload["source_commit"], str) or not COMMIT_RE.fullmatch(payload["source_commit"]):
        raise ReporterError("reporter submission has an invalid source commit")
    if not isinstance(payload["artifact_sha256"], str) or not SHA256_RE.fullmatch(payload["artifact_sha256"]):
        raise ReporterError("reporter submission has an invalid artifact digest")
    if payload["consent_to_publish"] is not True:
        raise ReporterError("reporter did not consent to publish the evidence")
    if payload["quality_observation"] not in {"dpdfnet-better", "equivalent", "gtcrn-better"}:
        raise ReporterError("quality_observation is invalid")

    environment = payload["environment"]
    if not isinstance(environment, dict):
        raise ReporterError("environment must be an object")
    exact(
        environment,
        {"windows_version", "cpu_model", "audio_device", "audio_driver", "reaper_version", "nvda_version", "osara_version"},
        "reporter environment",
    )
    for name, value in environment.items():
        bounded(value, name)

    runs = payload["runs"]
    if not isinstance(runs, list) or not 3 <= len(runs) <= 16:
        raise ReporterError("reporter submission must contain 3..=16 buffer runs")
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ReporterError(f"run {index} must be an object")
        exact(
            run,
            {"requested_buffer_frames", "host_evidence_url", "host_evidence_sha256", "audible_xruns", "continuous_audio"},
            f"run {index}",
        )
        integer(run["requested_buffer_frames"], f"run {index} requested_buffer_frames", 16, 8192)
        if not isinstance(run["host_evidence_url"], str) or not ATTACHMENT_RE.fullmatch(run["host_evidence_url"]):
            raise ReporterError(f"run {index} has an invalid host evidence URL")
        if not isinstance(run["host_evidence_sha256"], str) or not SHA256_RE.fullmatch(run["host_evidence_sha256"]):
            raise ReporterError(f"run {index} has an invalid host evidence digest")
        audible_xruns = run["audible_xruns"]
        continuous_audio = boolean(run["continuous_audio"], f"run {index} continuous_audio")
        if audible_xruns is None:
            if continuous_audio:
                raise ReporterError(
                    f"run {index} audible_xruns may be null only when continuous_audio is false"
                )
        else:
            integer(audible_xruns, f"run {index} audible_xruns")

    accessibility = payload["accessibility"]
    if not isinstance(accessibility, dict):
        raise ReporterError("accessibility must be an object")
    exact(
        accessibility,
        {"nvda_active", "osara_active", "parameters_announced", "values_announced", "all_adjustable", "focus_stable", "host_or_plugin_crashes"},
        "accessibility",
    )
    for name in ("nvda_active", "osara_active", "values_announced", "all_adjustable", "focus_stable"):
        boolean(accessibility[name], f"accessibility {name}")
    announced = accessibility["parameters_announced"]
    if not isinstance(announced, list) or len(announced) > 16:
        raise ReporterError("parameters_announced must be a bounded array")
    for index, name in enumerate(announced):
        bounded(name, f"parameters_announced {index}")
    integer(accessibility["host_or_plugin_crashes"], "host_or_plugin_crashes")


def validate_metrics(metrics: Any, label: str) -> dict[str, int]:
    if not isinstance(metrics, dict):
        raise ReporterError(f"{label} must be an object")
    names = {"overload_blocks", "late_blocks", "invalid_blocks", "worker_errors"}
    exact(metrics, names, label)
    return {name: integer(metrics[name], f"{label} {name}") for name in names}


def normalize_host_evidence(
    submission: dict[str, Any],
    run: dict[str, Any],
    payload: bytes,
) -> dict[str, Any]:
    digest = hashlib.sha256(payload).hexdigest()
    if digest != run["host_evidence_sha256"]:
        raise ReporterError("host evidence attachment digest mismatch")
    host = parse_json(payload, "host evidence")
    schema = host.get("schema")
    common = {
        "schema", "schema_version", "source_commit", "model_id", "model_sha256", "plugin_id",
        "sample_rate_hz", "channels", "chunk_frames", "latency_frames", "processed_frames",
        "active_seconds", "measurement", "worker_started", "finished_gracefully", "metrics",
        "lifetime_metrics", "environment",
    }
    if schema == "denoize-dpdfnet-clap-host-run-v1":
        if host.get("schema_version") != 1:
            raise ReporterError("host evidence v1 has the wrong schema version")
        exact(host, common, "host evidence v1")
        effective_buffer = None
    elif schema == "denoize-dpdfnet-clap-host-run-v2":
        if host.get("schema_version") != 2:
            raise ReporterError("host evidence v2 has the wrong schema version")
        exact(host, common | {"host_audio_configuration", "callback_frames"}, "host evidence v2")
        configuration = host["host_audio_configuration"]
        callbacks = host["callback_frames"]
        if not isinstance(configuration, dict) or not isinstance(callbacks, dict):
            raise ReporterError("host buffer evidence must be represented by objects")
        exact(configuration, {"min_frames_count", "max_frames_count"}, "host audio configuration")
        exact(callbacks, {"calls", "minimum", "maximum"}, "host callback frames")
        activation_minimum = integer(configuration["min_frames_count"], "activation minimum", 1, 1_048_576)
        activation_maximum = integer(configuration["max_frames_count"], "activation maximum", 1, 1_048_576)
        callback_calls = integer(callbacks["calls"], "callback calls", 1)
        observed_minimum = integer(callbacks["minimum"], "observed callback minimum", 1, 1_048_576)
        observed_maximum = integer(callbacks["maximum"], "observed callback maximum", 1, 1_048_576)
        if not activation_minimum <= activation_maximum:
            raise ReporterError("host activation frame bounds are reversed")
        if not activation_minimum <= observed_minimum <= observed_maximum <= activation_maximum:
            raise ReporterError("observed callback frames are outside the host activation bounds")
        effective_buffer = {
            "activation_minimum": activation_minimum,
            "activation_maximum": activation_maximum,
            "callback_calls": callback_calls,
            "observed_minimum": observed_minimum,
            "observed_maximum": observed_maximum,
        }
    else:
        raise ReporterError("unsupported host evidence schema")

    if host.get("source_commit") != submission["source_commit"]:
        raise ReporterError("host evidence source commit differs from the reporter submission")
    if host.get("model_id") != MODEL_ID or host.get("model_sha256") != MODEL_SHA256 or host.get("plugin_id") != PLUGIN_ID:
        raise ReporterError("host evidence identifies the wrong DPDFNet model or plug-in")
    sample_rate = integer(host.get("sample_rate_hz"), "host evidence sample_rate_hz", 8_000, 192_000)
    integer(host.get("channels"), "host evidence channels", 1, 2)
    integer(host.get("chunk_frames"), "host evidence chunk_frames", 1, 1_048_576)
    integer(host.get("latency_frames"), "host evidence latency_frames", 1, 1_048_576)
    processed_frames = integer(host.get("processed_frames"), "host evidence processed_frames", 1)
    active_seconds = host.get("active_seconds")
    if isinstance(active_seconds, bool) or not isinstance(active_seconds, (int, float)) or not 0 < active_seconds <= 1_000_000:
        raise ReporterError("host evidence active_seconds must be a bounded positive number")
    measurement = host.get("measurement")
    if not isinstance(measurement, dict):
        raise ReporterError("host evidence measurement must be an object")
    exact(measurement, {"warmup_frames", "measured_frames"}, "host evidence measurement")
    warmup_frames = integer(measurement["warmup_frames"], "host evidence warmup_frames")
    measured_frames = integer(measurement["measured_frames"], "host evidence measured_frames", 1)
    if warmup_frames > processed_frames or measured_frames != processed_frames - warmup_frames:
        raise ReporterError("host evidence measurement frames are inconsistent")
    worker_started = boolean(host.get("worker_started"), "host evidence worker_started")
    finished_gracefully = boolean(host.get("finished_gracefully"), "host evidence finished_gracefully")
    metrics = validate_metrics(host.get("metrics"), "host evidence metrics")
    validate_metrics(host.get("lifetime_metrics"), "host evidence lifetime metrics")
    environment = host.get("environment")
    if not isinstance(environment, dict):
        raise ReporterError("host evidence environment must be an object")
    exact(environment, {"os", "arch"}, "host evidence environment")
    if environment != {"os": "windows", "arch": "x86_64"}:
        raise ReporterError("reporter host evidence must come from Windows x86-64")

    url = run["host_evidence_url"]
    name = Path(urllib.parse.urlparse(url).path).name
    return {
        "requested_buffer_frames": run["requested_buffer_frames"],
        "sample_rate_hz": sample_rate,
        "duration_seconds": measured_frames / sample_rate,
        "effective_buffer_frames": effective_buffer,
        "worker_started": worker_started,
        "finished_gracefully": finished_gracefully,
        **metrics,
        "audible_xruns": run["audible_xruns"],
        "continuous_audio": run["continuous_audio"],
        "host_evidence": {
            "url": url,
            "name": name,
            "size_bytes": len(payload),
            "sha256": digest,
            "schema": schema,
        },
    }


def gate(identifier: str, passed: bool) -> dict[str, Any]:
    return {
        "id": identifier,
        "observed": int(passed),
        "operator": "greater-or-equal",
        "limit": 1,
        "passed": passed,
    }


def reporter_checks(payload: dict[str, Any], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    version = re.fullmatch(r"([0-9]+)\.([0-9]+)(?:\.[0-9]+)?", payload["environment"]["reaper_version"])
    version_passed = version is not None and tuple(map(int, version.groups()[:2])) >= (7, 79)
    requested = [run["requested_buffer_frames"] for run in runs]
    accessibility = payload["accessibility"]
    checks = [
        gate("reaper-version", version_passed),
        gate(
            "run-duration-rate",
            all(run["sample_rate_hz"] == 48_000 and run["duration_seconds"] >= 300 for run in runs),
        ),
        gate(
            "requested-buffer-coverage",
            len(set(requested)) == len(requested)
            and min(requested) <= 128
            and 480 in requested
            and max(requested) >= 1024,
        ),
        gate("effective-buffer-observed", all(run["effective_buffer_frames"] is not None for run in runs)),
        gate(
            "realtime-worker",
            all(
                run["worker_started"]
                and run["finished_gracefully"]
                and all(run[name] == 0 for name in ("overload_blocks", "late_blocks", "invalid_blocks", "worker_errors"))
                for run in runs
            ),
        ),
        gate(
            "audible-continuity",
            all(run["audible_xruns"] == 0 and run["continuous_audio"] for run in runs),
        ),
        gate(
            "nvda-osara",
            accessibility["nvda_active"]
            and accessibility["osara_active"]
            and accessibility["parameters_announced"] == PARAMETERS
            and accessibility["values_announced"]
            and accessibility["all_adjustable"]
            and accessibility["focus_stable"],
        ),
        gate("no-host-plugin-crashes", accessibility["host_or_plugin_crashes"] == 0),
    ]
    return checks


def build_v2_document(
    payload: dict[str, Any],
    github: dict[str, Any],
    loader: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    validate_payload_v2(payload)
    if loader is None:
        loader = attachment
    urls: set[str] = set()
    runs = []
    for run in payload["runs"]:
        url = run["host_evidence_url"]
        if url in urls:
            raise ReporterError("each buffer run must use a distinct host evidence attachment")
        urls.add(url)
        runs.append(normalize_host_evidence(payload, run, loader(url)))
    checks = reporter_checks(payload, runs)
    return {
        "schema": "denoize-dpdfnet-reporter-evidence-v2",
        "schema_version": 2,
        "github": github,
        "payload": payload,
        "runs": runs,
        "checks": checks,
        "accepted": all(check["passed"] for check in checks),
    }


def generate(args: argparse.Namespace) -> bool:
    match = COMMENT_RE.fullmatch(args.comment_url)
    if not match:
        raise ReporterError("comment URL must refer to issue #221 in penguin425/denoize")
    comment_id = int(match.group(1))
    issue = api("/repos/penguin425/denoize/issues/221")
    comment = api(f"/repos/penguin425/denoize/issues/comments/{comment_id}")
    issue_login = issue.get("user", {}).get("login")
    comment_login = comment.get("user", {}).get("login")
    if issue_login != "UlisesMilani" or comment_login != issue_login:
        raise ReporterError("the evidence comment was not authored by the issue #221 reporter")
    if comment.get("html_url") != args.comment_url:
        raise ReporterError("GitHub API comment URL differs from the requested URL")
    body = comment.get("body")
    if not isinstance(body, str) or len(body.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ReporterError("comment body is missing or too large")
    matches = re.findall(r"```json\s*\n(\{.*?\})\s*\n```", body, flags=re.DOTALL)
    if len(matches) != 1:
        raise ReporterError("comment must contain exactly one fenced JSON evidence object")
    payload = parse_json(matches[0], "comment evidence")
    github = {
        "repository": "penguin425/denoize",
        "issue": 221,
        "comment_id": comment_id,
        "comment_url": args.comment_url,
        "author": comment_login,
        "created_at": comment.get("created_at"),
        "updated_at": comment.get("updated_at"),
        "comment_body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }
    if payload.get("schema") == "denoize-dpdfnet-reporter-submission-v1":
        validate_payload_v1(payload)
        document = {
            "schema": "denoize-dpdfnet-reporter-evidence-v1",
            "schema_version": 1,
            "github": github,
            "payload": payload,
            "accepted": True,
        }
    elif payload.get("schema") == "denoize-dpdfnet-reporter-submission-v2":
        document = build_v2_document(payload, github)
    else:
        raise ReporterError("unsupported reporter-submission schema")
    output = args.output
    if output.exists() or output.is_symlink():
        raise ReporterError(f"refusing to replace existing reporter evidence: {output}")
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
    return document["accepted"]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--comment-url", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--allow-rejected", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        accepted = generate(args)
    except (ReporterError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not accepted and not args.allow_rejected:
        print("error: reporter evidence did not pass every promotion gate", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
