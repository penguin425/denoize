#!/usr/bin/env python3
"""Bind a DPDFNet DAW stress run and paced CLAP worker run to one platform."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


MODEL_SHA256 = "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_PEAK_RSS_BYTES = 512 * 1024 * 1024
WORKER_WALL_LOWER_RATIO = 0.95
WORKER_WALL_UPPER_RATIO = 1.05
WORKER_WALL_UPPER_SLACK_SECONDS = 0.25


class EvidenceError(RuntimeError):
    pass


def load(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    original = path
    if original.is_symlink() or not original.is_file():
        raise EvidenceError(f"{label} is not a regular file: {original}")
    path = original.resolve()
    size = path.stat().st_size
    if not 1 <= size <= MAX_JSON_BYTES:
        raise EvidenceError(f"{label} size must be in 1..={MAX_JSON_BYTES}")
    payload = path.read_bytes()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"invalid {label} JSON: {error}") from error
    if not isinstance(document, dict):
        raise EvidenceError(f"{label} must be a JSON object")
    return document, payload


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} must be finite")
    return result


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{label} must be an integer")
    return value


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path, payload: bytes) -> dict[str, Any]:
    return {"name": path.name, "size_bytes": len(payload), "sha256": sha256(payload)}


def check(identifier: str, observed: int | float, operator: str, limit: int | float) -> dict[str, Any]:
    passed = observed <= limit if operator == "less-or-equal" else observed >= limit
    return {"id": identifier, "observed": observed, "operator": operator, "limit": limit, "passed": passed}


def generate(args: argparse.Namespace) -> bool:
    stress_path = args.stress.resolve()
    worker_path = args.worker.resolve()
    stress, stress_payload = load(args.stress, "stress run")
    worker, worker_payload = load(args.worker, "worker run")
    if stress.get("schema") != "denoize-dpdfnet-gtcrn-stress-v1":
        raise EvidenceError("unsupported stress-run schema")
    if stress.get("model") != "dpdfnet2_48khz_stereo_independent_daw_path":
        raise EvidenceError("stress run did not exercise the production DPDFNet DAW path")
    if stress.get("model_file_sha256") != MODEL_SHA256 or stress.get("state_size") != 56_436:
        raise EvidenceError("stress run did not bind the pinned DPDFNet-2 model")
    if stress.get("parallel_streams") != 1:
        raise EvidenceError("promotion stress run must measure one DAW stream")
    realtime_paced = stress.get("realtime_paced")
    if not isinstance(realtime_paced, bool):
        raise EvidenceError("stress run must declare whether it was real-time paced")
    environment = stress.get("environment")
    if not isinstance(environment, dict):
        raise EvidenceError("stress run lacks an environment object")
    source_commit = environment.get("source_commit")
    if not isinstance(source_commit, str) or not COMMIT_RE.fullmatch(source_commit):
        raise EvidenceError("stress run lacks a lowercase 40-character source commit")
    operating_system = environment.get("os")
    if operating_system not in {"linux", "macos", "windows"}:
        raise EvidenceError("stress run operating system is unsupported")
    target = environment.get("target")
    expected_target_fragment = {"linux": "linux", "macos": "apple", "windows": "windows"}[operating_system]
    if not isinstance(target, str) or expected_target_fragment not in target:
        raise EvidenceError("stress target does not match its operating system")
    os_version = environment.get("os_version")
    cpu_model = environment.get("cpu_model")
    hardware_tier = environment.get("hardware_tier")
    runner_label = environment.get("runner_label")
    if not isinstance(os_version, str) or not 1 <= len(os_version) <= 256:
        raise EvidenceError("stress run lacks a bounded OS version")
    if not isinstance(cpu_model, str) or not 1 <= len(cpu_model) <= 256:
        raise EvidenceError("stress run lacks a bounded CPU model")
    if hardware_tier not in {"portable-ci", "lowest-supported"}:
        raise EvidenceError("hardware tier must be portable-ci or lowest-supported")
    logical_cpus = integer(environment.get("logical_parallelism"), "logical_parallelism")
    if not 1 <= logical_cpus <= 1024:
        raise EvidenceError("logical_parallelism is outside 1..=1024")
    if hardware_tier == "portable-ci":
        expected_runner = {
            "linux": "ubuntu-24.04",
            "macos": "macos-15",
            "windows": "windows-2025",
        }[operating_system]
        if runner_label != expected_runner:
            raise EvidenceError(
                f"portable {operating_system} evidence must run on {expected_runner}"
            )
    elif (
        operating_system != "linux"
        or target != "x86_64-unknown-linux-gnu"
        or runner_label != "ubuntu-slim"
        or logical_cpus != 1
    ):
        raise EvidenceError(
            "lowest-supported evidence must run on one logical CPU under "
            "x86_64 ubuntu-slim"
        )
    if not realtime_paced:
        raise EvidenceError("promotion stress evidence must be real-time paced")

    timing = stress.get("timing")
    memory = stress.get("memory")
    robustness = stress.get("robustness")
    if not isinstance(timing, dict) or not isinstance(memory, dict) or not isinstance(robustness, dict):
        raise EvidenceError("stress run lacks timing, memory, or robustness evidence")
    seconds = integer(stress.get("requested_seconds_per_stream"), "requested_seconds_per_stream")
    calls = integer(stress.get("calls"), "calls")
    measured_audio_seconds = number(
        stress.get("measured_audio_seconds"), "measured_audio_seconds"
    )
    if measured_audio_seconds <= 0.0:
        raise EvidenceError("stress run measured no audio")
    wall_p99_9_ms = number(timing.get("p99_9_ms"), "p99_9_ms")
    wall_maximum_ms = number(timing.get("maximum_ms"), "maximum_ms")
    wall_summed_rtf = number(timing.get("summed_compute_rtf"), "summed_compute_rtf")
    wall_calls_over_budget = integer(timing.get("calls_over_budget"), "calls_over_budget")
    peak_rss = integer(memory.get("peak_rss_bytes"), "peak_rss_bytes")
    if timing.get("budget_ms") != 10.0:
        raise EvidenceError("stress run must use a 10 ms DAW deadline")
    process_cpu_config = {
        "macos": ("CLOCK_PROCESS_CPUTIME_ID", "macOS"),
        "windows": ("GetProcessTimes", "Windows"),
    }.get(operating_system)
    process_cpu_summed_rtf = None
    if process_cpu_config is not None:
        process_cpu_clock, process_cpu_label = process_cpu_config
        process_cpu = timing.get("process_cpu")
        if not isinstance(process_cpu, dict):
            raise EvidenceError(f"{process_cpu_label} stress run lacks process CPU timing")
        if process_cpu.get("clock") != process_cpu_clock:
            raise EvidenceError(
                f"{process_cpu_label} stress run used an unsupported process CPU clock"
            )
        if integer(process_cpu.get("sample_count"), "process CPU sample_count") != calls:
            raise EvidenceError("process CPU timing sample count does not match stress calls")
        if process_cpu.get("budget_ms") != 10.0:
            raise EvidenceError("process CPU timing must use a 10 ms DAW deadline")
        expected_distribution_eligibility = operating_system != "windows"
        if (
            process_cpu.get("per_call_distribution_gate_eligible")
            is not expected_distribution_eligibility
        ):
            raise EvidenceError(
                f"{process_cpu_label} process CPU distribution eligibility is invalid"
            )
        minimum_nonzero_delta_ms = number(
            process_cpu.get("minimum_nonzero_delta_ms"),
            "process CPU minimum_nonzero_delta_ms",
        )
        if minimum_nonzero_delta_ms <= 0.0:
            raise EvidenceError("process CPU timing has no positive clock delta")
        total_run_cpu_ms = number(
            process_cpu.get("total_run_cpu_ms"), "process CPU total_run_cpu_ms"
        )
        if total_run_cpu_ms <= 0.0:
            raise EvidenceError("process CPU timing has no positive run total")
        process_cpu_summed_rtf = number(
            process_cpu.get("summed_compute_rtf"),
            "process CPU summed_compute_rtf",
        )
        expected_process_cpu_rtf = total_run_cpu_ms / 1_000.0 / measured_audio_seconds
        if not math.isclose(
            process_cpu_summed_rtf,
            expected_process_cpu_rtf,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise EvidenceError("process CPU run total and summed RTF are inconsistent")
    if operating_system == "macos":
        deadline_clock = "process-cpu"
        compute_rtf_clock = "process-cpu"
        p99_9_ms = number(process_cpu.get("p99_9_ms"), "process CPU p99_9_ms")
        maximum_ms = number(process_cpu.get("maximum_ms"), "process CPU maximum_ms")
        summed_rtf = process_cpu_summed_rtf
        calls_over_budget = integer(
            process_cpu.get("calls_over_budget"),
            "process CPU calls_over_budget",
        )
    elif operating_system == "windows":
        # GetProcessTimes is accurate when accumulated across the complete run,
        # but its per-call deltas are quantized on hosted Windows runners. Keep
        # wall time for the direct-call distribution and use process CPU time
        # only for the aggregate compute-capacity check.
        deadline_clock = "monotonic-wall"
        compute_rtf_clock = "process-cpu"
        p99_9_ms = wall_p99_9_ms
        maximum_ms = wall_maximum_ms
        summed_rtf = process_cpu_summed_rtf
        calls_over_budget = wall_calls_over_budget
    else:
        deadline_clock = "monotonic-wall"
        compute_rtf_clock = "monotonic-wall"
        p99_9_ms = wall_p99_9_ms
        maximum_ms = wall_maximum_ms
        summed_rtf = wall_summed_rtf
        calls_over_budget = wall_calls_over_budget
    if robustness.get("independent_stream_bit_exact") is not True or robustness.get("empty_input_exact") is not True:
        raise EvidenceError("stress reset/empty-input robustness did not pass")
    finite_geometry = robustness.get("finite_geometry")
    if not isinstance(finite_geometry, list) or {item.get("sample_rate") for item in finite_geometry if isinstance(item, dict)} != {8000, 16000, 44100, 48000, 96000}:
        raise EvidenceError("stress run lacks all five sample-rate probes")
    if any(item.get("exact_length") is not True or item.get("all_finite") is not True for item in finite_geometry):
        raise EvidenceError("one or more sample-rate robustness probes failed")

    if worker.get("schema") != "denoize-dpdfnet-worker-run-v2" or worker.get("schema_version") != 2:
        raise EvidenceError("unsupported paced-worker schema")
    if worker.get("source_commit") != source_commit:
        raise EvidenceError("stress and worker evidence bind different commits")
    if (
        worker.get("model_id") != "dpdfnet2-48khz-hr"
        or worker.get("model_sha256") != MODEL_SHA256
        or worker.get("plugin_id") != "org.penguin425.denoize.neural-hq"
    ):
        raise EvidenceError("worker run did not bind the pinned DPDFNet model")
    for field, expected in (("sample_rate_hz", 48_000), ("channels", 2), ("chunk_frames", 480), ("latency_frames", 11_520)):
        if isinstance(worker.get(field), bool) or not isinstance(worker.get(field), int) or worker.get(field) != expected:
            raise EvidenceError(f"worker {field} must be integer {expected}")
    if worker.get("channel_mode") != "independent":
        raise EvidenceError("worker run did not exercise the 48 kHz independent stereo gate geometry")
    worker_environment = worker.get("environment")
    if not isinstance(worker_environment, dict) or worker_environment.get("os") != operating_system:
        raise EvidenceError("stress and worker evidence bind different operating systems")
    worker_binding = {
        "arch": environment.get("arch"),
        "logical_parallelism": logical_cpus,
        "target": target,
        "cpu_model": cpu_model,
        "hardware_tier": hardware_tier,
        "runner_label": runner_label,
    }
    for name, expected in worker_binding.items():
        if worker_environment.get(name) != expected:
            raise EvidenceError(
                f"stress and worker evidence bind different {name} values"
            )
    runs = worker.get("runs")
    if not isinstance(runs, list) or len(runs) != 3 or sorted(run.get("callback_frames") for run in runs if isinstance(run, dict)) != [144, 480, 1024]:
        raise EvidenceError("worker run must contain exactly one 144, 480, and 1024-frame run")
    aggregate_neural = [0, 0]
    aggregate_scheduling = aggregate_processing = aggregate_total = 0
    paced_blocks = None
    for run in runs:
        callback_frames = integer(run.get("callback_frames"), "worker callback_frames")
        callback_calls = integer(run.get("callback_calls"), "worker callback_calls")
        paced = integer(run.get("paced_blocks"), "worker paced_blocks")
        measured = integer(run.get("measured_frames"), "worker measured_frames")
        finite_count = integer(run.get("finite_frames"), "worker finite_frames")
        neural = run.get("neural_frames_per_channel")
        if not isinstance(neural, list) or len(neural) != 2:
            raise EvidenceError("worker run lacks two per-channel neural frame counts")
        neural = [integer(value, "worker neural_frames_per_channel") for value in neural]
        if measured != callback_calls * callback_frames or measured != math.ceil((11_520 + paced * 480) / callback_frames) * callback_frames:
            raise EvidenceError("paced worker frame accounting is inconsistent")
        if finite_count != measured or any(value < 480 or value > measured - 11_520 for value in neural):
            raise EvidenceError("worker finite or per-channel neural frame gate failed")
        expected_wall = measured / 48_000
        actual_wall = number(run.get("measurement_wall_seconds"), "worker measurement_wall_seconds")
        if actual_wall < expected_wall * WORKER_WALL_LOWER_RATIO or actual_wall > expected_wall * WORKER_WALL_UPPER_RATIO + WORKER_WALL_UPPER_SLACK_SECONDS:
            raise EvidenceError("paced worker wall time does not represent its real-time schedule")
        metrics = run.get("metrics")
        if not isinstance(metrics, dict):
            raise EvidenceError("worker run lacks metrics")
        values = {name: integer(metrics.get(name), f"worker {name}") for name in ("overload_blocks", "late_blocks", "invalid_blocks", "worker_errors")}
        if any(value < 0 or value > 9_007_199_254_740_991 for value in values.values()):
            raise EvidenceError("worker metric counters must be non-negative safe integers")
        queues = run.get("queues_after_run")
        if not isinstance(queues, dict):
            raise EvidenceError("worker run lacks queues_after_run")
        for queue_name in ("input", "output", "ready"):
            queue_value = queues.get(queue_name)
            if isinstance(queue_value, bool) or not isinstance(queue_value, int) or not 0 <= queue_value <= 56:
                raise EvidenceError(f"worker queue {queue_name} must be an integer in 0..56")
        scheduling = values["overload_blocks"] + values["late_blocks"]
        processing = values["invalid_blocks"] + values["worker_errors"]
        aggregate_scheduling += scheduling; aggregate_processing += processing; aggregate_total += scheduling + processing
        paced_blocks = paced if paced_blocks is None else min(paced_blocks, paced)
        # The aggregate gate reports the weakest channel across geometries.
        aggregate_neural = [min(aggregate_neural[index], neural[index]) for index in range(2)] if any(aggregate_neural) else neural
    # The released CLAP path exposes a 24-chunk/240 ms buffered worker
    # contract, not each synchronous 10 ms model invocation as a host
    # deadline. Retain the direct-call distribution as capacity diagnostics,
    # while gating the production contract with aggregate compute and the
    # independently paced worker below.
    direct_call_deadline_gate_eligible = False
    wall_clock_worker_gate_eligible = hardware_tier == "portable-ci"
    checks = [
        check("minimum-stress-seconds", seconds, "greater-or-equal", 60),
        check("minimum-stress-calls", calls, "greater-or-equal", 6000),
        check("stress-summed-rtf", summed_rtf, "less-or-equal", 1.0),
        check("stress-peak-rss-bytes", peak_rss, "less-or-equal", MAX_PEAK_RSS_BYTES),
        check(
            "minimum-paced-worker-blocks",
            paced_blocks,
            "greater-or-equal",
            seconds * 100,
        ),
        check(
            (
                "worker-error-counters"
                if wall_clock_worker_gate_eligible
                else "worker-processing-errors"
            ),
            aggregate_total if wall_clock_worker_gate_eligible else aggregate_processing,
            "less-or-equal",
            0,
        ),
        check("worker-stereo-finite-frames", 1, "greater-or-equal", 1),
        check("worker-independent-stereo", 1, "greater-or-equal", 1),
        check("worker-neural-frames-per-channel", min(aggregate_neural), "greater-or-equal", 480),
    ]
    accepted = all(item["passed"] for item in checks)
    document = {
        "schema": "denoize-dpdfnet-platform-evidence-v3",
        "schema_version": 3,
        "source_commit": source_commit,
        "model_id": "dpdfnet2-48khz-hr",
        "model_sha256": MODEL_SHA256,
        "platform": {
            "os": operating_system,
            "os_version": os_version,
            "target": target,
            "arch": environment.get("arch"),
            "cpu_model": cpu_model,
            "logical_cpus": logical_cpus,
            "hardware_tier": hardware_tier,
            "runner_label": runner_label,
        },
        "inputs": {
            "stress": file_record(stress_path, stress_payload),
            "paced_worker": file_record(worker_path, worker_payload),
        },
        "measurement": {
            "stress_seconds": seconds,
            "stress_calls": calls,
            "stress_realtime_paced": realtime_paced,
            "deadline_clock": deadline_clock,
            "compute_rtf_clock": compute_rtf_clock,
            "direct_call_deadline_gate_eligible": direct_call_deadline_gate_eligible,
            "wall_clock_worker_gate_eligible": wall_clock_worker_gate_eligible,
            "p99_9_ms": p99_9_ms,
            "maximum_ms": maximum_ms,
            "deadline_misses": calls_over_budget,
            "summed_compute_rtf": summed_rtf,
            "wall_p99_9_ms": wall_p99_9_ms,
            "wall_maximum_ms": wall_maximum_ms,
            "wall_deadline_misses": wall_calls_over_budget,
            "wall_summed_compute_rtf": wall_summed_rtf,
            "peak_rss_bytes": peak_rss,
            "worker_channels": 2,
            "worker_channel_mode": "independent",
            "worker_callback_frames": [144, 480, 1024],
            "worker_neural_frames_per_channel": aggregate_neural,
            "paced_worker_blocks": paced_blocks,
            "worker_error_counter_total": aggregate_total,
            "worker_scheduling_counter_total": aggregate_scheduling,
            "worker_processing_error_count": aggregate_processing,
        },
        "checks": checks,
        "accepted": accepted,
    }
    payload = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    output = args.output
    if output.exists() or output.is_symlink():
        raise EvidenceError(f"refusing to replace existing platform evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as destination:
            descriptor = -1
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not accepted:
        failed = "; ".join(
            f"{item['id']}: observed={item['observed']!r} "
            f"{item['operator']} limit={item['limit']!r}"
            for item in checks
            if not item["passed"]
        )
        print(f"failed DPDFNet platform checks: {failed}", file=sys.stderr)
    return accepted


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--stress", type=Path, required=True)
    result.add_argument("--worker", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--allow-rejected", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        accepted = generate(args)
    except (EvidenceError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not accepted and not args.allow_rejected:
        print("error: DPDFNet platform promotion thresholds were not met", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
