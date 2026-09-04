#!/usr/bin/env python3
"""Exercise the DPDFNet blind-listening and promotion evidence contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import wave

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent.parent
PREPARE = ROOT / "scripts/prepare-dpdfnet-blind-listening.py"
SCORE = ROOT / "scripts/score-dpdfnet-blind-listening.py"
PLATFORM = ROOT / "scripts/generate-dpdfnet-platform-evidence.py"
PROMOTION = ROOT / "scripts/generate-dpdfnet-promotion-evidence.py"
FETCH_REPORTER = ROOT / "scripts/fetch-dpdfnet-reporter-evidence.py"
SCHEMAS = {
    "protocol": ROOT / "schemas/denoize-dpdfnet-blind-protocol-v1.schema.json",
    "answer": ROOT / "schemas/denoize-dpdfnet-blind-answer-key-v1.schema.json",
    "response": ROOT / "schemas/denoize-dpdfnet-blind-listener-response-v1.schema.json",
    "result": ROOT / "schemas/denoize-dpdfnet-blind-listening-result-v1.schema.json",
    "worker": ROOT / "schemas/denoize-dpdfnet-worker-run-v1.schema.json",
    "clap_host": ROOT / "schemas/denoize-dpdfnet-clap-host-run-v1.schema.json",
    "clap_host_v2": ROOT / "schemas/denoize-dpdfnet-clap-host-run-v2.schema.json",
    "platform_v1": ROOT / "schemas/denoize-dpdfnet-platform-evidence-v1.schema.json",
    "platform": ROOT / "schemas/denoize-dpdfnet-platform-evidence-v2.schema.json",
    "reaper": ROOT / "schemas/denoize-dpdfnet-reaper-automated-evidence-v1.schema.json",
    "reporter_v1": ROOT / "schemas/denoize-dpdfnet-reporter-evidence-v1.schema.json",
    "reporter_v2": ROOT / "schemas/denoize-dpdfnet-reporter-evidence-v2.schema.json",
    "promotion": ROOT / "schemas/denoize-dpdfnet-promotion-evidence-v1.schema.json",
}


def run(arguments: list[str], *, success: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(arguments, check=False, capture_output=True, text=True)
    if success and result.returncode != 0:
        raise AssertionError(f"command failed ({result.returncode}): {' '.join(arguments)}\n{result.stderr}")
    if not success and result.returncode == 0:
        raise AssertionError(f"command unexpectedly passed: {' '.join(arguments)}")
    return result


def write_wav(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(48_000)
        samples = [((index * 997 + seed * 101) % 20_000) - 10_000 for index in range(480)]
        output.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def fixture(root: Path) -> tuple[Path, Path]:
    matrix = root / "matrix-result.json"
    audio = root / "candidates"
    definitions = [
        (f"recorded-{index}", "noise-matrix", f"freesound-{2530 + index}")
        for index in range(4)
    ]
    definitions.extend(
        (f"babble-{index}", "noise-matrix", "three-talker-babble") for index in range(3)
    )
    definitions.extend((f"clean-{index}", "clean-preservation", None) for index in range(3))
    definitions.extend((f"synthetic-{index}", "noise-matrix", "pink") for index in range(2))
    cases = []
    for case_index, (identifier, kind, noise) in enumerate(definitions):
        cases.append({"id": identifier, "kind": kind, "noise": noise})
        for file_index, name in enumerate(("clean", "noisy", "dpdfnet2", "gtcrn")):
            write_wav(audio / identifier / f"{name}.wav", case_index * 10 + file_index)
    matrix.write_text(json.dumps({"cases": cases}, sort_keys=True) + "\n", encoding="utf-8")
    return matrix, audio


def prepare(root: Path, matrix: Path, audio: Path) -> tuple[Path, Path]:
    key = root / "randomization.key"
    key.write_bytes(bytes(range(32)))
    bundle = root / "public-bundle"
    answer = root / "private-answer-key.json"
    run(
        [
            sys.executable,
            str(PREPARE),
            "--matrix-result",
            str(matrix),
            "--audio-dir",
            str(audio),
            "--randomization-key",
            str(key),
            "--output-dir",
            str(bundle),
            "--answer-key",
            str(answer),
        ]
    )
    return bundle, answer


def responses(root: Path, protocol: dict, answer: dict, *, candidate: bool) -> Path:
    output = root / ("candidate-responses" if candidate else "baseline-responses")
    output.mkdir()
    answer_by_id = {trial["trial_id"]: trial for trial in answer["trials"]}
    target = "dpdfnet2-48khz-hr" if candidate else "gtcrn-dns3"
    for listener_index in range(20):
        trials = []
        for trial in protocol["trials"]:
            key = answer_by_id[trial["trial_id"]]
            preference = "a" if key["a_model"] == target else "b"
            trials.append({"trial_id": trial["trial_id"], "preference": preference})
        document = {
            "schema": "denoize-dpdfnet-blind-listener-response-v1",
            "schema_version": 1,
            "protocol_sha256": answer["protocol_sha256"],
            "listener_id": f"listener-{listener_index:02d}",
            "consent": True,
            "trials": trials,
        }
        (output / f"listener-{listener_index:02d}.json").write_text(
            json.dumps(document, sort_keys=True) + "\n", encoding="utf-8"
        )
    return output


def platform_fixture(root: Path) -> tuple[Path, Path]:
    commit = "0123456789abcdef0123456789abcdef01234567"
    stress = {
        "schema": "denoize-dpdfnet-gtcrn-stress-v1",
        "model": "dpdfnet2_48khz_stereo_linked_daw_path",
        "model_file_sha256": "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b",
        "state_size": 56_436,
        "parallel_streams": 1,
        "requested_seconds_per_stream": 60,
        "realtime_paced": True,
        "calls": 6_000,
        "timing": {
            "p99_9_ms": 8.0,
            "maximum_ms": 9.0,
            "budget_ms": 10.0,
            "calls_over_budget": 0,
            "summed_compute_rtf": 0.5,
        },
        "memory": {"peak_rss_bytes": 128 * 1024 * 1024},
        "robustness": {
            "independent_stream_bit_exact": True,
            "empty_input_exact": True,
            "finite_geometry": [
                {"sample_rate": rate, "exact_length": True, "all_finite": True}
                for rate in (8_000, 16_000, 44_100, 48_000, 96_000)
            ],
        },
        "environment": {
            "source_commit": commit,
            "os": "linux",
            "target": "x86_64-unknown-linux-gnu",
            "os_version": "fixture-linux",
            "cpu_model": "fixture-cpu",
            "hardware_tier": "portable-ci",
            "runner_label": "ubuntu-24.04",
            "logical_parallelism": 2,
            "arch": "x86_64",
        },
    }
    worker = {
        "schema": "denoize-dpdfnet-worker-run-v1",
        "schema_version": 1,
        "source_commit": commit,
        "model_id": "dpdfnet2-48khz-hr",
        "model_sha256": "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b",
        "plugin_id": "org.penguin425.denoize.neural-hq",
        "sample_rate_hz": 48_000,
        "channels": 1,
        "chunk_frames": 480,
        "latency_frames": 11_520,
        "paced_blocks": 6_000,
        "measured_frames": 2_891_520,
        "finite_frames": 2_891_520,
        "neural_frames": 2_880_000,
        "measurement_wall_seconds": 60.25,
        "metrics": {
            "overload_blocks": 0,
            "late_blocks": 0,
            "invalid_blocks": 0,
            "worker_errors": 0,
        },
        "queues_after_run": {"input": 0, "output": 0, "ready": 0},
        "environment": {
            "os": "linux",
            "arch": "x86_64",
            "logical_parallelism": 2,
            "target": "x86_64-unknown-linux-gnu",
            "cpu_model": "fixture-cpu",
            "hardware_tier": "portable-ci",
            "runner_label": "ubuntu-24.04",
        },
    }
    stress_path = root / "stress.json"
    worker_path = root / "worker.json"
    stress_path.write_text(json.dumps(stress) + "\n", encoding="utf-8")
    worker_path.write_text(json.dumps(worker) + "\n", encoding="utf-8")
    return stress_path, worker_path


def load_promotion_module():
    specification = importlib.util.spec_from_file_location(
        "denoize_dpdfnet_promotion", PROMOTION
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_reporter_module():
    specification = importlib.util.spec_from_file_location(
        "denoize_dpdfnet_reporter", FETCH_REPORTER
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def file_record(name: str) -> dict[str, object]:
    return {"name": name, "size_bytes": 1, "sha256": "0" * 64}


def composite_fixtures(
    root: Path, listening: dict, platform: dict, validators: dict
) -> tuple[SimpleNamespace, object]:
    module = load_promotion_module()
    commit = "0123456789abcdef0123456789abcdef01234567"
    summary = {
        "schema": "denoize-dpdfnet-gtcrn-evaluation-summary-v1",
        "source_commit": commit,
        "matrix_result_sha256": listening["source_matrix_sha256"],
        "models": {
            "dpdfnet2-48khz-hr": module.MODEL_SHA256,
            "gtcrn-dns3": module.GTCRN_SHA256,
        },
        "quality": {
            "case_counts": {
                "noise_matrix": 280,
                "clean_preservation": 10,
                "speakers": 10,
                "noises": 7,
            },
            "dpdfnet2_vs_gtcrn": {
                "si_sdr_improvement_db": {
                    "speaker_cluster_bootstrap_95ci": [0.1, 0.3]
                },
                "stoi_improvement": {
                    "speaker_cluster_bootstrap_95ci": [0.01, 0.03]
                },
                "musical_noise": {
                    "speaker_cluster_bootstrap_95ci": [0.01, 0.03]
                },
            },
            "strata": {
                "noise": {
                    "three-talker-babble": {
                        "speaker_cluster_bootstrap_95ci": [0.01, 0.03]
                    }
                }
            },
        },
    }
    summary_path = root / "evaluation-summary.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    listening_path = root / "listening-result.json"
    listening_path.write_text(json.dumps(listening) + "\n", encoding="utf-8")

    reaper = {
        "schema": "denoize-dpdfnet-reaper-automated-evidence-v1",
        "schema_version": 1,
        "source_commit": commit,
        "model_id": "dpdfnet2-48khz-hr",
        "model_sha256": module.MODEL_SHA256,
        "host": {
            "name": "REAPER",
            "version": "7.79",
            "operating_system": "windows",
            "sample_rate_hz": 48_000,
            "buffer_frames": 480,
            "active_seconds": 60.0,
        },
        "accessibility_api": {
            "osara_style_parameter_path": True,
            "parameters_readable_and_adjustable": 4,
            "nvda_human_verified": False,
        },
        "measurement": {
            "warmup_frames": 46_080,
            "measured_frames": 2_880_000,
        },
        "worker_metrics": {
            "overload_blocks": 0,
            "late_blocks": 0,
            "invalid_blocks": 0,
            "worker_errors": 0,
        },
        "lifetime_worker_metrics": {
            "overload_blocks": 24,
            "late_blocks": 6,
            "invalid_blocks": 5,
            "worker_errors": 0,
        },
        "process": {
            "wall_seconds": 60.0,
            "cpu_seconds": 10.0,
            "peak_working_set_bytes": 128 * 1024 * 1024,
            "logical_processors": 2,
        },
        "inputs": {
            "parameters": file_record("parameters.tsv"),
            "clap_host_run": file_record("clap-host-run.json"),
            "process_metrics": file_record("process.json"),
        },
        "accepted_automated": True,
    }
    validators["reaper"].validate(reaper)
    reaper_path = root / "reaper-automated.json"
    reaper_path.write_text(json.dumps(reaper) + "\n", encoding="utf-8")

    artifact = root / "experimental.zip"
    artifact.write_bytes(b"attested experimental archive fixture")
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    reporter_module = load_reporter_module()
    host_payloads = {}
    submission_runs = []
    for index, frames in enumerate((128, 480, 1024)):
        url = f"https://github.com/user-attachments/files/{12345678 + index}/dpdfnet-{frames}.json"
        host = {
            "schema": "denoize-dpdfnet-clap-host-run-v2",
            "schema_version": 2,
            "source_commit": commit,
            "model_id": "dpdfnet2-48khz-hr",
            "model_sha256": module.MODEL_SHA256,
            "plugin_id": "org.penguin425.denoize.neural-hq",
            "sample_rate_hz": 48_000,
            "channels": 1,
            "chunk_frames": 480,
            "latency_frames": 11_520,
            "processed_frames": 14_446_080,
            "active_seconds": 301.0,
            "measurement": {
                "warmup_frames": 46_080,
                "measured_frames": 14_400_000,
            },
            "host_audio_configuration": {
                "min_frames_count": frames,
                "max_frames_count": frames,
            },
            "callback_frames": {
                "calls": 100_000,
                "minimum": frames,
                "maximum": frames,
            },
            "worker_started": True,
            "finished_gracefully": True,
            "metrics": {
                "overload_blocks": 0,
                "late_blocks": 0,
                "invalid_blocks": 0,
                "worker_errors": 0,
            },
            "lifetime_metrics": {
                "overload_blocks": 0,
                "late_blocks": 0,
                "invalid_blocks": 0,
                "worker_errors": 0,
            },
            "environment": {"os": "windows", "arch": "x86_64"},
        }
        validators["clap_host_v2"].validate(host)
        encoded = (json.dumps(host, sort_keys=True) + "\n").encode()
        host_payloads[url] = encoded
        submission_runs.append(
            {
                "requested_buffer_frames": frames,
                "host_evidence_url": url,
                "host_evidence_sha256": hashlib.sha256(encoded).hexdigest(),
                "audible_xruns": 0,
                "continuous_audio": True,
            }
        )
    reporter_payload = {
        "schema": "denoize-dpdfnet-reporter-submission-v2",
        "schema_version": 2,
        "source_commit": commit,
        "artifact_sha256": artifact_sha256,
        "environment": {
            "windows_version": "fixture",
            "cpu_model": "fixture",
            "audio_device": "fixture",
            "audio_driver": "fixture",
            "reaper_version": "7.79",
            "nvda_version": "fixture",
            "osara_version": "fixture",
        },
        "runs": submission_runs,
        "accessibility": {
            "nvda_active": True,
            "osara_active": True,
            "parameters_announced": [
                "Bypass",
                "Mix",
                "Output Gain",
                "Overload Fallback",
            ],
            "values_announced": True,
            "all_adjustable": True,
            "focus_stable": True,
            "host_or_plugin_crashes": 0,
        },
        "quality_observation": "dpdfnet-better",
        "consent_to_publish": True,
    }
    reporter = reporter_module.build_v2_document(
        reporter_payload,
        {
            "repository": "penguin425/denoize",
            "issue": 221,
            "comment_id": 1,
            "comment_url": "https://github.com/penguin425/denoize/issues/221#issuecomment-1",
            "author": "UlisesMilani",
            "created_at": "2026-09-02T00:00:00Z",
            "updated_at": "2026-09-02T00:00:00Z",
            "comment_body_sha256": "1" * 64,
        },
        loader=lambda url: host_payloads[url],
    )
    validators["reporter_v2"].validate(reporter)
    assert reporter["accepted"] is True
    inconsistent_reporter = json.loads(json.dumps(reporter))
    inconsistent_reporter["accepted"] = False
    try:
        module.reporter_passed(inconsistent_reporter)
    except module.PromotionError as error:
        assert "accepted flag differs" in str(error)
    else:
        raise AssertionError("inconsistent reporter result unexpectedly passed")

    failed_payload = json.loads(json.dumps(reporter_payload))
    failed_hosts = dict(host_payloads)
    failed_url = failed_payload["runs"][0]["host_evidence_url"]
    failed_host = json.loads(failed_hosts[failed_url])
    failed_host["metrics"]["overload_blocks"] = 7
    failed_encoded = (json.dumps(failed_host, sort_keys=True) + "\n").encode()
    failed_hosts[failed_url] = failed_encoded
    failed_payload["runs"][0]["host_evidence_sha256"] = hashlib.sha256(
        failed_encoded
    ).hexdigest()
    failed_reporter = reporter_module.build_v2_document(
        failed_payload,
        reporter["github"],
        loader=lambda url: failed_hosts[url],
    )
    validators["reporter_v2"].validate(failed_reporter)
    assert failed_reporter["accepted"] is False
    assert failed_reporter["runs"][0]["overload_blocks"] == 7
    assert next(
        check for check in failed_reporter["checks"] if check["id"] == "realtime-worker"
    )["passed"] is False

    unknown_xruns_payload = json.loads(json.dumps(reporter_payload))
    unknown_xruns_payload["runs"][0]["audible_xruns"] = None
    unknown_xruns_payload["runs"][0]["continuous_audio"] = False
    unknown_xruns_reporter = reporter_module.build_v2_document(
        unknown_xruns_payload,
        reporter["github"],
        loader=lambda url: host_payloads[url],
    )
    validators["reporter_v2"].validate(unknown_xruns_reporter)
    assert unknown_xruns_reporter["accepted"] is False
    assert unknown_xruns_reporter["runs"][0]["audible_xruns"] is None
    assert next(
        check
        for check in unknown_xruns_reporter["checks"]
        if check["id"] == "audible-continuity"
    )["passed"] is False

    inconsistent_unknown_xruns = json.loads(json.dumps(unknown_xruns_payload))
    inconsistent_unknown_xruns["runs"][0]["continuous_audio"] = True
    try:
        reporter_module.build_v2_document(
            inconsistent_unknown_xruns,
            reporter["github"],
            loader=lambda url: host_payloads[url],
        )
    except reporter_module.ReporterError as error:
        assert "may be null only when continuous_audio is false" in str(error)
    else:
        raise AssertionError("unknown audible XRUNs with continuous audio unexpectedly passed")

    for invalid_audible_xruns in ("unknown", -1, True):
        invalid_xruns_payload = json.loads(json.dumps(reporter_payload))
        invalid_xruns_payload["runs"][0]["audible_xruns"] = invalid_audible_xruns
        try:
            reporter_module.build_v2_document(
                invalid_xruns_payload,
                reporter["github"],
                loader=lambda url: host_payloads[url],
            )
        except reporter_module.ReporterError:
            pass
        else:
            raise AssertionError(
                f"invalid audible XRUN value unexpectedly passed: {invalid_audible_xruns!r}"
            )

    missing_xruns_payload = json.loads(json.dumps(reporter_payload))
    del missing_xruns_payload["runs"][0]["audible_xruns"]
    try:
        reporter_module.build_v2_document(
            missing_xruns_payload,
            reporter["github"],
            loader=lambda url: host_payloads[url],
        )
    except reporter_module.ReporterError:
        pass
    else:
        raise AssertionError("missing audible XRUN field unexpectedly passed")

    failed_body = "```json\n" + json.dumps(failed_payload) + "\n```"
    failed_comment_url = (
        "https://github.com/penguin425/denoize/issues/221#issuecomment-2"
    )

    def reporter_api(path: str):
        if path == "/repos/penguin425/denoize/issues/221":
            return {"user": {"login": "UlisesMilani"}}
        assert path == "/repos/penguin425/denoize/issues/comments/2"
        return {
            "user": {"login": "UlisesMilani"},
            "html_url": failed_comment_url,
            "body": failed_body,
            "created_at": "2026-09-03T00:00:00Z",
            "updated_at": "2026-09-03T00:00:00Z",
        }

    reporter_module.api = reporter_api
    reporter_module.attachment = lambda url: failed_hosts[url]
    preserved_path = root / "reporter-rejected.json"
    assert reporter_module.generate(
        SimpleNamespace(comment_url=failed_comment_url, output=preserved_path)
    ) is False
    preserved = json.loads(preserved_path.read_text(encoding="utf-8"))
    validators["reporter_v2"].validate(preserved)
    assert preserved["runs"][0]["overload_blocks"] == 7
    assert preserved["accepted"] is False

    legacy_payload = json.loads(json.dumps(reporter_payload))
    legacy_hosts = {}
    for run in legacy_payload["runs"]:
        url = run["host_evidence_url"]
        legacy = json.loads(host_payloads[url])
        legacy["schema"] = "denoize-dpdfnet-clap-host-run-v1"
        legacy["schema_version"] = 1
        del legacy["host_audio_configuration"]
        del legacy["callback_frames"]
        encoded = (json.dumps(legacy, sort_keys=True) + "\n").encode()
        legacy_hosts[url] = encoded
        run["host_evidence_sha256"] = hashlib.sha256(encoded).hexdigest()
    legacy_reporter = reporter_module.build_v2_document(
        legacy_payload,
        reporter["github"],
        loader=lambda url: legacy_hosts[url],
    )
    validators["reporter_v2"].validate(legacy_reporter)
    assert legacy_reporter["accepted"] is False
    assert all(run["effective_buffer_frames"] is None for run in legacy_reporter["runs"])
    assert next(
        check
        for check in legacy_reporter["checks"]
        if check["id"] == "effective-buffer-observed"
    )["passed"] is False
    reporter_path = root / "reporter.json"
    reporter_path.write_text(json.dumps(reporter) + "\n", encoding="utf-8")

    platform_paths = []
    platform_bundles = []
    configurations = [
        (
            "linux",
            "x86_64-unknown-linux-gnu",
            "x86_64",
            "portable-ci",
            "ubuntu-24.04",
            2,
        ),
        (
            "macos",
            "aarch64-apple-darwin",
            "aarch64",
            "portable-ci",
            "macos-15",
            3,
        ),
        (
            "windows",
            "x86_64-pc-windows-msvc",
            "x86_64",
            "portable-ci",
            "windows-2025",
            4,
        ),
        (
            "linux",
            "x86_64-unknown-linux-gnu",
            "x86_64",
            "lowest-supported",
            "ubuntu-slim",
            1,
        ),
    ]
    for index, (
        operating_system,
        target,
        arch,
        tier,
        runner_label,
        logical_cpus,
    ) in enumerate(configurations):
        document = json.loads(json.dumps(platform))
        document["platform"].update(
            {
                "os": operating_system,
                "os_version": f"fixture-{operating_system}",
                "target": target,
                "arch": arch,
                "cpu_model": f"fixture-{operating_system}-{tier}",
                "hardware_tier": tier,
                "runner_label": runner_label,
                "logical_cpus": logical_cpus,
            }
        )
        document["measurement"]["stress_realtime_paced"] = True
        document["measurement"]["deadline_clock"] = (
            "process-cpu"
            if operating_system in {"macos", "windows"}
            else "monotonic-wall"
        )
        validators["platform"].validate(document)
        path = root / f"platform-{index}.json"
        path.write_text(json.dumps(document) + "\n", encoding="utf-8")
        bundle = root / f"platform-{index}.sigstore.json"
        bundle.write_text("{}\n", encoding="utf-8")
        platform_paths.append(path)
        platform_bundles.append(bundle)

    artifact_bundle = root / "artifact.sigstore.json"
    artifact_bundle.write_text("{}\n", encoding="utf-8")

    def verify_attestation(subject: Path, bundle: Path, source_commit: str):
        assert source_commit == commit
        return {
            "subject_sha256": hashlib.sha256(subject.read_bytes()).hexdigest(),
            "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
            "verified_attestations": 1,
            "repository": "penguin425/denoize",
            "signer_workflow": module.SIGNER_WORKFLOW,
            "source_commit": commit,
        }

    def inspect_windows_archive(path: Path, source_commit: str):
        assert source_commit == commit and path == artifact
        return {
            "archive": module.file_record(path, path.read_bytes()),
            "package_manifest_sha256": "2" * 64,
            "target": "x86_64-pc-windows-msvc",
            "plugin_sha256": "3" * 64,
            "model_sha256": module.MODEL_SHA256,
            "descriptor_count": 3,
        }

    module.verify_attestation = verify_attestation
    module.inspect_windows_archive = inspect_windows_archive
    arguments = SimpleNamespace(
        source_commit=commit,
        evaluation_summary=summary_path,
        listening_result=listening_path,
        reaper_automated=reaper_path,
        reporter_evidence=reporter_path,
        artifact=artifact,
        artifact_attestation=artifact_bundle,
        platform_evidence=platform_paths,
        platform_attestation=platform_bundles,
        output=root / "promotion.json",
        allow_rejected=False,
    )
    return arguments, module


def main() -> int:
    validators = {}
    for name, path in SCHEMAS.items():
        document = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(document)
        validators[name] = Draft202012Validator(document)
    with tempfile.TemporaryDirectory(prefix="denoize-dpdfnet-promotion-") as temporary:
        root = Path(temporary)
        clap_host = {
            "schema": "denoize-dpdfnet-clap-host-run-v1",
            "schema_version": 1,
            "source_commit": "0123456789abcdef0123456789abcdef01234567",
            "model_id": "dpdfnet2-48khz-hr",
            "model_sha256": "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b",
            "plugin_id": "org.penguin425.denoize.neural-hq",
            "sample_rate_hz": 48_000,
            "channels": 2,
            "chunk_frames": 480,
            "latency_frames": 11_520,
            "processed_frames": 2_976_000,
            "active_seconds": 61.8,
            "measurement": {
                "warmup_frames": 46_080,
                "measured_frames": 2_929_920,
            },
            "worker_started": True,
            "finished_gracefully": True,
            "metrics": {
                "overload_blocks": 0,
                "late_blocks": 0,
                "invalid_blocks": 0,
                "worker_errors": 0,
            },
            "lifetime_metrics": {
                "overload_blocks": 24,
                "late_blocks": 6,
                "invalid_blocks": 5,
                "worker_errors": 0,
            },
            "environment": {"os": "windows", "arch": "x86_64"},
        }
        validators["clap_host"].validate(clap_host)
        matrix, audio = fixture(root)
        bundle, answer_path = prepare(root, matrix, audio)
        protocol_path = bundle / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        answer = json.loads(answer_path.read_text(encoding="utf-8"))
        validators["protocol"].validate(protocol)
        validators["answer"].validate(answer)
        public_payload = protocol_path.read_text(encoding="utf-8") + (bundle / "index.html").read_text(encoding="utf-8")
        assert "dpdfnet2-48khz-hr" not in public_payload
        assert "gtcrn-dns3" not in public_payload
        assert len(protocol["trials"]) == 16
        assert sum(trial["role"] == "core" for trial in answer["trials"]) == 12
        assert sum(trial["role"] == "repeat" for trial in answer["trials"]) == 4

        passing_responses = responses(root, protocol, answer, candidate=True)
        for path in passing_responses.glob("*.json"):
            validators["response"].validate(json.loads(path.read_text(encoding="utf-8")))
        passing_result = root / "passing-result.json"
        run(
            [
                sys.executable,
                str(SCORE),
                "--protocol",
                str(protocol_path),
                "--answer-key",
                str(answer_path),
                "--responses",
                str(passing_responses),
                "--output",
                str(passing_result),
            ]
        )
        passing = json.loads(passing_result.read_text(encoding="utf-8"))
        validators["result"].validate(passing)
        assert passing["accepted"] is True
        assert passing["listeners"]["retained"] == 20
        assert passing["overall"]["dpdfnet_preference_score"] == 1.0
        assert passing["duplicate_consistency"]["inconsistencies"] == 0

        rejected_responses = responses(root, protocol, answer, candidate=False)
        rejected_result = root / "rejected-result.json"
        run(
            [
                sys.executable,
                str(SCORE),
                "--protocol",
                str(protocol_path),
                "--answer-key",
                str(answer_path),
                "--responses",
                str(rejected_responses),
                "--output",
                str(rejected_result),
                "--allow-rejected",
            ]
        )
        rejected = json.loads(rejected_result.read_text(encoding="utf-8"))
        validators["result"].validate(rejected)
        assert rejected["accepted"] is False
        assert rejected["overall"]["dpdfnet_preference_score"] == 0.0

        duplicate = run(
            [
                sys.executable,
                str(PREPARE),
                "--matrix-result",
                str(matrix),
                "--audio-dir",
                str(audio),
                "--randomization-key",
                str(root / "randomization.key"),
                "--output-dir",
                str(bundle),
                "--answer-key",
                str(root / "another-answer.json"),
            ],
            success=False,
        )
        assert "refusing to replace" in duplicate.stderr

        matrix_link = root / "matrix-link.json"
        matrix_link.symlink_to(matrix)
        linked_matrix = run(
            [
                sys.executable,
                str(PREPARE),
                "--matrix-result",
                str(matrix_link),
                "--audio-dir",
                str(audio),
                "--randomization-key",
                str(root / "randomization.key"),
                "--output-dir",
                str(root / "linked-matrix-bundle"),
                "--answer-key",
                str(root / "linked-matrix-answer.json"),
            ],
            success=False,
        )
        assert "not a regular file" in linked_matrix.stderr

        answer_link = root / "answer-link.json"
        answer_link.symlink_to(answer_path)
        linked_answer = run(
            [
                sys.executable,
                str(SCORE),
                "--protocol",
                str(protocol_path),
                "--answer-key",
                str(answer_link),
                "--responses",
                str(passing_responses),
                "--output",
                str(root / "linked-answer-result.json"),
            ],
            success=False,
        )
        assert "not a regular file" in linked_answer.stderr

        tampered_bundle = root / "tampered-bundle"
        shutil.copytree(bundle, tampered_bundle)
        with (tampered_bundle / protocol["trials"][0]["audio"]["a"]["path"]).open("ab") as output:
            output.write(b"tamper")
        tampered = run(
            [
                sys.executable,
                str(SCORE),
                "--protocol",
                str(tampered_bundle / "protocol.json"),
                "--answer-key",
                str(answer_path),
                "--responses",
                str(passing_responses),
                "--output",
                str(root / "tampered-result.json"),
            ],
            success=False,
        )
        assert "fingerprint mismatch" in tampered.stderr

        platform_root = root / "platform"
        platform_root.mkdir()
        stress_path, worker_path = platform_fixture(platform_root)
        validators["worker"].validate(json.loads(worker_path.read_text(encoding="utf-8")))
        platform_result = platform_root / "platform.json"
        run(
            [
                sys.executable,
                str(PLATFORM),
                "--stress",
                str(stress_path),
                "--worker",
                str(worker_path),
                "--output",
                str(platform_result),
            ]
        )
        platform = json.loads(platform_result.read_text(encoding="utf-8"))
        validators["platform"].validate(platform)
        assert platform["accepted"] is True
        assert len(platform["checks"]) == 11

        unpaced_stress = json.loads(stress_path.read_text(encoding="utf-8"))
        unpaced_stress["realtime_paced"] = False
        unpaced_stress_path = platform_root / "unpaced-stress.json"
        unpaced_stress_path.write_text(
            json.dumps(unpaced_stress) + "\n", encoding="utf-8"
        )
        unpaced_result = run(
            [
                sys.executable,
                str(PLATFORM),
                "--stress",
                str(unpaced_stress_path),
                "--worker",
                str(worker_path),
                "--output",
                str(platform_root / "unpaced-platform.json"),
            ],
            success=False,
        )
        assert "promotion stress evidence must be real-time paced" in unpaced_result.stderr

        fast_worker = json.loads(worker_path.read_text(encoding="utf-8"))
        fast_worker["measurement_wall_seconds"] = 50.0
        fast_worker_path = platform_root / "fast-worker.json"
        fast_worker_path.write_text(
            json.dumps(fast_worker) + "\n", encoding="utf-8"
        )
        fast_worker_result = run(
            [
                sys.executable,
                str(PLATFORM),
                "--stress",
                str(stress_path),
                "--worker",
                str(fast_worker_path),
                "--output",
                str(platform_root / "fast-worker-platform.json"),
            ],
            success=False,
        )
        assert "completed too quickly" in fast_worker_result.stderr

        slow_worker = json.loads(worker_path.read_text(encoding="utf-8"))
        slow_worker["measurement_wall_seconds"] = 204.0
        slow_worker_path = platform_root / "slow-worker.json"
        slow_worker_path.write_text(
            json.dumps(slow_worker) + "\n", encoding="utf-8"
        )
        slow_worker_result = run(
            [
                sys.executable,
                str(PLATFORM),
                "--stress",
                str(stress_path),
                "--worker",
                str(slow_worker_path),
                "--output",
                str(platform_root / "slow-worker-platform.json"),
            ],
            success=False,
        )
        assert "completed too slowly" in slow_worker_result.stderr

        short_worker = json.loads(worker_path.read_text(encoding="utf-8"))
        short_worker["paced_blocks"] = 100
        short_worker["measured_frames"] = 59_520
        short_worker["finite_frames"] = 59_520
        short_worker["neural_frames"] = 48_000
        short_worker["measurement_wall_seconds"] = 1.25
        short_worker_path = platform_root / "short-worker.json"
        short_worker_path.write_text(
            json.dumps(short_worker) + "\n", encoding="utf-8"
        )
        short_worker_platform_path = platform_root / "short-worker-platform.json"
        run(
            [
                sys.executable,
                str(PLATFORM),
                "--stress",
                str(stress_path),
                "--worker",
                str(short_worker_path),
                "--output",
                str(short_worker_platform_path),
                "--allow-rejected",
            ]
        )
        short_worker_platform = json.loads(
            short_worker_platform_path.read_text(encoding="utf-8")
        )
        validators["platform"].validate(short_worker_platform)
        assert short_worker_platform["accepted"] is False
        short_worker_check = next(
            check
            for check in short_worker_platform["checks"]
            if check["id"] == "minimum-paced-worker-blocks"
        )
        assert short_worker_check == {
            "id": "minimum-paced-worker-blocks",
            "observed": 100,
            "operator": "greater-or-equal",
            "limit": 6_000,
            "passed": False,
        }

        preempted_stress = json.loads(stress_path.read_text(encoding="utf-8"))
        preempted_stress["timing"].update(
            {
                "p99_9_ms": 8.654417,
                "maximum_ms": 13.000958,
                "calls_over_budget": 1,
                "calls_over_budget_fraction": 1 / 6000,
            }
        )
        preempted_stress_path = platform_root / "preempted-stress.json"
        preempted_stress_path.write_text(
            json.dumps(preempted_stress) + "\n", encoding="utf-8"
        )
        preempted_platform_path = platform_root / "preempted-platform.json"
        run(
            [
                sys.executable,
                str(PLATFORM),
                "--stress",
                str(preempted_stress_path),
                "--worker",
                str(worker_path),
                "--output",
                str(preempted_platform_path),
            ]
        )
        preempted_platform = json.loads(
            preempted_platform_path.read_text(encoding="utf-8")
        )
        validators["platform"].validate(preempted_platform)
        assert preempted_platform["accepted"] is True
        by_id = {item["id"]: item for item in preempted_platform["checks"]}
        assert by_id["stress-maximum-ms"]["limit"] == 20.0
        assert by_id["stress-deadline-misses"]["limit"] == 6

        mac_stress = json.loads(stress_path.read_text(encoding="utf-8"))
        mac_stress["environment"].update(
            {
                "os": "macos",
                "arch": "aarch64",
                "target": "aarch64-apple-darwin",
                "os_version": "macOS-15.7.9-fixture",
                "cpu_model": "Apple M1 (Virtual)",
                "runner_label": "macos-15",
                "logical_parallelism": 3,
            }
        )
        mac_stress["timing"].update(
            {
                "p99_9_ms": 11.37075,
                "maximum_ms": 16.252792,
                "calls_over_budget": 17,
                "calls_over_budget_fraction": 17 / 6000,
                "summed_compute_rtf": 0.39796,
                "process_cpu": {
                    "clock": "CLOCK_PROCESS_CPUTIME_ID",
                    "sample_count": 6_000,
                    "budget_ms": 10.0,
                    "p99_9_ms": 8.2,
                    "maximum_ms": 9.3,
                    "calls_over_budget": 0,
                    "summed_compute_rtf": 0.36,
                },
            }
        )
        mac_worker = json.loads(worker_path.read_text(encoding="utf-8"))
        mac_worker["environment"].update(
            {
                "os": "macos",
                "arch": "aarch64",
                "target": "aarch64-apple-darwin",
                "cpu_model": "Apple M1 (Virtual)",
                "runner_label": "macos-15",
                "logical_parallelism": 3,
            }
        )
        mac_stress_path = platform_root / "mac-stress.json"
        mac_worker_path = platform_root / "mac-worker.json"
        mac_stress_path.write_text(
            json.dumps(mac_stress) + "\n", encoding="utf-8"
        )
        mac_worker_path.write_text(
            json.dumps(mac_worker) + "\n", encoding="utf-8"
        )
        mac_platform_path = platform_root / "mac-platform.json"
        run(
            [
                sys.executable,
                str(PLATFORM),
                "--stress",
                str(mac_stress_path),
                "--worker",
                str(mac_worker_path),
                "--output",
                str(mac_platform_path),
            ]
        )
        mac_platform = json.loads(mac_platform_path.read_text(encoding="utf-8"))
        validators["platform"].validate(mac_platform)
        assert mac_platform["accepted"] is True
        assert mac_platform["measurement"]["deadline_clock"] == "process-cpu"
        assert mac_platform["measurement"]["p99_9_ms"] == 8.2
        assert mac_platform["measurement"]["wall_p99_9_ms"] == 11.37075
        assert all(check["passed"] is True for check in mac_platform["checks"])

        mac_without_cpu = json.loads(json.dumps(mac_stress))
        del mac_without_cpu["timing"]["process_cpu"]
        mac_without_cpu_path = platform_root / "mac-without-cpu-stress.json"
        mac_without_cpu_path.write_text(
            json.dumps(mac_without_cpu) + "\n", encoding="utf-8"
        )
        missing_cpu = run(
            [
                sys.executable,
                str(PLATFORM),
                "--stress",
                str(mac_without_cpu_path),
                "--worker",
                str(mac_worker_path),
                "--output",
                str(platform_root / "mac-without-cpu-platform.json"),
            ],
            success=False,
        )
        assert "macOS stress run lacks process CPU timing" in missing_cpu.stderr

        windows_stress = json.loads(stress_path.read_text(encoding="utf-8"))
        windows_stress["environment"].update(
            {
                "os": "windows",
                "arch": "x86_64",
                "target": "x86_64-pc-windows-msvc",
                "os_version": "Microsoft Windows NT 10.0.26100.0",
                "cpu_model": "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz",
                "runner_label": "windows-2025",
                "logical_parallelism": 4,
            }
        )
        windows_stress["timing"].update(
            {
                "p99_9_ms": 10.3638,
                "maximum_ms": 14.518,
                "calls_over_budget": 12,
                "calls_over_budget_fraction": 12 / 6000,
                "summed_compute_rtf": 0.59573,
                "process_cpu": {
                    "clock": "GetProcessTimes",
                    "sample_count": 6_000,
                    "budget_ms": 10.0,
                    "p99_9_ms": 8.7,
                    "maximum_ms": 9.8,
                    "calls_over_budget": 0,
                    "summed_compute_rtf": 0.56,
                },
            }
        )
        windows_worker = json.loads(worker_path.read_text(encoding="utf-8"))
        windows_worker["environment"].update(
            {
                "os": "windows",
                "arch": "x86_64",
                "target": "x86_64-pc-windows-msvc",
                "cpu_model": "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz",
                "runner_label": "windows-2025",
                "logical_parallelism": 4,
            }
        )
        windows_stress_path = platform_root / "windows-stress.json"
        windows_worker_path = platform_root / "windows-worker.json"
        windows_stress_path.write_text(
            json.dumps(windows_stress) + "\n", encoding="utf-8"
        )
        windows_worker_path.write_text(
            json.dumps(windows_worker) + "\n", encoding="utf-8"
        )
        windows_platform_path = platform_root / "windows-platform.json"
        run(
            [
                sys.executable,
                str(PLATFORM),
                "--stress",
                str(windows_stress_path),
                "--worker",
                str(windows_worker_path),
                "--output",
                str(windows_platform_path),
            ]
        )
        windows_platform = json.loads(
            windows_platform_path.read_text(encoding="utf-8")
        )
        validators["platform"].validate(windows_platform)
        assert windows_platform["accepted"] is True
        assert windows_platform["measurement"]["deadline_clock"] == "process-cpu"
        assert windows_platform["measurement"]["p99_9_ms"] == 8.7
        assert windows_platform["measurement"]["wall_p99_9_ms"] == 10.3638
        assert all(check["passed"] is True for check in windows_platform["checks"])

        windows_without_cpu = json.loads(json.dumps(windows_stress))
        del windows_without_cpu["timing"]["process_cpu"]
        windows_without_cpu_path = platform_root / "windows-without-cpu-stress.json"
        windows_without_cpu_path.write_text(
            json.dumps(windows_without_cpu) + "\n", encoding="utf-8"
        )
        missing_windows_cpu = run(
            [
                sys.executable,
                str(PLATFORM),
                "--stress",
                str(windows_without_cpu_path),
                "--worker",
                str(windows_worker_path),
                "--output",
                str(platform_root / "windows-without-cpu-platform.json"),
            ],
            success=False,
        )
        assert (
            "Windows stress run lacks process CPU timing"
            in missing_windows_cpu.stderr
        )

        windows_wrong_clock = json.loads(json.dumps(windows_stress))
        windows_wrong_clock["timing"]["process_cpu"]["clock"] = (
            "CLOCK_PROCESS_CPUTIME_ID"
        )
        windows_wrong_clock_path = platform_root / "windows-wrong-clock-stress.json"
        windows_wrong_clock_path.write_text(
            json.dumps(windows_wrong_clock) + "\n", encoding="utf-8"
        )
        wrong_windows_clock = run(
            [
                sys.executable,
                str(PLATFORM),
                "--stress",
                str(windows_wrong_clock_path),
                "--worker",
                str(windows_worker_path),
                "--output",
                str(platform_root / "windows-wrong-clock-platform.json"),
            ],
            success=False,
        )
        assert (
            "Windows stress run used an unsupported process CPU clock"
            in wrong_windows_clock.stderr
        )

        lowest_stress = json.loads(stress_path.read_text(encoding="utf-8"))
        lowest_stress["environment"].update(
            {
                "logical_parallelism": 1,
                "hardware_tier": "lowest-supported",
                "runner_label": "ubuntu-slim",
            }
        )
        lowest_stress["realtime_paced"] = True
        lowest_worker = json.loads(worker_path.read_text(encoding="utf-8"))
        lowest_worker["environment"].update(
            {
                "logical_parallelism": 1,
                "hardware_tier": "lowest-supported",
                "runner_label": "ubuntu-slim",
            }
        )
        lowest_stress_path = platform_root / "lowest-stress.json"
        lowest_worker_path = platform_root / "lowest-worker.json"
        lowest_stress_path.write_text(
            json.dumps(lowest_stress) + "\n", encoding="utf-8"
        )
        lowest_worker_path.write_text(
            json.dumps(lowest_worker) + "\n", encoding="utf-8"
        )
        lowest_platform_path = platform_root / "lowest-platform.json"
        run(
            [
                sys.executable,
                str(PLATFORM),
                "--stress",
                str(lowest_stress_path),
                "--worker",
                str(lowest_worker_path),
                "--output",
                str(lowest_platform_path),
            ]
        )
        lowest_platform = json.loads(
            lowest_platform_path.read_text(encoding="utf-8")
        )
        validators["platform"].validate(lowest_platform)
        assert lowest_platform["accepted"] is True
        assert lowest_platform["platform"]["logical_cpus"] == 1
        assert lowest_platform["platform"]["runner_label"] == "ubuntu-slim"
        assert lowest_platform["measurement"]["stress_realtime_paced"] is True

        unpaced_lowest = json.loads(json.dumps(lowest_stress))
        unpaced_lowest["realtime_paced"] = False
        unpaced_lowest_path = platform_root / "unpaced-lowest-stress.json"
        unpaced_lowest_path.write_text(
            json.dumps(unpaced_lowest) + "\n", encoding="utf-8"
        )
        unpaced_lowest_result = run(
            [
                sys.executable,
                str(PLATFORM),
                "--stress",
                str(unpaced_lowest_path),
                "--worker",
                str(lowest_worker_path),
                "--output",
                str(platform_root / "unpaced-lowest-platform.json"),
            ],
            success=False,
        )
        assert "must be real-time paced" in unpaced_lowest_result.stderr

        false_lowest = json.loads(json.dumps(lowest_stress))
        false_lowest["environment"]["logical_parallelism"] = 2
        false_lowest_path = platform_root / "false-lowest-stress.json"
        false_lowest_path.write_text(
            json.dumps(false_lowest) + "\n", encoding="utf-8"
        )
        false_lowest_result = run(
            [
                sys.executable,
                str(PLATFORM),
                "--stress",
                str(false_lowest_path),
                "--worker",
                str(lowest_worker_path),
                "--output",
                str(platform_root / "false-lowest-platform.json"),
            ],
            success=False,
        )
        assert "one logical CPU" in false_lowest_result.stderr

        rejected_stress = json.loads(stress_path.read_text(encoding="utf-8"))
        rejected_stress["timing"]["p99_9_ms"] = 11.0
        rejected_stress_path = platform_root / "rejected-stress.json"
        rejected_stress_path.write_text(json.dumps(rejected_stress) + "\n", encoding="utf-8")
        rejected_platform_path = platform_root / "rejected-platform.json"
        run(
            [
                sys.executable,
                str(PLATFORM),
                "--stress",
                str(rejected_stress_path),
                "--worker",
                str(worker_path),
                "--output",
                str(rejected_platform_path),
                "--allow-rejected",
            ]
        )
        rejected_platform = json.loads(rejected_platform_path.read_text(encoding="utf-8"))
        validators["platform"].validate(rejected_platform)
        assert rejected_platform["accepted"] is False

        promotion_root = root / "promotion"
        promotion_root.mkdir()
        arguments, module = composite_fixtures(
            promotion_root, passing, platform, validators
        )
        assert module.generate(arguments) is True
        promotion = json.loads(arguments.output.read_text(encoding="utf-8"))
        validators["promotion"].validate(promotion)
        assert promotion["accepted"] is True
        assert len(promotion["platforms"]) == 4
        assert {entry["os"] for entry in promotion["platforms"]} == {
            "linux",
            "macos",
            "windows",
        }
        assert sum(
            entry["hardware_tier"] == "lowest-supported"
            for entry in promotion["platforms"]
        ) == 1

        legacy_portable = json.loads(
            arguments.platform_evidence[0].read_text(encoding="utf-8")
        )
        legacy_portable["schema"] = "denoize-dpdfnet-platform-evidence-v1"
        legacy_portable["schema_version"] = 1
        for field in (
            "stress_realtime_paced",
            "deadline_clock",
            "wall_p99_9_ms",
            "wall_maximum_ms",
            "wall_deadline_misses",
            "wall_summed_compute_rtf",
        ):
            del legacy_portable["measurement"][field]
        validators["platform_v1"].validate(legacy_portable)
        legacy_portable_path = promotion_root / "platform-portable-v1.json"
        legacy_portable_path.write_text(
            json.dumps(legacy_portable) + "\n", encoding="utf-8"
        )
        mixed_versions = SimpleNamespace(**vars(arguments))
        mixed_versions.platform_evidence = [
            legacy_portable_path,
            *arguments.platform_evidence[1:],
        ]
        mixed_versions.output = promotion_root / "promotion-mixed-platform-versions.json"
        assert module.generate(mixed_versions) is True
        validators["promotion"].validate(
            json.loads(mixed_versions.output.read_text(encoding="utf-8"))
        )

        legacy_lowest = json.loads(
            arguments.platform_evidence[3].read_text(encoding="utf-8")
        )
        legacy_lowest["schema"] = "denoize-dpdfnet-platform-evidence-v1"
        legacy_lowest["schema_version"] = 1
        for field in (
            "stress_realtime_paced",
            "deadline_clock",
            "wall_p99_9_ms",
            "wall_maximum_ms",
            "wall_deadline_misses",
            "wall_summed_compute_rtf",
        ):
            del legacy_lowest["measurement"][field]
        validators["platform_v1"].validate(legacy_lowest)
        legacy_lowest_path = promotion_root / "platform-lowest-v1.json"
        legacy_lowest_path.write_text(
            json.dumps(legacy_lowest) + "\n", encoding="utf-8"
        )
        v1_lowest = SimpleNamespace(**vars(arguments))
        v1_lowest.platform_evidence = [
            *arguments.platform_evidence[:3],
            legacy_lowest_path,
        ]
        v1_lowest.output = promotion_root / "promotion-v1-lowest.json"
        try:
            module.generate(v1_lowest)
        except module.PromotionError as error:
            assert "real-time-paced v2 schema" in str(error)
        else:
            raise AssertionError("v1 lowest-supported evidence unexpectedly passed")

        falsely_paced_lowest = json.loads(
            arguments.platform_evidence[3].read_text(encoding="utf-8")
        )
        falsely_paced_lowest["measurement"]["stress_realtime_paced"] = False
        assert validators["platform"].is_valid(falsely_paced_lowest) is False
        falsely_paced_lowest_path = promotion_root / "platform-lowest-unpaced-v2.json"
        falsely_paced_lowest_path.write_text(
            json.dumps(falsely_paced_lowest) + "\n", encoding="utf-8"
        )
        unpaced_v2_lowest = SimpleNamespace(**vars(arguments))
        unpaced_v2_lowest.platform_evidence = [
            *arguments.platform_evidence[:3],
            falsely_paced_lowest_path,
        ]
        unpaced_v2_lowest.output = promotion_root / "promotion-unpaced-v2-lowest.json"
        try:
            module.generate(unpaced_v2_lowest)
        except module.PromotionError as error:
            assert "must record real-time pacing" in str(error)
        else:
            raise AssertionError("unpaced v2 lowest-supported evidence unexpectedly passed")

        no_lowest = SimpleNamespace(**vars(arguments))
        no_lowest.platform_evidence = arguments.platform_evidence[:3]
        no_lowest.platform_attestation = arguments.platform_attestation[:3]
        no_lowest.output = promotion_root / "promotion-no-lowest.json"
        assert module.generate(no_lowest) is False
        rejected_promotion = json.loads(
            no_lowest.output.read_text(encoding="utf-8")
        )
        validators["promotion"].validate(rejected_promotion)
        assert rejected_promotion["accepted"] is False
        lowest_check = next(
            check
            for check in rejected_promotion["checks"]
            if check["id"] == "lowest-supported-hardware"
        )
        assert lowest_check["passed"] is False

        duplicate_platform = SimpleNamespace(**vars(arguments))
        duplicate_platform.platform_evidence = [
            *arguments.platform_evidence[:3],
            arguments.platform_evidence[0],
        ]
        duplicate_platform.platform_attestation = [
            *arguments.platform_attestation[:3],
            arguments.platform_attestation[0],
        ]
        duplicate_platform.output = promotion_root / "promotion-duplicate.json"
        try:
            module.generate(duplicate_platform)
        except module.PromotionError as error:
            assert "duplicate portable-ci platform evidence for linux" in str(error)
        else:
            raise AssertionError("duplicate platform evidence unexpectedly passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
