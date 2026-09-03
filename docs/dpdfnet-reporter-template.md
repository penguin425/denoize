# Issue #221 reporter test

Use the attested Windows experimental CLAP archive from the DPDFNet promotion
workflow. Verify it with `gh attestation verify`, then test `denoize Neural HQ`
in REAPER 7.79 or newer with NVDA and OSARA.

## Capture each run

Fully exit REAPER before every run. From PowerShell, set a new evidence path
and launch REAPER from that same shell so it inherits the variables:

```powershell
$env:DENOIZE_EVIDENCE_SOURCE_COMMIT = "40_HEX_COMMIT"
$env:DENOIZE_NEURAL_HOST_EVIDENCE = "$PWD\dpdfnet-128.json"
& "C:\Program Files\REAPER (x64)\reaper.exe"
```

Configure the requested buffer in REAPER, play continuous audio for at least
five measured minutes, then stop playback and exit REAPER. Do not reuse an
evidence filename: the plug-in deliberately refuses to replace an existing
file. Repeat at a requested buffer of 480 frames, one buffer no larger than 128
frames, and one buffer no smaller than 1024 frames.

The generated JSON is the authoritative source for overload, late, invalid,
worker-error, duration, and actual callback-frame measurements. Do not copy or
guess those counters. Record audible XRUNs and continuous-audio status while
listening. Hash each JSON before uploading it to issue #221:

```powershell
Get-FileHash .\dpdfnet-128.json -Algorithm SHA256
```

Upload the three JSON files to the issue, then use their GitHub attachment URLs
and hashes in exactly one fenced JSON object:

```json
{
  "schema": "denoize-dpdfnet-reporter-submission-v2",
  "schema_version": 2,
  "source_commit": "40_HEX_COMMIT",
  "artifact_sha256": "64_HEX_ARCHIVE_DIGEST",
  "environment": {
    "windows_version": "Windows version and build",
    "cpu_model": "CPU model",
    "audio_device": "Audio interface",
    "audio_driver": "Driver type and version",
    "reaper_version": "7.79",
    "nvda_version": "NVDA version",
    "osara_version": "OSARA version"
  },
  "runs": [
    {
      "requested_buffer_frames": 128,
      "host_evidence_url": "https://github.com/user-attachments/files/12345678/dpdfnet-128.json",
      "host_evidence_sha256": "64_HEX_HOST_EVIDENCE_DIGEST",
      "audible_xruns": 0,
      "continuous_audio": true
    },
    {
      "requested_buffer_frames": 480,
      "host_evidence_url": "https://github.com/user-attachments/files/12345679/dpdfnet-480.json",
      "host_evidence_sha256": "64_HEX_HOST_EVIDENCE_DIGEST",
      "audible_xruns": 0,
      "continuous_audio": true
    },
    {
      "requested_buffer_frames": 1024,
      "host_evidence_url": "https://github.com/user-attachments/files/12345680/dpdfnet-1024.json",
      "host_evidence_sha256": "64_HEX_HOST_EVIDENCE_DIGEST",
      "audible_xruns": 0,
      "continuous_audio": true
    }
  ],
  "accessibility": {
    "nvda_active": true,
    "osara_active": true,
    "parameters_announced": ["Bypass", "Mix", "Output Gain", "Overload Fallback"],
    "values_announced": true,
    "all_adjustable": true,
    "focus_stable": true,
    "host_or_plugin_crashes": 0
  },
  "quality_observation": "dpdfnet-better",
  "consent_to_publish": true
}
```

Report observed failures exactly. The importer preserves a structurally valid
failed submission, its raw host files, normalized counters, and individual gate
results with `accepted: false`; it does not require failures to be rewritten as
zero. Host-evidence v1 files are also retained, but cannot pass the new
effective-buffer provenance gate because they did not record callback sizes.
