param(
  [Parameter(Mandatory = $true)][string]$PluginBinary,
  [Parameter(Mandatory = $true)][string]$ModelRoot,
  [Parameter(Mandatory = $true)][string]$OutputDirectory,
  [Parameter(Mandatory = $true)][string]$SourceCommit,
  [int]$DurationSeconds = 60
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($SourceCommit -notmatch '^[0-9a-f]{40}$') {
  throw "SourceCommit must be a lowercase 40-character SHA-1"
}
if ($DurationSeconds -lt 60 -or $DurationSeconds -gt 3600) {
  throw "DurationSeconds must be between 60 and 3600"
}
$PluginBinary = (Resolve-Path -LiteralPath $PluginBinary).Path
$ModelRoot = (Resolve-Path -LiteralPath $ModelRoot).Path
$model = Join-Path $ModelRoot "dpdfnet2-48khz-hr/dpdfnet2_48khz_hr.onnx"
if (-not (Test-Path -LiteralPath $PluginBinary -PathType Leaf)) {
  throw "experimental CLAP binary is missing"
}
if (-not (Test-Path -LiteralPath $model -PathType Leaf)) {
  throw "pinned DPDFNet model is missing"
}
if ((Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash.ToLowerInvariant() -ne
  "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b") {
  throw "pinned DPDFNet model digest differs"
}

$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$result = Join-Path $OutputDirectory "reaper-parameters.tsv"
$hostRun = Join-Path $OutputDirectory "clap-host-run.json"
$processMetrics = Join-Path $OutputDirectory "reaper-process.json"
$evidencePath = Join-Path $OutputDirectory "reaper-automated-evidence.json"
foreach ($path in @($result, $hostRun, $processMetrics, $evidencePath)) {
  if (Test-Path -LiteralPath $path) {
    throw "refusing to replace existing evidence: $path"
  }
}

$installer = Join-Path $env:RUNNER_TEMP "reaper779_x64-install.exe"
$reaperDir = Join-Path $env:RUNNER_TEMP "reaper779-dpdfnet"
$pluginDir = Join-Path $env:RUNNER_TEMP "denoize-dpdfnet-clap"
$resourceDir = Join-Path $env:RUNNER_TEMP "reaper-dpdfnet-resource"
$tone = Join-Path $env:RUNNER_TEMP "dpdfnet-tone.wav"
Invoke-WebRequest -Uri https://www.reaper.fm/files/7.x/reaper779_x64-install.exe -OutFile $installer
if ((Get-FileHash $installer -Algorithm SHA256).Hash -ne
  "F07714D894A073DF40E88568F8AA524A74230F574B0688DF681F9B7C0877F9DF") {
  throw "unexpected REAPER installer digest"
}
New-Item -ItemType Directory -Force $reaperDir, $pluginDir, $resourceDir | Out-Null
Start-Process -FilePath $installer -ArgumentList "/S", "/D=$reaperDir" -Wait
$reaper = Join-Path $reaperDir "reaper.exe"
$plugin = Join-Path $pluginDir "denoize.clap"
Copy-Item -LiteralPath $PluginBinary -Destination $plugin

@(
  "[reaper]",
  "loadlastproj=0",
  "warnmaxram64=0",
  "[audioconfig]",
  "mode=5",
  "dummy_blocksize=480",
  "dummy_srate=48000"
) | Set-Content -LiteralPath (Join-Path $resourceDir "reaper.ini") -Encoding ascii

$sampleRate = 48000
$evidenceWarmupFrames = 46080
$measurementDelaySeconds = $DurationSeconds + 1
# Keep the media item longer than the measured interval. A repeated short item
# introduces transport discontinuities and recurrent-state resets into what is
# intended to be a sustained real-time scheduling measurement.
$sampleCount = $sampleRate * ($DurationSeconds + 10)
$dataSize = $sampleCount * 2
$writer = [System.IO.BinaryWriter]::new([System.IO.File]::Create($tone))
try {
  $writer.Write([System.Text.Encoding]::ASCII.GetBytes("RIFF"))
  $writer.Write([int](36 + $dataSize))
  $writer.Write([System.Text.Encoding]::ASCII.GetBytes("WAVEfmt "))
  $writer.Write([int]16)
  $writer.Write([int16]1)
  $writer.Write([int16]1)
  $writer.Write([int]$sampleRate)
  $writer.Write([int]($sampleRate * 2))
  $writer.Write([int16]2)
  $writer.Write([int16]16)
  $writer.Write([System.Text.Encoding]::ASCII.GetBytes("data"))
  $writer.Write([int]$dataSize)
  for ($sample = 0; $sample -lt $sampleCount; $sample++) {
    $value = [int16](12000 * [Math]::Sin(2 * [Math]::PI * 440 * $sample / $sampleRate))
    $writer.Write($value)
  }
} finally {
  $writer.Dispose()
}

Add-Type @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public static class DpdfnetReaperDialog {
    public delegate bool EnumWindowsProc(IntPtr window, IntPtr data);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern IntPtr FindWindow(string className, string windowName);
    [DllImport("user32.dll")] public static extern IntPtr GetDlgItem(IntPtr dialog, int id);
    [DllImport("user32.dll")] public static extern int GetDlgCtrlID(IntPtr window);
    [DllImport("user32.dll")] public static extern IntPtr GetParent(IntPtr window);
    [DllImport("user32.dll")] public static extern bool EnumChildWindows(IntPtr parent, EnumWindowsProc callback, IntPtr data);
    [DllImport("user32.dll")] public static extern IntPtr SendMessage(IntPtr window, uint message, IntPtr word, IntPtr data);
    [DllImport("user32.dll", CharSet = CharSet.Unicode, EntryPoint = "SendMessageW")]
    public static extern IntPtr SendMessageString(IntPtr window, uint message, IntPtr word, StringBuilder data);
    public static IntPtr[] Descendants(IntPtr parent) {
        var values = new List<IntPtr>();
        EnumChildWindows(parent, delegate(IntPtr window, IntPtr data) { values.Add(window); return true; }, IntPtr.Zero);
        return values.ToArray();
    }
    public static string ComboItem(IntPtr combo, int index) {
        var value = new StringBuilder(1024);
        SendMessageString(combo, 0x0148, new IntPtr(index), value);
        return value.ToString();
    }
}
"@

$env:CLAP_PATH = $pluginDir
$env:DENOIZE_MODEL_DIR = $ModelRoot
$env:DENOIZE_EVIDENCE_SOURCE_COMMIT = $SourceCommit
$env:DENOIZE_NEURAL_HOST_EVIDENCE = $hostRun
$env:DENOIZE_REAPER_RESULT = $result
$env:DENOIZE_REAPER_PLUGIN = "denoize Neural HQ"
$env:DENOIZE_REAPER_OSARA_STYLE = "1"
$env:DENOIZE_REAPER_NORMALIZED = "0"
$env:DENOIZE_REAPER_PLAY_DELAY = "3"
$env:DENOIZE_REAPER_SET_DELAY = [string]$measurementDelaySeconds
$env:DENOIZE_REAPER_AUDIO = $tone
$env:DENOIZE_REAPER_PLAY = "1"
$env:DENOIZE_REAPER_REPEAT = "0"
$env:DENOIZE_REAPER_OPEN_AUDIO_DEVICE = "1"
$env:DENOIZE_REAPER_REMOVE_FX = "1"
$env:DENOIZE_REAPER_PLUGIN_PARAMETER_COUNT = "4"
$env:DENOIZE_REAPER_BYPASS_LATCH = "1"

$arguments = @(
  "-newinst", "-cfgfile", (Join-Path $resourceDir "reaper.ini"),
  (Join-Path $env:GITHUB_WORKSPACE "scripts/reaper-clap-parameter-smoke.lua")
)
$startedAt = Get-Date
$process = Start-Process -FilePath $reaper -ArgumentList $arguments -PassThru
$cpuBefore = $process.TotalProcessorTime.TotalSeconds
$preferencesConfigured = $false
try {
  $deadline = (Get-Date).AddSeconds($DurationSeconds + 90)
  do {
    Start-Sleep -Milliseconds 500
    $dialog = [DpdfnetReaperDialog]::FindWindow("#32770", "REAPER")
    if ($dialog -ne [IntPtr]::Zero) {
      $yes = [DpdfnetReaperDialog]::GetDlgItem($dialog, 6)
      if ($yes -ne [IntPtr]::Zero) {
        [void][DpdfnetReaperDialog]::SendMessage($yes, 0x00F5, [IntPtr]::Zero, [IntPtr]::Zero)
      }
    }
    $preferences = [DpdfnetReaperDialog]::FindWindow("#32770", "REAPER Preferences")
    if (-not $preferencesConfigured -and $preferences -ne [IntPtr]::Zero) {
      $selected = $false
      foreach ($child in [DpdfnetReaperDialog]::Descendants($preferences)) {
        if ([DpdfnetReaperDialog]::GetDlgCtrlID($child) -ne 1000) { continue }
        $count = [int][DpdfnetReaperDialog]::SendMessage($child, 0x0146, [IntPtr]::Zero, [IntPtr]::Zero)
        for ($index = 0; $index -lt $count; $index++) {
          if ([DpdfnetReaperDialog]::ComboItem($child, $index) -eq "Dummy Audio") {
            [void][DpdfnetReaperDialog]::SendMessage($child, 0x014E, [IntPtr]$index, [IntPtr]::Zero)
            $parent = [DpdfnetReaperDialog]::GetParent($child)
            [void][DpdfnetReaperDialog]::SendMessage($parent, 0x0111, [IntPtr](1000 -bor (1 -shl 16)), $child)
            $selected = $true
            break
          }
        }
      }
      if ($selected) {
        Start-Sleep -Milliseconds 500
        $apply = [DpdfnetReaperDialog]::GetDlgItem($preferences, 1144)
        if ($apply -ne [IntPtr]::Zero) {
          [void][DpdfnetReaperDialog]::SendMessage($apply, 0x00F5, [IntPtr]::Zero, [IntPtr]::Zero)
        }
        $ok = [DpdfnetReaperDialog]::GetDlgItem($preferences, 1)
        [void][DpdfnetReaperDialog]::SendMessage($ok, 0x00F5, [IntPtr]::Zero, [IntPtr]::Zero)
        $preferencesConfigured = $true
      }
    }
    if (Test-Path -LiteralPath $result) {
      $lines = Get-Content -LiteralPath $result
      if ($lines -contains "complete`t0") { break }
    }
    if ($process.HasExited) { throw "REAPER exited before completing the DPDFNet run" }
  } while ((Get-Date) -lt $deadline)
  if (-not (Test-Path -LiteralPath $result) -or
    -not ((Get-Content -LiteralPath $result) -contains "complete`t0")) {
    throw "REAPER did not complete the DPDFNet run before the deadline"
  }
  $hostDeadline = (Get-Date).AddSeconds(15)
  while (-not (Test-Path -LiteralPath $hostRun) -and (Get-Date) -lt $hostDeadline) {
    Start-Sleep -Milliseconds 250
  }
  if (-not (Test-Path -LiteralPath $hostRun)) {
    throw "the CLAP instance did not publish graceful host evidence"
  }
  $process.Refresh()
  $wallSeconds = ((Get-Date) - $startedAt).TotalSeconds
  $cpuSeconds = $process.TotalProcessorTime.TotalSeconds - $cpuBefore
  $processDocument = [ordered]@{
    schema = "denoize-dpdfnet-reaper-process-v1"
    schema_version = 1
    wall_seconds = $wallSeconds
    cpu_seconds = $cpuSeconds
    peak_working_set_bytes = [int64]$process.PeakWorkingSet64
    logical_processors = [Environment]::ProcessorCount
  }
  $processDocument | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath $processMetrics -Encoding utf8NoBOM
} finally {
  if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
}

$lines = Get-Content -LiteralPath $result
$parameters = @($lines | Where-Object { $_ -match '^parameter\t' })
$osara = @($lines | Where-Object { $_ -match '^osara\t' })
$bypassLatch = @($lines | Where-Object { $_ -match '^bypass-latch\t' })
$names = @($parameters | ForEach-Object { ($_ -split "`t")[2] })
$expectedNames = @("Bypass", "Mix", "Output Gain", "Overload Fallback")
if ($parameters.Count -ne 4 -or $osara.Count -ne 4 -or
  $bypassLatch.Count -ne 4 -or
  @(Compare-Object $expectedNames $names).Count -ne 0 -or
  $lines -notcontains "performance`tno-anticipative-fx`ttrue" -or
  $lines -notcontains "transport`trepeat`tfalse" -or
  $lines -notcontains "bypass-latch-summary`t0" -or
  $lines -notcontains "removed`ttrue" -or $lines -notcontains "complete`t0") {
  throw "REAPER/OSARA-style parameter evidence did not pass"
}
$expectedBypassStages = @(
  @{ Name = "on-1"; Target = 1.0; Applied = "true" },
  @{ Name = "off"; Target = 0.0; Applied = "true" },
  @{ Name = "on-2"; Target = 1.0; Applied = "true" },
  @{ Name = "on-2-held"; Target = 1.0; Applied = "false" }
)
for ($index = 0; $index -lt $expectedBypassStages.Count; $index++) {
  $fields = $bypassLatch[$index] -split "`t"
  $expected = $expectedBypassStages[$index]
  if ($fields.Count -ne 10 -or $fields[1] -ne $expected.Name -or
    [Math]::Abs([double]$fields[2] - $expected.Target) -gt 1e-9 -or
    $fields[3] -ne $expected.Applied -or $fields[4] -ne "true" -or
    [Math]::Abs([double]$fields[5] - $expected.Target) -gt 1e-9 -or
    $fields[6] -ne "true" -or $fields[8] -ne "false" -or
    (([int]$fields[9] -band 1) -ne 1)) {
    throw "REAPER did not retain the repeated Bypass sequence"
  }
}
$hostEvidence = Get-Content -LiteralPath $hostRun -Raw | ConvertFrom-Json
if ($hostEvidence.schema -ne "denoize-dpdfnet-clap-host-run-v2" -or
  $hostEvidence.schema_version -ne 2 -or
  $hostEvidence.model_id -ne "dpdfnet2-48khz-hr" -or
  $hostEvidence.source_commit -ne $SourceCommit -or
  -not $hostEvidence.worker_started -or -not $hostEvidence.finished_gracefully -or
  $hostEvidence.active_seconds -lt $measurementDelaySeconds -or
  $hostEvidence.measurement.warmup_frames -ne $evidenceWarmupFrames -or
  $hostEvidence.measurement.measured_frames -lt ($DurationSeconds * $sampleRate) -or
  $hostEvidence.measurement.measured_frames -ne ($hostEvidence.processed_frames - $hostEvidence.measurement.warmup_frames) -or
  $hostEvidence.host_audio_configuration.min_frames_count -lt 1 -or
  $hostEvidence.host_audio_configuration.max_frames_count -lt
    $hostEvidence.host_audio_configuration.min_frames_count -or
  $hostEvidence.callback_frames.calls -lt 1 -or
  $hostEvidence.callback_frames.minimum -lt
    $hostEvidence.host_audio_configuration.min_frames_count -or
  $hostEvidence.callback_frames.maximum -lt $hostEvidence.callback_frames.minimum -or
  $hostEvidence.callback_frames.maximum -gt
    $hostEvidence.host_audio_configuration.max_frames_count -or
  $hostEvidence.metrics.overload_blocks -ne 0 -or $hostEvidence.metrics.late_blocks -ne 0 -or
  $hostEvidence.metrics.invalid_blocks -ne 0 -or $hostEvidence.metrics.worker_errors -ne 0 -or
  $hostEvidence.lifetime_metrics.worker_errors -ne 0) {
  Write-Host "Rejected real REAPER host evidence:"
  Write-Host ($hostEvidence | ConvertTo-Json -Depth 10)
  throw "real REAPER DPDFNet worker evidence did not pass"
}
$processResult = Get-Content -LiteralPath $processMetrics -Raw | ConvertFrom-Json
$fileRecord = {
  param([string]$Path)
  $item = Get-Item -LiteralPath $Path
  [ordered]@{
    name = $item.Name
    size_bytes = [int64]$item.Length
    sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}
$evidence = [ordered]@{
  schema = "denoize-dpdfnet-reaper-automated-evidence-v1"
  schema_version = 1
  source_commit = $SourceCommit
  model_id = "dpdfnet2-48khz-hr"
  model_sha256 = "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b"
  host = [ordered]@{
    name = "REAPER"
    version = "7.79"
    operating_system = "windows"
    sample_rate_hz = 48000
    buffer_frames = 480
    active_seconds = [double]$hostEvidence.active_seconds
  }
  accessibility_api = [ordered]@{
    osara_style_parameter_path = $true
    parameters_readable_and_adjustable = 4
    nvda_human_verified = $false
  }
  measurement = [ordered]@{
    warmup_frames = [int64]$hostEvidence.measurement.warmup_frames
    measured_frames = [int64]$hostEvidence.measurement.measured_frames
  }
  worker_metrics = $hostEvidence.metrics
  lifetime_worker_metrics = $hostEvidence.lifetime_metrics
  process = [ordered]@{
    wall_seconds = [double]$processResult.wall_seconds
    cpu_seconds = [double]$processResult.cpu_seconds
    peak_working_set_bytes = [int64]$processResult.peak_working_set_bytes
    logical_processors = [int]$processResult.logical_processors
  }
  inputs = [ordered]@{
    parameters = & $fileRecord $result
    clap_host_run = & $fileRecord $hostRun
    process_metrics = & $fileRecord $processMetrics
  }
  accepted_automated = $true
}
$evidence | ConvertTo-Json -Depth 10 |
  Set-Content -LiteralPath $evidencePath -Encoding utf8NoBOM
Write-Host "REAPER automated evidence: $evidencePath"
