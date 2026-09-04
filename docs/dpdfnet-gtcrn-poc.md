# DPDFNet issue #221 evaluation

Measured on 2026-09-02 against `denoize` commit
`2ffa6849621c86e0a28abab8e95e8eef1fc0c02a`. The follow-up integration branch
now contains a production backend and an opt-in direct CLAP descriptor, but
DPDFNet is not in normal release plug-in builds and does not replace GTCRN.

## Decision

- **Go:** continue with DPDFNet-2 48 kHz HR as an explicitly selectable HQ
  plug-in prototype.
- **No-go:** do not integrate DPDFNet-8 in the tract CPU path. It missed every
  10 ms deadline while providing no material quality advantage.
- **No-go for now:** do not silently replace GTCRN or change the default. The
  DPDFNet worker/reset/state path now passes on this host, but Windows/REAPER,
  lower-tier cross-platform performance, and blinded listening remain open.
- Keep GTCRN as the compact/low-resource option. Its current released worker
  gate passed on this host, so the Windows overload report is not reproduced
  here and should be investigated on the reporter's machine.

This recommendation is stronger than the original three-case PoC: it now rests
on 280 paired noisy cases, ten source-preservation cases, five sample-rate
probes, a 70-case ViSQOL subset, sustained and concurrent hop timing, RSS,
real-model reset/geometry checks, ONNX Runtime numerical validation, and paced
release-profile worker gates for both GTCRN and the opt-in DPDFNet descriptor.

## Quality matrix

The primary matrix contains ten 48 kHz VCTK speakers, seven noise types, and
four SNRs (`-5`, `0`, `5`, and `15` dB): 280 paired conditions in total. Noise
types are two pinned DeepFilterNet/Freesound recordings plus deterministic
white, pink, 60 Hz hum, impulsive, and three-talker babble fixtures. Mixtures
were measured after PCM16 quantization; requested-SNR error was between
`-0.000324` and `+0.000174` dB.

DPDFNet's causal output was advanced by its four-hop/40 ms content delay. The
reference, noisy input, DPDFNet output, and GTCRN output were then cropped to
the same interval, avoiding both time-offset bias and zero-padded-tail bias.
Every case starts with new recurrent state.

Confidence intervals below use a deterministic speaker-cluster bootstrap with
20,000 resamples. Repeated noise/SNR conditions from one speaker are therefore
not treated as independent speakers.

| 280 noisy cases | DPDFNet-2 | GTCRN | Paired advantage for DPDFNet-2 | 95% CI | DPDFNet-2 wins |
|---|---:|---:|---:|---:|---:|
| SI-SDR improvement | **+9.878 dB** | +5.785 dB | **+4.093 dB** | +3.048 to +5.258 | 94.6% |
| STOI improvement | **+0.0467** | +0.0179 | **+0.0288** | +0.0199 to +0.0378 | 76.8% |
| Musical-noise screen (lower) | **0.0168** | 0.0196 | **0.00286 lower** | 0.00101 to 0.00532 | 60.4% |
| Pumping screen (lower) | 0.1077 | 0.1118 | 0.00414 lower | -0.00080 to +0.00983 | 58.2% |
| Transient-loss screen (lower) | **0.3232** | 0.5133 | **0.1902 lower** | 0.1066 to 0.2755 | 82.5% |

Pumping is directionally better but its confidence interval includes zero; it
must not be presented as a confirmed advantage. The artifact metrics are
deterministic screening signals, not listening-test scores.

The reference-aware ViSQOL audio-mode subset contains 60 noisy cases: all ten
speakers, both recorded noises and babble, at `-5` and `5` dB. It produced no
missing scores.

| 60-case ViSQOL subset | DPDFNet-2 | GTCRN | Paired advantage | 95% CI | DPDFNet-2 wins |
|---|---:|---:|---:|---:|---:|
| Enhanced MOS-LQO | **2.950** | 2.172 | **+0.778** | +0.567 to +1.032 | 98.3% |
| Improvement over noisy | **+0.363** | -0.415 | **+0.778** | +0.562 to +1.022 | 98.3% |

GTCRN's negative ViSQOL change is consistent with a 48→16→48 kHz round trip
being penalized by wideband audio mode even when speech intelligibility is
preserved. ViSQOL is consequently corroborating evidence, not a stand-alone
speech-quality verdict.

### Noise and SNR strata

DPDFNet-2's paired SI-SDR-improvement advantage over GTCRN stayed positive in
every aggregate stratum, but the babble lower bound is only just above zero.

| Noise | Paired advantage | 95% CI | DPDFNet-2 wins |
|---|---:|---:|---:|
| Freesound 2530 | **+3.899 dB** | +3.140 to +4.747 | 100.0% |
| Freesound 573577 | **+8.430 dB** | +5.250 to +12.038 | 100.0% |
| White | **+1.667 dB** | +0.789 to +2.594 | 90.0% |
| Pink | **+3.588 dB** | +2.928 to +4.301 | 100.0% |
| 60 Hz hum | **+4.543 dB** | +3.297 to +6.012 | 97.5% |
| Impulsive | **+4.661 dB** | +3.579 to +5.696 | 97.5% |
| Three-talker babble | +1.866 dB | +0.008 to +4.039 | 77.5% |

Nine of DPDFNet-2's fifteen SI-SDR losses were babble cases. The worst case was
`p228`, babble at `-5` dB: DPDFNet-2 trailed GTCRN by `7.542` dB. DPDFNet is a
general denoiser, not a target-speaker separator; competing speech is a real
boundary and prevents an unconditional replacement claim.

| Input SNR | Paired advantage | 95% CI | DPDFNet-2 wins |
|---|---:|---:|---:|
| -5 dB | **+4.417 dB** | +3.550 to +5.394 | 92.9% |
| 0 dB | **+4.195 dB** | +3.296 to +5.216 | 97.1% |
| 5 dB | **+3.967 dB** | +2.880 to +5.181 | 97.1% |
| 15 dB | **+3.795 dB** | +2.229 to +5.533 | 91.4% |

### DPDFNet-2 versus DPDFNet-8

The issue reporter's observation that versions 2 and 8 sound similar is
supported at aggregate level, while version 2 is slightly better on STOI.

| 280 noisy cases | DPDFNet-2 | DPDFNet-8 | Paired DPDFNet-2 difference | 95% CI |
|---|---:|---:|---:|---:|
| SI-SDR improvement | +9.878 dB | +9.941 dB | -0.062 dB | -0.186 to +0.060 |
| STOI improvement | **+0.0467** | +0.0369 | **+0.00986** | +0.00119 to +0.01885 |
| ViSQOL improvement (60 cases) | +0.3626 | +0.3658 | -0.00325 | -0.02094 to +0.01467 |

Their aligned waveforms had median cosine similarity `0.9990` (mean `0.9933`,
fifth percentile `0.9607`). That similarity does not justify version 8's much
higher compute cost.

### Source-preservation conflict

Ten unmodified VCTK source clips expose a metric conflict. These recordings
are corpus sources, not guaranteed anechoic/noiseless masters, so model changes
can be interpreted as either cleanup or unwanted modification.

| Source input | DPDFNet-2 | GTCRN | Paired DPDFNet-2 difference | 95% CI |
|---|---:|---:|---:|---:|
| SI-SDR against source | **18.751 dB** | 14.984 dB | **+3.767 dB** | +1.977 to +5.982 |
| ViSQOL MOS-LQO | **3.579** | 2.177 | **+1.402** | +1.064 to +1.799 |
| STOI | 0.8366 | **0.9006** | **-0.0640** | -0.1065 to -0.0280 |
| Pumping screen (lower) | 0.0895 | **0.0446** | -0.0449 | -0.0528 to -0.0342 |

DPDFNet preserves full-band waveform quality better by SI-SDR and audio-mode
ViSQOL, while GTCRN scores better on STOI and the pumping screen. A blinded
speech-focused listening test is therefore a required promotion gate rather
than an optional check.

## Paper and architecture context

The DPDFNet paper describes a causal, single-channel extension of
DeepFilterNet2 that adds dual-path blocks for long-range temporal and
cross-band modeling. It also adds an over-attenuation loss and always-on
fine-tuning. The paper reports gains on VoiceBank+DEMAND, DNS4, and a separate
multilingual low-SNR set; those claims motivate this evaluation but do not
substitute for denoize's own workload evidence.

Upstream reports `dpdfnet2_48khz_hr` at 2.58 M parameters and 2.42 GMAC, and
`dpdfnet8_48khz_hr` at 3.63 M parameters and 7.17 GMAC. GTCRN's maintained
figures are only 48.2 K parameters and 33 MMAC/s. DPDFNet-2 therefore has about
73 times GTCRN's published operation count, yet it ran slightly faster in this
tract benchmark. Larger dense kernels can use this CPU/runtime more efficiently
than many small operations; the result is an implementation observation, not a
portable refutation of the theoretical complexity gap.

DPDFNet's official streaming API retains RNN state, accepts arbitrary caller
chunks, emits its first result after one 20 ms window, and recommends a 480
sample/10 ms callback at 48 kHz. The production Rust adapter now wraps the
fixed-hop frontend with arbitrary-chunk buffering, exact-length resampling,
reset, finite-value sanitization, and explicit four-hop alignment.

## Realtime performance

CPU measurements used release/fat-LTO Rust 1.98.0, tract 0.23.4, and one AMD
Ryzen 9 3950X under WSL2 Linux x86-64. DPDFNet single-stream runs contain 120
seconds/12,000 measured hops after 100 warm-up calls; multi-stream runs contain
60 seconds per stream. All streams share one optimized graph and retain
independent recurrent/frontend state.

| Path | Streams | Mean | p99 | p99.9 | Maximum | Calls over budget | Aggregate throughput | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DPDFNet-2, 10 ms native hop | 1 | **4.204 ms** | **4.960** | **5.944** | 8.804 | **0/12,000** | 2.38x | 44.0 MiB |
| DPDFNet-2, 10 ms native hop | 4 | 4.319 ms | 5.244 | 6.808 | 7.867 | **0/24,000** | 9.24x | 50.7 MiB |
| DPDFNet-8, 10 ms native hop | 1 | 11.385 ms | 12.750 | 14.444 | 16.864 | **12,000/12,000** | 0.88x | 60.5 MiB |
| GTCRN, 16 ms native hop | 1 | 7.635 ms | 8.435 | 9.129 | 11.531 | 0/7,500 over 16 ms | 2.10x | 29.6 MiB |
| GTCRN, 48 kHz stereo-linked DAW block | 1 | 4.767 ms | 15.850 | 16.679 | 19.515 | 1,887/12,000 over 10 ms | 2.10x | 28.4 MiB |

DPDFNet-2 reproduces the issue reporter's ONNX Runtime result closely (their
mean `4.92` ms, maximum `7.77` ms, no 10 ms overrun). DPDFNet-8 does not: their
runtime saw occasional overruns, while tract exceeded 10 ms on effectively
every hop. Runtime-specific measurement is mandatory.

GTCRN DAW-call p50 was only `0.0048` ms because 48 kHz host blocks alternate
between resampler buffering and a 16 ms native inference. Its 15.7% of calls
above 10 ms does not by itself mean plug-in overload: summed RTF was `0.477`,
and the released plug-in has a separate 24-block/240 ms worker deadline.

The real ignored release gate
`neural::tests::pinned_gtcrn_release_worker_meets_sustained_deadlines` was run
with the authenticated model and 100 paced callback blocks. It passed with
zero overload, late, invalid, or worker-error assertions. The frequent Windows
overload reported in issue #221 is therefore not reproduced on this machine.

The opt-in gate
`neural::tests::pinned_dpdfnet2_release_worker_meets_sustained_deadlines` was
then run against the authenticated DPDFNet-2 model through the production CLAP
worker. Its 100 paced 48 kHz blocks also passed with zero overload, late,
invalid, stale-generation, or worker-error failures. This establishes the
Linux/WSL2 reference path only; it does not answer the reporter's Windows host
behavior.

## Model and adapter contract

- DPDFNet-2 model: official `dpdfnet2_48khz_hr.onnx`, revision
  `dd6818d00f50c836fed43a6243ebe49116de5964`, SHA-256
  `7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b`,
  10,493,337 bytes, 56,436-float state.
- DPDFNet-8 model: same revision, SHA-256
  `7b3afbb260a08fe9af3d16e3bda992971be1e7e951d1dee7c2d235f5c43f5631`,
  14,857,107 bytes, 90,228-float state.
- GTCRN model: revision `3862c44808dca492ea5a8a145d2dc2a1028d08c8`,
  SHA-256
  `b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87`,
  535,190 bytes.

The official DPDFNet-8 HR ONNX incorrectly carries the same
`profile=dpdfnet2_48khz_hr` metadata as DPDFNet-2. Production code must bind the
authenticated digest and state geometry; the profile string alone is unsafe.
The PoC accepts only the two observed official state geometries.

Both Rust adapters matched a one-thread ONNX Runtime 1.22.1 reference on the
same aligned fixture:

| Model | Maximum absolute error | RMS error | Correlation |
|---|---:|---:|---:|
| DPDFNet-2 | `1.5447e-5` | `8.3059e-6` | `0.9999999867` |
| DPDFNet-8 | `1.5330e-5` | `7.1150e-6` | `0.9999999901` |

Real-model stress checks found bit-exact independent streams, finite/exact
length output at 8, 16, 44.1, 48, and 96 kHz, and safe high-level handling of
NaN, infinities, and out-of-range input. Six metadata/window/geometry unit
tests pass. The existing GTCRN arbitrary-partition test also passes.

The production adapter accepts arbitrary host blocks and sample rates, produces
exactly the input presentation length, sanitizes non-finite samples, and
compensates the measured 40 ms content offset before the worker publishes a
result. Synthetic partition/reset tests and the real-model CLAP gate cover
reset, discontinuity generation, stale-result rejection, queue overload,
callback allocation, finite output, and fixed 240 ms latency reporting. The
fixed-hop model API remains available only for comparison measurements.

## Reproduction and artifacts

Run the complete acquisition, matrix, stress, ViSQOL, bootstrap, and listening
bundle workflow with:

```console
python3 scripts/run-dpdfnet-gtcrn-evaluation.py --stress-seconds 60
```

The runner downloads only pinned assets, verifies every byte count and SHA-256,
generates deterministic fixtures, and writes raw JSON plus a compact Markdown
summary under `/tmp/denoize-dpdfnet-gtcrn-evaluation` by default. The evaluated
fixture fingerprint was
`7cf6ee9a0e8767ae9c73e2e092a062358b2e4fe5ca08d9e4254cce8eecbb7691`.
Raw result digests from this run were:

- matrix JSON: `75964595877130c906260fe00fed005d0e1a8ffa4011007eee49f3f441710566`
- ViSQOL JSON: `fb4400ee4bd36f07177bbbd12f4f6499818cdd8ef1ff82792002ceb5fc437986`

The formal paired-preference bundle contains twelve core trials and four
hidden repeats across recorded noise, babble, source preservation, and
synthetic noise. A secret HMAC key randomizes opaque trial IDs, A/B assignment,
and trial order; the public protocol contains neither model identities nor the
answer key. The scorer applies the predeclared duplicate-consistency screens,
requires at least twenty retained listeners, and uses a 20,000-resample
listener-cluster bootstrap. Keep the randomization key and answer key private
until responses are frozen.

The VCTK mirror is pinned per file and the underlying corpus is CC BY 4.0.
DeepFilterNet's asset manifest identifies Freesound 2530 as CC BY 3.0 and
Freesound 573577 as CC0 1.0. Evaluation audio is downloaded to the work
directory and is not redistributed in this repository.

## Remaining promotion gates

The branch has completed the former first integration gate: DPDFNet-2 now has
its own `dpdfnet` backend, managed artifact/license/provenance, model ID,
`org.penguin425.denoize.neural-hq` state identity, arbitrary-block stream, and
opt-in `experimental-dpdfnet-hq` CLAP descriptor. Cross-profile state is
rejected and the standard two-descriptor release build is unchanged.

1. Have the issue reporter run signed experimental Windows builds in REAPER,
   including CLAP with NVDA/OSARA, and capture CPU, overload, late-block, and
   host-buffer settings.
2. Run a randomized blinded listening test, explicitly including the babble
   failures and source-preservation conflicts.
3. Repeat p99.9/deadline/RSS measurements on the lowest supported CPU tier and
   on macOS/Windows; do not generalize this Ryzen/WSL2 result.
4. Produce signed experimental CLAP artifacts and host evidence without adding
   the descriptor to VST3, AUv3, or LV2 release claims prematurely.
5. Keep GTCRN selectable until real plug-in evidence supports changing the
   default. DPDFNet-8 should remain excluded from tract CPU builds.

For this gate, the reproducible lowest supported tier is the GitHub-hosted
`ubuntu-slim` runner: x86-64 Linux, one logical CPU, and 5 GB RAM. Promotion
evidence rejects any `lowest-supported` record that is not generated on that
runner with exactly one available CPU. The workflow builds the executables on
the normal Linux runner, attests them, verifies those attestations offline on
`ubuntu-slim`, and performs only inference and evidence generation there to
stay inside its 15-minute job limit. See the
[GitHub-hosted runner specification](https://docs.github.com/en/actions/reference/runners/github-hosted-runners#supported-runners-and-hardware-resources).

Hosted runners are not real-time schedulers. Every promotion direct-call probe
therefore presents one 480-frame block every 10 ms instead of saturating its
runner continuously. Sleep time is excluded from each call measurement, and an
overrun is carried into the next scheduled call rather than resetting the
clock. The three `portable-ci` records keep the direct-call gate. Linux and
Windows apply it to monotonic wall time; macOS applies it to process-wide
`CLOCK_PROCESS_CPUTIME_ID`, with wall-time tails retained as diagnostics.
Windows still uses its `GetProcessTimes` kernel-plus-user total for aggregate
compute RTF, but not for the per-call distribution: although the API expresses
values in 100 ns units, the hosted runner produced 15.625 ms accounting steps.
Treating those quantized deltas as a tail distribution would turn normal calls
into alternating zero/15.625 ms samples. `QueryThreadCycleTime` is not used as
a substitute because Microsoft explicitly warns not to convert its cycle count
to elapsed time. Every eligible portable direct-call clock requires p99.9 at or
below 10 ms, at most 0.1% of calls above 10 ms, and no single call above 20 ms.

The one-CPU `ubuntu-slim` record retains the same p99.9, maximum, and 10 ms
miss count but marks `direct_call_deadline_gate_eligible=false`. That container
is the reproducible memory/parallelism floor, not a real-time scheduler, and
the released plug-in does not expose each synchronous model invocation as a
10 ms host deadline. Its public contract is the separately paced, buffered
worker below. Lowest-tier acceptance therefore requires aggregate RTF at or
below 1.0, the RSS bound, full finite/neural output, and zero inference or
unsafe-output errors. It marks `wall_clock_worker_gate_eligible=false`, so
overload/late counters caused by that shared container's scheduling remain
visible without being presented as a minimum-CPU compute failure. This
distinction was added after two exact-source one-CPU runs reported direct
p99.9 values of 28.97 and 21.45 ms at RTF 0.582 and 0.482, while a complete
6,000-block worker run had all four counters at zero. The portable three-OS
direct and worker thresholds are unchanged.

The production CLAP worker is independently paced for the full requested
measurement (6,000 blocks for the 60-second gate). On each `portable-ci`
platform it remains the strict wall-clock scheduling gate: overload, late,
invalid, and worker-error counters must all be zero, and process CPU timing
cannot hide a worker miss. The one-CPU capacity record runs the same production
worker and still rejects inference errors, unsafe output, incomplete finite
frames, or absence of neural output, but leaves overload/late as scheduling
diagnostics. The input and result queues each span all 24 scheduler chunks,
rather than ending after 16 chunks, so a recoverable pause cannot force a
recurrent-state discontinuity before the declared deadline.
Input fixtures are prepared before measurement, then whole 480-frame blocks are
submitted on one absolute 10 ms clock; processing time is carried into the next
deadline instead of added to a relative sleep. Evidence is rejected if the
measured wall time falls below 95% or exceeds 105% plus 250 ms of the complete
scheduled window (including latency priming), so a slow feeder cannot hide
production-worker deadline failures.

## Sources

- [Issue #221](https://github.com/penguin425/denoize/issues/221)
- [DPDFNet source and profiles](https://github.com/ceva-ip/DPDFNet/tree/1333776d470f01ecf4a533f098f4e8aeb3d00b89)
- [DPDFNet paper](https://arxiv.org/abs/2512.16420)
- [Official DPDFNet model revision](https://huggingface.co/Ceva-IP/DPDFNet/tree/dd6818d00f50c836fed43a6243ebe49116de5964)
- [GTCRN source and model](https://github.com/Xiaobin-Rong/gtcrn/tree/3862c44808dca492ea5a8a145d2dc2a1028d08c8)
- [GTCRN paper](https://ieeexplore.ieee.org/document/10448310)
- [VCTK corpus](https://datashare.ed.ac.uk/handle/10283/3443)
- [DeepFilterNet fixture licensing](https://github.com/Rikorose/DeepFilterNet/blob/d375b2d8309e0935d165700c91da9de862a99c31/assets/README.md)
- [Apple `clock_gettime` contract](https://github.com/apple-oss-distributions/Libc/blob/main/gen/clock_gettime.3)
- [Microsoft `GetProcessTimes` contract](https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes)
- [Microsoft `QueryThreadCycleTime` contract](https://learn.microsoft.com/windows/win32/api/realtimeapiset/nf-realtimeapiset-querythreadcycletime)
