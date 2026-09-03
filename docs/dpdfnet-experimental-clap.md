# DPDFNet experimental CLAP

This package is for issue #221 testing. It is not a normal denoize release.

It adds one CLAP-only plug-in, `denoize Neural HQ`
(`org.penguin425.denoize.neural-hq`), backed by the pinned
`dpdfnet2-48khz-hr` model. VST3, AUv3, and LV2 are not expanded by this
experiment, and `denoize Neural` continues to use GTCRN.

## Run it

1. Verify the archive with `gh attestation verify ARCHIVE --repo penguin425/denoize`.
2. Extract the archive.
3. Set `DENOIZE_MODEL_DIR` to the extracted package's `models` directory. The
   bundled model is already under `models/dpdfnet2-48khz-hr/`.
4. Copy `denoize.clap` to a CLAP search directory or point the host's CLAP
   path at the extracted directory.
5. In REAPER, insert **CLAP: denoize Neural HQ** on a 48 kHz session.

Follow `REPORTER.md` from the package for the three five-minute runs and the
exact host-evidence capture procedure. It distinguishes REAPER's requested
buffer from the activation bounds and callback sizes observed by the plug-in,
and preserves failed measurements without changing their counters.

Do not publish the private answer key from a blinded listening test until all
responses have been collected and frozen.
