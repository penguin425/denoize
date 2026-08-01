# Releasing denoize

One version tag publishes two distributions:

- GitHub Release archives with classical DSP, RNNoise, and DeepFilterNet.
- The crates.io CLI/library package with classical DSP and RNNoise.

DeepFilterNet 0.5.6 is currently a Git-only dependency. Cargo registries cannot
publish packages with an unreleased Git dependency, so it is intentionally
excluded from `Cargo.crates-io.toml`. Do not add it to that manifest until a
compatible DeepFilterNet release exists on crates.io.

## One-time repository setup

1. Create a crates.io account and an API token scoped to publishing `denoize`.
2. In the GitHub repository, create an Actions environment named `crates-io`.
3. Add the token as the environment secret `CRATES_IO_TOKEN`.
4. Optionally require approval on the `crates-io` environment.

The token is only exposed to the `publish-crate` job. GitHub release jobs use
the repository's built-in `GITHUB_TOKEN`.

## Release checklist

1. Update the root, crates.io, and desktop manifest versions and all lockfiles.
2. Regenerate the CLI reference and verify that it is committed:

   ```sh
   bash scripts/generate-cli-docs.sh
   bash scripts/generate-cli-docs.sh --check
   ```

3. Update release notes or user-facing documentation.
4. Run:

   ```sh
   cargo test --locked --all-targets --features full
   bash scripts/publish-crates-io.sh --dry-run
   ```

5. Commit and push the release change.
6. Tag that exact commit and push the tag:

   ```sh
   git tag -a v0.1.0 -m "denoize v0.1.0"
   git push origin v0.1.0
   ```

The workflow first validates and tests the tag, builds and uploads every OS
archive, publishes the crate, and verifies the complete release asset set. The
verification checks that every CLI archive, desktop installer, signature, and
`latest.json` is present and non-empty, validates the SHA-256 manifests, and
confirms that updater metadata points at uploaded assets. Only then is the
GitHub draft release published. If any step fails, the GitHub release remains a
draft.

The same release check can be run against an existing release with:

```sh
bash scripts/verify-release-assets.sh v0.5.7
```

Pull requests also run the desktop packaging matrix on Linux, macOS (Intel and
Apple Silicon), and Windows. Those CI builds disable signing and updater
artifact generation; the tagged release workflow enables both with the
repository's signing key.
