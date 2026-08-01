#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)

check=false
if [[ "${1:-}" == "--check" ]]; then
  check=true
  shift
fi

binary=${1:-"$repo_dir/target/debug/denoize"}
output=${2:-"$repo_dir/docs/cli.md"}

if [[ ! -x "$binary" ]]; then
  echo "CLI binary not found at $binary; building it first." >&2
  cargo build --locked --no-default-features --bin denoize \
    --manifest-path "$repo_dir/Cargo.toml"
fi

if [[ ! -x "$binary" ]]; then
  echo "CLI binary is not executable: $binary" >&2
  exit 1
fi

mkdir -p "$(dirname "$output")"

temporary_output=$(mktemp "${TMPDIR:-/tmp}/denoize-cli-docs.XXXXXX")
trap 'rm -f "$temporary_output"' EXIT
{
  echo '# denoize CLI reference'
  echo
  echo '```text'
  "$binary" --help
  echo '```'
} > "$temporary_output"

if [[ "$check" == true ]]; then
  diff -u "$output" "$temporary_output"
else
  mv "$temporary_output" "$output"
  trap - EXIT
fi
