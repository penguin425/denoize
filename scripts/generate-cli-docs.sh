#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
binary=${1:-"$repo_dir/target/debug/denoize"}
output=${2:-"$repo_dir/docs/cli.md"}

mkdir -p "$(dirname "$output")"
{
  echo '# denoize CLI reference'
  echo
  echo '```text'
  "$binary" --help
  echo '```'
} > "$output"
