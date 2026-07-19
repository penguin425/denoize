#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

manifest_backup=$(mktemp)
lock_backup=$(mktemp)
cp Cargo.toml "$manifest_backup"
cp Cargo.lock "$lock_backup"
restore_files() {
  cp "$manifest_backup" Cargo.toml
  cp "$lock_backup" Cargo.lock
  rm -f "$manifest_backup" "$lock_backup"
}
trap restore_files EXIT

cp Cargo.crates-io.toml Cargo.toml
rm Cargo.lock
cargo generate-lockfile

if [[ "${1:-}" == "--test" ]]; then
  shift
  cargo test --locked --all-targets --features full "$@"
  exit 0
fi

cargo publish --locked --allow-dirty "$@"
