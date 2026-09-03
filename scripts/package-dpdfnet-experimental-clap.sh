#!/usr/bin/env bash

set -euo pipefail

if (( $# != 5 )); then
  echo "usage: $0 TARGET SOURCE_COMMIT OUTPUT_DIR PLUGIN_BINARY MODEL_FILE" >&2
  exit 2
fi

target=$1
source_commit=$2
output_dir=$3
binary=$4
model=$5

if [[ ! "$source_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commit must be a lowercase 40-character SHA-1" >&2
  exit 2
fi

case "$target" in
  aarch64-apple-darwin|x86_64-apple-darwin)
    platform=macos
    archive_extension=tar.gz
    ;;
  x86_64-unknown-linux-gnu)
    platform=linux
    archive_extension=tar.gz
    ;;
  x86_64-pc-windows-msvc)
    platform=windows
    archive_extension=zip
    ;;
  *)
    echo "unsupported experimental CLAP target: $target" >&2
    exit 2
    ;;
esac

if [[ ! -f "$binary" || -L "$binary" ]]; then
  echo "CLAP binary is not a regular file: $binary" >&2
  exit 1
fi
if [[ ! -f "$model" || -L "$model" ]]; then
  echo "DPDFNet model is not a regular file: $model" >&2
  exit 1
fi

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum < "$1" | awk '{print $1}'
  else
    shasum -a 256 < "$1" | awk '{print $1}'
  fi
}

expected_model_sha256=7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b
model_sha256=$(sha256_file "$model")
if [[ "$model_sha256" != "$expected_model_sha256" ]]; then
  echo "unexpected DPDFNet model digest: $model_sha256" >&2
  exit 1
fi

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=$(mkdir -p "$output_dir" && cd "$output_dir" && pwd)
short_commit=${source_commit:0:12}
package="denoize-dpdfnet-experimental-${short_commit}-${target}"
archive="$output_dir/$package.$archive_extension"
if [[ -e "$archive" || -L "$archive" ]]; then
  echo "refusing to replace existing experimental archive: $archive" >&2
  exit 1
fi

staging_root=$(mktemp -d "${TMPDIR:-/tmp}/denoize-dpdfnet-clap.XXXXXX")
cleanup() {
  find "$staging_root" -depth -delete
}
trap cleanup EXIT
package_dir="$staging_root/$package"
mkdir -p "$package_dir/models/dpdfnet2-48khz-hr"

version=$(python3 - "$repo_dir/Cargo.toml" 2>/dev/null <<'PY' || true
import pathlib
import sys
import tomllib

print(tomllib.loads(pathlib.Path(sys.argv[1]).read_text())["package"]["version"])
PY
)
if [[ -z "$version" ]]; then
  version=0.0.0
fi

if [[ "$platform" == macos ]]; then
  executable_dir="$package_dir/denoize.clap/Contents/MacOS"
  mkdir -p "$executable_dir"
  cp "$binary" "$executable_dir/denoize"
  chmod 755 "$executable_dir/denoize"
  cat > "$package_dir/denoize.clap/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>denoize</string>
  <key>CFBundleIdentifier</key><string>org.penguin425.denoize.experimental-dpdfnet</string>
  <key>CFBundleName</key><string>denoize DPDFNet experimental</string>
  <key>CFBundlePackageType</key><string>BNDL</string>
  <key>CFBundleShortVersionString</key><string>$version</string>
  <key>CFBundleVersion</key><string>$version</string>
</dict></plist>
EOF
else
  cp "$binary" "$package_dir/denoize.clap"
  [[ "$platform" == windows ]] || chmod 755 "$package_dir/denoize.clap"
fi

cp "$model" "$package_dir/models/dpdfnet2-48khz-hr/dpdfnet2_48khz_hr.onnx"
cp "$repo_dir/models/licenses/dpdfnet2-48khz-hr-Apache-2.0.txt" \
  "$package_dir/models/dpdfnet2-48khz-hr/"
cp "$repo_dir/models/provenance/dpdfnet2-48khz-hr.json" \
  "$package_dir/models/dpdfnet2-48khz-hr/"
cp "$repo_dir/docs/dpdfnet-experimental-clap.md" "$package_dir/EXPERIMENTAL.md"
cp "$repo_dir/docs/dpdfnet-reporter-template.md" "$package_dir/REPORTER.md"
cp "$repo_dir/README.md" "$repo_dir/LICENSE" "$repo_dir/THIRD_PARTY.md" "$package_dir/"
cp -R "$repo_dir/LICENSES" "$package_dir/"

plugin_sha256=$(sha256_file "$binary")
python3 - \
  "$package_dir/manifest.json" "$source_commit" "$target" "$version" \
  "$plugin_sha256" "$model_sha256" <<'PY'
import json
from pathlib import Path
import sys

output, commit, target, version, plugin_sha256, model_sha256 = sys.argv[1:]
document = {
    "schema": "denoize-dpdfnet-experimental-clap-package-v1",
    "schema_version": 1,
    "source_commit": commit,
    "target": target,
    "version": version,
    "scope": {
        "format": "clap",
        "descriptor_count": 3,
        "experimental_descriptor_id": "org.penguin425.denoize.neural-hq",
        "vst3_extended": False,
        "auv3_extended": False,
        "lv2_extended": False,
    },
    "plugin_sha256": plugin_sha256,
    "model": {
        "id": "dpdfnet2-48khz-hr",
        "filename": "models/dpdfnet2-48khz-hr/dpdfnet2_48khz_hr.onnx",
        "sha256": model_sha256,
    },
}
Path(output).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ "$archive_extension" == tar.gz ]]; then
  tar -C "$staging_root" -czf "$archive" "$package"
elif command -v 7z >/dev/null 2>&1; then
  (cd "$staging_root" && 7z a -tzip "$archive" "$package" >/dev/null)
elif command -v zip >/dev/null 2>&1; then
  (cd "$staging_root" && zip -qr "$archive" "$package")
elif command -v python3 >/dev/null 2>&1; then
  (cd "$staging_root" && python3 -m zipfile -c "$archive" "$package")
else
  echo "7z, zip, or Python is required to create the Windows archive" >&2
  exit 1
fi

if [[ ! -s "$archive" || -L "$archive" ]]; then
  echo "experimental CLAP archive was not created: $archive" >&2
  exit 1
fi
printf '%s\n' "$archive"
