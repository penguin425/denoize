//! Versioned external-model manifest and verified local cache.

use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
pub struct ModelInfo {
    pub name: &'static str,
    pub backend: &'static str,
    pub filename: &'static str,
    pub url: &'static str,
    pub revision: &'static str,
    pub sha256: &'static str,
    pub license: &'static str,
    pub sample_rate: u32,
}

pub const MODELS: &[ModelInfo] = &[ModelInfo {
    name: "gtcrn-dns3",
    backend: "gtcrn",
    filename: "gtcrn_simple.onnx",
    url: "https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/3862c44808dca492ea5a8a145d2dc2a1028d08c8/stream/onnx_models/gtcrn_simple.onnx",
    revision: "3862c44808dca492ea5a8a145d2dc2a1028d08c8",
    sha256: "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87",
    license: "MIT",
    sample_rate: 16_000,
}];

pub fn find(name: &str) -> Option<&'static ModelInfo> {
    MODELS
        .iter()
        .find(|model| model.name == name || model.backend == name)
}

pub fn cache_dir() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("DENOIZE_MODEL_DIR") {
        return Ok(PathBuf::from(path));
    }
    #[cfg(target_os = "windows")]
    if let Some(path) = std::env::var_os("LOCALAPPDATA") {
        return Ok(PathBuf::from(path).join("denoize").join("models"));
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(path).join("denoize").join("models"));
    }
    std::env::var_os("HOME")
        .map(|path| PathBuf::from(path).join(".cache/denoize/models"))
        .ok_or_else(|| "cannot locate model cache; set DENOIZE_MODEL_DIR".into())
}

pub fn path(model: &ModelInfo) -> Result<PathBuf, String> {
    Ok(cache_dir()?.join(model.name).join(model.filename))
}

pub fn verify(model: &ModelInfo) -> Result<PathBuf, String> {
    let path = path(model)?;
    if !path.is_file() {
        return Err(format!("model is not installed: {}", path.display()));
    }
    let actual = sha256(&path)?;
    if actual != model.sha256 {
        return Err(format!(
            "checksum mismatch for {}: expected {}, got {}",
            path.display(),
            model.sha256,
            actual
        ));
    }
    Ok(path)
}

pub fn install(model: &ModelInfo) -> Result<PathBuf, String> {
    if let Ok(path) = verify(model) {
        return Ok(path);
    }
    let destination = path(model)?;
    let parent = destination
        .parent()
        .ok_or_else(|| "invalid model cache path".to_string())?;
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    let partial = destination.with_extension("onnx.part");
    let downloaded = partial.metadata().map(|meta| meta.len()).unwrap_or(0);
    let mut request = ureq::get(model.url).set("User-Agent", "denoize-model-manager");
    if downloaded > 0 {
        request = request.set("Range", &format!("bytes={downloaded}-"));
    }
    let response = request
        .call()
        .map_err(|error| format!("failed to download {}: {error}", model.url))?;
    let resumed = downloaded > 0 && response.status() == 206;
    let mut output = OpenOptions::new()
        .create(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed)
        .open(&partial)
        .map_err(|error| format!("failed to open {}: {error}", partial.display()))?;
    let mut reader = response.into_reader();
    std::io::copy(&mut reader, &mut output)
        .map_err(|error| format!("failed to save {}: {error}", partial.display()))?;
    output
        .flush()
        .map_err(|error| format!("failed to flush {}: {error}", partial.display()))?;
    let actual = sha256(&partial)?;
    if actual != model.sha256 {
        return Err(format!(
            "downloaded model checksum mismatch: expected {}, got {} (partial kept at {})",
            model.sha256,
            actual,
            partial.display()
        ));
    }
    std::fs::rename(&partial, &destination).map_err(|error| {
        format!(
            "failed to move {} to {}: {error}",
            partial.display(),
            destination.display()
        )
    })?;
    Ok(destination)
}

fn sha256(path: &Path) -> Result<String, String> {
    let mut input =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = input
            .read(&mut buffer)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_has_pinned_integrity_and_metadata() {
        for model in MODELS {
            assert_eq!(model.sha256.len(), 64);
            assert_eq!(model.revision.len(), 40);
            assert!(model.url.contains(model.revision));
            assert!(model.sample_rate > 0);
            assert!(!model.license.is_empty());
        }
    }
}
