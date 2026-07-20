//! Cross-container audio metadata preservation.

use std::path::Path;

use lofty::config::WriteOptions;
use lofty::file::{AudioFile, TaggedFileExt};
use lofty::tag::Tag;

pub fn read(input: &Path) -> Result<Option<Tag>, String> {
    let source = lofty::read_from_path(input)
        .map_err(|error| format!("read metadata from {}: {error}", input.display()))?;
    Ok(source.primary_tag().or_else(|| source.first_tag()).cloned())
}

pub fn write(mut tag: Tag, output: &Path) -> Result<(), String> {
    let mut destination = lofty::read_from_path(output)
        .map_err(|error| format!("read output metadata from {}: {error}", output.display()))?;
    let target_type = destination.primary_tag_type();
    if tag.tag_type() != target_type {
        tag.re_map(target_type);
    }
    destination.insert_tag(tag);
    destination
        .save_to_path(output, WriteOptions::default())
        .map_err(|error| format!("write metadata to {}: {error}", output.display()))
}

/// Copy the primary input tag to the output's native tag type.
/// Returns `false` when the input has no metadata.
pub fn copy(input: &Path, output: &Path) -> Result<bool, String> {
    let Some(tag) = read(input)? else {
        return Ok(false);
    };
    write(tag, output)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lofty::tag::{Accessor, Tag, TagExt, TagType};

    #[test]
    fn copies_wav_metadata() {
        let root = std::env::temp_dir().join(format!("denoize-metadata-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let input = root.join("input.wav");
        let output = root.join("output.wav");
        let audio = crate::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 1_600]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        crate::write_wav(&input, &audio).unwrap();
        crate::write_wav(&output, &audio).unwrap();
        let mut tag = Tag::new(TagType::RiffInfo);
        tag.set_title("Metadata survives".into());
        tag.set_artist("denoize".into());
        tag.save_to_path(&input, WriteOptions::default()).unwrap();

        assert!(copy(&input, &output).unwrap());
        let tagged = lofty::read_from_path(&output).unwrap();
        let copied = tagged.primary_tag().unwrap();
        assert_eq!(copied.title().as_deref(), Some("Metadata survives"));
        assert_eq!(copied.artist().as_deref(), Some("denoize"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
