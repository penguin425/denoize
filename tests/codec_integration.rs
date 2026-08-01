use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use denoize::{decode_file, metadata, read_audio, write_audio, write_wav, Audio, EncodeOptions};
use hound::SampleFormat;
use lofty::config::WriteOptions;
use lofty::tag::{Accessor, Tag, TagExt, TagType};

struct TestWorkspace {
    path: PathBuf,
}

impl TestWorkspace {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before Unix epoch")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("denoize-codec-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(&path).expect("create codec test workspace");
        Self { path }
    }

    fn file(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TestWorkspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn fixture(channels: usize, frames: usize) -> Audio {
    let sample_rate = 44_100;
    let channels = (0..channels)
        .map(|channel| {
            (0..frames)
                .map(|frame| {
                    let time = frame as f64 / sample_rate as f64;
                    let frequency = 220.0 + channel as f64 * 73.0;
                    let level = 0.18 + channel as f64 * 0.02;
                    (std::f64::consts::TAU * frequency * time).sin() * level
                })
                .collect()
        })
        .collect();
    Audio {
        sample_rate,
        channels,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    }
}

fn assert_duration(decoded: &denoize::decode::DecodedPcm, input: &Audio, codec: &str) {
    let input_seconds = input.frames() as f64 / input.sample_rate as f64;
    let output_seconds = decoded.frames() as f64 / decoded.sample_rate as f64;
    assert!(
        (output_seconds - input_seconds).abs() < 0.15,
        "{codec} duration changed from {input_seconds:.3}s to {output_seconds:.3}s"
    );
}

fn assert_tag(path: &Path) {
    let tag = metadata::read(path)
        .expect("read output metadata")
        .expect("output should contain a tag");
    assert_eq!(tag.title().as_deref(), Some("Integration fixture"));
    assert_eq!(tag.artist().as_deref(), Some("denoize tests"));
}

#[test]
fn wav_and_flac_preserve_multichannel_shape() {
    let workspace = TestWorkspace::new();
    let input = fixture(4, 44_100 / 2);
    let wav = workspace.file("surround.wav");
    let flac = workspace.file("surround.flac");

    write_wav(&wav, &input).expect("write multichannel WAV");
    let wav_audio = read_audio(&wav).expect("read multichannel WAV");
    assert_eq!(wav_audio.sample_rate, input.sample_rate);
    assert_eq!(wav_audio.channels(), 4);
    assert_eq!(wav_audio.frames(), input.frames());

    write_audio(&flac, &input, EncodeOptions::default()).expect("write multichannel FLAC");
    let decoded = decode_file(&flac).expect("decode multichannel FLAC");
    assert_eq!(decoded.sample_rate, input.sample_rate);
    assert_eq!(decoded.n_channels(), 4);
    assert_eq!(decoded.frames(), input.frames());
    for (expected, actual) in input.channels.iter().zip(&decoded.channels) {
        let max_error = expected
            .iter()
            .zip(actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0, f64::max);
        assert!(max_error < 2.0 / 32_768.0, "FLAC PCM error {max_error}");
    }
}

#[test]
fn stereo_lossy_codecs_preserve_channel_layout_and_duration() {
    let workspace = TestWorkspace::new();
    let input = fixture(2, 44_100 / 2);

    for (extension, codec) in [("opus", "Ogg Opus"), ("mp3", "MP3")] {
        let output = workspace.file(&format!("stereo.{extension}"));
        write_audio(&output, &input, EncodeOptions::default())
            .unwrap_or_else(|error| panic!("write {codec}: {error}"));
        let decoded =
            decode_file(&output).unwrap_or_else(|error| panic!("decode {codec}: {error}"));
        assert_eq!(decoded.n_channels(), 2, "{codec} channel count");
        assert_duration(&decoded, &input, codec);
    }
}

#[cfg(feature = "m4a-encode")]
#[test]
fn adts_aac_preserves_stereo_layout_and_duration() {
    let workspace = TestWorkspace::new();
    let input = fixture(2, 44_100 / 2);
    let output = workspace.file("stereo.aac");

    write_audio(&output, &input, EncodeOptions::default()).expect("write ADTS AAC");
    let decoded = decode_file(&output).expect("decode ADTS AAC");
    assert_eq!(decoded.n_channels(), 2);
    assert_duration(&decoded, &input, "ADTS AAC");
}

#[cfg(feature = "m4a-encode")]
#[test]
fn m4a_preserves_stereo_layout_and_duration() {
    let workspace = TestWorkspace::new();
    let input = fixture(2, 44_100 / 2);
    let output = workspace.file("stereo.m4a");

    write_audio(&output, &input, EncodeOptions::default()).expect("write M4A");
    let decoded = decode_file(&output).expect("decode M4A");
    assert_eq!(decoded.n_channels(), 2);
    assert_duration(&decoded, &input, "M4A");
}

#[test]
fn metadata_copies_across_wav_flac_and_mp3() {
    let workspace = TestWorkspace::new();
    let input = workspace.file("tagged.wav");
    let audio = fixture(2, 44_100 / 4);
    write_wav(&input, &audio).expect("write tagged input");

    let mut tag = Tag::new(TagType::RiffInfo);
    tag.set_title("Integration fixture".into());
    tag.set_artist("denoize tests".into());
    tag.save_to_path(&input, WriteOptions::default())
        .expect("write input metadata");

    for (extension, codec) in [("flac", "FLAC"), ("mp3", "MP3")] {
        let output = workspace.file(&format!("tagged.{extension}"));
        write_audio(&output, &audio, EncodeOptions::default())
            .unwrap_or_else(|error| panic!("write {codec}: {error}"));
        assert!(metadata::copy(&input, &output)
            .unwrap_or_else(|error| panic!("copy metadata to {codec}: {error}")));
        assert_tag(&output);
    }
}

#[cfg(feature = "m4a-encode")]
#[test]
fn metadata_copies_to_m4a() {
    let workspace = TestWorkspace::new();
    let input = workspace.file("tagged-m4a.wav");
    let output = workspace.file("tagged.m4a");
    let audio = fixture(2, 44_100 / 4);
    write_wav(&input, &audio).expect("write tagged input");

    let mut tag = Tag::new(TagType::RiffInfo);
    tag.set_title("Integration fixture".into());
    tag.set_artist("denoize tests".into());
    tag.save_to_path(&input, WriteOptions::default())
        .expect("write input metadata");

    write_audio(&output, &audio, EncodeOptions::default()).expect("write M4A");
    assert!(metadata::copy(&input, &output).expect("copy metadata to M4A"));
    assert_tag(&output);
}
