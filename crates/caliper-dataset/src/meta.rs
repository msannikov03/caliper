//! `meta/info.json` model and the v3.0 path templates.

use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Default max data-parquet file size before rolling to the next file
/// (`lerobot.datasets.utils.DEFAULT_DATA_FILE_SIZE_IN_MB`).
pub const DEFAULT_DATA_FILE_SIZE_IN_MB: f64 = 100.0;
/// Default `video_files_size_in_mb` (`DEFAULT_VIDEO_FILE_SIZE_IN_MB`); recorded
/// in `info.json` even for video-less datasets, exactly like lerobot.
pub const DEFAULT_VIDEO_FILE_SIZE_IN_MB: f64 = 200.0;
/// Default max number of files per `chunk-XXX` directory
/// (`lerobot.datasets.utils.DEFAULT_CHUNK_SIZE`).
pub const DEFAULT_CHUNK_SIZE: u64 = 1000;
/// v3.0 data path template (`lerobot.datasets.utils.DEFAULT_DATA_PATH`).
pub const DEFAULT_DATA_PATH: &str = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet";
/// v3.0 episodes-metadata path template (`DEFAULT_EPISODES_PATH`).
pub const DEFAULT_EPISODES_PATH: &str =
    "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet";

/// One entry of `info.json`'s `features` map.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureInfo {
    pub dtype: String,
    pub shape: Vec<u64>,
    /// `null`, a list of element names, or (for cameras) a nested mapping —
    /// kept as raw JSON to stay lossless.
    #[serde(default)]
    pub names: serde_json::Value,
    /// lerobot's converter stamps every feature with the dataset fps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
}

/// `meta/info.json` — field set matches what lerobot 0.4.4 writes for v3.0
/// (NO v2.1 `total_chunks` / `total_videos`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Info {
    pub codebase_version: String,
    /// `null` is legal (lerobot's `robot_type: str | None`).
    pub robot_type: Option<String>,
    pub total_episodes: u64,
    pub total_frames: u64,
    pub total_tasks: u64,
    pub chunks_size: u64,
    /// Integral values serialize as JSON integers (matching lerobot);
    /// fractional targets are allowed so tests can roll files at tiny sizes.
    #[serde(with = "mb_size")]
    pub data_files_size_in_mb: f64,
    #[serde(with = "mb_size")]
    pub video_files_size_in_mb: f64,
    pub fps: u32,
    pub splits: BTreeMap<String, String>,
    pub data_path: String,
    pub video_path: Option<String>,
    pub features: BTreeMap<String, FeatureInfo>,
}

/// Serialize an `f64` megabyte size as a JSON integer when integral (lerobot
/// writes `100`, not `100.0`), while still accepting/producing fractions.
mod mb_size {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        if v.fract() == 0.0 && *v >= 0.0 && *v <= u64::MAX as f64 {
            s.serialize_u64(*v as u64)
        } else {
            s.serialize_f64(*v)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        f64::deserialize(d)
    }
}

/// Render a v3.0 path template — Python `str.format` restricted to the
/// `{chunk_index}` / `{file_index}` placeholders with an optional `:0Nd` spec,
/// which is all lerobot's templates use.
pub fn format_chunk_file_path(
    template: &str,
    chunk_index: u64,
    file_index: u64,
) -> Result<String, Error> {
    let mut out = String::with_capacity(template.len() + 8);
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let close = after.find('}').ok_or_else(|| {
            Error::Format(format!("unbalanced '{{' in path template '{template}'"))
        })?;
        let placeholder = &after[..close];
        let (name, spec) = match placeholder.split_once(':') {
            Some((n, s)) => (n, s),
            None => (placeholder, "d"),
        };
        let value = match name {
            "chunk_index" => chunk_index,
            "file_index" => file_index,
            other => {
                return Err(Error::Format(format!(
                    "unsupported placeholder '{{{other}}}' in path template '{template}'"
                )));
            }
        };
        let width = spec
            .strip_suffix('d')
            .and_then(|w| {
                if w.is_empty() {
                    Some(0)
                } else {
                    w.parse::<usize>().ok()
                }
            })
            .ok_or_else(|| {
                Error::Format(format!(
                    "unsupported format spec '{spec}' in path template '{template}'"
                ))
            })?;
        out.push_str(&format!("{value:0width$}"));
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// lerobot's `update_chunk_file_indices`: advance to the next file, wrapping
/// into the next chunk directory after `chunks_size` files.
pub fn next_chunk_file(chunk_index: u64, file_index: u64, chunks_size: u64) -> (u64, u64) {
    if file_index + 1 >= chunks_size {
        (chunk_index + 1, 0)
    } else {
        (chunk_index, file_index + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_default_templates() {
        assert_eq!(
            format_chunk_file_path(DEFAULT_DATA_PATH, 0, 0).unwrap(),
            "data/chunk-000/file-000.parquet"
        );
        assert_eq!(
            format_chunk_file_path(DEFAULT_DATA_PATH, 12, 345).unwrap(),
            "data/chunk-012/file-345.parquet"
        );
        assert_eq!(
            format_chunk_file_path(DEFAULT_EPISODES_PATH, 1, 2).unwrap(),
            "meta/episodes/chunk-001/file-002.parquet"
        );
    }

    #[test]
    fn formats_nonstandard_specs() {
        assert_eq!(
            format_chunk_file_path("x/{chunk_index}/{file_index:d}", 3, 4).unwrap(),
            "x/3/4"
        );
        assert!(format_chunk_file_path("x/{episode_index:03d}", 0, 0).is_err());
        assert!(format_chunk_file_path("x/{chunk_index:03f}", 0, 0).is_err());
        assert!(format_chunk_file_path("x/{chunk_index", 0, 0).is_err());
    }

    #[test]
    fn chunk_file_advance_wraps() {
        assert_eq!(next_chunk_file(0, 0, 1000), (0, 1));
        assert_eq!(next_chunk_file(0, 998, 1000), (0, 999));
        assert_eq!(next_chunk_file(0, 999, 1000), (1, 0));
        assert_eq!(next_chunk_file(2, 1, 2), (3, 0));
    }

    #[test]
    fn mb_size_serializes_integers_as_json_integers() {
        let mut info = Info {
            codebase_version: "v3.0".into(),
            robot_type: Some("t".into()),
            total_episodes: 0,
            total_frames: 0,
            total_tasks: 0,
            chunks_size: DEFAULT_CHUNK_SIZE,
            data_files_size_in_mb: DEFAULT_DATA_FILE_SIZE_IN_MB,
            video_files_size_in_mb: DEFAULT_VIDEO_FILE_SIZE_IN_MB,
            fps: 50,
            splits: BTreeMap::new(),
            data_path: DEFAULT_DATA_PATH.into(),
            video_path: None,
            features: BTreeMap::new(),
        };
        let s = serde_json::to_string(&info).unwrap();
        assert!(s.contains("\"data_files_size_in_mb\":100"), "{s}");
        assert!(!s.contains("100.0"), "{s}");
        info.data_files_size_in_mb = 0.25;
        let s = serde_json::to_string(&info).unwrap();
        assert!(s.contains("\"data_files_size_in_mb\":0.25"), "{s}");
        let back: Info = serde_json::from_str(&s).unwrap();
        assert_eq!(back.data_files_size_in_mb, 0.25);
        assert_eq!(back.video_files_size_in_mb, 200.0);
    }
}
