//! [`DatasetReader`] — opens a LeRobotDataset v3.0 from disk, resolving
//! episodes purely from the `meta/episodes` offset columns and honoring the
//! per-dataset `info.json` path template. Reads both this crate's output and
//! datasets produced by lerobot's own tooling (the v2.1→v3.0 converter emits
//! `List<Float32>` vector columns where we write `FixedSizeList<Float32>`;
//! both layouts are handled).

use crate::meta::{Info, format_chunk_file_path};
use crate::{CODEBASE_VERSION, Error};
use arrow::array::{
    Array, FixedSizeListArray, Float32Array, Float64Array, Int64Array, LargeListArray,
    LargeStringArray, ListArray, StringArray,
};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// One row of `meta/episodes` (the offset columns; per-episode stats columns
/// are present on disk but not surfaced here).
#[derive(Clone, Debug)]
pub struct EpisodeMeta {
    pub episode_index: i64,
    pub tasks: Vec<String>,
    pub length: u64,
    pub data_chunk_index: u64,
    pub data_file_index: u64,
    /// Global frame index of this episode's first frame.
    pub dataset_from_index: i64,
    /// Global frame index one past this episode's last frame.
    pub dataset_to_index: i64,
}

/// All frames of one episode, column-major.
#[derive(Clone, Debug)]
pub struct EpisodeData {
    pub episode_index: i64,
    pub tasks: Vec<String>,
    pub timestamps: Vec<f64>,
    pub frame_indices: Vec<i64>,
    pub global_indices: Vec<i64>,
    pub task_indices: Vec<i64>,
    /// `float32` vector features (e.g. `observation.state`, `action`), one
    /// `Vec<f32>` per frame, keyed by feature name.
    pub features: BTreeMap<String, Vec<Vec<f32>>>,
}

impl EpisodeData {
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }
}

/// LeRobotDataset v3.0 reader. See the [module docs](self).
pub struct DatasetReader {
    root: PathBuf,
    info: Info,
    episodes: Vec<EpisodeMeta>,
    tasks: Vec<String>,
}

impl DatasetReader {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, Error> {
        let root = root.as_ref().to_path_buf();
        let info_path = root.join("meta/info.json");
        let raw = std::fs::read_to_string(&info_path)
            .map_err(|e| Error::Format(format!("cannot read {}: {e}", info_path.display())))?;
        // Version-gate FIRST, on a lenient parse: a v2.x dataset is missing
        // v3.0-only fields, and the typed deserialization would report those
        // (e.g. `missing field data_files_size_in_mb`) instead of the real
        // story — the dataset is an older format generation.
        let version = serde_json::from_str::<serde_json::Value>(&raw)
            .ok()
            .and_then(|v| {
                v.get("codebase_version")
                    .and_then(|c| c.as_str().map(String::from))
            });
        if let Some(v) = &version
            && !v.starts_with("v3.")
        {
            return Err(Error::Format(format!(
                "codebase_version '{v}' is not {CODEBASE_VERSION} (convert v2.x datasets with \
                 lerobot's converter first)"
            )));
        }
        let info: Info = serde_json::from_str(&raw)?;
        if !info.codebase_version.starts_with("v3.") {
            return Err(Error::Format(format!(
                "codebase_version '{}' is not {CODEBASE_VERSION} (convert v2.x datasets with \
                 lerobot's converter first)",
                info.codebase_version
            )));
        }
        let mut episodes = load_episodes(&root)?;
        episodes.sort_by_key(|e| e.episode_index);
        let tasks = load_tasks(&root)?;
        Ok(Self {
            root,
            info,
            episodes,
            tasks,
        })
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    pub fn fps(&self) -> u32 {
        self.info.fps
    }

    pub fn total_episodes(&self) -> usize {
        self.episodes.len()
    }

    /// Episode metadata rows, sorted by `episode_index`.
    pub fn episodes(&self) -> &[EpisodeMeta] {
        &self.episodes
    }

    /// Task strings ordered by `task_index`.
    pub fn tasks(&self) -> &[String] {
        &self.tasks
    }

    /// Read every frame of episode `idx` (position in [`episodes`](Self::episodes)).
    ///
    /// The episode's rows are located purely from the `meta/episodes` offsets:
    /// its data file comes from `data/chunk_index` + `data/file_index` via the
    /// dataset's own `data_path` template, and the row range inside that file
    /// is `dataset_from_index..dataset_to_index` rebased to the file's first
    /// frame (the minimum `dataset_from_index` among episodes in the file).
    pub fn read_episode(&self, idx: usize) -> Result<EpisodeData, Error> {
        let ep = self
            .episodes
            .get(idx)
            .ok_or_else(|| Error::Format(format!("episode {idx} of {}", self.episodes.len())))?;
        let file_start = self
            .episodes
            .iter()
            .filter(|e| {
                e.data_chunk_index == ep.data_chunk_index && e.data_file_index == ep.data_file_index
            })
            .map(|e| e.dataset_from_index)
            .min()
            .expect("episode itself always matches");
        let start = usize::try_from(ep.dataset_from_index - file_start)
            .map_err(|_| Error::Format("negative episode offset".into()))?;
        let end = usize::try_from(ep.dataset_to_index - file_start)
            .map_err(|_| Error::Format("negative episode offset".into()))?;

        let rel = format_chunk_file_path(
            &self.info.data_path,
            ep.data_chunk_index,
            ep.data_file_index,
        )?;
        let path = self.root.join(rel);
        let file = File::open(&path)
            .map_err(|e| Error::Format(format!("cannot open {}: {e}", path.display())))?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        let mut out = EpisodeData {
            episode_index: ep.episode_index,
            tasks: ep.tasks.clone(),
            timestamps: Vec::new(),
            frame_indices: Vec::new(),
            global_indices: Vec::new(),
            task_indices: Vec::new(),
            features: BTreeMap::new(),
        };
        let vector_features: Vec<&String> = self
            .info
            .features
            .iter()
            .filter(|(name, f)| f.dtype == "float32" && name.as_str() != "timestamp")
            .map(|(name, _)| name)
            .collect();
        for name in &vector_features {
            out.features.insert((*name).clone(), Vec::new());
        }

        let mut row_offset = 0usize;
        for batch in reader {
            let batch = batch?;
            let rows = batch.num_rows();
            let lo = start.max(row_offset);
            let hi = end.min(row_offset + rows);
            if lo < hi {
                let slice = batch.slice(lo - row_offset, hi - lo);
                self.extract_rows(&slice, &vector_features, &mut out)?;
            }
            row_offset += rows;
            if row_offset >= end {
                break;
            }
        }
        if out.timestamps.len() != ep.length as usize {
            return Err(Error::Format(format!(
                "episode {}: expected {} frames from offsets, data file yielded {}",
                ep.episode_index,
                ep.length,
                out.timestamps.len()
            )));
        }
        Ok(out)
    }

    fn extract_rows(
        &self,
        batch: &RecordBatch,
        vector_features: &[&String],
        out: &mut EpisodeData,
    ) -> Result<(), Error> {
        extract_f64_scalars(column(batch, "timestamp")?, &mut out.timestamps)?;
        for (name, dst) in [
            ("frame_index", &mut out.frame_indices),
            ("index", &mut out.global_indices),
            ("task_index", &mut out.task_indices),
        ] {
            extract_i64_scalars(column(batch, name)?, name, dst)?;
        }
        // Consistency guard: every extracted row must belong to this episode.
        let mut ep_col = Vec::new();
        extract_i64_scalars(
            column(batch, "episode_index")?,
            "episode_index",
            &mut ep_col,
        )?;
        if ep_col.iter().any(|&e| e != out.episode_index) {
            return Err(Error::Format(format!(
                "rows resolved for episode {} carry a different episode_index — \
                 meta/episodes offsets disagree with the data file",
                out.episode_index
            )));
        }
        for name in vector_features {
            let arr = column(batch, name.as_str())?;
            let dst = out.features.get_mut(*name).expect("pre-inserted");
            extract_f32_rows(arr, name, dst)?;
        }
        Ok(())
    }
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a dyn Array, Error> {
    let idx = batch
        .schema()
        .index_of(name)
        .map_err(|_| Error::Format(format!("data file missing column '{name}'")))?;
    Ok(batch.column(idx).as_ref())
}

fn extract_f32_rows(arr: &dyn Array, name: &str, out: &mut Vec<Vec<f32>>) -> Result<(), Error> {
    let values_of = |v: arrow::array::ArrayRef| -> Result<Vec<f32>, Error> {
        if let Some(f) = v.as_any().downcast_ref::<Float32Array>() {
            Ok((0..f.len()).map(|j| f.value(j)).collect())
        } else if let Some(f) = v.as_any().downcast_ref::<Float64Array>() {
            Ok((0..f.len()).map(|j| f.value(j) as f32).collect())
        } else {
            Err(Error::Format(format!(
                "column '{name}': unsupported list element type"
            )))
        }
    };
    if let Some(a) = arr.as_any().downcast_ref::<FixedSizeListArray>() {
        for i in 0..a.len() {
            out.push(values_of(a.value(i))?);
        }
    } else if let Some(a) = arr.as_any().downcast_ref::<ListArray>() {
        for i in 0..a.len() {
            out.push(values_of(a.value(i))?);
        }
    } else if let Some(a) = arr.as_any().downcast_ref::<LargeListArray>() {
        for i in 0..a.len() {
            out.push(values_of(a.value(i))?);
        }
    } else if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
        // shape [1] scalar float feature stored as a plain column
        for i in 0..a.len() {
            out.push(vec![a.value(i)]);
        }
    } else {
        return Err(Error::Format(format!(
            "column '{name}': expected a float32 list column, got {:?}",
            arr.data_type()
        )));
    }
    Ok(())
}

fn extract_f64_scalars(arr: &dyn Array, out: &mut Vec<f64>) -> Result<(), Error> {
    if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
        out.extend((0..a.len()).map(|i| f64::from(a.value(i))));
    } else if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        out.extend((0..a.len()).map(|i| a.value(i)));
    } else {
        return Err(Error::Format(format!(
            "column 'timestamp': expected float, got {:?}",
            arr.data_type()
        )));
    }
    Ok(())
}

fn extract_i64_scalars(arr: &dyn Array, name: &str, out: &mut Vec<i64>) -> Result<(), Error> {
    let a = arr.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
        Error::Format(format!(
            "column '{name}': expected int64, got {:?}",
            arr.data_type()
        ))
    })?;
    out.extend((0..a.len()).map(|i| a.value(i)));
    Ok(())
}

fn extract_strings(arr: &dyn Array, i: usize) -> Result<Vec<String>, Error> {
    let list_values = |v: arrow::array::ArrayRef| -> Result<Vec<String>, Error> {
        if let Some(s) = v.as_any().downcast_ref::<StringArray>() {
            Ok((0..s.len()).map(|j| s.value(j).to_string()).collect())
        } else if let Some(s) = v.as_any().downcast_ref::<LargeStringArray>() {
            Ok((0..s.len()).map(|j| s.value(j).to_string()).collect())
        } else {
            Err(Error::Format("'tasks' list has non-string elements".into()))
        }
    };
    if let Some(a) = arr.as_any().downcast_ref::<ListArray>() {
        list_values(a.value(i))
    } else if let Some(a) = arr.as_any().downcast_ref::<LargeListArray>() {
        list_values(a.value(i))
    } else if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        Ok(vec![a.value(i).to_string()])
    } else if let Some(a) = arr.as_any().downcast_ref::<LargeStringArray>() {
        Ok(vec![a.value(i).to_string()])
    } else {
        Err(Error::Format(format!(
            "'tasks': unsupported type {:?}",
            arr.data_type()
        )))
    }
}

/// Read every `meta/episodes/*/*.parquet` (sorted, like lerobot's
/// `load_nested_dataset`) and pull the offset columns.
fn load_episodes(root: &Path) -> Result<Vec<EpisodeMeta>, Error> {
    let dir = root.join("meta/episodes");
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for chunk in sorted_dir(&dir)? {
        if chunk.is_dir() {
            for f in sorted_dir(&chunk)? {
                if f.extension().is_some_and(|e| e == "parquet") {
                    files.push(f);
                }
            }
        }
    }
    let mut episodes = Vec::new();
    for path in files {
        let file = File::open(&path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        for batch in reader {
            let batch = batch?;
            let get = |name: &str| -> Result<Vec<i64>, Error> {
                let idx = batch
                    .schema()
                    .index_of(name)
                    .map_err(|_| Error::Format(format!("meta/episodes missing column '{name}'")))?;
                let mut v = Vec::new();
                extract_i64_scalars(batch.column(idx).as_ref(), name, &mut v)?;
                Ok(v)
            };
            let episode_index = get("episode_index")?;
            let chunk_index = get("data/chunk_index")?;
            let file_index = get("data/file_index")?;
            let from_index = get("dataset_from_index")?;
            let to_index = get("dataset_to_index")?;
            let length = get("length")?;
            let tasks_idx = batch
                .schema()
                .index_of("tasks")
                .map_err(|_| Error::Format("meta/episodes missing column 'tasks'".into()))?;
            for i in 0..batch.num_rows() {
                episodes.push(EpisodeMeta {
                    episode_index: episode_index[i],
                    tasks: extract_strings(batch.column(tasks_idx).as_ref(), i)?,
                    length: u64::try_from(length[i])
                        .map_err(|_| Error::Format("negative episode length".into()))?,
                    data_chunk_index: u64::try_from(chunk_index[i])
                        .map_err(|_| Error::Format("negative chunk index".into()))?,
                    data_file_index: u64::try_from(file_index[i])
                        .map_err(|_| Error::Format("negative file index".into()))?,
                    dataset_from_index: from_index[i],
                    dataset_to_index: to_index[i],
                });
            }
        }
    }
    Ok(episodes)
}

/// Read `meta/tasks.parquet` → task strings ordered by `task_index`. The task
/// string lives in the pandas index column (usually `__index_level_0__`); we
/// take the first string-typed column that is not `task_index`.
fn load_tasks(root: &Path) -> Result<Vec<String>, Error> {
    let path = root.join("meta/tasks.parquet");
    let file = File::open(&path)
        .map_err(|e| Error::Format(format!("cannot open {}: {e}", path.display())))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut pairs: Vec<(i64, String)> = Vec::new();
    for batch in reader {
        let batch = batch?;
        let ti = batch
            .schema()
            .index_of("task_index")
            .map_err(|_| Error::Format("tasks.parquet missing 'task_index'".into()))?;
        let mut indices = Vec::new();
        extract_i64_scalars(batch.column(ti).as_ref(), "task_index", &mut indices)?;
        let string_col = (0..batch.num_columns())
            .find(|&c| {
                c != ti
                    && (batch.column(c).as_any().is::<StringArray>()
                        || batch.column(c).as_any().is::<LargeStringArray>())
            })
            .ok_or_else(|| Error::Format("tasks.parquet has no task string column".into()))?;
        for (i, &task_index) in indices.iter().enumerate() {
            let task = extract_strings(batch.column(string_col).as_ref(), i)?
                .pop()
                .unwrap_or_default();
            pairs.push((task_index, task));
        }
    }
    pairs.sort_by_key(|(i, _)| *i);
    Ok(pairs.into_iter().map(|(_, t)| t).collect())
}

fn sorted_dir(dir: &Path) -> Result<Vec<PathBuf>, Error> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|e| e.path())
        .collect();
    entries.sort();
    Ok(entries)
}
