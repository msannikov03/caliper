//! [`DatasetWriter`] — streams frames into the LeRobotDataset v3.0 layout.
//!
//! Lifecycle: [`DatasetWriter::create`] → [`add_frame`](DatasetWriter::add_frame)
//! (repeat) → [`save_episode`](DatasetWriter::save_episode) (repeat) →
//! [`finalize`](DatasetWriter::finalize). Episode frames are written to the
//! current `data/chunk-XXX/file-XXX.parquet` as one row group per episode
//! (exactly like lerobot's `pq.ParquetWriter.write_table` per episode), files
//! roll by the same predictive size rule lerobot uses, and episode metadata is
//! buffered and written to `meta/episodes/` on finalize. Dropping the writer
//! auto-finalizes, so a forgotten `finalize()` can never truncate the dataset.

use crate::meta::{
    DEFAULT_CHUNK_SIZE, DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_DATA_PATH, DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB, FeatureInfo, Info, format_chunk_file_path, next_chunk_file,
};
use crate::stats::{FeatureStats, aggregate_stats};
use crate::{CODEBASE_VERSION, Error};
use arrow::array::{
    Array, ArrayRef, FixedSizeListBuilder, Float32Array, Float32Builder, Float64Builder,
    Int64Array, Int64Builder, LargeStringArray, ListBuilder, StringBuilder,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Feature names lerobot adds implicitly to every dataset — the writer owns
/// these columns, so user features may not shadow them.
pub const RESERVED_FEATURES: [&str; 5] = [
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
];

/// One user data feature: a fixed-width `float32` vector per frame
/// (e.g. `observation.state` with one element per joint).
#[derive(Clone, Debug)]
pub struct FeatureSpec {
    pub name: String,
    pub dim: usize,
    /// Optional per-element names (joint names); `names: null` when absent.
    pub names: Option<Vec<String>>,
}

impl FeatureSpec {
    pub fn vector(name: impl Into<String>, dim: usize, names: Option<Vec<String>>) -> Self {
        Self {
            name: name.into(),
            dim,
            names,
        }
    }
}

/// Dataset-level configuration for [`DatasetWriter::create`].
#[derive(Clone, Debug)]
pub struct DatasetSpec {
    pub fps: u32,
    pub robot_type: String,
    pub features: Vec<FeatureSpec>,
    /// Data files roll to the next `file-XXX.parquet` once they would exceed
    /// this size (default 100, like lerobot). Fractional values are honored so
    /// tests can roll at tiny sizes; the value is recorded in `info.json`.
    pub data_files_size_in_mb: f64,
    /// Max files per `chunk-XXX` directory (default 1000, like lerobot).
    pub chunks_size: u64,
}

impl DatasetSpec {
    pub fn new(fps: u32, robot_type: impl Into<String>, features: Vec<FeatureSpec>) -> Self {
        Self {
            fps,
            robot_type: robot_type.into(),
            features,
            data_files_size_in_mb: DEFAULT_DATA_FILE_SIZE_IN_MB,
            chunks_size: DEFAULT_CHUNK_SIZE,
        }
    }
}

struct EpisodeRecord {
    episode_index: i64,
    task: String,
    length: u64,
    chunk_index: u64,
    file_index: u64,
    from_index: i64,
    to_index: i64,
    stats: BTreeMap<String, FeatureStats>,
}

struct OpenDataFile {
    writer: ArrowWriter<File>,
    chunk_index: u64,
    file_index: u64,
    frames: u64,
}

/// Streaming LeRobotDataset v3.0 writer. See the [module docs](self).
pub struct DatasetWriter {
    root: PathBuf,
    spec: DatasetSpec,
    /// Per-feature flattened `f32` buffer for the episode being recorded.
    buf: Vec<Vec<f32>>,
    buf_times: Vec<f64>,
    tasks: Vec<String>,
    episodes: Vec<EpisodeRecord>,
    global_index: i64,
    data: Option<OpenDataFile>,
    next_chunk: u64,
    next_file: u64,
    finalized: bool,
}

impl DatasetWriter {
    /// Create a fresh dataset at `root`. Fails if `root` already holds one.
    pub fn create(root: impl AsRef<Path>, spec: DatasetSpec) -> Result<Self, Error> {
        if spec.features.is_empty() {
            return Err(Error::State("dataset needs at least one feature".into()));
        }
        if spec.fps == 0 {
            return Err(Error::State("fps must be positive".into()));
        }
        if spec.chunks_size == 0 {
            return Err(Error::State("chunks_size must be positive".into()));
        }
        if !spec.data_files_size_in_mb.is_finite() || spec.data_files_size_in_mb <= 0.0 {
            return Err(Error::State(
                "data_files_size_in_mb must be positive".into(),
            ));
        }
        let mut seen: Vec<&str> = Vec::new();
        for f in &spec.features {
            if f.name.contains('/') {
                return Err(Error::State(format!(
                    "feature name '{}' may not contain '/'",
                    f.name
                )));
            }
            if RESERVED_FEATURES.contains(&f.name.as_str()) {
                return Err(Error::State(format!(
                    "feature name '{}' is reserved",
                    f.name
                )));
            }
            if f.dim == 0 {
                return Err(Error::State(format!(
                    "feature '{}' must have dim >= 1",
                    f.name
                )));
            }
            if let Some(names) = &f.names
                && names.len() != f.dim
            {
                return Err(Error::State(format!(
                    "feature '{}': {} names for dim {}",
                    f.name,
                    names.len(),
                    f.dim
                )));
            }
            if seen.contains(&f.name.as_str()) {
                return Err(Error::State(format!("duplicate feature '{}'", f.name)));
            }
            seen.push(&f.name);
        }
        let root = root.as_ref().to_path_buf();
        if root.join("meta/info.json").exists() {
            return Err(Error::State(format!(
                "refusing to overwrite existing dataset at {}",
                root.display()
            )));
        }
        fs::create_dir_all(root.join("meta"))?;
        fs::create_dir_all(root.join("data"))?;
        let n_features = spec.features.len();
        Ok(Self {
            root,
            spec,
            buf: vec![Vec::new(); n_features],
            buf_times: Vec::new(),
            tasks: Vec::new(),
            episodes: Vec::new(),
            global_index: 0,
            data: None,
            next_chunk: 0,
            next_file: 0,
            finalized: false,
        })
    }

    /// Append one frame with an auto timestamp of `frame_index / fps` seconds.
    /// `values` must name every declared feature exactly once.
    pub fn add_frame(&mut self, values: &[(&str, &[f64])]) -> Result<(), Error> {
        let t = self.buf_times.len() as f64 / f64::from(self.spec.fps);
        self.add_frame_at(values, t)
    }

    /// Append one frame with an explicit timestamp (seconds).
    pub fn add_frame_at(&mut self, values: &[(&str, &[f64])], timestamp: f64) -> Result<(), Error> {
        if self.finalized {
            return Err(Error::State("writer already finalized".into()));
        }
        if values.len() != self.spec.features.len() {
            return Err(Error::State(format!(
                "frame has {} features, dataset declares {}",
                values.len(),
                self.spec.features.len()
            )));
        }
        for (name, _) in values {
            if !self.spec.features.iter().any(|f| f.name == *name) {
                return Err(Error::State(format!("unknown feature '{name}'")));
            }
        }
        // Validate everything before mutating any buffer, so a failed frame
        // never leaves the per-feature buffers ragged.
        let mut ordered: Vec<&[f64]> = Vec::with_capacity(self.spec.features.len());
        for feat in &self.spec.features {
            let (_, v) = values
                .iter()
                .find(|(n, _)| *n == feat.name)
                .ok_or_else(|| Error::State(format!("missing feature '{}'", feat.name)))?;
            if v.len() != feat.dim {
                return Err(Error::Shape {
                    name: feat.name.clone(),
                    expected: feat.dim,
                    got: v.len(),
                });
            }
            ordered.push(v);
        }
        for (buf, v) in self.buf.iter_mut().zip(ordered) {
            buf.extend(v.iter().map(|&x| x as f32));
        }
        self.buf_times.push(timestamp);
        Ok(())
    }

    /// Close the buffered frames as one episode tagged with `task`, writing
    /// its row group to the current data file (rolling to a new file first if
    /// the size target would be exceeded).
    pub fn save_episode(&mut self, task: &str) -> Result<(), Error> {
        if self.finalized {
            return Err(Error::State("writer already finalized".into()));
        }
        let len = self.buf_times.len();
        if len == 0 {
            return Err(Error::State(
                "no frames buffered; call add_frame first".into(),
            ));
        }
        let episode_index = self.episodes.len() as i64;
        let task_index = self.intern_task(task);

        let batch = self.build_episode_batch(episode_index, task_index)?;
        self.roll_data_file_if_needed(len as u64)?;
        if self.data.is_none() {
            self.open_data_file(batch.schema())?;
        }
        let data = self.data.as_mut().expect("data file just opened");
        data.writer.write(&batch)?;
        // Flush so the on-disk size drives the next rolling decision (one row
        // group per episode, matching lerobot's per-episode write_table).
        data.writer.flush()?;
        data.frames += len as u64;
        let (chunk_index, file_index) = (data.chunk_index, data.file_index);

        let stats = self.episode_stats(episode_index, task_index);
        self.episodes.push(EpisodeRecord {
            episode_index,
            task: task.to_string(),
            length: len as u64,
            chunk_index,
            file_index,
            from_index: self.global_index,
            to_index: self.global_index + len as i64,
            stats,
        });
        self.global_index += len as i64;
        for buf in &mut self.buf {
            buf.clear();
        }
        self.buf_times.clear();
        Ok(())
    }

    /// Number of episodes saved so far.
    pub fn total_episodes(&self) -> usize {
        self.episodes.len()
    }

    /// Number of frames buffered in the episode currently being recorded.
    pub fn buffered_frames(&self) -> usize {
        self.buf_times.len()
    }

    /// Flush everything and write the `meta/` sidecars. Errors if frames were
    /// buffered but never saved via [`save_episode`](Self::save_episode) — the
    /// `Drop` guard, by contrast, silently discards such frames because drop
    /// cannot report.
    pub fn finalize(mut self) -> Result<PathBuf, Error> {
        if !self.buf_times.is_empty() {
            return Err(Error::State(format!(
                "{} buffered frames not saved; call save_episode(task) before finalize",
                self.buf_times.len()
            )));
        }
        self.finalize_impl()?;
        Ok(self.root.clone())
    }

    fn finalize_impl(&mut self) -> Result<(), Error> {
        if self.finalized {
            return Ok(());
        }
        // Mark first: a failed finalize must not run again from Drop.
        self.finalized = true;
        if let Some(data) = self.data.take() {
            data.writer.close()?;
        }
        self.write_episodes_parquet()?;
        self.write_tasks_parquet()?;
        let stats_maps: Vec<BTreeMap<String, FeatureStats>> =
            self.episodes.iter().map(|e| e.stats.clone()).collect();
        let aggregated = aggregate_stats(&stats_maps);
        fs::write(
            self.root.join("meta/stats.json"),
            serde_json::to_string_pretty(&aggregated)?,
        )?;
        fs::write(
            self.root.join("meta/info.json"),
            serde_json::to_string_pretty(&self.build_info())?,
        )?;
        Ok(())
    }

    // ---- internals ----

    fn intern_task(&mut self, task: &str) -> i64 {
        if let Some(i) = self.tasks.iter().position(|t| t == task) {
            return i as i64;
        }
        self.tasks.push(task.to_string());
        (self.tasks.len() - 1) as i64
    }

    fn build_episode_batch(
        &self,
        episode_index: i64,
        task_index: i64,
    ) -> Result<RecordBatch, Error> {
        let len = self.buf_times.len();
        let mut fields: Vec<Field> = Vec::new();
        let mut columns: Vec<ArrayRef> = Vec::new();
        for (feat, buf) in self.spec.features.iter().zip(&self.buf) {
            let mut b = FixedSizeListBuilder::new(Float32Builder::new(), feat.dim as i32);
            for row in buf.chunks_exact(feat.dim) {
                for &x in row {
                    b.values().append_value(x);
                }
                b.append(true);
            }
            let arr = b.finish();
            fields.push(Field::new(&feat.name, arr.data_type().clone(), true));
            columns.push(Arc::new(arr));
        }
        let timestamp =
            Float32Array::from(self.buf_times.iter().map(|&t| t as f32).collect::<Vec<_>>());
        let frame_index = Int64Array::from((0..len as i64).collect::<Vec<_>>());
        let episode = Int64Array::from(vec![episode_index; len]);
        let index = Int64Array::from(
            (self.global_index..self.global_index + len as i64).collect::<Vec<_>>(),
        );
        let task = Int64Array::from(vec![task_index; len]);
        for (name, arr) in [
            ("timestamp", Arc::new(timestamp) as ArrayRef),
            ("frame_index", Arc::new(frame_index) as ArrayRef),
            ("episode_index", Arc::new(episode) as ArrayRef),
            ("index", Arc::new(index) as ArrayRef),
            ("task_index", Arc::new(task) as ArrayRef),
        ] {
            fields.push(Field::new(name, arr.data_type().clone(), true));
            columns.push(arr);
        }
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?)
    }

    /// lerobot's predictive roll: before writing an episode of `incoming`
    /// frames, if `current_size + avg_size_per_frame * incoming` would reach
    /// the target, close the current file and advance the (chunk, file) index.
    fn roll_data_file_if_needed(&mut self, incoming: u64) -> Result<(), Error> {
        let Some(data) = &self.data else {
            return Ok(());
        };
        // Size from the writer's own accounting, NOT fs::metadata: parquet's
        // TrackedWrite buffers through a BufWriter, so the on-disk length lags
        // the logical size (at tiny test targets it reads 0 and never rolls).
        let size_mb = data.writer.bytes_written() as f64 / (1024.0 * 1024.0);
        let avg = if data.frames > 0 {
            size_mb / data.frames as f64
        } else {
            0.0
        };
        if size_mb + avg * incoming as f64 >= self.spec.data_files_size_in_mb {
            let data = self.data.take().expect("checked above");
            let (chunk_index, file_index) = (data.chunk_index, data.file_index);
            data.writer.close()?;
            let (c, f) = next_chunk_file(chunk_index, file_index, self.spec.chunks_size);
            self.next_chunk = c;
            self.next_file = f;
        }
        Ok(())
    }

    fn open_data_file(&mut self, schema: SchemaRef) -> Result<(), Error> {
        let rel = format_chunk_file_path(DEFAULT_DATA_PATH, self.next_chunk, self.next_file)?;
        let path = self.root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = File::create(&path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let writer = ArrowWriter::try_new(file, schema, Some(props))?;
        self.data = Some(OpenDataFile {
            writer,
            chunk_index: self.next_chunk,
            file_index: self.next_file,
            frames: 0,
        });
        Ok(())
    }

    /// Per-episode stats over the values **as stored** (after the f32 round),
    /// for every feature including the implicit scalar ones — exactly the
    /// entries lerobot's converter flattens into the episodes parquet.
    fn episode_stats(&self, episode_index: i64, task_index: i64) -> BTreeMap<String, FeatureStats> {
        let len = self.buf_times.len();
        let mut stats = BTreeMap::new();
        for (feat, buf) in self.spec.features.iter().zip(&self.buf) {
            let rows: Vec<Vec<f64>> = buf
                .chunks_exact(feat.dim)
                .map(|row| row.iter().map(|&x| f64::from(x)).collect())
                .collect();
            stats.insert(feat.name.clone(), FeatureStats::compute(&rows, feat.dim));
        }
        let scalar = |vals: Vec<f64>| -> FeatureStats {
            let rows: Vec<Vec<f64>> = vals.into_iter().map(|v| vec![v]).collect();
            FeatureStats::compute(&rows, 1)
        };
        stats.insert(
            "timestamp".into(),
            scalar(
                self.buf_times
                    .iter()
                    .map(|&t| f64::from(t as f32))
                    .collect(),
            ),
        );
        stats.insert(
            "frame_index".into(),
            scalar((0..len as i64).map(|i| i as f64).collect()),
        );
        stats.insert(
            "episode_index".into(),
            scalar(vec![episode_index as f64; len]),
        );
        stats.insert(
            "index".into(),
            scalar(
                (self.global_index..self.global_index + len as i64)
                    .map(|i| i as f64)
                    .collect(),
            ),
        );
        stats.insert("task_index".into(), scalar(vec![task_index as f64; len]));
        stats
    }

    /// `meta/episodes/chunk-000/file-000.parquet` — column set and order
    /// matching lerobot's converter output: offsets, `tasks`, flattened
    /// `stats/<feature>/<stat>` lists, then the metadata file indices.
    /// Like the converter's `write_episodes`, all rows go to one file.
    fn write_episodes_parquet(&self) -> Result<(), Error> {
        if self.episodes.is_empty() {
            return Ok(());
        }
        let mut fields: Vec<Field> = Vec::new();
        let mut columns: Vec<ArrayRef> = Vec::new();
        let push_i64 =
            |name: &str, vals: Vec<i64>, fields: &mut Vec<Field>, cols: &mut Vec<ArrayRef>| {
                fields.push(Field::new(name, DataType::Int64, true));
                cols.push(Arc::new(Int64Array::from(vals)) as ArrayRef);
            };
        push_i64(
            "episode_index",
            self.episodes.iter().map(|e| e.episode_index).collect(),
            &mut fields,
            &mut columns,
        );
        push_i64(
            "data/chunk_index",
            self.episodes.iter().map(|e| e.chunk_index as i64).collect(),
            &mut fields,
            &mut columns,
        );
        push_i64(
            "data/file_index",
            self.episodes.iter().map(|e| e.file_index as i64).collect(),
            &mut fields,
            &mut columns,
        );
        push_i64(
            "dataset_from_index",
            self.episodes.iter().map(|e| e.from_index).collect(),
            &mut fields,
            &mut columns,
        );
        push_i64(
            "dataset_to_index",
            self.episodes.iter().map(|e| e.to_index).collect(),
            &mut fields,
            &mut columns,
        );
        push_i64(
            "length",
            self.episodes.iter().map(|e| e.length as i64).collect(),
            &mut fields,
            &mut columns,
        );

        let mut tasks_b = ListBuilder::new(StringBuilder::new());
        for e in &self.episodes {
            tasks_b.values().append_value(&e.task);
            tasks_b.append(true);
        }
        let tasks_arr = tasks_b.finish();
        fields.push(Field::new("tasks", tasks_arr.data_type().clone(), true));
        columns.push(Arc::new(tasks_arr));

        // stats/<feature>/<stat>: features sorted (BTreeMap order), stats in
        // the converter's alphabetical count/max/mean/min/std order.
        for feat_name in self.episodes[0].stats.keys() {
            for stat in ["count", "max", "mean", "min", "std"] {
                let name = format!("stats/{feat_name}/{stat}");
                if stat == "count" {
                    let mut b = ListBuilder::new(Int64Builder::new());
                    for e in &self.episodes {
                        for &c in &e.stats[feat_name].count {
                            b.values().append_value(c as i64);
                        }
                        b.append(true);
                    }
                    let arr = b.finish();
                    fields.push(Field::new(&name, arr.data_type().clone(), true));
                    columns.push(Arc::new(arr));
                } else {
                    let mut b = ListBuilder::new(Float64Builder::new());
                    for e in &self.episodes {
                        let s = &e.stats[feat_name];
                        let vals = match stat {
                            "max" => &s.max,
                            "mean" => &s.mean,
                            "min" => &s.min,
                            _ => &s.std,
                        };
                        for &v in vals {
                            b.values().append_value(v);
                        }
                        b.append(true);
                    }
                    let arr = b.finish();
                    fields.push(Field::new(&name, arr.data_type().clone(), true));
                    columns.push(Arc::new(arr));
                }
            }
        }
        let n = self.episodes.len();
        push_i64(
            "meta/episodes/chunk_index",
            vec![0; n],
            &mut fields,
            &mut columns,
        );
        push_i64(
            "meta/episodes/file_index",
            vec![0; n],
            &mut fields,
            &mut columns,
        );

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
        let rel = format_chunk_file_path(DEFAULT_EPISODES_PATH, 0, 0)?;
        let path = self.root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = File::create(&path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    /// `meta/tasks.parquet`: `task_index` + the task string as a pandas index
    /// column (`__index_level_0__`), with the `pandas` schema metadata lerobot
    /// needs to restore `DataFrame.index` on `pd.read_parquet`.
    fn write_tasks_parquet(&self) -> Result<(), Error> {
        let task_index = Int64Array::from((0..self.tasks.len() as i64).collect::<Vec<_>>());
        let names = LargeStringArray::from(self.tasks.clone());
        let pandas_meta = serde_json::json!({
            "index_columns": ["__index_level_0__"],
            "column_indexes": [{
                "name": null, "field_name": null, "pandas_type": "unicode",
                "numpy_type": "str", "metadata": {"encoding": "UTF-8"},
            }],
            "columns": [
                {"name": "task_index", "field_name": "task_index", "pandas_type": "int64",
                 "numpy_type": "int64", "metadata": null},
                {"name": null, "field_name": "__index_level_0__", "pandas_type": "object",
                 "numpy_type": "str", "metadata": null},
            ],
            "attributes": {},
            "creator": {"library": "caliper-dataset", "version": env!("CARGO_PKG_VERSION")},
            "pandas_version": "2.0.0",
        });
        let metadata = HashMap::from([("pandas".to_string(), pandas_meta.to_string())]);
        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("task_index", DataType::Int64, true),
                Field::new("__index_level_0__", DataType::LargeUtf8, true),
            ],
            metadata,
        ));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(task_index) as ArrayRef,
                Arc::new(names) as ArrayRef,
            ],
        )?;
        let file = File::create(self.root.join("meta/tasks.parquet"))?;
        // pandas restores `DataFrame.index` from a top-level `pandas` entry in
        // the parquet footer key-value metadata. parquet-rs only embeds arrow
        // schema metadata inside the base64 `ARROW:schema` blob, so write the
        // footer entry explicitly.
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_key_value_metadata(Some(vec![KeyValue::new(
                "pandas".to_string(),
                pandas_meta.to_string(),
            )]))
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    fn build_info(&self) -> Info {
        let mut features = BTreeMap::new();
        for f in &self.spec.features {
            features.insert(
                f.name.clone(),
                FeatureInfo {
                    dtype: "float32".into(),
                    shape: vec![f.dim as u64],
                    names: match &f.names {
                        Some(n) => serde_json::json!(n),
                        None => serde_json::Value::Null,
                    },
                    fps: Some(self.spec.fps),
                },
            );
        }
        for (name, dtype) in [
            ("timestamp", "float32"),
            ("frame_index", "int64"),
            ("episode_index", "int64"),
            ("index", "int64"),
            ("task_index", "int64"),
        ] {
            features.insert(
                name.to_string(),
                FeatureInfo {
                    dtype: dtype.into(),
                    shape: vec![1],
                    names: serde_json::Value::Null,
                    fps: Some(self.spec.fps),
                },
            );
        }
        let mut splits = BTreeMap::new();
        if !self.episodes.is_empty() {
            splits.insert("train".to_string(), format!("0:{}", self.episodes.len()));
        }
        Info {
            codebase_version: CODEBASE_VERSION.into(),
            robot_type: Some(self.spec.robot_type.clone()),
            total_episodes: self.episodes.len() as u64,
            total_frames: self.global_index as u64,
            total_tasks: self.tasks.len() as u64,
            chunks_size: self.spec.chunks_size,
            data_files_size_in_mb: self.spec.data_files_size_in_mb,
            video_files_size_in_mb: DEFAULT_VIDEO_FILE_SIZE_IN_MB,
            fps: self.spec.fps,
            splits,
            data_path: DEFAULT_DATA_PATH.into(),
            video_path: None,
            features,
        }
    }
}

impl Drop for DatasetWriter {
    /// Auto-finalize: flush the parquet footer and write all buffered episode
    /// metadata, making the lerobot "forgot to finalize" footgun structurally
    /// impossible. Errors are unreportable in drop and are discarded; call
    /// [`finalize`](Self::finalize) explicitly to observe them.
    fn drop(&mut self) {
        let _ = self.finalize_impl();
    }
}
