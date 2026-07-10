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
//!
//! Camera data: declare [`FeatureKind::Image`] features and record through
//! [`add_frame_with_images`](DatasetWriter::add_frame_with_images) with one
//! pre-encoded PNG per image feature per frame — encoding stays on the
//! producer side; the writer validates each frame by decoding (shape/8-bit),
//! stores the bytes verbatim in the `struct<bytes, path>` layout lerobot's
//! own writer embeds, and computes the per-channel `(c, 1, 1)` stats lerobot
//! expects in `meta/stats.json`.

use crate::meta::{
    DEFAULT_CHUNK_SIZE, DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_DATA_PATH, DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB, FeatureInfo, Info, format_chunk_file_path, next_chunk_file,
};
use crate::stats::{FeatureStats, aggregate_stats};
use crate::{CODEBASE_VERSION, Error};
use arrow::array::{
    Array, ArrayRef, BinaryBuilder, FixedSizeListBuilder, Float32Array, Float32Builder,
    Float64Builder, Int64Array, Int64Builder, LargeStringArray, ListBuilder, StringBuilder,
    StructArray,
};
use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef};
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

/// What one user data feature holds per frame.
#[derive(Clone, Debug)]
pub enum FeatureKind {
    /// A fixed-width `float32` vector (e.g. `observation.state`, one element
    /// per joint).
    Vector {
        dim: usize,
        /// Optional per-element names (joint names); `names: null` when absent.
        names: Option<Vec<String>>,
    },
    /// A camera frame stored as lerobot's `dtype: "image"`: the data parquet
    /// carries an Arrow `struct<bytes: binary, path: string>` column whose
    /// `bytes` is one complete PNG file per frame (HF `datasets.Image`
    /// storage). The writer takes PRE-ENCODED PNG bytes — encoding stays on
    /// the producer side — and validates each frame's decoded
    /// height/width/channels against this spec.
    Image {
        height: usize,
        width: usize,
        channels: usize,
    },
}

/// One user data feature. See [`FeatureKind`] for the per-frame payloads.
#[derive(Clone, Debug)]
pub struct FeatureSpec {
    pub name: String,
    pub kind: FeatureKind,
}

impl FeatureSpec {
    pub fn vector(name: impl Into<String>, dim: usize, names: Option<Vec<String>>) -> Self {
        Self {
            name: name.into(),
            kind: FeatureKind::Vector { dim, names },
        }
    }

    /// An image feature of `height` × `width` pixels with `channels` samples
    /// per pixel (3 = RGB, 1 = grayscale, 4 = RGBA — must match what the PNG
    /// frames decode to).
    pub fn image(name: impl Into<String>, height: usize, width: usize, channels: usize) -> Self {
        Self {
            name: name.into(),
            kind: FeatureKind::Image {
                height,
                width,
                channels,
            },
        }
    }

    pub fn is_image(&self) -> bool {
        matches!(self.kind, FeatureKind::Image { .. })
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
    /// Episode-level task strings (the `tasks` list column of `meta/episodes`).
    /// Always exactly one for `save_episode`; edit ops may merge episodes with
    /// different tasks, producing a multi-entry list (lerobot's own converter
    /// emits a list here too).
    tasks: Vec<String>,
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

/// Per-channel pixel statistics of ONE decoded frame, in lerobot's normalized
/// [0, 1] scale (`pixel / 255`). Folded per episode in `episode_stats`.
struct PixelStats {
    min: Vec<f64>,
    max: Vec<f64>,
    sum: Vec<f64>,
    sumsq: Vec<f64>,
    /// Pixels per frame (`height * width`) — constant per feature by
    /// construction, so folding across frames stays exact.
    pixels: u64,
}

/// One buffered image frame: the caller's PNG bytes verbatim + the channel
/// stats computed from the validating decode at `add_frame` time.
struct ImageFrame {
    png: Vec<u8>,
    stats: PixelStats,
}

/// Per-feature episode buffer, matching the [`FeatureKind`] of its spec.
enum FeatureBuf {
    /// Flattened `f32` rows (`dim` values per frame).
    Vector(Vec<f32>),
    /// One pre-encoded PNG per frame.
    Image(Vec<ImageFrame>),
}

/// Streaming LeRobotDataset v3.0 writer. See the [module docs](self).
pub struct DatasetWriter {
    root: PathBuf,
    spec: DatasetSpec,
    /// Per-feature buffer for the episode being recorded (parallel to
    /// `spec.features`).
    buf: Vec<FeatureBuf>,
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
            match &f.kind {
                FeatureKind::Vector { dim, names } => {
                    if *dim == 0 {
                        return Err(Error::State(format!(
                            "feature '{}' must have dim >= 1",
                            f.name
                        )));
                    }
                    if let Some(names) = names
                        && names.len() != *dim
                    {
                        return Err(Error::State(format!(
                            "feature '{}': {} names for dim {}",
                            f.name,
                            names.len(),
                            dim
                        )));
                    }
                }
                FeatureKind::Image {
                    height,
                    width,
                    channels,
                } => {
                    if *height == 0 || *width == 0 {
                        return Err(Error::State(format!(
                            "image feature '{}': height and width must be >= 1, got {height}x{width}",
                            f.name
                        )));
                    }
                    // PNG carries 1 (gray), 2 (gray+alpha), 3 (RGB) or 4
                    // (RGBA) samples per pixel; anything else can never match
                    // a decoded frame.
                    if !(1..=4).contains(channels) {
                        return Err(Error::State(format!(
                            "image feature '{}': channels must be 1..=4, got {channels}",
                            f.name
                        )));
                    }
                }
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
        let buf = spec
            .features
            .iter()
            .map(|f| match f.kind {
                FeatureKind::Vector { .. } => FeatureBuf::Vector(Vec::new()),
                FeatureKind::Image { .. } => FeatureBuf::Image(Vec::new()),
            })
            .collect();
        Ok(Self {
            root,
            spec,
            buf,
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
    /// `values` must name every declared vector feature exactly once; datasets
    /// with image features must use
    /// [`add_frame_with_images`](Self::add_frame_with_images).
    pub fn add_frame(&mut self, values: &[(&str, &[f64])]) -> Result<(), Error> {
        let t = self.buf_times.len() as f64 / f64::from(self.spec.fps);
        self.add_frame_at_with_images(values, &[], t)
    }

    /// Append one frame with an explicit timestamp (seconds). Vector-feature
    /// datasets only — see [`add_frame`](Self::add_frame).
    pub fn add_frame_at(&mut self, values: &[(&str, &[f64])], timestamp: f64) -> Result<(), Error> {
        self.add_frame_at_with_images(values, &[], timestamp)
    }

    /// Append one frame carrying camera data, with an auto timestamp of
    /// `frame_index / fps` seconds. `values` must name every declared vector
    /// feature exactly once and `images` every declared image feature exactly
    /// once; each image is one complete pre-encoded PNG file, validated by
    /// decoding (8-bit, height/width/channels must match the spec) — the
    /// bytes are stored verbatim.
    pub fn add_frame_with_images(
        &mut self,
        values: &[(&str, &[f64])],
        images: &[(&str, &[u8])],
    ) -> Result<(), Error> {
        let t = self.buf_times.len() as f64 / f64::from(self.spec.fps);
        self.add_frame_at_with_images(values, images, t)
    }

    /// [`add_frame_with_images`](Self::add_frame_with_images) with an explicit
    /// timestamp (seconds).
    pub fn add_frame_at_with_images(
        &mut self,
        values: &[(&str, &[f64])],
        images: &[(&str, &[u8])],
        timestamp: f64,
    ) -> Result<(), Error> {
        if self.finalized {
            return Err(Error::State("writer already finalized".into()));
        }
        let n_vec = self.spec.features.iter().filter(|f| !f.is_image()).count();
        let n_img = self.spec.features.len() - n_vec;
        if values.len() != n_vec {
            return Err(Error::State(format!(
                "frame has {} vector features, dataset declares {n_vec}",
                values.len()
            )));
        }
        if images.len() != n_img {
            return Err(Error::State(format!(
                "frame has {} image features, dataset declares {n_img}{}",
                images.len(),
                if n_img > 0 && images.is_empty() {
                    " (use add_frame_with_images for datasets with image features)"
                } else {
                    ""
                }
            )));
        }
        for (name, _) in values {
            match self.spec.features.iter().find(|f| f.name == *name) {
                Some(f) if f.is_image() => {
                    return Err(Error::State(format!(
                        "feature '{name}' is an image feature; pass it in `images`"
                    )));
                }
                Some(_) => {}
                None => return Err(Error::State(format!("unknown feature '{name}'"))),
            }
        }
        for (name, _) in images {
            match self.spec.features.iter().find(|f| f.name == *name) {
                Some(f) if !f.is_image() => {
                    return Err(Error::State(format!(
                        "feature '{name}' is a vector feature; pass it in `values`"
                    )));
                }
                Some(_) => {}
                None => return Err(Error::State(format!("unknown image feature '{name}'"))),
            }
        }
        // Validate everything (including a full decode of every PNG) before
        // mutating any buffer, so a failed frame never leaves the per-feature
        // buffers ragged.
        enum Ordered<'a> {
            Vector(&'a [f64]),
            Image(&'a [u8], PixelStats),
        }
        let mut ordered: Vec<Ordered<'_>> = Vec::with_capacity(self.spec.features.len());
        for feat in &self.spec.features {
            match &feat.kind {
                FeatureKind::Vector { dim, .. } => {
                    let (_, v) = values
                        .iter()
                        .find(|(n, _)| *n == feat.name)
                        .ok_or_else(|| Error::State(format!("missing feature '{}'", feat.name)))?;
                    if v.len() != *dim {
                        return Err(Error::Shape {
                            name: feat.name.clone(),
                            expected: *dim,
                            got: v.len(),
                        });
                    }
                    ordered.push(Ordered::Vector(v));
                }
                FeatureKind::Image {
                    height,
                    width,
                    channels,
                } => {
                    let (_, png) =
                        images
                            .iter()
                            .find(|(n, _)| *n == feat.name)
                            .ok_or_else(|| {
                                Error::State(format!("missing image feature '{}'", feat.name))
                            })?;
                    let stats = decode_png_stats(&feat.name, png, *height, *width, *channels)?;
                    ordered.push(Ordered::Image(png, stats));
                }
            }
        }
        for (buf, v) in self.buf.iter_mut().zip(ordered) {
            match (buf, v) {
                (FeatureBuf::Vector(buf), Ordered::Vector(v)) => {
                    buf.extend(v.iter().map(|&x| x as f32));
                }
                (FeatureBuf::Image(buf), Ordered::Image(png, stats)) => {
                    buf.push(ImageFrame {
                        png: png.to_vec(),
                        stats,
                    });
                }
                // Buffers are built from the same spec that `ordered` walked.
                _ => unreachable!("buffer kind matches spec kind by construction"),
            }
        }
        self.buf_times.push(timestamp);
        Ok(())
    }

    /// Close the buffered frames as one episode tagged with `task`, writing
    /// its row group to the current data file (rolling to a new file first if
    /// the size target would be exceeded).
    pub fn save_episode(&mut self, task: &str) -> Result<(), Error> {
        let t = task.to_string();
        let frame_tasks = vec![t.clone(); self.buf_times.len()];
        self.save_episode_with_tasks(&[t], &frame_tasks)
    }

    /// Like [`save_episode`](Self::save_episode) but with a per-frame task and
    /// an explicit episode-level task list — what the offline edit ops need to
    /// merge episodes with different tasks without collapsing their frames'
    /// `task_index` values. `frame_tasks` must have one entry per buffered
    /// frame; `episode_tasks` is stored in the `tasks` column of
    /// `meta/episodes` and every string (from both lists) is interned into
    /// `meta/tasks.parquet`.
    pub(crate) fn save_episode_with_tasks(
        &mut self,
        episode_tasks: &[String],
        frame_tasks: &[String],
    ) -> Result<(), Error> {
        if self.finalized {
            return Err(Error::State("writer already finalized".into()));
        }
        let len = self.buf_times.len();
        if len == 0 {
            return Err(Error::State(
                "no frames buffered; call add_frame first".into(),
            ));
        }
        if episode_tasks.is_empty() {
            return Err(Error::State("episode needs at least one task".into()));
        }
        if frame_tasks.len() != len {
            return Err(Error::State(format!(
                "{} frame tasks for {} buffered frames",
                frame_tasks.len(),
                len
            )));
        }
        let episode_index = self.episodes.len() as i64;
        for t in episode_tasks {
            self.intern_task(t);
        }
        let frame_task_idx: Vec<i64> = frame_tasks.iter().map(|t| self.intern_task(t)).collect();

        let batch = self.build_episode_batch(episode_index, &frame_task_idx)?;
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

        let stats = self.episode_stats(episode_index, &frame_task_idx);
        self.episodes.push(EpisodeRecord {
            episode_index,
            tasks: episode_tasks.to_vec(),
            length: len as u64,
            chunk_index,
            file_index,
            from_index: self.global_index,
            to_index: self.global_index + len as i64,
            stats,
        });
        self.global_index += len as i64;
        for buf in &mut self.buf {
            match buf {
                FeatureBuf::Vector(b) => b.clear(),
                FeatureBuf::Image(b) => b.clear(),
            }
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
        // Image entries serialize with lerobot's (c, 1, 1) nesting so
        // normalization layers can broadcast them against CHW tensors; vector
        // entries keep the flat lists lerobot writes for 1-D features.
        let image_names = self.image_feature_names();
        let entries: BTreeMap<&String, StatsJson<'_>> = aggregated
            .iter()
            .map(|(name, s)| {
                let entry = if image_names.contains(&name.as_str()) {
                    StatsJson::Image(NestedFeatureStats::from(s))
                } else {
                    StatsJson::Flat(s)
                };
                (name, entry)
            })
            .collect();
        fs::write(
            self.root.join("meta/stats.json"),
            serde_json::to_string_pretty(&entries)?,
        )?;
        fs::write(
            self.root.join("meta/info.json"),
            serde_json::to_string_pretty(&self.build_info())?,
        )?;
        Ok(())
    }

    // ---- internals ----

    /// Names of the declared image features (for stats-shape decisions).
    fn image_feature_names(&self) -> Vec<&str> {
        self.spec
            .features
            .iter()
            .filter(|f| f.is_image())
            .map(|f| f.name.as_str())
            .collect()
    }

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
        frame_task_idx: &[i64],
    ) -> Result<RecordBatch, Error> {
        let len = self.buf_times.len();
        let mut fields: Vec<Field> = Vec::new();
        let mut columns: Vec<ArrayRef> = Vec::new();
        for (feat, buf) in self.spec.features.iter().zip(&self.buf) {
            let arr: ArrayRef = match buf {
                FeatureBuf::Vector(buf) => {
                    let FeatureKind::Vector { dim, .. } = &feat.kind else {
                        unreachable!("buffer kind matches spec kind by construction");
                    };
                    let mut b = FixedSizeListBuilder::new(Float32Builder::new(), *dim as i32);
                    for row in buf.chunks_exact(*dim) {
                        for &x in row {
                            b.values().append_value(x);
                        }
                        b.append(true);
                    }
                    Arc::new(b.finish())
                }
                // lerobot's `dtype: "image"` storage — the HF `datasets.Image`
                // arrow layout: `struct<bytes: binary, path: string>` (field
                // order bytes,path), `bytes` = one complete PNG file, `path` =
                // the per-episode frame basename lerobot's own writer embeds.
                FeatureBuf::Image(buf) => {
                    let mut bytes_b = BinaryBuilder::new();
                    let mut path_b = StringBuilder::new();
                    for (i, frame) in buf.iter().enumerate() {
                        bytes_b.append_value(&frame.png);
                        path_b.append_value(format!("frame-{i:06}.png"));
                    }
                    let struct_fields = Fields::from(vec![
                        Field::new("bytes", DataType::Binary, true),
                        Field::new("path", DataType::Utf8, true),
                    ]);
                    let arr = StructArray::try_new(
                        struct_fields,
                        vec![
                            Arc::new(bytes_b.finish()) as ArrayRef,
                            Arc::new(path_b.finish()) as ArrayRef,
                        ],
                        None,
                    )?;
                    Arc::new(arr)
                }
            };
            fields.push(Field::new(&feat.name, arr.data_type().clone(), true));
            columns.push(arr);
        }
        let timestamp =
            Float32Array::from(self.buf_times.iter().map(|&t| t as f32).collect::<Vec<_>>());
        let frame_index = Int64Array::from((0..len as i64).collect::<Vec<_>>());
        let episode = Int64Array::from(vec![episode_index; len]);
        let index = Int64Array::from(
            (self.global_index..self.global_index + len as i64).collect::<Vec<_>>(),
        );
        let task = Int64Array::from(frame_task_idx.to_vec());
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
    fn episode_stats(
        &self,
        episode_index: i64,
        frame_task_idx: &[i64],
    ) -> BTreeMap<String, FeatureStats> {
        let len = self.buf_times.len();
        let mut stats = BTreeMap::new();
        for (feat, buf) in self.spec.features.iter().zip(&self.buf) {
            let s = match buf {
                FeatureBuf::Vector(buf) => {
                    let FeatureKind::Vector { dim, .. } = &feat.kind else {
                        unreachable!("buffer kind matches spec kind by construction");
                    };
                    let rows: Vec<Vec<f64>> = buf
                        .chunks_exact(*dim)
                        .map(|row| row.iter().map(|&x| f64::from(x)).collect())
                        .collect();
                    FeatureStats::compute(&rows, *dim)
                }
                FeatureBuf::Image(buf) => fold_image_stats(buf),
            };
            stats.insert(feat.name.clone(), s);
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
        stats.insert(
            "task_index".into(),
            scalar(frame_task_idx.iter().map(|&t| t as f64).collect()),
        );
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
            for t in &e.tasks {
                tasks_b.values().append_value(t);
            }
            tasks_b.append(true);
        }
        let tasks_arr = tasks_b.finish();
        fields.push(Field::new("tasks", tasks_arr.data_type().clone(), true));
        columns.push(Arc::new(tasks_arr));

        // stats/<feature>/<stat>: features sorted (BTreeMap order), stats in
        // the converter's alphabetical count/max/mean/min/std order. Image
        // features nest their value cells as (c, 1, 1) — the shape lerobot's
        // own episodes parquet carries — while `count` stays a flat frame
        // count for every feature. (lerobot drops all stats/ columns when
        // loading; the authoritative aggregate lives in meta/stats.json.)
        let image_names = self.image_feature_names();
        for feat_name in self.episodes[0].stats.keys() {
            let is_image = image_names.contains(&feat_name.as_str());
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
                    continue;
                }
                let pick = |s: &'static str, e: &EpisodeRecord| -> Vec<f64> {
                    let st = &e.stats[feat_name];
                    match s {
                        "max" => st.max.clone(),
                        "mean" => st.mean.clone(),
                        "min" => st.min.clone(),
                        _ => st.std.clone(),
                    }
                };
                if is_image {
                    // (c, 1, 1): List<List<List<f64>>> — per row, `c` outer
                    // entries of `[[v]]`.
                    let mut b =
                        ListBuilder::new(ListBuilder::new(ListBuilder::new(Float64Builder::new())));
                    for e in &self.episodes {
                        for v in pick(stat, e) {
                            b.values().values().values().append_value(v);
                            b.values().values().append(true);
                            b.values().append(true);
                        }
                        b.append(true);
                    }
                    let arr = b.finish();
                    fields.push(Field::new(&name, arr.data_type().clone(), true));
                    columns.push(Arc::new(arr));
                } else {
                    let mut b = ListBuilder::new(Float64Builder::new());
                    for e in &self.episodes {
                        for v in pick(stat, e) {
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
            let info = match &f.kind {
                FeatureKind::Vector { dim, names } => FeatureInfo {
                    dtype: "float32".into(),
                    shape: vec![*dim as u64],
                    names: match names {
                        Some(n) => serde_json::json!(n),
                        None => serde_json::Value::Null,
                    },
                    fps: Some(self.spec.fps),
                },
                // Exactly the entry lerobot's own `LeRobotDataset.create`
                // writes for `dtype: "image"`: shape is (h, w, c) and names
                // MUST be this list — `dataset_to_policy_features` checks
                // `names[2] in ("channel", "channels")` to flip HWC→CHW.
                FeatureKind::Image {
                    height,
                    width,
                    channels,
                } => FeatureInfo {
                    dtype: "image".into(),
                    shape: vec![*height as u64, *width as u64, *channels as u64],
                    names: serde_json::json!(["height", "width", "channels"]),
                    fps: Some(self.spec.fps),
                },
            };
            features.insert(f.name.clone(), info);
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

/// Decode-validate one PNG frame and compute its per-channel [`PixelStats`]
/// (normalized to lerobot's [0, 1] scale). The bytes themselves are stored
/// verbatim — this decode exists to reject wrong-shaped frames LOUDLY at
/// `add_frame` time and to feed the image stats lerobot expects.
fn decode_png_stats(
    name: &str,
    bytes: &[u8],
    height: usize,
    width: usize,
    channels: usize,
) -> Result<PixelStats, Error> {
    let err = |msg: String| Error::Format(format!("image feature '{name}': {msg}"));
    // png 0.18's Decoder wants Read + Seek — a Cursor over the slice provides both.
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    // Normalize palette / sub-8-bit images to plain 8-bit samples so the
    // channel check below sees real channels, whatever the encoder chose.
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder
        .read_info()
        .map_err(|e| err(format!("not a decodable PNG: {e}")))?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| err("PNG output size overflows".into()))?;
    let mut buf = vec![0u8; size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| err(format!("PNG decode failed: {e}")))?;
    if info.bit_depth != png::BitDepth::Eight {
        return Err(err(format!(
            "expected 8-bit samples, decoded {:?}",
            info.bit_depth
        )));
    }
    let got_c = info.color_type.samples();
    if (info.height as usize, info.width as usize, got_c) != (height, width, channels) {
        return Err(err(format!(
            "decoded {}x{}x{got_c} (h x w x channels), spec declares {height}x{width}x{channels}",
            info.height, info.width
        )));
    }
    let data = &buf[..info.buffer_size()];
    if data.len() != height * width * channels {
        return Err(err(format!(
            "decoded byte count {} != h*w*c = {}",
            data.len(),
            height * width * channels
        )));
    }
    let mut st = PixelStats {
        min: vec![f64::INFINITY; channels],
        max: vec![f64::NEG_INFINITY; channels],
        sum: vec![0.0; channels],
        sumsq: vec![0.0; channels],
        pixels: (height * width) as u64,
    };
    for px in data.chunks_exact(channels) {
        for (j, &b) in px.iter().enumerate() {
            let v = f64::from(b) / 255.0;
            if v < st.min[j] {
                st.min[j] = v;
            }
            if v > st.max[j] {
                st.max[j] = v;
            }
            st.sum[j] += v;
            st.sumsq[j] += v * v;
        }
    }
    Ok(st)
}

/// Fold per-frame pixel stats into one episode-level [`FeatureStats`]:
/// per-channel population stats over EVERY pixel of every frame, with
/// lerobot's `count = [n_frames]` convention (pixels-per-frame is constant
/// per feature, so the frame-weighted aggregation in `aggregate_stats` stays
/// exact for images too).
fn fold_image_stats(frames: &[ImageFrame]) -> FeatureStats {
    let c = frames.first().map_or(0, |f| f.stats.min.len());
    let mut min = vec![f64::INFINITY; c];
    let mut max = vec![f64::NEG_INFINITY; c];
    let mut sum = vec![0.0; c];
    let mut sumsq = vec![0.0; c];
    let mut total_px = 0u64;
    for f in frames {
        for j in 0..c {
            if f.stats.min[j] < min[j] {
                min[j] = f.stats.min[j];
            }
            if f.stats.max[j] > max[j] {
                max[j] = f.stats.max[j];
            }
            sum[j] += f.stats.sum[j];
            sumsq[j] += f.stats.sumsq[j];
        }
        total_px += f.stats.pixels;
    }
    let denom = total_px.max(1) as f64;
    let mean: Vec<f64> = sum.iter().map(|s| s / denom).collect();
    let std: Vec<f64> = sumsq
        .iter()
        .zip(&mean)
        .map(|(sq, m)| (sq / denom - m * m).max(0.0).sqrt())
        .collect();
    FeatureStats {
        min,
        max,
        mean,
        std,
        count: vec![frames.len() as u64],
    }
}

/// `meta/stats.json` value shapes: vector features keep [`FeatureStats`]'s
/// flat lists; image features nest each value as `(c, 1, 1)` — exactly the
/// shape lerobot's own writer emits so normalization broadcasts over CHW.
/// Untagged, so both serialize as the bare stats object.
#[derive(serde::Serialize)]
#[serde(untagged)]
enum StatsJson<'a> {
    Flat(&'a FeatureStats),
    Image(NestedFeatureStats),
}

/// [`FeatureStats`] with `(c, 1, 1)`-nested values (field order matches
/// `FeatureStats` so both variants pretty-print with the same key order).
#[derive(serde::Serialize)]
struct NestedFeatureStats {
    min: Vec<[[f64; 1]; 1]>,
    max: Vec<[[f64; 1]; 1]>,
    mean: Vec<[[f64; 1]; 1]>,
    std: Vec<[[f64; 1]; 1]>,
    count: Vec<u64>,
}

impl From<&FeatureStats> for NestedFeatureStats {
    fn from(s: &FeatureStats) -> Self {
        let nest = |v: &[f64]| v.iter().map(|&x| [[x]]).collect();
        Self {
            min: nest(&s.min),
            max: nest(&s.max),
            mean: nest(&s.mean),
            std: nest(&s.std),
            count: s.count.clone(),
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
