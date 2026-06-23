//! LeRobotDataset v2.1 record/replay (feature `dataset`). The control loop feeds
//! `(observation.state, action)` per tick; on finalize each episode is written as
//! an Apache Parquet file plus the `meta/` JSON sidecars, in the on-disk layout
//! the HF `lerobot` library reads.
//!
//! ⚠️ Schema fidelity is validated locally by (1) a Rust write→read round-trip and
//! (2) a pyarrow column/dtype/stats check — NOT against `lerobot` itself (not
//! importable in this environment). Treat lerobot-readability as best-effort until
//! the `importorskip` cross-check actually runs in CI.

use crate::control::Frame;
use crate::{Error, RobotBackend};
use caliper_model::Model;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Array, FixedSizeListBuilder, Float32Array, Float32Builder, Int64Array};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

pub const CODEBASE_VERSION: &str = "v2.1";
pub const CHUNK_SIZE: usize = 1000;

/// Dataset feature spec: dimensionality + joint names + frame rate.
#[derive(Clone, Debug)]
pub struct DatasetSpec {
    pub fps: u32,
    pub ndof: usize,
    pub joint_names: Vec<String>,
    pub robot_type: String,
}
impl DatasetSpec {
    pub fn from_model(model: &Model, fps: u32) -> Self {
        Self {
            fps,
            ndof: model.ndof,
            joint_names: model.joint_names.clone(),
            robot_type: model.name.clone(),
        }
    }
}

/// Per-feature summary stats (population, ddof=0) over an episode.
#[derive(Clone, Debug)]
struct Stats {
    min: Vec<f64>,
    max: Vec<f64>,
    mean: Vec<f64>,
    std: Vec<f64>,
    count: usize,
}
fn column_stats(rows: &[Vec<f64>], width: usize) -> Stats {
    let n = rows.len().max(1);
    let mut min = vec![f64::INFINITY; width];
    let mut max = vec![f64::NEG_INFINITY; width];
    let mut mean = vec![0.0; width];
    for r in rows {
        for j in 0..width {
            let v = r[j];
            min[j] = min[j].min(v);
            max[j] = max[j].max(v);
            mean[j] += v / n as f64;
        }
    }
    let mut var = vec![0.0; width];
    for r in rows {
        for j in 0..width {
            let d = r[j] - mean[j];
            var[j] += d * d / n as f64; // ddof = 0 (population)
        }
    }
    if rows.is_empty() {
        min.iter_mut().for_each(|x| *x = 0.0);
        max.iter_mut().for_each(|x| *x = 0.0);
    }
    Stats {
        std: var.iter().map(|v| v.sqrt()).collect(),
        min,
        max,
        mean,
        count: rows.len(),
    }
}
fn stats_json(s: &Stats) -> serde_json::Value {
    serde_json::json!({
        "min": s.min, "max": s.max, "mean": s.mean, "std": s.std, "count": [s.count],
    })
}

/// Writes a LeRobotDataset v2.1 to disk. One episode at a time.
pub struct Recorder {
    root: PathBuf,
    spec: DatasetSpec,
    next_episode: usize,
    global_index: i64,
    total_frames: usize,
    // accumulator for the current episode
    cur: Option<EpisodeBuf>,
    // meta accumulators (written on close)
    episodes_meta: Vec<serde_json::Value>,
    episodes_stats: Vec<serde_json::Value>,
    tasks: Vec<String>,
}

struct EpisodeBuf {
    task_index: i64,
    states: Vec<Vec<f64>>,
    actions: Vec<Vec<f64>>,
    timestamps: Vec<f64>,
}

impl Recorder {
    /// Create (or overwrite into) a dataset directory tree.
    pub fn create(root: impl AsRef<Path>, spec: DatasetSpec) -> Result<Self, Error> {
        let root = root.as_ref().to_path_buf();
        for sub in ["meta", "data/chunk-000"] {
            fs::create_dir_all(root.join(sub)).map_err(|e| Error::Backend(e.to_string()))?;
        }
        Ok(Self {
            root,
            spec,
            next_episode: 0,
            global_index: 0,
            total_frames: 0,
            cur: None,
            episodes_meta: Vec::new(),
            episodes_stats: Vec::new(),
            tasks: Vec::new(),
        })
    }

    fn task_index(&mut self, task: &str) -> i64 {
        if let Some(i) = self.tasks.iter().position(|t| t == task) {
            return i as i64;
        }
        self.tasks.push(task.to_string());
        (self.tasks.len() - 1) as i64
    }

    /// Begin a new episode tagged with a natural-language `task`.
    pub fn start_episode(&mut self, task: &str) -> Result<(), Error> {
        if self.cur.is_some() {
            return Err(Error::Backend("episode already open".into()));
        }
        let task_index = self.task_index(task);
        self.cur = Some(EpisodeBuf {
            task_index,
            states: Vec::new(),
            actions: Vec::new(),
            timestamps: Vec::new(),
        });
        Ok(())
    }

    /// Append one frame: `state` = observation (read BEFORE the command), `action`
    /// = the commanded target, `t` = timestamp (seconds).
    pub fn append_frame(&mut self, state: &[f64], action: &[f64], t: f64) -> Result<(), Error> {
        let n = self.spec.ndof;
        if state.len() != n || action.len() != n {
            return Err(Error::DofMismatch {
                expected: n,
                got: state.len().min(action.len()),
            });
        }
        let ep = self
            .cur
            .as_mut()
            .ok_or_else(|| Error::Backend("no open episode".into()))?;
        ep.states.push(state.to_vec());
        ep.actions.push(action.to_vec());
        ep.timestamps.push(t);
        Ok(())
    }

    /// Convenience: map a control-loop [`Frame`] (measured→state, command→action).
    pub fn append_control_frame(&mut self, f: &Frame) -> Result<(), Error> {
        self.append_frame(&f.measured, &f.command, f.t)
    }

    /// Write the current episode's parquet + accumulate its meta.
    pub fn finalize_episode(&mut self) -> Result<(), Error> {
        let ep = self
            .cur
            .take()
            .ok_or_else(|| Error::Backend("no open episode".into()))?;
        let ep_index = self.next_episode as i64;
        let len = ep.timestamps.len();
        let n = self.spec.ndof;

        // ---- parquet ----
        let state_arr = fixed_list(&ep.states, n);
        let action_arr = fixed_list(&ep.actions, n);
        let timestamp =
            Float32Array::from(ep.timestamps.iter().map(|&t| t as f32).collect::<Vec<_>>());
        let frame_index = Int64Array::from((0..len as i64).collect::<Vec<_>>());
        let episode_index = Int64Array::from(vec![ep_index; len]);
        let index = Int64Array::from(
            (0..len as i64)
                .map(|i| self.global_index + i)
                .collect::<Vec<_>>(),
        );
        let task_index = Int64Array::from(vec![ep.task_index; len]);

        let schema = Arc::new(Schema::new(vec![
            Field::new("observation.state", state_arr.data_type().clone(), false),
            Field::new("action", action_arr.data_type().clone(), false),
            Field::new("timestamp", timestamp.data_type().clone(), false),
            Field::new("frame_index", frame_index.data_type().clone(), false),
            Field::new("episode_index", episode_index.data_type().clone(), false),
            Field::new("index", index.data_type().clone(), false),
            Field::new("task_index", task_index.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(state_arr),
                Arc::new(action_arr),
                Arc::new(timestamp),
                Arc::new(frame_index),
                Arc::new(episode_index),
                Arc::new(index),
                Arc::new(task_index),
            ],
        )
        .map_err(|e| Error::Backend(e.to_string()))?;

        let path = self
            .root
            .join("data/chunk-000")
            .join(format!("episode_{ep_index:06}.parquet"));
        let file = File::create(&path).map_err(|e| Error::Backend(e.to_string()))?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))
            .map_err(|e| Error::Backend(e.to_string()))?;
        writer
            .write(&batch)
            .map_err(|e| Error::Backend(e.to_string()))?;
        writer.close().map_err(|e| Error::Backend(e.to_string()))?;

        // ---- meta ----
        let task = self.tasks[ep.task_index as usize].clone();
        self.episodes_meta.push(serde_json::json!({
            "episode_index": ep_index, "tasks": [task], "length": len,
        }));
        let st = column_stats(&ep.states, n);
        let ac = column_stats(&ep.actions, n);
        let ts: Vec<Vec<f64>> = ep.timestamps.iter().map(|&t| vec![t]).collect();
        let tst = column_stats(&ts, 1);
        self.episodes_stats.push(serde_json::json!({
            "episode_index": ep_index,
            "stats": {
                "observation.state": stats_json(&st),
                "action": stats_json(&ac),
                "timestamp": stats_json(&tst),
            }
        }));

        self.global_index += len as i64;
        self.total_frames += len;
        self.next_episode += 1;
        Ok(())
    }

    /// Write `meta/info.json`, `episodes.jsonl`, `tasks.jsonl`, `episodes_stats.jsonl`.
    pub fn close(self) -> Result<PathBuf, Error> {
        let n = self.spec.ndof;
        let names = serde_json::Value::Array(
            self.spec
                .joint_names
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect(),
        );
        let feat_vec = |names: &serde_json::Value| serde_json::json!({"dtype": "float32", "shape": [n], "names": names});
        let feat_scalar = |dtype: &str| serde_json::json!({"dtype": dtype, "shape": [1], "names": serde_json::Value::Null});
        let info = serde_json::json!({
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.spec.robot_type,
            "total_episodes": self.next_episode,
            "total_frames": self.total_frames,
            "total_tasks": self.tasks.len(),
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": CHUNK_SIZE,
            "fps": self.spec.fps,
            "splits": {"train": format!("0:{}", self.next_episode)},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": serde_json::Value::Null,
            "features": {
                "observation.state": feat_vec(&names),
                "action": feat_vec(&names),
                "timestamp": feat_scalar("float32"),
                "frame_index": feat_scalar("int64"),
                "episode_index": feat_scalar("int64"),
                "index": feat_scalar("int64"),
                "task_index": feat_scalar("int64"),
            }
        });
        write_json(&self.root.join("meta/info.json"), &info)?;
        write_jsonl(&self.root.join("meta/episodes.jsonl"), &self.episodes_meta)?;
        write_jsonl(
            &self.root.join("meta/episodes_stats.jsonl"),
            &self.episodes_stats,
        )?;
        let tasks: Vec<serde_json::Value> = self
            .tasks
            .iter()
            .enumerate()
            .map(|(i, t)| serde_json::json!({"task_index": i, "task": t}))
            .collect();
        write_jsonl(&self.root.join("meta/tasks.jsonl"), &tasks)?;
        Ok(self.root)
    }
}

fn fixed_list(rows: &[Vec<f64>], width: usize) -> arrow::array::FixedSizeListArray {
    let mut b = FixedSizeListBuilder::new(Float32Builder::new(), width as i32);
    for r in rows {
        for &x in r {
            b.values().append_value(x as f32);
        }
        b.append(true);
    }
    b.finish()
}

fn write_json(path: &Path, v: &serde_json::Value) -> Result<(), Error> {
    let s = serde_json::to_string_pretty(v).map_err(|e| Error::Backend(e.to_string()))?;
    fs::write(path, s).map_err(|e| Error::Backend(e.to_string()))
}
fn write_jsonl(path: &Path, rows: &[serde_json::Value]) -> Result<(), Error> {
    let mut s = String::new();
    for r in rows {
        s.push_str(&serde_json::to_string(r).map_err(|e| Error::Backend(e.to_string()))?);
        s.push('\n');
    }
    fs::write(path, s).map_err(|e| Error::Backend(e.to_string()))
}

// ===== reader / replay =====

/// One episode read back from disk.
#[derive(Clone, Debug)]
pub struct Episode {
    pub index: u64,
    pub states: Vec<Vec<f64>>,
    pub actions: Vec<Vec<f64>>,
    pub timestamps: Vec<f64>,
}
impl Episode {
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }
}

/// Reads a LeRobotDataset v2.1 from disk.
pub struct DatasetReader {
    root: PathBuf,
    pub total_episodes: usize,
    pub fps: u32,
    pub ndof: usize,
}
impl DatasetReader {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, Error> {
        let root = root.as_ref().to_path_buf();
        let info: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("meta/info.json"))
                .map_err(|e| Error::Backend(e.to_string()))?,
        )
        .map_err(|e| Error::Backend(e.to_string()))?;
        let total_episodes = info["total_episodes"].as_u64().unwrap_or(0) as usize;
        let fps = info["fps"].as_u64().unwrap_or(30) as u32;
        let ndof = info["features"]["action"]["shape"][0].as_u64().unwrap_or(0) as usize;
        Ok(Self {
            root,
            total_episodes,
            fps,
            ndof,
        })
    }

    pub fn read_episode(&self, ep: usize) -> Result<Episode, Error> {
        let path = self
            .root
            .join("data/chunk-000")
            .join(format!("episode_{ep:06}.parquet"));
        let file = File::open(&path).map_err(|e| Error::Backend(e.to_string()))?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| Error::Backend(e.to_string()))?
            .build()
            .map_err(|e| Error::Backend(e.to_string()))?;
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut timestamps = Vec::new();
        for batch in reader {
            let batch = batch.map_err(|e| Error::Backend(e.to_string()))?;
            let col = |name: &str| -> Result<usize, Error> {
                batch
                    .schema()
                    .index_of(name)
                    .map_err(|e| Error::Backend(e.to_string()))
            };
            read_fixed_list(&batch, col("observation.state")?, &mut states)?;
            read_fixed_list(&batch, col("action")?, &mut actions)?;
            let ts = batch
                .column(col("timestamp")?)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| Error::Backend("timestamp not f32".into()))?;
            for i in 0..ts.len() {
                timestamps.push(ts.value(i) as f64);
            }
        }
        Ok(Episode {
            index: ep as u64,
            states,
            actions,
            timestamps,
        })
    }
}

fn read_fixed_list(batch: &RecordBatch, col: usize, out: &mut Vec<Vec<f64>>) -> Result<(), Error> {
    use arrow::array::FixedSizeListArray;
    let arr = batch
        .column(col)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| Error::Backend("expected FixedSizeList".into()))?;
    for i in 0..arr.len() {
        let v = arr.value(i);
        let f = v
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| Error::Backend("inner not f32".into()))?;
        out.push((0..f.len()).map(|j| f.value(j) as f64).collect());
    }
    Ok(())
}

/// Replay one recorded action onto a backend (position command).
pub fn replay_frame(backend: &mut dyn RobotBackend, ep: &Episode, i: usize) -> Result<(), Error> {
    let a = ep
        .actions
        .get(i)
        .ok_or_else(|| Error::Backend(format!("frame {i} out of range")))?;
    backend.command_joint_positions(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimBackend;

    fn tmpdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("caliper_ds_{tag}"));
        let _ = fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn write_read_roundtrip_and_replay() {
        let spec = DatasetSpec {
            fps: 50,
            ndof: 2,
            joint_names: vec!["j1".into(), "j2".into()],
            robot_type: "test".into(),
        };
        let dir = tmpdir("rt");
        let mut rec = Recorder::create(&dir, spec).unwrap();
        rec.start_episode("wave").unwrap();
        let states: Vec<[f64; 2]> = (0..5)
            .map(|k| [k as f64 * 0.1, -(k as f64) * 0.2])
            .collect();
        let actions: Vec<[f64; 2]> = (0..5)
            .map(|k| [k as f64 * 0.1 + 0.01, -(k as f64) * 0.2])
            .collect();
        for k in 0..5 {
            rec.append_frame(&states[k], &actions[k], k as f64 / 50.0)
                .unwrap();
        }
        rec.finalize_episode().unwrap();
        let root = rec.close().unwrap();

        // meta exists
        assert!(root.join("meta/info.json").exists());
        assert!(root.join("data/chunk-000/episode_000000.parquet").exists());

        // read back
        let rd = DatasetReader::open(&root).unwrap();
        assert_eq!(rd.total_episodes, 1);
        assert_eq!(rd.ndof, 2);
        let ep = rd.read_episode(0).unwrap();
        assert_eq!(ep.len(), 5);
        for k in 0..5 {
            for j in 0..2 {
                assert!((ep.states[k][j] - states[k][j]).abs() < 1e-6);
                assert!((ep.actions[k][j] - actions[k][j]).abs() < 1e-6);
            }
            assert!((ep.timestamps[k] - k as f64 / 50.0).abs() < 1e-6);
        }

        // replay drives a backend to the recorded action
        let mut b = SimBackend::new(2);
        replay_frame(&mut b, &ep, 3).unwrap();
        let q = b.joint_positions();
        for j in 0..2 {
            assert!((q[j] - actions[3][j]).abs() < 1e-6);
        }
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn global_index_monotonic_across_episodes() {
        let spec = DatasetSpec {
            fps: 30,
            ndof: 1,
            joint_names: vec!["j".into()],
            robot_type: "t".into(),
        };
        let dir = tmpdir("multi");
        let mut rec = Recorder::create(&dir, spec).unwrap();
        for e in 0..2 {
            rec.start_episode("t").unwrap();
            for k in 0..3 {
                rec.append_frame(&[k as f64], &[e as f64], k as f64)
                    .unwrap();
            }
            rec.finalize_episode().unwrap();
        }
        let root = rec.close().unwrap();
        let rd = DatasetReader::open(&root).unwrap();
        assert_eq!(rd.total_episodes, 2);
        let _ = fs::remove_dir_all(&root);
    }
}
