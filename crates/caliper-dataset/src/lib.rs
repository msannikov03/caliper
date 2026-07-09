//! Native LeRobotDataset **v3.0** writer/reader.
//!
//! Writes and reads the on-disk format of `lerobot` 0.4.4 (`codebase_version`
//! `"v3.0"`), video-less this wave — lerobot loads such datasets fine. The
//! format was reverse-engineered from two ground truths:
//!
//! 1. the installed `lerobot` sources (`datasets/lerobot_dataset.py`,
//!    `datasets/utils.py`, `datasets/compute_stats.py`,
//!    `datasets/v30/convert_dataset_v21_to_v30.py`), and
//! 2. a reference dataset produced by recording through Caliper's v2.1
//!    `Recorder` and running lerobot's official v2.1→v3.0 converter — checked
//!    in at `oracle/fixtures/datasets/ref_v30/` (see its `README.md`; the
//!    reader-vs-converter test runs against it hermetically).
//!
//! On-disk layout written by [`DatasetWriter`]:
//!
//! ```text
//! meta/info.json                          # codebase_version, features, sizes, templates
//! meta/tasks.parquet                      # task_index + task string (pandas index column)
//! meta/stats.json                         # aggregated min/max/mean/std/count per feature
//! meta/episodes/chunk-000/file-000.parquet# episode offsets + per-episode stats columns
//! data/chunk-XXX/file-XXX.parquet         # episodes concatenated, size-rolled
//! ```
//!
//! The **lerobot footgun** — forgetting `finalize()` and losing the parquet
//! footer plus all episode metadata — is structurally impossible here: the
//! writer auto-finalizes on `Drop` (footers flushed, buffered episode
//! metadata written). An explicit [`DatasetWriter::finalize`] is still the
//! right way to end a recording because it can report errors.

mod error;
mod meta;
mod reader;
mod stats;
mod writer;

pub use error::Error;
pub use meta::{FeatureInfo, Info, format_chunk_file_path};
pub use reader::{DatasetReader, EpisodeData, EpisodeMeta};
pub use stats::{FeatureStats, aggregate_stats};
pub use writer::{DatasetSpec, DatasetWriter, FeatureSpec};

/// `codebase_version` this crate writes and accepts.
pub const CODEBASE_VERSION: &str = "v3.0";
