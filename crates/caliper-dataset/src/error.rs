//! Crate error type.

/// Errors from writing or reading a LeRobotDataset v3.0.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("parquet: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// The on-disk dataset violates the v3.0 contract (missing file, wrong
    /// column type, unsupported template, …).
    #[error("invalid dataset: {0}")]
    Format(String),
    /// A frame value had the wrong number of elements for its feature.
    #[error("feature '{name}': expected {expected} values, got {got}")]
    Shape {
        name: String,
        expected: usize,
        got: usize,
    },
    /// The writer was used out of order (e.g. `save_episode` with no frames).
    #[error("writer state: {0}")]
    State(String),
    /// An offline edit operation was given invalid arguments (bad episode
    /// index, non-adjacent merge, out-of-range split frame, …) or found
    /// leftovers of a crashed previous edit next to the dataset.
    #[error("edit: {0}")]
    Edit(String),
}
