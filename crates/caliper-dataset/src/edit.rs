//! Offline dataset-level **edit operations** — delete / split / merge episodes
//! plus a caliper-only per-episode tags sidecar. Every op leaves a valid,
//! lerobot-loadable v3.0 dataset behind.
//!
//! # Implementation shape: read → rewrite → atomic swap
//!
//! Each op streams the source dataset through [`DatasetReader`] with the
//! transformation applied and re-records it with [`DatasetWriter`] into a
//! **sibling temp directory** (`<root>.caliper-edit-tmp`), then swaps:
//!
//! 1. build the full new dataset in `<root>.caliper-edit-tmp`
//! 2. `rename(<root> → <root>.caliper-edit-old)`
//! 3. `rename(<root>.caliper-edit-tmp → <root>)`
//! 4. `remove_dir_all(<root>.caliper-edit-old)`
//!
//! **Crash behavior**: the original is untouched until step 2. A crash before
//! step 2 leaves at most a stray `.caliper-edit-tmp` sibling; a crash between
//! steps 2 and 3 leaves the dataset at `.caliper-edit-old` (rename it back by
//! hand); a crash between 3 and 4 leaves a stale `.caliper-edit-old` copy.
//! Ops refuse to run while either sibling exists, so a crashed edit is always
//! surfaced instead of silently clobbered.
//!
//! # What the rewrite preserves — and what it doesn't
//!
//! Preserved: fps, `robot_type` (including `null`), feature set
//! (dims/element names/per-feature fps), per-frame values and timestamps,
//! per-frame `task_index` semantics, `data_files_size_in_mb` / `chunks_size`,
//! and — via a raw-JSON merge, since [`Info`] itself **drops unknown fields**
//! on deserialize — any unknown top-level `info.json` keys. Regenerated:
//! episode/index numbering (dense), `meta/tasks.parquet` (unused tasks
//! dropped, indices remapped first-seen), all per-episode stats bookkeeping in
//! `meta/episodes`, aggregated `meta/stats.json`, totals, and `splits` (reset
//! to `train: 0:N`). Image (`dtype: "image"`) features ride along
//! byte-for-byte: episode transforms are index-level, so each surviving
//! frame's encoded PNG streams through the writer verbatim (re-validated by
//! decode on the way; image stats are recomputed with everything else). Not
//! supported (rejected loudly, dataset untouched): any other feature layout —
//! notably `dtype: "video"` datasets, whose frames live outside the data
//! parquet where an index-level rewrite cannot carry them.
//!
//! # Timestamp semantics
//!
//! v3.0 stores one `timestamp` per frame that **restarts per episode**
//! (lerobot's writer records `frame_index / fps`; delta-timestamp windowing
//! compares values within one episode only). The ops keep that invariant:
//! deletion copies timestamps verbatim; the second half of a **split** is
//! rebased to start at 0; the second episode of a **merge** is rebased to
//! continue `1/fps` after the first (for a continuous-clock source recorded
//! at exactly `1/fps` cadence this reproduces the original values).
//!
//! # Tags sidecar
//!
//! Free-form per-episode tags live in `meta/caliper_tags.json` — a caliper
//! extension. lerobot 0.4.4 never globs `meta/` (it loads `info.json`,
//! `stats.json`, `tasks.parquet` and `episodes/*/*.parquet` by exact path), so
//! the extra file is invisible to it; the oracle proves a tagged dataset still
//! loads. Edit ops remap tag keys alongside episode renumbering.

use crate::Error;
use crate::meta::Info;
use crate::reader::DatasetReader;
use crate::writer::{DatasetSpec, DatasetWriter, FeatureSpec, RESERVED_FEATURES};
use std::collections::BTreeMap;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

/// Dataset-relative path of the tags sidecar.
pub const TAGS_FILE: &str = "meta/caliper_tags.json";

// ===== tags sidecar =====

/// Read `meta/caliper_tags.json` → episode index → tags. Missing file = no
/// tags (returns an empty map); a present-but-malformed file is an error.
pub fn read_tags(root: impl AsRef<Path>) -> Result<BTreeMap<u64, Vec<String>>, Error> {
    let path = root.as_ref().join(TAGS_FILE);
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let raw = fs::read_to_string(&path)?;
    // JSON object keys are strings; parse them back to episode indices.
    let map: BTreeMap<String, Vec<String>> = serde_json::from_str(&raw)?;
    let mut out = BTreeMap::new();
    for (k, v) in map {
        let idx: u64 = k
            .parse()
            .map_err(|_| Error::Format(format!("{TAGS_FILE}: non-integer episode key '{k}'")))?;
        out.insert(idx, v);
    }
    Ok(out)
}

/// Write the tags sidecar (episodes with an empty tag list are omitted).
/// `root` must already be a dataset (`meta/` must exist).
pub fn write_tags(root: impl AsRef<Path>, tags: &BTreeMap<u64, Vec<String>>) -> Result<(), Error> {
    let meta = root.as_ref().join("meta");
    if !meta.is_dir() {
        return Err(Error::Format(format!(
            "{} is not a dataset (no meta/ directory)",
            root.as_ref().display()
        )));
    }
    let map: BTreeMap<String, &Vec<String>> = tags
        .iter()
        .filter(|(_, v)| !v.is_empty())
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    fs::write(
        root.as_ref().join(TAGS_FILE),
        serde_json::to_string_pretty(&map)?,
    )?;
    Ok(())
}

// ===== edit operations =====

/// Remove `episodes` (positions in the reader's episode order) and renumber
/// the survivors densely: `episode_index`, `index`/frame offsets and
/// `task_index` are all rewritten, now-unused tasks are dropped from
/// `meta/tasks.parquet`, per-episode + aggregated stats are recomputed, and
/// tags are remapped. Deleting every episode is refused (an episode-less
/// dataset is not lerobot-loadable).
pub fn delete_episodes(root: impl AsRef<Path>, episodes: &[usize]) -> Result<(), Error> {
    let root = dataset_root(root.as_ref())?;
    let reader = DatasetReader::open(&root)?;
    let n = reader.total_episodes();
    if episodes.is_empty() {
        return Err(Error::Edit("no episodes to delete".into()));
    }
    let mut drop: Vec<usize> = episodes.to_vec();
    drop.sort_unstable();
    drop.dedup();
    if let Some(&bad) = drop.iter().find(|&&e| e >= n) {
        return Err(Error::Edit(format!("episode {bad} of {n}")));
    }
    if drop.len() == n {
        return Err(Error::Edit(
            "refusing to delete every episode (the result would not be loadable)".into(),
        ));
    }
    let survivors: Vec<usize> = (0..n).filter(|e| !drop.contains(e)).collect();
    let plan: Vec<PlannedEpisode> = survivors
        .iter()
        .map(|&e| PlannedEpisode {
            slices: vec![Slice::full(e, &reader)],
            tasks: reader.episodes()[e].tasks.clone(),
        })
        .collect();
    let old_tags = read_tags(&root)?;
    let new_tags: BTreeMap<u64, Vec<String>> = survivors
        .iter()
        .enumerate()
        .filter_map(|(new, &old)| old_tags.get(&(old as u64)).map(|t| (new as u64, t.clone())))
        .collect();
    rewrite(&root, &reader, &plan, &new_tags)
}

/// Split episode `episode` into two at local frame `frame` (`0 < frame < len`):
/// frames `[0, frame)` and `[frame, len)` become adjacent episodes, both
/// keeping the original episode's tasks and tags; the second half's
/// timestamps are rebased to start at 0. Later episodes shift up by one.
pub fn split_episode(root: impl AsRef<Path>, episode: usize, frame: usize) -> Result<(), Error> {
    let root = dataset_root(root.as_ref())?;
    let reader = DatasetReader::open(&root)?;
    let n = reader.total_episodes();
    if episode >= n {
        return Err(Error::Edit(format!("episode {episode} of {n}")));
    }
    let len = reader.episodes()[episode].length as usize;
    if frame == 0 || frame >= len {
        return Err(Error::Edit(format!(
            "split frame must be in (0, {len}), got {frame}"
        )));
    }
    let mut plan = Vec::with_capacity(n + 1);
    for e in 0..n {
        let tasks = reader.episodes()[e].tasks.clone();
        if e == episode {
            plan.push(PlannedEpisode {
                slices: vec![Slice::new(e, 0..frame)],
                tasks: tasks.clone(),
            });
            plan.push(PlannedEpisode {
                slices: vec![Slice::new(e, frame..len)],
                tasks,
            });
        } else {
            plan.push(PlannedEpisode {
                slices: vec![Slice::full(e, &reader)],
                tasks,
            });
        }
    }
    let old_tags = read_tags(&root)?;
    let mut new_tags = BTreeMap::new();
    for (old, tags) in &old_tags {
        let old = *old as usize;
        if old < episode {
            new_tags.insert(old as u64, tags.clone());
        } else if old == episode {
            // both halves inherit the split episode's tags
            new_tags.insert(old as u64, tags.clone());
            new_tags.insert(old as u64 + 1, tags.clone());
        } else {
            new_tags.insert(old as u64 + 1, tags.clone());
        }
    }
    rewrite(&root, &reader, &plan, &new_tags)
}

/// Merge two **adjacent** episodes (`second == first + 1`) into one. The
/// merged episode keeps `first`'s task list, unioned with `second`'s when they
/// differ (each frame keeps its own `task_index`); `second`'s timestamps are
/// rebased to continue `1/fps` after `first`'s last frame, keeping the
/// per-episode-relative timestamp invariant lerobot expects. Tags are the
/// union of both episodes'; later episodes shift down by one.
pub fn merge_episodes(root: impl AsRef<Path>, first: usize, second: usize) -> Result<(), Error> {
    let root = dataset_root(root.as_ref())?;
    let reader = DatasetReader::open(&root)?;
    let n = reader.total_episodes();
    if first >= n || second >= n {
        return Err(Error::Edit(format!("episode {} of {n}", first.max(second))));
    }
    if second != first + 1 {
        return Err(Error::Edit(format!(
            "episodes {first} and {second} are not adjacent (need second == first + 1)"
        )));
    }
    let mut plan = Vec::with_capacity(n - 1);
    for e in 0..n {
        if e == second {
            continue;
        }
        if e == first {
            let mut tasks = reader.episodes()[first].tasks.clone();
            for t in &reader.episodes()[second].tasks {
                if !tasks.contains(t) {
                    tasks.push(t.clone());
                }
            }
            plan.push(PlannedEpisode {
                slices: vec![Slice::full(first, &reader), Slice::full(second, &reader)],
                tasks,
            });
        } else {
            plan.push(PlannedEpisode {
                slices: vec![Slice::full(e, &reader)],
                tasks: reader.episodes()[e].tasks.clone(),
            });
        }
    }
    let old_tags = read_tags(&root)?;
    let mut new_tags = BTreeMap::new();
    for (old, tags) in &old_tags {
        let old = *old as usize;
        if old < second {
            let entry: &mut Vec<String> = new_tags.entry(old.min(first) as u64).or_default();
            // `first` may collect from both `first` and `second` (old == first
            // stays at `first`; old == second folds into it below).
            for t in tags {
                if !entry.contains(t) {
                    entry.push(t.clone());
                }
            }
        } else if old == second {
            let entry: &mut Vec<String> = new_tags.entry(first as u64).or_default();
            for t in tags {
                if !entry.contains(t) {
                    entry.push(t.clone());
                }
            }
        } else {
            new_tags.insert(old as u64 - 1, tags.clone());
        }
    }
    rewrite(&root, &reader, &plan, &new_tags)
}

// ===== rewrite plumbing =====

/// One contiguous run of source frames feeding a new episode.
struct Slice {
    episode: usize,
    range: Range<usize>,
}

impl Slice {
    fn new(episode: usize, range: Range<usize>) -> Self {
        Self { episode, range }
    }
    fn full(episode: usize, reader: &DatasetReader) -> Self {
        Self {
            episode,
            range: 0..reader.episodes()[episode].length as usize,
        }
    }
}

/// One episode of the rewritten dataset: source slices in order + its
/// episode-level task list.
struct PlannedEpisode {
    slices: Vec<Slice>,
    tasks: Vec<String>,
}

/// Canonicalize + sanity-check the dataset root.
fn dataset_root(root: &Path) -> Result<PathBuf, Error> {
    let canon = fs::canonicalize(root)
        .map_err(|e| Error::Format(format!("cannot resolve {}: {e}", root.display())))?;
    if !canon.is_dir() {
        return Err(Error::Format(format!(
            "{} is not a directory",
            canon.display()
        )));
    }
    Ok(canon)
}

/// Rebuild `FeatureSpec`s from `info.json`. Editable features are flat
/// `float32` vectors and `dtype: "image"` camera features (whose PNG bytes
/// stream through the rewrite verbatim); anything else (video, multi-dim
/// arrays) is rejected loudly — the rewrite would silently drop it otherwise.
/// Vector element names survive when they are a plain string list
/// (`names: null` otherwise).
fn feature_specs(info: &Info) -> Result<Vec<FeatureSpec>, Error> {
    let mut specs = Vec::new();
    for (name, f) in &info.features {
        if RESERVED_FEATURES.contains(&name.as_str()) {
            continue;
        }
        if f.dtype == "image" {
            let &[h, w, c] = &f.shape[..] else {
                return Err(Error::Edit(format!(
                    "image feature '{name}' has shape {:?} — expected (height, width, channels)",
                    f.shape
                )));
            };
            specs.push(FeatureSpec::image(
                name.clone(),
                h as usize,
                w as usize,
                c as usize,
            ));
            continue;
        }
        if f.dtype != "float32" || f.shape.len() != 1 {
            return Err(Error::Edit(format!(
                "feature '{name}' is dtype '{}' shape {:?} — edit ops only support flat float32 \
                 vector and image features (video datasets are not editable)",
                f.dtype, f.shape
            )));
        }
        let names: Option<Vec<String>> = f.names.as_array().and_then(|a| {
            a.iter()
                .map(|v| v.as_str().map(String::from))
                .collect::<Option<Vec<_>>>()
        });
        specs.push(FeatureSpec::vector(
            name.clone(),
            f.shape[0] as usize,
            names,
        ));
    }
    if specs.is_empty() {
        return Err(Error::Edit("dataset declares no user features".into()));
    }
    Ok(specs)
}

/// Stream `plan` through a fresh writer in a sibling temp dir, then atomically
/// swap it in (see the module docs for the exact crash story).
fn rewrite(
    root: &Path,
    reader: &DatasetReader,
    plan: &[PlannedEpisode],
    tags: &BTreeMap<u64, Vec<String>>,
) -> Result<(), Error> {
    let info = reader.info();
    let specs = feature_specs(info)?;
    let name = root
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| Error::Format(format!("bad dataset path {}", root.display())))?;
    let parent = root
        .parent()
        .ok_or_else(|| Error::Format(format!("dataset {} has no parent dir", root.display())))?;
    let tmp = parent.join(format!("{name}.caliper-edit-tmp"));
    let old = parent.join(format!("{name}.caliper-edit-old"));
    for leftover in [&tmp, &old] {
        if leftover.exists() {
            return Err(Error::Edit(format!(
                "{} exists — leftover of a crashed edit; inspect/remove it first",
                leftover.display()
            )));
        }
    }

    let result = build_into(&tmp, reader, info, specs, plan, tags);
    if let Err(e) = result {
        let _ = fs::remove_dir_all(&tmp); // best-effort cleanup; original untouched
        return Err(e);
    }

    // Swap. The original is untouched until this first rename.
    fs::rename(root, &old)?;
    if let Err(e) = fs::rename(&tmp, root) {
        // Try to restore the original so the failure is not destructive.
        let _ = fs::rename(&old, root);
        let _ = fs::remove_dir_all(&tmp);
        return Err(e.into());
    }
    fs::remove_dir_all(&old)?;
    Ok(())
}

/// Write the whole transformed dataset into `tmp` (no swap here).
fn build_into(
    tmp: &Path,
    reader: &DatasetReader,
    info: &Info,
    specs: Vec<FeatureSpec>,
    plan: &[PlannedEpisode],
    tags: &BTreeMap<u64, Vec<String>>,
) -> Result<(), Error> {
    let fps = info.fps;
    let vector_names: Vec<String> = specs
        .iter()
        .filter(|s| !s.is_image())
        .map(|s| s.name.clone())
        .collect();
    let image_names: Vec<String> = specs
        .iter()
        .filter(|s| s.is_image())
        .map(|s| s.name.clone())
        .collect();
    let mut spec = DatasetSpec::new(fps, info.robot_type.clone().unwrap_or_default(), specs);
    spec.data_files_size_in_mb = info.data_files_size_in_mb;
    spec.chunks_size = info.chunks_size;
    let mut writer = DatasetWriter::create(tmp, spec)?;

    let all_tasks = reader.tasks();
    for planned in plan {
        let mut frame_tasks: Vec<String> = Vec::new();
        // Timestamp of the last frame appended for THIS new episode, used to
        // rebase follow-on slices (merge) to continue at 1/fps cadence.
        let mut last_t: Option<f64> = None;
        for (slice_no, slice) in planned.slices.iter().enumerate() {
            let ep = reader.read_episode(slice.episode)?;
            if slice.range.end > ep.len() {
                return Err(Error::Edit(format!(
                    "slice {:?} out of range for episode {} (len {})",
                    slice.range,
                    slice.episode,
                    ep.len()
                )));
            }
            // Rebase rule (see module docs): the first slice keeps its
            // timestamps unless it starts mid-episode (split's second half →
            // start at 0); later slices continue after the previous one.
            let base = ep.timestamps[slice.range.start];
            let offset = match (slice_no, slice.range.start) {
                (0, 0) => 0.0,
                (0, _) => -base,
                _ => last_t.expect("previous slice appended frames") + 1.0 / f64::from(fps) - base,
            };
            for i in slice.range.clone() {
                let t = ep.timestamps[i] + offset;
                let mut values: Vec<(&str, &[f64])> = Vec::with_capacity(vector_names.len());
                let mut rows: Vec<Vec<f64>> = Vec::with_capacity(vector_names.len());
                for name in &vector_names {
                    let feat = ep.features.get(name).ok_or_else(|| {
                        Error::Format(format!(
                            "episode {} is missing feature '{name}'",
                            slice.episode
                        ))
                    })?;
                    rows.push(feat[i].iter().map(|&x| f64::from(x)).collect());
                }
                for (name, row) in vector_names.iter().zip(&rows) {
                    values.push((name.as_str(), row.as_slice()));
                }
                // Image bytes ride along untouched: episode transforms are
                // index-level, so frame `i`'s PNG is appended verbatim (the
                // writer re-validates each frame's decode against the spec).
                let mut images: Vec<(&str, &[u8])> = Vec::with_capacity(image_names.len());
                for name in &image_names {
                    let png = ep
                        .images
                        .get(name)
                        .and_then(|frames| frames.get(i))
                        .ok_or_else(|| {
                            Error::Format(format!(
                                "episode {} is missing image feature '{name}' frame {i}",
                                slice.episode
                            ))
                        })?;
                    images.push((name.as_str(), png.as_slice()));
                }
                writer.add_frame_at_with_images(&values, &images, t)?;
                let ti = usize::try_from(ep.task_indices[i])
                    .ok()
                    .filter(|&x| x < all_tasks.len());
                let task = ti.map(|x| all_tasks[x].clone()).ok_or_else(|| {
                    Error::Format(format!(
                        "episode {}: frame {i} has task_index {} outside tasks.parquet ({})",
                        slice.episode,
                        ep.task_indices[i],
                        all_tasks.len()
                    ))
                })?;
                frame_tasks.push(task);
                last_t = Some(t);
            }
        }
        writer.save_episode_with_tasks(&planned.tasks, &frame_tasks)?;
    }
    writer.finalize()?;

    preserve_unknown_info_fields(tmp, reader)?;
    write_tags(tmp, tags)?;
    Ok(())
}

/// [`Info`] drops unknown `info.json` fields on deserialize, so preserving
/// them goes through raw JSON: any top-level key present in the source file
/// but absent from the rewritten one is copied over, and `robot_type` is
/// copied verbatim (the writer cannot represent `null`). Everything the
/// writer regenerates (totals, splits, data_path, features) is kept from the
/// rewrite.
fn preserve_unknown_info_fields(tmp: &Path, reader: &DatasetReader) -> Result<(), Error> {
    let src_raw = fs::read_to_string(reader.root().join("meta/info.json"))?;
    let src: serde_json::Value = serde_json::from_str(&src_raw)?;
    let dst_path = tmp.join("meta/info.json");
    let mut dst: serde_json::Value = serde_json::from_str(&fs::read_to_string(&dst_path)?)?;
    let (Some(src_map), Some(dst_map)) = (src.as_object(), dst.as_object_mut()) else {
        return Ok(());
    };
    for (k, v) in src_map {
        if k == "robot_type" || !dst_map.contains_key(k) {
            dst_map.insert(k.clone(), v.clone());
        }
    }
    fs::write(&dst_path, serde_json::to_string_pretty(&dst)?)?;
    Ok(())
}
