//! The **dataset doctor** — pre-train diagnostics over a native
//! LeRobotDataset v3.0. Every check targets a failure mode that is invisible
//! at record time, silent during training, and fatal to the resulting policy:
//! stale `meta/stats.json` (the normalization killer), dead joints,
//! contradictory demonstrations, echo/lag action labels, coverage holes,
//! frozen tails, dead cameras, accidental double-records.
//!
//! Entry point: [`analyze`] → [`DataReport`], a serde-serializable list of
//! [`Finding`]s (stable `D0xx` codes, plain-English messages naming the
//! feature/episode/dof and the consequence for training, plus a fix hint) with
//! [`DataReport::render_text`] for humans.
//!
//! # Streaming and determinism
//!
//! The analyzer makes **two streaming passes**, reading one episode at a time
//! through [`DatasetReader`] — the full dataset is never resident:
//!
//! 1. running per-dof stats (Welford), pairwise-correlation sums, per-episode
//!    anomaly checks (timestamps, frozen tail), episode content signatures,
//!    image-frame diagnostics, and a seeded reservoir subsample of
//!    (state, action) frames;
//! 2. per-dof histograms and action-saturation counts, which need the pass-1
//!    min/max as bin bounds.
//!
//! All subsampling is driven by a splitmix64 RNG seeded from
//! [`AnalyzeOptions::seed`], so a report is a pure function of the dataset
//! bytes and the options.
//!
//! # Feature-name conventions
//!
//! Checks that relate actions to observations key off lerobot's conventional
//! names: the feature `"action"` and the feature `"observation.state"`. When
//! either is absent those checks (echo, tiny-range, contradiction sampling)
//! are skipped; everything per-feature runs on every `float32` vector feature.

use crate::Error;
use crate::reader::DatasetReader;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

/// lerobot's conventional proprioception feature name.
const STATE_FEATURE: &str = "observation.state";
/// lerobot's conventional action feature name.
const ACTION_FEATURE: &str = "action";
/// Pairwise correlation is only accumulated up to this many dofs per feature
/// (the sums grow as dim²; robot vector features are far below this).
const MAX_CORR_DIM: usize = 32;
/// Cap on emitted cross-episode duplicate-pair findings.
const MAX_DUP_PAIRS: usize = 8;

/// How bad a [`Finding`] is for training.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// Training on this data is broken or actively poisoned.
    Error,
    /// Training will likely produce a degraded or misbehaving policy.
    Warning,
    /// Worth knowing; may be intentional.
    Info,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Info => "info",
        })
    }
}

/// One diagnostic result. `code` is stable (`D001`…) so callers can filter;
/// `message` names the feature/episode/dof concerned and states the
/// consequence for training; `fix_hint` says what to do about it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Finding {
    pub code: String,
    pub severity: Severity,
    /// Feature the finding is about, when it concerns one.
    pub feature: Option<String>,
    /// `episode_index` the finding is about, when it concerns one.
    pub episode: Option<i64>,
    /// Feature element (dof) the finding is about, when it concerns one.
    pub dof: Option<usize>,
    pub message: String,
    pub fix_hint: String,
}

/// Recomputed whole-dataset statistics of one `float32` vector feature —
/// exposed so callers (and tests) can cross-check the analyzer's arithmetic
/// against ground truth.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureSummary {
    pub dim: usize,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    /// Fraction of histogram bins visited per dof (from pass 2). Empty when
    /// the histogram pass was skipped (too few frames); a degenerate dof
    /// (zero range) reports 1.0 — its single value trivially covers its span.
    pub bin_occupancy: Vec<f64>,
}

/// The dataset doctor's report. Serializes to JSON via serde;
/// [`render_text`](Self::render_text) produces the human version.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataReport {
    pub root: String,
    pub total_episodes: usize,
    pub total_frames: u64,
    pub fps: u32,
    /// Recomputed stats per `float32` vector feature.
    pub features: BTreeMap<String, FeatureSummary>,
    /// Sorted most-severe first, then by code / feature / episode / dof.
    pub findings: Vec<Finding>,
}

impl DataReport {
    /// Human-readable rendering of the report.
    pub fn render_text(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "dataset doctor — {}", self.root);
        let _ = writeln!(
            s,
            "episodes: {}, frames: {}, fps: {}",
            self.total_episodes, self.total_frames, self.fps
        );
        let count = |sev: Severity| self.findings.iter().filter(|f| f.severity == sev).count();
        let _ = writeln!(
            s,
            "findings: {} error, {} warning, {} info",
            count(Severity::Error),
            count(Severity::Warning),
            count(Severity::Info)
        );
        if self.findings.is_empty() {
            let _ = writeln!(s, "\nno problems found.");
            return s;
        }
        for f in &self.findings {
            let _ = writeln!(s, "\n[{}] {} — {}", f.severity, f.code, f.message);
            let _ = writeln!(s, "    fix: {}", f.fix_hint);
        }
        s
    }
}

/// Thresholds and knobs for [`analyze`]. `Default` is tuned so that a healthy
/// teleop dataset produces zero findings; every field is a plain number so a
/// caller can tighten or loosen a single check.
#[derive(Clone, Debug)]
pub struct AnalyzeOptions {
    /// D001: a dof whose whole-dataset std is below this never moves.
    pub eps_std: f64,
    /// D002: relative tolerance for `meta/stats.json` vs recomputed mean/std.
    pub stats_rtol: f64,
    /// D002: absolute tolerance floor for the same comparison.
    pub stats_atol: f64,
    /// D003: fraction of frames pinned at one value (min, max, or a single
    /// histogram bin) above which an action dof counts as saturated/collapsed.
    pub saturation_frac: f64,
    /// D004: `rms(action - state) / rms_std(state)` below this = echo labels.
    pub echo_rms_ratio: f64,
    /// D005: action-dof range below this fraction of the median state-dof
    /// range = numerically tiny actions.
    pub tiny_action_ratio: f64,
    /// D006: reservoir size for the contradiction subsample.
    pub knn_samples: usize,
    /// D006: normalized state RMS distance below which two frames count as
    /// "the same situation".
    pub knn_state_eps: f64,
    /// D006: normalized action RMS distance above which same-situation frames
    /// count as contradictory.
    pub knn_action_min: f64,
    /// D006: emit at most this many worst contradictory pairs.
    pub max_pair_findings: usize,
    /// D007: histogram bins per dof.
    pub coverage_bins: usize,
    /// D007: flag a dof visiting fewer than this fraction of its bins.
    pub coverage_min_occupancy: f64,
    /// D008: mean |pairwise correlation| above this = corridor-shaped data.
    pub corridor_min_abs_corr: f64,
    /// D009: robust (MAD-based) z-score above which an episode length is an
    /// outlier.
    pub length_outlier_z: f64,
    /// D010: relative deviation of a frame-to-frame dt from `1/fps` above
    /// which timestamps count as irregular.
    pub timestamp_rtol: f64,
    /// D011: episode tail length (frames) checked for being frozen. Values
    /// below 2 disable the check.
    pub frozen_tail_frames: usize,
    /// D013: fraction of consecutive duplicate image frames above which an
    /// episode's camera stream is flagged.
    pub dup_frame_frac: f64,
    /// D014: total start→end brightness drift (on the [0, 1] scale) above
    /// which an episode's camera stream is flagged.
    pub brightness_drift: f64,
    /// Seed for the deterministic reservoir subsample.
    pub seed: u64,
}

impl Default for AnalyzeOptions {
    fn default() -> Self {
        Self {
            eps_std: 1e-8,
            stats_rtol: 1e-4,
            stats_atol: 1e-6,
            saturation_frac: 0.5,
            echo_rms_ratio: 0.05,
            tiny_action_ratio: 0.01,
            knn_samples: 256,
            knn_state_eps: 0.1,
            knn_action_min: 1.5,
            max_pair_findings: 3,
            coverage_bins: 20,
            coverage_min_occupancy: 0.5,
            corridor_min_abs_corr: 0.95,
            length_outlier_z: 4.0,
            timestamp_rtol: 0.05,
            frozen_tail_frames: 5,
            dup_frame_frac: 0.25,
            brightness_drift: 0.25,
            seed: 0xCA11_9E12,
        }
    }
}

/// Run every diagnostic over the dataset at `root`. See the [module
/// docs](self) for the check catalog and the streaming/determinism contract.
pub fn analyze(root: impl AsRef<Path>, opts: AnalyzeOptions) -> Result<DataReport, Error> {
    let reader = DatasetReader::open(root.as_ref())?;
    let mut a = Analyzer::new(&reader, &opts)?;
    a.pass1()?;
    a.pass2()?;
    a.evaluate()?;
    Ok(a.into_report(root.as_ref()))
}

// ===== internals =====

/// Streaming population mean/variance/min/max of one dof (Welford).
#[derive(Clone)]
struct Welford {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl Welford {
    fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn push(&mut self, v: f64) {
        self.n += 1;
        let d = v - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (v - self.mean);
        if v < self.min {
            self.min = v;
        }
        if v > self.max {
            self.max = v;
        }
    }

    fn std(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            (self.m2 / self.n as f64).sqrt()
        }
    }

    fn range(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.max - self.min
        }
    }
}

/// splitmix64 — the same tiny deterministic generator the rest of the
/// workspace uses for seeded pseudo-randomness.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn fnv1a(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;

/// Whole-dataset accumulators of one `float32` vector feature.
struct FeatAcc {
    dim: usize,
    /// Element names from `info.json` (empty when `names` is null/non-list).
    names: Vec<String>,
    dims: Vec<Welford>,
    /// Σ xᵢ·xⱼ for the upper triangle (i < j), empty when dim is 1 or above
    /// [`MAX_CORR_DIM`].
    cross: Vec<f64>,
    frames: u64,
}

impl FeatAcc {
    fn cross_idx(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < j);
        i * self.dim - i * (i + 1) / 2 + (j - i - 1)
    }
}

/// One reservoir-sampled frame for the contradiction check.
struct Sample {
    episode: i64,
    frame: usize,
    state: Vec<f32>,
    action: Vec<f32>,
}

/// Decoded-frame digest for the image checks.
struct FrameLook {
    /// Mean over color channels (alpha excluded), [0, 1] scale.
    brightness: f64,
    /// Min/max color sample of the frame (0..=255).
    min: u8,
    max: u8,
    /// Hash of the full decoded buffer (duplicate detection).
    hash: u64,
}

struct Analyzer<'a> {
    reader: &'a DatasetReader,
    opts: &'a AnalyzeOptions,
    findings: Vec<Finding>,
    feats: BTreeMap<String, FeatAcc>,
    /// Episode lengths by reader position.
    lengths: Vec<u64>,
    /// Episode content signature by reader position (see `episode_signature`).
    signatures: Vec<u64>,
    /// Σ (action - state)² and its value count, when dims match.
    echo_sumsq: f64,
    echo_n: u64,
    samples: Vec<Sample>,
    frames_seen: u64,
    rng: SplitMix,
    /// Per feature per dof: bins visited + max single-bin count (pass 2).
    histograms: BTreeMap<String, Vec<HistDof>>,
    /// Action-dof saturation counts (pass 2): (at_min, at_max).
    action_saturation: Vec<(u64, u64)>,
    has_state: bool,
    has_action: bool,
}

#[derive(Clone)]
struct HistDof {
    counts: Vec<u64>,
}

impl<'a> Analyzer<'a> {
    fn new(reader: &'a DatasetReader, opts: &'a AnalyzeOptions) -> Result<Self, Error> {
        let mut feats = BTreeMap::new();
        for (name, fi) in &reader.info().features {
            if fi.dtype != "float32" || name == "timestamp" {
                continue;
            }
            let dim = usize::try_from(fi.shape.iter().product::<u64>())
                .map_err(|_| Error::Format(format!("feature '{name}': shape overflows")))?;
            if dim == 0 {
                return Err(Error::Format(format!("feature '{name}': empty shape")));
            }
            let names = fi
                .names
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|v| v.as_str().unwrap_or_default().to_string())
                        .collect::<Vec<_>>()
                })
                .filter(|n| n.len() == dim)
                .unwrap_or_default();
            let cross = if (2..=MAX_CORR_DIM).contains(&dim) {
                vec![0.0; dim * (dim - 1) / 2]
            } else {
                Vec::new()
            };
            feats.insert(
                name.clone(),
                FeatAcc {
                    dim,
                    names,
                    dims: vec![Welford::new(); dim],
                    cross,
                    frames: 0,
                },
            );
        }
        let has_state = feats.contains_key(STATE_FEATURE);
        let has_action = feats.contains_key(ACTION_FEATURE);
        Ok(Self {
            reader,
            opts,
            findings: Vec::new(),
            feats,
            lengths: Vec::new(),
            signatures: Vec::new(),
            echo_sumsq: 0.0,
            echo_n: 0,
            samples: Vec::new(),
            frames_seen: 0,
            rng: SplitMix(opts.seed),
            histograms: BTreeMap::new(),
            action_saturation: Vec::new(),
            has_state,
            has_action,
        })
    }

    // ---- pass 1 ----

    fn pass1(&mut self) -> Result<(), Error> {
        for idx in 0..self.reader.total_episodes() {
            let ep = self.reader.read_episode(idx)?;
            self.lengths.push(ep.len() as u64);
            self.check_timestamps(&ep);
            self.accumulate_features(&ep)?;
            self.check_frozen_tail(&ep);
            self.signatures.push(self.episode_signature(&ep));
            self.sample_frames(&ep);
            self.check_images(&ep);
        }
        Ok(())
    }

    fn accumulate_features(&mut self, ep: &crate::EpisodeData) -> Result<(), Error> {
        for (name, acc) in &mut self.feats {
            let rows = ep.features.get(name).ok_or_else(|| {
                Error::Format(format!(
                    "episode {}: feature '{name}' missing from data",
                    ep.episode_index
                ))
            })?;
            for row in rows {
                if row.len() != acc.dim {
                    return Err(Error::Format(format!(
                        "episode {}: feature '{name}' row has {} values, info.json declares {}",
                        ep.episode_index,
                        row.len(),
                        acc.dim
                    )));
                }
                for (w, &v) in acc.dims.iter_mut().zip(row) {
                    w.push(f64::from(v));
                }
                if !acc.cross.is_empty() {
                    for i in 0..acc.dim {
                        for j in (i + 1)..acc.dim {
                            let k = acc.cross_idx(i, j);
                            acc.cross[k] += f64::from(row[i]) * f64::from(row[j]);
                        }
                    }
                }
                acc.frames += 1;
            }
        }
        // Echo accumulation: action vs state, same frame, when dims match.
        if self.has_state && self.has_action {
            let (s_rows, a_rows) = (&ep.features[STATE_FEATURE], &ep.features[ACTION_FEATURE]);
            for (s, a) in s_rows.iter().zip(a_rows) {
                if s.len() == a.len() {
                    for (&sv, &av) in s.iter().zip(a) {
                        let d = f64::from(av) - f64::from(sv);
                        self.echo_sumsq += d * d;
                        self.echo_n += 1;
                    }
                }
            }
        }
        Ok(())
    }

    /// D010 — timestamp irregularity vs the declared fps.
    fn check_timestamps(&mut self, ep: &crate::EpisodeData) {
        if ep.len() < 2 {
            return;
        }
        let expected = 1.0 / f64::from(self.reader.fps());
        let mut worst: Option<(usize, f64, f64)> = None; // (frame, dt, rel dev)
        for (k, pair) in ep.timestamps.windows(2).enumerate() {
            let dt = pair[1] - pair[0];
            let dev = (dt - expected).abs() / expected;
            if (dt <= 0.0 || dev > self.opts.timestamp_rtol)
                && worst.is_none_or(|(_, _, w)| dev > w)
            {
                worst = Some((k + 1, dt, dev));
            }
        }
        if let Some((frame, dt, _)) = worst {
            self.findings.push(Finding {
                code: "D010".into(),
                severity: Severity::Warning,
                feature: None,
                episode: Some(ep.episode_index),
                dof: None,
                message: format!(
                    "episode {}: timestamps are irregular — worst at frame {frame}, dt = {dt:.6}s \
                     where fps {} implies {expected:.6}s; delta-timestamp windowing and action \
                     chunking will pair misaligned frames",
                    ep.episode_index,
                    self.reader.fps()
                ),
                fix_hint:
                    "re-record with a steady clock, or rewrite timestamps to frame_index/fps \
                           if the frames really are evenly spaced"
                        .into(),
            });
        }
    }

    /// D011 — frozen tail: the last M frames are bit-identical across every
    /// vector feature.
    fn check_frozen_tail(&mut self, ep: &crate::EpisodeData) {
        let m = self.opts.frozen_tail_frames;
        if m < 2 || ep.len() < m || self.feats.is_empty() {
            return;
        }
        let len = ep.len();
        let frozen = self.feats.keys().all(|name| {
            let rows = &ep.features[name];
            let last = &rows[len - 1];
            rows[len - m..len - 1]
                .iter()
                .all(|r| rows_bit_equal(r, last))
        });
        if frozen {
            self.findings.push(Finding {
                code: "D011".into(),
                severity: Severity::Warning,
                feature: None,
                episode: Some(ep.episode_index),
                dof: None,
                message: format!(
                    "episode {}: the last {m} frames are bit-identical across every vector \
                     feature — the robot froze before the recording stopped; the policy will \
                     learn to stall at the end of the task",
                    ep.episode_index
                ),
                fix_hint: "trim the frozen tail (split the episode at the last moving frame and \
                           delete the remainder via the edit ops)"
                    .into(),
            });
        }
    }

    /// Content signature for the cross-episode duplicate check: FNV-1a over
    /// the state sequence (all vector features when no `observation.state`).
    fn episode_signature(&self, ep: &crate::EpisodeData) -> u64 {
        let mut h = FNV_OFFSET;
        fnv1a(&mut h, &(ep.len() as u64).to_le_bytes());
        let keys: Vec<&String> = if self.has_state {
            self.feats.keys().filter(|k| *k == STATE_FEATURE).collect()
        } else {
            self.feats.keys().collect()
        };
        for name in keys {
            fnv1a(&mut h, name.as_bytes());
            for row in &ep.features[name] {
                for &v in row {
                    fnv1a(&mut h, &v.to_bits().to_le_bytes());
                }
            }
        }
        h
    }

    /// Deterministic reservoir subsample of (state, action) frames for the
    /// contradiction check.
    fn sample_frames(&mut self, ep: &crate::EpisodeData) {
        if !(self.has_state && self.has_action) || self.opts.knn_samples == 0 {
            return;
        }
        let (s_rows, a_rows) = (&ep.features[STATE_FEATURE], &ep.features[ACTION_FEATURE]);
        for (frame, (s, a)) in s_rows.iter().zip(a_rows).enumerate() {
            self.frames_seen += 1;
            let sample = || Sample {
                episode: ep.episode_index,
                frame,
                state: s.clone(),
                action: a.clone(),
            };
            if self.samples.len() < self.opts.knn_samples {
                self.samples.push(sample());
            } else {
                let j = (self.rng.next() % self.frames_seen) as usize;
                if j < self.samples.len() {
                    self.samples[j] = sample();
                }
            }
        }
    }

    /// D012/D013/D014 — image-frame diagnostics for every `dtype: "image"`
    /// feature of one episode.
    fn check_images(&mut self, ep: &crate::EpisodeData) {
        for (name, frames) in &ep.images {
            let mut black = 0u64;
            let mut white = 0u64;
            let mut constant = 0u64;
            let mut undecodable = 0u64;
            let mut first_error = String::new();
            let mut dups = 0u64;
            let mut prev_hash: Option<u64> = None;
            let mut brightness = Vec::with_capacity(frames.len());
            for bytes in frames {
                let look = match decode_frame(bytes) {
                    Ok(l) => l,
                    Err(e) => {
                        undecodable += 1;
                        if first_error.is_empty() {
                            first_error = e;
                        }
                        prev_hash = None;
                        continue;
                    }
                };
                if look.max == 0 {
                    black += 1;
                } else if look.min == 255 {
                    white += 1;
                } else if look.min == look.max {
                    constant += 1;
                }
                if prev_hash == Some(look.hash) {
                    dups += 1;
                }
                prev_hash = Some(look.hash);
                brightness.push(look.brightness);
            }
            if undecodable > 0 {
                self.findings.push(Finding {
                    code: "D012".into(),
                    severity: Severity::Error,
                    feature: Some(name.clone()),
                    episode: Some(ep.episode_index),
                    dof: None,
                    message: format!(
                        "episode {} feature '{name}': {undecodable} of {} frames cannot be \
                         decoded ({first_error}) — the dataloader will crash or silently drop \
                         these frames",
                        ep.episode_index,
                        frames.len()
                    ),
                    fix_hint: "re-encode or delete the affected episode; check the camera \
                               pipeline that produced the bytes"
                        .into(),
                });
            }
            if black + white + constant > 0 {
                let what = [
                    (black, "all-black"),
                    (white, "all-white"),
                    (constant, "single-color"),
                ]
                .iter()
                .filter(|(n, _)| *n > 0)
                .map(|(n, w)| format!("{n} {w}"))
                .collect::<Vec<_>>()
                .join(", ");
                self.findings.push(Finding {
                    code: "D012".into(),
                    severity: Severity::Warning,
                    feature: Some(name.clone()),
                    episode: Some(ep.episode_index),
                    dof: None,
                    message: format!(
                        "episode {} feature '{name}': {what} frames out of {} — the camera fed \
                         no usable signal; a vision policy trained on this input is blind here",
                        ep.episode_index,
                        frames.len()
                    ),
                    fix_hint: "check camera power/exposure/connection for this take and delete \
                               the episode if the whole stream is dead"
                        .into(),
                });
            }
            if frames.len() >= 2 {
                let frac = dups as f64 / (frames.len() - 1) as f64;
                if frac > self.opts.dup_frame_frac {
                    self.findings.push(Finding {
                        code: "D013".into(),
                        severity: Severity::Info,
                        feature: Some(name.clone()),
                        episode: Some(ep.episode_index),
                        dof: None,
                        message: format!(
                            "episode {} feature '{name}': {:.0}% of consecutive frames are exact \
                             duplicates — the camera delivered fewer real frames than the \
                             recorded fps claims, so visual dynamics are slower than labeled",
                            ep.episode_index,
                            frac * 100.0
                        ),
                        fix_hint: "check the camera's true frame rate against the dataset fps; \
                                   consider recording at the rate the camera actually sustains"
                            .into(),
                    });
                }
            }
            if brightness.len() >= 2 {
                let drift = ls_slope(&brightness) * (brightness.len() - 1) as f64;
                if drift.abs() > self.opts.brightness_drift {
                    self.findings.push(Finding {
                        code: "D014".into(),
                        severity: Severity::Info,
                        feature: Some(name.clone()),
                        episode: Some(ep.episode_index),
                        dof: None,
                        message: format!(
                            "episode {} feature '{name}': mean brightness drifts by {drift:.2} \
                             (on the 0–1 scale) from start to end — auto-exposure or lighting \
                             changed mid-take; the policy may key on brightness instead of the \
                             scene",
                            ep.episode_index
                        ),
                        fix_hint: "lock camera exposure/white balance and keep lighting constant \
                                   during recording"
                            .into(),
                    });
                }
            }
        }
    }

    // ---- pass 2 ----

    /// Histograms (bounds = pass-1 min/max) and action saturation counts.
    fn pass2(&mut self) -> Result<(), Error> {
        let total: u64 = self.lengths.iter().sum();
        let bins = self.opts.coverage_bins;
        if total == 0 || bins == 0 {
            return Ok(());
        }
        for (name, acc) in &self.feats {
            self.histograms.insert(
                name.clone(),
                vec![
                    HistDof {
                        counts: vec![0; bins]
                    };
                    acc.dim
                ],
            );
        }
        if self.has_action {
            self.action_saturation = vec![(0, 0); self.feats[ACTION_FEATURE].dim];
        }
        for idx in 0..self.reader.total_episodes() {
            let ep = self.reader.read_episode(idx)?;
            for (name, acc) in &self.feats {
                let hist = self.histograms.get_mut(name).expect("pre-inserted");
                let rows = ep.features.get(name).ok_or_else(|| {
                    Error::Format(format!(
                        "episode {}: feature '{name}' missing from data",
                        ep.episode_index
                    ))
                })?;
                for row in rows {
                    for (j, &v) in row.iter().enumerate() {
                        let w = &acc.dims[j];
                        let range = w.range();
                        if range < self.opts.eps_std {
                            continue;
                        }
                        let v = f64::from(v);
                        let bin = (((v - w.min) / range * bins as f64) as usize).min(bins - 1);
                        hist[j].counts[bin] += 1;
                        if name == ACTION_FEATURE {
                            let tol = (range * 1e-6).max(1e-12);
                            if (v - w.min).abs() <= tol {
                                self.action_saturation[j].0 += 1;
                            }
                            if (v - w.max).abs() <= tol {
                                self.action_saturation[j].1 += 1;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // ---- post-pass evaluation ----

    fn evaluate(&mut self) -> Result<(), Error> {
        self.eval_variance_collapse();
        self.eval_stats_json()?;
        self.eval_action_scale();
        self.eval_contradictions();
        self.eval_coverage();
        self.eval_corridor();
        self.eval_length_outliers();
        self.eval_duplicates()?;
        Ok(())
    }

    fn dof_label(acc: &FeatAcc, j: usize) -> String {
        match acc.names.get(j) {
            Some(n) if !n.is_empty() => format!("dof {j} ('{n}')"),
            _ => format!("dof {j}"),
        }
    }

    /// D001 — per-dof variance collapse across the whole dataset.
    fn eval_variance_collapse(&mut self) {
        for (name, acc) in &self.feats {
            if acc.frames < 2 {
                continue;
            }
            for (j, w) in acc.dims.iter().enumerate() {
                if w.std() < self.opts.eps_std {
                    self.findings.push(Finding {
                        code: "D001".into(),
                        severity: Severity::Warning,
                        feature: Some(name.clone()),
                        episode: None,
                        dof: Some(j),
                        message: format!(
                            "feature '{name}' {}: constant at {:.6} across all {} frames — this \
                             joint never moves; the policy will learn to ignore it, and std-based \
                             normalization divides by ~zero",
                            Self::dof_label(acc, j),
                            w.mean,
                            acc.frames
                        ),
                        fix_hint: "verify the recording pipeline actually drives/reads this dof, \
                                   or drop it from the feature"
                            .into(),
                    });
                }
            }
        }
    }

    /// D002 — `meta/stats.json` vs recomputed mean/std (the silent
    /// normalization killer).
    fn eval_stats_json(&mut self) -> Result<(), Error> {
        if self.feats.is_empty() || self.lengths.iter().sum::<u64>() == 0 {
            return Ok(());
        }
        let path = self.reader.root().join("meta/stats.json");
        let raw = match std::fs::read_to_string(&path) {
            Ok(r) => r,
            Err(_) => {
                self.findings.push(Finding {
                    code: "D002".into(),
                    severity: Severity::Error,
                    feature: None,
                    episode: None,
                    dof: None,
                    message: "meta/stats.json is missing — lerobot normalizes with these values; \
                              without them training cannot even start"
                        .to_string(),
                    fix_hint: "regenerate the stats (rewrite the dataset through the caliper edit \
                               ops or lerobot's compute_stats)"
                        .into(),
                });
                return Ok(());
            }
        };
        let stats: serde_json::Value = serde_json::from_str(&raw)?;
        for (name, acc) in &self.feats {
            let Some(entry) = stats.get(name) else {
                self.findings.push(Finding {
                    code: "D002".into(),
                    severity: Severity::Error,
                    feature: Some(name.clone()),
                    episode: None,
                    dof: None,
                    message: format!(
                        "feature '{name}' has no entry in meta/stats.json — normalization will \
                         fail or silently pass raw values through"
                    ),
                    fix_hint: "regenerate meta/stats.json from the data".into(),
                });
                continue;
            };
            for (stat, actual) in [
                (
                    "mean",
                    acc.dims.iter().map(|w| w.mean).collect::<Vec<f64>>(),
                ),
                ("std", acc.dims.iter().map(Welford::std).collect()),
            ] {
                let stored = entry.get(stat).and_then(json_f64_list);
                let Some(stored) = stored else {
                    self.findings.push(Finding {
                        code: "D002".into(),
                        severity: Severity::Error,
                        feature: Some(name.clone()),
                        episode: None,
                        dof: None,
                        message: format!(
                            "feature '{name}': meta/stats.json '{stat}' is missing or not a flat \
                             number list — normalization cannot use it"
                        ),
                        fix_hint: "regenerate meta/stats.json from the data".into(),
                    });
                    continue;
                };
                if stored.len() != acc.dim {
                    self.findings.push(Finding {
                        code: "D002".into(),
                        severity: Severity::Error,
                        feature: Some(name.clone()),
                        episode: None,
                        dof: None,
                        message: format!(
                            "feature '{name}': meta/stats.json '{stat}' has {} entries but the \
                             feature has {} dofs — stats belong to a different schema",
                            stored.len(),
                            acc.dim
                        ),
                        fix_hint: "regenerate meta/stats.json from the data".into(),
                    });
                    continue;
                }
                let worst = stored
                    .iter()
                    .zip(&actual)
                    .enumerate()
                    .map(|(j, (&s, &a))| {
                        let tol =
                            self.opts.stats_atol + self.opts.stats_rtol * s.abs().max(a.abs());
                        (j, s, a, (s - a).abs() - tol)
                    })
                    .max_by(|x, y| x.3.total_cmp(&y.3));
                if let Some((j, stored_v, actual_v, excess)) = worst
                    && excess > 0.0
                {
                    self.findings.push(Finding {
                        code: "D002".into(),
                        severity: Severity::Error,
                        feature: Some(name.clone()),
                        episode: None,
                        dof: Some(j),
                        message: format!(
                            "feature '{name}' {}: meta/stats.json {stat} = {stored_v:.6} but the \
                             data's actual {stat} is {actual_v:.6} — every input is normalized \
                             with the wrong {stat}, silently mis-scaling training",
                            Self::dof_label(acc, j)
                        ),
                        fix_hint: "regenerate meta/stats.json (rewrite through the caliper edit \
                                   ops or lerobot's compute_stats) — stale stats usually mean the \
                                   data was edited without recomputing them"
                            .into(),
                    });
                }
            }
        }
        Ok(())
    }

    /// D003/D004/D005 — action-scale anomalies.
    fn eval_action_scale(&mut self) {
        let Some(action) = self.feats.get(ACTION_FEATURE) else {
            return;
        };
        // D003: saturation at min/max (from pass-2 exact counts) or collapse
        // onto a single histogram bin.
        if action.frames > 0 {
            let n = action.frames as f64;
            for (j, w) in action.dims.iter().enumerate() {
                if w.range() < self.opts.eps_std {
                    continue; // constant dof — D001's business
                }
                let (at_min, at_max) = self.action_saturation.get(j).copied().unwrap_or((0, 0));
                let hist_peak = self
                    .histograms
                    .get(ACTION_FEATURE)
                    .and_then(|h| h.get(j))
                    .and_then(|d| d.counts.iter().copied().max().map(|m| (m, d)));
                let mut msg: Option<String> = None;
                if at_max as f64 / n >= self.opts.saturation_frac {
                    msg = Some(format!(
                        "{:.0}% of frames sit exactly at the maximum ({:.4})",
                        at_max as f64 / n * 100.0,
                        w.max
                    ));
                } else if at_min as f64 / n >= self.opts.saturation_frac {
                    msg = Some(format!(
                        "{:.0}% of frames sit exactly at the minimum ({:.4})",
                        at_min as f64 / n * 100.0,
                        w.min
                    ));
                } else if let Some((peak, dof)) = hist_peak
                    && peak as f64 / n >= self.opts.saturation_frac
                {
                    let bin = dof
                        .counts
                        .iter()
                        .position(|&c| c == peak)
                        .unwrap_or_default();
                    let width = w.range() / dof.counts.len() as f64;
                    msg = Some(format!(
                        "{:.0}% of frames collapse to a single value near {:.4}",
                        peak as f64 / n * 100.0,
                        w.min + (bin as f64 + 0.5) * width
                    ));
                }
                if let Some(msg) = msg {
                    self.findings.push(Finding {
                        code: "D003".into(),
                        severity: Severity::Warning,
                        feature: Some(ACTION_FEATURE.into()),
                        episode: None,
                        dof: Some(j),
                        message: format!(
                            "feature 'action' {}: {msg} — saturated/clipped commands; the policy \
                             mostly sees one label and will slam that value at deployment",
                            Self::dof_label(action, j)
                        ),
                        fix_hint: "check teleop gain and command clipping; re-record with the dof \
                                   actually exercised through its range"
                            .into(),
                    });
                }
            }
        }
        // D004: action ≈ state (echo/lag labels).
        if let Some(state) = self.feats.get(STATE_FEATURE)
            && self.echo_n > 0
        {
            let diff_rms = (self.echo_sumsq / self.echo_n as f64).sqrt();
            let state_var_mean =
                state.dims.iter().map(|w| w.std() * w.std()).sum::<f64>() / state.dim as f64;
            let state_rms = state_var_mean.sqrt();
            if state_rms > self.opts.eps_std && diff_rms < self.opts.echo_rms_ratio * state_rms {
                self.findings.push(Finding {
                    code: "D004".into(),
                    severity: Severity::Warning,
                    feature: Some(ACTION_FEATURE.into()),
                    episode: None,
                    dof: None,
                    message: format!(
                        "'action' is nearly identical to 'observation.state' (rms difference \
                         {diff_rms:.6} vs state spread {state_rms:.4}) — echo/lag labels; the \
                         policy can minimize loss by copying its input and will never move the \
                         robot on its own"
                    ),
                    fix_hint: "if this is position control at high fps, train on delta actions \
                               or targets shifted by one step instead of the raw echo"
                        .into(),
                });
            }
        }
        // D005: action range tiny vs the state ranges.
        if let Some(state) = self.feats.get(STATE_FEATURE)
            && action.frames > 0
        {
            let mut state_ranges: Vec<f64> = state.dims.iter().map(Welford::range).collect();
            state_ranges.sort_by(f64::total_cmp);
            let median_state_range = state_ranges[state_ranges.len() / 2];
            if median_state_range > self.opts.eps_std {
                for (j, w) in action.dims.iter().enumerate() {
                    let range = w.range();
                    if range >= self.opts.eps_std
                        && range < self.opts.tiny_action_ratio * median_state_range
                    {
                        self.findings.push(Finding {
                            code: "D005".into(),
                            severity: Severity::Warning,
                            feature: Some(ACTION_FEATURE.into()),
                            episode: None,
                            dof: Some(j),
                            message: format!(
                                "feature 'action' {}: range {range:.2e} vs a typical state range \
                                 of {median_state_range:.2e} — actions are numerically tiny; \
                                 after normalization, sensor noise dominates the learning signal",
                                Self::dof_label(action, j)
                            ),
                            fix_hint: "check action units/scaling against the state (e.g. rad vs \
                                       deg, normalized vs raw) before training"
                                .into(),
                        });
                    }
                }
            }
        }
    }

    /// D006 — contradictory demos: near-identical states with divergent
    /// actions, over the deterministic reservoir subsample.
    fn eval_contradictions(&mut self) {
        if self.samples.len() < 2 {
            return;
        }
        let (Some(state), Some(action)) = (
            self.feats.get(STATE_FEATURE),
            self.feats.get(ACTION_FEATURE),
        ) else {
            return;
        };
        let s_std: Vec<f64> = state.dims.iter().map(Welford::std).collect();
        let a_std: Vec<f64> = action.dims.iter().map(Welford::std).collect();
        let mut pairs: Vec<(f64, f64, usize, usize)> = Vec::new(); // (a_dist, s_dist, i, j)
        for i in 0..self.samples.len() {
            for j in (i + 1)..self.samples.len() {
                let (si, sj) = (&self.samples[i], &self.samples[j]);
                let Some(s_dist) = norm_dist(&si.state, &sj.state, &s_std, self.opts.eps_std)
                else {
                    continue;
                };
                if s_dist >= self.opts.knn_state_eps {
                    continue;
                }
                let Some(a_dist) = norm_dist(&si.action, &sj.action, &a_std, self.opts.eps_std)
                else {
                    continue;
                };
                if a_dist > self.opts.knn_action_min {
                    pairs.push((a_dist, s_dist, i, j));
                }
            }
        }
        pairs.sort_by(|x, y| y.0.total_cmp(&x.0));
        for &(a_dist, s_dist, i, j) in pairs.iter().take(self.opts.max_pair_findings) {
            let (si, sj) = (&self.samples[i], &self.samples[j]);
            self.findings.push(Finding {
                code: "D006".into(),
                severity: Severity::Warning,
                feature: Some(ACTION_FEATURE.into()),
                episode: Some(si.episode),
                dof: None,
                message: format!(
                    "episode {} frame {} vs episode {} frame {}: states are nearly identical \
                     (normalized distance {s_dist:.3}) but the actions diverge (normalized \
                     distance {a_dist:.2}) — contradictory supervision; behavior cloning \
                     averages these into an action nobody demonstrated",
                    si.episode, si.frame, sj.episode, sj.frame
                ),
                fix_hint: "review the two demonstrations and delete the wrong one, or condition \
                           the policy on the missing context (task/goal) that distinguishes them"
                    .into(),
            });
        }
    }

    /// D007 — per-dof histogram occupancy (coverage holes).
    fn eval_coverage(&mut self) {
        let bins = self.opts.coverage_bins;
        let total: u64 = self.lengths.iter().sum();
        // Occupancy is meaningless when frames can't plausibly fill the bins.
        if bins == 0 || total < 2 * bins as u64 {
            return;
        }
        for (name, acc) in &self.feats {
            let Some(hist) = self.histograms.get(name) else {
                continue;
            };
            for (j, w) in acc.dims.iter().enumerate() {
                if w.range() < self.opts.eps_std {
                    continue;
                }
                let visited = hist[j].counts.iter().filter(|&&c| c > 0).count();
                let occupancy = visited as f64 / bins as f64;
                if occupancy < self.opts.coverage_min_occupancy {
                    self.findings.push(Finding {
                        code: "D007".into(),
                        severity: Severity::Info,
                        feature: Some(name.clone()),
                        episode: None,
                        dof: Some(j),
                        message: format!(
                            "feature '{name}' {}: only {visited} of {bins} bins between min \
                             ({:.4}) and max ({:.4}) are ever visited — coverage holes; the \
                             policy has no data for most of this dof's span and will extrapolate \
                             there",
                            Self::dof_label(acc, j),
                            w.min,
                            w.max
                        ),
                        fix_hint: "collect demonstrations passing through the unvisited region, \
                                   or accept that the policy is undefined there (if this dof is \
                                   intentionally discrete, ignore this)"
                            .into(),
                    });
                }
            }
        }
    }

    /// D008 — corridor-shaped data: all dofs of a feature move in lockstep.
    fn eval_corridor(&mut self) {
        for (name, acc) in &self.feats {
            if acc.cross.is_empty() || acc.frames < 2 {
                continue;
            }
            let n = acc.frames as f64;
            let mut sum_abs = 0.0;
            let mut pairs = 0usize;
            for i in 0..acc.dim {
                for j in (i + 1)..acc.dim {
                    let (wi, wj) = (&acc.dims[i], &acc.dims[j]);
                    if wi.std() < self.opts.eps_std || wj.std() < self.opts.eps_std {
                        continue;
                    }
                    let cov = acc.cross[acc.cross_idx(i, j)] / n - wi.mean * wj.mean;
                    sum_abs += (cov / (wi.std() * wj.std())).abs();
                    pairs += 1;
                }
            }
            if pairs == 0 {
                continue;
            }
            let mean_abs = sum_abs / pairs as f64;
            if mean_abs > self.opts.corridor_min_abs_corr {
                self.findings.push(Finding {
                    code: "D008".into(),
                    severity: Severity::Info,
                    feature: Some(name.clone()),
                    episode: None,
                    dof: None,
                    message: format!(
                        "feature '{name}': mean |pairwise correlation| across its {} dofs is \
                         {mean_abs:.2} — corridor-shaped data (the dofs move in lockstep along \
                         one path); any state off that corridor is out-of-distribution",
                        acc.dim
                    ),
                    fix_hint: "vary the demonstrations (different starts, goals, speeds) so the \
                               dofs decorrelate and the policy sees the workspace, not one line \
                               through it"
                        .into(),
                });
            }
        }
    }

    /// D009 — episode length outliers via robust MAD z-scores.
    fn eval_length_outliers(&mut self) {
        if self.lengths.len() < 4 {
            return;
        }
        let mut sorted = self.lengths.clone();
        sorted.sort_unstable();
        let median = sorted[sorted.len() / 2] as f64;
        let mut devs: Vec<f64> = self
            .lengths
            .iter()
            .map(|&l| (l as f64 - median).abs())
            .collect();
        devs.sort_by(f64::total_cmp);
        let mad = devs[devs.len() / 2];
        // Floor the scale at one frame so identical-length datasets don't
        // divide by zero and flag ±1-frame jitter.
        let scale = (1.4826 * mad).max(1.0);
        for (pos, &len) in self.lengths.iter().enumerate() {
            let z = (len as f64 - median).abs() / scale;
            if z > self.opts.length_outlier_z {
                let episode = self.reader.episodes()[pos].episode_index;
                self.findings.push(Finding {
                    code: "D009".into(),
                    severity: Severity::Info,
                    feature: None,
                    episode: Some(episode),
                    dof: None,
                    message: format!(
                        "episode {episode}: length {len} frames vs a median of {median:.0} \
                         (robust z = {z:.1}) — likely a stuck recording, a concatenated take, or \
                         an aborted demo; it will dominate (or starve) the sampling of its task",
                    ),
                    fix_hint: "review the episode; split or delete it via the edit ops if it is \
                               not one clean demonstration"
                        .into(),
                });
            }
        }
    }

    /// D015 — cross-episode duplicates: identical state sequences.
    fn eval_duplicates(&mut self) -> Result<(), Error> {
        let mut groups: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (pos, &sig) in self.signatures.iter().enumerate() {
            groups.entry(sig).or_default().push(pos);
        }
        let mut emitted = 0usize;
        for positions in groups.values().filter(|p| p.len() > 1) {
            for k in 1..positions.len() {
                if emitted >= MAX_DUP_PAIRS {
                    return Ok(());
                }
                let (a, b) = (positions[0], positions[k]);
                // The signature is a hash — confirm byte equality before
                // accusing anyone of double-recording.
                let ea = self.reader.read_episode(a)?;
                let eb = self.reader.read_episode(b)?;
                let keys: Vec<&String> = if self.has_state {
                    self.feats.keys().filter(|n| *n == STATE_FEATURE).collect()
                } else {
                    self.feats.keys().collect()
                };
                let states_equal = ea.len() == eb.len()
                    && keys.iter().all(|name| {
                        ea.features[*name]
                            .iter()
                            .zip(&eb.features[*name])
                            .all(|(x, y)| rows_bit_equal(x, y))
                    });
                if !states_equal {
                    continue;
                }
                let all_equal = self.feats.keys().all(|name| {
                    ea.features[name]
                        .iter()
                        .zip(&eb.features[name])
                        .all(|(x, y)| rows_bit_equal(x, y))
                });
                let detail = if all_equal {
                    "identical state sequences and identical actions — an accidental \
                     double-record that silently double-weights this demonstration"
                } else {
                    "identical state sequences but different actions — the same trajectory was \
                     labeled twice with conflicting commands"
                };
                self.findings.push(Finding {
                    code: "D015".into(),
                    severity: Severity::Warning,
                    feature: None,
                    episode: Some(ea.episode_index),
                    dof: None,
                    message: format!(
                        "episode {} and episode {}: {detail}",
                        ea.episode_index, eb.episode_index
                    ),
                    fix_hint: "delete one of the copies via the edit ops".into(),
                });
                emitted += 1;
            }
        }
        Ok(())
    }

    // ---- report assembly ----

    fn into_report(mut self, root: &Path) -> DataReport {
        let severity_rank = |s: Severity| match s {
            Severity::Error => 0u8,
            Severity::Warning => 1,
            Severity::Info => 2,
        };
        self.findings.sort_by(|a, b| {
            (
                severity_rank(a.severity),
                &a.code,
                &a.feature,
                a.episode,
                a.dof,
            )
                .cmp(&(
                    severity_rank(b.severity),
                    &b.code,
                    &b.feature,
                    b.episode,
                    b.dof,
                ))
        });
        let bins = self.opts.coverage_bins;
        let features = self
            .feats
            .iter()
            .map(|(name, acc)| {
                let occupancy = self
                    .histograms
                    .get(name)
                    .map(|hist| {
                        acc.dims
                            .iter()
                            .zip(hist)
                            .map(|(w, h)| {
                                if w.range() < self.opts.eps_std {
                                    1.0
                                } else {
                                    h.counts.iter().filter(|&&c| c > 0).count() as f64 / bins as f64
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                (
                    name.clone(),
                    FeatureSummary {
                        dim: acc.dim,
                        mean: acc.dims.iter().map(|w| w.mean).collect(),
                        std: acc.dims.iter().map(Welford::std).collect(),
                        // ±∞ sentinels of an empty accumulator would not
                        // survive JSON serialization — report 0.0 instead.
                        min: acc
                            .dims
                            .iter()
                            .map(|w| if w.n == 0 { 0.0 } else { w.min })
                            .collect(),
                        max: acc
                            .dims
                            .iter()
                            .map(|w| if w.n == 0 { 0.0 } else { w.max })
                            .collect(),
                        bin_occupancy: occupancy,
                    },
                )
            })
            .collect();
        DataReport {
            root: root.display().to_string(),
            total_episodes: self.reader.total_episodes(),
            total_frames: self.lengths.iter().sum(),
            fps: self.reader.fps(),
            features,
            findings: self.findings,
        }
    }
}

/// Bitwise row equality (f32 payloads compared exactly, NaN == NaN).
fn rows_bit_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Per-dim std-normalized RMS distance, skipping degenerate dims. `None` when
/// lengths differ or no dim has usable spread.
fn norm_dist(a: &[f32], b: &[f32], std: &[f64], eps: f64) -> Option<f64> {
    if a.len() != b.len() || a.len() != std.len() {
        return None;
    }
    let mut sum = 0.0;
    let mut used = 0usize;
    for ((&x, &y), &s) in a.iter().zip(b).zip(std) {
        if s < eps {
            continue;
        }
        let d = (f64::from(x) - f64::from(y)) / s;
        sum += d * d;
        used += 1;
    }
    if used == 0 {
        None
    } else {
        Some((sum / used as f64).sqrt())
    }
}

/// Least-squares slope of `ys` against 0..len (per-frame brightness trend).
fn ls_slope(ys: &[f64]) -> f64 {
    let n = ys.len() as f64;
    let xbar = (n - 1.0) / 2.0;
    let ybar = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &y) in ys.iter().enumerate() {
        let dx = i as f64 - xbar;
        num += dx * (y - ybar);
        den += dx * dx;
    }
    if den > 0.0 { num / den } else { 0.0 }
}

fn json_f64_list(v: &serde_json::Value) -> Option<Vec<f64>> {
    v.as_array()?
        .iter()
        .map(serde_json::Value::as_f64)
        .collect()
}

/// Decode one stored image frame into its diagnostic digest. 8-bit frames
/// only (what the writer accepts and lerobot produces); anything else is
/// reported as undecodable by the caller.
fn decode_frame(bytes: &[u8]) -> Result<FrameLook, String> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("not a decodable PNG: {e}"))?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| "PNG output size overflows".to_string())?;
    let mut buf = vec![0u8; size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("PNG decode failed: {e}"))?;
    if info.bit_depth != png::BitDepth::Eight {
        return Err(format!("unsupported bit depth {:?}", info.bit_depth));
    }
    let data = &buf[..info.buffer_size()];
    let samples = info.color_type.samples();
    if samples == 0 || data.is_empty() {
        return Err("empty frame".to_string());
    }
    // Alpha is uniform padding for these checks — measure color channels only.
    let color_ch = match samples {
        2 => 1,
        4 => 3,
        n => n,
    };
    let mut min = u8::MAX;
    let mut max = u8::MIN;
    let mut sum = 0u64;
    let mut count = 0u64;
    for px in data.chunks_exact(samples) {
        for &b in &px[..color_ch] {
            if b < min {
                min = b;
            }
            if b > max {
                max = b;
            }
            sum += u64::from(b);
            count += 1;
        }
    }
    let mut hash = FNV_OFFSET;
    fnv1a(&mut hash, data);
    Ok(FrameLook {
        brightness: sum as f64 / count as f64 / 255.0,
        min,
        max,
        hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_matches_analytic_population_stats() {
        let mut w = Welford::new();
        for v in [1.0, 2.0, 3.0, 4.0] {
            w.push(v);
        }
        assert_eq!(w.mean, 2.5);
        assert!((w.std() - 1.25f64.sqrt()).abs() < 1e-12);
        assert_eq!((w.min, w.max), (1.0, 4.0));
    }

    #[test]
    fn ls_slope_recovers_a_linear_ramp() {
        let ys: Vec<f64> = (0..10).map(|i| 0.5 + 0.03 * i as f64).collect();
        assert!((ls_slope(&ys) - 0.03).abs() < 1e-12);
        assert!(ls_slope(&[0.7; 8]).abs() < 1e-12);
    }

    #[test]
    fn norm_dist_skips_degenerate_dims() {
        // Second dim has zero std → only the first contributes.
        let d = norm_dist(&[1.0, 5.0], &[3.0, 9.0], &[2.0, 0.0], 1e-8).unwrap();
        assert!((d - 1.0).abs() < 1e-12);
        assert!(norm_dist(&[1.0], &[2.0], &[0.0], 1e-8).is_none());
    }

    #[test]
    fn decode_frame_rejects_garbage() {
        assert!(decode_frame(&[0x00, 0x01, 0x02]).is_err());
        assert!(decode_frame(b"not a png at all").is_err());
    }

    #[test]
    fn splitmix_is_deterministic() {
        let mut a = SplitMix(42);
        let mut b = SplitMix(42);
        let seq_a: Vec<u64> = (0..8).map(|_| a.next()).collect();
        let seq_b: Vec<u64> = (0..8).map(|_| b.next()).collect();
        assert_eq!(seq_a, seq_b);
    }
}
