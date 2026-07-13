//! The dataset doctor through the public API: every `D0xx` check gets a
//! positive test (a dataset crafted via the writer WITH the defect) and a
//! negative test (a clean dataset produces zero findings), plus analytic
//! cross-checks of the recomputed numbers, determinism, and serde/text
//! rendering.

use caliper_dataset::{
    AnalyzeOptions, DataReport, DatasetSpec, DatasetWriter, FeatureSpec, Severity, analyze,
};
use std::fs;
use std::path::{Path, PathBuf};

fn tmpdir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("caliper_dataset_dr_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    p
}

/// Deterministic pseudo-random f64 in [-1, 1) — splitmix-style, no deps.
fn noise(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn spec_sa() -> DatasetSpec {
    DatasetSpec::new(
        50,
        "test_bot",
        vec![
            FeatureSpec::vector("observation.state", 2, Some(vec!["j1".into(), "j2".into()])),
            FeatureSpec::vector("action", 2, None),
        ],
    )
}

/// A healthy action label: a smooth, non-echo, decorrelated function of the
/// state (the two mixes have zero covariance for i.i.d. state dims).
fn act(s: [f64; 2]) -> [f64; 2] {
    [0.6 * s[0] + 0.25 * s[1], 0.6 * s[1] - 0.25 * s[0]]
}

fn add(w: &mut DatasetWriter, s: [f64; 2], a: [f64; 2]) {
    w.add_frame(&[("observation.state", &s), ("action", &a)])
        .unwrap();
}

/// A defect-free teleop-style dataset: 6 episodes of slightly varying length,
/// uniform state noise, smooth actions.
fn clean_dataset(dir: &Path) -> PathBuf {
    let mut w = DatasetWriter::create(dir, spec_sa()).unwrap();
    let mut seed = 11;
    for (ep, len) in [38usize, 39, 40, 41, 42, 43].into_iter().enumerate() {
        for _ in 0..len {
            let s = [noise(&mut seed), noise(&mut seed)];
            add(&mut w, s, act(s));
        }
        w.save_episode(if ep % 2 == 0 { "wave" } else { "reach" })
            .unwrap();
    }
    w.finalize().unwrap()
}

fn run(root: &Path) -> DataReport {
    analyze(root, AnalyzeOptions::default()).unwrap()
}

fn has_code(r: &DataReport, code: &str) -> bool {
    r.findings.iter().any(|f| f.code == code)
}

// ===== negative: a clean dataset is a clean bill of health =====

#[test]
fn clean_dataset_yields_zero_findings() {
    let dir = tmpdir("clean");
    let root = clean_dataset(&dir);
    let r = run(&root);
    assert!(r.findings.is_empty(), "{}", r.render_text());
    assert_eq!(r.total_episodes, 6);
    assert_eq!(r.total_frames, 243);
    assert_eq!(r.fps, 50);
    assert!(r.render_text().contains("no problems found"));
}

#[test]
fn empty_dataset_yields_zero_findings() {
    let dir = tmpdir("empty");
    let w = DatasetWriter::create(&dir, spec_sa()).unwrap();
    let root = w.finalize().unwrap();
    let r = run(&root);
    assert!(r.findings.is_empty(), "{}", r.render_text());
    assert_eq!(r.total_frames, 0);
}

// ===== analytic cross-check of the recomputed numbers =====

#[test]
fn recomputed_stats_match_analytic_ground_truth() {
    let dir = tmpdir("analytic");
    let spec = DatasetSpec::new(
        50,
        "test_bot",
        vec![FeatureSpec::vector("observation.state", 1, None)],
    );
    let mut w = DatasetWriter::create(&dir, spec).unwrap();
    for v in [1.0, 2.0, 3.0, 4.0] {
        w.add_frame(&[("observation.state", &[v])]).unwrap();
    }
    w.save_episode("t").unwrap();
    let root = w.finalize().unwrap();
    let r = run(&root);
    assert!(r.findings.is_empty(), "{}", r.render_text());
    let s = &r.features["observation.state"];
    assert!((s.mean[0] - 2.5).abs() < 1e-12);
    // population std of {1,2,3,4} = sqrt(1.25)
    assert!((s.std[0] - 1.25f64.sqrt()).abs() < 1e-12);
    assert_eq!((s.min[0], s.max[0]), (1.0, 4.0));
    // 1→bin 0, 2→bin 6, 3→bin 13, 4→bin 19 of 20: exactly 4 bins visited.
    assert!((s.bin_occupancy[0] - 0.2).abs() < 1e-12);
}

// ===== D001 variance collapse =====

#[test]
fn d001_flags_a_dof_that_never_moves() {
    let dir = tmpdir("d001");
    let mut w = DatasetWriter::create(&dir, spec_sa()).unwrap();
    let mut seed = 3;
    for _ in 0..2 {
        for _ in 0..30 {
            let x = noise(&mut seed);
            // dof 1 is welded at 0.5; actions are a smooth function of x with
            // zero cross-covariance (x vs x²).
            add(&mut w, [x, 0.5], [0.6 * x, 0.5 - 0.4 * x * x]);
        }
        w.save_episode("t").unwrap();
    }
    let root = w.finalize().unwrap();
    let r = run(&root);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "D001")
        .expect("D001 expected");
    assert_eq!(f.feature.as_deref(), Some("observation.state"));
    assert_eq!(f.dof, Some(1));
    assert!(f.message.contains("j2"), "{}", f.message);
    assert!(f.message.contains("never moves"), "{}", f.message);
}

// ===== D002 stats.json mismatch =====

#[test]
fn d002_flags_stale_stats_json() {
    let dir = tmpdir("d002");
    let root = clean_dataset(&dir);
    let path = root.join("meta/stats.json");
    let mut v: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    let old = v["observation.state"]["mean"][0].as_f64().unwrap();
    v["observation.state"]["mean"][0] = serde_json::json!(old + 0.5);
    fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();

    let r = run(&root);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "D002")
        .expect("D002 expected");
    assert_eq!(f.severity, Severity::Error);
    assert_eq!(f.feature.as_deref(), Some("observation.state"));
    assert_eq!(f.dof, Some(0));
    assert!(f.message.contains("mean"), "{}", f.message);
    // Sorted most-severe first: the error leads the report.
    assert_eq!(r.findings[0].code, "D002");
}

// ===== D003 action saturation / collapse =====

#[test]
fn d003_flags_actions_pinned_at_the_maximum() {
    let dir = tmpdir("d003max");
    let mut w = DatasetWriter::create(&dir, spec_sa()).unwrap();
    let mut seed = 5;
    for _ in 0..2 {
        for i in 0..40 {
            let s = [noise(&mut seed), noise(&mut seed)];
            // 90% of dof-0 commands are clipped at exactly 1.0.
            let a0 = if i % 10 < 9 { 1.0 } else { noise(&mut seed) };
            add(&mut w, s, [a0, 0.5 * s[1]]);
        }
        w.save_episode("t").unwrap();
    }
    let root = w.finalize().unwrap();
    let r = run(&root);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "D003")
        .expect("D003 expected");
    assert_eq!(f.feature.as_deref(), Some("action"));
    assert_eq!(f.dof, Some(0));
    assert!(f.message.contains("maximum"), "{}", f.message);
}

#[test]
fn d003_flags_actions_collapsed_to_one_value() {
    let dir = tmpdir("d003zero");
    let mut w = DatasetWriter::create(&dir, spec_sa()).unwrap();
    let mut seed = 6;
    for _ in 0..2 {
        for i in 0..40 {
            let s = [noise(&mut seed), noise(&mut seed)];
            // 90% of dof-0 commands are 0.0 — not an extreme, so only the
            // histogram-peak branch can catch it.
            let a0 = if i % 10 < 9 { 0.0 } else { noise(&mut seed) };
            add(&mut w, s, [a0, 0.5 * s[1]]);
        }
        w.save_episode("t").unwrap();
    }
    let root = w.finalize().unwrap();
    let r = run(&root);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "D003")
        .expect("D003 expected");
    assert!(
        f.message.contains("collapse to a single value"),
        "{}",
        f.message
    );
}

#[test]
fn d003_ignores_saturation_below_threshold() {
    let dir = tmpdir("d003neg");
    let mut w = DatasetWriter::create(&dir, spec_sa()).unwrap();
    let mut seed = 7;
    for _ in 0..2 {
        for i in 0..40 {
            let s = [noise(&mut seed), noise(&mut seed)];
            // Only 30% at the limit — under the 50% default.
            let a0 = if i % 10 < 3 { 1.0 } else { noise(&mut seed) };
            add(&mut w, s, [a0, 0.5 * s[1]]);
        }
        w.save_episode("t").unwrap();
    }
    let root = w.finalize().unwrap();
    let r = run(&root);
    assert!(!has_code(&r, "D003"), "{}", r.render_text());
}

// ===== D004 echo labels =====

#[test]
fn d004_flags_action_that_echoes_the_state() {
    let dir = tmpdir("d004");
    let mut w = DatasetWriter::create(&dir, spec_sa()).unwrap();
    let mut seed = 9;
    for _ in 0..2 {
        for _ in 0..30 {
            let s = [noise(&mut seed), noise(&mut seed)];
            add(&mut w, s, s); // action == state, verbatim
        }
        w.save_episode("t").unwrap();
    }
    let root = w.finalize().unwrap();
    let r = run(&root);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "D004")
        .expect("D004 expected");
    assert!(f.message.contains("copying its input"), "{}", f.message);
}

// ===== D005 tiny action range =====

#[test]
fn d005_flags_numerically_tiny_actions() {
    let dir = tmpdir("d005");
    let mut w = DatasetWriter::create(&dir, spec_sa()).unwrap();
    let mut seed = 13;
    for _ in 0..2 {
        for _ in 0..30 {
            let s = [noise(&mut seed), noise(&mut seed)];
            // dof 0 spans ~0.002 while the state spans ~2 — wrong units.
            add(&mut w, s, [0.001 * s[0], 0.6 * s[1]]);
        }
        w.save_episode("t").unwrap();
    }
    let root = w.finalize().unwrap();
    let r = run(&root);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "D005")
        .expect("D005 expected");
    assert_eq!(f.feature.as_deref(), Some("action"));
    assert_eq!(f.dof, Some(0));
    assert!(f.message.contains("numerically tiny"), "{}", f.message);
}

// ===== D006 contradictory demos =====

fn contradiction_dataset(dir: &Path) -> PathBuf {
    let mut w = DatasetWriter::create(dir, spec_sa()).unwrap();
    let mut seed = 17;
    let states: Vec<[f64; 2]> = (0..40)
        .map(|_| [noise(&mut seed), noise(&mut seed)])
        .collect();
    // Episode 0: near +1 actions everywhere.
    for s in &states {
        add(
            &mut w,
            *s,
            [1.0 + 0.01 * noise(&mut seed), 1.0 + 0.01 * noise(&mut seed)],
        );
    }
    w.save_episode("t").unwrap();
    // Episode 1: the SAME situations (jittered so it is not a byte-duplicate)
    // labeled with the opposite command.
    for s in &states {
        add(
            &mut w,
            [s[0] + 0.001, s[1] + 0.001],
            [
                -1.0 + 0.01 * noise(&mut seed),
                -1.0 + 0.01 * noise(&mut seed),
            ],
        );
    }
    w.save_episode("t").unwrap();
    w.finalize().unwrap()
}

#[test]
fn d006_flags_contradictory_demonstrations() {
    let dir = tmpdir("d006");
    let root = contradiction_dataset(&dir);
    let r = run(&root);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "D006")
        .expect("D006 expected");
    assert!(f.message.contains("nearly identical"), "{}", f.message);
    assert!(f.message.contains("actions diverge"), "{}", f.message);
    // Worst pairs are capped.
    let n = r.findings.iter().filter(|f| f.code == "D006").count();
    assert!(n <= AnalyzeOptions::default().max_pair_findings);
}

#[test]
fn analyze_is_deterministic() {
    let dir = tmpdir("determinism");
    let root = contradiction_dataset(&dir);
    let a = serde_json::to_string(&run(&root)).unwrap();
    let b = serde_json::to_string(&run(&root)).unwrap();
    assert_eq!(a, b);
}

// ===== D007 coverage holes =====

#[test]
fn d007_flags_coverage_holes() {
    let dir = tmpdir("d007");
    let mut w = DatasetWriter::create(&dir, spec_sa()).unwrap();
    let mut seed = 19;
    for ep in 0..3 {
        for i in 0..40 {
            // dof 0 lives in two tight clusters at the ends of its span; the
            // middle 90% of the range is never visited.
            let sign = if (ep + i) % 2 == 0 { 1.0 } else { -1.0 };
            let s = [sign + 0.02 * noise(&mut seed), noise(&mut seed)];
            add(&mut w, s, [0.5 * s[0], 0.5 * s[1]]);
        }
        w.save_episode("t").unwrap();
    }
    let root = w.finalize().unwrap();
    let r = run(&root);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "D007" && f.feature.as_deref() == Some("observation.state"))
        .expect("D007 expected");
    assert_eq!(f.dof, Some(0));
    assert!(f.message.contains("coverage holes"), "{}", f.message);
    let s = &r.features["observation.state"];
    assert!(
        s.bin_occupancy[0] <= 0.2,
        "occupancy {}",
        s.bin_occupancy[0]
    );
    assert!(
        s.bin_occupancy[1] >= 0.5,
        "occupancy {}",
        s.bin_occupancy[1]
    );
}

// ===== D008 corridor-shaped data =====

#[test]
fn d008_flags_lockstep_dofs() {
    let dir = tmpdir("d008");
    let mut w = DatasetWriter::create(&dir, spec_sa()).unwrap();
    let mut seed = 23;
    for _ in 0..2 {
        for _ in 0..40 {
            let x = noise(&mut seed);
            // dof 1 = 2 · dof 0 exactly: the data is a line, not a workspace.
            add(&mut w, [x, 2.0 * x], [0.5 * x, 0.4 - 0.3 * x * x]);
        }
        w.save_episode("t").unwrap();
    }
    let root = w.finalize().unwrap();
    let r = run(&root);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "D008")
        .expect("D008 expected");
    assert_eq!(f.feature.as_deref(), Some("observation.state"));
    assert!(f.message.contains("corridor"), "{}", f.message);
}

// ===== D009 episode length outliers =====

#[test]
fn d009_flags_a_length_outlier() {
    let dir = tmpdir("d009");
    let mut w = DatasetWriter::create(&dir, spec_sa()).unwrap();
    let mut seed = 29;
    for len in [12usize, 12, 12, 12, 12, 12, 90] {
        for _ in 0..len {
            let s = [noise(&mut seed), noise(&mut seed)];
            add(&mut w, s, act(s));
        }
        w.save_episode("t").unwrap();
    }
    let root = w.finalize().unwrap();
    let r = run(&root);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "D009")
        .expect("D009 expected");
    assert_eq!(f.episode, Some(6));
    assert!(f.message.contains("median"), "{}", f.message);
}

// ===== D010 timestamp irregularity =====

#[test]
fn d010_flags_irregular_timestamps() {
    let dir = tmpdir("d010");
    let mut w = DatasetWriter::create(&dir, spec_sa()).unwrap();
    let mut seed = 31;
    for i in 0..20 {
        let s = [noise(&mut seed), noise(&mut seed)];
        let a = act(s);
        // A 0.3 s stall between frames 9 and 10 of a 50 fps stream.
        let t = i as f64 / 50.0 + if i >= 10 { 0.3 } else { 0.0 };
        w.add_frame_at(&[("observation.state", &s), ("action", &a)], t)
            .unwrap();
    }
    w.save_episode("t").unwrap();
    let root = w.finalize().unwrap();
    let r = run(&root);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "D010")
        .expect("D010 expected");
    assert_eq!(f.episode, Some(0));
    assert!(f.message.contains("frame 10"), "{}", f.message);
}

// ===== D011 frozen tail =====

#[test]
fn d011_flags_a_frozen_tail() {
    let dir = tmpdir("d011");
    let mut w = DatasetWriter::create(&dir, spec_sa()).unwrap();
    let mut seed = 37;
    for _ in 0..20 {
        let s = [noise(&mut seed), noise(&mut seed)];
        add(&mut w, s, act(s));
    }
    // The robot freezes: 6 bit-identical frames before the recording stops.
    let s = [0.3, -0.2];
    for _ in 0..6 {
        add(&mut w, s, act(s));
    }
    w.save_episode("t").unwrap();
    let root = w.finalize().unwrap();
    let r = run(&root);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "D011")
        .expect("D011 expected");
    assert_eq!(f.episode, Some(0));
    assert!(f.message.contains("froze"), "{}", f.message);
}

// ===== image checks (D012/D013/D014) =====

const IH: usize = 4;
const IW: usize = 4;
const CAM: &str = "observation.images.cam";

fn spec_img() -> DatasetSpec {
    DatasetSpec::new(
        50,
        "cam_bot",
        vec![
            FeatureSpec::vector("observation.state", 1, None),
            FeatureSpec::image(CAM, IH, IW, 3),
        ],
    )
}

fn png_rgb(px: &[u8]) -> Vec<u8> {
    assert_eq!(px.len(), IH * IW * 3);
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, IW as u32, IH as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().unwrap();
    wr.write_image_data(px).unwrap();
    wr.finish().unwrap();
    out
}

fn add_img(w: &mut DatasetWriter, seed: &mut u64, png: &[u8]) {
    let s = [noise(seed)];
    w.add_frame_with_images(&[("observation.state", &s)], &[(CAM, png)])
        .unwrap();
}

/// Textured, per-frame-varying, brightness-stable pixels for frame `k`.
fn lively_pixels(k: usize) -> Vec<u8> {
    (0..IH * IW * 3)
        .map(|p| (((k * 31 + p * 7) % 191) + 32) as u8)
        .collect()
}

#[test]
fn clean_image_dataset_yields_zero_findings() {
    let dir = tmpdir("imgclean");
    let mut w = DatasetWriter::create(&dir, spec_img()).unwrap();
    let mut seed = 41;
    for ep in 0..2 {
        for i in 0..10 {
            add_img(&mut w, &mut seed, &png_rgb(&lively_pixels(ep * 100 + i)));
        }
        w.save_episode("t").unwrap();
    }
    let root = w.finalize().unwrap();
    let r = run(&root);
    assert!(r.findings.is_empty(), "{}", r.render_text());
}

#[test]
fn d012_flags_black_white_and_constant_frames() {
    let dir = tmpdir("d012");
    let mut w = DatasetWriter::create(&dir, spec_img()).unwrap();
    let mut seed = 43;
    for value in [0u8, 255, 128] {
        let png = png_rgb(&[value; IH * IW * 3]);
        for _ in 0..4 {
            add_img(&mut w, &mut seed, &png);
        }
        w.save_episode("t").unwrap();
    }
    let root = w.finalize().unwrap();
    let r = run(&root);
    for (ep, what) in [(0, "all-black"), (1, "all-white"), (2, "single-color")] {
        let f = r
            .findings
            .iter()
            .find(|f| f.code == "D012" && f.episode == Some(ep))
            .unwrap_or_else(|| panic!("D012 expected for episode {ep}"));
        assert_eq!(f.feature.as_deref(), Some(CAM));
        assert!(f.message.contains(what), "{}", f.message);
    }
}

#[test]
fn d013_flags_duplicated_consecutive_frames() {
    let dir = tmpdir("d013");
    let mut w = DatasetWriter::create(&dir, spec_img()).unwrap();
    let mut seed = 47;
    // A,A,B,B,C,C,D,D — 4 of 7 transitions are duplicates (57% > 25%).
    for i in 0..8 {
        add_img(&mut w, &mut seed, &png_rgb(&lively_pixels(i / 2)));
    }
    w.save_episode("t").unwrap();
    let root = w.finalize().unwrap();
    let r = run(&root);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "D013")
        .expect("D013 expected");
    assert!(f.message.contains("57%"), "{}", f.message);
    assert!(!has_code(&r, "D012"), "{}", r.render_text());
}

#[test]
fn d014_flags_brightness_drift() {
    let dir = tmpdir("d014");
    let mut w = DatasetWriter::create(&dir, spec_img()).unwrap();
    let mut seed = 53;
    for i in 0..10usize {
        // Mean brightness ramps 15 → 195 of 255 (drift ≈ 0.7 on [0, 1]).
        let px: Vec<u8> = (0..IH * IW * 3)
            .map(|p| (15 + i * 20 + (p % 5)) as u8)
            .collect();
        add_img(&mut w, &mut seed, &png_rgb(&px));
    }
    w.save_episode("t").unwrap();
    let root = w.finalize().unwrap();
    let r = run(&root);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "D014")
        .expect("D014 expected");
    assert!(f.message.contains("brightness drifts"), "{}", f.message);
}

#[test]
fn image_checks_ignore_below_threshold_defects() {
    let dir = tmpdir("imgneg");
    let mut w = DatasetWriter::create(&dir, spec_img()).unwrap();
    let mut seed = 59;
    // One duplicate of 7 transitions (14% < 25%) and a mild 0.055 brightness
    // drift (< 0.25): both under their thresholds.
    for i in 0..8usize {
        let k = if i == 3 { 2 } else { i }; // frames 2 and 3 identical
        let px: Vec<u8> = (0..IH * IW * 3)
            .map(|p| ((100 + k * 2 + (p % 3)) % 256) as u8)
            .collect();
        add_img(&mut w, &mut seed, &png_rgb(&px));
    }
    w.save_episode("t").unwrap();
    let root = w.finalize().unwrap();
    let r = run(&root);
    assert!(r.findings.is_empty(), "{}", r.render_text());
}

// ===== D015 cross-episode duplicates =====

#[test]
fn d015_flags_an_accidental_double_record() {
    let dir = tmpdir("d015");
    let mut w = DatasetWriter::create(&dir, spec_sa()).unwrap();
    let mut seed = 61;
    let frames: Vec<[f64; 2]> = (0..30)
        .map(|_| [noise(&mut seed), noise(&mut seed)])
        .collect();
    // The same take saved twice, then one genuine episode.
    for _ in 0..2 {
        for s in &frames {
            add(&mut w, *s, act(*s));
        }
        w.save_episode("t").unwrap();
    }
    for _ in 0..30 {
        let s = [noise(&mut seed), noise(&mut seed)];
        add(&mut w, s, act(s));
    }
    w.save_episode("t").unwrap();
    let root = w.finalize().unwrap();
    let r = run(&root);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "D015")
        .expect("D015 expected");
    assert!(f.message.contains("episode 0"), "{}", f.message);
    assert!(f.message.contains("episode 1"), "{}", f.message);
    assert!(f.message.contains("double-record"), "{}", f.message);
}

// ===== report plumbing =====

#[test]
fn report_serializes_and_renders() {
    let dir = tmpdir("plumbing");
    let root = contradiction_dataset(&dir);
    let r = run(&root);
    assert!(!r.findings.is_empty());

    let json = serde_json::to_string(&r).unwrap();
    let back: DataReport = serde_json::from_str(&json).unwrap();
    assert_eq!(back.findings.len(), r.findings.len());
    assert_eq!(back.total_frames, r.total_frames);

    let text = r.render_text();
    assert!(text.contains("D006"), "{text}");
    assert!(text.contains("fix:"), "{text}");

    // Findings are sorted most-severe first.
    let ranks: Vec<u8> = r
        .findings
        .iter()
        .map(|f| match f.severity {
            Severity::Error => 0,
            Severity::Warning => 1,
            Severity::Info => 2,
        })
        .collect();
    let mut sorted = ranks.clone();
    sorted.sort_unstable();
    assert_eq!(ranks, sorted);
}
