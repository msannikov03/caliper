//! Writer → reader round-trips over the public API: fidelity, drop-guard
//! auto-finalize, size-based file rolling, stats parity, determinism, and the
//! error paths that keep buffers consistent.

use caliper_dataset::{DatasetReader, DatasetSpec, DatasetWriter, Error, FeatureSpec};
use std::fs;
use std::path::PathBuf;

fn tmpdir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("caliper_dataset_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    p
}

fn spec2() -> DatasetSpec {
    DatasetSpec::new(
        50,
        "test_bot",
        vec![
            FeatureSpec::vector("observation.state", 2, Some(vec!["j1".into(), "j2".into()])),
            FeatureSpec::vector("action", 2, None),
        ],
    )
}

/// Deterministic pseudo-random f64 in [-1, 1) — splitmix-style, no deps.
fn noise(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn record_episode(
    w: &mut DatasetWriter,
    frames: usize,
    task: &str,
    seed: &mut u64,
) -> Vec<[f64; 2]> {
    let mut states = Vec::new();
    for _ in 0..frames {
        let s = [noise(seed), noise(seed)];
        let a = [noise(seed), noise(seed)];
        w.add_frame(&[("observation.state", &s), ("action", &a)])
            .unwrap();
        states.push(s);
    }
    w.save_episode(task).unwrap();
    states
}

#[test]
fn write_then_read_roundtrip() {
    let dir = tmpdir("rt");
    let mut w = DatasetWriter::create(&dir, spec2()).unwrap();
    let mut seed = 7;
    let ep0 = record_episode(&mut w, 5, "wave", &mut seed);
    let ep1 = record_episode(&mut w, 8, "reach", &mut seed);
    let ep2 = record_episode(&mut w, 3, "wave", &mut seed);
    assert_eq!(w.total_episodes(), 3);
    let root = w.finalize().unwrap();

    // v3.0 layout on disk (no v2.1 leftovers)
    assert!(root.join("data/chunk-000/file-000.parquet").exists());
    assert!(
        root.join("meta/episodes/chunk-000/file-000.parquet")
            .exists()
    );
    assert!(root.join("meta/tasks.parquet").exists());
    assert!(root.join("meta/stats.json").exists());
    let info: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("meta/info.json")).unwrap()).unwrap();
    assert_eq!(info["codebase_version"], "v3.0");
    assert_eq!(info["total_episodes"], 3);
    assert_eq!(info["total_frames"], 16);
    assert_eq!(info["total_tasks"], 2);
    assert_eq!(info["splits"]["train"], "0:3");
    assert_eq!(
        info["data_path"],
        "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    );
    assert!(
        info.get("total_chunks").is_none(),
        "v2.1 field must not appear"
    );
    assert!(
        info.get("total_videos").is_none(),
        "v2.1 field must not appear"
    );
    assert_eq!(info["chunks_size"], 1000);
    assert_eq!(info["data_files_size_in_mb"], 100);
    assert_eq!(info["video_files_size_in_mb"], 200);
    assert!(info["video_path"].is_null());
    assert_eq!(info["features"]["observation.state"]["names"][0], "j1");
    assert_eq!(info["features"]["action"]["names"], serde_json::Value::Null);
    assert_eq!(info["features"]["timestamp"]["dtype"], "float32");

    let r = DatasetReader::open(&root).unwrap();
    assert_eq!(r.total_episodes(), 3);
    assert_eq!(r.fps(), 50);
    assert_eq!(r.tasks(), &["wave".to_string(), "reach".to_string()][..]);

    let metas = r.episodes();
    assert_eq!(metas[0].length, 5);
    assert_eq!(metas[1].length, 8);
    assert_eq!(metas[2].length, 3);
    assert_eq!(
        (metas[1].dataset_from_index, metas[1].dataset_to_index),
        (5, 13)
    );
    assert_eq!(metas[2].tasks, ["wave".to_string()]);

    for (idx, want) in [(0usize, &ep0), (1, &ep1), (2, &ep2)] {
        let ep = r.read_episode(idx).unwrap();
        assert_eq!(ep.len(), want.len());
        assert_eq!(ep.episode_index, idx as i64);
        let states = &ep.features["observation.state"];
        for (i, row) in want.iter().enumerate() {
            for j in 0..2 {
                assert!((f64::from(states[i][j]) - row[j]).abs() < 1e-6);
            }
            // default timestamps are frame_index / fps, resetting per episode
            assert!((ep.timestamps[i] - i as f64 / 50.0).abs() < 1e-5);
            assert_eq!(ep.frame_indices[i], i as i64);
        }
    }
    // global index continues across episodes; task interning is stable
    let ep1_read = r.read_episode(1).unwrap();
    assert_eq!(ep1_read.global_indices, (5..13).collect::<Vec<i64>>());
    assert!(ep1_read.task_indices.iter().all(|&t| t == 1));
    let ep2_read = r.read_episode(2).unwrap();
    assert!(ep2_read.task_indices.iter().all(|&t| t == 0));

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn drop_auto_finalizes() {
    let dir = tmpdir("drop");
    let mut seed = 42;
    let states;
    {
        let mut w = DatasetWriter::create(&dir, spec2()).unwrap();
        states = record_episode(&mut w, 6, "t", &mut seed);
        // no finalize() — the guard must flush footers + episode metadata
    }
    let r = DatasetReader::open(&dir).unwrap();
    assert_eq!(r.total_episodes(), 1);
    let ep = r.read_episode(0).unwrap();
    assert_eq!(ep.len(), 6);
    for (i, row) in states.iter().enumerate() {
        assert!((f64::from(ep.features["observation.state"][i][0]) - row[0]).abs() < 1e-6);
    }
    let info: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("meta/info.json")).unwrap()).unwrap();
    assert_eq!(info["total_episodes"], 1);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn drop_discards_unsaved_frames_but_keeps_saved_episodes() {
    let dir = tmpdir("drop_partial");
    let mut seed = 1;
    {
        let mut w = DatasetWriter::create(&dir, spec2()).unwrap();
        record_episode(&mut w, 4, "t", &mut seed);
        // buffered frames never saved as an episode
        w.add_frame(&[
            ("observation.state", &[0.1, 0.2][..]),
            ("action", &[0.0, 0.0][..]),
        ])
        .unwrap();
        assert_eq!(w.buffered_frames(), 1);
    }
    let r = DatasetReader::open(&dir).unwrap();
    assert_eq!(r.total_episodes(), 1);
    assert_eq!(r.read_episode(0).unwrap().len(), 4);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn files_roll_at_tiny_size_target_and_wrap_chunks() {
    let dir = tmpdir("roll");
    let mut spec = spec2();
    spec.data_files_size_in_mb = 1e-6; // every next episode exceeds the target
    spec.chunks_size = 2; // wrap into chunk-001 after 2 files
    let mut w = DatasetWriter::create(&dir, spec).unwrap();
    let mut seed = 3;
    let eps: Vec<Vec<[f64; 2]>> = (0..3)
        .map(|_| record_episode(&mut w, 4, "t", &mut seed))
        .collect();
    let root = w.finalize().unwrap();

    assert!(root.join("data/chunk-000/file-000.parquet").exists());
    assert!(root.join("data/chunk-000/file-001.parquet").exists());
    assert!(root.join("data/chunk-001/file-000.parquet").exists());

    let r = DatasetReader::open(&root).unwrap();
    let metas = r.episodes();
    assert_eq!(
        metas
            .iter()
            .map(|m| (m.data_chunk_index, m.data_file_index))
            .collect::<Vec<_>>(),
        vec![(0, 0), (0, 1), (1, 0)]
    );
    // offsets stay global even across files, and rows resolve correctly
    assert_eq!(
        (metas[2].dataset_from_index, metas[2].dataset_to_index),
        (8, 12)
    );
    for (idx, want) in eps.iter().enumerate() {
        let ep = r.read_episode(idx).unwrap();
        assert_eq!(ep.len(), 4);
        for (i, row) in want.iter().enumerate() {
            assert!((f64::from(ep.features["observation.state"][i][1]) - row[1]).abs() < 1e-6);
        }
    }
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn stats_match_hand_computed_lerobot_aggregation() {
    let dir = tmpdir("stats");
    let mut w = DatasetWriter::create(
        &dir,
        DatasetSpec::new(10, "t", vec![FeatureSpec::vector("x", 1, None)]),
    )
    .unwrap();
    // episode A: {1, 2, 3}; episode B: {10, 14}
    for v in [1.0, 2.0, 3.0] {
        w.add_frame(&[("x", &[v][..])]).unwrap();
    }
    w.save_episode("a").unwrap();
    for v in [10.0, 14.0] {
        w.add_frame(&[("x", &[v][..])]).unwrap();
    }
    w.save_episode("b").unwrap();
    let root = w.finalize().unwrap();

    let stats: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("meta/stats.json")).unwrap()).unwrap();
    let x = &stats["x"];
    // pooled population stats over {1,2,3,10,14}
    let pooled = [1.0f64, 2.0, 3.0, 10.0, 14.0];
    let mean = pooled.iter().sum::<f64>() / 5.0;
    let var = pooled.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 5.0;
    assert_eq!(x["count"][0], 5);
    assert_eq!(x["min"][0], 1.0);
    assert_eq!(x["max"][0], 14.0);
    assert!((x["mean"][0].as_f64().unwrap() - mean).abs() < 1e-9);
    assert!((x["std"][0].as_f64().unwrap() - var.sqrt()).abs() < 1e-9);
    // implicit features carry stats too (converter parity)
    for feat in [
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ] {
        assert!(stats.get(feat).is_some(), "missing stats for {feat}");
        assert_eq!(stats[feat]["count"][0], 5, "{feat}");
    }
    assert_eq!(stats["index"]["max"][0], 4.0);
    assert_eq!(stats["task_index"]["max"][0], 1.0);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn identical_recordings_produce_identical_bytes() {
    let write = |dir: &PathBuf| {
        let mut w = DatasetWriter::create(dir, spec2()).unwrap();
        let mut seed = 99;
        record_episode(&mut w, 5, "a", &mut seed);
        record_episode(&mut w, 7, "b", &mut seed);
        w.finalize().unwrap()
    };
    let r1 = write(&tmpdir("det1"));
    let r2 = write(&tmpdir("det2"));
    for rel in [
        "data/chunk-000/file-000.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "meta/tasks.parquet",
        "meta/info.json",
        "meta/stats.json",
    ] {
        assert_eq!(
            fs::read(r1.join(rel)).unwrap(),
            fs::read(r2.join(rel)).unwrap(),
            "non-deterministic output: {rel}"
        );
    }
    let _ = fs::remove_dir_all(&r1);
    let _ = fs::remove_dir_all(&r2);
}

#[test]
fn error_paths_stay_consistent() {
    let dir = tmpdir("errs");
    let mut w = DatasetWriter::create(&dir, spec2()).unwrap();

    // wrong width
    let err = w
        .add_frame(&[
            ("observation.state", &[0.0][..]),
            ("action", &[0.0, 0.0][..]),
        ])
        .unwrap_err();
    assert!(
        matches!(
            &err,
            Error::Shape {
                expected: 2,
                got: 1,
                ..
            }
        ),
        "{err}"
    );
    // unknown feature name
    assert!(
        w.add_frame(&[
            ("observation.state", &[0.0, 0.0][..]),
            ("nope", &[0.0, 0.0][..])
        ])
        .is_err()
    );
    // wrong feature count
    assert!(
        w.add_frame(&[("observation.state", &[0.0, 0.0][..])])
            .is_err()
    );
    // failed frames must not leave ragged buffers
    assert_eq!(w.buffered_frames(), 0);
    // empty episode
    assert!(w.save_episode("t").is_err());
    // finalize with unsaved frames
    w.add_frame(&[
        ("observation.state", &[0.0, 0.0][..]),
        ("action", &[0.0, 0.0][..]),
    ])
    .unwrap();
    assert!(matches!(w.finalize(), Err(Error::State(_))));
    let _ = fs::remove_dir_all(&dir);

    // invalid specs
    let bad = DatasetSpec::new(50, "t", vec![FeatureSpec::vector("index", 1, None)]);
    assert!(DatasetWriter::create(tmpdir("errs2"), bad).is_err());
    let bad = DatasetSpec::new(50, "t", vec![FeatureSpec::vector("a/b", 1, None)]);
    assert!(DatasetWriter::create(tmpdir("errs3"), bad).is_err());

    // refusing to overwrite an existing dataset
    let dir2 = tmpdir("errs4");
    let mut w = DatasetWriter::create(&dir2, spec2()).unwrap();
    let mut seed = 5;
    record_episode(&mut w, 2, "t", &mut seed);
    let root = w.finalize().unwrap();
    assert!(DatasetWriter::create(&root, spec2()).is_err());
    let _ = fs::remove_dir_all(&root);
}
