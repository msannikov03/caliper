//! Reader vs the OFFICIAL lerobot toolchain: `oracle/fixtures/datasets/ref_v30`
//! was produced by lerobot 0.4.4's v2.1→v3.0 converter (see the fixture
//! README for provenance/regeneration), so these tests prove
//! [`DatasetReader`] consumes lerobot's own on-disk output — hermetically, no
//! Python at test time. The converter writes vector columns as
//! `List<Float32>` (not our `FixedSizeList`), exercising that decode path.

use caliper_dataset::DatasetReader;
use std::path::PathBuf;

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures/datasets/ref_v30")
}

#[test]
fn reads_converter_produced_dataset() {
    let r = DatasetReader::open(fixture_root()).unwrap();
    assert_eq!(r.info().codebase_version, "v3.0");
    assert_eq!(r.info().robot_type.as_deref(), Some("showcase6"));
    assert_eq!(r.fps(), 50);
    assert_eq!(r.total_episodes(), 3);
    assert_eq!(
        r.tasks(),
        &["reach a pose".to_string(), "wave".to_string()][..]
    );
    assert_eq!(r.info().data_files_size_in_mb, 100.0);
    assert_eq!(r.info().chunks_size, 1000);

    let metas = r.episodes();
    assert_eq!(
        metas.iter().map(|m| m.length).collect::<Vec<_>>(),
        vec![25, 30, 20]
    );
    assert_eq!(
        metas
            .iter()
            .map(|m| (m.dataset_from_index, m.dataset_to_index))
            .collect::<Vec<_>>(),
        vec![(0, 25), (25, 55), (55, 75)]
    );
    assert!(
        metas
            .iter()
            .all(|m| m.data_chunk_index == 0 && m.data_file_index == 0)
    );
    assert_eq!(metas[0].tasks, vec!["reach a pose".to_string()]);
    assert_eq!(metas[2].tasks, vec!["wave".to_string()]);
}

#[test]
fn episode_frames_decode_correctly() {
    let r = DatasetReader::open(fixture_root()).unwrap();
    for (idx, len, from) in [(0usize, 25usize, 0i64), (1, 30, 25), (2, 20, 55)] {
        let ep = r.read_episode(idx).unwrap();
        assert_eq!(ep.len(), len);
        assert_eq!(ep.episode_index, idx as i64);
        assert_eq!(ep.frame_indices, (0..len as i64).collect::<Vec<_>>());
        assert_eq!(
            ep.global_indices,
            (from..from + len as i64).collect::<Vec<_>>()
        );
        let states = &ep.features["observation.state"];
        let actions = &ep.features["action"];
        assert!(states.iter().all(|s| s.len() == 6));
        assert!(actions.iter().all(|a| a.len() == 6));
        assert!(states.iter().flatten().all(|v| v.is_finite()));
        // the recording's control-loop clock is continuous across episodes at
        // 50 fps: t = global_index / 50
        for (i, &t) in ep.timestamps.iter().enumerate() {
            assert!(
                (t - (from + i as i64) as f64 / 50.0).abs() < 1e-4,
                "t={t} i={i}"
            );
        }
    }
    // spot values pinned from the committed fixture (converter output)
    let ep0 = r.read_episode(0).unwrap();
    assert!(
        ep0.features["observation.state"][0]
            .iter()
            .all(|&v| v == 0.0)
    );
    let ep1 = r.read_episode(1).unwrap();
    assert!((ep1.features["observation.state"][0][0] - 0.168_419_46).abs() < 1e-6);
    assert!(ep1.task_indices.iter().all(|&t| t == 0));
    let ep2 = r.read_episode(2).unwrap();
    assert!(ep2.task_indices.iter().all(|&t| t == 1));
}
