//! Reader vs the OFFICIAL lerobot toolchain, image edition:
//! `oracle/fixtures/datasets/img_v30` was produced by lerobot 0.4.4's OWN
//! v3.0 writer (`LeRobotDataset.create` + `add_frame` + `save_episode` +
//! `finalize`, `use_videos=False`) with a `dtype: "image"` camera feature —
//! see the fixture README for provenance/regeneration. These tests prove
//! [`DatasetReader`] decodes lerobot's embedded `struct<bytes, path>` PNG
//! column hermetically, no Python at test time.

use caliper_dataset::DatasetReader;
use std::path::PathBuf;

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures/datasets/img_v30")
}

#[test]
fn reads_lerobot_written_image_dataset() {
    let r = DatasetReader::open(fixture_root()).unwrap();
    assert_eq!(r.info().codebase_version, "v3.0");
    assert_eq!(r.info().robot_type.as_deref(), Some("caliper"));
    assert_eq!(r.fps(), 30);
    assert_eq!(r.total_episodes(), 2);
    assert_eq!(r.tasks(), &["demo".to_string()][..]);

    let cam = &r.info().features["observation.images.cam"];
    assert_eq!(cam.dtype, "image");
    assert_eq!(cam.shape, vec![32, 32, 3]);
    assert_eq!(
        cam.names,
        serde_json::json!(["height", "width", "channels"])
    );

    for idx in 0..2 {
        let ep = r.read_episode(idx).unwrap();
        assert_eq!(ep.len(), 3);
        let frames = &ep.images["observation.images.cam"];
        assert_eq!(frames.len(), 3);
        for png in frames {
            assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "complete PNG files");
            assert!(png.len() > 50);
        }
        // vector features decode alongside the image column
        assert!((f64::from(ep.features["observation.state"][2][1]) - 0.4).abs() < 1e-6);
        assert!((f64::from(ep.features["action"][1][0]) - 0.3).abs() < 1e-6);
    }

    // frames are distinct per (episode, frame) — pinned from the generator
    let e0 = r.read_episode(0).unwrap();
    let e1 = r.read_episode(1).unwrap();
    let f0 = &e0.images["observation.images.cam"];
    let f1 = &e1.images["observation.images.cam"];
    assert_ne!(f0[0], f0[1]);
    assert_ne!(f0[0], f1[0]);
    // byte lengths pinned from the committed fixture (PIL PNG encoder)
    assert_eq!(f0.iter().map(Vec::len).collect::<Vec<_>>(), [117, 118, 120]);
    assert_eq!(f1.iter().map(Vec::len).collect::<Vec<_>>(), [121, 123, 123]);
}
