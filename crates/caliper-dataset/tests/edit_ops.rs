//! Offline edit ops: every op must leave a reader-loadable dataset with exact
//! expected frame values, dense renumbering, recomputed stats, remapped tasks
//! and tags — and invalid arguments must leave the original untouched.

use caliper_dataset::edit::{
    delete_episodes, merge_episodes, read_tags, split_episode, write_tags,
};
use caliper_dataset::{DatasetReader, DatasetSpec, DatasetWriter, Error, FeatureSpec};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

const FPS: u32 = 50;

fn tmpdir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("caliper_edit_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    p
}

/// 3 episodes over one 2-dim feature; deterministic values `10*ep + frame`
/// in dim 0 and `-(10*ep + frame)` in dim 1, so any frame is identifiable.
/// Tasks: ep0 "wave", ep1 "reach" (only user), ep2 "wave".
fn build(tag: &str, lengths: [usize; 3]) -> PathBuf {
    let dir = tmpdir(tag);
    let mut w = DatasetWriter::create(
        &dir,
        DatasetSpec::new(
            FPS,
            "edit_bot",
            vec![FeatureSpec::vector(
                "observation.state",
                2,
                Some(vec!["j1".into(), "j2".into()]),
            )],
        ),
    )
    .unwrap();
    for (ep, (&len, task)) in lengths.iter().zip(["wave", "reach", "wave"]).enumerate() {
        for i in 0..len {
            let v = (10 * ep + i) as f64;
            w.add_frame(&[("observation.state", &[v, -v][..])]).unwrap();
        }
        w.save_episode(task).unwrap();
    }
    w.finalize().unwrap()
}

fn frame_val(r: &DatasetReader, ep: usize, frame: usize) -> f64 {
    f64::from(r.read_episode(ep).unwrap().features["observation.state"][frame][0])
}

fn tags(pairs: &[(u64, &[&str])]) -> BTreeMap<u64, Vec<String>> {
    pairs
        .iter()
        .map(|(k, v)| (*k, v.iter().map(|s| s.to_string()).collect()))
        .collect()
}

#[test]
fn delete_renumbers_remaps_tasks_and_recomputes_stats() {
    let root = build("delete", [4, 3, 5]);
    write_tags(
        &root,
        &tags(&[(0, &["keep"]), (1, &["bad-demo"]), (2, &["success"])]),
    )
    .unwrap();

    delete_episodes(&root, &[1]).unwrap();

    let r = DatasetReader::open(&root).unwrap();
    assert_eq!(r.total_episodes(), 2);
    // dropped-task remap: "reach" was only used by ep1 and must be gone
    assert_eq!(r.tasks(), &["wave".to_string()][..]);
    let metas = r.episodes();
    assert_eq!(
        metas.iter().map(|m| m.episode_index).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        metas
            .iter()
            .map(|m| (m.dataset_from_index, m.dataset_to_index))
            .collect::<Vec<_>>(),
        vec![(0, 4), (4, 9)] // dense global offsets after the delete
    );
    // survivors carry the ORIGINAL frame values of old ep0 and ep2
    assert_eq!(frame_val(&r, 0, 3), 3.0);
    assert_eq!(frame_val(&r, 1, 0), 20.0);
    assert_eq!(frame_val(&r, 1, 4), 24.0);
    let ep1 = r.read_episode(1).unwrap();
    assert_eq!(ep1.episode_index, 1);
    assert_eq!(ep1.global_indices, (4..9).collect::<Vec<i64>>());
    assert_eq!(ep1.frame_indices, (0..5).collect::<Vec<i64>>());
    assert!(ep1.task_indices.iter().all(|&t| t == 0));
    // timestamps of a surviving episode are copied verbatim
    for (i, &t) in ep1.timestamps.iter().enumerate() {
        assert!((t - i as f64 / f64::from(FPS)).abs() < 1e-5);
    }

    // aggregated stats recomputed over the SURVIVING frames only
    let stats: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("meta/stats.json")).unwrap()).unwrap();
    let pooled: Vec<f64> = (0..4)
        .map(|i| i as f64)
        .chain((0..5).map(|i| (20 + i) as f64))
        .collect();
    let mean = pooled.iter().sum::<f64>() / pooled.len() as f64;
    let x = &stats["observation.state"];
    assert_eq!(x["count"][0], 9);
    assert_eq!(x["min"][0], 0.0);
    assert_eq!(x["max"][0], 24.0);
    assert!((x["mean"][0].as_f64().unwrap() - mean).abs() < 1e-9);
    assert_eq!(stats["task_index"]["max"][0], 0.0);
    assert_eq!(stats["index"]["max"][0], 8.0);

    // info totals rewritten
    let info: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("meta/info.json")).unwrap()).unwrap();
    assert_eq!(info["total_episodes"], 2);
    assert_eq!(info["total_frames"], 9);
    assert_eq!(info["total_tasks"], 1);
    assert_eq!(info["splits"]["train"], "0:2");
    assert_eq!(info["robot_type"], "edit_bot");
    assert_eq!(info["features"]["observation.state"]["names"][0], "j1");

    // tags remapped: old 0 → 0, old 2 → 1, old 1 gone
    assert_eq!(
        read_tags(&root).unwrap(),
        tags(&[(0, &["keep"]), (1, &["success"])])
    );

    // no swap leftovers next to the dataset
    let name = root.file_name().unwrap().to_str().unwrap();
    let parent = root.parent().unwrap();
    assert!(!parent.join(format!("{name}.caliper-edit-tmp")).exists());
    assert!(!parent.join(format!("{name}.caliper-edit-old")).exists());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn delete_multiple_and_unknown_info_fields_survive() {
    let root = build("delete2", [4, 3, 5]);
    // inject an unknown top-level info.json field — must survive the rewrite
    let info_path = root.join("meta/info.json");
    let mut info: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&info_path).unwrap()).unwrap();
    info.as_object_mut()
        .unwrap()
        .insert("caliper_custom".into(), serde_json::json!({"a": 1}));
    fs::write(&info_path, serde_json::to_string_pretty(&info).unwrap()).unwrap();

    delete_episodes(&root, &[0, 2]).unwrap();

    let r = DatasetReader::open(&root).unwrap();
    assert_eq!(r.total_episodes(), 1);
    assert_eq!(r.tasks(), &["reach".to_string()][..]);
    assert_eq!(frame_val(&r, 0, 0), 10.0);
    let info: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&info_path).unwrap()).unwrap();
    assert_eq!(info["caliper_custom"]["a"], 1);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn split_produces_two_episodes_with_rebased_timestamps() {
    let root = build("split", [4, 3, 5]);
    write_tags(&root, &tags(&[(2, &["success"])])).unwrap();

    split_episode(&root, 2, 2).unwrap();

    let r = DatasetReader::open(&root).unwrap();
    assert_eq!(r.total_episodes(), 4);
    assert_eq!(
        r.episodes().iter().map(|m| m.length).collect::<Vec<_>>(),
        vec![4, 3, 2, 3]
    );
    // both halves keep the task
    assert_eq!(r.episodes()[2].tasks, vec!["wave".to_string()]);
    assert_eq!(r.episodes()[3].tasks, vec!["wave".to_string()]);
    // values: part a = frames 0..2 of old ep2, part b = frames 2..5
    assert_eq!(frame_val(&r, 2, 0), 20.0);
    assert_eq!(frame_val(&r, 2, 1), 21.0);
    assert_eq!(frame_val(&r, 3, 0), 22.0);
    assert_eq!(frame_val(&r, 3, 2), 24.0);
    // second half restarts per-episode-relative bookkeeping
    let b = r.read_episode(3).unwrap();
    assert_eq!(b.frame_indices, vec![0, 1, 2]);
    assert_eq!(b.global_indices, (9..12).collect::<Vec<i64>>());
    for (i, &t) in b.timestamps.iter().enumerate() {
        assert!((t - i as f64 / f64::from(FPS)).abs() < 1e-5, "t={t}");
    }
    // tags: both halves inherit
    assert_eq!(
        read_tags(&root).unwrap(),
        tags(&[(2, &["success"]), (3, &["success"])])
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn merge_unions_tasks_and_keeps_per_frame_task_indices() {
    let root = build("merge", [4, 3, 5]);
    write_tags(&root, &tags(&[(0, &["a"]), (1, &["b", "a"]), (2, &["c"])])).unwrap();

    merge_episodes(&root, 0, 1).unwrap();

    let r = DatasetReader::open(&root).unwrap();
    assert_eq!(r.total_episodes(), 2);
    let metas = r.episodes();
    assert_eq!(metas[0].length, 7);
    assert_eq!(metas[1].length, 5);
    // episode-level task union, ep_a's tasks first
    assert_eq!(
        metas[0].tasks,
        vec!["wave".to_string(), "reach".to_string()]
    );
    // frames keep their own task_index (wave=0 for old ep0 frames, reach for old ep1)
    let m = r.read_episode(0).unwrap();
    let reach_idx = r.tasks().iter().position(|t| t == "reach").unwrap() as i64;
    assert!(m.task_indices[..4].iter().all(|&t| t == 0));
    assert!(m.task_indices[4..].iter().all(|&t| t == reach_idx));
    // values concatenate in order
    assert_eq!(
        m.features["observation.state"]
            .iter()
            .map(|v| f64::from(v[0]))
            .collect::<Vec<_>>(),
        vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
    );
    // timestamps stay per-episode-relative and continue at 1/fps across the seam
    for (i, &t) in m.timestamps.iter().enumerate() {
        assert!((t - i as f64 / f64::from(FPS)).abs() < 1e-5, "i={i} t={t}");
    }
    assert_eq!(m.frame_indices, (0..7).collect::<Vec<i64>>());
    assert_eq!(m.global_indices, (0..7).collect::<Vec<i64>>());
    // trailing episode shifted down, values intact
    assert_eq!(frame_val(&r, 1, 0), 20.0);
    // tags: union of a+b (deduped, a's order first), old 2 → 1
    assert_eq!(
        read_tags(&root).unwrap(),
        tags(&[(0, &["a", "b"]), (1, &["c"])])
    );
    // aggregated stats cover all 12 frames still
    let stats: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("meta/stats.json")).unwrap()).unwrap();
    assert_eq!(stats["observation.state"]["count"][0], 12);
    assert_eq!(stats["episode_index"]["max"][0], 1.0);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn edited_datasets_chain_and_stay_loadable() {
    // delete → split → merge in sequence; each step must load cleanly.
    let root = build("chain", [4, 3, 5]);
    delete_episodes(&root, &[1]).unwrap();
    split_episode(&root, 1, 2).unwrap();
    merge_episodes(&root, 1, 2).unwrap(); // undo the split
    let r = DatasetReader::open(&root).unwrap();
    assert_eq!(r.total_episodes(), 2);
    assert_eq!(
        r.episodes().iter().map(|m| m.length).collect::<Vec<_>>(),
        vec![4, 5]
    );
    assert_eq!(frame_val(&r, 1, 4), 24.0);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn validation_errors_leave_dataset_untouched() {
    let root = build("errs", [4, 3, 5]);
    let before = fs::read(root.join("meta/info.json")).unwrap();

    assert!(matches!(delete_episodes(&root, &[]), Err(Error::Edit(_))));
    assert!(matches!(delete_episodes(&root, &[3]), Err(Error::Edit(_))));
    assert!(matches!(
        delete_episodes(&root, &[0, 1, 2]),
        Err(Error::Edit(_))
    ));
    assert!(matches!(split_episode(&root, 3, 1), Err(Error::Edit(_))));
    assert!(matches!(split_episode(&root, 0, 0), Err(Error::Edit(_))));
    assert!(matches!(split_episode(&root, 0, 4), Err(Error::Edit(_))));
    assert!(matches!(merge_episodes(&root, 0, 2), Err(Error::Edit(_))));
    assert!(matches!(merge_episodes(&root, 1, 0), Err(Error::Edit(_))));
    assert!(matches!(merge_episodes(&root, 2, 3), Err(Error::Edit(_))));

    assert_eq!(fs::read(root.join("meta/info.json")).unwrap(), before);
    let r = DatasetReader::open(&root).unwrap();
    assert_eq!(r.total_episodes(), 3);
    assert_eq!(frame_val(&r, 2, 4), 24.0);
    let _ = fs::remove_dir_all(&root);
}

// ===== image datasets: PNG bytes must ride along through every op =====

const IH: usize = 4;
const IW: usize = 3;

/// Deterministic RGB PNG for (episode, frame) — every pixel identifies its
/// source frame, so the byte-equality assertions below prove exact
/// index-level streaming (not just "some image survived").
fn png_rgb(ep: usize, frame: usize) -> Vec<u8> {
    let px: Vec<u8> = (0..IH * IW * 3)
        .map(|i| (((10 * ep + frame) * 7 + i) % 256) as u8)
        .collect();
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, IW as u32, IH as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().unwrap();
    w.write_image_data(&px).unwrap();
    w.finish().unwrap();
    out
}

/// Like [`build`] but with a camera feature next to the vector; returns the
/// per-episode PNGs exactly as recorded.
fn build_img(tag: &str, lengths: [usize; 3]) -> (PathBuf, Vec<Vec<Vec<u8>>>) {
    let dir = tmpdir(tag);
    let mut w = DatasetWriter::create(
        &dir,
        DatasetSpec::new(
            FPS,
            "edit_bot",
            vec![
                FeatureSpec::vector("observation.state", 2, None),
                FeatureSpec::image("observation.images.cam", IH, IW, 3),
            ],
        ),
    )
    .unwrap();
    let mut pngs = Vec::new();
    for (ep, (&len, task)) in lengths.iter().zip(["wave", "reach", "wave"]).enumerate() {
        let mut ep_pngs = Vec::new();
        for i in 0..len {
            let v = (10 * ep + i) as f64;
            let png = png_rgb(ep, i);
            w.add_frame_with_images(
                &[("observation.state", &[v, -v][..])],
                &[("observation.images.cam", &png)],
            )
            .unwrap();
            ep_pngs.push(png);
        }
        w.save_episode(task).unwrap();
        pngs.push(ep_pngs);
    }
    (w.finalize().unwrap(), pngs)
}

fn ep_pngs(r: &DatasetReader, ep: usize) -> Vec<Vec<u8>> {
    r.read_episode(ep).unwrap().images["observation.images.cam"].clone()
}

#[test]
fn delete_streams_image_bytes_through() {
    let (root, pngs) = build_img("img_delete", [4, 3, 5]);
    delete_episodes(&root, &[1]).unwrap();
    let r = DatasetReader::open(&root).unwrap();
    assert_eq!(r.total_episodes(), 2);
    // surviving episodes' PNGs round-trip byte-exactly through the rewrite
    assert_eq!(ep_pngs(&r, 0), pngs[0]);
    assert_eq!(ep_pngs(&r, 1), pngs[2]);
    // the image feature survives in info.json with its shape intact
    let f = &r.info().features["observation.images.cam"];
    assert_eq!(f.dtype, "image");
    assert_eq!(f.shape, vec![IH as u64, IW as u64, 3]);
    // vector features still ride along next to the images
    assert_eq!(frame_val(&r, 1, 4), 24.0);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn split_slices_image_bytes() {
    let (root, pngs) = build_img("img_split", [4, 3, 5]);
    split_episode(&root, 2, 2).unwrap();
    let r = DatasetReader::open(&root).unwrap();
    assert_eq!(r.total_episodes(), 4);
    // untouched episodes copy verbatim; the split halves slice the byte vectors
    assert_eq!(ep_pngs(&r, 0), pngs[0]);
    assert_eq!(ep_pngs(&r, 1), pngs[1]);
    assert_eq!(ep_pngs(&r, 2), pngs[2][..2].to_vec());
    assert_eq!(ep_pngs(&r, 3), pngs[2][2..].to_vec());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn merge_concatenates_image_bytes() {
    let (root, pngs) = build_img("img_merge", [4, 3, 5]);
    merge_episodes(&root, 0, 1).unwrap();
    let r = DatasetReader::open(&root).unwrap();
    assert_eq!(r.total_episodes(), 2);
    // merged episode keeps BOTH halves' frames, in order, byte-exact
    let want: Vec<Vec<u8>> = pngs[0].iter().chain(&pngs[1]).cloned().collect();
    assert_eq!(ep_pngs(&r, 0), want);
    assert_eq!(ep_pngs(&r, 1), pngs[2]);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn tags_roundtrip_and_tolerate_missing_file() {
    let root = build("tags", [4, 3, 5]);
    assert_eq!(read_tags(&root).unwrap(), BTreeMap::new());
    let t = tags(&[(0, &["success"]), (2, &["bad-demo", "retry"])]);
    write_tags(&root, &t).unwrap();
    assert_eq!(read_tags(&root).unwrap(), t);
    // empty tag lists are dropped on write
    let mut with_empty = t.clone();
    with_empty.insert(1, vec![]);
    write_tags(&root, &with_empty).unwrap();
    assert_eq!(read_tags(&root).unwrap(), t);
    // not a dataset → refused
    assert!(write_tags(std::env::temp_dir().join("nope_ds"), &t).is_err());
    let _ = fs::remove_dir_all(&root);
}
