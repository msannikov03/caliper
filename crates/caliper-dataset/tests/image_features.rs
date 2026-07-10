//! Image (`dtype: "image"`) features through the public writer/reader API:
//! byte-exact PNG round-trips, the Arrow `struct<bytes: binary, path: string>`
//! data-parquet contract (hand-pinned from a dump of lerobot 0.4.4's own
//! writer output — see `oracle/fixtures/datasets/img_v30/README` provenance),
//! the `(c, 1, 1)` stats shapes, and the loud validation paths.

use arrow::array::{Array, BinaryArray, ListArray, StringArray, StructArray};
use arrow::datatypes::DataType;
use caliper_dataset::{DatasetReader, DatasetSpec, DatasetWriter, FeatureSpec};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs;
use std::fs::File;
use std::path::PathBuf;

const H: usize = 4;
const W: usize = 3;

fn tmpdir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("caliper_dataset_img_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    p
}

fn spec_img() -> DatasetSpec {
    DatasetSpec::new(
        30,
        "cam_bot",
        vec![
            FeatureSpec::vector("observation.state", 2, Some(vec!["j1".into(), "j2".into()])),
            FeatureSpec::image("observation.images.cam", H, W, 3),
        ],
    )
}

/// Deterministic RGB pixels for frame `k`: HWC, value = (k*7 + index) % 256.
fn pixels(k: usize) -> Vec<u8> {
    (0..H * W * 3).map(|i| ((k * 7 + i) % 256) as u8).collect()
}

fn png_rgb(h: usize, w: usize, px: &[u8]) -> Vec<u8> {
    assert_eq!(px.len(), h * w * 3);
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().unwrap();
    wr.write_image_data(px).unwrap();
    wr.finish().unwrap();
    out
}

/// Record `frames` frames into the open episode, returning the PNGs appended.
fn record_frames(w: &mut DatasetWriter, ep: usize, frames: usize) -> Vec<Vec<u8>> {
    let mut pngs = Vec::new();
    for i in 0..frames {
        let png = png_rgb(H, W, &pixels(ep * 10 + i));
        let s = [0.1 * i as f64, 0.2 * i as f64];
        w.add_frame_with_images(
            &[("observation.state", &s)],
            &[("observation.images.cam", &png)],
        )
        .unwrap();
        pngs.push(png);
    }
    pngs
}

#[test]
fn image_roundtrip_is_byte_exact() {
    let dir = tmpdir("rt");
    let mut w = DatasetWriter::create(&dir, spec_img()).unwrap();
    let ep0 = record_frames(&mut w, 0, 3);
    w.save_episode("grab").unwrap();
    let ep1 = record_frames(&mut w, 1, 2);
    w.save_episode("wave").unwrap();
    let root = w.finalize().unwrap();

    let r = DatasetReader::open(&root).unwrap();
    assert_eq!(r.total_episodes(), 2);
    let f = &r.info().features["observation.images.cam"];
    assert_eq!(f.dtype, "image");
    assert_eq!(f.shape, vec![H as u64, W as u64, 3]);
    assert_eq!(f.names, serde_json::json!(["height", "width", "channels"]));

    for (idx, pngs) in [(0usize, &ep0), (1, &ep1)] {
        let ep = r.read_episode(idx).unwrap();
        assert_eq!(ep.len(), pngs.len());
        let got = &ep.images["observation.images.cam"];
        assert_eq!(got, pngs, "episode {idx} PNG bytes must round-trip");
        // vector features unaffected by the image column
        assert_eq!(ep.features["observation.state"].len(), pngs.len());
        assert!((f64::from(ep.features["observation.state"][1][0]) - 0.1).abs() < 1e-6);
    }
}

/// The exact Arrow layout lerobot's own writer embeds for `dtype: "image"`:
/// `struct<bytes: binary, path: string>` (field order bytes, path), `path` =
/// per-episode basename `frame-XXXXXX.png`, one row group per episode.
#[test]
fn data_parquet_matches_lerobot_image_contract() {
    let dir = tmpdir("contract");
    let mut w = DatasetWriter::create(&dir, spec_img()).unwrap();
    let pngs = record_frames(&mut w, 0, 2);
    w.save_episode("grab").unwrap();
    record_frames(&mut w, 1, 2);
    w.save_episode("grab").unwrap();
    let root = w.finalize().unwrap();

    let file = File::open(root.join("data/chunk-000/file-000.parquet")).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    assert_eq!(
        builder.metadata().num_row_groups(),
        2,
        "one row group per episode, like lerobot's per-episode write_table"
    );
    let field = builder
        .schema()
        .field_with_name("observation.images.cam")
        .unwrap();
    let DataType::Struct(children) = field.data_type() else {
        panic!("image column must be a struct, got {:?}", field.data_type());
    };
    assert_eq!(children.len(), 2);
    assert_eq!(children[0].name(), "bytes");
    assert_eq!(children[0].data_type(), &DataType::Binary);
    assert_eq!(children[1].name(), "path");
    assert_eq!(children[1].data_type(), &DataType::Utf8);

    let mut reader = builder.build().unwrap();
    let batch = reader.next().unwrap().unwrap();
    let col = batch
        .column(batch.schema().index_of("observation.images.cam").unwrap())
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap()
        .clone();
    let bytes = col
        .column_by_name("bytes")
        .unwrap()
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap()
        .clone();
    let paths = col
        .column_by_name("path")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .clone();
    assert_eq!(bytes.value(0), &pngs[0][..]);
    assert_eq!(bytes.value(1), &pngs[1][..]);
    assert_eq!(paths.value(0), "frame-000000.png");
    assert_eq!(paths.value(1), "frame-000001.png");
    // PNG magic — the stored bytes are complete PNG files
    assert_eq!(&bytes.value(0)[..8], b"\x89PNG\r\n\x1a\n");
}

/// stats.json image entries are `(c, 1, 1)`-nested and computed over every
/// pixel in lerobot's normalized [0, 1] scale; vector entries stay flat.
/// The per-episode stats/ columns in meta/episodes nest identically.
#[test]
fn image_stats_shapes_and_values() {
    let dir = tmpdir("stats");
    let mut w = DatasetWriter::create(&dir, spec_img()).unwrap();
    let frames = 3usize;
    record_frames(&mut w, 0, frames);
    w.save_episode("grab").unwrap();
    let root = w.finalize().unwrap();

    // expected per-channel mean over all pixels of all frames, /255
    let mut sum = [0.0f64; 3];
    for i in 0..frames {
        for (j, &b) in pixels(i).iter().enumerate() {
            sum[j % 3] += f64::from(b) / 255.0;
        }
    }
    let n_px = (frames * H * W) as f64;
    let expected_mean: Vec<f64> = sum.iter().map(|s| s / n_px).collect();

    let stats: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("meta/stats.json")).unwrap()).unwrap();
    let cam = &stats["observation.images.cam"];
    assert_eq!(cam["count"], serde_json::json!([frames]));
    for key in ["min", "max", "mean", "std"] {
        let v = cam[key].as_array().unwrap();
        assert_eq!(v.len(), 3, "{key}: one entry per channel");
        for ch in v {
            let ch = ch.as_array().unwrap();
            assert_eq!(ch.len(), 1, "{key}: (c,1,1) nesting");
            assert_eq!(ch[0].as_array().unwrap().len(), 1);
        }
    }
    for (j, &m) in expected_mean.iter().enumerate() {
        let got = cam["mean"][j][0][0].as_f64().unwrap();
        assert!((got - m).abs() < 1e-12, "channel {j}: {got} vs {m}");
        let std = cam["std"][j][0][0].as_f64().unwrap();
        assert!(std > 0.0, "gradient pixels have nonzero std");
        let lo = cam["min"][j][0][0].as_f64().unwrap();
        let hi = cam["max"][j][0][0].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&lo) && lo <= hi && hi <= 1.0);
    }
    // vector features keep flat lists
    let state_mean = stats["observation.state"]["mean"].as_array().unwrap();
    assert_eq!(state_mean.len(), 2);
    assert!(state_mean[0].is_number());

    // per-episode stats columns nest the same way (count stays flat)
    let file = File::open(root.join("meta/episodes/chunk-000/file-000.parquet")).unwrap();
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let batch = reader.next().unwrap().unwrap();
    let mean_col = batch
        .column(
            batch
                .schema()
                .index_of("stats/observation.images.cam/mean")
                .unwrap(),
        )
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap()
        .clone();
    let row = mean_col.value(0); // (c, 1, 1): outer len 3
    assert_eq!(row.len(), 3);
    let mid = row.as_any().downcast_ref::<ListArray>().unwrap().value(0);
    assert_eq!(mid.len(), 1);
    let inner = mid.as_any().downcast_ref::<ListArray>().unwrap().value(0);
    assert_eq!(inner.len(), 1);
    let count_col = batch
        .column(
            batch
                .schema()
                .index_of("stats/observation.images.cam/count")
                .unwrap(),
        )
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap()
        .clone();
    assert_eq!(count_col.value(0).len(), 1, "count stays a flat [n_frames]");
}

#[test]
fn image_validation_is_loud() {
    let dir = tmpdir("loud");
    let mut w = DatasetWriter::create(&dir, spec_img()).unwrap();
    let s = [0.0, 0.0];
    let good = png_rgb(H, W, &pixels(0));

    // plain add_frame on an image dataset points at the right API
    let err = w
        .add_frame(&[("observation.state", &s[..])])
        .unwrap_err()
        .to_string();
    assert!(err.contains("add_frame_with_images"), "{err}");

    // wrong-shape PNG (transposed dims)
    let bad_shape = png_rgb(W, H, &pixels(0));
    assert!(
        w.add_frame_with_images(
            &[("observation.state", &s)],
            &[("observation.images.cam", &bad_shape)],
        )
        .is_err()
    );

    // bytes that are not a PNG at all
    assert!(
        w.add_frame_with_images(
            &[("observation.state", &s)],
            &[("observation.images.cam", b"not a png".as_slice())],
        )
        .is_err()
    );

    // image passed under a vector feature name / vector under an image name
    assert!(
        w.add_frame_with_images(
            &[("observation.images.cam", &s[..])],
            &[("observation.state", &good)],
        )
        .is_err()
    );

    // failed frames must not leave ragged buffers behind
    w.add_frame_with_images(
        &[("observation.state", &s)],
        &[("observation.images.cam", &good)],
    )
    .unwrap();
    assert_eq!(w.buffered_frames(), 1);
    w.save_episode("ok").unwrap();
    let root = w.finalize().unwrap();
    let r = DatasetReader::open(&root).unwrap();
    assert_eq!(r.read_episode(0).unwrap().len(), 1);
}

#[test]
fn image_spec_validation() {
    for spec in [
        FeatureSpec::image("cam", 0, 8, 3),
        FeatureSpec::image("cam", 8, 0, 3),
        FeatureSpec::image("cam", 8, 8, 0),
        FeatureSpec::image("cam", 8, 8, 5),
    ] {
        let dir = tmpdir("spec");
        assert!(
            DatasetWriter::create(&dir, DatasetSpec::new(30, "t", vec![spec.clone()])).is_err(),
            "accepted bad image spec {spec:?}"
        );
    }
}
