//! Criterion benchmarks for the kinematics hot paths: forward kinematics and the
//! geometric Jacobian on a realistic 6R arm (`showcase6`).
use caliper_kinematics::{JacFrame, fk_joints, fk_tip, jacobian};
use caliper_model::Model;
use caliper_spatial::Se3;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::path::Path;

fn load(name: &str) -> Model {
    let p = format!(
        "{}/../../oracle/fixtures/robots/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    Model::from_urdf(Path::new(&p)).expect("load fixture")
}

fn bench_kinematics(c: &mut Criterion) {
    let m = load("showcase6.urdf");
    let q = [0.2_f64, -0.4, 0.5, 0.1, 0.3, -0.2];
    let frame = m.tip_frame();
    let mut out = vec![Se3::identity(); m.ndof];

    c.bench_function("fk_joints/showcase6", |b| {
        b.iter(|| fk_joints(black_box(&m), black_box(&q), black_box(&mut out)))
    });

    c.bench_function("fk_tip/showcase6", |b| {
        b.iter(|| fk_tip(black_box(&m), black_box(&q)))
    });

    c.bench_function("jacobian/showcase6", |b| {
        b.iter(|| {
            jacobian(
                black_box(&m),
                black_box(&q),
                black_box(frame),
                JacFrame::World,
            )
        })
    });
}

criterion_group!(benches, bench_kinematics);
criterion_main!(benches);
