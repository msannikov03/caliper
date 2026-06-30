//! Criterion benchmarks for the dynamics hot paths on a realistic 6R arm
//! (`showcase6`): inverse dynamics (RNEA), the joint-space inertia matrix (CRBA),
//! and forward dynamics (articulated-body acceleration via CRBA + Cholesky).
use caliper_dynamics::{GRAVITY_EARTH, crba, forward_dynamics, rnea};
use caliper_model::Model;
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

fn bench_dynamics(c: &mut Criterion) {
    let m = load("showcase6.urdf");
    let q = [0.2_f64, -0.4, 0.5, 0.1, 0.3, -0.2];
    let qd = [0.1_f64, 0.2, -0.1, 0.05, -0.05, 0.15];
    let qdd = [0.05_f64, -0.1, 0.2, 0.0, 0.1, -0.05];
    let tau = [1.0_f64, -2.0, 0.5, 0.2, -0.3, 0.1];
    let g = GRAVITY_EARTH;

    c.bench_function("rnea/showcase6", |b| {
        b.iter(|| {
            rnea(
                black_box(&m),
                black_box(&q),
                black_box(&qd),
                black_box(&qdd),
                black_box(&g),
            )
        })
    });

    c.bench_function("crba/showcase6", |b| {
        b.iter(|| crba(black_box(&m), black_box(&q)))
    });

    c.bench_function("forward_dynamics/showcase6", |b| {
        b.iter(|| {
            forward_dynamics(
                black_box(&m),
                black_box(&q),
                black_box(&qd),
                black_box(&tau),
                black_box(&g),
            )
        })
    });
}

criterion_group!(benches, bench_dynamics);
criterion_main!(benches);
