//! Criterion benchmark for the collision query (self-collision + world) on the
//! `collide_arm` fixture, covering a clear and a near-contact configuration.
use caliper_collision::{CollisionModel, WorldScene};
use caliper_model::Model;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::path::Path;
use std::sync::Arc;

fn load(name: &str) -> Arc<Model> {
    let p = format!(
        "{}/../../oracle/fixtures/robots/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    Arc::new(Model::from_urdf(Path::new(&p)).expect("load fixture"))
}

fn bench_collision(c: &mut Criterion) {
    let m = load("collide_arm.urdf");
    let cm = CollisionModel::new(m, WorldScene::new().with_ground(-0.05), 0.0);
    let clear = [0.0_f64, 0.0, 0.0];
    let folded = [0.0_f64, std::f64::consts::PI, std::f64::consts::PI];

    c.bench_function("query_clear/collide_arm", |b| {
        b.iter(|| cm.query(black_box(&clear)))
    });

    c.bench_function("query_folded/collide_arm", |b| {
        b.iter(|| cm.query(black_box(&folded)))
    });
}

criterion_group!(benches, bench_collision);
criterion_main!(benches);
