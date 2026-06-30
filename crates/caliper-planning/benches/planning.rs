//! Criterion benchmark for the RRT-Connect planner: a collision-free plan on the
//! `collide_arm` fixture in an empty scene. The planner is deterministic (fixed
//! seed in `PlannerConfig::default`), so each iteration does identical work.
use caliper_collision::WorldScene;
use caliper_model::Model;
use caliper_planning::{Planner, PlannerConfig};
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

fn bench_planning(c: &mut Criterion) {
    let m = load("collide_arm.urdf");
    let start = [0.0_f64, 0.0, 0.0];
    let goal = [0.4_f64, -0.4, 0.4];

    c.bench_function("rrt_connect/collide_arm", |b| {
        b.iter(|| {
            let planner = Planner::new(Arc::clone(&m), WorldScene::new(), PlannerConfig::default());
            planner.plan(black_box(&start), black_box(&goal))
        })
    });
}

criterion_group!(benches, bench_planning);
criterion_main!(benches);
