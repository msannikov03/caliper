//! Collision-aware motion planning: RRT-Connect a smoothed, collision-free
//! joint-space path through a scene with a ground plane and an obstacle box, then
//! independently re-verify it.
//!
//! Run with:
//!     cargo run -p caliper --example plan_demo
//!
//! NOTE: this example names `caliper::collision::WorldScene`, i.e. it relies on
//! the umbrella `caliper` crate re-exporting `caliper_collision` as `collision`
//! (the world-scene type the planner consumes). The other examples only touch the
//! always-available re-exports (model/kinematics/ik/dynamics).

use caliper::collision::WorldScene;
use caliper::model::Model;
use caliper::planning::{Planner, PlannerConfig, path_length};
use std::sync::Arc;

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../oracle/fixtures/robots")
        .join(name)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(Model::from_urdf(&fixture("showcase6.urdf"))?);

    // A static world: a ground plane plus a box obstacle off to one side.
    let scene = WorldScene::new()
        .with_ground(0.0)
        .add_box([0.4, 0.0, 0.4], [0.1, 0.1, 0.1]);

    let cfg = PlannerConfig::default(); // deterministic (seeded splitmix64)
    let planner = Planner::new(model.clone(), scene, cfg);
    if planner.uncovered_frames() > 0 {
        println!(
            "warning: {} frame(s) have no collider (not collision-checked)",
            planner.uncovered_frames()
        );
    }

    let start = vec![0.0; model.ndof];
    let goal = vec![0.8, -0.4, 0.6, 0.0, 0.5, 0.0];

    let path = planner.plan(&start, &goal)?;
    println!("planned {} waypoints", path.len());
    println!("joint-space path length: {:.3} rad", path_length(&path));

    // The planner guarantees endpoints; re-verify every edge at finer resolution.
    let ok = planner.verify_path(&path);
    println!(
        "independent collision re-verification: {}",
        if ok { "PASS" } else { "FAIL" }
    );
    assert!(ok, "planned path must be collision-free");

    Ok(())
}
