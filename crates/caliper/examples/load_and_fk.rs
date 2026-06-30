//! Load a URDF and read the tool pose — the "hello world" of the engine.
//!
//! Run with:
//!     cargo run -p caliper --example load_and_fk
//!
//! Everything is reached through the umbrella `caliper` crate, which re-exports
//! the engine modules (`model`, `kinematics`, `spatial`, ...). The robot is the
//! oracle test fixture, located relative to this crate via `CARGO_MANIFEST_DIR`.

use caliper::kinematics::{fk_frame, fk_tip};
use caliper::model::Model;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    // crates/caliper -> repo root -> oracle/fixtures/robots
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../oracle/fixtures/robots")
        .join(name)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("caliper v{}", caliper::VERSION);

    let model = Model::from_urdf(&fixture("showcase6.urdf"))?;
    println!("loaded `{}` — {} DOF", model.name, model.ndof);
    println!("joints: {:?}", model.joint_names);

    let tip = model.tip_frame();
    println!("tool frame: `{}`", model.frame_name(tip));

    // Forward kinematics at the zero configuration.
    let q0 = vec![0.0; model.ndof];
    let home = fk_tip(&model, &q0);
    let [x, y, z] = home.translation();
    println!("tip @ q=0:      [{x:+.4}, {y:+.4}, {z:+.4}] m");

    // ...and at a non-trivial pose. `fk_frame` lets you target any frame.
    let q = vec![0.3, -0.5, 0.8, 0.0, 0.7, -0.2];
    let pose = fk_frame(&model, &q, tip);
    let [x, y, z] = pose.translation();
    println!("tip @ q=[..]:   [{x:+.4}, {y:+.4}, {z:+.4}] m");

    Ok(())
}
