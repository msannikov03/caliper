//! Inverse kinematics two ways: the general numeric solver and, when the arm is a
//! canonical spherical-wrist 6R, the closed-form analytic solver.
//!
//! Run with:
//!     cargo run -p caliper --example ik_demo
//!
//! We pick a known joint vector, run FK to get a reachable target pose, then ask
//! IK to recover a configuration that hits it.

use caliper::ik::{IkOpts, analytic_ik_6r, ik};
use caliper::kinematics::fk_frame;
use caliper::model::Model;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../oracle/fixtures/robots")
        .join(name)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::from_urdf(&fixture("showcase6.urdf"))?;
    let tip = model.tip_frame();

    // A reachable target = FK of a known configuration.
    let q_true = vec![0.4, -0.6, 0.9, 0.2, -0.5, 0.3];
    let target = fk_frame(&model, &q_true, tip);
    let [tx, ty, tz] = target.translation();
    println!("target tip: [{tx:+.4}, {ty:+.4}, {tz:+.4}] m");

    // --- Numeric (damped least squares, multi-restart) ---
    let seed = vec![0.0; model.ndof];
    let res = ik(&model, tip, &target, &seed, &IkOpts::default());
    println!(
        "numeric : success={} residual={:.2e} iters={} restarts={}",
        res.success, res.residual, res.iters, res.restarts_used
    );
    if res.success {
        let reached = fk_frame(&model, &res.q, tip);
        let err = (reached.translation_vec() - target.translation_vec()).norm();
        println!("          position error {err:.2e} m");
    }

    // --- Analytic (closed form) — falls back gracefully if the structure is not
    // a recognised spherical-wrist 6R. ---
    match analytic_ik_6r(&model, tip, &target, Some(&seed)) {
        Some(branches) if !branches.is_empty() => {
            println!("analytic: {} branch(es) found", branches.len());
            let best = &branches[0]; // seed-nearest branch is placed first
            let reached = fk_frame(&model, best, tip);
            let err = (reached.translation_vec() - target.translation_vec()).norm();
            println!("          nearest branch position error {err:.2e} m");
        }
        Some(_) => println!("analytic: structure recognised but pose unreachable"),
        None => println!("analytic: not a spherical-wrist 6R — use the numeric solver"),
    }

    Ok(())
}
