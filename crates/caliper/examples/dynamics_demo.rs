//! Rigid-body dynamics: gravity-compensation torque (inverse dynamics) and a short
//! forward simulation of the arm falling under gravity with no actuation.
//!
//! Run with:
//!     cargo run -p caliper --example dynamics_demo
//!
//! Uses RNEA for inverse dynamics and the `Simulator` (CRBA + forward dynamics +
//! a symplectic integrator) for the rollout.

use caliper::dynamics::{GRAVITY_EARTH, Simulator, rnea};
use caliper::model::Model;
use std::sync::Arc;

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../oracle/fixtures/robots")
        .join(name)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::from_urdf(&fixture("showcase6.urdf"))?;
    if !model.has_inertia {
        eprintln!("model has no inertial data — dynamics unavailable");
        return Ok(());
    }

    let q = vec![0.2, -0.5, 0.8, 0.0, 0.6, 0.0];
    let zeros = vec![0.0; model.ndof];

    // Inverse dynamics with qd = qdd = 0 yields the gravity torque G(q): the
    // static hold torque each joint must supply to balance gravity.
    let g_tau = rnea(&model, &q, &zeros, &zeros, &GRAVITY_EARTH)?;
    println!("gravity torque G(q) [N·m]:");
    for (i, name) in model.joint_names.iter().enumerate() {
        println!("  {name:>4}: {:+.4}", g_tau[i]);
    }

    // Forward sim: release the arm from `q` at rest with zero applied torque and
    // let gravity act. Damping is on by default, so it settles rather than ringing.
    let mut sim = Simulator::new(Arc::new(model.clone()))?;
    sim.reset_to(&q, &zeros)?;
    sim.set_torque(&zeros)?;

    let dt = 1.0 / 1000.0;
    let total = 0.5; // seconds
    sim.step_n(dt, (total / dt) as usize)?;

    println!("after {total:.2}s of free fall under gravity (no torque):");
    println!("  t      = {:.3} s", sim.time());
    println!("  q      = {:?}", round(sim.q()));
    println!("  qd     = {:?}", round(sim.qd()));
    println!("  energy = {:.4} J", sim.total_energy());

    Ok(())
}

fn round(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| (x * 1e3).round() / 1e3).collect()
}
