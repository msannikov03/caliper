//! Integration tests that ACTUALLY link and run MuJoCo — all behind the
//! `mujoco` feature, so a default `cargo test` needs no MuJoCo on the machine.
//!
//! Run with:
//!   MUJOCO_DYNAMIC_LINK_DIR=... DYLD_LIBRARY_PATH=... \
//!     cargo test -p caliper-sim-mujoco --features mujoco
#![cfg(feature = "mujoco")]

use caliper_hal::{ControlLoop, Gains, HoldSetpoint, RobotBackend};
use caliper_model::Model;
use caliper_sim_mujoco::mjcf::{Actuation, MjcfOptions};
use caliper_sim_mujoco::{MujocoBackend, MujocoSim};
use std::path::Path;
use std::sync::Arc;

fn model(name: &str) -> Arc<Model> {
    Arc::new(
        Model::from_urdf(Path::new(&format!(
            "{}/../../oracle/fixtures/robots/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        )))
        .unwrap(),
    )
}

/// (0) MJCF-from-model round-trips through the real MuJoCo compiler.
#[test]
fn mjcf_roundtrip_loads() {
    for name in ["dyn_pendulum2.urdf", "showcase6.urdf", "collide_arm.urdf"] {
        let m = model(name);
        let sim = MujocoSim::from_caliper_model(&m).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(sim.ndof(), m.ndof, "{name}");
        assert_eq!(sim.qpos().len(), m.ndof, "{name}");
        assert_eq!(sim.nu(), 0, "{name}: TorqueDirect must emit no actuators");
        assert_eq!(sim.skipped_hull_colliders(), 0, "{name}");
    }
}

/// Raw-MJCF entry: joints mapped in document order, stepping works.
#[test]
fn raw_mjcf_loads_and_steps() {
    let xml = r#"
      <mujoco model="raw2">
        <compiler angle="radian"/>
        <option timestep="0.001"/>
        <worldbody>
          <body name="a" pos="0 0 0.1">
            <joint name="ja" type="hinge" axis="0 1 0"/>
            <inertial pos="0 0 0.1" mass="1" diaginertia="0.01 0.01 0.002"/>
            <body name="b" pos="0 0 0.2">
              <joint name="jb" type="hinge" axis="0 1 0"/>
              <inertial pos="0 0 0.1" mass="1" diaginertia="0.01 0.01 0.002"/>
            </body>
          </body>
        </worldbody>
      </mujoco>"#;
    let mut sim = MujocoSim::from_mjcf(xml).unwrap();
    assert_eq!(sim.ndof(), 2);
    assert_eq!(sim.joint_names(), ["ja", "jb"]);
    sim.set_state(&[0.2, -0.1], &[0.0, 0.0]).unwrap();
    sim.step(0.05).unwrap();
    assert!((sim.time() - 0.05).abs() < 1e-12);
    // dt that is not a multiple of h fails loudly.
    assert!(sim.step(0.0015).is_err());
}

/// (a) An arm under gravity with zero torque sags: qpos changes, stays finite.
#[test]
fn gravity_sag_zero_torque() {
    let m = model("showcase6.urdf");
    let mut sim = MujocoSim::from_caliper_model(&m).unwrap();
    let q0 = [0.1, 0.3, -0.2, 0.2, 0.1, 0.1];
    sim.set_state(&q0, &[0.0; 6]).unwrap();
    sim.step(0.5).unwrap();
    let q = sim.qpos();
    assert!(q.iter().all(|x| x.is_finite()));
    let moved = q
        .iter()
        .zip(q0.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(moved > 1e-3, "arm did not sag under gravity: {q:?}");
}

/// (b) CONTACT: a 1-dof pendulum with a sphere tip swings down onto a ground
/// plane and rests there — contacts non-empty with a sane normal and depth.
#[test]
fn sphere_settles_on_ground_plane() {
    let urdf = r#"<?xml version="1.0"?>
      <robot name="tap">
        <link name="base"/>
        <link name="arm">
          <inertial><origin xyz="0 0 0.2" rpy="0 0 0"/><mass value="0.5"/>
            <inertia ixx="0.007" ixy="0" ixz="0" iyy="0.007" iyz="0" izz="0.0002"/></inertial>
          <collision><origin xyz="0 0 0.4" rpy="0 0 0"/>
            <geometry><sphere radius="0.06"/></geometry></collision>
        </link>
        <joint name="j1" type="revolute">
          <parent link="base"/><child link="arm"/>
          <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
          <limit lower="-6.28" upper="6.28" effort="50" velocity="20"/>
        </joint>
      </robot>"#;
    let path = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tap.urdf");
    std::fs::write(&path, urdf).unwrap();
    let m = Model::from_urdf(&path).unwrap();

    // Plane at z=-0.35; the sphere center orbit has radius 0.4, so hanging
    // straight down (q=π) the sphere bottom (-0.46) is well below the plane —
    // it must come to rest ON the plane instead.
    let opt = MjcfOptions {
        ground_plane: Some(-0.35),
        joint_damping: 0.5,
        ..Default::default()
    };
    let mut sim = MujocoSim::from_caliper_model_with(&m, &opt).unwrap();
    sim.set_state(&[2.0], &[0.0]).unwrap();
    assert_eq!(sim.ncon(), 0, "must start contact-free");
    sim.step(3.0).unwrap(); // fall + settle (damped)
    let contacts = sim.contacts();
    assert!(!contacts.is_empty(), "no contact after settling");
    let c = &contacts[0];
    assert!(
        c.geom1 == "caliper_ground" || c.geom2 == "caliper_ground",
        "contact not with the ground plane: {c:?}"
    );
    assert!(c.depth > 0.0, "non-penetrating contact reported: {c:?}");
    assert!(
        c.normal[2].abs() > 0.9,
        "ground contact normal should be ±z: {c:?}"
    );
    // Contact point sits on the plane, near the sphere's world position.
    assert!(
        (c.pos[2] - (-0.35)).abs() < 0.06,
        "contact z off-plane: {c:?}"
    );
    // The wrench has a positive normal component (plane pushes back).
    let f = sim.contact_force(0);
    assert!(f[0] > 0.0, "no repulsive normal force: {f:?}");
    // And the joint is finite / at rest-ish under damping.
    assert!(sim.qvel()[0].abs() < 1.0);
}

/// (c) Determinism: identical runs are BITWISE identical (same binary + same
/// pinned libmujoco; warmstart included because both runs share every step).
#[test]
fn bitwise_deterministic_runs() {
    let m = model("dyn_pendulum2.urdf");
    let run = || {
        let mut sim = MujocoSim::from_caliper_model(&m).unwrap();
        sim.reset(&[0.3, -0.2]).unwrap();
        let mut trace: Vec<u64> = Vec::new();
        for k in 0..500 {
            let t = k as f64 * 1e-3;
            let tau = [0.5 * (2.0 * t).sin(), 0.3 * (3.0 * t).cos()];
            sim.set_joint_torques(&tau).unwrap();
            sim.step(1e-3).unwrap();
            for q in sim.qpos() {
                trace.push(q.to_bits());
            }
        }
        trace
    };
    assert_eq!(run(), run(), "two identical runs diverged bitwise");
}

/// (d) The EXISTING ControlLoop + SafetyMonitor drive a MuJoCo backend
/// unchanged: computed-torque toward a hold target converges.
#[test]
fn control_loop_drives_mujoco_backend() {
    let m = model("showcase6.urdf");
    let backend = MujocoBackend::new(&m).unwrap();
    let dt = 1e-3; // == MJCF timestep default, so loop tick == one mj_step
    let mut lp = ControlLoop::new(backend, m.clone(), dt)
        .unwrap()
        .with_gains(Gains { kp: 50.0, kd: 14.0 });
    let target = vec![0.2, -0.3, 0.3, 0.2, -0.2, 0.2];
    let e0: f64 = target.iter().map(|t| t * t).sum::<f64>().sqrt(); // q0 = 0
    let mut sp = HoldSetpoint::new(target.clone());
    lp.run_to(&mut sp, 2000).unwrap();
    assert!(!lp.monitor().is_estopped(), "safety monitor tripped");
    let q = lp.backend().joint_positions();
    let e1: f64 = q
        .iter()
        .zip(target.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    assert!(
        e1 < 0.1 * e0,
        "tracking error did not decrease enough: {e0} -> {e1} (q = {q:?})"
    );
}

/// Position-servo variant: Position mode drives ctrl, Torque mode is honestly
/// unsupported.
#[test]
fn position_servo_variant() {
    let m = model("dyn_pendulum2.urdf");
    let opt = MjcfOptions {
        actuation: Actuation::PositionServo { kp: 30.0, kv: 3.0 },
        ..Default::default()
    };
    let mut b = MujocoBackend::with_options(&m, &opt).unwrap();
    assert_eq!(b.sim().nu(), 2);
    assert!(matches!(
        b.set_mode(caliper_hal::ControlMode::Torque),
        Err(caliper_hal::Error::UnsupportedMode(_))
    ));
    b.enable().unwrap();
    let target = [0.4, -0.3];
    b.command_joint_positions(&target).unwrap();
    for _ in 0..3000 {
        b.step(1e-3).unwrap();
    }
    let q = b.joint_positions();
    for i in 0..2 {
        assert!(
            (q[i] - target[i]).abs() < 0.15,
            "servo did not track: q={q:?} target={target:?}"
        );
    }
}

/// (e) Cross-check: caliper's own gravity Simulator vs MuJoCo on the
/// contact-free 2-link pendulum. Different integrators (caliper symplectic
/// Euler vs MuJoCo Euler with implicit damping), same h=1e-4, same uniform
/// damping 0.1 — qpos must agree within 2e-2 rad over a 0.3 s horizon
/// (loose tolerance is deliberate and documented; this catches sign/axis/
/// inertia mapping bugs, not integrator truncation differences).
#[test]
fn cross_check_against_caliper_simulator() {
    let m = model("dyn_pendulum2.urdf");
    let q0 = [0.3, -0.2];

    let mut cal = caliper_dynamics::Simulator::new(m.clone()).unwrap();
    cal.h_max = 1e-4;
    cal.set_state(&q0, &[0.0, 0.0]).unwrap(); // default damping = 0.1/joint

    let opt = MjcfOptions {
        timestep: 1e-4,
        joint_damping: 0.1,
        ..Default::default()
    };
    let mut mj = MujocoSim::from_caliper_model_with(&m, &opt).unwrap();
    mj.set_state(&q0, &[0.0, 0.0]).unwrap();

    for _ in 0..3000 {
        cal.step(1e-4).unwrap();
        mj.step(1e-4).unwrap();
    }
    let qm = mj.qpos();
    for (i, (qc, qmj)) in cal.q().iter().zip(qm.iter()).enumerate().take(2) {
        let d = (qc - qmj).abs();
        assert!(
            d < 2e-2,
            "joint {i}: caliper {qc} vs mujoco {qmj} (|Δ|={d})"
        );
    }
    // Sanity: the pendulum actually moved (the check above is not trivially
    // comparing two frozen states).
    assert!((qm[0] - q0[0]).abs() + (qm[1] - q0[1]).abs() > 0.05);
}
