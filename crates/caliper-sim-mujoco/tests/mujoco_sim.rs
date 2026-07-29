//! Integration tests that ACTUALLY link and run MuJoCo — all behind the
//! `mujoco` feature, so a default `cargo test` needs no MuJoCo on the machine.
//!
//! Run with:
//!   MUJOCO_DYNAMIC_LINK_DIR=... DYLD_LIBRARY_PATH=... \
//!     cargo test -p caliper-sim-mujoco --features mujoco
#![cfg(feature = "mujoco")]

use caliper_hal::{ControlLoop, Gains, HoldSetpoint, RobotBackend};
use caliper_model::Model;
use caliper_sim_mujoco::mjcf::{Actuation, MjcfOptions, PropShape, PropSpec};
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

/// Raw-MJCF entry, UNNAMED joint: legal MJCF that used to die with
/// `MissingJoint("joint0")` because the placeholder fallback name can never
/// resolve through `name_to_id` — joints are now addressed by document id.
#[test]
fn raw_mjcf_unnamed_joint_resolves_by_id() {
    let xml = r#"
      <mujoco model="anon">
        <compiler angle="radian"/>
        <option timestep="0.001"/>
        <worldbody>
          <body name="a" pos="0 0 0.1">
            <joint type="hinge" axis="0 1 0"/>
            <inertial pos="0 0 0.1" mass="1" diaginertia="0.01 0.01 0.002"/>
            <body name="b" pos="0 0 0.2">
              <joint name="jb" type="slide" axis="0 0 1"/>
              <inertial pos="0 0 0.1" mass="1" diaginertia="0.01 0.01 0.002"/>
            </body>
          </body>
        </worldbody>
      </mujoco>"#;
    let mut sim = MujocoSim::from_mjcf(xml).expect("unnamed joints are valid MJCF");
    assert_eq!(sim.ndof(), 2);
    assert_eq!(sim.joint_names(), ["joint0", "jb"]);
    sim.set_state(&[0.2, 0.05], &[0.0, 0.0]).unwrap();
    sim.step(0.05).unwrap();
    assert!(sim.qpos().iter().all(|x| x.is_finite()));
}

/// Sanitization seam: the generator emits `left_arm` for
/// `<joint name="left arm">` (and unescapes `&amp;` at XML parse), so the sim
/// must resolve the REGISTERED spelling — building from the generator's own
/// output used to fail with `MissingJoint`. The caliper spelling stays the
/// user-facing one.
#[test]
fn sanitized_joint_and_prop_names_resolve() {
    let urdf = r#"<?xml version="1.0"?>
      <robot name="spacey">
        <link name="base"/>
        <link name="arm link">
          <inertial><origin xyz="0 0 0.2" rpy="0 0 0"/><mass value="0.5"/>
            <inertia ixx="0.007" ixy="0" ixz="0" iyy="0.007" iyz="0" izz="0.0002"/></inertial>
        </link>
        <joint name="left arm" type="revolute">
          <parent link="base"/><child link="arm link"/>
          <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
          <limit lower="-3.14" upper="3.14" effort="50" velocity="20"/>
        </joint>
      </robot>"#;
    let path = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("spacey.urdf");
    std::fs::write(&path, urdf).unwrap();
    let m = Model::from_urdf(&path).unwrap();
    assert_eq!(m.joint_names, ["left arm"], "fixture premise");
    let opt = MjcfOptions {
        props: vec![PropSpec {
            name: "my ball".into(), // whitespace in a prop name too
            shape: PropShape::Sphere { r: 0.05 },
            pos: [0.5, 0.0, 0.3],
            quat: None,
            mass: 0.1,
            rgba: None,
            material: None,
        }],
        ..Default::default()
    };
    let mut sim = MujocoSim::from_caliper_model_with(&m, &opt)
        .expect("the generator's own output must resolve");
    assert_eq!(
        sim.joint_names(),
        ["left arm"],
        "caliper spelling preserved"
    );
    assert_eq!(sim.prop_names(), ["my ball"]);
    sim.set_state(&[0.3], &[0.0]).unwrap();
    sim.step(0.05).unwrap();
    assert!(sim.qpos()[0].is_finite());
    // the prop rides along under its MuJoCo-registered body name
    assert!(sim.body_pose("prop_my_ball").is_ok());
    assert_eq!(sim.prop_poses().len(), 1);
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

/// (b2) HULL MESHES: a model exported WITH `export_hull_meshes` — a pendulum
/// whose only collider is a mesh (unit-cube STL → ConvexHull) — LOADS in the
/// real MuJoCo compiler with zero skipped colliders, and the hull geom makes
/// a sane contact when the cube swings down onto a ground plane.
#[test]
fn hull_mesh_collider_loads_and_contacts() {
    // Same layout as `sphere_settles_on_ground_plane`, but the tip collider
    // is the shared unit-cube fixture (corners ±0.5) scaled to a 0.12 m cube,
    // referenced by ABSOLUTE path so the temp URDF resolves it.
    let stl = format!(
        "{}/../../oracle/fixtures/robots/unit_cube.stl",
        env!("CARGO_MANIFEST_DIR")
    );
    let urdf = format!(
        r#"<?xml version="1.0"?>
      <robot name="hulltap">
        <link name="base"/>
        <link name="arm">
          <inertial><origin xyz="0 0 0.2" rpy="0 0 0"/><mass value="0.5"/>
            <inertia ixx="0.007" ixy="0" ixz="0" iyy="0.007" iyz="0" izz="0.0002"/></inertial>
          <collision><origin xyz="0 0 0.4" rpy="0 0 0"/>
            <geometry><mesh filename="{stl}" scale="0.12 0.12 0.12"/></geometry></collision>
        </link>
        <joint name="j1" type="revolute">
          <parent link="base"/><child link="arm"/>
          <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
          <limit lower="-6.28" upper="6.28" effort="50" velocity="20"/>
        </joint>
      </robot>"#
    );
    let path = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("hulltap.urdf");
    std::fs::write(&path, urdf).unwrap();
    let m = Model::from_urdf(&path).unwrap();

    // Cube center orbit radius 0.4, half-extent 0.06: hanging straight down
    // the cube bottom (-0.46) is below a plane at z=-0.35 — it must come to
    // rest ON the plane instead.
    let opt = MjcfOptions {
        ground_plane: Some(-0.35),
        joint_damping: 0.5,
        export_hull_meshes: true,
        ..Default::default()
    };
    let mut sim = MujocoSim::from_caliper_model_with(&m, &opt).unwrap();
    assert_eq!(
        sim.skipped_hull_colliders(),
        0,
        "hull collider skipped despite export_hull_meshes"
    );
    sim.set_state(&[2.0], &[0.0]).unwrap();
    assert_eq!(sim.ncon(), 0, "must start contact-free");
    sim.step(3.0).unwrap(); // fall + settle (damped)
    let contacts = sim.contacts();
    assert!(!contacts.is_empty(), "no contact after settling");
    let c = contacts
        .iter()
        .find(|c| c.geom1 == "caliper_ground" || c.geom2 == "caliper_ground")
        .expect("no contact with the ground plane");
    assert!(
        c.geom1 == "col0_arm" || c.geom2 == "col0_arm",
        "ground contact is not with the hull geom: {c:?}"
    );
    assert!(c.depth > 0.0, "non-penetrating contact reported: {c:?}");
    assert!(
        c.normal[2].abs() > 0.9,
        "ground contact normal should be ±z: {c:?}"
    );
    // Contact point sits on the plane, within the cube's half-diagonal.
    assert!(
        (c.pos[2] - (-0.35)).abs() < 0.11,
        "contact z off-plane: {c:?}"
    );
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

/// disable() DE-ENERGIZES, not just latches: a torque commanded while enabled
/// used to stay in `qfrc_applied` and keep driving the arm through later
/// `step`s — after disable the motion must be bit-identical to a
/// never-commanded backend.
#[test]
fn disable_clears_stale_torques() {
    let m = model("dyn_pendulum2.urdf");
    let mut b = MujocoBackend::new(&m).unwrap();
    b.enable().unwrap();
    b.command_joint_torques(&[3.0, -2.0]).unwrap();
    b.disable().unwrap();
    assert!(
        b.sim().mj_data().qfrc_applied().iter().all(|&f| f == 0.0),
        "stale applied torques survive disable()"
    );
    let mut passive = MujocoBackend::new(&m).unwrap();
    for _ in 0..200 {
        b.step(1e-3).unwrap();
        passive.step(1e-3).unwrap();
    }
    assert_eq!(
        b.joint_positions(),
        passive.joint_positions(),
        "disabled backend still driven by pre-disable torque"
    );
}

/// Servo variant of the same semantic: disable freezes `ctrl` at the CURRENT
/// position, so a stale target commanded before disable stops pulling.
#[test]
fn disable_freezes_servo_target() {
    let m = model("dyn_pendulum2.urdf");
    let opt = MjcfOptions {
        actuation: Actuation::PositionServo { kp: 30.0, kv: 3.0 },
        ..Default::default()
    };
    let mut b = MujocoBackend::with_options(&m, &opt).unwrap();
    b.enable().unwrap();
    b.command_joint_positions(&[0.8, -0.6]).unwrap();
    for _ in 0..100 {
        b.step(1e-3).unwrap(); // partway toward the target
    }
    b.disable().unwrap();
    let q_hold = b.joint_positions();
    assert_eq!(
        b.sim().mj_data().ctrl(),
        q_hold.as_slice(),
        "ctrl not frozen at the disable pose"
    );
}

/// (f) PROPS: a free box dropped above the ground plane settles ON it
/// (z ≈ half-height) with a live ground contact; the same pose is readable
/// through `prop_poses` and name-resolved `body_pose`.
#[test]
fn box_prop_settles_on_plane() {
    let m = model("dyn_pendulum2.urdf");
    let opt = MjcfOptions {
        ground_plane: Some(0.0),
        joint_damping: 0.5,
        props: vec![PropSpec {
            name: "crate".into(),
            shape: PropShape::Box {
                half: [0.05, 0.05, 0.05],
            },
            pos: [0.6, 0.0, 0.4],
            quat: None,
            mass: 0.2,
            rgba: Some([0.8, 0.2, 0.2, 1.0]),
            material: None,
        }],
        ..Default::default()
    };
    let mut sim = MujocoSim::from_caliper_model_with(&m, &opt).unwrap();
    assert_eq!(sim.prop_names(), ["crate"]);
    // initial pose = the spec, identity orientation
    let (p0, q0) = sim.body_pose("prop_crate").unwrap();
    assert!((p0[2] - 0.4).abs() < 1e-12 && (q0[0] - 1.0).abs() < 1e-12);
    sim.step(2.0).unwrap(); // fall 0.35 m + settle
    let props = sim.prop_poses();
    assert_eq!(props.len(), 1);
    let (name, pos, quat) = &props[0];
    assert_eq!(name, "crate");
    assert!(
        (pos[2] - 0.05).abs() < 0.01,
        "box not resting on the plane: z = {}",
        pos[2]
    );
    assert!(
        (pos[0] - 0.6).abs() < 0.05 && pos[1].abs() < 0.05,
        "box slid: {pos:?}"
    );
    assert!(quat[0].abs() > 0.99, "box tumbled: {quat:?}");
    let contacts = sim.contacts();
    assert!(
        contacts
            .iter()
            .any(|c| c.geom1 == "caliper_ground" || c.geom2 == "caliper_ground"),
        "no ground contact after settling: {contacts:?}"
    );
    // body_pose agrees with prop_poses; unknown bodies fail loudly
    let (bp, bq) = sim.body_pose("prop_crate").unwrap();
    assert_eq!(bp, *pos);
    assert_eq!(bq, *quat);
    assert!(sim.body_pose("nope").is_err());
}

/// (g) Prop trajectories are bitwise deterministic across identical runs.
#[test]
fn prop_pose_determinism() {
    let m = model("dyn_pendulum2.urdf");
    let opt = MjcfOptions {
        ground_plane: Some(0.0),
        props: vec![PropSpec {
            name: "b".into(),
            shape: PropShape::Sphere { r: 0.05 },
            pos: [0.5, 0.1, 0.5],
            quat: None,
            mass: 0.1,
            rgba: None,
            material: None,
        }],
        ..Default::default()
    };
    let run = || {
        let mut sim = MujocoSim::from_caliper_model_with(&m, &opt).unwrap();
        let mut trace: Vec<u64> = Vec::new();
        for _ in 0..300 {
            sim.step(1e-3).unwrap();
            for (_, p, q) in sim.prop_poses() {
                trace.extend(p.iter().chain(q.iter()).map(|x| x.to_bits()));
            }
        }
        trace
    };
    assert_eq!(run(), run(), "prop trajectories diverged bitwise");
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

// ===== grasp welds (B2) =====

/// The grasp-weld fixture: `gripper_arm` hanging over a 5 cm cube that rests
/// on the ground plane, with the jaw already 1 mm into the cube's top face —
/// so a grasp can be tested without racing a falling prop.
fn grasp_rig(attach: bool) -> MujocoSim {
    let m = model("gripper_arm.urdf");
    let opt = MjcfOptions {
        ground_plane: Some(0.0),
        attach_link: attach.then(|| "jaw".to_string()),
        props: vec![PropSpec {
            name: "cube".into(),
            shape: PropShape::Box { half: [0.05; 3] },
            pos: [0.0, 0.0, 0.05],
            quat: None,
            mass: 0.05,
            rgba: None,
            material: None,
        }],
        ..Default::default()
    };
    MujocoSim::from_caliper_model_with(&m, &opt).unwrap()
}

/// Pose of the cube relative to the jaw body — the quantity a weld is supposed
/// to hold constant.
fn cube_rel_jaw(sim: &MujocoSim) -> ([f64; 3], [f64; 4]) {
    let (jp, jq) = sim.body_pose("b_gripper").unwrap();
    let (_, pp, pq) = sim.prop_poses().into_iter().next().unwrap();
    // R_jawᵀ·(x_cube − x_jaw), q_jaw⁻¹⊗q_cube, with quats as [w,x,y,z].
    let q = nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
        jq[0], jq[1], jq[2], jq[3],
    ));
    let p = nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
        pq[0], pq[1], pq[2], pq[3],
    ));
    let d =
        q.inverse_transform_vector(&(nalgebra::Vector3::from(pp) - nalgebra::Vector3::from(jp)));
    let r = q.inverse() * p;
    ([d.x, d.y, d.z], [r.w, r.i, r.j, r.k])
}

/// A model built WITHOUT `attach_link` has no welds at all, and says so
/// instead of pretending a grasp happened.
#[test]
fn no_attach_link_means_no_welds() {
    let mut sim = grasp_rig(false);
    assert!(sim.weld_props().is_empty());
    let err = sim.set_weld_active("cube", true, true).unwrap_err();
    assert!(err.to_string().contains("no grasp weld"), "got: {err}");
    assert!(sim.weld_active("cube").is_err());
}

/// Welds are emitted inactive, name-addressed by prop, and refuse unknown props.
#[test]
fn welds_start_inactive_and_are_addressed_by_prop() {
    let mut sim = grasp_rig(true);
    assert_eq!(sim.weld_props(), ["cube"]);
    assert!(!sim.weld_active("cube").unwrap());
    let err = sim.set_weld_active("nope", true, true).unwrap_err();
    assert!(err.to_string().contains("`nope`"), "got: {err}");
    // an inactive weld does not hold anything: the cube is still a free body
    sim.forward();
    assert!(!sim.weld_active("cube").unwrap());
}

/// The jaw touches the cube, and `props_touching_robot` reports contact with
/// the ROBOT only — the cube also rests on the ground, which must not count.
#[test]
fn prop_contact_with_the_robot_is_distinguished_from_the_ground() {
    let mut sim = grasp_rig(true);
    sim.forward();
    assert!(sim.ncon() >= 2, "expected jaw + ground contacts");
    assert!(
        sim.contacts()
            .iter()
            .any(|c| c.geom1 == "caliper_ground" || c.geom2 == "caliper_ground"),
        "the cube should be resting on the ground"
    );
    assert_eq!(sim.props_touching_robot(), ["cube"]);

    // Lift the arm clear: the cube still touches the ground, but nothing else.
    sim.set_state(&[1.2, 0.0, 0.0], &[0.0; 3]).unwrap();
    sim.forward();
    assert!(
        sim.props_touching_robot().is_empty(),
        "a cube touching only the ground is not touching the robot"
    );
}

/// Activation with `capture_relpose` is CONTINUOUS: the prop does not snap to
/// the pose the weld was authored with. The same activation without capture
/// yanks it back — that contrast is the whole point of the capture.
#[test]
fn relpose_capture_makes_activation_snap_free() {
    // Move the arm well away from the pose the document was compiled at, so
    // the authored relative pose is stale by tens of centimeters.
    let moved = [0.9, 0.0, 0.0];

    let mut with = grasp_rig(true);
    with.set_state(&moved, &[0.0; 3]).unwrap();
    with.forward();
    let before = with.prop_poses()[0].1;
    with.set_weld_active("cube", true, true).unwrap();
    with.step_once();
    let after = with.prop_poses()[0].1;
    let snap = (nalgebra::Vector3::from(after) - nalgebra::Vector3::from(before)).norm();
    assert!(
        snap < 1e-3,
        "captured activation moved the prop {:.3} mm — it must be continuous",
        snap * 1000.0
    );

    let mut without = grasp_rig(true);
    without.set_state(&moved, &[0.0; 3]).unwrap();
    without.forward();
    let before = without.prop_poses()[0].1;
    without.set_weld_active("cube", true, false).unwrap();
    for _ in 0..50 {
        without.step_once();
    }
    let after = without.prop_poses()[0].1;
    let pull = (nalgebra::Vector3::from(after) - nalgebra::Vector3::from(before)).norm();
    assert!(
        pull > 0.05,
        "without capture the weld should drag the prop to its authored pose, moved {pull:.4} m"
    );
}

/// The full grasp: attach, carry the prop through a swing, release, and see it
/// fall. Plus reset, which must drop whatever was held.
#[test]
fn welded_prop_is_carried_then_released() {
    let mut sim = grasp_rig(true);
    sim.forward();
    assert_eq!(sim.props_touching_robot(), ["cube"]);

    sim.set_weld_active("cube", true, true).unwrap();
    let rel0 = cube_rel_jaw(&sim);
    let z0 = sim.prop_poses()[0].1[2];

    // Swing j1: the jaw sweeps up and sideways, so a carried cube must leave
    // the ground with it.
    for _ in 0..600 {
        sim.set_joint_torques(&[12.0, 0.0, 0.0]).unwrap();
        sim.step_once();
    }
    let held = sim.prop_poses()[0].1;
    assert!(
        held[2] > z0 + 0.02,
        "the carried cube never left the ground (z {z0} → {})",
        held[2]
    );
    // The relative pose is what the weld holds — MuJoCo's weld is a SOFT
    // constraint, so allow a millimetre-scale sag, not a rigid identity.
    let rel1 = cube_rel_jaw(&sim);
    let dp = (nalgebra::Vector3::from(rel1.0) - nalgebra::Vector3::from(rel0.0)).norm();
    assert!(
        dp < 5e-3,
        "carried cube drifted {:.2} mm in the jaw frame",
        dp * 1000.0
    );

    // Release: the cube keeps its state and falls under gravity.
    sim.set_weld_active("cube", false, false).unwrap();
    assert!(!sim.weld_active("cube").unwrap());
    let dropped_from = sim.prop_poses()[0].1[2];
    for _ in 0..800 {
        sim.set_joint_torques(&[12.0, 0.0, 0.0]).unwrap();
        sim.step_once();
    }
    let landed = sim.prop_poses()[0].1[2];
    assert!(
        landed < dropped_from - 0.01,
        "released cube did not fall (z {dropped_from} → {landed})"
    );

    // Reset releases everything, whatever was held.
    sim.set_weld_active("cube", true, true).unwrap();
    assert!(sim.weld_active("cube").unwrap());
    sim.reset(&[0.0, 0.0, 0.0]).unwrap();
    assert!(
        !sim.weld_active("cube").unwrap(),
        "reset must drop a held prop"
    );
}
