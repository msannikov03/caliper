//! Contact-stability LINTER integration tests — real MuJoCo rollouts over
//! deliberately broken (and one deliberately clean) contact scenes. All
//! behind the `mujoco` feature, exactly like `mujoco_sim.rs`:
//!
//!   MUJOCO_DYNAMIC_LINK_DIR=... DYLD_LIBRARY_PATH=... \
//!     cargo test -p caliper-sim-mujoco --features mujoco
#![cfg(feature = "mujoco")]

use caliper_model::Model;
use caliper_sim_mujoco::MujocoSim;
use caliper_sim_mujoco::lint::{LintCode, LintOptions, lint_contact_stability};
use caliper_sim_mujoco::mjcf::{ContactMaterial, MjcfOptions, PropShape, PropSpec};
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

fn box_prop(mass: f64, z: f64, material: Option<ContactMaterial>) -> PropSpec {
    PropSpec {
        name: "crate".into(),
        shape: PropShape::Box {
            half: [0.05, 0.05, 0.05],
        },
        pos: [0.6, 0.0, z],
        quat: None,
        mass,
        rgba: None,
        material,
    }
}

/// NEGATIVE: a sane scene — Rigid material, box dropped from just above the
/// plane, damped pendulum — settles clean: zero findings. And the verdict is
/// deterministic across two identical fresh rollouts.
#[test]
fn stable_scene_is_finding_free() {
    let m = model("dyn_pendulum2.urdf");
    let opt = MjcfOptions {
        ground_plane: Some(0.0),
        joint_damping: 0.5,
        default_material: Some(ContactMaterial::Rigid),
        props: vec![box_prop(0.2, 0.2, None)],
        ..Default::default()
    };
    let lint_opts = LintOptions {
        settle_duration: 2.0, // the box needs the fall + rest time
        ..Default::default()
    };
    let run = || {
        let mut sim = MujocoSim::from_caliper_model_with(&m, &opt).unwrap();
        lint_contact_stability(&mut sim, &lint_opts).unwrap()
    };
    let findings = run();
    assert!(
        findings.is_empty(),
        "clean scene produced findings: {findings:?}"
    );
    // determinism: fresh identical sim ⇒ identical (empty) verdict
    assert!(run().is_empty());
}

/// C001: the canonical explosion recipe — solref timeconst 100× BELOW the
/// 2×timestep stability bound, nearly undamped, rock-hard solimp, a heavy box
/// starting deep inside the plane. The linter must call it an explosion and
/// point at solref/timestep.
#[test]
fn stiff_underdamped_penetrating_scene_is_c001() {
    let m = model("dyn_pendulum2.urdf");
    let opt = MjcfOptions {
        timestep: 0.01,
        ground_plane: Some(0.0),
        default_material: Some(ContactMaterial::Custom {
            solref: (1e-4, 0.01), // timeconst << 2h = 0.02, dampratio ~ 0
            solimp: (0.99, 0.999, 1e-5),
            friction: (1.0, 0.005, 0.0001),
        }),
        // half-height 0.05 at z = 0.01: starts 0.04 m INSIDE the plane
        props: vec![box_prop(10.0, 0.01, None)],
        ..Default::default()
    };
    let mut sim = MujocoSim::from_caliper_model_with(&m, &opt).unwrap();
    let findings = lint_contact_stability(&mut sim, &LintOptions::default()).unwrap();
    assert!(
        findings.iter().any(|f| f.code == LintCode::C001Explosion),
        "no C001 on the explosion scene: {findings:?}"
    );
    let c001 = findings
        .iter()
        .find(|f| f.code == LintCode::C001Explosion)
        .unwrap();
    assert!(
        c001.suggestion.contains("solref timeconst") && c001.suggestion.contains("timestep"),
        "unhelpful suggestion: {}",
        c001.suggestion
    );
}

/// C002: an ultra-soft contact (impedance capped at 0.1, ramping over 5 cm)
/// under a heavy box — it sinks into the floor and STAYS there. The linter
/// must report persistent penetration (and NOT an explosion: the scene is
/// perfectly calm, just buried).
#[test]
fn ultra_soft_contact_under_heavy_box_is_c002() {
    let m = model("dyn_pendulum2.urdf");
    let opt = MjcfOptions {
        ground_plane: Some(0.0),
        joint_damping: 0.5,
        default_material: Some(ContactMaterial::Custom {
            solref: (0.05, 1.0),       // slow, critically damped — no ringing
            solimp: (0.05, 0.1, 0.05), // nearly no push-back for 5 cm
            friction: (1.0, 0.005, 0.0001),
        }),
        // drop from just above the plane so all the action is sinking
        props: vec![box_prop(20.0, 0.06, None)],
        ..Default::default()
    };
    let lint_opts = LintOptions {
        settle_duration: 2.0,
        ..Default::default()
    };
    let mut sim = MujocoSim::from_caliper_model_with(&m, &opt).unwrap();
    let findings = lint_contact_stability(&mut sim, &lint_opts).unwrap();
    assert!(
        findings.iter().any(|f| f.code == LintCode::C002Penetration),
        "no C002 on the buried-box scene: {findings:?}"
    );
    assert!(
        !findings.iter().any(|f| f.code == LintCode::C001Explosion),
        "calm sinking misread as an explosion: {findings:?}"
    );
    let c002 = findings
        .iter()
        .find(|f| f.code == LintCode::C002Penetration)
        .unwrap();
    assert!(
        c002.suggestion.contains("solimp"),
        "unhelpful suggestion: {}",
        c002.suggestion
    );
}

/// Bad lint options are rejected before any stepping happens.
#[test]
fn bad_lint_options_rejected_at_entry() {
    let m = model("dyn_pendulum2.urdf");
    let mut sim = MujocoSim::from_caliper_model(&m).unwrap();
    let bad = LintOptions {
        observe_duration: -1.0,
        ..Default::default()
    };
    assert!(lint_contact_stability(&mut sim, &bad).is_err());
    assert_eq!(sim.time(), 0.0, "rejected options must not step the sim");
}
