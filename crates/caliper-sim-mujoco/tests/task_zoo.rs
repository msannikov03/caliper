//! The TASK ZOO — every `tasks/*.caliper-task.json` shipped at the repo root,
//! swept through the real loader.
//!
//! The zoo is the "works out of the box" promise: a file in here is one a user
//! can open in Studio or hand to `caliper-learn eval --task` on a fresh clone,
//! with no fetch step and no hand-editing. That promise is only as good as the
//! sweep below, which is why this is a test and not a README paragraph.
//!
//! No `mujoco` feature gate: reading, validating and JUDGING a task needs no
//! MuJoCo (see `task.rs`), so the whole zoo is checked in the default lane.
//! The proof that each scene also COMPILES and steps in a real sim lives on the
//! python side (`learn/tests/test_task_zoo.py`), which has mujoco.

use caliper_sim_mujoco::task::success::SuccessState;
use caliper_sim_mujoco::task::{TASK_VERSION, TaskSpec, load_task};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn zoo_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tasks"))
}

/// Every task file in the zoo, in sorted (i.e. numbered) order.
fn zoo_files() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(zoo_dir())
        .expect("the zoo directory exists")
        .map(|e| e.expect("readable dir entry").path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".caliper-task.json"))
        })
        .collect();
    out.sort();
    // A glob that silently matched nothing would make every assertion below
    // vacuous — the one failure mode a sweep cannot detect by itself.
    assert!(
        out.len() >= 4,
        "expected at least 4 zoo tasks, found {}",
        out.len()
    );
    out
}

fn label(path: &Path) -> String {
    path.file_name().unwrap().to_string_lossy().into_owned()
}

fn zoo() -> Vec<(PathBuf, TaskSpec)> {
    zoo_files()
        .into_iter()
        .map(|p| {
            let spec = load_task(&p).unwrap_or_else(|e| panic!("{} does not load: {e}", label(&p)));
            (p, spec)
        })
        .collect()
}

#[test]
fn every_zoo_task_loads_and_names_itself_uniquely() {
    let mut names: BTreeSet<String> = BTreeSet::new();
    for (path, spec) in zoo() {
        let what = label(&path);
        assert_eq!(spec.version, TASK_VERSION, "{what}");
        assert!(
            names.insert(spec.name.clone()),
            "{what}: task name `{}` is already used by another zoo task — names \
             are how a run is identified in an eval report",
            spec.name
        );
        // Round-trip: what the loader holds re-parses to the same task.
        let again = TaskSpec::from_json_str(&spec.to_json().unwrap(), &zoo_dir()).unwrap();
        assert_eq!(again, spec, "{what} does not round-trip");
    }
}

#[test]
fn every_zoo_task_ships_its_robot_and_its_scene_in_repo() {
    for (path, spec) in zoo() {
        let what = label(&path);
        // A zoo task must work on a fresh clone: relative path, in-repo file,
        // no `caliper fetch` step in front of it.
        assert!(
            !Path::new(&spec.robot).is_absolute(),
            "{what}: robot `{}` is an absolute path — it would not resolve on \
             anyone else's machine",
            spec.robot
        );
        assert!(
            spec.robot_path().is_file(),
            "{what}: robot resolves to {}, which is not a file",
            spec.robot_path().display()
        );
        // Every prop passes the engine's own prop rulebook (dimensions, mass,
        // material ranges, analytic inertia).
        let props = spec
            .prop_specs()
            .unwrap_or_else(|e| panic!("{what}: prop specs: {e}"));
        assert!(
            !props.is_empty(),
            "{what}: a zoo task needs something to act on"
        );
    }
}

#[test]
fn every_zoo_task_scores_props_and_zones_its_scene_declares() {
    for (path, spec) in zoo() {
        let what = label(&path);
        let pred = spec
            .success
            .as_ref()
            .unwrap_or_else(|| panic!("{what}: a zoo task must define a verdict"));
        // The loader already refuses both of these; the sweep is what makes
        // sure it was ASKED, on every shipped file.
        let known: BTreeSet<&str> = spec.scene.props.iter().map(|p| p.name.as_str()).collect();
        for prop in pred.prop_names() {
            assert!(
                known.contains(prop),
                "{what}: success scores `{prop}`, not in {known:?}"
            );
        }
        // Zone NAMES are resolved at load time, so a surviving name would mean
        // the resolver was skipped.
        let json = spec.to_json().unwrap();
        for zone in &spec.scene.zones {
            assert!(
                !json.contains(&format!("\"zone\": \"{}\"", zone.name)),
                "{what}: zone `{}` survived as a name in the resolved task",
                zone.name
            );
        }
        // Every verdict is a sentence a report can print.
        assert!(!pred.describe().is_empty(), "{what}");
    }
}

/// The invariant that is easy to violate and expensive to notice: a predicate
/// that already holds at the spawn pose.
///
/// `evaluate()` LATCHES success — one true step anywhere in the episode marks
/// the episode successful — so a task whose verdict is true at t=0 reports
/// 100% for a policy that does nothing at all. Every zoo task must start
/// unsolved.
#[test]
fn no_zoo_task_is_already_solved_at_its_start_state() {
    for (path, spec) in zoo() {
        let what = label(&path);
        // The spawn state: every prop where the file puts it, at rest (which is
        // also what a real reset produces, so `settled_speed` is answerable).
        let spawn = SuccessState::from_positions(
            spec.scene
                .props
                .iter()
                .map(|p| (p.name.clone(), p.pos))
                .collect::<Vec<_>>(),
        )
        .with_velocities(
            spec.scene
                .props
                .iter()
                .map(|p| (p.name.clone(), [0.0, 0.0, 0.0]))
                .collect::<Vec<_>>(),
        );
        let mut tracker = spec.success_tracker().unwrap();
        tracker
            .reset(Some(&spawn))
            .unwrap_or_else(|e| panic!("{what}: reset on the spawn state: {e}"));
        let verdict = tracker
            .judge(&spawn)
            .unwrap_or_else(|e| panic!("{what}: judging the spawn state: {e}"));
        assert!(
            !verdict,
            "{what} is BORN SOLVED — `{}` already holds at the start pose, so a \
             do-nothing policy scores 100%",
            spec.success.as_ref().unwrap().describe()
        );
    }
}

/// Physical sanity a JSON schema cannot check: the zoo's robot is
/// `gripper_arm`, whose jaw reaches a ring 0.323–0.401 m from its shoulder at
/// (x, z) = (0, 0.5) — see the reach arithmetic in the URDF's header. A prop
/// whose top face is off that ring cannot be grasped at all, and a task built
/// on one is decoration, not a task.
#[test]
fn every_zoo_prop_has_a_graspable_top_face() {
    // The jaw's own annulus about the shoulder, in the arm's x-z plane.
    const SHOULDER_Z: f64 = 0.5;
    const R_MIN: f64 = 0.323; // fully folded elbow (|j2| = 1.5 rad)
    const R_MAX: f64 = 0.401; // straight arm

    for (path, spec) in zoo() {
        let what = label(&path);
        for prop in &spec.scene.props {
            let half = prop
                .half_extents
                .unwrap_or_else(|| panic!("{what}: zoo props are boxes"));
            let (x, top) = (prop.pos[0], prop.pos[2] + half[2]);
            let r = (x * x + (SHOULDER_Z - top).powi(2)).sqrt();
            assert!(
                (R_MIN..=R_MAX).contains(&r),
                "{what}: prop `{}` has its top face at (x={x:.3}, z={top:.3}), \
                 {r:.3} m from the shoulder — outside the jaw's {R_MIN}..{R_MAX} m \
                 reach, so nothing can ever pick it up",
                prop.name
            );
            assert!(
                prop.pos[1].abs() < 1e-9,
                "{what}: prop `{}` sits at y={}, off the arm's x-z plane (both \
                 hinges turn about Y)",
                prop.name,
                prop.pos[1]
            );
        }
    }
}
