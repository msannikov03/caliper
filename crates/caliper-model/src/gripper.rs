//! Gripper-channel detection: WHICH movable joint (if any) is the robot's
//! gripper, and what "open" / "closed" mean on it.
//!
//! Pure model inspection — no simulation, no MuJoCo — so every face (Studio,
//! Python, CLI) resolves the SAME joint from the SAME rules instead of each
//! guessing its own.
//!
//! # The rules, and why
//!
//! A gripper joint is recognized by NAME, matched case-insensitively against
//! [`GRIPPER_NAME_HINTS`] over BOTH the joint's own name and its child link's
//! name. The child link matters because real descriptions often don't name the
//! joint at all: SO-101 numbers its joints `1`..`6` and puts the semantics in
//! the links (`... → wrist → gripper → jaw`), so the jaw joint `6` is only
//! findable through its child link. Panda names the joint itself
//! (`panda_finger_joint1`); both spellings are covered.
//!
//! Candidates are then filtered and ranked:
//! - **Both limits required.** Open/closed targets ARE the joint limits (inset
//!   by [`GRIPPER_TARGET_INSET`]); a limitless joint has no closed end, so it
//!   is not a gripper channel — a name match without limits yields `None`
//!   rather than a channel whose targets would have to be invented.
//! - **Mimic joints are skipped.** A parallel gripper's second finger is a
//!   `<mimic>` of the first (Panda's `panda_finger_joint2`, Gen3 lite's tips):
//!   it is DRIVEN, not commanded, so commanding it would fight its source.
//! - **Most distal wins.** Model joints are in topological order, so the LAST
//!   match is the deepest in the chain — on SO-101 that separates the wrist
//!   joint (whose child link is named `gripper`) from the jaw joint after it.

use crate::Model;

/// Substrings that name a gripper joint or its child link, lowercased. Matched
/// as substrings, so `panda_finger_joint1` and `left_jaw` both hit.
pub const GRIPPER_NAME_HINTS: [&str; 5] = ["gripper", "finger", "jaw", "claw", "hand"];

/// Fraction of the joint RANGE the open/closed targets are pulled in from the
/// limits. A PD target sitting exactly on a limit slams into MuJoCo's limit
/// constraint every tick; 2% keeps the hold target inside the mechanism.
pub const GRIPPER_TARGET_INSET: f64 = 0.02;

/// True if `s` contains any [`GRIPPER_NAME_HINTS`] entry, case-insensitively.
pub fn is_gripper_name(s: &str) -> bool {
    let lower = s.to_lowercase();
    GRIPPER_NAME_HINTS.iter().any(|h| lower.contains(h))
}

/// The joint's own output frame: the first frame anchored to `joint` with an
/// identity offset, i.e. the URDF CHILD LINK of that joint (the compiler
/// registers it at the moment the joint is created, before any fixed-folded
/// descendant). `None` for an out-of-range index.
pub fn child_link_frame(m: &Model, joint: usize) -> Option<usize> {
    if joint >= m.ndof {
        return None;
    }
    m.frames.iter().position(|f| f.anchor == Some(joint))
}

/// Name of the joint's child link — the link a gripper welds a grasped prop
/// to. `None` for an out-of-range index.
pub fn child_link_name(m: &Model, joint: usize) -> Option<&str> {
    child_link_frame(m, joint).map(|f| m.frames[f].name.as_str())
}

/// The robot's gripper joint, or `None` when nothing qualifies. See the module
/// docs for the exact rules (name hints over joint + child link, both limits
/// required, mimics skipped, most distal wins).
pub fn find_gripper_joint(m: &Model) -> Option<usize> {
    // Most distal wins, so search from the deepest joint backwards.
    (0..m.ndof).rfind(|&i| {
        m.mimic[i].is_none()
            && gripper_targets(m, i, true).is_some()
            && (is_gripper_name(&m.joint_names[i])
                || child_link_name(m, i).is_some_and(is_gripper_name))
    })
}

/// `(open target, closed target)` for `joint`: its limits pulled in by
/// [`GRIPPER_TARGET_INSET`] of the range. `closed_at_lo` picks which end of
/// the range CLOSES the jaw — true for the common convention (0 = shut,
/// positive = open), false for a joint that closes toward its upper limit.
///
/// `None` when the joint has no limits, they are not finite, or the range is
/// empty — none of which can host an open/closed channel.
pub fn gripper_targets(m: &Model, joint: usize, closed_at_lo: bool) -> Option<(f64, f64)> {
    let (lo, hi) = (*m.limits.get(joint)?)?;
    if !(lo.is_finite() && hi.is_finite() && hi > lo) {
        return None;
    }
    let inset = GRIPPER_TARGET_INSET * (hi - lo);
    let (lo_t, hi_t) = (lo + inset, hi - inset);
    Some(if closed_at_lo {
        (hi_t, lo_t)
    } else {
        (lo_t, hi_t)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn fixture(dir: &str, name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../oracle/fixtures")
            .join(dir)
            .join(name)
    }

    fn robot(name: &str) -> Model {
        Model::from_urdf(&fixture("robots", name)).expect("fixture loads")
    }

    fn corpus(name: &str) -> Model {
        Model::from_urdf(&fixture("corpus", name)).expect("corpus fixture loads")
    }

    #[test]
    fn name_hints_are_case_insensitive_substrings() {
        assert!(is_gripper_name("gripper"));
        assert!(is_gripper_name("panda_finger_joint1"));
        assert!(is_gripper_name("LEFT_JAW"));
        assert!(is_gripper_name("Claw"));
        assert!(is_gripper_name("robotiq_hand_e"));
        assert!(!is_gripper_name("j2"));
        assert!(!is_gripper_name("wrist_roll"));
        assert!(!is_gripper_name(""));
    }

    #[test]
    fn a_plain_arm_has_no_gripper() {
        let m = robot("dyn_pendulum2.urdf");
        assert_eq!(find_gripper_joint(&m), None);
        let m = robot("showcase6.urdf");
        assert_eq!(find_gripper_joint(&m), None);
    }

    #[test]
    fn so101_detects_through_its_child_link() {
        // SO-101 numbers its joints `1`..`6`; only the LINK names say what they
        // are (`wrist → gripper → jaw`). Joint 5's child link is `gripper` and
        // joint 6's is `jaw` — both match, and the distal one (6) must win.
        let m = corpus("so101_new_calib.urdf");
        let g = find_gripper_joint(&m).expect("so101 must expose a gripper channel");
        assert_eq!(m.joint_names[g], "6");
        assert_eq!(child_link_name(&m, g), Some("jaw"));
        let (open, closed) = gripper_targets(&m, g, true).unwrap();
        let (lo, hi) = m.limits[g].unwrap();
        assert!(closed > lo && closed < open && open < hi);
    }

    #[test]
    fn panda_picks_the_driving_finger_not_the_mimic() {
        // panda_finger_joint2 is `<mimic>`-driven by joint1: commanding it
        // would fight its source, so the DRIVING joint is the channel even
        // though the mimic is later in kinematic order.
        let m = corpus("panda.urdf");
        let g = find_gripper_joint(&m).expect("panda must expose a gripper channel");
        assert_eq!(m.joint_names[g], "panda_finger_joint1");
        assert!(m.mimic[g].is_none());
        assert_eq!(child_link_name(&m, g), Some("panda_leftfinger"));
    }

    #[test]
    fn child_link_is_the_joints_own_output_frame() {
        let m = robot("dyn_pendulum2.urdf");
        assert_eq!(child_link_name(&m, 0), Some("l1"));
        assert_eq!(child_link_name(&m, 1), Some("l2"));
        assert_eq!(child_link_name(&m, 2), None);
        assert_eq!(child_link_frame(&m, 9), None);
    }

    #[test]
    fn targets_inset_from_the_limits_and_respect_the_closed_end() {
        let m = robot("gripper_arm.urdf");
        let g = find_gripper_joint(&m).expect("the fixture has a `gripper` joint");
        assert_eq!(m.joint_names[g], "gripper");
        let (lo, hi) = m.limits[g].unwrap();
        let inset = GRIPPER_TARGET_INSET * (hi - lo);
        assert_eq!(gripper_targets(&m, g, true), Some((hi - inset, lo + inset)));
        assert_eq!(
            gripper_targets(&m, g, false),
            Some((lo + inset, hi - inset))
        );
    }

    #[test]
    fn a_limitless_name_match_is_not_a_channel() {
        // A continuous joint has no limits, so there is no closed end to aim
        // at — the detector must decline rather than invent one.
        let urdf = r#"<?xml version="1.0"?>
          <robot name="freehand">
            <link name="base"/><link name="hand"/>
            <joint name="wrist_to_hand" type="continuous">
              <parent link="base"/><child link="hand"/>
              <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
            </joint>
          </robot>"#;
        let path =
            std::env::temp_dir().join(format!("caliper_freehand_{}.urdf", std::process::id()));
        std::fs::write(&path, urdf).unwrap();
        let m = Model::from_urdf(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(is_gripper_name(&m.joint_names[0]));
        assert_eq!(gripper_targets(&m, 0, true), None);
        assert_eq!(find_gripper_joint(&m), None);
    }
}
