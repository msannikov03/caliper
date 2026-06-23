use super::*;
use caliper_kinematics::{fk_frame, fk_tip};
use caliper_model::Model;
use caliper_spatial::Se3;
use nalgebra::Vector3;
use std::path::Path;

fn load(name: &str) -> Model {
    let p = format!(
        "{}/../../oracle/fixtures/robots/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    Model::from_urdf(Path::new(&p)).unwrap()
}
fn lim(m: &Model) -> MotionLimits {
    MotionLimits::from_model(m, &MotionLimitsConfig::default()).unwrap()
}
struct R(u64);
impl R {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

// ---- ANALYTICAL single-DOF 7-segment reference ----
#[test]
fn scurve_matches_analytic_7segment() {
    // L=10, vmax=1.5, amax=2, jmax=4 => full 7-segment (cruise + accel plateau)
    let p = crate::scurve::plan_scurve(10.0, 1.5, 2.0, 4.0);
    let tj = 2.0 / 4.0; // amax/jmax
    let ta = 1.5 / 2.0 - tj; // vmax/amax - tj = 0.25
    assert!((p.tj - tj).abs() < 1e-9, "tj {}", p.tj);
    assert!((p.ta - ta).abs() < 1e-9, "ta {}", p.ta);
    assert!(p.tv > 0.0, "cruise must exist");
    // total-time identity T = L/v + v/a + a/j
    let tref = 10.0 / 1.5 + 1.5 / 2.0 + 2.0 / 4.0;
    assert!(
        (p.total() - tref).abs() < 1e-9,
        "total {} vs {}",
        p.total(),
        tref
    );
    // displacement integrates to L
    let (dp, v, a) = p.sample(p.total());
    assert!((dp - 10.0).abs() < 1e-9 && v.abs() < 1e-9 && a.abs() < 1e-9);
    // peak vel == vmax
    let mut vmax = 0.0f64;
    let n = 2000;
    let dt = p.total() / n as f64;
    for k in 0..=n {
        vmax = vmax.max(p.sample(k as f64 * dt).1.abs());
    }
    assert!((vmax - 1.5).abs() < 1e-6);
}

#[test]
fn scurve_short_move_no_cruise() {
    let p = crate::scurve::plan_scurve(0.1, 1.5, 2.0, 4.0);
    assert!(p.tv.abs() < 1e-12, "no cruise");
    let mut vmax = 0.0f64;
    let n = 2000;
    let dt = p.total() / n as f64;
    for k in 0..=n {
        vmax = vmax.max(p.sample(k as f64 * dt).1.abs());
    }
    assert!(vmax < 1.5 + 1e-9);
    let (dp, _, _) = p.sample(p.total());
    assert!((dp - 0.1).abs() < 1e-9);
}

// ---- FEASIBILITY (limits respected over the whole horizon) ----
#[test]
fn move_j_respects_limits() {
    for name in [
        "toy.urdf",
        "showcase6.urdf",
        "redundant7.urdf",
        "prismatic.urdf",
    ] {
        let m = load(name);
        let l = lim(&m);
        let mut rng = R(0xBEEF);
        for _ in 0..5 {
            let q0: Vec<f64> = (0..m.ndof)
                .map(|i| {
                    let (lo, hi) = m.limits[i].unwrap_or((-1.0, 1.0));
                    lo + rng.f() * (hi - lo)
                })
                .collect();
            let q1: Vec<f64> = (0..m.ndof)
                .map(|i| {
                    let (lo, hi) = m.limits[i].unwrap_or((-1.0, 1.0));
                    lo + rng.f() * (hi - lo)
                })
                .collect();
            let traj = move_j(&m, &q0, &q1, &l).unwrap();
            let t = traj.duration();
            if t < 1e-9 {
                continue;
            }
            let n = 4000usize;
            let dt = t / n as f64;
            for k in 1..n {
                let s = traj.sample(k as f64 * dt);
                for i in 0..m.ndof {
                    assert!(s.qd[i].abs() <= l.vmax[i] * (1.0 + 1e-6), "{name} vel j{i}");
                    assert!(
                        s.qdd[i].abs() <= l.amax[i] * (1.0 + 1e-6),
                        "{name} acc j{i}"
                    );
                    let jk = (traj.sample(k as f64 * dt + dt).qdd[i]
                        - traj.sample(k as f64 * dt - dt).qdd[i])
                        / (2.0 * dt);
                    assert!(
                        jk.abs() <= l.jmax[i] * (1.0 + 1e-2),
                        "{name} jerk j{i}={jk}"
                    );
                }
            }
        }
    }
}

// ---- BOUNDARY / C2 ----
#[test]
fn move_j_boundary_and_c2() {
    for name in ["toy.urdf", "showcase6.urdf", "redundant7.urdf"] {
        let m = load(name);
        let l = lim(&m);
        let mut rng = R(0x1234);
        for _ in 0..5 {
            let q0: Vec<f64> = (0..m.ndof)
                .map(|i| {
                    let (lo, hi) = m.limits[i].unwrap_or((-1.0, 1.0));
                    lo + rng.f() * (hi - lo)
                })
                .collect();
            let q1: Vec<f64> = (0..m.ndof)
                .map(|i| {
                    let (lo, hi) = m.limits[i].unwrap_or((-1.0, 1.0));
                    lo + rng.f() * (hi - lo)
                })
                .collect();
            let traj = move_j(&m, &q0, &q1, &l).unwrap();
            let a = traj.sample(0.0);
            let b = traj.sample(traj.duration());
            for i in 0..m.ndof {
                assert!((a.q[i] - q0[i]).abs() < 1e-12 && (b.q[i] - q1[i]).abs() < 1e-12);
                assert!(a.qd[i].abs() < 1e-12 && b.qd[i].abs() < 1e-12);
                assert!(a.qdd[i].abs() < 1e-12 && b.qdd[i].abs() < 1e-12);
            }
            let t = traj.duration();
            if t < 1e-9 {
                continue;
            }
            let n = 4000;
            let dt = t / n as f64;
            let mut prev = traj.sample(0.0);
            for k in 1..=n {
                let s = traj.sample(k as f64 * dt);
                for i in 0..m.ndof {
                    assert!((s.qd[i] - prev.qd[i]).abs() <= l.amax[i] * dt * (1.0 + 1e-3));
                    assert!((s.qdd[i] - prev.qdd[i]).abs() <= l.jmax[i] * dt * (1.0 + 1e-2));
                }
                prev = s;
            }
        }
    }
}

// ---- MOVE_L STRAIGHTNESS ----
#[test]
fn move_l_traces_straight_line() {
    let m = load("showcase6.urdf");
    let l = lim(&m);
    let qa = [0.3, -0.4, 0.6, 0.2, -0.5, 0.1];
    let ta = fk_tip(&m, &qa);
    let pa = ta.translation_vec();
    // a comfortably-reachable straight move (0.05 m, orientation held) — straightness
    // is the property under test; reachability of larger/rotating moves varies.
    let pb = pa + Vector3::new(0.0, 0.05, 0.0);
    let goal = Se3::from_parts(pb, ta.0.rotation);
    let opts = CartesianMoveOpts::defaults(l);
    let f = m.tip_frame();
    let traj = move_l(&m, f, &qa, &goal, &opts).unwrap();
    assert!(traj.completed, "reachable MOVE_L must complete");
    let u = (pb - pa).normalize();
    let n = 200;
    for k in 0..=n {
        let t = traj.duration() * k as f64 / n as f64;
        let q = traj.sample(t).q;
        let p = fk_frame(&m, &q, f).translation_vec();
        let d = ((p - pa) - ((p - pa).dot(&u)) * u).norm();
        assert!(d < 1e-3, "lateral dev {d} at t={t}");
    }
}

// On an unreachable target, Abort hard-errors; the default (Truncate) returns a
// best-effort prefix flagged completed=false with the reached path fraction.
#[test]
fn move_l_abort_on_unreachable() {
    let m = load("showcase6.urdf");
    let l = lim(&m);
    let qa = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
    let ta = fk_tip(&m, &qa);
    let goal = Se3::from_parts(
        ta.translation_vec() + Vector3::new(5.0, 5.0, 5.0),
        ta.0.rotation,
    );
    let opts = CartesianMoveOpts {
        on_failure: OnFailure::Abort,
        ..CartesianMoveOpts::defaults(l)
    };
    let r = move_l(&m, m.tip_frame(), &qa, &goal, &opts);
    assert!(
        matches!(r, Err(MotionError::Unreachable { .. })),
        "expected Unreachable, got {r:?}"
    );
}

#[test]
fn move_l_truncate_returns_prefix() {
    let m = load("showcase6.urdf");
    let l = lim(&m);
    let qa = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
    let ta = fk_tip(&m, &qa);
    let goal = Se3::from_parts(
        ta.translation_vec() + Vector3::new(5.0, 5.0, 5.0),
        ta.0.rotation,
    );
    let opts = CartesianMoveOpts::defaults(l); // default = Truncate
    let traj = move_l(&m, m.tip_frame(), &qa, &goal, &opts).expect("truncate yields a prefix");
    assert!(!traj.completed, "should be a truncated prefix");
    assert!(
        traj.reached < 1.0 && traj.reached >= 0.0,
        "reached {}",
        traj.reached
    );
    assert!(traj.duration() > 0.0, "prefix must have positive duration");
}
