//! `caliper report` — cycle-time + path-quality report assembly.
//!
//! Pure glue: sample planned [`Trajectory`] segments onto rows, run the
//! engine's [`path_report`], and render the result as a tidy table or JSON.
use caliper::kinematics::{PathReport, PathRows, path_report};
use caliper::model::Model;
use caliper::motion::{MotionLimits, Trajectory};

/// Owned per-sample rows harvested from back-to-back trajectory segments,
/// concatenated onto one monotone clock (segment k+1 starts where k ended).
pub struct SampledRows {
    pub times: Vec<f64>,
    pub q: Vec<Vec<f64>>,
    pub qd: Vec<Vec<f64>>,
    pub qdd: Vec<Vec<f64>>,
}

impl SampledRows {
    pub fn rows(&self) -> PathRows<'_> {
        PathRows {
            times: &self.times,
            q: &self.q,
            qd: &self.qd,
            qdd: &self.qdd,
        }
    }
}

/// Sample `n` points per segment (uniform in time, endpoints included) and
/// concatenate on a shared clock. `n` is clamped to >= 2 so every segment
/// contributes both endpoints.
pub fn sample_segments(segments: &[Trajectory], n: usize) -> SampledRows {
    let n = n.max(2);
    let mut out = SampledRows {
        times: vec![],
        q: vec![],
        qd: vec![],
        qdd: vec![],
    };
    let mut offset = 0.0;
    for traj in segments {
        for (t, s) in traj.sample_grid(n) {
            out.times.push(offset + t);
            out.q.push(s.q);
            out.qd.push(s.qd);
            out.qdd.push(s.qdd);
        }
        offset += traj.duration();
    }
    out
}

/// Whole-path report over back-to-back segments: sample, then fold through the
/// engine's [`path_report`] against the shared `limits`.
pub fn report_segments(
    model: &Model,
    frame: usize,
    segments: &[Trajectory],
    n: usize,
    limits: &MotionLimits,
) -> PathReport {
    let rows = sample_segments(segments, n);
    path_report(model, frame, &rows.rows(), &limits.vmax, &limits.amax)
}

/// Percent string for a utilization fraction ("62.1%"), infinity-safe.
fn pct(x: f64) -> String {
    if x.is_finite() {
        format!("{:.1}%", x * 100.0)
    } else {
        "inf".into()
    }
}

/// A `(joint, value)` pair as "62.1% (j_name)"; `None` (0-DOF) as "n/a".
fn worst_str(model: &Model, w: Option<(usize, f64)>) -> String {
    match w {
        Some((j, v)) => format!("{} ({})", pct(v), model.joint_names[j]),
        None => "n/a".into(),
    }
}

/// Print the human table: cycle time (+ per-segment), conditioning, per-joint
/// limit margins and velocity/acceleration utilization.
pub fn print_table(model: &Model, rep: &PathReport, seg_durations: &[f64]) {
    println!(
        "  cycle time      : {:.4} s  ({} samples)",
        rep.cycle_time, rep.samples
    );
    if seg_durations.len() > 1 {
        for (k, d) in seg_durations.iter().enumerate() {
            println!("    segment [{k}]   : {d:.4} s");
        }
    }
    println!(
        "  manipulability  : min {:.4e}   mean {:.4e}",
        rep.min_manipulability, rep.mean_manipulability
    );
    println!(
        "  sigma_min       : min {:.4e}  @ t={:.3}s",
        rep.min_sigma_min, rep.t_min_sigma
    );
    println!("  joint            limit-margin   vel-util   acc-util");
    for i in 0..model.ndof {
        let margin = if rep.limit_margin[i].is_finite() {
            format!("{:>12.4}", rep.limit_margin[i])
        } else {
            format!("{:>12}", "unbounded")
        };
        println!(
            "    [{i}] {:<10} {margin}   {:>8}   {:>8}",
            model.joint_names[i],
            pct(rep.vel_utilization[i]),
            pct(rep.acc_utilization[i])
        );
    }
    println!(
        "  worst vel util  : {}",
        worst_str(model, rep.worst_vel_utilization())
    );
    println!(
        "  worst acc util  : {}",
        worst_str(model, rep.worst_acc_utilization())
    );
    match rep.min_limit_margin() {
        Some((j, m)) => println!(
            "  tightest margin : {m:.4} (rad|m) on {}{}",
            model.joint_names[j],
            if m < 0.0 { "  ⚠ LIMIT VIOLATED" } else { "" }
        ),
        None => println!("  tightest margin : n/a (no position limits)"),
    }
}

/// Machine output: the full report as one JSON object (snake_case keys;
/// unbounded margins and infinite utilizations serialize as `null`).
pub fn to_json(model: &Model, rep: &PathReport, seg_durations: &[f64]) -> serde_json::Value {
    let finite = |x: f64| -> serde_json::Value {
        if x.is_finite() {
            serde_json::json!(x)
        } else {
            serde_json::Value::Null
        }
    };
    let vecf = |v: &[f64]| -> Vec<serde_json::Value> { v.iter().map(|&x| finite(x)).collect() };
    serde_json::json!({
        "cycle_time": rep.cycle_time,
        "samples": rep.samples,
        "segment_durations": seg_durations,
        "min_manipulability": rep.min_manipulability,
        "mean_manipulability": rep.mean_manipulability,
        "min_sigma_min": rep.min_sigma_min,
        "t_min_sigma": rep.t_min_sigma,
        "joints": model.joint_names,
        "limit_margin": vecf(&rep.limit_margin),
        "vel_utilization": vecf(&rep.vel_utilization),
        "acc_utilization": vecf(&rep.acc_utilization),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliper::motion::{MotionLimitsConfig, move_j};
    use std::path::Path;

    fn load(name: &str) -> Model {
        let p = format!(
            "{}/../../oracle/fixtures/robots/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        );
        Model::from_urdf(Path::new(&p)).unwrap()
    }

    /// Two deterministic MOVE_J legs on showcase6: g0 → g1 → g2. The start is
    /// deliberately OFF the all-zeros home pose — showcase6's spherical wrist
    /// is exactly singular there (wrist axes align, σ_min = 0), so a path from
    /// home makes min_sigma_min legitimately 0; the strict σ_min > 0 assertion
    /// needs a nowhere-singular path. (That home IS singular is pinned by
    /// `report_flags_singular_home` below.)
    fn fixture() -> (Model, Vec<Trajectory>, MotionLimits) {
        let m = load("showcase6.urdf");
        let limits = MotionLimits::from_model(&m, &MotionLimitsConfig::default()).unwrap();
        let g0 = vec![0.1, -0.3, 0.4, 0.2, -0.5, 0.1];
        let g1 = vec![0.4, -0.6, 0.5, 0.3, -0.4, 0.2];
        let g2 = vec![-0.2, 0.3, -0.5, 0.1, 0.6, -0.3];
        let t1 = move_j(&m, &g0, &g1, &limits).unwrap();
        let t2 = move_j(&m, &g1, &g2, &limits).unwrap();
        (m, vec![t1, t2], limits)
    }

    /// The report DETECTS a singular pose: a leg starting at showcase6's
    /// all-zeros home (wrist singular) reports σ_min ≈ 0 at t ≈ 0 — the
    /// feature working, not a defect.
    #[test]
    fn report_flags_singular_home() {
        let m = load("showcase6.urdf");
        let limits = MotionLimits::from_model(&m, &MotionLimitsConfig::default()).unwrap();
        let t1 = move_j(
            &m,
            &vec![0.0; m.ndof],
            &[0.4, -0.6, 0.5, 0.3, -0.4, 0.2],
            &limits,
        )
        .unwrap();
        let rep = report_segments(&m, m.tip_frame(), std::slice::from_ref(&t1), 60, &limits);
        assert!(
            rep.min_sigma_min < 1e-6,
            "home wrist singularity not flagged: σ_min = {}",
            rep.min_sigma_min
        );
        assert!(
            rep.t_min_sigma < 0.2 * rep.cycle_time,
            "σ_min should be at the singular start"
        );
    }

    #[test]
    fn sample_segments_shares_one_monotone_clock() {
        let (_, segs, _) = fixture();
        let rows = sample_segments(&segs, 25);
        assert_eq!(rows.times.len(), 50); // 25 per segment
        assert!(rows.times.windows(2).all(|w| w[1] >= w[0]));
        let total = segs.iter().map(|t| t.duration()).sum::<f64>();
        assert!((rows.times.last().unwrap() - total).abs() < 1e-9);
        // the seam is continuous: segment 2's first q == segment 1's last q
        for (a, b) in rows.q[24].iter().zip(&rows.q[25]) {
            assert!((a - b).abs() < 1e-9, "seam continuity");
        }
    }

    #[test]
    fn report_covers_full_cycle_and_respects_limits() {
        let (m, segs, limits) = fixture();
        let rep = report_segments(&m, m.tip_frame(), &segs, 60, &limits);
        let total = segs.iter().map(|t| t.duration()).sum::<f64>();
        assert!((rep.cycle_time - total).abs() < 1e-9);
        assert_eq!(rep.samples, 120);
        // a planned S-curve stays inside its own limits (small sampling slack)
        for i in 0..m.ndof {
            assert!(rep.vel_utilization[i] <= 1.001, "vel util joint {i}");
            assert!(rep.acc_utilization[i] <= 1.001, "acc util joint {i}");
        }
        // showcase6 has position limits on every joint → margins all finite,
        // and the path stays inside them
        assert!(rep.limit_margin.iter().all(|mg| mg.is_finite()));
        assert!(rep.min_limit_margin().unwrap().1 > 0.0);
        // conditioning fields are populated and coherent
        assert!(rep.min_sigma_min > 0.0);
        assert!(rep.min_manipulability <= rep.mean_manipulability);
        // determinism: a second identical run reproduces the report exactly
        let rep2 = report_segments(&m, m.tip_frame(), &segs, 60, &limits);
        assert_eq!(rep.min_sigma_min, rep2.min_sigma_min);
        assert_eq!(rep.vel_utilization, rep2.vel_utilization);
    }

    #[test]
    fn json_maps_nonfinite_to_null() {
        let m = load("toy.urdf");
        let limits = MotionLimits::from_model(&m, &MotionLimitsConfig::default()).unwrap();
        let traj = move_j(&m, &[0.0, 0.0], &[0.5, -0.5], &limits).unwrap();
        let mut rep = report_segments(&m, m.tip_frame(), std::slice::from_ref(&traj), 20, &limits);
        rep.limit_margin[0] = f64::INFINITY; // pretend joint 0 is unbounded
        let j = to_json(&m, &rep, &[traj.duration()]);
        assert!(j["limit_margin"][0].is_null());
        assert!(j["limit_margin"][1].is_number());
        assert_eq!(j["samples"], 20);
    }
}
