//! `caliper report` — cycle-time + path-quality report assembly.
//!
//! Pure glue: sample planned [`Trajectory`] segments onto rows, run the
//! engine's [`path_report`], and render the result as a tidy table or JSON.
use caliper::kinematics::{
    Finding, LintLimits, LintOptions, LintSeverity, PathReport, PathRows, lint_path, path_report,
};
use caliper::model::Model;
use caliper::motion::{MotionLimits, Trajectory};
use caliper_collision::{CollisionModel, WorldScene};
use std::sync::Arc;

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

/// Contiguous `true` windows of `flags` as inclusive `(first, last)` index pairs.
fn windows(flags: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut k = 0;
    while k < flags.len() {
        if flags[k] {
            let start = k;
            while k < flags.len() && flags[k] {
                k += 1;
            }
            out.push((start, k - 1));
        } else {
            k += 1;
        }
    }
    out
}

/// Collision lint over sampled rows — the collision-margin half of the
/// trajectory linter. Lives CLI-side because caliper-kinematics cannot depend
/// on caliper-collision (the engine linter `lint_path` covers T001–T007).
///
/// - `T008` (Error): the path is in collision (self or world), one finding per
///   contiguous time window.
/// - `T009` (Warning): the path passes within `clearance` of an obstacle or
///   itself. caliper-collision has no distance query, so this is a boolean
///   re-query with every collider inflated by `clearance` — conservative:
///   collider-pair gaps (self and world boxes) are flagged anywhere below
///   2×`clearance` (both sides inflate), ground gaps below 1×.
pub fn lint_collision(
    model: &Arc<Model>,
    scene: &WorldScene,
    rows: &SampledRows,
    clearance: f64,
) -> Vec<Finding> {
    let mut out = Vec::new();
    let ns = rows.times.len();
    if ns == 0 {
        return out;
    }
    let clearance = if clearance.is_finite() {
        clearance.max(0.0)
    } else {
        0.0
    };
    let cm0 = CollisionModel::new(model.clone(), scene.clone(), 0.0);
    let cmc =
        (clearance > 0.0).then(|| CollisionModel::new(model.clone(), scene.clone(), clearance));
    let mut hit0 = vec![false; ns];
    let mut hitc = vec![false; ns];
    for k in 0..ns {
        match cm0.query(&rows.q[k]) {
            Ok(rep) => hit0[k] = rep.has_collision(),
            // rows sampled from a foreign robot (dim mismatch / non-finite q)
            // are themselves a hard finding, not a panic
            Err(e) => {
                out.push(Finding {
                    code: "T008",
                    severity: LintSeverity::Error,
                    message: format!(
                        "collision query failed at t={:.3} s: {e}; rows must come from this robot's own trajectory",
                        rows.times[k]
                    ),
                    fix_hint: "sample the rows from a trajectory planned on the same model (q length = ndof, finite)".into(),
                    joint: None,
                    time: Some(rows.times[k]),
                    value: None,
                });
                return out;
            }
        }
        if let Some(cm) = &cmc {
            // same q already validated by the margin-0 query above
            hitc[k] = cm.query(&rows.q[k]).is_ok_and(|r| r.has_collision());
        }
    }
    for (a, b) in windows(&hit0) {
        out.push(Finding {
            code: "T008",
            severity: LintSeverity::Error,
            message: format!(
                "path in collision from t={:.3} s to t={:.3} s ({} of {ns} samples)",
                rows.times[a],
                rows.times[b],
                b - a + 1,
            ),
            fix_hint: "re-plan the segment (Planner::plan / plan_optimal) around the obstacle or move the waypoints".into(),
            joint: None,
            time: Some(rows.times[a]),
            value: None,
        });
    }
    // near-miss windows: inside the clearance envelope but not actually colliding
    let near: Vec<bool> = hitc.iter().zip(&hit0).map(|(c, h)| *c && !*h).collect();
    for (a, b) in windows(&near) {
        out.push(Finding {
            code: "T009",
            severity: LintSeverity::Warning,
            message: format!(
                "path passes within the clearance margin {clearance:.3} m of an obstacle or itself from t={:.3} s to t={:.3} s",
                rows.times[a],
                rows.times[b],
            ),
            fix_hint: "re-plan with at least this planner margin, or move the waypoints away from the obstacle".into(),
            joint: None,
            time: Some(rows.times[a]),
            value: Some(clearance),
        });
    }
    out
}

/// The whole trajectory lint over sampled rows: the engine linter
/// ([`lint_path`], T001–T007, engineering-default thresholds) plus the
/// CLI-side collision lint ([`lint_collision`], T008/T009), merged
/// most-severe-first (then by code, then chronologically).
pub fn lint_rows(
    model: &Arc<Model>,
    frame: usize,
    rows: &SampledRows,
    limits: &MotionLimits,
    scene: &WorldScene,
    clearance: f64,
) -> Vec<Finding> {
    let mut out = lint_path(
        model,
        frame,
        &rows.rows(),
        &LintLimits {
            vmax: &limits.vmax,
            amax: &limits.amax,
            jmax: &limits.jmax,
        },
        &LintOptions::default(),
    );
    out.extend(lint_collision(model, scene, rows, clearance));
    out.sort_by(|a, b| {
        b.severity
            .cmp(&a.severity) // LintSeverity orders Warning < Error
            .then_with(|| a.code.cmp(b.code))
            .then_with(|| {
                a.time
                    .partial_cmp(&b.time)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    out
}

/// Print the LINT section of the human report: findings errors-first (the
/// [`lint_rows`] order) with their codes and fix hints.
pub fn print_lint(findings: &[Finding]) {
    println!();
    if findings.is_empty() {
        println!("  LINT: clean — no findings");
        return;
    }
    let errors = findings
        .iter()
        .filter(|f| f.severity == LintSeverity::Error)
        .count();
    println!(
        "  LINT: {} error(s), {} warning(s)",
        errors,
        findings.len() - errors
    );
    for f in findings {
        let tag = match f.severity {
            LintSeverity::Error => "ERROR",
            LintSeverity::Warning => "WARN ",
        };
        println!("    [{}] {tag} {}", f.code, f.message);
        println!("           fix: {}", f.fix_hint);
    }
}

/// Lint findings as a JSON array (snake_case keys; `joint`/`time`/`value`
/// are `null` when the finding has no such anchor).
pub fn lint_to_json(findings: &[Finding]) -> serde_json::Value {
    serde_json::Value::Array(
        findings
            .iter()
            .map(|f| {
                serde_json::json!({
                    "code": f.code,
                    "severity": match f.severity {
                        LintSeverity::Error => "error",
                        LintSeverity::Warning => "warning",
                    },
                    "message": f.message,
                    "fix_hint": f.fix_hint,
                    "joint": f.joint,
                    "time": f.time,
                    "value": f.value,
                })
            })
            .collect(),
    )
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

    // ===== lint_collision =====

    /// `SampledRows` at the given configs with a unit clock and zero rates
    /// (the collision lint only reads `times` + `q`).
    fn rows_at(q: Vec<Vec<f64>>) -> SampledRows {
        let n = q.first().map(|r| r.len()).unwrap_or(0);
        let z = vec![vec![0.0; n]; q.len()];
        SampledRows {
            times: (0..q.len()).map(|k| k as f64).collect(),
            q,
            qd: z.clone(),
            qdd: z,
        }
    }

    /// T008 positive: collide_arm folded to q = [0, π, π] self-collides
    /// (l1 ↔ l3, per the fixture's own doc), the straight q = 0 samples are
    /// clear ⇒ exactly one Error window at the middle sample.
    #[test]
    fn lint_collision_flags_folded_self_collision() {
        let m = Arc::new(load("collide_arm.urdf"));
        let scene = WorldScene::new();
        let folded = vec![0.0, std::f64::consts::PI, std::f64::consts::PI];
        // fixture precondition, straight from the collision checker
        let cm = CollisionModel::new(m.clone(), scene.clone(), 0.0);
        assert!(cm.query(&folded).unwrap().has_collision());
        assert!(!cm.query(&[0.0; 3]).unwrap().has_collision());

        let rows = rows_at(vec![vec![0.0; 3], folded, vec![0.0; 3]]);
        let f = lint_collision(&m, &scene, &rows, 0.0);
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].code, "T008");
        assert_eq!(f[0].severity, LintSeverity::Error);
        assert_eq!(f[0].time, Some(1.0));
        assert!(f[0].message.contains("1 of 3 samples"), "{}", f[0].message);
    }

    /// T009 clearance probe against an analytic gap. collide_arm at q = 0:
    /// link boxes span x ∈ [−0.06, 0.06]; a world box with its near face at
    /// x = 0.15 leaves a 0.09 m gap. Both sides inflate by the clearance, so
    /// the probe trips iff 2·clearance ≥ 0.09: 0.05 warns, 0.04 stays clean.
    #[test]
    fn lint_collision_clearance_probe_known_gap() {
        let m = Arc::new(load("collide_arm.urdf"));
        let scene = WorldScene::new().add_box([0.2, 0.0, 0.45], [0.05, 0.05, 0.05]);
        let rows = rows_at(vec![vec![0.0; 3], vec![0.0; 3]]);

        let f = lint_collision(&m, &scene, &rows, 0.05); // 2·0.05 = 0.10 > 0.09
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].code, "T009");
        assert_eq!(f[0].severity, LintSeverity::Warning);
        assert_eq!(f[0].time, Some(0.0));
        assert_eq!(f[0].value, Some(0.05));

        let clean = lint_collision(&m, &scene, &rows, 0.04); // 2·0.04 = 0.08 < 0.09
        assert!(clean.is_empty(), "{clean:?}");
    }

    /// Negative + degenerate: a clear path lints clean (with and without a
    /// clearance probe), and empty rows are a no-op.
    #[test]
    fn lint_collision_clean_and_empty() {
        let m = Arc::new(load("collide_arm.urdf"));
        let scene = WorldScene::new().add_box([1.0, 0.0, 0.45], [0.05, 0.05, 0.05]);
        let rows = rows_at(vec![vec![0.0; 3], vec![0.1, 0.1, 0.1]]);
        assert!(lint_collision(&m, &scene, &rows, 0.0).is_empty());
        assert!(lint_collision(&m, &scene, &rows, 0.02).is_empty());
        let empty = rows_at(vec![]);
        assert!(lint_collision(&m, &scene, &empty, 0.05).is_empty());
    }

    // ===== lint_rows / lint_to_json (the `caliper report` LINT section) =====

    /// Positive: a path that folds into self-collision (T008 Error, exactly
    /// as `lint_collision_flags_folded_self_collision` pins) AND loops joints
    /// 2π back to the start (T005 Warning, travel ≫ net) merges into one list
    /// with every Error strictly before every Warning.
    #[test]
    fn lint_rows_merges_and_sorts_errors_first() {
        let m = Arc::new(load("collide_arm.urdf"));
        let limits = MotionLimits::from_model(&m, &MotionLimitsConfig::default()).unwrap();
        let scene = WorldScene::new();
        let folded = vec![0.0, std::f64::consts::PI, std::f64::consts::PI];
        let rows = rows_at(vec![vec![0.0; 3], folded, vec![0.0; 3]]);
        let f = lint_rows(&m, m.tip_frame(), &rows, &limits, &scene, 0.0);
        let codes: Vec<&str> = f.iter().map(|x| x.code).collect();
        assert!(codes.contains(&"T008"), "{codes:?}");
        assert!(codes.contains(&"T005"), "{codes:?}");
        // exactly one collision window, at the folded middle sample
        let t008: Vec<_> = f.iter().filter(|x| x.code == "T008").collect();
        assert_eq!(t008.len(), 1);
        assert_eq!(t008[0].time, Some(1.0));
        // sorted most-severe-first: no Error after the first Warning
        let first_warn = f
            .iter()
            .position(|x| x.severity == LintSeverity::Warning)
            .unwrap();
        assert!(
            f[first_warn..]
                .iter()
                .all(|x| x.severity == LintSeverity::Warning),
            "{codes:?}"
        );
    }

    /// Negative: a planned MOVE_J inside its own limits, with no scene, lints
    /// completely clean end-to-end through the same glue.
    #[test]
    fn lint_rows_clean_on_planned_move() {
        let m = Arc::new(load("toy.urdf"));
        let limits = MotionLimits::from_model(&m, &MotionLimitsConfig::default()).unwrap();
        let traj = move_j(&m, &[0.0, 0.0], &[0.5, -0.5], &limits).unwrap();
        let rows = sample_segments(std::slice::from_ref(&traj), 40);
        let f = lint_rows(&m, m.tip_frame(), &rows, &limits, &WorldScene::new(), 0.0);
        assert!(f.is_empty(), "{f:?}");
    }

    /// The JSON projection carries every machine field and maps absent
    /// anchors to null.
    #[test]
    fn lint_json_fields_and_nulls() {
        let findings = vec![Finding {
            code: "T001",
            severity: LintSeverity::Error,
            message: "m".into(),
            fix_hint: "h".into(),
            joint: Some(2),
            time: Some(0.5),
            value: None,
        }];
        let v = lint_to_json(&findings);
        assert_eq!(v[0]["code"], "T001");
        assert_eq!(v[0]["severity"], "error");
        assert_eq!(v[0]["joint"], 2);
        assert_eq!(v[0]["time"], 0.5);
        assert!(v[0]["value"].is_null());
        assert_eq!(lint_to_json(&[]), serde_json::Value::Array(vec![]));
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
