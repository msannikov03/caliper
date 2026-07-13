//! Forward kinematics, Jacobians, and singularity analysis.
use caliper_model::{JointKind, Model};
use caliper_spatial::{Se3, exp_prismatic, exp_revolute};
use nalgebra::{DMatrix, DVector, Matrix3, SymmetricEigen, Vector3, Vector6};

/// Local exp of movable joint `i` at configuration `q`.
#[inline]
fn joint_local(model: &Model, i: usize, q: &[f64]) -> Se3 {
    match model.kind[i] {
        JointKind::Revolute => exp_revolute(&model.axis[i], q[i]),
        JointKind::Prismatic => exp_prismatic(&model.axis[i], q[i]),
    }
}

/// Forward kinematics: world pose of every movable joint frame, in topological
/// order. `q.len() == out.len() == model.ndof`. No allocation.
pub fn fk_joints(model: &Model, q: &[f64], out: &mut [Se3]) {
    debug_assert_eq!(q.len(), model.ndof);
    debug_assert_eq!(out.len(), model.ndof);
    for i in 0..model.ndof {
        let parent_world = match model.parent[i] {
            Some(p) => out[p],
            None => Se3::identity(),
        };
        out[i] = parent_world
            .compose(&model.parent_to_joint[i])
            .compose(&joint_local(model, i, q));
    }
}

/// World pose of a link frame, given precomputed movable-joint world poses.
pub fn frame_pose(model: &Model, joint_world: &[Se3], frame: usize) -> Se3 {
    let lf = &model.frames[frame];
    let base = match lf.anchor {
        Some(j) => joint_world[j],
        None => Se3::identity(),
    };
    base.compose(&lf.offset)
}

/// World pose of a single frame, computed independently by walking the
/// root→frame ancestor chain (the cross-check path against [`fk_joints`]).
pub fn fk_frame(model: &Model, q: &[f64], frame: usize) -> Se3 {
    let lf = &model.frames[frame];
    let base = match lf.anchor {
        Some(j) => fk_movable(model, q, j),
        None => Se3::identity(),
    };
    base.compose(&lf.offset)
}

fn fk_movable(model: &Model, q: &[f64], j: usize) -> Se3 {
    let parent_world = match model.parent[j] {
        Some(p) => fk_movable(model, q, p),
        None => Se3::identity(),
    };
    parent_world
        .compose(&model.parent_to_joint[j])
        .compose(&joint_local(model, j, q))
}

/// World pose of the model's tip frame.
pub fn fk_tip(model: &Model, q: &[f64]) -> Se3 {
    fk_frame(model, q, model.tip_frame())
}

// ===== reduced-space wrappers for mimic joints =====
//
// URDF `<mimic joint="src" multiplier="m" offset="b"/>` constrains
// `q[i] = m·q[src] + b`, so joint i is not an independent dof. Caliper keeps the
// FULL space untouched (every existing function still takes `q.len() == ndof`);
// these wrappers work in the REDUCED space (independent dofs only, ordered as
// `Model::independent_dofs`) by expanding through the validated full-space math.
// On a model without mimics they degrade exactly to the plain functions.

/// Reduced-space FK: expand `q_red` (`len == ndof_independent()`) through the
/// mimic constraints, then run the full-space [`fk_frame`].
pub fn fk_frame_reduced(model: &Model, q_red: &[f64], frame: usize) -> Se3 {
    fk_frame(model, &model.expand_mimic(q_red), frame)
}

/// Reduced-space geometric Jacobian (6 × ndof_independent, rows `[v; ω]`) of
/// `frame` at the expanded configuration, plus the frame pose.
///
/// Chain rule: with `q_full = E(q_red)`, `∂q_full[d]/∂q_red[k] = 1` for the
/// independent dof `d` in slot `k`, and `∂q_full[i]/∂q_red[k] = m_i` for every
/// mimic joint `i` whose source is `d`. Hence
/// `J_red[:,k] = J_full[:,d] + Σ_i m_i · J_full[:,i]` — exact, built on the
/// oracle-validated full-space [`jacobian`] (mimic chains are rejected at
/// compile, so the sum never recurses).
pub fn jacobian_reduced(
    model: &Model,
    q_red: &[f64],
    frame: usize,
    jframe: JacFrame,
) -> (Se3, DMatrix<f64>) {
    let q_full = model.expand_mimic(q_red);
    let (ee, j_full) = jacobian(model, &q_full, frame, jframe);
    let indep = model.independent_dofs();
    let mut j_red = DMatrix::<f64>::zeros(6, indep.len());
    for (k, &d) in indep.iter().enumerate() {
        let mut col = j_full.column(d).into_owned();
        for (i, mi) in model.mimic.iter().enumerate() {
            if let Some(mi) = mi
                && mi.source == d
            {
                col += j_full.column(i) * mi.multiplier;
            }
        }
        j_red.set_column(k, &col);
    }
    (ee, j_red)
}

/// Reference frame for a geometric Jacobian.
#[derive(Clone, Copy, Debug)]
pub enum JacFrame {
    /// World-aligned at the frame origin — Pinocchio `LOCAL_WORLD_ALIGNED`.
    World,
    /// Frame-local — Pinocchio `LOCAL`.
    Body,
}

/// Geometric Jacobian (6 × ndof, rows `[v; ω]`) of `frame`, fused with FK.
/// Joints not on the root→`frame` path get zero columns. Returns `(ee_pose, J)`.
pub fn jacobian(model: &Model, q: &[f64], frame: usize, jframe: JacFrame) -> (Se3, DMatrix<f64>) {
    let mut jw = vec![Se3::identity(); model.ndof];
    fk_joints(model, q, &mut jw);
    let ee = frame_pose(model, &jw, frame);
    let p_e = ee.translation_vec();

    // ancestor mask: which movable joints lie on the path to `frame`
    let mut is_anc = vec![false; model.ndof];
    let mut cur = model.frames[frame].anchor;
    while let Some(j) = cur {
        is_anc[j] = true;
        cur = model.parent[j];
    }

    let mut jac = DMatrix::<f64>::zeros(6, model.ndof);
    for i in 0..model.ndof {
        if !is_anc[i] {
            continue;
        }
        let z = jw[i].rotation() * model.axis[i]; // world joint axis
        let p_i = jw[i].translation_vec();
        let (lin, ang) = match model.kind[i] {
            JointKind::Revolute => (z.cross(&(p_e - p_i)), z), // [ z×(p_e−p_i) ; z ]
            JointKind::Prismatic => (z, Vector3::zeros()),     // [ z ; 0 ]
        };
        jac[(0, i)] = lin.x;
        jac[(1, i)] = lin.y;
        jac[(2, i)] = lin.z;
        jac[(3, i)] = ang.x;
        jac[(4, i)] = ang.y;
        jac[(5, i)] = ang.z;
    }

    match jframe {
        JacFrame::World => (ee, jac),
        JacFrame::Body => {
            let rt = ee.rotation().transpose();
            (ee, rotate_twist_rows(&jac, &rt))
        }
    }
}

/// Left-multiply each column's linear and angular 3-blocks by `rm`.
fn rotate_twist_rows(j: &DMatrix<f64>, rm: &Matrix3<f64>) -> DMatrix<f64> {
    let mut out = j.clone();
    for c in 0..j.ncols() {
        let lin = rm * Vector3::new(j[(0, c)], j[(1, c)], j[(2, c)]);
        let ang = rm * Vector3::new(j[(3, c)], j[(4, c)], j[(5, c)]);
        out[(0, c)] = lin.x;
        out[(1, c)] = lin.y;
        out[(2, c)] = lin.z;
        out[(3, c)] = ang.x;
        out[(4, c)] = ang.y;
        out[(5, c)] = ang.z;
    }
    out
}

/// How the smallest singular direction is lost at a singularity.
///
/// `kind` is the TASK-space heuristic (from the smallest left-singular vector):
/// an *advisory* label. The robust, frame-independent contract is
/// `offending_joints` (the JOINT-space nullspace). At an exact rank-deficiency
/// the left-singular vector is numerically arbitrary, so `kind` may read
/// `Boundary` even at a textbook wrist/elbow lock — assert on `offending_joints`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SingularityKind {
    None,
    Wrist,
    Elbow,
    Boundary,
}

/// Tolerances for singularity analysis + the governor (metric / rad robots).
#[derive(Clone, Copy, Debug)]
pub struct SingularityParams {
    /// Relative nullspace tolerance: `σ < eps_null·σ_max` ⇒ a null direction.
    pub eps_null: f64,
    /// `σ_min` below which the governor engages.
    pub eps_activate: f64,
    /// Maximum DLS damping `λ`.
    pub lambda_max: f64,
}
impl Default for SingularityParams {
    fn default() -> Self {
        Self {
            eps_null: 1e-6,
            eps_activate: 1e-2,
            lambda_max: 1e-1,
        }
    }
}

/// A typed, structured singularity report — Caliper's signature feature.
#[derive(Clone, Debug)]
pub struct SingularityReport {
    pub manipulability: f64,
    pub condition_number: f64,
    pub sigma_min: f64,
    pub kind: SingularityKind,
    pub offending_joints: Vec<usize>,
    /// `ndof × m` (m = number of near-zero singular values).
    pub nullspace_basis: DMatrix<f64>,
    /// `ndof`, the unit right-singular vector of `σ_min`.
    pub escape_direction: DVector<f64>,
    /// The three smallest singular values, ascending.
    pub sigma: [f64; 3],
}

/// A manipulability ellipsoid: 3 principal axes (unit) + 3 radii (= the singular
/// values of the sliced Jacobian block), sorted by DESCENDING radius. Axes are in
/// the frame of the Jacobian passed in — pass a World (LOCAL_WORLD_ALIGNED)
/// Jacobian to draw it in world space at the tip.
#[derive(Clone, Copy, Debug)]
pub struct ManipulabilityEllipsoid {
    /// Unit principal axes, `axes[0]` = major (fastest tip motion).
    pub axes: [Vector3<f64>; 3],
    /// Principal radii (velocity gains), descending. `radii[k]` pairs with `axes[k]`.
    pub radii: [f64; 3],
    /// Yoshikawa volume measure of the block, √det(J_b·J_bᵀ) = ∏ radii.
    pub volume: f64,
}

impl ManipulabilityEllipsoid {
    /// Translational (linear-velocity) ellipsoid from rows 0..3 of a 6×n geometric
    /// Jacobian. The one to draw at the tip.
    pub fn translational(j6: &DMatrix<f64>) -> Self {
        Self::from_block(j6, 0)
    }
    /// Angular-velocity ellipsoid from rows 3..6 (orientation manipulability).
    pub fn angular(j6: &DMatrix<f64>) -> Self {
        Self::from_block(j6, 3)
    }

    fn from_block(j6: &DMatrix<f64>, row0: usize) -> Self {
        let n = j6.ncols();
        if n == 0 || j6.nrows() < row0 + 3 {
            return Self {
                axes: [Vector3::x(), Vector3::y(), Vector3::z()],
                radii: [0.0; 3],
                volume: 0.0,
            };
        }
        let jb = j6.rows(row0, 3).into_owned(); // 3 × n
        // Build the 3×3 SPD core as a STATIC Matrix3 so SymmetricEigen's
        // eigenvectors are Const<3> columns and `.into()` to Vector3 compiles.
        // (DMatrix→Matrix3 `.into()` does NOT exist in nalgebra 0.35.)
        let a = Matrix3::from_fn(|r, c| jb.row(r).dot(&jb.row(c)));
        let eig = SymmetricEigen::new(a);
        let mut pairs: [(f64, Vector3<f64>); 3] = [
            (
                eig.eigenvalues[0].max(0.0).sqrt(),
                eig.eigenvectors.column(0).into(),
            ),
            (
                eig.eigenvalues[1].max(0.0).sqrt(),
                eig.eigenvectors.column(1).into(),
            ),
            (
                eig.eigenvalues[2].max(0.0).sqrt(),
                eig.eigenvectors.column(2).into(),
            ),
        ];
        // descending radius; total_cmp is infallible (eigenvalues are finite reals).
        pairs.sort_by(|x, y| y.0.total_cmp(&x.0));
        let fix = |mut v: Vector3<f64>| -> Vector3<f64> {
            let nrm = v.norm();
            if nrm > 0.0 {
                v /= nrm;
            }
            // deterministic sign: largest-|component| positive (prevents drag flicker)
            let k = (0..3)
                .max_by(|&i, &j| {
                    v[i].abs()
                        .partial_cmp(&v[j].abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            if v[k] < 0.0 {
                v = -v;
            }
            v
        };
        ManipulabilityEllipsoid {
            axes: [fix(pairs[0].1), fix(pairs[1].1), fix(pairs[2].1)],
            radii: [pairs[0].0, pairs[1].0, pairs[2].0],
            volume: pairs[0].0 * pairs[1].0 * pairs[2].0,
        }
    }
}

/// World-frame translational manipulability ellipsoid at `frame`, in one call.
pub fn manipulability_ellipsoid(model: &Model, q: &[f64], frame: usize) -> ManipulabilityEllipsoid {
    let (_, j) = jacobian(model, q, frame, JacFrame::World);
    ManipulabilityEllipsoid::translational(&j)
}

/// Ergonomic singularity + conditioning report for `frame` at `q`. Convenience
/// wrapper: builds the World (LOCAL_WORLD_ALIGNED) Jacobian and runs the single-SVD
/// [`Jacobian::analyze`]. Pinned to World so manipulability / condition_number / σ
/// match the oracle's Pinocchio LWA cross-check and the Studio world-drawn ellipsoid.
pub fn analyze(
    model: &Model,
    q: &[f64],
    frame: usize,
    params: &SingularityParams,
) -> SingularityReport {
    let (_, j) = jacobian(model, q, frame, JacFrame::World);
    Jacobian(j).analyze(params)
}

/// A geometric Jacobian wrapper carrying the SVD-based singularity analysis.
pub struct Jacobian(pub DMatrix<f64>);

/// Degenerate (no-information) report for an `n`-column Jacobian: used for a legal
/// 0-DOF / empty / non-finite Jacobian, or if nalgebra fails to return SVD factors.
fn degenerate_report(n: usize) -> SingularityReport {
    SingularityReport {
        manipulability: 0.0,
        condition_number: f64::INFINITY,
        sigma_min: 0.0,
        kind: SingularityKind::None,
        offending_joints: vec![],
        nullspace_basis: DMatrix::zeros(n, 0),
        escape_direction: DVector::zeros(n),
        sigma: [0.0; 3],
    }
}

impl Jacobian {
    /// True iff every element is finite. nalgebra's Golub–Reinsch SVD does NOT
    /// terminate on a NaN/Inf matrix, so every SVD path guards on this first.
    fn all_finite(&self) -> bool {
        self.0.iter().all(|x| x.is_finite())
    }

    /// Singular values (descending). Empty for a 0-DOF or non-finite Jacobian.
    pub fn singular_values(&self) -> DVector<f64> {
        if self.0.ncols() == 0 || self.0.nrows() == 0 || !self.all_finite() {
            return DVector::zeros(0);
        }
        self.0.clone().svd(false, false).singular_values
    }
    /// Yoshikawa manipulability = product of singular values (0 for a 0-DOF or
    /// non-finite Jacobian).
    pub fn manipulability(&self) -> f64 {
        if self.0.ncols() == 0 || self.0.nrows() == 0 || !self.all_finite() {
            return 0.0; // empty SVD has no singular values -> product()==1.0 footgun
        }
        self.singular_values().iter().product()
    }

    /// The single SVD → the full [`SingularityReport`]. Returns a sentinel for a
    /// legal 0-DOF (fixed-only) robot instead of running an empty SVD.
    pub fn analyze(&self, p: &SingularityParams) -> SingularityReport {
        let j = &self.0;
        let n = j.ncols();
        // 0-DOF, empty, or non-finite (NaN/Inf would hang the SVD) → degenerate report.
        if n == 0 || j.nrows() == 0 || !self.all_finite() {
            return degenerate_report(n);
        }
        let svd = j.clone().svd(true, true);
        let s = &svd.singular_values;
        let k = s.len();
        // Recoverable: if nalgebra somehow fails to return the SVD factors (it
        // should not, given the all_finite guard above), fall back to a degenerate
        // report rather than panicking.
        let (u, vt) = match (svd.u.as_ref(), svd.v_t.as_ref()) {
            (Some(u), Some(vt)) => (u, vt),
            _ => return degenerate_report(n),
        };

        let sigma_max = s[0];
        let sigma_min = s[k - 1];
        let manipulability: f64 = s.iter().product();
        let condition_number = if sigma_min > 0.0 {
            sigma_max / sigma_min
        } else {
            f64::INFINITY
        };

        let mut sigma = [0.0; 3];
        for (i, slot) in sigma.iter_mut().enumerate() {
            *slot = if k > i { s[k - 1 - i] } else { 0.0 };
        }

        // Nullspace via the n×n symmetric eigendecomposition of JᵀJ. The compact SVD
        // of a 6×n Jacobian only yields min(6,n) right-singular vectors in `vt`, so for
        // a redundant (n>6) arm it omits the structural self-motion modes (σ=0 directions
        // beyond the 6 tracked). JᵀJ has eigenvalues σ², INCLUDING those zeros, so its
        // eigenvectors span the FULL right nullspace. Select σ² below (eps_null·σ_max)².
        let tol = p.eps_null * sigma_max;
        let tol_sq = tol * tol;
        let jtj = j.transpose() * j; // n × n, symmetric PSD; eigenvalues = σ²
        let eig = SymmetricEigen::new(jtj);
        let null_idx: Vec<usize> = (0..n).filter(|&i| eig.eigenvalues[i] < tol_sq).collect();
        let mut nullspace_basis = DMatrix::<f64>::zeros(n, null_idx.len());
        for (c, &i) in null_idx.iter().enumerate() {
            nullspace_basis.set_column(c, &eig.eigenvectors.column(i).into_owned());
        }

        let escape_direction = DVector::from_iterator(n, vt.row(k - 1).iter().copied());
        let u_min = Vector6::from_iterator(u.column(k - 1).iter().copied());
        let kind = classify(&u_min, sigma_min, p.eps_activate);

        let emax = escape_direction
            .iter()
            .fold(0.0_f64, |a, &x| a.max(x.abs()));
        let offending_joints = (0..n)
            .filter(|&i| emax > 0.0 && escape_direction[i].abs() > 0.5 * emax)
            .collect();

        SingularityReport {
            manipulability,
            condition_number,
            sigma_min,
            kind,
            offending_joints,
            nullspace_basis,
            escape_direction,
            sigma,
        }
    }
}

/// Classify the lost direction from the smallest left-singular vector
/// `u_min = [v(0..3); ω(3..6)]`. (Per-topology geometric tests can refine this.)
fn classify(u_min: &Vector6<f64>, sigma_min: f64, eps: f64) -> SingularityKind {
    if sigma_min >= eps {
        return SingularityKind::None;
    }
    let lin = Vector3::new(u_min[0], u_min[1], u_min[2]).norm();
    let ang = Vector3::new(u_min[3], u_min[4], u_min[5]).norm();
    if ang > 1.5 * lin {
        SingularityKind::Wrist // orientation DOF collapsed
    } else if lin > 1.5 * ang {
        SingularityKind::Boundary // translation DOF collapsed at a reach edge
    } else {
        SingularityKind::Elbow // mixed translation + orientation
    }
}

/// Wraps a solver's output to stay safe near singularities.
pub struct SingularityGovernor {
    pub params: SingularityParams,
}

impl SingularityGovernor {
    pub fn new(params: SingularityParams) -> Self {
        Self { params }
    }
    /// Smooth, C¹ damping `λ²` that ramps in as `σ_min` drops below `eps_activate`.
    pub fn damping_sq(&self, sigma_min: f64) -> f64 {
        let e = self.params.eps_activate;
        if sigma_min >= e {
            0.0
        } else {
            let r = sigma_min / e;
            self.params.lambda_max.powi(2) * (1.0 - r * r)
        }
    }
    /// Scale a commanded 6-twist near a singularity: attenuate components along
    /// ill-conditioned directions when *approaching*, let them escape when
    /// *leaving*. `u`,`s` come from the same SVD used for the report.
    pub fn scale_twist(
        &self,
        v_cmd: &Vector6<f64>,
        u: &DMatrix<f64>,
        s: &DVector<f64>,
        prev_sigma_min: f64,
    ) -> Vector6<f64> {
        let k = s.len();
        // Guard the empty / dim-mismatched SVD (e.g. a 0-DOF robot): there is no
        // direction to project onto or scale, so pass the command through unchanged.
        if k == 0 || u.ncols() < k {
            return *v_cmd;
        }
        let lambda2 = self.damping_sq(s[k - 1]);
        let approaching = s[k - 1] < prev_sigma_min;
        let mut out = Vector6::zeros();
        for i in 0..k {
            let ui = Vector6::from_iterator(u.column(i).iter().copied());
            let c = ui.dot(v_cmd);
            let sg = s[i];
            let gain = if approaching {
                sg * sg / (sg * sg + lambda2)
            } else {
                1.0
            };
            out += (c * gain) * ui;
        }
        out
    }
}

// ===== path-level report (OLP cycle-time / conditioning / limit metrics) =====

/// Pre-sampled trajectory rows for [`path_report`]: parallel per-sample arrays.
/// caliper-kinematics has no trajectory type, so callers (CLI / Studio) sample
/// their `Trajectory` onto rows and pass them in — the report math stays pure.
#[derive(Clone, Copy, Debug)]
pub struct PathRows<'a> {
    /// Sample times (s), non-decreasing.
    pub times: &'a [f64],
    /// Joint positions per sample (`times.len()` rows of `ndof`).
    pub q: &'a [Vec<f64>],
    /// Joint velocities per sample.
    pub qd: &'a [Vec<f64>],
    /// Joint accelerations per sample.
    pub qdd: &'a [Vec<f64>],
}

/// Aggregate path-quality metrics along a sampled trajectory — the numbers an
/// OLP "cycle-time + reach" report is made of. Produced by [`path_report`].
#[derive(Clone, Debug)]
pub struct PathReport {
    /// Total cycle time (s): last sample time − first.
    pub cycle_time: f64,
    /// Number of samples analyzed.
    pub samples: usize,
    /// Smallest Yoshikawa manipulability over the path.
    pub min_manipulability: f64,
    /// Mean Yoshikawa manipulability over the path.
    pub mean_manipulability: f64,
    /// Worst conditioning: the smallest σ_min over the path.
    pub min_sigma_min: f64,
    /// Time (s) of the worst-σ_min sample.
    pub t_min_sigma: f64,
    /// Per joint: min distance to the nearer position limit over the path
    /// (rad | m). `f64::INFINITY` for an unbounded joint; negative = violated.
    pub limit_margin: Vec<f64>,
    /// Per joint: max |q̇| / vmax over the path (1.0 = at the limit).
    pub vel_utilization: Vec<f64>,
    /// Per joint: max |q̈| / amax over the path.
    pub acc_utilization: Vec<f64>,
}

impl PathReport {
    /// `(joint, value)` of the largest per-joint entry; `None` for 0 DOF.
    fn worst(v: &[f64]) -> Option<(usize, f64)> {
        v.iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }
    /// Worst joint velocity utilization as `(joint, max|q̇|/vmax)`.
    pub fn worst_vel_utilization(&self) -> Option<(usize, f64)> {
        Self::worst(&self.vel_utilization)
    }
    /// Worst joint acceleration utilization as `(joint, max|q̈|/amax)`.
    pub fn worst_acc_utilization(&self) -> Option<(usize, f64)> {
        Self::worst(&self.acc_utilization)
    }
    /// Tightest limit margin as `(joint, margin)`; `None` when every joint is
    /// unbounded (or 0 DOF) — infinity is "no limit", not a margin.
    pub fn min_limit_margin(&self) -> Option<(usize, f64)> {
        self.limit_margin
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, m)| m.is_finite())
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }
}

/// Compute a [`PathReport`] for `frame` over pre-sampled rows: cycle time,
/// min/mean manipulability + min σ_min (one Jacobian SVD per sample, the same
/// math as [`Jacobian::analyze`]), per-joint position-limit margins against
/// `model.limits`, and per-joint velocity/acceleration utilization against
/// `vmax`/`amax` (the caller's `MotionLimits` arrays).
///
/// Pure and total: empty rows yield a zeroed report; a non-positive `vmax[i]` /
/// `amax[i]` reports `INFINITY` utilization for any motion on that joint.
pub fn path_report(
    model: &Model,
    frame: usize,
    rows: &PathRows,
    vmax: &[f64],
    amax: &[f64],
) -> PathReport {
    let n = model.ndof;
    debug_assert_eq!(rows.q.len(), rows.times.len());
    debug_assert_eq!(rows.qd.len(), rows.times.len());
    debug_assert_eq!(rows.qdd.len(), rows.times.len());
    debug_assert_eq!(vmax.len(), n);
    debug_assert_eq!(amax.len(), n);

    let mut report = PathReport {
        cycle_time: 0.0,
        samples: rows.times.len(),
        min_manipulability: 0.0,
        mean_manipulability: 0.0,
        min_sigma_min: 0.0,
        t_min_sigma: 0.0,
        limit_margin: vec![f64::INFINITY; n],
        vel_utilization: vec![0.0; n],
        acc_utilization: vec![0.0; n],
    };
    let (Some(&t0), Some(&t1)) = (rows.times.first(), rows.times.last()) else {
        return report;
    };
    report.cycle_time = t1 - t0;
    report.min_manipulability = f64::INFINITY;
    report.min_sigma_min = f64::INFINITY;

    // `ratio` is total: a degenerate limit turns any motion into INFINITY
    // utilization instead of a NaN that would poison the max-fold.
    let ratio = |x: f64, lim: f64| -> f64 {
        if lim > 0.0 {
            x.abs() / lim
        } else if x == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    };

    let mut manip_sum = 0.0;
    for (k, t) in rows.times.iter().enumerate() {
        let (_, j) = jacobian(model, &rows.q[k], frame, JacFrame::World);
        // one SVD per sample (Jacobian::analyze's underlying decomposition);
        // empty == 0-DOF / non-finite → the degenerate 0-manipulability sample.
        let sv = Jacobian(j).singular_values();
        let (manip, sigma_min) = if sv.is_empty() {
            (0.0, 0.0) // empty product would be 1.0 — the footgun manipulability() guards
        } else {
            (sv.iter().product(), sv[sv.len() - 1])
        };
        manip_sum += manip;
        report.min_manipulability = report.min_manipulability.min(manip);
        if sigma_min < report.min_sigma_min {
            report.min_sigma_min = sigma_min;
            report.t_min_sigma = *t;
        }
        for i in 0..n {
            if let Some((lo, hi)) = model.limits[i] {
                let margin = (rows.q[k][i] - lo).min(hi - rows.q[k][i]);
                report.limit_margin[i] = report.limit_margin[i].min(margin);
            }
            report.vel_utilization[i] =
                report.vel_utilization[i].max(ratio(rows.qd[k][i], vmax[i]));
            report.acc_utilization[i] =
                report.acc_utilization[i].max(ratio(rows.qdd[k][i], amax[i]));
        }
    }
    report.mean_manipulability = manip_sum / report.samples as f64;
    report
}

// ===== trajectory linter (typed findings layered on path_report) =====

/// Severity of a lint [`Finding`]: `Error` = a hard limit is violated and the
/// trajectory must not run; `Warning` = it runs but deserves a second look.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LintSeverity {
    Warning,
    Error,
}

/// One typed trajectory-lint finding (stable code `T001`..). The message is
/// self-contained (field, got-value, expected range/unit, location); `joint`,
/// `time`, and `value` carry the same facts machine-readably so faces (CLI
/// table, Studio) never parse the message text.
#[derive(Clone, Debug)]
pub struct Finding {
    /// Stable lint code (`T001`..`T007`; `T008`+ reserved for face-side checks).
    pub code: &'static str,
    pub severity: LintSeverity,
    /// Human message naming the field, the got-value, and the expected range.
    pub message: String,
    /// A concrete way out of the defect.
    pub fix_hint: String,
    /// Offending joint; `None` for whole-path findings (singular corridor).
    pub joint: Option<usize>,
    /// Sample time (s) where the defect is worst (or the corridor's worst point).
    pub time: Option<f64>,
    /// The offending measured value (margin, utilization, ratio, jerk, σ_min).
    pub value: Option<f64>,
}

/// Per-joint motion limits the linter checks against — the caller's
/// `MotionLimits` arrays, passed as slices exactly like [`path_report`]
/// (caliper-kinematics has no trajectory/limit types of its own).
#[derive(Clone, Copy, Debug)]
pub struct LintLimits<'a> {
    /// Per-joint |q̇| bound (rad/s | m/s).
    pub vmax: &'a [f64],
    /// Per-joint |q̈| bound (rad/s² | m/s²).
    pub amax: &'a [f64],
    /// Per-joint jerk bound (rad/s³ | m/s³); the jerk-spike check scales it.
    pub jmax: &'a [f64],
}

/// Thresholds for [`lint_path`]. Defaults are metric-robot engineering
/// choices, not physics — tune per cell.
#[derive(Clone, Copy, Debug)]
pub struct LintOptions {
    /// T002/T003: flag utilization above `1 + utilization_tol` (a planned
    /// S-curve legitimately grazes 100% between samples, so exact `> 1.0`
    /// would false-positive on sampling slack).
    pub utilization_tol: f64,
    /// T004: "near" a position limit means within this margin (rad | m).
    pub near_limit_margin: f64,
    /// T004: flag when the near-limit dwell exceeds this fraction of samples.
    pub near_limit_dwell_frac: f64,
    /// T005: flag when a joint's total travel exceeds this multiple of its
    /// net start→end change (the "360° detour").
    pub travel_ratio_max: f64,
    /// T005: ignore joints whose total travel is below this (rad | m) — a
    /// tiny wiggle is noise, not a detour.
    pub travel_min: f64,
    /// T006: a jerk spike is a finite-difference jerk above this multiple of jmax.
    pub jerk_spike_ratio: f64,
    /// T007: σ_min below this marks a singular corridor (default matches
    /// [`SingularityParams::default`]'s `eps_activate`).
    pub sigma_min_threshold: f64,
}
impl Default for LintOptions {
    fn default() -> Self {
        Self {
            utilization_tol: 1e-2,
            near_limit_margin: 0.05,
            near_limit_dwell_frac: 0.25,
            travel_ratio_max: 3.0,
            travel_min: 0.5,
            jerk_spike_ratio: 1.5,
            sigma_min_threshold: 1e-2,
        }
    }
}

/// Lint a sampled trajectory: run [`path_report`] and turn its folded metrics —
/// plus per-sample passes for locations, dwell, travel, jerk, and the σ_min
/// corridor — into typed [`Finding`]s:
///
/// - `T001` (Error): position limit violated (negative limit margin).
/// - `T002` (Error): velocity utilization > 100% (+tolerance).
/// - `T003` (Error): acceleration utilization > 100% (+tolerance).
/// - `T004` (Warning): sustained dwell within `near_limit_margin` of a
///   position limit for more than `near_limit_dwell_frac` of samples.
/// - `T005` (Warning): total joint travel ≫ net start→end change — the
///   wrap-around/360° detour, located at the peak excursion off the chord.
/// - `T006` (Warning): finite-difference jerk (fd on `qdd`) above
///   `jerk_spike_ratio × jmax`. Duplicate seam times (dt ≤ 0) are skipped;
///   an infinite `jmax[i]` (e.g. TOPP output) disables the check for joint i.
/// - `T007` (Warning): singular corridor — one finding per contiguous time
///   window with σ_min below `sigma_min_threshold` (same per-sample SVD as
///   [`path_report`] / [`Jacobian::analyze`]).
///
/// Pure and total like [`path_report`]: empty rows or a 0-DOF model lint clean.
/// Findings are emitted in code order, so Errors always precede Warnings.
pub fn lint_path(
    model: &Model,
    frame: usize,
    rows: &PathRows,
    limits: &LintLimits,
    opts: &LintOptions,
) -> Vec<Finding> {
    let n = model.ndof;
    debug_assert_eq!(limits.jmax.len(), n);
    let mut out = Vec::new();
    let ns = rows.times.len();
    if ns == 0 || n == 0 {
        return out;
    }
    let report = path_report(model, frame, rows, limits.vmax, limits.amax);

    // Unit label per joint (position / velocity / accel / jerk order).
    let unit = |i: usize, order: u32| -> &'static str {
        match (model.kind[i], order) {
            (JointKind::Revolute, 0) => "rad",
            (JointKind::Revolute, 1) => "rad/s",
            (JointKind::Revolute, 2) => "rad/s^2",
            (JointKind::Revolute, _) => "rad/s^3",
            (JointKind::Prismatic, 0) => "m",
            (JointKind::Prismatic, 1) => "m/s",
            (JointKind::Prismatic, 2) => "m/s^2",
            (JointKind::Prismatic, _) => "m/s^3",
        }
    };

    // Per-sample location pass: path_report keeps only the folded numbers, so
    // recover WHERE each extreme happens (+ dwell counts, travel, σ per sample).
    let mut t_margin = vec![rows.times[0]; n]; // time of the tightest margin
    let mut t_vel = vec![rows.times[0]; n]; // time of max |q̇|
    let mut t_acc = vec![rows.times[0]; n]; // time of max |q̈|
    let mut max_qd = vec![0.0_f64; n];
    let mut max_qdd = vec![0.0_f64; n];
    let mut min_margin = vec![f64::INFINITY; n];
    let mut near_count = vec![0_usize; n];
    let mut travel = vec![0.0_f64; n];
    let mut sigma = Vec::with_capacity(ns); // σ_min per sample
    for (k, &t) in rows.times.iter().enumerate() {
        let (_, j) = jacobian(model, &rows.q[k], frame, JacFrame::World);
        let sv = Jacobian(j).singular_values();
        sigma.push(if sv.is_empty() { 0.0 } else { sv[sv.len() - 1] });
        for i in 0..n {
            if let Some((lo, hi)) = model.limits[i] {
                let margin = (rows.q[k][i] - lo).min(hi - rows.q[k][i]);
                if margin < min_margin[i] {
                    min_margin[i] = margin;
                    t_margin[i] = t;
                }
                if margin < opts.near_limit_margin {
                    near_count[i] += 1;
                }
            }
            let qd = rows.qd[k][i].abs();
            if qd > max_qd[i] {
                max_qd[i] = qd;
                t_vel[i] = t;
            }
            let qdd = rows.qdd[k][i].abs();
            if qdd > max_qdd[i] {
                max_qdd[i] = qdd;
                t_acc[i] = t;
            }
            if k > 0 {
                travel[i] += (rows.q[k][i] - rows.q[k - 1][i]).abs();
            }
        }
    }
    let last = ns - 1;

    // T001 — position limit violated (negative margin). Error.
    for (i, &t_marg) in t_margin.iter().enumerate().take(n) {
        let margin = report.limit_margin[i];
        if margin.is_finite() && margin < 0.0 {
            // a finite margin exists only when the joint has limits
            let (lo, hi) = model.limits[i].expect("finite margin implies limits");
            out.push(Finding {
                code: "T001",
                severity: LintSeverity::Error,
                message: format!(
                    "joint {i} `{}`: position limit violated by {:.4} {} at t={:.3} s (limits [{lo:.4}, {hi:.4}])",
                    model.joint_names[i],
                    -margin,
                    unit(i, 0),
                    t_marg,
                ),
                fix_hint: "re-plan the segment endpoints (or pick another IK branch) so q stays inside the position limits".into(),
                joint: Some(i),
                time: Some(t_margin[i]),
                value: Some(margin),
            });
        }
    }

    // T002 — velocity limit exceeded. Error.
    for i in 0..n {
        let util = report.vel_utilization[i];
        if util > 1.0 + opts.utilization_tol {
            out.push(Finding {
                code: "T002",
                severity: LintSeverity::Error,
                message: format!(
                    "joint {i} `{}`: velocity limit exceeded: max |qd| {:.4} {} vs vmax {:.4} ({:.1}% at t={:.3} s)",
                    model.joint_names[i],
                    max_qd[i],
                    unit(i, 1),
                    limits.vmax[i],
                    util * 100.0,
                    t_vel[i],
                ),
                fix_hint: "re-time the trajectory (retime_waypoints / longer segment time) or raise vmax".into(),
                joint: Some(i),
                time: Some(t_vel[i]),
                value: Some(util),
            });
        }
    }

    // T003 — acceleration limit exceeded. Error.
    for i in 0..n {
        let util = report.acc_utilization[i];
        if util > 1.0 + opts.utilization_tol {
            out.push(Finding {
                code: "T003",
                severity: LintSeverity::Error,
                message: format!(
                    "joint {i} `{}`: acceleration limit exceeded: max |qdd| {:.4} {} vs amax {:.4} ({:.1}% at t={:.3} s)",
                    model.joint_names[i],
                    max_qdd[i],
                    unit(i, 2),
                    limits.amax[i],
                    util * 100.0,
                    t_acc[i],
                ),
                fix_hint: "re-time the trajectory with a longer duration or raise amax".into(),
                joint: Some(i),
                time: Some(t_acc[i]),
                value: Some(util),
            });
        }
    }

    // T004 — sustained near-limit dwell. Warning (skipped when T001 already
    // fired for the joint: a violation supersedes "close to the limit").
    for i in 0..n {
        let frac = near_count[i] as f64 / ns as f64;
        if min_margin[i].is_finite() && min_margin[i] >= 0.0 && frac > opts.near_limit_dwell_frac {
            out.push(Finding {
                code: "T004",
                severity: LintSeverity::Warning,
                message: format!(
                    "joint {i} `{}`: within {:.3} {} of a position limit for {:.1}% of samples (> {:.1}% allowed); tightest margin {:.4} at t={:.3} s",
                    model.joint_names[i],
                    opts.near_limit_margin,
                    unit(i, 0),
                    frac * 100.0,
                    opts.near_limit_dwell_frac * 100.0,
                    min_margin[i],
                    t_margin[i],
                ),
                fix_hint: "re-seed IK (or shift the waypoints) away from the joint limit to keep escape headroom".into(),
                joint: Some(i),
                time: Some(t_margin[i]),
                value: Some(frac),
            });
        }
    }

    // T005 — wrap-around detour: total travel ≫ net start→end change. Warning.
    for (i, &trav) in travel.iter().enumerate().take(n) {
        let net = (rows.q[last][i] - rows.q[0][i]).abs();
        if trav > opts.travel_min && trav > opts.travel_ratio_max * net {
            // locate the detour: peak excursion off the straight time-chord q0→qend
            let (t0, t1) = (rows.times[0], rows.times[last]);
            let span = t1 - t0;
            let mut t_peak = t0;
            let mut dev = -1.0_f64;
            for (k, &t) in rows.times.iter().enumerate() {
                let alpha = if span > 0.0 { (t - t0) / span } else { 0.0 };
                let chord = rows.q[0][i] + alpha * (rows.q[last][i] - rows.q[0][i]);
                let d = (rows.q[k][i] - chord).abs();
                if d > dev {
                    dev = d;
                    t_peak = t;
                }
            }
            let ratio = trav / net; // inf when the joint loops back exactly
            out.push(Finding {
                code: "T005",
                severity: LintSeverity::Warning,
                message: format!(
                    "joint {i} `{}`: travels {:.4} {u} for a net change of {:.4} {u} (ratio {ratio:.1} > {:.1}) — wrap-around detour; peak excursion {dev:.4} {u} at t={t_peak:.3} s",
                    model.joint_names[i],
                    trav,
                    net,
                    opts.travel_ratio_max,
                    u = unit(i, 0),
                ),
                fix_hint: "pick the IK branch (or unwound joint angle) nearest the previous waypoint before planning this segment".into(),
                joint: Some(i),
                time: Some(t_peak),
                value: Some(ratio),
            });
        }
    }

    // T006 — jerk spikes: finite difference on qdd vs jerk_spike_ratio × jmax.
    for i in 0..n {
        let threshold = opts.jerk_spike_ratio * limits.jmax[i];
        if !threshold.is_finite() {
            continue; // jmax = inf (e.g. TOPP output): explicitly jerk-unlimited
        }
        let mut worst = 0.0_f64;
        let mut t_worst = rows.times[0];
        for k in 0..last {
            let dt = rows.times[k + 1] - rows.times[k];
            if dt <= 0.0 {
                continue; // duplicate seam time on a concatenated clock
            }
            let jerk = ((rows.qdd[k + 1][i] - rows.qdd[k][i]) / dt).abs();
            if jerk > worst {
                worst = jerk;
                t_worst = rows.times[k + 1];
            }
        }
        if worst > threshold {
            out.push(Finding {
                code: "T006",
                severity: LintSeverity::Warning,
                message: format!(
                    "joint {i} `{}`: jerk spike {worst:.4} {u} at t={t_worst:.3} s exceeds {:.1} x jmax = {threshold:.4} {u}",
                    model.joint_names[i],
                    opts.jerk_spike_ratio,
                    u = unit(i, 3),
                ),
                fix_hint: "re-time with the jerk-limited S-curve (move_j / retime_waypoints) or blend the acceleration step over more samples".into(),
                joint: Some(i),
                time: Some(t_worst),
                value: Some(worst),
            });
        }
    }

    // T007 — singular corridor: contiguous windows of σ_min below threshold.
    let mut k = 0;
    while k < ns {
        if sigma[k] < opts.sigma_min_threshold {
            let start = k;
            let mut worst = sigma[k];
            let mut t_worst = rows.times[k];
            while k < ns && sigma[k] < opts.sigma_min_threshold {
                if sigma[k] < worst {
                    worst = sigma[k];
                    t_worst = rows.times[k];
                }
                k += 1;
            }
            out.push(Finding {
                code: "T007",
                severity: LintSeverity::Warning,
                message: format!(
                    "singular corridor: σ_min falls to {worst:.4e} (< {:.4e}) between t={:.3} s and t={:.3} s (worst at t={t_worst:.3} s)",
                    opts.sigma_min_threshold,
                    rows.times[start],
                    rows.times[k - 1],
                ),
                fix_hint: "re-pose the path away from the singular region (see `analyze` escape_direction) or accept DLS damping through it".into(),
                joint: None,
                time: Some(t_worst),
                value: Some(worst),
            });
        } else {
            k += 1;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, UnitQuaternion};
    use std::f64::consts::PI;
    use std::path::Path;

    fn load(name: &str) -> Model {
        let p = format!(
            "{}/../../oracle/fixtures/robots/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        );
        Model::from_urdf(Path::new(&p)).unwrap()
    }
    fn toy() -> Model {
        load("toy.urdf")
    }

    #[test]
    fn fk_home_pose() {
        let m = toy();
        let p = fk_tip(&m, &[0.0, 0.0]).translation();
        assert!((p[0] - 0.2).abs() < 1e-12);
        assert!(p[1].abs() < 1e-12);
        assert!((p[2] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn fk_revolute_rotates_tip() {
        let m = toy();
        let p = fk_tip(&m, &[PI / 2.0, 0.0]).translation();
        assert!(p[0].abs() < 1e-12);
        assert!((p[1] - 0.2).abs() < 1e-12);
        assert!((p[2] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn fk_frame_matches_fk_joints() {
        let m = toy();
        let q = [0.3, -0.4];
        let mut jw = vec![Se3::identity(); m.ndof];
        fk_joints(&m, &q, &mut jw);
        for f in 0..m.frames.len() {
            let a = fk_frame(&m, &q, f);
            let b = frame_pose(&m, &jw, f);
            assert!((a.0.to_homogeneous() - b.0.to_homogeneous()).norm() < 1e-12);
        }
    }

    fn rot_log(m: &Matrix3<f64>) -> Vector3<f64> {
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*m)).scaled_axis()
    }

    fn fd_jacobian(m: &Model, q: &[f64], frame: usize, h: f64) -> DMatrix<f64> {
        let mut jfd = DMatrix::<f64>::zeros(6, m.ndof);
        for i in 0..m.ndof {
            let mut qp = q.to_vec();
            let mut qm = q.to_vec();
            qp[i] += h;
            qm[i] -= h;
            let tp = fk_frame(m, &qp, frame);
            let tm = fk_frame(m, &qm, frame);
            let v = (tp.translation_vec() - tm.translation_vec()) / (2.0 * h);
            let w = rot_log(&(tp.rotation() * tm.rotation().transpose())) / (2.0 * h);
            jfd[(0, i)] = v.x;
            jfd[(1, i)] = v.y;
            jfd[(2, i)] = v.z;
            jfd[(3, i)] = w.x;
            jfd[(4, i)] = w.y;
            jfd[(5, i)] = w.z;
        }
        jfd
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        let m = toy();
        let frame = m.tip_frame();
        for q in [[0.3, -0.4], [1.0, 0.5], [-0.7, 1.2]] {
            let (_, jac) = jacobian(&m, &q, frame, JacFrame::World);
            let jfd = fd_jacobian(&m, &q, frame, 1e-6);
            assert!(
                (&jac - &jfd).norm() < 1e-6,
                "analytic Jacobian vs finite-difference"
            );
        }
    }

    #[test]
    fn jacobian_body_world_consistent() {
        // J_body = blkdiag(Rᵀ,Rᵀ)·J_world  ⇒  rotating J_body by R recovers J_world
        let m = toy();
        let q = [0.6, -0.9];
        let f = m.tip_frame();
        let (ee, jw) = jacobian(&m, &q, f, JacFrame::World);
        let (_, jb) = jacobian(&m, &q, f, JacFrame::Body);
        let back = rotate_twist_rows(&jb, &ee.rotation());
        assert!((&back - &jw).norm() < 1e-12);
    }

    fn singular_synthetic() -> Jacobian {
        // 6×3 with column 2 == column 1 ⇒ rank 2 ⇒ a 1-D nullspace.
        let mut m = DMatrix::<f64>::zeros(6, 3);
        m[(0, 0)] = 1.0;
        m[(1, 1)] = 1.0;
        m[(2, 2)] = 1.0;
        let c1 = m.column(1).into_owned();
        m.set_column(2, &c1);
        Jacobian(m)
    }

    #[test]
    fn analyze_svd_identities() {
        let m = toy();
        let (_, j) = jacobian(&m, &[0.4, 0.7], m.tip_frame(), JacFrame::World);
        let jac = Jacobian(j.clone());
        let rep = jac.analyze(&SingularityParams::default());
        assert!((rep.escape_direction.norm() - 1.0).abs() < 1e-9);
        // ‖J · v_min‖ == σ_min  (SVD identity)
        assert!(((&j * &rep.escape_direction).norm() - rep.sigma_min).abs() < 1e-9);
        assert!((rep.manipulability - jac.manipulability()).abs() < 1e-12);
        assert_eq!(rep.kind, SingularityKind::None); // generic config is well-conditioned
    }

    #[test]
    fn analyze_detects_rank_deficiency() {
        let jac = singular_synthetic();
        let rep = jac.analyze(&SingularityParams::default());
        assert!(rep.sigma_min < 1e-9);
        assert!(rep.condition_number > 1e10);
        assert_eq!(rep.nullspace_basis.ncols(), 1);
        assert_ne!(rep.kind, SingularityKind::None);
        let null = rep.nullspace_basis.column(0).into_owned();
        assert!((&jac.0 * &null).norm() < 1e-9); // J · nullspace ≈ 0
    }

    #[test]
    fn redundant_arm_has_structural_nullspace() {
        // A 7-DOF arm at a GENERIC (full task-rank) config still has a 1-D self-motion
        // nullspace: dim ker(J) = n - rank(J) = 7 - 6 = 1. The compact 6×7 SVD's v_t
        // only carries 6 right vectors and would report 0 columns; the JᵀJ
        // eigendecomposition must recover the structural σ=0 direction.
        let m = load("redundant7.urdf");
        assert_eq!(m.ndof, 7);
        let f = m.tip_frame();
        let q = [0.3, -0.4, 0.6, 0.2, -0.5, 0.1, 0.35]; // generic, non-singular
        let (_, j) = jacobian(&m, &q, f, JacFrame::World);
        let rep = Jacobian(j.clone()).analyze(&SingularityParams::default());
        // full task rank here ⇒ not flagged singular, yet a self-motion mode exists
        assert_eq!(rep.kind, SingularityKind::None);
        assert!(
            rep.nullspace_basis.ncols() >= 1,
            "redundant arm must expose >=1 nullspace column (got {})",
            rep.nullspace_basis.ncols()
        );
        // every reported nullspace column is genuinely in ker(J): ‖J·n‖ ≈ 0
        for c in 0..rep.nullspace_basis.ncols() {
            let n_c = rep.nullspace_basis.column(c).into_owned();
            assert!((n_c.norm() - 1.0).abs() < 1e-9, "nullspace col {c} unit");
            assert!(
                (&j * &n_c).norm() < 1e-9,
                "‖J·n‖ for nullspace col {c} must vanish"
            );
        }
    }

    #[test]
    fn governor_damping_is_continuous_and_monotone() {
        let g = SingularityGovernor::new(SingularityParams::default());
        let e = g.params.eps_activate;
        assert_eq!(g.damping_sq(e + 1.0), 0.0); // off above threshold
        assert!(g.damping_sq(e - 1e-12) < 1e-6); // continuous at the boundary
        assert!(g.damping_sq(0.0) > 0.0); // engaged at the singularity
        assert!(g.damping_sq(e * 0.5) > g.damping_sq(e * 0.9)); // more damping as σ→0
    }

    #[test]
    fn prismatic_fk_and_jacobian() {
        let m = load("prismatic.urdf");
        let frame = m.tip_frame();
        let home = fk_tip(&m, &[0.0, 0.0]).translation();
        let moved = fk_tip(&m, &[0.0, 0.25]).translation();
        assert!((moved[0] - (home[0] + 0.25)).abs() < 1e-12);
        assert!((moved[1] - home[1]).abs() < 1e-12 && (moved[2] - home[2]).abs() < 1e-12);
        // analytic Jacobian (incl. the prismatic [z;0] column) vs finite-difference
        let (_, jac) = jacobian(&m, &[0.3, 0.2], frame, JacFrame::World);
        let jfd = fd_jacobian(&m, &[0.3, 0.2], frame, 1e-6);
        assert!((&jac - &jfd).norm() < 1e-6);
    }

    #[test]
    fn branched_ancestor_masking() {
        let m = load("branched.urdf");
        let q = vec![0.3; m.ndof];
        let fa = m.frame_id("tipA").unwrap();
        let fb = m.frame_id("tipB").unwrap();
        let (_, ja) = jacobian(&m, &q, fa, JacFrame::World);
        let (_, jb) = jacobian(&m, &q, fb, JacFrame::World);
        let col = |name: &str| m.joint_names.iter().position(|n| n == name).unwrap();
        // tipA does not depend on j3; tipB does not depend on j2
        assert!(ja.column(col("j3")).norm() < 1e-15);
        assert!(jb.column(col("j2")).norm() < 1e-15);
        // but each DOES depend on its own branch joint
        assert!(ja.column(col("j2")).norm() > 1e-6);
        assert!(jb.column(col("j3")).norm() > 1e-6);
    }

    #[test]
    fn zero_dof_analyze_does_not_panic() {
        let m = load("fixed_only.urdf");
        assert_eq!(m.ndof, 0);
        let (_, j) = jacobian(&m, &[], m.tip_frame(), JacFrame::World);
        let rep = Jacobian(j).analyze(&SingularityParams::default());
        assert_eq!(rep.kind, SingularityKind::None);
        assert!((fk_tip(&m, &[]).translation()[2] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn manipulability_ellipsoid_matches_jv_svd() {
        // translational ellipsoid radii == singular values of the linear block Jv,
        // axes are unit + orthogonal + sorted descending.
        let m = load("showcase6.urdf");
        let q = [0.3, -0.4, 0.6, 0.2, -0.5, 0.1];
        let f = m.tip_frame();
        let (_, j) = jacobian(&m, &q, f, JacFrame::World);
        let ell = ManipulabilityEllipsoid::translational(&j);
        // reference: SVD of Jv (top 3 rows)
        let jv = j.rows(0, 3).into_owned();
        let mut sv: Vec<f64> = jv
            .svd(false, false)
            .singular_values
            .iter()
            .copied()
            .collect();
        sv.sort_by(|a, b| b.total_cmp(a)); // descending (infallible)
        #[allow(clippy::needless_range_loop)] // parallel-index radii/axes/sv reads clearest
        for k in 0..3 {
            assert!((ell.radii[k] - sv[k]).abs() < 1e-9, "radius {k}");
            assert!((ell.axes[k].norm() - 1.0).abs() < 1e-9, "axis {k} unit");
        }
        assert!(ell.radii[0] >= ell.radii[1] && ell.radii[1] >= ell.radii[2]);
        // axes orthonormal (distinct eigenvalues here)
        assert!(ell.axes[0].dot(&ell.axes[1]).abs() < 1e-7);
        assert!(ell.axes[0].dot(&ell.axes[2]).abs() < 1e-7);
        assert!((ell.volume - ell.radii[0] * ell.radii[1] * ell.radii[2]).abs() < 1e-12);
    }

    #[test]
    fn free_analyze_matches_world_jacobian_analyze() {
        let m = load("showcase6.urdf");
        let q = [0.3, -0.4, 0.6, 0.2, -0.5, 0.1];
        let f = m.tip_frame();
        let rep = analyze(&m, &q, f, &SingularityParams::default());
        let (_, j) = jacobian(&m, &q, f, JacFrame::World);
        let want = Jacobian(j).analyze(&SingularityParams::default());
        assert!((rep.manipulability - want.manipulability).abs() < 1e-12);
        assert!((rep.sigma_min - want.sigma_min).abs() < 1e-12);
        assert_eq!(rep.kind, SingularityKind::None);
    }

    // ===== reduced-space mimic wrappers =====

    /// Central finite difference of `fk_frame_reduced` per INDEPENDENT dof —
    /// the numeric reference for `jacobian_reduced` (same idiom as `fd_jacobian`,
    /// but stepping the reduced coordinates through the mimic expansion).
    fn fd_jacobian_reduced(m: &Model, q_red: &[f64], frame: usize, h: f64) -> DMatrix<f64> {
        let n = m.ndof_independent();
        let mut jfd = DMatrix::<f64>::zeros(6, n);
        for k in 0..n {
            let mut qp = q_red.to_vec();
            let mut qm = q_red.to_vec();
            qp[k] += h;
            qm[k] -= h;
            let tp = fk_frame_reduced(m, &qp, frame);
            let tm = fk_frame_reduced(m, &qm, frame);
            let v = (tp.translation_vec() - tm.translation_vec()) / (2.0 * h);
            let w = rot_log(&(tp.rotation() * tm.rotation().transpose())) / (2.0 * h);
            jfd[(0, k)] = v.x;
            jfd[(1, k)] = v.y;
            jfd[(2, k)] = v.z;
            jfd[(3, k)] = w.x;
            jfd[(4, k)] = w.y;
            jfd[(5, k)] = w.z;
        }
        jfd
    }

    #[test]
    fn fk_reduced_equals_fk_of_expanded() {
        let m = load("gripper_mimic.urdf");
        assert_eq!(m.ndof_independent(), 2);
        let q_red = [0.7, 0.03];
        let q_full = m.expand_mimic(&q_red);
        for f in 0..m.frames.len() {
            let a = fk_frame_reduced(&m, &q_red, f);
            let b = fk_frame(&m, &q_full, f);
            // definitionally the same call path — must be EXACTLY equal
            assert_eq!(
                a.0.to_homogeneous(),
                b.0.to_homogeneous(),
                "frame {} ({})",
                f,
                m.frame_name(f)
            );
        }
    }

    #[test]
    fn jacobian_reduced_matches_finite_difference() {
        let m = load("gripper_mimic.urdf");
        for frame in [m.frame_id("finger_r").unwrap(), m.frame_id("palm").unwrap()] {
            for q_red in [[0.7, 0.03], [-0.4, 0.01], [1.2, 0.0]] {
                let (_, j) = jacobian_reduced(&m, &q_red, frame, JacFrame::World);
                assert_eq!(j.ncols(), 2);
                let jfd = fd_jacobian_reduced(&m, &q_red, frame, 1e-6);
                assert!(
                    (&j - &jfd).amax() < 1e-6,
                    "reduced Jacobian vs FD, frame {frame}: analytic=\n{j}\nfd=\n{jfd}"
                );
            }
        }
    }

    #[test]
    fn jacobian_reduced_chain_rule_identity() {
        // J_red[:,k] == J_full[:,d] + m_i·J_full[:,i] for each mimic i sourced at d.
        let m = load("gripper_mimic.urdf");
        let q_red = [0.7, 0.03];
        let q_full = m.expand_mimic(&q_red);
        let frame = m.frame_id("finger_r").unwrap();
        let (ee_r, j_red) = jacobian_reduced(&m, &q_red, frame, JacFrame::World);
        let (ee_f, j_full) = jacobian(&m, &q_full, frame, JacFrame::World);
        assert_eq!(ee_r.0.to_homogeneous(), ee_f.0.to_homogeneous());
        // dof 0 (arm) drives mimic 1 (wrist, m=0.5); dof 2 (finger1) drives 3 (m=-1)
        let want0 = j_full.column(0) + j_full.column(1) * 0.5;
        let want1 = j_full.column(2) + j_full.column(3) * (-1.0);
        assert!((j_red.column(0) - want0).norm() < 1e-15);
        assert!((j_red.column(1) - want1).norm() < 1e-15);
    }

    #[test]
    fn reduced_wrappers_degrade_without_mimic() {
        // On a mimic-free model the wrappers must EXACTLY reproduce the plain calls.
        let m = load("showcase6.urdf");
        assert!(!m.has_mimic());
        let q = [0.3, -0.4, 0.6, 0.2, -0.5, 0.1];
        let f = m.tip_frame();
        assert_eq!(
            fk_frame_reduced(&m, &q, f).0.to_homogeneous(),
            fk_frame(&m, &q, f).0.to_homogeneous()
        );
        for jf in [JacFrame::World, JacFrame::Body] {
            let (er, jr) = jacobian_reduced(&m, &q, f, jf);
            let (ep, jp) = jacobian(&m, &q, f, jf);
            assert_eq!(er.0.to_homogeneous(), ep.0.to_homogeneous());
            assert_eq!(jr, jp);
        }
    }

    /// A NaN/Inf Jacobian must yield a degenerate report immediately — nalgebra's
    /// SVD never terminates on a non-finite matrix, so the guard is load-bearing.
    /// (If the guard regresses this test hangs and the suite times out.)
    #[test]
    fn analyze_rejects_nonfinite_without_hanging() {
        let mut j = DMatrix::<f64>::zeros(6, 3);
        j[(0, 0)] = f64::NAN;
        j[(2, 1)] = f64::INFINITY;
        let jac = Jacobian(j);
        let rep = jac.analyze(&SingularityParams::default());
        assert_eq!(rep.kind, SingularityKind::None);
        assert_eq!(rep.sigma_min, 0.0);
        assert_eq!(rep.manipulability, 0.0);
        assert_eq!(jac.manipulability(), 0.0);
        assert_eq!(jac.singular_values().len(), 0);
    }

    // ===== path_report =====

    /// (times, q, qd, qdd) rows for the fixture below.
    type Rows = (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>);

    /// Deterministic 3-sample fixture on toy.urdf (limits ±3.14 on both joints).
    fn toy_rows() -> Rows {
        let times = vec![0.0, 0.5, 1.25];
        let q = vec![vec![0.0, 0.0], vec![0.3, -0.4], vec![0.6, 3.0]];
        let qd = vec![vec![0.0, 0.0], vec![1.5, -0.8], vec![0.0, 0.0]];
        let qdd = vec![vec![0.0, 0.0], vec![-2.0, 4.5], vec![0.0, 0.0]];
        (times, q, qd, qdd)
    }

    #[test]
    fn path_report_hand_checked_metrics() {
        let m = toy();
        let (times, q, qd, qdd) = toy_rows();
        let rows = PathRows {
            times: &times,
            q: &q,
            qd: &qd,
            qdd: &qdd,
        };
        let rep = path_report(&m, m.tip_frame(), &rows, &[3.0, 3.0], &[10.0, 10.0]);
        assert_eq!(rep.samples, 3);
        assert!((rep.cycle_time - 1.25).abs() < 1e-12);
        // limit margins: joint 0 tightest at q=0.6 → min(0.6+3.14, 3.14-0.6)=2.54;
        // joint 1 tightest at q=3.0 → 3.14-3.0 = 0.14.
        assert!((rep.limit_margin[0] - 2.54).abs() < 1e-12);
        assert!((rep.limit_margin[1] - 0.14).abs() < 1e-12);
        let (j, margin) = rep.min_limit_margin().unwrap();
        assert_eq!(j, 1);
        assert!((margin - 0.14).abs() < 1e-12);
        // utilization: max|qd| = [1.5, 0.8] / 3.0; max|qdd| = [2.0, 4.5] / 10.0.
        assert!((rep.vel_utilization[0] - 0.5).abs() < 1e-12);
        assert!((rep.vel_utilization[1] - 0.8 / 3.0).abs() < 1e-12);
        assert_eq!(rep.worst_vel_utilization().unwrap(), (0, 0.5));
        assert!((rep.acc_utilization[0] - 0.2).abs() < 1e-12);
        assert!((rep.acc_utilization[1] - 0.45).abs() < 1e-12);
        assert_eq!(rep.worst_acc_utilization().unwrap().0, 1);
    }

    #[test]
    fn path_report_matches_per_sample_analyze() {
        // min/mean manipulability and min σ_min must agree with running the full
        // Jacobian::analyze at every sample — the report is the same SVD, folded.
        let m = toy();
        let (times, q, qd, qdd) = toy_rows();
        let rows = PathRows {
            times: &times,
            q: &q,
            qd: &qd,
            qdd: &qdd,
        };
        let rep = path_report(&m, m.tip_frame(), &rows, &[3.0, 3.0], &[10.0, 10.0]);
        let mut min_manip = f64::INFINITY;
        let mut sum = 0.0;
        let mut min_sigma = f64::INFINITY;
        let mut t_min = 0.0;
        for (k, t) in times.iter().enumerate() {
            let a = analyze(&m, &q[k], m.tip_frame(), &SingularityParams::default());
            min_manip = min_manip.min(a.manipulability);
            sum += a.manipulability;
            if a.sigma_min < min_sigma {
                min_sigma = a.sigma_min;
                t_min = *t;
            }
        }
        assert!((rep.min_manipulability - min_manip).abs() < 1e-12);
        assert!((rep.mean_manipulability - sum / 3.0).abs() < 1e-12);
        assert!((rep.min_sigma_min - min_sigma).abs() < 1e-12);
        assert_eq!(rep.t_min_sigma, t_min);
        // determinism: same rows → identical report
        let rep2 = path_report(&m, m.tip_frame(), &rows, &[3.0, 3.0], &[10.0, 10.0]);
        assert_eq!(rep.min_sigma_min, rep2.min_sigma_min);
        assert_eq!(rep.mean_manipulability, rep2.mean_manipulability);
    }

    #[test]
    fn path_report_empty_and_degenerate_limits() {
        let m = toy();
        let rows = PathRows {
            times: &[],
            q: &[],
            qd: &[],
            qdd: &[],
        };
        let rep = path_report(&m, m.tip_frame(), &rows, &[3.0, 3.0], &[10.0, 10.0]);
        assert_eq!(rep.samples, 0);
        assert_eq!(rep.cycle_time, 0.0);
        assert_eq!(rep.min_manipulability, 0.0);
        assert_eq!(rep.min_sigma_min, 0.0);
        assert!(rep.limit_margin.iter().all(|mg| mg.is_infinite()));
        assert!(rep.min_limit_margin().is_none()); // infinity = "no limit", not a margin
        // a zero vmax must flag moving joints as INFINITY, not NaN
        let (times, q, qd, qdd) = toy_rows();
        let rows = PathRows {
            times: &times,
            q: &q,
            qd: &qd,
            qdd: &qdd,
        };
        let rep = path_report(&m, m.tip_frame(), &rows, &[0.0, 3.0], &[10.0, 10.0]);
        assert!(rep.vel_utilization[0].is_infinite());
        assert!(rep.vel_utilization[1].is_finite());
    }

    // ===== lint_path =====

    /// Zero rows for `n` joints at the given `q` samples (lint fixtures that
    /// only exercise position-based checks).
    fn zero_rows(times: Vec<f64>, q: Vec<Vec<f64>>) -> Rows {
        let n = q[0].len();
        let z = vec![vec![0.0; n]; q.len()];
        (times, q, z.clone(), z)
    }

    fn lint_toy(rows: &Rows, vmax: &[f64], amax: &[f64], jmax: &[f64]) -> Vec<Finding> {
        let m = toy();
        let pr = PathRows {
            times: &rows.0,
            q: &rows.1,
            qd: &rows.2,
            qdd: &rows.3,
        };
        let limits = LintLimits { vmax, amax, jmax };
        lint_path(&m, m.tip_frame(), &pr, &limits, &LintOptions::default())
    }

    /// A well-behaved path on toy.urdf (small motion, far from every limit,
    /// never singular — a planar 2R with nonzero link lengths has σ_min > 0
    /// everywhere) lints completely clean. The negative control for T001–T007.
    #[test]
    fn lint_clean_path_zero_findings() {
        let rows = (
            vec![0.0, 0.5, 1.0],
            vec![vec![0.0, 0.0], vec![0.1, -0.1], vec![0.2, -0.2]],
            vec![vec![0.0, 0.0], vec![0.3, 0.3], vec![0.0, 0.0]],
            vec![vec![0.0, 0.0], vec![0.5, 0.5], vec![0.0, 0.0]],
        );
        let f = lint_toy(&rows, &[3.0, 3.0], &[10.0, 10.0], &[50.0, 50.0]);
        assert!(f.is_empty(), "clean path must lint clean, got {f:?}");
    }

    /// T001 positive: q beyond the +3.14 limit. Ground truth: margin
    /// 3.14 − 3.3 = −0.16, tightest at every sample (t = first sample).
    #[test]
    // 3.14 below is toy.urdf's literal joint limit, not an approximation of π.
    #[allow(clippy::approx_constant)]
    fn lint_flags_position_limit_violation() {
        let rows = zero_rows(vec![0.0, 1.0], vec![vec![0.0, 3.3], vec![0.0, 3.3]]);
        let f = lint_toy(&rows, &[3.0, 3.0], &[10.0, 10.0], &[50.0, 50.0]);
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].code, "T001");
        assert_eq!(f[0].severity, LintSeverity::Error);
        assert_eq!(f[0].joint, Some(1));
        assert!((f[0].value.unwrap() - (3.14 - 3.3)).abs() < 1e-12);
        assert!(f[0].message.contains("j2"), "{}", f[0].message);
    }

    /// T002 positive: |q̇| = 4 vs vmax = 3 ⇒ utilization exactly 4/3 at t=1.
    #[test]
    fn lint_flags_velocity_limit() {
        let rows = (
            vec![0.0, 1.0],
            vec![vec![0.0, 0.0], vec![0.1, 0.0]],
            vec![vec![0.0, 0.0], vec![4.0, 0.0]],
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        );
        let f = lint_toy(&rows, &[3.0, 3.0], &[10.0, 10.0], &[50.0, 50.0]);
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].code, "T002");
        assert_eq!(f[0].severity, LintSeverity::Error);
        assert_eq!(f[0].joint, Some(0));
        assert_eq!(f[0].time, Some(1.0));
        assert!((f[0].value.unwrap() - 4.0 / 3.0).abs() < 1e-15);
    }

    /// T003 positive: |q̈| = 45 vs amax = 10 ⇒ utilization exactly 4.5. jmax is
    /// generous (threshold 1.5·200 = 300 > fd jerk 45), so no T006 rides along.
    #[test]
    fn lint_flags_acceleration_limit() {
        let rows = (
            vec![0.0, 1.0],
            vec![vec![0.0, 0.0], vec![0.1, 0.0]],
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            vec![vec![0.0, 0.0], vec![0.0, 45.0]],
        );
        let f = lint_toy(&rows, &[3.0, 3.0], &[10.0, 10.0], &[200.0, 200.0]);
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].code, "T003");
        assert_eq!(f[0].severity, LintSeverity::Error);
        assert_eq!(f[0].joint, Some(1));
        assert!((f[0].value.unwrap() - 4.5).abs() < 1e-15);
    }

    /// T004 positive: joint 0 parks at 3.10, margin 3.14 − 3.10 = 0.04 <
    /// near_limit_margin (0.05) for 3/3 samples ⇒ dwell fraction exactly 1.
    #[test]
    fn lint_flags_near_limit_dwell() {
        let rows = zero_rows(
            vec![0.0, 1.0, 2.0],
            vec![vec![3.10, 0.0], vec![3.10, 0.0], vec![3.10, 0.0]],
        );
        let f = lint_toy(&rows, &[3.0, 3.0], &[10.0, 10.0], &[50.0, 50.0]);
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].code, "T004");
        assert_eq!(f[0].severity, LintSeverity::Warning);
        assert_eq!(f[0].joint, Some(0));
        assert!((f[0].value.unwrap() - 1.0).abs() < 1e-15);
    }

    /// T005 positive: joint 0 runs 0 → 2 → 0.1. Ground truth: travel
    /// 2 + 1.9 = 3.9, net 0.1 ⇒ ratio 39 > 3; the chord 0→0.1 puts the peak
    /// excursion at the middle sample (t = 1).
    #[test]
    fn lint_flags_travel_detour() {
        let rows = zero_rows(
            vec![0.0, 1.0, 2.0],
            vec![vec![0.0, 0.0], vec![2.0, 0.0], vec![0.1, 0.0]],
        );
        let f = lint_toy(&rows, &[3.0, 3.0], &[10.0, 10.0], &[50.0, 50.0]);
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].code, "T005");
        assert_eq!(f[0].severity, LintSeverity::Warning);
        assert_eq!(f[0].joint, Some(0));
        assert_eq!(f[0].time, Some(1.0));
        assert!((f[0].value.unwrap() - 39.0).abs() < 1e-9);
    }

    /// T006 positive: qdd steps 0 → 5 over dt = 0.1 ⇒ fd jerk exactly 50,
    /// above 1.5 × jmax = 15, landing at the second sample (t = 0.1).
    #[test]
    fn lint_flags_jerk_spike() {
        let rows = (
            vec![0.0, 0.1, 0.2],
            vec![vec![0.0, 0.0], vec![0.01, 0.0], vec![0.02, 0.0]],
            vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]],
            vec![vec![0.0, 0.0], vec![5.0, 0.0], vec![0.0, 0.0]],
        );
        let f = lint_toy(&rows, &[3.0, 3.0], &[10.0, 10.0], &[10.0, 10.0]);
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].code, "T006");
        assert_eq!(f[0].severity, LintSeverity::Warning);
        assert_eq!(f[0].joint, Some(0));
        assert_eq!(f[0].time, Some(0.1));
        assert!((f[0].value.unwrap() - 50.0).abs() < 1e-9);
    }

    /// T007 positive on showcase6: the all-zeros home is wrist-singular
    /// (σ_min < 1e-6, pinned by the CLI report test) while the generic config
    /// has σ_min ≥ eps_activate (pinned by `free_analyze_matches_...` via
    /// `kind == None`). One contiguous window around the singular sample, and
    /// the reported σ_min cross-checks against `analyze` exactly. jmax = ∞
    /// (TOPP-style) exercises the jerk-check opt-out.
    #[test]
    fn lint_flags_singular_corridor() {
        let m = load("showcase6.urdf");
        let f = m.tip_frame();
        let generic = vec![0.3, -0.4, 0.6, 0.2, -0.5, 0.1];
        let home = vec![0.0; 6];
        let p = SingularityParams::default();
        // fixture preconditions (self-diagnosing, both pinned elsewhere)
        assert!(analyze(&m, &generic, f, &p).sigma_min > 1e-2);
        assert!(analyze(&m, &home, f, &p).sigma_min < 1e-2);
        let (times, q) = (
            vec![0.0, 1.0, 2.0],
            vec![generic.clone(), home.clone(), generic.clone()],
        );
        let z = vec![vec![0.0; 6]; 3];
        let rows = PathRows {
            times: &times,
            q: &q,
            qd: &z,
            qdd: &z,
        };
        let limits = LintLimits {
            vmax: &[10.0; 6],
            amax: &[10.0; 6],
            jmax: &[f64::INFINITY; 6],
        };
        // disable dwell/detour so the test pins the corridor check alone
        // (showcase6's exact limit values are not this test's business)
        let opts = LintOptions {
            near_limit_dwell_frac: 1.1,
            travel_min: 100.0,
            ..LintOptions::default()
        };
        let findings = lint_path(&m, f, &rows, &limits, &opts);
        assert_eq!(findings.len(), 1, "{findings:?}");
        let c = &findings[0];
        assert_eq!(c.code, "T007");
        assert_eq!(c.severity, LintSeverity::Warning);
        assert_eq!(c.joint, None);
        assert_eq!(c.time, Some(1.0));
        // numeric cross-check: the corridor's worst σ_min IS analyze's σ_min
        let want = analyze(&m, &home, f, &p).sigma_min;
        assert!((c.value.unwrap() - want).abs() < 1e-12);

        // two disjoint singular windows ⇒ two T007 findings
        let q2 = vec![home.clone(), generic.clone(), home];
        let rows2 = PathRows {
            times: &times,
            q: &q2,
            qd: &z,
            qdd: &z,
        };
        let findings2 = lint_path(&m, f, &rows2, &limits, &opts);
        assert_eq!(findings2.len(), 2, "{findings2:?}");
        assert!(findings2.iter().all(|x| x.code == "T007"));
        assert_eq!(findings2[0].time, Some(0.0));
        assert_eq!(findings2[1].time, Some(2.0));
    }

    /// Total on degenerate inputs: empty rows and a 0-DOF model lint clean.
    #[test]
    fn lint_empty_and_zero_dof() {
        let m = toy();
        let empty = PathRows {
            times: &[],
            q: &[],
            qd: &[],
            qdd: &[],
        };
        let limits = LintLimits {
            vmax: &[3.0, 3.0],
            amax: &[10.0, 10.0],
            jmax: &[50.0, 50.0],
        };
        assert!(lint_path(&m, m.tip_frame(), &empty, &limits, &LintOptions::default()).is_empty());

        let m0 = load("fixed_only.urdf");
        assert_eq!(m0.ndof, 0);
        let times = [0.0, 1.0];
        let q: Vec<Vec<f64>> = vec![vec![], vec![]];
        let rows = PathRows {
            times: &times,
            q: &q,
            qd: &q,
            qdd: &q,
        };
        let l0 = LintLimits {
            vmax: &[],
            amax: &[],
            jmax: &[],
        };
        assert!(lint_path(&m0, m0.tip_frame(), &rows, &l0, &LintOptions::default()).is_empty());
    }
}

/// Property-based (fuzz) tests over bounded, finite random configurations.
#[cfg(test)]
mod proptests {
    use super::*;
    use nalgebra::{Rotation3, UnitQuaternion};
    use proptest::prelude::*;
    use std::path::Path;

    fn load(name: &str) -> Model {
        let p = format!(
            "{}/../../oracle/fixtures/robots/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        );
        Model::from_urdf(Path::new(&p)).unwrap()
    }

    fn rot_log(m: &Matrix3<f64>) -> Vector3<f64> {
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*m)).scaled_axis()
    }

    /// Central finite-difference of FK, the independent reference for the analytic
    /// Jacobian (same idiom as the deterministic unit test).
    fn fd_jacobian(m: &Model, q: &[f64], frame: usize, h: f64) -> DMatrix<f64> {
        let mut jfd = DMatrix::<f64>::zeros(6, m.ndof);
        for i in 0..m.ndof {
            let mut qp = q.to_vec();
            let mut qm = q.to_vec();
            qp[i] += h;
            qm[i] -= h;
            let tp = fk_frame(m, &qp, frame);
            let tm = fk_frame(m, &qm, frame);
            let v = (tp.translation_vec() - tm.translation_vec()) / (2.0 * h);
            let w = rot_log(&(tp.rotation() * tm.rotation().transpose())) / (2.0 * h);
            jfd[(0, i)] = v.x;
            jfd[(1, i)] = v.y;
            jfd[(2, i)] = v.z;
            jfd[(3, i)] = w.x;
            jfd[(4, i)] = w.y;
            jfd[(5, i)] = w.z;
        }
        jfd
    }

    /// Assert FK yields a valid SE(3) and the analytic Jacobian matches central FD,
    /// per-element to ~1e-6, for an arbitrary bounded configuration `q`.
    fn check_config(m: &Model, q: &[f64]) -> Result<(), TestCaseError> {
        let frame = m.tip_frame();
        // FK is SE(3)-valid: rotation orthonormal with det +1.
        let r = fk_tip(m, q).rotation();
        prop_assert!((r.transpose() * r - Matrix3::identity()).norm() < 1e-9);
        prop_assert!((r.determinant() - 1.0).abs() < 1e-9);
        // analytic Jacobian == central finite-difference (per-element ~1e-6).
        let (_, jac) = jacobian(m, q, frame, JacFrame::World);
        let jfd = fd_jacobian(m, q, frame, 1e-6);
        prop_assert!((&jac - &jfd).amax() < 1e-5);
        Ok(())
    }

    proptest! {
        /// 2-DOF arm: FK valid + analytic Jacobian matches FD over random q.
        #[test]
        fn fk_se3_valid_and_jacobian_matches_fd_2dof(
            q in prop::collection::vec(-2.5f64..2.5, 2),
        ) {
            let m = load("toy.urdf");
            prop_assume!(q.len() == m.ndof);
            check_config(&m, &q)?;
        }

        /// 6-DOF arm: FK valid + analytic Jacobian matches FD over random q.
        #[test]
        fn fk_se3_valid_and_jacobian_matches_fd_6dof(
            q in prop::collection::vec(-2.5f64..2.5, 6),
        ) {
            let m = load("showcase6.urdf");
            prop_assume!(q.len() == m.ndof);
            check_config(&m, &q)?;
        }
    }
}
