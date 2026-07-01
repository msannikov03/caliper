//! Inverse dynamics (RNEA), joint-space mass matrix (CRBA), forward dynamics, and
//! a semi-implicit-Euler Simulator. Body-frame spatial algebra in [v;ω] order,
//! reusing caliper-spatial's adjoint / adjoint_inv / small_adjoint / SpatialInertia.
//! Cross-validated against Pinocchio (oracle/tests/test_pinocchio_dynamics.py).
use caliper_kinematics::fk_joints;
use caliper_model::{JointKind, Model};
use caliper_spatial::{Se3, Twist, exp_prismatic, exp_revolute, small_adjoint};
use nalgebra::{DMatrix, DVector, Matrix6, Vector3, Vector6};
use std::sync::Arc;

/// Default Earth gravity in the URDF world (Z-up); matches Pinocchio's
/// `model.gravity.linear = (0,0,-9.81)`.
pub const GRAVITY_EARTH: Vector3<f64> = Vector3::new(0.0, 0.0, -9.81);

#[derive(thiserror::Error, Debug)]
pub enum DynError {
    #[error("model has no inertial data; dynamics needs <inertial> blocks on every link")]
    NoInertia,
    #[error("expected {expected} dofs, got {got}")]
    Dim { expected: usize, got: usize },
    #[error("mass matrix not positive-definite (Cholesky failed) at this configuration")]
    NotSpd,
    #[error("simulation diverged: non-finite state")]
    Diverged,
}

fn check_dims(model: &Model, slices: &[(&str, usize)]) -> Result<(), DynError> {
    if !model.has_inertia {
        return Err(DynError::NoInertia);
    }
    for &(_, len) in slices {
        if len != model.ndof {
            return Err(DynError::Dim {
                expected: model.ndof,
                got: len,
            });
        }
    }
    Ok(())
}

/// Local exp of movable joint `i` at `q[i]` (mirrors kinematics joint-local).
#[inline]
fn joint_local(model: &Model, i: usize, q: &[f64]) -> Se3 {
    match model.kind[i] {
        JointKind::Revolute => exp_revolute(&model.axis[i], q[i]),
        JointKind::Prismatic => exp_prismatic(&model.axis[i], q[i]),
    }
}

/// Motion subspace `S_i` (6-vector, joint frame), [v;ω] order:
/// revolute `[0;axis]`, prismatic `[axis;0]`.
#[inline]
fn subspace(model: &Model, i: usize) -> Vector6<f64> {
    let a = model.axis[i];
    match model.kind[i] {
        JointKind::Revolute => Vector6::new(0.0, 0.0, 0.0, a.x, a.y, a.z),
        JointKind::Prismatic => Vector6::new(a.x, a.y, a.z, 0.0, 0.0, 0.0),
    }
}

/// child-from-parent 6×6 inverse adjoint: maps a twist from the PARENT-link frame
/// into link `i`'s frame, `ⁱX_parent = Ad( parent_to_joint[i] · jointLocal(i) )⁻¹`.
#[inline]
fn i_x_parent(model: &Model, i: usize, q: &[f64]) -> Matrix6<f64> {
    model.parent_to_joint[i]
        .compose(&joint_local(model, i, q))
        .adjoint_inv()
}

/// Inverse dynamics (RNEA): `τ = ID(q, qd, qdd)` including gravity + Coriolis.
/// Gravity enters as a fictitious base acceleration `a₀ = [−gravity; 0]` (the
/// standard trick; matches Pinocchio, which seeds `a₀ = −model.gravity`). So a
/// static link feels its weight and `rnea(q,0,0,g)` returns the gravity torque.
pub fn rnea(
    model: &Model,
    q: &[f64],
    qd: &[f64],
    qdd: &[f64],
    gravity: &Vector3<f64>,
) -> Result<DVector<f64>, DynError> {
    check_dims(
        model,
        &[("q", q.len()), ("qd", qd.len()), ("qdd", qdd.len())],
    )?;
    let n = model.ndof;
    let mut v = vec![Vector6::<f64>::zeros(); n]; // spatial velocity, link frame
    let mut a = vec![Vector6::<f64>::zeros(); n]; // spatial acceleration, link frame
    let mut f = vec![Vector6::<f64>::zeros(); n]; // spatial force, link frame
    let a_base = Vector6::new(-gravity.x, -gravity.y, -gravity.z, 0.0, 0.0, 0.0);

    // FORWARD pass: v, a, f per link (parent[i] < i guaranteed by topo order).
    for i in 0..n {
        let xi = i_x_parent(model, i, q);
        let s = subspace(model, i);
        let (vp, ap) = match model.parent[i] {
            Some(p) => (v[p], a[p]),
            None => (Vector6::zeros(), a_base),
        };
        v[i] = xi * vp + s * qd[i];
        // a_i = X·a_parent + S·qdd + ad_{v_i}(S·qd) (Featherstone velocity-product term).
        let advs = small_adjoint(&Twist(v[i])) * (s * qd[i]);
        a[i] = xi * ap + s * qdd[i] + advs;
        // Newton–Euler: f = G·a + v×*(G·v), where the force cross v×* = −crm(v)ᵀ.
        let g = model.inertia[i].matrix();
        let adv_t = small_adjoint(&Twist(v[i])).transpose();
        f[i] = g * a[i] - adv_t * (g * v[i]);
    }

    // BACKWARD pass: τ_i = Sᵀ f_i ; propagate force to parent by Adᵀ.
    let mut tau = DVector::zeros(n);
    for i in (0..n).rev() {
        let s = subspace(model, i);
        tau[i] = s.dot(&f[i]);
        if let Some(p) = model.parent[i] {
            let xi = i_x_parent(model, i, q); // child-from-parent
            let contrib = xi.transpose() * f[i]; // Ad(child→parent)ᵀ f_child
            f[p] += contrib;
        }
    }
    Ok(tau)
}

/// Joint-space mass matrix `M(q)` (CRBA): symmetric PD, ndof×ndof, joint order.
pub fn crba(model: &Model, q: &[f64]) -> Result<DMatrix<f64>, DynError> {
    check_dims(model, &[("q", q.len())])?;
    let n = model.ndof;
    let mut ic: Vec<Matrix6<f64>> = (0..n).map(|i| *model.inertia[i].matrix()).collect();
    let xs: Vec<Matrix6<f64>> = (0..n).map(|i| i_x_parent(model, i, q)).collect();

    // backward composite roll-up: Ic_parent += Xᵀ·Ic_child·X
    for i in (0..n).rev() {
        if let Some(p) = model.parent[i] {
            let rolled = xs[i].transpose() * ic[i] * xs[i];
            ic[p] += rolled;
        }
    }

    let mut m = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        let si = subspace(model, i);
        let mut fcol = ic[i] * si; // composite force for unit qdd_i
        m[(i, i)] = si.dot(&fcol);
        let mut j = i;
        while let Some(p) = model.parent[j] {
            fcol = xs[j].transpose() * fcol; // propagate into parent frame
            j = p;
            let val = subspace(model, j).dot(&fcol);
            m[(i, j)] = val;
            m[(j, i)] = val;
        }
    }
    Ok(m)
}

/// Forward dynamics: `qdd = M(q)⁻¹ (τ − C(q,qd)qd − G(q))`.
/// `bias = rnea(q,qd,0,g) = C·qd + G` ; `M = crba(q)` ; solved by Cholesky.
pub fn forward_dynamics(
    model: &Model,
    q: &[f64],
    qd: &[f64],
    tau: &[f64],
    gravity: &Vector3<f64>,
) -> Result<DVector<f64>, DynError> {
    check_dims(
        model,
        &[("q", q.len()), ("qd", qd.len()), ("tau", tau.len())],
    )?;
    let zeros = vec![0.0; model.ndof];
    let bias = rnea(model, q, qd, &zeros, gravity)?;
    let m = crba(model, q)?;
    let rhs = DVector::from_row_slice(tau) - bias;
    let chol = m.cholesky().ok_or(DynError::NotSpd)?; // NEVER .expect — surface NotSpd
    Ok(chol.solve(&rhs))
}

/// Total gravitational potential energy `PE = −Σ mᵢ · gravity · r_com_i` (world).
/// For `gravity=[0,0,−9.81]`, `PE = Σ mᵢ·9.81·z`.
pub fn potential_energy(model: &Model, joint_world: &[Se3], gravity: &Vector3<f64>) -> f64 {
    let mut pe = 0.0;
    for (si, jw) in model.inertia.iter().zip(joint_world.iter()) {
        let m = si.mass();
        if m <= 0.0 {
            continue;
        }
        // COM offset c in the link frame, recovered from the (0,3) block −m·[c]×.
        let g = si.matrix();
        let cx = -g[(2, 4)] / m; // c.x
        let cy = -g[(0, 5)] / m; // c.y
        let cz = -g[(1, 3)] / m; // c.z
        let r = jw.0 * nalgebra::Point3::new(cx, cy, cz);
        pe += -m * gravity.dot(&r.coords);
    }
    pe
}

/// Total mass of the movable mechanism: `Σ mᵢ` over every movable link's
/// composite inertia (including any folded fixed-welded descendants). The fixed
/// base is not part of this sum (its inertia is dropped for a fixed-base model),
/// matching Pinocchio's `centerOfMass`, which excludes the universe body.
pub fn total_mass(model: &Model) -> f64 {
    model.inertia.iter().map(|si| si.mass()).sum()
}

/// Mass-weighted world center of mass `r_com = Σ mᵢ·r_com_i / Σ mᵢ` at `q`.
///
/// Reuses `fk_joints` for each movable link's world placement and the same
/// COM-offset extraction as [`potential_energy`] (recovered from the `−m·[c]×`
/// block of the spatial inertia). The fixed base is excluded, so this matches
/// Pinocchio's `centerOfMass` (which sums only the movable joint subtrees).
/// Errors [`DynError::NoInertia`] when the model lacks `<inertial>` data.
pub fn center_of_mass(model: &Model, q: &[f64]) -> Result<Vector3<f64>, DynError> {
    check_dims(model, &[("q", q.len())])?;
    let mut jw = vec![Se3::identity(); model.ndof];
    fk_joints(model, q, &mut jw);
    let mut m_total = 0.0;
    let mut weighted = Vector3::zeros();
    for (si, jw_i) in model.inertia.iter().zip(jw.iter()) {
        let m = si.mass();
        if m <= 0.0 {
            continue;
        }
        // COM offset c in the link frame, recovered from the (0,3) block −m·[c]×
        // (same indices as potential_energy).
        let g = si.matrix();
        let cx = -g[(2, 4)] / m; // c.x
        let cy = -g[(0, 5)] / m; // c.y
        let cz = -g[(1, 3)] / m; // c.z
        let r = jw_i.0 * nalgebra::Point3::new(cx, cy, cz);
        weighted += m * r.coords;
        m_total += m;
    }
    // has_inertia guaranteed a nonzero movable mass; guard anyway so a degenerate
    // all-zero-mass model returns the origin instead of a NaN division.
    if m_total <= 0.0 {
        return Ok(Vector3::zeros());
    }
    Ok(weighted / m_total)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegratorKind {
    Symplectic,
    Rk4,
}

/// A torque-driven gravity simulator (fixed-base, no contact). Semi-implicit
/// Euler by default; symplectic so a passive swing conserves energy.
pub struct Simulator {
    model: Arc<Model>,
    q: DVector<f64>,
    qd: DVector<f64>,
    q_home: DVector<f64>,
    pub gravity: Vector3<f64>,
    pub damping: DVector<f64>,
    tau_applied: DVector<f64>,
    pub integrator: IntegratorKind,
    pub h_max: f64,
    pub max_substeps: usize,
    pub qd_clamp: Option<f64>,
    t: f64,
}

impl Simulator {
    pub fn new(model: Arc<Model>) -> Result<Self, DynError> {
        if !model.has_inertia {
            return Err(DynError::NoInertia);
        }
        let n = model.ndof;
        Ok(Self {
            q: DVector::zeros(n),
            qd: DVector::zeros(n),
            q_home: DVector::zeros(n),
            gravity: GRAVITY_EARTH,
            damping: DVector::from_element(n, 0.1),
            tau_applied: DVector::zeros(n),
            integrator: IntegratorKind::Symplectic,
            h_max: 1e-3,
            max_substeps: 64,
            qd_clamp: Some(1e3),
            t: 0.0,
            model,
        })
    }
    pub fn set_torque(&mut self, tau: &[f64]) -> Result<(), DynError> {
        if tau.len() != self.model.ndof {
            return Err(DynError::Dim {
                expected: self.model.ndof,
                got: tau.len(),
            });
        }
        if !tau.iter().all(|x| x.is_finite()) {
            return Err(DynError::Diverged);
        }
        self.tau_applied = DVector::from_row_slice(tau);
        Ok(())
    }
    pub fn set_gravity(&mut self, g: Vector3<f64>) {
        self.gravity = g;
    }
    pub fn set_damping(&mut self, b: &[f64]) -> Result<(), DynError> {
        if b.len() != self.model.ndof {
            return Err(DynError::Dim {
                expected: self.model.ndof,
                got: b.len(),
            });
        }
        self.damping = DVector::from_row_slice(b);
        Ok(())
    }
    pub fn reset(&mut self) {
        self.q = self.q_home.clone();
        self.qd.fill(0.0);
        self.tau_applied.fill(0.0);
        self.t = 0.0;
    }
    pub fn reset_to(&mut self, q0: &[f64], qd0: &[f64]) -> Result<(), DynError> {
        let n = self.model.ndof;
        if q0.len() != n || qd0.len() != n {
            return Err(DynError::Dim {
                expected: n,
                got: q0.len(),
            });
        }
        self.q = DVector::from_row_slice(q0);
        self.qd = DVector::from_row_slice(qd0);
        self.q_home = self.q.clone();
        self.t = 0.0;
        Ok(())
    }
    /// Set `(q, qd)` in place WITHOUT touching `t` or the reset home pose.
    /// Additive helper for a `PhysicsSimBackend` position-teleport; unlike
    /// [`reset_to`](Self::reset_to) it preserves the simulator clock and home.
    pub fn set_state(&mut self, q: &[f64], qd: &[f64]) -> Result<(), DynError> {
        let n = self.model.ndof;
        if q.len() != n || qd.len() != n {
            return Err(DynError::Dim {
                expected: n,
                got: q.len(),
            });
        }
        if !q.iter().chain(qd.iter()).all(|x| x.is_finite()) {
            return Err(DynError::Diverged);
        }
        self.q = DVector::from_row_slice(q);
        self.qd = DVector::from_row_slice(qd);
        Ok(())
    }
    pub fn q(&self) -> &[f64] {
        self.q.as_slice()
    }
    pub fn qd(&self) -> &[f64] {
        self.qd.as_slice()
    }
    pub fn time(&self) -> f64 {
        self.t
    }
    pub fn ndof(&self) -> usize {
        self.model.ndof
    }

    pub fn total_energy(&self) -> f64 {
        let m = match crba(&self.model, self.q.as_slice()) {
            Ok(m) => m,
            Err(_) => return f64::NAN,
        };
        let ke = 0.5 * (self.qd.transpose() * &m * &self.qd)[(0, 0)];
        let mut jw = vec![Se3::identity(); self.model.ndof];
        fk_joints(&self.model, self.q.as_slice(), &mut jw);
        let pe = potential_energy(&self.model, &jw, &self.gravity);
        ke + pe
    }

    /// Clamp `|qd| ≤ qd_clamp` in place. A non-finite or negative bound is a
    /// no-op (guards against `f64::clamp(min>max)` panicking on a bad `Some(c)`).
    #[inline]
    fn apply_qd_clamp(&mut self) {
        if let Some(c) = self.qd_clamp
            && c.is_finite()
            && c >= 0.0
        {
            self.qd.iter_mut().for_each(|v| *v = v.clamp(-c, c));
        }
    }

    fn micro_step(&mut self, h: f64) -> Result<(), DynError> {
        // tau_total = tau_applied − damping ⊙ qd  (viscous, folded into FD input)
        let tau = &self.tau_applied - self.damping.component_mul(&self.qd);
        let qdd = forward_dynamics(
            &self.model,
            self.q.as_slice(),
            self.qd.as_slice(),
            tau.as_slice(),
            &self.gravity,
        )?;
        if !qdd.iter().all(|x| x.is_finite()) {
            return Err(DynError::Diverged);
        }
        self.qd += h * &qdd; // semi-implicit: velocity first
        self.apply_qd_clamp();
        let qd_new = self.qd.clone();
        self.q += h * &qd_new; // position from the NEW velocity
        if !self.q.iter().chain(self.qd.iter()).all(|x| x.is_finite()) {
            return Err(DynError::Diverged);
        }
        self.t += h;
        Ok(())
    }

    /// Advance by `dt` with internal substeps (`h ≤ h_max`, clamped to max_substeps).
    ///
    /// Rejects invalid public inputs up front (B5): a non-finite `dt`, a
    /// non-finite/non-positive `h_max`, or `max_substeps == 0` would otherwise
    /// produce a degenerate substep count or `NaN` state — all surface as
    /// [`DynError::Diverged`]. If honoring `h ≤ h_max` would need more than
    /// `max_substeps` substeps the step is rejected rather than silently
    /// coarsening `h` past `h_max` (C4).
    pub fn step(&mut self, dt: f64) -> Result<(), DynError> {
        // dt must be finite AND strictly positive: a negative dt slips past the
        // `n_ideal > max_substeps` budget check (ceil is <= 0) and integrates
        // backward in time with a negative h.
        let valid = dt.is_finite()
            && dt > 0.0
            && self.h_max.is_finite()
            && self.h_max > 0.0
            && self.max_substeps >= 1;
        if !valid {
            return Err(DynError::Diverged);
        }
        let n_ideal = (dt / self.h_max).ceil();
        if n_ideal > self.max_substeps as f64 {
            // Would require h > h_max to fit the substep budget; refuse instead
            // of silently coarsening the integration step.
            return Err(DynError::Diverged);
        }
        let n = (n_ideal as usize).clamp(1, self.max_substeps);
        let h = dt / n as f64;
        match self.integrator {
            IntegratorKind::Symplectic => {
                for _ in 0..n {
                    self.micro_step(h)?;
                }
            }
            IntegratorKind::Rk4 => {
                for _ in 0..n {
                    self.rk4_step(h)?;
                }
            }
        }
        Ok(())
    }
    /// Fixed micro-step variant for tests/determinism.
    pub fn step_n(&mut self, h: f64, n: usize) -> Result<(), DynError> {
        for _ in 0..n {
            self.micro_step(h)?;
        }
        Ok(())
    }

    // RK4 on the (q,qd) ODE with the held tau − B·qd force; validation-only.
    fn rk4_step(&mut self, h: f64) -> Result<(), DynError> {
        let accel = |q: &DVector<f64>, qd: &DVector<f64>| -> Result<DVector<f64>, DynError> {
            let tau = &self.tau_applied - self.damping.component_mul(qd);
            forward_dynamics(
                &self.model,
                q.as_slice(),
                qd.as_slice(),
                tau.as_slice(),
                &self.gravity,
            )
        };
        let (q0, v0) = (self.q.clone(), self.qd.clone());
        let a1 = accel(&q0, &v0)?;
        let (k1q, k1v) = (v0.clone(), a1);
        let a2 = accel(&(&q0 + 0.5 * h * &k1q), &(&v0 + 0.5 * h * &k1v))?;
        let (k2q, k2v) = (&v0 + 0.5 * h * &k1v, a2);
        let a3 = accel(&(&q0 + 0.5 * h * &k2q), &(&v0 + 0.5 * h * &k2v))?;
        let (k3q, k3v) = (&v0 + 0.5 * h * &k2v, a3);
        let a4 = accel(&(&q0 + h * &k3q), &(&v0 + h * &k3v))?;
        let (k4q, k4v) = (&v0 + h * &k3v, a4);
        self.q = &q0 + (h / 6.0) * (&k1q + 2.0 * &k2q + 2.0 * &k3q + &k4q);
        self.qd = &v0 + (h / 6.0) * (&k1v + 2.0 * &k2v + 2.0 * &k3v + &k4v);
        self.apply_qd_clamp(); // C3: honor qd_clamp here too, matching micro_step
        if !self.q.iter().chain(self.qd.iter()).all(|x| x.is_finite()) {
            return Err(DynError::Diverged);
        }
        self.t += h;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / ((1u64 << 53) as f64)
        }
        fn unit(&mut self) -> f64 {
            self.f() * 3.0 - 1.5
        }
    }
    fn load(name: &str) -> Model {
        Model::from_urdf(Path::new(&format!(
            "{}/../../oracle/fixtures/robots/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        )))
        .unwrap()
    }

    #[test]
    fn no_inertia_errors() {
        let m = load("toy.urdf");
        assert!(matches!(
            rnea(&m, &[0.0; 2], &[0.0; 2], &[0.0; 2], &GRAVITY_EARTH),
            Err(DynError::NoInertia)
        ));
    }
    #[test]
    fn crba_is_symmetric_and_spd() {
        for name in ["dyn_pendulum2.urdf", "showcase6.urdf"] {
            let m = load(name);
            let mut r = Rng(0xC0);
            let q: Vec<f64> = (0..m.ndof).map(|_| r.unit()).collect();
            let mm = crba(&m, &q).unwrap();
            assert!((&mm - &mm.transpose()).norm() < 1e-12);
            assert!(mm.clone().cholesky().is_some(), "{name} M not SPD");
        }
    }
    #[test]
    fn crba_column_equals_rnea_unit_accel() {
        // M[:,k] == rnea(q,0,e_k,g=0)
        let m = load("dyn_pendulum2.urdf");
        let mut r = Rng(0xC1);
        let q: Vec<f64> = (0..m.ndof).map(|_| r.unit()).collect();
        let mm = crba(&m, &q).unwrap();
        let z = vec![0.0; m.ndof];
        for k in 0..m.ndof {
            let mut e = vec![0.0; m.ndof];
            e[k] = 1.0;
            let col = rnea(&m, &q, &z, &e, &Vector3::zeros()).unwrap();
            for i in 0..m.ndof {
                assert!((mm[(i, k)] - col[i]).abs() < 1e-9, "M[{i},{k}]");
            }
        }
    }
    #[test]
    fn fd_roundtrip_recovers_qdd() {
        for name in ["dyn_pendulum2.urdf", "showcase6.urdf"] {
            for g in [GRAVITY_EARTH, Vector3::zeros()] {
                let m = load(name);
                let mut r = Rng(0xC2);
                for _ in 0..50 {
                    let q: Vec<f64> = (0..m.ndof).map(|_| r.unit()).collect();
                    let qd: Vec<f64> = (0..m.ndof).map(|_| r.unit()).collect();
                    let qdd: Vec<f64> = (0..m.ndof).map(|_| r.unit()).collect();
                    let tau = rnea(&m, &q, &qd, &qdd, &g).unwrap();
                    let qdd2 = forward_dynamics(&m, &q, &qd, tau.as_slice(), &g).unwrap();
                    let qdd_v = DVector::from_row_slice(&qdd);
                    assert!((qdd2 - qdd_v).norm() < 1e-9, "{name} g={g:?}");
                }
            }
        }
    }
    #[test]
    fn passive_swing_energy_bounded() {
        for name in ["dyn_pendulum2.urdf", "showcase6.urdf"] {
            let m = Arc::new(load(name));
            let mut s = Simulator::new(m.clone()).unwrap();
            // RK4 + no damping is the tight energy-conservation witness (the
            // default symplectic integrator only bounds energy to O(h), which for
            // a stiff 6-DOF swing is a few %); RK4 keeps a passive swing to <1e-3.
            s.integrator = IntegratorKind::Rk4;
            s.h_max = 5e-4;
            s.set_damping(&vec![0.0; m.ndof]).unwrap();
            s.qd_clamp = None;
            let q0: Vec<f64> = (0..m.ndof)
                .map(|i| if i == 1 { 1.0 } else { 0.3 })
                .collect();
            s.reset_to(&q0, &vec![0.0; m.ndof]).unwrap();
            let e0 = s.total_energy();
            let mut worst = 0.0f64;
            let mut moved = 0.0f64;
            for _ in 0..4000 {
                s.step(5e-4).unwrap();
                worst = worst.max((s.total_energy() - e0).abs() / e0.abs().max(1e-6));
                for (qi, q0i) in s.q().iter().zip(q0.iter()) {
                    moved = moved.max((qi - q0i).abs());
                }
            }
            assert!(moved > 0.1, "{name} swing didn't move");
            assert!(worst < 3e-3, "{name} energy drift {worst:e}");
        }
    }

    // B5: step must reject invalid public inputs with an Err (never panic / NaN).
    #[test]
    fn step_rejects_invalid_params() {
        let m = Arc::new(load("dyn_pendulum2.urdf"));
        // max_substeps == 0
        let mut s = Simulator::new(m.clone()).unwrap();
        s.max_substeps = 0;
        assert!(s.step(1e-3).is_err());
        // non-finite dt
        let mut s = Simulator::new(m.clone()).unwrap();
        assert!(s.step(f64::NAN).is_err());
        assert!(s.step(f64::INFINITY).is_err());
        // non-positive dt (negative would otherwise integrate backward in time)
        let mut s = Simulator::new(m.clone()).unwrap();
        assert!(s.step(-1e-3).is_err());
        assert!(s.step(0.0).is_err());
        // non-positive / non-finite h_max
        let mut s = Simulator::new(m.clone()).unwrap();
        s.h_max = 0.0;
        assert!(s.step(1e-3).is_err());
        let mut s = Simulator::new(m.clone()).unwrap();
        s.h_max = -1e-3;
        assert!(s.step(1e-3).is_err());
        let mut s = Simulator::new(m).unwrap();
        s.h_max = f64::NAN;
        assert!(s.step(1e-3).is_err());
    }

    // C4: needing more than max_substeps to honor h_max surfaces as an error
    // instead of silently coarsening h beyond h_max.
    #[test]
    fn step_surfaces_excessive_substeps() {
        let m = Arc::new(load("dyn_pendulum2.urdf"));
        let mut s = Simulator::new(m).unwrap();
        s.h_max = 1e-4;
        s.max_substeps = 2;
        assert!(s.step(1e-3).is_err()); // dt/h_max = 10 > 2 substeps
        assert!(s.step(2e-4).is_ok()); // dt/h_max = 2 fits the budget exactly
    }

    // B8: a negative qd_clamp must NOT panic (would be f64::clamp(min>max));
    // clamping is simply skipped for a non-finite/negative bound.
    #[test]
    fn qd_clamp_negative_does_not_panic() {
        let m = Arc::new(load("dyn_pendulum2.urdf"));
        let mut s = Simulator::new(m.clone()).unwrap();
        s.qd_clamp = Some(-5.0);
        s.set_torque(&vec![1.0; s.ndof()]).unwrap();
        assert!(s.step(1e-3).is_ok());
        let mut s = Simulator::new(m).unwrap();
        s.qd_clamp = Some(f64::NAN);
        s.set_torque(&vec![1.0; s.ndof()]).unwrap();
        assert!(s.step(1e-3).is_ok());
    }

    // C3: rk4_step must honor qd_clamp just like micro_step.
    #[test]
    fn rk4_honors_qd_clamp() {
        let m = Arc::new(load("dyn_pendulum2.urdf"));
        let mut s = Simulator::new(m).unwrap();
        s.integrator = IntegratorKind::Rk4;
        s.qd_clamp = Some(0.5);
        s.set_torque(&vec![100.0; s.ndof()]).unwrap();
        for _ in 0..50 {
            s.step(1e-3).unwrap();
        }
        assert!(
            s.qd().iter().all(|&v| v.abs() <= 0.5 + 1e-9),
            "rk4_step ignored qd_clamp: {:?}",
            s.qd()
        );
    }
}

/// Property-based (fuzz) tests over bounded, finite random states.
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use std::path::Path;

    fn load(name: &str) -> Model {
        Model::from_urdf(Path::new(&format!(
            "{}/../../oracle/fixtures/robots/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        )))
        .unwrap()
    }

    /// RNEA→forward-dynamics round-trip: τ = rnea(q,qd,qdd) must invert back to qdd
    /// via forward_dynamics(q,qd,τ). (rnea is affine in qdd: τ = M(q)qdd + bias.)
    fn check_roundtrip(m: &Model, q: &[f64], qd: &[f64], qdd: &[f64]) -> Result<(), TestCaseError> {
        let g = GRAVITY_EARTH;
        let tau = rnea(m, q, qd, qdd, &g).unwrap();
        let qdd2 = forward_dynamics(m, q, qd, tau.as_slice(), &g).unwrap();
        prop_assert!((qdd2 - DVector::from_row_slice(qdd)).norm() < 1e-7);
        Ok(())
    }

    /// CRBA mass matrix is symmetric and positive-definite at any configuration.
    fn check_crba(m: &Model, q: &[f64]) -> Result<(), TestCaseError> {
        let mm = crba(m, q).unwrap();
        prop_assert!((&mm - &mm.transpose()).norm() < 1e-12);
        prop_assert!(mm.cholesky().is_some(), "M not SPD");
        Ok(())
    }

    proptest! {
        /// 2-DOF pendulum: forward-dynamics inverts RNEA, M is symmetric + PD.
        #[test]
        fn fd_inverts_rnea_2dof(
            q in prop::collection::vec(-1.5f64..1.5, 2),
            qd in prop::collection::vec(-1.5f64..1.5, 2),
            qdd in prop::collection::vec(-1.5f64..1.5, 2),
        ) {
            let m = load("dyn_pendulum2.urdf");
            check_roundtrip(&m, &q, &qd, &qdd)?;
            check_crba(&m, &q)?;
        }

        /// 6-DOF arm: forward-dynamics inverts RNEA, M is symmetric + PD.
        #[test]
        fn fd_inverts_rnea_6dof(
            q in prop::collection::vec(-1.5f64..1.5, 6),
            qd in prop::collection::vec(-1.5f64..1.5, 6),
            qdd in prop::collection::vec(-1.5f64..1.5, 6),
        ) {
            let m = load("showcase6.urdf");
            check_roundtrip(&m, &q, &qd, &qdd)?;
            check_crba(&m, &q)?;
        }
    }
}
