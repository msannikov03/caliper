//! Python bindings (`import caliper`) — the scripting / analysis face.
use caliper::dynamics::{self, DynError, GRAVITY_EARTH};
use caliper::hal::{
    ControlLoop as EngineLoop, DatasetReader as EngineReader, DatasetSpec, Gains, HoldSetpoint,
    JointMap, LeaderFollowerSource, PhysicsSimBackend, Recorder as EngineRecorder, RobotBackend,
    SafetyConfig, SafetyMonitor as EngineMonitor, SimBackend,
};
use caliper::ik::{IkOpts, ik};
use caliper::kinematics::{JacFrame, Jacobian, SingularityKind, SingularityParams, jacobian};
use caliper::model::Model;
use caliper::motion::{
    CartesianMoveOpts, MotionLimits as EngineLimits, MotionLimitsConfig, Trajectory as EngineTraj,
    move_j, move_l,
};
use caliper::planning::reach::{ReachChecker as EngineReach, ReachConfig, ReachStatus};
use caliper::planning::{PlanError, Planner as EnginePlanner, PlannerConfig};
use caliper::spatial::Se3;
use caliper_collision::{CollisionError, CollisionModel as EngineCollision, WorldScene};
use nalgebra::{DMatrix, Matrix3, UnitQuaternion, Vector3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

/// Engine version string.
#[pyfunction]
fn version() -> &'static str {
    caliper::VERSION
}

/// A robot model loaded from URDF.
#[pyclass]
struct Robot {
    inner: caliper::model::Robot,
}

#[pymethods]
impl Robot {
    /// Load a robot from a URDF file.
    #[staticmethod]
    fn from_urdf(path: &str) -> PyResult<Self> {
        caliper::model::Robot::from_urdf(std::path::Path::new(path))
            .map(|inner| Robot { inner })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn ndof(&self) -> usize {
        self.inner.ndof()
    }

    #[getter]
    fn joint_names(&self) -> Vec<String> {
        self.inner.joint_names.clone()
    }

    /// Name of the default tip frame (the last-registered link frame).
    fn tip_frame(&self) -> String {
        let model = &self.inner.model;
        model.frame_name(model.tip_frame()).to_string()
    }

    /// Names of every queryable link frame, in registration order.
    fn frame_names(&self) -> Vec<String> {
        self.inner
            .model
            .frames
            .iter()
            .map(|f| f.name.clone())
            .collect()
    }

    /// Forward kinematics: world pose of `frame` (default = tip frame) at `q`,
    /// returned as a 4×4 row-major homogeneous matrix.
    #[pyo3(signature = (q, frame=None))]
    fn fk(&self, q: Vec<f64>, frame: Option<&str>) -> PyResult<Vec<Vec<f64>>> {
        let model = &self.inner.model;
        if q.len() != model.ndof {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "q has length {}, expected ndof={}",
                q.len(),
                model.ndof
            )));
        }
        let f = resolve_frame(model, frame)?;
        let pose = caliper::kinematics::fk_frame(model, &q, f);
        let r = pose.rotation();
        let t = pose.translation();
        Ok(vec![
            vec![r[(0, 0)], r[(0, 1)], r[(0, 2)], t[0]],
            vec![r[(1, 0)], r[(1, 1)], r[(1, 2)], t[1]],
            vec![r[(2, 0)], r[(2, 1)], r[(2, 2)], t[2]],
            vec![0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Geometric Jacobian (6×ndof, rows `[v; ω]`) of `frame` (default = tip) at
    /// `q`. `reference` is `"world"` (default, LOCAL_WORLD_ALIGNED) or `"body"`
    /// (LOCAL). Returned as a 6×N row-major matrix.
    #[pyo3(signature = (q, frame=None, reference=None))]
    fn jacobian(
        &self,
        q: Vec<f64>,
        frame: Option<&str>,
        reference: Option<&str>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let model = &self.inner.model;
        if q.len() != model.ndof {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "q has length {}, expected ndof={}",
                q.len(),
                model.ndof
            )));
        }
        let f = resolve_frame(model, frame)?;
        let jframe = match reference.unwrap_or("world") {
            "world" => caliper::kinematics::JacFrame::World,
            "body" => caliper::kinematics::JacFrame::Body,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown reference frame `{other}` (expected \"world\" or \"body\")"
                )));
            }
        };
        let (_, jac) = caliper::kinematics::jacobian(model, &q, f, jframe);
        let (rows, cols) = (jac.nrows(), jac.ncols());
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                row.push(jac[(r, c)]);
            }
            out.push(row);
        }
        Ok(out)
    }

    /// Inverse kinematics. `target` is a 4×4 COLUMN-MAJOR homogeneous matrix
    /// (outer Vec = 4 columns, each length 4; `target[col][row]`). NOTE fk()
    /// returns ROW-MAJOR — to feed fk() into ik() you must transpose. Returns a
    /// dict {success, q, residual, iters, restarts_used}. residual is the SE(3)
    /// log6 norm (mixed linear+angular), not metres.
    #[pyo3(signature = (target, seed, frame=None))]
    fn ik(
        &self,
        py: Python<'_>,
        target: Vec<Vec<f64>>,
        seed: Vec<f64>,
        frame: Option<&str>,
    ) -> PyResult<Py<PyDict>> {
        let model = &self.inner.model;
        if seed.len() != model.ndof {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "seed has length {}, expected ndof={}",
                seed.len(),
                model.ndof
            )));
        }
        if target.len() != 4 || target.iter().any(|c| c.len() != 4) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target must be a 4x4 column-major matrix (4 columns of length 4)",
            ));
        }
        finite_or_err("seed", &seed)?;
        for col in &target {
            finite_or_err("target", col)?;
        }
        let f = resolve_frame(model, frame)?;
        // column-major: target[col][row]
        let rot = Matrix3::new(
            target[0][0],
            target[1][0],
            target[2][0],
            target[0][1],
            target[1][1],
            target[2][1],
            target[0][2],
            target[1][2],
            target[2][2],
        );
        let trans = Vector3::new(target[3][0], target[3][1], target[3][2]);
        // from_matrix projects onto SO(3) (the caller-supplied basis may be slightly
        // non-orthonormal), matching the Studio se3_from_col_major path.
        let quat = UnitQuaternion::from_matrix(&rot);
        let target_se3 = Se3::from_parts(trans, quat);
        let res = ik(model, f, &target_se3, &seed, &IkOpts::default());
        let d = PyDict::new(py);
        d.set_item("success", res.success)?;
        d.set_item("q", res.q)?;
        d.set_item("residual", res.residual)?;
        d.set_item("iters", res.iters)?;
        d.set_item("restarts_used", res.restarts_used)?;
        Ok(d.into())
    }

    /// Singularity analysis at `q` (World / LOCAL_WORLD_ALIGNED). Dict with every
    /// SingularityReport field. `kind` is lowercase: "none"|"wrist"|"elbow"|"boundary".
    #[pyo3(signature = (q, frame=None))]
    fn analyze(&self, py: Python<'_>, q: Vec<f64>, frame: Option<&str>) -> PyResult<Py<PyDict>> {
        let model = &self.inner.model;
        if q.len() != model.ndof {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "q has length {}, expected ndof={}",
                q.len(),
                model.ndof
            )));
        }
        finite_or_err("q", &q)?;
        let f = resolve_frame(model, frame)?;
        let (_, jac) = jacobian(model, &q, f, JacFrame::World);
        let rep = Jacobian(jac).analyze(&SingularityParams::default());
        let d = PyDict::new(py);
        d.set_item("manipulability", rep.manipulability)?;
        d.set_item("condition_number", rep.condition_number)?;
        d.set_item("sigma_min", rep.sigma_min)?;
        d.set_item("kind", kind_str(rep.kind))?;
        d.set_item("offending_joints", rep.offending_joints)?;
        d.set_item("nullspace_basis", dmatrix_to_rows(&rep.nullspace_basis))?;
        d.set_item(
            "escape_direction",
            rep.escape_direction.iter().copied().collect::<Vec<f64>>(),
        )?;
        d.set_item("sigma", rep.sigma.to_vec())?;
        Ok(d.into())
    }

    /// Yoshikawa manipulability (∏σ) at `q` (World frame).
    #[pyo3(signature = (q, frame=None))]
    fn manipulability(&self, q: Vec<f64>, frame: Option<&str>) -> PyResult<f64> {
        let model = &self.inner.model;
        if q.len() != model.ndof {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "q has length {}, expected ndof={}",
                q.len(),
                model.ndof
            )));
        }
        finite_or_err("q", &q)?;
        let f = resolve_frame(model, frame)?;
        let (_, jac) = jacobian(model, &q, f, JacFrame::World);
        Ok(Jacobian(jac).manipulability())
    }

    /// Translational manipulability ellipsoid at the frame origin, WORLD coords.
    /// Returns (axes, radii): `axes` is 3×3 with COLUMN c = the c-th principal
    /// axis (unit eigenvector of Jv·Jvᵀ); `radii[c]` = sqrt(eigenvalue) = the
    /// singular value along that axis. Order is eigen-order (unsorted, paired).
    #[pyo3(signature = (q, frame=None))]
    fn ellipsoid(&self, q: Vec<f64>, frame: Option<&str>) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
        let model = &self.inner.model;
        if q.len() != model.ndof {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "q has length {}, expected ndof={}",
                q.len(),
                model.ndof
            )));
        }
        finite_or_err("q", &q)?;
        let f = resolve_frame(model, frame)?;
        let (_, jac) = jacobian(model, &q, f, JacFrame::World);
        let jv = jac.rows(0, 3).into_owned(); // 3 × N
        // STATIC Matrix3 core (no DMatrix→Matrix3 .into(), which is absent in 0.35).
        let a = Matrix3::from_fn(|r, c| jv.row(r).dot(&jv.row(c)));
        let eig = nalgebra::SymmetricEigen::new(a);
        let mut axes = vec![vec![0.0_f64; 3]; 3];
        let mut radii = vec![0.0_f64; 3];
        for c in 0..3 {
            let v = eig.eigenvectors.column(c);
            axes[0][c] = v[0];
            axes[1][c] = v[1];
            axes[2][c] = v[2];
            radii[c] = eig.eigenvalues[c].max(0.0).sqrt();
        }
        Ok((axes, radii))
    }

    /// Rest-to-rest jerk-limited time-synchronized joint move.
    #[pyo3(signature = (q_start, q_goal, limits=None))]
    fn move_j(
        &self,
        q_start: Vec<f64>,
        q_goal: Vec<f64>,
        limits: Option<MotionLimits>,
    ) -> PyResult<Trajectory> {
        let model = &self.inner.model;
        if q_start.len() != model.ndof || q_goal.len() != model.ndof {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "q_start/q_goal must have length ndof={}",
                model.ndof
            )));
        }
        finite_or_err("q_start", &q_start)?;
        finite_or_err("q_goal", &q_goal)?;
        let lim = motion_limits(model, limits)?;
        move_j(model, &q_start, &q_goal, &lim)
            .map(|inner| Trajectory { inner })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Cartesian straight line (MOVE_L) from q_start to a 4×4 COLUMN-MAJOR target
    /// (same convention as ik(): target[col][row]).
    #[pyo3(signature = (q_start, target, frame=None, limits=None))]
    fn move_l(
        &self,
        q_start: Vec<f64>,
        target: Vec<Vec<f64>>,
        frame: Option<&str>,
        limits: Option<MotionLimits>,
    ) -> PyResult<Trajectory> {
        let model = &self.inner.model;
        if q_start.len() != model.ndof {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "q_start length must be ndof={}",
                model.ndof
            )));
        }
        if target.len() != 4 || target.iter().any(|c| c.len() != 4) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target must be 4x4 column-major",
            ));
        }
        finite_or_err("q_start", &q_start)?;
        for col in &target {
            finite_or_err("target", col)?;
        }
        let f = resolve_frame(model, frame)?;
        let rot = Matrix3::new(
            target[0][0],
            target[1][0],
            target[2][0],
            target[0][1],
            target[1][1],
            target[2][1],
            target[0][2],
            target[1][2],
            target[2][2],
        );
        let trans = Vector3::new(target[3][0], target[3][1], target[3][2]);
        let goal = Se3::from_parts(trans, UnitQuaternion::from_matrix(&rot));
        let lim = motion_limits(model, limits)?;
        let opts = CartesianMoveOpts::defaults(lim);
        move_l(model, f, &q_start, &goal, &opts)
            .map(|inner| Trajectory { inner })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Inverse dynamics (RNEA): tau = ID(q,qd,qdd) incl. gravity + Coriolis (ndof).
    #[pyo3(signature = (q, qd, qdd, gravity=None))]
    fn rnea(
        &self,
        q: Vec<f64>,
        qd: Vec<f64>,
        qdd: Vec<f64>,
        gravity: Option<[f64; 3]>,
    ) -> PyResult<Vec<f64>> {
        let m = &self.inner.model;
        for (l, x) in [("q", &q), ("qd", &qd), ("qdd", &qdd)] {
            if x.len() != m.ndof {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{l} length {} != ndof {}",
                    x.len(),
                    m.ndof
                )));
            }
            finite_or_err(l, x)?;
        }
        dynamics::rnea(m, &q, &qd, &qdd, &grav(gravity))
            .map(|t| t.as_slice().to_vec())
            .map_err(dyn_err)
    }

    /// Joint-space mass matrix M(q) (CRBA): ndof×ndof symmetric PD, row-major.
    fn crba(&self, q: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        let m = &self.inner.model;
        if q.len() != m.ndof {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "q length {} != ndof {}",
                q.len(),
                m.ndof
            )));
        }
        finite_or_err("q", &q)?;
        dynamics::crba(m, &q)
            .map(|mm| dmatrix_to_rows(&mm))
            .map_err(dyn_err)
    }

    /// Forward dynamics: qdd = M(q)⁻¹ (tau − C(q,qd)qd − g(q)) (ndof).
    #[pyo3(signature = (q, qd, tau, gravity=None))]
    fn forward_dynamics(
        &self,
        q: Vec<f64>,
        qd: Vec<f64>,
        tau: Vec<f64>,
        gravity: Option<[f64; 3]>,
    ) -> PyResult<Vec<f64>> {
        let m = &self.inner.model;
        for (l, x) in [("q", &q), ("qd", &qd), ("tau", &tau)] {
            if x.len() != m.ndof {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{l} length {} != ndof {}",
                    x.len(),
                    m.ndof
                )));
            }
            finite_or_err(l, x)?;
        }
        dynamics::forward_dynamics(m, &q, &qd, &tau, &grav(gravity))
            .map(|a| a.as_slice().to_vec())
            .map_err(dyn_err)
    }

    /// Gravity torque only: g(q) = rnea(q,0,0,gravity) (ndof).
    #[pyo3(signature = (q, gravity=None))]
    fn gravity_torque(&self, q: Vec<f64>, gravity: Option<[f64; 3]>) -> PyResult<Vec<f64>> {
        let m = &self.inner.model;
        if q.len() != m.ndof {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "q length {} != ndof {}",
                q.len(),
                m.ndof
            )));
        }
        finite_or_err("q", &q)?;
        let z = vec![0.0; m.ndof];
        dynamics::rnea(m, &q, &z, &z, &grav(gravity))
            .map(|t| t.as_slice().to_vec())
            .map_err(dyn_err)
    }

    /// True iff the URDF carried `<inertial>` on every link (dynamics available).
    #[getter]
    fn has_inertia(&self) -> bool {
        self.inner.model.has_inertia
    }

    fn __repr__(&self) -> String {
        format!(
            "Robot(name='{}', ndof={})",
            self.inner.name,
            self.inner.ndof()
        )
    }
}

/// A planned trajectory you can sample (MATLAB-style for plotting).
#[pyclass]
struct Trajectory {
    inner: EngineTraj,
}

#[pymethods]
impl Trajectory {
    #[getter]
    fn duration(&self) -> f64 {
        self.inner.duration()
    }
    #[getter]
    fn ndof(&self) -> usize {
        self.inner.ndof()
    }
    #[getter]
    fn completed(&self) -> bool {
        self.inner.completed
    }
    #[getter]
    fn reached(&self) -> f64 {
        self.inner.reached
    }
    #[getter]
    fn vel_limit(&self) -> Vec<f64> {
        self.inner.limits().vmax.clone()
    }
    #[getter]
    fn accel_limit(&self) -> Vec<f64> {
        self.inner.limits().amax.clone()
    }
    #[getter]
    fn jerk_limit(&self) -> Vec<f64> {
        self.inner.limits().jmax.clone()
    }

    /// (q, qd, qdd) at time t (clamped to [0,duration]).
    fn sample(&self, t: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let s = self.inner.sample(t);
        (s.q, s.qd, s.qdd)
    }
    fn q_at(&self, t: f64) -> Vec<f64> {
        self.inner.q_at(t)
    }

    /// (times, q[N][ndof], qd[N][ndof], qdd[N][ndof]); times[0]=0, last=duration.
    #[allow(clippy::type_complexity)]
    fn sample_uniform(&self, dt: f64) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // clamp: dt<=0/NaN would make n overflow / allocate unbounded.
        let dt = if dt.is_finite() && dt > 1e-4 {
            dt.min(10.0)
        } else {
            1e-2
        };
        let dur = self.inner.duration();
        let n = ((dur / dt).ceil() as usize).max(1) + 1;
        let mut times = Vec::with_capacity(n);
        let (mut q, mut qd, mut qdd) = (vec![], vec![], vec![]);
        for k in 0..n {
            let t = (k as f64 * dt).min(dur);
            let s = self.inner.sample(t);
            times.push(t);
            q.push(s.q);
            qd.push(s.qd);
            qdd.push(s.qdd);
        }
        (times, q, qd, qdd)
    }
    fn __repr__(&self) -> String {
        format!(
            "Trajectory(ndof={}, duration={:.3}s, completed={})",
            self.inner.ndof(),
            self.inner.duration(),
            self.inner.completed
        )
    }
}

/// Per-joint motion limits (vel/accel/jerk) — MATLAB-friendly override.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct MotionLimits {
    inner: EngineLimits,
}

#[pymethods]
impl MotionLimits {
    #[new]
    fn new(vel: Vec<f64>, accel: Vec<f64>, jerk: Vec<f64>) -> PyResult<Self> {
        if vel.len() != accel.len() || vel.len() != jerk.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "vel/accel/jerk length mismatch",
            ));
        }
        Ok(MotionLimits {
            inner: EngineLimits {
                vmax: vel,
                amax: accel,
                jmax: jerk,
            },
        })
    }
    #[staticmethod]
    #[pyo3(signature = (robot, accel_ratio=5.0, jerk_ratio=10.0, vel_scale=1.0, default_vel=1.0))]
    fn from_robot(
        robot: &Robot,
        accel_ratio: f64,
        jerk_ratio: f64,
        vel_scale: f64,
        default_vel: f64,
    ) -> PyResult<Self> {
        let cfg = MotionLimitsConfig {
            default_vel,
            accel_ratio,
            jerk_ratio,
            vel_scale,
            overrides: vec![],
        };
        EngineLimits::from_model(&robot.inner.model, &cfg)
            .map(|inner| MotionLimits { inner })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
    #[getter]
    fn vel(&self) -> Vec<f64> {
        self.inner.vmax.clone()
    }
    #[getter]
    fn accel(&self) -> Vec<f64> {
        self.inner.amax.clone()
    }
    #[getter]
    fn jerk(&self) -> Vec<f64> {
        self.inner.jmax.clone()
    }
}

/// Resolve the limits arg: an explicit MotionLimits, or defaults from the model.
fn motion_limits(
    model: &caliper::model::Model,
    limits: Option<MotionLimits>,
) -> PyResult<EngineLimits> {
    match limits {
        Some(l) => Ok(l.inner),
        None => EngineLimits::from_model(model, &MotionLimitsConfig::default())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
    }
}

/// Resolve a frame name to its index, defaulting to the tip frame.
fn resolve_frame(model: &caliper::model::Model, frame: Option<&str>) -> PyResult<usize> {
    match frame {
        None => Ok(model.tip_frame()),
        Some(name) => model.frame_id(name).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("unknown frame `{name}`"))
        }),
    }
}

/// Lowercase kind tag for the Python dict (idiomatic snake on the Python side).
fn kind_str(k: SingularityKind) -> &'static str {
    match k {
        SingularityKind::None => "none",
        SingularityKind::Wrist => "wrist",
        SingularityKind::Elbow => "elbow",
        SingularityKind::Boundary => "boundary",
    }
}

fn grav(g: Option<[f64; 3]>) -> Vector3<f64> {
    g.map(|a| Vector3::new(a[0], a[1], a[2]))
        .unwrap_or(GRAVITY_EARTH)
}
fn dyn_err(e: DynError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}
fn hal_err(e: caliper::hal::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}
fn col_err(e: CollisionError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}
fn plan_err(e: PlanError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}
fn world_scene(ground: Option<f64>, boxes: Option<Vec<([f64; 3], [f64; 3])>>) -> WorldScene {
    let mut s = WorldScene::new();
    if let Some(z) = ground {
        s = s.with_ground(z);
    }
    for (c, h) in boxes.unwrap_or_default() {
        s = s.add_box(c, h);
    }
    s
}
/// 12 numbers (9 row-major rotation, then tx,ty,tz) → `Se3`.
fn se3_from_12(t: &[f64]) -> PyResult<Se3> {
    if t.len() != 12 || !t.iter().all(|x| x.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "target needs 12 finite values (9 row-major R then tx,ty,tz)",
        ));
    }
    let rot = Matrix3::new(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]);
    Ok(Se3::from_parts(
        Vector3::new(t[9], t[10], t[11]),
        UnitQuaternion::from_matrix(&rot),
    ))
}

/// A torque-driven gravity simulator (fixed-base, no contact).
#[pyclass]
struct Simulator {
    inner: dynamics::Simulator,
    dt: f64,
}

#[pymethods]
impl Simulator {
    #[new]
    #[pyo3(signature = (robot, dt=1e-3, gravity=None, damping=0.0, substeps=4))]
    fn new(
        robot: &Robot,
        dt: f64,
        gravity: Option<[f64; 3]>,
        damping: f64,
        substeps: usize,
    ) -> PyResult<Self> {
        if !(damping.is_finite() && damping >= 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "damping must be finite and >= 0 (negative damping injects energy)",
            ));
        }
        let model = std::sync::Arc::new(robot.inner.model.clone());
        let mut inner = dynamics::Simulator::new(model).map_err(dyn_err)?;
        inner.set_gravity(grav(gravity));
        inner
            .set_damping(&vec![damping; robot.inner.model.ndof])
            .map_err(dyn_err)?;
        inner.h_max = (dt / substeps.max(1) as f64).max(1e-6);
        Ok(Simulator {
            inner,
            dt: if dt.is_finite() && dt > 0.0 { dt } else { 1e-3 },
        })
    }
    fn step(&mut self) -> PyResult<()> {
        self.inner.step(self.dt).map_err(dyn_err)
    }
    fn step_n(&mut self, n: usize) -> PyResult<()> {
        for _ in 0..n {
            self.inner.step(self.dt).map_err(dyn_err)?;
        }
        Ok(())
    }
    #[getter]
    fn q(&self) -> Vec<f64> {
        self.inner.q().to_vec()
    }
    #[getter]
    fn qd(&self) -> Vec<f64> {
        self.inner.qd().to_vec()
    }
    #[getter]
    fn time(&self) -> f64 {
        self.inner.time()
    }
    #[getter]
    fn energy(&self) -> f64 {
        self.inner.total_energy()
    }
    fn set_torque(&mut self, tau: Vec<f64>) -> PyResult<()> {
        finite_or_err("tau", &tau)?;
        self.inner.set_torque(&tau).map_err(dyn_err)
    }
    fn set_gravity(&mut self, g: [f64; 3]) {
        self.inner.set_gravity(Vector3::new(g[0], g[1], g[2]));
    }
    fn set_damping(&mut self, d: Vec<f64>) -> PyResult<()> {
        if d.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "damping must be finite and >= 0 (negative damping injects energy)",
            ));
        }
        self.inner.set_damping(&d).map_err(dyn_err)
    }
    #[pyo3(signature = (q0=None, qd0=None))]
    fn reset(&mut self, q0: Option<Vec<f64>>, qd0: Option<Vec<f64>>) -> PyResult<()> {
        match (q0, qd0) {
            (None, None) => {
                self.inner.reset();
                Ok(())
            }
            (q, v) => {
                let n = self.inner.ndof();
                let q = q.unwrap_or_else(|| vec![0.0; n]);
                let v = v.unwrap_or_else(|| vec![0.0; n]);
                finite_or_err("q0", &q)?;
                finite_or_err("qd0", &v)?;
                self.inner.reset_to(&q, &v).map_err(dyn_err)
            }
        }
    }
    /// Bake a rollout: (times[N], q[N][ndof], qd[N][ndof]) over `horizon`.
    #[pyo3(signature = (horizon, sample_dt=None))]
    #[allow(clippy::type_complexity)]
    fn rollout(
        &mut self,
        horizon: f64,
        sample_dt: Option<f64>,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        if !(horizon.is_finite() && horizon > 0.0 && horizon <= 600.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "horizon must be finite, > 0, and <= 600 s",
            ));
        }
        let sdt = sample_dt.unwrap_or(self.dt).max(1e-4);
        let n = ((horizon / sdt).ceil() as usize).max(1);
        let (mut ts, mut qs, mut qds) = (vec![], vec![], vec![]);
        ts.push(self.inner.time());
        qs.push(self.inner.q().to_vec());
        qds.push(self.inner.qd().to_vec());
        for _ in 0..n {
            self.inner.step(sdt).map_err(dyn_err)?;
            ts.push(self.inner.time());
            qs.push(self.inner.q().to_vec());
            qds.push(self.inner.qd().to_vec());
        }
        Ok((ts, qs, qds))
    }
    fn __repr__(&self) -> String {
        let qdmax = self.inner.qd().iter().fold(0.0f64, |a, &x| a.max(x.abs()));
        format!(
            "Simulator(ndof={}, t={:.3}s, |qd|max={:.3})",
            self.inner.ndof(),
            self.inner.time(),
            qdmax
        )
    }
}

/// Reject non-finite inputs (NaN/Inf) before they reach nalgebra's SVD, which
/// does not terminate on a non-finite matrix.
fn finite_or_err(label: &str, xs: &[f64]) -> PyResult<()> {
    if xs.iter().any(|x| !x.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label} contains a non-finite value (NaN/Inf)"
        )));
    }
    Ok(())
}

fn dmatrix_to_rows(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|r| (0..m.ncols()).map(|c| m[(r, c)]).collect())
        .collect()
}

// ===== Phase 5: control loop, dataset, collision, safety, teleop =====

/// A deterministic computed-torque control loop over a physical sim backend.
#[pyclass]
struct ControlLoop {
    inner: EngineLoop<PhysicsSimBackend>,
    ndof: usize,
}

#[pymethods]
impl ControlLoop {
    #[new]
    #[pyo3(signature = (robot, dt=1e-3, kp=100.0, kd=20.0, gravity=None, start=None))]
    fn new(
        robot: &Robot,
        dt: f64,
        kp: f64,
        kd: f64,
        gravity: Option<[f64; 3]>,
        start: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let m = &robot.inner.model;
        if !m.has_inertia {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "robot has no <inertial> data; control needs dynamics",
            ));
        }
        let model = Arc::new(m.clone());
        let mut backend = PhysicsSimBackend::new(model.clone()).map_err(hal_err)?;
        if let Some(q0) = start {
            if q0.len() != m.ndof {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "start length {} != ndof {}",
                    q0.len(),
                    m.ndof
                )));
            }
            finite_or_err("start", &q0)?;
            backend
                .set_state(&q0, &vec![0.0; m.ndof])
                .map_err(hal_err)?;
        }
        let mut inner = EngineLoop::new(backend, model, dt)
            .map_err(hal_err)?
            .with_gains(Gains { kp, kd });
        if let Some(g) = gravity {
            inner = inner.with_gravity(grav(Some(g)));
        }
        Ok(ControlLoop {
            inner,
            ndof: m.ndof,
        })
    }

    /// Regulate to `goal` for `ticks` steps (no recording).
    fn run_to(&mut self, goal: Vec<f64>, ticks: usize) -> PyResult<()> {
        self.check_goal(&goal)?;
        let mut sp = HoldSetpoint::new(goal);
        self.inner.run_to(&mut sp, ticks).map_err(hal_err)
    }

    /// Regulate to `goal`, recording each tick. Returns (times, states, actions),
    /// where `states` = measured q (observation.state) and `actions` = commanded q.
    #[allow(clippy::type_complexity)]
    fn rollout_to(
        &mut self,
        goal: Vec<f64>,
        ticks: usize,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        self.check_goal(&goal)?;
        let mut sp = HoldSetpoint::new(goal);
        let frames = self.inner.run_record(&mut sp, ticks).map_err(hal_err)?;
        let times = frames.iter().map(|f| f.t).collect();
        let states = frames.iter().map(|f| f.measured.clone()).collect();
        let actions = frames.iter().map(|f| f.command.clone()).collect();
        Ok((times, states, actions))
    }

    /// Latch an e-stop (loop + backend).
    fn estop(&mut self) -> PyResult<()> {
        self.inner.estop().map_err(hal_err)
    }

    #[getter]
    fn q(&self) -> Vec<f64> {
        self.inner.backend().joint_positions()
    }
    #[getter]
    fn qd(&mut self) -> PyResult<Vec<f64>> {
        Ok(self
            .inner
            .backend_mut()
            .read_state()
            .map_err(hal_err)?
            .qd_or_zero())
    }
    #[getter]
    fn time(&self) -> f64 {
        self.inner.time()
    }
    #[getter]
    fn tick(&self) -> u64 {
        self.inner.tick()
    }
    fn __repr__(&self) -> String {
        format!(
            "ControlLoop(ndof={}, t={:.3}s)",
            self.ndof,
            self.inner.time()
        )
    }
}

impl ControlLoop {
    fn check_goal(&self, goal: &[f64]) -> PyResult<()> {
        if goal.len() != self.ndof {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "goal length {} != ndof {}",
                goal.len(),
                self.ndof
            )));
        }
        finite_or_err("goal", goal)
    }
}

/// Writes a LeRobotDataset v2.1 episode to disk.
#[pyclass]
struct Recorder {
    inner: Option<EngineRecorder>,
}

#[pymethods]
impl Recorder {
    #[new]
    #[pyo3(signature = (robot, out, fps=30))]
    fn new(robot: &Robot, out: &str, fps: u32) -> PyResult<Self> {
        let spec = DatasetSpec::from_model(&robot.inner.model, fps);
        let inner = EngineRecorder::create(out, spec).map_err(hal_err)?;
        Ok(Recorder { inner: Some(inner) })
    }
    fn start_episode(&mut self, task: &str) -> PyResult<()> {
        self.get()?.start_episode(task).map_err(hal_err)
    }
    fn append(&mut self, state: Vec<f64>, action: Vec<f64>, t: f64) -> PyResult<()> {
        finite_or_err("state", &state)?;
        finite_or_err("action", &action)?;
        self.get()?
            .append_frame(&state, &action, t)
            .map_err(hal_err)
    }
    fn finalize_episode(&mut self) -> PyResult<()> {
        self.get()?.finalize_episode().map_err(hal_err)
    }
    /// Finalize the dataset (writes meta/) and return its path. Consumes the recorder.
    fn close(&mut self) -> PyResult<String> {
        let rec = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("recorder already closed"))?;
        rec.close()
            .map(|p| p.display().to_string())
            .map_err(hal_err)
    }
}

impl Recorder {
    fn get(&mut self) -> PyResult<&mut EngineRecorder> {
        self.inner
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("recorder is closed"))
    }
}

/// Reads a LeRobotDataset v2.1 from disk.
#[pyclass]
struct DatasetReader {
    inner: EngineReader,
}

#[pymethods]
impl DatasetReader {
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        EngineReader::open(path)
            .map(|inner| DatasetReader { inner })
            .map_err(hal_err)
    }
    #[getter]
    fn total_episodes(&self) -> usize {
        self.inner.total_episodes
    }
    #[getter]
    fn ndof(&self) -> usize {
        self.inner.ndof
    }
    #[getter]
    fn fps(&self) -> u32 {
        self.inner.fps
    }
    /// Read an episode → (states, actions, timestamps).
    #[allow(clippy::type_complexity)]
    fn read_episode(&self, episode: usize) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>)> {
        let e = self.inner.read_episode(episode).map_err(hal_err)?;
        Ok((e.states, e.actions, e.timestamps))
    }
}

/// Configuration-space collision checker (self + world).
#[pyclass]
struct CollisionModel {
    inner: EngineCollision,
}

#[pymethods]
impl CollisionModel {
    #[new]
    #[pyo3(signature = (robot, ground=None, boxes=None, margin=0.0))]
    fn new(
        robot: &Robot,
        ground: Option<f64>,
        boxes: Option<Vec<([f64; 3], [f64; 3])>>,
        margin: f64,
    ) -> PyResult<Self> {
        let model = Arc::new(robot.inner.model.clone());
        let mut scene = WorldScene::new();
        if let Some(z) = ground {
            scene = scene.with_ground(z);
        }
        for (c, h) in boxes.unwrap_or_default() {
            scene = scene.add_box(c, h);
        }
        Ok(CollisionModel {
            inner: EngineCollision::new(model, scene, margin),
        })
    }
    #[getter]
    fn num_colliders(&self) -> usize {
        self.inner.num_colliders()
    }
    #[getter]
    fn uncovered_frames(&self) -> usize {
        self.inner.uncovered_frames()
    }
    /// Query at `q` → dict(collision, self_pairs, world_hits, colliding_frames).
    fn query(&self, py: Python<'_>, q: Vec<f64>) -> PyResult<Py<PyDict>> {
        finite_or_err("q", &q)?;
        let r = self.inner.query(&q).map_err(col_err)?;
        let d = PyDict::new(py);
        d.set_item("collision", r.has_collision())?;
        let pairs: Vec<(usize, usize)> = r.self_pairs.clone();
        d.set_item("self_pairs", pairs)?;
        d.set_item("world_hits", r.world_hits.clone())?;
        d.set_item("colliding_frames", r.colliding_frames.clone())?;
        Ok(d.into())
    }
}

/// The pure safety monitor: position clamp, velocity rate-limit, e-stop latch.
#[pyclass]
struct SafetyMonitor {
    inner: EngineMonitor,
    dt: f64,
}

#[pymethods]
impl SafetyMonitor {
    #[new]
    #[pyo3(signature = (robot, q0, dt=1e-3))]
    fn new(robot: &Robot, q0: Vec<f64>, dt: f64) -> PyResult<Self> {
        if q0.len() != robot.inner.model.ndof {
            return Err(pyo3::exceptions::PyValueError::new_err("q0 length != ndof"));
        }
        finite_or_err("q0", &q0)?;
        let cfg = SafetyConfig::from_model(&robot.inner.model);
        Ok(SafetyMonitor {
            inner: EngineMonitor::new(cfg, &q0),
            dt,
        })
    }
    /// Sanitize a desired target → (safe_q, dict(clamped_position, limited_velocity, estopped, ok)).
    fn gate(&mut self, py: Python<'_>, desired: Vec<f64>) -> PyResult<(Vec<f64>, Py<PyDict>)> {
        finite_or_err("desired", &desired)?;
        let (safe, v) = self.inner.gate(&desired, self.dt);
        let d = PyDict::new(py);
        d.set_item("clamped_position", v.clamped_position)?;
        d.set_item("limited_velocity", v.limited_velocity)?;
        d.set_item("estopped", v.estopped)?;
        d.set_item("ok", v.ok())?;
        Ok((safe, d.into()))
    }
    fn estop(&mut self) {
        self.inner.estop();
    }
    fn clear_estop(&mut self) {
        self.inner.clear_estop();
    }
    #[getter]
    fn is_estopped(&self) -> bool {
        self.inner.is_estopped()
    }
    #[getter]
    fn warn_count(&self) -> u64 {
        self.inner.warn_count
    }
}

/// Leader-follower teleop in pure sim: a follower control loop tracks a leader.
/// `unsendable` because it holds a `Box<dyn RobotBackend>` (Send but not Sync);
/// the object is GIL-bound, which is fine for a single-threaded teleop demo.
#[pyclass(unsendable)]
struct LeaderFollower {
    loopy: EngineLoop<SimBackend>,
    src: LeaderFollowerSource,
    ndof: usize,
}

#[pymethods]
impl LeaderFollower {
    #[new]
    #[pyo3(signature = (robot, dt=1e-3))]
    fn new(robot: &Robot, dt: f64) -> PyResult<Self> {
        let n = robot.inner.model.ndof;
        let model = Arc::new(robot.inner.model.clone());
        let follower = SimBackend::new(n);
        let leader = Box::new(SimBackend::new(n));
        let src = LeaderFollowerSource::new(leader, JointMap::identity(n));
        let loopy = EngineLoop::new(follower, model, dt).map_err(hal_err)?;
        Ok(LeaderFollower {
            loopy,
            src,
            ndof: n,
        })
    }
    /// Move the leader to `lead`, step the follower once, return the follower q.
    fn step(&mut self, lead: Vec<f64>) -> PyResult<Vec<f64>> {
        if lead.len() != self.ndof {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "lead length != ndof",
            ));
        }
        finite_or_err("lead", &lead)?;
        self.src
            .leader_mut()
            .command_joint_positions(&lead)
            .map_err(hal_err)?;
        self.loopy.step(&mut self.src, None).map_err(hal_err)?;
        Ok(self.loopy.backend().joint_positions())
    }
}

/// Collision-aware RRT-Connect motion planner (Phase 6).
#[pyclass]
struct Planner {
    inner: EnginePlanner,
    model: Arc<Model>,
}

#[pymethods]
impl Planner {
    #[new]
    #[pyo3(signature = (robot, ground=None, boxes=None, seed=0xCA11, step=0.3, margin=0.0))]
    fn new(
        robot: &Robot,
        ground: Option<f64>,
        boxes: Option<Vec<([f64; 3], [f64; 3])>>,
        seed: u64,
        step: f64,
        margin: f64,
    ) -> PyResult<Self> {
        let model = Arc::new(robot.inner.model.clone());
        let scene = world_scene(ground, boxes);
        let cfg = PlannerConfig {
            seed,
            step,
            margin,
            ..PlannerConfig::default()
        };
        Ok(Planner {
            inner: EnginePlanner::new(model.clone(), scene, cfg),
            model,
        })
    }
    #[getter]
    fn uncovered_frames(&self) -> usize {
        self.inner.uncovered_frames()
    }
    /// Plan a collision-free waypoint path to a joint goal.
    fn plan(&self, start: Vec<f64>, goal: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        finite_or_err("start", &start)?;
        finite_or_err("goal", &goal)?;
        self.inner.plan(&start, &goal).map_err(plan_err)
    }
    /// Plan to a Cartesian goal pose (12 numbers); `frame` index defaults to tip.
    #[pyo3(signature = (start, target, frame=None))]
    fn plan_to_pose(
        &self,
        start: Vec<f64>,
        target: Vec<f64>,
        frame: Option<usize>,
    ) -> PyResult<Vec<Vec<f64>>> {
        finite_or_err("start", &start)?;
        let se3 = se3_from_12(&target)?;
        let f = frame.unwrap_or_else(|| self.model.tip_frame());
        self.inner
            .plan_to_pose(&start, &se3, f, &start)
            .map_err(plan_err)
    }
    /// Plan + retime → (times, q[N][ndof], qd[N][ndof]).
    #[pyo3(signature = (start, goal, dt=0.02))]
    #[allow(clippy::type_complexity)]
    fn plan_trajectory(
        &self,
        start: Vec<f64>,
        goal: Vec<f64>,
        dt: f64,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        finite_or_err("start", &start)?;
        finite_or_err("goal", &goal)?;
        let limits = EngineLimits::from_model(&self.model, &MotionLimitsConfig::default())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let traj = self
            .inner
            .plan_trajectory(&start, &goal, &limits, dt)
            .map_err(plan_err)?;
        let dur = traj.duration();
        let n = ((dur / dt).ceil() as usize).max(1);
        let (mut ts, mut qs, mut qds) = (vec![], vec![], vec![]);
        for k in 0..=n {
            let t = (k as f64 * dt).min(dur);
            let s = traj.sample(t);
            ts.push(t);
            qs.push(s.q);
            qds.push(s.qd);
        }
        Ok((ts, qs, qds))
    }
    /// Independently re-verify a path is collision-free (finer resolution).
    fn verify(&self, path: Vec<Vec<f64>>) -> bool {
        self.inner.verify_path(&path)
    }
}

/// Collision-aware reachability checker (Phase 6).
#[pyclass]
struct ReachChecker {
    inner: EngineReach,
}

#[pymethods]
impl ReachChecker {
    #[new]
    #[pyo3(signature = (robot, ground=None, boxes=None, frame=None, seeds=8))]
    fn new(
        robot: &Robot,
        ground: Option<f64>,
        boxes: Option<Vec<([f64; 3], [f64; 3])>>,
        frame: Option<usize>,
        seeds: usize,
    ) -> PyResult<Self> {
        let model = Arc::new(robot.inner.model.clone());
        let scene = world_scene(ground, boxes);
        let cfg = ReachConfig {
            frame,
            seeds,
            ..ReachConfig::default()
        };
        Ok(ReachChecker {
            inner: EngineReach::new(model, scene, cfg),
        })
    }
    /// Reachability of a Cartesian pose (12 numbers) →
    /// dict(status: "reachable"|"blocked"|"unreachable", residual, q).
    fn status(&self, py: Python<'_>, target: Vec<f64>) -> PyResult<Py<PyDict>> {
        let se3 = se3_from_12(&target)?;
        let v = self.inner.status(&se3);
        let d = PyDict::new(py);
        d.set_item(
            "status",
            match v.status {
                ReachStatus::Reachable => "reachable",
                ReachStatus::Blocked => "blocked",
                ReachStatus::Unreachable => "unreachable",
            },
        )?;
        d.set_item("residual", v.residual)?;
        match v.q {
            Some(q) => d.set_item("q", q)?,
            None => d.set_item("q", py.None())?,
        }
        Ok(d.into())
    }
    fn reachable(&self, target: Vec<f64>) -> PyResult<bool> {
        Ok(self.inner.reachable(&se3_from_12(&target)?))
    }
}

#[pymodule]
fn _caliper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", caliper::VERSION)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<Robot>()?;
    m.add_class::<Trajectory>()?;
    m.add_class::<MotionLimits>()?;
    m.add_class::<Simulator>()?;
    m.add_class::<ControlLoop>()?;
    m.add_class::<Recorder>()?;
    m.add_class::<DatasetReader>()?;
    m.add_class::<CollisionModel>()?;
    m.add_class::<SafetyMonitor>()?;
    m.add_class::<LeaderFollower>()?;
    m.add_class::<Planner>()?;
    m.add_class::<ReachChecker>()?;
    Ok(())
}
