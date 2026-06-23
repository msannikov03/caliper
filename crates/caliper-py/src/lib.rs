//! Python bindings (`import caliper`) — the scripting / analysis face.
use caliper::ik::{IkOpts, ik};
use caliper::kinematics::{JacFrame, Jacobian, SingularityKind, SingularityParams, jacobian};
use caliper::spatial::Se3;
use nalgebra::{DMatrix, Matrix3, UnitQuaternion, Vector3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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

    fn __repr__(&self) -> String {
        format!(
            "Robot(name='{}', ndof={})",
            self.inner.name,
            self.inner.ndof()
        )
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

#[pymodule]
fn _caliper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", caliper::VERSION)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<Robot>()?;
    Ok(())
}
