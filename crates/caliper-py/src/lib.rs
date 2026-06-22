//! Python bindings (`import caliper`) — the scripting / analysis face.
use pyo3::prelude::*;

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

#[pymodule]
fn _caliper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", caliper::VERSION)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<Robot>()?;
    Ok(())
}
