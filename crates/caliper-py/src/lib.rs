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

    fn __repr__(&self) -> String {
        format!(
            "Robot(name='{}', ndof={})",
            self.inner.name,
            self.inner.ndof()
        )
    }
}

#[pymodule]
fn _caliper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", caliper::VERSION)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<Robot>()?;
    Ok(())
}
