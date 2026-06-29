//! Caliper — a modern, open robotics engine.
//!
//! Umbrella crate re-exporting the engine modules. The three faces
//! (Studio app, CLI, Python bindings) use this crate as their primary entry
//! point, though they also depend on individual sub-crates directly where they
//! need types this facade does not re-export.
pub use caliper_dynamics as dynamics;
pub use caliper_hal as hal;
pub use caliper_ik as ik;
pub use caliper_kinematics as kinematics;
pub use caliper_model as model;
pub use caliper_motion as motion;
pub use caliper_planning as planning;
pub use caliper_spatial as spatial;

/// Engine version (from Cargo).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod prelude {
    pub use crate::hal::{RobotBackend, SimBackend};
    pub use crate::kinematics::Jacobian;
    pub use crate::model::Robot;
    pub use crate::motion::{MotionLimits, Trajectory, move_j};
    pub use crate::spatial::Se3;
}
