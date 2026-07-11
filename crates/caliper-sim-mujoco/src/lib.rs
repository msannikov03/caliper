//! MuJoCo contact simulation behind a clean seam.
//!
//! Two layers, both engine-side (no Studio/py surface this wave):
//!
//! 1. [`mjcf`] — generate a MINIMAL MJCF document from a caliper
//!    [`Model`](caliper_model::Model): kinematic tree, hinge/slide joints,
//!    inertials, collision geoms (primitives always, convex-hull meshes
//!    opt-in), optional ground plane, optional position actuators. Pure
//!    string work — always compiled, no MuJoCo needed.
//! 2. `MujocoSim` / `MujocoBackend` (feature `mujoco` — the names are not
//!    doc-linked because the items only exist with the feature) — a thin safe
//!    layer
//!    over `mujoco-rs` 5.0.0 (pinned; tracks MuJoCo 3.9.0 exactly) plus a
//!    [`caliper_hal::RobotBackend`] impl, so the EXISTING
//!    `ControlLoop`/`SafetyMonitor`/teleop/recording stack drives a contact
//!    sim unchanged.
//!
//! # Scope (honest)
//! - Fixed-base articulated trees of 1-dof joints (hinge/slide) only — plus
//!   free-floating primitive PROPS ([`mjcf::MjcfOptions::props`]): each is a
//!   `<freejoint>` body emitted after the robot, tracked by name and read back
//!   via `MujocoSim::prop_poses`/`body_pose` (feature `mujoco`).
//! - Collision export covers the exact primitives (box/sphere/cylinder/capsule)
//!   always. `ConvexHull` mesh colliders export as inline-vertex `<mesh>`
//!   assets when [`mjcf::MjcfOptions::export_hull_meshes`] is on (MuJoCo
//!   convex-hulls vertex-only meshes natively — no mesh files on disk); OFF by
//!   default, in which case the generator counts and reports them via
//!   [`mjcf::MjcfDocument::skipped_hull_colliders`] instead of silently
//!   dropping coverage.
//! - Torque actuation writes `qfrc_applied` directly (no actuators — the recon
//!   verdict); position actuation uses MJCF `<position>` servos. The two are
//!   mutually exclusive per built model, because a position servo keeps acting
//!   whatever you write elsewhere (see [`mjcf::Actuation`]).
//! - Determinism: MuJoCo is single-threaded per `mjData` by default and
//!   bit-deterministic for the same binary + identical integration state
//!   (warmstart included). We never opt into `mjThreadPool` and never enable
//!   noisy sensors. Cross-version/cross-platform bitwise equality is NOT
//!   guaranteed — the MuJoCo release is pinned (3.9.0).
//!
//! # Linking (feature `mujoco`)
//! `mujoco-rs` links a shared `libmujoco` it does not build. Set
//! `MUJOCO_DYNAMIC_LINK_DIR` to a directory containing `libmujoco.dylib`/`.so`
//! (see `scripts/fetch_mujoco.sh`) before `cargo build --features mujoco`, and
//! have `DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH` include it at run time.

pub mod mjcf;

#[cfg(feature = "mujoco")]
mod backend;
#[cfg(feature = "mujoco")]
mod sim;

#[cfg(feature = "mujoco")]
pub use backend::MujocoBackend;
#[cfg(feature = "mujoco")]
pub use sim::{Contact, MujocoSim};

/// `true` when this build actually links MuJoCo (feature `mujoco`).
pub const MUJOCO_ENABLED: bool = cfg!(feature = "mujoco");

/// Everything that can go wrong in this crate. (Note: no `source`-named String
/// field anywhere — `thiserror` reserves that name for error chaining.)
#[derive(thiserror::Error, Debug)]
pub enum MujocoError {
    /// The caliper model lacks usable `<inertial>` data; MuJoCo requires mass
    /// on every moving body, exactly like `caliper_dynamics::Simulator`.
    #[error("model has no usable inertia (has_inertia=false)")]
    NoInertia,
    #[error("mjcf generation: {0}")]
    Mjcf(String),
    #[error("mujoco model load failed: {0}")]
    Load(String),
    #[error("expected {expected} values, got {got}")]
    Dim { expected: usize, got: usize },
    #[error("non-finite value in {what}")]
    NonFinite { what: &'static str },
    #[error("step dt {dt} is not a positive integer multiple of the model timestep {h}")]
    BadDt { dt: f64, h: f64 },
    #[error("joint `{0}` missing from the compiled MuJoCo model")]
    MissingJoint(String),
    #[error("body `{0}` missing from the compiled MuJoCo model")]
    MissingBody(String),
    /// A raw MJCF model contains a joint kind the seam cannot map onto the
    /// flat `q`/`qd` vectors (free/ball joints — fixed-base 1-dof trees only).
    #[error("unsupported MuJoCo joint type for `{0}` (only hinge/slide are mapped)")]
    UnsupportedJoint(String),
    #[error("mujoco backend: {0}")]
    Backend(String),
}

/// Every MuJoCo failure surfaces through the HAL as a loud
/// [`caliper_hal::Error::Backend`] — never a silent success.
impl From<MujocoError> for caliper_hal::Error {
    fn from(e: MujocoError) -> Self {
        caliper_hal::Error::Backend(e.to_string())
    }
}
