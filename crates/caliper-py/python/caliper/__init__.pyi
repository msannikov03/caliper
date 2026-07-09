"""Type stubs for the Caliper Python bindings (PEP 561).

Hand-written to mirror `crates/caliper-py/src/lib.rs` exactly — every
`#[pyclass]`, `#[pymethods]` and `#[pyfunction]` with its real Python
signature and return type. Do not add methods that are not in `lib.rs`.

Conventions carried over from the Rust side:
  * FK poses are returned as 4x4 ROW-MAJOR homogeneous matrices.
  * EVERY pose-accepting method (`ik()` / `analytic_ik()` / `move_l()` /
    `move_c()` / `Planner.plan_to_pose()` / `ReachChecker.status()` /
    `reachable()` / `calibrate_joint_offsets()`) takes the SAME `_Pose`:
    a 4x4 COLUMN-MAJOR nested matrix (`pose[col][row]`) or its flat
    16-element column-major equivalent.
  * `Planner.plan_to_pose()` additionally grandfathers the LEGACY flat
    12-element row-major form (9 row-major R, then tx, ty, tz).
  * Frame arguments take a frame NAME; `Planner.plan_to_pose()` and
    `ReachChecker` also still accept a raw frame index (back-compat).
"""

import os
from typing import Any, Callable, Optional, Sequence, Union

__version__: str

# A joint-space / flat float vector accepted at the FFI boundary (any sequence).
_Vec = Sequence[float]
# A gravity 3-vector, or a box (center, half-extents) spec.
_Vec3 = Sequence[float]
# World-scene axis-aligned boxes: (center, half_extents) pairs.
_Boxes = Sequence[tuple[_Vec3, _Vec3]]
# A Cartesian pose: 4x4 COLUMN-MAJOR nested (`pose[col][row]`) or flat-16
# column-major — the ONE convention shared by every pose-accepting method.
_Pose = Union[Sequence[Sequence[float]], Sequence[float]]
# A frame argument: a NAME (unified), or a raw index where back-compat allows.
_Frame = Union[str, int]

def version() -> str:
    """Engine version string."""
    ...

def run_graph(robot: "Robot", graph_json: str) -> dict[str, Any]:
    """Execute a Caliper compute graph against `robot`.

    Returns a dict with `terminal_clip` ({times, qs, qds} or None),
    `scopes` (list of {node_id, signal, t, y}), and `diagnostics`.
    """
    ...

def validate_graph(robot: "Robot", graph_json: str) -> dict[str, Any]:
    """Validate a compute graph against `robot` without executing it.

    Returns the diagnostics dict ({ok, topo_order, cycle, node_errors,
    edge_errors}).
    """
    ...

def log6(m: Sequence[Sequence[float]]) -> list[float]:
    """SE(3) log map of a 4x4 ROW-MAJOR homogeneous transform → twist `[v; w]`."""
    ...

def exp6(twist: _Vec) -> list[list[float]]:
    """SE(3) exp map of a length-6 screw `[v; w]` → 4x4 ROW-MAJOR transform."""
    ...

def calibrate_joint_offsets(
    robot: "Robot",
    observations: Sequence[tuple[_Vec, _Pose]],
    frame: Optional[str] = ...,
    max_iters: int = ...,
    lambda_: float = ...,  # exposed to Python as the positional arg `lambda`
    tol_step: float = ...,
    tol_residual: float = ...,
) -> dict[str, Any]:
    """Estimate per-joint zero offsets so that `FK(q_k + delta) ~= T_k`.

    `observations` is a list of `(q, pose)` pairs; `pose` is a 4x4
    COLUMN-MAJOR homogeneous matrix (`pose[col][row]`, or flat-16). Returns
    `{offsets, rms_residual, iters, converged}`.
    """
    ...

# --- Pure-Python interop exporters, re-exported from `caliper/interop.py` —
# --- NOT from `lib.rs` (the "mirror lib.rs exactly" rule covers Rust only).

def export_lerobot_calibration(
    offsets: Sequence[float],
    motor_names: Sequence[str],
    path: Optional[Union[str, os.PathLike[str]]] = ...,
    *,
    resolution: int = ...,
    bus: str = ...,
) -> dict[str, dict[str, int]]:
    """caliper-calib joint offsets → lerobot's per-robot-id calibration JSON.

    Returns `{motor_name: {id, drive_mode, homing_offset, range_min,
    range_max}}` (lerobot `MotorCalibration` fields, all ints); also written
    to `path` as indent-4 JSON when given.
    """
    ...

def export_robomimic_hdf5(
    dataset_root: Union[str, os.PathLike[str]],
    out_path: Union[str, os.PathLike[str]],
) -> dict[str, Any]:
    """Caliper-recorded LeRobotDataset v2.1 root → robomimic HDF5.

    Writes `data/demo_N/{obs/...,actions,rewards,dones}` + `total`/`env_args`
    attrs. Needs `h5py` (not a caliper dependency). Returns
    `{out_path, demos, total}`.
    """
    ...

class Robot:
    """A robot model loaded from URDF."""

    @staticmethod
    def from_urdf(path: str) -> "Robot":
        """Load a robot from a URDF file."""
        ...

    @property
    def name(self) -> str: ...
    @property
    def ndof(self) -> int: ...
    @property
    def joint_names(self) -> list[str]: ...
    @property
    def joint_limits(self) -> list[Optional[tuple[float, float]]]:
        """Per-joint `(lo, hi)`; `None` for an unbounded (continuous) joint."""
        ...
    @property
    def has_inertia(self) -> bool:
        """True iff every link carried `<inertial>` (dynamics available)."""
        ...

    def tip_frame(self) -> str:
        """Name of the default tip frame (the last-registered link frame)."""
        ...

    def frame_names(self) -> list[str]:
        """Names of every queryable link frame, in registration order."""
        ...

    def fk(self, q: _Vec, frame: Optional[str] = ...) -> list[list[float]]:
        """Forward kinematics → 4x4 ROW-MAJOR homogeneous matrix."""
        ...

    def jacobian(
        self,
        q: _Vec,
        frame: Optional[str] = ...,
        reference: Optional[str] = ...,
    ) -> list[list[float]]:
        """Geometric Jacobian (6xN, rows `[v; w]`). `reference` = "world"|"body"."""
        ...

    def ik(
        self,
        target: _Pose,
        seed: _Vec,
        frame: Optional[str] = ...,
    ) -> dict[str, Any]:
        """Numeric IK against a 4x4 COLUMN-MAJOR target (nested or flat-16).

        Returns `{success, q, residual, iters, restarts_used}`.
        """
        ...

    def analytic_ik(
        self,
        target: _Pose,
        seed: Optional[_Vec] = ...,
        frame: Optional[str] = ...,
    ) -> Optional[list[list[float]]]:
        """Closed-form IK for a spherical-wrist 6R arm.

        `None` when the model is not a recognised spherical-wrist 6R; otherwise
        the list of branch configs (seed-nearest first), or `[]` when
        recognised-but-unreachable.
        """
        ...

    def analyze(self, q: _Vec, frame: Optional[str] = ...) -> dict[str, Any]:
        """Singularity analysis at `q` (World frame).

        Dict with manipulability, condition_number, sigma_min, kind
        ("none"|"wrist"|"elbow"|"boundary"), offending_joints,
        nullspace_basis, escape_direction, sigma.
        """
        ...

    def manipulability(self, q: _Vec, frame: Optional[str] = ...) -> float:
        """Yoshikawa manipulability at `q` (World frame)."""
        ...

    def ellipsoid(
        self, q: _Vec, frame: Optional[str] = ...
    ) -> tuple[list[list[float]], list[float]]:
        """Translational manipulability ellipsoid → (axes 3x3, radii)."""
        ...

    def move_j(
        self,
        q_start: _Vec,
        q_goal: _Vec,
        limits: Optional["MotionLimits"] = ...,
    ) -> "Trajectory":
        """Rest-to-rest jerk-limited time-synchronized joint move."""
        ...

    def move_l(
        self,
        q_start: _Vec,
        target: _Pose,
        frame: Optional[str] = ...,
        limits: Optional["MotionLimits"] = ...,
    ) -> "Trajectory":
        """Cartesian straight line (MOVE_L) to a 4x4 COLUMN-MAJOR target
        (nested or flat-16)."""
        ...

    def move_c(
        self,
        q_start: _Vec,
        via: _Vec,
        target: _Pose,
        frame: Optional[str] = ...,
        limits: Optional["MotionLimits"] = ...,
    ) -> "Trajectory":
        """Cartesian circular arc (MOVE_C) THROUGH `via` ([x, y, z], meters)
        to a 4x4 COLUMN-MAJOR end pose (nested or flat-16); the short sweep
        passes the via en route to the end."""
        ...

    def retime_time_optimal(
        self,
        waypoints: Sequence[_Vec],
        vmax: Optional[_Vec] = ...,
        amax: Optional[_Vec] = ...,
        dt: float = ...,
    ) -> "Trajectory":
        """Time-optimal (acceleration-limited, corner-stop bang-bang TOPP)
        retiming of a waypoint path (each row length = ndof).

        `vmax`/`amax` are per-joint bounds (pass both or neither; default =
        model limits). Jerk is NOT limited — `jerk_limit` reports inf.
        """
        ...

    def resolved_rate(
        self, q: _Vec, v: _Vec, frame: Optional[str] = ...
    ) -> list[float]:
        """Joint velocities realizing the desired end-effector spatial velocity
        `v` (length-6 `[v; w]`, world-aligned) at `q`, via the
        manipulability-gated damped pseudo-inverse."""
        ...

    def nullspace_step(
        self, q: _Vec, v: _Vec, z: _Vec, frame: Optional[str] = ...
    ) -> list[float]:
        """Resolved-rate with a null-space secondary objective:
        `qd = J+ v + (I - J+ J) z`. Only the component of `z` (length ndof)
        inside ker(J) is applied, leaving the end-effector velocity unchanged."""
        ...

    def rnea(
        self,
        q: _Vec,
        qd: _Vec,
        qdd: _Vec,
        gravity: Optional[_Vec3] = ...,
    ) -> list[float]:
        """Inverse dynamics (RNEA): tau = ID(q, qd, qdd)."""
        ...

    def crba(self, q: _Vec) -> list[list[float]]:
        """Joint-space mass matrix M(q) (CRBA), ndof x ndof, row-major."""
        ...

    def forward_dynamics(
        self,
        q: _Vec,
        qd: _Vec,
        tau: _Vec,
        gravity: Optional[_Vec3] = ...,
    ) -> list[float]:
        """Forward dynamics: qdd = M(q)^-1 (tau - C qd - g)."""
        ...

    def gravity_torque(
        self, q: _Vec, gravity: Optional[_Vec3] = ...
    ) -> list[float]:
        """Gravity torque only: g(q)."""
        ...

    def __repr__(self) -> str: ...

class Trajectory:
    """A planned trajectory you can sample."""

    @property
    def duration(self) -> float: ...
    @property
    def ndof(self) -> int: ...
    @property
    def completed(self) -> bool: ...
    @property
    def reached(self) -> float: ...
    @property
    def vel_limit(self) -> list[float]: ...
    @property
    def accel_limit(self) -> list[float]: ...
    @property
    def jerk_limit(self) -> list[float]: ...

    def sample(
        self, t: float
    ) -> tuple[list[float], list[float], list[float]]:
        """(q, qd, qdd) at time t (clamped to [0, duration])."""
        ...

    def q_at(self, t: float) -> list[float]: ...

    def sample_uniform(
        self, dt: float
    ) -> tuple[list[float], list[list[float]], list[list[float]], list[list[float]]]:
        """(times, q, qd, qdd) sampled uniformly at `dt`."""
        ...

    def __repr__(self) -> str: ...

class MotionLimits:
    """Per-joint motion limits (vel/accel/jerk)."""

    def __init__(self, vel: _Vec, accel: _Vec, jerk: _Vec) -> None: ...

    @staticmethod
    def from_robot(
        robot: "Robot",
        accel_ratio: float = ...,
        jerk_ratio: float = ...,
        vel_scale: float = ...,
        default_vel: float = ...,
    ) -> "MotionLimits": ...

    @property
    def vel(self) -> list[float]: ...
    @property
    def accel(self) -> list[float]: ...
    @property
    def jerk(self) -> list[float]: ...

class Simulator:
    """A torque-driven gravity simulator (fixed-base, no contact)."""

    def __init__(
        self,
        robot: "Robot",
        dt: float = ...,
        gravity: Optional[_Vec3] = ...,
        damping: float = ...,
        substeps: int = ...,
    ) -> None: ...

    def step(self) -> None: ...
    def step_n(self, n: int) -> None: ...

    @property
    def q(self) -> list[float]: ...
    @property
    def qd(self) -> list[float]: ...
    @property
    def time(self) -> float: ...
    @property
    def energy(self) -> float: ...

    def set_torque(self, tau: _Vec) -> None: ...
    def set_gravity(self, g: _Vec3) -> None: ...
    def set_damping(self, d: _Vec) -> None: ...
    def reset(
        self, q0: Optional[_Vec] = ..., qd0: Optional[_Vec] = ...
    ) -> None: ...

    def rollout(
        self, horizon: float, sample_dt: Optional[float] = ...
    ) -> tuple[list[float], list[list[float]], list[list[float]]]:
        """Bake a rollout: (times, q, qd) over `horizon`."""
        ...

    def __repr__(self) -> str: ...

class ControlLoop:
    """A deterministic computed-torque control loop over a physical sim backend."""

    def __init__(
        self,
        robot: "Robot",
        dt: float = ...,
        kp: float = ...,
        kd: float = ...,
        gravity: Optional[_Vec3] = ...,
        start: Optional[_Vec] = ...,
    ) -> None: ...

    def run_to(self, goal: _Vec, ticks: int) -> None:
        """Regulate to `goal` for `ticks` steps (no recording)."""
        ...

    def rollout_to(
        self, goal: _Vec, ticks: int
    ) -> tuple[list[float], list[list[float]], list[list[float]]]:
        """Regulate to `goal`, recording → (times, states, actions)."""
        ...

    def step_with_target(self, action: _Vec) -> list[float]:
        """Step one tick toward `action`; return post-step measured q."""
        ...

    def run_stream(
        self,
        goal: _Vec,
        ticks: int,
        callback: Callable[[dict[str, Any]], Optional[bool]],
        emit_every: int = ...,
    ) -> int:
        """Regulate to `goal`, calling `callback(frame)` on every
        `emit_every`-th tick with a dict {tick, t, measured, measured_qd,
        command, warn} — bit-identical to the `rollout_to` sequence.

        The callback returns False to cancel cooperatively (None / any other
        value continues). Returns the number of ticks actually executed.
        """
        ...

    def estop(self) -> None: ...

    @property
    def q(self) -> list[float]: ...
    @property
    def qd(self) -> list[float]: ...
    @property
    def time(self) -> float: ...
    @property
    def tick(self) -> int: ...
    @property
    def last_warn(self) -> bool:
        """True iff the most recent `step_with_target` saturated the command."""
        ...

    def __repr__(self) -> str: ...

class Recorder:
    """Writes a LeRobotDataset v2.1 episode to disk."""

    def __init__(self, robot: "Robot", out: str, fps: int = ...) -> None: ...

    def start_episode(self, task: str) -> None: ...
    def append(self, state: _Vec, action: _Vec, t: float) -> None: ...
    def finalize_episode(self) -> None: ...
    def close(self) -> str:
        """Finalize the dataset (writes meta/) and return its path."""
        ...

class DatasetReader:
    """Reads a LeRobotDataset v2.1 from disk."""

    @staticmethod
    def open(path: str) -> "DatasetReader": ...

    @property
    def total_episodes(self) -> int: ...
    @property
    def ndof(self) -> int: ...
    @property
    def fps(self) -> int: ...

    def read_episode(
        self, episode: int
    ) -> tuple[list[list[float]], list[list[float]], list[float]]:
        """Read an episode → (states, actions, timestamps)."""
        ...

class CollisionModel:
    """Configuration-space collision checker (self + world)."""

    def __init__(
        self,
        robot: "Robot",
        ground: Optional[float] = ...,
        boxes: Optional[_Boxes] = ...,
        margin: float = ...,
    ) -> None: ...

    @property
    def num_colliders(self) -> int: ...
    @property
    def uncovered_frames(self) -> int: ...

    def query(self, q: _Vec) -> dict[str, Any]:
        """Query at `q` → dict(collision, self_pairs, world_hits, colliding_frames)."""
        ...

    def contacts(
        self, q: _Vec
    ) -> list[tuple[int, int, dict[str, Any]]]:
        """Penetration contacts (EPA) for every self-colliding link pair at `q`.

        Each item is `(frame_a, frame_b, {normal: [3], depth: float,
        witness: [3]})` with `frame_a < frame_b` (indices, as in `query()`'s
        `self_pairs`); translate A by `-depth * normal` to separate the pair.
        World geometry (ground/boxes) is not included — see `query()`.
        """
        ...

class SafetyMonitor:
    """The pure safety monitor: position clamp, velocity rate-limit, e-stop latch."""

    def __init__(self, robot: "Robot", q0: _Vec, dt: float = ...) -> None: ...

    def gate(self, desired: _Vec) -> tuple[list[float], dict[str, Any]]:
        """Sanitize a desired target → (safe_q, dict(clamped_position,
        limited_velocity, estopped, ok))."""
        ...

    def estop(self) -> None: ...
    def clear_estop(self) -> None: ...

    @property
    def is_estopped(self) -> bool: ...
    @property
    def warn_count(self) -> int: ...

class LeaderFollower:
    """Leader-follower teleop in pure sim: a follower control loop tracks a leader."""

    def __init__(self, robot: "Robot", dt: float = ...) -> None: ...

    def step(self, lead: _Vec) -> list[float]:
        """Move the leader to `lead`, step the follower once, return follower q."""
        ...

class Planner:
    """Collision-aware RRT-Connect motion planner."""

    def __init__(
        self,
        robot: "Robot",
        ground: Optional[float] = ...,
        boxes: Optional[_Boxes] = ...,
        seed: int = ...,
        step: float = ...,
        margin: float = ...,
    ) -> None: ...

    @property
    def uncovered_frames(self) -> int: ...

    def plan(self, start: _Vec, goal: _Vec) -> list[list[float]]:
        """Plan a collision-free waypoint path to a joint goal."""
        ...

    def plan_optimal(
        self, start: _Vec, goal: _Vec, iters: int
    ) -> list[list[float]]:
        """Plan an asymptotically-optimal (RRT*) smoothed joint-space path."""
        ...

    def plan_prm(
        self, start: _Vec, goal: _Vec, samples: int, k: int
    ) -> list[list[float]]:
        """Plan a collision-free, smoothed path with a PRM (Probabilistic
        RoadMap): `samples` free milestones, each wired to its `k` nearest
        free neighbours; shortest roadmap path, shortcut-smoothed.
        Deterministic for a given seed/samples/k."""
        ...

    def plan_to_pose(
        self, start: _Vec, target: _Pose, frame: Optional[_Frame] = ...
    ) -> list[list[float]]:
        """Plan to a Cartesian goal pose — a 4x4 COLUMN-MAJOR matrix (nested
        `target[col][row]` or flat-16, the same convention as `Robot.ik()`).

        The LEGACY flat 12-element row-major form (9 R then tx, ty, tz) is
        still accepted for back-compat. `frame` is a frame name (or a raw
        index, back-compat), defaulting to the tip.
        """
        ...

    def plan_trajectory(
        self, start: _Vec, goal: _Vec, dt: float = ...
    ) -> tuple[list[float], list[list[float]], list[list[float]]]:
        """Plan + retime → (times, q, qd)."""
        ...

    def verify(self, path: Sequence[Sequence[float]]) -> bool:
        """Independently re-verify a path is collision-free (finer resolution)."""
        ...

class ReachChecker:
    """Collision-aware reachability checker.

    `frame` is a frame name (or a raw index, back-compat); default = tip.
    """

    def __init__(
        self,
        robot: "Robot",
        ground: Optional[float] = ...,
        boxes: Optional[_Boxes] = ...,
        frame: Optional[_Frame] = ...,
        seeds: int = ...,
    ) -> None: ...

    def status(self, target: _Pose) -> dict[str, Any]:
        """Reachability of a Cartesian pose — a 4x4 COLUMN-MAJOR matrix
        (nested `target[col][row]` or flat-16, the same convention as
        `Robot.ik()`) →
        dict(status: "reachable"|"blocked"|"unreachable", residual, q)."""
        ...

    def reachable(self, target: _Pose) -> bool:
        """True iff the pose (same convention as `status()`) is reachable."""
        ...
