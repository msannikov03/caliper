# Contact simulation (MuJoCo)

`caliper-sim-mujoco` puts MuJoCo behind caliper's existing backend seam, so the
same `ControlLoop` / `SafetyMonitor` / teleop / recording stack that drives
`PhysicsSimBackend` (contact-free) can drive a full contact simulation
unchanged. Engine-side only for now — no Studio or Python surface yet.

Two layers:

- **`mjcf`** — generates a *minimal* MJCF document from a caliper `Model`:
  kinematic tree, hinge/slide joints, inertials (converted from caliper's
  link-origin spatial inertias to MuJoCo's about-COM convention), primitive
  collision geoms, an optional ground plane, and optional `<position>`
  actuators. Pure string work: always compiled and tested, no MuJoCo needed.
- **`MujocoSim` / `MujocoBackend`** (cargo feature `mujoco`) — a thin safe
  layer over the pinned `mujoco-rs 5.0.0` wrapper (tracks MuJoCo **3.9.0**
  exactly), plus a `caliper_hal::RobotBackend` implementation.

## Actuation — chosen at construction

A MuJoCo `<position>` servo applies force on *every* step, so it cannot
coexist with direct torque injection. The variant is therefore fixed when the
model is built:

| Variant (`mjcf::Actuation`) | `Torque` mode | `Position` mode |
|---|---|---|
| `TorqueDirect` (default) | writes `qfrc_applied` directly — no actuators at all | **non-physical teleport** (mirrors `PhysicsSimBackend`) |
| `PositionServo { kp, kv }` | `UnsupportedMode` — honest error | writes servo targets to `ctrl`; MuJoCo computes the torque |

`estop()` latches, zeroes `qfrc_applied`, and (on the servo variant) freezes
`ctrl` at the current position — a zeroed servo target would actively drive to
`q = 0`, the opposite of a stop.

## Determinism

- MuJoCo runs single-threaded per `mjData`; caliper never opts into
  `mjThreadPool` and generates no noisy sensors.
- `MujocoSim::reset()` restores the *full* integration state (time, warmstart
  included), so two identical command sequences are **bitwise identical** —
  there is a test asserting exactly that.
- Bitwise reproducibility holds per binary + per MuJoCo release only. That is
  why the wrapper is pinned exactly (`mujoco-rs = "=5.0.0"` ↔ MuJoCo 3.9.0).
- `step(dt)` only accepts integer multiples of the model timestep — no silent
  remainder drift.

## Verification

Feature-gated integration tests cover: MJCF round-trips through the real
MuJoCo compiler; gravity sag with zero torque; a sphere-tipped pendulum
settling **on a ground plane** (contact list non-empty, ±z normal, positive
depth and normal force); bitwise-identical repeat runs; the existing
`ControlLoop` converging through a `MujocoBackend`; and a cross-check of
caliper's own gravity `Simulator` vs MuJoCo on the 2-link pendulum
(|Δq| < 2·10⁻² rad over 0.3 s at h = 10⁻⁴ — a deliberately loose tolerance:
the integrators differ, and the check exists to catch sign/axis/inertia
mapping bugs, not truncation error).

## Honest scope & gaps

- Fixed-base trees of 1-dof joints only (free/ball joints are rejected).
- **Mesh colliders are not exported**: `CollisionShape::ConvexHull` entries
  are *counted* (`skipped_hull_colliders`) rather than silently dropped, so a
  MuJoCo model can have less collision coverage than `caliper-collision` on
  the same robot. MJCF mesh assets are deferred.
- URDF is not fed to MuJoCo directly (MuJoCo parses URDF but cannot express
  actuators/solver options there); caliper generates MJCF instead, and joint
  addressing is resolved **by name** at load — never by assuming index order.
- Caliper's `Model` does not carry URDF `<dynamics damping>`; MJCF damping is
  a uniform knob (`MjcfOptions::joint_damping`), not a translation.
- Velocity mode is unsupported (as everywhere else in the HAL).

## Building with MuJoCo

The default build needs nothing. Enabling the seam links a **shared
libmujoco 3.9.0** that `mujoco-rs` does not download on macOS:

```bash
scripts/fetch_mujoco.sh                       # pinned official release
export MUJOCO_DYNAMIC_LINK_DIR=~/.cache/caliper/mujoco-3.9.0
export DYLD_LIBRARY_PATH=$MUJOCO_DYNAMIC_LINK_DIR:$DYLD_LIBRARY_PATH  # macOS
cargo test -p caliper-sim-mujoco --features mujoco
```

CI runs only the default (MuJoCo-free) build of this crate; the feature-gated
tests are a local/gated lane until a cached-artifact CI job is added.
