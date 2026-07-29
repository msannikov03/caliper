# Contact simulation (MuJoCo)

`caliper-sim-mujoco` puts MuJoCo behind caliper's existing backend seam, so the
same `ControlLoop` / `SafetyMonitor` / teleop / recording stack that drives
`PhysicsSimBackend` (contact-free) can drive a full contact simulation
unchanged. Faces: Studio's Simulate mode drives the live sim in mujoco
builds (with the `C001`–`C003` stability lint run after every bake), and
Python reaches the MJCF generator via `model_to_mjcf` (incl. `material=` /
`actuators=`); the `MujocoSim`/`MujocoBackend` layer itself has no Python
binding yet.

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

## Live session (Studio)

Studio's Simulate mode can also run the sim **live** instead of baking a clip:
a background thread owns a `ControlLoop` over the `MujocoBackend` (or, in
MuJoCo-free builds, the builtin contact-free integrator) and steps it at a
fixed **1 ms** physics timestep while a PD servo holds a live-mutable joint
target. Each emitted state carries the joint positions/velocities, world
frames, tip position, prop poses, and the **live contact count**; it streams
to the viewport at render rate — nominally 60 Hz, actually
1/(17 · 1 ms) ≈ 58.8 Hz after decimating to a whole number of physics steps.

Design points, stated plainly:

- **Fixed timestep, wall-clock paced.** An accumulator converts elapsed wall
  time into whole physics steps; catch-up debt is capped at 0.25 s — beyond
  that, excess time is dropped rather than spiraling into ever-larger step
  batches.
- **Pause freezes; it does not de-energize.** Pausing stops stepping *and*
  stops accumulating wall time, with the servo target untouched — the arm
  holds exactly where it is. This is deliberately not `estop()`/`disable()`:
  those de-energize, and a de-energized arm falls.
- **Reset rides the determinism anchor.** Reset reseeds the backend —
  `MujocoSim::reset()`, the full `mj_resetData` (warmstart included, the same
  mechanism behind the bitwise-reproducibility test above) — and rebuilds the
  control loop at the reset pose, so the session clock and tick counter
  restart at zero. Reset works while paused and still emits one state, so the
  viewport always matches the sim.
- **Errors end the session loudly.** A step error or a non-finite state ends
  the session with an `error: …` reason; there is no silent freeze.
- **Builtin fallback.** MuJoCo-free builds run the identical session on
  `PhysicsSimBackend` — gravity only: no contacts, no ground reaction, and
  props are rejected with a clear error rather than silently dropped.
- **Driving it.** Every input edits the *PD hold target*, never the streamed
  pose — the sim remains the single source of truth for where the arm is, so
  input and stream cannot fight. Joint sliders become live target editors
  (the measured pose is drawn as a ghost tick, making servo lag visible), the
  IK gizmo retargets the tip (solutions seeded from the target, not the
  lagging measurement, so consecutive drags compose), `[`/`]` select a joint
  and `-`/`=`/arrows jog it at a rate scaled to the joint type, and a gamepad
  drives the tip in cartesian world axes (0.15 stick deadband with a cubic
  response curve; A pauses, B resets). Space freezes and unfreezes. At most
  one target update is sent per rendered frame.

**Grasping — a weld heuristic, stated plainly.** A live session with a
gripper channel can pick props up, and it does it the way sim teleop rigs
actually do: not finger-friction physics, but an explicitly-labeled **weld**.
The gripper joint is auto-detected by name (`gripper`/`finger`/`jaw`/… on the
joint *or its child link* — SO-101's joints are named `1`–`6`, the semantics
live in the links; mimic joints are skipped so Panda resolves to the driving
finger) or overridden explicitly; open/close is just a PD target move to the
joint's limits (inset 2% so the hold target never slams a stop). When the
gripper is *commanded* closed and a prop is in contact with the robot, the
prop welds to the attach link — with its relative pose captured at that
instant, so activation is snap-free (measured < 1 mm across the activation
tick) — and opening releases it to fall naturally. One prop at a time; reset
releases; a MuJoCo weld is a soft constraint, so a carried prop sags ~1–2 mm
under a hard swing. Two honest limits: contact with *any* robot geom counts
(a prop leaning on the forearm can be taken), and `closed` in the stream is
the command, not a measurement — a gripper squeezing a prop reads closed
while its joint never reaches the closed target.

**Live vs. bake — both exist because they answer different questions.** A bake
is a fixed command sequence through the deterministic sim: reproducible
clip-for-clip, and the `C001`–`C003` stability lint runs over the finished
rollout. A live session is paced by the wall clock and driven by whatever the
UI sends, so it is for *watching and interacting*, not for reproducible
artifacts.

**Recording teleop episodes.** While driving a live session you can record
straight into a native [LeRobotDataset v3.0](./learning.md) — the same format
the training side reads, no conversion step. Capture happens *in the session
thread at exact tick decimation* (default 50 fps from the 1 kHz loop; any fps
that divides the tick rate), writing `observation.state` (measured joints)
and `action` (the PD hold target) per frame, so timestamps are exact
`k/fps` — never wall-clock-sampled. A take is start/stop with a per-episode
task label; stopping either saves the episode or discards it. Pausing
mid-take freezes capture and resumes the same take with no timestamp gap.
Resetting mid-take **discards the take** (a reset invalidates the
demonstration) and says so. Ending the session finalizes the open dataset. A
Studio-recorded dataset loads directly in real lerobot — verified against
lerobot 0.6.0.

The `live_*` Tauri commands and `live://` events behind this are
**Studio-internal IPC, not a public API** — they fall in the same not-promised
bucket as Studio UI layout in the
[stability contract](../reference/stability.md). Script against the CLI/Python
faces instead.

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

## Shipping the app with contact sim (macOS)

The recipe above serves *development*: the dylib lives in a cache directory
with an absolute install id, so the binary only runs on the machine that
fetched it. To ship a Studio `.app`/`.dmg` with the `mujoco` feature **on**,
the bundle must carry `libmujoco` itself and resolve it via `@rpath`:

```bash
scripts/bundle_mujoco.sh    # fetch + stage src-tauri/vendor/ (gitignored)
cd apps/studio
MUJOCO_DYNAMIC_LINK_DIR="$PWD/src-tauri/vendor" npm run tauri build -- \
  --features mujoco \
  --config "$PWD/src-tauri/tauri.mujoco.conf.json"
```

How the pieces fit (each step verified against the tauri 2.x sources):

1. `bundle_mujoco.sh` copies the pinned dylib into
   `apps/studio/src-tauri/vendor/`, rewrites its install id to
   `@rpath/libmujoco.3.9.0.dylib`, and ad-hoc re-signs it
   (`install_name_tool` invalidates signatures, which SIGKILLs on Apple
   Silicon). Linking against **this** copy is what stamps the relocatable
   `@rpath/...` load command into the executable — the bundler never rewrites
   install names after the fact.
2. `tauri.mujoco.conf.json` is a *separate overlay* config, merged over
   `tauri.conf.json` by `--config` (JSON Merge Patch). It adds
   `bundle.macOS.frameworks = ["vendor/libmujoco.3.9.0.dylib"]`. It cannot
   live in the default config: tauri-build hard-errors whenever a listed
   dylib is missing, which would break every ordinary build.
3. With a non-empty `frameworks` list, tauri-build links the executable with
   `-Wl,-rpath,@executable_path/../Frameworks`, and the bundler copies the
   dylib into `Contents/Frameworks/` and signs it with the app. At launch,
   `@rpath/libmujoco.3.9.0.dylib` resolves inside the bundle — no
   `DYLD_LIBRARY_PATH`, no per-machine paths.

Honest caveats:

- The `.dmg` grows by ~9 MB (the universal2 `libmujoco.3.9.0.dylib` is
  8.6 MB).
- The staged dylib is ad-hoc signed, then re-signed with whatever identity
  signs the app (currently an Apple Development cert, not notarized) — the
  usual right-click → Open applies, same as the plain release.
- `--features mujoco` **without** the overlay config produces a binary whose
  `@rpath/libmujoco...` load command resolves nowhere — always pass both
  flags together (or neither).
- macOS only; the default MuJoCo-free bundle is completely unaffected.
