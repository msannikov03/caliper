# Control & safety

`caliper-hal` is the hardware/simulation abstraction layer plus a deterministic
control stack. It turns a robot — real or simulated — into a uniform,
tick-driven contract.

## The backend contract

`RobotBackend` is the real backend contract: control modes, lifecycle/safety,
atomic state readback, and a tick-driven `step(dt)`. `SimBackend` is the built-in
simulated implementation. Everything is **clock-free**: nothing advances until
the loop calls `step(dt)`, and `t == tick * dt`. There is no `Instant::now` and
no wall clock, so a rollout is bit-for-bit reproducible and testable without any
real robot.

## The control loop

`ControlLoop` is a deterministic **computed-torque** controller. The design
lesson baked in here is important: a fixed-gain PD controller diverges on
low-inertia wrists, so Caliper uses computed-torque (model-based) control, which
gives one gain pair that works across any robot. The loop saturates the
*command* (not merely the position reference), so limits are actually enforced on
what is sent to the actuator.

## Safety

`SafetyMonitor` is a pure (side-effect-free) safety layer. `SafetyCheck` is the
pluggable predicate the collision crate implements, so collision rejection slots
directly into the safety path. `ControlLoop.step_with_target` exposes a
`last_warn` channel so a caller (or the learning sidecar) can see when the safety
layer intervened.

## Setpoint sources and teleop

`Setpoint` sources drive the loop, including a **teleop** leader–follower source
(one arm's state commands another). Because everything is tick-driven, teleop is
just another deterministic setpoint stream.

## Dataset record / replay

With the `dataset` feature, `caliper-hal` records and replays a **LeRobotDataset
v2.1** — the standard schema used for imitation-learning data — which the
Phase-7 learning sidecar consumes.

## Hardware skeletons

Feature-gated **CAN** and **Dynamixel** hardware backends exist as *skeletons*.
They are the interface stubs for real actuators; the physics, control, and safety
above are fully implemented and tested against the simulated backend.

## How control/safety is verified

The computed-torque decoupling and the safety monitor are **re-derived-correct**,
with the control law validated on a 2-DOF pendulum (a case with a known
closed-form). They were further hardened during the audit (input validation,
NaN/limit guards). The dataset path is validated by **pyarrow schema + NumPy
statistics** — note that `lerobot` itself is not importable in the test env, so
the check is against the *schema*, not against lerobot's own reader.
