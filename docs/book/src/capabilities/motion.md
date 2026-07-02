# Motion

`caliper-motion` produces smooth, jerk-limited trajectories with O(1)
closed-form sampling: a `Trajectory` answers `sample(t)` in constant time
(position/velocity/acceleration), so it plays back cheaply and deterministically.

## Jerk-limited S-curve profiles

The core is a **7-segment S-curve** (jerk-limited trapezoidal) profile. It
respects velocity, acceleration, and jerk limits (`MotionLimits`).

- **MOVE_J** — joint-space, **time-synchronized**: all joints start and finish
  together, driven by the slowest joint's limits.
- **MOVE_L** — Cartesian straight-line motion of the tool frame.
- **MOVE_C** — Cartesian circular/arc motion.

The Cartesian entry points validate their caps, `dt`, and goal finiteness
(non-finite goals are rejected), symmetric with the joint-space path — this was
tightened in the audit.

MOVE_C fits the unique circle through the start / via / end tip positions and
sweeps it the short way, so the parameterization passes **through** the via
point on its way to the end (the naive arc frame could sweep the long way
round — regression-tested). It is wired to every face: `move_c` in Rust and
Python, and `caliper move --target ... --via tx,ty,tz` on the CLI, with oracle
coverage (endpoint + via reached within joint velocity limits).

## Waypoint retiming

`retime_waypoints` takes a joint-space waypoint path (for example, the output of
the planner) and turns it into a playable, jerk-limited `Trajectory`. This is
how a planned path becomes something a control loop or Studio can execute and
record.

## Time-optimal parameterization (TOPP)

`caliper-motion` also includes a **time-optimal, acceleration-limited
parameterization** of a joint-space waypoint path (`topp`), with **corner stops**
at every interior waypoint.

The reasoning is explicit in the code: a piecewise-linear path `q(s)` has a
discontinuous tangent at every interior waypoint, so `q''(s)` is an unbounded
Dirac there. Joint acceleration along the path is `q̈ᵢ = q'ᵢ·s̈ + q''ᵢ·ṡ²`; the
`q''·ṡ²` term explodes at a corner unless the path velocity `ṡ` is zero there.
Caliper therefore drives `ṡ → 0` at each interior waypoint so the spike
vanishes. Per segment the tangent `q'(s) = Δq` is constant, giving the two
scalar bounds

```text
|q̇ᵢ| = |Δqᵢ|·ṡ ≤ vmaxᵢ   ⟺   ṡ ≤ minᵢ vmaxᵢ/|Δqᵢ|
|q̈ᵢ| = |Δqᵢ|·s̈ ≤ amaxᵢ   ⟺   s̈ ≤ minᵢ amaxᵢ/|Δqᵢ|
```

and a rest-to-rest bang-bang (trapezoid/triangle) profile in `s` over `[0,1]` is
time-optimal subject to those bounds. Segments are concatenated (rest between
them) and resampled onto a uniform `dt` grid.

## How motion is verified

All of `caliper-motion` is **re-derived-correct but self-consistent-only**:
there is no third-party trajectory oracle (nothing Ruckig-class) wired in. The
profiles are checked against Ruckig-class jerk-limited *expectations* and by
property tests (endpoint exactness, monotonicity, limit adherence) rather than
against an external reference implementation. This is one of the places where the
trust comes from re-derivation plus invariants, not from an external cross-check.
