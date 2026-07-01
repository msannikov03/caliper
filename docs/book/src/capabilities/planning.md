# Planning

`caliper-planning` is a pure-CPU, dependency-light motion-planning crate. It
plans **collision-free joint-space waypoint paths** and, on request, retimes them
into playable trajectories.

## Sampling-based planners

- **RRT-Connect** — bidirectional, joint-space, with
  `caliper_collision::CollisionModel` (self + world) plus joint limits as the
  validity check. This is the default planner (`Planner`).
- **RRT\*** — the asymptotically-optimal variant (`rrtstar`).
- **PRM** — a probabilistic roadmap (`prm`).

All three are **deterministic**: they use a seeded splitmix64 PRNG (no `rand`),
so a given seed yields the same plan every time and the planners are fully
unit-testable with no hardware.

## Smoothing and retiming

A raw sampled path is jagged. `Planner` **shortcut-smooths** the result
(repeatedly attempting to replace sub-paths with collision-free straight-line
shortcuts). To play or record the plan, `Planner::plan_trajectory` (via the
motion crate's `retime`) turns the collision-free waypoint path into a
jerk-limited `caliper_motion::Trajectory`.

## Reachability

`caliper_planning::reach` provides reachability analysis — a three-way
classification of goals as reachable / blocked / out-of-reach that is
collision-aware.

## Trajectory optimization (CHOMP)

`caliper-trajopt` is a separate, CHOMP-style **collision-aware trajectory
optimizer**. Given an initial waypoint path, it refines the *interior* waypoints
(endpoints held fixed) by gradient descent on

```text
cost(path) = w_smooth · smoothness(path) + w_obs · obstacle(path)
```

- **smoothness** is the classic CHOMP quadratic — summed squared
  finite-difference accelerations `‖q[i-1] − 2·q[i] + q[i+1]‖²` over the interior.
  Its analytic gradient (the pentadiagonal `AᵀA q` form) is cross-validated in
  the tests against a finite-difference of the cost.
- **obstacle** is a smooth proximity penalty: each robot collider is reduced to
  a body sphere, placed by FK and inflated by its bounding radius, and scored
  against an `ObstacleField` signed-distance field with the standard CHOMP hinge
  potential.

## How planning is verified

The planners and smoothing are **re-derived-correct but self-consistent-only**.
The important honesty here is the nature of the collision guarantee: it is a
**sampled-at-resolution** guarantee. The planner checks configurations at a
discrete resolution along each edge, so a sufficiently narrow passage can be
*tunneled* — the plan can be reported collision-free while a thin obstacle slips
between samples. There are also no dedicated narrow-passage / near-π / at-limit
stress fixtures yet; coverage is random sampling. These are documented
limitations, not defects.
