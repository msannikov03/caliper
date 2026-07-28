# Pose forms

Every Cartesian pose in Caliper is a 4×4 homogeneous transform, but the
**memory layout differs between outputs and inputs** — this page is the one
table to check before wiring poses between calls. It exists because the
mismatch fails *silently* (see [the transpose trap](#the-transpose-trap)
below).

## The one rule

> **FK outputs are row-major. Pose inputs are column-major.**

| Surface | Direction | Form |
|---|---|---|
| `Robot.fk`, `exp6` (Python) | output | 4×4 **ROW-major** nested list — `m[row][col]`, `np.array(m)` is the matrix |
| `log6` (Python) | input | 4×4 **ROW-major** nested list (same layout `fk` returns) |
| `Robot.ik` / `analytic_ik` / `move_l` / `move_c`, `Planner.plan_to_pose`, `ReachChecker.status` / `reachable`, `calibrate_joint_offsets` (Python) | input | 4×4 **COLUMN-major** nested (`pose[col][row]`) **or** flat 16-element column-major |
| `Planner.plan_to_pose` (Python, legacy only) | input | also grandfathers the flat **12-element row-major** form (9 rotation entries, then `tx, ty, tz`) — new code should use the 4×4 form |
| `ik` / `move` / `plan` / `reach` `--target` (CLI) | input | **12 numbers: 9 row-major rotation entries, then `tx, ty, tz`** |
| `calibrate --observations` JSON (CLI) | input | per observation: flat-16 **column-major**, or a 4×4 nested (row-major) matrix |
| Studio backend DTOs (`frames`) | output | flat-16 **column-major** `Mat4` (the three.js/WebGL layout) |

Why the split: `fk` returning a list of rows is what NumPy/`np.array` users
expect, while the pose-input side is the column-major convention `ik()` (and
the Studio/three.js path) always used — both are frozen under the [stability
contract](stability.md), so neither can quietly flip. The flat-16
column-major form is the standard graphics-stack layout (three.js, WebGL,
`np.flatten(order="F")`).

## The transpose trap

The most common new-user mistake, and it does not error:

```python
q2 = robot.ik(robot.fk(q), seed)["q"]      # WRONG — silently solves the transpose
```

`fk` returns a nested list of ROWS; `ik` reads a nested list as COLUMNS
(`pose[col][row]`). The same bytes therefore parse as the **transpose**: the
target rotation becomes `Rᵀ` (the inverse rotation) and the target
translation becomes the homogeneous `[0, 0, 0]` bottom row — so IK happily
"solves" toward a wrong orientation at the origin, converging or not, with no
exception anywhere.

Transpose at the boundary and the round-trip works:

```python
import numpy as np

T = np.array(robot.fk(q))                       # 4x4, row-major → real matrix
res = robot.ik(T.T.tolist(), seed)              # transpose = column-major nested
# equivalently, the flat-16 column-major form:
res = robot.ik(T.flatten(order="F").tolist(), seed)
# without numpy:
res = robot.ik([list(col) for col in zip(*robot.fk(q))], seed)
```

The same applies to every pose-accepting call in the table above —
`move_l`, `plan_to_pose`, `ReachChecker.status`, `calibrate_joint_offsets` —
they all share `ik`'s input convention, so one `T.T` habit covers the whole
surface.

## Rotation handling

Pose inputs project the rotation block onto SO(3) (`from_matrix`), so a
slightly non-orthonormal basis is tolerated and cleaned up. This is a
feature for numerically-drifted matrices — and part of why the transpose trap
is silent: a transposed rotation is still a perfectly valid rotation. An
orthonormality/handedness *acceptance* check is a candidate future
tightening, not current behavior.
