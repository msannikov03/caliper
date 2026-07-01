# Kinematics & IK

## Forward kinematics

`caliper-kinematics` computes forward kinematics from a frozen `Model`: given a
joint configuration `q`, it places every frame in the world. It also produces
the **geometric Jacobian** in two flavors:

- **world** (LWA-style), and
- **body** (LOCAL).

These two flavors match the two reference frames Pinocchio exposes, which is why
the oracle can check them directly. FK and the world Jacobian are
**Pinocchio-validated** to residuals on the order of 1e-9…1e-15.

## Singularity analysis

The kinematics crate also computes singularity metrics from the Jacobian: the
singular-value spectrum, Yoshikawa **manipulability**, and the **condition
number**, plus the manipulability ellipsoid (eigendecomposition) and a
redundant-arm nullspace. The scalar metrics (σ, manipulability, condition
number) are cross-checked against a NumPy SVD in the oracle. The ellipsoid
eigendecomposition and the nullspace are re-derived-correct but validated only
against Caliper itself (no external reference); a "singular joint"
classification is treated as **advisory**, not a hard guarantee.

## Inverse kinematics

`caliper-ik` provides two IK paths:

1. **Iterative CLIK** — damped-least-squares / Levenberg–Marquardt closed-loop
   inverse kinematics with:
   - manipulability-gated damping (more damping near singularities),
   - per-step clamping,
   - joint-limit handling,
   - multi-restart to escape poor local basins.

2. **Analytic 6R IK** (`caliper_ik::analytic`) — a closed-form solver for the
   standard 6R wrist-partitioned geometry, returning the discrete set of
   branch solutions.

### How IK is verified

The IK **solver** is validated by the FK∘IK round-trip: solve for `q`, run FK,
and confirm the resulting pose matches the target to tolerance. That closure is
strong evidence the solver converges to correct configurations, but note the
honest caveat — it is *self-consistent* against Caliper's own FK, not checked
against an independent task-space DLS reference. Because FK itself *is*
externally validated against Pinocchio, a correct FK∘IK closure is meaningful,
but a defect shared between FK and IK would not be caught by this test alone.
