# Dynamics & simulation

`caliper-dynamics` implements the standard rigid-body dynamics algorithms on the
frozen `Model`:

- **RNEA** — the Recursive Newton–Euler Algorithm for inverse dynamics: given
  `(q, q̇, q̈)`, compute the joint torques `τ`.
- **CRBA** — the Composite Rigid Body Algorithm for the joint-space mass
  (inertia) matrix `M(q)`.
- **Forward dynamics** — given `(q, q̇, τ)`, compute `q̈` (using `M` and the
  RNEA bias term).
- **`Simulator`** — a semi-implicit (symplectic-style) Euler integrator with
  gravity, advanced by explicit `step(dt)` calls.

## Conventions

Spatial quantities follow the `[v; ω]` twist ordering and are
Pinocchio-compatible, matching `caliper-spatial`. This alignment is what lets the
oracle compare against Pinocchio directly.

## How dynamics is verified

RNEA, CRBA, and forward dynamics are **externally cross-validated against
Pinocchio** to residuals on the order of ~1e-9. (An earlier RNEA sign bug was in
fact caught by exactly this external cross-validation, which is part of why the
oracle exists.) The `Simulator`'s integrators are checked for energy-bounded
behavior rather than against an external reference.

> **Honest note.** The `Simulator` is validated as energy-bounded, and its
> `energy` reporting assumes Earth gravity — a documented assumption, not a bug.
> The native simulation path has **no collision** built in; collision is a
> separate crate that plugs into the control/safety layer.
