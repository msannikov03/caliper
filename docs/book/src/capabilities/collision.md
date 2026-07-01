# Collision

`caliper-collision` is a self-contained, **pure-nalgebra** collision checker —
no `parry`/`rapier` (they were deliberately rejected to keep the dependency
surface lean). `CollisionModel` builds primitive colliders from a `Model`'s
parsed `<collision>` geometry, places them by forward kinematics at a
configuration `q`, and reports:

- **self-collisions** — between link pairs, excluding an auto-seeded adjacency
  allowlist (adjacent links are expected to touch), and
- **world collisions** — against a ground half-space and world boxes.

It implements `caliper_hal::SafetyCheck`, so the control loop / safety layer can
reject a colliding command.

## Geometry

- **Box ↔ box** — the separating-axis theorem (15 axes, Ericson), including the
  edge-edge degeneracy.
- **Sphere/box** and **half-space** — closed form.
- **Cylinders** — conservatively approximated by their tight oriented bounding
  box (this errs *toward* detecting a collision, which is the safe direction).
- **Capsules** — swept spheres (a core segment ⊕ a sphere of `radius`):
  capsule ↔ sphere/half-space/capsule use closed-form point-segment /
  segment-segment distances; capsule ↔ box/convex reuse GJK via the capsule's
  exact support function.
- **Mesh** — arrives as the convex hull of its vertices and is checked with
  **GJK** (boolean origin-in-Minkowski-difference).

## Penetration depth (EPA)

On overlap, `CollisionModel::contacts` runs **EPA** (the Expanding Polytope
Algorithm) on top of GJK to recover a `Contact` for each colliding pair: a unit
separation `normal` (the outward normal of the Minkowski difference), a
penetration `depth`, and a `witness` point.

## Honesty about coverage

The checker is **re-derived-correct** and, by construction, **cannot
under-report** for the geometry it handles. But two things are called out
explicitly:

- Colliders that cannot be reduced to the supported primitives (some mesh /
  capsule cases) are **surfaced loudly** via an `uncovered_frames` report rather
  than silently dropped — you always know what was *not* checked.
- The native `Simulator` has no collision; collision is only enforced where the
  `SafetyCheck` is wired in (the control/safety layer).
