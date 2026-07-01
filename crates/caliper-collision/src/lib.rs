//! Collision detection for Caliper — a self-contained, pure-nalgebra checker.
//!
//! [`CollisionModel`] builds primitive colliders from a [`Model`]'s parsed
//! `<collision>` geometry, places them by forward kinematics at a configuration
//! `q`, and reports self-collisions (excluding an auto-seeded adjacency allowlist)
//! plus world collisions (ground half-space + boxes). It implements
//! [`SafetyCheck`](caliper_hal::SafetyCheck), so the control loop / safety layer
//! can reject colliding commands.
//!
//! Geometry: oriented-box ↔ oriented-box uses the separating-axis theorem (15
//! axes, Ericson); sphere/box and half-space cases are closed-form. Cylinders are
//! conservatively approximated by their tight oriented bounding box (errs toward
//! detecting a collision — safe). CAPSULES are swept spheres (a core segment ⊕ a
//! sphere of `radius`): capsule ↔ sphere/half-space/capsule use closed-form
//! point-segment / segment-segment distances, and capsule ↔ box/convex reuse GJK
//! via the capsule's exact support function. MESH colliders arrive as a convex
//! hull of vertices ([`CollisionShape::ConvexHull`]) and are checked with GJK
//! (boolean origin-in-Minkowski-difference). Everything is deterministic and
//! dependency-free.
//!
//! PENETRATION DEPTH: on overlap, [`CollisionModel::contacts`] runs EPA (the
//! expanding polytope algorithm) on each self-colliding pair to recover a
//! [`Contact`] (separation `normal`, penetration `depth`, and a `witness` point).
//! It is additive: the boolean [`CollisionModel::query`] / [`CollisionReport`] are
//! unchanged.
//!
//! ⚠ SCOPE: box/sphere/cylinder/capsule/convex-hull(mesh) `<collision>` geometry is
//! checked. A `<mesh>` the loader could not read carries no collider, so that part
//! is NOT checked — a report can read "clear" while such a link interpenetrates.
//! [`CollisionModel::uncovered_frames`] returns that count; callers should surface
//! it rather than trust a "clear" verdict blindly.

use caliper_hal::SafetyCheck;
use caliper_kinematics::fk_frame;
use caliper_model::{CollisionShape, Model};
use nalgebra::{Matrix3, Vector3};
use std::collections::HashSet;
use std::sync::Arc;

#[derive(thiserror::Error, Debug)]
pub enum CollisionError {
    #[error("expected {expected} dofs, got {got}")]
    Dim { expected: usize, got: usize },
    #[error("non-finite configuration")]
    NonFinite,
}

/// Static world geometry the arm can collide with.
#[derive(Clone, Debug, Default)]
pub struct WorldScene {
    /// A solid ground half-space: everything at `z ≤ ground_z` is solid.
    ground_z: Option<f64>,
    /// Axis-aligned obstacle boxes: `(center, half_extents)`.
    boxes: Vec<([f64; 3], [f64; 3])>,
}
impl WorldScene {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_ground(mut self, z: f64) -> Self {
        self.ground_z = Some(z);
        self
    }
    pub fn add_box(mut self, center: [f64; 3], half: [f64; 3]) -> Self {
        // A negative or non-finite half-extent would poison the SAT: `f64::clamp`
        // panics when `min > max` (negative half flips the bounds) and a `NaN`
        // extent makes every comparison `false`, so the box would silently "never
        // separate" (i.e. collide with everything). The builder returns `Self` and
        // cannot surface an error, so sanitize to a finite, non-negative box.
        let half = [
            sanitize_extent(half[0]),
            sanitize_extent(half[1]),
            sanitize_extent(half[2]),
        ];
        self.boxes.push((center, half));
        self
    }
}

/// Clamp a box half-extent to a finite, non-negative value (NaN/∞/negative → 0).
fn sanitize_extent(x: f64) -> f64 {
    if x.is_finite() { x.max(0.0) } else { 0.0 }
}

/// Result of a collision query. Pairs/hits are canonically sorted → deterministic.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CollisionReport {
    /// Self-colliding `(frame_a, frame_b)` pairs with `frame_a < frame_b`.
    pub self_pairs: Vec<(usize, usize)>,
    /// Frames intersecting world geometry.
    pub world_hits: Vec<usize>,
    /// Union of all frames involved in any collision.
    pub colliding_frames: Vec<usize>,
}
impl CollisionReport {
    pub fn has_collision(&self) -> bool {
        !self.self_pairs.is_empty() || !self.world_hits.is_empty()
    }
}

/// An oriented collision primitive in world coordinates.
#[derive(Clone, Debug)]
enum Prim {
    /// Oriented box: center, orientation (columns = local axes), half-extents.
    Obb {
        c: Vector3<f64>,
        r: Matrix3<f64>,
        h: Vector3<f64>,
    },
    Sphere {
        c: Vector3<f64>,
        radius: f64,
    },
    /// Convex hull (mesh collider) in WORLD coordinates. `margin` inflates the
    /// hull by a sphere of that radius at GJK support time (conservative, exact
    /// Minkowski sum). Never empty (a ConvexHull collider always has >= 3 points).
    Convex {
        points: Vec<Vector3<f64>>,
        margin: f64,
    },
    /// Capsule = a swept sphere: the set of points within `radius` of the core
    /// segment `[a, b]` (both endpoints in WORLD coordinates). A degenerate
    /// `a == b` reduces to a sphere. Its support function is exact, so it plugs
    /// straight into GJK/EPA.
    Capsule {
        a: Vector3<f64>,
        b: Vector3<f64>,
        radius: f64,
    },
}

/// A penetration contact recovered by EPA for an overlapping primitive pair.
#[derive(Clone, Debug, PartialEq)]
pub struct Contact {
    /// Unit separation axis = the outward normal of the Minkowski difference
    /// `A ⊖ B` at the boundary point closest to the origin. Translating `A` by
    /// `-depth * normal` (or `B` by `+depth * normal`) just resolves the overlap.
    pub normal: Vector3<f64>,
    /// Penetration depth ≥ 0 (the minimum translation distance along `normal`).
    pub depth: f64,
    /// A witness point on the surface of `A`, in world coordinates (the deepest
    /// point of `A` inside `B`, recovered from the EPA face's support points).
    pub witness: Vector3<f64>,
}

/// A configuration-space collision checker over a robot model + a world scene.
pub struct CollisionModel {
    model: Arc<Model>,
    scene: WorldScene,
    margin: f64,
    /// Allowlisted `(a,b)` frame pairs (`a<b`) skipped in self-collision.
    allowed: HashSet<(usize, usize)>,
}

impl CollisionModel {
    /// Build a checker; auto-seeds the adjacency allowlist. `margin` inflates each
    /// collider (conservative).
    pub fn new(model: Arc<Model>, scene: WorldScene, margin: f64) -> Self {
        let allowed = seed_allowlist(&model);
        Self {
            model,
            scene,
            margin: margin.max(0.0),
            allowed,
        }
    }
    pub fn num_colliders(&self) -> usize {
        self.model.collision.len()
    }
    /// Frames NOT fully collision-covered (unsafe-by-omission hint for the UI/docs):
    /// frames with NO collider, PLUS frames that carry a primitive collider but also
    /// had a mesh collider DROPPED (only partially covered — a query can still
    /// report "clear" for the dropped part).
    pub fn uncovered_frames(&self) -> usize {
        let dropped: HashSet<usize> = self.model.dropped_collider_frames.iter().copied().collect();
        let fully_covered: HashSet<usize> = self
            .model
            .collision
            .iter()
            .map(|g| g.frame)
            .filter(|f| !dropped.contains(f))
            .collect();
        self.model.frames.len() - fully_covered.len()
    }
    pub fn allowlisted(&self, a: usize, b: usize) -> bool {
        let key = if a < b { (a, b) } else { (b, a) };
        self.allowed.contains(&key)
    }

    /// Query collisions at configuration `q`.
    pub fn query(&self, q: &[f64]) -> Result<CollisionReport, CollisionError> {
        if q.len() != self.model.ndof {
            return Err(CollisionError::Dim {
                expected: self.model.ndof,
                got: q.len(),
            });
        }
        if !q.iter().all(|x| x.is_finite()) {
            return Err(CollisionError::NonFinite);
        }

        let geoms = &self.model.collision;
        let placed: Vec<(Prim, usize)> = geoms
            .iter()
            .map(|g| {
                let world = fk_frame(&self.model, q, g.frame).0 * g.origin.0;
                let r = world.rotation.to_rotation_matrix().into_inner();
                let c = world.translation.vector;
                (prim_for(&g.shape, c, r, self.margin), g.frame)
            })
            .collect();

        let mut report = CollisionReport::default();
        let mut frames: HashSet<usize> = HashSet::new();

        // self-collision (skip same-link and allowlisted pairs)
        for i in 0..placed.len() {
            for j in (i + 1)..placed.len() {
                let (fa, fb) = (placed[i].1, placed[j].1);
                if fa == fb {
                    continue;
                }
                let key = if fa < fb { (fa, fb) } else { (fb, fa) };
                if self.allowed.contains(&key) {
                    continue;
                }
                if intersects(&placed[i].0, &placed[j].0) {
                    report.self_pairs.push(key);
                    frames.insert(fa);
                    frames.insert(fb);
                }
            }
        }

        // world collision: ground half-space (n=+z, solid z ≤ ground_z) + boxes
        for (prim, frame) in &placed {
            let mut hit = false;
            if let Some(z) = self.scene.ground_z {
                hit |= prim_below_plane(prim, Vector3::new(0.0, 0.0, 1.0), z);
            }
            for (center, half) in &self.scene.boxes {
                let obb = Prim::Obb {
                    c: Vector3::new(center[0], center[1], center[2]),
                    r: Matrix3::identity(),
                    h: Vector3::new(
                        half[0] + self.margin,
                        half[1] + self.margin,
                        half[2] + self.margin,
                    ),
                };
                hit |= intersects(prim, &obb);
            }
            if hit {
                report.world_hits.push(*frame);
                frames.insert(*frame);
            }
        }

        report.self_pairs.sort_unstable();
        report.self_pairs.dedup();
        report.world_hits.sort_unstable();
        report.world_hits.dedup();
        let mut fv: Vec<usize> = frames.into_iter().collect();
        fv.sort_unstable();
        report.colliding_frames = fv;
        Ok(report)
    }

    /// Penetration contacts for every self-colliding link pair at `q`, via EPA.
    /// Additive to [`CollisionModel::query`] (which stays a pure boolean): for each
    /// overlapping, non-allowlisted, distinct-frame pair this returns
    /// `(frame_a, frame_b, contact)` with `frame_a < frame_b`. The contact's
    /// `witness` lies on the lower-indexed frame's collider (treated as `A`); its
    /// `normal`/`depth` are the minimum translation that separates them (move `A`
    /// by `-depth * normal`). Pairs whose overlap is a bare touch (zero depth) or
    /// whose EPA degenerates are omitted. Deterministically sorted by frame pair.
    ///
    /// World geometry (ground / boxes) is NOT included — it has no `frame_b`; use
    /// [`CollisionModel::query`]'s `world_hits` for those.
    pub fn contacts(&self, q: &[f64]) -> Result<Vec<(usize, usize, Contact)>, CollisionError> {
        if q.len() != self.model.ndof {
            return Err(CollisionError::Dim {
                expected: self.model.ndof,
                got: q.len(),
            });
        }
        if !q.iter().all(|x| x.is_finite()) {
            return Err(CollisionError::NonFinite);
        }

        let placed: Vec<(Prim, usize)> = self
            .model
            .collision
            .iter()
            .map(|g| {
                let world = fk_frame(&self.model, q, g.frame).0 * g.origin.0;
                let r = world.rotation.to_rotation_matrix().into_inner();
                let c = world.translation.vector;
                (prim_for(&g.shape, c, r, self.margin), g.frame)
            })
            .collect();

        let mut out: Vec<(usize, usize, Contact)> = Vec::new();
        for i in 0..placed.len() {
            for j in (i + 1)..placed.len() {
                let (fi, fj) = (placed[i].1, placed[j].1);
                if fi == fj {
                    continue;
                }
                let key = if fi < fj { (fi, fj) } else { (fj, fi) };
                if self.allowed.contains(&key) {
                    continue;
                }
                // Orient A = lower-indexed frame so the witness/normal convention is
                // tied to the canonical `(frame_a, frame_b)` order.
                let (pa, pb) = if fi < fj {
                    (&placed[i].0, &placed[j].0)
                } else {
                    (&placed[j].0, &placed[i].0)
                };
                if !intersects(pa, pb) {
                    continue;
                }
                if let Some(tetra) = gjk_tetra(pa, pb)
                    && let Some(contact) = epa(pa, pb, tetra)
                {
                    out.push((key.0, key.1, contact));
                }
            }
        }
        out.sort_by_key(|a| (a.0, a.1));
        Ok(out)
    }
}

impl SafetyCheck for CollisionModel {
    fn check(&self, q: &[f64]) -> Result<(), String> {
        match self.query(q) {
            Ok(r) if r.has_collision() => {
                Err(format!("collision at frames {:?}", r.colliding_frames))
            }
            Ok(_) => Ok(()),
            Err(e) => Err(e.to_string()),
        }
    }
}

/// Convert a parsed shape at world `(c, r)` (+margin) into an oriented primitive.
/// Cylinders become their tight OBB (Z-aligned local; conservative). A ConvexHull
/// is transformed vertex-by-vertex into world coords (`c + r·p`); its margin is
/// applied at GJK support time.
fn prim_for(shape: &CollisionShape, c: Vector3<f64>, r: Matrix3<f64>, margin: f64) -> Prim {
    match shape {
        CollisionShape::Box { half } => Prim::Obb {
            c,
            r,
            h: Vector3::new(half.x + margin, half.y + margin, half.z + margin),
        },
        CollisionShape::Sphere { radius } => Prim::Sphere {
            c,
            radius: radius + margin,
        },
        CollisionShape::Cylinder { radius, length } => Prim::Obb {
            c,
            r,
            h: Vector3::new(radius + margin, radius + margin, length / 2.0 + margin),
        },
        // Z-aligned capsule (URDF): core segment from `-length/2` to `+length/2`
        // along the shape's local Z (= world `r·ẑ`), swept by `radius`. The margin
        // inflates the swept radius (conservative), exactly as for a sphere.
        CollisionShape::Capsule { radius, length } => {
            let axis = Vector3::new(r[(0, 2)], r[(1, 2)], r[(2, 2)]); // world local-Z
            let half = length / 2.0;
            Prim::Capsule {
                a: c - axis * half,
                b: c + axis * half,
                radius: radius + margin,
            }
        }
        CollisionShape::ConvexHull { points } => Prim::Convex {
            points: points.iter().map(|p| c + r * p.coords).collect(),
            margin,
        },
    }
}

/// Seed the self-collision allowlist (adjacent or rigidly co-anchored links).
fn seed_allowlist(model: &Model) -> HashSet<(usize, usize)> {
    let mut allowed = HashSet::new();
    let geoms = &model.collision;
    let anchor = |f: usize| model.frames[f].anchor;
    for i in 0..geoms.len() {
        for j in (i + 1)..geoms.len() {
            let (fa, fb) = (geoms[i].frame, geoms[j].frame);
            if fa == fb {
                continue;
            }
            let adjacent = match (anchor(fa), anchor(fb)) {
                (Some(x), Some(y)) => {
                    x == y || model.parent[x] == Some(y) || model.parent[y] == Some(x)
                }
                // Two base/world-fixed frames are one rigid body → always allowed.
                (None, None) => true,
                // A movable-anchored frame vs the base is NOT auto-allowlisted: the
                // base shares no joint with a movable link in the adjacency sense (a
                // root link can fold back and genuinely strike the base), so that
                // pair must stay checked. Only the base itself / base-attached fixed
                // frames (the (None,None) arm) are co-located with the base.
                (None, Some(_)) | (Some(_), None) => false,
            };
            if adjacent {
                let key = if fa < fb { (fa, fb) } else { (fb, fa) };
                allowed.insert(key);
            }
        }
    }
    allowed
}

// ===== primitive intersection (pure nalgebra) =====

fn intersects(a: &Prim, b: &Prim) -> bool {
    match (a, b) {
        (
            Prim::Obb {
                c: ca,
                r: ra,
                h: ha,
            },
            Prim::Obb {
                c: cb,
                r: rb,
                h: hb,
            },
        ) => obb_obb(ca, ra, ha, cb, rb, hb),
        (
            Prim::Sphere { c, radius },
            Prim::Obb {
                c: cb,
                r: rb,
                h: hb,
            },
        )
        | (
            Prim::Obb {
                c: cb,
                r: rb,
                h: hb,
            },
            Prim::Sphere { c, radius },
        ) => sphere_obb(c, *radius, cb, rb, hb),
        (Prim::Sphere { c: a, radius: ra }, Prim::Sphere { c: b, radius: rb }) => {
            (a - b).norm() <= ra + rb
        }
        // ----- capsule (swept-sphere) closed forms -----
        // capsule ↔ sphere: point-to-segment distance ≤ r_cap + r_sphere.
        (Prim::Capsule { a: p, b: q, radius }, Prim::Sphere { c: sc, radius: sr })
        | (Prim::Sphere { c: sc, radius: sr }, Prim::Capsule { a: p, b: q, radius }) => {
            let rr = radius + sr;
            point_segment_dist2(sc, p, q) <= rr * rr
        }
        // capsule ↔ capsule: segment-to-segment distance ≤ r1 + r2.
        (
            Prim::Capsule {
                a: a1,
                b: b1,
                radius: r1,
            },
            Prim::Capsule {
                a: a2,
                b: b2,
                radius: r2,
            },
        ) => {
            let rr = r1 + r2;
            segment_segment_dist2(a1, b1, a2, b2) <= rr * rr
        }
        // Any remaining pair involving a convex hull (mesh) OR a capsule → GJK
        // boolean test. The capsule's exact swept-sphere support function makes
        // GJK exact for capsule ↔ box / capsule ↔ convex (and it is conservative —
        // never under-reports — on the iteration cap). The closed forms above stay
        // the gold path for primitive ↔ primitive; GJK is cross-validated against
        // `obb_obb` and the closed forms over many poses (see tests).
        (Prim::Convex { .. }, _)
        | (_, Prim::Convex { .. })
        | (Prim::Capsule { .. }, _)
        | (_, Prim::Capsule { .. }) => gjk_intersect(a, b),
    }
}

/// `true` if the primitive dips into the solid half-space `{ x : x·n ≤ d }`.
fn prim_below_plane(p: &Prim, n: Vector3<f64>, d: f64) -> bool {
    match p {
        Prim::Sphere { c, radius } => c.dot(&n) - radius <= d,
        Prim::Obb { c, r, h } => {
            let reach = h.x * r.column(0).dot(&n).abs()
                + h.y * r.column(1).dot(&n).abs()
                + h.z * r.column(2).dot(&n).abs();
            c.dot(&n) - reach <= d
        }
        // The convex hull dips into the half-space iff its lowest vertex does
        // (the support point along -n); margin lowers it by `margin·|n|`, |n|==1.
        Prim::Convex { points, margin } => points.iter().any(|p| p.dot(&n) - *margin <= d),
        // The capsule dips in iff its lowest core endpoint, lowered by `radius`
        // (|n|==1), reaches the half-space.
        Prim::Capsule { a, b, radius } => a.dot(&n).min(b.dot(&n)) - *radius <= d,
    }
}

/// Sphere ↔ oriented box: distance from the center to the box ≤ radius.
fn sphere_obb(
    sc: &Vector3<f64>,
    sr: f64,
    c: &Vector3<f64>,
    r: &Matrix3<f64>,
    h: &Vector3<f64>,
) -> bool {
    let d = r.transpose() * (sc - c); // center in box-local coords
    let cl = Vector3::new(
        d.x.clamp(-h.x, h.x),
        d.y.clamp(-h.y, h.y),
        d.z.clamp(-h.z, h.z),
    );
    (d - cl).norm_squared() <= sr * sr
}

/// Oriented-box ↔ oriented-box via the separating-axis theorem (Ericson, RTCD).
fn obb_obb(
    ca: &Vector3<f64>,
    ra: &Matrix3<f64>,
    ha: &Vector3<f64>,
    cb: &Vector3<f64>,
    rb: &Matrix3<f64>,
    hb: &Vector3<f64>,
) -> bool {
    const EPS: f64 = 1e-9;
    let a = [ha.x, ha.y, ha.z];
    let b = [hb.x, hb.y, hb.z];
    // R[i][j] = A_i · B_j  (and its abs, with epsilon for near-parallel edges)
    let mut rmat = [[0.0; 3]; 3];
    let mut absr = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rmat[i][j] = ra.column(i).dot(&rb.column(j));
            absr[i][j] = rmat[i][j].abs() + EPS;
        }
    }
    // translation in A's frame
    let tt = cb - ca;
    let t = [
        tt.dot(&ra.column(0)),
        tt.dot(&ra.column(1)),
        tt.dot(&ra.column(2)),
    ];
    // axes A0,A1,A2
    for i in 0..3 {
        let rb_ = b[0] * absr[i][0] + b[1] * absr[i][1] + b[2] * absr[i][2];
        if t[i].abs() > a[i] + rb_ {
            return false;
        }
    }
    // axes B0,B1,B2
    for j in 0..3 {
        let ra_ = a[0] * absr[0][j] + a[1] * absr[1][j] + a[2] * absr[2][j];
        let tj = t[0] * rmat[0][j] + t[1] * rmat[1][j] + t[2] * rmat[2][j];
        if tj.abs() > ra_ + b[j] {
            return false;
        }
    }
    // 9 cross-product axes A_i × B_j
    let cross = [
        // (i, j, ra, rb, t)
        (
            a[1] * absr[2][0] + a[2] * absr[1][0],
            b[1] * absr[0][2] + b[2] * absr[0][1],
            t[2] * rmat[1][0] - t[1] * rmat[2][0],
        ),
        (
            a[1] * absr[2][1] + a[2] * absr[1][1],
            b[0] * absr[0][2] + b[2] * absr[0][0],
            t[2] * rmat[1][1] - t[1] * rmat[2][1],
        ),
        (
            a[1] * absr[2][2] + a[2] * absr[1][2],
            b[0] * absr[0][1] + b[1] * absr[0][0],
            t[2] * rmat[1][2] - t[1] * rmat[2][2],
        ),
        (
            a[0] * absr[2][0] + a[2] * absr[0][0],
            b[1] * absr[1][2] + b[2] * absr[1][1],
            t[0] * rmat[2][0] - t[2] * rmat[0][0],
        ),
        (
            a[0] * absr[2][1] + a[2] * absr[0][1],
            b[0] * absr[1][2] + b[2] * absr[1][0],
            t[0] * rmat[2][1] - t[2] * rmat[0][1],
        ),
        (
            a[0] * absr[2][2] + a[2] * absr[0][2],
            b[0] * absr[1][1] + b[1] * absr[1][0],
            t[0] * rmat[2][2] - t[2] * rmat[0][2],
        ),
        (
            a[0] * absr[1][0] + a[1] * absr[0][0],
            b[1] * absr[2][2] + b[2] * absr[2][1],
            t[1] * rmat[0][0] - t[0] * rmat[1][0],
        ),
        (
            a[0] * absr[1][1] + a[1] * absr[0][1],
            b[0] * absr[2][2] + b[2] * absr[2][0],
            t[1] * rmat[0][1] - t[0] * rmat[1][1],
        ),
        (
            a[0] * absr[1][2] + a[1] * absr[0][2],
            b[0] * absr[2][1] + b[1] * absr[2][0],
            t[1] * rmat[0][2] - t[0] * rmat[1][2],
        ),
    ];
    for (ra_, rb_, tv) in cross {
        if tv.abs() > ra_ + rb_ {
            return false;
        }
    }
    true // no separating axis → intersecting
}

// ===== capsule (swept-sphere) closed-form distances (pure nalgebra) =====

/// Squared distance from point `p` to the segment `[a, b]`. A degenerate segment
/// (`a == b`) reduces to point-to-point. Exact closed form.
fn point_segment_dist2(p: &Vector3<f64>, a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    let ab = b - a;
    let ab2 = ab.norm_squared();
    if ab2 <= 1e-30 {
        return (p - a).norm_squared(); // degenerate segment = point
    }
    let t = ((p - a).dot(&ab) / ab2).clamp(0.0, 1.0);
    (p - (a + ab * t)).norm_squared()
}

/// Squared distance between segments `[p1, q1]` and `[p2, q2]` (Ericson, RTCD
/// §5.1.9, "ClosestPtSegmentSegment"). Handles either or both segments degenerate.
/// Exact closed form — the basis for the capsule ↔ capsule boolean.
fn segment_segment_dist2(
    p1: &Vector3<f64>,
    q1: &Vector3<f64>,
    p2: &Vector3<f64>,
    q2: &Vector3<f64>,
) -> f64 {
    const EPS: f64 = 1e-12;
    let d1 = q1 - p1; // direction of segment 1
    let d2 = q2 - p2; // direction of segment 2
    let r = p1 - p2;
    let a = d1.dot(&d1); // squared length of segment 1
    let e = d2.dot(&d2); // squared length of segment 2
    let f = d2.dot(&r);

    let (s, t);
    if a <= EPS && e <= EPS {
        // both segments are points
        return r.norm_squared();
    }
    if a <= EPS {
        // first segment is a point
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(&r);
        if e <= EPS {
            // second segment is a point
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            // general non-degenerate case
            let b = d1.dot(&d2);
            let denom = a * e - b * b; // always ≥ 0
            let s0 = if denom > EPS {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0 // parallel segments → pick an arbitrary point on segment 1
            };
            let t0 = (b * s0 + f) / e;
            // clamp t to [0,1] and recompute s for the clamped t
            if t0 < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t0 > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            } else {
                t = t0;
                s = s0;
            }
        }
    }
    let c1 = p1 + d1 * s;
    let c2 = p2 + d2 * t;
    (c1 - c2).norm_squared()
}

// ===== GJK boolean intersection (pure nalgebra) =====
//
// Tests whether two convex primitives overlap by deciding if the origin lies in
// their Minkowski difference A ⊖ B. Matches the rest of the module's BOOLEAN
// contract (no distance/penetration). Touching (origin on the boundary) is
// treated as a collision, consistent with the `<=` SAT. The simplex evolution is
// the well-known Muratori/Moran form; it is cross-validated against the validated
// `obb_obb` SAT over many random poses in the tests below.

const GJK_MAX_ITERS: usize = 64;

/// Support point of a primitive in direction `d` (need not be unit). For a sphere
/// / convex hull the margin is folded in here (Minkowski sum with a sphere).
fn prim_support(p: &Prim, d: &Vector3<f64>) -> Vector3<f64> {
    match p {
        Prim::Obb { c, r, h } => {
            // farthest corner: c + R · (sign(Rᵀd) ⊙ h)
            let dl = r.transpose() * d;
            let s = Vector3::new(
                if dl.x >= 0.0 { h.x } else { -h.x },
                if dl.y >= 0.0 { h.y } else { -h.y },
                if dl.z >= 0.0 { h.z } else { -h.z },
            );
            c + r * s
        }
        Prim::Sphere { c, radius } => {
            let n = d.norm();
            if n > 0.0 { c + d * (*radius / n) } else { *c }
        }
        Prim::Convex { points, margin } => {
            // farthest vertex along d, then push out by margin·d̂ (rounded hull)
            let mut best = points[0];
            let mut bestdot = best.dot(d);
            for &v in &points[1..] {
                let vd = v.dot(d);
                if vd > bestdot {
                    bestdot = vd;
                    best = v;
                }
            }
            let n = d.norm();
            if *margin > 0.0 && n > 0.0 {
                best + d * (*margin / n)
            } else {
                best
            }
        }
        // Swept sphere: farther segment endpoint along d, pushed out by radius·d̂.
        Prim::Capsule { a, b, radius } => {
            let base = if a.dot(d) >= b.dot(d) { *a } else { *b };
            let n = d.norm();
            if n > 0.0 {
                base + d * (*radius / n)
            } else {
                base
            }
        }
    }
}

/// A representative interior/center point, used only to seed the search direction.
fn prim_center(p: &Prim) -> Vector3<f64> {
    match p {
        Prim::Obb { c, .. } | Prim::Sphere { c, .. } => *c,
        Prim::Convex { points, .. } => points.iter().sum::<Vector3<f64>>() / points.len() as f64,
        Prim::Capsule { a, b, .. } => (a + b) * 0.5,
    }
}

/// Support of the Minkowski difference A ⊖ B in direction `d`.
#[inline]
fn mink_support(a: &Prim, b: &Prim, d: &Vector3<f64>) -> Vector3<f64> {
    prim_support(a, d) - prim_support(b, &(-d))
}

/// GJK boolean overlap test. A faithful port of the canonical
/// Muratori/Moran simplex evolution (`a` is always the most-recently-added
/// vertex; `simplex3`/`simplex4` keep a consistent winding so the tetra face
/// normals come out outward). Returns `true` on overlap or touching.
fn gjk_intersect(pa: &Prim, pb: &Prim) -> bool {
    // The Minkowski-difference support is symmetric in (pa,pb) for origin
    // containment; `s(dir)` below is A⊖B.
    let s = |dir: &Vector3<f64>| mink_support(pa, pb, dir);

    let mut search = prim_center(pa) - prim_center(pb);
    if search.norm_squared() < 1e-24 {
        search = Vector3::new(1.0, 0.0, 0.0);
    }

    // first two simplex points
    let mut c = s(&search);
    search = -c;
    let mut b = s(&search);
    if b.dot(&search) < 0.0 {
        return false; // no overlap
    }
    // perpendicular to edge cb, toward the origin
    let cb = c - b;
    search = cb.cross(&(-b)).cross(&cb);
    if search.norm_squared() < 1e-24 {
        // origin lies on the line cb → pick any axis not parallel to it
        search = cb.cross(&Vector3::new(1.0, 0.0, 0.0));
        if search.norm_squared() < 1e-24 {
            search = cb.cross(&Vector3::new(0.0, 0.0, -1.0));
        }
    }

    let mut a; // newest vertex
    let mut d = Vector3::zeros();
    let mut simp_dim = 2usize;

    for _ in 0..GJK_MAX_ITERS {
        a = s(&search);
        if a.dot(&search) < 0.0 {
            return false; // farthest point short of the origin → separated
        }
        simp_dim += 1;
        if simp_dim == 3 {
            simplex3(&mut a, &mut b, &mut c, &mut d, &mut simp_dim, &mut search);
        } else if simplex4(&mut a, &mut b, &mut c, &mut d, &mut simp_dim, &mut search) {
            return true;
        }
        if search.norm_squared() < 1e-24 {
            // origin on the simplex (touching) → treat as collision
            return true;
        }
    }
    true // no decision within the cap → assume overlap (conservative)
}

/// Triangle simplex update (Moran). `a` is newest. Reduces to an edge or prepares
/// a winding-consistent triangle base for the tetra step.
fn simplex3(
    a: &mut Vector3<f64>,
    b: &mut Vector3<f64>,
    c: &mut Vector3<f64>,
    d: &mut Vector3<f64>,
    simp_dim: &mut usize,
    search: &mut Vector3<f64>,
) {
    let n = (*b - *a).cross(&(*c - *a)); // triangle normal
    let ao = -*a;
    *simp_dim = 2;
    if (*b - *a).cross(&n).dot(&ao) > 0.0 {
        // closest to edge AB
        *c = *a;
        *search = (*b - *a).cross(&ao).cross(&(*b - *a));
        return;
    }
    if n.cross(&(*c - *a)).dot(&ao) > 0.0 {
        // closest to edge AC
        *b = *a;
        *search = (*c - *a).cross(&ao).cross(&(*c - *a));
        return;
    }
    *simp_dim = 3;
    if n.dot(&ao) > 0.0 {
        // above the triangle
        *d = *c;
        *c = *b;
        *b = *a;
        *search = n;
    } else {
        // below the triangle
        *d = *b;
        *b = *a;
        *search = -n;
    }
}

/// Tetrahedron simplex update (Moran). `a` is the tip (newest); BCD is the base.
/// Returns `true` iff the origin is enclosed.
fn simplex4(
    a: &mut Vector3<f64>,
    b: &mut Vector3<f64>,
    c: &mut Vector3<f64>,
    d: &mut Vector3<f64>,
    simp_dim: &mut usize,
    search: &mut Vector3<f64>,
) -> bool {
    let abc = (*b - *a).cross(&(*c - *a));
    let acd = (*c - *a).cross(&(*d - *a));
    let adb = (*d - *a).cross(&(*b - *a));
    let ao = -*a;
    *simp_dim = 3;
    if abc.dot(&ao) > 0.0 {
        *d = *c;
        *c = *b;
        *b = *a;
        *search = abc;
        return false;
    }
    if acd.dot(&ao) > 0.0 {
        *b = *a;
        *search = acd;
        return false;
    }
    if adb.dot(&ao) > 0.0 {
        *c = *d;
        *d = *b;
        *b = *a;
        *search = adb;
        return false;
    }
    true // enclosed
}

// ===== EPA penetration depth (pure nalgebra) =====
//
// On overlap, EPA (the Expanding Polytope Algorithm) recovers the minimum
// translation vector. It seeds from the GJK terminal tetrahedron enclosing the
// origin in the Minkowski difference `A ⊖ B`, then repeatedly pushes the closest
// boundary face outward (querying the support in that face's normal) until the
// face is on the true boundary. The face normal is the separation axis, its
// distance to the origin is the penetration depth, and a barycentric blend of the
// support points-on-A that built the face is the witness point on A. Works for any
// primitive pair through the shared `prim_support` (box/sphere/convex/capsule).

const EPA_MAX_ITERS: usize = 96;
/// Convergence threshold: stop once the support in the face normal advances the
/// boundary by less than this (in the Minkowski-difference metric, metres).
const EPA_TOL: f64 = 1e-10;

/// A Minkowski-difference vertex carrying the support point on `A` that produced
/// it, so EPA can map a boundary point back to a witness on `A`'s surface.
#[derive(Clone, Copy, Debug)]
struct Sv {
    /// `supportA(d) - supportB(-d)` — the vertex in the Minkowski difference.
    v: Vector3<f64>,
    /// `supportA(d)` — the contributing point on `A` (world coords).
    sa: Vector3<f64>,
}

/// Support of the Minkowski difference in direction `d`, tracking the point on A.
#[inline]
fn mink_sv(a: &Prim, b: &Prim, d: &Vector3<f64>) -> Sv {
    let pa = prim_support(a, d);
    let pb = prim_support(b, &(-d));
    Sv { v: pa - pb, sa: pa }
}

/// An EPA polytope face: a triangle (indices into the vertex list), its OUTWARD
/// unit normal, and the (non-negative) distance from the origin to its plane.
struct Face {
    i: usize,
    j: usize,
    k: usize,
    normal: Vector3<f64>,
    dist: f64,
}

/// Build a face from three vertices, orienting the normal OUTWARD (away from the
/// polytope interior reference `centroid`). `None` if the triangle is degenerate.
fn make_face(verts: &[Sv], i: usize, j: usize, k: usize, centroid: &Vector3<f64>) -> Option<Face> {
    let (vi, vj, vk) = (verts[i].v, verts[j].v, verts[k].v);
    let mut n = (vj - vi).cross(&(vk - vi));
    let nn = n.norm();
    if nn < 1e-18 {
        return None; // degenerate (collinear) triangle
    }
    n /= nn;
    // Orient outward and keep the winding consistent with the stored order so that
    // shared horizon edges cancel as reversed pairs during expansion.
    let (i, j, k, n) = if n.dot(&(vi - centroid)) < 0.0 {
        (i, k, j, -n)
    } else {
        (i, j, k, n)
    };
    let dist = n.dot(&verts[i].v).max(0.0);
    Some(Face {
        i,
        j,
        k,
        normal: n,
        dist,
    })
}

/// Push a horizon edge, cancelling it against an already-present reverse edge
/// (the shared edge of two visible faces is interior, not on the horizon).
fn add_edge(edges: &mut Vec<(usize, usize)>, a: usize, b: usize) {
    if let Some(pos) = edges.iter().position(|&(x, y)| x == b && y == a) {
        edges.swap_remove(pos);
    } else {
        edges.push((a, b));
    }
}

/// Barycentric coords `(u, v, w)` of `p` w.r.t. triangle `(a, b, c)` (Ericson).
/// `None` if the triangle is degenerate.
fn barycentric(
    p: &Vector3<f64>,
    a: &Vector3<f64>,
    b: &Vector3<f64>,
    c: &Vector3<f64>,
) -> Option<(f64, f64, f64)> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(&v0);
    let d01 = v0.dot(&v1);
    let d11 = v1.dot(&v1);
    let d20 = v2.dot(&v0);
    let d21 = v2.dot(&v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-30 {
        return None;
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    Some((1.0 - v - w, v, w))
}

/// Build a [`Contact`] from a converged closest face: project the origin onto the
/// face plane, recover its barycentric coords, and blend the per-vertex points-on-A
/// into a witness point on A.
fn contact_from_face(verts: &[Sv], f: &Face) -> Option<Contact> {
    let (vi, vj, vk) = (verts[f.i].v, verts[f.j].v, verts[f.k].v);
    let proj = f.normal * f.dist; // closest point on the face plane to the origin
    let (u, v, w) = barycentric(&proj, &vi, &vj, &vk)?;
    let witness = verts[f.i].sa * u + verts[f.j].sa * v + verts[f.k].sa * w;
    Some(Contact {
        normal: f.normal,
        depth: f.dist,
        witness,
    })
}

/// GJK that, on overlap, returns the terminal tetrahedron enclosing the origin in
/// `A ⊖ B` (each vertex tracks its point-on-A for EPA). `None` when separated or
/// only touching (the boundary case is ill-conditioned for EPA and not a true
/// penetration). Mirrors [`gjk_intersect`]'s simplex evolution on [`Sv`] vertices.
fn gjk_tetra(pa: &Prim, pb: &Prim) -> Option<[Sv; 4]> {
    let s = |dir: &Vector3<f64>| mink_sv(pa, pb, dir);

    let mut search = prim_center(pa) - prim_center(pb);
    if search.norm_squared() < 1e-24 {
        search = Vector3::new(1.0, 0.0, 0.0);
    }
    let mut c = s(&search);
    search = -c.v;
    let mut b = s(&search);
    if b.v.dot(&search) < 0.0 {
        return None;
    }
    let cb = c.v - b.v;
    search = cb.cross(&(-b.v)).cross(&cb);
    if search.norm_squared() < 1e-24 {
        search = cb.cross(&Vector3::new(1.0, 0.0, 0.0));
        if search.norm_squared() < 1e-24 {
            search = cb.cross(&Vector3::new(0.0, 0.0, -1.0));
        }
    }

    let mut a;
    let mut d = Sv {
        v: Vector3::zeros(),
        sa: Vector3::zeros(),
    };
    let mut simp_dim = 2usize;
    for _ in 0..GJK_MAX_ITERS {
        a = s(&search);
        if a.v.dot(&search) < 0.0 {
            return None; // farthest point short of the origin → separated
        }
        simp_dim += 1;
        if simp_dim == 3 {
            simplex3_sv(&mut a, &mut b, &mut c, &mut d, &mut simp_dim, &mut search);
        } else if simplex4_sv(&mut a, &mut b, &mut c, &mut d, &mut simp_dim, &mut search) {
            return Some([a, b, c, d]);
        }
        if search.norm_squared() < 1e-24 {
            return None; // origin on the simplex boundary (touching) → no EPA
        }
    }
    None
}

/// [`simplex3`] specialised to [`Sv`] vertices (decisions use `.v`; `.sa` rides along).
fn simplex3_sv(
    a: &mut Sv,
    b: &mut Sv,
    c: &mut Sv,
    d: &mut Sv,
    simp_dim: &mut usize,
    search: &mut Vector3<f64>,
) {
    let n = (b.v - a.v).cross(&(c.v - a.v));
    let ao = -a.v;
    *simp_dim = 2;
    if (b.v - a.v).cross(&n).dot(&ao) > 0.0 {
        *c = *a;
        *search = (b.v - a.v).cross(&ao).cross(&(b.v - a.v));
        return;
    }
    if n.cross(&(c.v - a.v)).dot(&ao) > 0.0 {
        *b = *a;
        *search = (c.v - a.v).cross(&ao).cross(&(c.v - a.v));
        return;
    }
    *simp_dim = 3;
    if n.dot(&ao) > 0.0 {
        *d = *c;
        *c = *b;
        *b = *a;
        *search = n;
    } else {
        *d = *b;
        *b = *a;
        *search = -n;
    }
}

/// [`simplex4`] specialised to [`Sv`] vertices. Returns `true` iff origin enclosed.
fn simplex4_sv(
    a: &mut Sv,
    b: &mut Sv,
    c: &mut Sv,
    d: &mut Sv,
    simp_dim: &mut usize,
    search: &mut Vector3<f64>,
) -> bool {
    let abc = (b.v - a.v).cross(&(c.v - a.v));
    let acd = (c.v - a.v).cross(&(d.v - a.v));
    let adb = (d.v - a.v).cross(&(b.v - a.v));
    let ao = -a.v;
    *simp_dim = 3;
    if abc.dot(&ao) > 0.0 {
        *d = *c;
        *c = *b;
        *b = *a;
        *search = abc;
        return false;
    }
    if acd.dot(&ao) > 0.0 {
        *b = *a;
        *search = acd;
        return false;
    }
    if adb.dot(&ao) > 0.0 {
        *c = *d;
        *d = *b;
        *b = *a;
        *search = adb;
        return false;
    }
    true
}

/// EPA on the GJK terminal tetra `tetra` (enclosing the origin in `A ⊖ B`).
/// Returns the penetration [`Contact`] (normal/depth/witness), or `None` if the
/// polytope degenerates. Convention: `normal` is the OUTWARD Minkowski-difference
/// normal at the closest boundary point, so `A` and `B` separate by translating
/// `A` by `-depth * normal`.
fn epa(pa: &Prim, pb: &Prim, tetra: [Sv; 4]) -> Option<Contact> {
    let s = |dir: &Vector3<f64>| mink_sv(pa, pb, dir);
    let mut verts: Vec<Sv> = tetra.to_vec();
    // A fixed interior reference: the seed tetra's centroid stays inside the
    // (only-outward-growing) polytope, so it orients every face's normal.
    let centroid = (verts[0].v + verts[1].v + verts[2].v + verts[3].v) / 4.0;

    let mut faces: Vec<Face> = Vec::new();
    for (i, j, k) in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
        if let Some(f) = make_face(&verts, i, j, k, &centroid) {
            faces.push(f);
        }
    }
    if faces.len() < 4 {
        return None; // degenerate seed tetra
    }

    for _ in 0..EPA_MAX_ITERS {
        // closest face to the origin
        let ci = (0..faces.len()).min_by(|&x, &y| {
            faces[x]
                .dist
                .partial_cmp(&faces[y].dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        let normal = faces[ci].normal;
        let dist = faces[ci].dist;

        // farthest support along the closest-face normal
        let p = s(&normal);
        if p.v.dot(&normal) - dist < EPA_TOL {
            return contact_from_face(&verts, &faces[ci]); // converged
        }

        // expand: drop every face the new point can see; stitch the horizon to it
        let np = p.v;
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        let mut keep: Vec<Face> = Vec::with_capacity(faces.len());
        for f in faces.drain(..) {
            if f.normal.dot(&(np - verts[f.i].v)) > 1e-12 {
                add_edge(&mut horizon, f.i, f.j);
                add_edge(&mut horizon, f.j, f.k);
                add_edge(&mut horizon, f.k, f.i);
            } else {
                keep.push(f);
            }
        }
        faces = keep;
        let ni = verts.len();
        verts.push(p);
        for (e0, e1) in horizon {
            if let Some(f) = make_face(&verts, e0, e1, ni, &centroid) {
                faces.push(f);
            }
        }
        if faces.is_empty() {
            return None;
        }
    }

    // iteration cap: return the best face we have (best-effort, still a valid lower
    // bound on the depth → conservative).
    let ci = (0..faces.len()).min_by(|&x, &y| {
        faces[x]
            .dist
            .partial_cmp(&faces[y].dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    contact_from_face(&verts, &faces[ci])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn model(name: &str) -> Arc<Model> {
        Arc::new(
            Model::from_urdf(Path::new(&format!(
                "{}/../../oracle/fixtures/robots/{}",
                env!("CARGO_MANIFEST_DIR"),
                name
            )))
            .unwrap(),
        )
    }

    #[test]
    fn partially_covered_frame_is_reported_uncovered() {
        // l1 has a box collider AND a dropped mesh; base has none. Both must count
        // as not-fully-covered (the OLD count would have been 1 = base only).
        let m = model("collide_mixed.urdf");
        assert_eq!(
            m.dropped_collider_frames.len(),
            1,
            "l1's dropped mesh tracked"
        );
        let cm = CollisionModel::new(m, WorldScene::new(), 0.0);
        assert_eq!(cm.num_colliders(), 1, "l1's box is the only primitive");
        assert_eq!(
            cm.uncovered_frames(),
            2,
            "base (no collider) + l1 (partial: mesh dropped) both uncovered"
        );
    }

    #[test]
    fn obb_sat_closed_form() {
        let i = Matrix3::identity();
        let h = Vector3::new(0.5, 0.5, 0.5);
        // overlapping unit boxes 0.5 apart
        assert!(obb_obb(
            &Vector3::zeros(),
            &i,
            &h,
            &Vector3::new(0.5, 0.0, 0.0),
            &i,
            &h
        ));
        // separated by 1.2 (> 1.0 combined) → clear
        assert!(!obb_obb(
            &Vector3::zeros(),
            &i,
            &h,
            &Vector3::new(1.2, 0.0, 0.0),
            &i,
            &h
        ));
    }

    #[test]
    fn sphere_box_closed_form() {
        let i = Matrix3::identity();
        let h = Vector3::new(0.5, 0.5, 0.5);
        assert!(sphere_obb(
            &Vector3::new(0.8, 0.0, 0.0),
            0.4,
            &Vector3::zeros(),
            &i,
            &h
        )); // 0.3 gap < r
        assert!(!sphere_obb(
            &Vector3::new(1.2, 0.0, 0.0),
            0.4,
            &Vector3::zeros(),
            &i,
            &h
        )); // 0.7 gap > r
    }

    #[test]
    fn allowlist_excludes_nonadjacent() {
        let m = model("collide_arm.urdf");
        let cm = CollisionModel::new(m.clone(), WorldScene::new(), 0.0);
        let f = |name: &str| m.frame_id(name).unwrap();
        assert!(cm.allowlisted(f("l1"), f("l2")));
        assert!(cm.allowlisted(f("l2"), f("l3")));
        assert!(
            !cm.allowlisted(f("l1"), f("l3")),
            "non-adjacent must NOT be auto-allowed"
        );
    }

    #[test]
    fn folded_self_collides_extended_clear() {
        let m = model("collide_arm.urdf");
        let cm = CollisionModel::new(m, WorldScene::new(), 0.0);
        let clear = cm.query(&[0.0, 0.0, 0.0]).unwrap();
        assert!(!clear.has_collision(), "extended must be clear: {clear:?}");
        let folded = cm
            .query(&[0.0, std::f64::consts::PI, std::f64::consts::PI])
            .unwrap();
        assert!(folded.has_collision(), "folded must self-collide");
        assert!(!folded.self_pairs.is_empty());
    }

    #[test]
    fn ground_collision() {
        let m = model("collide_arm.urdf");
        // ground just below the base so the upright arm (l1 bottom at z=0) is clear
        let cm = CollisionModel::new(m, WorldScene::new().with_ground(-0.05), 0.0);
        assert!(cm.query(&[0.0, 0.0, 0.0]).unwrap().world_hits.is_empty());
        let down = cm.query(&[std::f64::consts::PI, 0.0, 0.0]).unwrap();
        assert!(!down.world_hits.is_empty(), "arm below ground must hit");
    }

    #[test]
    fn box_obstacle_and_fk_consistency() {
        let m = model("collide_shapes.urdf");
        // sphere collider sits at the j1 frame (world origin); a box there → hit
        let cm = CollisionModel::new(
            m.clone(),
            WorldScene::new().add_box([0.0, 0.0, 0.0], [0.2, 0.2, 0.2]),
            0.0,
        );
        assert!(cm.query(&[0.0]).unwrap().has_collision());
        // far box → clear (FK consistency: collider really is near origin)
        let cm2 = CollisionModel::new(
            m,
            WorldScene::new().add_box([5.0, 0.0, 0.0], [0.2, 0.2, 0.2]),
            0.0,
        );
        assert!(!cm2.query(&[0.0]).unwrap().has_collision());
    }

    #[test]
    fn dim_and_finite_guards() {
        let m = model("collide_arm.urdf");
        let cm = CollisionModel::new(m, WorldScene::new(), 0.0);
        assert!(matches!(
            cm.query(&[0.0, 0.0]),
            Err(CollisionError::Dim { .. })
        ));
        assert!(matches!(
            cm.query(&[0.0, f64::NAN, 0.0]),
            Err(CollisionError::NonFinite)
        ));
    }

    // ---- D2: edge-edge (cross-product) separating axes ----

    /// Two long thin rods crossing at a genuine 3-D skew, chosen so that ALL six
    /// face axes overlap and the ONLY separating axis is the edge-edge axis
    /// `A0 × B0`. Rod A lies along x; rod B's long axis is `(0,0.6,0.8)` with its
    /// cross-section rotated 45° so the cross axis `(0,-0.8,0.6)` coincides with no
    /// face normal. The combined reach along that axis is ≈0.2814 (per H·0.6), so
    /// the contact threshold is H≈0.469.
    fn skew_rods(h: f64) -> bool {
        let c = std::f64::consts::FRAC_1_SQRT_2;
        let s8 = 0.8 * c; // 0.565685…
        let s6 = 0.6 * c; // 0.424264…
        let i = Matrix3::identity();
        let rb = Matrix3::from_columns(&[
            Vector3::new(0.0, 0.6, 0.8),
            Vector3::new(c, -s8, s6),
            Vector3::new(c, s8, -s6),
        ]);
        let ext = Vector3::new(2.0, 0.1, 0.1);
        obb_obb(
            &Vector3::zeros(),
            &i,
            &ext,
            &Vector3::new(0.0, 0.0, h),
            &rb,
            &ext,
        )
    }

    #[test]
    fn edge_edge_axis_separates() {
        // H = 0.6 > 0.469 → the cross axis A0×B0 separates → NO collision. If the
        // cross-product axes were missing/under-reported, all 6 face axes overlap
        // and obb_obb would WRONGLY report a collision here.
        assert!(
            !skew_rods(0.6),
            "skew rods separated only along an edge-edge axis must report clear"
        );
    }

    #[test]
    fn edge_edge_axis_contacts() {
        // H = 0.2 < 0.469 → no axis separates → the rods interpenetrate. Confirms
        // the edge-edge axis term does not spuriously over-separate (false clear).
        assert!(
            skew_rods(0.2),
            "skew rods inside the edge-edge contact threshold must collide"
        );
    }

    #[test]
    fn parallel_edges_degenerate_no_false_separation() {
        // Identically-oriented overlapping boxes → every A_i × B_j is ~0 (parallel
        // edges, degenerate axes). The EPS guard on `absr` must keep those axes from
        // falsely separating: overlap MUST be reported as a collision.
        let i = Matrix3::identity();
        let h = Vector3::new(0.5, 0.5, 0.5);
        assert!(
            obb_obb(
                &Vector3::zeros(),
                &i,
                &h,
                &Vector3::new(0.3, 0.0, 0.0),
                &i,
                &h
            ),
            "overlapping parallel-edge boxes must not be split by a degenerate axis"
        );
    }

    // ---- D3: base is not auto-allowlisted against a movable root link ----

    #[test]
    fn base_not_allowlisted_against_movable_root() {
        // Give the (collider-less) base its own collider, then confirm the base is
        // NOT auto-allowlisted against l1 — a root link folding back onto the base
        // is a real self-collision that must stay checked. Adjacent movable links
        // (l1-l2) remain allowlisted; non-adjacent (l1-l3) remain checked.
        let mut m = Model::from_urdf(Path::new(&format!(
            "{}/../../oracle/fixtures/robots/collide_arm.urdf",
            env!("CARGO_MANIFEST_DIR")
        )))
        .unwrap();
        let base = m.frame_id("base").unwrap();
        let l1 = m.frame_id("l1").unwrap();
        let l2 = m.frame_id("l2").unwrap();
        let l3 = m.frame_id("l3").unwrap();
        // clone an existing box collider and re-anchor it to the base frame
        let mut g = m.collision[0].clone();
        g.frame = base;
        m.collision.push(g);
        let cm = CollisionModel::new(Arc::new(m), WorldScene::new(), 0.0);
        assert!(
            !cm.allowlisted(base, l1),
            "base must NOT be auto-allowlisted against a movable root link"
        );
        assert!(
            cm.allowlisted(l1, l2),
            "adjacent movable links stay allowlisted"
        );
        assert!(!cm.allowlisted(l1, l3), "non-adjacent links stay checked");
    }

    // ---- mesh / convex-hull (GJK) coverage + cross-validation ----

    /// A deterministic splitmix64 stream → f64 in [0,1) (no `rand` crate).
    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.f()
        }
    }

    fn rand_rot(rng: &mut Rng) -> Matrix3<f64> {
        let axis = Vector3::new(
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        );
        let axis = if axis.norm() < 1e-6 {
            Vector3::new(0.0, 0.0, 1.0)
        } else {
            axis.normalize()
        };
        let ang = rng.range(-std::f64::consts::PI, std::f64::consts::PI);
        nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), ang).into_inner()
    }

    /// The 8 corners of an oriented box, in world coordinates.
    fn cube_points(c: Vector3<f64>, r: Matrix3<f64>, h: Vector3<f64>) -> Vec<Vector3<f64>> {
        let mut p = Vec::with_capacity(8);
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    p.push(c + r * Vector3::new(sx * h.x, sy * h.y, sz * h.z));
                }
            }
        }
        p
    }

    /// A convex-hull primitive shaped as the 8 corners of an oriented box.
    fn cube_prim(c: Vector3<f64>, r: Matrix3<f64>, h: Vector3<f64>) -> Prim {
        Prim::Convex {
            points: cube_points(c, r, h),
            margin: 0.0,
        }
    }

    #[test]
    fn convex_cube_matches_obb_overlap_and_separation() {
        let i = Matrix3::identity();
        let h = Vector3::new(0.5, 0.5, 0.5);
        let obb = Prim::Obb {
            c: Vector3::zeros(),
            r: i,
            h,
        };
        // overlapping (0.5 apart): convex-cube must collide with the OBB, and the
        // result must match obb_obb on the equivalent boxes.
        let near = cube_prim(Vector3::new(0.5, 0.0, 0.0), i, h);
        assert!(
            intersects(&near, &obb),
            "overlapping convex vs obb must collide"
        );
        assert!(intersects(&obb, &near), "symmetry");
        // separated (1.2 apart): must be clear, matching the SAT.
        let far = cube_prim(Vector3::new(1.2, 0.0, 0.0), i, h);
        assert!(
            !intersects(&far, &obb),
            "separated convex vs obb must be clear"
        );
    }

    #[test]
    fn gjk_matches_obb_sat_randomized() {
        // Cross-validate GJK against the validated OBB SAT over many random poses,
        // using ONLY boundary-free placements so the boolean verdicts are
        // unambiguous: (1) B's center placed INSIDE A → guaranteed overlap, and
        // (2) B pushed beyond A along world-x with a positive gap → guaranteed
        // separation (x is a separating axis). Any transcription error in the GJK
        // simplex routines would surface as a mismatch here.
        let mut rng = Rng(0xCAFE_F00D_1234_5678);
        let ident = Matrix3::identity();
        for _ in 0..400 {
            let ha = Vector3::new(
                rng.range(0.1, 0.5),
                rng.range(0.1, 0.5),
                rng.range(0.1, 0.5),
            );
            let hb = Vector3::new(
                rng.range(0.1, 0.5),
                rng.range(0.1, 0.5),
                rng.range(0.1, 0.5),
            );
            let rb = rand_rot(&mut rng);
            let obb_a = Prim::Obb {
                c: Vector3::zeros(),
                r: ident,
                h: ha,
            };

            // --- guaranteed OVERLAP: center of B sits inside A ---
            let c_in = Vector3::new(
                rng.range(-ha.x, ha.x),
                rng.range(-ha.y, ha.y),
                rng.range(-ha.z, ha.z),
            );
            let conv_b = cube_prim(c_in, rb, hb);
            let conv_a = cube_prim(Vector3::zeros(), ident, ha);
            let obb_b = Prim::Obb {
                c: c_in,
                r: rb,
                h: hb,
            };
            // SAT ground truth
            assert!(
                obb_obb(&Vector3::zeros(), &ident, &ha, &c_in, &rb, &hb),
                "sanity: B's center inside A must overlap per SAT"
            );
            assert!(intersects(&conv_b, &obb_a), "convex(B) vs obb(A) overlap");
            assert!(
                intersects(&conv_a, &conv_b),
                "convex(A) vs convex(B) overlap"
            );
            assert!(intersects(&conv_a, &obb_b), "convex(A) vs obb(B) overlap");

            // --- guaranteed SEPARATION: push B beyond A along world x ---
            let proj_bx =
                hb.x * rb[(0, 0)].abs() + hb.y * rb[(0, 1)].abs() + hb.z * rb[(0, 2)].abs();
            let cx = ha.x + proj_bx + 0.05; // 5 cm gap → x is a separating axis
            let c_out = Vector3::new(cx, rng.range(-0.2, 0.2), rng.range(-0.2, 0.2));
            let conv_out = cube_prim(c_out, rb, hb);
            let obb_out = Prim::Obb {
                c: c_out,
                r: rb,
                h: hb,
            };
            assert!(
                !obb_obb(&Vector3::zeros(), &ident, &ha, &c_out, &rb, &hb),
                "sanity: gapped B must be separated per SAT"
            );
            assert!(!intersects(&conv_out, &obb_a), "convex(B) vs obb(A) clear");
            assert!(
                !intersects(&conv_a, &conv_out),
                "convex(A) vs convex(B) clear"
            );
            assert!(!intersects(&conv_a, &obb_out), "convex(A) vs obb(B) clear");
        }
    }

    #[test]
    fn convex_vs_sphere_matches_sphere_obb() {
        // A convex cube is geometrically the same body as the equivalent OBB, so
        // GJK(convex, sphere) must agree with the closed-form sphere_obb.
        let i = Matrix3::identity();
        let h = Vector3::new(0.5, 0.5, 0.5);
        let cube = cube_prim(Vector3::zeros(), i, h);
        for &(sc, sr) in &[
            ([0.8, 0.0, 0.0], 0.4), // 0.3 gap < r → hit
            ([1.2, 0.0, 0.0], 0.4), // 0.7 gap > r → clear
            ([0.0, 0.0, 0.0], 0.1), // inside
            ([0.9, 0.9, 0.0], 0.3), // near a corner, clear
        ] {
            let sphere = Prim::Sphere {
                c: Vector3::new(sc[0], sc[1], sc[2]),
                radius: sr,
            };
            let want = sphere_obb(
                &Vector3::new(sc[0], sc[1], sc[2]),
                sr,
                &Vector3::zeros(),
                &i,
                &h,
            );
            assert_eq!(
                intersects(&cube, &sphere),
                want,
                "convex/sphere must match sphere_obb for {sc:?} r={sr}"
            );
        }
    }

    #[test]
    fn convex_margin_inflates() {
        // With a 5 cm gap the cubes are clear; a margin >= half the gap on the
        // convex side must close it (Minkowski-sum inflation in the support fn).
        let i = Matrix3::identity();
        let h = Vector3::new(0.5, 0.5, 0.5);
        let obb = Prim::Obb {
            c: Vector3::new(1.05, 0.0, 0.0),
            r: i,
            h,
        };
        let bare = Prim::Convex {
            points: cube_points(Vector3::zeros(), i, h),
            margin: 0.0,
        };
        assert!(!intersects(&bare, &obb), "0.05 gap, no margin → clear");
        let inflated = Prim::Convex {
            points: cube_points(Vector3::zeros(), i, h),
            margin: 0.06,
        };
        assert!(intersects(&inflated, &obb), "margin must close the gap");
    }

    #[test]
    fn convex_below_plane() {
        // cube spanning z ∈ [-0.5, 0.5]: dips into a half-space at z ≤ 0.1, clear
        // below z ≤ -1.0. Validates the convex arm of prim_below_plane.
        let cube = cube_prim(
            Vector3::zeros(),
            Matrix3::identity(),
            Vector3::new(0.5, 0.5, 0.5),
        );
        let zup = Vector3::new(0.0, 0.0, 1.0);
        assert!(prim_below_plane(&cube, zup, 0.1));
        assert!(!prim_below_plane(&cube, zup, -1.0));
    }

    #[test]
    fn mesh_link_is_now_covered_and_collides() {
        // collide_mesh.urdf: l1 carries unit_cube.stl → a ConvexHull collider. The
        // mesh frame is now COVERED (only the collider-less base is uncovered), and
        // the convex collider participates in real world-collision queries via GJK.
        let m = model("collide_mesh.urdf");
        assert!(
            m.dropped_collider_frames.is_empty(),
            "loaded mesh must not be dropped"
        );
        let cm = CollisionModel::new(m.clone(), WorldScene::new(), 0.0);
        assert_eq!(cm.num_colliders(), 1, "the convex-hull collider");
        assert_eq!(
            cm.uncovered_frames(),
            1,
            "only `base` is uncovered now; l1's mesh is covered"
        );
        // a world box overlapping the cube (centered at origin at q=0) → collision
        let hit = CollisionModel::new(
            m.clone(),
            WorldScene::new().add_box([0.0, 0.0, 0.0], [0.2, 0.2, 0.2]),
            0.0,
        );
        assert!(
            hit.query(&[0.0]).unwrap().has_collision(),
            "convex hull (mesh) must collide with an overlapping world box"
        );
        // a distant box → clear (FK consistency through the convex path)
        let clear = CollisionModel::new(
            m.clone(),
            WorldScene::new().add_box([5.0, 0.0, 0.0], [0.2, 0.2, 0.2]),
            0.0,
        );
        assert!(!clear.query(&[0.0]).unwrap().has_collision());
        // ground intersecting the cube (z ≤ 0.1) → world hit; far below → clear
        let ground = CollisionModel::new(m.clone(), WorldScene::new().with_ground(0.1), 0.0);
        assert!(!ground.query(&[0.0]).unwrap().world_hits.is_empty());
        let deep = CollisionModel::new(m, WorldScene::new().with_ground(-2.0), 0.0);
        assert!(deep.query(&[0.0]).unwrap().world_hits.is_empty());
    }

    // ---- B13: degenerate world-box extents are sanitized ----

    #[test]
    fn add_box_sanitizes_degenerate_extents() {
        let m = model("collide_shapes.urdf");
        // A non-finite extent must not turn the SAT into a blanket "collide with
        // everything": it is clamped to a finite zero box, so a far box stays clear.
        let cm = CollisionModel::new(
            m.clone(),
            WorldScene::new().add_box([10.0, 10.0, 10.0], [f64::NAN, 0.2, 0.2]),
            0.0,
        );
        assert!(
            !cm.query(&[0.0]).unwrap().has_collision(),
            "a NaN-extent box far from the arm must not register a collision"
        );
        // Negative extents clamp to zero (and must not panic via f64::clamp bounds).
        let cm2 = CollisionModel::new(
            m,
            WorldScene::new().add_box([10.0, 0.0, 0.0], [-5.0, -5.0, -5.0]),
            0.0,
        );
        assert!(
            !cm2.query(&[0.0]).unwrap().has_collision(),
            "negative extents must clamp to a degenerate, non-colliding box"
        );
    }

    // ===== capsule (swept-sphere) closed-form cross-validation =====

    #[test]
    fn capsule_sphere_closed_form() {
        // Capsule core along x from (-0.5,0,0) to (0.5,0,0), radius 0.1.
        let cap = Prim::Capsule {
            a: Vector3::new(-0.5, 0.0, 0.0),
            b: Vector3::new(0.5, 0.0, 0.0),
            radius: 0.1,
        };
        // sphere above the middle: surface gap = z - (0.1 + r_s).
        let clear = Prim::Sphere {
            c: Vector3::new(0.0, 0.0, 0.25),
            radius: 0.1,
        }; // dist 0.25 > 0.2 → clear
        assert!(!intersects(&cap, &clear));
        let hit = Prim::Sphere {
            c: Vector3::new(0.0, 0.0, 0.15),
            radius: 0.1,
        }; // dist 0.15 < 0.2 → hit
        assert!(intersects(&cap, &hit));
        assert!(intersects(&hit, &cap), "symmetry");
        // beyond the cap end (clamped to the endpoint): point-to-(0.5,0,0) distance
        // probes the hemispherical cap. 0.2 == r1+r2 → touching counts (`<=`).
        let touch = Prim::Sphere {
            c: Vector3::new(0.7, 0.0, 0.0),
            radius: 0.1,
        };
        assert!(intersects(&cap, &touch), "0.2 == r1+r2 touch must collide");
        let past = Prim::Sphere {
            c: Vector3::new(0.71, 0.0, 0.0),
            radius: 0.1,
        };
        assert!(!intersects(&cap, &past), "0.21 > 0.2 must be clear");
    }

    #[test]
    fn capsule_capsule_closed_form() {
        // Reference x-capsule at z=0, radius 0.1.
        let a = Prim::Capsule {
            a: Vector3::new(-0.5, 0.0, 0.0),
            b: Vector3::new(0.5, 0.0, 0.0),
            radius: 0.1,
        };
        // parallel x-capsule lifted by 0.15: segment-segment dist 0.15 < 0.2 → hit.
        let near = Prim::Capsule {
            a: Vector3::new(-0.5, 0.0, 0.15),
            b: Vector3::new(0.5, 0.0, 0.15),
            radius: 0.1,
        };
        assert!(intersects(&a, &near));
        // lifted by 0.25: dist 0.25 > 0.2 → clear.
        let far = Prim::Capsule {
            a: Vector3::new(-0.5, 0.0, 0.25),
            b: Vector3::new(0.5, 0.0, 0.25),
            radius: 0.1,
        };
        assert!(!intersects(&a, &far));
        // perpendicular (y-axis) capsule crossing over the origin at z=0.18: the
        // segment-segment closest distance is the pure z gap 0.18 < 0.2 → hit. A
        // naive endpoint-only distance (≈0.52) would WRONGLY report clear, so this
        // exercises the interior-interior branch of segment_segment_dist2.
        let perp = Prim::Capsule {
            a: Vector3::new(0.0, -0.5, 0.18),
            b: Vector3::new(0.0, 0.5, 0.18),
            radius: 0.1,
        };
        assert!(intersects(&a, &perp));
    }

    #[test]
    fn capsule_halfspace_closed_form() {
        // vertical capsule core z in [-0.2,0.2], radius 0.1 → lowest point z=-0.3.
        let cap = Prim::Capsule {
            a: Vector3::new(0.0, 0.0, -0.2),
            b: Vector3::new(0.0, 0.0, 0.2),
            radius: 0.1,
        };
        let zup = Vector3::new(0.0, 0.0, 1.0);
        assert!(
            prim_below_plane(&cap, zup, -0.25),
            "cap bottom -0.3 dips into z<=-0.25"
        );
        assert!(
            !prim_below_plane(&cap, zup, -0.35),
            "cap bottom -0.3 clears z<=-0.35"
        );
    }

    #[test]
    fn capsule_obb_via_gjk_known_distance() {
        // Horizontal capsule (core along x, radius 0.1) at height z=d, spanning over
        // a box of half 0.3 at the origin. The segment passes above the +z face, so
        // the closest distance is d - 0.3; collide iff d - 0.3 <= 0.1 → d <= 0.4.
        let box_ = Prim::Obb {
            c: Vector3::zeros(),
            r: Matrix3::identity(),
            h: Vector3::new(0.3, 0.3, 0.3),
        };
        let cap = |d: f64| Prim::Capsule {
            a: Vector3::new(-0.5, 0.0, d),
            b: Vector3::new(0.5, 0.0, d),
            radius: 0.1,
        };
        assert!(intersects(&cap(0.35), &box_), "0.05 gap < r → collide");
        assert!(intersects(&box_, &cap(0.35)), "symmetry");
        assert!(!intersects(&cap(0.45), &box_), "0.15 gap > r → clear");
    }

    #[test]
    fn capsule_convex_matches_capsule_obb() {
        // GJK(capsule, convex-cube) must agree with GJK(capsule, the equivalent OBB)
        // — the convex hull is the same body, so the swept-sphere verdicts must match
        // across the contact threshold.
        let i = Matrix3::identity();
        let h = Vector3::new(0.3, 0.3, 0.3);
        let cube = cube_prim(Vector3::zeros(), i, h);
        let obb = Prim::Obb {
            c: Vector3::zeros(),
            r: i,
            h,
        };
        for &d in &[0.35_f64, 0.45, 0.2] {
            let cap = Prim::Capsule {
                a: Vector3::new(-0.5, 0.0, d),
                b: Vector3::new(0.5, 0.0, d),
                radius: 0.1,
            };
            assert_eq!(
                intersects(&cap, &cube),
                intersects(&cap, &obb),
                "capsule/convex vs capsule/obb must agree at d={d}"
            );
        }
    }

    #[test]
    fn capsule_link_is_now_covered_and_collides() {
        // collide_capsule.urdf: l1 carries a <capsule> (radius 0.1, length 0.4). The
        // capsule frame is now COVERED (only the collider-less base is uncovered) and
        // participates in real world-collision queries.
        let m = model("collide_capsule.urdf");
        assert!(
            m.dropped_collider_frames.is_empty(),
            "a parsed capsule must not be dropped"
        );
        let cm = CollisionModel::new(m.clone(), WorldScene::new(), 0.0);
        assert_eq!(cm.num_colliders(), 1, "the capsule collider");
        assert_eq!(
            cm.uncovered_frames(),
            1,
            "only `base` is uncovered now; l1's capsule is covered"
        );
        // capsule core spans z in [-0.2,0.2], r0.1 at the origin; a world box there
        // overlaps it.
        let hit = CollisionModel::new(
            m.clone(),
            WorldScene::new().add_box([0.0, 0.0, 0.0], [0.2, 0.2, 0.2]),
            0.0,
        );
        assert!(
            hit.query(&[0.0]).unwrap().has_collision(),
            "capsule must collide with an overlapping world box"
        );
        // distant box → clear (FK consistency through the capsule path)
        let clear = CollisionModel::new(
            m.clone(),
            WorldScene::new().add_box([5.0, 0.0, 0.0], [0.2, 0.2, 0.2]),
            0.0,
        );
        assert!(!clear.query(&[0.0]).unwrap().has_collision());
        // ground catching the capsule's lowest point (z=-0.3); far below → clear
        let ground = CollisionModel::new(m.clone(), WorldScene::new().with_ground(-0.25), 0.0);
        assert!(!ground.query(&[0.0]).unwrap().world_hits.is_empty());
        let deep = CollisionModel::new(m, WorldScene::new().with_ground(-1.0), 0.0);
        assert!(deep.query(&[0.0]).unwrap().world_hits.is_empty());
    }

    // ===== EPA penetration depth =====

    #[test]
    fn epa_two_boxes_exact_depth_and_normal() {
        // Two unit boxes overlapping 0.1 along x: A at origin, B at +0.9x (each half
        // 0.5). The Minkowski difference is a box, so EPA converges EXACTLY — depth
        // 0.1 along +x, witness on A's +x face (x=0.5).
        let i = Matrix3::identity();
        let h = Vector3::new(0.5, 0.5, 0.5);
        let a = Prim::Obb {
            c: Vector3::zeros(),
            r: i,
            h,
        };
        let b = Prim::Obb {
            c: Vector3::new(0.9, 0.0, 0.0),
            r: i,
            h,
        };
        let tetra = gjk_tetra(&a, &b).expect("overlap must yield a GJK tetra");
        let ct = epa(&a, &b, tetra).expect("EPA must return a contact");
        assert!((ct.depth - 0.1).abs() < 1e-9, "depth {}", ct.depth);
        assert!(
            (ct.normal - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-9,
            "normal {:?}",
            ct.normal
        );
        assert!(
            (ct.normal.norm() - 1.0).abs() < 1e-12,
            "normal must be unit"
        );
        assert!(
            (ct.witness.x - 0.5).abs() < 1e-9,
            "witness on A's +x face, got {:?}",
            ct.witness
        );
        // Resolution check: translating A by slightly MORE than -depth*normal must
        // clear the boolean overlap (the reported MTV genuinely separates them).
        let a_clear = Prim::Obb {
            c: -ct.normal * (ct.depth + 1e-6),
            r: i,
            h,
        };
        assert!(
            !intersects(&a_clear, &b),
            "translating A past -depth*normal must remove the overlap"
        );
    }

    #[test]
    fn epa_two_spheres_exact_depth_and_normal() {
        // Smooth-support cross-check: spheres r0.5 @ origin and r0.4 @ +0.7x overlap
        // by 0.9-0.7 = 0.2 along +x. The Minkowski difference is a perfect sphere, so
        // EPA converges cleanly. Witness on A's surface along +x = (0.5,0,0).
        let a = Prim::Sphere {
            c: Vector3::zeros(),
            radius: 0.5,
        };
        let b = Prim::Sphere {
            c: Vector3::new(0.7, 0.0, 0.0),
            radius: 0.4,
        };
        let tetra = gjk_tetra(&a, &b).expect("overlap");
        let ct = epa(&a, &b, tetra).expect("contact");
        // Depth converges exactly; the NORMAL carries a polytope-order angular
        // approximation on a curved (sphere) Minkowski surface — 1e-3 is the honest
        // bar for a smooth support (box-box pins the exact 1e-9 polytope case).
        assert!((ct.depth - 0.2).abs() < 1e-6, "depth {}", ct.depth);
        assert!(
            (ct.normal - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-3,
            "normal {:?}",
            ct.normal
        );
        assert!(
            (ct.witness - Vector3::new(0.5, 0.0, 0.0)).norm() < 1e-2,
            "witness {:?}",
            ct.witness
        );
    }

    #[test]
    fn epa_sphere_into_box_matches_analytic() {
        // Sphere (r0.3) centered at +0.6x penetrates box A (half 0.5): the sphere's
        // left tip reaches x=0.3, the box's +x face is at 0.5 → depth 0.2 along +x.
        // EPA's depth is a LOWER bound on the true penetration (the polytope is
        // inscribed) and converges to it; the box's discrete corner support limits
        // flat-face precision, so bracket tightly-but-safely rather than demand
        // machine precision. The box-box / sphere-sphere tests pin the exact path.
        let i = Matrix3::identity();
        let box_a = Prim::Obb {
            c: Vector3::zeros(),
            r: i,
            h: Vector3::new(0.5, 0.5, 0.5),
        };
        let sphere_b = Prim::Sphere {
            c: Vector3::new(0.6, 0.0, 0.0),
            radius: 0.3,
        };
        let tetra = gjk_tetra(&box_a, &sphere_b).expect("overlap");
        let ct = epa(&box_a, &sphere_b, tetra).expect("contact");
        assert!(
            ct.depth > 0.2 - 1e-3 && ct.depth <= 0.2 + 1e-9,
            "depth {} must bracket the analytic 0.2",
            ct.depth
        );
        assert!(
            ct.normal.x > 0.999 && ct.normal.y.abs() < 1e-2 && ct.normal.z.abs() < 1e-2,
            "normal {:?} must point ~+x",
            ct.normal
        );
        assert!(
            (ct.witness.x - 0.5).abs() < 1e-6,
            "witness on the box's +x face, got {:?}",
            ct.witness
        );
    }

    #[test]
    fn contacts_report_penetration_on_self_collision() {
        // End-to-end through FK: the folded arm self-collides (l1 vs l3 boxes), and
        // contacts() must surface that pair with a positive penetration depth and a
        // unit normal, while a clear config yields none.
        let m = model("collide_arm.urdf");
        let cm = CollisionModel::new(m.clone(), WorldScene::new(), 0.0);
        let pi = std::f64::consts::PI;
        assert!(cm.query(&[0.0, pi, pi]).unwrap().has_collision());
        let cts = cm.contacts(&[0.0, pi, pi]).unwrap();
        assert!(
            !cts.is_empty(),
            "folded self-collision must yield a contact"
        );
        let l1 = m.frame_id("l1").unwrap();
        let l3 = m.frame_id("l3").unwrap();
        assert!(
            cts.iter()
                .any(|&(a, b, _)| (a, b) == (l1.min(l3), l1.max(l3))),
            "the (l1,l3) pair must be reported, got {:?}",
            cts.iter().map(|&(a, b, _)| (a, b)).collect::<Vec<_>>()
        );
        for (a, b, c) in &cts {
            assert!(a < b, "canonical frame order a<b");
            assert!(c.depth > 0.0, "penetration depth must be positive");
            assert!(
                (c.normal.norm() - 1.0).abs() < 1e-9,
                "contact normal must be unit"
            );
            assert!(c.depth.is_finite() && c.witness.iter().all(|x| x.is_finite()));
        }
        assert!(
            cm.contacts(&[0.0, 0.0, 0.0]).unwrap().is_empty(),
            "a clear config yields no contacts"
        );
        assert!(
            matches!(cm.contacts(&[0.0]), Err(CollisionError::Dim { .. })),
            "dim guard"
        );
        assert!(matches!(
            cm.contacts(&[0.0, f64::NAN, 0.0]),
            Err(CollisionError::NonFinite)
        ));
    }
}
