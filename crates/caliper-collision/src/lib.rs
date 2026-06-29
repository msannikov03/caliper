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
//! detecting a collision — safe). Everything is deterministic and dependency-free.
//!
//! ⚠ SCOPE: only box/sphere/cylinder `<collision>` primitives are checked. Links
//! whose collision geometry is a MESH or capsule carry no collider (no mesh
//! loader), so they are NOT checked — a report can read "clear" while such a link
//! interpenetrates. [`CollisionModel::uncovered_frames`] returns that count;
//! callers should surface it rather than trust a "clear" verdict blindly. A
//! conservative fallback collider for mesh links is future work.

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
#[derive(Clone, Copy, Debug)]
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
    /// Frames carrying NO collider (unsafe-by-omission hint for the UI/docs).
    pub fn uncovered_frames(&self) -> usize {
        let covered: HashSet<usize> = self.model.collision.iter().map(|g| g.frame).collect();
        self.model.frames.len() - covered.len()
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
                (prim_for(g.shape, c, r, self.margin), g.frame)
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
/// Cylinders become their tight OBB (Z-aligned local; conservative).
fn prim_for(shape: CollisionShape, c: Vector3<f64>, r: Matrix3<f64>, margin: f64) -> Prim {
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
}
