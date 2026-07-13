//! Rigid-body mass properties from geometry, for repairing missing `<inertial>`s.
//!
//! Everything is computed at UNIT density and scaled by the caller's density
//! (kg/m³) when converted to a [`SpatialInertia`]:
//!
//! - Triangle meshes: the Mirtich/Eberly DIVERGENCE-THEOREM integrals
//!   ([`mesh_props`]) — each triangle contributes a signed tetrahedron against
//!   the origin, so volume, COM, and the full inertia tensor (off-diagonals
//!   included) come out of one pass over the faces. Requires a CLOSED,
//!   consistently wound mesh; a globally reversed winding is detected (negative
//!   volume) and flipped, but an open or mixed-winding mesh integrates to a
//!   near-zero volume and is REFUSED (`None`) rather than silently writing a
//!   garbage inertial. (The repo's own `unit_cube.stl` hull fixture is exactly
//!   such a cloud — see the refusal test.)
//! - Primitives (box/sphere/cylinder/capsule): closed-form reference formulas,
//!   the capsule composed as cylinder + two hemispheres via parallel axis.
//!
//! Tests pin every path to analytic ground truth: exact for cube meshes and a
//! translated cube (parallel-axis proof), tolerance-bounded for an icosphere
//! and a faceted cylinder, and exact capsule↔sphere agreement at zero length.

use caliper_spatial::{SpatialInertia, skew};
use nalgebra::{Matrix3, Point3, Vector3};
use std::f64::consts::PI;

/// Mass properties at UNIT density: `mass == volume`.
#[derive(Clone, Debug)]
pub struct MassProps {
    /// Volume (m³) — the unit-density mass.
    pub volume: f64,
    /// Center of mass, in the shape's own frame.
    pub com: Vector3<f64>,
    /// Inertia tensor about the COM, shape axes, unit density.
    pub inertia_com: Matrix3<f64>,
}

impl MassProps {
    /// Convert to a link-frame spatial inertia at `density` kg/m³ (transform
    /// with the shape's origin afterwards to move it onto the link).
    pub fn spatial(&self, density: f64) -> SpatialInertia {
        SpatialInertia::from_mass_com_inertia(
            density * self.volume,
            self.com,
            self.inertia_com * density,
        )
    }
}

/// Recover `(mass, com, inertia_about_com)` from a spatial inertia — the exact
/// inverse of [`SpatialInertia::from_mass_com_inertia`], used to write a summed
/// composite back out as URDF `<inertial>` fields.
pub fn extract_mass_com_inertia(g: &SpatialInertia) -> (f64, Vector3<f64>, Matrix3<f64>) {
    let m = g.matrix();
    let mass = m[(0, 0)];
    // lower-left block is m·[c]×; read c off the skew entries
    let mcx = m.fixed_view::<3, 3>(3, 0);
    let com = Vector3::new(mcx[(2, 1)], mcx[(0, 2)], mcx[(1, 0)]) / mass;
    let cx = skew(&com);
    // from_mass_com_inertia stored I_o = I_com − m·[c]×[c]× in the lower-right
    let i_com = m.fixed_view::<3, 3>(3, 3) + mass * (cx * cx);
    // symmetrize away accumulation noise before it lands in a file
    let i_com = (i_com + i_com.transpose()) * 0.5;
    (mass, com, i_com)
}

/// Eberly's subexpressions for one coordinate of one triangle.
#[allow(clippy::many_single_char_names)]
fn subexpr(w0: f64, w1: f64, w2: f64) -> (f64, f64, f64, f64, f64, f64) {
    let t0 = w0 + w1;
    let f1 = t0 + w2;
    let t1 = w0 * w0;
    let t2 = t1 + w1 * t0;
    let f2 = t2 + w2 * f1;
    let f3 = w0 * t1 + w1 * t2 + w2 * f2;
    let g0 = f2 + w0 * (f1 + w0);
    let g1 = f2 + w1 * (f1 + w1);
    let g2 = f2 + w2 * (f1 + w2);
    (f1, f2, f3, g0, g1, g2)
}

/// Mass properties of a triangle soup (`tris.len() % 3 == 0`, three vertices
/// per face, the layout `caliper_model::stl::parse_stl` produces). `None` when
/// the soup cannot be a closed solid: fewer than 4 faces, non-finite vertices,
/// or a |signed volume| indistinguishable from zero (open or inconsistently
/// wound mesh) — see the module docs.
pub fn mesh_props(tris: &[Point3<f64>]) -> Option<MassProps> {
    if tris.len() < 12 || !tris.len().is_multiple_of(3) {
        return None;
    }
    let mut scale: f64 = 0.0;
    let mut intg = [0.0f64; 10];
    for t in tris.chunks_exact(3) {
        let (p0, p1, p2) = (t[0], t[1], t[2]);
        for p in [&p0, &p1, &p2] {
            if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
                return None;
            }
            scale = scale.max(p.coords.amax());
        }
        let d = (p1 - p0).cross(&(p2 - p0));
        let (f1x, f2x, f3x, g0x, g1x, g2x) = subexpr(p0.x, p1.x, p2.x);
        let (_f1y, f2y, f3y, g0y, g1y, g2y) = subexpr(p0.y, p1.y, p2.y);
        let (_f1z, f2z, f3z, g0z, g1z, g2z) = subexpr(p0.z, p1.z, p2.z);
        intg[0] += d.x * f1x;
        intg[1] += d.x * f2x;
        intg[2] += d.y * f2y;
        intg[3] += d.z * f2z;
        intg[4] += d.x * f3x;
        intg[5] += d.y * f3y;
        intg[6] += d.z * f3z;
        intg[7] += d.x * (p0.y * g0x + p1.y * g1x + p2.y * g2x);
        intg[8] += d.y * (p0.z * g0y + p1.z * g1y + p2.z * g2y);
        intg[9] += d.z * (p0.x * g0z + p1.x * g1z + p2.x * g2z);
    }
    const MULT: [f64; 10] = [
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 24.0,
        1.0 / 24.0,
        1.0 / 60.0,
        1.0 / 60.0,
        1.0 / 60.0,
        1.0 / 120.0,
        1.0 / 120.0,
        1.0 / 120.0,
    ];
    for (v, m) in intg.iter_mut().zip(MULT) {
        *v *= m;
    }
    // globally reversed (inward) winding integrates to the negated values
    if intg[0] < 0.0 {
        for v in &mut intg {
            *v = -*v;
        }
    }
    let vol = intg[0];
    // refuse open / mixed-winding meshes: their signed volume collapses toward 0
    if !vol.is_finite() || vol <= 1e-12 * scale.powi(3).max(f64::MIN_POSITIVE) {
        return None;
    }
    let com = Vector3::new(intg[1], intg[2], intg[3]) / vol;
    let m = vol; // unit density
    let ixx = intg[5] + intg[6] - m * (com.y * com.y + com.z * com.z);
    let iyy = intg[4] + intg[6] - m * (com.z * com.z + com.x * com.x);
    let izz = intg[4] + intg[5] - m * (com.x * com.x + com.y * com.y);
    let ixy = -(intg[7] - m * com.x * com.y);
    let iyz = -(intg[8] - m * com.y * com.z);
    let ixz = -(intg[9] - m * com.z * com.x);
    Some(MassProps {
        volume: vol,
        com,
        inertia_com: Matrix3::new(ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz),
    })
}

/// Solid box from HALF-extents (URDF `size`/2).
pub fn box_props(half: Vector3<f64>) -> MassProps {
    let (a, b, c) = (2.0 * half.x, 2.0 * half.y, 2.0 * half.z);
    let v = a * b * c;
    MassProps {
        volume: v,
        com: Vector3::zeros(),
        inertia_com: Matrix3::from_diagonal(&Vector3::new(
            v * (b * b + c * c) / 12.0,
            v * (a * a + c * c) / 12.0,
            v * (a * a + b * b) / 12.0,
        )),
    }
}

/// Solid sphere.
pub fn sphere_props(r: f64) -> MassProps {
    let v = 4.0 / 3.0 * PI * r.powi(3);
    MassProps {
        volume: v,
        com: Vector3::zeros(),
        inertia_com: Matrix3::from_diagonal_element(0.4 * v * r * r),
    }
}

/// Solid Z-aligned cylinder (URDF convention), `l` the FULL length.
pub fn cylinder_props(r: f64, l: f64) -> MassProps {
    let v = PI * r * r * l;
    MassProps {
        volume: v,
        com: Vector3::zeros(),
        inertia_com: Matrix3::from_diagonal(&Vector3::new(
            v * (3.0 * r * r + l * l) / 12.0,
            v * (3.0 * r * r + l * l) / 12.0,
            v * r * r / 2.0,
        )),
    }
}

/// Solid Z-aligned capsule (URDF convention: `l` is the CORE segment length,
/// hemispherical caps of radius `r` beyond it). Composed as a cylinder plus
/// two hemispheres: each hemisphere has `I = 2/5·m·r²` about its base center
/// and its COM `3r/8` above the base, so shifting to the capsule center is
/// two parallel-axis hops. At `l = 0` this reduces EXACTLY to the sphere.
pub fn capsule_props(r: f64, l: f64) -> MassProps {
    let vc = PI * r * r * l; // cylinder volume (= unit-density mass)
    let vh = 2.0 / 3.0 * PI * r.powi(3); // one hemisphere
    let izz = vc * r * r / 2.0 + 2.0 * (0.4 * vh * r * r);
    let z_com = l / 2.0 + 3.0 * r / 8.0; // hemisphere COM height above center
    let ixx = vc * (3.0 * r * r + l * l) / 12.0
        + 2.0 * (0.4 * vh * r * r - vh * (3.0 * r / 8.0).powi(2) + vh * z_com * z_com);
    MassProps {
        volume: vc + 2.0 * vh,
        com: Vector3::zeros(),
        inertia_com: Matrix3::from_diagonal(&Vector3::new(ixx, ixx, izz)),
    }
}

// ===== test mesh generators (shared with repair tests) =====

/// Deterministic closed test meshes with OUTWARD winding. `cfg(test)` only —
/// repair tests reuse them to write well-formed STL files.
#[cfg(test)]
pub mod testmesh {
    use nalgebra::{Point3, Vector3};

    /// Axis-aligned cube of half-extent `h` centered at `center`: 12 outward
    /// triangles built per face from `u × v = n` bases (winding correct by
    /// construction).
    pub fn cube_tris(h: f64, center: Vector3<f64>) -> Vec<Point3<f64>> {
        let mut out = Vec::with_capacity(36);
        for axis in 0..3 {
            for sgn in [1.0f64, -1.0] {
                let mut n = Vector3::zeros();
                n[axis] = sgn;
                let mut u = Vector3::zeros();
                u[(axis + 1) % 3] = 1.0;
                let v = n.cross(&u);
                let o = center + n * h;
                let a = Point3::from(o - u * h - v * h);
                let b = Point3::from(o + u * h - v * h);
                let c = Point3::from(o + u * h + v * h);
                let d = Point3::from(o - u * h + v * h);
                out.extend([a, b, c, a, c, d]);
            }
        }
        out
    }

    /// Icosphere of radius `r`: subdivided icosahedron with vertices projected
    /// onto the sphere. `level = 3` → 1280 faces (volume within ~0.9% of the
    /// true sphere, inertia within ~1.5%).
    pub fn icosphere_tris(r: f64, level: usize) -> Vec<Point3<f64>> {
        let t = (1.0 + 5.0f64.sqrt()) / 2.0;
        let raw = [
            [-1.0, t, 0.0],
            [1.0, t, 0.0],
            [-1.0, -t, 0.0],
            [1.0, -t, 0.0],
            [0.0, -1.0, t],
            [0.0, 1.0, t],
            [0.0, -1.0, -t],
            [0.0, 1.0, -t],
            [t, 0.0, -1.0],
            [t, 0.0, 1.0],
            [-t, 0.0, -1.0],
            [-t, 0.0, 1.0],
        ];
        let proj = |v: Vector3<f64>| Point3::from(v.normalize() * r);
        let vs: Vec<Point3<f64>> = raw
            .iter()
            .map(|a| proj(Vector3::new(a[0], a[1], a[2])))
            .collect();
        const FACES: [[usize; 3]; 20] = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        let mut tris: Vec<[Point3<f64>; 3]> = FACES
            .iter()
            .map(|f| [vs[f[0]], vs[f[1]], vs[f[2]]])
            .collect();
        for _ in 0..level {
            let mut next = Vec::with_capacity(tris.len() * 4);
            for [a, b, c] in tris {
                let ab = proj((a.coords + b.coords) / 2.0);
                let bc = proj((b.coords + c.coords) / 2.0);
                let ca = proj((c.coords + a.coords) / 2.0);
                next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
            }
            tris = next;
        }
        tris.into_iter().flatten().collect()
    }

    /// Faceted Z-aligned cylinder (radius `r`, full length `l`, `n` segments):
    /// outward side quads split into triangles plus cap fans. `n = 64` puts
    /// volume/inertia within ~0.4% of the smooth cylinder.
    pub fn cylinder_tris(r: f64, l: f64, n: usize) -> Vec<Point3<f64>> {
        let mut out = Vec::with_capacity(n * 12);
        let ct = Point3::new(0.0, 0.0, l / 2.0);
        let cb = Point3::new(0.0, 0.0, -l / 2.0);
        for k in 0..n {
            let (a0, a1) = (
                2.0 * std::f64::consts::PI * k as f64 / n as f64,
                2.0 * std::f64::consts::PI * (k + 1) as f64 / n as f64,
            );
            let b0 = Point3::new(r * a0.cos(), r * a0.sin(), -l / 2.0);
            let b1 = Point3::new(r * a1.cos(), r * a1.sin(), -l / 2.0);
            let t0 = Point3::new(r * a0.cos(), r * a0.sin(), l / 2.0);
            let t1 = Point3::new(r * a1.cos(), r * a1.sin(), l / 2.0);
            out.extend([b0, b1, t1, b0, t1, t0]); // side (outward)
            out.extend([ct, t0, t1]); // top cap (+z)
            out.extend([cb, b1, b0]); // bottom cap (−z)
        }
        out
    }

    /// Serialize a triangle soup as ASCII STL (normals zeroed; the loader
    /// ignores them).
    pub fn ascii_stl(tris: &[Point3<f64>]) -> String {
        let mut s = String::from("solid t\n");
        for t in tris.chunks_exact(3) {
            s.push_str("facet normal 0 0 0\nouter loop\n");
            for p in t {
                s.push_str(&format!("vertex {} {} {}\n", p.x, p.y, p.z));
            }
            s.push_str("endloop\nendfacet\n");
        }
        s.push_str("endsolid t\n");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::testmesh::*;
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    #[test]
    fn cube_mesh_matches_analytic_box_exactly() {
        let p = mesh_props(&cube_tris(0.5, Vector3::zeros())).unwrap();
        let a = box_props(Vector3::new(0.5, 0.5, 0.5));
        assert!((p.volume - 1.0).abs() < 1e-12, "vol {}", p.volume);
        assert!(p.com.norm() < 1e-12);
        assert!((p.inertia_com - a.inertia_com).norm() < 1e-12);
        // 1 m³ cube: I = m·a²/6 = 1/6 on the diagonal
        assert!((p.inertia_com[(0, 0)] - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn translated_cube_proves_parallel_axis() {
        // shifting the SAME cube must move only the COM: the tensor about the
        // COM is translation-invariant, so any error here is a parallel-axis bug.
        let p = mesh_props(&cube_tris(0.5, Vector3::new(1.0, 2.0, 3.0))).unwrap();
        assert!((p.volume - 1.0).abs() < 1e-9);
        assert!((p.com - Vector3::new(1.0, 2.0, 3.0)).norm() < 1e-9);
        let centered = mesh_props(&cube_tris(0.5, Vector3::zeros())).unwrap();
        assert!(
            (p.inertia_com - centered.inertia_com).norm() < 1e-9,
            "COM tensor must not change under translation"
        );
    }

    #[test]
    fn icosphere_matches_analytic_sphere_within_tolerance() {
        let p = mesh_props(&icosphere_tris(1.0, 3)).unwrap();
        let a = sphere_props(1.0);
        assert!(rel(p.volume, a.volume) < 0.02, "vol {}", p.volume);
        assert!(p.com.norm() < 1e-12, "centrosymmetric → COM at 0");
        for i in 0..3 {
            assert!(
                rel(p.inertia_com[(i, i)], a.inertia_com[(i, i)]) < 0.03,
                "I[{i}{i}] = {}",
                p.inertia_com[(i, i)]
            );
        }
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            assert!(p.inertia_com[(i, j)].abs() < 1e-12 * p.inertia_com[(0, 0)].abs().max(1.0));
        }
    }

    #[test]
    fn faceted_cylinder_matches_analytic_within_tolerance() {
        let p = mesh_props(&cylinder_tris(0.5, 2.0, 64)).unwrap();
        let a = cylinder_props(0.5, 2.0);
        assert!(rel(p.volume, a.volume) < 0.005, "vol {}", p.volume);
        assert!(p.com.norm() < 1e-12);
        assert!(rel(p.inertia_com[(0, 0)], a.inertia_com[(0, 0)]) < 0.01);
        assert!(rel(p.inertia_com[(2, 2)], a.inertia_com[(2, 2)]) < 0.01);
    }

    #[test]
    fn reversed_winding_is_normalized() {
        let mut tris = cube_tris(0.5, Vector3::zeros());
        for t in tris.chunks_exact_mut(3) {
            t.swap(1, 2); // flip every face inward
        }
        let p = mesh_props(&tris).unwrap();
        assert!((p.volume - 1.0).abs() < 1e-12, "global flip is recovered");
    }

    #[test]
    fn open_or_degenerate_meshes_are_refused() {
        assert!(mesh_props(&[]).is_none(), "empty");
        let tri = [
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        assert!(mesh_props(&tri).is_none(), "single face");
        // open flat quad: 4 faces' worth of area but zero enclosed volume
        let quad: Vec<Point3<f64>> = (0..4)
            .flat_map(|k| {
                let o = k as f64;
                [
                    Point3::new(o, 0.0, 0.0),
                    Point3::new(o + 1.0, 0.0, 0.0),
                    Point3::new(o, 1.0, 0.0),
                ]
            })
            .collect();
        assert!(mesh_props(&quad).is_none(), "open surface has no volume");
        let cube = cube_tris(0.5, Vector3::zeros());
        assert!(mesh_props(&cube[..35]).is_none(), "len % 3 != 0");
    }

    #[test]
    fn capsule_at_zero_length_is_exactly_a_sphere() {
        let c = capsule_props(0.7, 0.0);
        let s = sphere_props(0.7);
        assert!((c.volume - s.volume).abs() < 1e-12);
        assert!((c.inertia_com - s.inertia_com).norm() < 1e-12);
    }

    #[test]
    fn primitive_formulas_match_reference_values() {
        // hand-checked reference numbers at unit density
        let b = box_props(Vector3::new(0.5, 1.0, 1.5)); // 1×2×3 box, V=6
        assert!((b.volume - 6.0).abs() < 1e-12);
        assert!((b.inertia_com[(0, 0)] - 6.0 * (4.0 + 9.0) / 12.0).abs() < 1e-12);
        let s = sphere_props(2.0); // V = 32π/3, I = 2/5·V·4
        assert!((s.volume - 32.0 * PI / 3.0).abs() < 1e-12);
        assert!((s.inertia_com[(0, 0)] - 0.4 * s.volume * 4.0).abs() < 1e-12);
        let c = cylinder_props(1.0, 2.0); // V = 2π
        assert!((c.volume - 2.0 * PI).abs() < 1e-12);
        assert!((c.inertia_com[(2, 2)] - c.volume / 2.0).abs() < 1e-12);
    }

    #[test]
    fn spatial_extraction_round_trips() {
        let p = mesh_props(&cube_tris(0.5, Vector3::new(0.2, -0.1, 0.3))).unwrap();
        let (m, com, i_com) = extract_mass_com_inertia(&p.spatial(1000.0));
        assert!((m - 1000.0 * p.volume).abs() < 1e-9);
        assert!((com - p.com).norm() < 1e-12);
        assert!((i_com - p.inertia_com * 1000.0).norm() < 1e-6);
    }

    #[test]
    fn repo_unit_cube_fixture_is_refused_as_inconsistently_wound() {
        // oracle/fixtures/robots/unit_cube.stl is a HULL fixture: a vertex
        // cloud whose faces are not consistently wound (its signed volume is
        // exactly 0). The integrator must refuse it, not report volume 0.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../oracle/fixtures/robots/unit_cube.stl");
        let bytes = std::fs::read(path).unwrap();
        let cloud = caliper_model::stl::parse_stl(&bytes).unwrap();
        assert_eq!(cloud.len(), 36, "12 faces");
        assert!(mesh_props(&cloud).is_none());
    }
}
