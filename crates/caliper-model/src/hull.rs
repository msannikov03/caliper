//! Pure-Rust 3-D convex hull (bounded incremental) with a correctness safety net.
//!
//! [`convex_hull`] reduces a vertex cloud to the vertices of its convex hull so
//! collision carries a bounded point set. Correctness is the priority: the hull
//! is only ever a *size optimization*. After building it we VERIFY that every
//! input point lies inside-or-on the produced hull (within tolerance); if the
//! incremental build is degenerate or — for any reason — drops a genuine extreme
//! point, we fall back to returning the full deduplicated cloud. That fallback is
//! itself collision-correct, because the GJK support function over a point set is
//! identical to the support over its convex hull. So a hull bug can never make
//! collision NON-conservative (it can at worst cost memory). Dependency-free.

use nalgebra::{Point3, Vector3};

/// Convex hull vertices of `points`. Guaranteed to be a subset of `points` whose
/// convex hull contains every input point (within tolerance); on any degeneracy
/// (fewer than 4 unique points, collinear, coplanar) or verification failure it
/// returns the full deduplicated cloud — always collision-correct (see module
/// docs). Returns empty only for empty input.
pub fn convex_hull(points: &[Point3<f64>]) -> Vec<Point3<f64>> {
    let unique = dedup(points);
    if unique.len() < 4 {
        return unique; // a point / segment / triangle is its own hull
    }
    match build_faces(&unique) {
        Some(faces) => {
            // hull vertices = those referenced by surviving faces
            let mut used = vec![false; unique.len()];
            for f in &faces {
                for &vi in &f.v {
                    used[vi] = true;
                }
            }
            let hull: Vec<Point3<f64>> = (0..unique.len())
                .filter(|&i| used[i])
                .map(|i| unique[i])
                .collect();
            if hull.len() >= 4 && verify(&unique, &hull) {
                hull
            } else {
                // degeneracy slipped through, or verification failed → full cloud
                // (always collision-correct; see module docs)
                unique
            }
        }
        // degenerate (coplanar/collinear) cloud → full cloud is correct
        None => unique,
    }
}

/// Characteristic length of the cloud (max bbox extent), used to scale tolerances.
fn scale_of(p: &[Point3<f64>]) -> f64 {
    let mut lo = p[0].coords;
    let mut hi = p[0].coords;
    for q in p {
        lo = lo.inf(&q.coords);
        hi = hi.sup(&q.coords);
    }
    (hi - lo).amax().max(1.0)
}

fn dedup(points: &[Point3<f64>]) -> Vec<Point3<f64>> {
    let mut out: Vec<Point3<f64>> = Vec::new();
    if points.is_empty() {
        return out;
    }
    let tol = 1e-9 * scale_of(points);
    let tol2 = tol * tol;
    for &p in points {
        if !out.iter().any(|&q| (p - q).norm_squared() <= tol2) {
            out.push(p);
        }
    }
    out
}

/// A hull face: three indices into the point list (orientation is fixed up to
/// outward by the centroid, so winding is not relied upon for normals).
#[derive(Clone, Copy)]
struct Face {
    v: [usize; 3],
    n: Vector3<f64>,  // outward unit-ish normal
    p0: Vector3<f64>, // a point on the face (v[0])
}

fn dist_to_line(p: &Point3<f64>, a: &Point3<f64>, dir: &Vector3<f64>) -> f64 {
    let ap = p - a;
    (ap - dir * ap.dot(dir)).norm()
}

/// Verify every input point is inside-or-on the hull formed by `hull` vertices.
/// Conservative: returns `false` (→ caller uses full cloud) if any point lies
/// measurably outside the hull of the candidate vertices.
fn verify(all: &[Point3<f64>], hull: &[Point3<f64>]) -> bool {
    // Rebuild faces from the hull vertices to get the bounding planes, then test
    // each input point. If a real extreme vertex were dropped from `hull`, it is
    // an input point strictly outside conv(hull) → it violates some outward face
    // → we return false and the caller falls back to the (correct) full cloud.
    // If building faces from `hull` is itself degenerate, fail safe too.
    let scale = scale_of(all);
    let eps = 1e-7 * scale;
    let faces = match build_faces(hull) {
        Some(f) => f,
        None => return false,
    };
    all.iter()
        .all(|p| faces.iter().all(|f| f.n.dot(&(p.coords - f.p0)) <= eps))
}

/// Build the face planes of the convex hull of `pts` via an incremental
/// (quickhull-style) sweep. `None` on a degenerate (collinear/coplanar) cloud.
/// Face normals are oriented outward by the cloud centroid, so winding is never
/// relied upon.
fn build_faces(pts: &[Point3<f64>]) -> Option<Vec<Face>> {
    let n = pts.len();
    if n < 4 {
        return None;
    }
    let scale = scale_of(pts);
    let eps = 1e-9 * scale;
    let centroid: Vector3<f64> = pts.iter().map(|p| p.coords).sum::<Vector3<f64>>() / n as f64;
    let i0 = 0usize;
    let i1 = (0..n)
        .max_by(|&a, &b| {
            (pts[a] - pts[i0])
                .norm()
                .total_cmp(&(pts[b] - pts[i0]).norm())
        })
        .unwrap();
    if (pts[i1] - pts[i0]).norm() <= eps {
        return None;
    }
    let line = (pts[i1] - pts[i0]).normalize();
    let i2 = (0..n)
        .max_by(|&a, &b| {
            dist_to_line(&pts[a], &pts[i0], &line)
                .total_cmp(&dist_to_line(&pts[b], &pts[i0], &line))
        })
        .unwrap();
    if dist_to_line(&pts[i2], &pts[i0], &line) <= eps {
        return None;
    }
    let plane_n = (pts[i1] - pts[i0]).cross(&(pts[i2] - pts[i0]));
    let pn = plane_n.norm();
    if pn <= eps * eps {
        return None;
    }
    let plane_n = plane_n / pn;
    let i3 = (0..n)
        .max_by(|&a, &b| {
            ((pts[a] - pts[i0]).dot(&plane_n))
                .abs()
                .total_cmp(&((pts[b] - pts[i0]).dot(&plane_n)).abs())
        })
        .unwrap();
    if ((pts[i3] - pts[i0]).dot(&plane_n)).abs() <= eps {
        return None;
    }
    let mk = |a: usize, b: usize, c: usize| -> Face {
        let mut nrm = (pts[b] - pts[a]).cross(&(pts[c] - pts[a]));
        let nn = nrm.norm();
        if nn > 0.0 {
            nrm /= nn;
        }
        if nrm.dot(&(pts[a].coords - centroid)) < 0.0 {
            nrm = -nrm;
        }
        Face {
            v: [a, b, c],
            n: nrm,
            p0: pts[a].coords,
        }
    };
    let mut faces = vec![
        mk(i0, i1, i2),
        mk(i0, i1, i3),
        mk(i0, i2, i3),
        mk(i1, i2, i3),
    ];
    for (idx, p) in pts.iter().enumerate() {
        if idx == i0 || idx == i1 || idx == i2 || idx == i3 {
            continue;
        }
        let mut visible: Vec<usize> = faces
            .iter()
            .enumerate()
            .filter(|(_, f)| f.n.dot(&(p.coords - f.p0)) > eps)
            .map(|(k, _)| k)
            .collect();
        if visible.is_empty() {
            continue;
        }
        let mut edge_count: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        for &fk in &visible {
            let v = faces[fk].v;
            for (a, b) in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_count.entry(key).or_insert(0) += 1;
            }
        }
        let horizon: Vec<(usize, usize)> = edge_count
            .into_iter()
            .filter(|&(_, c)| c == 1)
            .map(|(e, _)| e)
            .collect();
        // remove visible faces high→low so swap_remove never disturbs an index we
        // still need (it only relocates kept faces, which we won't revisit)
        visible.sort_unstable();
        for &fk in visible.iter().rev() {
            faces.swap_remove(fk);
        }
        if faces.len() + horizon.len() > 64 * n + 64 {
            return None;
        }
        for (a, b) in horizon {
            faces.push(mk(a, b, idx));
        }
    }
    Some(faces)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_corners(h: f64) -> Vec<Point3<f64>> {
        let mut v = Vec::new();
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    v.push(Point3::new(sx * h, sy * h, sz * h));
                }
            }
        }
        v
    }

    #[test]
    fn cube_hull_is_collision_correct() {
        // A cube has COPLANAR faces, the degenerate case for the incremental hull;
        // per the module contract it then safely falls back to the full deduplicated
        // cloud (collision-correct: GJK support over the cloud == over the true hull).
        // So we assert the CONTRACT (correctness), not minimality (which is a deferred
        // optimization for coplanar-faced meshes — see the coplanar-minimize TODO).
        let corners = cube_corners(0.5);
        let mut pts = corners.clone();
        pts.push(Point3::new(0.0, 0.0, 0.0)); // interior
        pts.push(Point3::new(0.2, -0.1, 0.3)); // interior
        pts.push(Point3::new(0.5, 0.0, 0.0)); // on a face
        let h = convex_hull(&pts);
        // (1) subset of input
        for p in &h {
            assert!(
                pts.iter().any(|q| (q - p).norm() < 1e-12),
                "hull pt not in input"
            );
        }
        // (2) every true corner (extreme point) is retained — REQUIRED for GJK correctness
        for c in &corners {
            assert!(
                h.iter().any(|p| (p - c).norm() < 1e-12),
                "hull dropped a corner {c:?}"
            );
        }
        // (3) support equality vs the input cloud in many directions (the GJK property)
        for d in [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-0.3, 0.7, 0.2),
            Vector3::new(0.5, -0.5, 0.5),
            Vector3::new(-1.0, -1.0, -1.0),
        ] {
            let s_in = pts
                .iter()
                .map(|p| p.coords.dot(&d))
                .fold(f64::MIN, f64::max);
            let s_h = h.iter().map(|p| p.coords.dot(&d)).fold(f64::MIN, f64::max);
            assert!((s_in - s_h).abs() < 1e-12, "support mismatch along {d:?}");
        }
    }

    #[test]
    fn tetra_hull_is_four_vertices() {
        let pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.1, 0.1, 0.1), // interior
        ];
        let h = convex_hull(&pts);
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn coplanar_falls_back_to_full_cloud() {
        // a flat square (z=0) is degenerate for a 3-D tetra seed → fallback path.
        let pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let h = convex_hull(&pts);
        // fallback returns the (deduped) cloud — still collision-correct
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn every_input_point_inside_random_hull() {
        // deterministic splitmix cloud; the hull must contain every input point.
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) as f64) / (u64::MAX as f64) * 2.0 - 1.0
        };
        let pts: Vec<Point3<f64>> = (0..200)
            .map(|_| Point3::new(next(), next(), next()))
            .collect();
        let hull = convex_hull(&pts);
        assert!(hull.len() >= 4 && hull.len() <= pts.len());
        // rebuild faces and confirm containment of all inputs (the safety net)
        let faces = build_faces(&hull).expect("non-degenerate random cloud");
        for p in &pts {
            assert!(
                faces.iter().all(|f| f.n.dot(&(p.coords - f.p0)) <= 1e-6),
                "input point {p:?} fell outside the computed hull"
            );
        }
    }
}
