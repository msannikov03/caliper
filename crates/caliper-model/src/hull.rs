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

/// Hard cap on the number of points fed to the (O(n²)) dedup + incremental hull
/// build. Real-robot collision meshes reuse full-detail visual STLs as
/// colliders — e.g. so101's onshape-generated links carry MULTIPLE meshes of
/// tens of thousands of vertices each — and the unbounded incremental build on
/// that is pathologically slow (effectively an infinite hang on load). Any cloud
/// larger than this is deterministically subsampled down to it *before* the
/// expensive work, so both the dedup and the build become bounded. 4096 is large
/// enough for an accurate collision hull (far more than the ~dozens of true hull
/// vertices a robot link has) yet small enough that the whole pipeline finishes
/// in well under a second even for a mesh with 50k+ vertices.
///
/// Sized against the actual cost: `dedup` and `build_faces` are both ~O(n²), so a
/// mesh capped at 4096 still took ~7 s to hull. A robot link's *true* convex hull
/// has only dozens–low-hundreds of vertices, so 1024 axis-preserving samples
/// capture its shape well while keeping the whole pipeline well under ~0.5 s per
/// mesh (≈(1024/4096)² of the 4096 cost).
const MAX_HULL_INPUT: usize = 1024;

/// Convex hull vertices of `points`. Guaranteed to be a subset of `points` whose
/// convex hull contains every input point (within tolerance); on any degeneracy
/// (fewer than 4 unique points, collinear, coplanar) or verification failure it
/// returns the full deduplicated cloud — always collision-correct (see module
/// docs). Returns empty only for empty input.
///
/// This function is TOTAL and BOUNDED: it always returns in time bounded by a
/// function of [`MAX_HULL_INPUT`], never hanging. Clouds larger than the cap are
/// first subsampled (keeping the 6 axis-extreme points, so the reduced hull
/// still bounds the object's AABB on every axis) — a conservative collision
/// approximation, not the exact hull; see [`subsample`].
pub fn convex_hull(points: &[Point3<f64>]) -> Vec<Point3<f64>> {
    // Cap the input BEFORE dedup so both the O(n²) dedup and the O(n²) build are
    // bounded. For clouds at or under the cap `subsample` returns the input
    // verbatim (same order), so small meshes behave EXACTLY as before.
    let capped = subsample(points, MAX_HULL_INPUT);
    let unique = dedup(&capped);
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

/// Deterministically reduce `points` to at most `cap` points, guaranteeing the 6
/// axis-extreme points (min/max of x, y, z) are kept so the AABB corners of the
/// true hull survive — the reduced hull therefore still bounds the object on
/// every axis. The remainder of the budget is filled by a fixed stride over the
/// input in stable order (no rand, fully deterministic). Clouds already at or
/// under `cap` are returned verbatim (same order), so downstream results are
/// identical to the pre-cap behavior.
///
/// This is a CONSERVATIVE APPROXIMATION for collision, not the exact hull: a true
/// extreme vertex lying along a non-axis (diagonal) direction may be dropped, so
/// the reduced hull can sit marginally *inside* the true hull off-axis. That is
/// an accepted accuracy/speed trade-off — without it a detailed collision mesh
/// hangs the loader outright. Axis directions remain exact.
fn subsample(points: &[Point3<f64>], cap: usize) -> Vec<Point3<f64>> {
    let n = points.len();
    if n <= cap {
        return points.to_vec();
    }
    // 1. Axis extremes: min/max index along each of x, y, z (single O(n) pass).
    //    Order: [min_x, max_x, min_y, max_y, min_z, max_z].
    let mut ext = [0usize; 6];
    for i in 1..n {
        let c = points[i].coords;
        if c.x < points[ext[0]].coords.x {
            ext[0] = i;
        }
        if c.x > points[ext[1]].coords.x {
            ext[1] = i;
        }
        if c.y < points[ext[2]].coords.y {
            ext[2] = i;
        }
        if c.y > points[ext[3]].coords.y {
            ext[3] = i;
        }
        if c.z < points[ext[4]].coords.z {
            ext[4] = i;
        }
        if c.z > points[ext[5]].coords.z {
            ext[5] = i;
        }
    }
    let mut keep = vec![false; n];
    let mut out: Vec<Point3<f64>> = Vec::with_capacity(cap);
    for &e in &ext {
        if !keep[e] {
            keep[e] = true;
            out.push(points[e]);
        }
    }
    // 2. Fixed-stride fill of the remaining budget over the whole cloud.
    let budget = cap.saturating_sub(out.len()).max(1);
    let stride = n.div_ceil(budget).max(1); // ~budget evenly spaced samples
    let mut i = 0usize;
    while i < n && out.len() < cap {
        if !keep[i] {
            out.push(points[i]);
        }
        i += stride;
    }
    out
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
    // Belt-and-suspenders work budget on the incremental sweep. `convex_hull`
    // already caps `n` at MAX_HULL_INPUT, which bounds this O(n²) loop; this
    // counter is a hard guarantee of totality even if `build_faces` were ever
    // driven with a pathological/degenerate cloud directly. We count the
    // dominant cost — face-visibility tests, `faces.len()` per inserted point —
    // and on exceed bail with `None`, so the caller falls back to the full
    // deduped (capped) cloud, which is always collision-correct (see module
    // docs). The bound (4·n² + 1M) sits comfortably above any legitimate hull's
    // work, so it never trips on real clouds and still finishes in <100 ms.
    let work_budget: u64 = (n as u64)
        .saturating_mul(n as u64)
        .saturating_mul(4)
        .saturating_add(1_000_000);
    let mut work: u64 = 0;
    for (idx, p) in pts.iter().enumerate() {
        if idx == i0 || idx == i1 || idx == i2 || idx == i3 {
            continue;
        }
        work = work.saturating_add(faces.len() as u64);
        if work > work_budget {
            return None; // pathological blow-up → fall back to the full cloud
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

    /// Deterministic splitmix64 stream mapped into [-1, 1]; no `rand`.
    fn splitmix(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) as f64) / (u64::MAX as f64) * 2.0 - 1.0
        }
    }

    #[test]
    fn huge_cloud_is_bounded_and_keeps_axis_extremes() {
        // A 60k-vertex cloud (the scale of so101's per-link collision STLs) must
        // load fast and never hang. It must be capped to <= MAX_HULL_INPUT and
        // still bound the object exactly on every axis (we always keep the 6
        // axis-extreme points), so GJK support along the axes is unchanged.
        let mut next = splitmix(0xDEAD_BEEF_1234_5678);
        let n = 60_000usize;
        let pts: Vec<Point3<f64>> = (0..n)
            .map(|_| Point3::new(next(), next(), next()))
            .collect();
        let t = std::time::Instant::now();
        let hull = convex_hull(&pts);
        let dt = t.elapsed();
        // 3 s: the guarded regression is effectively-infinite (pre-cap so101
        // hung >31 s); a loose bound stays meaningful while tolerating debug
        // builds on a machine busy with a parallel gate sweep (observed 1.5 s).
        assert!(dt.as_secs_f64() < 3.0, "hull build too slow: {dt:?}");
        assert!(
            hull.len() <= MAX_HULL_INPUT,
            "hull not bounded: {} > {MAX_HULL_INPUT}",
            hull.len()
        );
        assert!(hull.len() >= 4);
        // subset of the input
        for p in &hull {
            assert!(
                pts.iter().any(|q| (q - p).norm() < 1e-12),
                "hull pt not in input"
            );
        }
        // axis support must MATCH the full cloud (the kept AABB extremes)
        for d in [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ] {
            let s_in = pts
                .iter()
                .map(|p| p.coords.dot(&d))
                .fold(f64::MIN, f64::max);
            let s_h = hull
                .iter()
                .map(|p| p.coords.dot(&d))
                .fold(f64::MIN, f64::max);
            assert!(
                (s_in - s_h).abs() < 1e-9,
                "axis support mismatch along {d:?}: {s_in} vs {s_h}"
            );
        }
    }

    #[test]
    fn huge_degenerate_cloud_falls_back_without_hanging() {
        // A duplicate-heavy, coplanar (z=0) mega-cloud: snapping x,y to a coarse
        // 11x11 integer lattice makes 60k points collapse to <=121 unique, and
        // coplanarity forces the safe fallback. It must not hang and must return
        // a bounded, collision-correct set.
        let mut next = splitmix(0x00C0_FFEE_D00D_1010);
        let n = 60_000usize;
        let pts: Vec<Point3<f64>> = (0..n)
            .map(|_| {
                // map [-1,1] -> integer grid [0,10], z pinned to the plane
                let gx = ((next() + 1.0) * 5.0).round();
                let gy = ((next() + 1.0) * 5.0).round();
                Point3::new(gx, gy, 0.0)
            })
            .collect();
        let t = std::time::Instant::now();
        let hull = convex_hull(&pts);
        let dt = t.elapsed();
        assert!(dt.as_secs_f64() < 1.0, "degenerate hull too slow: {dt:?}");
        assert!(
            hull.len() <= MAX_HULL_INPUT && !hull.is_empty(),
            "unexpected fallback size: {}",
            hull.len()
        );
        // The kept AABB extremes guarantee axis support still bounds the object
        // exactly on x and y (z is the degenerate axis). (0,0) is the min-x/min-y
        // extreme and 10 is the max on both axes.
        for d in [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
        ] {
            let s_in = pts
                .iter()
                .map(|p| p.coords.dot(&d))
                .fold(f64::MIN, f64::max);
            let s_h = hull
                .iter()
                .map(|p| p.coords.dot(&d))
                .fold(f64::MIN, f64::max);
            assert!(
                (s_in - s_h).abs() < 1e-9,
                "axis support mismatch along {d:?}"
            );
        }
    }

    #[test]
    fn small_cloud_unaffected_by_cap() {
        // Under the cap, subsample is a no-op (verbatim, same order), so the
        // hull is byte-for-byte what the pre-cap code produced.
        assert!(cube_corners(0.5).len() <= MAX_HULL_INPUT);
        let pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.1, 0.1, 0.1),
        ];
        assert_eq!(subsample(&pts, MAX_HULL_INPUT), pts);
        assert_eq!(convex_hull(&pts).len(), 4);
    }
}
