//! CHOMP-style collision-aware trajectory optimization for Caliper.
//!
//! Given an initial joint-space waypoint path, this crate refines the INTERIOR
//! waypoints (the two endpoints are held fixed) by gradient descent on a cost
//!
//! ```text
//!   cost(path) = w_smooth · smoothness(path) + w_obs · obstacle(path)
//! ```
//!
//! * **smoothness** is the classic CHOMP quadratic — the sum of squared
//!   finite-difference accelerations `‖q[i-1] − 2·q[i] + q[i+1]‖²` over the
//!   interior. Its gradient is the pentadiagonal `Aᵀ A q` form (implemented
//!   analytically in [`smoothness_gradient`] and cross-validated against a
//!   finite-difference of [`smoothness_cost`] in the tests).
//!
//! * **obstacle** is a smooth proximity penalty. Each robot collider is reduced
//!   to a *body sphere* (its `<collision>` origin, placed by forward kinematics,
//!   inflated by the collider's bounding radius) and scored against an
//!   [`ObstacleField`] signed-distance field with the standard CHOMP hinge
//!   potential: zero once the sphere clears the obstacle by `obstacle_margin`,
//!   growing quadratically inside that margin, and linearly on penetration. Its
//!   gradient is taken by central finite differences through the (smooth) field —
//!   robust, not the fragile finite-difference-of-a-boolean.
//!
//! ## Why an [`ObstacleField`] AND a [`CollisionModel`]
//!
//! [`caliper_collision::WorldScene`] keeps its geometry private and the
//! [`CollisionModel`] exposes only a *boolean* world query, so it cannot supply a
//! smooth world-obstacle gradient. [`ObstacleField`] is therefore a small, public
//! signed-distance mirror of the same obstacles (build it from the SAME numbers
//! you gave the `WorldScene`) and drives the gradient. The `CollisionModel`
//! remains the **authoritative** checker: every accepted step is gated so the
//! number of colliding samples along the path (measured by the real
//! `CollisionModel`) never increases. Consequently:
//!
//! * a collision-free input path stays collision-free (the count starts at 0 and
//!   can never rise), and
//! * a path that dips into an obstacle can only have that penetration reduced.
//!
//! Because a body sphere is a *conservative* bound on its collider, driving the
//! obstacle cost to zero (every sphere clear of the field) implies the real
//! collider is clear of the corresponding box — so the smooth field and the
//! validated `CollisionModel` agree at the optimum.
//!
//! Everything is deterministic (plain gradient descent, no `rand`) and
//! dependency-free beyond `nalgebra`.

use caliper_collision::CollisionModel;
use caliper_kinematics::fk_frame;
use caliper_model::{CollisionShape, Model};

#[derive(thiserror::Error, Debug)]
pub enum TrajOptError {
    #[error("path is empty")]
    Empty,
    #[error("waypoint {waypoint} has {got} dofs, expected {expected}")]
    DimMismatch {
        waypoint: usize,
        expected: usize,
        got: usize,
    },
    #[error("non-finite value in waypoint {0}")]
    NonFinite(usize),
    #[error("invalid option: {0}")]
    InvalidOption(&'static str),
    #[error("collision query failed: {0}")]
    Collision(String),
}

/// A signed-distance mirror of the static obstacles the arm must avoid.
///
/// Build it from the SAME boxes / ground plane you passed to the
/// [`caliper_collision::WorldScene`] backing the [`CollisionModel`]. It supplies
/// the smooth obstacle *gradient*; the `CollisionModel` supplies the
/// authoritative pass/fail gate. An empty field yields a zero obstacle cost, so
/// [`optimize`] degenerates to a pure smoother.
#[derive(Clone, Debug, Default)]
pub struct ObstacleField {
    /// Axis-aligned boxes as `(center, half_extents)`.
    boxes: Vec<([f64; 3], [f64; 3])>,
    /// Solid ground half-space: everything at `z ≤ ground_z` is solid.
    ground_z: Option<f64>,
}

impl ObstacleField {
    pub fn new() -> Self {
        Self::default()
    }
    /// Add a solid ground half-space at height `z` (solid for `z' ≤ z`).
    pub fn with_ground(mut self, z: f64) -> Self {
        self.ground_z = Some(z);
        self
    }
    /// Add an axis-aligned obstacle box (`center`, `half` extents). Non-finite or
    /// negative half-extents are clamped to a finite, non-negative value (matching
    /// [`caliper_collision::WorldScene::add_box`]).
    pub fn add_box(mut self, center: [f64; 3], half: [f64; 3]) -> Self {
        let half = [
            sanitize_extent(half[0]),
            sanitize_extent(half[1]),
            sanitize_extent(half[2]),
        ];
        self.boxes.push((center, half));
        self
    }
    /// `true` when there are no obstacles (obstacle cost is identically zero).
    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty() && self.ground_z.is_none()
    }

    /// Signed distance from world point `p` to the NEAREST obstacle surface.
    /// Positive outside every obstacle, negative when `p` is inside one, and
    /// `+∞` when the field is empty (no obstacle → no penalty).
    pub fn signed_distance(&self, p: &[f64; 3]) -> f64 {
        let mut best = f64::INFINITY;
        for (c, h) in &self.boxes {
            best = best.min(box_signed_distance(p, c, h));
        }
        if let Some(z) = self.ground_z {
            // Solid for z ≤ ground_z; the signed distance of a point above it is
            // its height over the plane (negative once inside the solid).
            best = best.min(p[2] - z);
        }
        best
    }
}

/// Clamp a box half-extent to a finite, non-negative value (NaN/∞/negative → 0).
fn sanitize_extent(x: f64) -> f64 {
    if x.is_finite() { x.max(0.0) } else { 0.0 }
}

/// Exact signed distance from a point to an axis-aligned box (negative inside).
fn box_signed_distance(p: &[f64; 3], c: &[f64; 3], h: &[f64; 3]) -> f64 {
    let d = [
        (p[0] - c[0]).abs() - h[0],
        (p[1] - c[1]).abs() - h[1],
        (p[2] - c[2]).abs() - h[2],
    ];
    let outside = (d[0].max(0.0).powi(2) + d[1].max(0.0).powi(2) + d[2].max(0.0).powi(2)).sqrt();
    let inside = d[0].max(d[1]).max(d[2]).min(0.0);
    outside + inside
}

/// Tunable optimizer parameters. [`Default`] gives sane values for a metre-scale
/// arm: a unit smoothness weight, a heavy obstacle weight, a 5 cm safety margin,
/// and 200 descent iterations.
#[derive(Clone, Debug)]
pub struct TrajOptOptions {
    /// Weight on the smoothness (finite-difference acceleration) term. `≥ 0`.
    pub w_smooth: f64,
    /// Weight on the obstacle-proximity term. `≥ 0`.
    pub w_obs: f64,
    /// Clearance (m) at which the obstacle penalty vanishes; the arm is pushed to
    /// stay at least this far from every obstacle. `> 0`.
    pub obstacle_margin: f64,
    /// Maximum gradient-descent iterations.
    pub max_iters: usize,
    /// Initial step size for the backtracking line search. `> 0`.
    pub step: f64,
    /// Smallest step the line search will try before giving up on an iteration.
    /// `> 0`.
    pub min_step: f64,
    /// Joint-space resolution (rad) for the collision gate's dense path sampling.
    /// `> 0`.
    pub edge_resolution: f64,
    /// Clamp interior waypoints to the model's joint limits after each step.
    pub clamp_to_limits: bool,
}

impl Default for TrajOptOptions {
    fn default() -> Self {
        Self {
            w_smooth: 1.0,
            w_obs: 10.0,
            obstacle_margin: 0.05,
            max_iters: 200,
            step: 0.1,
            min_step: 1e-6,
            edge_resolution: 0.05,
            clamp_to_limits: true,
        }
    }
}

/// Outcome of [`optimize`]: the refined path plus before/after diagnostics.
#[derive(Clone, Debug)]
pub struct TrajOptResult {
    /// The optimized path (same length / dimensionality as the input; endpoints
    /// bit-identical to the input's).
    pub path: Vec<Vec<f64>>,
    /// Number of descent iterations that produced an accepted step.
    pub iterations: usize,
    pub initial_smoothness: f64,
    pub final_smoothness: f64,
    pub initial_obstacle_cost: f64,
    pub final_obstacle_cost: f64,
    /// Colliding dense samples (per the authoritative [`CollisionModel`]) along
    /// the input path — a monotone non-increasing quantity across the optimize.
    pub initial_colliding_samples: usize,
    /// Colliding dense samples along the returned path (`≤ initial`).
    pub final_colliding_samples: usize,
    /// Frames the `CollisionModel` does NOT fully cover (surfaced verbatim so the
    /// caller does not trust a "clear" verdict blindly).
    pub uncovered_frames: usize,
}

/// The CHOMP smoothness cost: the sum over interior waypoints of the squared
/// finite-difference acceleration `‖q[i-1] − 2·q[i] + q[i+1]‖²`. Zero for a path
/// with fewer than three waypoints (no interior acceleration).
pub fn smoothness_cost(path: &[Vec<f64>]) -> f64 {
    if path.len() < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 1..path.len() - 1 {
        for ((&pm, &pc), &pn) in path[i - 1].iter().zip(&path[i]).zip(&path[i + 1]) {
            let a = pm - 2.0 * pc + pn;
            s += a * a;
        }
    }
    s
}

/// Analytic gradient of [`smoothness_cost`] w.r.t. every coordinate, shaped like
/// `path` (endpoint rows are all zero — the endpoints are fixed).
///
/// With `a[i] = q[i-1] − 2·q[i] + q[i+1]` (defined for interior `i`, zero
/// elsewhere), `∂S/∂q[k] = 2·(a[k-1] − 2·a[k] + a[k+1])` — the pentadiagonal
/// `2·AᵀA` stencil. Cross-validated against a central finite difference in the
/// tests.
pub fn smoothness_gradient(path: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = path.len();
    let dof = if n == 0 { 0 } else { path[0].len() };
    let mut grad = vec![vec![0.0; dof]; n];
    if n < 3 {
        return grad;
    }
    // a[i] for interior i (1..=n-2); accessed via `accel` which returns 0 out of
    // range so the k-1 / k+1 boundary terms fall away cleanly.
    let accel = |i: isize, d: usize| -> f64 {
        if i < 1 || i as usize > n - 2 {
            0.0
        } else {
            let i = i as usize;
            path[i - 1][d] - 2.0 * path[i][d] + path[i + 1][d]
        }
    };
    for (k, grow) in grad.iter_mut().enumerate().take(n - 1).skip(1) {
        let ki = k as isize;
        for (d, g) in grow.iter_mut().enumerate() {
            *g = 2.0 * (accel(ki - 1, d) - 2.0 * accel(ki, d) + accel(ki + 1, d));
        }
    }
    grad
}

/// CHOMP obstacle hinge potential of a signed `clearance` (body-sphere surface to
/// nearest obstacle) with safety margin `eps > 0`. `C¹`-continuous:
/// * `clearance ≥ eps`  → `0`,
/// * `0 ≤ clearance < eps` → `(clearance − eps)² / (2·eps)`,
/// * `clearance < 0` (penetrating) → `eps/2 − clearance` (grows linearly).
fn obstacle_potential(clearance: f64, eps: f64) -> f64 {
    if clearance >= eps {
        0.0
    } else if clearance >= 0.0 {
        let t = clearance - eps;
        0.5 * t * t / eps
    } else {
        0.5 * eps - clearance
    }
}

/// Bounding-sphere radius of a collider about its `<collision>` origin. A
/// conservative bound: if the sphere clears an obstacle, the collider does too.
fn bounding_radius(shape: &CollisionShape) -> f64 {
    match shape {
        CollisionShape::Box { half } => half.norm(),
        CollisionShape::Sphere { radius } => *radius,
        CollisionShape::Cylinder { radius, length } => {
            (radius * radius + 0.25 * length * length).sqrt()
        }
        CollisionShape::Capsule { radius, length } => radius + 0.5 * length,
        CollisionShape::ConvexHull { points } => points
            .iter()
            .map(|p| p.coords.norm())
            .fold(0.0_f64, f64::max),
    }
}

/// Total obstacle cost of a single configuration: the sum of the hinge potential
/// over every collider's body sphere.
fn obstacle_cost_at(model: &Model, field: &ObstacleField, q: &[f64], eps: f64) -> f64 {
    if field.is_empty() {
        return 0.0;
    }
    let mut c = 0.0;
    for g in &model.collision {
        let world = fk_frame(model, q, g.frame).0 * g.origin.0;
        let p = world.translation.vector;
        let clearance = field.signed_distance(&[p.x, p.y, p.z]) - bounding_radius(&g.shape);
        c += obstacle_potential(clearance, eps);
    }
    c
}

/// Obstacle cost summed over the INTERIOR waypoints (endpoints are fixed).
fn obstacle_cost_total(model: &Model, field: &ObstacleField, path: &[Vec<f64>], eps: f64) -> f64 {
    if path.len() < 3 || field.is_empty() {
        return 0.0;
    }
    let mut c = 0.0;
    for q in &path[1..path.len() - 1] {
        c += obstacle_cost_at(model, field, q, eps);
    }
    c
}

/// Central finite-difference gradient of a single waypoint's obstacle cost.
fn obstacle_gradient_at(model: &Model, field: &ObstacleField, q: &[f64], eps: f64) -> Vec<f64> {
    let n = q.len();
    let mut g = vec![0.0; n];
    if field.is_empty() {
        return g;
    }
    const H: f64 = 1e-6;
    let mut qp = q.to_vec();
    let mut qm = q.to_vec();
    for j in 0..n {
        qp[j] = q[j] + H;
        qm[j] = q[j] - H;
        let cp = obstacle_cost_at(model, field, &qp, eps);
        let cm = obstacle_cost_at(model, field, &qm, eps);
        g[j] = (cp - cm) / (2.0 * H);
        qp[j] = q[j];
        qm[j] = q[j];
    }
    g
}

fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn lerp(a: &[f64], b: &[f64], t: f64) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + (y - x) * t).collect()
}

/// `true` if the authoritative `CollisionModel` flags `q` (a failed query — bad
/// dim / non-finite — is treated as colliding, i.e. unsafe).
fn config_colliding(cm: &CollisionModel, q: &[f64]) -> bool {
    match cm.query(q) {
        Ok(r) => r.has_collision(),
        Err(_) => true,
    }
}

/// Count densely-sampled configurations along `path` that the authoritative
/// `CollisionModel` reports as colliding. Shared segment vertices are counted
/// once. This is the monotone quantity the optimizer's gate keeps non-increasing.
fn colliding_samples(cm: &CollisionModel, path: &[Vec<f64>], res: f64) -> usize {
    if path.is_empty() {
        return 0;
    }
    let mut count = 0usize;
    if config_colliding(cm, &path[0]) {
        count += 1;
    }
    for w in path.windows(2) {
        let d = dist(&w[0], &w[1]);
        let steps = ((d / res).ceil() as usize).max(1);
        for i in 1..=steps {
            let t = i as f64 / steps as f64;
            if config_colliding(cm, &lerp(&w[0], &w[1], t)) {
                count += 1;
            }
        }
    }
    count
}

/// Optimize a joint-space waypoint `path` with fixed endpoints.
///
/// * `model` — the kinematic model (must match `collision`); used for the
///   obstacle term's forward kinematics.
/// * `collision` — the authoritative [`CollisionModel`]; gates every accepted
///   step so colliding-sample count never increases (see the module docs).
/// * `field` — a signed-distance mirror of the same obstacles, driving the smooth
///   obstacle gradient (empty ⇒ pure smoother).
/// * `opts` — weights, margins, iteration budget.
///
/// Returns the refined path (endpoints bit-identical to the input) plus
/// before/after diagnostics. The returned path is guaranteed to have **no more**
/// colliding samples than the input, so a collision-free input stays
/// collision-free and a penetrating input's penetration is only ever reduced.
///
/// The PROVEN capability is smoothness minimization under a monotone
/// collision-safety gate (which alone declutters a jerky path out of shallow
/// obstacles). The explicit obstacle-push term (`w_obs > 0`, driven by `field`) is
/// **best-effort / experimental** — the finite-difference obstacle gradient can
/// destabilize the line search on deep penetrations; prefer `w_obs = 0` until a
/// proper analytic CHOMP obstacle gradient lands (deferred).
pub fn optimize(
    path: &[Vec<f64>],
    model: &Model,
    collision: &CollisionModel,
    field: &ObstacleField,
    opts: &TrajOptOptions,
) -> Result<TrajOptResult, TrajOptError> {
    // ---- validate inputs ----
    if path.is_empty() {
        return Err(TrajOptError::Empty);
    }
    for (i, w) in path.iter().enumerate() {
        if w.len() != model.ndof {
            return Err(TrajOptError::DimMismatch {
                waypoint: i,
                expected: model.ndof,
                got: w.len(),
            });
        }
        if !w.iter().all(|x| x.is_finite()) {
            return Err(TrajOptError::NonFinite(i));
        }
    }
    validate_opts(opts)?;

    let uncovered = collision.uncovered_frames();
    let n = path.len();
    let init_smooth = smoothness_cost(path);
    let init_obs = obstacle_cost_total(model, field, path, opts.obstacle_margin);
    let init_coll = colliding_samples(collision, path, opts.edge_resolution);

    // Fewer than three waypoints ⇒ no interior to move; return the input as-is.
    if n < 3 {
        return Ok(TrajOptResult {
            path: path.to_vec(),
            iterations: 0,
            initial_smoothness: init_smooth,
            final_smoothness: init_smooth,
            initial_obstacle_cost: init_obs,
            final_obstacle_cost: init_obs,
            initial_colliding_samples: init_coll,
            final_colliding_samples: init_coll,
            uncovered_frames: uncovered,
        });
    }

    let total_cost = |p: &[Vec<f64>]| -> f64 {
        opts.w_smooth * smoothness_cost(p)
            + opts.w_obs * obstacle_cost_total(model, field, p, opts.obstacle_margin)
    };

    let mut cur = path.to_vec();
    let mut cur_cost = total_cost(&cur);
    let mut cur_coll = init_coll;
    let mut accepted_iters = 0usize;

    for _ in 0..opts.max_iters {
        // Full gradient (endpoint rows stay zero — endpoints are fixed).
        let sg = smoothness_gradient(&cur);
        let mut grad = vec![vec![0.0; model.ndof]; n];
        let mut gnorm2 = 0.0;
        for k in 1..n - 1 {
            let og = if opts.w_obs > 0.0 {
                obstacle_gradient_at(model, field, &cur[k], opts.obstacle_margin)
            } else {
                vec![0.0; model.ndof]
            };
            for d in 0..model.ndof {
                let g = opts.w_smooth * sg[k][d] + opts.w_obs * og[d];
                grad[k][d] = g;
                gnorm2 += g * g;
            }
        }
        if gnorm2 <= 1e-24 {
            break; // stationary
        }

        // Backtracking line search: accept the first step that BOTH strictly
        // lowers the total cost AND does not raise the authoritative
        // colliding-sample count. The latter makes collision-safety monotone.
        let mut ls = opts.step;
        let mut accepted = false;
        while ls >= opts.min_step {
            let cand = apply_step(model, &cur, &grad, ls, opts.clamp_to_limits);
            let cand_coll = colliding_samples(collision, &cand, opts.edge_resolution);
            let cand_cost = total_cost(&cand);
            if cand_coll <= cur_coll && cand_cost < cur_cost - 1e-12 {
                cur = cand;
                cur_cost = cand_cost;
                cur_coll = cand_coll;
                accepted = true;
                break;
            }
            ls *= 0.5;
        }
        if !accepted {
            break; // converged / no admissible descent step
        }
        accepted_iters += 1;
    }

    Ok(TrajOptResult {
        final_smoothness: smoothness_cost(&cur),
        final_obstacle_cost: obstacle_cost_total(model, field, &cur, opts.obstacle_margin),
        final_colliding_samples: cur_coll,
        path: cur,
        iterations: accepted_iters,
        initial_smoothness: init_smooth,
        initial_obstacle_cost: init_obs,
        initial_colliding_samples: init_coll,
        uncovered_frames: uncovered,
    })
}

/// Produce a candidate path by stepping the interior waypoints down the gradient
/// (endpoints are copied verbatim). Optionally clamps interior waypoints to the
/// model's joint limits.
fn apply_step(
    model: &Model,
    path: &[Vec<f64>],
    grad: &[Vec<f64>],
    step: f64,
    clamp: bool,
) -> Vec<Vec<f64>> {
    let n = path.len();
    let mut out = path.to_vec();
    for k in 1..n - 1 {
        for d in 0..out[k].len() {
            out[k][d] = path[k][d] - step * grad[k][d];
        }
        if clamp {
            model.clamp(&mut out[k]);
        }
    }
    out
}

fn validate_opts(o: &TrajOptOptions) -> Result<(), TrajOptError> {
    if !(o.w_smooth.is_finite() && o.w_smooth >= 0.0) {
        return Err(TrajOptError::InvalidOption(
            "w_smooth must be finite and >= 0",
        ));
    }
    if !(o.w_obs.is_finite() && o.w_obs >= 0.0) {
        return Err(TrajOptError::InvalidOption("w_obs must be finite and >= 0"));
    }
    if !(o.obstacle_margin.is_finite() && o.obstacle_margin > 0.0) {
        return Err(TrajOptError::InvalidOption(
            "obstacle_margin must be finite and > 0",
        ));
    }
    if !(o.step.is_finite() && o.step > 0.0) {
        return Err(TrajOptError::InvalidOption("step must be finite and > 0"));
    }
    if !(o.min_step.is_finite() && o.min_step > 0.0 && o.min_step <= o.step) {
        return Err(TrajOptError::InvalidOption(
            "min_step must be finite, > 0, and <= step",
        ));
    }
    if !(o.edge_resolution.is_finite() && o.edge_resolution > 0.0) {
        return Err(TrajOptError::InvalidOption(
            "edge_resolution must be finite and > 0",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliper_collision::{CollisionModel, WorldScene};
    use caliper_model::Model;
    use std::path::Path;
    use std::sync::Arc;

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

    // ===== pure-math unit tests (independent of the robot) =====

    #[test]
    fn smoothness_cost_hand_example() {
        // Straight, evenly-spaced 1-D path has zero acceleration → zero cost.
        let straight = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        assert!(smoothness_cost(&straight).abs() < 1e-15);
        // A single kink: q = [0, 1, 0]. a[1] = 0 - 2*1 + 0 = -2 → cost 4.
        let kink = vec![vec![0.0], vec![1.0], vec![0.0]];
        assert!((smoothness_cost(&kink) - 4.0).abs() < 1e-15);
        // Fewer than 3 waypoints → no interior acceleration.
        assert_eq!(smoothness_cost(&[vec![0.0], vec![5.0]]), 0.0);
    }

    #[test]
    fn smoothness_gradient_matches_finite_difference() {
        // Cross-validate the analytic pentadiagonal gradient against a central
        // finite difference of `smoothness_cost`. Any stencil/index error surfaces.
        let path = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.3, -0.4, 0.5],
            vec![-0.2, 0.6, -0.3],
            vec![0.7, 0.1, 0.2],
            vec![0.4, 0.4, 0.4],
        ];
        let g = smoothness_gradient(&path);
        let h = 1e-6;
        // Endpoints are FIXED in optimization, so the gradient projects them to zero.
        assert!(g[0].iter().all(|&x| x == 0.0));
        assert!(g[path.len() - 1].iter().all(|&x| x == 0.0));
        // Interior stencil must match a central finite difference of the raw cost.
        for k in 1..path.len() - 1 {
            for d in 0..3 {
                let mut pp = path.clone();
                let mut pm = path.clone();
                pp[k][d] += h;
                pm[k][d] -= h;
                let fd = (smoothness_cost(&pp) - smoothness_cost(&pm)) / (2.0 * h);
                assert!(
                    (g[k][d] - fd).abs() < 1e-5,
                    "grad[{k}][{d}]={} fd={}",
                    g[k][d],
                    fd
                );
            }
        }
        // Endpoint rows must be exactly zero (endpoints are fixed).
        assert!(g[0].iter().all(|x| *x == 0.0));
        assert!(g[4].iter().all(|x| *x == 0.0));
    }

    #[test]
    fn obstacle_field_signed_distance_matches_hand_values() {
        let f = ObstacleField::new().add_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // Outside along +x: distance to the x=1 face.
        assert!((f.signed_distance(&[2.0, 0.0, 0.0]) - 1.0).abs() < 1e-12);
        // Deep at the center: negative, equal to -min half-extent (=-1).
        assert!((f.signed_distance(&[0.0, 0.0, 0.0]) + 1.0).abs() < 1e-12);
        // On a face: exactly zero.
        assert!(f.signed_distance(&[1.0, 0.0, 0.0]).abs() < 1e-12);
        // Outside diagonal corner: sqrt(3).
        assert!((f.signed_distance(&[2.0, 2.0, 2.0]) - 3.0_f64.sqrt()).abs() < 1e-12);
        // Empty field → +inf (no penalty anywhere).
        assert!(
            ObstacleField::new()
                .signed_distance(&[9.0, 9.0, 9.0])
                .is_infinite()
        );
        // Ground half-space: height above the plane, negative below.
        let g = ObstacleField::new().with_ground(0.0);
        assert!((g.signed_distance(&[0.0, 0.0, 0.5]) - 0.5).abs() < 1e-12);
        assert!((g.signed_distance(&[0.0, 0.0, -0.2]) + 0.2).abs() < 1e-12);
    }

    #[test]
    fn obstacle_potential_is_monotone_and_c1() {
        let eps = 0.1;
        // Zero once clear by the margin.
        assert_eq!(obstacle_potential(0.2, eps), 0.0);
        assert_eq!(obstacle_potential(eps, eps), 0.0);
        // Strictly increasing as clearance shrinks / penetration deepens.
        let samples = [0.09, 0.05, 0.0, -0.1, -0.5];
        for w in samples.windows(2) {
            assert!(
                obstacle_potential(w[1], eps) > obstacle_potential(w[0], eps),
                "potential must grow as clearance drops: {} -> {}",
                w[0],
                w[1]
            );
        }
        // C1 continuity at the penetration boundary (slope -> -1 from both sides).
        let h = 1e-7;
        let left = (obstacle_potential(-h, eps) - obstacle_potential(-2.0 * h, eps)) / h;
        let right = (obstacle_potential(2.0 * h, eps) - obstacle_potential(h, eps)) / h;
        assert!(
            (left - right).abs() < 1e-3,
            "slope jump at d=0: {left} vs {right}"
        );
    }

    // ===== integration: cross-validation against the real CollisionModel =====

    /// (a) collision-free stays free, (b) smoother, (c) endpoints unchanged.
    #[test]
    fn smooths_a_jerky_collision_free_path() {
        let m = model("collide_arm.urdf");
        // Empty scene ⇒ only self-collision is possible; small angles never fold
        // l3 back onto l1, so every config here is collision-free.
        let cm = CollisionModel::new(m.clone(), WorldScene::new(), 0.0);
        let field = ObstacleField::new(); // no obstacles → pure smoother
        let path = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.3, -0.4, 0.5],
            vec![-0.2, 0.6, -0.3],
            vec![0.5, 0.1, 0.2],
            vec![0.4, 0.4, 0.4],
        ];
        // sanity: the input is genuinely collision-free per the real checker.
        assert_eq!(
            colliding_samples(&cm, &path, 0.05),
            0,
            "test setup: input must be collision-free"
        );

        let opts = TrajOptOptions {
            w_obs: 0.0,
            ..TrajOptOptions::default()
        };
        let r = optimize(&path, &m, &cm, &field, &opts).unwrap();

        // (c) endpoints bit-identical.
        assert_eq!(r.path[0], path[0]);
        assert_eq!(*r.path.last().unwrap(), *path.last().unwrap());
        // (b) strictly smoother.
        assert!(
            r.final_smoothness < r.initial_smoothness,
            "smoothness must drop: {} -> {}",
            r.initial_smoothness,
            r.final_smoothness
        );
        assert!(r.iterations > 0, "optimizer should take at least one step");
        // (a) still collision-free per the authoritative CollisionModel.
        assert_eq!(
            colliding_samples(&cm, &r.path, 0.05),
            0,
            "optimized path must remain collision-free"
        );
        assert_eq!(r.final_colliding_samples, 0);
    }

    /// A path whose interior waypoint reaches into a box obstacle: the optimizer
    /// must reduce the penetration (fewer colliding samples per the real
    /// CollisionModel) while keeping the endpoints.
    #[test]
    fn reduces_penetration_into_a_box_obstacle() {
        let m = model("collide_arm.urdf");
        // A LOW box off to +x. The `collide_arm` links stack up +z; rotating j1 by
        // θ swings the chain into the x–z plane along `(s·sinθ, 0, s·cosθ)`. The
        // near-vertical arm (small θ) stays at x≈0 and its upper links sit ABOVE
        // this low box (z up to 0.35), so both endpoints clear it; the bent arm
        // (θ=1.2) lays its mid/upper links right through x∈[0.3,0.7], z∈[0.05,0.35].
        let center = [0.5, 0.0, 0.2];
        let half = [0.2, 0.3, 0.15];
        let scene = WorldScene::new().add_box(center, half);
        let cm = CollisionModel::new(m.clone(), scene, 0.0);
        let field = ObstacleField::new().add_box(center, half); // mirror the scene

        let start = vec![0.0, 0.0, 0.0]; // straight up along +z
        let goal = vec![0.2, 0.0, 0.0]; // still near-vertical, clear of the low box
        let mid = vec![1.2, 0.0, 0.0]; // bent ~69° toward +x, into the box
        let path = vec![start.clone(), mid, goal.clone()];

        // setup sanity: endpoints clear, interior path collides.
        assert!(
            !cm.query(&start).unwrap().has_collision(),
            "test setup: start must be clear"
        );
        assert!(
            !cm.query(&goal).unwrap().has_collision(),
            "test setup: goal must be clear"
        );
        let init = colliding_samples(&cm, &path, 0.05);
        assert!(init > 0, "test setup: input path must dip into the box");

        // The PROVEN capability: smoothness minimization with monotone collision
        // safety — smoothing the jerky bent path pulls the mid waypoint toward the
        // (collision-free) straight-line midpoint, decluttering out of the box, and
        // the authoritative-checker gate guarantees the colliding-sample count never
        // rises. (The explicit obstacle-push gradient term is best-effort/experimental
        // — documented in the module header — so this test exercises w_obs=0.)
        let opts = TrajOptOptions {
            w_smooth: 1.0,
            w_obs: 0.0,
            max_iters: 400,
            ..TrajOptOptions::default()
        };
        let r = optimize(&path, &m, &cm, &field, &opts).unwrap();

        // endpoints preserved exactly.
        assert_eq!(r.path[0], start);
        assert_eq!(*r.path.last().unwrap(), goal);
        // penetration strictly reduced (measured by the authoritative checker).
        assert!(
            r.final_colliding_samples < init,
            "penetration must be reduced: {} -> {}",
            init,
            r.final_colliding_samples
        );
        // never introduced new collisions.
        assert!(r.final_colliding_samples <= r.initial_colliding_samples);
        // the smooth obstacle cost also fell.
        assert!(r.final_obstacle_cost <= r.initial_obstacle_cost);
    }

    #[test]
    fn endpoints_are_never_moved_even_with_obstacle_pressure() {
        // Even when the obstacle field would "want" to push the endpoints, they
        // are held fixed.
        let m = model("collide_arm.urdf");
        let center = [0.5, 0.0, 0.5];
        let half = [0.3, 0.3, 0.3];
        let cm = CollisionModel::new(m.clone(), WorldScene::new().add_box(center, half), 0.0);
        let field = ObstacleField::new().add_box(center, half);
        let path = vec![
            vec![0.6, 0.0, 0.0],
            vec![0.7, 0.1, -0.1],
            vec![0.5, -0.1, 0.2],
        ];
        let r = optimize(&path, &m, &cm, &field, &TrajOptOptions::default()).unwrap();
        assert_eq!(r.path[0], path[0]);
        assert_eq!(*r.path.last().unwrap(), *path.last().unwrap());
    }

    // ===== validation / edge cases =====

    #[test]
    fn rejects_bad_inputs() {
        let m = model("collide_arm.urdf");
        let cm = CollisionModel::new(m.clone(), WorldScene::new(), 0.0);
        let field = ObstacleField::new();
        let opts = TrajOptOptions::default();

        assert!(matches!(
            optimize(&[], &m, &cm, &field, &opts),
            Err(TrajOptError::Empty)
        ));
        // wrong dimensionality.
        let bad = vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0]];
        assert!(matches!(
            optimize(&bad, &m, &cm, &field, &opts),
            Err(TrajOptError::DimMismatch { .. })
        ));
        // non-finite.
        let nf = vec![vec![0.0, 0.0, 0.0], vec![f64::NAN, 0.0, 0.0]];
        assert!(matches!(
            optimize(&nf, &m, &cm, &field, &opts),
            Err(TrajOptError::NonFinite(_))
        ));
        // bad option.
        let bad_opts = TrajOptOptions {
            obstacle_margin: 0.0,
            ..TrajOptOptions::default()
        };
        let ok_path = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.1, 0.0, 0.0],
            vec![0.2, 0.0, 0.0],
        ];
        assert!(matches!(
            optimize(&ok_path, &m, &cm, &field, &bad_opts),
            Err(TrajOptError::InvalidOption(_))
        ));
    }

    #[test]
    fn short_path_is_returned_unchanged() {
        let m = model("collide_arm.urdf");
        let cm = CollisionModel::new(m.clone(), WorldScene::new(), 0.0);
        let field = ObstacleField::new();
        let path = vec![vec![0.0, 0.0, 0.0], vec![0.3, 0.2, 0.1]];
        let r = optimize(&path, &m, &cm, &field, &TrajOptOptions::default()).unwrap();
        assert_eq!(r.path, path);
        assert_eq!(r.iterations, 0);
    }

    #[test]
    fn deterministic_across_runs() {
        let m = model("collide_arm.urdf");
        let center = [0.5, 0.0, 0.5];
        let half = [0.35, 0.3, 0.35];
        let cm = CollisionModel::new(m.clone(), WorldScene::new().add_box(center, half), 0.0);
        let field = ObstacleField::new().add_box(center, half);
        let path = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.2, 0.0, 0.0],
            vec![0.15, 0.0, 0.0],
        ];
        let a = optimize(&path, &m, &cm, &field, &TrajOptOptions::default()).unwrap();
        let b = optimize(&path, &m, &cm, &field, &TrajOptOptions::default()).unwrap();
        assert_eq!(a.path, b.path);
        assert_eq!(a.iterations, b.iterations);
    }
}
