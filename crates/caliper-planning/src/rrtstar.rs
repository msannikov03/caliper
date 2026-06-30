//! RRT* — single-tree, asymptotically-optimal sampling planner.
//!
//! Same validity model and joint-space geometry as RRT-Connect (`rrt.rs`): edges
//! are collision-checked at `edge_resolution` via the caller-supplied
//! `motion_valid`, samples are drawn from the per-joint bounds with the same
//! deterministic splitmix64 PRNG (`rng.rs`) and the same fixed draw order (bias
//! coin first, then one uniform per joint). The difference is optimization: RRT*
//! grows a single start-rooted tree, connects each new node through the
//! minimal-cost-to-come near neighbour (cost = joint-space path length), and
//! *rewires* near neighbours through the new node whenever that is cheaper. The
//! near-neighbour radius shrinks as `r(n) = min(gamma*(ln n / n)^(1/d), eta)`,
//! which (with a large enough `gamma`) gives almost-sure convergence to the
//! shortest path. The goal is not a tree node; instead every node records whether
//! a straight edge to the goal is collision-free, and the returned plan is the
//! cheapest such node's root-path followed by the goal.

use crate::rng::Rng;
use crate::rrt::{dist, steer};
use std::f64::consts::PI;

/// RRT* tuning, derived from [`crate::PlannerConfig`] by the caller.
pub(crate) struct StarParams {
    /// PRNG seed (same seed ⇒ identical tree ⇒ identical plan).
    pub seed: u64,
    /// Sampling/iteration budget (> 0, validated by the caller).
    pub iters: usize,
    /// Max joint-space steer step (rad) — also the near-radius cap `eta`.
    pub step: f64,
    /// Probability of sampling the goal directly.
    pub goal_bias: f64,
}

/// Volume of the unit `d`-ball, via `V_d = (2*pi/d) * V_{d-2}`, `V_0=1`, `V_1=2`.
fn unit_ball_volume(d: usize) -> f64 {
    let even = d.is_multiple_of(2);
    let mut v = if even { 1.0 } else { 2.0 };
    let mut k = if even { 2 } else { 3 };
    while k <= d {
        v *= 2.0 * PI / (k as f64);
        k += 2;
    }
    v
}

/// The RRT* radius constant. Uses the bounding-box volume as an upper bound on the
/// free-space measure (over-estimating only inflates the radius, which preserves
/// optimality), with a small safety factor so it stays strictly above the
/// asymptotic-optimality threshold. Falls back to `2*step` if the geometry is
/// degenerate.
fn radius_gamma(bounds: &[(f64, f64)], step: f64) -> f64 {
    let d = bounds.len() as f64;
    let mu: f64 = bounds.iter().map(|&(lo, hi)| (hi - lo).max(0.0)).product();
    let zeta = unit_ball_volume(bounds.len());
    let g = 2.0 * (1.0 + 1.0 / d).powf(1.0 / d) * (mu / zeta).powf(1.0 / d) * 1.1;
    if g.is_finite() && g > 0.0 {
        g.max(2.0 * step)
    } else {
        2.0 * step
    }
}

/// A start-rooted tree tracking cost-to-come and children, so rewiring can
/// reparent a node and push the cost delta to its whole subtree.
struct StarTree {
    nodes: Vec<Vec<f64>>,
    parent: Vec<usize>,
    children: Vec<Vec<usize>>,
    cost: Vec<f64>,
}

impl StarTree {
    fn new(root: Vec<f64>) -> Self {
        Self {
            nodes: vec![root],
            parent: vec![usize::MAX],
            children: vec![Vec::new()],
            cost: vec![0.0],
        }
    }
    fn len(&self) -> usize {
        self.nodes.len()
    }
    /// Nearest node to `q` (linear scan, squared Euclidean) — matches `Tree`.
    fn nearest(&self, q: &[f64]) -> usize {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, n) in self.nodes.iter().enumerate() {
            let d: f64 = n.iter().zip(q).map(|(x, y)| (x - y) * (x - y)).sum();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }
    /// Indices of nodes within `r` of `q`, in ascending index order (deterministic).
    fn near(&self, q: &[f64], r: f64) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|&i| dist(&self.nodes[i], q) <= r)
            .collect()
    }
    fn add(&mut self, q: Vec<f64>, parent: usize, cost: f64) -> usize {
        let i = self.nodes.len();
        self.nodes.push(q);
        self.parent.push(parent);
        self.children.push(Vec::new());
        self.cost.push(cost);
        if parent != usize::MAX {
            self.children[parent].push(i);
        }
        i
    }
    /// Reparent `node` onto `new_parent` and refresh `node`'s subtree costs.
    fn reparent(&mut self, node: usize, new_parent: usize) {
        let old = self.parent[node];
        if old != usize::MAX {
            self.children[old].retain(|&c| c != node);
        }
        self.parent[node] = new_parent;
        self.children[new_parent].push(node);
        // Propagate the cost change down the subtree (parent cost already current).
        let mut stack = vec![node];
        while let Some(u) = stack.pop() {
            let p = self.parent[u];
            self.cost[u] = self.cost[p] + dist(&self.nodes[p], &self.nodes[u]);
            for &c in &self.children[u] {
                stack.push(c);
            }
        }
    }
    /// Root→`idx` configuration path (inclusive).
    fn path_to(&self, idx: usize) -> Vec<Vec<f64>> {
        let mut chain = Vec::new();
        let mut i = idx;
        while i != usize::MAX {
            chain.push(self.nodes[i].clone());
            i = self.parent[i];
        }
        chain.reverse();
        chain
    }
}

/// Run RRT* from `start` toward `goal` within the iteration budget. `motion_valid`
/// must return `true` iff the straight segment a→b is collision-free at the dense
/// planner resolution (this also validates the segment endpoints). Returns the
/// lowest-cost root→goal waypoint path found, or `None` if the goal was never
/// connected. Endpoints of the returned path equal `start` and `goal` exactly.
pub(crate) fn rrt_star(
    start: &[f64],
    goal: &[f64],
    bounds: &[(f64, f64)],
    p: &StarParams,
    motion_valid: impl Fn(&[f64], &[f64]) -> bool,
) -> Option<Vec<Vec<f64>>> {
    let d = start.len();
    let gamma = radius_gamma(bounds, p.step);
    let eta = p.step;

    let mut rng = Rng::new(p.seed);
    let mut tree = StarTree::new(start.to_vec());
    // Per-node: does a collision-free straight edge to the goal exist, and its cost.
    let mut to_goal: Vec<Option<f64>> = vec![goal_edge(start, goal, &motion_valid)];

    for _ in 0..p.iters {
        // --- sample (fixed RNG order: bias coin, then one uniform per joint) ---
        let q_rand: Vec<f64> = if rng.unit() < p.goal_bias {
            goal.to_vec()
        } else {
            bounds.iter().map(|&(lo, hi)| rng.range(lo, hi)).collect()
        };

        let near_idx = tree.nearest(&q_rand);
        let q_new = steer(&tree.nodes[near_idx], &q_rand, p.step);
        // Skip a no-op step (q_rand already at the nearest node).
        if dist(&tree.nodes[near_idx], &q_new) < 1e-12 {
            continue;
        }
        if !motion_valid(&tree.nodes[near_idx], &q_new) {
            continue;
        }

        // r(n) over the CURRENT node count, capped at eta.
        let n = tree.len() as f64;
        let r = if n > 1.0 {
            (gamma * (n.ln() / n).powf(1.0 / d as f64)).min(eta)
        } else {
            0.0
        };
        let near = tree.near(&q_new, r);

        // --- choose parent: minimal cost-to-come with a collision-free edge ---
        // The nearest node's edge is already validated; it is the baseline parent.
        let mut best_parent = near_idx;
        let mut best_cost = tree.cost[near_idx] + dist(&tree.nodes[near_idx], &q_new);
        for &m in &near {
            if m == near_idx {
                continue;
            }
            let c = tree.cost[m] + dist(&tree.nodes[m], &q_new);
            if c + 1e-12 < best_cost && motion_valid(&tree.nodes[m], &q_new) {
                best_cost = c;
                best_parent = m;
            }
        }

        let new_idx = tree.add(q_new.clone(), best_parent, best_cost);
        to_goal.push(goal_edge(&q_new, goal, &motion_valid));

        // --- rewire: route near nodes through the new node when cheaper ---
        for &m in &near {
            if m == best_parent || m == new_idx {
                continue;
            }
            let c = best_cost + dist(&q_new, &tree.nodes[m]);
            if c + 1e-12 < tree.cost[m] && motion_valid(&q_new, &tree.nodes[m]) {
                tree.reparent(m, new_idx);
            }
        }
    }

    // --- extract the cheapest goal-connected node (costs reflect all rewiring) ---
    let mut best: Option<(usize, f64)> = None;
    for (i, edge) in to_goal.iter().enumerate() {
        if let Some(edge) = edge {
            let total = tree.cost[i] + edge;
            if best.is_none_or(|(_, b)| total < b) {
                best = Some((i, total));
            }
        }
    }
    let (bi, _) = best?;
    let mut path = tree.path_to(bi);
    // Append the goal unless this node IS the goal (exact, from a goal-biased steer).
    if dist(path.last().unwrap(), goal) > 1e-12 {
        path.push(goal.to_vec());
    }
    Some(path)
}

/// `Some(len)` if a straight edge `q`→`goal` is collision-free, else `None`.
fn goal_edge(
    q: &[f64],
    goal: &[f64],
    motion_valid: &impl Fn(&[f64], &[f64]) -> bool,
) -> Option<f64> {
    if motion_valid(q, goal) {
        Some(dist(q, goal))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_ball_volumes() {
        assert!((unit_ball_volume(1) - 2.0).abs() < 1e-12);
        assert!((unit_ball_volume(2) - PI).abs() < 1e-12);
        assert!((unit_ball_volume(3) - 4.0 * PI / 3.0).abs() < 1e-12);
    }

    /// In a fully free 2-D box, RRT* should drive the path toward the straight
    /// line — a property a broken (non-optimizing) implementation would fail.
    #[test]
    fn free_space_approaches_optimal() {
        let bounds = [(-1.0, 1.0), (-1.0, 1.0)];
        let start = [-0.9, -0.9];
        let goal = [0.9, 0.9];
        let p = StarParams {
            seed: 0xCA11,
            iters: 4000,
            step: 0.2,
            goal_bias: 0.1,
        };
        let path = rrt_star(&start, &goal, &bounds, &p, |_, _| true).unwrap();
        assert_eq!(path.first().unwrap().as_slice(), &start);
        assert_eq!(path.last().unwrap().as_slice(), &goal);
        let len = crate::smooth::path_length(&path);
        let straight = dist(&start, &goal);
        assert!(
            len >= straight - 1e-9,
            "below the straight-line lower bound"
        );
        assert!(
            len <= straight * 1.10,
            "RRT* should near the optimum: {len} vs {straight}"
        );
    }

    /// Same seed ⇒ identical tree ⇒ identical path.
    #[test]
    fn deterministic_same_seed() {
        let bounds = [(-1.0, 1.0), (-1.0, 1.0)];
        let start = [-0.9, -0.9];
        let goal = [0.9, 0.9];
        let mk = || StarParams {
            seed: 7,
            iters: 800,
            step: 0.2,
            goal_bias: 0.1,
        };
        let a = rrt_star(&start, &goal, &bounds, &mk(), |_, _| true).unwrap();
        let b = rrt_star(&start, &goal, &bounds, &mk(), |_, _| true).unwrap();
        assert_eq!(a, b);
    }
}
