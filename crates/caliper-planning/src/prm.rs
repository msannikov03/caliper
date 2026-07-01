//! PRM — Probabilistic RoadMap, a *multi-query* sampling planner.
//!
//! Same validity model and joint-space geometry as RRT-Connect (`rrt.rs`) and
//! RRT* (`rrtstar.rs`): milestones are rejection-sampled from the per-joint
//! bounds with the same deterministic splitmix64 PRNG (`rng.rs`), edges are
//! collision-checked at `edge_resolution` via the caller-supplied `motion_valid`,
//! and a milestone is kept only when `config_free` holds. The difference is
//! *structure reuse*: a [`Roadmap`] of collision-free milestones is built ONCE
//! (each milestone wired to its `k` nearest collision-free neighbours), then any
//! number of start/goal queries are answered cheaply by connecting the endpoints
//! to the roadmap and running Dijkstra over joint-space edge lengths.
//!
//! Everything is deterministic: milestones are drawn in a fixed RNG order,
//! neighbours are ranked by distance with an index tie-break, and Dijkstra picks
//! the lowest-index node on cost ties — so a given seed yields the same roadmap
//! and the same query path. The returned waypoint path's endpoints equal the
//! query's `start`/`goal` exactly; the caller shortcut-smooths it.

use crate::rng::Rng;
use crate::rrt::dist;

/// PRM tuning, derived from [`crate::PlannerConfig`] + the query by the caller.
pub(crate) struct PrmParams {
    /// PRNG seed (same seed ⇒ identical roadmap ⇒ identical query path).
    pub seed: u64,
    /// Number of collision-free milestones to sample (> 0, validated by caller).
    pub samples: usize,
    /// Neighbours each node is connected to (> 0, validated by caller).
    pub k: usize,
}

/// A reusable roadmap of collision-free milestones and their collision-free
/// `k`-nearest-neighbour edges. Build once, query many times.
pub(crate) struct Roadmap {
    /// Milestone configurations (all collision-free at build time).
    nodes: Vec<Vec<f64>>,
    /// Undirected adjacency: `adj[i]` = `(neighbour, edge_length)` pairs.
    adj: Vec<Vec<(usize, f64)>>,
}

/// Rejection-sample `params.samples` collision-free milestones and wire each to
/// its `params.k` nearest collision-free neighbours. Sampling draws one uniform
/// per joint (fixed RNG order) and retries on a colliding draw, capped so a tiny
/// free-space cannot livelock. `config_free(q)` gates a milestone; `motion_valid`
/// gates an edge (both endpoints + the dense segment between them).
pub(crate) fn build_roadmap(
    bounds: &[(f64, f64)],
    params: &PrmParams,
    config_free: impl Fn(&[f64]) -> bool,
    motion_valid: impl Fn(&[f64], &[f64]) -> bool,
) -> Roadmap {
    let mut rng = Rng::new(params.seed);
    let mut nodes: Vec<Vec<f64>> = Vec::with_capacity(params.samples);

    // Bound total draws so a nearly-blocked scene fails fast instead of looping
    // forever; still generous enough to fill a modest free-space.
    let max_attempts = params.samples.saturating_mul(40).saturating_add(1000);
    let mut attempts = 0;
    while nodes.len() < params.samples && attempts < max_attempts {
        attempts += 1;
        let q: Vec<f64> = bounds.iter().map(|&(lo, hi)| rng.range(lo, hi)).collect();
        if config_free(&q) {
            nodes.push(q);
        }
    }

    let adj = connect_knn(&nodes, params.k, &motion_valid);
    Roadmap { nodes, adj }
}

/// Build the undirected k-NN adjacency over `nodes`: for each node, rank the
/// others by joint-space distance (index tie-break) and add a collision-free edge
/// to each of the `k` nearest, deduplicating symmetric pairs.
fn connect_knn(
    nodes: &[Vec<f64>],
    k: usize,
    motion_valid: &impl Fn(&[f64], &[f64]) -> bool,
) -> Vec<Vec<(usize, f64)>> {
    let n = nodes.len();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        let mut cand: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, dist(&nodes[i], &nodes[j])))
            .collect();
        cand.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        for &(j, d) in cand.iter().take(k) {
            // symmetric — skip if the pair is already recorded (from j's pass)
            if adj[i].iter().any(|&(x, _)| x == j) {
                continue;
            }
            if motion_valid(&nodes[i], &nodes[j]) {
                adj[i].push((j, d));
                adj[j].push((i, d));
            }
        }
    }
    adj
}

impl Roadmap {
    /// Number of milestones in the roadmap.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Answer a query: connect `start`/`goal` to their `k` nearest collision-free
    /// milestones (plus a direct `start`→`goal` shortcut when it is collision-free)
    /// and return the shortest resulting waypoint path via Dijkstra over edge
    /// lengths, or `None` if the roadmap does not connect them. The returned path's
    /// first/last configs equal `start`/`goal` exactly.
    pub fn query(
        &self,
        start: &[f64],
        goal: &[f64],
        k: usize,
        motion_valid: impl Fn(&[f64], &[f64]) -> bool,
    ) -> Option<Vec<Vec<f64>>> {
        let n = self.nodes.len();
        let start_idx = n;
        let goal_idx = n + 1;

        // Augment the roadmap with the two query nodes.
        let mut adj = self.adj.clone();
        adj.push(Vec::new()); // start
        adj.push(Vec::new()); // goal

        // Connect each query endpoint to its k nearest collision-free milestones.
        for (qi, q) in [(start_idx, start), (goal_idx, goal)] {
            let mut cand: Vec<(usize, f64)> =
                (0..n).map(|j| (j, dist(q, &self.nodes[j]))).collect();
            cand.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            let mut added = 0;
            for &(j, d) in cand.iter() {
                if added >= k {
                    break;
                }
                if motion_valid(q, &self.nodes[j]) {
                    adj[qi].push((j, d));
                    adj[j].push((qi, d));
                    added += 1;
                }
            }
        }

        // Trivial-connect fallback: a direct collision-free start→goal edge.
        if motion_valid(start, goal) {
            let d = dist(start, goal);
            adj[start_idx].push((goal_idx, d));
            adj[goal_idx].push((start_idx, d));
        }

        let prev = dijkstra(&adj, start_idx, goal_idx)?;

        // Reconstruct goal→start, then reverse; map indices back to configs.
        let cfg = |i: usize| -> Vec<f64> {
            if i == start_idx {
                start.to_vec()
            } else if i == goal_idx {
                goal.to_vec()
            } else {
                self.nodes[i].clone()
            }
        };
        let mut chain = vec![goal_idx];
        let mut cur = goal_idx;
        while cur != start_idx {
            cur = prev[cur];
            chain.push(cur);
        }
        chain.reverse();
        Some(chain.into_iter().map(cfg).collect())
    }
}

/// Dense (O(V²)) Dijkstra over an undirected weighted adjacency. Returns the
/// predecessor array on success (goal reachable), else `None`. Ties are broken
/// deterministically toward the lowest node index (strict `<` frontier + strict
/// `<` relaxation), so the recovered path is reproducible. Edge lengths are
/// non-negative joint-space distances, which Dijkstra requires.
fn dijkstra(adj: &[Vec<(usize, f64)>], src: usize, dst: usize) -> Option<Vec<usize>> {
    let v = adj.len();
    let mut d = vec![f64::INFINITY; v];
    let mut prev = vec![usize::MAX; v];
    let mut done = vec![false; v];
    d[src] = 0.0;
    loop {
        // Lowest-cost unvisited node (lowest index on ties).
        let mut u = usize::MAX;
        let mut best = f64::INFINITY;
        for i in 0..v {
            if !done[i] && d[i] < best {
                best = d[i];
                u = i;
            }
        }
        if u == usize::MAX || u == dst {
            break;
        }
        done[u] = true;
        for &(w, cost) in &adj[u] {
            let nd = d[u] + cost;
            if nd < d[w] {
                d[w] = nd;
                prev[w] = u;
            }
        }
    }
    if d[dst].is_finite() { Some(prev) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dijkstra_shortest_on_hand_graph() {
        // 0 --1-- 1 --1-- 3   and   0 --1-- 2 --5-- 3
        // shortest 0→3 is 0-1-3 (len 2), not 0-2-3 (len 6).
        let adj = vec![
            vec![(1usize, 1.0), (2, 1.0)],
            vec![(0, 1.0), (3, 1.0)],
            vec![(0, 1.0), (3, 5.0)],
            vec![(1, 1.0), (2, 5.0)],
        ];
        let prev = dijkstra(&adj, 0, 3).unwrap();
        // reconstruct
        let mut chain = vec![3usize];
        let mut cur = 3;
        while cur != 0 {
            cur = prev[cur];
            chain.push(cur);
        }
        chain.reverse();
        assert_eq!(chain, vec![0, 1, 3]);
    }

    #[test]
    fn dijkstra_disconnected_returns_none() {
        let adj = vec![vec![(1usize, 1.0)], vec![(0, 1.0)], vec![]];
        assert!(dijkstra(&adj, 0, 2).is_none());
    }

    /// A fully-free 2-D box: enough milestones + a k-NN roadmap must connect two
    /// corners, and the recovered path is a real polyline start→goal.
    #[test]
    fn free_space_roadmap_connects() {
        let bounds = [(-1.0, 1.0), (-1.0, 1.0)];
        let params = PrmParams {
            seed: 0xCA11,
            samples: 200,
            k: 8,
        };
        let rm = build_roadmap(&bounds, &params, |_| true, |_, _| true);
        assert_eq!(rm.len(), 200);
        let start = [-0.9, -0.9];
        let goal = [0.9, 0.9];
        let path = rm.query(&start, &goal, 8, |_, _| true).unwrap();
        assert_eq!(path.first().unwrap().as_slice(), &start);
        assert_eq!(path.last().unwrap().as_slice(), &goal);
        // every hop is finite and the polyline is at least the straight line
        let len = crate::smooth::path_length(&path);
        assert!(len >= dist(&start, &goal) - 1e-9);
    }

    /// Same seed ⇒ identical roadmap ⇒ identical query path.
    #[test]
    fn deterministic_same_seed() {
        let bounds = [(-1.0, 1.0), (-1.0, 1.0)];
        let mk = || PrmParams {
            seed: 42,
            samples: 120,
            k: 6,
        };
        let start = [-0.5, -0.5];
        let goal = [0.7, 0.6];
        let a = build_roadmap(&bounds, &mk(), |_| true, |_, _| true)
            .query(&start, &goal, 6, |_, _| true)
            .unwrap();
        let b = build_roadmap(&bounds, &mk(), |_| true, |_, _| true)
            .query(&start, &goal, 6, |_, _| true)
            .unwrap();
        assert_eq!(a, b);
    }
}
