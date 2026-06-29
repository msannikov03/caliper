//! Shortcut smoothing: repeatedly try to replace a sub-path between two points
//! with a single straight (collision-checked) edge. Deterministic (seeded RNG),
//! monotone non-increasing in path length, and every kept edge is re-validated —
//! so the output is still collision-free.

use crate::rng::Rng;
use crate::rrt::dist;

/// Total joint-space length of a waypoint path.
pub fn path_length(path: &[Vec<f64>]) -> f64 {
    path.windows(2).map(|w| dist(&w[0], &w[1])).sum()
}

/// `shortcut_iters` rounds of random shortcutting. `valid_edge(a,b)` must return
/// `true` iff the straight segment a→b is collision-free (dense check). Never
/// increases length; preserves the endpoints.
pub fn shortcut_smooth(
    mut path: Vec<Vec<f64>>,
    iters: usize,
    seed: u64,
    valid_edge: impl Fn(&[f64], &[f64]) -> bool,
) -> Vec<Vec<f64>> {
    if path.len() <= 2 {
        return path;
    }
    let mut rng = Rng::new(seed);
    for _ in 0..iters {
        if path.len() <= 2 {
            break;
        }
        // pick i < j with at least one node strictly between them
        let n = path.len();
        let i = (rng.unit() * (n - 2) as f64) as usize; // 0..=n-3
        let span = n - 2 - i; // remaining room for j
        let j = i + 2 + (rng.unit() * (span as f64 + 1.0)) as usize; // i+2 ..= n-1
        let j = j.min(n - 1);
        if j <= i + 1 {
            continue;
        }
        if valid_edge(&path[i], &path[j]) {
            // drop the intermediate nodes (i, j] exclusive of j
            path.drain(i + 1..j);
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_and_endpoints() {
        // a zig-zag in free space collapses to the straight line, endpoints kept
        let path = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
        ];
        let before = path_length(&path);
        let out = shortcut_smooth(path.clone(), 200, 0xCA11, |_, _| true); // everything valid
        assert_eq!(out.first().unwrap(), &path[0]);
        assert_eq!(out.last().unwrap(), path.last().unwrap());
        assert!(path_length(&out) <= before + 1e-12, "must not grow");
        assert!(out.len() == 2, "free space collapses to a straight line");
    }

    #[test]
    fn deterministic() {
        let path = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 0.0],
        ];
        let a = shortcut_smooth(path.clone(), 50, 7, |_, _| true);
        let b = shortcut_smooth(path.clone(), 50, 7, |_, _| true);
        assert_eq!(a, b);
    }

    #[test]
    fn respects_invalid_edges() {
        // no shortcut is valid → path unchanged
        let path = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let out = shortcut_smooth(path.clone(), 100, 1, |_, _| false);
        assert_eq!(out, path);
    }
}
