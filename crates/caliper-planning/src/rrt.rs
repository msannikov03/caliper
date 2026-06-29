//! RRT tree + joint-space geometry helpers (Euclidean metric).

/// A unidirectional search tree of joint configurations.
pub(crate) struct Tree {
    nodes: Vec<Vec<f64>>,
    parent: Vec<usize>,
}

impl Tree {
    pub fn new(root: Vec<f64>) -> Self {
        Self {
            nodes: vec![root],
            parent: vec![usize::MAX], // root has no parent
        }
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn node(&self, i: usize) -> &[f64] {
        &self.nodes[i]
    }
    pub fn root(&self) -> &[f64] {
        &self.nodes[0]
    }
    /// Add `q` as a child of `parent`; returns its index.
    pub fn add(&mut self, q: Vec<f64>, parent: usize) -> usize {
        self.nodes.push(q);
        self.parent.push(parent);
        self.nodes.len() - 1
    }
    /// Index of the nearest node to `q` (linear scan, squared Euclidean).
    pub fn nearest(&self, q: &[f64]) -> usize {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, n) in self.nodes.iter().enumerate() {
            let d = dist_sq(n, q);
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }
    /// Root→`idx` configuration path (inclusive).
    pub fn path_to(&self, idx: usize) -> Vec<Vec<f64>> {
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

#[inline]
pub(crate) fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}
#[inline]
pub(crate) fn dist(a: &[f64], b: &[f64]) -> f64 {
    dist_sq(a, b).sqrt()
}
#[inline]
pub(crate) fn lerp(a: &[f64], b: &[f64], t: f64) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + (y - x) * t).collect()
}

/// Step from `from` toward `to`, capped at length `step`. Returns `to` itself if
/// it is already within `step` (so an extend can "Reach" the target).
pub(crate) fn steer(from: &[f64], to: &[f64], step: f64) -> Vec<f64> {
    let d = dist(from, to);
    if d <= step || d == 0.0 {
        return to.to_vec();
    }
    let s = step / d;
    from.iter().zip(to).map(|(x, y)| x + (y - x) * s).collect()
}
