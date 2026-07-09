//! Per-episode feature statistics and lerobot-compatible aggregation.
//!
//! Semantics mirror `lerobot.datasets.compute_stats`: per-episode stats are
//! population (`ddof=0`) min/max/mean/std per element plus a frame `count`;
//! `meta/stats.json` aggregates episodes with a count-weighted mean and the
//! exact parallel-variance combination `aggregate_feature_stats` uses.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// min/max/mean/std per element of one feature, plus the sample count.
/// Serializes to the `{"min": [...], ..., "count": [n]}` shape lerobot writes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureStats {
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub count: Vec<u64>,
}

impl FeatureStats {
    /// Population stats over `rows` (each row is one frame, `width` elements).
    /// Empty input is rejected upstream (`save_episode` requires frames).
    pub fn compute(rows: &[Vec<f64>], width: usize) -> Self {
        let n = rows.len();
        let denom = n.max(1) as f64;
        let mut min = vec![f64::INFINITY; width];
        let mut max = vec![f64::NEG_INFINITY; width];
        let mut mean = vec![0.0; width];
        for row in rows {
            for (j, &v) in row.iter().enumerate() {
                if v < min[j] {
                    min[j] = v;
                }
                if v > max[j] {
                    max[j] = v;
                }
                mean[j] += v / denom;
            }
        }
        let mut var = vec![0.0; width];
        for row in rows {
            for (j, &v) in row.iter().enumerate() {
                let d = v - mean[j];
                var[j] += d * d / denom;
            }
        }
        if n == 0 {
            min = vec![0.0; width];
            max = vec![0.0; width];
        }
        Self {
            min,
            max,
            mean,
            std: var.iter().map(|v| v.sqrt()).collect(),
            count: vec![n as u64],
        }
    }

    fn total_count(&self) -> u64 {
        self.count.first().copied().unwrap_or(0)
    }
}

/// Aggregate per-episode stats into dataset-level stats, feature by feature,
/// with lerobot's weighted-mean + parallel-variance algorithm:
///
/// ```text
/// mean = Σ count_e · mean_e / Σ count_e
/// var  = Σ count_e · (var_e + (mean_e - mean)²) / Σ count_e
/// ```
pub fn aggregate_stats(
    episodes: &[BTreeMap<String, FeatureStats>],
) -> BTreeMap<String, FeatureStats> {
    let mut keys: Vec<&String> = episodes.iter().flat_map(|e| e.keys()).collect();
    keys.sort();
    keys.dedup();

    let mut out = BTreeMap::new();
    for key in keys {
        let stats: Vec<&FeatureStats> = episodes.iter().filter_map(|e| e.get(key)).collect();
        out.insert(key.clone(), aggregate_feature(&stats));
    }
    out
}

fn aggregate_feature(stats: &[&FeatureStats]) -> FeatureStats {
    let width = stats.first().map_or(0, |s| s.mean.len());
    let total: u64 = stats.iter().map(|s| s.total_count()).sum();
    let denom = (total.max(1)) as f64;

    let mut min = vec![f64::INFINITY; width];
    let mut max = vec![f64::NEG_INFINITY; width];
    let mut mean = vec![0.0; width];
    for s in stats {
        let w = s.total_count() as f64;
        for j in 0..width {
            if s.min[j] < min[j] {
                min[j] = s.min[j];
            }
            if s.max[j] > max[j] {
                max[j] = s.max[j];
            }
            mean[j] += s.mean[j] * w / denom;
        }
    }
    let mut var = vec![0.0; width];
    for s in stats {
        let w = s.total_count() as f64;
        for j in 0..width {
            let d = s.mean[j] - mean[j];
            var[j] += (s.std[j] * s.std[j] + d * d) * w / denom;
        }
    }
    FeatureStats {
        min,
        max,
        mean,
        std: var.iter().map(|v| v.sqrt()).collect(),
        count: vec![total],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(vals: &[f64]) -> Vec<Vec<f64>> {
        vals.iter().map(|&v| vec![v]).collect()
    }

    #[test]
    fn per_episode_stats_are_population() {
        let s = FeatureStats::compute(&rows(&[1.0, 2.0, 3.0, 4.0]), 1);
        assert_eq!(s.min, vec![1.0]);
        assert_eq!(s.max, vec![4.0]);
        assert_eq!(s.mean, vec![2.5]);
        // population std of {1,2,3,4} = sqrt(1.25)
        assert!((s.std[0] - 1.25f64.sqrt()).abs() < 1e-12);
        assert_eq!(s.count, vec![4]);
    }

    #[test]
    fn aggregation_equals_pooled_population_stats() {
        // Aggregating episode stats must equal stats over the pooled samples —
        // the parallel variance combination is exact for population variance.
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 12.0, 14.0, 16.0, 18.0];
        let ep_a = BTreeMap::from([("x".to_string(), FeatureStats::compute(&rows(&a), 1))]);
        let ep_b = BTreeMap::from([("x".to_string(), FeatureStats::compute(&rows(&b), 1))]);
        let agg = aggregate_stats(&[ep_a, ep_b]);

        let pooled: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
        let direct = FeatureStats::compute(&rows(&pooled), 1);
        let got = &agg["x"];
        assert_eq!(got.count, vec![8]);
        assert!((got.mean[0] - direct.mean[0]).abs() < 1e-12);
        assert!((got.std[0] - direct.std[0]).abs() < 1e-12);
        assert_eq!(got.min[0], 1.0);
        assert_eq!(got.max[0], 18.0);
    }

    #[test]
    fn aggregation_unions_feature_keys() {
        let ep_a = BTreeMap::from([("x".to_string(), FeatureStats::compute(&rows(&[1.0]), 1))]);
        let ep_b = BTreeMap::from([("y".to_string(), FeatureStats::compute(&rows(&[2.0]), 1))]);
        let agg = aggregate_stats(&[ep_a, ep_b]);
        assert_eq!(agg.len(), 2);
        assert_eq!(agg["x"].count, vec![1]);
        assert_eq!(agg["y"].count, vec![1]);
    }
}
