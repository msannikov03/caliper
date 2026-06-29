//! Collision-aware reachability: can a frame reach a Cartesian pose with a
//! collision-free configuration? IK is collision-UNAWARE, so we try several
//! deterministic seeds — a pose counts as `Reachable` only if some IK solution is
//! both within tolerance AND collision-free; `Blocked` if every reachable solution
//! collides; `Unreachable` if IK never converges.

use crate::rng::Rng;
use caliper_collision::{CollisionModel, WorldScene};
use caliper_ik::{IkOpts, ik};
use caliper_model::Model;
use caliper_spatial::Se3;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReachStatus {
    /// IK converged AND the solution is collision-free.
    Reachable,
    /// IK converged but every solution found collides (self/world).
    Blocked,
    /// IK did not converge from any seed (out of the frame's workspace).
    Unreachable,
}

#[derive(Clone, Debug)]
pub struct ReachVerdict {
    pub status: ReachStatus,
    /// Best (smallest) IK residual seen across seeds.
    pub residual: f64,
    /// A collision-free IK solution when `status == Reachable`.
    pub q: Option<Vec<f64>>,
}

#[derive(Clone, Debug)]
pub struct ReachConfig {
    /// Frame to reach with; `None` = the model's tip frame.
    pub frame: Option<usize>,
    /// IK seeds to try (seed 0 = home/zeros; the rest are random-but-deterministic).
    pub seeds: usize,
    pub seed: u64,
    pub margin: f64,
    pub unbounded_range: f64,
}
impl Default for ReachConfig {
    fn default() -> Self {
        Self {
            frame: None,
            seeds: 8,
            seed: 0xCA11,
            margin: 0.0,
            unbounded_range: std::f64::consts::PI,
        }
    }
}

/// Collision-aware reachability checker over a robot + world scene.
pub struct ReachChecker {
    model: Arc<Model>,
    collision: CollisionModel,
    frame: usize,
    bounds: Vec<(f64, f64)>,
    cfg: ReachConfig,
}

impl ReachChecker {
    pub fn new(model: Arc<Model>, scene: WorldScene, cfg: ReachConfig) -> Self {
        let frame = cfg.frame.unwrap_or_else(|| model.tip_frame());
        let bounds = (0..model.ndof)
            .map(|i| model.limits[i].unwrap_or((-cfg.unbounded_range, cfg.unbounded_range)))
            .collect();
        let collision = CollisionModel::new(model.clone(), scene, cfg.margin);
        Self {
            model,
            collision,
            frame,
            bounds,
            cfg,
        }
    }

    fn collision_free(&self, q: &[f64]) -> bool {
        self.collision
            .query(q)
            .map(|r| !r.has_collision())
            .unwrap_or(false)
    }

    /// Reachability verdict for a Cartesian `target` pose of the configured frame.
    pub fn status(&self, target: &Se3) -> ReachVerdict {
        let n = self.model.ndof;
        let mut rng = Rng::new(self.cfg.seed);
        let mut best_res = f64::INFINITY;
        let mut any_converged = false;
        for k in 0..self.cfg.seeds.max(1) {
            // seed 0 = zeros (home); the rest deterministic-random within limits
            let seed_q: Vec<f64> = if k == 0 {
                vec![0.0; n]
            } else {
                self.bounds
                    .iter()
                    .map(|&(lo, hi)| rng.range(lo, hi))
                    .collect()
            };
            let res = ik(&self.model, self.frame, target, &seed_q, &IkOpts::default());
            if res.success {
                any_converged = true;
                best_res = best_res.min(res.residual);
                if self.collision_free(&res.q) {
                    return ReachVerdict {
                        status: ReachStatus::Reachable,
                        residual: res.residual,
                        q: Some(res.q),
                    };
                }
            } else {
                best_res = best_res.min(res.residual);
            }
        }
        ReachVerdict {
            status: if any_converged {
                ReachStatus::Blocked
            } else {
                ReachStatus::Unreachable
            },
            residual: best_res,
            q: None,
        }
    }

    /// Convenience boolean.
    pub fn reachable(&self, target: &Se3) -> bool {
        self.status(target).status == ReachStatus::Reachable
    }

    /// Fraction of `targets` that are collision-free reachable → (reachable, total).
    pub fn sweep(&self, targets: &[Se3]) -> (usize, usize) {
        let r = targets.iter().filter(|t| self.reachable(t)).count();
        (r, targets.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliper_kinematics::fk_frame;
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

    // collide_arm has real <collision> geometry, so a world box can actually block
    // it (showcase6 is visual-only → 0 colliders → nothing to block).
    #[test]
    fn reachable_in_free_space() {
        let m = model("collide_arm.urdf");
        let q = vec![0.3, -0.3, 0.3]; // near-extended, self-collision-free
        let tip = m.tip_frame();
        let target = fk_frame(&m, &q, tip);
        let rc = ReachChecker::new(m, WorldScene::new(), ReachConfig::default());
        let v = rc.status(&target);
        assert_eq!(
            v.status,
            ReachStatus::Reachable,
            "residual {:.2e}",
            v.residual
        );
    }

    #[test]
    fn unreachable_far_pose() {
        // 10 m away — far outside the arm's workspace → Unreachable
        let m = model("collide_arm.urdf");
        let target = Se3::from_parts(
            nalgebra::Vector3::new(10.0, 0.0, 0.0),
            nalgebra::UnitQuaternion::identity(),
        );
        let rc = ReachChecker::new(m, WorldScene::new(), ReachConfig::default());
        assert_eq!(rc.status(&target).status, ReachStatus::Unreachable);
    }

    #[test]
    fn blocked_by_world_box() {
        // a kinematically-reachable pose, but a big box enveloping that region makes
        // every IK solution collide → Blocked.
        let m = model("collide_arm.urdf");
        let q = vec![0.3, -0.3, 0.3];
        let tip = m.tip_frame();
        let target = fk_frame(&m, &q, tip);
        let c = target.translation();
        let scene = WorldScene::new().add_box([c[0], c[1], c[2]], [1.0, 1.0, 1.0]);
        let rc = ReachChecker::new(m, scene, ReachConfig::default());
        let v = rc.status(&target);
        assert_eq!(
            v.status,
            ReachStatus::Blocked,
            "residual {:.2e}",
            v.residual
        );
    }
}
