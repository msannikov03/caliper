//! Collision-aware motion planning for Caliper.
//!
//! [`Planner`] runs **RRT-Connect** (bidirectional, joint-space) with
//! [`caliper_collision::CollisionModel`] as the validity check (self + world) plus
//! joint limits, then **shortcut-smooths** the result. Everything is deterministic
//! (a seeded splitmix64 PRNG — no `rand`), so a given seed yields the same plan and
//! the whole thing is unit-testable with no hardware. A planned path is a
//! collision-free joint-space WAYPOINT path; turn it into a playable, recordable
//! `caliper_motion::Trajectory` with [`Planner::plan_trajectory`] (see `retime`).
//!
//! Reachability analysis lives in [`reach`]. Planning is a pure-CPU, dependency-
//! light crate (the cuRobo GPU sidecar is deferred).

mod prm;
mod rng;
mod rrt;
mod rrtstar;
mod smooth;

pub mod reach;

pub use smooth::path_length;

use caliper_collision::{CollisionModel, WorldScene};
use caliper_ik::{IkOpts, ik};
use caliper_model::Model;
use caliper_motion::{MotionLimits, Trajectory, retime_waypoints};
use caliper_spatial::Se3;
use prm::{PrmParams, build_roadmap};
use rng::Rng;
use rrt::{Tree, dist, lerp, steer};
use rrtstar::{StarParams, rrt_star};
use std::sync::Arc;

/// Tunable planner parameters. Defaults are sensible for a ~6-DOF arm.
#[derive(Clone, Debug)]
pub struct PlannerConfig {
    /// PRNG seed — same seed ⇒ identical plan.
    pub seed: u64,
    /// Max RRT iterations before giving up.
    pub max_iters: usize,
    /// Max joint-space extend step (rad).
    pub step: f64,
    /// Probability of biasing a sample toward the opposite tree's root.
    pub goal_bias: f64,
    /// Collision-check spacing along an edge (rad).
    pub edge_resolution: f64,
    /// Shortcut-smoothing rounds.
    pub shortcut_iters: usize,
    /// Collision inflation margin (m).
    pub margin: f64,
    /// Sampling half-range for joints with no URDF limit (rad).
    pub unbounded_range: f64,
}
impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            seed: 0xCA11,
            max_iters: 10_000,
            step: 0.3,
            goal_bias: 0.05,
            edge_resolution: 0.05,
            shortcut_iters: 200,
            margin: 0.0,
            unbounded_range: std::f64::consts::PI,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum PlanError {
    #[error("expected {expected} joints, got {got}")]
    Dim { expected: usize, got: usize },
    #[error("non-finite value in {0}")]
    NonFinite(&'static str),
    #[error("invalid planner config: {0}")]
    InvalidConfig(&'static str),
    #[error("{which} joint {joint} = {value} is outside bounds [{lo}, {hi}]")]
    OutOfBounds {
        which: &'static str,
        joint: usize,
        value: f64,
        lo: f64,
        hi: f64,
    },
    #[error("start configuration is in collision: {0}")]
    StartInCollision(String),
    #[error("goal configuration is in collision: {0}")]
    GoalInCollision(String),
    #[error("collision query failed: {0}")]
    Collision(String),
    #[error("no collision-free path found within {0} iterations")]
    Unreachable(usize),
    #[error("goal pose unreachable by IK (residual {0:.3e})")]
    IkUnreachable(f64),
    #[error("retiming failed: {0}")]
    Retime(String),
}

enum Extend {
    Reached,
    Advanced,
    Trapped,
}

/// A collision-aware joint-space planner over a robot model + a static world scene.
pub struct Planner {
    model: Arc<Model>,
    collision: CollisionModel,
    cfg: PlannerConfig,
    bounds: Vec<(f64, f64)>,
}

impl Planner {
    /// Build a planner. `scene` holds the world obstacles (ground/boxes); `cfg`
    /// the algorithm parameters.
    pub fn new(model: Arc<Model>, scene: WorldScene, cfg: PlannerConfig) -> Self {
        let bounds = (0..model.ndof)
            .map(|i| model.limits[i].unwrap_or((-cfg.unbounded_range, cfg.unbounded_range)))
            .collect();
        let collision = CollisionModel::new(model.clone(), scene, cfg.margin);
        Self {
            model,
            collision,
            cfg,
            bounds,
        }
    }

    pub fn ndof(&self) -> usize {
        self.model.ndof
    }
    /// Frames with NO collider (mesh/none) — collisions there are NOT detected.
    pub fn uncovered_frames(&self) -> usize {
        self.collision.uncovered_frames()
    }
    pub fn config(&self) -> &PlannerConfig {
        &self.cfg
    }

    /// Plan a collision-free, smoothed joint-space waypoint path from `start` to
    /// `goal`. The returned path's endpoints equal `start`/`goal` and every edge is
    /// collision-free at the planner resolution (verify independently with
    /// [`verify_path`](Self::verify_path)).
    pub fn plan(&self, start: &[f64], goal: &[f64]) -> Result<Vec<Vec<f64>>, PlanError> {
        self.validate_cfg()?;
        let n = self.model.ndof;
        for (name, q) in [("start", start), ("goal", goal)] {
            if q.len() != n {
                return Err(PlanError::Dim {
                    expected: n,
                    got: q.len(),
                });
            }
            if !q.iter().all(|x| x.is_finite()) {
                return Err(PlanError::NonFinite(name));
            }
        }
        self.check_bounds(start, "start")?;
        self.check_bounds(goal, "goal")?;
        self.check_endpoint(start, true)?;
        self.check_endpoint(goal, false)?;

        let mut rng = Rng::new(self.cfg.seed);
        let mut ta = Tree::new(start.to_vec()); // start-rooted
        let mut tb = Tree::new(goal.to_vec()); // goal-rooted

        for iter in 0..self.cfg.max_iters {
            let even = iter % 2 == 0;
            let bias_root = if even {
                tb.root().to_vec()
            } else {
                ta.root().to_vec()
            };
            let q_rand = self.sample(&mut rng, &bias_root);

            let grew = if even {
                self.extend(&mut ta, &q_rand)
            } else {
                self.extend(&mut tb, &q_rand)
            };
            if !matches!(grew, Extend::Trapped) {
                let (grow_idx, q_new) = if even {
                    (ta.len() - 1, ta.node(ta.len() - 1).to_vec())
                } else {
                    (tb.len() - 1, tb.node(tb.len() - 1).to_vec())
                };
                let connected = if even {
                    self.connect(&mut tb, &q_new)
                } else {
                    self.connect(&mut ta, &q_new)
                };
                if let Some(other_idx) = connected {
                    let (sidx, gidx) = if even {
                        (grow_idx, other_idx)
                    } else {
                        (other_idx, grow_idx)
                    };
                    let raw = assemble(&ta, &tb, sidx, gidx);
                    return Ok(self.smooth(raw));
                }
            }
        }
        Err(PlanError::Unreachable(self.cfg.max_iters))
    }

    /// Plan an **asymptotically-optimal** (RRT*) collision-free, smoothed
    /// joint-space waypoint path from `start` to `goal` within `iters` samples.
    ///
    /// Unlike [`plan`](Self::plan) (RRT-Connect — fast, *feasible*), this grows a
    /// single start-rooted tree that keeps rewiring toward lower path length, so
    /// with enough iterations it converges to (near-)shortest joint-space paths.
    /// Same validity model and resolution as `plan`; same determinism (a given
    /// `seed`/`iters` yields the same path). Endpoints equal `start`/`goal` and
    /// every edge is collision-free at the planner resolution (independently
    /// re-checkable with [`verify_path`](Self::verify_path)). The raw RRT* path is
    /// shortcut-smoothed (monotone non-increasing in length) before returning.
    pub fn plan_optimal(
        &self,
        start: &[f64],
        goal: &[f64],
        iters: usize,
    ) -> Result<Vec<Vec<f64>>, PlanError> {
        self.validate_cfg()?;
        if iters == 0 {
            return Err(PlanError::InvalidConfig("iters must be > 0"));
        }
        let n = self.model.ndof;
        for (name, q) in [("start", start), ("goal", goal)] {
            if q.len() != n {
                return Err(PlanError::Dim {
                    expected: n,
                    got: q.len(),
                });
            }
            if !q.iter().all(|x| x.is_finite()) {
                return Err(PlanError::NonFinite(name));
            }
        }
        self.check_bounds(start, "start")?;
        self.check_bounds(goal, "goal")?;
        self.check_endpoint(start, true)?;
        self.check_endpoint(goal, false)?;

        let params = StarParams {
            seed: self.cfg.seed,
            iters,
            step: self.cfg.step,
            goal_bias: self.cfg.goal_bias,
        };
        let raw = rrt_star(start, goal, &self.bounds, &params, |a, b| {
            self.motion_valid(a, b)
        })
        .ok_or(PlanError::Unreachable(iters))?;
        Ok(self.smooth(raw))
    }

    /// Plan a collision-free, smoothed joint-space waypoint path from `start` to
    /// `goal` with a **PRM** (Probabilistic RoadMap — a multi-query sampling
    /// planner). Rejection-samples `samples` collision-free milestones inside the
    /// joint bounds, wires each to its `k` nearest collision-free neighbours, then
    /// connects `start`/`goal` and returns the shortest roadmap path (Dijkstra over
    /// joint-space edge lengths), shortcut-smoothed.
    ///
    /// Unlike [`plan`](Self::plan) (RRT-Connect) and [`plan_optimal`](Self::plan_optimal)
    /// (RRT*), which grow a fresh tree per query, PRM's strength is a reusable
    /// roadmap; this MVP builds + queries it in one call. Same validity model and
    /// resolution, same determinism (a given `seed`/`samples`/`k` yields the same
    /// path). Endpoints equal `start`/`goal` and every edge is collision-free at the
    /// planner resolution (independently re-checkable with
    /// [`verify_path`](Self::verify_path)). `Unreachable` if the roadmap fails to
    /// connect the endpoints (try more `samples` or a larger `k`).
    pub fn plan_prm(
        &self,
        start: &[f64],
        goal: &[f64],
        samples: usize,
        k: usize,
    ) -> Result<Vec<Vec<f64>>, PlanError> {
        self.validate_cfg()?;
        if samples == 0 {
            return Err(PlanError::InvalidConfig("samples must be > 0"));
        }
        if k == 0 {
            return Err(PlanError::InvalidConfig("k must be > 0"));
        }
        let n = self.model.ndof;
        for (name, q) in [("start", start), ("goal", goal)] {
            if q.len() != n {
                return Err(PlanError::Dim {
                    expected: n,
                    got: q.len(),
                });
            }
            if !q.iter().all(|x| x.is_finite()) {
                return Err(PlanError::NonFinite(name));
            }
        }
        self.check_bounds(start, "start")?;
        self.check_bounds(goal, "goal")?;
        self.check_endpoint(start, true)?;
        self.check_endpoint(goal, false)?;

        let params = PrmParams {
            seed: self.cfg.seed,
            samples,
            k,
        };
        let roadmap = build_roadmap(
            &self.bounds,
            &params,
            |q| self.config_free(q),
            |a, b| self.motion_valid(a, b),
        );
        let raw = roadmap
            .query(start, goal, k, |a, b| self.motion_valid(a, b))
            .ok_or(PlanError::Unreachable(samples))?;
        Ok(self.smooth(raw))
    }

    /// Plan + retime into a playable/recordable [`Trajectory`] (collision-free
    /// waypoints → per-joint-limited path-scalar S-curves).
    pub fn plan_trajectory(
        &self,
        start: &[f64],
        goal: &[f64],
        limits: &MotionLimits,
        dt: f64,
    ) -> Result<Trajectory, PlanError> {
        let path = self.plan(start, goal)?;
        retime_waypoints(&path, limits, dt).map_err(|e| PlanError::Retime(e.to_string()))
    }

    /// Plan to a Cartesian goal pose of `frame`: IK the pose (from `ik_seed`) to a
    /// joint goal, then plan to it. `IkUnreachable` if IK does not converge.
    pub fn plan_to_pose(
        &self,
        start: &[f64],
        target: &Se3,
        frame: usize,
        ik_seed: &[f64],
    ) -> Result<Vec<Vec<f64>>, PlanError> {
        let res = ik(&self.model, frame, target, ik_seed, &IkOpts::default());
        if !res.success {
            return Err(PlanError::IkUnreachable(res.residual));
        }
        self.plan(start, &res.q)
    }

    /// Independently re-verify that every edge of `path` is collision-free, at a
    /// FINER resolution than planning used (a stricter guarantee than the planner's
    /// own check). Endpoints included.
    pub fn verify_path(&self, path: &[Vec<f64>]) -> bool {
        if path.is_empty() {
            return false;
        }
        let res = self.cfg.edge_resolution * 0.5;
        // endpoints
        if !self.config_free(&path[0]) || !self.config_free(path.last().unwrap()) {
            return false;
        }
        for w in path.windows(2) {
            let d = dist(&w[0], &w[1]);
            let steps = ((d / res).ceil() as usize).max(1);
            for i in 0..=steps {
                let t = i as f64 / steps as f64;
                if !self.config_free(&lerp(&w[0], &w[1], t)) {
                    return false;
                }
            }
        }
        true
    }

    // ---- internals ----

    /// Reject a config that would livelock or overflow the planner. `edge_resolution`
    /// of 0 turns `(d/0).ceil() as usize` into `usize::MAX` (an effectively infinite
    /// collision-check loop); a `step` of 0 does the same in `connect`.
    fn validate_cfg(&self) -> Result<(), PlanError> {
        let c = &self.cfg;
        if !(c.step.is_finite() && c.step > 0.0) {
            return Err(PlanError::InvalidConfig("step must be finite and > 0"));
        }
        if !(c.edge_resolution.is_finite() && c.edge_resolution > 0.0) {
            return Err(PlanError::InvalidConfig(
                "edge_resolution must be finite and > 0",
            ));
        }
        if !(c.goal_bias.is_finite() && (0.0..=1.0).contains(&c.goal_bias)) {
            return Err(PlanError::InvalidConfig("goal_bias must be in [0, 1]"));
        }
        if !(c.unbounded_range.is_finite() && c.unbounded_range > 0.0) {
            return Err(PlanError::InvalidConfig(
                "unbounded_range must be finite and > 0",
            ));
        }
        Ok(())
    }

    /// Reject an endpoint outside the configured per-joint bounds (random samples are
    /// already bounded; user-supplied endpoints are not).
    fn check_bounds(&self, q: &[f64], which: &'static str) -> Result<(), PlanError> {
        for (i, (&v, &(lo, hi))) in q.iter().zip(self.bounds.iter()).enumerate() {
            if v < lo || v > hi {
                return Err(PlanError::OutOfBounds {
                    which,
                    joint: i,
                    value: v,
                    lo,
                    hi,
                });
            }
        }
        Ok(())
    }

    fn check_endpoint(&self, q: &[f64], is_start: bool) -> Result<(), PlanError> {
        match self.collision.query(q) {
            Ok(r) if r.has_collision() => {
                let msg = format!("frames {:?}", r.colliding_frames);
                Err(if is_start {
                    PlanError::StartInCollision(msg)
                } else {
                    PlanError::GoalInCollision(msg)
                })
            }
            Ok(_) => Ok(()),
            Err(e) => Err(PlanError::Collision(e.to_string())),
        }
    }

    fn config_free(&self, q: &[f64]) -> bool {
        if !q.iter().all(|x| x.is_finite()) {
            return false;
        }
        self.collision
            .query(q)
            .map(|r| !r.has_collision())
            .unwrap_or(false)
    }

    fn motion_valid(&self, a: &[f64], b: &[f64]) -> bool {
        let d = dist(a, b);
        let steps = ((d / self.cfg.edge_resolution).ceil() as usize).max(1);
        for i in 1..=steps {
            let t = i as f64 / steps as f64;
            if !self.config_free(&lerp(a, b, t)) {
                return false;
            }
        }
        true
    }

    fn sample(&self, rng: &mut Rng, bias_root: &[f64]) -> Vec<f64> {
        // draw the bias coin first (fixed RNG order for determinism)
        let bias = rng.unit() < self.cfg.goal_bias;
        if bias {
            return bias_root.to_vec();
        }
        self.bounds
            .iter()
            .map(|&(lo, hi)| rng.range(lo, hi))
            .collect()
    }

    fn extend(&self, tree: &mut Tree, target: &[f64]) -> Extend {
        let near = tree.nearest(target);
        let from = tree.node(near).to_vec();
        let q_new = steer(&from, target, self.cfg.step);
        if self.motion_valid(&from, &q_new) {
            let reached = dist(&q_new, target) < 1e-9;
            tree.add(q_new, near);
            if reached {
                Extend::Reached
            } else {
                Extend::Advanced
            }
        } else {
            Extend::Trapped
        }
    }

    /// Repeatedly extend `tree` toward the fixed `target` until it reaches or traps.
    /// Returns the index of the reached node on success.
    fn connect(&self, tree: &mut Tree, target: &[f64]) -> Option<usize> {
        let cap =
            (dist(tree.node(tree.nearest(target)), target) / self.cfg.step).ceil() as usize + 4;
        for _ in 0..cap {
            match self.extend(tree, target) {
                Extend::Reached => return Some(tree.len() - 1),
                Extend::Advanced => continue,
                Extend::Trapped => return None,
            }
        }
        None
    }

    fn smooth(&self, raw: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let edge = |a: &[f64], b: &[f64]| self.motion_valid(a, b);
        smooth::shortcut_smooth(raw, self.cfg.shortcut_iters, self.cfg.seed ^ 0x5, edge)
    }
}

/// Stitch the start-tree path (root→junction) with the reversed goal-tree path
/// (junction→goal), dropping the duplicated junction node.
fn assemble(ta: &Tree, tb: &Tree, sidx: usize, gidx: usize) -> Vec<Vec<f64>> {
    let mut path = ta.path_to(sidx); // start → junction
    let mut tail = tb.path_to(gidx); // goal-root → junction
    tail.reverse(); // junction → goal
    path.extend(tail.into_iter().skip(1)); // skip duplicate junction
    path
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn arm_planner(scene: WorldScene) -> Planner {
        Planner::new(model("collide_arm.urdf"), scene, PlannerConfig::default())
    }

    #[test]
    fn connects_start_to_goal_collision_free() {
        let p = arm_planner(WorldScene::new());
        let start = vec![0.0, 0.0, 0.0];
        let goal = vec![0.4, -0.4, 0.4];
        let path = p.plan(&start, &goal).unwrap();
        assert!(path.len() >= 2);
        assert_eq!(&path[0], &start);
        assert_eq!(path.last().unwrap(), &goal);
        assert!(p.verify_path(&path), "planned path must be collision-free");
    }

    #[test]
    fn deterministic_same_seed() {
        let a = arm_planner(WorldScene::new())
            .plan(&[0.0, 0.0, 0.0], &[0.4, -0.4, 0.4])
            .unwrap();
        let b = arm_planner(WorldScene::new())
            .plan(&[0.0, 0.0, 0.0], &[0.4, -0.4, 0.4])
            .unwrap();
        assert_eq!(a, b, "same seed ⇒ identical path");
    }

    #[test]
    fn plans_with_world_scene_collision_free() {
        // A ground half-space + a box obstacle are present. The endpoints are clear
        // (asserted against the same CollisionModel the planner uses); the returned
        // path must be collision-free under that world (verified at finer resolution).
        let scene = WorldScene::new()
            .with_ground(-0.1)
            .add_box([0.6, 0.0, 0.3], [0.15, 0.15, 0.15]);
        let start = vec![0.0, 0.0, 0.0];
        let goal = vec![0.4, -0.4, 0.4];
        // self-check the scenario is well-posed: both endpoints collision-free
        let cm = CollisionModel::new(model("collide_arm.urdf"), scene.clone(), 0.0);
        assert!(
            !cm.query(&start).unwrap().has_collision(),
            "start must be clear"
        );
        assert!(
            !cm.query(&goal).unwrap().has_collision(),
            "goal must be clear"
        );
        let p = arm_planner(scene);
        let path = p.plan(&start, &goal).unwrap();
        assert!(
            p.verify_path(&path),
            "path must be collision-free under the world"
        );
    }

    #[test]
    fn start_in_collision_errors() {
        // folded pose self-collides (l1↔l3)
        let p = arm_planner(WorldScene::new());
        let folded = vec![0.0, std::f64::consts::PI, std::f64::consts::PI];
        let err = p.plan(&folded, &[0.0, 0.0, 0.0]).unwrap_err();
        assert!(matches!(err, PlanError::StartInCollision(_)), "got {err:?}");
    }

    #[test]
    fn goal_in_collision_errors() {
        let p = arm_planner(WorldScene::new());
        let folded = vec![0.0, std::f64::consts::PI, std::f64::consts::PI];
        let err = p.plan(&[0.0, 0.0, 0.0], &folded).unwrap_err();
        assert!(matches!(err, PlanError::GoalInCollision(_)), "got {err:?}");
    }

    #[test]
    fn plan_trajectory_endpoints_and_duration() {
        let p = arm_planner(WorldScene::new());
        let lim = MotionLimits {
            vmax: vec![1.0; 3],
            amax: vec![3.0; 3],
            jmax: vec![20.0; 3],
        };
        let start = vec![0.0, 0.0, 0.0];
        let goal = vec![0.4, -0.4, 0.4];
        let traj = p.plan_trajectory(&start, &goal, &lim, 1e-3).unwrap();
        assert!(traj.duration() > 0.0);
        let s0 = traj.q_at(0.0);
        let s1 = traj.q_at(traj.duration());
        for i in 0..3 {
            assert!((s0[i] - start[i]).abs() < 1e-6, "start joint {i}");
            assert!((s1[i] - goal[i]).abs() < 1e-3, "goal joint {i}");
        }
    }

    #[test]
    fn invalid_config_errors() {
        let start = vec![0.0, 0.0, 0.0];
        let goal = vec![0.4, -0.4, 0.4];
        let bad = |mutate: fn(&mut PlannerConfig)| {
            let mut cfg = PlannerConfig::default();
            mutate(&mut cfg);
            Planner::new(model("collide_arm.urdf"), WorldScene::new(), cfg).plan(&start, &goal)
        };
        // each of these would livelock/overflow without validation
        for res in [
            bad(|c| c.edge_resolution = 0.0),
            bad(|c| c.edge_resolution = -0.05),
            bad(|c| c.edge_resolution = f64::NAN),
            bad(|c| c.step = 0.0),
            bad(|c| c.step = -0.3),
            bad(|c| c.goal_bias = -0.1),
            bad(|c| c.goal_bias = 1.5),
            bad(|c| c.unbounded_range = 0.0),
            bad(|c| c.unbounded_range = f64::INFINITY),
        ] {
            assert!(
                matches!(res, Err(PlanError::InvalidConfig(_))),
                "expected InvalidConfig, got {res:?}"
            );
        }
    }

    #[test]
    fn endpoint_out_of_bounds_errors() {
        let p = arm_planner(WorldScene::new());
        // collide_arm joints are revolute with finite limits; push well past them.
        let far = vec![100.0, 0.0, 0.0];
        let start = vec![0.0, 0.0, 0.0];
        assert!(
            matches!(
                p.plan(&far, &start),
                Err(PlanError::OutOfBounds { which: "start", .. })
            ),
            "out-of-bounds start must error"
        );
        assert!(
            matches!(
                p.plan(&start, &far),
                Err(PlanError::OutOfBounds { which: "goal", .. })
            ),
            "out-of-bounds goal must error"
        );
    }

    // ---- RRT* (plan_optimal) cross-validation ----

    #[test]
    fn optimal_connects_collision_free_endpoints() {
        let p = arm_planner(WorldScene::new());
        let start = vec![0.0, 0.0, 0.0];
        let goal = vec![0.4, -0.4, 0.4];
        let path = p.plan_optimal(&start, &goal, 1500).unwrap();
        assert!(path.len() >= 2);
        assert_eq!(&path[0], &start, "endpoint must equal start exactly");
        assert_eq!(
            path.last().unwrap(),
            &goal,
            "endpoint must equal goal exactly"
        );
        assert!(
            p.verify_path(&path),
            "RRT* path must be collision-free at fine resolution"
        );
        // Any valid path is at least the straight-line joint distance.
        assert!(path_length(&path) >= dist(&start, &goal) - 1e-9);
    }

    #[test]
    fn optimal_deterministic_same_seed() {
        let a = arm_planner(WorldScene::new())
            .plan_optimal(&[0.0, 0.0, 0.0], &[0.4, -0.4, 0.4], 1200)
            .unwrap();
        let b = arm_planner(WorldScene::new())
            .plan_optimal(&[0.0, 0.0, 0.0], &[0.4, -0.4, 0.4], 1200)
            .unwrap();
        assert_eq!(a, b, "same seed ⇒ identical RRT* path");
    }

    #[test]
    fn optimal_not_worse_than_rrt_connect_free_space() {
        // Both planners share the validity model + smoother; with enough iterations
        // RRT* must not return a longer path than RRT-Connect. (A wrong RRT* that
        // ignores cost / rewiring would routinely lose this.)
        let p = arm_planner(WorldScene::new());
        let start = vec![0.0, 0.0, 0.0];
        let goal = vec![0.4, -0.4, 0.4];
        let connect = p.plan(&start, &goal).unwrap();
        let star = p.plan_optimal(&start, &goal, 3000).unwrap();
        assert!(p.verify_path(&star));
        assert!(
            path_length(&star) <= path_length(&connect) + 1e-6,
            "RRT* {:.6} should be <= RRT-Connect {:.6}",
            path_length(&star),
            path_length(&connect),
        );
    }

    #[test]
    fn optimal_with_world_scene_not_worse_than_rrt_connect() {
        let scene = WorldScene::new()
            .with_ground(-0.1)
            .add_box([0.6, 0.0, 0.3], [0.15, 0.15, 0.15]);
        let start = vec![0.0, 0.0, 0.0];
        let goal = vec![0.4, -0.4, 0.4];
        let cm = CollisionModel::new(model("collide_arm.urdf"), scene.clone(), 0.0);
        assert!(
            !cm.query(&start).unwrap().has_collision(),
            "start must be clear"
        );
        assert!(
            !cm.query(&goal).unwrap().has_collision(),
            "goal must be clear"
        );
        let p = arm_planner(scene);
        let connect = p.plan(&start, &goal).unwrap();
        let star = p.plan_optimal(&start, &goal, 5000).unwrap();
        assert!(
            p.verify_path(&star),
            "RRT* path must be collision-free under the world"
        );
        assert_eq!(&star[0], &start);
        assert_eq!(star.last().unwrap(), &goal);
        assert!(
            path_length(&star) <= path_length(&connect) + 1e-6,
            "RRT* {:.6} should be <= RRT-Connect {:.6}",
            path_length(&star),
            path_length(&connect),
        );
    }

    #[test]
    fn optimal_invalid_iters_errors() {
        let p = arm_planner(WorldScene::new());
        assert!(matches!(
            p.plan_optimal(&[0.0, 0.0, 0.0], &[0.4, -0.4, 0.4], 0),
            Err(PlanError::InvalidConfig(_))
        ));
    }

    #[test]
    fn optimal_guards_dim_bounds_collision() {
        let p = arm_planner(WorldScene::new());
        assert!(matches!(
            p.plan_optimal(&[0.0, 0.0], &[0.0, 0.0, 0.0], 100),
            Err(PlanError::Dim { .. })
        ));
        assert!(matches!(
            p.plan_optimal(&[100.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 100),
            Err(PlanError::OutOfBounds { which: "start", .. })
        ));
        let folded = vec![0.0, std::f64::consts::PI, std::f64::consts::PI];
        assert!(matches!(
            p.plan_optimal(&[0.0, 0.0, 0.0], &folded, 100),
            Err(PlanError::GoalInCollision(_))
        ));
    }

    #[test]
    fn dim_and_finite_guards() {
        let p = arm_planner(WorldScene::new());
        assert!(matches!(
            p.plan(&[0.0, 0.0], &[0.0, 0.0, 0.0]),
            Err(PlanError::Dim { .. })
        ));
        assert!(matches!(
            p.plan(&[0.0, f64::NAN, 0.0], &[0.0, 0.0, 0.0]),
            Err(PlanError::NonFinite(_))
        ));
    }

    // ---- PRM (plan_prm) cross-validation ----

    #[test]
    fn prm_connects_collision_free_endpoints() {
        let p = arm_planner(WorldScene::new());
        let start = vec![0.0, 0.0, 0.0];
        let goal = vec![0.4, -0.4, 0.4];
        let path = p.plan_prm(&start, &goal, 250, 8).unwrap();
        assert!(path.len() >= 2);
        assert_eq!(&path[0], &start, "endpoint must equal start exactly");
        assert_eq!(
            path.last().unwrap(),
            &goal,
            "endpoint must equal goal exactly"
        );
        // Cross-validate against the engine's own dense collision check.
        assert!(
            p.verify_path(&path),
            "PRM path must be collision-free at fine resolution"
        );
        // Any valid path is at least the straight-line joint distance.
        assert!(path_length(&path) >= dist(&start, &goal) - 1e-9);
    }

    #[test]
    fn prm_deterministic_same_seed() {
        let a = arm_planner(WorldScene::new())
            .plan_prm(&[0.0, 0.0, 0.0], &[0.4, -0.4, 0.4], 200, 6)
            .unwrap();
        let b = arm_planner(WorldScene::new())
            .plan_prm(&[0.0, 0.0, 0.0], &[0.4, -0.4, 0.4], 200, 6)
            .unwrap();
        assert_eq!(a, b, "same seed ⇒ identical PRM path");
    }

    #[test]
    fn prm_with_world_scene_collision_free() {
        // Ground half-space + a box obstacle. Endpoints self-checked clear against
        // the same CollisionModel the planner uses; the returned roadmap path must
        // be collision-free under that world (verified at finer resolution).
        let scene = WorldScene::new()
            .with_ground(-0.1)
            .add_box([0.6, 0.0, 0.3], [0.15, 0.15, 0.15]);
        let start = vec![0.0, 0.0, 0.0];
        let goal = vec![0.4, -0.4, 0.4];
        let cm = CollisionModel::new(model("collide_arm.urdf"), scene.clone(), 0.0);
        assert!(
            !cm.query(&start).unwrap().has_collision(),
            "start must be clear"
        );
        assert!(
            !cm.query(&goal).unwrap().has_collision(),
            "goal must be clear"
        );
        let p = arm_planner(scene);
        let path = p.plan_prm(&start, &goal, 300, 10).unwrap();
        assert_eq!(&path[0], &start);
        assert_eq!(path.last().unwrap(), &goal);
        assert!(
            p.verify_path(&path),
            "PRM path must be collision-free under the world"
        );
    }

    #[test]
    fn prm_trivial_query_is_short() {
        // start/goal are directly connectable in free space → the roadmap's
        // direct-edge fallback yields a straight, minimal path (post-smoothing a
        // single segment), not a long detour through milestones.
        let p = arm_planner(WorldScene::new());
        let start = vec![0.0, 0.0, 0.0];
        let goal = vec![0.2, -0.1, 0.15];
        assert!(
            p.motion_valid(&start, &goal),
            "test premise: endpoints are directly connectable"
        );
        let path = p.plan_prm(&start, &goal, 150, 6).unwrap();
        assert!(p.verify_path(&path));
        // A directly-connectable query smooths to the straight line.
        assert!(
            (path_length(&path) - dist(&start, &goal)).abs() < 1e-9,
            "trivial query should collapse to the straight segment, got len {:.6} vs {:.6}",
            path_length(&path),
            dist(&start, &goal),
        );
    }

    #[test]
    fn prm_invalid_samples_or_k_errors() {
        let p = arm_planner(WorldScene::new());
        let start = vec![0.0, 0.0, 0.0];
        let goal = vec![0.4, -0.4, 0.4];
        assert!(matches!(
            p.plan_prm(&start, &goal, 0, 6),
            Err(PlanError::InvalidConfig(_))
        ));
        assert!(matches!(
            p.plan_prm(&start, &goal, 100, 0),
            Err(PlanError::InvalidConfig(_))
        ));
    }

    #[test]
    fn prm_guards_dim_bounds_collision() {
        let p = arm_planner(WorldScene::new());
        assert!(matches!(
            p.plan_prm(&[0.0, 0.0], &[0.0, 0.0, 0.0], 100, 6),
            Err(PlanError::Dim { .. })
        ));
        assert!(matches!(
            p.plan_prm(&[0.0, f64::NAN, 0.0], &[0.0, 0.0, 0.0], 100, 6),
            Err(PlanError::NonFinite(_))
        ));
        assert!(matches!(
            p.plan_prm(&[100.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 100, 6),
            Err(PlanError::OutOfBounds { which: "start", .. })
        ));
        assert!(matches!(
            p.plan_prm(&[0.0, 0.0, 0.0], &[0.0, 100.0, 0.0], 100, 6),
            Err(PlanError::OutOfBounds { which: "goal", .. })
        ));
        let folded = vec![0.0, std::f64::consts::PI, std::f64::consts::PI];
        assert!(matches!(
            p.plan_prm(&folded, &[0.0, 0.0, 0.0], 100, 6),
            Err(PlanError::StartInCollision(_))
        ));
        assert!(matches!(
            p.plan_prm(&[0.0, 0.0, 0.0], &folded, 100, 6),
            Err(PlanError::GoalInCollision(_))
        ));
    }
}
