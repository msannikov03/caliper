//! `caliper` — the command-line face of the engine.
use caliper::calib::{CalibOptions, calibrate_joint_offsets};
use caliper::dynamics::{self, GRAVITY_EARTH, Simulator};
use caliper::hal::{
    ControlLoop, DatasetReader, DatasetSpec, Gains, HoldSetpoint, JointMap, LeaderFollowerSource,
    PhysicsSimBackend, Recorder, RobotBackend, SimBackend, replay_frame,
};
use caliper::ik::{IkOpts, analytic_ik_6r, ik};
use caliper::kinematics::{JacFrame, Jacobian, SingularityParams, fk_frame, jacobian};
use caliper::motion::{
    CartesianMoveOpts, MotionLimits, MotionLimitsConfig, move_j, move_l, retime_time_optimal,
};
use caliper::planning::path_length;
use caliper::planning::reach::{ReachChecker, ReachConfig, ReachStatus};
use caliper::planning::{Planner, PlannerConfig};
use caliper::spatial::Se3;
use caliper_collision::{CollisionModel, WorldScene};
use clap::{Parser, Subcommand};
use nalgebra::{Matrix3, Matrix4, UnitQuaternion, Vector3};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "caliper",
    version,
    about = "Caliper — a modern robotics engine"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print engine info.
    Info,
    /// Load a robot from a URDF file and print its structure.
    Load {
        /// Path to a .urdf file.
        urdf: PathBuf,
    },
    /// Forward kinematics for a joint configuration (Phase 1).
    Fk {
        /// Path to a .urdf file.
        urdf: PathBuf,
        /// Comma-separated joint values, e.g. --joints 0.1,0.2,0.0
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        joints: Vec<f64>,
    },
    /// Inverse kinematics: solve `frame` to a target pose.
    Ik {
        /// Path to a .urdf file.
        urdf: PathBuf,
        /// Target as 12 numbers: 9 row-major rotation then tx,ty,tz.
        /// e.g. --target 1,0,0,0,1,0,0,0,1,0.3,0.0,0.2
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        target: Vec<f64>,
        /// Comma-separated seed config (length = ndof).
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        seed: Vec<f64>,
        /// Optional frame name; defaults to the tip frame.
        #[arg(long)]
        frame: Option<String>,
        /// Use the closed-form analytic solver (spherical-wrist 6R only); falls
        /// back with a clear notice + non-zero exit if the model is not 6R.
        #[arg(long)]
        analytic: bool,
    },
    /// Singularity / manipulability analysis at a configuration.
    Analyze {
        /// Path to a .urdf file.
        urdf: PathBuf,
        /// Comma-separated joint values (length = ndof).
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        joints: Vec<f64>,
        /// Optional frame name; defaults to the tip frame.
        #[arg(long)]
        frame: Option<String>,
    },
    /// Plan a jerk-limited move and print sampled waypoints.
    Move {
        urdf: PathBuf,
        /// MOVE_J goal config (length = ndof). Mutually exclusive with --target.
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        goal: Option<Vec<f64>>,
        /// MOVE_L Cartesian target: 12 numbers (9 row-major R then tx,ty,tz), as `ik`.
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        target: Option<Vec<f64>>,
        /// Start config (length = ndof). Defaults to all-zeros (home).
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        start: Option<Vec<f64>>,
        /// Sample period seconds for the printed table.
        #[arg(long, default_value_t = 0.05)]
        dt: f64,
        #[arg(long)]
        frame: Option<String>,
        /// Time-optimal (acceleration-limited, corner-stop bang-bang TOPP)
        /// retiming instead of the jerk-limited S-curve.
        /// Joint-space --goal only (not Cartesian --target).
        #[arg(long)]
        time_optimal: bool,
    },
    /// Inverse/forward dynamics at a configuration (Phase 4).
    Dyn {
        urdf: PathBuf,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        joints: Vec<f64>,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        vel: Option<Vec<f64>>,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        accel: Option<Vec<f64>>,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        gravity: Option<Vec<f64>>,
        #[arg(long)]
        mass_matrix: bool,
    },
    /// Time-step the passive/forced dynamics and print q + total energy (Phase 4).
    Sim {
        urdf: PathBuf,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        start: Option<Vec<f64>>,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        torque: Option<Vec<f64>>,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        gravity: Option<Vec<f64>>,
        #[arg(long, default_value_t = 0.0)]
        damping: f64,
        #[arg(long, default_value_t = 2.0)]
        duration: f64,
        #[arg(long, default_value_t = 1e-3)]
        dt: f64,
        #[arg(long, default_value_t = 0.1)]
        print_dt: f64,
    },
    /// Run the deterministic control loop on a physical sim to a goal (Phase 5).
    Run {
        urdf: PathBuf,
        /// Goal joint configuration (length = ndof).
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        goal: Vec<f64>,
        /// Start config (length = ndof). Defaults to all-zeros.
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        start: Option<Vec<f64>>,
        #[arg(long, default_value_t = 100.0)]
        kp: f64,
        #[arg(long, default_value_t = 20.0)]
        kd: f64,
        #[arg(long, default_value_t = 1e-3)]
        dt: f64,
        #[arg(long, default_value_t = 4000)]
        ticks: usize,
    },
    /// Leader-follower teleop demo (pure sim): a follower tracks a scripted leader.
    Teleop {
        urdf: PathBuf,
        #[arg(long, default_value_t = 3000)]
        ticks: usize,
        #[arg(long, default_value_t = 1e-3)]
        dt: f64,
    },
    /// Run a control loop and record a LeRobotDataset v2.1 episode (Phase 5).
    Record {
        urdf: PathBuf,
        /// Output dataset directory.
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        goal: Vec<f64>,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        start: Option<Vec<f64>>,
        #[arg(long, default_value_t = 2000)]
        ticks: usize,
        #[arg(long, default_value_t = 50)]
        fps: u32,
        #[arg(long, default_value = "caliper demo")]
        task: String,
    },
    /// Replay a recorded LeRobotDataset episode through a sim backend (Phase 5).
    Replay {
        urdf: PathBuf,
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long, default_value_t = 0)]
        episode: usize,
    },
    /// Check self/world collisions at a configuration (Phase 5).
    Collide {
        urdf: PathBuf,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        joints: Vec<f64>,
        /// Add a solid ground half-space at z = <ground>.
        #[arg(long, allow_hyphen_values = true)]
        ground: Option<f64>,
        /// Collider inflation margin (m).
        #[arg(long, default_value_t = 0.0)]
        margin: f64,
        /// Also print EPA penetration contacts (normal, depth, witness) per
        /// self-colliding pair.
        #[arg(long)]
        contacts: bool,
    },
    /// Plan a collision-free path to a joint goal or Cartesian --target (Phase 6).
    Plan {
        urdf: PathBuf,
        /// Joint-space goal (length = ndof). Mutually exclusive with --target.
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        goal: Option<Vec<f64>>,
        /// Cartesian goal: 12 numbers (9 row-major R then tx,ty,tz), as `ik`.
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        target: Option<Vec<f64>>,
        /// Start config (length = ndof). Defaults to all-zeros.
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        start: Option<Vec<f64>>,
        /// Solid ground half-space at z = <ground>.
        #[arg(long, allow_hyphen_values = true)]
        ground: Option<f64>,
        /// Obstacle box: 6 numbers cx,cy,cz,hx,hy,hz.
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        obstacle: Option<Vec<f64>>,
        /// PRNG seed (determinism).
        #[arg(long, default_value_t = 0xCA11)]
        seed: u64,
        /// Frame for a Cartesian --target; defaults to the tip frame.
        #[arg(long)]
        frame: Option<String>,
        /// Use the asymptotically-optimal RRT* planner instead of RRT-Connect.
        /// Joint-space --goal only (not Cartesian --target).
        #[arg(long)]
        optimal: bool,
        /// RRT* sample budget (only with --optimal).
        #[arg(long, default_value_t = 4000)]
        iters: usize,
        /// Use the PRM (Probabilistic RoadMap) planner instead of RRT-Connect.
        /// Joint-space --goal only (not Cartesian --target).
        #[arg(long)]
        prm: bool,
        /// PRM milestone budget (only with --prm).
        #[arg(long, default_value_t = 400)]
        samples: usize,
        /// PRM nearest-neighbour degree (only with --prm).
        #[arg(long, default_value_t = 8)]
        k: usize,
    },
    /// Calibrate joint-zero offsets from measured tip poses (kinematic calibration).
    ///
    /// Observations JSON schema (--observations <file.json>):
    ///   {"observations": [
    ///     {"q": [j0, j1, ...],            // commanded config, length = ndof
    ///      "pose": [16 numbers]},         // measured tip pose, 4x4 COLUMN-MAJOR
    ///     {"q": [...],
    ///      "pose": [[r00,r01,r02,tx],     // OR a 4x4 nested (row-major) homogeneous
    ///               [r10,r11,r12,ty],     //    matrix
    ///               [r20,r21,r22,tz],
    ///               [0,0,0,1]]}
    ///   ]}
    /// Each `pose` is FK(q + delta) for the unknown true offset `delta`; the solver
    /// recovers `delta` so that FK(q + delta) matches every measured pose.
    ///
    /// --self-test synthesizes observations from a known --offset via FK, so the
    /// command is runnable headlessly and demonstrates exact offset recovery.
    Calibrate {
        urdf: PathBuf,
        /// Measured frame; defaults to the tip frame.
        #[arg(long)]
        frame: Option<String>,
        /// Observations JSON file (schema above). Mutually exclusive with --self-test.
        #[arg(long)]
        observations: Option<PathBuf>,
        /// Synthesize observations from FK with a known offset (headless demo).
        #[arg(long)]
        self_test: bool,
        /// True offset for --self-test (length = ndof). Defaults to a small canned offset.
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        offset: Option<Vec<f64>>,
    },
    /// Collision-aware reachability of a Cartesian pose (Phase 6).
    Reach {
        urdf: PathBuf,
        /// Cartesian pose: 12 numbers (9 row-major R then tx,ty,tz).
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        target: Vec<f64>,
        #[arg(long, allow_hyphen_values = true)]
        ground: Option<f64>,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        obstacle: Option<Vec<f64>>,
        #[arg(long)]
        frame: Option<String>,
    },
    /// Run or validate a Caliper node graph (.caliper-graph.json) (Phase 8).
    Graph {
        #[command(subcommand)]
        action: GraphCmd,
    },
}

#[derive(Subcommand)]
enum GraphCmd {
    /// Execute a graph against a robot and print a result summary.
    Run {
        /// Path to a .urdf file.
        urdf: PathBuf,
        /// Path to a .caliper-graph.json document.
        graph: PathBuf,
        /// Optional path to write the terminal clip as JSON (ClipData).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Validate a graph against a robot (no execution).
    Validate {
        /// Path to a .urdf file.
        urdf: PathBuf,
        /// Path to a .caliper-graph.json document.
        graph: PathBuf,
    },
}

fn resolve_frame(model: &caliper::model::Model, frame: &Option<String>) -> anyhow::Result<usize> {
    match frame {
        None => Ok(model.tip_frame()),
        Some(name) => model
            .frame_id(name)
            .ok_or_else(|| anyhow::anyhow!("unknown frame `{name}`")),
    }
}

fn grav_vec(g: &Option<Vec<f64>>) -> anyhow::Result<Vector3<f64>> {
    match g {
        None => Ok(GRAVITY_EARTH),
        Some(v) => {
            anyhow::ensure!(v.len() == 3, "--gravity needs 3 values x,y,z");
            anyhow::ensure!(v.iter().all(|x| x.is_finite()), "--gravity must be finite");
            Ok(Vector3::new(v[0], v[1], v[2]))
        }
    }
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::Info => {
            println!("Caliper engine v{}", caliper::VERSION);
            println!("a modern, open robotics engine — kinematics · IK · singularity · control");
        }
        Cmd::Load { urdf } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            println!("robot: {}", robot.name);
            println!("dof:   {}", robot.ndof());
            for (i, j) in robot.joint_names.iter().enumerate() {
                println!("  [{i}] {j}");
            }
        }
        Cmd::Fk { urdf, joints } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            let m = &robot.model;
            anyhow::ensure!(
                joints.len() == m.ndof,
                "expected {} joint value(s), got {}",
                m.ndof,
                joints.len()
            );
            let pose = caliper::kinematics::fk_tip(m, &joints);
            let p = pose.translation();
            let r = pose.0.rotation.euler_angles();
            let tip = m.frame_name(m.tip_frame());
            println!(
                "FK '{}' → tip frame '{}'  (q = {:?})",
                robot.name, tip, joints
            );
            println!("  position : [{:.5}, {:.5}, {:.5}]", p[0], p[1], p[2]);
            println!("  rpy      : [{:.5}, {:.5}, {:.5}]", r.0, r.1, r.2);
        }
        Cmd::Ik {
            urdf,
            target,
            seed,
            frame,
            analytic,
        } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            let m = &robot.model;
            anyhow::ensure!(
                target.len() == 12,
                "expected 12 target values (9 row-major R then tx,ty,tz), got {}",
                target.len()
            );
            anyhow::ensure!(
                seed.len() == m.ndof,
                "expected {} seed value(s), got {}",
                m.ndof,
                seed.len()
            );
            anyhow::ensure!(
                target.iter().chain(seed.iter()).all(|x| x.is_finite()),
                "target/seed contains a non-finite value (NaN/Inf)"
            );
            let f = resolve_frame(m, &frame)?;
            let rot = Matrix3::new(
                target[0], target[1], target[2], target[3], target[4], target[5], target[6],
                target[7], target[8],
            );
            let trans = Vector3::new(target[9], target[10], target[11]);
            // from_matrix projects onto SO(3) (the supplied basis may be non-orthonormal).
            let quat = UnitQuaternion::from_matrix(&rot);
            let target_se3 = Se3::from_parts(trans, quat);
            let tip = m.frame_name(f);
            if analytic {
                match analytic_ik_6r(m, f, &target_se3, Some(seed.as_slice())) {
                    None => {
                        eprintln!(
                            "IK (analytic) '{}' -> frame '{tip}': model is NOT a spherical-wrist 6R",
                            robot.name
                        );
                        eprintln!("  in the canonical alignment the closed form needs;");
                        eprintln!("  rerun without --analytic to use the numeric IK solver.");
                        std::process::exit(2);
                    }
                    Some(branches) if branches.is_empty() => {
                        eprintln!("IK (analytic) '{}' -> frame '{tip}'", robot.name);
                        eprintln!(
                            "  recognised spherical-wrist 6R, but the pose is UNREACHABLE (0 branches)"
                        );
                        std::process::exit(1);
                    }
                    Some(branches) => {
                        // `seed` was given, so branches[0] is the seed-nearest solution.
                        let best = &branches[0];
                        let ee = fk_frame(m, best, f);
                        let residual = ee.inverse().compose(&target_se3).log().0.norm();
                        println!("IK (analytic) '{}' -> frame '{tip}'", robot.name);
                        println!("  branches: {}", branches.len());
                        let qs: Vec<String> = best.iter().map(|v| format!("{v:.6}")).collect();
                        println!("  q (seed-nearest): [{}]", qs.join(", "));
                        println!("  FK residual     : {residual:.6e}");
                    }
                }
            } else {
                let res = ik(m, f, &target_se3, &seed, &IkOpts::default());
                println!("IK '{}' -> frame '{}'", robot.name, tip);
                println!("  success : {}", res.success);
                println!("  iters   : {} (restarts {})", res.iters, res.restarts_used);
                println!("  residual: {:.6e}", res.residual);
                let qs: Vec<String> = res.q.iter().map(|v| format!("{v:.6}")).collect();
                println!("  q       : [{}]", qs.join(", "));
            }
        }
        Cmd::Analyze {
            urdf,
            joints,
            frame,
        } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            let m = &robot.model;
            anyhow::ensure!(
                joints.len() == m.ndof,
                "expected {} joint value(s), got {}",
                m.ndof,
                joints.len()
            );
            anyhow::ensure!(
                joints.iter().all(|x| x.is_finite()),
                "joints contains a non-finite value (NaN/Inf)"
            );
            let f = resolve_frame(m, &frame)?;
            let (_, jac) = jacobian(m, &joints, f, JacFrame::World);
            let rep = Jacobian(jac).analyze(&SingularityParams::default());
            let tip = m.frame_name(f);
            println!(
                "ANALYZE '{}' -> frame '{}'  (q = {:?})",
                robot.name, tip, joints
            );
            println!("  manipulability  : {:.6e}", rep.manipulability);
            if rep.condition_number.is_finite() {
                println!("  condition_number: {:.6e}", rep.condition_number);
            } else {
                println!("  condition_number: inf");
            }
            println!("  sigma_min       : {:.6e}", rep.sigma_min);
            println!(
                "  sigma (3 small) : [{:.6e}, {:.6e}, {:.6e}]",
                rep.sigma[0], rep.sigma[1], rep.sigma[2]
            );
            println!("  kind            : {:?}", rep.kind);
            println!("  offending_joints: {:?}", rep.offending_joints);
            let esc: Vec<String> = rep
                .escape_direction
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect();
            println!("  escape_direction: [{}]", esc.join(", "));
            println!(
                "  nullspace       : {}x{}",
                rep.nullspace_basis.nrows(),
                rep.nullspace_basis.ncols()
            );
            for c in 0..rep.nullspace_basis.ncols() {
                let col: Vec<String> = rep
                    .nullspace_basis
                    .column(c)
                    .iter()
                    .map(|v| format!("{v:.6}"))
                    .collect();
                println!("    [{c}] [{}]", col.join(", "));
            }
        }
        Cmd::Move {
            urdf,
            goal,
            target,
            start,
            dt,
            frame,
            time_optimal,
        } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            let m = &robot.model;
            let start = start.unwrap_or_else(|| vec![0.0; m.ndof]);
            anyhow::ensure!(start.len() == m.ndof, "start needs {} values", m.ndof);
            anyhow::ensure!(
                dt.is_finite() && dt > 0.0,
                "--dt must be a positive finite number"
            );
            anyhow::ensure!(
                goal.is_some() ^ target.is_some(),
                "pass exactly one of --goal / --target"
            );
            let limits = MotionLimits::from_model(m, &MotionLimitsConfig::default())?;
            anyhow::ensure!(
                start.iter().all(|x| x.is_finite()),
                "start contains a non-finite value"
            );
            let (traj, kind) = if let Some(goal) = goal {
                anyhow::ensure!(goal.len() == m.ndof, "goal needs {} values", m.ndof);
                anyhow::ensure!(
                    goal.iter().all(|x| x.is_finite()),
                    "goal contains a non-finite value"
                );
                if time_optimal {
                    // Fine internal knot grid; --dt only paces the printed table.
                    let traj = retime_time_optimal(&[start.clone(), goal], &limits, 1e-3)?;
                    (traj, "MOVE_J (time-optimal)")
                } else {
                    (move_j(m, &start, &goal, &limits)?, "MOVE_J")
                }
            } else {
                anyhow::ensure!(
                    !time_optimal,
                    "--time-optimal supports only joint-space --goal, not Cartesian --target"
                );
                let t = target.unwrap();
                anyhow::ensure!(
                    t.len() == 12 && t.iter().all(|x| x.is_finite()),
                    "target needs 12 finite values (9 row-major R then tx,ty,tz)"
                );
                let rot = Matrix3::new(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]);
                let trans = Vector3::new(t[9], t[10], t[11]);
                let goal_se3 = Se3::from_parts(trans, UnitQuaternion::from_matrix(&rot));
                let f = resolve_frame(m, &frame)?;
                let opts = CartesianMoveOpts::defaults(limits.clone());
                (move_l(m, f, &start, &goal_se3, &opts)?, "MOVE_L")
            };
            println!(
                "{kind} '{}'  duration {:.4}s  ({} dof, completed={})",
                robot.name,
                traj.duration(),
                m.ndof,
                traj.completed
            );
            let mut t = 0.0;
            let dur = traj.duration();
            let mut viol = false;
            loop {
                let s = traj.sample(t);
                let qstr: Vec<String> = s.q.iter().map(|v| format!("{v:.4}")).collect();
                let qdmax = s.qd.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
                let qddmax = s.qdd.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
                for (i, &v) in s.qd.iter().enumerate() {
                    if v.abs() > limits.vmax[i] * 1.001 {
                        viol = true;
                    }
                }
                println!(
                    "  {t:7.3}  [{}]   |qd|<={qdmax:6.3} |qdd|<={qddmax:6.3}",
                    qstr.join(", ")
                );
                if t >= dur {
                    break;
                }
                t = (t + dt).min(dur);
            }
            println!("  within-limits: {}", if viol { "FAIL" } else { "PASS" });
        }
        Cmd::Dyn {
            urdf,
            joints,
            vel,
            accel,
            gravity,
            mass_matrix,
        } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            let m = &robot.model;
            anyhow::ensure!(
                m.has_inertia,
                "model '{}' has no <inertial> data; dynamics needs mass/inertia on every link",
                robot.name
            );
            anyhow::ensure!(
                joints.len() == m.ndof,
                "expected {} joint value(s), got {}",
                m.ndof,
                joints.len()
            );
            let qd = vel.unwrap_or_else(|| vec![0.0; m.ndof]);
            let qdd = accel.unwrap_or_else(|| vec![0.0; m.ndof]);
            anyhow::ensure!(
                qd.len() == m.ndof && qdd.len() == m.ndof,
                "--vel/--accel must have {} values",
                m.ndof
            );
            anyhow::ensure!(
                joints.iter().chain(&qd).chain(&qdd).all(|x| x.is_finite()),
                "joints/vel/accel contains a non-finite value"
            );
            let g = grav_vec(&gravity)?;
            let tau = dynamics::rnea(m, &joints, &qd, &qdd, &g)?;
            let gq = dynamics::rnea(m, &joints, &vec![0.0; m.ndof], &vec![0.0; m.ndof], &g)?;
            println!(
                "DYN '{}'  (g=[{:.3},{:.3},{:.3}])",
                robot.name, g.x, g.y, g.z
            );
            let f = |v: &[f64]| {
                v.iter()
                    .map(|x| format!("{x:8.4}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            println!("  tau               = [{}]   (N·m / N)", f(tau.as_slice()));
            println!("  gravity-only g(q) = [{}]", f(gq.as_slice()));
            if mass_matrix {
                let mm = dynamics::crba(m, &joints)?;
                println!("  M(q) =");
                for r in 0..mm.nrows() {
                    let row: Vec<String> = (0..mm.ncols())
                        .map(|c| format!("{:8.4}", mm[(r, c)]))
                        .collect();
                    println!("    [{}]", row.join(", "));
                }
            }
        }
        Cmd::Sim {
            urdf,
            start,
            torque,
            gravity,
            damping,
            duration,
            dt,
            print_dt,
        } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            let m = &robot.model;
            anyhow::ensure!(
                m.has_inertia,
                "model '{}' has no <inertial> data; sim needs mass/inertia",
                robot.name
            );
            anyhow::ensure!(
                dt.is_finite() && dt > 0.0 && duration.is_finite() && duration > 0.0,
                "--dt and --duration must be positive finite numbers"
            );
            anyhow::ensure!(
                print_dt.is_finite() && print_dt > 0.0,
                "--print_dt must be a positive finite number"
            );
            anyhow::ensure!(damping.is_finite(), "--damping must be finite");
            let q0 = start.unwrap_or_else(|| vec![0.0; m.ndof]);
            anyhow::ensure!(q0.len() == m.ndof, "--start needs {} values", m.ndof);
            anyhow::ensure!(
                q0.iter().all(|x| x.is_finite()),
                "--start contains a non-finite value"
            );
            let model = Arc::new(m.clone());
            let mut sim = Simulator::new(model)?;
            sim.set_gravity(grav_vec(&gravity)?);
            sim.set_damping(&vec![damping; m.ndof])?;
            // Keep the integrator's substepping (default h_max=1e-3) and qd clamp on:
            // step(dt) subdivides a coarse --dt, so it stays stable instead of
            // diverging into a spurious NotSpd abort.
            if let Some(tau) = torque {
                anyhow::ensure!(tau.len() == m.ndof, "--torque needs {} values", m.ndof);
                sim.set_torque(&tau)?;
            }
            sim.reset_to(&q0, &vec![0.0; m.ndof])?;
            let e0 = sim.total_energy();
            println!(
                "SIM '{}'  (dt={dt}, damping={damping}, g={:?})",
                robot.name,
                grav_vec(&gravity)?.as_slice()
            );
            println!("    t      q                              |qd|max     E_total");
            let mut next_print = 0.0;
            let mut t = 0.0;
            let mut settled = false;
            loop {
                if t >= next_print - 1e-12 {
                    let qstr = sim
                        .q()
                        .iter()
                        .map(|x| format!("{x:6.3}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let qdmax = sim.qd().iter().fold(0.0f64, |a, &x| a.max(x.abs()));
                    println!(
                        "  {t:6.3}  [{qstr}]   {qdmax:7.3}   {:10.5}",
                        sim.total_energy()
                    );
                    if qdmax < 1e-3 && damping > 0.0 && t > 0.1 {
                        settled = true;
                    }
                    next_print += print_dt;
                }
                if t >= duration {
                    break;
                }
                sim.step(dt)?;
                t += dt;
            }
            let drift = (sim.total_energy() - e0).abs() / e0.abs().max(1e-6);
            println!("  energy drift: {:.3e} ({:.4}%)", drift, drift * 100.0);
            if settled {
                println!("  settled (|qd|max < 1e-3)");
            }
        }
        Cmd::Run {
            urdf,
            goal,
            start,
            kp,
            kd,
            dt,
            ticks,
        } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            let m = &robot.model;
            anyhow::ensure!(
                m.has_inertia,
                "model '{}' has no <inertial>; control needs dynamics",
                robot.name
            );
            anyhow::ensure!(goal.len() == m.ndof, "--goal needs {} values", m.ndof);
            let start = start.unwrap_or_else(|| vec![0.0; m.ndof]);
            anyhow::ensure!(start.len() == m.ndof, "--start needs {} values", m.ndof);
            anyhow::ensure!(
                goal.iter().chain(&start).all(|x| x.is_finite()),
                "goal/start must be finite"
            );
            anyhow::ensure!(dt.is_finite() && dt > 0.0, "--dt must be > 0");
            anyhow::ensure!(
                kp.is_finite() && kd.is_finite(),
                "--kp and --kd must be finite"
            );
            let model = Arc::new(m.clone());
            let mut backend = PhysicsSimBackend::new(model.clone())?;
            backend.set_state(&start, &vec![0.0; m.ndof])?;
            let mut loopy = ControlLoop::new(backend, model, dt)?.with_gains(Gains { kp, kd });
            let mut sp = HoldSetpoint::new(goal.clone());
            loopy.run_to(&mut sp, ticks)?;
            let q = loopy.backend().joint_positions();
            let err = q
                .iter()
                .zip(&goal)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            println!(
                "RUN '{}'  ticks={ticks} dt={dt} kp={kp} kd={kd}",
                robot.name
            );
            println!("  goal : [{}]", fmt_vec(&goal));
            println!("  final: [{}]", fmt_vec(&q));
            println!(
                "  ||q - goal|| = {err:.3e}  ->  {}",
                if err < 1e-2 { "PASS" } else { "FAIL" }
            );
        }
        Cmd::Teleop { urdf, ticks, dt } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            let m = &robot.model;
            anyhow::ensure!(dt.is_finite() && dt > 0.0, "--dt must be > 0");
            let n = m.ndof;
            let model = Arc::new(m.clone());
            let follower = SimBackend::new(n);
            let leader = Box::new(SimBackend::new(n));
            let mut src = LeaderFollowerSource::new(leader, JointMap::identity(n));
            let mut loopy = ControlLoop::new(follower, model, dt)?;
            let mut worst = 0.0f64;
            for k in 0..ticks {
                let t = k as f64 * dt;
                // gentle leader sweep, starting from zero (no initial jump)
                let lead: Vec<f64> = (0..n)
                    .map(|i| 0.3 * (0.5 * t * (1.0 + 0.2 * i as f64)).sin())
                    .collect();
                src.leader_mut().command_joint_positions(&lead)?;
                loopy.step(&mut src, None)?;
                let fq = loopy.backend().joint_positions();
                let e = fq
                    .iter()
                    .zip(&lead)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                worst = worst.max(e);
            }
            println!(
                "TELEOP '{}'  {n} dof, {ticks} ticks (leader→follower)",
                robot.name
            );
            println!(
                "  max |follower - leader| = {worst:.3e}  ->  {}",
                if worst < 5e-2 { "PASS" } else { "FAIL" }
            );
        }
        Cmd::Record {
            urdf,
            out,
            goal,
            start,
            ticks,
            fps,
            task,
        } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            let m = &robot.model;
            anyhow::ensure!(m.has_inertia, "model '{}' has no <inertial>", robot.name);
            anyhow::ensure!(goal.len() == m.ndof, "--goal needs {} values", m.ndof);
            anyhow::ensure!(fps > 0, "--fps must be > 0");
            let start = start.unwrap_or_else(|| vec![0.0; m.ndof]);
            anyhow::ensure!(start.len() == m.ndof, "--start needs {} values", m.ndof);
            anyhow::ensure!(
                goal.iter().chain(&start).all(|x| x.is_finite()),
                "goal/start must be finite"
            );
            let model = Arc::new(m.clone());
            let mut backend = PhysicsSimBackend::new(model.clone())?;
            backend.set_state(&start, &vec![0.0; m.ndof])?;
            let dt = 1.0 / fps as f64;
            let mut loopy = ControlLoop::new(backend, model, dt)?;
            let mut sp = HoldSetpoint::new(goal.clone());
            let frames = loopy.run_record(&mut sp, ticks)?;
            let mut rec = Recorder::create(&out, DatasetSpec::from_model(m, fps))?;
            rec.start_episode(&task)?;
            for f in &frames {
                rec.append_control_frame(f)?;
            }
            rec.finalize_episode()?;
            let root = rec.close()?;
            println!("RECORD '{}'  ->  {}", robot.name, root.display());
            println!("  episode 0: {ticks} frames @ {fps} fps, task = '{task}'");
            println!("  data/chunk-000/episode_000000.parquet  (LeRobotDataset v2.1)");
        }
        Cmd::Replay {
            urdf,
            dataset,
            episode,
        } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            let m = &robot.model;
            let rd = DatasetReader::open(&dataset)?;
            anyhow::ensure!(
                rd.ndof == m.ndof,
                "dataset ndof {} != robot ndof {}",
                rd.ndof,
                m.ndof
            );
            let ep = rd.read_episode(episode)?;
            anyhow::ensure!(!ep.is_empty(), "episode {episode} is empty");
            let mut b = SimBackend::new(m.ndof);
            println!(
                "REPLAY '{}'  episode {episode}: {} frames @ {} fps",
                robot.name,
                ep.len(),
                rd.fps
            );
            for i in 0..ep.len() {
                replay_frame(&mut b, &ep, i)?;
                if i < 3 || i + 1 == ep.len() {
                    println!(
                        "  [{i:4}] t={:.3}  action=[{}]",
                        ep.timestamps[i],
                        fmt_vec(&b.joint_positions())
                    );
                }
            }
            let last = ep.len() - 1;
            let err = b
                .joint_positions()
                .iter()
                .zip(&ep.actions[last])
                .map(|(a, c)| (a - c).abs())
                .fold(0.0, f64::max);
            println!(
                "  round-trip |q - action_last| = {err:.3e}  ->  {}",
                if err < 1e-6 { "PASS" } else { "FAIL" }
            );
        }
        Cmd::Collide {
            urdf,
            joints,
            ground,
            margin,
            contacts,
        } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            let m = &robot.model;
            anyhow::ensure!(
                joints.len() == m.ndof,
                "expected {} joint value(s), got {}",
                m.ndof,
                joints.len()
            );
            anyhow::ensure!(
                joints.iter().all(|x| x.is_finite()),
                "joints contains a non-finite value"
            );
            let model = Arc::new(m.clone());
            anyhow::ensure!(margin.is_finite(), "--margin must be finite");
            let mut scene = WorldScene::new();
            if let Some(z) = ground {
                anyhow::ensure!(z.is_finite(), "--ground must be finite");
                scene = scene.with_ground(z);
            }
            let cm = CollisionModel::new(model, scene, margin);
            let rep = cm.query(&joints)?;
            println!(
                "COLLIDE '{}'  ({} colliders)",
                robot.name,
                cm.num_colliders()
            );
            let uncovered = cm.uncovered_frames();
            if uncovered > 0 {
                println!(
                    "  ⚠ {uncovered} frame(s) have NO collider (mesh/none) — collisions there are NOT detected"
                );
            }
            println!("  collision: {}", rep.has_collision());
            for (a, b) in &rep.self_pairs {
                println!("  self : {} <-> {}", m.frame_name(*a), m.frame_name(*b));
            }
            for f in &rep.world_hits {
                println!("  world: {}", m.frame_name(*f));
            }
            if contacts {
                let cs = cm.contacts(&joints)?;
                if cs.is_empty() {
                    println!("  contacts: (none)");
                } else {
                    println!("  contacts ({}):", cs.len());
                    for (a, b, c) in &cs {
                        println!(
                            "    {} <-> {}  depth={:.5}  normal=[{:.4}, {:.4}, {:.4}]  witness=[{:.4}, {:.4}, {:.4}]",
                            m.frame_name(*a),
                            m.frame_name(*b),
                            c.depth,
                            c.normal.x,
                            c.normal.y,
                            c.normal.z,
                            c.witness.x,
                            c.witness.y,
                            c.witness.z
                        );
                    }
                }
            }
        }
        Cmd::Plan {
            urdf,
            goal,
            target,
            start,
            ground,
            obstacle,
            seed,
            frame,
            optimal,
            iters,
            prm,
            samples,
            k,
        } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            let m = &robot.model;
            anyhow::ensure!(
                goal.is_some() ^ target.is_some(),
                "pass exactly one of --goal / --target"
            );
            anyhow::ensure!(!(optimal && prm), "pass at most one of --optimal / --prm");
            let start = start.unwrap_or_else(|| vec![0.0; m.ndof]);
            anyhow::ensure!(start.len() == m.ndof, "--start needs {} values", m.ndof);
            anyhow::ensure!(
                start.iter().all(|x| x.is_finite()),
                "--start contains a non-finite value"
            );
            let model = Arc::new(m.clone());
            let scene = world_scene(ground, &obstacle)?;
            let cfg = PlannerConfig {
                seed,
                ..PlannerConfig::default()
            };
            let planner = Planner::new(model, scene, cfg);
            let path = if let Some(goal) = goal {
                anyhow::ensure!(goal.len() == m.ndof, "--goal needs {} values", m.ndof);
                anyhow::ensure!(
                    goal.iter().all(|x| x.is_finite()),
                    "--goal contains a non-finite value"
                );
                if optimal {
                    anyhow::ensure!(iters > 0, "--iters must be > 0");
                    planner.plan_optimal(&start, &goal, iters)?
                } else if prm {
                    anyhow::ensure!(samples > 0, "--samples must be > 0");
                    anyhow::ensure!(k > 0, "--k must be > 0");
                    planner.plan_prm(&start, &goal, samples, k)?
                } else {
                    planner.plan(&start, &goal)?
                }
            } else {
                anyhow::ensure!(
                    !optimal && !prm,
                    "--optimal/--prm support only joint-space --goal, not Cartesian --target"
                );
                let t = target.unwrap();
                let goal_se3 = target_to_se3(&t)?;
                let f = resolve_frame(m, &frame)?;
                planner.plan_to_pose(&start, &goal_se3, f, &start)?
            };
            let free = planner.verify_path(&path);
            println!("PLAN '{}'  seed={seed}", robot.name);
            println!(
                "  planner     : {}",
                if optimal {
                    format!("RRT* (optimal, iters={iters})")
                } else if prm {
                    format!("PRM (samples={samples}, k={k})")
                } else {
                    "RRT-Connect".to_string()
                }
            );
            println!("  waypoints   : {}", path.len());
            println!("  path length : {:.4} rad", path_length(&path));
            if planner.uncovered_frames() > 0 {
                println!(
                    "  ⚠ {} frame(s) have NO collider (mesh/none) — not collision-checked",
                    planner.uncovered_frames()
                );
            }
            println!(
                "  collision-free (re-verified): {}",
                if free { "PASS" } else { "FAIL" }
            );
        }
        Cmd::Calibrate {
            urdf,
            frame,
            observations,
            self_test,
            offset,
        } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            let m = &robot.model;
            let f = resolve_frame(m, &frame)?;
            anyhow::ensure!(
                observations.is_some() ^ self_test,
                "pass exactly one of --observations <file.json> / --self-test"
            );
            let (obs, truth) = if self_test {
                let truth = match offset {
                    Some(o) => {
                        anyhow::ensure!(o.len() == m.ndof, "--offset needs {} values", m.ndof);
                        anyhow::ensure!(
                            o.iter().all(|x| x.is_finite()),
                            "--offset contains a non-finite value"
                        );
                        o
                    }
                    None => default_offset(m.ndof),
                };
                (synth_observations(m, f, &truth), Some(truth))
            } else {
                anyhow::ensure!(
                    offset.is_none(),
                    "--offset only applies to --self-test (it is the synthesized true offset)"
                );
                let path = observations.unwrap();
                (load_observations(&path, m.ndof)?, None)
            };
            let res = calibrate_joint_offsets(m, f, &obs, CalibOptions::default())?;
            println!("CALIBRATE '{}' -> frame '{}'", robot.name, m.frame_name(f));
            println!("  observations: {}", obs.len());
            println!("  offsets     : [{}]", fmt6(&res.offsets));
            println!("  rms_residual: {:.6e}", res.rms_residual);
            println!("  iters       : {}", res.iters);
            println!("  converged   : {}", res.converged);
            if let Some(truth) = truth {
                let err = res
                    .offsets
                    .iter()
                    .zip(&truth)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                println!("  true offset : [{}]", fmt6(&truth));
                println!(
                    "  max|recovered - true| = {err:.3e}  ->  {}",
                    if err < 1e-6 { "PASS" } else { "FAIL" }
                );
            }
        }
        Cmd::Reach {
            urdf,
            target,
            ground,
            obstacle,
            frame,
        } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            let m = &robot.model;
            let pose = target_to_se3(&target)?;
            let f = resolve_frame(m, &frame)?;
            let model = Arc::new(m.clone());
            let scene = world_scene(ground, &obstacle)?;
            let rc = ReachChecker::new(
                model,
                scene,
                ReachConfig {
                    frame: Some(f),
                    ..ReachConfig::default()
                },
            );
            let v = rc.status(&pose);
            let label = match v.status {
                ReachStatus::Reachable => "REACHABLE",
                ReachStatus::Blocked => "BLOCKED (collision)",
                ReachStatus::Unreachable => "UNREACHABLE (out of workspace)",
            };
            println!("REACH '{}' -> frame '{}'", robot.name, m.frame_name(f));
            println!("  status   : {label}");
            println!("  residual : {:.3e}", v.residual);
            if let Some(q) = v.q {
                println!("  config   : [{}]", fmt_vec(&q));
            }
        }
        Cmd::Graph { action } => match action {
            GraphCmd::Run { urdf, graph, out } => {
                let robot = caliper::model::Robot::from_urdf(&urdf)?;
                let doc = load_graph(&graph)?;
                let res = match caliper::graph::run(&doc, &robot) {
                    Ok(res) => res,
                    Err(e) => {
                        print_graph_error(&e);
                        std::process::exit(1);
                    }
                };
                let name = doc.metadata.name.as_deref().unwrap_or(&robot.name);
                println!("GRAPH RUN '{}'  ({} node(s))", name, doc.nodes.len());
                println!("  diagnostics: ok");
                match &res.terminal_clip {
                    Some(clip) => {
                        let dur = clip.times.last().copied().unwrap_or(0.0)
                            - clip.times.first().copied().unwrap_or(0.0);
                        println!("  terminal clip: {} sample(s), {:.4}s", clip.len(), dur);
                    }
                    None => println!("  terminal clip: (none)"),
                }
                if res.scopes.is_empty() {
                    println!("  scopes: (none)");
                } else {
                    println!("  scopes:");
                    for s in &res.scopes {
                        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
                        for &v in &s.y {
                            lo = lo.min(v);
                            hi = hi.max(v);
                        }
                        if s.y.is_empty() {
                            lo = 0.0;
                            hi = 0.0;
                        }
                        println!(
                            "    {} / {}: {} point(s)  min={:.5} max={:.5}",
                            s.node_id,
                            s.signal,
                            s.y.len(),
                            lo,
                            hi
                        );
                    }
                }
                if let Some(path) = out {
                    match &res.terminal_clip {
                        Some(clip) => {
                            let json = serde_json::to_string_pretty(clip)?;
                            std::fs::write(&path, json)?;
                            println!("  wrote terminal clip -> {}", path.display());
                        }
                        None => {
                            anyhow::bail!("--out given but the graph produced no terminal clip")
                        }
                    }
                }
            }
            GraphCmd::Validate { urdf, graph } => {
                let robot = caliper::model::Robot::from_urdf(&urdf)?;
                let doc = load_graph(&graph)?;
                let diag = caliper::graph::validate(&doc, &robot.model);
                let name = doc.metadata.name.as_deref().unwrap_or(&robot.name);
                println!("GRAPH VALIDATE '{}'  ({} node(s))", name, doc.nodes.len());
                if diag.is_ok() {
                    println!("  status: ok");
                    println!("  topo order: [{}]", diag.topo_order.join(", "));
                } else {
                    print_diagnostics(&diag, false);
                    std::process::exit(1);
                }
            }
        },
    }
    Ok(())
}

/// Read + deserialize a `.caliper-graph.json` document.
fn load_graph(path: &std::path::Path) -> anyhow::Result<caliper::graph::GraphDoc> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read graph `{}`: {e}", path.display()))?;
    let doc: caliper::graph::GraphDoc = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("failed to parse graph `{}`: {e}", path.display()))?;
    Ok(doc)
}

/// Print structured validation diagnostics (node/edge errors, cycle, topo order).
fn print_diagnostics(diag: &caliper::graph::Diagnostics, to_stderr: bool) {
    macro_rules! out {
        ($($a:tt)*) => { if to_stderr { eprintln!($($a)*) } else { println!($($a)*) } };
    }
    out!("  status: INVALID");
    for e in &diag.node_errors {
        out!("  node `{}`: {}", e.node_id, e.message);
    }
    for e in &diag.edge_errors {
        out!("  edge [{}]: {}", e.edge_index, e.message);
    }
    if !diag.cycle.is_empty() {
        out!("  cycle: [{}]", diag.cycle.join(", "));
    }
    if !diag.topo_order.is_empty() {
        out!("  topo order: [{}]", diag.topo_order.join(", "));
    }
}

/// Print a `GraphError` (node failure or validation diagnostics) entirely to stderr.
fn print_graph_error(e: &caliper::graph::GraphError) {
    match e {
        caliper::graph::GraphError::Node { node_id, message } => {
            eprintln!("graph error: node `{node_id}` failed: {message}");
        }
        caliper::graph::GraphError::Validation { diagnostics } => {
            eprintln!("graph error: validation failed");
            print_diagnostics(diagnostics, true);
        }
    }
}

/// Build a `WorldScene` from optional `--ground` + a 6-number `--obstacle`.
fn world_scene(ground: Option<f64>, obstacle: &Option<Vec<f64>>) -> anyhow::Result<WorldScene> {
    let mut scene = WorldScene::new();
    if let Some(z) = ground {
        anyhow::ensure!(z.is_finite(), "--ground must be finite");
        scene = scene.with_ground(z);
    }
    if let Some(b) = obstacle {
        anyhow::ensure!(
            b.len() == 6 && b.iter().all(|x| x.is_finite()),
            "--obstacle needs 6 finite values cx,cy,cz,hx,hy,hz"
        );
        scene = scene.add_box([b[0], b[1], b[2]], [b[3], b[4], b[5]]);
    }
    Ok(scene)
}

/// Parse 12 numbers (9 row-major rotation, then tx,ty,tz) into an `Se3`.
fn target_to_se3(t: &[f64]) -> anyhow::Result<Se3> {
    anyhow::ensure!(
        t.len() == 12 && t.iter().all(|x| x.is_finite()),
        "target needs 12 finite values (9 row-major R then tx,ty,tz)"
    );
    let rot = Matrix3::new(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]);
    let trans = Vector3::new(t[9], t[10], t[11]);
    Ok(Se3::from_parts(trans, UnitQuaternion::from_matrix(&rot)))
}

fn fmt_vec(v: &[f64]) -> String {
    v.iter()
        .map(|x| format!("{x:.4}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Format a vector at 6 decimals (calibration offsets are small).
fn fmt6(v: &[f64]) -> String {
    v.iter()
        .map(|x| format!("{x:.6}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A small deterministic canned offset of length `n`, used when `--self-test` is run
/// without an explicit `--offset`. Magnitudes stay small so FK stays well-conditioned.
fn default_offset(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            sign * 0.05 * (1.0 + 0.4 * i as f64)
        })
        .collect()
}

/// Synthesize calibration observations `Tₖ = FK(qₖ + truth)` for deterministic random
/// configs, mirroring the crate's own self-test so `--self-test` demonstrates exact
/// offset recovery headlessly. Deterministic splitmix64 PRNG (repo style, no `rand`).
fn synth_observations(
    model: &caliper::model::Model,
    frame: usize,
    truth: &[f64],
) -> Vec<(Vec<f64>, Se3)> {
    let mut state: u64 = 0xCA11_BACE_D1FF_0001;
    let mut next_u64 = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    (0..16)
        .map(|_| {
            let q: Vec<f64> = (0..model.ndof)
                .map(|_| {
                    let u = (next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                    -1.0 + 2.0 * u
                })
                .collect();
            let phi: Vec<f64> = q.iter().zip(truth).map(|(a, b)| a + b).collect();
            let t = fk_frame(model, &phi, frame);
            (q, t)
        })
        .collect()
}

/// Load + validate calibration observations from a JSON file (schema documented on the
/// `calibrate` subcommand). Each `pose` may be 16 column-major numbers or a 4x4 nested
/// row-major homogeneous matrix.
fn load_observations(path: &std::path::Path, ndof: usize) -> anyhow::Result<Vec<(Vec<f64>, Se3)>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read observations `{}`: {e}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("failed to parse observations `{}`: {e}", path.display()))?;
    let arr = v
        .get("observations")
        .and_then(|x| x.as_array())
        .ok_or_else(|| anyhow::anyhow!("observations JSON must have an `observations` array"))?;
    anyhow::ensure!(!arr.is_empty(), "observations array is empty");
    let mut out = Vec::with_capacity(arr.len());
    for (i, o) in arr.iter().enumerate() {
        let q = o
            .get("q")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow::anyhow!("observation {i}: missing `q` array"))?;
        let q: Vec<f64> = q
            .iter()
            .map(|n| {
                n.as_f64()
                    .ok_or_else(|| anyhow::anyhow!("observation {i}: `q` has a non-number"))
            })
            .collect::<anyhow::Result<_>>()?;
        anyhow::ensure!(
            q.len() == ndof,
            "observation {i}: q has {} value(s), expected ndof = {ndof}",
            q.len()
        );
        anyhow::ensure!(
            q.iter().all(|x| x.is_finite()),
            "observation {i}: q contains a non-finite value"
        );
        let pose = o
            .get("pose")
            .ok_or_else(|| anyhow::anyhow!("observation {i}: missing `pose`"))?;
        let se3 = parse_pose(pose).map_err(|e| anyhow::anyhow!("observation {i}: {e}"))?;
        out.push((q, se3));
    }
    Ok(out)
}

/// Parse a 4x4 homogeneous transform from JSON: either 16 column-major numbers or a
/// 4x4 nested (row-major) array. The rotation block is re-projected onto SO(3).
fn parse_pose(v: &serde_json::Value) -> anyhow::Result<Se3> {
    let arr = v.as_array().ok_or_else(|| {
        anyhow::anyhow!("`pose` must be an array (16 column-major numbers or a 4x4 nested matrix)")
    })?;
    let h: Matrix4<f64> = if arr.len() == 16 {
        let mut m = [0.0_f64; 16];
        for (k, e) in arr.iter().enumerate() {
            m[k] = e
                .as_f64()
                .ok_or_else(|| anyhow::anyhow!("`pose` has a non-number element"))?;
        }
        Matrix4::from_column_slice(&m)
    } else if arr.len() == 4 {
        let mut m = Matrix4::zeros();
        for (r, row) in arr.iter().enumerate() {
            let row = row
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("`pose` 4x4: row {r} is not an array"))?;
            anyhow::ensure!(row.len() == 4, "`pose` 4x4: row {r} needs 4 numbers");
            for (c, e) in row.iter().enumerate() {
                m[(r, c)] = e
                    .as_f64()
                    .ok_or_else(|| anyhow::anyhow!("`pose` 4x4: row {r} has a non-number"))?;
            }
        }
        m
    } else {
        anyhow::bail!("`pose` must be 16 column-major numbers or a 4x4 nested array");
    };
    anyhow::ensure!(
        h.iter().all(|x| x.is_finite()),
        "`pose` contains a non-finite value"
    );
    let rot = h.fixed_view::<3, 3>(0, 0).into_owned();
    let trans = Vector3::new(h[(0, 3)], h[(1, 3)], h[(2, 3)]);
    Ok(Se3::from_parts(trans, UnitQuaternion::from_matrix(&rot)))
}
