//! `caliper` — the command-line face of the engine.
use caliper::dynamics::{self, GRAVITY_EARTH, Simulator};
use caliper::hal::{
    ControlLoop, DatasetReader, DatasetSpec, Gains, HoldSetpoint, JointMap, LeaderFollowerSource,
    PhysicsSimBackend, Recorder, RobotBackend, SimBackend, replay_frame,
};
use caliper::ik::{IkOpts, ik};
use caliper::kinematics::{JacFrame, Jacobian, SingularityParams, jacobian};
use caliper::motion::{CartesianMoveOpts, MotionLimits, MotionLimitsConfig, move_j, move_l};
use caliper::planning::path_length;
use caliper::planning::reach::{ReachChecker, ReachConfig, ReachStatus};
use caliper::planning::{Planner, PlannerConfig};
use caliper::spatial::Se3;
use caliper_collision::{CollisionModel, WorldScene};
use clap::{Parser, Subcommand};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
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
            let res = ik(m, f, &target_se3, &seed, &IkOpts::default());
            let tip = m.frame_name(f);
            println!("IK '{}' -> frame '{}'", robot.name, tip);
            println!("  success : {}", res.success);
            println!("  iters   : {} (restarts {})", res.iters, res.restarts_used);
            println!("  residual: {:.6e}", res.residual);
            let qs: Vec<String> = res.q.iter().map(|v| format!("{v:.6}")).collect();
            println!("  q       : [{}]", qs.join(", "));
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
            let (traj, kind) = if let Some(goal) = goal {
                anyhow::ensure!(goal.len() == m.ndof, "goal needs {} values", m.ndof);
                (move_j(m, &start, &goal, &limits)?, "MOVE_J")
            } else {
                let t = target.unwrap();
                anyhow::ensure!(
                    t.len() == 12,
                    "target needs 12 values (9 row-major R then tx,ty,tz)"
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
                dt.is_finite() && dt > 0.0 && duration > 0.0,
                "--dt and --duration must be positive"
            );
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
            let mut scene = WorldScene::new();
            if let Some(z) = ground {
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
        } => {
            let robot = caliper::model::Robot::from_urdf(&urdf)?;
            let m = &robot.model;
            anyhow::ensure!(
                goal.is_some() ^ target.is_some(),
                "pass exactly one of --goal / --target"
            );
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
                planner.plan(&start, &goal)?
            } else {
                let t = target.unwrap();
                let goal_se3 = target_to_se3(&t)?;
                let f = resolve_frame(m, &frame)?;
                planner.plan_to_pose(&start, &goal_se3, f, &start)?
            };
            let free = planner.verify_path(&path);
            println!("PLAN '{}'  seed={seed}", robot.name);
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
    }
    Ok(())
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
