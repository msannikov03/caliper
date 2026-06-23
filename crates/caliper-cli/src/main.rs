//! `caliper` — the command-line face of the engine.
use caliper::dynamics::{self, GRAVITY_EARTH, Simulator};
use caliper::ik::{IkOpts, ik};
use caliper::kinematics::{JacFrame, Jacobian, SingularityParams, jacobian};
use caliper::motion::{CartesianMoveOpts, MotionLimits, MotionLimitsConfig, move_j, move_l};
use caliper::spatial::Se3;
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
            let model = Arc::new(m.clone());
            let mut sim = Simulator::new(model)?;
            sim.set_gravity(grav_vec(&gravity)?);
            sim.set_damping(&vec![damping; m.ndof])?;
            sim.h_max = dt;
            sim.qd_clamp = None;
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
    }
    Ok(())
}
