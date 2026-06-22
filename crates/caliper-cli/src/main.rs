//! `caliper` — the command-line face of the engine.
use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
    }
    Ok(())
}
