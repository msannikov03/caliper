# CLI

The `caliper-cli` face exposes the engine as a command-line tool. Each
subcommand parses its arguments and dispatches to the engine — no math lives in
the CLI.

## Subcommands

Verified, run-tested subcommands include:

| Command | Purpose |
|--------|---------|
| `info` | Print engine version / build info. |
| `load` | Load and summarize a URDF model. |
| `fk` | Forward kinematics for a joint vector. |
| `ik` | Inverse kinematics to a target pose. |
| `analyze` | Singularity / manipulability analysis. |
| `move` | Generate a jerk-limited motion. |
| `plan` | RRT plan to a goal (with a ground plane). |
| `reach` | Reachability analysis. |
| `dyn` | Dynamics (RNEA/CRBA/forward). |
| `sim` | Simulate a rollout. |
| `run` | Run a control rollout. |
| `teleop` | Leader–follower teleoperation. |
| `record` / `replay` | LeRobot dataset record / replay. |
| `graph` | Load and run a `.caliper-graph.json` dataflow graph. |

## Examples

```sh
cargo run -p caliper-cli -- info
cargo run -p caliper-cli -- fk    robot.urdf --joints 0.1,0.2,0.0,0.0,0.0,0.0
cargo run -p caliper-cli -- ik    robot.urdf --target 1,0,0,0,1,0,0,0,1,0.3,0.0,0.2
cargo run -p caliper-cli -- move  robot.urdf --target 1,0,0,0,1,0,0,0,1,0.3,0.0,0.2
cargo run -p caliper-cli -- plan  robot.urdf --goal 0.5,0.2,-0.3,0,0,0 --ground 0.0
cargo run -p caliper-cli -- graph robot.urdf my.caliper-graph.json
```
