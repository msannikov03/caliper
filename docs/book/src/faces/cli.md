# CLI

The `caliper-cli` face exposes the engine as a command-line tool. Each
subcommand parses its arguments and dispatches to the engine — no math lives in
the CLI.

## Subcommands

The full verb set (from the clap `Cmd` enum in `crates/caliper-cli/src/main.rs`):

| Command | Purpose |
|--------|---------|
| `info` | Print engine version / build info. |
| `load` | Load and summarize a URDF model. |
| `fk` | Forward kinematics for a joint vector. |
| `ik` | Inverse kinematics to a target pose (`--analytic` for the closed-form 6R solver). |
| `analyze` | Singularity / manipulability analysis at a configuration (`--json`). |
| `move` | Jerk-limited MOVE_J / MOVE_L / MOVE_C (`--via`); `--topp` for time-optimal retiming. |
| `dyn` | Dynamics at a configuration (RNEA / CRBA / forward). |
| `sim` | Time-step the passive/forced dynamics (q + energy trace). |
| `run` | Deterministic control-loop rollout on a physical sim to a goal. |
| `teleop` | Leader–follower teleoperation demo (pure sim). |
| `record` / `replay` | LeRobotDataset record (v3.0 default, `--format v21`) / replay. |
| `collide` | Self/world collision check at a configuration (`--contacts` for EPA depth, `--json`). |
| `plan` | Collision-free planning: RRT-Connect, `--optimal` (RRT\*), `--prm`. |
| `calibrate` | Joint-zero offset calibration from measured tip poses (`--self-test`). |
| `reach` | Collision-aware reachability of a Cartesian pose (`--json`). |
| `report` | Cycle-time + path-quality report **plus the trajectory lint** (`T001`–`T009`); `--strict` exits non-zero on Error findings. |
| `mjcf` | Export the robot as an MJCF (MuJoCo XML) model. |
| `graph` | `run` / `validate` a `.caliper-graph.json` dataflow graph. |
| `doctor` | **Asset doctor**: diagnose a URDF/xacro (`A001`–`A014`); `--repair` writes a repaired copy. |
| `data doctor` | **Dataset doctor**: pre-training diagnostics over a LeRobotDataset v3.0 (`D001`–`D015`). |
| `data delete` / `split` / `merge` / `tag` | **Dataset edit**: offline episode surgery + the caliper tags sidecar (atomic rewrite, always lerobot-loadable). |

See [Doctors & trajectory lint](../capabilities/doctors.md) for the full
check catalogs.

### `doctor` — asset doctor

```sh
# diagnose only: plain-English findings, most-severe first
cargo run -p caliper-cli -- doctor robot.urdf

# machine-readable
cargo run -p caliper-cli -- doctor robot.urdf --json

# apply every mechanical repair to a COPY (robot.repaired.urdf next to the
# input; the input file is never modified), then re-diagnose the copy
cargo run -p caliper-cli -- doctor robot.urdf --repair

# computed inertials at a custom uniform density (kg/m^3; default 1000)
cargo run -p caliper-cli -- doctor robot.urdf --repair --density 2700 --out fixed.urdf
```

Findings never change the exit code — the report is the product. The command
only errors when the file cannot even be inspected.

### `data doctor` — dataset doctor

```sh
# the root is the directory containing meta/ and data/
cargo run -p caliper-cli -- data doctor ~/datasets/pick_place
cargo run -p caliper-cli -- data doctor ~/datasets/pick_place --json
```

Deterministic: the same dataset bytes always produce the same report. A
healthy dataset reports zero findings.

### `data delete` / `split` / `merge` / `tag` — dataset edit

The offline edit ops (`caliper-dataset::edit`, the same engine behind the
Python `dataset_*` functions and Studio's Data-mode edit bar). Every op
rewrites through the native v3.0 writer into a sibling temp dir and swaps in
atomically, so the result is always lerobot-loadable; survivors are
renumbered densely, tasks remapped, stats recomputed, tags remapped.

```sh
# delete episodes 0 and 3 (refuses to delete ALL episodes)
cargo run -p caliper-cli -- data delete ~/datasets/pick_place --episodes 0,3

# split episode 2 at local frame 150 (both halves keep task + tags)
cargo run -p caliper-cli -- data split ~/datasets/pick_place --episode 2 --frame 150

# merge adjacent episodes 4 and 5 (tasks unioned; timestamps continue 1/fps)
cargo run -p caliper-cli -- data merge ~/datasets/pick_place --first 4 --second 5

# tags sidecar (meta/caliper_tags.json — a caliper extension lerobot ignores)
cargo run -p caliper-cli -- data tag ~/datasets/pick_place                          # list
cargo run -p caliper-cli -- data tag ~/datasets/pick_place --episode 2 --add good,retry
cargo run -p caliper-cli -- data tag ~/datasets/pick_place --episode 2 --remove retry
cargo run -p caliper-cli -- data tag ~/datasets/pick_place --episode 2 --clear
```

### `report` — path report + trajectory lint

```sh
# two-segment MOVE_J with a ground plane, near-miss margin 2 cm, CI-strict
cargo run -p caliper-cli -- report robot.urdf \
  --goal 0.5,0.2,-0.3,0,0,0 --goal 0,0,0,0,0,0 \
  --ground 0.0 --clearance 0.02 --strict
```

## Examples

```sh
cargo run -p caliper-cli -- info
cargo run -p caliper-cli -- fk    robot.urdf --joints 0.1,0.2,0.0,0.0,0.0,0.0
cargo run -p caliper-cli -- ik    robot.urdf --target 1,0,0,0,1,0,0,0,1,0.3,0.0,0.2
cargo run -p caliper-cli -- move  robot.urdf --target 1,0,0,0,1,0,0,0,1,0.3,0.0,0.2
cargo run -p caliper-cli -- plan  robot.urdf --goal 0.5,0.2,-0.3,0,0,0 --ground 0.0
cargo run -p caliper-cli -- graph run robot.urdf my.caliper-graph.json
cargo run -p caliper-cli -- doctor robot.urdf --repair
cargo run -p caliper-cli -- data doctor ~/datasets/pick_place
```
