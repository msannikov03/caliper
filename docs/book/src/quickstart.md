# Zero to moving in 10 minutes

One guided path through the whole loop on a **real robot description**: load →
inspect → diagnose → plan → simulate → record a dataset → diagnose *that* →
train a tiny policy → judge the result. Every command below is copy-paste real;
where output is shown, it is what the tools actually print (numbers are
deterministic unless marked otherwise).

Along the way you meet the two things Caliper insists on that most stacks
skip: **doctors before you spend** (asset doctor, dataset doctor, trajectory
lint) and **verdicts after you train** (eval, profile, autopsy) — because
"everybody just starts training and hopes for the best" is exactly the failure
mode this engine exists to close.

## 0 · Install

**Studio (macOS, Apple Silicon):** grab the `.dmg` from
[Releases](https://github.com/msannikov03/caliper/releases). The app is signed
with a Development certificate but **not yet notarized**, so on first open:
right-click the app → **Open** → Open (or allow it under *System Settings →
Privacy & Security*). That's it — one file, no ROS, no GPU, no cloud.

**CLI + Python face (from source — also where the sample robots live):**

```sh
git clone https://github.com/msannikov03/caliper && cd caliper
cargo build --release -p caliper-cli
alias caliper="$PWD/target/release/caliper"      # for this shell session
```

```sh
# the pip route: build the Python bindings into a venv with maturin
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release -m crates/caliper-py/Cargo.toml
```

For steps 6–9 (learning) also install the sidecar and the sim renderer:

```sh
pip install -e learn          # caliper_learn: torch + numpy
pip install mujoco pillow     # camera collector + closed-loop eval
```

> Requires a recent stable Rust (edition 2024; MSRV 1.89) and Python ≥ 3.11.
> Everything below runs from the repo root.

## 1 · Get a robot

The binary ships a small **robot zoo** — real-robot URDFs (Franka Panda,
SO-100, SO-101, Kinova Gen3 lite) embedded in the executable, vendored
verbatim with their licenses; no network involved. We'll use the **SO-101** —
the arm half the hobby-robotics world is building right now:

```sh
caliper fetch --list                    # table the zoo (name, dof, license, source)
URDF="$(caliper fetch so101_new_calib | head -1)"
```

`fetch` materializes the file (default: `~/.cache/caliper/zoo/`), prints its
absolute path on the first line — hence the `head -1` — and then says exactly
what you got: license, source, and the doctor findings this file is *known*
to raise (the meshes are deliberately not embedded; more on that in step 3).

```sh
caliper load "$URDF"
```

```text
robot: so101_new_calib
dof:   6
  [0] 1
  [1] 2
  ...
  [5] 6
```

Six revolute joints, loaded and frozen into a kinematic model. (Yes, the
vendor named the joints `1`–`6`.)

## 2 · Open it in Studio and jog

Launch *Caliper Studio* → **Open URDF…** (⌘O) → pick the fetched
`so101_new_calib.urdf` (step 1 printed its path). You land in **Jog** mode: drag
the joint sliders, or grab the tip gizmo to drive IK live, with the
singularity HUD tracking manipulability as you go. A first-run tour walks you
through the five modes and ⌘K (replay it any time: ⌘K → *Show tour*).

> The zoo ships the URDF **without its STL meshes** (~58 MB that aren't ours
> to embed), so Studio shows the frame skeleton rather than the full body.
> For the full visual treatment pick a bundled sample (e.g. `visual_arm`) from
> the samples dropdown, or File → Open a complete
> [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) checkout.

Studio remembers your session — robot, pose, mode — and restores it on the
next launch.

## 3 · Run the doctor on it

Would this file actually survive physics, collision, MJCF export? Ask before
finding out the hard way:

```sh
caliper doctor "$URDF"
```

```text
asset doctor: 17 error(s), 17 warning(s), 0 info(s)

ERROR (17)
  [A003] collision mesh `assets/base_motor_holder_so101_v1.stl` on link `base` cannot be resolved (tried …)
  ...
```

Exactly the `A003` findings `fetch` warned about: every missing mesh named,
with the exact search paths tried. On your own CAD exports this is the class
of defect that otherwise surfaces one crash at a time, or never. Findings are
**data, not errors**: the exit code stays 0 (the report
*is* the product). Mechanical defects get `--repair`, which writes a fixed
**copy** and never touches your input:

```sh
caliper doctor my_export.urdf --repair        # → my_export.repaired.urdf
```

Studio runs this doctor automatically on every load. Full check catalog
(`A001`–`A014`): [Doctors & trajectory lint](capabilities/doctors.md).

## 4 · Plan a move — and get a verdict on it

Kinematics don't need meshes. Plan a collision-checked path and a
jerk-limited trajectory to a joint goal:

```sh
caliper plan   "$URDF" --goal 0.3,-0.4,0.6,0.4,0.5,0.3
caliper report "$URDF" --goal 0.3,-0.4,0.6,0.4,0.5,0.3
```

`plan` prints the RRT-Connect waypoints (deterministic — seeded PRNG, same
path every run). `report` is the pre-flight verdict on the motion itself:

```text
  cycle time      : 0.3374 s  (100 samples)
  manipulability  : min …   mean …
  sigma_min       : min …  @ t=…s
  joint            limit-margin   vel-util   acc-util
  ...

  LINT: 0 error(s), 1 warning(s)
    [T007] WARN  singular corridor: σ_min falls to 6.0755e-8 (< 1.0000e-2) between t=0.000 s and t=0.102 s (worst at t=0.000 s)
           fix: re-pose the path away from the singular region (see `analyze` escape_direction) or accept DLS damping through it
```

And there's the point of the lint: the all-zeros home pose is a **singularity**,
and the first tenth of a second of this move runs through its corridor —
something you'd otherwise discover as a velocity spike on hardware. The full
catalog (`T001`–`T009`) covers limit violations, 360° detours, jerk spikes and
collision near-misses; `--strict` turns Error findings into a non-zero exit for
CI, and `--json` makes everything machine-readable.

## 5 · Simulate it

The SO-101 file carries inertial data, so dynamics work out of the box:

```sh
caliper sim "$URDF" --duration 1.5 --damping 0.5
```

You get a time-stepped table of `q` and total energy under gravity, ending
with the honest number that says whether the integrator held together:

```text
  energy drift: …
```

In Studio, switch to **Simulate** (⌘3) for the same engine interactively:
gravity drop, computed-torque drive-to-goal, RRT plan, collision check. In
builds with the MuJoCo feature, a **Builtin | Contact** toggle appears — drop
free props on the robot and watch real contact dynamics on the same playback
transport ([contact simulation](capabilities/contact-sim.md)).

## 6 · Record a sim dataset (with a camera)

Time to make training data. The sidecar's camera collector plans
collision-free reaches on a bundled 3-dof fixture (`collide_arm`), renders an
over-the-shoulder MuJoCo camera per frame, and writes a **native
LeRobotDataset v3.0** — images as pre-encoded PNGs, no ffmpeg:

```sh
python -m caliper_learn.collect_sim demo_ds -n 4 --fps 30 --max-frames 80
```

```text
demo_ds
```

Deterministic given `--seed`: reruns produce byte-identical image bytes. (It
defaults to the vendored `collide_arm` fixture — the camera scene is built
from the robot's own inertials and geometry, and the mesh-less SO-101 zoo file
has nothing for a camera to see. Pass `--urdf` for a robot with resolvable
geometry.)

No MuJoCo installed? The engine records a control-loop episode by itself:

```sh
caliper record oracle/fixtures/robots/collide_arm.urdf --out demo_ctl --goal 0.4,-0.3,0.5
```

## 7 · Run the dataset doctor

Before a single GPU-second is spent, ask whether this data can train anything:

```sh
caliper data doctor demo_ds
```

```text
dataset doctor — demo_ds
...
```

Fifteen checks (`D001`–`D015`) stream the dataset in two passes: per-dof
variance collapse, stale `stats.json` (the silent normalization killer),
saturated/echoed actions, contradictory demos, coverage holes, frozen tails,
dead cameras, duplicate episodes. Same contract as every doctor: findings are
data, stable codes, `--json` for machines. Studio's Data mode (⌘5) has the
same doctor behind a button — findings click through to the offending episode.

## 8 · Train a tiny BC policy

Pure-PyTorch, CPU, about a minute — the point is the loop, not the score:

```python
# train_tiny.py — run inside the venv: python train_tiny.py
from caliper_learn.data import DataConfig, make_datasets
from caliper_learn.policy import build_policy
from caliper_learn.train import TrainConfig, fit
from caliper_learn.checkpoint import save_checkpoint

train, val, stats, meta = make_datasets(DataConfig(root="demo_ds"))
policy = build_policy(
    "bc_mlp",
    {"obs_dim": meta["obs_dim"], "action_dim": meta["action_dim"]},
    stats=stats,
    seed=0,
)
hist = fit(policy, train, val, TrainConfig(steps=300, batch_size=32))
print(f"final train loss: {hist['final_train']:.4f}")
save_checkpoint(policy, "bc_tiny.pt")
```

Then close the loop in sim — deploy at the **collection cadence** (`dt = 1/fps`;
deploying a lookahead policy at the wrong rate is a classic silent failure,
documented in [Learning sidecar](capabilities/learning.md)):

```python
import caliper
import numpy as np
from caliper_learn.deploy import rollout_policy

robot = caliper.Robot.from_urdf("oracle/fixtures/robots/collide_arm.urdf")
goal = [0.4, -0.3, 0.5]
res = rollout_policy(policy, robot, goal, ticks=120, dt=1 / 30, fps=30)
print("final |q - goal|:", np.abs(np.array(res.states[-1]) - goal).max())
```

Four episodes and 300 steps won't reach the goal — expect the residual to
*shrink*, not vanish. That gap is precisely what the next step is for.

## 9 · Judge the result — eval and the autopsy

The loss went down. Did the *policy* actually work? Ask the eval harness —
seeded closed-loop episodes with Wilson-95 confidence bounds, so 0/5 stays
honest instead of hiding behind an average:

```python
from caliper_learn.deploy import make_obs
from caliper_learn.eval import EvalConfig, evaluate, reach_eval_task, render_text

task = reach_eval_task(robot, "l3", [0.147, 0.0, 0.575], fps=30)
step = lambda s: policy.predict(make_obs(s[: robot.ndof], goal, policy.obs_dim))
print(render_text(evaluate(step, task, EvalConfig(n_episodes=5))))
```

And when a policy trained in the lerobot ecosystem "does nothing" on deploy,
run the full post-mortem — dataset doctor + policy debugger + eval + latency
profile, one report, one verdict paragraph:

```sh
caliper-learn autopsy <checkpoint_dir> demo_ds --urdf <robot.urdf> \
    --frame <tip> --target 0.147 0.0 0.575
```

It takes lerobot-Hub-convention checkpoints (safetensors only — no pickle is
ever deserialized) and answers the question that burns the most hours: *is it
a data problem, a model problem, or a deploy-loop problem?* Codes, thresholds
and a full walkthrough: [Verdicts — eval, profiling & the Policy
Autopsy](capabilities/verdicts.md).

## Where to next

- [The three faces](faces/index.md) — everything above, on CLI, Python, and Studio.
- [Doctors & trajectory lint](capabilities/doctors.md) — every check code explained.
- [Studio dataflow graph](capabilities/studio-graph.md) — wire the same engine as a Simulink-style graph (⌘4).
- [Verification](verification.md) — why you can trust the numbers: Pinocchio/NumPy cross-validation at 1e-9…1e-15, through the shipped bindings.
