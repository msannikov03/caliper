# caliper_learn — Phase 7 (Learning) sidecar

A minimal, **pure-torch** behavior-cloning sidecar for Caliper. No lerobot / hydra /
diffusers in the core — the BC-MLP, ACT-lite transformer, and optional DDPM head are
hand-written stdlib torch (only the Hub-deploy loader in `hub.py` lazily imports
lerobot, when actually loading a Hub checkpoint). It builds on the Caliper PyO3 bindings (`caliper.{Robot, Planner,
ControlLoop, Recorder, DatasetReader}`).

Pipeline: **collect** (sim demos → LeRobotDataset v2.1) → **data** (torch Dataset,
goal-conditioned, train-only norm stats) → **policy** (BC-MLP / ACT-lite / diffusion)
→ **train** (`fit`) → **checkpoint** → **deploy** (closed-loop in sim via
`ControlLoop.step_with_target`).

## Setup (one env: torch + caliper together)

```sh
cd ~/GitHub/caliper
env -u CONDA_PREFIX uv pip install --python .venv/bin/python torch        # CPU/MPS wheel
env -u CONDA_PREFIX .venv/bin/maturin develop -m crates/caliper-py/Cargo.toml   # build `caliper` FIRST
env -u CONDA_PREFIX uv pip install --python .venv/bin/python -e learn      # then caliper_learn
```

## Run the tests (CPU, seconds, no GPU)

```sh
env -u CONDA_PREFIX .venv/bin/python -m pytest learn -v
```

## Sim camera datasets (image-conditioned BC)

`collect_camera_dataset(out, n_episodes=..., fps=30)` replaces a physical camera:
it renders every planner-labelled frame through a MuJoCo offscreen camera
(`sim_camera.SimCameraScene`, built from the robot via `caliper.model_to_mjcf` +
an auto-scaled over-the-shoulder `<camera>`) and writes a native LeRobotDataset
v3.0 with a `dtype: "image"` feature via `caliper.RecorderV3` — PIL-encoded PNG
bytes (compress level 6, lerobot parity), byte-deterministic given the seed. Needs
`mujoco` + `Pillow` (lazy imports; both live in the repo `.venv`). Gate:
`oracle/tests/test_sim_camera.py` — real-lerobot load + a 2-step image-ACT train.

## Vectorized simulation envs (RL/data-gen substrate)

`vec_env.VecSimEnv` is Caliper's ONE vectorized env: N MuJoCo instances (one shared
compiled `MjModel`, N cheap `MjData`) over any caliper Robot, gymnasium.vector-style
`reset`/`step` semantics without importing gymnasium, per-env auto-reset, seeded and
bitwise-deterministic. Substrate, not framework: reward/termination are user hooks
(`set_task`), external RL libraries do the learning; `reach_task` is the single built-in
example and `rollout_random` the smoke/data-gen helper. Actions are qpos targets tracked
by an internal PD + gravity compensation (`model_to_mjcf` emits torque-direct MJCF, no
actuators). `obs_images=True` adds a per-env offscreen camera (GL context each — keep N small).

```python
import caliper
from caliper_learn import VecSimEnv, reach_task

robot = caliper.Robot.from_urdf("arm.urdf")
env = VecSimEnv(robot, num_envs=8, fps=50, seed=0)
env.set_task(*reach_task(robot, "tool0", [0.4, 0.0, 0.3], tol=0.02))
obs = env.reset()
obs, reward, terminated, truncated, info = env.step(actions)  # (8, ndof) qpos targets
```

## Deploying lerobot Hub checkpoints

`hub.load_lerobot_policy(dir)` loads a lerobot-0.4.4-convention checkpoint (config.json +
model.safetensors + policy_{pre,post}processor.json) and `runner.run_policy(policy, cl, fps=50, ticks=...)`
drives it through the safety-monitored `ControlLoop` — ACT, state-only observations this wave
(image checkpoints raise `NotImplementedError`).

**Security stance:** SAFETENSORS ONLY, in-process, no network. lerobot's own remote-inference
path (`PolicyServer`) deserializes pickle over an open port (CVE-2026-25874); Caliper instead
loads weights as pure tensors (`safetensors.torch`, never `torch.load`) and refuses any
checkpoint directory containing `.bin`/`.pt`/`.pth`/`.ckpt`/`.pkl` files.

## The Policy Autopsy

Everyone instruments training (loss curves, LR schedules) and nobody instruments the
loop that actually fails: **dataset → checkpoint → closed-loop deploy**. A policy that
"converged" and then does nothing on the robot dies somewhere in that loop — echo
labels, stale normalization stats, a cadence mismatch, an action queue that can't fit
the tick budget — and every one of those is invisible in the loss curve. The autopsy
runs every diagnostic that applies (dataset doctor `D001`–`D015`, policy debugger
`P001`–`P008`, seeded eval `E001`–`E003` with Wilson-95 CIs, latency profile
`L001`–`L003`) and merges them into ONE report whose verdict leads with the most
severe, most upstream cause:

```python
import caliper
from caliper_learn import autopsy, reach_eval_task

robot = caliper.Robot.from_urdf("arm.urdf")
task = reach_eval_task(robot, "tool0", [0.4, 0.0, 0.3], tol=0.05, fps=50)
rep = autopsy("runs/003000/pretrained_model", "data/demos", robot=robot, task=task)
print(rep.verdict)  # e.g. "The dataset has 0 error(s) and 4 warning(s) (D001, D004, ...)"
```

Or from the terminal (`caliper-learn` is installed with the package; every subcommand
takes `--json` and exits 1 on any error-severity finding, so CI can gate on it):

```sh
caliper-learn autopsy runs/003000/pretrained_model data/demos \
    --urdf arm.urdf --frame tool0 --target 0.40 0.00 0.30
caliper-learn debug runs/003000/pretrained_model --dataset data/demos --urdf arm.urdf
```

The pieces are usable standalone. **Seeded eval** — because training loss predicts
nothing about closed-loop competence, and 3/5 successes is a coin flip, not "60%":

```python
from caliper_learn import EvalConfig, evaluate, render_text, sweep

result = evaluate(policy, task, EvalConfig(n_episodes=20, base_seed=0))
print(render_text(result))          # per-seed rows + Wilson-95 CI + findings
sweep({"ckpt_1k": "runs/001000/pretrained_model",
       "ckpt_3k": "runs/003000/pretrained_model"}, task)  # same seeds, ranked
```

**Latency profile** — the honest achievable rate is `1 / p95(tick)`, and for chunked
policies the refill tick (the one that re-runs the network) is reported separately
from the near-free queue pops:

```python
from caliper_learn import profile_rollout

loop = caliper.ControlLoop(robot, dt=1 / 50, start=[0.0] * robot.ndof)
print(profile_rollout(policy, loop, ticks=200, fps=50).render_text())
```

State-only observations this wave (image probes/eval land with the vision wave); the
full finding catalog lives in the book: `docs/book/src/capabilities/verdicts.md`.

## ⚠️ Deferred: real GPU training

Everything here is proven ONLY by **seeded CPU oracles** — a 2-sample overfit-smoke
(loss → 0), checkpoint round-trip, and a closed-loop sim rollout. Training a real
ACT/Diffusion policy on the 4090s (`compute`/`compute2`) is the documented next step:

```python
from caliper_learn.policy import build_policy, seed_all
from caliper_learn.train import fit, TrainConfig

seed_all(0)  # before build_policy: weight init draws from the global RNG
policy = build_policy("act_lite", {"obs_dim": obs_dim, "action_dim": action_dim}, stats)
fit(policy, train_ds, val_ds, TrainConfig(device="cuda", epochs=...))
```

The `device="cuda"` path is **UNVERIFIED until run on a GPU box** and is **never
auto-run** — launch it yourself on a real dataset.
