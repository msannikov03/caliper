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

## Deploying lerobot Hub checkpoints

`hub.load_lerobot_policy(dir)` loads a lerobot-0.4.4-convention checkpoint (config.json +
model.safetensors + policy_{pre,post}processor.json) and `runner.run_policy(policy, cl, fps=50, ticks=...)`
drives it through the safety-monitored `ControlLoop` — ACT, state-only observations this wave
(image checkpoints raise `NotImplementedError`).

**Security stance:** SAFETENSORS ONLY, in-process, no network. lerobot's own remote-inference
path (`PolicyServer`) deserializes pickle over an open port (CVE-2026-25874); Caliper instead
loads weights as pure tensors (`safetensors.torch`, never `torch.load`) and refuses any
checkpoint directory containing `.bin`/`.pt`/`.pth`/`.ckpt`/`.pkl` files.

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
