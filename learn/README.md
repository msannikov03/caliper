# caliper_learn — Phase 7 (Learning) sidecar

A minimal, **pure-torch** behavior-cloning sidecar for Caliper. No lerobot / hydra /
diffusers — the BC-MLP, ACT-lite transformer, and optional DDPM head are hand-written
stdlib torch. It builds on the Caliper PyO3 bindings (`caliper.{Robot, Planner,
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
