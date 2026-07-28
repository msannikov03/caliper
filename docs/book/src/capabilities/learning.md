# Learning sidecar

`learn/caliper_learn` is the Phase-7 **behavior-cloning sidecar**: a minimal,
**pure-torch** imitation-learning package. It is a Python package *outside* the
Cargo workspace, built on the Caliper PyO3 bindings
(`caliper.{Robot, Planner, ControlLoop, Recorder, DatasetReader}`).

"Pure-torch" is a deliberate constraint: **no `lerobot`, no `hydra`, no
`diffusers`** at runtime. The BC-MLP, an ACT-lite transformer, and an optional
DDPM (diffusion) head are hand-written stdlib PyTorch.

## Pipeline

```text
collect  →  data  →  policy  →  train  →  checkpoint  →  deploy
```

- **collect** — generate sim demonstrations into a LeRobotDataset v2.1 (one-step
  lookahead + a terminal frame). (The engine's dataset faces also write the
  **v3.0 native** layout — see [Control & safety](control-safety.md); the
  sidecar's own collector still emits v2.1.)
- **data** — a goal-conditioned torch `Dataset` with **train-only** normalization
  statistics.
- **policy** — `build_policy` for `bc_mlp`, `act_lite`, or the diffusion head;
  normalization stats are stored as model buffers so they round-trip with the
  weights.
- **train** — `fit` on CPU.
- **checkpoint** — save/restore round-trip.
- **deploy** — closed-loop in sim via `ControlLoop.step_with_target`.

## Hard-won lessons (baked into the code)

These are documented because getting them wrong produces silently-wrong results:

- **Train and deploy must share cadence.** Collecting at `fps=50` but deploying
  at the default `dt=1e-3` consumed the one-step lookahead ~20× too fast (only
  ~42% of the gap closed). Deploy at `dt = 1/fps`.
- **ACT deploy must mirror the dataset's windowed history.** A degenerate
  repeated-observation history nullifies the temporal encoder.
- **Normalization round-trip tests are false-greens unless the stats are
  non-identity.** The buffer round-trip must use real (non-identity) stats to
  mean anything.
- **Seed *before* building the policy.** A train-loop seed does not cover weight
  init, because the model is built before `fit` runs — call `seed_all(0)` before
  `build_policy`.

## Deploying a real lerobot checkpoint (the payoff leg)

The sidecar can also run a **real lerobot Hub-convention ACT checkpoint**
closed-loop in the deterministic sim — no lerobot server, no network, CPU
only:

```python
import caliper
from caliper_learn import load_lerobot_policy, run_policy

robot = caliper.Robot.from_urdf("robot.urdf")
policy = load_lerobot_policy("outputs/train/act_reach/checkpoints/last/pretrained_model")
loop = caliper.ControlLoop(robot, dt=1 / 50)   # dt MUST match the training fps
result = run_policy(policy, loop, fps=50, ticks=400)
print(result.warn_ticks, result.times[-1])
```

- **`load_lerobot_policy(path, device="cpu") -> LoadedPolicy`** — loads a
  LOCAL checkpoint directory (model.safetensors + config.json +
  policy_{pre,post}processor.json). **Safetensors only**: any pickle-format
  file (.bin/.pt/.pth/.ckpt/.pkl/.pickle) raises `CheckpointSecurityError` —
  pickles execute arbitrary code on load. ACT policies with state-like input
  features are supported this wave; VISUAL features raise a named
  `NotImplementedError`.
- **`LoadedPolicy`** — wraps the policy + its pre/post processor pipelines
  behind `reset()` / `predict(obs_dict) -> action`. Action chunking uses
  lerobot's OWN `select_action` semantics (one action popped per call,
  replan every `n_action_steps`, temporal ensembling when configured), so
  the in-sim loop consumes chunks exactly like `lerobot-eval` would.
- **`run_policy(policy, control_loop, *, fps, ticks, obs_builder=None)`** —
  the closed-loop runner: builds observations from measured state, vets every
  commanded target through the `SafetyMonitor` inside `ControlLoop`, and
  returns a `HubRolloutResult` (times/states/actions + `warn_ticks`). Any
  object with `reset()`/`predict()` works, so home-grown policies use the
  same runner.

This is the leg the [latency profiler and debugger](verdicts.md) judge:
`profile_rollout` takes the same `LoadedPolicy`, and `analyze_policy` takes
the same checkpoint directory `load_lerobot_policy` loads.

## Diagnostics on top of the pipeline

The sidecar also carries the W2 verdict stack — the seeded eval harness
(`E001`–`E003`), the deploy-loop latency profiler (`L001`–`L003`), the policy
deploy debugger (`P001`–`P008`), and the autopsy that merges them with the
dataset doctor under a single verdict, plus the `caliper-learn` console
script. They get their own chapter:
[Verdicts — eval, profiling & the Policy Autopsy](verdicts.md).

## Honesty about verification

Everything in the sidecar is proven **only by seeded CPU oracles** — a 2-sample
overfit smoke test (loss → 0), a checkpoint round-trip, and a closed-loop sim
rollout. Real GPU training of an ACT / diffusion policy is the documented next
step and is **deliberately never auto-run**. No trained policy or learned
capability is claimed here — only that the pipeline is correct and reproducible
at small scale on CPU.
