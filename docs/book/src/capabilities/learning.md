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
