"""Seeded evaluation harness: the answer to "my loss went down but the policy
does nothing" and "checkpoint selection is a seed lottery".

Training loss predicts almost nothing about closed-loop competence (BC covariate
shift, chunk-cadence mismatches, normalization drift all hide behind a pretty
loss curve), so the only honest metric is rollouts: N seeded episodes on
`VecSimEnv`, success = the task's termination_fn fired, aggregated with a
Wilson 95% interval so a 3/5 result is reported as the coin-flip it is instead
of "60%". Everything is deterministic — same `EvalConfig` produces a
byte-identical serialized `EvalResult` (tested with exact equality), and every
episode's seed is in the output, so any single failing episode can be
reproduced in isolation.

What runs where:

- `evaluate(policy, task, cfg)` — one policy, N seeded episodes, per-episode
  rows + aggregates + plain-English findings (stable codes, fix hints —
  the caliper-doctor pattern).
- `sweep(checkpoints, task, cfg)` — the checkpoint-selection answer: evaluate
  each candidate under the SAME seeds and return a ranked table (success rate,
  then mean return). Candidates are lerobot Hub checkpoint directories (loaded
  via `hub.load_lerobot_policy`) or in-memory policies/callables, so a scripted
  baseline ranks in the same table as a trained checkpoint.
- `render_text` / `to_json` — tidy report / deterministic JSON (sorted keys).

Policies: a `hub.LoadedPolicy` (dict observations — measured q is mapped onto
every state-like input feature, mirroring `runner.default_obs_builder`), OR any
callable `state -> action` where `state` is the env's (2*ndof,) float32
[qpos, qvel] vector and the action is a (ndof,) qpos target. Objects with a
`.reset()` get it called at every episode start (fresh ACT queue / RNG).

Scope this wave: state-only observations. Image observations (`obs_images` on
`VecSimEnv` + VISUAL input features on Hub checkpoints) arrive with the vision
wave; the episode loop below is the extension point — build the obs dict from
`obs["image"]` alongside `obs["state"]`.

Heavy deps (mujoco via VecSimEnv, torch/lerobot via hub) stay lazy: importing
this module is cheap, matching the rest of the package.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .vec_env import RewardFn, TerminationFn

# ----- findings (the caliper-doctor pattern: stable code + message + fix) ----

# Stable check ids — machine-matchable, never renumbered.
ALL_EPISODES_FAILED = "E001"
SEED_LOTTERY = "E002"
ZERO_REWARD_SIGNAL = "E003"


@dataclass(frozen=True)
class EvalFinding:
    """One diagnosed evaluation smell. `message` states what was observed and
    why it matters; `fix_hint` says what to do about it."""

    code: str
    severity: str  # "error" | "warn" | "info"
    message: str
    fix_hint: Optional[str] = None


# ----- task / config / result ------------------------------------------------


@dataclass
class EvalTask:
    """What to evaluate on: a robot + the reach_task-shaped hooks.

    `reward_fn` / `termination_fn` have the `VecSimEnv.set_task` signature
    (qpos copy, qvel copy, env index). Success is DEFINED as termination_fn
    returning True within `max_steps`; with `termination_fn=None` no episode
    can succeed. Episode initial states are the env's seeded jitter
    (`init_jitter` fraction of each joint's limit range around its midpoint).
    `distance_fn(qpos) -> float`, when present, is reported per episode as
    `final_distance` (reach tasks: see `reach_eval_task`).
    """

    robot: object
    reward_fn: Optional[RewardFn]
    termination_fn: Optional[TerminationFn]
    max_steps: int = 200
    fps: int = 50
    init_jitter: float = 0.2
    distance_fn: Optional[Callable[[np.ndarray], float]] = None


@dataclass(frozen=True)
class EvalConfig:
    """N seeded episodes; episode k runs on seed `base_seed + k`. Same config
    (and a deterministic policy) → byte-identical serialized result."""

    n_episodes: int = 20
    base_seed: int = 0


@dataclass(frozen=True)
class EpisodeResult:
    """One episode. `seed` reproduces it exactly (env init jitter + stepping);
    `episode_return` is the summed per-step reward; `final_distance` is
    `task.distance_fn` at the terminal qpos (None when the task has none)."""

    seed: int
    success: bool
    steps: int
    episode_return: float
    final_distance: Optional[float]


@dataclass(frozen=True)
class EvalResult:
    n_episodes: int
    n_success: int
    success_rate: float
    ci95_low: float  # Wilson 95% interval on the success rate
    ci95_high: float
    mean_return: float
    median_return: float
    mean_steps_to_success: Optional[float]  # over successful episodes; None if 0
    episodes: tuple[EpisodeResult, ...]
    findings: tuple[EvalFinding, ...]


@dataclass(frozen=True)
class SweepEntry:
    """One row of the sweep ranking. `path` is the checkpoint directory when the
    candidate was loaded from disk, None for in-memory policies/baselines."""

    name: str
    path: Optional[str]
    result: EvalResult


# ----- statistics -------------------------------------------------------------

_Z95 = 1.959963984540054  # Phi^-1(0.975)


def wilson_interval(successes: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%).

    Chosen over the normal approximation because eval runs are small and rates
    sit at the edges: 0/N and N/N still get honest, non-degenerate intervals
    (a normal interval collapses to width zero there).
    """
    if n < 1:
        raise ValueError(f"need n >= 1, got n={n}")
    if not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n, got {successes}/{n}")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    # Wilson provably contains p-hat; enforce it against float residue too
    # (at p=0, center-half rounds to ~5.6e-17, an ulp ABOVE the true 0).
    return (min(max(0.0, center - half), p), max(min(1.0, center + half), p))


# ----- policy adapter ---------------------------------------------------------


def _adapt_policy(policy, ndof: int):
    """Normalize the two supported policy kinds to `(step, reset)`.

    step(state (2*ndof,) f32) -> (ndof,) float64 action; reset() per episode.
    """
    if hasattr(policy, "predict") and hasattr(getattr(policy, "config", None), "input_features"):
        # hub.LoadedPolicy: dict observations. Map measured q onto every
        # state-like input feature — same contract as runner.default_obs_builder.
        features = policy.config.input_features
        names = policy.config.state_feature_names

        def step(state: np.ndarray) -> np.ndarray:
            q = np.asarray(state[:ndof], dtype=np.float32)
            obs = {}
            for name in names:
                (_, shape) = features[name]
                if shape != q.shape:
                    raise ValueError(
                        f"checkpoint feature '{name}' expects shape {shape} but the "
                        f"robot has {q.shape[0]} dof — this checkpoint was trained "
                        "for a different robot"
                    )
                obs[name] = q
            return np.asarray(policy.predict(obs), dtype=np.float64)

        return step, policy.reset

    if callable(policy) or hasattr(policy, "predict"):
        fn = policy.predict if hasattr(policy, "predict") else policy

        def step(state: np.ndarray) -> np.ndarray:
            return np.asarray(fn(state), dtype=np.float64)

        return step, getattr(policy, "reset", lambda: None)

    raise TypeError(
        f"unsupported policy {type(policy).__name__}: pass a hub.LoadedPolicy or a "
        "callable state -> action (state = (2*ndof,) [qpos, qvel] float32)"
    )


# ----- the harness ------------------------------------------------------------


def evaluate(policy, task: EvalTask, cfg: EvalConfig = EvalConfig()) -> EvalResult:
    """Run `policy` for `cfg.n_episodes` seeded episodes on `task`.

    Episode k: `env.reset(seed=cfg.base_seed + k)` (seeded init jitter), then up
    to `task.max_steps` control steps at `task.fps`; success = termination_fn
    fired. Deterministic for deterministic policies: same cfg → byte-identical
    `to_json(result)` (a policy with its own unseeded RNG breaks this — seed it
    in `reset()`, the way the diffusion head does).
    """
    if cfg.n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {cfg.n_episodes}")
    if task.max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {task.max_steps}")

    from .vec_env import VecSimEnv  # lazy: pulls in mujoco

    ndof = int(task.robot.ndof)
    step_fn, reset_fn = _adapt_policy(policy, ndof)

    episodes: list[EpisodeResult] = []
    with VecSimEnv(
        task.robot, 1, fps=task.fps, seed=cfg.base_seed, init_jitter=task.init_jitter
    ) as env:
        env.set_task(task.reward_fn, task.termination_fn)
        for k in range(cfg.n_episodes):
            seed = cfg.base_seed + k
            obs = env.reset(seed=seed)
            reset_fn()
            ep_return, steps, success = 0.0, 0, False
            final_state = obs["state"][0]
            for _t in range(task.max_steps):
                action = step_fn(obs["state"][0])
                if action.shape != (ndof,):
                    raise ValueError(
                        f"policy action shape {action.shape} != ({ndof},) at "
                        f"episode seed {seed}, step {steps}"
                    )
                obs, r, te, _tr, info = env.step(action[None, :])
                ep_return += float(r[0])
                steps += 1
                if te[0]:
                    success = True
                    final_state = info["final_observation"][0]  # pre-auto-reset
                    break
                final_state = obs["state"][0]
            final_distance = (
                float(task.distance_fn(np.asarray(final_state[:ndof], dtype=np.float64)))
                if task.distance_fn is not None
                else None
            )
            episodes.append(
                EpisodeResult(
                    seed=seed,
                    success=success,
                    steps=steps,
                    episode_return=ep_return,
                    final_distance=final_distance,
                )
            )

    return _aggregate(episodes)


def _aggregate(episodes: list[EpisodeResult]) -> EvalResult:
    n = len(episodes)
    n_success = sum(1 for e in episodes if e.success)
    lo, hi = wilson_interval(n_success, n)
    returns = [e.episode_return for e in episodes]
    steps_ok = [e.steps for e in episodes if e.success]
    return EvalResult(
        n_episodes=n,
        n_success=n_success,
        success_rate=n_success / n,
        ci95_low=lo,
        ci95_high=hi,
        mean_return=statistics.fmean(returns),
        median_return=float(statistics.median(returns)),
        mean_steps_to_success=statistics.fmean(steps_ok) if steps_ok else None,
        episodes=tuple(episodes),
        findings=tuple(_diagnose(episodes, n_success, lo, hi)),
    )


def _diagnose(
    episodes: list[EpisodeResult], n_success: int, lo: float, hi: float
) -> list[EvalFinding]:
    """The mined failure modes, named in plain English with stable codes."""
    n = len(episodes)
    findings: list[EvalFinding] = []
    if n_success == 0:
        findings.append(
            EvalFinding(
                ALL_EPISODES_FAILED,
                "warn",
                f"0/{n} episodes reached termination — the policy never solved the task.",
                fix_hint=(
                    "Training loss says nothing about this. Check the deploy cadence "
                    "(eval fps vs the collection fps — action chunks consumed at the "
                    "wrong rate), the observation feature mapping (right robot, right "
                    "dof count, right feature names), and sweep() the other checkpoints "
                    "before blaming the data."
                ),
            )
        )
    elif n_success < n and (hi - lo) > 0.5:
        findings.append(
            EvalFinding(
                SEED_LOTTERY,
                "warn",
                f"success flips seed-to-seed: {n_success}/{n}, 95% CI "
                f"[{lo:.2f}, {hi:.2f}] spans {hi - lo:.2f} — this run cannot "
                "distinguish checkpoints.",
                fix_hint=(
                    "Raise EvalConfig.n_episodes (the interval shrinks ~1/sqrt(n)); "
                    "picking a checkpoint on this few seeds is the seed lottery."
                ),
            )
        )
    if all(e.episode_return == 0.0 for e in episodes):
        findings.append(
            EvalFinding(
                ZERO_REWARD_SIGNAL,
                "warn",
                f"every episode returned exactly 0.0 across {n} seeds — no reward "
                "signal reached the evaluator.",
                fix_hint=(
                    "Wire a reward_fn into EvalTask (e.g. reach_eval_task): success "
                    "alone cannot rank near-misses, so returns stay uninformative."
                ),
            )
        )
    return findings


# ----- the one built-in task factory ------------------------------------------


def reach_eval_task(
    robot,
    frame: str,
    target_pos,
    *,
    tol: float = 0.05,
    max_steps: int = 200,
    fps: int = 50,
    init_jitter: float = 0.2,
) -> EvalTask:
    """`vec_env.reach_task` packaged as an EvalTask, plus the matching
    `distance_fn` so episodes report `final_distance` (how close a FAILED
    episode got — the difference between "almost" and "nowhere near")."""
    from .vec_env import reach_task  # local: keep symmetry with evaluate's lazy env

    reward_fn, termination_fn = reach_task(robot, frame, target_pos, tol=tol)
    target = np.asarray(target_pos, dtype=np.float64).reshape(3)

    def distance_fn(qpos: np.ndarray) -> float:
        pose = robot.fk([float(v) for v in qpos], frame)
        p = np.array([pose[0][3], pose[1][3], pose[2][3]])
        return float(np.linalg.norm(p - target))

    return EvalTask(
        robot=robot,
        reward_fn=reward_fn,
        termination_fn=termination_fn,
        max_steps=max_steps,
        fps=fps,
        init_jitter=init_jitter,
        distance_fn=distance_fn,
    )


# ----- checkpoint sweep --------------------------------------------------------


def sweep(checkpoints, task: EvalTask, cfg: EvalConfig = EvalConfig()) -> list[SweepEntry]:
    """Evaluate every candidate under the SAME seeds and rank them — the
    checkpoint-selection answer.

    `checkpoints`: a `{name: candidate}` dict, or a sequence of candidates.
    Each candidate is a lerobot Hub checkpoint directory (str/Path, loaded via
    `hub.load_lerobot_policy`) or an in-memory policy/callable (so a scripted
    baseline ranks in the same table). Ranked by success rate, then mean
    return, then name (a stable total order).
    """
    if isinstance(checkpoints, dict):
        items = list(checkpoints.items())
    else:
        items = [
            (str(c) if isinstance(c, (str, Path)) else f"policy_{i}", c)
            for i, c in enumerate(checkpoints)
        ]
    if not items:
        raise ValueError("sweep needs at least one candidate")

    entries: list[SweepEntry] = []
    for name, cand in items:
        if isinstance(cand, (str, Path)):
            from .hub import load_lerobot_policy  # lazy: pulls in torch (+lerobot)

            policy, path = load_lerobot_policy(cand), str(cand)
        else:
            policy, path = cand, None
        entries.append(SweepEntry(name=name, path=path, result=evaluate(policy, task, cfg)))

    entries.sort(key=lambda e: (-e.result.success_rate, -e.result.mean_return, e.name))
    return entries


# ----- reporting ---------------------------------------------------------------


def to_json(obj, *, indent: int | None = None) -> str:
    """Deterministic JSON for an EvalResult or a sweep ranking (sorted keys —
    byte-identical for identical results; the determinism oracle keys on this)."""
    if isinstance(obj, EvalResult):
        payload = asdict(obj)
    elif isinstance(obj, (list, tuple)) and all(isinstance(e, SweepEntry) for e in obj):
        payload = [asdict(e) for e in obj]
    else:
        raise TypeError(f"to_json takes an EvalResult or a list of SweepEntry, got {type(obj).__name__}")
    return json.dumps(payload, sort_keys=True, indent=indent)


def render_text(obj) -> str:
    """Tidy human report for an EvalResult or a sweep ranking. Per-episode seeds
    are always shown: any single episode can be re-run in isolation."""
    if isinstance(obj, EvalResult):
        return _render_result(obj)
    if isinstance(obj, (list, tuple)) and all(isinstance(e, SweepEntry) for e in obj):
        return _render_sweep(list(obj))
    raise TypeError(f"render_text takes an EvalResult or a list of SweepEntry, got {type(obj).__name__}")


def _fmt_opt(v: Optional[float], spec: str = ".3f") -> str:
    return "-" if v is None else format(v, spec)


def _render_result(r: EvalResult) -> str:
    lines = [
        f"episodes: {r.n_success}/{r.n_episodes} succeeded  "
        f"success_rate={r.success_rate:.3f}  "
        f"wilson95=[{r.ci95_low:.3f}, {r.ci95_high:.3f}]",
        f"return: mean={r.mean_return:.4f} median={r.median_return:.4f}  "
        f"steps-to-success: mean={_fmt_opt(r.mean_steps_to_success, '.1f')}",
        f"{'seed':>8} {'success':>8} {'steps':>6} {'return':>12} {'final_dist':>11}",
    ]
    for e in r.episodes:
        lines.append(
            f"{e.seed:>8} {('yes' if e.success else 'no'):>8} {e.steps:>6} "
            f"{e.episode_return:>12.4f} {_fmt_opt(e.final_distance, '.4f'):>11}"
        )
    for f in r.findings:
        lines.append(f"[{f.severity.upper()}] {f.code}: {f.message}")
        if f.fix_hint:
            lines.append(f"    fix: {f.fix_hint}")
    return "\n".join(lines)


def _render_sweep(entries: list[SweepEntry]) -> str:
    lines = [f"{'rank':>4}  {'success':>9} {'wilson95':>16} {'mean_ret':>10} {'steps':>7}  name"]
    for i, e in enumerate(entries, start=1):
        r = e.result
        lines.append(
            f"{i:>4}  {r.n_success:>4}/{r.n_episodes:<4} "
            f"[{r.ci95_low:.2f}, {r.ci95_high:.2f}] {r.mean_return:>10.4f} "
            f"{_fmt_opt(r.mean_steps_to_success, '.1f'):>7}  {e.name}"
        )
    return "\n".join(lines)
