"""Closed-loop deployment in sim: each tick read the observation, ask the policy
for an action, and step the engine toward it via caliper.ControlLoop.step_with_target.

The policy is any object with `.predict(obs)->action` and `.obs_dim` (the StubPolicy
here, or a trained `caliper_learn` Policy). CPU + deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class StubPolicy:
    """Constant-action policy (no torch) — a deterministic baseline for the deploy
    oracle. obs_dim defaults to len(action) (extend if goal-conditioned)."""

    def __init__(self, action, obs_dim: int | None = None):
        self._a = np.asarray(action, dtype=np.float32)
        self.action_dim = len(self._a)
        self.obs_dim = obs_dim if obs_dim is not None else self.action_dim

    def predict(self, obs):
        return self._a.copy()


def make_obs(q, goal, obs_dim: int) -> np.ndarray:
    """Build the policy observation from the measured q and the goal, matching the
    policy's obs_dim (q, or concat(q, goal) when goal-conditioned)."""
    q = np.asarray(q, dtype=np.float32)
    if obs_dim == q.shape[0]:
        return q
    if obs_dim == 2 * q.shape[0]:
        return np.concatenate([q, np.asarray(goal, dtype=np.float32)])
    raise ValueError(f"obs_dim {obs_dim} != ndof {q.shape[0]} or 2*ndof")


@dataclass
class RolloutResult:
    times: list = field(default_factory=list)
    states: list = field(default_factory=list)  # measured q acted on (the obs), T x ndof
    actions: list = field(default_factory=list)  # action the policy emitted, T x ndof


def rollout_policy(
    policy,
    robot,
    goal,
    *,
    ticks: int = 200,
    dt: float = 1e-3,
    start=None,
    record_to: str | None = None,
    fps: int = 50,
) -> RolloutResult:
    """Run `policy` closed-loop on `robot` (a caliper.Robot) toward `goal` for
    `ticks` steps. Optionally record the rollout as a LeRobotDataset at `record_to`.

    Determinism: deterministic for deterministic policies (e.g. BCMLP/ACTLite); the
    diffusion head samples from its own seeded generator (reset() reseeds it). Note
    the action's lookahead horizon was calibrated at the collection cadence
    (dt = 1/fps); for faithful replay deploy at the same dt (the 1e-3 default queries
    far more often than collection and drives second-order dynamics, not the planned
    kinematic path — ordinary BC covariate shift applies).
    """
    import caliper

    n = robot.ndof
    start = [0.0] * n if start is None else list(start)
    cl = caliper.ControlLoop(robot, dt=dt, start=start)
    if hasattr(policy, "reset"):
        policy.reset()  # fresh per-rollout state (ACT history, diffusion RNG)
    rec = None
    if record_to is not None:
        rec = caliper.Recorder(robot, str(record_to), fps=fps)
        rec.start_episode("policy rollout")

    res = RolloutResult()
    for k in range(ticks):
        q = cl.q  # measured q (the observation source)
        obs = make_obs(q, goal, policy.obs_dim)
        action = np.asarray(policy.predict(obs), dtype=np.float32).tolist()
        res.times.append(k * dt)
        res.states.append(list(q))
        res.actions.append(action)
        if rec is not None:
            rec.append(list(q), action, k / fps)
        cl.step_with_target(action)

    if rec is not None:
        rec.finalize_episode()
        rec.close()
    return res
