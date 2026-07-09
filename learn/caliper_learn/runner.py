"""Drive a loaded Hub policy through Caliper's safety-monitored control loop.

Generalizes `deploy.rollout_policy` to dict-observation policies (`hub.LoadedPolicy`):
each tick we build the observation dict the checkpoint declares, ask the policy for
ONE action (lerobot's `select_action` pops from its internal chunk queue and only
re-runs the network every `n_action_steps` ticks — see hub.LoadedPolicy), and feed
it to `caliper.ControlLoop.step_with_target`, so the Rust SafetyMonitor stays in the
loop for every commanded target. In-process, no network, no policy server.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HubRolloutResult:
    times: list = field(default_factory=list)
    states: list = field(default_factory=list)  # measured q the policy saw, T x ndof
    actions: list = field(default_factory=list)  # action commanded, T x action_dim
    warn_ticks: int = 0  # ticks on which the SafetyMonitor raised a warning


def default_obs_builder(policy):
    """Map the measured joint state onto every state-like input feature.

    lerobot's ACT requires at least one image or `observation.environment_state`
    input (configuration_act.py `validate_features`), so state-only checkpoints
    carry `observation.environment_state` (ENV) and usually also `observation.state`
    (STATE). In Caliper's sim deploy both are the same signal — measured q — so we
    feed q to each declared state-like feature, after checking the shapes match.
    """
    features = policy.config.input_features

    def build(cl) -> dict[str, np.ndarray]:
        q = np.asarray(cl.q, dtype=np.float32)
        obs = {}
        for name in policy.config.state_feature_names:
            (_, shape) = features[name]
            if shape != q.shape:
                raise ValueError(
                    f"checkpoint feature '{name}' expects shape {shape} but the robot "
                    f"has {q.shape[0]} dof — this checkpoint was trained for a "
                    "different robot"
                )
            obs[name] = q
        return obs

    return build


def run_policy(policy, control_loop, *, fps: int = 50, ticks: int = 200, obs_builder=None) -> HubRolloutResult:
    """Run `policy` closed-loop on `control_loop` for `ticks` steps.

    - `policy`: a `hub.LoadedPolicy` (or anything with `.reset()` and
      `.predict(obs_dict) -> 1-D action`).
    - `control_loop`: a `caliper.ControlLoop` built by the caller (dt should match
      the checkpoint's training cadence, normally 1/fps — the P7 lesson: deploying
      at a different dt than collection consumes action chunks at the wrong rate).
    - `fps`: timestamp base for the result (`times[k] = k / fps`).
    - `obs_builder(control_loop) -> dict`: override how observations are built;
      defaults to mapping measured q onto every state-like input feature.

    The SafetyMonitor inside ControlLoop vets every commanded target; warnings are
    surfaced per-tick via `control_loop.last_warn` and counted in the result.
    """
    if obs_builder is None:
        obs_builder = default_obs_builder(policy)
    if hasattr(policy, "reset"):
        policy.reset()  # fresh action queue / temporal ensemble per rollout

    res = HubRolloutResult()
    for k in range(ticks):
        q = list(control_loop.q)
        obs = obs_builder(control_loop)
        action = np.asarray(policy.predict(obs), dtype=np.float32)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError(f"policy emitted a non-finite action at tick {k}: {action}")
        res.times.append(k / fps)
        res.states.append(q)
        res.actions.append(action.tolist())
        control_loop.step_with_target(action.tolist())
        if control_loop.last_warn:
            res.warn_ticks += 1
    return res
