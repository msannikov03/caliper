"""Generate varied, deterministic sim demonstrations and record them as a
LeRobotDataset v2.1 — the training data for the BC sidecar.

KEY framing (the BC-learnability fix): the action label is a ONE-STEP LOOKAHEAD
along a smooth planned path (state = measured q_k, action = next setpoint q_{k+1}),
NOT a constant per-episode goal. A constant goal would make the map one-to-many
(many states share one label) → unlearnable; the lookahead delta is a small,
state-dependent target with a true zero-loss regression solution. The episode goal
(final pose) is reconstructed at dataloader time, not stored (the v2.1 schema fixes
observation.state/action width = ndof).

torch-free: needs only `caliper` + numpy, so this runs before torch is installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_FIXTURES_ENV = "CALIPER_FIXTURES"
_DEFAULT_FIXTURE = {"planner": "collide_arm", "control": "showcase6"}
_DEFAULT_BOXES = [((0.6, 0.0, 0.3), (0.15, 0.15, 0.15))]
_UNBOUNDED = float(np.pi)


def _fixtures_dir() -> Path:
    env = os.environ.get(_FIXTURES_ENV)
    if env:
        return Path(env)
    # learn/caliper_learn/collect.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2] / "oracle" / "fixtures" / "robots"


def _resolve_urdf(backend: str, urdf: str | None) -> str:
    if urdf is not None:
        return urdf
    return str(_fixtures_dir() / f"{_DEFAULT_FIXTURE[backend]}.urdf")


def _bounds(robot, unbounded: float = _UNBOUNDED) -> np.ndarray:
    """(ndof, 2) sampling bounds from the URDF joint limits (None -> ±unbounded)."""
    lims = robot.joint_limits
    out = np.empty((len(lims), 2), dtype=float)
    for i, lim in enumerate(lims):
        out[i] = (-unbounded, unbounded) if lim is None else (lim[0], lim[1])
    return out


def _sample_free(rng, bounds, cm, max_tries: int) -> list[float]:
    """Rejection-sample a collision-free config (config-space, no IK needed)."""
    for _ in range(max_tries):
        q = [float(rng.uniform(lo, hi)) for lo, hi in bounds]
        if not cm.query(q)["collision"]:
            return q
    raise RuntimeError(f"no collision-free config found in {max_tries} tries")


def collect_demos(
    out_dir: str | os.PathLike,
    *,
    n_episodes: int = 8,
    backend: str = "planner",
    urdf: str | None = None,
    fps: int = 50,
    seed0: int = 0,
    ground: float = -0.1,
    boxes=None,
    dt: float = 0.02,
    ctrl_ticks: int = 300,
    fixed_goal: list[float] | None = None,
    max_goal_tries: int = 200,
    task_template: str = "reach pose {ep}",
) -> str:
    """Record `n_episodes` demonstrations into a LeRobotDataset v2.1 at `out_dir`.

    `backend="planner"` (default, collide_arm): per episode, rejection-sample a
    free start+goal, plan a collision-free retimed trajectory, and store one-step
    lookahead (q_k -> q_{k+1}) frames (plus a terminal hold-at-goal frame).
    `backend="control"` (showcase6): drive a fixed shared goal from varied starts via
    the computed-torque loop. NOTE the control backend's action label is the safety-
    gated position COMMAND, which ramps then holds at the fixed goal — so it is a
    goal-conditioned regulator target (near-constant per episode), NOT a one-step
    lookahead; it is only meaningfully learnable with goal_conditioned=True. Returns
    the dataset root path. Deterministic in `seed0` (byte-identical reruns).
    """
    import caliper  # runtime dep (built via maturin), not a packaging dep

    if backend not in _DEFAULT_FIXTURE:
        raise ValueError(f"backend must be 'planner' or 'control', got {backend!r}")
    boxes = _DEFAULT_BOXES if boxes is None else boxes
    robot = caliper.Robot.from_urdf(_resolve_urdf(backend, urdf))
    n = robot.ndof
    rec = caliper.Recorder(robot, str(out_dir), fps=fps)

    if backend == "planner":
        cm = caliper.CollisionModel(robot, ground=ground, boxes=boxes, margin=0.0)
        bounds = _bounds(robot)
        for ep in range(n_episodes):
            rng = np.random.default_rng(seed0 + ep)
            start = _sample_free(rng, bounds, cm, max_goal_tries)
            goal = _sample_free(rng, bounds, cm, max_goal_tries)
            planner = caliper.Planner(robot, ground=ground, boxes=boxes, seed=seed0 + ep)
            _ts, qs, _qds = planner.plan_trajectory(start, goal, dt=dt)
            if len(qs) < 2:
                continue  # degenerate (start==goal); skip
            rec.start_episode(task_template.format(ep=ep))
            for k in range(len(qs) - 1):
                rec.append(qs[k], qs[k + 1], k / fps)  # state=q_k, action=q_{k+1}
            # terminal frame: settle at the true goal so the reconstructed episode
            # goal (states[-1]) == the planner terminal q_{T-1}, not q_{T-2}, and the
            # policy learns a "hold at goal" sample.
            last = len(qs) - 1
            rec.append(qs[last], qs[last], last / fps)
            rec.finalize_episode()
    else:  # control backend: a single regulator f(q)->action, varied starts
        if not robot.has_inertia:
            raise ValueError("control backend needs a robot with <inertial> data")
        goal = fixed_goal if fixed_goal is not None else [0.0] * n
        bounds = _bounds(robot) * 0.4  # modest starts so the loop settles
        for ep in range(n_episodes):
            rng = np.random.default_rng(seed0 + ep)
            q0 = [float(rng.uniform(lo, hi)) for lo, hi in bounds]
            cl = caliper.ControlLoop(robot, dt=1.0 / fps, start=q0)
            times, states, actions = cl.rollout_to(goal, ctrl_ticks)
            rec.start_episode(task_template.format(ep=ep))
            for s, a, t in zip(states, actions, times):
                rec.append(s, a, t)
            rec.finalize_episode()

    return rec.close()


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Collect Caliper sim demos -> LeRobotDataset v2.1")
    p.add_argument("out", help="output dataset directory")
    p.add_argument("-n", "--episodes", type=int, default=8)
    p.add_argument("--backend", default="planner", choices=["planner", "control"])
    p.add_argument("--urdf", default=None)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.02)
    a = p.parse_args(argv)
    root = collect_demos(
        a.out, n_episodes=a.episodes, backend=a.backend, urdf=a.urdf,
        fps=a.fps, seed0=a.seed, dt=a.dt,
    )
    print(root)


if __name__ == "__main__":
    main()
