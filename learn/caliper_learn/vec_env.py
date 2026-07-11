"""Vectorized MuJoCo simulation env over a caliper Robot — the RL/data-gen
substrate. Caliper ships ONE vectorized env; external RL libraries (SB3,
CleanRL, torchrl, ...) drive it. We never build an RL framework here: reward
and termination are user-supplied hooks (`set_task`), defaulting to
zero-reward / never-terminate.

Design notes (read before touching):

- ONE compiled `mujoco.MjModel`, N `mujoco.MjData` instances. The model is the
  heavy shared object (geoms, meshes, compiled kinematics); an `MjData` is the
  per-instance state buffer — O(nq + nbody) floats, a few KB for a 6-dof arm —
  so N environments cost roughly ONE model plus N tiny state blocks.

- Control: the Python-exposed `caliper.model_to_mjcf` always emits
  `Actuation::TorqueDirect` — NO `<actuator>` block (the Rust exporter's
  `PositionServo` variant is not reachable from PyO3), so `data.ctrl` does not
  exist (`nu == 0`). The honest path is therefore an internal PD on the qpos
  target, applied per physics substep through `data.qfrc_applied` as a
  COMPUTED-TORQUE law: `tau = M(q) @ (kp*(target - qpos) - kd*qvel) + qfrc_bias`
  with `M` from `mj_fullM(qM)` and `qfrc_bias` MuJoCo's gravity+Coriolis force.
  The mass-matrix scaling is load-bearing, not a nicety: a fixed-gain PD
  (tau = kp*e - kd*qd + bias) diverges on low-inertia links — kp=100 against a
  ~1e-3 kg·m² link is an ω≈300 rad/s error dynamic that a 1 ms integrator
  cannot follow (QACC blows up; MuJoCo silently auto-resets the state, which
  masquerades as 'the arm sits at home'). Scaling by M gives unit-inertia
  error dynamics with ONE gain pair for any robot — the repo's Phase-5 lesson,
  reproduced here empirically before this law replaced the bare PD.
  `ctrl_mode="pd"` is the only mode; the parameter exists so a real
  actuator-servo mode can be added if `model_to_mjcf` ever exposes one.

- Timing: `fps` is the CONTROL rate. Each `step()` runs
  `round(1 / (fps * timestep))` physics substeps per env (default
  timestep 1e-3 → 20 substeps at fps=50).

- Determinism: per-env RNG streams `default_rng(seed + i)`; `reset(seed=...)`
  reseeds, `reset()` continues the streams (gymnasium semantics). MuJoCo
  stepping is deterministic, so same seed + same actions → byte-identical
  trajectories. Auto-resets draw from the env's own stream, preserving this.

- `obs_images=True` builds one `SimCameraScene` PER env. Each scene owns its
  own MjModel copy AND an offscreen GL renderer context — memory and GL
  handles scale linearly with N, and rendering is serial. Keep N small
  (2-8) for image observations; state-only scales much further.

- API mirrors `gymnasium.vector` semantics WITHOUT importing gymnasium:
  `reset() -> obs`, `step(actions) -> (obs, reward, terminated, truncated,
  info)`, same-step auto-reset (a done env returns its RESET observation;
  the terminal one is in `info["final_observation"][i]`).

Heavy deps (mujoco, caliper) are imported lazily, matching the rest of the
package: `import caliper_learn` stays cheap.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .collect import _bounds

# User hooks: (qpos copy, qvel copy, env index) -> reward / done.
RewardFn = Callable[[np.ndarray, np.ndarray, int], float]
TerminationFn = Callable[[np.ndarray, np.ndarray, int], bool]


class VecSimEnv:
    """N independent MuJoCo instances of one caliper Robot (see module doc).

    Actions are qpos TARGETS (N, ndof), tracked by an internal PD; observations
    are `{"state": (N, 2*ndof) float32}` = [qpos, qvel], plus
    `"image": (N, H, W, 3) uint8` when `obs_images=True`.
    """

    def __init__(
        self,
        robot,
        num_envs: int = 1,
        *,
        fps: int = 50,
        ctrl_mode: str = "pd",
        kp: float = 100.0,
        kd: float = 20.0,
        obs_images: bool = False,
        image_size: tuple[int, int] = (64, 64),
        seed: int = 0,
        ground: float | None = None,
        extra_xml: str = "",
        init_jitter: float = 0.2,
        max_episode_steps: int | None = None,
        timestep: float = 1e-3,
        joint_damping: float = 0.0,
    ):
        import caliper  # lazy runtime dep (built via maturin)
        import mujoco  # lazy: keep caliper_learn importable without mujoco

        if ctrl_mode != "pd":
            raise ValueError(
                f"ctrl_mode must be 'pd' (got {ctrl_mode!r}): caliper.model_to_mjcf "
                "emits TorqueDirect MJCF (no <actuator> block), so a MuJoCo position "
                "servo is not available — targets are tracked by an internal PD."
            )
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if not (np.isfinite(kp) and kp > 0.0 and np.isfinite(kd) and kd >= 0.0):
            raise ValueError(f"need finite kp > 0 and kd >= 0, got kp={kp} kd={kd}")
        if not 0.0 <= init_jitter <= 1.0:
            raise ValueError(f"init_jitter must be in [0, 1], got {init_jitter}")
        substeps = int(round(1.0 / (fps * timestep)))
        if substeps < 1:
            raise ValueError(
                f"fps={fps} finer than the physics timestep={timestep} "
                "(need 1/(fps*timestep) >= 1)"
            )

        self._mujoco = mujoco
        self.robot = robot
        self.num_envs = int(num_envs)
        self.ndof = int(robot.ndof)
        self.fps = int(fps)
        self.kp, self.kd = float(kp), float(kd)
        self._substeps = substeps
        self._init_jitter = float(init_jitter)
        self._max_episode_steps = max_episode_steps
        self._seed0 = int(seed)

        xml = caliper.model_to_mjcf(
            robot, ground=ground, extra_xml=extra_xml or None,
            timestep=timestep, joint_damping=joint_damping,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        if self.model.nq != self.ndof:
            raise ValueError(
                f"MJCF nq={self.model.nq} != robot ndof={self.ndof} "
                "(exporter emits one hinge/slide per joint; this should not happen)"
            )
        self._data = [mujoco.MjData(self.model) for _ in range(self.num_envs)]

        # Per-joint sampling bounds (URDF limits; unbounded -> ±pi), midpoints.
        self._bounds = _bounds(robot)
        self._mid = self._bounds.mean(axis=1)
        self._half = 0.5 * (self._bounds[:, 1] - self._bounds[:, 0])

        self._scenes = None
        if obs_images:
            from .sim_camera import SimCameraScene

            h, w = int(image_size[0]), int(image_size[1])
            self._scenes = [
                SimCameraScene.from_robot(robot, width=w, height=h, ground=ground)
                for _ in range(self.num_envs)
            ]

        # Task hooks: substrate default = zero reward, never terminate.
        self._reward_fn: Optional[RewardFn] = None
        self._termination_fn: Optional[TerminationFn] = None

        self._rngs = [np.random.default_rng(self._seed0 + i) for i in range(self.num_envs)]
        self._elapsed = np.zeros(self.num_envs, dtype=np.int64)

    # ----- task hooks ------------------------------------------------------

    def set_task(
        self,
        reward_fn: Optional[RewardFn],
        termination_fn: Optional[TerminationFn] = None,
    ) -> None:
        """Install user reward/termination hooks (called with qpos copy, qvel
        copy, env index AFTER each control step). `None` restores the defaults
        (zero reward / never terminate). Tasks live in user code — this class
        is the substrate, not a task zoo."""
        self._reward_fn = reward_fn
        self._termination_fn = termination_fn

    # ----- gym.vector-style API --------------------------------------------

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset ALL envs. `seed` reseeds the per-env RNG streams
        (`default_rng(seed + i)`); omit it to continue the current streams."""
        if seed is not None:
            self._rngs = [np.random.default_rng(int(seed) + i) for i in range(self.num_envs)]
        for i in range(self.num_envs):
            self._reset_env(i)
        return self._obs()

    def step(self, actions):
        """Apply qpos targets `actions` (N, ndof) for one control period.

        Returns `(obs, reward (N,) float64, terminated (N,) bool,
        truncated (N,) bool, info)`. Done envs auto-reset same-step: `obs`
        holds their fresh reset observation and
        `info["final_observation"][i]` the terminal state vector
        (`info["reset_mask"]` flags which envs reset)."""
        acts = np.asarray(actions, dtype=np.float64)
        if acts.shape != (self.num_envs, self.ndof):
            raise ValueError(
                f"actions shape {acts.shape} != ({self.num_envs}, {self.ndof})"
            )
        if not np.all(np.isfinite(acts)):
            raise ValueError("actions must be finite")

        mujoco = self._mujoco
        reward = np.zeros(self.num_envs, dtype=np.float64)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        final_obs: list[Optional[np.ndarray]] = [None] * self.num_envs

        nv = self.model.nv
        m_dense = np.zeros((nv, nv), dtype=np.float64)
        for i, d in enumerate(self._data):
            target = acts[i]
            for _ in range(self._substeps):
                # Computed torque: unit-inertia error dynamics via the mass
                # matrix (see the module doc — a bare PD explodes on
                # low-inertia links). qM/qfrc_bias are valid from the
                # preceding mj_step/mj_forward on this data.
                a_des = self.kp * (target - d.qpos) - self.kd * d.qvel
                mujoco.mj_fullM(self.model, m_dense, d.qM)
                d.qfrc_applied[:] = m_dense @ a_des + d.qfrc_bias
                mujoco.mj_step(self.model, d)
            self._elapsed[i] += 1

            qpos, qvel = d.qpos.copy(), d.qvel.copy()
            if self._reward_fn is not None:
                reward[i] = float(self._reward_fn(qpos, qvel, i))
            if self._termination_fn is not None:
                terminated[i] = bool(self._termination_fn(qpos, qvel, i))
            if self._max_episode_steps is not None:
                truncated[i] = (
                    not terminated[i] and self._elapsed[i] >= self._max_episode_steps
                )
            if terminated[i] or truncated[i]:
                final_obs[i] = np.concatenate([qpos, qvel]).astype(np.float32)
                self._reset_env(i)  # same-step auto-reset (gym.vector semantics)

        reset_mask = terminated | truncated
        info: dict = {"reset_mask": reset_mask}
        if reset_mask.any():
            info["final_observation"] = final_obs
        return self._obs(), reward, terminated, truncated, info

    # ----- helpers ----------------------------------------------------------

    def _reset_env(self, i: int) -> None:
        """Seeded initial-state jitter: uniform within `init_jitter` fraction
        of each joint's limit range around its midpoint; zero velocity."""
        d = self._data[i]
        self._mujoco.mj_resetData(self.model, d)
        u = self._rngs[i].uniform(-1.0, 1.0, size=self.ndof)
        d.qpos[:] = self._mid + self._init_jitter * self._half * u
        d.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, d)  # populate qfrc_bias for the PD
        self._elapsed[i] = 0

    def _obs(self) -> dict[str, np.ndarray]:
        state = np.empty((self.num_envs, 2 * self.ndof), dtype=np.float32)
        for i, d in enumerate(self._data):
            state[i, : self.ndof] = d.qpos
            state[i, self.ndof :] = d.qvel
        obs = {"state": state}
        if self._scenes is not None:
            imgs = [self._scenes[i].render(d.qpos) for i, d in enumerate(self._data)]
            obs["image"] = np.stack(imgs).astype(np.uint8, copy=False)
        return obs

    def action_bounds(self) -> np.ndarray:
        """(ndof, 2) qpos-target sampling bounds (URDF limits, ±pi if unbounded)."""
        return self._bounds.copy()

    def close(self) -> None:
        if self._scenes is not None:
            for s in self._scenes:
                s.close()
            self._scenes = None

    def __enter__(self) -> "VecSimEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def reach_task(robot, frame: str, target_pos, tol: float = 0.05):
    """THE one built-in example task (docs + tests — not a task zoo): reach a
    world-space point with `frame`. Returns `(reward_fn, termination_fn)` for
    `VecSimEnv.set_task`: reward = -distance(fk(qpos, frame), target_pos),
    terminate when distance < `tol`."""
    target = np.asarray(target_pos, dtype=np.float64).reshape(3)

    def _dist(qpos: np.ndarray) -> float:
        pose = robot.fk([float(v) for v in qpos], frame)  # 4x4 row-major
        p = np.array([pose[0][3], pose[1][3], pose[2][3]])
        return float(np.linalg.norm(p - target))

    def reward_fn(qpos, qvel, i_env) -> float:
        return -_dist(qpos)

    def termination_fn(qpos, qvel, i_env) -> bool:
        return _dist(qpos) < tol

    return reward_fn, termination_fn


def rollout_random(env: VecSimEnv, steps: int, *, seed: int = 0) -> dict[str, np.ndarray]:
    """Smoke/data-gen helper: reset `env`, drive `steps` uniform-random qpos
    targets (within `env.action_bounds()`), stack the results. Returns
    `{"states": (steps, N, 2*ndof) f32, "actions": (steps, N, ndof) f64,
    "rewards": (steps, N) f64, "terminated"/"truncated": (steps, N) bool}`
    plus `"images": (steps, N, H, W, 3) u8` when the env renders images."""
    rng = np.random.default_rng(seed)
    b = env.action_bounds()
    env.reset(seed=seed)
    states, actions, rewards, terms, truncs, images = [], [], [], [], [], []
    for _ in range(steps):
        a = rng.uniform(b[:, 0], b[:, 1], size=(env.num_envs, env.ndof))
        obs, r, te, tr, _info = env.step(a)
        states.append(obs["state"])
        actions.append(a)
        rewards.append(r)
        terms.append(te)
        truncs.append(tr)
        if "image" in obs:
            images.append(obs["image"])
    out = {
        "states": np.stack(states),
        "actions": np.stack(actions),
        "rewards": np.stack(rewards),
        "terminated": np.stack(terms),
        "truncated": np.stack(truncs),
    }
    if images:
        out["images"] = np.stack(images)
    return out
