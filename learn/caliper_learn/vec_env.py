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

- `randomization=RandomizationSpec(...)` (see `randomize.py`) draws per-env
  parameters at every reset, seeded from THAT env's stream (drawn before the
  qpos jitter, so a fixed seed reproduces both). Runtime fields (gains,
  spawn, camera) are applied in place. MODEL-level fields (mass, damping,
  friction, gravity) break the one-shared-MjModel design above — each such
  env gets its OWN MjModel recompiled from the randomized MJCF at every
  reset. The honest cost: N × the model memory (the very thing the shared
  design avoids) plus an XML parse + compile per reset (~ms for a small
  arm). That is the price of physics randomization; state-only N stays cheap
  because MjData is still tiny. The current draws are `info["randomization"]`
  on every step (plain diffable dicts, index-aligned with envs; an auto-reset
  env's terminal-transition draw moves to `info["final_randomization"][i]`,
  mirroring `final_observation`). Camera
  jitter moves the RENDER scene's camera only; mass/damping do not affect
  rendering, so scenes are not rebuilt.

- PROPS (free-floating objects) live in the scene through `extra_xml`: the
  Python `caliper.model_to_mjcf` has no structured `props=` argument (the
  Rust `MjcfOptions::props` path is not exposed through PyO3 yet), so a prop
  is a `<body><freejoint/>...</body>` passed verbatim. Their free joints add
  7 qpos / 6 qvel EACH, and `extra_xml` is injected BEFORE the robot bodies,
  so the robot's dofs are NOT the qpos prefix in that case. Every robot-side
  read/write therefore goes through `_q_sel`/`_v_sel`, resolved by JOINT NAME
  (`mujoco_name` = caliper's spelling with whitespace → `_`, the same
  resolution `caliper-sim-mujoco`'s own sim layer does). With no props the
  selections are `slice(0, ndof)` and every operation is bit-identical to the
  prefix code they replaced. The computed-torque law stays exact under props:
  a free body is its OWN kinematic tree, so `M` has no robot↔prop coupling
  and contact forces reach the arm as external forces (which is the point —
  the arm should feel the object).

- `success=<predicate>` (see `success.py`) scores the scene: each env gets its
  OWN clone (predicates capture per-episode baselines), reset at every episode
  start, and `info["success"]` carries the per-env verdict, index-aligned with
  the returned obs. On an auto-reset the TERMINAL verdict moves to
  `info["final_success"][i]`, mirroring `final_observation`. Success is
  REPORTED, not terminating: pass a `termination_fn` if the episode should end
  (the eval harness does exactly that).

- API mirrors `gymnasium.vector` semantics WITHOUT importing gymnasium:
  `reset() -> obs`, `step(actions) -> (obs, reward, terminated, truncated,
  info)`, same-step auto-reset (a done env returns its RESET observation;
  the terminal one is in `info["final_observation"][i]`).

Heavy deps (mujoco, caliper) are imported lazily, matching the rest of the
package: `import caliper_learn` stays cheap.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Optional

import numpy as np

from .collect import _bounds
from .randomize import RandomizationSpec, apply_to_env, apply_to_mjcf, sample
from .success import SuccessPredicate, as_predicate, clone, state_from_mujoco

# User hooks: (qpos copy, qvel copy, env index) -> reward / done.
RewardFn = Callable[[np.ndarray, np.ndarray, int], float]
TerminationFn = Callable[[np.ndarray, np.ndarray, int], bool]


def _mujoco_name(name: str) -> str:
    """The identifier MuJoCo actually REGISTERS for a caliper joint name:
    whitespace becomes `_` (MuJoCo names carry no spaces), while the XML
    escaping the exporter adds is undone by the parser at load time. Mirrors
    `caliper-sim-mujoco`'s `mjcf::mujoco_name`, which is how the Rust sim layer
    resolves the same joints — one spelling rule, two languages."""
    return "".join("_" if ch.isspace() else ch for ch in name)


def _selection(addresses: list[int]):
    """Address list → a `slice` when the run is contiguous ascending (so the
    hot loop indexes VIEWS), else an index array."""
    a = np.asarray(addresses, dtype=np.intp)
    if a.size and bool(np.all(np.diff(a) == 1)):
        return slice(int(a[0]), int(a[-1]) + 1)
    return a


def _robot_selectors(mujoco, model, robot):
    """Locate the robot's qpos/qvel entries in the compiled model, by joint
    NAME rather than by position.

    Props enter the scene through `extra_xml`, which the exporter injects
    BEFORE the robot bodies — each free joint then takes 7 qpos / 6 qvel ahead
    of the arm, so `qpos[:ndof]` would silently address the prop. Returns
    `(q_sel, v_sel, v_ix)`, the last being the 2-D selection that cuts the
    robot's block out of the dense mass matrix. With no props these are
    `slice(0, ndof)` and the arithmetic is bit-identical to plain prefix
    slicing.
    """
    qadr: list[int] = []
    vadr: list[int] = []
    for name in robot.joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _mujoco_name(name))
        if jid < 0:
            raise ValueError(
                f"robot joint {name!r} is not in the compiled MJCF — the exporter "
                "emits one hinge/slide per joint under that name, so either "
                "extra_xml redefined it or this model did not come from "
                "caliper.model_to_mjcf"
            )
        kind = model.jnt_type[jid]
        if kind not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(
                f"robot joint {name!r} compiled as MuJoCo joint type {int(kind)}, "
                "expected hinge or slide (one qpos/qvel entry each)"
            )
        qadr.append(int(model.jnt_qposadr[jid]))
        vadr.append(int(model.jnt_dofadr[jid]))
    q_sel, v_sel = _selection(qadr), _selection(vadr)
    v_ix = (v_sel, v_sel) if isinstance(v_sel, slice) else np.ix_(vadr, vadr)
    return q_sel, v_sel, v_ix


class VecSimEnv:
    """N independent MuJoCo instances of one caliper Robot (see module doc).

    Actions are qpos TARGETS (N, ndof), tracked by an internal PD; observations
    are `{"state": (N, 2*ndof) float32}` = [qpos, qvel], plus
    `"image": (N, H, W, 3) uint8` when `obs_images=True`. Observations and
    actions cover the ROBOT only — props in the scene are addressed by name
    through `success_state()` / the `success` predicate, never by index.
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
        randomization: RandomizationSpec | None = None,
        success=None,
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
        if not math.isclose(substeps * fps * timestep, 1.0, rel_tol=1e-6):
            eff_fps = 1.0 / (substeps * timestep)
            warnings.warn(
                f"fps={fps} does not divide the physics timestep={timestep}: each "
                f"control step runs {substeps} substeps = "
                f"{substeps * timestep * 1e3:.3f} ms, an effective rate of "
                f"{eff_fps:.2f} Hz. Anything timestamped at {fps} Hz (recorded "
                f"datasets, deploy cadence) silently drifts "
                f"{abs(1.0 / fps - substeps * timestep) * 1e3:.3f} ms per tick — "
                "pick fps and timestep so that 1/(fps*timestep) is an integer",
                stacklevel=2,
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
        self._q_sel, self._v_sel, self._v_ix = _robot_selectors(
            mujoco, self.model, robot
        )
        self._data = [mujoco.MjData(self.model) for _ in range(self.num_envs)]

        # Domain randomization (see the module doc for the memory tradeoff):
        # _models[i] is the shared base model until a MODEL-level draw
        # replaces it with that env's own recompiled copy at reset.
        self._rand = randomization
        self._base_xml = xml
        self._models = [self.model] * self.num_envs
        self._draws: list[Optional[dict]] = [None] * self.num_envs
        self._kp = np.full(self.num_envs, self.kp, dtype=np.float64)
        self._kd = np.full(self.num_envs, self.kd, dtype=np.float64)

        # Per-joint sampling bounds (URDF limits; unbounded -> ±pi), midpoints.
        self._bounds = _bounds(robot)
        self._mid = self._bounds.mean(axis=1)
        self._half = 0.5 * (self._bounds[:, 1] - self._bounds[:, 0])

        self._scenes = None
        if obs_images:
            from .sim_camera import SimCameraScene

            h, w = int(image_size[0]), int(image_size[1])
            # The render scene must be the SAME scene: `extra_xml` goes through
            # too, or a prop the policy is supposed to grasp would be missing
            # from the very images it is trained on. The scene's camera adds no
            # dofs, so its qpos layout matches this env's and `_obs` can hand
            # it the full state (props included, at their live pose).
            self._scenes = [
                SimCameraScene.from_robot(
                    robot, width=w, height=h, ground=ground, extra_xml=extra_xml
                )
                for _ in range(self.num_envs)
            ]

        # Task hooks: substrate default = zero reward, never terminate.
        self._reward_fn: Optional[RewardFn] = None
        self._termination_fn: Optional[TerminationFn] = None

        # Success predicate: ONE spec, N independent clones — `Lifted` captures
        # a per-episode baseline, so a shared instance would score env 1 against
        # env 0's start (see success.clone).
        self._success: Optional[list[SuccessPredicate]] = None
        if success is not None:
            spec = as_predicate(success)
            self._success = [clone(spec) for _ in range(self.num_envs)]
            self.success_predicate: Optional[SuccessPredicate] = spec
        else:
            self.success_predicate = None

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
        (`info["reset_mask"]` flags which envs reset). With randomization,
        `info["randomization"]` is index-aligned with the RETURNED obs (a
        reset env's entry is its fresh draw); the draw that governed a reset
        env's terminal transition is preserved in
        `info["final_randomization"][i]` (the final_observation analog).
        With a `success` predicate, `info["success"]` is the (N,) bool verdict
        on the RETURNED state and `info["final_success"][i]` the verdict on a
        reset env's TERMINAL state (that analog once more)."""
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
        final_rand: list[Optional[dict]] = [None] * self.num_envs
        success = np.zeros(self.num_envs, dtype=bool)
        final_success: list[Optional[bool]] = [None] * self.num_envs

        nv = self.model.nv
        m_dense = np.zeros((nv, nv), dtype=np.float64)
        for i in range(self.num_envs):
            d, m = self._data[i], self._models[i]
            target = acts[i]
            for _ in range(self._substeps):
                # Computed torque: unit-inertia error dynamics via the mass
                # matrix (see the module doc — a bare PD explodes on
                # low-inertia links). qM/qfrc_bias are valid from the
                # preceding mj_step/mj_forward on this data. Gains are
                # per-env (kp_scale/kd_scale randomization).
                # `_q_sel`/`_v_sel` are the robot's own entries — plain slices
                # unless props share the model (see the module doc).
                a_des = (
                    self._kp[i] * (target - d.qpos[self._q_sel])
                    - self._kd[i] * d.qvel[self._v_sel]
                )
                mujoco.mj_fullM(m, m_dense, d.qM)
                d.qfrc_applied[self._v_sel] = (
                    m_dense[self._v_ix] @ a_des + d.qfrc_bias[self._v_sel]
                )
                mujoco.mj_step(m, d)
            self._elapsed[i] += 1

            qpos = np.asarray(d.qpos[self._q_sel]).copy()
            qvel = np.asarray(d.qvel[self._v_sel]).copy()
            if self._reward_fn is not None:
                reward[i] = float(self._reward_fn(qpos, qvel, i))
            if self._termination_fn is not None:
                terminated[i] = bool(self._termination_fn(qpos, qvel, i))
            if self._max_episode_steps is not None:
                truncated[i] = (
                    not terminated[i] and self._elapsed[i] >= self._max_episode_steps
                )
            if self._success is not None:
                success[i] = self._judge(i)
            if terminated[i] or truncated[i]:
                final_obs[i] = np.concatenate([qpos, qvel]).astype(np.float32)
                # Capture the draw that governed THIS step before _reset_env
                # overwrites it with the next episode's draw (the
                # final_observation analog — otherwise it is lost).
                final_rand[i] = self._draws[i]
                final_success[i] = bool(success[i])
                self._reset_env(i)  # same-step auto-reset (gym.vector semantics)
                if self._success is not None:
                    # Re-judge the FRESH state so info["success"] stays aligned
                    # with the observation actually returned.
                    success[i] = self._judge(i)

        reset_mask = terminated | truncated
        info: dict = {"reset_mask": reset_mask}
        if reset_mask.any():
            info["final_observation"] = final_obs
            if self._rand is not None:
                info["final_randomization"] = final_rand
            if self._success is not None:
                info["final_success"] = final_success
        if self._rand is not None:
            info["randomization"] = list(self._draws)  # index-aligned with envs
        if self._success is not None:
            info["success"] = success
        return self._obs(), reward, terminated, truncated, info

    # ----- helpers ----------------------------------------------------------

    def _reset_env(self, i: int) -> None:
        """Seeded initial-state jitter: uniform within `init_jitter` fraction
        of each joint's limit range around its midpoint; zero velocity.

        With `randomization`, the draw comes FIRST from the same per-env
        stream (fixed order → a seed reproduces draw + jitter together);
        MODEL-level draws recompile this env's own MjModel from the
        randomized MJCF (the documented per-env memory cost)."""
        if self._rand is not None:
            draw = sample(self._rand, self._rngs[i], self.ndof)
            self._draws[i] = draw
            if self._rand.has_model_params():
                model = self._mujoco.MjModel.from_xml_string(
                    apply_to_mjcf(draw, self._base_xml)
                )
                assert model.nq == self.model.nq  # same tree, edited params only
                self._models[i] = model
                self._data[i] = self._mujoco.MjData(model)
        d, m = self._data[i], self._models[i]
        self._mujoco.mj_resetData(m, d)  # props return to their MJCF spawn pose
        u = self._rngs[i].uniform(-1.0, 1.0, size=self.ndof)
        d.qpos[self._q_sel] = self._mid + self._init_jitter * self._half * u
        d.qvel[self._v_sel] = 0.0
        if self._rand is not None:
            apply_to_env(self._draws[i], self, index=i)  # gains/spawn/camera
        self._mujoco.mj_forward(m, d)  # populate qfrc_bias for the PD
        self._elapsed[i] = 0
        if self._success is not None:
            # Capture this episode's reference (Lifted) from the fresh state.
            self._success[i].reset(state_from_mujoco(m, d))

    def _judge(self, i: int):
        """This env's success predicate on its CURRENT state (props read out of
        the live model/data — see `success.state_from_mujoco`)."""
        return bool(self._success[i](state_from_mujoco(self._models[i], self._data[i])))

    def success_state(self, index: int = 0):
        """The `SuccessState` env `index` is currently in — every prop's world
        center and linear velocity, by name. Available with or without a
        `success` predicate: this is how a caller scores a scene by hand or
        debugs why a predicate is not firing."""
        if not 0 <= index < self.num_envs:
            raise ValueError(f"env index {index} out of range for num_envs={self.num_envs}")
        return state_from_mujoco(self._models[index], self._data[index])

    def _obs(self) -> dict[str, np.ndarray]:
        state = np.empty((self.num_envs, 2 * self.ndof), dtype=np.float32)
        for i, d in enumerate(self._data):
            state[i, : self.ndof] = d.qpos[self._q_sel]
            state[i, self.ndof :] = d.qvel[self._v_sel]
        obs = {"state": state}
        if self._scenes is not None:
            # Full qpos: the scene shares this env's joint layout (see __init__),
            # so props render where they actually are.
            imgs = [
                self._scenes[i].render(np.asarray(d.qpos)) for i, d in enumerate(self._data)
            ]
            obs["image"] = np.stack(imgs).astype(np.uint8, copy=False)
        return obs

    def action_bounds(self) -> np.ndarray:
        """(ndof, 2) qpos-target sampling bounds (URDF limits, ±pi if unbounded)."""
        return self._bounds.copy()

    @property
    def randomization_draws(self) -> list[Optional[dict]]:
        """Per-env current randomization draws (None entries when the env has
        not reset yet or no spec was given) — `reset()` returns only obs, so
        this is how a caller logs the draws it just reset into. Treat the
        dicts as read-only; they are the same objects step() reports in
        `info["randomization"]`."""
        return list(self._draws)

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
    # Offset past the env streams: reset(seed=seed) reseeds env i to
    # default_rng(seed + i), so default_rng(seed) would be byte-identical to
    # env 0's stream — the "random" actions would deterministically replay
    # env 0's reset-jitter/randomization draws.
    rng = np.random.default_rng(seed + env.num_envs)
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
