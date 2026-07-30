"""Policy-in-the-loop bridge: drive a LIVE Studio sim from a trained checkpoint.

Studio (Rust) owns the world — it steps the contact sim, applies targets and
draws. This module owns inference: it is spawned as
``<env-python> -m caliper_learn.cli drive CKPT --urdf URDF`` and speaks a
line-oriented JSON protocol over stdin/stdout, one line per message:

    python -> {"type":"ready","ndof":N,"chunk":C,"policyType":"act","device":"cpu"}
    Studio -> {"type":"obs","tick":T,"t":S,"q":[...],"qd":[...]}
    python -> {"type":"action","tick":T,"q":[...]}
    Studio -> {"type":"stop"}
    python -> {"type":"error","message":"..."}   (then exit nonzero)

STDOUT IS THE WIRE. Nothing but protocol lines may ever reach it, so
`drive_loop` redirects `sys.stdout` to stderr for its whole lifetime and writes
to the file object it was handed *before* the redirect (the lerobot-chatter
lesson: `hub.load_lerobot_policy` already redirects the loader's prints, and a
policy that prints per tick would otherwise corrupt every frame).

CHUNKING. The loop keeps a chunk buffer: `select_actions(state)` may return one
action `(ndof,)` or a whole chunk `(C, ndof)`; one buffered action is consumed
per obs and the policy is re-queried only when the buffer drains — plain
receding-horizon deployment, the cadence `profile.ChunkStats` models. A
`hub.LoadedPolicy` is the C=1 case on the wire because lerobot's own
`select_action` owns the queue internally (it pops one action per call and
re-runs the network every `n_action_steps` ticks — see `hub.LoadedPolicy`), so
`HubPolicySource` returns exactly one action per obs, mirroring
`runner.run_policy` tick for tick. The `chunk` field of the ready line reports
the checkpoint's `n_action_steps` regardless: it is the RE-PLAN PERIOD Studio
wants to display, not a promise about where the buffering happens.

Scope: state-based observations only. Live Studio has no camera stream, so a
checkpoint declaring image features is refused at load, naming the camera keys
it wants (honest scope — same gate as `hub`, worded for this path).
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np

from .hub import _STATE_FEATURE_TYPES


class BridgeError(RuntimeError):
    """A fatal bridge problem. Its message is what Studio receives as
    ``{"type":"error","message":...}`` before the process exits nonzero."""


# ----- messages ---------------------------------------------------------------


def _send(writer, msg: dict) -> None:
    """One protocol line. Compact separators keep per-tick bytes down; the flush
    is mandatory — Studio blocks on the reply."""
    writer.write(json.dumps(msg, separators=(",", ":")) + "\n")
    writer.flush()


def send_error(writer, message: str) -> None:
    """Report a fatal problem on the wire (caller exits nonzero afterwards)."""
    _send(writer, {"type": "error", "message": str(message)})


# ----- limits -----------------------------------------------------------------


def limit_arrays(limits, ndof: int) -> tuple[np.ndarray, np.ndarray]:
    """`(lo, hi)` float arrays from `robot.joint_limits`-shaped input.

    A limitless joint (URDF `continuous`, `None` in `joint_limits`) gets
    ±inf — the Rust SafetyMonitor skips position clamping for those too, so
    clamping them here would invent a constraint the engine does not have.
    """
    if limits is None:
        return np.full(ndof, -np.inf), np.full(ndof, np.inf)
    lo = np.empty(ndof, dtype=np.float64)
    hi = np.empty(ndof, dtype=np.float64)
    if len(limits) != ndof:
        raise BridgeError(f"limits has {len(limits)} entries but the robot has {ndof} dof")
    for i, lim in enumerate(limits):
        if lim is None:
            lo[i], hi[i] = -np.inf, np.inf
        else:
            lo[i], hi[i] = float(lim[0]), float(lim[1])
    return lo, hi


# ----- policy sources ---------------------------------------------------------


class HubPolicySource:
    """A `hub.LoadedPolicy` behind the drive loop's tiny policy interface.

    `select_actions` returns ONE action per observation: lerobot's
    `select_action` pops its internal queue and re-runs the network every
    `n_action_steps` ticks by itself, so buffering on top of it would consume
    chunks at half rate. This mirrors `runner.run_policy` /
    `eval._adapt_policy` exactly, normalization included (the pre/post
    processor pipelines live inside `predict`).
    """

    def __init__(self, policy, ndof: int, device: str = "cpu"):
        self._policy = policy
        self._ndof = int(ndof)
        # Feature shapes are vetted once by `load_drive_policy`, and `predict`
        # re-checks them per call — no per-tick validation needed here.
        self._names = policy.config.state_feature_names
        self.chunk = int(getattr(policy, "n_action_steps", 1) or 1)
        self.policy_type = str(policy.config.type)
        self.device = device

    def reset(self) -> None:
        self._policy.reset()

    def select_actions(self, state) -> np.ndarray:
        q = np.asarray(state[: self._ndof], dtype=np.float32)
        obs = {name: q for name in self._names}
        return np.asarray(self._policy.predict(obs), dtype=np.float64)


def _camera_keys(ckpt_dir) -> list[str]:
    """Image/other non-state input features declared by the checkpoint config,
    read WITHOUT loading any weights (so the refusal is instant and honest)."""
    cfg_file = Path(ckpt_dir) / "config.json"
    if not cfg_file.exists():
        return []
    try:
        raw = json.loads(cfg_file.read_text())
    except (OSError, ValueError):
        return []
    feats = raw.get("input_features") or {}
    if not isinstance(feats, dict):
        return []
    return sorted(
        name
        for name, f in feats.items()
        if isinstance(f, dict) and str(f.get("type")) not in _STATE_FEATURE_TYPES
    )


def load_drive_policy(checkpoint: str, urdf: str, device: str = "cpu"):
    """`(source, ndof, limits)` for `drive`: the checkpoint, vetted against the robot.

    Raises `BridgeError` (never a bare traceback) for every problem Studio must
    be told about in plain English: image-observation checkpoints, ndof
    mismatch, unreadable URDF, missing torch/lerobot.
    """
    try:
        import caliper
    except ImportError as e:  # pragma: no cover - caliper is a hard runtime dep
        raise BridgeError(f"the 'caliper' extension module is not importable: {e}") from e

    try:
        robot = caliper.Robot.from_urdf(urdf)
    except Exception as e:
        raise BridgeError(f"could not load the robot URDF {urdf}: {e}") from e
    ndof = int(robot.ndof)

    cams = _camera_keys(checkpoint)
    if cams:
        raise BridgeError(
            f"this checkpoint needs image observations {cams}, and the live Studio "
            "sim has no camera stream to feed them — drive is state-based only. "
            "Evaluate an image policy offline instead (`caliper-learn eval`, whose "
            "sim camera can render those keys)."
        )

    from .hub import load_lerobot_policy

    try:
        policy = load_lerobot_policy(checkpoint, device=device)
    except Exception as e:
        raise BridgeError(f"could not load the checkpoint {checkpoint}: {e}") from e

    if int(policy.action_dim) != ndof:
        raise BridgeError(
            f"checkpoint action dim {policy.action_dim} != robot dof {ndof} "
            f"({urdf}) — this checkpoint was trained for a different robot"
        )
    for name in policy.config.state_feature_names:
        (_, shape) = policy.config.input_features[name]
        if tuple(shape) != (ndof,):
            raise BridgeError(
                f"checkpoint feature '{name}' expects shape {tuple(shape)} but the "
                f"robot has {ndof} dof — this checkpoint was trained for a different robot"
            )

    return HubPolicySource(policy, ndof, device=device), ndof, list(robot.joint_limits)


# ----- the loop ---------------------------------------------------------------


def drive_loop(policy, reader, writer, ndof: int, limits=None) -> int:
    """Serve `policy` over the drive protocol until stop/EOF. Returns an exit code.

    - `policy`: anything with `select_actions(state) -> (ndof,) | (C, ndof)`;
      `reset()` is called once before the first obs when present. `state` is the
      `(2*ndof,)` float32 `[qpos, qvel]` vector — the same observation shape
      `eval._adapt_policy` hands to callable policies.
    - `reader` / `writer`: the protocol streams (`.readline()` / `.write()`).
    - `limits`: `robot.joint_limits` (per-dof `(lo, hi)` or `None`); commanded
      targets are clamped into them, since the engine would clamp anyway and a
      silently-clamped target is the P003 failure Studio must not mistake for
      the policy's intent. A one-line session summary (ticks served, policy
      queries, clamped ticks) goes to STDERR at exit — never onto the wire.

    Every reply echoes the tick of the obs it answers, in order: one action per
    obs, no skipping. Returns 0 on a clean stop, 1 after a fatal error (which is
    reported on the wire first).
    """
    ndof = int(ndof)
    lo, hi = limit_arrays(limits, ndof)
    buffer: list[np.ndarray] = []
    clamped_ticks = 0
    queries = 0
    served = 0

    # Everything from here on treats `writer` as the only sink: library prints
    # (a chatty policy, torch warnings) go to stderr, never onto the wire.
    with contextlib.redirect_stdout(sys.stderr):
        if hasattr(policy, "reset"):
            policy.reset()
        _send(
            writer,
            {
                "type": "ready",
                "ndof": ndof,
                "chunk": int(getattr(policy, "chunk", 1) or 1),
                "policyType": str(getattr(policy, "policy_type", type(policy).__name__)),
                "device": str(getattr(policy, "device", "cpu")),
            },
        )

        while True:
            line = reader.readline()
            if not line:
                break  # Studio closed the pipe: same as stop
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError as e:
                send_error(writer, f"malformed protocol line ({e}): {line[:200]!r}")
                return 1
            if not isinstance(msg, dict):
                send_error(writer, f"protocol message is not an object: {line[:200]!r}")
                return 1

            kind = msg.get("type")
            if kind == "stop":
                break
            if kind != "obs":
                continue  # forward compatibility: unknown message types are ignored

            tick = msg.get("tick", 0)
            try:
                state, err = _state_from_obs(msg, ndof)
            except Exception as e:  # pragma: no cover - defensive
                send_error(writer, f"bad obs at tick {tick}: {e}")
                return 1
            if err is not None:
                send_error(writer, f"bad obs at tick {tick}: {err}")
                return 1

            if not buffer:
                try:
                    chunk = np.asarray(policy.select_actions(state), dtype=np.float64)
                except Exception as e:
                    send_error(writer, f"policy failed at tick {tick}: {type(e).__name__}: {e}")
                    return 1
                queries += 1
                if chunk.ndim == 1:
                    chunk = chunk[None, :]
                if chunk.ndim != 2 or chunk.shape[1] != ndof or chunk.shape[0] < 1:
                    send_error(
                        writer,
                        f"policy returned action shape {tuple(chunk.shape)} at tick {tick}; "
                        f"expected ({ndof},) or (C, {ndof})",
                    )
                    return 1
                buffer = list(chunk)

            action = buffer.pop(0)
            if not np.all(np.isfinite(action)):
                send_error(
                    writer,
                    f"policy emitted a non-finite action at tick {tick}: "
                    f"{np.array2string(action, precision=4)}",
                )
                return 1
            target = np.clip(action, lo, hi)
            if not np.array_equal(target, action):
                clamped_ticks += 1
            _send(writer, {"type": "action", "tick": tick, "q": [float(v) for v in target]})
            served += 1

    # One line, at exit, on stderr (Studio's log pane) — the wire stays clean.
    summary = f"caliper-learn drive: served {served} tick(s) from {queries} policy quer"
    summary += "y" if queries == 1 else "ies"
    if clamped_ticks:
        summary += (
            f"; clamped the commanded target into the joint limits on {clamped_ticks} "
            "tick(s) — the policy is asking for motion the robot cannot make "
            "(see `caliper-learn debug`, P003)"
        )
    print(summary, file=sys.stderr)
    return 0


def _state_from_obs(msg: dict, ndof: int) -> tuple[np.ndarray, str | None]:
    """`[q, qd]` (2*ndof,) float32 from an obs message, or `(_, reason)`.

    `qd` is optional (defaults to zeros) — Studio may not have velocities for a
    kinematic-only session, and state-feature checkpoints read only `q`.
    """
    q = msg.get("q")
    if not isinstance(q, list):
        return np.zeros(2 * ndof, dtype=np.float32), "missing 'q' array"
    if len(q) != ndof:
        return np.zeros(2 * ndof, dtype=np.float32), f"q has {len(q)} values, robot has {ndof} dof"
    qd = msg.get("qd")
    if qd is None:
        qd = [0.0] * ndof
    if not isinstance(qd, list) or len(qd) != ndof:
        return (
            np.zeros(2 * ndof, dtype=np.float32),
            f"qd has {0 if not isinstance(qd, list) else len(qd)} values, robot has {ndof} dof",
        )
    try:
        state = np.asarray(list(q) + list(qd), dtype=np.float32)
    except (TypeError, ValueError) as e:
        return np.zeros(2 * ndof, dtype=np.float32), f"non-numeric values ({e})"
    if not np.all(np.isfinite(state)):
        return state, "q/qd contain non-finite values"
    return state, None


def drive_main(
    checkpoint: str,
    urdf: str,
    *,
    device: str = "cpu",
    reader=None,
    writer=None,
) -> int:
    """CLI adapter: load, then serve. Any load failure is reported on the wire.

    The writer is captured BEFORE any loading happens, so the loader's own
    chatter (redirected to stderr inside `hub`) can never race the wire.
    """
    reader = sys.stdin if reader is None else reader
    writer = sys.stdout if writer is None else writer
    try:
        with contextlib.redirect_stdout(sys.stderr):
            source, ndof, limits = load_drive_policy(checkpoint, urdf, device=device)
    except BridgeError as e:
        send_error(writer, str(e))
        return 1
    except Exception as e:  # never let a traceback be the only thing Studio sees
        send_error(writer, f"{type(e).__name__}: {e}")
        return 1
    return drive_loop(source, reader, writer, ndof, limits)
