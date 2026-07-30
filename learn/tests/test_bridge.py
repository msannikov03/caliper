"""The policy-in-the-loop bridge: protocol shape, chunk cadence, stdout purity.

The scripted-policy tests need neither torch nor lerobot — `drive_loop` is a
pure protocol machine over anything with `select_actions`. The last test drives
a REAL tiny ACT checkpoint end to end (same recipe as test_hub_deploy) to prove
the hub adapter feeds normalization and the chunk queue correctly.
"""

import io
import json
import sys

import numpy as np
import pytest

import caliper  # conftest importorskips this for the whole directory  # noqa: F401
from caliper_learn.bridge import BridgeError, drive_loop, limit_arrays

DOF = 6
URDF = "oracle/fixtures/robots/showcase6.urdf"


# ----- scripted policies ------------------------------------------------------


class ScriptedPolicy:
    """Emits `chunk` rows per query, each row = base + query index, so the tick
    a value came from is readable in the output. Counts queries and resets."""

    def __init__(self, ndof=3, chunk=1, policy_type="scripted", device="cpu"):
        self.ndof = ndof
        self.chunk = chunk
        self.policy_type = policy_type
        self.device = device
        self.queries = 0
        self.resets = 0
        self.seen = []

    def reset(self):
        self.resets += 1

    def select_actions(self, state):
        self.seen.append(np.asarray(state, dtype=np.float64).copy())
        out = np.full((self.chunk, self.ndof), float(self.queries), dtype=np.float64)
        out += np.arange(self.chunk, dtype=np.float64)[:, None] * 0.01
        self.queries += 1
        if self.chunk == 1:
            return out[0]  # the self-buffered shape (hub.LoadedPolicy's)
        return out


class ChattyPolicy(ScriptedPolicy):
    """A policy that prints on every query — the lerobot-stdout-poisoning case."""

    def select_actions(self, state):
        print("chatty: about to infer")
        sys.stdout.write("more chatter without a newline")
        return super().select_actions(state)


def _obs(tick, q, qd=None):
    msg = {"type": "obs", "tick": tick, "t": tick / 50.0, "q": [float(v) for v in q]}
    if qd is not None:
        msg["qd"] = [float(v) for v in qd]
    return json.dumps(msg) + "\n"


def _run(policy, lines, ndof=3, limits=None):
    """Drive `policy` over a canned script; return (rc, parsed messages, raw stdout)."""
    reader = io.StringIO("".join(lines))
    writer = io.StringIO()
    rc = drive_loop(policy, reader, writer, ndof, limits)
    raw = writer.getvalue()
    msgs = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return rc, msgs, raw


# ----- ready / basic exchange -------------------------------------------------


def test_ready_line_then_one_action_per_obs():
    p = ScriptedPolicy(ndof=3, chunk=1, policy_type="act", device="cpu")
    ticks = [7, 8, 9]
    rc, msgs, _ = _run(p, [_obs(t, [0.1 * t] * 3) for t in ticks] + ['{"type":"stop"}\n'])
    assert rc == 0
    assert msgs[0] == {
        "type": "ready",
        "ndof": 3,
        "chunk": 1,
        "policyType": "act",
        "device": "cpu",
    }
    assert p.resets == 1, "reset() must be called once before serving"
    actions = msgs[1:]
    assert [m["type"] for m in actions] == ["action"] * 3
    assert [m["tick"] for m in actions] == ticks, "replies must echo obs ticks in order"
    assert all(len(m["q"]) == 3 for m in actions)


def test_qd_is_appended_to_the_state_vector():
    """State handed to the policy is [q, qd] (2*ndof,) — eval's convention."""
    p = ScriptedPolicy(ndof=3)
    q, qd = [0.1, 0.2, 0.3], [-1.0, 0.0, 1.0]
    rc, _, _ = _run(p, [_obs(0, q, qd), '{"type":"stop"}\n'])
    assert rc == 0
    assert np.allclose(p.seen[0], q + qd)


def test_missing_qd_defaults_to_zeros():
    p = ScriptedPolicy(ndof=3)
    rc, _, _ = _run(p, [_obs(0, [0.5, 0.5, 0.5]), '{"type":"stop"}\n'])
    assert rc == 0
    assert np.allclose(p.seen[0], [0.5, 0.5, 0.5, 0.0, 0.0, 0.0])


def test_eof_without_stop_exits_cleanly():
    """Studio dying closes the pipe; that is a stop, not an error."""
    p = ScriptedPolicy(ndof=3)
    rc, msgs, _ = _run(p, [_obs(0, [0.0] * 3)])
    assert rc == 0
    assert msgs[-1]["type"] == "action"


def test_stop_before_any_obs():
    p = ScriptedPolicy(ndof=3)
    rc, msgs, _ = _run(p, ['{"type":"stop"}\n'])
    assert rc == 0 and len(msgs) == 1 and msgs[0]["type"] == "ready"
    assert p.queries == 0


def test_unknown_message_types_are_ignored():
    """Forward compatibility: Studio may add message kinds we do not know."""
    p = ScriptedPolicy(ndof=3)
    rc, msgs, _ = _run(
        p,
        ['{"type":"hello","v":2}\n', "\n", _obs(1, [0.0] * 3), '{"type":"stop"}\n'],
    )
    assert rc == 0
    assert [m["type"] for m in msgs] == ["ready", "action"]


# ----- chunking ---------------------------------------------------------------


def test_chunk_of_four_is_consumed_before_requerying():
    """9 obs with chunk=4 -> queries at obs 0, 4, 8 (three network calls), and the
    values prove which query each action came from."""
    p = ScriptedPolicy(ndof=3, chunk=4)
    rc, msgs, _ = _run(
        p, [_obs(t, [0.0] * 3) for t in range(9)] + ['{"type":"stop"}\n']
    )
    assert rc == 0
    assert msgs[0]["chunk"] == 4
    assert p.queries == 3, f"expected 3 re-plans over 9 ticks, got {p.queries}"
    actions = msgs[1:]
    assert len(actions) == 9
    # query index = floor(tick/4), position in chunk = tick % 4
    for t, m in enumerate(actions):
        expected = float(t // 4) + 0.01 * (t % 4)
        assert m["q"] == pytest.approx([expected] * 3), f"tick {t} served the wrong chunk row"


def test_chunk_boundary_is_exact_at_the_last_element():
    """Exactly 4 obs with chunk=4 must cost exactly one query (no early refill)."""
    p = ScriptedPolicy(ndof=3, chunk=4)
    rc, _, _ = _run(p, [_obs(t, [0.0] * 3) for t in range(4)] + ['{"type":"stop"}\n'])
    assert rc == 0 and p.queries == 1
    p2 = ScriptedPolicy(ndof=3, chunk=4)
    _run(p2, [_obs(t, [0.0] * 3) for t in range(5)] + ['{"type":"stop"}\n'])
    assert p2.queries == 2, "the 5th obs must trigger the re-query"


def test_chunked_policy_sees_the_obs_of_the_refill_tick():
    """Receding horizon: the re-query happens on the tick that drained the buffer,
    with THAT tick's state — not a stale one."""
    p = ScriptedPolicy(ndof=3, chunk=2)
    lines = [_obs(t, [float(t)] * 3) for t in range(4)] + ['{"type":"stop"}\n']
    _run(p, lines)
    assert len(p.seen) == 2
    assert p.seen[0][:3] == pytest.approx([0.0] * 3)
    assert p.seen[1][:3] == pytest.approx([2.0] * 3)


# ----- stdout purity ----------------------------------------------------------


def test_stdout_stays_pure_with_a_chatty_policy(capsys):
    """Nothing but protocol lines on the wire, even when the policy prints."""
    p = ChattyPolicy(ndof=3, chunk=1)
    rc, msgs, raw = _run(p, [_obs(t, [0.0] * 3) for t in range(3)] + ['{"type":"stop"}\n'])
    assert rc == 0
    for line in raw.splitlines():
        parsed = json.loads(line)  # every line parses — no chatter interleaved
        assert parsed["type"] in {"ready", "action"}
    assert len(msgs) == 4
    captured = capsys.readouterr()
    assert "chatty" in captured.err, "policy chatter must be routed to stderr"
    assert "chatty" not in captured.out


def test_stdout_is_restored_after_the_loop(capsys):
    p = ScriptedPolicy(ndof=3)
    _run(p, ['{"type":"stop"}\n'])
    print("back to normal")
    assert "back to normal" in capsys.readouterr().out


# ----- error paths ------------------------------------------------------------


def test_ndof_mismatch_in_obs_is_a_fatal_error():
    p = ScriptedPolicy(ndof=3)
    rc, msgs, _ = _run(p, [_obs(3, [0.0, 0.0])])
    assert rc == 1
    assert msgs[-1]["type"] == "error"
    assert "2 values" in msgs[-1]["message"] and "3 dof" in msgs[-1]["message"]
    assert p.queries == 0


def test_non_finite_action_is_an_error_not_a_silent_nan():
    class NaNPolicy(ScriptedPolicy):
        def select_actions(self, state):
            return np.array([0.0, float("nan"), 0.0])

    rc, msgs, _ = _run(NaNPolicy(ndof=3), [_obs(11, [0.0] * 3)])
    assert rc == 1
    assert msgs[-1]["type"] == "error" and "non-finite" in msgs[-1]["message"]
    assert "11" in msgs[-1]["message"]


def test_wrong_action_width_is_an_error():
    class WidePolicy(ScriptedPolicy):
        def select_actions(self, state):
            return np.zeros(5)

    rc, msgs, _ = _run(WidePolicy(ndof=3), [_obs(0, [0.0] * 3)])
    assert rc == 1
    assert "shape" in msgs[-1]["message"] and msgs[-1]["type"] == "error"


def test_policy_exception_is_reported_on_the_wire():
    class BoomPolicy(ScriptedPolicy):
        def select_actions(self, state):
            raise RuntimeError("the model exploded")

    rc, msgs, _ = _run(BoomPolicy(ndof=3), [_obs(2, [0.0] * 3)])
    assert rc == 1
    assert msgs[-1]["type"] == "error"
    assert "the model exploded" in msgs[-1]["message"]


def test_malformed_line_is_fatal():
    rc, msgs, _ = _run(ScriptedPolicy(ndof=3), ["{not json\n"])
    assert rc == 1
    assert msgs[-1]["type"] == "error" and "malformed" in msgs[-1]["message"]


def test_non_finite_obs_is_rejected():
    rc, msgs, _ = _run(ScriptedPolicy(ndof=3), ['{"type":"obs","tick":0,"q":[0,null,0]}\n'])
    assert rc == 1
    assert msgs[-1]["type"] == "error"


# ----- limits -----------------------------------------------------------------


def test_actions_are_clamped_into_the_joint_limits():
    class BigPolicy(ScriptedPolicy):
        def select_actions(self, state):
            return np.array([5.0, -5.0, 0.25])

    limits = [(-1.0, 1.0), (-1.0, 1.0), None]  # third joint is limitless
    rc, msgs, _ = _run(BigPolicy(ndof=3), [_obs(0, [0.0] * 3)], limits=limits)
    assert rc == 0
    assert msgs[-1]["q"] == pytest.approx([1.0, -1.0, 0.25])


def test_clamping_is_reported_on_stderr_not_on_the_wire(capsys):
    class BigPolicy(ScriptedPolicy):
        def select_actions(self, state):
            return np.array([5.0, 0.0, 0.0])

    rc, msgs, _ = _run(
        BigPolicy(ndof=3),
        [_obs(t, [0.0] * 3) for t in range(2)] + ['{"type":"stop"}\n'],
        limits=[(-1.0, 1.0)] * 3,
    )
    assert rc == 0
    assert {m["type"] for m in msgs} == {"ready", "action"}
    err = capsys.readouterr().err
    assert "clamped" in err and "2 tick(s)" in err
    assert "served 2 tick(s) from 2 policy queries" in err


def test_limitless_joints_are_never_clamped():
    class BigPolicy(ScriptedPolicy):
        def select_actions(self, state):
            return np.array([99.0])

    rc, msgs, _ = _run(BigPolicy(ndof=1), [_obs(0, [0.0])], ndof=1, limits=[None])
    assert rc == 0 and msgs[-1]["q"] == pytest.approx([99.0])


def test_limit_arrays_length_mismatch_raises():
    with pytest.raises(BridgeError, match="2 entries"):
        limit_arrays([(-1.0, 1.0), (-1.0, 1.0)], 3)


def test_limit_arrays_none_is_unbounded():
    lo, hi = limit_arrays(None, 2)
    assert np.all(np.isneginf(lo)) and np.all(np.isposinf(hi))


# ----- the real checkpoint ----------------------------------------------------


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from test_hub_deploy import _make_tiny_checkpoint

    return _make_tiny_checkpoint(tmp_path_factory.mktemp("drive_ckpt"), dof=DOF)


def test_real_checkpoint_drives_ten_ticks(tiny_ckpt):
    """End to end through the hub adapter: ready line reports the checkpoint's
    re-plan period, and every commanded target is finite and inside the limits."""
    from caliper_learn.bridge import load_drive_policy

    source, ndof, limits = load_drive_policy(str(tiny_ckpt), URDF)
    assert ndof == DOF
    assert source.chunk == 4 and source.policy_type == "act"

    lines = [_obs(t, list(np.linspace(-0.3, 0.3, DOF) * (t + 1) / 10)) for t in range(10)]
    rc, msgs, _ = _run(source, lines + ['{"type":"stop"}\n'], ndof=DOF, limits=limits)
    assert rc == 0
    assert msgs[0] == {
        "type": "ready",
        "ndof": DOF,
        "chunk": 4,
        "policyType": "act",
        "device": "cpu",
    }
    actions = msgs[1:]
    assert len(actions) == 10
    assert [m["tick"] for m in actions] == list(range(10))
    lo, hi = limit_arrays(limits, DOF)
    for m in actions:
        a = np.asarray(m["q"], dtype=np.float64)
        assert a.shape == (DOF,) and np.all(np.isfinite(a))
        assert np.all(a >= lo - 1e-9) and np.all(a <= hi + 1e-9)


def test_hub_adapter_matches_run_policy_tick_for_tick(tiny_ckpt):
    """The bridge must be the SAME deployment as `runner.run_policy`: identical
    actions for identical observations, chunk-queue refills included."""
    from caliper_learn.bridge import HubPolicySource
    from caliper_learn.hub import load_lerobot_policy

    qs = [(np.linspace(-0.3, 0.3, DOF) * (k + 1) / 10).astype(np.float32) for k in range(10)]

    ref = load_lerobot_policy(str(tiny_ckpt))
    ref.reset()
    expected = [
        ref.predict({"observation.state": q, "observation.environment_state": q}) for q in qs
    ]

    source = HubPolicySource(load_lerobot_policy(str(tiny_ckpt)), DOF)
    rc, msgs, _ = _run(
        source,
        [_obs(k, list(q)) for k, q in enumerate(qs)] + ['{"type":"stop"}\n'],
        ndof=DOF,
    )
    assert rc == 0
    for k, (m, e) in enumerate(zip(msgs[1:], expected)):
        assert np.allclose(m["q"], e, atol=1e-6), f"tick {k} diverged from run_policy"


def test_image_checkpoint_is_refused_naming_the_camera_keys(tiny_ckpt, tmp_path):
    """Live Studio has no camera stream — say so, and name what was wanted."""
    import shutil

    from caliper_learn.bridge import load_drive_policy

    for f in tiny_ckpt.iterdir():
        shutil.copy(f, tmp_path / f.name)
    cj = json.loads((tmp_path / "config.json").read_text())
    cj["input_features"]["observation.images.top"] = {"type": "VISUAL", "shape": [3, 96, 96]}
    (tmp_path / "config.json").write_text(json.dumps(cj))
    with pytest.raises(BridgeError, match="observation.images.top"):
        load_drive_policy(str(tmp_path), URDF)


def test_wrong_robot_is_refused_at_load(tiny_ckpt):
    """A 6-dof checkpoint against a 3-dof arm: caught before the first tick."""
    from caliper_learn.bridge import load_drive_policy

    with pytest.raises(BridgeError, match="different robot"):
        load_drive_policy(str(tiny_ckpt), "oracle/fixtures/robots/collide_arm.urdf")


def test_drive_main_reports_load_failure_on_the_wire(tmp_path):
    """A bad checkpoint is an {"type":"error"} line + nonzero exit, not a traceback."""
    from caliper_learn.bridge import drive_main

    writer = io.StringIO()
    rc = drive_main(
        str(tmp_path / "nope"), URDF, reader=io.StringIO(""), writer=writer
    )
    assert rc == 1
    msgs = [json.loads(x) for x in writer.getvalue().splitlines() if x.strip()]
    assert len(msgs) == 1 and msgs[0]["type"] == "error"


def test_cli_drive_end_to_end(monkeypatch, capsys, tiny_ckpt):
    """`caliper-learn drive CKPT --urdf URDF` over piped stdin."""
    from caliper_learn.cli import main

    monkeypatch.setattr(sys, "stdin", io.StringIO(_obs(0, [0.0] * DOF) + '{"type":"stop"}\n'))
    rc = main(["drive", str(tiny_ckpt), "--urdf", URDF])
    assert rc == 0
    out = capsys.readouterr().out
    msgs = [json.loads(x) for x in out.splitlines() if x.strip()]
    assert [m["type"] for m in msgs] == ["ready", "action"]


def test_cli_drive_needs_exactly_one_robot_source(tiny_ckpt):
    from caliper_learn.cli import main

    with pytest.raises(SystemExit, match="exactly one robot source"):
        main(["drive", str(tiny_ckpt)])
