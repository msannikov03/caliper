"""Latency profiler oracles — hermetic (fake policy + fake control loop, no
caliper/torch on the path), timings vary so tests assert STRUCTURE exactly and
magnitudes only where the fake's deliberate sleeps guarantee them:

- positive: a chunky policy (sleep every 5th call) -> refill p95 >> pop p95,
  bimodal detection recovers period 5, L001 fires at an fps the sleep cannot
  sustain; L002 fires when inference is the whole tick; L003 fires on an
  alternating-latency step.
- negative: a fast callable at an easy fps -> ZERO findings, no chunk detected.
- shape: to_json round-trips with the documented keys; structure (keys, finding
  codes forced by the fakes, chunk period) is exactly equal across runs.
"""

import json
import time

import numpy as np

from caliper_learn.profile import (
    BUDGET_EXCEEDED,
    HIGH_JITTER,
    INFERENCE_DOMINATES,
    profile_rollout,
)


class FakeLoop:
    """Stands in for caliper.ControlLoop: .q + .step_with_target, optional
    per-tick sleep in the step stage (tick index -> seconds)."""

    def __init__(self, ndof=3, step_sleep=None):
        self._q = [0.0] * ndof
        self._step_sleep = step_sleep
        self._k = 0

    @property
    def q(self):
        return list(self._q)

    def step_with_target(self, action):
        if self._step_sleep is not None:
            time.sleep(self._step_sleep(self._k))
        self._k += 1
        self._q = list(action)
        return self.q


def make_chunky_callable(period=5, sleep_s=0.005, ndof=3):
    """A plain callable that 'refills' (sleeps) on every `period`-th call and
    pops (returns instantly) otherwise — the ACT select_action timing shape."""
    calls = {"n": 0}

    def predict(obs):
        if calls["n"] % period == 0:
            time.sleep(sleep_s)
        calls["n"] += 1
        return np.zeros(ndof, dtype=np.float32)

    return predict


class ChunkyPolicy:
    """Same timing shape, but declares its chunk config like hub.LoadedPolicy."""

    n_action_steps = 5

    def __init__(self, sleep_s=0.005, ndof=3):
        self._sleep_s = sleep_s
        self._ndof = ndof
        self._k = 0

    def reset(self):
        self._k = 0

    def predict(self, obs):
        if self._k % self.n_action_steps == 0:
            time.sleep(self._sleep_s)
        self._k += 1
        return np.zeros(self._ndof, dtype=np.float32)


# ---- positive: refill spike + L001 -----------------------------------------


def test_refill_spike_and_l001_at_unsustainable_fps():
    """5 ms sleep on every 5th tick vs a 1 ms budget: bimodal detection must
    find period 5, the refill p95 must dwarf the pop p95, and L001 must fire."""
    report = profile_rollout(
        make_chunky_callable(period=5, sleep_s=0.005), FakeLoop(), ticks=25, fps=1000
    )

    assert report.chunk is not None
    assert report.chunk.source == "bimodal"
    assert report.chunk.period == 5
    assert report.chunk.n_refill_ticks == 5
    assert report.chunk.pop is not None
    assert report.chunk.refill.p95 > 10 * report.chunk.pop.p95

    codes = [f.code for f in report.findings]
    assert BUDGET_EXCEEDED in codes
    assert report.frac_over_budget > 0.05
    assert report.achievable_hz < 1000  # honest headline: p95, not the mean
    l001 = next(f for f in report.findings if f.code == BUDGET_EXCEEDED)
    assert l001.severity == "error"
    assert "Hz" in l001.message and l001.fix_hint  # says by how much + what to do

    text = report.render_text()
    for needle in ("obs_build", "inference", "step", "total", BUDGET_EXCEEDED, "refill"):
        assert needle in text


def test_chunk_config_beats_bimodality():
    """A policy declaring n_action_steps=5 is split by config, not by guessing;
    also exercises the zero-arg control-loop factory path."""
    report = profile_rollout(ChunkyPolicy(sleep_s=0.005), lambda: FakeLoop(), ticks=25, fps=1000)
    assert report.chunk is not None
    assert report.chunk.source == "config"
    assert report.chunk.period == 5
    assert report.chunk.n_refill_ticks == 5
    assert report.chunk.refill.p95 > 10 * report.chunk.pop.p95


# ---- positive: L002 / L003 --------------------------------------------------


def test_l002_inference_dominates():
    """3 ms inference on EVERY tick within an 8 ms budget: no budget violation,
    but the report must say inference is the bottleneck and quantify it."""

    def slow_every_tick(obs):
        time.sleep(0.003)
        return np.zeros(3, dtype=np.float32)

    report = profile_rollout(slow_every_tick, FakeLoop(), ticks=40, fps=125)
    codes = [f.code for f in report.findings]
    assert INFERENCE_DOMINATES in codes
    l002 = next(f for f in report.findings if f.code == INFERENCE_DOMINATES)
    assert "%" in l002.message  # quantified share of the tick
    assert l002.fix_hint


def test_l003_high_jitter():
    """Step latency alternating 0 / 12 ms inside a 20 ms budget: every tick fits,
    but the cadence is unstable — L003 must fire."""
    loop = FakeLoop(step_sleep=lambda k: 0.012 if k % 2 else 0.0)
    report = profile_rollout(
        lambda obs: np.zeros(3, dtype=np.float32), loop, ticks=30, fps=50
    )
    codes = [f.code for f in report.findings]
    assert HIGH_JITTER in codes
    l003 = next(f for f in report.findings if f.code == HIGH_JITTER)
    assert l003.severity == "warn"
    assert l003.fix_hint


# ---- negative: clean input, zero findings -----------------------------------


def test_fast_callable_no_findings():
    """A microsecond-fast callable at 10 Hz: zero findings, zero over-budget
    ticks, and no phantom chunking from scheduler noise (absolute-floor guard)."""
    report = profile_rollout(
        lambda obs: np.zeros(3, dtype=np.float32), FakeLoop(), ticks=50, fps=10
    )
    assert report.findings == []
    assert report.frac_over_budget == 0.0
    assert report.chunk is None
    assert report.achievable_hz > 10
    assert "no findings" in report.render_text()


# ---- shape + structural determinism ------------------------------------------


def test_json_shape():
    report = profile_rollout(
        make_chunky_callable(period=5, sleep_s=0.005), FakeLoop(), ticks=25, fps=1000
    )
    d = json.loads(report.to_json())
    assert set(d) == {
        "fps",
        "ticks",
        "budget_s",
        "achievable_hz",
        "jitter_s",
        "frac_over_budget",
        "overhead_s",
        "stages",
        "chunk",
        "findings",
    }
    assert set(d["stages"]) == {"obs_build", "inference", "step", "total"}
    for s in d["stages"].values():
        assert set(s) == {"p50", "p95", "p99", "max"}
    assert set(d["chunk"]) == {"source", "period", "n_refill_ticks", "refill", "pop"}
    for f in d["findings"]:
        assert set(f) == {"code", "severity", "message", "fix_hint"}
    assert d["fps"] == 1000 and d["ticks"] == 25 and d["budget_s"] == 1e-3


def test_structure_is_deterministic_across_runs():
    """Timings vary run to run; STRUCTURE must not: same keys, same chunk
    identification, and the sleep-forced L001 present both times (exact equality
    on everything the fakes pin down)."""

    def run():
        return profile_rollout(
            make_chunky_callable(period=5, sleep_s=0.005), FakeLoop(), ticks=25, fps=1000
        )

    a, b = run(), run()
    da, db = a.to_dict(), b.to_dict()
    assert set(da) == set(db)
    assert set(da["stages"]) == set(db["stages"])
    assert (da["chunk"]["source"], da["chunk"]["period"], da["chunk"]["n_refill_ticks"]) == (
        db["chunk"]["source"],
        db["chunk"]["period"],
        db["chunk"]["n_refill_ticks"],
    )
    assert BUDGET_EXCEEDED in [f["code"] for f in da["findings"]]
    assert BUDGET_EXCEEDED in [f["code"] for f in db["findings"]]
    # findings are sorted (severity rank, code) — the order is itself structure
    assert [f["code"] for f in da["findings"]] == sorted(
        [f["code"] for f in da["findings"]],
        key=lambda c: ({"L001": 0, "L003": 1, "L002": 2}[c], c),
    )
