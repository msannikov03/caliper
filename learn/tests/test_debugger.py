"""Hermetic debugger unit tests (fake policy/robot, no torch/lerobot on the
path — the full checkpoint-level positives/negatives live in test_autopsy.py):

- P003 must be SUPPRESSED for limitless joints: `_bounds` fabricates ±pi for
  them (sampling convenience), but the SafetyMonitor skips position clamping
  when a joint has no limits — "clamped every tick" would be a false diagnosis.
- P004's action-stats lookup must fall back per-KEY: a postprocessor stats file
  that exists but carries no 'action' entry used to skip the comparison
  entirely (`(post_stats or pre_stats).get("action")`).
"""

from types import SimpleNamespace

import numpy as np

from caliper_learn.debugger import (
    ACTION_COLLAPSE,
    ACTION_SATURATION,
    NORMALIZATION_MISMATCH,
    _behavioral_checks,
    _check_normalization,
    _DatasetView,
)


class _FakeRobot:
    """Only what `_bounds` and the P003 gate read: dof 0 limited, dof 1 not."""

    joint_limits = [(-1.0, 1.0), None]


class _ConstPolicy:
    """Always returns the same far-out-of-range action for both dofs."""

    config = SimpleNamespace(
        state_feature_names=["observation.state"],
        input_features={"observation.state": (None, (2,))},
    )

    def reset(self):
        pass

    def predict(self, obs):
        return np.array([5.0, 5.0])


def test_p003_suppressed_for_limitless_joints():
    """Regression: dof 1 has NO URDF limits — safety.rs skips clamping for it,
    yet P003 fired against the fabricated ±pi sampling bounds and claimed the
    SafetyMonitor would clamp it every tick. Only the genuinely limited dof 0
    may be reported (its 5.0 constant sits outside [-1, 1] on 100% of probes)."""
    findings = _behavioral_checks(_ConstPolicy(), None, _FakeRobot(), {}, {})
    p003_dofs = sorted(f.dof for f in findings if f.code == ACTION_SATURATION)
    assert p003_dofs == [0]
    # sanity: the collapse check still sees the constant output
    assert any(f.code == ACTION_COLLAPSE for f in findings)


def test_p004_falls_back_to_pre_stats_when_post_lacks_action():
    """Regression: with a non-empty post_stats dict lacking 'action',
    `(post_stats or pre_stats).get("action")` returned None and the action
    normalization check silently skipped — a ~10-data-std mean shift stored in
    the preprocessor went unreported."""
    rng = np.random.default_rng(0)
    ds = _DatasetView(
        states=rng.normal(size=(64, 2)), actions=rng.normal(size=(64, 2)), fps=50
    )
    cfg = SimpleNamespace(state_feature_names=[])
    pre = {"action": {"mean": ds.actions.mean(0) + 10.0, "std": ds.actions.std(0)}}
    post = {"observation.state": {"mean": np.zeros(2), "std": np.ones(2)}}  # no 'action'
    findings = _check_normalization(cfg, pre, post, ds)
    assert [f.code for f in findings] == [NORMALIZATION_MISMATCH]
    assert findings[0].feature == "action" and findings[0].severity == "error"
