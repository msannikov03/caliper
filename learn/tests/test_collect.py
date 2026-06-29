import json

import numpy as np
import pytest

caliper = pytest.importorskip("caliper")
from caliper_learn.collect import collect_demos


def test_schema_and_nonconstant(tmp_path):
    root = collect_demos(str(tmp_path / "ds"), n_episodes=3, seed0=0, fps=50)
    rd = caliper.DatasetReader.open(root)
    assert rd.total_episodes == 3 and rd.ndof == 3 and rd.fps == 50
    s, a, t = rd.read_episode(0)
    assert len(s) == len(a) == len(t) > 0
    assert all(len(r) == 3 for r in s) and all(len(r) == 3 for r in a)
    assert t == sorted(t)
    # BC-learnability guard: actions must NOT be constant (one-step lookahead)
    assert np.asarray(a).std() > 1e-4
    # v2.1 schema sanity
    info = json.loads((root and __import__("pathlib").Path(root) / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1" and info["total_episodes"] == 3


def test_deterministic_reruns(tmp_path):
    r1 = collect_demos(str(tmp_path / "a"), n_episodes=3, seed0=0)
    r2 = collect_demos(str(tmp_path / "b"), n_episodes=3, seed0=0)
    rd1, rd2 = caliper.DatasetReader.open(r1), caliper.DatasetReader.open(r2)
    assert rd1.total_episodes == rd2.total_episodes == 3
    # byte-identical across ALL episodes and ALL arrays (states, actions, timestamps)
    for ep in range(3):
        e1, e2 = rd1.read_episode(ep), rd2.read_episode(ep)
        for arr1, arr2 in zip(e1, e2):
            assert arr1 == arr2, f"episode {ep} differs across reruns"
