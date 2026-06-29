import numpy as np
import pytest

pytest.importorskip("caliper")
torch = pytest.importorskip("torch")

from caliper_learn.data import DataConfig, make_datasets, split_episodes
from caliper_learn.policy import build_policy


def test_frame_dataset_shapes(tiny_dataset):
    train, _, stats, meta = make_datasets(DataConfig(root=tiny_dataset, mode="frame", goal_conditioned=True))
    assert meta["obs_dim"] == 2 * meta["ndof"] and meta["action_dim"] == meta["ndof"]
    o, a = train[0]
    assert o.shape == (meta["obs_dim"],) and a.shape == (meta["action_dim"],)
    assert o.dtype == torch.float32
    # norm stats well-formed (no zero std)
    assert (stats.in_std > 0).all() and (stats.out_std > 0).all()


def test_split_no_leakage():
    tr, va = split_episodes(10, 0.2, 0)
    assert set(tr).isdisjoint(set(va))
    assert sorted(tr + va) == list(range(10))


def test_window_dataset_shapes(tiny_dataset):
    train, _, _, meta = make_datasets(
        DataConfig(root=tiny_dataset, mode="window", history=4, chunk=4, goal_conditioned=True)
    )
    oh, ac, mk = train[0]
    assert oh.shape == (4, meta["obs_dim"])
    assert ac.shape == (4, meta["action_dim"])
    assert mk.shape == (4,)


def test_goal_conditioned_obs(tiny_dataset):
    # not goal-conditioned -> obs_dim == ndof
    _, _, _, meta = make_datasets(DataConfig(root=tiny_dataset, goal_conditioned=False))
    assert meta["obs_dim"] == meta["ndof"]


@pytest.mark.parametrize("name,extra", [("bc_mlp", {}), ("act_lite", {"history": 4, "chunk": 4})])
def test_policy_forward_and_norm_buffers(tiny_dataset, name, extra):
    _, _, stats, meta = make_datasets(DataConfig(root=tiny_dataset))
    cfg = {"obs_dim": meta["obs_dim"], "action_dim": meta["action_dim"], **extra}
    p = build_policy(name, cfg, stats)
    a = p.predict(np.zeros(meta["obs_dim"], dtype=np.float32))
    assert a.shape == (meta["action_dim"],)
    sd = p.state_dict()
    for k in ("in_mean", "in_std", "out_mean", "out_std"):
        assert k in sd, f"{name} missing norm buffer {k}"
