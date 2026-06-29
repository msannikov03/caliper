import numpy as np
import pytest

pytest.importorskip("caliper")
torch = pytest.importorskip("torch")
from torch.utils.data import TensorDataset

from caliper_learn.policy import build_policy, seed_all
from caliper_learn.train import TrainConfig, fit


def test_diffusion_loss_drops_and_predicts():
    seed_all(0)
    obs_dim, action_dim = 6, 3
    g = torch.Generator().manual_seed(0)
    obs = torch.randn(8, obs_dim, generator=g)
    act = torch.randn(8, action_dim, generator=g)
    ds = TensorDataset(obs, act)
    p = build_policy(
        "diffusion",
        {"obs_dim": obs_dim, "action_dim": action_dim, "hidden": 64, "steps": 20},
    )
    hist = fit(p, ds, cfg=TrainConfig(steps=600, batch_size=8, lr=1e-3, log_every=1, seed=0))
    init, final = hist["train_loss"][0], hist["final_train"]
    # noise-prediction MSE is stochastic; just require a clear downward trend
    assert final < 0.7 * init, f"diffusion loss did not drop: {init}->{final}"
    # reverse-chain sampling produces a finite action of the right shape
    a = p.predict(np.zeros(obs_dim, dtype=np.float32))
    assert a.shape == (action_dim,) and np.isfinite(a).all()
