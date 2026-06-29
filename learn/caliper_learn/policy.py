"""Minimal pure-torch policies (no lerobot/hydra/diffusers).

`Policy` is the shared base: it carries normalization as registered BUFFERS (so
they ride inside `state_dict` and train/deploy share identical normalization),
exposes `loss(batch)` (driven generically by the train loop) and `predict(obs)`
(raw obs in → raw action out). `build_policy(name, cfg, stats)` is the registry.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


def seed_all(seed: int) -> None:
    """Seed python/numpy/torch and request deterministic algorithms (CPU oracles)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # older torch
        torch.use_deterministic_algorithms(True)


class Policy(nn.Module):
    """Base policy: normalization buffers + the loss/predict contract."""

    name = "base"

    def __init__(self, obs_dim: int, action_dim: int, cfg: dict):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = dict(cfg)
        self.register_buffer("in_mean", torch.zeros(obs_dim))
        self.register_buffer("in_std", torch.ones(obs_dim))
        self.register_buffer("out_mean", torch.zeros(action_dim))
        self.register_buffer("out_std", torch.ones(action_dim))

    # ---- normalization ----
    def set_norm(self, stats) -> None:
        self.in_mean.copy_(torch.as_tensor(stats.in_mean, dtype=torch.float32))
        self.in_std.copy_(torch.as_tensor(stats.in_std, dtype=torch.float32))
        self.out_mean.copy_(torch.as_tensor(stats.out_mean, dtype=torch.float32))
        self.out_std.copy_(torch.as_tensor(stats.out_std, dtype=torch.float32))

    def _norm_in(self, x):
        return (x - self.in_mean) / self.in_std

    def _norm_out(self, a):
        return (a - self.out_mean) / self.out_std

    def _denorm_out(self, y):
        return y * self.out_std + self.out_mean

    def config_dict(self) -> dict:
        return {"name": self.name, "obs_dim": self.obs_dim, "action_dim": self.action_dim, **self.cfg}

    def reset(self) -> None:
        """Clear any per-rollout state (history buffers, RNG). No-op by default;
        deploy.rollout_policy calls this at the start of each episode."""

    # ---- to be specialized ----
    def forward(self, x):  # normed obs -> normed action
        raise NotImplementedError

    def loss(self, batch):
        obs, act = batch[0], batch[1]
        pred = self.forward(self._norm_in(obs))
        return nn.functional.mse_loss(pred, self._norm_out(act))

    @torch.no_grad()
    def predict(self, obs):
        """Raw obs (obs_dim,) or (B, obs_dim) → raw action, same leading shape."""
        was_1d = False
        t = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        if t.dim() == 1:
            t = t.unsqueeze(0)
            was_1d = True
        y = self._denorm_out(self.forward(self._norm_in(t)))
        y = y.cpu().numpy().astype(np.float32)
        return y[0] if was_1d else y


class BCMLP(Policy):
    name = "bc_mlp"

    def __init__(self, obs_dim, action_dim, hidden: int = 256, depth: int = 2):
        super().__init__(obs_dim, action_dim, {"hidden": hidden, "depth": depth})
        layers: list[nn.Module] = [nn.Linear(obs_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ACTLite(Policy):
    """A small transformer-encoder over an obs history → an action chunk."""

    name = "act_lite"

    def __init__(
        self,
        obs_dim,
        action_dim,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 2,
        ff: int = 256,
        history: int = 8,
        chunk: int = 16,
    ):
        super().__init__(
            obs_dim,
            action_dim,
            {
                "d_model": d_model,
                "nhead": nhead,
                "layers": layers,
                "ff": ff,
                "history": history,
                "chunk": chunk,
            },
        )
        self.history = history
        self.chunk = chunk
        self.inp = nn.Linear(obs_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, history, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=ff, batch_first=True, dropout=0.0
        )
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, chunk * action_dim)
        self._buf: deque | None = None
        self._first = None

    def forward(self, obs_hist):
        # obs_hist: [B, H, obs_dim] (already normed)
        b = obs_hist.shape[0]
        h = self.inp(obs_hist) + self.pos
        h = self.enc(h)
        pooled = h.mean(dim=1)  # [B, d_model]
        out = self.head(pooled).view(b, self.chunk, self.action_dim)
        return out

    def loss(self, batch):
        obs_hist, act_chunk, mask = batch
        pred = self.forward(self._norm_in(obs_hist))  # [B,K,A]
        target = self._norm_out(act_chunk)
        per = nn.functional.mse_loss(pred, target, reduction="none").mean(dim=-1)  # [B,K]
        m = mask
        return (per * m).sum() / m.sum().clamp_min(1.0)

    def reset(self) -> None:
        """Start a fresh rolling obs history (call at episode start during deploy)."""
        self._buf = deque(maxlen=self.history)
        self._first = None

    @torch.no_grad()
    def predict(self, obs):
        """Single raw obs (obs_dim,) → first action of the predicted chunk.

        Maintains a rolling, left-padded obs history identical to the one
        ACTWindowDataset builds at train time (window[j] = obs[max(0, cur-H+1+j)]),
        so the temporal encoder sees in-distribution input rather than a degenerate
        repeat of the current obs. Returns the first action of the chunk (receding
        horizon: re-query every tick); chunk ensembling is left for future work.
        """
        o = np.asarray(obs, dtype=np.float32)
        if self._buf is None:
            self.reset()
        if self._first is None:
            self._first = o
        self._buf.append(o)
        pad = self.history - len(self._buf)
        window = [self._first] * pad + list(self._buf)  # left-pad with the episode's first obs
        hist = torch.as_tensor(np.stack(window)).view(1, self.history, self.obs_dim)
        chunk = self._denorm_out(self.forward(self._norm_in(hist)))  # [1,K,A]
        return chunk[0, 0].cpu().numpy().astype(np.float32)


_REGISTRY = {"bc_mlp": BCMLP, "act_lite": ACTLite}


def build_policy(name: str, cfg: dict, stats=None, seed: int | None = None) -> Policy:
    """Construct a policy from the registry. Pass `seed` to seed weight init
    reproducibly (calls seed_all BEFORE instantiation) — otherwise init draws from
    the ambient global torch RNG, so the caller must seed_all() before build_policy
    for reproducible weights (TrainConfig.seed only governs shuffling + stochastic loss)."""
    if seed is not None:
        seed_all(seed)
    cfg = dict(cfg)
    cfg.pop("name", None)
    if name == "diffusion":
        from .diffusion import DiffusionHead

        cls = DiffusionHead
    else:
        cls = _REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"unknown policy {name!r} (have {sorted(_REGISTRY) + ['diffusion']})")
    p = cls(**cfg)
    if stats is not None:
        p.set_norm(stats)
    return p
