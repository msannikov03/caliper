"""OPTIONAL minimal DDPM action head — a small obs-conditioned noise predictor,
hand-written (no diffusers). Trains by noise-prediction MSE; `predict` runs the
DDPM reverse chain. Kept isolated so its (stochastic) test is cleanly skippable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .policy import Policy


class DiffusionHead(Policy):
    name = "diffusion"

    def __init__(self, obs_dim, action_dim, hidden: int = 256, steps: int = 100, sample_seed: int = 0):
        super().__init__(obs_dim, action_dim, {"hidden": hidden, "steps": steps, "sample_seed": sample_seed})
        self.steps = steps
        self.sample_seed = sample_seed
        self._gen: torch.Generator | None = None  # reverse-chain RNG (reproducible)
        betas = torch.linspace(1e-4, 0.02, steps)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def _eps(self, obs_n, a_t, t):
        # t: [B] long -> normalized scalar feature [B,1]
        tf = (t.float() / self.steps).unsqueeze(-1)
        return self.net(torch.cat([obs_n, a_t, tf], dim=-1))

    def forward(self, x):  # not used directly; predict() runs the reverse chain
        raise NotImplementedError("DiffusionHead uses loss()/predict(), not forward()")

    def loss(self, batch):
        obs, act = batch[0], batch[1]
        obs_n = self._norm_in(obs)
        a0 = self._norm_out(act)
        b = a0.shape[0]
        t = torch.randint(0, self.steps, (b,), device=a0.device)
        noise = torch.randn_like(a0)
        ab = self.abar[t].unsqueeze(-1)
        a_t = ab.sqrt() * a0 + (1.0 - ab).sqrt() * noise
        pred = self._eps(obs_n, a_t, t)
        return nn.functional.mse_loss(pred, noise)

    def reset(self) -> None:
        """Reseed the reverse-chain generator (deterministic deploy across reruns)."""
        self._gen = torch.Generator().manual_seed(self.sample_seed)

    @torch.no_grad()
    def predict(self, obs):
        if self._gen is None:
            self.reset()
        g = self._gen
        t1 = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        was_1d = t1.dim() == 1
        if was_1d:
            t1 = t1.unsqueeze(0)
        obs_n = self._norm_in(t1)
        b = obs_n.shape[0]
        a = torch.randn(b, self.action_dim, generator=g)
        for i in reversed(range(self.steps)):
            t = torch.full((b,), i, dtype=torch.long)
            eps = self._eps(obs_n, a, t)
            beta = self.betas[i]
            alpha = self.alphas[i]
            ab = self.abar[i]
            mean = (a - beta / (1.0 - ab).sqrt() * eps) / alpha.sqrt()
            a = mean + (beta.sqrt() * torch.randn(a.shape, generator=g) if i > 0 else 0.0)
        y = self._denorm_out(a).cpu().numpy().astype(np.float32)
        return y[0] if was_1d else y
