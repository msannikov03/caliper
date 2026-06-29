"""Generic training loop driving any Policy polymorphically via `policy.loss(batch)`.

Proven locally by a CPU overfit-smoke (see tests). `device="cuda"` is the DOCUMENTED
4090 path — it is NEVER auto-run here; the only proven bar is the CPU smoke.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .policy import seed_all


@dataclass
class TrainConfig:
    epochs: int = 100
    steps: Optional[int] = None  # if set, a step-budgeted loop (the smoke uses this)
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    drop_last: bool = False
    device: str = "cpu"  # "cpu" | "mps" | "cuda" (cuda = documented, never auto-run)
    seed: int = 0
    log_every: int = 50
    ckpt_path: Optional[str] = None


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        name = "cpu"
    return torch.device(name)


def _to(batch, device):
    return tuple(x.to(device) for x in batch)


def fit(
    policy,
    train_ds,
    val_ds=None,
    cfg: TrainConfig = TrainConfig(),
    progress: Optional[Callable[[int, float, Optional[float]], None]] = None,
) -> dict:
    """Train `policy` on `train_ds`. Returns a history dict
    {train_loss, val_loss, steps, best_val, final_train}."""
    seed_all(cfg.seed)
    device = resolve_device(cfg.device)
    policy.to(device)
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=cfg.drop_last,
        generator=gen,
    )
    if len(loader) == 0:
        raise ValueError(
            "empty training loader: dataset is empty, or smaller than batch_size with drop_last=True"
        )
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    hist = {"train_loss": [], "val_loss": [], "steps": [], "best_val": None, "final_train": None}
    step = 0
    last = float("nan")
    target_steps = cfg.steps if cfg.steps is not None else cfg.epochs * max(1, len(loader))
    done = False
    while not done:
        for batch in loader:
            policy.train()
            batch = _to(batch, device)
            opt.zero_grad()
            loss = policy.loss(batch)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step()
            last = float(loss.detach().cpu())
            step += 1
            if step % cfg.log_every == 0 or step == target_steps:
                v = _eval(policy, val_ds, cfg, device) if val_ds else None
                hist["train_loss"].append(last)
                hist["val_loss"].append(v)
                hist["steps"].append(step)
                if v is not None and math.isfinite(v) and (hist["best_val"] is None or v < hist["best_val"]):
                    hist["best_val"] = v
                    if cfg.ckpt_path:
                        save_checkpoint(policy, cfg.ckpt_path)
                if progress:
                    progress(step, last, v)
            if step >= target_steps:
                done = True
                break
    hist["final_train"] = last
    if cfg.ckpt_path and (val_ds is None or hist["best_val"] is None):
        save_checkpoint(policy, cfg.ckpt_path)  # no val → save last
    return hist


@torch.no_grad()
def _eval(policy, val_ds, cfg, device) -> float:
    policy.eval()
    loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    tot, nb = 0.0, 0
    for batch in loader:
        tot += float(policy.loss(_to(batch, device)).cpu())
        nb += 1
    return tot / nb if nb else float("nan")  # no fake-perfect 0.0 on an empty val set
