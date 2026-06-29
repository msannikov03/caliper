"""Dataset IO for the BC sidecar: read a LeRobotDataset (via caliper.DatasetReader),
reconstruct the episode goal (final measured q), split at the EPISODE level (no
frame leakage), compute train-only normalization stats, and expose frame (BC) and
window (ACT) torch Datasets.

obs = q (or concat(q, goal) when goal_conditioned) ; action = the recorded command.
Datasets are normalization-FREE — normalization lives in the policy's buffers so
train and deploy share one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Episode:
    index: int
    states: np.ndarray  # (T, ndof) f32
    actions: np.ndarray  # (T, ndof) f32
    timestamps: np.ndarray  # (T,) f32
    goal: np.ndarray  # (ndof,) f32 — final measured q


@dataclass
class NormStats:
    in_mean: np.ndarray
    in_std: np.ndarray
    out_mean: np.ndarray
    out_std: np.ndarray


@dataclass
class DataConfig:
    root: str
    mode: str = "frame"  # "frame" (BC) | "window" (ACT)
    history: int = 8
    chunk: int = 16
    goal_conditioned: bool = True
    val_frac: float = 0.2
    split_seed: int = 0


def load_episodes(root: str) -> tuple[list[Episode], dict]:
    import caliper

    rd = caliper.DatasetReader.open(str(root))
    eps: list[Episode] = []
    for i in range(rd.total_episodes):
        s, a, t = rd.read_episode(i)
        s = np.asarray(s, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        t = np.asarray(t, dtype=np.float32)
        if len(s) == 0:
            raise ValueError(f"episode {i} has no frames")
        eps.append(Episode(i, s, a, t, s[-1].copy()))
    return eps, {"ndof": rd.ndof, "fps": rd.fps}


def split_episodes(n: int, val_frac: float = 0.2, seed: int = 0) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    nval = int(round(n * val_frac)) if n > 1 else 0
    val = sorted(idx[:nval].tolist())
    train = sorted(idx[nval:].tolist())
    if not train:  # tiny-n guard: never leave train empty
        train, val = val, []
    return train, val


def _obs_rows(ep: Episode, goal_conditioned: bool) -> np.ndarray:
    if not goal_conditioned:
        return ep.states
    g = np.broadcast_to(ep.goal, ep.states.shape)
    return np.concatenate([ep.states, g], axis=1)


def _mean_std(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = x.mean(axis=0).astype(np.float32)
    s = x.std(axis=0).astype(np.float32)
    s[s < 1e-6] = 1.0  # guard constant dims
    return m, s


def compute_norm_stats(eps, train_idx, goal_conditioned: bool) -> NormStats:
    obs = np.concatenate([_obs_rows(eps[i], goal_conditioned) for i in train_idx], axis=0)
    act = np.concatenate([eps[i].actions for i in train_idx], axis=0)
    im, isd = _mean_std(obs)
    om, osd = _mean_std(act)
    return NormStats(im, isd, om, osd)


class BCFrameDataset(Dataset):
    """One (obs, action) pair per recorded frame."""

    def __init__(self, eps, idx, goal_conditioned: bool):
        obs = [_obs_rows(eps[i], goal_conditioned) for i in idx]
        act = [eps[i].actions for i in idx]
        self.obs = np.concatenate(obs, axis=0).astype(np.float32)
        self.act = np.concatenate(act, axis=0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, i):
        return torch.from_numpy(self.obs[i]), torch.from_numpy(self.act[i])


class ACTWindowDataset(Dataset):
    """Per anchor frame: an obs history (left-padded) + an action chunk (right-padded
    with a valid mask). Drives the ACT-lite policy."""

    def __init__(self, eps, idx, goal_conditioned: bool, history: int, chunk: int):
        H, K = history, chunk
        self.obs, self.act, self.mask = [], [], []
        for i in idx:
            o = _obs_rows(eps[i], goal_conditioned).astype(np.float32)
            a = eps[i].actions.astype(np.float32)
            T = len(o)
            for k in range(T):
                hist = np.stack([o[max(0, k - H + 1 + j)] for j in range(H)])
                ch, mk = [], []
                for j in range(K):
                    if k + j < T:
                        ch.append(a[k + j])
                        mk.append(1.0)
                    else:
                        ch.append(a[-1])
                        mk.append(0.0)
                self.obs.append(hist)
                self.act.append(np.stack(ch))
                self.mask.append(np.asarray(mk, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.obs[i]),
            torch.from_numpy(self.act[i]),
            torch.from_numpy(self.mask[i]),
        )


def make_datasets(cfg: DataConfig):
    """-> (train_ds, val_ds|None, NormStats, meta{ndof,fps,obs_dim,action_dim,goal_conditioned,mode})."""
    eps, meta = load_episodes(cfg.root)
    if not eps:
        raise ValueError(f"dataset at {cfg.root!r} has no episodes")
    ndof = meta["ndof"]
    obs_dim = 2 * ndof if cfg.goal_conditioned else ndof
    train_idx, val_idx = split_episodes(len(eps), cfg.val_frac, cfg.split_seed)
    stats = compute_norm_stats(eps, train_idx, cfg.goal_conditioned)
    if cfg.mode == "frame":
        train = BCFrameDataset(eps, train_idx, cfg.goal_conditioned)
        val = BCFrameDataset(eps, val_idx, cfg.goal_conditioned) if val_idx else None
    elif cfg.mode == "window":
        train = ACTWindowDataset(eps, train_idx, cfg.goal_conditioned, cfg.history, cfg.chunk)
        val = (
            ACTWindowDataset(eps, val_idx, cfg.goal_conditioned, cfg.history, cfg.chunk)
            if val_idx
            else None
        )
    else:
        raise ValueError(f"mode must be 'frame' or 'window', got {cfg.mode!r}")
    meta.update(
        obs_dim=obs_dim,
        action_dim=ndof,
        goal_conditioned=cfg.goal_conditioned,
        mode=cfg.mode,
    )
    return train, val, stats, meta
