"""Checkpoint save/load — a plain dict (config + state_dict incl. norm buffers),
never a pickled module. `load_checkpoint` rebuilds the policy from the registry and
loads weights, so deploy needs no knowledge of which policy it is.
"""

from __future__ import annotations

import torch

from .policy import build_policy

FORMAT = "caliper.learn/1"


def save_checkpoint(policy, path: str) -> None:
    torch.save(
        {
            "format": FORMAT,
            "policy": policy.name,
            "config": policy.config_dict(),
            "state_dict": policy.state_dict(),
            "torch_version": torch.__version__,
        },
        str(path),
    )


def load_checkpoint(path: str, map_location: str = "cpu"):
    d = torch.load(str(path), map_location=map_location, weights_only=False)
    if d.get("format") != FORMAT:
        raise ValueError(f"unrecognized checkpoint format {d.get('format')!r}")
    policy = build_policy(d["policy"], d["config"])
    policy.load_state_dict(d["state_dict"])
    policy.eval()
    return policy
