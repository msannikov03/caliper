"""caliper_learn — a minimal, pure-torch behavior-cloning sidecar for Caliper.

Pipeline: collect (sim demos -> LeRobotDataset) -> data (torch Dataset) -> policy
(BC-MLP / ACT-lite / optional diffusion) -> train (CPU overfit-smoke) -> deploy
(closed-loop in sim via caliper.ControlLoop.step_with_target).

NO lerobot/hydra/diffusers — everything is hand-written stdlib torch + numpy. The
actual ACT/Diffusion training run on a GPU (the 4090s) is a documented, never-auto-
run deferral; locally everything is proven by seeded CPU oracles.

DEPLOY leg (`hub` + `runner`): load lerobot-Hub-convention checkpoints —
SAFETENSORS ONLY, in-process, no pickle, no network service — and drive them
through Caliper's safety-monitored ControlLoop. `hub` lazily imports lerobot only
when a Hub checkpoint is actually loaded; the core sidecar stays lerobot-free.
"""

__version__ = "0.1.0"

from .collect import collect_demos
from .hub import CheckpointSecurityError, LoadedPolicy, load_lerobot_policy
from .runner import run_policy

__all__ = [
    "collect_demos",
    "load_lerobot_policy",
    "LoadedPolicy",
    "CheckpointSecurityError",
    "run_policy",
    "__version__",
]
