"""caliper_learn — a minimal, pure-torch behavior-cloning sidecar for Caliper.

Pipeline: collect (sim demos -> LeRobotDataset) -> data (torch Dataset) -> policy
(BC-MLP / ACT-lite / optional diffusion) -> train (CPU overfit-smoke) -> deploy
(closed-loop in sim via caliper.ControlLoop.step_with_target).

NO lerobot/hydra/diffusers — everything is hand-written stdlib torch + numpy. The
actual ACT/Diffusion training run on a GPU (the 4090s) is a documented, never-auto-
run deferral; locally everything is proven by seeded CPU oracles.
"""

__version__ = "0.1.0"

from .collect import collect_demos

__all__ = ["collect_demos", "__version__"]
