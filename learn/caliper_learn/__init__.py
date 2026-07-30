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

W2 DIAGNOSTICS: `eval` (seeded closed-loop harness + checkpoint `sweep`),
`profile` (deploy-loop latency, honest achievable Hz), `debugger`
(P001..P008 — "is it a vision problem or a control problem"), `autopsy`
(dataset doctor + debugger + eval + profile merged under one verdict), and the
`caliper-learn` console script (`cli.main`). Heavy deps (mujoco, lerobot,
safetensors reads) stay lazy inside those modules, matching the package rule.

W4 DATA ENGINE: `randomize` (seeded, diffable domain-randomization draws —
MJCF edits for model params, in-place application for runtime params, wired
into `VecSimEnv(randomization=...)`) and `coverage_gen` (the doctor→generator
loop: data_doctor D007 coverage holes → targeted planner episodes → doctor
re-run, `caliper-learn coverage`). `video` closes the last lerobot-parity
hole: dtype-"video" camera storage (per-episode mp4s, lerobot-mirrored encode
settings, `attach_video_metadata` post-write bridge) — `collect_camera_dataset
(video=True)`.

SUCCESS PREDICATES (`success`): composable, JSON-describable tests over the
SCENE rather than the arm — `Lifted`, `PlacedInZone`, `AllOf`/`AnyOf` — shared
by `VecSimEnv(success=...)` (per-step `info["success"]`), the eval harness
(`EvalTask.success_predicate`, Wilson-95 over predicate-scored episodes) and
the autopsy verdict, so a grasp/place run has ONE definition of "it worked".

TASK ARTIFACTS (`task`): `load_task` reads a `*.caliper-task.json` — robot +
scene (free props, target zones) + start pose + gripper + success predicate +
time budget — the same schema `caliper_sim_mujoco::task` writes, verified
against it by a shared parity table. `VecSimEnv.from_task(task)` and
`eval_task_from_file(path)` / `caliper-learn eval --task` consume it, so a
manipulation task is ONE diffable file instead of an argument list that can
disagree with itself.
"""

# Version identity: single-sourced from the installed distribution's metadata
# (pyproject.toml [project].version) so a bump never needs a code edit here.
try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("caliper_learn")
except Exception:  # graceful fallback: plain-checkout import, not pip-installed
    __version__ = "0.1.0"

from .autopsy import AutopsyReport, autopsy
from .collect import collect_demos
from .collect_sim import collect_camera_dataset
from .coverage_gen import CoverageReport, generate_coverage
from .debugger import (
    ACTION_COLLAPSE,
    ACTION_SATURATION,
    CADENCE_MISMATCH,
    CHUNK_CONFIG_ANOMALY,
    DEAD_INPUT,
    DOF_ACTION_COLLAPSE,
    NONFINITE_ACTION,
    NORMALIZATION_MISMATCH,
    PolicyFinding,
    analyze_policy,
    render_policy_findings,
)
from .eval import (
    ALL_EPISODES_FAILED,
    SEED_LOTTERY,
    ZERO_REWARD_SIGNAL,
    EpisodeResult,
    EvalConfig,
    EvalFinding,
    EvalResult,
    EvalTask,
    SweepEntry,
    eval_task_from_file,
    evaluate,
    reach_eval_task,
    render_text,
    sweep,
    to_json,
    wilson_interval,
)
from .hub import CheckpointSecurityError, LoadedPolicy, load_lerobot_policy
from .profile import (
    BUDGET_EXCEEDED,
    HIGH_JITTER,
    INFERENCE_DOMINATES,
    ChunkStats,
    Finding,
    LatencyReport,
    StageStats,
    profile_rollout,
)
from .randomize import RandomizationSpec, apply_to_env, apply_to_mjcf
from .randomize import sample as sample_randomization
from .runner import run_policy
from .sim_camera import SimCameraScene
from .success import (
    AllOf,
    AnyOf,
    Lifted,
    PlacedInZone,
    SuccessPredicate,
    SuccessState,
    Zone,
    state_from_mujoco,
)
from .success import as_predicate as as_success_predicate
from .success import from_dict as success_from_dict
from .task import TASK_VERSION, LearnTask, load_task, task_from_dict
from .vec_env import VecSimEnv, reach_task, rollout_random
from .video import VideoRecorder, attach_video_metadata, encode_episode_video

__all__ = [
    "collect_demos",
    "collect_camera_dataset",
    "SimCameraScene",
    "VecSimEnv",
    "reach_task",
    "rollout_random",
    # domain randomization + coverage generator
    "RandomizationSpec",
    "sample_randomization",
    "apply_to_mjcf",
    "apply_to_env",
    "generate_coverage",
    "CoverageReport",
    # success predicates (VecSimEnv success=, EvalTask.success_predicate)
    "SuccessPredicate",
    "SuccessState",
    "Zone",
    "Lifted",
    "PlacedInZone",
    "AllOf",
    "AnyOf",
    "as_success_predicate",
    "success_from_dict",
    "state_from_mujoco",
    # task artifacts (*.caliper-task.json)
    "LearnTask",
    "load_task",
    "task_from_dict",
    "TASK_VERSION",
    # dtype-"video" camera storage (mp4 encode + post-write bridge)
    "VideoRecorder",
    "attach_video_metadata",
    "encode_episode_video",
    "load_lerobot_policy",
    "LoadedPolicy",
    "CheckpointSecurityError",
    "run_policy",
    # eval harness
    "EvalTask",
    "EvalConfig",
    "EvalResult",
    "EpisodeResult",
    "EvalFinding",
    "SweepEntry",
    "evaluate",
    "sweep",
    "reach_eval_task",
    "eval_task_from_file",
    "render_text",
    "to_json",
    "wilson_interval",
    "ALL_EPISODES_FAILED",
    "SEED_LOTTERY",
    "ZERO_REWARD_SIGNAL",
    # latency profiler
    "profile_rollout",
    "LatencyReport",
    "StageStats",
    "ChunkStats",
    "Finding",
    "BUDGET_EXCEEDED",
    "INFERENCE_DOMINATES",
    "HIGH_JITTER",
    # policy debugger
    "analyze_policy",
    "PolicyFinding",
    "render_policy_findings",
    "ACTION_COLLAPSE",
    "DOF_ACTION_COLLAPSE",
    "ACTION_SATURATION",
    "NORMALIZATION_MISMATCH",
    "CADENCE_MISMATCH",
    "DEAD_INPUT",
    "NONFINITE_ACTION",
    "CHUNK_CONFIG_ANOMALY",
    # autopsy
    "autopsy",
    "AutopsyReport",
    "__version__",
]
