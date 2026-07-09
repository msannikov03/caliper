"""Load lerobot-Hub-convention policy checkpoints (SAFETENSORS ONLY, local dirs only).

Security stance (the whole point of this module)
------------------------------------------------
lerobot's own remote-inference path (`PolicyServer`) deserializes *pickle* over an
open port (CVE-2026-25874). Caliper never does that: this loader reads weights as
pure tensors from `model.safetensors` / processor `*.safetensors` files, entirely
in-process, with no network service and no `torch.load` anywhere on the path. Any
pickle-bearing file (`.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl`, `.pickle`) in the
checkpoint directory is refused loudly before anything is deserialized.

Checkpoint layout — VERIFIED against the installed lerobot 0.4.4 (.venv) by saving
a tiny ACTPolicy through `save_pretrained` + `make_act_pre_post_processors`:

- ``config.json``            draccus dump of ``ACTConfig``: ``"type": "act"``,
                             ``input_features``/``output_features`` as
                             ``{name: {"type": STATE|ENV|VISUAL|ACTION, "shape": [..]}}``,
                             ``normalization_mapping``, ``chunk_size``,
                             ``n_action_steps``, arch dims, ``device`` ...
- ``model.safetensors``      raw module weights ONLY. In lerobot 0.4.x normalization
                             stats are NOT buffers in the state dict any more — they
                             moved to processor pipelines (``lerobot/processor/
                             migrate_policy_normalization.py`` exists precisely to
                             migrate old buffer-style checkpoints).
- ``policy_preprocessor.json``   pipeline: rename_observations_processor →
                             to_batch_processor → device_processor →
                             normalizer_processor, whose ``state_file`` (e.g.
                             ``policy_preprocessor_step_3_normalizer_processor
                             .safetensors``) holds flat ``{feature}.mean`` /
                             ``{feature}.std`` tensors.
- ``policy_postprocessor.json``  unnormalizer_processor (its own ``*.safetensors``
                             stats) → device_processor(cpu).
- ``train_config.json``      ``TrainPipelineConfig`` (dataset repo, optimizer,
                             steps, ...) — training provenance only, NOT needed for
                             inference; we ignore it.

Loader path chosen: lazily import lerobot's own ``ACTPolicy`` +
``PolicyProcessorPipeline`` and drive their local-directory loading. Re-implementing
ACT (ResNet backbone, transformer encoder/decoder, VAE encoder, sinusoidal tables)
in pure torch would be ~1000 lines of drift risk for zero behavioral difference.
The no-pickle guarantee holds on this path — VERIFIED in the installed source:

- ``lerobot/policies/pretrained.py``: ``from_pretrained`` → ``_load_as_safetensor``
  → ``safetensors.torch.load_model`` (lines 29/111/146). Never ``torch.load``.
- ``lerobot/processor/pipeline.py``: ``from_pretrained`` reads the pipeline json and
  loads each step's ``state_file`` via ``safetensors.torch.load_file`` (line 46).

On top of that we scan the directory OURSELVES and refuse pickle-suffixed files
before handing anything to lerobot.

Scope this wave: ACT with state-only observations (STATE / ENV features). A
checkpoint declaring VISUAL input features raises ``NotImplementedError`` naming the
feature (vision arrives with the sim wave). Non-ACT types raise as well.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# File suffixes that are (or may be) pickle containers. NEVER loaded.
PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle")

_PICKLE_MSG = (
    "refusing to load {path}: '{name}' is a pickle-format file. Caliper loads policy "
    "weights from safetensors ONLY — pickle deserialization executes arbitrary code "
    "(this is the lerobot PolicyServer vulnerability class, CVE-2026-25874). "
    "Re-export the checkpoint with lerobot's save_pretrained (model.safetensors)."
)

# State-like feature types we can serve from ControlLoop.q; VISUAL is gated.
_STATE_FEATURE_TYPES = {"STATE", "ENV"}


class CheckpointSecurityError(RuntimeError):
    """A checkpoint was refused for security reasons (pickle-format weights)."""


@dataclass(frozen=True)
class ACTCheckpointConfig:
    """The subset of lerobot's ACTConfig (config.json) that this loader consumes.

    Everything else in config.json (arch dims etc.) is passed through untouched to
    lerobot's own ACTConfig parser — we only *inspect* these fields.
    """

    type: str
    input_features: dict[str, tuple[str, tuple[int, ...]]]  # name -> (ftype, shape)
    output_features: dict[str, tuple[str, tuple[int, ...]]]
    chunk_size: int
    n_action_steps: int
    n_obs_steps: int = 1
    temporal_ensemble_coeff: float | None = None
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def action_dim(self) -> int:
        (_, shape) = self.output_features["action"]
        return int(shape[0])

    @property
    def state_feature_names(self) -> list[str]:
        """Input features servable from proprioceptive state (STATE + ENV kinds)."""
        return [n for n, (t, _) in self.input_features.items() if t in _STATE_FEATURE_TYPES]


def _parse_features(d: dict) -> dict[str, tuple[str, tuple[int, ...]]]:
    return {name: (str(f["type"]), tuple(int(s) for s in f["shape"])) for name, f in d.items()}


def _read_checkpoint_config(config_file: Path) -> ACTCheckpointConfig:
    with open(config_file) as f:
        raw = json.load(f)
    return ACTCheckpointConfig(
        type=str(raw.get("type", "")),
        input_features=_parse_features(raw.get("input_features", {})),
        output_features=_parse_features(raw.get("output_features", {})),
        chunk_size=int(raw.get("chunk_size", 1)),
        n_action_steps=int(raw.get("n_action_steps", 1)),
        n_obs_steps=int(raw.get("n_obs_steps", 1)),
        temporal_ensemble_coeff=raw.get("temporal_ensemble_coeff"),
        raw=raw,
    )


def _security_scan(root: Path) -> None:
    """Refuse the whole checkpoint if ANY pickle-suffixed file is present."""
    offenders = sorted(
        p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in PICKLE_SUFFIXES
    )
    if offenders:
        raise CheckpointSecurityError(_PICKLE_MSG.format(path=root, name=offenders[0]))


class LoadedPolicy:
    """A lerobot Hub checkpoint ready for in-process inference on CPU.

    Wraps lerobot's policy module + its pre/post processor pipelines behind the tiny
    interface the deploy runner needs: ``reset()`` and ``predict(obs_dict) -> action``.

    Action chunking is handled by lerobot's OWN `select_action` semantics — VERIFIED
    in lerobot/policies/act/modeling_act.py (lines 92-122): the policy keeps an
    internal `deque(maxlen=n_action_steps)`, pops ONE action per call, and refills it
    from `predict_action_chunk()[:, :n_action_steps]` when empty (or runs the
    temporal ensembler when `temporal_ensemble_coeff` is set). Calling
    ``predict`` once per control tick therefore replans every `n_action_steps`
    ticks, exactly like `lerobot-eval` / `predict_action` in lerobot's own loop
    (lerobot/utils/control_utils.py:107-113: preprocess → select_action →
    postprocess), which this method mirrors.
    """

    def __init__(self, policy, preprocessor, postprocessor, config: ACTCheckpointConfig, path: Path):
        self._policy = policy
        self._pre = preprocessor
        self._post = postprocessor
        self.config = config
        self.path = path
        self.action_dim = config.action_dim
        self.n_action_steps = config.n_action_steps
        self.chunk_size = config.chunk_size

    def reset(self) -> None:
        """Clear the action queue / temporal ensemble. Call at episode start."""
        self._policy.reset()

    @torch.no_grad()
    def predict(self, obs: dict[str, "np.ndarray | torch.Tensor"]) -> np.ndarray:
        """One control tick: obs dict in (unbatched), one action out (1-D float32).

        `obs` must carry every input feature the checkpoint declares (e.g.
        ``{"observation.state": q, "observation.environment_state": q}``). The
        preprocessor pipeline adds the batch dim (to_batch_processor) and applies
        the checkpoint's normalization; the postprocessor unnormalizes the action.
        """
        missing = [n for n in self.config.input_features if n not in obs]
        if missing:
            raise KeyError(
                f"observation is missing input features {missing}; checkpoint expects "
                f"{sorted(self.config.input_features)}"
            )
        batch = {
            name: torch.as_tensor(np.asarray(obs[name]), dtype=torch.float32)
            for name in self.config.input_features
        }
        for name, t in batch.items():
            (_, shape) = self.config.input_features[name]
            if tuple(t.shape) != shape:
                raise ValueError(f"feature '{name}' has shape {tuple(t.shape)}, checkpoint expects {shape}")
        batch = self._pre(batch)
        action = self._policy.select_action(batch)
        action = self._post(action)
        return action.squeeze(0).to("cpu", torch.float32).numpy()


def load_lerobot_policy(path_or_dir: str | Path, device: str = "cpu") -> LoadedPolicy:
    """Load a local lerobot-Hub-convention checkpoint directory. SAFETENSORS ONLY.

    Local directories only — this loader never touches the network (no Hub download,
    no policy server). Raises:

    - ``CheckpointSecurityError`` — the path is, or the directory contains, a
      pickle-format file (.bin/.pt/.pth/.ckpt/.pkl/.pickle).
    - ``NotImplementedError`` — non-ACT policy type, or an input feature this wave
      does not support (VISUAL/image features; named in the message).
    - ``FileNotFoundError`` — missing model.safetensors / config.json /
      policy_{pre,post}processor.json (old-convention checkpoints with stats baked
      into the state dict need lerobot's migrate_policy_normalization first).
    """
    root = Path(path_or_dir)
    if root.is_file() or (not root.exists() and root.suffix):
        # A direct file path: only ever acceptable if it were safetensors, but the
        # convention needs the sidecar jsons — so direct files are always an error;
        # pickle suffixes get the security message.
        if root.suffix.lower() in PICKLE_SUFFIXES:
            raise CheckpointSecurityError(_PICKLE_MSG.format(path=root.parent, name=root.name))
        raise ValueError(
            f"{root} is a file; pass the checkpoint *directory* "
            "(config.json + model.safetensors + policy_{pre,post}processor.json)"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {root}")

    _security_scan(root)

    config_file = root / "config.json"
    model_file = root / "model.safetensors"
    pre_file = root / "policy_preprocessor.json"
    post_file = root / "policy_postprocessor.json"
    for p in (config_file, model_file):
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found in {root} — not a lerobot Hub checkpoint")
    for p in (pre_file, post_file):
        if not p.exists():
            raise FileNotFoundError(
                f"{p.name} not found in {root}. lerobot 0.4.x keeps normalization in "
                "processor pipelines; old checkpoints with stats in the state dict must "
                "be migrated first (lerobot/processor/migrate_policy_normalization.py)."
            )

    cfg = _read_checkpoint_config(config_file)
    if cfg.type != "act":
        raise NotImplementedError(
            f"policy type '{cfg.type}' is not supported yet — this wave loads ACT only"
        )
    visual = [n for n, (t, _) in cfg.input_features.items() if t not in _STATE_FEATURE_TYPES]
    if visual:
        raise NotImplementedError(
            f"checkpoint requires unsupported (non-state) input feature(s) {visual} — "
            "image/VISUAL observations arrive with the sim wave; this wave is state-only"
        )
    if not cfg.state_feature_names:
        raise ValueError(f"checkpoint declares no state input features: {root}")
    if "observation.state" not in cfg.input_features:
        # lerobot's own ACT forward unconditionally dereferences
        # batch["observation.state"].device (modeling_act.py lines 431 and 455 in
        # 0.4.4), so an ENV-only checkpoint crashes inside lerobot itself — verified
        # empirically. Gate it here with an honest message instead.
        raise NotImplementedError(
            "checkpoint has no 'observation.state' input feature; lerobot 0.4.4's ACT "
            "forward requires it to be present in the batch (modeling_act.py:431/455) "
            "even when only environment-state features are declared"
        )

    # Lazy import: lerobot is only needed when actually deploying a Hub checkpoint.
    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "loading lerobot Hub checkpoints needs the 'lerobot' package in this env "
            "(it is a caliper_learn extra, not a hard dependency)"
        ) from e

    # Parse config via lerobot's own parser, then pin the device (a cuda-trained
    # checkpoint stores device='cuda'; we run in-process on the requested device).
    lr_config = PreTrainedConfig.from_pretrained(str(root))
    lr_config.device = device
    # strict=True: refuse silently-mismatched weights (default in lerobot is False).
    policy = ACTPolicy.from_pretrained(str(root), config=lr_config, strict=True)
    policy.eval()

    # Processor pipelines: json + safetensors stats. Device overrides + transition
    # converters mirror lerobot's make_pre_post_processors pretrained branch
    # (lerobot/policies/factory.py:267-286) exactly.
    pre = PolicyProcessorPipeline.from_pretrained(
        str(root),
        config_filename=pre_file.name,
        overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {"device": device},
        },
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    post = PolicyProcessorPipeline.from_pretrained(
        str(root),
        config_filename=post_file.name,
        overrides={
            "unnormalizer_processor": {"device": device},
            "device_processor": {"device": "cpu"},
        },
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    loaded = LoadedPolicy(policy, pre, post, cfg, root)
    loaded.reset()
    return loaded
