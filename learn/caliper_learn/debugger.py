"""The policy deploy debugger: dataset + checkpoint in, "why does my trained
policy do nothing" out.

`analyze_policy(policy_dir, dataset_root=None, robot=None)` inspects a
lerobot-Hub-convention checkpoint (via the safetensors-only `hub` loader),
probes its forward pass on dataset-replayed observations, and names the mined
deploy failure modes in plain English with stable codes (the caliper-doctor
pattern — every finding carries `message`, `fix_hint`, and machine fields):

- P001 action collapse       every probe returns the same action — the network
                             output is a constant (classic zeroed/dead weights,
                             or a policy that learned the dataset mean).
- P002 per-dof collapse      a dof the DATA moves but the policy never does
                             (needs `dataset_root`).
- P003 saturation            actions pinned at / beyond the joint limits — the
                             SafetyMonitor will clamp every tick (needs `robot`).
- P004 normalization mismatch  the processor's train-time stats disagree with
                             stats recomputed from the dataset — the killer:
                             predictions are silently scaled/shifted garbage
                             (needs `dataset_root`).
- P005 cadence mismatch      the checkpoint's recorded training fps disagrees
                             with the dataset fps — the P7-lesson class: action
                             chunks consumed at the wrong rate (needs
                             `dataset_root` + a `train_config.json` that
                             declares an fps).
- P006 dead input            perturbing one state dim never changes the action —
                             the policy ignores that joint. Image-input probes
                             are honestly NotImplemented this wave (the loader
                             gates VISUAL checkpoints anyway).
- P007 non-finite forward    NaN/inf anywhere in the probed actions; the other
                             behavioral checks are skipped (their math would be
                             garbage on garbage).
- P008 chunk-config anomaly  `n_action_steps` vs `chunk_size` weirdness. Checked
                             STATICALLY from config.json before the model is
                             loaded — lerobot's own parser crashes on the worst
                             of these, so the debugger must speak first.

Thresholds are module constants CALIBRATED empirically (2026-07-13, tiny ACT on
the showcase6 sinusoid dataset): a random-init policy's per-dof normalized
action spread is >= 0.17 and its dead-input response >= 0.19, while zeroed
weights give exactly 0.0 for both — the 0.05 / 0.02 cuts sit well inside that
gap. Deterministic: probe states are a fixed linspace subsample of the dataset
(or a seeded rng around the processor's own state stats), the policy is
`reset()` before every probe forward, and findings are sorted severity-then-
code-then-dof — same inputs, identical findings.

Heavy deps stay as lazy as the rest of the package: `caliper`, `safetensors`,
and lerobot (via `hub.load_lerobot_policy`) are imported inside the functions
that need them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .hub import _read_checkpoint_config

# Stable check ids — machine-matchable, never renumbered.
ACTION_COLLAPSE = "P001"
DOF_ACTION_COLLAPSE = "P002"
ACTION_SATURATION = "P003"
NORMALIZATION_MISMATCH = "P004"
CADENCE_MISMATCH = "P005"
DEAD_INPUT = "P006"
NONFINITE_ACTION = "P007"
CHUNK_CONFIG_ANOMALY = "P008"

# --- calibrated thresholds (see module docstring for the measured gap) --------
# P001/P002: per-dof std of probed actions, normalized by the reference action
# std. Random-init measures >= 0.17; a collapsed output is exactly 0.
_COLLAPSE_SPREAD = 0.05
# P002 only judges dofs the data actually moves.
_DATA_MOVES_STD = 1e-4
# P003: fraction of probed actions outside a joint's limits.
_SATURATION_FRAC = 0.2
# P004: |mean shift| in units of the data std, and the std ratio band.
_STATS_MEAN_Z = 0.25
_STATS_STD_RATIO = 1.5
# P006: max action response to a one-dim state perturbation, normalized by the
# reference action std. Random-init measures >= 0.19; a dead input is 0.
_DEAD_INPUT_DELTA = 0.02
# Probe budget: forwards are cheap (CPU, tiny obs), keep the counts fixed for
# determinism and speed.
_N_PROBES = 32
_N_DEAD_BASES = 3

_SEVERITY_RANK = {"error": 0, "warn": 1, "info": 2}


@dataclass(frozen=True)
class PolicyFinding:
    """One diagnosed policy smell. `message` states what was observed and why
    it matters; `fix_hint` says what to do about it; `feature`/`dof`/`value`
    are the machine anchors (None when the finding is checkpoint-wide)."""

    code: str
    severity: str  # "error" | "warn" | "info"
    message: str
    fix_hint: Optional[str] = None
    feature: Optional[str] = None
    dof: Optional[int] = None
    value: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "fix_hint": self.fix_hint,
            "feature": self.feature,
            "dof": self.dof,
            "value": self.value,
        }


# ----- dataset access ----------------------------------------------------------


@dataclass(frozen=True)
class _DatasetView:
    """All frames of the dataset, flattened: what the debugger compares against."""

    states: np.ndarray  # (T, ndof) float64
    actions: np.ndarray  # (T, ndof) float64
    fps: int


def _load_dataset(root) -> _DatasetView:
    """Read every episode via the caliper readers (v3.0 first, v2.1 fallback)."""
    import caliper  # lazy runtime dep (built via maturin)

    last_err = None
    for opener in (caliper.DatasetReaderV3.open, caliper.DatasetReader.open):
        try:
            rd = opener(str(root))
            break
        except ValueError as e:
            last_err = e
    else:
        raise ValueError(f"cannot open dataset at {root}: {last_err}") from last_err
    states, actions = [], []
    for i in range(rd.total_episodes):
        s, a, _t = rd.read_episode(i)
        states.append(np.asarray(s, dtype=np.float64))
        actions.append(np.asarray(a, dtype=np.float64))
    if not states:
        raise ValueError(f"dataset at {root} has no episodes")
    return _DatasetView(
        states=np.concatenate(states), actions=np.concatenate(actions), fps=int(rd.fps)
    )


# ----- processor stats (the train-time normalization ground truth) --------------


def _read_flat_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """`{feature}.{mean|std}` flat safetensors -> {feature: {stat: array}}."""
    from safetensors import safe_open  # lazy: only needed for P004

    out: dict[str, dict[str, np.ndarray]] = {}
    with safe_open(str(path), framework="np") as f:
        for key in f.keys():
            feature, _, stat = key.rpartition(".")
            out.setdefault(feature, {})[stat] = np.asarray(f.get_tensor(key), dtype=np.float64)
    return out


def _load_processor_stats(root: Path):
    """(normalizer stats, unnormalizer stats) from the processor safetensors.

    lerobot 0.4.x file convention (verified against the installed package):
    `policy_preprocessor_step_<i>_normalizer_processor.safetensors` /
    `policy_postprocessor_step_<i>_unnormalizer_processor.safetensors`, each a
    flat `{feature}.mean` / `{feature}.std` tensor dict. Missing files return
    empty dicts — the stats checks then skip honestly.
    """
    pre = sorted(root.glob("policy_preprocessor_step_*_normalizer_processor.safetensors")) or sorted(
        root.glob("policy_preprocessor*.safetensors")
    )
    post = sorted(
        root.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors")
    ) or sorted(root.glob("policy_postprocessor*.safetensors"))
    return (
        _read_flat_stats(pre[0]) if pre else {},
        _read_flat_stats(post[0]) if post else {},
    )


# ----- static checks (config.json / train_config.json — no model load) ----------


def _check_chunk_config(cfg) -> list[PolicyFinding]:
    """P008 — chunk-config anomalies, judged from config.json alone (lerobot's
    own parser crashes on the invalid combinations, so this must run first)."""
    findings = []
    if cfg.n_action_steps > cfg.chunk_size:
        findings.append(
            PolicyFinding(
                CHUNK_CONFIG_ANOMALY,
                "error",
                f"n_action_steps={cfg.n_action_steps} > chunk_size={cfg.chunk_size} — the "
                "action queue would pop more steps than one forward pass produces; lerobot's "
                "own config validation rejects this checkpoint, it cannot run.",
                fix_hint=(
                    "set n_action_steps <= chunk_size in config.json (it was edited after "
                    "training, or the training config was invalid)"
                ),
                value=float(cfg.n_action_steps),
            )
        )
    if cfg.temporal_ensemble_coeff is not None and cfg.n_action_steps != 1:
        findings.append(
            PolicyFinding(
                CHUNK_CONFIG_ANOMALY,
                "error",
                f"temporal_ensemble_coeff={cfg.temporal_ensemble_coeff} with "
                f"n_action_steps={cfg.n_action_steps} — lerobot's temporal ensembling "
                "requires n_action_steps == 1 (the ensembler replaces the queue); this "
                "checkpoint cannot run.",
                fix_hint="set n_action_steps=1, or drop temporal_ensemble_coeff",
                value=float(cfg.n_action_steps),
            )
        )
    if cfg.chunk_size > 1 and cfg.n_action_steps == 1 and cfg.temporal_ensemble_coeff is None:
        findings.append(
            PolicyFinding(
                CHUNK_CONFIG_ANOMALY,
                "info",
                f"chunk_size={cfg.chunk_size} but n_action_steps=1 with no temporal "
                "ensembling — the full network re-runs EVERY control tick and predicts "
                f"{cfg.chunk_size} actions of which {cfg.chunk_size - 1} are thrown away.",
                fix_hint=(
                    "raise n_action_steps to amortize the forward pass over more ticks, or "
                    "enable temporal ensembling if per-tick replanning is intentional"
                ),
                value=float(cfg.chunk_size),
            )
        )
    return findings


def _declared_train_fps(root: Path) -> Optional[int]:
    """The training fps, if the checkpoint's train_config.json declares one
    (lerobot's EnvConfig carries `fps`; some configs put it at the top level)."""
    tc = root / "train_config.json"
    if not tc.exists():
        return None
    try:
        raw = json.loads(tc.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for holder in (raw, raw.get("env") or {}, raw.get("dataset") or {}):
        fps = holder.get("fps") if isinstance(holder, dict) else None
        if fps is not None:
            return int(fps)
    return None


def _check_cadence(root: Path, ds: Optional[_DatasetView]) -> list[PolicyFinding]:
    """P005 — training fps vs dataset fps. The repo's P7 lesson class: a chunk
    policy deployed at the wrong cadence consumes its action queue at the wrong
    rate (collect-at-50-deploy-at-1000 closed only ~42% of the gap)."""
    if ds is None:
        return []
    train_fps = _declared_train_fps(root)
    if train_fps is None or train_fps == ds.fps:
        return []
    return [
        PolicyFinding(
            CADENCE_MISMATCH,
            "error",
            f"the checkpoint's train_config.json declares fps={train_fps} but the dataset "
            f"is fps={ds.fps} — action chunks will be consumed at the wrong rate at deploy "
            "(each queued action is 'worth' a different amount of real time than the one "
            "it was trained to be).",
            fix_hint=(
                f"deploy at dt = 1/{ds.fps} (the collection cadence) and retrain if the "
                "checkpoint really was trained on differently-timed data — train and "
                "deploy must share one fps"
            ),
            value=float(train_fps),
        )
    ]


def _check_normalization(cfg, pre_stats, post_stats, ds: _DatasetView) -> list[PolicyFinding]:
    """P004 — processor stats vs stats recomputed from the dataset itself.

    State-like input features are compared against the recomputed state
    mean/std (deploy feeds measured q to every one of them — the runner
    contract), the action feature against the recomputed action stats. A
    mismatch means the policy normalizes/unnormalizes with numbers from a
    different dataset: every prediction is silently shifted and scaled.
    """
    findings = []
    s_mean, s_std = ds.states.mean(axis=0), ds.states.std(axis=0)
    a_mean, a_std = ds.actions.mean(axis=0), ds.actions.std(axis=0)
    targets = [(name, pre_stats.get(name), s_mean, s_std) for name in cfg.state_feature_names]
    targets.append(("action", (post_stats or pre_stats).get("action"), a_mean, a_std))

    for feature, proc, d_mean, d_std in targets:
        if not proc or "mean" not in proc or "std" not in proc:
            continue  # no stats stored for this feature — nothing to compare
        p_mean, p_std = proc["mean"].reshape(-1), proc["std"].reshape(-1)
        if p_mean.shape != d_mean.shape:
            findings.append(
                PolicyFinding(
                    NORMALIZATION_MISMATCH,
                    "error",
                    f"feature '{feature}': processor stats have {p_mean.shape[0]} dims but "
                    f"the dataset has {d_mean.shape[0]} — this checkpoint was trained on a "
                    "different robot or feature layout.",
                    fix_hint="retrain on this dataset (or point the debugger at the right one)",
                    feature=feature,
                )
            )
            continue
        bad_dofs, worst = [], 0.0
        for j in range(len(d_mean)):
            z = abs(p_mean[j] - d_mean[j]) / max(d_std[j], 1e-3)
            ratio_bad = False
            if d_std[j] >= _DATA_MOVES_STD or p_std[j] >= _DATA_MOVES_STD:
                ratio = p_std[j] / max(d_std[j], 1e-6)
                ratio_bad = not (1.0 / _STATS_STD_RATIO <= ratio <= _STATS_STD_RATIO)
                worst = max(worst, z, ratio if ratio_bad else 0.0)
            else:
                worst = max(worst, z)  # constant in both — only the mean can disagree
            if z > _STATS_MEAN_Z or ratio_bad:
                bad_dofs.append(j)
        if bad_dofs:
            findings.append(
                PolicyFinding(
                    NORMALIZATION_MISMATCH,
                    "error",
                    f"feature '{feature}': the checkpoint's normalization stats disagree "
                    f"with stats recomputed from the dataset on dof(s) {bad_dofs} — the "
                    "policy was trained against different data statistics, so its inputs/"
                    "outputs are silently shifted and scaled at deploy.",
                    fix_hint=(
                        "the classic causes: the dataset was edited/regrown after training, "
                        "or the wrong checkpoint is paired with this dataset. Retrain, or "
                        "regenerate the processor stats from THIS dataset"
                    ),
                    feature=feature,
                    dof=bad_dofs[0],
                    value=float(worst),
                )
            )
    return findings


# ----- behavioral probes (the loaded model) --------------------------------------


def _probe_states(ds: Optional[_DatasetView], pre_stats, ndof: int) -> np.ndarray:
    """(N, ndof) float32 probe q's. Dataset-replayed when possible (a fixed
    linspace subsample — real states the policy claims to handle); otherwise a
    seeded grid around the processor's OWN state stats (the distribution the
    checkpoint says it was trained on)."""
    if ds is not None:
        idx = np.linspace(0, len(ds.states) - 1, _N_PROBES).round().astype(int)
        return ds.states[idx, :ndof].astype(np.float32)
    stats = pre_stats.get("observation.state") or {}
    mean = stats.get("mean", np.zeros(ndof)).reshape(-1)[:ndof]
    std = stats.get("std", np.ones(ndof)).reshape(-1)[:ndof]
    u = np.random.default_rng(0).uniform(-2.0, 2.0, size=(_N_PROBES, ndof))
    return (mean + std * u).astype(np.float32)


def _forward(policy, q: np.ndarray) -> np.ndarray:
    """One fresh forward: reset first so the chunk queue never serves a stale
    action from the previous probe (lerobot pops a queue every predict)."""
    policy.reset()
    obs = {name: q for name in policy.config.state_feature_names}
    return np.asarray(policy.predict(obs), dtype=np.float64)


def _behavioral_checks(policy, ds, robot, pre_stats, post_stats) -> list[PolicyFinding]:
    ndof = int(policy.config.input_features[policy.config.state_feature_names[0]][1][0])
    probes = _probe_states(ds, pre_stats, ndof)
    actions = np.stack([_forward(policy, q) for q in probes])

    # P007 first: everything downstream is garbage-on-garbage if this fires.
    if not np.all(np.isfinite(actions)):
        bad = np.argwhere(~np.isfinite(actions))
        return [
            PolicyFinding(
                NONFINITE_ACTION,
                "error",
                f"the forward pass produced non-finite actions on {len(np.unique(bad[:, 0]))} "
                f"of {len(probes)} probe states (first at probe {bad[0][0]}, action dof "
                f"{bad[0][1]}) — NaN/inf weights or a numerically exploding layer.",
                fix_hint=(
                    "inspect model.safetensors for NaN/inf tensors and the training run for "
                    "loss spikes; the other behavioral checks were skipped (they cannot be "
                    "trusted on non-finite output)"
                ),
                dof=int(bad[0][1]),
            )
        ]

    findings: list[PolicyFinding] = []
    if ds is not None:
        ref_std = ds.actions.std(axis=0)
    elif "action" in (post_stats or {}):
        ref_std = post_stats["action"]["std"].reshape(-1)
    else:
        ref_std = np.ones(actions.shape[1])
    spread = actions.std(axis=0) / np.maximum(ref_std, 1e-6)

    # P001 — global collapse: every probe returns (essentially) the same action.
    if float(spread.max()) < _COLLAPSE_SPREAD:
        findings.append(
            PolicyFinding(
                ACTION_COLLAPSE,
                "error",
                f"across {len(probes)} varied probe states every action is (nearly) "
                f"identical — max per-dof spread is {spread.max():.4f} of the data's action "
                "scale. The policy has collapsed to a constant (typically the dataset mean).",
                fix_hint=(
                    "training loss can look fine here — a mean-predictor is the L2 optimum "
                    "for unlearnable labels. Check for zeroed/corrupted weights, then whether "
                    "the state->action map is one-to-many (the repo's one-step-lookahead "
                    "labeling exists precisely to fix that)"
                ),
                value=float(spread.max()),
            )
        )

    # P002 — per-dof collapse: the data moves this dof, the policy never does.
    if ds is not None:
        data_std = ds.actions.std(axis=0)
        for j in range(actions.shape[1]):
            if data_std[j] > _DATA_MOVES_STD and float(spread[j]) < _COLLAPSE_SPREAD:
                findings.append(
                    PolicyFinding(
                        DOF_ACTION_COLLAPSE,
                        "warn",
                        f"action dof {j} never moves across the probes (spread "
                        f"{spread[j]:.4f} of its data scale) while the dataset moves it "
                        f"(std {data_std[j]:.4f}) — the policy has written this joint off.",
                        fix_hint=(
                            "look at this dof's loss contribution and normalization std; a "
                            "per-dof std that is wrongly large makes its normalized targets "
                            "vanish, and the network learns to ignore the joint"
                        ),
                        dof=j,
                        value=float(spread[j]),
                    )
                )

    # P003 — saturation against the robot's joint limits.
    if robot is not None:
        from .collect import _bounds  # URDF limits, ±pi when unbounded

        bounds = _bounds(robot)
        if bounds.shape[0] == actions.shape[1]:
            outside = (actions < bounds[:, 0]) | (actions > bounds[:, 1])
            for j in range(actions.shape[1]):
                frac = float(outside[:, j].mean())
                if frac > _SATURATION_FRAC:
                    findings.append(
                        PolicyFinding(
                            ACTION_SATURATION,
                            "warn",
                            f"action dof {j} lands outside the joint limits "
                            f"[{bounds[j, 0]:.3f}, {bounds[j, 1]:.3f}] on {frac:.0%} of "
                            "probes — at deploy the SafetyMonitor clamps it every tick and "
                            "the commanded motion is not what the policy 'intended'.",
                            fix_hint=(
                                "almost always an unnormalization-scale problem (check P004 "
                                "and the action std in the postprocessor stats) — genuine "
                                "limit-riding demonstrations are rare"
                            ),
                            dof=j,
                            value=frac,
                        )
                    )

    # P006 — dead inputs: perturbing one state dim never changes the action.
    if ds is not None:
        s_std = ds.states.std(axis=0)[:ndof]
    else:
        s_std = (pre_stats.get("observation.state") or {}).get("std", np.ones(ndof)).reshape(-1)[
            :ndof
        ]
    base_idx = np.linspace(0, len(probes) - 1, _N_DEAD_BASES).round().astype(int)
    rel_ref = np.maximum(ref_std, 1e-6)
    for j in range(ndof):
        delta = max(0.1, 0.25 * float(s_std[j]))
        response = 0.0
        for b in probes[base_idx]:
            hi, lo = b.copy(), b.copy()
            hi[j] += delta
            lo[j] -= delta
            diff = np.abs(_forward(policy, hi) - _forward(policy, lo)) / rel_ref
            response = max(response, float(diff.max()))
        if response < _DEAD_INPUT_DELTA:
            findings.append(
                PolicyFinding(
                    DEAD_INPUT,
                    "warn",
                    f"perturbing state dim {j} by ±{delta:.3f} never changes the action "
                    f"(max normalized response {response:.4f}) — the policy ignores this "
                    "joint's measurement.",
                    fix_hint=(
                        "if the task genuinely needs this joint the policy cannot close the "
                        "loop on it; check the feature wiring and whether this dim was "
                        "constant in training (see the dataset doctor's D001)"
                    ),
                    dof=j,
                    value=response,
                )
            )
    return findings


# ----- the entry point ------------------------------------------------------------


def analyze_policy(policy_dir, dataset_root=None, robot=None) -> list[PolicyFinding]:
    """Debug a checkpoint against (optionally) its dataset and robot.

    - `policy_dir`: lerobot-Hub-convention checkpoint directory (loaded with the
      safetensors-only `hub` loader — same security stance).
    - `dataset_root`: the LeRobotDataset it was trained on; unlocks P002/P004/
      P005 and dataset-replayed probes for the behavioral checks.
    - `robot`: a `caliper.Robot`; unlocks P003 (joint-limit saturation).

    Returns findings sorted severity-then-code-then-dof; `[]` means every
    reachable check passed. Static config checks run FIRST, and when they find
    an error-grade P008 the model load is skipped entirely (lerobot's own
    parser would crash on such a config — the debugger's whole job is to say
    so calmly). Deterministic for a given (checkpoint, dataset, robot) triple.
    """
    root = Path(policy_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {root}")
    config_file = root / "config.json"
    if not config_file.exists():
        raise ValueError(f"{root} has no config.json — not a lerobot Hub checkpoint")
    cfg = _read_checkpoint_config(config_file)

    findings = _check_chunk_config(cfg)
    ds = _load_dataset(dataset_root) if dataset_root is not None else None
    findings += _check_cadence(root, ds)
    pre_stats, post_stats = _load_processor_stats(root)
    if ds is not None:
        findings += _check_normalization(cfg, pre_stats, post_stats, ds)

    if not any(f.code == CHUNK_CONFIG_ANOMALY and f.severity == "error" for f in findings):
        from .hub import load_lerobot_policy  # lazy: pulls in torch + lerobot

        policy = load_lerobot_policy(root)
        findings += _behavioral_checks(policy, ds, robot, pre_stats, post_stats)

    findings.sort(
        key=lambda f: (
            _SEVERITY_RANK[f.severity],
            f.code,
            f.dof if f.dof is not None else -1,
            f.feature or "",
        )
    )
    return findings


def render_policy_findings(findings: list[PolicyFinding]) -> str:
    """Tidy human report for a list of policy findings."""
    if not findings:
        return "no findings — every reachable policy check passed.\n"
    out = [f"policy findings ({len(findings)}):"]
    for f in findings:
        anchor = "".join(
            [
                f" feature={f.feature}" if f.feature is not None else "",
                f" dof={f.dof}" if f.dof is not None else "",
            ]
        )
        out.append(f"  [{f.code}] ({f.severity}){anchor} {f.message}")
        if f.fix_hint:
            out.append(f"         fix: {f.fix_hint}")
    return "\n".join(out) + "\n"
