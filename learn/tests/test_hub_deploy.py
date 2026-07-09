"""DEPLOY leg: lerobot-Hub-convention checkpoints -> caliper_learn.hub -> runner.

Deterministic CPU oracles:
- round-trip parity: a tiny ACT saved through the INSTALLED lerobot, loaded via our
  safetensors-only loader, must match lerobot's own from_pretrained + processor
  pipelines action-for-action (including the chunk-queue refill).
- security: pickle-format files (.bin/.pt) are refused with the security message.
- honest gates: VISUAL features / missing processor jsons raise clearly.
- e2e: the loaded policy drives ControlLoop on showcase6 with the SafetyMonitor in
  the loop; no NaNs, warnings observable.
"""

import json
import shutil

import numpy as np
import pytest
import torch

import caliper  # conftest importorskips this for the whole directory  # noqa: F401

lerobot = pytest.importorskip("lerobot")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.act.processor_act import make_act_pre_post_processors  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402

from caliper_learn.hub import CheckpointSecurityError, load_lerobot_policy  # noqa: E402
from caliper_learn.runner import run_policy  # noqa: E402

DOF = 6  # matches showcase6
URDF = "oracle/fixtures/robots/showcase6.urdf"


def _make_tiny_checkpoint(d, dof=DOF, seed=0):
    """Save a tiny state-only ACT through the installed lerobot (the Hub convention
    itself is the oracle: whatever save_pretrained writes is what we must load)."""
    torch.manual_seed(seed)
    cfg = ACTConfig(
        # ACT requires >=1 image or the environment state (validate_features), so a
        # state-only checkpoint carries ENV (+ STATE) features.
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(dof,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(dof,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(dof,))},
        chunk_size=8,
        n_action_steps=4,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,  # deterministic in eval
        device="cpu",
    )
    policy = ACTPolicy(cfg)
    policy.eval()
    ar = torch.arange(dof, dtype=torch.float32)
    # NON-identity stats so a normalization bug cannot hide behind zeros/ones.
    stats = {
        "observation.state": {"mean": 0.1 * ar, "std": 1 + 0.05 * ar},
        "observation.environment_state": {"mean": -0.05 * ar, "std": 0.5 + 0.1 * ar},
        "action": {"mean": 0.02 * ar, "std": 0.8 + 0.02 * ar},
    }
    pre, post = make_act_pre_post_processors(cfg, stats)
    policy.save_pretrained(d)
    pre.save_pretrained(d)
    post.save_pretrained(d)
    return d


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory):
    return _make_tiny_checkpoint(tmp_path_factory.mktemp("act_ckpt"))


def test_round_trip_matches_lerobot(tiny_ckpt):
    """Our loader == lerobot's own loading path, action-for-action, 10 sequential
    ticks (n_action_steps=4, so this crosses two chunk-queue refills)."""
    ours = load_lerobot_policy(tiny_ckpt)
    assert ours.action_dim == DOF
    assert ours.n_action_steps == 4
    assert set(ours.config.state_feature_names) == {
        "observation.state",
        "observation.environment_state",
    }

    ref = ACTPolicy.from_pretrained(str(tiny_ckpt))
    ref.eval()
    ref_pre, ref_post = make_pre_post_processors(ref.config, pretrained_path=str(tiny_ckpt))

    ours.reset()
    ref.reset()
    for k in range(10):
        q = (np.linspace(-0.3, 0.3, DOF) * (k + 1) / 10).astype(np.float32)
        obs = {"observation.state": q, "observation.environment_state": q}
        a_ours = ours.predict(obs)
        batch = {n: torch.as_tensor(v) for n, v in obs.items()}
        with torch.no_grad():
            a_ref = ref_post(ref.select_action(ref_pre(batch))).squeeze(0).numpy()
        assert np.allclose(a_ours, a_ref, atol=1e-6), f"tick {k} diverged"
        assert np.all(np.isfinite(a_ours))


def test_rejects_pickle_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.bin").write_bytes(b"\x80\x04")  # pickle magic
    with pytest.raises(CheckpointSecurityError, match="pickle"):
        load_lerobot_policy(tmp_path)


def test_rejects_pickle_file_path(tmp_path):
    p = tmp_path / "policy.pt"
    p.write_bytes(b"\x80\x04")
    with pytest.raises(CheckpointSecurityError, match="pickle"):
        load_lerobot_policy(p)


def test_rejects_pickle_alongside_safetensors(tiny_ckpt, tmp_path):
    """Even a valid safetensors checkpoint is refused if a pickle file sits next to
    it — the directory as a whole is untrusted."""
    for f in tiny_ckpt.iterdir():
        shutil.copy(f, tmp_path / f.name)
    (tmp_path / "optimizer.ckpt").write_bytes(b"\x80\x04")
    with pytest.raises(CheckpointSecurityError, match="pickle"):
        load_lerobot_policy(tmp_path)


def test_vision_checkpoint_gated(tiny_ckpt, tmp_path):
    """A checkpoint declaring VISUAL inputs raises NotImplementedError NAMING the
    feature (honest gate — vision comes with the sim wave)."""
    for f in tiny_ckpt.iterdir():
        shutil.copy(f, tmp_path / f.name)
    cj = json.loads((tmp_path / "config.json").read_text())
    cj["input_features"]["observation.images.top"] = {"type": "VISUAL", "shape": [3, 96, 96]}
    (tmp_path / "config.json").write_text(json.dumps(cj))
    with pytest.raises(NotImplementedError, match="observation.images.top"):
        load_lerobot_policy(tmp_path)


def test_env_only_checkpoint_gated(tiny_ckpt, tmp_path):
    """lerobot 0.4.4's ACT forward unconditionally dereferences
    batch['observation.state'] (modeling_act.py:431/455), so an ENV-only checkpoint
    would crash inside lerobot — our loader gates it with an honest message."""
    for f in tiny_ckpt.iterdir():
        shutil.copy(f, tmp_path / f.name)
    cj = json.loads((tmp_path / "config.json").read_text())
    del cj["input_features"]["observation.state"]
    (tmp_path / "config.json").write_text(json.dumps(cj))
    with pytest.raises(NotImplementedError, match="observation.state"):
        load_lerobot_policy(tmp_path)


def test_missing_processor_json_raises(tiny_ckpt, tmp_path):
    """Old-convention checkpoints (stats in the state dict, no processor jsons) get
    a clear migration message instead of silently-unnormalized inference."""
    for f in tiny_ckpt.iterdir():
        if f.name.startswith("policy_preprocessor"):
            continue
        shutil.copy(f, tmp_path / f.name)
    with pytest.raises(FileNotFoundError, match="policy_preprocessor"):
        load_lerobot_policy(tmp_path)


def test_e2e_showcase6_smoke(tiny_ckpt):
    """The loaded policy drives ControlLoop.step_with_target on showcase6 for 50
    ticks: no NaNs, SafetyMonitor warnings observable (an untrained policy commands
    junk — the monitor SHOULD be busy; what matters is that it is in the loop)."""
    policy = load_lerobot_policy(tiny_ckpt)
    robot = caliper.Robot.from_urdf(URDF)
    assert robot.ndof == DOF
    cl = caliper.ControlLoop(robot, dt=1.0 / 50, start=[0.0] * DOF)
    res = run_policy(policy, cl, fps=50, ticks=50)
    S = np.asarray(res.states)
    A = np.asarray(res.actions)
    assert S.shape == (50, DOF) and A.shape == (50, DOF)
    assert np.all(np.isfinite(S)) and np.all(np.isfinite(A))
    assert isinstance(cl.last_warn, bool)  # safety observability stays accessible
    assert 0 <= res.warn_ticks <= 50
    assert res.times[1] - res.times[0] == pytest.approx(1.0 / 50)
