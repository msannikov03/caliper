"""Debugger + autopsy + CLI tests: every P-check has a positive (crafted
defect detected) AND a negative (clean input, zero findings) case, determinism
is exact-equality, the autopsy verdict leads with the most severe section, and
the CLI is smoked by calling main() directly (no subprocess).

The fixtures mirror the proven oracle recipes: the clean dataset is the
test_doctors sinusoid+affine construction (doctor-clean by design), and the
tiny ACT checkpoint is the test_hub_deploy construction — except its processor
stats are computed FROM the dataset, so the clean pairing really is clean
(P004 compares those stats against recomputed dataset stats)."""

import json
import math
import shutil

import numpy as np
import pytest
import torch

import caliper  # conftest importorskips this for the whole directory  # noqa: F401

mujoco = pytest.importorskip("mujoco")
lerobot = pytest.importorskip("lerobot")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.act.processor_act import make_act_pre_post_processors  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from caliper_learn.autopsy import AutopsyReport, autopsy  # noqa: E402
from caliper_learn.cli import main  # noqa: E402
from caliper_learn.debugger import (  # noqa: E402
    ACTION_COLLAPSE,
    ACTION_SATURATION,
    CADENCE_MISMATCH,
    CHUNK_CONFIG_ANOMALY,
    DEAD_INPUT,
    DOF_ACTION_COLLAPSE,
    NONFINITE_ACTION,
    NORMALIZATION_MISMATCH,
    analyze_policy,
    render_policy_findings,
)
from caliper_learn.eval import EvalConfig, reach_eval_task  # noqa: E402

FPS = 25
DOF = 6  # showcase6
URDF = "oracle/fixtures/robots/showcase6.urdf"

# The test_doctors clean-dataset construction: decorrelated per-dof sinusoids
# (coverage, no corridor) + a fixed affine action policy (deterministic labels,
# no echo, comparable range) — verified doctor-clean.
_AFFINE_C = [1.8, -1.6, 1.5, -1.4, 1.7, -1.3]
_AFFINE_B = [0.4, -0.3, 0.25, 0.35, -0.45, 0.3]
_FREQS = [0.37, 0.53, 0.71, 0.89, 1.07, 1.31]
_AMPS = [0.8, 0.7, 0.6, 0.5, 0.45, 0.4]


def _codes(findings):
    return {f.code for f in findings}


def _write_clean_dataset(root, robot):
    rec = caliper.RecorderV3(robot, str(root), FPS)
    for e, phase in enumerate((0.0, 1.1, 2.3)):
        rec.start_episode(f"sweep {e}")
        for k in range(150):
            t = k / FPS
            s = [
                a * math.sin(2.0 * math.pi * fr * t + phase + 0.9 * i)
                for i, (a, fr) in enumerate(zip(_AMPS, _FREQS))
            ]
            rec.append(s, [c * si + b for c, si, b in zip(_AFFINE_C, s, _AFFINE_B)], t)
        rec.finalize_episode()
    rec.close()
    return root


def _write_defective_dataset(root, robot):
    """The test_doctors defect census: state dof 1 never moves (D001), action
    echoes the state (D004), episode 4 is a length outlier (D009), episode 0's
    tail is frozen (D011)."""
    rec = caliper.RecorderV3(robot, str(root), FPS)
    for e in range(5):
        frames = 8 if e == 4 else 60
        rec.start_episode("defective demo")
        for k in range(frames):
            kk = min(k, frames - 6) if e == 0 else k
            arg = 2.0 * math.pi * 0.41 * (kk / FPS) + 0.7 * e
            s = [math.sin(arg + 0.8 * i) for i in range(DOF)]
            s[1] = 0.5
            rec.append(s, list(s), k / FPS)
        rec.finalize_episode()
    rec.close()
    return root


def _dataset_stats(root):
    rd = caliper.DatasetReaderV3.open(str(root))
    S, A = [], []
    for i in range(rd.total_episodes):
        s, a, _t = rd.read_episode(i)
        S.append(np.asarray(s))
        A.append(np.asarray(a))
    S, A = np.concatenate(S), np.concatenate(A)
    return S.mean(0), S.std(0), A.mean(0), A.std(0)


def _make_ckpt(d, dataset_root, seed=0):
    """Tiny state-only ACT (the test_hub_deploy construction) whose processor
    stats are the dataset's own recomputed stats — a genuinely matched pair."""
    s_mean, s_std, a_mean, a_std = _dataset_stats(dataset_root)
    torch.manual_seed(seed)
    cfg = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(DOF,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(DOF,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(DOF,))},
        chunk_size=8,
        n_action_steps=4,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        device="cpu",
    )
    policy = ACTPolicy(cfg)
    policy.eval()
    tt = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
    stats = {
        "observation.state": {"mean": tt(s_mean), "std": tt(s_std)},
        "observation.environment_state": {"mean": tt(s_mean), "std": tt(s_std)},
        "action": {"mean": tt(a_mean), "std": tt(a_std)},
    }
    pre, post = make_act_pre_post_processors(cfg, stats)
    policy.save_pretrained(d)
    pre.save_pretrained(d)
    post.save_pretrained(d)
    return d


def _copy_ckpt(src, dst):
    for f in src.iterdir():
        shutil.copy(f, dst / f.name)
    return dst


@pytest.fixture(scope="module")
def robot():
    return caliper.Robot.from_urdf(URDF)


@pytest.fixture(scope="module")
def clean_ds(tmp_path_factory, robot):
    return _write_clean_dataset(tmp_path_factory.mktemp("clean") / "ds", robot)


@pytest.fixture(scope="module")
def clean_ckpt(tmp_path_factory, clean_ds):
    return _make_ckpt(tmp_path_factory.mktemp("ckpt"), clean_ds)


@pytest.fixture(scope="module")
def zeroed_ckpt(tmp_path_factory, clean_ckpt):
    """The classic dead checkpoint: every weight zeroed — the network output is
    exactly the unnormalizer mean, for every input."""
    d = _copy_ckpt(clean_ckpt, tmp_path_factory.mktemp("ckpt_zero"))
    w = load_file(str(d / "model.safetensors"))
    save_file({k: torch.zeros_like(v) for k, v in w.items()}, str(d / "model.safetensors"))
    return d


# ----- the clean pair: one negative case covering EVERY check ---------------------


def test_clean_pair_zero_findings(clean_ckpt, clean_ds, robot):
    findings = analyze_policy(clean_ckpt, clean_ds, robot=robot)
    assert findings == [], [f.to_dict() for f in findings]
    assert "no findings" in render_policy_findings(findings)


# ----- P001 / P002 / P006: zeroed weights ------------------------------------------


def test_zeroed_weights_fire_collapse_and_dead_inputs(zeroed_ckpt, clean_ds, robot):
    findings = analyze_policy(zeroed_ckpt, clean_ds, robot=robot)
    codes = _codes(findings)
    assert {ACTION_COLLAPSE, DOF_ACTION_COLLAPSE, DEAD_INPUT} <= codes, sorted(codes)
    # NOT a saturation or NaN problem — a constant-at-the-mean is finite and in-range
    assert ACTION_SATURATION not in codes and NONFINITE_ACTION not in codes
    p001 = [f for f in findings if f.code == ACTION_COLLAPSE]
    assert len(p001) == 1 and p001[0].severity == "error"
    assert p001[0].value is not None and p001[0].value < 0.05
    # every data-moving dof is individually named, with machine anchors
    assert {f.dof for f in findings if f.code == DOF_ACTION_COLLAPSE} == set(range(DOF))
    assert {f.dof for f in findings if f.code == DEAD_INPUT} == set(range(DOF))
    for f in findings:
        assert f.message and f.fix_hint  # plain-English + actionable, always


# ----- P004: normalization mismatch -------------------------------------------------


def test_wrong_processor_stats_fire_p004(tmp_path, clean_ckpt, clean_ds):
    d = _copy_ckpt(clean_ckpt, tmp_path)
    f = d / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    stats = load_file(str(f))
    stats["observation.state.mean"] = stats["observation.state.mean"] + 3.0  # ~6 data stds
    save_file(stats, str(f))
    findings = analyze_policy(d, clean_ds)
    p004 = [x for x in findings if x.code == NORMALIZATION_MISMATCH]
    assert p004, [x.to_dict() for x in findings]
    assert any(x.feature == "observation.state" for x in p004)
    assert all(x.severity == "error" for x in p004)
    # gating negative: without a dataset there is nothing to compare against
    assert NORMALIZATION_MISMATCH not in _codes(analyze_policy(d, None))


# ----- P003: saturation --------------------------------------------------------------


def test_huge_unnormalizer_std_fires_p003(tmp_path, clean_ckpt, clean_ds, robot):
    d = _copy_ckpt(clean_ckpt, tmp_path)
    f = d / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    stats = load_file(str(f))
    stats["action.std"] = stats["action.std"] * 500.0  # actions land ~hundreds of rad
    save_file(stats, str(f))
    findings = analyze_policy(d, clean_ds, robot=robot)
    p003 = [x for x in findings if x.code == ACTION_SATURATION]
    assert p003 and all(x.severity == "warn" and x.dof is not None for x in p003)
    # the root cause is also named: the tampered action stats mismatch the data
    assert NORMALIZATION_MISMATCH in _codes(findings)
    # gating negative: no robot, no limits to saturate against
    assert ACTION_SATURATION not in _codes(analyze_policy(d, clean_ds))


# ----- P005: cadence mismatch ---------------------------------------------------------


def test_fps_mismatch_fires_p005(tmp_path, clean_ckpt, clean_ds):
    d = tmp_path / "bad"
    d.mkdir()
    _copy_ckpt(clean_ckpt, d)
    (d / "train_config.json").write_text(json.dumps({"env": {"fps": 30}}))
    findings = analyze_policy(d, clean_ds)
    p005 = [x for x in findings if x.code == CADENCE_MISMATCH]
    assert len(p005) == 1 and p005[0].severity == "error"
    assert "30" in p005[0].message and str(FPS) in p005[0].message
    assert p005[0].value == 30.0

    # negative: a train_config that AGREES with the dataset stays silent
    d2 = tmp_path / "good"
    d2.mkdir()
    _copy_ckpt(clean_ckpt, d2)
    (d2 / "train_config.json").write_text(json.dumps({"env": {"fps": FPS}}))
    assert CADENCE_MISMATCH not in _codes(analyze_policy(d2, clean_ds))


# ----- P007: non-finite forward --------------------------------------------------------


def test_nan_weights_fire_p007_and_skip_the_rest(tmp_path, clean_ckpt, clean_ds, robot):
    d = _copy_ckpt(clean_ckpt, tmp_path)
    w = load_file(str(d / "model.safetensors"))
    w["model.action_head.bias"] = w["model.action_head.bias"] * float("nan")
    save_file(w, str(d / "model.safetensors"))
    findings = analyze_policy(d, clean_ds, robot=robot)
    p007 = [x for x in findings if x.code == NONFINITE_ACTION]
    assert len(p007) == 1 and p007[0].severity == "error"
    # the other behavioral checks are skipped — their math is garbage on NaN
    assert not _codes(findings) & {ACTION_COLLAPSE, DOF_ACTION_COLLAPSE, DEAD_INPUT, ACTION_SATURATION}


# ----- P008: chunk-config anomalies -----------------------------------------------------


def test_chunk_anomaly_skips_the_model_load(tmp_path, clean_ckpt, clean_ds):
    """n_action_steps > chunk_size crashes lerobot's own config parser — the
    debugger must diagnose it statically and never attempt the load (this test
    completing without an exception IS the proof)."""
    d = _copy_ckpt(clean_ckpt, tmp_path)
    cj = json.loads((d / "config.json").read_text())
    cj["n_action_steps"] = 16  # > chunk_size=8
    (d / "config.json").write_text(json.dumps(cj))
    findings = analyze_policy(d, clean_ds)
    p008 = [x for x in findings if x.code == CHUNK_CONFIG_ANOMALY]
    assert len(p008) == 1 and p008[0].severity == "error"
    assert "16" in p008[0].message and "8" in p008[0].message
    # behavioral checks never ran
    assert not _codes(findings) & {ACTION_COLLAPSE, DEAD_INPUT}


def test_chunk_anomaly_ensembling_misconfig(tmp_path, clean_ckpt):
    d = _copy_ckpt(clean_ckpt, tmp_path)
    cj = json.loads((d / "config.json").read_text())
    cj["temporal_ensemble_coeff"] = 0.01  # requires n_action_steps == 1; ours is 4
    (d / "config.json").write_text(json.dumps(cj))
    findings = analyze_policy(d)
    assert any(
        x.code == CHUNK_CONFIG_ANOMALY and x.severity == "error" and "ensembl" in x.message
        for x in findings
    )


def test_chunk_anomaly_replans_every_tick_is_info(tmp_path, clean_ckpt, clean_ds):
    d = _copy_ckpt(clean_ckpt, tmp_path)
    cj = json.loads((d / "config.json").read_text())
    cj["n_action_steps"] = 1  # valid but wasteful: full forward every tick
    (d / "config.json").write_text(json.dumps(cj))
    findings = analyze_policy(d, clean_ds)
    p008 = [x for x in findings if x.code == CHUNK_CONFIG_ANOMALY]
    assert len(p008) == 1 and p008[0].severity == "info"
    # info-grade P008 does NOT block the load; the behavioral negative holds
    assert not _codes(findings) - {CHUNK_CONFIG_ANOMALY}


# ----- determinism / error paths ----------------------------------------------------------


def test_debugger_determinism_exact(zeroed_ckpt, clean_ds, robot):
    a = analyze_policy(zeroed_ckpt, clean_ds, robot=robot)
    b = analyze_policy(zeroed_ckpt, clean_ds, robot=robot)
    assert a == b  # frozen dataclasses, exact equality
    assert json.dumps([f.to_dict() for f in a]) == json.dumps([f.to_dict() for f in b])


def test_debugger_error_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_policy(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="config.json"):
        analyze_policy(empty)


# ----- autopsy -----------------------------------------------------------------------------


def test_autopsy_defective_dataset_leads_the_verdict(tmp_path_factory, robot):
    """Defective data + a policy matched to it: the verdict must lead with the
    dataset (upstream causes first), and the debugger must NOT blame the policy
    for a dof the DATA never moves (the P002 gate)."""
    ds = _write_defective_dataset(tmp_path_factory.mktemp("defective") / "ds", robot)
    ck = _make_ckpt(tmp_path_factory.mktemp("ckpt_def"), ds)
    rep = autopsy(ck, ds)
    assert isinstance(rep, AutopsyReport)
    assert rep.dataset["clean"] is False
    assert {"D001", "D004", "D009", "D011"} <= {f["code"] for f in rep.dataset["findings"]}
    assert rep.eval_result is None and rep.latency is None
    # the data-dead dof 1 is a DATASET problem, not a policy one
    assert 1 not in {f.dof for f in rep.policy_findings if f.code == DOF_ACTION_COLLAPSE}
    v = rep.verdict.lower()
    assert "dataset" in v and v.index("dataset") < v.index("policy")
    txt = rep.render_text()
    assert "VERDICT" in txt and "dataset doctor (D)" in txt and "policy debugger (P)" in txt
    payload = json.loads(rep.to_json())
    assert payload["verdict"] == rep.verdict
    assert payload["eval"] is None and payload["latency"] is None


def test_autopsy_clean_pair_full_sections(clean_ckpt, clean_ds, robot):
    """robot+task unlock the E and L sections; the merged report carries all
    four, and the clean D/P sections say so."""
    q_probe = [0.3] * DOF
    pose = robot.fk(q_probe, "flange")
    target = [pose[0][3], pose[1][3], pose[2][3]]
    task = reach_eval_task(robot, "flange", target, tol=0.1, max_steps=6, fps=FPS)
    rep = autopsy(
        clean_ckpt,
        clean_ds,
        robot=robot,
        task=task,
        cfg=EvalConfig(n_episodes=2, base_seed=0),
        profile_ticks=8,
    )
    assert rep.dataset["clean"] is True
    assert rep.policy_findings == []
    assert rep.eval_result is not None and rep.eval_result.n_episodes == 2
    assert rep.latency is not None and rep.latency.ticks == 8
    assert rep.verdict.lower().startswith("no defects found") or "clean" in rep.verdict.lower()
    assert "closed-loop" in rep.verdict
    txt = rep.render_text()
    assert "closed-loop eval (E)" in txt and "deploy latency (L)" in txt
    payload = json.loads(rep.to_json())
    assert payload["eval"]["n_episodes"] == 2
    assert payload["latency"]["ticks"] == 8


def test_autopsy_verdict_determinism_without_timing(tmp_path_factory, robot):
    ds = _write_defective_dataset(tmp_path_factory.mktemp("defective2") / "ds", robot)
    ck = _make_ckpt(tmp_path_factory.mktemp("ckpt_def2"), ds)
    a, b = autopsy(ck, ds), autopsy(ck, ds)
    assert a.verdict == b.verdict
    assert a.to_json() == b.to_json()  # byte-identical without the L-section


# ----- CLI (main() called directly — no subprocess) -----------------------------------------


def test_cli_debug_clean_and_broken(capsys, clean_ckpt, zeroed_ckpt, clean_ds):
    rc = main(["debug", str(clean_ckpt), "--dataset", str(clean_ds), "--urdf", URDF, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0 and payload["findings"] == []

    rc = main(["debug", str(zeroed_ckpt), "--dataset", str(clean_ds), "--json"])
    payload = json.loads(capsys.readouterr().out)
    codes = {f["code"] for f in payload["findings"]}
    assert rc == 1 and ACTION_COLLAPSE in codes  # error-severity finding -> exit 1

    # human rendering path
    rc = main(["debug", str(clean_ckpt), "--dataset", str(clean_ds)])
    assert rc == 0 and "no findings" in capsys.readouterr().out


def test_cli_autopsy_smoke(capsys, clean_ckpt, clean_ds):
    rc = main(["autopsy", str(clean_ckpt), str(clean_ds), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["dataset"]["clean"] is True and payload["policy_findings"] == []
    assert payload["verdict"]


def test_cli_eval_smoke(capsys, clean_ckpt, robot):
    pose = robot.fk([0.3] * DOF, "flange")
    rc = main(
        [
            "eval",
            str(clean_ckpt),
            "--urdf",
            URDF,
            "--frame",
            "flange",
            "--target",
            str(pose[0][3]),
            str(pose[1][3]),
            str(pose[2][3]),
            "--episodes",
            "2",
            "--max-steps",
            "5",
            "--fps",
            str(FPS),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_episodes"] == 2 and len(payload["episodes"]) == 2
    # eval findings are warn-grade — never an error exit
    assert rc == 0


def test_cli_profile_smoke(capsys, clean_ckpt):
    rc = main(["profile", str(clean_ckpt), "--urdf", URDF, "--ticks", "8", "--fps", str(FPS), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ticks"] == 8 and payload["fps"] == FPS and "achievable_hz" in payload
    # exit code is self-consistent with the report (timing-dependent findings)
    has_error = any(f["severity"] == "error" for f in payload["findings"])
    assert rc == (1 if has_error else 0)
