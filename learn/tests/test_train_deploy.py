import numpy as np
import pytest

caliper = pytest.importorskip("caliper")
torch = pytest.importorskip("torch")
from torch.utils.data import TensorDataset

from caliper_learn.checkpoint import load_checkpoint, save_checkpoint
from caliper_learn.collect import collect_demos
from caliper_learn.data import DataConfig, NormStats, make_datasets
from caliper_learn.deploy import StubPolicy, make_obs, rollout_policy
from caliper_learn.policy import build_policy, seed_all
from caliper_learn.train import TrainConfig, fit

URDF = "oracle/fixtures/robots/collide_arm.urdf"


def _two_sample_ds(obs_dim, action_dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(2, obs_dim, generator=g)
    act = torch.randn(2, action_dim, generator=g)
    return TensorDataset(obs, act), obs, act


def test_bc_overfit_smoke():
    seed_all(0)
    obs_dim, action_dim = 6, 3
    ds, _, _ = _two_sample_ds(obs_dim, action_dim)
    p = build_policy("bc_mlp", {"obs_dim": obs_dim, "action_dim": action_dim, "hidden": 64})
    hist = fit(p, ds, cfg=TrainConfig(steps=400, batch_size=2, lr=1e-3, log_every=1, seed=0))
    init, final = hist["train_loss"][0], hist["final_train"]
    assert final < 1e-3, f"BC did not overfit 2 samples: final={final}"
    assert final < 0.05 * init, f"loss barely moved: {init}->{final}"


def test_checkpoint_round_trip(tmp_path):
    seed_all(0)
    obs_dim, action_dim = 6, 3
    ds, _, _ = _two_sample_ds(obs_dim, action_dim)
    # NON-identity normalization so the round-trip actually exercises buffer
    # persistence AND the norm path (identity defaults would prove nothing).
    stats = NormStats(
        in_mean=np.full(obs_dim, 0.7, np.float32),
        in_std=np.full(obs_dim, 1.5, np.float32),
        out_mean=np.full(action_dim, -0.3, np.float32),
        out_std=np.full(action_dim, 2.0, np.float32),
    )
    p = build_policy("bc_mlp", {"obs_dim": obs_dim, "action_dim": action_dim, "hidden": 64}, stats)
    fit(p, ds, cfg=TrainConfig(steps=50, batch_size=2, seed=0))
    obs = np.linspace(-1, 1, obs_dim, dtype=np.float32)
    before = p.predict(obs)
    ck = tmp_path / "p.pt"
    save_checkpoint(p, str(ck))
    q = load_checkpoint(str(ck))
    after = q.predict(obs)
    assert np.allclose(before, after, atol=1e-6), f"{before} != {after}"
    # norm buffers survived AND are the non-identity values we set (not defaults)
    sd_b, sd_a = p.state_dict(), q.state_dict()
    assert not torch.allclose(sd_a["in_mean"], torch.zeros(obs_dim))
    assert not torch.allclose(sd_a["out_std"], torch.ones(action_dim))
    for k in ("in_mean", "in_std", "out_mean", "out_std"):
        assert torch.allclose(sd_b[k], sd_a[k])


def test_act_loss_drops(tiny_dataset):
    seed_all(0)
    train, _, stats, meta = make_datasets(
        DataConfig(root=tiny_dataset, mode="window", history=4, chunk=4, val_frac=0.0)
    )
    p = build_policy(
        "act_lite",
        {
            "obs_dim": meta["obs_dim"],
            "action_dim": meta["action_dim"],
            "d_model": 32,
            "nhead": 2,
            "layers": 1,
            "ff": 32,
            "history": 4,
            "chunk": 4,
        },
        stats,
    )
    hist = fit(p, train, cfg=TrainConfig(steps=300, batch_size=16, lr=1e-3, log_every=1, seed=0))
    init, final = hist["train_loss"][0], hist["final_train"]
    assert final < 0.5 * init, f"ACT loss did not drop: {init}->{final}"


def test_deploy_stub_regulates_and_deterministic():
    robot = caliper.Robot.from_urdf(URDF)
    n = robot.ndof
    goal = [0.2, -0.2, 0.2][:n]
    start = [0.0] * n
    pol = StubPolicy(goal, obs_dim=2 * n)
    r1 = rollout_policy(pol, robot, goal, ticks=1500, dt=1e-3, start=start)
    r2 = rollout_policy(pol, robot, goal, ticks=1500, dt=1e-3, start=start)
    assert r1.states == r2.states, "deploy not deterministic"
    d0 = np.linalg.norm(np.array(start) - np.array(goal))
    dT = np.linalg.norm(np.array(r1.states[-1]) - np.array(goal))
    assert np.isfinite(np.array(r1.states)).all()
    assert dT < 0.25 * d0, f"stub did not regulate: d0={d0} dT={dT}"


def test_e2e_collect_train_deploy(tmp_path):
    seed_all(0)
    root = collect_demos(str(tmp_path / "ds"), n_episodes=4, seed0=0, fps=50)
    train, _, stats, meta = make_datasets(
        DataConfig(root=root, mode="frame", goal_conditioned=True, val_frac=0.0)
    )
    p = build_policy(
        "bc_mlp",
        {"obs_dim": meta["obs_dim"], "action_dim": meta["action_dim"], "hidden": 128},
        stats,
    )
    fit(p, train, cfg=TrainConfig(steps=2000, batch_size=64, lr=1e-3, seed=0))

    # in-distribution goal/start: episode 0's recorded endpoints
    rd = caliper.DatasetReader.open(root)
    s, _a, _t = rd.read_episode(0)
    start, goal = list(s[0]), list(s[-1])
    # deploy at the COLLECTION cadence (dt = 1/fps) so the action's one-step lookahead
    # horizon matches what the policy was trained on (finding: timescale mismatch).
    res = rollout_policy(p, caliper.Robot.from_urdf(URDF), goal, ticks=600, dt=1.0 / 50, start=start)
    assert np.isfinite(np.array(res.states)).all()
    d0 = np.linalg.norm(np.array(start) - np.array(goal))
    dT = np.linalg.norm(np.array(res.states[-1]) - np.array(goal))
    # regulation bar (not just "moved a little"): must close most of the gap.
    assert dT < 0.4 * d0, f"trained BC did not regulate to goal: d0={d0} dT={dT}"


def test_fit_empty_loader_raises():
    obs_dim, action_dim = 6, 3
    one = TensorDataset(torch.zeros(1, obs_dim), torch.zeros(1, action_dim))
    p = build_policy("bc_mlp", {"obs_dim": obs_dim, "action_dim": action_dim, "hidden": 8})
    with pytest.raises(ValueError, match="empty training loader"):
        fit(p, one, cfg=TrainConfig(steps=5, batch_size=64, drop_last=True))


def test_control_loop_last_warn():
    robot = caliper.Robot.from_urdf(URDF)
    n = robot.ndof
    cl = caliper.ControlLoop(robot, dt=1e-3, start=[0.0] * n)
    cl.step_with_target([0.05] * n)
    assert isinstance(cl.last_warn, bool)  # observability getter present


def test_act_deploy_rolling_history_deterministic(tiny_dataset):
    seed_all(0)
    train, _, stats, meta = make_datasets(
        DataConfig(root=tiny_dataset, mode="window", history=4, chunk=4, val_frac=0.0)
    )
    p = build_policy(
        "act_lite",
        {"obs_dim": meta["obs_dim"], "action_dim": meta["action_dim"],
         "d_model": 16, "nhead": 2, "layers": 1, "ff": 16, "history": 4, "chunk": 4},
        stats,
    )
    robot = caliper.Robot.from_urdf(URDF)
    goal = [0.2, -0.2, 0.2][: robot.ndof]
    r1 = rollout_policy(p, robot, goal, ticks=50, dt=1e-3, start=[0.0] * robot.ndof)
    r2 = rollout_policy(p, robot, goal, ticks=50, dt=1e-3, start=[0.0] * robot.ndof)
    assert r1.states == r2.states  # reset() makes ACT rollouts reproducible
    assert np.isfinite(np.array(r1.states)).all()


def test_make_obs_shapes():
    n = 3
    q = np.arange(n, dtype=np.float32)
    goal = np.arange(n, dtype=np.float32) + 10
    assert make_obs(q, goal, n).shape == (n,)
    assert make_obs(q, goal, 2 * n).tolist() == list(range(n)) + list(range(10, 10 + n))
    with pytest.raises(ValueError):
        make_obs(q, goal, n + 1)
