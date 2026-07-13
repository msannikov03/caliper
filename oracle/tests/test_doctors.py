"""Cross-face oracle for the W1 doctors: asset doctor, dataset doctor, lint.

Everything here goes through the Python face (`caliper.doctor`,
`caliper.data_doctor`, `caliper.lint_path`) — the engines' own Rust suites pin
the per-check math; this file proves the FACES deliver it:

  (a) asset doctor — finds every crafted defect in the doctor_* fixtures
      (positives), reports doctor_clean.urdf spotless (the shared negative),
      and repair=True writes a COPY that compiles via Robot.from_urdf WITH
      has_inertia and re-diagnoses to zero findings; computed inertials are
      LINEAR in the density argument (analytic cross-check through
      gravity_torque differences);
  (b) dataset doctor — flags a writer-crafted defective LeRobotDataset v3.0
      (constant dof D001, echo labels D004, length outlier D009, frozen tail
      D011) and passes a clean sim-collected dataset (Simulator rollouts under
      decorrelated sinusoidal torques, actions from a fixed affine policy);
  (c) trajectory lint — clean on a planned move_j, and flags hand-crafted
      position/velocity violations with the exact analytic margins.

Deterministic by construction: fixed phases, no RNG, and repeated doctor runs
must return identical dicts.

NOTE the repo's oracle/fixtures/robots/unit_cube.stl has INCONSISTENT winding
(signed volume 0 — pinned by the doctor's own Rust tests), so the repair tests
stage doctor_repairable.urdf into a temp dir next to a properly wound cube.
"""

import math
import pathlib

import pytest

import caliper

pytestmark = pytest.mark.skipif(
    not all(hasattr(caliper, f) for f in ("doctor", "data_doctor", "lint_path")),
    reason="caliper lacks the doctor bindings — rebuild (maturin develop)",
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROBOTS = ROOT / "oracle" / "fixtures" / "robots"

FPS = 25
NDOF = 6  # showcase6


def _codes(report):
    return {f["code"] for f in report["findings"]}


# ===== (a) asset doctor =====


def test_doctor_clean_fixture_is_spotless():
    rep = caliper.doctor(str(ROBOTS / "doctor_clean.urdf"))
    assert rep["clean"] is True
    assert rep["findings"] == []
    assert (rep["errors"], rep["warnings"], rep["infos"]) == (0, 0, 0)
    assert rep["repair"] is None


@pytest.mark.parametrize(
    "fixture, expected",
    [
        ("doctor_bad_inertia.urdf", {"A001", "A002"}),
        ("doctor_mesh_missing.urdf", {"A003"}),
        ("doctor_dup_mesh.urdf", {"A003", "A004"}),
        ("doctor_visual_only.urdf", {"A005"}),
        ("doctor_onshape.urdf", {"A010"}),
        ("doctor_mimic.urdf", {"A011", "A012"}),
        ("doctor_zero_axis.urdf", {"A008"}),
        ("doctor_xacro_leftover.urdf", {"A013"}),
    ],
)
def test_doctor_finds_each_crafted_defect(fixture, expected):
    rep = caliper.doctor(str(ROBOTS / fixture))
    assert expected <= _codes(rep), (fixture, rep["findings"])
    assert rep["clean"] is False


def test_doctor_single_finding_fixtures_report_nothing_else():
    # These fixtures document their finding as the ONLY one in the report.
    for fixture, code, severity in [
        ("doctor_visual_only.urdf", "A005", "warn"),
        ("doctor_onshape.urdf", "A010", "info"),
        ("doctor_zero_axis.urdf", "A008", "error"),
    ]:
        rep = caliper.doctor(str(ROBOTS / fixture))
        got = [(f["code"], f["severity"]) for f in rep["findings"]]
        assert got == [(code, severity)], (fixture, rep["findings"])


def test_doctor_mimic_exact_census():
    rep = caliper.doctor(str(ROBOTS / "doctor_mimic.urdf"))
    assert sorted(f["code"] for f in rep["findings"]) == ["A011", "A012", "A012"]
    assert rep["errors"] == 3
    # every finding names the field and carries the machine columns
    for f in rep["findings"]:
        assert f["message"]
        assert isinstance(f["auto_fixable"], bool)


def test_doctor_is_deterministic():
    p = str(ROBOTS / "doctor_bad_inertia.urdf")
    assert caliper.doctor(p) == caliper.doctor(p)


def test_doctor_rejects_bad_args(tmp_path):
    p = str(ROBOTS / "doctor_clean.urdf")
    with pytest.raises(ValueError, match="repair"):
        caliper.doctor(p, out=str(tmp_path / "x.urdf"))  # out without repair
    with pytest.raises(ValueError, match="density"):
        caliper.doctor(p, repair=True, out=str(tmp_path / "y.urdf"), density=0.0)
    assert not (tmp_path / "y.urdf").exists()  # rejected BEFORE writing
    with pytest.raises(ValueError):
        caliper.doctor(str(ROBOTS / "no_such_robot.urdf"))


# --- repair round-trip (staged: see the module docstring's winding note) ---


def _write_cube_stl(path):
    """A properly wound unit cube ([-0.5, 0.5]^3): 12 outward-facing triangles."""
    tris = []
    for axis in range(3):
        for sgn in (-1.0, 1.0):
            u, v = (axis + 1) % 3, (axis + 2) % 3
            if sgn < 0:
                u, v = v, u  # swap keeps the winding outward on the -face

            def corner(du, dv, axis=axis, sgn=sgn, u=u, v=v):
                p = [0.0, 0.0, 0.0]
                p[axis] = 0.5 * sgn
                p[u] = 0.5 * du
                p[v] = 0.5 * dv
                return p

            a, b, c, d = corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)
            tris += [(a, b, c), (a, c, d)]
    lines = ["solid cube"]
    for a, b, c in tris:
        e1 = [b[i] - a[i] for i in range(3)]
        e2 = [c[i] - a[i] for i in range(3)]
        n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ]
        norm = math.sqrt(sum(x * x for x in n)) or 1.0
        n = [x / norm for x in n]
        lines.append(f"  facet normal {n[0]} {n[1]} {n[2]}")
        lines.append("    outer loop")
        for p in (a, b, c):
            lines.append(f"      vertex {p[0]} {p[1]} {p[2]}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid cube")
    path.write_text("\n".join(lines) + "\n")


def _stage_repairable(tmp_path):
    staged = tmp_path / "doctor_repairable.urdf"
    staged.write_text((ROBOTS / "doctor_repairable.urdf").read_text())
    _write_cube_stl(tmp_path / "unit_cube.stl")
    return staged


def test_doctor_repair_round_trip(tmp_path):
    staged = _stage_repairable(tmp_path)
    # the ORIGINAL cannot even compile: j2's <limit> has no velocity=
    with pytest.raises(ValueError):
        caliper.Robot.from_urdf(str(staged))
    out = tmp_path / "repaired.urdf"
    rep = caliper.doctor(str(staged), repair=True, out=str(out))
    # fixture-documented census of the ORIGINAL: 3 errors, 3 warnings
    assert (rep["errors"], rep["warnings"]) == (3, 3), rep["findings"]
    applied = rep["repair"]["applied"]
    assert {a["code"] for a in applied} == {"A001", "A007", "A009", "A014"}
    assert len(applied) == 6  # A001 x2 (l1 box, l2 mesh), A007 x2, A009, A014
    assert rep["repair"]["skipped"] == []
    assert rep["repair"]["mesh_copies"] == []
    assert rep["repair"]["out"] == str(out)
    # the repaired COPY compiles WITH inertia and re-diagnoses spotless
    r = caliper.Robot.from_urdf(str(out))
    assert r.has_inertia
    after = caliper.doctor(str(out))
    assert after["findings"] == [], after["findings"]
    # the input was never modified
    assert staged.read_text() == (ROBOTS / "doctor_repairable.urdf").read_text()


def test_doctor_repair_density_is_linear(tmp_path):
    """Computed inertials scale linearly in density: gravity torques of the
    repaired robots have EQUAL successive differences across 1000/2000/3000
    kg/m^3 (the fixed explicit inertials cancel in the differences)."""
    staged = _stage_repairable(tmp_path)
    # The fixture as-is is gravity-DEGENERATE: the heavy (repaired) links hang
    # off vertical-axis joints (no gravity moment about a gravity-parallel
    # axis), and the one horizontal joint's downstream COM sits on its own
    # axis — gravity_torque is identically zero at ANY mass, hiding density.
    # Tilt j1 horizontal (still non-unit, so the A009 repair keeps firing).
    staged.write_text(staged.read_text().replace('axis xyz="0 0 2"', 'axis xyz="0 2 0"'))
    q = [0.3, 0.7, -0.4]  # bent config: torques generically non-zero
    taus = []
    for density in (1000.0, 2000.0, 3000.0):
        out = tmp_path / f"repaired_{int(density)}.urdf"
        caliper.doctor(str(staged), repair=True, out=str(out), density=density)
        taus.append(caliper.Robot.from_urdf(str(out)).gravity_torque(q))
    d21 = [b - a for a, b in zip(taus[0], taus[1])]
    d32 = [b - a for a, b in zip(taus[1], taus[2])]
    scale = max(max(abs(x) for x in d21), 1.0)
    assert max(abs(x - y) for x, y in zip(d21, d32)) < 1e-8 * scale, (d21, d32)
    assert max(abs(x) for x in d21) > 1e-6  # density actually flows through


def test_doctor_repair_default_out_path(tmp_path):
    staged = _stage_repairable(tmp_path)
    rep = caliper.doctor(str(staged), repair=True)
    expected = tmp_path / "doctor_repairable.repaired.urdf"
    assert rep["repair"]["out"] == str(expected)
    assert expected.exists()
    assert caliper.doctor(str(expected))["findings"] == []


# ===== (b) dataset doctor =====


def _showcase():
    return caliper.Robot.from_urdf(str(ROBOTS / "showcase6.urdf"))


# A fixed affine policy of the state: a deterministic action label, so
# near-identical states NEVER carry contradictory actions (D006-immune, the
# normalization cancels the per-dof scale), while staying far from an echo of
# the state (D004) and spanning a comparable range (D005).
_AFFINE_C = [1.8, -1.6, 1.5, -1.4, 1.7, -1.3]
_AFFINE_B = [0.4, -0.3, 0.25, 0.35, -0.45, 0.3]


def _policy(state):
    return [c * s + b for c, s, b in zip(_AFFINE_C, state, _AFFINE_B)]


def _sim_episode(robot, phase, frames):
    """Torque-driven zero-gravity Simulator rollout at exactly FPS: per-dof
    sinusoidal torques at mutually non-locking frequencies decorrelate the
    dofs (no D008 corridor) and sweep each dof's range repeatedly (D007
    coverage); damping keeps the motion bounded."""
    freqs = [0.37, 0.53, 0.71, 0.89, 1.07, 1.31]
    amps = [0.8, 0.7, 0.6, 0.5, 0.45, 0.4]
    sim = caliper.Simulator(robot, dt=1e-3, gravity=[0.0, 0.0, 0.0], damping=0.8)
    steps = int(round((1.0 / FPS) / 1e-3))
    states, times = [], []
    for k in range(frames):
        t = k / FPS
        tau = [
            a * math.sin(2.0 * math.pi * fr * t + phase + 0.9 * i)
            for i, (a, fr) in enumerate(zip(amps, freqs))
        ]
        sim.set_torque(tau)
        sim.step_n(steps)
        states.append(sim.q)
        times.append(t)
    return states, times


@pytest.fixture(scope="module")
def clean_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("doctor_clean_ds") / "ds"
    r = _showcase()
    rec = caliper.RecorderV3(r, str(root), FPS)
    for e, phase in enumerate((0.0, 1.1, 2.3)):
        states, times = _sim_episode(r, phase, 150)
        rec.start_episode(f"sweep {e}")
        for s, t in zip(states, times):
            rec.append(s, _policy(s), t)
        rec.finalize_episode()
    rec.close()
    return root


@pytest.fixture(scope="module")
def defective_root(tmp_path_factory):
    """Writer-crafted defects: state dof 1 never moves anywhere (D001),
    action == state exactly (D004 echo), episode 4 is 8 frames against a
    median of 60 (D009), and episode 0's last 6 frames are bit-identical
    across every feature (D011). Timestamps stay regular."""
    root = tmp_path_factory.mktemp("doctor_defective_ds") / "ds"
    r = _showcase()
    rec = caliper.RecorderV3(r, str(root), FPS)
    for e in range(5):
        frames = 8 if e == 4 else 60
        rec.start_episode("defective demo")
        for k in range(frames):
            kk = min(k, frames - 6) if e == 0 else k  # freeze episode 0's tail
            arg = 2.0 * math.pi * 0.41 * (kk / FPS) + 0.7 * e
            s = [math.sin(arg + 0.8 * i) for i in range(NDOF)]
            s[1] = 0.5  # this dof never moves in the whole dataset
            rec.append(s, list(s), k / FPS)  # action == state: echo labels
        rec.finalize_episode()
    rec.close()
    return root


def test_data_doctor_passes_clean_sim_collected_dataset(clean_root):
    rep = caliper.data_doctor(str(clean_root))
    assert rep["total_episodes"] == 3
    assert rep["total_frames"] == 450
    assert rep["fps"] == FPS
    assert rep["findings"] == [], rep["findings"]
    assert rep["clean"] is True
    assert set(rep["features"]) == {"observation.state", "action"}
    for feat in rep["features"].values():
        assert feat["dim"] == NDOF
        assert len(feat["mean"]) == NDOF
        # every dof genuinely moved
        assert all(s > 1e-6 for s in feat["std"]), feat["std"]


def test_data_doctor_flags_crafted_defects(defective_root):
    rep = caliper.data_doctor(str(defective_root))
    codes = _codes(rep)
    assert {"D001", "D004", "D009", "D011"} <= codes, sorted(codes)
    assert rep["clean"] is False
    d001 = [f for f in rep["findings"] if f["code"] == "D001"]
    assert any(f["feature"] == "observation.state" and f["dof"] == 1 for f in d001)
    assert all(f["severity"] == "warning" for f in d001)
    d004 = [f for f in rep["findings"] if f["code"] == "D004"]
    assert d004 and all(f["feature"] == "action" for f in d004)
    assert any(f["episode"] == 4 for f in rep["findings"] if f["code"] == "D009")
    assert any(f["episode"] == 0 for f in rep["findings"] if f["code"] == "D011")
    for f in rep["findings"]:
        assert f["message"] and f["fix_hint"]


def test_data_doctor_is_deterministic(defective_root):
    a = caliper.data_doctor(str(defective_root))
    b = caliper.data_doctor(str(defective_root))
    assert a == b


def test_data_doctor_rejects_non_dataset(tmp_path):
    with pytest.raises(ValueError):
        caliper.data_doctor(str(tmp_path / "not_a_dataset"))


# ===== (c) trajectory lint =====


def _toy():
    return caliper.Robot.from_urdf(str(ROBOTS / "toy.urdf"))


def test_lint_path_clean_on_planned_move():
    r = _toy()
    traj = r.move_j([0.0, 0.0], [0.5, -0.5])
    times, q, qd, qdd = traj.sample_uniform(0.01)
    assert caliper.lint_path(r, times, q, qd, qdd) == []


def test_lint_path_flags_position_violation_with_exact_margin():
    r = _toy()  # toy's limits are exactly [-3.14, 3.14] per joint
    times = [0.0, 0.1]
    q = [[0.0, 0.0], [4.0, 0.0]]
    z = [[0.0, 0.0], [0.0, 0.0]]
    findings = caliper.lint_path(r, times, q, z, z)
    assert [f["code"] for f in findings] == ["T001"], findings
    f = findings[0]
    assert f["severity"] == "error"
    assert f["joint"] == 0
    assert f["time"] == 0.1
    assert abs(f["value"] - (3.14 - 4.0)) < 1e-12  # the hand-computed margin
    assert "position limit" in f["message"] and f["fix_hint"]


def test_lint_path_flags_velocity_violation_with_exact_utilization():
    r = _toy()  # toy's velocity limit is 3 rad/s per joint
    times = [0.0, 0.1]
    q = [[0.0, 0.0], [0.5, 0.0]]
    qd = [[0.0, 0.0], [50.0, 0.0]]
    z = [[0.0, 0.0], [0.0, 0.0]]
    findings = caliper.lint_path(r, times, q, qd, z)
    t002 = [f for f in findings if f["code"] == "T002"]
    assert len(t002) == 1, findings
    assert t002[0]["severity"] == "error"
    assert t002[0]["joint"] == 0
    assert abs(t002[0]["value"] - 50.0 / 3.0) < 1e-12
    # explicit limits override the model's own
    lim = caliper.MotionLimits([100.0, 100.0], [1000.0, 1000.0], [10000.0, 10000.0])
    assert caliper.lint_path(r, times, q, qd, z, limits=lim) == []


def test_lint_path_validates_inputs():
    r = _toy()
    z2 = [[0.0, 0.0]]
    with pytest.raises(ValueError, match="one row per sample"):
        caliper.lint_path(r, [0.0, 0.1], z2, z2, z2)
    with pytest.raises(ValueError, match="ndof"):
        caliper.lint_path(r, [0.0], [[0.0, 0.0, 0.0]], z2, z2)
    with pytest.raises(ValueError, match="non-decreasing"):
        caliper.lint_path(r, [0.1, 0.0], z2 * 2, z2 * 2, z2 * 2)
    with pytest.raises(ValueError, match="unknown frame"):
        caliper.lint_path(r, [0.0], z2, z2, z2, frame="nope")
    # empty rows lint clean (total, like the engine)
    assert caliper.lint_path(r, [], [], [], []) == []
