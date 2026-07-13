"""Phase-0 oracle: the two faces (CLI + Python bindings) must agree on the
same engine. Full FK/Jacobian cross-validation vs Pinocchio lands in Phase 1."""
import pathlib
import subprocess

import pytest

import caliper

ROOT = pathlib.Path(__file__).resolve().parents[2]
URDF = ROOT / "oracle" / "fixtures" / "robots" / "toy.urdf"
CLI = ROOT / "target" / "debug" / "caliper"


def test_version():
    assert caliper.version() == "0.1.0"
    assert caliper.__version__ == "0.1.0"


def test_load():
    r = caliper.Robot.from_urdf(str(URDF))
    assert r.name == "toy2"
    assert r.ndof == 2
    assert r.joint_names == ["j1", "j2"]


@pytest.mark.skipif(not CLI.exists(), reason="CLI binary not built")
def test_face_parity():
    """The Python face and the CLI face must report the identical model."""
    out = subprocess.run(
        [str(CLI), "load", str(URDF)], capture_output=True, text=True
    ).stdout
    r = caliper.Robot.from_urdf(str(URDF))
    assert r.name in out
    assert f"dof:   {r.ndof}" in out
    # version identity: both faces report the one workspace crate version
    assert caliper.__version__ in subprocess.run(
        [str(CLI), "--version"], capture_output=True, text=True
    ).stdout


# FK / Jacobian cross-validation vs Pinocchio now lives in test_pinocchio.py
# (gated on the optional `pinocchio` dependency + the extended PyO3 bindings).
