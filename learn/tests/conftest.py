import pytest

caliper = pytest.importorskip("caliper")
from caliper_learn.collect import collect_demos  # noqa: E402


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    """A small deterministic planner dataset (collide_arm) shared across tests."""
    d = tmp_path_factory.mktemp("ds")
    return collect_demos(str(d), n_episodes=4, seed0=0, fps=50)
