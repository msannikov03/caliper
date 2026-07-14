"""The doctor→generator loop: turn `caliper.data_doctor` coverage findings
(D007 — per-dof histogram bins never visited) into TARGETED new demonstrations
that fill them, and prove the fix by re-running the doctor.

`generate_coverage` in one breath:

1. doctor the input dataset (`caliper.data_doctor` — v3.0) and keep its
   per-dof `bin_occupancy` + D007 count as the BEFORE;
2. replay every input episode into a NEW dataset at `out_root` (the input is
   never mutated — append-by-copy, the only honest append for a format whose
   stats/metadata are whole-dataset artifacts);
3. plan new episodes with `collect.py`'s machinery (seeded rejection-sampled
   collision-free start, `caliper.Planner` trajectory, one-step-lookahead
   frames + terminal hold) whose GOALS sit in the least-occupied histogram
   bins — the histogram is recomputed after every episode, so consecutive
   episodes chase different holes instead of piling into the first one;
4. re-run the doctor on the merged output and report the before/after
   occupancy delta — the closed loop, in one `CoverageReport`.

Binning for goal TARGETING uses the robot's joint limits (`collect._bounds`),
not the observed min/max: the point is to reach where the data has never
been, and the observed span of a hole-ridden dataset understates the
reachable one. The REPORTED occupancy is the doctor's own (bins between
observed min/max), because that is the number D007 fires on — the loop
closes against the doctor's metric, not a private one.

Scope: state/action datasets. Image features are refused up front (episode
replay would silently drop the pixels — better loud than lossy).

Deterministic in `seed`: episode `ep` draws from `default_rng(seed + ep)` and
seeds its planner the same way (the `collect_demos` scheme), so reruns produce
byte-identical datasets. torch-free: needs only `caliper` + numpy.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from .collect import _DEFAULT_BOXES, _bounds, _sample_free

STATE_FEATURE = "observation.state"
_D007 = "D007"


@dataclass(frozen=True)
class CoverageReport:
    """Before/after of one doctor→generator pass. Occupancies are the data
    doctor's per-dof `bin_occupancy` (fraction of bins visited between the
    observed min and max); `min_*` is the worst dof — the number that decides
    whether D007 keeps firing."""

    out_root: str
    episodes_replayed: int
    episodes_added: int
    occupancy_before: tuple[float, ...]
    occupancy_after: tuple[float, ...]
    min_occupancy_before: float
    min_occupancy_after: float
    d007_before: int
    d007_after: int
    error_findings_after: int  # doctor errors on the OUTPUT (CI gate signal)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=indent)

    def render_text(self) -> str:
        lines = [
            f"coverage: {self.episodes_replayed} episode(s) replayed + "
            f"{self.episodes_added} targeted episode(s) -> {self.out_root}",
            f"min bin occupancy: {self.min_occupancy_before:.3f} -> "
            f"{self.min_occupancy_after:.3f}  (D007 findings: {self.d007_before} -> "
            f"{self.d007_after})",
            f"{'dof':>4} {'before':>8} {'after':>8}",
        ]
        for j, (b, a) in enumerate(zip(self.occupancy_before, self.occupancy_after)):
            lines.append(f"{j:>4} {b:>8.3f} {a:>8.3f}")
        if self.error_findings_after:
            lines.append(
                f"[ERROR] the output dataset carries {self.error_findings_after} "
                "error-severity doctor finding(s) — run caliper.data_doctor on it"
            )
        return "\n".join(lines)


def _doctor_occupancy(report: dict) -> list[float]:
    feats = report.get("features", {}).get(STATE_FEATURE)
    occ = feats.get("bin_occupancy") if feats else None
    if not occ:
        raise ValueError(
            f"data doctor reported no bin occupancy for '{STATE_FEATURE}' — "
            "the dataset has no frames (the doctor's histogram pass only runs "
            "on non-empty data); collect episodes first"
        )
    return list(occ)


def _count_d007(report: dict) -> int:
    return sum(
        1
        for f in report["findings"]
        if f["code"] == _D007 and f["feature"] == STATE_FEATURE
    )


def _bin_rows(rows: np.ndarray, bounds: np.ndarray, bins: int) -> np.ndarray:
    """(T, ndof) states -> (T, ndof) int bin indices over the joint-limit span."""
    span = bounds[:, 1] - bounds[:, 0]
    idx = np.floor((rows - bounds[:, 0]) / span * bins).astype(np.int64)
    return np.clip(idx, 0, bins - 1)


def _target_goal(rng, hist: np.ndarray, bounds: np.ndarray, cm, max_tries: int) -> list[float]:
    """A collision-free goal aimed at the least-occupied bins.

    Per dof, candidate bins are ranked by occupancy (stable sort → ties go to
    the lower bin, deterministically); each failed try widens the pool by one
    rank every 5 tries, so an unreachable bin combination degrades gracefully
    toward broader sampling instead of spinning forever. Falls back to a
    plain free sample if every try collides (the goal is then untargeted but
    the episode still adds data)."""
    ndof, bins = hist.shape
    widths = (bounds[:, 1] - bounds[:, 0]) / bins
    order = np.argsort(hist, axis=1, kind="stable")  # (ndof, bins), emptiest first
    for t in range(max_tries):
        pool = min(1 + t // 5, bins)
        q = []
        for j in range(ndof):
            b = int(order[j, int(rng.integers(pool))])
            lo = bounds[j, 0] + b * widths[j]
            q.append(float(rng.uniform(lo, lo + widths[j])))
        if not cm.query(q)["collision"]:
            return q
    return _sample_free(rng, bounds, cm, max_tries)


def generate_coverage(
    dataset_root: str | os.PathLike,
    robot,
    out_root: str | os.PathLike,
    *,
    episodes: int = 4,
    seed: int = 0,
    ground: float = -0.1,
    boxes=None,
    bins: int = 20,
    max_goal_tries: int = 200,
    task_template: str = "coverage fill {ep}",
) -> CoverageReport:
    """Fill `dataset_root`'s coverage holes: write input + `episodes` targeted
    planner episodes to a NEW LeRobotDataset v3.0 at `out_root` and report the
    doctor's before/after occupancy (module doc has the full loop).

    `robot` must be the dataset's robot (`ndof` is checked; goal targeting
    bins over ITS joint limits). `seed` follows the `collect_demos` scheme —
    episode `ep` uses `default_rng(seed + ep)` for sampling AND the planner.
    """
    import caliper  # runtime dep (built via maturin), not a packaging dep

    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")
    boxes = _DEFAULT_BOXES if boxes is None else boxes

    before = caliper.data_doctor(str(dataset_root))
    occ_before = _doctor_occupancy(before)

    rd = caliper.DatasetReaderV3.open(str(dataset_root))
    if rd.image_features:
        raise ValueError(
            f"dataset carries image feature(s) {rd.image_features} — episode "
            "replay would drop the pixels; coverage generation supports "
            "state/action datasets only"
        )
    ndof = int(robot.ndof)
    if rd.ndof != ndof:
        raise ValueError(f"dataset ndof={rd.ndof} != robot ndof={ndof}")
    fps = int(rd.fps)

    bounds = _bounds(robot)
    hist = np.zeros((ndof, bins), dtype=np.int64)
    rec = caliper.RecorderV3(robot, str(out_root), fps=fps)

    # 1) replay the input verbatim (never mutate it) and seed the histogram.
    for i in range(rd.total_episodes):
        states, actions, ts = rd.read_episode(i)
        tasks = rd.episode_tasks(i)
        rec.start_episode(tasks[0] if tasks else f"replayed episode {i}")
        for s, a, t in zip(states, actions, ts):
            rec.append(s, a, t)
        rec.finalize_episode()
        for row in _bin_rows(np.asarray(states, dtype=np.float64), bounds, bins):
            hist[np.arange(ndof), row] += 1

    # 2) targeted episodes: goals in the emptiest bins, hist updated per
    # episode so the next one chases the next hole.
    cm = caliper.CollisionModel(robot, ground=ground, boxes=boxes, margin=0.0)
    added = 0
    for ep in range(episodes):
        rng = np.random.default_rng(seed + ep)
        start = _sample_free(rng, bounds, cm, max_goal_tries)
        goal = _target_goal(rng, hist, bounds, cm, max_goal_tries)
        planner = caliper.Planner(robot, ground=ground, boxes=boxes, seed=seed + ep)
        _ts, qs, _qds = planner.plan_trajectory(start, goal, dt=1.0 / fps)
        if len(qs) < 2:
            continue  # degenerate (start==goal); skip, as collect_demos does
        rec.start_episode(task_template.format(ep=ep))
        for k in range(len(qs) - 1):  # state=q_k, action=q_{k+1} (collect.py)
            rec.append(qs[k], qs[k + 1], k / fps)
        last = len(qs) - 1  # terminal hold-at-goal frame
        rec.append(qs[last], qs[last], last / fps)
        rec.finalize_episode()
        added += 1
        for row in _bin_rows(np.asarray(qs, dtype=np.float64), bounds, bins):
            hist[np.arange(ndof), row] += 1

    root = rec.close()

    # 3) close the loop: the doctor's verdict on the merged dataset.
    after = caliper.data_doctor(root)
    occ_after = _doctor_occupancy(after)
    return CoverageReport(
        out_root=root,
        episodes_replayed=rd.total_episodes,
        episodes_added=added,
        occupancy_before=tuple(occ_before),
        occupancy_after=tuple(occ_after),
        min_occupancy_before=min(occ_before),
        min_occupancy_after=min(occ_after),
        d007_before=_count_d007(before),
        d007_after=_count_d007(after),
        error_findings_after=sum(1 for f in after["findings"] if f["severity"] == "error"),
    )
