"""Latency profiler for the deploy loop: is the requested control rate honest?

`profile_rollout` drives a policy through the same three stages as
`runner.run_policy` (obs build -> predict -> `step_with_target`) and times each
stage per tick with `time.perf_counter_ns`. The headline is deliberately
pessimistic: `achievable_hz = 1 / p95(per-tick total)` — the rate the loop holds
on 95% of ticks, not the average that hides the spikes.

Chunk-awareness (the spike that matters): lerobot-style `select_action` pops an
internal action queue and only re-runs the network every `n_action_steps` ticks
(see `hub.LoadedPolicy`), so mean inference time is a lie — the refill tick is
the one that must fit the budget. Refill ticks are identified from the policy's
chunk config (`n_action_steps`) when available, else detected from timing
bimodality, and their inference p95 is reported SEPARATELY from pop-tick p95.

Overhead accounting: the timing scaffold itself (four `perf_counter_ns` calls +
list appends per tick) is measured on an empty loop before profiling and its
median per-tick cost is subtracted from every tick's total (clamped at zero), so
the report charges the policy and the engine, not the profiler. Per-stage times
carry at most one timestamp call (~tens of ns) of un-subtracted overhead.

Findings follow the caliper-doctor pattern: plain-English messages with stable
codes (`L001` budget exceeded, `L002` inference dominates, `L003` high jitter)
and a `fix_hint` each. This is a profiler, not a deploy path — it does not
validate actions or record datasets; use `runner.run_policy` for real rollouts.
No RNG is used anywhere in this module (timings vary; structure does not).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

# Stable finding codes (string-typed on Finding so reports serialize plainly).
BUDGET_EXCEEDED = "L001"
INFERENCE_DOMINATES = "L002"
HIGH_JITTER = "L003"

# L001: fraction of ticks over the fps budget above which the rate is dishonest
# (5% ~ "the p95 tick does not fit the budget").
_L001_FRAC = 0.05
# L002: inference must be the majority of the median tick AND the tick must be a
# material fraction of the budget — a 5 us loop that is "90% inference" is fine.
_L002_SHARE = 0.6
_L002_MIN_BUDGET_FRAC = 0.2
# L003: tick-period std beyond max(25% of budget, 1 ms) means unstable cadence.
_L003_BUDGET_FRAC = 0.25
_L003_FLOOR_S = 1e-3
# Bimodality: a refill spike must exceed BOTH 3x the median inference time and
# median + 200 us (absolute floor so scheduler noise on a microsecond-fast
# policy is never classified as a chunk refill).
_BIMODAL_FACTOR = 3.0
_BIMODAL_FLOOR_NS = 200_000

_SEVERITY_RANK = {"error": 0, "warn": 1, "info": 2}


@dataclass(frozen=True)
class Finding:
    """One profiler finding: stable `code`, plain-English `message` naming the
    consequence, `fix_hint` saying what to do about it (doctor pattern)."""

    code: str
    severity: str  # "error" | "warn" | "info"
    message: str
    fix_hint: str | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "fix_hint": self.fix_hint,
        }


@dataclass(frozen=True)
class StageStats:
    """Wall-time percentiles for one stage, in seconds."""

    p50: float
    p95: float
    p99: float
    max: float

    @classmethod
    def from_ns(cls, samples_ns) -> "StageStats":
        a = np.asarray(samples_ns, dtype=np.float64) * 1e-9
        return cls(
            p50=float(np.percentile(a, 50)),
            p95=float(np.percentile(a, 95)),
            p99=float(np.percentile(a, 99)),
            max=float(a.max()),
        )

    def to_dict(self) -> dict:
        return {"p50": self.p50, "p95": self.p95, "p99": self.p99, "max": self.max}


@dataclass(frozen=True)
class ChunkStats:
    """Refill-vs-pop split of the inference stage for chunked policies.

    `refill` is the inference time on ticks where the action queue is (expected
    to be) refilled by a network forward pass — the spike that must fit the
    budget. `pop` is everything else (queue pops, near-free). `source` says how
    refill ticks were identified: "config" (the policy's `n_action_steps`) or
    "bimodal" (detected from the timing distribution).
    """

    source: str  # "config" | "bimodal"
    period: int | None  # ticks between refills (None if undetermined)
    n_refill_ticks: int
    refill: StageStats
    pop: StageStats | None  # None when every measured tick was a refill

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "period": self.period,
            "n_refill_ticks": self.n_refill_ticks,
            "refill": self.refill.to_dict(),
            "pop": self.pop.to_dict() if self.pop is not None else None,
        }


@dataclass
class LatencyReport:
    """Per-stage latency percentiles + the honest achievable rate + findings."""

    fps: int
    ticks: int
    budget_s: float  # 1 / fps, the per-tick deadline
    achievable_hz: float  # 1 / p95(per-tick total) — the honest headline
    jitter_s: float  # std of the tick period (start-to-start)
    frac_over_budget: float  # fraction of ticks whose total exceeded budget_s
    overhead_s: float  # instrumentation baseline subtracted from each total
    stages: dict[str, StageStats] = field(default_factory=dict)
    chunk: ChunkStats | None = None
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fps": self.fps,
            "ticks": self.ticks,
            "budget_s": self.budget_s,
            "achievable_hz": self.achievable_hz,
            "jitter_s": self.jitter_s,
            "frac_over_budget": self.frac_over_budget,
            "overhead_s": self.overhead_s,
            "stages": {name: s.to_dict() for name, s in self.stages.items()},
            "chunk": self.chunk.to_dict() if self.chunk is not None else None,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def render_text(self) -> str:
        ms = 1e3
        out = [
            f"Latency profile — {self.ticks} ticks @ {self.fps} Hz "
            f"(budget {self.budget_s * ms:.3f} ms/tick)",
            f"  achievable: ~{self.achievable_hz:.0f} Hz (1 / p95 tick time); "
            f"{self.frac_over_budget:.1%} of ticks over budget",
            f"  jitter (std of tick period): {self.jitter_s * ms:.3f} ms; "
            f"instrumentation overhead subtracted: {self.overhead_s * ms:.4f} ms/tick",
            f"  {'stage':<12}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}{'max ms':>10}",
        ]
        for name in ("obs_build", "inference", "step", "total"):
            s = self.stages[name]
            out.append(
                f"  {name:<12}{s.p50 * ms:>10.3f}{s.p95 * ms:>10.3f}"
                f"{s.p99 * ms:>10.3f}{s.max * ms:>10.3f}"
            )
        if self.chunk is not None:
            c = self.chunk
            period = f"every {c.period} ticks" if c.period is not None else "period undetermined"
            pop95 = f"{c.pop.p95 * ms:.3f} ms" if c.pop is not None else "n/a"
            out.append(
                f"  chunk queue ({c.source}): refills {period} ({c.n_refill_ticks} seen) — "
                f"refill p95 {c.refill.p95 * ms:.3f} ms vs pop p95 {pop95}"
            )
        if not self.findings:
            out.append(f"no findings — the loop holds {self.fps} Hz with headroom.")
        else:
            out.append(f"findings ({len(self.findings)}):")
            for f in self.findings:
                out.append(f"  [{f.code}] ({f.severity}) {f.message}")
                if f.fix_hint:
                    out.append(f"         fix: {f.fix_hint}")
        return "\n".join(out) + "\n"


def _instrumentation_baseline(ticks: int) -> float:
    """Median per-tick cost (ns) of the timing scaffold itself: the same four
    `perf_counter_ns` calls + appends a real tick pays, on an empty loop."""
    t0s, samples = [], []
    for _ in range(max(ticks, 64)):
        t0 = time.perf_counter_ns()
        t1 = time.perf_counter_ns()  # noqa: F841 — mirrors the real loop's shape
        t2 = time.perf_counter_ns()  # noqa: F841
        t3 = time.perf_counter_ns()
        t0s.append(t0)
        samples.append(t3 - t0)
    return float(np.median(samples))


def _default_obs_builder(policy):
    """Mirror the deploy paths: dict observations for Hub-style policies (every
    declared state-like feature <- measured q, as in `runner.default_obs_builder`),
    a bare float32 q vector otherwise."""
    config = getattr(policy, "config", None)
    if config is not None and getattr(config, "state_feature_names", None):
        from .runner import default_obs_builder

        return default_obs_builder(policy)
    return lambda cl: np.asarray(cl.q, dtype=np.float32)


def _detect_chunk(inference_ns: list[int], policy, warmup: int) -> ChunkStats | None:
    """Split inference times into refill vs pop ticks.

    Prefers the policy's own chunk config (`n_action_steps` > 1: after reset(),
    lerobot's queue refills on tick 0 and then every n ticks — measured tick k is
    absolute tick k + warmup). Falls back to bimodality: refill ticks are the
    spikes far above the median (see `_BIMODAL_FACTOR` / `_BIMODAL_FLOOR_NS`).
    Returns None for policies with no detectable chunking.
    """
    n = getattr(policy, "n_action_steps", None)
    if isinstance(n, int) and n > 1:
        refill_idx = [k for k in range(len(inference_ns)) if (k + warmup) % n == 0]
        source, period = "config", n
    else:
        med = float(np.median(inference_ns))
        cut = max(_BIMODAL_FACTOR * med, med + _BIMODAL_FLOOR_NS)
        refill_idx = [k for k, v in enumerate(inference_ns) if v > cut]
        if not refill_idx or len(refill_idx) == len(inference_ns):
            return None  # unimodal: no chunking visible in the timings
        source = "bimodal"
        diffs = np.diff(refill_idx)
        period = int(np.bincount(diffs).argmax()) if len(diffs) else None
    refill = [inference_ns[k] for k in refill_idx]
    pop = [v for k, v in enumerate(inference_ns) if k not in set(refill_idx)]
    if not refill:
        return None  # config period longer than the profiled window
    return ChunkStats(
        source=source,
        period=period,
        n_refill_ticks=len(refill),
        refill=StageStats.from_ns(refill),
        pop=StageStats.from_ns(pop) if pop else None,
    )


def _make_findings(report: LatencyReport) -> list[Finding]:
    ms = 1e3
    findings: list[Finding] = []
    total, inf = report.stages["total"], report.stages["inference"]
    budget = report.budget_s

    if report.frac_over_budget > _L001_FRAC:
        findings.append(
            Finding(
                code=BUDGET_EXCEEDED,
                severity="error",
                message=(
                    f"the {report.fps} Hz budget ({budget * ms:.3f} ms/tick) was exceeded on "
                    f"{report.frac_over_budget:.1%} of ticks; p95 tick time is "
                    f"{total.p95 * ms:.3f} ms ({total.p95 / budget:.1f}x the budget) — the max "
                    f"sustainable rate is ~{report.achievable_hz:.0f} Hz"
                ),
                fix_hint=(
                    f"run the loop at <= {report.achievable_hz:.0f} Hz (and retrain/collect at "
                    "that fps — deploy cadence must match collection), or cut the dominant "
                    "stage in the table above"
                ),
            )
        )

    share = inf.p50 / total.p50 if total.p50 > 0 else 0.0
    if share > _L002_SHARE and total.p95 > _L002_MIN_BUDGET_FRAC * budget:
        findings.append(
            Finding(
                code=INFERENCE_DOMINATES,
                severity="info",
                message=(
                    f"inference dominates the tick: {share:.0%} of the median tick "
                    f"({inf.p50 * ms:.3f} of {total.p50 * ms:.3f} ms); obs build and engine "
                    "step are not the bottleneck"
                ),
                fix_hint=(
                    "shrink or torch.compile the model, or for chunked policies raise "
                    "n_action_steps so the forward pass amortizes over more ticks (watch the "
                    "refill p95 — that single tick still has to fit the budget)"
                ),
            )
        )

    jitter_cut = max(_L003_BUDGET_FRAC * budget, _L003_FLOOR_S)
    if report.jitter_s > jitter_cut:
        findings.append(
            Finding(
                code=HIGH_JITTER,
                severity="warn",
                message=(
                    f"tick period jitter (std) is {report.jitter_s * ms:.3f} ms against a "
                    f"{budget * ms:.3f} ms budget — the loop cadence is unstable even if the "
                    "average holds"
                ),
                fix_hint=(
                    "look for periodic spikes first (chunk refills — see the refill/pop "
                    "split), then background load and GC pauses; pin the process or lower "
                    "the fps until the period stabilizes"
                ),
            )
        )

    findings.sort(key=lambda f: (_SEVERITY_RANK[f.severity], f.code))
    return findings


def profile_rollout(
    policy,
    control_loop,
    *,
    ticks: int = 200,
    fps: int = 50,
    obs_builder=None,
    warmup: int = 0,
) -> LatencyReport:
    """Profile `policy` driving `control_loop` for `ticks` steps at an `fps` budget.

    - `policy`: a `hub.LoadedPolicy` / anything with `.predict(obs) -> action`
      (`.reset()` is called first when present), or a plain callable
      `obs -> action`.
    - `control_loop`: a ready `caliper.ControlLoop`-like object (needs `.q` and
      `.step_with_target`), or a zero-arg factory returning one.
    - `obs_builder(control_loop) -> obs`: override observation construction;
      defaults to the deploy conventions (see `_default_obs_builder`).
    - `warmup`: ticks executed before measurement starts (excluded from all
      stats; useful to keep one-time lazy init out of the refill numbers).

    Timing methodology: `time.perf_counter_ns` around each stage; the median
    per-tick cost of an empty instrumented loop is measured first and subtracted
    from every tick's total (clamped at zero) — see the module docstring.
    """
    if ticks < 2:
        raise ValueError(f"ticks must be >= 2 to measure a tick period, got {ticks}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    cl = control_loop
    if not hasattr(cl, "step_with_target") and callable(cl):
        cl = cl()
    if not (hasattr(cl, "step_with_target") and hasattr(cl, "q")):
        raise TypeError(
            "control_loop must expose .q and .step_with_target(action) (a "
            "caliper.ControlLoop or a zero-arg factory returning one)"
        )

    predict = policy.predict if hasattr(policy, "predict") else policy
    if not callable(predict):
        raise TypeError("policy must have .predict(obs) or be callable itself")
    if obs_builder is None:
        obs_builder = _default_obs_builder(policy)
    if hasattr(policy, "reset"):
        policy.reset()  # fresh action queue: refill phase starts at absolute tick 0

    overhead_ns = _instrumentation_baseline(ticks)

    for _ in range(warmup):
        cl.step_with_target(np.asarray(predict(obs_builder(cl)), dtype=np.float32).tolist())

    obs_ns: list[int] = []
    inf_ns: list[int] = []
    step_ns: list[int] = []
    t0s: list[int] = []
    for _ in range(ticks):
        t0 = time.perf_counter_ns()
        obs = obs_builder(cl)
        t1 = time.perf_counter_ns()
        action = predict(obs)
        t2 = time.perf_counter_ns()
        cl.step_with_target(np.asarray(action, dtype=np.float32).tolist())
        t3 = time.perf_counter_ns()
        t0s.append(t0)
        obs_ns.append(t1 - t0)
        inf_ns.append(t2 - t1)
        step_ns.append(t3 - t2)

    total_ns = [
        max(o + i + s - overhead_ns, 0.0) for o, i, s in zip(obs_ns, inf_ns, step_ns)
    ]
    budget_s = 1.0 / fps
    budget_ns = budget_s * 1e9
    total_p95_ns = float(np.percentile(total_ns, 95))
    achievable_hz = 1e9 / total_p95_ns if total_p95_ns > 0 else float("inf")
    periods_ns = np.diff(np.asarray(t0s, dtype=np.float64))
    jitter_s = float(np.std(periods_ns)) * 1e-9
    frac_over = float(np.mean([t > budget_ns for t in total_ns]))

    report = LatencyReport(
        fps=fps,
        ticks=ticks,
        budget_s=budget_s,
        achievable_hz=achievable_hz,
        jitter_s=jitter_s,
        frac_over_budget=frac_over,
        overhead_s=overhead_ns * 1e-9,
        stages={
            "obs_build": StageStats.from_ns(obs_ns),
            "inference": StageStats.from_ns(inf_ns),
            "step": StageStats.from_ns(step_ns),
            "total": StageStats.from_ns(total_ns),
        },
        chunk=_detect_chunk(inf_ns, policy, warmup),
    )
    report.findings = _make_findings(report)
    return report
