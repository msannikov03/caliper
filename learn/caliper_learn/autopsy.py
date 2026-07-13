"""The autopsy: dataset + policy in, ONE report with a verdict out.

`autopsy(policy_dir, dataset_root, ...)` runs every W1/W2 diagnostic that
applies and merges them into a single `AutopsyReport`:

- D-section  `caliper.data_doctor` on the dataset (LeRobotDataset v3.0 —
  convert v2.x first; the doctor's own error says how).
- P-section  `debugger.analyze_policy` on the checkpoint, dataset-aware.
- E-section  `eval.evaluate` — seeded closed-loop success rate with a Wilson
  95% interval. Only when `robot` AND `task` are given (rollouts need a sim).
- L-section  `profile.profile_rollout` — per-stage latency + the honest
  achievable Hz, on a fresh `caliper.ControlLoop`. Same gating.

The VERDICT paragraph is template-based and honest: it leads with the most
severe finding class (dataset errors predict training failure and outrank
everything — fix upstream first), names the concrete policy pathologies in
plain English (which dofs are dead, whether stats mismatch), and only then
reports the closed-loop numbers. Ties break toward the dataset: data problems
cause policy problems, not the other way around.

Rendering: `AutopsyReport.render_text()` / `.to_json()` (sorted keys). The
D/P/E sections are deterministic for a given (dataset, checkpoint, task)
triple; the L-section is wall-clock timing and is honestly not.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional

from .debugger import PolicyFinding, analyze_policy, render_policy_findings
from .eval import EvalConfig, EvalResult, EvalTask
from .eval import render_text as render_eval
from .profile import LatencyReport

# Severity spellings differ across the doctors (Rust data_doctor says
# "warning", the Python doctors say "warn") — rank them together.
_RANK = {"error": 0, "warn": 1, "warning": 1, "info": 2}

# One plain-English clause per policy code, for the verdict paragraph.
_P_CLAUSES = {
    "P001": "its actions have collapsed to a constant",
    "P002": "it never moves dof(s) {dofs} that the data moves",
    "P003": "its actions saturate the joint limits on dof(s) {dofs}",
    "P004": "its normalization stats do not match this dataset",
    "P005": "its training cadence disagrees with the dataset fps",
    "P006": "it ignores input joint(s) {dofs}",
    "P007": "its forward pass produces non-finite actions",
    "P008": "its chunk configuration is inconsistent",
}


@dataclass
class AutopsyReport:
    """Everything the autopsy learned, one object. `dataset` is the raw
    `caliper.data_doctor` dict; `eval_result`/`latency` are None when no
    robot+task were supplied (the report says so instead of guessing)."""

    policy_dir: str
    dataset_root: str
    verdict: str
    dataset: dict
    policy_findings: list[PolicyFinding]
    eval_result: Optional[EvalResult] = None
    latency: Optional[LatencyReport] = None

    def to_dict(self) -> dict:
        return {
            "policy_dir": self.policy_dir,
            "dataset_root": self.dataset_root,
            "verdict": self.verdict,
            "dataset": self.dataset,
            "policy_findings": [f.to_dict() for f in self.policy_findings],
            "eval": asdict(self.eval_result) if self.eval_result is not None else None,
            "latency": self.latency.to_dict() if self.latency is not None else None,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    def render_text(self) -> str:
        d = self.dataset
        out = [
            "== Caliper autopsy ==",
            f"policy:  {self.policy_dir}",
            f"dataset: {self.dataset_root}",
            "",
            f"VERDICT: {self.verdict}",
            "",
            f"-- dataset doctor (D) — {d['total_episodes']} episodes, "
            f"{d['total_frames']} frames @ {d['fps']} fps --",
        ]
        if d["findings"]:
            for f in d["findings"]:
                anchor = "".join(
                    [
                        f" feature={f['feature']}" if f.get("feature") is not None else "",
                        f" episode={f['episode']}" if f.get("episode") is not None else "",
                        f" dof={f['dof']}" if f.get("dof") is not None else "",
                    ]
                )
                out.append(f"  [{f['code']}] ({f['severity']}){anchor} {f['message']}")
                if f.get("fix_hint"):
                    out.append(f"         fix: {f['fix_hint']}")
        else:
            out.append("  clean — no findings.")
        out += ["", "-- policy debugger (P) --", render_policy_findings(self.policy_findings).rstrip()]
        out += ["", "-- closed-loop eval (E) --"]
        if self.eval_result is not None:
            out.append(render_eval(self.eval_result))
        else:
            out.append("  not run — pass robot= and task= for seeded rollouts.")
        out += ["", "-- deploy latency (L) --"]
        if self.latency is not None:
            out.append(self.latency.render_text().rstrip())
        else:
            out.append("  not run — pass robot= and task= to profile the control loop.")
        return "\n".join(out) + "\n"


def _count(findings, key) -> tuple[int, int]:
    """(errors, warnings) over dict- or dataclass-shaped findings."""
    sevs = [key(f) for f in findings]
    return (
        sum(1 for s in sevs if _RANK.get(s, 2) == 0),
        sum(1 for s in sevs if _RANK.get(s, 2) == 1),
    )


def _section_rank(err: int, warn: int) -> int:
    return 0 if err else (1 if warn else 3)


def _policy_clauses(findings: list[PolicyFinding]) -> list[str]:
    clauses = []
    for code in sorted({f.code for f in findings}):
        dofs = sorted({f.dof for f in findings if f.code == code and f.dof is not None})
        clause = _P_CLAUSES.get(code, f"it trips check {code}")
        clauses.append(clause.format(dofs=dofs))
    return clauses


def _verdict(
    dataset: dict,
    policy_findings: list[PolicyFinding],
    eval_result: Optional[EvalResult],
    latency: Optional[LatencyReport],
) -> str:
    d_err, d_warn = _count(dataset["findings"], lambda f: f["severity"])
    p_err, p_warn = _count(policy_findings, lambda f: f.severity)
    d_codes = sorted({f["code"] for f in dataset["findings"]})

    if dataset["findings"]:
        d_clause = (
            f"the dataset has {d_err} error(s) and {d_warn} warning(s) "
            f"({', '.join(d_codes)}) that predict training failure"
        )
    else:
        d_clause = "the dataset is clean"
    if policy_findings:
        extra = "additionally " if dataset["findings"] else ""
        p_clause = f"the policy {extra}has issues: " + "; ".join(
            _policy_clauses(policy_findings)
        )
    else:
        p_clause = "the policy checks are clean"

    # Lead with the most severe section; ties go to the dataset (upstream first).
    sentences = (
        [d_clause, p_clause]
        if _section_rank(d_err, d_warn) <= _section_rank(p_err, p_warn)
        else [p_clause, d_clause]
    )

    if eval_result is not None:
        e = eval_result
        clause = (
            f"closed-loop: {e.n_success}/{e.n_episodes} episodes succeeded "
            f"(95% CI [{e.ci95_low:.2f}, {e.ci95_high:.2f}])"
        )
        if e.n_success == 0:
            clause += " — the policy never solved the task"
        sentences.append(clause)
    if latency is not None:
        if any(f.code == "L001" for f in latency.findings):
            sentences.append(
                f"the deploy loop cannot hold {latency.fps} Hz "
                f"(achievable ~{latency.achievable_hz:.0f} Hz)"
            )
        else:
            sentences.append(f"the deploy loop holds {latency.fps} Hz with headroom")

    if not dataset["findings"] and not policy_findings:
        sentences[0] = "no defects found — dataset and policy checks are both clean"
        del sentences[1]
    verdict = "; ".join(sentences) + "."
    return verdict[0].upper() + verdict[1:]


def autopsy(
    policy_dir,
    dataset_root,
    robot=None,
    task: Optional[EvalTask] = None,
    cfg: Optional[EvalConfig] = None,
    *,
    profile_ticks: int = 100,
) -> AutopsyReport:
    """Run the full post-mortem on a (checkpoint, dataset) pair.

    - `robot` + `task` (an `EvalTask`, e.g. `reach_eval_task(...)`) unlock the
      E-section (seeded rollouts, `cfg` = `EvalConfig`, default 20 episodes)
      and the L-section (`profile_ticks` ticks at `task.fps` on a fresh
      `ControlLoop` started at q=0 — the robot needs inertial data).
    - `robot` alone also unlocks the debugger's joint-limit saturation check.

    The policy is loaded ONCE (safetensors-only hub loader) and shared by the
    E and L sections; the P-section performs its own gated load so that a
    statically-broken checkpoint (P008) is still reported instead of crashing.
    Raises `ValueError` if the dataset cannot be read by the dataset doctor
    (v3.0 on-disk format) — an unreadable dataset has no honest autopsy.
    """
    import caliper  # lazy runtime dep (built via maturin)

    dataset = caliper.data_doctor(str(dataset_root))
    policy_findings = analyze_policy(policy_dir, dataset_root, robot=robot)

    eval_result = None
    latency = None
    if robot is not None and task is not None:
        from .eval import evaluate  # lazy: pulls in mujoco via VecSimEnv
        from .hub import load_lerobot_policy  # lazy: pulls in lerobot
        from .profile import profile_rollout

        policy = load_lerobot_policy(policy_dir)
        eval_result = evaluate(policy, task, cfg if cfg is not None else EvalConfig())
        loop = caliper.ControlLoop(robot, dt=1.0 / task.fps, start=[0.0] * int(robot.ndof))
        latency = profile_rollout(policy, loop, ticks=profile_ticks, fps=task.fps)

    return AutopsyReport(
        policy_dir=str(policy_dir),
        dataset_root=str(dataset_root),
        verdict=_verdict(dataset, policy_findings, eval_result, latency),
        dataset=dataset,
        policy_findings=policy_findings,
        eval_result=eval_result,
        latency=latency,
    )
