"""`caliper-learn` — the console face of the W2 diagnostics.

Subcommands (each takes `--json` for machine output; human text otherwise):

- `caliper-learn debug   CKPT [--dataset ROOT] [--urdf PATH]`
      the policy deploy debugger (P001..P008).
- `caliper-learn autopsy CKPT ROOT [--urdf PATH [--frame F --target X Y Z]]`
      the full post-mortem: dataset doctor + debugger (+ closed-loop eval and
      latency profile when a robot and a reach task are given).
- `caliper-learn eval    CKPT --urdf PATH --frame F --target X Y Z`
      seeded closed-loop evaluation with Wilson-95 aggregates.
- `caliper-learn profile CKPT --urdf PATH`
      deploy-loop latency profile (honest achievable Hz).
- `caliper-learn coverage ROOT OUT --urdf PATH`
      the doctor→generator loop: replay ROOT + targeted planner episodes
      into OUT, report the before/after bin-occupancy delta (D007).

Exit code: 1 when any error-severity finding was reported, else 0 — so CI can
gate on "the doctor found something red" without parsing output. Heavy deps
(caliper, torch/lerobot, mujoco) load lazily per subcommand; `--help` is
instant.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

# Severity spellings across the doctors: Rust says "warning", Python "warn".
_ERROR = "error"


def _add_task_args(p: argparse.ArgumentParser, *, required: bool) -> None:
    p.add_argument("--urdf", required=required, help="robot URDF (caliper.Robot.from_urdf)")
    p.add_argument("--frame", required=required, help="frame the reach task targets")
    p.add_argument(
        "--target",
        nargs=3,
        type=float,
        required=required,
        metavar=("X", "Y", "Z"),
        help="world-space reach target",
    )
    p.add_argument("--tol", type=float, default=0.05, help="reach success tolerance (m)")
    p.add_argument("--episodes", type=int, default=20, help="seeded eval episodes")
    p.add_argument("--seed", type=int, default=0, help="base seed (episode k = seed + k)")
    p.add_argument("--max-steps", type=int, default=200, help="max control steps per episode")
    p.add_argument("--fps", type=int, default=50, help="control rate (MUST match collection)")


def _robot(urdf: str):
    import caliper  # lazy runtime dep

    return caliper.Robot.from_urdf(urdf)


def _task(args, robot):
    from .eval import reach_eval_task

    return reach_eval_task(
        robot,
        args.frame,
        args.target,
        tol=args.tol,
        max_steps=args.max_steps,
        fps=args.fps,
    )


def _has_error(severities) -> bool:
    return any(s == _ERROR for s in severities)


def _cmd_debug(args) -> int:
    from .debugger import analyze_policy, render_policy_findings

    robot = _robot(args.urdf) if args.urdf else None
    findings = analyze_policy(args.policy_dir, args.dataset, robot=robot)
    if args.json:
        payload = {"policy_dir": args.policy_dir, "findings": [f.to_dict() for f in findings]}
        print(json.dumps(payload, sort_keys=True, indent=2))
    else:
        print(render_policy_findings(findings), end="")
    return 1 if _has_error(f.severity for f in findings) else 0


def _cmd_autopsy(args) -> int:
    from .autopsy import autopsy
    from .eval import EvalConfig

    robot = _robot(args.urdf) if args.urdf else None
    task = None
    if robot is not None and args.frame and args.target:
        task = _task(args, robot)
    rep = autopsy(
        args.policy_dir,
        args.dataset_root,
        robot=robot,
        task=task,
        cfg=EvalConfig(n_episodes=args.episodes, base_seed=args.seed),
        profile_ticks=args.ticks,
    )
    print(rep.to_json(indent=2) if args.json else rep.render_text(), end="\n" if args.json else "")
    sevs = [f["severity"] for f in rep.dataset["findings"]]
    sevs += [f.severity for f in rep.policy_findings]
    if rep.latency is not None:
        sevs += [f.severity for f in rep.latency.findings]
    return 1 if _has_error(sevs) else 0


def _cmd_eval(args) -> int:
    from .eval import EvalConfig, evaluate, render_text, to_json
    from .hub import load_lerobot_policy

    robot = _robot(args.urdf)
    result = evaluate(
        load_lerobot_policy(args.policy_dir),
        _task(args, robot),
        EvalConfig(n_episodes=args.episodes, base_seed=args.seed),
    )
    print(to_json(result, indent=2) if args.json else render_text(result))
    return 1 if _has_error(f.severity for f in result.findings) else 0


def _cmd_profile(args) -> int:
    import caliper  # lazy runtime dep

    from .hub import load_lerobot_policy
    from .profile import profile_rollout

    robot = _robot(args.urdf)
    loop = caliper.ControlLoop(robot, dt=1.0 / args.fps, start=[0.0] * int(robot.ndof))
    report = profile_rollout(
        load_lerobot_policy(args.policy_dir), loop, ticks=args.ticks, fps=args.fps
    )
    print(report.to_json() if args.json else report.render_text(), end="\n" if args.json else "")
    return 1 if _has_error(f.severity for f in report.findings) else 0


def _cmd_coverage(args) -> int:
    from .coverage_gen import generate_coverage

    rep = generate_coverage(
        args.dataset_root,
        _robot(args.urdf),
        args.out_root,
        episodes=args.episodes,
        seed=args.seed,
        bins=args.bins,
    )
    print(rep.to_json(indent=2) if args.json else rep.render_text())
    return 1 if rep.error_findings_after else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caliper-learn",
        description="Caliper learning diagnostics: debug / autopsy / eval / profile.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("debug", help="policy deploy debugger (P001..P008)")
    p.add_argument("policy_dir", help="lerobot Hub checkpoint directory")
    p.add_argument("--dataset", default=None, help="training dataset root (unlocks P002/P004/P005)")
    p.add_argument("--urdf", default=None, help="robot URDF (unlocks P003 saturation)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_debug)

    p = sub.add_parser("autopsy", help="dataset doctor + debugger (+ eval/profile with a task)")
    p.add_argument("policy_dir", help="lerobot Hub checkpoint directory")
    p.add_argument("dataset_root", help="LeRobotDataset v3.0 root")
    _add_task_args(p, required=False)
    p.add_argument("--ticks", type=int, default=100, help="latency-profile ticks")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_autopsy)

    p = sub.add_parser("eval", help="seeded closed-loop evaluation (Wilson-95)")
    p.add_argument("policy_dir", help="lerobot Hub checkpoint directory")
    _add_task_args(p, required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_eval)

    p = sub.add_parser("profile", help="deploy-loop latency profile")
    p.add_argument("policy_dir", help="lerobot Hub checkpoint directory")
    p.add_argument("--urdf", required=True, help="robot URDF (needs inertial data)")
    p.add_argument("--ticks", type=int, default=200)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_profile)

    p = sub.add_parser(
        "coverage",
        help="doctor→generator loop: fill D007 coverage holes with targeted episodes",
    )
    p.add_argument("dataset_root", help="input LeRobotDataset v3.0 root (never mutated)")
    p.add_argument("out_root", help="output dataset root (input replay + new episodes)")
    p.add_argument("--urdf", required=True, help="the dataset's robot URDF")
    p.add_argument("-n", "--episodes", type=int, default=4, help="targeted episodes to add")
    p.add_argument("--seed", type=int, default=0, help="base seed (episode k = seed + k)")
    p.add_argument("--bins", type=int, default=20, help="histogram bins per dof for targeting")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_coverage)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
