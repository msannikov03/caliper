"""`caliper-learn` — the console face of the W2 diagnostics.

Subcommands (each takes `--json` for machine output; human text otherwise):

- `caliper-learn debug   CKPT [--dataset ROOT] [--urdf PATH]`
      the policy deploy debugger (P001..P008).
- `caliper-learn autopsy CKPT ROOT [--task FILE | --urdf PATH --frame F --target X Y Z]`
      the full post-mortem: dataset doctor + debugger (+ closed-loop eval and
      latency profile when a task is given).
- `caliper-learn eval    CKPT (--task FILE | --urdf PATH --frame F --target X Y Z)`
      seeded closed-loop evaluation with Wilson-95 aggregates.
- `caliper-learn profile CKPT --urdf PATH`
      deploy-loop latency profile (honest achievable Hz).
- `caliper-learn coverage ROOT OUT (--urdf PATH | --task FILE)`
      the doctor→generator loop: replay ROOT + targeted planner episodes
      into OUT, report the before/after bin-occupancy delta (D007).
- `caliper-learn drive CKPT (--urdf PATH | --task FILE)`
      policy-in-the-loop: serve the checkpoint over the stdio JSONL bridge so
      Studio's live sim can be driven by it (see `bridge`). NOT a human
      subcommand — stdout is the protocol wire; Studio spawns it.

THE TWO TASK FORMS. `--urdf/--frame/--target` is the built-in REACH task:
success = the arm's tip got within `--tol` of a point. `--task FILE` is a
`*.caliper-task.json` ARTIFACT: robot + scene (props) + start pose + rate +
time budget + a success predicate over the SCENE ("the cube is in the bin").
The artifact is why there is no bare `--success` flag — a predicate names props
that only the file it ships with can put in the world. `--fps` / `--max-steps`
default to the file's `fps` / `horizonS` when a `--task` supplies them, and to
50 / 200 otherwise; passing them explicitly always wins.

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
    """The task-selection flags: a `*.caliper-task.json` artifact, or the
    built-in reach task. `required` demands that ONE of the two forms is given
    (`eval` cannot run without a task; `autopsy` can — it just skips E and L)."""
    p.add_argument(
        "--task",
        default=None,
        help="a *.caliper-task.json artifact (robot + scene + success predicate)",
    )
    p.add_argument("--urdf", default=None, help="robot URDF (caliper.Robot.from_urdf)")
    p.add_argument("--frame", default=None, help="frame the reach task targets")
    p.add_argument(
        "--target",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="world-space reach target",
    )
    p.add_argument("--tol", type=float, default=0.05, help="reach success tolerance (m)")
    p.add_argument("--episodes", type=int, default=20, help="seeded eval episodes")
    p.add_argument("--seed", type=int, default=0, help="base seed (episode k = seed + k)")
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="max control steps per episode (default: the task file's horizonS, else 200)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=None,
        help="control rate, MUST match collection (default: the task file's fps, else 50)",
    )
    p.set_defaults(task_required=required)


def _robot(urdf: str):
    import caliper  # lazy runtime dep

    return caliper.Robot.from_urdf(urdf)


def _reach_args(args) -> bool:
    """Is the reach form fully specified?"""
    return bool(args.urdf and args.frame and args.target)


def _task(args):
    """`(task, robot)` for these flags: the `EvalTask` they name and the robot it
    runs on. Either may be None — `--urdf` alone means "no task, but here is the
    robot" (autopsy's P003 saturation check), and autopsy with neither form runs
    the D and P sections only.

    Loud when the forms are mixed or half-specified: silently ignoring `--frame`
    because `--task` also appeared would hide which task a reported success rate
    belongs to.
    """
    from .eval import eval_task_from_file, reach_eval_task

    if args.task:
        if args.urdf or args.frame or args.target:
            raise SystemExit(
                "--task already carries the robot and the success criterion: drop "
                "--urdf/--frame/--target, or drop --task to run the reach task"
            )
        over = {}
        if args.fps is not None:
            over["fps"] = args.fps
        if args.max_steps is not None:
            over["max_steps"] = args.max_steps
        task = eval_task_from_file(args.task, **over)
        return task, task.robot
    robot = _robot(args.urdf) if args.urdf else None
    if _reach_args(args):
        return (
            reach_eval_task(
                robot,
                args.frame,
                args.target,
                tol=args.tol,
                max_steps=200 if args.max_steps is None else args.max_steps,
                fps=50 if args.fps is None else args.fps,
            ),
            robot,
        )
    if args.frame or args.target:
        raise SystemExit(
            "the reach task needs all of --urdf PATH --frame F --target X Y Z"
        )
    if args.task_required:
        raise SystemExit(
            "no task: pass --task FILE (a *.caliper-task.json), or the reach form "
            "--urdf PATH --frame F --target X Y Z"
        )
    return None, robot


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

    task, robot = _task(args)
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

    task, _ = _task(args)  # the robot rides inside the task
    result = evaluate(
        load_lerobot_policy(args.policy_dir),
        task,
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

    if bool(args.urdf) == bool(args.task):
        raise SystemExit(
            "coverage needs exactly one robot source: --urdf PATH, or --task FILE "
            "(a *.caliper-task.json, whose robot is used — its scene and success "
            "criterion play no part in coverage generation, which replays the "
            "dataset and plans free-space episodes)"
        )
    if args.task:
        from .task import load_task

        robot = _robot(str(load_task(args.task).robot_path))
    else:
        robot = _robot(args.urdf)
    rep = generate_coverage(
        args.dataset_root,
        robot,
        args.out_root,
        episodes=args.episodes,
        seed=args.seed,
        bins=args.bins,
    )
    print(rep.to_json(indent=2) if args.json else rep.render_text())
    return 1 if rep.error_findings_after else 0


def _cmd_drive(args) -> int:
    from .bridge import drive_main

    if bool(args.urdf) == bool(args.task):
        raise SystemExit(
            "drive needs exactly one robot source: --urdf PATH, or --task FILE "
            "(a *.caliper-task.json, whose robot is used — observations stay "
            "state-based, so its scene and success criterion play no part here)"
        )
    if args.task:
        from .task import load_task

        urdf = str(load_task(args.task).robot_path)
    else:
        urdf = args.urdf
    return drive_main(args.policy_dir, urdf, device=args.device)


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

    p = sub.add_parser(
        "autopsy",
        help="dataset doctor + debugger (+ eval/profile with --task or the reach form)",
    )
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
    p.add_argument("--urdf", default=None, help="the dataset's robot URDF")
    p.add_argument(
        "--task",
        default=None,
        help="a *.caliper-task.json to take the robot from (instead of --urdf)",
    )
    p.add_argument("-n", "--episodes", type=int, default=4, help="targeted episodes to add")
    p.add_argument("--seed", type=int, default=0, help="base seed (episode k = seed + k)")
    p.add_argument("--bins", type=int, default=20, help="histogram bins per dof for targeting")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_coverage)

    p = sub.add_parser(
        "drive",
        help="serve a checkpoint over the stdio JSONL bridge (Studio spawns this)",
    )
    p.add_argument("policy_dir", help="lerobot Hub checkpoint directory")
    p.add_argument("--urdf", default=None, help="the robot the sim is running")
    p.add_argument(
        "--task",
        default=None,
        help="a *.caliper-task.json to take the robot from (instead of --urdf)",
    )
    p.add_argument("--device", default="cpu", help="inference device (default: cpu)")
    p.set_defaults(fn=_cmd_drive)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
