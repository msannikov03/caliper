# The task artifact — `*.caliper-task.json`

A task file is the shareable unit of a manipulation problem: robot, scene,
gripper, success criterion, horizon — one JSON file that every face consumes.
Studio opens it (robot loaded, props placed, zones drawn, live session and
recording pre-configured); the learning sidecar builds envs and eval tasks
from it (`VecSimEnv.from_task`, `caliper-learn eval --task`, `autopsy
--task`); the success criterion inside is judged by the **same predicate
semantics in Rust and Python**, pinned by a shared parity table both suites
run.

```json
{
  "version": 1,
  "name": "pick-cube",
  "robot": "../robots/gripper_arm.urdf",
  "q0": [0.0, 0.0, 0.02],
  "scene": {
    "ground": 0.0,
    "props": [
      { "name": "cube", "kind": "box",
        "halfExtents": [0.05, 0.05, 0.05],
        "pos": [0.0, 0.0, 0.05], "mass": 0.05,
        "rgba": [0.9, 0.4, 0.2, 1.0], "material": "wood" }
    ],
    "zones": [
      { "name": "bin", "center": [0.4, 0.2, 0.02],
        "half": [0.05, 0.05, 0.02], "rgba": [0.2, 0.8, 0.4, 0.35] }
    ]
  },
  "gripper": { "joint": "gripper", "closed": "lo" },
  "success": { "kind": "placed_in_zone", "prop": "cube",
               "zone": "bin", "settled_speed": 0.01 },
  "horizonS": 20.0,
  "fps": 50
}
```

## Field semantics

- **`version`** — must be `1`. The schema is covered by the
  [stability contract](./stability.md): within version 1 it only grows;
  loaders reject unknown versions loudly.
- **`robot`** — URDF path, resolved **relative to the task file's directory**
  (absolute paths work too). The file must exist at load time.
- **`q0`** — optional start pose (defaults to zeros).
- **`scene.props`** — the same prop vocabulary the contact sim uses
  (`box`/`sphere`/`cylinder`, quaternion w-first, analytic inertia from
  `mass`); `material` is a preset name (`rigid`/`rubber`/`foam`/`steel`/
  `wood`, case-insensitive) or a custom
  `{solref, solimp, friction}` dict — identical to the Python face's
  `material=` kwarg.
- **`scene.zones`** — named axis-aligned boxes. Zones are *evaluator-side*:
  they never enter the physics, they are drawn as translucent boxes in Studio
  and referenced by success predicates. `rgba` is a render hint only.
- **`gripper`** — optional override for the
  [gripper channel](../capabilities/contact-sim.md) (auto-detection covers
  sanely-named robots); `closed` says which limit end means closed
  (default `"lo"`).
- **`success`** — a [success predicate](../capabilities/verdicts.md)
  (`lifted` / `placed_in_zone` / `all_of` / `any_of`). A zone given as a
  **string** resolves by name against `scene.zones` at load time (and is
  stored resolved). Semantics are identical in both implementations — plain
  `>=` comparisons, inclusive zone faces, and a `settled_speed` check that
  *errors* on a velocity-less state rather than passing a fly-through.
- **`horizonS`**, **`fps`** — optional episode horizon and recording fps
  defaults.

**Strictness is the point.** Unknown keys — at *every* level — are rejected,
not ignored. A typo in a task file must never silently weaken the success
test or drop a prop.

## Consumers

| Face | What a task file does |
|---|---|
| Studio | *Open task…* loads the robot, places props (materials included), draws zones, pre-fills the recording task label + fps, passes the gripper override and success predicate to the live session — the `SUCCESS` badge then judges every streamed state |
| `caliper_learn` | `load_task(path)` → `VecSimEnv.from_task(...)` (props + success + q0 + ground) and `eval`/`autopsy` `--task <file>` on the CLI |
| Rust | `caliper_sim_mujoco::task::{load_task, TaskSpec}` + `SuccessTracker` — the same evaluator Studio streams live |

## Parity, not promises

Rust and Python each implement the predicate evaluator; a shared table of
predicate × state → verdict cases (including two knife-edge rows sitting
exactly on a zone face and exactly at the settled-speed limit, and two rows
that must *raise*) runs in both suites. If the implementations ever drift,
a test goes red on whichever side moved.
