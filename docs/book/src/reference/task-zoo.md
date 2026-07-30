# The task zoo

Five [task artifacts](./task-artifact.md) that run on a fresh clone. They live
in `tasks/` at the repo root, reference a robot that ships with the source, and
need no download step:

```
tasks/01_lift.caliper-task.json
tasks/02_place.caliper-task.json
tasks/03_hold_high.caliper-task.json
tasks/04_sort.caliper-task.json
tasks/05_precise_place.caliper-task.json
```

The zoo exists so "open a task" and "score a policy against a task" are things
you can do in one command, before you have authored anything yourself. It is
also the regression suite for the task format: every file is swept through both
loaders, built into a real MuJoCo scene and checked for two failure modes a
JSON schema cannot see — a verdict that is *already true* at the start pose, and
a verdict that *no reachable pose* can satisfy.

## The tasks

All five drive `gripper_arm`, a 3-dof arm (`j1`, `j2`, and a prismatic
`gripper`) that hangs over the origin, at 50 fps.

| File | Name | Goal | Predicate | Horizon |
|---|---|---|---|---|
| `01_lift` | `zoo-lift-cube` | Raise the wooden cube 8 cm off the ground | `lifted` (`ref: initial`) | 8 s |
| `02_place` | `zoo-place-cube` | Carry the cube into a bin 15 cm to one side and let it settle | `placed_in_zone` + `settled_speed` | 15 s |
| `03_hold_high` | `zoo-hold-cube-high` | Hold a heavier steel cube out at arm's length, above 30 cm, and stop moving | `all_of` [`lifted` (`ref: absolute`), `placed_in_zone`] | 15 s |
| `04_sort` | `zoo-sort-blocks` | Move *either* of two blocks across to the dock | `any_of` [two `placed_in_zone`] | 30 s |
| `05_precise_place` | `zoo-precise-place` | The same carry as `02`, onto a pad half as wide, with a tighter settle | `placed_in_zone` + `settled_speed` | 20 s |

Between them they exercise every predicate kind, both `lifted` references, four
of the five [contact-material](../capabilities/contact-sim.md) presets (wood,
steel, rubber, foam) and both single- and multi-prop scenes.

Difficulty is graduated, and it is a real ordering rather than a label: sweeping
the joint grid for poses that satisfy each verdict gives roughly 24 000 for
`01`, 7 000 for `04`, 4 000 for `03`, 1 800 for `02` and 800 for `05`.

## Opening one

In Studio, *Open task…* (toolbar or ⌘K) loads the robot, places the props with
their materials, draws the zones as translucent boxes, pre-fills the recording
label and fps, and hands the success predicate to the live session — the
`SUCCESS` badge then judges every streamed state.

From Python:

```python
from caliper_learn.task import load_task
from caliper_learn.vec_env import VecSimEnv

task = load_task("tasks/02_place.caliper-task.json")
with VecSimEnv.from_task(task, num_envs=8) as env:
    obs = env.reset(seed=0)
```

To score a checkpoint against one:

```bash
caliper-learn eval --task tasks/01_lift.caliper-task.json path/to/checkpoint
caliper-learn autopsy --task tasks/01_lift.caliper-task.json path/to/checkpoint
```

`--task` supplies the robot, scene, start pose, fps and step budget, so
`--urdf` / `--frame` / `--target` are not needed (and mixing the two is a loud
error). The eval reports a success rate with a Wilson 95% interval — see
[verdicts](../capabilities/verdicts.md).

## Constraints these tasks are built around

The zoo is small and its scenes are modest, for reasons worth stating plainly
rather than discovering by hand.

**Grasping is a weld heuristic, one prop at a time.** Closing the gripper on a
prop welds it to the jaw; there is no friction-based grip and no second weld.
`04_sort` therefore asks for *either* block via `any_of`, not both — a task
requiring two simultaneous carries would not be expressible, and a task
requiring two sequential ones would be scored by a predicate that cannot
remember the first.

**The robot must ship with the source.** Task files resolve `robot` relative to
themselves, and a zoo task that needed `caliper fetch` first would not run on a
fresh clone. `gripper_arm` is the only in-repo robot with a gripper joint,
collision geometry and inertials on every link, so all five use it. The robot
zoo behind [`caliper fetch`](../quickstart.md) (so101 and friends) ships visual
meshes without collision geometry: those robots can be posed and rendered, but
nothing in a scene can ever touch them, so no zoo task references one.

**Every prop has to sit where the jaw can reach it.** `gripper_arm`'s jaw sweeps
an annulus 0.323–0.401 m from its shoulder at `(x, z) = (0, 0.5)`, in the x-z
plane only (both hinges turn about Y). At ground level that annulus narrows to
a patch a few centimetres wide around the origin, which is why the ground cubes
sit *at* the origin and why `04_sort`'s blocks are tall — their raised top faces
are what brings them back into reach. Props off the ring are decoration; the
Rust sweep fails on one.

**No task is scored by "return it to where it started".** `evaluate()` latches
success — one true step marks the whole episode — so a predicate satisfied at
the start pose reports 100% for a policy that does nothing at all. That rules
out a literal return-to-origin task, and it is why `05_precise_place` tests
release precision with a narrow pad *offset* from the spawn point instead. Both
suites assert that every zoo verdict is false at t=0.

## What the suites check

`crates/caliper-sim-mujoco/tests/task_zoo.rs` (no `mujoco` feature needed —
reading and judging a task never touches MuJoCo) sweeps the directory for: every
file loading and round-tripping, names unique across the zoo, robot paths
relative and resolvable, props passing the engine's own rulebook, scored props
and named zones existing, no verdict true at the spawn state, and every prop's
top face inside the jaw's annulus.

`learn/tests/test_task_zoo.py` adds what only a real engine can answer: each
task compiles into a `VecSimEnv`, resets, reports its props where the file put
them, steps, and judges live states without raising. It also pins a *witness*
per task — a grasp pose whose jaw meets the prop's top face, plus a carry pose
whose welded prop position satisfies the verdict — computed through real forward
kinematics, so editing a zone out of the arm's reach turns the task red instead
of quietly unsolvable.
