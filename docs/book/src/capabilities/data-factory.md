# Data factory — randomization, coverage, materials & video

Once you can [record](./learning.md) and [doctor](./doctors.md) datasets, the
next problem is *making enough good data* without a robot. The data factory is
the sim-side toolkit for that: vary the world deterministically, fill the holes
the [dataset doctor](./doctors.md) finds, give contacts sane material
behaviour, and store camera streams as real MP4 video that lerobot reads.

All of this lives in the `caliper_learn` sidecar (Python) and
`caliper-sim-mujoco` (Rust, the optional `mujoco` feature). It is a **data
factory, not an RL framework** — you get the substrate; you bring the task.

## Domain randomization

`caliper_learn.randomize` turns one nominal scene into a distribution of
scenes, seeded so a run is reproducible and its draw is a diffable JSON record
(you can commit the exact randomization a dataset was collected under).

```python
from caliper_learn import RandomizationSpec, sample
from caliper_learn.randomize import apply_to_mjcf, apply_to_env

spec = RandomizationSpec(
    mass=(0.8, 1.2),          # per-body mass multiplier range
    joint_damping=(0.5, 2.0), # multiplier
    gains=(0.9, 1.1),         # kp/kd multiplier
    camera_pos=0.02,          # absolute jitter (m)
    spawn_pose=0.01,          # absolute jitter (m / rad)
    gravity=0.1,              # absolute jitter on |g|
)
draw = sample(spec, rng, ndof)     # a plain, JSON-serializable dict
```

- **`sample(spec, rng, ndof)`** draws once from every enabled field in a fixed
  order, so disabling one field never reshuffles the others' random stream —
  the same integer seed gives byte-identical draws.
- **`apply_to_mjcf(draw, mjcf)`** edits model-level parameters (mass — which
  also scales inertia — joint damping/frictionloss, gravity) as a structural
  XML edit and returns a new MJCF string. Only `b_`-prefixed robot bodies are
  touched; props are left alone.
- **`apply_to_env(draw, env)`** applies the runtime parameters (controller
  gains, spawn offset clipped into joint bounds, camera jitter around a
  snapshotted base pose so resets never drift).

### In a vectorized env

`VecSimEnv` takes a spec directly and draws **per environment** at every reset:

```python
from caliper_learn import VecSimEnv
env = VecSimEnv(robot, num_envs=8, randomization=spec, seed=0)
obs = env.reset()                 # each env gets its own draw
env.randomization_draws           # the 8 draws, also in info['randomization']
```

Model-level draws recompile that env's `MjModel` from the randomized MJCF at
reset. That is a real cost — one model plus one XML compile per env per reset —
documented in the module so you size `num_envs` accordingly. Runtime-only
randomization (gains, spawn, camera) has no rebuild cost.

## The doctor → generator loop

The [dataset doctor's](./doctors.md) `D007` coverage finding tells you *which
joint-limit bins your data never visits*. `coverage_gen` closes that loop:
read the finding, plan new episodes whose goals land in the emptiest bins,
append them to a **new** dataset (the input is never mutated), and re-run the
doctor to show the occupancy delta.

```
caliper-learn coverage INPUT_DATASET OUTPUT_DATASET --urdf robot.urdf -n 40 --seed 0
```

```python
from caliper_learn import generate_coverage
report = generate_coverage(dataset_root, robot, out_root, episodes=40, seed=0)
report.occupancy_before, report.occupancy_after   # min-bin occupancy
report.d007_before, report.d007_after             # finding counts
```

The histogram updates as episodes are planned, so consecutive episodes chase
*different* holes; it widens the goal window and falls back to free sampling
when a bin is hard to reach. Runs are deterministic — the same seed produces a
byte-identical output dataset. In a smoke test on a deliberately corridor-shaped
dataset it raised min-bin occupancy 0.2 → 0.7 and drove `D007` findings 3 → 0.

## Contact materials

Tuning MuJoCo's `solref`/`solimp`/`friction` by hand is the classic dark art —
stiff contacts jitter or explode, soft ones penetrate. `ContactMaterial` turns
it into a named choice with derivations documented in the source:

| Preset | Use for | Character |
|---|---|---|
| `Rigid` | metal-on-metal, hard stops | stiff, near-inelastic |
| `Steel` | tools, structural parts | very stiff, low friction |
| `Wood` | props, fixtures | stiff, medium friction |
| `Rubber` | grippers, feet, bumpers | soft, high friction |
| `Foam` | soft props, padding | very soft, damped |
| `Custom{solref, solimp, friction}` | your own | validated on construction |

Set a scene default or override per prop:

```rust
let opts = MjcfOptions {
    default_material: Some(ContactMaterial::Foam),   // ground + unmarked props
    props: vec![PropSpec { material: Some(ContactMaterial::Steel), ..cube }],
    ..Default::default()
};
```

`Custom` is validated on build (positive `solref` timeconst/dampratio, `solimp`
`dmin`/`dmax` in `(0,1)` with `dmin ≤ dmax`) — a bad tuple is rejected loudly,
not silently clamped.

## Contact stability linter

With the `mujoco` feature, `lint_contact_stability` runs a settle rollout and
reports how a scene misbehaves, each finding with a concrete fix:

- **`C001` explosion** — `|qacc|`/energy grows during settling (the "spins
  uncontrollably" class). *Fix: raise the `solref` timeconst to ≥ 2× the
  timestep, or reduce the timestep.*
- **`C002` penetration** — persistent contact depth after settling. *Fix:
  stiffen the material or the `solimp` `dmax`.*
- **`C003` jitter** — contact force oscillates after settling. *Fix: increase
  `solref` damping or switch to a damped preset.*

`C001` suppresses `C002`/`C003` (depth and force stats are meaningless mid
blow-up). The classifier core (`classify_stability`) is pure and always
compiled; only the rollout that produces a trace needs MuJoCo.

## Convex decomposition seam

A single convex hull is a poor collider for a concave part (a cup collides like
a solid blob). The `ColliderDecomposer` trait is the seam for real convex
decomposition (CoACD-class); the shipped `NaiveDecomposer` is the identity —
one piece, the existing hull — and `MjcfOptions.hull_decomposer` plumbs
multi-piece output through to MJCF (`<mesh>` asset + `<geom>` per piece). The
seam is here so a decomposer can drop in without touching the exporter; the
heavy algorithm is deliberately **not** vendored yet.

## MP4 video features

Datasets can store camera streams as real MP4 video (dtype `video`) instead of
per-frame PNGs — the layout modern lerobot policies expect. `caliper_learn.video`
mirrors lerobot 0.4.4's own encode settings (`libsvtav1`, `yuv420p`, `g=2`,
`crf=30`; H.264 alternative) so the output is byte-compatible.

```python
from caliper_learn.video import available, encode_episode_video, VideoRecorder
available()   # (bool, reason) — probes PyAV → ffmpeg → unavailable
```

`VideoRecorder` buffers a camera stream per episode and writes the v3.0
`videos/{key}/chunk-XXX/file-XXX.mp4` layout (one episode per file,
`from`/`to_timestamp` bookkeeping). Because the Rust writer does not yet emit
video columns, `attach_video_metadata` is a deterministic pyarrow post-write
that appends the four `videos/{key}/*` columns to `meta/episodes`, the pixel
stats to `stats.json`, and the feature entry to `info.json` — documented as the
bridge until the Rust writer grows native video columns. A recorded sim video
dataset loads directly in real lerobot and decodes to frames matching the
renders within measured codec tolerance (≈0.011 mean-abs-diff vs a 0.05 gate).

> **Encoder availability.** Encoding needs PyAV or an `ffmpeg` on `PATH`; the
> gate test skips honestly when neither is present. Decoding for the lerobot
> round-trip uses torchcodec, as lerobot itself does.
