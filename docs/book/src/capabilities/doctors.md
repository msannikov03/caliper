# Doctors & trajectory lint

Caliper ships three diagnostic engines. Each one turns a class of silent,
late-surfacing failures into an explicit, plain-English report **before** you
pay for them:

| Doctor | Input | Codes | Catches |
|---|---|---|---|
| **Asset doctor** (`caliper-doctor`) | a `.urdf` / `.xacro` file | `A001`–`A014` | CAD-export defects that break loading, physics, or collision coverage |
| **Dataset doctor** (`caliper-dataset::analyze`) | a LeRobotDataset v3.0 root | `D001`–`D015` | data defects that are invisible at record time and fatal to a trained policy |
| **Trajectory lint** (`caliper-kinematics::lint_path` + face-side collision lint) | a sampled trajectory | `T001`–`T009` | limit violations and path-quality hazards before a trajectory runs |

Shared contract, all three:

- **Findings are data, not errors.** A finding never changes an exit code or
  throws; the report *is* the product. Commands only error when the input
  cannot even be inspected (unreadable file, unparseable dataset).
- **Stable codes.** Every check has a fixed code you can filter on in JSON
  output; the sets below are exhaustive as of this writing.
- **Sorted most-severe-first**, with per-severity counts.
- **Severities**: *Error* = broken or actively wrong if ignored; *Warning* =
  runs/loads but behaves worse than you think; *Info* = worth knowing, nothing
  wrong per se.

Where to run them:

| | CLI | Python | Studio |
|---|---|---|---|
| Asset doctor | `caliper doctor robot.urdf [--repair]` | `caliper.doctor(path, repair=…)` | automatic on robot load — findings appear in the error-banner area, with **Repair & reload** when a mechanical fix exists |
| Dataset doctor | `caliper data doctor <root>` | `caliper.data_doctor(root)` | Data mode → **Doctor** button |
| Trajectory lint | `caliper report … [--strict]` | `caliper.lint_path(robot, …)` | — |

---

## Asset doctor (`A001`–`A014`)

Real-world URDFs — CAD exports above all — routinely carry defects that the
rest of the stack surfaces late, one at a time, or not at all: a silently
dropped collider here, `has_inertia = false` there, an MJCF export MuJoCo
rejects. `diagnose` runs every check in one pass; `repair` emits a **repaired
copy** — the input file is never touched.

The doctor parses XML itself (leniently) instead of going through `urdf-rs`:
half the point is diagnosing files `urdf-rs` rejects outright, like a
`<limit>` without `velocity=` or a `.urdf` full of xacro leftovers. `.xacro`
input is expanded first via `caliper_model::xacro`.

### Check catalog

**A001 — missing/zero `<inertial>` on a non-root link** *(Error,
auto-fixable)*. Any movable link (or fixed link folded onto one) without a
real `<inertial>` flips the whole model to `has_inertia = false`, which gates
off every dynamics entry point (simulation, computed-torque control, gravity
compensation). Repair: `compute_inertials` fills mass/COM/tensor from the
link's collision (else visual) geometry at a uniform density — analytic
formulas for primitives, divergence-theorem integrals for meshes. It never
overwrites an explicit inertial with positive mass, and never touches the
root link (see A010).

**A002 — implausible inertia** *(Error)*. Non-finite entries, a zero tensor
with positive mass, a negative principal moment, or a triangle-inequality
violation (any principal moment exceeding the sum of the other two — physically
impossible for real mass). Checked on the tensor's **eigenvalues**, so
converter-dropped off-diagonal terms are caught even when the diagonal looks
sane. Consequence: integrators go unstable or silently wrong. No auto-fix —
the true values live in your CAD; recovering them mechanically would be a
guess.

**A003 — mesh unresolvable or unloadable** *(Error on `<collision>`, Warning
on `<visual>`)*. The message lists every path that was tried
(relative/absolute/`file://`/`package://`). A dropped *collision* mesh is an
Error because the engine then checks **nothing** for that link — a collision
query can report "clear" while the real link is in contact. A dropped visual
only degrades rendering. No auto-fix (the doctor cannot invent your mesh
file), but the finding names exactly what to restore and where it was
expected.

**A004 — duplicate mesh basenames pointing at different files** *(Warning,
auto-fixable)*. Two links referencing `hand.stl` from different directories
work locally, but any pipeline that flattens assets into one folder (bundlers,
converters, most sim importers) silently makes both links use *one* of the
files. Repair: `dedupe_mesh_basenames` renames later duplicates
(`m2__hand.stl`, …) in the document and returns a **file-copy plan** — the
engine performs no file writes; executing the copies is the calling face's
job (the CLI and Studio both do it).

**A005 — link has `<visual>` but no `<collision>`** *(Warning)*. The link is
invisible to collision checking and planning: paths can be planned straight
through it. Often intentional (decorative geometry) — hence a Warning — but on
an arm segment it usually means the exporter dropped the collision block.

**A006 — collision mesh above the 1024-vertex hull cap** *(Info)*. Caliper
convex-hulls mesh colliders and subsamples above the cap; the hull may be
slightly loose. Nothing is wrong — worth knowing when you see near-miss
distances that disagree with CAD by millimetres.

**A007 — revolute joint without usable position limits** *(Warning,
auto-fixable)*. Missing `<limit>`, or a degenerate `lower == upper` range on a
joint that is supposed to move. IK and planning then treat the joint as
unbounded (or frozen), and datasets recorded through it can contain
wound-up configurations. Deliberately `continuous` joints are exempt.
Repair: `inject_limits` writes a conservative ±π range, marked as such in the
repair log.

**A008 — zero-length / unparseable joint axis** *(Error)*. An axis of
`0 0 0` (or garbage text) makes the joint's motion undefined; most loaders
either reject the file or silently substitute a default axis that sends FK to
the wrong place. No auto-fix — the doctor cannot know which axis you meant.

**A009 — non-unit joint axis** *(Warning, auto-fixable)*. Some parsers
normalize, some don't: the same file produces different kinematics in
different tools, and velocity/effort limits change meaning by the axis norm.
Repair: `normalize_axes` rescales to a unit vector (direction preserved).

**A010 — zero-mass root link** *(Info, heuristic)*. The signature of
onshape-to-robot and several other CAD exporters. Harmless in itself (the
root never moves), which is why the repair pipeline deliberately **skips the
root** when computing inertials — flagged so you know the file's provenance.

**A011 — mimic references an unknown joint** *(Error)*. The mimic joint's
motion is undefined; loaders that tolerate it produce a robot whose FK
disagrees with the file's intent.

**A012 — mimic chain (incl. self-mimic)** *(Error)*. A mimic whose source is
itself a mimic (or itself). Resolution order is undefined across tools;
caliper's compiler rejects it outright.

**A013 — xacro leftovers in a `.urdf`** *(Error, or Warning when
`xmlns:xacro` is declared and in-process expansion succeeds)*. An unexpanded
`$(find …)`, `${…}` or `<xacro:…>` tag in a plain `.urdf` means URDF parsers
fail on the tags or silently misread the values. With the namespace declared,
caliper *can* expand it in-process (the rest of the report then describes the
expanded model) — but most other URDF consumers will not, so it stays a
portability Warning. Fix: run the file through xacro once and ship the
expanded `.urdf`. (A real `.xacro` file is simply expanded — leftovers are
its normal content, not a finding.)

**A014 — `<limit>` missing `velocity=`** *(Error, auto-fixable)*. `urdf-rs`
(and therefore every Rust-stack consumer) rejects the **whole file** over this
one attribute; a missing `effort=` safely defaults to 0 in caliper's own
loader but not everywhere. Repair: `inject_limits` writes the mandatory
`velocity="1"` (conservative) so the file at least loads.

### Repair semantics

- **Everything is opt-in.** Every repair rewrites physics-relevant fields, so
  each `RepairOpts` flag is off by default; `RepairOpts::all()` (what the CLI
  `--repair`, Python `repair=True`, and Studio's *Repair & reload* use)
  enables all four: `compute_inertials`, `normalize_axes`,
  `dedupe_mesh_basenames`, `inject_limits`.
- **The input file is never modified.** Repair returns the repaired document;
  faces write it as a sibling `<stem>.repaired.urdf` so relative mesh
  references keep resolving.
- **Nothing fails silently.** What could not be fixed lands in `skipped` with
  the reason (e.g. a link with no geometry to integrate).
- **Verify the copy.** The findings in a repair run describe the *original*
  file; re-run `diagnose` on the output (the CLI and Studio do this
  automatically and show the after-report).

> **Density caveat.** `compute_inertials` assumes a **uniform material
> density**, default 1000 kg/m³ (water — a sane mid-range for
> printed/machined robot parts; override with `--density` / `density=`).
> Real links are not uniform-density solids: computed inertials are
> *placeholders* good enough for stable simulation and roughly-scaled
> dynamics, not a substitute for CAD-derived values in dynamics-critical
> control. The repair log marks every computed inertial so you can audit
> them.

---

## Dataset doctor (`D001`–`D015`)

Pre-training diagnostics over a native LeRobotDataset v3.0. Every check
targets a failure mode that is **invisible at record time, silent during
training, and fatal to the resulting policy**. The analyzer makes two
streaming passes (one episode resident at a time), recomputes all statistics
from the raw bytes, and is fully deterministic (seeded subsampling): a report
is a pure function of the dataset bytes and the options. The default
thresholds (`AnalyzeOptions::default`) are tuned so a healthy teleop dataset
produces **zero findings**.

Checks relating actions to observations key off lerobot's conventional
feature names `action` and `observation.state`; when either is absent those
checks are skipped. Everything per-feature runs on every `float32` vector
feature.

**D001 — dead dof** *(Warning)*. A dof whose whole-dataset std is ~0 never
moves. Why it kills training: std-based normalization divides by ~zero
(exploding inputs or NaNs, depending on the stack), and the policy learns the
dof is irrelevant — if the joint was supposed to move, that behaviour is
unlearnable from this data.

**D002 — stale/missing `meta/stats.json`** *(Error)*. The doctor recomputes
mean/std per dof and compares against the stored stats (missing file, missing
feature entry, wrong length, or values beyond tolerance). Why it kills
training: lerobot normalizes **with the stored values**. Stats that describe
different bytes (classic cause: the dataset was edited or concatenated
without recomputing) shift and scale every input systematically — the policy
trains in one coordinate system and deploys in another. This is the single
most common "trained fine, deploys as garbage" cause the doctor can prove.

**D003 — saturated/collapsed action dof** *(Warning)*. More than half the
frames (configurable) pinned at the dof's min, max, or a single histogram
bin. Why: the label distribution is nearly constant — the policy mostly sees
one value and will slam that value at deployment; usually a teleop gain or
command-clipping problem.

**D004 — echo/lag action labels** *(Warning)*. `action` is nearly identical
to `observation.state` (RMS difference below a small fraction of the state's
spread). Why: the policy can minimize its loss by *copying its input* —
it will never move the robot on its own. Classic cause: logging the measured
position as the "action" in high-fps position control. Fix: train on delta
actions or one-step-shifted targets.

**D005 — numerically tiny actions** *(Warning)*. An action dof's range is
orders of magnitude below the typical state range (unit mismatch: rad vs deg,
normalized vs raw). Why: after normalization, sensor noise dominates the
learning signal for that dof.

**D006 — contradictory demonstrations** *(Warning)*. Near-identical states
(on a seeded reservoir subsample) with widely divergent actions. Why:
behavior cloning averages the modes into an action *nobody demonstrated* —
the mean of "go left" and "go right" is "drive into the obstacle in the
middle". Fix: delete the wrong demo, or condition the policy on the missing
context (task/goal) that distinguishes them.

**D007 — coverage holes** *(Info)*. A dof visits fewer than half its
histogram bins between its own min and max. Why: the policy has no data for
most of that dof's span and will extrapolate there — fine if the region is
intentionally unreachable, dangerous if deployment passes through it.

**D008 — corridor-shaped data** *(Info)*. Mean |pairwise correlation| across
a feature's dofs near 1: the dofs move in lockstep along one path. Why: the
dataset spans a 1-D manifold of the workspace; any state off that corridor is
out-of-distribution at deployment. Fix: vary starts, goals and speeds.

**D009 — episode-length outlier** *(Info)*. Robust (MAD-based) z-score on
episode lengths. Why: a 10× episode is usually a stuck recording, a
concatenated take, or an aborted demo — and it dominates (or starves) the
sampling of its task.

**D010 — irregular timestamps** *(Warning)*. Frame-to-frame dt deviating
from `1/fps`. Why: delta-timestamp windowing and action-chunking (ACT, delta
actions) pair frames by *time*; misaligned frames mean the model learns from
mismatched (state, action) pairs.

**D011 — frozen tail** *(Warning)*. The last N frames of an episode are
bit-identical across every vector feature: the robot froze before the
recording stopped. Why: the policy learns to stall at the end of the task —
seen as "the arm approaches the goal and stops short" at deployment. Fix:
trim via the edit ops (split at the last moving frame, delete the remainder).

**D012 — dead camera** *(Error when frames cannot be decoded; Warning for
black/white/constant streams)*. Why: undecodable frames crash or silently
skip in the dataloader; a black stream means a vision policy is blind on an
input it is supposed to use — it trains anyway, keying on whatever else
correlates.

**D013 — duplicated consecutive camera frames** *(Info)*. A large fraction
of consecutive frames byte-identical: the camera delivered fewer real frames
than the recorded fps claims. Why: visual dynamics are slower than labeled,
which skews anything trained with frame-stacking or optical flow.

**D014 — brightness drift** *(Info)*. Mean image brightness drifts
substantially start→end within an episode (auto-exposure hunting, lighting
change). Why: the policy can key on brightness as a spurious progress signal
instead of the scene. Fix: lock exposure/white balance.

**D015 — duplicate episodes** *(Warning)*. Cross-episode identical state
sequences (accidental double-record or copy). Why: duplicates over-weight one
demonstration and leak between train/val splits, inflating validation
metrics.

The report also carries the recomputed per-feature summaries (`dim`, `mean`,
`std`, `min`, `max`, per-dof histogram bin occupancy) so you can cross-check
the analyzer's arithmetic against your own.

In **Studio**, findings with an episode reference are clickable — the episode
table jumps to the row so the offending take can be inspected, split, or
deleted on the spot. Any structural edit clears the report (it described the
pre-edit bytes); run the Doctor again after editing.

---

## Trajectory lint (`T001`–`T009`)

Typed findings over a *sampled* trajectory (`times`/`q`/`qd`/`qdd` rows —
exactly what `Trajectory.sample_uniform` produces). `T001`–`T007` live in the
engine (`caliper_kinematics::lint_path`, layered on `path_report`);
`T008`/`T009` are the collision half, computed face-side (the CLI `report`
verb) because the kinematics crate cannot depend on the collision crate.
Errors mean *do not run this trajectory*; Warnings deserve a second look.
Thresholds (`LintOptions`) are metric-robot engineering defaults, not physics
— tune per cell.

| Code | Severity | Finding |
|---|---|---|
| `T001` | Error | position limit violated (negative margin), located at the worst sample |
| `T002` | Error | velocity utilization > 100 % (+ tolerance) vs `vmax` |
| `T003` | Error | acceleration utilization > 100 % (+ tolerance) vs `amax` |
| `T004` | Warning | sustained dwell within a small margin of a position limit (default > 25 % of samples) — no escape headroom left |
| `T005` | Warning | wrap-around detour: total joint travel ≫ net start→end change (the "360° spin"), located at the peak excursion off the chord |
| `T006` | Warning | finite-difference jerk spike above `1.5 × jmax` (disabled per joint when `jmax` is infinite, e.g. TOPP output) |
| `T007` | Warning | singular corridor: one finding per contiguous window with σ_min below threshold |
| `T008` | Error | path in collision (self or world), one finding per contiguous time window |
| `T009` | Warning | near-miss: path passes within the clearance margin of an obstacle or itself (conservative boolean re-query with inflated colliders: pair gaps flag below 2× the margin, ground below 1×) |

Every finding carries the offending `joint`, `time`, and measured `value`
machine-readably — faces never parse message text. `caliper report --strict`
exits non-zero on any Error-severity finding, which is the CI hook.
