# Build program (2026-07)

This page is the honest ledger of the July 2026 build program: a two-round
research effort (competitive landscape, then a pain-point mining of what people
actually struggle with in robotics software) that produced a four-wave plan,
and what each wave actually shipped. A ✗ or *deferred* here is deliberate — the
goal is that this table never lies about the state of the app.

The program's thesis: Caliper occupies a two-front position — **lighter than
everything** (one artifact, no GPU, offline) and **more legible than
everything** (doctors and verdicts on top of a single-owner codebase that also
owns the dataset format and the sim). The waves build the second front while
telling the story of the first.

## W1 — Doctors (trust at every input) — ✅ shipped

| Planned | Shipped |
|---|---|
| Asset doctor: lint + auto-repair URDF/MJCF | ✅ `caliper-doctor` crate, `A001`–`A014`, opt-in repair (inertia-from-mesh via divergence-theorem integrals pinned to analytic ground truth) |
| Dataset doctor: pre-train linter | ✅ `caliper-dataset::analyze`, `D001`–`D015`, streaming two-pass |
| Trajectory linter | ✅ `caliper-kinematics::lint_path`, `T001`–`T009` (incl. the "360° detour" detector + collision-margin) |
| Loud-error edge guardrails | ✅ ~20 CLI/Python messages upgraded to name-the-field/got/expected/fix |
| Faces + Studio + docs | ✅ CLI (`doctor`, `data doctor`, `report --strict`), Python, Studio (auto-diagnose + Repair&reload, Data-mode Doctor panel), Doctors chapter, capability matrix |

## W2 — Verdicts (train→deploy legibility) — ✅ shipped

| Planned | Shipped |
|---|---|
| Seeded eval harness | ✅ `caliper_learn.eval`, `E001`–`E003`, Wilson-95 CIs, `sweep` checkpoint ranking, per-episode seeds |
| Policy deploy debugger | ✅ `caliper_learn.debugger`, `P001`–`P008` (incl. normalization-mismatch and cadence-mismatch, the mined killers) |
| Latency profiler | ✅ `caliper_learn.profile`, `L001`–`L003`, chunk-aware refill-vs-pop p95, honest achievable-Hz |
| **The Policy Autopsy** (flagship) | ✅ `caliper_learn.autopsy` — data doctor (D) + debugger (P) + eval (E) + latency (L) under one verdict; `caliper-learn` console script |
| Verdicts docs | ✅ Verdicts chapter, capability-matrix rows |

*Scope note:* the autopsy is CLI/Python only — policy inference is Python-side,
so there is no Studio autopsy panel. Stated plainly in the chapter.

## W3 — Reach (make the won arguments universally true) — ✅ shipped (config items owner-gated)

| Planned | Shipped |
|---|---|
| Robot zoo | ✅ `caliper fetch <name>`/`--list` (embedded corpus URDFs; meshes not embedded, doctor-error set documented per robot) |
| Benchmark harness + metrics page | ✅ `scripts/measure_lightweight.sh` (+ self-test), *Lightweight, measured* page |
| Stability contract | ✅ *Stability contract* page (semver, deprecation, dataset compat matrix), `CHANGELOG.md` |
| Headless CI recipe | ✅ *Headless CI recipe* page (run-twice-diff determinism) |
| Version identity | ✅ `caliper.__version__`, `caliper --version`, `caliper_learn.__version__`, CLI↔Python parity smoke |
| Zero-to-moving quickstart | ✅ quickstart chapter (every command verified against real surfaces) |
| Studio first-run tour | ✅ 6-step dismissible overlay + palette "Show tour" |
| Notarize macOS / tested Linux runtime | ⏳ **owner-gated** — needs the Apple Developer-ID cert; Linux wheel CI exists but is not runtime-verified |

## W4 — Data factory — ✅ shipped

| Planned | Shipped |
|---|---|
| Domain randomization API | ✅ `caliper_learn.randomize` (CI-diffable seeded draws) + `VecSimEnv(randomization=)` |
| Coverage generator (doctor→generator loop) | ✅ `caliper_learn.coverage_gen` + `caliper-learn coverage` |
| Contact material presets + stability linter | ✅ `ContactMaterial` presets, `lint_contact_stability` (`C001`–`C003`) |
| Convex decomposition | ◑ **seam only** — `ColliderDecomposer` trait + identity impl; CoACD-class algorithm deliberately not vendored (per the research: leave the seam, don't build it) |
| MP4 video encoding | ✅ `caliper_learn.video` (dtype `video`, lerobot-exact, real round-trip); ◑ video meta columns via a pyarrow post-write bridge until the Rust writer grows them natively |
| Data factory docs + this audit | ✅ Data factory chapter, capability-matrix rows, this page |

## Deliberately not built (traps the research flagged)

- **A ROS bridge / ROS-compat layer** — the mined value is *escape from* ROS;
  interop stays at the artifact level (URDF, MJCF, LeRobotDataset).
- **A photorealistic / GPU renderer or in-house physics** — MuJoCo embedded is
  the ceiling; the weight advantage is the point.
- **VLA / foundation-model training infrastructure** — Caliper *produces*
  datasets and *debugs* any policy; it does not compete with H100-scale
  training.
- **Cloud / fleet features** — offline-capable is an invariant.
- **A general RL framework** — the vectorized env is a substrate; task and
  learner are yours.

## Owner-gated / owner-supplied (not code)

Apple Developer-ID cert + notarization · PyPI / crates.io tokens · the
crates.io umbrella name (`caliper` is taken) · a GPU training run on real
hardware · the human GUI review (now spanning Jog / Motion / Simulate+Contact /
Graph / Data modes + the first-run tour).
