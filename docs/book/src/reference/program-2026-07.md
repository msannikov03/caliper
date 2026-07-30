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
| MP4 video encoding | ✅ `caliper_learn.video` (dtype `video`, lerobot-exact, real round-trip); video meta columns now emitted **natively by the Rust writer** (the pyarrow bridge is a repair tool — see the follow-on table) |
| Data factory docs + this audit | ✅ Data factory chapter, capability-matrix rows, this page |

## Follow-on — human demonstration loop (in progress)

The next program replaces bake-then-replay with hands-on interaction inside
Studio. Only what is built is listed as built:

| Phase | Status |
|---|---|
| **A1 — live sim session** | ✅ built — Studio's Simulate mode steps the sim (MuJoCo, or the builtin integrator in default builds) live in a background thread: fixed 1 ms timestep, PD hold target, ~60 Hz state stream, pause-as-freeze, deterministic reset, live contact count ([details](../capabilities/contact-sim.md#live-session-studio)) |
| **A2 — input layer** | ✅ built — the live session is drivable: joint sliders edit the hold target (measured pose ghosted so PD lag is visible), IK gizmo retargets the tip live, keyboard jog (`[`/`]` select, `-`/`=`/arrows move), gamepad cartesian tip drive (deadband + cubic response, A = pause, B = reset), Space = freeze. Inputs write only the hold target — never the streamed pose — so input and stream cannot fight |
| **A3 — teleop episode recording** | ✅ built — record takes from a live session straight into a native LeRobotDataset v3.0, captured in the sim thread at exact tick decimation (default 50 fps; timestamps are `k/fps`, never wall-clock): per-episode task labels, save/discard per take, finish-dataset, open-in-Data. Reset discards the take. Acceptance verified: a Studio-recorded dataset loads in real lerobot 0.6.0 |
| **B1/B2 — gripper channel + weld attach-on-grasp** | ✅ built — gripper auto-detection (joint or child-link name, mimic-aware, override supported), open/close on the shared PD hold target, and an explicitly-labeled weld grasp: commanded-closed + contact ⇒ weld with relpose captured at activation (snap-free < 1 mm), open ⇒ natural release; one prop at a time; reset releases ([details](../capabilities/contact-sim.md)) |
| **B3 — success predicates** | ✅ built — `lifted` / `placed_in_zone` / combinators with an exact JSON schema (the seed of the coming task artifact), reported by `VecSimEnv(success=…)`, scored by the eval harness (wilson unchanged), named in the autopsy verdict |
| **C — task artifact** | ✅ built — `*.caliper-task.json` v1, every face consumes it, Rust/Python predicate parity pinned by a shared table ([details](./task-artifact.md)) |
| **D1 — episode replay** | ✅ built — recorded episodes re-perform on the 3D robot in Data mode; located doctor findings jump to their pose |
| **D2 — verdict viewers** | ✅ built — eval/debug/profile/autopsy `--json` render in Data mode (Wilson bar, findings, verdict line) |
| **D3 — task zoo** | ✅ built — five solvability-witnessed starters in `tasks/` ([details](./task-zoo.md)) |
| **F1 — native video features** | ✅ built — the writer emits video metadata in one pass; the pyarrow bridge is a repair tool now |
| **F2 — load time** | ✅ built — parallel hull priming; so101 5.0 ms release / 0.42 s debug (bit-identical hulls) |
| **F4 — decomposition recon** | ✅ concluded: **skip vendoring** — the seam is points-only by design, CoACD's dylib alone outweighs the entire dmg and is nondeterministic multicore; if ever needed, parry's pure-Rust VHACD behind an optional feature is the pick. The recon's real yield: the hull-builder orientation bug (48% of real meshes falling back) — found and fixed |
| **G1/G2 — CI** | ✅ built — monthly newest-lerobot pairing watch (skips = failures) + a Linux runtime job |

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
