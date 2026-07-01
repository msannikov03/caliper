# Studio dataflow graph

`caliper-graph` is the Phase-8 **dataflow graph executor** — the engine behind
Studio's Simulink-style node editor, and equally usable from the CLI and Python.

## What it is

A deterministic, serde-serializable graph IR (`GraphDoc`, persisted as the
`.caliper-graph.json` schema) that composes existing engine ops into a runnable
pipeline. All three faces (the Studio Tauri backend, the CLI's `graph`
subcommand, and the PyO3 `run_graph`) (de)serialize a `GraphDoc`, then call
`validate` and `run`.

**No new math lives here.** Every COMPUTE node dispatches to an *existing* engine
free function or type. The crate is pure Rust and lean — `serde` plus the engine
crates plus `nalgebra`, with no `rand` and no `tokio`.

## Shape of the IR

- `ir` — the persisted schema: `PortType`, `NodeKind`, `Node`, `Edge`,
  `GraphDoc`, `PortValue`, `ClipData`, `ReportData`.
- `validate` — returns `Diagnostics` (per-node / per-edge errors) plus a
  topological order, or a cycle report.
- `exec` — `run` returns a `GraphResult` or a `GraphError`.

## Node kinds

The `NodeKind` enum covers source, compute, and sink nodes, each mapping to
engine functionality:

- **Sources / configs** — `StartConfig`, `GoalPose`, `NamedConfig`.
- **Compute** — `Ik`, `MoveJ`, `MoveL`, `PlanRrt`, `Control` (`kp`/`kd`),
  `GravityDrop`, `CollisionCheck`.
- **Sinks / views** — `View` (3D scene), `Scope` (a named signal plotted over
  time), `Report`.

## Determinism

The executor is deterministic: `PlanRrt` is seeded, and control/dynamics
rollouts are tick-driven and clock-free (consistent with the rest of the
engine). A wired graph like `StartConfig → GoalPose → Ik → MoveL → View + Scope`
produces the same result on every face and every run.

## How the graph is verified

The executor's **dispatch is verified faithful** — a `MoveJ`/`MoveL` node
produces the same result as calling `caliper_motion::move_j` / the Cartesian path
directly (a parity test), and the executor is deterministic. It has its own
oracle coverage. Note the scope of this: it verifies that the *graph wrapper*
faithfully calls the engine, on top of the engine's own verification — it does
not add a new independent check of the underlying math.
