# Stability contract

What you can rely on release over release, stated plainly. Caliper is pre-1.0
software; this page says exactly what that does and does not license us to
break, so "0.x" never becomes an excuse.

## Versioning policy (pre-1.0 semver)

One version number covers the whole surface: the workspace crates, the CLI,
the Python package, and Caliper Studio all ship as the same `0.MINOR.PATCH`
(release CI refuses a tag whose version disagrees with `Cargo.toml` and
`pyproject.toml`).

- **Patch (`0.x.y` → `0.x.y+1`) — never breaks.** Bug fixes, docs,
  performance. No public API, CLI flag, wire format, or file format changes
  behavior-visibly, with one exception: a fix to output that was *wrong* (a
  bug is not an interface).
- **Minor (`0.x` → `0.x+1`) — may break, but only with a receipt.** Every
  breaking change appears in `CHANGELOG.md` under **Changed** or **Removed**
  with what broke and what to migrate to. A break that is not in the
  changelog is a bug — report it.
- **Post-1.0** this collapses to standard [semver](https://semver.org/):
  breaking changes require a major version.

## Deprecation policy

Nothing public disappears without a warning you had a release to see:

1. The release that deprecates something keeps it working and makes it warn —
   a `#[deprecated]` attribute in Rust, a `DeprecationWarning` in Python, a
   stderr notice in the CLI — naming the replacement.
2. The **next minor** release at the earliest may remove it.

Precedent: the `record` default format flip (v2.1 → v3.0) kept the old
behavior reachable (`--format v21`), announced the change in the flag's own
`--help` text, and documented it in the changelog.

## LeRobotDataset compatibility matrix

The dataset formats are an interface with someone else's loader on the other
end, so every cell of this matrix is pinned by an oracle test that runs the
*real* `lerobot` package, not a schema lookalike:

| Format | Write | Read | Proof |
|---|---|---|---|
| **v3.0** (native) | ✓ default — `caliper record`, `RecorderV3` | ✓ `DatasetReaderV3`, `replay` auto-detects | our recording loads **directly** in lerobot 0.4.4 (windowing + padding asserted, one verified-decreasing SGD step); cross-direction: our reader reads a lerobot-written dataset; edits stay loadable — `oracle/tests/test_dataset_v3.py` |
| **v2.1** (legacy) | ✓ `caliper record --format v21`, `Recorder` | ✓ `DatasetReader`, `replay` auto-detects | schema + stats validated via pyarrow (`oracle/tests/test_lerobot_dataset.py`); full round-trip through lerobot's own v2.1→v3.0 converter and back into a real `LeRobotDataset` load — `oracle/tests/test_lerobot_roundtrip.py` |

The lerobot-version fine print, pinned by test rather than hoped
(`oracle/tests/test_lerobot_roundtrip.py`): lerobot **< 0.4** loads our v2.1
datasets natively; lerobot **≥ 0.4** dropped v2.x reading entirely and
rejects them with its own `BackwardCompatibilityError` version gate — that is
lerobot's contract, not a Caliper bug, and it is why v3.0 is the default. Any
*other* parse error against our metadata fails the oracle.

Within a minor release the on-disk bytes we write for a given format do not
change meaning; a format-affecting change is a **Changed** entry by the policy
above.

## MSRV and language floors

- **Rust:** MSRV **1.89** (`rust-version` in the workspace `Cargo.toml`).
  Raising it is a **Changed** changelog entry in a minor release, never a
  patch.
- **Python:** **≥ 3.10**, via a single `abi3-py310` wheel — one wheel per
  platform covers every later CPython, so a new Python release does not need
  a new Caliper release.

## The promise list

These are properties, not aspirations — each is enforced by something that
runs in CI:

- **Single artifact per face.** One CLI binary, one `.dmg`, one abi3 wheel.
  No ROS workspace, no conda environment, no driver installation.
  ([Lightweight, measured](./lightweight.md) keeps the sizes honest.)
- **No GPU requirement, anywhere.** The engine is pure CPU; the optional
  MuJoCo contact backend is CPU; even the learning sidecar trains on CPU (GPU
  is an optimization, never a floor). Enforced culturally by CI running
  everything on GPU-less runners, and structurally by
  `scripts/assert-lean.sh`, which fails the build if the core facade crate
  ever pulls a GPU/sim/transport dependency.
- **Offline-capable.** Nothing in the engine or faces phones home. The oracle
  sets `HF_HUB_OFFLINE=1` so an accidental network dependency fails loudly;
  the policy runner loads checkpoints from local directories only.
- **Deterministic seeded simulation.** The engine is clock-free — state
  advances only on `step(dt)` — and the one randomized component uses a
  seeded splitmix64 PRNG. Same input + same seed = same bytes out, pinned by
  determinism tests and usable as a CI assertion
  ([Headless CI recipe](./headless-ci.md)).

## What is *not* promised (pre-1.0)

Honesty cuts both ways: Rust API shapes, CLI human-readable (non-`--json`)
output text, Studio UI layout, and the learning sidecar's Python internals
may all change at minor releases — with changelog receipts, per the policy
above. If you are scripting against the CLI, prefer the `--json` outputs;
their fields only grow within a minor.
