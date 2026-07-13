# Lightweight, measured

"Lightweight" is a claim, and claims ship with numbers. This page is the
metrics table for Caliper's footprint, the script that produces every number
in it, and the incumbent figures it is being compared against — with
citations, because the comparison is only fair if you can check it.

The rule: **no value appears here without a measurement behind it.** Anything
we have not measured yet says `TBD — run scripts/measure_lightweight.sh`, not
a hopeful estimate.

## The metrics table

| Claim | Target | Measured | Measured on |
|---|---|---|---|
| Studio install (`.dmg`) | ≤ 250 MB | **10.4 MB** (aarch64 `.dmg`, MuJoCo dylib bundled) | Apple M3 Max, 36 GB, macOS, rev `1dd648c`, 2026-07-13 |
| Python wheel | ≤ 100 MB | TBD — run `scripts/measure_lightweight.sh` | — |
| Cold CLI → robot loaded + FK | ≤ 5 s | TBD — run `scripts/measure_lightweight.sh` ¹ | — |
| RAM, full app + sim | ≤ 1 GB | TBD (Studio); headless plan+sim peak RSS below ¹ | — |
| `pip install` (wheel) | ≤ 30 s | TBD — run `scripts/measure_lightweight.sh` | — |
| Record overhead vs realtime | ≤ 1.1× | TBD — run `scripts/measure_lightweight.sh` ¹ | — |
| Seeded rollouts bit-identical | always | **yes** — machine-verified (see below) | CI, every push |

¹ A smoke run against a **debug** binary (explicitly tainted as
"not representative" by the script — debug Rust is typically 10–100× slower
than release) measured: cold CLI→FK on a real 6-dof URDF **6.4 ms**, robot
load (panda) **8.0 ms**, headless plan+sim peak RSS **11.5 MB**, record
overhead **0.43× realtime** (i.e. more than 2× faster than realtime) — every
one of them orders of magnitude inside its target *before* optimization. The
release numbers replace these TBDs the first time the script runs against a
release build; until then the table refuses to print them as measured.

Bit-identical determinism is not a benchmark artifact but an engine property:
the core is clock-free, the one randomized component (sampling planners) uses
a seeded splitmix64 PRNG, and it is pinned by tests that run on every push —
the graph-executor determinism oracle (`oracle/tests/test_graph.py`), the
seeded-planner tests, and the run-twice byte-compare the measurement script
performs (`seeded_plan_deterministic` in its JSON output).

## Reproducing the numbers

```sh
# from the repo root; build the artifacts you want measured first:
cargo build --release -p caliper-cli
maturin build --release -m crates/caliper-py/Cargo.toml   # optional: wheel size
bash scripts/measure_lightweight.sh
```

The script writes `target/metrics/lightweight.json` (machine-stamped: CPU,
RAM, OS, git rev, which binary was measured) and a ready-to-paste markdown
row-set (`target/metrics/lightweight.md`). It is honest by construction:

- artifacts that are absent are reported as `skipped: <reason + how to build>`,
  never guessed;
- pointing it at a debug binary (`CALIPER_BIN=…`) taints every timing with a
  "debug build — not representative" caveat in the provenance;
- it uses [hyperfine](https://github.com/sharkdp/hyperfine) when installed and
  falls back to a median-of-3 wall-clock loop when not, and says which it used;
- `MEASURE_PIP=1` additionally times a `pip install` of the wheel into a
  throwaway venv (opt-in because it creates an environment).

The script has its own test suite (`scripts/test_measure_lightweight.sh`) —
positive and negative cases per behavior, including "absent artifact must skip,
not invent" and "debug override must taint".

## What the incumbents cost (with citations)

These are the vendors' own published figures at the time of writing (2026-07);
follow the links, they may have changed.

| Stack | Install | Hardware floor | Time to first robot |
|---|---|---|---|
| **NVIDIA Isaac Sim** | ~10 GB-class download, 50 GB disk recommended | **RTX GPU required** (min. GeForce RTX 3070-class); 32 GB RAM min, 64 GB recommended [\[req\]](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) | first launch compiles shaders — minutes, on qualifying hardware only |
| **MoveIt 2** | full ROS 2 desktop install (multi-GB); prebuilt binaries are Ubuntu-via-apt, everything else is a colcon source build [\[install\]](https://moveit.ai/install-moveit2/binary/) | no GPU, but a supported Ubuntu/ROS 2 pairing | a workspace build from source is commonly tens of minutes |
| **lerobot** (pip) | `pip install lerobot` pulls PyTorch (+CUDA wheels on Linux, ~2.5 GB for torch alone [\[pypi\]](https://pypi.org/project/torch/#files)) plus the `av` FFmpeg wheel — a multi-GB environment [\[lerobot\]](https://pypi.org/project/lerobot/) | CPU works; GPU needed for serious training | minutes of dependency resolution + download |
| **Caliper** | one 10 MB `.dmg`, one CLI binary, one abi3 wheel | **no GPU, no ROS, no CUDA** — a laptop | see the table above |

To be fair to the incumbents: Isaac Sim is a photorealistic GPU simulator,
MoveIt is a full ROS 2 planning framework, and lerobot ships an entire
training stack — they carry that weight because they do things Caliper does
not (rendering-quality sim, ROS integration, GPU training). The comparison is
not "Caliper does everything they do, smaller"; it is "for the
load-a-robot / plan / simulate / record / deploy loop, you do not have to pay
their entry price."

## Related pages

- [Stability contract](./stability.md) — what these artifacts promise release
  over release.
- [Headless CI recipe](./headless-ci.md) — the "runs on a free CPU runner"
  claim, as a copy-paste job.
