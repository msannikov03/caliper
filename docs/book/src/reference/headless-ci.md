# Headless CI recipe

Caliper's engine needs no GPU, no display, and no hardware, which means your
robot regression tests can run on the free GitHub Actions tier. This page is a
copy-paste job that builds the CLI, runs a seeded simulation + eval batch on a
plain `ubuntu-latest` runner, and *proves* determinism by running the seeded
work twice and diffing the bytes.

## The job

Drop this into `.github/workflows/robot-eval.yml` of a project that depends on
Caliper (swap the URDF and goals for your robot):

```yaml
name: robot-eval
on: [push, pull_request]

jobs:
  seeded-sim-eval:
    runs-on: ubuntu-latest        # free tier — no GPU anywhere in this job
    env:
      URDF: oracle/fixtures/corpus/so100.urdf   # your robot here
      GOAL: 0.3,0.2,-0.1,0,0,0                  # in-limit joint goal
    steps:
      # Get Caliper. Until the crates.io / PyPI names are settled, build from
      # the repo (Swatinem/rust-cache makes rebuilds incremental-fast):
      - uses: actions/checkout@v4
        with:
          repository: msannikov03/caliper   # or your fork / a vendored copy
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Build the CLI once
        run: cargo build --release -p caliper-cli

      - name: Eval batch — seeded plans across a seed sweep + machine-readable report
        run: |
          mkdir -p out
          for seed in 1 2 3 4 5; do
            ./target/release/caliper plan "$URDF" --goal "$GOAL" --seed "$seed" \
              > "out/plan_seed$seed.txt"
          done
          ./target/release/caliper report "$URDF" --goal "$GOAL" --json > out/report.json

      - name: Simulate + record an episode, headlessly
        run: |
          ./target/release/caliper sim "$URDF" --duration 2.0 > out/sim.txt
          ./target/release/caliper record "$URDF" --out out/dataset \
            --goal "$GOAL" --ticks 500 --fps 50

      # THE DETERMINISM ASSERTION: run the seeded work twice; the bytes must
      # match. `diff` exits non-zero on any drift, failing the job — flaky
      # simulation cannot hide.
      - name: Determinism — same seed, bit-identical output
        run: |
          ./target/release/caliper plan "$URDF" --goal "$GOAL" --seed 42 > a.txt
          ./target/release/caliper plan "$URDF" --goal "$GOAL" --seed 42 > b.txt
          diff a.txt b.txt
          ./target/release/caliper report "$URDF" --goal "$GOAL" --json > r1.json
          ./target/release/caliper report "$URDF" --goal "$GOAL" --json > r2.json
          diff r1.json r2.json

      # Gate on quality, not just "it ran": --strict makes `report` exit
      # non-zero when the trajectory linter finds errors (limit violations,
      # wrap-around detours, singular corridors…).
      - name: Trajectory lint gate
        run: ./target/release/caliper report "$URDF" --goal "$GOAL" --strict

      - uses: actions/upload-artifact@v4
        with:
          name: eval-out
          path: out/
```

Total cold time is dominated by the one release build of the CLI (cached
across runs by `rust-cache`); the measured work itself is seconds — see
[Lightweight, measured](./lightweight.md).

## Why `diff` is a valid determinism test here

It would not be for most simulators. It is for Caliper because the engine is
clock-free (state advances only on `step(dt)`, nothing reads the wall clock)
and the only randomness is a seeded splitmix64 PRNG — so a seeded run's output
is a pure function of its inputs, byte for byte, and the strictest possible
assertion (`diff`) is also the simplest. If that diff ever fires, it is a real
regression in the determinism contract
([Stability contract](./stability.md)), not noise to be tolerated.

## Variants

**Python instead of the CLI** — same free runner, engine driven through the
bindings (this is exactly how Caliper's own `python` CI job works):

```yaml
      - uses: astral-sh/setup-uv@v5
      - run: uv venv && uv pip install maturin numpy
      - run: uv run maturin develop -m crates/caliper-py/Cargo.toml
      - run: |
          uv run python - <<'EOF'
          import caliper
          robot = caliper.Robot.from_urdf("oracle/fixtures/corpus/so100.urdf")
          # ... seeded planner / sim / dataset assertions ...
          EOF
```

Prebuilt abi3 wheels (macOS arm64 + manylinux x86_64) are attached to each
[GitHub release](https://github.com/msannikov03/caliper/releases) — `pip
install <wheel-url>` skips the Rust toolchain entirely once you pin a release.

**Policy eval batch** — the learning sidecar's eval harness
(`caliper-learn eval --json`, seeded success-scored rollouts with Wilson-95
aggregates) runs on the same CPU runners; it additionally needs
`uv pip install -e learn` and CPU torch. Budget accordingly: torch is the one
heavyweight download in that variant, and it belongs to the *training* side of
the fence, never to the engine.

**Contact simulation** — the optional MuJoCo backend is also CPU-only and
headless-capable; enable the `mujoco` feature and fetch the pinned dylib with
`scripts/fetch_mujoco.sh` first. The default recipe above deliberately uses
only the built-in simulator so the job stays dependency-free.
