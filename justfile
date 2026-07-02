# Caliper dev tasks — `just <task>`
# Note: the `studio` Tauri app is excluded from the plain engine checks because
# it needs a built frontend (dist/); build it with `just app`.
default:
    @just --list

build:
    cargo build --workspace --exclude studio

test:
    cargo test --workspace --exclude studio

fmt:
    cargo fmt

fmt-check:
    cargo fmt --check

lint:
    cargo clippy --workspace --exclude studio --all-targets -- -D warnings

# Assert the core crate stays lightweight (no heavy/GPU/sim deps)
light:
    cargo build -p caliper
    bash scripts/assert-lean.sh

# Build + install the Python bindings into the active venv
py:
    maturin develop -m crates/caliper-py/Cargo.toml

# Run the Python cross-validation / face-parity oracle
# (needs the repo .venv: maturin + pytest + numpy + pin + pyarrow installed)
oracle:
    python -m pytest oracle -v

# Run the Phase-7 learning sidecar tests (pure-torch BC oracle, CPU, seconds).
# Needs torch + caliper (PyO3) + caliper_learn installed into the repo .venv:
#   env -u CONDA_PREFIX uv pip install --python .venv/bin/python torch
#   env -u CONDA_PREFIX .venv/bin/maturin develop -m crates/caliper-py/Cargo.toml
#   env -u CONDA_PREFIX uv pip install --python .venv/bin/python -e learn
learn:
    python -m pytest learn -v

# Run the Caliper Studio desktop app (Tauri dev)
app:
    npm --prefix apps/studio install
    npm --prefix apps/studio run tauri dev

# Full local CI gate (engine)
ci: fmt-check lint test light
    @echo "local CI green ✓"

# build the mdBook docs site (needs mdbook on PATH)
docs:
    mdbook build docs/book
