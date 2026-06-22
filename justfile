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
oracle:
    python -m pytest oracle -v

# Run the Caliper Studio desktop app (Tauri dev)
app:
    npm --prefix apps/studio install
    npm --prefix apps/studio run tauri dev

# Full local CI gate (engine)
ci: fmt-check lint test light
    @echo "local CI green ✓"
