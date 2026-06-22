# Caliper dev tasks — `just <task>`
default:
    @just --list

build:
    cargo build --workspace

test:
    cargo test --workspace

fmt:
    cargo fmt

fmt-check:
    cargo fmt --check

lint:
    cargo clippy --workspace --all-targets -- -D warnings

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

# Full local CI gate
ci: fmt-check lint test light
    @echo "local CI green ✓"
