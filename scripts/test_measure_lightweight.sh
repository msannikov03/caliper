#!/usr/bin/env bash
# Tests for scripts/measure_lightweight.sh — positive and negative per behavior.
# Pure bash + python3 (json validation); runs the real script against the real
# repo, so it needs no fixtures. Exits non-zero on the first failing assertion.
#
#   bash scripts/test_measure_lightweight.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/scripts/measure_lightweight.sh"
JSON="$ROOT/target/metrics/lightweight.json"
PY="python3"
if [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"; fi

PASS=0
fail() { echo "FAIL: $1" >&2; exit 1; }
ok()   { PASS=$((PASS + 1)); echo "  ok: $1"; }

assert_json() { # assert_json <python-expr over report dict> <label>
  "$PY" -c "
import json, sys
report = json.load(open('$JSON'))
machine, metrics = report['machine'], report['metrics']
assert $1, '$2'
" || fail "$2"
  ok "$2"
}

echo "[1] default run (positive: completes, valid JSON, full metric set)"
bash "$SCRIPT" >/dev/null
[[ -f "$JSON" ]] || fail "lightweight.json not written"
ok "exit 0 + JSON written"
assert_json "set(metrics) == {'dmg_size_mb','wheel_size_mb','cold_fk_ms','robot_load_ms','peak_rss_mb','pip_install_s','record_overhead_x','seeded_plan_deterministic','python_import_ms'}" \
  "every metric present (value or skipped) — none silently missing"
assert_json "all(('value' in e) != ('skipped' in e) for e in metrics.values())" \
  "each entry is exactly one of value|skipped"
assert_json "all(e['skipped'] for e in metrics.values() if 'skipped' in e)" \
  "every skip carries a non-empty reason"
assert_json "machine['git_rev'] and machine['cpu'] and machine['os']" \
  "machine provenance stamped"

echo "[2] idempotence (positive: second run overwrites cleanly)"
bash "$SCRIPT" >/dev/null
"$PY" -c "import json; json.load(open('$JSON'))" || fail "second run corrupted JSON"
ok "re-run emits valid JSON"

echo "[3] artifact-absent honesty (negative: absent wheel => skipped, not a number)"
if ls "$ROOT"/target/wheels/*.whl >/dev/null 2>&1; then
  echo "  skip: a wheel exists; the absent-artifact branch is not reachable here"
else
  assert_json "'skipped' in metrics['wheel_size_mb'] and 'maturin' in metrics['wheel_size_mb']['skipped']" \
    "absent wheel reported as skipped with a how-to-build reason"
fi

echo "[4] CLI-binary handling"
if [[ -x "$ROOT/target/debug/caliper" && ! -x "$ROOT/target/release/caliper" ]]; then
  # positive: explicit override measures, and taints provenance with the debug caveat
  CALIPER_BIN="$ROOT/target/debug/caliper" bash "$SCRIPT" >/dev/null
  assert_json "'value' in metrics['cold_fk_ms'] and metrics['cold_fk_ms']['value'] > 0" \
    "override binary: cold FK measured (positive)"
  assert_json "metrics['seeded_plan_deterministic']['value'] is True" \
    "override binary: seeded plan bit-identical (positive)"
  assert_json "machine['caveat'] and 'debug' in machine['caveat']" \
    "debug override taints provenance with a caveat (negative: numbers cannot pass as release)"
  bash "$SCRIPT" >/dev/null  # restore the honest default report
elif [[ -x "$ROOT/target/release/caliper" ]]; then
  assert_json "'value' in metrics['cold_fk_ms']" "release binary present: cold FK measured"
else
  assert_json "'skipped' in metrics['cold_fk_ms'] and 'cargo build --release' in metrics['cold_fk_ms']['skipped']" \
    "no binary at all: CLI metrics skipped with the build hint"
fi

echo "[5] bogus CALIPER_BIN (negative: hard error, not a silent skip)"
if CALIPER_BIN=/nonexistent/caliper bash "$SCRIPT" >/dev/null 2>&1; then
  fail "nonexistent CALIPER_BIN should exit non-zero"
fi
ok "nonexistent CALIPER_BIN exits non-zero"

echo "all $PASS assertions passed"
