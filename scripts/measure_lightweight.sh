#!/usr/bin/env bash
# Measure the "lightweight" claims ON THIS MACHINE and emit JSON + a markdown
# row-set. Every number in docs/book/src/reference/lightweight.md is either
# produced by this script or marked TBD — the metrics ARE the marketing, so no
# value ships without a measurement behind it.
#
# Honest by construction:
#   - artifacts that are not present are reported as {"skipped": "<why>"},
#     never guessed;
#   - a debug binary (CALIPER_BIN override) taints every timing with a
#     "debug build — not representative" caveat;
#   - output is stamped with machine info + git rev so a number can never be
#     quoted without its provenance.
#
# Usage:
#   bash scripts/measure_lightweight.sh
#   CALIPER_BIN=target/debug/caliper bash scripts/measure_lightweight.sh   # smoke only
#   MEASURE_PIP=1 bash scripts/measure_lightweight.sh   # also time a venv pip install (needs a wheel)
#
# Idempotent: writes target/metrics/lightweight.{json,md} (overwritten each
# run, gitignored via target/) and prints the markdown to stdout. Exit 0 even
# when everything is skipped; non-zero only on real errors (e.g. a CALIPER_BIN
# that does not exist).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/target/metrics"
mkdir -p "$OUT_DIR"
JSON_OUT="$OUT_DIR/lightweight.json"
MD_OUT="$OUT_DIR/lightweight.md"

PY="python3"
if [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"; fi

# Fixtures: small + large real-robot URDFs vendored for the oracle corpus.
FK_URDF="$ROOT/oracle/fixtures/corpus/so100.urdf"      # 6 dof
FK_JOINTS="0.1,0.2,0.0,0.0,0.0,0.0"
LOAD_URDF="$ROOT/oracle/fixtures/corpus/panda.urdf"    # largest corpus fixture
PLAN_GOAL="0.3,0.2,-0.1,0.0,0.0,0.0"                   # in-limit for so100 (joint 2 is [-pi, 0])
RECORD_TICKS=500
RECORD_FPS=50

# --- binary selection -------------------------------------------------------
BIN=""
BIN_NOTE=""
DEBUG_CAVEAT=""
if [[ -n "${CALIPER_BIN:-}" ]]; then
  if [[ ! -x "$CALIPER_BIN" ]]; then
    echo "error: CALIPER_BIN='$CALIPER_BIN' does not exist or is not executable" >&2
    exit 1
  fi
  BIN="$CALIPER_BIN"
  BIN_NOTE="CALIPER_BIN override"
  if [[ "$BIN" == */debug/* ]]; then
    DEBUG_CAVEAT="debug build — timings not representative of a release binary"
  fi
elif [[ -x "$ROOT/target/release/caliper" ]]; then
  BIN="$ROOT/target/release/caliper"
  BIN_NOTE="release build"
fi
CLI_SKIP_REASON="release binary absent — build with: cargo build --release -p caliper-cli"

# --- helpers ----------------------------------------------------------------
# Accumulate metrics as KEY<TAB>JSON-VALUE lines; python assembles the report
# at the end (bash string-splicing of JSON is how numbers get silently
# corrupted).
METRICS_TSV="$(mktemp)"
TMP_WORK="$(mktemp -d)"
trap 'rm -f "$METRICS_TSV"; rm -rf "$TMP_WORK"' EXIT

emit() { # emit <key> <json-value>
  printf '%s\t%s\n' "$1" "$2" >>"$METRICS_TSV"
}

skip() { # skip <key> <reason>
  emit "$1" "$("$PY" -c 'import json,sys; print(json.dumps({"skipped": sys.argv[1]}))' "$2")"
}

# median wall-clock ms over N fresh-process runs (each run is a cold process;
# the OS page cache is warm after run 1 — reported as such).
wall_ms() { # wall_ms <runs> <cmd...>
  local runs="$1"; shift
  "$PY" - "$runs" "$@" <<'EOF'
import subprocess, sys, time, statistics
runs = int(sys.argv[1]); cmd = sys.argv[2:]
samples = []
for _ in range(runs):
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    samples.append((time.perf_counter() - t0) * 1e3)
print(f"{statistics.median(samples):.1f}")
EOF
}

# peak RSS in MB of one command run (BSD /usr/bin/time -l reports bytes,
# GNU time -v reports KB).
peak_rss_mb() { # peak_rss_mb <cmd...>
  local tf="$TMP_WORK/time.$$.txt"
  if /usr/bin/time -l true 2>/dev/null 1>&2; then
    /usr/bin/time -l "$@" >/dev/null 2>"$tf" || return 1
    awk '/maximum resident set size/ {printf "%.1f", $1/1048576}' "$tf"
  else
    /usr/bin/time -v "$@" >/dev/null 2>"$tf" || return 1
    awk -F': ' '/Maximum resident set size/ {printf "%.1f", $2/1024}' "$tf"
  fi
}

file_size_mb() { # file_size_mb <path>
  "$PY" -c 'import os,sys; print(f"{os.path.getsize(sys.argv[1])/1048576:.1f}")' "$1"
}

# --- artifact sizes ---------------------------------------------------------
DMG="$(ls -t "$ROOT"/target/release/bundle/dmg/*.dmg 2>/dev/null | head -1 || true)"
if [[ -n "$DMG" ]]; then
  emit dmg_size_mb "$("$PY" -c 'import json,sys; print(json.dumps({"value": float(sys.argv[1]), "unit": "MB", "artifact": sys.argv[2]}))' "$(file_size_mb "$DMG")" "$(basename "$DMG")")"
else
  skip dmg_size_mb "artifact absent — build with: npm --prefix apps/studio run tauri build"
fi

WHEEL="$(ls -t "$ROOT"/target/wheels/*.whl 2>/dev/null | head -1 || true)"
if [[ -n "$WHEEL" ]]; then
  emit wheel_size_mb "$("$PY" -c 'import json,sys; print(json.dumps({"value": float(sys.argv[1]), "unit": "MB", "artifact": sys.argv[2]}))' "$(file_size_mb "$WHEEL")" "$(basename "$WHEEL")")"
else
  skip wheel_size_mb "artifact absent — build with: maturin build --release -m crates/caliper-py/Cargo.toml"
fi

# --- CLI timings ------------------------------------------------------------
if [[ -n "$BIN" ]]; then
  # Cold CLI → robot: fresh process spawn + URDF load + FK, median wall time.
  if command -v hyperfine >/dev/null 2>&1; then
    HJSON="$TMP_WORK/hyperfine.json"
    hyperfine --warmup 1 --runs 5 --export-json "$HJSON" \
      "$BIN fk $FK_URDF --joints $FK_JOINTS" >/dev/null
    FK_MS="$("$PY" -c 'import json,sys; print(f"{json.load(open(sys.argv[1]))["results"][0]["median"]*1e3:.1f}")' "$HJSON")"
    FK_METHOD="hyperfine (5 runs, 1 warmup, median)"
  else
    FK_MS="$(wall_ms 3 "$BIN" fk "$FK_URDF" --joints "$FK_JOINTS")"
    FK_METHOD="3-run wall-clock loop, median (hyperfine not installed)"
  fi
  emit cold_fk_ms "$("$PY" -c 'import json,sys; print(json.dumps({"value": float(sys.argv[1]), "unit": "ms", "method": sys.argv[2]}))' "$FK_MS" "$FK_METHOD")"

  # Robot load: parse + compile the largest corpus URDF (panda, 9 dof w/ mimic).
  LOAD_MS="$(wall_ms 3 "$BIN" load "$LOAD_URDF")"
  emit robot_load_ms "$("$PY" -c 'import json,sys; print(json.dumps({"value": float(sys.argv[1]), "unit": "ms", "urdf": "panda.urdf", "method": "3-run wall-clock loop, median (includes process spawn)"}))' "$LOAD_MS")"

  # Peak RSS: max over a plan run and a sim run (the heaviest headless paths).
  PLAN_RSS="$(peak_rss_mb "$BIN" plan "$FK_URDF" --goal "$PLAN_GOAL" --seed 42 || true)"
  SIM_RSS="$(peak_rss_mb "$BIN" sim "$FK_URDF" --duration 2.0 || true)"
  if [[ -n "$PLAN_RSS" && -n "$SIM_RSS" ]]; then
    emit peak_rss_mb "$("$PY" -c 'import json,sys; p,s=float(sys.argv[1]),float(sys.argv[2]); print(json.dumps({"value": max(p,s), "unit": "MB", "plan": p, "sim": s, "method": "/usr/bin/time max RSS over plan + sim runs"}))' "$PLAN_RSS" "$SIM_RSS")"
  else
    skip peak_rss_mb "plan/sim run failed under /usr/bin/time"
  fi

  # Record overhead vs realtime: N ticks at FPS is N/FPS seconds of data;
  # overhead = wall / nominal (values < 1.0 mean faster than realtime).
  REC_OUT="$TMP_WORK/record_ds"
  REC_MS="$(wall_ms 1 "$BIN" record "$FK_URDF" --out "$REC_OUT" --goal "$PLAN_GOAL" --ticks "$RECORD_TICKS" --fps "$RECORD_FPS")"
  emit record_overhead_x "$("$PY" -c 'import json,sys; wall=float(sys.argv[1])/1e3; nominal=float(sys.argv[2])/float(sys.argv[3]); print(json.dumps({"value": round(wall/nominal,3), "unit": "x realtime", "wall_s": round(wall,3), "nominal_s": nominal, "ticks": int(sys.argv[2]), "fps": int(sys.argv[3])}))' "$REC_MS" "$RECORD_TICKS" "$RECORD_FPS")"

  # Determinism: two seeded plan runs must be byte-identical.
  "$BIN" plan "$FK_URDF" --goal "$PLAN_GOAL" --seed 42 >"$TMP_WORK/plan_a.txt" 2>&1
  "$BIN" plan "$FK_URDF" --goal "$PLAN_GOAL" --seed 42 >"$TMP_WORK/plan_b.txt" 2>&1
  if cmp -s "$TMP_WORK/plan_a.txt" "$TMP_WORK/plan_b.txt"; then
    emit seeded_plan_deterministic '{"value": true, "method": "two plan --seed 42 runs, byte-compared"}'
  else
    emit seeded_plan_deterministic '{"value": false, "method": "two plan --seed 42 runs, byte-compared"}'
  fi
else
  for k in cold_fk_ms robot_load_ms peak_rss_mb record_overhead_x seeded_plan_deterministic; do
    skip "$k" "$CLI_SKIP_REASON"
  done
fi

# --- python import ----------------------------------------------------------
if "$PY" -c 'import caliper' 2>/dev/null; then
  IMPORT_MS="$("$PY" -c 'import time; t=time.perf_counter(); import caliper; print(f"{(time.perf_counter()-t)*1e3:.1f}")')"
  emit python_import_ms "$("$PY" -c 'import json,sys; print(json.dumps({"value": float(sys.argv[1]), "unit": "ms", "python": sys.argv[2]}))' "$IMPORT_MS" "$PY")"
else
  skip python_import_ms "caliper not importable in $PY — run: maturin develop -m crates/caliper-py/Cargo.toml"
fi

# --- pip install (opt-in: creates a throwaway venv) --------------------------
if [[ "${MEASURE_PIP:-0}" == "1" && -n "$WHEEL" ]]; then
  PIP_VENV="$TMP_WORK/pipvenv"
  "$PY" -m venv "$PIP_VENV"
  PIP_MS="$(wall_ms 1 "$PIP_VENV/bin/pip" install --no-index "$WHEEL")"
  emit pip_install_s "$("$PY" -c 'import json,sys; print(json.dumps({"value": round(float(sys.argv[1])/1e3,1), "unit": "s", "method": "pip install --no-index <wheel> into a fresh venv"}))' "$PIP_MS")"
elif [[ -n "$WHEEL" ]]; then
  skip pip_install_s "opt-in — re-run with MEASURE_PIP=1 (creates a throwaway venv)"
else
  skip pip_install_s "wheel absent — build with: maturin build --release -m crates/caliper-py/Cargo.toml"
fi

# --- assemble JSON + markdown ------------------------------------------------
GIT_REV="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ //' || echo unknown)"
MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || "$PY" -c 'import os; print(os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES"))' 2>/dev/null || echo 0)"
MEM_GB="$("$PY" -c 'import sys; b=int(sys.argv[1]); print(f"{b/2**30:.0f}" if b else "unknown")' "$MEM_BYTES")"

"$PY" - "$METRICS_TSV" "$JSON_OUT" "$MD_OUT" <<EOF
import json, sys, platform, datetime

tsv, json_out, md_out = sys.argv[1], sys.argv[2], sys.argv[3]
metrics = {}
with open(tsv) as f:
    for line in f:
        key, _, val = line.rstrip("\n").partition("\t")
        metrics[key] = json.loads(val)

report = {
    "machine": {
        "measured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "cpu": """$CPU""".strip(),
        "mem_gb": """$MEM_GB""".strip(),
        "git_rev": "$GIT_REV",
        "binary": "${BIN:-none}",
        "binary_note": "$BIN_NOTE",
        "caveat": "$DEBUG_CAVEAT" or None,
    },
    "metrics": metrics,
}
with open(json_out, "w") as f:
    json.dump(report, f, indent=2)
    f.write("\n")

m = report["machine"]
prov = f'{m["cpu"]}, {m["mem_gb"]} GB, {m["os"]}, rev {m["git_rev"]}'
if m["caveat"]:
    prov += f' — {m["caveat"]}'

def cell(entry):
    if "skipped" in entry:
        return f'skipped: {entry["skipped"]}'
    if isinstance(entry["value"], bool):
        return "yes" if entry["value"] else "**NO**"
    unit = entry.get("unit", "")
    return f'{entry["value"]} {unit}'.strip()

rows = [
    ("Studio install (.dmg)", "dmg_size_mb"),
    ("Python wheel", "wheel_size_mb"),
    ("Cold CLI → FK on a real robot", "cold_fk_ms"),
    ("Robot load (panda URDF)", "robot_load_ms"),
    ("Peak RSS, plan + sim", "peak_rss_mb"),
    ("pip install (wheel, offline)", "pip_install_s"),
    ("Record overhead vs realtime", "record_overhead_x"),
    ("Seeded plan bit-identical", "seeded_plan_deterministic"),
    ("python -c 'import caliper'", "python_import_ms"),
]
lines = [
    "| Metric | Measured | Measured on |",
    "|---|---|---|",
]
for label, key in rows:
    lines.append(f'| {label} | {cell(metrics[key])} | {prov} |')
md = "\n".join(lines) + "\n"
with open(md_out, "w") as f:
    f.write(md)
print(md)
print(f"wrote {json_out}")
print(f"wrote {md_out}")
EOF
