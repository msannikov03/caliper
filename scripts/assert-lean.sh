#!/usr/bin/env bash
# Fail if the lightweight core crate pulls a heavy/GPU/sim/transport dependency.
set -euo pipefail
# Banlist covers GPU/sim/transport AND the dataset stack (arrow/parquet): all are
# opt-in features, so the DEFAULT facade tree must show none of them.
banned='parry|rapier|socketcan|serialport|arrow|parquet|tch|torch|wgpu|mujoco|tokio'
tree="$(cargo tree -p caliper --edges normal 2>/dev/null)"
if echo "$tree" | grep -Eiq "$banned"; then
  echo "FAIL: core crate 'caliper' pulled a heavy dependency:"
  echo "$tree" | grep -Ei "$banned"
  exit 1
fi
echo "core is lean ✓ (no heavy deps in caliper)"
