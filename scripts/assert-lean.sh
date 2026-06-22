#!/usr/bin/env bash
# Fail if the lightweight core crate pulls a heavy/GPU/sim/transport dependency.
set -euo pipefail
banned='parry|rapier|socketcan|tch|torch|wgpu|mujoco|tokio'
tree="$(cargo tree -p caliper --edges normal 2>/dev/null)"
if echo "$tree" | grep -Eiq "$banned"; then
  echo "FAIL: core crate 'caliper' pulled a heavy dependency:"
  echo "$tree" | grep -Ei "$banned"
  exit 1
fi
echo "core is lean ✓ (no heavy deps in caliper)"
