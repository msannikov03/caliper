#!/usr/bin/env bash
# Stage libmujoco so the Caliper Studio .app/.dmg can SHIP contact sim (macOS).
#
# fetch_mujoco.sh solves the DEV problem: dylib in a cache dir with an
# ABSOLUTE install id, so `cargo test --features mujoco` finds it on this
# machine. This script solves the SHIP problem: the .app must carry the dylib
# itself and resolve it via @rpath on any machine.
#
# What it does:
#   1. fetches the pinned MuJoCo 3.9.0 dylib (reuses scripts/fetch_mujoco.sh
#      and its cache — no re-download if already fetched)
#   2. stages a COPY into apps/studio/src-tauri/vendor/  (gitignored)
#   3. rewrites the copy's install id to @rpath/libmujoco.3.9.0.dylib and
#      ad-hoc re-signs it (install_name_tool invalidates the signature, and an
#      invalidated signature SIGKILLs on Apple Silicon)
#   4. adds the unversioned libmujoco.dylib symlink that `-lmujoco` resolves
#      at LINK time (mujoco-rs looks for exactly $DIR/libmujoco.dylib)
#   5. prints the exact bundle-build command
#
# Why this works (verified against tauri-cli 2.11.3 / tauri-build 2.6.3 /
# mujoco-rs 5.0.0 sources):
#   - tauri.mujoco.conf.json is merged over tauri.conf.json by `--config`
#     (JSON Merge Patch, RFC 7396). It adds
#     bundle.macOS.frameworks = ["vendor/libmujoco.3.9.0.dylib"]
#     (frameworks paths resolve relative to src-tauri/). It lives in a
#     SEPARATE overlay conf because tauri-build hard-errors on a listed-but-
#     missing dylib — the default conf must keep building with nothing staged.
#   - tauri-cli hands the merged config to build scripts via TAURI_CONFIG;
#     tauri-build, seeing a non-empty frameworks list, (a) copies the dylib to
#     target/Frameworks (so dev runs work too) and (b) links the executable
#     with -Wl,-rpath,@executable_path/../Frameworks. No custom build.rs rpath
#     is needed.
#   - tauri-bundler copies each frameworks entry into Contents/Frameworks and
#     signs it together with the app. It does NOT rewrite install names —
#     which is why step 3 sets the id BEFORE linking: the executable records
#     the dylib's install id as its LC_LOAD_DYLIB, so linking against this
#     staged copy yields @rpath/libmujoco.3.9.0.dylib (linking against the
#     fetch cache would bake in an absolute per-machine path).
#   - MUJOCO_DYNAMIC_LINK_DIR must be an ABSOLUTE path containing
#     libmujoco.dylib (mujoco-rs build.rs panics otherwise).
#
# Usage:  scripts/bundle_mujoco.sh [cache-dir]
set -euo pipefail

MJ_VERSION="3.9.0"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDIO_TAURI="$REPO_ROOT/apps/studio/src-tauri"
VENDOR="$STUDIO_TAURI/vendor"
CACHE="${1:-$HOME/.cache/caliper/mujoco-$MJ_VERSION}"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "bundle_mujoco.sh is macOS-only (it stages the .app/.dmg dylib)." >&2
  echo "On Linux, use scripts/fetch_mujoco.sh + LD_LIBRARY_PATH for dev." >&2
  exit 1
fi

"$REPO_ROOT/scripts/fetch_mujoco.sh" "$CACHE"

SRC="$CACHE/libmujoco.$MJ_VERSION.dylib"
[ -f "$SRC" ] || { echo "expected $SRC after fetch — aborting" >&2; exit 1; }

mkdir -p "$VENDOR"
cp -f "$SRC" "$VENDOR/libmujoco.$MJ_VERSION.dylib"
install_name_tool -id "@rpath/libmujoco.$MJ_VERSION.dylib" \
  "$VENDOR/libmujoco.$MJ_VERSION.dylib"
codesign --force -s - "$VENDOR/libmujoco.$MJ_VERSION.dylib"
ln -sf "libmujoco.$MJ_VERSION.dylib" "$VENDOR/libmujoco.dylib"

echo
echo "staged: $VENDOR/libmujoco.$MJ_VERSION.dylib (install id @rpath/..., ad-hoc signed)"
echo
echo "Build the MuJoCo-enabled bundle with:"
echo
echo "  cd \"$REPO_ROOT/apps/studio\""
echo "  MUJOCO_DYNAMIC_LINK_DIR=\"$VENDOR\" npm run tauri build -- \\"
echo "    --features mujoco \\"
echo "    --config \"$STUDIO_TAURI/tauri.mujoco.conf.json\""
echo
echo "Note: env -u CONDA_PREFIX before npm/cargo if conda is active (repo rule)."
