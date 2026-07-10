#!/usr/bin/env bash
# Fetch the PINNED MuJoCo 3.9.0 shared library that `caliper-sim-mujoco`
# (cargo feature `mujoco`, via mujoco-rs 5.0.0) links against. mujoco-rs never
# builds or downloads MuJoCo itself on macOS — this script does, from the
# official pinned GitHub release, into a cache dir, and prints the env vars
# needed for build + run.
#
# Usage:  scripts/fetch_mujoco.sh [dest-dir]
# Then:   export MUJOCO_DYNAMIC_LINK_DIR=<dest>
#         export DYLD_LIBRARY_PATH=<dest>:$DYLD_LIBRARY_PATH   # macOS
#         export LD_LIBRARY_PATH=<dest>:$LD_LIBRARY_PATH       # Linux
#         cargo test -p caliper-sim-mujoco --features mujoco
set -euo pipefail

MJ_VERSION="3.9.0"
DEST="${1:-$HOME/.cache/caliper/mujoco-$MJ_VERSION}"
BASE="https://github.com/google-deepmind/mujoco/releases/download/$MJ_VERSION"

mkdir -p "$DEST"
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin)
    # macOS ships as a dmg wrapping mujoco.framework (universal2). We extract
    # the versioned dylib and add an UNVERSIONED symlink because `-lmujoco`
    # resolves `libmujoco.dylib`.
    if [ -f "$DEST/libmujoco.$MJ_VERSION.dylib" ]; then
      echo "already fetched: $DEST/libmujoco.$MJ_VERSION.dylib"
    else
      DMG="$DEST/mujoco-$MJ_VERSION-macos-universal2.dmg"
      curl -fL --retry 3 -o "$DMG" "$BASE/mujoco-$MJ_VERSION-macos-universal2.dmg"
      MNT="$(mktemp -d)"
      hdiutil attach -nobrowse -readonly -mountpoint "$MNT" "$DMG" >/dev/null
      trap 'hdiutil detach "$MNT" >/dev/null || true' EXIT
      # The framework wraps a versioned dylib; copy whatever it calls itself.
      FOUND="$(find "$MNT" -name "libmujoco*.dylib" -type f | head -n1)"
      [ -n "$FOUND" ] || { echo "no libmujoco dylib inside the dmg" >&2; exit 1; }
      cp "$FOUND" "$DEST/"
      hdiutil detach "$MNT" >/dev/null
      trap - EXIT
      rm -f "$DMG"
    fi
    ( cd "$DEST" && ln -sf "$(basename "$(ls libmujoco*.dylib | grep -v '^libmujoco\.dylib$' | head -n1)")" libmujoco.dylib )
    RUNVAR="DYLD_LIBRARY_PATH"
    ;;
  Linux)
    case "$ARCH" in
      x86_64)  TARBALL="mujoco-$MJ_VERSION-linux-x86_64.tar.gz" ;;
      aarch64) TARBALL="mujoco-$MJ_VERSION-linux-aarch64.tar.gz" ;;
      *) echo "unsupported Linux arch: $ARCH" >&2; exit 1 ;;
    esac
    if ls "$DEST"/libmujoco.so* >/dev/null 2>&1; then
      echo "already fetched: $DEST"
    else
      curl -fL --retry 3 -o "$DEST/$TARBALL" "$BASE/$TARBALL"
      tar -xzf "$DEST/$TARBALL" -C "$DEST" --strip-components=1
      rm -f "$DEST/$TARBALL"
      # The .so lives in lib/; flatten it next to the symlink -lmujoco wants.
      if [ -d "$DEST/lib" ]; then cp "$DEST"/lib/libmujoco.so* "$DEST/"; fi
    fi
    ( cd "$DEST" && VER="$(ls libmujoco.so.* 2>/dev/null | head -n1)" && [ -n "$VER" ] && ln -sf "$VER" libmujoco.so || true )
    RUNVAR="LD_LIBRARY_PATH"
    ;;
  *)
    echo "unsupported OS: $OS" >&2; exit 1 ;;
esac

# macOS: the release dylib's install name is framework-relative
# (@rpath/mujoco.framework/...), and SIP strips DYLD_LIBRARY_PATH through
# /usr/bin/env — so rewrite the ID to the absolute cache path (no env needed
# at runtime) and ad-hoc re-sign (an invalidated signature SIGKILLs on
# Apple Silicon).
if [ "$OS" = "Darwin" ] && [ -f "$DEST/libmujoco.$MJ_VERSION.dylib" ]; then
  install_name_tool -id "$DEST/libmujoco.$MJ_VERSION.dylib" "$DEST/libmujoco.$MJ_VERSION.dylib" 2>/dev/null
  codesign --force -s - "$DEST/libmujoco.$MJ_VERSION.dylib"
fi

echo
echo "MuJoCo $MJ_VERSION ready in: $DEST"
echo "export MUJOCO_DYNAMIC_LINK_DIR=\"$DEST\""
echo "export $RUNVAR=\"$DEST:\${$RUNVAR:-}\""
