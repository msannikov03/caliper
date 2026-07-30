# Releasing Caliper (macOS, from this machine)

The release unit is a **signed + notarized dmg with auto-update artifacts**,
published to a GitHub release. Users install once; the app offers later
versions itself (toolbar chip) from the signed `latest.json` on the newest
release.

## Secrets (never in the repo)

| What | Where | Notes |
|---|---|---|
| Developer ID cert | login keychain | `Developer ID Application: Business Lion LLC (2RHJ5N9TTT)` |
| Notary API key | `~/Downloads/AuthKey_5D3A7A3552.p8` | Key ID `5D3A7A3552`, issuer `6c69efbc-e664-4159-a39f-4e7f776b3198`; pass DIRECTLY (`--key/--key-id/--issuer`) — keychain profiles are unreadable non-interactively |
| Updater signing key | `~/Downloads/caliper-updater.key` (+ `.pub`) | minisign; pubkey is baked into `tauri.conf.json`. **Losing it means shipped apps can never auto-update again.** Keep 0600. |

## Flow

1. Bump the version in **five places** (they are gated for consistency):
   workspace `Cargo.toml`, `crates/caliper-py/pyproject.toml`,
   `apps/studio/src-tauri/tauri.conf.json`, `apps/studio/src-tauri/Cargo.toml`,
   `apps/studio/package.json`. Cut the `Unreleased` section of `CHANGELOG.md`
   into a dated release section.
2. Full gate sweep green, commit, push.
3. Build signed (codesign must run FOREGROUND — background shells cannot
   reach the keychain key):
   ```sh
   scripts/bundle_mujoco.sh
   cd apps/studio
   MUJOCO_DYNAMIC_LINK_DIR="$PWD/src-tauri/vendor" \
   APPLE_SIGNING_IDENTITY="Developer ID Application: Business Lion LLC (2RHJ5N9TTT)" \
   TAURI_SIGNING_PRIVATE_KEY_PATH="$HOME/Downloads/caliper-updater.key" \
   npm run tauri build -- --features mujoco --config "$PWD/src-tauri/tauri.mujoco.conf.json"
   ```
   Artifacts land in `target/release/bundle/`: the `.dmg`, and (because
   `createUpdaterArtifacts` is on) `macos/Caliper Studio.app.tar.gz` + `.sig`.
4. Notarize the dmg (submit WITHOUT `--wait` if a Tailscale exit node is
   active — long polls drop; poll `notarytool info` instead), then staple the
   app and the dmg, then `spctl --assess` both.
5. Write `latest.json` (the auto-update manifest):
   ```json
   { "version": "X.Y.Z", "pub_date": "<RFC3339>",
     "platforms": { "darwin-aarch64": {
       "signature": "<contents of the .sig file>",
       "url": "https://github.com/msannikov03/caliper/releases/download/vX.Y.Z/Caliper.Studio_aarch64.app.tar.gz" } } }
   ```
6. Tag + release: `git tag vX.Y.Z && git push origin vX.Y.Z`, then
   `gh release create vX.Y.Z <dmg> <app.tar.gz> latest.json --title … --notes …`.
   The tag also triggers the release workflow (version-check, wheels attached
   to the same release, PyPI publish if `PYPI_API_TOKEN` is set; the crates
   dry-run is non-blocking until a first real crates.io publish exists).
7. Keep **only the newest release public-facing**: older releases are deleted
   (tags stay). `releases/latest/download/latest.json` — the endpoint baked
   into shipped apps — always points at the newest release automatically.

## Auto-update facts

- The plugin verifies `latest.json`'s minisign signature against the baked-in
  pubkey before trusting anything; the `.tar.gz` signature is checked before
  install. Notarization is preserved (the update artifact is the signed app).
- Update = in-place `.app` swap + relaunch; user data is untouched (Studio
  keeps state in `localStorage` + OS log/config dirs, outside the bundle).
- The check runs once, ~4 s after launch, and degrades to a log line offline.
- `CFBundleVersion`-style monotonicity: the updater compares semver — the
  shipped version must strictly increase.
