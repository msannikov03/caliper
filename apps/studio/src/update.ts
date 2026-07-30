// ============================================================
// update.ts — auto-update surface over the tauri updater plugin.
// The plugin verifies latest.json's minisign signature against the
// pubkey baked into tauri.conf.json before anything is trusted; this
// module only sequences it: one silent check shortly after launch
// (result logged, so headless runs can prove the loop), then install
// strictly on the user's click. Nothing updates without consent.
// ============================================================
import { check, type Update } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";
import { info, warn } from "@tauri-apps/plugin-log";

export interface UpdateOffer {
  version: string;
  /** release notes body, if the manifest carries one */
  notes: string | null;
}

// Module-scope: the Update object holds the download handle; the offer is
// the render-safe projection of it.
let pending: Update | null = null;

/** One launch-time check. Network/manifest failures are logged, never thrown —
 *  an unreachable update endpoint must not degrade the app. */
export async function checkForUpdate(): Promise<UpdateOffer | null> {
  try {
    const u = await check();
    if (u) {
      pending = u;
      void info(`update check: v${u.version} available (running ${u.currentVersion})`);
      return { version: u.version, notes: u.body ?? null };
    }
    void info("update check: up to date");
    return null;
  } catch (e) {
    void warn(`update check failed (offline or endpoint unreachable): ${String(e)}`);
    return null;
  }
}

/** Download, verify (plugin-side signature check), install, relaunch.
 *  Throws on failure so the caller can surface it. */
export async function installPendingUpdate(): Promise<void> {
  if (!pending) throw new Error("no update pending — check first");
  void info(`update: downloading v${pending.version}`);
  await pending.downloadAndInstall();
  void info("update: installed, relaunching");
  await relaunch();
}

/** test seam */
export function _resetUpdate(): void {
  pending = null;
}
