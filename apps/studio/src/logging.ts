// Webview-side crash observability helpers (pure logic, headless-testable).
//
// main.tsx forwards uncaught errors + unhandled promise rejections to the
// tauri-plugin-log backend so they land in the same rotating log file as the
// Rust side (macOS: ~/Library/Logs/com.sannikov.studio/studio.log). These
// helpers keep that path pure: message formatting + a rate-limit dedupe guard
// so an error thrown in a loop (e.g. per-RAF) can't flood the log file.

/// Returns a guard: `true` = log this message, `false` = suppress (the same
/// message was already logged within the last `windowMs`). Distinct messages
/// are tracked independently; the map is pruned so memory stays bounded.
export function makeDedupe(
  windowMs = 1000,
  now: () => number = Date.now,
): (key: string) => boolean {
  const lastSent = new Map<string, number>();
  return (key: string): boolean => {
    const t = now();
    const prev = lastSent.get(key);
    if (prev !== undefined && t - prev < windowMs) return false;
    // Prune expired entries opportunistically so a stream of unique messages
    // (e.g. errors embedding a counter) cannot grow the map without bound.
    if (lastSent.size >= 256) {
      for (const [k, v] of lastSent) {
        if (t - v >= windowMs) lastSent.delete(k);
      }
      // Pathological case: 256 distinct messages inside one window — reset
      // rather than grow (losing dedupe state is better than losing memory).
      if (lastSent.size >= 256) lastSent.clear();
    }
    lastSent.set(key, t);
    return true;
  };
}

/// Human-readable one-liner for an uncaught `window.onerror` event.
export function formatErrorEvent(ev: {
  message?: string;
  filename?: string;
  lineno?: number;
  colno?: number;
  error?: unknown;
}): string {
  const where = ev.filename
    ? ` at ${ev.filename}:${ev.lineno ?? 0}:${ev.colno ?? 0}`
    : "";
  const stack =
    ev.error instanceof Error && ev.error.stack ? `\n${ev.error.stack}` : "";
  return `uncaught error: ${ev.message ?? "<no message>"}${where}${stack}`;
}

/// Human-readable one-liner for an unhandled promise rejection reason.
export function formatRejection(reason: unknown): string {
  if (reason instanceof Error) {
    return `unhandled rejection: ${reason.message}${reason.stack ? `\n${reason.stack}` : ""}`;
  }
  if (typeof reason === "string") return `unhandled rejection: ${reason}`;
  try {
    return `unhandled rejection: ${JSON.stringify(reason)}`;
  } catch {
    return `unhandled rejection: ${String(reason)}`;
  }
}
