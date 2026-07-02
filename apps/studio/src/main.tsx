import React from "react";
import ReactDOM from "react-dom/client";
import { error as logError } from "@tauri-apps/plugin-log";
import { formatErrorEvent, formatRejection, makeDedupe } from "./logging";

// Forward uncaught webview errors + unhandled rejections to the backend log
// file (macOS: ~/Library/Logs/com.sannikov.studio/studio.log) so frontend
// crashes leave the same trace as Rust panics. Registered before React mounts
// so even render-time crashes are captured. Deliberately NOT hooking
// console.error (too noisy) — only genuinely uncaught failures. The dedupe
// guard suppresses repeats of the same message within 1 s so an error loop
// can't flood the file; `.catch` keeps a broken IPC bridge from spawning its
// own unhandled rejections (which would recurse into this handler).
const shouldLog = makeDedupe(1000);
window.addEventListener("error", (ev) => {
  const msg = formatErrorEvent(ev);
  if (shouldLog(msg)) void logError(msg).catch(() => {});
});
window.addEventListener("unhandledrejection", (ev) => {
  const msg = formatRejection(ev.reason);
  if (shouldLog(msg)) void logError(msg).catch(() => {});
});

// Offline-safe fonts for Tauri (no Google Fonts CDN) — Inter (UI) + JetBrains Mono (numerics/eyebrows).
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/inter/700.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "@fontsource/jetbrains-mono/600.css";
import "@fontsource/jetbrains-mono/700.css";

// Design tokens + base body styles load before the app's own stylesheet.
import "./design/tokens.css";

import App from "./App";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
