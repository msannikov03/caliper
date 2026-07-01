import React from "react";
import ReactDOM from "react-dom/client";

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
