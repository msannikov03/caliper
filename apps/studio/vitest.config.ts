import { defineConfig } from "vitest/config";

// Headless unit-test harness for Studio frontend logic.
// Environment: jsdom (provides window/document/RAF/performance.now).
// All Tauri and rendering deps are mocked in the individual test files.
export default defineConfig({
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
  },
});
