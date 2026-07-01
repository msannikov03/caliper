// Global test setup for the Studio headless unit tests.
//
// jsdom provides: window, document, requestAnimationFrame, cancelAnimationFrame,
// performance.now — no extra polyfills are needed for the logic-only tests here.
//
// Per-module mocks (Tauri IPC, @xyflow/react) live in the individual test files
// via vi.mock so each suite is self-contained.
