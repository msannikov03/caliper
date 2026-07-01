# The three faces

Every face is a thin shell over the same engine. They differ only in how you
reach the engine, not in what the engine does.

- **[CLI](./cli.md)** — the fastest way to try the engine from a shell; each
  subcommand parses arguments and calls the engine.
- **[Python](./python.md)** — `import caliper`, built with maturin/PyO3;
  scriptable like MATLAB/NumPy, and the surface the oracle runs through.
- **[Studio](./studio.md)** — *Caliper Studio*, a Tauri + React desktop app with
  a 3D scene and a dataflow node editor.

Because there is one implementation of the math, a result computed via the CLI,
via Python, and via a Studio graph node is the same result.
