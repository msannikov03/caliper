# License plan

Caliper uses a **split license**, matching the kind of artifact rather than
applying one blanket license to everything. The corresponding `LICENSE-*` files
live at the repository root.

| Artifact | License | File |
|----------|---------|------|
| **Software** — the Rust engine, the three faces, and tooling | **Apache-2.0** | `LICENSE-APACHE` |
| **Hardware** — any open-hardware designs | **CERN-OHL-W** (weakly-reciprocal open hardware) | `LICENSE-CERN-OHL-W` |
| **Documentation** — docs and written material (including this book) | **CC-BY** | `LICENSE-CC-BY` |

## Why split

- **Apache-2.0** is a permissive, patent-grant software license — appropriate for
  an engine meant to be embedded and built on.
- **CERN-OHL-W** is the standard weakly-reciprocal license for open hardware —
  the right instrument for physical designs, which Apache/CC do not cover well.
- **CC-BY** is the natural fit for prose and documentation.

## Status

These are **repo-level** `LICENSE-*` files per artifact type, not per-crate
`license` fields. During the build-out, individual crate metadata may still carry
a permissive `MIT OR Apache-2.0` placeholder while the split-license files are
finalized; the intent above is the plan of record.
