//! Asset doctor: diagnose and repair URDF robot descriptions.
//!
//! Real-world URDFs — CAD exports above all — routinely carry defects that
//! the rest of the stack only surfaces late, one at a time, or not at all: a
//! silently dropped collider here, `has_inertia = false` there, an MJCF
//! export that MuJoCo rejects. [`diagnose`] runs every check in ONE pass and
//! returns a [`DoctorReport`] of plain-English [`Finding`]s (each naming the
//! offending field, its value, and the consequence); [`repair`] emits a
//! REPAIRED COPY — the input file is never touched — for the defects that
//! have a mechanical fix.
//!
//! The doctor parses XML itself (leniently) instead of going through urdf-rs:
//! half the point is diagnosing files urdf-rs REJECTS outright, like a
//! `<limit>` without `velocity=` or a `.urdf` full of xacro leftovers.
//! `.xacro` input is expanded first via `caliper_model::xacro`.
//!
//! Checks (stable codes — each has BOTH a positive and a negative test):
//!
//! | code | severity | check | auto-fix ([`RepairOpts`]) |
//! |---|---|---|---|
//! | A001 | Error | missing/zero `<inertial>` on a non-root link | `compute_inertials` (divergence-theorem mesh integrals / analytic primitives × density) |
//! | A002 | Error | implausible inertia: non-finite, zero tensor with mass, negative principal moment, or triangle-inequality violation (checked on EIGENVALUES, so converter-dropped off-diagonals are caught) | — |
//! | A003 | Error/Warn | mesh unresolvable (says which paths were tried) or unloadable — Error on `<collision>` (collider silently dropped), Warn on `<visual>` | — |
//! | A004 | Warn | duplicate mesh basenames pointing at DIFFERENT files | `dedupe_mesh_basenames` (rename + copy plan) |
//! | A005 | Warn | link has `<visual>` but no `<collision>` (uncheckable) | — |
//! | A006 | Info | collision mesh above the 1024-vertex hull cap (subsampled) | — |
//! | A007 | Warn | revolute joint without usable position limits | `inject_limits` (±π, marked conservative) |
//! | A008 | Error | zero-length / unparseable joint axis | — |
//! | A009 | Warn | non-unit joint axis | `normalize_axes` |
//! | A010 | Info | `[heuristic]` zero-mass root link — onshape-to-robot signature | — |
//! | A011 | Error | mimic references an unknown joint | — |
//! | A012 | Error | mimic chain (incl. self-mimic) | — |
//! | A013 | Error/Warn | xacro leftovers in a `.urdf` (Warn when `xmlns:xacro` is declared and expansion succeeds) | — |
//! | A014 | Error | `<limit>` missing `velocity=` (urdf-rs rejects the whole file; `effort=` safely defaults to 0) | `inject_limits` |

use std::path::{Path, PathBuf};

mod checks;
mod massprops;
mod repair;
mod resolve;
mod xml;

pub use repair::{MeshCopy, RepairAction, RepairOpts, RepairOutcome, repair};

/// Stable finding codes. String-typed on [`Finding`] so reports serialize
/// naturally; these constants are the single source of truth.
pub mod codes {
    pub const MISSING_INERTIAL: &str = "A001";
    pub const IMPLAUSIBLE_INERTIA: &str = "A002";
    pub const MESH_UNRESOLVABLE: &str = "A003";
    pub const DUPLICATE_MESH_BASENAME: &str = "A004";
    pub const VISUAL_WITHOUT_COLLISION: &str = "A005";
    pub const COLLISION_MESH_HUGE: &str = "A006";
    pub const REVOLUTE_NO_LIMITS: &str = "A007";
    pub const ZERO_AXIS: &str = "A008";
    pub const AXIS_NOT_NORMALIZED: &str = "A009";
    pub const CAD_ZERO_MASS_ROOT: &str = "A010";
    pub const MIMIC_UNKNOWN_SOURCE: &str = "A011";
    pub const MIMIC_CHAIN: &str = "A012";
    pub const XACRO_LEFTOVERS: &str = "A013";
    pub const LIMIT_MISSING_ATTRS: &str = "A014";
}

/// Finding severity: how broken things are if the finding is ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Severity {
    /// Blocks compiling, exporting, or simulating — or yields wrong physics.
    Error,
    /// Loads fine but behaves worse than the user thinks (dropped coverage,
    /// unbounded joints, portability traps).
    Warn,
    /// Worth knowing; nothing is wrong per se.
    Info,
}

impl Severity {
    fn rank(self) -> u8 {
        match self {
            Severity::Error => 0,
            Severity::Warn => 1,
            Severity::Info => 2,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Severity::Error => "ERROR",
            Severity::Warn => "WARN",
            Severity::Info => "INFO",
        }
    }
}

/// One diagnosed defect. `message` names the field and value and states the
/// consequence; `fix_hint` says what to do about it; `auto_fixable` means a
/// [`RepairOpts`] flag fixes it mechanically.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Finding {
    /// Stable check id from [`codes`].
    pub code: String,
    pub severity: Severity,
    pub message: String,
    pub fix_hint: Option<String>,
    pub auto_fixable: bool,
}

impl Finding {
    fn new(code: &str, severity: Severity, message: String) -> Self {
        Finding {
            code: code.to_string(),
            severity,
            message,
            fix_hint: None,
            auto_fixable: false,
        }
    }
    pub(crate) fn error(code: &str, message: String) -> Self {
        Self::new(code, Severity::Error, message)
    }
    pub(crate) fn warn(code: &str, message: String) -> Self {
        Self::new(code, Severity::Warn, message)
    }
    pub(crate) fn info(code: &str, message: String) -> Self {
        Self::new(code, Severity::Info, message)
    }
    pub(crate) fn hint(mut self, hint: &str) -> Self {
        self.fix_hint = Some(hint.to_string());
        self
    }
    pub(crate) fn auto(mut self) -> Self {
        self.auto_fixable = true;
        self
    }
}

/// The result of [`diagnose`]: findings sorted most-severe-first plus counts.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DoctorReport {
    pub findings: Vec<Finding>,
    pub errors: usize,
    pub warnings: usize,
    pub infos: usize,
}

impl DoctorReport {
    /// Sort findings (severity, then code) and tally the counts.
    pub fn new(mut findings: Vec<Finding>) -> Self {
        findings.sort_by(|a, b| {
            (a.severity.rank(), a.code.as_str()).cmp(&(b.severity.rank(), b.code.as_str()))
        });
        let count = |s: Severity| findings.iter().filter(|f| f.severity == s).count();
        DoctorReport {
            errors: count(Severity::Error),
            warnings: count(Severity::Warn),
            infos: count(Severity::Info),
            findings,
        }
    }

    pub fn is_clean(&self) -> bool {
        self.findings.is_empty()
    }
    pub fn has_errors(&self) -> bool {
        self.errors > 0
    }

    /// Human-readable report, grouped by severity, one finding per entry with
    /// its code, fix hint, and an `(auto-fixable)` marker where repair helps.
    pub fn render_text(&self) -> String {
        if self.is_clean() {
            return "asset doctor: no findings — the description looks healthy\n".to_string();
        }
        let mut out = format!(
            "asset doctor: {} error(s), {} warning(s), {} info(s)\n",
            self.errors, self.warnings, self.infos
        );
        for sev in [Severity::Error, Severity::Warn, Severity::Info] {
            let group: Vec<&Finding> = self.findings.iter().filter(|f| f.severity == sev).collect();
            if group.is_empty() {
                continue;
            }
            out.push_str(&format!("\n{} ({})\n", sev.label(), group.len()));
            for f in group {
                out.push_str(&format!("  [{}] {}\n", f.code, f.message));
                if let Some(hint) = &f.fix_hint {
                    let auto = if f.auto_fixable {
                        " (auto-fixable)"
                    } else {
                        ""
                    };
                    out.push_str(&format!("         fix: {hint}{auto}\n"));
                }
            }
        }
        out
    }

    /// The report as pretty JSON (fields as in the struct).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("plain data serializes")
    }
}

/// Why a description could not even be inspected (defects found DURING
/// inspection are [`Finding`]s, not errors).
#[derive(thiserror::Error, Debug)]
pub enum DoctorError {
    #[error("read {path:?}: {err}")]
    Io { path: PathBuf, err: String },
    #[error("{path:?} is not parseable XML: {err}")]
    Xml { path: PathBuf, err: String },
    #[error("{path:?}: root element is `<{found}>`, expected `<robot>`")]
    NotARobot { path: PathBuf, found: String },
    #[error("xacro expansion of {path:?} failed: {err}")]
    Xacro { path: PathBuf, err: String },
}

/// A loaded description: the (possibly xacro-expanded) `<robot>` DOM, findings
/// produced during loading (A013), and the base dir for mesh resolution.
pub(crate) struct Loaded {
    pub robot: xml::Element,
    pub pre: Vec<Finding>,
    pub dir: Option<PathBuf>,
}

pub(crate) fn load(path: &Path) -> Result<Loaded, DoctorError> {
    let text = std::fs::read_to_string(path).map_err(|e| DoctorError::Io {
        path: path.to_path_buf(),
        err: e.to_string(),
    })?;
    let dir = path.parent().map(Path::to_path_buf);
    let parse = |t: &str| {
        let root = xml::parse_document(t).map_err(|e| DoctorError::Xml {
            path: path.to_path_buf(),
            err: e.to_string(),
        })?;
        if root.name != "robot" {
            return Err(DoctorError::NotARobot {
                path: path.to_path_buf(),
                found: root.name,
            });
        }
        Ok(root)
    };
    let expand = |t: &str| {
        caliper_model::xacro::expand(t, dir.as_deref()).map_err(|e| DoctorError::Xacro {
            path: path.to_path_buf(),
            err: e.to_string(),
        })
    };

    // a .xacro file is simply expanded — leftovers are its NORMAL content
    let ext_is_xacro = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("xacro"));
    if ext_is_xacro {
        let robot = parse(&expand(&text)?)?;
        return Ok(Loaded {
            robot,
            pre: Vec::new(),
            dir,
        });
    }

    let raw = parse(&text)?;
    // A013 triage for .urdf files: with xmlns:xacro caliper CAN expand it (a
    // portability Warn, and we diagnose the expansion); without, leftovers mean
    // the file will not load anywhere (Error, diagnose the raw DOM best-effort).
    if raw.attr("xmlns:xacro").is_some() {
        return match expand(&text) {
            Ok(expanded) => {
                let robot = parse(&expanded)?;
                let pre = vec![
                    Finding::warn(
                        codes::XACRO_LEFTOVERS,
                        format!(
                            "`{}` is a .urdf that declares xmlns:xacro: caliper expands it \
                         in-process, but most URDF consumers will not — they will see raw \
                         xacro constructs (the rest of this report describes the EXPANDED \
                         model)",
                            path.display()
                        ),
                    )
                    .hint("run the file through xacro once and ship the expanded .urdf"),
                ];
                Ok(Loaded { robot, pre, dir })
            }
            Err(e) => {
                let pre = vec![Finding::error(
                    codes::XACRO_LEFTOVERS,
                    format!(
                        "`{}` is a .urdf with xacro content that even caliper's expander \
                         rejects ({e}): no URDF consumer will load this file (this report \
                         covers the raw, unexpanded document)",
                        path.display()
                    ),
                )];
                Ok(Loaded {
                    robot: raw,
                    pre,
                    dir,
                })
            }
        };
    }
    let leftovers = checks::xacro_leftovers(&raw);
    let mut pre = Vec::new();
    if !leftovers.is_empty() {
        pre.push(
            Finding::error(
                codes::XACRO_LEFTOVERS,
                format!(
                    "{} xacro leftover(s) in a plain .urdf (no xmlns:xacro), e.g. {}: URDF \
                     parsers (urdf-rs included) will fail on the tags or silently misread \
                     the ${{..}} values",
                    leftovers.len(),
                    leftovers[0]
                ),
            )
            .hint("this file looks like an unexpanded xacro renamed to .urdf — expand it"),
        );
    }
    Ok(Loaded {
        robot: raw,
        pre,
        dir,
    })
}

/// Diagnose the URDF (or xacro) description at `path`. Errors only when the
/// file cannot even be inspected; every defect found IN it is a [`Finding`].
pub fn diagnose(path: &Path) -> Result<DoctorReport, DoctorError> {
    let loaded = load(path)?;
    let mut findings = loaded.pre;
    findings.extend(checks::run(&loaded.robot, loaded.dir.as_deref()));
    Ok(DoctorReport::new(findings))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> DoctorReport {
        DoctorReport::new(vec![
            Finding::info(codes::CAD_ZERO_MASS_ROOT, "an info".to_string()),
            Finding::warn(codes::REVOLUTE_NO_LIMITS, "a warn".to_string())
                .hint("inject")
                .auto(),
            Finding::error(codes::MISSING_INERTIAL, "an error".to_string()).hint("compute"),
        ])
    }

    #[test]
    fn report_sorts_most_severe_first_and_counts() {
        let r = sample();
        assert_eq!((r.errors, r.warnings, r.infos), (1, 1, 1));
        assert_eq!(r.findings[0].severity, Severity::Error);
        assert_eq!(r.findings[2].severity, Severity::Info);
        assert!(r.has_errors() && !r.is_clean());
        assert!(DoctorReport::new(Vec::new()).is_clean());
    }

    #[test]
    fn render_text_groups_by_severity_with_codes_and_hints() {
        let text = sample().render_text();
        let (e, w) = (
            text.find("ERROR (1)").unwrap(),
            text.find("WARN (1)").unwrap(),
        );
        assert!(e < w && w < text.find("INFO (1)").unwrap());
        assert!(text.contains("[A001] an error"));
        assert!(text.contains("fix: inject (auto-fixable)"));
        assert!(text.contains("1 error(s), 1 warning(s), 1 info(s)"));
        let clean = DoctorReport::new(Vec::new()).render_text();
        assert!(clean.contains("no findings"));
    }

    #[test]
    fn report_round_trips_through_json() {
        let r = sample();
        let back: DoctorReport = serde_json::from_str(&r.to_json()).unwrap();
        assert_eq!(back.findings.len(), 3);
        assert_eq!(back.errors, 1);
        assert_eq!(back.findings[0].code, codes::MISSING_INERTIAL);
        assert_eq!(back.findings[1].fix_hint.as_deref(), Some("inject"));
        assert!(back.findings[1].auto_fixable);
    }
}
