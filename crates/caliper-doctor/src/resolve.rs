//! Mesh-path resolution that also REPORTS what was tried.
//!
//! ⚠ LOCKSTEP MIRROR of `caliper_model::resolve_mesh_path` /
//! `package_search_dirs` (those are `pub(crate)` there, and the doctor needs
//! the failed CANDIDATES, which the model never records). If the model's
//! search order changes, change it here identically — the whole value of the
//! A003 finding is telling the user exactly where caliper looked.

use std::path::{Path, PathBuf};

/// Outcome of resolving one `<mesh filename=...>` reference.
pub struct Resolution {
    /// Absolute (canonicalized) path of the first existing candidate, or `None`.
    pub resolved: Option<PathBuf>,
    /// Every candidate path that was checked, in search order.
    pub tried: Vec<PathBuf>,
}

/// Resolve a URDF mesh `filename` exactly like `caliper_model::resolve_mesh_path`:
/// `file://` stripped; plain relative → `urdf_dir/<raw>`; absolute → itself;
/// `package://<pkg>/<rest>` → `urdf_dir/<rest>`, then for every search root `A`
/// (urdf dir + up to 6 ancestor levels, then each `CALIPER_PACKAGE_PATH` root):
/// `A/<pkg>/<rest>` and `A/<rest>`. First existing file wins.
pub fn resolve_mesh(raw: &str, urdf_dir: Option<&Path>) -> Resolution {
    let name = raw.strip_prefix("file://").unwrap_or(raw);
    if let Some(pkg_rest) = name.strip_prefix("package://") {
        let Some((pkg, rest)) = pkg_rest.split_once('/') else {
            return Resolution {
                resolved: None,
                tried: Vec::new(),
            };
        };
        if pkg.is_empty() || rest.is_empty() {
            return Resolution {
                resolved: None,
                tried: Vec::new(),
            };
        }
        let mut tried: Vec<PathBuf> = Vec::new();
        if let Some(dir) = urdf_dir {
            tried.push(dir.join(rest));
        }
        for a in package_search_dirs(urdf_dir) {
            tried.push(a.join(pkg).join(rest));
            tried.push(a.join(rest));
        }
        let resolved = tried.iter().find(|c| c.is_file()).cloned().map(absolutize);
        return Resolution { resolved, tried };
    }
    let p = Path::new(name);
    let cand = if p.is_absolute() {
        p.to_path_buf()
    } else {
        match urdf_dir {
            Some(dir) => dir.join(p),
            None => {
                return Resolution {
                    resolved: None,
                    tried: Vec::new(),
                };
            }
        }
    };
    let resolved = cand.is_file().then(|| absolutize(cand.clone()));
    Resolution {
        resolved,
        tried: vec![cand],
    }
}

/// Mirror of `caliper_model::package_search_dirs`: the URDF dir's ancestors
/// (itself first, then up to 6 parent levels), then every root in
/// `CALIPER_PACKAGE_PATH` (colon-separated).
fn package_search_dirs(urdf_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(dir) = urdf_dir {
        dirs.extend(dir.ancestors().take(7).map(Path::to_path_buf));
    }
    if let Ok(roots) = std::env::var("CALIPER_PACKAGE_PATH") {
        dirs.extend(
            roots
                .split(':')
                .filter(|s| !s.is_empty())
                .map(PathBuf::from),
        );
    }
    dirs
}

/// Mirror of `caliper_model::absolutize`.
fn absolutize(p: PathBuf) -> PathBuf {
    std::fs::canonicalize(&p).unwrap_or(p)
}

/// Render a tried-candidates list for a finding message, capped so a
/// package:// miss (15+ candidates) stays readable.
pub fn fmt_tried(tried: &[PathBuf]) -> String {
    if tried.is_empty() {
        return "nothing (the reference is malformed or there is no base directory)".to_string();
    }
    const CAP: usize = 6;
    let mut parts: Vec<String> = tried
        .iter()
        .take(CAP)
        .map(|p| format!("`{}`", p.display()))
        .collect();
    if tried.len() > CAP {
        parts.push(format!("… and {} more", tried.len() - CAP));
    }
    parts.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn robots_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures/robots")
    }

    #[test]
    fn resolves_relative_and_dot_relative_to_same_file() {
        let dir = robots_dir();
        let a = resolve_mesh("unit_cube.stl", Some(&dir)).resolved.unwrap();
        let b = resolve_mesh("./unit_cube.stl", Some(&dir))
            .resolved
            .unwrap();
        assert_eq!(a, b, "canonicalization unifies spellings");
    }

    #[test]
    fn package_uri_resolves_via_ancestor_search_like_the_model() {
        // demo_pkg sits one level above robots/, same as the caliper-model test.
        let dir = robots_dir();
        let r = resolve_mesh("package://demo_pkg/meshes/part.stl", Some(&dir));
        assert!(r.resolved.is_some(), "tried: {:?}", r.tried);
    }

    #[test]
    fn miss_records_candidates() {
        let dir = robots_dir();
        let r = resolve_mesh("missing/nope.stl", Some(&dir));
        assert!(r.resolved.is_none());
        assert_eq!(r.tried.len(), 1);
        assert!(r.tried[0].ends_with("missing/nope.stl"));
        let msg = fmt_tried(&r.tried);
        assert!(msg.contains("nope.stl"), "{msg}");
    }
}
