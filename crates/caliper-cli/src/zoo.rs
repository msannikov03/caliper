//! The robot zoo: real-robot URDFs EMBEDDED in the binary.
//!
//! `caliper fetch <name>` materializes one of the vendored corpus URDFs
//! (`oracle/fixtures/corpus/`, licenses in its README) without any network —
//! the four files are small text and ship inside the executable via
//! `include_str!`. Meshes are deliberately NOT embedded (~58 MB): visuals fall
//! back (`path = None`), collision meshes drop LOUDLY via
//! `dropped_collider_frames`, and the kinematics stay exact.
//!
//! Honesty over polish: the files are vendored VERBATIM, so the upstream
//! defects (massless attachment frames) and the non-embedded meshes surface in
//! `caliper doctor` as Error findings. Each entry documents exactly which
//! codes to expect ([`ZooEntry::known_doctor_errors`]) instead of papering
//! over them, and the unit tests pin that set.
//!
//! NOTE (packaging): `include_str!` reaches OUTSIDE the crate root into the
//! repo's `oracle/fixtures/corpus/` — fine for workspace builds and CI, but
//! `cargo package`/`cargo publish` of caliper-cli would need the corpus
//! vendored under the crate first.

use std::path::{Path, PathBuf};

/// One embedded robot: the URDF text plus the metadata `--list` tables and
/// `fetch` prints (name/dof/license/source from the corpus README).
pub struct ZooEntry {
    /// Fetch key, also the output file stem.
    pub name: &'static str,
    /// Human-readable robot name.
    pub robot: &'static str,
    /// Output basename, `<name>.urdf`.
    pub file_name: &'static str,
    /// The vendored URDF text, verbatim.
    pub urdf: &'static str,
    /// Full-space dof — mimic joints counted, matching `Robot::ndof()`
    /// (pinned against the same files in `oracle/tests/test_corpus.py`).
    pub dof: usize,
    /// Upstream license (SPDX id).
    pub license: &'static str,
    /// Upstream repository + in-repo path the file was vendored from.
    pub source: &'static str,
    /// `caliper doctor` Error codes this file is KNOWN to report (sorted,
    /// deduped). Empty = doctor-clean. Documented honestly instead of
    /// silently shipping a file that fails the doctor.
    pub known_doctor_errors: &'static [&'static str],
    /// One line explaining WHY those errors are expected.
    pub doctor_note: &'static str,
}

/// One shared caveat line — the same trade-off for every entry.
pub const MESH_CAVEAT: &str = "meshes are NOT embedded — visuals fall back, \
     collision meshes drop LOUDLY (dropped_collider_frames), kinematics exact";

/// The registry. Order = the corpus README.
pub const REGISTRY: &[ZooEntry] = &[
    ZooEntry {
        name: "panda",
        robot: "Franka Emika Panda (arm + hand)",
        file_name: "panda.urdf",
        urdf: include_str!("../../../oracle/fixtures/corpus/panda.urdf"),
        dof: 9,
        license: "BSD-2-Clause",
        source: "https://github.com/Gepetto/example-robot-data \
                 (robots/panda_description/urdf/panda.urdf)",
        known_doctor_errors: &["A001", "A003"],
        doctor_note: "A001: panda_link8/panda_hand_tcp are massless attachment \
                      frames upstream; A003: collision meshes not embedded",
    },
    ZooEntry {
        name: "so101_new_calib",
        robot: "SO-101 arm (TheRobotStudio)",
        file_name: "so101_new_calib.urdf",
        urdf: include_str!("../../../oracle/fixtures/corpus/so101_new_calib.urdf"),
        dof: 6,
        license: "Apache-2.0",
        source: "https://github.com/TheRobotStudio/SO-ARM100 \
                 (Simulation/SO101/so101_new_calib.urdf)",
        known_doctor_errors: &["A003"],
        doctor_note: "A003: collision meshes not embedded",
    },
    ZooEntry {
        name: "so100",
        robot: "SO-100 arm (TheRobotStudio)",
        file_name: "so100.urdf",
        urdf: include_str!("../../../oracle/fixtures/corpus/so100.urdf"),
        dof: 6,
        license: "Apache-2.0",
        source: "https://github.com/TheRobotStudio/SO-ARM100 \
                 (Simulation/SO100/so100.urdf)",
        known_doctor_errors: &["A003"],
        doctor_note: "A003: collision meshes not embedded",
    },
    ZooEntry {
        name: "gen3_lite",
        robot: "Kinova Gen3 lite",
        file_name: "gen3_lite.urdf",
        urdf: include_str!("../../../oracle/fixtures/corpus/gen3_lite.urdf"),
        dof: 10,
        license: "BSD-3-Clause",
        source: "https://github.com/Kinovarobotics/ros2_kortex \
                 (kortex_description/robots/gen3_lite.urdf)",
        known_doctor_errors: &["A001", "A003"],
        doctor_note: "A001: end_effector_link/dummy_link/tool_frame are \
                      massless tool frames upstream; A003: collision meshes \
                      not embedded",
    },
];

/// Exact-name lookup (no fuzzy matching — the registry is four names).
pub fn find(name: &str) -> Option<&'static ZooEntry> {
    REGISTRY.iter().find(|e| e.name == name)
}

/// Every fetchable name, registry order (for error messages and usage).
pub fn names() -> Vec<&'static str> {
    REGISTRY.iter().map(|e| e.name).collect()
}

/// Default output directory: `~/.cache/caliper/zoo/`.
/// ($HOME, with the Windows $USERPROFILE fallback — no `dirs` dependency.)
pub fn default_dir() -> anyhow::Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .filter(|h| !h.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("cannot determine the home directory; pass --dir"))?;
    Ok(home.join(".cache").join("caliper").join("zoo"))
}

/// What [`fetch`] did to the file on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FetchStatus {
    /// The file did not exist and was written.
    Fetched,
    /// The file already matched the embedded URDF byte-for-byte — untouched.
    AlreadyPresent,
    /// The file existed but DIFFERED from the embedded URDF (stale or edited
    /// cache copy) and was rewritten to the embedded truth.
    Refreshed,
}

impl FetchStatus {
    pub fn describe(self) -> &'static str {
        match self {
            FetchStatus::Fetched => "fetched (embedded copy written)",
            FetchStatus::AlreadyPresent => "already present — identical file left untouched",
            FetchStatus::Refreshed => {
                "refreshed — the cached copy differed from the embedded URDF and was rewritten"
            }
        }
    }
}

/// Materialize `entry` as `<dir>/<name>.urdf`, creating `dir` as needed.
/// Idempotent: an identical existing file is left alone. Returns the ABSOLUTE
/// path plus what happened.
pub fn fetch(entry: &ZooEntry, dir: &Path) -> anyhow::Result<(PathBuf, FetchStatus)> {
    std::fs::create_dir_all(dir)
        .map_err(|e| anyhow::anyhow!("cannot create `{}`: {e}", dir.display()))?;
    let path = dir.join(entry.file_name);
    let status = match std::fs::read_to_string(&path) {
        Ok(existing) if existing == entry.urdf => FetchStatus::AlreadyPresent,
        Ok(_) => {
            write_urdf(&path, entry)?;
            FetchStatus::Refreshed
        }
        Err(_) => {
            write_urdf(&path, entry)?;
            FetchStatus::Fetched
        }
    };
    let abs = std::path::absolute(&path)
        .map_err(|e| anyhow::anyhow!("cannot absolutize `{}`: {e}", path.display()))?;
    Ok((abs, status))
}

fn write_urdf(path: &Path, entry: &ZooEntry) -> anyhow::Result<()> {
    std::fs::write(path, entry.urdf)
        .map_err(|e| anyhow::anyhow!("failed to write `{}`: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliper_doctor::Severity;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Fresh scratch directory per call (unique via pid + counter),
    /// mirroring caliper-doctor's test helper.
    fn temp_dir(tag: &str) -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "caliper-zoo-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn registry_lists_all_four_corpus_robots_with_metadata() {
        let got: Vec<&str> = REGISTRY.iter().map(|e| e.name).collect();
        assert_eq!(got, ["panda", "so101_new_calib", "so100", "gen3_lite"]);
        for e in REGISTRY {
            assert!(e.urdf.contains("<robot"), "[{}] not a URDF", e.name);
            assert_eq!(e.file_name, format!("{}.urdf", e.name));
            assert!(e.dof > 0, "[{}] dof missing", e.name);
            assert!(!e.license.is_empty(), "[{}] license missing", e.name);
            assert!(
                e.source.starts_with("https://github.com/"),
                "[{}] source must attribute the upstream repo",
                e.name
            );
            assert!(
                !e.doctor_note.is_empty(),
                "[{}] doctor note missing",
                e.name
            );
            let mut sorted = e.known_doctor_errors.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted, e.known_doctor_errors,
                "[{}] known_doctor_errors must be sorted + deduped",
                e.name
            );
        }
    }

    #[test]
    fn find_is_exact_and_rejects_unknown_names() {
        assert!(find("panda").is_some());
        assert!(find("so100").is_some());
        assert!(find("Panda").is_none(), "lookup must be exact, not fuzzy");
        assert!(find("panda.urdf").is_none());
        assert!(find("ur5").is_none());
        assert!(find("").is_none());
    }

    #[test]
    fn fetched_urdfs_load_and_doctor_errors_match_the_documented_set() {
        // Hermetic mesh resolution: a dev machine's CALIPER_PACKAGE_PATH could
        // make the "not embedded" collision meshes resolve and silently shrink
        // the expected A003 set.
        // SAFETY: single mutation, and this is the only test in this binary
        // that touches mesh resolution (repo precedent: caliper-model tests).
        unsafe { std::env::remove_var("CALIPER_PACKAGE_PATH") };
        let dir = temp_dir("doctor");
        for e in REGISTRY {
            let (path, status) = fetch(e, &dir).unwrap();
            assert_eq!(status, FetchStatus::Fetched, "[{}]", e.name);
            assert!(
                path.is_absolute(),
                "[{}] fetch must return an absolute path",
                e.name
            );

            // 1) the fetched file LOADS via caliper, at the documented dof
            let robot = caliper::model::Robot::from_urdf(&path)
                .unwrap_or_else(|err| panic!("[{}] failed to load: {err}", e.name));
            assert_eq!(robot.ndof(), e.dof, "[{}] dof drifted", e.name);

            // 2) doctor Error codes == exactly the documented set (empty =
            //    clean); anything extra OR missing is a registry lie
            let rep = caliper_doctor::diagnose(&path).unwrap();
            let mut got: Vec<&str> = rep
                .findings
                .iter()
                .filter(|f| f.severity == Severity::Error)
                .map(|f| f.code.as_str())
                .collect();
            got.sort_unstable();
            got.dedup();
            assert_eq!(
                got,
                e.known_doctor_errors,
                "[{}] doctor Error codes drifted from the documented set:\n{}",
                e.name,
                rep.render_text()
            );
        }
    }

    #[test]
    fn fetch_is_idempotent_and_refreshes_stale_copies() {
        let dir = temp_dir("idem");
        let e = find("so100").unwrap();
        let (p1, s1) = fetch(e, &dir).unwrap();
        assert_eq!(s1, FetchStatus::Fetched);
        // second fetch: same path, file untouched
        let (p2, s2) = fetch(e, &dir).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(s2, FetchStatus::AlreadyPresent);
        assert_eq!(std::fs::read_to_string(&p2).unwrap(), e.urdf);
        // a stale/edited cached copy is rewritten to the embedded truth
        std::fs::write(&p1, "<robot name=\"stale\"/>").unwrap();
        let (p3, s3) = fetch(e, &dir).unwrap();
        assert_eq!(p1, p3);
        assert_eq!(s3, FetchStatus::Refreshed);
        assert_eq!(std::fs::read_to_string(&p3).unwrap(), e.urdf);
    }

    #[test]
    fn fetch_into_an_uncreatable_dir_errors() {
        // negative: the destination "directory" is a plain file
        let dir = temp_dir("neg");
        let blocker = dir.join("blocked");
        std::fs::write(&blocker, "not a directory").unwrap();
        let err = fetch(find("panda").unwrap(), &blocker).unwrap_err();
        assert!(err.to_string().contains("cannot create"), "{err}");
    }

    #[test]
    fn default_dir_is_under_the_user_cache() {
        let d = default_dir().unwrap();
        assert!(
            d.ends_with(Path::new(".cache/caliper/zoo")),
            "unexpected default dir: {}",
            d.display()
        );
    }
}
