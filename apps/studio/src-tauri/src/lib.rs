//! Caliper Studio — Tauri backend. Wires the `caliper` engine to the UI.
use serde::Serialize;

#[derive(Serialize)]
struct RobotInfo {
    name: String,
    dof: usize,
    joint_names: Vec<String>,
}

/// Engine version (proves the UI is talking to the real Rust core).
#[tauri::command]
fn engine_version() -> String {
    caliper::VERSION.to_string()
}

/// Load a robot from a URDF path and return its structure.
///
/// Security: only `.urdf`/`.xacro` files are accepted (so this is not a general
/// arbitrary-file-read primitive), and errors are mapped to a generic message
/// rather than leaking filesystem detail to the webview. When wired into the UI,
/// file selection should go through the native dialog plugin, not a raw path.
#[tauri::command]
fn load_robot(path: String) -> Result<RobotInfo, String> {
    let p = std::path::Path::new(&path);
    let ext_ok = p
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("urdf") || e.eq_ignore_ascii_case("xacro"));
    if !ext_ok {
        return Err("only .urdf or .xacro files are supported".into());
    }
    let robot = caliper::model::Robot::from_urdf(p)
        .map_err(|_| "failed to load robot from the given URDF".to_string())?;
    Ok(RobotInfo {
        name: robot.name.clone(),
        dof: robot.ndof(),
        joint_names: robot.joint_names.clone(),
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![engine_version, load_robot])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
