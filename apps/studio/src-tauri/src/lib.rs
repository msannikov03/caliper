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
#[tauri::command]
fn load_robot(path: String) -> Result<RobotInfo, String> {
    let robot =
        caliper::model::Robot::from_urdf(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
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
