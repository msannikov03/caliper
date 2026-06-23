use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NamedPose {
    pub name: String,
    pub q: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct PoseLibrary {
    poses: Vec<NamedPose>,
}

impl PoseLibrary {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn upsert(&mut self, name: String, q: Vec<f64>) {
        if let Some(p) = self.poses.iter_mut().find(|p| p.name == name) {
            p.q = q;
        } else {
            self.poses.push(NamedPose { name, q });
        }
    }
    pub fn get(&self, name: &str) -> Option<&NamedPose> {
        self.poses.iter().find(|p| p.name == name)
    }
    pub fn remove(&mut self, name: &str) {
        self.poses.retain(|p| p.name != name);
    }
    pub fn list(&self) -> &[NamedPose] {
        &self.poses
    }
    pub fn clear(&mut self) {
        self.poses.clear();
    }
}
