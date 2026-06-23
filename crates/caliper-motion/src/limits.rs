use crate::MotionError;
use caliper_model::Model;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotionLimits {
    pub vmax: Vec<f64>,
    pub amax: Vec<f64>,
    pub jmax: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct JointOverride {
    pub joint: usize,
    pub vmax: Option<f64>,
    pub amax: Option<f64>,
    pub jmax: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct MotionLimitsConfig {
    /// Fallback vmax when the URDF velocity limit is None (rad/s | m/s).
    pub default_vel: f64,
    /// amax = accel_ratio * vmax  (1/s). Engineering choice, NOT physics.
    pub accel_ratio: f64,
    /// jmax = jerk_ratio * amax   (1/s).
    pub jerk_ratio: f64,
    /// Global de-rate on the URDF velocity (1.0 = full speed; 0.5 = slow mode).
    pub vel_scale: f64,
    pub overrides: Vec<JointOverride>,
}

impl Default for MotionLimitsConfig {
    fn default() -> Self {
        Self {
            default_vel: 1.0,
            accel_ratio: 5.0,
            jerk_ratio: 10.0,
            vel_scale: 1.0,
            overrides: vec![],
        }
    }
}

impl MotionLimits {
    pub fn ndof(&self) -> usize {
        self.vmax.len()
    }

    /// Build per-dof (vmax,amax,jmax). vmax from URDF velocity (de-rated), accel
    /// and jerk synthesized from the config ratios, then per-joint overrides.
    pub fn from_model(model: &Model, cfg: &MotionLimitsConfig) -> Result<Self, MotionError> {
        let n = model.ndof;
        let mut vmax = vec![0.0; n];
        let mut amax = vec![0.0; n];
        let mut jmax = vec![0.0; n];
        for i in 0..n {
            let v = cfg.vel_scale * model.vel_limit[i].unwrap_or(cfg.default_vel);
            vmax[i] = v;
            amax[i] = cfg.accel_ratio * v;
            jmax[i] = cfg.jerk_ratio * amax[i];
        }
        for o in &cfg.overrides {
            if o.joint >= n {
                return Err(MotionError::DimMismatch);
            }
            if let Some(v) = o.vmax {
                vmax[o.joint] = v;
            }
            if let Some(a) = o.amax {
                amax[o.joint] = a;
            }
            if let Some(j) = o.jmax {
                jmax[o.joint] = j;
            }
        }
        for i in 0..n {
            // reject non-positive OR non-finite (one combined test avoids the
            // neg_cmp_op_on_partial_ord lint and still catches NaN).
            if !(vmax[i] > 0.0 && amax[i] > 0.0 && jmax[i] > 0.0) {
                return Err(MotionError::BadLimit(i));
            }
        }
        Ok(Self { vmax, amax, jmax })
    }
}
