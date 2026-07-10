//! `MujocoSim` — a thin, deterministic, headless seam over `mujoco-rs`.
//!
//! Design rules:
//! - Joint addressing is resolved by NAME through `mj_name2id` +
//!   `jnt_qposadr`/`jnt_dofadr` at construction — never by assuming MuJoCo's
//!   qpos order matches caliper's joint order (it does for our generated MJCF,
//!   but the map makes that a fact, not an assumption).
//! - Only fixed-base trees of 1-dof joints (hinge/slide) are mapped; anything
//!   else fails loudly at load.
//! - Determinism: MuJoCo is single-threaded per `mjData` (we never opt into
//!   `mjThreadPool`) and bit-deterministic for the same binary + identical
//!   integration state; `reset()` restores the FULL state (time, warmstart
//!   included), so two identical command sequences produce bit-identical
//!   trajectories. Cross-version bitwise equality is not guaranteed — MuJoCo
//!   is pinned at 3.9.0 via the exact `mujoco-rs` pin.

use crate::MujocoError;
use crate::mjcf::{self, MjcfOptions};
use caliper_model::Model;
use mujoco_rs::wrappers::mj_data::MjData;
use mujoco_rs::wrappers::mj_model::{MjModel, MjtJoint, MjtObj};
use std::sync::Arc;

/// One detected contact, in world coordinates.
#[derive(Clone, Debug)]
pub struct Contact {
    /// Geom names (generated MJCF names geoms `col{i}_{link}`; the optional
    /// ground plane is `caliper_ground`). Unnamed geoms fall back to `geom{id}`.
    pub geom1: String,
    pub geom2: String,
    /// Contact point (midpoint between surfaces).
    pub pos: [f64; 3],
    /// Contact normal, pointing from geom1 toward geom2.
    pub normal: [f64; 3],
    /// Penetration depth (= −MuJoCo `dist`; positive when penetrating).
    pub depth: f64,
}

/// A loaded MuJoCo model + data with a flat `q`/`qd` interface in a fixed
/// joint order (caliper's order when built from a caliper [`Model`], MuJoCo's
/// document order when built [`from_mjcf`](MujocoSim::from_mjcf)).
pub struct MujocoSim {
    data: MjData<Arc<MjModel>>,
    /// Model timestep `h`; [`step`](Self::step) takes integer multiples.
    h: f64,
    qpos_adr: Vec<usize>,
    dof_adr: Vec<usize>,
    joint_names: Vec<String>,
    nu: usize,
    skipped_hull_colliders: usize,
    /// `(prop name, MuJoCo body id)` for every free prop passed at build,
    /// in [`MjcfOptions::props`] order (empty for raw-MJCF loads).
    props: Vec<(String, usize)>,
}

impl MujocoSim {
    /// Build from a caliper model with default [`MjcfOptions`].
    pub fn from_caliper_model(m: &Model) -> Result<Self, MujocoError> {
        Self::from_caliper_model_with(m, &MjcfOptions::default())
    }

    /// Build from a caliper model: generate minimal MJCF, load it, and map
    /// caliper joint order → MuJoCo addresses by name.
    pub fn from_caliper_model_with(m: &Model, opt: &MjcfOptions) -> Result<Self, MujocoError> {
        let doc = mjcf::mjcf_from_model(m, opt)?;
        let mj =
            MjModel::from_xml_string(&doc.xml).map_err(|e| MujocoError::Load(e.to_string()))?;
        let sim = Self::from_parts(
            mj,
            m.joint_names.clone(),
            doc.skipped_hull_colliders,
            &doc.prop_bodies,
        )?;
        // Our MJCF contains exactly the caliper joints — anything else is a
        // generator bug, not a user error.
        debug_assert_eq!(sim.qpos_adr.len(), m.ndof);
        Ok(sim)
    }

    /// Load a raw MJCF string. Every joint in the document must be hinge or
    /// slide (fixed-base articulated models only); the flat `q` order is
    /// MuJoCo's joint order.
    pub fn from_mjcf(xml: &str) -> Result<Self, MujocoError> {
        let mj = MjModel::from_xml_string(xml).map_err(|e| MujocoError::Load(e.to_string()))?;
        let names: Vec<String> = (0..mj.njnt() as usize)
            .map(|id| {
                mj.id_to_name(MjtObj::mjOBJ_JOINT, id)
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("joint{id}"))
            })
            .collect();
        Self::from_parts(mj, names, 0, &[])
    }

    fn from_parts(
        mj: MjModel,
        joint_names: Vec<String>,
        skipped_hull_colliders: usize,
        prop_bodies: &[(String, String)],
    ) -> Result<Self, MujocoError> {
        let mut qpos_adr = Vec::with_capacity(joint_names.len());
        let mut dof_adr = Vec::with_capacity(joint_names.len());
        {
            let types = mj.jnt_type();
            let qadr = mj.jnt_qposadr();
            let dadr = mj.jnt_dofadr();
            for name in &joint_names {
                let id = mj
                    .name_to_id(MjtObj::mjOBJ_JOINT, name)
                    .ok_or_else(|| MujocoError::MissingJoint(name.clone()))?;
                match types[id] {
                    MjtJoint::mjJNT_HINGE | MjtJoint::mjJNT_SLIDE => {}
                    _ => return Err(MujocoError::UnsupportedJoint(name.clone())),
                }
                qpos_adr.push(qadr[id] as usize);
                dof_adr.push(dadr[id] as usize);
            }
        }
        let h = mj.opt().timestep;
        if !(h.is_finite() && h > 0.0) {
            return Err(MujocoError::Load(format!("model timestep {h} invalid")));
        }
        let nu = mj.nu() as usize;
        // Resolve prop body ids by NAME while we still own the model — our
        // MJCF emitted these bodies, so a miss is a generator bug surfaced loud.
        let mut props = Vec::with_capacity(prop_bodies.len());
        for (pname, bname) in prop_bodies {
            let id = mj
                .name_to_id(MjtObj::mjOBJ_BODY, bname)
                .ok_or_else(|| MujocoError::MissingBody(bname.clone()))?;
            props.push((pname.clone(), id));
        }
        let mut data = MjData::new(Arc::new(mj));
        data.forward(); // consistent derived quantities before first read
        Ok(Self {
            data,
            h,
            qpos_adr,
            dof_adr,
            joint_names,
            nu,
            skipped_hull_colliders,
            props,
        })
    }

    // ---- introspection ----
    pub fn ndof(&self) -> usize {
        self.qpos_adr.len()
    }
    pub fn joint_names(&self) -> &[String] {
        &self.joint_names
    }
    /// The MuJoCo integration timestep `h` baked into the model.
    pub fn timestep(&self) -> f64 {
        self.h
    }
    /// Number of MuJoCo actuators (0 for `Actuation::TorqueDirect`).
    pub fn nu(&self) -> usize {
        self.nu
    }
    /// Hull/mesh colliders the MJCF generator had to skip (0 for raw MJCF).
    /// Non-zero = LESS collision coverage than caliper-collision has.
    pub fn skipped_hull_colliders(&self) -> usize {
        self.skipped_hull_colliders
    }
    pub fn time(&self) -> f64 {
        self.data.time()
    }

    // ---- state ----
    pub fn qpos(&self) -> Vec<f64> {
        let qp = self.data.qpos();
        self.qpos_adr.iter().map(|&a| qp[a]).collect()
    }
    pub fn qvel(&self) -> Vec<f64> {
        let qv = self.data.qvel();
        self.dof_adr.iter().map(|&a| qv[a]).collect()
    }

    /// Seed `(q, qd)` WITHOUT advancing time, then recompute derived
    /// quantities (`mj_forward`).
    pub fn set_state(&mut self, q: &[f64], qd: &[f64]) -> Result<(), MujocoError> {
        self.check(q, "q")?;
        self.check(qd, "qd")?;
        {
            let qp = self.data.qpos_mut();
            for (i, &a) in self.qpos_adr.iter().enumerate() {
                qp[a] = q[i];
            }
        }
        {
            let qv = self.data.qvel_mut();
            for (i, &a) in self.dof_adr.iter().enumerate() {
                qv[a] = qd[i];
            }
        }
        self.data.forward();
        Ok(())
    }

    /// Full deterministic reset: `mj_resetData` (zeros time, velocities,
    /// controls, applied forces AND the solver warmstart), then seed `q0` and
    /// recompute. Two runs from the same `reset` + identical commands are
    /// bitwise identical (same binary + libmujoco).
    pub fn reset(&mut self, q0: &[f64]) -> Result<(), MujocoError> {
        self.check(q0, "q0")?;
        self.data.reset();
        {
            let qp = self.data.qpos_mut();
            for (i, &a) in self.qpos_adr.iter().enumerate() {
                qp[a] = q0[i];
            }
        }
        self.data.forward();
        Ok(())
    }

    // ---- commands ----
    /// Write generalized joint torques into `qfrc_applied` (all mapped dofs,
    /// every call — stale values never linger). Persists across steps until
    /// overwritten, exactly like `Simulator::set_torque`.
    pub fn set_joint_torques(&mut self, tau: &[f64]) -> Result<(), MujocoError> {
        self.check(tau, "tau")?;
        let qf = self.data.qfrc_applied_mut();
        for (i, &a) in self.dof_adr.iter().enumerate() {
            qf[a] = tau[i];
        }
        Ok(())
    }

    /// Write the raw actuator vector (`ctrl`); length must equal [`nu`](Self::nu).
    /// For `Actuation::PositionServo` models this is one target position per
    /// joint, in caliper joint order (the generator emits actuators in that
    /// order).
    pub fn set_ctrl(&mut self, ctrl: &[f64]) -> Result<(), MujocoError> {
        if ctrl.len() != self.nu {
            return Err(MujocoError::Dim {
                expected: self.nu,
                got: ctrl.len(),
            });
        }
        if !ctrl.iter().all(|x| x.is_finite()) {
            return Err(MujocoError::NonFinite { what: "ctrl" });
        }
        self.data.ctrl_mut().copy_from_slice(ctrl);
        Ok(())
    }

    // ---- integration ----
    /// Advance by `dt`, which must be a positive integer multiple of the model
    /// timestep `h` (within 1e-9 relative) — no silent remainder, no hidden
    /// sub-step drift between two sims stepped with the same `dt`.
    pub fn step(&mut self, dt: f64) -> Result<(), MujocoError> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(MujocoError::BadDt { dt, h: self.h });
        }
        let k = (dt / self.h).round();
        if k < 1.0 || (k * self.h - dt).abs() > 1e-9 * dt.max(1.0) {
            return Err(MujocoError::BadDt { dt, h: self.h });
        }
        for _ in 0..k as u64 {
            self.data.step();
        }
        Ok(())
    }

    /// One raw `mj_step` of the model timestep.
    pub fn step_once(&mut self) {
        self.data.step();
    }

    /// Recompute derived quantities (incl. the contact list) for the CURRENT
    /// state without advancing time.
    pub fn forward(&mut self) {
        self.data.forward();
    }

    // ---- contacts ----
    pub fn ncon(&self) -> usize {
        self.data.ncon() as usize
    }

    /// The contact list from the last `step`/`forward`, with geom ids resolved
    /// to names. `frame[0..3]` is the MuJoCo contact normal (geom1 → geom2);
    /// `depth = −dist` (positive = penetrating).
    pub fn contacts(&self) -> Vec<Contact> {
        let model = self.data.model();
        let name = |id: i32| -> String {
            model
                .id_to_name(MjtObj::mjOBJ_GEOM, id as usize)
                .map(str::to_string)
                .unwrap_or_else(|| format!("geom{id}"))
        };
        self.data
            .contact()
            .iter()
            .map(|c| Contact {
                geom1: name(c.geom1),
                geom2: name(c.geom2),
                pos: c.pos,
                normal: [c.frame[0], c.frame[1], c.frame[2]],
                depth: -c.dist,
            })
            .collect()
    }

    /// Contact wrench `[normal force, 2×friction, 3×torque]` in the contact
    /// frame for contact index `i` (`[0.0; 6]` when out of range).
    pub fn contact_force(&self, i: usize) -> [f64; 6] {
        self.data.contact_force(i)
    }

    // ---- bodies & props ----
    /// World pose of a named MJCF body from the last `step`/`forward`:
    /// `(xpos, xquat)` with the quaternion in MuJoCo order `[w, x, y, z]`.
    /// Robot bodies are `b_{joint}`, props `prop_{name}` (see [`mjcf`]).
    pub fn body_pose(&self, name: &str) -> Result<([f64; 3], [f64; 4]), MujocoError> {
        let id = self
            .data
            .model()
            .name_to_id(MjtObj::mjOBJ_BODY, name)
            .ok_or_else(|| MujocoError::MissingBody(name.to_string()))?;
        Ok((self.data.xpos()[id], self.data.xquat()[id]))
    }

    /// Prop names passed at build, in [`MjcfOptions::props`] order.
    pub fn prop_names(&self) -> Vec<&str> {
        self.props.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// `(name, world pos, world quat [w,x,y,z])` for every prop, in build
    /// order, from the last `step`/`forward`. Empty for raw-MJCF loads.
    pub fn prop_poses(&self) -> Vec<(String, [f64; 3], [f64; 4])> {
        let xp = self.data.xpos();
        let xq = self.data.xquat();
        self.props
            .iter()
            .map(|(n, id)| (n.clone(), xp[*id], xq[*id]))
            .collect()
    }

    // ---- escape hatches ----
    /// Raw `mujoco-rs` data handle (full mjData surface, incl. `ffi()`).
    pub fn mj_data(&self) -> &MjData<Arc<MjModel>> {
        &self.data
    }
    pub fn mj_data_mut(&mut self) -> &mut MjData<Arc<MjModel>> {
        &mut self.data
    }

    fn check(&self, xs: &[f64], what: &'static str) -> Result<(), MujocoError> {
        if xs.len() != self.ndof() {
            return Err(MujocoError::Dim {
                expected: self.ndof(),
                got: xs.len(),
            });
        }
        if !xs.iter().all(|x| x.is_finite()) {
            return Err(MujocoError::NonFinite { what });
        }
        Ok(())
    }
}
