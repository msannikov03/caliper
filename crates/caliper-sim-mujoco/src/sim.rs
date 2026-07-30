//! `MujocoSim` — a thin, deterministic, headless seam over `mujoco-rs`.
//!
//! Design rules:
//! - Joint addressing is resolved at construction through `jnt_qposadr` /
//!   `jnt_dofadr` — never by assuming MuJoCo's qpos order matches caliper's
//!   joint order (it does for our generated MJCF, but the map makes that a
//!   fact, not an assumption). Caliper builds resolve joint ids by the
//!   SANITIZED name the generator emitted ([`mjcf::mujoco_name`] — MuJoCo
//!   registers `left_arm` for `<joint name="left arm">`); raw-MJCF loads take
//!   every joint by document id, so unnamed joints are fine.
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
use mujoco_rs::mujoco_c::mjNEQDATA;
use mujoco_rs::wrappers::mj_data::MjData;
use mujoco_rs::wrappers::mj_model::{MjModel, MjtJoint, MjtObj};
use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use std::sync::Arc;

/// Numbers per equality constraint in `mjModel::eq_data`. For a WELD they are
/// laid out `[anchor(3), relpose position(3), relpose quaternion w,x,y,z(4),
/// torquescale(1)]` — verified against MuJoCo 3.9.0 by round-tripping an MJCF
/// weld with distinct values through the compiler.
const NEQDATA: usize = mjNEQDATA as usize;

/// Offset of a weld's relative-pose block within its `eq_data` row.
const WELD_RELPOSE: usize = 3;

/// MuJoCo world quaternion (`[w, x, y, z]`) → nalgebra unit quaternion.
fn unit_quat(q: [f64; 4]) -> UnitQuaternion<f64> {
    UnitQuaternion::new_normalize(Quaternion::new(q[0], q[1], q[2], q[3]))
}

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
    /// `(prop name, MuJoCo equality-constraint id)` for every prop's grasp
    /// weld, in the same order. Empty unless the model was built with
    /// [`MjcfOptions::attach_link`].
    welds: Vec<(String, usize)>,
}

impl MujocoSim {
    /// Build from a caliper model with default [`MjcfOptions`].
    pub fn from_caliper_model(m: &Model) -> Result<Self, MujocoError> {
        Self::from_caliper_model_with(m, &MjcfOptions::default())
    }

    /// Build from a caliper model: generate minimal MJCF, load it, and map
    /// caliper joint order → MuJoCo addresses by the SANITIZED name the
    /// generator emitted ([`mjcf::mujoco_name`]); the caliper spelling stays
    /// the user-facing one in [`joint_names`](Self::joint_names) and errors.
    pub fn from_caliper_model_with(m: &Model, opt: &MjcfOptions) -> Result<Self, MujocoError> {
        let doc = mjcf::mjcf_from_model(m, opt)?;
        let mj =
            MjModel::from_xml_string(&doc.xml).map_err(|e| MujocoError::Load(e.to_string()))?;
        // The generator SANITIZES identifiers, so MuJoCo registered
        // `mujoco_name(..)` of each caliper joint name (`left arm` →
        // `left_arm`) — resolve that spelling, but keep (and report errors
        // with) the caliper spelling the user knows.
        let mut joint_ids = Vec::with_capacity(m.joint_names.len());
        for name in &m.joint_names {
            let id = mj
                .name_to_id(MjtObj::mjOBJ_JOINT, &mjcf::mujoco_name(name))
                .ok_or_else(|| MujocoError::MissingJoint(name.clone()))?;
            joint_ids.push(id);
        }
        let sim = Self::from_parts(
            mj,
            m.joint_names.clone(),
            joint_ids,
            doc.skipped_hull_colliders,
            &doc.prop_bodies,
            &doc.grasp_welds,
        )?;
        // Our MJCF contains exactly the caliper joints — anything else is a
        // generator bug, not a user error.
        debug_assert_eq!(sim.qpos_adr.len(), m.ndof);
        Ok(sim)
    }

    /// Load a raw MJCF string. Every joint in the document must be hinge or
    /// slide (fixed-base articulated models only); the flat `q` order is
    /// MuJoCo's joint order. Joints are addressed by document ID, so unnamed
    /// joints (legal MJCF) load fine — they show up in
    /// [`joint_names`](Self::joint_names) as the placeholder `joint{id}`.
    pub fn from_mjcf(xml: &str) -> Result<Self, MujocoError> {
        let mj = MjModel::from_xml_string(xml).map_err(|e| MujocoError::Load(e.to_string()))?;
        let (names, ids): (Vec<String>, Vec<usize>) = (0..mj.njnt() as usize)
            .map(|id| {
                let name = mj
                    .id_to_name(MjtObj::mjOBJ_JOINT, id)
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("joint{id}"));
                (name, id)
            })
            .unzip();
        Self::from_parts(mj, names, ids, 0, &[], &[])
    }

    /// `joint_names[i]` is the user-facing spelling for MuJoCo joint
    /// `joint_ids[i]` — the caller has already RESOLVED the ids (by sanitized
    /// name for caliper builds, by document order for raw MJCF), so no name
    /// lookup happens here.
    fn from_parts(
        mj: MjModel,
        joint_names: Vec<String>,
        joint_ids: Vec<usize>,
        skipped_hull_colliders: usize,
        prop_bodies: &[(String, String)],
        grasp_welds: &[(String, String)],
    ) -> Result<Self, MujocoError> {
        debug_assert_eq!(joint_names.len(), joint_ids.len());
        let mut qpos_adr = Vec::with_capacity(joint_names.len());
        let mut dof_adr = Vec::with_capacity(joint_names.len());
        {
            let types = mj.jnt_type();
            let qadr = mj.jnt_qposadr();
            let dadr = mj.jnt_dofadr();
            for (name, &id) in joint_names.iter().zip(&joint_ids) {
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
        // MJCF emitted these bodies, so a miss is a generator bug surfaced
        // loud. The doc spelling in `prop_bodies` is XML-ESCAPED; MuJoCo
        // registers the unescaped attribute value, i.e. `mujoco_name(..)` of
        // the prop name.
        let mut props = Vec::with_capacity(prop_bodies.len());
        for (pname, bname) in prop_bodies {
            let registered = format!("prop_{}", mjcf::mujoco_name(pname));
            let id = mj
                .name_to_id(MjtObj::mjOBJ_BODY, &registered)
                .ok_or_else(|| MujocoError::MissingBody(bname.clone()))?;
            props.push((pname.clone(), id));
        }
        // Same story for the grasp welds: the generator emitted them, so a
        // miss is a generator bug. MuJoCo registers `mujoco_name(..)` of the
        // weld name the document carries XML-ESCAPED.
        let mut welds = Vec::with_capacity(grasp_welds.len());
        for (pname, wname) in grasp_welds {
            let registered = format!("grasp_{}", mjcf::mujoco_name(pname));
            let id = mj
                .name_to_id(MjtObj::mjOBJ_EQUALITY, &registered)
                .ok_or_else(|| MujocoError::MissingWeld(wname.clone()))?;
            welds.push((pname.clone(), id));
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
            welds,
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
    ///
    /// Grasp welds are released: `mj_resetData` restores `d->eq_active` from
    /// the model's `eq_active0` (all welds are authored inactive), and this
    /// call deactivates them explicitly too, so the guarantee does not rest on
    /// a MuJoCo implementation detail. A captured relative pose is left in the
    /// model — it is dead data while the weld is inactive, and every
    /// activation overwrites it.
    pub fn reset(&mut self, q0: &[f64]) -> Result<(), MujocoError> {
        self.check(q0, "q0")?;
        self.data.reset();
        self.deactivate_all_welds();
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

    /// `(name, world LINEAR velocity)` for every prop, in build order, from the
    /// last `step`/`forward`. Empty for raw-MJCF loads.
    ///
    /// A prop rides a `<freejoint>`, and MuJoCo defines that joint's first
    /// three `qvel` entries as the body's linear velocity in WORLD coordinates
    /// — the exact three numbers `caliper_learn.success.state_from_mujoco`
    /// reads, so a "has it come to rest" check answers the same in both faces.
    /// A prop body with no free joint is a generator bug and reported as an
    /// error rather than as a fabricated zero velocity (which would read as
    /// "settled").
    pub fn prop_velocities(&self) -> Result<Vec<(String, [f64; 3])>, MujocoError> {
        let model = self.data.model();
        let types = model.jnt_type();
        let bodies = model.jnt_bodyid();
        let dof_adr = model.jnt_dofadr();
        let qv = self.data.qvel();
        self.props
            .iter()
            .map(|(name, bid)| {
                let jid = (0..types.len())
                    .find(|&j| types[j] == MjtJoint::mjJNT_FREE && bodies[j] as usize == *bid)
                    .ok_or_else(|| {
                        MujocoError::Backend(format!(
                            "prop `{name}` has no free joint, so its velocity cannot be read"
                        ))
                    })?;
                let a = dof_adr[jid] as usize;
                Ok((name.clone(), [qv[a], qv[a + 1], qv[a + 2]]))
            })
            .collect()
    }

    // ---- grasp welds ----
    //
    // An HONEST FAKE, and labeled as one: a real parallel gripper grasps by
    // friction between two finger geoms, which needs finger dynamics no
    // teleop-collection rig actually simulates. Instead each prop carries one
    // `<weld>` equality constraint to the gripper's link, emitted INACTIVE by
    // the generator ([`MjcfOptions::attach_link`]); "grasping" activates it and
    // "releasing" deactivates it. While active the prop is rigidly (if
    // compliantly — a MuJoCo weld is a soft constraint) carried by the arm;
    // once released it is a free body again, with whatever velocity it had.

    /// Prop names that carry a grasp weld, in build order (empty unless the
    /// model was built with [`MjcfOptions::attach_link`]).
    pub fn weld_props(&self) -> Vec<&str> {
        self.welds.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Is this prop's grasp weld currently holding it?
    pub fn weld_active(&self, prop: &str) -> Result<bool, MujocoError> {
        let eq = self.weld_id(prop)?;
        Ok(self.data.eq_active()[eq])
    }

    /// Attach (`active = true`) or release a prop's grasp weld.
    ///
    /// `capture_relpose` is the whole trick: a weld authored in the document
    /// holds the two bodies at the relative pose the MODEL was compiled with,
    /// so activating it mid-run would SNAP the prop to wherever that was.
    /// With `capture_relpose` on, the constraint's target relative pose is
    /// first overwritten with the pose the prop has RIGHT NOW relative to the
    /// attach body, so activation is continuous — the prop stays exactly where
    /// it is and simply stops moving relative to the gripper. Pass `false`
    /// only to deliberately re-use the authored pose.
    ///
    /// Ignored on release (there is nothing to capture).
    pub fn set_weld_active(
        &mut self,
        prop: &str,
        active: bool,
        capture_relpose: bool,
    ) -> Result<(), MujocoError> {
        let eq = self.weld_id(prop)?;
        if active && capture_relpose {
            self.capture_weld_relpose(eq);
        }
        self.data.eq_active_mut()[eq] = active;
        Ok(())
    }

    /// Release every grasp weld. Cheap and idempotent — the "drop everything"
    /// move behind a reset or a session teardown.
    pub fn deactivate_all_welds(&mut self) {
        let ids: Vec<usize> = self.welds.iter().map(|(_, id)| *id).collect();
        let active = self.data.eq_active_mut();
        for id in ids {
            active[id] = false;
        }
    }

    /// Props currently in contact with a ROBOT geom — not with the ground
    /// plane, not with each other — in build order. This is the "is the
    /// gripper actually touching it" test a grasp heuristic gates on.
    ///
    /// Robot vs. world vs. prop is decided by BODY, not by geom name: a
    /// contact's geoms resolve to their bodies, prop bodies are the ones the
    /// generator emitted for [`MjcfOptions::props`], world geoms (the ground
    /// plane and any `extra_worldbody_xml`) belong to body 0, and everything
    /// else is the robot.
    pub fn props_touching_robot(&self) -> Vec<&str> {
        let gbody = self.data.model().geom_bodyid();
        let contacts = self.data.contact();
        let prop_ids: Vec<usize> = self.props.iter().map(|(_, id)| *id).collect();
        let body_of = |g: i32| -> Option<usize> {
            (g >= 0 && (g as usize) < gbody.len()).then(|| gbody[g as usize] as usize)
        };
        self.props
            .iter()
            .filter(|(_, id)| {
                contacts.iter().any(|c| {
                    let (Some(b1), Some(b2)) = (body_of(c.geom1), body_of(c.geom2)) else {
                        return false;
                    };
                    let other = if b1 == *id {
                        b2
                    } else if b2 == *id {
                        b1
                    } else {
                        return false;
                    };
                    other != 0 && !prop_ids.contains(&other)
                })
            })
            .map(|(n, _)| n.as_str())
            .collect()
    }

    fn weld_id(&self, prop: &str) -> Result<usize, MujocoError> {
        self.welds
            .iter()
            .find(|(n, _)| n == prop)
            .map(|(_, id)| *id)
            .ok_or_else(|| MujocoError::MissingWeld(prop.to_string()))
    }

    /// Freeze the CURRENT relative pose of the weld's body2 (the prop) with
    /// respect to body1 (the attach link) into the constraint:
    ///
    /// ```text
    /// relpose.pos  = R1ᵀ · (x2 − x1)
    /// relpose.quat = q1⁻¹ ⊗ q2
    /// ```
    ///
    /// `mjModel::eq_data` is the only place MuJoCo keeps a weld's target pose
    /// (there is no per-`mjData` copy), so this writes the model — which is
    /// exactly what the runtime-attach recipe does, and is why every raw
    /// pointer in this crate is confined to this file.
    fn capture_weld_relpose(&mut self, eq: usize) {
        let model = self.data.model();
        let b1 = model.eq_obj1id()[eq] as usize;
        let b2 = model.eq_obj2id()[eq] as usize;
        let (x1, x2) = (self.data.xpos()[b1], self.data.xpos()[b2]);
        let (q1, q2) = (self.data.xquat()[b1], self.data.xquat()[b2]);
        let q1 = unit_quat(q1);
        let rel_p = q1.inverse_transform_vector(&(Vector3::from(x2) - Vector3::from(x1)));
        let rel_q = q1.inverse() * unit_quat(q2);
        let row = [
            rel_p.x, rel_p.y, rel_p.z, rel_q.w, rel_q.i, rel_q.j, rel_q.k,
        ];
        // SAFETY: `eq_data` is a `*mut mjtNum` field of the C-owned mjModel —
        // its pointee is a separate allocation from the `&mjModel` we read it
        // out of, so writing through it aliases nothing Rust holds. `eq` came
        // from `weld_id`, i.e. a name MuJoCo resolved in THIS model, so it is
        // `< neq` and the row `[eq·NEQDATA, (eq+1)·NEQDATA)` is in bounds.
        // `mujoco-rs` exposes `eq_data` read-only (no `eq_data_mut`), and the
        // model is shared through an `Arc` by `MjData`, so this pointer is the
        // only write path; MuJoCo reads it on the next `mj_forward`/`mj_step`.
        unsafe {
            let p = model.ffi().eq_data.add(NEQDATA * eq + WELD_RELPOSE);
            for (k, v) in row.iter().enumerate() {
                *p.add(k) = *v;
            }
        }
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
