//! Caliper Phase-8 **dataflow graph executor**.
//!
//! A deterministic, serde-serializable graph IR ([`GraphDoc`]) that composes the
//! existing engine ops (FK / IK / motion / planning / dynamics / control /
//! collision) into a runnable pipeline. The three faces — the Studio Tauri
//! backend, the CLI, and the PyO3 bindings — all (de)serialize a `GraphDoc`
//! (the `.caliper-graph.json` schema), then call [`validate`] and [`run`].
//!
//! NO new math lives here: every COMPUTE node dispatches to an EXISTING engine
//! free fn / type. The crate is pure-Rust and lean (serde + the engine crates +
//! nalgebra; no `rand`/`tokio`). Determinism: `PlanRrt` is seeded; control /
//! dynamics rollouts are tick-driven and clock-free.
//!
//! ## Shape
//! - [`ir`] — the persisted schema: [`PortType`], [`NodeKind`], [`Node`],
//!   [`Edge`], [`GraphDoc`], [`PortValue`], [`ClipData`], [`ReportData`].
//! - [`validate`] (module) — [`validate`] returns [`Diagnostics`] (per-node /
//!   per-edge errors + a topo order or a cycle report).
//! - [`exec`] — [`run`] returns a [`GraphResult`] or a [`GraphError`].

pub mod exec;
pub mod ir;
pub mod validate;

pub use exec::{
    CLIP_DT, CONTROL_DT, GraphError, GraphResult, ScopeSeries, bake_trajectory, pose_value, run,
};
pub use ir::{
    ClipData, Edge, GraphDoc, GraphMeta, InPort, Node, NodeKind, OutPort, PortRef, PortType,
    PortValue, ReportData, pose_to_se3, se3_to_pose,
};
pub use validate::{Diagnostics, EdgeDiag, NodeDiag, Signal, parse_signal, validate};

#[cfg(test)]
mod tests {
    use super::*;
    use caliper_kinematics::fk_tip;
    use caliper_model::Robot;
    use caliper_motion::{CartesianMoveOpts, MotionLimits, MotionLimitsConfig, move_j, move_l};
    use std::path::Path;

    fn robot(name: &str) -> Robot {
        Robot::from_urdf(Path::new(&format!(
            "{}/../../oracle/fixtures/robots/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        )))
        .unwrap()
    }

    fn node(id: &str, kind: NodeKind) -> Node {
        Node {
            id: id.into(),
            kind,
        }
    }
    fn edge(from: &str, fp: &str, to: &str, tp: &str) -> Edge {
        Edge {
            from: from.into(),
            from_port: PortRef::Name(fp.into()),
            to: to.into(),
            to_port: PortRef::Name(tp.into()),
        }
    }

    fn limits(r: &Robot) -> MotionLimits {
        MotionLimits::from_model(&r.model, &MotionLimitsConfig::default()).unwrap()
    }

    fn clips_eq(a: &ClipData, b: &ClipData, eps: f64) -> bool {
        if a.times.len() != b.times.len() || a.qs.len() != b.qs.len() {
            return false;
        }
        for (ra, rb) in a.qs.iter().zip(&b.qs) {
            if ra.len() != rb.len() || ra.iter().zip(rb).any(|(x, y)| (x - y).abs() > eps) {
                return false;
            }
        }
        a.times
            .iter()
            .zip(&b.times)
            .all(|(x, y)| (x - y).abs() <= eps)
    }

    #[test]
    fn validation_rejects_cycle() {
        let r = robot("showcase6.urdf");
        // two IK nodes feeding each other's seed (Config→Config) form a 2-cycle.
        let doc = GraphDoc {
            nodes: vec![
                node(
                    "a",
                    NodeKind::Ik {
                        frame: None,
                        seed: None,
                    },
                ),
                node(
                    "b",
                    NodeKind::Ik {
                        frame: None,
                        seed: None,
                    },
                ),
            ],
            edges: vec![
                edge("a", "config", "b", "seed"),
                edge("b", "config", "a", "seed"),
            ],
            ..Default::default()
        };
        let d = validate(&doc, &r.model);
        assert!(!d.cycle.is_empty(), "cycle must be reported");
        assert!(d.topo_order.is_empty());
        assert!(!d.is_ok());
    }

    #[test]
    fn validation_rejects_type_mismatch() {
        let r = robot("showcase6.urdf");
        // StartConfig.config (Config) → View.clip (Clip) is a type mismatch.
        let doc = GraphDoc {
            nodes: vec![
                node("s", NodeKind::StartConfig { q: vec![0.0; 6] }),
                node("v", NodeKind::View {}),
            ],
            edges: vec![edge("s", "config", "v", "clip")],
            ..Default::default()
        };
        let d = validate(&doc, &r.model);
        assert!(
            !d.edge_errors.is_empty(),
            "type mismatch must be an edge error"
        );
        assert!(!d.is_ok());
    }

    #[test]
    fn validation_rejects_wrong_ndof_config() {
        let r = robot("showcase6.urdf"); // ndof 6
        let doc = GraphDoc {
            nodes: vec![node("s", NodeKind::StartConfig { q: vec![0.0; 3] })],
            edges: vec![],
            ..Default::default()
        };
        let d = validate(&doc, &r.model);
        assert!(
            d.node_errors.iter().any(|e| e.node_id == "s"),
            "wrong-ndof config must be a node error"
        );
    }

    #[test]
    fn pipeline_movel_matches_direct() {
        let r = robot("showcase6.urdf");
        let m = &r.model;
        let start = vec![0.0, -0.3, 0.5, 0.0, 0.4, 0.0];
        let q_goal = vec![0.2, -0.1, 0.6, 0.1, 0.3, -0.1];
        let goal_pose = se3_to_pose(&fk_tip(m, &q_goal));

        let doc = GraphDoc {
            nodes: vec![
                node("start", NodeKind::StartConfig { q: start.clone() }),
                node("goal", NodeKind::GoalPose { m: goal_pose }),
                node(
                    "ik",
                    NodeKind::Ik {
                        frame: None,
                        seed: None,
                    },
                ),
                node("mv", NodeKind::MoveL { frame: None }),
                node("view", NodeKind::View {}),
            ],
            edges: vec![
                edge("start", "config", "ik", "seed"),
                edge("goal", "pose", "ik", "pose"),
                edge("start", "config", "mv", "start"),
                edge("goal", "pose", "mv", "goal"),
                edge("mv", "clip", "view", "clip"),
            ],
            ..Default::default()
        };

        let res = run(&doc, &r).expect("pipeline runs");
        let clip = res.terminal_clip.expect("terminal clip present");
        assert!(clip.len() > 1, "terminal clip non-empty");

        // parity: the View clip equals calling move_l directly.
        let lim = limits(&r);
        let opts = CartesianMoveOpts::defaults(lim);
        let goal_se3 = pose_to_se3(&goal_pose);
        let direct = move_l(m, m.tip_frame(), &start, &goal_se3, &opts).unwrap();
        let direct_clip = bake_trajectory(&direct, CLIP_DT);
        assert!(
            clips_eq(&clip, &direct_clip, 1e-9),
            "MoveL pipeline clip must match move_l directly"
        );
    }

    #[test]
    fn pipeline_movej_matches_direct() {
        let r = robot("showcase6.urdf");
        let m = &r.model;
        let start = vec![0.0; 6];
        let goal = vec![0.4, -0.3, 0.5, 0.2, -0.4, 0.1];
        let doc = GraphDoc {
            nodes: vec![
                node("s", NodeKind::StartConfig { q: start.clone() }),
                node("g", NodeKind::StartConfig { q: goal.clone() }),
                node("mj", NodeKind::MoveJ {}),
                node("v", NodeKind::View {}),
            ],
            edges: vec![
                edge("s", "config", "mj", "start"),
                edge("g", "config", "mj", "goal"),
                edge("mj", "clip", "v", "clip"),
            ],
            ..Default::default()
        };
        let res = run(&doc, &r).unwrap();
        let clip = res.terminal_clip.unwrap();
        let direct = bake_trajectory(&move_j(m, &start, &goal, &limits(&r)).unwrap(), CLIP_DT);
        assert!(clip.len() > 1);
        assert!(clips_eq(&clip, &direct, 1e-9), "MoveJ parity");
    }

    #[test]
    fn planrrt_is_deterministic() {
        let r = robot("collide_arm.urdf"); // 3-dof
        let doc = GraphDoc {
            nodes: vec![
                node(
                    "s",
                    NodeKind::StartConfig {
                        q: vec![0.0, 0.0, 0.0],
                    },
                ),
                node(
                    "g",
                    NodeKind::StartConfig {
                        q: vec![0.4, -0.4, 0.4],
                    },
                ),
                node(
                    "plan",
                    NodeKind::PlanRrt {
                        seed: 0xCA11,
                        ground: None,
                        boxes: vec![],
                    },
                ),
                node("v", NodeKind::View {}),
            ],
            edges: vec![
                edge("s", "config", "plan", "start"),
                edge("g", "config", "plan", "goal"),
                edge("plan", "clip", "v", "clip"),
            ],
            ..Default::default()
        };
        let a = run(&doc, &r).unwrap().terminal_clip.unwrap();
        let b = run(&doc, &r).unwrap().terminal_clip.unwrap();
        assert!(a.len() > 1, "plan clip non-empty");
        assert!(clips_eq(&a, &b, 1e-12), "same seed ⇒ identical clip");
    }

    #[test]
    fn control_requires_inertia() {
        let r = robot("toy.urdf"); // 2-dof, no <inertial>
        assert!(!r.model.has_inertia);
        let doc = GraphDoc {
            nodes: vec![
                node("s", NodeKind::StartConfig { q: vec![0.0, 0.0] }),
                node("g", NodeKind::StartConfig { q: vec![0.1, 0.1] }),
                node(
                    "c",
                    NodeKind::Control {
                        kp: 100.0,
                        kd: 20.0,
                    },
                ),
                node("v", NodeKind::View {}),
            ],
            edges: vec![
                edge("s", "config", "c", "start"),
                edge("g", "config", "c", "goal"),
                edge("c", "clip", "v", "clip"),
            ],
            ..Default::default()
        };
        let err = run(&doc, &r).unwrap_err();
        match err {
            GraphError::Validation { diagnostics } => {
                assert!(
                    diagnostics
                        .node_errors
                        .iter()
                        .any(|e| e.message.contains("has_inertia")),
                    "Control on a non-inertia robot must fail validation"
                );
            }
            other => panic!("expected Validation error, got {other:?}"),
        }
    }

    #[test]
    fn scope_extracts_correct_series_length() {
        let r = robot("showcase6.urdf");
        let start = vec![0.0; 6];
        let goal = vec![0.3, -0.2, 0.4, 0.1, -0.3, 0.2];
        let doc = GraphDoc {
            nodes: vec![
                node("s", NodeKind::StartConfig { q: start }),
                node("g", NodeKind::StartConfig { q: goal }),
                node("mj", NodeKind::MoveJ {}),
                node(
                    "sc",
                    NodeKind::Scope {
                        signal: "q0".into(),
                    },
                ),
            ],
            edges: vec![
                edge("s", "config", "mj", "start"),
                edge("g", "config", "mj", "goal"),
                edge("mj", "clip", "sc", "clip"),
            ],
            ..Default::default()
        };
        let res = run(&doc, &r).unwrap();
        assert_eq!(res.scopes.len(), 1);
        let series = &res.scopes[0];
        // no View → terminal clip is the last produced clip (MoveJ).
        let clip = res.terminal_clip.as_ref().unwrap();
        assert!(clip.len() > 1);
        assert_eq!(series.y.len(), clip.len(), "scope y length == clip length");
        assert_eq!(series.t.len(), clip.len(), "scope t length == clip length");
        // q0 series equals each row's joint 0.
        for (k, row) in clip.qs.iter().enumerate() {
            assert!((series.y[k] - row[0]).abs() < 1e-12);
        }
    }

    #[test]
    fn graphdoc_roundtrips_through_serde_json_like() {
        // The IR must survive a serialize→deserialize cycle (faces persist it).
        // We avoid a serde_json dependency by checking the value-level invariants
        // a face relies on: ports/types are stable and resolvable.
        let k = NodeKind::PlanRrt {
            seed: 7,
            ground: Some(-0.1),
            boxes: vec![([0.5, 0.0, 0.3], [0.1, 0.1, 0.1])],
        };
        assert_eq!(k.type_name(), "planRrt");
        assert_eq!(k.in_port_names(), vec!["start", "goal"]);
        assert_eq!(k.out_port_names(), vec!["clip"]);
        assert_eq!(
            PortRef::Name("goal".into()).resolve(&k.in_port_names()),
            Some(1)
        );
        assert_eq!(PortRef::Index(0).resolve(&k.out_port_names()), Some(0));
    }
}
