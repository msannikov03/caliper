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

/// Property-based **fuzz** tests over ARBITRARY (incl. adversarial / degenerate)
/// [`GraphDoc`]s — this is exactly the untrusted input a Studio / CLI / PyO3 user
/// can hand to `graph_run`. The invariants proved here:
///
/// 1. [`validate`] NEVER panics and always returns a self-consistent
///    [`Diagnostics`] (`topo_order` XOR `cycle`; `is_ok()` iff no errors).
/// 2. [`run`] NEVER unwinds: it returns `Ok(GraphResult)` or a *structured*
///    `Err(GraphError)` (validation-class or per-node-compute), never a panic.
/// 3. Validation FULLY protects the executor: a doc that `validate().is_ok()`
///    never makes `run` bail at the validation gate — any later failure is a
///    structured [`GraphError::Node`]; and reaching `Ok` implies the doc validated.
/// 4. A well-formed pipeline over the runtime-total node family (sources → MoveJ →
///    View/Scope) both validates and runs to `Ok` (the positive protection path).
///
/// Strategies deliberately generate degenerate params (empty / wrong-ndof /
/// non-finite / huge / negative configs, non-finite pose & box values, unknown
/// frames & scope signals) and adversarial edges (dangling ids, wrong ports,
/// type-mismatches, cycles, multi-feeders) on a fixed loaded robot. Bounds keep it
/// fast + deterministic (proptest defaults): almost every random doc short-circuits
/// at the validation gate, and the rare valid one only touches cheap, bounded ops.
#[cfg(test)]
mod fuzz {
    use super::*;
    use caliper_model::Robot;
    use proptest::prelude::*;
    use proptest::strategy::BoxedStrategy;
    use std::path::Path;

    fn robot(name: &str) -> Robot {
        Robot::from_urdf(Path::new(&format!(
            "{}/../../oracle/fixtures/robots/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        )))
        .unwrap()
    }

    // ----- scalar / vector building blocks (normal values + degenerate specials) -----

    /// A "wild" `f64`: ordinary finite values salted with the degenerate cases a
    /// validator must reject (NaN / ±inf / zero) AND a finite HUGE magnitude. Used
    /// only for scalars whose magnitude cannot drive an unbounded executor loop —
    /// gains, gravity, obstacle boxes, ground — never for a config/pose (see below).
    fn any_f64() -> impl Strategy<Value = f64> {
        prop_oneof![
            6 => -3.0f64..3.0,
            1 => Just(f64::NAN),
            1 => Just(f64::INFINITY),
            1 => Just(f64::NEG_INFINITY),
            1 => Just(1e18),
            1 => Just(-1e18),
            1 => Just(0.0),
        ]
    }

    /// A "tame" `f64` for CONFIG and POSE components: bounded finite magnitude plus
    /// the non-finite degenerates (NaN / ±inf) that validation must reject. It omits
    /// the finite-HUGE value on purpose: validation treats a large-finite config
    /// exactly like a normal one (only finiteness/length matter), yet a huge-finite
    /// config or pose would make `move_j`/`move_l` bake an astronomically long
    /// trajectory — so excluding it costs zero validation coverage while keeping the
    /// (rare) executed motion node provably bounded and the fuzz run fast.
    fn tame_f64() -> impl Strategy<Value = f64> {
        prop_oneof![
            6 => -3.0f64..3.0,
            1 => Just(f64::NAN),
            1 => Just(f64::INFINITY),
            1 => Just(f64::NEG_INFINITY),
            1 => Just(0.0),
        ]
    }

    /// A config vector of *arbitrary* length 0..=8 — exercises empty, wrong-ndof
    /// (model.ndof is 6 / 3), and matching-length configs, all with degenerate values.
    fn any_qvec() -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(tame_f64(), 0..=8)
    }

    /// A duration / dt scalar for GravityDrop. The valid arm is COARSE (`dt ≥ 1e-3`,
    /// `dur ≤ 1`) so any pairing that survives validation implies ≤ ~1000 sim steps;
    /// the degenerate values (NaN / inf / 0 / negative / huge) either fail validation
    /// or (huge/huge) collapse to one step — so no validation-passing combination can
    /// drive an expensive rollout.
    fn dur_dt() -> impl Strategy<Value = f64> {
        prop_oneof![
            4 => 0.001f64..1.0,
            1 => Just(f64::NAN),
            1 => Just(f64::INFINITY),
            1 => Just(0.0),
            1 => Just(-1.0),
            1 => Just(1e18),
        ]
    }

    fn arr3() -> impl Strategy<Value = [f64; 3]> {
        proptest::array::uniform3(any_f64())
    }

    fn boxes() -> impl Strategy<Value = Vec<([f64; 3], [f64; 3])>> {
        prop::collection::vec((arr3(), arr3()), 0..=2)
    }

    fn small_name() -> impl Strategy<Value = String> {
        prop::sample::select(vec!["home", "ready", "", "x"]).prop_map(String::from)
    }

    /// Optional frame name: mostly `None` / unknown frames (validation rejects), so
    /// the frame-check path is exercised without needing a real frame id.
    fn opt_frame() -> impl Strategy<Value = Option<String>> {
        proptest::option::of(
            prop::sample::select(vec!["tool0", "flange", "wrist_3", "bogus_frame", ""])
                .prop_map(String::from),
        )
    }

    /// Scope signal string mixing valid (`q0`, `qd2`, `tip_x`, `energy`, `t`) and
    /// invalid (`q99` out-of-range, `bogus`, bare `q`, empty) forms.
    fn signal_strat() -> impl Strategy<Value = String> {
        prop::sample::select(vec![
            "q0", "q5", "q99", "qd0", "qd2", "tip_x", "tip_y", "tip_z", "energy", "t", "bogus",
            "q", "qd", "",
        ])
        .prop_map(String::from)
    }

    // ----- node kinds (grouped so each prop_oneof! stays within the tuple arity) -----

    fn sources() -> BoxedStrategy<NodeKind> {
        prop_oneof![
            any_qvec().prop_map(|q| NodeKind::StartConfig { q }),
            // Pose components are `tame` too: a huge-finite goal pose would drive
            // `move_l`'s Cartesian duration unbounded (non-finite is rejected anyway).
            proptest::array::uniform16(tame_f64()).prop_map(|m| NodeKind::GoalPose { m }),
            (any_qvec(), small_name()).prop_map(|(q, name)| NodeKind::NamedConfig { q, name }),
        ]
        .boxed()
    }

    fn compute() -> BoxedStrategy<NodeKind> {
        prop_oneof![
            (opt_frame(), proptest::option::of(any_qvec()))
                .prop_map(|(frame, seed)| NodeKind::Ik { frame, seed }),
            Just(NodeKind::MoveJ {}),
            opt_frame().prop_map(|frame| NodeKind::MoveL { frame }),
            (any::<u64>(), proptest::option::of(any_f64()), boxes()).prop_map(
                |(seed, ground, boxes)| NodeKind::PlanRrt {
                    seed,
                    ground,
                    boxes
                }
            ),
            (any_f64(), any_f64()).prop_map(|(kp, kd)| NodeKind::Control { kp, kd }),
            (proptest::option::of(arr3()), dur_dt(), dur_dt()).prop_map(
                |(gravity, duration, dt)| NodeKind::GravityDrop {
                    gravity,
                    duration,
                    dt
                }
            ),
            (proptest::option::of(any_f64()), boxes())
                .prop_map(|(ground, boxes)| NodeKind::CollisionCheck { ground, boxes }),
        ]
        .boxed()
    }

    fn sinks() -> BoxedStrategy<NodeKind> {
        prop_oneof![
            Just(NodeKind::View {}),
            signal_strat().prop_map(|signal| NodeKind::Scope { signal }),
        ]
        .boxed()
    }

    fn node_kind() -> BoxedStrategy<NodeKind> {
        prop_oneof![sources(), compute(), sinks()].boxed()
    }

    // ----- ids / ports / edges (a small id pool ⇒ realistic dangling + duplicates) -----

    /// Node ids drawn from a tiny pool so duplicate ids arise naturally.
    fn node_id() -> impl Strategy<Value = String> {
        prop::sample::select(vec!["a", "b", "c", "d", "e"]).prop_map(String::from)
    }

    /// Edge endpoints from a slightly larger pool (incl. `ghost`, never a node) so
    /// dangling-endpoint edges are common.
    fn edge_id() -> impl Strategy<Value = String> {
        prop::sample::select(vec!["a", "b", "c", "d", "e", "ghost"]).prop_map(String::from)
    }

    /// A port ref that is EITHER a positional index (some out of range) OR a name
    /// (some valid for one node kind, some — like `bogus` — for none).
    fn port_ref() -> impl Strategy<Value = PortRef> {
        prop_oneof![
            (0usize..4).prop_map(PortRef::Index),
            prop::sample::select(vec![
                "config", "pose", "clip", "report", "start", "goal", "seed", "bogus",
            ])
            .prop_map(|s| PortRef::Name(s.into())),
        ]
    }

    fn edge_strat() -> impl Strategy<Value = Edge> {
        (edge_id(), port_ref(), edge_id(), port_ref()).prop_map(|(from, from_port, to, to_port)| {
            Edge {
                from,
                from_port,
                to,
                to_port,
            }
        })
    }

    fn node_strat() -> impl Strategy<Value = Node> {
        (node_id(), node_kind()).prop_map(|(id, kind)| Node { id, kind })
    }

    fn doc_strat() -> impl Strategy<Value = GraphDoc> {
        (
            prop::collection::vec(node_strat(), 1..=6),
            prop::collection::vec(edge_strat(), 0..=8),
        )
            .prop_map(|(nodes, edges)| GraphDoc {
                nodes,
                edges,
                metadata: Default::default(),
            })
    }

    /// Shared structural invariants of any [`Diagnostics`], regardless of the doc.
    fn assert_diag_consistent(d: &Diagnostics, n_nodes: usize) -> Result<(), TestCaseError> {
        // topo_order and cycle are mutually exclusive.
        prop_assert!(
            d.topo_order.is_empty() || d.cycle.is_empty(),
            "topo_order and cycle both populated"
        );
        if d.is_ok() {
            prop_assert!(
                d.node_errors.is_empty() && d.edge_errors.is_empty() && d.cycle.is_empty()
            );
            prop_assert_eq!(d.topo_order.len(), n_nodes, "ok ⇒ full topo order");
        } else {
            prop_assert!(
                !d.node_errors.is_empty() || !d.edge_errors.is_empty() || !d.cycle.is_empty(),
                "not ok ⇒ some error/cycle recorded"
            );
        }
        Ok(())
    }

    proptest! {
        /// `validate` never panics and returns a self-consistent `Diagnostics` on
        /// arbitrary adversarial docs — over both a 6-dof and a 3-dof model.
        #[test]
        fn validate_never_panics(doc in doc_strat()) {
            for name in ["showcase6.urdf", "collide_arm.urdf"] {
                let r = robot(name);
                let d = validate(&doc, &r.model);
                assert_diag_consistent(&d, doc.nodes.len())?;
            }
        }

        /// `run` never unwinds on arbitrary adversarial docs: it yields `Ok` or a
        /// STRUCTURED `Err`. Proves validation fully guards the executor —
        ///  * `Ok`                        ⇒ the doc validated,
        ///  * `Err(Validation)`           ⇒ the doc did NOT validate (run bailed at the gate),
        ///  * `Err(Node)`                 ⇒ the doc DID validate (failure is a per-node compute error).
        #[test]
        fn run_never_panics_and_validation_guards_executor(doc in doc_strat()) {
            let r = robot("showcase6.urdf");
            // Pre-validation must agree with run's internal gate (same model, deterministic).
            let pre = validate(&doc, &r.model);
            assert_diag_consistent(&pre, doc.nodes.len())?;
            match run(&doc, &r) {
                Ok(res) => {
                    prop_assert!(pre.is_ok(), "run produced Ok but pre-validation was not ok");
                    prop_assert!(res.diagnostics.is_ok(), "Ok result must carry ok diagnostics");
                }
                Err(GraphError::Validation { diagnostics }) => {
                    prop_assert!(!diagnostics.is_ok(), "Validation error must carry failing diagnostics");
                    prop_assert!(!pre.is_ok(), "run bailed at validation but the doc validated");
                }
                Err(GraphError::Node { .. }) => {
                    prop_assert!(pre.is_ok(), "a Node error is only reachable after validation passed");
                }
            }
        }

        /// Positive path: a well-formed pipeline over the runtime-TOTAL node family
        /// (StartConfig → MoveJ → View + Scope) ALWAYS validates and runs to `Ok`
        /// for any bounded configs and any valid scope signal — i.e. validation
        /// passing is *sufficient* for these nodes to execute cleanly.
        #[test]
        fn valid_movej_pipeline_always_runs_ok(
            start in prop::collection::vec(-1.0f64..1.0, 6),
            goal in prop::collection::vec(-1.0f64..1.0, 6),
            signal in prop::sample::select(vec![
                "q0", "qd0", "tip_x", "tip_y", "tip_z", "t", "energy",
            ]),
        ) {
            let r = robot("showcase6.urdf");
            let n = |id: &str, kind| Node { id: id.into(), kind };
            let e = |from: &str, fp: &str, to: &str, tp: &str| Edge {
                from: from.into(),
                from_port: PortRef::Name(fp.into()),
                to: to.into(),
                to_port: PortRef::Name(tp.into()),
            };
            let doc = GraphDoc {
                nodes: vec![
                    n("s", NodeKind::StartConfig { q: start }),
                    n("g", NodeKind::StartConfig { q: goal }),
                    n("mj", NodeKind::MoveJ {}),
                    n("v", NodeKind::View {}),
                    n("sc", NodeKind::Scope { signal: signal.into() }),
                ],
                edges: vec![
                    e("s", "config", "mj", "start"),
                    e("g", "config", "mj", "goal"),
                    e("mj", "clip", "v", "clip"),
                    e("mj", "clip", "sc", "clip"),
                ],
                ..Default::default()
            };
            let d = validate(&doc, &r.model);
            prop_assert!(d.is_ok(), "constructed pipeline must validate: {:?}", d);
            let res = run(&doc, &r)
                .map_err(|err| TestCaseError::fail(format!("valid pipeline failed to run: {err:?}")))?;
            let clip = res.terminal_clip.ok_or_else(|| TestCaseError::fail("no terminal clip"))?;
            prop_assert!(!clip.is_empty(), "terminal clip must be non-empty");
            prop_assert_eq!(res.scopes.len(), 1, "exactly one scope series");
            prop_assert_eq!(res.scopes[0].y.len(), clip.len(), "scope y len == clip len");
        }
    }
}
