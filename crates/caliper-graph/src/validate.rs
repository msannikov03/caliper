//! Structured graph validation (no panics). [`validate`] returns [`Diagnostics`]
//! with per-node and per-edge errors plus a topological order (or a cycle report).

use crate::ir::{GraphDoc, NodeKind, PortType};
use caliper_model::Model;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// A per-node validation error.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NodeDiag {
    pub node_id: String,
    pub message: String,
}

/// A per-edge validation error (indexed into `GraphDoc.edges`).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EdgeDiag {
    pub edge_index: usize,
    pub message: String,
}

/// The result of [`validate`]. `topo_order` is filled iff the graph is a DAG with
/// all endpoints resolvable; otherwise `cycle` lists the nodes left in the cycle.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct Diagnostics {
    pub node_errors: Vec<NodeDiag>,
    pub edge_errors: Vec<EdgeDiag>,
    /// Node ids in a valid execution order (empty if a cycle was found).
    pub topo_order: Vec<String>,
    /// Node ids participating in a cycle (empty if the graph is a DAG).
    pub cycle: Vec<String>,
}

impl Diagnostics {
    /// `true` iff there are no node/edge errors and no cycle.
    pub fn is_ok(&self) -> bool {
        self.node_errors.is_empty() && self.edge_errors.is_empty() && self.cycle.is_empty()
    }
}

/// A 1-D series a [`NodeKind::Scope`] can extract from a clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signal {
    Time,
    Q(usize),
    Qd(usize),
    TipX,
    TipY,
    TipZ,
    Energy,
}

/// Parse a `Scope` signal string. Recognized: `t`, `q<i>`, `qd<i>`, `tip_x`,
/// `tip_y`, `tip_z`, `energy`.
pub fn parse_signal(s: &str) -> Option<Signal> {
    match s {
        "t" => Some(Signal::Time),
        "tip_x" => Some(Signal::TipX),
        "tip_y" => Some(Signal::TipY),
        "tip_z" => Some(Signal::TipZ),
        "energy" => Some(Signal::Energy),
        _ => {
            if let Some(rest) = s.strip_prefix("qd") {
                rest.parse::<usize>().ok().map(Signal::Qd)
            } else if let Some(rest) = s.strip_prefix('q') {
                rest.parse::<usize>().ok().map(Signal::Q)
            } else {
                None
            }
        }
    }
}

fn all_finite(xs: &[f64]) -> bool {
    xs.iter().all(|x| x.is_finite())
}

fn boxes_finite(boxes: &[([f64; 3], [f64; 3])]) -> bool {
    boxes.iter().all(|(c, h)| all_finite(c) && all_finite(h))
}

/// Validate a document against a model. Never panics; returns structured errors.
pub fn validate(doc: &GraphDoc, model: &Model) -> Diagnostics {
    let n = model.ndof;
    let mut d = Diagnostics::default();

    // ---- node id index (first occurrence wins; duplicates are errors) ----
    let mut id_to_idx: HashMap<&str, usize> = HashMap::new();
    for (i, node) in doc.nodes.iter().enumerate() {
        if id_to_idx.contains_key(node.id.as_str()) {
            d.node_errors.push(NodeDiag {
                node_id: node.id.clone(),
                message: format!("duplicate node id `{}`", node.id),
            });
        } else {
            id_to_idx.insert(node.id.as_str(), i);
        }
    }

    // ---- per-node parameter checks ----
    for node in &doc.nodes {
        let mut err = |msg: String| {
            d.node_errors.push(NodeDiag {
                node_id: node.id.clone(),
                message: msg,
            })
        };
        let check_frame = |frame: &Option<String>, err: &mut dyn FnMut(String)| {
            if let Some(f) = frame
                && model.frame_id(f).is_none()
            {
                err(format!("unknown frame `{f}`"));
            }
        };
        match &node.kind {
            NodeKind::StartConfig { q } | NodeKind::NamedConfig { q, .. } => {
                if q.len() != n {
                    err(format!("config length {} != model.ndof {n}", q.len()));
                } else if !all_finite(q) {
                    err("config contains a non-finite value".into());
                }
            }
            NodeKind::GoalPose { m } => {
                if !all_finite(m) {
                    err("pose matrix contains a non-finite value".into());
                }
            }
            NodeKind::Ik { frame, seed } => {
                check_frame(frame, &mut err);
                if let Some(s) = seed {
                    if s.len() != n {
                        err(format!("seed length {} != model.ndof {n}", s.len()));
                    } else if !all_finite(s) {
                        err("seed contains a non-finite value".into());
                    }
                }
            }
            NodeKind::MoveJ {} => {}
            NodeKind::MoveL { frame } => check_frame(frame, &mut err),
            NodeKind::PlanRrt { ground, boxes, .. } => {
                if let Some(g) = ground
                    && !g.is_finite()
                {
                    err("ground z must be finite".into());
                }
                if !boxes_finite(boxes) {
                    err("obstacle box has a non-finite value".into());
                }
            }
            NodeKind::Control { kp, kd } => {
                if !kp.is_finite() || !kd.is_finite() {
                    err("kp/kd must be finite".into());
                }
                if !model.has_inertia {
                    err("Control requires a model with <inertial> data (has_inertia)".into());
                }
            }
            NodeKind::GravityDrop {
                gravity,
                duration,
                dt,
            } => {
                if !(duration.is_finite() && *duration > 0.0) {
                    err("duration must be finite and > 0".into());
                }
                if !(dt.is_finite() && *dt > 0.0) {
                    err("dt must be finite and > 0".into());
                }
                if let Some(g) = gravity
                    && !all_finite(g)
                {
                    err("gravity must be finite".into());
                }
                if !model.has_inertia {
                    err("GravityDrop requires a model with <inertial> data (has_inertia)".into());
                }
            }
            NodeKind::CollisionCheck { ground, boxes } => {
                if let Some(g) = ground
                    && !g.is_finite()
                {
                    err("ground z must be finite".into());
                }
                if !boxes_finite(boxes) {
                    err("obstacle box has a non-finite value".into());
                }
            }
            NodeKind::View {} => {}
            NodeKind::Scope { signal } => match parse_signal(signal) {
                None => err(format!("unknown scope signal `{signal}`")),
                Some(Signal::Q(i)) | Some(Signal::Qd(i)) if i >= n => {
                    err(format!("scope joint index {i} >= model.ndof {n}"))
                }
                Some(Signal::Energy) if !model.has_inertia => {
                    err("scope signal `energy` requires has_inertia".into())
                }
                _ => {}
            },
        }
    }

    // ---- edge endpoint / port resolution + type compatibility ----
    // Per-node, per-input-port: which edge feeds it (to detect missing/duplicate).
    let mut feeders: Vec<Vec<Option<usize>>> = doc
        .nodes
        .iter()
        .map(|node| vec![None; node.kind.in_ports().len()])
        .collect();
    // Edges with both endpoints valid (for the DAG pass): (from_idx, to_idx).
    let mut dag_edges: Vec<(usize, usize)> = Vec::new();

    for (ei, edge) in doc.edges.iter().enumerate() {
        let mut eerr = |msg: String| {
            d.edge_errors.push(EdgeDiag {
                edge_index: ei,
                message: msg,
            })
        };
        let from_idx = id_to_idx.get(edge.from.as_str());
        let to_idx = id_to_idx.get(edge.to.as_str());
        let (fi, ti) = match (from_idx, to_idx) {
            (Some(&fi), Some(&ti)) => (fi, ti),
            _ => {
                if from_idx.is_none() {
                    eerr(format!(
                        "edge `from` references unknown node `{}`",
                        edge.from
                    ));
                }
                if to_idx.is_none() {
                    eerr(format!("edge `to` references unknown node `{}`", edge.to));
                }
                continue;
            }
        };
        dag_edges.push((fi, ti));

        let from_kind = &doc.nodes[fi].kind;
        let to_kind = &doc.nodes[ti].kind;
        let out_names = from_kind.out_port_names();
        let in_names = to_kind.in_port_names();

        let op = match edge.from_port.resolve(&out_names) {
            Some(i) => i,
            None => {
                eerr(format!(
                    "`from` port {:?} not found on node `{}` (out ports: {:?})",
                    edge.from_port, edge.from, out_names
                ));
                continue;
            }
        };
        let ip = match edge.to_port.resolve(&in_names) {
            Some(i) => i,
            None => {
                eerr(format!(
                    "`to` port {:?} not found on node `{}` (in ports: {:?})",
                    edge.to_port, edge.to, in_names
                ));
                continue;
            }
        };

        let out_ty: PortType = from_kind.out_ports()[op].ty;
        let in_port = to_kind.in_ports()[ip];
        if !in_port.types.contains(&out_ty) {
            eerr(format!(
                "type mismatch: `{}.{}` is {:?} but `{}.{}` expects {:?}",
                edge.from, out_names[op], out_ty, edge.to, in_names[ip], in_port.types
            ));
            continue;
        }

        match feeders[ti][ip] {
            None => feeders[ti][ip] = Some(ei),
            Some(prev) => eerr(format!(
                "input port `{}.{}` already fed by edge #{prev}",
                edge.to, in_names[ip]
            )),
        }
    }

    // ---- required input ports must be connected ----
    for (ni, node) in doc.nodes.iter().enumerate() {
        for (pi, port) in node.kind.in_ports().iter().enumerate() {
            if port.required && feeders[ni][pi].is_none() {
                d.node_errors.push(NodeDiag {
                    node_id: node.id.clone(),
                    message: format!("required input port `{}` is not connected", port.name),
                });
            }
        }
    }

    // ---- DAG check (Kahn) over the resolvable edges ----
    let count = doc.nodes.len();
    let mut indeg = vec![0usize; count];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); count];
    for &(f, t) in &dag_edges {
        adj[f].push(t);
        indeg[t] += 1;
    }
    let mut queue: VecDeque<usize> = (0..count).filter(|&i| indeg[i] == 0).collect();
    let mut order: Vec<usize> = Vec::with_capacity(count);
    let mut indeg_work = indeg.clone();
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &v in &adj[u] {
            indeg_work[v] -= 1;
            if indeg_work[v] == 0 {
                queue.push_back(v);
            }
        }
    }
    if order.len() == count {
        d.topo_order = order.into_iter().map(|i| doc.nodes[i].id.clone()).collect();
    } else {
        let placed: HashSet<usize> = order.into_iter().collect();
        d.cycle = (0..count)
            .filter(|i| !placed.contains(i))
            .map(|i| doc.nodes[i].id.clone())
            .collect();
        d.node_errors.push(NodeDiag {
            node_id: d.cycle.first().cloned().unwrap_or_default(),
            message: format!("graph has a cycle through nodes {:?}", d.cycle),
        });
    }

    d
}
