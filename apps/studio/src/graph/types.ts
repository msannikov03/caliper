// Shared types for the Phase-8 Graph mode (Simulink-style dataflow editor).
// These mirror the caliper-graph serde shapes (all camelCase) reached through the
// Tauri `graph_*` commands. See crates/caliper-graph/src/{ir,validate,exec}.rs.

import type { Node, Edge } from "@xyflow/react";
import type { TrajectoryDto } from "../store";
import type { KindName } from "./spec";

/** Per-node run/validation status, drives the header status ring. */
export type NodeStatus = "idle" | "running" | "ok" | "error";

/**
 * The `data` payload carried by every xyflow node. The index signature keeps it
 * assignable to React Flow's `Record<string, unknown>` node-data constraint while
 * still giving us typed fields.
 */
export interface CaliperNodeData {
  /** The node kind (== the serde `"type"` tag and the xyflow node `type`). */
  kind: KindName;
  /** Inline UI parameter bag; projected to the kind's JSON params on serialize. */
  params: Record<string, unknown>;
  /** Status ring state. */
  status: NodeStatus;
  /** Last validation / run error message for this node (tooltip + ring). */
  error?: string;
  [key: string]: unknown;
}

export type CNode = Node<CaliperNodeData>;
export type CEdge = Edge;

// ---- caliper-graph::validate::Diagnostics (camelCase) ----
export interface NodeDiag {
  nodeId: string;
  message: string;
}
export interface EdgeDiag {
  edgeIndex: number;
  message: string;
}
export interface Diagnostics {
  nodeErrors: NodeDiag[];
  edgeErrors: EdgeDiag[];
  topoOrder: string[];
  cycle: string[];
}

/** A `Scope` node's extracted 1-D series (caliper-graph::exec::ScopeSeries). */
export interface GraphScope {
  nodeId: string;
  signal: string;
  t: number[];
  y: number[];
}

/**
 * What the `graph_run` Tauri command resolves to. The backend bakes the graph's
 * terminal clip into a full `TrajectoryDto` (with render frames) so the persistent
 * GL preview can play it back; it also forwards the scopes + diagnostics. We keep
 * `terminalClip`/`diagnostics` optional so the face is resilient to either shape.
 */
export interface GraphRunResult {
  trajectory?: TrajectoryDto | null;
  scopes?: GraphScope[];
  diagnostics?: Diagnostics;
}

/** True iff diagnostics carry no node/edge errors and no cycle. */
export function diagnosticsOk(d: Diagnostics): boolean {
  return (
    d.nodeErrors.length === 0 && d.edgeErrors.length === 0 && d.cycle.length === 0
  );
}
