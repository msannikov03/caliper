import type { ReactNode } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps, NodeTypes } from "@xyflow/react";
import { useStore } from "../store";
import type { CaliperNodeData, NodeStatus } from "./types";
import type { KindName } from "./spec";
import {
  NODE_SPECS,
  PORT_COLORS,
  inHandleColor,
  signalOptions,
} from "./spec";
import { ScopeChart } from "./ScopeChart";

const PI = Math.PI;
type Box = [[number, number, number], [number, number, number]];

function dataOf(p: NodeProps): CaliperNodeData {
  return p.data as unknown as CaliperNodeData;
}
function handleTop(i: number, n: number): string {
  return `${((i + 1) / (n + 1)) * 100}%`;
}

// ---- shared node chrome: title, status ring, typed color-coded handles ----
function NodeShell({
  kind,
  status,
  error,
  children,
}: {
  kind: KindName;
  status: NodeStatus;
  error?: string;
  children?: ReactNode;
}) {
  const spec = NODE_SPECS[kind];
  return (
    <div className={`gnode gnode-${spec.category} gn-${status}`} title={error ?? spec.blurb}>
      <div className="gnode-head">
        <span className={`gnode-ring ${status}`} />
        <span className="gnode-title">{spec.label}</span>
      </div>
      {spec.inputs.map((port, i) => (
        <Handle
          key={`in-${port.name}`}
          type="target"
          position={Position.Left}
          id={port.name}
          className="ghandle"
          style={{ top: handleTop(i, spec.inputs.length), background: inHandleColor(port) }}
        />
      ))}
      {spec.inputs.map((port, i) => (
        <span
          key={`inl-${port.name}`}
          className="ghandle-label in"
          style={{ top: handleTop(i, spec.inputs.length) }}
        >
          {port.name}
          {port.required ? "" : "?"}
        </span>
      ))}
      {spec.outputs.map((port, i) => (
        <Handle
          key={`out-${port.name}`}
          type="source"
          position={Position.Right}
          id={port.name}
          className="ghandle"
          style={{ top: handleTop(i, spec.outputs.length), background: PORT_COLORS[port.type] }}
        />
      ))}
      {spec.outputs.map((port, i) => (
        <span
          key={`outl-${port.name}`}
          className="ghandle-label out"
          style={{ top: handleTop(i, spec.outputs.length) }}
        >
          {port.name}
        </span>
      ))}
      <div className="gnode-body">{children}</div>
    </div>
  );
}

// ---- reusable inline field controls (all marked nodrag) ----
function NumField({
  label,
  value,
  onChange,
  step = 0.01,
  min,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
}) {
  return (
    <label className="gfield">
      <span>{label}</span>
      <input
        className="nodrag"
        type="number"
        step={step}
        min={min}
        value={Number.isFinite(value) ? value : 0}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (!Number.isNaN(v)) onChange(v);
        }}
      />
    </label>
  );
}

function TextField({
  label,
  value,
  placeholder,
  onChange,
}: {
  label: string;
  value: string;
  placeholder?: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="gfield">
      <span>{label}</span>
      <input
        className="nodrag"
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

function CheckRow({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="gcheck">
      <input
        className="nodrag"
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      {label}
    </label>
  );
}

function ConfigSliders({ id, q }: { id: string; q: number[] }) {
  const robot = useStore((s) => s.robot);
  const update = useStore((s) => s.updateNodeParams);
  if (!robot) return null;
  return (
    <div className="gconf">
      {robot.jointNames.map((nm, i) => {
        const kind = robot.jointKinds[i];
        const lim = robot.limits[i];
        const [lo, hi] = lim ?? (kind === "prismatic" ? [-0.5, 0.5] : [-PI, PI]);
        const v = q[i] ?? 0;
        return (
          <div className="gconf-row" key={i}>
            <span className="gconf-name" title={nm}>
              {nm}
            </span>
            <input
              className="nodrag"
              type="range"
              min={lo}
              max={hi}
              step={0.001}
              value={v}
              onChange={(e) => {
                const nq = q.slice();
                nq[i] = parseFloat(e.target.value);
                update(id, { q: nq });
              }}
            />
            <span className="gconf-val">{v.toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}

function BoxesEditor({ id, boxes }: { id: string; boxes: Box[] }) {
  const update = useStore((s) => s.updateNodeParams);
  const set = (b: Box[]) => update(id, { boxes: b });
  return (
    <div className="gboxes">
      <div className="gboxes-head">
        <span>obstacle boxes</span>
        <button
          className="nodrag"
          onClick={() => set([...boxes, [[0, 0, 0.2], [0.1, 0.1, 0.1]]])}
        >
          +
        </button>
      </div>
      {boxes.map((bx, bi) => (
        <div className="gbox-row" key={bi}>
          {[0, 1, 2, 3, 4, 5].map((ci) => {
            const part = ci < 3 ? 0 : 1;
            const k = ci % 3;
            return (
              <input
                key={ci}
                className="nodrag"
                type="number"
                step={0.05}
                value={bx[part][k]}
                title={part === 0 ? `center ${"xyz"[k]}` : `half ${"xyz"[k]}`}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (Number.isNaN(v)) return;
                  const nb = boxes.map((r) => [[...r[0]], [...r[1]]] as Box);
                  nb[bi][part][k] = v;
                  set(nb);
                }}
              />
            );
          })}
          <button className="nodrag" onClick={() => set(boxes.filter((_, j) => j !== bi))}>
            ×
          </button>
        </div>
      ))}
    </div>
  );
}

// ===== the 12 node kinds =====
function StartConfigNode(p: NodeProps) {
  const d = dataOf(p);
  return (
    <NodeShell kind="startConfig" status={d.status} error={d.error}>
      <ConfigSliders id={p.id} q={(d.params.q as number[]) ?? []} />
    </NodeShell>
  );
}

function NamedConfigNode(p: NodeProps) {
  const d = dataOf(p);
  const update = useStore((s) => s.updateNodeParams);
  return (
    <NodeShell kind="namedConfig" status={d.status} error={d.error}>
      <TextField
        label="name"
        value={(d.params.name as string) ?? ""}
        onChange={(v) => update(p.id, { name: v })}
      />
      <ConfigSliders id={p.id} q={(d.params.q as number[]) ?? []} />
    </NodeShell>
  );
}

function GoalPoseNode(p: NodeProps) {
  const d = dataOf(p);
  const update = useStore((s) => s.updateNodeParams);
  const pr = d.params;
  const field = (label: string, key: string, step: number) => (
    <NumField
      label={label}
      step={step}
      value={(pr[key] as number) ?? 0}
      onChange={(v) => update(p.id, { [key]: v })}
    />
  );
  return (
    <NodeShell kind="goalPose" status={d.status} error={d.error}>
      <div className="gpose-grid">
        {field("x", "x", 0.01)}
        {field("y", "y", 0.01)}
        {field("z", "z", 0.01)}
        {field("roll", "roll", 0.05)}
        {field("pitch", "pitch", 0.05)}
        {field("yaw", "yaw", 0.05)}
      </div>
    </NodeShell>
  );
}

function IkNode(p: NodeProps) {
  const d = dataOf(p);
  const update = useStore((s) => s.updateNodeParams);
  return (
    <NodeShell kind="ik" status={d.status} error={d.error}>
      <TextField
        label="frame"
        placeholder="(tip)"
        value={(d.params.frame as string) ?? ""}
        onChange={(v) => update(p.id, { frame: v })}
      />
    </NodeShell>
  );
}

function MoveJNode(p: NodeProps) {
  const d = dataOf(p);
  return <NodeShell kind="moveJ" status={d.status} error={d.error} />;
}

function MoveLNode(p: NodeProps) {
  const d = dataOf(p);
  const update = useStore((s) => s.updateNodeParams);
  return (
    <NodeShell kind="moveL" status={d.status} error={d.error}>
      <TextField
        label="frame"
        placeholder="(tip)"
        value={(d.params.frame as string) ?? ""}
        onChange={(v) => update(p.id, { frame: v })}
      />
    </NodeShell>
  );
}

function PlanRrtNode(p: NodeProps) {
  const d = dataOf(p);
  const update = useStore((s) => s.updateNodeParams);
  const groundOn = !!d.params.groundOn;
  return (
    <NodeShell kind="planRrt" status={d.status} error={d.error}>
      <NumField
        label="seed"
        step={1}
        min={0}
        value={(d.params.seed as number) ?? 1}
        onChange={(v) => update(p.id, { seed: Math.max(0, Math.trunc(v)) })}
      />
      <CheckRow
        label="ground plane"
        checked={groundOn}
        onChange={(v) => update(p.id, { groundOn: v })}
      />
      {groundOn && (
        <NumField
          label="ground z"
          step={0.01}
          value={(d.params.ground as number) ?? 0}
          onChange={(v) => update(p.id, { ground: v })}
        />
      )}
      <BoxesEditor id={p.id} boxes={(d.params.boxes as Box[]) ?? []} />
    </NodeShell>
  );
}

function ControlNode(p: NodeProps) {
  const d = dataOf(p);
  const update = useStore((s) => s.updateNodeParams);
  return (
    <NodeShell kind="control" status={d.status} error={d.error}>
      <NumField
        label="kp"
        step={1}
        value={(d.params.kp as number) ?? 100}
        onChange={(v) => update(p.id, { kp: v })}
      />
      <NumField
        label="kd"
        step={1}
        value={(d.params.kd as number) ?? 20}
        onChange={(v) => update(p.id, { kd: v })}
      />
    </NodeShell>
  );
}

function GravityDropNode(p: NodeProps) {
  const d = dataOf(p);
  const update = useStore((s) => s.updateNodeParams);
  return (
    <NodeShell kind="gravityDrop" status={d.status} error={d.error}>
      <CheckRow
        label="earth gravity"
        checked={!!d.params.gravityOn}
        onChange={(v) => update(p.id, { gravityOn: v })}
      />
      <NumField
        label="duration"
        step={0.1}
        value={(d.params.duration as number) ?? 2}
        onChange={(v) => update(p.id, { duration: v })}
      />
      <NumField
        label="dt"
        step={0.0005}
        value={(d.params.dt as number) ?? 0.001}
        onChange={(v) => update(p.id, { dt: v })}
      />
    </NodeShell>
  );
}

function CollisionCheckNode(p: NodeProps) {
  const d = dataOf(p);
  const update = useStore((s) => s.updateNodeParams);
  const groundOn = !!d.params.groundOn;
  return (
    <NodeShell kind="collisionCheck" status={d.status} error={d.error}>
      <CheckRow
        label="ground plane"
        checked={groundOn}
        onChange={(v) => update(p.id, { groundOn: v })}
      />
      {groundOn && (
        <NumField
          label="ground z"
          step={0.01}
          value={(d.params.ground as number) ?? 0}
          onChange={(v) => update(p.id, { ground: v })}
        />
      )}
      <BoxesEditor id={p.id} boxes={(d.params.boxes as Box[]) ?? []} />
    </NodeShell>
  );
}

function ViewNode(p: NodeProps) {
  const d = dataOf(p);
  return (
    <NodeShell kind="view" status={d.status} error={d.error}>
      <div className="gnode-note">→ plays in the 3D preview</div>
    </NodeShell>
  );
}

function ScopeNode(p: NodeProps) {
  const d = dataOf(p);
  const update = useStore((s) => s.updateNodeParams);
  const robot = useStore((s) => s.robot);
  const series = useStore((s) => s.graphScopes.find((x) => x.nodeId === p.id));
  const signal = (d.params.signal as string) ?? "q0";
  const opts = signalOptions(robot?.ndof ?? 0);
  return (
    <NodeShell kind="scope" status={d.status} error={d.error}>
      <label className="gfield">
        <span>signal</span>
        <select
          className="nodrag"
          value={signal}
          onChange={(e) => update(p.id, { signal: e.target.value })}
        >
          {opts.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      </label>
      {series && series.t.length > 0 && series.signal === signal ? (
        <ScopeChart t={series.t} y={series.y} label={signal} />
      ) : (
        <div className="gscope-empty">run to plot {signal}</div>
      )}
    </NodeShell>
  );
}

export const nodeTypes: NodeTypes = {
  startConfig: StartConfigNode,
  goalPose: GoalPoseNode,
  namedConfig: NamedConfigNode,
  ik: IkNode,
  moveJ: MoveJNode,
  moveL: MoveLNode,
  planRrt: PlanRrtNode,
  control: ControlNode,
  gravityDrop: GravityDropNode,
  collisionCheck: CollisionCheckNode,
  view: ViewNode,
  scope: ScopeNode,
};
