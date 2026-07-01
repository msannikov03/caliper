import { useEffect, useState } from "react";
import { ReactFlow, Background, BackgroundVariant, Controls, MiniMap } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import "./graph.css";
import { useStore } from "../store";
import { nodeTypes } from "./nodes";
import { KIND_ORDER, NODE_SPECS } from "./spec";
import type { KindName, NodeCategory } from "./spec";

const CATEGORY_LABEL: Record<NodeCategory, string> = {
  source: "Sources",
  compute: "Compute",
  sink: "Sinks",
};

function Palette() {
  const addGraphNode = useStore((s) => s.addGraphNode);
  const cats: NodeCategory[] = ["source", "compute", "sink"];
  return (
    <div className="g-palette">
      <div className="g-palette-title">Add node</div>
      {cats.map((cat) => (
        <div className="g-palette-group" key={cat}>
          <div className="g-palette-cat">{CATEGORY_LABEL[cat]}</div>
          {KIND_ORDER.filter((k) => NODE_SPECS[k].category === cat).map((k: KindName) => (
            <button
              key={k}
              className={`g-palette-btn pb-${cat}`}
              title={NODE_SPECS[k].blurb}
              onClick={() => addGraphNode(k)}
            >
              {NODE_SPECS[k].label}
            </button>
          ))}
        </div>
      ))}
    </div>
  );
}

export function GraphEditor() {
  const nodes = useStore((s) => s.graphNodes);
  const edges = useStore((s) => s.graphEdges);
  const onNodesChange = useStore((s) => s.onGraphNodesChange);
  const onEdgesChange = useStore((s) => s.onGraphEdgesChange);
  const onConnect = useStore((s) => s.onGraphConnect);
  const runGraph = useStore((s) => s.runGraph);
  const runGraphLive = useStore((s) => s.runGraphLive);
  const validateGraph = useStore((s) => s.validateGraph);
  const banner = useStore((s) => s.graphBanner);
  const graphName = useStore((s) => s.graphName);
  const setGraphName = useStore((s) => s.setGraphName);
  const saveGraph = useStore((s) => s.saveGraph);
  const loadGraph = useStore((s) => s.loadGraph);
  const refreshGraphList = useStore((s) => s.refreshGraphList);
  const saved = useStore((s) => s.graphSaved);
  const robot = useStore((s) => s.robot);
  const [loadSel, setLoadSel] = useState("");

  useEffect(() => {
    void refreshGraphList();
  }, [refreshGraphList]);

  return (
    <div className="graph-editor">
      <div className="g-toolbar">
        <button className="g-run" disabled={!robot} onClick={() => void runGraph()}>
          ▶ Run
        </button>
        <button
          className="g-run-live"
          disabled={!robot}
          title="Run, then stream the Scope traces in for a live feel (build-checked only)"
          onClick={() => void runGraphLive()}
        >
          ⦿ Run Live
        </button>
        <button className="g-validate" disabled={!robot} onClick={() => void validateGraph()}>
          ✓ Validate
        </button>
        <span className="g-sep" />
        <input
          className="g-name"
          placeholder="graph name"
          value={graphName}
          onChange={(e) => setGraphName(e.target.value)}
        />
        <button
          disabled={!graphName.trim()}
          onClick={() => void saveGraph(graphName.trim())}
        >
          Save
        </button>
        <select
          className="g-load"
          value={loadSel}
          onChange={(e) => setLoadSel(e.target.value)}
        >
          <option value="">load…</option>
          {saved.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
        <button disabled={!loadSel} onClick={() => void loadGraph(loadSel)}>
          Load
        </button>
        <button title="refresh saved list" onClick={() => void refreshGraphList()}>
          ⟳
        </button>
      </div>
      {banner && <div className="g-banner">{banner}</div>}
      <div className="g-canvas">
        <Palette />
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          colorMode="dark"
          fitView
          minZoom={0.2}
          maxZoom={2.5}
          defaultEdgeOptions={{ animated: false }}
          proOptions={{ hideAttribution: false }}
        >
          <Background
            variant={BackgroundVariant.Dots}
            gap={22}
            size={1}
            color="rgba(255,255,255,0.06)"
          />
          <Controls showInteractive={false} />
          <MiniMap
            pannable
            zoomable
            bgColor="#0b0d10"
            nodeColor="#22262e"
            nodeStrokeColor="rgba(255,255,255,0.12)"
            maskColor="rgba(11,13,16,0.66)"
          />
        </ReactFlow>
      </div>
    </div>
  );
}
