import { useState } from "react";
import { useStore } from "../store";

export function PosePanel() {
  const robot = useStore((s) => s.robot);
  const poses = useStore((s) => s.poses);
  const savePose = useStore((s) => s.savePose);
  const deletePose = useStore((s) => s.deletePose);
  const planMoveToPose = useStore((s) => s.planMoveToPose);
  const [name, setName] = useState("");
  if (!robot) return null;
  return (
    <aside className="pose-panel">
      <div className="pose-title">Poses</div>
      <div className="pose-add">
        <input value={name} placeholder="name" onChange={(e) => setName(e.target.value)} />
        <button
          disabled={!name}
          onClick={() => {
            void savePose(name);
            setName("");
          }}
        >
          Save current
        </button>
      </div>
      <ul>
        {poses.map((p) => (
          <li key={p.name}>
            <span>{p.name}</span>
            <button onClick={() => void planMoveToPose(p.name)}>Move to</button>
            <button onClick={() => void deletePose(p.name)}>×</button>
          </li>
        ))}
      </ul>
    </aside>
  );
}
