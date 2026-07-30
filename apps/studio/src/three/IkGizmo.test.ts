// Unit test for the gizmo's drag-release path.
//
// The component itself needs a WebGL canvas and a react-three-fiber tree, so
// what is covered here is the decision the fix is made of: releaseDrag is the
// whole body of `onDragEnd`, and the same call is what the component's hide/
// unmount effect runs when a drag is interrupted instead. The wiring (an effect
// keyed on the `hidden` condition) is by inspection; the invariant it restores
// is not.

import { describe, it, expect, vi } from "vitest";
import { releaseDrag } from "./IkGizmo";
import type { DragRefs } from "./IkGizmo";

/** Drag refs as the component holds them mid-drag. */
function midDrag(rafHandle = 0): DragRefs {
  return {
    raf: { current: rafHandle },
    dragging: { current: true },
    held: { current: {} as never },
  };
}

describe("releaseDrag — a drag that never gets its onDragEnd", () => {
  it("gives OrbitControls back and forgets the drag", () => {
    const controls = { enabled: false }; // onDragStart disabled it
    const refs = midDrag();

    releaseDrag(refs, controls);

    expect(controls.enabled).toBe(true);
    expect(refs.dragging.current).toBe(false);
    expect(refs.held.current).toBeNull();
  });

  it("cancels the queued drag-follow frame so no target lands after the handle is gone", () => {
    const cancel = vi.spyOn(globalThis, "cancelAnimationFrame");
    const refs = midDrag(7);

    releaseDrag(refs, { enabled: false });

    expect(cancel).toHaveBeenCalledWith(7);
    expect(refs.raf.current).toBe(0);
    cancel.mockRestore();
  });

  it("survives a scene with no controls at all", () => {
    const refs = midDrag();
    expect(() => releaseDrag(refs, undefined)).not.toThrow();
    expect(refs.dragging.current).toBe(false);
  });

  it("leaves controls alone when no drag of ours is up", () => {
    // the handle can hide with nothing being dragged, and something else may
    // legitimately have the camera disabled — that is not ours to re-enable
    const controls = { enabled: false };
    const refs: DragRefs = {
      raf: { current: 0 },
      dragging: { current: false },
      held: { current: null },
    };

    releaseDrag(refs, controls);

    expect(controls.enabled).toBe(false);
  });

  it("is idempotent — a hide right after a normal drag end changes nothing", () => {
    const controls = { enabled: false };
    const refs = midDrag(3);

    releaseDrag(refs, controls); // onDragEnd
    controls.enabled = false; // something else takes the camera afterwards
    releaseDrag(refs, controls); // the hide effect, on the same refs

    expect(controls.enabled).toBe(false);
    expect(refs.raf.current).toBe(0);
  });
});
