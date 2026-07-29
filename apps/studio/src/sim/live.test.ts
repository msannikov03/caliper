// Headless unit tests for the pure live-session helpers (src/sim/live.ts):
// the adoption of a started session, the two-stage event filter (stash/drop at
// arrival, apply/keep/drop at flush time) that keeps a ~60 Hz stream down to
// one set() per frame, and the two store patches.
//
// The interesting cases are all ORDERING ones: a state event can arrive before
// the live_start reply that names its session, and an "ended" of a superseded
// session can arrive after its successor is already running.

import { describe, it, expect } from "vitest";
import {
  classifyLiveState,
  liveEndedPatch,
  liveFlushFate,
  liveInfoFromStarted,
  liveStatePatch,
} from "./live";
import type { LiveInfo, LiveStartedDto, LiveStateEvent } from "./live";
import type { PropTrack } from "./props";

/** Static prop shape row as `live_start` returns it: no baked frames. */
function mockPropTrack(name: string): PropTrack {
  return {
    name,
    kind: "box",
    halfExtents: [0.02, 0.02, 0.02],
    radius: null,
    length: null,
    rgba: null,
    frames: [],
  };
}

function mockStarted(over: Partial<LiveStartedDto> = {}): LiveStartedDto {
  return {
    sessionId: 1,
    engine: "mujoco",
    h: 0.002,
    emitHz: 59.5,
    ndof: 2,
    props: [mockPropTrack("box0")],
    ...over,
  };
}

function mockInfo(over: Partial<LiveInfo> = {}): LiveInfo {
  return {
    sessionId: 1,
    engine: "mujoco",
    paused: false,
    t: 0.5,
    tick: 250,
    ncon: 0,
    props: [mockPropTrack("box0")],
    ...over,
  };
}

function mockEvent(over: Partial<LiveStateEvent> = {}): LiveStateEvent {
  return {
    sessionId: 1,
    tick: 300,
    t: 0.6,
    q: [0.1, 0.2],
    qd: [0, 0],
    frames: [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]],
    tip: [0.3, 0, 0.2],
    ncon: 2,
    props: [[0, 0, 0.05, 1, 0, 0, 0]],
    paused: false,
    target: [0, 0],
    ...over,
  };
}

describe("liveInfoFromStarted", () => {
  it("adopts identity + static props and zeroes the streamed scalars", () => {
    const info = liveInfoFromStarted(mockStarted({ sessionId: 7, engine: "builtin" }));
    expect(info.sessionId).toBe(7);
    expect(info.engine).toBe("builtin");
    expect(info.paused).toBe(false);
    expect(info.t).toBe(0);
    expect(info.tick).toBe(0);
    expect(info.ncon).toBe(0);
  });

  it("carries the static prop rows through by reference (they never change)", () => {
    const dto = mockStarted();
    expect(liveInfoFromStarted(dto).props).toBe(dto.props);
  });
});

describe("classifyLiveState — arrival filter", () => {
  it("stashes an event of the current session", () => {
    expect(classifyLiveState(mockInfo({ sessionId: 4 }), 4, 0)).toBe("stash");
  });

  it("drops an event of an older session (a superseded stream still draining)", () => {
    expect(classifyLiveState(mockInfo({ sessionId: 4 }), 3, 0)).toBe("drop");
  });

  it("stashes an event of a NEWER session while its start is being adopted", () => {
    expect(classifyLiveState(mockInfo({ sessionId: 4 }), 5, 1)).toBe("stash");
    // …and even with no start in flight: the newer session is authoritative
    expect(classifyLiveState(mockInfo({ sessionId: 4 }), 5, 0)).toBe("stash");
  });

  it("stashes with no session yet WHEN a start is in flight (event beat the reply)", () => {
    expect(classifyLiveState(null, 1, 1)).toBe("stash");
  });

  it("drops with no session and no start in flight (nothing wants this)", () => {
    expect(classifyLiveState(null, 1, 0)).toBe("drop");
  });
});

describe("liveFlushFate — flush filter", () => {
  it("drops when nothing is stashed", () => {
    expect(liveFlushFate(mockInfo(), null)).toBe("drop");
    expect(liveFlushFate(null, null)).toBe("drop");
  });

  it("applies an event of the adopted session", () => {
    expect(liveFlushFate(mockInfo({ sessionId: 2 }), mockEvent({ sessionId: 2 }))).toBe("apply");
  });

  it("keeps an early event: the session it names is not adopted yet", () => {
    expect(liveFlushFate(null, mockEvent())).toBe("keep");
    expect(liveFlushFate(mockInfo({ sessionId: 2 }), mockEvent({ sessionId: 3 }))).toBe("keep");
  });

  it("drops a stale event whose session we already moved past", () => {
    expect(liveFlushFate(mockInfo({ sessionId: 3 }), mockEvent({ sessionId: 2 }))).toBe("drop");
  });
});

describe("liveStatePatch", () => {
  it("routes pose + frames through the fields playback already drives", () => {
    const ev = mockEvent();
    const patch = liveStatePatch(mockInfo(), ev);
    expect(patch.q).toBe(ev.q);
    expect(patch.frames).toBe(ev.frames);
    expect(patch.livePropPoses).toBe(ev.props);
  });

  it("advances only the streamed scalars and preserves session identity", () => {
    const prev = mockInfo({ sessionId: 9, engine: "builtin" });
    const patch = liveStatePatch(prev, mockEvent({ t: 1.25, tick: 600, ncon: 3, paused: true }));
    expect(patch.live.sessionId).toBe(9);
    expect(patch.live.engine).toBe("builtin");
    expect(patch.live.props).toBe(prev.props);
    expect(patch.live).toMatchObject({ t: 1.25, tick: 600, ncon: 3, paused: true });
  });

  it("does not mutate the previous slice", () => {
    const prev = mockInfo({ t: 0.5, ncon: 0 });
    liveStatePatch(prev, mockEvent({ t: 9, ncon: 5 }));
    expect(prev.t).toBe(0.5);
    expect(prev.ncon).toBe(0);
  });
});

describe("liveEndedPatch", () => {
  it("clears the slice silently for the two benign ends", () => {
    for (const reason of ["stopped", "superseded"]) {
      const patch = liveEndedPatch(reason);
      expect(patch).toEqual({ live: null, livePropPoses: [] });
      expect(patch.error).toBeUndefined();
    }
  });

  it("clears the slice AND surfaces the banner for a backend error", () => {
    const patch = liveEndedPatch("error: mujoco step diverged");
    expect(patch.live).toBeNull();
    expect(patch.livePropPoses).toEqual([]);
    expect(patch.error).toBe("live sim ended — error: mujoco step diverged");
  });

  it("treats an unrecognized reason as loud, not benign", () => {
    expect(liveEndedPatch("stopped ").error).toBe("live sim ended — stopped ");
  });
});
