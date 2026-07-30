// Headless unit tests for the pure live-session helpers (src/sim/live.ts):
// the adoption of a started session, the two-stage event filter (stash/drop at
// arrival, apply/keep/drop at flush time) that keeps a ~60 Hz stream down to
// one set() per frame, the two store patches, and the recording slice (A3):
// a dataset that outlives its takes, and the stream transition that is the
// ONLY notice a take died without a command behind it.
//
// The interesting cases are all ORDERING ones: a state event can arrive before
// the live_start reply that names its session, and an "ended" of a superseded
// session can arrive after its successor is already running.

import { describe, it, expect } from "vitest";
import {
  classifyLiveState,
  gripperControl,
  gripperSeated,
  liveEndedPatch,
  liveFlushFate,
  liveInfoFromStarted,
  liveRecAfterStop,
  liveRecFromStarted,
  liveRecPatch,
  liveStatePatch,
  reconcileGripper,
  NO_GRIPPER_TITLE,
} from "./live";
import type {
  GripperInfo,
  LiveInfo,
  LiveRecInfo,
  LiveStartedDto,
  LiveStateEvent,
} from "./live";
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

/** A gripper channel as `live_start` reports it: the jaw joint's limits inset
 *  2%, closing toward `lo` (the usual convention). */
function mockChannel(over: Partial<GripperInfo> = {}): GripperInfo {
  return { joint: "gripper", index: 1, openTarget: 0.04, closedTarget: 0.0, ...over };
}

function mockStarted(over: Partial<LiveStartedDto> = {}): LiveStartedDto {
  return {
    sessionId: 1,
    engine: "mujoco",
    h: 0.002,
    emitHz: 59.5,
    ndof: 2,
    props: [mockPropTrack("box0")],
    gripper: null,
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
    gripper: null,
    gripperState: null,
    held: null,
    success: null,
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
    recording: false,
    recFrames: 0,
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

  it("adopts the gripper channel and starts with nothing commanded or held", () => {
    const ch = mockChannel();
    const info = liveInfoFromStarted(mockStarted({ gripper: ch }));
    expect(info.gripper).toBe(ch);
    expect(info.gripperState).toBeNull();
    expect(info.held).toBeNull();
  });

  it("reads a session with no gripper (or an older backend) as no channel", () => {
    expect(liveInfoFromStarted(mockStarted({ gripper: null })).gripper).toBeNull();
    const { gripper: _dropped, ...noField } = mockStarted();
    expect(liveInfoFromStarted(noField).gripper).toBeNull();
  });

  it("starts with no verdict (the first state event brings one, or never does)", () => {
    expect(liveInfoFromStarted(mockStarted()).success).toBeNull();
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

  it("carries the streamed success verdict, per instant and unlatched", () => {
    const held = liveStatePatch(mockInfo(), mockEvent({ success: true }));
    expect(held.live.success).toBe(true);
    // a session that succeeded and then lost it reads false again — the badge
    // shows what the stream says, it never remembers a verdict
    expect(liveStatePatch(held.live, mockEvent({ success: false })).live.success).toBe(false);
  });

  it("reads an absent (or null) verdict as nobody scoring the session", () => {
    expect(liveStatePatch(mockInfo({ success: true }), mockEvent()).live.success).toBeNull();
    expect(liveStatePatch(mockInfo(), mockEvent({ success: null })).live.success).toBeNull();
  });

  it("carries the streamed gripper command + held prop into the slice", () => {
    const ch = mockChannel();
    const prev = mockInfo({ gripper: ch });
    const ev = mockEvent({ gripper: { closed: true, q: 0.031 }, held: "box0" });
    const patch = liveStatePatch(prev, ev);
    // the CHANNEL is static and rides through; only the command/measurement move
    expect(patch.live.gripper).toBe(ch);
    expect(patch.live.gripperState).toEqual({ closed: true, q: 0.031 });
    expect(patch.live.held).toBe("box0");
  });

  it("reports a released prop and a re-opened jaw, not the last state that was true", () => {
    const prev = mockInfo({
      gripper: mockChannel(),
      gripperState: { closed: true, q: 0.031 },
      held: "box0",
    });
    const ev = mockEvent({ gripper: { closed: false, q: 0.04 }, held: null });
    const patch = liveStatePatch(prev, ev);
    expect(patch.live.gripperState).toEqual({ closed: false, q: 0.04 });
    expect(patch.live.held).toBeNull();
  });

  it("reads an event with no gripper fields (builtin, older backend) as neither", () => {
    const prev = mockInfo({ gripperState: { closed: true, q: 0 }, held: "box0" });
    const patch = liveStatePatch(prev, mockEvent());
    expect(patch.live.gripperState).toBeNull();
    expect(patch.live.held).toBeNull();
  });
});

// ---- gripper + grasp (B): the command, the measurement, and the button ----

describe("gripperControl — what the panel's gripper button says and does", () => {
  it("offers the closing half of the toggle while the jaw is commanded open", () => {
    const c = gripperControl(mockChannel(), { closed: false, q: 0.04 });
    expect(c.label).toBe("gripper: open ▸ close");
    expect(c.closed).toBe(false);
    expect(c.disabled).toBe(false);
    expect(c.title).toContain("close the gripper");
    expect(c.title).toContain("gripper"); // and names the joint it drives
  });

  it("offers the opening half once it is commanded closed", () => {
    const c = gripperControl(mockChannel({ joint: "jaw" }), { closed: true, q: 0.031 });
    expect(c.label).toBe("gripper: closed ▸ open");
    expect(c.closed).toBe(true);
    expect(c.title).toContain("open the gripper");
    expect(c.title).toContain("jaw");
  });

  it("reads a jaw squeezing a prop as CLOSED — the flag is the command, not the position", () => {
    // short of closedTarget because something is in the way; the button must
    // still offer "open", or the human could not let go
    expect(gripperControl(mockChannel(), { closed: true, q: 0.028 }).label).toBe(
      "gripper: closed ▸ open",
    );
  });

  it("starts open before the first state event lands", () => {
    expect(gripperControl(mockChannel(), null).closed).toBe(false);
  });

  it("disables itself and says WHY on a robot with no gripper joint", () => {
    const c = gripperControl(null, null);
    expect(c.disabled).toBe(true);
    expect(c.title).toBe(NO_GRIPPER_TITLE);
    expect(c.title).toMatch(/gripper.*finger.*jaw|override/);
  });
});

describe("gripperSeated — did the jaw reach what was commanded?", () => {
  const ch = mockChannel(); // open 0.04 → closed 0.0, span 0.04

  it("is seated at (and within a tenth of the span of) the commanded target", () => {
    expect(gripperSeated(ch, { closed: true, q: 0 })).toBe(true);
    expect(gripperSeated(ch, { closed: true, q: 0.004 })).toBe(true);
    expect(gripperSeated(ch, { closed: false, q: 0.04 })).toBe(true);
  });

  it("is NOT seated while the jaw is stopped short (travelling, or on a prop)", () => {
    expect(gripperSeated(ch, { closed: true, q: 0.028 })).toBe(false);
    expect(gripperSeated(ch, { closed: false, q: 0.01 })).toBe(false);
  });

  it("claims nothing with no measurement at all", () => {
    expect(gripperSeated(ch, null)).toBe(false);
    expect(gripperSeated(ch, { closed: true, q: NaN })).toBe(false);
  });
});

describe("reconcileGripper — our command outranks an event that predates it", () => {
  it("passes the stream through when nothing is pending", () => {
    const ev = { closed: true, q: 0.03 };
    const out = reconcileGripper(ev, null);
    expect(out.state).toBe(ev);
    expect(out.settled).toBe(false);
  });

  it("settles the moment the stream reports the intent we sent", () => {
    const ev = { closed: true, q: 0.03 };
    const out = reconcileGripper(ev, true);
    expect(out.state).toBe(ev);
    expect(out.settled).toBe(true);
  });

  it("holds our intent over an older event, keeping its measurement", () => {
    // the event was emitted before live_gripper landed: it still says open
    const out = reconcileGripper({ closed: false, q: 0.04 }, true);
    expect(out.state).toEqual({ closed: true, q: 0.04 });
    expect(out.settled).toBe(false);
  });

  it("has nothing to reconcile on a session with no gripper channel", () => {
    expect(reconcileGripper(null, true)).toEqual({ state: null, settled: false });
    expect(reconcileGripper(null, null)).toEqual({ state: null, settled: false });
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

// ---- recording (A3): the dataset outlives each take ----

function mockRec(over: Partial<LiveRecInfo> = {}): LiveRecInfo {
  return {
    root: "/tmp/teleop_panda",
    fps: 50,
    recording: false,
    task: null,
    frames: 0,
    episodesSaved: 0,
    ...over,
  };
}

describe("liveRecFromStarted", () => {
  it("adopts the root the BACKEND named, not the one we asked with", () => {
    const rec = liveRecFromStarted(
      null,
      { root: "/data/teleop", fps: 25, recordEvery: 20, episodeIndex: 0 },
      "pick the cube",
    );
    expect(rec).toEqual({
      root: "/data/teleop",
      fps: 25,
      recording: true,
      task: "pick the cube",
      frames: 0,
      episodesSaved: 0,
    });
  });

  it("keeps the episode count when the take continues the open dataset", () => {
    const prev = mockRec({ root: "/data/teleop", episodesSaved: 3 });
    const rec = liveRecFromStarted(
      prev,
      { root: "/data/teleop", fps: 50, recordEvery: 10, episodeIndex: 3 },
      "again",
    );
    expect(rec.episodesSaved).toBe(3);
    expect(rec.recording).toBe(true);
  });

  it("restarts the count when the reply names a different dataset", () => {
    const prev = mockRec({ root: "/data/old", episodesSaved: 3 });
    const rec = liveRecFromStarted(
      prev,
      { root: "/data/new", fps: 50, recordEvery: 10, episodeIndex: 0 },
      "t",
    );
    expect(rec.episodesSaved).toBe(0);
  });
});

describe("liveRecAfterStop", () => {
  it("counts a saved take and ends it", () => {
    const rec = liveRecAfterStop(mockRec({ recording: true, task: "t", frames: 120 }), {
      saved: true,
      episodeIndex: 0,
      frames: 120,
    });
    expect(rec).toMatchObject({ recording: false, task: null, frames: 0, episodesSaved: 1 });
  });

  it("ends a discarded take without counting it", () => {
    const rec = liveRecAfterStop(
      mockRec({ recording: true, task: "t", frames: 40, episodesSaved: 2 }),
      { saved: false, episodeIndex: null, frames: 40 },
    );
    expect(rec).toMatchObject({ recording: false, episodesSaved: 2 });
  });
});

describe("liveRecPatch — the stream's view of the take", () => {
  it("says nothing when no dataset is open", () => {
    expect(liveRecPatch(null, mockEvent({ recording: true, recFrames: 9 }), true)).toEqual({
      rec: null,
      discarded: false,
    });
  });

  it("counts the take's frames as they stream in", () => {
    const prev = mockRec({ recording: true, task: "t", frames: 10 });
    const ev = mockEvent({ recording: true, recFrames: 11 });
    const { rec, discarded } = liveRecPatch(prev, ev, true);
    expect(rec).toMatchObject({ recording: true, task: "t", frames: 11 });
    expect(discarded).toBe(false);
  });

  it("returns the SAME slice when nothing moved (no repaint per streamed frame)", () => {
    const prev = mockRec({ recording: true, task: "t", frames: 11 });
    expect(liveRecPatch(prev, mockEvent({ recording: true, recFrames: 11 }), true).rec).toBe(prev);
  });

  it("reads an armed true→false transition as a take that died on its own", () => {
    const prev = mockRec({ recording: true, task: "t", frames: 11 });
    const { rec, discarded } = liveRecPatch(prev, mockEvent({ recording: false }), true);
    expect(rec).toMatchObject({ recording: false, task: null, frames: 0 });
    expect(discarded).toBe(true);
  });

  it("ignores a not-yet-armed false: it is an older event crossing the start reply", () => {
    const prev = mockRec({ recording: true, task: "t", frames: 0 });
    const { rec, discarded } = liveRecPatch(prev, mockEvent({ recording: false }), false);
    expect(rec).toBe(prev);
    expect(discarded).toBe(false);
  });

  it("stays quiet between takes (a stopped take does not discard twice)", () => {
    const prev = mockRec({ episodesSaved: 1 });
    const { rec, discarded } = liveRecPatch(prev, mockEvent({ recording: false }), true);
    expect(rec).toBe(prev);
    expect(discarded).toBe(false);
  });
});
