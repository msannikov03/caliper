// Headless unit tests for the pure task-artifact helpers (src/sim/task.ts):
// what a loaded task contributes to a `live_start` request, what rates the
// record panel may offer once a task declares one, and what the verdict badge
// says for each of the three things `live://state.success` can be.
//
// The load-bearing case is the THIRD state: `null` means nobody is scoring the
// session, which must never render as "not yet".

import { describe, it, expect } from "vitest";
import { recFpsChoices, successBadge, taskInfoFromDto, taskLiveStartFields } from "./task";
import type { TaskInfo, TaskSpecDto } from "./task";

const PREDICATE = { kind: "lifted", prop: "cube", height: 0.05, ref: "initial" };

function mockDto(over: Partial<TaskSpecDto> = {}): TaskSpecDto {
  return {
    name: "lift-cube",
    robotPath: "/fx/robots/gripper_arm.urdf",
    q0: [0, 0, 0.02],
    ground: 0,
    props: [
      {
        name: "cube",
        kind: "box",
        halfExtents: [0.05, 0.05, 0.05],
        pos: [0, 0, 0.05],
        mass: 0.05,
        material: "wood",
      },
    ],
    zones: [{ name: "bin", center: [0.4, 0.2, 0.02], half: [0.05, 0.05, 0.02], rgba: null }],
    gripper: { joint: "gripper", closed: "lo" },
    success: PREDICATE,
    successDescription: "cube is lifted 0.05 m above its initial height",
    horizonS: 20,
    fps: 50,
    ...over,
  };
}

function mockTask(over: Partial<TaskInfo> = {}): TaskInfo {
  return { ...taskInfoFromDto(mockDto(), "/tasks/lift_cube.caliper-task.json"), ...over };
}

describe("taskInfoFromDto", () => {
  it("keeps the whole reply and remembers the file it came from", () => {
    const t = taskInfoFromDto(mockDto(), "/tasks/lift_cube.caliper-task.json");
    expect(t.path).toBe("/tasks/lift_cube.caliper-task.json");
    expect(t.name).toBe("lift-cube");
    expect(t.robotPath).toBe("/fx/robots/gripper_arm.urdf");
    expect(t.q0).toEqual([0, 0, 0.02]);
    expect(t.zones[0].name).toBe("bin");
    expect(t.success).toBe(PREDICATE); // the predicate is carried, never rebuilt
  });
});

describe("taskLiveStartFields", () => {
  it("sends nothing at all without a task", () => {
    expect(taskLiveStartFields(null)).toEqual({});
  });

  it("carries the ground, the gripper override and the predicate verbatim", () => {
    expect(taskLiveStartFields(mockTask({ ground: 0.15 }))).toEqual({
      ground: 0.15,
      gripperJoint: "gripper",
      gripperClosed: "lo",
      success: PREDICATE,
    });
  });

  it("omits each half of the gripper override the task leaves out", () => {
    expect(taskLiveStartFields(mockTask({ gripper: { joint: "jaw" } }))).toEqual({
      ground: 0,
      gripperJoint: "jaw",
      success: PREDICATE,
    });
    expect(taskLiveStartFields(mockTask({ gripper: { closed: "hi" } }))).toEqual({
      ground: 0,
      gripperClosed: "hi",
      success: PREDICATE,
    });
    expect(taskLiveStartFields(mockTask({ gripper: null }))).toEqual({
      ground: 0,
      success: PREDICATE,
    });
  });

  it("omits the predicate when the task declares none (no verdict is computed)", () => {
    const f = taskLiveStartFields(mockTask({ success: null, successDescription: null }));
    expect(f).toEqual({ ground: 0, gripperJoint: "gripper", gripperClosed: "lo" });
    expect("success" in f).toBe(false);
  });
});

describe("recFpsChoices", () => {
  const BASE = [25, 50, 100] as const;

  it("is the plain choice list without a task rate", () => {
    expect(recFpsChoices(BASE, null)).toEqual([25, 50, 100]);
    expect(recFpsChoices(BASE, undefined)).toEqual([25, 50, 100]);
  });

  it("does not duplicate a task rate the list already offers", () => {
    expect(recFpsChoices(BASE, 50)).toEqual([25, 50, 100]);
  });

  it("adds an off-list task rate in order (the task's rate must be selectable)", () => {
    expect(recFpsChoices(BASE, 30)).toEqual([25, 30, 50, 100]);
    expect(recFpsChoices(BASE, 200)).toEqual([25, 50, 100, 200]);
  });

  it("ignores a nonsense rate", () => {
    expect(recFpsChoices(BASE, 0)).toEqual([25, 50, 100]);
    expect(recFpsChoices(BASE, -5)).toEqual([25, 50, 100]);
    expect(recFpsChoices(BASE, NaN)).toEqual([25, 50, 100]);
  });
});

describe("successBadge", () => {
  const DESC = "cube is lifted 0.05 m above its initial height";

  it("renders nothing when nothing is scoring the session", () => {
    expect(successBadge(null, DESC)).toBeNull();
    expect(successBadge(undefined, DESC)).toBeNull();
  });

  it("fills the badge on a holding verdict", () => {
    const b = successBadge(true, DESC)!;
    expect(b.label).toBe("SUCCESS ✓");
    expect(b.className).toBe("badge success");
    expect(b.title).toBe(DESC);
  });

  it("stays muted while the verdict has not happened", () => {
    const b = successBadge(false, DESC)!;
    expect(b.label).toBe("success: not yet");
    expect(b.className).toBe("badge");
    expect(b.title).toBe(DESC);
  });

  it("explains itself when the task carries no description", () => {
    expect(successBadge(true, null)!.title).toBe("the success predicate holds");
    expect(successBadge(false, null)!.title).toBe("the success predicate does not hold yet");
  });
});
