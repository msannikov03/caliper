// Headless unit tests for the pure doctor-surface logic (src/doctor/doctor.ts):
// severity chip mapping, banner capping, the repair gate, the severity tally,
// and the finding→episode-row jump. Every helper gets a positive AND a
// negative case (finds/acts vs clean/no-op).

import { describe, it, expect } from "vitest";
import {
  canRepair,
  doctorSummary,
  findingEpisode,
  findingRefs,
  sevClass,
  sevLabel,
  visibleFindings,
} from "./doctor";
import type { DataDoctorFinding, DoctorFinding, DoctorReport } from "./doctor";

function assetFinding(p: Partial<DoctorFinding> = {}): DoctorFinding {
  return {
    code: "A001",
    severity: "error",
    message: "link `l1` has no <inertial>",
    fixHint: null,
    autoFixable: false,
    ...p,
  };
}

function dataFinding(p: Partial<DataDoctorFinding> = {}): DataDoctorFinding {
  return {
    code: "D001",
    severity: "warning",
    feature: "observation.state",
    episode: null,
    dof: null,
    message: "dof never moves",
    fixHint: "check the pipeline",
    ...p,
  };
}

function report(findings: DoctorFinding[]): DoctorReport {
  const n = (s: string) => findings.filter((f) => f.severity === s).length;
  return {
    findings,
    errors: n("error"),
    warnings: n("warn"),
    infos: n("info"),
    repair: null,
    after: null,
  };
}

describe("sevClass / sevLabel", () => {
  it("maps every severity of BOTH doctors to a distinct chip", () => {
    expect(sevClass("error")).toBe("sev-error");
    expect(sevClass("warn")).toBe("sev-warn");
    expect(sevClass("warning")).toBe("sev-warn"); // dataset spelling, same chip
    expect(sevClass("info")).toBe("sev-info");
    expect(sevLabel("error")).toBe("ERR");
    expect(sevLabel("warn")).toBe("WARN");
    expect(sevLabel("warning")).toBe("WARN");
    expect(sevLabel("info")).toBe("INFO");
  });
});

describe("visibleFindings", () => {
  const five = [1, 2, 3, 4, 5];

  it("caps an over-long list and counts the hidden tail", () => {
    const { shown, hidden } = visibleFindings(five, 3);
    expect(shown).toEqual([1, 2, 3]); // prefix = most severe (engine pre-sorts)
    expect(hidden).toBe(2);
  });

  it("passes a short list through untouched (negative)", () => {
    expect(visibleFindings(five, 5)).toEqual({ shown: five, hidden: 0 });
    expect(visibleFindings([], 3)).toEqual({ shown: [], hidden: 0 });
  });

  it("treats a non-positive cap as 'show everything'", () => {
    expect(visibleFindings(five, 0)).toEqual({ shown: five, hidden: 0 });
    expect(visibleFindings(five, -1)).toEqual({ shown: five, hidden: 0 });
  });

  it("never aliases the input array", () => {
    const { shown } = visibleFindings(five, 5);
    expect(shown).not.toBe(five);
  });
});

describe("canRepair", () => {
  it("is true when ANY finding is auto-fixable (positive)", () => {
    const r = report([assetFinding(), assetFinding({ code: "A009", autoFixable: true })]);
    expect(canRepair(r)).toBe(true);
  });

  it("is false with no auto-fixable findings, an empty report, or null", () => {
    expect(canRepair(report([assetFinding()]))).toBe(false);
    expect(canRepair(report([]))).toBe(false);
    expect(canRepair(null)).toBe(false);
  });
});

describe("doctorSummary", () => {
  it("tallies with correct plurals and drops zero counts", () => {
    expect(doctorSummary(2, 1, 0)).toBe("2 errors · 1 warning");
    expect(doctorSummary(1, 0, 3)).toBe("1 error · 3 infos");
    expect(doctorSummary(0, 0, 1)).toBe("1 info");
  });

  it("reports a clean bill of health (negative)", () => {
    expect(doctorSummary(0, 0, 0)).toBe("no findings");
  });
});

describe("findingEpisode", () => {
  it("returns a valid episode row (positive)", () => {
    expect(findingEpisode(dataFinding({ episode: 3 }), 6)).toBe(3);
    expect(findingEpisode(dataFinding({ episode: 0 }), 1)).toBe(0);
  });

  it("rejects missing, out-of-range, and non-integer refs (negative)", () => {
    expect(findingEpisode(dataFinding(), 6)).toBeNull(); // whole-dataset finding
    expect(findingEpisode(dataFinding({ episode: 6 }), 6)).toBeNull();
    expect(findingEpisode(dataFinding({ episode: -1 }), 6)).toBeNull();
    expect(findingEpisode(dataFinding({ episode: 2.5 }), 6)).toBeNull();
    expect(findingEpisode(dataFinding({ episode: 2 }), 0)).toBeNull(); // empty table
  });
});

describe("findingRefs", () => {
  it("renders every present ref, in ep/dof/feature order (positive)", () => {
    const f = dataFinding({ episode: 4, dof: 1, feature: "action" });
    expect(findingRefs(f)).toEqual(["ep 4", "dof 1", "action"]);
  });

  it("skips absent refs, including dof 0 kept and episode 0 kept", () => {
    expect(findingRefs(dataFinding({ episode: 0, dof: 0, feature: null }))).toEqual([
      "ep 0",
      "dof 0",
    ]);
  });

  it("is empty for a dataset-wide finding (negative)", () => {
    expect(findingRefs(dataFinding({ feature: null }))).toEqual([]);
  });
});
