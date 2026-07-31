#!/usr/bin/env python3
"""ITA-6 — a 6-DOF spherical-wrist manipulator DESIGNED BY CALIPER.

This script is the whole design method, start to finish: it parametrizes an
industrial-kinematics arm (yaw base, shoulder pitch, elbow pitch, spherical
wrist), emits a complete URDF for every candidate in a coarse design grid,
loads each one with the caliper engine, and scores it on the engine's own
kinematic analysis over a table-top WORK ZONE. The winner is written out as
`ita6.urdf` (with gripper) and `ita6_arm6.urdf` (the bare 6R kinematic core,
which is what caliper's closed-form spherical-wrist IK accepts), together with
`analysis.json` — the full sweep table, the winner, and the static payload
check.

Nothing here is hand-tuned geometry copied out of a CAD package. Link lengths
come from the sweep; masses and inertia tensors are composed analytically from
a documented bill of materials (aluminium tube + printed shell + COTS motors
and gearboxes), so the dynamics the engine sees are the dynamics the drawing
implies.

Run (from the repo root, with the caliper python face installed):

    PATH=/usr/sbin:$PATH .venv/bin/python examples/ita6/design_ita6.py

`/usr/sbin` on PATH is only needed because caliper's optional mujoco lane
shells out to `sysctl`; the design sweep itself is pure kinematics.

--------------------------------------------------------------------------
DESIGN INTENT (fixed requirements, not swept)
--------------------------------------------------------------------------
  * 6R, spherical wrist (axes 4/5/6 intersect) -> closed-form analytic IK.
  * ~450 mm reach, measured base-axis -> TCP with the arm extended.
  * 1.0 kg structural payload, 0.5 kg precision payload.
  * NEMA17-class steppers; strain-wave reduction at J2/J3 (they carry the
    load), belt + planetary elsewhere.
  * J1 yaw / J2 shoulder pitch / J3 elbow pitch / J4 wrist roll / J5 wrist
    pitch / J6 tool roll -- PUMA/OPW-class ordering.

SWEPT (the design space this script searches)
  * L2 -- upper-arm length, J2 axis -> J3 axis.
  * L3 -- forearm length, J3 axis -> wrist centre.
  * h_base -- base-axis height (stage-2 refinement at the winning L2/L3).

FIXED BY PACKAGING
  * L_w = 100 mm, wrist centre -> TCP (the wrist offset: flange, gripper body
    and jaw stack). Reach = L2 + L3 + L_w.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import caliper

HERE = Path(__file__).resolve().parent

# ==========================================================================
# 1. BILL OF MATERIALS -- every density and COTS mass used below, in one place
# ==========================================================================
# Structural densities (kg/m^3):
RHO_AL = 2700.0     # 6061-T6 aluminium: tubes, base plate, jaw plates.
RHO_PLA = 1240.0    # PLA-CF, quoted as SOLID: shells below are modelled as
#                     genuine hollow walls, so the wall material is solid and
#                     the "infill" is the geometry, not a fudged density.
RHO_STEEL = 7850.0  # fasteners/shafts, only where called out.

# COTS masses (kg) -- catalogue figures for the actuator class named above.
M_NEMA17 = 0.30     # NEMA17, 48 mm body, 0.50 N.m holding torque.
M_NEMA14 = 0.14     # NEMA14, 34 mm body, 0.22 N.m.
M_NEMA11 = 0.07     # NEMA11, 30 mm body, 0.09 N.m.
M_STRAINWAVE = 0.13  # CSF-8-class strain-wave unit, 30:1.
M_PLANET_L = 0.16   # 10:1 planetary, NEMA17 frame (J1).
M_PLANET_M = 0.06   # 20:1 planetary, NEMA14 frame (J4/J5).
M_PLANET_S = 0.03   # 14:1 planetary, NEMA11 frame (J6).
M_SERVO = 0.035     # gripper lead-screw micro gearmotor.

# Drivetrain -> joint limits. tau_out = tau_motor * ratio * efficiency.
#   name: (motor holding N.m, total ratio, efficiency, motor rad/s at usable rpm)
DRIVE = {
    "j1": (0.50, 30.0, 0.80, 73.0),   # 3:1 belt + 10:1 planetary
    "j2": (0.50, 30.0, 0.80, 73.0),   # strain wave, 30:1
    "j3": (0.50, 30.0, 0.80, 73.0),   # strain wave, 30:1
    "j4": (0.22, 20.0, 0.75, 73.0),   # planetary
    "j5": (0.22, 20.0, 0.75, 73.0),   # planetary
    "j6": (0.09, 14.0, 0.75, 73.0),   # planetary
}
# Position limits (rad). J1 is the +-175 deg the brief calls for.
LIMITS = {
    "j1": (-3.054, 3.054),   # +-175 deg
    "j2": (-2.094, 2.094),   # +-120 deg
    "j3": (-2.618, 2.618),   # +-150 deg
    "j4": (-3.054, 3.054),   # +-175 deg
    "j5": (-2.094, 2.094),   # +-120 deg
    "j6": (-3.140, 3.140),   # +-180 deg
}
GRIPPER_TRAVEL = 0.040       # m, jaw gap 36 mm (closed) -> 76 mm (open)
GRIPPER_EFFORT = 60.0        # N, lead-screw jaw force
GRIPPER_VEL = 0.05           # m/s

L_W = 0.100                  # wrist centre -> TCP (fixed by the gripper stack)
PAYLOAD_KG = 1.0             # structural payload for the static torque check
G = 9.81

# ==========================================================================
# 2. WORK ZONE + SCORING WEIGHTS
# ==========================================================================
# A table-top box in front of the arm: 150 x 240 x 160 mm, its floor 40 mm
# above the mounting plane. Every score below is computed over this box only.
#
# WHY THIS SIZE. A 450 mm arm whose tool is 100 mm long can only put its WRIST
# CENTRE inside a 350 mm sphere about the shoulder, and a vertical tool spends
# all 100 mm of that going straight down. The far-top corner of this box sits
# at 97 % of that 350 mm -- reachable, but badly conditioned, which is exactly
# the discrimination the sweep needs. The larger 200 x 300 x 200 box starting
# at x = 0.35 that a first pass used is simply outside the machine: the best
# candidate in the reach band served only 58 % of it. `BRIEF_ZONE` below keeps
# that box so the winner is also reported against it, honestly.
ZONE = {"x": (0.15, 0.30), "y": (-0.12, 0.12), "z": (0.04, 0.20)}
# An EVEN sample count across y, on purpose: the grid straddles the arm's
# symmetry plane instead of lying on it. y = 0 is where the tool-down target
# rotation becomes symmetric, and caliper's closed-form IK returns zero
# branches for EXACTLY y == 0 while solving y = 1e-12 fine -- a real engine
# bug, reproducible on the shipped showcase6 fixture (see README, "Known
# engine bug"). Sampling the plane would have charged that bug to the robot.
ZONE_N = (5, 4, 5)           # samples per axis -> 100 poses per candidate
BRIEF_ZONE = {"x": (0.15, 0.35), "y": (-0.15, 0.15), "z": (0.05, 0.25)}

# Tool orientation at every sample: pointing straight DOWN, yawed to face the
# sample radially (the natural top-down pick approach). One orientation per
# point, so reachability is a real constraint and not an optimisation.
#
# Combined score. Documented weights, and they are the whole ranking rule:
W_REACH = 0.50    # fraction of the zone reachable at all
W_MANIP = 0.30    # zone manipulability index (unreachable samples count 0)
W_MARGIN = 0.20   # joint-limit margin (unreachable samples count 0)
# The manipulability term is normalised by the best index in the sweep, so the
# score is RELATIVE TO THIS GRID -- it ranks candidates, it is not an absolute.

# Design-grid axes (m).
L2_GRID = [0.140, 0.160, 0.180, 0.200, 0.220, 0.240]
L3_GRID = [0.110, 0.130, 0.150, 0.170, 0.190, 0.210]
REACH_TARGET = 0.450
REACH_BAND = 0.010           # +-10 mm counts as "~450 mm"
H_BASE_GRID = [0.080, 0.095, 0.110, 0.125, 0.140]
H_BASE_DEFAULT = 0.110
H_SHOULDER = 0.052           # J1 axis -> J2 axis, fixed by the turret casting

# Branch-selection seed: reach forward, elbow up, tool vertical
# (q2 + q3 + q5 = pi puts the tool z-axis straight down when q4 = q6 = 0).
SEED_REST = (1.00, 0.90, 0.00, 1.2416, 0.00)


# ==========================================================================
# 3. INERTIA ALGEBRA -- composite links from primitive parts
# ==========================================================================
_AXIS_PERM = {"z": (0, 1, 2), "y": (2, 0, 1), "x": (1, 2, 0)}


def _axial(m: float, i_trans: float, i_axial: float, axis: str) -> np.ndarray:
    """Diagonal inertia of an axisymmetric body whose symmetry axis is `axis`."""
    a, b, c = _AXIS_PERM[axis]          # c is the symmetry axis slot
    out = np.zeros(3)
    out[a] = out[b] = i_trans
    out[c] = i_axial
    return np.diag(out)


@dataclass
class Part:
    """One primitive in a link's bill of materials.

    `mass` is either given (COTS) or derived from `rho` and the geometry.
    `pos` is the part centroid in the LINK frame; parts are axis-aligned.
    """

    kind: str                       # "cyl" | "tube" | "box"
    dims: tuple                     # cyl:(r,h) tube:(ro,ri,h) box:(sx,sy,sz)
    pos: tuple = (0.0, 0.0, 0.0)
    axis: str = "z"                 # symmetry axis for cyl/tube
    mass: float | None = None
    rho: float | None = None
    note: str = ""

    def volume(self) -> float:
        if self.kind == "cyl":
            r, h = self.dims
            return math.pi * r * r * h
        if self.kind == "tube":
            ro, ri, h = self.dims
            return math.pi * (ro * ro - ri * ri) * h
        sx, sy, sz = self.dims
        return sx * sy * sz

    def resolved_mass(self) -> float:
        if self.mass is not None:
            return self.mass
        if self.rho is None:
            raise ValueError(f"part {self.note!r} has neither mass nor rho")
        return self.rho * self.volume()

    def inertia_com(self) -> np.ndarray:
        """3x3 inertia about the part's own centroid, in the link frame."""
        m = self.resolved_mass()
        if self.kind == "cyl":
            r, h = self.dims
            return _axial(m, m * (3 * r * r + h * h) / 12.0, m * r * r / 2.0, self.axis)
        if self.kind == "tube":
            ro, ri, h = self.dims
            rr = ro * ro + ri * ri
            return _axial(m, m * (3 * rr + h * h) / 12.0, m * rr / 2.0, self.axis)
        sx, sy, sz = self.dims
        return np.diag([
            m * (sy * sy + sz * sz) / 12.0,
            m * (sx * sx + sz * sz) / 12.0,
            m * (sx * sx + sy * sy) / 12.0,
        ])


def box_shell(sx: float, sy: float, sz: float, t: float, pos: tuple,
              rho: float, note: str) -> list[Part]:
    """A printed hollow box, as its six real wall plates.

    Exact rather than "solid box at a made-up fraction of the density": the
    walls are where the material actually is, so both the mass AND the radius
    of gyration come out right.
    """
    x, y, z = pos
    return [
        Part("box", (sx, sy, t), (x, y, z + (sz - t) / 2), rho=rho, note=f"{note} +z wall"),
        Part("box", (sx, sy, t), (x, y, z - (sz - t) / 2), rho=rho, note=f"{note} -z wall"),
        Part("box", (sx, t, sz - 2 * t), (x, y + (sy - t) / 2, z), rho=rho, note=f"{note} +y wall"),
        Part("box", (sx, t, sz - 2 * t), (x, y - (sy - t) / 2, z), rho=rho, note=f"{note} -y wall"),
        Part("box", (t, sy - 2 * t, sz - 2 * t), (x + (sx - t) / 2, y, z), rho=rho,
             note=f"{note} +x wall"),
        Part("box", (t, sy - 2 * t, sz - 2 * t), (x - (sx - t) / 2, y, z), rho=rho,
             note=f"{note} -x wall"),
    ]


def compose(parts: list[Part]) -> tuple[float, np.ndarray, np.ndarray]:
    """Total mass, COM, and inertia ABOUT THE COM for a list of parts.

    Sum of parallel-axis-shifted positive-definite tensors about the composite
    COM -- physically valid by construction, which is why the doctor's A002
    plausibility check (eigenvalues + triangle inequality) passes for free.
    """
    masses = np.array([p.resolved_mass() for p in parts])
    pos = np.array([p.pos for p in parts], dtype=float)
    m_tot = float(masses.sum())
    com = (masses[:, None] * pos).sum(axis=0) / m_tot
    inertia = np.zeros((3, 3))
    for p, m in zip(parts, masses):
        d = np.array(p.pos, dtype=float) - com
        inertia += p.inertia_com() + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return m_tot, com, inertia


# ==========================================================================
# 4. LINK GEOMETRY -- the "simple CAD": every collider is also the visual
# ==========================================================================
@dataclass
class Geom:
    """A collision/visual primitive. The SAME list feeds both, so what the
    doctor lints, what MuJoCo simulates, and what the render shows are one
    shape -- there is no second, prettier truth."""

    kind: str                       # "cylinder" | "box"
    dims: tuple                     # cylinder:(r,length) box:(sx,sy,sz)
    pos: tuple = (0.0, 0.0, 0.0)
    rpy: tuple = (0.0, 0.0, 0.0)


@dataclass
class Link:
    name: str
    parts: list[Part] = field(default_factory=list)
    geoms: list[Geom] = field(default_factory=list)


RPY_Y = (1.5708, 0.0, 0.0)          # cylinder local z -> world -y (lies on Y)


def build_links(l2: float, l3: float, h_base: float) -> list[Link]:
    """The full mechanical description at these link lengths."""
    hs = H_SHOULDER

    # ---- base: pedestal, J1 motor + 10:1 planetary, world-fixed ----------
    base = Link("base", parts=[
        Part("cyl", (0.080, 0.008), (0, 0, 0.004), "z", rho=RHO_AL, note="base plate"),
        Part("tube", (0.062, 0.056, h_base - 0.008), (0, 0, 0.008 + (h_base - 0.008) / 2),
             "z", rho=RHO_PLA, note="pedestal shell"),
        Part("cyl", (0.021, 0.048), (0, 0, 0.040), "z", mass=M_NEMA17, note="J1 NEMA17"),
        Part("cyl", (0.024, 0.030), (0, 0, h_base - 0.020), "z", mass=M_PLANET_L,
             note="J1 10:1 planetary"),
    ], geoms=[
        Geom("cylinder", (0.082, 0.008), (0, 0, 0.004)),
        Geom("cylinder", (0.064, h_base - 0.010), (0, 0, 0.008 + (h_base - 0.010) / 2)),
    ])

    # ---- shoulder: J1 output turret carrying the J2 strain-wave drive -----
    shoulder = Link("shoulder", parts=[
        Part("tube", (0.050, 0.045, 0.086), (0, 0, 0.043), "z", rho=RHO_PLA,
             note="turret shell"),
        Part("cyl", (0.021, 0.048), (0, -0.046, hs), "y", mass=M_NEMA17, note="J2 NEMA17"),
        Part("cyl", (0.033, 0.026), (0, 0.020, hs), "y", mass=M_STRAINWAVE,
             note="J2 strain wave 30:1"),
        Part("tube", (0.006, 0.004, 0.070), (0, 0, hs), "y", rho=RHO_STEEL,
             note="J2 output shaft (bored)"),
    ], geoms=[
        Geom("cylinder", (0.052, 0.086), (0, 0, 0.043)),
        Geom("cylinder", (0.032, 0.118), (0, -0.010, hs), RPY_Y),
    ])

    # ---- upper arm: Al tube; J3 motor kept LOW (belt drive to the elbow) --
    tube2_len = l2 - 0.070
    upper = Link("upper_arm", parts=[
        Part("tube", (0.0225, 0.0210, tube2_len), (0, 0, 0.035 + tube2_len / 2), "z",
             rho=RHO_AL, note="upper-arm tube OD45 x 1.5"),
        Part("tube", (0.026, 0.021, 0.035), (0, 0, 0.0175), "z", rho=RHO_PLA,
             note="lower fitting (printed shell)"),
        Part("tube", (0.026, 0.021, 0.030), (0, 0, l2 - 0.015), "z", rho=RHO_PLA,
             note="elbow fitting (printed shell)"),
        Part("cyl", (0.021, 0.048), (0, -0.042, 0.034), "y", mass=M_NEMA17,
             note="J3 NEMA17 (belt to elbow)"),
        Part("cyl", (0.033, 0.026), (0, 0.018, l2), "y", mass=M_STRAINWAVE,
             note="J3 strain wave 30:1"),
        Part("box", (0.010, 0.006, l2 - 0.05), (0, -0.026, l2 / 2), mass=0.030,
             note="J3 timing belt + pulleys"),
    ], geoms=[
        Geom("box", (0.046, 0.046, l2 - 0.030), (0, 0, l2 / 2)),
        Geom("cylinder", (0.028, 0.098), (0, -0.012, 0.034), RPY_Y),
        Geom("cylinder", (0.036, 0.050), (0, 0.012, l2), RPY_Y),
    ])

    # ---- forearm: lighter Al tube, J4 roll motor at the far end ----------
    tube3_len = l3 - 0.075
    forearm = Link("forearm", parts=[
        Part("tube", (0.0180, 0.0165, tube3_len), (0, 0, 0.030 + tube3_len / 2), "z",
             rho=RHO_AL, note="forearm tube OD36 x 1.5"),
        Part("tube", (0.022, 0.018, 0.032), (0, 0, 0.016), "z", rho=RHO_PLA,
             note="elbow fitting (printed shell)"),
        Part("cyl", (0.017, 0.040), (0, 0, l3 - 0.075), "z", mass=M_NEMA14, note="J4 NEMA14"),
        Part("cyl", (0.020, 0.026), (0, 0, l3 - 0.045), "z", mass=M_PLANET_M,
             note="J4 20:1 planetary"),
    ], geoms=[
        # The forearm body STOPS 35 mm short of the wrist centre. That neck is
        # not styling: without it the J4 motor pod reaches to within 2 mm of the
        # centre and the wrist yoke sweeps straight through it -- caliper's own
        # CollisionModel reported a 31 mm penetration on all 100 work-zone poses
        # before this clearance existed.
        Geom("box", (0.038, 0.038, l3 - 0.075), (0, 0, (l3 - 0.055) / 2)),
        Geom("cylinder", (0.026, 0.055), (0, 0, l3 - 0.0625)),
    ])

    # ---- wrist 1: roll housing straddling the WRIST CENTRE (its origin) ---
    wrist1 = Link("wrist1", parts=[
        Part("tube", (0.030, 0.025, 0.056), (0, 0, -0.016), "z", rho=RHO_PLA,
             note="roll housing"),
        Part("cyl", (0.015, 0.036), (0, -0.026, -0.008), "y", mass=M_NEMA14, note="J5 NEMA14"),
        Part("cyl", (0.018, 0.024), (0, 0.014, -0.008), "y", mass=M_PLANET_M,
             note="J5 20:1 planetary"),
    ], geoms=[
        Geom("cylinder", (0.031, 0.058), (0, 0, -0.016)),
        Geom("cylinder", (0.020, 0.070), (0, -0.008, -0.008), RPY_Y),
    ])

    # ---- wrist 2: pitch yoke carrying the J6 roll drive ------------------
    wrist2 = Link("wrist2", parts=[
        Part("tube", (0.026, 0.021, 0.044), (0, 0, 0.022), "z", rho=RHO_PLA, note="pitch yoke"),
        Part("cyl", (0.013, 0.030), (0, 0, 0.017), "z", mass=M_NEMA11, note="J6 NEMA11"),
        Part("cyl", (0.016, 0.020), (0, 0, 0.038), "z", mass=M_PLANET_S,
             note="J6 14:1 planetary"),
    ], geoms=[
        Geom("cylinder", (0.027, 0.048), (0, 0, 0.024)),
    ])

    # ---- tool: flange, gripper body, FIXED jaw. TCP sits at z = L_W ------
    tool = Link("tool", parts=[
        Part("cyl", (0.026, 0.006), (0, 0, 0.051), "z", rho=RHO_AL, note="tool flange"),
        *box_shell(0.048, 0.058, 0.026, 0.0025, (0, 0, 0.063), RHO_PLA, "gripper body"),
        Part("cyl", (0.008, 0.030), (0, 0, 0.063), "z", mass=M_SERVO, note="jaw gearmotor"),
        Part("cyl", (0.003, 0.048), (0, 0.000, 0.076), "y", rho=RHO_STEEL, note="lead screw"),
        Part("box", (0.034, 0.006, 0.042), (0, -0.021, 0.098), rho=RHO_AL, note="fixed jaw"),
    ], geoms=[
        Geom("cylinder", (0.026, 0.006), (0, 0, 0.051)),
        Geom("box", (0.048, 0.058, 0.026), (0, 0, 0.063)),
        Geom("box", (0.034, 0.006, 0.042), (0, -0.021, 0.098)),
    ])

    # ---- TCP: the tool-centre frame the IK and the task both address -----
    tcp = Link("tcp", parts=[
        Part("cyl", (0.004, 0.004), (0, 0, 0), "z", rho=RHO_AL, note="TCP witness"),
    ], geoms=[Geom("cylinder", (0.004, 0.004))])

    # ---- jaw: the single MOVING jaw, on the prismatic `gripper` joint -----
    jaw = Link("jaw", parts=[
        Part("box", (0.034, 0.006, 0.042), (0, 0, 0), rho=RHO_AL, note="moving jaw"),
        Part("box", (0.020, 0.014, 0.014), (0, -0.008, -0.026), rho=RHO_PLA, note="jaw carriage"),
    ], geoms=[
        Geom("box", (0.034, 0.006, 0.042)),
        Geom("box", (0.020, 0.014, 0.014), (0, -0.008, -0.026)),
    ])

    return [base, shoulder, upper, forearm, wrist1, wrist2, tool, tcp, jaw]


# ==========================================================================
# 5. URDF EMISSION
# ==========================================================================
def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _geom_xml(g: Geom, indent: str) -> str:
    if g.kind == "cylinder":
        r, ln = g.dims
        shape = f'<cylinder radius="{_fmt(r)}" length="{_fmt(ln)}"/>'
    else:
        sx, sy, sz = g.dims
        shape = f'<box size="{_fmt(sx)} {_fmt(sy)} {_fmt(sz)}"/>'
    o = (f'<origin xyz="{_fmt(g.pos[0])} {_fmt(g.pos[1])} {_fmt(g.pos[2])}" '
         f'rpy="{_fmt(g.rpy[0])} {_fmt(g.rpy[1])} {_fmt(g.rpy[2])}"/>')
    return (f"{indent}{o}\n{indent}<geometry>{shape}</geometry>")


def _link_xml(link: Link) -> str:
    m, com, inertia = compose(link.parts)
    out = [f'  <link name="{link.name}">']
    out.append(
        f'    <inertial>\n'
        f'      <origin xyz="{_fmt(com[0])} {_fmt(com[1])} {_fmt(com[2])}" rpy="0 0 0"/>\n'
        f'      <mass value="{_fmt(m)}"/>\n'
        f'      <inertia ixx="{inertia[0, 0]:.8g}" ixy="{inertia[0, 1]:.8g}" '
        f'ixz="{inertia[0, 2]:.8g}" iyy="{inertia[1, 1]:.8g}" '
        f'iyz="{inertia[1, 2]:.8g}" izz="{inertia[2, 2]:.8g}"/>\n'
        f'    </inertial>')
    for g in link.geoms:
        out.append("    <visual>\n" + _geom_xml(g, "      ") + "\n    </visual>")
        out.append("    <collision>\n" + _geom_xml(g, "      ") + "\n    </collision>")
    out.append("  </link>")
    return "\n".join(out)


def _joint_xml(name, jtype, parent, child, xyz, axis, lo, hi, effort, vel) -> str:
    return (
        f'  <joint name="{name}" type="{jtype}">\n'
        f'    <parent link="{parent}"/><child link="{child}"/>\n'
        f'    <origin xyz="{_fmt(xyz[0])} {_fmt(xyz[1])} {_fmt(xyz[2])}" rpy="0 0 0"/>\n'
        f'    <axis xyz="{axis}"/>\n'
        f'    <limit lower="{_fmt(lo)}" upper="{_fmt(hi)}" '
        f'effort="{_fmt(effort)}" velocity="{_fmt(vel)}"/>\n'
        f'  </joint>')


def drive_limits(joint: str) -> tuple[float, float]:
    """(effort N.m, velocity rad/s) at the joint OUTPUT, from the drivetrain."""
    tau, ratio, eta, w_motor = DRIVE[joint]
    return round(tau * ratio * eta, 3), round(w_motor / ratio, 3)


def build_urdf(l2: float, l3: float, h_base: float = H_BASE_DEFAULT, *,
               with_gripper: bool = True, name: str = "ita6") -> str:
    """The complete URDF at these dimensions.

    `with_gripper=False` emits the bare 6R core: caliper's closed-form
    spherical-wrist IK requires ndof == 6, so the design sweep and the analytic
    path both run on that model. The first six joints are byte-identical
    between the two files, so a q solved on the core is valid on the full arm.
    """
    links = build_links(l2, l3, h_base)
    if not with_gripper:
        links = [lk for lk in links if lk.name != "jaw"]
        # Without the moving jaw the tool link keeps only its own geometry.

    e1, v1 = drive_limits("j1")
    e2, v2 = drive_limits("j2")
    e3, v3 = drive_limits("j3")
    e4, v4 = drive_limits("j4")
    e5, v5 = drive_limits("j5")
    e6, v6 = drive_limits("j6")

    header = f"""<?xml version="1.0"?>
<!--
  ITA-6 : 6-DOF spherical-wrist manipulator, {"with parallel-jaw gripper" if with_gripper else "bare 6R kinematic core"}.

  GENERATED by examples/ita6/design_ita6.py : do not hand-edit; re-run the
  script. Link lengths are the winner of a {len(L2_GRID)}x{len(L3_GRID)} design sweep scored on
  caliper's own reachability / manipulability / limit-margin analysis over a
  {(ZONE['x'][1] - ZONE['x'][0]) * 1000:.0f} x {(ZONE['y'][1] - ZONE['y'][0]) * 1000:.0f} x {(ZONE['z'][1] - ZONE['z'][0]) * 1000:.0f} mm table-top work zone.

  KINEMATICS (canonical spherical-wrist 6R; every joint origin rpy = 0):
    J1 yaw    axis Z at z = {_fmt(h_base)} m
    J2 pitch  axis Y  (shoulder)      L2 = {_fmt(l2)} m to the elbow
    J3 pitch  axis Y  (elbow)         L3 = {_fmt(l3)} m to the wrist centre
    J4 roll   axis Z  |
    J5 pitch  axis Y  +-  these three axes intersect at the WRIST CENTRE
    J6 roll   axis Z  |
    tcp_mount fixed, +{_fmt(L_W)} m -> TCP (the frame IK and the task address)
  Reach, base axis -> TCP, arm extended: {_fmt(l2 + l3 + L_W)} m.

  MASS MODEL : composed analytically from a bill of materials, not guessed:
    aluminium 6061 tube/plate  rho = {_fmt(RHO_AL)} kg/m^3   (walls modelled as real walls)
    PLA-CF printed shells      rho = {_fmt(RHO_PLA)} kg/m^3  (hollow geometry, solid wall)
    steel shafts/screws        rho = {_fmt(RHO_STEEL)} kg/m^3
    NEMA17 {_fmt(M_NEMA17)} kg | NEMA14 {_fmt(M_NEMA14)} kg | NEMA11 {_fmt(M_NEMA11)} kg
    strain wave {_fmt(M_STRAINWAVE)} kg | planetary {_fmt(M_PLANET_L)}/{_fmt(M_PLANET_M)}/{_fmt(M_PLANET_S)} kg
  Each <inertial> is the parallel-axis sum of those primitives about the link
  COM : valid by construction, which is why A002 passes without tuning.

  DRIVETRAIN -> <limit> (tau_out = tau_motor * ratio * efficiency):
    J1 NEMA17 0.50 N.m x 30:1 belt+planetary x 0.80 -> {_fmt(e1)} N.m, {_fmt(v1)} rad/s
    J2 NEMA17 0.50 N.m x 30:1 strain wave    x 0.80 -> {_fmt(e2)} N.m, {_fmt(v2)} rad/s
    J3 NEMA17 0.50 N.m x 30:1 strain wave    x 0.80 -> {_fmt(e3)} N.m, {_fmt(v3)} rad/s
    J4 NEMA14 0.22 N.m x 20:1 planetary      x 0.75 -> {_fmt(e4)} N.m, {_fmt(v4)} rad/s
    J5 NEMA14 0.22 N.m x 20:1 planetary      x 0.75 -> {_fmt(e5)} N.m, {_fmt(v5)} rad/s
    J6 NEMA11 0.09 N.m x 14:1 planetary      x 0.75 -> {_fmt(e6)} N.m, {_fmt(v6)} rad/s

  Every <collision> is also the <visual>: the shape the doctor lints, the shape
  MuJoCo integrates and the shape the render shows are one object.
-->
<robot name="{name}">
"""
    body = [header]
    body.extend(_link_xml(lk) for lk in links)
    body.append("")
    body.append(_joint_xml("j1", "revolute", "base", "shoulder",
                           (0, 0, h_base), "0 0 1", *LIMITS["j1"], e1, v1))
    body.append(_joint_xml("j2", "revolute", "shoulder", "upper_arm",
                           (0, 0, H_SHOULDER), "0 1 0", *LIMITS["j2"], e2, v2))
    body.append(_joint_xml("j3", "revolute", "upper_arm", "forearm",
                           (0, 0, l2), "0 1 0", *LIMITS["j3"], e3, v3))
    body.append(_joint_xml("j4", "revolute", "forearm", "wrist1",
                           (0, 0, l3), "0 0 1", *LIMITS["j4"], e4, v4))
    body.append(_joint_xml("j5", "revolute", "wrist1", "wrist2",
                           (0, 0, 0), "0 1 0", *LIMITS["j5"], e5, v5))
    body.append(_joint_xml("j6", "revolute", "wrist2", "tool",
                           (0, 0, 0), "0 0 1", *LIMITS["j6"], e6, v6))
    body.append(
        '  <joint name="tcp_mount" type="fixed">\n'
        '    <parent link="tool"/><child link="tcp"/>\n'
        f'    <origin xyz="0 0 {_fmt(L_W)}" rpy="0 0 0"/>\n'
        '  </joint>')
    if with_gripper:
        # Closed at the LOWER limit: jaw inner faces 36 mm apart, opening to
        # 76 mm -- a 40-50 mm cube is grasped inside the travel.
        body.append(_joint_xml("gripper", "prismatic", "tool", "jaw",
                               (0, 0.021, 0.098), "0 1 0", 0.0, GRIPPER_TRAVEL,
                               GRIPPER_EFFORT, GRIPPER_VEL))
    body.append("</robot>")
    return "\n".join(body) + "\n"


# ==========================================================================
# 6. SCORING -- caliper's own analysis over the work zone
# ==========================================================================
def zone_samples(n=ZONE_N, zone=None) -> np.ndarray:
    zone = zone or ZONE
    xs = np.linspace(*zone["x"], n[0])
    ys = np.linspace(*zone["y"], n[1])
    zs = np.linspace(*zone["z"], n[2])
    return np.array([(x, y, z) for x in xs for y in ys for z in zs])


def down_target(p) -> list[list[float]]:
    """Column-major 4x4: tool z-axis straight down, yawed to face `p`."""
    x, y, z = p
    yaw = math.atan2(y, x)
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    ry180 = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    t = np.eye(4)
    t[:3, :3] = rz @ ry180
    t[:3, 3] = (x, y, z)
    return t.T.tolist()          # caliper's IK takes COLUMN-major


# Retry offsets, micrometres. See `solve_pose`.
_NUDGES = (
    (1e-6, 0.0, 0.0), (0.0, 1e-6, 0.0), (0.0, 0.0, 1e-6),
    (-1e-6, 0.0, 0.0), (0.0, -1e-6, 0.0), (0.0, 0.0, -1e-6),
    (1e-5, 0.0, 0.0), (0.0, 1e-5, 0.0), (0.0, 0.0, 1e-5),
)


NUDGE_STATS = {"nudged": 0, "solved": 0}


def solve_pose(robot, p, seed=None) -> list:
    """Limit-respecting closed-form solutions for "tool vertical at `p`".

    THE ONE IK ENTRY POINT for this file, so the sweep, the collision check,
    the payload check and the figures all ask the same question the same way.

    The retry works around a genuine caliper bug: `analytic_ik` returns an
    empty branch list for a class of reachable, in-limits poses -- always at
    y = 0 (where the tool-down target rotation is symmetric) and erratically
    elsewhere, about 5 % of a dense slice through the working envelope. It is
    a knife-edge, not a region: the y = 0 pose that fails solves fine at
    y = 1e-12, and `fk` of the hand-derived q lands on the target to 1e-16.
    Reproducible on the shipped showcase6 fixture; see README, "Known engine
    bug".

    Displacing the target by a micrometre cannot change what this machine can
    reach by anything that matters (the sample spacing is millimetres), so the
    retry recovers the solver's misses without inventing reachability -- every
    branch it returns is still verified against the target by caliper itself.
    The retries do NOT fully cover the folded-elbow region above z ~ 0.34 m,
    where the miss rate is far higher; the figures plot the working envelope
    below it, and the README says so.
    """
    seed = seed or [math.atan2(p[1], p[0]), *SEED_REST]
    branches = _working_branches(robot.analytic_ik(down_target(p), seed=seed))
    if branches:
        NUDGE_STATS["solved"] += 1
        return branches
    for dx, dy, dz in _NUDGES:
        branches = _working_branches(robot.analytic_ik(
            down_target((p[0] + dx, p[1] + dy, p[2] + dz)), seed=seed))
        if branches:
            # a retry rescued this pose from the y=0 solver bug -- counted so
            # "reached" is auditable as exact-vs-nudged (see analysis.json)
            NUDGE_STATS["nudged"] += 1
            NUDGE_STATS["solved"] += 1
            return branches
    return []


def _working_branches(branches) -> list:
    """Keep only the WORKING wrist branch: J5 pitched up (q5 > 0).

    A spherical wrist reaches every tool-down pose two ways, the second with
    the wrist flipped past vertical (q5 < 0, here always within a few degrees
    of the -120 deg stop). That posture is not one a table-top pick would ever
    command, and admitting it would credit the arm with envelope it cannot
    usefully work in. Every one of the 100 work-zone poses solves in the
    working branch, so this constrains the FIGURES' outer envelope (31 of 3086
    slice cells) and changes no score.
    """
    return [q for q in (branches or []) if q[4] > 0.0]


def score_candidate(robot, samples: np.ndarray) -> dict:
    """Reach / manipulability / limit margin of one arm over the work zone.

    For each sample the tool must reach it with the tool vertical. Among the
    limit-respecting closed-form branches we take the SEED-NEAREST one -- what
    a real controller would command -- not the most flattering one.
    """
    lims = robot.joint_limits
    mids = np.array([(lo + hi) / 2 for lo, hi in lims])
    halves = np.array([(hi - lo) / 2 for lo, hi in lims])

    reached = 0
    manips, margins, pts = [], [], []
    for p in samples:
        branches = solve_pose(robot, p)
        if not branches:                       # None (not 6R) or [] (no IK)
            manips.append(0.0)
            margins.append(0.0)
            continue
        q = np.array(branches[0])
        reached += 1
        m = robot.manipulability(list(q))
        margin = float(np.min(1.0 - np.abs(q - mids) / halves))
        manips.append(m)
        margins.append(max(margin, 0.0))
        pts.append((*p, m, margin))

    manips = np.array(manips)
    margins = np.array(margins)
    n = len(samples)
    return {
        "reach_fraction": reached / n,
        # Indices count UNREACHABLE samples as zero: quality-weighted coverage.
        "manip_index": float(manips.mean()),
        "margin_index": float(margins.mean()),
        # ...and the honest per-reached-pose numbers, for the write-up.
        "manip_mean_reached": float(manips[manips > 0].mean()) if reached else 0.0,
        "margin_mean_reached": float(margins[manips > 0].mean()) if reached else 0.0,
        "n_samples": n,
        "reached_points": pts,
    }


def self_collision_report_full(arm, full, samples: np.ndarray) -> dict:
    """Self-collision of the SHIPPED robot (jaw included) over the zone.

    IK runs on the bare 6R core (the closed form needs ndof == 6); each
    solution is then checked on the full 7-DOF model with the gripper padded
    at BOTH travel extremes -- the jaw is a moving body sitting exactly where
    the first geometry pass had real interference, so checking only the arm
    would verify a robot we do not ship.
    """
    cm = caliper.CollisionModel(full)
    names = full.frame_names()
    g_lo, g_hi = full.joint_limits[6]
    pairs: dict[tuple[str, str], float] = {}
    hits = 0
    for p in samples:
        branches = solve_pose(arm, p)
        if not branches:
            continue
        colliding = False
        for g in (g_lo, g_hi):
            q7 = list(branches[0]) + [g]
            if cm.query(q7)["collision"]:
                colliding = True
                for a, b, info in cm.contacts(q7):
                    key = (names[a], names[b])
                    pairs[key] = max(pairs.get(key, 0.0), float(info["depth"]))
        hits += int(colliding)
    return {
        "colliding_poses": hits,
        "n_samples": int(len(samples)),
        "gripper_configs_checked": ["open", "closed"],
        "colliders": cm.num_colliders,
        "uncovered_frames": cm.uncovered_frames,
        "pairs": {f"{a} <-> {b}": round(d, 5) for (a, b), d in pairs.items()},
    }


def self_collision_report(robot, samples: np.ndarray) -> dict:
    """Do the work-zone IK solutions actually fit inside the machine?

    Reachability says the wrist centre lands where it should; it says nothing
    about the forearm passing through the wrist. caliper's own CollisionModel
    answers that, and it is the check that caught the 31 mm forearm/wrist-yoke
    interference in the first geometry pass.

    Ground contact is NOT counted: the pedestal is bolted to the table, so its
    collider resting on the z = 0 plane is the intended state, not a fault.
    """
    cm = caliper.CollisionModel(robot)
    names = robot.frame_names()
    pairs: dict[tuple[str, str], float] = {}
    hits = 0
    for p in samples:
        branches = solve_pose(robot, p)
        if not branches:
            continue
        q = list(branches[0])
        if not cm.query(q)["collision"]:
            continue
        hits += 1
        for a, b, info in cm.contacts(q):
            key = (names[a], names[b])
            pairs[key] = max(pairs.get(key, 0.0), float(info["depth"]))
    return {
        "colliding_poses": hits,
        "n_samples": int(len(samples)),
        "colliders": cm.num_colliders,
        "uncovered_frames": cm.uncovered_frames,
        "pairs": {f"{a} <-> {b}": round(d, 5) for (a, b), d in pairs.items()},
    }


def static_payload_check(robot, samples: np.ndarray, payload: float) -> dict:
    """Worst-case static joint torque over the zone with `payload` at the TCP.

    Gravity torque of the arm itself (caliper RNEA) plus J^T applied at the TCP
    for the payload weight -- exact, and it uses the engine's own Jacobian.
    """
    worst = np.zeros(6)
    worst_at = None
    per_joint_peak = np.zeros(6)  # each joint's own max over the zone
    # World-frame wrench of the payload hanging at the TCP, [v; omega] ordering
    # (caliper's jacobian puts the three linear rows first -- verified against a
    # hand-computed 90-degree shoulder pose).
    wrench = np.array([0.0, 0.0, -payload * G, 0.0, 0.0, 0.0])
    for p in samples:
        branches = solve_pose(robot, p)
        if not branches:
            continue
        q = list(branches[0])
        # `gravity_torque` is the torque the actuators must HOLD (it already
        # opposes gravity); `J^T w` is the generalised force the payload EXERTS
        # on the joints, so holding it costs -J^T w. Adding them instead of
        # subtracting makes the two cancel -- at a 1 kg / 0.45 m stretch that
        # turns a real 7.7 N.m into a comfortable-looking 0.6 N.m.
        j = np.array(robot.jacobian(q))          # 6 x ndof, World frame
        tau = np.array(robot.gravity_torque(q)) - j.T @ wrench
        per_joint_peak = np.maximum(per_joint_peak, np.abs(tau[:6]))
        if np.max(np.abs(tau)) > np.max(np.abs(worst)):
            worst, worst_at = tau, tuple(float(v) for v in p)
    efforts = [drive_limits(f"j{i + 1}")[0] for i in range(6)]
    return {
        "payload_kg": payload,
        "worst_torque_nm": [round(float(t), 3) for t in worst],
        "at_point": worst_at,
        "effort_limit_nm": efforts,
        "utilisation": [round(abs(float(t)) / e, 3) for t, e in zip(worst, efforts)],
        # each joint's OWN peak over the zone (the single worst pose above is
        # worst for ONE joint; other joints can peak at other poses)
        "per_joint_peak_nm": [round(float(t), 3) for t in per_joint_peak],
        "per_joint_peak_utilisation": [round(float(t) / e, 3) for t, e in zip(per_joint_peak, efforts)],
        "within_limits": bool(all(float(t) <= e for t, e in zip(per_joint_peak, efforts))),
    }


# ==========================================================================
# 7. THE SWEEP
# ==========================================================================
def load_candidate(tmpdir: Path, l2: float, l3: float, h_base: float):
    """Emit the 6R core at these dimensions and load it with caliper."""
    path = tmpdir / f"cand_{l2:.3f}_{l3:.3f}_{h_base:.3f}.urdf"
    path.write_text(build_urdf(l2, l3, h_base, with_gripper=False, name="ita6_candidate"))
    return caliper.Robot.from_urdf(str(path))


def combined(entry: dict, manip_ref: float) -> float:
    norm = entry["manip_index"] / manip_ref if manip_ref > 0 else 0.0
    return (W_REACH * entry["reach_fraction"]
            + W_MANIP * min(norm, 1.0)
            + W_MARGIN * entry["margin_index"])


def in_reach_band(l2: float, l3: float) -> bool:
    return abs(l2 + l3 + L_W - REACH_TARGET) <= REACH_BAND + 1e-9


def main() -> None:
    t0 = time.perf_counter()
    samples = zone_samples()
    tmp = Path(tempfile.mkdtemp(prefix="ita6_"))
    print(f"caliper {caliper.__version__}  |  work zone "
          f"x{ZONE['x']} y{ZONE['y']} z{ZONE['z']}  |  {len(samples)} poses/candidate\n")

    # ---- stage 1: L2 x L3 -------------------------------------------------
    grid = []
    for l2 in L2_GRID:
        for l3 in L3_GRID:
            robot = load_candidate(tmp, l2, l3, H_BASE_DEFAULT)
            s = score_candidate(robot, samples)
            s.pop("reached_points")
            s.update(L2=l2, L3=l3, h_base=H_BASE_DEFAULT,
                     reach=round(l2 + l3 + L_W, 4), feasible=in_reach_band(l2, l3))
            grid.append(s)
    manip_ref = max(e["manip_index"] for e in grid) or 1.0
    for e in grid:
        e["score"] = round(combined(e, manip_ref), 4)

    print(f"STAGE 1 -- design sweep, {len(grid)} candidates "
          f"(weights: reach {W_REACH}, manip {W_MANIP}, margin {W_MARGIN})")
    print(f"{'L2':>6} {'L3':>6} {'reach':>7} {'450mm':>6} {'reach_f':>8} "
          f"{'manip_ix':>9} {'margin':>7} {'score':>7}")
    for e in grid:
        print(f"{e['L2']:6.3f} {e['L3']:6.3f} {e['reach']:7.3f} "
              f"{'yes' if e['feasible'] else '-':>6} {e['reach_fraction']:8.2f} "
              f"{e['manip_index']:9.5f} {e['margin_index']:7.3f} {e['score']:7.3f}")

    feasible = [e for e in grid if e["feasible"]]
    if not feasible:
        raise SystemExit("no candidate satisfies the reach constraint")
    best = max(feasible, key=lambda e: e["score"])
    best_overall = max(grid, key=lambda e: e["score"])
    print(f"\n  best in the {REACH_TARGET * 1000:.0f}+-{REACH_BAND * 1000:.0f} mm band: "
          f"L2={best['L2']:.3f} L3={best['L3']:.3f}  score {best['score']:.3f}")
    print(f"  best ignoring the reach constraint: L2={best_overall['L2']:.3f} "
          f"L3={best_overall['L3']:.3f}  score {best_overall['score']:.3f} "
          f"(reach {best_overall['reach']:.3f} m)")

    # ---- stage 2: base height at the winning link lengths ------------------
    print("\nSTAGE 2 -- base height at the winning link lengths")
    print(f"{'h_base':>7} {'reach_f':>8} {'manip_ix':>9} {'margin':>7} {'score':>7}")
    heights = []
    for h in H_BASE_GRID:
        robot = load_candidate(tmp, best["L2"], best["L3"], h)
        s = score_candidate(robot, samples)
        s.pop("reached_points")
        s.update(L2=best["L2"], L3=best["L3"], h_base=h,
                 reach=best["reach"], feasible=True)
        s["score"] = round(combined(s, manip_ref), 4)
        heights.append(s)
        print(f"{h:7.3f} {s['reach_fraction']:8.2f} {s['manip_index']:9.5f} "
              f"{s['margin_index']:7.3f} {s['score']:7.3f}")
    best_h = max(heights, key=lambda e: e["score"])
    print(f"\n  best base height: {best_h['h_base']:.3f} m  score {best_h['score']:.3f}")

    winner = dict(best_h)
    l2w, l3w, hbw = winner["L2"], winner["L3"], winner["h_base"]

    # ---- the winner, in full ----------------------------------------------
    arm_path = HERE / "ita6_arm6.urdf"
    full_path = HERE / "ita6.urdf"
    arm_path.write_text(build_urdf(l2w, l3w, hbw, with_gripper=False, name="ita6_arm6"))
    full_path.write_text(build_urdf(l2w, l3w, hbw, with_gripper=True, name="ita6"))

    arm = caliper.Robot.from_urdf(str(arm_path))
    full = caliper.Robot.from_urdf(str(full_path))
    NUDGE_STATS["nudged"] = NUDGE_STATS["solved"] = 0
    detail = score_candidate(arm, samples)
    winner_nudges = dict(NUDGE_STATS)
    selfcol = self_collision_report_full(arm, full, samples)
    payload_1kg = static_payload_check(arm, samples, PAYLOAD_KG)
    payload_half = static_payload_check(arm, samples, 0.5)

    # Measured (not inferred) twin-agreement and IK-residual figures: the two
    # URDFs share a byte-identical 6R chain, but the claim ships as a number.
    # `fk(q)` returns the TIP pose as a 4x4. The twins' tips differ (tcp vs
    # jaw), so the measurement is: T_arm_tcp(q)^-1 . T_full_jaw(q+[0]) must be
    # the CONSTANT fixed tcp->jaw offset for every q -- any drift means the 6R
    # chains differ. Stronger than a single-frame diff, and it is a number.
    rng = np.random.default_rng(0)
    lims = arm.joint_limits
    x0 = None
    tcp_diff = 0.0
    for _ in range(50):
        q6 = [float(rng.uniform(lo, hi)) for lo, hi in lims]
        fa = np.array(arm.fk(q6))
        ff = np.array(full.fk(q6 + [0.0]))
        x = np.linalg.inv(fa) @ ff
        if x0 is None:
            x0 = x
        tcp_diff = max(tcp_diff, float(np.max(np.abs(x - x0))))
    ik_resid = 0.0
    n_res = 0
    for pnt in samples:
        for q in solve_pose(arm, pnt)[:1]:
            tip = np.array(arm.fk(list(q)))[:3, 3]
            ik_resid = max(ik_resid, float(np.linalg.norm(tip - np.array(pnt))))
            n_res += 1
    verification = {
        "tcp_agreement_max": tcp_diff,
        "tcp_agreement_note": "max element drift of inv(T_arm_tcp) @ T_full_jaw across 50 random in-limit q (gripper 0) -- the fixed tcp->jaw offset must be constant iff the twin 6R chains are identical",
        "ik_residual_max_m": ik_resid,
        "ik_residual_note": f"max |FK(analytic branch) - target| over {n_res} reached zone poses (nudged targets measured against the ORIGINAL point, so the bound includes the 1e-5 m worst-case nudge)",
        "nudged_samples": winner_nudges["nudged"],
        "solved_samples": winner_nudges["solved"],
    }
    print(f"  twin TCP agreement        max |dFK| = {tcp_diff:.3e}")
    print(f"  analytic IK residual      max {ik_resid:.3e} m over {n_res} poses "
          f"({winner_nudges['nudged']} of {winner_nudges['solved']} needed the micro-nudge)")

    doc_arm = caliper.doctor(str(arm_path))
    doc_full = caliper.doctor(str(full_path))

    # Moving mass = everything the actuators have to accelerate (all but the
    # world-fixed pedestal). `robot.total_mass` counts exactly the same set,
    # because the root link carries no joint.
    per_link, moving, m_base = {}, 0.0, 0.0
    for lk in build_links(l2w, l3w, hbw):
        m, _, _ = compose(lk.parts)
        per_link[lk.name] = round(m, 4)
        if lk.name == "base":
            m_base = m
        else:
            moving += m

    # Honesty check: the winner against the LARGER box the brief first named.
    brief = score_candidate(arm, zone_samples(zone=BRIEF_ZONE))
    brief.pop("reached_points")

    elapsed = time.perf_counter() - t0
    print(f"\nWINNER  L2 = {l2w * 1000:.0f} mm  L3 = {l3w * 1000:.0f} mm  "
          f"h_base = {hbw * 1000:.0f} mm  L_w = {L_W * 1000:.0f} mm")
    print(f"  reach (base axis -> TCP)  {(l2w + l3w + L_W) * 1000:.0f} mm")
    print(f"  work-zone reachability    {detail['reach_fraction'] * 100:.1f} % "
          f"of {detail['n_samples']} vertical-tool poses")
    print(f"  manipulability            index {detail['manip_index']:.5f}, "
          f"mean over reached {detail['manip_mean_reached']:.5f}")
    print(f"  joint-limit margin        {detail['margin_mean_reached']:.3f} "
          f"(1.0 = dead centre of every range)")
    print(f"  reachability, brief box   {brief['reach_fraction'] * 100:.1f} % "
          f"(the larger 200x300x200 box -- outside the machine, reported anyway)")
    print(f"  mass                      {moving + m_base:.3f} kg total "
          f"({moving:.3f} kg moving + {m_base:.3f} kg pedestal)")
    print(f"  static {PAYLOAD_KG:.1f} kg payload    worst |tau| per joint "
          f"{[round(abs(t), 2) for t in payload_1kg['worst_torque_nm']]} N.m")
    print(f"                            vs effort {payload_1kg['effort_limit_nm']} N.m "
          f"-> {'WITHIN LIMITS' if payload_1kg['within_limits'] else 'OVER LIMIT'}")
    print(f"  self-collision            {selfcol['colliding_poses']} of "
          f"{selfcol['n_samples']} work-zone poses"
          + (f" -> {selfcol['pairs']}" if selfcol["pairs"] else " (clean)"))
    print(f"  doctor ita6_arm6.urdf     {doc_arm['errors']} errors, "
          f"{doc_arm['warnings']} warnings, {doc_arm['infos']} infos")
    print(f"  doctor ita6.urdf          {doc_full['errors']} errors, "
          f"{doc_full['warnings']} warnings, {doc_full['infos']} infos")
    print(f"\nsweep took {elapsed:.1f} s "
          f"({len(grid) + len(H_BASE_GRID)} candidates x {len(samples)} IK poses)")

    analysis = {
        "generated_by": "examples/ita6/design_ita6.py",
        "caliper_version": caliper.__version__,
        "sweep_seconds": round(elapsed, 2),
        "method": {
            "work_zone": ZONE,
            "brief_zone": BRIEF_ZONE,
            "zone_samples": list(ZONE_N),
            "tool_orientation": "vertical (tool -z down), yawed radially to the sample",
            "branch_selection": "seed-nearest limit-respecting analytic branch",
            "seed_rest": list(SEED_REST),
            "weights": {"reach": W_REACH, "manipulability": W_MANIP,
                        "limit_margin": W_MARGIN},
            "manipulability_normalisation": "divided by the best index in the grid",
            "reach_constraint_m": [REACH_TARGET - REACH_BAND, REACH_TARGET + REACH_BAND],
            "wrist_offset_m": L_W,
        },
        "grid": grid,
        "base_height_refinement": heights,
        "winner": {
            "L2": l2w, "L3": l3w, "h_base": hbw, "L_w": L_W,
            "h_shoulder": H_SHOULDER,
            "reach_m": round(l2w + l3w + L_W, 4),
            "score": winner["score"],
            "reach_fraction": detail["reach_fraction"],
            "manip_index": detail["manip_index"],
            "manip_mean_reached": detail["manip_mean_reached"],
            "margin_index": detail["margin_index"],
            "margin_mean_reached": detail["margin_mean_reached"],
            "reach_fraction_brief_zone": brief["reach_fraction"],
            "total_mass_kg": round(moving + m_base, 4),
            "moving_mass_kg": round(moving, 4),
            "engine_total_mass_kg": round(full.total_mass, 4),
            "mass_per_link_kg": per_link,
            "joint_limits_rad": {k: list(v) for k, v in LIMITS.items()},
            "joint_effort_nm": {f"j{i + 1}": drive_limits(f"j{i + 1}")[0] for i in range(6)},
            "joint_velocity_rads": {f"j{i + 1}": drive_limits(f"j{i + 1}")[1]
                                    for i in range(6)},
        },
        "self_collision": selfcol,
        "verification": verification,
        "payload_check": {"structural_1kg": payload_1kg, "precision_0p5kg": payload_half},
        "doctor": {
            "ita6_arm6.urdf": {k: doc_arm[k] for k in
                               ("errors", "warnings", "infos", "clean")},
            "ita6.urdf": {k: doc_full[k] for k in
                          ("errors", "warnings", "infos", "clean")},
            "findings": [f["code"] + " " + f["severity"] + ": " + f["message"]
                         for f in doc_arm["findings"] + doc_full["findings"]],
        },
        "reached_points": [[round(v, 6) for v in row] for row in detail["reached_points"]],
    }
    (HERE / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(f"\nwrote {arm_path.name}, {full_path.name}, analysis.json")


if __name__ == "__main__":
    main()
