#!/usr/bin/env python3
"""Poster figures for ITA-6 — all three rendered from the real model.

    figures/ita6_render.png      MuJoCo offscreen render of ita6.urdf via
                                 caliper_learn.sim_camera (the arm posed over
                                 its work zone with the task cube).
    figures/ita6_workspace.png   Reachable set with the tool vertical, coloured
                                 by manipulability: side slice + top slice,
                                 work zone outlined.
    figures/ita6_optimum.png     The design sweep itself — combined score over
                                 the L2 x L3 grid, the 450 mm reach band, and
                                 the optimum.

Run AFTER design_ita6.py (it reads ita6.urdf and analysis.json):

    MUJOCO_DYNAMIC_LINK_DIR=~/.cache/caliper/mujoco-3.9.0 \\
      PATH=/usr/sbin:$PATH .venv/bin/python examples/ita6/make_figures.py

Colours follow a single-hue sequential blue ramp for magnitude and one orange
accent for the optimum — no rainbow maps, so the figures stay readable in
greyscale print and for colour-vision-deficient readers.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FIGS = HERE / "figures"

import caliper  # noqa: E402
from design_ita6 import BRIEF_ZONE, ZONE, solve_pose  # noqa: E402

# --- palette (single-hue sequential blue + one orange accent) ---------------
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ACCENT = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d3"
UNREACHED = "#f2f1ee"
DPI = 200


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({
        "font.family": "sans-serif",
        # Arial first: the poster is set in Arial, and matplotlib silently
        # falls through this list rather than failing if it is absent.
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 11,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    cmap = LinearSegmentedColormap.from_list("caliper_blue", BLUE_RAMP)
    cmap.set_bad(UNREACHED)
    return plt, cmap


# ==========================================================================
# FIGURE A — the render
# ==========================================================================
# Link colours: structure in cool greys, actuator pods darker, the moving jaw
# in the accent so the gripper reads at poster distance.
LINK_RGBA = {
    "base": "0.36 0.38 0.42 1",
    "shoulder": "0.80 0.82 0.85 1",
    "upper_arm": "0.86 0.88 0.90 1",
    "forearm": "0.86 0.88 0.90 1",
    "wrist1": "0.62 0.66 0.72 1",
    "wrist2": "0.62 0.66 0.72 1",
    "tool": "0.45 0.48 0.54 1",
    "tcp": "0.92 0.41 0.20 1",
    "jaw": "0.92 0.41 0.20 1",
}

# A flat white skybox and a soft headlight: MuJoCo's default is a blue haze
# gradient, which no poster wants behind a hero shot.
STYLE_XML = """  <visual>
    <global offwidth="3200" offheight="2400"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.65 0.65 0.65" specular="0.12 0.12 0.12"/>
    <rgba haze="1 1 1 1"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map znear="0.01" zfar="30"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1"
             width="512" height="512"/>
  </asset>
"""


def style_mjcf(xml: str) -> str:
    """Inject the white studio look and per-link colours into caliper's MJCF."""
    xml = xml.replace("  <worldbody>", STYLE_XML + "  <worldbody>", 1)

    def colour(match: re.Match) -> str:
        head, link = match.group(0), match.group(2)
        rgba = LINK_RGBA.get(link)
        return f'{head} rgba="{rgba}"' if rgba else head

    # caliper names every exported collider `col<i>_<link>` (mjcf.rs), which is
    # the hook for colouring the render without touching the URDF.
    xml = re.sub(r'<geom name="(col\d+_(\w+))"', colour, xml)
    xml = xml.replace('name="caliper_ground" type="plane"',
                      'name="caliper_ground" type="plane" rgba="0.93 0.93 0.92 1"')
    return xml


def look_at_camera_xml(target, distance: float, az_deg: float, el_deg: float,
                       name: str = "hero", fovy: float = 45.0) -> str:
    """A world-fixed `<camera>` aimed at `target` from (azimuth, elevation).

    `sim_camera.camera_xml` frames the WORLD ORIGIN, which for an arm working
    out at +x leaves the subject in one corner. This aims where the robot is.
    MuJoCo's `xyaxes` is the camera's right vector then its up vector; it looks
    along -(x cross y), so z is the target-to-camera direction.
    """
    t = np.asarray(target, dtype=float)
    az, el = math.radians(az_deg), math.radians(el_deg)
    z = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
                  math.sin(el)])
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pos = t + distance * z
    fmt = lambda v: " ".join(f"{c:.6g}" for c in v)  # noqa: E731
    return (f'<light pos="{fmt(t + [0.2, -0.3, 1.4])}" dir="-0.15 0.2 -1" '
            f'directional="true"/>'
            f'<camera name="{name}" pos="{fmt(pos)}" '
            f'xyaxes="{fmt(x)} {fmt(y)}" fovy="{fovy}"/>')


def crop_to_content(rgb: np.ndarray, pad_frac: float = 0.035,
                    tol: int = 6) -> np.ndarray:
    """Trim the white surround to the rendered subject, keeping a small margin.

    A perspective render of an off-centre subject always leaves uneven white
    borders; cropping to content is what makes the figure sit on a poster
    without hand-tweaking the camera for every pose.
    """
    ink = np.any(rgb < (255 - tol), axis=2)
    rows, cols = np.any(ink, axis=1), np.any(ink, axis=0)
    if not rows.any() or not cols.any():
        return rgb
    r0, r1 = np.flatnonzero(rows)[[0, -1]]
    c0, c1 = np.flatnonzero(cols)[[0, -1]]
    pad = int(round(pad_frac * max(r1 - r0, c1 - c0)))
    r0, r1 = max(r0 - pad, 0), min(r1 + pad + 1, rgb.shape[0])
    c0, c1 = max(c0 - pad, 0), min(c1 + pad + 1, rgb.shape[1])
    return rgb[r0:r1, c0:c1]


def zone_wireframe_xml(zone: dict, rgba: str, thickness: float = 0.0016) -> str:
    """Twelve thin boxes = the wireframe edges of a zone (MJCF has no wire mode)."""
    (x0, x1), (y0, y1), (z0, z1) = zone["x"], zone["y"], zone["z"]
    t = thickness
    out = []
    for y in (y0, y1):
        for z in (z0, z1):
            out.append(f'<geom type="box" size="{(x1 - x0) / 2} {t} {t}" '
                       f'pos="{(x0 + x1) / 2} {y} {z}" rgba="{rgba}" contype="0" '
                       f'conaffinity="0"/>')
    for x in (x0, x1):
        for z in (z0, z1):
            out.append(f'<geom type="box" size="{t} {(y1 - y0) / 2} {t}" '
                       f'pos="{x} {(y0 + y1) / 2} {z}" rgba="{rgba}" contype="0" '
                       f'conaffinity="0"/>')
    for x in (x0, x1):
        for y in (y0, y1):
            out.append(f'<geom type="box" size="{t} {t} {(z1 - z0) / 2}" '
                       f'pos="{x} {y} {(z0 + z1) / 2}" rgba="{rgba}" contype="0" '
                       f'conaffinity="0"/>')
    return "".join(out)


def figure_render(analysis: dict, width: int = 3000, height: int = 2250) -> Path:
    from caliper_learn.sim_camera import SimCameraScene

    task = json.loads((HERE / "ita6.caliper-task.json").read_text())
    robot = caliper.Robot.from_urdf(str(HERE / "ita6.urdf"))
    cube = task["scene"]["props"][0]
    zone = task["scene"]["zones"][0]

    # Table slab under the work zone, the work-zone wireframe, and the drop bin.
    zc, zh = zone["center"], zone["half"]
    extra = (
        '<geom name="table" type="box" size="0.34 0.30 0.006" pos="0.22 0 -0.006" '
        'rgba="0.88 0.87 0.84 1" contype="0" conaffinity="0"/>'
        + zone_wireframe_xml(ZONE, "0.22 0.52 0.84 0.75", thickness=0.0011)
        + f'<geom name="bin" type="box" size="{zh[0]} {zh[1]} 0.002" '
          f'pos="{zc[0]} {zc[1]} 0.001" rgba="0.16 0.47 0.84 0.45" '
          f'contype="0" conaffinity="0"/>'
    )

    cam = look_at_camera_xml(target=(0.17, 0.0, 0.16), distance=0.82,
                             az_deg=-52.0, el_deg=20.0)
    xml = caliper.model_to_mjcf(robot, ground=0.0, extra_xml=cam + extra,
                                props=[cube])
    scene = SimCameraScene(style_mjcf(xml), width=width, height=height,
                           camera="hero")

    q = list(task["q0"])                       # the task's own pre-grasp pose
    cp = cube["pos"]
    qpos = np.array(q + [cp[0], cp[1], cp[2], 1.0, 0.0, 0.0, 0.0])
    rgb = crop_to_content(scene.render(qpos))
    scene.close()

    from PIL import Image
    out = FIGS / "ita6_render.png"
    Image.fromarray(rgb).save(out)
    return out


# ==========================================================================
# FIGURE B — the reachable set
# ==========================================================================
def reach_slice(robot, us, vs, point) -> np.ndarray:
    """Manipulability over a 2D slice; NaN where the tool-down pose is unreachable."""
    grid = np.full((len(vs), len(us)), np.nan)
    for j, v in enumerate(vs):
        for i, u in enumerate(us):
            branches = solve_pose(robot, point(u, v))
            if branches:
                grid[j, i] = robot.manipulability(list(branches[0]))
    return grid


def figure_workspace(analysis: dict, n: int = 150) -> Path:
    plt, cmap = _mpl()
    robot = caliper.Robot.from_urdf(str(HERE / "ita6_arm6.urdf"))
    w = analysis["winner"]

    # Side slice, in a vertical plane just off the symmetry plane (y = 0 is the
    # one place caliper's closed-form IK reports false negatives -- see README).
    # z stops at 0.32 m, just under the top of the tool-vertical envelope.
    # Above it lies a folded-elbow over-the-top region the closed form only
    # finds intermittently (the same engine bug `solve_pose` documents), so
    # plotting it would show solver coverage rather than robot geometry.
    y_slice = 0.03
    xs = np.linspace(-0.02, 0.50, n)
    zs = np.linspace(0.0, 0.32, n)
    side = reach_slice(robot, xs, zs, lambda x, z: (x, y_slice, z))

    # Top slice, at the middle height of the work zone.
    z_slice = round((ZONE["z"][0] + ZONE["z"][1]) / 2, 3)
    xs2 = np.linspace(-0.02, 0.50, n)
    ys2 = np.linspace(-0.26, 0.26, n)
    top = reach_slice(robot, xs2, ys2, lambda x, y: (x, y, z_slice))

    vmax = float(np.nanmax([np.nanmax(side), np.nanmax(top)]))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))

    im = axes[0].pcolormesh(xs, zs, side, cmap=cmap, vmin=0.0, vmax=vmax,
                            shading="nearest", rasterized=True)
    axes[0].add_patch(plt.Rectangle(
        (ZONE["x"][0], ZONE["z"][0]), ZONE["x"][1] - ZONE["x"][0],
        ZONE["z"][1] - ZONE["z"][0], fill=False, ec=ACCENT, lw=2.0, zorder=5))
    axes[0].plot([0, 0], [0, w["h_base"]], color=INK, lw=3, solid_capstyle="butt",
                 zorder=6)
    axes[0].plot([0], [w["h_base"] + w["h_shoulder"]], marker="o", ms=6,
                 color=INK, zorder=6)
    axes[0].set_xlabel("x  (m)")
    axes[0].set_ylabel("z  (m)")
    axes[0].set_aspect("equal")
    axes[0].text(0.015, w["h_base"] + w["h_shoulder"] + 0.012, "J2", color=INK,
                 fontsize=10)
    axes[0].text(ZONE["x"][0], ZONE["z"][1] + 0.015, "work zone", color=ACCENT,
                 fontsize=10, fontweight="bold")
    axes[0].text(0.985, 0.03, f"side slice,  y = {y_slice:.2f} m",
                 transform=axes[0].transAxes, ha="right", fontsize=10, color=INK_2)

    axes[1].pcolormesh(xs2, ys2, top, cmap=cmap, vmin=0.0, vmax=vmax,
                       shading="nearest", rasterized=True)
    axes[1].add_patch(plt.Rectangle(
        (ZONE["x"][0], ZONE["y"][0]), ZONE["x"][1] - ZONE["x"][0],
        ZONE["y"][1] - ZONE["y"][0], fill=False, ec=ACCENT, lw=2.0, zorder=5))
    axes[1].add_patch(plt.Rectangle(
        (BRIEF_ZONE["x"][0], BRIEF_ZONE["y"][0]),
        BRIEF_ZONE["x"][1] - BRIEF_ZONE["x"][0],
        BRIEF_ZONE["y"][1] - BRIEF_ZONE["y"][0], fill=False, ec=INK_2, lw=1.0,
        ls=(0, (4, 3)), zorder=4))
    axes[1].plot([0], [0], marker="o", ms=6, color=INK, zorder=6)
    axes[1].set_xlabel("x  (m)")
    axes[1].set_ylabel("y  (m)")
    axes[1].set_aspect("equal")
    axes[1].text(0.985, 0.03, f"top slice,  z = {z_slice:.2f} m",
                 transform=axes[1].transAxes, ha="right", fontsize=10, color=INK_2)

    for ax in axes:
        ax.tick_params(length=3, width=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02)
    cbar.set_label("manipulability  $\\sqrt{\\det JJ^{\\mathsf{T}}}$   (m$^3$)")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=3, width=0.8)

    out = FIGS / "ita6_workspace.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ==========================================================================
# FIGURE C — the design sweep
# ==========================================================================
def figure_optimum(analysis: dict) -> Path:
    plt, cmap = _mpl()
    grid = analysis["grid"]
    l2s = sorted({e["L2"] for e in grid})
    l3s = sorted({e["L3"] for e in grid})
    score = np.full((len(l3s), len(l2s)), np.nan)
    reach = np.full_like(score, np.nan)
    for e in grid:
        score[l3s.index(e["L3"]), l2s.index(e["L2"])] = e["score"]
        reach[l3s.index(e["L3"]), l2s.index(e["L2"])] = e["reach"]

    w = analysis["winner"]
    band = analysis["method"]["reach_constraint_m"]

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    mm2, mm3 = [v * 1000 for v in l2s], [v * 1000 for v in l3s]
    im = ax.pcolormesh(mm2, mm3, score, cmap=cmap, shading="nearest",
                       rasterized=True)

    for j, l3 in enumerate(mm3):
        for i, l2 in enumerate(mm2):
            v = score[j, i]
            ax.text(l2, l3, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                    color="white" if v > 0.55 else INK)

    # Headroom above the top row so the winner callout never lands on a value.
    step3 = mm3[1] - mm3[0]
    ax.set_ylim(mm3[0] - step3 / 2, mm3[-1] + step3 / 2 + 26)

    # The 450 mm reach requirement, as a band on the design space.
    cs = ax.contour(mm2, mm3, reach, levels=[band[0], band[1]], colors=INK_2,
                    linewidths=1.1, linestyles="dashed")
    ax.clabel(cs, fmt=lambda v: f"{v * 1000:.0f} mm", fontsize=9, inline=True)

    ax.plot(w["L2"] * 1000, w["L3"] * 1000, marker="*", ms=26, color=ACCENT,
            mec="white", mew=1.4, zorder=8)
    ax.annotate(f"ITA-6:  L2 = {w['L2'] * 1000:.0f} mm,  L3 = {w['L3'] * 1000:.0f} mm"
                f"   (score {w['score']:.3f})",
                xy=(w["L2"] * 1000, w["L3"] * 1000 + step3 / 2),
                xytext=(w["L2"] * 1000, mm3[-1] + step3 / 2 + 15),
                fontsize=11, fontweight="bold", color=ACCENT, ha="center",
                va="center",
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.2,
                                shrinkA=2, shrinkB=6))
    # Say out loud that the dark corner is NOT the answer: it is a longer arm.
    fig.text(0.5, -0.015,
             "dashed band = the 450 mm reach requirement;  "
             "higher scores outside it belong to longer arms",
             fontsize=10, color=INK_2, ha="center", va="top")

    ax.set_xlabel("upper-arm length  L2  (mm)")
    ax.set_ylabel("forearm length  L3  (mm)")
    ax.set_xticks(mm2)
    ax.set_yticks(mm3)
    ax.tick_params(length=3, width=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("combined score   0.50 reach + 0.30 manipulability + 0.20 margin")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=3, width=0.8)

    out = FIGS / "ita6_optimum.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    analysis = json.loads((HERE / "analysis.json").read_text())
    from PIL import Image
    for fn in (figure_render, figure_optimum, figure_workspace):
        out = fn(analysis)
        w, h = Image.open(out).size
        print(f"{out.relative_to(HERE.parent.parent)}  {w} x {h}")


if __name__ == "__main__":
    main()
