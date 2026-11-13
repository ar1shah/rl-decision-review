"""Field geometry (uu) and drawing for the 2D renderer (ROADMAP 1.5).

Coordinates are Unreal units: x side to side, y goal to goal, z up. Blue defends -y (drawn at
the bottom), Orange defends +y (top).

The constants here (field, goals, ball, car hitboxes) move to `rldr.state.geometry` in 2.1, so
the detectors share one copy (touch contact range and challenge logic need the hitboxes).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Polygon, Rectangle

# --------------------------------------------------------------------------- field geometry

SIDE_WALL_X = 4096.0
BACK_WALL_Y = 5120.0
CORNER_CUT = 8064.0  # 45-degree corner walls lie on |x| + |y| = 8064 (checked in 1.3)
CEILING_Z = 2044.0
GOAL_HALF_WIDTH = 893.0  # goal mouth x in [-893, 893]
GOAL_DEPTH = 880.0  # net box behind the goal line, y from +-5120 to +-6000
GOAL_HEIGHT = 642.0
BALL_RADIUS = 92.75
CENTER_CIRCLE_R = 1000.0  # cosmetic only, approximate

# Standard hitboxes (length, width, height) in uu, keyed by `players.car` (hitbox family).
HITBOXES: dict[str, tuple[float, float, float]] = {
    "Octane": (118.01, 84.20, 36.16),
    "Dominus": (127.93, 83.28, 31.30),
    "Plank": (128.82, 84.67, 29.39),
    "Breakout": (131.49, 80.52, 30.30),
    "Hybrid": (127.02, 82.19, 34.16),
    "Merc": (120.72, 76.71, 41.66),
}
DEFAULT_HITBOX = "Octane"

TEAM_COLORS = {0: "#2f6fdb", 1: "#f28c28"}
TEAM_NAMES = {0: "Blue", 1: "Orange"}
FIELD_FACE = "#eef3ea"
LINE_COLOR = "#9aa59a"
PAD_COLOR = "#e0b000"

# Standard Soccar boost pads, used when a cached replay predates meta["boost_pads"].
# Same 34 positions as fd["boost_pads"] on the lab replays (tested).
_SMALL_HALF = [  # (x, y) on Blue's half; mirrored to Orange's by y -> -y
    (0, -4240),
    (-1792, -4184),
    (1792, -4184),
    (-940, -3308),
    (940, -3308),
    (0, -2816),
    (-3584, -2484),
    (3584, -2484),
    (-1788, -2300),
    (1788, -2300),
    (-2048, -1036),
    (2048, -1036),
    (0, -1024),
]
_SMALL_MID = [(-1024, 0), (1024, 0)]
_BIG = [(-3072, -4096), (3072, -4096), (-3584, 0), (3584, 0), (-3072, 4096), (3072, 4096)]


def _standard_pads() -> pd.DataFrame:
    small = _SMALL_HALF + [(x, -y) for x, y in _SMALL_HALF] + _SMALL_MID
    rows = [{"x": float(x), "y": float(y), "z": 70.0, "size": "small"} for x, y in small]
    rows += [{"x": float(x), "y": float(y), "z": 73.0, "size": "big"} for x, y in _BIG]
    return pd.DataFrame(rows).sort_values(["y", "x"], ignore_index=True)


STANDARD_PADS = _standard_pads()


def hitbox_for(car: str | None) -> tuple[float, float, float]:
    """(length, width, height) for a hitbox family; Octane when unknown or missing."""
    if isinstance(car, str) and car in HITBOXES:
        return HITBOXES[car]
    return HITBOXES[DEFAULT_HITBOX]


def pads_from_meta(meta: Mapping[str, Any]) -> pd.DataFrame:
    """Boost pads (x, y, z, size) from `ParsedReplay.meta["boost_pads"]`, else the standard set."""
    pads = meta.get("boost_pads")
    if not pads:
        return STANDARD_PADS.copy()
    return pd.DataFrame(pads)[["x", "y", "z", "size"]]


def field_outline() -> np.ndarray:
    """The 8 floor-outline vertices (x, y), counter-clockwise from the +x wall's top end."""
    cx = CORNER_CUT - BACK_WALL_Y  # x where a corner wall meets a back wall (2944)
    cy = CORNER_CUT - SIDE_WALL_X  # y where a corner wall meets a side wall (3968)
    return np.array(
        [
            (SIDE_WALL_X, cy),
            (cx, BACK_WALL_Y),
            (-cx, BACK_WALL_Y),
            (-SIDE_WALL_X, cy),
            (-SIDE_WALL_X, -cy),
            (-cx, -BACK_WALL_Y),
            (cx, -BACK_WALL_Y),
            (SIDE_WALL_X, -cy),
        ]
    )


def goal_box(team: int) -> tuple[float, float, float, float]:
    """Net box behind a team's goal line as (x0, y0, width, height) for a Rectangle."""
    sign = -1.0 if team == 0 else 1.0
    y0 = sign * BACK_WALL_Y if sign > 0 else -BACK_WALL_Y - GOAL_DEPTH
    return (-GOAL_HALF_WIDTH, y0, 2 * GOAL_HALF_WIDTH, GOAL_DEPTH)


# --------------------------------------------------------------------------- drawing

TOP_XLIM = (-SIDE_WALL_X - 250, SIDE_WALL_X + 250)
TOP_YLIM = (-BACK_WALL_Y - GOAL_DEPTH - 150, BACK_WALL_Y + GOAL_DEPTH + 150)
SIDE_XLIM = TOP_YLIM
SIDE_YLIM = (-120, CEILING_Z + 120)


def draw_field(ax: Axes, pads: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None) -> None:
    """Top-down field to scale: walls, corner cuts, goals + net boxes, center markings, pads."""
    pads = STANDARD_PADS if pads is None else pd.DataFrame(pads)
    ax.add_patch(
        Polygon(field_outline(), closed=True, facecolor=FIELD_FACE, edgecolor="#555", lw=1.5)
    )
    for team in (0, 1):
        x0, y0, w, h = goal_box(team)
        color = TEAM_COLORS[team]
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=color, alpha=0.18, edgecolor=color))
        line_y = -BACK_WALL_Y if team == 0 else BACK_WALL_Y
        ax.plot([-GOAL_HALF_WIDTH, GOAL_HALF_WIDTH], [line_y, line_y], color=color, lw=3)
    ax.plot([-SIDE_WALL_X, SIDE_WALL_X], [0, 0], color=LINE_COLOR, lw=1)
    ax.add_patch(Circle((0, 0), CENTER_CIRCLE_R, fill=False, edgecolor=LINE_COLOR, lw=1))
    for _, p in pads.iterrows():
        big = p["size"] == "big"
        ax.add_patch(
            Circle(
                (p["x"], p["y"]),
                160 if big else 70,
                facecolor=PAD_COLOR,
                alpha=0.55 if big else 0.4,
                edgecolor="none",
            )
        )
    ax.set_xlim(*TOP_XLIM)
    ax.set_ylim(*TOP_YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_side_field(ax: Axes) -> None:
    """Side profile (y horizontal, z up): floor, ceiling, back walls above the goals, nets."""
    ax.add_patch(
        Rectangle(
            (-BACK_WALL_Y, 0), 2 * BACK_WALL_Y, CEILING_Z, facecolor=FIELD_FACE, edgecolor="none"
        )
    )
    ax.plot([-BACK_WALL_Y, BACK_WALL_Y], [0, 0], color="#555", lw=1.5)
    ax.plot([-BACK_WALL_Y, BACK_WALL_Y], [CEILING_Z, CEILING_Z], color="#555", lw=1)
    for team, sign in ((0, -1.0), (1, 1.0)):
        wall = sign * BACK_WALL_Y
        color = TEAM_COLORS[team]
        ax.plot([wall, wall], [GOAL_HEIGHT, CEILING_Z], color="#555", lw=1.5)
        back = wall + sign * GOAL_DEPTH
        ax.add_patch(
            Rectangle(
                (min(wall, back), 0),
                GOAL_DEPTH,
                GOAL_HEIGHT,
                facecolor=color,
                alpha=0.18,
                edgecolor=color,
            )
        )
        ax.plot([wall, wall], [0, GOAL_HEIGHT], color=color, lw=3)
    ax.set_xlim(*SIDE_XLIM)
    ax.set_ylim(*SIDE_YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
