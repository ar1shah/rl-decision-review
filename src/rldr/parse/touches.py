"""Touch sources for ROADMAP 1.3.

Three independent views of "who touched the ball when":

* `team_touches`    : `fd["touch_events"]` (the replay's own touch replication). Team only;
                      the player is always null.
* `stats_touches`   : `sa.get_stats_events(path)` stream `touch` (library heuristic). Has the
                      player.
* `physics_touches` : our own. A sudden change in ball velocity between consecutive frames,
                      credited to the nearest car on the field, with ball-surface contacts
                      (floor, walls, ceiling, corners, goal frame) labelled so bounces next to a
                      car can be told apart from hits.

The final touch rule is chosen from the evidence in `scripts/touches.py` and confirmed with
Ari before it is built (see ROADMAP 1.3).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from rldr.parse.frames import ParsedReplay, player_key

# --------------------------------------------------------------------------- field geometry
# Standard Soccar arena (uu). Checked against ball extents on both lab replays: ball z
# 85..1955, |x| <= 4011, |y| <= 5046 outside the goal mouth, |x|+|y| <= 7939 (corner planes
# at |x|+|y| = 8064, minus R*sqrt(2) for the ball center).
BALL_RADIUS = 92.75
FLOOR_Z = 0.0
CEILING_Z = 2044.0
SIDE_WALL_X = 4096.0
BACK_WALL_Y = 5120.0
CORNER_SUM = 8064.0  # corner planes: |x| + |y| = CORNER_SUM
GOAL_HALF_WIDTH = 893.0
GOAL_HEIGHT = 642.0

SURFACE_MARGIN = 40.0  # extra slack beyond the ball radius for "touching a surface"
BOUNCE_NORMAL_COS = np.cos(np.radians(30))  # Δv this aligned with the normal = a bounce

# --------------------------------------------------------------------------- physics params
PHYSICS_DV_MIN = 300.0  # uu/s change in ball velocity between consecutive frames
CONTACT_RANGE = 250.0  # car center to ball center; ball radius + car half-length + slack


# --------------------------------------------------------------------------- source tables


def stats_touches(stats_events: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """The stats `touch` stream as a table: frame, player_key, team, ball_speed_change.

    Uses `payload.frame` (it lines up with `touch_events`; `meta.timing.start_frame` is
    usually one frame later).
    """
    rows = []
    for e in stats_events:
        if e["meta"]["stream"] != "touch":
            continue
        p = e["payload"]["payload"]
        rows.append(
            {
                "frame": int(p["frame"]),
                "player_key": player_key(p["player"]),
                "team": 0 if p["is_team_0"] else 1,
                "ball_speed_change": float(p.get("ball_speed_change") or 0.0),
            }
        )
    cols = ["frame", "player_key", "team", "ball_speed_change"]
    return pd.DataFrame(rows, columns=cols).sort_values("frame", ignore_index=True)


def team_touches(touch_events: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """`fd["touch_events"]` as a table: frame, team (player is always null there)."""
    rows = [
        {"frame": int(t["frame"]), "team": 0 if t["team_is_team_0"] else 1} for t in touch_events
    ]
    return pd.DataFrame(rows, columns=["frame", "team"]).sort_values("frame", ignore_index=True)


# --------------------------------------------------------------------------- surfaces


def surface_contacts(pos: np.ndarray, margin: float = SURFACE_MARGIN) -> list[list[str]]:
    """Which arena surfaces each ball position (n, 3) is touching, within radius + margin.

    Surfaces: floor, ceiling, side_wall, back_wall, corner, goal. "goal" covers the goal frame
    (posts, crossbar) and the inside of the net: any ball near the back-wall plane that isn't
    cleanly inside the open goal mouth.
    """
    reach = BALL_RADIUS + margin
    x, y, z = np.abs(pos[:, 0]), np.abs(pos[:, 1]), pos[:, 2]
    near_back = y > BACK_WALL_Y - reach
    open_mouth = (x < GOAL_HALF_WIDTH - reach) & (z < GOAL_HEIGHT - reach)
    in_mouth_band = (x < GOAL_HALF_WIDTH + reach) & (z < GOAL_HEIGHT + reach)
    checks = {
        "floor": z < FLOOR_Z + reach,
        "ceiling": z > CEILING_Z - reach,
        "side_wall": x > SIDE_WALL_X - reach,
        "back_wall": near_back & ~in_mouth_band,
        "corner": (x + y) > CORNER_SUM - reach * np.sqrt(2),
        "goal": near_back & in_mouth_band & ~(open_mouth & (y < BACK_WALL_Y)),
    }
    return [[s for s, hit in checks.items() if hit[i]] for i in range(len(pos))]


def _surface_normal(surface: str, pos: np.ndarray) -> np.ndarray:
    """Outward normal (pointing from the field into the surface) at a ball position."""
    sx, sy = np.sign(pos[0]) or 1.0, np.sign(pos[1]) or 1.0
    if surface == "floor":
        return np.array([0.0, 0.0, -1.0])
    if surface == "ceiling":
        return np.array([0.0, 0.0, 1.0])
    if surface == "side_wall":
        return np.array([sx, 0.0, 0.0])
    if surface in ("back_wall", "goal"):
        return np.array([0.0, sy, 0.0])
    return np.array([sx, sy, 0.0]) / np.sqrt(2)  # corner


def is_bounce(surface: str, pos: np.ndarray, v_before: np.ndarray, v_after: np.ndarray) -> bool:
    """Bounce signature: the ball was moving into the surface, now moves out of it, and the
    velocity change points along the surface normal (a car hit also changes the ball's motion
    along the surface). For the goal frame, any of the three axes may be the normal.
    """
    dv = v_after - v_before
    ndv = np.linalg.norm(dv)
    if ndv == 0:
        return False
    normals = [_surface_normal(surface, pos)]
    if surface == "goal":
        normals += [np.array([np.sign(pos[0]) or 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    for n in normals:
        if v_before @ n > 0 and v_after @ n < 0 and abs(dv @ n) / ndv > BOUNCE_NORMAL_COS:
            return True
    return False


# --------------------------------------------------------------------------- physics touches


def physics_touches(
    parsed: ParsedReplay, dv_min: float = PHYSICS_DV_MIN, margin: float = SURFACE_MARGIN
) -> pd.DataFrame:
    """Candidate touches from ball velocity jumps, with nearest-car attribution.

    One row per frame f where |v_ball(f) - v_ball(f-1)| > dv_min, both frames live and the ball
    awake at f (this excludes the goal frame, where the ball stops in the net). The kickoff hit
    (ball asleep at f-1, awake at f) is kept.

    Columns: frame, dv, ball x/y/z (at f-1, before the change), player_key / team / dist of the
    nearest on-field car (min of the car-ball distance at f-1 and f; demolished or absent cars
    are skipped), second_key / second_dist (next nearest), surfaces (list), bounce (bool: some
    surface shows a bounce signature).
    """
    e = parsed.entities
    frames = parsed.frames
    ball = e[e["entity"] == "ball"].sort_values("frame")
    pos = ball[["x", "y", "z"]].to_numpy(dtype=float)
    vel = ball[["vx", "vy", "vz"]].to_numpy(dtype=float)
    sleeping = ball["sleeping"].fillna(False).astype(bool).to_numpy()
    live = frames["is_live"].to_numpy()

    dv = np.r_[0.0, np.linalg.norm(np.diff(vel, axis=0), axis=1)]
    ok = (dv > dv_min) & live & np.r_[False, live[:-1]] & ~sleeping
    cand = np.flatnonzero(ok)

    keys = list(parsed.players["player_key"])
    team_of = dict(zip(parsed.players["player_key"], parsed.players["team"], strict=True))
    cars = e[e["entity"] != "ball"]
    car_pos = {}
    for k in keys:
        c = cars[cars["entity"] == k].sort_values("frame")
        p = c[["x", "y", "z"]].to_numpy(dtype=float, copy=True)
        on_field = (c["is_present"] & ~c["is_demolished"]).to_numpy()
        p[~on_field] = np.nan  # demolished / absent cars can't touch the ball
        car_pos[k] = p

    rows = []
    surfaces = surface_contacts(pos[cand - 1], margin) if len(cand) else []
    for i, f in enumerate(cand):
        dists = []
        for k in keys:
            d = np.fmin(  # NaN only if the car is off the field on both frames
                np.linalg.norm(car_pos[k][f - 1] - pos[f - 1]),
                np.linalg.norm(car_pos[k][f] - pos[f]),
            )
            if np.isfinite(d):
                dists.append((d, k))
        dists.sort()
        (d1, k1), (d2, k2) = (dists + [(np.inf, None)] * 2)[:2]
        surf = surfaces[i]
        bounce = any(is_bounce(s, pos[f - 1], vel[f - 1], vel[f]) for s in surf)
        rows.append(
            {
                "frame": int(f),
                "dv": float(dv[f]),
                "x": pos[f - 1, 0],
                "y": pos[f - 1, 1],
                "z": pos[f - 1, 2],
                "player_key": k1,
                "team": team_of.get(k1) if k1 else None,
                "dist": float(d1),
                "second_key": k2,
                "second_dist": float(d2),
                "surfaces": surf,
                "bounce": bounce,
            }
        )
    cols = ["frame", "dv", "x", "y", "z", "player_key", "team", "dist"]
    cols += ["second_key", "second_dist", "surfaces", "bounce"]
    return pd.DataFrame(rows, columns=cols)
