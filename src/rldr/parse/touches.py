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

`build_touches` combines them with the touch rule agreed with Ari from the evidence in
`scripts/touches.py` (see `resolve_touches`, ROADMAP 1.3 and the Decision Log).
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


def ball_arrays(parsed: ParsedReplay) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ball position (n, 3), velocity (n, 3) and sleeping (n,), indexed by frame."""
    e = parsed.entities
    ball = e[e["entity"] == "ball"].sort_values("frame")
    pos = ball[["x", "y", "z"]].to_numpy(dtype=float, copy=True)
    vel = ball[["vx", "vy", "vz"]].to_numpy(dtype=float, copy=True)
    sleeping = ball["sleeping"].fillna(False).astype(bool).to_numpy()
    return pos, vel, sleeping


def car_positions(parsed: ParsedReplay) -> dict[str, np.ndarray]:
    """Car position (n, 3) per player_key, NaN while demolished or absent (can't touch)."""
    cars = parsed.entities[parsed.entities["entity"] != "ball"]
    out = {}
    for k in parsed.players["player_key"]:
        c = cars[cars["entity"] == k].sort_values("frame")
        p = c[["x", "y", "z"]].to_numpy(dtype=float, copy=True)
        p[~(c["is_present"] & ~c["is_demolished"]).to_numpy()] = np.nan
        out[k] = p
    return out


def car_ball_dist(car: np.ndarray, ball: np.ndarray, f: int) -> float:
    """Car center to ball center around a touch at frame f: the smaller of the distance at
    f-1 and at f (the contact happens between the two). NaN if the car is off the field on both.
    """
    return float(
        np.fmin(np.linalg.norm(car[f - 1] - ball[f - 1]), np.linalg.norm(car[f] - ball[f]))
    )


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
    pos, vel, sleeping = ball_arrays(parsed)
    live = parsed.frames["is_live"].to_numpy()

    dv = np.r_[0.0, np.linalg.norm(np.diff(vel, axis=0), axis=1)]
    ok = (dv > dv_min) & live & np.r_[False, live[:-1]] & ~sleeping
    cand = np.flatnonzero(ok)

    keys = list(parsed.players["player_key"])
    team_of = dict(zip(parsed.players["player_key"], parsed.players["team"], strict=True))
    car_pos = car_positions(parsed)

    rows = []
    surfaces = surface_contacts(pos[cand - 1], margin) if len(cand) else []
    for i, f in enumerate(cand):
        dists = []
        for k in keys:
            d = car_ball_dist(car_pos[k], pos, f)
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


# --------------------------------------------------------------------------- the touch rule
# Agreed with Ari from the evidence in scripts/touches.py (ROADMAP 1.3, Decision Log).

TOUCH_DEDUP_FRAMES = 2  # same player, stats touches this close = one touch
TOUCH_MATCH_FRAMES = 3  # stats touch <-> physics touch, same player
CONTEST_GAP = 30.0  # uu: another car in contact range this close to the nearest = 50/50
PHYSICS_ONLY_DV_MIN = 500.0  # uu/s, for touches the stats stream does not have
PHYSICS_ONLY_CLEAR_FRAMES = 10  # ...and no stats touch (any player) within this many frames

TOUCH_RULE_COLUMNS = [
    "frame",
    "player_key",
    "team",
    "stats_frame",
    "touch_source",
    "contested",
    "contest_type",
    "alt_player_key",
    "source_conflict",
]


def dedup_stats(st: pd.DataFrame, window: int = TOUCH_DEDUP_FRAMES) -> pd.DataFrame:
    """Drop a stats touch when the same player has a kept touch at most `window` frames before.

    A touch on a frame where the stats stream also credits another player is never dropped:
    it is half of a contested pair (0AAC 4:37: iDrxp 1112, then iDrxp + Ari on 1113).
    """
    shared = st["frame"].map(st.groupby("frame")["player_key"].nunique()) > 1
    keep = []
    last: dict[str, int] = {}
    for i, (f, k, pair) in enumerate(zip(st["frame"], st["player_key"], shared, strict=True)):
        if k in last and f - last[k] <= window and not pair:
            continue
        last[k] = f
        keep.append(i)
    return st.iloc[keep].reset_index(drop=True)


def _closest(c: pd.DataFrame, f: int) -> pd.Series:
    """Row of `c` nearest to frame f (ties go to the earlier frame)."""
    rank = (c["frame"] - f).abs() * 2 + (c["frame"] > f)
    return c.loc[rank.idxmin()]


def _match(near: pd.DataFrame, k: str) -> pd.Series | None:
    """The physics touch (within the match window) that stands for player k's stats touch.

    Candidates: k is the nearest car, or k is second-nearest and also in contact range. Rows
    where k is the nearest car or within CONTEST_GAP of it come first, nearest in time; a row
    where k is in range but far behind the nearest car (a conflict) only if there is no other.
    """
    mine = near["player_key"] == k
    second = (near["second_key"] == k) & (near["second_dist"] <= CONTACT_RANGE)
    close = second & (near["second_dist"] - near["dist"] <= CONTEST_GAP)
    for sel in (mine | close, second & ~close):
        if sel.any():
            return near[sel]
    return None


def resolve_touches(st: pd.DataFrame, ph: pd.DataFrame, team_of: dict[str, int]) -> pd.DataFrame:
    """Apply the touch rule to a stats touch table and a physics candidate table.

    `st`: frame, player_key (the stats `touch` stream). `ph`: physics candidates as returned by
    `physics_touches` (frame, dv, player_key, dist, second_key, second_dist, surfaces, bounce).

    1. The stats stream is the base list; same-player repeats within 2 frames are one touch.
    2. Each stats touch is matched to a physics touch within +-3 frames where the credited
       player is the nearest car in contact range (or the second-nearest, also in range). When
       it matches, the touch moves to the physics frame (where the ball's velocity actually
       jumps) and `touch_source = "both"`; otherwise it keeps the stats frame and
       `touch_source = "stats"`.
    3. Contested (`contested`, `alt_player_key`, `contest_type` = "opponent" / "teammate"):
       the stats stream credits 2+ players on one frame (one row each), or physics has another
       car in contact range within 30 uu of the nearest. Who won a 50/50 is not decided here.
    4. `source_conflict`: physics puts a different car on the ball (the credited player is not
       in contact range, or is in range but more than 30 uu further than the nearest car).
       `alt_player_key` is the physics car.
    5. Physics-only touches are added when |dv| >= 500, exactly one car is in contact range,
       the ball is not near any arena surface, and the stats stream has no touch within +-10
       frames. `touch_source = "physics"`.
    """
    st = dedup_stats(st.sort_values("frame", kind="stable").reset_index(drop=True))
    ph = ph[ph["dist"] <= CONTACT_RANGE]
    players_at = st.groupby("frame")["player_key"].agg(list).to_dict()

    rows = []
    for f, k in zip(st["frame"], st["player_key"], strict=True):
        f = int(f)
        r = {"frame": f, "player_key": k, "stats_frame": f, "touch_source": "stats"}
        r |= {"contested": False, "alt_player_key": None, "source_conflict": False}
        near = ph[(ph["frame"] - f).abs() <= TOUCH_MATCH_FRAMES]
        cands = _match(near, k)
        if cands is not None:
            c = _closest(cands, f)
            r |= {"frame": int(c["frame"]), "touch_source": "both"}
            alt = c["second_key"] if c["player_key"] == k else c["player_key"]
            in_range = c["second_dist"] <= CONTACT_RANGE
            if in_range and c["second_dist"] - c["dist"] <= CONTEST_GAP:
                r |= {"contested": True, "alt_player_key": alt}
            elif c["player_key"] != k:
                r |= {"source_conflict": True, "alt_player_key": alt}
        elif len(near):
            c = _closest(near, f)
            r |= {"source_conflict": True, "alt_player_key": c["player_key"]}
        others = [o for o in players_at[f] if o != k]
        if others:  # stats credits several players on this frame: the pair is the contest
            r |= {"contested": True, "alt_player_key": others[0]}
        rows.append(r)

    stats_frames = st["frame"].to_numpy()
    for c in ph.itertuples():
        if (
            c.dv >= PHYSICS_ONLY_DV_MIN
            and c.second_dist > CONTACT_RANGE
            and not c.surfaces
            and not c.bounce
            and not (np.abs(stats_frames - c.frame) <= PHYSICS_ONLY_CLEAR_FRAMES).any()
        ):
            rows.append(
                {
                    "frame": int(c.frame),
                    "player_key": c.player_key,
                    "stats_frame": None,
                    "touch_source": "physics",
                    "contested": False,
                    "alt_player_key": None,
                    "source_conflict": False,
                }
            )

    cols = [c for c in TOUCH_RULE_COLUMNS if c not in ("team", "contest_type")]
    df = pd.DataFrame(rows, columns=cols)
    # Two stats touches by one player can land on the same physics frame: keep one.
    df = df.sort_values("frame", kind="stable").drop_duplicates(["frame", "player_key"])
    df["team"] = df["player_key"].map(team_of)
    contested = df["contested"].astype(bool).to_numpy()
    same_team = (df["alt_player_key"].map(team_of) == df["team"]).to_numpy()
    df["contest_type"] = np.where(
        contested, np.where(same_team, "teammate", "opponent"), None
    ).astype(object)
    df["stats_frame"] = df["stats_frame"].astype("Int64")
    return df[TOUCH_RULE_COLUMNS].reset_index(drop=True)


def touch_confidence(
    touch_source: pd.Series,
    contest_type: pd.Series,
    source_conflict: pd.Series,
    car_dist: pd.Series,
) -> np.ndarray:
    """high / medium / low per touch (agreed with Ari).

    * low: teammate-contested (a double-commit signal), or the sources disagree on the car.
    * medium: one source only, or an opponent 50/50 with the credited car in contact range.
    * high: stats and physics agree on the player and nobody else is on the ball.
    An opponent 50/50 with the credited car out of contact range is low.
    """
    teammate = (contest_type == "teammate").to_numpy(dtype=bool)
    opponent = (contest_type == "opponent").to_numpy(dtype=bool)
    conflict = source_conflict.astype(bool).to_numpy()
    in_range = (car_dist <= CONTACT_RANGE).to_numpy(dtype=bool)
    both = (touch_source == "both").to_numpy(dtype=bool)
    return np.select(
        [teammate | conflict, opponent & in_range, opponent, both],
        ["low", "medium", "low", "high"],
        default="medium",
    )


def build_touches(parsed: ParsedReplay, stats_events: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Touch rows for the events table (type "touch").

    Columns: the rule columns (see `resolve_touches`), `confidence`, and per-touch physics:
    ball x/y/z at frame-1 (just before contact), ball_speed_before (frame-1), ball_speed_after
    (frame), ball_dv = |v(frame) - v(frame-1)|, car_ball_dist (credited car, see
    `car_ball_dist`).
    """
    team_of = dict(zip(parsed.players["player_key"], parsed.players["team"], strict=True))
    df = resolve_touches(stats_touches(stats_events), physics_touches(parsed), team_of)
    pos, vel, _ = ball_arrays(parsed)
    car_pos = car_positions(parsed)
    f = df["frame"].to_numpy(dtype=np.int64)
    df["type"] = "touch"
    df[["x", "y", "z"]] = pos[f - 1]
    df["ball_speed_before"] = np.linalg.norm(vel[f - 1], axis=1)
    df["ball_speed_after"] = np.linalg.norm(vel[f], axis=1)
    df["ball_dv"] = np.linalg.norm(vel[f] - vel[f - 1], axis=1)
    df["car_ball_dist"] = [
        car_ball_dist(car_pos[k], pos, int(fr)) for k, fr in zip(df["player_key"], f, strict=True)
    ]
    df["confidence"] = touch_confidence(
        df["touch_source"], df["contest_type"], df["source_conflict"], df["car_ball_dist"]
    )
    return df
