"""Discrete event table built from a replay (ROADMAP 1.3).

One row per event, sorted by frame. Common columns for every type:

    frame, time, clock, is_overtime, run, type, player_key, team, x, y, z

`frame` is the raw network frame index (same as `ParsedReplay.frames`); time, clock,
is_overtime and run are joined from there. `x, y, z` is where it happened: the ball for
goals / kickoffs / stat events, the victim for demos, the pad for boost pickups.
`team` for a demo is the attacker's team.

Type-specific columns are nullable and only filled for their type:

    goal          team0_score, team1_score (running score after the goal)
    demo          other_player_key (victim; player_key is the attacker), attacker_x/y/z
    kickoff       countdown_frame, first_touch_frame (frame = first live frame)
    boost_pickup  pad_id, pad_size ("big" / "small"), is_live
    shot/save/assist  ball_speed (ball at the stat frame), shot_on_target (shots only)

Touches are added separately (see ROADMAP 1.3: the touch rule is decided from evidence).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from rldr.parse.frames import COUNTDOWN_STATE, LIVE_STATE, ParsedReplay, player_key

# Same-frame events are ordered so that cause comes before effect.
EVENT_TYPES = ["kickoff", "touch", "shot", "save", "goal", "assist", "demo", "boost_pickup"]
STAT_KINDS = {"Shot": "shot", "Save": "save", "Assist": "assist"}

COMMON_COLUMNS = [
    "frame",
    "time",
    "clock",
    "is_overtime",
    "run",
    "type",
    "player_key",
    "team",
    "x",
    "y",
    "z",
]
# Type-specific columns and their dtypes (all nullable).
EXTRA_COLUMNS: dict[str, str] = {
    "team0_score": "Int16",
    "team1_score": "Int16",
    "other_player_key": "string",
    "attacker_x": "float64",
    "attacker_y": "float64",
    "attacker_z": "float64",
    "countdown_frame": "Int64",
    "first_touch_frame": "Int64",
    "pad_id": "string",
    "pad_size": "string",
    "is_live": "boolean",
    "ball_speed": "float64",
    "shot_on_target": "boolean",
}
EVENT_COLUMNS = COMMON_COLUMNS + list(EXTRA_COLUMNS)


def _key(remote_id: dict[str, Any] | None) -> str | None:
    return None if remote_id is None else player_key(remote_id)


def _xyz(v: dict[str, float] | None) -> tuple[float, float, float]:
    return (np.nan, np.nan, np.nan) if v is None else (v["x"], v["y"], v["z"])


def _ball_xyz(parsed: ParsedReplay, frames: Sequence[int]) -> np.ndarray:
    """Ball position at each frame, shape (len(frames), 3)."""
    ball = parsed.entities[parsed.entities["entity"] == "ball"].set_index("frame")
    return ball.loc[list(frames), ["x", "y", "z"]].to_numpy(dtype=float).reshape(-1, 3)


# --------------------------------------------------------------------------- per-type builders


def goal_events(goals: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """From `fd["goal_events"]`: scorer, scoring team, running score."""
    rows = [
        {
            "frame": int(g["frame"]),
            "type": "goal",
            "player_key": _key(g.get("player")),
            "team": 0 if g["scoring_team_is_team_0"] else 1,
            "team0_score": g["team_zero_score"],
            "team1_score": g["team_one_score"],
        }
        for g in goals
    ]
    return pd.DataFrame(rows)


def demo_events(demos: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """From `fd["demolish_infos"]`: attacker (player_key), victim, both locations."""
    rows = []
    for d in demos:
        vx, vy, vz = _xyz(d.get("victim_location"))
        ax, ay, az = _xyz(d.get("attacker_location"))
        rows.append(
            {
                "frame": int(d["frame"]),
                "type": "demo",
                "player_key": _key(d.get("attacker")),
                "other_player_key": _key(d.get("victim")),
                "x": vx,
                "y": vy,
                "z": vz,
                "attacker_x": ax,
                "attacker_y": ay,
                "attacker_z": az,
            }
        )
    return pd.DataFrame(rows)


def kickoff_frames(states: Sequence[str], ball_sleeping: Sequence[bool]) -> list[dict[str, int]]:
    """Every Countdown -> Active transition.

    Returns dicts with `countdown_frame` (first frame of that countdown), `frame` (first live
    frame) and `first_touch_frame` (first frame from there on where the ball is awake; the ball
    sleeps at the kickoff spot until it's hit), or -1 if the ball never wakes up.
    A countdown that never goes live (e.g. a forfeit right after a goal) is not a kickoff.
    """
    states = np.asarray(states, dtype=object)
    asleep = np.asarray(ball_sleeping, dtype=bool)
    out = []
    for f in range(1, len(states)):
        if states[f - 1] == COUNTDOWN_STATE and states[f] == LIVE_STATE:
            cd = f - 1
            while cd > 0 and states[cd - 1] == COUNTDOWN_STATE:
                cd -= 1
            awake = np.flatnonzero(~asleep[f:])
            out.append(
                {
                    "frame": f,
                    "countdown_frame": cd,
                    "first_touch_frame": f + int(awake[0]) if len(awake) else -1,
                }
            )
    return out


def kickoff_events(parsed: ParsedReplay) -> pd.DataFrame:
    states = parsed.frames["game_state"].astype(str).to_numpy()
    ball = parsed.entities[parsed.entities["entity"] == "ball"]
    sleeping = ball["sleeping"].fillna(False).astype(bool).to_numpy()
    ks = kickoff_frames(states, sleeping)
    if not ks:
        return pd.DataFrame()
    df = pd.DataFrame(ks).assign(type="kickoff")
    df["first_touch_frame"] = df["first_touch_frame"].where(df["first_touch_frame"] >= 0)
    df[["x", "y", "z"]] = _ball_xyz(parsed, df["frame"])
    return df


def pickup_events(
    pad_events: Sequence[dict[str, Any]], pads: Sequence[dict[str, Any]]
) -> pd.DataFrame:
    """`PickedUp` boost pad events with pad size and position from `fd["boost_pads"]`.

    `Available` (pad respawn) events are skipped.
    """
    by_id = {p["pad_id"]: p for p in pads}
    rows = []
    for e in pad_events:
        kind = e.get("kind")
        if not (isinstance(kind, dict) and "PickedUp" in kind):
            continue
        pad = by_id.get(e["pad_id"])
        x, y, z = _xyz(pad["position"] if pad else None)
        rows.append(
            {
                "frame": int(e["frame"]),
                "type": "boost_pickup",
                "player_key": _key(e.get("player")),
                "pad_id": e["pad_id"],
                "pad_size": pad["size"].lower() if pad else None,
                "x": x,
                "y": y,
                "z": z,
            }
        )
    return pd.DataFrame(rows)


def stat_events(player_stats: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Shot / Save / Assist from `fd["player_stat_events"]`.

    These are the game's own stat credits. The frame is when the stat was credited, which
    trails the touch that earned it (by a variable amount; see ROADMAP 1.3 Notes).
    """
    rows = []
    for e in player_stats:
        kind = STAT_KINDS.get(e.get("kind"))
        if kind is None:
            continue
        shot = e.get("shot") or {}
        crossing = shot.get("projected_goal_line_crossing") or {}
        rows.append(
            {
                "frame": int(e["frame"]),
                "type": kind,
                "player_key": _key(e.get("player")),
                "team": 0 if e["is_team_0"] else 1,
                "shot_on_target": crossing.get("inside_goal_mouth") if kind == "shot" else None,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- assembly


def _as_dtype(col: pd.Series, dtype: str) -> pd.Series:
    """Cast a mixed object column (values, None, NaN, pd.NA) to a nullable dtype."""
    if dtype == "float64":
        return pd.to_numeric(col, errors="raise").astype("float64")
    if dtype in ("Int8", "Int16", "Int64", "boolean"):
        return pd.Series(col.astype(object).where(col.notna(), None), index=col.index).astype(dtype)
    return col.astype(object).where(col.notna(), None).astype(dtype)


def finalize_events(parsed: ParsedReplay, parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-type frames, fill common columns, fix dtypes, sort."""
    parts = [p for p in parts if len(p)]
    if parts:
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["frame", "type"])
    for col in EVENT_COLUMNS:
        if col not in df:
            df[col] = pd.NA

    df["frame"] = df["frame"].astype(np.int64)
    info = parsed.frames.set_index("frame").loc[
        df["frame"], ["time", "clock", "is_overtime", "run", "is_live", "game_state"]
    ]
    df["time"] = info["time"].to_numpy()
    df["clock"] = info["clock"].to_numpy()
    df["is_overtime"] = info["is_overtime"].to_numpy()
    df["run"] = info["run"].to_numpy()
    pickups = df["type"] == "boost_pickup"
    df.loc[pickups, "is_live"] = info["is_live"].to_numpy()[pickups.to_numpy()]

    # Team from the roster when the event didn't carry one.
    team_of = dict(zip(parsed.players["player_key"], parsed.players["team"], strict=True))
    missing = df["team"].isna() & df["player_key"].notna()
    df.loc[missing, "team"] = df.loc[missing, "player_key"].map(team_of)

    # Ball position where the event is about the ball; speed only for stat events (at the goal
    # frame the ball has already stopped in the net, so its speed there means nothing).
    ball = parsed.entities[parsed.entities["entity"] == "ball"].set_index("frame")
    about_ball = df["type"].isin(["goal", "shot", "save", "assist"]).to_numpy()
    if about_ball.any():
        b = ball.loc[df.loc[about_ball, "frame"]]
        df.loc[about_ball, ["x", "y", "z"]] = b[["x", "y", "z"]].to_numpy()
    stat = df["type"].isin(list(STAT_KINDS.values())).to_numpy()
    if stat.any():
        v = ball.loc[df.loc[stat, "frame"], ["vx", "vy", "vz"]].to_numpy(dtype=float)
        df.loc[stat, "ball_speed"] = np.linalg.norm(v, axis=1)

    df["type"] = pd.Categorical(df["type"], categories=EVENT_TYPES)
    dtypes = {"player_key": "string", "team": "Int8", "x": "float64", "y": "float64"}
    for col, dtype in (dtypes | {"z": "float64"} | EXTRA_COLUMNS).items():
        df[col] = _as_dtype(df[col], dtype)
    df = df.sort_values(["frame", "type"], kind="stable", ignore_index=True)
    return df[EVENT_COLUMNS]


def build_events(
    parsed: ParsedReplay,
    fd: dict[str, Any],
    touches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Events table from `parsed` (1.2 tables) and `sa.get_replay_frames_data(path)`.

    `touches` (optional, same column layout) is merged in once the touch rule exists.
    """
    parts = [
        kickoff_events(parsed),
        goal_events(fd["goal_events"]),
        demo_events(fd["demolish_infos"]),
        pickup_events(fd["boost_pad_events"], fd["boost_pads"]),
        stat_events(fd.get("player_stat_events", [])),
    ]
    if touches is not None:
        parts.append(touches)
    return finalize_events(parsed, parts)
