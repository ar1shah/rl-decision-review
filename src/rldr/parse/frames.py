"""Normalized frame tables built from a replay (ROADMAP 1.2).

Source is `subtr_actor.get_replay_frames_data` at the native network-frame rate (~30 Hz),
so the `frame` index here is the same raw frame index every event (goals, demos, touches)
refers to. See ROADMAP 1.1 / 1.2 Notes for the data quirks handled below.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import subtr_actor as sa

BOOST_RAW_MAX = 255.0
LIVE_STATE = "Active"
COUNTDOWN_STATE = "Countdown"
OVERTIME_ATTRIBUTE = "TAGame.GameEvent_Soccar_TA:bOverTime"


# --------------------------------------------------------------------------- small helpers


def boost_to_pct(raw: Any) -> Any:
    """Convert replay boost (0-255) to 0-100. NaN passes through; scalars stay scalars."""
    out = np.asarray(raw, dtype=float) * 100.0 / BOOST_RAW_MAX
    return float(out) if out.ndim == 0 else out


def player_key(remote_id: dict[str, Any]) -> str:
    """Stable player key from a platform-tagged remote id, e.g. 'Steam:7656...'.

    PlayStation ids are dicts ({name, online_id, ...}); their `online_id` is the stable part.
    """
    if not isinstance(remote_id, dict) or len(remote_id) != 1:
        raise ValueError(f"unexpected remote id: {remote_id!r}")
    ((platform, value),) = remote_id.items()
    if isinstance(value, dict):
        value = value.get("online_id") or value.get("name")
    return f"{platform}:{value}"


def platform_of(key: str) -> str:
    """Platform prefix of a player key: Steam / Epic / Xbox / PlayStation / ..."""
    return key.split(":", 1)[0]


def parse_clock(text: str) -> tuple[int, bool]:
    """'2:10' -> (130, False); '+0:30' -> (30, True) for overtime (NOTES.md convention)."""
    s = text.strip()
    overtime = s.startswith("+")
    minutes, seconds = s.lstrip("+").split(":")
    if not 0 <= int(seconds) < 60:
        raise ValueError(f"bad clock {text!r}")
    return int(minutes) * 60 + int(seconds), overtime


def format_clock(clock: int, is_overtime: bool = False) -> str:
    """Inverse of parse_clock: (130, False) -> '2:10'; (30, True) -> '+0:30'."""
    c = int(clock)
    return f"{'+' if is_overtime else ''}{c // 60}:{c % 60:02d}"


def quat_to_euler(qw: Any, qx: Any, qy: Any, qz: Any) -> tuple[Any, Any, Any]:
    """Quaternion (w, x, y, z) -> (yaw, pitch, roll) in radians, ZYX order.

    Physical meaning (verified on wall-driving frames, ROADMAP 1.5):
      yaw   = heading in the x-y plane, atan2(forward_y, forward_x); kickoff cars face the ball
      pitch > 0 = nose up (forward_z = sin(pitch)); flipped from the standard ZYX sign
      roll  > 0 = right side down (car on the -x wall facing +y has roll ~ +90 deg)
    The quaternion itself rotates the car's local (forward, left, up) axes into world space.
    """
    qw, qx, qy, qz = (np.asarray(q, dtype=float) for q in (qw, qx, qy, qz))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    pitch = -np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    return yaw, pitch, roll


# --------------------------------------------------------------------------- demos


def demolished_mask(
    n_frames: int, demo_frames: Sequence[int], has_data: np.ndarray, sleeping: np.ndarray
) -> np.ndarray:
    """True from each demo frame until the car respawns.

    Respawn = first later frame with real data and `sleeping == False`. While demolished the
    car's rigid body is frozen at the demo spot with `sleeping == True`, sometimes followed by
    `Empty` frames (all inside the window).
    """
    mask = np.zeros(n_frames, dtype=bool)
    alive = np.asarray(has_data, dtype=bool) & ~np.asarray(sleeping, dtype=bool)
    for f in demo_frames:
        later = np.flatnonzero(alive[f + 1 :])
        end = f + 1 + int(later[0]) if len(later) else n_frames
        mask[f:end] = True
    return mask


def presence_mask(has_data: np.ndarray, demolished: np.ndarray) -> np.ndarray:
    """A car is present unless its frame is Empty outside a demo window.

    Empty-and-not-demolished happens at kickoff resets (and when a player has left).
    Use `is_present & ~is_demolished` for "this car is actually on the field".
    """
    return np.asarray(has_data, dtype=bool) | np.asarray(demolished, dtype=bool)


# --------------------------------------------------------------------------- clock / overtime


def overtime_flag_frame(
    network_frames: Sequence[dict[str, Any]], objects: Sequence[str]
) -> int | None:
    """First frame where the replicated `bOverTime` attribute is set true, if ever."""
    ids = {i for i, name in enumerate(objects) if name == OVERTIME_ATTRIBUTE}
    if not ids:
        return None
    for k, frame in enumerate(network_frames):
        for update in frame.get("updated_actors", ()):
            if update.get("object_id") in ids and update.get("attribute") == {"Boolean": True}:
                return k
    return None


def overtime_fallback_frame(
    seconds_remaining: np.ndarray,
    states: np.ndarray,
    goal_frames: Sequence[int],
    goal_team_zero: Sequence[bool],
) -> int | None:
    """Fallback OT start when no bOverTime flag exists (unverified on a real OT replay).

    The first Countdown after the clock reaches 0, if the score is tied at that point.
    """
    zero = np.flatnonzero(np.asarray(seconds_remaining) == 0)
    if not len(zero):
        return None
    after = np.flatnonzero(np.asarray(states[zero[0] :]) == COUNTDOWN_STATE)
    if not len(after):
        return None
    start = int(zero[0] + after[0])
    blue = sum(1 for f, t0 in zip(goal_frames, goal_team_zero, strict=True) if f < start and t0)
    orange = sum(
        1 for f, t0 in zip(goal_frames, goal_team_zero, strict=True) if f < start and not t0
    )
    return start if blue == orange else None


def compute_clock(
    seconds_remaining: np.ndarray,
    states: np.ndarray,
    times: np.ndarray,
    overtime_start: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (clock, is_overtime).

    Regulation: clock = SecondsRemaining (what the in-game clock shows).
    Overtime (from `overtime_start`): clock = whole seconds elapsed since the OT kickoff went
    live, frozen outside live play. If the raw counter itself counts up during OT it is used
    as-is; otherwise elapsed time is derived from timestamps. Unverified on a real OT replay.
    """
    secs = np.asarray(seconds_remaining, dtype=np.int64)
    states = np.asarray(states, dtype=object)
    times = np.asarray(times, dtype=float)
    n = len(secs)
    clock = secs.copy()
    is_ot = np.zeros(n, dtype=bool)
    if overtime_start is None:
        return clock, is_ot

    is_ot[overtime_start:] = True
    raw_ot = secs[overtime_start:]
    if len(raw_ot) > 1 and np.all(np.diff(raw_ot) >= 0) and raw_ot[-1] > raw_ot[0]:
        return clock, is_ot  # raw counter counts up in OT: already elapsed seconds

    live = is_ot & (states == LIVE_STATE)
    elapsed = np.zeros(n)
    if live.any():
        t0 = times[int(np.argmax(live))]
        running = np.where(live, times - t0, np.nan)
        # Freeze outside live play: carry the last live value forward, 0 before the kickoff.
        idx = np.where(live, np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        running = np.where(np.isnan(running), running[idx], running)
        elapsed = np.nan_to_num(running, nan=0.0)
    clock[is_ot] = np.floor(np.maximum(elapsed[is_ot], 0.0)).astype(np.int64)
    return clock, is_ot


def clock_run(clock: np.ndarray, is_overtime: np.ndarray, is_live: np.ndarray) -> np.ndarray:
    """Per-frame index of the stretch of live play within one clock second (0, 1, ...).

    After a goal the kickoff restarts at the same clock, which stays frozen until the kickoff
    touch, so one clock second can hold several separate stretches of live play. Run 0 is the
    first stretch in that second, run 1 the next (e.g. the kickoff after a goal). Dead frames
    (countdown, post-goal) keep the run of the last stretch before them (0 if none yet).
    """
    clock = np.asarray(clock)
    ot = np.asarray(is_overtime, dtype=bool)
    live = np.asarray(is_live, dtype=bool)
    n = len(clock)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    new_group = np.r_[True, (clock[1:] != clock[:-1]) | (ot[1:] != ot[:-1])]
    prev_live = np.r_[False, live[:-1]]
    starts = live & (~prev_live | new_group)
    group_id = np.cumsum(new_group) - 1
    total = np.cumsum(starts)
    before_group = np.r_[0, total[:-1]][new_group][group_id]  # starts before each group
    return np.maximum(total - before_group - 1, 0).astype(np.int64)


# --------------------------------------------------------------------------- tables

CAR_FLAGS = (
    "boost_active",
    "jump_active",
    "dodge_active",
    "double_jump_active",
    "powerslide_active",
)
ENTITY_COLUMNS = (
    ["frame", "entity", "team"]
    + ["x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "qw", "qx", "qy", "qz"]
    + ["yaw", "pitch", "roll", "boost", *CAR_FLAGS, "sleeping", "is_present", "is_demolished"]
)
# Car state that is meaningless while demolished (frozen at the demo spot, boost draining).
_DEMO_NULLED = ("pos", "vel", "ang", "quat", "boost")


@dataclass
class ParsedReplay:
    """Normalized tables for one replay.

    frames   : one row per raw network frame (time, clock, game state, is_live, is_overtime,
               run)
    entities : long format, one row per (frame, entity); entity is "ball" or a player key
    players  : player_key, name, team, platform, car
    meta     : replay-level facts (map, team size, score, forfeit, overtime source)
    events   : discrete events (goals, demos, kickoffs, pickups, ...); see rldr.parse.events.
               Filled by load_replay; None when built with build_replay alone.
    """

    frames: pd.DataFrame
    entities: pd.DataFrame
    players: pd.DataFrame
    meta: dict[str, Any]
    events: pd.DataFrame | None = None

    def at_clock(self, clock: str, live_only: bool = True) -> pd.DataFrame:
        """Entity rows (plus frame info and player name) for one clock second, e.g. '3:06'.

        The in-game clock shows each second for ~30 frames, so this returns every frame in
        that second. `live_only` drops countdown / post-goal frames that share the clock.

        One clock second can cover several separate stretches of play: after a goal the
        kickoff restarts at the same clock, and the clock stays frozen until the kickoff touch.
        The `run` column (0, 1, ...) numbers those live stretches in frame order, so run 0 of a
        goal's clock is the play before the goal and run 1 is the next kickoff (see clock_run).
        """
        seconds, overtime = parse_clock(clock)
        f = self.frames
        sel = f[(f["clock"] == seconds) & (f["is_overtime"] == overtime)]
        if live_only:
            sel = sel[sel["is_live"]]
        out = sel.merge(self.entities, on="frame", how="inner")
        names = dict(zip(self.players["player_key"], self.players["name"], strict=True))
        names["ball"] = "ball"
        out.insert(out.columns.get_loc("entity") + 1, "name", out["entity"].astype(str).map(names))
        return out


def _variant_data(frame: Any) -> dict[str, Any] | None:
    """The payload of a serialized `{"Data": {...}}` frame, or None for "Empty" etc."""
    return frame.get("Data") if isinstance(frame, dict) else None


def _vec3(v: dict[str, float] | None) -> tuple[float, float, float]:
    return (np.nan, np.nan, np.nan) if v is None else (v["x"], v["y"], v["z"])


def _rigid_bodies(frames: Sequence[Any]) -> dict[str, np.ndarray]:
    """Rigid body arrays for one entity's frames. Empty frames and null fields become NaN."""
    n = len(frames)
    out = {
        "pos": np.full((n, 3), np.nan),
        "vel": np.full((n, 3), np.nan),
        "ang": np.full((n, 3), np.nan),
        "quat": np.full((n, 4), np.nan),
        "sleeping": np.zeros(n, dtype=bool),
        "has_data": np.zeros(n, dtype=bool),
    }
    for i, frame in enumerate(frames):
        data = _variant_data(frame)
        if data is None:
            continue
        rb = data["rigid_body"]
        out["has_data"][i] = True
        out["pos"][i] = _vec3(rb["location"])
        out["vel"][i] = _vec3(rb.get("linear_velocity"))
        out["ang"][i] = _vec3(rb.get("angular_velocity"))
        r = rb.get("rotation")
        if r is not None:
            out["quat"][i] = (r["w"], r["x"], r["y"], r["z"])
        out["sleeping"][i] = bool(rb.get("sleeping"))
    return out


def _nullable_bool(values: np.ndarray, valid: np.ndarray) -> pd.arrays.BooleanArray:
    """Boolean column that is NA where `valid` is False."""
    return pd.arrays.BooleanArray(np.asarray(values, dtype=bool), ~np.asarray(valid, dtype=bool))


def _entity_frame(
    entity: str, team: int | None, rb: dict[str, np.ndarray], extra: dict[str, Any]
) -> pd.DataFrame:
    n = len(rb["has_data"])
    yaw, pitch, roll = quat_to_euler(*rb["quat"].T)
    cols: dict[str, Any] = {
        "frame": np.arange(n, dtype=np.int64),
        "entity": [entity] * n,
        "team": pd.array([team] * n, dtype="Int8"),
    }
    for key, names in (
        ("pos", ("x", "y", "z")),
        ("vel", ("vx", "vy", "vz")),
        ("ang", ("wx", "wy", "wz")),
        ("quat", ("qw", "qx", "qy", "qz")),
    ):
        for j, col in enumerate(names):
            cols[col] = rb[key][:, j]
    cols |= {"yaw": yaw, "pitch": pitch, "roll": roll}
    cols |= extra
    return pd.DataFrame(cols)


def _ball_entity(ball_frames: Sequence[Any]) -> pd.DataFrame:
    rb = _rigid_bodies(ball_frames)
    # Velocities are null while the ball sleeps (kickoff spawn, post-goal): it isn't moving.
    for key in ("vel", "ang"):
        rb[key][rb["has_data"] & np.isnan(rb[key]).any(axis=1)] = 0.0
    n = len(ball_frames)
    no_flags = np.zeros(n, dtype=bool)
    extra = {
        "boost": np.full(n, np.nan),
        **{flag: _nullable_bool(no_flags, no_flags) for flag in CAR_FLAGS},
        "sleeping": _nullable_bool(rb["sleeping"], rb["has_data"]),
        "is_present": rb["has_data"],
        "is_demolished": np.zeros(n, dtype=bool),
    }
    return _entity_frame("ball", None, rb, extra)


def _car_entity(
    key: str, team: int, car_frames: Sequence[Any], demo_frames: Sequence[int]
) -> pd.DataFrame:
    n = len(car_frames)
    rb = _rigid_bodies(car_frames)
    demolished = demolished_mask(n, demo_frames, rb["has_data"], rb["sleeping"])
    present = presence_mask(rb["has_data"], demolished)
    valid = rb["has_data"] & ~demolished  # frames whose recorded car state we trust

    boost_raw = np.full(n, np.nan)
    flags = {flag: np.zeros(n, dtype=bool) for flag in CAR_FLAGS}
    for i, frame in enumerate(car_frames):
        data = _variant_data(frame)
        if data is None:
            continue
        boost_raw[i] = data.get("boost_amount", np.nan)
        for flag in CAR_FLAGS:
            flags[flag][i] = bool(data.get(flag))
    rb["boost"] = boost_to_pct(boost_raw)
    for field in _DEMO_NULLED:
        rb[field][demolished] = np.nan

    extra = {
        "boost": rb["boost"],
        **{flag: _nullable_bool(v, valid) for flag, v in flags.items()},
        "sleeping": _nullable_bool(rb["sleeping"], rb["has_data"]),
        "is_present": present,
        "is_demolished": demolished,
    }
    return _entity_frame(key, team, rb, extra)


def _players_table(meta: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for team, side in ((0, "team_zero"), (1, "team_one")):
        for p in meta[side]:
            key = player_key(p["remote_id"])
            rows.append(
                {
                    "player_key": key,
                    "name": p["name"],
                    "team": team,
                    "platform": platform_of(key),
                    "car": p.get("car_hitbox_family"),
                }
            )
    players = pd.DataFrame(rows, columns=["player_key", "name", "team", "platform", "car"])
    return players.astype({"team": "int8"})


def boost_pads(pads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """`fd["boost_pads"]` as plain JSON rows {pad_id, x, y, z, size ("big" / "small")}.

    Sorted by (y, x) so the order doesn't depend on the replay's actor ids.
    """
    rows = [
        {
            "pad_id": p["pad_id"],
            "x": float(p["position"]["x"]),
            "y": float(p["position"]["y"]),
            "z": float(p["position"]["z"]),
            "size": str(p["size"]).lower(),
        }
        for p in pads
    ]
    return sorted(rows, key=lambda r: (r["y"], r["x"]))


def build_replay(replay_id: str, raw: dict[str, Any], fd: dict[str, Any]) -> ParsedReplay:
    """Build the tables from `sa.parse_replay(bytes)` and `sa.get_replay_frames_data(path)`."""
    names = raw["names"]
    md = fd["frame_data"]["metadata_frames"]
    n = len(md)
    times = np.array([m["time"] for m in md], dtype=float)
    secs = np.array([m["seconds_remaining"] for m in md], dtype=np.int64)
    state_ids = [m["replicated_game_state_name"] for m in md]
    states = np.array([names[s] if s is not None else "Unknown" for s in state_ids], dtype=object)

    goals = fd["goal_events"]
    ot_start = overtime_flag_frame(raw["network_frames"]["frames"], raw["objects"])
    ot_source = "flag" if ot_start is not None else None
    if ot_start is None:
        ot_start = overtime_fallback_frame(
            secs,
            states,
            [g["frame"] for g in goals],
            [g["scoring_team_is_team_0"] for g in goals],
        )
        ot_source = "fallback" if ot_start is not None else None
    clock, is_ot = compute_clock(secs, states, times, ot_start)
    is_live = states == LIVE_STATE

    frames = pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.int64),
            "time": times,
            "seconds_remaining": secs,
            "clock": clock,
            "game_state": pd.Categorical(states.astype(str)),
            "is_live": is_live,
            "is_overtime": is_ot,
            "run": clock_run(clock, is_ot, is_live),
        }
    )

    players = _players_table(fd["meta"])
    team_of = dict(zip(players["player_key"], players["team"], strict=True))
    demos: dict[str, list[int]] = {}
    for d in fd["demolish_infos"]:
        demos.setdefault(player_key(d["victim"]), []).append(int(d["frame"]))

    parts = [_ball_entity(fd["frame_data"]["ball_data"]["frames"])]
    for remote_id, pdata in fd["frame_data"]["players"]:
        key = player_key(remote_id)
        if key not in team_of:  # not in the roster meta: use the per-frame team flag
            first = next(d for d in map(_variant_data, pdata["frames"]) if d is not None)
            team_of[key] = 0 if first["is_team_0"] else 1
        parts.append(_car_entity(key, int(team_of[key]), pdata["frames"], demos.get(key, [])))
    for part in parts:
        if len(part) != n:
            raise ValueError(f"{part['entity'].iloc[0]}: {len(part)} frames, expected {n}")

    order = ["ball", *players["player_key"]]
    order += sorted({p["entity"].iloc[0] for p in parts} - set(order))
    entities = pd.concat(parts, ignore_index=True)
    entities["entity"] = pd.Categorical(entities["entity"], categories=order)
    entities = entities.sort_values(["frame", "entity"], kind="stable", ignore_index=True)
    entities = entities[ENTITY_COLUMNS]

    props = raw["properties"]
    meta = {
        "replay_id": replay_id,
        "map": props.get("MapName"),
        "team_size": props.get("TeamSize"),
        "forfeit": bool(props.get("bForfeit", False)),
        "team0_score": props.get("Team0Score", 0),
        "team1_score": props.get("Team1Score", 0),
        "n_frames": n,
        "overtime_source": ot_source,
        "boost_pads": boost_pads(fd.get("boost_pads", [])),
    }
    return ParsedReplay(frames=frames, entities=entities, players=players, meta=meta)


def _parse_replay(path: Path) -> ParsedReplay:
    """Parse a .replay file from scratch (no cache), including events."""
    from rldr.parse.events import build_events  # events builds on this module

    raw = sa.parse_replay(path.read_bytes())
    fd = sa.get_replay_frames_data(str(path))
    parsed = build_replay(path.stem, raw, fd)
    parsed.events = build_events(parsed, fd, sa.get_stats_events(str(path))["events"])
    return parsed


def load_replay(
    path: str | Path, use_cache: bool = True, cache_root: str | Path | None = None
) -> ParsedReplay:
    """Normalized tables for a .replay file (see ParsedReplay), including events.

    With `use_cache` (default), reads `data/<replay_id>/` when it is valid for this replay file
    and this version of the parsing code, otherwise parses and (re)writes it; see
    rldr.parse.cache. `use_cache=False` always parses and never touches the cache.
    """
    path = Path(path)
    if not use_cache:
        return _parse_replay(path)
    from rldr.parse import cache  # cache imports this module

    root = cache.DEFAULT_CACHE_ROOT if cache_root is None else Path(cache_root)
    digest = cache.replay_hash(path)
    cached = cache.read_cache(path.stem, digest, root)
    if cached is not None:
        return cached
    parsed = _parse_replay(path)
    cache.write_cache(parsed, digest, root)
    return parsed
