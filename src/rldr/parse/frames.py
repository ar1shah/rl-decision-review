"""Normalized frame tables built from a replay (ROADMAP 1.2).

Source is `subtr_actor.get_replay_frames_data` at the native network-frame rate (~30 Hz),
so the `frame` index here is the same raw frame index every event (goals, demos, touches)
refers to. See ROADMAP 1.1 / 1.2 Notes for the data quirks handled below.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

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

    Yaw is verified against kickoff spawns (cars face the ball). Pitch/roll signs are
    standard-math conventions and not yet verified against Rocket League.
    """
    qw, qx, qy, qz = (np.asarray(q, dtype=float) for q in (qw, qx, qy, qz))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
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
