"""Print the events table of a replay (ROADMAP 1.3), for checking by eye.

    uv run python scripts/events.py replays/<file>.replay                 # summary
    uv run python scripts/events.py replays/<file>.replay --clock 3:47    # events near 3:47
    uv run python scripts/events.py replays/<file>.replay --type touch --type save

`--clock` shows every event within `--window` seconds (default 2) of that clock, so a save
credited a second after its touch still shows up. Players are shown by name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from rldr.parse.events import EVENT_TYPES
from rldr.parse.frames import format_clock, load_replay, parse_clock

TEAMS = {0: "Blue", 1: "Orange"}


def summary(events: pd.DataFrame) -> str:
    lines = ["counts: " + ", ".join(f"{t} {(events['type'] == t).sum()}" for t in EVENT_TYPES)]
    t = events[events["type"] == "touch"]
    for col in ("touch_source", "confidence", "contest_type"):
        counts = t[col].fillna("-").value_counts().to_dict()
        lines.append(f"touches by {col}: {counts}")
    lines.append(f"touches with source_conflict: {int(t['source_conflict'].sum())}")
    return "\n".join(lines)


def detail(row: pd.Series, name) -> str:
    typ = row["type"]
    if typ == "touch":
        alt = f" vs {name(row['alt_player_key'])}" if pd.notna(row["alt_player_key"]) else ""
        tag = row["contest_type"] if pd.notna(row["contest_type"]) else ""
        tag += " conflict" if row["source_conflict"] else ""
        return (
            f"{row['confidence']:<6} {row['touch_source']:<7} "
            f"ball {row['ball_speed_before']:>4.0f}->{row['ball_speed_after']:>4.0f} "
            f"car {row['car_ball_dist']:>3.0f}uu {tag}{alt}".rstrip()
        )
    if typ == "goal":
        return f"score {row['team0_score']}-{row['team1_score']}"
    if typ == "demo":
        return f"victim {name(row['other_player_key'])}"
    if typ == "kickoff":
        return f"countdown {row['countdown_frame']}, first touch {row['first_touch_frame']}"
    if typ == "boost_pickup":
        return f"{row['pad_size']} pad {row['pad_id']}" + ("" if row["is_live"] else " (dead)")
    if typ in ("shot", "save", "assist"):
        touch = f"touch {row['touch_frame']}" if pd.notna(row["touch_frame"]) else ""
        target = " on target" if row["shot_on_target"] is True else ""
        return f"ball {row['ball_speed']:.0f}{target} {touch}".rstrip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("replay", type=Path)
    parser.add_argument("--clock", help="in-game clock, M:SS or +M:SS")
    parser.add_argument("--window", type=int, default=2, help="seconds around --clock")
    parser.add_argument("--type", action="append", choices=EVENT_TYPES, help="repeatable")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # player names can be non-ASCII on Windows

    replay = load_replay(args.replay)
    ev = replay.events
    names = dict(zip(replay.players["player_key"], replay.players["name"], strict=True))

    def name(k) -> str:
        return names.get(k, "") if pd.notna(k) else ""

    print(f"{replay.meta['replay_id']}  {replay.meta['map']}")
    print(summary(ev))
    if args.clock is None and not args.type:
        return
    if args.clock is not None:
        secs, ot = parse_clock(args.clock)
        ev = ev[(ev["is_overtime"] == ot) & ((ev["clock"] - secs).abs() <= args.window)]
    if args.type:
        ev = ev[ev["type"].isin(args.type)]
    print()
    for _, r in ev.iterrows():
        team = TEAMS.get(int(r["team"]), "") if pd.notna(r["team"]) else ""
        print(
            f"{r['frame']:>6} {format_clock(r['clock'], r['is_overtime']):>6} "
            f"{r['type']:<12} {name(r['player_key']):<16} {team:<6} {detail(r, name)}"
        )


if __name__ == "__main__":
    main()
