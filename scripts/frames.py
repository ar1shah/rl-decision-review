"""Print ball and car state at an in-game clock time (ROADMAP 1.2), for checking by eye.

    uv run python scripts/frames.py replays/<file>.replay --clock 3:06
    uv run python scripts/frames.py replays/<file>.replay --clock +0:30   # overtime

One clock second can hold several separate stretches of play (after a goal the kickoff
restarts at the same clock), so each stretch ("run") is printed on its own, at its first and
last live frame.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from rldr.parse.frames import format_clock, load_replay

TEAMS = {0: "Blue", 1: "Orange"}


def describe(rows) -> str:
    lines = [f"  {'name':<16} {'team':<6} {'x':>7} {'y':>7} {'z':>6} {'speed':>6} {'boost':>5}"]
    for _, r in rows.iterrows():
        team = "" if r["entity"] == "ball" else TEAMS[int(r["team"])]
        if r["is_demolished"]:
            lines.append(f"  {r['name']:<16} {team:<6} {'(demolished)':>36}")
            continue
        if not r["is_present"]:
            lines.append(f"  {r['name']:<16} {team:<6} {'(not on field)':>36}")
            continue
        speed = math.hypot(r["vx"], r["vy"], r["vz"])
        boost = "" if r["entity"] == "ball" else f"{r['boost']:.0f}"
        lines.append(
            f"  {r['name']:<16} {team:<6} {r['x']:>7.0f} {r['y']:>7.0f} {r['z']:>6.0f} "
            f"{speed:>6.0f} {boost:>5}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("replay", type=Path)
    parser.add_argument("--clock", required=True, help="in-game clock, M:SS or +M:SS")
    parser.add_argument("--all-states", action="store_true", help="include non-live frames")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # player names can be non-ASCII on Windows

    replay = load_replay(args.replay)
    rows = replay.at_clock(args.clock, live_only=not args.all_states)
    print(f"{replay.meta['replay_id']}  {replay.meta['map']}  clock {args.clock}")
    if rows.empty:
        print("  no frames at that clock")
        return
    for run, group in rows.groupby("run"):
        for frame in sorted({group["frame"].min(), group["frame"].max()}):
            at = group[group["frame"] == frame]
            first = at.iloc[0]
            clock = format_clock(first["clock"], first["is_overtime"])
            print(
                f"\nrun {run}  frame {frame}  t={first['time']:.2f}s  clock {clock}  "
                f"{first['game_state']}"
            )
            print(describe(at))


if __name__ == "__main__":
    main()
