"""Explore raw subtr-actor-py output for one or more replays (ROADMAP 1.1).

Read-only: prints what the parser gives us and where each thing lives. Writes nothing.

    uv run python scripts/explore.py                      # every .replay in replays/
    uv run python scripts/explore.py replays/<file>.replay
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import subtr_actor as sa

REPLAY_DIR = Path(__file__).resolve().parent.parent / "replays"

# Goal clocks Ari wrote down in replays/NOTES.md: (clock "M:SS", scorer name in replay data).
EXPECTED_GOALS: dict[str, list[tuple[str, str]]] = {
    "2C980F9741798CF5E4D1A0BFF902C275": [
        ("3:06", "Degnis98"),
        ("1:35", "Degnis98"),
        ("0:54", "CohenLovesJill"),
        ("0:38", "ari ツ"),
        ("0:02", "CohenLovesJill"),
    ],
    "0AACF51B43A578E00DD400BD009DF81E": [
        ("4:54", "NickyFuego1406"),
        ("4:37", "iDrxp6442"),
        ("3:43", "iDrxp6442"),
        ("1:49", "iDrxp6442"),
        ("1:38", "ari ツ"),
    ],
}

# Every feature adder name accepted by get_column_headers in subtr-actor-py 1.5.1. The library
# has no listing function, so these were found by probing; printed with their columns below.
GLOBAL_ADDERS = [
    "BallRigidBody",
    "BallRigidBodyBasis",
    "BallRigidBodyQuaternions",
    "BallRigidBodyNoVelocities",
    "VelocityAddedBallRigidBodyNoVelocities",
    "InterpolatedBallRigidBodyNoVelocities",
    "BallHasBeenHit",
    "SecondsRemaining",
    "CurrentTime",
    "FrameTime",
    "ReplicatedStateName",
    "ReplicatedGameStateTimeRemaining",
    "ThreatModelValues",
    "ThreatModelFeatures",
    "ThreatEntityFeatures",
]
PLAYER_ADDERS = [
    "PlayerRigidBody",
    "PlayerRigidBodyBasis",
    "PlayerRigidBodyQuaternions",
    "PlayerRigidBodyNoVelocities",
    "VelocityAddedPlayerRigidBodyNoVelocities",
    "InterpolatedPlayerRigidBodyNoVelocities",
    "PlayerBoost",
    "PlayerJump",
    "PlayerAnyJump",
    "PlayerDodgeRefreshed",
    "PlayerDemolishedBy",
    "PlayerBallDistance",
    "PlayerRelativeBallPosition",
    "PlayerRelativeBallVelocity",
    "PlayerLocalRelativeBallPosition",
    "PlayerLocalRelativeBallVelocity",
]

# The ndarray layout used for the row-count and interpolation checks.
ND_GLOBAL = ["BallRigidBody", "CurrentTime", "SecondsRemaining", "ReplicatedStateName"]
ND_PLAYER = ["PlayerRigidBody", "PlayerBoost", "PlayerDemolishedBy"]


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def short(obj: Any, n: int = 240) -> str:
    s = json.dumps(obj, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + " ..."


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def player_key(pid: Any) -> str:
    """Flatten a platform-tagged player id ({"Steam": "..."}, {"PlayStation": {...}}) to text."""
    if pid is None:
        return "None"
    (platform, value), *_ = pid.items()
    if isinstance(value, dict):
        value = value.get("online_id") or value.get("name") or short(value, 60)
    return f"{platform}:{value}"


def variant(frame: Any) -> str:
    """Rust enum variant name of a serialized frame: 'Data', 'Empty', ..."""
    return next(iter(frame)) if isinstance(frame, dict) else str(frame)


def explore(path: Path) -> None:
    replay_id = path.stem
    print("\n" + "#" * 78 + f"\n# {path.name}\n" + "#" * 78)

    raw = sa.parse_replay(path.read_bytes())
    props = raw["properties"]
    fd = sa.get_replay_frames_data(path)
    meta = fd["meta"]

    # Map flattened player id -> (name, team) from the frames_data meta.
    roster: dict[str, tuple[str, int]] = {}
    for team, key in ((0, "team_zero"), (1, "team_one")):
        for p in meta[key]:
            roster[player_key(p["remote_id"])] = (p["name"], team)

    def who(pid: Any) -> str:
        return roster.get(player_key(pid), (player_key(pid), -1))[0]

    # ---------------------------------------------------------------- 1. metadata
    section("1. Metadata (parse_replay()['properties'] + get_replay_frames_data()['meta'])")
    print(f"map           : {props.get('MapName')}")
    print(
        f"mode          : {props.get('TeamSize')}v{props.get('TeamSize')}  "
        f"game_type={meta['game_type']}"
    )
    print(
        f"score         : Blue(team 0) {props.get('Team0Score')} - "
        f"{props.get('Team1Score')} Orange(team 1)"
    )
    print(f"bForfeit      : {props.get('bForfeit')}")
    print(
        f"TotalSecondsPlayed={props.get('TotalSecondsPlayed')}  "
        f"NumFrames={props.get('NumFrames')}  "
        f"RecordFPS={props.get('RecordFPS')}"
    )
    print("header keys   :", sorted(props))
    print("players (PlayerStats):")
    for ps in props.get("PlayerStats", []):
        plat = ps.get("Platform", {}).get("value")
        print(
            f"  team {ps['Team']}  {ps['Name']:<16} {plat:<24} goals={ps['Goals']} "
            f"assists={ps['Assists']} saves={ps['Saves']} shots={ps['Shots']} score={ps['Score']}"
        )
    print("frames_data meta players (name, car, remote_id):")
    for key in ("team_zero", "team_one"):
        for p in meta[key]:
            print(
                f"  {key:<9} {p['name']:<16} {p.get('car_hitbox_family')!s:<8} "
                f"{player_key(p['remote_id'])}"
            )

    # ---------------------------------------------------------------- 2. features
    section("2. ndarray feature adders and their columns (get_column_headers)")
    for name in GLOBAL_ADDERS:
        cols = sa.get_column_headers(global_feature_adders=[name], player_feature_adders=[])
        print(f"  global {name:<40} {short(cols['global_headers'], 140)}")
    for name in PLAYER_ADDERS:
        cols = sa.get_column_headers(global_feature_adders=[], player_feature_adders=[name])
        print(f"  player {name:<40} {short(cols['player_headers'], 140)}")
    default = sa.get_column_headers()
    print(
        f"  defaults: global={default['global_headers'][:3]}... player={default['player_headers']}"
    )

    # ---------------------------------------------------------------- 3. frames
    section("3. Raw network frames and timing")
    net = raw["network_frames"]["frames"]
    t_raw = np.array([f["time"] for f in net])
    dt = np.diff(t_raw)
    md = fd["frame_data"]["metadata_frames"]
    t_md = np.array([m["time"] for m in md])
    duration = t_raw[-1] - t_raw[0]
    print(f"raw network frames : {len(net)}   (header NumFrames={props.get('NumFrames')})")
    print(f"time range         : {t_raw[0]:.3f} -> {t_raw[-1]:.3f} s  (duration {duration:.2f} s)")
    print(
        f"dt mean/min/max    : {dt.mean():.4f} / {dt.min():.4f} / {dt.max():.4f} s  "
        f"-> {1 / dt.mean():.1f} Hz (header RecordFPS={props.get('RecordFPS')})"
    )
    print(
        f"dt percentiles     : p1={np.percentile(dt, 1):.4f} p50={np.median(dt):.4f} "
        f"p99={np.percentile(dt, 99):.4f}"
    )
    names = raw["names"]
    gaps = np.where(dt > 0.2)[0]
    in_play = dt[dt <= 0.2]
    print(f"rate excluding gaps>0.2s: {1 / in_play.mean():.1f} Hz over {in_play.sum():.1f} s")
    print(f"gaps > 0.2 s       : {len(gaps)} totalling {dt[gaps].sum():.1f} s")
    for i in gaps:
        a, b = md[i], md[i + 1]
        print(
            f"   after raw frame {i}: {dt[i]:.2f} s at clock {clock(a['seconds_remaining'])}  "
            f"{names[a['replicated_game_state_name']]} -> {names[b['replicated_game_state_name']]}"
            f"  (goal replay not recorded)"
        )
    print(
        f"frames_data rows   : metadata={len(md)} "
        f"ball={len(fd['frame_data']['ball_data']['frames'])} "
        f"players={[len(p[1]['frames']) for p in fd['frame_data']['players']]}"
    )
    print(f"frames_data time == raw frame time for every frame: {np.allclose(t_md, t_raw)}")

    # ---------------------------------------------------------------- 4. row count
    section("4. Why the ndarray has fewer rows than raw frames")
    for fps in (None, 10.0, 30.0):
        nd_meta, nd = sa.get_ndarray_with_info_from_replay_filepath(
            path, global_feature_adders=ND_GLOBAL, player_feature_adders=ND_PLAYER, fps=fps
        )
        heads = nd_meta["column_headers"]["global_headers"]
        ct = nd[:, heads.index("current time")]
        step = np.diff(ct)
        eff = 10.0 if fps is None else fps
        print(
            f"  fps={fps!s:<5} rows={nd.shape[0]:>6} cols={nd.shape[1]}  "
            f"duration*fps={duration * eff:8.1f}  time {ct[0]:.3f}->{ct[-1]:.3f}  "
            f"step mean={step.mean():.4f} min={step.min():.4f} max={step.max():.4f}"
        )
    print("  -> fps defaults to 10.0: rows are a fixed-rate RESAMPLE of the whole replay span, not")
    print("     raw frames with some dropped. Countdowns and post-goal frames are kept.")

    # Interpolated or sample-and-hold? Compare resampled ball x/y to the raw frames around it.
    nd_meta, nd = sa.get_ndarray_with_info_from_replay_filepath(
        path, global_feature_adders=ND_GLOBAL, player_feature_adders=ND_PLAYER, fps=10.0
    )
    g = nd_meta["column_headers"]["global_headers"]
    ball = fd["frame_data"]["ball_data"]["frames"]
    bx = np.array(
        [b["Data"]["rigid_body"]["location"]["x"] if variant(b) == "Data" else np.nan for b in ball]
    )
    ct = nd[:, g.index("current time")]
    nx = nd[:, g.index("Ball - position x")]
    prev = nxt = between = other = nearest = 0
    for i in range(len(ct)):
        j = int(np.searchsorted(t_raw, ct[i], side="right")) - 1
        if j < 0 or j + 1 >= len(t_raw) or bx[j] == bx[j + 1]:
            continue  # ball not moving across this pair: can't tell hold from interpolation
        lo, hi = sorted((bx[j], bx[j + 1]))
        near = j if ct[i] - t_raw[j] <= t_raw[j + 1] - ct[i] else j + 1
        nearest += bool(np.isclose(nx[i], bx[near], atol=0.01))
        if np.isclose(nx[i], bx[j], atol=0.01):
            prev += 1
        elif np.isclose(nx[i], bx[j + 1], atol=0.01):
            nxt += 1
        elif lo <= nx[i] <= hi:
            between += 1
        else:
            other += 1
    print(
        f"  ball x at each 10fps row vs the raw frames around it (moving ball only): "
        f"= previous raw frame {prev}, = next raw frame {nxt}, interpolated {between}, "
        f"other {other}; = nearest-in-time raw frame {nearest}"
    )
    in_gap = before = after = 0
    for i in gaps:
        rows = np.where((ct > t_raw[i]) & (ct < t_raw[i + 1]))[0]
        in_gap += len(rows)
        before += int(np.sum(np.isclose(nx[rows], bx[i], atol=0.01)))
        after += int(np.sum(np.isclose(nx[rows], bx[i + 1], atol=0.01)))
    print(
        f"  10fps rows inside unrecorded goal-replay gaps: {in_gap} "
        f"(ball x equals frame before gap: {before}, frame after gap: {after})"
    )
    print(
        f"  ndarray 'current time' is the sample time; seconds remaining at row 0 = "
        f"{nd[0, g.index('seconds remaining')]}"
    )

    # ---------------------------------------------------------------- 5. where things live
    section("5. Where things live")
    print(
        "ball rigid body : get_replay_frames_data()['frame_data']['ball_data']['frames'][i]['Data']"
        "['rigid_body'] (location, rotation quat, linear/angular velocity)"
    )
    print(f"   variants      : {dict(Counter(variant(b) for b in ball))}")
    print("   ndarray       : global 'Ball - position x' ... via BallRigidBody")
    print(
        "car rigid body  : frame_data['players'][k] = [remote_id, {'frames': [...]}], "
        "frames[i]['Data']['rigid_body']"
    )
    for pid, pdata in fd["frame_data"]["players"]:
        frames = pdata["frames"]
        print(f"   {who(pid):<16} variants={dict(Counter(variant(f) for f in frames))}")
    first = next(
        f["Data"] for f in fd["frame_data"]["players"][0][1]["frames"] if variant(f) == "Data"
    )
    print(f"   player frame keys: {sorted(first)}")

    boosts = np.array(
        [
            f["Data"]["boost_amount"]
            for _, p in fd["frame_data"]["players"]
            for f in p["frames"]
            if variant(f) == "Data"
        ]
    )
    nd_heads = nd_meta["column_headers"]["player_headers"]
    n_g = len(g)
    n_p = len(nd_heads)
    bi = nd_heads.index("boost level (raw replay units)")
    nd_boost = np.concatenate(
        [nd[:, n_g + k * n_p + bi] for k in range(len(fd["frame_data"]["players"]))]
    )
    print(
        f"boost           : frames[i]['Data']['boost_amount'] min={boosts.min()} "
        f"max={boosts.max()} "
        f"(kickoff value {first['boost_amount']}); ndarray 'boost level (raw replay units)' "
        f"min={np.nanmin(nd_boost)} max={np.nanmax(nd_boost)}"
    )
    print("   -> both are RAW 0-255 (kickoff 85 = 33.3%). Normalize with * 100 / 255.")
    secs = [m["seconds_remaining"] for m in md]
    print(
        f"SecondsRemaining: frame_data['metadata_frames'][i]['seconds_remaining'] (int) "
        f"range {min(secs)}..{max(secs)}; ndarray global 'seconds remaining'"
    )
    state_names = Counter(m["replicated_game_state_name"] for m in md)
    print(
        "game state      : metadata_frames[i]['replicated_game_state_name'] (index into "
        "parse_replay()['names']) and ['replicated_game_state_time_remaining'] "
        "(kickoff countdown)"
    )
    for sid, n in sorted(state_names.items()):
        idx = [i for i, m in enumerate(md) if m["replicated_game_state_name"] == sid]
        runs = 1 + int(np.sum(np.diff(idx) > 1))
        print(f"   id {sid:<4} name={raw['names'][sid]!r:<26} frames={n:<6} contiguous runs={runs}")
    countdown = Counter(m["replicated_game_state_time_remaining"] for m in md)
    print(f"   time_remaining values: {dict(sorted(countdown.items()))}")

    touches = fd["touch_events"]
    null_touch = sum(1 for t in touches if t["player"] is None)
    print(
        f"touches         : get_replay_frames_data()['touch_events'] n={len(touches)} "
        f"(player null on {null_touch}); keys={sorted(touches[0])}"
    )
    print(f"   sample        : {short(touches[0])}")
    if null_touch:
        nulls = [t for t in touches if t["player"] is None]
        print(
            f"   null-player touch clocks: "
            f"{[clock(md[t['frame']]['seconds_remaining']) for t in nulls][:20]}"
        )
    by_player = Counter(who(t["player"]) if t["player"] else "None" for t in touches)
    print(f"   per player    : {dict(by_player)}")
    stats = sa.get_stats_events(path)["events"]
    streams = Counter(e["meta"]["stream"] for e in stats)
    st_touch = [e for e in stats if e["meta"]["stream"] == "touch"]
    st_touch_null = sum(1 for e in st_touch if e["payload"]["payload"].get("player") is None)
    print(
        f"touches (attributed): get_stats_events()['events'] where meta.stream=='touch' "
        f"n={len(st_touch)} (player null on {st_touch_null})"
    )
    print(f"   payload keys  : {sorted(st_touch[0]['payload']['payload'])}")
    print(
        f"   per player    : "
        f"{dict(Counter(who(e['payload']['payload']['player']) for e in st_touch))}"
    )
    raw_touch_frames = {t["frame"] for t in touches}
    st_touch_frames = {e["payload"]["payload"]["frame"] for e in st_touch}
    print(
        f"   frames shared with touch_events: {len(raw_touch_frames & st_touch_frames)} "
        f"(touch_events only {len(raw_touch_frames - st_touch_frames)}, "
        f"stats only {len(st_touch_frames - raw_touch_frames)})"
    )
    print(f"stats streams   : {dict(streams.most_common())}")
    print(
        f"goals           : get_replay_frames_data()['goal_events'] n={len(fd['goal_events'])}; "
        f"header properties['Goals'] n={len(props.get('Goals', []))}"
    )
    print(
        "demos           : get_replay_frames_data()['demolish_infos'] "
        f"n={len(fd['demolish_infos'])}"
    )
    for d in fd["demolish_infos"]:
        print(
            f"   {clock(d['seconds_remaining'])}  {who(d['attacker'])} -> {who(d['victim'])}  "
            f"frame={d['frame']}"
        )
    pads = fd["boost_pads"]
    pad_events = fd["boost_pad_events"]
    kinds = Counter(variant(e["kind"]) for e in pad_events)
    picked = [e for e in pad_events if variant(e["kind"]) == "PickedUp"]
    size = {p["pad_id"]: p["size"] for p in pads}
    print(
        f"boost pads      : get_replay_frames_data()['boost_pads'] n={len(pads)} "
        f"sizes={dict(Counter(p['size'] for p in pads))}"
    )
    print(
        f"pad pickups     : get_replay_frames_data()['boost_pad_events'] n={len(pad_events)} "
        f"kinds={dict(kinds)}"
    )
    print(f"   PickedUp sample: {short(picked[0])}")
    print(
        f"   PickedUp by size: {dict(Counter(size.get(e['pad_id'], '?') for e in picked))}  "
        f"player null on {sum(1 for e in picked if e['player'] is None)}"
    )
    print(
        f"other streams   : player_stat_events={len(fd['player_stat_events'])} "
        f"{dict(Counter(e['kind'] for e in fd['player_stat_events']))}, "
        f"replay_tick_marks={len(fd['replay_tick_marks'])}, "
        f"dodge_refreshed_events={len(fd['dodge_refreshed_events'])}, "
        f"player_camera_events={len(fd['player_camera_events'])}"
    )
    ticks = [
        (t["description"], clock(md[t["frame"]]["seconds_remaining"]))
        for t in fd["replay_tick_marks"]
    ]
    print(f"   tick marks    : {ticks}")

    # ---------------------------------------------------------------- 6. goal clocks
    section("6. Goal clocks vs replays/NOTES.md")
    header_goals = props.get("Goals", [])
    expected = EXPECTED_GOALS.get(replay_id)
    ok = True
    for k, ge in enumerate(fd["goal_events"]):
        f = ge["frame"]
        by_frame = md[f]["seconds_remaining"]
        by_time = md[int(np.argmin(np.abs(t_md - ge["time"])))]["seconds_remaining"]
        hdr_frame = header_goals[k]["frame"] if k < len(header_goals) else None
        scorer = who(ge["player"])
        line = (
            f"  goal {k + 1}: {clock(by_frame)} (by time {clock(by_time)})  {scorer:<16} "
            f"score {ge['team_zero_score']}-{ge['team_one_score']}  "
            f"goal_event.frame={f} header.frame={hdr_frame} time={ge['time']:.2f}"
        )
        if expected and k < len(expected):
            exp_clock, exp_scorer = expected[k]
            match = clock(by_frame) == exp_clock and scorer == exp_scorer
            ok &= match
            line += f"  NOTES {exp_clock} {exp_scorer}: {'MATCH' if match else 'MISMATCH'}"
        print(line)
    if expected:
        ok &= len(expected) == len(fd["goal_events"])
        print(f"  all goals match NOTES.md: {ok}")
    final = fd["goal_events"][-1] if fd["goal_events"] else None
    if final:
        print(
            f"  final goal score {final['team_zero_score']}-{final['team_one_score']} vs header "
            f"{props.get('Team0Score')}-{props.get('Team1Score')}"
        )
    nd_rows_at_goal = [int(np.argmin(np.abs(ct - ge["time"]))) for ge in fd["goal_events"]]
    print(
        f"  header Goals[].frame indexes RAW network frames (== goal_events.frame). As ndarray "
        f"row indexes they would be wrong; nearest 10fps rows by time are {nd_rows_at_goal}"
    )

    # ---------------------------------------------------------------- 7. weirdness
    section("7. Things that look missing or weird")
    t_end_play = t_md[np.array(secs) == 0]
    print(
        f"  clock reaches 0 at t={t_end_play[0]:.2f}s"
        if len(t_end_play)
        else f"  clock never reaches 0 (last clock {clock(secs[-1])}) -> forfeit/early end"
    )
    overtime = any(s > p for p, s in zip(secs[:-1], secs[1:], strict=True) if p == 0)
    print(f"  overtime (clock goes back up after 0): {overtime}")
    for pid, pdata in fd["frame_data"]["players"]:
        empties = [i for i, f in enumerate(pdata["frames"]) if variant(f) != "Data"]
        if empties:
            print(
                f"  {who(pid)}: {len(empties)} non-Data frames at raw frames "
                f"{empties[:10]}{' ...' if len(empties) > 10 else ''} "
                f"(clocks {sorted({clock(secs[i]) for i in empties})})"
            )
    null_vel_states = Counter(
        names[md[i]["replicated_game_state_name"]]
        for i, b in enumerate(ball)
        if variant(b) == "Data" and b["Data"]["rigid_body"]["linear_velocity"] is None
    )
    print(
        f"  ball frames with null linear_velocity (ball 'sleeping' before kickoff touch), "
        f"by game state: {dict(null_vel_states)}"
    )
    players_by_key = {
        player_key(pid): pdata["frames"] for pid, pdata in fd["frame_data"]["players"]
    }
    for d in fd["demolish_infos"]:
        frames = players_by_key[player_key(d["victim"])]
        f0 = d["frame"]
        frozen = 0
        for f in frames[f0:]:
            if variant(f) != "Data" or f["Data"]["rigid_body"]["linear_velocity"] is not None:
                break
            frozen += 1
        loc0 = (
            frames[f0]["Data"]["rigid_body"]["location"] if variant(frames[f0]) == "Data" else None
        )
        print(
            f"  demo victim {who(d['victim'])} at {clock(d['seconds_remaining'])}: "
            f"{frozen} frames ({frozen / 30:.1f}s @30Hz) with null velocity after the demo, "
            f"position held at {short(loc0, 80)}"
        )
    demo_col = nd_heads.index("player demolished by")
    demo_vals = Counter(
        float(v)
        for k in range(len(fd["frame_data"]["players"]))
        for v in nd[:, n_g + k * n_p + demo_col]
        if not np.isnan(v)
    )
    print(f"  ndarray 'player demolished by' value counts: {dict(demo_vals.most_common(6))}")
    nan_rows = int(np.isnan(nd).any(axis=1).sum())
    print(f"  ndarray (10fps) rows containing any NaN: {nan_rows} / {nd.shape[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("replays", nargs="*", type=Path, help="replay files (default: replays/*)")
    args = parser.parse_args()
    paths = args.replays or sorted(REPLAY_DIR.glob("*.replay"))
    if not paths:
        raise SystemExit(f"no .replay files found in {REPLAY_DIR}")
    for path in paths:
        explore(path)


if __name__ == "__main__":
    main()
