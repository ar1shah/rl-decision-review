"""Validate the 1.2 frame tables against the real replays and replays/NOTES.md."""

import math

import numpy as np
import pytest

from rldr.parse.frames import format_clock, player_key

# Goal clocks and scorers Ari wrote down in replays/NOTES.md.
EXPECTED_GOALS = {
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
EXPECTED_DEMOS = {"2C980F9741798CF5E4D1A0BFF902C275": 6, "0AACF51B43A578E00DD400BD009DF81E": 1}
GOAL_REPLAY_GAP_S = 0.2


def _car_frames(replay, key):
    for remote_id, pdata in replay.fd["frame_data"]["players"]:
        if player_key(remote_id) == key:
            return pdata["frames"]
    raise KeyError(key)


def _is_data(frame):
    return isinstance(frame, dict) and "Data" in frame


# --------------------------------------------------------------------------- clock


def test_goal_clocks_match_notes(replay):
    p = replay.parsed
    names = dict(zip(p.players["player_key"], p.players["name"], strict=True))
    got = []
    for goal in replay.fd["goal_events"]:
        row = p.frames.iloc[goal["frame"]]
        got.append(
            (format_clock(row["clock"], row["is_overtime"]), names[player_key(goal["player"])])
        )
    assert got == EXPECTED_GOALS[p.meta["replay_id"]]


def test_no_overtime_in_these_replays(replay):
    p = replay.parsed
    assert not p.frames["is_overtime"].any()
    assert p.meta["overtime_source"] is None


def test_frame_index_matches_raw_timestamps(replay):
    p = replay.parsed
    md = replay.fd["frame_data"]["metadata_frames"]
    assert len(p.frames) == len(md) == p.meta["n_frames"]
    assert p.frames["frame"].tolist() == list(range(len(md)))
    assert np.allclose(p.frames["time"], [m["time"] for m in md])
    assert (np.diff(p.frames["time"]) > 0).all()


# --------------------------------------------------------------------------- dead time


def test_is_live_only_during_active(replay):
    f = replay.parsed.frames
    assert set(f["game_state"].astype(str)) <= {"Countdown", "Active", "PostGoalScored"}
    assert (f["is_live"] == (f["game_state"] == "Active")).all()
    assert not f.loc[f["game_state"].isin(["Countdown", "PostGoalScored"]), "is_live"].any()


def test_is_live_false_across_goal_replay_gaps(replay):
    f = replay.parsed.frames
    gaps = np.flatnonzero(np.diff(f["time"]) > GOAL_REPLAY_GAP_S)
    assert len(gaps) == len(replay.fd["goal_events"])  # one unrecorded goal replay per goal
    for i in gaps:
        assert not f.at[i, "is_live"] and not f.at[i + 1, "is_live"]
        assert f.at[i, "game_state"] == "PostGoalScored"
        assert f.at[i + 1, "game_state"] == "Countdown"


def test_clock_only_changes_during_live_play(replay):
    f = replay.parsed.frames
    changed = np.flatnonzero(np.diff(f["clock"]) != 0) + 1
    assert f.loc[changed, "is_live"].all()


# --------------------------------------------------------------------------- entities


def test_entity_table_shape_and_keys(replay):
    p = replay.parsed
    n_players = len(replay.fd["frame_data"]["players"])
    assert len(p.entities) == p.meta["n_frames"] * (1 + n_players)
    assert p.players["player_key"].is_unique
    assert set(p.entities["entity"].astype(str)) == {"ball", *p.players["player_key"]}
    assert not p.entities.duplicated(["frame", "entity"]).any()
    assert p.entities["frame"].is_monotonic_increasing
    cars = p.entities[p.entities["entity"] != "ball"]
    team_of = dict(zip(p.players["player_key"], p.players["team"], strict=True))
    assert (cars["team"].astype(int) == cars["entity"].astype(str).map(team_of)).all()


def test_ball_velocity_never_null(replay):
    ball = replay.parsed.entities.query("entity == 'ball'")
    assert ball[["x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz"]].notna().all().all()
    sleeping = ball["sleeping"].astype(bool)
    assert sleeping.any()
    assert (ball.loc[sleeping, ["vx", "vy", "vz"]] == 0).all().all()


def test_boost_is_0_to_100(replay):
    e = replay.parsed.entities
    cars = e[e["entity"] != "ball"]
    boost = cars["boost"].dropna()
    assert boost.between(0, 100).all()
    assert boost.max() == pytest.approx(100.0)
    kickoff = cars[cars["frame"] == 0]["boost"]
    assert kickoff.to_numpy() == pytest.approx([100 / 3] * len(kickoff))
    assert e.loc[e["entity"] == "ball", "boost"].isna().all()


def test_kickoff_cars_face_the_ball(replay):
    e = replay.parsed.entities
    first = e[e["frame"] == 0]
    ball = first[first["entity"] == "ball"].iloc[0]
    for _, car in first[first["entity"] != "ball"].iterrows():
        bearing = math.atan2(ball["y"] - car["y"], ball["x"] - car["x"])
        diff = (car["yaw"] - bearing + math.pi) % (2 * math.pi) - math.pi
        assert abs(math.degrees(diff)) < 10, car["entity"]


# --------------------------------------------------------------------------- demos / presence


def test_demolished_frames_have_null_car_state(replay):
    e = replay.parsed.entities
    demo = e[e["is_demolished"]]
    assert len(demo) > 0
    cols = ["x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "qw", "yaw", "boost"]
    assert demo[cols].isna().all().all()
    assert demo["is_present"].all()
    assert demo["boost_active"].isna().all()


def test_demolished_windows_match_demo_events_and_sleeping(replay):
    p = replay.parsed
    demos = replay.fd["demolish_infos"]
    assert len(demos) == EXPECTED_DEMOS[p.meta["replay_id"]]
    e = p.entities
    for key in p.players["player_key"]:
        car = e[e["entity"] == key].set_index("frame")
        frames = _car_frames(replay, key)
        demo_frames = {d["frame"] for d in demos if player_key(d["victim"]) == key}
        # A window starts exactly at each demo event...
        starts = set(np.flatnonzero(np.diff(np.r_[0, car["is_demolished"].astype(int)]) == 1))
        assert starts == demo_frames, key
        # ...and every recorded frame inside it is the frozen, sleeping car.
        for f in np.flatnonzero(car["is_demolished"]):
            if _is_data(frames[f]):
                assert frames[f]["Data"]["rigid_body"]["sleeping"], (key, f)
        # The car only sleeps while demolished.
        sleeping = car["sleeping"].fillna(False).astype(bool)
        assert not (sleeping & ~car["is_demolished"]).any(), key


def test_not_present_only_on_reset_empty_frames(replay):
    p = replay.parsed
    e = p.entities
    for key in p.players["player_key"]:
        car = e[e["entity"] == key].set_index("frame")
        frames = _car_frames(replay, key)
        empty = np.array([not _is_data(f) for f in frames])
        expected = empty & ~car["is_demolished"].to_numpy()
        assert (~car["is_present"].to_numpy() == expected).all(), key
        absent = np.flatnonzero(expected)
        assert (p.frames.loc[absent, "game_state"] == "Countdown").all(), key
        assert car.loc[absent, ["x", "y", "z", "boost"]].isna().all().all(), key


# --------------------------------------------------------------------------- done-when check


def test_positions_at_degnis_goal_3_06(load_lab_replay):
    """ROADMAP 1.2 done-when: the 3:06 goal (Degnis98 scores on Blue) looks right."""
    p = load_lab_replay("2C980F9741798CF5E4D1A0BFF902C275").parsed
    rows = p.at_clock("3:06")
    assert sorted(rows["run"].unique()) == [0, 1]  # play before the goal, then the next kickoff
    before_goal = rows[rows["run"] == 0]
    last = before_goal[before_goal["frame"] == before_goal["frame"].max()].set_index("name")
    ball, scorer = last.loc["ball"], last.loc["Degnis98"]
    assert ball["y"] < -5000  # in Blue's goal
    assert math.dist((ball["x"], ball["y"]), (scorer["x"], scorer["y"])) < 1500
    assert last.loc["ari ツ", "is_demolished"]  # Ari was demolished just before this goal
    kickoff = rows[rows["run"] == 1]
    assert kickoff["frame"].min() > before_goal["frame"].max()
