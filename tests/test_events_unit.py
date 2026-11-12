"""Synthetic unit tests for the 1.3 event builders (no replay files needed)."""

import numpy as np

from rldr.parse.events import (
    demo_events,
    goal_events,
    kickoff_frames,
    pickup_events,
    stat_events,
)
from rldr.parse.frames import clock_run

STEAM = {"Steam": "1"}
EPIC = {"Epic": "abc"}


# --------------------------------------------------------------------------- clock_run


def test_clock_run_numbers_live_stretches_within_a_second():
    # clock 186: play, goal (dead), countdown (dead), kickoff (live again at the same clock)
    clock = np.array([187, 186, 186, 186, 186, 186, 186, 185])
    live = np.array([1, 1, 1, 0, 0, 1, 1, 1], dtype=bool)
    ot = np.zeros(8, dtype=bool)
    assert clock_run(clock, ot, live).tolist() == [0, 0, 0, 0, 0, 1, 1, 0]


def test_clock_run_opening_countdown_is_run_0():
    clock = np.array([300, 300, 300, 300, 299])
    live = np.array([0, 0, 1, 1, 1], dtype=bool)
    assert clock_run(clock, np.zeros(5, dtype=bool), live).tolist() == [0, 0, 0, 0, 0]


def test_clock_run_overtime_is_its_own_group():
    clock = np.array([0, 0, 0, 0])
    live = np.array([1, 0, 0, 1], dtype=bool)
    ot = np.array([0, 0, 1, 1], dtype=bool)
    assert clock_run(clock, ot, live).tolist() == [0, 0, 0, 0]


def test_clock_run_empty():
    assert clock_run(np.array([]), np.array([]), np.array([])).tolist() == []


# --------------------------------------------------------------------------- kickoffs


def test_kickoff_frames_from_countdown_to_active():
    states = ["Countdown"] * 3 + ["Active"] * 4 + ["PostGoalScored"] * 2
    states += ["Countdown"] * 2 + ["Active"] * 3
    asleep = [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0]
    assert kickoff_frames(states, asleep) == [
        {"frame": 3, "countdown_frame": 0, "first_touch_frame": 5},
        {"frame": 11, "countdown_frame": 9, "first_touch_frame": 12},
    ]


def test_kickoff_frames_skip_countdown_that_never_goes_live():
    # forfeit right after a goal: the replay ends in Countdown
    states = ["Countdown", "Active", "Active", "PostGoalScored", "Countdown", "Countdown"]
    ks = kickoff_frames(states, [1, 0, 0, 1, 1, 1])
    assert [k["frame"] for k in ks] == [1]


def test_kickoff_without_touch_has_no_first_touch():
    ks = kickoff_frames(["Countdown", "Active", "Active"], [1, 1, 1])
    assert ks[0]["first_touch_frame"] == -1


# --------------------------------------------------------------------------- other builders


def test_goal_events_running_score():
    goals = [
        {"frame": 10, "player": STEAM, "scoring_team_is_team_0": False},
        {"frame": 20, "player": EPIC, "scoring_team_is_team_0": True},
    ]
    goals[0] |= {"team_zero_score": 0, "team_one_score": 1}
    goals[1] |= {"team_zero_score": 1, "team_one_score": 1}
    df = goal_events(goals)
    assert df["player_key"].tolist() == ["Steam:1", "Epic:abc"]
    assert df["team"].tolist() == [1, 0]
    assert df[["team0_score", "team1_score"]].to_numpy().tolist() == [[0, 1], [1, 1]]


def test_demo_events_attacker_victim_and_locations():
    demo = {
        "frame": 5,
        "attacker": STEAM,
        "victim": EPIC,
        "attacker_location": {"x": 1.0, "y": 2.0, "z": 3.0},
        "victim_location": {"x": 4.0, "y": 5.0, "z": 6.0},
    }
    row = demo_events([demo]).iloc[0]
    assert (row["player_key"], row["other_player_key"]) == ("Steam:1", "Epic:abc")
    assert (row["x"], row["y"], row["z"]) == (4.0, 5.0, 6.0)
    assert (row["attacker_x"], row["attacker_y"], row["attacker_z"]) == (1.0, 2.0, 3.0)


def test_pickup_events_lookup_pad_and_skip_respawns():
    pads = [
        {"pad_id": "P1", "position": {"x": 3072.0, "y": -4096.0, "z": 73.0}, "size": "Big"},
        {"pad_id": "P2", "position": {"x": 0.0, "y": -4240.0, "z": 70.0}, "size": "Small"},
    ]
    events = [
        {"frame": 3, "kind": {"PickedUp": {"sequence": 1}}, "pad_id": "P1", "player": STEAM},
        {"frame": 4, "kind": "Available", "pad_id": "P1", "player": None},
        {"frame": 9, "kind": {"PickedUp": {"sequence": 2}}, "pad_id": "P2", "player": EPIC},
    ]
    df = pickup_events(events, pads)
    assert df["frame"].tolist() == [3, 9]
    assert df["pad_size"].tolist() == ["big", "small"]
    assert df["player_key"].tolist() == ["Steam:1", "Epic:abc"]
    assert (df.iloc[0]["x"], df.iloc[0]["y"]) == (3072.0, -4096.0)


def test_stat_events_keep_shot_save_assist_only():
    stats = [
        {"frame": 1, "kind": "Shot", "player": STEAM, "is_team_0": True},
        {"frame": 2, "kind": "Save", "player": EPIC, "is_team_0": False},
        {"frame": 3, "kind": "Assist", "player": STEAM, "is_team_0": True},
        {"frame": 4, "kind": "Something", "player": STEAM, "is_team_0": True},
    ]
    stats[0]["shot"] = {"projected_goal_line_crossing": {"inside_goal_mouth": True}}
    df = stat_events(stats)
    assert df["type"].tolist() == ["shot", "save", "assist"]
    assert df["team"].tolist() == [0, 1, 0]
    assert df["shot_on_target"].tolist()[0] is True
