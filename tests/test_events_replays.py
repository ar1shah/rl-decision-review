"""Validate the 1.3 events table against the real replays and replays/NOTES.md."""

import numpy as np
import pandas as pd
import pytest
from conftest import EXPECTED_DEMOS, EXPECTED_GOALS

from rldr.parse.events import EVENT_COLUMNS, build_events
from rldr.parse.frames import format_clock, player_key

BIG_PAD_XY = {(sx * 3072.0, sy * 4096.0) for sx in (-1, 1) for sy in (-1, 1)}
BIG_PAD_XY |= {(-3584.0, 0.0), (3584.0, 0.0)}
TICK_TO_EVENT_MAX_S = 4.0  # replay tick marks sit ~1-2.5 s before the event they mark


_EVENTS: dict[str, pd.DataFrame] = {}


@pytest.fixture
def events(replay) -> pd.DataFrame:
    rid = replay.parsed.meta["replay_id"]
    if rid not in _EVENTS:
        _EVENTS[rid] = build_events(replay.parsed, replay.fd, replay.stats)
    return _EVENTS[rid]


def _names(replay) -> dict[str, str]:
    p = replay.parsed.players
    return dict(zip(p["player_key"], p["name"], strict=True))


def test_schema_and_order(events, replay):
    assert list(events.columns) == EVENT_COLUMNS
    assert events["frame"].is_monotonic_increasing
    assert events["frame"].between(0, replay.parsed.meta["n_frames"] - 1).all()
    has_player = events["player_key"].notna()
    players = replay.parsed.players
    team_of = dict(zip(players["player_key"], players["team"], strict=True))
    assert (
        events.loc[has_player, "team"] == events.loc[has_player, "player_key"].map(team_of)
    ).all()


def test_time_and_clock_come_from_frames(events, replay):
    f = replay.parsed.frames.set_index("frame").loc[events["frame"]]
    assert np.allclose(events["time"], f["time"])
    assert (events["clock"].to_numpy() == f["clock"].to_numpy()).all()
    assert (events["run"].to_numpy() == f["run"].to_numpy()).all()


# --------------------------------------------------------------------------- goals


def test_goal_count_matches_scoreboard(events, replay):
    goals = events[events["type"] == "goal"]
    meta = replay.parsed.meta
    assert len(goals) == meta["team0_score"] + meta["team1_score"]
    last = goals.iloc[-1]
    assert (last["team0_score"], last["team1_score"]) == (meta["team0_score"], meta["team1_score"])
    assert (goals["team"] == 0).sum() == meta["team0_score"]


def test_goal_clocks_and_scorers_match_notes(events, replay):
    goals = events[events["type"] == "goal"]
    names = _names(replay)
    got = [
        (format_clock(c, ot), names[k])
        for c, ot, k in zip(goals["clock"], goals["is_overtime"], goals["player_key"], strict=True)
    ]
    assert got == EXPECTED_GOALS[replay.parsed.meta["replay_id"]]


def test_goals_are_in_a_net(events):
    goals = events[events["type"] == "goal"]
    # Blue (team 0) scores in Orange's net at +y, Orange in Blue's at -y.
    assert (goals["y"].abs() > 5120).all()
    assert (np.sign(goals["y"]) == np.where(goals["team"] == 0, 1, -1)).all()


# --------------------------------------------------------------------------- demos


def test_demo_count_and_keys_match_demolish_infos(events, replay):
    demos = events[events["type"] == "demo"]
    infos = replay.fd["demolish_infos"]
    assert len(demos) == len(infos) == EXPECTED_DEMOS[replay.parsed.meta["replay_id"]]
    assert demos["player_key"].tolist() == [player_key(d["attacker"]) for d in infos]
    assert demos["other_player_key"].tolist() == [player_key(d["victim"]) for d in infos]


def test_demo_victim_is_demolished_from_the_demo_frame(events, replay):
    e = replay.parsed.entities
    for _, d in events[events["type"] == "demo"].iterrows():
        car = e[e["entity"] == d["other_player_key"]].set_index("frame")["is_demolished"]
        assert car.loc[d["frame"]] and not car.loc[d["frame"] - 1]


# --------------------------------------------------------------------------- kickoffs


def test_kickoff_count_is_goals_plus_one_unless_game_ends_first(events, replay):
    kickoffs = events[events["type"] == "kickoff"]
    goals = events[events["type"] == "goal"]
    f = replay.parsed.frames
    ended_before_kickoff = not f.loc[f["frame"] > goals["frame"].max(), "is_live"].any()
    expected = len(goals) + 1 - int(ended_before_kickoff)
    assert len(kickoffs) == expected
    # the forfeit replay is the one that ends before the next kickoff
    assert ended_before_kickoff == replay.parsed.meta["forfeit"]


def test_kickoffs_start_live_at_center_after_countdown(events, replay):
    kickoffs = events[events["type"] == "kickoff"]
    f = replay.parsed.frames.set_index("frame")
    assert (f.loc[kickoffs["frame"], "game_state"] == "Active").all()
    assert (f.loc[kickoffs["frame"] - 1, "game_state"] == "Countdown").all()
    assert (f.loc[kickoffs["countdown_frame"].astype(int), "game_state"] == "Countdown").all()
    assert np.allclose(kickoffs[["x", "y"]], 0.0)
    assert (kickoffs["first_touch_frame"] > kickoffs["frame"]).all()
    # the first kickoff is at 5:00; every later one follows a goal (run 1 of the goal's clock)
    assert kickoffs.iloc[0]["clock"] == 300
    assert (kickoffs.iloc[1:]["run"] == 1).all()


# --------------------------------------------------------------------------- boost pads


def test_pickups_match_pad_events_and_pad_table(events, replay):
    pickups = events[events["type"] == "boost_pickup"]
    raw = [e for e in replay.fd["boost_pad_events"] if isinstance(e["kind"], dict)]
    assert len(pickups) == len(raw) > 0
    assert pickups["player_key"].notna().all()
    assert set(pickups["pad_size"]) == {"big", "small"}
    big = pickups[pickups["pad_size"] == "big"]
    assert set(zip(big["x"], big["y"], strict=True)) == BIG_PAD_XY
    pads = {p["pad_id"]: p for p in replay.fd["boost_pads"]}
    assert len(pads) == 34 and sum(p["size"] == "Big" for p in pads.values()) == 6


# --------------------------------------------------------------------------- shots / saves


def test_stat_event_counts(events, replay):
    raw = replay.fd["player_stat_events"]
    for kind, typ in (("Shot", "shot"), ("Save", "save"), ("Assist", "assist")):
        assert (events["type"] == typ).sum() == sum(e["kind"] == kind for e in raw)


@pytest.mark.parametrize("kind", ["Save", "Goal"])
def test_tick_marks_pair_one_to_one_with_events(events, replay, kind):
    """Every TeamNSave / TeamNGoal tick mark is followed by its own event of that team."""
    typ = kind.lower()
    marks = [t for t in replay.fd["replay_tick_marks"] if t["description"].endswith(kind)]
    evs = events[events["type"] == typ]
    assert len(marks) == len(evs)
    for mark, (_, ev) in zip(marks, evs.iterrows(), strict=True):
        team = int(mark["description"][len("Team")])
        assert ev["team"] == team
        assert 0 < ev["time"] - mark["time"] < TICK_TO_EVENT_MAX_S


def test_notes_saves_are_stat_saves(events, replay):
    """NOTES.md saves (Ari's memory, clock to the second) match the game's save credits."""
    rid = replay.parsed.meta["replay_id"]
    expected = {
        "2C980F9741798CF5E4D1A0BFF902C275": [("3:47", "CohenLovesJill"), ("0:22", "ari ツ")],
        "0AACF51B43A578E00DD400BD009DF81E": [],
    }[rid]
    orange = {
        "2C980F9741798CF5E4D1A0BFF902C275": ["1:58"],
        "0AACF51B43A578E00DD400BD009DF81E": ["2:37"],
    }
    saves = events[events["type"] == "save"]
    names = _names(replay)

    def near(clock):  # Ari's clock can be off by a second either way
        secs = int(clock.split(":")[0]) * 60 + int(clock.split(":")[1])
        return saves[(saves["clock"] - secs).abs() <= 1]

    for clock, who in expected:
        assert who in {names[k] for k in near(clock)["player_key"]}, clock
    for clock in orange[rid]:
        assert (near(clock)["team"] == 1).any(), clock


# --------------------------------------------------------------------------- touches


def _touches(events):
    return events[events["type"] == "touch"]


def test_touch_rows_are_complete(events):
    t = _touches(events)
    assert len(t) > 50
    assert t["player_key"].notna().all() and t["team"].notna().all()
    assert set(t["touch_source"]) <= {"both", "stats", "physics"}
    assert set(t["confidence"]) <= {"high", "medium", "low"}
    assert (t["stats_frame"].isna() == (t["touch_source"] == "physics")).all()
    contested = t["contested"].astype(bool)
    assert t.loc[contested, "contest_type"].isin(["opponent", "teammate"]).all()
    assert t.loc[~contested, "contest_type"].isna().all()
    flagged = contested | t["source_conflict"].astype(bool)
    assert t.loc[flagged, "alt_player_key"].notna().all()
    assert (t.loc[flagged, "alt_player_key"] != t.loc[flagged, "player_key"]).all()
    assert not t.duplicated(["frame", "player_key"]).any()
    assert (
        t[["ball_speed_before", "ball_speed_after", "ball_dv", "car_ball_dist"]].notna().all().all()
    )


def test_no_touches_in_non_live_frames(events, replay):
    live = replay.parsed.frames.set_index("frame")["is_live"]
    assert live.loc[_touches(events)["frame"]].all()


def test_both_source_touches_are_in_contact_range(events):
    t = _touches(events)
    both = t[(t["touch_source"] == "both") & ~t["source_conflict"].astype(bool)]
    assert (both["car_ball_dist"] <= 250).all()


def test_scoring_team_last_touch_is_the_scorer(events):
    t = _touches(events)
    for _, g in events[events["type"] == "goal"].iterrows():
        prior = t[(t["team"] == g["team"]) & (t["frame"] <= g["frame"])]
        last = prior[prior["frame"] == prior["frame"].max()]
        assert g["player_key"] in set(last["player_key"]), g["clock"]


def test_saver_is_their_team_last_toucher(events):
    t = _touches(events)
    for _, s in events[events["type"] == "save"].iterrows():
        prior = t[(t["team"] == s["team"]) & (t["frame"] <= s["frame"])]
        last = prior[prior["frame"] == prior["frame"].max()]
        assert s["player_key"] in set(last["player_key"]), s["clock"]
        assert s["touch_frame"] == last["frame"].iloc[0]


def test_every_shot_and_save_has_the_credited_players_touch(events):
    stat = events[events["type"].isin(["shot", "save"])]
    assert stat["touch_frame"].notna().all()
    assert (stat["frame"] - stat["touch_frame"]).between(0, 90).all()
    assert events.loc[~events["type"].isin(["shot", "save"]), "touch_frame"].isna().all()


def test_kickoff_first_touch_has_a_touch_row(events):
    t = _touches(events)
    for _, k in events[events["type"] == "kickoff"].iterrows():
        assert (t["frame"] == k["first_touch_frame"]).any(), k["clock"]


def test_notes_saves_have_the_savers_touch(events, replay):
    """NOTES.md saves: the touch behind each save credit is by the right player / team."""
    rid = replay.parsed.meta["replay_id"]
    names = _names(replay)
    expected = {  # clock -> player name (or "Orange")
        "2C980F9741798CF5E4D1A0BFF902C275": {
            "3:47": "CohenLovesJill",
            "1:58": "Orange",
            "0:22": "ari ツ",
        },
        "0AACF51B43A578E00DD400BD009DF81E": {"2:37": "Orange"},
    }[rid]
    t = _touches(events).set_index(["frame", "player_key"])
    saves = events[events["type"] == "save"]
    for clock, who in expected.items():
        secs = int(clock.split(":")[0]) * 60 + int(clock.split(":")[1])
        near = saves[(saves["clock"] - secs).abs() <= 1]
        if who == "Orange":
            near = near[near["team"] == 1]
        else:
            near = near[near["player_key"].map(names) == who]
        assert len(near), clock
        s = near.iloc[0]
        touch = t.loc[(s["touch_frame"], s["player_key"])]
        assert touch["team"] == s["team"], clock


def test_notes_347_is_a_teammate_50_50_and_022_is_ari(events, replay):
    if replay.parsed.meta["replay_id"] != "2C980F9741798CF5E4D1A0BFF902C275":
        pytest.skip("2C98 only")
    names = _names(replay)
    t = _touches(events)
    t = t.assign(name=t["player_key"].map(names))
    at_347 = t[(t["clock"] == 227) & (t["name"].isin(["CohenLovesJill", "ari ツ"]))]
    pair = at_347[at_347["frame"] == 2276]
    assert set(pair["name"]) == {"CohenLovesJill", "ari ツ"}
    assert (pair["contest_type"] == "teammate").all() and (pair["confidence"] == "low").all()
    ari_022 = t[(t["name"] == "ari ツ") & t["frame"].between(9200, 9204)]
    assert len(ari_022) == 1 and ari_022.iloc[0]["ball_dv"] > 2000
