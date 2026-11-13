"""Synthetic unit tests for the 1.3 touch rule (no replay files needed)."""

import numpy as np
import pandas as pd

from rldr.parse.touches import (
    CONTACT_RANGE,
    dedup_stats,
    resolve_touches,
    surface_contacts,
    touch_confidence,
)

# A and B are teammates (team 0); X is an opponent (team 1).
TEAM_OF = {"A": 0, "B": 0, "X": 1}
FAR = np.inf


def stats(*rows):
    return pd.DataFrame(rows, columns=["frame", "player_key"])


def phys(*rows):
    """(frame, dv, player, dist, second, second_dist, surfaces, bounce)"""
    cols = ["frame", "dv", "player_key", "dist", "second_key", "second_dist", "surfaces"]
    df = pd.DataFrame([r[:7] for r in rows], columns=cols)
    df["bounce"] = [r[7] if len(r) > 7 else False for r in rows]
    return df


NO_PHYS = phys()


def one(df, **match):
    sel = np.ones(len(df), dtype=bool)
    for k, v in match.items():
        sel &= (df[k] == v).to_numpy()
    assert sel.sum() == 1, df
    return df[sel].iloc[0]


# --------------------------------------------------------------------------- dedup


def test_dedup_merges_same_player_within_two_frames_only():
    st = stats((100, "A"), (101, "A"), (102, "X"), (103, "A"), (110, "A"))
    out = dedup_stats(st)
    assert list(zip(out["frame"], out["player_key"], strict=True)) == [
        (100, "A"),
        (102, "X"),
        (103, "A"),
        (110, "A"),
    ]


# --------------------------------------------------------------------------- matching


def test_stats_touch_moves_to_matching_physics_frame():
    t = resolve_touches(stats((100, "A")), phys((102, 900, "A", 150, "X", 600, [])), TEAM_OF)
    r = one(t, player_key="A")
    assert (r["frame"], r["stats_frame"], r["touch_source"]) == (102, 100, "both")
    assert not r["contested"] and not r["source_conflict"]


def test_physics_more_than_three_frames_away_does_not_match():
    t = resolve_touches(stats((100, "A")), phys((104, 900, "A", 150, "X", 600, [])), TEAM_OF)
    r = one(t, player_key="A")
    assert (r["frame"], r["touch_source"]) == (100, "stats")


def test_physics_out_of_contact_range_is_ignored():
    p = phys((101, 900, "A", CONTACT_RANGE + 1, "X", 900, []))
    r = one(resolve_touches(stats((100, "A")), p, TEAM_OF), player_key="A")
    assert r["touch_source"] == "stats" and not r["source_conflict"]


def test_soft_stats_touch_without_physics_is_kept():
    r = one(resolve_touches(stats((100, "A")), NO_PHYS, TEAM_OF), player_key="A")
    assert (r["frame"], r["touch_source"]) == (100, "stats")


def test_dedup_keeps_a_touch_that_is_half_of_a_same_frame_pair():
    # 0AAC 4:37: X at 100, then X and A together at 101 (X is the scorer).
    out = dedup_stats(stats((100, "X"), (101, "A"), (101, "X")))
    assert len(out) == 3


def test_close_in_time_50_50_beats_a_later_frame_where_the_player_is_nearest():
    # 2C98 last kickoff: X's stats touch at 100 is a 50/50 there (X second, 8 uu behind A);
    # X is only nearest at 103.
    p = phys((100, 2900, "A", 207, "X", 215, []), (103, 780, "X", 177, "A", 191, []))
    r = one(resolve_touches(stats((100, "X")), p, TEAM_OF), player_key="X")
    assert (r["frame"], r["contested"], r["alt_player_key"]) == (100, True, "A")


def test_conflicting_match_only_when_nothing_better():
    p = phys((100, 900, "X", 140, "A", 230, []), (102, 900, "A", 150, None, FAR, []))
    r = one(resolve_touches(stats((100, "A")), p, TEAM_OF), player_key="A")
    assert (r["frame"], r["source_conflict"]) == (102, False)


def test_two_stats_touches_on_one_physics_frame_become_one():
    st = stats((100, "A"), (103, "A"))
    t = resolve_touches(st, phys((102, 900, "A", 150, None, FAR, [])), TEAM_OF)
    assert len(t) == 1 and t.iloc[0]["frame"] == 102


# --------------------------------------------------------------------------- contested


def test_same_frame_opponent_pair_is_two_contested_rows():
    st = stats((100, "A"), (100, "X"))
    t = resolve_touches(st, phys((100, 900, "A", 170, "X", 175, [])), TEAM_OF)
    assert len(t) == 2
    a, x = one(t, player_key="A"), one(t, player_key="X")
    assert a["contested"] and x["contested"]
    assert (a["alt_player_key"], x["alt_player_key"]) == ("X", "A")
    assert a["contest_type"] == x["contest_type"] == "opponent"
    assert a["touch_source"] == x["touch_source"] == "both"  # X matches as second car in range


def test_same_frame_teammate_pair_is_teammate_contest():
    st = stats((100, "A"), (100, "B"))
    t = resolve_touches(st, phys((100, 900, "A", 170, "B", 175, [])), TEAM_OF)
    assert set(t["contest_type"]) == {"teammate"}


def test_physics_second_car_close_makes_a_single_stats_touch_contested():
    t = resolve_touches(stats((100, "A")), phys((100, 900, "A", 170, "X", 195, [])), TEAM_OF)
    r = one(t, player_key="A")
    assert r["contested"] and r["alt_player_key"] == "X" and r["contest_type"] == "opponent"


def test_second_car_in_range_but_far_behind_is_not_contested():
    t = resolve_touches(stats((100, "A")), phys((100, 900, "A", 150, "X", 240, [])), TEAM_OF)
    r = one(t, player_key="A")
    assert not r["contested"] and pd.isna(r["contest_type"]) and r["alt_player_key"] is None


# --------------------------------------------------------------------------- conflicts


def test_credited_player_second_and_far_behind_is_a_conflict():
    t = resolve_touches(stats((100, "A")), phys((101, 900, "X", 140, "A", 230, [])), TEAM_OF)
    r = one(t, player_key="A")
    assert r["source_conflict"] and not r["contested"] and r["alt_player_key"] == "X"
    assert r["frame"] == 101


def test_credited_player_out_of_range_is_a_conflict_on_the_stats_frame():
    t = resolve_touches(stats((100, "A")), phys((101, 900, "X", 140, "B", 600, [])), TEAM_OF)
    r = one(t, player_key="A")
    assert r["source_conflict"] and r["alt_player_key"] == "X"
    assert (r["frame"], r["touch_source"]) == (100, "stats")


# --------------------------------------------------------------------------- physics-only


def test_physics_only_touch_needs_strong_clean_and_alone():
    p = phys(
        (200, 800, "A", 150, "X", 900, []),  # added
        (300, 400, "A", 150, "X", 900, []),  # too soft
        (400, 800, "A", 150, "X", 200, []),  # another car in range
        (500, 800, "A", 150, "X", 900, ["floor"]),  # near a surface
        (600, 800, "A", 150, "X", 900, [], True),  # bounce
        (705, 800, "A", 150, "X", 900, []),  # stats touch by someone within 10 frames
    )
    t = resolve_touches(stats((696, "X")), p, TEAM_OF)
    added = t[t["touch_source"] == "physics"]
    assert added["frame"].tolist() == [200]
    assert added["stats_frame"].isna().all()


# --------------------------------------------------------------------------- confidence


def test_confidence_levels():
    rows = [
        # source, contest_type, conflict, car dist, expected
        ("both", None, False, 150, "high"),
        ("stats", None, False, 150, "medium"),
        ("physics", None, False, 150, "medium"),
        ("both", "opponent", False, 170, "medium"),
        ("both", "opponent", False, CONTACT_RANGE + 50, "low"),
        ("both", "teammate", False, 170, "low"),
        ("stats", None, True, 400, "low"),
        ("both", None, True, 230, "low"),
    ]
    df = pd.DataFrame(rows, columns=["src", "ct", "conflict", "dist", "want"])
    got = touch_confidence(df["src"], df["ct"], df["conflict"], df["dist"])
    assert got.tolist() == df["want"].tolist()


# --------------------------------------------------------------------------- surfaces


def test_surface_contacts():
    pos = np.array(
        [
            [0.0, 0.0, 93.0],  # resting on the floor
            [0.0, 0.0, 1000.0],  # mid-air
            [4000.0, 0.0, 1000.0],  # side wall
            [0.0, -5100.0, 1000.0],  # back wall above the goal
            [0.0, 5050.0, 300.0],  # entering the open goal mouth: nothing
            [880.0, 5100.0, 300.0],  # goal post
            [0.0, 5300.0, 300.0],  # inside the net
        ]
    )
    got = surface_contacts(pos)
    assert got[0] == ["floor"]
    assert got[1] == []
    assert got[2] == ["side_wall"]
    assert got[3] == ["back_wall"]
    assert got[4] == []
    assert got[5] == ["goal"]
    assert got[6] == ["goal"]
