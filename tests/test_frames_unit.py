"""Unit tests for rldr.parse.frames helpers (synthetic data, no replay files needed)."""

import math

import numpy as np
import pytest

from rldr.parse.frames import (
    boost_to_pct,
    compute_clock,
    demolished_mask,
    format_clock,
    overtime_fallback_frame,
    overtime_flag_frame,
    parse_clock,
    platform_of,
    player_key,
    presence_mask,
    quat_to_euler,
)

# --------------------------------------------------------------------------- boost


@pytest.mark.parametrize(("raw", "pct"), [(0, 0.0), (85, 100 / 3), (255, 100.0), (127.5, 50.0)])
def test_boost_to_pct_scalar(raw, pct):
    assert boost_to_pct(raw) == pytest.approx(pct)


def test_boost_to_pct_array_keeps_nan():
    out = boost_to_pct(np.array([0.0, 255.0, np.nan]))
    assert out[:2] == pytest.approx([0.0, 100.0])
    assert np.isnan(out[2])


# --------------------------------------------------------------------------- ids


@pytest.mark.parametrize(
    ("remote_id", "key", "platform"),
    [
        ({"Steam": "76561198833913591"}, "Steam:76561198833913591", "Steam"),
        ({"Epic": "a7f54509d625455c"}, "Epic:a7f54509d625455c", "Epic"),
        ({"Xbox": "2535416587616781"}, "Xbox:2535416587616781", "Xbox"),
        (
            {"PlayStation": {"name": "Cohen", "online_id": "7802313240122656448", "unknown1": []}},
            "PlayStation:7802313240122656448",
            "PlayStation",
        ),
    ],
)
def test_player_key(remote_id, key, platform):
    assert player_key(remote_id) == key
    assert platform_of(key) == platform


def test_player_key_rejects_garbage():
    with pytest.raises(ValueError):
        player_key({})


# --------------------------------------------------------------------------- clock text


@pytest.mark.parametrize(
    ("text", "value"),
    [("5:00", (300, False)), ("3:06", (186, False)), ("0:02", (2, False)), ("+0:30", (30, True))],
)
def test_parse_and_format_clock_round_trip(text, value):
    assert parse_clock(text) == value
    assert format_clock(*value) == text


def test_parse_clock_rejects_bad_seconds():
    with pytest.raises(ValueError):
        parse_clock("1:75")


# --------------------------------------------------------------------------- clock mapping


def _states(*runs):
    """_states(("Countdown", 2), ("Active", 3)) -> array of state names."""
    return np.array([s for s, n in runs for _ in range(n)], dtype=object)


def test_regulation_clock_is_seconds_remaining():
    secs = np.array([300, 300, 299, 298, 298])
    states = _states(("Countdown", 2), ("Active", 3))
    clock, ot = compute_clock(secs, states, np.arange(5) / 30, overtime_start=None)
    assert clock.tolist() == secs.tolist()
    assert not ot.any()


def _overtime_match():
    """Regulation ends 1-1 at 0:00, then an OT kickoff and ~2.5 s of OT play, then a goal."""
    states = _states(
        ("Active", 3),  # 0:01, 0:00, 0:00 (ball in the air)
        ("Countdown", 3),  # OT kickoff countdown (frames 3-5)
        ("Active", 6),  # OT live play (frames 6-11)
        ("PostGoalScored", 2),  # OT goal
    )
    secs = np.array([1, 0, 0] + [0] * 3 + [0] * 6 + [0] * 2)
    times = np.array([0, 1, 1.5, 2, 3, 4, 5, 5.5, 6, 6.9, 7.2, 7.6, 8, 9], dtype=float)
    return secs, states, times


def test_overtime_clock_from_timestamps():
    secs, states, times = _overtime_match()
    clock, ot = compute_clock(secs, states, times, overtime_start=3)
    assert ot.tolist() == [False] * 3 + [True] * 11
    # Countdown before the OT kickoff shows +0:00; live play counts up from the kickoff (t=5);
    # the post-goal frames freeze at the last live value (t=7.6 -> 2 s).
    assert clock.tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert [format_clock(c, o) for c, o in zip(clock, ot, strict=True)][-1] == "+0:02"


def test_overtime_clock_uses_raw_counter_when_it_counts_up():
    secs, states, times = _overtime_match()
    secs = secs.copy()
    secs[6:] = [0, 0, 1, 1, 2, 2, 2, 2]
    clock, _ = compute_clock(secs, states, times, overtime_start=3)
    assert clock[6:].tolist() == [0, 0, 1, 1, 2, 2, 2, 2]


def test_overtime_flag_frame():
    objects = ["Foo", "TAGame.GameEvent_Soccar_TA:bOverTime", "Bar"]
    frames = [
        {"updated_actors": [{"object_id": 0, "attribute": {"Int": 3}}]},
        {"updated_actors": [{"object_id": 1, "attribute": {"Boolean": False}}]},
        {"updated_actors": []},
        {"updated_actors": [{"object_id": 1, "attribute": {"Boolean": True}}]},
    ]
    assert overtime_flag_frame(frames, objects) == 3
    assert overtime_flag_frame(frames[:3], objects) is None
    assert overtime_flag_frame(frames, ["Foo"]) is None


def test_overtime_fallback_needs_tied_score():
    secs, states, _ = _overtime_match()
    # One goal each before the clock hit 0 -> tied -> OT starts at the first Countdown after 0.
    assert overtime_fallback_frame(secs, states, [0, 1], [True, False]) == 3
    # 2-1 at 0:00: that Countdown is not overtime (and real matches would just end).
    assert overtime_fallback_frame(secs, states, [0, 1, 2], [True, False, True]) is None
    # The OT goal itself (after the start) doesn't affect the tie check.
    assert overtime_fallback_frame(secs, states, [0, 1, 12], [True, False, True]) == 3


def test_overtime_fallback_none_when_clock_never_hits_zero():
    secs = np.array([300, 299, 98, 98])
    states = _states(("Active", 3), ("Countdown", 1))
    assert overtime_fallback_frame(secs, states, [], []) is None


# --------------------------------------------------------------------------- demos / presence


def test_demolished_mask_spans_frozen_and_empty_frames_until_respawn():
    # frames:     0  1  2  3  4  5  6  7  8  9
    has_data = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 1], dtype=bool)  # 4-5 Empty (respawn gap)
    sleeping = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)  # frozen at demo spot
    mask = demolished_mask(10, [2], has_data, sleeping)
    assert np.flatnonzero(mask).tolist() == [2, 3, 4, 5]


def test_demolished_mask_runs_to_end_without_respawn():
    has_data = np.array([1, 1, 1, 0], dtype=bool)
    sleeping = np.array([0, 1, 1, 0], dtype=bool)
    assert demolished_mask(4, [1], has_data, sleeping).tolist() == [False, True, True, True]


def test_is_present_false_only_for_empty_frames_outside_demos():
    # frames 1-2: kickoff-reset Empty frames (not present); frame 6: Empty inside a demo window.
    has_data = np.array([1, 0, 0, 1, 1, 1, 0, 1], dtype=bool)
    sleeping = np.array([0, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
    demolished = demolished_mask(8, [4], has_data, sleeping)
    assert np.flatnonzero(demolished).tolist() == [4, 5, 6]
    present = presence_mask(has_data, demolished)
    assert present.tolist() == [True, False, False, True, True, True, True, True]
    on_field = present & ~demolished
    assert np.flatnonzero(on_field).tolist() == [0, 3, 7]


# --------------------------------------------------------------------------- rotation


def test_quat_to_euler_pure_yaw():
    # 45 degree rotation about z: the blue left-kickoff spawn at (-2048, -2560).
    half = math.radians(45) / 2
    yaw, pitch, roll = quat_to_euler(math.cos(half), 0.0, 0.0, math.sin(half))
    assert float(yaw) == pytest.approx(math.radians(45))
    assert float(pitch) == pytest.approx(0.0)
    assert float(roll) == pytest.approx(0.0)
