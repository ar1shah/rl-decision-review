"""Renderer state, windows and output files (ROADMAP 1.5). Headless (Agg, set in conftest).

Most tests use a tiny synthetic replay; the last ones check the 2C98 moments from NOTES.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from rldr.parse.events import build_events
from rldr.parse.frames import ENTITY_COLUMNS, ParsedReplay
from rldr.viz import clip as clip_mod
from rldr.viz.clip import gif_durations, render_clip, segments, select_frames, zoom_box
from rldr.viz.field import HITBOXES
from rldr.viz.scene import ReplayView, label_offsets, quat_axes, render_snapshot

# --------------------------------------------------------------------------- synthetic replay
#
# frame  0-1  Countdown  clock 0:10            Bob absent (kickoff reset Empty frames)
#        2-5  Active     clock 0:10            kickoff at 2, Ann touches at 3, Ann demos Cat at 5
#        6-8  Active     clock 0:09            Ann + Bob touch together at 7 (teammate 50/50)
#        9    PostGoal   clock 0:09            Ann's goal, 1-0
#       10-11 Countdown  clock 0:09

N = 12
STATES = ["Countdown"] * 2 + ["Active"] * 7 + ["PostGoalScored"] + ["Countdown"] * 2
CLOCK = [10] * 6 + [9] * 6
PLAYERS = [("A", "Ann", 0, "Octane"), ("B", "Bob", 0, "Dominus"), ("C", "Cat", 1, None)]
DEMO_SPOT = (100.0, 200.0, 17.0)


def _entity(key: str, team, xs, present=None, demolished=None) -> pd.DataFrame:
    df = pd.DataFrame({c: np.nan for c in ENTITY_COLUMNS}, index=range(N))
    df["frame"] = np.arange(N)
    df["entity"] = key
    df["team"] = pd.array([team] * N, dtype="Int8")
    df["x"], df["y"], df["z"] = xs, -1000.0, 17.0 if key != "ball" else 93.0
    df[["vx", "vy", "vz"]] = (500.0, 0.0, 0.0)
    df[["qw", "qx", "qy", "qz"]] = (1.0, 0.0, 0.0, 0.0)
    df["boost"] = 50.0 if key != "ball" else np.nan
    df["is_present"] = True if present is None else present
    df["is_demolished"] = False if demolished is None else demolished
    dead = df["is_demolished"] | ~df["is_present"]
    df.loc[dead, ["x", "y", "z", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "boost"]] = np.nan
    return df


def _events() -> pd.DataFrame:
    rows = [
        {"frame": 2, "type": "kickoff", "countdown_frame": 0},
        {"frame": 3, "type": "touch", "player_key": "A", "team": 0},
        {"frame": 5, "type": "demo", "player_key": "A", "team": 0, "other_player_key": "C"},
        {"frame": 7, "type": "touch", "player_key": "A", "team": 0, "contest_type": "teammate"},
        {"frame": 7, "type": "touch", "player_key": "B", "team": 0, "contest_type": "teammate"},
        {"frame": 9, "type": "goal", "player_key": "A", "team": 0},
    ]
    ev = pd.DataFrame(rows)
    ev[["x", "y", "z"]] = (0.0, -1000.0, 93.0)
    ev.loc[ev["type"] == "demo", ["x", "y", "z"]] = DEMO_SPOT
    ev["alt_player_key"] = None
    ev.loc[(ev["frame"] == 7) & (ev["player_key"] == "A"), "alt_player_key"] = "B"
    ev["team0_score"] = pd.array([None] * 5 + [1], dtype="Int16")
    ev["team1_score"] = pd.array([None] * 5 + [0], dtype="Int16")
    ev["team"] = ev["team"].astype("Int8")
    return ev


@pytest.fixture(scope="module")
def synthetic() -> ReplayView:
    frames = pd.DataFrame(
        {
            "frame": np.arange(N),
            "time": np.arange(N) / 30.0,
            "seconds_remaining": CLOCK,
            "clock": CLOCK,
            "game_state": pd.Categorical(STATES),
            "is_live": [s == "Active" for s in STATES],
            "is_overtime": False,
            "run": 0,
        }
    )
    x = np.linspace(-500, 500, N)
    ents = [
        _entity("ball", None, x),
        _entity("A", 0, x - 300),
        _entity("B", 0, x + 300, present=np.arange(N) >= 2),
        _entity("C", 1, x, demolished=np.arange(N) >= 5),
    ]
    players = pd.DataFrame(
        [
            {"player_key": k, "name": n, "team": t, "platform": "Steam", "car": c}
            for k, n, t, c in PLAYERS
        ]
    )
    replay = ParsedReplay(
        frames=frames,
        entities=pd.concat(ents, ignore_index=True).sort_values(["frame"], kind="stable"),
        players=players,
        meta={"replay_id": "SYNTHETIC"},
        events=_events(),
    )
    return ReplayView(replay)


# --------------------------------------------------------------------------- state


def test_state_score_cars_and_hitboxes(synthetic):
    s = synthetic.state(4)
    assert s.score == (0, 0) and s.clock == "0:10" and s.is_live
    by = {c.name: c for c in s.cars}
    assert set(by) == {"Ann", "Bob", "Cat"}
    assert by["Bob"].hitbox == HITBOXES["Dominus"]
    assert by["Cat"].hitbox == HITBOXES["Octane"]  # unknown car falls back to Octane
    assert synthetic.state(9).score == (1, 0)  # the goal frame already shows the new score


def test_absent_car_is_omitted(synthetic):
    assert {c.name for c in synthetic.state(0).cars} == {"Ann", "Cat"}


def test_demolished_car_drawn_at_demo_spot(synthetic):
    cat = next(c for c in synthetic.state(6).cars if c.name == "Cat")
    assert cat.demolished
    assert tuple(cat.pos) == DEMO_SPOT
    assert cat.vel is None and cat.forward is None


def test_marks_merge_same_frame_touches(synthetic):
    kinds = [(m.frame, m.kind) for m in synthetic.marks]
    assert kinds == [(3, "touch"), (5, "demo"), (7, "teammate")]
    pair = synthetic.marks[-1]
    assert pair.names == ["Ann", "Bob"]
    demo = synthetic.marks[1]
    assert demo.names == ["Ann", "Cat"] and tuple(demo.pos) == DEMO_SPOT
    t = synthetic.time
    assert [m.frame for m in synthetic.marks_between(t[3], t[7])] == [5, 7]


def test_quat_axes_identity_and_yaw():
    fwd, left, up = quat_axes(1.0, 0.0, 0.0, 0.0)
    assert (
        np.allclose(fwd, [1, 0, 0]) and np.allclose(left, [0, 1, 0]) and np.allclose(up, [0, 0, 1])
    )
    half = np.radians(90) / 2
    fwd, _, _ = quat_axes(np.cos(half), 0.0, 0.0, np.sin(half))
    assert np.allclose(fwd, [0, 1, 0])


def test_label_offsets_stack_close_cars():
    pos = [np.array([0.0, 0.0]), np.array([50.0, 40.0]), np.array([3000.0, 0.0])]
    dy = label_offsets(pos)
    assert dy[0] == 0 and dy[1] > 0 and dy[2] == 0


# --------------------------------------------------------------------------- windows


def test_clock_frames_and_window_drops_dead_frames(synthetic):
    assert synthetic.clock_frames("0:10").tolist() == [2, 3, 4, 5]
    idx = select_frames(synthetic, clock="0:10", seconds=10)
    assert idx.tolist() == [2, 3, 4, 5]  # countdown frames 0-1 dropped
    dead = select_frames(synthetic, clock="0:10", seconds=10, include_dead=True)
    assert dead.tolist() == [0, 1, 2, 3, 4, 5]


def test_window_leads_into_end_of_clock_and_keeps_goal_frame(synthetic):
    idx = select_frames(synthetic, clock="0:09", seconds=0.1)
    assert idx[-1] == 9  # run ends in a goal: the (non-live) goal frame is kept
    assert idx.tolist() == [6, 7, 8, 9]
    longer = select_frames(synthetic, clock="0:09", seconds=10, after=1.0)
    assert longer.tolist() == [2, 3, 4, 5, 6, 7, 8, 9]  # post-goal countdown still dropped


def test_frames_and_kickoff_windows(synthetic):
    assert select_frames(synthetic, frames=(1, 4)).tolist() == [2, 3, 4]
    kick = select_frames(synthetic, kickoff=0, seconds=0.05)
    assert kick.tolist() == [0, 1, 2, 3]  # countdown kept, then 0.05 s of live play
    with pytest.raises(ValueError):
        select_frames(synthetic, clock="0:30", seconds=2)


def test_segments_split_at_cuts(synthetic):
    assert segments(synthetic, np.array([2, 3, 4, 8, 9])).tolist() == [0, 0, 0, 1, 1]


def test_zoom_box_holds_all_points():
    pts = np.array([[0.0, 0.0], [2000.0, 100.0]])
    center, half = zoom_box(pts, 500.0)
    assert np.allclose(center, [1000.0, 50.0])
    assert half >= 1000.0
    assert zoom_box(pts[:1], 800.0)[1] == 800.0


# --------------------------------------------------------------------------- output files


def test_snapshot_png_written(synthetic, tmp_path):
    path = render_snapshot(synthetic, 6, tmp_path / "snap.png", dpi=50)
    assert path.exists() and path.stat().st_size > 5000
    assert Image.open(path).size == (350, 580)


def test_gif_written_at_15_fps_every_second_frame(synthetic, tmp_path):
    path = render_clip(synthetic, np.arange(2, 9), tmp_path / "clip", dpi=40)
    assert path.suffix == ".gif" and path.stat().st_size > 1000
    gif = Image.open(path)
    assert gif.n_frames == 4  # frames 2, 4, 6, 8
    durations = []
    for i in range(gif.n_frames):
        gif.seek(i)
        durations.append(gif.info["duration"])
    assert durations == gif_durations(4, 15) == [70, 60, 70, 70]


def test_gif_durations_average_to_fps():
    d = gif_durations(90, 15)
    assert set(d) <= {60, 70} and sum(d) == 6000


def test_mp4_without_ffmpeg_falls_back_to_gif(synthetic, tmp_path, monkeypatch):
    monkeypatch.setattr(clip_mod.shutil, "which", lambda _: None)
    with pytest.warns(UserWarning, match="ffmpeg"):
        path = render_clip(synthetic, np.arange(2, 5), tmp_path / "x.mp4", mp4=True, dpi=40)
    assert path.suffix == ".gif" and path.exists()


def test_dotted_clip_name_keeps_its_name(synthetic, tmp_path):
    path = render_clip(synthetic, np.arange(2, 5), tmp_path / "a_2s+0.5s", dpi=40)
    assert path.name == "a_2s+0.5s.gif"


# --------------------------------------------------------------------------- 2C98 moments

LAB = "2C980F9741798CF5E4D1A0BFF902C275"


@pytest.fixture(scope="module")
def lab_view(load_lab_replay) -> ReplayView:
    r = load_lab_replay(LAB)
    r.parsed.events = build_events(r.parsed, r.fd, r.stats)
    return ReplayView(r.parsed)


def test_lab_pads_from_meta(lab_view):
    assert len(lab_view.pads) == 34 and (lab_view.pads["size"] == "big").sum() == 6


def test_lab_3_06_clip_shows_demo_and_ends_on_goal(lab_view):
    idx = select_frames(lab_view, clock="3:06", seconds=6)
    assert idx[-1] == 3472  # Degnis98's goal frame
    assert 3390 in idx  # shotofsam demos Ari at 3:09
    assert lab_view.state(3472).score == (0, 1)
    ari = next(c for c in lab_view.state(3400).cars if c.name == "ari ツ")
    assert ari.demolished
    assert ari.pos[:2] == pytest.approx((-3733.2, -750.6), abs=1)


def test_lab_3_47_teammate_double_touch(lab_view):
    mark = next(m for m in lab_view.marks if m.frame == 2276)
    assert mark.kind == "teammate"
    assert sorted(mark.names) == ["CohenLovesJill", "ari ツ"]
