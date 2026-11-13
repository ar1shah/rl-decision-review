"""Replay state at a frame, and drawing it top-down + side view (ROADMAP 1.5).

`ReplayView` turns a `ParsedReplay` into per-entity numpy arrays once, so both snapshots and
clips look up any frame in O(1). `Renderer` owns one figure (field drawn once) and redraws
only the moving parts for each frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from rldr.parse.frames import ParsedReplay, format_clock, parse_clock
from rldr.viz.field import (
    BALL_RADIUS,
    TEAM_COLORS,
    TOP_XLIM,
    TOP_YLIM,
    draw_field,
    draw_side_field,
    hitbox_for,
    pads_from_meta,
)

TRAIL_SECONDS = 1.5  # trail length in clips
MARK_SECONDS = 0.5  # how long a touch / demo mark stays visible (fading)
VEL_ARROW_SECONDS = 0.5  # velocity arrow = where the car is in this many seconds
BALL_GROWTH_Z = 1000.0  # ball drawn at radius * (1 + z / BALL_GROWTH_Z)
BALL_Z_LABEL = 250.0  # label ball height above this z
CAR_Z_LABEL = 120.0  # label car height above this z (a car on the floor sits at z ~17)
SIDE_NOSE_LEN = 300.0  # side view: nose line length for a car pointing fully along y/z
BOOST_BAR_LEN = 300.0
DEMO_COLOR = "#d62728"
TEAMMATE_COLOR = "#c2185b"
BALL_TRAIL_COLOR = "#333333"

_FONT_CANDIDATES = [
    "DejaVu Sans",
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "Microsoft YaHei",
    "Noto Sans CJK JP",
    "Noto Sans JP",
]


def _font_family() -> list[str]:
    """DejaVu first, then CJK-capable fonts that exist here (names like 'ari ツ', 'Rubeen 福')."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    return [n for n in _FONT_CANDIDATES if n in have] or ["sans-serif"]


FONTS = _font_family()


# --------------------------------------------------------------------------- state


def quat_axes(qw: Any, qx: Any, qy: Any, qz: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-space (forward, left, up) unit vectors of a car from its quaternion (w, x, y, z).

    Verified on the lab replays: forward matches velocity while driving, and up points into the
    field on side walls (ROADMAP 1.5 Notes). Works on scalars or arrays; returns shape (..., 3).
    """
    w, x, y, z = (np.asarray(q, dtype=float) for q in (qw, qx, qy, qz))
    forward = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
    left = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1)
    up = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)
    return forward, left, up


@dataclass
class CarState:
    key: str
    name: str
    team: int
    hitbox: tuple[float, float, float]
    demolished: bool
    pos: np.ndarray  # demo location while demolished
    vel: np.ndarray | None = None
    forward: np.ndarray | None = None
    boost: float = float("nan")


@dataclass
class FrameState:
    frame: int
    time: float
    clock: str
    run: int
    game_state: str
    is_live: bool
    score: tuple[int, int]
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    cars: list[CarState]


@dataclass
class Mark:
    """A touch (one per frame, 50/50 rows merged) or a demo, drawn for MARK_SECONDS."""

    kind: str  # "touch" / "opponent" (contested 50/50) / "teammate" / "demo"
    frame: int
    time: float
    pos: np.ndarray
    names: list[str]
    team: int | None = None
    victim_pos: np.ndarray | None = None


@dataclass
class Trails:
    """Recent positions per entity for a clip frame; NaN rows break the line."""

    top: dict[str, np.ndarray] = field(default_factory=dict)  # entity -> (k, 2) x, y
    side_ball: np.ndarray | None = None  # (k, 2) y, z


class ReplayView:
    """Per-entity arrays over all frames of a ParsedReplay (needs `events`)."""

    def __init__(self, replay: ParsedReplay):
        if replay.events is None:
            raise ValueError("ReplayView needs a replay with events (use load_replay)")
        self.replay = replay
        self.replay_id = str(replay.meta.get("replay_id", "replay"))
        self.pads = pads_from_meta(replay.meta)
        f = replay.frames
        self.n = len(f)
        self.time = f["time"].to_numpy(dtype=float)
        self.clock = f["clock"].to_numpy()
        self.is_overtime = f["is_overtime"].to_numpy(dtype=bool)
        self.run = f["run"].to_numpy()
        self.game_state = f["game_state"].astype(str).to_numpy()
        self.is_live = f["is_live"].to_numpy(dtype=bool)

        p = replay.players
        self.names = dict(zip(p["player_key"], p["name"], strict=True))
        self.teams = {k: int(t) for k, t in zip(p["player_key"], p["team"], strict=True)}
        self.hitboxes = {k: hitbox_for(c) for k, c in zip(p["player_key"], p["car"], strict=True)}

        self.ball: dict[str, np.ndarray] = {}
        self.cars: dict[str, dict[str, np.ndarray]] = {}
        for entity, g in replay.entities.groupby("entity", observed=True, sort=False):
            g = g.sort_values("frame")
            if len(g) != self.n or (g["frame"].to_numpy() != np.arange(self.n)).any():
                raise ValueError(f"{entity}: entity rows don't cover every frame")
            arrays = {
                "pos": g[["x", "y", "z"]].to_numpy(dtype=float),
                "vel": g[["vx", "vy", "vz"]].to_numpy(dtype=float),
                "quat": g[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float),
                "boost": g["boost"].to_numpy(dtype=float),
                "present": g["is_present"].to_numpy(dtype=bool),
                "demolished": g["is_demolished"].to_numpy(dtype=bool),
            }
            if entity == "ball":
                self.ball = arrays
            else:
                key = str(entity)
                arrays["forward"] = quat_axes(*arrays["quat"].T)[0]
                self.cars[key] = arrays
                if key not in self.teams:  # not in the roster: fall back to the entity table
                    self.teams[key] = int(g["team"].iloc[0])
                    self.names.setdefault(key, key)
                    self.hitboxes.setdefault(key, hitbox_for(None))

        ev = replay.events
        goals = ev[ev["type"] == "goal"].sort_values("frame")
        self._goal_frames = goals["frame"].to_numpy()
        self.goal_frames = set(self._goal_frames.tolist())
        self._goal_scores = goals[["team0_score", "team1_score"]].to_numpy(dtype=int)
        demos = ev[ev["type"] == "demo"].sort_values("frame")
        self._demos: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for victim, g in demos.groupby("other_player_key"):
            self._demos[str(victim)] = (
                g["frame"].to_numpy(),
                g[["x", "y", "z"]].to_numpy(dtype=float),
            )
        self.marks = self._build_marks(ev)
        self._mark_times = np.array([m.time for m in self.marks])
        self.kickoffs = ev[ev["type"] == "kickoff"].sort_values("frame").reset_index(drop=True)

    # ------------------------------------------------------------------ lookups

    def name(self, key: Any) -> str:
        return self.names.get(str(key), str(key)) if pd.notna(key) else "?"

    def score_at(self, frame: int) -> tuple[int, int]:
        i = np.searchsorted(self._goal_frames, frame, side="right") - 1
        return (0, 0) if i < 0 else (int(self._goal_scores[i, 0]), int(self._goal_scores[i, 1]))

    def demo_position(self, key: str, frame: int) -> np.ndarray | None:
        """Victim location of the latest demo on `key` at or before `frame`."""
        if key not in self._demos:
            return None
        frames, pos = self._demos[key]
        i = np.searchsorted(frames, frame, side="right") - 1
        return None if i < 0 else pos[i]

    def clock_frames(self, clock: str, run: int = 0, live_only: bool = True) -> np.ndarray:
        """Frame indexes of one clock second ('M:SS' / '+M:SS') and run."""
        seconds, ot = parse_clock(clock)
        sel = (self.clock == seconds) & (self.is_overtime == ot) & (self.run == run)
        if live_only:
            sel &= self.is_live
        return np.flatnonzero(sel)

    def clock_label(self, frame: int) -> str:
        return format_clock(self.clock[frame], bool(self.is_overtime[frame]))

    def state(self, frame: int) -> FrameState:
        if not 0 <= frame < self.n:
            raise IndexError(f"frame {frame} outside 0..{self.n - 1}")
        cars = []
        for key, a in self.cars.items():
            common = dict(
                key=key, name=self.name(key), team=self.teams[key], hitbox=self.hitboxes[key]
            )
            if a["demolished"][frame]:
                where = self.demo_position(key, frame)
                if where is not None:
                    cars.append(CarState(demolished=True, pos=where, **common))
                continue
            if not a["present"][frame] or np.isnan(a["pos"][frame]).any():
                continue  # absent (kickoff reset Empty frames, left the game)
            cars.append(
                CarState(
                    demolished=False,
                    pos=a["pos"][frame],
                    vel=np.nan_to_num(a["vel"][frame]),
                    forward=a["forward"][frame],
                    boost=float(a["boost"][frame]),
                    **common,
                )
            )
        return FrameState(
            frame=int(frame),
            time=float(self.time[frame]),
            clock=self.clock_label(frame),
            run=int(self.run[frame]),
            game_state=str(self.game_state[frame]),
            is_live=bool(self.is_live[frame]),
            score=self.score_at(frame),
            ball_pos=self.ball["pos"][frame],
            ball_vel=self.ball["vel"][frame],
            cars=cars,
        )

    # ------------------------------------------------------------------ marks

    def _build_marks(self, ev: pd.DataFrame) -> list[Mark]:
        marks = []
        for frame, g in ev[ev["type"] == "touch"].groupby("frame"):
            keys = list(dict.fromkeys(g["player_key"].dropna().astype(str)))
            alts = [str(k) for k in g["alt_player_key"].dropna() if str(k) not in keys]
            contest = set(g["contest_type"].dropna().astype(str))
            kind = (
                "teammate" if "teammate" in contest else "opponent" if "opponent" in contest else ""
            )
            kind = kind or "touch"
            names = [self.name(k) for k in keys + (alts if kind != "touch" else [])]
            first = g.iloc[0]
            marks.append(
                Mark(
                    kind=kind,
                    frame=int(frame),
                    time=float(self.time[int(frame)]),
                    pos=first[["x", "y", "z"]].to_numpy(dtype=float),
                    names=names,
                    team=None if pd.isna(first["team"]) else int(first["team"]),
                )
            )
        for _, d in ev[ev["type"] == "demo"].iterrows():
            marks.append(
                Mark(
                    kind="demo",
                    frame=int(d["frame"]),
                    time=float(self.time[int(d["frame"])]),
                    pos=d[["x", "y", "z"]].to_numpy(dtype=float),
                    names=[self.name(d["player_key"]), self.name(d["other_player_key"])],
                    team=None if pd.isna(d["team"]) else int(d["team"]),
                )
            )
        return sorted(marks, key=lambda m: m.frame)

    def marks_between(self, t0: float, t1: float) -> list[Mark]:
        """Marks with t0 < time <= t1."""
        if not self.marks:
            return []
        lo = np.searchsorted(self._mark_times, t0, side="right")
        hi = np.searchsorted(self._mark_times, t1, side="right")
        return self.marks[lo:hi]


# --------------------------------------------------------------------------- drawing


def car_polygon(pos: np.ndarray, forward: np.ndarray, hitbox: Sequence[float]) -> np.ndarray:
    """Top-down car outline (x, y): hitbox rectangle with a pointed nose, turned by yaw."""
    length, width = hitbox[0], hitbox[1]
    yaw = np.arctan2(forward[1], forward[0])
    f = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-f[1], f[0]])
    local = [
        (-length / 2, -width / 2),
        (length * 0.3, -width / 2),
        (length / 2 + 20, 0.0),
        (length * 0.3, width / 2),
        (-length / 2, width / 2),
    ]
    return np.array([pos[:2] + a * f + b * left for a, b in local])


LABEL_STEP = 260.0  # vertical spacing between stacked car labels (uu)


def label_offsets(positions: Sequence[np.ndarray], width: float = 900.0) -> list[float]:
    """Extra y offset per car label so labels of nearby cars stack instead of overlapping.

    Labels are placed lowest car first; a label that would land within `width` in x and
    LABEL_STEP in y of an already placed one moves up until it is clear.
    """
    order = sorted(range(len(positions)), key=lambda i: positions[i][1])
    placed: list[tuple[float, float]] = []
    out = [0.0] * len(positions)
    for i in order:
        x, y = positions[i][0], positions[i][1]
        dy = 0.0
        while any(abs(x - px) < width and abs(y + dy - py) < LABEL_STEP for px, py in placed):
            dy += LABEL_STEP
        placed.append((x, y + dy))
        out[i] = dy
    return out


class Renderer:
    """One figure: header, top-down field, side view (y horizontal, z up) underneath."""

    FIG_SIZE = (7.0, 11.6)

    def __init__(self, view: ReplayView, dpi: int = 100):
        self.view = view
        self.fig = plt.figure(figsize=self.FIG_SIZE, dpi=dpi, facecolor="white")
        self.ax_top = self.fig.add_axes((0.02, 0.125, 0.96, 0.828))
        self.ax_side = self.fig.add_axes((0.02, 0.008, 0.96, 0.112))
        draw_field(self.ax_top, view.pads)
        draw_side_field(self.ax_side)
        self.header = self.fig.text(
            0.5, 0.975, "", ha="center", va="center", fontsize=10, fontfamily=FONTS
        )
        self._dynamic: list[Any] = []

    def close(self) -> None:
        plt.close(self.fig)

    def zoom(self, center: Sequence[float], half_width: float) -> None:
        """Show only a box around `center` (x, y) in the top view; keeps the axes' shape."""
        x0, x1 = TOP_XLIM
        y0, y1 = TOP_YLIM
        half_height = half_width * (y1 - y0) / (x1 - x0)
        self.ax_top.set_xlim(center[0] - half_width, center[0] + half_width)
        self.ax_top.set_ylim(center[1] - half_height, center[1] + half_height)

    # ------------------------------------------------------------------ helpers

    def _add(self, artist: Any) -> Any:
        self._dynamic.append(artist)
        return artist

    def _text(self, ax: Any, x: float, y: float, s: str, **kw: Any) -> None:
        kw.setdefault("fontsize", 7)
        self._add(ax.text(x, y, s, fontfamily=FONTS, zorder=9, **kw))

    def _clear(self) -> None:
        for artist in self._dynamic:
            artist.remove()
        self._dynamic.clear()

    # ------------------------------------------------------------------ frame

    def draw(
        self,
        state: FrameState,
        trails: Trails | None = None,
        marks: Sequence[Mark] = (),
        note: str = "",
    ) -> None:
        self._clear()
        v = self.view
        s0, s1 = state.score
        self.header.set_text(
            f"{v.replay_id[:4]}   {state.clock} (run {state.run})   Blue {s0} – {s1} Orange   "
            f"{state.game_state}   frame {state.frame}   t={state.time:.2f}s"
            + (f"   {note}" if note else "")
        )
        if trails is not None:
            self._draw_trails(trails)
        offsets = label_offsets([c.pos for c in state.cars])
        for car, dy in zip(state.cars, offsets, strict=True):
            self._draw_car(car, dy)
        self._draw_ball(state.ball_pos)
        for mark in marks:
            self._draw_mark(mark, max(0.0, 1.0 - (state.time - mark.time) / MARK_SECONDS))

    def _draw_trails(self, trails: Trails) -> None:
        for entity, pts in trails.top.items():
            color = BALL_TRAIL_COLOR if entity == "ball" else TEAM_COLORS[self.view.teams[entity]]
            self._trail(self.ax_top, pts, color, lw=2.0 if entity == "ball" else 1.5)
        if trails.side_ball is not None:
            self._trail(self.ax_side, trails.side_ball, BALL_TRAIL_COLOR, lw=1.5)

    def _trail(self, ax: Any, pts: np.ndarray, color: str, lw: float) -> None:
        if len(pts) < 2:
            return
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        ok = ~np.isnan(segs).any(axis=(1, 2))
        if not ok.any():
            return
        alphas = np.linspace(0.05, 0.7, len(segs))[ok]
        rgba = np.array([to_rgba(color, a) for a in alphas])
        self._add(ax.add_collection(LineCollection(segs[ok], colors=rgba, linewidths=lw, zorder=3)))

    def _draw_car(self, car: CarState, label_dy: float = 0.0) -> None:
        color = TEAM_COLORS[car.team]
        x, y, z = car.pos
        top, side = self.ax_top, self.ax_side
        if car.demolished:
            self._add(top.plot(x, y, marker="X", ms=12, color=DEMO_COLOR, mec="k", zorder=6)[0])
            label = f"{car.name} (demo)"
            self._text(top, x, y + 230 + label_dy, label, ha="center", color="#777")
            self._add(side.plot(y, z, marker="X", ms=7, color=DEMO_COLOR, mec="k", zorder=6)[0])
            return
        poly = car_polygon(car.pos, car.forward, car.hitbox)
        zorder = 6 + min(max(z, 0.0), 2044.0) / 2044.0  # higher things draw on top
        self._add(top.add_patch(Polygon(poly, fc=color, ec="k", lw=0.6, zorder=zorder)))
        speed = float(np.hypot(car.vel[0], car.vel[1]))
        if speed > 50:
            end = (x + car.vel[0] * VEL_ARROW_SECONDS, y + car.vel[1] * VEL_ARROW_SECONDS)
            self._add(
                top.add_patch(
                    FancyArrowPatch(
                        (x, y), end, arrowstyle="-|>", mutation_scale=8, color=color, lw=1, zorder=5
                    )
                )
            )
        # Name, then a boost bar with the number.
        self._text(top, x, y + 300 + label_dy, car.name, ha="center", va="bottom", color="k")
        bx, by = x - BOOST_BAR_LEN / 2, y + 200 + label_dy
        frac = 0.0 if np.isnan(car.boost) else car.boost / 100
        self._add(
            top.add_patch(
                Rectangle((bx, by), BOOST_BAR_LEN, 60, fc="white", ec="#666", lw=0.5, zorder=7)
            )
        )
        self._add(
            top.add_patch(
                Rectangle((bx, by), BOOST_BAR_LEN * frac, 60, fc=color, ec="none", zorder=8)
            )
        )
        boost = "?" if np.isnan(car.boost) else f"{car.boost:.0f}"
        if z > CAR_Z_LABEL:
            boost += f"  z {z:.0f}"
        self._text(top, bx + BOOST_BAR_LEN + 40, by + 30, boost, va="center", fontsize=6)
        if label_dy:  # stacked label: leader line back to the car
            self._add(top.plot([x, x], [y + 60, by], color="#888", lw=0.5, zorder=5)[0])
        # Side view: a dot plus a nose line from the forward vector's (y, z).
        fy, fz = car.forward[1], car.forward[2]
        self._add(side.plot(y, z, "o", ms=3.5, color=color, mec="k", mew=0.4, zorder=6)[0])
        self._add(
            side.plot(
                [y, y + SIDE_NOSE_LEN * fy], [z, z + SIDE_NOSE_LEN * fz], color=color, lw=1.5
            )[0]
        )

    def _draw_ball(self, pos: np.ndarray) -> None:
        x, y, z = pos
        top, side = self.ax_top, self.ax_side
        self._add(
            top.add_patch(Circle((x, y), BALL_RADIUS, fc="k", alpha=0.25, ec="none", zorder=4))
        )
        r = BALL_RADIUS * (1 + max(z, 0) / BALL_GROWTH_Z)
        zorder = 6 + min(max(z, 0.0), 2044.0) / 2044.0
        # Translucent fill so a car under the ball (aerial, goal-line save) still shows.
        self._add(
            top.add_patch(Circle((x, y), r, fc=(1, 1, 1, 0.55), ec="k", lw=1.2, zorder=zorder))
        )
        if z > BALL_Z_LABEL:
            self._text(top, x + r + 60, y, f"z {z:.0f}", va="center", fontsize=6, color="#333")
        self._add(side.add_patch(Circle((y, z), BALL_RADIUS, fc="white", ec="k", lw=1, zorder=7)))

    def _draw_mark(self, mark: Mark, alpha: float) -> None:
        if alpha <= 0:
            return
        x, y = mark.pos[0], mark.pos[1]
        top = self.ax_top
        names = mark.names
        if mark.kind == "demo":
            self._add(top.plot(x, y, marker="*", ms=22, color=DEMO_COLOR, alpha=alpha, zorder=8)[0])
            label = f"demo: {names[0]} → {names[1]}"
            self._text(top, x + 250, y - 250, label, color=DEMO_COLOR, alpha=alpha, fontsize=8)
            return
        if mark.kind == "teammate":
            ring = Circle((x, y), 320, fill=False, ec=TEAMMATE_COLOR, lw=2, ls="--", alpha=alpha)
            label, color = "teammates: " + " + ".join(names), TEAMMATE_COLOR
        elif mark.kind == "opponent":
            ring = Circle((x, y), 320, fc=(1, 1, 1, 0.3 * alpha), ec="k", lw=2, alpha=alpha)
            label, color = "50/50: " + " vs ".join(names), "k"
        else:
            color = TEAM_COLORS.get(mark.team, "k") if mark.team is not None else "k"
            ring = Circle((x, y), 260, fc=to_rgba(color, 0.35 * alpha), ec=color, lw=1.2)
            label = "touch: " + ", ".join(names)
        self._add(top.add_patch(ring))
        ring.set_zorder(8)
        r = ring.get_radius()
        self._text(
            top, x, y - r - 60, label, ha="center", va="top", color=color, alpha=alpha, fontsize=8
        )

    # ------------------------------------------------------------------ output

    def save_png(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, facecolor="white")
        return path


def render_snapshot(
    view: ReplayView,
    frame: int,
    path: str | Path,
    dpi: int = 120,
    marks: bool = True,
    zoom: float | None = None,
) -> Path:
    """PNG of one frame. Touch / demo marks from the last MARK_SECONDS are drawn too.

    `zoom` = half-width (uu) of a top-view box centered on the ball.
    """
    renderer = Renderer(view, dpi=dpi)
    try:
        state = view.state(frame)
        if zoom:
            renderer.zoom(state.ball_pos, zoom)
        recent = view.marks_between(state.time - MARK_SECONDS, state.time) if marks else []
        renderer.draw(state, marks=recent)
        return renderer.save_png(path)
    finally:
        renderer.close()
