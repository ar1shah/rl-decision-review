"""Clip windows and animated clips (ROADMAP 1.5).

Window rules (all by real `time`, not frame counts):
  clock + seconds   N seconds leading into the END of that clock second's live run, plus
                    `after` seconds past it. E.g. 3:06 run 0 for 6 s ends on the goal frame.
  frames A:B        exactly those frames (inclusive).
  kickoff N         from kickoff N's countdown to `seconds` after it goes live. The countdown
                    frames are kept even without include_dead.
Non-live frames (countdown, post-goal) are dropped unless include_dead, except goal frames (the
first PostGoalScored frame, where the ball is in the net and the score flips). A jump in the kept
frames is a "cut": trails restart there and the header shows [cut] briefly.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from rldr.viz.field import TOP_XLIM, TOP_YLIM
from rldr.viz.scene import MARK_SECONDS, TRAIL_SECONDS, Renderer, ReplayView, Trails

GIF_FPS = 15  # GIFs keep every 2nd raw frame (raw is ~30 Hz)
MP4_FPS = 30
CUT_GAP_SECONDS = 0.2  # a time jump bigger than this between kept frames is a cut
CUT_NOTE_SECONDS = 0.4
DEFAULT_KICKOFF_SECONDS = 3.0


def gif_durations(n: int, fps: float) -> list[int]:
    """Per-frame GIF durations (ms) that average exactly 1000 / fps.

    GIF stores durations in 10 ms units, so a flat 66.7 ms (15 fps) becomes 60 ms and plays
    11% fast. Rounding the cumulative time instead gives 70, 60, 70, ... (real time).
    """
    edges = [round(i * 1000 / fps, -1) for i in range(n + 1)]
    return [int(b - a) for a, b in zip(edges[:-1], edges[1:], strict=True)]


class RealTimePillowWriter(PillowWriter):
    """PillowWriter with per-frame durations from gif_durations."""

    def finish(self) -> None:
        self._frames[0].save(
            self.outfile,
            save_all=True,
            append_images=self._frames[1:],
            duration=gif_durations(len(self._frames), self.fps),
            loop=0,
        )


def select_frames(
    view: ReplayView,
    *,
    clock: str | None = None,
    run: int = 0,
    seconds: float | None = None,
    after: float = 0.0,
    frames: tuple[int, int] | None = None,
    kickoff: int | None = None,
    include_dead: bool = False,
) -> np.ndarray:
    """Frame indexes for a clip (see module doc for the window rules)."""
    t = view.time
    keep_dead_before = -1  # frames before this index keep dead frames (kickoff countdown)
    if frames is not None:
        a, b = frames
        if not 0 <= a <= b < view.n:
            raise ValueError(f"frames {a}:{b} outside 0..{view.n - 1}")
        idx = np.arange(a, b + 1)
    elif kickoff is not None:
        if not 0 <= kickoff < len(view.kickoffs):
            raise ValueError(f"kickoff {kickoff} not in 0..{len(view.kickoffs) - 1}")
        k = view.kickoffs.iloc[kickoff]
        start, live = int(k["countdown_frame"]), int(k["frame"])
        span = DEFAULT_KICKOFF_SECONDS if seconds is None else seconds
        end = int(np.searchsorted(t, t[live] + span, side="right")) - 1
        idx = np.arange(start, end + 1)
        keep_dead_before = live
    elif clock is not None:
        if seconds is None:
            raise ValueError("clock windows need seconds")
        live = view.clock_frames(clock, run)
        if not len(live):
            raise ValueError(f"no live frames at {clock} run {run}")
        end = int(live[-1])
        if end + 1 in view.goal_frames:  # the run ends in a goal: show the goal frame too
            end += 1
        idx = np.flatnonzero((t >= t[end] - seconds) & (t <= t[end] + after))
    else:
        raise ValueError("give clock + seconds, frames, or kickoff")
    if not include_dead:
        # Goal frames are the first PostGoalScored frame; keep them so the score flips.
        keep = view.is_live[idx] | (idx < keep_dead_before) | np.isin(idx, list(view.goal_frames))
        idx = idx[keep]
    if not len(idx):
        raise ValueError("window has no frames to show (try include_dead)")
    return idx


ZOOM_MARGIN = 400.0


def zoom_box(points: np.ndarray, half_width: float) -> tuple[np.ndarray, float]:
    """(center, half_width) of a box that holds every point (with a margin), at least as wide
    as `half_width`. The box keeps the top view's shape, so the height limit is scaled."""
    lo, hi = np.nanmin(points, axis=0), np.nanmax(points, axis=0)
    center = (lo + hi) / 2
    x0, x1 = TOP_XLIM
    y0, y1 = TOP_YLIM
    shape = (y1 - y0) / (x1 - x0)
    need = max((hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2 / shape) + ZOOM_MARGIN
    return center, max(half_width, need)


def segments(view: ReplayView, idx: np.ndarray) -> np.ndarray:
    """Segment id per kept frame; a new segment starts at every cut."""
    if not len(idx):
        return np.zeros(0, dtype=int)
    gap = (np.diff(idx) > 1) | (np.diff(view.time[idx]) > CUT_GAP_SECONDS)
    return np.r_[0, np.cumsum(gap)]


def trails_for(view: ReplayView, idx: np.ndarray, seg: np.ndarray, i: int) -> Trails:
    """Trail points ending at idx[i]: same segment, last TRAIL_SECONDS."""
    t_now = view.time[idx[i]]
    j = i
    while j > 0 and seg[j - 1] == seg[i] and t_now - view.time[idx[j - 1]] <= TRAIL_SECONDS:
        j -= 1
    hist = idx[j : i + 1]
    top = {"ball": view.ball["pos"][hist, :2]}
    for key, a in view.cars.items():
        pts = a["pos"][hist, :2].copy()
        pts[a["demolished"][hist] | ~a["present"][hist]] = np.nan
        top[key] = pts
    return Trails(top=top, side_ball=view.ball["pos"][hist][:, [1, 2]])


def render_clip(
    view: ReplayView,
    idx: np.ndarray,
    path: str | Path,
    mp4: bool = False,
    fps: int | None = None,
    dpi: int = 80,
    zoom: float | None = None,
) -> Path:
    """Animate frames `idx` to a GIF (default, 15 fps) or MP4 (30 fps, needs ffmpeg).

    Returns the path written; an MP4 request without ffmpeg warns and writes a GIF instead.
    `zoom` = minimum half-width (uu) of a fixed top-view box around the ball's path over the
    clip (centered on it, grown to fit; see zoom_box).
    """
    path = Path(path)
    if mp4 and shutil.which("ffmpeg") is None:
        warnings.warn("ffmpeg not found; writing a GIF instead", stacklevel=2)
        mp4 = False
    ext = ".mp4" if mp4 else ".gif"
    if path.suffix.lower() in (".gif", ".mp4"):
        path = path.with_suffix(ext)
    else:  # no extension yet (names may contain dots, e.g. "2s+0.5s")
        path = path.with_name(path.name + ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    step = 1 if mp4 else 2
    rate = fps or (MP4_FPS if mp4 else GIF_FPS)
    seg = segments(view, idx)
    shown = list(range(0, len(idx), step))
    if shown[-1] != len(idx) - 1:
        shown.append(len(idx) - 1)
    cut_times = view.time[idx[1:][np.diff(seg) > 0]] if len(idx) > 1 else np.array([])

    renderer = Renderer(view, dpi=dpi)
    if zoom:
        renderer.zoom(*zoom_box(view.ball["pos"][idx, :2], zoom))
    t_first = view.time[idx[0]]

    def update(k: int) -> list:
        i = shown[k]
        frame = int(idx[i])
        state = view.state(frame)
        t = state.time
        marks = view.marks_between(max(t - MARK_SECONDS, t_first - 1e-9), t)
        recent_cut = ((t - cut_times >= 0) & (t - cut_times < CUT_NOTE_SECONDS)).any()
        renderer.draw(
            state,
            trails=trails_for(view, idx, seg, i),
            marks=marks,
            note="[cut]" if recent_cut else "",
        )
        return []

    try:
        anim = FuncAnimation(renderer.fig, update, frames=len(shown), blit=False)
        writer = FFMpegWriter(fps=rate) if mp4 else RealTimePillowWriter(fps=rate)
        anim.save(path, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": "white"})
    finally:
        renderer.close()
    return path
