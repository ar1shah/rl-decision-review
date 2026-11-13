"""Render a moment of a replay top-down (ROADMAP 1.5): PNG snapshot or animated clip.

    uv run python scripts/render.py replays/<file>.replay --clock 3:06            # snapshot
    uv run python scripts/render.py replays/<file>.replay --clock 3:06 --run 1    # next kickoff
    uv run python scripts/render.py replays/<file>.replay --frame 2276            # exact frame
    uv run python scripts/render.py replays/<file>.replay --clock 3:06 --seconds 6  # clip
    uv run python scripts/render.py replays/<file>.replay --clock 3:47 --seconds 3 --after 1
    uv run python scripts/render.py replays/<file>.replay --frames 3300:3480
    uv run python scripts/render.py replays/<file>.replay --kickoff 0 --seconds 3
    uv run python scripts/render.py replays/<file>.replay --kickoff 0              # countdown PNG
    uv run python scripts/render.py replays/<file>.replay --frame 2276 --zoom 1200  # close-up

Snapshot default frame: the first live frame of that clock second (run 0 unless --run).
Clip with --clock: N seconds leading into the END of that clock second (+ --after).
Clips are GIFs (15 fps) by default; --mp4 writes a 30 fps MP4 when ffmpeg is installed.
Non-live frames are skipped in clips unless --include-dead. Output goes to out/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from rldr.parse.frames import load_replay  # noqa: E402
from rldr.viz.clip import DEFAULT_KICKOFF_SECONDS, render_clip, select_frames  # noqa: E402
from rldr.viz.scene import ReplayView, render_snapshot  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def _frames_arg(text: str) -> tuple[int, int]:
    a, b = text.split(":")
    return int(a), int(b)


def _num(x: float) -> str:
    """Dot-free number for file names: 0.5 -> '0p5', 6.0 -> '6'."""
    return f"{x:g}".replace(".", "p")


def _clock_tag(clock: str) -> str:
    return clock.replace("+", "ot").replace(":", "-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("replay", type=Path)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--clock", help="in-game clock, M:SS or +M:SS (overtime)")
    where.add_argument("--frame", type=int, help="snapshot of this raw frame")
    where.add_argument("--frames", type=_frames_arg, help="clip of frames A:B (inclusive)")
    where.add_argument("--kickoff", type=int, help="kickoff number (0 = opening kickoff)")
    parser.add_argument("--run", type=int, default=0, help="stretch of play within the clock")
    parser.add_argument("--seconds", type=float, help="clip length in seconds (makes a clip)")
    parser.add_argument("--after", type=float, default=0.0, help="clip seconds past the clock")
    parser.add_argument("--include-dead", action="store_true", help="keep non-live frames")
    parser.add_argument("--mp4", action="store_true", help="MP4 (needs ffmpeg) instead of GIF")
    parser.add_argument("--fps", type=int, help="override output frame rate")
    parser.add_argument("--dpi", type=int, help="override dpi (PNG 120, clips 80)")
    parser.add_argument("--zoom", type=float, help="top view: half-width (uu) around the ball")
    parser.add_argument("--out", type=Path, help="output file (default: out/<auto name>)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    replay = load_replay(args.replay)
    view = ReplayView(replay)
    rid = view.replay_id[:4]
    clip = args.frames is not None or args.seconds is not None

    if clip:
        idx = select_frames(
            view,
            clock=args.clock,
            run=args.run,
            seconds=args.seconds,
            after=args.after,
            frames=args.frames,
            kickoff=args.kickoff,
            include_dead=args.include_dead,
        )
        if args.frames is not None:
            name = f"{rid}_f{args.frames[0]}-{args.frames[1]}"
        elif args.kickoff is not None:
            name = f"{rid}_kickoff{args.kickoff}_{_num(args.seconds or DEFAULT_KICKOFF_SECONDS)}s"
        else:
            name = f"{rid}_{_clock_tag(args.clock)}_r{args.run}_{_num(args.seconds)}s"
            if args.after:
                name += f"+{_num(args.after)}s"
        out = args.out or OUT_DIR / name
        if args.zoom:
            out = out.with_name(out.name + f"_z{_num(args.zoom)}")
        path = render_clip(
            view,
            idx,
            out,
            mp4=args.mp4,
            fps=args.fps,
            dpi=args.dpi or 80,
            zoom=args.zoom,
        )
        print(
            f"{path}  ({len(idx)} frames {idx[0]}..{idx[-1]}, "
            f"{view.time[idx[-1]] - view.time[idx[0]]:.1f} s, "
            f"{path.stat().st_size / 1e6:.2f} MB)"
        )
        return

    if args.frame is not None:
        frame = args.frame
    elif args.kickoff is not None:
        frame = int(view.kickoffs.iloc[args.kickoff]["countdown_frame"])
    else:
        live = view.clock_frames(args.clock, args.run)
        if not len(live):
            sys.exit(f"no live frames at {args.clock} run {args.run}")
        frame = int(live[0])
    tag = _clock_tag(view.clock_label(frame))
    zoom = f"_z{_num(args.zoom)}" if args.zoom else ""
    out = args.out or OUT_DIR / f"{rid}_{tag}_r{view.run[frame]}_f{frame}{zoom}.png"
    path = render_snapshot(view, frame, out, dpi=args.dpi or 120, zoom=args.zoom)
    print(f"{path}  (frame {frame}, {path.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
