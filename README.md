# RL Decision Review

A Rocket League replay analyzer that finds the decisions that cost your team the most: double commits, last-man challenges, boost starving, rotation cuts and overcommits. For each one it says who was at fault, what happened, and what the better play was.

It works from `.replay` files, not video. Replays are recorded game state (positions, velocities, boost, touches), so the analysis is deterministic detectors running on that data. An LLM is only used at the end to write the coaching text.

> **Status:** early. Project setup is done; replay parsing is next. See [ROADMAP.md](ROADMAP.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 for you.

```
uv sync
```

## Commands

```
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
```

## Layout

```
src/rldr/      # analysis package: parse, state, detectors, danger, report, viz
scripts/       # CLI entry points
tests/         # pytest tests
labels/        # ground-truth labels for test replays
replays/       # .replay files (gitignored) + NOTES.md
```

`ROADMAP.md` tracks the work; `CLAUDE.md` holds project context and Rocket League domain notes.
