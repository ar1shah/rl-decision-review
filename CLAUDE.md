# CLAUDE.md

Context for Claude Code working in this repo. Read this and `ROADMAP.md` before starting any task.

## What this project is

**RL Decision Review**: a Rocket League replay analyzer that finds the decisions that cost a team the most in a match (double commits, last-man challenges, boost starving, rotation cuts, overcommits). It blames the right player, explains what happened, and suggests the better option. The output is a shareable report link.

It does NOT analyze video. `.replay` files are recorded game state (positions, velocities, boost, touches at ~30 Hz). We parse that data and run deterministic detectors on it. An LLM is only used at the very end to write the coaching text from detector output.

Owner: Ari (former Grand Champion in 2v2). His judgment on what counts as a mistake is the ground truth for tuning detectors.

## Workflow rules

1. **Always work from a ROADMAP.md subsection.** Each task maps to an ID like `2.4`. Stay inside its scope. If you find something that belongs elsewhere, add it to the right subsection in ROADMAP.md instead of doing it now.
2. **Update ROADMAP.md when you finish work:**
   - Check off `- [x]` tasks that are done.
   - Change the subsection's status tag (`[TODO]` to `[IN PROGRESS]` to `[DONE]`).
   - Add anything you learned (data quirks, thresholds, decisions) to that subsection's `Notes:` line.
   - Add new decisions to the Decision Log at the bottom of ROADMAP.md.
3. **Ask before guessing on Rocket League logic.** If a rule about rotation, roles, or blame is ambiguous, ask Ari. Do not invent game knowledge.
4. **Validate visually.** Anything involving positions or game state should be checkable with the 2D renderer against a moment Ari remembers (see `replays/NOTES.md`).
5. **Small, testable steps.** Write a test or a validation script for each piece before moving on.
6. **Never commit `.replay` files or generated data** (see `.gitignore`).
7. **Commit often.** Commit after every meaningful working step (a passing test, a working script, a docs update), and always when a ROADMAP subsection is finished. Don't wait to be asked. Keep commits small and focused, run `ruff` and `pytest` before committing, and use messages that start with the subsection ID, e.g. `1.2: add clock mapping and tests`.
8. **No AI attribution in git.** Do NOT add `Co-Authored-By: Claude`, "Generated with Claude Code", or any other Claude/AI attribution line to commit messages or PR descriptions. Commits are authored by Ari only.

## Tech stack

| Layer | Choice |
|---|---|
| Language (analysis) | Python 3.12, managed with `uv` |
| Replay parsing | `subtr-actor-py` (built on boxcars) |
| Data | pandas, numpy, parquet cache |
| Plots / clips | matplotlib (2D top-down) |
| Tests / lint | pytest, ruff |
| API (Phase 5) | FastAPI |
| Frontend (Phase 5) | Next.js on Vercel |
| Storage / share links (Phase 5) | Supabase |
| LLM write-ups (Phase 6) | Claude API, Haiku-tier by default |

## Project structure (target)

```
rl-review/
  CLAUDE.md
  ROADMAP.md
  pyproject.toml
  replays/             # .replay files (gitignored) + NOTES.md (committed)
  data/                # parsed parquet cache (gitignored)
  out/                 # rendered plots / clips (gitignored)
  src/rldr/
    parse/             # replay -> raw data -> normalized frames + events
    state/             # field geometry, goal-side, roles, time-to-ball, possession
    detectors/         # one module per detector, shared Flag schema
    danger/            # danger score (heuristic, later ML)
    report/            # rank flags, build report JSON
    viz/               # 2D renderer, clips
  scripts/             # CLI entry points (parse, render, analyze, eval)
  tests/
  labels/              # ground-truth labels for lab replays (YAML)
  web/                 # Phase 5 frontend
  api/                 # Phase 5 backend
```

## Commands

Fill these in as they get created.

```
uv sync                                   # install deps (creates .venv on Python 3.12)
uv run pytest                             # tests
uv run ruff check .                       # lint
uv run ruff format .                      # format (use --check to verify only)
uv add <pkg>                              # add a runtime dep (uv add --dev <pkg> for dev-only)
uv run python scripts/explore.py [replays/<file>.replay]  # dump raw parser output (1.1); default: all replays
uv run python scripts/frames.py replays/<file>.replay --clock 3:06  # ball/car state at a clock (1.2); +M:SS = OT
```

In code: `from rldr.parse.frames import load_replay` gives `.frames`, `.entities`, `.players`, `.meta`, and `.at_clock("M:SS")`.

Not created yet (planned):

```
uv run python scripts/parse.py replays/<file>.replay                # 1.4
uv run python scripts/render.py replays/<file>.replay --clock 2:10  # 1.5
```

Windows note: uv was installed with winget. If `uv` isn't found in an already-open terminal, open a new one (PATH was updated on install).

## Rocket League domain reference

### Field coordinates (Unreal units, "uu")
- **x**: side to side, about -4096 to +4096
- **y**: goal to goal, about -5120 to +5120. **Blue defends the -y goal, Orange defends the +y goal.**
- **z**: up, 0 (floor) to about 2044 (ceiling)
- Goal mouth: about 1786 wide (x from -893 to +893), 642 tall
- Ball radius: about 92.75 uu
- Big boost pads (100): (±3072, ±4096) and (±3584, 0). Everything else is a small pad (12).

Treat these as approximations. Verify against parsed data and note corrections in ROADMAP.md.

### Speeds
- Car max speed: 2300 uu/s. Supersonic: about 2200+.
- Ball max speed: 6000 uu/s.

### Replay data quirks
- **Boost is 0 to 255 in replays.** Always convert to 0 to 100 in the normalized frame table.
- Frame timing is not perfectly uniform. Use real timestamps, not frame index times 1/30.
- Map in-game clock (`SecondsRemaining`) to frames. Overtime counts up instead of down, so handle that explicitly.
- Player inputs are only partially recorded. Don't build logic that depends on exact inputs.

### Glossary
- **Goal-side**: player is between the ball and their own net.
- **First man**: closest goal-side player to the ball; the one who should be playing it.
- **Second man**: support, positioned for the follow-up or to cover.
- **Last man**: furthest back toward own net. In 2s, whoever isn't first man.
- **Double commit**: two teammates go for the same ball at about the same time and both lose it or get beaten.
- **Last-man challenge**: last man leaves goal-side to challenge, loses it, and the net is open.
- **Rotation cut**: a player re-enters play in front of a teammate who is already first man, instead of rotating back.
- **Overcommit**: too many teammates in front of the ball when the opponent gets clean possession.
- **Danger score**: estimated chance of conceding in the next few seconds from the current game state.
- **Decision cost**: danger after a decision minus danger before. Used to rank flags.

## Detector design principles

- **Outcome-anchored:** start from danger spikes (goals against, open shots, clean opponent possession) and rewind 3 to 8 seconds to find the decision that caused it. Don't flag weird stuff that didn't matter.
- **Every flag has:** type, clock time and frame range, blamed player(s), blame confidence (high/medium/low/shared), severity (decision cost), a plain-language "what happened", and a "better option".
- **Forced plays aren't mistakes.** If danger was already high and the challenge was the only option, don't flag it (or label it "forced").
- **Mode-aware thresholds:** 2s and 3s have different norms. Thresholds live in one config file, not scattered through code.
- **Wrong blame is worse than no blame.** When unsure, use "shared" or skip the flag.
