# ROADMAP

Living checklist for RL Decision Review. Claude Code updates this as work gets done (see CLAUDE.md, Workflow rules).

## How to use this

1. Pick the next `[TODO]` subsection (respect "Depends on").
2. Paste that subsection into Claude chat and ask: *"Turn this into a Claude Code prompt."*
3. Give that prompt to Claude Code.
4. Claude Code checks off tasks, updates the status tag, and fills in Notes.

Status tags: `[TODO]` · `[IN PROGRESS]` · `[DONE]` · `[BLOCKED]`

---

## Phase 0: Project Setup

### 0.1 Repo and environment `[DONE]`
**Goal:** An empty but working Python project matching the structure in CLAUDE.md.
- [x] `git init`, add `.gitignore` (ignore `replays/*.replay`, `data/`, `out/`, `.venv/`, `.env`)
- [x] Set up `uv` project with Python 3.12 and `pyproject.toml`
- [x] Add deps: `subtr-actor-py`, `pandas`, `numpy`, `pyarrow`, `matplotlib`, `pyyaml`; dev deps: `pytest`, `ruff`
- [x] Create the folder skeleton from CLAUDE.md (`src/rldr/...`, `scripts/`, `tests/`, `labels/`, `replays/`, `data/`, `out/`)
- [x] One passing placeholder test
- [x] Fill in the Commands section of CLAUDE.md

**Done when:** `uv run pytest` and `uv run ruff check .` both pass.
**Depends on:** nothing
**Notes:** Done 2026-11-12. `uv run pytest` (1 passed), `uv run ruff check .`, and `uv run ruff format --check .` all pass.
- Tooling: uv 0.13.7 (installed via winget), uv-managed CPython 3.12.16 pinned in `.python-version`. Git repo on branch `main`, nothing committed yet.
- Package is `rldr` (src layout, `uv_build` backend), so `import rldr` works in tests and scripts. `uv.lock` should be committed (reproducible installs).
- Resolved versions: subtr-actor-py 1.5.1, pandas 3.1.1, numpy 2.5.4, pyarrow 26.0.1, matplotlib 3.11.2, pyyaml 6.0.3, pytest 9.2.0, ruff 0.17.5. Note pandas 3.x (copy-on-write on by default, string dtype changes) when writing 1.2.
- Ruff config in `pyproject.toml`: line length 100, target py312, rules E/F/W/I/UP/B.
- `src/rldr/` subpackages created empty: `parse`, `state`, `detectors`, `danger`, `report`, `viz`. `web/` and `api/` left for Phase 5. `data/` and `out/` exist locally but are gitignored (no `.gitkeep`); `scripts/` and `labels/` have `.gitkeep`.
- Windows quirk: `uv python install 3.12` reports "Missing expected target directory for Python minor version link" every time on this machine, but the install itself is fine and the venv points directly at `cpython-3.12.16-windows-x86_64-none\python.exe`. If uv ever complains about Python, run `uv python install 3.12 --reinstall`.

### 0.2 Replay inventory `[DONE]`
**Goal:** A known set of test replays with notes about what happens in them.
- [x] Put 1 to 3 recent 2s/3s replays in `replays/`
- [x] For each, fill in `replays/NOTES.md` with at least 2 memorable moments (clock time + what happened)
- [x] Include at least one replay with overtime or a forfeit if possible

**Done when:** every replay in `replays/` has an entry in NOTES.md.
**Depends on:** nothing (Ari does this manually)
**Notes:** 2 ranked 2s replays (one full, one forfeit at 1:38). Ari = `ari ツ` in data, `rawun` in-game. Parser goal clocks matched Ari's notes exactly. No overtime replay yet; grab one later for 1.6.

---

## Phase 1: Parsing

### 1.1 Explore the raw parser output `[TODO]`
**Goal:** Understand exactly what subtr-actor-py gives us before designing our own format.
- [ ] Script `scripts/explore.py` that parses one replay and prints: metadata (map, mode, players, teams, score), available feature names/columns, frame count, duration
- [ ] Find and document where these live: ball and car rigid bodies, boost, `SecondsRemaining`, touches, goals, demos, boost pad pickups
- [ ] Note anything missing or weird

**Done when:** a short summary of the raw data shape is written in Notes below.
**Depends on:** 0.1, 0.2
**Notes:** Early findings from a quick test parse in chat (2026-11-12), verify in code:
- subtr-actor-py parses both replays. Python module is `subtr_actor`. Useful functions: `get_replay_meta`, `get_ndarray_with_info_from_replay_filepath`, `parse_replay`, `get_replay_frames_data`, `get_stats_events`, `get_stats_timeline`.
- Raw network frames run ~26 per second (not 30). 10164 raw frames for a 5 min game.
- **Gotcha:** `get_ndarray_with_info_from_replay_filepath` returns far fewer rows than raw frames (3876 vs 10164). Header `Goals[].frame` are RAW network frame indexes, so they do NOT index the ndarray. Map by time: `parse_replay(...)["network_frames"]["frames"][i]["time"]`, then match to the ndarray `current time` column. This gave correct-looking goal clocks for every goal. Figure out why rows are dropped (dead time? frames without all players?) before trusting the ndarray as the frame table.
- `SecondsRemaining` feature works as the in-game clock. Boost comes as raw 0-255.
- Header has `bForfeit`, `TeamSize`, `Team0Score/Team1Score`, `PlayerStats` (name, team, goals, saves, shots, platform), `Goals`, `HighLights`.
- Ari is `ari ツ` on Steam, on team 0 (Blue) in both replays.

### 1.2 Normalized frame table `[TODO]`
**Goal:** One clean DataFrame that everything downstream uses.
- [ ] `src/rldr/parse/frames.py`: replay path to long-format DataFrame with columns like `frame, time, clock, is_overtime, entity (ball/player id), team, x, y, z, vx, vy, vz, rotation (quat or pitch/yaw/roll), boost (0-100), is_demolished`
- [ ] Player metadata table: `player_id, name, team, platform`
- [ ] Clock mapping: `clock` matches in-game clock, overtime handled
- [ ] Handle frames during goal replays and kickoff countdowns (flag them as dead time)
- [ ] Unit tests for boost conversion and clock mapping

**Done when:** for a moment in NOTES.md, filtering the table by that clock time gives sensible positions.
**Depends on:** 1.1
**Notes:**

### 1.3 Event extraction `[TODO]`
**Goal:** A table of discrete events with frame and clock time.
- [ ] Touches (who, when, ball velocity before/after)
- [ ] Goals (scorer, team, clock)
- [ ] Demos (attacker, victim)
- [ ] Kickoffs (start frame of each)
- [ ] Boost pad pickups (player, big or small, location) if the data supports it
- [ ] Tests: goal count matches the scoreboard

**Done when:** goals and kickoffs line up with NOTES.md and the final score.
**Depends on:** 1.2
**Notes:**

### 1.4 Parquet cache `[TODO]`
**Goal:** Parse once, reuse fast.
- [ ] `scripts/parse.py <replay>` writes `data/<replay_id>/frames.parquet`, `events.parquet`, `meta.json`
- [ ] Loader that reads from cache if present, re-parses if the replay changed

**Done when:** a second load is near-instant.
**Depends on:** 1.2, 1.3
**Notes:**

### 1.5 2D top-down renderer `[TODO]`
**Goal:** See any moment of the game to validate the data by eye.
- [ ] `src/rldr/viz/field.py`: draw field outline, goals, big boost pads
- [ ] Draw ball and cars at a given clock time (team colors, name labels, boost amount, velocity arrows)
- [ ] `scripts/render.py <replay> --clock M:SS` saves a PNG to `out/`
- [ ] Clip mode: `--clock M:SS --seconds 5` saves a GIF/MP4 of the window with short trails

**Done when:** Ari checks 2+ moments from NOTES.md and they look right.
**Depends on:** 1.4
**Notes:**

### 1.6 Robustness pass `[TODO]`
**Goal:** Make sure parsing doesn't break on edge cases.
- [ ] Run parse on every replay in `replays/` and report failures
- [ ] Handle overtime, forfeits, players leaving/joining, demos
- [ ] Basic sanity checks (positions in field bounds, no NaN storms, boost 0-100)

**Done when:** all replays parse and pass sanity checks.
**Depends on:** 1.5
**Notes:**

---

## Phase 2: Game State

### 2.1 Field geometry and helpers `[TODO]`
**Goal:** Shared math so detectors don't reinvent it.
- [ ] `src/rldr/state/geometry.py`: own-goal position per team, distance helpers, "direction toward" helpers, field thirds (defensive/mid/offensive per team)
- [ ] Constants in one place (field size, pads, speeds), verified against parsed data
- [ ] Unit tests

**Done when:** tests pass and constants match observed data.
**Depends on:** 1.6
**Notes:**

### 2.2 Goal-side `[TODO]`
**Goal:** Per frame, per player: are they between the ball and their own net?
- [ ] Define goal-side (start simple: closer to own goal than the ball along y; then refine with a cone/angle check)
- [ ] Add `is_goal_side` column
- [ ] Render overlay showing goal-side status

**Done when:** overlay matches Ari's judgment on sample moments.
**Depends on:** 2.1
**Notes:**

### 2.3 Time to ball `[TODO]`
**Goal:** Estimate how soon each player can reach the ball.
- [ ] v1: distance / max(current speed, floor), with a penalty when facing away from the ball
- [ ] Add `time_to_ball` column
- [ ] Sanity check against actual next touches

**Done when:** the player who touches next usually had the lowest time-to-ball shortly before.
**Depends on:** 2.1
**Notes:**

### 2.4 Rotation roles `[TODO]`
**Goal:** Label each player first man / second man / last man per frame.
- [ ] Role logic using goal-side + time-to-ball + distance to own net
- [ ] Smooth roles over time so they don't flicker every frame
- [ ] Different handling for 2s vs 3s
- [ ] Add `role` column; render roles on the overlay

**Done when:** Ari agrees with roles on 5+ sample moments.
**Depends on:** 2.2, 2.3
**Notes:**

### 2.5 Possession `[TODO]`
**Goal:** Know which team (and player) controls the ball.
- [ ] Last touch team/player
- [ ] "Controlled possession" (dribble or repeated touches within a short window) vs loose ball
- [ ] Add `possession_team`, `possession_player`, `is_controlled` columns

**Done when:** possession changes line up with touches and feel right on the overlay.
**Depends on:** 1.3, 2.1
**Notes:**

---

## Phase 3: Detectors (v1)

### 3.0 Detector framework `[TODO]`
**Goal:** A shared structure so every detector outputs the same shape.
- [ ] `Flag` dataclass: `type, clock, frame_start, frame_end, blamed_players, blame_confidence (high/medium/low/shared), severity, what_happened, better_option, debug`
- [ ] Detector interface: takes game-state frames + events, returns list of Flags
- [ ] `config/thresholds.yaml` with per-mode thresholds
- [ ] `scripts/analyze.py <replay>` runs all detectors, prints flags, optionally renders a clip per flag

**Done when:** a dummy detector runs end to end and produces a clip.
**Depends on:** 2.4, 2.5
**Notes:**

### 3.1 Lab replay and labels `[TODO]`
**Goal:** Ground truth to grade detectors against.
- [ ] Ari records a private 2v2 with staged mistakes (see the table in NOTES.md template): 3 double commits, 2 last-man challenges, 2 boost detours, 2 rotation cuts, 3+ clean plays
- [ ] Labels file `labels/<replay>.yaml`: clock, type, blamed player, or `clean`
- [ ] Also label 2+ real ranked replays (key plays only)

**Done when:** labels exist for the lab replay and at least 2 ranked replays.
**Depends on:** 0.2 (Ari does the recording)
**Notes:**

### 3.2 Evaluation harness `[TODO]`
**Goal:** One command that grades all detectors against labels.
- [ ] `scripts/eval.py`: match flags to labels within a time tolerance (about ±2s)
- [ ] Report per detector: hits, misses, false flags, correct blame %
- [ ] Clean-play labels count as false positives if flagged

**Done when:** running eval prints a readable scorecard.
**Depends on:** 3.0, 3.1
**Notes:**

### 3.3 Double commit detector `[TODO]`
**Goal:** Catch two teammates going for the same ball at once.
- [ ] Both within ~1500 uu of ball, both moving toward it, time-to-ball within ~0.4s of each other
- [ ] Outcome check: opponent wins it or gets past both, danger goes up
- [ ] Blame: the player who was not first man; "shared" if roles were unclear
- [ ] Tune thresholds with eval harness

**Done when:** catches all staged double commits, no clean-play flags, correct blame.
**Depends on:** 3.2
**Notes:**

### 3.4 Last-man challenge detector `[TODO]`
**Goal:** Catch the last man leaving net and losing the challenge.
- [ ] Last man leaves goal-side to challenge, opponent touches next or whiff detected
- [ ] "Forced" exception when danger was already high
- [ ] Split blame if second man was out of position

**Done when:** catches staged last-man challenges, skips forced ones.
**Depends on:** 3.2
**Notes:**

### 3.5 Boost starve detector `[TODO]`
**Goal:** Catch boost decisions that left the team exposed.
- [ ] On danger spikes: defender boost < ~20 when they needed to make a play
- [ ] Rewind for cause: big-pad detour while last/second man and ball moving toward own net
- [ ] Better option: small-pad path back to net

**Done when:** catches staged boost detours.
**Depends on:** 3.2
**Notes:**

### 3.6 Rotation cut detector `[TODO]`
**Goal:** Catch players cutting in front of the first man.
- [ ] Teammate is first man; player turns back into play in front of them instead of rotating back post
- [ ] Outcome: double commit or both caught in front of the ball

**Done when:** catches staged cuts without overlapping double-commit flags.
**Depends on:** 3.2, 3.3
**Notes:**

### 3.7 Overcommit detector `[TODO]`
**Goal:** Catch too many players in front of the ball when the opponent gets possession.
- [ ] Count teammates past the ball at opponent's clean possession
- [ ] Blame whoever pushed up last

**Done when:** flags make sense on ranked replays.
**Depends on:** 3.2
**Notes:**

---

## Phase 4: Danger Score and Ranking

### 4.1 Heuristic danger score `[TODO]`
**Goal:** A 0 to 1 estimate of conceding soon, per frame per team.
- [ ] Inputs: ball distance to own net, ball velocity toward net, defenders goal-side, opponent near ball, team boost
- [ ] Weights in config
- [ ] Plot danger over the match; spikes should line up with goals against and big chances

**Done when:** every goal against is preceded by a clear danger spike.
**Depends on:** 2.4, 2.5
**Notes:**

### 4.2 Decision cost and top 3 `[TODO]`
**Goal:** Rank flags by how much they hurt.
- [ ] Severity = danger after decision minus danger before
- [ ] Dedupe overlapping flags (one moment, one flag)
- [ ] Output: top 3 per player and top 3 for the team

**Done when:** the top 3 matches what Ari would pick on 2+ replays.
**Depends on:** 3.3, 4.1
**Notes:**

### 4.3 Training dataset (later) `[TODO]`
**Goal:** Data to learn a better danger score.
- [ ] Pull a few thousand high-rank 2s replays from the ballchasing.com API
- [ ] Build features per frame + label "conceded within 10s"

**Done when:** dataset saved and documented.
**Depends on:** 1.6, 2.5
**Notes:**

### 4.4 Learned danger model (later) `[TODO]`
**Goal:** Replace the heuristic with a trained model.
- [ ] Train a simple model (gradient boosting first), evaluate vs heuristic
- [ ] Swap in behind the same interface

**Done when:** the model beats the heuristic on held-out replays.
**Depends on:** 4.3
**Notes:**

---

## Phase 5: Web App MVP

### 5.1 Analysis API `[TODO]`
**Goal:** Upload a replay, get a report.
- [ ] FastAPI: `POST /analyze` (upload) returns report JSON; `GET /report/{id}`
- [ ] Report JSON schema: match meta, players, flags with clips data (positions for the window)

**Depends on:** 4.2
**Notes:**

### 5.2 Frontend: upload and report page `[TODO]`
**Goal:** The demo-able product.
- [ ] Next.js app in `web/`: upload page, report page
- [ ] Report: per-player tabs, top 3 flags as cards (clock, what happened, better option, blame)

**Depends on:** 5.1
**Notes:**

### 5.3 Moment viewer `[TODO]`
**Goal:** Watch each flagged moment in the browser.
- [ ] 2D canvas replay of the flag window (or evaluate `@rlrml/player` 3D player)
- [ ] Highlight blamed player, show roles

**Depends on:** 5.2
**Notes:**

### 5.4 Storage and share links `[TODO]`
**Goal:** Reports persist and can be sent to teammates.
- [ ] Supabase: store report JSON + replay reference
- [ ] Public share URL per report

**Depends on:** 5.2
**Notes:**

### 5.5 Deploy `[TODO]`
- [ ] Frontend on Vercel, API hosted (Fly/Railway/Render)
- [ ] Env vars and limits (file size, rate limit)

**Depends on:** 5.4
**Notes:**

---

## Phase 6: AI Coaching Layer

### 6.1 Compact match summary `[TODO]`
**Goal:** Turn flags + stats into a small text summary for the LLM (target under ~10k tokens).
- [ ] Never send raw frames
- [ ] Include flags, roles at those moments, key stats

**Depends on:** 4.2
**Notes:**

### 6.2 LLM write-ups `[TODO]`
**Goal:** Coach-style explanations and a "top habits to fix" summary.
- [ ] Prompt + Claude API call (Haiku-tier default), cost logging per report
- [ ] Guardrail: the LLM explains flags, it doesn't invent new ones

**Depends on:** 6.1
**Notes:**

### 6.3 Tone modes `[TODO]`
- [ ] Normal coach mode
- [ ] Opt-in roast mode

**Depends on:** 6.2
**Notes:**

---

## Phase 7: Polish and Pitch

### 7.1 Trophi.ai pitch package `[TODO]`
- [ ] Short demo video (upload, report, flagged moment)
- [ ] One-pager: problem, approach, detector accuracy from eval harness
- [ ] Clean README with screenshots

**Depends on:** 5.5
**Notes:**

---

## Decision Log

| Date | Decision | Why |
|---|---|---|
| 2026-11-12 | Parse replay data, not video | Replays are recorded game state; video adds cost and noise |
| 2026-11-12 | subtr-actor-py as parser | Actively maintained, Python bindings, numeric feature output |
| 2026-11-12 | Deterministic detectors first, LLM last | Detectors must be right before the AI explains them |
| 2026-11-12 | Outcome-anchored detection | Only flag decisions that actually led to danger |
| 2026-11-12 | Python package named `rldr`, src layout, `uv_build` backend | Matches CLAUDE.md structure; src layout keeps tests honest about the installed package |
| 2026-11-12 | Ruff rules E/F/W/I/UP/B, line length 100 | Catches real bugs and import order without being noisy on numeric code |
