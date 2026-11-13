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

### 1.1 Explore the raw parser output `[DONE]`
**Goal:** Understand exactly what subtr-actor-py gives us before designing our own format.
- [x] Script `scripts/explore.py` that parses one replay and prints: metadata (map, mode, players, teams, score), available feature names/columns, frame count, duration
- [x] Find and document where these live: ball and car rigid bodies, boost, `SecondsRemaining`, touches, goals, demos, boost pad pickups
- [x] Note anything missing or weird

**Done when:** a short summary of the raw data shape is written in Notes below.
**Depends on:** 0.1, 0.2
**Notes:** Done 2026-11-12, verified in code with `uv run python scripts/explore.py` on both replays (subtr-actor-py 1.5.1, module `subtr_actor`). Supersedes the early chat findings; two of those were wrong (frame rate, "dropped rows").
- **Three useful entry points:**
  - `parse_replay(bytes)`: header `properties` (`MapName`, `TeamSize`, `Team0Score/Team1Score`, `bForfeit` (key only present when true), `TotalSecondsPlayed`, `NumFrames`, `PlayerStats`, `Goals`, `HighLights`), `network_frames.frames[i].time`, and `names` (string table the game-state ids index into).
  - `get_replay_frames_data(path)`: **native rate, one entry per raw network frame**, plus pre-extracted events. This is the richest source.
  - `get_stats_events(path)["events"]`: library-computed event streams, each `{meta: {stream, primary_player, timing}, payload: {kind, payload}}`.
  - `get_ndarray_with_info_from_replay_filepath`: convenience resampled numeric array (see row-count note).
- **Frame rate:** raw frames are ~30 Hz during play (median dt 0.0333 s on `0AAC`, 0.0347 s on `2C98`; header `RecordFPS=30`). The earlier "~26 Hz" was the mean dt, dragged down by **unrecorded goal replays**: after every goal there is one 4.5-9 s gap between consecutive raw frames (state `PostGoalScored` -> `Countdown`). 5 gaps per replay here, ~35 s total. Timestamps are otherwise near-uniform but not exact; use `time`, not index/30.
- **Why the ndarray has fewer rows (answered):** nothing is dropped. `fps` defaults to **10.0**, so rows = replay span x 10 (3876 = 387.5 s x 10; 2866 = 286.6 s x 10). `fps=30` gives 11627 rows (more than the 10164 raw frames). Resampling is **not interpolation**: each row copies the first raw frame at or after the sample time. Rows that land in a goal-replay gap (342 and 365 rows) are filled with the frame after the gap, i.e. fake held data. Countdown and post-goal frames are kept. Header `Goals[].frame` and all event `frame` fields are raw network frame indexes, never ndarray rows.
- **Where things live** (`fd = get_replay_frames_data(path)`):
  - Clock / state: `fd.frame_data.metadata_frames[i]` = `{time, seconds_remaining (int), replicated_game_state_name, replicated_game_state_time_remaining}`. State ids are indexes into `parse_replay()["names"]`: seen `Countdown`, `Active`, `PostGoalScored` (the numeric ids differ per replay, so always resolve through `names`). `replicated_game_state_time_remaining` is the kickoff countdown (3 during countdown, else 0).
  - Ball: `fd.frame_data.ball_data.frames[i].Data.rigid_body` = `{location, rotation (quat w,x,y,z), linear_velocity, angular_velocity, sleeping}`. Velocities are **null while the ball sleeps** (kickoff spawn until first touch, and post-goal), ~1100 frames per replay.
  - Cars: `fd.frame_data.players[k] = [remote_id, {frames: [...]}]`, `frames[i]` is `{"Data": {...}}` or `"Empty"`. Data keys: `rigid_body`, `boost_amount`, `boost_active`, `jump_active`, `dodge_active`, `double_jump_active`, `powerslide_active`, `is_team_0`, `player_name`, `camera`, `input` (empty). `Empty` frames are rare (5-34 per player) and sit at goal resets and demo respawns.
  - Boost: `boost_amount` is **raw 0-255** in both frames_data and the ndarray (kickoff = 85 = 33.3%). Normalize `* 100 / 255`.
  - Goals: `fd.goal_events` (`frame`, `time`, `player`, `scoring_team_is_team_0`, running score). Same frames as header `Goals[]` (header is 1 frame later on 3 of 10 goals).
  - Demos: `fd.demolish_infos` (`frame`, `time`, `seconds_remaining`, attacker/victim ids, locations, velocities). 6 in `2C98`, 1 in `0AAC`. Also `get_stats_events` stream `demolition`, and ndarray player feature `PlayerDemolishedBy` (-1 = not demoed, else 0-3; presumably the attacker's player index, unverified).
  - Boost pads: `fd.boost_pads` (34 pads: id, position, `Small`/`Big`; 28 small, 6 big) and `fd.boost_pad_events` (`PickedUp` with player + position, or `Available` = respawn). Player is never null on pickups. Also stats stream `boost_pickups`.
  - Touches: see quirk below.
- **ndarray feature adders** (no listing function exists; found by probing, full columns printed by the script): global `BallRigidBody`, `BallRigidBodyBasis`, `BallRigidBodyQuaternions`, `BallRigidBodyNoVelocities`, `VelocityAddedBallRigidBodyNoVelocities`, `InterpolatedBallRigidBodyNoVelocities`, `BallHasBeenHit`, `SecondsRemaining`, `CurrentTime`, `FrameTime`, `ReplicatedStateName`, `ReplicatedGameStateTimeRemaining`, `ThreatModelValues`, `ThreatModelFeatures`, `ThreatEntityFeatures`; player `PlayerRigidBody`, `PlayerRigidBodyBasis`, `PlayerRigidBodyQuaternions`, `PlayerRigidBodyNoVelocities`, `VelocityAddedPlayerRigidBodyNoVelocities`, `InterpolatedPlayerRigidBodyNoVelocities`, `PlayerBoost`, `PlayerJump`, `PlayerAnyJump`, `PlayerDodgeRefreshed`, `PlayerDemolishedBy`, `PlayerBallDistance`, `PlayerRelativeBallPosition`, `PlayerRelativeBallVelocity`, `PlayerLocalRelativeBallPosition`, `PlayerLocalRelativeBallVelocity`. Defaults: `BallRigidBody` + `PlayerRigidBody, PlayerBoost, PlayerAnyJump`. Ndarray rotation from `*RigidBody` is 3 values (likely Euler), not the quat.
- **Goal clocks validated:** `metadata_frames[goal_event.frame].seconds_remaining` gives the clock; matching by `time` gives the same answer. All 10 goals (5 per replay) match NOTES.md clock and scorer. Final scores 3-2 and 4-1 match the header; `bForfeit=True` on `0AAC`, whose clock stops at 1:38.
- **Quirks / missing:**
  - **`fd.touch_events` never has a player** (`player: null` on 123/123 and 84/84), only `team_is_team_0`. The attributed source is `get_stats_events` stream `touch` (183 and 110 touches, all with player, plus ball position/speed change, role, play_depth). The two sources disagree a lot: only 73 of 123 `touch_events` frames appear in the stats stream on `2C98`. 1.3 must pick one and validate against NOTES.md saves.
  - **Demo victims:** the car is not removed. For ~2.9 s after a demo its frames are still `Data` with position frozen at the demo spot, velocity null, and boost draining to 0, then 1-2 `Empty` frames, then respawn with boost 85. 1.2 must flag `is_demolished` from `demolish_infos` (or null velocity), not trust that position.
  - Player ids are platform-tagged dicts (`{"Steam": "..."}`, `{"Epic": "..."}`, `{"Xbox": "..."}`, `{"PlayStation": {name, online_id, ...}}`). `PlayerStats.Platform` shows `PS4` / `Dingo` (Xbox) / `Epic` / `Steam`.
  - `replay_tick_marks` (`Team0Save`, `Team1Goal`, `User`, ...) sit 1-2 s before the event clock; fine as hints, not as event times.
  - `get_stats_events` also offers streams we may reuse or compare against later: `kickoff`, `fifty_fifty`, `whiff`, `beaten_to_ball`, `possession`, `player_possession`, `rotation_role`, `first_man_change`, `depth_role`, `shadow_defense`, `bump`, etc. Treat these as library heuristics, not ground truth.
  - No overtime replay yet, so overtime clock behavior is untested (script checks for it).

### 1.2 Normalized frame table `[DONE]`
**Goal:** One clean DataFrame that everything downstream uses.
- [x] `src/rldr/parse/frames.py`: replay path to long-format DataFrame with columns like `frame, time, clock, is_overtime, entity (ball/player id), team, x, y, z, vx, vy, vz, rotation (quat or pitch/yaw/roll), boost (0-100), is_demolished`
- [x] Player metadata table: `player_id, name, team, platform`
- [x] Clock mapping: `clock` matches in-game clock, overtime handled
- [x] Handle frames during goal replays and kickoff countdowns (flag them as dead time)
- [x] Unit tests for boost conversion and clock mapping

**Done when:** for a moment in NOTES.md, filtering the table by that clock time gives sensible positions.
**Depends on:** 1.1
**Notes:** Done 2026-11-12. 53 tests pass (26 synthetic unit tests, 27 against both replays). The done-when is checked by a test plus `scripts/frames.py`: at 3:06 the ball is in Blue's net next to Degnis98, at 0:38 it's in Orange's net next to Ari, and at 3:47 Cohen is in the net under the ball.
- **API:** `load_replay(path) -> ParsedReplay` with `.frames`, `.entities`, `.players`, `.meta`, `.at_clock("M:SS" | "+M:SS")`. `build_replay(id, raw, fd)` takes already-parsed output (tests reuse it). Load takes ~1 s. Not yet cached (1.4).
- **`frames`** (one row per raw network frame; `frame` = the raw index every event uses): `frame, time, seconds_remaining, clock, game_state` (category), `is_live` (state `Active` only), `is_overtime`. Goal replays are absent from the data (time gaps), so they never get rows. The frames on each side of a gap are `PostGoalScored` / `Countdown`, i.e. not live.
- **`entities`** (long format, `n_frames x (1 + n_players)` rows, sorted by frame then ball-first): `frame, entity` (`"ball"` or player key), `team` (Int8, NA for ball), `x y z vx vy vz wx wy wz qw qx qy qz yaw pitch roll boost` (0-100), `boost_active jump_active dodge_active double_jump_active powerslide_active` (nullable boolean), `sleeping, is_present, is_demolished`. Time and clock live only in `frames`; join on `frame`.
- **Player key** = `"<Platform>:<id>"` from `remote_id` (`Steam:7656...`, `Epic:...`, `Xbox:...`, `PlayStation:<online_id>`). Don't use `PlayerStats.OnlineID` (it's `'0'` for Epic). `players` = `player_key, name, team, platform, car`.
- **Clock:** `clock` = `SecondsRemaining` in regulation. It only changes during live play. After a goal, the kickoff restarts at the same clock, and the clock stays frozen until the kickoff touch. So one clock second can contain two stretches of play. `at_clock` numbers them in a `run` column (run 0 of a goal's clock = before the goal, run 1 = next kickoff).
- **Overtime (UNVERIFIED, no OT replay yet):** the header has no OT key. Instead we read the replicated network attribute `TAGame.GameEvent_Soccar_TA:bOverTime` (`parse_replay()["network_frames"]["frames"][k]["updated_actors"]`, value `{"Boolean": true}`). `is_overtime` is true from the first frame it's set. If it's never set, the fallback is the first `Countdown` after the clock hits 0 with a tied score (from `goal_events`). `meta["overtime_source"]` says which was used (`flag` / `fallback` / None). OT `clock` = whole seconds since the OT kickoff went live, frozen outside live play, shown `+M:SS`. If the raw counter counts up in OT, it is used directly. Only tested on synthetic data; see 1.6.
- **Demos:** `is_demolished` runs from the `demolish_infos` frame until respawn (the first later frame with data and `sleeping == False`). It includes `Empty` frames in that window and can span a goal reset (Ari, 2C98 2:04, 169 frames). While demolished, position, velocity, angular velocity, rotation, boost and car flags are NaN/NA. Car `rigid_body.sleeping` is true exactly during demo windows (tested).
- **`is_present`:** False only for `Empty` car frames outside demo windows. Today those are the first 1-3 `Countdown` frames of a goal reset; later, players who left. **Use `is_present & ~is_demolished` for "this car is actually on the field."** The ball is always present.
- **Ball:** null velocity and angular velocity while sleeping (kickoff spawn, post-goal) become 0.
- **Rotation:** quaternion `w,x,y,z` plus ZYX yaw/pitch/roll in radians. Yaw is verified (kickoff cars face the ball within 10°). Pitch/roll signs are not yet verified against RL; check them in the 1.5 renderer.

### 1.3 Event extraction `[DONE]`
**Goal:** A table of discrete events with frame and clock time.
- [x] Touches (who, when, ball velocity before/after): stats stream + physics, with contested / confidence (rule below)
- [x] Goals (scorer, team, clock)
- [x] Demos (attacker, victim)
- [x] Kickoffs (start frame of each)
- [x] Boost pad pickups (player, big or small, location) if the data supports it
- [x] Shots / saves / assists from the game's own stat credits (added; ground truth for touches)
- [x] Tests: goal count matches the scoreboard; demos, kickoffs, saves, scorers, touches (see `tests/test_events_replays.py`, `tests/test_touches_unit.py`)

**Done when:** goals and kickoffs line up with NOTES.md and the final score.
**Depends on:** 1.2
**Notes:** From 1.1: goals, demos, and boost pad pickups are pre-extracted in `get_replay_frames_data` (`goal_events`, `demolish_infos`, `boost_pad_events` + `boost_pads`). Touches: `touch_events` has no player; `get_stats_events` stream `touch` does, but the two disagree on many frames. Decide which to trust (or derive our own) using the NOTES.md saves. `get_stats_events` also has a `kickoff` stream. From 1.2: kickoff starts are the `Countdown -> Active` transitions in `frames.game_state`; the ball is `sleeping` from then until the kickoff touch, and the clock is frozen until that touch too.
- **Events table (built, `src/rldr/parse/events.py`):** `build_events(parsed, fd)`; `load_replay` fills `ParsedReplay.events`. Common columns `frame, time, clock, is_overtime, run, type, player_key, team, x, y, z`, plus nullable per-type columns (see the module docstring). Same-frame order: kickoff, touch, shot, save, goal, assist, demo, boost_pickup. `frames` gained a `run` column (`clock_run`), the same numbering `at_clock` used.
- **Goals / demos / kickoffs / pickups:** goals 5+5, which matches the scoreboard and NOTES.md. Demos 6+1 (`team` = attacker's team; victim in `other_player_key`). Kickoffs: 6 on 2C98 (goals + 1) and 5 on 0AAC: the forfeit replay ends in `Countdown` after the last goal, so that countdown is not a kickoff. `first_touch_frame` = first frame the ball wakes up. Pickups: 375 / 269 `PickedUp` events (the stats `boost_pickups` stream has fewer: 339 / 237). 10-15 per replay happen in non-live frames (`is_live` column). Big pads verified at (±3072, ±4096), (±3584, 0).
- **Shot / save / assist** from `fd["player_stat_events"]` (the game's own credits; 12/6/1 and 9/3/1). The credit frame **trails** the touch: saves by 4-11 frames (one at 30), shots by up to 73. `replay_tick_marks` `TeamNSave` / `TeamNGoal` pair one-to-one with those events, 1.5-2.5 s earlier. NOTES.md saves all match a game save credit: 3:47 Cohen, 1:58 Degnis98, 0:22 = Ari at frame 9208 (clock 0:23; Cohen also saves at 0:21), 2:37 NickyFuego1406.
- **Touch evidence (`uv run python scripts/touches.py`), 2C98 / 0AAC:**
  - `touch_events` is mostly a "hitting team changed" signal. It contains 87-88% of stats touches that change the hitting team but only 18-23% of same-team touches, and it misses goal shots (the last entry before the 0:54 / 1:49 goals is 213 / 279 frames earlier). Not usable for touches.
  - Stats stream vs physics (ball |Δv| > 300 uu/s, nearest on-field car within 250 uu, minus bounce-signature frames), same player within ±3 f: 84% / 82% of stats touches are in physics; 94% / 96% of physics touches are in stats. When they match, the physics frame equals the stats `payload.frame` 78% of the time and is otherwise mostly 1-2 frames later.
  - Stats-only touches are mostly soft (stats ball speed change < 300; 14 / 13), then same-frame pairs (6 / 3) and attribution conflicts (7 / 2). Physics-only touches: 11 / 4, mostly the stats stream crediting the second-nearest car.
  - Center-distance can't settle 50/50s: 29 / 185 and 18 / 106 physics touches have a second car in contact range within 30 uu of the nearest (26 / 16 of them opponents). Example: 3:47, Cohen 176 uu vs Ari 178 uu.
  - Goals: the scoring team's last touch = scorer on 10/10 for both stats and physics.
  - Saves/shots: the credited player is their team's last toucher before the credit on 29/30 for both. The exception is 0AAC 4:12: Ari's shot is credited after iDrxp6442 touched it in a 50/50 at Nicky's save, so the game keeps the shot with the shooter.
- **Ball-surface exclusion (Ari):** the geometry was checked against ball extents (floor z ≈ 93, ceiling 2044, side walls 4096, back walls 5120, corner planes |x|+|y| = 8064, goal 893 x 642).
  - Proximity only (ball within radius + 40 uu of a surface) removes 109 / 46 of 207 / 114 in-range candidates, mostly floor. **86% / 96% of those are stats-confirmed touches by the same player**, i.e. ground hits.
  - Proximity + bounce signature (velocity into the surface reverses and Δv is within 30° of the normal) removes 22 / 8; 64% / 88% of those are still stats-confirmed, since pops and pinches look like bounces.
  - So the surface rule should only gate physics-only additions, not filter physics as a whole.
- **Touch rule (built, `rldr.parse.touches.build_touches`, agreed with Ari):**
  - Stats `touch` stream is the base list. Same-player repeats within 2 frames are one touch, except a touch that is half of a same-frame pair (0AAC 4:37: iDrxp at 1112, then iDrxp + Ari together at 1113; deduping that pair broke the scorer check).
  - Each stats touch is matched to an in-range physics touch within ±3 frames. Candidates are frames where the player is the nearest car, or second and within 30 uu of it, closest in time first. A frame where the player is in range but more than 30 uu behind the nearest car is used only as a fallback. Picking "nearest car" frames first was wrong at the 2C98 0:02 kickoff: shotofsam's 50/50 at 10070 got moved to 10073. On a match the touch moves to the physics frame (`touch_source=both`, `stats_frame` keeps the original).
  - `contested` + `alt_player_key` + `contest_type` (`opponent` / `teammate`): stats credits 2+ players on one frame (one row each), or physics has another car in contact range within 30 uu. The teammate case is a double-commit signal for 3.3. Who won a 50/50 is not decided here (2.5).
  - `source_conflict`: physics puts a different car on the ball. `alt_player_key` = physics' car.
  - Physics-only additions need all of: |Δv| >= 500, exactly one car in range, no surface contact, no bounce, and no stats touch within ±10 frames. Only 1 in each replay set (2C98 4:19 Degnis98).
  - Confidence: high = both sources agree, not contested. medium = one source, or an opponent 50/50 with the credited car in range. low = teammate-contested, a source conflict, or an opponent 50/50 with the car out of range.
  - `x, y, z` for touches is the ball at frame-1 (just before contact). `ball_speed_before/after` are frame-1 / frame. `car_ball_dist` is the credited car (min of f-1, f).
  - Shot / save rows get `touch_frame` = the credited player's last touch within 90 frames before the credit.
- **Touch counts (2C98 / 0AAC):** 180 / 106 touches. Source: both 170 / 99, stats only 9 / 7, physics only 1 / 0.
  - Confidence: **high 139 / 76, medium 30 / 24, low 11 / 6.**
  - Contested: opponent 21 / 18, teammate 7 / 3. Source conflicts: 5 / 3.
  - 3:47 (2C98) comes out as a teammate 50/50 (Ari + Cohen on frame 2276, both low) followed by Cohen's save credit. 0:22 is Ari at 9203 (Δv 2724). 2:37 (0AAC) is Nicky at 5288.
- **Checks that pass on both replays:**
  - Every goal's scorer is among their team's last touchers.
  - Every save credit's player is among their team's last touchers, and `touch_frame` is that touch.
  - Every shot and save has the credited player's touch within 90 frames.
  - Every kickoff's `first_touch_frame` has a touch row.
  - No touch rows in non-live frames.
  - Rows that are both-source and not in conflict have the car within 250 uu.
- CLI: `uv run python scripts/events.py replays/<file>.replay [--clock M:SS] [--type touch]`.

### 1.4 Parquet cache `[DONE]`
**Goal:** Parse once, reuse fast.
- [x] `scripts/parse.py <replay>` writes `data/<replay_id>/frames.parquet`, `entities.parquet`, `players.parquet`, `events.parquet`, `meta.json`
- [x] Loader that reads from cache if present, re-parses if the replay changed (`load_replay(path, use_cache=True)`)
- [x] Invalidate on replay hash, `CACHE_VERSION`, parse-code hash, or subtr-actor version change
- [x] Round-trip test: cached tables equal fresh ones exactly, dtypes included
- [x] Cold vs cached load times recorded below

**Done when:** a second load is near-instant.
**Depends on:** 1.2, 1.3
**Notes:** Done 2026-11-12. 21 new tests (15 synthetic in `tests/test_cache_unit.py`, 6 on the lab replays in `tests/test_cache_replays.py`); 148 pass in total.
- **API:** `rldr.parse.cache` (`CACHE_VERSION`, `cache_status`, `read_cache`, `write_cache`, `code_hash`). `load_replay(path, use_cache=True, cache_root=None)` reads `data/<replay_id>/` on a hit, otherwise parses and rewrites it. `use_cache=False` parses and never touches the cache. `replay_id` = file stem. `scripts/frames.py` and `scripts/events.py` now use the cache automatically. Tests still parse from raw `sa.*` (conftest), except the round-trip test, which writes to `tmp_path`.
- **Layout:** one parquet file per table (`index=False`, default snappy) plus `meta.json` = `{"cache": {version, code_sha256, subtr_actor_version, replay_sha256, written_at}, "meta": ParsedReplay.meta}`. `meta.json` is deleted first and written last (temp file + `os.replace`), so an interrupted write reads as "no cache" and never as a half-valid cache.
- **Invalidation** (`cache_status` reasons, checked in this order): `no cache`, `unreadable` (bad JSON; a corrupt parquet file is also a miss), `version changed`, `code changed`, `parser changed`, `hash changed`, `missing table`. A miss is never an error; it just re-parses.
  - `code_sha256` (Ari) = sha256 over every `.py` under `src/rldr/parse/`, sorted by relative path (path + bytes, so renames count). Any edit to parsing or touch logic invalidates the cache with no version bump. CRLF is read as LF, so git autocrlf on checkout doesn't cause a spurious re-parse. Keep bumping `CACHE_VERSION` on intentional schema changes anyway (it documents them).
  - subtr-actor version is also checked: a parser upgrade can change output with no schema change.
- **Dtypes:** pandas 3.1.1 + pyarrow 26 round-trip every column exactly with no workarounds: categoricals (categories and order, incl. unused ones like `type`), `Int8/Int16/Int64` with NA, `boolean` with NA, `string` with NA, pandas-3 `str` (players). The test checks `dtypes.equals`, category lists, and `assert_frame_equal(check_exact=True)` per table, plus that the fresh tables actually contain all six tricky dtypes.
- **Timings** (`uv run python scripts/parse.py`, this laptop):

  | Replay | cold parse | write | cached load | cache size (replay) |
  |---|---|---|---|---|
  | 2C98 (10164 frames) | 2.72-2.78 s | 0.07 s | 0.015-0.05 s | 5.29 MB (1.17 MB) |
  | 0AAC (7498 frames) | 1.86-1.91 s | 0.07-0.08 s | 0.05 s | 3.87 MB (0.85 MB) |

  Cached load includes hashing the replay and the code. In a fresh process, end to end (incl. imports): 3.27 s -> 0.60 s (2C98) and 2.44 s -> 0.60 s (0AAC). Of that 0.60 s, 0.57 s is importing pandas/pyarrow/subtr_actor, so the cache read itself is ~30 ms.
- `entities.parquet` is 96% of the cache. zstd or float32 could shrink it if storage ever matters (Phase 5); not needed now.
- Verified: `scripts/events.py ... --clock 3:47` output is identical from cache and from a fresh parse; appending a byte to a copy of a replay gives `hash changed` and a re-parse.

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
- [ ] Verify overtime on a real OT replay: `is_overtime` from the `bOverTime` flag (and that the tied-at-0:00 fallback agrees), OT `clock` vs the in-game `+M:SS`, and add its goal clocks to NOTES.md + tests (1.2 OT logic is synthetic-tested only)
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
**Notes:** From 1.3: touches carry `contested` / `contest_type` / `alt_player_key`. Same-frame pairs are one row per player, and who won a 50/50 is decided here, from where the ball goes next (Ari). Use `confidence` to avoid assigning possession on `low` touches alone.

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
**Notes:** From 1.3: touches with `contest_type == "teammate"` (two teammates on the ball within a frame or 30 uu of each other) are a direct double-commit signal: 7 on 2C98, 3 on 0AAC, including 3:47 (Ari + Cohen).

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
| 2026-11-12 | Build the frame table (1.2) from `get_replay_frames_data` at native ~30 Hz, not the resampled ndarray | The ndarray defaults to 10 fps, copies the next raw frame instead of interpolating, and fills unrecorded goal-replay gaps with fake rows; frames_data keeps real timestamps and shares frame indexes with all events |
| 2026-11-12 | Frame table = a per-frame index (`frames`) plus a long-format `entities` table keyed by `"<Platform>:<id>"`, joined on `frame` | Clock/state live in one place; entity rows stay lean; platform ids are stable where display names aren't |
| 2026-11-12 | Demolished cars get NaN position/velocity/rotation/boost; `is_present` marks Empty frames outside demos | The recorded state is frozen at the demo spot with fake boost drain; detectors must not treat a dead or absent car as on the field |
| 2026-11-12 | Overtime from the replicated `GameEvent_Soccar_TA:bOverTime` attribute, falling back to "first kickoff after 0:00 with a tied score" | The header has no OT key; the network attribute is the game's own flag. Unverified until we have an OT replay (1.6) |
| 2026-11-12 | Events table is one wide frame-sorted table, nullable per-type columns, same `frame` index as `frames` | One place for detectors to look; parquet-friendly for 1.4; joins to the frame table without remapping |
| 2026-11-12 | Touch rule: stats `touch` stream as the base, physics (ball Δv + nearest car) for timing, contest and confidence; `touch_events` not used | `touch_events` is a team-change signal that misses goal shots. Stats and physics agree on 82-96% (same player, ±3 f), and both get 10/10 scorers and 29/30 saves/shots. Physics alone can't settle 50/50s by distance, so contests are flagged instead of guessed |
| 2026-11-12 | Ball-surface exclusion only gates physics-only touches | Applied to all physics candidates it removed 86-96% real (stats-confirmed) ground/wall hits |
| 2026-11-12 | 50/50s get one row per player with `contest_type` (opponent / teammate); the winner is decided in 2.5 from where the ball goes | Wrong blame is worse than no blame; the touch frame alone can't say who won. Teammate contests feed the double-commit detector |
| 2026-11-12 | Touch confidence: high = sources agree, uncontested; medium = one source or opponent 50/50 in range; low = teammate contest or source conflict | Lets detectors skip or soften blame on uncertain touches (Ari) |
| 2026-11-12 | Parquet cache: one file per table + `meta.json` in `data/<replay_id>/`; `load_replay` caches by default | Parsing takes 2-3 s per replay, a cached read ~30 ms. Parquet keeps pandas nullable/categorical dtypes exactly; meta.json written last marks a complete cache |
| 2026-11-12 | Cache valid only if replay sha256, `CACHE_VERSION`, parse-code hash (all `.py` in `rldr/parse`) and subtr-actor version all match | A stale cache silently breaks detectors. The code hash catches logic changes nobody remembered to version (Ari); `CACHE_VERSION` stays for intentional schema bumps |
| 2026-11-12 | Git history rewritten once to strip a Claude co-author trailer (force-pushed `main`); `.claude/settings.json` committed to disable it | Commits are authored by Ari only (CLAUDE.md rule 8) |
