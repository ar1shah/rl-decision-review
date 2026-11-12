# Replay Notes

Ground-truth notes for each replay. Clock = in-game clock when it happened (use `+0:30` for overtime).
`.replay` files live in this folder but are gitignored. This file is committed.

---

## Template

### `<filename>.replay`
- **Mode:** 2s / 3s
- **Type:** ranked / casual / lab
- **My car / name:**
- **Teammate(s):**
- **Final score:**

| Clock | What happened | Whose fault (if a mistake) |
|---|---|---|
| 2:10 | I scored off a backboard read | n/a |
| 0:45 | Teammate and I double committed at mid, they scored | teammate |

---

## Lab replay plan (private 2v2, for Phase 3)

Do these on purpose and write down the clock time for each:

| # | Play | Who's at fault |
|---|---|---|
| 1-3 | Double commit (both arrive at the ball together) | Alternate who should have peeled |
| 4-5 | Last-man challenge (leave net, lose it) | Me |
| 6-7 | Boost detour (last man takes far corner 100 while ball comes toward our net) | Me |
| 8-9 | Rotation cut (turn back in front of teammate who's going) | Me |
| 10+ | Clean plays (good rotation, correct forced challenge, good 50) | Nobody |

---

## Replays

> **Name note:** Ari shows as `ari ツ` in replay data and `rawun` in-game. Same player (team 0 / Blue in both replays).

### `2C980F9741798CF5E4D1A0BFF902C275.replay`
- **Mode:** 2s
- **Type:** ranked, full game, no overtime
- **Me:** rawun / `ari ツ` (Blue)
- **Teammate:** CohenLovesJill
- **Opponents (Orange):** Degnis98, shotofsam
- **Final score:** 3-2 Blue

| Clock | What happened | Source | Whose fault (if a mistake) |
|---|---|---|---|
| 3:47 | CohenLovesJill save | Ari | n/a |
| 3:06 | Degnis98 scores (0-1) | Ari + parser | unknown |
| 1:58 | Orange save | Ari | n/a |
| 1:35 | Degnis98 scores (0-2) | parser only | unknown |
| 0:54 | CohenLovesJill scores (1-2) | Ari + parser | n/a |
| 0:38 | rawun scores (2-2) | Ari + parser | n/a |
| 0:22 | rawun save | Ari | n/a |
| 0:02 | CohenLovesJill scores (3-2) | Ari + parser | n/a |

### `0AACF51B43A578E00DD400BD009DF81E.replay`
- **Mode:** 2s
- **Type:** ranked, early forfeit by Orange at 1:38
- **Me:** rawun / `ari ツ` (Blue)
- **Teammate:** iDrxp6442
- **Opponents (Orange):** NickyFuego1406, Rubeen 福
- **Final score:** 4-1 Blue (forfeit)

| Clock | What happened | Source | Whose fault (if a mistake) |
|---|---|---|---|
| 4:54 | NickyFuego1406 scores (0-1) | Ari + parser | unknown |
| 4:37 | iDrxp6442 scores, assist rawun (1-1) | Ari + parser | n/a |
| 3:43 | iDrxp6442 scores (2-1) | Ari + parser | n/a |
| 2:37 | Orange save | Ari | n/a |
| 1:49 | iDrxp6442 scores (3-1) | Ari + parser | n/a |
| 1:38 | rawun scores (4-1) | Ari + parser | n/a |
| 1:38 | Orange forfeits | Ari | n/a |

**Validation:** every goal clock from the parser (mapped by timestamp) matches Ari's in-game notes. Saves (3:47, 1:58, 0:22, 2:37) are good test points for touch detection in 1.3.
