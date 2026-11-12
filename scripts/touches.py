"""Compare the three touch sources (ROADMAP 1.3) and print the evidence for the touch rule.

    uv run python scripts/touches.py [replays/<file>.replay ...]   # default: all replays

Sections per replay:
  1. source counts
  2. physics candidates: contact range and ball-surface exclusion (both variants)
  3. agreement matrix (±0/±1/±3 frames, same team / same player)
  4. disagreement buckets (why sources miss each other)
  5. touch_events vs "hitting team changed"
  6. every Save / Shot stat event: which sources have the credited player's touch before it
  7. every goal: scoring team's last touch per source
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import subtr_actor as sa

from rldr.parse.frames import build_replay, format_clock, player_key
from rldr.parse.touches import (
    CONTACT_RANGE,
    physics_touches,
    stats_touches,
    team_touches,
)

REPLAY_DIR = Path(__file__).resolve().parent.parent / "replays"
TOLS = (0, 1, 3)
STAT_WINDOWS = (10, 20, 40, 90)  # frames before a Save/Shot credit to look for the touch


def match(a: pd.DataFrame, b: pd.DataFrame, tol: int, on: str | None) -> np.ndarray:
    """For each row of `a`: is there a row of `b` within ±tol frames (and same `on` value)?"""
    if not len(a) or not len(b):
        return np.zeros(len(a), dtype=bool)
    close = np.abs(a["frame"].to_numpy()[:, None] - b["frame"].to_numpy()[None, :]) <= tol
    if on is not None:
        close &= a[on].to_numpy()[:, None] == b[on].to_numpy()[None, :]
    return close.any(axis=1)


def nearest_offset(a: pd.DataFrame, b: pd.DataFrame, on: str | None) -> np.ndarray:
    """Signed frame offset (b - a) to the nearest matching row of `b`, or a large number."""
    out = np.full(len(a), 10**6)
    for i, (f, v) in enumerate(zip(a["frame"], a[on] if on else [None] * len(a), strict=True)):
        bb = b if on is None else b[b[on] == v]
        if len(bb):
            d = bb["frame"].to_numpy() - f
            out[i] = d[np.argmin(np.abs(d))]
    return out


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def section(title: str) -> None:
    print(f"\n--- {title}")


def report(path: Path) -> None:
    raw = sa.parse_replay(path.read_bytes())
    fd = sa.get_replay_frames_data(str(path))
    stats_ev = sa.get_stats_events(str(path))["events"]
    parsed = build_replay(path.stem, raw, fd)
    names = dict(zip(parsed.players["player_key"], parsed.players["name"], strict=True))
    clock = parsed.frames["clock"].to_numpy()

    def clk(f: int) -> str:
        return format_clock(clock[f], bool(parsed.frames.at[f, "is_overtime"]))

    def nm(k) -> str:
        return names.get(k, str(k))

    st = stats_touches(stats_ev)
    te = team_touches(fd["touch_events"])
    ph_all = physics_touches(parsed)

    print(f"\n==================== {path.stem} ({len(parsed.frames)} frames)")

    # ------------------------------------------------------------------ 1. counts
    section("1. sources")
    print(f"stats touch stream : {len(st)}  (players credited: {st['player_key'].nunique()})")
    print(f"touch_events       : {len(te)}  (team only)")
    print(f"physics candidates : {len(ph_all)}  (|dv| > threshold, live, ball awake)")

    # ------------------------------------------------------------------ 2. physics filters
    section("2. physics candidates: contact range and surface exclusion")
    bins = [0, 150, 200, 250, 300, 400, 600, np.inf]
    hist = np.histogram(ph_all["dist"], bins=bins)[0]
    labels = [f"<{b}" for b in bins[1:-1]] + [">=600"]
    print("nearest car distance:", dict(zip(labels, hist.tolist(), strict=True)))
    in_range = ph_all[ph_all["dist"] <= CONTACT_RANGE].reset_index(drop=True)
    print(f"within contact range ({CONTACT_RANGE:.0f} uu): {len(in_range)}")
    in_range["stats_same_player"] = match(in_range, st, 3, "player_key")

    near_surface = in_range["surfaces"].map(len) > 0
    by_surface = Counter(s for ss in in_range.loc[near_surface, "surfaces"] for s in ss)
    print(
        f"near any surface (radius + margin): {near_surface.sum()}  by surface: {dict(by_surface)}"
    )
    for label, removed in (
        ("A. proximity only", near_surface),
        ("B. proximity + bounce", in_range["bounce"]),
    ):
        r = in_range[removed]
        lost = int(r["stats_same_player"].sum())
        print(
            f"  {label:<22} removes {len(r):>3}; of those, stats stream has the same player's "
            f"touch (±3f) on {pct(lost, len(r))}"
        )
        surf = Counter(s for ss in r["surfaces"] for s in ss)
        print(f"  {'':<22} by surface: {dict(surf)}")
    ph = in_range[~in_range["bounce"]].reset_index(drop=True)
    print(f"physics source used below = in range, minus variant B bounces: {len(ph)}")

    # ------------------------------------------------------------------ 3. agreement
    section("3. agreement: share of ROW touches that COLUMN also has within ±tol frames")
    srcs = {"stats": st, "touch_events": te, "physics": ph}
    for tol in TOLS:
        print(f" tol ±{tol}f, same team:")
        for a_name, a in srcs.items():
            cells = [
                f"{b_name}={pct(int(match(a, b, tol, 'team').sum()), len(a))}"
                for b_name, b in srcs.items()
                if b is not a
            ]
            print(f"   {a_name:<13} " + "   ".join(cells))
    for tol in TOLS:
        sp = int(match(st, ph, tol, "player_key").sum())
        ps = int(match(ph, st, tol, "player_key").sum())
        print(
            f" tol ±{tol}f, same PLAYER: stats in physics {pct(sp, len(st))}; "
            f"physics in stats {pct(ps, len(ph))}"
        )

    # ------------------------------------------------------------------ 4. buckets
    section("4. disagreement buckets")
    # stats touches with no same-player physics touch within ±3
    pair_frames = set(st.loc[st.duplicated("frame", keep=False), "frame"])
    st_b = Counter()
    st_examples: dict[str, list[str]] = {}
    for _, s in st[~match(st, ph, 3, "player_key")].iterrows():
        f = s["frame"]
        prev_same = st[(st["player_key"] == s["player_key"]) & st["frame"].between(f - 2, f - 1)]
        cand = ph_all[ph_all["frame"].between(f - 3, f + 3)]
        off = nearest_offset(pd.DataFrame([s]), ph, "player_key")[0]
        if f in pair_frames:
            b = "same-frame pair (stats credits 2 players)"
        elif len(prev_same):
            b = "near-duplicate (same player 1-2 frames earlier)"
        elif s["ball_speed_change"] < 300:
            b = "soft touch (ball speed change < 300)"
        elif len(cand) and (cand["dist"] > CONTACT_RANGE).all():
            b = "physics saw it, nearest car outside contact range"
        elif len(cand) and cand["bounce"].any():
            b = "physics saw it, labelled a surface bounce"
        elif len(cand):
            b = "physics credits another player"
        elif abs(off) <= 10:
            b = "physics same player, 4-10 frames off"
        else:
            b = "unexplained"
        st_b[b] += 1
        st_examples.setdefault(b, []).append(f"{clk(f)}@{f} {nm(s['player_key'])}")
    print(f" stats touches with no same-player physics touch (±3f): {sum(st_b.values())}")
    for b, n in st_b.most_common():
        print(f"   {n:>3}  {b}   e.g. {', '.join(st_examples[b][:3])}")

    ph_b = Counter()
    ph_examples: dict[str, list[str]] = {}
    for _, p in ph[~match(ph, st, 3, "player_key")].iterrows():
        f = p["frame"]
        near = st[st["frame"].between(f - 3, f + 3)]
        if len(near):
            other = near.iloc[0]["player_key"]
            ok = p["second_key"] == other and p["second_dist"] <= CONTACT_RANGE
            b = (
                "stats credits the 2nd-nearest car (also in range)"
                if ok
                else "stats credits another player"
            )
        elif p["surfaces"]:
            b = "near a surface, no bounce signature (car + surface?)"
        elif abs(nearest_offset(pd.DataFrame([p]), st, "player_key")[0]) <= 10:
            b = "stats same player, 4-10 frames off"
        else:
            b = "no stats touch nearby"
        ph_b[b] += 1
        ph_examples.setdefault(b, []).append(f"{clk(f)}@{f} {nm(p['player_key'])} dv={p['dv']:.0f}")
    print(f" physics touches with no same-player stats touch (±3f): {sum(ph_b.values())}")
    for b, n in ph_b.most_common():
        print(f"   {n:>3}  {b}   e.g. {', '.join(ph_examples[b][:3])}")

    # touch_events vs stats
    te_off = nearest_offset(te, st, "team")
    miss = np.abs(te_off) > 3
    near_off = Counter(int(o) for o in te_off[miss] if abs(o) <= 15)
    far = int((np.abs(te_off) > 15).sum())
    print(
        f" touch_events with no same-team stats touch (±3f): {int(miss.sum())}; "
        f"offsets 4-15f: {sum(near_off.values())}, none within 15f: {far}"
    )

    # ------------------------------------------------------------------ 5. team change
    section("5. is touch_events a 'hitting team changed' signal?")
    st_live = st.drop_duplicates("frame")  # one row per frame for this question
    change = np.r_[True, st_live["team"].to_numpy()[1:] != st_live["team"].to_numpy()[:-1]]
    has_te = match(st_live, te, 3, "team")
    seen_change = pct(int((has_te & change).sum()), int(change.sum()))
    seen_same = pct(int((has_te & ~change).sum()), int((~change).sum()))
    print(f" stats touches that change the hitting team: {seen_change} in touch_events")
    print(f" stats touches by the same team as before  : {seen_same} in touch_events")
    te_change = np.r_[True, te["team"].to_numpy()[1:] != te["team"].to_numpy()[:-1]]
    switch = pct(int(te_change[1:].sum()), len(te) - 1)
    print(f" consecutive touch_events that switch team : {switch}")

    # ------------------------------------------------------------------ 6. saves / shots
    section("6. Save / Shot credits: touch by the credited player before the stat frame")
    pse = [e for e in fd["player_stat_events"] if e["kind"] in ("Save", "Shot")]
    lags: dict[str, list[int]] = {"stats": [], "physics": [], "touch_events": []}
    hits = {k: Counter() for k in lags}
    last_ok = Counter()
    last_miss: dict[str, list[str]] = {"stats": [], "physics": []}
    head = f" {'kind':<5} {'clock':>6} {'frame':>6} {'player':<16}"
    print(f"{head} lag(frames) stats / physics / touch_events(team)")
    for ev in pse:
        f, k = int(ev["frame"]), player_key(ev["player"])
        team = 0 if ev["is_team_0"] else 1
        for name, src in (("stats", st), ("physics", ph)):
            prior = src[(src["team"] == team) & (src["frame"] <= f)]
            if len(prior) and prior.iloc[-1]["player_key"] == k:
                last_ok[name] += 1
            else:
                who = nm(prior.iloc[-1]["player_key"]) if len(prior) else "nobody"
                last_miss[name].append(f"{ev['kind']} {clk(f)} {nm(k)} (last: {who})")
    for ev in pse:
        f, k = int(ev["frame"]), player_key(ev["player"])
        team = 0 if ev["is_team_0"] else 1
        row = []
        for name, src, col, val in (
            ("stats", st, "player_key", k),
            ("physics", ph, "player_key", k),
            ("touch_events", te, "team", team),
        ):
            prior = src[(src[col] == val) & src["frame"].between(f - max(STAT_WINDOWS), f)]
            if len(prior):
                lag = f - int(prior["frame"].max())
                lags[name].append(lag)
                for w in STAT_WINDOWS:
                    hits[name][w] += lag <= w
                row.append(f"{lag:>3}")
            else:
                row.append("  -")
        print(f" {ev['kind']:<5} {clk(f):>6} {f:>6} {nm(k):<16} {' / '.join(row)}")
    for name in lags:
        cells = ", ".join(f"<={w}f {pct(hits[name][w], len(pse))}" for w in STAT_WINDOWS)
        print(f" {name:<13} found: {cells}")
    for name in last_miss:
        misses = "; ".join(last_miss[name]) or "none"
        print(
            f" {name}: credited player is their team's last toucher before the credit on "
            f"{pct(last_ok[name], len(pse))}; misses: {misses}"
        )

    # ------------------------------------------------------------------ 7. goals
    section("7. goals: scoring team's last touch before the goal frame")
    for g in fd["goal_events"]:
        f, scorer = int(g["frame"]), player_key(g["player"])
        team = 0 if g["scoring_team_is_team_0"] else 1
        cells = []
        for name, src in (("stats", st), ("physics", ph), ("touch_events", te)):
            prior = src[(src["team"] == team) & (src["frame"] <= f)]
            if not len(prior):
                cells.append(f"{name}: none")
                continue
            last = prior.iloc[-1]
            who = nm(last["player_key"]) if "player_key" in src else f"team {team}"
            ok = (
                ""
                if "player_key" not in src
                else (" OK" if last["player_key"] == scorer else " WRONG")
            )
            cells.append(f"{name}: {who} {f - int(last['frame'])}f before{ok}")
        print(f" {clk(f)} {nm(scorer):<16} " + " | ".join(cells))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("replays", nargs="*", type=Path)
    args = ap.parse_args()
    paths = args.replays or sorted(REPLAY_DIR.glob("*.replay"))
    pd.set_option("display.width", 200)
    for p in paths:
        report(p)


if __name__ == "__main__":
    main()
