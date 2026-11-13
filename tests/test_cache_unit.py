"""Parquet cache (ROADMAP 1.4) on synthetic tables: hits, every miss reason, write order."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rldr.parse import cache, frames
from rldr.parse.frames import ParsedReplay, load_replay

REPLAY_ID = "FAKE0001"


def fake_parsed() -> ParsedReplay:
    """Tiny tables covering every dtype the real tables use."""
    n = 4
    frame_tbl = pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.int64),
            "time": np.linspace(0.0, 0.1, n),
            "clock": np.array([300, 300, 299, 299], dtype=np.int64),
            "game_state": pd.Categorical(["Countdown", "Active", "Active", "PostGoalScored"]),
            "is_live": np.array([False, True, True, False]),
        }
    )
    entities = pd.DataFrame(
        {
            "frame": np.repeat(np.arange(n, dtype=np.int64), 2),
            "entity": pd.Categorical(["ball", "Steam:1"] * n, categories=["ball", "Steam:1", "x"]),
            "team": pd.array([pd.NA, 0] * n, dtype="Int8"),
            "x": [0.0, np.nan] * n,
            "boost_active": pd.array([pd.NA, True, pd.NA, False] * 2, dtype="boolean"),
        }
    )
    players = pd.DataFrame(
        {"player_key": ["Steam:1"], "name": ["ari ツ"], "team": np.array([0], dtype=np.int8)}
    )
    events = pd.DataFrame(
        {
            "frame": np.array([1, 2], dtype=np.int64),
            "type": pd.Categorical(["kickoff", "touch"], categories=["kickoff", "touch", "goal"]),
            "player_key": pd.array([pd.NA, "Steam:1"], dtype="string"),
            "touch_frame": pd.array([pd.NA, pd.NA], dtype="Int64"),
            "contested": pd.array([pd.NA, False], dtype="boolean"),
        }
    )
    meta = {"replay_id": REPLAY_ID, "map": "cs_p", "forfeit": False, "overtime_source": None}
    return ParsedReplay(frame_tbl, entities, players, meta, events)


def assert_same(a: ParsedReplay, b: ParsedReplay) -> None:
    for name in cache.TABLES:
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name), check_exact=True)
    assert a.meta == b.meta


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A fake replay file, a cache root, and a counting stand-in for the real parser."""
    replay = tmp_path / f"{REPLAY_ID}.replay"
    replay.write_bytes(b"not really a replay")
    root = tmp_path / "data"
    calls: list[Path] = []

    def fake_parse(path: Path) -> ParsedReplay:
        calls.append(path)
        return fake_parsed()

    monkeypatch.setattr(frames, "_parse_replay", fake_parse)
    return replay, root, calls


def status(replay: Path, root: Path) -> str:
    return cache.cache_status(REPLAY_ID, cache.replay_hash(replay), root)


def test_second_load_is_a_cache_hit(setup):
    replay, root, calls = setup
    assert status(replay, root) == cache.NO_CACHE
    first = load_replay(replay, cache_root=root)
    assert status(replay, root) == cache.HIT
    second = load_replay(replay, cache_root=root)
    assert len(calls) == 1
    assert_same(first, second)
    assert sorted(p.name for p in (root / REPLAY_ID).iterdir()) == sorted(
        [f"{t}.parquet" for t in cache.TABLES] + [cache.META_FILE]
    )


def test_meta_json_records_cache_keys(setup):
    replay, root, _ = setup
    load_replay(replay, cache_root=root)
    doc = json.loads((root / REPLAY_ID / cache.META_FILE).read_text(encoding="utf-8"))
    assert doc["meta"] == fake_parsed().meta
    assert doc["cache"]["version"] == cache.CACHE_VERSION
    assert doc["cache"]["replay_sha256"] == cache.replay_hash(replay)
    assert doc["cache"]["code_sha256"] == cache.current_code_hash()
    assert doc["cache"]["subtr_actor_version"] == cache.parser_version()


def test_replay_bytes_changed_reparses(setup):
    replay, root, calls = setup
    load_replay(replay, cache_root=root)
    replay.write_bytes(b"a different replay")
    assert status(replay, root) == cache.HASH_CHANGED
    load_replay(replay, cache_root=root)
    assert len(calls) == 2
    assert status(replay, root) == cache.HIT


def test_cache_version_bump_reparses(setup, monkeypatch):
    replay, root, calls = setup
    load_replay(replay, cache_root=root)
    monkeypatch.setattr(cache, "CACHE_VERSION", cache.CACHE_VERSION + 1)
    assert status(replay, root) == cache.VERSION_CHANGED
    load_replay(replay, cache_root=root)
    assert len(calls) == 2


def test_code_change_reparses(setup, monkeypatch):
    replay, root, calls = setup
    load_replay(replay, cache_root=root)
    monkeypatch.setattr(cache, "current_code_hash", lambda: "0" * 64)
    assert status(replay, root) == cache.CODE_CHANGED
    load_replay(replay, cache_root=root)
    assert len(calls) == 2
    assert status(replay, root) == cache.HIT


def test_parser_upgrade_reparses(setup, monkeypatch):
    replay, root, calls = setup
    load_replay(replay, cache_root=root)
    monkeypatch.setattr(cache, "parser_version", lambda: "999.0.0")
    assert status(replay, root) == cache.PARSER_CHANGED
    load_replay(replay, cache_root=root)
    assert len(calls) == 2


def test_code_hash_tracks_py_files(tmp_path):
    src = tmp_path / "parse"
    (src / "sub").mkdir(parents=True)
    (src / "a.py").write_text("A = 1\n")
    (src / "sub" / "b.py").write_text("B = 2\n")
    base = cache.code_hash(src)
    assert cache.code_hash(src) == base

    (src / "notes.txt").write_text("not code")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00bytecode")
    assert cache.code_hash(src) == base  # only .py files count

    (src / "a.py").write_text("A = 2\n")
    assert cache.code_hash(src) != base  # edit
    (src / "a.py").write_text("A = 1\n")
    assert cache.code_hash(src) == base

    (src / "c.py").write_text("")
    assert cache.code_hash(src) != base  # new file, even empty
    (src / "c.py").unlink()
    (src / "sub" / "b.py").rename(src / "sub" / "b2.py")
    assert cache.code_hash(src) != base  # rename


def test_current_code_hash_covers_parse_package():
    assert cache.current_code_hash() == cache.code_hash(cache.PARSE_DIR)
    assert (cache.PARSE_DIR / "touches.py").exists()


def test_missing_table_reparses(setup):
    replay, root, calls = setup
    load_replay(replay, cache_root=root)
    (root / REPLAY_ID / "entities.parquet").unlink()
    assert status(replay, root) == cache.MISSING_TABLE
    load_replay(replay, cache_root=root)
    assert len(calls) == 2


def test_corrupt_meta_reparses(setup):
    replay, root, calls = setup
    load_replay(replay, cache_root=root)
    (root / REPLAY_ID / cache.META_FILE).write_text("{not json", encoding="utf-8")
    assert status(replay, root) == cache.UNREADABLE
    load_replay(replay, cache_root=root)
    assert len(calls) == 2


def test_corrupt_parquet_reparses(setup):
    replay, root, calls = setup
    load_replay(replay, cache_root=root)
    (root / REPLAY_ID / "events.parquet").write_bytes(b"garbage")
    digest = cache.replay_hash(replay)
    assert cache.read_cache_with_status(REPLAY_ID, digest, root) == (None, cache.UNREADABLE)
    parsed = load_replay(replay, cache_root=root)
    assert len(calls) == 2
    assert_same(parsed, fake_parsed())
    assert status(replay, root) == cache.HIT


def test_use_cache_false_never_touches_cache(setup):
    replay, root, calls = setup
    load_replay(replay, use_cache=False, cache_root=root)
    load_replay(replay, use_cache=False, cache_root=root)
    assert len(calls) == 2
    assert not root.exists()


def test_interrupted_write_leaves_no_valid_cache(setup, monkeypatch):
    replay, root, _ = setup
    load_replay(replay, cache_root=root)
    original = pd.DataFrame.to_parquet
    written = []

    def flaky(self, *args, **kwargs):
        if written:
            raise OSError("disk full")
        written.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", flaky)
    with pytest.raises(OSError):
        cache.write_cache(fake_parsed(), cache.replay_hash(replay), root)
    assert not (root / REPLAY_ID / cache.META_FILE).exists()
    assert status(replay, root) == cache.NO_CACHE


def test_write_cache_requires_events(tmp_path):
    parsed = fake_parsed()
    parsed.events = None
    with pytest.raises(ValueError):
        cache.write_cache(parsed, "0" * 64, tmp_path)


def test_default_root_is_repo_data_dir():
    assert cache.DEFAULT_CACHE_ROOT == Path(__file__).resolve().parent.parent / "data"
