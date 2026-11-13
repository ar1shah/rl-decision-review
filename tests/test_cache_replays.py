"""Parquet round-trip on the lab replays (ROADMAP 1.4): cached tables equal fresh ones exactly.

Skips when the (gitignored) replay files are absent.
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import REPLAY_DIR, REPLAY_IDS

from rldr.parse import cache
from rldr.parse.frames import ParsedReplay, load_replay


@pytest.fixture(scope="module")
def fresh_replays() -> dict[str, ParsedReplay]:
    return {}


@pytest.fixture(params=REPLAY_IDS, ids=lambda rid: rid[:4])
def fresh(request, fresh_replays) -> ParsedReplay:
    """A replay parsed from scratch (no cache), once per module."""
    rid = request.param
    path = REPLAY_DIR / f"{rid}.replay"
    if not path.exists():
        pytest.skip(f"replay not available: {path.name}")
    if rid not in fresh_replays:
        fresh_replays[rid] = load_replay(path, use_cache=False)
    return fresh_replays[rid]


def assert_tables_identical(a: ParsedReplay, b: ParsedReplay) -> None:
    for name in cache.TABLES:
        left, right = getattr(a, name), getattr(b, name)
        assert left.dtypes.equals(right.dtypes), name  # Int8 / boolean / string / category
        for col in left.columns:
            if isinstance(left[col].dtype, pd.CategoricalDtype):
                assert list(left[col].cat.categories) == list(right[col].cat.categories), col
                assert left[col].cat.ordered == right[col].cat.ordered, col
        pd.testing.assert_frame_equal(
            left, right, check_exact=True, check_categorical=True, obj=name
        )
    assert a.meta == b.meta


def test_parquet_round_trip_is_exact(fresh, tmp_path):
    digest = "0" * 64
    cache.write_cache(fresh, digest, tmp_path)
    back = cache.read_cache(fresh.meta["replay_id"], digest, tmp_path)
    assert back is not None
    assert_tables_identical(fresh, back)


def test_fresh_tables_have_the_tricky_dtypes(fresh):
    """Guard that the round-trip test covers nullable and categorical columns at all."""
    dtypes = {str(t) for name in cache.TABLES for t in getattr(fresh, name).dtypes}
    assert {"Int8", "Int16", "Int64", "boolean", "string", "category"} <= dtypes


def test_load_replay_cached_matches_fresh(fresh, tmp_path):
    path = REPLAY_DIR / f"{fresh.meta['replay_id']}.replay"
    rid, digest = path.stem, cache.replay_hash(path)
    assert cache.cache_status(rid, digest, tmp_path) == cache.NO_CACHE
    load_replay(path, cache_root=tmp_path)  # miss: parses and writes
    assert cache.cache_status(rid, digest, tmp_path) == cache.HIT
    assert_tables_identical(fresh, load_replay(path, cache_root=tmp_path))
