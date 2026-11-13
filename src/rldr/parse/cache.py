"""Parquet cache for parsed replays (ROADMAP 1.4).

Layout, one folder per replay (`replay_id` = the .replay file's stem):

    data/<replay_id>/frames.parquet
                     entities.parquet
                     players.parquet
                     events.parquet
                     meta.json        {"cache": {...validity keys...}, "meta": ParsedReplay.meta}

`meta.json` is written last and deleted first on a rewrite, so it marks a complete cache. A cache
is only used when all of these match the current state:

    version              CACHE_VERSION (bump on intentional schema changes)
    code_sha256          hash of every .py file in rldr/parse (any parsing/touch logic change)
    subtr_actor_version  installed subtr-actor-py (parser upgrades can change output)
    replay_sha256        the .replay file's bytes

Anything else (missing file, unreadable JSON or parquet) is a miss, never an error.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from functools import lru_cache
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import pandas as pd

from rldr.parse.frames import ParsedReplay

# Bump whenever a table's columns, dtypes or meaning change on purpose. Code changes in
# rldr/parse invalidate the cache on their own (code hash); this is for explicit schema bumps.
CACHE_VERSION = 1

TABLES = ("frames", "entities", "players", "events")
META_FILE = "meta.json"
PARSE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_ROOT = PARSE_DIR.parents[2] / "data"  # src/rldr/parse -> repo root / data

HIT = "hit"
# Miss reasons, in the order cache_status checks them.
NO_CACHE = "no cache"
UNREADABLE = "unreadable"
VERSION_CHANGED = "version changed"
CODE_CHANGED = "code changed"
PARSER_CHANGED = "parser changed"
HASH_CHANGED = "hash changed"
MISSING_TABLE = "missing table"


def replay_hash(path: str | Path) -> str:
    """sha256 of the replay file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash(src_dir: str | Path = PARSE_DIR) -> str:
    """sha256 over every .py file under `src_dir`, sorted by relative path.

    Each file contributes its relative path and its bytes, so edits, additions, deletions and
    renames all change the hash. CRLF is read as LF: git's autocrlf rewrites line endings on
    checkout, which shouldn't count as a code change.
    """
    src_dir = Path(src_dir)
    h = hashlib.sha256()
    for rel in sorted(p.relative_to(src_dir).as_posix() for p in src_dir.rglob("*.py")):
        h.update(rel.encode() + b"\0")
        h.update((src_dir / rel).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


@lru_cache(maxsize=1)
def current_code_hash() -> str:
    """Code hash of the installed rldr/parse package (computed once per process)."""
    return code_hash(PARSE_DIR)


def parser_version() -> str:
    return package_version("subtr-actor-py")


def cache_dir(replay_id: str, root: str | Path = DEFAULT_CACHE_ROOT) -> Path:
    return Path(root) / replay_id


def _cache_keys(digest: str) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "code_sha256": current_code_hash(),
        "subtr_actor_version": parser_version(),
        "replay_sha256": digest,
    }


def _read_meta(folder: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("cache"), dict):
        return None
    if not isinstance(doc.get("meta"), dict):
        return None
    return doc


def cache_status(replay_id: str, digest: str, root: str | Path = DEFAULT_CACHE_ROOT) -> str:
    """`HIT`, or the first reason the cache for `replay_id` can't be used (see module doc)."""
    folder = cache_dir(replay_id, root)
    if not (folder / META_FILE).exists():
        return NO_CACHE
    doc = _read_meta(folder)
    if doc is None:
        return UNREADABLE
    stored, current = doc["cache"], _cache_keys(digest)
    for key, reason in (
        ("version", VERSION_CHANGED),
        ("code_sha256", CODE_CHANGED),
        ("subtr_actor_version", PARSER_CHANGED),
        ("replay_sha256", HASH_CHANGED),
    ):
        if stored.get(key) != current[key]:
            return reason
    if not all((folder / f"{name}.parquet").exists() for name in TABLES):
        return MISSING_TABLE
    return HIT


def read_cache_with_status(
    replay_id: str, digest: str, root: str | Path = DEFAULT_CACHE_ROOT
) -> tuple[ParsedReplay | None, str]:
    """(tables, HIT) when the cache is valid and readable, else (None, miss reason)."""
    status = cache_status(replay_id, digest, root)
    if status != HIT:
        return None, status
    folder = cache_dir(replay_id, root)
    doc = _read_meta(folder)
    if doc is None:
        return None, UNREADABLE
    try:
        tables = {name: pd.read_parquet(folder / f"{name}.parquet") for name in TABLES}
    except Exception:  # corrupt or truncated parquet: re-parse rather than fail
        return None, UNREADABLE
    return ParsedReplay(meta=doc["meta"], **tables), HIT


def read_cache(
    replay_id: str, digest: str, root: str | Path = DEFAULT_CACHE_ROOT
) -> ParsedReplay | None:
    """Cached tables for `replay_id`, or None if there is no valid cache."""
    return read_cache_with_status(replay_id, digest, root)[0]


def write_cache(parsed: ParsedReplay, digest: str, root: str | Path = DEFAULT_CACHE_ROOT) -> Path:
    """Write all tables plus meta.json to `root/<replay_id>/`; returns that folder.

    meta.json is removed first and written last (atomically), so an interrupted write leaves a
    folder that reads as a miss.
    """
    if parsed.events is None:
        raise ValueError("cannot cache a ParsedReplay without events (use load_replay)")
    folder = cache_dir(parsed.meta["replay_id"], root)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / META_FILE).unlink(missing_ok=True)
    for name in TABLES:
        getattr(parsed, name).to_parquet(folder / f"{name}.parquet", index=False)
    keys = _cache_keys(digest) | {"written_at": datetime.now(UTC).isoformat(timespec="seconds")}
    tmp = folder / (META_FILE + ".tmp")
    tmp.write_text(json.dumps({"cache": keys, "meta": parsed.meta}, indent=2), encoding="utf-8")
    os.replace(tmp, folder / META_FILE)
    return folder
