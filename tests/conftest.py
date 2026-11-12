"""Shared fixtures. Replay files are gitignored, so replay tests skip when they're absent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import subtr_actor as sa

from rldr.parse.frames import ParsedReplay, build_replay

REPLAY_DIR = Path(__file__).resolve().parent.parent / "replays"
REPLAY_IDS = ["2C980F9741798CF5E4D1A0BFF902C275", "0AACF51B43A578E00DD400BD009DF81E"]


@dataclass
class LoadedReplay:
    """A parsed replay plus the raw parser output, for cross-checking the tables."""

    parsed: ParsedReplay
    raw: dict[str, Any]
    fd: dict[str, Any]


@pytest.fixture(scope="session")
def load_lab_replay():
    cache: dict[str, LoadedReplay] = {}

    def load(replay_id: str) -> LoadedReplay:
        if replay_id not in cache:
            path = REPLAY_DIR / f"{replay_id}.replay"
            if not path.exists():
                pytest.skip(f"replay not available: {path.name}")
            raw = sa.parse_replay(path.read_bytes())
            fd = sa.get_replay_frames_data(str(path))
            cache[replay_id] = LoadedReplay(build_replay(replay_id, raw, fd), raw, fd)
        return cache[replay_id]

    return load


@pytest.fixture(params=REPLAY_IDS, ids=lambda rid: rid[:4])
def replay(request, load_lab_replay) -> LoadedReplay:
    return load_lab_replay(request.param)
