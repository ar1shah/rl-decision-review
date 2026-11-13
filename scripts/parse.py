"""Parse replays into the parquet cache (ROADMAP 1.4) and report cache status and timings.

    uv run python scripts/parse.py                              # every replay in replays/
    uv run python scripts/parse.py replays/<file>.replay        # one or more replays
    uv run python scripts/parse.py replays/<file>.replay --force

For each replay: whether the cache was valid (or why not), the cold parse and write time when
it had to parse, the cached load time (`load_replay` on a hit, incl. hashing the replay), and
the size of each cached table. `--force` re-parses and rewrites even when the cache is valid.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rldr.parse import cache
from rldr.parse.frames import load_replay

REPLAY_DIR = Path(__file__).resolve().parent.parent / "replays"


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def process(path: Path, root: Path, force: bool) -> None:
    digest = cache.replay_hash(path)
    status = cache.cache_status(path.stem, digest, root)
    line = f"{path.stem}  cache: {status}"
    if force or status != cache.HIT:
        t0 = time.perf_counter()
        parsed = load_replay(path, use_cache=False)
        t1 = time.perf_counter()
        cache.write_cache(parsed, digest, root)
        t2 = time.perf_counter()
        line += f"{' (forced)' if force and status == cache.HIT else ''}"
        line += f" -> parsed {t1 - t0:.2f} s, wrote {t2 - t1:.2f} s"
    t0 = time.perf_counter()
    load_replay(path, cache_root=root)
    line += f" | cached load {time.perf_counter() - t0:.3f} s"
    if cache.cache_status(path.stem, digest, root) != cache.HIT:
        raise RuntimeError(f"{path.name}: cache still invalid after writing")
    print(line)

    folder = cache.cache_dir(path.stem, root)
    files = [folder / f"{t}.parquet" for t in cache.TABLES] + [folder / cache.META_FILE]
    sizes = ", ".join(f"{f.name} {size_mb(f):.2f}" for f in files)
    total = sum(size_mb(f) for f in files)
    print(f"    {folder}  {total:.2f} MB (replay {size_mb(path):.2f} MB): {sizes}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("replays", nargs="*", type=Path, help="default: every replays/*.replay")
    parser.add_argument("--force", action="store_true", help="re-parse even if cached")
    parser.add_argument("--cache-root", type=Path, default=cache.DEFAULT_CACHE_ROOT)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    paths = args.replays or sorted(REPLAY_DIR.glob("*.replay"))
    if not paths:
        sys.exit(f"no replays given and none in {REPLAY_DIR}")
    for path in paths:
        process(path, args.cache_root, args.force)


if __name__ == "__main__":
    main()
