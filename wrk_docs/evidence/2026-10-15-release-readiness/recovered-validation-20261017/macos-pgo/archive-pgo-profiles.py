#!/usr/bin/env python3
"""Copy stable LLVM raw profiles into durable evidence while the PGO build runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import time
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--build-root", type=pathlib.Path, required=True)
    parser.add_argument("--archive-dir", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    args = parser.parse_args()

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    stable: dict[pathlib.Path, tuple[int, int, int]] = {}
    copied: dict[pathlib.Path, dict[str, object]] = {}
    merge_seen_utc: str | None = None
    watch_start_utc = utc_now()
    print(f"watch_start_utc={watch_start_utc}", flush=True)
    print(f"watched_pid={args.pid}", flush=True)

    def scan(final: bool = False) -> None:
        nonlocal merge_seen_utc
        for run_root in sorted((args.build_root / "runs").glob("*")):
            profile_dir = run_root / "profiles"
            for source in sorted(profile_dir.glob("*.profraw")):
                try:
                    stat = source.stat()
                except FileNotFoundError:
                    continue
                previous = stable.get(source)
                if previous is not None and previous[:2] == (stat.st_size, stat.st_mtime_ns):
                    stable_count = previous[2] + 1
                else:
                    stable_count = 1
                stable[source] = (stat.st_size, stat.st_mtime_ns, stable_count)
                if source in copied or (not final and stable_count < 2):
                    continue

                profdata = run_root / "encoder.profdata"
                before_merge = not profdata.exists()
                destination = args.archive_dir / run_root.name / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                source_after = source.stat()
                if (source_after.st_size, source_after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    raise RuntimeError(f"raw profile changed while being archived: {source}")
                source_hash = sha256(source)
                destination_hash = sha256(destination)
                if source_hash != destination_hash:
                    raise RuntimeError(f"archived raw profile hash mismatch: {source}")
                record = {
                    "source": str(source),
                    "archive": str(destination),
                    "size_bytes": stat.st_size,
                    "sha256": source_hash,
                    "copied_utc": utc_now(),
                    "copied_before_profdata_merge": before_merge,
                }
                copied[source] = record
                print(f"archived={destination} bytes={stat.st_size} sha256={source_hash} before_merge={before_merge}", flush=True)

            if (run_root / "encoder.profdata").exists() and merge_seen_utc is None:
                merge_seen_utc = utc_now()
                print(f"profdata_merge_seen_utc={merge_seen_utc}", flush=True)

    while pid_exists(args.pid):
        scan()
        time.sleep(0.1)

    scan(final=True)
    records = sorted(copied.values(), key=lambda item: (str(item["source"])))
    manifest = {
        "watch_start_utc": watch_start_utc,
        "watch_terminal_utc": utc_now(),
        "watched_pid": args.pid,
        "build_root": str(args.build_root),
        "profile_count": len(records),
        "all_copies_match_source_hash": True,
        "copied_before_profdata_merge_count": sum(
            bool(record["copied_before_profdata_merge"]) for record in records
        ),
        "profdata_merge_seen_utc": merge_seen_utc,
        "profiles": records,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "watch_terminal_utc", "watched_pid", "profile_count",
        "copied_before_profdata_merge_count", "profdata_merge_seen_utc",
    )}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
