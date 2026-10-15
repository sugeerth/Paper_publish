#!/usr/bin/env python3
"""Measure the growth of the Vega-Lite grammar across its tagged releases.

Vega-Lite publishes a machine-readable JSON Schema with every release, which
makes it one of the few visualization grammars whose expressiveness can be
measured directly rather than argued about. For each release tag this script
reads ``build/vega-lite-schema.json`` out of the git object store and records
its size and the number of type definitions it declares.

Usage::

    python3 scripts/schema_growth.py --repo .cache/vega-lite

The clone must have blobs available (a plain ``git clone``, or a blobless
clone with network access so missing objects can be fetched on demand).
Releases before v2.0.0 did not ship the built schema at that path and are
skipped; the script reports how many.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = "build/vega-lite-schema.json"
# Release tags only: vX.Y.Z or X.Y.Z, no prereleases.
TAG_RE = re.compile(r"^v?\d+\.\d+\.\d+$")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, errors="replace"
    )


def tag_date(repo: Path, tag: str) -> str:
    return git(repo, "log", "-1", "--format=%aI", tag).strip()[:10]


def schema_at(repo: Path, tag: str) -> tuple[int, int] | None:
    """(bytes, number of definitions) for the schema at `tag`, or None."""
    try:
        blob = git(repo, "show", f"{tag}:{SCHEMA_PATH}")
    except subprocess.CalledProcessError:
        return None
    try:
        doc = json.loads(blob)
    except json.JSONDecodeError:
        return None
    defs = doc.get("definitions") or doc.get("$defs") or {}
    return len(blob.encode()), len(defs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=ROOT / ".cache" / "repos" / "vega-lite")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "schema.json")
    args = ap.parse_args()

    if not (args.repo / ".git").is_dir():
        sys.exit(f"no git repository at {args.repo}")

    tags = [t for t in git(args.repo, "tag").split() if TAG_RE.match(t)]

    def key(t: str) -> tuple[int, ...]:
        return tuple(int(p) for p in t.lstrip("v").split("."))

    rows, skipped = [], 0
    for tag in sorted(tags, key=key):
        result = schema_at(args.repo, tag)
        if result is None:
            skipped += 1
            continue
        size, ndefs = result
        rows.append({
            "tag": tag,
            "date": tag_date(args.repo, tag),
            "bytes": size,
            "definitions": ndefs,
        })
        print(f"  {tag:10s} {rows[-1]['date']}  {size:>9,}B  {ndefs:>4d} defs",
              file=sys.stderr)

    if not rows:
        sys.exit("no schemas read; does the clone have blobs?")

    rows.sort(key=lambda r: r["date"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"path": SCHEMA_PATH, "releases": rows, "tags_without_schema": skipped},
        indent=2) + "\n")
    print(f"wrote {args.out} ({len(rows)} releases, {skipped} tags without a "
          f"built schema)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
