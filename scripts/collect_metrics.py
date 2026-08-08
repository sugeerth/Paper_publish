#!/usr/bin/env python3
"""Collect repository metrics for the visualization toolkits studied in the paper.

The paper reports a snapshot of GitHub metadata. This script is what produced
that snapshot, and re-running it regenerates ``data/metrics.json`` so the
numbers in the paper can be checked or refreshed.

Usage::

    GITHUB_TOKEN=ghp_... python3 scripts/collect_metrics.py

A token is optional but strongly recommended: unauthenticated requests are
limited to 60/hour, which is not enough for the contributor pagination probes.

If the network is unavailable the script leaves the existing snapshot in place
and exits non-zero rather than writing partial data.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://api.github.com"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "metrics.json"

# The study corpus: general-purpose, actively distributed visualization
# toolkits, split by whether the project originated in published
# visualization research.
CORPUS = [
    # slug,                      family,        research_origin
    ("d3/d3",                    "web",         True),
    ("vega/vega",                "web",         True),
    ("vega/vega-lite",           "web",         True),
    ("uwdata/mosaic",            "web",         True),
    ("observablehq/plot",        "web",         False),
    ("apache/echarts",           "web",         True),
    ("plotly/plotly.js",         "web",         False),
    ("chartjs/Chart.js",         "web",         False),
    ("visgl/deck.gl",            "web",         True),
    ("matplotlib/matplotlib",    "notebook",    False),
    ("vega/altair",              "notebook",    True),
    ("bokeh/bokeh",              "notebook",    False),
    ("holoviz/holoviews",        "notebook",    False),
    ("plotly/plotly.py",         "notebook",    False),
    ("mwaskom/seaborn",          "notebook",    False),
]


def _request(path: str, token: str | None) -> tuple[dict | list, dict]:
    """GET an API path, returning the decoded body and the response headers."""
    req = urllib.request.Request(f"{API}{path}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "vis-oss-study")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode()), dict(resp.headers)


def contributor_count(slug: str, token: str | None) -> int | None:
    """Total contributors, read off the Link header of a 1-per-page request.

    GitHub does not expose a contributor count directly. Asking for a single
    contributor per page makes the ``last`` page number equal to the count.
    """
    try:
        _, headers = _request(
            f"/repos/{slug}/contributors?per_page=1&anon=true", token
        )
    except urllib.error.HTTPError:
        return None
    link = headers.get("Link", "")
    for part in link.split(","):
        if 'rel="last"' in part:
            url = part.split(";")[0].strip().strip("<>")
            return int(url.rsplit("page=", 1)[1])
    return 1


def release_count(slug: str, token: str | None) -> int | None:
    try:
        _, headers = _request(f"/repos/{slug}/releases?per_page=1", token)
    except urllib.error.HTTPError:
        return None
    link = headers.get("Link", "")
    for part in link.split(","):
        if 'rel="last"' in part:
            url = part.split(";")[0].strip().strip("<>")
            return int(url.rsplit("page=", 1)[1])
    return 0


def collect(slug: str, family: str, research: bool, token: str | None) -> dict:
    repo, _ = _request(f"/repos/{slug}", token)
    assert isinstance(repo, dict)
    return {
        "slug": slug,
        "family": family,
        "research_origin": research,
        "stars": repo["stargazers_count"],
        "forks": repo["forks_count"],
        "open_issues": repo["open_issues_count"],
        "watchers": repo["subscribers_count"],
        "language": repo.get("language"),
        "license": (repo.get("license") or {}).get("spdx_id"),
        "created_at": repo["created_at"],
        "pushed_at": repo["pushed_at"],
        "archived": repo["archived"],
        "contributors": contributor_count(slug, token),
        "releases": release_count(slug, token),
    }


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("warning: no GITHUB_TOKEN set; may hit the 60/hour rate limit",
              file=sys.stderr)

    rows = []
    for slug, family, research in CORPUS:
        try:
            rows.append(collect(slug, family, research, token))
            print(f"  ok  {slug}", file=sys.stderr)
        except (urllib.error.URLError, OSError) as exc:
            print(f"fail  {slug}: {exc}", file=sys.stderr)
            return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "repositories": rows,
        },
        indent=2,
    ) + "\n")
    print(f"wrote {OUT} ({len(rows)} repositories)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
