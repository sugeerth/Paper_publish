#!/usr/bin/env python3
"""Mine the commit histories of the visualization toolkits studied in the paper.

Every number the paper reports about maintenance comes from here. The script
clones each repository blobless (history without file contents, which is two
orders of magnitude smaller and enough for authorship analysis), walks
``git log``, and writes ``data/repos.json`` plus ``data/ecosystem.json``.

Usage::

    python3 scripts/analyze_history.py                 # clone into .cache/ and analyse
    python3 scripts/analyze_history.py --cache DIR     # reuse existing clones
    python3 scripts/analyze_history.py --no-fetch      # analyse only what is cached

Methodology notes, stated here because they determine the results:

* Merge commits are excluded (``--no-merges``). They are bookkeeping, and
  counting them would inflate whoever happens to press the merge button.
* Author names are lower-cased and passed through ``ALIASES`` to merge the
  handle/real-name pairs that otherwise split one person across two entries.
  This is the one manual step in the pipeline; the table is small and explicit
  so it can be audited.
* Bot commits are classified by author name (``is_bot``) and reported
  separately rather than dropped, because their *share* is a finding.
* "Last 12 months" is measured against ``WINDOW_START``, pinned to a constant
  so the snapshot is reproducible rather than relative to when you run it.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Pinned so re-running reproduces the paper's snapshot exactly.
SNAPSHOT = "2026-08-08"
WINDOW_START = "2025-08-08"

# --- the study corpus ------------------------------------------------------
# group: "lineage"    -- the Protovis -> D3 -> Vega -> Vega-Lite -> Mosaic line
#        "satellite"  -- research prototypes published from the same lab
#        "comparison" -- other general-purpose visualization toolkits
CORPUS = {
    # name              (github slug,                  group,        research_origin)
    "protovis":         ("mbostock/protovis",          "lineage",    True),
    "d3":               ("d3/d3",                      "lineage",    True),
    "vega":             ("vega/vega",                  "lineage",    True),
    "vega-lite":        ("vega/vega-lite",             "lineage",    True),
    "mosaic":           ("uwdata/mosaic",              "lineage",    True),

    "lyra":             ("vega/lyra",                  "satellite",  True),
    "voyager":          ("vega/voyager",               "satellite",  True),
    "draco":            ("uwdata/draco",               "satellite",  True),
    "falcon":           ("uwdata/falcon",              "satellite",  True),
    "altair":           ("vega/altair",                "satellite",  True),
    "arquero":          ("uwdata/arquero",             "satellite",  True),
    "ipyvega":          ("vega/ipyvega",               "satellite",  True),
    "vega-editor":      ("vega/editor",                "satellite",  True),
    "vega-embed":       ("vega/vega-embed",            "satellite",  True),
    "vega-tooltip":     ("vega/vega-tooltip",          "satellite",  True),
    "vega-datasets":    ("vega/vega-datasets",         "satellite",  True),

    "matplotlib":       ("matplotlib/matplotlib",      "comparison", False),
    "bokeh":            ("bokeh/bokeh",                "comparison", False),
    "plotly-py":        ("plotly/plotly.py",           "comparison", False),
    "chartjs":          ("chartjs/Chart.js",           "comparison", False),
    "echarts":          ("apache/echarts",             "comparison", True),
    "deckgl":           ("visgl/deck.gl",              "comparison", True),
    "ggplot2":          ("tidyverse/ggplot2",          "comparison", False),
    "plot":             ("observablehq/plot",          "comparison", False),
    "gephi":            ("gephi/gephi",                "comparison", True),
}

# The d3-* core modules, analysed as one family: D3 was split into ~30 packages,
# so counting only d3/d3 would badly understate the work.
D3_MODULES = [
    "d3-array", "d3-force", "d3-geo", "d3-hierarchy", "d3-interpolate",
    "d3-scale", "d3-scale-chromatic", "d3-selection", "d3-shape",
    "d3-transition",
]
for _m in D3_MODULES:
    CORPUS[_m] = (f"d3/{_m}", "d3-family", True)

# Repositories aggregated into the "Vega ecosystem" figure.
VEGA_ECOSYSTEM = [
    "vega", "vega-lite", "altair", "mosaic", "lyra", "voyager", "draco",
    "falcon", "arquero", "ipyvega", "vega-editor", "vega-embed",
    "vega-tooltip", "vega-datasets",
]
D3_FAMILY = ["d3"] + D3_MODULES

# One person, two git identities. Merging these is the only manual judgement
# in the pipeline, so the table is kept explicit and auditable.
ALIASES = {
    'kanit "ham" wongsuphasawat': "kanit wongsuphasawat",
    "jheer": "jeffrey heer",
    "mbostock": "mike bostock",
    "michael bostock": "mike bostock",
    "domoritz": "dominik moritz",
    "jakevdp": "jake vanderplas",
    "arvindsatya1": "arvind satyanarayan",
    "hadley": "hadley wickham",
}

BOT_NAMES = {"renovate", "semantic-release-bot", "github-actions", "codecov"}


def is_bot(name: str) -> bool:
    n = name.lower()
    return (
        "bot]" in n
        or n.endswith("-bot")
        or "dependabot" in n
        or "greenkeeper" in n
        or n in BOT_NAMES
    )


def canonical(name: str) -> str:
    n = name.strip().lower()
    return ALIASES.get(n, n)


def clone(slug: str, dest: Path) -> bool:
    if (dest / ".git").is_dir():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git", "clone", "--filter=blob:none", "--quiet",
        f"https://github.com/{slug}.git", str(dest),
    ]
    return subprocess.call(cmd) == 0


def commits(path: Path) -> list[tuple[str, str]]:
    """(canonical author, ISO date) for every non-merge commit, oldest last."""
    out = subprocess.check_output(
        ["git", "-C", str(path), "log", "--no-merges", "--pretty=format:%aN\t%aI"],
        text=True, errors="replace",
    )
    rows = []
    for line in out.split("\n"):
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        rows.append((canonical(parts[0]), parts[1]))
    return rows


def bus_factor(counter: collections.Counter, fraction: float) -> int:
    """How many authors it takes to account for `fraction` of the commits."""
    total = sum(counter.values())
    if not total:
        return 0
    seen = 0
    for i, (_, n) in enumerate(counter.most_common(), start=1):
        seen += n
        if seen >= total * fraction:
            return i
    return len(counter)


def analyse(name: str, path: Path) -> dict:
    rows = commits(path)
    human = collections.Counter()
    per_year_human: dict[int, int] = collections.Counter()
    per_year_bot: dict[int, int] = collections.Counter()
    recent_human = collections.Counter()
    last_human = ""
    first = ""

    for author, date in rows:
        year = int(date[:4])
        first = min(first, date) if first else date
        if is_bot(author):
            per_year_bot[year] += 1
            continue
        human[author] += 1
        per_year_human[year] += 1
        last_human = max(last_human, date)
        if date >= WINDOW_START:
            recent_human[author] += 1

    total = len(rows)
    total_human = sum(human.values())
    recent_total = sum(recent_human.values())
    recent_bots = sum(
        1 for a, d in rows if is_bot(a) and d >= WINDOW_START
    )
    slug, group, research = CORPUS[name]

    return {
        "name": name,
        "slug": slug,
        "group": group,
        "research_origin": research,
        "commits_total": total,
        "commits_human": total_human,
        "commits_bot": total - total_human,
        "bot_share": (total - total_human) / total if total else 0.0,
        "authors": len(human),
        "first_commit": first[:10],
        "last_human_commit": last_human[:10],
        "top_author": human.most_common(1)[0][0] if human else None,
        "top_author_share": (
            human.most_common(1)[0][1] / total_human if total_human else 0.0
        ),
        "bus_factor_50": bus_factor(human, 0.5),
        "bus_factor_80": bus_factor(human, 0.8),
        "human_commits_12mo": recent_total,
        "authors_12mo": len(recent_human),
        "bot_share_12mo": (
            recent_bots / (recent_bots + recent_total)
            if (recent_bots + recent_total) else 0.0
        ),
        "per_year_human": {str(y): n for y, n in sorted(per_year_human.items())},
        "per_year_bot": {str(y): n for y, n in sorted(per_year_bot.items())},
    }


def aggregate(names: list[str], cache: Path, label: str) -> dict:
    """Pool several repositories and measure concentration across the whole set."""
    human = collections.Counter()
    reach: dict[str, set[str]] = collections.defaultdict(set)
    recent = collections.Counter()
    used = []
    for name in names:
        path = cache / name
        if not (path / ".git").is_dir():
            continue
        used.append(name)
        for author, date in commits(path):
            if is_bot(author):
                continue
            human[author] += 1
            reach[author].add(name)
            if date >= WINDOW_START:
                recent[author] += 1

    total = sum(human.values())
    return {
        "label": label,
        "repositories": used,
        "human_commits": total,
        "humans": len(human),
        "bus_factor_50": bus_factor(human, 0.5),
        "bus_factor_80": bus_factor(human, 0.8),
        "human_commits_12mo": sum(recent.values()),
        "humans_12mo": len(recent),
        "top_authors": [
            {
                "name": a,
                "commits": n,
                "share": n / total if total else 0.0,
                "repositories": len(reach[a]),
            }
            for a, n in human.most_common(10)
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=ROOT / ".cache" / "repos")
    ap.add_argument("--no-fetch", action="store_true",
                    help="analyse only repositories already cached")
    args = ap.parse_args()

    rows = []
    for name, (slug, _group, _research) in CORPUS.items():
        path = args.cache / name
        if not (path / ".git").is_dir():
            if args.no_fetch:
                print(f"skip  {name} (not cached)", file=sys.stderr)
                continue
            if not clone(slug, path):
                print(f"fail  {name} ({slug})", file=sys.stderr)
                continue
        rows.append(analyse(name, path))
        print(f"  ok  {name}", file=sys.stderr)

    if not rows:
        print("no repositories analysed", file=sys.stderr)
        return 1

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "repos.json").write_text(json.dumps(
        {"snapshot": SNAPSHOT, "window_start": WINDOW_START,
         "repositories": rows}, indent=2) + "\n")

    ecosystems = [
        aggregate(VEGA_ECOSYSTEM, args.cache, "Vega ecosystem"),
        aggregate(D3_FAMILY, args.cache, "D3 family"),
    ]
    (DATA / "ecosystem.json").write_text(json.dumps(
        {"snapshot": SNAPSHOT, "window_start": WINDOW_START,
         "ecosystems": ecosystems}, indent=2) + "\n")

    print(f"wrote data/repos.json ({len(rows)} repositories) "
          f"and data/ecosystem.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
