#!/usr/bin/env bash
# Usage: make_pdf.sh <book_dir> "<Title>" "<Subtitle>" <output.pdf>
set -euo pipefail
BOOK="$1"; TITLE="$2"; SUB="$3"; OUT="$4"
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
ls "$BOOK"/chapters/*.md | sort -V > "$TMP/list"
pandoc $(cat "$TMP/list") -f gfm-tex_math_dollars -t html5 -s --toc --toc-depth=2 \
  --metadata title="$TITLE" --metadata subtitle="$SUB" \
  --css "$HERE/style.css" --embed-resources -o "$TMP/book.html"
NODE_PATH="$(npm root -g)" node "$HERE/print.js" "$TMP/book.html" "$OUT" "$TITLE"
cp "$TMP/book.html" "${OUT%.pdf}.html"
echo "built $OUT ($(wc -l < "$TMP/list") chapters)"
