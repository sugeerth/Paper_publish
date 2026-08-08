# Who Maintains the Grammar?

Source, data, and build for a short paper measuring who actually maintains the
visualization community's most-used research software — the
Protovis → D3 → Vega → Vega-Lite → Mosaic lineage — against ten comparison
toolkits.

**Headline findings** (snapshot 8 August 2026): three people account for half of
26,902 human commits across 14 lineage repositories; Vega-Lite's human
contribution fell 97% from its 2015 peak to 55 commits in the last twelve
months; bots write 58% of its recent commits and over 80% in three sibling
packages; its JSON Schema has been byte-identical for six releases; and five of
the ten published systems have had no human commit in a year, including a
best-paper winner and a test-of-time winner.

## Building the PDF

```sh
make          # regenerate figures and macros, then build paper/main.pdf
make paper    # build the PDF only, from the committed data
make clean    # remove LaTeX intermediates
```

Requires TeX Live with `latexmk`, `pdflatex`, `bibtex`, and the `titlesec`,
`booktabs`, `microtype`, and `pgfplots` packages. On Debian/Ubuntu:

```sh
apt-get install texlive-latex-base texlive-latex-recommended \
                texlive-latex-extra texlive-fonts-recommended latexmk
```

`paper/visstyle.sty` is a small self-contained approximation of the IEEE VGTC
two-column short-paper layout, written so the paper compiles on a stock TeX
Live install — neither `vgtc.cls` nor `IEEEtran.cls` is in the base packages,
and a paper a reviewer cannot compile is not reproducible. Swap in the official
`vgtc.cls` at camera-ready time.

## Reproducing the data

```sh
make history   # clone 35 repos blobless, mine git history -> data/repos.json,
               #   data/ecosystem.json   (needs network; ~200 MB of clones)
make schema    # read the Vega-Lite JSON Schema at each release tag
               #   -> data/schema.json
make figures   # data/*.json -> figures/*.tex and paper/numbers.tex
```

The snapshot date and the twelve-month window are pinned as constants in
`scripts/analyze_history.py`, so re-running reproduces the paper's numbers
rather than drifting with the calendar. Every figure quoted in the prose is a
macro in the generated `paper/numbers.tex`, so the text cannot disagree with
the data.

### Method notes

- Merge commits are excluded; they are bookkeeping and would credit whoever
  pressed the button.
- Author names are lower-cased and passed through a small explicit alias table
  merging handle/real-name pairs. This is the only manual judgement in the
  pipeline, and it biases concentration *downward* — it makes the central
  finding conservative.
- Bot commits are classified by author name and reported separately rather than
  dropped, because their share is itself a result.

`data/issues.json` is the one exception to reproducibility: issue state lives on
GitHub, not in the git history, so those two counts are a recorded API snapshot.
The paper says so in a footnote.

## Layout

| Path | Contents |
| --- | --- |
| `paper/main.tex` | The paper |
| `paper/visstyle.sty` | Two-column VGTC-like style |
| `paper/numbers.tex` | Generated — every number quoted in the prose |
| `paper/refs.bib` | Bibliography |
| `scripts/analyze_history.py` | Clones and mines the commit histories |
| `scripts/schema_growth.py` | Measures the Vega-Lite grammar across releases |
| `scripts/make_figures.py` | Data → pgfplots figures, table, and macros |
| `data/` | Committed snapshots, so the paper builds offline |
| `figures/` | Generated figure and table sources |
