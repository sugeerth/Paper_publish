# Research-Origin Visualization Software: a short-paper study

This repository contains a short paper, its LaTeX source, the data snapshot it
reports, and the scripts that produced both.

## Building the PDF

```sh
make            # regenerate figures, then build paper/main.pdf
make paper      # build the PDF only, using the committed data snapshot
make clean      # remove LaTeX intermediates
```

The build needs a TeX Live installation with `latexmk`, `pdflatex`, `bibtex`,
and the `titlesec`, `booktabs`, `microtype`, and `pgfplots` packages. On
Debian/Ubuntu:

```sh
apt-get install texlive-latex-base texlive-latex-recommended \
                texlive-latex-extra texlive-fonts-recommended latexmk
```

`paper/visstyle.sty` is a small self-contained approximation of the IEEE VGTC
two-column short-paper layout, written so the paper compiles on a stock TeX
Live install. Swap `\usepackage{visstyle}` for the official `vgtc.cls` at
camera-ready time.

## Refreshing the data

```sh
GITHUB_TOKEN=<token> python3 scripts/collect_metrics.py   # -> data/metrics.json
python3 scripts/make_figures.py                           # -> figures/*.tex
```

`data/metrics.json` is committed so the paper builds without network access.
The `collected_at` field records when the snapshot was taken; every figure in
the paper is derived from that one file.

## Layout

| Path | Contents |
| --- | --- |
| `paper/main.tex` | The paper |
| `paper/visstyle.sty` | Two-column VGTC-like style |
| `paper/refs.bib` | Bibliography |
| `scripts/collect_metrics.py` | Pulls repository metadata from the GitHub API |
| `scripts/make_figures.py` | Turns the snapshot into pgfplots figures |
| `data/metrics.json` | Committed snapshot of the study corpus |
| `figures/` | Generated figure sources |
