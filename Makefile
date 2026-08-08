PAPER := paper/main
LATEXMK := latexmk
LATEXMKFLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all paper figures data clean distclean watch

all: figures paper

## Build the PDF (runs pdflatex/bibtex as many times as needed)
paper:
	$(LATEXMK) $(LATEXMKFLAGS) -outdir=paper $(PAPER).tex

## Regenerate the figures from the collected repository metrics
figures: data
	python3 scripts/make_figures.py

## Refresh the raw repository metrics from the GitHub API
data:
	python3 scripts/collect_metrics.py

watch:
	$(LATEXMK) $(LATEXMKFLAGS) -outdir=paper -pvc $(PAPER).tex

clean:
	$(LATEXMK) -c -outdir=paper $(PAPER).tex

distclean:
	$(LATEXMK) -C -outdir=paper $(PAPER).tex
