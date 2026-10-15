LATEXMK := latexmk
LATEXMKFLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all paper figures history schema clean distclean watch

all: figures paper

## Build the PDF. Run from paper/ because latexmk's -outdir interacts badly
## with the relative \input paths used for the generated figures.
paper:
	cd paper && $(LATEXMK) $(LATEXMKFLAGS) main.tex

## Regenerate figures, the maintenance table, and paper/numbers.tex
figures:
	python3 scripts/make_figures.py

## Re-mine the commit histories (clones into .cache/, needs network)
history:
	python3 scripts/analyze_history.py

## Re-measure the Vega-Lite schema across its release tags
schema:
	python3 scripts/schema_growth.py

watch:
	cd paper && $(LATEXMK) $(LATEXMKFLAGS) -pvc main.tex

clean:
	cd paper && $(LATEXMK) -c main.tex

distclean:
	cd paper && $(LATEXMK) -C main.tex
