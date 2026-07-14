# Build the Phase IIa short paper
# Requires: pdflatex, bibtex (TeX Live or MiKTeX)

MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib tables/*.tex
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out
	rm -f $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz

distclean: clean
	rm -f $(MAIN).pdf

.PHONY: all clean distclean

arxiv: $(MAIN).pdf
	tar czf arxiv-gauge-v1.tar.gz $(MAIN).tex $(MAIN).bbl figures/*.pdf tables/*.tex
