# Build the paper
# Requires: pdflatex, bibtex (from a TeX distribution like TeX Live or MiKTeX)

MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
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

# Package only figures actually referenced by the manuscript (arXiv
# flags unreferenced files for deletion otherwise)
arxiv: $(MAIN).pdf
	tar czf arxiv-v1.tar.gz $(MAIN).tex $(MAIN).bbl tables/*.tex 	    $$(grep -oE 'figures/[a-z_]+\.pdf' $(MAIN).tex | sort -u)
