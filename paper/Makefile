# Build the paper
# Requires: pdflatex, bibtex (from a TeX distribution like TeX Live or MiKTeX)

MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

# 12 pt single-column review build for MLST submission (IOP prefers >= 12 pt).
# Produces main_submission.pdf; the twocolumn build above is unchanged.
submission: $(MAIN).tex references.bib
	sed 's/,twocolumn,/,preprint,/' $(MAIN).tex > main_submission.tex
	pdflatex main_submission
	bibtex main_submission
	pdflatex main_submission
	pdflatex main_submission

# Double-anonymous review build (IOP anonymisation checklist): 12 pt single
# column with author block, funding/COI statements, and repo URLs removed or
# replaced by the anonymized mirror. Upload the PDF only -- the generated .tex
# still contains identity in \else branches.
anonymous: $(MAIN).tex references.bib
	sed -e 's/,twocolumn,/,preprint,/' -e 's/anonymousfalse/anonymoustrue/' $(MAIN).tex > main_anonymous.tex
	pdflatex main_anonymous
	bibtex main_anonymous
	pdflatex main_anonymous
	pdflatex main_anonymous
	@grep -q "qft-graph-XXXX" main_anonymous.tex && echo "*** WARNING: anonymized-mirror placeholder URL still present -- create the mirror at https://anonymous.4open.science and update main.tex ***" || true

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out
	rm -f main_submission.* main_anonymous.*
	rm -f $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz

distclean: clean
	rm -f $(MAIN).pdf

.PHONY: all clean distclean

# Package only figures actually referenced by the manuscript (arXiv
# flags unreferenced files for deletion otherwise)
arxiv: $(MAIN).pdf
	tar czf arxiv-v1.tar.gz $(MAIN).tex $(MAIN).bbl tables/*.tex 	    $$(grep -oE 'figures/[a-z_]+\.pdf' $(MAIN).tex | sort -u)
