# Build the LaTeX report. Figures come from ../../scripts/make_figures.py.
#
#   make            # build the PDF
#   make report.pdf
#   make clean      # remove aux files
#   make distclean  # also remove the PDF

LATEX = pdflatex -interaction=nonstopmode -halt-on-error

all: report.pdf

%.pdf: %.tex
	$(LATEX) $<
	$(LATEX) $<

clean:
	rm -f *.aux *.log *.out *.toc

distclean: clean
	rm -f report.pdf

.PHONY: all clean distclean
