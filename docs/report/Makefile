# Build the LaTeX reports. Figures come from ../../scripts/make_figures.py.
#
#   make            # build both PDFs
#   make report_ru.pdf
#   make clean      # remove aux files
#   make distclean  # also remove PDFs

LATEX = pdflatex -interaction=nonstopmode -halt-on-error

all: report_ru.pdf report_en.pdf

%.pdf: %.tex
	$(LATEX) $<
	$(LATEX) $<

clean:
	rm -f *.aux *.log *.out *.toc

distclean: clean
	rm -f report_ru.pdf report_en.pdf

.PHONY: all clean distclean
