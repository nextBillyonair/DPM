PDFLATEX:=pdflatex
BIBTEX:=bibtex


TEX_SRC=presentation.tex

${TEX_SRC:.tex=.pdf}:%.pdf: %.tex
	$(PDFLATEX) $<
	$(PDFLATEX) $<
	$(PDFLATEX) $<

clean:
	rm -f ${TEX_SRC:.tex=.pdf} *.out *.aux *.log *.bbl *.blg *.nav *.snm *.toc


.PHONY: clean
