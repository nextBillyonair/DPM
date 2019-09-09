PDFLATEX:=pdflatex
BIBTEX:=bibtex


TEX_SRC=doc.tex

${TEX_SRC:.tex=.pdf}:%.pdf: %.tex subsections/*.tex #%.bib
	$(PDFLATEX) $<
	# $(BIBTEX) ${<:.tex=}
	$(PDFLATEX) $<
	$(PDFLATEX) $<
	rm -f *.out *.aux *.log *.bbl *.blg

clean:
	rm -f *.out *.aux *.log *.bbl *.blg


.PHONY: clean
