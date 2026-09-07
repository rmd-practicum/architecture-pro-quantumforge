PUML := $(shell find . -type f -name '*.puml')
SVG  := $(PUML:.puml=.svg)

.PHONY: diagrams
diagrams: $(SVG)

%.svg: %.puml
	plantuml -nometadata -svg $<

.PHONY: clean-diagrams
clean-diagrams:
	$(RM) $(SVG)
