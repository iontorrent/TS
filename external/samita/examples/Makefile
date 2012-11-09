EXAMPLES=align filter gff pileup reference metadata samdiff bedvalidator

CLEAN=$(addsuffix -clean,$(EXAMPLES))

.PHONY: all clean $(EXAMPLES) $(CLEAN)

all: $(EXAMPLES)

$(EXAMPLES):: %: %/Makefile
	@echo "BUILDING EXAMPLES: $@ "
	$(MAKE) -C $@

$(CLEAN):
	-$(MAKE) -C $(@:-clean=) clean

clean: $(CLEAN)
