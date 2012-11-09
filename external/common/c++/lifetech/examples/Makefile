EXAMPLES=logging

CLEAN=$(addsuffix -clean,$(EXAMPLES))

.PHONY: all clean $(EXAMPLES) $(CLEAN)

all: $(EXAMPLES)

$(EXAMPLES):: %: %/Makefile
	@echo "BUILDING EXAMPLES: $@ "
	$(MAKE) -C $@

$(CLEAN):
	-$(MAKE) -C $(@:-clean=) clean

clean: $(CLEAN)
