SAMITA_ROOT=../..
include $(SAMITA_ROOT)/build/common.inc

.PHONY: all debug clean

EXAMPLES=align_example

all: $(EXAMPLES)

$(EXAMPLES): LDLIBS=-lsamita++ $(LIBS)

debug: CXXFLAGS=$(DEBUG_FLAGS)
debug: all

clean:
	rm -f $(EXAMPLES)
