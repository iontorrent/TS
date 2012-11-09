SAMITA_ROOT=../..
include $(SAMITA_ROOT)/build/common.inc

.PHONY: all debug clean

EXAMPLES=remarkpairs

all: $(EXAMPLES)

$(EXAMPLES): LDLIBS=-lsamita++ $(LIBS)

remarkpairs: remarkpairs_cmdline.o
remarkpairs: override CC=$(CXX)

debug: CXXFLAGS=$(DEBUG_FLAGS)
debug: all

clean:
	rm -f $(EXAMPLES) *.o

## Generated Arguments
%.cpp %.h: %.ggo
	-gengetopt -i $< --output-dir=.

