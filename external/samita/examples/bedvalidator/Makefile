SAMITA_ROOT=../..
include $(SAMITA_ROOT)/build/common.inc

.PHONY: all debug clean

EXAMPLES=BedValidator

all: $(EXAMPLES)

$(EXAMPLES): LDLIBS=-lsamita++ $(LIBS)

BedValidator.o: $(SAMITA_ROOT)/samita/common/BedValidator.hpp

debug: CXXFLAGS=$(DEBUG_FLAGS)
debug: all

clean:
	rm -f $(EXAMPLES) *.o
