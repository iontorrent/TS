SAMITA_ROOT=../..
include $(SAMITA_ROOT)/build/common.inc

.PHONY: all debug clean

EXAMPLES=PileupTry

all: $(EXAMPLES)

$(EXAMPLES): LDLIBS=-lsamita++ $(LIBS)

sam_pileup: CFLAGS=$(DEBUG_FLAGS)
sam_pileup: CPPFLAGS+=-I$(THIRDPARTY_INC)/samtools
sam_pileup: LDLIBS=-lbam -lz

debug: CXXFLAGS=$(DEBUG_FLAGS)
debug: all

clean:
	rm -f $(EXAMPLES)
