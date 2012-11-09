SAMITA_ROOT=../..
include $(SAMITA_ROOT)/build/common.inc

.PHONY: all debug clean

EXAMPLES=ExtractHeader

all: $(EXAMPLES)

$(EXAMPLES): CC=$(CXX)
$(EXAMPLES): LDLIBS=-lsamita++ $(LIBS)

debug: CXXFLAGS=$(DEBUG_FLAGS)
debug: all

clean:
	rm -f $(EXAMPLES) *.o

ExtractHeader.o: ExtractHeader.cpp \
  ../../samita/align/align_reader.hpp ../../samita/common/types.hpp \
  ../../samita/common/interval.hpp ../../samita/exception/exception.hpp \
  ../../samita/common/quality_value.hpp ../../samita/align/cigar.hpp \
  ../../samita/sam/bam.hpp ../../samita/align/align_reader_util.hpp \
  ../../samita/sam/bam_metadata.hpp
