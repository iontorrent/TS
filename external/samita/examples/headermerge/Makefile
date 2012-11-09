SAMITA_ROOT=../..
include $(SAMITA_ROOT)/build/common.inc

.PHONY: all debug clean

EXAMPLES=headermerge

all: $(EXAMPLES)

$(EXAMPLES): LDLIBS=-lsamita++ $(LIBS)

#headermerge: CFLAGS=$(DEBUG_FLAGS)
#headermerge: CPPFLAGS+=-I$(THIRDPARTY_INC)/samtools
#headermerge: LDLIBS=-lbam -lz
headermerge: headermerge_cmdline.cpp

headermerge: override CC=$(CXX)

debug: CXXFLAGS=$(DEBUG_FLAGS)
debug: all

clean:
	rm -f $(EXAMPLES) *.o

## Generated Arguments
%.cpp %.h: %.ggo
	-gengetopt -i $< --output-dir=.

headermerge.o: headermerge.cpp \
  ../../samita/common/types.hpp \
  ../../samita/common/interval.hpp \
  ../../samita/exception/exception.hpp \
  ../../../common/c++/include/lifetech/string/util.hpp \
  ../../samita/common/quality_value.hpp \
  ../../samita/align/cigar.hpp \
  ../../samita/align/align_writer.hpp \
  ../../samita/align/align_reader.hpp \
  ../../samita/sam/bam.hpp \
  ../../samita/align/align_reader_util.hpp \
  headermerge_cmdline.h
