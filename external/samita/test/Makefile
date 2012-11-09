SAMITA_ROOT=..
include $(SAMITA_ROOT)/build/common.inc

.PHONY: all clean test debug bam-test xsq-test

all: testRunner
	$(MAKE) -C xsq

clean:
	rm -f *.o
	rm -f testRunner
	rm -f testPerformance
	rm -f result.xml
	rm -f data/test.sorted.a.bam.bai
	rm -f data/test.sorted.b.bam.bai
	$(MAKE) -C xsq clean

debug: CXXFLAGS=$(DEBUG_FLAGS)
debug: all 

CXXFLAGS+=-g -rdynamic

test: bam-test xsq-test

xsq-test: debug
	$(MAKE) -C xsq test

bam-test: testRunner
	-LD_LIBRARY_PATH=$(LIFETECH_LIB):$(SAMITA_LIB):$(THIRDPARTY_LIB) ./testRunner
	#LD_LIBRARY_PATH=$(LIFETECH_LIB):$(SAMITA_LIB):$(COMMON_LIB) ./testPerformance

## Force C++ compiler for linking
override CC=$(CXX)

## LDLIBS must be prefixed with these values
testRunner testPerformance: LDLIBS:=-lsamita++ $(THIRD_PARTY_LIBS) -Wl,-no-as-needed -lcppunit -ldl
## Force use of samita++ in ../lib
testRunner testPerformance: LDFLAGS+=-Wl,-rpath,'$$ORIGIN/../lib'

testRunner: testRunner.o

testRunner.o: testRunner.cpp common/IntervalTest.hpp \
  ../samita/common/interval.hpp ../samita/exception/exception.hpp \
  ../../common/c++/include/lifetech/string/util.hpp \
  common/FeatureTest.hpp ../samita/common/feature.hpp \
  ../samita/common/types.hpp ../samita/common/quality_value.hpp \
  ../samita/align/cigar.hpp align/AlignTest.hpp \
  ../samita/align/align_reader.hpp ../samita/sam/bam.hpp gff/GffTest.hpp \
  ../samita/gff/gff.hpp ../samita/common/record_stream.hpp \
  fastq/FastqTest.hpp ../samita/fastq/fastq.hpp \
  reference/ReferenceTest.hpp ../samita/reference/reference.hpp \
  sam/BasTest.hpp ../samita/sam/bas.hpp filter/FilterTest.hpp \
  ../samita/filter/filter.hpp ../samita/filter/mate_filter.hpp \
  pileup/PileupTest.hpp ../samita/pileup/pileup.hpp \
  ../samita/pileup/pileup_builder.hpp ../samita/pileup/pileup.hpp \
  sam/MetadataTest.hpp ../samita/sam/bam_metadata.hpp
testPerformance.o: testPerformance.cpp ../samita/align/align_reader.hpp \
  ../samita/common/types.hpp ../samita/common/interval.hpp \
  ../samita/exception/exception.hpp \
  ../../common/c++/include/lifetech/string/util.hpp \
  ../samita/common/quality_value.hpp ../samita/align/cigar.hpp \
  ../samita/sam/bam.hpp
