SAMITA_ROOT=../..
include $(SAMITA_ROOT)/build/common.inc

.PHONY: all clean test debug 

all: xsqTestRunner testRunner

clean:
	rm -f xsqTestRunner
	rm -f xsqTestRunner.o
	rm -f testPerformance
	rm -f result.xml
	rm -f data/test.sorted.a.bam.bai
	rm -f data/test.sorted.b.bam.bai

debug: CXXFLAGS=$(DEBUG_FLAGS)
debug: all 

override CXXFLAGS+=-g
override LDFLAGS+=-rdynamic

test: debug
	-LD_LIBRARY_PATH=$(LIFETECH_LIB):$(SAMITA_LIB):$(THIRDPARTY_LIB) ./xsqTestRunner unit
	#LD_LIBRARY_PATH=$(LIFETECH_LIB):$(SAMITA_LIB):$(COMMON_LIB) ./testPerformance

override CC=$(CXX)

## LDLIBS must be prefixed with these values
xsqTestRunner testPerformance testRunner: LDLIBS:=-lsamita++ $(LDLIBS) -Wl,-no-as-needed -rdynamic -lcppunit -ldl -ltbb

xsqTestRunner testPerformance testRunner: override CPPFLAGS+=-DH5_USE_16_API -pthread

xsqTestRunner.o: xsqTestRunner.cpp XsqTest.hpp $(SAMITA_ROOT)/samita/xsq/xsq_io.hpp
