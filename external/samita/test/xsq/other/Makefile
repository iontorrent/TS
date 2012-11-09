SAMITA_ROOT=../../..
include $(SAMITA_ROOT)/build/common.inc

.PHONY: all clean test debug 

all: Feasibility

clean:
	rm -f Feasibility

debug: CXXFLAGS=$(DEBUG_FLAGS)
debug: all 

CXXFLAGS+=-g -rdynamic

test: debug
	-LD_LIBRARY_PATH=$(LIFETECH_LIB):$(SAMITA_LIB):$(THIRDPARTY_LIB) ./Feasibility

export CC=$(CXX)

## LDLIBS must be prefixed with these values
xsqTestRunner testPerformance: LDLIBS:=-lsamita++ -lcppunit -ldl $(LDLIBS)

Feasibility: Feasibility.cpp
	$(CC) $(CXXFLAGS) $(CPPFLAGS) Feasibility.cpp $(LDFLAGS) $(LDLIBS) -o $@
