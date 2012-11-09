LIFETECH_ROOT=..
include $(LIFETECH_ROOT)/build/common.inc

.PHONY: all clean test debug 

CXX?=g++

all: testRunner

clean:
	rm -f testRunner
	
debug: CXXFLAGS=$(DEBUG_FLAGS)
debug: all 

test: all
	./testRunner 

testPerformance:

testRunner:

% : %.cpp
	$(CXX) $(CXXFLAGS) $< $(HEADER_FLAGS) $(LIBRARY_FLAGS) -o $@ $(LIFETECH_LIBS) -l cppunit $(THIRD_PARTY_LIBS) 
