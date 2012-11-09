LIFETECH_ROOT=../..
include $(LIFETECH_ROOT)/build/common.inc

.PHONY: all debug clean

CXX?=g++
EXAMPLES=logging_example

all: $(EXAMPLES)

debug: CXXFLAGS=$(DEBUG_FLAGS)
debug: all

clean:
	rm -f $(EXAMPLES)
	
% : %.cpp
	$(CXX) $(CXXFLAGS) $< $(HEADER_FLAGS) $(LIBRARY_FLAGS) -o $@ $(LIFETECH_LIBS) $(THIRD_PARTY_LIBS) 
