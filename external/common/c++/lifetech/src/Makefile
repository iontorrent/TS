LIFETECH_ROOT=..
include $(LIFETECH_ROOT)/build/common.inc

LIB=$(LIFETECH_LIB)/liblifetech.so

CXX?=g++

SOURCES=logging/util.cpp

#override CXXFLAGS+=-fno-strict-aliasing
#override CXXFLAGS+=-pthread

.PHONY: all debug clean

all: $(LIB) $(OBJ) 

debug: override CXXFLAGS=$(DEBUG_FLAGS)
debug: all

clean:
	rm -f *.o $(LIB)
	
$(LIB) :
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(SHLIB_FLAGS) $(SOURCES) $(HEADER_FLAGS) $(LIBRARY_FLAGS) -o $(LIB)
