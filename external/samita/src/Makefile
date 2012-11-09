SAMITA_ROOT=..
include $(SAMITA_ROOT)/build/common.inc

LIBSAMITA=$(SAMITA_ROOT)/lib/libsamita++

SOURCES=align/align_reader.cpp
SOURCES+=align/align_writer.cpp
SOURCES+=reference/reference.cpp
SOURCES+=sam/bam.cpp
SOURCES+=sam/bam_metadata.cpp
SOURCES+=pileup/pileup.cpp
SOURCES+=xsq/xsq_io.cpp

OBJ := $(SOURCES:.cpp=.o)

.PHONY: all debug clean

all: depend $(LIBSAMITA).so $(LIBSAMITA).a

debug: CXXFLAGS=$(DEBUG_FLAGS) -fPIC
debug: all

clean:
	rm -f depend depend.* $(OBJ) $(LIBSAMITA).so $(LIBSAMITA).a

xsq/xsq_io.o: override CPPFLAGS+=-DH5_USE_16_API

# Force C++ compiler for linking
#$(LIBSAMITA).so: override CC=$(CXX)

$(LIBSAMITA).so: override LDFLAGS+=-shared
$(LIBSAMITA).so $(LIBSAMITA).a: LDLIBS+=-llog4cxx -lapr-1 -laprutil-1 -lboost_iostreams -lz

## Object files in a shared library must be compiled with PIC
$(OBJ): override CPPFLAGS+=-fPIC

$(LIBSAMITA).so: $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(LIBSAMITA).a: CC=$(CXX)
$(LIBSAMITA).a: LDFLAGS+=-static

$(LIBSAMITA).a: $(OBJ)
	$(AR) -crus $@ $^

## Auto build dependency tree for library code
## Don't use CPPFLAGS -I, or boost and samtools are pulled in (boost is huge)
## Executables have just executable as target, or .o is generated and C is used for linking instead of C++
depend: Makefile $(SOURCES)
	@echo "Rebuilding dependency list"
	@set -e; rm -fv depend; \
	  for i in $(SOURCES); do \
	    $(CXX) $(CPPFLAGS) -MT $${i%.cpp}.o -MG -MM $$i >> depend.$$$$; \
	  done; \
	  mv depend.$$$$ depend

ifeq (depend, $(wildcard depend))
      include depend
endif
