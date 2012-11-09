CXXFLAGS=-Wall -DNDEBUG -O2 -pthread

SRCDIR=../../samita/graph
THIRDPARTY_ROOT?=../../../../3rdParty
INCDIRS=-I../../../sam_format/samtools-0.1.7a -I$(SRCDIR) -I$(THIRDPARTY_ROOT)/include -I/usr/include/boost-1_33_1
LIBDIRS=-L$(THIRDPARTY_ROOT)/lib -L/usr/local/lib -L/usr/lib64/boost-1_33_1

TESTS=graph_test

.PHONY: all clean test debug 

all: $(TESTS)

clean:
	rm -f $(TESTS)
	rm -f *.gcov
	rm -f *.gcda
	rm -f *.gcno
	
debug: CXXFLAGS=-g -Wall -DDEBUG -pthread
debug: all 

test: $(TESTS)
	./graph_test
	@diff test_graph.dot wonderland.dot && echo "Test OK"

% : %.cpp
	$(CXX) $(CXXFLAGS) $< $(INCDIRS) -o $@ \
		     -L$(SRCDIR) $(LIBDIRS)  \
		     -l z -l boost_iostreams -l cppunit -ldl
