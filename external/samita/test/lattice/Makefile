CXXFLAGS=-Wall -DNDEBUG -O2

SRC=../src

INCDIRS=-I$(SRC) -I/usr/include/ -I../../graph/src -I../../../common/c++/include/  -I../../../sam_format/samtools-0.1.7a/ -I../../../samita/src -I../../../common/c++/cppunit/include -I/usr/include/boost-1_33_1/
LIBDIRS=-L../../../common/c++/cppunit/lib -L/usr/local/lib -L/usr/lib64/boost-1_33_1

TESTS=interval_lattice_test

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
	./interval_lattice_test

% : %.cpp
	$(CXX) $(CXXFLAGS) $< $(INCDIRS) -o $@ \
		     -L$(SRC_DIR) $(LIBDIRS)  \
		     -lz -lboost_iostreams -lcppunit -ldl
