CXX?=g++
CXXFLAGS=-Wall -DNDEBUG -O2 -fPIC

SRC=../src

INCDIRS=-I$(SRC) -I/usr/include/ -I../../graph/src -I../../../common/c++/include/ -I../../../sam_format/samtools-0.1.7a/ -I../../../samita/src -I../../../common/c++/cppunit/include -I/usr/include/boost-1_33_1/
LIBDIRS=-L../../../common/c++/cppunit/lib -L/usr/local/lib -L../../../common/c++/lib -L../../../sam_format/samtools-0.1.7a -L../../../samita/src -L/usr/lib64/boost-1_33_1

.PHONY: all clean test debug 

all : interval_lattice.cpp
	$(CXX) $(CXXFLAGS) interval_lattice.cpp $(INCDIRS) -o cgat \
		     -L$(SRC_DIR) $(LIBDIRS)  \
		     -lz -lbam -lsamita++ -llog4cxx -lapr-1 -laprutil-1 -lexpat -lpthread -lboost_iostreams -lcppunit -ldl -lboost_program_options
	mv cgat ../bin
			     
clean:
	rm -f $(TESTS)
	rm -f *.gcov
	rm -f *.gcda
	rm -f *.gcno
	
debug: CXXFLAGS=-g -Wall -DDEBUG -pthread
debug: all


