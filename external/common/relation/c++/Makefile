CXX?=g++
BIN=test_graph
CXXFLAGS=-Wall -pedantic -DNDEBUG -O2
LIBINC=
LIB=

SRC=test_graph.cpp
OBJ=$(SRC:.cpp=.o)

.PHONY: all debug clean docs clean_docs

all: $(LIB) $(OBJ)
	$(CXX) $(OBJ) -o $(BIN) -lboost_serialization

debug: CXXFLAGS=-g -Wall -DDEBUG
debug: all

docs:
	doxygen doxygen.conf

clean_docs:
	rm -rf ../doc

clean: clean_docs
	rm -f *.o $(LIB)

test_graph.o: test_graph.cpp
	$(CXX) $(CXXFLAGS) -c test_graph.cpp $(LIBINC) -o test_graph.o
