CPPFLAGS=-fPIC -I../../buildTools
SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)
LIB = libkmeans.a

$(LIB): $(OBJ)
	ar rcs $(LIB) $(OBJ)
