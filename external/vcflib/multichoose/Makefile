all: multichoose multipermute

#multichoose_recursive: multichoose_recursive.cpp
#	g++ multichoose_recursive.cpp -o multichoose_recursive

multichoose: multichoose.cpp multichoose.h
	g++ multichoose.cpp -o multichoose

multipermute: multipermute.cpp multipermute.h
	g++ multipermute.cpp -o multipermute

cmultichoose: multichoose.c
	gcc multichoose.c -o cmultichoose

clean:
	rm cmultichoose
	rm multichoose
