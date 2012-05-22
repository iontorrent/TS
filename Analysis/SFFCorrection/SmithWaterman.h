/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SMITHWATERMAN_H
#define SMITHWATERMAN_H

 #include <string>
 #include <iostream>
 #include "Alignment.h"
 #include "Sequence.h"
 #include "Cell.h"
 #define STOP 0
 #define LEFT 1
 #define DIAGONAL 2
 #define UP 3

namespace ion {
	class SmithWaterman {
	  public:

	  SmithWaterman() {
	  }
	  ~SmithWaterman() {
	  }

	  Alignment* align(std::string s1, std::string s2, float o, float e, float m);

	  private:

	  Cell* construct(std::string seq1, std::string seq2, float o, float e, float m, unsigned char pointers[],
					  int  sizeOfVerticalGaps[], int sizeOfHorizontalGaps[]) ;


	  Alignment* traceback(std::string s1, std::string s2, unsigned char pointers[], Cell *cell,
							int sizeOfVerticalGaps[], int sizeOfHorizontalGaps[]);

	  std::string reverse(char* a, int len) ;
	   std::string reverseComplement(const char *a, int len);
	  float maximum(float a, float b, float c, float d);

	};
 }

#endif // SMITHWATERMAN_H
