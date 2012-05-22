/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

 #include "SmithWaterman.h"

 namespace ion {
	Alignment* SmithWaterman::align(std::string s1, std::string s2, float o, float e, float mp) {

		//fprintf(stdout, "Sequence1 = %s \n", s1.c_str());
		//fprintf(stdout, "Sequence2 = %s \n", s2.c_str());

		std::string seq1 = s1;
		std::string seq2 = s2;

		//std::cout << "add of seq1 = " << &seq1 << " add of s1 =" << &s1 << std::endl;

		int m = s1.length() + 1;
		int n = s2.length() + 1;
                if (m>600 || n>600)
                    std::cout << "m: " << m << " , n: " << n << std::endl;	
		unsigned char* pointers = new unsigned char[m*n];
		// Initializes the boundaries of the traceback matrix to STOP.
		for (int i = 0, k = 0; i < m; i++, k += n) {
			pointers[k] = STOP;
		}
		for (int j = 1; j < n; j++) {
			pointers[j] = STOP;
		}

		int* sizesOfVerticalGaps = new int[m * n];
		int* sizesOfHorizontalGaps = new int[m * n];
		for (int i = 0, k = 0; i < m; i++, k += n) {
			for (int j = 0; j < n; j++) {
				sizesOfVerticalGaps[k + j] = sizesOfHorizontalGaps[k + j] = 1;
			}
		}

		//fprintf(stdout, "Calling construct m = %d, n = %d\n", m, n);
		Cell *cell = construct(seq1, seq2, o, e, mp, pointers, sizesOfVerticalGaps, sizesOfHorizontalGaps);
		//fprintf(stdout, "Calling traceback s1 = , s2 =  \n") ;
		//Cell *cell = new Cell();
		Alignment* alignment = traceback(s1, s2,  pointers, cell, sizesOfVerticalGaps, sizesOfHorizontalGaps);
		//alignment->setName1();
		//alignment->setName2();
		//alignment.setMatrix(matrix);
		alignment->setOpen(o);
		alignment->setExtend(e);
	        //std::cout << "end of traceback call" << std::endl;
		delete cell;
		delete[] pointers;
                delete[] sizesOfVerticalGaps;
                delete[] sizesOfHorizontalGaps;
		return alignment;
	}

	Cell* SmithWaterman::construct(std::string seq1, std::string seq2, float o,
			float e, float mp, unsigned char pointers[], int sizesOfVerticalGaps[],
			int sizesOfHorizontalGaps[]) {


		//char* a1 = new char[seq1.length()+1];
		//char* a2 = new char[seq2.length()+1];
		//strcpy(a1, seq1.c_str());
		//strcpy(a2, seq2.c_str());

 		std::string a1 = seq1;
		std::string a2 = seq2;

		int m = seq1.length() + 1;
		int n = seq2.length() + 1;

		//fprintf(stdout, "construct s1 = %s, s2 =%s \n", a1.c_str(), a2.c_str());
		//std::cout << "add of seq1 in construct = " << &seq1 << std::endl;

		float f; // score of alignment x1...xi to y1...yi if xi aligns to yi
		float g[n]; // score if xi aligns to a gap after yi
		float h; // score if yi aligns to a gap after xi
		float v[n]; // best score of alignment x1...xi to y1...yi
		float vDiagonal;

		g[0] = -999999.99;
		h = -999999.99;
		v[0] = 0;

		for (int j = 1; j < n; j++) {
			g[j] = -999999.99;
			v[j] = 0;
		}

		float similarityScore, g1, g2, h1, h2;

		Cell *cell = new Cell();

		for (int i = 1, k = n; i < m; i++, k += n) {
			h = -999999.99;
			vDiagonal = v[0];
			for (int j = 1, l = k + 1; j < n; j++, l++) {
				if (a1.at(i - 1) == a2.at(j - 1)) similarityScore = 1;
				else
					similarityScore = mp;


				// Fill the matrices
				f = vDiagonal + similarityScore;

				g1 = g[j] - e;
				g2 = v[j] - o;
				if (g1 > g2) {
					g[j] = g1;
					sizesOfVerticalGaps[l] = (sizesOfVerticalGaps[l - n] + 1);
				} else {
					g[j] = g2;
				}

				h1 = h - e;
				h2 = v[j - 1] - o;
				if (h1 > h2) {
					h = h1;
					sizesOfHorizontalGaps[l] = (sizesOfHorizontalGaps[l - 1] + 1);
				} else {
					h = h2;
				}

				vDiagonal = v[j];
				v[j] = maximum(f, g[j], h, 0);



				// Determine the traceback direction
				if (v[j] == 0) {
					pointers[l] = STOP;
				} else if (v[j] == f) {
					pointers[l] = DIAGONAL;
				} else if (v[j] == g[j]) {
					pointers[l] = UP;
				} else {
					pointers[l] = LEFT;
				}

				// Set the traceback start at the current cell i, j and score
				if (v[j] > cell->getScore()) {
					cell->set(i, j, v[j]);
				}
			}

		}

		//std::cout << "add of seq1 in construct end = " << &seq1 << std::endl;
		//fprintf(stdout, "construct end s1 = %s \n", seq1.c_str());
		//std::cout << "add of a1 " << &a1 << " add of a2 " << &a2 << std::endl;
		//delete[] a1;
		//delete[] a2;
		//std::cout << " after delete a1 & a2" << std::endl;

		return cell;
	}

	Alignment* SmithWaterman::traceback(std::string s1, std::string s2,
			unsigned char  pointers[], Cell *cell, int sizesOfVerticalGaps[],
			int sizesOfHorizontalGaps[]) {

		//fprintf(stdout, "traceback s1 = %s, s2 = %s \n", s1.c_str() , s2.c_str() );
		//char* a1 = new char[s1->getSequence().size()+1];
		//char* a2 = new char[s2->getSequence().size()+1];
		//strcpy(a1, s1->getSequence().c_str());
		//strcpy(a2, s2->getSequence().c_str());

		//float[][] scores = m.getScores();

		std::string a1 = s1;
		std::string a2 = s2;

		int n = s2.length() + 1;

		//fprintf(stdout, "length of s2 = %d \n", n);

		Alignment *alignment = new Alignment();
		alignment->setScore(cell->getScore());

		int maxlen = s1.length() + s2.length(); // maximum length after the
												// aligned sequences

		//fprintf(stdout, "maxlen = %d \n", maxlen);

		char reversed1[maxlen]; // reversed sequence #1
		char reversed2[maxlen]; // reversed sequence #2
		char reversed3[maxlen]; // reversed markup

		//initialize reversed
		for (int i = 0; i < maxlen; i++) {
			reversed1[i] = '\0';
			reversed2[i] = '\0';
			reversed3[i] = '\0';
		}
		int len1 = 0; // length of sequence #1 after alignment
		int len2 = 0; // length of sequence #2 after alignment
		int len3 = 0; // length of the markup line

		int lenBase1 = 0; //length of sequence1 bases not including gaps
		int lenBase2 = 0; //length of sequence 2 bases not including gaps

		int identity = 0; // count of identitcal pairs
		int similarity = 0; // count of similar pairs
		int gaps = 0; // count of gaps

		char c1, c2;

		int i = cell->getRow(); // traceback start row
		int j = cell->getCol(); // traceback start col
		int k = i * n;

		int stillGoing = 1; // traceback flag: true -> continue & false
								   // -> stop
		//fprintf(stdout, "row = %d, col = %d, k = %d \n", i, j, n);

		while (stillGoing) {
			switch (pointers[k + j]) {
			case UP:
				for (int l = 0, len = sizesOfVerticalGaps[k + j]; l < len; l++) {
					reversed1[len1++] = a1.at(--i);
					reversed2[len2++] = '-';
					reversed3[len3++] = ' ';
					k -= n;
					gaps++;
					lenBase1++;
				}
				break;
			case DIAGONAL:
				c1 = a1.at(--i);
				c2 = a2.at(--j);
				k -= n;
				lenBase1++;
				lenBase2++;
				reversed1[len1++] = c1;
				reversed2[len2++] = c2;
				if (c1 == c2) {
					reversed3[len3++] = '|';
					identity++;
					similarity++;
				} else {
					reversed3[len3++] = '.';
				}
				break;
			case LEFT:
				for (int l = 0, len = sizesOfHorizontalGaps[k + j]; l < len; l++) {
					reversed1[len1++] = '-';
					reversed2[len2++] = a2.at(--j);
					reversed3[len3++] = ' ';
					gaps++;
					lenBase2++;
				}
				break;
			case STOP:
				stillGoing = 0;
			}
		}
		//fprintf(stdout, "finished stillGoing \n");
		//std::cout << "traceback len1 = " << len1 << std::endl;
		//std::cout << "traceback rev1 = " << reversed1 << std::endl;

		//std::cout << "traceback rev1 len = " << strlen(reversed1) << std::endl;
		alignment->setSequence1(reverse(reversed1, len1));
		alignment->setStart1(i);
		alignment->setLengthOfSequence1(lenBase1);
		alignment->setSequence2(reverse(reversed2, len2));
		alignment->setStart2(j);
		alignment->setLengthOfSequence2(lenBase2);
		alignment->setMarkupLine(reverse(reversed3, len3));
		alignment->setIdentity(identity);
		alignment->setGaps(gaps);
		alignment->setSimilarity(similarity);

		//delete[] a1;
		//delete[] a2;

		return alignment;
	}

	float SmithWaterman::maximum(float a, float b, float c, float d) {
		if (a > b) {
			if (a > c) {
				return a > d ? a : d;
			} else {
				return c > d ? c : d;
			}
		} else if (b > c) {
			return b > d ? b : d;
		} else {
			return c > d ? c : d;
		}
	}

	/**
	 * Reverses an array of chars
	 *
	 * @param a
	 * @param len
	 * @return the input array of char reserved
	 */
	std::string SmithWaterman::reverse(char* a, int len) {
		char* b = new char[len + 1];

		for (int i = len - 1, j = 0; i >= 0; i--, j++) {
			b[j] = a[i];
		}
		b[len] = '\0';

		std::string s (b);
	        //std::cout << " len in reverse = " << len << std::endl;
                //std::cout << " string in reversed = " << s << std::endl;
		delete[] b;
		return s;
	}

	std::string SmithWaterman::reverseComplement(const char *a, int len) {
		char *b = new char[len + 1];
		for (int i = len -1, j = 0; i >= 0; i--, j++) {
			if (a[i] == 'A')
				b[j] = 'T';
			else if (a[i] == 'T')
				b[j] = 'A';
			else if (a[i] == 'C')
				b[j] = 'G';
			else if (a[i] == 'G')
				b[j] = 'C';
			else
				b[j] = a[i];
		}
		b[len] = '\0';
		std::string s(b);
		delete [] b;
		return s;
	}

 }

