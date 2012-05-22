/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

 #include <string>
 #include "string.h"
 #include "Alignment.h"

 #define GAP '-'


 namespace ion {

		float Alignment::getExtend() {
			return extend;
		}

		void Alignment::setExtend(float ext) {
			extend = ext;
		}

		std::string Alignment::getName1() {
			return name1;
		}

		void Alignment::setName1(std::string name) {
			name1 = name;
		}

		std::string Alignment::getName2() {
			return name2;
		}

		void Alignment::setName2(std::string name) {
			name2 = name;
		}

		float Alignment::getOpen() {
			return open;
		}

		void Alignment::setOpen(float o) {
			open = o;
		}

		float Alignment::getScore() {
			return score;
		}

		void Alignment::setScore(float s) {
			score = s;
		}

		std::string Alignment::getSequence1 () {
			return sequence1;
		}

		void Alignment::setSequence1 ( std::string seq) {
			sequence1 = seq;
		}

		std::string Alignment::getSequence2 () {
			return sequence2;
		}

		void Alignment::setSequence2 ( std::string seq) {
			sequence2 = seq;
		}

		int Alignment::getStart1() {
			return start1;
		}

		void Alignment::setStart1(int start) {
			start1 = start;
		}

		int Alignment::getStart2() {
			return start2;
		}

		void Alignment::setStart2(int start) {
			start2 = start;
		}

		int Alignment::getGaps() {
			return gaps;
		}

		void Alignment::setGaps(int g) {
			gaps = g;
		}

		int Alignment::getIdentity() {
			return identity;
		}

		void Alignment::setIdentity(int i) {
			identity = i;
		}

		std::string Alignment::getMarkupLine() {
			return markupLine;
		}

		void Alignment::setMarkupLine(std::string markup) {
			markupLine = markup;
		}

		int Alignment::getSimilarity() {
			return similarity;
		}

		void Alignment::setSimilarity(int s) {
			similarity = s;
		}

		void Alignment::setLengthOfSequence1(int l) {
			lengthSequence1 = l;
		}

		int Alignment::getLengthOfSequence1() {
			return lengthSequence1;
		}

		void Alignment::setLengthOfSequence2(int l) {
			lengthSequence2 = l;
		}

		int Alignment::getLengthOfSequence2() {
			return lengthSequence2;
		}


		float Alignment::calculateScore() {
		   float tempScore = 0;
		    int prev1Gap = 0;
			int prev2Gap = 0;
			char c1, c2;
			for (int i = 0, n = sequence1.length(); i < n; i++ ) {
				c1 = sequence1.at(i);
				c2 = sequence2.at(i);

				if (c1 == GAP) {
					if (prev1Gap)
						tempScore -= extend;
					else
						tempScore -= open;

					prev1Gap = 1;
				}
				else if (c2 == GAP) {
					if (prev2Gap)
						tempScore -= extend;
					else
						tempScore -= open;

					prev2Gap = 1;
				}
				else {
					if (c1 == c2)
						tempScore += 1;
					else
						tempScore -= 1;

					prev1Gap = 0;
					prev2Gap = 0;
				}

			}

			return tempScore;

		}

		int Alignment::checkScore() {
			if (calculateScore() == score)
				return 1;
			else
				return 0;
		}



 }
