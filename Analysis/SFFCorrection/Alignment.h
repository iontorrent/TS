/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ALIGNMENT_H
#define ALIGNMENT_H

 #include <string>
 #define GAP '-'


 namespace ion {
	class Alignment {
		public:
		Alignment() {
			open = 0;
			extend = 0;
			score = 0;
			start1 = 0;
			start2 = 0;
			identity = 0;
			similarity = 0;
			gaps = 0;
			lengthSequence1 = 0;
			lengthSequence2 = 0;
		}
		~Alignment() {
		}

		float getExtend();

		void setExtend(float extend);

		std::string getName1();

		void setName1(std::string name);

		std::string getName2();

		void setName2(std::string name);

		float getOpen();

		void setOpen(float open);

		float getScore();

		void setScore(float score);

		std::string getSequence1 ();

		void setSequence1 ( std::string sequence);

		std::string getSequence2 ();

		void setSequence2 ( std::string sequence);

		int getStart1();

		void setStart1(int start);

		int getStart2();

		void setStart2(int start);

		int getGaps();

		void setGaps(int gaps);

		int getIdentity();

		void setIdentity(int identity);

		std::string getMarkupLine();

		void setMarkupLine(std::string markup);

		int getSimilarity();

		void setSimilarity(int similarity);

		float calculateScore();

		int checkScore();

		int getLengthOfSequence1();

		int getLengthOfSequence2();

		void setLengthOfSequence1(int length);

		void setLengthOfSequence2(int length);

		private:
		float open;
		float extend;
		float score;
		std::string sequence1;
		std::string name1;
		int start1;
		std::string sequence2;
		std::string name2;
		int start2;

		std::string markupLine;
		int identity;
		int similarity;
		int gaps;
		int lengthSequence1;
		int lengthSequence2;


	};

 }

#endif // ALIGNMENT_H
