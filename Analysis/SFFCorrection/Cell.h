/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef CELL_H
#define CELL_H

 #include <string>

 namespace ion {
	class Cell {
		public:
		Cell();
		Cell(int row, int col, float score);
		~Cell() {
		};

		int getRow();
		int getCol();

		float getScore();

		void setRow(int row);

		void setCol(int col);

		void setScore(float score);

		void set (int row, int col, float score);

		private:
		int row;
		int col;
		float score;

	};
 }

#endif // CELL_H
