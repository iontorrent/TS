/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

 #include <string>
 #include "Cell.h"

 namespace ion {
		Cell::Cell() {
		  row  =0;
		  col =0;
		  score = 0;
		}

		Cell::Cell(int r, int c, float s) {
			row = r;
			col = c;
			score = s;
		}

		int Cell::getRow() {
			return row;
		}
		int Cell::getCol() {
			return col;
		}

		float Cell::getScore() {
			return score;
		}

		void Cell::setRow(int r) {
			row = r;
		}

		void Cell::setCol(int c) {
			col = c;
		}

		void Cell::setScore(float s) {
			score = s;
		}

		void Cell::set (int r, int c, float s) {
			row = r;
			col = c;
			score = s;
		}

 }

