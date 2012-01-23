/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REGION_H
#define REGION_H

struct Region {
//	int x, y;	// lower left corner X & Y
	int row, col;	//upper left corner Row and Column
	int w, h;	// width & height of region
};

#endif // REGION_H

