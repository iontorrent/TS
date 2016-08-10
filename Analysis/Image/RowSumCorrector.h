/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * RowSumCorrector.h
 *
 *  Created on: Aug 13, 2015
 *      Author: ionadmin
 */

#ifndef ROWSUMCORRECTOR_H_
#define ROWSUMCORRECTOR_H_

#ifdef BB_DC
#include "datacollect_global.h"
#else
#include "Image.h"
int CorrectRowAverages(char *srcdir, char *datname, Image *img);
int GenerateRowCorrFile(char *srcdir, char *name);
#endif

struct rowcorr_header {
    uint32_t magic;
    uint32_t version;
    uint32_t rows;			// number of rows
    uint32_t frames;		// number of frames per file
    uint32_t framerate;		// number of frames per second
    uint32_t pixsInSums;	// number of pixels per sum
    uint32_t sumsPerRow;	// number of sum per row,
};

#define ROWCORR_MAGIC_VALUE    0xFF115E3B



short int *CreateRowSumCorrection(short int *image, int rows, int cols, int frames);
int WriteRowSumCorrection(char *fname, short int *corr, int rows, int frames);



#endif /* ROWSUMCORRECTOR_H_ */
