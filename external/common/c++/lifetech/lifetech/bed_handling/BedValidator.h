/*
 * Copyright (c) 2011 Life Technologies Corporation. All rights reserved.
 */

#ifndef BEDVALIDATOR_H_
#define BEDVALIDATOR_H_

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include "loggerutil.h"
using namespace std;

/* Column constants  */
#define  CHROM_IDX 		1
#define  CHROMSTART_IDX 	2
#define  CHROMEND_IDX 		3
#define  NAME_IDX 		4
#define  SCORE_IDX 		5
#define  STRAND_IDX 		6
#define  THICKSTART_IDX 	7
#define  THICKEND_IDX 		8
#define  ITEMRGB_IDX 		9
#define  BLOCKCOUNT_IDX 	10
#define  BLOCKSIZES_IDX 	11
#define  BLOCKSTARTS_IDX 	12

/* Constants used in code */
#define SCORE_MIN 		0
#define SCORE_MAX		1000
#define MIN_COLUMNS 		3

#define int64 int64_t

const char STRAND_PLUS		= '+';
const char STRAND_MINUS 	= '-';

class BedValidator
{

public:

	BedValidator(const string &bedFnIn);
	~BedValidator();
	
	void setExpectedChromNames ( const vector<string> &bamChromNamesIn );
	void setExpectedChromSizes (const vector<int64> &bamChromSizesIn );
	
	bool validate();

private:

	bool checkWithBAMHeader(const string trackName, const string chromName, const int64 chromStart, const int64 chromEnd);
	bool getTrackName(string &trackName, const string trackLine, int &trackCount);

	/*Input Bed file name*/
	string mBedFn;
	
	/* Bed file data container */
	vector<string> mBamChromNames;
	vector<int64> mBamChromSizes;
};

#endif
