/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

/*
 * RowSumData.h
 *
 *  Created on: Jan 21, 2014
 *      Author: awong
 */

#ifndef ROWSUMDATA_H
#define ROWSUMDATA_H

#include <fstream>
#include<vector>
#include "../datahdr.h"
#include <algorithm>

// rowsum data class for reading and storing sensing electrode data
// rowsum data is always full chip (all rows).  Thus row arguments is always in full chip coordinates

// TODO would there be a condition that different process is trying to read the same rowsum file? any problem?


class RowSumData {
public:
	RowSumData();
	virtual ~RowSumData();

	// read rowsum data from file.  The raw data is stored in rowsum private member.
	// returns 0 if success
	// returns 1 if error opening file
	// returns 2 if error reading or closing file
	int LoadRowSumData(const std::string fileName, const std::vector<unsigned int> startRowList, const std::vector<unsigned int> endRowList);

	// return the sensing electrode trace in full resolution
	std::vector<float> RowTrace(const unsigned int row);

	// return the 4-row averaged sensing electrode trace in full resolution
	std::vector<float> RowTrace_avg4(const unsigned int row);

	// return the 4-row averaged sensing electrode trace in full resolution with trace offset
	// adjusted to zero by subtracting the average of the first numOffsetFrames points.
	std::vector<float> RowTrace_avg4(
			const unsigned int row,
			const unsigned int numOffsetFrames);

	// return the 4-row averaged sensing electrode trace in time compressed resolution with trace offset
	// adjusted to zero by subtracting the average of the first numOffsetFrames points.
	// the frames are compressed according to the new time stamps, normalized by the rowsum frame interval.
	std::vector<float> RowTrace_avg4(
			const unsigned int row,
			const unsigned int numOffsetFrames,
			const std::vector<unsigned int> newTimeStamps);

	// returns total number of rows
	unsigned int NumRows();

	// return frame rate in msec
	float FrameRate();

	// return time stamp.
	// normalized = true: returns time stamps normalized by frame intervals.
	// normalized = false: returns time stamps in msec.
	std::vector<unsigned int> TimeStamps(bool normalized);



	unsigned int Row2avg4row(const unsigned int wholeChipRow); // convert row in whole chip coordinate to the corresponding row in senseTrace_avg4

	void Row2RowsumRows(const unsigned int row, std::vector<unsigned int> & rowsumRows);


	std::vector<float> CompressFrames(
			const std::vector<float> & rowTrace,
			std::vector<float> & rowTrace_vfc,
			const std::vector<unsigned int> & timeStampOrig,
			const std::vector<unsigned int> & timeStampNew);

	// Return if the 4-row-averaged sensing electrode trace is valid.  row is whole chip row.
	bool isValidAvg4(unsigned int row);




private:

	// compute sensing electrode traces for in (row,frame) format.
	void ComputeSenseTrace();

	// filter the trace with median filter.
	void medianFilterTrace(const unsigned int order);

	// calculate median
	float median(const float * v1, const unsigned int order);

	// compute sense electrode traces, averaged over four rows that are acquired simultaneously.
	void ComputeSenseTrace_avg4();

	// adjust the offset of trace to zero by zeroing the first numOffsetFrames frames.
	void ZeroTrace(std::vector<float> & trace, const unsigned int numOffsetFrames);

	std::vector<uint32_t> rowsum;      // row sum raw data from file
	std::vector<std::vector<float> > senseTrace; // rowsum data averaged by pixelsInSum. (Only the needed rows are populated)
	std::vector<std::vector<float> > senseTrace_avg4; // 4 row averaged sensing electrode data

	std::vector< bool > isValid_avg4;  // whether the 4-row-averaged sensing electrode data is valid

	float baseFrameRate;  // in msec defined the same rate as in raw struct.  (Need to be float.  int does not have enough precision when calculating frame numbers.)
	uint32_t pixelsInSum; // number of pixels summed in each rowsum data
	uint32_t sumsPerRow; // number of rowsum values per row
	uint32_t sskey;  // The first field in the header of a rowsum file
	uint32_t version;  // version number of rowsum data file
	uint32_t numRows;  // total number of rows of the chip
	uint32_t numFrames; // number of frames in the rowsum data

	std::vector<unsigned int> startRowList; // first row of isfet data in whole chip coordinate
	std::vector<unsigned int> endRowList;  // last + 1 row of isfet data in whole chip coordinate

	std::string fileName;  // file name of the rowsum file
	std::vector<bool> isValid; // indicate if a rowsum trace is valid.
	std::vector<bool> isNeeded; // if a row is needed from the data.  If isNeeded rows will go through further processing

	uint32_t lowPin, highPin;  // low and high values for pinned rowsum
	unsigned int numPinnedLow, numPinnedHigh; // number of low and high pin values in the rows needed.

	bool DEBUG;
	bool topHalfOnly;

};


#endif // ROWSUMDATA_H
