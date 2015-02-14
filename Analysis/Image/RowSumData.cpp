/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * RowSumData.cpp
 *
 *  Created on: Jan 21, 2014
 *      Author: ionadmin
 */

#include "RowSumData.h"

RowSumData::RowSumData() {

	// default parameters
	lowPin = 0;
	highPin = 32760;
	numPinnedLow = 0;
	numPinnedHigh = 0;

	DEBUG = true;
	topHalfOnly = false;

//	float y[] = {1.1, 0.3, 5.2, 4.2, 2.5, 0.7};
//	std::vector< float > x(y, y+ sizeof(y)/sizeof(float));
//	printf("%5.2f %5.2f %5.2f %5.2f %5.2f %5.2f medians are: \n", y[0], y[1], y[2], y[3], y[4], y[5]);
//	fflush(stdout);
//	printf("median(0, 3) = %5.2f\n", median(&x[0], 3));
//	printf("median(0, 5) = %5.2f\n", median(&x[0], 5));
//
//	printf("median(1, 3) = %5.2f\n", median(&x[1], 3));
//	printf("median(1, 5) = %5.2f\n", median(&x[1], 5));
//
//	fflush(stdout);

}

RowSumData::~RowSumData() {
	// TODO Auto-generated destructor stub
}

int RowSumData::LoadRowSumData(const std::string fileName, const std::vector<unsigned int> startRowList, const std::vector<unsigned int> endRowList){




	// save a copy
	this->fileName = fileName;
	this->startRowList = startRowList;
	this->endRowList = endRowList;




	// open file
	std::ifstream rowSumFile;
	rowSumFile.open(fileName.c_str(), std::ifstream::in | std::ifstream::binary);


	if (rowSumFile){
		// opened successfully

		// read version number in header
		rowSumFile.read((char*) &sskey, 4);
		rowSumFile.read((char*) &version, 4);

		// printf("DEBUG sskey = %u, version = %u\n", sskey, version);

		if (version == (uint32_t) 1){
			// read version 1 header
			uint32_t buffer[5];
			rowSumFile.read((char*) buffer, sizeof(uint32_t)*5);

			numRows = buffer[0];
			numFrames = buffer[1];

			baseFrameRate = ((1./(float) buffer[2])*1000.);  // convert frame/sec to msec/frame

			pixelsInSum = buffer[3];
			sumsPerRow = buffer[4];
			// printf("DEBUG rows %u, numFrames %u, frameRate %10.5f, pixelsInSum %u, sumsPerRow %u\n", numRows, numFrames, baseFrameRate, pixelsInSum, sumsPerRow);

		} else {
			baseFrameRate = ((1./15.)*1000.);
			pixelsInSum = 1288*12;
			sumsPerRow = 1;
		}



		if (version == (uint32_t) 0){
			uint32_t buffer[2];
			rowSumFile.read((char*) buffer, sizeof(uint32_t)*2);
			numRows = buffer[0];
			numFrames = buffer[1];
		}

		int numDataPoints =numRows * numFrames * sumsPerRow; // total number of rowsum data points


		rowsum.resize(numDataPoints); //allocate memory
		rowSumFile.read((char*) &rowsum[0], sizeof(uint32_t)*numDataPoints); //read all rowsum data

		printf("DEBUG rowsum data: %u %u %u %u ...\n", rowsum[0], rowsum[1], rowsum[2], rowsum[3]);

		rowSumFile.close();

		// override some parameter values as the ones in file are not correct
		pixelsInSum = 8;

		// preset all rows are valid
		isValid.clear();
		isValid.resize(numRows, true);


		// update needed - only do computation on needed rows.
		isNeeded.clear();
		isNeeded.resize(numRows, false);
		std::vector<unsigned int> rowsNeeded;



		for (unsigned int i = 0; i < startRowList.size(); i++){
			for (unsigned int r =startRowList[i]; r < endRowList[i]; r++){
				Row2RowsumRows(r, rowsNeeded);
				for (unsigned int rr = 0; rr < rowsNeeded.size(); rr++){
					isNeeded[rowsNeeded[rr]] = true;

				}
			}
		}



		//fflush(stdout);



		if (rowSumFile.fail()){
			// error reading or closing file
			return 2;
		} else {
			// everything went well
			ComputeSenseTrace();  // calculate sensing electrode trace from rowsum data and mark pinned not valid
			//medianFilterTrace(5); // median filter traces
			ComputeSenseTrace_avg4(); // compute 4-row average sensing electrode trace.
			return 0;
		}


	} else {
		// error opening file
		return 1;
	}




}


void RowSumData::Row2RowsumRows(const unsigned int row, std::vector<unsigned int> & rowsNeeded){
	unsigned int r = row;

	// make sure it is in the first half of total number of rows
	if (r > numRows/2-1){
		r = numRows - 1 - r; // flip to first half row equivalent.
	}

	// make sure the starting row is right
	if (r % 2 == 1){
		// odd
		r -= 1;
	}

	if (topHalfOnly){
		rowsNeeded.resize(2);
		rowsNeeded[0] = numRows - 1 - r;
		rowsNeeded[1] = numRows - 1 - (r+1);


	} else {

	rowsNeeded.resize(4);
	rowsNeeded[0] = r;
	rowsNeeded[1] = r+1;
	rowsNeeded[2] = numRows - 1 - r;
	rowsNeeded[3] = numRows - 1 - (r+1);
	}



}

unsigned int RowSumData::Row2avg4row(const unsigned int wholeChipRow){

	unsigned int r = wholeChipRow;
	if (r > numRows/2-1){
		r = numRows - 1 - r; // flip to first half row equivalent.
	}
	return (r/2);  // 0 or 1 -> 0; 2 or 3 -> 1; ....

}

std::vector<float> RowSumData::RowTrace(const unsigned int row){
	return senseTrace[row];
}

std::vector<float> RowSumData::RowTrace_avg4(const unsigned int row){


	// convert whole chip row to 4-row averaged row r
	return senseTrace_avg4[Row2avg4row(row)];

}

std::vector<float> RowSumData::RowTrace_avg4(
		const unsigned int row,
		const unsigned int numOffsetFrames){

	// calculate trace offset by averaging a number of initial data points
	std::vector<float> rt(RowTrace_avg4(row));

	ZeroTrace(rt, numOffsetFrames);

	return rt;

}


std::vector<float> RowSumData::RowTrace_avg4(
		const unsigned int row,
		const unsigned int numOffsetFrames,
		const std::vector<unsigned int> newTimeStamps){

	if (! isNeeded[row]){
		printf("Warning: RowSumData row %u requested but not marked as needed!\n", row);
	}

	std::vector<float> & rowTrace = senseTrace_avg4[Row2avg4row(row)];

	std::vector<unsigned int> rowsumTimeStamps(TimeStamps(true));

	// do variable frame compression
	std::vector <float> rowTrace_vfc;
	CompressFrames(rowTrace, rowTrace_vfc, rowsumTimeStamps, newTimeStamps);

	// zero the trace
	ZeroTrace(rowTrace_vfc, numOffsetFrames);




	return rowTrace_vfc;

}

std::vector<float> RowSumData::CompressFrames(
		const std::vector<float> & rowTrace,
		std::vector<float> & rowTrace_vfc,
		const std::vector<unsigned int> & timeStampOrig,
		const std::vector<unsigned int> & timeStampNew){

	rowTrace_vfc.resize(timeStampNew.size(), 0.);
	for (unsigned int f = 0; f < rowTrace_vfc.size(); f++){
		rowTrace_vfc[f] = 0.f;
	}

	unsigned int i = 0;
	unsigned int cnt = 0;
	for (unsigned int f = 0; f < timeStampOrig.size(); f++){
		rowTrace_vfc[i] +=  rowTrace[f];
		cnt ++;

		//printf("In CompressFrames, timeStampNew[%d] %d, timeStampOrig[%d] %d rowTrace[f] %10.5f rowTrace_vfc[f] %10.5f\n", i, timeStampNew[i], f, timeStampOrig[f], rowTrace[f], rowTrace_vfc[i]);

		if (timeStampNew[i] == timeStampOrig[f]){
			rowTrace_vfc[i] /= (float) cnt;
 //			printf("In CompressFrames loop, rowTrace_vfc[i] %10.5f\n", rowTrace_vfc[i]);
			cnt = 0;
			i ++;

			if (i == timeStampNew.size())
				break;
		}



	}

	return rowTrace_vfc;

}

void RowSumData::ZeroTrace(std::vector<float> & trace, const unsigned int numOffsetFrames){

	if (numOffsetFrames > trace.size()){
		if (DEBUG){
			printf("Warning RowSumDataZeroTrace number of offset frames larger than trace size.");
		}
		return;
	}

	if (numOffsetFrames > 0){
		float offset = 0.;
		for (unsigned int f = 0; f < numOffsetFrames; f++){
			offset += trace[f];
		}
		offset /= (float) numOffsetFrames;

		// subtract offset from trace
		for (unsigned int f = 0; f < trace.size(); f++)
			trace[f] -= offset;
	}

}

void RowSumData::ComputeSenseTrace(){

	//printf("In computeSenseTrace\n"); fflush(stdout);

	// allocate memory
	//senseTrace.resize(numRows, std::vector<float>(numFrames, 0.));
	senseTrace.resize(numRows);
	for (unsigned int row = 0; row < senseTrace.size(); row++){
		if (isNeeded[row]){
			senseTrace[row] = std::vector<float>(numFrames);
		}
	}

	// parse the left FPGA readings (every other elements in rowsum)
	// and rearrange them in (row, frame) format
	unsigned int row = 0;
	unsigned int frame = 0;
	uint32_t val;
	for (unsigned int i = 0; i < rowsum.size(); i+=2){
		// only parse the needed row
		if (isNeeded[row]){
			// check if there's any pinned values.
			val = rowsum[i];
			if (val <= lowPin){
				numPinnedLow ++;
				isValid[row] = false;
			} else if (val >= highPin){
				numPinnedHigh ++;
				isValid[row] = false;
			} else {
				senseTrace[row][frame] = (float) val/ (float) pixelsInSum;
			}
		}
		row++;
		if (row == numRows){
			row = 0;
			frame ++;
		}
	}


	unsigned int numValid = 0;
	unsigned int numNeeded = 0;
	for (unsigned int r = 0; r < senseTrace.size(); r++){
		if (isNeeded[r]){
			numNeeded ++;
			if (isValid[r]){
				numValid ++;
			}


		}
	}

	printf("%s: %u out of %u needed rows are valid\n", fileName.c_str(), numValid, numNeeded);
	fflush(stdout);


}

void RowSumData::medianFilterTrace(const unsigned int order){
	// order must be odd

	//printf("In medianFilterTrace\n");
	//fflush(stdout);

	const unsigned int valEachSide = (order-1)/2;

	unsigned int cnt = 0;
	for (unsigned int row = 0; row < senseTrace.size(); row++){
		if (isNeeded[row] && isValid[row]){
			// printf("row %u needed\n", row);
			cnt ++;
			std::vector<float> newTrace(senseTrace[row]);
			for (unsigned int frame = valEachSide; frame < senseTrace[row].size() - valEachSide; frame ++){
				newTrace[frame] = median(&senseTrace[row][frame - valEachSide], order);
			}

//			if (row == 6400) {
//				for (unsigned int f = 0; f < senseTrace[row].size(); f++){
//					printf("%5.3f, %5.3f\n", senseTrace[row][f], newTrace[f]);
//				}

//			}

			senseTrace[row] = newTrace;

		}
	}

	if (DEBUG){
		printf("%s: number of trace median filtered is %u \n", fileName.c_str(), cnt);
		fflush(stdout);
	}

}

float RowSumData::median(const float * v1, const unsigned int order){

	// perform partial sort to find median
	// Find minimum (order-1)/2 + 1 times

	unsigned int num = (order - 1)/2 + 1;
	std::vector < bool > taken(order, false);
	int minIndex;
	for (unsigned int i = 0; i < num; i++){
		minIndex = -1;
		for (unsigned int j = 0; j < order; j++){

			// first element
			if (minIndex == -1 && ! taken[j]){
				minIndex = j;
				continue;
			}

			if (! taken[j] && *(v1 + j) < *(v1 + minIndex)  ){
				// new minimum
				minIndex = j;
			}
		}

		// should always be a minIndex
		taken[minIndex] = true;

	}


	return *(v1 + minIndex);

}

bool RowSumData::isValidAvg4(unsigned int row){
	return isValid_avg4[Row2avg4row(row)];
}

void RowSumData::ComputeSenseTrace_avg4(){
	// allocate memory

	senseTrace_avg4.clear();
	senseTrace_avg4.resize(numRows/4);
	isValid_avg4.clear();
	isValid_avg4.resize(numRows/4, false);

	//printf("numRows %d, emptyFloatVec empty? %d senseTrace_avg4[1000].empty()? %d\n", numRows, emptyFloatVec.empty(), senseTrace_avg4[1000].empty());


	// make sure senseTrace has been computed.
	if (senseTrace.empty()){
		printf("Error in ComputeSenseTrace_avg4:  senseTrace is empty."); fflush(stdout);
	}


	int cnt_rowConsidered = 0;
	int cnt_valid = 0;


	// average over rows that are acquired simultaneously.
	// that is rows (0, 1, last, last-1), (2,3, last-2, last-3), ...
	std::vector<unsigned int> rowsumRows;
	for (unsigned int i = 0; i < startRowList.size(); i++){
		for (unsigned int row = startRowList[i]; row < endRowList[i]; row++){

			unsigned int avg4Row = Row2avg4row(row);

	//		printf("row %d, avg4Row %d, before empty check senseTrace_avg4[avg4Row].empty() %d\n", row, avg4Row, senseTrace_avg4[avg4Row].empty());
			if (! senseTrace_avg4[avg4Row].empty())
				continue;  // already calculated

			cnt_rowConsidered ++;
			//		printf("computeSenseTrace_avg4 row %u \n", row); fflush(stdout);

			// allocate memory, preset to 0
			senseTrace_avg4[avg4Row] = std::vector<float>(numFrames, 0.);

			Row2RowsumRows(row, rowsumRows);

			int norm = 0;
			for (unsigned int i = 0; i < rowsumRows.size(); i++){
				const unsigned int rsr = rowsumRows[i];
				if (isValid[rsr]){
					norm ++;
					for (unsigned int frame = 0; frame < numFrames; frame++){
						senseTrace_avg4[avg4Row][frame] += senseTrace[rsr][frame];
					}

				}
			}
			if (norm > 0) {
				for (unsigned int frame = 0; frame < numFrames; frame++){
					senseTrace_avg4[avg4Row][frame] /= (float) norm;
				}
				isValid_avg4[avg4Row] = true;
				cnt_valid++;
			} else {
				isValid_avg4[avg4Row] = false;
			}
		}
	}



	if (DEBUG){
		printf("%s: ComputeSenseTrace_avg4: %u rows try to compute, %u rows valid.\n", fileName.c_str(), cnt_rowConsidered, cnt_valid);
		fflush(stdout);
	}

}


std::vector<unsigned int> RowSumData::TimeStamps(bool normalized){
	std::vector<unsigned int> ts(numFrames, 0);
	for (unsigned int f = 0; f < ts.size(); f++){
		ts[f] = (unsigned int) ( (float) (f+1) * (normalized? 1. : baseFrameRate ) );
	}
	return ts;

}

unsigned int RowSumData::NumRows(){
	return numRows;
}

float RowSumData::FrameRate(){
	return baseFrameRate;
}



