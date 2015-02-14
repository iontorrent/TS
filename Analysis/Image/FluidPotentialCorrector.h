/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * FluidPotentialCorrector.h
 *
 *  Created on: Jan 20, 2014
 *      Author: awong
 */

#ifndef FLUIDPOTENTIALCORRECTOR_H
#define FLUIDPOTENTIALCORRECTOR_H

#include <string>
#include <vector>
#include <map>
#include "RowSumData.h"
#include "Mask.h"
#include "RawImage.h"
#include <cmath>
#include <armadillo>


class FluidPotentialCorrector {
public:

//	// default constructor
//	FluidPotentialCorrector();
//
//	// constructor to set default scale factor
//	FluidPotentialCorrector(const double scaleFactor);

	// constructor to load file and set default scale factor
	FluidPotentialCorrector(
			RawImage *raw,
			Mask *mask,
			const std::string fileName,
			const double scaleFactor,
			const unsigned int startRow,
			const unsigned int endRow,
			const double noiseThresh);

	// constructor for initializing the object first and load files later
	FluidPotentialCorrector(const double noiseThresh);

	virtual ~FluidPotentialCorrector();

	// load rowsum sensing electrode raw data
	int loadSensingElectrodeData(const std::string fileName,
			unsigned int startRow,
			unsigned int endRow);

	// load rowsum sensing electrode raw data
	int loadSensingElectrodeDataThumbnail(const std::string fileName,
			const unsigned int numRows);

	// set flag for signaling we are dealing with a thumbnail dataset
	void setIsThumbnail();

	// set region size
	void setImage(RawImage *raw,
			Mask *mask,
			const unsigned int regionSizeRow,
			const unsigned int regionSizeCol,
			const char nucChar);

	// perform fluid potential correction on whole image with region size (regionXSize, regionYSize)
	void doCorrection();

	void correctWithLastGoodFlow();

	void saveAverageFlowTraces();

	// perform fluid potential correction on wells within [rowStart, rowEnd) and [colStart, colEnd)
	void applyScaleFactor(
			const unsigned int rowStart,
			const unsigned int rowEnd,
			const unsigned int colStart,
			const unsigned int colEnd,
			const float scaleFactor,
			const std::vector<double> & senseTrace
			);

	//
	bool readRowSumIsSuccess();
	void setThreshold(const double threshold);


private:

	RawImage *raw;
	Mask *mask;
	char nucChar;


	std::string rowsumFileName;
	int readRowSumReturnValue;
	double noiseThresh;
	double scaleFactorDefault;
	bool scaleFactorCorrection;
	bool lastGoodFlowCorrection;
	bool useDefaultScaleFactor;
	bool correctSenseDrift;
	unsigned int numThumbnailRows;
	unsigned int numThumbnailCols;
	unsigned int numThumbnailRegionsRow;
	unsigned int numThumbnailRegionsCol;


	RowSumData rowsum;
	bool isThumbnail;
	unsigned int numRowsWholeChip;
	unsigned int regionSizeRow, regionSizeCol;
	std::vector< std::vector<float> > senseTrace;
	std::vector<bool> senseTraceIsValid;
	//std::vector< std::vector< std::vector<double> > > senseTraceAvgRegion;

	std::map< char, std::vector< std::vector < std::vector < double > > > > lastGoodAvgTraces;

	unsigned int ThumbnailRow2wholeChipRow(const unsigned int row);

	float findScaleFactor_fitSG(
			const float scaleFactorMin,
			const float scaleFactorMax,
			const float scaleFactorBin,
			const unsigned int rowStart,
			const unsigned int rowEnd,
			const unsigned int colStart,
			const unsigned int colEnd,
			std::vector<double> & senseTraceAvg
			);


	// adjust the offset of trace to zero by zeroing the first numOffsetFrames frames.
	void ZeroTrace(std::vector<double> & trace, const unsigned int numOffsetFrames);

	// Correct the drift of a trace by minimizing the least square of the first numPointsStart data points in the beginning and the last numPointsEnd data points at the end.
	void CorrectTraceDrift(std::vector<double> & trace, const unsigned int numPointsStart, const unsigned int numPointsEnd);


	bool rowsumNoiseTooSmall();

	// convert a single trace first to vfc according to the block, then to uniform
	void convertTraceUniformToThumbnail(
			std::vector<double> & trace,
			const unsigned int tn_region_row,
			const unsigned int tn_region_col);


	std::vector<unsigned int> timeStampThumbnailRegionVFC(const unsigned int row, const unsigned int col);

	std::vector<double> uniform2vfc(
			const std::vector<double> & trace,
			const std::vector<unsigned int> & timeStamp,
			const std::vector<unsigned int> & timeStamp_vfc);

	std::vector<double> vfc2thumbnail(
			const std::vector<double> trace_vfc,
			const std::vector<unsigned int> timeStamp_vfc,
			const std::vector<unsigned int> timeStamps);

	bool DEBUG;

	std::vector<unsigned int> timeStamps;



};

#endif // FLUIDPOTENTIALCORRECTOR_H
