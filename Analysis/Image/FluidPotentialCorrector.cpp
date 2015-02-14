/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * FluidPotentialCorrector.cpp
 *
 *  Created on: Jan 20, 2014
 *      Author: awong
 */

#include "FluidPotentialCorrector.h"

//FluidPotentialCorrector::FluidPotentialCorrector() {
//	scaleFactorDefault = 0.;
//	isThumbnail = false;
//	numRowsWholeChip = 10656; // P1
//	blockSize = 100;
//
//}
//
//FluidPotentialCorrector::FluidPotentialCorrector(const double scaleFactor){
//	scaleFactorDefault = scaleFactor;
//	isThumbnail = false;
//	numRowsWholeChip = 10656;  //P1
//	blockSize = 100;
//}

FluidPotentialCorrector::FluidPotentialCorrector(
		RawImage *raw,
		Mask *mask,
		const std::string fileName,
		const double scaleFactor,
		const unsigned int startRow,
		const unsigned int endRow,
		const double noiseThresh){
	const std::vector<unsigned int> startRowVector(1, startRow);
	const std::vector<unsigned int> endRowVector(1, endRow);

	readRowSumReturnValue = rowsum.LoadRowSumData(fileName, startRowVector, endRowVector);



	scaleFactorDefault = scaleFactor;
	isThumbnail = false;
	numRowsWholeChip = 10656; // P1
	regionSizeRow = 100;
	regionSizeCol = 100;
	this->raw = raw;
	this->mask = mask;
	rowsumFileName = fileName;


	DEBUG = 1;
	this->noiseThresh = noiseThresh;

	scaleFactorCorrection = true;
	lastGoodFlowCorrection = false;
	useDefaultScaleFactor = false;
	correctSenseDrift = false;

	numThumbnailRows = 100;
	numThumbnailCols = 100;

	numThumbnailRegionsRow = 8;
	numThumbnailRegionsCol = 12;

//	std::vector<float> tr(rowsum.rowTrace_avg4(rowsum.getNumRows() - 1));
//	for (unsigned int f = 0; f < tr.size(); f++){
//		printf("%10.4f, ", tr[f]);
//	}
//	printf("\n");

}

FluidPotentialCorrector::FluidPotentialCorrector(const double noiseThresh){

	isThumbnail = false;
	numRowsWholeChip = 10656; // P1
	regionSizeRow = 100;
	regionSizeCol = 100;
	DEBUG = 1;
	this->noiseThresh = noiseThresh;

	readRowSumReturnValue = 0;
	scaleFactorDefault = 1.;

	scaleFactorCorrection = true;
	lastGoodFlowCorrection = false;
	useDefaultScaleFactor = false;
	correctSenseDrift = false;

	numThumbnailRows = 100;
	numThumbnailCols = 100;

	numThumbnailRegionsRow = 8;
	numThumbnailRegionsCol = 12;

}


void FluidPotentialCorrector::setIsThumbnail(){
	isThumbnail = true;

}

void FluidPotentialCorrector::setImage(RawImage *raw,
		Mask *mask,
		const unsigned int regionSizeRow,
		const unsigned int regionSizeCol,
		const char nucChar){
	this->raw = raw;
	this->mask = mask;
	this->regionSizeRow = regionSizeRow;
	this->regionSizeCol = regionSizeCol;
	this->nucChar = nucChar;

}

bool FluidPotentialCorrector::readRowSumIsSuccess(){
	return !((bool) readRowSumReturnValue );

}

FluidPotentialCorrector::~FluidPotentialCorrector() {
	// TODO Auto-generated destructor stub
}


int FluidPotentialCorrector::loadSensingElectrodeData(
		const std::string fileName,
		unsigned int startRow,
		unsigned int endRow){
	std::vector<unsigned int> startRowVector(1, startRow);
	std::vector<unsigned int> endRowVector(1, endRow);
	readRowSumReturnValue = rowsum.LoadRowSumData(fileName, startRowVector, endRowVector);
	return readRowSumReturnValue;
}

int FluidPotentialCorrector::loadSensingElectrodeDataThumbnail(
		const std::string fileName,
		const unsigned int numRows){

	std::vector<unsigned int> startRowVector;
	std::vector<unsigned int> endRowVector;

	// parse region size
	for (unsigned int r = 0; r<numRows; r+=regionSizeRow){
		unsigned int rwhole = ThumbnailRow2wholeChipRow(r);
		startRowVector.push_back(rwhole);
		endRowVector.push_back(rwhole + regionSizeRow);
		if ( numRows - r  < 2*regionSizeRow ){
			// for avoiding last region's size smaller than regionSizeRow
			*(endRowVector.end()-1) =  ThumbnailRow2wholeChipRow(numRows-1)+1;
			break;
		}
	}

//	for (unsigned int i = 0; i < startRowVector.size(); i++){
//		printf("startrow %d endrow %d\n", startRowVector[i], endRowVector[i]);
//	}
	readRowSumReturnValue = rowsum.LoadRowSumData(fileName, startRowVector, endRowVector);
	return readRowSumReturnValue;
}


unsigned int FluidPotentialCorrector::ThumbnailRow2wholeChipRow(const unsigned int row){
	if (isThumbnail){
		// convert from thumbnail to whole chip row
		const int thumbnailSize = 100;
		const int blockNum = row/thumbnailSize;  // integer division gives the block number in row
		const int rowInBlock = row % thumbnailSize;
		const int blockSizeRow = numRowsWholeChip/8;  // for P1 only
		const int blockSizeCol = numRowsWholeChip/12;
		const int blockOffset = blockSizeRow/2 - thumbnailSize/2;

		//printf("thumbnailSize %u blockNum %u rowInBlock %u blockSize %u blockOffset %u\n", thumbnailSize, blockNum, rowInBlock, blockSize, blockOffset);
		return blockSizeRow*blockNum + rowInBlock + blockOffset;


	} else {
		return row;
	}


}

void FluidPotentialCorrector::doCorrection(){

	//	printf("0 tranformRow %u\n", (uint32_t) transformRow(0));
	//	printf("99 tranformRow %u\n", (uint32_t) transformRow(99));
	//	printf("100 tranformRow %u\n", (uint32_t) transformRow(100));
	//	printf("256 tranformRow %u\n", (uint32_t) transformRow(256));
	//	printf("regionXSize %d regionYSize %d\n", regionXSize, regionYSize);


	//printf("In doCorrection, beginning\n"); fflush(stdout);

	short int *image = raw->image;
	const unsigned int rows = raw->rows;
	const unsigned int cols = raw->cols;
	const unsigned int frames = raw->frames;

	//printf("numRows %u numCols %u numFrames %u row_offset %d\n", rows, cols, frames, raw->chip_offset_y);

//	// check if region size is valid
//	if (! isThumbnail){
//		// default region size for block data
////		regionSizeCol = 100;
////		regionSizeRow = 100;
//	} else {
//
//		if( cols %regionSizeCol != 0 || rows%regionSizeRow != 0){
//			//skip correction
//			fprintf (stdout, "FluidPotentialCorrector thumbnail image skipped: Region sizes are not compatible with image(%d x %d): %d x %d\n", rows, cols, regionSizeRow, regionSizeCol);
//			return;
//		}
//
//	}

	const unsigned int frameStride = rows * cols;


	// Get time stamps for raw image
	timeStamps.assign(raw->timestamps, raw->timestamps + frames);
	for (unsigned int f = 0; f < timeStamps.size(); f++){
		timeStamps[f] = (int) roundf((float) timeStamps[f]/ rowsum.FrameRate() );
	//	printf("timeStamps[%d] %d", f, timeStamps[f]);
	}

	//printf("In doCorrection, after timeStamps\n"); fflush(stdout);
	// populate rowsum sensing electrode traces
	const unsigned int row_offset = raw->chip_offset_y;
	senseTrace.resize(rows);
	senseTraceIsValid.resize(rows);
		//printf("In doCorrection, r %d rows %d\n", r, rows); fflush(stdout);
	if (isThumbnail){
		for (unsigned int r = 0; r < rows; r++){
			unsigned int rwhole = ThumbnailRow2wholeChipRow(r);
			senseTrace[r] =  rowsum.RowTrace_avg4(rwhole , 5, timeStamps );
			senseTraceIsValid[r] = rowsum.isValidAvg4(rwhole);
		}
	} else {
		for (unsigned int r = 0; r < rows; r++){
			senseTrace[r] =  rowsum.RowTrace_avg4( row_offset + r, 5, timeStamps );
			senseTraceIsValid[r] = rowsum.isValidAvg4(row_offset+ r);
		}

	}

//	for (unsigned int f = 0; f < frames; f++){
//		printf("senseTrace[50] %10.5f\n", senseTrace[50][f]);
//	}

	if (rowsumNoiseTooSmall()){
		printf("Fluid potential corrector:  sensing electrode noise too small (< %5.2f).  Correction skipped.\n", noiseThresh);
		if (lastGoodFlowCorrection){
			saveAverageFlowTraces();
		}
		return;
	}

	if (scaleFactorCorrection){
		// go through all regions and perform correction
		//double scaleFactor = scaleFactorDefault;
		const unsigned int numColBlocks = cols / regionSizeCol;
		const unsigned int numRowBlocks = rows / regionSizeRow;

		std::vector< std::vector<float> > scaleFactor(numRowBlocks);
		std::vector< std::vector< std::vector<double> > > senseTraceAvgRegion(numRowBlocks);

		// find scale factor for each region
		const float scaleFactorMin = 0.;
		const float scaleFactorMax = 10.;
		const float scaleFactorBin = 0.2;
		for(unsigned int rb = 0; rb < numRowBlocks; rb++){
			scaleFactor[rb].resize(numColBlocks);
			senseTraceAvgRegion[rb].resize(numColBlocks);

			for(unsigned int cb = 0; cb < numColBlocks; cb++){
				unsigned int rowStart = rb * regionSizeRow;
				unsigned int rowEnd = (rb + 1) * regionSizeRow;
				if (rb == regionSizeRow - 1) rowEnd = rows;

				unsigned int colStart = cb * regionSizeCol;
				unsigned int colEnd = (cb + 1) * regionSizeCol;
				if (cb == regionSizeCol - 1) colEnd = cols;

				//            if ( (rb==7) && (cb == 3)){
				//            	DEBUG = true;
				//            }


				if (useDefaultScaleFactor){
					scaleFactor[rb][cb] = scaleFactorDefault;
				} else {
					scaleFactor[rb][cb] = findScaleFactor_fitSG(scaleFactorMin, scaleFactorMax, scaleFactorBin, rowStart, rowEnd, colStart, colEnd, senseTraceAvgRegion[rb][cb]);
				}
				//printf("block r=%u, c=%u: scale factor = %10.3f\n", rb, cb, scaleFactor[rb][cb]);
				//			DEBUG = false;

			}
		}

		if (DEBUG){
			printf("scaleFactor %s\n", rowsumFileName.c_str());
			for (unsigned int r = 0; r < scaleFactor.size(); r++){
				for (unsigned int c = 0; c < scaleFactor[r].size(); c++){
					printf("%7.3f, ", scaleFactor[r][c]);
				}
				printf("\n");
				fflush(stdout);
			}
		}



		for(unsigned int rb = 0; rb < numRowBlocks; rb++){
			for(unsigned int cb = 0; cb < numColBlocks; cb++){

				//			// go through all pixels in region and collect good traces
				//			std::vector< std::vector<float> > traces;

				unsigned int rowStart = rb * regionSizeRow;
				unsigned int rowEnd = (rb + 1) * regionSizeRow;
				if (rb == regionSizeRow - 1) rowEnd = rows;

				unsigned int colStart = cb * regionSizeCol;
				unsigned int colEnd = (cb + 1) * regionSizeCol;
				if (cb == regionSizeCol - 1) colEnd = cols;

				applyScaleFactor(rowStart, rowEnd, colStart, colEnd, scaleFactor[rb][cb], senseTraceAvgRegion[rb][cb]);
			}
		}

	}

	if (lastGoodFlowCorrection){
		correctWithLastGoodFlow();
	}


}


void FluidPotentialCorrector:: saveAverageFlowTraces(){

	printf("In saveAverageFlowTraces\n");fflush(stdout);

	short int *image = raw->image;
	const unsigned int rows = raw->rows;
	const unsigned int cols = raw->cols;
	const unsigned int frames = raw->frames;
	const unsigned int frameStride = rows * cols;
	const unsigned int numColBlocks = cols / regionSizeCol;
	const unsigned int numRowBlocks = rows / regionSizeRow;

	// initialize last good average trace
	if (lastGoodAvgTraces[nucChar].empty()){
		lastGoodAvgTraces[nucChar].resize(numRowBlocks);
		for(unsigned int rb = 0; rb < numRowBlocks; rb++){
			lastGoodAvgTraces[nucChar][rb].resize(numColBlocks);
		}
	}





	unsigned int ind;
	std::vector <double> avgTrace(frames);
	unsigned int numSavedAvgTraces = 0;
	for(unsigned int rb = 0; rb < numRowBlocks; rb++){

		for(unsigned int cb = 0; cb < numColBlocks; cb++){
			unsigned int rowStart = rb * regionSizeRow;
			unsigned int rowEnd = (rb + 1) * regionSizeRow;
			if (rb == regionSizeRow - 1) rowEnd = rows;

			unsigned int colStart = cb * regionSizeCol;
			unsigned int colEnd = (cb + 1) * regionSizeCol;
			if (cb == regionSizeCol - 1) colEnd = cols;



			//printf("rb %d, cb %d\n", rb, cb);

			// initialize avgTrace
			for (unsigned int f = 0; f < avgTrace.size(); f++)
				avgTrace[f] = 0.;
			unsigned int count = 0;


			// Find average trace over the region
			for (unsigned int r = rowStart; r < rowEnd; r++){
				for (unsigned int c = colStart; c < colEnd; c++){
					ind =  c +  r*cols;
					//printf("r %d c %d cols %d ind %d\n", r, c, cols, ind); fflush(stdout);
					if ( (( *mask ) [ind] & MaskPinned)==0 ){
						// well (c, r) is not pinned
						for (unsigned int f = 0; f <frames; f++){
							avgTrace[f] += (double) image[ind + f*frameStride];
							count ++;
						}
					}
				}
			}

			if (count > 0){
				for (unsigned int f = 0; f <frames; f++)
					avgTrace[f] /= (double) count;
				lastGoodAvgTraces[nucChar][rb][cb] = avgTrace;
				numSavedAvgTraces ++;

			} else {

			}




		}
	}


	printf("%c %d avg trace saved.\n", nucChar, numSavedAvgTraces);

}




void FluidPotentialCorrector:: correctWithLastGoodFlow(){


	short int *image = raw->image;
	const unsigned int rows = raw->rows;
	const unsigned int cols = raw->cols;
	const unsigned int frames = raw->frames;
	const unsigned int frameStride = rows * cols;
	const unsigned int numColBlocks = cols / regionSizeCol;
	const unsigned int numRowBlocks = rows / regionSizeRow;


	if (lastGoodAvgTraces[nucChar].empty()){
		printf("FluidPotentialCorrector: cannot correct with last good flow: not exist.\n");
		return;
	}

	unsigned int ind;
	std::vector <double> avgTrace(frames);
	unsigned int numSavedAvgTraces = 0;
	for(unsigned int rb = 0; rb < numRowBlocks; rb++){

		for(unsigned int cb = 0; cb < numColBlocks; cb++){
			unsigned int rowStart = rb * regionSizeRow;
			unsigned int rowEnd = (rb + 1) * regionSizeRow;
			if (rb == regionSizeRow - 1) rowEnd = rows;

			unsigned int colStart = cb * regionSizeCol;
			unsigned int colEnd = (cb + 1) * regionSizeCol;
			if (cb == regionSizeCol - 1) colEnd = cols;

			// initialize avgTrace
			for (unsigned int f = 0; f < avgTrace.size(); f++)
				avgTrace[f] = 0.;
			unsigned int count = 0;

			// Find average trace over the region
			for (unsigned int r = rowStart; r < rowEnd; r++){
				for (unsigned int c = colStart; c < colEnd; c++){
					ind =  c +  r*cols;
					//printf("r %d c %d cols %d ind %d\n", r, c, cols, ind); fflush(stdout);
					if ( (( *mask ) [ind] & MaskPinned)==0 ){
						// well (c, r) is not pinned
						for (unsigned int f = 0; f <frames; f++){
							avgTrace[f] += (double) image[ind + f*frameStride];
							count ++;
						}
					}
				}
			}
			if (count > 0){
				// find average trace

				for (unsigned int f = 0; f <frames; f++){
					avgTrace[f] /= (double) count;
					if (rb == 5 && cb ==5){
						printf("%10.5f, %10.5f\n", avgTrace[f], lastGoodAvgTraces[nucChar][rb][cb][f]);
					}
				}


				// apply correction

				for (unsigned int r = rowStart; r < rowEnd; r++){
					for (unsigned int c = colStart; c < colEnd; c++){
						ind =  c +  r*cols;
						if ( (( *mask ) [ind] & MaskPinned)==0 ){
							// well (c, r) is not pinned
							for (unsigned int f = 0; f <frames; f++){
								image[ind + f*frameStride] = (short int) ( (double) image[ind + f*frameStride] - avgTrace[f] + lastGoodAvgTraces[nucChar][rb][cb][f]);							}
						}
					}
				}














			} else {

			}




		}
	}


	printf("%c %d avg trace saved.\n", nucChar, numSavedAvgTraces);

}



bool FluidPotentialCorrector::rowsumNoiseTooSmall(){

	const unsigned int frames = raw->frames;
	const unsigned int rows = raw->rows;
	const unsigned int cols = raw->cols;

	std::vector<double> senseTraceAvg(frames), senseTraceAvgFit(frames);
	int count = 0;
	for (unsigned int r = 0; r < rows; r++){
		if (senseTraceIsValid[r]){
			for (unsigned int f = 0; f < senseTraceAvg.size(); f++){
				senseTraceAvg[f] += (double) senseTrace[r][f];  //senseTrace is the local, already avg4
			};
			count ++;
		}
	}

	//printf("senseTraceAvg\n");
	for (unsigned int f = 0; f < senseTraceAvg.size(); f++){
		senseTraceAvg[f] /= (double) count;
		//printf("%d, %10.5f\n", f, senseTraceAvg[f]);
	};

	const unsigned int sgOrder = 7;
	const double sgCoeff[sgOrder] = { -0.0952,    0.1429,    0.2857,    0.3333,    0.2857,    0.1429,   -0.0952};


	// filter with  Savitzky-Golay  (will be able to save a few steps by computing error directly)
	const int sgHalf = (sgOrder-1)/2;
	senseTraceAvgFit = senseTraceAvg;
	for (unsigned int f = sgHalf; f < frames -sgHalf; f++  ){
		double val = 0.;
		for (unsigned int i = 0; i < sgOrder; i++){
			val += sgCoeff[i]*senseTraceAvg[f - sgHalf + i];
		}
		senseTraceAvgFit[f] = val;
	}

	// calculate square error sum
	double noise = 0.;
	int countNoise = 0;
	for (unsigned int f = sgHalf; f < frames -sgHalf; f++  ){
		noise += (senseTraceAvgFit[f] -senseTraceAvg[f])*(senseTraceAvgFit[f] -senseTraceAvg[f]);
		countNoise ++;
	}
	noise = sqrt(noise/ (double)(countNoise-1) );

	if (DEBUG){
		printf("%s: Rowsum noise is %5.3f\n", rowsumFileName.c_str(), noise);
	}

	if (noise < noiseThresh)
		return true;
	else
		return false;


}

void FluidPotentialCorrector::applyScaleFactor(
		const unsigned int rowStart,
		const unsigned int rowEnd,
		const unsigned int colStart,
		const unsigned int colEnd,
		const float scaleFactor,
		const std::vector<double> & senseTrace
		){

	if (scaleFactor == 0.f)
		return;

	short int *image = raw->image;
	const unsigned int rows = raw->rows;
	const unsigned int cols = raw->cols;
	const unsigned int frames = raw->frames;
	const unsigned int frameStride = rows * cols;
	const unsigned int row_offset = raw->chip_offset_y;
	//float scaleFactor = scaleFactorDefault;

	// Get time stamps for raw image
	std::vector<unsigned int> timeStamps(raw->timestamps, raw->timestamps + frames);
	for (unsigned int f = 0; f < timeStamps.size(); f++){
		timeStamps[f] = (int) roundf((float) timeStamps[f]/ rowsum.FrameRate() );
	}

	//
	unsigned int ind;
	for (unsigned int r = rowStart; r < rowEnd; r++){

		for (unsigned int c = colStart; c < colEnd; c++){
			ind =  c +  r*cols  ;
			//printf("%u, %u\n", r, c);

			for (unsigned int f = 0; f <frames; f++){
				if (r == 1000 && c == 1000) {
					printf("%u, %10.4f, %10.4f, ", image[ind + f*frameStride], scaleFactor, senseTrace[f]);
				}
				//image[ind + f*frameStride] = (short int) ( (float) image[ind + f*frameStride] - scaleFactor * senseTrace[r][f]);
				image[ind + f*frameStride] = (short int) ( (float) image[ind + f*frameStride] - scaleFactor * senseTrace[f]);

				if (r == 1000 && c == 1000) {
					printf("%u\n", image[ind + f*frameStride]);
				}

			}

		}

	}




}

float FluidPotentialCorrector::findScaleFactor_fitSG(
		const float scaleFactorMin,
		const float scaleFactorMax,
		const float scaleFactorBin,
		const unsigned int rowStart,
		const unsigned int rowEnd,
		const unsigned int colStart,
		const unsigned int colEnd,
		std::vector<double> & senseTraceAvg
		){

	double scaleFactorOpt = 0.;
	const short int *image = raw->image;
	const unsigned int rows = raw->rows;
	const unsigned int cols = raw->cols;
	const unsigned int frames = raw->frames;
	const unsigned int frameStride = rows * cols;
	const unsigned int sgOrder = 7;
	const double sgCoeff[sgOrder] = { -0.0952,    0.1429,    0.2857,    0.3333,    0.2857,    0.1429,   -0.0952};

//	{-0.0839,    0.0210,    0.1026,    0.1608,    0.1958,    0.2075,    0.1958,    0.1608, 0.1026,    0.0210,   -0.0839};


	// calculate mean isfet and sense trace
	std::vector<double> isfetTraceAvg(frames,0);
	senseTraceAvg.resize(frames,0.);
	unsigned int countIsfet = 0;
	for (unsigned int r = rowStart; r < rowEnd; r++){
		unsigned int countIsfetThisRow = 0;
		if (senseTraceIsValid[r]){

			for (unsigned int c = colStart; c < colEnd; c++){
				const unsigned int ind =  c +  r*cols;

				if ( (( *mask ) [ind] & MaskPinned)==0 ){
					// (r,c) is not a pinned well and has a valid sensing electrode rowsum trace

					for (unsigned f = 0; f < frames; f++){
						isfetTraceAvg[f] += (double) image[ind + f*frameStride];
					}
					countIsfetThisRow ++;
				}
			}
			countIsfet += countIsfetThisRow;

			for (unsigned f = 0; f < frames; f++){
				senseTraceAvg[f] += (double) senseTrace[r][f] * (double) countIsfetThisRow;
			}

		}
	}

	// skip current region if no valid trace
	if (countIsfet == 0)
		return 0.;

	for (unsigned f = 0; f < frames; f++){
		isfetTraceAvg[f] /= (double) countIsfet;
		senseTraceAvg[f] /= (double) countIsfet;

	}

	// transform to vfc then interpolate back to uniform
	if (isThumbnail){
		const unsigned int regionTNRow = rowStart/numThumbnailRows;
		const unsigned int regionTNCol = colStart/numThumbnailCols;
		convertTraceUniformToThumbnail(senseTraceAvg, regionTNRow, regionTNCol);
	}

	// zero the isfet and the sense electrode traces
	ZeroTrace(isfetTraceAvg, 5);
	ZeroTrace(senseTraceAvg, 5);

	// correct sensing electrode trace drift
	if (correctSenseDrift){
		CorrectTraceDrift(senseTraceAvg, 5, 5);
	}


	// calculate optimal scale factor
	std::vector <double> isfetTraceCorr(frames, 0);
	std::vector <double> isfetTraceCorrSG;
	std::vector <double> err;
	double minErr = -1;
	scaleFactorOpt = 0.;
	unsigned int iOpt = 0, i = 0;;
	for (double sf = scaleFactorMin; sf <= scaleFactorMax; sf += scaleFactorBin){
		// calculate corrected trace with a scale factor
		for (unsigned int f = 0; f < frames; f++){
			isfetTraceCorr[f] = isfetTraceAvg[f] - sf * senseTraceAvg[f];
		}

		// filter with  Savitzky-Golay  (will be able to save a few steps by computing error directly)
		const int sgHalf = (sgOrder-1)/2;
		isfetTraceCorrSG = isfetTraceCorr;
		for (unsigned int f = sgHalf; f < frames -sgHalf; f++  ){
			double val = 0.;
			for (unsigned int i = 0; i < sgOrder; i++){
				val += sgCoeff[i]*isfetTraceCorr[f - sgHalf + i];
			}
			isfetTraceCorrSG[f] = val;
		}

		// calculate square error sum
		double errThis = 0.;
		for (unsigned int f = sgHalf; f < frames -sgHalf; f++  ){
			errThis += (isfetTraceCorrSG[f] -isfetTraceCorr[f])*(isfetTraceCorrSG[f] -isfetTraceCorr[f]);
		}

		if (minErr == -1 || errThis < minErr){
			minErr = errThis;
			scaleFactorOpt = sf;
			iOpt = i;

		}

//		if (DEBUG){
//			printf("%7.3f, %7.3f\n", sf, errThis);
//		}

		err.push_back(errThis);
		i ++;



	}

	// refine scaleFactor
	if (iOpt >0 && iOpt < i-1){   // i is total number of scale factors tried
		double x0 = scaleFactorOpt - scaleFactorBin;
		double y0 = err[iOpt - 1];
		double x1 = scaleFactorOpt;
		double y1 = err[iOpt];
		double x2 = scaleFactorOpt + scaleFactorBin;
		double y2 = err[iOpt + 1];

		arma::mat A = arma::mat(3,3);   // mat, vec are default double
		arma::vec b = arma::vec(3);

		A(0,0) = x0*x0*x0*x0 + x1*x1*x1*x1 + x2*x2*x2*x2;
		A(0,1) = x0*x0*x0 + x1*x1*x1 + x2*x2*x2;
		A(0,2) = x0*x0 + x1*x1 + x2*x2;

		A(1,0) = A(0,1);
		A(1,1) = A(0,2);
		A(1,2) = x0 + x1 + x2;

		A(2,0) = A(0,2);
		A(2,1) = A(1,2);
		A(2,2) = 3.;

		b(0) = x0*x0*y0 + x1*x1*y1 + x2*x2*y2;
		b(1) = x0*y0 + x1*y1 + x2*y2;
		b(2) = y0 + y1 + y2;

		arma::mat p = solve(A, b);

		const double scaleFactorOptNew = -p(1)/p(0)/2.;


//		if (DEBUG){
//			printf("scaleFactorOpt %5.5f scaleFactorOptNew %5.5f\n", scaleFactorOpt, scaleFactorOptNew);
//			printf("x = %5.5f, %5.5f, %5.5f\n", x0, x1, x2);
//			printf("y = %5.5f, %5.5f, %5.5f\n", y0, y1, y2);
//			printf("p = %5.5f, %5.5f, %5.5f\n", p(0), p(1), p(2));
//
//		}

		if (x0 < scaleFactorOptNew && scaleFactorOptNew < x2)
			scaleFactorOpt = scaleFactorOptNew;









	}


//	if (DEBUG){
//		printf("%s \n", rowsumFileName.c_str());
//		printf("rowStart %u rowEnd %u \n", rowStart, rowEnd);
//		printf("optimal scale factor is %10.3f \n", scaleFactorOpt);
//		printf("isfetTraceAvg, senseTraceAvg, isfetTraceCorr, isfetTraceCorrSG\n");
//		for (unsigned int f = 0; f < frames; f++){
//			printf("%7.3f, %7.3f, %7.3f, %7.3f\n", isfetTraceAvg[f],  senseTraceAvg[f],  isfetTraceCorr[f], isfetTraceCorrSG[f]);
//		}
//		fflush(stdout);
//	}


	return  (float) scaleFactorOpt;



}

void FluidPotentialCorrector::convertTraceUniformToThumbnail(
		std::vector<double> & trace,
		const unsigned int row,
		const unsigned int col){

	// get vfc time stamps for the thumbnail region
	const std::vector<unsigned int> timeStamp_vfc = timeStampThumbnailRegionVFC(row, col);
	// convert trace from uniform to vfc
	const std::vector<double> trace_vfc = uniform2vfc(trace, timeStamps, timeStamp_vfc);
	// interpoloate vfc trace to thumbnail
	const std::vector<double> trace_tn = vfc2thumbnail(trace_vfc, timeStamp_vfc, timeStamps);

	//	if (row == 5  && col == 5){
//
//		// print full res
//		printf("full res trace\n");
//		for (unsigned int i = 0; i < timeStamps.size(); i++){
//			printf("%d\t%10.5f\n", timeStamps[i], trace[i]);
//		}
//
//		// print vfc
//		printf("vfc trace\n");
//		for (unsigned int i = 0; i < timeStamp_vfc.size(); i++){
//			printf("%d\t%10.5f\n", timeStamp_vfc[i], trace_vfc[i]);
//		}
//
//		// print full res
//		printf("tn trace\n");
//		for (unsigned int i = 0; i < timeStamps.size(); i++){
//			printf("%d\t%10.5f\n", timeStamps[i], trace_tn[i]);
//		}
//
//
//	}

	trace = trace_tn;



}


std::vector<double> FluidPotentialCorrector::vfc2thumbnail(
		const std::vector<double> trace_vfc,
		const std::vector<unsigned int> timeStamp_vfc,
		const std::vector<unsigned int> timeStamps){

//	function trace_tn = vfc2thumbnailTrace(ftimes_vfc, trace_vfc, ftimes_tn)
//
//	trace_tn = zeros(size(trace_vfc, 1), length(ftimes_tn));
//	framePrev = 1;
//	valuePrev = trace_vfc(:, 1);
//	for i = 1:length(ftimes_vfc)
//	    frame = ftimes_vfc(i);
//	    value = trace_vfc(:, i);
//
//	    % move last point to the end of thumbnail
//	    if i == length(ftimes_vfc)
//	        frame = ftimes_tn(end);
//	    end;
//
//	    % interpolation  between points * . . . *
//	    for f = framePrev+1:frame-1
//	        trace_tn(:, f) = valuePrev + (f-framePrev)/(frame-framePrev) * (value - valuePrev);
//	    end;
//	    trace_tn(:,frame) = value;  % set current point value
//
//	    framePrev = frame;
//	    valuePrev = value;
//	end;

	std::vector<double> trace_tn(timeStamps.size(), 0.);
	unsigned int framePrev = 2; // first frame starts at 2
	double valuePrev = trace_vfc[0];
	for (unsigned int i = 0; i < timeStamp_vfc.size();i++){
		unsigned int frame = timeStamp_vfc[i];
		double value = trace_vfc[i];

		// move last point to the end of thumbnail
		if (i == timeStamp_vfc.size()-1)
			frame = * (timeStamps.end()-1);

		// printf("frame %d, value %10.5f \n", frame, value);fflush(stdout);

		// interpolate between points * . . . *
		for (unsigned int f = framePrev+2; f <= frame - 2; f+=2){
			trace_tn[f/2-1] = valuePrev +  (double)(f - framePrev) / (double) (frame -framePrev) * (value - valuePrev);
		//	printf("interpolate frame %d, value %10.5f \n", f, trace_tn[f/2-1]);fflush(stdout);
		}
		trace_tn[frame/2-1] = value;
		framePrev = frame;
		valuePrev = value;

	}

	return trace_tn;



}

std::vector<double> FluidPotentialCorrector::uniform2vfc(
		const std::vector<double> & trace,
		const std::vector<unsigned int> & timeStamp,
		const std::vector<unsigned int> & timeStamp_vfc){

	std::vector<double> trace_vfc(timeStamp_vfc.size(), 0.);
	for (unsigned int f = 0; f < trace_vfc.size(); f++){
		trace_vfc[f] = 0.f;
	}

	unsigned int i = 0;
	unsigned int cnt = 0;
	for (unsigned int f = 0; f < timeStamp.size(); f++){
		trace_vfc[i] +=  trace[f];
		cnt ++;

		//printf("In CompressFrames, timeStamp_vfc[%d] %d, timeStamp[%d] %d trace[f] %10.5f trace_vfc[f] %10.5f\n", i, timeStamp_vfc[i], f, timeStamp[f], trace[f], trace_vfc[i]);

		if (timeStamp_vfc[i] == timeStamp[f]){
			trace_vfc[i] /= (float) cnt;
 //			printf("In CompressFrames loop, trace_vfc[i] %10.5f\n", rowTrace_vfc[i]);
			cnt = 0;
			i ++;

			if (i == timeStamp_vfc.size())
				break;
		}



	}

	return trace_vfc;

}

void FluidPotentialCorrector::ZeroTrace(std::vector<double> & trace, const unsigned int numOffsetFrames){

	if (numOffsetFrames > trace.size()){
		if (DEBUG){
			printf("Warning RowSumDataZeroTrace number of offset frames larger than trace size.\n");
		}
		return;
	}

	if (numOffsetFrames > 0){
		float offset = 0.;
		for (unsigned int f = 0; f < numOffsetFrames; f++){
			offset += trace[f];
		}
		offset /= (double) numOffsetFrames;

		// subtract offset from trace
		for (unsigned int f = 0; f < trace.size(); f++)
			trace[f] -= offset;
	}

}

void FluidPotentialCorrector::CorrectTraceDrift(std::vector<double> & trace, const unsigned int numPointsStart, const unsigned int numPointsEnd){

	// find rotation factor beta
	double numerator = 0., denominator = 0.;
	for (unsigned int f = 0; f < trace.size(); f++){

		// consider only the first numPointsStart and the last  numPointsEnd points
		if (f < numPointsStart || f >= trace.size() - numPointsEnd){
			numerator += trace[f] * (double) f;
			denominator += (double) (f*f);
		}
	}
	const double beta = numerator / denominator;

	// apply correction
	for (unsigned int f = 0; f < trace.size(); f++){
		trace[f] = trace[f] - beta * (double) f;
	}
}

void FluidPotentialCorrector::setThreshold(const double threshold){
	this->noiseThresh = threshold;
}

std::vector<unsigned int> FluidPotentialCorrector::timeStampThumbnailRegionVFC(const unsigned int row, const unsigned int col){
	std::vector< std::vector < std::vector <unsigned int>  > > timeStampVFC;

	timeStampVFC.resize(numThumbnailRegionsRow);
	for (unsigned int r = 0; r < numThumbnailRegionsRow; r++)
		timeStampVFC[r].resize(numThumbnailRegionsCol);


	const unsigned int tsvfc_r0c0 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106};
	const unsigned int tsvfc_r0c1 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106,122};
	const unsigned int tsvfc_r0c2 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106,122};
	const unsigned int tsvfc_r0c3 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134};
	const unsigned int tsvfc_r0c4 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134};
	const unsigned int tsvfc_r0c5 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134};
	const unsigned int tsvfc_r0c6 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134,150};
	const unsigned int tsvfc_r0c7 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r0c8 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r0c9 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r0c10 [] = {2,18,34,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,94,98,102,106,110,118,126,142,158,174};
	const unsigned int tsvfc_r0c11 [] = {2,18,34,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,110,114,118,122,126,130,134,138,142,146,150,154,158,162,166,170,174,178,186};
	const unsigned int tsvfc_r1c0 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106};
	const unsigned int tsvfc_r1c1 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106,122};
	const unsigned int tsvfc_r1c2 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106,122};
	const unsigned int tsvfc_r1c3 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,78,86,94,110,126};
	const unsigned int tsvfc_r1c4 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134};
	const unsigned int tsvfc_r1c5 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134};
	const unsigned int tsvfc_r1c6 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134,150};
	const unsigned int tsvfc_r1c7 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r1c8 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r1c9 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r1c10 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,102,110,118,134,150,166};
	const unsigned int tsvfc_r1c11 [] = {2,18,34,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,94,98,102,106,110,118,126,142,158,174};
	const unsigned int tsvfc_r2c0 [] = {2,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,62,66,74,82,98};
	const unsigned int tsvfc_r2c1 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106,122};
	const unsigned int tsvfc_r2c2 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106,122};
	const unsigned int tsvfc_r2c3 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,78,86,94,110,126};
	const unsigned int tsvfc_r2c4 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,78,86,94,110,126};
	const unsigned int tsvfc_r2c5 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134};
	const unsigned int tsvfc_r2c6 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134,150};
	const unsigned int tsvfc_r2c7 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r2c8 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r2c9 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r2c10 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,102,110,118,134,150,166};
	const unsigned int tsvfc_r2c11 [] = {2,18,34,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,94,98,102,106,110,118,126,142,158,174};
	const unsigned int tsvfc_r3c0 [] = {2,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,62,66,74,82,98};
	const unsigned int tsvfc_r3c1 [] = {2,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,62,66,74,82,98,114};
	const unsigned int tsvfc_r3c2 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106,122};
	const unsigned int tsvfc_r3c3 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,78,86,94,110,126};
	const unsigned int tsvfc_r3c4 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,78,86,94,110,126};
	const unsigned int tsvfc_r3c5 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134};
	const unsigned int tsvfc_r3c6 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134,150};
	const unsigned int tsvfc_r3c7 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r3c8 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r3c9 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r3c10 [] = {2,18,34,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,94,98,102,106,110,118,126,142,158,174};
	const unsigned int tsvfc_r3c11 [] = {2,18,34,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,94,98,102,106,110,118,126,142,158,174};
	const unsigned int tsvfc_r4c0 [] = {2,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,62,66,74,82,98};
	const unsigned int tsvfc_r4c1 [] = {2,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,62,66,74,82,98,114};
	const unsigned int tsvfc_r4c2 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106,122};
	const unsigned int tsvfc_r4c3 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,78,86,94,110,126};
	const unsigned int tsvfc_r4c4 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,78,86,94,110,126};
	const unsigned int tsvfc_r4c5 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134};
	const unsigned int tsvfc_r4c6 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134,150};
	const unsigned int tsvfc_r4c7 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r4c8 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r4c9 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r4c10 [] = {2,18,34,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,94,98,102,106,110,118,126,142,158,174};
	const unsigned int tsvfc_r4c11 [] = {2,18,34,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,94,98,102,106,110,118,126,142,158,174};
	const unsigned int tsvfc_r5c0 [] = {2,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,62,66,74,82,98};
	const unsigned int tsvfc_r5c1 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106,122};
	const unsigned int tsvfc_r5c2 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106,122};
	const unsigned int tsvfc_r5c3 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,78,86,94,110,126};
	const unsigned int tsvfc_r5c4 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,78,86,94,110,126};
	const unsigned int tsvfc_r5c5 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134};
	const unsigned int tsvfc_r5c6 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134,150};
	const unsigned int tsvfc_r5c7 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r5c8 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r5c9 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r5c10 [] = {2,18,34,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,94,98,102,106,110,118,126,142,158,174};
	const unsigned int tsvfc_r5c11 [] = {2,18,34,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,94,98,102,106,110,118,126,142,158,174};
	const unsigned int tsvfc_r6c0 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106};
	const unsigned int tsvfc_r6c1 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106,122};
	const unsigned int tsvfc_r6c2 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106,122};
	const unsigned int tsvfc_r6c3 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,78,86,94,110,126};
	const unsigned int tsvfc_r6c4 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134};
	const unsigned int tsvfc_r6c5 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134};
	const unsigned int tsvfc_r6c6 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134,150};
	const unsigned int tsvfc_r6c7 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r6c8 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r6c9 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r6c10 [] = {2,18,34,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,94,98,102,106,110,118,126,142,158,174};
	const unsigned int tsvfc_r6c11 [] = {2,18,34,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,102,106,110,114,118,126,134,150,166,182};
	const unsigned int tsvfc_r7c0 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,102,106,110,114,118};
	const unsigned int tsvfc_r7c1 [] = {2,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,70,74,82,90,106,122};
	const unsigned int tsvfc_r7c2 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,90,98,114,130};
	const unsigned int tsvfc_r7c3 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134};
	const unsigned int tsvfc_r7c4 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134};
	const unsigned int tsvfc_r7c5 [] = {2,18,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,78,82,86,94,102,118,134};
	const unsigned int tsvfc_r7c6 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,102,110,126,142,158};
	const unsigned int tsvfc_r7c7 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r7c8 [] = {2,18,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,86,90,94,98,106,114,130,146,162};
	const unsigned int tsvfc_r7c9 [] = {2,18,34,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,94,98,102,106,114,122,138,154,170};
	const unsigned int tsvfc_r7c10 [] = {2,18,34,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,94,98,102,106,110,118,126,142,158,174};
	const unsigned int tsvfc_r7c11 [] = {2,18,34,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,110,114,118,122,126,130,134,138,142,146,150,154,158,162,166,170,174,178,186};

	const unsigned int * tsvfc;
	unsigned int tsvfc_len = 0;
	if (row == 0 && col == 0) {tsvfc = tsvfc_r0c0; tsvfc_len = 31;}
	if (row == 0 && col == 1) {tsvfc = tsvfc_r0c1; tsvfc_len = 32;}
	if (row == 0 && col == 2) {tsvfc = tsvfc_r0c2; tsvfc_len = 32;}
	if (row == 0 && col == 3) {tsvfc = tsvfc_r0c3; tsvfc_len = 34;}
	if (row == 0 && col == 4) {tsvfc = tsvfc_r0c4; tsvfc_len = 34;}
	if (row == 0 && col == 5) {tsvfc = tsvfc_r0c5; tsvfc_len = 34;}
	if (row == 0 && col == 6) {tsvfc = tsvfc_r0c6; tsvfc_len = 35;}
	if (row == 0 && col == 7) {tsvfc = tsvfc_r0c7; tsvfc_len = 36;}
	if (row == 0 && col == 8) {tsvfc = tsvfc_r0c8; tsvfc_len = 36;}
	if (row == 0 && col == 9) {tsvfc = tsvfc_r0c9; tsvfc_len = 36;}
	if (row == 0 && col == 10) {tsvfc = tsvfc_r0c10; tsvfc_len = 38;}
	if (row == 0 && col == 11) {tsvfc = tsvfc_r0c11; tsvfc_len = 55;}
	if (row == 1 && col == 0) {tsvfc = tsvfc_r1c0; tsvfc_len = 31;}
	if (row == 1 && col == 1) {tsvfc = tsvfc_r1c1; tsvfc_len = 32;}
	if (row == 1 && col == 2) {tsvfc = tsvfc_r1c2; tsvfc_len = 32;}
	if (row == 1 && col == 3) {tsvfc = tsvfc_r1c3; tsvfc_len = 33;}
	if (row == 1 && col == 4) {tsvfc = tsvfc_r1c4; tsvfc_len = 34;}
	if (row == 1 && col == 5) {tsvfc = tsvfc_r1c5; tsvfc_len = 34;}
	if (row == 1 && col == 6) {tsvfc = tsvfc_r1c6; tsvfc_len = 35;}
	if (row == 1 && col == 7) {tsvfc = tsvfc_r1c7; tsvfc_len = 36;}
	if (row == 1 && col == 8) {tsvfc = tsvfc_r1c8; tsvfc_len = 36;}
	if (row == 1 && col == 9) {tsvfc = tsvfc_r1c9; tsvfc_len = 36;}
	if (row == 1 && col == 10) {tsvfc = tsvfc_r1c10; tsvfc_len = 37;}
	if (row == 1 && col == 11) {tsvfc = tsvfc_r1c11; tsvfc_len = 38;}
	if (row == 2 && col == 0) {tsvfc = tsvfc_r2c0; tsvfc_len = 31;}
	if (row == 2 && col == 1) {tsvfc = tsvfc_r2c1; tsvfc_len = 32;}
	if (row == 2 && col == 2) {tsvfc = tsvfc_r2c2; tsvfc_len = 32;}
	if (row == 2 && col == 3) {tsvfc = tsvfc_r2c3; tsvfc_len = 33;}
	if (row == 2 && col == 4) {tsvfc = tsvfc_r2c4; tsvfc_len = 33;}
	if (row == 2 && col == 5) {tsvfc = tsvfc_r2c5; tsvfc_len = 34;}
	if (row == 2 && col == 6) {tsvfc = tsvfc_r2c6; tsvfc_len = 35;}
	if (row == 2 && col == 7) {tsvfc = tsvfc_r2c7; tsvfc_len = 36;}
	if (row == 2 && col == 8) {tsvfc = tsvfc_r2c8; tsvfc_len = 36;}
	if (row == 2 && col == 9) {tsvfc = tsvfc_r2c9; tsvfc_len = 36;}
	if (row == 2 && col == 10) {tsvfc = tsvfc_r2c10; tsvfc_len = 37;}
	if (row == 2 && col == 11) {tsvfc = tsvfc_r2c11; tsvfc_len = 38;}
	if (row == 3 && col == 0) {tsvfc = tsvfc_r3c0; tsvfc_len = 31;}
	if (row == 3 && col == 1) {tsvfc = tsvfc_r3c1; tsvfc_len = 32;}
	if (row == 3 && col == 2) {tsvfc = tsvfc_r3c2; tsvfc_len = 32;}
	if (row == 3 && col == 3) {tsvfc = tsvfc_r3c3; tsvfc_len = 33;}
	if (row == 3 && col == 4) {tsvfc = tsvfc_r3c4; tsvfc_len = 33;}
	if (row == 3 && col == 5) {tsvfc = tsvfc_r3c5; tsvfc_len = 34;}
	if (row == 3 && col == 6) {tsvfc = tsvfc_r3c6; tsvfc_len = 35;}
	if (row == 3 && col == 7) {tsvfc = tsvfc_r3c7; tsvfc_len = 36;}
	if (row == 3 && col == 8) {tsvfc = tsvfc_r3c8; tsvfc_len = 36;}
	if (row == 3 && col == 9) {tsvfc = tsvfc_r3c9; tsvfc_len = 36;}
	if (row == 3 && col == 10) {tsvfc = tsvfc_r3c10; tsvfc_len = 38;}
	if (row == 3 && col == 11) {tsvfc = tsvfc_r3c11; tsvfc_len = 38;}
	if (row == 4 && col == 0) {tsvfc = tsvfc_r4c0; tsvfc_len = 31;}
	if (row == 4 && col == 1) {tsvfc = tsvfc_r4c1; tsvfc_len = 32;}
	if (row == 4 && col == 2) {tsvfc = tsvfc_r4c2; tsvfc_len = 32;}
	if (row == 4 && col == 3) {tsvfc = tsvfc_r4c3; tsvfc_len = 33;}
	if (row == 4 && col == 4) {tsvfc = tsvfc_r4c4; tsvfc_len = 33;}
	if (row == 4 && col == 5) {tsvfc = tsvfc_r4c5; tsvfc_len = 34;}
	if (row == 4 && col == 6) {tsvfc = tsvfc_r4c6; tsvfc_len = 35;}
	if (row == 4 && col == 7) {tsvfc = tsvfc_r4c7; tsvfc_len = 36;}
	if (row == 4 && col == 8) {tsvfc = tsvfc_r4c8; tsvfc_len = 36;}
	if (row == 4 && col == 9) {tsvfc = tsvfc_r4c9; tsvfc_len = 36;}
	if (row == 4 && col == 10) {tsvfc = tsvfc_r4c10; tsvfc_len = 38;}
	if (row == 4 && col == 11) {tsvfc = tsvfc_r4c11; tsvfc_len = 38;}
	if (row == 5 && col == 0) {tsvfc = tsvfc_r5c0; tsvfc_len = 31;}
	if (row == 5 && col == 1) {tsvfc = tsvfc_r5c1; tsvfc_len = 32;}
	if (row == 5 && col == 2) {tsvfc = tsvfc_r5c2; tsvfc_len = 32;}
	if (row == 5 && col == 3) {tsvfc = tsvfc_r5c3; tsvfc_len = 33;}
	if (row == 5 && col == 4) {tsvfc = tsvfc_r5c4; tsvfc_len = 33;}
	if (row == 5 && col == 5) {tsvfc = tsvfc_r5c5; tsvfc_len = 34;}
	if (row == 5 && col == 6) {tsvfc = tsvfc_r5c6; tsvfc_len = 35;}
	if (row == 5 && col == 7) {tsvfc = tsvfc_r5c7; tsvfc_len = 36;}
	if (row == 5 && col == 8) {tsvfc = tsvfc_r5c8; tsvfc_len = 36;}
	if (row == 5 && col == 9) {tsvfc = tsvfc_r5c9; tsvfc_len = 36;}
	if (row == 5 && col == 10) {tsvfc = tsvfc_r5c10; tsvfc_len = 38;}
	if (row == 5 && col == 11) {tsvfc = tsvfc_r5c11; tsvfc_len = 38;}
	if (row == 6 && col == 0) {tsvfc = tsvfc_r6c0; tsvfc_len = 31;}
	if (row == 6 && col == 1) {tsvfc = tsvfc_r6c1; tsvfc_len = 32;}
	if (row == 6 && col == 2) {tsvfc = tsvfc_r6c2; tsvfc_len = 32;}
	if (row == 6 && col == 3) {tsvfc = tsvfc_r6c3; tsvfc_len = 33;}
	if (row == 6 && col == 4) {tsvfc = tsvfc_r6c4; tsvfc_len = 34;}
	if (row == 6 && col == 5) {tsvfc = tsvfc_r6c5; tsvfc_len = 34;}
	if (row == 6 && col == 6) {tsvfc = tsvfc_r6c6; tsvfc_len = 35;}
	if (row == 6 && col == 7) {tsvfc = tsvfc_r6c7; tsvfc_len = 36;}
	if (row == 6 && col == 8) {tsvfc = tsvfc_r6c8; tsvfc_len = 36;}
	if (row == 6 && col == 9) {tsvfc = tsvfc_r6c9; tsvfc_len = 36;}
	if (row == 6 && col == 10) {tsvfc = tsvfc_r6c10; tsvfc_len = 38;}
	if (row == 6 && col == 11) {tsvfc = tsvfc_r6c11; tsvfc_len = 38;}
	if (row == 7 && col == 0) {tsvfc = tsvfc_r7c0; tsvfc_len = 43;}
	if (row == 7 && col == 1) {tsvfc = tsvfc_r7c1; tsvfc_len = 32;}
	if (row == 7 && col == 2) {tsvfc = tsvfc_r7c2; tsvfc_len = 33;}
	if (row == 7 && col == 3) {tsvfc = tsvfc_r7c3; tsvfc_len = 34;}
	if (row == 7 && col == 4) {tsvfc = tsvfc_r7c4; tsvfc_len = 34;}
	if (row == 7 && col == 5) {tsvfc = tsvfc_r7c5; tsvfc_len = 34;}
	if (row == 7 && col == 6) {tsvfc = tsvfc_r7c6; tsvfc_len = 35;}
	if (row == 7 && col == 7) {tsvfc = tsvfc_r7c7; tsvfc_len = 36;}
	if (row == 7 && col == 8) {tsvfc = tsvfc_r7c8; tsvfc_len = 36;}
	if (row == 7 && col == 9) {tsvfc = tsvfc_r7c9; tsvfc_len = 37;}
	if (row == 7 && col == 10) {tsvfc = tsvfc_r7c10; tsvfc_len = 38;}
	if (row == 7 && col == 11) {tsvfc = tsvfc_r7c11; tsvfc_len = 55;}


	std::vector<unsigned int> tsvfc_vec (tsvfc, tsvfc + tsvfc_len );

//	printf("%d %d sizeof(tsvfc) %ld sizeof(tsvfc[0]) %ld\n", row, col, sizeof(tsvfc), sizeof(tsvfc[0]) );
//	printf("%d %d sizeof(tsvfc_r0c0) %ld sizeof(tsvfc_r0c0[0]) %ld\n", row, col, sizeof(tsvfc_r0c0), sizeof(tsvfc_r0c0[0]) );

	if ( row >= numThumbnailRegionsRow || col >= numThumbnailRegionsCol){
		printf("FluidPotentialCorrector::timeStampThumbnailRegionVFC: Error - row %d and col %d out of Range!\n", row, col);
	}

	return tsvfc_vec;

}
