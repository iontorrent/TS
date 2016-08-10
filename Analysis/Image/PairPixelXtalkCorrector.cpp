/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

#include "PairPixelXtalkCorrector.h"

PairPixelXtalkCorrector::PairPixelXtalkCorrector()
{
}

void PairPixelXtalkCorrector::CorrectThumbnailFromFile(RawImage *raw , const char  * xtalkFileName){
	const int numRows = 8;
	const int numCols = 12;
	const float xtalk_fraction_default = 0.2f;
	float xtalk_fraction1[numRows][numCols], xtalk_fraction2[numRows][numCols];

	// initialize
	for (int r=0; r <numRows;r++){
		for (int c=0; c<numCols;c++){
			xtalk_fraction1[r][c] = xtalk_fraction_default;
			xtalk_fraction2[r][c] = xtalk_fraction_default;
		}
	}

	// read from file
	std::ifstream file(xtalkFileName);
	std::string line;
	if (file.is_open()){
		while (getline(file,line)){
			int row, col;
			float xtalk1, xtalk2;
			sscanf(line.c_str(), "%d, %d, %f, %f", &row, &col, &xtalk1, &xtalk2);
			if (xtalk1 >= 0.f && xtalk1 < 0.5f && xtalk2 >= 0.f && xtalk2 < 0.5f && (xtalk1 != 0.f && xtalk2 != 0.f)){
				xtalk_fraction1[row][col] = xtalk1;
				xtalk_fraction2[row][col] = xtalk2;

			}


			//fprintf(stdout, "row %d, col %d, xtalk %1.5f\n", row, col, xtalk);
		}
	}

	// print out crosstlak values
	if (0){
		for (int r=0; r <numRows;r++){
			for (int c=0; c<numCols;c++){
				fprintf(stdout, "%1.3f, ", xtalk_fraction1[r][c]);
			}
			fprintf(stdout, "\n");
		}
		fprintf(stdout, "\n");
		for (int r=0; r <numRows;r++){
			for (int c=0; c<numCols;c++){
				fprintf(stdout, "%1.3f, ", xtalk_fraction2[r][c]);
			}
			fprintf(stdout, "\n");
		}

	}



    int nRows = raw->rows;
    int nCols = raw->cols;
    int nFrames = raw->frames;

    int phase = (raw->chip_offset_y)%2;

    /*-----------------------------------------------------------------------------------------------------------*/
    // doublet xtalk correction - electrical xtalk between two neighboring pixels in the same column is xtalk_fraction
    //
    // Model is:
    // p1 = (1-xtalk_fraction)*c1 + xtalk_fraction * c2
    // p2 = (1-xtalk_fraction)*c2 + xtalk_fraction * c1
    // where p1,p2 - observed values, and c1,c2 - actual values. We solve the system for c1,c2.
    /*-----------------------------------------------------------------------------------------------------------*/
    for( int f=0; f<nFrames; ++f ){
        for( int c=0; c<nCols; ++c ){
            for(int r=phase; r<nRows-1; r+=2 ){
                float p1_0 = raw->image[r*raw->cols+c];
            	float p1 = raw->image[f*raw->frameStride+r*raw->cols+c] - p1_0;
            	float p2_0 = raw->image[(r+1)*raw->cols+c];
                float p2 = raw->image[f*raw->frameStride+(r+1)*raw->cols+c] - p2_0;
                float xt1 = xtalk_fraction1[r/100][c/100];
                float xt2 = xtalk_fraction2[r/100][c/100];

                float denominator = (1.f-xt1*xt2);
                float gain = 0.8f;

                // preserve offset
                raw->image[f*raw->frameStride+r*raw->cols+c] = p1_0 + (p1-xt1*p2)/denominator*gain;
                raw->image[f*raw->frameStride+(r+1)*raw->cols+c] = p2_0 + (p2-xt2*p1)/denominator*gain;
                //fprintf(stdout,"%d, %d, %d, %d, %1.4f\n", r, c, r/100, c/100, xt);

//                raw->image[f*raw->frameStride+r*raw->cols+c] = p1_0 + ((p1 - xtalk_fraction*p2)/(1.0f-xtalk_fraction));;
//                raw->image[f*raw->frameStride+(r+1)*raw->cols+c] = p2_0 + ((p2 - xtalk_fraction*p1)/(1.0f-xtalk_fraction));

//                short p1 = raw->image[f*raw->frameStride+r*raw->cols+c];
//                short p2 = raw->image[f*raw->frameStride+(r+1)*raw->cols+c];
//                raw->image[f*raw->frameStride+r*raw->cols+c] = ((1-xtalk_fraction)*p1-xtalk_fraction*p2)/denominator;
//                raw->image[f*raw->frameStride+(r+1)*raw->cols+c] = ((1-xtalk_fraction)*p2-xtalk_fraction*p1)/denominator;
            }
        }
    }








}

//Caution - this code is awaiting final P2 chip. It should be tested when valid data
//is available.
void PairPixelXtalkCorrector::Correct(RawImage *raw, float xtalk_fraction)
{
    int nRows = raw->rows;
    int nCols = raw->cols;
    int nFrames = raw->frames;

    int phase = (raw->chip_offset_y)%2;
    float denominator = (1.f-2.f*xtalk_fraction);
    /*-----------------------------------------------------------------------------------------------------------*/
    // doublet xtalk correction - electrical xtalk between two neighboring pixels in the same column is xtalk_fraction
    //
    // Model is:
    // p1 = (1-xtalk_fraction)*c1 + xtalk_fraction * c2
    // p2 = (1-xtalk_fraction)*c2 + xtalk_fraction * c1
    // where p1,p2 - observed values, and c1,c2 - actual values. We solve the system for c1,c2.
    /*-----------------------------------------------------------------------------------------------------------*/
    for( int f=0; f<nFrames; ++f ){
        for( int c=0; c<nCols; ++c ){
            for(int r=phase; r<nRows-1; r+=2 ){
            	// conform to datacollect definition
                float p1_0 = raw->image[r*raw->cols+c];
            	float p1 = raw->image[f*raw->frameStride+r*raw->cols+c] - p1_0;
            	float p2_0 = raw->image[(r+1)*raw->cols+c];
                float p2 = raw->image[f*raw->frameStride+(r+1)*raw->cols+c] - p2_0;
                // preserve offset
                raw->image[f*raw->frameStride+r*raw->cols+c] = p1_0 + ((p1 - xtalk_fraction*p2)/(1.0f-xtalk_fraction));;
                raw->image[f*raw->frameStride+(r+1)*raw->cols+c] = p2_0 + ((p2 - xtalk_fraction*p1)/(1.0f-xtalk_fraction));

//                raw->image[f*raw->frameStride+r*raw->cols+c] = p1_0 + ((1-xtalk_fraction)*p1-xtalk_fraction*p2)/denominator;
//                raw->image[f*raw->frameStride+(r+1)*raw->cols+c] = p2_0 + ((1-xtalk_fraction)*p2-xtalk_fraction*p1)/denominator;



//                short p1 = raw->image[f*raw->frameStride+r*raw->cols+c];
//                short p2 = raw->image[f*raw->frameStride+(r+1)*raw->cols+c];
//                raw->image[f*raw->frameStride+r*raw->cols+c] = ((1-xtalk_fraction)*p1-xtalk_fraction*p2)/denominator;
//                raw->image[f*raw->frameStride+(r+1)*raw->cols+c] = ((1-xtalk_fraction)*p2-xtalk_fraction*p1)/denominator;
            }
        }
    }
}
