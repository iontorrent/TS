/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

#include "PairPixelXtalkCorrector.h"

PairPixelXtalkCorrector::PairPixelXtalkCorrector()
{
}

//Caution - this code is awaiting final P2 chip. It should be tested when valid data
//is available.
void PairPixelXtalkCorrector::Correct(RawImage *raw, float xtalk_fraction)
{
    int nRows = raw->rows;
    int nCols = raw->cols;
    int nFrames = raw->frames;

    int phase = (raw->chip_offset_y)%2;
    float denominator = (1-xtalk_fraction*xtalk_fraction);
    /*-----------------------------------------------------------------------------------------------------------*/
    // doublet xtalk correction - electrical xtalk between two neighboring pixels in the same column is xtalk_fraction
    //
    // Model is:
    // p1 = c1 + xtalk_fraction * c2
    // p2 = c2 + xtalk_fraction * c1
    // where p1,p2 - observed values, and c1,c2 - actual values. We solve the system for c1,c2.
    /*-----------------------------------------------------------------------------------------------------------*/
    for( int f=0; f<nFrames; ++f ){
        for( int c=0; c<nCols; ++c ){
            for(int r=phase; r<nRows-1; r+=2 ){
                short p1 = raw->image[f*raw->frameStride+r*raw->cols+c];
                short p2 = raw->image[f*raw->frameStride+(r+1)*raw->cols+c];
                raw->image[f*raw->frameStride+r*raw->cols+c] = (p1-xtalk_fraction*p2)/denominator;
                raw->image[f*raw->frameStride+(r+1)*raw->cols+c] = (p2-xtalk_fraction*p1)/denominator;
            }
        }
    }
}
