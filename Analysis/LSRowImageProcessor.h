/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef LSROWIMAGEPROCESSOR_H
#define LSROWIMAGEPROCESSOR_H

#include <stdlib.h>
#include <string.h>
#include "ChipIdDecoder.h"
#include "ChannelXTCorrection.h"

#define NUM_CHANS   4

class LSRowImageProcessor
{
public:

    LSRowImageProcessor(int correction_span = 3)
    {
        nLen = correction_span*2+1;
        indicies = new int[nLen];

        int ndx = 0;
        for (int dist=-correction_span*NUM_CHANS;dist <= correction_span*NUM_CHANS;dist+=NUM_CHANS)
            indicies[ndx++]=dist;
    }

    // generate an electrical cross-talk correction from the lsrow image
    // This must be called before any of the other methods (besides the constructor)
    // if the file pointed to by lsimg_path does not exist, the method returns false
    // and no correction is generated.  If lsimg_path does exist, then a correction
    // is generated and the method returns true
    ChannelXTCorrection *GenerateCorrection(const char *lsimg_path);

    ~LSRowImageProcessor()
    {
        delete [] indicies;
    }

private:

    bool GenerateGroupCorrection(int group_num,float *vect_output,int rows,int cols,uint16_t *img);
    void AccumulateMatrixData(double *lhs,double *rhs,double *amat,double bval);

    int nLen;
    int *indicies;
};


#endif  // LSROWIMAGEPROCESSOR_H 



