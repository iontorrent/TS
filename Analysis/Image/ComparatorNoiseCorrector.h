/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef COMPARATORNOISECORRECTOR_H
#define COMPARATORNOISECORRECTOR_H

#include "Mask.h"
#include "RawImage.h"

class ComparatorNoiseCorrector
{
public:
    void CorrectComparatorNoise(RawImage *raw,Mask *mask, bool verbose);
    void CorrectComparatorNoiseThumbnail(RawImage *raw,Mask *mask, int regionXSize, int regionYSize, bool verbose);
private:
    int  DiscoverComparatorPhase(float *psigs,int *c_avg_num,int n_comparators,int nframes);
    void NNSubtractComparatorSigs(float *pnn,float *psigs,int *mask,int span,int n_comparators,int nframes);
    void CalcComparatorSigRMS(float *prms,float *pnn,int n_comparators,int nframes);
    void MaskAbove90thPercentile(int *mask,float *prms,int n_comparators);
    void MaskIQR(int *mask,float *prms,int n_comparators, bool verbose=false);
};

#endif // COMPARATORNOISECORRECTOR_H
