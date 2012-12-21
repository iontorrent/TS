/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef COMPARATORNOISECORRECTOR_H
#define COMPARATORNOISECORRECTOR_H

#include "Mask.h"
#include "RawImage.h"

class ComparatorNoiseCorrector
{
public:
    void CorrectComparatorNoise(RawImage *raw,Mask *mask, bool verbose);
    void CorrectComparatorNoise(short *image, int rows, int cols, int frames, Mask *mask, bool verbose);
    void CorrectComparatorNoiseThumbnail(RawImage *raw,Mask *mask, int regionXSize, int regionYSize, bool verbose);
    
void CorrectComparatorNoiseThumbnail(short *image, int rows, int cols, int frames, Mask *mask, int regionXSize, int regionYSize, bool verbose);
    ComparatorNoiseCorrector() {
      mComparator_sigs = NULL;
      mComparator_noise = NULL;
      mSigsSize = 0;
    }
    ~ComparatorNoiseCorrector() {
      FreeMem();
    }

    void FreeMem() {
      if (mComparator_sigs != NULL) {
        delete [] mComparator_sigs;
        delete [] mComparator_noise;
        mSigsSize = 0;
      }
    }
    void Allocate(int size) {
      if (size > mSigsSize || mComparator_sigs == NULL) {
        FreeMem();
        mComparator_sigs = new float[size];
        mComparator_noise = new float[size/2];
      }
    }

private:
    int  DiscoverComparatorPhase(float *psigs,int *c_avg_num,int n_comparators,int nframes);
    void NNSubtractComparatorSigs(float *pnn,float *psigs,int *mask,int span,int n_comparators,int nframes);
    void CalcComparatorSigRMS(float *prms,float *pnn,int n_comparators,int nframes);
    void MaskAbove90thPercentile(int *mask,float *prms,int n_comparators);
    void MaskIQR(int *mask,float *prms,int n_comparators, bool verbose=false);
    float *mComparator_sigs;
    float *mComparator_noise;
    int mSigsSize;
};

#endif // COMPARATORNOISECORRECTOR_H
