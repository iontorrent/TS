/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef COMPARATORNOISECORRECTOR_H
#define COMPARATORNOISECORRECTOR_H

#include "Mask.h"
#include "RawImage.h"
#include "RandSchrange.h"

class ComparatorNoiseCorrector
{
public:
    void CorrectComparatorNoise(RawImage *raw,Mask *mask, bool verbose,bool aggressive_corection = false, bool beadfind_image = false);
    void CorrectComparatorNoise(short *image, int rows, int cols, int frames, Mask *mask, bool verbose,bool aggressive_correction, bool beadfind_image = false);
    void CorrectComparatorNoiseThumbnail(RawImage *raw,Mask *mask, int regionXSize, int regionYSize, bool verbose);
    void CorrectComparatorNoiseThumbnail(short *image, int rows, int cols, int frames, Mask *mask, int regionXSize, int regionYSize, bool verbose);

    ComparatorNoiseCorrector() {
      mComparator_sigs = NULL;
      mComparator_noise = NULL;
      mComparator_hf_noise = NULL;
      mSigsSize = 0;
      NNSpan = 1;
    }
    ~ComparatorNoiseCorrector() {
      FreeMem();
    }

    void FreeMem() {
      if (mComparator_sigs != NULL) {
        delete [] mComparator_sigs;
        delete [] mComparator_noise;
        delete [] mComparator_hf_noise;
        mSigsSize = 0;
      }
    }
    void Allocate(int size) {
      if (size > mSigsSize || mComparator_sigs == NULL) {
        FreeMem();
        mComparator_sigs = new float[size];
        mComparator_noise = new float[size/2];
        mComparator_hf_noise = new float[size/2];
      }
    }

protected:
   void CorrectComparatorNoise_internal(short *image, int rows, int cols, int frames, Mask *mask, bool verbose,bool aggressive_correction,int row_start=-1,int row_end=-1);

private:
    int  DiscoverComparatorPhase(float *psigs,int *c_avg_num,int n_comparators,int nframes);
    void NNSubtractComparatorSigs(float *pnn,float *psigs,int *mask,int span,int n_comparators,int nframes,float *hfnoise=NULL);
    void HighPassFilter(float *pnn,int n_comparators,int nframes,int span);
    void CalcComparatorSigRMS(float *prms,float *pnn,int n_comparators,int nframes);
    void MaskAbove90thPercentile(int *mask,float *prms,int n_comparators);
    void MaskIQR(int *mask,float *prms,int n_comparators, bool verbose=false);
    void MaskUsingDynamicStdCutoff(int *mask,float *prms,int n_comparators, float std_mult, bool verbose=false);
    void GetPrincComp(float *pcomp,float *pnn,int *mask,int n_comparators,int nframes);
    void FilterUsingPrincComp(float *pnn,float *pcomp,int n_comparators,int nframes);
    float *mComparator_sigs;
    float *mComparator_noise;
    float *mComparator_hf_noise;
    RandSchrange mRand;
    int mSigsSize;
    int NNSpan;
};

#endif // COMPARATORNOISECORRECTOR_H
