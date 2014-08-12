/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef COMPARATORNOISECORRECTOR_H
#define COMPARATORNOISECORRECTOR_H


#ifdef BB_DC
typedef short int Mask;

typedef struct RawImage {
	int rows;
	int cols;
};

#else
#include "Mask.h"
#include "RawImage.h"
#include "RandSchrange.h"
#include <stddef.h>
#endif

#define MAX_CNC_THREADS 256
#define MAX_CNC_PCA_ITERS 40

class ComparatorNoiseCorrector
{
public:
    char *AllocateStructs(int threadNum, int _rows, int _cols, int _frames);
    void FreeStructs(int threadNum, bool force=false, char *ptr=NULL);
    void CorrectComparatorNoise(RawImage *raw,Mask *mask,bool verbose,bool aggressive_corection = false,
                                bool beadfind_image = false, int threadNum=-1);
    /**
     * @warning - if mask is NULL and data in image has been dc offset, CNC will think that all the
     * data is pinned low and zero everything out.
     */
    void CorrectComparatorNoise(short *image, int rows, int cols, int frames,
    		Mask *mask,bool verbose,bool aggressive_correction, bool beadfind_image = false,
    		int threadNum=-1);
    void CorrectComparatorNoiseThumbnail(RawImage *raw,Mask *mask, int regionXSize, int regionYSize, bool verbose);
    void CorrectComparatorNoiseThumbnail(short *image, int rows, int cols, int frames, Mask *mask, int regionXSize, int regionYSize, bool verbose);
	void justGenerateMask(RawImage *raw, int threadNum);

    ComparatorNoiseCorrector() {
      mComparator_sigs = NULL;
      mComparator_noise = NULL;
      mComparator_hf_noise = NULL;
      mCorrection = NULL;
      mSigsSize = 0;
      NNSpan = 1;
      mMaskGenerated=0;
      rows=0;
      cols=0;
      frames=0;
      image=NULL;
      initVars();
    }

    ~ComparatorNoiseCorrector() {
    }

protected:
   void CorrectComparatorNoise_internal(bool verbose,
		   bool aggressive_correction,int row_start=-1,int row_end=-1);

private:
   int  DiscoverComparatorPhase(float *psigs,int n_comparators,int nframes, bool hfonly);
   int  DiscoverComparatorPhase_tn(float *psigs,int n_comparators,int nframes, bool hfonly);
    void NNSubtractComparatorSigs(float *pnn,float *psigs,int *mask,int span,int n_comparators,int nframes,float *hfnoise=NULL);
    void HighPassFilter(float *pnn,int n_comparators,int nframes,int span);
    void CalcComparatorSigRMS(float *prms,float *pnn,int n_comparators,int nframes);
    void MaskAbove90thPercentile(int *mask,float *prms,int n_comparators);
    void MaskIQR(int *mask,float *prms,int n_comparators, bool verbose=false);
    void MaskUsingDynamicStdCutoff(int *mask,float *prms,int n_comparators, float std_mult, bool verbose=false);
    void GetPrincComp(float *pcomp,float *pnn,int *mask,int n_comparators,int nframes);
    void FilterUsingPrincComp(float *pnn,float *pcomp,int n_comparators,int nframes);
	void GenerateIntMask(Mask *_mask);

    void GenerateMask(float *mask=NULL);
    void SumColumns(int row_start, int row_end);
    void SetMeanToZero(float *inp);
    void ApplyCorrection(int  phase, int row_start, int row_end, short int *correction);
    void TransposeData();
    void BuildCorrection(bool hfonly);
    
    // list of allocated structures
    float *mComparator_sigs; // [cols*frames*4];
    int    mComparator_sigs_len; // [cols*frames*4];
    float *mComparator_noise; // [cols*frames*2];
    int    mComparator_noise_len; // [cols*frames*2];
    float *mComparator_hf_noise; // [cols*frames*2];
    int    mComparator_hf_noise_len; // [cols*frames*2];
    float *mComparator_rms; //[cols*2];
    int    mComparator_rms_len; //[cols*2];
    int   *mComparator_mask; //[cols*2];
    int    mComparator_mask_len; //[cols*2];
    float *mComparator_hf_rms; //[cols*2];
    int    mComparator_hf_rms_len; //[cols*2];
    int   *mComparator_hf_mask; //[cols*2];
    int    mComparator_hf_mask_len; //[cols*2];
    float *mPcomp; //[frames];
    int    mPcomp_len; //[frames];
    float *mAvg_num; //[cols*frames*4];
    int    mAvg_num_len; //[cols*frames*4];
    short int *mCorrection; // [cols*frames*2]
    int        mCorrection_len; // [cols*frames*2]
    float   *mMask; // [cols*rows];
    int    mMask_len; // [cols*rows];
#define ALLIGN_LEN(a) (((a) & ~(32-1))?(((a)+32)& ~(32-1)):(a))
    void initVars()
    {
        mComparator_sigs = NULL; // [cols*frames*4];
        mComparator_sigs_len = ALLIGN_LEN(cols*frames*4*sizeof(mComparator_sigs[0])); // [cols*frames*4];
        mComparator_noise = NULL; // [cols*frames*2];
        mComparator_noise_len = ALLIGN_LEN(cols*frames*2*sizeof(mComparator_sigs[0])); // [cols*frames*2];
        mComparator_hf_noise = NULL; // [cols*frames*2];
        mComparator_hf_noise_len = ALLIGN_LEN(cols*frames*2*sizeof(mComparator_hf_noise[0])); // [cols*frames*2];
        mComparator_rms = NULL; //[cols*2];
        mComparator_rms_len = ALLIGN_LEN(cols*2*sizeof(mComparator_rms[0])); //[cols*2];
        mComparator_mask = NULL; //[cols*2];
        mComparator_mask_len = ALLIGN_LEN(cols*2*sizeof(mComparator_mask[0])); //[cols*2];
        mComparator_hf_rms = NULL; //[cols*2];
        mComparator_hf_rms_len = ALLIGN_LEN(cols*2*sizeof(mComparator_hf_rms[0])); //[cols*2];
        mComparator_hf_mask = NULL; //[cols*2];
        mComparator_hf_mask_len = ALLIGN_LEN(cols*2*sizeof(mComparator_hf_mask[0])); //[cols*2];
        mPcomp = NULL; //[frames];
        mPcomp_len = ALLIGN_LEN(frames*sizeof(mPcomp[0])); //[frames];
        mAvg_num = NULL; //[cols*frames*4];
        mAvg_num_len = ALLIGN_LEN(cols*frames*4*sizeof(mAvg_num[0])); //[cols*frames*4];
        mCorrection = NULL; // [cols*frames*2]
        mCorrection_len = ALLIGN_LEN(cols*frames*2*sizeof(mCorrection[0])); // [cols*frames*2]
        mMask = NULL; // [cols*rows];
        mMask_len = ALLIGN_LEN(cols*rows*sizeof(mMask[0])); // [cols*rows];

    }

    short int *image;
    int rows;
    int cols;
    int frames;
    RandSchrange mRand;
    int mMaskGenerated;


    //    RandSchrange mRand;
    int mSigsSize;
    int NNSpan;

    static char *mAllocMem[MAX_CNC_THREADS];
    static int   mAllocMemLen[MAX_CNC_THREADS];
};

#endif // COMPARATORNOISECORRECTOR_H
