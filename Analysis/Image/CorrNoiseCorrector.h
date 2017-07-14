/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef CORRNOISECORRECTOR_H
#define CORRNOISECORRECTOR_H

#include <stddef.h>

#ifdef BB_DC
#include "ComparatorNoiseCorrector.h"
//typedef short int Mask;
//
//typedef struct  {
//	int rows;
//	int cols;
//	short int *image;
//	int frames;
//}RawImage;

#else
#include "Mask.h"
#include "RawImage.h"
#endif

#define MAX_RNC_THREADS 256

class CorrNoiseCorrector
{
public:
    char *AllocateStructs(int threadNum, int _rows, int _cols, int _frames);
    void FreeStructs(int threadNum, bool force=false, char *ptr=NULL);
    double CorrectCorrNoise(RawImage *raw, int correctRows, int thumbnail, bool override=false,
    		bool verbose=false, int threadNum=-1, int avg=0);
    /**
     * @warning - if mask is NULL and data in image has been dc offset, CNC will think that all the
     * data is pinned low and zero everything out.
     */
    double CorrectCorrNoise(short *image, int rows, int cols, int frames, int correctRows,
    		int thumbnail, bool overrride=false, bool verbose=false, int threadNum=-1,
    		int avg=0, int frame_mult=0);

    CorrNoiseCorrector() {
      mCorr_sigs = NULL;
      mCorr_noise = NULL;
//      mCorrection = NULL;
      NNSpan = 1;
      rows=0;
      cols=0;
      frames=0;
      ncomp=4;
      image=NULL;
      sumTime=0;
      applyTime = 0;
      tm1=0;
      tm2=0;
      tm2_1=0;
      tm2_2=0;
      tm2_3=0;
      nnsubTime=0;
      mskTime=0;
      allocTime=0;
      maskTime=0;
      mainTime=0;
      aggTime=0;
      totalTime=0;
      CorrLen=0;
      thumbnail=0;
      CorrAvg=0;
      fmult=1;
      initVars();
    }

    ~CorrNoiseCorrector() {
    }

protected:
   double CorrectRowNoise_internal(bool verbose, int correctRows);

private:
    void NNSubtractComparatorSigs(int row_span, int time_span, int correctRows);
    void SumRows();
    void SumCols();
    void SetMeanToZero();
    double ApplyCorrection_rows();
    void ApplyCorrection_cols();
    void DebugSaveRowNoise(int correctRows);
    void DebugSaveComparatorSigs(int correctRows);
    void smoothRowAvgs(float weight);
    void FixouterPixels();
    void ReZeroPinnedPixels_cpFirstFrame();

    // list of allocated structures
    float *mCorr_sigs; // [cols*frames*4];
    int    mCorr_sigs_len; // [cols*frames*4];
    float *mCorr_noise; // [cols*frames*4];
    int    mCorr_noise_len; // [cols*frames*4];
    short int *mCorr_mask;
    int        mCorr_mask_len;
//    short int *mCorrection; // [cols*frames*2]
//    int        mCorrection_len; // [cols*frames*2]
#define ALLIGN_LEN(a) (((a) & ~(32-1))?(((a)+32)& ~(32-1)):(a))
    void initVars()
    {
        mCorr_sigs = NULL; // [cols*frames*4];
        mCorr_sigs_len = ALLIGN_LEN(CorrLen*frames*ncomp*sizeof(mCorr_sigs[0])); // [cols*frames*4];
        mCorr_noise = NULL; // [cols*frames*4];
        mCorr_noise_len = ALLIGN_LEN(CorrLen*frames*ncomp*sizeof(mCorr_sigs[0])); // [cols*frames*4];
//        mCorrection = NULL; // [cols*frames*4]
//        mCorrection_len = ALLIGN_LEN(CorrLen*frames*4*sizeof(mCorrection[0])); // [cols*frames*4]
    }

    short int *image;
    int rows;
    int cols;
    int frames;
    int ncomp;
    int CorrLen;
    int fmult; // number of frames to skip

    double sumTime;
    double applyTime;
    double tm1;
    double tm2;
    double tm2_1;
    double tm2_2;
    double tm2_3;
    double nnsubTime;
    double mskTime;
    double allocTime;
    double maskTime;
    double mainTime;
    double aggTime;
    double totalTime;
	uint64_t CorrAvg;

    int NNSpan;
    int thumbnail;

    static char *mAllocMem[MAX_RNC_THREADS];
    static int   mAllocMemLen[MAX_RNC_THREADS];
};

#endif // CORRNOISECORRECTOR_H
