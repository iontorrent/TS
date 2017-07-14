/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef ADVCOMPR_H
#define ADVCOMPR_H

#include <malloc.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#ifdef WIN32
#include "deInterlace.h"
#define AdvComprPrintf printf
#else
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <float.h>
#ifndef BB_DC
#include "ByteSwapUtils.h"
#include "Image.h"
#define AdvComprPrintf printf
#else
#include "datacollect_global.h"
#define AdvComprPrintf DTRACE
#endif
#include "ComparatorNoiseCorrector.h"
#include "CorrNoiseCorrector.h"
#include "datahdr.h"
#include "deInterlace.h"
#endif
#include "Vecs.h"

#define ADVCOMPR_KEY 0xebad
// Block header for a AdvCompr file
typedef struct{
	uint32_t key;

	uint16_t ver;
	uint16_t unused;
	uint16_t nBasisVec;
	uint16_t npts;

	uint32_t datalength;


	uint16_t rstart;
	uint16_t rend;
	uint16_t cstart;
	uint16_t cend;

}AdvComprHeader_t; // 128 bits


// Timing block for debug.
// passed to all functions for timing analysis
typedef struct {
	double Extract;
	double populate;
	double mean;
	double xtalk;
	double ccn;
	double rcn;
	double clear;
	double subtract;
	double CompressBlock;
	double createSample;
	double write;
	double getBits;
	double PackBits;
	double UnPackBits;
	double overall;
    double inCompress;
    double SubtractMean;
    double SetMeanToZero;
    double extractVect;
        
}AdvCompr_Timing_t;


#define ADVC_MAX_REGIONS 98
#define BLK_SZ_X (16*18) // must be a multiple of 16 for avx
#define BLK_SZ_Y (16*18) // must be a multiple of 16 for avx
#define BLK_SZ (BLK_SZ_X*BLK_SZ_Y)

#define TRCS_ACCESS(nt,np) (trcs[(np)*ntrcsL+(nt)])
#define TRC_COEFF_ACC(nv,tr) (trcs_coeffs[(nv)*ntrcs+(tr)])
#define COEFF_ACCESS(npt,nvec) (basis_vectors[(nvec)*npts+(npt)])

class AdvCompr{
public:

	AdvCompr(FILE_HANDLE _fd, short int *raw, int _w, int _h, int _nImgPts,
			int _nUncompImgPts, int *_timestamps_in, int *_timestamps_out,
			int _testUncompr, char *fname, char *options, int frameRate=15,
			int RemoveRowNoise=1);
	~AdvCompr();

	/*  un-compress the image into memory */
	int  UnCompress(int threadNum=-1);

	/* compress the image to a file */
	int  Compress(int threadNum=-1, uint32_t region=0, int targetAvg=8192, int timeTransform=1);

    void SetDoCNC(bool value) { do_cnc = value; }
    static void SetGain(int region, int w, int h, float conv, uint16_t *gainPtr);
    static void ClearGain(int region);
    static float *ReadGain(int region, uint32_t cols, uint32_t rows, char *srcPath);
    static void WriteGain(int region, int w, int h, char *destPath);
    static float *ReSetGain(int region, int w, int h, float *gainPtr);
    static void xtalkCorrect_raw(float xtalk_fraction, int w, int h, int npts, short int *raw);

private:
        AdvCompr() {} // no default constructor
	double AdvCTimer();
	void Write(void *buf, int len);
	void DumpTiming(const char *label);
	void DumpDetailedTiming(const char *label);
	void ClearVarStates(int lnvects);
	void Write(FILE_HANDLE fd, void *buf, int len);
	int  Read(FILE_HANDLE fd, void *buf, int len);
	void UnPackBits(uint64_t *trcs_coeffs_buffer, uint64_t *bufferEnd);
	float findt0(int *idx);
	int   PopulateTraceBlock_ComputeMean();
	void  ComputeMeanTrace();
	int   CompressBlock();
	void  SubtractMeanFromTraces();
	int   GetBitsNeeded();
	uint64_t PackBits(uint64_t *buffer, uint64_t *bufferEnd);
	int   SaveResidual(int add);
	void  SaveRawTrace();
	int   WriteTraceBlock();
	void  ComputeGain();
	void  SetMeanOfFramesToZero_SubMean(int earlyFrames);
	void  doTestComprPre();
	void  doTestComprPost();
	int   ExtractTraceBlock();
	void  ParseOptions(char *options);
	void  CreateSampleMatrix();
	float ExtractVectors();
	void  NormalizeVectors();
	void  ComputeEmphasisVector(float *gv, float mult, float adder, float width);
	void  AddDcOffsetToMeanTrace();
	void  AddSavedMeanToTrace(float *saved_mean);
	void  SetMeanTraceToZero();
	void  smoothBeginning();
	void  adjustForDrift();
	void  ApplyGain_FullChip(float targetAvg);
	void  ComputeGain_FullChip();
	void  ZeroBelow(float thresh);
	void  NeighborSubtract(short int *raw, int h, int w, int npts, float *saved_mean);
	void  ZeroTraceMean(short int *raw, int h, int w, int npts);
	void  TimeTransform(int *timestamps_compr, int *newtimestamps_compr,
			int frameRate, int &tmp_npts, int &tmp_nUncompImgPts);
	void TimeTransform_trace(int npts, int npts_newfr, int *timestamps_compr,
			int *timestamps_newfr, float *vectors, float * nvectors, int nvect);
    void xtalkCorrect(float xtalk_fraction);

    void SmoothMean();
    void ClearBeginningOfTrace();
    void PostClearFrontPorch();
    static float *AllocateGainCorrection(uint32_t region, uint32_t len);
    void findPinned();
    void ApplyGain_FullChip_xtalkcorr(float xtalk_fraction, int w, int h, int npts,
    		short unsigned int *raw);
    void ApplyGain_FullChip_xtalkcorr_sumRows(float xtalk_fraction,
    		int w, int h, int npts, short unsigned int *raw);

    void NNSubtractComparatorSigs(int span);
    void NNSubtractComparatorSigs_tn(int span);



	int nPcaBasisVectors;       // number of pca basis vectors
	int nPcaFakeBasisVectors;   // number of made up pca basis vectors
	int nSplineBasisVectors;    // number of spline basis vectors
	int nPcaSplineBasisVectors; // number of pca basis vectors through spline code
	int nDfcCoeff;              // number of dfc coefficients

	int nBasisVectors;          // sum of all the above coefficients
	char *inp_options;              // passed in options arguments

	short int *raw;             // raw image
	float *trcs;                // raw image block loaded into floats, mean subtracted and zeroed
	float *trcs_coeffs;         // coefficients per trace
	int   *trcs_state;          // pinned mask -1, as well as number of coeffs per trace later
	float *basis_vectors;       // pca/spline basis vectors
	float *mean_trc;            // mean trace, subtracted from every trace
    bool do_cnc;                // should cnc correction be peformed.
    bool do_rnc;                // should rnc correction be peformed.
	int w;
	int h;
	int ntrcs;            // num trcs in the block
	int ntrcsL;           // a rounded version of ntrcs
	int npts;             // number of frames in the traces
	int nUncompImgPts;    // un-vfc number of frames in the trace
	int t0est;            // estimate of t0 in frames.
	int t0estIdx;         // frame index for t0
    int doGain;           // internal flag used for controling gain correction
    int pinnedTrcs;       // number of pinned trcs

    int frameRate;        // incomming data frame rate
    int npts_newfr;
    int nUncompImgPts_newfr;
    int *timestamps_newfr;
    int timestamps_newfr_len;


	float *minVals;       // coefficient minimum values
	float *maxVals;       // coefficient maximum values
	int   *bitsNeeded;    // number of bits to use when saving a coefficient
	float *sampleMatrix;  // sample matrix for pca algorithm
	uint32_t nSamples;    // size of the sample matrix

	int  *timestamps_compr;   // vfc compressed timestamps
	int  *timestamps_uncompr; // un-vfc compressed timestamps
	int testType;
	int cblocks;              // number of x-blocks
	int rblocks;              // number of y-blocks
	char SplineStrategy[128]; // spline options
	struct _file_hdr FileHdrM;// main dat file hdr
	_expmt_hdr_v4 FileHdrV4;  // sub  dat file hdr
	char fname[1024];         // file name to save
	FILE_HANDLE fd;           // fd to save to
	AdvComprHeader_t hdr;
	float tikhonov_k;
	int spline_order;
	int timeTransform; // set to 1 if timetransform is requested
	int CorrectRowNoise;      // set to 1 for row noise correction

	float *gainCorr; // set to the correct region
	char *GblMemPtr;
	uint32_t GblMemLen;
	int ThreadNum;
	float *mMask;
	uint32_t mMask_len;
    float *mCorr_sigs; // [cols*frames*4];
    float *mCorr_noise; // [cols*frames*4];

	uint32_t trcs_len;
	uint32_t trcs_coeffs_len;
	uint32_t basis_vectors_len;
	uint32_t mean_trc_len;
	uint32_t trcs_state_len;
	uint32_t minVals_len;
	uint32_t maxVals_len;
	uint32_t bitsNeeded_len;
	uint32_t sampleMatrix_len;
    uint32_t mCorr_sigs_len;
    uint32_t mCorr_noise_len;

#define TARGET_SAMPLE_RATE 10
#define TARGET_MIN_SAMPLES 800

#define SETLEN(vn,ln,l,t) ln = (l)*sizeof(t) + VEC8F_SIZE_B; ln &= ~(VEC8F_SIZE_B-1); memLen += ln;
#define ALLOC_PTR(vn,ln,vt) vn = (vt *)memPtr; memPtr += ln;
	void ALLOC_STRUCTURES(int nv)
	{
		uint32_t memLen=VEC8F_SIZE_B; // for initial alignment
		SETLEN(trcs,trcs_len,(ntrcs)*(npts),float);
		SETLEN(trcs_coeffs,trcs_coeffs_len,(ntrcs)*nv,float);
		SETLEN(basis_vectors,basis_vectors_len,nv*(npts),float);
		SETLEN(mean_trc,mean_trc_len,(npts),float);
		SETLEN(trcs_state,trcs_state_len,(ntrcs),int);
		SETLEN(minVals,minVals_len,nv,float);
		SETLEN(maxVals,maxVals_len,nv,float);
		SETLEN(bitsNeeded,bitsNeeded_len,nv,int);
		SETLEN(mMask,mMask_len,w*h,float);
		SETLEN(timestamps_newfr,timestamps_newfr_len,npts*2,int);
		SETLEN(mCorr_sigs,mCorr_sigs_len,2*h*npts,float);
		SETLEN(mCorr_noise,mCorr_noise_len,2*h*npts,float);

//		int sample_rate = TARGET_SAMPLE_RATE;//ntrcsL/TARGET_SAMPLES;
//		if(sample_rate < 1)
//			sample_rate=1;
		nSamples = (ntrcsL/TARGET_SAMPLE_RATE)+1;
		if(nSamples < TARGET_MIN_SAMPLES)
		{
			nSamples = (ntrcsL > TARGET_MIN_SAMPLES)?TARGET_MIN_SAMPLES:ntrcsL;
		}

		nSamples &= ~(0x1f);// align nSamples to make it easy to work on...

		SETLEN(sampleMatrix,sampleMatrix_len,(nSamples*npts),float);

		// memLen now contains the amount of memory we need
		if(ThreadNum >= 0 && ThreadNum < ADVC_MAX_REGIONS)
		{
			if (threadMem[ThreadNum] == NULL || threadMemLen[ThreadNum] < memLen)
			{
				if(threadMem[ThreadNum])
					free(threadMem[ThreadNum]);
				threadMem[ThreadNum]=(char *)malloc(memLen);
				threadMemLen[ThreadNum] = memLen;
//				printf("allocating %d bytes to thread %d (%d %d %d %d %d %d %d %d %d)\n",
//						memLen,ThreadNum,
//						trcs_len,trcs_coeffs_len,basis_vectors_len,mean_trc_len,trcs_state_len,
//						minVals_len,maxVals_len,bitsNeeded_len,sampleMatrix_len);
			}
			GblMemPtr=threadMem[ThreadNum];
			GblMemLen=memLen;
		}
		else
		{
			if(GblMemLen<memLen)
			{
				if(GblMemPtr)
					free(GblMemPtr);
				GblMemPtr=(char *)malloc(memLen);
				GblMemLen=memLen;
//				printf("allocating %d bytes to thread %d (%d %d %d %d %d %d %d %d %d) nsamples=%d\n",
//						memLen,ThreadNum,
//						trcs_len,trcs_coeffs_len,basis_vectors_len,mean_trc_len,trcs_state_len,
//						minVals_len,maxVals_len,bitsNeeded_len,sampleMatrix_len,nSamples);
			}
		}
                // this gives us aligned memory without calling memalign() which is problematic on win32
		char *memPtr=(char *)((((uint64_t)GblMemPtr) + VEC8F_SIZE_B) & ~(VEC8F_SIZE_B-1));

		ALLOC_PTR(trcs,trcs_len,float);
		ALLOC_PTR(trcs_coeffs,trcs_coeffs_len,float);
		ALLOC_PTR(basis_vectors,basis_vectors_len,float);
		ALLOC_PTR(mean_trc,mean_trc_len,float);
		ALLOC_PTR(trcs_state,trcs_state_len,int);
		ALLOC_PTR(minVals,minVals_len,float);
		ALLOC_PTR(maxVals,maxVals_len,float);
		ALLOC_PTR(bitsNeeded,bitsNeeded_len,int);
		ALLOC_PTR(sampleMatrix,sampleMatrix_len,float);
		ALLOC_PTR(mMask,mMask_len,float);
		ALLOC_PTR(timestamps_newfr,timestamps_newfr_len, int);
		ALLOC_PTR(mCorr_sigs,mCorr_sigs_len, float);
		ALLOC_PTR(mCorr_noise,mCorr_noise_len, float);
	}

    void FREE_STRUCTURES(bool doFree)
    {
		trcs=NULL; trcs_len=0;
		trcs_coeffs=NULL; trcs_coeffs_len=0;
		basis_vectors=NULL; basis_vectors_len=0;
		mean_trc=NULL; mean_trc_len=0;
		trcs_state=NULL; trcs_state_len=0;
		minVals=NULL; minVals_len=0;
		maxVals=NULL; maxVals_len=0;
		bitsNeeded=NULL; bitsNeeded_len=0;
		sampleMatrix=NULL; sampleMatrix_len=0;
		mMask=NULL; mMask_len=0;
		mCorr_sigs=NULL; mCorr_sigs_len=0;
		mCorr_noise=NULL; mCorr_noise_len=0;
//		printf("freeing thread %d doFree=%d\n",ThreadNum,doFree);

		if(doFree || !(ThreadNum >= 0 && ThreadNum < ADVC_MAX_REGIONS))
		{
			free(GblMemPtr);
			GblMemPtr=NULL;
			if(ThreadNum >= 0 && ThreadNum < ADVC_MAX_REGIONS)
			{
				threadMem[ThreadNum]=NULL;
				threadMemLen[ThreadNum] = 0;
			}

		}

    }

    void CLEAR_STRUCTURES()
    {
//		v8f *memPtr=(v8f *)((((uint64_t)GblMemPtr) + VEC8F_SIZE_B) & ~(VEC8F_SIZE_B-1));
//		v8f zeroV=LD_VEC8F(0.0f);
//		uint32_t memLen=(GblMemLen - VEC8F_SIZE_B) & ~((VEC8F_SIZE_B)-1);
//
//		memLen /= VEC8F_SIZE_B;
//
//		for(uint32_t cnt=0;cnt<memLen;cnt++)
//		{
//			memPtr[cnt] = zeroV;
//		}


//		memset(trcs,0,trcs_len);
//		memset(trcs_coeffs,0,trcs_coeffs_len);
//		memset(basis_vectors,0,basis_vectors_len);
//		memset(mean_trc,0,mean_trc_len);
//		memset(trcs_state,0,trcs_state_len);
//		memset(minVals,0,minVals_len);
//		memset(maxVals,0,maxVals_len);
//		memset(bitsNeeded,0,bitsNeeded_len);
//		memset(sampleMatrix,0,sampleMatrix_len);
    }

	AdvCompr_Timing_t timing;

	static float *gainCorrection[ADVC_MAX_REGIONS];
	static uint32_t gainCorrectionSize[ADVC_MAX_REGIONS];
	static char *threadMem[ADVC_MAX_REGIONS];
	static uint32_t threadMemLen[ADVC_MAX_REGIONS];
};

#ifndef WIN32
#ifndef BB_DC
// handles test compression/uncompression in Analysis
int AdvComprTest(const char *fname, Image *image, char *options, bool do_cnc=true);
#endif
#endif

// handles loading windows inside the image
int AdvComprUnCompress(char *fname, short int *raw, int w, int h, int nImgPts,
		int *_timestamps, int startFrame, int endFrame, int mincols,
		int minrows, int maxcols, int maxrows, int ignoreErrors);



#endif // ADVCOMPR_H
