/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <float.h>
#include <sys/time.h>
#include <string.h>
#include <time.h>

#include "Utils.h"
#include "CorrNoiseCorrector.h"
#include "deInterlace.h"
#include "Vecs.h"
#ifdef __AVX__
#include <xmmintrin.h>
#include "immintrin.h"
#endif
#ifndef BB_DC
#include "ChipIdDecoder.h"
#else
#include "datacollect_global.h"
#endif
//#define DBG_SAVETEMPS 1
//#define USE_ONE_AVERAGE 1
//#define SMOOTH_NNAVG 5
//#define DBG_PRINT_TIMES 1
//#define SCALE_TRACES 1
//#define AVG_END_FRAMES 4
//#define AVG_START_FRAMES 4
#define THROW_AWAY_START_FRAMES 1
//#define LOOSE_DC_OFFSET 1

#ifdef DBG_SAVETEMPS
#include "crop/Acq.h"
#endif

#define mAvg_num_ACC(idx,comparator)  (mAvg_num[(idx)+((comparator)*CorrLen)])
#define SIGS_ACC(idx,comparator,frame) (mCorr_sigs[((comparator)*CorrLen*frames) + ((frame) *CorrLen) + (idx)])
#define NOISE_ACC(idx,comparator,frame) (mCorr_noise[((comparator)*CorrLen*frames) + ((frame) *CorrLen) + (idx)])
//#define CORR_ACC(idx,comparator,frame) (mCorrection[((comparator)*CorrLen*frames) + ((frame) *CorrLen) + (idx)])


char *CorrNoiseCorrector::mAllocMem[MAX_RNC_THREADS] = {NULL};
int CorrNoiseCorrector::mAllocMemLen[MAX_RNC_THREADS] = {0};


static double CNCTimer()
{
#ifdef WIN32
	double rc=0.0;
#if 0
	LPSYSTEMTIME  t;
	GetLocalTime (t);
    rc =  t->wMinute*60*1000;
	rc += t->wSecond*1000;
	rc += t->wMilliseconds;
#endif
	return (rc);
#else
  struct timeval tv;
  gettimeofday ( &tv, NULL );
  double curT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );
  return ( curT );
#endif
}

#ifndef BB_DC
double CorrNoiseCorrector::CorrectCorrNoise(RawImage *raw, int correctRows, int thumbnail, bool override,
		bool verbose, int threadNum, int avg)
{
	if ((raw->imageState & IMAGESTATE_ComparatorCorrected) == 0)
	  return CorrectCorrNoise(raw->image,raw->rows,raw->cols,raw->frames,correctRows,thumbnail,override, verbose,threadNum,avg);
	return 0;
}
#endif


double CorrNoiseCorrector::CorrectCorrNoise(short *_image, int _rows, int _cols, int _frames, int correctRows,
		int _thumbnail, bool override, bool verbose, int threadNum, int avg, int frame_mult)
{
	char *allocPtr;
	double rc=0;

#ifndef BB_DC
	if( !override && !ChipIdDecoder::IsPtwo() )
		return rc;
#else
	if(!override && eg.ChipInfo.ChipMajorRev < 0x20)
		return rc;
#endif

	double wholeStart,start;
	wholeStart = start = CNCTimer();

	image=_image;
	thumbnail=_thumbnail;

	CorrAvg=(uint64_t)avg;

	rows=_rows;
	cols=_cols;
	frames=_frames;
	if(frame_mult)
		fmult=frame_mult;

	mCorr_mask_len = rows*cols*2;
	mCorr_mask=(short int *)memalign(64,mCorr_mask_len);

    FixouterPixels(); // make sure we don't pin good pixels

	if(correctRows & 1){
	   printf("correcting row correlated noise %s\n",(thumbnail?"thumbnail":"normal"));
       if(thumbnail){
         NNSpan = 50;
         ncomp=12;
       }
       else{
         NNSpan = 500;
         ncomp=1;
       }
	   CorrLen=_rows;
	   allocPtr=AllocateStructs(threadNum, _rows, _cols, _frames);
	   rc = CorrectRowNoise_internal( verbose,1);
	   FreeStructs(threadNum,false,allocPtr);

	}
	if((correctRows & 2) || !correctRows){
		printf("correcting col correlated noise %s\n",(thumbnail?"thumbnail":"normal"));
       if(thumbnail){
         NNSpan = 50;
         ncomp=12;
       }
       else{
         NNSpan = 100;
         ncomp=2;
       }
		CorrLen=_cols;
		allocPtr=AllocateStructs(threadNum, _rows, _cols, _frames);
		CorrectRowNoise_internal( verbose,0);
		FreeStructs(threadNum,false,allocPtr);
	}

	ReZeroPinnedPixels_cpFirstFrame();
	free(mCorr_mask);

   totalTime = CNCTimer()-wholeStart;
#ifdef DBG_PRINT_TIMES
   printf("CNC took %.2lf alloc=%.2f mask=%.2f main=%.2f agg=%.2f sum=%.2f apply=%.2f tm1=%.2f tm2=%.2f(%.2f/%.2f) nn=%.2f msk=%.2f pca=%.2f\n",
		   totalTime,allocTime,maskTime,mainTime,aggTime,sumTime,applyTime,tm1,tm2,tm2_1,tm2_2,nnsubTime,mskTime,tm2_3);
#endif
   return rc;
}

char *CorrNoiseCorrector::AllocateStructs(int threadNum, int _rows, int _cols, int _frames)
{
	int len=0;
	char *allocBuffer=NULL;
	char **allocPtr = &allocBuffer;
	int allocLen=0;
	int *allocLenPtr = &allocLen;
	char *aptr;

	initVars();

	// need to put the vectorizable stuff up front to make sure it's properly alligned
	len += mCorr_sigs_len;
    len += mCorr_noise_len;
//    len += mCorrection_len;

    if(threadNum >= 0 && threadNum < MAX_RNC_THREADS)
    {
    	allocPtr = &mAllocMem[threadNum];
    	allocLenPtr = &mAllocMemLen[threadNum];
    }

	if(*allocLenPtr < len)
	{
		if(*allocPtr)
		{
//			printf("Freeing old memory tn=%d\n",threadNum);
			free(*allocPtr);
		}

//		printf("allocating new memory tn=%d len=%d\n",threadNum,len);
		*allocPtr = (char *)memalign(VEC8F_SIZE_B,len);
		*allocLenPtr = len;
	}
	// check for failed alloc here..

	aptr = *allocPtr; // allocate the single large buffer as needed

	mCorr_sigs = (float *)aptr;  aptr += mCorr_sigs_len;
    mCorr_noise = (float *)aptr; aptr += mCorr_noise_len;
//    mCorrection = (short int *)aptr; aptr += mCorrection_len;

   	return *allocPtr;
}

void CorrNoiseCorrector::FreeStructs(int threadNum, bool force, char *ptr)
{
	if(force || threadNum < 0)
	{
		if(threadNum >= 0)
		{
			free(mAllocMem[threadNum]);
			mAllocMem[threadNum]=NULL;
			mAllocMemLen[threadNum]=0;
		}
		else if(ptr)
		{
			free(ptr);
		}
	}
}

double CorrNoiseCorrector::CorrectRowNoise_internal( bool verbose, int correctRows)
{
	double rc = 0;

  memset(mCorr_sigs,0,mCorr_sigs_len);
  memset(mCorr_noise,0,mCorr_noise_len);

  // first, create the average comparator signals
  // making sure to avoid pinned pixels
  if(correctRows)
	  SumRows();
  else
	  SumCols();
  
  double startTime=CNCTimer();

	  // subtract DC offset from average comparator signals
	  SetMeanToZero();

#ifdef DBG_SAVETEMPS
	DebugSaveComparatorSigs(correctRows);
#endif

  tm1 += CNCTimer()-startTime;
  startTime = CNCTimer();
  double tm2_1_startTime = CNCTimer();

  // now neighbor-subtract the comparator signals
  NNSubtractComparatorSigs(500,1,correctRows);

#ifdef DBG_SAVETEMPS
	DebugSaveRowNoise(correctRows);
#endif

  tm2_1 += CNCTimer()-tm2_1_startTime;

  tm2 += CNCTimer()-startTime;

  if(correctRows){
	  // smooth the average trace, and add diff to all row noise.
//	  smoothRowAvgs(0.5);
	  rc = ApplyCorrection_rows();
  }
  else
	  ApplyCorrection_cols();

  return rc;
}


#ifdef DBG_SAVETEMPS



void CorrNoiseCorrector::DebugSaveComparatorSigs(int correctRows)
{
	  Image loader2;

	  loader2.LoadRaw ( "acq_0000.dat", 0, true, false, false );

	int y,frame;
	int frameStride=rows*cols;
	short *srcPtr;
//	int lw=cols/VEC8_SIZE;

	// now subtract each neighbor-subtracted comparator signal from the
	// pixels that are connected to that comparator

     	for (frame = 0; frame < frames; frame++)
    	{
    		for (y = 0; y < rows; y++)
    		{
    			for(int x = 0; x < cols; x++)
    			{
    				srcPtr = &loader2.raw->image[frame * frameStride + y * cols + x];
    			    if(correctRows)
        				*srcPtr = (short)(8192.0f + (float)SIGS_ACC(y,x/(cols/ncomp),frame));
        			else
        				*srcPtr = (short)(8192.0f + (float)SIGS_ACC(x,y/(rows/ncomp),frame));
    			}
    		}
    	}

        char newName[256];
        sprintf(newName,"acq_0000.dat_oldsig");

        Acq saver;
        saver.SetData ( &loader2 );
        saver.WriteVFC(newName, 0, 0, cols, rows);

}

void CorrNoiseCorrector::DebugSaveRowNoise(int correctRows)
{
	int y,frame;
	int frameStride=rows*cols;
	short *srcPtr;
	float cor;

	  Image loader2;

	  loader2.LoadRaw ( "acq_0000.dat", 0, true, false,false );
	// now subtract each neighbor-subtracted comparator signal from the
	// pixels that are connected to that comparator

	for (frame = 0; frame < frames; frame++)
	{
		for (y = 0; y < rows; y++)
		{
			for(int x = 0; x < cols; x++)
			{
				srcPtr = &loader2.raw->image[frame * frameStride + y * cols + x];
			    if(correctRows)
			    	cor = NOISE_ACC(y,x/(cols/ncomp),frame);
			    else
			    	cor = NOISE_ACC(x,y/(rows/ncomp),frame);

				*srcPtr = (short)(8192.0f + cor);
			}
		}
	}
    char newName[1024];

    sprintf(newName,"acq_0000.dat_oldnoise");

    Acq saver;
    saver.SetData ( &loader2 );
    saver.WriteVFC(newName, 0, 0, cols, rows);

}

#endif


// make sure our applied correction won't pin any currently un-pinned pixels
void CorrNoiseCorrector::FixouterPixels()
{
	int frameStride=rows*cols*fmult;
	int endFrmOffset = (frames-1)*frameStride;
	for (int idx = 0; idx < rows*cols; idx++){
		short int *srcPtr = (short int *) (&image[idx]);
		short int *endPtr = (short int *) (&image[idx+endFrmOffset]);
		if(*srcPtr <= 8 || *endPtr <= 8){ // pinned
			mCorr_mask[idx]=0;
			for (int frame = 0; frame < frames; frame++){
				srcPtr[frame*frameStride] = 0;
			}
		}
		else if(*srcPtr >= (16384-8) || *endPtr >= (16384-8)){
			mCorr_mask[idx]=0;
			for (int frame = 0; frame < frames; frame++){
				srcPtr[frame*frameStride] = 0;
			}
		}
		else{
			mCorr_mask[idx]=1;
			// not pinned
#ifdef LOOSE_DC_OFFSET
			float avg = 0;
			int initial_frames=4;
			for (int frame = 0; frame < initial_frames; frame++){
				avg += srcPtr[frame*frameStride];
			}
			avg /= (float)initial_frames;
			avg -= 8192;

			for (int frame = 0; frame < frames; frame++){
				srcPtr[frame*frameStride] -= (short int)avg;
			}

#else
			if(*srcPtr < 1000 || *endPtr < 1000){
				// shift this pixel up by 1000
				for (int frame = 0; frame < frames; frame++){
					srcPtr[frame*frameStride] += 1000;
				}
			}
			else if(*srcPtr > (16384-1000) || *endPtr > (16384-1000)){
				// shift this pixel down by 1000
				for (int frame = 0; frame < frames; frame++){
					srcPtr[frame*frameStride] -= 1000;
				}
			}
#endif
		}
	}
}

void CorrNoiseCorrector::ReZeroPinnedPixels_cpFirstFrame()
{
	int frameStride=rows*cols*fmult;
#ifdef SCALE_TRACES
	int64_t avg_height=0;
	int64_t avg_cnt=0;
#endif

//	int endFrmOffset = (frames-1)*frameStride;
	for (int idx = 0; idx < rows*cols; idx++){
		short int *srcPtr = (short int *) (&image[idx]);
		if(mCorr_mask[idx] == 0){ // pinned
			// pin the trace
			for (int frame = 0; frame < frames; frame++){
				srcPtr[frame*frameStride]=0;
			}
		}
//		else if(*srcPtr == 16384){
//			// pin the trace
//			// we are going to overwrite the first frame anyway..
//			for (int frame = 0; frame < frames; frame++){
//				srcPtr[frame*frameStride]=16384; // mark this as a pinned pixel
//			}
//		}
		else{
			// not pinned
			// average the first ten frames together for gain correction
#ifdef AVG_END_FRAMES
			int32_t EndAvg = 0;
			const int num_end_avg=AVG_END_FRAMES;

			for(int lidx=1;lidx<=num_end_avg;lidx++){
				EndAvg += srcPtr[(frames-lidx)*frameStride];
			}
			EndAvg /= num_end_avg;
			for(int lidx=1;lidx<=num_end_avg;lidx++){
				srcPtr[(frames-lidx)*frameStride] = EndAvg;
			}
#endif

#ifdef THROW_AWAY_START_FRAMES
			{
				int32_t StartAvg = 0;

				StartAvg = srcPtr[THROW_AWAY_START_FRAMES*frameStride];
				for(int lidx=0;lidx<THROW_AWAY_START_FRAMES;lidx++){
					srcPtr[lidx*frameStride] = StartAvg;
				}
			}
#endif

#ifdef AVG_START_FRAMES
			{
				// average the last four frames together for gain correction
				int32_t StartAvg = 0;
				const int num_start_avg=AVG_START_FRAMES;

				for(int lidx=0;lidx<num_start_avg;lidx++){
					StartAvg += srcPtr[lidx*frameStride];
				}
				StartAvg /= num_start_avg;
				for(int lidx=0;lidx<num_start_avg;lidx++){
					srcPtr[lidx*frameStride] = StartAvg;
				}
			}
#endif
#ifdef SCALE_TRACES
			avg_height += EndAvg-StartAvg;
			avg_cnt++;
#endif
		}
	}
#ifdef SCALE_TRACES
	printf("avg_height=%ld avg_cnt=%ld --> %ld\n",avg_height,avg_cnt,avg_height/avg_cnt);
	avg_height /= avg_cnt;
	avg_height=600;
	for (int idx = 0; idx < rows*cols; idx++){
		short int *srcPtr = (short int *) (&image[idx]);
		if(mCorr_mask[idx] != 0){ // not pinned
			// do gain correction right here...
			float StartAvg = srcPtr[0];
			float EndAvg   = srcPtr[(frames-1)*frameStride];

			float gain = avg_height/(EndAvg-StartAvg);
			for(int frame=(frames-1);frame >= 0; frame--){
				srcPtr[frame*frameStride] = StartAvg + gain * ((float)srcPtr[frame*frameStride]-StartAvg);
			}
		}
	}
#endif
}

// subtract the already computed correction from the image file
double CorrNoiseCorrector::ApplyCorrection_rows()
{
	int y;
	int frameStride=rows*cols*fmult;
	int frameStrideV=(rows*cols/8)*fmult;
	double startTime = CNCTimer();
	double TotalAvgNoiseSq=0;
	double TotalAvgNoiseSum=0;
//	v8s imgAvg=LD_VEC8S((short int)CorrAvg);


	for (y = 0; y < rows; y++){
		v8s *srcPtr = (v8s *) (&image[0 * frameStride + y * cols]);
		int x = 0;

		for (int reg = 0; reg < ncomp; reg++){

			{
				float avgNoiseSq=0;
				float avgNoiseSum = 0;
				for(int frame=0;frame<frames;frame++){
					float corr=NOISE_ACC(y,reg,frame);
					avgNoiseSq +=  corr*corr;
					avgNoiseSum += corr;
				}
				TotalAvgNoiseSq += avgNoiseSq;
				TotalAvgNoiseSum += avgNoiseSum;
			}

#if 0
			float noise_slope=0;
			float noise_offset=NOISE_ACC(y,reg,frame);
			float noise_cnt=0;
			if(reg > 0){
				noise_slope+=(NOISE_ACC(y,reg,frame)-NOISE_ACC(y,reg-1,frame))/(float)(cols/ncomp);
				noise_cnt++;
			}
			if(reg < 3){
				noise_slope+=(NOISE_ACC(y,reg+1,frame)-NOISE_ACC(y,reg,frame))/(float)(cols/ncomp);
				noise_cnt++;
			}
			noise_slope/=noise_cnt;
			for (float weight=-cols/8; weight < cols/8; weight++,x++){
				srcPtr[x] -= noise_offset + weight*noise_slope;
			}
#else
			for(;x<(reg+1)*(cols/ncomp);x+=8,srcPtr++){
//				v8s corr0=LD_VEC8S((short int)NOISE_ACC(y,reg,1));
				for (int frame = 0; frame < frames; frame++)
				{
					v8s corr=LD_VEC8S((short int)(NOISE_ACC(y,reg,frame)));
//					srcPtr[frame*frameStrideV] += imgAvg - srcPtr[1*frameStrideV] - corr + corr0;
//					srcPtr[frame*frameStrideV] = LD_VEC8S((short int)(8192 + SIGS_ACC(y,reg,frame) - NOISE_ACC(y,reg,frame)));
					srcPtr[frame*frameStrideV] -= corr;
				}
			}
#endif
		}
	}

	TotalAvgNoiseSum /= (double)(rows*ncomp*frames);
	TotalAvgNoiseSq /= (double)(rows*ncomp*frames);
//	printf("RTN: sq=%.1lf rms=%.1lf\n",TotalAvgNoiseSq,sqrt(TotalAvgNoiseSq));
	TotalAvgNoiseSq -= TotalAvgNoiseSum*TotalAvgNoiseSum;
	TotalAvgNoiseSq = sqrt(TotalAvgNoiseSq)/sqrt(2);
//	printf("ncomp=%d phase=%d row_start=%d row_end=%d\n",ncomp,phase,row_start,row_end);
	  applyTime += CNCTimer()-startTime;

	  return TotalAvgNoiseSq;
}

// subtract the already computed correction from the image file
void CorrNoiseCorrector::ApplyCorrection_cols()
{
	double startTime = CNCTimer();
	//printf("%s\n",__FUNCTION__);
	for (int frame = 0; frame < frames; frame++)
	{
		for(int y=0 ;y< rows; y++){
			v8f_u *corr=(v8f_u *)&NOISE_ACC(0,0,frame);
			v8s *srcPtr=(v8s *)&image[frame*fmult*rows*cols + y*cols];
			for(int x=0 ;x< cols; x+=8,corr++,srcPtr++){
				v8s_u corrS;
				CVT_VEC8F_VEC8S(corrS,(*corr));
				*srcPtr -= corrS.V;


//			int16_t sum[4]={0,0,0,0};
//			for(int j=0;j<4;j++){
//				sum[j] = NOISE_ACC(y,j,frame);
//			}
//			short int *srcPtr = (short int *) (&image[frame*rows*cols + x]);
//			int y = 0;
//			for (int reg = 0; reg < ncomp; reg++){
//				float noise_slope=0;
//				float noise_offset=NOISE_ACC(x,reg,frame);
//				float noise_cnt=0;
//				if(reg > 0){
//					noise_slope+=(NOISE_ACC(x,reg,frame)-NOISE_ACC(x,reg-1,frame))/(float)(rows/ncomp);
//					noise_cnt++;
//				}
//				if(reg < (ncomp-1)){
//					noise_slope+=(NOISE_ACC(x,reg+1,frame)-NOISE_ACC(x,reg,frame))/(float)(rows/ncomp);
//					noise_cnt++;
//				}
//				noise_slope/=noise_cnt;
//				for (float weight=-(float)rows/((float)ncomp*2.0f); weight < (float)rows/((float)ncomp*2.0f); weight++,y++){
//					srcPtr[y*cols] -= noise_offset + weight*noise_slope;
//				}
			}
		}
	}

//	printf("ncomp=%d phase=%d row_start=%d row_end=%d\n",ncomp,phase,row_start,row_end);
	  applyTime += CNCTimer()-startTime;
}

// sum the columns from row_start to row_end and put the answer in mCorr_sigs
// mMask has a 1 in it for every active column and a zero for every pinned pixel
void CorrNoiseCorrector::SumRows()
{
	int frame,x,y;
	double startTime=CNCTimer();

	int lcols=cols/8;
	for (frame = 0; frame < frames; frame++)
	{
		for (y = 0; y < rows; y++){
			short int *sptr = (short int *) (&image[frame * fmult * cols*rows + y * cols]);
			v8s  *maskSumPtr=(v8s *)&mCorr_mask[y*cols];
			for(int reg=0;reg<ncomp;reg++){
				v8s_u maskSum;
				v8f_u sum;
				v8f_u valU;
				maskSum.V=LD_VEC8S(0);
				sum.V=LD_VEC8F(0);
				for (x = 0; x < lcols/ncomp; x++)
				{
					LD_VEC8S_CVT_VEC8F(sptr,valU);
					sum.V += valU.V;
					maskSum.V+=*maskSumPtr++;
					sptr+=8;
				}
				float avg=0;
				float msksm=0;
				for(int j=0;j<8;j++){
					avg += sum.A[j];
					msksm += maskSum.A[j];
				}
				SIGS_ACC(y,reg,frame) = avg/msksm;
			}
		}
	}


	if(CorrAvg==0){
		for (int reg = 0; reg < ncomp; reg++){
			for (y = 0; y < rows; y+=2){
				CorrAvg += SIGS_ACC(y,reg,0);
			}
		}
		CorrAvg /= (uint64_t)(rows*ncomp/2);
	}

//	printf("AvgSig = ");
//	for (frame = 0; frame < frames; frame++){
//		float AvgSig=0;
//		for (int reg = 0; reg < ncomp; reg++){
//			for (y = 0; y < rows; y++){
//				AvgSig += SIGS_ACC(y,reg,frame);
//			}
//		}
//		AvgSig /= (uint64_t)(rows*ncomp);
//		printf(" %.2f",AvgSig);
//	}
//	printf("\n");


	sumTime += CNCTimer() - startTime;
}

// sum the columns from row_start to row_end and put the answer in mCorr_sigs
// mMask has a 1 in it for every active column and a zero for every pinned pixel
void CorrNoiseCorrector::SumCols()
{
	int frame;
//	int colsV=cols/8;

	double startTime=CNCTimer();

	for (frame = 0; frame < frames; frame++)
	{
		for (int x = 0; x < cols; x+=8){

			v8f_u sum;
			v8f_u valU;


			short int *sptr = (short int *) (&image[frame*fmult*cols*rows + x]);
//			v8s  *maskSumPtr=(v8s *)&mCorr_mask[x];
			for(int reg=0;reg<ncomp;reg++){
//				v8s_u maskSum;
				sum.V=LD_VEC8F(0);
//				maskSum.V=LD_VEC8S(0);

				for (int y = (reg)*(rows/ncomp); y < (reg+1)*(rows/ncomp); y++)
				{
					LD_VEC8S_CVT_VEC8F(sptr,valU);
					sum.V += valU.V;
//					maskSum.V+=*maskSumPtr++;
					sptr+=cols;
//					maskSumPtr += colsV;
				}
				for(int i=0;i<8;i++){
					SIGS_ACC(x+i,reg,frame) = sum.A[i]/(float)(/*maskSum.A[i]*/rows/ncomp);
				}
			}
		}
	}

	sumTime += CNCTimer() - startTime;
}

// sets the mean of the columns averages to zero
void CorrNoiseCorrector::SetMeanToZero()
{
	int frame, comparator;
	float dc;

	// get a pointer to where we will build the comparator signal average
	for (comparator=0;comparator<ncomp;comparator++)
	{
		for (int idx = 0; idx < CorrLen; idx++)
		{
			dc = 0;

			for (frame = 0; frame < frames; frame++)
				dc += SIGS_ACC(idx,comparator,frame);

			dc /= frames;

			// subtract dc offset
			for (frame = 0; frame < frames; frame++)
				SIGS_ACC(idx,comparator,frame) -= dc;
		}
	}

}

void CorrNoiseCorrector::smoothRowAvgs(float weight)
{
	float nn_avg[frames];
	float nn_avg_smooth[frames];
	float nn_avg_noise[frames];
	int span=8;

	// first, get an average of all the corrected row signals
	memset(nn_avg, 0, sizeof(nn_avg));
	memset(nn_avg_smooth, 0, sizeof(nn_avg_smooth));
	memset(nn_avg_noise, 0, sizeof(nn_avg_noise));
	for (int y = 0; y < CorrLen; y++) {
		for(int comp = 0;comp < ncomp; comp++){
			for (int frame = 0; frame < frames; frame++){
				nn_avg[frame] += SIGS_ACC(y,comp,frame);
			}
		}
	}
	for (int frame = 0; frame < frames; frame++){
		nn_avg[frame] /= ncomp*CorrLen;
	}

	// now, smooth into nn_avg_smooth
	//float prev=nn_avg[0];
	for (int frame = 0; frame < frames; frame++){
		int start_frame=frame-span;
		int end_frame=frame+span;
		if(start_frame<0)
			start_frame=0;
		if(end_frame> frames)
			end_frame=frames;
		float avg=0;
		for(int fr=start_frame;fr<end_frame;fr++)
			avg += nn_avg[fr];
		avg /= (float)(end_frame-start_frame);
		nn_avg_smooth[frame] = avg;
//		nn_avg_smooth[frame] = weight*prev + (1-weight)*nn_avg[frame];
		nn_avg_noise[frame] = nn_avg[frame]-nn_avg_smooth[frame];
	}
	for (int y = 0; y < CorrLen; y++) {
		for(int comp = 0;comp < ncomp; comp++){
			for (int frame = 0; frame < frames; frame++){
				NOISE_ACC(y,comp,frame) += nn_avg_noise[frame];
			}
		}
	}
}


// now neighbor-subtract the comparator signals
void CorrNoiseCorrector::NNSubtractComparatorSigs(int row_span, int time_span, int correctRows)
{
	float nn_avg[frames];
	float nn_avg_smoothed[frames];

	double startTime = CNCTimer();

	if(time_span<1)
		time_span=1;

	for (int y = 0; y < CorrLen; y++) {
		int my_rowspan = row_span;
		if(my_rowspan > CorrLen)
			my_rowspan=CorrLen;
//		if((y+my_rowspan) > CorrLen)
//			my_rowspan = CorrLen-y;
		int start_y = std::max(y-my_rowspan,0);
		int end_y   = std::min(y+my_rowspan,CorrLen);
		if(thumbnail/* && !correctRows*/){
			// keep the ns within the 100x100 block
			start_y = y - y%100; // the beginning of the 100x100 block
			end_y=start_y+100;
		}

		for(int comp = 0;comp < ncomp; comp++){
			memset(nn_avg, 0, sizeof(nn_avg));
			for (int frame = 0; frame < frames; frame++){
				int start_frame= std::max(frame-time_span+1,0);
				int end_frame  = std::min(frame+time_span,frames);
				for(int fr=start_frame;fr<end_frame;fr++){
					for(int ry=start_y;ry<end_y;ry++){
						nn_avg[frame] += SIGS_ACC(ry,comp,fr);
					}
				}
				nn_avg[frame] /= (end_y-start_y)*(end_frame-start_frame);
			}

#ifdef SMOOTH_NNAVG
			// now, smooth nn_avg
			for (int frame = 0; frame < frames; frame++){
				int span=SMOOTH_NNAVG;
				int start_fr=frame-span;
				int end_fr=frame+span;
				if(start_fr < 0)start_fr=0;
				if(end_fr > frames)end_fr=frames;

				nn_avg_smoothed[frame]=0;
				for(int fr=start_fr;fr<end_fr;fr++){
					nn_avg_smoothed[frame] += nn_avg[fr];
				}
				nn_avg_smoothed[frame] /= (end_fr-start_fr);
			}
#else
			for (int frame = 0; frame < frames; frame++){
				nn_avg_smoothed[frame] =nn_avg[frame];
			}
#endif
			for (int frame = 0; frame < frames; frame++){
				float tmp = SIGS_ACC(y,comp,frame) - nn_avg_smoothed[frame];
				if(tmp != tmp)
					tmp=0;

				NOISE_ACC(y,comp,frame) = tmp;

//				if(y<5 && comp==0)
//					printf("%d start_frame=%d end_frame=%d avg=%f noise=%f\n",frame,start_frame,end_frame,nn_avg[frame],NOISE_ACC(y,comp,frame));
			}
		}
	}

	nnsubTime += CNCTimer() - startTime;
}





