/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <float.h>
#include <sys/time.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "Utils.h"
#include "ComparatorNoiseCorrector.h"
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
//#define DBG_PRINT_TIMES 1

#ifdef DBG_SAVETEMPS
#include "crop/Acq.h"
#endif

#define mAvg_num_ACC(x,comparator)  (mAvg_num[(x)+((comparator)*cols)])
#define SIGS_ACC(x,comparator,frame) (mComparator_sigs[((comparator)*cols*frames) + ((frame) *cols) + (x)])

char *ComparatorNoiseCorrector::mAllocMem[MAX_CNC_THREADS] = {NULL};
int ComparatorNoiseCorrector::mAllocMemLen[MAX_CNC_THREADS] = {0};


double CNCTimer()
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

void ComparatorNoiseCorrector::CorrectComparatorNoise(RawImage *raw,Mask *mask, bool verbose,
		bool aggressive_correction, bool beadfind_image, int threadNum)
{
#ifndef BB_DC
	if ((raw->imageState & IMAGESTATE_ComparatorCorrected) == 0)
	  CorrectComparatorNoise(raw->image,raw->rows,raw->cols,raw->frames,mask,verbose,
			  aggressive_correction,beadfind_image, threadNum);
#endif
}

void ComparatorNoiseCorrector::CorrectComparatorNoiseThumbnail(RawImage *raw,Mask *mask, int regionXSize, int regionYSize, bool verbose)
{
  CorrectComparatorNoiseThumbnail(raw->image, raw->rows, raw->cols, raw->frames, mask, /*regionXSize*/100, /*regionYSize*/100, verbose);
}

void ComparatorNoiseCorrector::CorrectComparatorNoiseThumbnail(short *_image, int _rows, int _cols, int _frames, Mask *mask, int regionXSize, int regionYSize, bool verbose)
{
      CorrectComparatorNoise(_image, _rows, _cols, _frames, mask, verbose, 0, false, -1, /*regionXSize*/100, /*regionYSize*/100);

}

void ComparatorNoiseCorrector::CorrectComparatorNoise(short *_image, int _rows, int _cols, int _frames,
		Mask *_mask,bool verbose, bool aggressive_correction, bool beadfind_image, int threadNum, int _regionXSize, int _regionYSize)
{
	char *allocPtr;

	image=_image;
    regionXSize=_regionXSize;
    regionYSize=_regionYSize;

   double wholeStart,start;
   wholeStart = start = CNCTimer();
   allocPtr=AllocateStructs(threadNum, _rows, _cols, _frames);
   allocTime = CNCTimer()-start;
   start = CNCTimer();
   if(_mask == NULL)
	   GenerateMask(mMask);
   else
	   GenerateIntMask(_mask);
   maskTime = CNCTimer()-start;
   start = CNCTimer();


   NNSpan = 5;

   if(regionXSize > 0)
   {//thumbnail image
	   int nYPatches = rows / regionYSize;

	   for(int pRow = 0; pRow < nYPatches; pRow++){
		 //printf(" patch %d\n",pRow*regionYSize);
		 CorrectComparatorNoise_internal( verbose, aggressive_correction, pRow*regionYSize,(((pRow+1)*regionYSize)));
	   }
   }
   else if (!beadfind_image)
   {
      CorrectComparatorNoise_internal( verbose, aggressive_correction);
	  mainTime = CNCTimer()-start;

      if (aggressive_correction)
      {
         int blk_size = 96;
         int sub_blocks = rows / blk_size;
         start=CNCTimer();

         if (blk_size * sub_blocks < rows) {
           sub_blocks++;
         }

         for (int blk = 0; blk < sub_blocks; blk++)
         {
            int row_start = blk * blk_size;
            int row_end = (blk + 1) * blk_size;

            if (blk == sub_blocks - 1) row_end = rows;

            CorrectComparatorNoise_internal( verbose, aggressive_correction, row_start, row_end, true);
         }
     	  aggTime = CNCTimer()-start;
      }
   }
   else
   {
      // trick correction into only doing hf noise correction as this is all that works for beadfind images
      CorrectComparatorNoise_internal( verbose, aggressive_correction, 0, rows,true);
   }

//   ClearPinned(); // re-zero pinned pixels

   totalTime = CNCTimer()-wholeStart;
#ifdef DBG_PRINT_TIMES
   printf("CNC took %.2lf alloc=%.2f mask=%.2f main=%.2f agg=%.2f sum=%.2f apply=%.2f tm1=%.2f tm2=%.2f(%.2f/%.2f) nn=%.2f msk=%.2f pca=%.2f\n",
		   totalTime,allocTime,maskTime,mainTime,aggTime,sumTime,applyTime,tm1,tm2,tm2_1,tm2_2,nnsubTime,mskTime,tm2_3);
#endif
   FreeStructs(threadNum,false,allocPtr);
}

char *ComparatorNoiseCorrector::AllocateStructs(int threadNum, int _rows, int _cols, int _frames)
{
	int len=0;
	char *allocBuffer=NULL;
	char **allocPtr = &allocBuffer;
	int allocLen=0;
	int *allocLenPtr = &allocLen;
	char *aptr;

	rows=_rows;
	cols=_cols;
	frames=_frames;

	initVars();

	// need to put the vectorizable stuff up front to make sure it's properly alligned
	len += mComparator_sigs_len;
    len += mComparator_noise_len;
    len += mComparator_hf_noise_len;
    len += mComparator_rms_len;
    len += mComparator_mask_len;
    len += mComparator_hf_rms_len;
    len += mComparator_hf_mask_len;
    len += mPcomp_len;
    len += mAvg_num_len;
    len += mCorrection_len;
    len += mMask_len;

    if(threadNum >= 0 && threadNum < MAX_CNC_THREADS)
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

	mComparator_sigs = (float *)aptr;  aptr += mComparator_sigs_len;
    mComparator_noise = (float *)aptr; aptr += mComparator_noise_len;
    mComparator_hf_noise = (float *)aptr; aptr += mComparator_hf_noise_len;
    mComparator_rms = (float *)aptr; aptr += mComparator_rms_len;
    mComparator_mask = (int *)aptr; aptr += mComparator_mask_len;
    mComparator_hf_rms = (float *)aptr; aptr += mComparator_hf_rms_len;
    mComparator_hf_mask = (int *)aptr; aptr += mComparator_hf_mask_len;
    mPcomp = (float *)aptr; aptr += mPcomp_len;
    mAvg_num = (float *)aptr; aptr += mAvg_num_len;
    mCorrection = (short int *)aptr; aptr += mCorrection_len;
    mMask = (float *)aptr; aptr += mMask_len;

   	return *allocPtr;
}

void ComparatorNoiseCorrector::FreeStructs(int threadNum, bool force, char *ptr)
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

void ComparatorNoiseCorrector::CorrectComparatorNoise_internal(
		 bool verbose, bool aggressive_correction, int row_start, int row_end, bool hfonly)
{
  int phase=-1;

  memset(mComparator_sigs,0,mComparator_sigs_len);
  memset(mComparator_noise,0,mComparator_noise_len);
  memset(mComparator_hf_noise,0,mComparator_hf_noise_len);
  memset(mAvg_num,0,mAvg_num_len);
  memset(mComparator_mask,0,mComparator_mask_len);
  memset(mComparator_hf_mask,0,mComparator_hf_mask_len);
  memset(mCorrection,0,mCorrection_len);
  ncomp=4;

  if (row_start == -1)
  {
	row_start = 0;
	row_end = rows;
  }


  // first, create the average comparator signals
  // making sure to avoid pinned pixels
  SumColumns(row_start, row_end);
  
  double startTime=CNCTimer();

  // subtract DC offset from average comparator signals
  SetMeanToZero(mComparator_sigs);

#ifdef DBG_SAVETEMPS
	DebugSaveComparatorSigs(0);
#endif

  // now figure out which pair of signals go together
  // this function also combines pairs of signals accordingly
  // from this point forward, there are only cols*2 signals to deal with
  phase = DiscoverComparatorPhase(mComparator_sigs,cols*4);

  // change the data to be column major for the rest of the functions...



#ifdef DBG_SAVETEMPS
	DebugSaveComparatorSigs(1);
#endif
#ifdef DBG_SAVETEMPS
	DebugSaveAvgNum();
#endif
  tm1 += CNCTimer()-startTime;
  startTime = CNCTimer();
  {
      ResetMask();
      double tm2_1_startTime = CNCTimer();

      // now neighbor-subtract the comparator signals
      NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,mComparator_mask,NNSpan,cols*ncomp,frames);

      // measure noise in the neighbor-subtracted signals
      CalcComparatorSigRMS(mComparator_rms,mComparator_noise,cols*ncomp,frames);

      // find the noisiest 10%
      MaskIQR(mComparator_mask,mComparator_rms,cols*ncomp);
#ifdef DBG_SAVETEMPS
		DebugSaveComparatorNoise(0);
		DebugSaveComparatorRMS(0);
		DebugSaveComparatorMask(0);
#endif

      // neighbor-subtract again...avoiding noisiest 10%
      NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,mComparator_mask,NNSpan,cols*ncomp,frames);

      // measure noise in the neighbor-subtracted signals
      CalcComparatorSigRMS(mComparator_rms,mComparator_noise,cols*ncomp,frames);

      ResetMask();

      MaskIQR(mComparator_mask,mComparator_rms,cols*ncomp, verbose);

      // neighbor-subtract again...avoiding noisiest 10%
      NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,mComparator_mask,NNSpan,cols*ncomp,frames);
#ifdef DBG_SAVETEMPS
		DebugSaveComparatorRMS(1);
		DebugSaveComparatorNoise(1);
		DebugSaveComparatorMask(1);
#endif

      tm2_1 += CNCTimer()-tm2_1_startTime;

      if (aggressive_correction)
      {
         // Newly added stuff.
             // subtracts some of what we detect as comparator noise from neighbors before forming the nn average
             // this cleans things up a little
          double tm2_2_startTime = CNCTimer();


         // make another set of noise signals that have been run through a high-pass filter
         // filter low frequency noise out of noise signals
          double tm2_3_startTime = CNCTimer();
         memcpy(mComparator_hf_noise,mComparator_noise,sizeof(float)*cols*ncomp*frames);
         HighPassFilter(mComparator_hf_noise,cols*ncomp,frames,10);
         tm2_3 += CNCTimer() - tm2_3_startTime;

         // neighbor-subtract again...now with some rejection of what we think the noise is
         NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,mComparator_mask,NNSpan,cols*ncomp,frames,mComparator_hf_noise);

         // measure noise in the neighbor-subtracted signals
         CalcComparatorSigRMS(mComparator_rms,mComparator_noise,cols*ncomp,frames);

         ResetMask();

         //      MaskIQR(mComparator_mask,mComparator_rms,cols*2, verbose);
         MaskUsingDynamicStdCutoff(mComparator_mask,mComparator_rms,cols*ncomp,1.0f);

         // even if some comparators didn't make the cut with the raw noise signal
         // we can correct more agressively if we put the noise signal through the high pass filter

         // redo the high-pass fitler
         memcpy(mComparator_hf_noise,mComparator_noise,sizeof(float)*cols*ncomp*frames);
         // get first principal component
         GetPrincComp(mPcomp,mComparator_hf_noise,mComparator_mask,cols*ncomp,frames);
         FilterUsingPrincComp(mComparator_hf_noise,mPcomp,cols*ncomp,frames);

         // measure high frequency noise
         CalcComparatorSigRMS(mComparator_hf_rms,mComparator_hf_noise,cols*ncomp,frames);
         for ( int cndx=0;cndx < cols*ncomp;cndx++ )
         {
           if ( mAvg_num[cndx] == 0 )
           {
             mComparator_hf_mask[cndx] = 1;
           }
         }
         MaskUsingDynamicStdCutoff(mComparator_hf_mask,mComparator_hf_rms,cols*ncomp,2.0f);
         tm2_2 += CNCTimer()-tm2_2_startTime;
         
      }

      // blanks comparator signal averages that didn't have any pixels to average (probably redundant)
      for ( int cndx=0;cndx < cols*ncomp;cndx++ )
      {
        float *cptr;

        if ( mAvg_num[cndx] == 0 ) {
          // get a pointer to where we will build the comparator signal average
          cptr = mComparator_sigs + cndx*frames;
          memset(cptr,0,sizeof(float[frames]));
        }
      }
      tm2 += CNCTimer()-startTime;
      BuildCorrection(hfonly);

#ifdef DBG_SAVETEMPS
	DebugSaveCorrection(row_start, row_end);
#endif

	  ApplyCorrection(phase, row_start, row_end, mCorrection);
  }
}

void ComparatorNoiseCorrector::ResetMask()
{
	int end_cndx=cols*ncomp;

	// mask comparators that don't contain any un-pinned pixels
    for ( int cndx=0;cndx < end_cndx;cndx++ )
    {
      if ( mAvg_num[cndx] == 0 )
          mComparator_mask[cndx] = 1;
      else
          mComparator_mask[cndx] = 0;
    }
}

#ifdef DBG_SAVETEMPS
#define mCorrection_ACC(x,comparator,frame) (mCorrection[comparator * cols * frames + frame * cols + x])
#define mComparator_sigs_ACC(x,comparator,frame) (mComparator_sigs[((comparator)*cols*frames) + ((frame) *cols) + (x)])
#define mComparator_noise_ACC(x,comparator,frame) (mComparator_noise[(ncomp * x + comparator) * frames + frame])
#define mCorrection_ACC(x,comparator,frame) (mCorrection[comparator * cols * frames + frame * cols + x])

void ComparatorNoiseCorrector::DebugSaveAvgNum()
{
	int y,frame;
	int frameStride=rows*cols;
	short *srcPtr;
	float *corPtr;
//	int lw=cols/VEC8_SIZE;

	  Image loader2;

	  loader2.LoadRaw ( "acq_0000.dat", 0, true, false );
	// now subtract each neighbor-subtracted comparator signal from the
	// pixels that are connected to that comparator

	for (frame = 0; frame < frames; frame++)
	{
		for (y = 0; y < rows; y++)
		{
			for(int x = 0; x < cols; x++)
			{
				int comparator=y&(ncomp-1);
				srcPtr = &loader2.raw->image[frame * frameStride + y * cols + x];
				corPtr = &mAvg_num[x*2+comparator];//mAvg_num_ACC(x,comparator);

				*srcPtr = (short)(/*8192.0f +*/ *corPtr);
			}
		}
	}
    char *newName=(char *)"acq_0000.dat_oldavgnum";

    Acq saver;
    saver.SetData ( &loader2 );
    saver.WriteVFC(newName, 0, 0, cols, rows);

}

void ComparatorNoiseCorrector::DebugSaveComparatorSigs(int state)
{
	  Image loader2;

	  loader2.LoadRaw ( "acq_0000.dat", 0, true, false );




	int y,frame;
	int frameStride=rows*cols;
	short *srcPtr;
	float *corPtr;
//	int lw=cols/VEC8_SIZE;

	// now subtract each neighbor-subtracted comparator signal from the
	// pixels that are connected to that comparator

	for (frame = 0; frame < frames; frame++)
	{
		for (y = 0; y < rows; y++)
		{
			for(int x = 0; x < cols; x++)
			{
				int comparator=y&(ncomp-1);
				srcPtr = &loader2.raw->image[frame * frameStride + y * cols + x];
				if(!state)
					corPtr = &mComparator_sigs_ACC(x,comparator,frame);
				else
					corPtr = &mComparator_sigs[(2*x+comparator)*frames+frame];


				*srcPtr = (short)(8192.0f + *corPtr);
			}
		}
	}

        char newName[256];
        sprintf(newName,"acq_0000.dat_oldsig%d",state);

        Acq saver;
        saver.SetData ( &loader2 );
        saver.WriteVFC(newName, 0, 0, cols, rows);

}
void ComparatorNoiseCorrector::DebugSaveComparatorMask(int time)
{
	int y,frame;
	int frameStride=rows*cols;
	short *srcPtr;
	int *corPtr;
//	int lw=cols/VEC8_SIZE;

	  Image loader2;

	  loader2.LoadRaw ( "acq_0000.dat", 0, true, false );
	// now subtract each neighbor-subtracted comparator signal from the
	// pixels that are connected to that comparator

	for (frame = 0; frame < frames; frame++)
	{
		for (y = 0; y < rows; y++)
		{
			for(int x = 0; x < cols; x++)
			{
				int comparator=y&(ncomp-1);
				srcPtr = &loader2.raw->image[frame * frameStride + y * cols + x];
				corPtr = &mComparator_mask[x*ncomp + comparator];

				*srcPtr = (short)(8192.0f + *corPtr);
			}
		}
	}
    char newName[1024];
    sprintf(newName,"acq_0000.dat_oldmask%d",time);

    Acq saver;
    saver.SetData ( &loader2 );
    saver.WriteVFC(newName, 0, 0, cols, rows);
}

void ComparatorNoiseCorrector::DebugSaveComparatorNoise(int time)
{
	int y,frame;
	int frameStride=rows*cols;
	short *srcPtr;
	float *corPtr;
//	int lw=cols/VEC8_SIZE;

	  Image loader2;

	  loader2.LoadRaw ( "acq_0000.dat", 0, true, false );
	// now subtract each neighbor-subtracted comparator signal from the
	// pixels that are connected to that comparator

	for (frame = 0; frame < frames; frame++)
	{
		for (y = 0; y < rows; y++)
		{
			for(int x = 0; x < cols; x++)
			{
				int comparator=y&(ncomp-1);
				srcPtr = &loader2.raw->image[frame * frameStride + y * cols + x];
				corPtr = &mComparator_noise_ACC(x,comparator,frame);

				*srcPtr = (short)(8192.0f + *corPtr);
			}
		}
	}
    char newName[1024];

    sprintf(newName,"acq_0000.dat_oldnoise%d",time);

    Acq saver;
    saver.SetData ( &loader2 );
    saver.WriteVFC(newName, 0, 0, cols, rows);

}

void ComparatorNoiseCorrector::DebugSaveComparatorRMS(int time)
{
	int y,frame;
	int frameStride=rows*cols;
	short *srcPtr;
	float *corPtr;
//	int lw=cols/VEC8_SIZE;

	  Image loader2;

	  loader2.LoadRaw ( "acq_0000.dat", 0, true, false );
	// now subtract each neighbor-subtracted comparator signal from the
	// pixels that are connected to that comparator

	for (frame = 0; frame < frames; frame++)
	{
		for (y = 0; y < rows; y++)
		{
			for(int x = 0; x < cols; x++)
			{
				int comparator=y&(ncomp-1);
				srcPtr = &loader2.raw->image[frame * frameStride + y * cols + x];
				corPtr = &mComparator_rms[x*ncomp+comparator];

				*srcPtr = (short)(8192.0f + *corPtr);
			}
		}
	}
    char newName[1024];

    sprintf(newName,"acq_0000.dat_oldrms%d",time);

    Acq saver;
    saver.SetData ( &loader2 );
    saver.WriteVFC(newName, 0, 0, cols, rows);

}


void ComparatorNoiseCorrector::DebugSaveCorrection(int row_start, int row_end)
{
	int y,frame;
	int frameStride=rows*cols;
	short *srcPtr;
	short *corPtr;

	  Image loader2;

	  loader2.LoadRaw ( "acq_0000.dat", 0, true, false );

	// now subtract each neighbor-subtracted comparator signal from the
	// pixels that are connected to that comparator
	printf("%s:\n",__FUNCTION__);
	for (frame = 0; frame < frames; frame++)
	{
		for (y = row_start; y < row_end; y++)
		{
			for(int x = 0; x < cols; x++)
			{
				int comparator=y&(ncomp-1);
				srcPtr = &loader2.raw->image[frame * frameStride + y * cols + x];
				corPtr = &mCorrection_ACC(x,comparator,frame);

				*srcPtr = (short)(8192 + *corPtr);
			}
		}
	}

    char *newName=(char *)"acq_0000.dat_oldcorr";

    Acq saver;
    saver.SetData ( &loader2 );
    saver.WriteVFC(newName, 0, 0, cols, rows);
}
#endif

// generate a 16-bit array of correction signals.  one for each comparator
// rounding is weird here.  Its not the same as the old function.
void ComparatorNoiseCorrector::BuildCorrection(bool hfonly)
{
	int x, frame, comparator, cndx;
	short int *mcPtr;
	float val;

	// Build mCorrection from mComparator_noise and mComparator_hf_noise

	for (comparator = 0; comparator < ncomp; comparator++)
	{
		for (frame = 0; frame < frames; frame++)
		{
			mcPtr = &mCorrection[comparator * cols * frames + frame * cols];
			for (x = 0; x < cols; x++)
			{
				// figure out which comparator this pixel belongs to
				cndx = ncomp * x + comparator;

				//only perform correction on noisy comparators;
				if (mComparator_mask[cndx] && !hfonly)
				{
					// subtract nn comparator signal from this pixel's data
					val = mComparator_noise[cndx * frames + frame];
					if(val > 0)
					val += 0.5f;
					else
					val -= 0.5f;
					mcPtr[x] += (short int)val;
				}
				else if (mComparator_hf_mask[cndx])
				{
					// subtract nn comparator signal from this pixel's data
					val = mComparator_hf_noise[cndx * frames + frame];
					if(val > 0)
					val += 0.5f;
					else
					val -= 0.5f;
					mcPtr[x] += (short int)val;
				}
			}
		}
	}
}


// subtract the already computed correction from the image file
void ComparatorNoiseCorrector::ApplyCorrection(int  phase, int row_start, int row_end, short int *correction)
{
	int i,y,frame;
	int frameStride=rows*cols;
	v8s *srcPtr, *corPtr;
	int lw=cols/VEC8_SIZE;
	int corrComp[4] = {0,1,2,3};
	double startTime = CNCTimer();

//	printf("ncomp=%d phase=%d row_start=%d row_end=%d\n",ncomp,phase,row_start,row_end);
	if(ncomp == 2){
		if(phase==0){
		corrComp[0] = 1;
		corrComp[1] = 1;
		corrComp[2] = 0;
		corrComp[3] = 0;
		}
		else{
			corrComp[0] = 0;
			corrComp[1] = 1;
			corrComp[2] = 1;
			corrComp[3] = 0;
		}
	}

	// now subtract each neighbor-subtracted comparator signal from the
	// pixels that are connected to that comparator

	for (frame = 0; frame < frames; frame++)
	{
		for (y = row_start; y < row_end; y++)
		{
			srcPtr = (v8s *)(&image[frame * frameStride + y * cols]);
			corPtr = (v8s *) (correction + corrComp[(y-row_start)&3]*frames*cols + frame*cols);

			for (i = 0;i<lw;i++)
			{
				srcPtr[i] -= corPtr[i];
			}
		}
	}
	  applyTime += CNCTimer()-startTime;
}

// sum the columns from row_start to row_end and put the answer in mComparator_sigs
// mMask has a 1 in it for every active column and a zero for every pinned pixel
void ComparatorNoiseCorrector::SumColumns(int row_start, int row_end)
{
	int frame,x,y,comparator;
	int frameStride=rows*cols;

	double startTime=CNCTimer();


	memset(mAvg_num,0,mAvg_num_len);

#ifdef __AVX__
	if((cols%VEC8_SIZE) == 0)
	{
		int lw=cols/VEC8_SIZE;
		v8f_u valU;

		valU.V=LD_VEC8F(0);
		for (frame = 0; frame < frames; frame++)
		{
			for (y = row_start; y < row_end; y++)
			{
				comparator = (y - row_start) & 0x3;
				short int *sptr = &image[frame * frameStride + y * cols];
				v8f *dstPtr = (v8f *) (mComparator_sigs + comparator * cols * frames + frame * cols);
				v8f *sumPtr = (v8f *) (mAvg_num + comparator * cols * frames + frame * cols);
				v8f *mskPtr = (v8f *) (mMask + cols*y);

				for (x = 0; x < lw; x++,sptr+=VEC8_SIZE,dstPtr++,sumPtr++,mskPtr++)
				{
					LD_VEC8S_CVT_VEC8F(sptr,valU);

					*dstPtr += valU.V;
					*sumPtr += *mskPtr;
				}
			}
		}
		for (frame = 0; frame < frames; frame++)
		{
			for (comparator = 0; comparator < 4; comparator++)
			{
				float *dstPtr = (float *) (mComparator_sigs + comparator * cols * frames + frame * cols);
				float *sumPtr = (float *) (mAvg_num + comparator * cols * frames + frame * cols);
				for (x = 0; x < cols; x++)
				{
					float valU = (float)((sumPtr[x])?sumPtr[x]:1);
					dstPtr[x] /= valU;
				}
			}
		}
	}
	else
#endif
	{
		for (frame = 0; frame < frames; frame++)
		{
			for (y = row_start; y < row_end; y++)
			{
				comparator = (y - row_start) & 0x3;
				float valU;
				short int *srcPtr = (short int *) (&image[frame * frameStride + y * cols]);
				float *dstPtr = (float *) (mComparator_sigs + comparator * cols * frames + frame * cols);
				float *sumPtr = (float *) (mAvg_num + comparator * cols);
				float *mskPtr = (float *) (mMask + cols*y);
				if(frame==0){
					for (x = 0; x < cols; x++)
					{
						valU= (float)srcPtr[x];
						dstPtr[x] += valU;
						sumPtr[x] += mskPtr[x];
					}
				}
				else{
					for (x = 0; x < cols; x++)
					{
						valU= (float)srcPtr[x];
						dstPtr[x] += valU;
					}
				}
			}
		}
		for (frame = 0; frame < frames; frame++)
		{
			for (comparator = 0; comparator < 4; comparator++)
			{
				float *dstPtr = (float *) (mComparator_sigs + comparator * cols * frames + frame * cols);
				float *sumPtr = (float *) (mAvg_num + comparator * cols);
				for (x = 0; x < cols; x++)
				{
					float valU = (float)((sumPtr[x])?sumPtr[x]:1);
					dstPtr[x] /= valU;
				}
			}
		}
	}

	sumTime += CNCTimer() - startTime;

}

// sets the mean of the columns averages to zero
void ComparatorNoiseCorrector::SetMeanToZero(float *inp)
{
	int cndx, frame, comparator;
	v8f_u dc,framesU;
	v8f_u *srcPtr;
	int lw=cols/VEC8_SIZE;

	framesU.V = LD_VEC8F((float)frames);

	// get a pointer to where we will build the comparator signal average
	for (comparator=0;comparator<ncomp;comparator++)
	{
		srcPtr=(v8f_u *)(inp + comparator*frames*cols);
		for (cndx = 0; cndx < lw; cndx++)
		{
			dc.V = LD_VEC8F(0.0f);

			for (frame = 0; frame < frames; frame++)
				dc.V += srcPtr[lw*frame].V;

			dc.V /= framesU.V;

			// subtract dc offset
			for (frame = 0; frame < frames; frame++)
				srcPtr[lw*frame].V -= dc.V;
			srcPtr++;
		}
	}

}

int ComparatorNoiseCorrector::DiscoverComparatorPhase(float *psigs,int n_comparators)
{
  float phase_rms[2];
  int phase=-1;

#ifndef BB_DC
  if( !ChipIdDecoder::IsPzero() )
#else
	  if(eg.ChipInfo.interlace_type == 1) // two half rows inter-mingled
#endif
  { // 2 comparators per column case


  for ( phase = 0;phase < 2;phase++ )
  {
    phase_rms[phase] = 0.0f;
    int rms_num = 0;

    for ( int i=0;i < cols;i++ )
    {
      float *cptr_1a;
      float *cptr_1b;
      float *cptr_2a;
      float *cptr_2b;
      float rms_sum = 0.0f;

      // have to skip any columns that have all pinned pixels in any subset-average
//      if (( mAvg_num[i] == 0 ) && ( mAvg_num[i] == 1 ) && ( mAvg_num[i] == 2 ) && ( mAvg_num[i] == 3 ))
      if (( mAvg_num[(i + 0*cols)] == 0 ) && ( mAvg_num[(i + 1*cols)] == 0 ) &&
    		  ( mAvg_num[(i + 2*cols)] == 0 ) && ( mAvg_num[(i + 3*cols)] == 0 )){
//        fprintf (stdout, "Noisy column: %d; Comparator: %d.\n", i/4, i&0x3);
        continue;
      }

      // get a pointers to the comparator signals
      if ( phase==0 ) {
        cptr_1a = &SIGS_ACC(i,2,0);
        cptr_1b = &SIGS_ACC(i,3,0);
        cptr_2a = &SIGS_ACC(i,0,0);
        cptr_2b = &SIGS_ACC(i,1,0);
      }
      else
      {
        cptr_1a = &SIGS_ACC(i,0,0);
        cptr_1b = &SIGS_ACC(i,3,0);
        cptr_2a = &SIGS_ACC(i,1,0);
        cptr_2b = &SIGS_ACC(i,2,0);
      }

      for ( int frame=0;frame < frames;frame++ )
      {
    	  int idx= frame*cols;
        rms_sum += (cptr_1a[idx]-cptr_1b[idx])*(cptr_1a[idx]-cptr_1b[idx]);
        rms_sum += (cptr_2a[idx]-cptr_2b[idx])*(cptr_2a[idx]-cptr_2b[idx]);
      }
      phase_rms[phase] += rms_sum;
      rms_num++;
    }

    //make them comparable between different runs
    if(rms_num != 0){
        phase_rms[phase] /= (2*rms_num);
    }
  }

  if (phase_rms[0] == 0 || phase_rms[1] == 0){
    return -1; //special tag to indicate case of 0 rms
  }


//  float phaseMeasure=0;
  if ( phase_rms[0] < phase_rms[1] ){
    phase = 0;
//    phaseMeasure=phase_rms[1]/phase_rms[0];
  }
  else{
    phase = 1;
//  	phaseMeasure=phase_rms[0]/phase_rms[1];
  }

//  fprintf (stdout, "Phase: %d; phaseMeasure=%.1f RMS Phase Calcs = %f vs %f\n", phase, phaseMeasure, phase_rms[0], phase_rms[1]);
	  //get phase_rms values to check how reliable it is

  // now combine signals according to the detected phase
  for ( int i=0;i < cols;i++ )
  {
    float *cptr_1a;
    float *cptr_1b;
    float *cptr_2a;
    float *cptr_2b;
    float num_1a,num_1b,num_2a,num_2b;
    float num1;
    float num2;
    float scale1;
    float scale2;

    // get a pointers to the comparator signals
    if ( phase==0 ) {
	    cptr_1a = &SIGS_ACC(i,2,0);
	    cptr_1b = &SIGS_ACC(i,3,0);
	    cptr_2a = &SIGS_ACC(i,0,0);
	    cptr_2b = &SIGS_ACC(i,1,0);
	    num_1a = mAvg_num_ACC(i,2);
	    num_1b = mAvg_num_ACC(i,3);
	    num_2a = mAvg_num_ACC(i,0);
	    num_2b = mAvg_num_ACC(i,1);
    }
    else
    {
	    cptr_1a = &SIGS_ACC(i,0,0);
	    cptr_1b = &SIGS_ACC(i,3,0);
	    cptr_2a = &SIGS_ACC(i,1,0);
	    cptr_2b = &SIGS_ACC(i,2,0);
	    num_1a = mAvg_num_ACC(i,0);
	    num_1b = mAvg_num_ACC(i,3);
	    num_2a = mAvg_num_ACC(i,1);
	    num_2b = mAvg_num_ACC(i,2);
    }

    num1 = num_1a+num_1b;
    num2 = num_2a+num_2b;

    if ( num1 > 0 )
      scale1 = 1.0f/((float)num1);
    else
      scale1 = 0.0f;

    if ( num2 > 0 )
      scale2 = 1.0f/((float)num2);
    else
      scale2 = 0.0f;

    for ( int frame=0;frame < frames;frame++ )
    {
      // beware...we are doing this in place...need to be careful
      float sum1 = scale1*(cptr_1a[frame*cols]*num_1a+cptr_1b[frame*cols]*num_1b);
      float sum2 = scale2*(cptr_2a[frame*cols]*num_2a+cptr_2b[frame*cols]*num_2b);

//	  if(hfonly == 1)
//	  	printf("%d/%d) sum1 = %.2f %f  %.2f * %d + %.2f * %d\n",i,frame,sum1,scale1,cptr_1a[frame*cols],num_1a,cptr_1b[frame*cols],num_1b);
//      SIGS_ACC(i,2,frame) = sum1;
//      SIGS_ACC(i,3,frame) = sum2;
      mComparator_noise[(i*2+0)*frames + frame] = sum1;
      mComparator_noise[(i*2+1)*frames + frame] = sum2;
    }

    mComparator_rms[i*2+0] = num1;
    mComparator_rms[i*2+1] = num2;
//    mAvg_num_ACC(i,2) = num1;
//    mAvg_num_ACC(i,3) = num2;

  }

  //copy data back to the right place
  memcpy(mComparator_sigs,mComparator_noise,2*cols*frames*sizeof(mAvg_num[0]));
  memcpy(mAvg_num,mComparator_rms,2*cols*sizeof(mAvg_num[0]));
  ncomp=2;
//  printf("in two comparator cnc\n");

  }
  else
  { // 4 comparators per column
	  for ( int i=0;i < cols;i++ )
	  {
		    for ( int frame=0;frame < frames;frame++ )
		    {
			  mComparator_noise[(i*4+0)*frames + frame] = SIGS_ACC(i,0,frame);
			  mComparator_noise[(i*4+1)*frames + frame] = SIGS_ACC(i,1,frame);
			  mComparator_noise[(i*4+2)*frames + frame] = SIGS_ACC(i,2,frame);
			  mComparator_noise[(i*4+3)*frames + frame] = SIGS_ACC(i,3,frame);
		    }
		    mComparator_rms[i*4+0] = mAvg_num_ACC(i,0);
		    mComparator_rms[i*4+1] = mAvg_num_ACC(i,1);
		    mComparator_rms[i*4+2] = mAvg_num_ACC(i,2);
		    mComparator_rms[i*4+3] = mAvg_num_ACC(i,3);
	  }
	  memcpy(mComparator_sigs,mComparator_noise,4*cols*frames*sizeof(mAvg_num[0]));
	  memcpy(mAvg_num,mComparator_rms,4*cols*sizeof(mAvg_num[0]));
	  ncomp=4;
//	  printf("in four comparator cnc\n");
  }
  return phase;
}


// now neighbor-subtract the comparator signals
void ComparatorNoiseCorrector::NNSubtractComparatorSigs(float *pnn,float *psigs,int *mask,int span,int n_comparators,int nframes,float *hfnoise)
{
	double startTime = CNCTimer();

	float nn_avg[nframes];
	float zero_sig[nframes];

	memset(zero_sig, 0, sizeof(zero_sig));

	for (int i = 0; i < n_comparators; i++) {
		int nn_cnt = 0;
		float *cptr;
		float *n_cptr;
		float *chfptr = NULL;
		float *n_chfptr = NULL;
		float *nncptr;
		int i_c0 = i & ~(ncomp - 1);
		memset(nn_avg, 0, sizeof(nn_avg));

		// in case we weren't provided with high frequency noise correction for NNs, use all zeros instead
		chfptr = zero_sig;
		n_chfptr = zero_sig;

		// rounding down the starting point and adding one to the rhs properly centers
		// the neighbor average about the central column...except in cases where columns are
		// masked within the neighborhood.

		//same column but the other comparators
		for (int comparator = 0; comparator < ncomp; comparator++) {
			int cndx = i_c0 + comparator;

			if (!mask[cndx] && cndx != i) {
				// get a pointer to the comparator signal
				cptr = psigs + cndx * nframes;

				if (hfnoise != NULL) {
					chfptr = hfnoise + cndx * nframes;

					// add it to the average
					for (int frame = 0; frame < nframes; frame++)
						nn_avg[frame] += cptr[frame] - chfptr[frame];
				} else {
					// add it to the average
					for (int frame = 0; frame < nframes; frame++)
						nn_avg[frame] += cptr[frame];
				}

				nn_cnt++;
			}
		}

		for (int s = 1; s <= span; s++) {
			//i_c0 is even number
			for (int comparator = 0; comparator < ncomp; comparator++) {
				int cndx = i_c0 - ncomp * s + comparator;
				int n_cndx = i_c0 + ncomp * s + comparator;
				if (!(cndx < 0 || n_cndx >= n_comparators ||
						mask[cndx] || mask[n_cndx] ||
						(regionXSize
								&& (((cndx / ncomp) / regionXSize) != ((i / ncomp) / regionXSize) ||
								  ((n_cndx / ncomp) / regionXSize) != ((i / ncomp) / regionXSize))))) {
					// get a pointer to the comparator signal
					cptr = psigs + cndx * nframes;
					n_cptr = psigs + n_cndx * nframes;

					if (hfnoise != NULL) {
						chfptr = hfnoise + cndx * nframes;
						n_chfptr = hfnoise + n_cndx * nframes;
					}
					// add it to the average
					for (int frame = 0; frame < nframes; frame++) {
						nn_avg[frame] += cptr[frame] - chfptr[frame];
						nn_avg[frame] += n_cptr[frame] - n_chfptr[frame];
					}

					nn_cnt += 2;
				}
			}
		}

		if ((nn_cnt > 0)) {
			for (int frame = 0; frame < nframes; frame++)
				nn_avg[frame] /= nn_cnt;

			// now subtract the neighbor average
			cptr = psigs + i * nframes;
			nncptr = pnn + i * nframes;
			for (int frame = 0; frame < nframes; frame++)
				nncptr[frame] = cptr[frame] - nn_avg[frame];
		} else {
//      fprintf (stdout, "Default noise of 0 is set: %d\n", i);
			// not a good set of neighbors to use...just blank the correction
			// signal and do nothing.
			nncptr = pnn + i * nframes;
			for (int frame = 0; frame < nframes; frame++)
				nncptr[frame] = 0.0f;
		}
	}
	nnsubTime += CNCTimer() - startTime;
}

void ComparatorNoiseCorrector::HighPassFilter(float *pnn,int n_comparators,int nframes,int span)
{
  //v8f_u* trc_scratch = new v8f_u[nframes];
  v8f_u trc_scratch[nframes];

  for ( int i=0;i < n_comparators;i+=VEC8_SIZE )
  {
    float *cptr;

    // get a pointer to the comparator noise signal
    cptr = pnn + i*nframes;
    
    // make smooth version of noise signal
    for (int j=0;j < nframes;j++)
    {
        v8f_u sum;
        sum.V = LD_VEC8F(0.0f);
        float cnt=0;
        
        for (int idx=(j-span);idx <=(j+span);idx++)
        {
            if ((idx >= 0) && (idx < nframes) && (idx!=j))
            {
            	for(int k=0;k<VEC8_SIZE;k++)
            		sum.A[k] += cptr[idx+k*nframes];
                cnt=cnt+1.0f;
            }
        }
        
        v8f cntV = LD_VEC8F(cnt);

    	for(int k=0;k<VEC8_SIZE;k++)
    		trc_scratch[j].V = sum.V/cntV;
    }
    
    // now subtract off the smoothed signal to eliminate low frequency
    // components, most of which are residual background effects that the
    // neighbor subtraction algorithm doesn't completely fitler out
    // this unfortunately does also somtimes eliminate some real comparator
    // noise...
    for (int j=0;j < nframes;j++)
    {
    	for(int k=0;k<VEC8_SIZE;k++)
    		cptr[j+k*nframes] -= trc_scratch[j].A[k];
    }
  }
}

// measure noise in the neighbor-subtracted signals
void ComparatorNoiseCorrector::CalcComparatorSigRMS(float *prms,float *pnn,int n_comparators,int nframes)
{
  for ( int i=0;i < n_comparators;i++ )
  {
    float *cptr;
    float rms_sum = 0.0f;

    // get a pointer to the comparator signal
    cptr = pnn + i*nframes;

    // add it to the average
    for ( int frame=0;frame < nframes;frame++ )
      rms_sum += cptr[frame]*cptr[frame];

    prms[i] = sqrt(rms_sum/nframes);
//    fprintf (stdout, "RMS of Comparator %d: %f\n", i, prms[i]);
  }
}

// find the noisiest 10%
void ComparatorNoiseCorrector::MaskAbove90thPercentile(int *mask,float *prms,int n_comparators)
{
  float rms_sort[n_comparators];
  int i;

  memcpy(rms_sort,prms,sizeof(rms_sort));

  // sort the top 10%
  for ( i=0;i < (n_comparators/10);i++ )
  {
    for ( int j=i;j < n_comparators;j++ )
    {
      if ( rms_sort[j] > rms_sort[i] )
      {
        float tmp = rms_sort[j];
        rms_sort[j] = rms_sort[i];
        rms_sort[i] = tmp;
      }
    }
  }

  float rms_thresh = rms_sort[i-1];

//  printf("**************************** comparator noise threshold = %f\n",rms_thresh);

  for ( i=0;i < n_comparators;i++ )
  {
    if ( prms[i] >= rms_thresh )
      mask[i] = 1;
  }
}

// find the noisiest 10%
void ComparatorNoiseCorrector::MaskIQR(int *mask,float *prms,int n_comparators, bool verbose)
{
    double startTime = CNCTimer();

  float rms_sort[n_comparators];
  int i;

  memcpy(rms_sort,prms,sizeof(rms_sort));

  std::sort(rms_sort, rms_sort+n_comparators);

  float rms_thresh = rms_sort[n_comparators * 3 / 4 - 1] + 2.5 * (rms_sort[n_comparators * 3/4 - 1] - rms_sort[n_comparators * 1/4 - 1]) ;

  int noisyCount = 0;
  if(verbose)
    fprintf (stdout, "Noisy comparators:");
  for ( i=0;i < n_comparators;i++ )
  {
    if ( prms[i] >= rms_thresh ){
      mask[i] = 1;
      noisyCount ++;
      if(verbose)
        fprintf (stdout, " %d", i);
    }
  }
  if(verbose)
    fprintf (stdout, "\n");
//  fprintf (stdout, "\n%d noisy comparators; threshold: %f\n", noisyCount, rms_thresh);
  mskTime += CNCTimer()-startTime;
}

void ComparatorNoiseCorrector::MaskUsingDynamicStdCutoff(int *mask,float *prms,int n_comparators, float std_mult, bool verbose)
{
    float mean_rms;
    float std_rms;
    int cnt;
    float rms_threshold = 1E+20f;
    int i;
    
    for (int iter=0;iter < 2;iter++)
    {
        mean_rms = 0.0f;
        std_rms = 0.0f;
        cnt = 0;
        
        for (i=0;i < n_comparators;i++ )
        {
            if ((mask[i] == 0) && (prms[i] < rms_threshold))
            {
                mean_rms += prms[i];
                cnt++;
            }
        }

        // if not enough to analyze...just bail
        if (cnt < 10)
            return;

        mean_rms /= cnt;

        for (i=0;i < n_comparators;i++ )
        {
            if ((mask[i] == 0) && (prms[i] < rms_threshold))
                std_rms += (prms[i]-mean_rms)*(prms[i]-mean_rms);
        }
        
        std_rms = sqrt(std_rms/cnt);
        
        rms_threshold = mean_rms + std_mult*std_rms;
    }

//  printf("hf rms threshold = %f\n",rms_threshold);
    
  int noisyCount = 0;
  if(verbose)
    fprintf (stdout, "Noisy comparators:");
    
    // now set the mask according to the threshold
    for (i=0;i < n_comparators;i++ )
    {
        if ((mask[i] == 0) && (prms[i] >= rms_threshold))
        {
            mask[i] = 1;
            noisyCount ++;
            if(verbose)
              fprintf (stdout, " %d", i);
        }
    }
  if(verbose)
    fprintf (stdout, "\n");
}

// simple iterative formula that is good for getting the first principal component
void ComparatorNoiseCorrector::GetPrincComp(float *mPcomp,float *pnn,int *mask,int n_comparators,int nframes)
{
	float ptmp[nframes];
	float ttmp[nframes];
	float residual;
	
	for (int i=0;i < nframes;i++)
	{
          //		ptmp[i] = rand();
          ptmp[i] = mRand.Rand();
		ttmp[i] = 0.0f;
	}
	
	memset(mPcomp,0,sizeof(float)*nframes);
	
	residual = FLT_MAX;
 
        int iterations = 0;	
	while((residual > 0.001) && (iterations < MAX_CNC_PCA_ITERS))
	{
		memset(ttmp,0,sizeof(float)*nframes);

		for (int i=0;i < n_comparators;i++)
		{
			float sum=0.0f;
			float *cptr = pnn + i*nframes;
			
			if (mask[i] == 0)
			{
				for (int j=0;j < nframes;j++)
					sum += ptmp[j]*cptr[j];
				
				for (int j=0;j < nframes;j++)
					ttmp[j] += cptr[j]*sum;
			}
		}
				
		float tmag = 0.0f;
		for (int i=0;i < nframes;i++)
			tmag += ttmp[i]*ttmp[i];
		
		tmag = sqrt(tmag/nframes);

		for (int i=0;i < nframes;i++)
			mPcomp[i] = ttmp[i]/tmag;
			
		residual = 0.0f;
		for (int i=0;i < nframes;i++)
			residual += (mPcomp[i]-ptmp[i])*(mPcomp[i]-ptmp[i]);
		residual = sqrt(residual/nframes);
		memcpy(ptmp,mPcomp,sizeof(float)*nframes);

                iterations++;
	}
}

void ComparatorNoiseCorrector::FilterUsingPrincComp(float *pnn,float *mPcomp,int n_comparators,int nframes)
{
    float pdotp = 0.0f;
    for (int i=0;i < nframes;i++)
        pdotp += mPcomp[i]*mPcomp[i];
    
    for (int i=0;i < n_comparators;i++)
    {
        float sum=0.0f;
    	float *cptr = pnn + i*nframes;
        
        for (int j=0;j < nframes;j++)
            sum += cptr[j]*mPcomp[j];
        
        float scale = sum/pdotp;
        
        for (int j=0;j < nframes;j++)
            cptr[j] -= mPcomp[j]*scale;
    }
}

#define VEC_CMP_GE(a,b) ((a.A[0] >= b.A[0]) || (a.A[1] >= b.A[1]) || (a.A[2] >= b.A[2]) || (a.A[3] >= b.A[3]) || (a.A[4] >= b.A[4]) || (a.A[5] >= b.A[5]) || (a.A[6] >= b.A[6]) || (a.A[7] >= b.A[7]))

void ComparatorNoiseCorrector::GenerateIntMask(Mask *_mask)
{
#ifndef BB_DC
	int frameStride=rows*cols;
	int idx;

	for(idx=0;idx<frameStride;idx++)
	{
		if((*_mask)[idx] & MaskPinned)
			mMask[idx] = 0;
		else
			mMask[idx] = 1;
	}
#endif
}

// generates a mask of pinned pixels
void ComparatorNoiseCorrector::GenerateMask(float *_mask)
{
	int frm,idx;
	int frameStride=rows*cols;
	int len=frameStride/VEC8_SIZE;
	v8f_u *mskPtr,tmp;
	v8f_u high,low;
	int cnt=0;
	mMaskGenerated=1;

	high.V=LD_VEC8F(16380);
	low.V=LD_VEC8F(5);


	// initialize the mask to all 1's
	tmp.V=LD_VEC8F(1.0f);
	mskPtr=(v8f_u *)_mask;
	for(idx=0;idx<len;idx++)
		mskPtr[idx].V=tmp.V;

	for(frm=0;frm<frames;frm++)
	{
		mskPtr=(v8f_u *)_mask;
#if 0//def __AVX__
		v8f_u tmpV;
		uint16_t *src=(uint16_t *)(image + rows*cols*frm);
		for(idx=0;idx<len;idx++)
		{
			LD_VEC8S_CVT_VEC8F(src,tmpV);

			// now, do the pinned pixel comparisons
			mskPtr[idx].V += __builtin_ia32_cmpps256(tmpV.V , low.V, _CMP_LT_OS);
			mskPtr[idx].V += __builtin_ia32_cmpps256(high.V, tmpV.V, _CMP_LT_OS);
		}
#else
		v8s_u *src=(v8s_u *)(image + rows*cols*frm);
		for(idx=0;idx<len;idx++)
		{
			for(int k=0;k<VEC8_SIZE;k++)
			{
				if(src[idx].A[k] >= high.A[k] || low.A[k] >= src[idx].A[k])
					mskPtr[idx].A[k]=0;
			}
		}
#endif
	}

	// now, clear all pinned pixels so the column adds will be quick
	for(idx=0;idx<frameStride;idx++)
	{
		if(_mask[idx]==0)
		{
			cnt++;
			// this pixel is pinned..  clear all the frame values
			for(frm=0;frm<frames;frm++)
			{
				image[rows*cols*frm + idx] = 0;
			}
		}
	}

//	printf("found %d pinned pixels out of %d pixels\n",cnt,frameStride);
}

void ComparatorNoiseCorrector::ClearPinned()
{
	int frameStride=rows*cols;

	// now, clear all pinned pixels so the column adds will be quick
	for(int idx=0;idx<frameStride;idx++)
	{
		if(mMask[idx]==0)
		{
			// this pixel is pinned..  clear all the frame values
			for(int frm=0;frm<frames;frm++)
			{
				image[rows*cols*frm + idx] = 0;
			}
		}
	}

}


