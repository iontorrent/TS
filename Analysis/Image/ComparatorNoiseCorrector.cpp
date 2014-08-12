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
#include "ComparatorNoiseCorrector.h"
#include "deInterlace.h"
#include "Vecs.h"
#ifdef __AVX__
#include <xmmintrin.h>
#include "immintrin.h"
#endif

#define mAvg_num_ACC(x,comparator)  (mAvg_num[(x)+((comparator)*cols)])
#define SIGS_ACC(x,comparator,frame) (mComparator_sigs[((comparator)*cols*frames) + ((frame) *cols) + (x)])

char *ComparatorNoiseCorrector::mAllocMem[MAX_CNC_THREADS] = {NULL};
int ComparatorNoiseCorrector::mAllocMemLen[MAX_CNC_THREADS] = {0};

double sumTime=0;
double applyTime = 0;
double tm1=0;
double tm2=0;
double tm2_1=0;
double tm2_2=0;
double tm2_3=0;
double nnsubTime=0;
double mskTime=0;

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

//void ComparatorNoiseCorrector::justGenerateMask(RawImage *raw, int threadNum)
//{
//	//char *allocPtr;
//	image=raw->image;
//	/*allocPtr=*/AllocateStructs(threadNum, raw->rows, raw->cols, raw->frames);
//	GenerateMask(mMask);
//}

void ComparatorNoiseCorrector::CorrectComparatorNoise(short *_image, int _rows, int _cols, int _frames,
		Mask *_mask,bool verbose, bool aggressive_correction, bool beadfind_image, int threadNum)
{
	char *allocPtr;

	image=_image;

//   double allocTime,maskTime,mainTime,aggTime,totalTime;
//   double wholeStart,start;
//   wholeStart = start = CNCTimer();
   allocPtr=AllocateStructs(threadNum, _rows, _cols, _frames);
//   allocTime = CNCTimer()-start;

//   start = CNCTimer();
   if(_mask == NULL)
	   GenerateMask(mMask);
   else
	   GenerateIntMask(_mask);
//   maskTime = CNCTimer()-start;
//   start = CNCTimer();


   if (aggressive_correction)
     NNSpan = 4;
   else
      NNSpan = 1;

   if (!beadfind_image)
   {
      CorrectComparatorNoise_internal( verbose, aggressive_correction);
//	  mainTime = CNCTimer()-start;

      if (aggressive_correction)
      {
         int blk_size = 96;
         int sub_blocks = rows / blk_size;
//         start=CNCTimer();

         if (blk_size * sub_blocks < rows) {
           sub_blocks++;
         }

         for (int blk = 0; blk < sub_blocks; blk++)
         {
            int row_start = blk * blk_size;
            int row_end = (blk + 1) * blk_size;

            if (blk == sub_blocks - 1) row_end = rows;

            CorrectComparatorNoise_internal( verbose, aggressive_correction, row_start, row_end);
         }
//     	  aggTime = CNCTimer()-start;
      }
   }
   else
   {
      // trick correction into only doing hf noise correction as this is all that works for beadfind images
      CorrectComparatorNoise_internal( verbose, aggressive_correction, 0, rows);
   }
//   totalTime = CNCTimer()-wholeStart;
//   printf("CNC took %.2lf alloc=%.2f mask=%.2f main=%.2f agg=%.2f sum=%.2f apply=%.2f tm1=%.2f tm2=%.2f(%.2f/%.2f) nn=%.2f msk=%.2f pca=%.2f\n",
//		   totalTime,allocTime,maskTime,mainTime,aggTime,sumTime,applyTime,tm1,tm2,tm2_1,tm2_2,nnsubTime,mskTime,tm2_3);

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
		 bool verbose, bool aggressive_correction, int row_start, int row_end)
{
  int phase;
  bool hfonly = false;

  memset(mComparator_sigs,0,mComparator_sigs_len);
  memset(mComparator_noise,0,mComparator_noise_len);
  memset(mComparator_hf_noise,0,mComparator_hf_noise_len);
  memset(mAvg_num,0,mAvg_num_len);
  memset(mComparator_mask,0,mComparator_mask_len);
  memset(mComparator_hf_mask,0,mComparator_hf_mask_len);
  memset(mCorrection,0,mCorrection_len);

  if (row_start == -1)
  {
	row_start = 0;
	row_end = rows;
	hfonly = false;
  }
  else
	hfonly = true;


  // first, create the average comparator signals
  // making sure to avoid pinned pixels
  SumColumns(row_start, row_end);
  
  double startTime=CNCTimer();

  // subtract DC offset from average comparator signals
  SetMeanToZero(mComparator_sigs);

  // now figure out which pair of signals go together
  // this function also combines pairs of signals accordingly
  // from this point forward, there are only cols*2 signals to deal with
  phase = DiscoverComparatorPhase(mComparator_sigs,cols*4,frames,hfonly);

  // change the data to be column major for the rest of the functions...
  TransposeData();

  tm1 += CNCTimer()-startTime;
  startTime = CNCTimer();
  //special case of rms==0, assign phase to -1
  if(phase != -1){
      // mask comparators that don't contain any un-pinned pixels
      for ( int cndx=0;cndx < cols*2;cndx++ )
      {
        if ( mAvg_num[cndx] == 0 )
        {
          mComparator_mask[cndx] = 1;
          continue;
        }
      }
      double tm2_1_startTime = CNCTimer();

      // now neighbor-subtract the comparator signals
      NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,mComparator_mask,NNSpan,cols*2,frames);

      // measure noise in the neighbor-subtracted signals
      CalcComparatorSigRMS(mComparator_rms,mComparator_noise,cols*2,frames);

      // find the noisiest 10%
      MaskIQR(mComparator_mask,mComparator_rms,cols*2);

      // neighbor-subtract again...avoiding noisiest 10%
      NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,mComparator_mask,NNSpan,cols*2,frames);

      // measure noise in the neighbor-subtracted signals
      CalcComparatorSigRMS(mComparator_rms,mComparator_noise,cols*2,frames);

      // reset mComparator_mask
      memset(mComparator_mask,0,mComparator_mask_len);
      for ( int cndx=0;cndx < cols*2;cndx++ )
      {
        if ( mAvg_num[cndx] == 0 )
        {
          mComparator_mask[cndx] = 1;
        }
      }
      MaskIQR(mComparator_mask,mComparator_rms,cols*2, verbose);

      // neighbor-subtract again...avoiding noisiest 10%
      NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,mComparator_mask,NNSpan,cols*2,frames);

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
         memcpy(mComparator_hf_noise,mComparator_noise,sizeof(float)*cols*2*frames);
         HighPassFilter(mComparator_hf_noise,cols*2,frames,10);
         tm2_3 += CNCTimer() - tm2_3_startTime;

         // neighbor-subtract again...now with some rejection of what we think the noise is
         NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,mComparator_mask,NNSpan,cols*2,frames,mComparator_hf_noise);

         // measure noise in the neighbor-subtracted signals
         CalcComparatorSigRMS(mComparator_rms,mComparator_noise,cols*2,frames);


         // reset mComparator_mask
         memset(mComparator_mask,0,mComparator_mask_len);
         for ( int cndx=0;cndx < cols*2;cndx++ )
         {
           if ( mAvg_num[cndx] == 0 )
           {
             mComparator_mask[cndx] = 1;
           }
         }
         
   //      MaskIQR(mComparator_mask,mComparator_rms,cols*2, verbose);
         MaskUsingDynamicStdCutoff(mComparator_mask,mComparator_rms,cols*2,1.0f);

         // even if some comparators didn't make the cut with the raw noise signal
         // we can correct more agressively if we put the noise signal through the high pass filter

         // redo the high-pass fitler
         memcpy(mComparator_hf_noise,mComparator_noise,sizeof(float)*cols*2*frames);
         // get first principal component
         GetPrincComp(mPcomp,mComparator_hf_noise,mComparator_mask,cols*2,frames);
         FilterUsingPrincComp(mComparator_hf_noise,mPcomp,cols*2,frames);

         // measure high frequency noise
         CalcComparatorSigRMS(mComparator_hf_rms,mComparator_hf_noise,cols*2,frames);
         for ( int cndx=0;cndx < cols*2;cndx++ )
         {
           if ( mAvg_num[cndx] == 0 )
           {
             mComparator_hf_mask[cndx] = 1;
           }
         }
         MaskUsingDynamicStdCutoff(mComparator_hf_mask,mComparator_hf_rms,cols*2,2.0f);
         tm2_2 += CNCTimer()-tm2_2_startTime;
         
      }

      // blanks comparator signal averages that didn't have any pixels to average (probably redundant)
      for ( int cndx=0;cndx < cols*2;cndx++ )
      {
        float *cptr;

        if ( mAvg_num[cndx] == 0 ) {
          // get a pointer to where we will build the comparator signal average
          cptr = mComparator_sigs + cndx*frames;
          memset(cptr,0,sizeof(float[frames]));
        }
      }
      tm2 += CNCTimer()-startTime;
#if 1
      BuildCorrection(hfonly);

	  ApplyCorrection(phase, row_start, row_end, mCorrection);
#else
      // now subtract each neighbor-subtracted comparator signal from the
      // pixels that are connected to that comparator
      int i=row_start*cols;
      for ( int y=row_start;y<row_end;y++ ) {
        for ( int x=0;x<cols;x++ ) {
          int cndx;
          int frame;
          float *cptr;

          // if this pixel is pinned..skip it
          if ( ( mMask[i] != 0 )  )
          {
            // figure out which comparator this pixel belongs to
            cndx = 2*x;
            if (( (y&0x3) == (0+phase) ) || ( (y&0x3) == (1+phase) ))
                cndx++;

            //only perform correction on noisy comparators;
            if(mComparator_mask[cndx] && !hfonly){
              // get a pointer to where we will build the comparator signal average
              cptr = mComparator_noise + cndx*frames;
              // subtract nn comparator signal from this pixel's data
              for ( frame=0;frame<frames;frame++ )
                image[frame*rows*cols+i] -= cptr[frame];
            } else if (mComparator_hf_mask[cndx])
            {
              // get a pointer to where we will build the comparator signal average
              cptr = mComparator_hf_noise + cndx*frames;
              // subtract nn comparator signal from this pixel's data
              for ( frame=0;frame<frames;frame++ )
                image[frame*rows*cols+i] -= cptr[frame];
            }
          }

          i++;
        }
      }

#endif	  
  }
}

// changes the data to be column major instead of comparator major
void ComparatorNoiseCorrector::TransposeData()
{
	int x,frame,comparator;

	// flip mAvg_num
	{
		for(comparator=0;comparator<2;comparator++)
		{
			for(x=0;x<cols;x++)
				mAvg_num[x*2+comparator] = mAvg_num[(comparator+2)*cols+x];
		}
	}
	// flip mComparator_sigs
    {
		for(comparator=0;comparator<2;comparator++)
		{
			for(frame=0;frame<frames;frame++)
			{
				for(x=0;x<cols;x++)
					mComparator_sigs[(2*x+comparator)*frames+frame] = mComparator_sigs[(comparator+2)*cols*frames + frame*cols + x];
			}
		}
    }
}

// generate a 16-bit array of correction signals.  one for each comparator
// rounding is weird here.  Its not the same as the old function.
void ComparatorNoiseCorrector::BuildCorrection(bool hfonly)
{
	int x, frame, comparator, cndx;
	short int *mcPtr;
	float val;

	// Build mCorrection from mComparator_noise and mComparator_hf_noise

	for (comparator = 0; comparator < 2; comparator++)
	{
		for (frame = 0; frame < frames; frame++)
		{
			mcPtr = &mCorrection[comparator * cols * frames + frame * cols];
			for (x = 0; x < cols; x++)
			{
				// figure out which comparator this pixel belongs to
				cndx = 2 * x + comparator;

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
	double startTime = CNCTimer();

	// now subtract each neighbor-subtracted comparator signal from the
	// pixels that are connected to that comparator

	for (frame = 0; frame < frames; frame++)
	{
		for (y = row_start; y < row_end; y++)
		{
			srcPtr = (v8s *)(&image[frame * frameStride + y * cols]);
			if (( (y&0x3) == (0+phase) ) || ( (y&0x3) == (1+phase) ))
				corPtr = (v8s *) (correction + frames*cols + frame*cols);
			else
				corPtr = (v8s *) (correction + frame*cols);

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
				float *sumPtr = (float *) (mAvg_num + comparator * cols * frames + frame * cols);
				float *mskPtr = (float *) (mMask + cols*y);
				for (x = 0; x < cols; x++)
				{
					valU= (float)srcPtr[x];
					dstPtr[x] += valU;
					sumPtr[x] += mskPtr[x];
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
	for (comparator=0;comparator<4;comparator++)
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

#ifndef BB_DC
void ComparatorNoiseCorrector::CorrectComparatorNoiseThumbnail(RawImage *raw,Mask *mask, int regionXSize, int regionYSize, bool verbose) {
  CorrectComparatorNoiseThumbnail(raw->image, raw->rows, raw->cols, raw->frames, mask, regionXSize, regionYSize, verbose);
}

void ComparatorNoiseCorrector::CorrectComparatorNoiseThumbnail(short *_image, int _rows, int _cols, int _frames, Mask *mask, int regionXSize, int regionYSize, bool verbose)
{
  time_t cnc_start;
  time ( &cnc_start );
  MemUsage ( "Starting Comparator Noise Correction" );

  image=_image;

  char *allocPtr=AllocateStructs(-1, _rows, _cols, _frames);
  int frameStride = rows * cols;

  if( cols%regionXSize != 0 || rows%regionYSize != 0){
    //skip correction
    fprintf (stdout, "Region sizes are not compatible with image(%d x %d): %d x %d", rows, cols, regionYSize, regionXSize);
  }
  int nXPatches = cols / regionXSize;
  int nYPatches = rows / regionYSize;
  int phase;

//  Allocate(cols * 4 * frames);
  for(int pRow = 0; pRow < nYPatches; pRow++){
    for(int pCol = 0; pCol < nXPatches; pCol++){
        if(verbose)
          fprintf (stdout, "Patch y: %d, Patch x: %d\n", pRow, pCol);
        memset(mComparator_sigs,0,sizeof(float) * regionXSize*4*frames);
        memset(mComparator_noise,0,sizeof(float) * regionXSize*2*frames);
        memset(mAvg_num,0,sizeof(float)*regionXSize*4);
        memset(mComparator_mask,0,sizeof(int)*regionXSize*2);
        memset(mComparator_rms,0,sizeof(mComparator_rms));

        // first, create the average comparator signals
        // making sure to avoid pinned pixels

        for ( int y=0;y<regionYSize;y++ ) {
          for(int x=0;x<regionXSize;x++ ) {
            // if this pixel is pinned..skip it
            int imgInd = cols * (y+pRow*regionYSize) + x + pCol*regionXSize;
            if ( (( *mask ) [imgInd] & MaskPinned)==0 )
            {
              // figure out which comparator this pixel belongs to
              // since we don't know the phase just yet, we first split each column
              // up into 4 separate comparator signals, even though
              // there are really only 2 of them
              int cndx = 4*x + (y&0x3);

              // get a pointer to where we will build the comparator signal average
              float *cptr = mComparator_sigs + cndx*frames;

              // add this pixels' data in
              for (int frame=0;frame<frames;frame++ )
                cptr[frame] += image[frame*frameStride+imgInd];

              // count how many we added
              mAvg_num[cndx]++;
            }
          }
        }

        // divide by the number to make a proper average
        for ( int cndx=0;cndx < regionXSize*4;cndx++ )
        {
          if ( mAvg_num[cndx] == 0 )
            continue;

          // get a pointer to where we will build the comparator signal average
          float *cptr = mComparator_sigs + cndx*frames;

          // divide by corresponding count, extreme case: divide by zero if all pixels are pinned
          if(mAvg_num[cndx] != 0){
            for (int frame=0;frame<frames;frame++ ){
              cptr[frame] /= mAvg_num[cndx];
            }
          }
        }

        // subtract DC offset from average comparator signals
        for ( int cndx=0;cndx < regionXSize*4;cndx++ )
        {
          float *cptr;
          float dc = 0.0f;

          // get a pointer to where we will build the comparator signal average
          cptr = mComparator_sigs + cndx*frames;

          for (int frame=0;frame<frames;frame++ )
            dc += cptr[frame];

          dc /= frames;

          // subtract dc offset
          for (int frame=0;frame<frames;frame++ )
            cptr[frame] -= dc;
        }

        // now figure out which pair of signals go together
        // this function also combines pairs of signals accordingly
        // from this point forward, there are only cols*2 signals to deal with
        phase = DiscoverComparatorPhase_tn(mComparator_sigs,regionXSize*4,frames,false);

        if(phase == -1){
//          fprintf (stdout, "Comparator Noise Correction skipped\n");
          continue;
        }

        // mask comparators that don't contain any un-pinned pixels
        for ( int cndx=0;cndx < regionXSize*2;cndx++ )
        {
          if ( mAvg_num[cndx] == 0 )
          {
            mComparator_mask[cndx] = 1;
            continue;
          }
        }

        // now neighbor-subtract the comparator signals
        NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,mComparator_mask,NNSpan, regionXSize*2,frames);

        // measure noise in the neighbor-subtracted signals
        CalcComparatorSigRMS(mComparator_rms,mComparator_noise,regionXSize*2,frames);

        // find the noisiest 10%
        MaskIQR(mComparator_mask,mComparator_rms,regionXSize*2);

        // neighbor-subtract again...avoiding noisiest 10%
        NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,mComparator_mask,NNSpan,regionXSize*2,frames);

        // measure noise in the neighbor-subtracted signals
        CalcComparatorSigRMS(mComparator_rms,mComparator_noise,regionXSize*2,frames);

        // reset mComparator_mask
        memset(mComparator_mask,0,sizeof(mComparator_mask));
        for ( int cndx=0;cndx < regionXSize*2;cndx++ )
        {
          if ( mAvg_num[cndx] == 0 )
          {
            mComparator_mask[cndx] = 1;
          }
        }
        MaskIQR(mComparator_mask,mComparator_rms,regionXSize*2, verbose);

        // neighbor-subtract again...avoiding noisiest 10%
        NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,mComparator_mask,NNSpan,regionXSize*2,frames);

        for ( int cndx=0;cndx < regionXSize*2;cndx++ )
        {
          float *cptr;

          if ( mAvg_num[cndx] == 0 ) {
            // get a pointer to where we will build the comparator signal average
            cptr = mComparator_sigs + cndx*frames;
            memset(cptr,0,sizeof(float[frames]));
          }
        }

        // now subtract each neighbor-subtracted comparator signal from the
        // pixels that are connected to that comparator
        for ( int y=0;y<regionYSize;y++ ) {
          for(int x=0;x<regionXSize;x++ ) {
            int imgInd = cols * (y+pRow*regionYSize) + x + pCol*regionXSize;

            // if this pixel is pinned..skip it
            if ( (( *mask ) [imgInd] & MaskPinned)==0 )
            {
              // figure out which comparator this pixel belongs to
              int cndx = 2*x;
              if (( (y&0x3) == (0+phase) ) || ( (y&0x3) == (1+phase) ))
                  cndx++;

              //only perform correction on noisy comparators;
              if(mComparator_mask[cndx]){
                // get a pointer to where we will build the comparator signal average
                float *cptr = mComparator_noise + cndx*frames;
                // subtract nn comparator signal from this pixel's data
                for ( int frame=0;frame<frames;frame++ )
                  image[frame*frameStride+imgInd] -= cptr[frame];
              }
            }
          }
        }
    }
  }

  FreeStructs(-1,false,allocPtr);

  MemUsage ( "After Comparator Noise Correction" );
  time_t cnc_end;
  time ( &cnc_end );

  fprintf (stdout, "Comparator Noise Correction: %0.3lf sec.\n", difftime(cnc_end, cnc_start));
}
#endif

int ComparatorNoiseCorrector::DiscoverComparatorPhase(float *psigs,int n_comparators,int nframes, bool hfonly)
{
  float phase_rms[2];
  int phase;

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

      for ( int frame=0;frame < nframes;frame++ )
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

  if ( phase_rms[0] < phase_rms[1] )
    phase = 0;
  else
    phase = 1;

  //get phase_rms values to check how reliable it is
//  fprintf (stdout, "Phase: %d; RMS Phase Calcs = %f vs %f\n", phase, phase_rms[0], phase_rms[1]);

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

    for ( int frame=0;frame < nframes;frame++ )
    {
      // beware...we are doing this in place...need to be careful
      float sum1 = scale1*(cptr_1a[frame*cols]*num_1a+cptr_1b[frame*cols]*num_1b);
      float sum2 = scale2*(cptr_2a[frame*cols]*num_2a+cptr_2b[frame*cols]*num_2b);
	  
//	  if(hfonly == 1)
//	  	printf("%d/%d) sum1 = %.2f %f  %.2f * %d + %.2f * %d\n",i,frame,sum1,scale1,cptr_1a[frame*cols],num_1a,cptr_1b[frame*cols],num_1b);
	  
      SIGS_ACC(i,2,frame) = sum1;
      SIGS_ACC(i,3,frame) = sum2;
    }

    mAvg_num_ACC(i,2) = num1;
    mAvg_num_ACC(i,3) = num2;
  }

  return phase;
}

int ComparatorNoiseCorrector::DiscoverComparatorPhase_tn(float *psigs,int n_comparators,int nframes, bool hfonly)
{
  float phase_rms[2];
  int phase;

  for ( phase = 0;phase < 2;phase++ )
  {
    phase_rms[phase] = 0.0f;
    int rms_num = 0;

    for ( int i=0;i < n_comparators;i+=4 )
    {
      float *cptr_1a;
      float *cptr_1b;
      float *cptr_2a;
      float *cptr_2b;
      float rms_sum = 0.0f;

      // have to skip any columns that have all pinned pixels in any subset-average
//      if (( mAvg_num[i] == 0 ) && ( mAvg_num[i] == 1 ) && ( mAvg_num[i] == 2 ) && ( mAvg_num[i] == 3 ))
      if (( mAvg_num[i] == 0 ) && ( mAvg_num[i + 1] == 0 ) && ( mAvg_num[i + 2] == 0 ) && ( mAvg_num[i + 3] == 0 )){
//        fprintf (stdout, "Noisy column: %d; Comparator: %d.\n", i/4, i&0x3);
        continue;
      }

      // get a pointers to the comparator signals
      if ( phase==0 ) {
        cptr_1a = psigs + (i+2)*nframes;
        cptr_1b = psigs + (i+3)*nframes;
        cptr_2a = psigs + (i+0)*nframes;
        cptr_2b = psigs + (i+1)*nframes;
      }
      else
      {
        cptr_1a = psigs + (i+0)*nframes;
        cptr_1b = psigs + (i+3)*nframes;
        cptr_2a = psigs + (i+1)*nframes;
        cptr_2b = psigs + (i+2)*nframes;
      }

      for ( int frame=0;frame < nframes;frame++ )
      {
        rms_sum += (cptr_1a[frame]-cptr_1b[frame])*(cptr_1a[frame]-cptr_1b[frame]);
        rms_sum += (cptr_2a[frame]-cptr_2b[frame])*(cptr_2a[frame]-cptr_2b[frame]);
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

  if ( phase_rms[0] < phase_rms[1] )
    phase = 0;
  else
    phase = 1;


  //get phase_rms values to check how reliable it is
//  fprintf (stdout, "Phase: %d; RMS Phase Calcs = %f vs %f\n", phase, phase_rms[0], phase_rms[1]);

  // now combine signals according to the detected phase
  int cndx=0;
  for ( int i=0;i < n_comparators;i+=4 )
  {
    int ndx[4];
    float *cptr_1a;
    float *cptr_1b;
    float *cptr_2a;
    float *cptr_2b;
    int num_1a,num_1b,num_2a,num_2b;
    float *cptr_1;
    float *cptr_2;
    int num1;
    int num2;
    float scale1;
    float scale2;

    // get a pointers to the comparator signals
    if ( phase==0 ) {
      ndx[0] = i+2;
      ndx[1] = i+3;
      ndx[2] = i+0;
      ndx[3] = i+1;
    }
    else
    {
      ndx[0] = i+0;
      ndx[1] = i+3;
      ndx[2] = i+1;
      ndx[3] = i+2;
    }
    cptr_1a = psigs + ndx[0]*nframes;
    cptr_1b = psigs + ndx[1]*nframes;
    cptr_2a = psigs + ndx[2]*nframes;
    cptr_2b = psigs + ndx[3]*nframes;
    num_1a = mAvg_num[ndx[0]];
    num_1b = mAvg_num[ndx[1]];
    num_2a = mAvg_num[ndx[2]];
    num_2b = mAvg_num[ndx[3]];

    num1 = num_1a+num_1b;
    num2 = num_2a+num_2b;

    cptr_1 = psigs + (cndx+0)*nframes;
    cptr_2 = psigs + (cndx+1)*nframes;

    if ( num1 > 0 )
      scale1 = 1.0f/((float)num1);
    else
      scale1 = 0.0f;

    if ( num2 > 0 )
      scale2 = 1.0f/((float)num2);
    else
      scale2 = 0.0f;

    for ( int frame=0;frame < nframes;frame++ )
    {
      // beware...we are doing this in place...need to be careful
      float sum1 = scale1*(cptr_1a[frame]*num_1a+cptr_1b[frame]*num_1b);
      float sum2 = scale2*(cptr_2a[frame]*num_2a+cptr_2b[frame]*num_2b);

//      if(hfonly == 1)
//	  printf("%d/%d) sum1 = %.2f %f  %.2f * %d + %.2f * %d\n",i/4,frame,sum1,scale1,cptr_1a[frame],num_1a,cptr_1b[frame],num_1b);

      cptr_1[frame] = sum1;
      cptr_2[frame] = sum2;
    }

    mAvg_num[cndx+0] = num1;
    mAvg_num[cndx+1] = num2;
    cndx+=2;
  }

  return phase;
}
// now neighbor-subtract the comparator signals
void ComparatorNoiseCorrector::NNSubtractComparatorSigs(float *pnn,float *psigs,int *mask,int span,int n_comparators,int nframes,float *hfnoise)
{
    double startTime = CNCTimer();

  float nn_avg[nframes];
  float zero_sig[nframes];
  
  memset(zero_sig,0,sizeof(zero_sig));

  for ( int i=0;i < n_comparators;i++ )
  {
    int nn_cnt=0;
    float *cptr;
    float *n_cptr;
    float *chfptr=NULL;
    float *n_chfptr=NULL;
    float *nncptr;
    float centroid = 0.0f;
    int i_c0 = i & ~0x1;
    memset(nn_avg,0,sizeof(nn_avg));

	// in case we weren't provided with high frequency noise correction for NNs, use all zeros instead
	chfptr = zero_sig;
	n_chfptr = zero_sig;
    
    // rounding down the starting point and adding one to the rhs properly centers
    // the neighbor average about the central column...except in cases where columns are
    // masked within the neighborhood.

    //same column but the other comparator
    int offset[2] = {1, 0};
    int theOtherCompInd = i_c0 + offset[i-i_c0];
    if(!mask[theOtherCompInd]){
        // get a pointer to the comparator signal
        cptr = psigs + theOtherCompInd *nframes;

        if (hfnoise != NULL)
        {
            chfptr = hfnoise + theOtherCompInd *nframes;

            // add it to the average
            for ( int frame=0;frame < nframes;frame++ )
              nn_avg[frame] += cptr[frame] - chfptr[frame];
        }
        else
        {
            // add it to the average
            for ( int frame=0;frame < nframes;frame++ )
              nn_avg[frame] += cptr[frame];
        }


        nn_cnt++;
        centroid += theOtherCompInd;
    }

    for(int s = 1; s <= span; s++){
      //i_c0 is even number
      //odd
      int cndx = i_c0 - 2*s + 1;
      int n_cndx = i_c0 + 2*s + 1;
      if(!(cndx < 0  || n_cndx >= n_comparators || mask[cndx] || mask[n_cndx])) {
          // get a pointer to the comparator signal
          cptr = psigs + cndx*nframes;
          n_cptr = psigs + n_cndx*nframes;

          if (hfnoise != NULL)
          {
              chfptr = hfnoise + cndx*nframes;
              n_chfptr = hfnoise + n_cndx*nframes;
          }
          // add it to the average
          for ( int frame=0;frame < nframes;frame++ )
          {
              nn_avg[frame] += cptr[frame] - chfptr[frame];
              nn_avg[frame] += n_cptr[frame] - n_chfptr[frame];
          }

          nn_cnt++;
          centroid += cndx;
          nn_cnt++;
          centroid += n_cndx;
      }

      cndx--;
      n_cndx--;
      //even, symmetric
      if(!(cndx < 0  || n_cndx >= n_comparators || mask[cndx] || mask[n_cndx])) {
          // get a pointer to the comparator signal
          cptr = psigs + cndx*nframes;
          n_cptr = psigs + n_cndx*nframes;

          if (hfnoise != NULL)
          {
              chfptr = hfnoise + cndx*nframes;
              n_chfptr = hfnoise + n_cndx*nframes;
          }
          // add it to the average
          for ( int frame=0;frame < nframes;frame++ )
          {
              nn_avg[frame] += cptr[frame] - chfptr[frame];
              nn_avg[frame] += n_cptr[frame] - n_chfptr[frame];
          }
          nn_cnt++;
          centroid += cndx;
          nn_cnt++;
          centroid += cndx;
      }
    }

    if (( nn_cnt > 0 ) )
    {
      for ( int frame=0;frame < nframes;frame++ )
        nn_avg[frame] /= nn_cnt;

      // now subtract the neighbor average
      cptr = psigs + i*nframes;
      nncptr = pnn + i*nframes;
      for ( int frame=0;frame < nframes;frame++ )
        nncptr[frame] = cptr[frame] - nn_avg[frame];
    }
    else
    {
//      fprintf (stdout, "Default noise of 0 is set: %d\n", i);
      // not a good set of neighbors to use...just blank the correction
      // signal and do nothing.
      nncptr = pnn + i*nframes;
      for ( int frame=0;frame < nframes;frame++ )
        nncptr[frame] = 0.0f;
    }
  }
  nnsubTime += CNCTimer() - startTime;
}

void ComparatorNoiseCorrector::HighPassFilter(float *pnn,int n_comparators,int nframes,int span)
{
#ifdef __INTEL_COMPILER
  v8f_u* trc_scratch = new v8f_u[nframes];
#else
  v8f_u trc_scratch[nframes];
#endif

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
	int frameStride=rows*cols;
	int idx;

	for(idx=0;idx<frameStride;idx++)
	{
		if((*_mask)[idx] & MaskPinned)
			mMask[idx] = 0;
		else
			mMask[idx] = 1;
	}
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
#ifdef __AVX__
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


