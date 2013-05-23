/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iostream>
#include <assert.h>
#include "PinnedInFlow.h"
#include "Utils.h"
#include "IonErr.h"


using namespace std;

PinnedInFlow::PinnedInFlow(Mask *maskPtr, int numFlows)
{ 
  mNumWells = maskPtr->W() *maskPtr->H();
  mNumFlows = numFlows;

}

PinnedInFlow::~PinnedInFlow()
{
  mPinnedInFlow.clear();
  mPinsPerFlow.clear();
}


void PinnedInFlow::Initialize (Mask *maskPtr)
{
  // allocate
  mPinnedInFlow.resize(mNumWells);
  mPinsPerFlow.resize(mNumFlows);

  // wells marked as -1 are valid unpinned wells
  // wells that become pinned as flows load are set to that flow value
  // initial pinned wells are set to flow zero using the beadfind mask
  int w = maskPtr->W();
  int h = maskPtr->H();

  // all empty wells set to -1
  int nonpinned = 0;
  for (int y=0; y<h; y++)
  {
    for (int x=0; x<w; x++)
    {
      int i = maskPtr->ToIndex (y,x);
      if (maskPtr->Match (x,y,MaskPinned))   // pinned
      {
        mPinnedInFlow[i] = 0;
      }
      else   // not pinned
      {
        mPinnedInFlow[i] = -1;  // not-pinned coming into flow 0
        nonpinned++;
      }
    }
  }
  // fprintf(stdout, "PinnedInFlow::Initialize: %d non-pinned wells prior to flow 0\n", nonpinned);

  // now initialize the counts per flow to 0 as well
  for (int i=0; i<mNumFlows; i++)
    mPinsPerFlow[i] = 0;

}

void PinnedInFlow::UpdateMaskWithPinned (Mask *maskPtr)
{
  // side-effect is to set the well (pixel) in maskPtr with MaskPinned if 
  // the well was pinned in any flow
  int w = maskPtr->W();
  int h = maskPtr->H();

  for (int y=0; y<h; y++)
  {
    for (int x=0; x<w; x++)
    {
      int i = maskPtr->ToIndex (y,x);
      if (mPinnedInFlow[i] >= 0)
        (*maskPtr) [i] = MaskPinned; // this pixel is pinned high or low
    }
  }
}

void PinnedInFlow::SetPinned(int idx, int flow)
{
	int16_t currFlow;
	currFlow = mPinnedInFlow[idx];
	// pixel is pinned high or low
	if ((currFlow < 0) | (currFlow > flow))   // new pins per flow
	{
		currFlow = flow;
		mPinnedInFlow[idx] = flow;
		while (((short volatile *) &mPinnedInFlow[0])[idx] > currFlow)
		{
			// race condition, a later flow already updated this well, keep trying
			((short volatile *) &mPinnedInFlow[0])[idx] = currFlow;
		}
		mPinsPerFlow[flow]++; // this needs to be protected...
	}
}

#define PPIX_VEC_SIZE 8
#define PPIX_VEC_SIZE_B 32
#define MAX_GAIN_CORRECT 16383

typedef float vecf_t __attribute__ ((vector_size (PPIX_VEC_SIZE_B)));
typedef union {
	float A[PPIX_VEC_SIZE];
	vecf_t V;
}vecf_u;

int PinnedInFlow::Update (int flow, Image *img, float *gainPtr)
{
  // if any well at (x,y) is first pinned in this flow & this flow's img,
  // set the value in mPinnedInFlow[x,y] to that flow
  // const RawImage *raw = img->GetImage();
  // int rows = raw->rows;
  // int cols = raw->cols;
  const RawImage *raw = img->GetImage();    // the raw image

  int rows = raw->rows;
  int cols = raw->cols;
  int frames = raw->frames;
  int x, y, frame;
//  int pinnedLowCount = 0;
//  int pinnedHighCount = 0;
  int i = 0;
  const uint32_t pinHigh = GetPinHigh();
  const uint32_t pinLow = GetPinLow();
  int16_t *pixPtr;
  uint32_t val;
  double fval;
  bool isLow,isHigh;
  uint32_t frameStride=rows*cols;
  float gainFactor;
  uint32_t pinned;

//  double stopT,startT = TinyTimer();

  if (rows <= 0 || cols <= 0)
  {
    cout << "Why bad row/cols for flow: " << flow << " rows: " << rows << " cols: " << cols << endl;
    exit (EXIT_FAILURE);
  }

  if((cols % PPIX_VEC_SIZE) != 0)
  {
	// check for pinned pixels in this flow
	for (y = 0; y < rows; y++) {
		for (x = 0; x < cols; x++) {
			i = y * cols + x;
			pixPtr = raw->image + i;
			if (gainPtr)
				gainFactor = gainPtr[i];
			else
				gainFactor = 1.0f;
			pinned=0;

			for (frame = 0; frame < frames; frame++) {
				fval = (float) pixPtr[frame * frameStride];
				fval *= gainFactor;
				val = fval;
				if (val > MAX_GAIN_CORRECT)
					val = MAX_GAIN_CORRECT;

				pixPtr[frame * frameStride] = (int16_t) val;

				isLow = val <= pinLow;
				isHigh = val >= pinHigh;
				if (!pinned && (isLow || isHigh)) {
					pinned=1;
					SetPinned(i, flow);
				}
			} // end frame loop
		}  // end x loop
	} // end y loop
  }
  else
  {
	  int j,fs;
	  vecf_u gainFactorV;
	  vecf_u fvalV;
	  uint32_t pinnedA[PPIX_VEC_SIZE];

//	  printf("Doing vectorized code\n");
		// check for pinned pixels in this flow
		for (y = 0; y < rows; y++) {
			for (x = 0; x < cols; x+=PPIX_VEC_SIZE) {
				i = y * cols + x;
				pixPtr = raw->image + i;
				for(j=0;j<PPIX_VEC_SIZE;j++)
				{
					pinnedA[j]=0;
					if (gainPtr)
						gainFactorV.A[j] = gainPtr[i+j];
					else
						gainFactorV.A[j] = 1.0f;
				}

				for (frame = 0; frame < frames; frame++) {
					fs = frame*frameStride;
					for(j=0;j<PPIX_VEC_SIZE;j++){
						fvalV.A[j] = (float) pixPtr[fs + j];
					}
					fvalV.V *= gainFactorV.V;

					for(j=0;j<PPIX_VEC_SIZE;j++){
						val = (uint16_t)fvalV.A[j];
						if (val > MAX_GAIN_CORRECT)
							val = MAX_GAIN_CORRECT;
						pixPtr[fs + j] = val;

						if(!pinnedA[j] && (val >= pinHigh || val <= pinLow))
						{
							pinnedA[j]=1;
							SetPinned(i+j, flow);
						}
					}
				} // end frame loop
			}  // end x loop
		} // end y loop
  }
  // char s[512];
  // int n = sprintf(s,  "PinnedInFlow::UpdatePinnedWells: %d pinned pixels <=  %d or >= %d in flow %d (%d low, %d high)\n", (pinnedLowCount+pinnedHighCount), pinLow, pinHigh, flow, pinnedLowCount, pinnedHighCount);
  // assert(n<511);
  // fprintf (stdout,"%s", s);
//  mPinsPerFlow[flow] = pinnedLowCount+pinnedHighCount;

//  stopT = TinyTimer();
//  printf ( "PinnedInFlow::Update ... done %d pinned in %0.2lf sec\n",mPinsPerFlow[flow],stopT - startT );

  return (mPinsPerFlow[flow]);
}

int PinnedInFlow::Update (int flow, class SynchDat *img, float *gainPtr) {
  // if any well at (x,y) is first pinned in this flow & this flow's img,
  // set the value in mPinnedInFlow[x,y] to that flow
  // const RawImage *raw = img->GetImage();
  // int rows = raw->rows;
  // int cols = raw->cols;
  int rows = img->GetRows();
  int cols = img->GetCols();
  const short pinHigh = GetPinHigh();
  const short pinLow = GetPinLow();
  int pinned = 0;
  float gainFactor =1.0f;
  if (rows <= 0 || cols <= 0 ) {
      cout << "Why bad row/cols for flow: " << flow << " rows: " << rows << " cols: " << cols << endl;
      exit (EXIT_FAILURE);
  }
  for (size_t rIx = 0; rIx < img->GetNumBin(); rIx++) {
    TraceChunk &chunk = img->GetChunk(rIx);
    if (chunk.mWidth % PPIX_VEC_SIZE != 0) {
      for (size_t r = 0; r < chunk.mHeight; r++) {
	for (size_t c = 0; c < chunk.mWidth; c++) {
	  size_t chipIdx = (r + chunk.mRowStart) * cols + (c + chunk.mColStart);
	  int16_t *p = &chunk.mData[0] + r * chunk.mWidth + c;
	  if (gainPtr) {
	    gainFactor = gainPtr[chipIdx];
	  }
	  else {
	    gainFactor = 1.0f;
	  }
	  pinned = 0;
	  for (size_t frame = 0; frame < chunk.mDepth; frame++) {
	    //	    int val = min(MAX_GAIN_CORRECT, (int)(*p * gainFactor));
	    int val = (int)(*p * gainFactor);
	    if (val > MAX_GAIN_CORRECT)
	      val = MAX_GAIN_CORRECT;
	    *p = (int16_t)val;
	    bool isLow = val <= pinLow;
	    bool isHigh = val >= pinHigh;
	    if (!pinned && (isLow || isHigh)) {
	      pinned = 1;
	      SetPinned(chipIdx, flow);
	    }
	    p += chunk.mFrameStep;
	  }
	}
      }
    }
    else {
      int j;
      vecf_u gainFactorV;
      vecf_u fvalV;
      int32_t val;
      uint32_t pinnedA[PPIX_VEC_SIZE];
      for (size_t r = 0; r < chunk.mHeight; r++) {
	for (size_t c = 0; c < chunk.mWidth; c+=PPIX_VEC_SIZE) {
	  size_t chipIdx = (r + chunk.mRowStart) * cols + (c + chunk.mColStart);
	  int16_t *p = &chunk.mData[0] + r * chunk.mWidth + c;
	  for(j=0;j<PPIX_VEC_SIZE;j++){
	    pinnedA[j]=0;
	    if (gainPtr)
	      gainFactorV.A[j] = gainPtr[chipIdx+j];
	    else
	      gainFactorV.A[j] = 1.0f;
	  }
	  for (size_t frame = 0; frame < chunk.mDepth; frame++) {      
	    int16_t *t = p;
	    for (j = 0; j < PPIX_VEC_SIZE; j++) {
	      fvalV.A[j] = (float) *t;
	      t++;
	    }
	    t = p;
	    fvalV.V *= gainFactorV.V;
	    for(j=0; j < PPIX_VEC_SIZE; j++) {
	      val = (int32_t)fvalV.A[j];
	      if (val > MAX_GAIN_CORRECT)
		val = MAX_GAIN_CORRECT;
	      *t = (int16_t)val;
	      t++;
	      if(!pinnedA[j] && (val >= pinHigh || val <= pinLow)) {
		  pinnedA[j]=1;
		  SetPinned(chipIdx+j, flow);
	      }
	    }
	    p += chunk.mFrameStep;
	  }
	}
      }
    }
  }
  return mPinsPerFlow[flow];
}

// int PinnedInFlow::Update (int flow, class SynchDat *img, float *gainPtr)
// {
//   // if any well at (x,y) is first pinned in this flow & this flow's img,
//   // set the value in mPinnedInFlow[x,y] to that flow
//   // const RawImage *raw = img->GetImage();
//   // int rows = raw->rows;
//   // int cols = raw->cols;
//   int rows = img->GetRows();
//   int cols = img->GetCols();
//   int pinnedLowCount = 0;
//   int pinnedHighCount = 0;
//   const short pinHigh = GetPinHigh();
//   const short pinLow = GetPinLow();

//   if (rows <= 0 || cols <= 0)
//   {
//     cout << "Why bad row/cols for flow: " << flow << " rows: " << rows << " cols: " << cols << endl;
//     exit (EXIT_FAILURE);
//   }
//   for (size_t rIx = 0; rIx < img->GetNumBin(); rIx++) {
//     TraceChunk &chunk = img->GetChunk(rIx);
//     for (size_t r = 0; r < chunk.mHeight; r++) {
//       for (size_t c = 0; c < chunk.mWidth; c++) {
// 	size_t chipIdx = (r + chunk.mRowStart) * cols + (c + chunk.mColStart);
// 	short currFlow = mPinnedInFlow[chipIdx];
// 	size_t idx = r * chunk.mWidth + c;
// 	  for (size_t frame = 0; frame < chunk.mDepth; frame++) {
// 	    short val = chunk.mData[idx];
// 	    bool isLow = val <= pinLow;
// 	    bool isHigh = val >= pinHigh;
// 	    if (isLow || isHigh) {
// 	      // pixel is pinned high or low
// 	      if ( (currFlow < 0) | (currFlow > flow))   // new pins per flow
// 		{
// 		  currFlow = flow;
// 		  mPinnedInFlow[chipIdx] = flow;
// 		}
// 	      if (isLow)
// 		pinnedLowCount++;
// 	      else
// 		pinnedHighCount++;
// 	      break;
// 	    }
// 	    idx += chunk.mFrameStep;
// 	  }
// 	// Should this really just be spinning???
// 	while ( ( (short volatile *) &mPinnedInFlow[0]) [chipIdx] > currFlow) {
// 	  // race condition, a later flow already updated this well, keep trying
// 	  ( (short volatile *) &mPinnedInFlow[0]) [chipIdx] = currFlow;
// 	}
//       }
//     }
//   }
//   mPinsPerFlow[flow] = pinnedLowCount+pinnedHighCount;
//   return (pinnedLowCount+pinnedHighCount);
// }

void PinnedInFlow::DumpSummaryPinsPerFlow (char *experimentName)
{
  char fileName[2048];
  int n = sprintf (fileName,"%s/pinsPerFlow.txt", experimentName);
  assert (n<2048);

  FILE *fp = NULL;

  fopen_s (&fp, fileName, "w");
  if (!fp)
  {
    printf ("Could not open %s, err %s\n", fileName, strerror (errno));
  }
  else
  {
    for (int flow=0; flow < mNumFlows; flow++)
    {
      fprintf (fp, "%d\t%d\t%d\n", flow, mPinsPerFlow[flow], mNumWells);
    }
    fclose (fp);
  }
}


/*
void PinnedInFlow::DumpPinPerWell(char *experimentName, Mask *maskPtr)
{
  char fileName[2048];
  int n = sprintf (fileName, "%s/whenpinned.txt", experimentName);
  assert (n<2048);

  FILE *fp = NULL;
  if (fileName != NULL) {
    fopen_s (&fp, fileName, "w");
    if (!fp) {
      printf ("Could not open %s, err %s\n", fileName, strerror (errno));
    }
    else {
      int w = maskPtr->W();
      int h = maskPtr->H();
      for (int y=0;y<h;y++) {
 for (int x=0;x<w;x++) {
   int i = maskPtr->ToIndex(y,x);
   if (mPinnedInFlow[i]>=0) // saw a pin
     fprintf (fp, "%d\t%d\t%d\n", x, y, mPinnedInFlow[i]);
 }
      }
      fclose (fp);
    }
  }
}
*/
