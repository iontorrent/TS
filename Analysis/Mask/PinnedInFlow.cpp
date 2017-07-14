/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iostream>
#include <assert.h>
#include "PinnedInFlow.h"
#include "Utils.h"
#include "IonErr.h"
#include "ChipIdDecoder.h"
#include "Vecs.h"
#include <malloc.h>
#ifdef __AVX__
#include <xmmintrin.h>
#endif

using namespace std;

// get pin values high and low
short GetPinHigh()
{
  // should this be a supplied parameter like chip type?
  // no guarantee that chip type always defines pin values

  // default 16 bit
  short pin_high = 0x3fff;
  // default 14 bit
  if (ChipIdDecoder::IsProtonChip())
     pin_high = 16380;

  return (pin_high);
}

short GetPinLow()
{
  short pin_low = 0;
  return (pin_low);
}

void PinnedInFlow::InitMutex(){
  bool retry = false;
    do{
      int ierr  = pthread_mutex_init(&mutex_setPin, NULL);
      switch(ierr){
        case 0:
          //no error
          break;
        case EAGAIN:
          cout << "The system lacked the necessary resources (other than memory) to initialize another mutex" << endl;
          retry = !retry;
          break;
        case ENOMEM:
          cout << "Insufficient memory exists to initialize the mutex." << endl;
          break;
        case EPERM:
          cout << "The caller does not have the privilege to perform the operations needed to initialize the mutex" << endl;
          break;
        case EBUSY:
          cout << "The implementation has detected an attempt to re-initialize the object referenced by mutex, a previously initialized, but not yet destroyed, mutex." << endl;
          break;
        default:
          cout << " An error with unknown error code " << ierr << " occurred during mutex initialization" << endl;
          break;
      }
      if(retry) cout << "retry mutex initialization" <<endl;
      else assert(ierr == 0); //die if after retry we still get an error
    }while(retry);
}


PinnedInFlow::PinnedInFlow(Mask *maskPtr, int numFlows)
{ 
  mNumWells = maskPtr->W() *maskPtr->H();
  mNumFlows = numFlows;
  InitMutex();

}

PinnedInFlow::~PinnedInFlow()
{
  mPinnedInFlow.clear();
  mPinsPerFlow.clear();

  int ierr = pthread_mutex_destroy(&mutex_setPin);
  switch (ierr){
    case 0:
      //no error
      break;
    case EBUSY:
      cout << "The implementation has detected an attempt to destroy the object referenced by mutex while it is locked or referenced (for example, while being used in a pthread_cond_wait() or pthread_cond_timedwait()) by another thread." << endl;
      break;
    case EINVAL:
      cout << "The value specified by mutex is invalid." << endl;
      break;
    default:
      cout << " An error with unknown error code " << ierr << " occurred during mutex destruction" << endl;
      break;
  }

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

#if 1

	int ierr = pthread_mutex_lock (&mutex_setPin);
	switch(ierr){
	  case EINVAL:
	    cout << "The value specified by mutex does not refer to an initialised mutex object." << endl;
	    break;
	  case EAGAIN:
	    cout << "The mutex could not be acquired because the maximum number of recursive locks for mutex has been exceeded." << endl;
	    break;
	  default:
	    break;
	    //nop
	}
	assert(ierr == 0);
	//critical section perform test set on pinned mask
	currFlow = mPinnedInFlow[idx];
	if ((currFlow < 0) || (currFlow > flow))   // new pins per flow
	{
	  mPinnedInFlow[idx] = flow;
	  mPinsPerFlow[flow]++; // add pin to sum of pins in this flow
	  if(currFlow > 0) mPinsPerFlow[currFlow]--; //if already wrongly recorded for currFlow remove from currFlow sum
	}
	pthread_mutex_unlock (&mutex_setPin);

#else
	currFlow = mPinnedInFlow[idx];
	// pixel is pinned high or low
	if ((currFlow < 0) | (currFlow > flow))   // new pins per flow
	{
		currFlow = flow;
		mPinnedInFlow[idx] = flow;
		//NOT THREAD SAFE!  there can be a set in a different thread
		//after this thread already exited after the test,set,test cycle
		while (((short volatile *) &mPinnedInFlow[0])[idx] > currFlow)
		{
			// race condition, a later flow already updated this well, keep trying
			((short volatile *) &mPinnedInFlow[0])[idx] = currFlow;
		}
		mPinsPerFlow[flow]++; // this needs to be protected...
	}
#endif
}

#define MAX_GAIN_CORRECT 16383

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
  int frame;
//  int pinnedLowCount = 0;
//  int pinnedHighCount = 0;
  int idx;
  const uint32_t pinHigh = GetPinHigh();
  const uint32_t pinLow = GetPinLow();
//  uint32_t pinned;

//  double stopT,startT = TinyTimer();

  if (rows <= 0 || cols <= 0)
  {
    cout << "Why bad row/cols for flow: " << flow << " rows: " << rows << " cols: " << cols << endl;
    exit (EXIT_FAILURE);
  }

#ifdef __AVX__
	if ((cols % VEC8_SIZE) == 0) {
		int k, idx;
		int frameStride = rows * cols;
		short int *src;
#define MY_VEC_SIZE 8
#define MY_VECF v8f_u

		v8f_u highV, lowV;
		v8f_u dummy8 = { };
		MY_VECF tmpV;
		MY_VECF *gainPtrV;
		MY_VECF pinnedV;
		MY_VECF dummy = { };
		short int *sptr;

		highV.V = dummy8.V + (float) pinHigh;
		lowV.V = dummy8.V + (float) pinLow;

		src = (short int *) (raw->image);
		if (gainPtr) {
			gainPtrV = (MY_VECF *) gainPtr;
			for (idx = 0; idx < frameStride; idx += MY_VEC_SIZE, src +=
					MY_VEC_SIZE) {
				pinnedV.V = dummy.V + 0;

				for (frame = 0; frame < frames; frame++) {
					sptr = &src[frame * frameStride];

					LD_VS_VF(sptr, tmpV); // load values from sptr to tmpV
					// now, do the pinned pixel comparisons
					for (uint l = 0;
							l < sizeof(pinnedV.V8) / sizeof(pinnedV.V8[0]);
							l++) {
						pinnedV.V8[l] += __builtin_ia32_cmpps256(tmpV.V8[l],
								lowV.V, _CMP_LT_OS);
						pinnedV.V8[l] += __builtin_ia32_cmpps256(highV.V,
								tmpV.V8[l], _CMP_LT_OS);
					}

					// gain correct
					tmpV.V *= (*gainPtrV).V;

					v8s_u tmpS;
					CVT_VF_VS(tmpS, tmpV);
					((v8s_u *) sptr)->V = tmpS.V;
				}
				gainPtrV++;
				// if any of the 8 pixels are pinned
				int somePinned = 0;
				for (uint l = 0; l < sizeof(pinnedV.V8) / sizeof(pinnedV.V8[0]);
						l++) {
					if (__builtin_ia32_ptestnzc256((v4di) pinnedV.V8[l],
							(v4di ) { -1, -1, -1, -1 }))
						somePinned = 1;
				}

				if (somePinned) {
					for (k = 0; k < MY_VEC_SIZE; k++) {
						if (pinnedV.A[k])
							SetPinned(idx + k, flow);
					}
				}
			}
		} else {
			// no gain correction
			for (idx = 0; idx < frameStride; idx += MY_VEC_SIZE, src +=
					MY_VEC_SIZE) {
				pinnedV.V = dummy.V + 0;

				for (frame = 0; frame < frames; frame++) {
					sptr = &src[frame * frameStride];

					LD_VS_VF(sptr, tmpV);

					// now, do the pinned pixel comparisons
					for (uint l = 0;
							l < sizeof(pinnedV.V8) / sizeof(pinnedV.V8[0]);
							l++) {
						pinnedV.V8[l] += __builtin_ia32_cmpps256(tmpV.V8[l],
								lowV.V, _CMP_LT_OS);
						pinnedV.V8[l] += __builtin_ia32_cmpps256(highV.V,
								tmpV.V8[l], _CMP_LT_OS);
					}
				}
				// if any of the 8 pixels are pinned
				int somePinned = 0;
				for (uint l = 0; l < sizeof(pinnedV.V8) / sizeof(pinnedV.V8[0]);
						l++) {
					if (__builtin_ia32_ptestnzc256((v4di) pinnedV.V8[l],
							(v4di ) { -1, -1, -1, -1 }))
						somePinned = 1;
				}

				if (somePinned) {
					for (k = 0; k < MY_VEC_SIZE; k++) {
						if (pinnedV.A[k])
							SetPinned(idx + k, flow);
					}
				}
			}

		}
	}else
#endif
  {
	  int16_t *pixPtr;
	  int16_t *sPtr;
	  uint32_t val;
	  double fval;
	  bool isOutOfRange;
	  int frameStride=rows*cols;
	  float gainFactor;

	  // check for pinned pixels in this flow
	for (idx = 0; idx < frameStride; idx++) {
		pixPtr = raw->image + idx;
		if (gainPtr)
			gainFactor = gainPtr[idx];
		else
			gainFactor = 1.0f;
		isOutOfRange=0;
		for (frame = 0; frame < frames; frame++) {
			sPtr = &pixPtr[frame*frameStride];
			fval = (float) *sPtr;
			fval *= gainFactor;
			val = fval;
//				if (val > MAX_GAIN_CORRECT)
//					val = MAX_GAIN_CORRECT;

			*sPtr = (int16_t) val;

			if(val < pinLow || pinHigh < val)
				isOutOfRange =1;
		} // end frame loop
		if (isOutOfRange) {
			SetPinned(idx, flow);
		}
	} // end idx loop
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


int PinnedInFlow::QuickUpdate(int flow, Image *img)
{
	// if any well at (x,y) is first pinned in this flow & this flow's img,
	// set the value in mPinnedInFlow[x,y] to that flow
	const RawImage *raw = img->GetImage(); // the raw image
	int rows = raw->rows;
	int cols = raw->cols;


	if (rows <= 0 || cols <= 0)
	{
		cout << "Why bad row/cols for flow: " << flow << " rows: " << rows
				<< " cols: " << cols << endl;
		exit(EXIT_FAILURE);
	}
	int16_t *pixPtr=raw->image;
	int frameStride = rows * cols;

	// check for pinned pixels in this flow
	for (int idx = 0; idx < frameStride; idx++)
	{
		if (pixPtr[idx] == 0) // advCompr sets all values to zero if pinned..
		{
			SetPinned(idx, flow);
		}
	} // end idx loop
	return 0;
}

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
