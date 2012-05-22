/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iostream>
#include <assert.h>
#include "PinnedInFlow.h"
#include "Utils.h"
#include "IonErr.h"

using namespace std;

PinnedInFlow::PinnedInFlow(Mask *maskPtr, int numFlows)
{
  int pinnedInFlowLength = maskPtr->W() *maskPtr->H();
  mPinnedInFlow = new short[pinnedInFlowLength];
  mPinsPerFlow = new int [numFlows];
  mNumWells = pinnedInFlowLength;
  mNumFlows = numFlows;
}

PinnedInFlow::~PinnedInFlow()
{
  if (mPinnedInFlow!=NULL) delete[] mPinnedInFlow;
  if (mPinsPerFlow!=NULL) delete[] mPinsPerFlow;
}


void PinnedInFlow::Initialize (Mask *maskPtr)
{
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


int PinnedInFlow::Update (int flow, Image *img)
{
  // if any well at (x,y) is first pinned in this flow & this flow's img,
  // set the value in mPinnedInFlow[x,y] to that flow
  const RawImage *raw = img->GetImage();
  int rows = raw->rows;
  int cols = raw->cols;
  int x, y, frame;
  int pinnedLowCount = 0;
  int pinnedHighCount = 0;
  int i = 0;
  const short pinHigh = GetPinHigh();
  const short pinLow = GetPinLow();

  if (rows <= 0 || cols <= 0)
  {
    cout << "Why bad row/cols for flow: " << flow << " rows: " << rows << " cols: " << cols << endl;
    exit (EXIT_FAILURE);
  }

  for (y=0;y<rows;y++)
  {
    for (x=0;x<cols;x++)
    {
      short currFlow = mPinnedInFlow[i];
      // if (currFlow < 0) | (currFlow > flow) ){ // new pins per flow
      if (true)    // report all pins, not just new, more useful though slower
      {
        // check for pinned pixels in this flow
        for (frame=0;frame<raw->frames;frame++)
        {
          short val = raw->image[frame*raw->frameStride + i];
          bool isLow = val <= pinLow;
          bool isHigh = val >= pinHigh;
          if (isLow || isHigh)
          {
            // pixel is pinned high or low
            if ( (currFlow < 0) | (currFlow > flow))   // new pins per flow
            {
              currFlow = flow;
              mPinnedInFlow[i] = flow;
            }
            if (isLow)
              pinnedLowCount++;
            else
              pinnedHighCount++;
            break;
          }
        } // end frame loop
      } // end if
      while ( ( (short volatile *) mPinnedInFlow) [i] > currFlow)
      {
        // race condition, a later flow already updated this well, keep trying
        ( (short volatile *) mPinnedInFlow) [i] = currFlow;
      }
      i++;
    }  // end x loop
  } // end y loop

  // char s[512];
  // int n = sprintf(s,  "PinnedInFlow::UpdatePinnedWells: %d pinned pixels <=  %d or >= %d in flow %d (%d low, %d high)\n", (pinnedLowCount+pinnedHighCount), pinLow, pinHigh, flow, pinnedLowCount, pinnedHighCount);
  // assert(n<511);
  // fprintf (stdout,"%s", s);
  mPinsPerFlow[flow] = pinnedLowCount+pinnedHighCount;
  return (pinnedLowCount+pinnedHighCount);
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
