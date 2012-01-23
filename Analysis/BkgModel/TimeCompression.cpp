/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "TimeCompression.h"

TimeCompression::TimeCompression()
{
  frameNumber = NULL;
  deltaFrame = NULL;
  deltaFrameSeconds = NULL;
  frames_per_point = NULL;
  npts = 0;
}
 
void TimeCompression::DeAllocate()
{
  if (frameNumber != NULL) delete [] frameNumber;
  if (deltaFrame != NULL) delete [] deltaFrame;
  if (deltaFrameSeconds !=NULL) delete [] deltaFrameSeconds;
  if (frames_per_point != NULL) delete [] frames_per_point;
}

TimeCompression::~TimeCompression()
{
    DeAllocate();
}

// imgFrames, t_compression_start, -5, 16, 5
void TimeCompression::SetUpTime(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg)
{
  // just allocate worst-case arrays
  frameNumber = new float[imgFrames];
  deltaFrame = new float[imgFrames];
  deltaFrameSeconds = new float[imgFrames]; // save recalculation on this
  frames_per_point = new int[imgFrames];

  // starting point of non-averaged data
  int il = (int) t_comp_start+start_detailed_time;
  // stopping point of non-averaged data..intermediate average points
  int ir = (int)(t_comp_start+stop_detailed_time);

  if (il < 0) il = 0;
  if (ir > imgFrames) ir = imgFrames;

  int npt=0;
  float fnum_sum = 0.0;
  int fnum_num = 0;
  int irpts = 2;
  float last_fnum = 0.0;
  bool capture = false;

  for (int i=0;i < imgFrames;i++)
  {
    fnum_sum += i;
    fnum_num++;

    capture = false;

    if (i < il)
    {
      if ((fnum_num >= left_avg) || ((i+1) >= il))
        capture = true;
    }
    else if ((i >= il) && (i < ir))
    {
      capture = true;
    }
    else if (i >= ir)
    {
      if (fnum_num >= irpts)
      {
        capture = true;
        irpts += 4;
        if (irpts > 40) irpts = 40;
      }
    }

    if (i == (imgFrames - 1))
      capture = true;

    if (capture == true)
    {
      frameNumber[npt] = (fnum_sum / fnum_num);
      deltaFrame[npt] = frameNumber[npt] - last_fnum;
      deltaFrameSeconds[npt] = deltaFrame[npt]/FRAMESPERSECOND; // cache for hydrogen generation
      frames_per_point[npt] = fnum_num;
      last_fnum = frameNumber[npt];
      npt++;
      fnum_sum = 0.0;
      fnum_num = 0;
    }
  }

  npts = npt;
}