/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "TimeCompression.h"

TimeCompression::TimeCompression()
{
  frameNumber = NULL;
  deltaFrame = NULL;
  deltaFrameSeconds = NULL;
  frames_per_point = NULL;
  npts = 0;
  time_start = 0.0-0.1;  // every data point real by default
}
 
void TimeCompression::DeAllocate()
{
  if (frameNumber != NULL) delete [] frameNumber;
  if (deltaFrame != NULL) delete [] deltaFrame;
  if (deltaFrameSeconds !=NULL) delete [] deltaFrameSeconds;
  if (frames_per_point != NULL) delete [] frames_per_point;
  frameNumber = NULL;
  deltaFrame = NULL;
  deltaFrameSeconds = NULL;
  frames_per_point = NULL;
}

TimeCompression::~TimeCompression()
{
    DeAllocate();
}

void TimeCompression::Allocate(int imgFrames)
{
  // just allocate worst-case arrays
  frameNumber = new float[imgFrames];
  deltaFrame = new float[imgFrames];
  deltaFrameSeconds = new float[imgFrames]; // save recalculation on this
  frames_per_point = new int[imgFrames];
}

// placeholder:  do time compression correctly
void TimeCompression::SetUpTime(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg)
{
  if (false)
    SetUpOldTime(imgFrames, t_comp_start, start_detailed_time, stop_detailed_time, left_avg);
  else
    SetUpStandardTime(imgFrames, t_comp_start, start_detailed_time, stop_detailed_time, left_avg);
}

// imgFrames, t_compression_start, -5, 16, 5
void TimeCompression::SetUpOldTime(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg)
{
  DeAllocate(); // just in case!
  Allocate(imgFrames);
  
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

// more flexible: define frames per point
// then define compression by applying frames per point.
void TimeCompression::SetUpStandardTime(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg)
{
  DeAllocate();
  Allocate(imgFrames);
  //StandardFramesPerPoint(imgFrames,t_comp_start, start_detailed_time,stop_detailed_time, left_avg);
  StandardAgain(imgFrames,t_comp_start, start_detailed_time,stop_detailed_time, left_avg);
  CompressionFromFramesPerPoint();
}

int CompressOneStep(int &cur_sum, int step)
{
  int tstep = step;
  cur_sum -= tstep;
  if (cur_sum<0)
    tstep = step+cur_sum;
  return(tstep);
}

void TimeCompression::StandardAgain(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg)
{
  int i_start = (int) t_comp_start+start_detailed_time;
  int i_done = (int)(t_comp_start+stop_detailed_time);
  // go to i_start compressing aggressively
  int cur_pt =0;
  int cur_sum=i_start;
  int i=0;
  for (; (i<imgFrames) & (cur_sum>0); i++)
  {
    frames_per_point[cur_pt] = CompressOneStep(cur_sum,left_avg);
    cur_pt++;
  }
  // now do the middle time when we are at full detail
  cur_sum = i_done-i_start;
  for (; (i<imgFrames) & (cur_sum>0); i++)
  {
    frames_per_point[cur_pt] = CompressOneStep(cur_sum,1);
    cur_pt++;
  }
  // finally compress the tail very heavily indeed
  int try_step = 2;
  cur_sum = imgFrames-i_done;
 for (; (i<imgFrames) & (cur_sum>0); i++)
  {
    frames_per_point[cur_pt] = CompressOneStep(cur_sum,try_step);
    cur_pt++;
    try_step += 4;
    //try_step *=2;
  }
  npts = cur_pt;
  
}

void TimeCompression::ExponentialFramesPerPoint(int imgFrames, float t_comp_start, int start_detailed_time, float geom_ratio)
{
  // super aggressive compression
  int cur_pt =0;
  int cur_sum=imgFrames;
  int i_start = (int) t_comp_start+start_detailed_time;
  if (i_start<1) i_start = 1;
  frames_per_point[cur_pt] = i_start;
  cur_sum -= frames_per_point[cur_pt];
  cur_pt++;
  float now_level = 1.0;
  while (cur_sum>0){
    frames_per_point[cur_pt] = CompressOneStep(cur_sum,(int) now_level);
    cur_pt++;
    now_level *= geom_ratio;
  }
  npts = cur_pt;
}

void TimeCompression::HyperTime(int imgFrames, float t_comp_start, int start_detailed_time)
{
  // super aggressive compression
  int cur_pt =0;
  int cur_sum=imgFrames;
  int i_start = (int) t_comp_start+start_detailed_time;
  if (i_start<1) i_start = 1;
  frames_per_point[cur_pt] = i_start;
  cur_sum -= frames_per_point[cur_pt];
  cur_pt++;
  for (int i=0; (i<3) & (cur_sum>0); i++)
  {
    frames_per_point[cur_pt] = CompressOneStep(cur_sum,1);
    cur_pt++;
    frames_per_point[cur_pt] = CompressOneStep(cur_sum,2);
    cur_pt++;
  }
  for (int i=0; (i<3) & (cur_sum>0); i++)
  {
    frames_per_point[cur_pt] = CompressOneStep(cur_sum,2);
    cur_pt++;
    frames_per_point[cur_pt] = CompressOneStep(cur_sum,3);
    cur_pt++;
  }
  npts = cur_pt;
}

void TimeCompression::StandardFramesPerPoint(int imgFrames,float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg)
{
  // starting point of non-averaged data
  int il = (int) t_comp_start+start_detailed_time;
  // stopping point of non-averaged data..intermediate average points
  int ir = (int)(t_comp_start+stop_detailed_time);

  if (il < 0) il = 0;
  if (ir > imgFrames) ir = imgFrames;

  int npt=0;
  int fnum_num = 0;
  int irpts = 2;

  for (int i=0;i < imgFrames;i++)
  {
    fnum_num++;
    bool capture = false;

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
      frames_per_point[npt] = fnum_num;
      npt++;
      fnum_num = 0;
    }
  }

  npts = npt;
}

void TimeCompression::CompressionFromFramesPerPoint()
{
  // frames_per_point contains the time compression information
  // sum (frames_per_point) = imgFrames
  // npts already defined
  // allocation already assumed
  // apply time compression to 0:(imgFrames-1)
          int npt = 0;
        // do not shift real bead wells at all
        int cur_frame=0;
        float  last_fnum = 0.0;
        for (;npt < npts;npt++)   // real data
        {
            float avg;
            avg=0.0;
            for (int i=0;i<frames_per_point[npt];i++)
            {
                avg += i+cur_frame;
            }
            frameNumber[npt] = (avg / frames_per_point[npt]);
            deltaFrame[npt] = frameNumber[npt] - last_fnum;
            deltaFrameSeconds[npt] = deltaFrame[npt]/FRAMESPERSECOND; // cache for hydrogen generation

            last_fnum = frameNumber[npt];            
            cur_frame+=frames_per_point[npt];
        }
}
