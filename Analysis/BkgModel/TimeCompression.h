/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TIMECOMPRESSION_H
#define TIMECOMPRESSION_H

#include <stdio.h>
#include <vector>
#include <math.h>
#include "BkgMagicDefines.h"

// handle the time compression for bkgmodel

class TimeCompression
{
  public:

  // time compression information
  float   *frameNumber;   // for each averaged data point, the mean frame number
  float   *deltaFrame;    // the delta of each data point from the last
  float   *deltaFrameSeconds; // in seconds
  int     *frames_per_point;      // helper table used to construct average of incomming data
  int     npts;           // number of data points after time compression
  
  float time_start; // when real points exist in the data we take
 
  
  TimeCompression();
  ~TimeCompression();
  void Allocate(int imgFrames);
  void CompressionFromFramesPerPoint();
  void DeAllocate();
  void SetUpTime(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg); // interface
  void SetUpOldTime(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg);
  void SetUpStandardTime(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg);
  void StandardFramesPerPoint(int imgFrames,float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg);
  void ExponentialFramesPerPoint(int imgFrames, float t_comp_start, int start_detailed_time, float geom_ratio);
  void HyperTime(int imgFrames, float t_comp_start, int start_detailed_time);
  void StandardAgain(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg);
};


#endif // TIMECOMPRESSION_H