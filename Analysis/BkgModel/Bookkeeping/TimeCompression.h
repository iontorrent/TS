/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TIMECOMPRESSION_H
#define TIMECOMPRESSION_H

#include <stdio.h>
#include <vector>
#include <iostream>
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
  float frames_per_second;
  int     *frames_per_point;      // helper table used to construct average of incomming data
  int     npts;           // number of data points after time compression
  
  float time_start; // when real points exist in the data we take
  int choose_time; // switch between different time compression schema

  // list of points and their weight for each vfc compressed point to convert into bg time
  std::vector<std::vector<std::pair<float,int> > > mVfcAverage; 
  // Keep the sum around so don't have to recalculate every time.
  std::vector<float> mVfcTotalWeight; 
  
  TimeCompression();
  ~TimeCompression();
  void Allocate(int imgFrames);
  void CompressionFromFramesPerPoint();
  void DeAllocate();
  inline int Coverage(int s1, int e1, int s2, int e2) { 
    return  std::max(0, std::min(e1,e2) - std::max(s1,s2));
  }
  void SetUpTime(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg); // interface
  void SetUpOldTime(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg);
  void SetUpStandardTime(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg);
  void StandardFramesPerPoint(int imgFrames,float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg);
  void ExponentialFramesPerPoint(int imgFrames, float t_comp_start, int start_detailed_time, float geom_ratio);
  void HyperTime(int imgFrames, float t_comp_start, int start_detailed_time);
  void StandardAgain(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg);
  void HalfSpeedSampling(int imgFrames, float t_comp_start, int start_detailed_time, int stop_detailed_time, int left_avg);
  void SetupConvertVfcTimeSegments(int frames, int *timestamps, int baseFrameRate);
  void ConvertVfcSegments(size_t rowStart, size_t rowEnd, size_t colStart, size_t colEnd, 
                          size_t nRow, size_t nCol, size_t nFrame, short *source, float *output);
  void ReportVfcConversion(int frames, int *timestamps, int baseFrameRate, std::ostream &out);
};


#endif // TIMECOMPRESSION_H
