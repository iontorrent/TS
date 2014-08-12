/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef PARAMSTRUCTS_H
#define PARAMSTRUCTS_H

#include "BeadParams.h"
#include "RegionParams.h"
#include "CudaDefines.h"

struct ConstParams : public reg_params
{
  int start[MAX_NUM_FLOWS_IN_BLOCK_GPU];
  float deltaFrames[MAX_COMPRESSED_FRAMES_GPU];
  float frameNumber[MAX_COMPRESSED_FRAMES_GPU];

  float deltaFrames_std[MAX_COMPRESSED_FRAMES_GPU];
  int std_frames_per_point[MAX_COMPRESSED_FRAMES_GPU];
  int etf_interpolate_frame[MAX_UNCOMPRESSED_FRAMES_GPU];
  float etf_interpolateMul[MAX_UNCOMPRESSED_FRAMES_GPU];
  int non_zero_emphasis_frames[MAX_POISSON_TABLE_COL];
  int std_non_zero_emphasis_frames[MAX_POISSON_TABLE_COL];

  int flowIdxMap[MAX_NUM_FLOWS_IN_BLOCK_GPU]; 
  // confining fitted bead params within a range
  bound_params beadParamsMaxConstraints;
  bound_params beadParamsMinConstraints;
  bool useDarkMatterPCA;
  bool useRecompressTailRawTrace;
};

struct ConstXtalkParams 
{
  int neis;
  float multiplier[MAX_XTALK_NEIGHBOURS];
  float tau_top[MAX_XTALK_NEIGHBOURS];
  float tau_fluid[MAX_XTALK_NEIGHBOURS];  
};

#endif // PARAMSTRUCTS_H

