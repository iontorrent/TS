/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef PARAMSTRUCTS_H
#define PARAMSTRUCTS_H

#include "BeadParams.h"
#include "RegionParams.h"
#include "CudaDefines.h"

struct ConstParams : public reg_params
{
  int coarse_nuc_start[MAX_NUM_FLOWS_IN_BLOCK_GPU];
  int fine_nuc_start[MAX_NUM_FLOWS_IN_BLOCK_GPU];
  float deltaFrames[MAX_COMPRESSED_FRAMES_GPU];
  float frameNumber[MAX_COMPRESSED_FRAMES_GPU];
  int non_zero_crude_emphasis_frames[MAX_POISSON_TABLE_COL];
  int non_zero_fine_emphasis_frames[MAX_POISSON_TABLE_COL];
  int flowIdxMap[MAX_NUM_FLOWS_IN_BLOCK_GPU]; 
  // confining fitted bead params within a range
  bound_params beadParamsMaxConstraints;
  bound_params beadParamsMinConstraints;
  bool useDarkMatterPCA;
};

struct ConstXtalkParams 
{
  bool simpleXtalk;
  int neis;
  float multiplier[MAX_XTALK_NEIGHBOURS];
  float tau_top[MAX_XTALK_NEIGHBOURS];
  float tau_fluid[MAX_XTALK_NEIGHBOURS];  
};

#endif // PARAMSTRUCTS_H

