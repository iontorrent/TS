/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef PARAMSTRUCTS_H
#define PARAMSTRUCTS_H

#include "BeadParams.h"
#include "RegionParams.h"
#include "CudaDefines.h"

struct ConstParams : public reg_params
{
  int start[NUMFB]; // 4
  float deltaFrames[MAX_COMPRESSED_FRAMES_GPU];
  float frameNumber[MAX_COMPRESSED_FRAMES_GPU];
  int flowIdxMap[NUMFB]; 
  // confining fitted bead params within a range
  bound_params beadParamsMaxConstraints;
  bound_params beadParamsMinConstraints;
  bool useDarkMatterPCA;
//  float darkMatterComp[MAX_COMPRESSED_FRAMES_GPU*4];
};

struct ConstXtalkParams 
{
  int neis;
  float multiplier[MAX_XTALK_NEIGHBOURS];
  float tau_top[MAX_XTALK_NEIGHBOURS];
  float tau_fluid[MAX_XTALK_NEIGHBOURS];  
};

#endif // PARAMSTRUCTS_H

