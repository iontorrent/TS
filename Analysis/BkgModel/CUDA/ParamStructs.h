/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef PARAMSTRUCTS_H
#define PARAMSTRUCTS_H

#include "BeadParams.h"
#include "RegionParams.h"

struct ConstParams : public reg_params
{
  //reg_params reg;
  int start[NUMFB]; // 4
  //float krate; // 4
  //float d; // 4
  //float kmax; // 4
  float deltaFrames[MAX_COMPRESSED_FRAMES];
//  float nucRise[NUMFB*ISIG_SUB_STEPS_SINGLE_FLOW*MAX_COMPRESSED_FRAMES];  
  int flowIdxMap[NUMFB]; 
  // confining fitted bead params within a range
  bound_params beadParamsMaxConstraints;
  bound_params beadParamsMinConstraints;
};


#endif // PARAMSTRUCTS_H

