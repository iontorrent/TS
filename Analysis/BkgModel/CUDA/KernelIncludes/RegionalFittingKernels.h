/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
 * 
 *
 *  Created on: Aug 20, 2014
 *  Author: Mohit
 */

#ifndef REGIONALFITTINGKERNEL_H_
#define REGIONALFITTINGKERNEL_H_

#include "MathModel/PoissonCdf.h"
#include "Mask.h"
#include "EnumDefines.h"
#include "ImgRegParams.h"
#include "DeviceParamDefines.h"

__global__ 
void PerformMultiFlowRegionalFitting(
  const unsigned short * RegionMask,
  const short *observedTrace, // REGIONS X NUM_SAMPLES_RF x F
  const float *beadParamCube,
  const unsigned short *beadStateCube,
  const float *emphasisVec, //(MAX_POISSON_TABLE_COL)*F
  const int *nonZeroEmphFrames, 
  float *nucRise,
  float *scratchSpace,
  const size_t *numFramesRegion,
  const ConstantParamsRegion * constRegP,
  PerFlowParamsRegion * perFlowRegP,
  const PerNucParamsRegion * perNucRegP,
  const float * RegionFrameCube,
  const float * EmptyTracesRegion,
  const int *NumSamples,
  const size_t numFlows
);





#endif // REGIONALFITTINGKERNEL_H_
