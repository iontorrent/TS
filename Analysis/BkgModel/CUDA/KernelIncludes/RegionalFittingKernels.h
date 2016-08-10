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
  const float *beadParamCube,
  const unsigned short *beadStateCube,
  const float *crudeemphasisVec, //(MAX_POISSON_TABLE_COL)*F
  const int *crudenonZeroEmphFrames, 
  float *finenucRise,
  float *coarsenucRise,
  float *scratchSpace,
  const size_t *numFramesRegion,
  const ConstantParamsRegion * constRegP,
  PerFlowParamsRegion * perFlowRegP,
  const PerNucParamsRegion * perNucRegP,
  const float * RegionFrameCube,
  const int *NumSamples
);



#endif // REGIONALFITTINGKERNEL_H_
