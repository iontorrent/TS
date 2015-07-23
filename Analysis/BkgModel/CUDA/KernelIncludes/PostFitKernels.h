/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 * XTalkKernels.h
 *
 *  Created on: Oct 1, 2014
 *      Author: jakob
 */

#ifndef XTALKKERNELS_H_
#define XTALKKERNELS_H_


#include "MathModel/PoissonCdf.h"
#include "Mask.h"
#include "EnumDefines.h"
#include "ImgRegParams.h"
#include "DeviceParamDefines.h"



//one threadblock per region
//threadblock width has to be warp size
//threadblock height [1-8] (default 4)
//grid width = #regions per row
//grid height = #regions per col
__global__
void UpdateSignalMap_k(
    const unsigned short * RegionMask,
    const unsigned short  * bfMask,
    const float* BeadParamCube,
    float* ResultCube,
    float * AverageSignalRegion  // has to be inited to 0.0f for all regions
  //  float * beadAvg     // has to be inited to 0.0f for all regions
);


//one warp per image row
//threadblock width has to be warp size
//threadblock height [1-8] (default 4)
//grid width = #regions per row
//grid height = #image rows/(threadblock height)
__global__
void PostProcessingCorrections_k(
    const unsigned short * RegionMask,
    const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP,
    const unsigned short  * bfMask,
    const float* BeadParamCube,
    unsigned short * BeadStateMask,
    float* PolyClonalCube,
    float* ResultCube,
    float * regionAvgSignal  // has to be inited to 0.0f for all regions
   );




#endif //XTALKKERNELS_H_
