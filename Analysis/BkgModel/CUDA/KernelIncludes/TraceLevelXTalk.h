/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 * SingleFitKernel.h
 *
 *  Created on: Feb 5, 2014
 *      Author: jakob
 */

#ifndef TRACELEVELXTALK_H_
#define TRACELEVELXTALK_H_

#include <vector>

#include "EnumDefines.h"
#include "ImgRegParams.h"
#include "DeviceParamDefines.h"


////////////////////////////////////




__global__ void SimpleXTalkNeighbourContributionAndAccumulation_LocalMem(// Here FL stands for flows
    const unsigned short * RegionMask, //per Region
    const unsigned short  * bfMask, // per Bead
    const unsigned short  * bstateMask, //per Bead

    float * xTalkContribution,  // buffer XTalk contribution to this well NxF
    float * genericXTalkTracesRegion, // one trace of max compressed frames per thread block or per region (atomicAdd)
    int * numGenericXTalkTracesRegion, //one int per region to average after accumulation
    const short* RawTraces,  //NxF
    const float * EmptyTraceRegion, //FxR
    const float* BeadParamCube, //NxP
    const float* RegionFrameCube, //FxRxT bkgTrace, DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
    const ConstantParamsRegion * constRegP, // R
    const PerFlowParamsRegion * perFlowRegP, // R
    const PerNucParamsRegion * perNucRegP, //RxNuc
    const size_t * numFramesRegion, // R
    const bool * TestingGenericXtakSampleMask //ToDo: remove whne testing done
);


// same as above but split inot two kernels that are in theory way more efficient
__global__ void SimpleXTalkNeighbourContribution(// Here FL stands for flows
    const unsigned short * RegionMask, //per Region
    const unsigned short  * bfMask, // per Bead
    const unsigned short  * bstateMask, //per Bead

    float * myBaseXTalkContribution,  // buffer XTalk contribution of this well NxF

    const short* RawTraces,  //NxF
    const float* BeadParamCube, //NxP
    const float* RegionFrameCube, //FxRxT bkgTrace, DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
    const ConstantParamsRegion * constRegP, // R
    const PerFlowParamsRegion * perFlowRegP, // R
    const PerNucParamsRegion * perNucRegP, //RxNuc
    const size_t * numFramesRegion // R
);

__global__ void GenericXTalkAndNeighbourAccumulation(// Here FL stands for flows
    const unsigned short * RegionMask, //per Region
    const unsigned short  * bfMask, // per Bead
    const unsigned short  * bstateMask, //per Bead
    float * BaseXTalkContribution,  // XTalk of each single well
    float * xTalkContribution,  // buffer XTalk to store accumulated xtalk at each well
    float * genericXTalkTracesperBlock, // one trace of max compressed frames per thread block
    int * numGenericXTalkTracesRegion, //one int per region to average after accumulation
    const PerFlowParamsRegion * perFlowRegP, // R
    const size_t * numFramesRegion, // R
    const bool * TestingGenericXtakSampleMask //ToDo: remove whne testing done
);

//one thread block per region
__global__ void GenericXTalkAccumulation(// Here FL stands for flows
    float * genericXTalkTracesRegion, // one trace of max compressed frames per thread block or per region (atomicAdd)
    const float * genericXTalkTracesPerBlock,
    const int * numGenericXTalkTracesRegion, //one int per region to average after accumulation
    const size_t * numFrames,
    const int blocksPerRegion
    );


//////////////////////////////////





#endif // TRACELEVELXTALK_H_
