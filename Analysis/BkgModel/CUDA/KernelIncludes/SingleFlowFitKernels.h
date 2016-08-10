/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 * SingleFitKernel.h
 *
 *  Created on: Feb 5, 2014
 *      Author: jakob
 */

#ifndef SINGLEFLOWFITKERNEL_H_
#define SINGLEFLOWFITKERNEL_H_

#include "MathModel/PoissonCdf.h"
#include "Mask.h"
#include "EnumDefines.h"
#include "ImgRegParams.h"
#include "DeviceParamDefines.h"


//void copySymbolsToDevice( const ImgRegParams& irp, const ParamLimits& pl, const ConfigParams& cp);


////////////////////////////////////
//Wrapper Kernel

// execute with one thread-block per region row.
// if the thread block is too small to handle the
// whole row it will slide across the row of the region
// kernel parameters:
// thread block dimensions (n,1,1)  //n = number of threads per block)
// grid dimension ( numRegions.x, imgH, 1) // one block per region in x direction and one per img row in y direction
// const execParams ep, moved to constant memory as ExecP
// const ImgRegParams moved to constant memory as ImgRegP
//ToDo: still extracts values from __constant__ FlowParams FlowP; should not even
//touch this and leave access to this symbol to the function
__global__
void ExecuteThreadBlockPerRegion2DBlocks(
    const unsigned short * RegionMask,
    const unsigned short  * bfMask,
    unsigned short  * bstateMask,
    //per bead
    //in parameters
    const short * RawTraces, // NxF
    const float * BeadParamCube,
    const float* crudeemphasisVec, //(MAX_POISSON_TABLE_COL)*F
    const int * crudenonZeroEmphFrames,
    const float* fineemphasisVec, //(MAX_POISSON_TABLE_COL)*F
    const int * finenonZeroEmphFrames,
    const float* nucRise, // ISIG_SUB_STEPS_SINGLE_FLOW * F
        //in out parameters
    float* ResultCube,
    const size_t * numFramesRegion,  //constant memory?
    const int * numLBeadsRegion,  //constant memory?
    //per region
    const ConstantParamsRegion * constRegP,
    const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP,
    const float * RegionFrameCube,  //DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
    const float * EmptyTracesRegion

    //TODO remove only for debugging
    //int * numLBeads//,
    //float * fgBuffer
    );


__global__
void ExecuteThreadBlockPerRegion2DBlocksDense(
    const unsigned short * RegionMask,
    const unsigned short  * bfMask,
    unsigned short  * bstateMask,
    //per bead
    //in parameters
    const short * RawTraces, // NxF
    const float * BeadParamCube,
    const float* crudeemphasisVec, //(MAX_POISSON_TABLE_COL)*F
    const int * crudenonZeroEmphFrames,
    const float* fineemphasisVec, //(MAX_POISSON_TABLE_COL)*F
    const int * finenonZeroEmphFrames,
    const float* finenucRise, // ISIG_SUB_STEPS_SINGLE_FLOW * F
    const float* coarsenucRise, // ISIG_SUB_STEPS_SINGLE_FLOW * F
    //in out parameters
    float* ResultCube,
    const size_t * numFramesRegion,  //constant memory?
    const int * numLBeadsRegion,  //constant memory?
    //per region
    const ConstantParamsRegion * constRegP,
    const PerFlowParamsRegion * perFlowRegP,
    const PerNucParamsRegion * perNucRegP,
    const float * RegionFrameCube,  //DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
    //TraceLevelXTalk
    const float * XTalkPerBead,
    const float * genericXTalkRegion
);



/*
//as above but tries to align the thread blocks in each region to the segments boundaries
__global__
void ExecuteThreadBlockPerRegionRow_AlignExecution(  const unsigned short  * bfMask,
                                      unsigned short  * bstateMask,
                                      //per bead
                                      //in parameters
                                      const short * RawTraces, // NxF
                                      const float * BeadParamCube,
                                      const float * BeadStateCube,
                                      const float* emphasisVec, //(MAX_POISSON_TABLE_COL)*F
                                      const float* nucRise, // ISIG_SUB_STEPS_SINGLE_FLOW * F
                                      //in out parameters
                                      float* ResultCube,
                                      const size_t * numFrames,  //constant memory?
                                      const int * numLBeadsRegion,  //constant memory?
                                      //per region
                                      const ConstParams * regP,
                                      const float * RegionFrameCube  //DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
                                      //TODO remove only for debugging
                                      //int * numLBeads//,
                                      //float * fgBuffer
                                      );
*/







#endif // SINGLEFLOWFITKERNEL_H_
