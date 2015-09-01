/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
/*

 * GenerateBeadTraceKernels.h
 *
 *  Created on: Sep 9, 2013
 *      Author: Jakob Siegel
 */

#ifndef GENERATEBEADTRACEKERNELS_H
#define GENERATEBEADTRACEKERNELS_H

#include "Image.h"
#include "TimeCompression.h"
#include "Region.h"
#include "BeadTracker.h"
#include "BkgTrace.h"
#include "EnumDefines.h"
#include "ImgRegParams.h"
#include "DeviceParamDefines.h"


// one block per region (e.g. for P1: regionX =6, regionY=6. => 36 regions)
// block width has to be a warp size or a 2^k fraction of it
// need one shared memory value per thread to calculate sum
// kernel creates meta data itself:
//  number of life beads per region (and warp/thread block-row)
//  t0 average gets calculated on the fly
//  t0map not needed since t0map values directly calculated on the fly from t0est
// requires Symbol "RawImgRegP" in constant memory
__global__
void GenerateT0AvgAndNumLBeads(
    const unsigned short* bfmask,
    const float* t0Est,
    int * lBeadsRegion, //numLbeads of whole region
    float * T0avg // T0Avg per REgion //ToDo check if this is really needed of if updating the T0Est would be better
);

__global__
void GenerateT0AvgAndNumLBeads_New(
    unsigned short * RegionMask,
    const unsigned short* bfmask,
    const unsigned short* BeadStateMask,
    const float* t0Est,
    int * SampleRowPtr,
    int * NumSamples,
    int * lBeadsRegion, //numLbeads of whole region
    float * T0avg // T0Avg per REgion //ToDo check if this is really needed of if updating the T0Est would be better
);



__global__
void GenerateAllBeadTraceEmptyFromMeta_k (
    unsigned short * RegionMask,
    short * img,  //perwell    input and output
    const unsigned short * bfmask, //per well
    const float* t0Est, //per well
    const float * frameNumberRegion, // from timing compression
    const int * framesPerPointRegion,
    const size_t * nptsRegion,  //per region
    const int * numlBeadsRegion,
    const float * T0avg,  // ToDo: try already subtract T0 after calculating the average so this would not be needed here anymore!
    const ConstantParamsRegion * constRegP,
    const PerFlowParamsRegion * perFlowRegP,
    //one empty trace per thread block
    float * EmptyTraceAvgRegion, // contains result of emppty trace averaging per region
    float * EmptyTraceSumRegionTBlock, // has to be initialized to 0!! will contain sum of all empty trace frames for each row in a region
    int * EmptyTraceCountRegionTBlock, // has to be initialized to 0!! will contain number of empty traces summed up for each row in a region
    int * EmptyTraceComplete, //has to be initialized to 0!! completion counter per region for final sum ToDo: figure out if we can do without it
    //regional sample
        //inputs
        const float * BeadParamCube,
        const unsigned short * BeadStateMask,
        //meta data
        const int * SampleRowPtr,
        int * SampleRowCounter,
        //outputs
        unsigned short * SampleStateMask,
        short * SampleCompressedTraces,
        float * SampleParamCube,
        SampleCoordPair * SampleCoord
);

__global__
void ReduceEmptyAverage_k (
    unsigned short * RegionMask,
    float * EmptyTraceAvgRegion,
    const ConstantParamsRegion * constRegP,
    const PerFlowParamsRegion * perFlowRegP,
    const float * frameNumberRegion, // from timing compression
    const int * framesPerPointRegion,
    const size_t * nptsRegion,  //per region
    const float * EmptyTraceSumRegionTBlock, // has to be initialized to 0!! will contain sum of all empty trace frames for the row in a region
    const int * EmptyTraceCountRegionTBlock, // has to be initialized to 0!! will contain number of empty traces summed up for each warp in a region
    const size_t numTBlocksPerReg
    //float * dcoffsetdebug
);



#endif // GENERATEBEADTRACEKERNELS_H
