/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
/*

 * ImageProcessingKernels.h
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





// one block per region (regionX =6, regionY=6. => 36 regions)
// block width has to be a warp size or a fraction of it
// need one shared memory value per thread to calculate sum
// kernel creates meta data itself:
//  number of life beads per region (and warp/thread block-row)
//  t0 average gets calculated on the fly
//  t0map not needed since t0map values directly calculated on the fly from t0est
//
// LIMITATIONS: code only works for regions of width less than 32^2 (should hopefully never be an issue)
//
__global__
void GenerateMetaPerWarpForBlock_k (
    const unsigned short* bfmask,
    const float* t0Est,
    int imgWidth,
    int imgHeight,
    int regMaxWidth,
    int regMaxHeight,
    int * offsetWarp, //num beads per row, length == regHeight * (ceil(regMaxWidth/blockDim.x)+1)
    int * lBeadsRegion, //numLbeads of whole region
    float * T0avg
);


// grid size is grid.x = regionsX = 6;
//        grid.y = imgHeight = 1288
// blockDim.x has to be the same as used for GenerateMetaData Kernel!!!!!
// block size block.x = warpsize = 32 (or maybe 16, needs to be tested)
//         block.y = TBD = probably 8 or so,

// one block works on on row of a region, where the warps slide
// across the row with a stride of blockdim.x*blockdim.y
__global__
void GenerateAllBeadTraceFromMeta_k (
    FG_BUFFER_TYPE ** fgBufferRegion,
    int* framesPerPointRegion[],
    const int * nptsRegion,
    RawImage raw,
    const unsigned short * bfmask,
    const float* t0Est,
    int regMaxW,
    int regMaxH,
    //from meta data kernel:
    const int * WarpOffsets, //num beads per row, length == regHeight * (ceil(regMaxW/blockDim.x)+1)
    const int * lBeadsRegion, //numLbeads of whole region
    const float * T0avg
);



//same as above but grid size defines how many regions are handled.
//e.g. grid.x = 6 grid.y = 224 with regStartX = 0 and regStartY = 2
//will handle the third row of regions.

__global__
void GenerateAllBeadTraceFromMeta_N_Regions_k (
    FG_BUFFER_TYPE* fgBufferRegion[], // buffers for N regions
    int* framesPerPointRegion[],
    const int * nptsRegion,
    RawImage raw,
    const unsigned short * bfmask,
    const float* t0Est,
    int regMaxW,
    int regMaxH,
    //from meta data kernel:
    const int * WarpOffsets, //num beads per row, length == regHeight * (ceil(regMaxW/blockDim.x)+1)
    const int * lBeadsRegion, //numLbeads of whole region
    const float * T0avg,
    const int regStartX, // start at region x //for whole block  0
    const int regStartY // start at region y //for whole block  0  (how many regions are worked on is defined by grid size
);


template<typename TDest, typename TSrc>
__global__
void transposeDataKernel(TDest *dest, TSrc *source, int width, int height);


#endif // GENERATEBEADTRACEKERNELS_H
