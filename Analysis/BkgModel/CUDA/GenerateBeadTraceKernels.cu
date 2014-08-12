/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "GenerateBeadTraceKernels.h"
//#include "cuda_runtime.h"
#include "cuda_error.h"
#include "dumper.h"
#include "Mask.h"
#include "Image.h"
#include "TimeCompression.h"
#include "Region.h"
#include "BeadTracker.h"
#include "BkgTrace.h"

using namespace std;



//mask already points to value
__device__ inline
bool Match(const unsigned short * mask,  MaskType type)
{
  return ( ( *mask & type ? true : false ) );
}


// Accumulates sm values of blockDim.x and stores them in sm value of thread 0;
// does not distribute results in sm, hence
// all values in sm for threads with Idx.x > 0 are garbage afterwards
// only works if blockDim.x <= warpsize since no sync
template<typename T>
__device__ inline
void WarpSumNoSync(T * sm)
{
  int offset = blockDim.x >> 1;
  while(offset >= 1){
    if(threadIdx.x < offset) *sm += *(sm + offset);
    offset = offset >> 1;
  }
}

template<typename T>
__device__ inline
void WarpSumNoSync(T * sm, int n)
{
  int tid = threadIdx.x % n;
  int offset = n >> 1;
  while(offset >= 1){
    if(tid < offset) *sm += *(sm + offset);
    offset = offset >> 1;
  }
  *sm = *(sm-tid);
}


//sums up shared memory from sm[0] to sm[threadIdx.x]
template<typename T>
__device__ inline
T RunningSumToCurrentThread(T * sm)
{
  T s = 0;
  for(int i = 0 ; i<threadIdx.x; i++)
    s += sm[i];
  return s;
}

//Uncompresses the VFC image, shifts it by t0 and re-compresses according to the
//the frames per point passed in compFrms
//works on one single well in the raw.image at position l_coord
//all other passed pointers and values are specific to the well

__device__ inline
void LoadImgWOffset(FG_BUFFER_TYPE *fgptr,
    const int * compFrms,
    int nfrms,
    int numLBeads,
    float t0Shift,
    RawImage raw,  //raw image object and data pointers
    int l_coord // well offset in raw image
)
{
  int t0ShiftWhole;
  float multT;
  float t0ShiftFrac;
  int my_frame = 0,compFrm,curFrms,curCompFrms;

  float prev;
  float next;
  float tmpAdder;

  int interf,lastInterf=-1;
  FG_BUFFER_TYPE lastVal;
  int f_coord;

  if(t0Shift < 0 - (raw.uncompFrames-2))
    t0Shift = 0 - (raw.uncompFrames-2);
  if(t0Shift > (raw.uncompFrames-2))
    t0Shift = (raw.uncompFrames-2);

  t0ShiftWhole=floor(t0Shift);
  t0ShiftFrac = t0Shift - (float)t0ShiftWhole;

  int startFrame = (t0ShiftWhole < 0)?(0):(t0ShiftWhole);
  my_frame = raw.interpolatedFrames[startFrame]-1;
  compFrm = 0;
  tmpAdder=0.0f;
  curFrms=0;
  curCompFrms=compFrms[compFrm];


  while ((my_frame < raw.uncompFrames) && (compFrm < nfrms))
  {
    interf= raw.interpolatedFrames[my_frame];

    if(interf != lastInterf)
    {
      f_coord = l_coord+raw.frameStride*interf;
      next = raw.image[f_coord];
      if(interf > 0)
      {
        prev = raw.image[f_coord - raw.frameStride];
      }
      else
      {
        prev = next;
      }
    }

    // interpolate
    multT= raw.interpolatedMult[my_frame] - (t0ShiftFrac/raw.interpolatedDiv[my_frame]);
    tmpAdder += ( (prev)-(next) ) * (multT) + (next);

    if(++curFrms >= curCompFrms)
    {
      tmpAdder /= curCompFrms;
      lastVal = (FG_BUFFER_TYPE)(tmpAdder); //Mayb use rintf or round to get more precision
      fgptr[numLBeads*compFrm] = lastVal;
      compFrm++;
      curCompFrms = compFrms[compFrm];
      curFrms=0;
      tmpAdder= 0.0f;
    }

    if(t0ShiftWhole < 0)
      t0ShiftWhole++;
    else
      my_frame++;

  }
  if(compFrm > 0 && compFrm < nfrms)
  {
    //lastVal = fgptr[numLBeads*(compFrm-1)];  //TODO: keep last val in reg
    for(;compFrm < nfrms;compFrm++)
      fgptr[numLBeads*compFrm] =  lastVal;
  }
}




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
    //  Image *img,
    //  FG_BUFFER_TYPE *fg_buffers,  //int iFlowBuffer not needed since on device on one flow of fg_buffer present
    //  int numFrames,
    int imgWidth,
    int imgHeight,
    int regMaxWidth,
    int regMaxHeight,
    int * offsetWarp, //num beads per row, length == regHeight * (ceil(regMaxWidth/blockDim.x)+1)
    int * lBeadsRegion, //numLbeads of whole region

    float * T0avg
)
{

  //if(blockDim.x != warpsize) return; // block width needs to be warp size
  extern __shared__ int smemGenerateMeta[]; //one element per thread;


  int * sm_base = smemGenerateMeta;
  int * sm = smemGenerateMeta;

  //helper
  int offsetsPerRow = ((regMaxWidth + blockDim.x-1)/blockDim.x) + 1; //ceil(regMaxWidth/blockDim.x) + 1


  //region id and base offset
  int regId = blockIdx.y * gridDim.x + blockIdx.x;  // block x & y are a grid of regions
  int RegionOffset = blockIdx.y*(regMaxHeight*imgWidth) + blockIdx.x*regMaxWidth;
  bfmask += RegionOffset; //go to beginning of region in mask
  t0Est += RegionOffset;
  lBeadsRegion += regId; // point to absolute num l beads for region
  T0avg += regId;
  offsetWarp += regId*regMaxHeight*offsetsPerRow; //point at beginning of region in warp meta data

  //warp starting offset
  int workOnRow = threadIdx.y; //each row in threadblock (warp) starts working on different row
  int * offsetWarpLocal = offsetWarp + workOnRow*offsetsPerRow; //go to beginning of specific row in meta data
  bfmask += imgWidth*threadIdx.y; //go to beginning of row to work on in mask
  t0Est+=imgWidth*threadIdx.y;
  sm += blockDim.x * threadIdx.y; // go to beginning of threadblock row in SM


  //thread offsets within warp
  bfmask += threadIdx.x; // add thread offset to mask row base pointer
  t0Est+=threadIdx.x;
  sm += threadIdx.x; // add thread offset to SM row base pointer

  //determine dimensions and handle border regions
  int regWidth =  (imgWidth-(blockIdx.x*regMaxWidth)< regMaxWidth )?(imgWidth%regMaxWidth):(regMaxWidth);
  int regHeight = (imgHeight-(blockIdx.y*regMaxHeight) < regMaxHeight)?(imgHeight%regMaxHeight):(regMaxHeight);

  // if(threadIdx.x == 0 && threadIdx.y == 0) printf ("RegId: %d dim: %dx%d, offsetsPerRow: %d\n", regId, regWidth, regHeight, offsetsPerRow);
  int LbeadsWarp = 0;
  float t0Sum = 0.0f;
  int t0Cnt = 0;
  // iterate over rows of region
  while(workOnRow < regHeight){
    int wellOffset = threadIdx.x;
    int windowStart = 0;
    const unsigned short* bfmaskRow = bfmask;
    const float* t0EstRow = t0Est;
    //int lBeadsRow = 0;
    int localoffset = 0;
    if(threadIdx.x == 0) *offsetWarpLocal = localoffset;
    offsetWarpLocal++;
    //slide warp/window across row and create sum for of num live beads for each warp
    while(windowStart < regWidth){
      if(wellOffset < regWidth){
        *sm = Match(bfmaskRow,(MaskType)MaskLive)?1:0;
        if(!Match(bfmaskRow,(MaskType) (MaskPinned | MaskIgnore | MaskExclude)))
        {
          t0Sum += *t0EstRow;
          t0Cnt ++;
        }
      }
      else
        *sm = 0;

      // sum up num live beads for current window, no sync needed since window is one warp
      WarpSumNoSync(sm);

      //if(threadIdx.x == 0 && threadIdx.y == 1) printf("row: %d, warpstart: %d, sm: %d\n", workOnRow, windowStart, *sm );
      // add to offset live beads of row (only correct for thread Idx.x == 0)
      localoffset += *sm;

      wellOffset += blockDim.x; //shift window by block width
      windowStart += blockDim.x;
      bfmaskRow += blockDim.x;
      t0EstRow += blockDim.x;
      if(threadIdx.x == 0) *offsetWarpLocal = localoffset;

      offsetWarpLocal ++; // go to next address for next warp in this row
    } //row done
    // thread 0 stores sum of live beads for this row
    LbeadsWarp += localoffset; // add up all live beads this warp sees per Row (only correct for threadIdx.x == 0)
    //update local pointers to nex row to work on (stride is blockDim.y)
    workOnRow += blockDim.y;
    bfmask += blockDim.y*imgWidth;
    t0Est += blockDim.y*imgWidth;
    offsetWarpLocal = offsetWarp + workOnRow*offsetsPerRow; // go to beginning of storage for next row meta data
  }//region done

  *sm = LbeadsWarp; // put all live beads the warp has seen into SM (only correct for threads with Idx.x == 0

  //Synchronize to allow to sum up partial sums from all warps
  __syncthreads();

  int numlBeads = 0;
  //calculate num live beads for whole region in all threads
  for(int i=0; i<blockDim.y; i++){
    numlBeads+= sm_base[i*blockDim.x]; //all threads get correct sum from SM base pointer
  }

  //if no live beads in region -> die
  if(numlBeads == 0) return;
  if(threadIdx.x == 0 && threadIdx.y == 0) *lBeadsRegion = numlBeads; //TODO: remove if not needed outside anymore
  //Synchronize to guarantee SM is not needed anymore and can be overwritten
  __syncthreads();

  //TODO: see if numLBeads is even needed in global mem
  // if(threadIdx.x == 0) *lBeadsRegion = numlBeads;

  //calculate t0 Sum for region
  float* smf = (float*)sm;
  *smf = t0Sum; //put t0 partial sums into shared
  //reduce partial sums inside each warp to one, sum only correct for threads with Idx.x == 0 in each warp
  WarpSumNoSync(smf);

  // synchronize to guarantee partial sums are all completed and global value can be accumulated
  __syncthreads();

  float * smf_base = (float*)sm_base;
  t0Sum = 0.0f;
  //calculate t0 sum for whole region
  for(int i=0; i<blockDim.y; i++){
    t0Sum += smf_base[i*blockDim.x];
  }
  //each thread now has correct t0Sum value;

  __syncthreads();
  //calculate t0 cnt for region
  *sm = t0Cnt; //put t0 partial sums into shared
  //reduce partial sums inside each warp to one, sum only correct for threads with Idx.x == 0 in each warp
  WarpSumNoSync(sm);
  // synchronize to guarantee partial sums are all completed and global value can be accumulated
  __syncthreads();

  t0Cnt = 0;
  //calculate t0 sum for whole region
  for(int i=0; i<blockDim.y; i++){
    t0Cnt += sm_base[i*blockDim.x];
  }

  float t0avgRegion = t0Sum/t0Cnt;
  // each thread now has correct t0Avg for the reagion

  if(threadIdx.x == 0 && threadIdx.y == 0){
    //printf("GPU t0_sum %f t0_cnt: %d \n" , t0Sum, t0Cnt );
    *T0avg = t0avgRegion; //TODO: remove if not needed outside anymore
  }
  __syncthreads();

  //TODO: Maybe do T0 shift here instead of on the fly when generating the traces.

  int elementIdx = threadIdx.x + threadIdx.y * blockDim.x;
  int stride = blockDim.x*blockDim.y;
  offsetWarpLocal = offsetWarp+ elementIdx;

  int numElements = offsetsPerRow * regHeight;
  int prevRowN = 0;

  // update the per row warp offsets within a row to absolute offsets in the region.
  while((elementIdx/stride) * stride < numElements){
    //load batch of local offsets to sm
    if(elementIdx < numElements)
      *sm =  *offsetWarpLocal;
    else
      *sm = 0;

    __syncthreads(); // make sure everything is loaded
    //update batch of local offsets to global offsets (only #offsetsPerRow threads work here)
    if(threadIdx.y == 0 && threadIdx.x < offsetsPerRow){
      int workOnSm = 0;
      int eIdx = elementIdx;
      int * smLast = sm_base + offsetsPerRow-1; // last offset in row is num live beads in row
      int * smLocal = sm_base + threadIdx.x;
      while(workOnSm < stride && eIdx < numElements){

        *smLocal += prevRowN; //prevRowN is absolute number of live beads in all previous rows

        prevRowN = *smLast;  //update prevRowN to last field in this row
        smLast += offsetsPerRow;
        smLocal += offsetsPerRow;
        eIdx += offsetsPerRow;
        workOnSm += offsetsPerRow;
      }
    }

    __syncthreads(); // make sure everything is loaded

    //write back to global
    if(elementIdx < numElements)
      *offsetWarpLocal = *sm;

    //move to next batch
    offsetWarpLocal += stride;
    elementIdx += stride;
  }

}



// grid size is grid.x = regionsX = 6;
//        grid.y = imgHeight = 1288
// blockDim.x has to be the same as used for GenerateMetaData Kernel!!!!!
// block size block.x = warpsize = 32 (or maybe 16, needs to be tested)
//         block.y = TBD = probably 8 or so,

// one block works on on row of a region, where the warps slide
// across the row with a stride of blockdim.x*blockdim.y
__global__
void GenerateAllBeadTraceFromMeta_k (
    FG_BUFFER_TYPE * fgBufferRegion[],
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
)
{

  extern __shared__ int smemGenBeadTraces[];  //4 Byte per thread

  short * img = raw.image;
  int imgW = raw.cols;

  //helper
  int warpsPerRow = ((regMaxW + blockDim.x-1)/blockDim.x);
  int offsetsPerRow = warpsPerRow + 1;

  int stride = blockDim.x * blockDim.y; // stride within a row (only relevant if (blockDim.x*blockDim.y) < wells per row

  //int imgFrameStride = imgW * imgH;



  int * sm_base = smemGenBeadTraces;
  int * sm_warp_base = sm_base + threadIdx.y*blockDim.x;
  int * sm = sm_warp_base + threadIdx.x;


  //region params and
  int regy = blockIdx.y / regMaxH;
  int regId = regy * gridDim.x + blockIdx.x;  // block x & y are a grid of regions by regions*rows

  int rowToWorkOn = blockIdx.y - regy*regMaxH; //number of row within region



  int numFrames = nptsRegion[regId];
  int numLBeads = lBeadsRegion[regId];
  float T0average = T0avg[regId];
  const int * framesPerPoint = framesPerPointRegion[regId];
  FG_BUFFER_TYPE * fgPtr = fgBufferRegion[regId]; //+ regId*(regMaxW*regMaxH*numFrames);


  //move pointers to beginning of row to work on
  int RegionBaseOffset = regy * (regMaxH * imgW) + blockIdx.x*regMaxW ;  // offset from base to first element in region
  int AbsoluteRowOffset = RegionBaseOffset + rowToWorkOn*imgW; // offset of region plus offset of row the block works on



  bfmask += AbsoluteRowOffset; //go to beginning of region in mask
  t0Est += AbsoluteRowOffset;
  img += AbsoluteRowOffset;

  WarpOffsets += regId*(regMaxH*offsetsPerRow) + rowToWorkOn * offsetsPerRow; //point at beginning of region in warp meta data



  int regWidth =  (imgW-(blockIdx.x*regMaxW)< regMaxW )?(imgW%regMaxW):(regMaxW);
  //regHeight not needed since grid.y already defines the height with the number of blocks with one block per row
  //int regHeight = (imgH-(regy*regMaxH) < regMaxH)?(imgH%regMaxH):(regMaxH);

  //TODO: might need padding
  //  int fgFrameStride = regWidth*regHeight;


  //warp level
  //start offset per warp/thread for this row
  int warpIdx = threadIdx.y;
  WarpOffsets += warpIdx;
  int localIdx = warpIdx * blockDim.x + threadIdx.x;

  bfmask += localIdx;
  t0Est += localIdx;
  img += localIdx;

  int AbsoluteWellOffset = AbsoluteRowOffset + warpIdx * blockDim.x + threadIdx.x;
  //slide warps accross row
  while(warpIdx < warpsPerRow){

    //load mask for warp
    if(localIdx < regWidth)
      *sm = Match(bfmask,(MaskType)MaskLive)?1:0;
    else
      *sm=0;

    int localoffset = RunningSumToCurrentThread(sm_warp_base) + *WarpOffsets;

    int IamAlive = *sm;
    WarpSumNoSync(sm);

    int numLBeadsWarp = *sm_warp_base;


    //Do Work if IamAlive!

    /////////////////////////////////
    //generate bead traces for live
    //beads in warp
    if(numLBeadsWarp > 0){

      float * smf_warp_base = (float*) sm_warp_base;
      float * smf = (float*) sm;  // one float per thread

      //do t0 shift average over warp
      *smf = (IamAlive)?( *t0Est - T0average ):(0);

      WarpSumNoSync(smf);
      float localT0 = (*smf_warp_base)/ (float)numLBeadsWarp;

      LoadImgWOffset(&fgPtr[localoffset],framesPerPoint, numFrames, numLBeads, localT0, raw, AbsoluteWellOffset);

    }
    ////////////////////////////////

    warpIdx += blockDim.y; //shift by num warps
    WarpOffsets += blockDim.y;
    localIdx += stride;
    bfmask += stride;
    t0Est += stride;
    img += stride;
    AbsoluteWellOffset += stride;
  }

  //thread level


}

// grid size is grid.x = regionsX = 6;
//        grid.y = imgHeight = 1288
// blockDim.x has to be the same as used for GenerateMetaData Kernel!!!!!
// block size block.x = warpsize = 32 (or maybe 16, needs to be tested)
//        block.y = TBD = probably 8 or so,

// one block works on on row of a region, where the warps slide
// across the row with a stride of blockdim.x*blockdim.y
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
)
{

  extern __shared__ int smemGenBeadTraces[];  //4 Byte per thread

  short * img = raw.image;
  int imgW = raw.cols;

  //helper
  int warpsPerRow = ((regMaxW + blockDim.x-1)/blockDim.x);
  int offsetsPerRow = warpsPerRow + 1;

  int stride = blockDim.x * blockDim.y; // stride within a row (only relevant if (blockDim.x*blockDim.y) < wells per row

  //int imgFrameStride = imgW * imgH;



  int * sm_base = smemGenBeadTraces;
  int * sm_warp_base = sm_base + threadIdx.y*blockDim.x;
  int * sm = sm_warp_base + threadIdx.x;

  // create local block index padded by start offset ( blocks per region: 1 x maxRegH )
  // since we cannot map the grid on hte image anymore if we only work on section of
  // the block per kernel invocation
  dim3 gBlockIdx( (blockIdx.x + regStartX),(blockIdx.y+regStartY*regMaxH));

  // to allow for blocking the image we cannot use gridDim.x anymore to
  // determine number of regions in x direction
  int numRegsX = (imgW + regMaxW -1)/regMaxW;



  //region params and
  int regy = gBlockIdx.y / regMaxH;
  int regId = regy * numRegsX + gBlockIdx.x;  // block x & y are a grid of regions by regions*rows

  // since the regId does not map to the now smaller fg_buffer anymore we have to determine the output buffer id
  int fgOutputBufferId = (blockIdx.y / regMaxH)*gridDim.x + blockIdx.x;  // block x & y are a grid of regions by regions*rows

  int rowToWorkOn = gBlockIdx.y - regy*regMaxH; //number of row within region



  int numFrames = nptsRegion[regId];
  int numLBeads = lBeadsRegion[regId];
  float T0average = T0avg[regId];
  int * framesPerPoint = framesPerPointRegion[regId];

  //we now have fewer fg buffers than regions.
  FG_BUFFER_TYPE * fgPtr = fgBufferRegion[fgOutputBufferId];


  //move pointers to beginning of row to work on
  int RegionBaseOffset = regy * (regMaxH * imgW) + gBlockIdx.x*regMaxW ;  // offset from base to first element in region
  int AbsoluteRowOffset = RegionBaseOffset + rowToWorkOn*imgW; // offset of region plus offset of row the block works on



  bfmask += AbsoluteRowOffset; //go to beginning of region in mask
  t0Est += AbsoluteRowOffset;
  img += AbsoluteRowOffset;

  WarpOffsets += regId*(regMaxH*offsetsPerRow) + rowToWorkOn * offsetsPerRow; //point at beginning of region in warp meta data



  int regWidth =  (imgW-(gBlockIdx.x*regMaxW)< regMaxW )?(imgW%regMaxW):(regMaxW);
  //regHeight not needed since grid.y already defines the height with the number of blocks with one block per row
  //int regHeight = (imgH-(regy*regMaxH) < regMaxH)?(imgH%regMaxH):(regMaxH);

  //TODO: might need padding
  //  int fgFrameStride = regWidth*regHeight;


  //warp level
  //start offset per warp/thread for this row
  int warpIdx = threadIdx.y;
  WarpOffsets += warpIdx;
  int localIdx = warpIdx * blockDim.x + threadIdx.x;

  bfmask += localIdx;
  t0Est += localIdx;
  img += localIdx;

  int AbsoluteWellOffset = AbsoluteRowOffset + warpIdx * blockDim.x + threadIdx.x;
  //slide warps across row
  while(warpIdx < warpsPerRow){

    //load mask for warp
    if(localIdx < regWidth)
      *sm = Match(bfmask,(MaskType)MaskLive)?1:0;
    else
      *sm=0;

    int localoffset = RunningSumToCurrentThread(sm_warp_base) + *WarpOffsets;

    int IamAlive = *sm;
    WarpSumNoSync(sm);

    int numLBeadsWarp = *sm_warp_base;


    //Do Work if IamAlive!

    /////////////////////////////////
    //generate bead traces for live
    //beads in warp
    if(numLBeadsWarp > 0){

      float * smf_warp_base = (float*) sm_warp_base;
      float * smf = (float*) sm;  // one float per thread

      //do t0 shift average over warp
      *smf = (IamAlive)?( *t0Est - T0average ):(0);
      WarpSumNoSync(smf);
      float localT0 = (*smf_warp_base)/ (float)numLBeadsWarp;

      LoadImgWOffset( &fgPtr[localoffset],framesPerPoint, numFrames, numLBeads, localT0, raw, AbsoluteWellOffset);

    }
    ////////////////////////////////

    warpIdx += blockDim.y; //shift by num warps
    WarpOffsets += blockDim.y;
    localIdx += stride;
    bfmask += stride;
    t0Est += stride;
    img += stride;
    AbsoluteWellOffset += stride;
  }

  //thread level


}





//input width and height
template<typename TDest, typename TSrc>
__global__
void transposeDataKernel(TDest *dest, TSrc *source, int width, int height)
{
  __shared__ float tile[32][32+1];

  if(sizeof(TDest) > sizeof(float)){
    printf ("TRANSPOSE KERNEL ERROR: destination type cannot be larger than %d bytes\n", sizeof(float));
    return;
  }

  TDest * smPtr;

  int xIndexIn = blockIdx.x * 32 + threadIdx.x;
  int yIndexIn = blockIdx.y * 32 + threadIdx.y;


  int Iindex = xIndexIn + (yIndexIn)*width;

  int xIndexOut = blockIdx.y * 32 + threadIdx.x;
  int yIndexOut = blockIdx.x * 32 + threadIdx.y;

  int Oindex = xIndexOut + (yIndexOut)*height;

  smPtr = (TDest*) &tile[threadIdx.y][threadIdx.x];

  if(xIndexIn < width && yIndexIn < height) *smPtr = (TDest)source[Iindex];

  smPtr = (TDest*) &tile[threadIdx.x][threadIdx.y];

  __syncthreads();


  if(xIndexOut < height && yIndexOut < width) dest[Oindex] = *smPtr;
}


template __global__ void transposeDataKernel<float,float>(float*,float*,int,int);
template __global__ void transposeDataKernel<short,short>(short*,short*,int,int);
template __global__ void transposeDataKernel<float,short>(float*,short*,int,int);


/*



  // these are used by both the background and live-bead
  const int avgover = 8;  // 2^n < warp_size to remove the need for sync
  int l_coord;
  int rx,rxh,ry,ryh;
  FG_BUFFER_TYPE *fgPtr;


  int ibd = threadIdx.x + blockIdx.x*blockDim.x;
  int avg_offset = (threadIdx.x/avgover)*avgover;

  float* smlocalT0 = localT0 + threadIdx.x;
  float* avgT0 = localT0 + (threadIdx.x/avgover)*avgover;
  //CP+=sid; // update to point to const memory struct for this stream

  if(ibd >= numLBeads) ibd = numLBeads-1;  //

  numLBeads = ((numLBeads+31)/32)*32;  //needs to be padded to 128bytes if working with transposed data

  rx= ptrx[ibd];
  ry= ptry[ibd];


 *smlocalT0 = t0_map[rx+ry*regMaxW];

  rxh = rx + regCol; //const
  ryh = ry + regRow; //const

  l_coord = ryh*raw.cols+rxh;  //

  //fgPtr = &fg_buffers[bead_flow_t*nbdx+npts*iFlowBuffer];
  //fgPtr = fg_buffers + (numLBeads*npts*iFlowBuffer+ibd);
  fgPtr = fg_buffer + ibd;


  if(threadIdx.x%avgover == 0){
    for(int o = 1; o < avgover; o++){
 *smlocalT0 += smlocalT0[o]; // add up in shared mem
    }
 *smlocalT0 /= avgover; // calc shared mem avg for warp or block
  }

  //LoadImgWOffset(raw, fgPtr, time_cp->frames_per_point, npts, l_coord, localT0);
  LoadImgWOffset(raw, fgPtr, frames_per_point, numFrames, numLBeads, l_coord, *avgT0);
  //KeepEmptyScale(region, my_beads,img, iFlowBuffer);
  //      KeepEmptyScale(bead_scale_by_flow, rx, ry, numLBeads, regCol, regRow, regW, regH, rawCols, smooth_max_amplitude, iFlowBuffer);

}

 */




