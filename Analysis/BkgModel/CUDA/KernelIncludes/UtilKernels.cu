/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 * UtilKernels.cu
 *
 *  Created on: Mar 5, 2014
 *      Author: jakob
 */

#include "UtilKernels.h"


//Util device functions

/////////////////////////////////////////////////////////////
//WARP LEVEL NO SYNC FUNCTIONS, USE WITH CAUTION





/*********************************8
 *
 * shared memory pointer naming conventions:
 *
 * [v][type]sm[OffsetDescriptor][Usage]
 *
 * v:               if pointer name starts with a v it is declared a volatile to guarantee warp safe execution
 *
 * type:            (optional, only used for temporary type casts) can be i=int, f=float, d=double,...
 *
 * OffsetDesriptor: if none given: e.g. sm, ism, the pointer points to a unique address/offset used only by the current thread
 *                  Base: points to the first element of the per thread/per Warp memory. all warps in block point to the same address.
 *                  Warp: (sometimes shortened to W) pointer points to the first element of a buffer of warp size used only by the the threads in the current warp
 *
 * Usage:            suffix do indicate special use or different sm layout with num elements != num threads
 *
 *
 * Examples:
 *            ism:          pointer to thread specific address casted to int*
 *            vsmWarpTrace  address pointing to the first element of a trace-buffer specific to the current warp which his declared volatile
 *            smWarp    pointer to the first element of a buffer of warp size used only by the the threads in the current warp
 *
 *            volatile int * vism = (int*)&smWarp[threadIdx.x];
 *                          assigns the int* casted address for the current thread to a volatile declared pointer
 *
 *
 *
*/


// Accumulates sm values of blockDim.x and stores them in sm value of thread 0;
// does not distribute results in sm, hence
// all values in sm for threads with Idx.x > 0 are garbage afterwards
// only works if blockDim.x has a value of 2^n <= warpsize (32) since no sync is used.

// smP[0] will contain correct value
template<typename T>
__device__ inline
void WarpSumNoSync(T * sm)
{
  volatile T * vsm = sm;
  int offset = blockDim.x >> 1;
  while(offset >= 1){
    if(threadIdx.x < offset) *vsm += *(vsm + offset);
    offset = offset >> 1;
  }
}

//sums only up for groups of n where n has to be a warpsize/(2^k)
template<typename T>
__device__ inline
void WarpSumNoSync(T * sm, int n)
{
  volatile T * vsm = sm;
  int tid = threadIdx.x % n;
  int offset = n >> 1;
  while(offset >= 1){
    if(tid < offset) *vsm += *(vsm + offset);
    offset = offset >> 1;
  }
  *vsm = *(vsm-tid); //distribute to all threads sm address
}


//sums up shared memory from sm[0] to sm[threadIdx.x]
template<typename T>
__device__ inline
T WarpRunningSumToCurrentThread(T * smWarp)
{
  volatile T * vsmWarp = smWarp;
  T s = 0;
  for(int i = 0 ; i<blockDim.x; i++)
    s += (i<=threadIdx.x)?(vsmWarp[i]):0;
  return s;
}


//accumulates one trace with nframes per thread into the output Trace by accumulaton
//the input races frame by frame in shared memory. all operations are warp based and rely on
//blockDim.x == warp size
// output trace buffer has to be initialized with 0 since accumulated value gets added to current value.
//
template<typename T>  //blockDim.x == warpsize !! sm buffer must be at least smBase[numframes]
__device__ inline
void WarpTraceAccum(T* outTrace, T*smWarp, T*localTrace, const int nframes)
{
  volatile float * vsmW = smWarp;
  volatile float * vOut = outTrace;
  for(int f=0; f<nframes; f++){
    vsmW[threadIdx.x] = localTrace[f]; // each thread of the warp writes one frame
    WarpSumNoSync(&smWarp[threadIdx.x]);
    if(threadIdx.x == 0)
      vOut[f] += vsmW[0];
  }
}



//accumulates one frame of trace held
//validTrace is option and defaults to true, if some traces are not part of the accumulation for those threads/traces validTrace has to be set to false
template<typename T>  //blockDim.x == warpsize !! sm buffer must be at least smBase[numframes]
__device__ inline
void WarpTraceAccumSingleFrame(T* smWarpTrace, const int frame, T*smWarp, const T localFrameValue, const bool validTrace)
{
  //volatile float * vsmW = smWarp;
  volatile float * vsm = smWarp + threadIdx.x;
  volatile float * vOut = smWarpTrace;
  *vsm = (validTrace)?(localFrameValue):(0.0f);
  WarpSumNoSync(vsm);
  if(threadIdx.x == 0)
      vOut[frame] += *vsm;

}
//overloaded from above, where input trace instead of just the frame value is passed
//validTrace is option and defaults to true, if some traces are not part of the accumulation for those threads/traces validTrace has to be set to false
template<typename T>
__device__ inline
void WarpTraceAccumSingleFrame(T* smWarpTrace, const int frame, T*smWarp, const T * localTrace, const bool validTrace)
{
  WarpTraceAccumSingleFrame( smWarpTrace, frame, smWarp, localTrace[frame], validTrace);
}



//same as above but each thread also passes a valid trace flag which tells if there is a trace to accumulate
//the function will also accumulate the number of valid traces and return the value
//blockDim.x == warpsize !! sm buffer must be at least smBase[numframes]
//validTrace is option and defaults to true, if some traces are not part of the accumulation for those threads/traces validTrace has to be set to false
template<typename T>
__device__ inline
int WarpTraceAccumCount(T* smWarpTrace, const int nframes, T*smWarp, const T*localTrace, const bool validTrace)
{
  volatile int * vismW = (int*)smWarp;
  volatile float * vsmW = smWarp;
  volatile int*vism = vismW+threadIdx.x; /// !!!!!
  *vism = (validTrace)?(1):(0);
  WarpSumNoSync(vism);
  int numTraces = vismW[0];
  if(numTraces > 0){
    volatile float * vOut = smWarpTrace;
    for(int f=0; f<nframes; f++){
      vsmW[threadIdx.x] = (validTrace)?(localTrace[f]):(0); // each thread of the warp writes one frame
      WarpSumNoSync(&(smWarp[threadIdx.x]));
      if(threadIdx.x == 0)
        vOut[f] += vsmW[0];
    }
  }
  return numTraces;
}


// END WARP LEVEL FUNCTIONS
////////////////////////////////////////
//BLOCK LEVEL REDUCTION

//reduces shared memory and returns sum,
//no sync at the end make sure to sync before using smem after function call
template<typename T>
__device__ inline
T ReduceSharedMemory(T* smBase, T*sm)
{
  WarpSumNoSync(sm);
  //Synchronize to allow to sum up partial sums from all warps within block
  __syncthreads();

  T sum = 0;
  //calculate num live beads for whole region in all threads
  for(size_t i=0; i<blockDim.y; i++){
    sum += smBase[i*blockDim.x]; //all threads get correct sum from SM base pointer
  }
  return sum;
}



//accumulates num Warps traces with nframes which are stored consecutively in shared memory into one final trace
//parameters: outTrace: points to the first frame of a buffer of length nframes or larger
//            smTracesBase: points to the first frame of a buffer containing numWarps traces of length nframes.
//            nframes: number of frames.
template<typename T>
__device__ inline
void BlockTraceAccumfromWarps(T* outTrace, const T*smBaseTraces, const int nframes, const int maxCompFrames)
{
  int accumFrame = threadIdx.x + blockDim.x * threadIdx.y;
  int blockSize = blockDim.x * blockDim.y;

  __syncthreads();

  while (accumFrame < nframes){
    T accum = 0.0f;

    for(int i=0; i<blockDim.y; i++)
      accum += smBaseTraces[accumFrame+maxCompFrames*i];

    outTrace[accumFrame] = accum;
    accumFrame += blockSize;

  }
}

//same as above but result will be written to the first warps trace in shared memory.
template<typename T>
__device__ inline
void BlockTraceAccumfromWarpsInplace(T*smBaseTraces, const int nframes, const int maxCompFrames)
{
  int offset = threadIdx.x + blockDim.x * threadIdx.y;
  int blockSize = blockDim.x * blockDim.y;

  __syncthreads();

  while (offset < nframes){

    for(int i=1; i<blockDim.y; i++)
      smBaseTraces[offset] += smBaseTraces[offset+maxCompFrames*i];

    offset += blockSize;
  }
}

//as above but then stores the result to a global address
// atomicGlobalAccum: true does an atomic accumulation in global memory, if false value gets saved to global memory and current value get's overwritten.

template<typename T>
__device__ inline
void BlockTraceAccumfromWarpsInplaceToGlobal( T*gTrace,const size_t outFrameStride, T*smBaseTraces, const int nframes, const int maxCompFrames, const bool atomicGlobalAccum)
{
  int accumFrame = threadIdx.x + blockDim.x * threadIdx.y;
  int blockSize = blockDim.x * blockDim.y;
  gTrace += accumFrame * outFrameStride;
  volatile T * vsmBaseTrace = smBaseTraces;
  __syncthreads();
  while (accumFrame < nframes){

    for(int i=1; i<blockDim.y; i++)
      vsmBaseTrace[accumFrame] += vsmBaseTrace[accumFrame+maxCompFrames*i];

    if(atomicGlobalAccum)
      atomicAdd(gTrace,vsmBaseTrace[accumFrame]);
    else
      *gTrace = vsmBaseTrace[accumFrame];

    accumFrame += blockSize;
    gTrace += blockSize * outFrameStride;

  }
}


template<typename T>
__device__ inline
T BlockAccumPerThreadValuesAcrossWarpsSharedMem(T*smBase)
{
  __syncthreads();

  T sum = 0;
  for(int i=0; i<blockDim.y;i++)
      sum += smBase[threadIdx.x + i*blockDim.x];
  return sum;

}

//per warp value at each sm_warp_base: smBase[0],[warpsize],[2*warpsize].... will be accumulated in smBase[0] and then stored to global
// atomicGlobalAccum: true does an atomic accumulation in global memory, if false value gets saved to global memory and current value get's overwritten.
template<typename T>
__device__ inline
void BlockAccumValuePerWarpToGlobal( T*gValue, T*smBase, const bool atomicGlobalAccum)
{
  T sum = BlockAccumPerThreadValuesAcrossWarpsSharedMem(smBase);
  if (threadIdx.x==0 && threadIdx.y ==0){
    if(atomicGlobalAccum)
      atomicAdd(gValue,sum);
    else
      *gValue=sum;
  }
}

/////////////////////////////////////////////////////////



//checks if the passed bit is set in the mask
//for more than one bit passed
//default: match any, returns true if at least one of the bits is matched
//if matchAll == true: returns true only if all bits passed in type are set in mask
__host__ __device__ inline
bool  Match(const unsigned short * mask,  unsigned short type, bool matchAll)
{
  unsigned short afterAnd = LDG_LOAD(mask) & type;
  if(!matchAll)
    return (afterAnd ? true : false ); // at least one bit matches
  else
    return  afterAnd == type; // all bits matched
}


template<typename T>
__device__ inline
void clampT ( T &x, T a, T b)
{
  // Clamps x between a and b
  x = (x < a ? a : (x > b ? b : x));
}

//checks mask if well can be used as a empty reference
__device__ inline
bool useForEmpty(const unsigned short *bfmask)
{
  //ToDo add Duds if needed
  return ( Match(bfmask,(unsigned short)MaskReference) &&
      !Match(bfmask,(unsigned int)(MaskPinned | MaskIgnore)) // && !Match(bfmask,(MaskType)MaskIgnore)
  );
}

//interpolate and correct
template<typename T> __device__ inline
float iPc(const T lvalue, const T rvalue, const float frac, const float c)
{
  //return (1-frac)*lvalue + frac*rvalue - c;
  return (float)((rvalue - lvalue)*frac + lvalue - c);
}






//uncompresses trace from a buffer with a stride of framestride into the buffer uncompTrace with consecutive elements (stride = 1) of length CfP.getUmcompressedFrames
template<typename T>
__device__ inline
void GetUncompressTrace(float * uncompTrace, const ConstantFrameParams & CfP, const T * compTrace, const int frameStride  )
{


  if(CfP.getRawFrames() < CfP.getUncompFrames()){
    int my_frame = 0;
    float prev=*compTrace;
    float next=0.0f;
    *uncompTrace = prev;

    for ( my_frame=1; my_frame<CfP.getUncompFrames(); my_frame++ )
    {
      // need to make this faster!!!
      int interf= CfP.interpolatedFrames[my_frame];

      next = compTrace[frameStride*interf];
      prev = compTrace[frameStride*(interf-1)];

      // interpolate
      float mult = CfP.interpolatedMult[my_frame];
      uncompTrace[my_frame] = ( prev-next ) *mult + next;
    }

  }else{
    // the rare "uncompressed" case
    for ( int my_frame=0; my_frame<CfP.getUncompFrames(); my_frame++ )
    {
      uncompTrace[my_frame] = compTrace[my_frame*frameStride];
    }
  }
}


template<typename T>
__device__ inline
float ComputeDcOffsetForUncompressedTrace ( const T *bPtr, const int uncompressedFrames, const float t_start, const float t_end )
{
  float cnt = 0.0001f;
  float dc_zero = 0.000f;

  int above_t_start = ( int ) ceil ( t_start );
  int below_t_end = ( int ) floor ( t_end );

  //ToDo: figure out how to propagate back to CPU
  //assert (0 <= above_t_start) && (above_t_start-1 < imgFrames) && (0 <= below_t_end+1) && (below_t_end < imgFrames));

  for ( int pt = above_t_start; pt <= below_t_end; pt++ )
  {

    dc_zero += ( float ) ( bPtr[pt] );
    cnt += 1.0f;

  }

  // include values surrounding t_start & t_end weighted by overhang
  if ( above_t_start > 0 )
  {
    float overhang = ( above_t_start-t_start );
    dc_zero = dc_zero + bPtr[above_t_start-1]*overhang;
    cnt += overhang;
  }

  if ( below_t_end < ( uncompressedFrames-1 ) )
  {
    float overhang = ( t_end-below_t_end );
    dc_zero = dc_zero + bPtr[below_t_end+1]* ( t_end-below_t_end );
    cnt += overhang;
  }
  dc_zero /= cnt;

  return dc_zero;

}

template<typename T>
__device__ inline
float ComputeDcOffsetForCompressedTrace ( const T * fgPtr,
    const size_t frameStride,
    const float* frameNumber,
    const float t_start,
    const float t_end,
    const int numFrames
)
{
  float dc_zero = 0.0f;
  float cnt = 0.0f;
  int pt;
  int pt1 = 0;
  int pt2 = 0;
  // TODO: is this really "rezero frames before pH step start?"
  // this should be compatible with i_start from the nuc rise - which may change if we change the shape???
  for (pt = 0;frameNumber[pt] < t_end;pt++)
  {
    pt2 = pt+1;
    if (frameNumber[pt]>t_start)
    {
      if (pt1 == 0)
        pt1 = pt; // set to first point above t_start

      dc_zero += (float) (fgPtr[pt*frameStride]);
      cnt += 1.0f; // should this be frames_per_point????
      //cnt += time_cp->frames_per_point[pt];  // this somehow makes it worse????
    }
  }

  // include values surrounding t_start & t_end weighted by overhang
  if (pt1 > 0) {
    // timecp->frameNumber[pt1-1] < t_start <= timecp->frameNumber[pt1]
    // normalize to a fraction in the spirit of "this somehow makes it worse"
    float den = (frameNumber[pt1]-frameNumber[pt1-1]);
    if ( den > 0 ) {
      float overhang = (frameNumber[pt1] - t_start)/den;
      dc_zero = dc_zero + fgPtr[(pt1-1)*frameStride]*overhang;
      cnt += overhang;
    }
  }

  if ( (pt2 < numFrames) && (pt2>0) ) {
    // timecp->frameNumber[pt2-1] <= t_end < timecp->frameNumber[pt2]
    // normalize to a fraction in the spirit of "this somehow makes it worse
    float den = (frameNumber[pt2]-frameNumber[pt2-1]);
    if ( den > 0 ) {
      float overhang = (t_end - frameNumber[pt2-1])/den;
      dc_zero = dc_zero + fgPtr[pt2*frameStride]*overhang;
      cnt += overhang;
    }
  }


  return (cnt > 0.0f )?(dc_zero/cnt):(0.0f);
}


//empty trace handling
__device__ inline
void TShiftAndPseudoCompression ( float *fbkg, const float *bg, const float * frameNumber,const float tshift, const int npts,  int const uncompFrames, const float dcoffset )
{

  //fprintf(stdout, "tshift %f\n", tshift);

  for (int i=0;i < npts;i++){
    // get the frame number of this data point (might be fractional because this point could be
    // the average of several frames of data.  This number is the average time of all the averaged
    // data points
    float t=frameNumber[i];
    float fn=t-tshift;
    if (fn < 0.0f) fn = 0.0f;
    if (fn > (ConstFrmP.getUncompFrames()-2)) fn = ConstFrmP.getUncompFrames()-2;
    int ifn= (int) fn;
    float frac = fn - ifn;
    if(dcoffset == 0)
      fbkg[i]  = ( (1-frac) *bg[ifn] + frac*bg[ifn+1]);
    else
      fbkg[i]  = ( (1-frac) *bg[ifn] + frac*bg[ifn+1]) - dcoffset;

    //= ((tmp == tmp)?(tmp):(0)); //nan check //assert ( !isnan(fbkg[i]) );
  }
}

__device__ inline
void TShiftAndPseudoCompressionOneFrame ( //tshift always left shift
    float *fbkg, const float *bg, const float * frameNumber,const float tshift, const int thisFrame, const float dcoffset )
{

  //printf("tshift %f\n", tshift);

  //for (int i=0;i < npts;i++){
  int i = thisFrame;
  // get the frame number of this data point (might be fractional because this point could be
  // the average of several frames of data.  This number is the average time of all the averaged
  // data points
  float t=frameNumber[i];
  float fn=t-tshift;
  if (fn < 0.0f) fn = 0.0f;
  if (fn > (ConstFrmP.getUncompFrames()-2)) fn = ConstFrmP.getUncompFrames()-2;
  int ifn= (int) fn;
  float frac = fn - ifn;
  if(dcoffset == 0)
    fbkg[i]  = ( (1-frac) *bg[ifn] + frac*bg[ifn+1]);
  else
    fbkg[i]  = ( (1-frac) *bg[ifn] + frac*bg[ifn+1]) - dcoffset;
  //= ((tmp == tmp)?(tmp):(0)); //nan check //assert ( !isnan(fbkg[i]) );

}



//Util Kernels

//input width and height
template<typename TDest, typename TSrc>
__global__
void transposeDataKernel(TDest *dest, TSrc *source, int width, int height)
{
  __shared__ float tile[32][32+1];

  if(sizeof(TDest) > sizeof(float)){
    //printf ("TRANSPOSE KERNEL ERROR: destination type cannot be larger than %d bytes\n", sizeof(float));
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

// Reduces shared memory of power of 2 size (max 512 elements) and puts the sum in sm[0]
template<typename T>
__device__ inline
void ReduceAndAvgAtBlockLevel(T *sm, int N, bool avg) {
  if (blockDim.x >= 512) { if (threadIdx.x < 256) sm[threadIdx.x] += sm[threadIdx.x + 256]; } __syncthreads();
  if (blockDim.x >= 256) { if (threadIdx.x < 128) sm[threadIdx.x] += sm[threadIdx.x + 128]; } __syncthreads();
  if (blockDim.x >= 128) { if (threadIdx.x < 64) sm[threadIdx.x] += sm[threadIdx.x + 64]; } __syncthreads();

  if (threadIdx.x < 32) {
    if (blockDim.x >= 64) sm[threadIdx.x] += sm[threadIdx.x + 32];    
    if (blockDim.x >= 32) sm[threadIdx.x] += sm[threadIdx.x + 16];    
    if (blockDim.x >= 16) sm[threadIdx.x] += sm[threadIdx.x + 8];    
    if (blockDim.x >= 8)  sm[threadIdx.x] += sm[threadIdx.x + 4];    
    if (blockDim.x >= 4)  sm[threadIdx.x] += sm[threadIdx.x + 2];    
    if (blockDim.x >= 2)  sm[threadIdx.x] += sm[threadIdx.x + 1];    
  }

  if (threadIdx.x == 0 && avg)
    sm[0] /= (T)N;

  __syncthreads();
}

template<typename T>
__device__ inline
void SimplestReductionAndAverage(T *sm, int N, bool avg) { 
  T sum = 0.0f;
  if (threadIdx.x == 0) {
    for (int i=0; i<N; ++i)
      sum += sm[i];

    if (avg)
      sm[0] = sum / (T)N;
    else
      sm[0] = sum;
  }

  __syncthreads();
}

template __global__ void transposeDataKernel<float,float>(float*,float*,int,int);
template __global__ void transposeDataKernel<short,short>(short*,short*,int,int);
template __global__ void transposeDataKernel<float,short>(float*,short*,int,int);


