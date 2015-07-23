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

// Accumulates sm values of blockDim.x and stores them in sm value of thread 0;
// does not distribute results in sm, hence
// all values in sm for threads with Idx.x > 0 are garbage afterwards
// only works if blockDim.x <= warpsize since no sync

template<typename T>
__device__ inline
void WarpSumNoSync(T * smP)
{
  volatile T * sm = smP;
  int offset = blockDim.x >> 1;
  while(offset >= 1){
    if(threadIdx.x < offset) *sm += *(sm + offset);
    offset = offset >> 1;
  }
}

//sums only up for groups of n where n has to be a warpsize/(2^k)
template<typename T>
__device__ inline
void WarpSumNoSync(T * smP, int n)
{
  volatile T * sm = smP;
  int tid = threadIdx.x % n;
  int offset = n >> 1;
  while(offset >= 1){
    if(tid < offset) *sm += *(sm + offset);
    offset = offset >> 1;
  }
  *sm = *(sm-tid); //distribute to all threads sm address
}


//sums up shared memory from sm[0] to sm[threadIdx.x]
template<typename T>
__device__ inline
T RunningSumToCurrentThread(T * smP)
{
  volatile T * sm = smP;
  T s = 0;
  for(int i = 0 ; i<blockDim.x; i++)
    s += (i<=threadIdx.x)?(sm[i]):0;
  return s;
}


//reduces shared memory and returns sum,
//no sync at the end make sure to sync before using smem after function call
template<typename T>
__device__ inline
T ReduceSharedMemory(T* base, T*local)
{
  WarpSumNoSync(local);
  //Synchronize to allow to sum up partial sums from all warps within block
  __syncthreads();

  T sum = 0;
  //calculate num live beads for whole region in all threads
  for(size_t i=0; i<blockDim.y; i++){
    sum += base[i*blockDim.x]; //all threads get correct sum from SM base pointer
  }
  return sum;
}

// END WARP LEVEL FUNCTIONS
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
void ReduceAndAvgAtBlockLevel(T *sm, T N, bool avg) {
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


