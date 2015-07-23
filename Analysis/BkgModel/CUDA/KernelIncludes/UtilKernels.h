/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 * UtilKernels.h
 *
 *  Created on: Mar 5, 2014
 *      Author: jakob
 */

#ifndef UTILKERNELS_H_
#define UTILKERNELS_H_

#include "DeviceParamDefines.h"

//Util device functions
//WARP LEVEL NO SYNC FUNCTIONS, USE WITH CAUTION
template<typename T>//blockDim.x == warpsize !!
__device__ inline void WarpSumNoSync(T * smP);

template<typename T>//blockDim.x == warpsize !!
__device__ inline void WarpSumNoSync(T * smP, int n);

template<typename T>//blockDim.x == warpsize !!
__device__ inline T RunningSumToCurrentThread(T * sm);

template<typename T>  //blockDim.x == warpsize !!
__device__  inline T ReduceSharedMemory(T* base, T*local);


// Block level reduction 
// blockDim.x should be power of 2 and less than equal to 512
template<typename T>
__device__ inline void ReduceAndAvgAtBlockLevel(T *sm, T N, bool avg);

template<typename T>
__device__ inline void SimplestReductionAndAverage(T *sm, int N, bool avg); 


//checks is the passed bit is set in the mask
//for more than one bit passed:
//if matchAll is set to false Match() returns true if at least one of the bits of type is set in matched
//if matchAll is set to true all bits of the passed type value have to be set in the mask for Match() to return true
__host__ __device__ inline bool  Match(const unsigned short * mask,  unsigned short type, bool matchAll = false);

template<typename T>
__device__ inline void clampT ( T &x, T a, T b);

//checks mask if well can be used as a empty reference
__device__ inline bool useForEmpty(const unsigned short *bfmask);


//interpolate and correct
template<typename T> __device__ inline
float iPc(const T lvalue, const T rvalue, const float frac, const float c = 0.0f);


//uncompresses trace from a buffer with a stride of framestride into the buffer uncompTrace with consecutive elements (stride = 1) of length CfP.getUmcompressedFrames
template<typename T> __device__ inline
void GetUncompressTrace(float * uncompTrace, const ConstantFrameParams & CfP, const T * compTrace, const int frameStride  );



//dc offset calculation for a uncompressed trace based on time points between t_start and T_end
template<typename T> __device__ inline
float ComputeDcOffsetForUncompressedTrace ( const T *bPtr, const int uncompressedFrames, const float t_start, const float t_end );

template<typename T> __device__ inline
float ComputeDcOffsetForCompressedTrace ( const T * fgPtr, const size_t frameStride, const float* frameNumber, const float t_start, const float t_end, const int numFrames);

//tshift always left shift and pseudo empty avg compression
__device__ inline
void TShiftAndPseudoCompression ( float *out_buff, const float *my_buff, const float * frameNumber,const float tshift, const int npts,  int const uncompFrames, const float dcoffset = 0.0f);
__device__ inline
void TShiftAndPseudoCompressionOneFrame ( float *fbkg, const float *bg, const float * frameNumber,const float tshift, const int thisFrame,  const float dcoffset = 0.0f );

//Util Kernels

template<typename TDest, typename TSrc>
__global__ void transposeDataKernel(TDest *dest, TSrc *source, int width, int height);



#endif /* UTILKERNELS_H_ */
