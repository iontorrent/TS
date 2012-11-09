/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef STREAMINGKERNELS_H
#define STREAMINGKERNELS_H

#include "CudaDefines.h"
#include "BkgFitOptim.h"
#include "BkgMagicDefines.h"
#include "ObsoleteCuda.h"

#include "ParamStructs.h"
//#include "PoissonCdf.h"

#include "cuda_error.h"


extern "C" 
void copySingleFlowFitConstParamAsync(ConstParams* ptr, int offset, cudaStream_t stream);

extern "C"
void copyMultiFlowFitConstParamAsync(ConstParams* ptr, int offset, cudaStream_t stream);

extern "C" 
void initPoissonTables(int device, float **poiss_cdf);   // PoissonCDFApproxMemo& poiss_cache);

extern "C" 
void initPoissonTablesLUT(int device, void **poissLUT);   // PoissonCDFApproxMemo& poiss_cache);


extern "C" 
void destroyPoissonTables(int device);



extern "C"
void  PerFlowLevMarFit_Wrapper(dim3 grid, dim3 block, int smem, cudaStream_t stream,
  // inputs
  float* fg_buffers_base, // NxF
  float* emphasis,
  float* nucRise, 
  // bead params
//  float* pAmpl, // N
//  float* pKmult, // N
//  float* pdmult, // N
//  float* pR, // N
//  float* pgain, // N
//  float* pSP, // N
  float * pBeadParamsBase, //N
  bead_state* pState,

  // scratch space in global memory
  float* err, // NxF
  float* fval, // NxF
  float* tmp_fval, // NxF
  float* jac, // NxF 
  float* meanErr,
  // other inputs
  float minAmpl,
  float maxKmult,
  float minKmult, 
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  bool useDynamicEmphasis,
//  int * pMonitor,
  int sId = 0
); 


///////// Pre-processing kernel (bkg correct and well params calculation);
extern "C"
void PreSingleFitProcessing_Wrapper(dim3 grid, dim3 block, int smem, cudaStream_t stream,// Here FL stands for flows
  // inputs from data reorganization
  float* pCopies, // N
  float* pR, // N
  float* pdmult, // N
  float* pgain, // N
  float* pAmpl, // FLxN
  float* pkmult, // FLxN
  float* sbg, // FLxF 
  float* dark_matter, // FLxF
  float* fgbuffers, // FLxFxN
  int flowNum, // starting flow number to calculate absolute flow num
  int num_beads, // 4
  int num_frames, // 4
  int numEv, // 4

  // outputs
//  float* pSP, // FLxN
//  float* ptauB, 
  bool alternatingFit,
  int sId = 0
);


extern "C"
void BuildMatrix_Wrapper( dim3 grid, dim3 block, int smem, cudaStream_t stream, 
  float* pPartialDeriv, // S*FLxNxF   //scatch
  unsigned int * pDotProdMasks, // pxp
  int num_steps,
  int num_params,
  int num_beads,
  int num_frames,
  // outputs
  float* pJTJ, // (px(p+1)/2)xN
  float* pRHS, // pxN 
  int vec = 1  
  );

extern "C"
void MultiFlowLevMarFit_Wrapper(dim3 grid, dim3 block, int smem, cudaStream_t stream,
  // inputs
  int maxEmphasis,
  float restrict_clonal,
  float* pobservedTrace, 
  float* pival,
  float* pfval,
  float* pnucRise, // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
  float* psbg, // FLxF
  float* pemphasis, // MAX_HPLEN+1 xF // needs precomputation
  float* pnon_integer_penalty, // MAX_HPLEN
  float* pdarkMatterComp, // NUMNUC * F  
  float* pbeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep_t
  float* pevalBeadParams,
  float* plambda,
  float* pjtj, // jtj matrix generated from build matrix kernel
  float* pltr, // scratch space to write lower triangular matrix
  float* pb, // rhs vector
  float* pdelta,
  unsigned int* paramIdxMap, 
  int num_params,
  int num_beads,
  int num_frames,
  // outputs
  float* presidual, // N 
  int sId
  );

extern "C"
void ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow_Wrapper(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  // inputs
  int maxEmphasis,
  float restrict_clonal,
  float* pobservedTrace, 
  float* pival, // FLxNxF   //scatch
  float* pscratch_ival, // FLxNxF
  float* pnucRise, // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
  float* psbg, // FLxF
  float* pemphasis, // MAX_HPLEN+1 xF // needs precomputation
  float* pnon_integer_penalty, // MAX_HPLEN
  float* pdarkMatterComp, // NUMNUC * F  
  float* pbeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep_t
  CpuStep_t* psteps, // we need a specific struct describing this config for this well fit for GPU
  unsigned int* pDotProdMasks,
  float* pJTJ,
  float* pRHS,
  int num_params,
  int num_steps,
  int num_beads,
  int num_frames,
  // outputs
  float* presidual,
  float* poutput, // total bead params x FL x N x F. Need to decide on its layout 
  int sId
); 

extern "C"
void  PerFlowAlternatingFit_Wrapper(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  // inputs
  float* fg_buffers_base, // NxF
  float* emphasis, // (MAX_HPLEN+1)xF
  float* nucRise, 
  // bead params
  float* pAmpl, // N
  float* pKmult, // N
  float* pdmult, // N
  float* ptauB, // N
  float* pgain, // N
  float* pSP, // N
  float * pCopies, //N
  bead_state* pState,

  // scratch space in global memory
  float* err, // NxF
  float* fval, // NxF
  float* tmp_fval, // NxF
  float* meanErr,
  // other inputs 
  float minAmpl,
  float maxKmult,
  float minKmult,  
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  int sId
); 

extern "C"
void transposeData_Wrapper(dim3 grid, dim3 block, int smem, cudaStream_t stream,float *dest, float *source, int width, int height);

///////// Transpose Kernel
extern "C"
void transposeDataToFloat_Wrapper(dim3 grid, dim3 block, int smem, cudaStream_t stream,float *dest, FG_BUFFER_TYPE *source, int width, int height);




#endif // STREAMINGKERNELS_H
