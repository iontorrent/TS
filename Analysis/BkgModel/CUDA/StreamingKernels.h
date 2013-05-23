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
void copyFittingConstParamAsync(ConstParams* ptr, int offset, cudaStream_t stream);


extern "C"
void copyXtalkConstParamAsync(ConstXtalkParams* ptr, int offset, cudaStream_t stream);

extern "C" 
void initPoissonTables(int device, float **poiss_cdf);   // PoissonCDFApproxMemo& poiss_cache);

extern "C" 
void initPoissonTablesLUT(int device, void **poissLUT);   // PoissonCDFApproxMemo& poiss_cache);


extern "C" 
void destroyPoissonTables(int device);



extern "C"
void  PerFlowGaussNewtonFit_Wrapper(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
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

extern "C"
void  PerFlowHybridFit_Wrapper(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
  // inputs
  float* fg_buffers_base, // NxF
  float* emphasis, // F
  float* nucRise, 
  // bead params
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
  int sId = 0,
  int switchToLevMar = 3
);


extern "C"
void  PerFlowLevMarFit_Wrapper(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
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
//int * pMonitor,
  int sId = 0
); 



///////// Pre-processing kernel (bkg correct and well params calculation);
extern "C"
void PreSingleFitProcessing_Wrapper(dim3 grid, dim3 block, int smem, cudaStream_t stream,// Here FL stands for flows
  // inputs from data reorganization
  float* pCopies, // N
  float* pR, // N
  float* pgain, // N
  float* pAmpl, // FLxN
  float* sbg, // FLxF 
  float* dark_matter, // FLxF
  float* pPCA_vals,
  float* fgbuffers, // FLxFxN
  int flowNum, // starting flow number to calculate absolute flow num
  int num_beads, // 4
  int num_frames, // 4
  bool alternatingFit,
  int sId = 0
);

///////// Xtalk computation kernel wrapper
extern "C"
void NeighbourContributionToXtalk_Wrapper(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,// Here FL stands for flows
  // inputs from data reorganization
  float* pR, // N
  float* sbg, // FLxF 
  float* fgbuffers, // FLxFxN
  // other inputs 
  int startingFlowNum, // starting flow number to calculate absolute flow num
  int currentFlowIteration,
  int num_beads, // 4
  int num_frames, // 4
  float* scratch_buf,
  float* nei_talk,
  int sId =0
);

extern "C"
void XtalkAccumulationAndSignalCorrection_Wrapper(// Here FL stands for flows
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,// Here FL stands for flows
  int currentFlowIteration,
  float* fgbuffers, // FLxFxN
  int num_beads, // 4
  int num_frames, // 4
  int* neiIdxMap, // MAX_XTALK_NEIGHBOURS x N
  float* nei_xtalk, // neixNxF
  float* xtalk, // FLxN
  float* pCopies, // N
  float* pR, // N
  float* pgain, // N
  float* sbg, // FLxF 
  float* dark_matter, // FLxF
  float* pPCA_vals,
  int flowNum, // starting flow number to calculate absolute flow num
  int sId
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
void MultiFlowLevMarFit_Wrapper(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
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
  int l1type,
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
void TaubAdjustForExponentialTailFitting_Wrapper(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  float* fg_buffers,
  float* Ampl,
  float* pR,
  float* avg_trc,
  float* fval,
  float* tmp_fval,
  float* err,
  float* jac,
  int num_beads,
  int num_frames,
  float* tauAdjust,
  int sId
);

extern "C"
void ExponentialTailFitting_Wrapper(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  float* tauAdjust,
  float* Ampl,
  float* pR,
  float* fg_buffers,
  float* bkg_trace,
  float* tmp_fval,
  int num_beads,
  int num_frames,
  int flowNum,
  int sId
);

extern "C"
void transposeData_Wrapper(dim3 grid, dim3 block, int smem, cudaStream_t stream,float *dest, float *source, int width, int height);

///////// Transpose Kernel
extern "C"
void transposeDataToFloat_Wrapper(dim3 grid, dim3 block, int smem, cudaStream_t stream,float *dest, FG_BUFFER_TYPE *source, int width, int height);




#endif // STREAMINGKERNELS_H
