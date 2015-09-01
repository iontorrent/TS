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

namespace StreamingKernels {

void copySingleFlowFitConstParamAsync(ConstParams* ptr, int offset, cudaStream_t stream);

void copyMultiFlowFitConstParamAsync(ConstParams* ptr, int offset, cudaStream_t stream);

void copyFittingConstParamAsync(ConstParams* ptr, int offset, cudaStream_t stream);

void copyXtalkConstParamAsync(ConstXtalkParams* ptr, int offset, cudaStream_t stream);

void initPoissonTables(int device, float **poiss_cdf);   // PoissonCDFApproxMemo& poiss_cache);

void initPoissonTablesLUT(int device, void **poissLUT);   // PoissonCDFApproxMemo& poiss_cache);

void destroyPoissonTables(int device);

void  PerFlowGaussNewtonFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
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
  float* meanErr,
  // other inputs
  float minAmpl,
  float maxKmult,
  float minKmult, 
  float adjKmult,
  bool fitKmult,
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  bool useDynamicEmphasis,
//  int * pMonitor,
  int sId /*= 0*/,
  int flow_block_size
); 

void  PerFlowRelaxKmultGaussNewtonFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
  // inputs
  float* fg_buffers_base, // NxF
  float* emphasis,
  float* nucRise, 
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
  float adjKmult,
  bool fitKmult,
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  bool useDynamicEmphasis,
//  int * pMonitor,
  int sId /*= 0*/,
  int flow_block_size
); 


void  PerFlowHybridFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
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
  float* meanErr,
  // other inputs 
  float minAmpl,
  float maxKmult,
  float minKmult,  
  float adjKmult,
  bool fitKmult,
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  bool useDynamicEmphasis,
//  int * pMonitor,
  int sId /*= 0*/,
  int switchToLevMar /*= 3*/,
  int flow_block_size
);


void  PerFlowLevMarFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
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
  float* meanErr,
  // other inputs
  float minAmpl,
  float maxKmult,
  float minKmult, 
  float adjKmult,
  bool fitKmult,
  int realFnum,
  int num_beads, // 4
  int num_frames, // 4
  bool useDynamicEmphasis,
//int * pMonitor,
  int sId /*= 0*/,
  int flow_block_size
); 



///////// Pre-processing kernel (bkg correct and well params calculation);
void PreSingleFitProcessing(dim3 grid, dim3 block, int smem, cudaStream_t stream,// Here FL stands for flows
  // inputs from data reorganization
  float* pCopies, // N
  float* pR, // N
  float* pPhi, // N
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
  int sId /*= 0*/,
  int flow_block_size
);

///////// Xtalk computation kernel wrapper
void NeighbourContributionToXtalk(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,// Here FL stands for flows
  // inputs from data reorganization
  float* pR, // N
  float* pCopies, // N
  float* pPhi, // N
  float* sbg, // FLxF 
  float* fgbuffers, // FLxFxN
  bead_state* pState,
  // other inputs 
  int startingFlowNum, // starting flow number to calculate absolute flow num
  int currentFlowIteration,
  int num_beads, // 4
  int num_frames, // 4
  float* scratch_buf,
  float* nei_talk,
  int sId =0
);

void XtalkAccumulation(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  bead_state* pState,
  int num_beads, // 4
  int num_frames, // 4
  int* neiIdxMap, // MAX_XTALK_NEIGHBOURS x N
  float* nei_xtalk, // neixNxF
  float* xtalk, // NxF
  int sId
);

void CalculateGenericXtalkForSimpleModel(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  int num_beads, // 4
  int num_frames, // 4
//  int regW,
//  int regH,
  bead_state* pState,
  int* sampNeiIdxMap,
  float* nei_xtalk,
  float* xtalk, // NxF
  float* genericXtalk, // GENERIC_SIMPLE_XTALK_SAMPLE x F
  int sId);

void ComputeXtalkAndZeromerCorrectedTrace(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  int currentFlowIteration,
  float* fgbuffers, // FLxFxN
  int num_beads, // 4
  int num_frames, // 4
  float* genericXtalk, // neixNxF
  float* xtalk, // NxF
  float* pCopies, // N
  float* pR, // N
  float* pPhi,// N
  float* pgain, // N
  float* sbg, // FLxF 
  float* dark_matter, // FLxF
  float* pPCA_vals, 
  int flowNum, // starting flow number to calculate absolute flow num
  int sId
);

void BuildMatrix( dim3 grid, dim3 block, int smem, cudaStream_t stream, 
  float* pPartialDeriv, // S*FLxNxF   //scatch
  unsigned int * pDotProdMasks, // pxp
  int num_steps,
  int num_params,
  int num_beads,
  int num_frames,
  // outputs
  float* pJTJ, // (px(p+1)/2)xN
  float* pRHS, // pxN 
  int vec /*= 1  */,
  int flow_block_size
  );

void MultiFlowLevMarFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
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
  int sId,
  int flow_block_size
  );

void ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow(
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
  CpuStep* psteps, // we need a specific struct describing this config for this well fit for GPU
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
  int sId,
  int flow_block_size
); 

void  PerFlowAlternatingFit(
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

void TaubAdjustForExponentialTailFitting(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  bead_state* pState,
  float* fg_buffers,
  float* Ampl,
  float* pR,
  float* pCopies, // N
  float* pPhi, // N
  float* avg_trc,
  float* fval,
  float* tmp_fval,
  float* err,
  float* jac,
  int num_beads,
  int num_frames,
  float* tauAdjust,
  int sId,
  int flow_block_size
);

void ExponentialTailFitting(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  float bkg_scale_limit,
  float bkg_tail_dc_lower_bound,
  bead_state* pState,
  float* tauAdjust,
  float* Ampl,
  float* pR,
  float* pCopies, // N
  float* pPhi, // N
  float* fg_buffers,
  float* bkg_trace,
  float* tmp_fval,
  int num_beads,
  int num_frames,
  int flowNum,
  int sId,
  int flow_block_size
);

void ProjectionSearch(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  bead_state* pState,
  float* fg_buffers, // FLxFxN (already background and xtalk corrected if applicable))
  float* emphasisVec, // FxLAST_POISSON_TABLE_COL
  float* nucRise, // ISIG_SUB_STEPS_MULTI_FLOW*F*FL 
  float* pBeadParamsBase,
  float* fval, // NxF
  int realFnum, // starting flow number in block of 20 flows
  int num_beads,
  int num_frames,
  int sId,
  int flow_block_size
);

void transposeData(dim3 grid, dim3 block, int smem, cudaStream_t stream,float *dest, float *source, int width, int height);

///////// Transpose Kernel
void transposeDataToFloat(dim3 grid, dim3 block, int smem, cudaStream_t stream,float *dest, FG_BUFFER_TYPE *source, int width, int height);

void RecompressRawTracesForSingleFlowFit(
  dim3 grid, 
  dim3 block, 
  int smem, 
  cudaStream_t stream,
  float* fgbuffers, // FLxFxN
  float* scratch,
  int startFrame,
  int oldFrames,
  int newFrames,
  int numFlows,
  int num_beads,
  int sId);


} // namespace

#endif // STREAMINGKERNELS_H
