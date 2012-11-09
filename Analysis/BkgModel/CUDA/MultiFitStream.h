/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef MULTIFITSTREAM_H
#define MULTIFITSTREAM_H

// std headers
#include <iostream>
// cuda
#include "cuda_runtime.h"
#include "cuda_error.h"
#include "Utils.h"


#include "StreamManager.h"
#include "ParamStructs.h"
#include "JobWrapper.h"
#include "GpuMultiFlowFitControl.h"

#define CUDA_MULTIFLOW_NUM_FIT 2 

class MultiFitStream : public cudaStreamExecutionUnit
{
  int _myId; 
  GpuMultiFlowFitMatrixConfig* _fd[CUDA_MULTIFLOW_NUM_FIT];
  float _lambda_start[CUDA_MULTIFLOW_NUM_FIT];
  int _fit_iterations[CUDA_MULTIFLOW_NUM_FIT];
  int _clonal_restriction[CUDA_MULTIFLOW_NUM_FIT];
  float _restrict_clonal[CUDA_MULTIFLOW_NUM_FIT];

   //Execution specific values    
  static int _bpb;   // beads per block

  int _fitIter; 
  
  int _maxEmphasis;
  unsigned int _partialDerivMask; // mask to find out which params to compute partial derivative for


  ConstParams* _HostConstP;
  //Only members starting with _Host or _Dev are 
  //allocated. pointers with _h_p or _d_p are just pointers 
  //referencing memory inside the _Dev/_Host buffers!!!

  //host memory
  // fit specific inputs
  float* _pHostLambdaForBeadFit[CUDA_MULTIFLOW_NUM_FIT];
  unsigned int * _pHostJTJMatrixMapForDotProductComputation[CUDA_MULTIFLOW_NUM_FIT];
  unsigned int * _pHostBeadParamIdxMap[CUDA_MULTIFLOW_NUM_FIT];
  CpuStep_t* _pHostSteps[CUDA_MULTIFLOW_NUM_FIT]; // we need a specific struct describing this config for this well fit for GPU

  bead_params * _pHostBeadParams;
  FG_BUFFER_TYPE * _pHostFgBuffer;

  float* _pHostNucRise; // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
  float* _pHostSbg; // FLxF
  float* _pHostEmphasis; // MAX_HPLEN+1 xF // needs precomputation
  float* _pHostNon_integer_penalty; // MAX_HPLEN+1
  float* _pHostDarkMatterComp; // NUMNUC * F  

  // allocated memory
  float* _pDevObservedTrace; //fg
  float* _pDevObservedTraceTranspose; //fg
  float* _pDevNucRise; // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
  float* _pDevSbg; // FLxF
  float* _pDevEmphasis; // MAX_HPLEN+1 xF // needs precomputation
  float* _pDevNon_integer_penalty; // MAX_HPLEN+1
  float* _pDevDarkMatterComp; // NUMNUC * F  
  float* _pDevBeadParams;
  float* _pDevBeadParamsEval;
  float* _pDevBeadParamsTranspose;
  CpuStep_t* _pDevSteps; // we need a specific struct describing this config for this well fit for GPU
  // outputs
  float* _pDevIval; // FLxNxF
  float* _pDevScratch_ival; // FLxNxF
  float* _pDevResidual; // N

  unsigned int * _pDevJTJMatrixMapForDotProductComputation;
  unsigned int * _pDevBeadParamIdxMap; 
  float* _pDevLambdaForBeadFit;
  float* _pDevJTJ;
  float* _pDevRHS;
  float* _pDevLTR;

  // pointers pointing into allocated memory
  float* _pd_partialDerivsOutput;
  float* _pd_delta;
//  float* _pd_beadParamEval; 
  
  WorkSet _myJob;

protected:

  void cleanUp();

public:

  MultiFitStream(GpuMultiFlowFitControl& fitcontrol, WorkerInfoQueue* q);
  ~MultiFitStream();


  // implementatin of stream execution methods
  bool ValidJob();
  int handleResults();
  void ExecuteJob(int * control = NULL);


  static void setBeadsPerBLock(int bpb);

private:

  // fit invariant functions
  void resetPointers();
  void serializeFitInvariantInputs();
  void copyFitInvariantInputsToDevice();
  void copyBeadParamsToDevice();
  void copyBeadParamsToHost();
 
  void executeTransposeToFloat();
  void executeTransposeParams();
  void executeMultiFit(int fit_index); // need to do fit invariant and fit specific
  void executeTransposeParamsBack();

  void copyToHost();
  
  // fit specific functions
  void SetUpLambdaArray(int fit_index);  
  void prepareFitSpecificInputs(int fi_index);
  void serializeFitSpecificInputs(int fit_index);
  void copyFitSpecifcInputsToDevice(int fit_index);

  // some inputs being calculated for fitting
  void CalculateClonalityRestriction(int fit_index);
  void CalculateNonIntegerPenalty();

};

#endif // MULTIFITSTREAM_H
