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


struct MultiFitData
{
  CpuStep_t* Steps; // we need a specific struct describing this config for this well fit for GPU
  unsigned int * JTJMatrixMapForDotProductComputation;
  unsigned int * BeadParamIdxMap;
  float* LambdaForBeadFit;
};


class SimpleMultiFitStream : public cudaSimpleStreamExecutionUnit
{
  //Execution config    
  static int _bpb;   // beads per block
  static int _l1type; // 0:sm=l1, 1:sm>l1, 2:sm<l1
  static int _bpbPartialD;   // beads per block
  static int _l1typePartialD; // 0:sm=l1, 1:sm>l1, 2:sm<l1



  float _lambda_start[CUDA_MULTIFLOW_NUM_FIT];
  int _fit_iterations[CUDA_MULTIFLOW_NUM_FIT];
  int _clonal_restriction[CUDA_MULTIFLOW_NUM_FIT];
  float _restrict_clonal[CUDA_MULTIFLOW_NUM_FIT];


  int _fitIter; 
  
  int _maxEmphasis;
  unsigned int _partialDerivMask; // mask to find out which params to compute partial derivative for


  size_t _invariantCopyInSize;
  size_t _fitSpecificCopyInSize;


  ConstParams* _HostConstP;
  //Only members starting with _Host or _Dev are 
  //allocated. pointers with _h_p or _d_p are just pointers 
  //referencing memory inside the _Dev/_Host buffers!!!

  //host memory
  // fit specific inputs

  bead_params* _pHostBeadParams;
  FG_BUFFER_TYPE * _pHostFgBuffer;
  MultiFitData _HostFitData[CUDA_MULTIFLOW_NUM_FIT];
  float* _pHostNucRise; // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
  float* _pHostSbg; // FLxF
  float* _pHostEmphasis; // MAX_HPLEN+1 xF // needs precomputation
  float* _pHostNon_integer_penalty; // MAX_HPLEN+1
  float* _pHostDarkMatterComp; // NUMNUC * F  

  // allocated memory
  MultiFitData _DevFitData;
  FG_BUFFER_TYPE* _pDevObservedTrace; //fg
  float* _pDevObservedTraceTranspose; //fg
  float* _pDevNucRise; // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
  float* _pDevSbg; // FLxF
  float* _pDevEmphasis; // MAX_HPLEN+1 xF // needs precomputation
  float* _pDevNon_integer_penalty; // MAX_HPLEN+1
  float* _pDevDarkMatterComp; // NUMNUC * F  
  bead_params* _pDevBeadParams;
  float* _pDevBeadParamsEval;
  float* _pDevBeadParamsTranspose;
  // outputs
  float* _pDevIval; // FLxNxF
  float* _pDevScratch_ival; // FLxNxF
  float* _pDevResidual; // N

  float* _pDevJTJ;
  float* _pDevRHS;
  float* _pDevLTR;

  // pointers pointing into allocated memory
  float* _pd_partialDerivsOutput;
  float* _pd_delta;
//  float* _pd_beadParamEval; 
  

protected:

  void cleanUp();
  int l1DefaultSettingMultiFit();
  int l1DefaultSettingPartialD();

public:

  SimpleMultiFitStream(streamResources * res, WorkerInfoQueueItem item);
  ~SimpleMultiFitStream();


  // implementatin of stream execution methods
  int handleResults();
  void ExecuteJob();
  void printStatus();

  static void setBeadsPerBLockMultiF(int bpb);
  static void setL1SettingMultiF(int type); // 0:sm=l1, 1:sm>l1, 2:sm<l1
  static void setBeadsPerBLockPartialD(int bpb);
  static void setL1SettingPartialD(int type); // 0:sm=l1, 1:sm>l1, 2:sm<l1

  static void printSettings();

  static size_t getMaxHostMem();
  static size_t getMaxDeviceMem(int numFrames = 0, int numBeads = 0);

  int getBeadsPerBlockMultiFit();
  int getL1SettingMultiFit();
  int getBeadsPerBlockPartialD();
  int getL1SettingPartialD();

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
