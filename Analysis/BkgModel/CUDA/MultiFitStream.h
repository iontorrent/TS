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



struct MultiFitData
{
  TMemSegment<CpuStep> Steps; // we need a specific struct describing this config for this well fit for GPU
  TMemSegment<unsigned int> JTJMatrixMapForDotProductComputation;
  TMemSegment<unsigned int> BeadParamIdxMap;
  TMemSegment<float> LambdaForBeadFit;
  MemSegPair hdCopyGroup;
};


class SimpleMultiFitStream : public cudaSimpleStreamExecutionUnit
{
  //Execution config    
  static int _bpb;   // beads per block
  static int _l1type; // 0:sm=l1, 1:sm>l1, 2:sm<l1
  static int _bpbPartialD;   // beads per block
  static int _l1typePartialD; // 0:sm=l1, 1:sm>l1, 2:sm<l1

  WorkSet _myJob;

  float _lambda_start[CUDA_MULTIFLOW_NUM_FIT];
  int _fit_training_level[CUDA_MULTIFLOW_NUM_FIT];
  int _fit_iterations[CUDA_MULTIFLOW_NUM_FIT];
  int _clonal_restriction[CUDA_MULTIFLOW_NUM_FIT];
  float _restrict_clonal[CUDA_MULTIFLOW_NUM_FIT];

  // number of different fits performed in this multifit stream
  int _fitNum; 
  int _curFitLevel;

  int _maxEmphasis;
  unsigned int _partialDerivMask; // mask to find out which params to compute partial derivative for


  size_t _invariantCopyInSize;
  size_t _fitSpecificCopyInSize;


  TMemSegment<ConstParams> _hConstP;

  //host memory
  // fit specific inputs

  TMemSegPair<BeadParams> _hdBeadParams;

  TMemSegPair<FG_BUFFER_TYPE> _hdFgBuffer;  //former _pHostFgBuffer and _pDevObservedTrace
  TMemSegPair<float> _hdCoarseNucRise; // FL x ISIG_SUB_STEPS_MULTI_FLOW x F
  TMemSegPair<float> _hdSbg; // FLxF
  TMemSegPair<float> _hdEmphasis; // MAX_HPLEN+1 xF // needs precomputation
  TMemSegPair<float> _hdNon_integer_penalty; // MAX_HPLEN+1
  TMemSegPair<float> _hdDarkMatterComp; // NUMNUC * F

  MemSegPair  _hdInvariantCopyInGroup;

  MultiFitData _HostDeviceFitData[CUDA_MULTIFLOW_NUM_FIT];
  MultiFitData _DevFitData;





  // allocated memory
  TMemSegment<float> _dFgBufferTransposed; //fg

  TMemSegment<float> _dBeadParamsEval;
  TMemSegment<float> _dBeadParamsTranspose;

  // outputs
  TMemSegment<float> _dIval; // FLxNxF
  TMemSegment<float> _dScratch_ival; // FLxNxF
  TMemSegment<float> _dResidual; // N

  TMemSegment<float> _dJTJ;
  TMemSegment<float> _dRHS;
  TMemSegment<float> _dLTR;

  // pointers pointing into allocated memory
  TMemSegment<float> _dPartialDerivsOutput;
  TMemSegment<float> _dDelta;
  //  float* _pd_beadParamEval;


protected:

  void cleanUp();
  int l1DefaultSettingMultiFit();
  int l1DefaultSettingPartialD();

public:

  SimpleMultiFitStream(streamResources * res, WorkerInfoQueueItem item);
  ~SimpleMultiFitStream();


  // implementatin of stream execution methods
  bool InitJob();
  void ExecuteJob();
  int handleResults();

  void printStatus();
  //////////////

  static void setBeadsPerBlockMultiF(int bpb);
  static void setL1SettingMultiF(int type); // 0:sm=l1, 1:sm>l1, 2:sm<l1
  static void setBeadsPerBlockPartialD(int bpb);
  static void setL1SettingPartialD(int type); // 0:sm=l1, 1:sm>l1, 2:sm<l1

  //change worst case alloc to a different number of frames or beads to reduce over-allocation
  //best values based on empirical data
  //need to be called to guarantee resource preallocation
  static void requestResources(int global_max_flow_key, int global_max_flow_block_size, 
                               float deviceFraction /*= 1.0f*/ );

  static void printSettings();

  static size_t getMaxHostMem(int global_max_flow_key, int global_max_flow_block_size);
                               //if no params given returns absolute or defined worst case size
  static size_t getMaxDeviceMem(int global_max_flow_key, int global_max_flow_block_size, int numFrames /*= 0*/, int numBeads /*= 0*/, WorkSet *job = NULL);

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

  void preFitCpuSteps();
  void postFitProcessing();


};



#endif // MULTIFITSTREAM_H
