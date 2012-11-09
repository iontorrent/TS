/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef SINGLEFITSTREAM_H
#define SINGLEFITSTREAM_H

// std headers
#include <iostream>
// cuda
#include "cuda_runtime.h"
#include "cuda_error.h"
#include "Utils.h"

//#include "PoissonCdf.h"
#include "StreamManager.h"
#include "ParamStructs.h"
#include "JobWrapper.h"



class SingleFitStream : public cudaStreamExecutionUnit
{

  int _N;
  int _F;

  static int _bpb;
  static int _cntSingleFitStream[MAX_CUDA_DEVICES];

  int _devId;

  int _BeadBaseAllocSize;
  int _FgBufferAllocSizeHost;
  int _FgBufferAllocSizeDevice;
 
  int _padN; 
  int _copyInSize;
  int _copyOutSize;

  //Only members starting with _Host or _Dev are 
  //allocated. pointers with _h_p or _d_p are just pointers 
  //referencing memory inside the _Dev/_Host buffers!!!

  //host memory
  char * _HostBeadBase;
  ConstParams* _HostConstP;
  FG_BUFFER_TYPE * _HostFgBuffer;

  //host pointers
  bead_params* _h_pBeadParams; // N 
  bead_state* _h_pBeadState; // N
  float* _h_pDarkMatter; // NUMNUC*F
  float* _h_pEmphVector; // (MAX_HPLEN+1) *F
  float* _h_pShiftedBkg; //NUMFB*F  
	float* _h_pNucRise; // ISIG_SUB_STEPS_SINGLE_FLOW * F * NUMFB

  //device memory
  
  float* _DevWeightFgBuffer; // NxF*NUMFB
  FG_BUFFER_TYPE * _d_pFgBuffer;
  float * _DevFgBufferFloat;
  char * _DevBeadBase;
  float * _DevWorkBase;
  float * _DevBeadParamTransp;
  float * _DevBeadStateTransp;

  //device pointers
  bead_params* _d_pBeadParams; // N 
  bead_state* _d_pBeadState; // N
  float* _d_pDarkMatter; // NUMNUC*F
  float* _d_pEmphVector; // (MAX_HPLEN+1) *F
  float* _d_pShiftedBkg; //NUMFB*F  
	float* _d_pNucRise; // ISIG_SUB_STEPS_SINGLE_FLOW * F * NUMFB


  // device pointers to transposed 
  // bead params
  float* _d_pR; //N
  float* _d_pCopies; //N
  float* _d_pAmpl; // N
  float* _d_pKmult; // N
  float* _d_pDmult; // N
  float* _d_pGain; // N

  //device pointers to transposed
  //bead state 
  bool *bad_read;     // bad key, infinite signal,
  bool *clonal_read;  // am I a high quality read that can be used for some parameter fitting purposes?
  bool *corrupt;      // has this bead been lost for whatever reason
  bool *pinned;       // this bead got pinned in some flow
  bool *random_samp;  // process this bead no matter how the above flags are set
  // track classification entities across blocks of flows
  float *key_norm;
  float *ppf;
  float *ssq;
  // track cumulative average error looking to detect corruption
  float *avg_err;

  //device scratch space pointers
//  float* _d_pTauB; // FNUM* N
//  float* _d_pSP; // N*NUMFB
//  float* _d_pwtScale; //N 
   float* _d_err; // NxF
  float* _d_fval; // NxF
  float* _d_tmp_fval; // NxF
  float* _d_jac; // NxF 
  float* _d_pMeanErr; // N*NUMFB

  WorkSet _myJob;


//  int * _DevMonitor;   
//  int * _HostMonitor;
protected:

  void cleanUp();

public:

  SingleFitStream(WorkerInfoQueue * Q);
  ~SingleFitStream();


  void resetPointers();
  void serializeInputs();
  void preProcessCpuSteps();

  // implementatin of stream execution methods
  bool ValidJob();
  void ExecuteJob(int * control = NULL);
  int handleResults();

  static void setBeadsPerBLock(int bpb);

private:
  void prepareInputs();
  void copyToDevice();
  void executeKernel();
  void copyToHost();

};

#endif // SINGLEFITSTREAM_H
