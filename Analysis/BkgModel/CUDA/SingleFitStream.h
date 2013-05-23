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

#include "ChipIdDecoder.h"


class SimpleSingleFitStream : public cudaSimpleStreamExecutionUnit
{

  //Execution config
  static int _bpb; // beads per block
  static int _l1type;  // 0:sm=l1, 1:sm>l1, 2:sm<l1
  static int _fittype; // 0:gauss newton, 1:lev mar, 2:hybrid
  static int _hybriditer; //num iter after hyubrid switchens from GN to LM

  int _N;
  int _F;

  int _padN; 
  int _copyInSize;
  int _copyOutSize;

  //Only members starting with _Host or _Dev are 
  //allocated. pointers with _h_p or _d_p are just pointers 
  //referencing memory inside the _Dev/_Host buffers!!!

  //host memory
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
  float* _d_pAmpl; // N * NUMFB
  float* _d_pKmult; // N * NUMFB
  float* _d_pDmult; // N
  float* _d_pGain; // N
  float* _d_pTau_Adj; // N
  float* _d_pPCA_Vals; // N * NUM_DM_PCA

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
  float* _d_err; // NxF
  float* _d_fval; // NxF
  float* _d_tmp_fval; // NxF
  float* _d_jac; // 3xNxF 
  float* _d_pMeanErr; // NxNUMFB
  float* _d_avg_trc; // NxF

  // host and device pointer for xtalk
  ConstXtalkParams* _HostConstXtalkP;
  int* _HostNeiIdxMap;
  float* _d_pXtalk;
  float* _d_pNeiContribution;
  int* _d_pNeiIdxMap;
  float* _d_pXtalkScratch;

 //TODO Monitor
 // int * _DevMonitor;   
 // int * _HostMonitor;
 // static int * _IterBuffer;
protected:

  void cleanUp();
  int l1DefaultSetting();
  int BlockSizeDefaultSetting();
public:

  SimpleSingleFitStream(streamResources * res, WorkerInfoQueueItem info);
  ~SimpleSingleFitStream();


  void resetPointers();
  void serializeInputs();
  void preProcessCpuSteps();

  // implementatin of stream execution methods
  void ExecuteJob();
  int handleResults();

  int getBeadsPerBlock();
  int getL1Setting();

  static void setBeadsPerBLock(int bpb);
  static void setL1Setting(int type); // 0:sm=l1, 1:sm>l1, 2:sm<l1
  static void setFitType(int type); // 0:gauss newton, 1:lev mar
  static void setHybridIter(int hybridIter); 

  static void printSettings();


  static size_t getScratchSpaceAllocSize(WorkSet Job);
  static size_t getMaxHostMem();  
  static size_t getMaxDeviceMem(int numFrames = 0, int numBeads = 0);

  //static void printDeleteMonitor();

private:
  void prepareInputs();
  void copyToDevice();
  void executeKernel();
  void copyToHost();
};



#endif // SINGLEFITSTREAM_H
