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
  static int _hybriditer; //num iter after hybrid switchens from GN to LM

  WorkSet _myJob;

  int _N;
  int _F;

  int _padN; 


  //MemorySegmentPairs In and Output data pointers for async copy
  TMemSegPair<FG_BUFFER_TYPE> _hdFgBuffer; //FG_BUFFER_TYPE

  TMemSegPair<BeadParams> _hdBeadParams; //BeadParams*
  TMemSegPair<bead_state> _hdBeadState;  //bead_state*

  TMemSegPair<float> _hdDarkMatter; //float*
  TMemSegPair<float> _hdDarkEmphVector; //float*
  TMemSegPair<float> _hdShiftedBkg; //float*
  TMemSegPair<float> _hdEmphVector; //float*
  TMemSegPair<float> _hdNucRise; //float*
  TMemSegPair<float> _hdStdTimeCompEmphVec; //float*
  TMemSegPair<float> _hdStdTimeCompNucRise; //float*

  //MemorySegmentPairGroups to wrap copy into single call
  MemSegPair _hdCopyOutGroup;
  MemSegPair _hdCopyInGroup;

  //Host Only Buffers for Async Constant init
  TMemSegment<ConstParams> _hConstP;  //ConstParams*


  //Device Only Buffers
  TMemSegment<float> _dFgBufferFloat; //float*
  TMemSegment<float> _dWorkBase; //scartch space, float*
  TMemSegment<float> _dBeadParamTransp; //float*/int*

  // device pointers to transposed 
  // bead params
  TMemSegment<float> _dR; //N
  TMemSegment<float> _dCopies; //N
  TMemSegment<float> _dAmpl; // N * flow_block_size
  TMemSegment<float> _dKmult; // N * flow_block_size
  TMemSegment<float> _dDmult; // N
  TMemSegment<float> _dGain; // N
  TMemSegment<float> _dTau_Adj; // N
  TMemSegment<float> _dPCA_Vals; // N * NUM_DM_PCA
  TMemSegment<float> _dPhi; // N

  //device scratch space pointers
  TMemSegment<float> _derr; // NxF
  TMemSegment<float> _dfval; // NxF
  TMemSegment<float> _dtmp_fval; // NxF
  TMemSegment<float> _djac; // 3xNxF
  TMemSegment<float> _dMeanErr; // Nxflow_block_size
  TMemSegment<float> _davg_trc; // NxF

  //host xtalk pointers
  TMemSegment<ConstXtalkParams> _hConstXtalkP; //ConstXtalkParams*
  TMemSegment<int> _hNeiIdxMap; //int*
  TMemSegment<int> _hSampleNeiIdxMap; //int*

  //device Xtalk pointers
  TMemSegment<float> _dNeiContribution; //float*
  TMemSegment<float> _dXtalk; //float*
  TMemSegment<int> _dNeiIdxMap; //int*
  TMemSegment<int> _dSampleNeiIdxMap; //int*
  TMemSegment<float> _dGenericXtalk; //float*
  TMemSegment<float> _dXtalkScratch; //float*

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

  // implementation of interface stream execution methods
  bool InitJob();
  void ExecuteJob();
  int handleResults();

  void printStatus();
  //////////////

  int getBeadsPerBlock();
  int getL1Setting();


  //need to be called to guarantee resource preallocation
  static void requestResources(int flow_key, int flow_block_size, float deviceFraction );


  static void setBeadsPerBlock(int bpb);
  static void setL1Setting(int type); // 0:sm=l1, 1:sm>l1, 2:sm<l1
  static void setFitType(int type); // 0:gauss newton, 1:lev mar
  static void setHybridIter(int hybridIter);

  static void printSettings();

  static size_t getScratchSpaceAllocSize(const WorkSet & Job);
  static size_t getMaxHostMem(int flow_key, int flow_block_size);
  //if no params given returns absolute or defined worst case size
  static size_t getMaxDeviceMem(int flow_key, int flow_block_size, int numFrames/*=0*/, int numBeads/*=0*/);


private:
  void prepareInputs();
  void copyToDevice();
  void executeKernel();
  void copyToHost();
};



#endif // SINGLEFITSTREAM_H
