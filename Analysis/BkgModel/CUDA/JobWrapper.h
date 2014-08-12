/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef JOBWRAPPER_H
#define JOBWRAPPER_H

#include <iostream>

#include "BkgMagicDefines.h"
#include "ParamStructs.h"
#include "WorkerInfoQueue.h"
#include "BeadTracker.h"
#include "NucStepCache.h"
#include "GpuMultiFlowFitControl.h"
#include "CudaDefines.h"

#define CUDA_MULTIFLOW_NUM_FIT 2 



class BkgModelWorkInfo;


//////////////////////////////////////
//Workset class:
//Leftover class from testing code which wrapped the previously dumped data into a Work set.
//class was kept since it wraps all the accesses to the "private_data" field in the job item
//and knows the # of elements + padding of all the buffers
// therefore no changes to the single fit or multi fit code needs to be made if BgkModel code
//gets refactored.

class WorkSet 
{


  BkgModelWorkInfo * _info;
  GpuMultiFlowFitControl _multiFlowFitControl;
  GpuMultiFlowFitMatrixConfig* _fd[CUDA_MULTIFLOW_NUM_FIT];
  
  NucStep nucRise; 
  int _maxFrames; // only set if we don't want to determine the mem sizes for a specific number of frames or no item is set
  int _maxBeads; // only set if we don't want to determine the mem sizes for a specific number of frames or no item is set
  int _flow_block_size; 
  int _flow_key;
  
public:
  
  WorkSet( int flow_key, int flow_block_size );
  WorkSet(BkgModelWorkInfo * i);
  ~WorkSet(); 

  void setData(BkgModelWorkInfo * info);
  bool isSet() const;

  int getNumBeads()  const;  // will return maxBeads if no _info object is set
  int getNumFrames() const;  // will return max Frames if no _info object is set


  // only used if we don't want to determine the mem sizes for a specific number of frames/beads or no item is set
  void setMaxFrames(int numFrames);
  int getMaxFrames() const;
  void setMaxBeads(int numBeads);
  int getMaxBeads() const;

  int getFlowBlockSize() const;
  void setFlow(int flow_key, int flow_block_size);
  int getFlowKey() const;
  int getAbsoluteFlowNum();

  int getMaxSteps();
  int getMaxParams();

  int getNumSteps(int fit_index);
  int getNumParams(int fit_index);
 

  reg_params * getRegionParams();
  BeadParams * getBeadParams();
  BeadTracker * getBeadTracker();
  bead_state * getBeadState();
  float * getEmphVec(); 
  float * getDarkMatter();
  float * getShiftedBackground();
  int * getFlowIdxMap();
  FG_BUFFER_TYPE * getFgBuffer();
  float * getDeltaFrames();  
  int * getStartNuc();
  float * getCalculateNucRise();
  int * getStartNucCoarse();
  float * getCalculateNucRiseCoarse();
  bound_params * getBeadParamsMax();
  bound_params * getBeadParamsMin();
  float* getClonalCallScale(); 
  bool useDynamicEmphasis();

  CpuStep* getPartialDerivSteps(int fit_index);
  unsigned int* getJTJMatrixMap(int fit_index);
  unsigned int* getBeadParamIdxMap(int fit_index);




  int getFgBufferSize( bool padded = false);
  int getFgBufferSizeShort( bool padded = false);
  int getReusedFgBufferPartialDerivsSize( bool padded = false);
  int getRegionParamsSize( bool padded = false);
  int getBeadParamsSize( bool padded = false);
  int getBeadStateSize( bool padded = false);
  int getEmphVecSize( bool padded = false); 
  int getDarkMatterSize( bool padded = false);
  int getShiftedBackgroundSize( bool padded = false);
  int getFlowIdxMapSize( bool padded = false);
  int getDeltaFramesSize( bool padded = false);  
  int getStartNucSize( bool padded = false);
  int getNucRiseSize( bool padded = false);
  int getStartNucCoarseSize( bool padded = false);
  int getNucRiseCoarseSize( bool padded = false);
  int getBeadParamsMaxSize( bool padded = false);
  int getBeadParamsMinSize( bool padded = false);
  int getClonalCallScaleSize( bool padded = false); 

  int getPartialDerivStepsSize(int fit_index, bool padded = false);
  int getJTJMatrixMapSize(int fit_index, bool padded = false);
  int getBeadParamIdxMapSize(int fit_index, bool padded = false);
  int getParamMatrixSize(int fit_index, bool padded = false);
  int getParamRHSSize(int fit_index, bool padded = false);

  int getPartialDerivStepsMaxSize(bool padded = false);
  int getJTJMatrixMapMaxSize(bool padded = false);
  int getBeadParamIdxMapMaxSize(bool padded = false);
  int getParamMatrixMaxSize(bool padded = false);
  int getParamRHSMaxSize(bool padded= false);


  int getFlxFxB(bool padded = false); //Flows x Frames x Beads x sizeof(float)
  int getFxB(bool padded = false); // Frames x Beads x sizeof(float)
  int getFlxB(bool padded = false); // Flows x Beads x sizeof(float)
  int getFloatPerBead(bool padded = false);

  int getPaddedN() const;
  int padTo128Bytes(int size);
  int multipltbyPaddedN(int size);


//
  float getAmpLowLimit();
  float getkmultLowLimit();
  float getkmultHighLimit();
  float getkmultAdj();
  bool fitkmultAlways();
  float getClonalCallPenalty(); 
  float getMaxEmphasis();
  bool performAlternatingFit();
  
  void setUpFineEmphasisVectors();
//  BkgModelWorkInfo * getInfo();

//  static void setMaxBeads(int n);
//  static int getMaxBeads();



// CPU side Work
  void KeyNormalize();
  void PerformePCA();


  bool ValidJob();

  void setJobToPostFitStep();
  void setJobToRemainRegionFit();

  void putJobToCPU(WorkerInfoQueueItem item);
  void putJobToGPU(WorkerInfoQueueItem item);

  void printJobSummary();
  
  int getXtalkNeiIdxMapSize(bool padded);
  int getNumXtalkNeighbours();
  const int* getNeiIdxMapForXtalk();
  int* getXtalkNeiXCoords();
  int* getXtalkNeiYCoords();
  float* getXtalkNeiMultiplier();
  float* getXtalkNeiTauTop();
  float* getXtalkNeiTauFluid();
  bool performCrossTalkCorrection() const;

  void calculateCPUXtalkForBead(int ibd, float* buf);

  float* getFrameNumber();
  int getFrameNumberSize(bool padded = false);
  bool performExpTailFitting();
  bool performCalcPCADarkMatter();
  bool useDarkMatterPCA();
  bool InitializeAmplitude();
  
  // recompress raw traces using standard timing compression 
  // for single flow fitting
  int GetNumUnCompressedFrames();
  int GetNumStdCompressedFrames();
  int GetNumETFCompressedFrames();
  int* GetStdFramesPerPoint();
  int* GetETFInterpolationFrames();
  float* GetETFInterpolationMul();
  int GetETFStartFrame();
  int GetStdFramesPerPointSize(bool padded = false);
  int GetETFInterpolationMulSize(bool padded = false);
  int GetETFInterpolationFrameSize(bool padded = false); 
  float* GetStdTimeCompEmphasis();
  int GetStdTimeCompEmphasisSize(bool padded = false);
  float* GetStdTimeCompNucRise();
  int GetStdTimeCompNucRiseSize(bool padded = false);
  float* GetStdTimeCompDeltaFrame();
  int GetStdTimeCompDeltaFrameSize(bool padded = false);
  float* GetStdTimeCompFrameNumber();
  int GetStdTimeCompFrameNumberSize(bool padded = false);
  bool performRecompressionTailRawTrace();
  void setUpFineEmphasisVectorsForStdCompression();
  int* GetNonZeroEmphasisFrames();
  int* GetNonZeroEmphasisFramesForStdCompression();
  int GetNonZeroEmphasisFramesVecSize(bool padded = false);
};




#endif //JOBWRAPPER_H


