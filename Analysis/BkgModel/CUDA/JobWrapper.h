/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef JOBWRAPPER_H
#define JOBWRAPPER_H

#include <iostream>

#include "BkgMagicDefines.h"
#include "ParamStructs.h"
#include "WorkerInfoQueue.h"
#include "BeadTracker.h"
#include "GpuMultiFlowFitControl.h"
#include "CudaDefines.h"

#define CUDA_MULTIFLOW_NUM_FIT 2 

// workset (benchmark workset, can be wrapped into a job)

class BkgModelWorkInfo;


class WorkSet 
{


  BkgModelWorkInfo * _info;

  GpuMultiFlowFitMatrixConfig* _fd[CUDA_MULTIFLOW_NUM_FIT];
  
  int _maxFrames; // only set if we don't want to determine the mem sizes for a specific number of frames or no item is set
  int _maxBeads; // only set if we don't want to determine the mem sizes for a specific number of frames or no item is set

  
public:

  
  WorkSet();
  WorkSet(BkgModelWorkInfo * i);
  ~WorkSet(); 

  void setData(BkgModelWorkInfo * info);
  bool isSet();

  int getNumBeads();  // will return maxBeads if no _info object is set
  int getNumFrames();  // will return max Frames if no _info object is set


  // only used if we don't want to determine the mem sizes for a specific number of frames/beads or no item is set
  void setMaxFrames(int numFrames);
  int getMaxFrames();
  void setMaxBeads(int numBeads);
  int getMaxBeads();

  

  int getAbsoluteFlowNum();

  int getMaxSteps();
  int getMaxParams();

  int getNumSteps(int fit_index);
  int getNumParams(int fit_index);
 

 
  reg_params * getRegionParams();
  bead_params * getBeadParams();
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

  CpuStep_t* getPartialDerivSteps(int fit_index);
  unsigned int* getJTJMatrixMap(int fit_index);
  unsigned int* getBeadParamIdxMap(int fit_index);




  int getFgBufferSize( bool padded = false);
  int getFgBufferSizeShort( bool padded = false);
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

  int getPaddedN();
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
  bool performCrossTalkCorrection();

  void calculateCPUXtalkForBead(int ibd, float* buf);

  float* getFrameNumber();
  int getFrameNumberSize(bool padded = false);
  bool performExpTailFitting();
  bool performCalcPCADarkMatter();
  bool useDarkMatterPCA();
  bool InitializeAmplitude();

};




#endif //JOBWRAPPER_H


