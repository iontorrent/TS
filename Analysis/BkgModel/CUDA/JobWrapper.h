/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef JOBWRAPPER_H
#define JOBWRAPPER_H

#include <iostream>

#include "BkgMagicDefines.h"
#include "ParamStructs.h"
#include "WorkerInfoQueue.h"
#include "BeadTracker.h"

// workset (benchmark workset, can be wrapped into a job)

class BkgModelWorkInfo;


class WorkSet 
{


  BkgModelWorkInfo * _info;

public:

  
  WorkSet();
  ~WorkSet(); 

  void setData(BkgModelWorkInfo * info);
  bool isSet();

  int getNumBeads();  // will return maxBeads if no _info object is set
  int getNumFrames();  // will return max Frames if no _info object is set
  int getAbsoluteFlowNum();
    
  reg_params * getRegionParams();
  bead_params * getBeadParams();
  BeadTracker * getBeadTracker();
  bead_state * getState();
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


  int getFgBufferSize( bool padded = false);
  int getFgBufferSizeShort( bool padded = false);
  int getRegionParamsSize( bool padded = false);
  int getBeadParamsSize( bool padded = false);
  int getStateSize( bool padded = false);
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

  int getFlxFxB(bool padded = false); //Flows x Frames x Beads x sizeof(float)
  int getFxB(bool padded = false); // Frames x Beads x sizeof(float)
  int getFlxB(bool padded = false); // Flows x Beads x sizeof(float)

  int getPaddedN();
  int padTo128Bytes(int size);
  int multipltbyPaddedN(int size);


//
  float getAmpLowLimit();
  float getkmultLowLimit();
  float getkmultHighLimit();
  float getClonalCallPenalty(); 
  float getMaxEmphasis();
  bool performAlternatingFit();
  
  void setUpFineEmphasisVectors();
//  BkgModelWorkInfo * getInfo();

  static void setMaxBeads(int n);
  static int getMaxBeads();


  bool ValidJob();
  void KeyNormalize();

  void putPostFitStep(WorkerInfoQueueItem item);
  void putRemainRegionFit(WorkerInfoQueueItem item);
  

};




#endif //JOBWRAPPER_H


