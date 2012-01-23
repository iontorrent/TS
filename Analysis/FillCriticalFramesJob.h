/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FILLCRITICALFRAMESJOB_H
#define FILLCRITICALFRAMESJOB_H
#include <vector>
#include <string>
#include <stdio.h>
#include "PJob.h"
#include "Traces.h"
#include "Mask.h"
#include "Image.h"
#include "ReportSet.h"

class FillCriticalFramesJob : public PJob {

 public:
  
  FillCriticalFramesJob() {
    mTrace = NULL;
    mRegionXSize = mRegionYSize = 0;
  }
  
  void Init(Traces *trace, int regionXSize, int regionYSize) {
    mTrace = trace;
    mRegionXSize = regionXSize;
    mRegionYSize = regionYSize;
  }

  /** Process work. */
  virtual void Run() {
    mTrace->FillCriticalFrames();
    mTrace->CalcReference(mRegionXSize,mRegionYSize,mTrace->mGridMedian);
  }

  /** Cleanup any resources. */
  virtual void TearDown() {}

  /** Exit this pthread (killing thread) */
  void Exit() {
    pthread_exit(NULL);
  }

 private:
  Traces *mTrace;
  int mRegionXSize, mRegionYSize;
};


#endif // FILLCRITICALFRAMESJOB_H
