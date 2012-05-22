/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGMODELREPLAY_H
#define BKGMODELREPLAY_H

#include "RegionTrackerReplay.h"

class BkgModel;

// *********************************************************************
// calculation specific class

class BkgModelReplay {

 public:
  BkgModelReplay(bool makeRegionTracker);

  virtual ~BkgModelReplay();

  virtual void FitTimeVaryingRegion (int flow, double &elapsed_time, Timer &fit_timer);
  void SetRegionTracker(RegionTracker *_rt) { rt = _rt; };
  virtual RegionTracker *GetRegionTracker() { return (rt); };

 public:
  BkgModel *bkg;

 private:
  RegionTracker *rt;

  BkgModelReplay(); // do not use
};

// *********************************************************************
// reader specific class

class BkgModelReplayReader : public BkgModelReplay {

 public:
  BkgModelReplayReader (CommandLineOpts& clo, int regionindex);

  ~BkgModelReplayReader();

  void FitTimeVaryingRegion (int flow, double &elapsed_time, Timer &fit_timer);
  void SetRegionTracker(RegionTrackerReader *_rt) { rrt = _rt; };
  RegionTracker *GetRegionTracker() { return (rrt); };

 private:
  RegionTrackerReader *rrt;  // pointer to my_regions in bkg

  BkgModelReplayReader(); //do not use
};

// *********************************************************************
// recorder specific class

class BkgModelReplayRecorder : public BkgModelReplay {

 public:
  BkgModelReplayRecorder(CommandLineOpts& clo, int regionindex);
  ~BkgModelReplayRecorder();

  void FitTimeVaryingRegion (int flow, double &elapsed_time, Timer &fit_timer);
  void SetRegionTracker(RegionTrackerRecorder *_rt) { rrt = _rt; };
  RegionTracker *GetRegionTracker() { return (rrt); };

 private:
  RegionTrackerRecorder *rrt;  // pointer to my_regions in bkg

  BkgModelReplayRecorder(); //do not use
};


#endif // BKGMODELREPLAY_H
