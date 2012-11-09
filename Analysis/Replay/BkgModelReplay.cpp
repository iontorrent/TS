/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BkgModelReplay.h"
#include "SignalProcessingMasterFitter.h"

// *********************************************************************
// calculation specific class


BkgModelReplay::BkgModelReplay(bool makeRegionTracker)
{
  bkg = NULL;
  rt = NULL;
  if (makeRegionTracker)
    rt = new RegionTracker;
}

BkgModelReplay::~BkgModelReplay() {
  if (rt != NULL ) delete rt;
}

void BkgModelReplay::FitTimeVaryingRegion (int flow, double &elapsed_time, Timer &fit_timer)
{
//  bkg->FitTimeVaryingRegion(elapsed_time, fit_timer);
}

// *********************************************************************
// reader specific functions

BkgModelReplayReader::BkgModelReplayReader(CommandLineOpts& clo, int regionindex)
  : BkgModelReplay(false)
{
    rrt = new RegionTrackerReader(clo, regionindex);
}

BkgModelReplayReader::~BkgModelReplayReader()
{
  if (rrt != NULL ) delete rrt;
}


void BkgModelReplayReader::FitTimeVaryingRegion (int flow, double &elapsed_time, Timer &fit_timer)
{
//  bkg->region_data->AdaptiveEmphasis();
  rrt->Read(flow);
}


// *********************************************************************
// recorder specific functions

BkgModelReplayRecorder::BkgModelReplayRecorder(CommandLineOpts& clo, int regionindex)
  : BkgModelReplay(false)
{
  rrt = new RegionTrackerRecorder(clo, regionindex);
}

BkgModelReplayRecorder::~BkgModelReplayRecorder()
{
  if (rrt != NULL ) delete rrt;
}

void BkgModelReplayRecorder::FitTimeVaryingRegion (int flow, double &elapsed_time, Timer &fit_timer)
{
 // bkg->FitTimeVaryingRegion (elapsed_time, fit_timer);
  rrt->Write(flow);
}

