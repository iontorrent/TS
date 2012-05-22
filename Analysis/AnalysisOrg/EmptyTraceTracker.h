/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef EMPTYTRACETRACKER_H
#define EMPTYTRACETRACKER_H

#include <vector>

#include "EmptyTrace.h"
#include "EmptyTraceReplay.h"
#include "Region.h"
#include "GlobalDefaultsForBkgModel.h"
#include "ImageSpecClass.h"
#include "CommandLineOpts.h"

class EmptyTraceTracker
{
  public:
    EmptyTraceTracker (Region *_regions, RegionTiming *_regiontiming,
                       int totalRegions,
                       std::vector<float> &_sep_t0_est, ImageSpecClass &imgSpec,
		       CommandLineOpts &_clo);
    
    ~EmptyTraceTracker();

    void SetEmptyTracesFromImage (Image &img, PinnedInFlow &pinnedInFlow, int flow, Mask *bfmask);
    void SetEmptyTracesFromImageForRegion(Image &img, PinnedInFlow &pinnedInFlow, int flow, Mask *bfmask, Region& region, float t_mid_nuc);

    EmptyTrace *AllocateEmptyTrace (Region &region, int imgFrames);
    EmptyTrace *GetEmptyTrace (Region &region);

  private:
    EmptyTrace **emptyTracesForBMFitter;
    int numRegions;
    Region *regions;
    RegionTiming *regionTiming;
    std::vector<float> &sep_t0_est;
    int maxNumRegions;
    int imgFrames;

    int MaxNumRegions (Region *regions, int nregions);

    int *unallocated;
    CommandLineOpts& clo;

    // the time compression variables needed here should be parsed out
    GlobalDefaultsForBkgModel global_defaults;

};

#endif // EMPTYTRACETRACKER_H
