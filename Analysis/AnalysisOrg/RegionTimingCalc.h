/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REGIONTIMINGCALC_H
#define REGIONTIMINGCALC_H

// #include "ImageLoader.h"
#include "Region.h"
#include "WorkerInfoQueue.h"
#include "AvgKeyIncorporation.h"

struct GaussianExponentialParams;



struct TimingFitWorkOrder{
  int type;
  Region *regions;
  RegionTiming *region_time;
  AvgKeyIncorporation *kic;
   int r;
 };
 void FindStartingParametersForBkgModel(GaussianExponentialParams &my_initial_params, float &my_fit_residual, float *avg_sig, int avg_len);
 void FillOneRegionTimingParameters(RegionTiming *region_time, Region *regions, int r, AvgKeyIncorporation *kic);
 void FillRegionalTimingParameters(RegionTiming *region_time, Region *regions, int numRegions, AvgKeyIncorporation *kic);
 extern void *TimingFitWorker(void *arg);
 void threadedFillRegionalTimingParameters(RegionTiming *region_time, Region *regions, int numRegions, AvgKeyIncorporation *kic);
 

#endif // REGIONTIMINGCALC_H
