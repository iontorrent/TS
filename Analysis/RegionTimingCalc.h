/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REGIONTIMINGCALC_H
#define REGIONTIMINGCALC_H

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <libgen.h>
#include <limits.h>
#include <signal.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <armadillo>

#include "cudaWrapper.h"
#include "Flow.h"
#include "Image.h"
#include "Region.h"
#include "Mask.h"
#include "Separator.h"
#include "BkgModel.h"
#include "GaussianExponentialFit.h"
#include "WorkerInfoQueue.h"
#include "Stats.h"
#include "SampleStats.h"
#include "ReservoirSample.h"
#include "SampleQuantiles.h"
#include "CommandLineOpts.h"
#include "ImageSpecClass.h"
#include "ImageLoader.h"
#include "WellFileManipulation.h"
#include "DifferentialSeparator.h"
#include "TrackProgress.h"
#include "SpecialDataTypes.h"

struct RegionTiming{
  float t_mid_nuc;
  float t_sigma;
};

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