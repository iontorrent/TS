/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef GPUCONTROLOPTS_H
#define GPUCONTROLOPTS_H

#include "stdlib.h"
#include "stdio.h"
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <cstring>
#include <string>
#include <algorithm>
#include "OptBase.h"

// What does the bkg-model section of the software need to know?
class GpuControlOpts{
  public:
    // commandline options for GPU for background model computation
    float gpuWorkLoad;
    int gpuNumStreams;

    int gpuMultiFlowFit;
    int gpuThreadsPerBlockMultiFit;
    int gpuL1ConfigMultiFit;
    int gpuThreadsPerBlockPartialD;
    int gpuL1ConfigPartialD;

    int gpuSingleFlowFit;
    int gpuThreadsPerBlockSingleFit;
    int gpuL1ConfigSingleFit;
    int gpuSingleFlowFitType;
    int gpuHybridIterations;

    int doGpuOnlyFitting;

    int gpuAmpGuess; 

    bool gpuVerbose;

    // hold the device ids for the gpus to be used for computation
    std::vector<int> gpuDeviceIds;
    void DefaultGpuControl(void);
	void PrintHelp();
	void SetOpts(OptArgs &opts, Json::Value& json_params);
};

#endif // GPUCONTROLOPTS_H
