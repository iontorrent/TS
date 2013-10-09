/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <assert.h>
#include <iostream>
#include <sstream>
#include "GpuControlOpts.h"


void GpuControlOpts::DefaultGpuControl()
{

    gpuWorkLoad = 1.0;

    gpuMultiFlowFit = 1;
    gpuThreadsPerBlockMultiFit = 128;
    gpuL1ConfigMultiFit = -1;  // actual default is set hardware specific in MultiFitStream.cu
    gpuThreadsPerBlockPartialD = 128;
    gpuL1ConfigPartialD = -1;  // actual default is set hardware specific in MultiFitStream.cu

    gpuSingleFlowFit = 1;
    gpuThreadsPerBlockSingleFit = -1;
    gpuL1ConfigSingleFit = -1; // actual default is set hardware specific in SingleFitStream.cu

    // 0: GaussNewton, 1: LevMar 2:Hybrid (gpuHybridIterations Gauss Newton, then rest LevMar)
    // 3: Relaxing Kmult (two pass Gauss Newton)
    gpuSingleFlowFitType = 0; 
    gpuHybridIterations = 3;

    doGpuOnlyFitting = 1;

    gpuAmpGuess = 1;

    gpuVerbose = false;
}

