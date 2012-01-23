/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef CUDAWRAPPER_H 
#define CUDAWRAPPER_H 

#include "WorkerInfoQueue.h"
#include "BkgModel/MathOptim.h"

#ifdef ION_COMPILE_CUDA
#include "BkgModelCuda.h"
#endif

//
// Struct used to pack GPU parameters when BkgWorker is created
//

struct BkgFitWorkerGpuInfo
{
    bool dynamic_gpu_load;
    int gpu_index;
    void* queue;
};

// defined in Analysis.cpp
extern void *BkgFitWorker(void *arg, bool use_gpu);
extern void *DynamicBkgFitWorker(void *arg, bool use_gpu);

bool configureGpu(bool use_gpu_acceleration, std::vector<int> &valid_devices, int use_all_gpus, 
  int numGpuThreads, int &numBkgWorkers_gpu);
  
void* BkgFitWorkerGpu(void *arg);  
void InitConstantMemoryOnGpu(PoissonCDFApproxMemo& poiss_cache);

#endif // CUDAWRAPPER_H
