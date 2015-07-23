/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef CUDAWRAPPER_H 
#define CUDAWRAPPER_H 

#include "WorkerInfoQueue.h"
#include "BkgModel/MathModel/MathOptim.h"
#include "GpuControlOpts.h"
#include "SignalProcessingFitterQueue.h"


bool configureGpu(bool use_gpu_acceleration, std::vector<int> &valid_devices, int use_all_gpus,
  int &numBkgWorkers_gpu);

void configureKernelExecution(GpuControlOpts opts, int global_max_flow_key, 
                                                   int global_max_flow_max);

void* BkgFitWorkerGpu(void* arg); 
void InitConstantMemoryOnGpu(int device, PoissonCDFApproxMemo& poiss_cache);

void SimpleFitStreamExecutionOnGpu(WorkerInfoQueue* q, WorkerInfoQueue* errorQueue);
bool TryToAddSingleFitStream(void * vpsM, WorkerInfoQueue* q);

bool ProcessProtonBlockImageOnGPU(
    BkgModelWorkInfo* fitterInfo, 
    int flowBlockSize,
    int deviceId);

void* flowByFlowHandshakeWorker(void *arg);


#endif // CUDAWRAPPER_H
