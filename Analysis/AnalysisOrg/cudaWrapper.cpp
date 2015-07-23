/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <algorithm>
#include "cudaWrapper.h"
#include "SignalProcessingFitterQueue.h"

#ifdef ION_COMPILE_CUDA
#include "SingleFitStream.h"
#include "MultiFitStream.h"
#include "StreamingKernels.h"
#include "JobWrapper.h"
#include "LayoutTester.h"
#include "BkgGpuPipeline.h"
#endif

#include "RawWellsWriteJob.h"
#include "PJob.h"
#include "PJobExit.h"

#define DEBUG_RAWWELLS_THREAD 0

static void* cpuWorkerThread(void *arg)
{
  WorkerInfoQueue *wq = static_cast<WorkerInfoQueue*>(arg);

  while(true) {
    WorkerInfoQueueItem item = wq->GetItem();

    PJob *job = static_cast<PJob*>(item.private_data);

    if (job->IsEnd()) {
      //job->Exit();
      wq->DecrementDone();
      break;
    }  
    else {
      job->Run();
    }
    wq->DecrementDone();
  }

  return NULL;
}


void* BkgFitWorkerGpu(void *arg)
{
#ifdef ION_COMPILE_CUDA
  // Unpack GPU worker info
  BkgFitWorkerGpuInfo* info = static_cast<BkgFitWorkerGpuInfo*>(arg);

  // Wrapper to create a GPU worker and set GPU
  printf("GPU_INDEX %d\n", info->gpu_index);
  cudaSetDevice( info->gpu_index );
  cudaError_t err = cudaGetLastError();

  // See if device was set OK
  if ( err == cudaSuccess )
  {
    cudaDeviceProp cuda_props;
    int dev_id;
    cudaGetDevice( &dev_id );
    cudaGetDeviceProperties( &cuda_props, dev_id );

    printf( "CUDA %d: Created GPU BkgModel worker...  (%d:%s v%d.%d)\n",
        dev_id, dev_id, cuda_props.name, cuda_props.major, cuda_props.minor );

    SimpleFitStreamExecutionOnGpu(static_cast<WorkerInfoQueue*>(info->queue),
        static_cast<WorkerInfoQueue*>(info->fallbackQueue) );

    return NULL;
  }

  printf( "CUDA: Failed to initialize GPU worker... (%d: %s)\n",
      info->gpu_index, cudaGetErrorString(err) );

  // Note: could fall back to a CPU worker here by setting GPU flag to false,
  //       though I'm not sure if that's always appropriate

  return NULL;
#else 
  return NULL;
#endif
}


bool configureGpu(bool use_gpu_acceleration, std::vector<int> &valid_devices, int use_all_gpus,
    int &numBkgWorkers_gpu) {
#ifdef ION_COMPILE_CUDA
  const unsigned long long gpu_mem = 0.5 * 1024 * 1024 * 1024;

  if (!use_gpu_acceleration)
    return false;

  // Get number of GPUs in system
  int num_gpus = 0;
  cudaError_t err = cudaGetDeviceCount( &num_gpus );

  const cudaComputeVersion minRequired(2,0);
  cudaComputeVersion maxFound(0,0);
  cudaComputeVersion minFound(9,9);

  if (err != cudaSuccess) {
    printf("CUDA: No GPU device available. Defaulting to CPU only computation (return code %d: %s) &\n", err , cudaGetErrorString(err));
    return false;
  }

  if (valid_devices.size() == 0) {

    cudaDeviceProp dev_props;

    // Iterate over GPUs to find the highest compute device
    for ( int dev = 0; dev < num_gpus;  dev++ )
    {
      cudaError_t err= cudaGetDeviceProperties( &dev_props, dev );
      if(err == cudaSuccess){
        cudaComputeVersion found(dev_props.major,dev_props.minor);
        if(found >= minRequired){
          maxFound = (found>maxFound)?(found):(maxFound);
          minFound = (found<minFound)?(found):(minFound);
          printf( "CUDA possible compute device found with Id: %d model: %s compute version: %d.%d\n", dev, dev_props.name, dev_props.major, dev_props.minor );
        }
      }
    }
    //build actual valid device list
    for ( int dev = 0; dev < num_gpus;  dev++ )
    {

      cudaError_t err= cudaGetDeviceProperties( &dev_props, dev );
      if(err == cudaSuccess){
        cudaComputeVersion found(dev_props.major,dev_props.minor);

        //if device has min compute and min memory check for other constraints
        if(found >= minRequired && dev_props.totalGlobalMem > gpu_mem){
          if(!use_all_gpus)
          {//use only cards with max compute
           if(found == maxFound)
             if(valid_devices.size() == 0)
               printf( "CUDA only devices with the max compute version found: %d.%d will be used as compute devices\n", maxFound.getMajor(), maxFound.getMinor());
             valid_devices.push_back(dev);
          }
          else
          { //use all gpus that pass the minimum requirements
            if(valid_devices.size() == 0)
              printf( "CUDA all devices with compute version >= %d.%d and at least %lluMB memory will be used as compute devices\n", minRequired.getMajor(), minRequired.getMinor(), gpu_mem/(1024*1024));
            valid_devices.push_back(dev);
          }
        }
      }
    }

  }else{
    printf("CUDA Device list provided:\n");
    cudaDeviceProp dev_props;
    for ( size_t i = 0; i < valid_devices.size();  i++ ){
      cudaError_t err = cudaGetDeviceProperties( &dev_props, valid_devices[i]);
      if(err == cudaSuccess){
        printf( "CUDA %d: model: %s compute version: %d.%d\n", valid_devices[i], dev_props.name, dev_props.major, dev_props.minor );
      }else{
        printf( "CUDA ERROR: No CUDA compatible device found with id %d!\n", valid_devices[i] );
        exit(-1);
      }
    }

  }
  //what was this for??
  /*else {
    while (valid_devices.size() > 0) {
      if (valid_devices.back() > (num_gpus - 1)) {
        valid_devices.pop_back();
      }
      else {
        break;
      }
    }
  }*/

  // Set the number of GPU workers and tell CUDA about our list of valid devices
  if (valid_devices.size() > 0) {
    numBkgWorkers_gpu = int(valid_devices.size());
    cudaSetValidDevices( &valid_devices[0], int( valid_devices.size() ) );
  }
  else {
    printf("CUDA: No GPU device available. Defaulting to CPU only computation (return code %d: %s) &\n", err , cudaGetErrorString(err));
    return false;   
  }


  PoissonCDFApproxMemo poiss_cache; 
  poiss_cache.Allocate (MAX_POISSON_TABLE_COL,MAX_POISSON_TABLE_ROW,POISSON_TABLE_STEP);
  poiss_cache.GenerateValues(); // fill out my table


  for(int i=valid_devices.size()-1 ; i >= 0; i--){
    try{
      cout << "CUDA "<< valid_devices[i] << ": Creating Context and Constant memory on device with id: "<<  valid_devices[i]<< endl;
      InitConstantMemoryOnGpu(valid_devices[i],poiss_cache);
    }
    catch(cudaException &e) {
      cout << "CUDA "<< valid_devices[i] << ": Context could not be created. removing device with id: "<<  valid_devices[i] << " from valid device list" << endl;
      valid_devices.erase (valid_devices.begin()+i);
      numBkgWorkers_gpu -= 1;
      if(numBkgWorkers_gpu == 0) cout << "CUDA: no context could be created, defaulting to CPU only execution" << endl; 
    }

  }

  if(numBkgWorkers_gpu == 0) return false;

  return true;

#else

  return false;

#endif

}

void InitConstantMemoryOnGpu(int device, PoissonCDFApproxMemo& poiss_cache) {
#ifdef ION_COMPILE_CUDA
  StreamingKernels::initPoissonTablesLUT(device, (void**) poiss_cache.poissLUT);

#endif
}

void configureKernelExecution(
    GpuControlOpts opts, 
    int global_max_flow_key, 
    int global_max_flow_max
)
{
#ifdef ION_COMPILE_CUDA
  if(opts.gpuMultiFlowFit)
  {
    SimpleMultiFitStream::setBeadsPerBlockMultiF(opts.gpuThreadsPerBlockMultiFit);
    SimpleMultiFitStream::setL1SettingMultiF(opts.gpuL1ConfigMultiFit);
    SimpleMultiFitStream::setBeadsPerBlockPartialD(opts.gpuThreadsPerBlockPartialD);
    SimpleMultiFitStream::setL1SettingPartialD(opts.gpuL1ConfigPartialD);
    SimpleMultiFitStream::requestResources(global_max_flow_key, global_max_flow_max, 1.0f);  //0.80f
    SimpleMultiFitStream::printSettings();
  }

  // configure SingleFlowFit Execution
  if(opts.gpuSingleFlowFit)
  {
    SimpleSingleFitStream::setBeadsPerBlock(opts.gpuThreadsPerBlockSingleFit);
    SimpleSingleFitStream::setL1Setting(opts.gpuL1ConfigSingleFit);
    SimpleSingleFitStream::setFitType(opts.gpuSingleFlowFitType);
    SimpleSingleFitStream::requestResources(global_max_flow_key, global_max_flow_max, 1.0f); //0.74f
    //SimpleSingleFitStream::setHybridIter(opts.gpuHybridIterations);
    SimpleSingleFitStream::printSettings();
  }

  cudaSimpleStreamManager::setNumMaxStreams(opts.gpuNumStreams);
  cudaSimpleStreamManager::setVerbose(opts.gpuVerbose);
  cudaSimpleStreamExecutionUnit::setVerbose(opts.gpuVerbose);


#endif 
}


void SimpleFitStreamExecutionOnGpu(WorkerInfoQueue* q, WorkerInfoQueue * errorQueue )
{
#ifdef ION_COMPILE_CUDA
  int dev_id;

  cudaGetDevice( &dev_id );
  std::cout << "CUDA " << dev_id << ": Creating GPU StreamManager" << std::endl;

  cudaSimpleStreamManager  sM( q, errorQueue );

  sM.DoWork();

  std::cout << "CUDA " << dev_id << ": Destroying GPU StreamManager" << std::endl;

#endif
}

bool ProcessProtonBlockImageOnGPU(
    BkgModelWorkInfo* fitterInfo,
    int flowBlockSize,
    int deviceId)
{
#ifdef ION_COMPILE_CUDA
  return blockLevelSignalProcessing(fitterInfo, flowBlockSize, deviceId);
#endif
  return false;
}

void* flowByFlowHandshakeWorker(void *arg)
{
#if DEBUG_RAWWELLS_THREAD
  std::cout << "====> Started GPU-CPU handshake worker" << std::endl;
#endif
  GPUFlowByFlowPipelineInfo * info = static_cast<GPUFlowByFlowPipelineInfo*>(arg);


  //create job handles for raw wells writer threads
  size_t numFitters = (*(info->fitters)).size();
  std::vector<RawWellsWriteJob*> writeJobs;
  writeJobs.resize(numFitters);
  for (size_t i=0; i<numFitters; ++i) {
    writeJobs[i] = new RawWellsWriteJob((*(info->fitters))[i]);
  }

  // create work queue and worker threads
  size_t numRawWellWriterBkgThreads = 6;
  WorkerInfoQueue jobQueue(numRawWellWriterBkgThreads*numFitters);

  pthread_t worker;
  for (size_t i=0; i<numRawWellWriterBkgThreads; ++i) {
    pthread_create(&worker, NULL, cpuWorkerThread, &jobQueue);
    pthread_detach(worker);
  }

  int curFlow = info->startingFlow;
  bool done = false;
  while (!done) {

#if DEBUG_RAWWELLS_THREAD
    std::cout <<"====> Current handshake flow: " << curFlow << std::endl;
#endif

    float *ampBuf = info->ampEstimatesBuf->readOneBuffer();

    WorkerInfoQueueItem item;
    for (unsigned int i=0; i<numFitters; ++i) {
      /*(*(info->fitters))[i]->GetGlobalStage().setCurFlowByFlowAmpBuf(ampBuf, curFlow);
      (*(info->fitters))[i]->GetGlobalStage().WriteFlowByFlowToWells(
                                                 (*(info->fitters))[i]->region_data->region,
                                                 (*(info->fitters))[i]->region_data->my_beads);
       */
      writeJobs[i]->setCurFlow(curFlow);
      writeJobs[i]->setCurAmpBuffer(ampBuf);
      item.finished = false;
      item.private_data = writeJobs[i];       
      jobQueue.PutItem(item);      
    }
    jobQueue.WaitTillDone();

#if DEBUG_RAWWELLS_THREAD
    std::cout << "====> Finished Flow: " << curFlow << std::endl;
#endif

    info->rawWells->DoneUpThroughFlow(curFlow, info->packQueue, info->writeQueue);

    info->ampEstimatesBuf->updateReadPos();

    if (curFlow == (info->endingFlow - 1)) {
      done = true;  
#if DEBUG_RAWWELLS_THREAD
      std::cout <<"====> Last handshake flow: " << curFlow << " exiting" <<  std::endl;
#endif
      break;
    }

    curFlow++;    
  }

  // send kill jobs to the threads
  PJobExit exitJob;
  WorkerInfoQueueItem item;
  for (size_t i=0; i<numRawWellWriterBkgThreads; ++i) {
    //TODO: redundant..need to make work queue generic to allow exit jobs
    item.finished = true; 
    item.private_data = &exitJob;
    jobQueue.PutItem(item);
  }
  jobQueue.WaitTillDone();

#if DEBUG_RAWWELLS_THREAD
  std::cout << "====> Cleaned up local threads in handshake worker" << std::endl;
#endif


  // clear memory taken up by job creation
  for (size_t i=0; i<numFitters; ++i) {
    delete writeJobs[i];
  }

  return NULL;
}
