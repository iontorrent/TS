/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <algorithm>
#include "cudaWrapper.h"
#include "SignalProcessingFitterQueue.h"

//#define ION_COMPILE_CUDA


#ifdef ION_COMPILE_CUDA
#include "SingleFitStream.h"
#include "MultiFitStream.h"
#include "StreamingKernels.h"
#include "JobWrapper.h"
#endif

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
  const unsigned long long gpu_mem = 0.4 * 1024 * 1024 * 1024;

  if (!use_gpu_acceleration)
    return false;

  // Get number of GPUs in system
  int num_gpus = 0;
  cudaError_t err = cudaGetDeviceCount( &num_gpus );

  if (err != cudaSuccess) {
      printf("CUDA: No GPU device available. Defaulting to CPU only computation (return code %d: %s) &\n", err , cudaGetErrorString(err));
      return false;
  }

  if (valid_devices.size() == 0) {
    if ( use_all_gpus )
    {
      // Add all GPUs to the valid device list
      for ( int dev = 0; dev < num_gpus;  dev++ )
        valid_devices.push_back(dev);
    }
    else
    {
      // Only add the highest compute devices to the compute list
      int version = 0;
      int major = 0;
      int minor = 0;
      cudaDeviceProp dev_props;

      // Iterate over GPUs to find the highest compute device
      for ( int dev = 0; dev < num_gpus;  dev++ )
      {
        cudaGetDeviceProperties( &dev_props, dev );
        if ( (dev_props.major*10) + dev_props.minor > version )
        {
          version = (dev_props.major*10) + dev_props.minor;
          major = dev_props.major;
          minor = dev_props.minor;
        }
      }

      for ( int dev = 0; dev < num_gpus;  dev++ )
      {
        cudaGetDeviceProperties(&dev_props, dev);
        if (dev_props.major == major && dev_props.minor == minor) {
          if (dev_props.totalGlobalMem > gpu_mem) {
            valid_devices.push_back(dev);
          }
        }
      } 
    }
  }
  else {
    while (valid_devices.size() > 0) {
      if (valid_devices.back() > (num_gpus - 1)) {
        valid_devices.pop_back();
      }
      else {
        break;
      }
    }
  }

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
      //cudaSetDevice(valid_devices[i]);
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

