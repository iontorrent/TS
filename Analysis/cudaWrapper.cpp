/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "cudaWrapper.h"

void* BkgFitWorkerGpu(void *arg)
{
#ifdef ION_COMPILE_CUDA
    // Unpack GPU worker info
    BkgFitWorkerGpuInfo* info = reinterpret_cast<BkgFitWorkerGpuInfo*>(arg);

    // Wrapper to create a GPU worker and set GPU
    cudaSetDevice( info->gpu_index );
    cudaError_t err = cudaGetLastError();

    // See if device was set OK
    if ( err == cudaSuccess )
    {
        cudaDeviceProp cuda_props;
        int dev_id;
        cudaGetDevice( &dev_id );
        cudaGetDeviceProperties( &cuda_props, dev_id );

        printf( "Created CUDA BkgModel worker...  (%d: %s v%d.%d)\n", dev_id, cuda_props.name, cuda_props.major, cuda_props.minor );

        if (info->dynamic_gpu_load)
            return( DynamicBkgFitWorker( info->queue, true ) );
        else 
            return( BkgFitWorker( info->queue, true ) );
    }
        
    printf( "Failed to initialize CUDA worker... (%d: %s)\n", info->gpu_index, cudaGetErrorString(err) );

    // Note: could fall back to a CPU worker here by setting GPU flag to false,
    //       though I'm not sure if that's always appropriate

    return NULL;
#else 
    return NULL;
#endif
}


bool configureGpu(bool use_gpu_acceleration, std::vector<int> &valid_devices, int use_all_gpus, 
  int numGpuThreads, int &numBkgWorkers_gpu) {
#ifdef ION_COMPILE_CUDA
  const unsigned long long gpu_mem = 2.5 * 1024 * 1024 * 1024;

  if (!use_gpu_acceleration)
    return false;

  // Get number of GPUs in system
  int num_gpus = 0;
  cudaError_t err = cudaGetDeviceCount( &num_gpus );

  if (err != cudaSuccess) {
    printf("No GPU device available. Defaulting to CPU only computation\n");
    return false;
  }

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

  // Set the number of GPU workers and tell CUDA about our list of valid devices
  if (valid_devices.size() > 0) {
    numBkgWorkers_gpu = numGpuThreads * int(valid_devices.size());
    cudaSetValidDevices( &valid_devices[0], int( valid_devices.size() ) );
  }
  else {
    printf("No GPU device available. Defaulting to CPU only computation\n");
    return false;   
  }

  return true;

#else
  
  return false;

#endif

}

void InitConstantMemoryOnGpu(PoissonCDFApproxMemo& poiss_cache) {
#ifdef ION_COMPILE_CUDA
  BkgModelCuda::InitializeConstantMemory(poiss_cache);
#endif
}
