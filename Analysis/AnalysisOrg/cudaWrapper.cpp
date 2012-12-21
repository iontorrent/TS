/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "cudaWrapper.h"

#ifdef ION_COMPILE_CUDA
#include "SingleFitStream.h"
#include "MultiFitStream.h"
#include "StreamingKernels.h"
#include "JobWrapper.h"
#include "SignalProcessingFitterQueue.h"
#endif

void* BkgFitWorkerGpu(void *arg)
{
#ifdef ION_COMPILE_CUDA
    // Unpack GPU worker info
    BkgFitWorkerGpuInfo* info = static_cast<BkgFitWorkerGpuInfo*>(arg);

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

        printf( "CUDA: Created GPU BkgModel worker...  (%d: %s v%d.%d)\n", dev_id, cuda_props.name, cuda_props.major, cuda_props.minor );

        if (info->type == GPU_SINGLE_FLOW_FIT)
          return(SingleFlowFitGPUWorker( info->queue) );
        else if (info->type == GPU_MULTI_FLOW_FIT)
          return(MultiFlowFitGPUWorker( info->queue) );
    }
        
    printf( "CUDA: Failed to initialize GPU worker... (%d: %s)\n", info->gpu_index, cudaGetErrorString(err) );

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
  const unsigned long long gpu_mem = 2.5 * 1024 * 1024 * 1024;

  if (!use_gpu_acceleration)
    return false;

  // Get number of GPUs in system
  int num_gpus = 0;
  cudaError_t err = cudaGetDeviceCount( &num_gpus );

  if (err != cudaSuccess) {
    printf("CUDA: No GPU device available. Defaulting to CPU only computation\n");
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
    numBkgWorkers_gpu = int(valid_devices.size());
    cudaSetValidDevices( &valid_devices[0], int( valid_devices.size() ) );
  }
  else {
    printf("CUDA: No GPU device available. Defaulting to CPU only computation\n");
    return false;   
  }

 
  PoissonCDFApproxMemo poiss_cache; 
  poiss_cache.Allocate (MAX_POISSON_TABLE_COL,MAX_POISSON_TABLE_ROW,POISSON_TABLE_STEP);
  poiss_cache.GenerateValues(); // fill out my table


  for(int i=valid_devices.size()-1 ; i >= 0; i--){
    try{
      //cudaSetDevice(valid_devices[i]);
      cout << "CUDA: Creating Context and Contant memory on device with id: "<<  valid_devices[i]<< endl;
      InitConstantMemoryOnGpu(valid_devices[i],poiss_cache);
    }
    catch(cudaException &e) {
      cout << "CUDA: Context could not be created. removing device with id: "<<  valid_devices[i] << " from valid device list" << endl;
      valid_devices.erase (valid_devices.begin()+i);
      numBkgWorkers_gpu -= 1;
      if(numBkgWorkers_gpu == 0) cout << "CUDA: no context could be created, defaulting to CPU only execution" << endl; 
    }

  }

  if(numBkgWorkers_gpu == 0) return false;

  cudaStreamPool::initLockNotThreadSafe();

  return true;

#else
  
  return false;

#endif

}

void InitConstantMemoryOnGpu(int device, PoissonCDFApproxMemo& poiss_cache) {
#ifdef ION_COMPILE_CUDA
  initPoissonTablesLUT(device, (void**) poiss_cache.poissLUT);
#endif
}

void SingleFlowStreamExecutionOnGpu(WorkerInfoQueue* q) {
#ifdef ION_COMPILE_CUDA
  bool fallBackToCPU = false;
  cudaStreamManager sM;
  int dev_id;
  //stream config
  SingleFitStream::setBeadsPerBLock(128); // cuda block size
  cudaGetDevice( &dev_id );
 
//  SingleFitStream * temp;

  sM.printMemoryUsage();

  for(int i=0; i < NUM_CUDA_FIT_STREAMS-1; i++){

 /*  try{ // exception handling to allow fallback to CPU Fit if not a single strweam could be created
      sM.addStreamUnit(new SingleFitStream(q));
      cudaGetDevice( &dev_id );
      std::cout <<"CUDA: Device " <<  dev_id <<  " SingleFitStream " << i <<" created " << std::endl;
      sM.printMemoryUsage();
    }
    catch(cudaException& e)
    {
      cout << e.what() << endl;
      if(i > 0){ 
          cout << "CUDA: Device " << dev_id<< " could not create more than " << i << " SingleFitStreams" << std::endl;  
          sM.printMemoryUsage();
      }else{
        std::cout << "CUDA: Device " << dev_id << " no SingleFitStreams could be created >>>>>>>>>>>>>>>>> FALLING BACK TO CPU!"<< std::endl;
        fallBackToCPU = true;
      }
      break;
    }

*/
      if(!TryToAddSingleFitStream(&sM,q)){
        if(i == 0 ) fallBackToCPU = true;
        break;
      }
  } 


  
  if(!fallBackToCPU){ // GPU WORKGER
    int flownum = 0; 
    int tryedCreateSecondStream = false;
    while ( sM.DoWork(&flownum) ){ 
    
     if (!tryedCreateSecondStream && flownum >= NUMFB )
      {
        std::cout << "CUDA: starting second block of 20 flows, try to create second SingleFitStream " << std::endl;  
        TryToAddSingleFitStream(&sM,q);
        tryedCreateSecondStream = true;
      }
    };
    
    std::cout << "CUDA: Device " << dev_id << " GPU memory profile when GPU thread exits" << std::endl; 
    sM.printMemoryUsage();
  
  }else{ // FALLBACK CPU WORKER

    bool done = false;
    while(!done)
    {
      
      //reverte to be a blocking CPU thread if no GPU stream got created
      WorkerInfoQueueItem item = q->GetItem();
    
      if (item.finished == true){
        // we are no longer needed...go away!
        done = true;
        q->DecrementDone();
        continue;
      }
      // only single flow fit and post processing is done 
      DoSingleFlowFitAndPostProcessing(item);
      // indicate we finished that bit of work
      q->DecrementDone();
    }

      
  }

#endif
}

void MultiFlowStreamExecutionOnGpu(WorkerInfoQueue* q)
{
#ifdef ION_COMPILE_CUDA
  std::cout << "CUDA: Creating MultiFlowFit GPU workers" << std::endl;
  bool fallBackToCPU = false;
  cudaStreamManager sM;
  int dev_id;
  //stream config
  MultiFitStream::setBeadsPerBLock(128); // cuda block size
  
  sM.printMemoryUsage();

  GpuMultiFlowFitControl fit_control;
  for(int i=0; i < 1; i++)
  {  
    try{ // exception handling to allow fallback to CPU Fit if not a single strweam could be created
      sM.addStreamUnit(new MultiFitStream(fit_control, q));
      cudaGetDevice( &dev_id );
      std::cout <<"CUDA: Device " <<  dev_id <<  " MultiFitStream " << i <<" created " << std::endl;
      sM.printMemoryUsage();
    }
    catch(cudaException& e)
    {
      cout << e.what() << endl;
      if(i > 0){ 
          cout << "CUDA: Device " << dev_id<< " could not create more than " << i << " Multi Fit streams" << std::endl;  
          sM.printMemoryUsage();
      }else{
        std::cout << "CUDA: Device " << dev_id << " no Multi Fit streams could be created >>>>>>>>>>>>>>>>> FALLING BACK TO CPU!"<< std::endl;
        fallBackToCPU = true;
      }
      break;
    }
  } 

  if(!fallBackToCPU){ // GPU WORKGER
     
    while ( sM.DoWork() ) {};
    
    std::cout << "CUDA: Device " << dev_id << " GPU memory profile when GPU threads exits" << std::endl; 
    sM.printMemoryUsage();
  
  }

  if (fallBackToCPU)
  { // FALLBACK CPU WORKER

    bool done = false;
    while(!done)
    {
      
      //reverte to be a blocking CPU thread if no GPU stream got created
      WorkerInfoQueueItem item = q->GetItem();
    
      if (item.finished == true){
        // we are no longer needed...go away!
        done = true;
        q->DecrementDone();
        continue;
      }
      DoInitialBlockOfFlowsAllBeadFit(item);
      // indicate we finished that bit of work
      q->DecrementDone();
    }
  }

#endif
}


bool TryToAddSingleFitStream(void * vpsM, WorkerInfoQueue* q){
#ifdef ION_COMPILE_CUDA
  int dev_id = 0;
  cudaStreamManager * psM = (cudaStreamManager *) vpsM;
  SingleFitStream * temp;
  cudaGetDevice( &dev_id );
  int i;
    try{ // exception handling to allow fallback to CPU Fit if not a single strweam could be created
      temp =  new SingleFitStream(q);
      i = psM->addStreamUnit( temp);
      std::cout <<"CUDA: Device " <<  dev_id <<  " Single Fit stream " << i <<" created " << std::endl;
      psM->printMemoryUsage();
    }
    catch(cudaException& e)
    {
      cout << e.what() << endl;
      if(psM->getNumStreams() > 0){ 
        cout << "CUDA: Device " << dev_id<< " could not create more than " << psM->getNumStreams() << " Single Fit streams" << std::endl;       
        psM->printMemoryUsage();
      }else{
        std::cout << "CUDA: Device " << dev_id << " no Single Fit streams could be created >>>>>>>>>>>>>>>>> FALLING BACK TO CPU!"<< std::endl;
        return false;
      }
    }

#endif
  return true;
}



