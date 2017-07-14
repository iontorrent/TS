/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <algorithm>
#include "cudaWrapper.h"
#include "SignalProcessingFitterQueue.h"

#ifdef ION_COMPILE_CUDA
#include "SingleFitStream.h"
#include "MultiFitStream.h"
#include "StreamingKernels.h"
#include "JobWrapper.h"
#include "BkgGpuPipeline.h"
#endif

#include "RawWellsWriteJob.h"
#include "PJob.h"
#include "PJobExit.h"
#include "ChipIdDecoder.h"

using namespace std;

#define DEBUG_RAWWELLS_THREAD 0
#define MIN_CUDA_COMPUTE_VERSION 20

#define CUDA_WRONG_TIME_AND_PLACE  \
    {   \
       std::cout << "CUDAWRAPPER: WE SHOULD NOT EVEN BE HERE, THIS PART OF THE CODE SHOULD BE UNREACHABLE WITHOUT GPU: " << __FILE__ << " " << __LINE__ <<std::endl;  \
    }



///////////////////////////////////////////////
//gpuDeviceConfig Class

//PROTECTED:

int gpuDeviceConfig::getVersion(int devId)
{
  int ret = 0;
#ifdef ION_COMPILE_CUDA

  cudaDeviceProp dev_props;
  cudaError_t err = cudaGetDeviceProperties( &dev_props, devId );
  if (err != cudaSuccess) return 0;
  ret =  (dev_props.major*10) + dev_props.minor; // meh. works as long as minor single digit.
  cout << "CUDA: gpuDeviceConfig: device added for evaluation: " << devId << ":" << dev_props.name <<" v" << dev_props.major <<"." << dev_props.minor << " " << dev_props.totalGlobalMem/(1024.0*1024.0*1024.0) << "GB" << endl;

  if(maxComputeVersion == 0){
    maxComputeVersion = minComputeVersion = ret;
  }else{
    maxComputeVersion = max(maxComputeVersion,ret);
    minComputeVersion = min(minComputeVersion,ret);
  }
#else
  CUDA_WRONG_TIME_AND_PLACE
#endif
  return ret;
}

/*removes devices from valid device list that do not meet minimum memory requirement*/
void gpuDeviceConfig::applyMemoryConstraint(size_t minBytes)
{
#ifdef ION_COMPILE_CUDA
  cudaDeviceProp dev_props;
  /*
  for (unsigned i = validDevices.size(); i-->0; ){
    cudaGetDeviceProperties( &dev_props, validDevices[i] );
    if(dev_props.totalGlobalMem < minBytes)
      validDevices.erase(validDevices.begin()+i);
  }
   */
  for(std::vector<int>::iterator it = validDevices.begin(); it != validDevices.end();)
  {
    cudaGetDeviceProperties( &dev_props, *it);
    if(dev_props.totalGlobalMem < minBytes)
      it = validDevices.erase(it);
    else
      ++it;
  }
#else
  CUDA_WRONG_TIME_AND_PLACE
#endif

  return;

}

/*removes all devices from valid device list with a compute version below minVersion */
void gpuDeviceConfig::applyComputeConstraint(int minVersion)
{
#ifdef ION_COMPILE_CUDA
  cudaDeviceProp dev_props;
  for(std::vector<int>::iterator it = validDevices.begin(); it != validDevices.end(); )
  {
    cudaGetDeviceProperties( &dev_props, *it);
    int version = (dev_props.major*10) + dev_props.minor;
    if(version < minVersion)
      it = validDevices.erase(it);
    else
      ++it;
  }
#else
  CUDA_WRONG_TIME_AND_PLACE
#endif
  return;
}


void gpuDeviceConfig::initDeviceContexts(){

#ifdef ION_COMPILE_CUDA
  PoissonCDFApproxMemo poiss_cache;
  poiss_cache.Allocate (MAX_POISSON_TABLE_COL,MAX_POISSON_TABLE_ROW,POISSON_TABLE_STEP);
  poiss_cache.GenerateValues(); // fill out lookup table poissLUT etc...
  for(std::vector<int>::iterator it = validDevices.begin(); it != validDevices.end(); )
  {
    try{
      cout << "CUDA "<< *it << ": gpuDeviceConfig::initDeviceContexts: Creating Context and Constant memory on device with id: "<<  *it<< endl;

#if defined( __SSE__ )
      // lookup table generation above in 'poiss_cache.GenerateValues()' uses SSE, copy LUT to device only if SSE is available
      // Analysis/BkgModel/MathModel/PoissonCdf.cpp : void PoissonCDFApproxMemo::GenerateValues()
      StreamingKernels::initPoissonTablesLUT(*it, (void**) poiss_cache.poissLUT);
#endif
      it++;
    }
    catch(cudaException &e) {
      //throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
      cout << e.what() << endl;
      cout << "CUDA "<< *it << ": gpuDeviceConfig::initDeviceContexts: Context could not be created. removing device with id: "<<  *it << " from valid device list" << endl;
      it = validDevices.erase (it);
    }
  }
#else
  CUDA_WRONG_TIME_AND_PLACE
#endif
  return;
}



gpuDeviceConfig::gpuDeviceConfig():maxComputeVersion(0),minComputeVersion(0){ };


bool gpuDeviceConfig::setValidDevices(std::vector<int> &CmdlineDeviceList, bool useMaxComputeVersionOnly)
{

  validDevices.clear();
  maxComputeVersion = 0;
  minComputeVersion = 0;

#ifdef ION_COMPILE_CUDA

  size_t minMemory = 1.0 * 1024 * 1024 * 1024;
  int numGPUs = 0;

  validDevices.clear();

  cudaError_t err = cudaGetDeviceCount( &numGPUs );
  if (err != cudaSuccess) {
    printf("CUDA: gpuDeviceConfig: No GPU device available. Defaulting to CPU only computation (return code %d: %s) &\n", err , cudaGetErrorString(err));
    return false; // we are done here
  }
  //if deviced passed from command line:
  if (CmdlineDeviceList.size() > 0){ //overwrite automated device selection, no checks performed. it is assumed that if cmd line option is give someone knows what he/she is doing!
    for (std::vector<int>::iterator itDevId = CmdlineDeviceList.begin() ; itDevId != CmdlineDeviceList.end(); ++itDevId)
    {
      if(getVersion(*itDevId)){  //if device exists push to back
        validDevices.push_back(*itDevId);
      }else{
        printf("CUDA WARNING: gpuDeviceConfig: Device with device id %d provided through the command line could not be found and will be ignored!\n", *itDevId);
      }
    }
    if(validDevices.size() == 0)
      printf("CUDA WARNING: gpuDeviceConfig: THE DEVICE LIST PROVIDED TO THE COMMAND LINE DID NOT CONTAIN ANY VALID DEVICES!\n");

  }else{
    //add all devices to vector
    for ( int dev = 0; dev < numGPUs;  dev++ ){
      if(getVersion(dev)) //check valid device and record min/max compute
        validDevices.push_back(dev);
    }

    //apply constraints and remove deivses that do not pass.
    applyMemoryConstraint(minMemory);

    int minCompVersion = (useMaxComputeVersionOnly)?(maxComputeVersion):(MIN_CUDA_COMPUTE_VERSION);
    printf("CUDA: gpuDeviceConfig: minimum compute version used for pipeline: %.1f\n", (float)minCompVersion/10.0 );
    applyComputeConstraint(minCompVersion);

  }

  if (validDevices.size() == 0) {
    err = cudaGetLastError();
    printf("CUDA: gpuDeviceConfig: No GPU device available or device not valid. Defaulting to CPU only computation (return code %d: %s) &\n", err , cudaGetErrorString(err));
    return false;
  }

  initDeviceContexts();

  if(! (validDevices.size() > 0) )
    cout << "CUDA: gpuDeviceConfig: no context could be created, defaulting to CPU only execution" << endl;
  else
    cudaSetValidDevices( &validDevices[0], int( validDevices.size()));

#endif

  return (validDevices.size() > 0);
}


///////////////////////////////////////////////
//cpuWorkerThread Class
queueWorkerThread::queueWorkerThread(WorkerInfoQueue *q)
{
  wq=q;
}

bool queueWorkerThread::start()
{
  if(wq == NULL) return false;
  return StartInternalThread();
}

void queueWorkerThread::join(){
  JoinInternalThread();
}

void queueWorkerThread::InternalThreadFunction()
{
  while(true) {
    WorkerInfoQueueItem item = wq->GetItem();
    PJob *job = static_cast<PJob*>(item.private_data);
    if (job->IsEnd()) {
      job->Exit();
      wq->DecrementDone();
      break;
    }
    else {
      job->Run();
    }
    wq->DecrementDone();
  }
}


/////////////////////////////////////////////////////////
// cudaFlowByFlowHandShaker class

flowByFlowHandshaker::flowByFlowHandshaker()
{
  startingFlow = 20;
  endingFlow = 20;
  packQueue = NULL;
  writeQueue = NULL;
  rawWells = NULL;
  ampEstimatesBuf = NULL;
  fitters=NULL;

}

flowByFlowHandshaker::flowByFlowHandshaker(SemQueue * ppackQueue, SemQueue * pwriteQueue,ChunkyWells * pRawWells, std::vector<SignalProcessingMasterFitter*> * pfitters)
{
  startingFlow = 20;
  endingFlow = 20;
  packQueue=ppackQueue;
  writeQueue=pwriteQueue;
  rawWells = pRawWells;
  fitters = pfitters;
  ampEstimatesBuf = NULL;
}


flowByFlowHandshaker::~flowByFlowHandshaker()
{
  if (ampEstimatesBuf)
    delete ampEstimatesBuf;
}

void flowByFlowHandshaker::CreateRingBuffer( int numBuffers, int bufSize)
{
  if (!ampEstimatesBuf)
    ampEstimatesBuf = new RingBuffer<float>(numBuffers, bufSize);
}

//This is the actual thread function
void flowByFlowHandshaker::InternalThreadFunction()
{
//#if DEBUG_RAWWELLS_THREAD
  std::cout << "CUDA: flowByFlowHandshaker: Started GPU-CPU handshake worker" << std::endl;
//#endif
  assert(fitters != NULL);
  assert(ampEstimatesBuf != NULL);
  assert(packQueue != NULL && writeQueue !=NULL);
  assert(rawWells != NULL);

  int numFitters = fitters->size();
  //create job handles for raw wells writer threads
  std::vector<RawWellsWriteJob*> writeJobs;
  writeJobs.resize(numFitters);
  for (int i=0; i<numFitters; ++i) {
    writeJobs[i] = new RawWellsWriteJob((*(fitters))[i]);
  }

  // create work queue and worker threads
  int numRawWellWriterBkgThreads = 6;
  WorkerInfoQueue jobQueue(numRawWellWriterBkgThreads*numFitters);


  vector<queueWorkerThread *>workers;
  for (int i=0; i<numRawWellWriterBkgThreads; ++i) {
    queueWorkerThread * worker = new queueWorkerThread(&jobQueue);
    if(worker->start())
      workers.push_back(worker);
    else
      delete worker;
  }

  int curFlow = startingFlow;
  bool done = false;
  while (!done) {

#if DEBUG_RAWWELLS_THREAD
    std::cout <<"====> Current handshake flow: " << curFlow << std::endl;
#endif

    float *ampBuf = ampEstimatesBuf->readOneBuffer();
    WorkerInfoQueueItem item;
    for ( int i=0; i<numFitters; ++i) {
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

    rawWells->DoneUpThroughFlow(curFlow, *packQueue, *writeQueue);

    ampEstimatesBuf->updateReadPos();

    if (curFlow == (endingFlow - 1)) {
      done = true;
      break;
#if DEBUG_RAWWELLS_THREAD
      std::cout <<"====> Last handshake flow: " << curFlow << " exiting" <<  std::endl;
#endif
    }

    curFlow++;
  }

  // send kill jobs to the threads
  PJobExit exitJob;
  WorkerInfoQueueItem item;
  for (int i=0; i<numRawWellWriterBkgThreads; ++i) {
    //TODO: redundant..need to make work queue generic to allow exit jobs
    item.finished = true;
    item.private_data = &exitJob;
    jobQueue.PutItem(item);
  }
  //jobQueue.WaitTillDone();
  /*join all the worker threads*/

  //cout << "flowByFlowHandshaker: joining " << workers.size() << " worker threads." << endl;
  for (vector<queueWorkerThread*>::iterator it = workers.begin() ; it != workers.end(); ++it)
  {
    (*it)->join();
    delete (*it);
  }
  cout << "CUDA: flowByFlowHandshaker: " << workers.size() << " worker threads joined." <<endl;

  // clear memory taken up by job creation
  for (int i=0; i<numFitters; ++i) {
    delete writeJobs[i];
  }
}

bool flowByFlowHandshaker::start(){

  if(ampEstimatesBuf == NULL) return false;


  return StartInternalThread();
}

void flowByFlowHandshaker::join(){
  JoinInternalThread();
}



/////////////////////////////////////////////////////////
// gpuBkgFitWorker class

gpuBkgFitWorker::gpuBkgFitWorker(int devId):devId(devId),q(NULL),errorQueue(NULL){};


void gpuBkgFitWorker::createStreamManager(){

#ifdef ION_COMPILE_CUDA
  cout << "CUDA " << devId << ": gpuBkgFitWorker: Creating GPU StreamManager" << endl;
  cudaDeviceProp cuda_props;
  cudaGetDeviceProperties( &cuda_props, devId );

  cudaSimpleStreamManager  sM( q, errorQueue );

  cout << "CUDA " <<  devId <<": gpuBkgFitWorker: Created GPU BkgModel worker...  ("
              << devId <<":"
              << cuda_props.name
              << " v"<< cuda_props.major <<"."<< cuda_props.minor << ")" << endl;
  sM.DoWork();
  cout << "CUDA " << devId << ": gpuBkgFitWorker: Destroying GPU StreamManager" << endl;

#else
  CUDA_WRONG_TIME_AND_PLACE
#endif
  return;
}


void gpuBkgFitWorker::InternalThreadFunction(){

#ifdef ION_COMPILE_CUDA
  // Unpack GPU worker info
  // Wrapper to create a GPU worker and set GPU
  //cout << "GPU_INDEX " << devId << endl;
  cudaSetDevice( devId );
  cudaError_t err = cudaGetLastError();
  // See if device was set OK
  if ( err == cudaSuccess )
  {
    createStreamManager();

    return;
  }

  cout << "CUDA: gpuBkgFitWorker: Failed to initialize GPU worker... (" << devId <<": " << cudaGetErrorString(err)<<")" << endl;
#else
  CUDA_WRONG_TIME_AND_PLACE
#endif

  return;
}


bool gpuBkgFitWorker::start(){
#ifdef ION_COMPILE_CUDA
  //cudaSetDevice( devId );
  //cudaError_t err = cudaGetLastError();
  //if ( err == cudaSuccess ){
    if(q == NULL || errorQueue == NULL || errorQueue == q) return false;
    if( devId < 0) return false;


   return StartInternalThread();

  //}
  //cout << "CUDA: Failed to initialize GPU worker... (" << devId <<": " << cudaGetErrorString(err)<<")" << endl;
#else
  CUDA_WRONG_TIME_AND_PLACE
#endif
  return false;
}

void gpuBkgFitWorker::join(){
  JoinInternalThread();
}




/////////////////////////////////////////////////////////
// cudaWrapper class


cudaWrapper::cudaWrapper(){
  useGpu = false;
  useAllGpus = false;
  workQueue = NULL;
  Handshaker = NULL;
  configOpts = NULL;
  #ifdef ION_COMPILE_CUDA
  GpuPipeline = NULL;
  RegionalFitHistory=NULL;
#endif
}

cudaWrapper::~cudaWrapper(){

  if(BkgWorkerThreads.size() > 0)
    cout << "CUDA WARNING: cudaWrapper: destructor is called while there are still " << BkgWorkerThreads.size() << " Worker Threads in active state!" <<endl;

  destroyQueue();

  if(Handshaker != NULL)
    delete Handshaker;

#ifdef ION_COMPILE_CUDA
  if(GpuPipeline!=NULL)
    delete GpuPipeline;
  GpuPipeline = NULL;
  if(RegionalFitHistory == NULL)
    delete RegionalFitHistory;
#endif

}

bool cudaWrapper::checkChipSize()
{
  if (! ChipIdDecoder::BigEnoughForGPU())
  {
    printf("CUDA: cudaWrapper: GPU acceleration suppressed on small chips\n");
    useGpu = false;
    return false;
  }
  return true;
}
//formerly part of PlanMyComputation
void cudaWrapper::configureGpu(BkgModelControlOpts &bkg_control)
{

  configOpts = &bkg_control.gpuControl;
  // This will override gpuWorkLoad=1 and will only use GPU for chips which are allowed in the following function
  useGpu = (configOpts->gpuWorkLoad > 0);

  //only perform next steps if chiop is large enough and gpu compute turned on
  if( useGpu && checkChipSize())
  {
    useAllGpus = false; //ToDo add comandline param to overwrite
    bool useMaxComputeVersion = false; //ToDo addcommandline param to change this if needed

    //configure actual GPUs. if compile without CUDA this is a NoOp
    useGpu = deviceConfig.setValidDevices(configOpts->gpuDeviceIds,useMaxComputeVersion);
  }

  if (!useGpu)
    configOpts->gpuFlowByFlowExecution = false;

  cout << "CUDA: useGpuAcceleration: "<< useGpuAcceleration() << endl;

}



void cudaWrapper::createQueue(int numRegions)
{
  if(useGpuAcceleration()){ //assume that we will have one worker per valid device
    workQueue = new WorkerInfoQueue (numRegions * getNumValidDevices() + 1 );  // where the +1 comes from nobody remembers but ehre probably was a reason for it once
    //testSerial = float((int64_t)workQueue/12345.6f);
  }
}

void cudaWrapper::destroyQueue()
{
  if(workQueue)
    delete workQueue;

  workQueue = NULL;
}

WorkerInfoQueue * cudaWrapper::getQueue()
{
  return workQueue;
}

bool cudaWrapper::checkIfInitFlowByFlow(int currentFlow, bool restart){

 #ifdef ION_COMPILE_CUDA

  if(useFlowByFlowExecution()){
    if(switchAtFlow() == currentFlow){
      cout << "CUDA: cudaWrapper: flow " << switchAtFlow() << " reached, switching from old block of 20 flows to NEW flow by flow GPU pipeline!" <<endl;
      return true;
    }
    if(restart && GpuPipeline == NULL){
      cout << "CUDA: cudaWrapper: Initiating flow by flow GPU pipeline after restart!" <<endl;
      return true;
    }
  }
#endif

  return false ;
}


bool cudaWrapper::SpinUpThreads( WorkerInfoQueue* fallbackCPUQ)
{

  if(useGpuAcceleration()){

    const vector<int> validDevices = deviceConfig.getValidDevices();

    if(validDevices.size() == 0) return false;

    for(vector<int>::const_iterator cit = validDevices.begin(); cit != validDevices.end(); ++cit){
      gpuBkgFitWorker * worker = new gpuBkgFitWorker(*cit);
      worker->setWorkQueue(workQueue);
      worker->setFallBackQueue((fallbackCPUQ));
      if(worker->start()){
        BkgWorkerThreads.push_back(worker);
        cout << "CUDA: cudaWrapper::SpinUpThreads: " << BkgWorkerThreads.size() << " GPU Bkg worker threads created" << endl;
      }else{
        cout << "CUDA: cudaWrapper::SpinUpThreads: failed to create worker a thread" << endl;
        delete worker;
      }
    }
    //no threads created! ToDo: initiate fallback to gpu in caller
    if(BkgWorkerThreads.size() > 0) return true;

    /*error state! no GPU workers could be created*/
    destroyQueue();
    useGpu = false;
    cout << "CUDA ERROR: cudaWrapper::SpinUpThreads: Failed to create any GPU worker threads. cleaning up GPU pipeline and fall back to CPU only Execution!";
  }

  return false;
}


void cudaWrapper::UnSpinThreads()
{
  if(getQueue())
  { /*put finish jobs inti queue */
    WorkerInfoQueueItem item;
    item.finished = true;
    item.private_data = NULL;
    for (int i=0;i < getNumWorkers();i++)
      getQueue()->PutItem (item);
  }
  /*wait for worker threads to complete work and join */
  size_t numgputhreads = BkgWorkerThreads.size();
  for(vector<gpuBkgFitWorker*>::iterator it = BkgWorkerThreads.begin(); it != BkgWorkerThreads.end(); )
  {
    (*it)->join(); /*wait for worker thread to join */
    delete (*it); /* delete thread wrapper class */
    (*it) = NULL;
    it = BkgWorkerThreads.erase(it); /* erase pointer to thread wrapper class from vector and move iterator to next element*/
  }
  if(numgputhreads > 0 ) cout << "CUDA: cudaWrapper::UnSpinThreads: all " << numgputhreads << " GPU worker threads are joined." <<endl;
  destroyQueue();

}
void cudaWrapper::setUpAndStartFlowByFlowHandshakeWorker( const CommandLineOpts &inception_state,
                                                          const ImageSpecClass &my_image_spec,
                                                          std::vector<SignalProcessingMasterFitter*> * fitters,
                                                          SemQueue *packQueue,
                                                          SemQueue *writeQueue,
                                                          ChunkyWells *rawWells,
                                                          int startFlow)
{

  Handshaker = new flowByFlowHandshaker(packQueue,writeQueue,rawWells, fitters);
  Handshaker->setStartEndFlow(startFlow,inception_state.flow_context.endingFlow);
  Handshaker->CreateRingBuffer(20, my_image_spec.rows * my_image_spec.cols);
  Handshaker->start();
}

void cudaWrapper::joinFlowByFlowHandshakeWorker()
{
  if(Handshaker != NULL){
    Handshaker->join();
    cout << "CUDA: cudaWrapper::joinFlowByFlowHandshakeWorker: GPU-CPU handshake worker thread joined." <<endl;
  }
}

bool cudaWrapper::fullBlockSignalProcessing(BkgModelWorkInfo* bkinfo)
{

#ifdef ION_COMPILE_CUDA


    cudaSetDevice(deviceConfig.getFirstValidDevice());
    // create static GpuPipeline Object
    //allocates all permanent device and host buffers
    //initializes all persistent buffers on the device side
    //initializes the poissontables
    //runs the T0average num bead meta data kernel
    if(GpuPipeline == NULL)
      GpuPipeline = new BkgGpuPipeline(bkinfo, deviceConfig.getFirstValidDevice(),RegionalFitHistory );

    //Update Host Side Buffers and Pinned status
    Timer newPtime;

    // This needs to go as floworder information is available and can be copied to device memory
    // on the first call to this pipeline
    // Probably need new pinned mask to be transferred every flow so that ZeroOutPins
    // can be avoided
    // No need to increment the flow here. Again just need starting flow on device
    // and things can fly from there

    try{

     //update all per flow by flow data
     GpuPipeline->PerFlowDataUpdate(bkinfo);

     //backup current region param state, this function only does work the very first time it gets called
     //GpuPipeline->InitRegionalParamsAtFirstFlow();

     //GpuPipeline->DebugOutputDeviceBuffers();

     GpuPipeline->ExecuteGenerateBeadTrace();

     GpuPipeline->ExecuteCrudeEmphasisGeneration();
     GpuPipeline->ExecuteRegionalFitting();
     GpuPipeline->HandleRegionalFittingResults();

     GpuPipeline->ExecuteFineEmphasisGeneration();

     GpuPipeline->ExecuteTraceLevelXTalk();

     GpuPipeline->ExecuteSingleFlowFit();
     GpuPipeline->ExecutePostFitSteps();
     GpuPipeline->HandleResults(Handshaker->getRingBuffer()); // copy reg_params and single flow fit results to host

     }
     catch(exception &e){
       std::cout << "CUDA: cudaWrapper: New pipeline encountered issue during" << bkinfo->flow << ". Exiting with error code for retry!" << std::endl;
       std::cout << e.what() << std::endl;

       exit(-1);

     }

     std::cout << "New pipeline time for flow " << bkinfo->flow << ": " << newPtime.elapsed() << std::endl;

    return true;

#else
  CUDA_WRONG_TIME_AND_PLACE
#endif

    return false;
}
void cudaWrapper::collectHistroyForRegionalFitting(BkgModelWorkInfo* bkinfo, int flowBlockSize, int extractNumFlows)
{
#ifdef ION_COMPILE_CUDA

  if(extractNumFlows == 0) extractNumFlows = flowBlockSize;

  const RawImage * rpt = bkinfo->img->GetImage();
  const SpatialContext * loc = &bkinfo[0].inception_state->loc_context;
  ImgRegParams irP(rpt->cols, rpt->rows, loc->regionXSize, loc->regionYSize);
  int maxFrames = 0;
  for(size_t i=0; i < irP.getNumRegions(); i++)
  {
    int f = bkinfo[i].bkgObj->region_data->time_c.npts();
    maxFrames = (maxFrames <f )?(f):(maxFrames);
  }
  int uncompressedFrames= rpt->uncompFrames;

  //should come from command line:
  int maxNumHistoryFlows = extractNumFlows;
  int HistoryFromCPUPipeline = (flowBlockSize < extractNumFlows)?(flowBlockSize):(extractNumFlows);

  RegionalFitHistory = new HistoryCollection(irP, maxNumHistoryFlows, maxFrames,uncompressedFrames);
  RegionalFitHistory->extractHistoryAllRegionsAllFlows(bkinfo,flowBlockSize,HistoryFromCPUPipeline);



#else
  CUDA_WRONG_TIME_AND_PLACE
#endif

}

void cudaWrapper::mirrorDeviceBuffersToHostForSerialization()
{
#ifdef ION_COMPILE_CUDA

  if(RegionalFitHistory != NULL){
    RegionalFitHistory->CopySerializationDataFromDeviceToHost();
  }
  if(GpuPipeline != NULL){
    GpuPipeline->CopySerializationDataFromDeviceToHost();
  }


#else
  //nothing to do
#endif
}

void cudaWrapper::configureKernelExecution(
    int global_max_flow_key,
    int global_max_flow_max
)
{
#ifdef ION_COMPILE_CUDA
  if(configOpts->gpuMultiFlowFit)
  {
    SimpleMultiFitStream::setBeadsPerBlockMultiF(configOpts->gpuThreadsPerBlockMultiFit);
    SimpleMultiFitStream::setL1SettingMultiF(configOpts->gpuL1ConfigMultiFit);
    SimpleMultiFitStream::setBeadsPerBlockPartialD(configOpts->gpuThreadsPerBlockPartialD);
    SimpleMultiFitStream::setL1SettingPartialD(configOpts->gpuL1ConfigPartialD);
    SimpleMultiFitStream::requestResources(global_max_flow_key, global_max_flow_max, 1.0f);  //0.80f
    SimpleMultiFitStream::printSettings();
  }

  // configure SingleFlowFit Execution
  if(configOpts->gpuSingleFlowFit)
  {
    SimpleSingleFitStream::setBeadsPerBlock(configOpts->gpuThreadsPerBlockSingleFit);
    SimpleSingleFitStream::setL1Setting(configOpts->gpuL1ConfigSingleFit);
    SimpleSingleFitStream::setFitType(configOpts->gpuSingleFlowFitType);
    SimpleSingleFitStream::requestResources(global_max_flow_key, global_max_flow_max, 1.0f); //0.74f
    //SimpleSingleFitStream::setHybridIter(configOpts->gpuHybridIterations);
    SimpleSingleFitStream::printSettings();
  }

  cudaSimpleStreamManager::setNumMaxStreams(configOpts->gpuNumStreams);
  cudaSimpleStreamManager::setVerbose(configOpts->gpuVerbose);
  cudaSimpleStreamExecutionUnit::setVerbose(configOpts->gpuVerbose);

  if(configOpts->gpuDevMemoryPerProc > 0){
    size_t memToRequest = (size_t) configOpts->gpuDevMemoryPerProc * (1024.0*1024)/configOpts->gpuNumStreams;
    if( memToRequest < cudaResourcePool::getRequestDeviceMemory())
      cout << "CUDA WARNING: memory provided for fixed allocation " <<  memToRequest/(1024*1024)<< "MB is less than minimum required memory determined via dynamic allocator: " << cudaResourcePool::getRequestDeviceMemory()/(1024*1024) << "MB. This might lead to memory reallocation during runtime." <<endl;
    cudaResourcePool::setDeviceMemory(memToRequest);
  }

#else
  CUDA_WRONG_TIME_AND_PLACE
#endif
}








