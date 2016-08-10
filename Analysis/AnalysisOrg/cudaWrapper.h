/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef CUDAWRAPPER_H
#define CUDAWRAPPER_H

#include "WorkerInfoQueue.h"
#include "BkgModel/MathModel/MathOptim.h"
#include "GpuControlOpts.h"
#include "SignalProcessingFitterQueue.h"

#include "pThreadWrapper.h"
#include <vector>

//#define ION_COMPILE_CUDA

class BkgGpuPipeline;
class BkgModelWorkInfo;

#ifdef ION_COMPILE_CUDA
#include "SampleHistory.h"
//class HistoryCollection;
#endif



/*************************************************
* class gpuDeviceConfig
*
* manages the available cuda devices in the system
* allows for specific configurations/selection of GPU resources
* creates CUDA context on selected devices by initializing
* some constant memory
*
**************************************************/
class gpuDeviceConfig
{

  //hardware configuration
  int maxComputeVersion;
  int minComputeVersion;
  //int minDeviceMemory;
  std::vector<int> validDevices;



protected:

  int getVersion(int devId);
  int getMaxVersion(std::vector<int> &DeviceIds);

  void applyMemoryConstraint(size_t minBytes);
  void applyComputeConstraint(int minVersion); //23 = 2.3, 35 = 3.5 etc

  void initDeviceContexts();

public:

  gpuDeviceConfig();

  bool  setValidDevices(std::vector<int> &CmdlineDeviceList, bool useMaxComputeVersionOnly);  //if useMaxComputeVersionOnly == false:  any CUDA compatible device will be used without checking for copute version
  //std::vector<int> & getValidDevices(){return validDevices;}
  int getNumValidDevices(){return validDevices.size(); }
  const std::vector<int> getValidDevices(){return validDevices;}
  int getFirstValidDevice(){return validDevices[0];}

};



/*************************************************
* class queueWorkerThread
*
* manages a cpu thread which dequeues a PJob from a
* WorkerInfoQueue, runs it and marks it as dequeued
* when completed
*
**************************************************/
class queueWorkerThread: protected pThreadWrapper
{
  WorkerInfoQueue *wq;
protected:
  virtual void InternalThreadFunction();
public:
  queueWorkerThread(WorkerInfoQueue *q);
  bool start();
  void join();
};



/*************************************************
* class flowByFlowHandshaker
*
* manages the thread, the ring-buffer
* and handles to the resources needed for handshake
* between flow by flow Gpu pipeline and wells file writing
*
**************************************************/
class flowByFlowHandshaker : protected pThreadWrapper
{
    int startingFlow;
    int endingFlow;
    SemQueue *packQueue;
    SemQueue *writeQueue;
    ChunkyWells *rawWells;
    std::vector<SignalProcessingMasterFitter*> *fitters;

    RingBuffer<float> *ampEstimatesBuf;

protected:

  virtual void InternalThreadFunction();

public:
  flowByFlowHandshaker();
  flowByFlowHandshaker(SemQueue * ppackQueue, SemQueue * pwriteQueue,ChunkyWells * pRawWells, std::vector<SignalProcessingMasterFitter*> * pfitters);
  ~flowByFlowHandshaker();

  void setSemQueues(SemQueue * ppackQueue, SemQueue * pwriteQueue){packQueue=ppackQueue; writeQueue=pwriteQueue;}
  void setChunkyWells(ChunkyWells * pRawWells) {rawWells = pRawWells;}
  void setFitters(std::vector<SignalProcessingMasterFitter*> *pfitters){fitters = pfitters;}
  void setStartFlow(int startF){startingFlow = startF;}
  void setEndFlow(int endF){ endingFlow = endF; }
  void setStartEndFlow(int startF, int endF){startingFlow = startF; endingFlow = endF; }

  void CreateRingBuffer( int numBuffers, int bufSize); //bufSize = num floats per image: cols*rows
  RingBuffer<float> * getRingBuffer(){return ampEstimatesBuf; }

  bool start();
  void join();





};

/*************************************************
* class gpuBkgFitWorker
*
* one GPU Worker Thread for the block of 20 flow
* per region GPU pipeline
* class holds all the information the thread needs
* to execute and also maintains information about
* the GPU the thread is working on
**************************************************/
class gpuBkgFitWorker: protected pThreadWrapper{

  int devId; //physical device Id
  WorkerInfoQueue* q;
  WorkerInfoQueue * errorQueue;

protected:


  void createStreamManager();
  virtual void InternalThreadFunction();


public:

  gpuBkgFitWorker(int devId);

  void setWorkQueue(WorkerInfoQueue* Queue){ q = Queue ;};
  void setFallBackQueue(WorkerInfoQueue * fallbackQueue){errorQueue = fallbackQueue;} //provide queue to put job into if gpu execution fails

  bool start();
  void join();


};




/*************************************************
* class cudaWrapper
*
* manages the interface between CPU code and GPU code
* handles gpu configuration and resources
* provides interface for GPU pipelines
*
**************************************************/
class cudaWrapper{


  //execution configuartion
  bool useGpu;
  bool useAllGpus;
  gpuDeviceConfig deviceConfig;
  //old pipeline resources
  WorkerInfoQueue * workQueue;
  std::vector<gpuBkgFitWorker*> BkgWorkerThreads;

  GpuControlOpts * configOpts;

//new pipeline resources
#ifdef ION_COMPILE_CUDA
  HistoryCollection * RegionalFitHistory;
  BkgGpuPipeline * GpuPipeline;
#endif
  flowByFlowHandshaker * Handshaker;


protected:

   bool checkChipSize();
   void destroyQueue();

public:

  cudaWrapper();
  ~cudaWrapper();

  //configures the devices and creates contexts through gpuDeviceConfig class
  void configureGpu(BkgModelControlOpts &bkg_control);

  int getNumWorkers(){ return BkgWorkerThreads.size();} //returns actual number of currently active gpu workers
  int getNumValidDevices(){ return deviceConfig.getNumValidDevices();}

  //check flags
  bool useGpuAcceleration() { return useGpu; }

  void configureKernelExecution( int global_max_flow_key, int global_max_flow_max );


  void createQueue(int numRegions);


//old pipeline control
  WorkerInfoQueue * getQueue();
  bool SpinUpThreads(WorkerInfoQueue* fallbackCPUQ);
  void UnSpinThreads();


  void setUpAndStartFlowByFlowHandshakeWorker(  const CommandLineOpts &inception_state,
                                                const ImageSpecClass &my_image_spec,
                                                std::vector<SignalProcessingMasterFitter*> * fitters,
                                                SemQueue *packQueue,
                                                SemQueue *writeQueue,
                                                ChunkyWells *rawWells,
                                                int startingFlow);

  void collectHistroyForRegionalFitting(BkgModelWorkInfo* bkinfo, int flowBlockSize, int extractNumFlows);
  void mirrorDeviceBuffersToHostForSerialization();


  void joinFlowByFlowHandshakeWorker();


  int switchAtFlow() { return configOpts->switchToFlowByFlowAt; }
  bool usePostFitHandshakeWorker() { return configOpts->postFitHandshakeWorker; }
  bool handshakeCreated() { return (Handshaker != NULL); }
  bool useFlowByFlowExecution() { return configOpts->gpuFlowByFlowExecution; };
  bool isCurrentFlowExecutedAsFlowByFlow(int currentFlow) { return ( useFlowByFlowExecution() && currentFlow >= switchAtFlow()); }
  bool ampGuessOnGpu(){ return (configOpts->gpuSingleFlowFit && configOpts->gpuAmpGuess); }

  bool checkIfInitFlowByFlow(int currentFlow, bool restart);
  bool fullBlockSignalProcessing(BkgModelWorkInfo* bkinfo);



private:
  // Serialization section
 friend class boost::serialization::access;
  template<typename Archive>
   void load(Archive& ar, const unsigned version)
   {
#ifdef ION_COMPILE_CUDA
    ar  & RegionalFitHistory;
#endif


   }
 template<typename Archive>
   void save(Archive& ar, const unsigned version) const
   {
#ifdef ION_COMPILE_CUDA
     cout << "STORE STORE STORE cudaWrapper " <<  endl;
     ar &  RegionalFitHistory;
#endif
   }

 BOOST_SERIALIZATION_SPLIT_MEMBER()


};








#endif // CUDAWRAPPER_H
