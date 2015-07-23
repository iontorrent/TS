/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef STREAMMANAGER_H
#define STREAMMANAGER_H

// std headers
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
// cuda

#include "WorkerInfoQueue.h"
#include "BkgMagicDefines.h"
#include "cuda_runtime.h"
#include "cuda_error.h"
#include "Utils.h"
#include "CudaDefines.h"
#include "ResourcePool.h"

//#define TASK_TIMING

//#define BLOCKING_EVENT


enum cudaStreamState 
{
  Init = 0,
  GetJob,  // > GetJob 
  Working,
  Waiting,  // states between GetJob and Sleeping hold a valid job reference
  ContinueWork,
  Sleeping, // < Sleeping 
  Blocking,
  Sync,
  Exit
};

////////////////////////////////////
/// SIMPLE IMPLEMENTATION


class TimeKeeper{

  Timer _T;
  double _timesum;
  int _activeCnt;
  int _jobCnt;
  int _errCnt;

public:
  
  TimeKeeper();

  void start();
  void stop();
  void stopAfterError();
  double getTime();
  double getAvgTime();
  int getJobCnt();
  int getErrorCnt();

};





class cudaSimpleStreamExecutionUnit
{


protected:
  
  static int _seuCnt;
  int _seuNum;
  int _computeVersion;

  string _name;

  static bool _verbose;
  
  WorkerInfoQueueItem _item;

  streamResources * _resource;

  cudaStream_t _stream;

  cudaStreamState _state;
  

public:

  cudaSimpleStreamExecutionUnit( streamResources * resources,  WorkerInfoQueueItem item );
//  cudaStreamExecutionUnit();
  virtual ~cudaSimpleStreamExecutionUnit();

  //retuns true if work to do, and false if not
  bool execute();

  void * getJobData();
  bool checkComplete();  
 
  void setName(std::string);
  string getName();
  string getLogHeader();

  int getStreamId();
  int getSeuNum();
  
  void setCompute(int compute);
  int getCompute();

  WorkerInfoQueueItem getItem();
//  void signalErrorToResource(); TODO

 // int getNumFrames();
//  int getNumBeads();


////////////////////////
// virtual interface functions needed for Stream Execution

 //inits job from the private_data field in the item
 //returns true if job is a valid job
  virtual bool InitJob() ;

// function should contain data preparation
// followed by all async cuda calls needed to execute the job
// 1. prepare data on host
// 2. copy async to device
// 3. call kernels
// 4  copy async back to host
//
// ExecuteJob will be executed at least once 
// plus one more time for every time handleResults() returns a vale != 0
// see Multiflowfit for example
  virtual void ExecuteJob() = 0;

//handles host side results, when async work is completed. returns 0 when done. 
//if return != 0 go back to ExecuteJob since more async work needs to be 
//performed. when done this method hands the item on to the next queue, provided
//either inside the item or the derived SEU class.
  virtual int handleResults() = 0; 

//this method is called by the streamManager in case of an exception during the SEU execution
//implementation should print a summary of all relevant input data sizes/values which could
//help to determine the reason for an error.
  virtual void printStatus() = 0;
////////////////////////


  //Factory Method to produce job specific Stream Execution Units
  static cudaSimpleStreamExecutionUnit * makeExecutionUnit(streamResources * resources, WorkerInfoQueueItem item);


// static helpers
  static size_t paddTo(size_t n, size_t padding)
  {
    return ((n+padding-1)/padding)*padding;
  }

  static void setVerbose(bool v);
  static bool Verbose();



};


//////////////////////////////////////////////////////////////////////////////////////
// StreamManager


class cudaSimpleStreamManager
{

  static bool _verbose;
  static int _maxNumStreams;

  int _executionErrorCount;

  WorkerInfoQueue * _inQ;
  WorkerInfoQueue * _fallBackQ;

  WorkerInfoQueueItem _item;
  cudaResourcePool * _resourcePool;
  bool _resourceAllocDone;

  vector<cudaSimpleStreamExecutionUnit *> _activeSEU;
//  map<int,int> _taskcounter;
  int _tasks;
  int _devId;
  int _computeVersion;
  map<string, TimeKeeper> _timer; //keep separate timer for each SEU type, distinguished by SEU name
  
  bool _GPUerror;

  // max and sum of all beads/frames of all jobs handled
  size_t _sumBeads;
  int _maxBeads;
  size_t _sumFrames;
  int  _maxFrames;
    
protected:


  //size_t getMaxHostSize(int flow_block_size);
  //size_t getMaxDeviceSize(int maxFrames /*= 0*/, int maxBeads /*= 0*/, int flow_block_size);

  void allocateResources();
  void freeResources();
 

  int availableResources(); 
  void getJob();
  void addSEU();
  void executeSEU();
  void moveToCPU();
  
  bool executionComplete(cudaSimpleStreamExecutionUnit* seu );

  //bookkeeping
  void recordBeads(int n);
  void recordFrames(int n);
 
  bool checkItem(); //checks if job item is a valid job
  bool isFinishItem(); //returns true if finish item

public:

  //factory method to produce job type specific SEUs
  cudaSimpleStreamManager( WorkerInfoQueue * inQ, WorkerInfoQueue * fallbackQ );
  ~cudaSimpleStreamManager();

  bool DoWork();
 
  int getNumStreams(); //returns actual number of stream
  void printMemoryUsage();
/*
  void startTimer();
  void stopTimer();
  double getBusyTime();
  double getAvgTimePerJob();
*/

  string getLogHeader();

////////////////////////
// static helpers
  static void setNumMaxStreams(int numStreams);
  static int getNumMaxStreams();
  static void setVerbose(bool v);


};


#endif //STREAMMANAGER_H
