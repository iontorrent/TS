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
#include "JobWrapper.h"
#include "WorkerInfoQueue.h"
#include "BkgMagicDefines.h"
#include "cuda_runtime.h"
#include "cuda_error.h"
#include "Utils.h"
#include "CudaDefines.h"
#include "ResourcePool.h"

//#define MAX_NUM_STREAMS 2

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

public:
  
  TimeKeeper();

  void start();
  void stop();
  double getTime();
  double getAvgTime();
  int getJobCnt();

};





class cudaSimpleStreamExecutionUnit
{

// sharedStreamData * _sd;
  
protected:
  
  static int _seuCnt;
  int _seuNum;
  int _computeVersion;

  string _name;

  static bool _verbose;
  
  WorkerInfoQueueItem _item;

  WorkSet _myJob;

  streamResources * _resources;

  cudaStream_t _stream;
  MemoryResource * _Host;
  MemoryResource * _Device;

  cudaStreamState _state;
  

public:

	cudaSimpleStreamExecutionUnit( streamResources * resources,  WorkerInfoQueueItem item );
//	cudaStreamExecutionUnit();
	virtual ~cudaSimpleStreamExecutionUnit();

  //retuns true if work to do, and false if not
  bool execute();

  void * getJobData();
  bool checkComplete();  
 
  void setName(char *);
  string getName();
  string getLogHeader();

  int getStreamId();
  int getSeuNum();
  
  void setCompute(int compute);
  int getCompute();

  WorkerInfoQueueItem getItem();
//  void signalErrorToResource(); TODO

  bool InitValidateJob(); 

////////////////////////
// virtual interface functions needed for Stream Execution



// function should contain data preperation 
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
//performed
  virtual int handleResults() = 0; 


////////////////////////
// static helpers   
  static void setVerbose(bool flag);
  static bool Verbose();

};


//////////////////////////////////////////////////////////////////////////////////////
// StreamManager


class cudaSimpleStreamManager
{

  static bool _verbose;

  WorkerInfoQueue * _inQ;

  WorkerInfoQueueItem _item;
  cudaResourcePool * _resourcePool;

  vector<cudaSimpleStreamExecutionUnit *> _activeSEU;
//  map<int,int> _taskcounter;
  int _tasks;
  int _devId;
  int _computeVersion;
  int _maxNumStreams;
  map<string, TimeKeeper> _timer;
  
  bool _GPUerror;  


protected:

  size_t getMaxHostSize();
  size_t getMaxDeviceSize(int maxFrames = 0, int maxBeads = 0);

  void allocateResources();
  void freeResources();
 

  int availableResources(); 
  void getJob();
  void addSEU();
  void executeSEU();
  void moveToCPU();
  
  bool executionComplete(cudaSimpleStreamExecutionUnit* seu );
 
public:

  cudaSimpleStreamManager( WorkerInfoQueue * inQ, int numStreams = MAX_NUM_STREAMS);
  ~cudaSimpleStreamManager();
  

  bool DoWork();


 
  int getNumStreams();
  void printMemoryUsage();
/*
  void startTimer();
  void stopTimer();
  double getBusyTime();
  double getAvgTimePerJob();
*/

  string getLogHeader();


};


#endif //STREAMMANAGER_H
