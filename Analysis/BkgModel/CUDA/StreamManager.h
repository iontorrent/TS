/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef STREAMMANAGER_H
#define STREAMMANAGER_H

// std headers
#include <iostream>
// cuda
#include "WorkerInfoQueue.h"
#include "BkgMagicDefines.h"
#include "cuda_runtime.h"
#include "cuda_error.h"
#include "Utils.h"
#include "CudaDefines.h"


//#define MAX_NUM_STREAMS 2

//#define TASK_TIMING

#define MAX_NUM_DEVICES 4

//#define BLOCKING_EVENT

/////////////////
//Singleton Class that manages a pool af MAX_NUM_STREAMS stream for up to MAX_NUM_DEVICES cuda devices
// the streams for a device get created when the first thread tries to obtain a stream for that device from the Pool
// the created streams on a given device get destroyed when the last stream for that device is released.

class cudaStreamPool
{

  static cudaStreamPool * _pInstance;

  static pthread_mutex_t _lock;
  static bool _init;

  int _numHandles[MAX_NUM_DEVICES];
  cudaStream_t _streams[MAX_NUM_DEVICES][MAX_NUM_STREAMS];
  bool _inUse[MAX_NUM_DEVICES][MAX_NUM_STREAMS];

  static int _devId; // carefull! is set when obtaining the lock and is no longer valid after release of Lock 

  //private constructors -> Singleton
  cudaStreamPool();
  cudaStreamPool(cudaStreamPool const&);  
  cudaStreamPool& operator=(cudaStreamPool const&);

  void createStreams();
  void destroyStreams();

  static void Lock();
  static void UnLock();
 
public:

  static void initLockNotThreadSafe();  //has to be called before multithreaded part to be thread safe;
  static void destroyLockNotThreadSafe();  //has to be called before multithreaded part to be thread safe;

  static cudaStreamPool * Instance();

  cudaStream_t getStream();
  void releaseStream(cudaStream_t); 

};




enum cudaStreamState 
{
  GetJob,
  Working,
  Waiting,
  ContinueWork,
  Sleeping,
  Blocking,
  Sync,
  Exit
};


// shared data object stored in the stream manager to get rid of 
// static variables to allow for multiple instances of StreamManagers
class sharedStreamData
{
  int _seuCounter;
  int _seuWorking;
  int _numBlock;
  int _tasksComplete;    
  bool _done;
  bool _sleepMask[MAX_NUM_STREAMS];


  cudaEvent_t _event;
  bool _waitEventMask[MAX_NUM_STREAMS];

  
 
  double _timesum;
  Timer _T;
  
  WorkerInfoQueue * _inQ; 

  public:
    
  sharedStreamData();
  ~sharedStreamData();

  int incSEUcnt();
  int decSEUcnt();

  bool setQ(WorkerInfoQueue * Q);
  bool isSet();
 
  void setSleeping(int id);
  void wakeUpCall();

  bool isSleeping(int id);
  bool allSleeping();

#ifdef BLOCKING_EVENT
  void createEvent();
  void setEvent(int id, cudaStream_t stream);
  void waitEvent(int id);
  bool allWaiting();
  void clearEvent(int id);
  void clearEvent();
#endif

  void setDone();
  bool isDone();

  // timing and control function
 
  void incTasksComplete();
  int getTasksComplete(); 

  void startTimer();
  void stopTimer();
  double getBusyTime();
  double getAvgTimePerJob();
};


class cudaStreamExecutionUnit
{

// sharedStreamData * _sd;

protected:
sharedStreamData * _sd;

   
  int _seuId;
  int _tasks;
  static bool _verbose;
  char _name[128];

  WorkerInfoQueue * _inQ; 
  WorkerInfoQueueItem _item;

  cudaStreamPool * _streamPool;
	
  cudaStream_t _stream;
 
  cudaStreamState _state;

public:

	cudaStreamExecutionUnit(WorkerInfoQueue * Q );
//	cudaStreamExecutionUnit();
	virtual ~cudaStreamExecutionUnit();

  //retuns true if wokr to do, and falase if not
  bool execute(int * control = NULL);

  bool getNewJob();  //returns true if job available
  void getNewJobBlocking();  //blocks while no Job available
  bool checkComplete();  


  void * getJobData();
  bool noMoreWork();

  virtual bool ValidJob() = 0; // Sets the job for the fitter object and checks if valid
  /*virtual void prepareInputs() = 0; // if needed move host data to page locked buffers or reorganize   
  virtual void copyToDevice() = 0;  // has to be implemented in a non blocking/async way
  virtual void executeKernel()= 0; // has to be implemented in a non blocking/async way
  virtual void copyToHost() = 0;  // has to be implemented in a non blocking/async way*/
  virtual int handleResults() = 0; //handles host side results, when async work is completed. returns 0 when done. if retunr != 0 go back to working and don't finish task
  virtual void ExecuteJob(int * control = NULL) = 0;
  
   void init(int id, sharedStreamData *sd);
  //void setId(int id); // only to be used by Stream Manager

  void setName(char *);
  char * getName();
  void  printInfo();
  int getId();
  //int getNumBlocks();
  int getNumTasks();


  
  static void setVerbose(bool flag);
  static bool Verbose();

};






class cudaStreamManager
{

  sharedStreamData _sd[MAX_NUM_STREAMS];
  cudaStreamExecutionUnit * _Stream[MAX_NUM_STREAMS];
  int _numStreams;
  int _dev_id;
 // int _tasks;
 // int _blocked;
 
public:

  cudaStreamManager();
  //cudaStreamManager( WorkerInfoQueue * inQ, int maxBeads, int blocksize, int howMany = 1);

  ~cudaStreamManager();
  

  int getNumStreams(); 
 
  void destroyStreams();  

  bool DoWork(int * control = NULL);

  //int createStreamUnit(WorkerInfoQueue * inQ, int maxBeads, int blocksize, int howMany = 1);  
  int addStreamUnit( cudaStreamExecutionUnit* seu);  


//protected:
// meant for internal use will be protected after testing 
  //int addStreamUnit(StreamExecutionUnit * StExUn);
  void destroyStream(int id);

  static void printMemoryUsage();


};





#endif //STREAMMANAGER_H
