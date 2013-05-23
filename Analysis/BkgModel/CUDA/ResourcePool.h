/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef RESOURCEPOOL_H
#define RESOURCEPOOL_H

// std headers
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
// cuda
#include "WorkerInfoQueue.h"
#include "BkgMagicDefines.h"
#include "cuda_runtime.h"
#include "cuda_error.h"
#include "Utils.h"
#include "CudaDefines.h"
#include "MemoryManager.h"


/////////////////
//Singleton Class that manages a pool af MAX_NUM_STREAMS stream for up to MAX_NUM_DEVICES cuda devices
// the streams for a device get created when the first thread tries to obtain a stream for that device from the Pool
// the created streams on a given device get destroyed when the last stream for that device is released.
/*
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

*/

///////////////////////////////////////////////////////////////////////
// Not Singleton, one per GPU 

class streamResources
{
  
  bool _inUse;
  int _streamId;
  MemoryResource _HMem;
  MemoryResource _DMem;
  cudaStream_t _stream;
  int _devId;

public:


  streamResources(size_t hostSize, size_t deviceSize , int id);
  ~streamResources();

  cudaStream_t getStream();
  MemoryResource * getHostMem();
  MemoryResource * getDevMem();
  int getStreamId();
  int getDevId();  


  bool aquire();
  void release();
  bool isSet();
   
  string getLogHeader();

};



class cudaResourcePool
{

  //int _numHandles;
  vector<streamResources *> _sRes;

  size_t _HostSize;
  size_t _DeviceSize;
  int _devId;

public:

  cudaResourcePool(size_t hostsize, size_t devicesize, int numStreams = MAX_NUM_STREAMS);
  ~cudaResourcePool();

  streamResources *  getResource();
  void releaseResource(streamResources *& res); 

  void tryAddResource(unsigned int numStreams);   
  int getNumStreams();

  void poolCleaning();

  void printMemoryUsage();
  string getLogHeader();

};



#endif //RESOURCEPOOL_H


