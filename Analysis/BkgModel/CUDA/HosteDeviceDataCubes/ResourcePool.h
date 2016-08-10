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



///////////////////////////////////////////////////////////////////////
// Not Singleton anymore, one per GPU

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

  //TODO: to be removeed after all pointers in Multi Fit got replaced by MemSegment objects
  MemoryResource * getHostMem();
  MemoryResource * getDevMem();

  MemSegment getHostSegment(size_t size);
  MemSegment getDevSegment(size_t size);
  MemSegPair GetHostDevPair(size_t size);


  //returns the size of the current Pair, Host or Device group and resets the group counter
  size_t StartNewSegGroup();
  size_t StartNewHostSegGroup();
  size_t StartNewDeviceSegGroup();
  //accumulates a group of same sized segments to allow copying multiple buffers in one copy
  //the elements of the group have to be acquired consecutively. A new group starts with a call to
  //StartNewPairGroup or with the first call to GetHostDevPair() after a GetHostSegment() or
  //GetDeviceSegment() call to prevent miss matching pairs.
  MemSegPair GetCurrentPairGroup();
  //returns last host/device segment group since last call to StartNewPairGroup, StartNewHostGroup, StartNewDevGroup or the first call to
  //GetHostDevPair after a a GetHostSegment() orGetDeviceSegment() call. To build a group on the Host or the
  //Device that also spans elements in MemSegPairs use the first and last Device or Host segment to create a group through
  //the MemSegment(first, last) constructor.
  MemSegment GetCurrentHostGroup();
  MemSegment GetCurrentDeviceGroup();

  int getStreamId();
  int getDevId();  

  //check if memory resource contains enough memory and is able to reallocate
  bool checkDeviceMemory(size_t size){ return _DMem.checkMemory(size); }

  bool aquire();
  void release();
  bool isSet();

  string Status();

  string getLogHeader();

};



class cudaResourcePool
{

  //int _numHandles;
  vector<streamResources *> _sRes;

  static size_t _SrequestedDeviceSize;
  static size_t _SrequestedHostSize;

  size_t _HostSize;
  size_t _DeviceSize;

  size_t _HostPersistentSize;
  size_t _DeviceersistentSize;

  int _devId;

public:

  cudaResourcePool(int numStreams = MAX_ALLOWED_NUM_STREAMS );
  ~cudaResourcePool();

  streamResources *  getResource();
  void releaseResource(streamResources *& res); 

  void tryAddResource(unsigned int numStreams);   
  int getNumStreams();

  void poolCleaning();

  void printMemoryUsage();
  string getLogHeader();


  static size_t requestDeviceMemory(size_t size);
  static void setDeviceMemory(size_t size);
  static size_t getRequestDeviceMemory(){return _SrequestedDeviceSize; }
  static size_t requestHostMemory(size_t size);

};



#endif //RESOURCEPOOL_H


