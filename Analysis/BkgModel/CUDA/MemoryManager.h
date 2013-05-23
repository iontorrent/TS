/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef MEMORYMANAGER_H
#define MEMORYMANAGER_H

// std headers
#include <iostream>
#include <string>
#include <sstream>
// cuda
#include "cuda_runtime.h"
#include "cuda_error.h"
#include "Utils.h"
#include "CudaDefines.h"


enum MemoryType{
  UnDefined,
  HostMem,
  HostPageLocked,
  DeviceGlobal,
  DeviceTexture
};


class MemoryResource
{
  char * _basePointer;
  char * _returnPointer;
  size_t _sizeBytes;
  size_t _accumilatedSize;
 
  MemoryType _type;

protected:
  //checks available device memory if already allocated memory is freed 
  size_t checkAvailableDevMem();

public:

  MemoryResource();
  MemoryResource(size_t size, MemoryType type);
  ~MemoryResource();

  bool reallocate(size_t, MemoryType type = UnDefined); // if no type given will try to realloc original type if valid
  void destroy(); // will free all allocatede memory but remember the type

  void * getSegment(size_t size);
  void releaseAll();

  size_t startNewSegGroup();
  size_t getCurrentSegGroupSize();
  size_t memoryAvailable();

  //check if memory resource contains enough memory and is able to reallocate
  bool checkMemory(size_t size); //throws exception if memory insufficient and reallocation not possible

  bool isSet();

  string getLogHeader();

};





#endif //MEMORYMANAGER_H
