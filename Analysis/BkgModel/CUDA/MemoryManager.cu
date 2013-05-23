/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

// patch for CUDA5.0/GCC4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <iostream>

#include "MemoryManager.h"




MemoryResource::MemoryResource()
{
  _sizeBytes = 0;
  _type = UnDefined;
  _basePointer = NULL;
  _returnPointer = NULL;
  _accumilatedSize = 0;
}

MemoryResource::MemoryResource(size_t size, MemoryType type)
{
  _sizeBytes = 0;
  _type = type;
  _basePointer = NULL;
  _returnPointer = NULL;
  _accumilatedSize = 0;

  reallocate(size, type);

}


MemoryResource::~MemoryResource()
{
  destroy();
}


bool MemoryResource::reallocate(size_t size, MemoryType type)
{
  //if already allocated and reallocation is needed cleanup first
  if(_sizeBytes > 0 || _basePointer != NULL) destroy();

  _sizeBytes = size;
  _type = (type == UnDefined)?(_type):(type); 
  int devId;
  cudaGetDevice(&devId);
  cudaError_t err = cudaSuccess;
    switch(_type){
      case HostPageLocked:
//      cout << getLogHeader() << " creating Host memory " << _sizeBytes  << endl;
        cudaHostAlloc(&_basePointer, _sizeBytes, cudaHostAllocDefault); 
        break;
      case DeviceGlobal:
//        cout << getLogHeader() << " creating Device memory " << _sizeBytes  << endl;
        cudaMalloc(&_basePointer, _sizeBytes);  
        break;
      case HostMem:
        _basePointer =(char*)malloc(_sizeBytes);
        break;
      case UnDefined:
      default:
        _basePointer = NULL;
    }
 
  err = cudaGetLastError();

  if ( err != cudaSuccess || _basePointer == NULL){
    _basePointer = NULL;
    _sizeBytes = 0;
    _basePointer = NULL;
    throw cudaAllocationError(err, __FILE__, __LINE__);                              
  } 
  _returnPointer = _basePointer;
  _accumilatedSize = 0;
  
  return true;

}


void MemoryResource::destroy()
{

  if(_basePointer != NULL){ 
    switch(_type){
      case HostMem:
        free(_basePointer);
        break;
      case HostPageLocked:
        cudaFreeHost(_basePointer); //CUDA_ERROR_CHECK();
        break;
      case DeviceGlobal:
        cudaFree(_basePointer); //CUDA_ERROR_CHECK();
        break;
      case UnDefined:
      default:
        break;
    }
  }

  _sizeBytes = 0;
  _basePointer = NULL;
  _returnPointer = NULL;
  _accumilatedSize = 0;

}


void * MemoryResource::getSegment(size_t size)
{
 
  if(_type == UnDefined || _basePointer == NULL)   throw cudaNotEnoughMemForStream(__FILE__,__LINE__);
  //return NULL;
  size_t segSize = ((size + 128 -1)/128) *128; 
/*
    switch(_type){
      case DeviceGlobal:
        cout << "CUDA: getSegment Device " << segSize <<" "  << _accumilatedSize <<  " free:"  << memoryAvailable() << endl;
        break;
      default:
        cout << "CUDA: getSegment Host   " << segSize <<" "  << _accumilatedSize <<  " free:"  << memoryAvailable()  << endl;
    }
*/
 
  if(_returnPointer + segSize <= _basePointer + _sizeBytes ){
    void * ret = (void*)_returnPointer; 
    _returnPointer += segSize;
    _accumilatedSize += segSize;
//    _returnPointer = ((_returnPointer+128-1)/128)*128;se

    return ret;
  }
  throw cudaNotEnoughMemForStream(__FILE__,__LINE__);
//    return NULL;
}

void MemoryResource::releaseAll()
{
  _returnPointer = _basePointer;

}

size_t MemoryResource::startNewSegGroup()
{
  size_t ret = _accumilatedSize;
  _accumilatedSize = 0;
  return ret;
}

size_t MemoryResource::getCurrentSegGroupSize()
{
  return _accumilatedSize;
}


size_t MemoryResource::memoryAvailable()
{
 return _sizeBytes - ((size_t)_returnPointer - (size_t)_basePointer);
}

size_t MemoryResource::checkAvailableDevMem()
{
  size_t free_byte ;
  size_t total_byte ;
  cudaMemGetInfo( &free_byte, &total_byte ) ;
  free_byte += memoryAvailable();
  return free_byte;
}

//check if memory resource contains enough memory and is able to reallocate
bool MemoryResource::checkMemory(size_t size)
{
  
  if(size >  _sizeBytes){

    if(_type == DeviceGlobal && false){  // DONT DO PRE CHECK ANYMORE SO ALLOCATION FAILS


      size_t free_byte  = checkAvailableDevMem();
      free_byte -=50*1024*1024; //allow for 50 meg buffer that might not  be avaialble because of context
      if (free_byte < size){  
        cout << getLogHeader() << " **** NOT ENOUGH MEMORY ON DEVICE TO REALLOCATED: " << size << " bytes needed, " << free_byte << " bytes available, no realocation performed, propagating up." << endl;
        
        throw cudaNotEnoughMemForStream(__FILE__,__LINE__);
      }
    }
    cout << getLogHeader() << " **** NOT ENOUGH MEMORY PREALLOCATED: " << _sizeBytes <<" bytes, try reallocating " << size << " bytes for current Job" << endl;
    reallocate(size);
    return false;
  }

  return true;
}


bool MemoryResource::isSet()
{
  if(_type == UnDefined) return false;
  if(_sizeBytes <= 0) return false;
  if(_basePointer == NULL) return false;
  return true;
}



string MemoryResource::getLogHeader()
{
  ostringstream headerinfo;

  int devId;
  cudaGetDevice(&devId);

  headerinfo << "CUDA " << devId << ":";

  return headerinfo.str();
}





