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
 // DeviceTexture
};


///////////////////////////////////////////////////////////////
//Memory Segment
//wraps a Host or Device pointer and tracks additional information
//about size and memory type. does not allocate or free any memory
// handles the copy between Host and device according to type
class MemSegment
{
protected:
  char * _basePointer;
  size_t _size;
  MemoryType _type;

public:
  MemSegment();
  MemSegment(void * ptr, size_t size, MemoryType type);
  MemSegment(const MemSegment& src);
  MemSegment(const MemSegment& first, const MemSegment& last);
  virtual ~MemSegment(){};


  size_t addSegment(const MemSegment& append);

  //will inflate/change the segment size by size bytes. no checks are performed to see if there
  //is enough memory pre allocated.
  //it is the programmers responsibility to prevent seg faults by not "over-inflating" the segment
  size_t inflateSegmentBy(size_t size);
  size_t changeSegmentSizeTo(size_t size);

  size_t getSize() {return _size;}

  //size_t size = 0: in the following calls if no size != 0 is provided the segment size is used.

  //copies from or to any kind of memory segment, if no size provided the smaller size is used
  //if stream is provided async copy will be performed if possible
  void copyAsync (const MemSegment& src, cudaStream_t sid, size_t size = 0);
  void copy (const MemSegment& src, size_t size = 0);

  //copies from or to host buffers, if no size provided the segment size is used
  //these function will not check if previously issued asynchronous copies are completed or not!
  void copyIn(const void * hostsrc, size_t size = 0);
  void copyOut(void * hostdst, size_t size = 0);
  //sets whole memory segment to value for each byte
  void memSet(int value, size_t size = 0);
  void memSetAsync(int value, cudaStream_t sid, size_t size = 0);
  //splits the segment, this gets resized to size of offset and the rest offset to end gets returned as new segment
  MemSegment splitAt(size_t offsetbytes);

  //returns segment pointer to be passed to GPU Kernel (should not be used for any other pointer arithmetics)
  void* getVoidPtr(){return (void*)_basePointer;}

  //checks if provided size fits into segment
  size_t checkSize(size_t size);

  void DebugOutput();

private:


  cudaMemcpyKind getCopyKind(const MemSegment& src);
  cudaMemcpyKind getCopyInKind();
  cudaMemcpyKind getCopyOutKind();

};




//////////////////////////////////////////////////////////////////////
//Memory Segment Pair:
//class that handles a pair of buffers, one host side and one device side
//buffer that can be copied asynchronously if the memory allows for it
//some functionality is limited to the Host or the device buffer
//the class does not do any consistency checks between the Host and Device
//buffer. The only check performed is the if the Host and Device side are
//of equal size. if that is not the case an exception of type cudaAllocationError
//is thrown
class MemSegPair
{
protected:
  MemSegment _Host;
  MemSegment _Device;

public:
  MemSegPair();
  MemSegPair(void * HostPtr, void * DevPtr, size_t size, MemoryType Htype=HostMem, MemoryType Dtype=DeviceGlobal);
  MemSegPair(const MemSegPair& src);
  MemSegPair(const MemSegment& host, const MemSegment& device);
  MemSegPair(const MemSegPair& first, const MemSegPair& last);
  virtual ~MemSegPair(){};

  size_t addSegment(const MemSegPair& append);

  //will inflate/change the segment size by size bytes. no checks are performed to see if there
  //is enough memory pre alloacted.
  //it is the programmers responsibity to prevent seg faults by not "overinflating" the segment
  size_t inflateSegmentBy(size_t size);
  size_t changeSegmentSizeTo(size_t size);


  size_t getSize();

  void copyToHostAsync (cudaStream_t sid, size_t size = 0);
  void copyToHost ( size_t size = 0);

  void copyToDeviceAsync (cudaStream_t sid, size_t size = 0);
  void copyToDevice ( size_t size = 0);

  //these function will not check if previously issued asynchronous copies are completed or not!
  void copyIn(const void * hostsrc, size_t size = 0);
  void copyOut(void * hostdst, size_t size = 0);

  //sets whole memory segment to value for each byte
  void memSet(int value);
  //splits the segments, this gets resized to size of offset and the rest offset to end gets returned as new segment

  MemSegPair splitAt(size_t offsetbytes);

  //just in case, should not be needed though
  MemSegment getDeviceSegment(){ return _Device; }
  MemSegment getHostSegment(){ return _Host; }

  //returns segment device pointer to be passed to GPU Kernel (should not be used for any other pointer aritmetics
  //device pointer for kernel use
  void * getVoidPtr(){return _Device.getVoidPtr();}
  //specific request for Host pointer since this should only used for hacky stuff
  void * getVoidHostPtr(){return _Host.getVoidPtr();}

  void DebugOutput();
};



////////////////////////////////////////////////


class MemoryResource
{
  char * _basePointer;
  char * _returnPointer;
  size_t _sizeBytes;

  MemSegment _CurrentSegment;
  MemSegment _CurrentSegGroup;

  MemoryType _type;

protected:
  //checks available device memory if already allocated memory is freed
  size_t checkAvailableDevMem();

public:

  MemoryResource();
  MemoryResource(size_t size, MemoryType type);
  ~MemoryResource();

  //static alloc and free functions
  static void* allocate(size_t size, MemoryType type);
  static void destroy(void * ptr, MemoryType type);

  bool reallocate(size_t, MemoryType type = UnDefined); // if no type given will try to realloc original type if valid
  void destroy(); // will free all allocatede memory but remember the type

  void * getSegment(size_t size);
  MemSegment getMemSegment(size_t size);
  MemSegment getCurrentSegGroup();

  void releaseAll();

  size_t startNewSegGroup();
  size_t getCurrentSegGroupSize();
  size_t getSize();
  size_t memoryUsed();
  size_t memoryAvailable();
  MemoryType getMemType();

  //check if memory resource contains enough memory and is able to reallocate
  bool checkMemory(size_t size); //throws exception if memory insufficient and reallocation not possible

  bool isSet();

  string getLogHeader();

};



/////////////////////////////////////////////////////
//TEMPLATE WRAPPER
//Templated Memory Segment.
//Allows for type specific access to a MemSegment Object.
//adds some type specific function and overloads the [] operator for
//array like access to host side arrays.
//throws cudaExecutionException if MemSegment is Device Memory
//and [] access is performed
template<typename T>
class TMemSegment : public MemSegment
{
public:
  TMemSegment():MemSegment(){};
  TMemSegment(T * ptr, size_t size, MemoryType type):MemSegment(ptr,size,type){};
  TMemSegment(const MemSegment& src):MemSegment(src){};
  TMemSegment(const MemSegment& first, const MemSegment& last):MemSegment(first,last){};
  virtual ~TMemSegment(){};

  T* getPtr(){return (T*)getVoidPtr();}

  T& operator[](size_t idx){
       if(_type != HostMem && _type != HostPageLocked ){
         cout << "CUDA Memory Manager Error: Access to Device Memory via [] operator not possible!" << endl;
         throw cudaExecutionException(cudaErrorInvalidHostPointer, __FILE__, __LINE__);
       }
       T* ptr = (T*)getVoidPtr();
       return ptr[idx];
  }

  //splits the segment, this gets resized to size of #idx of type T. the rest gets returned as new segment without type
  MemSegment splitAtIdx(size_t idx){ return splitAt(idx * sizeof(T)); }
  size_t numElements(){ return getSize()/sizeof(T); }

};






/////////////////////////////////////////////////////
//TEMPLATE WRAPPER
//Templated Memory Segment Pair.
//Allows for type specific access to a MemSegPair Object.
//ads some type specific function and overloads the [] operator for
//array like access of the Host side data.
template<typename T>
class TMemSegPair : public MemSegPair
{
public:
  TMemSegPair():MemSegPair(){};
  TMemSegPair(T * HostPtr, T * DevPtr, size_t size, MemoryType Htype=HostMem, MemoryType Dtype=DeviceGlobal):MemSegPair(HostPtr,DevPtr,size,Htype,Dtype){};
  TMemSegPair(const MemSegPair& src):MemSegPair(src){};
  TMemSegPair(const MemSegment& host, const MemSegment& device):MemSegPair(host,device){};
  TMemSegPair(const MemSegPair& first, const MemSegPair& last):MemSegPair(first,last){};
  virtual ~TMemSegPair(){};

  //returns device pointer e.g. for kernel execution
  T* getPtr(){return (T*)getVoidPtr();}
  //access to host pointer should not really be needed but for e.g. debugging here it is
  T* getHostPtr(){return (T*)getVoidHostPtr();}

  //by default return host array, mostly for debugging purpose or host side data update
  //so that host array is at same state as device array is the programmers responsibility
  T& operator[](size_t idx){
           TMemSegment<T> tmpSeg = _Host; // wrap into type specific template for [] access
           return tmpSeg[idx];
  }

  //splits the segment, this gets resized to size of #idx of type T. the rest gets returned as new segment without type
  MemSegPair splitAtIdx(size_t idx){ return splitAt(idx * sizeof(T)); }
  size_t numElements(){ return getSize()/sizeof(T); }

};






/////////////////////////////////////////////////////
//TEMPLATE memory allocator
//actually allocates the designated memory and frees it with destruction
//meant for debugging but also might be useful in other situations
template<typename T>
class TMemSegAlloc : public TMemSegment<T>
{

public:


  TMemSegAlloc(size_t size, MemoryType type)
  {
    TMemSegment<T>::_basePointer = MemoryResource::allocate(size,type);
    *this = TMemSegment<T>(TMemSegment<T>::_basePointer,size,type);     //create segment wrapper around allocated memory
  }

  virtual ~TMemSegAlloc()
  {
    MemoryResource::destroy(TMemSegment<T>::_basePointer,TMemSegment<T>::_type );
  }

};

///////////////////
//TEMPLATE using memory allocator
//creates two TMemSegAlloc objects that actually allocate and free the designated memory
//meant for debugging but also might be useful in other situations
template<typename T>
class TMemSegPairAlloc : public TMemSegPair<T>
{
  TMemSegPairAlloc(size_t size, MemoryType Htype=HostPageLocked, MemoryType Dtype=DeviceGlobal)
  {
    TMemSegPair<T>::_Host = TMemSegAlloc<T>(size,Htype);
    TMemSegPair<T>::_Device = TMemSegAlloc<T>(size,Dtype);
  };

  virtual ~TMemSegPairAlloc(){};

};

#endif //MEMORYMANAGER_H
