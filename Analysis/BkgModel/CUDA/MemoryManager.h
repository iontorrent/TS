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
#include <set>
#include <map>






enum MemoryType{
  UnDefined,
  HostMem,
  HostPageLocked,
  DeviceGlobal
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

  size_t readSizeFromFile(ifstream& myFile);
  MemoryType readMemTypeFromFile(ifstream& myFile);
  void readBufferFromFile(ifstream& myFile, size_t readsize=0);


private:

  cudaMemcpyKind getCopyKind(const MemSegment& src);
  cudaMemcpyKind getCopyInKind();
  cudaMemcpyKind getCopyOutKind();

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

  size_t getSize() const  {return _size;}
  MemoryType getType() const {return _type;}
  //size_t size = 0: in the following calls if no size != 0 is provided the segment size is used.

  //copies from or to any kind of memory segment, if no size provided the smaller size is used
  //if stream is provided async copy will be performed if possible
  void copyAsync (const MemSegment& src, cudaStream_t sid, size_t size = 0);
  void copy (const MemSegment& src, size_t size = 0);
  void copySubSet (const MemSegment& src, size_t srcOffset, size_t dstOffset, size_t size=0);

  //copies from or to host buffers, if no size provided the segment size is used
  //these function will not check if previously issued asynchronous copies are completed or not!
  void copyIn(const void * hostsrc, size_t size = 0);
  void copyOut(void * hostdst, size_t size = 0);
  void copyInSubSet(size_t dstOffset, void * src, size_t size );
  void copyOutSubSet(void * dst, size_t srcOffset, size_t size = 0 );
  //sets whole memory segment to value for each byte
  void memSet(int value, size_t size = 0);
  void memSetAsync(int value, cudaStream_t sid, size_t size);

  void memSet(int value, size_t offset, size_t size);
  void memSetAsync(int value, cudaStream_t sid, size_t offset, size_t size);

  //splits the segment, this gets resized to size of offset and the rest offset to end gets returned as new segment
  MemSegment splitAt(size_t offsetbytes);

  //returns segment pointer to be passed to GPU Kernel (should not be used for any other pointer arithmetics)
  void* getVoidPtr() {return (void*)_basePointer;}

  //checks if provided size fits into segment
  size_t checkSize(size_t size) const;

  void printSummary() const ;
  void printReadableSize(size_t size = 0) const;
  // debugging and dumping
  void DebugOutput();
  // writes or reads buffer to or from a file in binary format
  // size_t _size, MemoryType _type, void * blob
  void dumptoFile(ofstream& fstream);
  void readFromFile(ifstream& fstream);




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

  void dumpHosttoFile(ofstream& fstream){_Host.dumptoFile(fstream);}
  void readHostFromFile(ifstream& fstream){_Host.readFromFile(fstream);}

  void dumpDevicetoFile(ofstream& fstream){ _Device.dumptoFile(fstream);}
  void readDevbiceFromFile(ifstream& fstream){ _Device.readFromFile(fstream);}

  void DebugOutput();
};



////////////////////////////////////////////////


class MemoryResource
{
  char * _basePointer;
  char * _returnPointer;
  size_t _sizeBytes;
  size_t _align;
  vector<MemSegment*> _segments;

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
  static void* allocatePtr(size_t size, MemoryType type);
  static void destroyPtr(void * ptr, MemoryType type);


  void setPreAllocSize(size_t sizeBytes);
  void setPreAllocAlignment(size_t align);
  void setPreAllocMemType(MemoryType type);
  size_t addPreAllocChunk(size_t chunkSize);
  size_t addPreAllocSegment(MemSegment * mseg);
  size_t getPreAllocSize(){ return _sizeBytes; } //always returns sizeBytes, getSize only returns the size if a buffer was created


  void allocate(MemoryType memType);

  void reallocate(size_t, MemoryType type = UnDefined); // if no type given will try to realloc original type if valid
  void reallocate(const MemSegment & src,  MemoryType type = UnDefined);

  void destroy(); // will free all allocatede memory but remember the type and alignment

  void * getSegment(size_t size);
  MemSegment getMemSegment(size_t size);
  MemSegment getCurrentSegGroup();

  void releaseAll();

  size_t startNewSegGroup();
  size_t getCurrentSegGroupSize();
  size_t getSize();// returns the size if a physical buffer is available, otherwise it returns 0
  size_t getPaddedSize();
  size_t getPaddedSize(size_t size);
  size_t memoryUsed();
  size_t memoryAvailable();
  MemoryType getMemType();

  //check if memory resource contains enough memory and is able to reallocate
  bool checkMemory(size_t size); //throws exception if memory insufficient and reallocation not possible

  bool isSet() const;

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
  //class should be derived with protected base class and hide all the actual byte/buffer operations to the outside
  //only needed should be re-implemented on this level for type T

public:
  TMemSegment():MemSegment(){};
  TMemSegment(T * ptr, size_t size, MemoryType type):MemSegment(ptr,size,type){};
  TMemSegment(const MemSegment& src):MemSegment(src){};
  TMemSegment(const MemSegment& first, const MemSegment& last):MemSegment(first,last){};
  virtual ~TMemSegment(){};


  //initializes all array elements to value. T must possess a functioning operator= to guarantee correctness
  void arraySet(T value){
    for(size_t i=0; i<getNumElements(); i++) (*this)[i] = value;
  }


  T* getPtr(){return (T*)getVoidPtr();}

  T& operator[](size_t idx){
    if(_type != HostMem && _type != HostPageLocked ){
      cout << "CUDA Memory Manager Error: Access to Device Memory via [] operator not possible!" << endl;
      throw cudaExecutionException(cudaErrorInvalidHostPointer, __FILE__, __LINE__);
    }
    T* ptr = getPtr();
    return ptr[idx];
  }

  //splits the segment, this gets resized to size of #idx of type T. the rest gets returned as new segment without type
  MemSegment splitAtIdx(size_t idx){ return splitAt(idx * sizeof(T)); }

  T getElement(size_t offset) const {
    T* ptr = (T*)(this->_basePointer);
    return ptr[offset];
  }
  size_t getNumElements() const { return getSize()/sizeof(T); }

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




class MemTypeTracker
{

  bool _tracking;
  size_t _maxAllocated;
  size_t _currentAllocated;
  set<const MemSegment*> mySegments;

  void updateMax(){ _maxAllocated = max(_maxAllocated,_currentAllocated);}

public:

  MemTypeTracker():_tracking(false),_maxAllocated(0),_currentAllocated(0){};
  ~MemTypeTracker();
  void trackSegment(const MemSegment * segment);
  void releaseSegment(const MemSegment * segment);
  size_t getCurrent() const;
  size_t getMax() const;
  void print() const;
  bool isTracking() const;
  void reset();
};

class MemoryUsageTracker{

  map<MemoryType,MemTypeTracker> SegmentTracker;


public:
  ~MemoryUsageTracker();
  MemTypeTracker * trackSegment(const MemSegment * segment);
  void releaseSegment(const MemSegment * segment);
  MemTypeTracker * trackSegment(const MemSegment & segment);
  void releaseSegment(const MemSegment & segment);
  void printMemUsage();
  void printMemUsage(MemoryType type);

};


/////////////////////////////////////////////////////
//TEMPLATE memory allocator
//actually allocates the designated memory and frees it with destruction
//meant for debugging but also might be useful in other situations
template<typename T>
class TMemSegAlloc : public TMemSegment<T>
{
  MemoryResource _resource;
  MemTypeTracker * _tracker;



protected:


  void allocate(size_t size, MemoryType type){
    _resource.reallocate(size,type);
    TMemSegment<T>::_type = _resource.getMemType();
    TMemSegment<T>::_size = _resource.getSize();
    TMemSegment<T>::_basePointer = (void*)_resource.getSegment(size); //if local resource was created
    this->memSet(0);
  }

  void wrapp(T * ptr, size_t sizeBytes, MemoryType type){
      if(isAllocated()){
        cout << "MemoryManager Warning: trying to wrap pointer "<< static_cast<void *>(ptr) << " in already initialized MemSegObject. Original buffer will be deleted and replaced by external pointer!" << endl;
        _resource.destroy();
      }
      TMemSegment<T>::_type = type;
      TMemSegment<T>::_size = sizeBytes;
      TMemSegment<T>::_basePointer = (char*)ptr;
   }



public:

  //no allocation constructors
  TMemSegAlloc():TMemSegment<T>(),_resource(),_tracker(NULL) {};
  TMemSegAlloc(T* ptr, size_t size, MemoryType type):TMemSegment<T>(ptr,size,type),_resource(),_tracker(NULL){}

  //allocation constructors
  TMemSegAlloc(size_t size, MemoryType type):TMemSegment<T>(NULL,size,type),_resource(size,type),_tracker(NULL){
    // cout << "Allocating Buffer of type: " << type << " and size: " << size << endl;
    TMemSegment<T>::_basePointer = (char*) _resource.getSegment(size); //if local resource was created, overwrite copied segment info
    TMemSegment<T>::_type = type;
    //if(TMemSegment<T>::_type != DeviceGlobal)
    this->memSet(0);
  }

  TMemSegAlloc( TMemSegAlloc<T> & that, MemoryType type):TMemSegment<T>(that),_resource(that.getSize(),type),_tracker(NULL){
    //   cout << "Allocating and copying Buffer of type: " << that.getType() << " to " << type << " and size: " << that.getSize() << endl;
    TMemSegment<T>::_basePointer = (char*)_resource.getSegment(that.getSize()); //if local resource was created, overwrite copied segment info
    TMemSegment<T>::_type = type;
    this->copy(that);
  }

  TMemSegAlloc( TMemSegAlloc<T> & that):TMemSegment<T>(that),_resource(that.getSize(),that.getType()),_tracker(NULL){
    //    cout << "Allocating and copying Buffer of type: " << that.getType() << " and size: " << that.getSize() << endl;
    TMemSegment<T>::_basePointer = (char*)_resource.getSegment(that.getSize()); //if local resource was created, overwrite copied segment info
    this->copy(that);
  }


  TMemSegAlloc( ifstream myFile, MemoryType type = UnDefined){
    size_t readsize;
    MemoryType readtype;

    readsize = TMemSegment<T>::readSizeFromFile(myFile);
    readtype = TMemSegment<T>::readMemTypeFromFile(myFile);
    //over write type from outside
    if(type != UnDefined){
      readtype = type;

    }

    allocate(readsize,readtype);
    _tracker = NULL;
    TMemSegment<T>::readBufferFromFile(myFile);
  }


  TMemSegAlloc( MemSegment & that, MemoryType type):TMemSegment<T>(that),_resource(that.getSize(),type),_tracker(NULL){
    //  cout << "Allocating and copying Buffer of type: " << that.getType() << " to " << type << " and size: " << that.getSize() << endl;
    TMemSegment<T>::_basePointer = (char*)_resource.getSegment(that.getSize()); //if local resource was created, overwrite copied segment info
    TMemSegment<T>::_type = type;
    this->copy(that);
  }






  //assignment does not create new buffer and (this) will cause segfaults if the buffer used in that is deleted
  TMemSegAlloc<T>& operator=(const TMemSegAlloc<T>& that)
  {
    _resource.destroy();
    TMemSegment<T>::_type = that._type;
    TMemSegment<T>::_size = that._size;
    if(that.isAllocated()){
      _resource.reallocate(TMemSegment<T>::_size,TMemSegment<T>::_type);
      TMemSegment<T>::_basePointer =  (char*)_resource.getSegment(TMemSegment<T>::_size);
      this->copy(that);
    }else{
      TMemSegment<T>::_basePointer =  that._basePointer;
    }
    return *this;
  };





  void trackMe(MemoryUsageTracker & MemTracker)
  {

    if(isAllocated()){
      _tracker = MemTracker.trackSegment(this);
    }
  };
  bool isAllocated() const {return _resource.isSet();}

  virtual ~TMemSegAlloc(){
    if(isAllocated() && _tracker) _tracker->releaseSegment(this);
  };


};


/*
///////////////////
//TEMPLATE using memory allocator
//creates two TMemSegAlloc objects that actually allocate and free the designated memory
//meant for debugging but also might be useful in other situations
template<typename T>
class TMemSegPairAlloc : public TMemSegPair<T>
{
private:
  TMemSegPairAlloc(const TMemSegPairAlloc<T>& that); // no more copy allowed to prevent unwanted free
  TMemSegPairAlloc& operator=(const TMemSegPairAlloc<T>& that); // no more assignment allowed to prevent unwanted free
public:
  TMemSegPairAlloc(size_t size, MemoryType Htype=HostPageLocked, MemoryType Dtype=DeviceGlobal){
    TMemSegPair<T>::_Host = TMemSegment<T>((T*)MemoryResource::allocate(size,Htype),size,Htype);
    TMemSegPair<T>::_Device = TMemSegment<T>((T*)MemoryResource::allocate(size,Dtype),size,Dtype);
  };

  virtual ~TMemSegPairAlloc(){
    MemoryResource::destroy(TMemSegPair<T>::getHostPtr(),TMemSegPair<T>::_Host.getType());
    MemoryResource::destroy(TMemSegPair<T>::getPtr(),TMemSegPair<T>::_Device.getType());
  };
};
 */
#endif //MEMORYMANAGER_H


