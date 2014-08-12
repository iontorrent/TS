/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

// patch for CUDA5.0/GCC4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <iostream>

#include "MemoryManager.h"



MemSegment::MemSegment():_basePointer(NULL),_size(0),_type(UnDefined){};
MemSegment::MemSegment(void * ptr, size_t size, MemoryType type):_basePointer((char*)ptr),_size(size),_type(type){};
MemSegment::MemSegment(const MemSegment& src):_basePointer(src._basePointer),_size(src._size),_type(src._type){};
MemSegment::MemSegment(const MemSegment& first, const MemSegment& last)
{
  _basePointer = first._basePointer;
  _type = first._type;

  if(first._type != last._type){
    cout << "CUDA Memory Manager Warning: Segment accumulation failed first and last segment do not match type" << endl;
    _size = first._size;
  }
  else if(first._basePointer > last._basePointer)
  {
    cout << "CUDA Memory Manager Warning: Segment accumulation failed first and last base pointers are not in order" << endl;
    _size = first._size;
  }
  else
  {
    //create segment from first_basepointer to the end of last.
    _size = (last._basePointer + last._size) - first._basePointer;
  }
}


size_t MemSegment::addSegment(const MemSegment& append)
{
  if(_basePointer == NULL){
    _type = append._type;
    _size = append._size;
    _basePointer = append._basePointer;
    return _size;
  }

  if(_type != append._type){
      cout << "CUDA Memory Manager Warning: Appending Segment failed, segments do not match type" << endl;
    }
    else if(_basePointer > append._basePointer)
    {
      cout << "CUDA Memory Manager Warning: Appending Segment failed, first and last base pointers are not in order" << endl;
    }
    else if(_basePointer+_size != append._basePointer)
    {
      cout << "CUDA Memory Manager Warning: Appended Segment is not the imidiate next segment, segments inbetween are added automatically!" << endl;
      _size = (append._basePointer + append._size) - _basePointer;
    }
    else
    {
      //create segment from first_basepointer to the end of last.
      _size += append._size;
    }


  return _size;
}


size_t MemSegment::inflateSegmentBy(size_t size)
{
  _size += size;
  return size;
}


size_t MemSegment::changeSegmentSizeTo(size_t size)
{
  if(size > 0)
  {
    _size = size;
  }
  else
  {
    cout << "CUDA Memory Manager Warning: Segment rezise request to size <= 0, previouse size: " << _size << "!" << endl;
    throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
  }

  return _size;
}

void MemSegment::copyIn(const void * src, size_t size)
{

  cudaMemcpy(_basePointer,src,checkSize(size),getCopyInKind());
  CUDA_ERROR_CHECK();
}

void MemSegment::copyOut(void * dst, size_t size )
{
  cudaMemcpy(dst,_basePointer,checkSize(size),getCopyOutKind());
  CUDA_ERROR_CHECK();
}




void MemSegment::copyAsync (const MemSegment& src, cudaStream_t sid, size_t size)
{


  if(sid == 0) cout << "CUDA Memory Manager Warning: intended async-copy is using stream 0 turning it into non-async copy!" << endl;

  if(_type == HostPageLocked || _type == DeviceGlobal)
  {
    if(src._type == HostPageLocked || src._type == DeviceGlobal)
    {
      cudaMemcpyAsync(_basePointer, src._basePointer, checkSize(size), getCopyKind(src), sid);
    }
    else
    {
      cout << "CUDA Memory Manager Warning: intended async-copy is using non paged locked host memory turning it into a non-async copy!" << endl;
      cudaMemcpy(_basePointer, src._basePointer, checkSize(size), getCopyKind(src));
    }
  }
  else
  {
    cout << "CUDA Memory Manager Warning: intended async-copy is using non paged locked host memory turning it into a non-async copy!" << endl;
    cudaMemcpy(_basePointer, src._basePointer, checkSize(size), getCopyKind(src));
  }
}


void MemSegment::copy (const MemSegment& src, size_t size)
{

  cudaMemcpy(_basePointer, src._basePointer, checkSize(size), getCopyKind(src));
  CUDA_ERROR_CHECK();
}

void MemSegment::memSet(int value, size_t size)
{
  switch(_type){
    case DeviceGlobal:
      cudaMemset((void*)_basePointer, value, checkSize(size));
      break;
    case HostMem:
    case HostPageLocked:
    default:
      memset((void*)_basePointer, value, checkSize(size));
  }
}

void MemSegment::memSetAsync(int value, cudaStream_t sid, size_t size)
{
  switch(_type){
    case DeviceGlobal:
      cudaMemsetAsync((void*)_basePointer, value, checkSize(size), sid);
      break;
    case HostMem:
    case HostPageLocked:
    default:
      cout << "CUDA Memory Manager Warning: Asyncronouse Host Side memset not available!" << endl;
      memset((void*)_basePointer, value, checkSize(size));
  }
}


MemSegment MemSegment::splitAt(size_t offset)
{
  if(offset >= _size){
    cout << "CUDA Memory Manager Warning: tried to split segment at offset(" <<offset<<") >= segment size("<< _size << ")!" << endl;
    throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
  }
  //cout << "CUDA Memory Manager: splitting buffer of size " << _size << " into two buffers of sizes " << offset << " and " << _size-offset << endl;
  MemSegment tmp(_basePointer+offset, _size-offset, _type);
  changeSegmentSizeTo(offset);
  return tmp;
}

size_t MemSegment::checkSize(size_t size)
{
  if(size <= 0) return _size;

  if(size <= _size)
  {
    return size;
  }

  cout << "CUDA Memory Manager Warning: requested size (" << size <<") is larger than segment size (" << _size << ")!" << endl;
  throw cudaAllocationError(cudaErrorInvalidValue, __FILE__, __LINE__);

}


cudaMemcpyKind MemSegment::getCopyKind(const MemSegment& src)
{

  switch(_type){
    case DeviceGlobal:
      switch(src._type){
        case DeviceGlobal:
          return cudaMemcpyDeviceToDevice;
        case HostMem:
        case HostPageLocked:
        default:
          return cudaMemcpyHostToDevice;
      }
      //no break
        case HostMem:
        case HostPageLocked :
        default:
          switch(src._type){
            case DeviceGlobal:
              return cudaMemcpyDeviceToHost;
            case HostMem:
            case HostPageLocked:
            default:
              return cudaMemcpyHostToHost;
          }
  }
}


cudaMemcpyKind MemSegment::getCopyInKind()
{
  switch(_type){
    case DeviceGlobal:
      return cudaMemcpyHostToDevice;
    case HostMem:
    case HostPageLocked :
    default:
      return cudaMemcpyHostToHost;
  }
}
cudaMemcpyKind MemSegment::getCopyOutKind()
{
  switch(_type){
    case DeviceGlobal:
      return cudaMemcpyDeviceToHost;
    case HostMem:
    case HostPageLocked :
    default:
      return cudaMemcpyHostToHost;
  }
}


void MemSegment::DebugOutput()
 {
  cout << "CUDA Memory Manager Debug: MemSegment: " << getSize() <<", " << getVoidPtr() << endl;
 }


////////////////////////////////////////////////////////////////////////////////////////////


MemSegPair::MemSegPair():_Host(),_Device(){};
MemSegPair::MemSegPair(void* HostPtr, void* DevPtr, size_t size, MemoryType Htype, MemoryType Dtype ):_Host(HostPtr,size,Htype),_Device(DevPtr,size,Dtype){};
MemSegPair::MemSegPair(const MemSegPair& src):_Host(src._Host),_Device(src._Device){};
MemSegPair::MemSegPair(const MemSegment& host, const MemSegment& device):_Host(host),_Device(device)
{
  getSize();
}
MemSegPair::MemSegPair(const MemSegPair& first, const MemSegPair& last):_Host(first._Host,last._Host),_Device(first._Device,last._Device)
{
  getSize();
}


 size_t MemSegPair::addSegment(const MemSegPair& append)
 {
   _Host.addSegment(append._Host);
   _Device.addSegment(append._Device);
   return getSize();
 }

 size_t MemSegPair::inflateSegmentBy(size_t size)
 {
   _Host.inflateSegmentBy(size);
   _Device.inflateSegmentBy(size);
   return getSize();

 }

 size_t MemSegPair::changeSegmentSizeTo(size_t size)
 {
   _Host.changeSegmentSizeTo(size);
   _Device.changeSegmentSizeTo(size);
   return getSize();
 }

 size_t MemSegPair::getSize()
 {

   if(_Host.getSize() == _Device.getSize())
     return _Host.getSize();

   cout << "CUDA Memory Manager Error: Device-Host Segment size mismatch!" << endl;
   throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
 }

 void MemSegPair::copyToHostAsync (cudaStream_t sid, size_t size)
 {
   _Host.copyAsync(_Device, sid, size);
 }
 void MemSegPair::copyToHost ( size_t size)
 {
   _Host.copy(_Device, size);
 }
 void MemSegPair::copyToDeviceAsync (cudaStream_t sid, size_t size)
 {
   _Device.copyAsync(_Host,sid,size);
 }
 void MemSegPair::copyToDevice ( size_t size)
 {
   _Device.copy(_Host, size);
 }

 void MemSegPair::copyIn(const void * src, size_t size)
 {
   _Host.copyIn(src,size);
 }

 void MemSegPair::copyOut(void * dst, size_t size )
 {
   _Host.copyOut(dst,size);
 }
 void MemSegPair::memSet(int value)
 {
   _Host.memSet(value);
   _Device.memSet(value);
 }


 MemSegPair MemSegPair::splitAt(size_t offset)
 {
   MemSegPair tmp(_Host.splitAt(offset),_Device.splitAt(offset)); //split at offset
   changeSegmentSizeTo(offset); //resize current
   return tmp;
 }

void MemSegPair::DebugOutput()
{
  cout << "CUDA Memory Manager Debug: MemSegPair Host:" << _Host.getSize() <<", " << _Host.getVoidPtr() << " Device:" << _Device.getSize() <<", " << _Device.getVoidPtr() << endl;
}


///////////////////////////////////////////////////////////////////////////////////////////



MemoryResource::MemoryResource()
{
  _sizeBytes = 0;
  _type = UnDefined;
  _basePointer = NULL;
  _returnPointer = NULL;

}

MemoryResource::MemoryResource(size_t size, MemoryType type)
{
  _sizeBytes = 0;
  _type = type;
  _basePointer = NULL;
  _returnPointer = NULL;


  if(size >0)
    reallocate(size, type);

}


MemoryResource::~MemoryResource()
{
  destroy();
}


void* MemoryResource::allocate(size_t size, MemoryType type)
{
  void * ptr = NULL;
  cudaError_t err = cudaSuccess;

  switch(type){
    case HostPageLocked:
      cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
      break;
    case DeviceGlobal:
      cudaMalloc(&ptr, size);
      break;
    case HostMem:
      ptr =(char*)malloc(size);
      break;
    case UnDefined:
    default:
      ptr = NULL;
  }

  err = cudaGetLastError();

  if ( err != cudaSuccess || ptr == NULL){
    throw cudaAllocationError(err, __FILE__, __LINE__);
  }

  return ptr;
}

bool MemoryResource::reallocate(size_t size, MemoryType type)
{
  //if already allocated and reallocation is needed cleanup first
  if(_sizeBytes > 0 || _basePointer != NULL) destroy();

  _sizeBytes = size;
  _type = (type == UnDefined)?(_type):(type);

  try{
    _basePointer = (char *) MemoryResource::allocate(_sizeBytes,_type);
  }
  catch(cudaException &e)
  {
     e.Print();
    _basePointer = NULL;
    _sizeBytes = 0;
    _basePointer = NULL;
    throw cudaAllocationError(e.getCudaError(), __FILE__, __LINE__);
  } 
  _returnPointer = _basePointer;
  startNewSegGroup();


  return true;

}


void MemoryResource::destroy(void * ptr, MemoryType type)
{
  if(ptr != NULL){
      switch(type){
      case HostMem:
        free(ptr);
        break;
      case HostPageLocked:
        cudaFreeHost(ptr); //CUDA_ERROR_CHECK();
        break;
      case DeviceGlobal:
        cudaFree(ptr); //CUDA_ERROR_CHECK();
        break;
      case UnDefined:
      default:
        break;
      }
  }
}


void MemoryResource::destroy()
{

  MemoryResource::destroy((void*)_basePointer, _type);

  _sizeBytes = 0;
  _basePointer = NULL;
  //_returnPointer = NULL;

}


void * MemoryResource::getSegment(size_t size)
{

  if(_type == UnDefined || _basePointer == NULL)   throw cudaNotEnoughMemForStream(__FILE__,__LINE__);
  //return NULL;
  size_t segSize = ((size + 128 -1)/128) *128; 
  
    /*switch(_type){
      case DeviceGlobal:
        cout << "CUDA: getSegment Device " << segSize <<" "  << memoryUsed() <<  " free:"  << memoryAvailable() << endl;
        break;
      default:
        cout << "CUDA: getSegment Host   " << segSize <<" "  << memoryUsed() <<  " free:"  << memoryAvailable()  << endl;
    }*/
   

  if(_returnPointer + segSize <= _basePointer + _sizeBytes ){
    _CurrentSegment = MemSegment((void*)_returnPointer, segSize, _type);
    _CurrentSegGroup.inflateSegmentBy(segSize);
    _returnPointer += segSize;
    return _CurrentSegment.getVoidPtr();
  }

  throw cudaNotEnoughMemForStream(__FILE__,__LINE__);
}

MemSegment MemoryResource::getMemSegment(size_t size)
{
  getSegment(size);
  return _CurrentSegment;
}

MemSegment MemoryResource::getCurrentSegGroup()
{
  return _CurrentSegGroup;
}


void MemoryResource::releaseAll()
{
  _returnPointer = _basePointer;
  _CurrentSegment = MemSegment(_returnPointer, 0, _type);
  startNewSegGroup();

}

size_t MemoryResource::startNewSegGroup()
{
  size_t ret = _CurrentSegGroup.getSize();
  _CurrentSegGroup = MemSegment((void*)_returnPointer, 0, _type);
  return ret;
}

size_t MemoryResource::getCurrentSegGroupSize()
{
  return _CurrentSegGroup.getSize();
}

size_t MemoryResource::getSize()
{
  return _sizeBytes;
}

size_t MemoryResource::memoryAvailable()
{
  return _sizeBytes - memoryUsed();
}

size_t MemoryResource::memoryUsed()
{
  return ((size_t)_returnPointer - (size_t)_basePointer);
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
     // free_byte -=50*1024*1024; //allow for 50 meg buffer that might not  be available because of context
      if (free_byte < size){  
        cout << getLogHeader() << " **** NOT ENOUGH MEMORY ON DEVICE TO REALLOCATED: " << size << " bytes needed, " << free_byte << " bytes available, no reallocation performed, propagating up." << endl;
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





