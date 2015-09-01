/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>

#include "MemoryManager.h"
#include "cuda_runtime.h"

using namespace std;


MemSegment::MemSegment():_basePointer(NULL),_size(0),_type(UnDefined) {};
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

void MemSegment::copyInSubSet( size_t dstOffset, void * src,size_t size )
{
  cudaMemcpy(_basePointer + dstOffset, src, checkSize(dstOffset+size),getCopyInKind());
  CUDA_ERROR_CHECK();
}

void MemSegment::copyOutSubSet(void * dst, size_t srcOffset, size_t size )
{
  if(size == 0)
    size = _size - srcOffset; // if size == 0 copy from srcOffset to end of buffer

  //cout << "cudaMemcpy( " << dst <<", " <<(void*)_basePointer << " + " <<  srcOffset << ", " <<  "checkSize(" << srcOffset << " + " <<  size  << " ),getCopyOutKind())" << endl;
  cudaMemcpy(dst,(void*)(_basePointer + srcOffset) ,checkSize(srcOffset+size),getCopyOutKind());

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



//If no size, or size= 0 is provided src will be copied into local buffer. if a
//size mismatch is encountered the smaller one of the two sizes will be used to initialize the
//copy and a warning will be printed.
void MemSegment::copy (const MemSegment& src, size_t size)
{
  //src.checkSize(size); //should probably check if return value is larger this->getSize()
  if(size == 0){
    size_t dstSize = this->getSize();
    size_t srcSize = src.getSize();
    size = min(dstSize,srcSize);
    if(dstSize != srcSize){
      cout << "CUDA Memory Manager Warning: buffer size missmatch dst: ";
      printReadableSize(dstSize);
      cout << " != src: ";
      printReadableSize(srcSize);
      cout <<". Will use smaller buffer size of ";
      printReadableSize(size);
      cout << " to initiate copy." << endl;
    }
  }

  cudaMemcpy(_basePointer, src._basePointer, checkSize(size), getCopyKind(src));
  //cout << "cudaMemcpy(" << (void*)_basePointer << ", " << (void*)src._basePointer << ", " << checkSize(size) << ", " << getCopyKind(src) << ")" << endl;
  CUDA_ERROR_CHECK();
}


void MemSegment::copySubSet(const MemSegment& src, size_t srcOffset, size_t dstOffset, size_t size)
{

  src.checkSize(srcOffset + size); //check if the requested sub segment actually is within src segment
  size_t copysize = (size == 0)? (src.getSize() - srcOffset):(size); // no size provide copy all of src from offset to end otherwise try to copy size
  this->checkSize(dstOffset + copysize); //check if the requested sub segment fits into this
  cudaMemcpy(_basePointer + dstOffset, src._basePointer + srcOffset, copysize, getCopyKind(src));
  CUDA_ERROR_CHECK();
}

//sets first size bytes in segment to value
//size (optional): number of bytes to set. if not provided
//                 all bytes in the segment will beset to value
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

//
//same as memSet but only works on device memory and performs asynchronous memset if stream is provided
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

void MemSegment::memSet(int value, size_t offset, size_t size)
{
  char * tmpPtr = _basePointer + offset;
  checkSize(size+offset);
  switch(_type){
    case DeviceGlobal:
      cudaMemset((void*)tmpPtr, value,size);
      break;
    case HostMem:
    case HostPageLocked:
    default:
      memset((void*)tmpPtr, value, size);
  }
}

//
//same as memSet but only works on device memory and performs asynchronous memset if stream is provided
void MemSegment::memSetAsync(int value, cudaStream_t sid, size_t offset, size_t size)
{
  char * tmpPtr = _basePointer + offset;
  checkSize(size+offset);
  switch(_type){
    case DeviceGlobal:
      cudaMemsetAsync((void*)tmpPtr, value, size, sid);
      break;
    case HostMem:
    case HostPageLocked:
    default:
      cout << "CUDA Memory Manager Warning: Asynchronous Host Side memset not available!" << endl;
      memset((void*)tmpPtr, value, size);
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

size_t MemSegment::checkSize(size_t size) const
{
  if(size <= 0) return _size;

  if(size <= _size)
  {
    //if(size < _size) cout << "CUDA Memory Manager Warning: copying smaller segment of size " << size <<" into large segment of size " << _size << "!" << endl;
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

void MemSegment::printSummary() const
{
  cout << "MemSegment of type " << getType() << " of size: "<< getSize() <<" ( " << getSize()/(1024.0*1024.0) << " MB )" << endl;
}

void MemSegment::printReadableSize(size_t size) const
{
 double printSize = (size == 0)?(_size):(size);
 const string SizeString[] = {"Bytes","KB","MB","GB","TB"};
 int offset = 0;
 while(printSize > 999.0){
   printSize /= 1024.0;
   offset++;
 }
 cout << std::setprecision(3) << printSize << " " << SizeString[offset];
}



void MemSegment::DebugOutput()
{
  cout << "CUDA Memory Manager Debug: MemSegment: " << getSize() <<", " << getVoidPtr() << endl;
}

void MemSegment::dumptoFile(ofstream& myFile)
{
  myFile.write((const char*)&_size , sizeof(size_t));
  myFile.write((const char*)&_type , sizeof(MemoryType));


  char * tmpPtr = _basePointer;

  if(_type == DeviceGlobal){
    tmpPtr = new char[_size];
    this->copyOut(tmpPtr);
  }
  //copy into actual Memory segment
  myFile.write((const char*)tmpPtr , _size);

  //clean up
  if(_type == DeviceGlobal)
    delete tmpPtr;
}


size_t  MemSegment::readSizeFromFile(ifstream& myFile){
  size_t size;
  if(myFile.eof()){
    cerr << "File Error: does not contain any more buffers (reached EOF)" << endl;
    exit (-1);
  }
  myFile.read((char*)&size,sizeof(size_t));
  if(myFile.eof()){
    cerr << "File Error: does not contain any more buffers (reached EOF)" << endl;
    exit (-1);
  }
  if(size <= 0){
    cerr << "File Error: size field corrupted (<= 0)" << endl;
    exit (-1);
  }
  return  size;
}

MemoryType  MemSegment::readMemTypeFromFile(ifstream& myFile)
{
  MemoryType type;
  myFile.read((char*)&type,sizeof(MemoryType));
  return type;
}

void  MemSegment::readBufferFromFile(ifstream& myFile, size_t readsize){

  char * tmpPtr = _basePointer;
  if(_type == DeviceGlobal){
    tmpPtr = new char[_size];
  }

  if(readsize <= 0) readsize = _size;
  //read to temp pointer
  myFile.read((char*)tmpPtr, readsize);
  if (!myFile)
  {
    readsize = myFile.gcount();
    cout << "error: only " <<  readsize << " bytes could be read";
  }

  if(_type == DeviceGlobal){
    //copy into actual Memory segment
    cout << "copy buffer of size " << readsize << " read from file to device " << endl;
    this->copyIn((void*)tmpPtr,readsize);
    delete tmpPtr;
  }
}


void MemSegment::readFromFile(ifstream& myFile)
{
  size_t readsize;
  MemoryType readtype;

  readsize = readSizeFromFile(myFile);

  if(readsize > _size){
    cerr << "File Error: buffer in file exceeds buffer size: " << _size << " < " << readsize << endl;
    exit (-1);
  }

  readtype = readMemTypeFromFile(myFile);

  if(readtype != _type)
  {
    cerr << "File Warning: buffer in file Memory type miss-match "<<  _type << " != " << readtype  << endl;
  }

  readBufferFromFile(myFile,readsize);

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



/////////////////////////////////////////////////

MemTypeTracker::~MemTypeTracker(){
  reset();
}

void MemTypeTracker::trackSegment(const MemSegment * segment)
{
  if(segment){
    set<const MemSegment*>::iterator it;
    it = mySegments.find(segment);
    if(it == mySegments.end()){
     // cout << "Start tracking: ";
      //segment->printSummary();
      mySegments.insert(segment);
      _currentAllocated += segment->getSize();
      updateMax();
      _tracking = true;
    }else{
      if(segment->getSize() != (*it)->getSize())
        releaseSegment(segment);
        trackSegment(segment);
    }
  }
}
void MemTypeTracker::releaseSegment(const MemSegment * segment)
{
  if(segment){
    set<const MemSegment*>::iterator it;
    it = mySegments.find(segment);
    if(it != mySegments.end()){
      _currentAllocated -= segment->getSize();
      mySegments.erase (it);
    }
  }
}

size_t MemTypeTracker::getCurrent() const
{
  return _currentAllocated;
}
size_t MemTypeTracker::getMax() const
{
  return _maxAllocated;
}
void MemTypeTracker::print() const
{
  if(isTracking()){
    cout <<  "Currently tracking " << mySegments.size() << endl;
    for (set<const MemSegment*>::iterator it=mySegments.begin(); it!=mySegments.end(); ++it){
      (*it)->printSummary();
    }
    cout << "Currently allocated Memory: " << _currentAllocated << " ( " << _currentAllocated/(1024.0*1024.0) << " MB ) "<< endl;
    cout << "Maximum allocated Memory: " << _maxAllocated << " ( " << _maxAllocated/(1024.0*1024.0) << " MB ) "<< endl;
  }else
    cout << "No segments tracked yet." << endl;

}
bool MemTypeTracker::isTracking() const
{
  return _tracking;
}
void MemTypeTracker::reset()
{
  _tracking = false;
  _maxAllocated = 0;
  _currentAllocated=0;
  mySegments.clear();
}

/////////////

MemoryUsageTracker::~MemoryUsageTracker()
{
  printMemUsage();
}


MemTypeTracker * MemoryUsageTracker::trackSegment(const MemSegment * segment)
{
  if(segment){
    SegmentTracker[segment->getType()].trackSegment(segment);
    return & SegmentTracker[segment->getType()];
  }
  return NULL;
}
void MemoryUsageTracker::releaseSegment(const MemSegment * segment)
{
  if(segment){
    SegmentTracker[segment->getType()].releaseSegment(segment);
  }
}

MemTypeTracker * MemoryUsageTracker::trackSegment(const MemSegment & segment)
{
  return trackSegment(&segment);
}

void MemoryUsageTracker::releaseSegment(const MemSegment & segment)
{
  releaseSegment(&segment);
}

void MemoryUsageTracker::printMemUsage(){
  for (map<MemoryType,MemTypeTracker>::iterator it=SegmentTracker.begin(); it!=SegmentTracker.end(); ++it){
    cout << "Tracking Info for Memory Segments of Type " << (*it).first << ":" << endl;
    (*it).second.print();
  }
}

void MemoryUsageTracker::printMemUsage(MemoryType type)
{
  if(SegmentTracker.find(type) != SegmentTracker.end()){
    cout << "Tracking Info for Memory Segments of Type " << type << ":" << endl;
    SegmentTracker[type].print();
  }
}



///////////////////////////////////////////////////////////////////////////////////////////



MemoryResource::MemoryResource()
{
  _sizeBytes = 0;
  _type = UnDefined;
  _basePointer = NULL;
  _returnPointer = NULL;
  _align = 128;

}

MemoryResource::MemoryResource(size_t size, MemoryType type)
{
  _sizeBytes = 0;
  _type = type;
  _basePointer = NULL;
  _returnPointer = NULL;
  _align = 128;

  if(size >0)
    reallocate(size, type);

}


MemoryResource::~MemoryResource()
{
  destroy();
}



void MemoryResource::setPreAllocSize(size_t sizeBytes)
{
  if(_basePointer == NULL){
    _sizeBytes = sizeBytes;
  }else{
    cout << "CUDA MemoryManager: attempt to change pre-allocation setting for an already allocated buffer!" << endl;
    throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
  }
}

void MemoryResource::setPreAllocAlignment(size_t align) {
  if(_basePointer == NULL){
    _align = align;
  }else{
    cout << "CUDA MemoryManager: attempt to change pre-allocation setting for an already allocated buffer!" << endl;
    throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
  }
};

void MemoryResource::setPreAllocMemType(MemoryType type){
  _type= type;
  if(_basePointer == NULL){
    _type= type;
  }else{
    cout << "CUDA MemoryManager: attempt to change pre-allocation setting for an already allocated buffer!" << endl;
    throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
  }
};

//padds current buffer size to alignment and adds chunk. returns old size
size_t MemoryResource::addPreAllocChunk(size_t chunk){
  if(_basePointer == NULL){
    size_t current = _sizeBytes;
    _sizeBytes = getPaddedSize() + chunk; //padd original size and add new chunk
    return current;
  }else{
    cout << "CUDA MemoryManager: attempt to change pre-allocation setting for an already allocated buffer!" << endl;
    throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
  }
}

size_t MemoryResource::addPreAllocSegment(MemSegment * mseg){
  if(_basePointer == NULL){
      size_t current = _sizeBytes;
      _sizeBytes = getPaddedSize() + mseg->getSize(); //padd original size and add new chunk

      return current;
    }else{
      cout << "CUDA MemoryManager: attempt to change pre-allocation setting for an already allocated buffer!" << endl;
      throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
    }
}



void MemoryResource::allocate(MemoryType memType)
{
  if(_sizeBytes == 0 || _type == UnDefined){
    cout << "CUDA MemoryManager: attempt to allocate without providing type or size!" << endl;
    throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
  }
  if(_basePointer != NULL){
      cout << "CUDA MemoryManager: attempt to allocate already allocated buffer!" << endl;
      throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
  }

  reallocate(_sizeBytes,_type);

}


void MemoryResource::reallocate(size_t size, MemoryType type)
{
  //if already allocated and reallocation is needed cleanup first
  if(_sizeBytes > 0 || _basePointer != NULL) destroy();

#if DEBUG
  cout << "Allocating Buffer of type: " << type << " and size: " << size << endl;
#endif

  _sizeBytes = size;
  _type = (type == UnDefined)?(_type):(type); // only change type if type != undefined [default] is provided

  try{
    _basePointer = (char *) MemoryResource::allocatePtr(_sizeBytes,_type);
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

}

//allocates memory according to needs of MemSegment, if no MemoryType is provided the src type is used
void MemoryResource::reallocate(const MemSegment & src,  MemoryType type)
{
  if(type == UnDefined)
    type = src.getType();

  reallocate(src.getSize(), type);
}


//static function
void* MemoryResource::allocatePtr(size_t size, MemoryType type)
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


//static function
void MemoryResource::destroyPtr(void * ptr, MemoryType type)
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

  MemoryResource::destroyPtr((void*)_basePointer, _type);

  _sizeBytes = 0;
  _basePointer = NULL;
  _returnPointer = NULL;

}


void * MemoryResource::getSegment(size_t size)
{
  // if(_type == UnDefined || _basePointer == NULL)   throw cudaNotEnoughMemForStream(__FILE__,__LINE__);
  //return NULL;
  size_t segSizePadded = getPaddedSize(size);

  /*switch(_type){
      case DeviceGlobal:
        cout << "CUDA: getSegment Device " << segSize <<" "  << memoryUsed() <<  " free:"  << memoryAvailable() << endl;
        break;
      default:
        cout << "CUDA: getSegment Host   " << segSize <<" "  << memoryUsed() <<  " free:"  << memoryAvailable()  << endl;
    }*/

  if(isSet())
  {
    if (_returnPointer + size <= _basePointer + _sizeBytes){
      _CurrentSegment = MemSegment((void*)_returnPointer, size, _type);
      _CurrentSegGroup.inflateSegmentBy(segSizePadded);
      _returnPointer += segSizePadded;
      return _CurrentSegment.getVoidPtr();
    }
    throw cudaNotEnoughMemForStream(__FILE__,__LINE__);
  }else{
    //this is for an empty resource to determine the needed size
    _basePointer += segSizePadded;
    _CurrentSegment = MemSegment((void*)_returnPointer, size, _type);
    _CurrentSegGroup.inflateSegmentBy(segSizePadded);
    _returnPointer += segSizePadded;
    return _CurrentSegment.getVoidPtr();
  }

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
  return (isSet())?(_sizeBytes):(0);
}

size_t MemoryResource::getPaddedSize()
{
  return getPaddedSize(_sizeBytes);
}
size_t MemoryResource::getPaddedSize(size_t size)
{
  return ( (size + _align -1) / _align) * _align;
}


size_t MemoryResource::memoryAvailable()
{
  return (isSet())?(_sizeBytes - memoryUsed()):(0);
}

size_t MemoryResource::memoryUsed()
{
  return (isSet())?((size_t)_returnPointer - (size_t)_basePointer):(0);
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


bool MemoryResource::isSet() const
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









