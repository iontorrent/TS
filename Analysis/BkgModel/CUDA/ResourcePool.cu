/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

// patch for CUDA5.0/GCC4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <iostream>

#include "ResourcePool.h"


using namespace std;




streamResources::streamResources(size_t hostSize, size_t deviceSize, int id):_HMem(hostSize,HostPageLocked),_DMem(deviceSize, DeviceGlobal)
{

  _inUse = false;
  _stream = NULL;
  _streamId = id;
  cudaGetDevice(&_devId);

  cout << getLogHeader() << " acquiring cudaStream" << endl;

  if(_HMem.memoryAvailable() < hostSize || _DMem.memoryAvailable() < deviceSize ){
    throw cudaException(cudaErrorMemoryAllocation);
  }

  cudaStreamCreate(&_stream);


  cudaError_t err = cudaGetLastError();
  if(_stream ==NULL  || err != cudaSuccess){
    _stream=NULL;
    throw cudaStreamCreationError(__FILE__,__LINE__);
  }


}

streamResources::~streamResources()
{
  if(_stream != NULL) cudaStreamDestroy(_stream); //CUDA_ERROR_CHECK(); 
}

cudaStream_t streamResources::getStream()
{
  return _stream;
}


MemoryResource * streamResources::getHostMem()
{
  return &_HMem;
}

MemoryResource * streamResources::getDevMem()
{
  return &_DMem;
}


MemSegment streamResources::getHostSegment(size_t size)
{
  return _HMem.getMemSegment(size);
}
MemSegment streamResources::getDevSegment(size_t size)
{
  return _DMem.getMemSegment(size);
}
MemSegPair streamResources::GetHostDevPair(size_t size)
{
  return MemSegPair(_HMem.getMemSegment(size),_DMem.getMemSegment(size));
}

MemSegPair streamResources::GetCurrentPairGroup()
{
  if(_HMem.getCurrentSegGroupSize() == _DMem.getCurrentSegGroupSize())
    return MemSegPair(_HMem.getCurrentSegGroup(),_DMem.getCurrentSegGroup());

  cout << getLogHeader() << " Memory Manager Warning: No valid Memory Host Device Pair available, returning NULL Segment!" << endl;
  return MemSegPair();
}

MemSegment streamResources::GetCurrentHostGroup()
{
  return _HMem.getCurrentSegGroup();
}
MemSegment streamResources::GetCurrentDeviceGroup()
{
  return _DMem.getCurrentSegGroup();
}

size_t streamResources::StartNewSegGroup()
{
  size_t hs = StartNewHostSegGroup();
  size_t ds = StartNewDeviceSegGroup();
  return (hs == ds)?(hs):(0);
}


size_t streamResources::StartNewHostSegGroup()
{
  return _HMem.startNewSegGroup();
}
size_t streamResources::StartNewDeviceSegGroup()
{
  return _DMem.startNewSegGroup();
}


int streamResources::getStreamId()
{
  return _streamId;
}


bool streamResources::aquire()
{
  if(_inUse) return false;
  _inUse = true;
  return _inUse;
}

void streamResources::release()
{
  _inUse = false;
  _HMem.releaseAll();
  _DMem.releaseAll();
}

bool streamResources::isSet()
{
  if(_stream == NULL) return false;
  if(!_HMem.isSet()) return false;
  if(!_DMem.isSet()) return false;
  if(_inUse) return true;
  return true;
}


int streamResources::getDevId()
{
  return _devId;
}

string streamResources::Status()
{
  ostringstream output;
  size_t divMB = 1024*1024;
  output << getLogHeader() << " Host Memory allocated(used): " << _HMem.getSize()/divMB <<"(" << _HMem.memoryUsed()/divMB << ")MB Device Memory: " << _DMem.getSize()/divMB <<"(" << _DMem.memoryUsed()/divMB << ")MB";

  return output.str();
}


string streamResources::getLogHeader()
{
  ostringstream headerinfo;

  headerinfo << "CUDA " << _devId << " StreamResource " << _streamId << ":";

  return headerinfo.str();
}
////////////////////////////////////////////////////////////


size_t cudaResourcePool::_SrequestedDeviceSize = 0;
size_t cudaResourcePool::_SrequestedHostSize = 0;

cudaResourcePool::cudaResourcePool(int numStreams)
{
  _HostSize = _SrequestedHostSize;
  _DeviceSize = _SrequestedDeviceSize;
  cudaGetDevice(&_devId);


  tryAddResource(numStreams);

  if(_sRes.empty()) throw   cudaStreamCreationError( __FILE__,__LINE__);

}
/*
cudaResourcePool::cudaResourcePool(size_t hostsize, size_t devicesize, int numStreams)
{
  _HostSize = hostsize;
  _DeviceSize = devicesize;
  cudaGetDevice(&_devId);


  tryAddResource(numStreams);  

  if(_sRes.empty()) throw   cudaStreamCreationError( __FILE__,__LINE__);

}*/

void cudaResourcePool::tryAddResource(unsigned int numStreams)
{

  streamResources * srestmp;

  while(_sRes.size() < numStreams){
    int i = _sRes.size();
    srestmp = NULL;
    try{
      // remove print memory since we observed a segfault in the libcuda api call, see TS-7922
      //printMemoryUsage();
      cout << getLogHeader() << " trying to create stream resources" << i <<" with "<<  _HostSize/(1024.0*1024.0)  << "MB Host and " << _DeviceSize/(1024.0*1024.0) << "MB Device memory" << endl;

      srestmp = new streamResources(_HostSize, _DeviceSize, i);
      _sRes.push_back (srestmp);
    }
    catch(cudaException &e){
      cout << getLogHeader() << " creation of stream resource " << i << " failed! " <<  endl;
      //      getNumStreams();
      break;
    }
  }

}



cudaResourcePool::~cudaResourcePool()
{
  cout << getLogHeader() << " destroying ResourcePool" << endl;
  while (!_sRes.empty())
  {
    delete _sRes.back();
    _sRes.pop_back();
  }  
}

streamResources * cudaResourcePool::getResource()
{

  vector<streamResources *>::iterator it;

  for ( it=_sRes.begin() ; it < _sRes.end(); it++ ){
    if ((*it)->aquire()){ 
      return (*it);
    }
  }
  return NULL;
}

void cudaResourcePool::releaseResource(streamResources *& res)
{
  vector<streamResources *>::iterator it;
  for ( it=_sRes.begin() ; it < _sRes.end(); it++ ){
    if ((*it) == res){ 
      res->release();
      break;
    }
  }
  res = NULL;
}

// cleans out streamResources where reallocation of resources might have failed 
// during execution or for whatever reason could not be allocated.
void cudaResourcePool::poolCleaning()
{
  for(int i=(_sRes.size()-1); i >= 0 ; i--){ // iterate backwards for easy delete
    if(!_sRes[i]->isSet()){
      cout << getLogHeader() << " removing SR " << _sRes[i]->getStreamId() << " from StreamPool" << endl;
      delete _sRes[i]; // destroy resource object
      _sRes.erase(_sRes.begin()+i); //delete resource from list
    }
  }
}



int cudaResourcePool::getNumStreams()
{
  poolCleaning(); // clean out resources where reallocation of resources might have failed 
  return _sRes.size();
}

string cudaResourcePool::getLogHeader()
{
  ostringstream headerinfo;

  headerinfo << "CUDA " << _devId << " ResourcePool:";

  return headerinfo.str();
}

void cudaResourcePool::printMemoryUsage()
{
  size_t free_byte ;
  size_t total_byte ;
  cudaMemGetInfo( &free_byte, &total_byte ) ;

  double free_db = (double)free_byte ;
  double total_db = (double)total_byte ;
  double used_db = total_db - free_db ;
  cout << getLogHeader() << " GPU memory usage: used = " << used_db/1024.0/1024.0<< ", free = " << free_db/1024.0/1024.0<< " MB, total = "<< total_db/1024.0/1024.0<<" MB" << endl;
}

size_t cudaResourcePool::requestDeviceMemory(size_t size)
{
  _SrequestedDeviceSize= (_SrequestedDeviceSize> size)?(_SrequestedDeviceSize):(size);
  return _SrequestedDeviceSize;
}
size_t cudaResourcePool::requestHostMemory(size_t size)
{
  _SrequestedHostSize = (_SrequestedHostSize > size)?(_SrequestedHostSize):(size);
  return _SrequestedHostSize;
}


