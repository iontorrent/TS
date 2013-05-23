/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

// patch for CUDA5.0/GCC4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <iostream>

#include "ResourcePool.h"


using namespace std;

/*
cudaStreamPool * cudaStreamPool::_pInstance = NULL;
pthread_mutex_t cudaStreamPool::_lock;  //forward declaration od static lock
bool cudaStreamPool::_init = false;
int cudaStreamPool::_devId = -1;

/////////////////////////////////////////////////////////

void cudaStreamPool::initLockNotThreadSafe()
{

  if (!_init) pthread_mutex_init(&_lock, NULL);
  _init = true;
}

void cudaStreamPool::destroyLockNotThreadSafe()
{
  if (_init) pthread_mutex_destroy(&_lock);
  _init = false;
}

cudaStreamPool::cudaStreamPool()
{
  for(int i =0; i <MAX_NUM_DEVICES; i++) _numHandles[i] = 0;
}

void cudaStreamPool::Lock()
{
  pthread_mutex_lock(&_lock);
  cudaGetDevice(&_devId); CUDA_ERROR_CHECK();
}

void cudaStreamPool::UnLock()
{
  _devId = -1; // set devId to invaliValue to casue an error if I made a mistake here
  pthread_mutex_unlock(&_lock);
}


void cudaStreamPool::createStreams()
{
  if(_numHandles[_devId] == 0){ //if no stream has been taken out of the pool it was not created yet
    cout << "CUDA: Device "<< _devId << " creating StreamPool with " << MAX_NUM_STREAMS << "." << endl;
    for(int i=0; i< MAX_NUM_STREAMS; i++){ 
      _inUse[_devId][i] = false;
      _streams[_devId][i] = NULL;
      cudaStreamCreate(&_streams[_devId][i]);
      cudaError_t err = cudaGetLastError();  
      if(_streams[_devId][i] == NULL || err != cudaSuccess) _inUse[_devId][i]=true;   //throw cudaStreamCreationError(__FILE__,__LINE__); 
    }
  }
}

void cudaStreamPool::destroyStreams()
{
  cout << "CUDA: Device "<< _devId << " destroying StreamPool." << endl;

  for(int i=0; i< MAX_NUM_STREAMS; i++){ 
    _inUse[_devId][i] = false;
    cudaStreamCreate(&_streams[_devId][i]);
    if(_streams[_devId][i] != NULL){ 
      cudaStreamDestroy(_streams[_devId][i]); CUDA_ERROR_CHECK(); 
    }
  }
  cudaThreadSynchronize();
}



cudaStreamPool* cudaStreamPool::Instance()  // thread safe
{
  Lock();//pthread_mutex_lock(&_lock);
  if(_pInstance == NULL) _pInstance = new cudaStreamPool();
  UnLock(); //pthread_mutex_unlock(&_lock);
  return _pInstance;
}

cudaStream_t cudaStreamPool::getStream()
{
  cudaStream_t ret = NULL;
  Lock(); //pthread_mutex_lock(&_lock);

  createStreams(); // if not created yet, create streams on device

  for(int i=0; i < MAX_NUM_STREAMS; i++){
    if(_streams[_devId][i] != NULL && !_inUse[_devId][i]){
      _inUse[_devId][i] = true;
      ret = _streams[_devId][i];
      _numHandles[_devId]++;
 //     cout << "CUDA: Device "<< _devId << " Stream " << i << " ("<< ret <<") taken out of StreamPool, " << _numHandles[_devId] << " streams are in use" << endl;
      break;
    }
  }
  UnLock(); //pthread_mutex_unlock(&_lock);
  return ret; 
}

void cudaStreamPool::releaseStream(cudaStream_t stream)
{
  Lock(); //pthread_mutex_lock(&_lock);
  for(int i=0; i < MAX_NUM_STREAMS; i++){
    if(_streams[_devId][i] == stream){
      _inUse[_devId][i] = false;
      _numHandles[_devId]--;
//      cout << "CUDA: Device "<< _devId << " Stream " << i << " (" << stream << ") put back into StreamPool, " << _numHandles[_devId] << " streams are in use" << endl;

      break;
    }
  }

  if(_numHandles[_devId] == 0) destroyStreams();
  //del_pInstance
  UnLock(); //pthread_mutex_unlock(&_lock);
}

*/


///////////////////////////////////////////////////////




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


string streamResources::getLogHeader()
{
  ostringstream headerinfo;

  headerinfo << "CUDA " << _devId << " StreamResource " << _streamId << ":";

  return headerinfo.str();
}
////////////////////////////////////////////////////////////



 

cudaResourcePool::cudaResourcePool(size_t hostsize, size_t devicesize, int numStreams)
{
  _HostSize = hostsize;
  _DeviceSize = devicesize;
  cudaGetDevice(&_devId);


  tryAddResource(numStreams);  

  if(_sRes.empty()) throw   cudaStreamCreationError( __FILE__,__LINE__);

}

void cudaResourcePool::tryAddResource(unsigned int numStreams)
{

  streamResources * srestmp;

  while(_sRes.size() < numStreams){
    int i = _sRes.size();
    srestmp = NULL;
    try{
     printMemoryUsage();
     cout << getLogHeader() << " trying to create stream resources "<<  i  << endl;
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




