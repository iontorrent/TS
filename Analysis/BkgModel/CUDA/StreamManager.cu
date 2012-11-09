/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>

#include "cuda_error.h"
#include "cuda_runtime.h"


#include "Utils.h"
#include "StreamManager.h"


using namespace std;


bool cudaStreamExecutionUnit::_verbose = false;

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
      cout << "CUDA: Device "<< _devId << " Stream " << i << " ("<< ret <<") taken out of StreamPool, " << _numHandles[_devId] << " streams are in use" << endl;
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
      cout << "CUDA: Device "<< _devId << " Stream " << i << " (" << stream << ") put back into StreamPool, " << _numHandles[_devId] << " streams are in use" << endl;

      break;
    }
  }

  if(_numHandles[_devId] == 0) destroyStreams();
  //del_pInstance
  UnLock(); //pthread_mutex_unlock(&_lock);
}


///////////////////////////////////////////////////////

sharedStreamData::sharedStreamData()
{
  _seuCounter = 0;
  _seuWorking = 0;
  _numBlock = 0;
  _timesum = 0;
  _done = false;
  _tasksComplete = 0;
  wakeUpCall();
  _inQ = NULL;
  _event = NULL;

}

sharedStreamData::~sharedStreamData()
{
  //signal to that all jobsd are done 
  if (_inQ != NULL){
     cout << "signal to Queue that all jobs are done and cleanup is complete" << endl;  
    _inQ->DecrementDone();
  }
#ifdef BLOCKING_EVENT
  if(_event != NULL) cudaEventDestroy(_event);CUDA_ERROR_CHECK();
#endif
}


int sharedStreamData::incSEUcnt()
{
 _seuCounter++ ;
 return _seuCounter;
}
int sharedStreamData::decSEUcnt()
{
  _seuCounter--;
  return _seuCounter;
}

// sleep handling  
void sharedStreamData::setSleeping(int id)
{
  _sleepMask[id] = true;
}


void sharedStreamData::wakeUpCall()
{
  for(int i=0; i< MAX_NUM_STREAMS; i++) _sleepMask[i] = false;
}

bool sharedStreamData::isSleeping(int id)
{
  return _sleepMask[id];  
}

bool sharedStreamData::allSleeping()
{
  int seuSleeping = 0;
  for(int i=0; i< MAX_NUM_STREAMS; i++) if(_sleepMask[i]) seuSleeping++;
  return (seuSleeping==_seuCounter)?(true):(false);  
}

#ifdef BLOCKING_EVENT
// wait handling

void  sharedStreamData::createEvent()
{
  cout << "CUDA: creating Event to block cpu thread while GPU is busy." << endl;
  cudaEventCreateWithFlags(&_event, cudaEventBlockingSync); CUDA_ERROR_CHECK();
  clearEvent();
}


void sharedStreamData::setEvent(int id, cudaStream_t stream){
  if (_waitEventMask[id] != true)
  {
//    cout << "CUDA STREAM " << id << " EVENT: recording event" << endl;
    cudaEventRecord(_event, stream );
    _waitEventMask[id] = true;
  }
}
void sharedStreamData::waitEvent(int id){
  if (allWaiting())
  {
//    cout << "CUDA STREAM " << id << " EVENT: going into blocking mode" << endl;
    cudaEventSynchronize(_event); CUDA_ERROR_CHECK();
//    cout << "CUDA STREAM " << id << " EVENT: waking up" << endl; 
  }
}

void sharedStreamData::clearEvent(int id)
{
//  cout << "CUDA STREAM " << id << " EVENT: done waiting" << endl; 
  _waitEventMask[id] = false;
}
void sharedStreamData::clearEvent()
{
  for(int i=0; i< MAX_NUM_STREAMS; i++) _waitEventMask[i] = false;
}


bool sharedStreamData::allWaiting()
{
  int seuWaiting = 0;
  for(int i=0; i< MAX_NUM_STREAMS; i++) if(_waitEventMask[i]) seuWaiting++;
  return (seuWaiting==_seuCounter)?(true):(false);  
}
#endif


void sharedStreamData::setDone()
{
  _done = true;
}

bool sharedStreamData::isDone()
{
  return _done;
}

void sharedStreamData::incTasksComplete()
{
  _tasksComplete++;
}

int sharedStreamData::getTasksComplete()
{
  return _tasksComplete;
}


void sharedStreamData::startTimer(){
  if(_seuWorking == 0){
    _T.restart();
  }
  _seuWorking++;
}

void sharedStreamData::stopTimer(){
  _seuWorking--; 
  if(_seuWorking == 0){
    _timesum += _T.elapsed();
  }
}

double sharedStreamData::getBusyTime()
{
  return _timesum;
}

double sharedStreamData::getAvgTimePerJob()
{
  return (_tasksComplete > 0)?(_timesum/_tasksComplete):(0);
}


bool sharedStreamData::isSet()
{
  return (_inQ!=NULL);
}

bool sharedStreamData::setQ(WorkerInfoQueue * Q)
{
  if(!isSet()){ 
    _inQ = Q;
#ifdef BLOCKING_EVENT
    createEvent(); 
#endif
    return true;
  }
  if(_inQ != Q) return false;
  return true;
}

// set Static Class variable to check how many streams got invoked
/*int cudaStreamExecutionUnit::_seuCounter = 0;
int cudaStreamExecutionUnit::_seuSleeping = 0;
int cudaStreamExecutionUnit::_seuWorking = 0;
double cudaStreamExecutionUnit::_timesum = 0;
bool cudaStreamExecutionUnit::_done = false;
Timer cudaStreamExecutionUnit::_T;
i*/


///////////////////////////////////////////////////////////////////
// STREAM EXECUTION UNIT

cudaStreamExecutionUnit::cudaStreamExecutionUnit( WorkerInfoQueue *Q )
{

  _state = GetJob; 
  
  _streamPool = cudaStreamPool::Instance();  
  
  _tasks = 0;
  _seuId = -1;
 

  _inQ = Q;

  setName("StreamExecutionUnit");
  
  _stream = _streamPool->getStream();
 // cudaStreamCreate(&_stream);
  if(_stream == NULL ) throw cudaStreamCreationError(__FILE__,__LINE__); 


}


cudaStreamExecutionUnit::~cudaStreamExecutionUnit()
{
  if(_verbose) cout << _name << " " << _seuId << " Destroying" << endl;
	if(_sd != NULL && _seuId >= 0){ 
    _sd->decSEUcnt();

    cout << "CUDA: " << getName() << " " << _seuId << " Stream released: " << _stream << endl;
    if(_stream != NULL){ 
      //cudaStreamDestroy(_stream); CUDA_ERROR_CHECK();
      //cudaThreadSynchronize();
      _streamPool->releaseStream(_stream); 
    }
    
  }
}

void cudaStreamExecutionUnit::setName(char * name)
{
  strcpy(_name, name);
}

void cudaStreamExecutionUnit::init(int id, sharedStreamData * sd)
{
   
  _seuId = id;
  if(_verbose) cout << _name << " " << _seuId <<" init SEU "<< endl; 

  for(int i = 0; i < MAX_NUM_STREAMS ; i++){
    if(sd[i].setQ(_inQ)){ 
      _sd = &sd[i];
      if(_verbose) cout << _name << " " << _seuId <<" linked to shared data object "<< i << endl; 
      break;
    }
  }

     

  _sd->incSEUcnt();

}

//////////////////////////////
// TASK EXECUTION FUNCTIONS

bool cudaStreamExecutionUnit::execute(int * control)
{

  if(_seuId < 0  || _sd == NULL) return false; // not initialized 
//SEU state machine

  switch(_state)
  {

    case GetJob:
      if(_sd->isDone()){ 
        _state = Exit; if(_verbose) cout << _name << " " << _seuId << " -> Exiting" << endl;
        break;
      }
      if(getNewJob()){ 
             _state = Working; if(_verbose) cout << _name << " " << _seuId << " -> Working" << endl;
      }else{
        _state = Sleeping; if(_verbose) cout << _name << " " << _seuId << " -> Sleeping" << endl;
        _sd->setSleeping(_seuId);
      }
      break;

    case Working: 
      if(noMoreWork()){
        //_inQ->DecrementDone(); 
        _sd->setDone();
        _state = Exit; if(_verbose) cout << _name << " " << _seuId << " -> Exiting" << endl;
        break;
      }
      if(!ValidJob()){
        _inQ->DecrementDone();
        _state = GetJob; if(_verbose) cout << _name << " " << _seuId << " No Vaid Job ->  GetJob" << endl;
        break;
      }      
      _sd->wakeUpCall();
      _sd->startTimer();
    case ContinueWork:
      ExecuteJob(control);
      _state = Waiting; if(_verbose) cout << _name << " " << _seuId << " -> Waiting" << endl;
#ifdef BLOCKING_EVENT
      _sd->setEvent(_seuId, _stream); 
#endif
      break;
    
    case Waiting:
 #ifdef BLOCKING_EVENT
      _sd->waitEvent(_seuId);
#endif
      if(checkComplete()){ // check if compeleted   
        if( handleResults() > 0 ){
          _state = ContinueWork; if(_verbose) cout << _name << " " << _seuId << "-> ContinueWork" << endl;
        }else{ 
          _inQ->DecrementDone();
          _sd->incTasksComplete();
          _tasks++;
          _sd->stopTimer();
          if(_verbose) cout << _name << " " << _seuId << " completed: " << _tasks  << endl;
          _state = GetJob; if(_verbose) cout << _name << " " << _seuId << " -> Getjob" << endl;
        }
#ifdef BLOCKING_EVENT
        _sd->clearEvent(_seuId);
#endif
      }
      break;
   
    case Sleeping:
      if(_sd->isDone()){
        _state = Exit; if(_verbose) cout << _name << " " << _seuId << " -> Exiting" << endl;
        break;
      }
      if(!_sd->isSleeping(_seuId)){ 
        _state = GetJob; if(_verbose) cout << _name << " " << _seuId << " -> GetJob" << endl;
      }
      if(_sd->allSleeping()){ 
        _state = Blocking; if(_verbose) cout << _name << " " << _seuId << " -> Blocking" << endl;
      }else 
        break; // if all are sleeping drop through to Blocking (hack to prevent more than one stream reaching blocking state)
    
    case Blocking:
      getNewJobBlocking();     
      _sd->wakeUpCall();
      _state = Working; if(_verbose) cout << _name << " " << _seuId << " -> Working" << endl;
      break;
 
    case Exit:
    default:
      return false;
  }

  return true;
}



bool cudaStreamExecutionUnit::getNewJob()
{ 

  _item = _inQ->TryGetItem();
 
  if(_item.private_data != NULL)
    return true;

  return false;
}

void cudaStreamExecutionUnit::getNewJobBlocking()
{ 
  _item = _inQ->GetItem();
}


void * cudaStreamExecutionUnit::getJobData()
{ 
  return (void*)_item.private_data;
}


bool cudaStreamExecutionUnit::noMoreWork()
{ 
  return _item.finished;
}

bool cudaStreamExecutionUnit::checkComplete()
{
	if(cudaSuccess != cudaStreamQuery(_stream)	) return false;
 	return true;
}


int cudaStreamExecutionUnit::getNumTasks()
{
	return _tasks;
}


bool cudaStreamExecutionUnit::Verbose()
{
	return _verbose;
}

char * cudaStreamExecutionUnit::getName()
{
  return _name;
}

void cudaStreamExecutionUnit::printInfo()
{
  cout << _name << " " << _seuId;
}
/*
void cudaStreamExecutionUnit::setId(int id)
{
  _seuId = id;
}*/

int cudaStreamExecutionUnit::getId()
{
  return _seuId;
}


///////////////
// static functions

void cudaStreamExecutionUnit::setVerbose(bool flag)
{
  _verbose = flag;
}


//////////////////////////////////////////////////////
// STREAM MANAGER



cudaStreamManager::cudaStreamManager()
{

  cudaGetDevice( &_dev_id );

  cout << "CUDA: Device "<< _dev_id << " StreamManager created" << endl;
  for(int i = 0; i < MAX_NUM_STREAMS ; i++) _Stream[i] = NULL;
  _numStreams = 0;
}


cudaStreamManager::~cudaStreamManager()
{

  for(int i=0; i < MAX_NUM_STREAMS; i++){
    if(_sd[i].isSet()){
//      cout << "outside time: " << _sd[i].getOverallTime() << "s " << endl;
      cout << "CUDA: Device " << _dev_id << " All Streams finished: " << _sd[i].getTasksComplete() << " in " <<  _sd[i].getBusyTime()  << "s, time/job: " << _sd[i].getAvgTimePerJob() << endl; 
    }
  }
  destroyStreams();

}


int cudaStreamManager::addStreamUnit( cudaStreamExecutionUnit * seu)
{

  for(int i=0; i<MAX_NUM_STREAMS; i++){
    if(_Stream[i] ==NULL){ 
      _Stream[i] = seu; 
      _Stream[i] -> init(i, _sd);
      _numStreams++;
      return i;
    }
  }

  return -1;
 }


  
void cudaStreamManager::destroyStream(int id)
{

  if(_Stream[id] != NULL){ 
    cout << "CUDA: Device " << _dev_id << " destroying "<< _Stream[id]->getName()  << " " << id << " out of " << _numStreams << " after completing: " <<  _Stream[id]->getNumTasks() <<" jobs" <<  endl;
      delete _Stream[id];
    _Stream[id] = NULL;
    _numStreams--;
  } 
}

int cudaStreamManager::getNumStreams()
{
  return _numStreams;
}

void cudaStreamManager::destroyStreams()
{
  for(int i=0; i<MAX_NUM_STREAMS; i++){
    destroyStream(i);
  }
   //cudaThreadSynchronize();
}  

bool cudaStreamManager::DoWork(int * control)
{
  int notDone = 0;
  for(int i=0; i<MAX_NUM_STREAMS; i++){
    if(_Stream[i] != NULL)
      if(_Stream[i]->execute(control)) notDone++;
      else destroyStream(i);
  }
  //return true if still work to do
   return (notDone>0)?(true):(false);
}

void cudaStreamManager::printMemoryUsage()
{
  size_t free_byte ;
  size_t total_byte ;
  cudaMemGetInfo( &free_byte, &total_byte ) ;
  int dev_id;
  cudaGetDevice( &dev_id );

  double free_db = (double)free_byte ;
  double total_db = (double)total_byte ;
  double used_db = total_db - free_db ;
  cout << "CUDA: Device " << dev_id << " GPU memory usage: used = " << used_db/1024.0/1024.0<< ", free = " << free_db/1024.0/1024.0<< " MB, total = "<< total_db/1024.0/1024.0<<" MB" << endl;
}


