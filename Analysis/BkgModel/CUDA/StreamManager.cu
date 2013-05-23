/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

// patch for CUDA5.0/GCC4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <iostream>

#include "cuda_error.h"
#include "cuda_runtime.h"


#include "Utils.h"
#include "ResourcePool.h"
#include "StreamManager.h"
#include "SignalProcessingFitterQueue.h"
#include "SingleFitStream.h"
#include "MultiFitStream.h"

using namespace std;

bool cudaSimpleStreamExecutionUnit::_verbose = false;
int cudaSimpleStreamExecutionUnit::_seuCnt = 0;

bool cudaSimpleStreamManager::_verbose =false;


//////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
// SIMPLE STREAM EXECUTION UNIT


cudaSimpleStreamExecutionUnit::cudaSimpleStreamExecutionUnit( streamResources * resources,  WorkerInfoQueueItem item )
{

  _state = Init; 
  
  _seuNum = _seuCnt++;
  
  _computeVersion = 35; //default set to latest

  setName("StreamExecutionUnit");
  
  _item = item;
  _resources = resources;

  if(_resources == NULL) throw cudaStreamCreationError(__FILE__,__LINE__); 

  _stream = _resources->getStream();
  _Host = _resources->getHostMem();
  _Device = _resources->getDevMem();

}


cudaSimpleStreamExecutionUnit::~cudaSimpleStreamExecutionUnit()
{
  if(_verbose) cout << getLogHeader() << " Completed, releasing Stream Resources" << endl;
    _resources->release();    
}

void cudaSimpleStreamExecutionUnit::setName(char * name)
{
  _name = name;
}

//////////////////////////////
// TASK EXECUTION FUNCTIONS

bool cudaSimpleStreamExecutionUnit::execute()
{

  switch(_state)
  {

    case Init:
      if(!InitValidateJob()){
        _state = Exit; if(_verbose) cout << getLogHeader() << " No Valid Job ->  Exit" << endl;
        break;
      }     
      _state = Working; if(_verbose) cout << getLogHeader() << " Init -> Working" << endl;
      break;

    case Working:
      // starttimer() 
    case ContinueWork:
      ExecuteJob();
      _state = Waiting; if(_verbose) cout << getLogHeader() << " -> Waiting" << endl;
      break;
    
    case Waiting:
      if(checkComplete()){ // check if compeleted  
        if( handleResults() > 0 ){
          _state = ContinueWork; if(_verbose) cout << getLogHeader() << " -> ContinueWork" << endl;
        }else{
          //stopTimer();
          _state = Exit ; if(_verbose) cout << getLogHeader() << " -> Exit" << endl;
        }
      }
      break;
    case Exit:
    default:
      return false; // retunr false if all is done!
  }
  // return true if there is still work to be done
  return true;
}


void * cudaSimpleStreamExecutionUnit::getJobData()
{ 
  return (void*)_item.private_data;
}


WorkerInfoQueueItem cudaSimpleStreamExecutionUnit::getItem()
{ 
  return _item;
}

bool cudaSimpleStreamExecutionUnit::checkComplete()
{
  cudaError_t ret;
  ret = cudaStreamQuery(_stream);

	if( ret == cudaErrorNotReady	) return false;
	if( ret == cudaSuccess) return true;
  ret = cudaGetLastError(); 
  throw cudaExecutionException(ret, __FILE__,__LINE__);
//  return false;
}


/*
WorkerInfoQueueItem * cudaSimpleStreamExecutionUnit::getItemAndReset()
{
  _sd->setError();
  if( _state > GetJob && _state < Sleeping){ 
    _state = GetJob;
    return &_item;
  }
  return NULL;
}*/


bool cudaSimpleStreamExecutionUnit::Verbose()
{
	return _verbose;
}

string cudaSimpleStreamExecutionUnit::getName()
{
  return _name;
}

string cudaSimpleStreamExecutionUnit::getLogHeader()
{
  ostringstream headerinfo;

  headerinfo << "CUDA " << _resources->getDevId() << " SEU " << getSeuNum() << " " << getName() << " SR " << getStreamId()<< ":";

  return headerinfo.str();
}

int cudaSimpleStreamExecutionUnit::getSeuNum()
{
  return _seuNum;
}

int cudaSimpleStreamExecutionUnit::getStreamId()
{
  return _resources->getStreamId();
}


bool cudaSimpleStreamExecutionUnit::InitValidateJob() {

    _myJob.setData(static_cast<BkgModelWorkInfo *>(getJobData())); 

    return _myJob.ValidJob();
}


void cudaSimpleStreamExecutionUnit::setCompute(int compute)
{
  _computeVersion = compute;
}
int cudaSimpleStreamExecutionUnit::getCompute()
{
  return _computeVersion;
}




///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// SIMPLE STREAM MANAGER



cudaSimpleStreamManager::cudaSimpleStreamManager( WorkerInfoQueue * inQ, int numStreams)
{

  cudaGetDevice( &_devId );

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, _devId);

  _computeVersion = 10*deviceProp.major + deviceProp.minor; 
   
  cout << getLogHeader() << " init with up to "<< numStreams <<" Stream Execution Units" << endl;
  _inQ = inQ; 
  _resourcePool = NULL;
  _item.finished = false;
  _GPUerror = false;
  _maxNumStreams = numStreams;
  allocateResources();
  _tasks = 0;
 

}



cudaSimpleStreamManager::~cudaSimpleStreamManager()
{
  
  freeResources(); 
  
  ostringstream outputStr;
   
  outputStr << getLogHeader() << " handled " << _tasks <<" tasks."<< endl;

 for ( map< string, TimeKeeper>::iterator iter = _timer.begin(); iter != _timer.end(); ++iter )
      outputStr << getLogHeader() << " " << iter->first << " finished: " << iter->second.getJobCnt() << " in " << iter->second.getTime() << " time/job: " << iter->second.getAvgTime() << endl;

  cout << outputStr.str();

}




void cudaSimpleStreamManager::allocateResources()
{

  size_t maxHostSize = getMaxHostSize();
  size_t maxDeviceSize = getMaxDeviceSize(MAX_PREALLOC_COMPRESSED_FRAMES_GPU);


  try{
    _resourcePool = new cudaResourcePool(maxHostSize,maxDeviceSize,_maxNumStreams); // throws cudaException
    cout << getLogHeader() << " one Stream Execution Unit requires "<< maxHostSize/(1024.0*1024.0)  << "MB Host and " << maxDeviceSize/(1024.0*1024.0) << "MB Devcie memory" << endl;
    int n = getNumStreams();
    if (n > 0){ cout <<  getLogHeader() <<" successfully aquired resources for " << n << " Stream Execution Units" <<endl;
    }
    _GPUerror = false;

  }
  catch(exception &e){
    cout << e.what() << endl;
    cout << getLogHeader() << " No StreamResources could be aquired! retry pending. jobs will he handled by CPU for now!" << endl;
    _GPUerror = true;
    _resourcePool = NULL;
  }

}


  
void cudaSimpleStreamManager::freeResources()
{
  if(_resourcePool != NULL) delete _resourcePool;
  _resourcePool = NULL;

  if(_item.finished && _inQ !=NULL){
    cout << getLogHeader() << " signaling Queue that all jobs and cleanup completed" << endl;
    _inQ->DecrementDone();
  }
}

int cudaSimpleStreamManager::getNumStreams()
{
  if(_resourcePool != NULL) return _resourcePool->getNumStreams();
  return 0;
}

int cudaSimpleStreamManager::availableResources()
{
  // calcualte free SEUs #allocated stream resources - #active SEUS
  return getNumStreams() - _activeSEU.size();
}



size_t cudaSimpleStreamManager::getMaxHostSize()
{
  size_t ret = 0;
  if(SimpleSingleFitStream::getMaxHostMem() > ret) ret = SimpleSingleFitStream::getMaxHostMem();
  if(SimpleMultiFitStream::getMaxHostMem() > ret) ret = SimpleMultiFitStream::getMaxHostMem();
  return ret;
}

size_t cudaSimpleStreamManager::getMaxDeviceSize(int maxFrames, int maxBeads)
{
  size_t ret = 0;
  if(SimpleSingleFitStream::getMaxDeviceMem(maxFrames, maxBeads) > ret) ret = SimpleSingleFitStream::getMaxDeviceMem(maxFrames, maxBeads);
  if(SimpleMultiFitStream::getMaxDeviceMem(maxFrames, maxBeads) > ret) ret = SimpleMultiFitStream::getMaxDeviceMem(maxFrames,maxBeads);
  return ret;
}


void cudaSimpleStreamManager::moveToCPU()
{
  //get jobs and hand them over to the CPU Q after GPU error was encountered
  getJob();
  if(!_item.finished && _item.private_data != NULL ){ 
    if(_verbose) cout << getLogHeader()<< " job received, try to reallocate resources!" << endl;
    //try to allocate and recover before handing job to CPU
    if(_resourcePool == NULL){
      allocateResources();
      if( getNumStreams()  > 0){ 
        cout << getLogHeader()<< " managed to aquire streamResources, switching execution back to GPU!" << endl;
        addSEU();
        _GPUerror = false;
        return;
      }
    }
    if(_verbose) cout << getLogHeader() << " handing job on to CPU queue" << endl;
    BkgModelWorkInfo *info = (BkgModelWorkInfo *) (_item.private_data);
    info->pq->GetCpuQueue()->PutItem(_item); // if no matching SEU put to CpuQ
    _inQ->DecrementDone(); // signale to Q that a job is completed
  } 
}


void cudaSimpleStreamManager::getJob()
{
  if(!_item.finished){
    if(_activeSEU.empty()){
       if(_verbose) cout << getLogHeader()<< " blocking Job request" << endl;
      _item = _inQ->GetItem();  //if no active SEUs block on Q
    }
    else{
      _item = _inQ->TryGetItem();
    }

    if(_item.finished){
      cout << getLogHeader()<< " received finish job" << endl;
    }
  }
}


// dependiong on the type of job that was received from the inQ
// creates the according SEU type or hands it back to the CPU
// new GPU Jobs have to be added to this switch/case statement
void cudaSimpleStreamManager::addSEU()
{
  //create a SEU if item is not the finish item
  if(!_item.finished && _item.private_data != NULL ){ 
    if(_verbose) cout << getLogHeader()<< " job received, checking type and create SEU" << endl;
   
    BkgModelWorkInfo *info = (BkgModelWorkInfo *) (_item.private_data);
    cudaSimpleStreamExecutionUnit * tmpSeu = NULL;
    try{
      switch(info->type){
        case INITIAL_FLOW_BLOCK_ALLBEAD_FIT:
          if(_verbose) cout << getLogHeader()<< " got MultiFit " << endl;
          tmpSeu = new SimpleMultiFitStream( _resourcePool->getResource(), _item);          
          break;
        case SINGLE_FLOW_FIT:
          if(_verbose) cout << getLogHeader()<< " got SingleFit " << endl;
          tmpSeu = new SimpleSingleFitStream( _resourcePool->getResource(), _item);
          break;
        default:        
          if(_verbose) cout << getLogHeader()<< " received unknown item" << endl;
          info->pq->GetCpuQueue()->PutItem(_item); // if no matching SEU put to CpuQ
          _inQ->DecrementDone(); // signale to Q that a job is completed
      }
      //set the compute version accorsitn to the device the streamManager is initiated for 
      tmpSeu->setCompute(_computeVersion);
    }
    catch(cudaException &e){
      if(_resourcePool->getNumStreams() == 0){ 
        cout << " *** ERROR DURING STREAM UNIT CREATION, handing job back to CPU" << endl;
         info->pq->GetCpuQueue()->PutItem(_item); // if no matching SEU put to CpuQ
        _inQ->DecrementDone(); // signale to Q that a job is completed
        _GPUerror = true;
      }else{ 
         info->pq->GetGpuQueue()->PutItem(_item); // if no matching SEU put to CpuQ
        _inQ->DecrementDone(); // signale to Q that a job is completed
      }


    }
    if(tmpSeu != NULL){
      // startTimer();
      _timer[tmpSeu->getName()].start();
      _activeSEU.push_back(tmpSeu);
    }
  } 
}


//handles the execution of all SEU in the StreamManager
//cleans up the SEUs that completed their jobs
void cudaSimpleStreamManager::executeSEU()
{
  bool workDone = false;

  for(int i=(_activeSEU.size()-1); i >= 0 ; i--){ // iterate backwards for easy delete
    if(_activeSEU[i] != NULL){ // saftey, should not be possible to be NULL
      try{
         workDone = !(_activeSEU[i]->execute());
      }
      catch(cudaException &e){ 
                //if execution exception, get item and habd it back tio CPU
        //cout << e.what() << endl;
        if(_verbose) e.Print();
       
        _item = _activeSEU[i]->getItem(); 
        BkgModelWorkInfo *info = (BkgModelWorkInfo *) (_item.private_data);
     
        if(_resourcePool->getNumStreams() == 0){ 
          cout << getLogHeader() << "*** ERROR DURING STREAM EXECUTION, handing incomplete Job back to CPU for retry" << endl;
          info->pq->GetCpuQueue()->PutItem(_item); // if no matching SEU put to CpuQ
          _GPUerror = true;
        }else{ 
          cout << getLogHeader() << "*** ERROR DURING STREAM EXECUTION, " << _resourcePool->getNumStreams() << " StreamResources still avaiable, retry pending" << endl;
          info->pq->GetGpuQueue()->PutItem(_item); // if no matching SEU put to CpuQ
        }
        workDone = true; // mark work as done so SEU gets cleaned up
        _tasks--; // decrease task count for GPU

      } // end catch block

      if(workDone){
        _timer[_activeSEU[i]->getName()].stop();
        delete _activeSEU[i]; // destroy SEU object
        _activeSEU.erase(_activeSEU.begin()+i); //delete SEU from active list
        _inQ->DecrementDone(); // signale to Q that a job is completed
        _tasks++;
      }
    }
  }
}


//perform actual work, polls jobs from inQ and executes them until finish-job received
bool cudaSimpleStreamManager::DoWork()
{  
  
  if(_inQ == NULL){
      cout << getLogHeader() << " No valid queue handle provided!" << endl;
      return false;
  }
 
  bool notDone = true;
  while(notDone){
      if(_GPUerror ){
        moveToCPU();
      }else{
        if(availableResources() > 0){          // if resources availabel get job          
          getJob(); // get a Job from Q, block on Q if no job not already working
          addSEU(); // try to add whatever getJob aquired
        }      
      }
      // drive the statemachine of the state execution units
      // and clean up when a SEU is don
      executeSEU();
      // as long as no finish job received and there are still active SEUs 
      // in the list we are not none yet
      notDone =  (_activeSEU.empty() && _item.finished)?(false):(true);
  }
  return false;
}

string cudaSimpleStreamManager::getLogHeader()
{
  ostringstream headerinfo;

  headerinfo << "CUDA " << _devId << " StreamManager:";

  return headerinfo.str();
}



/////////////////////////////////

TimeKeeper::TimeKeeper()
{
  _timesum = 0;
  _activeCnt = 0;
  _jobCnt = 0;
}


void TimeKeeper::start(){
  if( _activeCnt == 0)
      _T.restart();
  _activeCnt++;
  _jobCnt++;
}

void TimeKeeper::stop(){
  _activeCnt--;
  if(_activeCnt == 0){
    _timesum += _T.elapsed();
  }
}

double TimeKeeper::getTime()
{
  return _timesum;
}

double TimeKeeper::getAvgTime()
{
  return (_jobCnt > 0)?(_timesum/_jobCnt):(0);
}
int TimeKeeper::getJobCnt()
{
  return _jobCnt;
}



