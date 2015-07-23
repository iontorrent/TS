/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>

#include "cuda_error.h"
#include "cuda_runtime.h"


#include "Utils.h"
#include "ResourcePool.h"
#include "StreamManager.h"
#include "SignalProcessingFitterQueue.h"

//job types
#include "SingleFitStream.h"
#include "MultiFitStream.h"
//#include "GenerateBeadTraceStream.h"


#define MAX_EXECUTION_ERRORS 10
#define NUM_ERRORS_TOGGLE_VERBOSE 5

using namespace std;

bool cudaSimpleStreamExecutionUnit::_verbose = false;
int cudaSimpleStreamExecutionUnit::_seuCnt = 0;

bool cudaSimpleStreamManager::_verbose =false;
int cudaSimpleStreamManager::_maxNumStreams = MAX_ALLOWED_NUM_STREAMS;

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
  _resource = resources;

  if(_resource == NULL) throw cudaStreamCreationError(__FILE__,__LINE__);

  _stream = _resource->getStream();

}


cudaSimpleStreamExecutionUnit::~cudaSimpleStreamExecutionUnit()
{
  if(_verbose) cout << getLogHeader() << " Completed, releasing Stream Resources" << endl;
  _resource->release();
}

void cudaSimpleStreamExecutionUnit::setName(std::string name)
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
    if(!InitJob()){
      _state = Exit; if(_verbose) cout << getLogHeader() << " No Valid Job ->  Exit" << endl;
      break;
    }
    _state = Working; if(_verbose) cout << getLogHeader() << " Init -> Working" << endl;
    break;

  case Working: // starting work state
  case ContinueWork:  //continue work state after non async code
    ExecuteJob();
    _state = Waiting; if(_verbose) cout << getLogHeader() << " -> Waiting" << endl;
    break;

  case Waiting:
    if(checkComplete()){ // check if completed
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
    return false; // return false if all is done!
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

  if( ret == cudaErrorNotReady  ) return false;
  if( ret == cudaSuccess) return true;
  ret = cudaGetLastError();
  throw cudaExecutionException(ret, __FILE__,__LINE__);
  //  return false;
}


void cudaSimpleStreamExecutionUnit::setVerbose(bool v)
{
  _verbose = v;
}

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

  headerinfo << "CUDA " << _resource->getDevId() << " SEU " << getSeuNum() << " " << getName() << " SR " << getStreamId()<< ":";

  return headerinfo.str();
}

int cudaSimpleStreamExecutionUnit::getSeuNum()
{
  return _seuNum;
}

int cudaSimpleStreamExecutionUnit::getStreamId()
{
  return _resource->getStreamId();
}

// need to be overloaded to return true if initiated correctly
bool cudaSimpleStreamExecutionUnit::InitJob() {

  return false;
}


//Factory Method to produce specialized SEUs
cudaSimpleStreamExecutionUnit * cudaSimpleStreamExecutionUnit::makeExecutionUnit(streamResources * resources, WorkerInfoQueueItem item)
{
  cudaSimpleStreamExecutionUnit * tmpSeu = NULL;

  ostringstream headerinfo;
  headerinfo << "CUDA " << resources->getDevId() << " SEU Factory SR "<< resources->getStreamId() << ":";

  int *type = (int*)item.private_data;

  switch(*type){
 /* case GENERATE_BEAD_TRACES:
      if(_verbose) cout << headerinfo.str()<< " creating GenerateAllBeadTraces " << endl;
      tmpSeu = new GenerateBeadTraceStream( resources, item);
      break;*/
  case INITIAL_FLOW_BLOCK_ALLBEAD_FIT:
    if(_verbose) cout << headerinfo.str()<< " creating MultiFit " << endl;
    tmpSeu = new SimpleMultiFitStream( resources, item);
    break;
  case SINGLE_FLOW_FIT:
    if(_verbose) cout << headerinfo.str()<< " creating SingleFit " << endl;
    tmpSeu = new SimpleSingleFitStream( resources, item);
    break;
  default:
    if(_verbose) cout << headerinfo.str()<< " received unknown item" << endl;

  }
  //set the compute version according to the device the streamManager is initiated for
  //tmpSeu->setCompute(_computeVersion);

  return tmpSeu;
}


void cudaSimpleStreamExecutionUnit::setCompute(int compute)
{
  _computeVersion = compute;
}
int cudaSimpleStreamExecutionUnit::getCompute()
{
  return _computeVersion;
}
/*
// have to find better way to propagate since _myJob no longer part of base class
// to keep the stream execution unit generic
int cudaSimpleStreamExecutionUnit::getNumFrames()
{
  int n=0;
  //if(_myJob.ValidJob()) n = _myJob.getNumFrames();
  return n;
}

int cudaSimpleStreamExecutionUnit::getNumBeads()
{
  int n=0;
  //if(_myJob.ValidJob()) n = _myJob.getNumBeads();
  return n;
}
 */


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// SIMPLE STREAM MANAGER



cudaSimpleStreamManager::cudaSimpleStreamManager( 
    WorkerInfoQueue * inQ, 
    WorkerInfoQueue * fallbackQ
  )
{

  cudaGetDevice( &_devId );

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, _devId);

  _computeVersion = 10*deviceProp.major + deviceProp.minor;
  cout << getLogHeader() << " init with up to "<< getNumMaxStreams() <<" Stream Execution Units" << endl;
  _inQ = inQ;
  _fallBackQ = fallbackQ;
  _resourcePool = NULL;
  _item.finished = false;
  _item.private_data = NULL;
  _GPUerror = false;
  //_maxNumStreams = numStreams; // now static

  _tasks = 0;
  _sumBeads=0;
  _maxBeads=0;
  _sumFrames=0;
  _maxFrames=0;
  _executionErrorCount=0;

  _resourceAllocDone = false; //not yet allocated, flag added to allow for allocation when first job arrives
  allocateResources();

}



cudaSimpleStreamManager::~cudaSimpleStreamManager()
{


  ostringstream outputStr;

  outputStr << getLogHeader() << " handled " << _tasks <<" tasks."<< endl;
  //  outputStr << getLogHeader() << " Beads max: "<< _maxBeads <<" avg: " << _sumBeads/_tasks << " Frames max: " <<  _maxFrames << " avg: " << _sumFrames/_tasks << endl;
  for ( map< string, TimeKeeper>::iterator iter = _timer.begin(); iter != _timer.end(); ++iter )
    outputStr << getLogHeader() << " " << iter->first << " finished: " << iter->second.getJobCnt() << " in " << iter->second.getTime() << " time/job: " << iter->second.getAvgTime() << " (exceptions: " <<  iter->second.getErrorCnt()<< ")" << endl;

  freeResources();

  cout << outputStr.str();

}



void cudaSimpleStreamManager::allocateResources()
{

  //size_t maxHostSize = getMaxHostSize();
  //allocate a lot of frames to handle exponential tail fit
  //size_t maxDeviceSize = getMaxDeviceSize(MAX_PREALLOC_COMPRESSED_FRAMES_GPU);

  if(_resourcePool !=NULL) delete _resourcePool;
  _resourcePool = NULL;

  try{
    _resourcePool = new cudaResourcePool(_maxNumStreams); // throws cudaException
    int n = getNumStreams();
    if (n > 0){ cout <<  getLogHeader() <<" successfully acquired resources for " << n << " Stream Execution Units" <<endl;
    }
    _GPUerror = false;

  }
  catch(exception &e){
    cout << e.what() << endl;
    cout << getLogHeader() << " No StreamResources could be acquired! retry pending. jobs will he handled by CPU for now!" << endl;
    _GPUerror = true;
    _resourcePool = NULL;
  }

   _resourceAllocDone = true;
}



void cudaSimpleStreamManager::freeResources()
{
  if(_resourcePool != NULL) delete _resourcePool;
  _resourcePool = NULL;

  if(isFinishItem() && _inQ !=NULL){
    cout << getLogHeader() << " signaling Queue that all jobs and cleanup completed" << endl;
    try{
      _inQ->DecrementDone();
    }
    catch(...){
      cout << getLogHeader() << " signal to Queue caused exception, Queue seems to be destroyed already!" << endl;
    }
  }
}

int cudaSimpleStreamManager::getNumStreams()
{
  if(_resourcePool != NULL) return _resourcePool->getNumStreams();
  return 0;
}

int cudaSimpleStreamManager::availableResources()
{
  // calculate free SEUs #allocated stream resources - #active SEUS
  return getNumStreams() - _activeSEU.size();
}


/*
// No Longer needed
size_t cudaSimpleStreamManager::getMaxHostSize(int flow_block_size)
{

  size_t ret = 0;
  ret = std::max( SimpleSingleFitStream::getMaxHostMem( flow_block_size ), ret );
  ret = std::max( SimpleMultiFitStream ::getMaxHostMem( flow_block_size ), ret );
  return ret;

}

size_t cudaSimpleStreamManager::getMaxDeviceSize(int maxFrames, int maxBeads, int flow_block_size)
{
  size_t ret = 0;
  ret = std::max( SimpleSingleFitStream::getMaxDeviceMem(flow_block_size, maxFrames, maxBeads), ret );
  ret = std::max( SimpleMultiFitStream ::getMaxDeviceMem(flow_block_size, maxFrames, maxBeads), ret );
  return ret;
}
*/

void cudaSimpleStreamManager::moveToCPU()
{
  //get jobs and hand them over to the CPU Q after GPU error was encountered
  getJob();
  if(checkItem()){
    if(_verbose) cout << getLogHeader()<< " job received, try to reallocate resources!" << endl;
    //try to allocate and recover before handing job to CPU
    if(_executionErrorCount < MAX_EXECUTION_ERRORS){
      if(getNumStreams() == 0 ){
        allocateResources();
      }
      if( getNumStreams()  > 0){
        cout << getLogHeader()<< " managed to acquire streamResources, switching execution back to GPU!" << endl;
        addSEU();
        _GPUerror = false;
        return;
      }
    }
    if(_verbose) cout << getLogHeader() << " handing job on to CPU queue" << endl;

    _fallBackQ->PutItem(_item); // if no matching SEU put to CpuQ
    _inQ->DecrementDone(); // signale to Q that a job is completed
  }
}


void cudaSimpleStreamManager::getJob()
{
  if(!isFinishItem()){
    if(_activeSEU.empty()){
      if(_verbose) cout << getLogHeader()<< " blocking Job request" << endl;
      _item = _inQ->GetItem();  //if no active SEUs block on Q
      if(!_resourceAllocDone) allocateResources(); // do allocation when first job received
    }
    else{
      _item = _inQ->TryGetItem();
    }
    if(isFinishItem()){
      cout << getLogHeader()<< " received finish job" << endl;
    }
  }
}


// Depending on the type of job that was received from the inQ
// creates the according SEU type or hands it back to the CPU
// new GPU Jobs have to be added to this switch/case statement
void cudaSimpleStreamManager::addSEU()
{
  //create a SEU if item is not the finish item
  if(checkItem()){
    if(_verbose) cout << getLogHeader()<< " job received, checking type and create SEU" << endl;
    cudaSimpleStreamExecutionUnit * tmpSeu = NULL;
    try{

      tmpSeu = cudaSimpleStreamExecutionUnit::makeExecutionUnit(_resourcePool->getResource(), _item);
      if (tmpSeu == NULL){
        if(_verbose) cout << getLogHeader()<< " received unknown item" << endl;
        _fallBackQ->PutItem(_item); // if no matching SEU put to Cpu
        _inQ->DecrementDone(); // Signal to Q that a job is completed

      }else{
        _timer[tmpSeu->getName()].start();
        _activeSEU.push_back(tmpSeu);
        //set the compute version according to the device the streamManager is initiated for
        tmpSeu->setCompute(_computeVersion);
      }

    }

    catch(cudaException &e){
      if(getNumStreams() == 0){
        cout << " *** ERROR DURING STREAM UNIT CREATION, handing job back to CPU" << endl;
        _fallBackQ->PutItem(_item);
        _inQ->DecrementDone(); // Signal to Q that a job is completed
        _GPUerror = true;
      }else{
        cout << " *** ERROR DURING STREAM UNIT CREATION, retry on GPU" << endl;
        _inQ->PutItem(_item); //
        _inQ->DecrementDone(); // Signal to Q that a job is completed
      }
    }
  }
}


//handles the execution of all SEU in the StreamManager
//cleans up the SEUs that completed their jobs
void cudaSimpleStreamManager::executeSEU()
{
  bool workDone = false;
  bool errorOccured = false;

  for(int i=(_activeSEU.size()-1); i >= 0 ; i--){ // iterate backwards for easy delete
    if(_activeSEU[i] != NULL){ // Safety, should not be possible to be NULL
      try{
        workDone = !(_activeSEU[i]->execute());
      }
      catch(cudaAllocationError &e){
        //if execution exception, get item and hand it back to CPU
        //cout << e.what() << endl;
        if(_verbose) e.Print();

        _item = _activeSEU[i]->getItem();

        if(getNumStreams() == 0){
          cout << getLogHeader() << "*** CUDA RESOURCE POOL EMPTY , handing incomplete Job back to CPU for retry" << endl;
          _fallBackQ->PutItem(_item); // if no matching SEU put to CpuQ
          _GPUerror = true;
        }else{
          cout << getLogHeader() << "*** CUDA STREAM RESOURCE COULD NOT BE ALLOCATED, " << getNumStreams() << " StreamResources still avaiable, retry pending" << endl;
          _inQ->PutItem(_item); // if no matching SEU put to CpuQ
        }
        workDone = true ; // Make work as done so SEU gets cleaned up
        errorOccured = true;


      } // end catch block

      catch(cudaException &e){
        //if execution exception, get item and hand it back to CPU

        e.Print();
        _item = _activeSEU[i]->getItem();

        cout << getLogHeader() << "*** ERROR DURING STREAM EXECUTION, handing incomplete Job back to CPU for retry" << endl;
        _fallBackQ->PutItem(_item); // if no matching SEU put to CpuQ
        workDone = true; // mark work as done so SEU gets cleaned up
        errorOccured = true;
        _executionErrorCount++;
        if(e.getCudaError() == cudaErrorLaunchFailure)
        {
          cout << getLogHeader() << "encountered Kernel Launch Failure. Stop retrying, set GPU error state" << endl;
          cout << getNumStreams() << " StreamResources available" << endl;
          _activeSEU[i]->printStatus();
          _executionErrorCount = MAX_EXECUTION_ERRORS + 1;
          _GPUerror = true;
        }else{
          if(_executionErrorCount == NUM_ERRORS_TOGGLE_VERBOSE){
            cout << getLogHeader() << "encountered " << NUM_ERRORS_TOGGLE_VERBOSE << " errors, turning on verbose mode for debugging" << endl;
            setVerbose(true);
            cudaSimpleStreamExecutionUnit::setVerbose(true);
          }
          if(_executionErrorCount >= MAX_EXECUTION_ERRORS){
            cout << getLogHeader() << "encountered " << MAX_EXECUTION_ERRORS << " errors. Stop retrying, set GPU error state" << endl;
            setVerbose(false);
            cudaSimpleStreamExecutionUnit::setVerbose(false);
            _GPUerror = true;
          }
        }
      } // end catch block

      // clean up if work for this SEU is done
      if(workDone){
        if(!errorOccured){
          _timer[_activeSEU[i]->getName()].stop();
          //   recordBeads(_activeSEU[i]->getNumBeads());
          //   recordFrames(_activeSEU[i]->getNumFrames());
          _tasks++;
        }else{
          _timer[_activeSEU[i]->getName()].stopAfterError();
        }
        delete _activeSEU[i]; // destroy SEU object
        _activeSEU.erase(_activeSEU.begin()+i); //delete SEU from active list
        _inQ->DecrementDone(); // Signal to Q that a job is completed
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
      if(availableResources() > 0 ) {          // if resources available get job
        getJob(); // get a Job from Q, block on Q if no job not already working
        addSEU(); // try to add whatever getJob acquired
      }
      else if ( getNumStreams() == 0 ) {      // Something's wrong, and the streams went away.
        cout << getLogHeader() << " all the streams went away. Falling back to CPU." << endl;
        _GPUerror = true;
      }
    }
    // drive the state machine of the state execution units
    // and clean up when a SEU is don
    executeSEU();
    // as long as no finish job received and there are still active SEUs
    // in the list we are not done yet
    notDone =  (_activeSEU.empty() && isFinishItem())?(false):(true);
  }
  return false;
}

//bookkeeping
/*
void cudaSimpleStreamManager::recordBeads(int n)
{
  _sumBeads+=n;
  _maxBeads = (_maxBeads>n)?(_maxBeads):(n);
}
void cudaSimpleStreamManager::recordFrames(int n)
{
  _sumFrames+=n;
  _maxFrames = (_maxFrames>n)?(_maxFrames):(n);
}
 */

void cudaSimpleStreamManager::setNumMaxStreams(int numMaxStreams)
{
  if(numMaxStreams <= MAX_ALLOWED_NUM_STREAMS ){
    _maxNumStreams = numMaxStreams;
  }else{
    cout << "CUDA tried to set number of streams to " << numMaxStreams << ", correcting to allowed maximum of " << MAX_ALLOWED_NUM_STREAMS <<  " streams " << endl;
    _maxNumStreams = MAX_ALLOWED_NUM_STREAMS;
  }
}

int cudaSimpleStreamManager::getNumMaxStreams()
{
  return _maxNumStreams;
}



void cudaSimpleStreamManager::setVerbose(bool v)
{
  _verbose = v;
}

string cudaSimpleStreamManager::getLogHeader()
{
  ostringstream headerinfo;

  headerinfo << "CUDA " << _devId << " StreamManager:";

  return headerinfo.str();
}


bool cudaSimpleStreamManager::checkItem()
{
  if(!_item.finished && _item.private_data != NULL ) return true;
  return false;
}

bool cudaSimpleStreamManager::isFinishItem()
{
  return _item.finished;
}

/////////////////////////////////

TimeKeeper::TimeKeeper()
{
  _timesum = 0;
  _activeCnt = 0;
  _jobCnt = 0;
  _errCnt = 0;
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

void TimeKeeper::stopAfterError(){
  stop();
  _jobCnt--;
  _errCnt++;
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

int TimeKeeper::getErrorCnt()
{
  return _errCnt;
}


