
#include <iostream>

#include "cudaStreamTemplate.h"
#include "cuda_error.h"

using namespace std;

//TODO: after implementing the template and adding the job type as a queue item the new SEU
//type needs to be added to the SEU factory method that produces the specialiced SEUs durign the execution





//basic config stuff:
int TemplateStream::_bpb = 128;
int TemplateStream::_l1type = -1;  // 0: SM=L1, 1: SM>L1,  2: L1>SM, -1:GPU default

//TODO: test for best setting

int TemplateStream::l1DefaultSetting()
{
  // 0: SM=L1, 1: SM>L1,  2: L1>SM, -1:GPU default
  if(_computeVersion == 20 ) return 1;
  if(_computeVersion == 35 ) return 1;
  return 0;
}

int TemplateStream::getBeadsPerBlock()
{
  return _bpb;
}

int TemplateStream::getL1Setting()
{
  if(_l1type < 0 || _l1type > 2){
    return l1DefaultSetting();
  }
  return _l1type;
}


/////////////////////////////////////////////////
//STREAM CLASS implementation

TemplateStream::TemplateStream(streamResources * res, WorkerInfoQueueItem item ) : cudaSimpleStreamExecutionUnit(res, item)
{
  setName("TemplateStream"); //set name: needed for log outputs and time keeping

  if(_verbose) cout << getLogHeader() << " created"  << endl;


}


TemplateStream::~TemplateStream()
{
  cleanUp();
}


void TemplateStream::cleanUp()
{

   if(_verbose) cout << getLogHeader() << " clean up"  << endl;
   CUDA_ERROR_CHECK();
}



//////////////////////////
// VIRTUAL MEMBER FUNCTIONS:
// INIT, ASYNC COPY FUNCTIONS, KERNEL EXECUTION AND DATA HANDLING

void GenerateBeadTraceStream::printStatus()
{

  cout << getLogHeader()  << " status: " << endl
  << " +------------------------------" << endl
  << " | block size: " << _threadBlockX << "x" << _threadBlockY  << endl
  //<< " | l1 setting: " << getL1Setting() << endl
  << " | state: " << _state << endl;
  if(_resources->isSet())
    cout << " | streamResource acquired successfully"<< endl;
  else
    cout << " | streamResource not acquired"<< endl;
   // _myJob.printJobSummary();
    cout << " +------------------------------" << endl;
}


bool GenerateBeadTraceStream::InitJob()
{
  return ValidJob();
}



void GenerateBeadTraceStream::ExecuteJob()
{

  //any cuda calls that are not async have to happen here to keep things clean
  prepareInputs();

  //the following 3 calls can only contain async cuda calls!!!
  copyToDevice();
  executeKernel();
  copyToHost();

}


int GenerateBeadTraceStream::handleResults()
{

  if(_verbose) cout <<  getLogHeader() << " Handling Results " << endl;

  return 0; //signal Job com[plete
}

////////////////////////////////////////
// Execution implementatin

//allocation and  serialization of buffers in host page locked host memory for async copy
void GenerateBeadTraceStream::prepareInputs()
{

}

// trigger async copies, no more sync cuda calls from this [point on until handle results
void GenerateBeadTraceStream::copyToDevice()
{
  //cout << "Copy data to GPU" << endl;
  if(_verbose) cout << getLogHeader() << " Async Copy To Device" << endl;


}


void GenerateBeadTraceStream::executeKernel()
{
  if(_verbose) cout << getLogHeader() << " Exec Async Kernels" << endl;

 }


void GenerateBeadTraceStream::copyToHost()
{

}

// end execution implementation
/////////////////////////////////////////////////////




//////////////////////////////////////////
// Static member function


size_t TemplateStream::getMaxHostMem()
{

  size_t ret = 0;
  //determine actual and worst case host buffer size
  return ret;

}

size_t TemplateStream::getMaxDeviceMem()
{

  size_t ret = 0;
  //determine actual or worst case host buffer size
  return ret;
}

void TemplateStream::setBeadsPerBLock(int bpb)
{
  _bpb = bpb;
}


void TemplateStream::setL1Setting(int type) // 0:sm=l1, 1:sm>l1, 2:sm<l1
{
 _l1type = type;
}


void TemplateStream::printSettings()
{

  cout << "CUDA TemplateStream SETTINGS: blocksize " << _bpb << " l1setting " << _l1type;
  switch(_l1type){
    case 0:
      cout << " (cudaFuncCachePreferEqual" << endl;;
      break;
    case 1:
      cout << " (cudaFuncCachePreferShared)" <<endl;
      break;
    case 2:
      cout << " (cudaFuncCachePreferL1)" << endl;
      break;
    default:
     cout << " GPU specific default" << endl;;
  }


}
