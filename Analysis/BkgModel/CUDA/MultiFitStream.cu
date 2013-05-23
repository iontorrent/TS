/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

// patch for CUDA5.0/GCC4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <iostream>

#include "cuda_error.h"
#include "cuda_runtime.h"

#include "StreamingKernels.h" 
#include "StreamManager.h"
#include "MultiFitStream.h"
#include "JobWrapper.h"


using namespace std;



int SimpleMultiFitStream::_bpb = 128;
int SimpleMultiFitStream::_l1type = -1;  // 0: SM=L1, 1: SM>L1,  2: L1>SM, -1:GPU default


int SimpleMultiFitStream::_bpbPartialD= 128;
int SimpleMultiFitStream::_l1typePartialD =-1;

int SimpleMultiFitStream::l1DefaultSettingMultiFit()
{
  // 0: SM=L1, 1: SM>L1,  2: L1>SM, -1:GPU default
  if(_computeVersion == 20 ) return 2;
  if(_computeVersion == 35 ) return 1;
  return 0;
}

int SimpleMultiFitStream::l1DefaultSettingPartialD()
{
  if(_computeVersion == 20 ) return 2;
  if(_computeVersion == 35 ) return 0;
  return 0;
}


/////////////////////////////////////////////////
//MULTI FIT STREAM CLASS

SimpleMultiFitStream::SimpleMultiFitStream(streamResources * res, WorkerInfoQueueItem item ) : cudaSimpleStreamExecutionUnit(res, item)
{
  setName("MultiFitStream");

  if(_verbose) cout << getLogHeader() << " created"  << endl;

  _lambda_start[0] = SMALL_LAMBDA;
  _lambda_start[1] = LARGER_LAMBDA;

  _fit_iterations[0] = 1;
  _fit_iterations[1] = 3;

  _clonal_restriction[0] = NO_NONCLONAL_PENALTY;
  _clonal_restriction[1] = FULL_NONCLONAL_PENALTY;




  // calculate clonality restriction
  for (int i=0; i<CUDA_MULTIFLOW_NUM_FIT; ++i)
  {
    CalculateClonalityRestriction(i);
  }

	//worst case scenario:

  _fitIter = 0;

  _HostConstP = NULL; // I 1

  _pHostBeadParams = NULL; // I 2

  _pHostFgBuffer = NULL; 
  _pHostNucRise = NULL; 
  _pHostSbg = NULL; 
  _pHostEmphasis = NULL; 
  _pHostNon_integer_penalty = NULL; 
  _pHostDarkMatterComp = NULL; 

  _pDevObservedTrace = NULL;
  _pDevObservedTraceTranspose = NULL;
  _pDevNucRise = NULL; 
  _pDevSbg = NULL; 
  _pDevEmphasis = NULL; 
  _pDevNon_integer_penalty = NULL; 
  _pDevDarkMatterComp = NULL; 
  _pDevBeadParams = NULL; 
  _pDevBeadParamsTranspose = NULL; 
  _pDevIval = NULL; 
  _pDevScratch_ival = NULL; 
  _pDevResidual = NULL;


  _DevFitData.Steps = NULL; 
  _DevFitData.LambdaForBeadFit = NULL;
  _DevFitData.JTJMatrixMapForDotProductComputation = NULL;
  _DevFitData.BeadParamIdxMap = NULL;

  _pDevJTJ = NULL;
  _pDevRHS = NULL;
  _pDevLTR = NULL;

  for (int i=0; i<CUDA_MULTIFLOW_NUM_FIT; ++i)
  {
    _HostFitData[i].Steps = NULL;
    _HostFitData[i].LambdaForBeadFit = NULL;
    _HostFitData[i].JTJMatrixMapForDotProductComputation = NULL;
    _HostFitData[i].BeadParamIdxMap = NULL;
  }


}



SimpleMultiFitStream::~SimpleMultiFitStream()
{
  cleanUp();
}

void SimpleMultiFitStream::cleanUp()
{

   if(_verbose) cout << getLogHeader() << " clean up"  << endl;
   CUDA_ERROR_CHECK();
}



void SimpleMultiFitStream::resetPointers()
{

  if(_verbose) cout << getLogHeader() << " resetting pointers for job with " << _myJob.getNumBeads() << "("<< _myJob.getPaddedN() <<") beads and " << _myJob.getNumFrames() << " frames" << endl;

// fit invariant inputs

   if(!_Device->checkMemory( getMaxDeviceMem(_myJob.getNumFrames(),_myJob.getNumBeads())))
      cout << getLogHeader() << " succesfully reallocated device memory to handle Job" << endl;




  _HostConstP  = (ConstParams*)_Host->getSegment(sizeof(ConstParams)); CUDA_ALLOC_CHECK(_HostConstP);  

  _pHostBeadParams =  (bead_params*) _Host->getSegment( _myJob.getBeadParamsSize(true)); 
  _pDevBeadParams =  (bead_params*) _Device->getSegment( _myJob.getBeadParamsSize(true));


  _Host->startNewSegGroup();

  _pHostFgBuffer = (FG_BUFFER_TYPE*)_Host->getSegment(_myJob.getFgBufferSizeShort(true)); CUDA_ALLOC_CHECK(_pHostFgBuffer); 
  _pDevObservedTrace = (FG_BUFFER_TYPE*)_Device->getSegment(_myJob.getFgBufferSizeShort(true)); CUDA_ALLOC_CHECK(_pDevObservedTrace);

  _pHostNucRise  = (float*)_Host->getSegment(_myJob.getNucRiseSize(true));  // ISIG_SUB_STEPS_SINGLE_FLOW * F * NUMFB 
  _pDevNucRise = (float*)_Device->getSegment(_myJob.getNucRiseSize(true)); // ISIG_SUB_STEPS_SINGLE_FLOW * F * NUMFB 

  _pHostSbg  = (float*)_Host->getSegment(_myJob.getShiftedBackgroundSize(true)); // NUMFB*F 
  _pDevSbg = (float*)_Device->getSegment(_myJob.getShiftedBackgroundSize(true));// NUMFB*F

  _pHostEmphasis  = (float*)_Host->getSegment(_myJob.getEmphVecSize(true)); // (MAX_POISSON_TABLE_COL)*F 
  _pDevEmphasis = (float*)_Device->getSegment(_myJob.getEmphVecSize(true)); // (MAX_POISSON_TABLE_COL)*F

  _pHostNon_integer_penalty  = (float*)_Host->getSegment(_myJob.getClonalCallScaleSize(true)); 
  _pDevNon_integer_penalty = (float*)_Device->getSegment(_myJob.getClonalCallScaleSize(true)); 

  _pHostDarkMatterComp  = (float*)_Host->getSegment(_myJob.getDarkMatterSize(true)); // NUMNUC*F 
  _pDevDarkMatterComp = (float*)_Device->getSegment(_myJob.getDarkMatterSize(true)); // NUMNUC*F


  _invariantCopyInSize = _Host->getCurrentSegGroupSize();

// fit variant inputs

   // fit specific host memory allocations
  for (int i=0; i<CUDA_MULTIFLOW_NUM_FIT; ++i) 
  {
    _HostFitData[i].Steps = (CpuStep_t*)_Host->getSegment( _myJob.getPartialDerivStepsMaxSize(true)); 
    _HostFitData[i].JTJMatrixMapForDotProductComputation = (unsigned int*)_Host->getSegment(_myJob.getJTJMatrixMapMaxSize(true));
    _HostFitData[i].BeadParamIdxMap = (unsigned int*)_Host->getSegment(_myJob.getBeadParamIdxMapMaxSize(true));  
    _HostFitData[i].LambdaForBeadFit = (float*)_Host->getSegment(_myJob.getFloatPerBead(true));  
  }


    _Device->startNewSegGroup();

    _DevFitData.Steps = (CpuStep_t*)_Device->getSegment( _myJob.getPartialDerivStepsMaxSize(true)); 
    _DevFitData.JTJMatrixMapForDotProductComputation = (unsigned int*)_Device->getSegment(_myJob.getJTJMatrixMapMaxSize(true));
    _DevFitData.BeadParamIdxMap = (unsigned int*)_Device->getSegment(_myJob.getBeadParamIdxMapMaxSize(true));  
    _DevFitData.LambdaForBeadFit = (float*)_Device->getSegment(_myJob.getFloatPerBead(true));  

    _fitSpecificCopyInSize = _Device->getCurrentSegGroupSize();

// Device work/scratch buffer:

  _pDevBeadParamsEval = (float*)_Device->getSegment( _myJob.getBeadParamsSize(true)); CUDA_ALLOC_CHECK(_pDevBeadParamsEval); 
  _pDevBeadParamsTranspose = (float*)_Device->getSegment( _myJob.getBeadParamsSize(true)); CUDA_ALLOC_CHECK(_pDevBeadParamsTranspose); 
  _pDevObservedTraceTranspose = (float*)_Device->getSegment( _myJob.getFgBufferSize(true)); CUDA_ALLOC_CHECK(_pDevObservedTraceTranspose); 
 // we need a specific struct describing this config for this well fit for GPU

  _pDevIval = (float*)_Device->getSegment( _myJob.getFxB(true)); CUDA_ALLOC_CHECK(_pDevIval); // FLxNxF
  _pDevScratch_ival = (float*)_Device->getSegment(_myJob.getFxB(true)); CUDA_ALLOC_CHECK(_pDevScratch_ival); // FLxNxF
  _pDevResidual = (float*)_Device->getSegment( _myJob.getFloatPerBead(true)); CUDA_ALLOC_CHECK(_pDevResidual); // FLxNxF


    // lev mar fit matrices
  _pDevJTJ = (float*)_Device->getSegment( _myJob.getParamMatrixMaxSize(true) ); CUDA_ALLOC_CHECK(_pDevJTJ);
  _pDevLTR = (float*)_Device->getSegment( _myJob.getParamMatrixMaxSize(true) ); CUDA_ALLOC_CHECK(_pDevLTR);
  _pDevRHS = (float*)_Device->getSegment( _myJob.getParamRHSMaxSize(true)); CUDA_ALLOC_CHECK(_pDevRHS);
    
  _pd_partialDerivsOutput = (float*)_pDevObservedTrace;
  _pd_delta = _pd_partialDerivsOutput + _myJob.getMaxSteps()*_myJob.getPaddedN()*_myJob.getNumFrames();

}





void SimpleMultiFitStream::serializeFitInvariantInputs()
{  //inputs

  if(_verbose) cout << getLogHeader() <<" serialize data for fit invariant asnync global mem copy" << endl;


  memcpy(_pHostFgBuffer, _myJob.getFgBuffer(), _myJob.getFgBufferSizeShort());
  memcpy(_pHostBeadParams, _myJob.getBeadParams() , _myJob.getBeadParamsSize());  
  memcpy(_pHostDarkMatterComp, _myJob.getDarkMatter(), _myJob.getDarkMatterSize()); 
  memcpy(_pHostSbg,_myJob.getShiftedBackground(), _myJob.getShiftedBackgroundSize()); 
  memcpy(_pHostEmphasis,_myJob.getEmphVec() , _myJob.getEmphVecSize());
  memcpy(_pHostNucRise, _myJob.getCalculateNucRiseCoarse() , _myJob.getNucRiseCoarseSize());  


  //const memory
  *((reg_params* )_HostConstP) = *(_myJob.getRegionParams()); // 4
  memcpy( _HostConstP->start, _myJob.getStartNucCoarse()  , _myJob.getStartNucCoarseSize() );
  memcpy( _HostConstP->deltaFrames, _myJob.getDeltaFrames() , _myJob.getDeltaFramesSize() );
  memcpy( _HostConstP->flowIdxMap, _myJob.getFlowIdxMap() , _myJob.getFlowIdxMapSize());  
  memcpy(&_HostConstP->beadParamsMaxConstraints, _myJob.getBeadParamsMax(), _myJob.getBeadParamsMaxSize());
  memcpy(&_HostConstP->beadParamsMinConstraints, _myJob.getBeadParamsMin(), _myJob.getBeadParamsMinSize());
  _HostConstP->useDarkMatterPCA = _myJob.useDarkMatterPCA();

}

void SimpleMultiFitStream::serializeFitSpecificInputs(int fit_index)
{
  //inputs
   if(_verbose) cout << getLogHeader() <<" serialize data for fit specific asnync global mem copy" << endl;
 
  memcpy(_HostFitData[fit_index].Steps, _myJob.getPartialDerivSteps(fit_index) , _myJob.getPartialDerivStepsSize(fit_index) ); 
  memcpy(_HostFitData[fit_index].JTJMatrixMapForDotProductComputation, _myJob.getJTJMatrixMap(fit_index), _myJob.getJTJMatrixMapSize(fit_index));  
  memcpy(_HostFitData[fit_index].BeadParamIdxMap, _myJob.getBeadParamIdxMap(fit_index), _myJob.getBeadParamIdxMapSize(fit_index) );

}



//////////////////////////
// ASYNC CUDA FUNCTIONS, KERNEL EXECUTION AND DATA HANDLING


void SimpleMultiFitStream::prepareFitSpecificInputs(
    int fit_index)
{
  //prepare environment for new job
  SetUpLambdaArray(fit_index); 
  serializeFitSpecificInputs(fit_index);      
}

void SimpleMultiFitStream::copyFitInvariantInputsToDevice()
{
  //cout << "Copy data to GPU" << endl;
  if(_verbose) cout << getLogHeader() << " Invariant Async Copy To Device" << endl;

//  cudaMemcpyAsync(_pDevObservedTrace,  _pHostFgBuffer, _invariantCopyInSize, cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();

  cudaMemcpyAsync(_pDevNon_integer_penalty, _pHostNon_integer_penalty,_myJob.getClonalCallScaleSize(), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK(); 
  cudaMemcpyAsync((FG_BUFFER_TYPE*)_pDevObservedTrace, _pHostFgBuffer, _myJob.getFgBufferSizeShort(), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();
  cudaMemcpyAsync(_pDevDarkMatterComp, _pHostDarkMatterComp, _myJob.getDarkMatterSize(), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK(); 
  cudaMemcpyAsync(_pDevSbg, _pHostSbg, _myJob.getShiftedBackgroundSize(), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK(); 
  cudaMemcpyAsync(_pDevEmphasis, _pHostEmphasis, _myJob.getEmphVecSize(), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();
  cudaMemcpyAsync(_pDevNucRise, _pHostNucRise, _myJob.getNucRiseCoarseSize(), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();  


//  copyMultiFlowFitConstParamAsync(_HostConstP, getStreamId(),_stream);CUDA_ERROR_CHECK();
  copyFittingConstParamAsync(_HostConstP, getStreamId(),_stream);CUDA_ERROR_CHECK();

}


void SimpleMultiFitStream::copyFitSpecifcInputsToDevice(int fit_index)
{
  //cout << "Copy data to GPU" << endl;
  if(_verbose) cout << getLogHeader() << " Fit Specific Async Copy To Device" << endl;

 // cudaMemcpyAsync(_DevFitData.Steps, _HostFitData[fit_index].Steps, _fitSpecificCopyInSize  , cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();
   cudaMemcpyAsync(_DevFitData.Steps, _HostFitData[fit_index].Steps, _myJob.getPartialDerivStepsSize(fit_index)  , cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK(); 

  cudaMemcpyAsync(_DevFitData.JTJMatrixMapForDotProductComputation, _HostFitData[fit_index].JTJMatrixMapForDotProductComputation , _myJob.getJTJMatrixMapSize(fit_index) , cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();  
 cudaMemcpyAsync(_DevFitData.BeadParamIdxMap, _HostFitData[fit_index].BeadParamIdxMap, _myJob.getBeadParamIdxMapSize(fit_index), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();  
 cudaMemcpyAsync(_DevFitData.LambdaForBeadFit, _HostFitData[fit_index].LambdaForBeadFit,_myJob.getFloatPerBead() , cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK(); 

}




void SimpleMultiFitStream::executeTransposeToFloat()
{
  //cout << "TransposeToFloat Kernel" << endl;

  int F = _myJob.getNumFrames();
  int padN = _myJob.getPaddedN();


  dim3 block(32,32);
  dim3 grid( (F*NUMFB+ block.x-1)/block.x , (padN+block.y-1)/block.y);
  
  transposeDataToFloat_Wrapper(grid, block, 0 ,_stream,_pDevObservedTraceTranspose, (FG_BUFFER_TYPE*)_pDevObservedTrace, F*NUMFB, padN);
  CUDA_ERROR_CHECK();
}

void SimpleMultiFitStream::executeTransposeParams()
{

  int padN = _myJob.getPaddedN();

  //cout << "TransposeParams Kernel" << endl;

  dim3 block(32,32);
  int StructLength = (sizeof(bead_params)/sizeof(float));

  if((sizeof(bead_params)%sizeof(float)) != 0 )
  { 
    cerr << getLogHeader() <<" Structure not a multiple of sizeof(float), transpose not possible" << endl;
    exit(-1);
  }

  dim3 grid((StructLength + block.x-1)/block.x , (padN+block.y-1)/block.y);

   CUDA_ERROR_CHECK();
   transposeData_Wrapper(grid, block, 0 ,_stream,_pDevBeadParamsTranspose, (float*)_pDevBeadParams, StructLength, padN);
//  cudaThreadSynchronize();CUDA_ERROR_CHECK();
}


void SimpleMultiFitStream::executeMultiFit(int fit_index)
{
  if(_verbose) cout << getLogHeader() << " Exec Async Kernels" << endl;

  //cout << "MultiFit Kernels" << endl;
  int F = _myJob.getNumFrames();
  int N = _myJob.getNumBeads();

  dim3 blockPD( getBeadsPerBlockPartialD(), 1);
  dim3 gridPD( (N+blockPD.x-1)/blockPD.x, 1 );

//  int StructLength = (sizeof(bead_params)/sizeof(float));

  CUDA_ERROR_CHECK();


  cudaMemcpyAsync(_pDevBeadParamsEval, _pDevBeadParamsTranspose, _myJob.getBeadParamsSize(true), cudaMemcpyDeviceToDevice, _stream ); CUDA_ERROR_CHECK(); 

  int sharedMem = _myJob.getEmphVecSize();
  for (int i=0; i<_fit_iterations[fit_index]; ++i) {

    cudaMemsetAsync(_pDevJTJ, 0, _myJob.getParamMatrixMaxSize(true), _stream); CUDA_ERROR_CHECK();
    cudaMemsetAsync(_pDevRHS, 0, _myJob.getParamRHSMaxSize(true), _stream); CUDA_ERROR_CHECK();

    ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow_Wrapper(
      getL1SettingPartialD(),
      gridPD,
      blockPD,
      sharedMem, 
      _stream,
      // inputs
      _myJob.getMaxEmphasis(),
      // weights for frames
      _restrict_clonal[fit_index],
      _pDevObservedTraceTranspose, 
      _pDevIval,
      _pDevScratch_ival,
      _pDevNucRise,
      _pDevSbg,
      _pDevEmphasis,
      _pDevNon_integer_penalty,
      _pDevDarkMatterComp,
      _pDevBeadParamsTranspose,
      _DevFitData.Steps,
      _DevFitData.JTJMatrixMapForDotProductComputation, // pxp
      _pDevJTJ,
      _pDevRHS,
      _myJob.getNumParams(fit_index),
      _myJob.getNumSteps(fit_index),
      N,
      F,
      _pDevResidual,
      _pd_partialDerivsOutput,
      getStreamId()  // stream id for offset in const memory
    );


    dim3 block( getBeadsPerBlockMultiFit(), 1);
    dim3 grid( (N+block.x-1)/block.x, 1 );

    MultiFlowLevMarFit_Wrapper(getL1SettingMultiFit(), grid, block, sharedMem, _stream,
      _myJob.getMaxEmphasis(),
      _restrict_clonal[fit_index],
      _pDevObservedTraceTranspose,
      _pDevIval,
      _pDevScratch_ival, // FLxNxFx2  //scratch for both ival and fval
      _pDevNucRise, // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
      _pDevSbg, // FLxF
      _pDevEmphasis, // MAX_POISSON_TABLE_COL xF // needs precomputation
      _pDevNon_integer_penalty, // MAX_HPXLEN
      _pDevDarkMatterComp, // NUMNUC * F  
      _pDevBeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep_t
      _pDevBeadParamsEval,
      _DevFitData.LambdaForBeadFit,
      _pDevJTJ, // jtj matrix
      _pDevLTR, // lower triangular matrix
      _pDevRHS, // rhs vector
      _pd_delta,
      _DevFitData.BeadParamIdxMap, 
      _myJob.getNumParams(fit_index),
      N,
      F,
      _pDevResidual, // N 
      getStreamId());
    }
 }

void SimpleMultiFitStream::executeTransposeParamsBack()
{

  //cout << "TransposeParamsBack Kernel" << endl;
  int padN = _myJob.getPaddedN();

  dim3 block(32,32);
  int StructLength = (sizeof(bead_params)/sizeof(float));

  dim3 grid ((padN+block.y-1)/block.y, (StructLength + block.x-1)/block.x );

  transposeData_Wrapper(grid, block, 0 ,_stream, (float*)_pDevBeadParams, _pDevBeadParamsTranspose, padN,  StructLength);
  CUDA_ERROR_CHECK();
}


void SimpleMultiFitStream::copyBeadParamsToDevice()
{
 cudaMemcpyAsync((bead_params*)_pDevBeadParams , _pHostBeadParams , _myJob.getBeadParamsSize(), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();
}


void SimpleMultiFitStream::copyBeadParamsToHost()
{
 cudaMemcpyAsync( _pHostBeadParams  , (bead_params*)_pDevBeadParams, _myJob.getBeadParamsSize(), cudaMemcpyDeviceToHost, _stream); CUDA_ERROR_CHECK();
}


int SimpleMultiFitStream::handleResults()
{

  if(_myJob.isSet()){

    if(_verbose) cout <<  getLogHeader() << " Handling Results "<< _fitIter << endl;


    memcpy( _myJob.getBeadParams(),_pHostBeadParams, _myJob.getBeadParamsSize());  
    _myJob.KeyNormalize();   // temporary call to key normalize till we put it into a GPU kernel

    if(_fitIter == 0 && _myJob.performCalcPCADarkMatter())
    {
      //PCA on CPUi
      _myJob.PerformePCA();     
      // update PCA flag
      _HostConstP->useDarkMatterPCA = _myJob.useDarkMatterPCA();      
      copyFittingConstParamAsync(_HostConstP, getStreamId(),_stream);CUDA_ERROR_CHECK();
      //update DarkMatterComp
      memcpy(_pHostDarkMatterComp, _myJob.getDarkMatter(),_myJob.getDarkMatterSize());  
      cudaMemcpyAsync(_pDevDarkMatterComp, _pHostDarkMatterComp, _myJob.getDarkMatterSize(), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK(); 
    }


    _fitIter++;

    // if not last iteratin yet copy bead data back topagelocked mem so device can get updated
    if(_fitIter < CUDA_MULTIFLOW_NUM_FIT){
      memcpy(_pHostBeadParams, _myJob.getBeadParams(),_myJob.getBeadParamsSize());  
      return 1;  //signal more work to be done;
    }

    _myJob.setJobToRemainRegionFit();
    _myJob.putJobToCPU(_item);
  }

  return 0; //signal Job com[plete
}



void SimpleMultiFitStream::SetUpLambdaArray(int fit_index) {
  for (int i=0; i<_myJob.getNumBeads(); ++i) {
    _HostFitData[fit_index].LambdaForBeadFit[i] = _lambda_start[fit_index];
  }
}

void SimpleMultiFitStream::ExecuteJob()
{

  
//  printInfo(); cout << " i: " <<  _fitIter << " numBeads: " << _myJob.getNumBeads() << " numFrames:" << _myJob.getNumFrames() << endl;
  

  if(_fitIter == 0){
    //CalculateCoarseNucRise();
    resetPointers();

    CalculateNonIntegerPenalty();
    serializeFitInvariantInputs();
    copyFitInvariantInputsToDevice();
  
    executeTransposeToFloat();
  }

  copyBeadParamsToDevice();
  prepareFitSpecificInputs(_fitIter);      
  copyFitSpecifcInputsToDevice(_fitIter);

  executeTransposeParams();
  executeMultiFit(_fitIter);
  executeTransposeParamsBack();

  copyBeadParamsToHost();

}

void SimpleMultiFitStream::CalculateClonalityRestriction(int fit_index)
{
  _restrict_clonal[fit_index] = 0;
  
  float hpmax = 2.0f;
  if (_clonal_restriction[fit_index] > 0)
  {
    if (hpmax > _clonal_restriction[fit_index])
      hpmax = _clonal_restriction[fit_index];

    _restrict_clonal[fit_index] = hpmax-0.5f;
  }
}

void SimpleMultiFitStream::CalculateNonIntegerPenalty()
{
  float clonal_call_scale[MAGIC_CLONAL_CALL_ARRAY_SIZE]; 
  float clonal_call_penalty;

  memcpy(clonal_call_scale, _myJob.getClonalCallScale(), _myJob.getClonalCallScaleSize());
  clonal_call_penalty = _myJob.getClonalCallPenalty();

  for (int i=0; i<MAGIC_CLONAL_CALL_ARRAY_SIZE; ++i)
  {
    _pHostNon_integer_penalty[i] = clonal_call_penalty * clonal_call_scale[i];
  }
}

int SimpleMultiFitStream::getBeadsPerBlockMultiFit()
{
  return _bpb;
}

int SimpleMultiFitStream::getL1SettingMultiFit()
{
  if(_l1type < 0 || _l1type > 2){
    return l1DefaultSettingMultiFit();
  }
  return _l1type;
}

int SimpleMultiFitStream::getBeadsPerBlockPartialD()
{
  return _bpbPartialD;
}

int SimpleMultiFitStream::getL1SettingPartialD()
{
  if(_l1typePartialD < 0 || _l1typePartialD > 2){
    return l1DefaultSettingPartialD();
  }
  return _l1typePartialD;
}





// Static member function


size_t SimpleMultiFitStream::getMaxHostMem()
{
  WorkSet Job;

  size_t ret = 0;

  ret += sizeof(ConstParams); 
  ret += Job.getFgBufferSizeShort(true);

  ret += Job.getBeadParamsSize(true); 
  ret += Job.getFgBufferSizeShort(true);  
  ret += Job.getNucRiseSize(true);   
  ret += Job.getShiftedBackgroundSize(true); 
  ret += Job.getEmphVecSize(true); 
  ret += Job.getClonalCallScaleSize(true); 
  ret += Job.getDarkMatterSize(true); 
  for (int i=0; i<CUDA_MULTIFLOW_NUM_FIT; ++i) 
  {
    ret += Job.getPartialDerivStepsMaxSize(true); 
    ret += Job.getJTJMatrixMapMaxSize(true);
    ret += Job.getBeadParamIdxMapMaxSize(true);  
    ret += Job.getFloatPerBead(true);  
  }

  return ret;

}

size_t SimpleMultiFitStream::getMaxDeviceMem(int numFrames, int numBeads)
{

  WorkSet Job;

  // if numFrames/numBeads are passed overwrite the predevined maxFrames/maxBeads
  // for the size calculation
  Job.setMaxFrames(numFrames);
  Job.setMaxBeads(numBeads);

  size_t ret = 0;

  ret += Job.getBeadParamsSize(true);  //  _pDevBeadParams
  ret += Job.getBeadParamsSize(true);  // _pDevBeadParamsEval
  ret += Job.getBeadParamsSize(true); // _pDevBeadParamsTranspose

  ret += Job.getFgBufferSizeShort(true); //   _pDevObservedTrace
  ret += Job.getFgBufferSize(true);  // _pDevObservedTraceTranspose

  ret += Job.getNucRiseSize(true); //   _pDevNucRise
  ret += Job.getShiftedBackgroundSize(true); //   _pDevSbg
  ret += Job.getEmphVecSize(true); // _pDevEmphasis
  ret += Job.getClonalCallScaleSize(true);  // _pDevNon_integer_penalty
  ret += Job.getDarkMatterSize(true);  // _pDevDarkMatterComp

  ret += Job.getPartialDerivStepsMaxSize(true);  // Steps
  ret += Job.getJTJMatrixMapMaxSize(true);  // JTJMatrixMapForDotProductComputation
  ret += Job.getBeadParamIdxMapMaxSize(true); // BeadParamIdxMap
  ret += Job.getFloatPerBead(true); //LambdaForBeadFit

  ret += Job.getFxB(true); // _pDevIval
  ret += Job.getFxB(true); // _pDevScratch_ival
  ret += Job.getFloatPerBead(true); // _pDevResidual
  ret += Job.getParamMatrixMaxSize(true); // _pDevJTJ
  ret += Job.getParamMatrixMaxSize(true); // _pDevLTR
  ret += Job.getParamRHSMaxSize(true); // _pDevRHS 

  return ret;
}

void SimpleMultiFitStream::setBeadsPerBLockMultiF(int bpb)
{
  _bpb = bpb;
}


void SimpleMultiFitStream::setL1SettingMultiF(int type) // 0:sm=l1, 1:sm>l1, 2:sm<l1
{
 _l1type = type;
}

void SimpleMultiFitStream::setBeadsPerBLockPartialD(int bpb)
{
  _bpbPartialD = bpb;
}

void SimpleMultiFitStream::setL1SettingPartialD(int type) // 0:sm=l1, 1:sm>l1, 2:sm<l1
{
  _l1typePartialD = type; 
}

void SimpleMultiFitStream::printSettings()
{

  cout << "CUDA MultiFitStream SETTINGS: blocksize " << _bpb << " l1setting " << _l1type;   
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
  cout << "CUDA PartialDerivative SETTINGS: blocksize " << _bpbPartialD << " l1setting " << _l1typePartialD;   
  switch(_l1typePartialD){
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
     cout << " GPU specific default" << endl;
  }

}





