/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>

#include "cuda_error.h"
#include "cuda_runtime.h"

#include "StreamingKernels.h" 
#include "StreamManager.h"
#include "MultiFitStream.h"
#include "JobWrapper.h"
 
using namespace std;


int validValue(int x, int l)
{
  if(x <= 0) cout << "MultiFitStream.cu:" << l << " invaliud value:" << x << endl;
  return x;
};


int MultiFitStream::_bpb = 128;

/////////////////////////////////////////////////
//MULTI FIT STREAM CLASS

MultiFitStream::MultiFitStream(GpuMultiFlowFitControl& fitcontrol, WorkerInfoQueue* Q) : cudaStreamExecutionUnit(Q) 
{
   
  _fd[0] = fitcontrol.GetMatrixConfig("FitWellAmplBuffering");
  _fd[1] = fitcontrol.GetMatrixConfig("FitWellPostKey");

  _lambda_start[0] = SMALL_LAMBDA;
  _lambda_start[1] = LARGER_LAMBDA;

  _fit_iterations[0] = 1;
  _fit_iterations[1] = 3;

  _clonal_restriction[0] = NO_NONCLONAL_PENALTY;
  _clonal_restriction[1] = FULL_NONCLONAL_PENALTY;

  setName("MultiFitStream");



  // calculate clonality restriction
  for (int i=0; i<CUDA_MULTIFLOW_NUM_FIT; ++i)
  {
    CalculateClonalityRestriction(i);
  }

	//worst case scenario:
  int maxSteps = fitcontrol.GetMaxSteps();
  int maxParams = fitcontrol.GetMaxParamsToFit();

  _fitIter = 0;

  _HostConstP = NULL;
  _pHostBeadParams = NULL;
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
  _pDevSteps = NULL; 
  _pDevIval = NULL; 
  _pDevScratch_ival = NULL; 
  _pDevResidual = NULL;

  _pDevJTJMatrixMapForDotProductComputation = NULL;
  _pDevLambdaForBeadFit = NULL;
  _pDevBeadParamIdxMap = NULL;
  _pDevJTJ = NULL;
  _pDevRHS = NULL;
  _pDevLTR = NULL;

  for (int i=0; i<CUDA_MULTIFLOW_NUM_FIT; ++i)
  {
    _pHostSteps[i] = NULL;
    _pHostLambdaForBeadFit[i] = NULL;
    _pHostJTJMatrixMapForDotProductComputation[i] = NULL;
    _pHostBeadParamIdxMap[i] = NULL;
  }

  int padN = _myJob.getPaddedN();

  try{
  
    cudaHostAlloc(&_HostConstP, sizeof(ConstParams), cudaHostAllocDefault); CUDA_ALLOC_CHECK(_HostConstP);  
    cudaHostAlloc(&_pHostBeadParams, _myJob.getBeadParamsSize(true), cudaHostAllocDefault ); CUDA_ALLOC_CHECK(_pHostBeadParams);
    cudaHostAlloc(&_pHostFgBuffer, _myJob.getFgBufferSizeShort(true), cudaHostAllocDefault ); CUDA_ALLOC_CHECK(_pHostFgBuffer);
    cudaHostAlloc(&_pHostNucRise,_myJob.getNucRiseCoarseSize(), cudaHostAllocDefault ); CUDA_ALLOC_CHECK(_pHostNucRise); // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
    cudaHostAlloc(&_pHostSbg, _myJob.getShiftedBackgroundSize(), cudaHostAllocDefault); CUDA_ALLOC_CHECK(_pHostSbg); // FLxF
    cudaHostAlloc(&_pHostEmphasis,_myJob.getEmphVecSize() , cudaHostAllocDefault ); CUDA_ALLOC_CHECK(_pHostEmphasis); // MAX_HPLEN+1 xF // needs precomputation
    cudaHostAlloc(&_pHostNon_integer_penalty, _myJob.getClonalCallScaleSize() , cudaHostAllocDefault ); CUDA_ALLOC_CHECK(_pHostNon_integer_penalty); // MAX_HPLEN+1
    cudaHostAlloc(&_pHostDarkMatterComp,_myJob.getDarkMatterSize(), cudaHostAllocDefault ); CUDA_ALLOC_CHECK(_pHostDarkMatterComp); // NUMNUC * F  


    // fit specific host memory allocations
    for (int i=0; i<CUDA_MULTIFLOW_NUM_FIT; ++i) 
    { 

      cudaHostAlloc(&_pHostSteps[i], sizeof(CpuStep_t)*maxSteps, cudaHostAllocDefault ); CUDA_ALLOC_CHECK(_pHostSteps[i]); // we need a specific struct describing this config for this well fit for GPU
      cudaHostAlloc(&_pHostJTJMatrixMapForDotProductComputation[i], sizeof(unsigned int)*maxParams*maxParams, cudaHostAllocDefault ); CUDA_ALLOC_CHECK(_pHostJTJMatrixMapForDotProductComputation[i]);  
      cudaHostAlloc(&_pHostBeadParamIdxMap[i], sizeof(unsigned int)*maxParams, cudaHostAllocDefault ); CUDA_ALLOC_CHECK(_pHostBeadParamIdxMap[i]);  
      cudaHostAlloc(&_pHostLambdaForBeadFit[i], sizeof(float)*padN , cudaHostAllocDefault ); CUDA_ALLOC_CHECK(_pHostLambdaForBeadFit[i]);  

    }

    //DEVICE ALLOCS

    cudaMalloc(&_pDevBeadParams, _myJob.getBeadParamsSize(true)); CUDA_ALLOC_CHECK(_pDevBeadParams);
    cudaMalloc(&_pDevBeadParamsEval, _myJob.getBeadParamsSize(true)); CUDA_ALLOC_CHECK(_pDevBeadParamsEval); 

    cudaMalloc(&_pDevBeadParamsTranspose, _myJob.getBeadParamsSize(true)); CUDA_ALLOC_CHECK(_pDevBeadParamsTranspose); 

    cudaMalloc(&_pDevObservedTrace, _myJob.getFgBufferSizeShort(true) ); CUDA_ALLOC_CHECK(_pDevObservedTrace);
    cudaMalloc(&_pDevObservedTraceTranspose, _myJob.getFgBufferSize(true)); CUDA_ALLOC_CHECK(_pDevObservedTraceTranspose); 

    cudaMalloc(&_pDevNucRise, _myJob.getNucRiseCoarseSize()); CUDA_ALLOC_CHECK(_pDevNucRise); // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
    cudaMalloc(&_pDevSbg,_myJob.getShiftedBackgroundSize()); CUDA_ALLOC_CHECK(_pDevSbg); // FLxF
    cudaMalloc(&_pDevEmphasis,_myJob.getEmphVecSize()); CUDA_ALLOC_CHECK(_pDevEmphasis); // MAX_HPLEN+1 xF // needs precomputation
    cudaMalloc(&_pDevNon_integer_penalty, _myJob.getClonalCallScaleSize()); CUDA_ALLOC_CHECK(_pDevNon_integer_penalty); // MAX_HPLEN+1
    cudaMalloc(&_pDevDarkMatterComp, _myJob.getDarkMatterSize()); CUDA_ALLOC_CHECK(_pDevDarkMatterComp); // NUMNUC * F  
    cudaMalloc(&_pDevSteps, sizeof(CpuStep_t)*maxSteps); CUDA_ALLOC_CHECK(_pDevSteps); // we need a specific struct describing this config for this well fit for GPU

    cudaMalloc(&_pDevIval, _myJob.getFxB(true)); CUDA_ALLOC_CHECK(_pDevIval); // FLxNxF
    cudaMalloc(&_pDevScratch_ival,_myJob.getFxB(true)); CUDA_ALLOC_CHECK(_pDevScratch_ival); // FLxNxF
    cudaMalloc(&_pDevResidual, padN*sizeof(float)); CUDA_ALLOC_CHECK(_pDevResidual); // FLxNxF

    cudaMalloc(&_pDevJTJMatrixMapForDotProductComputation,sizeof(unsigned int)*maxParams*maxParams); CUDA_ALLOC_CHECK(_pDevJTJMatrixMapForDotProductComputation);  
    cudaMalloc(&_pDevLambdaForBeadFit,padN *sizeof(float)); CUDA_ALLOC_CHECK(_pDevLambdaForBeadFit);
    cudaMalloc(&_pDevBeadParamIdxMap, maxParams*sizeof(unsigned int)); CUDA_ALLOC_CHECK(_pDevBeadParamIdxMap);

    // lev mar fit matrices
    cudaMalloc(&_pDevJTJ, padN*((maxParams*(maxParams + 1))/2)*sizeof(float)); CUDA_ALLOC_CHECK(_pDevJTJ);
    cudaMalloc(&_pDevLTR, padN*((maxParams*(maxParams + 1))/2)*sizeof(float)); CUDA_ALLOC_CHECK(_pDevLTR);
    cudaMalloc(&_pDevRHS, padN*maxParams*sizeof(float)); CUDA_ALLOC_CHECK(_pDevRHS);
    
    _pd_partialDerivsOutput = _pDevObservedTrace;
    _pd_delta = _pd_partialDerivsOutput + maxSteps*padN*_myJob.getNumFrames();
   // _pd_beadParamEval = _pd_delta + maxParams*padN; 


  }
  catch( cudaException& e)
  {
    cleanUp();
    throw e; 
  }

}



MultiFitStream::~MultiFitStream()
{
  cleanUp();
}

void MultiFitStream::cleanUp()
{
    //cout << "MultiFit cleanup... " << endl;

   if(_HostConstP != NULL) cudaFreeHost(_HostConstP);_HostConstP = NULL;
   if(_pHostBeadParams != NULL) cudaFreeHost(_pHostBeadParams);_pHostBeadParams = NULL;
   if(_pHostFgBuffer != NULL) cudaFreeHost(_pHostFgBuffer);_pHostFgBuffer = NULL;
   if(_pHostNucRise != NULL) cudaFreeHost(_pHostNucRise);_pHostNucRise = NULL; 
   if(_pHostSbg != NULL) cudaFreeHost(_pHostSbg);_pHostSbg = NULL; 
   if(_pHostEmphasis != NULL) cudaFreeHost(_pHostEmphasis);_pHostEmphasis = NULL; 
   if(_pHostNon_integer_penalty != NULL) cudaFreeHost(_pHostNon_integer_penalty);_pHostNon_integer_penalty = NULL; 
   if(_pHostDarkMatterComp != NULL) cudaFreeHost(_pHostDarkMatterComp);_pHostDarkMatterComp = NULL; 

   for (int i=0; i<CUDA_MULTIFLOW_NUM_FIT; ++i)
   {
     if(_pHostSteps[i] != NULL) cudaFreeHost(_pHostSteps[i]);_pHostSteps[i] = NULL;
     if(_pHostJTJMatrixMapForDotProductComputation[i]!=NULL) cudaFreeHost(_pHostJTJMatrixMapForDotProductComputation[i]); _pHostJTJMatrixMapForDotProductComputation[i] = NULL;
     if(_pHostBeadParamIdxMap[i] != NULL) cudaFreeHost(_pHostBeadParamIdxMap[i]);_pHostBeadParamIdxMap[i] = NULL;
     if(_pHostLambdaForBeadFit[i] != NULL) cudaFreeHost(_pHostLambdaForBeadFit[i]);_pHostLambdaForBeadFit[i] = NULL;
   }

   if(_pDevObservedTrace != NULL) cudaFree(_pDevObservedTrace);_pDevObservedTrace = NULL;
   if(_pDevObservedTraceTranspose != NULL) cudaFree(_pDevObservedTraceTranspose);_pDevObservedTraceTranspose = NULL;
   if(_pDevNucRise != NULL) cudaFree(_pDevNucRise);_pDevNucRise = NULL; 
   if(_pDevSbg != NULL) cudaFree(_pDevSbg);_pDevSbg = NULL; 
   if(_pDevEmphasis != NULL) cudaFree(_pDevEmphasis);_pDevEmphasis = NULL; 
   if(_pDevNon_integer_penalty != NULL) cudaFree(_pDevNon_integer_penalty);_pDevNon_integer_penalty = NULL; 
   if(_pDevDarkMatterComp != NULL) cudaFree(_pDevDarkMatterComp);_pDevDarkMatterComp = NULL; 
   if(_pDevBeadParams != NULL) cudaFree(_pDevBeadParams);_pDevBeadParams = NULL; 
   if(_pDevBeadParamsEval != NULL) cudaFree(_pDevBeadParamsEval);_pDevBeadParamsEval = NULL; 

   if(_pDevBeadParamsTranspose != NULL) cudaFree(_pDevBeadParamsTranspose);_pDevBeadParamsTranspose = NULL; 
   if(_pDevSteps != NULL) cudaFree(_pDevSteps);_pDevSteps = NULL; 
   if(_pDevIval != NULL) cudaFree(_pDevIval);_pDevIval = NULL; 
   if(_pDevScratch_ival != NULL) cudaFree(_pDevScratch_ival);_pDevScratch_ival = NULL; 
   if(_pDevResidual != NULL) cudaFree(_pDevResidual);_pDevResidual = NULL; 

   if(_pDevJTJMatrixMapForDotProductComputation!=NULL) cudaFree(_pDevJTJMatrixMapForDotProductComputation); _pDevJTJMatrixMapForDotProductComputation = NULL;
   if(_pDevBeadParamIdxMap != NULL) cudaFree(_pDevBeadParamIdxMap);_pDevBeadParamIdxMap = NULL;
   if(_pDevLambdaForBeadFit != NULL) cudaFree(_pDevLambdaForBeadFit);_pDevLambdaForBeadFit = NULL;
   if(_pDevJTJ != NULL) cudaFree(_pDevJTJ);_pDevJTJ = NULL;
   if(_pDevRHS != NULL) cudaFree(_pDevRHS);_pDevRHS = NULL;
   if(_pDevLTR != NULL) cudaFree(_pDevLTR);_pDevLTR = NULL;

   CUDA_ERROR_CHECK();
}



void MultiFitStream::resetPointers()
{
  // update live beads and number of frames and padding
  //_N = _myJob.getNumBeads();
  //_F = _myJob.getNumFrames();

  //_padN = ((_N+32-1)/32) * 32;i
/*  int padN = _myJob.getPaddedN(); 
  int maxSteps = fitcontrol.GetMaxSteps();
  int maxParams = fitcontrol.GetMaxParamsToFit();
  _pd_partialDerivsOutput = _pDevObservedTrace;
  _pd_delta = _pd_partialDerivsOutput + maxSteps*padN*_myJob.getNumFrames();
  _pd_beadParamEval = _pd_delta + maxParams*padN; 
*/


}





void MultiFitStream::serializeFitInvariantInputs()
{  //inputs
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
}

void MultiFitStream::serializeFitSpecificInputs(int fit_index)
{
  //inputs
  memcpy(_pHostSteps[fit_index], _fd[fit_index]->GetPartialDerivSteps(), sizeof(CpuStep_t)*_fd[fit_index]->GetNumSteps()); 
  memcpy(_pHostJTJMatrixMapForDotProductComputation[fit_index], _fd[fit_index]->GetJTJMatrixMapForDotProductComputation() , sizeof(unsigned int) * _fd[fit_index]->GetNumParamsToFit()*_fd[fit_index]->GetNumParamsToFit());  
  memcpy(_pHostBeadParamIdxMap[fit_index], _fd[fit_index]->GetParamIdxMap(), sizeof(unsigned int) * _fd[fit_index]->GetNumParamsToFit());
}



//////////////////////////
// ASYNC CUDA FUNCTIONS, KERNEL EXECUTION AND DATA HANDLING


void MultiFitStream::prepareFitSpecificInputs(
    int fit_index)
{
  //prepare environment for new job
  SetUpLambdaArray(fit_index); 
  serializeFitSpecificInputs(fit_index);      
}

void MultiFitStream::copyFitInvariantInputsToDevice()
{
  //cout << "Copy data to GPU" << endl;

  cudaMemcpyAsync(_pDevNon_integer_penalty, _pHostNon_integer_penalty,validValue(_myJob.getClonalCallScaleSize(),__LINE__), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK(); 
  cudaMemcpyAsync((FG_BUFFER_TYPE*)_pDevObservedTrace, _pHostFgBuffer, validValue(_myJob.getFgBufferSizeShort(),__LINE__), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();
  cudaMemcpyAsync(_pDevDarkMatterComp, _pHostDarkMatterComp, validValue(_myJob.getDarkMatterSize(),__LINE__), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK(); 
  cudaMemcpyAsync(_pDevSbg, _pHostSbg, validValue(_myJob.getShiftedBackgroundSize(),__LINE__), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK(); 
  cudaMemcpyAsync(_pDevEmphasis, _pHostEmphasis, validValue(_myJob.getEmphVecSize(),__LINE__), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();
  cudaMemcpyAsync(_pDevNucRise, _pHostNucRise, validValue(_myJob.getNucRiseCoarseSize(),__LINE__), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();  

  copyMultiFlowFitConstParamAsync(_HostConstP, getId(),_stream);CUDA_ERROR_CHECK();

   

}

void MultiFitStream::copyFitSpecifcInputsToDevice(int fit_index)
{
  //cout << "Copy data to GPU" << endl;

  cudaMemcpyAsync(_pDevSteps, _pHostSteps[fit_index], validValue(sizeof(CpuStep_t)*_fd[fit_index]->GetNumSteps(),__LINE__) , cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK(); 
  cudaMemcpyAsync(_pDevJTJMatrixMapForDotProductComputation, _pHostJTJMatrixMapForDotProductComputation[fit_index] , validValue(sizeof(unsigned int) *_fd[fit_index]->GetNumParamsToFit()*_fd[fit_index]->GetNumParamsToFit(),__LINE__) , cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();  
 cudaMemcpyAsync(_pDevBeadParamIdxMap, _pHostBeadParamIdxMap[fit_index], validValue(sizeof(unsigned int)*_fd[fit_index]->GetNumParamsToFit(),__LINE__) , cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();  
 cudaMemcpyAsync(_pDevLambdaForBeadFit, _pHostLambdaForBeadFit[fit_index], validValue(sizeof(float)*_myJob.getNumBeads(),__LINE__) , cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK(); 


}




void MultiFitStream::executeTransposeToFloat()
{
  //cout << "TransposeToFloat Kernel" << endl;

  int F = _myJob.getNumFrames();
  int padN = _myJob.getPaddedN();


  dim3 block(32,32);
  dim3 grid( (F*NUMFB+ block.x-1)/block.x , (padN+block.y-1)/block.y);
  
  transposeDataToFloat_Wrapper(grid, block, 0 ,_stream,_pDevObservedTraceTranspose, (FG_BUFFER_TYPE*)_pDevObservedTrace, F*NUMFB, padN);
  CUDA_ERROR_CHECK();
}

void MultiFitStream::executeTransposeParams()
{

  int padN = _myJob.getPaddedN();

  //cout << "TransposeParams Kernel" << endl;

  dim3 block(32,32);
  int StructLength = (sizeof(bead_params)/sizeof(float));

  if((sizeof(bead_params)%sizeof(float)) != 0 )
  { 
    cerr << "Structure not a multiple of sizeof(float), transpose not possible" << endl;
    exit(-1);
  }

  dim3 grid((StructLength + block.x-1)/block.x , (padN+block.y-1)/block.y);

   CUDA_ERROR_CHECK();
   transposeData_Wrapper(grid, block, 0 ,_stream,_pDevBeadParamsTranspose, _pDevBeadParams, StructLength, padN);
//  cudaThreadSynchronize();CUDA_ERROR_CHECK();
}


void MultiFitStream::executeMultiFit(int fit_index)
{

  //cout << "MultiFit Kernels" << endl;
  int F = _myJob.getNumFrames();
  int N = _myJob.getNumBeads();
  int padN = _myJob.getPaddedN();

  dim3 block( _bpb, 1);
  dim3 grid( (N+block.x-1)/block.x, 1 );

//  int StructLength = (sizeof(bead_params)/sizeof(float));

  CUDA_ERROR_CHECK();


  cudaMemcpyAsync(_pDevBeadParamsEval, _pDevBeadParamsTranspose, validValue(_myJob.getBeadParamsSize(true),__LINE__) , cudaMemcpyDeviceToDevice, _stream ); CUDA_ERROR_CHECK(); 

  int sharedMem = sizeof(float)*(MAX_HPLEN + 1)*F;
  for (int i=0; i<_fit_iterations[fit_index]; ++i) {

    cudaMemsetAsync(_pDevJTJ, 0, padN*(( _fd[fit_index]->GetNumParamsToFit()*( _fd[fit_index]->GetNumParamsToFit() + 1))/2)*sizeof(float), _stream); CUDA_ERROR_CHECK();
    cudaMemsetAsync(_pDevRHS, 0, padN*_fd[fit_index]->GetNumParamsToFit()*sizeof(float), _stream); CUDA_ERROR_CHECK();

    ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow_Wrapper(
      grid,
      block,
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
      _pDevSteps,
      _pDevJTJMatrixMapForDotProductComputation, // pxp
      _pDevJTJ,
      _pDevRHS,
      _fd[fit_index]->GetNumParamsToFit(),
      _fd[fit_index]->GetNumSteps(),
      N,
      F,
      _pDevResidual,
      _pd_partialDerivsOutput,
      getId()  // stream id for offset in const memory
    );

    MultiFlowLevMarFit_Wrapper(grid, block, sharedMem, _stream,
      _myJob.getMaxEmphasis(),
      _restrict_clonal[fit_index],
      _pDevObservedTraceTranspose,
      _pDevIval,
      _pDevScratch_ival, // FLxNxFx2  //scratch for both ival and fval
      _pDevNucRise, // FL x ISIG_SUB_STEPS_MULTI_FLOW x F 
      _pDevSbg, // FLxF
      _pDevEmphasis, // MAX_HPLEN+1 xF // needs precomputation
      _pDevNon_integer_penalty, // MAX_HPLEN
      _pDevDarkMatterComp, // NUMNUC * F  
      _pDevBeadParamsTranspose, // we will be indexing directly into it from the parameter indices provide by CpuStep_t
      _pDevBeadParamsEval,
      _pDevLambdaForBeadFit,
      _pDevJTJ, // jtj matrix
      _pDevLTR, // lower triangular matrix
      _pDevRHS, // rhs vector
      _pd_delta,
      _pDevBeadParamIdxMap, 
      _fd[fit_index]->GetNumParamsToFit(),
      N,
      F,
      _pDevResidual, // N 
      getId());
    }
 }

void MultiFitStream::executeTransposeParamsBack()
{

  //cout << "TransposeParamsBack Kernel" << endl;
  int padN = _myJob.getPaddedN();

  dim3 block(32,32);
  int StructLength = (sizeof(bead_params)/sizeof(float));

  dim3 grid ((padN+block.y-1)/block.y, (StructLength + block.x-1)/block.x );

  transposeData_Wrapper(grid, block, 0 ,_stream, _pDevBeadParams, _pDevBeadParamsTranspose, padN,  StructLength);
  CUDA_ERROR_CHECK();
}


void MultiFitStream::copyBeadParamsToDevice()
{
 cudaMemcpyAsync((bead_params*)_pDevBeadParams , _pHostBeadParams , validValue(_myJob.getBeadParamsSize(),__LINE__), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();
}


void MultiFitStream::copyBeadParamsToHost()
{
 cudaMemcpyAsync( _pHostBeadParams  , (bead_params*)_pDevBeadParams, validValue(_myJob.getBeadParamsSize(),__LINE__), cudaMemcpyDeviceToHost, _stream); CUDA_ERROR_CHECK();
}


int MultiFitStream::handleResults()
{

  if(_myJob.isSet()){
    memcpy( _myJob.getBeadParams(),_pHostBeadParams, _myJob.getBeadParamsSize());  
    _myJob.KeyNormalize();   // temporary call to key normalize till we put it into a GPU kernel
    _fitIter++;

    if(_fitIter < CUDA_MULTIFLOW_NUM_FIT){
      memcpy(_pHostBeadParams, _myJob.getBeadParams(),_myJob.getBeadParamsSize());  
      return 1;  //signal more work to be done;
    }
    _myJob.putRemainRegionFit(_item);
    _fitIter = 0;
  }
  return 0; //signal Job com[plete
}


bool MultiFitStream::ValidJob() {
  _myJob.setData((BkgModelWorkInfo *)getJobData()); 
  
  return _myJob.ValidJob();
}

void MultiFitStream::SetUpLambdaArray(int fit_index) {
  for (int i=0; i<_myJob.getNumBeads(); ++i) {
    _pHostLambdaForBeadFit[fit_index][i] = _lambda_start[fit_index];
  }
}

//static Function

void MultiFitStream::setBeadsPerBLock(int bpb)
{
  _bpb = bpb;
}

void MultiFitStream::ExecuteJob(int * control)
{

  
//  printInfo(); cout << " i: " <<  _fitIter << " numBeads: " << _myJob.getNumBeads() << " numFrames:" << _myJob.getNumFrames() << endl;
  

  if(_fitIter == 0){
    //CalculateCoarseNucRise();

    CalculateNonIntegerPenalty();
    resetPointers();
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

void MultiFitStream::CalculateClonalityRestriction(int fit_index)
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

void MultiFitStream::CalculateNonIntegerPenalty()
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
