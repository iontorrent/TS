/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>

#include "cuda_error.h"
#include "cuda_runtime.h"

#include "StreamingKernels.h" 
#include "StreamManager.h"
#include "SingleFitStream.h"
#include "JobWrapper.h"
#include "GpuMultiFlowFitControl.h"


//#define MIN_MEMORY_FOR_ONE_STREAM (450*1024*1024)
 
using namespace std;

int SingleFitStream::_cntSingleFitStream[MAX_CUDA_DEVICES] = {0};
int SingleFitStream::_bpb = 128;
   


/////////////////////////////////////////////////
//FIT STREAM CLASS
SingleFitStream::SingleFitStream(WorkerInfoQueue * Q) : cudaStreamExecutionUnit(Q)
{
	//worst case scenario:
  int N = GpuMultiFlowFitControl::GetMaxBeads();
  int F = GpuMultiFlowFitControl::GetMaxFrames();
  int bft = NUMFB*F;
  int padN = ((N+32-1)/32) * 32; 

  setName("SingleFitStream");

  cudaGetDevice(&_devId);
  _cntSingleFitStream[_devId]++;
  _BeadBaseAllocSize = 0;

  //over allocate to allow for 128 byte alignements of buffers
  _BeadBaseAllocSize += padN * sizeof(bead_params); //bead params
  _BeadBaseAllocSize += padN * sizeof(bead_state);  // bead state
	_BeadBaseAllocSize += sizeof(float) * NUMNUC*F +128; //darkmatter  
  _BeadBaseAllocSize += sizeof(float) * (MAX_HPLEN+1)*F +128;  // EmphVec 
	_BeadBaseAllocSize += sizeof(float) * bft + 128; //shifted background
  _BeadBaseAllocSize += sizeof(float) * ISIG_SUB_STEPS_SINGLE_FLOW * F * NUMFB + 128; // nucRise

  _FgBufferAllocSizeHost = bft * padN * sizeof( FG_BUFFER_TYPE); 
  _FgBufferAllocSizeDevice = bft * padN * sizeof(float);

  int ScratchSpaceAllocSize = 0;
  ScratchSpaceAllocSize += 4*padN*F;  
  ScratchSpaceAllocSize += 1*padN*NUMFB;
  ScratchSpaceAllocSize *= sizeof(float);

//  _HostMonitor = _DevMonitor = NULL;

  _HostBeadBase = NULL;
  _HostConstP = NULL;
  _HostFgBuffer = NULL;
  _DevWeightFgBuffer = NULL;
  _DevFgBufferFloat = NULL;
  _DevBeadBase = NULL;
  _DevWorkBase = NULL;
  _DevBeadParamTransp = NULL;
 
 try{
  
    cudaHostAlloc(&_HostFgBuffer, _FgBufferAllocSizeHost, cudaHostAllocDefault); CUDA_ALLOC_CHECK(_HostFgBuffer);  
    cudaHostAlloc(&_HostBeadBase, _BeadBaseAllocSize, cudaHostAllocDefault ); CUDA_ALLOC_CHECK(_HostBeadBase);  
    cudaHostAlloc(&_HostConstP, sizeof(ConstParams), cudaHostAllocDefault); CUDA_ALLOC_CHECK(_HostConstP);  

//    cudaHostAlloc(&_HostMonitor, (MAX_HPLEN+1) * sizeof(int), cudaHostAllocDefault); CUDA_ALLOC_CHECK(_HostMonitor);  
//    cudaMalloc( &_DevMonitor, (MAX_HPLEN+1) * sizeof(int));  CUDA_ALLOC_CHECK(_DevMonitor);

   
    //DEVICE ALLOCS
    cudaMalloc( &_DevWeightFgBuffer, _FgBufferAllocSizeHost);  CUDA_ALLOC_CHECK(_DevWeightFgBuffer);
    //cudaMalloc( &_DevWeightFgBuffer, _FgBufferAllocSizeDevice);  CUDA_ALLOC_CHECK(_DevWeightFgBuffer);
    cudaMalloc( &_DevFgBufferFloat, _FgBufferAllocSizeDevice);  CUDA_ALLOC_CHECK(_DevFgBufferFloat);
    cudaMalloc( &_DevWorkBase, ScratchSpaceAllocSize );  CUDA_ALLOC_CHECK(_DevWorkBase);
    cudaMalloc( &_DevBeadBase, _BeadBaseAllocSize);  CUDA_ALLOC_CHECK(_DevBeadBase);
    cudaMalloc( &_DevBeadParamTransp, padN*sizeof(bead_params));  CUDA_ALLOC_CHECK(_DevBeadParamTransp);


    // over allocation to prevent buffers at the end of the address space on the device/ lead to invalid pointer error
    // if pointers got bedn into that address space.

  }
  catch( cudaException& e)
  {
    cleanUp();
    throw e; 
  }

 // if(overAllocation != NULL) cudaFree(overAllocation);

  _d_pFgBuffer = (FG_BUFFER_TYPE*)_DevWeightFgBuffer;
/*
  std::cout << ">>>>> MEMORYMAP: Stream" << _cntSingleFitStream[_devId]  << " HostPLMemory Inputs " << _BeadBaseAllocSize <<" Bytes" << std::endl;
  std::cout << ">>>>> MEMORYMAP: Stream" << _cntSingleFitStream[_devId]  << " HostPLMemory fgBuff " << _FgBufferAllocSizeHost <<" Bytes" << std::endl;
  std::cout << ">>>>> MEMORYMAP: Stream" << _cntSingleFitStream[_devId]  << " HostPLMemory ConstMemStruct " << sizeof(ConstParams) <<" Bytes" << std::endl;
  std::cout << ">>>>> MEMORYMAP: Stream" << _cntSingleFitStream[_devId]  << " Constant ConstMemStruct " << sizeof(ConstParams) <<" Bytes" << std::endl;
  std::cout << ">>>>> MEMORYMAP: Stream" << _cntSingleFitStream[_devId]  << " Device Inputs " <<  _BeadBaseAllocSize<<" Bytes" << std::endl;
  std::cout << ">>>>> MEMORYMAP: Stream" << _cntSingleFitStream[_devId]  << " device WorkSpcae " << ScratchSpaceAllocSize <<" Bytes" << std::endl;
  std::cout << ">>>>> MEMORYMAP: Stream" << _cntSingleFitStream[_devId]  << " Device BeadParamsTransp " <<  padN * sizeof(bead_params) <<" Bytes" << std::endl;
  std::cout << ">>>>> MEMORYMAP: Stream" << _cntSingleFitStream[_devId]  << " Device FgBufferIn " <<  _FgBufferAllocSizeDevice <<" Bytes" << std::endl;
  std::cout << ">>>>> MEMORYMAP: Stream" << _cntSingleFitStream[_devId]  << " Device FgBufferTransp " <<  _FgBufferAllocSizeDevice  <<" Bytes" << std::endl;
*/

}



SingleFitStream::~SingleFitStream()
{
  cleanUp();
}

void SingleFitStream::cleanUp()
{
  if(_DevBeadParamTransp != NULL) cudaFree(_DevBeadParamTransp); _DevBeadParamTransp = NULL;CUDA_ERROR_CHECK();
  if(_DevBeadBase != NULL) cudaFree(_DevBeadBase); _DevBeadBase = NULL;CUDA_ERROR_CHECK();
  if(_DevWorkBase != NULL) cudaFree(_DevWorkBase); _DevWorkBase = NULL;CUDA_ERROR_CHECK();
  if(_DevFgBufferFloat != NULL) cudaFree(_DevFgBufferFloat); _DevFgBufferFloat = NULL;CUDA_ERROR_CHECK();
  if(_DevWeightFgBuffer != NULL) cudaFree(_DevWeightFgBuffer); _DevWeightFgBuffer = NULL;CUDA_ERROR_CHECK();

//  if(_DevMonitor != NULL) cudaFree(_DevMonitor); _DevMonitor = NULL;CUDA_ERROR_CHECK();
//  if(_HostMonitor != NULL) cudaFreeHost(_HostMonitor); _HostMonitor = NULL;CUDA_ERROR_CHECK();


  if(_HostFgBuffer != NULL) cudaFreeHost(_HostFgBuffer); _HostFgBuffer = NULL;CUDA_ERROR_CHECK();
  if(_HostConstP != NULL) cudaFreeHost(_HostConstP); _HostConstP = NULL;CUDA_ERROR_CHECK();
  if(_HostBeadBase != NULL) cudaFreeHost(_HostBeadBase); _HostBeadBase = NULL;CUDA_ERROR_CHECK();


  _cntSingleFitStream[_devId]--;
/*
  if(_cntSingleFitStream[_devId] == 0 )
  {
    if(_DevPoissBase[_devId] != NULL){
      if(_DevPoissBase[_devId] != NULL) cudaFree(_DevPoissBase[_devId]); _DevPoissBase[_devId] = NULL;CUDA_ERROR_CHECK();
      _init[_devId] = false;
    }
  }
*/
}



void SingleFitStream::resetPointers()
{

  _N = _myJob.getNumBeads();
  _F = _myJob.getNumFrames();

  _padN = ((_N+32-1)/32) * 32;

  //cout << getId() << " resetting pointers for job with " << _N << "("<< _padN <<") beads and " << _F << " frames" << endl;

  int offsetBytes = 0;
  
  //cerr << getId() << " setting pointers for " << sizeof(float)*(6*_N+2*_N*_F) << " bytes of input data" << endl; 
    //host bead pointer

  _h_pBeadParams = (bead_params*)_HostBeadBase;
  _d_pBeadParams = (bead_params*)_DevBeadBase; 

  offsetBytes += _padN * sizeof(bead_params);
  offsetBytes += 128 - (offsetBytes%128); 

  _h_pBeadState = (bead_state*)(_HostBeadBase + offsetBytes);
  _d_pBeadState = (bead_state*)(_DevBeadBase + offsetBytes);

  offsetBytes += _padN * sizeof(bead_state);
  offsetBytes += 128 - (offsetBytes%128); 
  //bead Params and State are our outputs. therefor:  
  _copyOutSize = offsetBytes;
  
  _h_pDarkMatter = (float*)(_HostBeadBase + offsetBytes); // NUMNUC*F 
  _d_pDarkMatter = (float*)(_DevBeadBase + offsetBytes); // NUMNUC*F

  offsetBytes += sizeof(float)*NUMNUC*_F;
  offsetBytes += 128 - (offsetBytes%128); 

  _h_pShiftedBkg = (float*)(_HostBeadBase + offsetBytes); // NUMFB*F 
  _d_pShiftedBkg = (float*)(_DevBeadBase + offsetBytes); // NUMFB*F

  offsetBytes += sizeof(float)*NUMFB*_F;
  offsetBytes += 128 - (offsetBytes%128); 

  _h_pEmphVector = (float*)(_HostBeadBase + offsetBytes); // (MAX_HPLEN+1)*F 
  _d_pEmphVector= (float*)(_DevBeadBase + offsetBytes); // (MAX_HPLEN+1)*F

  offsetBytes += sizeof(float)*(MAX_HPLEN+1)*_F;
  offsetBytes += 128 - (offsetBytes%128); 

  _h_pNucRise = (float*)(_HostBeadBase + offsetBytes); // ISIG_SUB_STEPS_SINGLE_FLOW * F * NUMFB 
  _d_pNucRise= (float*)(_DevBeadBase + offsetBytes); // ISIG_SUB_STEPS_SINGLE_FLOW * F * NUMFB 

  _copyInSize = offsetBytes + ISIG_SUB_STEPS_SINGLE_FLOW * _F * NUMFB * sizeof(float);

   
  

   //dev pointer after transpose (Structure of Arrays)
  _d_pCopies = _DevBeadParamTransp;
  _d_pR = _d_pCopies + _padN; // N 
  _d_pDmult = _d_pR + _padN; // N
  _d_pGain = _d_pDmult + _padN; // N
  _d_pAmpl =  _d_pGain + _padN; // N * NUMFB
  _d_pKmult = _d_pAmpl +_padN*NUMFB; // N * NUMFB

   //device scratch space pointers
  _d_err = _DevWorkBase; // NxF
  _d_fval = _d_err +_padN*_F; // NxF
  _d_tmp_fval = _d_fval + _padN*_F; // NxF
  _d_jac = _d_tmp_fval + _padN*_F; // NxF 
//  _d_pTauB = _d_jac + _padN *_F; // N * NUMFB
//  _d_pSP = _d_pTauB + _padN*NUMFB; //  N * NUMFB
//  _d_pMeanErr = _d_pSP + _padN*NUMFB; // N * NUMFB 
  _d_pMeanErr = _d_jac + _padN*_F; // N * NUMFB 

//  cudaMemset(_DevMonitor, 0 , (MAX_HPLEN+1)*sizeof(int));

}


void SingleFitStream::serializeInputs()
{
  
  //cout << getId() <<" serialize data for asnync global mem copy" << endl;
  memcpy(_HostFgBuffer,_myJob.getFgBuffer() ,sizeof(FG_BUFFER_TYPE)*_N*_F*NUMFB);
  // TODO: coud already start firtst async copy here


  memcpy(_h_pBeadParams, _myJob.getBeadParams(), sizeof(bead_params)*_N);  
  memcpy(_h_pBeadState, _myJob.getState(), sizeof(bead_state)*_N); 
  memcpy(_h_pDarkMatter, _myJob.getDarkMatter(), sizeof(float)*NUMNUC*_F); 
  memcpy(_h_pShiftedBkg, _myJob.getShiftedBackground(), sizeof(float)*NUMFB*_F); 
  memcpy(_h_pEmphVector, _myJob.getEmphVec(), sizeof(float)*(MAX_HPLEN+1)*_F);
  memcpy(_h_pNucRise, _myJob.getCalculateNucRise(), sizeof(float) * ISIG_SUB_STEPS_SINGLE_FLOW * _F * NUMFB);  

  //cout << getId() <<" collect data for async const mem copy" << endl;
  *((reg_params* )_HostConstP) = *(_myJob.getRegionParams()); // 4
  memcpy( _HostConstP->start, _myJob.getStartNuc(),sizeof(int)*NUMFB );
  memcpy( _HostConstP->deltaFrames, _myJob.getDeltaFrames(),sizeof(float)*_F );
  memcpy( _HostConstP->flowIdxMap, _myJob.getFlowIdxMap(), sizeof(int)*NUMFB);  

}



//////////////////////////
// ASYNC CUDA FUNCTIONS, KERNEL EXECUTION AND DATA HANDLING


void SingleFitStream::prepareInputs()
{
  //prepare environment for new job
  
  preProcessCpuSteps();
  resetPointers();
  serializeInputs();      
}

void SingleFitStream::copyToDevice()
{
 // move data to device
  //cout <<  getId() << " Async Copy To Device" << endl;
  //cout << getId() << " bead input size " << _copyInSize/(1024.0*1024.0) << "MB, output size " << _copyOutSize/(1024.0*1024.0) << "MB"<< endl;
  //cout << getId() << " fgBuffer input size " << ( sizeof(FG_BUFFER_TYPE)*_N*_F*NUMFB)/(1024.0*1024.0) << "MB"<< endl
;
  cudaMemcpyAsync( _DevWeightFgBuffer, _HostFgBuffer, _FgBufferAllocSizeHost, cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();
  cudaMemcpyAsync( _DevBeadBase, _HostBeadBase, _copyInSize, cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();


  copySingleFlowFitConstParamAsync(_HostConstP, getId() ,_stream);CUDA_ERROR_CHECK();
 //cudaMemcpyToSymbolAsync ( CP, _HostConstP, sizeof(ConstParams), getId()*sizeof(ConstParams),cudaMemcpyHostToDevice, _stream);CUDA_ERROR_CHECK();

}



void SingleFitStream::executeKernel()
{
  //cout <<  getId() << " Async Kernel Exec" << endl;

  dim3 block(32,32);
  dim3 grid( (_F*NUMFB+ block.x-1)/block.x , (_padN+block.y-1)/block.y);
  
  transposeDataToFloat_Wrapper (grid, block, 0 ,_stream,_DevFgBufferFloat, _d_pFgBuffer, _F*NUMFB, _padN);

  int StructLength = (sizeof(bead_params)/sizeof(float));
  if((sizeof(bead_params)%sizeof(float)) != 0 )
  { 
    cerr << "Structure not a multiple of sizeof(float), transpose not possible" << endl;
    exit(-1);
    /*
    float Copies;        // P is now copies per bead, normalized to 1E+6
    float R;  // ratio of bead buffering to empty buffering - one component of many in this calculation
    float dmult;  // diffusion alteration of this well
    float gain;  // responsiveness of this well
    float Ampl[NUMFB]; // homopolymer length mixture
    float kmult[NUMFB];  // individual flow multiplier to rate of enzyme action
    ...
  */
  }
 
  grid.x = (StructLength + block.x-1)/block.x  ;
  grid.y = (_padN+block.y-1)/block.y;
  transposeData_Wrapper( grid, block, 0 ,_stream, (float*)_DevBeadParamTransp, (float*)_d_pBeadParams, StructLength, _padN);

 
  //
    
  block.x = _bpb;
  block.y = 1;

  grid.y = 1;
  grid.x = (_N+block.x-1)/block.x;

  PreSingleFitProcessing_Wrapper( grid, block, 0 , _stream,
  // Here FL stands for flows
  // inputs from data reorganization
  _d_pCopies, // N
  _d_pR, // N
  _d_pDmult, // N
  _d_pGain, // N
  _d_pAmpl, // FLxN
  _d_pKmult, // FLxN
 _d_pShiftedBkg, // FLxF 
  _d_pDarkMatter, // FLxF
  _DevFgBufferFloat, // FLxFxN
  // other inputs 
  //TODO
   _myJob.getAbsoluteFlowNum(), // starting flow number to calculate absolute flow num
  _N, // 4
  _F, // 4
  MAX_HPLEN+1, // 4
  // outputs
//  _d_pSP, // FLxN
//  _d_pTauB, 
  //_myJob.performAlternatingFit(),
  false,
  getId()
);


/*if (_myJob.performAlternatingFit())
{
  PerFlowAlternatingFit_Wrapper( grid, block, sharedMem, _stream,
      // inputs
      _DevFgBufferFloat, 
      _d_pEmphVector, 
      _d_pNucRise,
      // bead params
      _d_pAmpl, // N
      _d_pKmult, // N
      _d_pDmult, // N
      _d_pTauB, // FNUM * N
      _d_pGain, // N
      _d_pSP, // FNUM * N
      _d_pCopies,
      _d_pBeadState,
      
      // scratch space in global memory
      _d_err, //
      _d_fval, // NxF
      _d_tmp_fval, // NxF
      _d_pMeanErr, 
    // other inputs
      _myJob.getAmpLowLimit(),
      _myJob.getkmultHighLimit(),
      _myJob.getkmultLowLimit(),
      _myJob.getAbsoluteFlowNum() , // real flow number 
      _myJob.getNumBeads(), // 4
      _myJob.getNumFrames(), // 4
      getId()  // stream id for offset in const memory
    );
}
*/
  
int sharedMem = sizeof(float) * (MAX_HPLEN + 1) * _myJob.getNumFrames();
PerFlowLevMarFit_Wrapper( grid, block, sharedMem, _stream,
      // inputs
      _DevFgBufferFloat, 
      _d_pEmphVector,
      _d_pNucRise,
      // bead params
//      _d_pAmpl, // N
//      _d_pKmult, // N
//      _d_pDmult, // N
//      _d_pR, // FNUM * N
//      _d_pGain, // N
//      _d_pSP, // FNUM * N
      _d_pCopies,
      _d_pBeadState,
      
      // scratch space in global memory
      _d_err, //
      _d_fval, // NxF
      _d_tmp_fval, // NxF
      _d_jac, //NxF 
      _d_pMeanErr, 
    // other inputs
      _myJob.getAmpLowLimit(),
      _myJob.getkmultHighLimit(),
      _myJob.getkmultLowLimit(),
      _myJob.getAbsoluteFlowNum() , // real flow number 
      _myJob.getNumBeads(), // 4
      _myJob.getNumFrames(), // 4
      _myJob.useDynamicEmphasis(),
//      _DevMonitor,
      getId()  // stream id for offset in const memory
    );

  block.x = 32;
  block.y = 32;

  grid.x = (_padN+block.y-1)/block.y;
  grid.y = (StructLength + block.x-1)/block.x;

  transposeData_Wrapper( grid, block, 0 ,_stream,(float*)_d_pBeadParams, (float*)_DevBeadParamTransp, _padN,  StructLength);

}

void SingleFitStream::copyToHost()
{
 //cout << getId() <<  " Async copy back" <<endl;
 cudaMemcpyAsync( _HostBeadBase, _DevBeadBase, _copyOutSize , cudaMemcpyDeviceToHost, _stream); CUDA_ERROR_CHECK();

// cudaMemcpyAsync( _HostMonitor, _DevMonitor, (MAX_HPLEN+1)*sizeof(int) , cudaMemcpyDeviceToHost, _stream); CUDA_ERROR_CHECK();


}


int SingleFitStream::handleResults()
{
  //cout << getId() <<  " Handling Results" <<endl;

  if(_myJob.isSet()){
    // for actual pipeline we have to copy the results back into original buffer
    memcpy(_myJob.getBeadParams(), _h_pBeadParams, sizeof(bead_params)*_N);       
    memcpy(_myJob.getState(), _h_pBeadState, sizeof(bead_state)*_N);       

    // to reuse the same job we have to move it to a temporal buffer:
    //memcpy(_myJob.getResu(), _h_pBeadParams, (sizeof(bead_params)+sizeof(bead_state))*_N);       

/*    cout << "POISSON TABLE ACCESSES: ";
    for(int i=0; i< MAX_HPLEN+1; i++){  
      cout << _HostMonitor[i] << " ";
    }
    cout << endl;*/
    //cout << getId() << " Move Item to follow up Q" <<endl;
  

    _myJob.putPostFitStep(_item);
    //BkgModelWorkInfo *info =  _myJob.getInfo();

    //cout << "Putting item in cpu queue" << endl;
    //cout << "Region---> row: " << info->bkgObj->region_data->region->row <<
    //    " col: " << info->bkgObj->region_data->region->col << " livebeads: " <<
    //    info->bkgObj->region_data->my_beads.numLBeads << endl;
    //info->type = POST_FIT_STEPS;
    //info->bkgObj->region_data->fitters_applied=TIME_TO_DO_PREWELL;
    //info->pq->GetCpuQueue()->PutItem(_item);

    // RESET TIMING AFTER FIRST 20 FLOWS
    //if( _myJob.getAbsoluteFlowNum() == 0) _sd->resetTimer();      
  }

  return 0;
}

bool SingleFitStream::ValidJob() {
    _myJob.setData((BkgModelWorkInfo *)getJobData()); 

/*    BkgModelWorkInfo *info =  _myJob.getInfo();
    if (info->bkgObj->region_data->fitters_applied == -1) {
      return false;
    }
    return true;*/

    return _myJob.ValidJob();
}

void SingleFitStream::preProcessCpuSteps() {
  _myJob.setUpFineEmphasisVectors();
}



//static Function


void SingleFitStream::setBeadsPerBLock(int bpb)
{
  _bpb = bpb;
}

void SingleFitStream::ExecuteJob(int * control)
{

  if(control != NULL) *control = _myJob.getAbsoluteFlowNum();

  prepareInputs();
  copyToDevice();
  executeKernel();
  copyToHost();
}

// Always the same
/*
void SingleFitStream::InitializeConstantMemory(PoissonCDFApproxMemo& poiss_cache)
{

  int DeviceId;
  cudaGetDevice(&DeviceId);



  if(!_init[DeviceId]){

    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;

    if(free_byte < MIN_MEMORY_FOR_ONE_STREAM) throw cudaNotEnoughMemForStream(__FILE__,__LINE__); 

    //cout << "init Poisson Tables" << endl;
    //PoissonCDFApproxMemo poiss_cache; 
    //poiss_cache.Allocate (MAX_HPLEN+1,MAX_POISSON_TABLE_ROW,POISSON_TABLE_STEP);
    //poiss_cache.GenerateValues(); // fill out my table
#ifndef USE_GLOBAL_POISSTABLE

    cudaMemcpyToSymbol (POISS_0_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[0], sizeof (float) *MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol (POISS_1_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[1], sizeof (float) *MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol (POISS_2_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[2], sizeof (float) *MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol (POISS_3_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[3], sizeof (float) *MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol (POISS_4_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[4], sizeof (float) *MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol (POISS_5_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[5], sizeof (float) *MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol (POISS_6_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[6], sizeof (float) *MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol (POISS_7_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[7], sizeof (float) *MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol (POISS_8_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[8], sizeof (float) *MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol (POISS_9_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[9], sizeof (float) *MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol (POISS_10_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[10], sizeof (float) *MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();
    cudaMemcpyToSymbol (POISS_11_APPROX_TABLE_CUDA, poiss_cache.poiss_cdf[11], sizeof (float) *MAX_POISSON_TABLE_ROW); CUDA_ERROR_CHECK();

#else
 
  int poissTableSize = (MAX_HPLEN + 1) * MAX_POISSON_TABLE_ROW * sizeof(float);
  cudaMalloc(&_DevPoissBase[DeviceId], poissTableSize); CUDA_ERROR_CHECK();
  float * ptr = _DevPoissBase[DeviceId];
  for(int i = 0; i< (MAX_HPLEN+1); i++)
  {
    cudaMemcpy(ptr, poiss_cache.poiss_cdf[i], sizeof(float)*MAX_POISSON_TABLE_ROW, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
    ptr += MAX_POISSON_TABLE_ROW;
  }
  cudaMemcpyToSymbol (POISS_APPROX_TABLE_CUDA_BASE , &_DevPoissBase[DeviceId]  , sizeof (float*)); CUDA_ERROR_CHECK();

#endif



#ifndef USE_CUDA_ERF
    cudaMemcpyToSymbol (ERF_APPROX_TABLE_CUDA, ERF_APPROX_TABLE, sizeof (ERF_APPROX_TABLE)); CUDA_ERROR_CHECK();
#endif

  }
  
 // std::cout << ">>>>> MEMORYMAP: PerProcess Constant Poisson " <<  (sizeof (float) *MAX_POISSON_TABLE_ROW * 12) <<" Bytes" << std::endl;


  _init[DeviceId] = true;

}
*/



