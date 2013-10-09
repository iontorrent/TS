/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

// patch for CUDA5.0/GCC4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

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

////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Simple


int SimpleSingleFitStream::_bpb = -1;
int SimpleSingleFitStream::_l1type = -1;
int SimpleSingleFitStream::_fittype = -1;  //0 GaussNewton, 1 LevMar, 2 Hybrid, 3 Relaxing Kmult Gauss Newton
int SimpleSingleFitStream::_hybriditer = -1; // LevMar after N iter of Gauss newton

int SimpleSingleFitStream::l1DefaultSetting()
{
  // 0: Equal, 1: Shared preferred, 2: L1 preferred
  if(_computeVersion == 20 ) return 2;
  if(_computeVersion >= 35 ) return 0;
  return 0;
}

int SimpleSingleFitStream::BlockSizeDefaultSetting()
{
  if(_computeVersion == 20 ) return 128;
  if(_computeVersion >= 35 ) return 256;
  return 128;
}
//for Monitor
//int * SimpleSingleFitStream::_IterBuffer = NULL;


/////////////////////////////////////////////////
//FIT STREAM CLASS
SimpleSingleFitStream::SimpleSingleFitStream(streamResources * res, WorkerInfoQueueItem item ) : cudaSimpleStreamExecutionUnit(res, item)
{


  setName("SingleFitStream");

  if(_verbose) cout << getLogHeader()  << " created " << endl;

  _HostConstP = NULL;
  _HostFgBuffer = NULL;
  _DevWeightFgBuffer = NULL;
  _DevFgBufferFloat = NULL;
  _DevBeadBase = NULL;
  _DevWorkBase = NULL;
  _DevBeadParamTransp = NULL;

}



SimpleSingleFitStream::~SimpleSingleFitStream()
{
  cleanUp();
}

void SimpleSingleFitStream::cleanUp()
{
 if(_verbose) cout << getLogHeader()  << " clean up" <<  endl;

}



void SimpleSingleFitStream::resetPointers()
{

  _N = _myJob.getNumBeads();
  _F = _myJob.getNumFrames();


  if(!_Device->checkMemory( getMaxDeviceMem(_F,_N)))
    cout << getLogHeader() << " Successfully reallocated device memory to handle Job" << endl;

  _padN = _myJob.getPaddedN();

  if(_verbose) cout << getLogHeader() << " resetting pointers for job with " << _N << "("<< _padN <<") beads and " << _F << " frames" << endl;

//  int ScratchSpaceAllocSize =0 ;
//  ScratchSpaceAllocSize += 4*_padN*_F;  
//  ScratchSpaceAllocSize += MAX_XTALK_NEIGHBOURS*_padN*_F;  
//  ScratchSpaceAllocSize += 1*_padN*NUMFB;
//  ScratchSpaceAllocSize *= sizeof(float);

       
  _HostConstP  = (ConstParams*)_Host->getSegment(sizeof(ConstParams)); CUDA_ALLOC_CHECK(_HostConstP);  

  if(_myJob.performCrossTalkCorrection()){
    _HostConstXtalkP  = (ConstXtalkParams*)_Host->getSegment(sizeof(ConstXtalkParams)); CUDA_ALLOC_CHECK(_HostConstXtalkP); 
  }

  _HostFgBuffer = (FG_BUFFER_TYPE*)_Host->getSegment(_myJob.getFgBufferSizeShort(true)); CUDA_ALLOC_CHECK(_HostFgBuffer); 

  if(_myJob.performCrossTalkCorrection()){
    _HostNeiIdxMap = (int*)_Host->getSegment(_myJob.getXtalkNeiIdxMapSize(true)); CUDA_ALLOC_CHECK(_HostNeiIdxMap); 
  }
    //DEVICE ALLOCS
  _DevWeightFgBuffer = (float*)_Device->getSegment(_myJob.getFgBufferSizeShort(true));  CUDA_ALLOC_CHECK(_DevWeightFgBuffer);
  _DevFgBufferFloat = (float*)_Device->getSegment(_myJob.getFgBufferSize(true));  CUDA_ALLOC_CHECK(_DevFgBufferFloat);
  _DevWorkBase =(float*) _Device->getSegment(getScratchSpaceAllocSize(_myJob)   );  CUDA_ALLOC_CHECK(_DevWorkBase);
//    _DevBeadBase = (char*)_Device->getSegment(_BeadBaseAllocSize);  CUDA_ALLOC_CHECK(_DevBeadBase);
  _DevBeadParamTransp = (float*)_Device->getSegment(_myJob.getBeadParamsSize(true));  CUDA_ALLOC_CHECK(_DevBeadParamTransp);

  _d_pFgBuffer = (FG_BUFFER_TYPE*)_DevWeightFgBuffer;


  //cerr << getId() << " setting pointers for " << sizeof(float)*(6*_N+2*_N*_F) << " bytes of input data" << endl; 
    //host bead pointer

  _Host->startNewSegGroup();

  _h_pBeadParams =  (bead_params*) _Host->getSegment( _myJob.getBeadParamsSize(true)); 
  _d_pBeadParams =  (bead_params*) _Device->getSegment( _myJob.getBeadParamsSize(true));

  _h_pBeadState = (bead_state*) _Host->getSegment( _myJob.getBeadStateSize(true));
  _d_pBeadState = (bead_state*) _Device->getSegment( _myJob.getBeadStateSize(true));

  //bead Params and State are our outputs. therefore:
  _copyOutSize = _Host->getCurrentSegGroupSize();
  
  _h_pDarkMatter = (float*)_Host->getSegment(_myJob.getDarkMatterSize(true)); // NUMNUC*F 
  _d_pDarkMatter = (float*)_Device->getSegment(_myJob.getDarkMatterSize(true)); // NUMNUC*F

  _h_pShiftedBkg = (float*)_Host->getSegment(_myJob.getShiftedBackgroundSize(true)); // NUMFB*F 
  _d_pShiftedBkg = (float*)_Device->getSegment(_myJob.getShiftedBackgroundSize(true));// NUMFB*F

  _h_pEmphVector = (float*)_Host->getSegment(_myJob.getEmphVecSize(true)); // (MAX_POISSON_TABLE_COL)*F 
  _d_pEmphVector= (float*)_Device->getSegment(_myJob.getEmphVecSize(true)); // (MAX_POISSON_TABLE_COL)*F

  _h_pNucRise = (float*)_Host->getSegment(_myJob.getNucRiseSize(true));  // ISIG_SUB_STEPS_SINGLE_FLOW * F * NUMFB 
  _d_pNucRise= (float*)_Device->getSegment(_myJob.getNucRiseSize(true)); // ISIG_SUB_STEPS_SINGLE_FLOW * F * NUMFB 

  _copyInSize = _Host->getCurrentSegGroupSize();

   
  

   //dev pointer after transpose (Structure of Arrays)
  _d_pCopies = _DevBeadParamTransp;
  _d_pR = _d_pCopies + _padN; // N 
  _d_pDmult = _d_pR + _padN; // N
  _d_pGain = _d_pDmult + _padN; // N
  _d_pAmpl =  _d_pGain + _padN; // N * NUMFB
  _d_pKmult = _d_pAmpl +_padN*NUMFB; // N * NUMFB
  _d_pPCA_Vals = _d_pKmult + _padN*NUMFB; // N*NUM_DM_PCA
  _d_pTau_Adj = _d_pPCA_Vals + _padN*NUM_DM_PCA; // N

   //device scratch space pointers
  _d_avg_trc = _DevWorkBase; // NxF
  _d_err = _d_avg_trc + _padN*_F; // NxF
  _d_fval = _d_err +_padN*_F; // NxF
  _d_tmp_fval = _d_fval + _padN*_F; // NxF
  _d_jac = _d_tmp_fval + _padN*_F; // 3*NxF Can be reduced in Taubadjust kernel 
  _d_pMeanErr = _d_jac + 3*_padN*_F; // N * NUMFB 

  // xtalk scratch space pointers
  _d_pNeiContribution = _DevWorkBase;
  _d_pXtalk = _d_pNeiContribution + _myJob.getNumXtalkNeighbours()*_padN*_F;
  _d_pXtalkScratch = _d_pXtalk + _padN*_F;
  _d_pNeiIdxMap = (int*)(_d_pXtalkScratch + 3*_padN*_F);

  //TODO Monitor
  //if(_IterBuffer == NULL){ 
  //  _IterBuffer = new int[ITER];
  // for(int i=0; i< ITER; i++) _IterBuffer[i] = 0;
  // }
  //cudaHostAlloc(&_HostMonitor, (ITER ) * sizeof(int), cudaHostAllocDefault);
  //cudaMalloc( &_DevMonitor, (ITER) * sizeof(int));  ;
  //cudaMemset(_DevMonitor, 0 , (ITER)*sizeof(int));
}


void SimpleSingleFitStream::serializeInputs()
{
  
  if(_verbose) cout <<  getLogHeader() <<" serialize data for async global mem copy" << endl;

  memcpy(_HostFgBuffer,_myJob.getFgBuffer() ,_myJob.getFgBufferSizeShort());

  memcpy(_h_pBeadParams, _myJob.getBeadParams(), _myJob.getBeadParamsSize());  
  memcpy(_h_pBeadState, _myJob.getBeadState(),_myJob.getBeadStateSize()); 
  memcpy(_h_pDarkMatter, _myJob.getDarkMatter(), _myJob.getDarkMatterSize()); 
  memcpy(_h_pShiftedBkg, _myJob.getShiftedBackground(), _myJob.getShiftedBackgroundSize()); 
  memcpy(_h_pEmphVector, _myJob.getEmphVec(), _myJob.getEmphVecSize());
  memcpy(_h_pNucRise, _myJob.getCalculateNucRise(), _myJob.getNucRiseSize());  

  //cout << getId() <<" collect data for async const mem copy" << endl;
  *((reg_params* )_HostConstP) = *(_myJob.getRegionParams()); // 4
  memcpy( _HostConstP->start, _myJob.getStartNuc(), _myJob.getStartNucSize() );
  memcpy( _HostConstP->deltaFrames, _myJob.getDeltaFrames(), _myJob.getDeltaFramesSize() );
  memcpy( _HostConstP->frameNumber, _myJob.getFrameNumber(), _myJob.getFrameNumberSize() );
  memcpy( _HostConstP->flowIdxMap, _myJob.getFlowIdxMap(), _myJob.getFlowIdxMapSize());  
  _HostConstP->useDarkMatterPCA = _myJob.useDarkMatterPCA();
//  cout << getLogHeader() << " useDarkMatterPCA: " << _myJob.useDarkMatterPCA() << endl;

if(_myJob.performCrossTalkCorrection()) {
  // copy neighbour map for xtalk
    _HostConstXtalkP->neis = _myJob.getNumXtalkNeighbours();
    memcpy( _HostConstXtalkP->multiplier, _myJob.getXtalkNeiMultiplier(),sizeof(float)*_myJob.getNumXtalkNeighbours());
    memcpy( _HostConstXtalkP->tau_top, _myJob.getXtalkNeiTauTop(),sizeof(float)*_myJob.getNumXtalkNeighbours());
    memcpy( _HostConstXtalkP->tau_fluid, _myJob.getXtalkNeiTauFluid(),sizeof(float)*_myJob.getNumXtalkNeighbours());
  
    memcpy(_HostNeiIdxMap, const_cast<int*>(_myJob.getNeiIdxMapForXtalk()), 
                          sizeof(int)*_myJob.getNumBeads()*_myJob.getNumXtalkNeighbours());

  /*std::cout << "====> Beads: " << _myJob.getNumBeads() << std::endl;
  for (int i=0; i<_HostConstXtalkP->neis; ++i) {
    std::cout << "cx=" << _HostConstXtalkP->cx[i];
    std::cout << "cy=" << _HostConstXtalkP->cy[i];
    std::cout << "multiplier=" << _HostConstXtalkP->multiplier[i];
    std::cout << "tau_top=" << _HostConstXtalkP->tau_top[i];
    std::cout << "tau_fluid=" << _HostConstXtalkP->tau_fluid[i];
    std::cout << std::endl;
   
    std::cout << _HostNeiIdxMap[i*_myJob.getNumBeads()] << " ";
    std::cout << std::endl;
  }*/
  }
}



//////////////////////////
// ASYNC CUDA FUNCTIONS, KERNEL EXECUTION AND DATA HANDLING


void SimpleSingleFitStream::prepareInputs()
{
  //prepare environment for new job
  
  preProcessCpuSteps();
  resetPointers();
  serializeInputs();      
}

void SimpleSingleFitStream::copyToDevice()
{
 // move data to device
  if(_verbose) cout << getLogHeader() << " Async Copy To Device" << endl;
  //cout << getId() << " bead input size " << _copyInSize/(1024.0*1024.0) << "MB, output size " << _copyOutSize/(1024.0*1024.0) << "MB"<< endl;
  //cout << getId() << " fgBuffer input size " << ( sizeof(FG_BUFFER_TYPE)*_N*_F*NUMFB)/(1024.0*1024.0) << "MB"<< endl
  
//  copySingleFlowFitConstParamAsync(_HostConstP, getStreamId() ,_stream);CUDA_ERROR_CHECK();
  copyFittingConstParamAsync(_HostConstP, getStreamId() ,_stream);CUDA_ERROR_CHECK();

  cudaMemcpyAsync( _DevWeightFgBuffer, _HostFgBuffer, _myJob.getFgBufferSizeShort(), cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();

  cudaMemcpyAsync( _d_pBeadParams, _h_pBeadParams, _copyInSize, cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();

 //cudaMemcpyToSymbolAsync ( CP, _HostConstP, sizeof(ConstParams), getId()*sizeof(ConstParams),cudaMemcpyHostToDevice, _stream);CUDA_ERROR_CHECK();

  // copy xtalk neighbor map

if(_myJob.performCrossTalkCorrection()) {
      copyXtalkConstParamAsync(_HostConstXtalkP, getStreamId() ,_stream);CUDA_ERROR_CHECK();
      cudaMemcpyAsync(_d_pNeiIdxMap, _HostNeiIdxMap, sizeof(int)*_myJob.getNumBeads()*_myJob.getNumXtalkNeighbours(),
      cudaMemcpyHostToDevice, _stream); CUDA_ERROR_CHECK();
  }

}



void SimpleSingleFitStream::executeKernel()
{
  if(_verbose) cout << getLogHeader() << " Async Kernel Exec" << endl;

  dim3 block(32,32);
  dim3 grid( (_F*NUMFB+ block.x-1)/block.x , (_padN+block.y-1)/block.y);


  
  transposeDataToFloat_Wrapper (grid, block, 0 ,_stream,_DevFgBufferFloat, _d_pFgBuffer, _F*NUMFB, _padN);

  int StructLength = (sizeof(bead_params)/sizeof(float));
  if((sizeof(bead_params)%sizeof(float)) != 0 )
  { 
    cerr << "Structure not a multiple of sizeof(float), transpose not possible" << endl;
    exit(-1);
  }
 
  grid.x = (StructLength + block.x-1)/block.x  ;
  grid.y = (_padN+block.y-1)/block.y;
  transposeData_Wrapper( 
      grid, 
      block, 
      0,
      _stream, 
      (float*)_DevBeadParamTransp, 
      (float*)_d_pBeadParams, 
      StructLength, 
     _padN);
    
  block.x = getBeadsPerBlock();
  block.y = 1;

  grid.y = 1;
  grid.x = (_N+block.x-1)/block.x;

// cross talk correction is perfomed for 3-seris chips only
if (_myJob.performCrossTalkCorrection()) {
  for (int fnum=0; fnum<NUMFB; ++fnum) {
    NeighbourContributionToXtalk_Wrapper(
      grid, 
      block, 
      0, 
      _stream,
      _d_pR, // N
      _d_pShiftedBkg + fnum*_F, // FLxF 
      _DevFgBufferFloat + fnum*_padN*_F, // FLxFxN
      _myJob.getAbsoluteFlowNum(), // starting flow number to calculate absolute flow num
      fnum,
      _N, // 4
      _F, // 4
      //xtalk arguments
      _d_pXtalkScratch,
      _d_pNeiContribution,
      getStreamId());

      XtalkAccumulationAndSignalCorrection_Wrapper(
          grid, 
	  block, 
	  0, 
	  _stream,
	  fnum,
	  _DevFgBufferFloat + fnum*_padN*_F, // FLxFxN
	  _N, // 4
	  _F, // 4
	  _d_pNeiIdxMap,
	  _d_pNeiContribution,
	  _d_pXtalk,
	  _d_pCopies, // N
	  _d_pR, // N
	  _d_pGain, // N
	  _d_pShiftedBkg + fnum*_F,
	  _d_pDarkMatter, // FLxF
          _d_pPCA_Vals,
	  _myJob.getAbsoluteFlowNum(), // starting flow number to calculate absolute flow num
	  getStreamId());
    }
}
else {
  PreSingleFitProcessing_Wrapper( grid, block, 0 , _stream,
  // Here FL stands for flows
  // inputs from data reorganization
  _d_pCopies, // N
  _d_pR, // N
  _d_pGain, // N
  _d_pAmpl, // FLxN
  _d_pShiftedBkg, // FLxF 
  _d_pDarkMatter, // FLxF
  _d_pPCA_Vals,
  _DevFgBufferFloat, // FLxFxN
  // other inputs 
   _myJob.getAbsoluteFlowNum(), // starting flow number to calculate absolute flow num
  _N, // 4
  _F, // 4
  //_myJob.performAlternatingFit(),
  false,
  getStreamId());
}

  // perform projection search for amplitude estimation
  if ((_myJob.getAbsoluteFlowNum() > 19) && _myJob.InitializeAmplitude()) {
    ProjectionSearch_Wrapper(
      grid,
      block,
      0,
      _stream,
      _DevFgBufferFloat,
      _d_pEmphVector,
      _d_pNucRise,
      _d_pCopies,
      _d_fval,
      _myJob.getAbsoluteFlowNum(),
      _N,
      _F,
      getStreamId());
  }

  // perform exponential tail fitting
  if (_myJob.performExpTailFitting()) {

    // only done in first 20 flows
    if (_myJob.getAbsoluteFlowNum() == 0) {
      TaubAdjustForExponentialTailFitting_Wrapper(
          grid, 
	  block, 
	  0, 
	  _stream,
	  _DevFgBufferFloat, // FLxFxN,
	  _d_pAmpl, // FLxN
	  _d_pR, // N
	  _d_avg_trc,
	  _d_fval,
	  _d_tmp_fval,
	  _d_err,
	  _d_jac,
	  _N,
	  _F,
	  _d_pTau_Adj, // output it is a per bead parameter
	  getStreamId());
     }

     ExponentialTailFitting_Wrapper(
         grid, 
         block, 
         0, 
         _stream,
         _d_pTau_Adj,
         _d_pAmpl,
         _d_pR,
         _DevFgBufferFloat,
         _d_pShiftedBkg,
         _d_tmp_fval,
         _N,
         _F,
         _myJob.getAbsoluteFlowNum(),
         getStreamId());
  }

  // perform single flow fitting 
  int sharedMem = _myJob.getEmphVecSize();
  switch(_fittype){
    case 1:
    PerFlowLevMarFit_Wrapper(getL1Setting(), grid, block, sharedMem, _stream,
      // inputs
      _DevFgBufferFloat, 
      _d_pEmphVector,
      _d_pNucRise,
      // bead params
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
      _myJob.getkmultAdj(),
      _myJob.fitkmultAlways(), 
      _myJob.getAbsoluteFlowNum() , // real flow number 
      _myJob.getNumBeads(), // 4
      _myJob.getNumFrames(), // 4
      _myJob.useDynamicEmphasis(),
      //  _DevMonitor, //TODO Monitor
      getStreamId()  // stream id for offset in const memory
    );
    break;
  case 2:
    PerFlowHybridFit_Wrapper(getL1Setting(), grid, block, sharedMem, _stream,
      // inputs
      _DevFgBufferFloat, 
      _d_pEmphVector,
      _d_pNucRise,
      // bead params
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
      _myJob.getkmultAdj(),
      _myJob.fitkmultAlways(), 
      _myJob.getAbsoluteFlowNum() , // real flow number 
      _myJob.getNumBeads(), // 4
      _myJob.getNumFrames(), // 4
      _myJob.useDynamicEmphasis(),
      // _DevMonitor, //TODO Monitor
      getStreamId()  // stream id for offset in const memory
    );
    break;
  case 3:
    PerFlowRelaxKmultGaussNewtonFit_Wrapper(getL1Setting(), grid, block, sharedMem, _stream,
      // inputs
      _DevFgBufferFloat, 
      _d_pEmphVector,
      _d_pNucRise,
      // bead params
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
      _myJob.getkmultAdj(),
      _myJob.fitkmultAlways(), 
      _myJob.getAbsoluteFlowNum() , // real flow number 
      _myJob.getNumBeads(), // 4
      _myJob.getNumFrames(), // 4
      _myJob.useDynamicEmphasis(),
      getStreamId()  // stream id for offset in const memory
    );
    break;
  case 0:
  default: 
    PerFlowGaussNewtonFit_Wrapper(getL1Setting(), grid, block, sharedMem, _stream,
      // inputs
      _DevFgBufferFloat, 
      _d_pEmphVector,
      _d_pNucRise,
      // bead params
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
      _myJob.getkmultAdj(),
      _myJob.fitkmultAlways(), 
      _myJob.getAbsoluteFlowNum() , // real flow number 
      _myJob.getNumBeads(), // 4
      _myJob.getNumFrames(), // 4
      _myJob.useDynamicEmphasis(),
      //    _DevMonitor, //TODO Monitor
      getStreamId()  // stream id for offset in const memory
    );
}
  block.x = 32;
  block.y = 32;

  grid.x = (_padN+block.y-1)/block.y;
  grid.y = (StructLength + block.x-1)/block.x;

  transposeData_Wrapper( 
      grid, 
      block, 
      0,
      _stream,
      (float*)_d_pBeadParams, 
      (float*)_DevBeadParamTransp, 
      _padN,  
      StructLength);
}

void SimpleSingleFitStream::copyToHost()
{
 //cout << getId() <<  " Async copy back" <<endl;
 cudaMemcpyAsync( _h_pBeadParams, _d_pBeadParams, _copyOutSize , cudaMemcpyDeviceToHost, _stream); CUDA_ERROR_CHECK();
 
 //TODO Monitor 
 //cudaMemcpyAsync( _HostMonitor, _DevMonitor, ITER*sizeof(int) , cudaMemcpyDeviceToHost, _stream); 
}


int SimpleSingleFitStream::handleResults()
{
  if(_verbose) cout << getLogHeader() <<  " Handling Results" <<endl;

  if(_myJob.isSet()){
    // for actual pipeline we have to copy the results back into original buffer
    memcpy(_myJob.getBeadParams(), _h_pBeadParams, _myJob.getBeadParamsSize());       
    memcpy(_myJob.getBeadState(), _h_pBeadState, _myJob.getBeadStateSize());       

    // to reuse the same job we have to move it to a temporal buffer:
    //memcpy(_myJob.getResu(), _h_pBeadParams, (sizeof(bead_params)+sizeof(bead_state))*_N);       

    _myJob.setJobToPostFitStep();
    _myJob.putJobToCPU(_item);
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
   

    //TODO Monitor 
    //for(int i=0;i<ITER;i++){
    //   _IterBuffer[i] += _HostMonitor[i];
    //   cout << i << " " << _IterBuffer[i] << endl;
    //}
  }



  return 0;
}


void SimpleSingleFitStream::preProcessCpuSteps() {
  _myJob.setUpFineEmphasisVectors();
}



int SimpleSingleFitStream::getBeadsPerBlock()
{
  if(_bpb < 0){
    return BlockSizeDefaultSetting();
  }
  return _bpb;
}

int SimpleSingleFitStream::getL1Setting()
{
  if(_l1type < 0 || _l1type > 2){
    return l1DefaultSetting();
  }
  return _l1type;
}

void SimpleSingleFitStream::printStatus()
{

	cout << getLogHeader()  << " status: " << endl
	<< " +------------------------------" << endl
	<< " | block size: " << getBeadsPerBlock()  << endl
	<< " | l1 setting: " << getL1Setting() << endl
	<< " | state: " << _state << endl;
	if(_resources->isSet())
		cout << " | streamResource acquired successfully"<< endl;
	else
		cout << " | streamResource not acquired"<< endl;
    _myJob.printJobSummary();
    cout << " +------------------------------" << endl;
}


/////////////////////////////////////////////////////////////////////////
//static Function


void SimpleSingleFitStream::ExecuteJob()
{
  prepareInputs();
  copyToDevice();
  executeKernel();
  copyToHost();
}


size_t SimpleSingleFitStream::getMaxHostMem()
{
  WorkSet Job;

  size_t ret = 0;

if(Job.performCrossTalkCorrection()){
    ret += sizeof(ConstXtalkParams); 
    ret += Job.getXtalkNeiIdxMapSize(true);
}
  ret += sizeof(ConstParams); 
  ret += Job.getFgBufferSizeShort(true);
  ret += Job.getBeadParamsSize(true); 
  ret += Job.getBeadStateSize(true);
  ret += Job.getDarkMatterSize(true); 
  ret += Job.getShiftedBackgroundSize(true);
  ret += Job.getEmphVecSize(true);
  ret += Job.getNucRiseSize(true);

  return ret;

}

size_t SimpleSingleFitStream::getScratchSpaceAllocSize(WorkSet Job)
{
  size_t ScratchSize = 0;
  ScratchSize += 7 * Job.getPaddedN() * Job.getNumFrames();  
  ScratchSize += 1* Job.getPaddedN() * NUMFB;

  if(Job.performCrossTalkCorrection()){
    ScratchSize += MAX_XTALK_NEIGHBOURS* Job.getPaddedN() * Job.getNumFrames();  
  }

  ScratchSize *= sizeof(float);

  return ScratchSize;
}



size_t SimpleSingleFitStream::getMaxDeviceMem(int numFrames, int numBeads)
{

  WorkSet Job;

  // if numFrames/numBeads are passed overwrite the predevined maxFrames/maxBeads
  // for the size calculation
  Job.setMaxFrames(numFrames);
  Job.setMaxBeads(numBeads);

  size_t ret = 0;

  ret = getScratchSpaceAllocSize(Job);

  ret += Job.getFgBufferSizeShort(true);
  ret += Job.getBeadParamsSize(true); 
  ret += Job.getBeadStateSize(true);
  ret += Job.getDarkMatterSize(true); 
  ret += Job.getShiftedBackgroundSize(true);
  ret += Job.getEmphVecSize(true);
  ret += Job.getNucRiseSize(true);

  ret += Job.getFgBufferSizeShort(true);
  ret += Job.getFgBufferSize(true);
  ret += Job.getBeadParamsSize(true); 

  //std::cout << "====> mem for single fit: " << (double)ret/(double)(1024*1024) << std::endl; 
  return ret;
}

void SimpleSingleFitStream::setBeadsPerBLock(int bpb)
{
  _bpb = bpb;
}

void SimpleSingleFitStream::setL1Setting(int type) // 0:sm=l1, 1:sm>l1, 2:sm<l1
{
 if( 0 <= type && type <= 2 ) _l1type = type; 
}

void SimpleSingleFitStream::setHybridIter(int hybridIter)
{
  _hybriditer = hybridIter;
}




void SimpleSingleFitStream::printSettings()
{

  cout << "CUDA SingleFitStream SETTINGS: block size " << _bpb  << " l1setting " << _l1type ;
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
     cout << " GPU specific default" << endl;
  }

}

void SimpleSingleFitStream::setFitType(int type) // 0:gauss newton, 1:lev mar
{ 
 _fittype = type; 
}




