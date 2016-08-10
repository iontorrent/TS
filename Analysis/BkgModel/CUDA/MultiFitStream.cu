/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>

#include "cuda_error.h"
#include "cuda_runtime.h"

#include "StreamingKernels.h"
#include "MultiFitStream.h"
#include "JobWrapper.h"
#include "GpuMultiFlowFitControl.h"
#include "SignalProcessingFitterQueue.h"

using namespace std;


#define DEBUG_SIZE 0

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

SimpleMultiFitStream::SimpleMultiFitStream(streamResources * res, WorkerInfoQueueItem item ) :
  cudaSimpleStreamExecutionUnit(res, item),
  _myJob( static_cast< BkgModelWorkInfo * >( item.private_data )->flow_key,
          static_cast< BkgModelWorkInfo * >( item.private_data )->inception_state->
              bkg_control.signal_chunks.flow_block_sequence.BlockAtFlow(
                static_cast< BkgModelWorkInfo * >( item.private_data )->flow )->size() )
{
  setName("MultiFitStream");

  if(_verbose) cout << getLogHeader() << " created"  << endl;

  // lambda values for each lev mar iteration
  _lambda_start[0] = SMALL_LAMBDA;
  _lambda_start[1] = LARGER_LAMBDA;


  _clonal_restriction[0] = NO_NONCLONAL_PENALTY;
  _clonal_restriction[1] = FULL_NONCLONAL_PENALTY;


  // calculate clonality restriction
  for (int i=0; i<CUDA_MULTIFLOW_NUM_FIT; ++i)
  {
    CalculateClonalityRestriction(i);
  }

  _fitNum = 0;
  _curFitLevel = 0;

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

  // multiple rounds of lev mar fits
  _fit_training_level[0] = _myJob.getPostKeyFitAllWellsTrainingLevel();
  _fit_training_level[1] = 1;

  // lev mar iterations within each training level
  _fit_iterations[0] = _myJob.getPostKeyFitAllWellsTrainingStep();
  _fit_iterations[1] = HAPPY_ALL_BEADS;

  // fit invariant inputs

  try{


    if(!_resource->checkDeviceMemory( getMaxDeviceMem(_myJob.getFlowKey(),_myJob.getFlowBlockSize(),_myJob.getNumFrames(),_myJob.getNumBeads(),&_myJob)))
      cout << getLogHeader() << " Successfully reallocated device memory to handle Job" << endl;


    _hConstP  = _resource->getHostSegment(sizeof(ConstParams));

    _hdBeadParams = _resource->GetHostDevPair(_myJob.getBeadParamsSize(true));



    _resource->StartNewSegGroup();

    // We reuse the same buffer for both _hdFgBuffer and _dPartialDerivsOutput+_dDelta.
    _hdFgBuffer = _resource->GetHostDevPair( _myJob.getReusedFgBufferPartialDerivsSize(true) );
    _hdCoarseNucRise  = _resource->GetHostDevPair(_myJob.getCoarseNucRiseSize(true));  // ISIG_SUB_STEPS_MULTI_FLOW * F * flow_block_size
    _hdSbg  = _resource->GetHostDevPair(_myJob.getShiftedBackgroundSize(true)); // flow_block_size*F
    _hdEmphasis  = _resource->GetHostDevPair(_myJob.getEmphVecSize(true)); // (MAX_POISSON_TABLE_COL)*F
    _hdNon_integer_penalty  = _resource->GetHostDevPair(_myJob.getClonalCallScaleSize(true));
    _hdDarkMatterComp  = _resource->GetHostDevPair(_myJob.getDarkMatterSize(true)); // NUMNUC*F

    _hdInvariantCopyInGroup = _resource->GetCurrentPairGroup();

    // fit variant inputs


    _resource->StartNewDeviceSegGroup();


    _DevFitData.Steps = _resource->getDevSegment(_myJob.getPartialDerivStepsMaxSize(true));
    _DevFitData.JTJMatrixMapForDotProductComputation = _resource->getDevSegment(_myJob.getJTJMatrixMapMaxSize(true));
    _DevFitData.BeadParamIdxMap = _resource->getDevSegment(_myJob.getBeadParamIdxMapMaxSize(true));
    _DevFitData.LambdaForBeadFit = _resource->getDevSegment(_myJob.getFloatPerBead(true));
    MemSegment FitVariantDataDeviceGroup = _resource->GetCurrentDeviceGroup();


    // fit specific host memory allocations
    for (int i=0; i<CUDA_MULTIFLOW_NUM_FIT; ++i)
    {
      _resource->StartNewHostSegGroup();
      _HostDeviceFitData[i].Steps = _resource->getHostSegment( _myJob.getPartialDerivStepsMaxSize(true));
      _HostDeviceFitData[i].JTJMatrixMapForDotProductComputation = _resource->getHostSegment(_myJob.getJTJMatrixMapMaxSize(true));
      _HostDeviceFitData[i].BeadParamIdxMap = _resource->getHostSegment(_myJob.getBeadParamIdxMapMaxSize(true));
      _HostDeviceFitData[i].LambdaForBeadFit = _resource->getHostSegment(_myJob.getFloatPerBead(true));
      MemSegment FitVariantDataHostGroup = _resource->GetCurrentHostGroup();
      //create copy pair for each fitting.
      _HostDeviceFitData[i].hdCopyGroup = MemSegPair(FitVariantDataHostGroup, FitVariantDataDeviceGroup);

    }

    // Device work/scratch buffer:

    _dBeadParamsEval = _resource->getDevSegment(_myJob.getBeadParamsSize(true));
    _dBeadParamsTranspose = _resource->getDevSegment(_myJob.getBeadParamsSize(true));
    _dFgBufferTransposed = _resource->getDevSegment(_myJob.getFgBufferSize(true));

    // we need a specific struct describing this config for this well fit for GPU
    _dIval = _resource->getDevSegment(_myJob.getFxB(true)); // FLxNxF
    _dScratch_ival = _resource->getDevSegment(_myJob.getFxB(true)); // FLxNxF
    _dResidual = _resource->getDevSegment(_myJob.getFloatPerBead(true)); // FLxNxF


    // lev mar fit matrices
    _dJTJ = _resource->getDevSegment(_myJob.getParamMatrixMaxSize(true) );
    _dLTR = _resource->getDevSegment(_myJob.getParamMatrixMaxSize(true) );
    _dRHS = _resource->getDevSegment(_myJob.getParamRHSMaxSize(true));

    //re-use fgBuffer device segment
    _dPartialDerivsOutput = _hdFgBuffer.getDeviceSegment();
    _dDelta = _dPartialDerivsOutput.splitAt( sizeof(float)*_myJob.getMaxSteps()*_myJob.getPaddedN()*
                                                           _myJob.getNumFrames() );

  }
  catch (cudaException &e)
  {
    cout << getLogHeader() << "Encountered Error during Resource Acquisition!" << endl;
    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
  }

  if(_verbose)cout << getLogHeader() << " " <<  _resource->Status() << endl;

}

void SimpleMultiFitStream::serializeFitInvariantInputs()
{  //inputs

  if(_verbose) cout << getLogHeader() <<" serialize data for fit invariant asnync global mem copy" << endl;

  try{

    _hdFgBuffer.copyIn(_myJob.getFgBuffer(), _myJob.getFgBufferSizeShort());
    _hdBeadParams.copyIn(_myJob.getBeadParams() , _myJob.getBeadParamsSize());
    _hdDarkMatterComp.copyIn(_myJob.getDarkMatter(), _myJob.getDarkMatterSize());
    _hdSbg.copyIn(_myJob.getShiftedBackground(), _myJob.getShiftedBackgroundSize());
    _hdEmphasis.copyIn(_myJob.getEmphVec() , _myJob.getEmphVecSize());
    _hdCoarseNucRise.copyIn(_myJob.getCoarseNucRise() , _myJob.getCoarseNucRiseSize());


    // a little hacky but we want to fill the structure in page locked memory with data
    ConstParams* tmpConstP = _hConstP.getPtr();
    //init the reg_param part (all we need from the reg params is non-dynamic)
    reg_params* tmpConstPCastToReg = (reg_params*)tmpConstP;
    *(tmpConstPCastToReg) = *(_myJob.getRegionParams()); // use the
    // init the rest of the ConstParam buffers
    memcpy( tmpConstP->coarse_nuc_start, _myJob.getCoarseNucStart()  , _myJob.getStartNucSize() );
    memcpy( tmpConstP->deltaFrames, _myJob.getDeltaFrames() , _myJob.getDeltaFramesSize() );
    memcpy( tmpConstP->flowIdxMap, _myJob.getFlowIdxMap() , _myJob.getFlowIdxMapSize());
    memcpy( tmpConstP->non_zero_crude_emphasis_frames, _myJob.GetNonZeroEmphasisFrames(),
        _myJob.GetNonZeroEmphasisFramesVecSize());
    memcpy(&tmpConstP->beadParamsMaxConstraints, _myJob.getBeadParamsMax(), _myJob.getBeadParamsMaxSize());
    memcpy(&tmpConstP->beadParamsMinConstraints, _myJob.getBeadParamsMin(), _myJob.getBeadParamsMinSize());
    tmpConstP->useDarkMatterPCA = _myJob.useDarkMatterPCA();

  }
  catch (cudaException &e)
  {
    cout << getLogHeader() << "Encountered Error during Input Serialization!" << endl;
    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
  }
}

void SimpleMultiFitStream::serializeFitSpecificInputs(int fit_index)
{
  //inputs
  if(_verbose) cout << getLogHeader() <<" serialize data for fit specific asnync global mem copy" << endl;

  try{

    _HostDeviceFitData[fit_index].Steps.copyIn(_myJob.getPartialDerivSteps(fit_index) , _myJob.getPartialDerivStepsSize(fit_index) );
    _HostDeviceFitData[fit_index].JTJMatrixMapForDotProductComputation.copyIn(_myJob.getJTJMatrixMap(fit_index), _myJob.getJTJMatrixMapSize(fit_index));
    _HostDeviceFitData[fit_index].BeadParamIdxMap.copyIn(_myJob.getBeadParamIdxMap(fit_index), _myJob.getBeadParamIdxMapSize(fit_index) );

  }

  catch (cudaException &e)
  {
    cout << getLogHeader() << "Encountered Error during Fit Specific Input Serialization!" << endl;
    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
  }

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

  try{
     //_hdNon_integer_penalty.copyToDeviceAsync(_stream,_myJob.getClonalCallScaleSize());
     //_hdFgBuffer.copyToDeviceAsync(_stream, _myJob.getFgBufferSizeShort());
     //_hdDarkMatterComp.copyToDeviceAsync(_stream, _myJob.getDarkMatterSize());
     //_hdSbg.copyToDeviceAsync(_stream, _myJob.getShiftedBackgroundSize());
     //_hdEmphasis.copyToDeviceAsync( _stream, _myJob.getEmphVecSize());
     //_hdNucRise.copyToDeviceAsync(_stream, _myJob.getNucRiseCoarseSize());
    _hdInvariantCopyInGroup.copyToDeviceAsync(_stream);

    //  copyMultiFlowFitConstParamAsync(_HostConstP, getStreamId(),_stream);CUDA_ERROR_CHECK();
    StreamingKernels::copyFittingConstParamAsync(_hConstP.getPtr(), getStreamId(),_stream);CUDA_ERROR_CHECK();
  }

  catch(cudaException &e)
  {
    cout << getLogHeader() << "Encountered Error during Copy to device!" << endl;
    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
  }

}


void SimpleMultiFitStream::copyFitSpecifcInputsToDevice(int fit_index)
{
  //cout << "Copy data to GPU" << endl;
  if(_verbose) cout << getLogHeader() << " Fit Specific Async Copy To Device" << endl;

  try{
    //_DevFitData.Steps.copyAsync(_HostDeviceFitData[fit_index].Steps, _stream, _myJob.getPartialDerivStepsSize(fit_index));
    //_DevFitData.JTJMatrixMapForDotProductComputation.copyAsync(_HostDeviceFitData[fit_index].JTJMatrixMapForDotProductComputation, _stream, _myJob.getJTJMatrixMapSize(fit_index));
    //_DevFitData.BeadParamIdxMap.copyAsync(_HostDeviceFitData[fit_index].BeadParamIdxMap, _stream, _myJob.getBeadParamIdxMapSize(fit_index));
    //_DevFitData.LambdaForBeadFit.copyAsync(_HostDeviceFitData[fit_index].LambdaForBeadFit,_stream,_myJob.getFloatPerBead());
    _HostDeviceFitData[fit_index].hdCopyGroup.copyToDeviceAsync(_stream);
  }

  catch(cudaException &e)
  {
    cout << getLogHeader() << "Encountered Error during Fit Specific Copy to device!" << endl;
    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
  }
}




void SimpleMultiFitStream::executeTransposeToFloat()
{
  //cout << "TransposeToFloat Kernel" << endl;

  int F = _myJob.getNumFrames();
  int padN = _myJob.getPaddedN();


  dim3 block(32,32);
  dim3 grid( (F*_myJob.getFlowBlockSize()+ block.x-1)/block.x , (padN+block.y-1)/block.y);

  StreamingKernels::transposeDataToFloat(grid, block, 0 ,_stream,_dFgBufferTransposed.getPtr(), _hdFgBuffer.getPtr(), F*_myJob.getFlowBlockSize(), padN);
  CUDA_ERROR_CHECK();
}

void SimpleMultiFitStream::executeTransposeParams()
{

  int padN = _myJob.getPaddedN();

  //cout << "TransposeParams Kernel" << endl;

  dim3 block(32,32);
  int StructLength = (sizeof(BeadParams)/sizeof(float));

  if((sizeof(BeadParams)%sizeof(float)) != 0 )
  {
    cerr << getLogHeader() <<" Structure not a multiple of sizeof(float), transpose not possible" << endl;
    exit(-1);
  }

  dim3 grid((StructLength + block.x-1)/block.x , (padN+block.y-1)/block.y);

   CUDA_ERROR_CHECK();
   StreamingKernels::transposeData(grid, block, 0 ,_stream,_dBeadParamsTranspose.getPtr(), (float*)_hdBeadParams.getPtr(), StructLength, padN);
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

//  int StructLength = (sizeof(BeadParams)/sizeof(float));

  CUDA_ERROR_CHECK();

  //async device to device copy
  _dBeadParamsEval.copyAsync(_dBeadParamsTranspose, _stream, _myJob.getBeadParamsSize(true));

  int sharedMem = _myJob.getEmphVecSize();
  for (int i=0; i<_fit_iterations[fit_index]; ++i) {

    //set scratchspace to 0
    _dJTJ.memSetAsync(0, _stream, _myJob.getParamMatrixMaxSize(true));
    _dRHS.memSetAsync(0, _stream, _myJob.getParamRHSMaxSize(true));

    StreamingKernels::ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow(
                        getL1SettingPartialD(),
			gridPD,
			blockPD,
			sharedMem,
			_stream,
			// inputs
			_myJob.getMaxEmphasis(),
			// weights for frames
			_restrict_clonal[fit_index],
			_dFgBufferTransposed.getPtr(),
			_dIval.getPtr(),
			_dScratch_ival.getPtr(),
			_hdCoarseNucRise.getPtr(),
			_hdSbg.getPtr(),
			_hdEmphasis.getPtr(),
			_hdNon_integer_penalty.getPtr(),
			_hdDarkMatterComp.getPtr(),
			_dBeadParamsTranspose.getPtr(),
			_DevFitData.Steps.getPtr(),
			_DevFitData.JTJMatrixMapForDotProductComputation.getPtr(), // pxp
			_dJTJ.getPtr(),
			_dRHS.getPtr(),
			_myJob.getNumParams(fit_index),
			_myJob.getNumSteps(fit_index),
			N,
			F,
			_dResidual.getPtr(),
			_dPartialDerivsOutput.getPtr(),
			getStreamId(), // stream id for offset in const memory
			_myJob.getFlowBlockSize());

    dim3 block( getBeadsPerBlockMultiFit(), 1);
    dim3 grid( (N+block.x-1)/block.x, 1 );

    StreamingKernels::MultiFlowLevMarFit(
		        getL1SettingMultiFit(),
			grid,
			block,
			sharedMem,
			_stream,
			_myJob.getMaxEmphasis(),
			_restrict_clonal[fit_index],
			_dFgBufferTransposed.getPtr(),
			_dIval.getPtr(),
			_dScratch_ival.getPtr(),
			_hdCoarseNucRise.getPtr(),
			_hdSbg.getPtr(),
			_hdEmphasis.getPtr(),
			_hdNon_integer_penalty.getPtr(),
			_hdDarkMatterComp.getPtr(),
			_dBeadParamsTranspose.getPtr(), // we will be indexing directly into it from the parameter indices provide by CpuStep
			_dBeadParamsEval.getPtr(),
			_DevFitData.LambdaForBeadFit.getPtr(),
			_dJTJ.getPtr(), // jtj matrix
			_dLTR.getPtr(), // lower triangular matrix
			_dRHS.getPtr(), // rhs vector
			_dDelta.getPtr(),
			_DevFitData.BeadParamIdxMap.getPtr(),
			_myJob.getNumParams(fit_index),
			N,
			F,
			_dResidual.getPtr(), // N
			getStreamId(),
			_myJob.getFlowBlockSize());
  }
  ++_curFitLevel;
}


void SimpleMultiFitStream::executeTransposeParamsBack()
{

  //cout << "TransposeParamsBack Kernel" << endl;
  int padN = _myJob.getPaddedN();

  dim3 block(32,32);
  int StructLength = (sizeof(BeadParams)/sizeof(float));

  dim3 grid ((padN+block.y-1)/block.y, (StructLength + block.x-1)/block.x );

  StreamingKernels::transposeData(grid, block, 0 ,_stream, (float*)_hdBeadParams.getPtr(), _dBeadParamsTranspose.getPtr(), padN,  StructLength);
  CUDA_ERROR_CHECK();
}


void SimpleMultiFitStream::copyBeadParamsToDevice()
{
  _hdBeadParams.copyToDeviceAsync(_stream,_myJob.getBeadParamsSize());
}


void SimpleMultiFitStream::copyBeadParamsToHost()
{
  _hdBeadParams.copyToHostAsync(_stream,  _myJob.getBeadParamsSize());
}


int SimpleMultiFitStream::handleResults()
{

  if(_myJob.isSet()){

    if(_verbose) cout <<  getLogHeader() << " Handling Results "<< _fitNum << endl;

    _hdBeadParams.copyOut(_myJob.getBeadParams(), _myJob.getBeadParamsSize());
    _myJob.KeyNormalize();   // temporary call to key normalize till we put it into a GPU kernel

    postFitProcessing();

    // if not last iteratin yet copy bead data back topagelocked mem so device can get updated
    if(_fitNum < CUDA_MULTIFLOW_NUM_FIT){
      _hdBeadParams.copyIn(_myJob.getBeadParams(),_myJob.getBeadParamsSize());
      return 1;  //signal more work to be done;
    }

    _myJob.setJobToRemainRegionFit();
    _myJob.putJobToCPU(_item);
  }

  return 0; //signal Job com[plete
}



void SimpleMultiFitStream::SetUpLambdaArray(int fit_index) {
  for (int i=0; i<_myJob.getNumBeads(); ++i) {
    _HostDeviceFitData[fit_index].LambdaForBeadFit[i] = _lambda_start[fit_index];
  }
}

void SimpleMultiFitStream::ExecuteJob()
{


  //  printInfo(); cout << " i: " <<  _fitNum << " numBeads: " << _myJob.getNumBeads() << " numFrames:" << _myJob.getNumFrames() << endl;


  if(_fitNum == 0 && _curFitLevel == 0){

    preFitCpuSteps();

    resetPointers();

    CalculateNonIntegerPenalty();
    serializeFitInvariantInputs();
    copyFitInvariantInputsToDevice();

    executeTransposeToFloat();
  }

  copyBeadParamsToDevice();
  prepareFitSpecificInputs(_fitNum);
  copyFitSpecifcInputsToDevice(_fitNum);

  executeTransposeParams();
  executeMultiFit(_fitNum);
  executeTransposeParamsBack();

  copyBeadParamsToHost();

}


bool SimpleMultiFitStream::InitJob() {

  _myJob.setData(static_cast<BkgModelWorkInfo *>(getJobData()));

  return _myJob.ValidJob();
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
  const float *clonal_call_scale = _myJob.getClonalCallScale();
  float clonal_call_penalty = _myJob.getClonalCallPenalty();

  for (int i=0; i<MAGIC_CLONAL_CALL_ARRAY_SIZE; ++i)
  {
    _hdNon_integer_penalty[i] = clonal_call_penalty * clonal_call_scale[i];
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

void SimpleMultiFitStream::printStatus()
{


  cout << getLogHeader()  << " status: " << endl
  << " +------------------------------" << endl
  << " | block size MultiFit: " << getBeadsPerBlockMultiFit()  << endl
  << " | l1 setting MultiFit: " << getL1SettingMultiFit() << endl
  << " | block size PartialD: " << getBeadsPerBlockPartialD() << endl
  << " | l1 setting PartialD: " << getL1SettingPartialD() << endl
  << " | state: " << _state << endl;
  if(_resource->isSet())
    cout << " | streamResource acquired successfully"<< endl;
  else
    cout << " | streamResource not acquired"<< endl;
    _myJob.printJobSummary();
    cout << " +------------------------------" << endl;

}




// Static member function


void SimpleMultiFitStream::requestResources(
    int global_max_flow_key,
    int global_max_flow_block_size,
    float deviceFraction
  )
{
  // We need to check values both with key=0 and key=max_key.
  // That way, we cover both extremes.
  size_t devAlloc = static_cast<size_t>( deviceFraction *
                     max( getMaxDeviceMem(global_max_flow_key, global_max_flow_block_size, 0, 0),
                          getMaxDeviceMem(0,                   global_max_flow_block_size, 0, 0) ) );
  size_t hostAlloc = max( getMaxHostMem(global_max_flow_key, global_max_flow_block_size),
                          getMaxHostMem(0,                   global_max_flow_block_size) );

  cout << "CUDA: MultiFitStream active and resources requested dev = "<< devAlloc/(1024.0*1024) << "MB ("<< (int)(deviceFraction*100)<<"%) host = " << hostAlloc/(1024.0*1024) << "MB" << endl;
  cudaResourcePool::requestDeviceMemory(devAlloc);
  cudaResourcePool::requestHostMemory(hostAlloc);


}



size_t SimpleMultiFitStream::getMaxHostMem(int flow_key, int flow_block_size)
{
  WorkSet Job( flow_key, flow_block_size );

  size_t ret = 0;

  ret += sizeof(ConstParams);
  ret += Job.getFgBufferSizeShort(true);

  ret += Job.getBeadParamsSize(true);
  ret += Job.getReusedFgBufferPartialDerivsSize(true);
  ret += Job.getCoarseNucRiseSize(true);
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

size_t SimpleMultiFitStream::getMaxDeviceMem(
    int flow_key,
    int flow_block_size,
    int numFrames,
    int numBeads,
    WorkSet *curJob
  )
{
  // create default job
  WorkSet dummyJob( flow_key, flow_block_size );
  // if numFrames/numBeads are passed overwrite the predevined maxFrames/maxBeads
  // for the size calculation
  if(numFrames >0) dummyJob.setMaxFrames(numFrames);
  if(numBeads> 0) dummyJob.setMaxBeads(numBeads);

  WorkSet *Job = NULL;
  if (curJob)
    Job = curJob;
  else
    Job = &dummyJob;


  size_t ret = 0;

  ret += Job->getBeadParamsSize(true);                  //  _hdBeadParams

  ret += Job->getReusedFgBufferPartialDerivsSize(true); //  _hdFgBuffer
  ret += Job->getCoarseNucRiseSize(true);                     //  _hdNucRise
  ret += Job->getShiftedBackgroundSize(true);           //  _hdSbg
  ret += Job->getEmphVecSize(true);                     //  _hdEmphasis
  ret += Job->getClonalCallScaleSize(true);             //  _hdNon_integer_penalty
  ret += Job->getDarkMatterSize(true);                  //  _hdDarkMatterComp

  ret += Job->getPartialDerivStepsMaxSize(true);        // _DevFitData.Steps
  ret += Job->getJTJMatrixMapMaxSize(true);             // _D...JTJMatrixMapForDotProductComputation
  ret += Job->getBeadParamIdxMapMaxSize(true);          // _DevFitData.BeadParamIdxMap
  ret += Job->getFloatPerBead(true);                    // _DevFitData.LambdaForBeadFit

  ret += Job->getBeadParamsSize(true);                  // _dBeadParamsEval
  ret += Job->getBeadParamsSize(true);                  // _dBeadParamsTranspose
  ret += Job->getFgBufferSize(true);                    // _dFgBufferTransposed

  ret += Job->getFxB(true);                             // _dIval
  ret += Job->getFxB(true);                             // _dcratch_ival
  ret += Job->getFloatPerBead(true);                    // _dResidual
  ret += Job->getParamMatrixMaxSize(true);              // _dJTJ
  ret += Job->getParamMatrixMaxSize(true);              // _dLTR
  ret += Job->getParamRHSMaxSize(true);                 // _dRHS

#if DEBUG_SIZE
  cout << "BP size: " << Job->getBeadParamsSize(true) << endl;
  cout << "Fgbuffer: " << Job->getReusedFgBufferPartialDerivsSize(true) << endl;
  cout << "Coarse rise: " <<  Job->getCoarseNucRiseSize(true) << endl;
  cout << "Shifted bkg: " << Job->getShiftedBackgroundSize(true) << endl;
  cout << "Emp size: " << Job->getEmphVecSize(true) << endl;
  cout << "Clonal scale: " << Job->getClonalCallScaleSize(true) << endl;
  cout << "Dark matter: " << Job->getDarkMatterSize(true) << endl;
  cout << "PartialDerivstepmax: " << Job->getPartialDerivStepsMaxSize(true) << endl;
  cout << "jtjmatrixmapsize: " << Job->getJTJMatrixMapMaxSize(true) << endl;
  cout << "BeadParamIdmap: " << Job->getBeadParamIdxMapMaxSize(true) << endl;
  cout << "Lambda: " << Job->getFloatPerBead(true) << endl;
  cout << "BP size: " << Job->getBeadParamsSize(true) << endl;
  cout << "BP size: " << Job->getBeadParamsSize(true) << endl;
  cout << "Fgbuffer size: " << Job->getFgBufferSize(true) << endl;
  cout << "FxB: " << Job->getFxB(true) << endl;
  cout << "FxB: " << Job->getFxB(true) << endl;
  cout << "Floatperbead: " << Job->getFloatPerBead(true) << endl;
  cout << "MatrixMax: " << Job->getParamMatrixMaxSize(true) << endl;
  cout << "MatrixMax: " << Job->getParamMatrixMaxSize(true) << endl;
  cout << "RHS Max: " << Job->getParamRHSMaxSize(true) << endl;
#endif




  return ret;
}

void SimpleMultiFitStream::setBeadsPerBlockMultiF(int bpb)
{
  _bpb = bpb;
}


void SimpleMultiFitStream::setL1SettingMultiF(int type) // 0:sm=l1, 1:sm>l1, 2:sm<l1
{
  _l1type = type;
}

void SimpleMultiFitStream::setBeadsPerBlockPartialD(int bpb)
{
  _bpbPartialD = bpb;
}

void SimpleMultiFitStream::setL1SettingPartialD(int type) // 0:sm=l1, 1:sm>l1, 2:sm<l1
{
  _l1typePartialD = type;
}

void SimpleMultiFitStream::printSettings()
{

  cout << "CUDA: MultiFitStream SETTINGS: blocksize = " << _bpb << " l1setting = " ;
  switch(_l1type){
    case 0:
      cout << "cudaFuncCachePreferEqual" << endl;;
      break;
    case 1:
      cout << "cudaFuncCachePreferShared" <<endl;
      break;
    case 2:
      cout << "cudaFuncCachePreferL1" << endl;
      break;
    default:
     cout << " GPU specific default" << endl;;
  }
  cout << "CUDA: PartialDerivative SETTINGS: blocksize = " << _bpbPartialD << " l1setting = ";
  switch(_l1typePartialD){
    case 0:
      cout << "cudaFuncCachePreferEqual" << endl;;
      break;
    case 1:
      cout << "cudaFuncCachePreferShared" <<endl;
      break;
    case 2:
      cout << "cudaFuncCachePreferL1" << endl;
      break;
    default:
     cout << "GPU specific default" << endl;
  }

}

void SimpleMultiFitStream::preFitCpuSteps()
{
  _myJob.prepareMultiFlowFitMatrixConfig();
  _myJob.performPreFitStepsForMultiFitStream();
}


void SimpleMultiFitStream::postFitProcessing()
{
  if (_curFitLevel == _fit_training_level[_fitNum]) {
    if (_fitNum == 0 && _myJob.performCalcPCADarkMatter())
    {
      //PCA on CPUi
      _myJob.PerformePCA();
      // update PCA flag
      ConstParams* tmpConstP = _hConstP.getPtr();
      tmpConstP->useDarkMatterPCA = _myJob.useDarkMatterPCA();
      StreamingKernels::copyFittingConstParamAsync(tmpConstP, getStreamId(),_stream);CUDA_ERROR_CHECK();
      //update DarkMatterComp
      _hdDarkMatterComp.copyIn(_myJob.getDarkMatter(),_myJob.getDarkMatterSize());
      _hdDarkMatterComp.copyToDeviceAsync(_stream, _myJob.getDarkMatterSize());
    }

    // go to next fit
    ++_fitNum;

    // reset current training level
    _curFitLevel = 0;
  }
}
