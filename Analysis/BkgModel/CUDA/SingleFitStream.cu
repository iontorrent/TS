/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
// patch for CUDA5.0/GCC4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <iostream>

#include "cuda_error.h"
#include "cuda_runtime.h"

#include "StreamingKernels.h"
#include "SingleFitStream.h"
#include "JobWrapper.h"
#include "GpuMultiFlowFitControl.h"
#include "SignalProcessingFitterQueue.h"


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
  // With recent rearrangements (10-31-13), the magic number for C2075 seems to be 160...
  // And really, I think I've tested it properly...
  if(_computeVersion == 20 ) return 160;
  if(_computeVersion >= 35 ) return 256;
  return 128;
}


/////////////////////////////////////////////////
//FIT STREAM CLASS
SimpleSingleFitStream::SimpleSingleFitStream(streamResources * res, WorkerInfoQueueItem item ) : 
  cudaSimpleStreamExecutionUnit(res, item),
  _myJob( static_cast< BkgModelWorkInfo * >( item.private_data )->flow_key,
          static_cast< BkgModelWorkInfo * >( item.private_data )->inception_state->
              bkg_control.signal_chunks.flow_block_sequence.BlockAtFlow(
                static_cast< BkgModelWorkInfo * >( item.private_data )->flow )->size() )
{


  setName("SingleFitStream");

  if(_verbose) cout << getLogHeader()  << " created " << endl;

  _N = 0;
  _F = 0;
  _padN = 0;

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


  if(!_resource->checkDeviceMemory(getMaxDeviceMem(_myJob.getFlowKey(),_myJob.getFlowBlockSize(), _F,_N )))
    cout << getLogHeader() << " Successfully reallocated device memory to handle Job" << endl;

  _padN = _myJob.getPaddedN();

  if(_verbose) cout << getLogHeader() << " resetting pointers for job with " << _N << "("<< _padN <<") beads and " << _F << " frames" << endl;

  try{

    //HOST DEVICE buffer pairs, Input and Output groups
    _hdFgBuffer = _resource->GetHostDevPair(_myJob.getFgBufferSizeShort(true));

    //fg buffers are copied first to overlap async copy with gathering of other input data
    _resource->StartNewSegGroup();

    _hdBeadParams =  _resource->GetHostDevPair(_myJob.getBeadParamsSize(true));
    _hdBeadState = _resource->GetHostDevPair( _myJob.getBeadStateSize(true));
    //bead Params and State are our outputs. therefore:
    _hdCopyOutGroup = _resource->GetCurrentPairGroup();


    //do not start new group since outputs are also parts of input group
    _hdDarkMatter = _resource->GetHostDevPair(_myJob.getDarkMatterSize(true)); // NUMNUC*F
    _hdShiftedBkg = _resource->GetHostDevPair(_myJob.getShiftedBackgroundSize(true)); // flow_block_size*F
    _hdEmphVector = _resource->GetHostDevPair(_myJob.getEmphVecSize(true)); // (MAX_POISSON_TABLE_COL)*F
    _hdStdTimeCompEmphVec = _resource->GetHostDevPair(_myJob.GetStdTimeCompEmphasisSize(true));
    _hdNucRise = _resource->GetHostDevPair(_myJob.getNucRiseSize(true));  // ISIG_SUB_STEPS_SINGLE_FLOW * F * flow_block_size
    _hdStdTimeCompNucRise = _resource->GetHostDevPair(_myJob.GetStdTimeCompNucRiseSize(true));  // ISIG_SUB_STEPS_SINGLE_FLOW * F * flow_block_size
    //all inputs are grouped now
    _hdCopyInGroup = _resource->GetCurrentPairGroup();


    //Device Only Memory Segments
    _dFgBufferFloat = _resource->getDevSegment(_myJob.getFgBufferSize(true));
    _dWorkBase = _resource->getDevSegment(getScratchSpaceAllocSize(_myJob)   );
    _dBeadParamTransp = _resource->getDevSegment(_myJob.getBeadParamsSize(true));

    //std::cout << "Memory used: " << _resource->getDevMem()->memoryUsed() << std::endl;
    //additional host pointers for Constant memory init
    _hConstP  = _resource->getHostSegment(sizeof(ConstParams));
    if(_myJob.performCrossTalkCorrection()){
      _hConstXtalkP  = _resource->getHostSegment(sizeof(ConstXtalkParams));
      _hNeiIdxMap = _resource->getHostSegment(_myJob.getXtalkNeiIdxMapSize(true));
    }


    //Reuse buffers on the device for other stuff ot create pointers to repacked data

    // We'll use this BeadParams as a reference to check against.
    // If someone tries to rearrange the data structures in BeadParams, we should complain.
    // Someday, we ought to access these chunks of data dynamically, and not be dependent on
    // BeadParams internals.
    // Checking for positive differences (>0) ensures that the fields are in the right order,
    // whatever size they happen to be.
    BeadParams dummy;

    //dev pointer after transpose (Structure of Arrays)
    size_t padNB = _padN*sizeof(float);
    _dCopies = _dBeadParamTransp;  //N

    assert( & dummy.R - & dummy.Copies == 1 );
    _dR = _dCopies.splitAt(padNB); // N

    assert( & dummy.dmult - & dummy.R == 1 );
    _dDmult = _dR.splitAt(padNB); // N

    assert( & dummy.gain - & dummy.dmult == 1 );
    _dGain = _dDmult.splitAt(padNB); // N

    assert( dummy.Ampl - & dummy.gain == 1 );
    _dAmpl = _dGain.splitAt(padNB); // N * flow_block_size

    assert( dummy.kmult - dummy.Ampl > 0 );
    _dKmult = _dAmpl.splitAt(padNB*(dummy.kmult - dummy.Ampl)); // N * flow_block_size

    assert( dummy.pca_vals - dummy.kmult > 0 );
    _dPCA_Vals = _dKmult.splitAt(padNB*(dummy.pca_vals - dummy.kmult)); // N*NUM_DM_PCA

    assert( & dummy.tau_adj - dummy.pca_vals == NUM_DM_PCA );
    _dTau_Adj = _dPCA_Vals.splitAt(padNB*NUM_DM_PCA); // N

    assert( & dummy.phi - & dummy.tau_adj == 1 );
    _dPhi = _dTau_Adj.splitAt(padNB); // N

    _dPhi.checkSize(padNB); // N

    //device scratch space pointers
    _davg_trc = _dWorkBase; // NxF
    _derr = _davg_trc.splitAt(padNB*_F); // NxF
    _dfval = _derr.splitAt(padNB*_F); // NxF
    _dtmp_fval = _dfval.splitAt(padNB*_F); // NxF
    _djac = _dtmp_fval.splitAt(padNB*_F); // 3*NxF Can be reduced in Taubadjust kernel
    _dMeanErr = _djac.splitAt(3*padNB *_F); // N * flow_block_size
    _dMeanErr.checkSize(padNB*_myJob.getFlowBlockSize());

    // xtalk scratch space pointers
    if(_myJob.performCrossTalkCorrection()){
      _dNeiContribution = _dWorkBase;
      _dXtalk = _dNeiContribution.splitAt(padNB *_myJob.getNumXtalkNeighbours()*_F);
      _dXtalkScratch = _dXtalk.splitAt(padNB*_F);
      _dNeiIdxMap = _dXtalkScratch.splitAt(padNB * 3*_F);
      _dNeiIdxMap.checkSize(padNB*_myJob.getNumXtalkNeighbours());
    }

  }
  catch(cudaException &e)
  {
    e.Print();
    cout << getLogHeader() << "Encountered Error during Resource Acquisition!" << endl;
    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
  }

  if(_verbose)cout << getLogHeader() << " " <<  _resource->Status() << endl;

}


void SimpleSingleFitStream::serializeInputs()
{

  if(_verbose) cout <<  getLogHeader() <<" serialize data for async global mem copy" << endl;
  try{
    _hdFgBuffer.copyIn(_myJob.getFgBuffer() ,_myJob.getFgBufferSizeShort());

    _hdBeadParams.copyIn(_myJob.getBeadParams(),_myJob.getBeadParamsSize());
    _hdBeadState.copyIn(_myJob.getBeadState(),_myJob.getBeadStateSize());
    _hdDarkMatter.copyIn(_myJob.getDarkMatter(), _myJob.getDarkMatterSize());
    _hdShiftedBkg.copyIn(_myJob.getShiftedBackground(), _myJob.getShiftedBackgroundSize());
    _hdEmphVector.copyIn(_myJob.getEmphVec(), _myJob.getEmphVecSize());
    _hdNucRise.copyIn(_myJob.getCalculateNucRise(), _myJob.getNucRiseSize());


    // a little hacky but we want to fill the structure in page locked memory with data
    ConstParams* tmpConstP = _hConstP.getPtr();
    //init the reg_param part (all we need from the reg params is non-dynamic)
    reg_params* tmpConstPCastToReg = (reg_params*)tmpConstP;
    *(tmpConstPCastToReg) = *(_myJob.getRegionParams()); // use the
    // init the rest of the ConstParam buffers
    memcpy( tmpConstP->start, _myJob.getStartNuc(), _myJob.getStartNucSize() );
    memcpy( tmpConstP->deltaFrames, _myJob.getDeltaFrames(), _myJob.getDeltaFramesSize() );
    memcpy( tmpConstP->frameNumber, _myJob.getFrameNumber(), _myJob.getFrameNumberSize() );
    memcpy( tmpConstP->flowIdxMap, _myJob.getFlowIdxMap(), _myJob.getFlowIdxMapSize());
    memcpy(tmpConstP->non_zero_emphasis_frames, _myJob.GetNonZeroEmphasisFrames(), 
        _myJob.GetNonZeroEmphasisFramesVecSize());
    tmpConstP->useDarkMatterPCA = _myJob.useDarkMatterPCA();
    tmpConstP->useRecompressTailRawTrace = (_myJob.performRecompressionTailRawTrace() && 
        _myJob.performExpTailFitting());

        // for recompressing traces
    if (_myJob.performExpTailFitting() && _myJob.performRecompressionTailRawTrace()) {
      _hdStdTimeCompEmphVec.copyIn(_myJob.GetStdTimeCompEmphasis(), _myJob.GetStdTimeCompEmphasisSize());
      _hdStdTimeCompNucRise.copyIn(_myJob.GetStdTimeCompNucRise(), _myJob.GetStdTimeCompNucRiseSize());
      memcpy(tmpConstP->std_frames_per_point, _myJob.GetStdFramesPerPoint(), _myJob.GetStdFramesPerPointSize());
      memcpy(tmpConstP->etf_interpolate_frame, _myJob.GetETFInterpolationFrames(), 
          _myJob.GetETFInterpolationFrameSize());
      memcpy(tmpConstP->etf_interpolateMul, _myJob.GetETFInterpolationMul(), 
          _myJob.GetETFInterpolationMulSize());
      memcpy(tmpConstP->deltaFrames_std, _myJob.GetStdTimeCompDeltaFrame(), 
          _myJob.GetStdTimeCompDeltaFrameSize());
      memcpy(tmpConstP->std_non_zero_emphasis_frames, 
          _myJob.GetNonZeroEmphasisFramesForStdCompression(), 
          _myJob.GetNonZeroEmphasisFramesVecSize());
    }

    if(_myJob.performCrossTalkCorrection()) {
      // copy neighbor map for xtalk
      ConstXtalkParams *tmpConstXtalkP = _hConstXtalkP.getPtr();
      tmpConstXtalkP->neis = _myJob.getNumXtalkNeighbours();
      memcpy( tmpConstXtalkP->multiplier, _myJob.getXtalkNeiMultiplier(),sizeof(float)*_myJob.getNumXtalkNeighbours());
      memcpy( tmpConstXtalkP->tau_top, _myJob.getXtalkNeiTauTop(),sizeof(float)*_myJob.getNumXtalkNeighbours());
      memcpy( tmpConstXtalkP->tau_fluid, _myJob.getXtalkNeiTauFluid(),sizeof(float)*_myJob.getNumXtalkNeighbours());

      _hNeiIdxMap.copyIn(const_cast<int*>(_myJob.getNeiIdxMapForXtalk()),
          sizeof(int)*_myJob.getNumBeads()*_myJob.getNumXtalkNeighbours());
    }
  }
  catch(cudaException &e)
  {
    cout << getLogHeader() << "Encountered Error during Input Serialization!" << endl;
    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
  }

}



//////////////////////////
// IMPLEMENTATION OF THE VIRTUAL INTERFACE
// ASYNC CUDA FUNCTIONS, KERNEL EXECUTION AND DATA HANDLING


bool SimpleSingleFitStream::InitJob() {

    _myJob.setData(static_cast<BkgModelWorkInfo *>(getJobData()));

    return _myJob.ValidJob();
}


void SimpleSingleFitStream::ExecuteJob()
{
  prepareInputs();
  copyToDevice();
  executeKernel();
  copyToHost();
}


int SimpleSingleFitStream::handleResults()
{
  if(_verbose) cout << getLogHeader() <<  " Handling Results" <<endl;

  if(_myJob.isSet()){
    // for actual pipeline we have to copy the results back into original buffer
  try{
    _hdBeadParams.copyOut(_myJob.getBeadParams(), _myJob.getBeadParamsSize());
    _hdBeadState.copyOut(_myJob.getBeadState(),_myJob.getBeadStateSize());

    _myJob.setJobToPostFitStep();
    _myJob.putJobToCPU(_item);
  }

  catch(cudaException &e)
  {
    cout << getLogHeader() << "Encountered Error during Result Handling!" << endl;
    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
  }

  }

  return 0;
}



void SimpleSingleFitStream::printStatus()
{


  cout << getLogHeader()  << " status: " << endl
  << " +------------------------------" << endl
  << " | block size: " << getBeadsPerBlock()  << endl
  << " | l1 setting: " << getL1Setting() << endl
  << " | state: " << _state << endl;
  if(_resource->isSet())
    cout << " | streamResource acquired successfully"<< endl;
  else
    cout << " | streamResource not acquired"<< endl;
    _myJob.printJobSummary();
    cout << " +------------------------------" << endl;
}



///////////////////////////////////////////////////////////////



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

  try{

    StreamingKernels::copyFittingConstParamAsync(_hConstP.getPtr(), getStreamId() ,_stream);CUDA_ERROR_CHECK();

    _hdFgBuffer.copyToDeviceAsync(_stream, _myJob.getFgBufferSizeShort());
    _hdCopyInGroup.copyToDeviceAsync(_stream);

    // copy xtalk neighbor map
    if(_myJob.performCrossTalkCorrection()) {
      StreamingKernels::copyXtalkConstParamAsync(_hConstXtalkP.getPtr(), getStreamId() ,_stream);CUDA_ERROR_CHECK();
      _dNeiIdxMap.copyAsync(_hNeiIdxMap, _stream, sizeof(int)*_myJob.getNumBeads()*_myJob.getNumXtalkNeighbours());
    }

  }
  catch(cudaException &e)
  {
    cout << getLogHeader() << "Encountered Error during Copy to device!" << endl;
    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
  }

}



void SimpleSingleFitStream::executeKernel()
{
  if(_verbose) cout << getLogHeader() << " Async Kernel Exec" << endl;

  dim3 block(32,32);
  dim3 grid( (_F*_myJob.getFlowBlockSize()+ block.x-1)/block.x , (_padN+block.y-1)/block.y);



  StreamingKernels::transposeDataToFloat (grid, block, 0 ,_stream,_dFgBufferFloat.getPtr(), _hdFgBuffer.getPtr(), _F*_myJob.getFlowBlockSize(), _padN);

  int StructLength = (sizeof(BeadParams)/sizeof(float));
  if((sizeof(BeadParams)%sizeof(float)) != 0 )
  { 
    cerr << "Structure not a multiple of sizeof(float), transpose not possible" << endl;
    exit(-1);
  }

  grid.x = (StructLength + block.x-1)/block.x  ;
  grid.y = (_padN+block.y-1)/block.y;
  StreamingKernels::transposeData( 
      grid, 
      block, 
      0,
      _stream, 
      (float*)_dBeadParamTransp.getPtr(),
      (float*)_hdBeadParams.getPtr(),
      StructLength, 
      _padN);

  block.x = getBeadsPerBlock();
  block.y = 1;

  grid.y = 1;
  grid.x = (_N+block.x-1)/block.x;


  // cross talk correction is performed for 3-series chips only
  if (_myJob.performCrossTalkCorrection()) {
    for (int fnum=0; fnum<_myJob.getFlowBlockSize(); ++fnum) {
      StreamingKernels::NeighbourContributionToXtalk(
        grid, 
        block, 
        0, 
        _stream,
        _dR.getPtr(), // N
        _dCopies.getPtr(), // N
        _dPhi.getPtr(), // N
        (float*)_hdShiftedBkg.getPtr() + fnum*_F, // FLxF
        (float*)_dFgBufferFloat.getPtr() + fnum*_padN*_F, // FLxFxN
        _myJob.getAbsoluteFlowNum(), // starting flow number to calculate absolute flow num
        fnum,
        _N, // 4
        _F, // 4
        //xtalk arguments
        _dXtalkScratch.getPtr(),
        _dNeiContribution.getPtr(),
        getStreamId());
  
        StreamingKernels::XtalkAccumulationAndSignalCorrection(
            grid, 
            block, 
            0, 
            _stream,
            fnum,
            (float*)_dFgBufferFloat.getPtr() + fnum*_padN*_F, // FLxFxN
            _N, // 4
            _F, // 4
            _dNeiIdxMap.getPtr(),
            _dNeiContribution.getPtr(),
            _dXtalk.getPtr(),
            _dCopies.getPtr(), // N
            _dR.getPtr(), // N
            _dPhi.getPtr(), // N
            _dGain.getPtr(), // N
            (float*)_hdShiftedBkg.getPtr() + fnum*_F,
            _hdDarkMatter.getPtr(), // FLxF
            _dPCA_Vals.getPtr(),
            _myJob.getAbsoluteFlowNum(), // starting flow number to calculate absolute flow num
            getStreamId());
      }
  }
  else {
    StreamingKernels::PreSingleFitProcessing( grid, block, 0 , _stream,
      // Here FL stands for flows
      // inputs from data reorganization
      _dCopies.getPtr(), // N
      _dR.getPtr(), // N
      _dPhi.getPtr(), // N
      _dGain.getPtr(), // N
      _dAmpl.getPtr(), // FLxN
      _hdShiftedBkg.getPtr(), // FLxF
      _hdDarkMatter.getPtr(), // FLxF
      _dPCA_Vals.getPtr(),
      _dFgBufferFloat.getPtr(), // FLxFxN
      // other inputs 
      _myJob.getAbsoluteFlowNum(), // starting flow number to calculate absolute flow num
      _N, // 4
      _F, // 4
      //_myJob.performAlternatingFit(),
      false,
      getStreamId(),
      _myJob.getFlowBlockSize());
  }

  // perform projection search for amplitude estimation
  if ((_myJob.getAbsoluteFlowNum() > 19) && _myJob.InitializeAmplitude()) {
    StreamingKernels::ProjectionSearch(
      grid,
      block,
      0,
      _stream,
      _hdBeadState.getPtr(),
      _dFgBufferFloat.getPtr(),
      _hdEmphVector.getPtr(),
      _hdNucRise.getPtr(),
      _dCopies.getPtr(),
      _dfval.getPtr(),
      _myJob.getAbsoluteFlowNum(),
      _N,
      _F,
      getStreamId(),
      _myJob.getFlowBlockSize());
  }

  // perform exponential tail fitting
  if (_myJob.performExpTailFitting()) {

    // only done in first 20 flows
    if (_myJob.getAbsoluteFlowNum() == 0) {
      StreamingKernels::TaubAdjustForExponentialTailFitting(
          grid, 
          block, 
          0, 
          _stream,
          _hdBeadState.getPtr(),
          _dFgBufferFloat.getPtr(), // FLxFxN,
          _dAmpl.getPtr(), // FLxN
          _dR.getPtr(), // N
          _dCopies.getPtr(), // N
          _dPhi.getPtr(), // N
          _davg_trc.getPtr(),
          _dfval.getPtr(),
          _dtmp_fval.getPtr(),
          _derr.getPtr(),
          _djac.getPtr(),
          _N,
          _F,
          _dTau_Adj.getPtr(), // output it is a per bead parameter
          getStreamId(),
          _myJob.getFlowBlockSize());
     }

     StreamingKernels::ExponentialTailFitting(
         grid, 
         block, 
         0, 
         _stream,
         _hdBeadState.getPtr(),
         _dTau_Adj.getPtr(),
         _dAmpl.getPtr(),
         _dR.getPtr(),
         _dCopies.getPtr(),
         _dPhi.getPtr(), // N
         _dFgBufferFloat.getPtr(),
         _hdShiftedBkg.getPtr(),
         _dtmp_fval.getPtr(),
         _N,
         _F,
         _myJob.getAbsoluteFlowNum(),
         getStreamId(),
         _myJob.getFlowBlockSize());

    if (_myJob.performRecompressionTailRawTrace())
      StreamingKernels::RecompressRawTracesForSingleFlowFit(
          grid, 
          block, 
          0, 
          _stream,
          _dFgBufferFloat.getPtr(), 
          _dtmp_fval.getPtr(),
          _myJob.GetETFStartFrame(),
          _F, // exponential tail fit compressed frames
          _myJob.GetNumStdCompressedFrames(),
          _myJob.getFlowBlockSize(),
          _N,
          getStreamId());
 
  }

  // decide some data buffers based on whether tail need to be recompressed 
  // or not. Need to refactor
  int sharedMem = _myJob.getEmphVecSize();
  float* dEmpVec = _hdEmphVector.getPtr();
  float* dNucRise = _hdNucRise.getPtr();
  int numFrames = _F;
  if (_myJob.performExpTailFitting() && _myJob.performRecompressionTailRawTrace()) {
    sharedMem = _myJob.GetStdTimeCompEmphasisSize();
    dEmpVec = _hdStdTimeCompEmphVec.getPtr();
    dNucRise = _hdStdTimeCompNucRise.getPtr();
    numFrames = _myJob.GetNumStdCompressedFrames();
  }

  //std::cout << "====================> Numframes: " << numFrames << std::endl;
  // perform single flow fitting 
  switch(_fittype){
  case 1:
    StreamingKernels::PerFlowLevMarFit(getL1Setting(), grid, block, sharedMem, _stream,
      // inputs
      _dFgBufferFloat.getPtr(),
      dEmpVec,
      dNucRise,
      // bead params
      _dCopies.getPtr(),
      _hdBeadState.getPtr(),
      // scratch space in global memory
      _derr.getPtr(), //
      _dfval.getPtr(), // NxF
      _dtmp_fval.getPtr(), // NxF
      _dMeanErr.getPtr(),
      // other inputs
      _myJob.getAmpLowLimit(),
      _myJob.getkmultHighLimit(),
      _myJob.getkmultLowLimit(),
      _myJob.getkmultAdj(),
      _myJob.fitkmultAlways(), 
      _myJob.getAbsoluteFlowNum() , // real flow number 
      _myJob.getNumBeads(), // 4
      numFrames,
      _myJob.useDynamicEmphasis(),
      getStreamId(), // stream id for offset in const memory
      _myJob.getFlowBlockSize()
    );
    break;
  case 2:
    StreamingKernels::PerFlowHybridFit(getL1Setting(), grid, block, sharedMem, _stream,
      // inputs
      _dFgBufferFloat.getPtr(),
      dEmpVec,
      dNucRise,
      // bead params
      _dCopies.getPtr(),
      _hdBeadState.getPtr(),
      // scratch space in global memory
      _derr.getPtr(), //
      _dfval.getPtr(), // NxF
      _dtmp_fval.getPtr(), // NxF
      _dMeanErr.getPtr(),
      // other inputs
      _myJob.getAmpLowLimit(),
      _myJob.getkmultHighLimit(),
      _myJob.getkmultLowLimit(),
      _myJob.getkmultAdj(),
      _myJob.fitkmultAlways(), 
      _myJob.getAbsoluteFlowNum() , // real flow number 
      _myJob.getNumBeads(), // 4
      numFrames,
      _myJob.useDynamicEmphasis(),
      getStreamId(), // stream id for offset in const memory
      3,              // switchToLevMar ???
      _myJob.getFlowBlockSize()
    );
    break;
  case 3:
    StreamingKernels::PerFlowRelaxKmultGaussNewtonFit(getL1Setting(), grid, block, sharedMem, _stream,
      // inputs
      _dFgBufferFloat.getPtr(),
      dEmpVec,
      dNucRise,
      // bead params
      _dCopies.getPtr(),
      _hdBeadState.getPtr(),
      // scratch space in global memory
      _derr.getPtr(), //
      _dfval.getPtr(), // NxF
      _dtmp_fval.getPtr(), // NxF
      _djac.getPtr(), //NxF 
      _dMeanErr.getPtr(),
      // other inputs
      _myJob.getAmpLowLimit(),
      _myJob.getkmultHighLimit(),
      _myJob.getkmultLowLimit(),
      _myJob.getkmultAdj(),
      _myJob.fitkmultAlways(), 
      _myJob.getAbsoluteFlowNum() , // real flow number 
      _myJob.getNumBeads(), // 4
      numFrames,
      _myJob.useDynamicEmphasis(),
      getStreamId(),  // stream id for offset in const memory
      _myJob.getFlowBlockSize()
    );
    break;
  case 0:
  default: 
    StreamingKernels::PerFlowGaussNewtonFit(getL1Setting(), grid, block, sharedMem, _stream,
      // inputs
      _dFgBufferFloat.getPtr(),
      dEmpVec,
      dNucRise,
      // bead params
      _dCopies.getPtr(),
      _hdBeadState.getPtr(),
      // scratch space in global memory
      _derr.getPtr(), //
      _dfval.getPtr(), // NxF
      _dtmp_fval.getPtr(), // NxF
      _dMeanErr.getPtr(),
      // other inputs
      _myJob.getAmpLowLimit(),
      _myJob.getkmultHighLimit(),
      _myJob.getkmultLowLimit(),
      _myJob.getkmultAdj(),
      _myJob.fitkmultAlways(), 
      _myJob.getAbsoluteFlowNum() , // real flow number 
      _myJob.getNumBeads(), // 4
      numFrames,
      _myJob.useDynamicEmphasis(),
      getStreamId(),  // stream id for offset in const memory
      _myJob.getFlowBlockSize()
    );
  }
  block.x = 32;
  block.y = 32;

  grid.x = (_padN+block.y-1)/block.y;
  grid.y = (StructLength + block.x-1)/block.x;

  StreamingKernels::transposeData( 
      grid, 
      block, 
      0,
      _stream,
      (float*)_hdBeadParams.getPtr(),
      (float*)_dBeadParamTransp.getPtr(),
      _padN,  
      StructLength);
}

void SimpleSingleFitStream::copyToHost()
{
  //cout << getId() <<  " Async copy back" <<endl;
  //cudaMemcpyAsync( _h_pBeadParams, _d_pBeadParams, _copyOutSize , cudaMemcpyDeviceToHost, _stream); CUDA_ERROR_CHECK();
  _hdCopyOutGroup.copyToHostAsync(_stream);
#if 0
  // To use this, you'll need to tweak JobWrapper.h to make BkgModelWorkInfo * _info public.
  cudaMemcpy( _h_pBeadParams, _d_pBeadParams, _copyOutSize , cudaMemcpyDeviceToHost); CUDA_ERROR_CHECK();
  ostringstream name;
  name << "dumpFile_" << getpid() << "_" << _myJob._info->bkgObj->region_data->region->index;
  ofstream out( name.str().c_str() );
  out << "N " << _N << "\n";
  out << "F " << _F << "\n";
  out << "padN " << _padN << "\n";
  out << "copyInSize " << _copyInSize << "\n";
  out << "copyOutSize " << _copyOutSize << "\n";
  out << "host state array: " << _h_pBeadState << "\n";
  out << "device state array: " << _d_pBeadState << "\n";

  // We've got N BeadParams...
  for( size_t i = 0 ; i < _N ; ++i ) {
    BeadParams &bp = _h_pBeadParams[i];
    out << i << ":\n";
    out << "  Copies " << bp.Copies << "\n";
    out << "  R " << bp.R << "\n";
    out << "  dmult " << bp.dmult << "\n";  
    out << "  gain " << bp.gain << "\n";
    out << "  Ampl, kmult " << "\n";
    for( size_t j = 0 ; j < _myJob.getFlowBlockSize() ; ++j )
    {
      out << "    " << j << ": " << bp.Ampl[j] << " " << bp.kmult[j] << "\n";
    }
    out << "  pca_vals " << "\n";
    for( size_t j = 0 ; j < NUM_DM_PCA ; ++j )
    {
      out << "    " << j << ": " << bp.pca_vals[j] << "\n";
    }
    out << "  tau_adj " << bp.tau_adj << "\n";
    //out << "  my_state (ptr) " << bp.my_state << "\n";
    out << "  trace_ndx " << bp.trace_ndx << "\n";
    out << "  x " << bp.x << "\n";
    out << "  y " << bp.y << "\n";
  }

  for( size_t i = 0 ; i < _N ; ++i ) {
    bead_state & bs = _h_pBeadState[i];
    out << "state " << i << ": " << "\n";
    out << "  bad_read " << bs.bad_read << "\n";
    out << "  clonal_read " << bs.clonal_read << "\n";
    out << "  corrupt " << bs.corrupt << "\n";
    out << "  pinned " << bs.pinned << "\n";
    out << "  random_samp " << bs.random_samp << "\n";
    out << "  key_norm " << bs.key_norm << "\n";
    out << "  ppf " << bs.ppf << "\n";
    out << "  ssq " << bs.ssq << "\n";
    out << "  avg_err " << bs.avg_err << "\n";
  }
#endif

}








void SimpleSingleFitStream::preProcessCpuSteps() {
  _myJob.setUpFineEmphasisVectors();

  if (_myJob.performExpTailFitting() && _myJob.performRecompressionTailRawTrace())
    _myJob.setUpFineEmphasisVectorsForStdCompression();
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



/////////////////////////////////////////////////////////////////////////
//static Function



void SimpleSingleFitStream::requestResources( int flow_key, int flow_block_size, float deviceFraction)
{
  size_t devAlloc = static_cast<size_t>( deviceFraction *
                        max( getMaxDeviceMem( flow_key, flow_block_size, 0, 0 ),
                             getMaxDeviceMem( 0,        flow_block_size, 0, 0 ) ) );
  size_t hostAlloc = max( getMaxHostMem(flow_key, flow_block_size),
                          getMaxHostMem(0,        flow_block_size) );
  cout << "CUDA SingleFitStream active and resources requested dev = "<< devAlloc/(1024.0*1024) << "MB ("<< (int)(deviceFraction*100)<<"%) host = " << hostAlloc/(1024.0*1024) << "MB" <<endl;
  cudaResourcePool::requestDeviceMemory(devAlloc);
  cudaResourcePool::requestHostMemory(hostAlloc);

}

size_t SimpleSingleFitStream::getMaxHostMem(int flow_key, int flow_block_size)
{
  WorkSet Job( flow_key, flow_block_size );

  size_t ret = 0;

  if(GpuMultiFlowFitControl::doGPUTraceLevelXtalk()){
    ret += Job.padTo128Bytes(sizeof(ConstXtalkParams)); 
    ret += Job.getXtalkNeiIdxMapSize(true);
  }
  ret += Job.padTo128Bytes(sizeof(ConstParams)); 
  ret += Job.getFgBufferSizeShort(true);
  ret += Job.getBeadParamsSize(true); 
  ret += Job.getBeadStateSize(true);
  ret += Job.getDarkMatterSize(true); 
  ret += Job.getShiftedBackgroundSize(true);
  ret += Job.getEmphVecSize(true);
  ret += Job.getNucRiseSize(true);
  ret += Job.GetStdTimeCompEmphasisSize(true);
  ret += Job.GetStdTimeCompNucRiseSize(true);

  return ret;

}

size_t SimpleSingleFitStream::getScratchSpaceAllocSize(const WorkSet & Job)
{
  size_t ScratchSize = 0;
  ScratchSize += 7 * Job.getPaddedN() * Job.getNumFrames();  
  ScratchSize += 1* Job.getPaddedN() * Job.getFlowBlockSize();

  if(GpuMultiFlowFitControl::doGPUTraceLevelXtalk()){
    ScratchSize += MAX_XTALK_NEIGHBOURS* Job.getPaddedN() * Job.getNumFrames();  
  }

  ScratchSize *= sizeof(float);

  return ScratchSize;
}




size_t SimpleSingleFitStream::getMaxDeviceMem( int flow_key, int flow_block_size, int numFrames, int numBeads)
{

  WorkSet Job( flow_key, flow_block_size );

  // if numFrames/numBeads are passed overwrite the predefined maxFrames/maxBeads
  // for the size calculation
  if(numFrames >0) Job.setMaxFrames(numFrames);
  if(numBeads> 0) Job.setMaxBeads(numBeads);

  size_t ret = 0;

  ret = getScratchSpaceAllocSize(Job);

  ret += Job.getFgBufferSizeShort(true);
  ret += Job.getBeadParamsSize(true); 
  ret += Job.getBeadStateSize(true);
  ret += Job.getDarkMatterSize(true); 
  ret += Job.getShiftedBackgroundSize(true);
  ret += Job.getEmphVecSize(true);
  ret += Job.getNucRiseSize(true);
  ret += Job.GetStdTimeCompEmphasisSize(true);
  ret += Job.GetStdTimeCompNucRiseSize(true);

  ret += Job.getFgBufferSizeShort(true);
  ret += Job.getFgBufferSize(true);
  ret += Job.getBeadParamsSize(true); 

  //cout << "getMAxDevice SingleFit N: " << N << "("<< Job.getPaddedN()  <<") F: " << F << " ret: " << ret/(1024.0*1024) << endl;
  //std::cout << "====> mem for single fit: " << ret << " bytes"  << std::endl; 
  return ret;
}

void SimpleSingleFitStream::setBeadsPerBlock(int bpb)
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

  cout << "CUDA SingleFitStream SETTINGS: blocksize = " << _bpb  << " l1setting = " ;
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
    cout << "GPU specific default" << endl;
  }

}

void SimpleSingleFitStream::setFitType(int type) // 0:gauss newton, 1:lev mar
{ 
  _fittype = type;
}




