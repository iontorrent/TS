/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

// patch for CUDA5.0/GCC4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <iostream>

#include "JobWrapper.h"
#include "SignalProcessingFitterQueue.h"
#include "GpuMultiFlowFitControl.h"
#include "DarkHalo.h"
using namespace std;


////////////////////////////////////////
// workset (represents one job


WorkSet::WorkSet( int flow_key, int flow_block_size)
{
  setFlow( flow_key, flow_block_size );
   
  _fd[0] = _multiFlowFitControl.GetMatrixConfig("FitWellAmplBuffering");
  _fd[1] = _multiFlowFitControl.GetMatrixConfig("FitWellPostKey");

  _maxFrames = 0; // only set if we don't want to determine the mem sizes for a specific number of frames or no item is set
  _maxBeads = 0; // only set if we don't want to determine the mem sizes for a specific number of beads or no item is set

 _info = NULL;
}

WorkSet::WorkSet(BkgModelWorkInfo * i)
{
  setData( i );

  _fd[0] = _multiFlowFitControl.GetMatrixConfig("FitWellAmplBuffering");
  _fd[1] = _multiFlowFitControl.GetMatrixConfig("FitWellPostKey");

  _maxFrames = 0; // only set if we don't want to determine the mem sizes for a specific number of frames or no item is set
  _maxBeads = 0; // only set if we don't want to determine the mem sizes for a specific number of beads or no item is set
}


WorkSet::~WorkSet()
{
}



void WorkSet::setMaxFrames(int frames)
{
  _maxFrames = frames;
}

int WorkSet::getMaxFrames() const
{
  int maxFrames;
  maxFrames =  (_maxFrames != 0)?(_maxFrames):(GpuMultiFlowFitControl::GetMaxFrames());
  assert(maxFrames);
  return maxFrames;
}
void WorkSet::setMaxBeads(int beads)
{
  _maxBeads = beads;
}

void WorkSet::setFlow(int flow_key, int flow_block_size)
{
  _flow_block_size = flow_block_size;
  _flow_key = flow_key;

  if ( _flow_block_size > MAX_NUM_FLOWS_IN_BLOCK_GPU )
  {
    // I'm sorry, Dave, I can't do that.
    fprintf( stderr, 
      "GPU acceleration requires that the number of flows in a block be less than %d.\n"
      "This limit is set at compile time in MAX_NUM_FLOWS_IN_BLOCK_GPU.\n", 
      MAX_NUM_FLOWS_IN_BLOCK_GPU );
    exit( -1 );
  }

  _multiFlowFitControl.SetFlowParams( flow_key, flow_block_size );

}

int WorkSet::getFlowBlockSize() const
{
  // We shouldn't be asking for this if some code path didn't set it.
  assert( _flow_block_size );
  return _flow_block_size;
}

int WorkSet::getFlowKey() const
{
  return _flow_key;
}

int WorkSet::getMaxBeads() const
{
  int maxBeads = (_maxBeads != 0)?(_maxBeads):(GpuMultiFlowFitControl::GetMaxBeads());
  assert(maxBeads);
  return maxBeads;
}

void WorkSet::setData(BkgModelWorkInfo * i)
{
  _info = i;
  setFlow( i->flow_key, i->inception_state->
              bkg_control.signal_chunks.flow_block_sequence.BlockAtFlow( i->flow )->size() );
}

bool WorkSet::isSet() const
{
 return (_info != NULL)?(true):(false);
}


int WorkSet::getNumBeads() const {

    if(isSet()){            
      return _info->bkgObj->region_data->my_beads.numLBeads; 
    }
  return getMaxBeads();// GpuMultiFlowFitControl::GetMaxBeads();
}


int WorkSet::getNumFrames() const { 

  if(isSet()){
    return _info->bkgObj->region_data->time_c.GetTimeCompressedFrames(); 
  }

  return getMaxFrames();  

}





int WorkSet::getMaxSteps()
{
  return _multiFlowFitControl.GetMaxSteps();
}

int WorkSet::getMaxParams()
{
  return _multiFlowFitControl.GetMaxParamsToFit(); 
}

int WorkSet::getNumSteps(int fit_index)
{
  return _fd[fit_index]->GetNumSteps();
}
int WorkSet::getNumParams(int fit_index)
{
  return _fd[fit_index]->GetNumParamsToFit();
}




int WorkSet::getAbsoluteFlowNum() 
{
  FlowBlockSequence::const_iterator flow_block = 
    _info->inception_state->bkg_control.signal_chunks.flow_block_sequence.
      BlockAtFlow( _info->flow );
  return flow_block->begin();
}

reg_params * WorkSet::getRegionParams() {   return  &_info->bkgObj->region_data->my_regions.rp;  }
BeadTracker * WorkSet::getBeadTracker(){   return &_info->bkgObj->region_data->my_beads; }
BeadParams * WorkSet::getBeadParams(){   return &_info->bkgObj->region_data->my_beads.params_nn[0]; }
bead_state * WorkSet::getBeadState(){   return &_info->bkgObj->region_data->my_beads.all_status[0]; }
float * WorkSet::getEmphVec(){   return &_info->bkgObj->region_data->emphasis_data.emphasis_vector_storage[0]; } 
float * WorkSet::getDarkMatter(){   return &_info->bkgObj->region_data->my_regions.missing_mass.dark_matter_compensator[0]; }
int * WorkSet::getFlowIdxMap(){   return _info->bkgObj->region_data_extras.my_flow->flow_ndx_map; }
FG_BUFFER_TYPE * WorkSet::getFgBuffer(){   return _info->bkgObj->region_data->my_trace.fg_buffers; }
float * WorkSet::getDeltaFrames(){   return &_info->bkgObj->region_data->time_c.deltaFrame[0]; }  
int * WorkSet::getStartNuc(){   return _info->bkgObj->region_data->my_regions.cache_step.i_start_fine_step; }


float * WorkSet::getShiftedBackground(){   
  _info->bkgObj->region_data->my_scratch.FillShiftedBkg (*_info->bkgObj->region_data->emptytrace, 
    _info->bkgObj->region_data->my_regions.rp.tshift, _info->bkgObj->region_data->time_c, true,
    getFlowBlockSize() );
  return _info->bkgObj->region_data->my_scratch.shifted_bkg; 
}

float * WorkSet::getCalculateNucRise(){   
  _info->bkgObj->region_data->my_regions.cache_step.CalculateNucRiseFineStep (&_info->bkgObj->region_data->my_regions.rp, _info->bkgObj->region_data->time_c, *_info->bkgObj->region_data_extras.my_flow); // the same for the whole region because time-shift happens per well
  return _info->bkgObj->region_data->my_regions.cache_step.nuc_rise_fine_step; 
}

float * WorkSet::getCalculateNucRiseCoarse()
{
  _info->bkgObj->region_data->my_regions.cache_step.CalculateNucRiseCoarseStep (&_info->bkgObj->region_data->my_regions.rp, _info->bkgObj->region_data->time_c, *_info->bkgObj->region_data_extras.my_flow);
  return _info->bkgObj->region_data->my_regions.cache_step.nuc_rise_coarse_step;
}

 
void WorkSet::setUpFineEmphasisVectors() {
  _info->bkgObj->region_data->SetFineEmphasisVectors();
}

float WorkSet::getAmpLowLimit() 
{ 
  return _info->bkgObj->getGlobalDefaultsForBkgModel().signal_process_control.AmplLowerLimit;
}

float WorkSet::getkmultLowLimit()
{
  return _info->bkgObj->getGlobalDefaultsForBkgModel().signal_process_control.single_flow_master.kmult_low_limit;
}

float WorkSet::getkmultHighLimit()
{
  return _info->bkgObj->getGlobalDefaultsForBkgModel().signal_process_control.single_flow_master.kmult_hi_limit;
}

float WorkSet::getkmultAdj()
{
  return _info->bkgObj->getGlobalDefaultsForBkgModel().signal_process_control.single_flow_master.krate_adj_limit;
}

bool WorkSet::fitkmultAlways()
{
  return _info->bkgObj->getGlobalDefaultsForBkgModel().signal_process_control.var_kmult_only;
}

float* WorkSet::getClonalCallScale() 
{
  return _info->bkgObj->getGlobalDefaultsForBkgModel().fitter_defaults.clonal_call_scale;
}

float WorkSet::getClonalCallPenalty() 
{
  return _info->bkgObj->getGlobalDefaultsForBkgModel().fitter_defaults.clonal_call_penalty;
}

int * WorkSet::getStartNucCoarse()
{
  return _info->bkgObj->region_data->my_regions.cache_step.i_start_coarse_step ;
}

bool WorkSet::performAlternatingFit()
{
  return _info->bkgObj->getGlobalDefaultsForBkgModel().signal_process_control.fit_alternate;
}

bool WorkSet::performRecompressionTailRawTrace()
{
  return _info->bkgObj->getGlobalDefaultsForBkgModel().signal_process_control.recompress_tail_raw_trace;
}

bound_params * WorkSet::getBeadParamsMax()
{
  return &_info->bkgObj->region_data->my_beads.params_high;
}

 bound_params * WorkSet::getBeadParamsMin()
{
  return &_info->bkgObj->region_data->my_beads.params_low;
}

float WorkSet::getMaxEmphasis()
{
  return _info->bkgObj->region_data->my_beads.max_emphasis;
};

bool WorkSet::useDynamicEmphasis()
{
  return true;
}


CpuStep* WorkSet::getPartialDerivSteps(int fit_index)
{
  return _fd[fit_index]->GetPartialDerivSteps();
}

unsigned int* WorkSet::getJTJMatrixMap(int fit_index)
{
  return _fd[fit_index]->GetJTJMatrixMapForDotProductComputation();
}

unsigned int* WorkSet::getBeadParamIdxMap(int fit_index)
{
  return _fd[fit_index]->GetParamIdxMap();
}


float * WorkSet::getFrameNumber() { 
  return &_info->bkgObj->region_data->time_c.frameNumber[0]; 
}  


//////////////////////////////////////////////////////////////////////////////////////
///SIZES:

//// N

int WorkSet::getBeadParamsSize(bool padded)
{
  int size = sizeof(BeadParams);
  return size*( (!padded)?(getNumBeads()):(getPaddedN()) ); 
}

int WorkSet::getBeadStateSize(bool padded)
{
  int size = sizeof(bead_state);
  return size*( (!padded)?(getNumBeads()):(getPaddedN()) ); 
}

int WorkSet::getFgBufferSize(bool padded)
{
  return getFlxFxB(padded);  
}


int WorkSet::getFgBufferSizeShort(bool padded)
{
  int size = sizeof(FG_BUFFER_TYPE)*getNumFrames()*getFlowBlockSize();
  return size*( (!padded)?(getNumBeads()):(getPaddedN()) ); 
}

int WorkSet::getReusedFgBufferPartialDerivsSize(bool padded)
{
  // We reuse the same buffer for both _hdFgBuffer and _dPartialDerivsOutput + _dDelta.
  // Make sure that it's the size of the bigger of the two.
  size_t dPartialDerivsOutputSize = sizeof(float) * getMaxSteps() * getPaddedN() * getNumFrames();
  size_t dDeltaSize = getParamRHSMaxSize(padded);
  size_t fgBufferSize = getFgBufferSizeShort(padded);

  return max( fgBufferSize, dPartialDerivsOutputSize + dDeltaSize );
}

int WorkSet::getFlxFxB(bool padded)
{
  int size = sizeof(float)*getFlowBlockSize()*getNumFrames();
  return size*( (!padded)?(getNumBeads()):(getPaddedN()) ); 
}

int WorkSet::getFxB(bool padded)
{
  int size = sizeof(float)*getNumFrames();
  return size*( (!padded)?(getNumBeads()):(getPaddedN()) ); 
}

int WorkSet::getFlxB(bool padded)
{
  int size = sizeof(float)*getFlowBlockSize();
  return size*( (!padded)?(getNumBeads()):(getPaddedN()) ); 
}

int WorkSet::getFloatPerBead(bool padded)
{
  int size = sizeof(float);  
  return size*( (!padded)?(getNumBeads()):(getPaddedN()) ); 
}
///// non-N

int WorkSet::getRegionParamsSize(bool padded)
{
  int size = sizeof(reg_params);
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getEmphVecSize(bool padded)
{
  int size = sizeof(float)*(MAX_POISSON_TABLE_COL)*getNumFrames();
  return (!padded)?(size):(padTo128Bytes(size));
}
 
int WorkSet::getDarkMatterSize(bool padded)
{
  int size = sizeof(float)*NUMNUC*getNumFrames();
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getShiftedBackgroundSize(bool padded)
{
  int size = sizeof(float)*getFlowBlockSize()*getNumFrames();
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getFlowIdxMapSize(bool padded)
{
  int size = sizeof(int)*getFlowBlockSize();
  return (!padded)?(size):(padTo128Bytes(size));
}



int WorkSet::getDeltaFramesSize(bool padded)
{
  int size = sizeof(float)*getNumFrames();
  return (!padded)?(size):(padTo128Bytes(size));
}
 
int WorkSet::getNucRiseSize(bool padded)
{
  int size = sizeof(float) * ISIG_SUB_STEPS_SINGLE_FLOW * getNumFrames() * getFlowBlockSize();
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getStartNucSize(bool padded)
{
  int size = sizeof(int)*getFlowBlockSize();
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getNucRiseCoarseSize(bool padded)
{
  int size = sizeof(float) * ISIG_SUB_STEPS_MULTI_FLOW * getNumFrames() * getFlowBlockSize();
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getStartNucCoarseSize(bool padded)
{
  int size = sizeof(int)*getFlowBlockSize();
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getBeadParamsMaxSize(bool padded)
{
  int size = sizeof(bound_params);
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getBeadParamsMinSize(bool padded)
{
  int size = sizeof(bound_params);
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getClonalCallScaleSize(bool padded)
{
  int size = sizeof(float)*MAGIC_CLONAL_CALL_ARRAY_SIZE;
  return (!padded)?(size):(padTo128Bytes(size));
}
 

int WorkSet::getPartialDerivStepsSize(int fit_index, bool padded)
{
  int size = sizeof(CpuStep)*_fd[fit_index]->GetNumSteps();
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getJTJMatrixMapSize(int fit_index, bool padded)
{
  int size = sizeof(unsigned int) * _fd[fit_index]->GetNumParamsToFit()*_fd[fit_index]->GetNumParamsToFit();  
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getBeadParamIdxMapSize(int fit_index, bool padded)
{
  int size = sizeof(unsigned int) * _fd[fit_index]->GetNumParamsToFit();  
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getParamMatrixSize(int fit_index, bool padded)
{
  int size = ((_fd[fit_index]->GetNumParamsToFit()*_fd[fit_index]->GetNumParamsToFit()+ 1)/2)*sizeof(float);
  return size * ((!padded)?(getNumBeads()):(getPaddedN()));
}

int WorkSet::getParamRHSSize(int fit_index, bool padded)
{
  int size = _fd[fit_index]->GetNumParamsToFit() *sizeof(float);
  return size * ((!padded)?(getNumBeads()):(getPaddedN()));
}




int WorkSet::getPartialDerivStepsMaxSize(bool padded)
{
  int size = sizeof(CpuStep)*getMaxSteps();
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getJTJMatrixMapMaxSize(bool padded)
{
  int size = sizeof(unsigned int) * getMaxParams()*getMaxParams();  
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getBeadParamIdxMapMaxSize(bool padded) 
{
  int size = sizeof(unsigned int) * getMaxParams();
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getParamMatrixMaxSize(bool padded)
{
  int size = ((getMaxParams()*(getMaxParams() + 1))/2)*sizeof(float);
  return size * ((!padded)?(getNumBeads()):(getPaddedN()));
}

int WorkSet::getParamRHSMaxSize(bool padded)
{
  int size = getMaxParams() *sizeof(float);
  return size * ((!padded)?(getNumBeads()):(getPaddedN()));
}

int WorkSet::getFrameNumberSize(bool padded)
{
  int size = sizeof(float)*getNumFrames();
  return (!padded)?(size):(padTo128Bytes(size));
}

//////////////////////////////////////////////////////////////////////////////////////
///

int WorkSet::getPaddedN() const{
  return ((getNumBeads()+32-1)/32)*32;
}


int WorkSet::padTo128Bytes(int size){
  return ((size+128-1)/128)*128;
}


bool WorkSet::ValidJob()
{
  
  if (_info->bkgObj->region_data->fitters_applied == -1 || _info == NULL) {
    return false;
  }
  return isSet();

}

void WorkSet::KeyNormalize()
{
  _info->bkgObj->region_data->my_beads.my_mean_copy_count = 
    _info->bkgObj->region_data->my_beads.KeyNormalizeReads(true, false, getFlowBlockSize());
}

void WorkSet::PerformePCA()
{
  FlowBlockSequence::const_iterator flow_block = 
    _info->inception_state->bkg_control.signal_chunks.flow_block_sequence.
      BlockAtFlow( _info->flow );
  _info->bkgObj->CPU_DarkMatterPCA( getFlowBlockSize(), flow_block->begin() );
}

void WorkSet::setJobToPostFitStep()
{
    _info->type = POST_FIT_STEPS;
    _info->bkgObj->region_data->fitters_applied=TIME_TO_DO_PREWELL;
}

void WorkSet::setJobToRemainRegionFit()
{
  _info->type = INITIAL_FLOW_BLOCK_REMAIN_REGIONAL_FIT;
  _info->bkgObj->region_data->fitters_applied=TIME_TO_DO_REMAIN_MULTI_FLOW_FIT_STEPS;
}

void WorkSet::putJobToCPU(WorkerInfoQueueItem item)
{
  _info->pq->GetCpuQueue()->PutItem(item);
}

void WorkSet::putJobToGPU(WorkerInfoQueueItem item)
{
  _info->pq->GetGpuQueue()->PutItem(item);
}

void WorkSet::printJobSummary()
{

  if( ValidJob() )
  {
    cout << " | Job Summary:" << endl
    << " | max beads: " << GpuMultiFlowFitControl::GetMaxBeads() << " max frames: " << GpuMultiFlowFitControl::GetMaxFrames() << endl
    << " | live beads: " << getNumBeads() <<" padded: "<< getPaddedN()  << endl
    << " | num frames: " << getNumFrames() << endl
    << " | flow num:   " << getAbsoluteFlowNum() << endl

    ; 
  }
  else{
   cout << "| No Valid Job Set" << endl;
  }

}

int WorkSet::getXtalkNeiIdxMapSize(bool padded)
{
  int size = sizeof(int) * MAX_XTALK_NEIGHBOURS;  
  return size*( (!padded)?(getNumBeads()):(getPaddedN()) ); 
}

int WorkSet::getNumXtalkNeighbours() {
  return _info->bkgObj->getXtalkExecute().xtalk_spec_p->nei_affected;  
}

const int* WorkSet::getNeiIdxMapForXtalk() {
  return _info->bkgObj->getXtalkExecute().GetNeighborIndexMap();
}

int* WorkSet::getXtalkNeiXCoords() {
  return &_info->bkgObj->getXtalkExecute().xtalk_spec_p->cx[0];  
}

int* WorkSet::getXtalkNeiYCoords() {
  return &_info->bkgObj->getXtalkExecute().xtalk_spec_p->cy[0];  
}

float* WorkSet::getXtalkNeiMultiplier() {
  return &_info->bkgObj->getXtalkExecute().xtalk_spec_p->multiplier[0];  
}

float* WorkSet::getXtalkNeiTauTop() {
  return &_info->bkgObj->getXtalkExecute().xtalk_spec_p->tau_top[0];  
}

float* WorkSet::getXtalkNeiTauFluid() {
  return &_info->bkgObj->getXtalkExecute().xtalk_spec_p->tau_fluid[0];  
}

void WorkSet::calculateCPUXtalkForBead(int ibd, float* buf) {
  FlowBlockSequence::const_iterator flow_block = 
    _info->inception_state->bkg_control.signal_chunks.flow_block_sequence.
      BlockAtFlow( _info->flow );
  _info->bkgObj->getXtalkExecute().ExecuteXtalkFlux(ibd, buf, flow_block->size(),
    flow_block->begin()
  ); 
}

bool WorkSet::performCrossTalkCorrection() const {
  if(isSet()) {
    return _info->bkgObj->getXtalkExecute().xtalk_spec_p->do_xtalk_correction;
  }
  return false;
}

bool WorkSet::performExpTailFitting() {
  return _info->bkgObj->getGlobalDefaultsForBkgModel().signal_process_control.exp_tail_fit;
}

bool WorkSet::performCalcPCADarkMatter() {
  return _info->bkgObj->getGlobalDefaultsForBkgModel().signal_process_control.pca_dark_matter;
}

bool WorkSet::useDarkMatterPCA() {
  return (  performCalcPCADarkMatter() && _info->bkgObj->region_data->my_regions.missing_mass.mytype == PCAVector)?(true):(false) ;
}

bool WorkSet::InitializeAmplitude() {
  if (_info->bkgObj->getGlobalDefaultsForBkgModel().signal_process_control.regional_sampling)
    return _info->bkgObj->getGlobalDefaultsForBkgModel().signal_process_control.amp_guess_on_gpu;
  else
    return false;
}

int WorkSet::GetNumUnCompressedFrames() {
  if(isSet())
    return _info->bkgObj->region_data->time_c.GetUncompressedFrames();

  return MAX_UNCOMPRESSED_FRAMES_GPU;
}

int WorkSet::GetNumStdCompressedFrames() {
  if(isSet())
    return _info->bkgObj->region_data->time_c.GetStdFrames(); 

  return getMaxFrames();
}

int WorkSet::GetNumETFCompressedFrames() {
  if(isSet())
    return _info->bkgObj->region_data->time_c.GetETFFrames();

  return getMaxFrames();
}

int* WorkSet::GetStdFramesPerPoint() {
  return &((_info->bkgObj->region_data->time_c.GetStdFramesPerPoint())[0]);
}

int* WorkSet::GetETFInterpolationFrames() {
  return &((_info->bkgObj->region_data->time_c.GetETFInterpolationFrame())[0]);
}

float* WorkSet::GetETFInterpolationMul() {
  return &((_info->bkgObj->region_data->time_c.GetETFInterpolationMul())[0]);
}

int WorkSet::GetETFStartFrame() {
  return _info->bkgObj->region_data->time_c.GetETFStartFrame();
}

int WorkSet::GetStdFramesPerPointSize(bool padded)
{
  int size = sizeof(int)*GetNumStdCompressedFrames();
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::GetETFInterpolationFrameSize(bool padded)
{
  int size = sizeof(int)*GetNumUnCompressedFrames();
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::GetETFInterpolationMulSize(bool padded)
{
  int size = sizeof(float)*GetNumUnCompressedFrames();
  return (!padded)?(size):(padTo128Bytes(size));
}


float * WorkSet::GetStdTimeCompEmphasis()
{   
  return &_info->bkgObj->region_data->std_time_comp_emphasis.emphasis_vector_storage[0]; 
} 

int WorkSet::GetStdTimeCompEmphasisSize(bool padded)
{
  int size = sizeof(float)*(MAX_POISSON_TABLE_COL)*GetNumStdCompressedFrames();
  return (!padded)?(size):(padTo128Bytes(size));
}
 
float * WorkSet::GetStdTimeCompNucRise()
{  
  nucRise.Alloc(GetNumStdCompressedFrames(), getFlowBlockSize()); 
  nucRise.CalculateNucRiseFineStep (
      &_info->bkgObj->region_data->my_regions.rp, 
      GetNumStdCompressedFrames(),
      _info->bkgObj->region_data->time_c.GetStdFrameNumber(), 
      *_info->bkgObj->region_data_extras.my_flow); // the same for the whole region because time-shift happens per well
  return nucRise.nuc_rise_fine_step; 
} 

int WorkSet::GetStdTimeCompNucRiseSize(bool padded)
{
  int size = sizeof(float) * ISIG_SUB_STEPS_SINGLE_FLOW * GetNumStdCompressedFrames() * getFlowBlockSize();
  return (!padded)?(size):(padTo128Bytes(size));
}
 
float* WorkSet::GetStdTimeCompDeltaFrame() {
  return &((_info->bkgObj->region_data->time_c.GetStdDeltaFrame())[0]);
}

float* WorkSet::GetStdTimeCompFrameNumber() {
  return &((_info->bkgObj->region_data->time_c.GetStdFrameNumber())[0]);
}

int WorkSet::GetStdTimeCompDeltaFrameSize(bool padded)
{
  int size = sizeof(float)*GetNumStdCompressedFrames();
  return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::GetStdTimeCompFrameNumberSize(bool padded)
{
  int size = sizeof(float)*GetNumStdCompressedFrames();
  return (!padded)?(size):(padTo128Bytes(size));
}

void WorkSet::setUpFineEmphasisVectorsForStdCompression() 
{
  _info->bkgObj->region_data->GenerateFineEmphasisForStdTimeCompression();
}

int* WorkSet::GetNonZeroEmphasisFrames()
{
  return &_info->bkgObj->region_data->emphasis_data.nonZeroEmphasisFrames[0];
}

int* WorkSet::GetNonZeroEmphasisFramesForStdCompression()
{
  return &_info->bkgObj->region_data->std_time_comp_emphasis.nonZeroEmphasisFrames[0];
}

int WorkSet::GetNonZeroEmphasisFramesVecSize(bool padded)
{
  int size = sizeof(int) * MAX_POISSON_TABLE_COL;
  return (!padded) ? (size) : (padTo128Bytes(size));  
}

