/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>

#include "JobWrapper.h"
#include "SignalProcessingFitterQueue.h"
#include "GpuMultiFlowFitControl.h"


using namespace std;


////////////////////////////////////////
// workset (represents one job


WorkSet::WorkSet()
{
 _info = NULL;
}

WorkSet::~WorkSet()
{
}



void WorkSet::setData(BkgModelWorkInfo * i)
{
 _info = i;
}

bool WorkSet::isSet()
{
 return (_info != NULL)?(true):(false);
}


int WorkSet::getNumBeads() {  

    
    if(isSet()){            
      if ((unsigned int)_info->bkgObj->region_data->my_beads.numLBeads > (unsigned int)GpuMultiFlowFitControl::GetMaxBeads()) { cerr << "numLBeads ("<< _info->bkgObj->region_data->my_beads.numLBeads << ") > MaxBeads (" << GpuMultiFlowFitControl::GetMaxBeads() << ") !" << endl; exit(-1);}
      return _info->bkgObj->region_data->my_beads.numLBeads; 
    }
  return  GpuMultiFlowFitControl::GetMaxBeads();
}


int WorkSet::getNumFrames() { 

  if(isSet()){
    if ((unsigned int)_info->bkgObj->region_data->time_c.GetTimeCompressedFrames() > (unsigned int) GpuMultiFlowFitControl::GetMaxFrames()) { cerr << "numframes ("<< _info->bkgObj->region_data->time_c.GetTimeCompressedFrames()<<")  > MaxFrames ("<< GpuMultiFlowFitControl::GetMaxFrames() <<") !" << endl; exit(-1);}
    return _info->bkgObj->region_data->time_c.GetTimeCompressedFrames(); 
  }
  return  GpuMultiFlowFitControl::GetMaxFrames();
}


int WorkSet::getAbsoluteFlowNum() { return _info->bkgObj->region_data->my_flow.buff_flow[0]; }

reg_params * WorkSet::getRegionParams() {   return  &_info->bkgObj->region_data->my_regions.rp;  }
BeadTracker * WorkSet::getBeadTracker(){   return &_info->bkgObj->region_data->my_beads; }
bead_params * WorkSet::getBeadParams(){   return &_info->bkgObj->region_data->my_beads.params_nn[0]; }
bead_state * WorkSet::getState(){   return &_info->bkgObj->region_data->my_beads.all_status[0]; }
float * WorkSet::getEmphVec(){   return &_info->bkgObj->region_data->emphasis_data.emphasis_vector_storage[0]; } 
float * WorkSet::getDarkMatter(){   return &_info->bkgObj->region_data->my_regions.missing_mass.dark_matter_compensator[0]; }
int * WorkSet::getFlowIdxMap(){   return _info->bkgObj->region_data->my_flow.flow_ndx_map; }
FG_BUFFER_TYPE * WorkSet::getFgBuffer(){   return _info->bkgObj->region_data->my_trace.fg_buffers; }
float * WorkSet::getDeltaFrames(){   return &_info->bkgObj->region_data->time_c.deltaFrame[0]; }  
int * WorkSet::getStartNuc(){   return _info->bkgObj->region_data->my_regions.cache_step.i_start_fine_step; }


float * WorkSet::getShiftedBackground(){   
  _info->bkgObj->region_data->my_scratch.FillShiftedBkg (*_info->bkgObj->region_data->emptytrace, _info->bkgObj->region_data->my_regions.rp.tshift, _info->bkgObj->region_data->time_c, true);
  return _info->bkgObj->region_data->my_scratch.shifted_bkg; 
}

float * WorkSet::getCalculateNucRise(){   
  _info->bkgObj->region_data->my_regions.cache_step.CalculateNucRiseFineStep (&_info->bkgObj->region_data->my_regions.rp, _info->bkgObj->region_data->time_c, _info->bkgObj->region_data->my_flow); // the same for the whole region because time-shift happens per well
  return _info->bkgObj->region_data->my_regions.cache_step.nuc_rise_fine_step; 
}

float * WorkSet::getCalculateNucRiseCoarse()
{
  _info->bkgObj->region_data->my_regions.cache_step.CalculateNucRiseCoarseStep (&_info->bkgObj->region_data->my_regions.rp, _info->bkgObj->region_data->time_c, _info->bkgObj->region_data->my_flow);
  return &_info->bkgObj->region_data->my_regions.cache_step.nuc_rise_coarse_step[0];
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
  return _info->bkgObj->getGlobalDefaultsForBkgModel().signal_process_control.kmult_low_limit;
}

float WorkSet::getkmultHighLimit()
{
  return _info->bkgObj->getGlobalDefaultsForBkgModel().signal_process_control.kmult_hi_limit;
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
  return (ChipIdDecoder::GetGlobalChipId() == ChipId900);
}
//////////////////////////////////////////////////////////////////////////////////////
///SIZES:

//// N

int WorkSet::getBeadParamsSize(bool padded)
{
	int size = sizeof(bead_params);
  return size*( (!padded)?(getNumBeads()):(getPaddedN()) ); 
}

int WorkSet::getStateSize(bool padded)
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
	int size = sizeof(FG_BUFFER_TYPE)*getNumFrames()*NUMFB;
  return size*( (!padded)?(getNumBeads()):(getPaddedN()) ); 
}

int WorkSet::getFlxFxB(bool padded)
{
  int size = sizeof(float)*NUMFB*getNumFrames();
  return size*( (!padded)?(getNumBeads()):(getPaddedN()) ); 
}

int WorkSet::getFxB(bool padded)
{
  int size = sizeof(float)*getNumFrames();
  return size*( (!padded)?(getNumBeads()):(getPaddedN()) ); 
}

int WorkSet::getFlxB(bool padded)
{
  int size = sizeof(float)*NUMFB;
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
	int size = sizeof(float)*NUMFB*getNumFrames();
	return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getFlowIdxMapSize(bool padded)
{
	int size = sizeof(int)*NUMFB;
	return (!padded)?(size):(padTo128Bytes(size));
}



int WorkSet::getDeltaFramesSize(bool padded)
{
	int size = sizeof(float)*getNumFrames();
	return (!padded)?(size):(padTo128Bytes(size));
}
 
int WorkSet::getNucRiseSize(bool padded)
{
	int size = sizeof(float) * ISIG_SUB_STEPS_SINGLE_FLOW * getNumFrames() * NUMFB;
	return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getStartNucSize(bool padded)
{
	int size = sizeof(int)*NUMFB;
	return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getNucRiseCoarseSize(bool padded)
{
	int size = sizeof(float) * ISIG_SUB_STEPS_MULTI_FLOW * getNumFrames() * NUMFB;
	return (!padded)?(size):(padTo128Bytes(size));
}

int WorkSet::getStartNucCoarseSize(bool padded)
{
	int size = sizeof(int)*NUMFB;
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
 





//////////////////////////////////////////////////////////////////////////////////////
///

int WorkSet::getPaddedN(){
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
  return true;

}

void WorkSet::KeyNormalize()
{
  _info->bkgObj->region_data->my_beads.my_mean_copy_count = _info->bkgObj->region_data->my_beads.KeyNormalizeReads(true);   
}

void WorkSet::putPostFitStep(WorkerInfoQueueItem item)
{
    _info->type = POST_FIT_STEPS;
    _info->bkgObj->region_data->fitters_applied=TIME_TO_DO_PREWELL;
    _info->pq->GetCpuQueue()->PutItem(item);
}

void WorkSet::putRemainRegionFit(WorkerInfoQueueItem item)
{
  _info->type = INITIAL_FLOW_BLOCK_REMAIN_REGIONAL_FIT;
  _info->bkgObj->region_data->fitters_applied=TIME_TO_DO_REMAIN_MULTI_FLOW_FIT_STEPS;
  _info->pq->GetCpuQueue()->PutItem(item);
}






