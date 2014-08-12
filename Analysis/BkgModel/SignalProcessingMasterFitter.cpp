/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <float.h>
#include <vector>
#include <memory>
#include <assert.h>
#include "LinuxCompat.h"
#include "SignalProcessingMasterFitter.h"
#include "RawWells.h"
#include "MathOptim.h"
#include "mixed.h"
#include "BkgDataPointers.h"
#include "DNTPRiseModel.h"
#include "DiffEqModel.h"
#include "DiffEqModelVec.h"
#include "MultiLevMar.h"
#include "DarkMatter.h"
#include "RefineFit.h"
#include "SpatialCorrelator.h"
#include "RefineTime.h"
#include "TraceCorrector.h"
#include "BkgMagicDefines.h"
#include "SampleQuantiles.h"

#define EMPHASIS_AMPLITUDE 1.0



// #define LIVE_WELL_WEIGHT_BG


/*--------------------------------Top level decisons start ----------------------------------*/

//
// Reads image file data and stores it in circular buffer
// This is >the< entry point for this functions
// Everything else is just setup
// this triggers the fit when enough data has accumulated
//
bool SignalProcessingMasterFitter::ProcessImage ( Image *img, int raw_flow, int flow_buffer_index, int flow_block_size )
{
  if ( NeverProcessRegion() ) {
    return false; // no error happened,nothing to do
  }

  AllocateRegionDataIfNeeded ( img );

  if ( img->doLocalRescaleRegionByEmptyWells() ) // locally rescale to the region
    img->LocalRescaleRegionByEmptyWells ( region_data->region );

  // Calculate average background for each well
  //  region_data->emptyTraceTracker->SetEmptyTracesFromImageForRegion ( *img, *global_state.pinnedInFlow, flow, global_state.bfmask, *region_data->region, region_data->t_mid_nuc_start);
  region_data->emptyTraceTracker->SetEmptyTracesFromImageForRegion ( *img, 
    *global_state.pinnedInFlow, raw_flow, global_state.bfmask, *region_data->region, region_data->t0_frame, flow_buffer_index);


  if ( region_data->LoadOneFlow ( img,global_defaults, *region_data_extras.my_flow, raw_flow, flow_block_size ) )
    return ( true ); // error happened when loading image


  return false;  // no error happened
}


//prototype GPU execution functions
// ProcessImage had to be broken into two function, before and after GPUGenerateBeadTraces.
bool SignalProcessingMasterFitter::InitProcessImageForGPU (
    Image *img, 
    int raw_flow, 
    int flow_buffer_index 
  )
{
  if ( NeverProcessRegion() ) {
    return false; // no error happened,nothing to do
  }

  AllocateRegionDataIfNeeded ( img );

  if ( img->doLocalRescaleRegionByEmptyWells() ) // locally rescale to the region
    img->LocalRescaleRegionByEmptyWells ( region_data->region );

  // Calculate average background for each well
  //  region_data->emptyTraceTracker->SetEmptyTracesFromImageForRegion ( *img, *global_state.pinnedInFlow, flow, global_state.bfmask, *region_data->region, region_data->t_mid_nuc_start);
  region_data->emptyTraceTracker->SetEmptyTracesFromImageForRegion ( *img, 
    *global_state.pinnedInFlow, raw_flow, global_state.bfmask, *region_data->region, 
    region_data->t0_frame, flow_buffer_index );

  //for GPU execution call Prepare Load Flow
  if ( region_data->PrepareLoadOneFlowGPU ( img,global_defaults, *region_data_extras.my_flow, 
                                            raw_flow ) )
    return ( true ); // error happened when loading image

  return false;  // no error happened
}

//prototype GPU execution functions
// ProcessImage had to be broken into two function, before and after GPUGenerateBeadTraces.
bool SignalProcessingMasterFitter::FinalizeProcessImageForGPU ( int flow_block_size )
{
	if ( NeverProcessRegion() ) {
	    return false; // no error happened,nothing to do
	 }

	if ( region_data->FinalizeLoadOneFlowGPU ( *region_data_extras.my_flow, flow_block_size ) )
	    return ( true ); // error happened when loading image

	  return false;  // no error happened
}



void SignalProcessingMasterFitter::AllocateRegionDataIfNeeded ( Image *img )
{
  if ( region_data->my_trace.NeedsAllocation() )
  {
    //SetupTimeAndBuffers ( region_data->sigma_start, region_data->t_mid_nuc_start, region_data->t_mid_nuc_start );
    SetupTimeAndBuffers ( region_data->sigma_start, region_data->t_mid_nuc_start, 
      region_data->t0_frame, region_data_extras.my_flow->GetMaxFlowCount(),
      region_data_extras.global_flow_max);
  }
}

bool SignalProcessingMasterFitter::ProcessImage ( 
    SynchDat &data, 
    int raw_flow, 
    int flow_buffer_index,
    int flow_block_size
  )
{
  if ( NeverProcessRegion() )
    return false; // no error happened,nothing to do

  AllocateRegionDataIfNeeded ( data );

  // Calculate average background for each well
  region_data->emptyTraceTracker->SetEmptyTracesFromImageForRegion ( data, 
    *global_state.pinnedInFlow, raw_flow, global_state.bfmask, *region_data->region, 
    GetTypicalMidNucTime (& (region_data->my_regions.rp.nuc_shape)), 
    region_data->my_regions.rp.nuc_shape.sigma, region_data->time_c.time_start, 
    &region_data->time_c, flow_buffer_index );

  if ( region_data->LoadOneFlow ( data,global_defaults, *region_data_extras.my_flow, raw_flow, flow_block_size ) )
  {
    return ( true ); // error happened when loading image
  }

  return false;  // no error happened
}

void SignalProcessingMasterFitter::AllocateRegionDataIfNeeded ( SynchDat &data )
{
  if ( region_data->my_trace.NeedsAllocation() )
  {
    TraceChunk &chunk = data.GetItemByRowCol ( get_region_row(), get_region_col() );
    // timing initialized to match chunk timing
    SetupTimeAndBuffers ( chunk.mSigma, chunk.mTMidNuc, chunk.mT0, 
      region_data_extras.my_flow->GetMaxFlowCount(),
      region_data_extras.global_flow_max);
    //    SetupTimeAndBuffers ( chunk.mSigma, chunk.mTMidNuc, chunk.mT0 );
    if ( chunk.RegionMatch ( *region_data->region ) )
    {
      if ( chunk.TimingMatch ( region_data->time_c.mTimePoints ) )
      {
        // both region and timing match chunk
        region_data->regionAndTimingMatchSdat = true;
      }
    }
  }
}

/*bool SignalProcessingMasterFitter::TestAndExecuteBlock ( int flow, bool last )
{
  if ( !NeverProcessRegion() and TriggerBlock ( last ) )
  {
    region_data->fitters_applied = TIME_TO_DO_UPSTREAM;
    ExecuteBlock ( flow, last );
  }
  return false;  // no error happened
}*/

bool SignalProcessingMasterFitter::TestAndTriggerComputation ( bool last )
{
  if ( !NeverProcessRegion() and TriggerBlock ( last ) )
  {
    region_data->fitters_applied = TIME_TO_DO_MULTIFLOW_REGIONAL_FIT;
    return true;
  }
  return false;
}

bool SignalProcessingMasterFitter::NeverProcessRegion()
{
  // or any other such criteria for skipping this whole region of the chip
  bool live_beads_absent = region_data->my_beads.numLBeads==0;
  return ( live_beads_absent );
}

bool SignalProcessingMasterFitter::TriggerBlock ( bool last )
{
  // When there are enough buffers filled,
  // do the computation and
  // pull out the results
  if ( !region_data_extras.my_flow->StopDropAndCalculate ( last ) )
    return false;
  else
    return ( true );
}

void SignalProcessingMasterFitter::DoPreComputationFiltering(int flow_block_size)
{
  // after we've loaded the dat files
  // perhaps we have some preprocessing to do before swinging into the compute
  if (global_defaults.signal_process_control.prefilter_beads){
    for (int iBuff=0; iBuff<flow_block_size; iBuff++)
      region_data->my_beads.ZeroOutPins(region_data->region, global_state.bfmask,
                                        *global_state.pinnedInFlow, 
                                        region_data_extras.my_flow->flow_ndx_map[iBuff], iBuff);
  }
}



void SignalProcessingMasterFitter::PreWellCorrectionFactors( int flow_block_size, int flow_block_start )
{
  if ( region_data->fitters_applied==TIME_TO_DO_PREWELL )
  {
    // anything that needs doing immediately before writing to wells
    // hypothetically, this all happens at the well level and doesn't involve the signal processing at all.
    PostModelProtonCorrection( flow_block_size, flow_block_start );
    // this will go away when we add sub-region empty well traces
    // UGLY: if empty-well normalization is turned on...the measured amplitude actually needs
    // to be corrected for the scaling that was done on the raw data...
    CompensateAmplitudeForEmptyWellNormalization( flow_block_size );
    region_data->fitters_applied = TIME_TO_DO_EXPORT;
  }
}

void SignalProcessingMasterFitter::CompensateAmplitudeForEmptyWellNormalization( int flow_block_size )
{
  // direct buffer access is a bad, bad hack.
  region_data->my_beads.CompensateAmplitudeForEmptyWellNormalization ( region_data->my_trace.bead_scale_by_flow, flow_block_size );
}

void SignalProcessingMasterFitter::PostModelProtonCorrection( int flow_block_size, int flow_block_start )
{
  // what the heck..why not?!??!?
  if ( global_defaults.signal_process_control.enable_well_xtalk_correction )
  {
    well_xtalk_corrector.AmplitudeCorrectAllFlows( flow_block_size, flow_block_start );
  }
}

void SignalProcessingMasterFitter::ExportAllAndReset ( int flow, bool last, int flow_block_size, const PolyclonalFilterOpts & opts, int flow_block_id, int flow_block_start )
{
  if ( region_data->fitters_applied == TIME_TO_DO_EXPORT )
  {
    // internal updates happen before external updates
    UpdateClonalFilterData ( flow, opts, flow_block_size, flow_block_start ); // export to "clonal filter" within regional data
    // phi_adj
    region_data->my_beads.UpdateAllPhi(flow, flow_block_size);


    // external updates here
    ExportStatusToMask(flow); // export status to bf mask

    ExportDataToWells( flow_block_start ); // export condensed data to wells - note we munged the amplitude values in Proton data

    if (write_debug_files)
      ExportDataToDataCubes ( last, flow, flow_block_id, flow_block_start ); // export hdf5 data if writing out - may want to put >this< before Proton wells correction



    //@TODO: we reset here rather than explicitly locking: no guarantee that data remains invariant if accessed after this point
    ResetForNextBlockOfData(); // finally done, reset for next block of flows
    region_data->fitters_applied = TIME_FOR_NEXT_BLOCK;
  }
}

void SignalProcessingMasterFitter::ExportStatusToMask(int flow)
{
  region_data->my_beads.WriteCorruptedToMask ( region_data->region, global_state.bfmask, global_state.washout_flow, flow );
}

void SignalProcessingMasterFitter::UpdateClonalFilterData ( int flow, const PolyclonalFilterOpts & opts, int flow_block_size, int flow_block_start )
{
  if ( global_defaults.signal_process_control.do_clonal_filter )
  {
    vector<float> copy_mult ( flow_block_size );
    for ( int f=0; f<flow_block_size; ++f )
      copy_mult[f] = region_data->my_regions.rp.CalculateCopyDrift ( flow_block_start + f );

    region_data->my_beads.UpdateClonalFilter ( flow, copy_mult, opts, flow_block_size, flow_block_start );
  }
}

void SignalProcessingMasterFitter::ExportDataToWells( int flow_block_start )
{
  for ( int fnum=0; fnum<region_data_extras.my_flow->flowBufferCount; fnum++ )
  {
    global_state.WriteAnswersToWells ( fnum,region_data->region,&region_data->my_regions,
      region_data->my_beads, flow_block_start );
  }
}

void SignalProcessingMasterFitter::ExportDataToDataCubes ( bool last, int last_flow, int flow_block_id, int flow_block_start )
{
  for ( int fnum=0; fnum<region_data_extras.my_flow->flowBufferCount; fnum++ )
  {
    global_state.WriteBeadParameterstoDataCubes ( fnum,last,region_data->region,
      region_data->my_beads, *region_data_extras.my_flow, region_data->my_trace, flow_block_id,
      flow_block_start );
  }
  // now regional parameters
  // so they are exported >before< we reset(!)
  int max_frames = global_defaults.signal_process_control.get_max_frames();
  global_state.WriteRegionParametersToDataCubes(region_data, &region_data_extras, max_frames, 
    region_data_extras.my_flow->flowBufferCount, flow_block_id, flow_block_start,
    last, last_flow );
}



void SignalProcessingMasterFitter::ResetForNextBlockOfData()
{
  region_data_extras.my_flow->ResetBuffersForWriting();
}

void SignalProcessingMasterFitter::MultiFlowRegionalFitting ( int flow, bool last, int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start )
{
  if ( region_data->fitters_applied == TIME_TO_DO_MULTIFLOW_REGIONAL_FIT)
  {
    SetFittersIfNeeded();

    if ( IsFirstBlock ( flow ) )
    {
      RegionalFittingForInitialFlowBlock( flow_key, flow_block_size, table, flow_block_start );
      region_data->fitters_applied = TIME_TO_DO_MULTIFLOW_FIT_ALL_WELLS;
    }
    else
    {
      RegionalFittingForLaterFlowBlock( flow_key, flow_block_size, table, flow_block_start );
      region_data->fitters_applied = TIME_TO_DO_DOWNSTREAM;
    }
  }
}


void SignalProcessingMasterFitter::FitEmbarassinglyParallelRefineFit ( int flow_block_size, int flow_block_start )
{
  if ( region_data->fitters_applied == TIME_TO_DO_DOWNSTREAM )
  {
    CPUxEmbarassinglyParallelRefineFit ( flow_block_size, flow_block_start );
    region_data->fitters_applied = TIME_TO_DO_PREWELL;
  }
}

bool SignalProcessingMasterFitter::IsFirstBlock ( int flow )
{
  return ( ( flow+1 ) <=region_data_extras.my_flow->GetMaxFlowCount() );  //@TODO:  this should access the >region_data<, not the >flow<
}

/*--------------------------------Top level decisons end ----------------------------------*/


/*--------------------------------- Allocation section start ------------------------------------------------------*/

void SignalProcessingMasterFitter::NothingInit()
{
  region_data = NULL;
  NothingFitters();
}




void SignalProcessingMasterFitter::NothingFitters()
{
  // fitters of all types
  math_poiss = NULL;
  refine_fit = NULL;
  axion_fit = NULL;
  refine_time_fit = NULL;
  trace_bkg_adj = NULL;
}


void SignalProcessingMasterFitter::SetComputeControlFlags ( bool enable_trace_xtalk_correction )
{
  trace_xtalk_spec.do_xtalk_correction = enable_trace_xtalk_correction;
}


// constructor used by Analysis pipeline
SignalProcessingMasterFitter::SignalProcessingMasterFitter ( 
    RegionalizedData *local_patch, 
    const SlicedChipExtras & local_extras,
    GlobalDefaultsForBkgModel &_global_defaults,
    const char *_results_folder, Mask *_mask, PinnedInFlow *_pinnedInFlow, 
    RawWells *_rawWells,
    Region *_region, set<int>& sample, 
    const vector<float>& sep_t0_est, bool debug_trace_enable,
    int _rows, int _cols, int _frames, int _uncompFrames, int *_timestamps,
    EmptyTraceTracker *_emptyTraceTracker, float sigma_guess,float t_mid_nuc_guess,
    float t0_frame_guess, bool nokey, SequenceItem* _seqList,int _numSeqListItems, bool restart,
    int16_t *_washout_flow, const CommandLineOpts *inception_state )
  : global_defaults ( _global_defaults )
{
  NothingInit();


  global_state.FillExternalLinks ( _mask,_pinnedInFlow,_rawWells, _washout_flow );
  global_state.MakeDirName ( _results_folder );

  region_data=local_patch;
  region_data_extras = local_extras;

  if ( !restart )
  {
    region_data->region = _region;
    region_data->my_trace.SetImageParams ( _rows,_cols,_frames,_uncompFrames,_timestamps );
    region_data->emptyTraceTracker = _emptyTraceTracker;
  }
  BkgModelInit ( debug_trace_enable,sigma_guess,t_mid_nuc_guess,t0_frame_guess,sep_t0_est,sample,
    nokey,_seqList,_numSeqListItems, restart );
}


void SignalProcessingMasterFitter::UpdateBeadBufferingFromExternalEstimates ( vector<float> *tauB, vector<float> *tauE )
{
  region_data->my_beads.InitBeadParamR ( global_state.bfmask, region_data->region, tauB, tauE );
}


void SignalProcessingMasterFitter::BkgModelInit ( bool debug_trace_enable,float sigma_guess,
                                                  float t_mid_nuc_guess, float t0_frame_guess, 
                                                  const vector<float>& sep_t0_est,
                                                  set<int>& sample, bool nokey, SequenceItem* _seqList,int _numSeqListItems, bool restart )
{

  if ( !restart )
  {
    // use separator estimate
    region_data->t_mid_nuc_start = t_mid_nuc_guess;
    region_data->sigma_start = sigma_guess;
    region_data->t0_frame = (int)(t0_frame_guess + VFC_T0_OFFSET + .5);
    region_data->my_trace.T0EstimateToMap ( sep_t0_est,region_data->region,global_state.bfmask );

    region_data->my_beads.InitBeadList ( global_state.bfmask,region_data->region, nokey, 
        _seqList,_numSeqListItems, sample, global_defaults.signal_process_control.AmplLowerLimit);
  }

  if ( ( !NeverProcessRegion() ) && debug_trace_enable )
    my_debug.DebugFileOpen ( global_state.dirName, region_data->region );

  // Rest of initialization delayed until SetupTimeAndBuffers() called.
}

void SignalProcessingMasterFitter::InitializeFlowBlock( int flow_block_size )
{
  // We do this at the very beginning of whenever fitting is supposed to happen.
  if ( region_data_extras.my_flow->GetMaxFlowCount() != flow_block_size )
  {
    region_data_extras.my_flow->SetMaxFlowCount( flow_block_size );
  }
}

void SignalProcessingMasterFitter::SetupTimeAndBuffers ( float sigma_guess,
                                                         float t_mid_nuc_guess,
                                                         float t0_offset,
                                                         int flow_block_size,
                                                         int global_flow_max )
{
  region_data->SetupTimeAndBuffers ( global_defaults,sigma_guess,t_mid_nuc_guess,t0_offset, 
                                     flow_block_size, global_flow_max );
}

void SignalProcessingMasterFitter::SetFittersIfNeeded()
{
  if ( axion_fit==NULL )
    SetUpFitObjects();
}


void SignalProcessingMasterFitter::SetUpFitObjects()
{

  // Axion - dark matter fitter
  axion_fit = new Axion ( *this );
  // Lightweight friend object like the CUDA object holding a fitter
  // >must< be after the time, region, beads are allocated
  refine_fit = new RefineFit ( *this );


  refine_time_fit = new RefineTime ( *this );
  trace_bkg_adj = new TraceCorrector ( *this );
  // set up my_search with the things it needs
  // set up for cross-talk
  InitXtalk();
}

void SignalProcessingMasterFitter::InitXtalk()
{
  trace_xtalk_spec.BootUpXtalkSpec ( ( region_data->region!=NULL ),
                                     global_defaults.chipType.c_str(), global_defaults.xtalk_name.c_str() );
  trace_xtalk_execute.CloseOverPointers ( 
    region_data->region, &trace_xtalk_spec,
    &region_data->my_beads, &region_data->my_regions, &region_data->time_c, math_poiss,
    &region_data->my_scratch, 
    region_data_extras.cur_bead_block, region_data_extras.cur_buffer_block,
    region_data_extras.my_flow, &region_data->my_trace,
    global_defaults.signal_process_control.use_vectorization );

  //global_defaults.well_xtalk_master.TestWrite();
  //well_xtalk_corrector.my_xtalk.TestWrite();
  well_xtalk_corrector.my_xtalk = global_defaults.well_xtalk_master; // generate instance from master template
  well_xtalk_corrector.SetRegionData(region_data, &region_data_extras); // point to the data we'll be using
  // set well_xtalk parameters here
}


void SignalProcessingMasterFitter::DestroyFitObjects()
{
  if ( axion_fit !=NULL ) delete axion_fit;
  if ( refine_fit !=NULL ) delete refine_fit;
  if ( refine_time_fit !=NULL ) delete refine_time_fit;
  if ( trace_bkg_adj !=NULL ) delete trace_bkg_adj;
}

//Destructor
SignalProcessingMasterFitter::~SignalProcessingMasterFitter()
{
  my_debug.DebugFileClose();
  DestroyFitObjects();
}


/*--------------------------------- Allocation section done ------------------------------------------------------*/

/*--------------------------------Control analysis flow start ----------------------------------------------------*/

void SignalProcessingMasterFitter::BootUpModel ( double &elapsed_time,Timer &fit_timer, int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start )
{
  // what are emphasis functions here?
  // they are set to maximum default 0 for each bead - make explicit so we >know<
  region_data->LimitedEmphasis();
  fit_timer.restart();
  //fit_control.FitInitial
  if ( global_defaults.signal_process_control.regional_sampling ){
    ChooseSampledForRegionParamFit( flow_block_size );
    //double sampling_time = fit_timer.elapsed();
    //printf("====> Time in sampling in bkg model: %.2f\n", sampling_time);
    FirstPassSampledRegionParamFit( flow_key, flow_block_size, table, flow_block_start );
  }
  else {
    FirstPassRegionParamFit( flow_key, flow_block_size, table, flow_block_start );
  }

  elapsed_time += fit_timer.elapsed();

  ////printf("====> Time in booting up bkg model: %.2f\n", elapsed_time);
}

// choose a sample of beads that will be used for regional parameter fitting
void SignalProcessingMasterFitter::ChooseSampledForRegionParamFit( int flow_block_size )
{
  assert ( global_defaults.signal_process_control.regional_sampling );

  switch (global_defaults.signal_process_control.regional_sampling_type) {
    case REGIONAL_SAMPLING_SYSTEMATIC:
    {
      region_data->my_beads.SetSampled();
      // fprintf(stdout, "Sampled %d beads from %d live in region %d\n",region_data->my_beads.NumberSampled(),region_data->my_beads.numLBeads, region_data->region->index);
      break;
    }
    case REGIONAL_SAMPLING_CLONAL_KEY_NORMALIZED:
    {
      std::vector<float> penalty(region_data->my_beads.numLBeads, 0);
      region_data->my_beads.ntarget = 200;  // beads to sample
      double pool_fraction = 0.02; // sampling_rate=1 for 100x100 thumbnail
      int sampling_rate = (region_data->my_beads.numLBeads*pool_fraction)/region_data->my_beads.ntarget;
      sampling_rate = std::max(1.0, (double)sampling_rate);
      region_data->CalculateFirstBlockClonalPenalty(global_defaults.data_control.nuc_flow_frame_width, penalty, global_defaults.signal_process_control.regional_sampling_type, flow_block_size);
      region_data->my_beads.SetSampled(penalty, sampling_rate);

      // fprintf(stdout, "Sampled %d beads from %d live in region %d with rate %d\n",region_data->my_beads.NumberSampled(),region_data->my_beads.numLBeads, region_data->region->index, sampling_rate);
    }
      break;
      // return;
    default:
      assert(false);  // this should never happen
  }
}

// first pass fit using all sampled beads
// initially chosen for regional parameter fitting
void SignalProcessingMasterFitter::FirstPassSampledRegionParamFit( int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start )
{
  assert ( global_defaults.signal_process_control.regional_sampling );

  MultiFlowLevMar first_lev_mar_fit( *this, flow_block_size, table );

  first_lev_mar_fit.MultiFlowSpecializedSampledLevMarFitParameters ( 1, 3, first_lev_mar_fit.fit_control.FitWellAmplBuffering, first_lev_mar_fit.fit_control.FitRegionTmidnucPlus, SMALL_LAMBDA , NO_NONCLONAL_PENALTY, flow_key, flow_block_size, flow_block_start );
  region_data->my_beads.my_mean_copy_count = region_data->my_beads.KeyNormalizeSampledReads ( true, flow_block_size );
  first_lev_mar_fit.MultiFlowSpecializedSampledLevMarFitParameters ( 1, 1, first_lev_mar_fit.fit_control.FitWellAmplBuffering, first_lev_mar_fit.fit_control.FitRegionTmidnucPlus, SMALL_LAMBDA , NO_NONCLONAL_PENALTY, flow_key, flow_block_size, flow_block_start );
  region_data->my_beads.my_mean_copy_count = region_data->my_beads.KeyNormalizeSampledReads ( true, flow_block_size );
  first_lev_mar_fit.MultiFlowSpecializedSampledLevMarFitParameters ( 1,1, first_lev_mar_fit.fit_control.FitWellAmplBuffering, first_lev_mar_fit.fit_control.FitRegionTmidnucPlus,SMALL_LAMBDA , NO_NONCLONAL_PENALTY, flow_key, flow_block_size, flow_block_start );
}

// first pass fit using all beads
// initially chosen for regional parameter fitting
void SignalProcessingMasterFitter::FirstPassRegionParamFit( int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start )
{
  assert ( !global_defaults.signal_process_control.regional_sampling );

  MultiFlowLevMar first_lev_mar_fit( *this, flow_block_size, table );

  first_lev_mar_fit.MultiFlowSpecializedLevMarFitParameters ( 1, 3, first_lev_mar_fit.fit_control.FitWellAmplBuffering, first_lev_mar_fit.fit_control.FitRegionTmidnucPlus, SMALL_LAMBDA , NO_NONCLONAL_PENALTY, flow_key, flow_block_size, flow_block_start );
  region_data->my_beads.my_mean_copy_count = region_data->my_beads.KeyNormalizeReads ( true, false, flow_block_size );
  first_lev_mar_fit.MultiFlowSpecializedLevMarFitParameters ( 1, 1, first_lev_mar_fit.fit_control.FitWellAmplBuffering, first_lev_mar_fit.fit_control.FitRegionTmidnucPlus, SMALL_LAMBDA, NO_NONCLONAL_PENALTY, flow_key, flow_block_size, flow_block_start );
  region_data->my_beads.my_mean_copy_count = region_data->my_beads.KeyNormalizeReads ( true, false, flow_block_size );
  first_lev_mar_fit.MultiFlowSpecializedLevMarFitParameters ( 1, 1, first_lev_mar_fit.fit_control.FitWellAmplBuffering, first_lev_mar_fit.fit_control.FitRegionTmidnucPlus, SMALL_LAMBDA , NO_NONCLONAL_PENALTY, flow_key, flow_block_size, flow_block_start );
}

void SignalProcessingMasterFitter::PostKeyFit (MultiFlowLevMar &post_key_fit, double &elapsed_time, Timer &fit_timer, int flow_key, int flow_block_size, int flow_block_start )
{
  //MultiFlowLevMar lev_mar_fit( *this, flow_block_size );

  region_data->LimitedEmphasis();

  fit_timer.restart();
  region_data->RezeroByCurrentTiming( flow_block_size ); // rezeroing??

  if ( !global_defaults.signal_process_control.regional_sampling )
    post_key_fit.MultiFlowSpecializedLevMarFitParameters ( NO_ADDITIONAL_WELL_ITERATIONS, 8, post_key_fit.fit_control.FitWellPostKey, post_key_fit.fit_control.FitRegionInit2, LARGER_LAMBDA , NO_NONCLONAL_PENALTY, flow_key, flow_block_size, flow_block_start );
  else
    post_key_fit.MultiFlowSpecializedSampledLevMarFitParameters ( NO_ADDITIONAL_WELL_ITERATIONS, 8, post_key_fit.fit_control.FitWellPostKey, post_key_fit.fit_control.FitRegionInit2, LARGER_LAMBDA , NO_NONCLONAL_PENALTY, flow_key, flow_block_size, flow_block_start );
  elapsed_time += fit_timer.elapsed();

  region_data->RezeroByCurrentTiming( flow_block_size );
  // try to use the first non-key cycle to help normalize everything and
  // identify incorporation model parameters
  //if(do_clonal_filter)
  //    my_beads.FindClonalReads();

  // new PCA-based dark matter must be computed across all wells, so it can't be done until after we've at least done some fitting of every well
  // if regional sampling is turned off...this is the place to fit it, though fitting it later (and using regional sampling) seems to work
  // better
  if ( global_defaults.signal_process_control.pca_dark_matter && !global_defaults.signal_process_control.regional_sampling)
    axion_fit->CalculatePCADarkMatter(false, flow_block_size, flow_block_start );

  fit_timer.restart();
  if ( !global_defaults.signal_process_control.regional_sampling )
    post_key_fit.MultiFlowSpecializedLevMarFitParameters ( NO_ADDITIONAL_WELL_ITERATIONS, 8, post_key_fit.fit_control.FitWellPostKey, post_key_fit.fit_control.FitRegionFull, LARGER_LAMBDA, FULL_NONCLONAL_PENALTY, flow_key, flow_block_size, flow_block_start );
  else
    post_key_fit.MultiFlowSpecializedSampledLevMarFitParameters ( NO_ADDITIONAL_WELL_ITERATIONS, 8, post_key_fit.fit_control.FitWellPostKey, post_key_fit.fit_control.FitRegionFull, LARGER_LAMBDA, FULL_NONCLONAL_PENALTY, flow_key, flow_block_size, flow_block_start );
  elapsed_time += fit_timer.elapsed();

}

void SignalProcessingMasterFitter::PostKeyFitAllWells ( double &elapsed_time, Timer &fit_timer, int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start )
{
  MultiFlowLevMar all_wells_lev_mar_fit( *this, flow_block_size, table );

  fit_timer.restart();
  all_wells_lev_mar_fit.MultiFlowSpecializedLevMarFitAllWells ( 1, all_wells_lev_mar_fit.fit_control.FitWellAmplBuffering, SMALL_LAMBDA, NO_NONCLONAL_PENALTY, flow_key, flow_block_size, flow_block_start );
  region_data->my_beads.my_mean_copy_count = region_data->my_beads.KeyNormalizeReads ( true, false, flow_block_size );

  // new PCA-based dark matter must be computed across all wells, so it can't be done until after we've at least done some fitting of every well
  // but it is also helpful to do this step before we finish the training....so it has been inserted here between the first and second step
  // of fitting all wells
  if ( global_defaults.signal_process_control.pca_dark_matter )
    axion_fit->CalculatePCADarkMatter(false, flow_block_size, flow_block_start);

  // only wells are fit here
  all_wells_lev_mar_fit.MultiFlowSpecializedLevMarFitAllWells ( HAPPY_ALL_BEADS, all_wells_lev_mar_fit.fit_control.FitWellPostKey, LARGER_LAMBDA, FULL_NONCLONAL_PENALTY, flow_key, flow_block_size, flow_block_start );
  elapsed_time += fit_timer.elapsed();
  region_data->my_beads.my_mean_copy_count = region_data->my_beads.KeyNormalizeReads ( true, false, flow_block_size );
}

void SignalProcessingMasterFitter::CPU_DarkMatterPCA( int flow_block_size, int flow_block_start )
{  
  axion_fit->CalculatePCADarkMatter(false, flow_block_size, flow_block_start );
}

void SignalProcessingMasterFitter::ApproximateDarkMatter ( const LevMarBeadAssistant & post_key_state, bool isSampled, int flow_block_size, int flow_block_start )
{
  region_data->LimitedEmphasis();
  // now figure out whatever remaining error there is in the fit, on average
  // if ( isSampled ) {
  //  region_data->my_beads.UpdateSampled( lev_mar_fit->lm_state.well_completed );
  // }
  
  if (global_defaults.signal_process_control.enable_dark_matter)
    axion_fit->CalculateDarkMatter ( FIRSTNFLOWSFORERRORCALC, post_key_state.residual, post_key_state.avg_resid*2.0, flow_block_size, flow_block_start );
  else
    region_data->my_regions.missing_mass.ResetDarkMatter();
}

void SignalProcessingMasterFitter::FitAmplitudeAndDarkMatter ( MultiFlowLevMar & fad_lev_mar_fit, double &elapsed_time, Timer &fit_timer, int flow_key, int flow_block_size, int flow_block_start )
{

  region_data->AdaptiveEmphasis();
  region_data->SetCrudeEmphasisVectors();

  //@TODO: should I be skipping low-quality bead refits here because we'll be getting their amplitudes in the refinement phase?
  fit_timer.restart();
  if (global_defaults.signal_process_control.enable_dark_matter)
    fad_lev_mar_fit.MultiFlowSpecializedLevMarFitParameters ( NO_ADDITIONAL_WELL_ITERATIONS, 8, fad_lev_mar_fit.fit_control.FitWellAmpl, fad_lev_mar_fit.fit_control.FitRegionDarkness, BIG_LAMBDA , NO_NONCLONAL_PENALTY, flow_key, flow_block_size, flow_block_start );
  else
    fad_lev_mar_fit.MultiFlowSpecializedLevMarFitParameters ( NO_ADDITIONAL_WELL_ITERATIONS, 8, fad_lev_mar_fit.fit_control.FitWellAmpl, fad_lev_mar_fit.fit_control.DontFitRegion, BIG_LAMBDA , NO_NONCLONAL_PENALTY, flow_key, flow_block_size, flow_block_start );

  elapsed_time += fit_timer.elapsed();
}

void SignalProcessingMasterFitter::FitWellParametersConditionalOnRegion ( MultiFlowLevMar & lev_mar_fit, double &elapsed_time, Timer &fit_timer )
{
  fit_timer.restart();
  lev_mar_fit.ChooseSkipBeads ( false );

  elapsed_time += fit_timer.elapsed();
}

void SignalProcessingMasterFitter::RegionalFittingForInitialFlowBlock( int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start )
{

  Timer fit_timer;
  double elapsed_time = 0;

  region_data->my_beads.ResetLocalBeadParams(); // start off with no data for amplitude/kmult
  region_data->my_regions.ResetLocalRegionParams( flow_block_size ); // start off with no per flow time shifts
  DoPreComputationFiltering( flow_block_size);  // already there may be beads that shouldn't be fit
  
  
  BootUpModel ( elapsed_time,fit_timer, flow_key, flow_block_size, table, flow_block_start );

  // now that we know something about the wells, select a good subset
  // by any filtration we think is good
  region_data->PickRepresentativeHighQualityWells ( global_defaults.signal_process_control.ssq_filter, flow_block_size );
  // fprintf(stdout, "Sampled %d beads from %d live in region %d with high-quality\n",region_data->my_beads.NumberSampled(),region_data->my_beads.numLBeads, region_data->region->index);

  MultiFlowLevMar post_key_fit( *this, flow_block_size, table );
  post_key_fit.ChooseSkipBeads ( false );
  // these two should be do-able only on representative wells
  PostKeyFit (post_key_fit, elapsed_time, fit_timer, flow_key, flow_block_size, flow_block_start );

  // if not doing PCA dark matter correction...do old style dark matter correction
  if ( !global_defaults.signal_process_control.pca_dark_matter )
    ApproximateDarkMatter (post_key_fit.lm_state , global_defaults.signal_process_control.regional_sampling, flow_block_size, flow_block_start );

  // Set things up for double exponential smoothing.
  region_data->my_regions.copy_drift_smoother.Initialize( & region_data->my_regions.rp );
  region_data->my_regions.ratio_drift_smoother.Initialize( & region_data->my_regions.rp );
}

void SignalProcessingMasterFitter::FitAllBeadsForInitialFlowBlock( int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start )
{
  if ( region_data->fitters_applied == TIME_TO_DO_MULTIFLOW_FIT_ALL_WELLS)
  {
    Timer fit_timer;
    double elapsed_time = 0;

    
    if (region_data->my_beads.doAllBeads)
      region_data->my_beads.IgnoreQuality(); // all beads processed in all flows

    if ( global_defaults.signal_process_control.regional_sampling )
      PostKeyFitAllWells ( elapsed_time, fit_timer, flow_key, flow_block_size, table, flow_block_start );

    region_data->fitters_applied = TIME_TO_DO_REMAIN_MULTI_FLOW_FIT_STEPS;
  }
}

void SignalProcessingMasterFitter::RemainingFitStepsForInitialFlowBlock( int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start )
{
  if ( region_data->fitters_applied == TIME_TO_DO_REMAIN_MULTI_FLOW_FIT_STEPS)
  {
    MultiFlowLevMar remaining_lev_mar_fit( *this, flow_block_size, table );
    Timer fit_timer;
    double elapsed_time = 0;

    remaining_lev_mar_fit.ChooseSkipBeads ( true );

    // the PCA dark matter correction does it's own fitting and this step is no longer needed
    // if PCA correction has been selected, otherwise do the old-style dark matter fit
    if ( !global_defaults.signal_process_control.pca_dark_matter )
      FitAmplitudeAndDarkMatter ( remaining_lev_mar_fit, elapsed_time, fit_timer, flow_key, flow_block_size, flow_block_start );

    // catch up well parameters on wells we didn't use in the regional estimates
    if ( remaining_lev_mar_fit.SkipBeads() )
      FitWellParametersConditionalOnRegion ( remaining_lev_mar_fit, elapsed_time, fit_timer );

#ifdef TUNE_INCORP_PARAMS_DEBUG
    DumpRegionParameters ( remaining_lev_mar_fit.lm_state.avg_resid );
#endif

    region_data->my_regions.RestrictRatioDrift();

    refine_time_fit->RefinePerFlowTimeEstimate ( region_data->my_regions.rp.nuc_shape.t_mid_nuc_shift_per_flow, flow_block_size, flow_block_start );

    region_data->fitters_applied = TIME_TO_DO_DOWNSTREAM;
  }
}

void SignalProcessingMasterFitter::GuessCrudeAmplitude ( double &elapsed_time, Timer &fit_timer, bool sampledOnly, int flow_block_size, int flow_block_start)
{
  region_data->SetCrudeEmphasisVectors();

  //@TODO: should I skip low-quality beads because I'm going to refit them later and I only need this for regional parameters?
  fit_timer.restart();
  //@TODO: badly organized data still requiring too much stuff passed around
  // I blame MultiFlowModel wrapper functions that don't exist
  region_data->LimitedEmphasis();
  my_search.ParasitePointers ( math_poiss, &region_data->my_trace,region_data->emptytrace,&region_data->my_scratch,region_data_extras.cur_bead_block, region_data_extras.cur_buffer_block, & ( region_data->my_regions ),&region_data->time_c,
    region_data_extras.my_flow,&region_data->emphasis_data );
  //  if (global_defaults.signal_process_control.generic_test_flag)
  //    my_search.BinarySearchAmplitude (region_data->my_beads, 0.5f,true); // apply search method to current bead list - wrong OO?  Method on bead list?
  //  else
  my_search.ProjectionSearchAmplitude ( region_data->my_beads, false, sampledOnly, flow_block_size, flow_block_start); // need all our speed
  //my_search.GoldenSectionAmplitude(my_beads);
  elapsed_time += fit_timer.elapsed();

}

void SignalProcessingMasterFitter::FitTimeVaryingRegion ( double &elapsed_time, Timer &fit_timer, int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start )
{
  MultiFlowLevMar tvr_lev_mar_fit( *this, flow_block_size, table );
  fit_timer.restart();
  // >NOW< we allow any emphasis level given our crude estimates for emphasis
  region_data->AdaptiveEmphasis();
  tvr_lev_mar_fit.ChooseSkipBeads ( true );
  tvr_lev_mar_fit.MultiFlowSpecializedLevMarFitParametersOnlyRegion ( 4, tvr_lev_mar_fit.fit_control.FitRegionTimeVarying, LARGER_LAMBDA , NO_NONCLONAL_PENALTY, flow_key, flow_block_size, flow_block_start );
  tvr_lev_mar_fit.ChooseSkipBeads ( false );
  elapsed_time += fit_timer.elapsed();
}

void SignalProcessingMasterFitter::RegionalFittingForLaterFlowBlock( int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start )
{
  Timer fit_timer;
  double elapsed_time = 0;

  region_data->my_beads.ResetLocalBeadParams(); // start off with no data for amplitude/kmult
  region_data->my_regions.ResetLocalRegionParams( flow_block_size ); // start off with no per flow time shifts
  DoPreComputationFiltering(flow_block_size);

  if (global_defaults.signal_process_control.regional_sampling)
  {
    GuessCrudeAmplitude (elapsed_time,fit_timer,global_defaults.signal_process_control.amp_guess_on_gpu, flow_block_size, flow_block_start);
  }
  else {
    GuessCrudeAmplitude (elapsed_time,fit_timer,false, flow_block_size, flow_block_start);
  }

  FitTimeVaryingRegion ( elapsed_time,fit_timer, flow_key, flow_block_size, table, flow_block_start );

  // Here is where we can do double exponential smoothing.
  region_data->my_regions.copy_drift_smoother.Smooth( & region_data->my_regions.rp );
  region_data->my_regions.ratio_drift_smoother.Smooth( & region_data->my_regions.rp );

  refine_time_fit->RefinePerFlowTimeEstimate ( region_data->my_regions.rp.nuc_shape.t_mid_nuc_shift_per_flow, flow_block_size, flow_block_start );
}

void SignalProcessingMasterFitter::RefineAmplitudeEstimates ( double &elapsed_time, Timer &fit_timer, int flow_block_size, int flow_block_start )
{
  // in either case, finish by fitting amplitude to finalized regional parameters
  region_data->SetFineEmphasisVectors();
  fit_timer.restart();
  refine_fit->FitAmplitudePerFlow ( flow_block_size, flow_block_start );
  double fit_ampl_time = fit_timer.elapsed();
  elapsed_time += fit_ampl_time;
  if ( ( my_debug.region_trace_file != NULL ) && ( my_debug.region_only_trace_file != NULL ) )
  {
    my_debug.DumpRegionTrace ( *this, flow_block_size, flow_block_start );
  }
}

void SignalProcessingMasterFitter::CPUxEmbarassinglyParallelRefineFit( int flow_block_size, int flow_block_start )
{
  Timer fit_timer;
  Timer total_timer;
  double elapsed_time=0;
  // now do a final amplitude estimation on each bead
  RefineAmplitudeEstimates ( elapsed_time,fit_timer, flow_block_size, flow_block_start );
  //printf("Time in single flow fit: %f\n", elapsed_time);
}

/*--------------------------------Control analysis flow end ----------------------------------------------------*/
