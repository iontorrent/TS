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
bool SignalProcessingMasterFitter::ProcessImage ( Image *img, int flow )
{
  if ( NeverProcessRegion() )
    return false; // no error happened,nothing to do

  AllocateRegionDataIfNeeded ( img );

  if ( img->doLocalRescaleRegionByEmptyWells() ) // locally rescale to the region
    img->LocalRescaleRegionByEmptyWells ( region_data->region );

  // Calculate average background for each well
  region_data->emptyTraceTracker->SetEmptyTracesFromImageForRegion ( *img, *global_state.pinnedInFlow, flow, global_state.bfmask, *region_data->region, region_data->t_mid_nuc_start);


  if ( region_data->LoadOneFlow ( img,global_defaults,flow ) )
    return ( true ); // error happened when loading image


  return false;  // no error happened
}

void SignalProcessingMasterFitter::AllocateRegionDataIfNeeded ( Image *img )
{
  if ( region_data->my_trace.NeedsAllocation() )
  {
    SetupTimeAndBuffers ( region_data->sigma_start, region_data->t_mid_nuc_start, region_data->t_mid_nuc_start );
  }
}

bool SignalProcessingMasterFitter::ProcessImage ( SynchDat &data, int flow )
{
  if ( NeverProcessRegion() )
    return false; // no error happened,nothing to do

  AllocateRegionDataIfNeeded ( data );

  // Calculate average background for each well
  region_data->emptyTraceTracker->SetEmptyTracesFromImageForRegion ( data, *global_state.pinnedInFlow, flow, global_state.bfmask, *region_data->region, GetTypicalMidNucTime (& (region_data->my_regions.rp.nuc_shape)) ,region_data->my_regions.rp.nuc_shape.sigma, region_data->time_c.time_start, &region_data->time_c ); 

  if ( region_data->LoadOneFlow ( data,global_defaults,flow ) )
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
    SetupTimeAndBuffers ( chunk.mSigma, chunk.mTMidNuc, chunk.mT0 );
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
  if ( !region_data->my_flow.StopDropAndCalculate ( last ) )
    return false;
  else
    return ( true );
}

void SignalProcessingMasterFitter::DoPreComputationFiltering()
{
  // after we've loaded the dat files
  // perhaps we have some preprocessing to do before swinging into the compute
  if (global_defaults.signal_process_control.prefilter_beads){
  for (int iBuff=0; iBuff<region_data->my_flow.numfb; iBuff++)
    region_data->my_beads.ZeroOutPins(region_data->region, global_state.bfmask, *global_state.pinnedInFlow, region_data->my_flow.flow_ndx_map[iBuff], iBuff);
  }
}



void SignalProcessingMasterFitter::PreWellCorrectionFactors()
{
  if ( region_data->fitters_applied==TIME_TO_DO_PREWELL )
  {
    // anything that needs doing immediately before writing to wells
    // hypothetically, this all happens at the well level and doesn't involve the signal processing at all.
    PostModelProtonCorrection();
    // this will go away when we add sub-region empty well traces
    // UGLY: if empty-well normalization is turned on...the measured amplitude actually needs
    // to be corrected for the scaling that was done on the raw data...
    CompensateAmplitudeForEmptyWellNormalization();
    region_data->fitters_applied = TIME_TO_DO_EXPORT;
  }
}

void SignalProcessingMasterFitter::CompensateAmplitudeForEmptyWellNormalization()
{
  // direct buffer access is a bad, bad hack.
  region_data->my_beads.CompensateAmplitudeForEmptyWellNormalization ( region_data->my_trace.bead_scale_by_flow );
}

void SignalProcessingMasterFitter::PostModelProtonCorrection()
{
  // what the heck..why not?!??!?
  if ( global_defaults.signal_process_control.proton_dot_wells_post_correction )
  {
    correct_spatial->AmplitudeCorrectAllFlows();
  }
}

void SignalProcessingMasterFitter::ExportAllAndReset ( int flow, bool last )
{
  if ( region_data->fitters_applied == TIME_TO_DO_EXPORT )
  {
    UpdateClonalFilterData ( flow ); // export to "clonal filter" within regional data

    ExportStatusToMask(flow); // export status to bf mask

    ExportDataToWells(); // export condensed data to wells - note we munged the amplitude values in Proton data
    ExportDataToDataCubes ( last ); // export hdf5 data if writing out - may want to put >this< before Proton wells correction

    //@TODO: we reset here rather than explicitly locking: no guarantee that data remains invariant if accessed after this point
    ResetForNextBlockOfData(); // finally done, reset for next block of flows
    region_data->fitters_applied = TIME_FOR_NEXT_BLOCK;
  }
}

void SignalProcessingMasterFitter::ExportStatusToMask(int flow)
{
  region_data->my_beads.WriteCorruptedToMask ( region_data->region, global_state.bfmask, global_state.washout_flow, flow );
}

void SignalProcessingMasterFitter::UpdateClonalFilterData ( int flow )
{
  if ( global_defaults.signal_process_control.do_clonal_filter )
  {
    vector<float> copy_mult ( NUMFB );
    for ( int f=0; f<NUMFB; ++f )
      copy_mult[f] = CalculateCopyDrift ( region_data->my_regions.rp, region_data->my_flow.buff_flow[f] );

    region_data->my_beads.UpdateClonalFilter ( flow, copy_mult );
  }
}

void SignalProcessingMasterFitter::ExportDataToWells()
{
  for ( int fnum=0; fnum<region_data->my_flow.flowBufferCount; fnum++ )
  {
    global_state.WriteAnswersToWells ( fnum,region_data->region,&region_data->my_regions,region_data->my_beads,region_data->my_flow );
  }
}

void SignalProcessingMasterFitter::ExportDataToDataCubes ( bool last )
{
  for ( int fnum=0; fnum<region_data->my_flow.flowBufferCount; fnum++ )
  {
    global_state.WriteBeadParameterstoDataCubes ( fnum,last,region_data->region,region_data->my_beads, region_data->my_flow, region_data->my_trace );
  }
  // now regional parameters
  // so they are exported >before< we reset(!)
  global_state.WriteRegionParametersToDataCubes(region_data);
}



void SignalProcessingMasterFitter::ResetForNextBlockOfData()
{
  region_data->my_flow.ResetBuffersForWriting();
}

void SignalProcessingMasterFitter::MultiFlowRegionalFitting ( int flow, bool last )
{
  if ( region_data->fitters_applied == TIME_TO_DO_MULTIFLOW_REGIONAL_FIT)
  {
    SetFittersIfNeeded();
    if ( IsFirstBlock ( flow ) )
    {
      RegionalFittingForInitialFlowBlock();
      region_data->fitters_applied = TIME_TO_DO_MULTIFLOW_FIT_ALL_WELLS;
    }
    else
    {
      RegionalFittingForLaterFlowBlock();
      region_data->fitters_applied = TIME_TO_DO_DOWNSTREAM;
    }
  }
}


void SignalProcessingMasterFitter::FitEmbarassinglyParallelRefineFit ()
{
  if ( region_data->fitters_applied == TIME_TO_DO_DOWNSTREAM )
  {
    CPUxEmbarassinglyParallelRefineFit ();
    region_data->fitters_applied = TIME_TO_DO_PREWELL;
  }
}

bool SignalProcessingMasterFitter::IsFirstBlock ( int flow )
{
  return ( ( flow+1 ) <=region_data->my_flow.numfb );  //@TODO:  this should access the >region_data<, not the >flow<
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
  lev_mar_fit = NULL;
  refine_fit = NULL;
  axion_fit = NULL;
  correct_spatial = NULL;
  refine_time_fit = NULL;
  trace_bkg_adj = NULL;
}


void SignalProcessingMasterFitter::SetComputeControlFlags ( bool enable_xtalk_correction )
{
  xtalk_spec.do_xtalk_correction = enable_xtalk_correction;
}


// constructor used by Analysis pipeline
SignalProcessingMasterFitter::SignalProcessingMasterFitter ( RegionalizedData *local_patch, GlobalDefaultsForBkgModel &_global_defaults, char *_results_folder, Mask *_mask, PinnedInFlow *_pinnedInFlow, RawWells *_rawWells, Region *_region, set<int>& sample,
                                                             vector<float>& sep_t0_est, bool debug_trace_enable,
                                                             int _rows, int _cols, int _frames, int _uncompFrames, int *_timestamps,
                                                             EmptyTraceTracker *_emptyTraceTracker,
                                                             float sigma_guess,float t_mid_nuc_guess,
                                                             SequenceItem* _seqList,int _numSeqListItems, bool restart, int16_t *_washout_flow )
  : global_defaults ( _global_defaults )
{
  NothingInit();


  global_state.FillExternalLinks ( _mask,_pinnedInFlow,_rawWells, _washout_flow );
  global_state.MakeDirName ( _results_folder );

  region_data=local_patch;
  if ( !restart )
  {
    region_data->region = _region;
    region_data->my_trace.SetImageParams ( _rows,_cols,_frames,_uncompFrames,_timestamps );
    region_data->emptyTraceTracker = _emptyTraceTracker;
  }
  BkgModelInit ( debug_trace_enable,sigma_guess,t_mid_nuc_guess,sep_t0_est,sample,_seqList,_numSeqListItems, restart );
}



// constructor used for testing outside of Analysis pipeline (doesn't require mask, region, or RawWells obects)
SignalProcessingMasterFitter::SignalProcessingMasterFitter ( GlobalDefaultsForBkgModel &_global_defaults, int _numLBeads, int numFrames,
    float sigma_guess,float t_mid_nuc_guess ) : global_defaults ( _global_defaults )
{
  NothingInit();

  global_state.MakeDirName ( "" );

  region_data->my_beads.numLBeads = _numLBeads;
  region_data->my_trace.SetImageParams ( 0,0,numFrames,numFrames,NULL );

  set<int> emptySample;
  vector<float> empty_t0;
  BkgModelInit ( false,sigma_guess,t_mid_nuc_guess,empty_t0,emptySample,NULL, 0, false );
}


void SignalProcessingMasterFitter::UpdateBeadBufferingFromExternalEstimates ( vector<float> *tauB, vector<float> *tauE )
{
  region_data->my_beads.InitBeadParamR ( global_state.bfmask, region_data->region, tauB, tauE );
}


//@TODO: bad duplicated code between the two inits: fix!!!
void SignalProcessingMasterFitter::BkgModelInit ( bool debug_trace_enable,float sigma_guess,
    float t_mid_nuc_guess,vector<float>& sep_t0_est,
    set<int>& sample, SequenceItem* _seqList,int _numSeqListItems, bool restart )
{

  if ( !restart )
  {
    // use separator estimate
    region_data->t_mid_nuc_start = t_mid_nuc_guess;
    region_data->sigma_start = sigma_guess;
    region_data->my_trace.T0EstimateToMap ( sep_t0_est,region_data->region,global_state.bfmask );

    //Image parameters
    region_data->my_flow.numfb = NUMFB;

    region_data->my_beads.InitBeadList ( global_state.bfmask,region_data->region,_seqList,_numSeqListItems, sample, global_defaults.signal_process_control.AmplLowerLimit );
  }

  if ( ( !NeverProcessRegion() ) && debug_trace_enable )
    my_debug.DebugFileOpen ( global_state.dirName, region_data->region );

  // Rest of initialization delayed until SetupTimeAndBuffers() called.
}

void SignalProcessingMasterFitter::SetupTimeAndBuffers ( float sigma_guess,
    float t_mid_nuc_guess,
    float t0_offset )
{
  region_data->SetupTimeAndBuffers ( global_defaults,sigma_guess,t_mid_nuc_guess,t0_offset );
  //SetUpFitObjects();
}

void SignalProcessingMasterFitter::SetFittersIfNeeded()
{
  if ( lev_mar_fit==NULL )
    SetUpFitObjects();
}


void SignalProcessingMasterFitter::SetUpFitObjects()
{

  // allocate a multiflow levmar fitter
  lev_mar_fit = new MultiFlowLevMar ( *this );
  // Axion - dark matter fitter
  axion_fit = new Axion ( *this );
  // Lightweight friend object like the CUDA object holding a fitter
  // >must< be after the time, region, beads are allocated
  refine_fit = new RefineFit ( *this );

  correct_spatial = new SpatialCorrelator ( *this );
  refine_time_fit = new RefineTime ( *this );
  trace_bkg_adj = new TraceCorrector ( *this );
  // set up my_search with the things it needs
  // set up for cross-talk
  InitXtalk();
}

void SignalProcessingMasterFitter::InitXtalk()
{
  xtalk_spec.BootUpXtalkSpec ( ( region_data->region!=NULL ), global_defaults.chipType.c_str(), global_defaults.xtalk_name.c_str() );
  xtalk_execute.CloseOverPointers ( region_data->region, &xtalk_spec,&region_data->my_beads, &region_data->my_regions,
                                    &region_data->time_c, math_poiss, &region_data->my_scratch, &region_data->my_flow,
                                    &region_data->my_trace, global_defaults.signal_process_control.use_vectorization );
}


void SignalProcessingMasterFitter::DestroyFitObjects()
{
  if ( lev_mar_fit!=NULL ) delete lev_mar_fit;
  if ( axion_fit !=NULL ) delete axion_fit;
  if ( refine_fit !=NULL ) delete refine_fit;
  if ( correct_spatial !=NULL ) delete correct_spatial;
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

void SignalProcessingMasterFitter::BootUpModel ( double &elapsed_time,Timer &fit_timer )
{
// what are emphasis functions here?
// they are set to maximum default 0 for each bead - make explicit so we >know<
  region_data->LimitedEmphasis();
  fit_timer.restart();
  //fit_control.FitInitial
  if ( global_defaults.signal_process_control.regional_sampling ){
    // set up sampling if needed
    region_data->my_beads.SetSampled();
  }

  if ( !global_defaults.signal_process_control.regional_sampling )
  {
    lev_mar_fit->MultiFlowSpecializedLevMarFitParameters ( 1, 3, lev_mar_fit->fit_control.FitWellAmplBuffering, lev_mar_fit->fit_control.FitRegionTmidnucPlus, SMALL_LAMBDA , NO_NONCLONAL_PENALTY );
    region_data->my_beads.my_mean_copy_count = region_data->my_beads.KeyNormalizeReads ( true );
    lev_mar_fit->MultiFlowSpecializedLevMarFitParameters ( 1, 1, lev_mar_fit->fit_control.FitWellAmplBuffering, lev_mar_fit->fit_control.FitRegionTmidnucPlus, SMALL_LAMBDA, NO_NONCLONAL_PENALTY );
    region_data->my_beads.my_mean_copy_count = region_data->my_beads.KeyNormalizeReads ( true );
    lev_mar_fit->MultiFlowSpecializedLevMarFitParameters ( 1, 1, lev_mar_fit->fit_control.FitWellAmplBuffering, lev_mar_fit->fit_control.FitRegionTmidnucPlus, SMALL_LAMBDA , NO_NONCLONAL_PENALTY );
  }
  else
  {
    lev_mar_fit->MultiFlowSpecializedSampledLevMarFitParameters ( 1, 3, lev_mar_fit->fit_control.FitWellAmplBuffering, lev_mar_fit->fit_control.FitRegionTmidnucPlus, SMALL_LAMBDA , NO_NONCLONAL_PENALTY );
    region_data->my_beads.my_mean_copy_count = region_data->my_beads.KeyNormalizeSampledReads ( true );
    lev_mar_fit->MultiFlowSpecializedSampledLevMarFitParameters ( 1, 1, lev_mar_fit->fit_control.FitWellAmplBuffering, lev_mar_fit->fit_control.FitRegionTmidnucPlus, SMALL_LAMBDA , NO_NONCLONAL_PENALTY );
    region_data->my_beads.my_mean_copy_count = region_data->my_beads.KeyNormalizeSampledReads ( true );
    lev_mar_fit->MultiFlowSpecializedSampledLevMarFitParameters ( 1,1, lev_mar_fit->fit_control.FitWellAmplBuffering, lev_mar_fit->fit_control.FitRegionTmidnucPlus,SMALL_LAMBDA , NO_NONCLONAL_PENALTY );
  }
  elapsed_time += fit_timer.elapsed();

}



void SignalProcessingMasterFitter::PostKeyFit ( double &elapsed_time, Timer &fit_timer )
{
  region_data->LimitedEmphasis();

  fit_timer.restart();
  region_data->RezeroByCurrentTiming(); // rezeroing??

  if ( !global_defaults.signal_process_control.regional_sampling )
    lev_mar_fit->MultiFlowSpecializedLevMarFitParameters ( NO_ADDITIONAL_WELL_ITERATIONS, 8, lev_mar_fit->fit_control.FitWellPostKey, lev_mar_fit->fit_control.FitRegionInit2, LARGER_LAMBDA , NO_NONCLONAL_PENALTY );
  else
    lev_mar_fit->MultiFlowSpecializedSampledLevMarFitParameters ( NO_ADDITIONAL_WELL_ITERATIONS, 8, lev_mar_fit->fit_control.FitWellPostKey, lev_mar_fit->fit_control.FitRegionInit2, LARGER_LAMBDA , NO_NONCLONAL_PENALTY );
  elapsed_time += fit_timer.elapsed();

  region_data->RezeroByCurrentTiming();
  // try to use the first non-key cycle to help normalize everything and
  // identify incorporation model parameters
  //if(do_clonal_filter)
  //    my_beads.FindClonalReads();

  fit_timer.restart();
  if ( !global_defaults.signal_process_control.regional_sampling )
    lev_mar_fit->MultiFlowSpecializedLevMarFitParameters ( NO_ADDITIONAL_WELL_ITERATIONS, 8, lev_mar_fit->fit_control.FitWellPostKey, lev_mar_fit->fit_control.FitRegionFull, LARGER_LAMBDA, FULL_NONCLONAL_PENALTY );
  else
    lev_mar_fit->MultiFlowSpecializedSampledLevMarFitParameters ( NO_ADDITIONAL_WELL_ITERATIONS, 8, lev_mar_fit->fit_control.FitWellPostKey, lev_mar_fit->fit_control.FitRegionFull, LARGER_LAMBDA, FULL_NONCLONAL_PENALTY );
  elapsed_time += fit_timer.elapsed();

}

void SignalProcessingMasterFitter::PostKeyFitAllWells ( double &elapsed_time, Timer &fit_timer )
{
  fit_timer.restart();
  lev_mar_fit->MultiFlowSpecializedLevMarFitAllWells ( 1, lev_mar_fit->fit_control.FitWellAmplBuffering, SMALL_LAMBDA, NO_NONCLONAL_PENALTY );
  region_data->my_beads.my_mean_copy_count = region_data->my_beads.KeyNormalizeReads ( true );
  // only wells are fit here
  lev_mar_fit->MultiFlowSpecializedLevMarFitAllWells ( HAPPY_ALL_BEADS, lev_mar_fit->fit_control.FitWellPostKey, LARGER_LAMBDA, FULL_NONCLONAL_PENALTY );
  elapsed_time += fit_timer.elapsed();
  region_data->my_beads.my_mean_copy_count = region_data->my_beads.KeyNormalizeReads ( true );
}

void SignalProcessingMasterFitter::ApproximateDarkMatter ( bool isSampled )
{
  region_data->LimitedEmphasis();
  // now figure out whatever remaining error there is in the fit, on average
  if ( isSampled ) {
    // lev_mar_fit->lm_state.UpdateBeadSampleList ( region_data->my_beads.high_quality ); //update the list of beads if we are sampling.
    region_data->my_beads.UpdateSampled( lev_mar_fit->lm_state.well_completed );
  }
  
    if (global_defaults.signal_process_control.enable_dark_matter)
      axion_fit->CalculateDarkMatter ( FIRSTNFLOWSFORERRORCALC, lev_mar_fit->lm_state.residual, lev_mar_fit->lm_state.avg_resid*2.0 );
   else
     region_data->my_regions.missing_mass.ResetDarkMatter();
}

void SignalProcessingMasterFitter::FitAmplitudeAndDarkMatter ( double &elapsed_time, Timer &fit_timer )
{

  region_data->AdaptiveEmphasis();
  region_data->SetCrudeEmphasisVectors();

  //@TODO: should I be skipping low-quality bead refits here because we'll be getting their amplitudes in the refinement phase?
  fit_timer.restart();
  if (global_defaults.signal_process_control.enable_dark_matter)
    lev_mar_fit->MultiFlowSpecializedLevMarFitParameters ( NO_ADDITIONAL_WELL_ITERATIONS, 8, lev_mar_fit->fit_control.FitWellAmpl, lev_mar_fit->fit_control.FitRegionDarkness, BIG_LAMBDA , NO_NONCLONAL_PENALTY );
  else
    lev_mar_fit->MultiFlowSpecializedLevMarFitParameters ( NO_ADDITIONAL_WELL_ITERATIONS, 8, lev_mar_fit->fit_control.FitWellAmpl, lev_mar_fit->fit_control.DontFitRegion, BIG_LAMBDA , NO_NONCLONAL_PENALTY );
    
  elapsed_time += fit_timer.elapsed();
}

void SignalProcessingMasterFitter::FitWellParametersConditionalOnRegion ( double &elapsed_time, Timer &fit_timer )
{
  fit_timer.restart();
  lev_mar_fit->ChooseSkipBeads ( false );

  elapsed_time += fit_timer.elapsed();
}

void SignalProcessingMasterFitter::RegionalFittingForInitialFlowBlock()
{
  Timer fit_timer;
  double elapsed_time = 0;

  region_data->my_beads.ResetLocalBeadParams(); // start off with no data for amplitude/kmult
  region_data->my_regions.ResetLocalRegionParams(); // start off with no per flow time shifts
  DoPreComputationFiltering();  // already there may be beads that shouldn't be fit
  
  
  BootUpModel ( elapsed_time,fit_timer );

  // now that we know something about the wells, select a good subset
  // by any filtration we think is good
  region_data->PickRepresentativeHighQualityWells ( global_defaults.signal_process_control.ssq_filter );

  lev_mar_fit->ChooseSkipBeads ( false );
  // these two should be do-able only on representative wells
  PostKeyFit ( elapsed_time, fit_timer );
  ApproximateDarkMatter ( global_defaults.signal_process_control.regional_sampling );
}

void SignalProcessingMasterFitter::FitAllBeadsForInitialFlowBlock()
{
  if ( region_data->fitters_applied == TIME_TO_DO_MULTIFLOW_FIT_ALL_WELLS) 
  {
    Timer fit_timer;
    double elapsed_time = 0;

    if ( global_defaults.signal_process_control.regional_sampling )
      PostKeyFitAllWells ( elapsed_time, fit_timer );

    region_data->fitters_applied = TIME_TO_DO_REMAIN_MULTI_FLOW_FIT_STEPS;
  }
}

void SignalProcessingMasterFitter::RemainingFitStepsForInitialFlowBlock()
{
  if ( region_data->fitters_applied == TIME_TO_DO_REMAIN_MULTI_FLOW_FIT_STEPS) 
  {
    Timer fit_timer;
    double elapsed_time = 0;

    lev_mar_fit->ChooseSkipBeads ( true );

    FitAmplitudeAndDarkMatter ( elapsed_time, fit_timer );

    // catch up well parameters on wells we didn't use in the regional estimates
    if ( lev_mar_fit->SkipBeads() )
      FitWellParametersConditionalOnRegion ( elapsed_time, fit_timer );

#ifdef TUNE_INCORP_PARAMS_DEBUG
  DumpRegionParameters ( lev_mar_fit->lm_state.avg_resid );
#endif

    region_data->my_regions.RestrictRatioDrift();
   
    refine_time_fit->RefinePerFlowTimeEstimate ( region_data->my_regions.rp.nuc_shape.t_mid_nuc_shift_per_flow );

    region_data->fitters_applied = TIME_TO_DO_DOWNSTREAM;
  }
}

void SignalProcessingMasterFitter::GuessCrudeAmplitude ( double &elapsed_time, Timer &fit_timer )
{
  region_data->SetCrudeEmphasisVectors();

  //@TODO: should I skip low-quality beads because I'm going to refit them later and I only need this for regional parameters?
  fit_timer.restart();
  //@TODO: badly organized data still requiring too much stuff passed around
  // I blame MultiFlowModel wrapper functions that don't exist
  region_data->LimitedEmphasis();
  my_search.ParasitePointers ( math_poiss, &region_data->my_trace,region_data->emptytrace,&region_data->my_scratch,& ( region_data->my_regions ),&region_data->time_c,&region_data->my_flow,&region_data->emphasis_data );
//  if (global_defaults.signal_process_control.generic_test_flag)
//    my_search.BinarySearchAmplitude (region_data->my_beads, 0.5f,true); // apply search method to current bead list - wrong OO?  Method on bead list?
//  else
  my_search.ProjectionSearchAmplitude ( region_data->my_beads, false ); // need all our speed
  //my_search.GoldenSectionAmplitude(my_beads);
  elapsed_time += fit_timer.elapsed();

}

void SignalProcessingMasterFitter::FitTimeVaryingRegion ( double &elapsed_time, Timer &fit_timer )
{
  fit_timer.restart();
  // >NOW< we allow any emphasis level given our crude estimates for emphasis
  region_data->AdaptiveEmphasis();
  lev_mar_fit->ChooseSkipBeads ( true );
  lev_mar_fit->MultiFlowSpecializedLevMarFitParametersOnlyRegion ( 4, lev_mar_fit->fit_control.FitRegionTimeVarying, LARGER_LAMBDA , NO_NONCLONAL_PENALTY );
  lev_mar_fit->ChooseSkipBeads ( false );
  elapsed_time += fit_timer.elapsed();
}

void SignalProcessingMasterFitter::RegionalFittingForLaterFlowBlock()
{
  Timer fit_timer;
  double elapsed_time = 0;

  region_data->my_beads.ResetLocalBeadParams(); // start off with no data for amplitude/kmult
  region_data->my_regions.ResetLocalRegionParams(); // start off with no per flow time shifts
  DoPreComputationFiltering();

  GuessCrudeAmplitude ( elapsed_time,fit_timer );
  FitTimeVaryingRegion ( elapsed_time,fit_timer );

  refine_time_fit->RefinePerFlowTimeEstimate ( region_data->my_regions.rp.nuc_shape.t_mid_nuc_shift_per_flow );
}

void SignalProcessingMasterFitter::RefineAmplitudeEstimates ( double &elapsed_time, Timer &fit_timer )
{
  // in either case, finish by fitting amplitude to finalized regional parameters
  region_data->SetFineEmphasisVectors();
  fit_timer.restart();
  refine_fit->FitAmplitudePerFlow ();
  double fit_ampl_time = fit_timer.elapsed();
  elapsed_time += fit_ampl_time;
  if ( ( my_debug.region_trace_file != NULL ) && ( my_debug.region_only_trace_file != NULL ) )
  {
    my_debug.DumpRegionTrace ( *this );
  }
}

void SignalProcessingMasterFitter::CPUxEmbarassinglyParallelRefineFit()
{
  Timer fit_timer;
  Timer total_timer;
  double elapsed_time=0;
  // now do a final amplitude estimation on each bead
  RefineAmplitudeEstimates ( elapsed_time,fit_timer );
  //printf("Time in single flow fit: %f\n", elapsed_time);
}

/*--------------------------------Control analysis flow end ----------------------------------------------------*/
