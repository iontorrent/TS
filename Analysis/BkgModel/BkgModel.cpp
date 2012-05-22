/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <float.h>
#include <vector>
#include <assert.h>
#include "LinuxCompat.h"
#include "BkgModel.h"
#include "RawWells.h"
#include "MathOptim.h"
#include "mixed.h"
#include "BkgDataPointers.h"


#ifdef ION_COMPILE_CUDA
#include "BkgModelCuda.h"
#endif

#include "DNTPRiseModel.h"
#include "DiffEqModel.h"
#include "DiffEqModelVec.h"
#include "MultiLevMar.h"
#include "DarkMatter.h"
#include "RefineFit.h"
#include "TraceCorrector.h"

#include "SampleQuantiles.h"

#define BKG_MODEL_DEBUG_DIR "/bkg_debug/"
#define EMPHASIS_AMPLITUDE 1.0



// #define LIVE_WELL_WEIGHT_BG


/*--------------------------------Top level decisons start ----------------------------------*/

//
// Reads image file data and stores it in circular buffer
// This is >the< entry point for this functions
// Everything else is just setup
// this triggers the fit when enough data has accumulated
//
bool BkgModel::ProcessImage (Image *img, int flow, bool last, bool learning, bool use_gpu)
{
  if (my_beads.numLBeads == 0)
    return false; // no error happened,nothing to do

  if (LoadOneFlow (img,flow))
    return (true); // error happened when loading image

  if (TriggerBlock (last))
    ExecuteBlock (flow, last,learning,use_gpu);

  return false;  // no error happened
}


bool BkgModel::LoadOneFlow (Image *img, int flow)
{
  const RawImage *raw = img->GetImage();
  if (!raw)
  {
    fprintf (stderr, "ERROR: no image data\n");
    return true;
  }

  AddOneFlowToBuffer (flow);

  UpdateTracesFromImage (img, flow);

  my_flow.Increment();

  return (false);
}

bool BkgModel::TriggerBlock (bool last)
{
  // When there are enough buffers filled,
  // do the computation and
  // pull out the results
  if ( (my_flow.flowBufferCount < my_flow.numfb) && !last)
    return false;
  else
    return (true);
}

void BkgModel::ExecuteBlock (int flow, bool last, bool learning, bool use_gpu)
{
  // do the fit
  FitModelForBlockOfFlows (flow,last,learning, use_gpu);
  
  FlushDataToWells (last);
}




void BkgModel::AddOneFlowToBuffer (int flow)
{
  // keep track of which flow # is in which buffer
  // also keep track of which nucleotide is associated with each flow
  my_flow.buff_flow[my_flow.flowBufferWritePos] = flow;
  my_flow.flow_ndx_map[my_flow.flowBufferWritePos] = global_defaults.glob_flow_ndx_map[flow%global_defaults.flow_order_len];
  my_flow.dbl_tap_map[my_flow.flowBufferWritePos] = global_defaults.IsDoubleTap (flow);

  // reset parameters for beads when we actually start fitting the beads
}

void BkgModel::UpdateTracesFromImage (Image *img, int flow)
{
  my_trace.SetRawTrace(); // buffers treated as raw traces
  
  // populate bead traces from image file and
  // time-shift traces for uniform start times; compress traces to flows buffer
  my_trace.GenerateAllBeadTrace (region,my_beads,img, my_flow.flowBufferWritePos);
  // subtract mean signal in time before flow starts from traces in flows buffer

  float t_mid_nuc =  GetTypicalMidNucTime (&my_regions->rp.nuc_shape);
  float t_offset_beads = my_regions->rp.nuc_shape.sigma;
  my_trace.RezeroBeads (time_c.time_start, t_mid_nuc-t_offset_beads,
                        my_flow.flowBufferWritePos);

  // calculate average trace across all empty wells in a region for a flow
  // to FileLoadWorker at Image load time, should be able to go here
  // emptyTraceTracker->SetEmptyTracesFromImageForRegion(*img, global_state.pinnedInFlow, flow, global_state.bfmask, *region, t_mid_nuc);
  emptytrace = emptyTraceTracker->GetEmptyTrace (*region);

  // sanity check images are what we think
  assert (emptytrace->imgFrames == my_trace.imgFrames);
}

void BkgModel::FlushDataToWells (bool last) // flush the traces, write data to wells
{
  if (my_flow.flowBufferCount > 0)
  {
    // Write Wells data
    while (my_flow.flowBufferCount > 0)
    {
      WriteAnswersToWells (my_flow.flowBufferReadPos);
      WriteBeadParameterstoDataCubes (my_flow.flowBufferReadPos, last);   // for bg_param.h5 output
      WriteDebugWells (my_flow.flowBufferReadPos); // detects if they're present, writes to them
      my_flow.Decrement();
    }
  }
}


// only call this when the buffers are full up and we're going to output something
// This is the big routine that fires off the optimizations we're doing
void BkgModel::FitModelForBlockOfFlows (int flow, bool last, bool learning, bool use_gpu)
{
  (void) learning;
  if (use_gpu) {}; // compiler happiness


#ifdef ION_COMPILE_CUDA
  if (use_gpu)
    GPUFitModelForBlockOfFlows (flow,last,learning);
  else
#endif
    CPUFitModelForBlockOfFlows (flow,last,learning);

  UpdateBeadStatusAfterFit (flow);

  my_beads.LimitBeadEvolution ( (flow+1) <=NUMFB,MAXRCHANGE,MAXCOPYCHANGE);

}

/*--------------------------------Top level decisons end ----------------------------------*/


/*--------------------------------- Allocation section start ------------------------------------------------------*/

void BkgModel::NothingInit()
{
  // I cannot believe I have to do this to make cppcheck happy
  region = NULL;
  emptyTraceTracker = NULL;
  emptytrace = NULL;
  lev_mar_fit = NULL;
  refine_fit = NULL;
  axion_fit = NULL;
  sigma_start = 0.0f;
  t_mid_nuc_start = 0.0f;
  do_clonal_filter = false;
  use_vectorization = false;
  AmplLowerLimit = 0.001f;
  mPtrs = NULL;
  math_poiss = NULL;
  my_regions = NULL;
  replay = NULL;
  trace_bkg_adj = NULL;

}

// constructor used by Analysis pipeline
BkgModel::BkgModel (char *_experimentName, Mask *_mask, PinnedInFlow *_pinnedInFlow, RawWells *_rawWells, Region *_region, set<int>& sample,
                    vector<float> *sep_t0_est, bool debug_trace_enable,
                    int _rows, int _cols, int _frames, int _uncompFrames, int *_timestamps, PoissonCDFApproxMemo *_math_poiss,
                    EmptyTraceTracker *_emptyTraceTracker,
                    BkgModelReplay *_replay,
                    float sigma_guess,float t_mid_nuc_guess,float dntp_uM,
                    bool enable_xtalk_correction,bool enable_clonal_filter,SequenceItem* _seqList,int _numSeqListItems)
{
  NothingInit();

  region = _region;

  global_state.rawWells = _rawWells;
  global_state.bfmask = _mask;
  global_state.pinnedInFlow = _pinnedInFlow;

  my_trace.SetImageParams (_rows,_cols,_frames,_uncompFrames,_timestamps);
  emptyTraceTracker = _emptyTraceTracker;
  replay = _replay;

  math_poiss = _math_poiss;
  xtalk_spec.do_xtalk_correction = enable_xtalk_correction;
  do_clonal_filter = enable_clonal_filter;



  BkgModelInit (_experimentName,debug_trace_enable,sigma_guess,t_mid_nuc_guess,dntp_uM,sep_t0_est,sample,_seqList,_numSeqListItems);
}

// constructor used by Analysis pipeline with tauB and tauE passed from Separator.
BkgModel::BkgModel (char *_experimentName, Mask *_mask, PinnedInFlow *_pinnedInFlow, RawWells *_rawWells, Region *_region, set<int>& sample,
                    vector<float> *sep_t0_est,vector<float> *tauB, vector<float> *tauE, bool debug_trace_enable,
                    int _rows, int _cols, int _frames, int _uncompFrames, int *_timestamps, PoissonCDFApproxMemo *_math_poiss,
                    EmptyTraceTracker *_emptyTraceTracker,
                    float sigma_guess,float t_mid_nuc_guess,float dntp_uM,
                    bool enable_xtalk_correction,bool enable_clonal_filter,SequenceItem* _seqList,int _numSeqListItems)
{
  NothingInit();
  region = _region;

  global_state.rawWells = _rawWells;
  global_state.bfmask = _mask;
  global_state.pinnedInFlow = _pinnedInFlow;

  my_trace.SetImageParams (_rows,_cols,_frames,_uncompFrames,_timestamps);
  emptyTraceTracker = _emptyTraceTracker;


  math_poiss = _math_poiss;
  xtalk_spec.do_xtalk_correction = enable_xtalk_correction;
  do_clonal_filter = enable_clonal_filter;


  BkgModelInit (_experimentName,debug_trace_enable,sigma_guess,t_mid_nuc_guess,dntp_uM,sep_t0_est,sample,_seqList,_numSeqListItems,tauB,tauE);
}

// constructor used for testing outside of Analysis pipeline (doesn't require mask, region, or RawWells obects)
BkgModel::BkgModel (int _numLBeads, int numFrames,
                    float sigma_guess,float t_mid_nuc_guess,float dntp_uM,
                    bool enable_xtalk_correction, bool enable_clonal_filter)
{
  NothingInit();

  global_state.rawWells = NULL;
  global_state.bfmask = NULL;
  global_state.pinnedInFlow = NULL;

  my_beads.numLBeads = _numLBeads;
  my_trace.SetImageParams (0,0,numFrames,numFrames,NULL);

  // make me a local cache for math
  math_poiss = new PoissonCDFApproxMemo;
  math_poiss->Allocate (MAX_HPLEN,512,0.05);
  math_poiss->GenerateValues();

  xtalk_spec.do_xtalk_correction = enable_xtalk_correction;
  do_clonal_filter = enable_clonal_filter;

  set<int> emptySample;
  BkgModelInit ("",false,sigma_guess,t_mid_nuc_guess,dntp_uM,NULL,emptySample,NULL, 0);
}

//@TODO: bad duplicated code between the two inits: fix!!!

//BkgModelInit with tauB and tauE
void BkgModel::BkgModelInit (char *_experimentName, bool debug_trace_enable,float sigma_guess,
                             float t_mid_nuc_guess,float dntp_uM,vector<float> *sep_t0_est, set<int>& sample, SequenceItem* _seqList,int _numSeqListItems, vector<float> *tauB, vector<float> *tauE)
{
  // initialize the rest of the parameters, including updating the beads
  BkgModelInit (_experimentName, debug_trace_enable, sigma_guess, t_mid_nuc_guess,dntp_uM, sep_t0_est,sample,_seqList,_numSeqListItems);
  // now add the tauB and tauE parameters as starting points for the bead parameters
  my_beads.InitBeadParamR (global_state.bfmask, region, tauB, tauE);
}


//@TODO: bad duplicated code between the two inits: fix!!!
void BkgModel::BkgModelInit (char *_experimentName, bool debug_trace_enable,float sigma_guess,
                             float t_mid_nuc_guess,float dntp_uM,vector<float> *sep_t0_est, set<int>& sample, SequenceItem* _seqList,int _numSeqListItems)
{
  global_state.dirName = (char *) malloc (strlen (_experimentName) +1);
  strncpy (global_state.dirName, _experimentName, strlen (_experimentName) +1); //@TODO: really?  why is this a local copy??

  sigma_start = sigma_guess;

  // ASSUMPTIONS:   15fps..  if that changes, this is hosed!!
  // use separator estimate
  t_mid_nuc_start = t_mid_nuc_guess;
  my_trace.T0EstimateToMap (sep_t0_est,region,global_state.bfmask);

  //Image parameters
  my_flow.numfb = NUMFB;

  my_beads.InitBeadList (global_state.bfmask,region,_seqList,_numSeqListItems, sample);

  if (replay == NULL)
  {
    replay = new BkgModelReplay (true);
  }
  replay->bkg = this;
  my_regions = replay->GetRegionTracker();

  if ( (my_beads.numLBeads > 0) && debug_trace_enable)
    DebugFileOpen();

  my_regions->InitRegionParams (t_mid_nuc_start,sigma_start,dntp_uM, global_defaults);

  SetTimeAndEmphasis();

  AllocTraceBuffers();

  AllocFitBuffers();

  // fix up multiflow lev_mar for operation
  InitLevMar();


  // Axion - dark matter fitter
  axion_fit = new Axion (*this);
  // Lightweight friend object like the CUDA object holding a fitter
  // >must< be after the time, region, beads are allocated
  refine_fit = new RefineFit (*this);

  trace_bkg_adj = new TraceCorrector(*this);

  // set up my_search with the things it needs
  // set up for cross-talk
  InitXtalk();
}

void BkgModel::InitXtalk()
{
  xtalk_spec.BootUpXtalkSpec ( (region!=NULL),global_defaults.chipType,global_defaults.xtalk_name);
  xtalk_execute.CloseOverPointers (region, &xtalk_spec,&my_beads, my_regions, &time_c, math_poiss, &my_scratch, &my_flow, &my_trace, use_vectorization);
}


void BkgModel::InitLevMar()
{
  // create matrix packing object(s)
  //Note:  scratch-space is used directly by the matrix packer objects to get the derivatives
  // so this object >must< persist in order to be used by the fit control object in the Lev_Mar fit.
  // Allocate directly the annoying pointers that the lev-mar object uses for control
  fit_control.AllocPackers (my_scratch.scratchSpace, global_defaults.no_RatioDrift_fit_first_20_flows, my_scratch.bead_flow_t, time_c.npts);
  // allocate a multiflow levmar fitter
  lev_mar_fit = new MultiFlowLevMar (*this);
  lev_mar_fit->lm_state.SetNonIntegerPenalty (global_defaults.clonal_call_scale,global_defaults.clonal_call_penalty,MAGIC_MAX_CLONAL_HP_LEVEL);
  lev_mar_fit->lm_state.AllocateBeadFitState (my_beads.numLBeads);
  lev_mar_fit->lm_state.AssignBeadsToRegionGroups();
}

//Destructor
BkgModel::~BkgModel()
{
  DebugFileClose();
  if (lev_mar_fit!=NULL) delete lev_mar_fit;
  if (axion_fit !=NULL) delete axion_fit;
  if (refine_fit !=NULL) delete refine_fit;
  if (trace_bkg_adj !=NULL) delete trace_bkg_adj;
  free (global_state.dirName); //@TODO: why is this a local copy?
  if (replay != NULL) delete replay;
}

void BkgModel::SetTimeAndEmphasis()
{
  time_c.choose_time = global_defaults.choose_time;
  time_c.SetUpTime (my_trace.imgFrames,t_mid_nuc_start,global_defaults.time_start_detail, global_defaults.time_stop_detail, global_defaults.time_left_avg);

  // check the points that we need
  if (CENSOR_ZERO_EMPHASIS>0)
  {
    EmphasisClass trial_emphasis;

    // assuming that our t_mid_nuc estimation is decent
    // see what the emphasis functions needed for "detailed" results are
    // and assume the "coarse" blank emphasis function will work out fine.
    trial_emphasis.SetDefaultValues (global_defaults.emp,global_defaults.emphasis_ampl_default, global_defaults.emphasis_width_default);
    trial_emphasis.SetupEmphasisTiming (time_c.npts, time_c.frames_per_point,time_c.frameNumber);
    trial_emphasis.CurrentEmphasis (t_mid_nuc_start, FINEXEMPHASIS);
//    int old_pts = time_c.npts;
    time_c.npts = trial_emphasis.ReportUnusedPoints (CENSOR_THRESHOLD, MIN_CENSOR); // threshold the points for the number actually needed by emphasis

    // don't bother monitoring this now
    //printf ("Saved: %f = %d of %d\n", (1.0*time_c.npts) / (1.0*old_pts), time_c.npts, old_pts);
    // now give the emphasis data structure (and everything else) using the "used" number of points
  }
  emphasis_data.SetDefaultValues (global_defaults.emp,global_defaults.emphasis_ampl_default, global_defaults.emphasis_width_default);
  emphasis_data.SetupEmphasisTiming (time_c.npts, time_c.frames_per_point,time_c.frameNumber);
  emphasis_data.CurrentEmphasis (t_mid_nuc_start, FINEXEMPHASIS);
}

// t_offset_beads = nuc_shape.sigma
// t_offset_empty = 4.0
void BkgModel::RezeroTraces (float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty, int fnum)
{
  // do these values make sense for offsets in RezeroBeads???
  my_trace.RezeroBeads (t_start, t_mid_nuc-t_offset_beads, fnum);
  emptytrace->RezeroReference (t_start, t_mid_nuc-t_offset_empty, fnum);
}

void BkgModel::RezeroTracesAllFlows (float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty)
{
  my_trace.RezeroBeadsAllFlows (t_start, t_mid_nuc-t_offset_beads);
  emptytrace->RezeroReferenceAllFlows (t_start, t_mid_nuc-t_offset_empty);
}

void BkgModel::AllocFitBuffers()
{
  // scratch space will be used directly by fit-control derivatives
  // so we need to make sure these structures match
  my_scratch.Allocate (time_c.npts,fit_control.fitParams.NumSteps);
  my_regions->AllocScratch (time_c.npts);
}

void BkgModel::AllocTraceBuffers()
{
  // now do the traces set up for time compression
  my_trace.Allocate (my_flow.numfb,NUMFB*time_c.npts,my_beads.numLBeads);
  my_trace.time_cp = &time_c; // point to the global time compression

}

/*--------------------------------- Allocation section done ------------------------------------------------------*/

/*--------------------------------Control analysis flow start ----------------------------------------------------*/

void BkgModel::BootUpModel (double &elapsed_time,Timer &fit_timer)
{
// what are emphasis functions here?
// they are set to maximum default 0 for each bead - make explicit so we >know<
  my_beads.AssignEmphasisForAllBeads (0);
  fit_timer.restart();
  //fit_control.FitInitial
  lev_mar_fit->MultiFlowSpecializedLevMarFitParameters (2, 5, fit_control.FitInitial, fit_control.FitRegionInit1, 0.1);
  elapsed_time += fit_timer.elapsed();

  my_beads.my_mean_copy_count = my_beads.KeyNormalizeReads (true);

  fit_timer.restart();
  lev_mar_fit->MultiFlowSpecializedLevMarFitParameters (1, 2, fit_control.FitInitial, fit_control.FitRegionInit1, 0.1);
  elapsed_time += fit_timer.elapsed();

  my_beads.my_mean_copy_count = my_beads.KeyNormalizeReads (true);

  fit_timer.restart();
  lev_mar_fit->MultiFlowSpecializedLevMarFitParameters (1, 2, fit_control.FitInitial, fit_control.FitRegionInit1, 0.1);
  elapsed_time += fit_timer.elapsed();

}

void BkgModel::PickRepresentativeHighQualityWells()
{
  my_beads.my_mean_copy_count = my_beads.KeyNormalizeReads (false); // retain fitted key values for snr purposes
  // use only high quality beads from now on when regional fitting
  if (global_defaults.ssq_filter>0.0f)
    my_beads.LowSSQRatioBeadsAreLowQuality (global_defaults.ssq_filter);
  my_beads.LowCopyBeadsAreLowQuality (my_beads.my_mean_copy_count);
  my_beads.KeyNormalizeReads (true); // force all beads to read the "true key" in the key flows
}

void BkgModel::PostKeyFit (double &elapsed_time, Timer &fit_timer)
{
  my_beads.AssignEmphasisForAllBeads (0);

  fit_timer.restart();
  RezeroTracesAllFlows (time_c.time_start, GetTypicalMidNucTime (& (my_regions->rp.nuc_shape)), my_regions->rp.nuc_shape.sigma, MAGIC_OFFSET_FOR_EMPTY_TRACE);

  lev_mar_fit->MultiFlowSpecializedLevMarFitParameters (1, 15, fit_control.FitPostKey, fit_control.FitRegionInit2, 1.0);
  elapsed_time += fit_timer.elapsed();

  RezeroTracesAllFlows (time_c.time_start, GetTypicalMidNucTime (& (my_regions->rp.nuc_shape)), my_regions->rp.nuc_shape.sigma, MAGIC_OFFSET_FOR_EMPTY_TRACE);
  // try to use the first non-key cycle to help normalize everything and
  // identify incorporation model parameters
  //if(do_clonal_filter)
  //    my_beads.FindClonalReads();

  fit_timer.restart();
  lev_mar_fit->MultiFlowSpecializedLevMarFitParameters (1, 15, fit_control.FitPostKey, fit_control.FitRegionFull, 1.0, 5);
  elapsed_time += fit_timer.elapsed();

}

void BkgModel::ApproximateDarkMatter()
{
  my_beads.AssignEmphasisForAllBeads (0);
  // now figure out whatever remaining error there is in the fit, on average
  axion_fit->CalculateDarkMatter (FIRSTNFLOWSFORERRORCALC,my_beads.high_quality, lev_mar_fit->lm_state.residual, lev_mar_fit->lm_state.avg_resid*2.0);
}



void BkgModel::FitAmplitudeAndDarkMatter (double &elapsed_time, Timer &fit_timer)
{
  my_beads.AssignEmphasisForAllBeads (emphasis_data.numEv-1);
  emphasis_data.CurrentEmphasis (GetTypicalMidNucTime (& (my_regions->rp.nuc_shape)), CRUDEXEMPHASIS); // why is this not per nuc?

  //@TODO: should I be skipping low-quality bead refits here because we'll be getting their amplitudes in the refinement phase?
  fit_timer.restart();
  lev_mar_fit->MultiFlowSpecializedLevMarFitParameters (1, 15, fit_control.FitAmpl, fit_control.FitRegionSlimErr, 10.0);
  elapsed_time += fit_timer.elapsed();
}

void BkgModel::FitWellParametersConditionalOnRegion (double &elapsed_time, Timer &fit_timer)
{
  fit_timer.restart();
  lev_mar_fit->lm_state.skip_beads = false;  // make sure we fit every bead here

  // catch up on wells we skipped while fitting regions
  // i.e. fit wells conditional on regional parameters now

// something wrong here(!)
//@TODO:  why does fitting the wells here wreck the fit?  Region parameters should be just fine...
  //lev_mar_fit->MultiFlowSpecializedLevMarFitParameters(3,0,fit_control.FitPostKey, fit_control.DontFitRegion, 1.0);

  // don't need to compensate for wells that didn't get dark matter adjusted, because we refit amplitude downstream anyway
  //lev_mar_fit->MultiFlowSpecializedLevMarFitParameters (5, 0, fit_control.FitAmpl, fit_control.DontFitRegion, 10.0); // refitting amplitude anyway

  elapsed_time += fit_timer.elapsed();
}

void BkgModel::FitInitialFlowBlockModel (double &elapsed_time, Timer &fit_timer)
{
  BootUpModel (elapsed_time,fit_timer);

  // now that we know something about the wells, select a good subset
  // by any filtration we think is good
  PickRepresentativeHighQualityWells();

  lev_mar_fit->lm_state.skip_beads=false;
  // these two should be do-able only on representative wells
  PostKeyFit (elapsed_time, fit_timer);


  ApproximateDarkMatter();

  lev_mar_fit->lm_state.skip_beads=true;

  FitAmplitudeAndDarkMatter (elapsed_time, fit_timer);

  // catch up well parameters on wells we didn't use in the regional estimates
  if (lev_mar_fit->lm_state.skip_beads)
    FitWellParametersConditionalOnRegion (elapsed_time, fit_timer);


#ifdef TUNE_INCORP_PARAMS_DEBUG
  DumpRegionParameters();
#endif

  my_regions->RestrictRatioDrift();
}

void BkgModel::GuessCrudeAmplitude (double &elapsed_time, Timer &fit_timer)
{
  emphasis_data.CurrentEmphasis (GetTypicalMidNucTime (& (my_regions->rp.nuc_shape)), CRUDEXEMPHASIS);

  //@TODO: should I skip low-quality beads because I'm going to refit them later and I only need this for regional parameters?
  fit_timer.restart();
  //@TODO: badly organized data still requiring too much stuff passed around
  // I blame MultiFlowModel wrapper functions that don't exist
  my_beads.AssignEmphasisForAllBeads (0);
  my_search.ParasitePointers (math_poiss, &my_trace,emptytrace,&my_scratch,my_regions,&time_c,&my_flow,&emphasis_data);
  if (global_defaults.generic_test_flag)
    my_search.BinarySearchAmplitude (my_beads, 0.5f,true); // apply search method to current bead list - wrong OO?  Method on bead list?
  else
    my_search.ProjectionSearchAmplitude (my_beads, false); // need all our speed
  //my_search.GoldenSectionAmplitude(my_beads);
  elapsed_time += fit_timer.elapsed();

}

void BkgModel::FitTimeVaryingRegion (double &elapsed_time, Timer &fit_timer)
{
  fit_timer.restart();
  // >NOW< we allow any emphasis level given our crude estimates for emphasis
  my_beads.AssignEmphasisForAllBeads (emphasis_data.numEv-1);
  lev_mar_fit->lm_state.skip_beads = true;  // we're not updating well parameters anyway, but make it look good
  lev_mar_fit->MultiFlowSpecializedLevMarFitParameters (0, 4, fit_control.DontFitWells, fit_control.FitRegionSlim, 1.0);
  lev_mar_fit->lm_state.skip_beads = false;
  elapsed_time += fit_timer.elapsed();
}


void BkgModel::FitLaterBlockOfFlows (int flow, double &elapsed_time, Timer &fit_timer)
{
  GuessCrudeAmplitude (elapsed_time,fit_timer);
  replay->FitTimeVaryingRegion (flow, elapsed_time,fit_timer);
  // FitTimeVaryingRegion (elapsed_time,fit_timer);
}


void BkgModel::RefineAmplitudeEstimates (double &elapsed_time, Timer &fit_timer)
{
  // in either case, finish by fitting amplitude to finalized regional parameters
  emphasis_data.CurrentEmphasis (GetTypicalMidNucTime (& (my_regions->rp.nuc_shape)), FINEXEMPHASIS);
  fit_timer.restart();
  refine_fit->FitAmplitudePerFlow (); // just like GPU code now
  double fit_ampl_time = fit_timer.elapsed();
  elapsed_time += fit_ampl_time;
}

void BkgModel::CPUFitModelForBlockOfFlows (int flow, bool last, bool learning)
{
  Timer fit_timer;
  Timer total_timer;
  double elapsed_time = 0;

  my_beads.ResetLocalBeadParams(); // start off with no data for amplitude/kmult
  my_regions->ResetLocalRegionParams(); // start off with no per flow time shifts

  if ( (flow+1) <=my_flow.numfb) // fit regions for this set of flows
  {
    FitInitialFlowBlockModel (elapsed_time,fit_timer);
  }
  else
  {
    FitLaterBlockOfFlows (flow, elapsed_time, fit_timer);
  }

  // background correct all the beads right in the fg_buffer...sorry
  //trace_bkg_adj->BackgroundCorrectAllBeadsInPlace();
  
  // now do a final amplitude estimation on each bead
  RefineAmplitudeEstimates (elapsed_time,fit_timer);

  printf (".");
  fflush (stdout);

  DumpRegionTrace (my_debug.region_trace_file);
}

void BkgModel::UpdateBeadStatusAfterFit (int flow)
{
  if (do_clonal_filter)
  {
    vector<float> copy_mult (NUMFB);
    for (int f=0; f<NUMFB; ++f)
      copy_mult[f] = CalculateCopyDrift (my_regions->rp, my_flow.buff_flow[f]);

    my_beads.UpdateClonalFilter (flow, copy_mult);
  }

  my_beads.WriteCorruptedToMask (region, global_state.bfmask);

  // turn off while debugging blackbird chip
  // my_beads.ZeroOutPins(region, bfmask, &pinnedInFlow, flow);
}


#ifdef ION_COMPILE_CUDA

void BkgModel::GPUBootUpModel (BkgModelCuda* bkg_model_cuda, double &elapsed_time,Timer &fit_timer)
{

  fit_timer.restart();
  my_beads.AssignEmphasisForAllBeads (0);
  bkg_model_cuda->MultiFlowSpecializedLevMarFitParameters (2, 5, fit_control.FitInitial, fit_control.FitRegionInit1, 0.1);
  elapsed_time += fit_timer.elapsed();

  my_beads.my_mean_copy_count = my_beads.KeyNormalizeReads (true);

  fit_timer.restart();
  bkg_model_cuda->MultiFlowSpecializedLevMarFitParameters (1, 2, fit_control.FitInitial, fit_control.FitRegionInit1, 0.1);
  elapsed_time += fit_timer.elapsed();

  my_beads.my_mean_copy_count = my_beads.KeyNormalizeReads (true);

  fit_timer.restart();
  bkg_model_cuda->MultiFlowSpecializedLevMarFitParameters (1, 2, fit_control.FitInitial, fit_control.FitRegionInit1, 0.1);
  elapsed_time += fit_timer.elapsed();

}

void BkgModel::GPUPostKeyFit (BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer)
{
  fit_timer.restart();
  my_beads.AssignEmphasisForAllBeads (0);

  RezeroTracesAllFlows (time_c.time_start, GetTypicalMidNucTime (& (my_regions->rp.nuc_shape)),  my_regions->rp.nuc_shape.sigma, MAGIC_OFFSET_FOR_EMPTY_TRACE);

  bkg_model_cuda->MultiFlowSpecializedLevMarFitParameters (1, 15, fit_control.FitPostKey, fit_control.FitRegionInit2, 1.0);
  elapsed_time += fit_timer.elapsed();

  RezeroTracesAllFlows (time_c.time_start, GetTypicalMidNucTime (& (my_regions->rp.nuc_shape)),  my_regions->rp.nuc_shape.sigma, MAGIC_OFFSET_FOR_EMPTY_TRACE);
  // try to use the first non-key cycle to help normalize everything and
  // identify incorporation model parameters
  //if(do_clonal_filter)
  //    my_beads.FindClonalReads();

  fit_timer.restart();
  bkg_model_cuda->MultiFlowSpecializedLevMarFitParameters (1, 15, fit_control.FitPostKey, fit_control.FitRegionFull, 1.0, 5);
  elapsed_time += fit_timer.elapsed();
}

void BkgModel::GPUFitAmplitudeAndDarkMatter (BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer)
{
  my_beads.AssignEmphasisForAllBeads (emphasis_data.numEv-1);
  emphasis_data.CurrentEmphasis (GetTypicalMidNucTime (& (my_regions->rp.nuc_shape)), CRUDEXEMPHASIS);

  fit_timer.restart();
  bkg_model_cuda->MultiFlowSpecializedLevMarFitParameters (1, 15, fit_control.FitAmpl, fit_control.FitRegionSlimErr, 10.0);
  elapsed_time += fit_timer.elapsed();
}

void BkgModel::GPUFitInitialFlowBlockModel (BkgModelCuda* bkg_model_cuda, double &elapsed_time, Timer &fit_timer)
{
  GPUBootUpModel (bkg_model_cuda, elapsed_time,fit_timer);

  PickRepresentativeHighQualityWells(); // same as CPU logic

  lev_mar_fit->lm_state.skip_beads=false;

  GPUPostKeyFit (bkg_model_cuda,elapsed_time, fit_timer);

  ApproximateDarkMatter();

  lev_mar_fit->lm_state.skip_beads=true;

  GPUFitAmplitudeAndDarkMatter (bkg_model_cuda,elapsed_time, fit_timer);

  lev_mar_fit->lm_state.skip_beads=false;
  // need catchup individual well parameters here
  // GPUFitWellParametersConditionalOnRegion (elapsed_time, fit_timer);

#ifdef TUNE_INCORP_PARAMS_DEBUG
  DumpRegionParameters();
#endif

  my_regions->RestrictRatioDrift();
}

void BkgModel::GPUGuessCrudeAmplitude (BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer)
{
  emphasis_data.CurrentEmphasis (GetTypicalMidNucTime (& (my_regions->rp.nuc_shape)), CRUDEXEMPHASIS);
  my_beads.AssignEmphasisForAllBeads (0);
  fit_timer.restart();
  bkg_model_cuda->BinarySearchAmplitude (0.5,true);
  elapsed_time += fit_timer.elapsed();

}

void BkgModel::GPUFitTimeVaryingRegion (BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer)
{
  fit_timer.restart();
  my_beads.AssignEmphasisForAllBeads (emphasis_data.numEv-1);
  lev_mar_fit->lm_state.skip_beads = true;
  bkg_model_cuda->MultiFlowSpecializedLevMarFitParameters (0, 4, fit_control.DontFitWells, fit_control.FitRegionSlim, 1.0);
  lev_mar_fit->lm_state.skip_beads = false;
  elapsed_time += fit_timer.elapsed();
}

void BkgModel::GPUFitLaterBlockOfFlows (BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer)
{
  GPUGuessCrudeAmplitude (bkg_model_cuda,elapsed_time,fit_timer);
  GPUFitTimeVaryingRegion (bkg_model_cuda,elapsed_time,fit_timer);
}


void BkgModel::GPURefineAmplitudeEstimates (BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer)
{
  fit_timer.restart();
  // in either case, finish by fitting amplitude to finalized regional parameters
  emphasis_data.CurrentEmphasis (GetTypicalMidNucTime (& (my_regions->rp.nuc_shape)), FINEXEMPHASIS);
  bkg_model_cuda->FitAmplitudePerFlow();
  elapsed_time += fit_timer.elapsed();

}
void BkgModel::GPUFitModelForBlockOfFlows (int flow, bool last, bool learning)
{
  Timer fit_timer;
  Timer total_timer;
  double elapsed_time = 0;

  my_beads.ResetLocalBeadParams(); // start off with no data for amplitude/kmult
  my_regions->ResetLocalRegionParams(); // start off with no per flow time shifts
  //
  // Create the GPU object
  //
  // The timing of this creation is critical to a thread-safe implementation. It is a
  // requirement of CUDA that GPU memory must be allocated, accessed, and freed by the
  // same thread.  If this requirement is violated, you will see "invalid argument"
  // errors at memory management calls and computation kernels. A future work around will
  // involve using the CUDA context model that NVIDIA has proposed.
  //
  BkgModelCuda* bkg_model_cuda = NULL;

  bkg_model_cuda = new BkgModelCuda (*this, fit_control.fitParams.NumSteps, fit_control.fitParams.Steps);

  if ( (flow+1) <=my_flow.numfb) // fit regions for this set of flows
  {
    GPUFitInitialFlowBlockModel (bkg_model_cuda,elapsed_time,fit_timer);
  }
  else
  {
    GPUFitLaterBlockOfFlows (bkg_model_cuda,elapsed_time, fit_timer);
  }

  GPURefineAmplitudeEstimates (bkg_model_cuda, elapsed_time,fit_timer);

  printf (".");
  fflush (stdout);

  DumpRegionTrace (my_debug.region_trace_file);

  if (bkg_model_cuda != NULL)
    delete bkg_model_cuda;

  //printf("r(%d,%d) Livebeads: %d, GPU fit process complete in: %f (%f in MultiFlowSpecializedLevMarFitParameters)\n",
  //       region->col, region->row, my_beads.numLBeads, total_timer.elapsed(), elapsed_time);

}

#endif  // CUDA code isolation




/*--------------------------------Control analysis flow end ----------------------------------------------------*/

// this puts our answers into the data structures where they belong
// should be the only point of contact with the external world, but isn't
void BkgModel::WriteAnswersToWells (int iFlowBuffer)
{
  // make absolutely sure we're upt to date
  my_regions->rp.copy_multiplier[iFlowBuffer] = CalculateCopyDrift (my_regions->rp, my_flow.buff_flow[iFlowBuffer]);
  //Write one flow's data to 1.wells
  for (int ibd=0;ibd < my_beads.numLBeads;ibd++)
  {
    float val = my_beads.params_nn[ibd].Ampl[iFlowBuffer] * my_beads.params_nn[ibd].Copies * my_regions->rp.copy_multiplier[iFlowBuffer];
    int x = my_beads.params_nn[ibd].x+region->col;
    int y = my_beads.params_nn[ibd].y+region->row;

    global_state.rawWells->WriteFlowgram (my_flow.buff_flow[iFlowBuffer], x, y, val);

  }
}

//@TODO: this is not actually a bkgmodel function but a function of my_beads?
void BkgModel::WriteBeadParameterstoDataCubes (int iFlowBuffer, bool last)
{
  if (mPtrs == NULL)
    return;

  int flow = my_flow.buff_flow[iFlowBuffer];
  for (int ibd=0;ibd < my_beads.numLBeads;ibd++)
  {
    int x = my_beads.params_nn[ibd].x+region->col;
    int y = my_beads.params_nn[ibd].y+region->row;
    struct bead_params &p = my_beads.params_nn[ibd];

    // use copyCube_element to copy DataCube element in BkgModel::WriteBeadParameterstoDataCubes
    mPtrs->copyCube_element (mPtrs->mAmpl,x,y,flow,p.Ampl[iFlowBuffer]);
    mPtrs->copyCube_element (mPtrs->mBeadDC,x,y,flow,my_trace.fg_dc_offset[ibd*NUMFB+iFlowBuffer]);
    mPtrs->copyCube_element (mPtrs->mKMult,x,y,flow,p.kmult[iFlowBuffer]);

    mPtrs->copyCube_element (mPtrs->mBeadInitParam,x,y,0,p.Copies);
    mPtrs->copyCube_element (mPtrs->mBeadInitParam,x,y,1,p.R);
    mPtrs->copyCube_element (mPtrs->mBeadInitParam,x,y,2,p.dmult);
    mPtrs->copyCube_element (mPtrs->mBeadInitParam,x,y,3,p.gain);

    if ( (iFlowBuffer+1) ==NUMFB || last)
    {
      mPtrs->copyMatrix_element (mPtrs->mBeadFblk_avgErr,x,y,p.my_state.avg_err);
      mPtrs->copyMatrix_element (mPtrs->mBeadFblk_clonal,x,y,p.my_state.clonal_read?1:0);
      mPtrs->copyMatrix_element (mPtrs->mBeadFblk_corrupt,x,y,p.my_state.corrupt?1:0);
    }
  }
}


void BkgModel::WriteDebugWells (int iFlowBuffer)
{
  if (my_debug.BkgDbg1 != NULL || my_debug.BkgDbg2 !=NULL || my_debug.BkgDebugKmult!=NULL)
  {

    for (int ibd=0;ibd < my_beads.numLBeads;ibd++)
    {
      int x = my_beads.params_nn[ibd].x+region->col;
      int y = my_beads.params_nn[ibd].y+region->row;


      // debug parameter output, if necessary
      float etbR,tauB;
      int NucID = my_flow.flow_ndx_map[iFlowBuffer];
      //Everything uses the same functions to compute this so we're compatible
      etbR = AdjustEmptyToBeadRatioForFlow (my_beads.params_nn[ibd].R,& (my_regions->rp),NucID,my_flow.buff_flow[iFlowBuffer]);
      tauB = ComputeTauBfromEmptyUsingRegionLinearModel (& (my_regions->rp),etbR);
      if (my_debug.BkgDbg1!=NULL)
        my_debug.BkgDbg1->WriteFlowgram (my_flow.buff_flow[iFlowBuffer],x,y,tauB);
      if (my_debug.BkgDbg2!=NULL)
        my_debug.BkgDbg2->WriteFlowgram (my_flow.buff_flow[iFlowBuffer],x,y,etbR);
      if (my_debug.BkgDebugKmult!=NULL)
        my_debug.BkgDebugKmult->WriteFlowgram (my_flow.buff_flow[iFlowBuffer],x,y,my_beads.params_nn[ibd].kmult[iFlowBuffer]); // kmultiplier to go with amplitude in main
    }
  }

}


void BkgModel::SendErrorVectorToHDF5 (bead_params *p, error_track &err_t)
{
  if (mPtrs !=NULL)
  {
    int x = p->x+region->col;
    int y = p->y+region->row;
    for (int fnum=0; fnum<NUMFB; fnum++)
    {
      // use copyCube_element to copy DataCube element to mResError
      mPtrs->copyCube_element (mPtrs->mResError,x,y,my_flow.buff_flow[fnum],err_t.mean_residual_error[fnum]);
    }
  }
}








/*--------------------------------debugging routines start ----------------------------------------------------*/

// debugging functions down here in the darkness
// so I don't have to read them every time I wander through the code
void BkgModel::MultiFlowComputeTotalSignalTrace (float *fval,struct bead_params *p,struct reg_params *reg_p,float *sbg)
{
  float sbg_local[my_scratch.bead_flow_t];

  // allow the background to be passed in to save processing
  if (sbg == NULL)
  {
    emptytrace->GetShiftedBkg (reg_p->tshift, time_c, sbg_local);
    sbg = sbg_local;
  }
  //@TODO possibly same for nuc_rise step
  MultiFlowComputeCumulativeIncorporationSignal (p,reg_p,my_scratch.ival,*my_regions,my_scratch.cur_bead_block,time_c,my_flow,math_poiss);
  MultiFlowComputeIncorporationPlusBackground (fval,p,reg_p,my_scratch.ival,sbg,*my_regions,my_scratch.cur_buffer_block,time_c,my_flow,use_vectorization, my_scratch.bead_flow_t);
}

void BkgModel::DebugFileOpen (void)
{
  if (region == NULL)
    return;

  char *fname;
  int name_len = strlen (global_state.dirName) + strlen (BKG_MODEL_DEBUG_DIR) + 64;
  struct stat fstatus;
  int         status;

  fname = new char[name_len];

  snprintf (fname,name_len,"%s%s",global_state.dirName,BKG_MODEL_DEBUG_DIR);
  status = stat (fname,&fstatus);

  if (status != 0)
  {
    // directory does not exist yet, create it
    mkdir (fname,S_IRWXU | S_IRWXG | S_IRWXO);
  }

  snprintf (fname,name_len,"%s%sdatax%dy%d.txt",global_state.dirName,BKG_MODEL_DEBUG_DIR,region->col,region->row);
  fopen_s (&my_debug.data_dbg_file,fname, "wt");

  snprintf (fname,name_len,"%s%stracex%dy%d.txt",global_state.dirName,BKG_MODEL_DEBUG_DIR,region->col,region->row);
  fopen_s (&my_debug.trace_dbg_file,fname, "wt");
  fprintf (my_debug.trace_dbg_file,"Background Fit Object Created x = %d, y = %d\n",region->col,region->row);
  fflush (my_debug.trace_dbg_file);

#ifdef FIT_ITERATION_DEBUG_TRACE
  snprintf (fname,name_len,"%s%siterx%dy%d.txt",global_state.dirName,BKG_MODEL_DEBUG_DIR,region->col,region->row);
  fopen_s (&my_debug.iter_dbg_file,fname,"wt");
#endif

  snprintf (fname,name_len,"%s/reg_tracex%dy%d.txt",global_state.dirName,region->col,region->row);
  fopen_s (&my_debug.region_trace_file,fname, "wt");

  snprintf (fname,name_len,"%s/reg_0mer_tracex%dy%d.txt",global_state.dirName,region->col,region->row);
  fopen_s (&my_debug.region_0mer_trace_file,fname, "wt");

  snprintf (fname,name_len,"%s/reg_1mer_tracex%dy%d.txt",global_state.dirName,region->col,region->row);
  fopen_s (&my_debug.region_1mer_trace_file,fname, "wt");

  delete [] fname;
}

void BkgModel::DebugFileClose (void)
{
#ifdef FIT_ITERATION_DEBUG_TRACE
  if (my_debug.iter_dbg_file != NULL)
    fclose (my_debug.iter_dbg_file);
#endif

  if (my_debug.trace_dbg_file != NULL)
    fclose (my_debug.trace_dbg_file);

  if (my_debug.data_dbg_file != NULL)
    fclose (my_debug.data_dbg_file);

  if (my_debug.region_trace_file != NULL)
    fclose (my_debug.region_trace_file);

  if (my_debug.region_0mer_trace_file != NULL)
    fclose (my_debug.region_0mer_trace_file);

  if (my_debug.region_1mer_trace_file != NULL)
    fclose (my_debug.region_1mer_trace_file);
}

void BkgModel::DebugIterations()
{
  DumpRegionParamsCSV (my_debug.iter_dbg_file,& (my_regions->rp));
  my_beads.DumpAllBeadsCSV (my_debug.iter_dbg_file);
}

void BkgModel::DebugBeadIteration (bead_params &eval_params, reg_params &eval_rp, int iter, int ibd)
{
  fprintf (my_debug.trace_dbg_file,"iter:% 3d,(% 5.3f, % 5.3f,% 6.2f, % 2.1f, % 5.3f, % 5.3f, % 5.3f) ",
           iter,eval_params.gain,eval_params.Copies,lev_mar_fit->lm_state.residual[ibd],eval_rp.nuc_shape.sigma,eval_params.R,my_regions->rp.RatioDrift,my_regions->rp.CopyDrift);
  fprintf (my_debug.trace_dbg_file,"% 3.2f,% 3.2f,% 3.2f,% 3.2f,",
           eval_params.Ampl[0],eval_params.Ampl[1],eval_params.Ampl[2],eval_params.Ampl[3]);
  fprintf (my_debug.trace_dbg_file,"% 3.2f,% 3.2f,% 3.2f,% 3.2f,",
           eval_params.Ampl[4],eval_params.Ampl[5],eval_params.Ampl[6],eval_params.Ampl[7]);
  fprintf (my_debug.trace_dbg_file,"% 2.1f,% 2.1f,% 2.1f,% 2.1f,",
           GetTypicalMidNucTime (&eval_rp.nuc_shape),GetTypicalMidNucTime (&eval_rp.nuc_shape),GetTypicalMidNucTime (&eval_rp.nuc_shape),GetTypicalMidNucTime (&eval_rp.nuc_shape)); // wrong! should be delayed

  fprintf (my_debug.trace_dbg_file,"(% 5.3f, % 5.3f, % 5.3f, % 5.3f) ",
           eval_rp.d[0],eval_rp.d[1],eval_rp.d[2],eval_rp.d[3]);
  fprintf (my_debug.trace_dbg_file,"(% 5.3f, % 5.3f, % 5.3f, % 5.3f, % 5.3f\n) ",
           eval_rp.krate[0],eval_rp.krate[1],eval_rp.krate[2],eval_rp.krate[3],eval_rp.sens);

  fflush (my_debug.trace_dbg_file);
}

void BkgModel::DumpRegionParameters()
{
  if (region != NULL)
    printf ("r(x,y) d(T,A,C,G) k(T,A,C,G)=(%d,%d) (%5.3f, %5.3f, %5.3f, %5.3f) (%5.3f, %5.3f, %5.3f, %5.3f) (%5.3f, %5.3f, %5.3f, %5.3f) %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f\n",
            region->col,region->row,
            my_regions->rp.d[0],my_regions->rp.d[1],my_regions->rp.d[2],my_regions->rp.d[3],
            my_regions->rp.krate[0],my_regions->rp.krate[1],my_regions->rp.krate[2],my_regions->rp.krate[3],
            my_regions->rp.kmax[0],my_regions->rp.kmax[1],my_regions->rp.kmax[2],my_regions->rp.kmax[3],
            lev_mar_fit->lm_state.avg_resid,my_regions->rp.tshift,my_regions->rp.tau_R_m,my_regions->rp.tau_R_o,my_regions->rp.nuc_shape.sigma,GetTypicalMidNucTime (& (my_regions->rp.nuc_shape)));
  if (region != NULL)
    printf ("---(%d,%d) (%5.3f, %5.3f, %5.3f, %5.3f) (%5.3f, %5.3f, %5.3f, %5.3f)\n",
            region->col,region->row,
            my_regions->rp.nuc_shape.t_mid_nuc_delay[0],my_regions->rp.nuc_shape.t_mid_nuc_delay[1],my_regions->rp.nuc_shape.t_mid_nuc_delay[2],my_regions->rp.nuc_shape.t_mid_nuc_delay[3],
            my_regions->rp.nuc_shape.sigma_mult[0],my_regions->rp.nuc_shape.sigma_mult[1],my_regions->rp.nuc_shape.sigma_mult[2],my_regions->rp.nuc_shape.sigma_mult[3]);
}

#define DUMP_N_VALUES(key,format,var,n) \
{\
    fprintf(my_fp,"%s",key);\
    for (int i=0;i<n;i++) fprintf(my_fp,format,var[i]);\
    fprintf(my_fp,"\n");\
}

// outputs a full trace of a region
// 1.) all region-wide parameters are output along w/ text identifiers so that they can be
//     parsed by downstream tools
// 2.) All pertinent data for a subset of live wells is output.  Per-well data is put in the trace file
//     in blocks of 20 flows (just as they are processed).  All parameters are output w/ text
//     identifiers so that they can be parsed later.  If there are more than 1000 live wells, some will
//     automatically be skipped in order to limit the debug output to ~1000 wells total.  Wells are skipped
//     in a sparse fashion in order to evenly represent the region
//
#define MAX_REGION_TRACE_WELLS  1000
void BkgModel::DumpRegionTrace (FILE *my_fp)
{
  if (my_fp)
  {
    // if this is the first time through, dump all the region-wide parameters that don't change
    if (my_flow.buff_flow[0] == 0)
    {
      fprintf (my_fp,"reg_row:\t%d\n",region->row);
      fprintf (my_fp,"reg_col:\t%d\n",region->col);
      fprintf (my_fp,"npts:\t%d\n",time_c.npts);
      fprintf (my_fp,"tshift:\t%f\n",my_regions->rp.tshift);
      fprintf (my_fp,"tau_R_m:\t%f\n",my_regions->rp.tau_R_m);
      fprintf (my_fp,"tau_R_o:\t%f\n",my_regions->rp.tau_R_o);
      fprintf (my_fp,"sigma:\t%f\n",my_regions->rp.nuc_shape.sigma);
      DUMP_N_VALUES ("krate:","\t%f",my_regions->rp.krate,NUMNUC);
      float tmp[NUMNUC];
      for (int i=0;i<NUMNUC;i++) tmp[i]=my_regions->rp.d[i];
      DUMP_N_VALUES ("d:","\t%f",tmp,NUMNUC);
      DUMP_N_VALUES ("kmax:","\t%f",my_regions->rp.kmax,NUMNUC);
      fprintf (my_fp,"sens:\t%f\n",my_regions->rp.sens);
      DUMP_N_VALUES ("NucModifyRatio:","\t%f",my_regions->rp.NucModifyRatio,NUMNUC);
      DUMP_N_VALUES ("ftimes:","\t%f",time_c.frameNumber,time_c.npts);
      DUMP_N_VALUES ("error_term:","\t%f",my_regions->missing_mass.dark_matter_compensator,my_regions->missing_mass.nuc_flow_t);  // one time_c.npts-long term per nuc
      fprintf (my_fp,"end_section:\n");
      // we don't output t_mid_nuc, CopyDrift, or RatioDrift here, because those can change every block of 20 flows
    }
// TODO: dump computed parameters taht are functions of apparently "basic" parameters
// because the routines to compute them are "hidden" in the code
    // now dump parameters and data that can be unique for every block of 20 flows
    DUMP_N_VALUES ("flows:","\t%d",my_flow.buff_flow,NUMFB);
    fprintf (my_fp,"CopyDrift:\t%f\n",my_regions->rp.CopyDrift);
    fprintf (my_fp,"RatioDrift:\t%f\n",my_regions->rp.RatioDrift);
    fprintf (my_fp,"t_mid_nuc:\t%f\n",GetTypicalMidNucTime (& (my_regions->rp.nuc_shape)));
    DUMP_N_VALUES ("nnum:","\t%d",my_flow.flow_ndx_map,NUMFB);
    fprintf (my_fp,"end_section:\n");

    float tmp[my_scratch.bead_flow_t];
    struct reg_params eval_rp = my_regions->rp;
//    float my_xtflux[my_scratch.bead_flow_t];
    float sbg[my_scratch.bead_flow_t];

    emptytrace->GetShiftedBkg (my_regions->rp.tshift, time_c, sbg);
    float skip_num = 1.0;
    if (my_beads.numLBeads > MAX_REGION_TRACE_WELLS)
    {
      skip_num = (float) (my_beads.numLBeads) /1000.0;
    }

    float skip_next = 0.0;
    for (int ibd=0;ibd < my_beads.numLBeads;ibd++)
    {
      if ( (float) ibd >= skip_next)
        skip_next += skip_num;
      else
        continue;

      struct bead_params *p = &my_beads.params_nn[ibd];
      fprintf (my_fp,"bead_row:%d\n",my_beads.params_nn[ibd].y);
      fprintf (my_fp,"bead_col:%d\n",my_beads.params_nn[ibd].x);
      float R_tmp[NUMFB],tau_tmp[NUMFB];
      for (int i=0;i < NUMFB;i++)
        R_tmp[i] = AdjustEmptyToBeadRatioForFlow (p->R,& (my_regions->rp),my_flow.flow_ndx_map[i],my_flow.buff_flow[i]);
      DUMP_N_VALUES ("R:","\t%f",R_tmp,NUMFB);
      for (int i=0;i < NUMFB;i++)
        tau_tmp[i] = ComputeTauBfromEmptyUsingRegionLinearModel (& (my_regions->rp),R_tmp[i]);
      DUMP_N_VALUES ("tau:","\t%f",tau_tmp,NUMFB);
      fprintf (my_fp,"P:%f\n",p->Copies);
      fprintf (my_fp,"gain:%f\n",p->gain);

      fprintf (my_fp,"dmult:%f\n",p->dmult);
//        fprintf(my_fp,"in_cnt:%d\n",in_cnt[my_beads.params_nn[ibd].y*region->w+my_beads.params_nn[ibd].x]);
      DUMP_N_VALUES ("Ampl:","\t%f",p->Ampl,NUMFB);
      DUMP_N_VALUES ("kmult:","\t%f",p->kmult,NUMFB);

      // run the model
      MultiFlowComputeTotalSignalTrace (my_scratch.fval,&my_beads.params_nn[ibd],& (my_regions->rp),sbg);

      struct bead_params eval_params = my_beads.params_nn[ibd];
      memset (eval_params.Ampl,0,sizeof (eval_params.Ampl));

      // calculate the model with all 0-mers to get synthetic background by itself
      MultiFlowComputeTotalSignalTrace (tmp,&eval_params,&eval_rp,sbg);

      // calculate proton flux from neighbors
      // why did this get commented out?????...it broke the below code that relies on my_xtflux being initialized!!!
      //CalcXtalkFlux(ibd,my_xtflux);

      // output values
      float tmp_signal[my_scratch.bead_flow_t];
      my_trace.MultiFlowFillSignalForBead (tmp_signal,ibd);
      DUMP_N_VALUES ("raw_data:","\t%0.1f", tmp_signal,my_scratch.bead_flow_t);
      DUMP_N_VALUES ("fit_data:","\t%.1f",my_scratch.fval,my_scratch.bead_flow_t);
      DUMP_N_VALUES ("avg_empty:","\t%.1f",sbg,my_scratch.bead_flow_t);
      DUMP_N_VALUES ("background:","\t%.1f",tmp,my_scratch.bead_flow_t);
//  temporary comment out until I figure out why CalcXtalkFlux is no longer called above
//      DUMP_N_VALUES ("xtalk:","\t%.1f",my_xtflux,my_scratch.bead_flow_t);
    }
    fprintf (my_fp,"end_section:\n");

    fflush (my_fp);
  } // end file exists
}

void BkgModel::doDebug (char *name, float diff, float *output)
{
#if 1
  (void) name;
  (void) diff;
  (void) output;
#else
  double avg=0;

//@TODO this is out of synch with the proper dealing with flows
  for (int i=0;i<my_scratch.bead_flow_t;i++)
    avg += *output++;

  avg /= my_scratch.bead_flow_t;
  printf ("%s(%f) %.1lf ",name,diff,avg);
#endif
}

char *BkgModel::findName (float *ptr)
{
  int i;
  for (i=0;i<fit_control.fitParams.NumSteps;i++)
  {
    if (ptr >= fit_control.fitParams.Steps[i].ptr && ptr < (fit_control.fitParams.Steps[i].ptr + my_scratch.bead_flow_t))
      return (fit_control.fitParams.Steps[i].name);
  }
  return (NULL);
}


//
// Copies image data from input arrays and does the fitting (if enough data has been supplied)
//  This version of ProcessImage does not require Image objects or Mask objects to work properly
//
bool BkgModel::ProcessImage (short *img, short *bkg, int flow, bool last, bool learning, bool use_gpu)
{
  if (my_beads.numLBeads == 0)
    return false;

  // keep track of which flow # is in which buffer
  my_flow.buff_flow[my_flow.flowBufferWritePos] = flow;
  // also keep track of which nucleotide is associated with each flow
  my_flow.flow_ndx_map[my_flow.flowBufferWritePos] = global_defaults.glob_flow_ndx_map[flow%global_defaults.flow_order_len];

  // emptytrace.FillEmptyTraceFromBuffer (bkg,my_flow.flowBufferWritePos);
  my_trace.FillBeadTraceFromBuffer (img,my_flow.flowBufferWritePos);
  // emptytrace.PrecomputeBackgroundSlopeForDeriv (my_flow.flowBufferWritePos);

  my_flow.Increment();

  if ( (my_flow.flowBufferCount >= my_flow.numfb) || last)
  {
    // do the fit
    FitModelForBlockOfFlows (flow,last,learning, use_gpu);

    while (my_flow.flowBufferCount > 0)
    {
      my_flow.Decrement();
    }
  }

  return false;
}

// interface to running in weird standalone mode from torrentR
int BkgModel::GetModelEvaluation (int iWell,struct bead_params *p,struct reg_params *rp,
                                  float **fg,float **bg,float **feval,float **isig,float **pf)
{
  SA.allocate (my_scratch.bead_flow_t);

  // make sure iWell is in the correct range
  if ( (iWell < 0) || (iWell > my_beads.numLBeads))
  {
    return (-1);
  }

  *fg = SA.fg;
  *bg = SA.bg;
  *feval = SA.feval;
  *isig = SA.isig;
  *pf = SA.pf;

  emptytrace->GetShiftedBkg (rp->tshift, time_c, SA.bg);
  // iterate over all data points doing the right thing

  //@TODO put nuc rise here
  MultiFlowComputeCumulativeIncorporationSignal (p,rp,SA.pf,*my_regions,my_scratch.cur_bead_block,time_c,my_flow,math_poiss);
  MultiFlowComputeIncorporationPlusBackground (SA.feval,p,rp,SA.pf,SA.bg,*my_regions,my_scratch.cur_buffer_block,time_c,my_flow, use_vectorization, my_scratch.bead_flow_t);

  // easiest way to get the incorporation signal w/ background subtracted off is to calculate
  // the function with Ampl=0 and subtract that from feval
  struct bead_params p_tmp = *p;
  memset (&p_tmp,0,sizeof (p_tmp.Ampl));
  MultiFlowComputeTotalSignalTrace (SA.isig,&p_tmp,rp,SA.bg);

  for (int fnum=0; fnum<my_flow.numfb; fnum++)
  {
    for (int i=0;i < time_c.npts;i++)
    {
      int tmp_index = i+fnum*time_c.npts;
      SA.isig[tmp_index] = SA.feval[tmp_index] - SA.isig[tmp_index];
    }
  }
  my_scratch.FillObserved (my_trace,iWell);

  for (int fnum=0;fnum < my_flow.numfb;fnum++)
  {

    for (int i=0;i < time_c.npts;i++)
      SA.fg[i+fnum*time_c.npts] = my_scratch.observed[i+fnum*time_c.npts];
  }

  return (my_scratch.bead_flow_t);
}


void BkgModel::DumpTimeAndEmphasisByRegion (FILE *my_fp)
{
  // this will be a dumb file format
  // each line has x,y, type, hash, data per time point
  // dump timing
  int i = 0;
  fprintf (my_fp,"%d\t%d\t", region->col, region->row);
  fprintf (my_fp,"time\t%d\t",time_c.npts);

  for (i=0; i<time_c.npts; i++)
    fprintf (my_fp,"%f\t",time_c.frameNumber[i]);
  for (; i<MAX_COMPRESSED_FRAMES; i++)
    fprintf (my_fp,"0.0\t");
  fprintf (my_fp,"\n");
  fprintf (my_fp,"%d\t%d\t", region->col, region->row);
  fprintf (my_fp,"frames_per_point\t%d\t",time_c.npts);
  for (i=0; i<time_c.npts; i++)
    fprintf (my_fp,"%d\t", time_c.frames_per_point[i]);
  for (; i<MAX_COMPRESSED_FRAMES; i++)
    fprintf (my_fp,"0\t");
  fprintf (my_fp,"\n");
  // dump emphasis
  for (int el=0; el<emphasis_data.numEv; el++)
  {
    fprintf (my_fp,"%d\t%d\t", region->col, region->row);
    fprintf (my_fp,"em\t%d\t",el);
    for (i=0; i<time_c.npts; i++)
      fprintf (my_fp,"%f\t",emphasis_data.EmphasisVectorByHomopolymer[el][i]);
    for (; i<MAX_COMPRESSED_FRAMES; i++)
      fprintf (my_fp,"0.0\t");
    fprintf (my_fp,"\n");
  }
}

// accessors for hdf5 dump of emptytrace object
int BkgModel::get_emptytrace_imgFrames()
{
  return ( (emptyTraceTracker->GetEmptyTrace (*region))->imgFrames);
}

float *BkgModel::get_emptytrace_bg_buffers()
{
  return ( (emptyTraceTracker->GetEmptyTrace (*region))->bg_buffers);
}

float *BkgModel::get_emptytrace_bg_dc_offset()
{
  return ( (emptyTraceTracker->GetEmptyTrace (*region))->bg_dc_offset);
}

void BkgModel::DumpEmptyTrace (FILE *my_fp)
{
  assert (emptyTraceTracker != NULL);   //sanity
  if (region!=NULL)
  {
    (emptyTraceTracker->GetEmptyTrace (*region))->DumpEmptyTrace (my_fp,region->col,region->row);
    // @TODO: investigate why doesn't this work???
    //emptytrace->DumpEmptyTrace (my_fp,region->col,region->row);
  }
}

void BkgModel::DumpTimeAndEmphasisByRegionH5 (int reg)
{
  if (mPtrs==NULL)
    return;

  ION_ASSERT (emphasis_data.numEv <= MAX_HPLEN+1, "emphasis_data.numEv > MAX_HPLEN+1");
  //ION_ASSERT(time_c.npts <= MAX_COMPRESSED_FRAMES, "time_c.npts > MAX_COMPRESSED_FRAMES");
  int npts = min (time_c.npts, MAX_COMPRESSED_FRAMES);

  // use copyCube_element to copy DataCube element to mEmphasisParam in BkgModel::DumpTimeAndEmphasisByRegionH5
  for (int hp=0; hp<emphasis_data.numEv; hp++)
  {
    for (int t=0; t< npts; t++)
      mPtrs->copyCube_element (mPtrs->mEmphasisParam,reg,hp,t,emphasis_data.EmphasisVectorByHomopolymer[hp][t]);
    for (int t=npts; t< MAX_COMPRESSED_FRAMES; t++)
      mPtrs->copyCube_element (mPtrs->mEmphasisParam,reg,hp,t,0); // pad 0's, memset faster here?
  }

  for (int hp=emphasis_data.numEv; hp<MAX_HPLEN+1; hp++)
  {
    for (int t=0; t< MAX_COMPRESSED_FRAMES; t++)
    {
      mPtrs->copyCube_element (mPtrs->mEmphasisParam,reg,hp,t,0); // pad 0's, memset faster here?
    }
  }

}



