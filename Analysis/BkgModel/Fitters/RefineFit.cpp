/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "RefineFit.h"
#include "DNTPRiseModel.h"
#include "TraceCorrector.h"

RefineFit::RefineFit (SignalProcessingMasterFitter &_bkg) :
  bkg (_bkg)
{
  InitSingleFlowFit();
  my_exp_tail_fit.SanityClause(bkg.global_defaults.signal_process_control.exp_tail_bkg_limit, bkg.global_defaults.signal_process_control.exp_tail_bkg_lower);
}


RefineFit::~RefineFit()
{

}

void RefineFit::InitSingleFlowFit()
{
  // fix up the fitter for refining amplitude/krate
  my_single_fit.AllocLevMar (bkg.region_data->time_c,bkg.math_poiss,
                             bkg.global_defaults.signal_process_control.single_flow_master.kmult_low_limit, bkg.global_defaults.signal_process_control.single_flow_master.kmult_hi_limit,
                             bkg.global_defaults.signal_process_control.AmplLowerLimit );
  my_single_fit.FillDecisionThreshold (bkg.global_defaults.signal_process_control.single_flow_master.krate_adj_limit);

   my_single_fit.gauss_newton_fit = bkg.global_defaults.signal_process_control.fit_gauss_newton;
   my_single_fit.always_slow = bkg.global_defaults.signal_process_control.always_start_slow;
}





void RefineFit::FitAmplitudePerBeadPerFlow (int ibd, NucStep &cache_step, int flow_block_size, int flow_block_start)
{

  BeadParams *p = &bkg.region_data->my_beads.params_nn[ibd];
  float block_signal_corrected[bkg.region_data->my_scratch.bead_flow_t];
  float block_signal_predicted[bkg.region_data->my_scratch.bead_flow_t];
  float block_signal_original[bkg.region_data->my_scratch.bead_flow_t]; // what were we before? correct-in-place erases information
  float block_signal_sbg[bkg.region_data->my_scratch.bead_flow_t]; // what background did we actually use: may not be stable in case of bugs


  float washoutThreshold = bkg.getWashoutThreshold();
  int washoutFlowDetection = bkg.getWashoutFlowDetection();

  error_track err_t; // temporary store fitting information, including errors for this bead this flow

  bkg.trace_bkg_adj->ReturnBackgroundCorrectedSignal (block_signal_corrected, block_signal_original, block_signal_sbg, ibd, flow_block_size, flow_block_start);

  if (bkg.global_defaults.signal_process_control.exp_tail_fit){

    // for example: this adjustment does extra background munging and needs debugging badly
    if (bkg.global_defaults.signal_process_control.exp_tail_bkg_adj){
      my_exp_tail_fit.AdjustBackground(block_signal_corrected,err_t.bkg_leakage, bkg.region_data->my_scratch.shifted_bkg,
                                       p,&bkg.region_data->my_regions.rp,
                                       bkg.region_data_extras.my_flow,bkg.region_data->time_c, flow_block_size, flow_block_start);

  }
  }

  for (int fnum=0;fnum < flow_block_size;fnum++)
  {
    float evect[bkg.region_data->time_c.npts()];
    bkg.region_data->emphasis_data.CustomEmphasis (evect, p->Ampl[fnum]);
    float *signal_corrected = &block_signal_corrected[fnum*bkg.region_data->time_c.npts()];
    float *signal_predicted = &block_signal_predicted[fnum*bkg.region_data->time_c.npts()];
    int NucID = bkg.region_data_extras.my_flow->flow_ndx_map[fnum];
    err_t.fit_type[fnum] =
        my_single_fit.FitOneFlow (fnum,evect,p,&err_t, signal_corrected,signal_predicted,
                                  NucID,cache_step.NucFineStep (fnum),cache_step.i_start_fine_step[fnum],
                                  flow_block_start,bkg.region_data->emphasis_data,bkg.region_data->my_regions);
    err_t.t_mid_nuc_actual[fnum] = cache_step.t_mid_nuc_actual[fnum];
    err_t.t_sigma_actual[fnum] = cache_step.t_sigma_actual[fnum];
  }


  // do things with my error vector for this bead

  // now detect corruption & store average error
  p->DetectCorruption (err_t, washoutThreshold, washoutFlowDetection, flow_block_size);
  // update error here to be passed to later block of flows
  // don't keep individual flow errors because we're surprisingly tight on memory
  p->UpdateCumulativeAvgError (err_t, flow_block_start + flow_block_size, flow_block_size); // current flow reached, 1-based
  if (bkg.global_defaults.signal_process_control.stop_beads){
  bool trip_flag = bkg.region_data->my_beads.UpdateSTOP(ibd, flow_block_size);
  // debugging, decrease number of high quality wells track when it happens
  if ((trip_flag) & (ibd % 143==0)){ // sample debug beads
    printf("STOP: %d %d %d %d %d %f\n", bkg.region_data->region->index,ibd, p->x, p->y, flow_block_start, bkg.region_data->my_beads.decay_ssq[ibd]);
  }
  }

  // send prediction to hdf5 if necessary
  CrazyDumpToHDF5(p,ibd,block_signal_predicted, block_signal_corrected, block_signal_original, block_signal_sbg, err_t, flow_block_start );
}

// this HDF5 dump has gotten way out of hand
// note we should >never< have both BeadParams and ibd
void RefineFit::CrazyDumpToHDF5(BeadParams *p, int ibd, float * block_signal_predicted, float *block_signal_corrected,
                                float *block_signal_original, float *block_signal_sbg,  error_track &err_t, int flow_block_start)
{
  if (! bkg.global_state.hasPointers())
    return;

  // if necessary, send the errors to HDF5
  // this sends bead error vector to bead_param.h5

  //bkg.global_state.SendErrorVectorToHDF5 (p,err_t, bkg.region_data->region,
  //                                        *bkg.region_data_extras.my_flow, flow_block_start );
  bkg.global_state.SendErrorVectorToWells (p,err_t, bkg.region_data->region,
                                            *bkg.region_data_extras.my_flow, flow_block_start );

  // sends debug-beads to region_param.h5, 1 per region
  CrazyDumpDebugBeads(p,ibd,block_signal_predicted, block_signal_corrected,  block_signal_original, block_signal_sbg, err_t, flow_block_start );

  // sends information to trace.h5
  // send beads_bestRegion all data
  if (bkg.region_data->isBestRegion)
  {
    CrazyDumpBestRegion(p,ibd,block_signal_predicted, block_signal_corrected, block_signal_original, block_signal_sbg,  err_t, flow_block_start );
  }
  // sends to trace.h5, specified beads all over
  CrazyDumpXyFlow(p,ibd,block_signal_predicted, block_signal_corrected,  block_signal_original, block_signal_sbg, err_t, flow_block_start );
  // sends to trace.h5, sampled beads in each region
  CrazyDumpRegionSamples(p,ibd,block_signal_predicted, block_signal_corrected, block_signal_original, block_signal_sbg,  err_t, flow_block_start );
}



void RefineFit::CrazyDumpDebugBeads(BeadParams *p, int ibd, float * block_signal_predicted, float *block_signal_corrected,  float *block_signal_original, float *block_signal_sbg,error_track &err_t, int flow_block_start)
{
  int max_frames = bkg.global_defaults.signal_process_control.get_max_frames();
  bkg.global_state.SendPredictedToHDF5(ibd, block_signal_predicted, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
  bkg.global_state.SendCorrectedToHDF5(ibd, block_signal_corrected, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
  bkg.global_state.SendXtalkToHDF5(ibd, bkg.region_data->my_scratch.cur_xtflux_block, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
}

void RefineFit::CrazyDumpXyFlow(BeadParams *p, int ibd, float * block_signal_predicted, float *block_signal_corrected, float *block_signal_original, float *block_signal_sbg,error_track &err_t, int flow_block_start)
{
  int max_frames = bkg.global_defaults.signal_process_control.get_max_frames();
  bkg.global_state.SendXyflow_Location_Keys_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendXyflow_Timeframe_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
  bkg.global_state.SendXyflow_R_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendXyflow_SP_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendXyflow_GainSens_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendXyflow_FitType_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, err_t.fit_type, flow_block_start );
  bkg.global_state.SendXyflow_Taub_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, err_t, flow_block_start );
  bkg.global_state.SendXyflow_Dmult_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendXyflow_Kmult_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendXyflow_HPlen_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendXyflow_MM_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendXyflow_Location_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendXyflow_Amplitude_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendXyflow_Residual_ToHDF5(ibd, err_t, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendXyflow_Predicted_ToHDF5(ibd, block_signal_predicted, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
  bkg.global_state.SendXyflow_Corrected_ToHDF5(ibd, block_signal_corrected, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
  bkg.global_state.SendXyflow_Predicted_Keys_ToHDF5(ibd, block_signal_predicted, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
  bkg.global_state.SendXyflow_Corrected_Keys_ToHDF5(ibd, block_signal_corrected, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
}

void RefineFit::CrazyDumpBestRegion(BeadParams *p, int ibd, float * block_signal_predicted, float *block_signal_corrected,  float *block_signal_original, float *block_signal_sbg,error_track &err_t, int flow_block_start)
{
  int max_frames = bkg.global_defaults.signal_process_control.get_max_frames();
  //printf("RefineFit::FitAmplitudePerBeadPerFlow... region %d is the bestRegion...sending bestRegion data to HDF5 for bead %d\n",bkg.region_data->region->index,ibd);
  // per bead parameters, in the first flow block only
  if (flow_block_start==0)
  {
    bkg.global_state.SendBestRegion_TimeframeToHDF5(*bkg.region_data, bkg.region_data_extras, max_frames);
    bkg.global_state.SendBestRegion_LocationToHDF5(ibd, *bkg.region_data );
    bkg.global_state.SendBestRegion_GainSensToHDF5(ibd, *bkg.region_data );
    bkg.global_state.SendBestRegion_DmultToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
    bkg.global_state.SendBestRegion_SPToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
    bkg.global_state.SendBestRegion_RToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  }

  // traces
  bkg.global_state.SendBestRegion_PredictedToHDF5(ibd, block_signal_predicted, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
  bkg.global_state.SendBestRegion_CorrectedToHDF5(ibd, block_signal_corrected, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
  bkg.global_state.SendBestRegion_OriginalToHDF5(ibd, block_signal_original, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
  bkg.global_state.SendBestRegion_SBGToHDF5(ibd, block_signal_sbg, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );

  // per bead per flow numbers - direct single-flow fit
  bkg.global_state.SendBestRegion_AmplitudeToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendBestRegion_KmultToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  // incidental data of debugging importance, per bead per flow
  // in theory these are computable, but there are enough bugs that reporting the actual values is useful
  bkg.global_state.SendBestRegion_TaubToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, err_t, flow_block_start );
  bkg.global_state.SendBestRegion_etbRToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, err_t, flow_block_start );
  // fit to data
  bkg.global_state.SendBestRegion_BkgLeakageToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, err_t, flow_block_start );
  bkg.global_state.SendBestRegion_InitAkToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, err_t, flow_block_start );
  bkg.global_state.SendBestRegion_TMS_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, err_t, flow_block_start );
  bkg.global_state.SendBestRegion_ResidualToHDF5(ibd, err_t, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendBestRegion_FitType_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, err_t.fit_type, flow_block_start );
  bkg.global_state.SendBestRegion_Converged_ToHDF5(ibd,  *bkg.region_data, bkg.region_data_extras, err_t, flow_block_start );
}

void RefineFit::CrazyDumpRegionSamples(BeadParams *p, int ibd, float * block_signal_predicted, float *block_signal_corrected,  float *block_signal_original, float *block_signal_sbg, error_track &err_t, int flow_block_start)
{
  int max_frames = bkg.global_defaults.signal_process_control.get_max_frames();
  int nAssigned = bkg.region_data->assign_sampleIndex();
  if (nAssigned>0 && bkg.region_data->isRegionSample(ibd)) // true = force regionSampleIndex assignment if not assigned
  {
    bkg.global_state.set_numLiveBeads(bkg.region_data->GetNumLiveBeads());
    bkg.global_state.set_nSampleOut(bkg.region_data->get_region_nSamples());
    int idx = bkg.region_data->get_sampleIndex(ibd);
    bkg.global_state.set_sampleIndex(idx);
    if (flow_block_start==0)
    {
      bkg.global_state.SendRegionSamples_TimeframeToHDF5(*bkg.region_data, bkg.region_data_extras, max_frames);
      bkg.global_state.SendRegionSamples_LocationToHDF5(ibd, *bkg.region_data );
      bkg.global_state.SendRegionSamples_RegionParamsToHDF5(ibd, *bkg.region_data );
      bkg.global_state.SendRegionSamples_GainSensToHDF5(ibd, *bkg.region_data );
      bkg.global_state.SendRegionSamples_DmultToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
      bkg.global_state.SendRegionSamples_SPToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
      bkg.global_state.SendRegionSamples_RToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
    }
    // traces
    bkg.global_state.SendRegionSamples_PredictedToHDF5(ibd, block_signal_predicted, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
    bkg.global_state.SendRegionSamples_CorrectedToHDF5(ibd, block_signal_corrected, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
    bkg.global_state.SendRegionSamples_OriginalToHDF5(ibd, block_signal_original, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
    bkg.global_state.SendRegionSamples_SBGToHDF5(ibd, block_signal_sbg, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
     // per bead per flow
    bkg.global_state.SendRegionSamples_AmplitudeToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
    bkg.global_state.SendRegionSamples_KmultToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
    // incidental data
    bkg.global_state.SendRegionSamples_TaubToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, err_t, flow_block_start );
    bkg.global_state.SendRegionSamples_etbRToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, err_t, flow_block_start );
    // other data
    bkg.global_state.SendRegionSamples_BkgLeakageToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, err_t, flow_block_start );
    bkg.global_state.SendRegionSamples_InitAkToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, err_t, flow_block_start );
    bkg.global_state.SendRegionSamples_ResidualToHDF5(ibd, err_t, *bkg.region_data, bkg.region_data_extras, flow_block_start );
    bkg.global_state.SendRegionSamples_FitType_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, err_t.fit_type, flow_block_start );
    bkg.global_state.SendRegionSamples_Converged_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, err_t, flow_block_start );
    // timing - match to beads for utility in reconstruction
    bkg.global_state.SendRegionSamples_TMS_ToHDF5(*bkg.region_data, bkg.region_data_extras,  flow_block_start );
   }
}



// fits all wells one flow at a time, using a LevMarFitter derived class
// only the amplitude term is fit
void RefineFit::FitAmplitudePerFlow ( int flow_block_size, int flow_block_start )
{
  bkg.region_data->my_regions.cache_step.CalculateNucRiseFineStep (&bkg.region_data->my_regions.rp,bkg.region_data->time_c,
                                                                   *bkg.region_data_extras.my_flow); // the same for the whole region because time-shift happens per well
  bkg.region_data->my_regions.cache_step.CalculateNucRiseCoarseStep (&bkg.region_data->my_regions.rp,bkg.region_data->time_c,*bkg.region_data_extras.my_flow); // use for xtalk

  bkg.region_data->my_scratch.FillShiftedBkg (*bkg.region_data->emptytrace, bkg.region_data->my_regions.rp.tshift, bkg.region_data->time_c, true, flow_block_size);

  if (bkg.trace_xtalk_spec.simple_model)
    bkg.trace_xtalk_execute.ComputeTypicalCrossTalk(bkg.trace_xtalk_execute.my_generic_xtalk, NULL, flow_block_size, flow_block_start ); // get the generic value set


  
  my_single_fit.SetUpEmphasisForLevMarOptimizer(&(bkg.region_data->emphasis_data));

  // allocate mResError
  //bkg.global_state.AllocDataCubeResErr(*bkg.region_data);
  for (int ibd = 0;ibd < bkg.region_data->my_beads.numLBeads;ibd++)
  {
    if (bkg.region_data->my_beads.params_nn[ibd].FitBeadLogic () or bkg.region_data->isRegionSample(ibd) or bkg.region_data->isBestRegion or ibd==bkg.region_data->my_beads.DEBUG_BEAD) // make sure debugging beads are preserved
      FitAmplitudePerBeadPerFlow (ibd,bkg.region_data->my_regions.cache_step, flow_block_size, flow_block_start);
  }

}
