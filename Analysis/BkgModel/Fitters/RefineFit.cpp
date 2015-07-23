/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "RefineFit.h"
#include "DNTPRiseModel.h"
#include "TraceCorrector.h"

RefineFit::RefineFit (SignalProcessingMasterFitter &_bkg) :
    bkg (_bkg)
{
   InitSingleFlowFit();
}

RefineFit::~RefineFit()
{

}

void RefineFit::InitSingleFlowFit()
{
  // fix up the fitter for refining amplitude/krate
  my_single_fit.AllocLevMar (bkg.region_data->time_c,bkg.math_poiss,
                             bkg.global_defaults.signal_process_control.single_flow_master.dampen_kmult,bkg.global_defaults.signal_process_control.var_kmult_only,
                             bkg.global_defaults.signal_process_control.single_flow_master.kmult_low_limit, bkg.global_defaults.signal_process_control.single_flow_master.kmult_hi_limit,
                             bkg.global_defaults.signal_process_control.AmplLowerLimit );
  my_single_fit.FillDecisionThreshold (bkg.global_defaults.signal_process_control.single_flow_master.krate_adj_limit);
  my_single_fit.SetRetryLimit (bkg.global_defaults.signal_process_control.single_flow_fit_max_retry);
  my_single_fit.fit_alt = bkg.global_defaults.signal_process_control.fit_alternate;
  my_single_fit.gauss_newton_fit = bkg.global_defaults.signal_process_control.fit_gauss_newton;
}

void RefineFit::SupplyMultiFlowSignal (float *block_signal_corrected, int ibd, int flow_block_size,
    int flow_block_start
  )
{
  if (bkg.region_data->my_trace.AlreadyAdjusted())
  {
    BeadParams *p = &bkg.region_data->my_beads.params_nn[ibd];
    reg_params *reg_p = &bkg.region_data->my_regions.rp;

    bkg.region_data->my_trace.MultiFlowFillSignalForBead (block_signal_corrected, ibd, flow_block_size);
    // calculate proton flux from neighbors
    bkg.region_data->my_scratch.ResetXtalkToZero();
//  bkg.xtalk_execute.ExecuteXtalkFlux (ibd,bkg.region_data->my_scratch.cur_xtflux_block);

    // set up current bead parameters by flow
    MathModel::FillBufferParamsBlockFlows (bkg.region_data_extras.cur_buffer_block,p,reg_p,
      bkg.region_data_extras.my_flow->flow_ndx_map,
      flow_block_start, flow_block_size);
    MathModel::FillIncorporationParamsBlockFlows (bkg.region_data_extras.cur_bead_block, p,reg_p,
      bkg.region_data_extras.my_flow->flow_ndx_map,
      flow_block_start, flow_block_size);
    // make my corrected signal
    // subtract computed zeromer signal
    // uses parameters above

    // should do trace-corrector here?
//  MultiCorrectBeadBkg (block_signal_corrected,p,
//                       bkg.region_data->my_scratch,*bkg.region_data_extras.my_flow,bkg.region_data->time_c,*bkg.region_data->my_regions,bkg.region_data->my_scratch.shifted_bkg,bkg.use_vectorization);
  }
  else
  {
    bkg.trace_bkg_adj->ReturnBackgroundCorrectedSignal (block_signal_corrected, ibd, flow_block_size, flow_block_start);
  }
}


//@TODO:  Revert local bead correction here
// For refining "time" shifts, we want to use sampled beads, so don't do bkg subtraction on everything and munge the buffers.
void RefineFit::FitAmplitudePerBeadPerFlow (int ibd, NucStep &cache_step, int flow_block_size, int flow_block_start)
{
  int fitType[flow_block_size];
  BeadParams *p = &bkg.region_data->my_beads.params_nn[ibd];
  float block_signal_corrected[bkg.region_data->my_scratch.bead_flow_t];
  float block_signal_predicted[bkg.region_data->my_scratch.bead_flow_t];
  float washoutThreshold = bkg.getWashoutThreshold();
  int washoutFlowDetection = bkg.getWashoutFlowDetection();

  SupplyMultiFlowSignal (block_signal_corrected, ibd, flow_block_size, flow_block_start);

  if (bkg.global_defaults.signal_process_control.exp_tail_fit)
     my_exp_tail_fit.CorrectTraces(block_signal_corrected,bkg.region_data->my_scratch.shifted_bkg,
        p,&bkg.region_data->my_regions.rp,
        bkg.region_data_extras.my_flow,bkg.region_data->time_c, flow_block_size, flow_block_start);

  error_track err_t; // temporary store errors for this bead this flow
  for (int fnum=0;fnum < flow_block_size;fnum++)
  {
    float evect[bkg.region_data->time_c.npts()];
    bkg.region_data->emphasis_data.CustomEmphasis (evect, p->Ampl[fnum]);
    float *signal_corrected = &block_signal_corrected[fnum*bkg.region_data->time_c.npts()];
    float *signal_predicted = &block_signal_predicted[fnum*bkg.region_data->time_c.npts()];
    int NucID = bkg.region_data_extras.my_flow->flow_ndx_map[fnum];
	  fitType[fnum] = 
      my_single_fit.FitOneFlow (fnum,evect,p,&err_t, signal_corrected,signal_predicted, 
        NucID,cache_step.NucFineStep (fnum),cache_step.i_start_fine_step[fnum],
        flow_block_start,bkg.region_data->time_c,bkg.region_data->emphasis_data,bkg.region_data->my_regions);
  }
  //int reg = bkg.region_data->region->index;
  //printf("RefineFit::FitAmplitudePerBeadPerFlow... (r,b)=(%d,%d) predicted[10]=%f corrected[10]=%f\n",reg,ibd,block_signal_predicted[10],block_signal_corrected[10]);
  // note: predicted[0:7]=0 most of the time, my_start=8 or 9

// do things with my error vector for this bead

  // now detect corruption & store average error
  p->DetectCorruption (err_t, washoutThreshold, washoutFlowDetection, flow_block_size);
  // update error here to be passed to later block of flows
  // don't keep individual flow errors because we're surprisingly tight on memory
  p->UpdateCumulativeAvgError (err_t, flow_block_start + flow_block_size, flow_block_size); // current flow reached, 1-based

  // send prediction to hdf5 if necessary
  CrazyDumpToHDF5(p,ibd,block_signal_predicted, block_signal_corrected, fitType, err_t, flow_block_start );
}

// this HDF5 dump has gotten way out of hand
// note we should >never< have both BeadParams and ibd
void RefineFit::CrazyDumpToHDF5(BeadParams *p, int ibd, float * block_signal_predicted, float *block_signal_corrected, int *fitType, error_track &err_t, int flow_block_start)
{
  // if necessary, send the errors to HDF5
  bkg.global_state.SendErrorVectorToHDF5 (p,err_t, bkg.region_data->region,
  *bkg.region_data_extras.my_flow, flow_block_start );
  int max_frames = bkg.global_defaults.signal_process_control.get_max_frames();
  //printf ("RefineFit::FitAmplitudePerBeadPerFlow... max_frames=%d\n",max_frames);
  bkg.global_state.SendPredictedToHDF5(ibd, block_signal_predicted, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
  bkg.global_state.SendCorrectedToHDF5(ibd, block_signal_corrected, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );

  bkg.global_state.SendXyflow_Location_Keys_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendXyflow_Timeframe_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
  bkg.global_state.SendXyflow_R_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendXyflow_SP_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendXyflow_GainSens_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  bkg.global_state.SendXyflow_FitType_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, fitType, flow_block_start );
  bkg.global_state.SendXyflow_Taub_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
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

  bkg.global_state.SendXtalkToHDF5(ibd, bkg.region_data->my_scratch.cur_xtflux_block, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
  // send beads_bestRegion predicted to hdf5 if necessary
  if (bkg.region_data->isBestRegion)
  {
    //printf("RefineFit::FitAmplitudePerBeadPerFlow... region %d is the bestRegion...sending bestRegion data to HDF5 for bead %d\n",bkg.region_data->region->index,ibd);
      bkg.global_state.SendBestRegion_TimeframeToHDF5(*bkg.region_data, bkg.region_data_extras, max_frames);
      bkg.global_state.SendBestRegion_LocationToHDF5(ibd, *bkg.region_data );
      bkg.global_state.SendBestRegion_GainSensToHDF5(ibd, *bkg.region_data );
      bkg.global_state.SendBestRegion_PredictedToHDF5(ibd, block_signal_predicted, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
      bkg.global_state.SendBestRegion_CorrectedToHDF5(ibd, block_signal_corrected, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
      bkg.global_state.SendBestRegion_AmplitudeToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
      bkg.global_state.SendBestRegion_ResidualToHDF5(ibd, err_t, *bkg.region_data, bkg.region_data_extras, flow_block_start );
      bkg.global_state.SendBestRegion_FitType_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, fitType, flow_block_start );
      bkg.global_state.SendBestRegion_KmultToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
      bkg.global_state.SendBestRegion_DmultToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
      bkg.global_state.SendBestRegion_TaubToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
      bkg.global_state.SendBestRegion_SPToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
      bkg.global_state.SendBestRegion_RToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
  }

  if (bkg.region_data->isRegionCenter(ibd))
  {
      bkg.global_state.SendRegionCenter_TimeframeToHDF5(*bkg.region_data, bkg.region_data_extras, max_frames);
      bkg.global_state.SendRegionCenter_LocationToHDF5(ibd, *bkg.region_data );
      bkg.global_state.SendRegionCenter_RegionParamsToHDF5(ibd, *bkg.region_data );
      bkg.global_state.SendRegionCenter_GainSensToHDF5(ibd, *bkg.region_data );
      bkg.global_state.SendRegionCenter_AmplitudeToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
      bkg.global_state.SendRegionCenter_ResidualToHDF5(ibd, err_t, *bkg.region_data, bkg.region_data_extras, flow_block_start );
      bkg.global_state.SendRegionCenter_FitType_ToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, fitType, flow_block_start );
      bkg.global_state.SendRegionCenter_KmultToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
      bkg.global_state.SendRegionCenter_DmultToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
      bkg.global_state.SendRegionCenter_TaubToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
      bkg.global_state.SendRegionCenter_SPToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
      bkg.global_state.SendRegionCenter_RToHDF5(ibd, *bkg.region_data, bkg.region_data_extras, flow_block_start );
      bkg.global_state.SendRegionCenter_PredictedToHDF5(ibd, block_signal_predicted, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
      bkg.global_state.SendRegionCenter_CorrectedToHDF5(ibd, block_signal_corrected, *bkg.region_data, bkg.region_data_extras, max_frames, flow_block_start );
  }
}

//  This is testing what the cross-talk routine produces
void RefineFit::TestXtalk(int flow_block_size, int flow_block_start)
{
  float xtalk_signal_typical[bkg.region_data->my_scratch.bead_flow_t];
  float xtalk_source_typical[bkg.region_data->my_scratch.bead_flow_t];
  
  bkg.trace_xtalk_execute.ComputeTypicalCrossTalk(xtalk_signal_typical, xtalk_source_typical, flow_block_size, flow_block_start);
  FILE *fp;
  char file_name[100];
  sprintf(file_name,"xtalk.typical.%04d.txt", flow_block_start + flow_block_size );
  fp = fopen(file_name,"at");
  int npts = bkg.region_data->time_c.npts();
  for (int fnum=0; fnum<flow_block_size; fnum++)
  {
    fprintf(fp,"%d\t%d\t%d\txtalk\t", bkg.region_data->region->row, bkg.region_data->region->col, 
      flow_block_start + fnum );
    int i;
    for ( i=0; i<npts; i++)
      fprintf(fp, "%f\t", xtalk_signal_typical[i+fnum*npts]);
    //for (; i<MAX_COMPRESSED_FRAMES; i++)
    //  fprintf(fp,"0.0\t");
    fprintf(fp,"\n");
    
    fprintf(fp,"%d\t%d\t%d\tsource\t", bkg.region_data->region->row, bkg.region_data->region->col, 
      flow_block_start + fnum );
    for ( i=0; i<npts; i++)
      fprintf(fp, "%f\t", xtalk_source_typical[i+fnum*npts]);
    //for (; i<MAX_COMPRESSED_FRAMES; i++)
    //  fprintf(fp,"0.0\t");
    fprintf(fp,"\n");
  }
  fclose(fp);
  
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

  for (int ibd = 0;ibd < bkg.region_data->my_beads.numLBeads;ibd++)
  {
    if (bkg.region_data->my_beads.params_nn[ibd].FitBeadLogic ())
      FitAmplitudePerBeadPerFlow (ibd,bkg.region_data->my_regions.cache_step, flow_block_size, flow_block_start);
  }
  
//    printf("krate fit reduction cnt:%d amt:%f\n",krate_cnt,krate_red);
}
