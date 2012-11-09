/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "RefineFit.h"
#include "DNTPRiseModel.h"
#include "TraceCorrector.h"

RefineFit::RefineFit (SignalProcessingMasterFitter &_bkg) :
    bkg (_bkg)
{
  local_emphasis = NULL;
  InitSingleFlowFit();
}

void RefineFit::SetupLocalEmphasis()
{
  local_emphasis = new EmphasisClass[NUMFB];
}

RefineFit::~RefineFit()
{
  if (local_emphasis!=NULL)
    delete[] local_emphasis;
}

void RefineFit::InitSingleFlowFit()
{
  // fix up the fitter for refining amplitude/krate
  my_single_fit.AllocLevMar (bkg.region_data->time_c,bkg.math_poiss,
                             bkg.global_defaults.signal_process_control.dampen_kmult,bkg.global_defaults.signal_process_control.var_kmult_only,
                             bkg.global_defaults.signal_process_control.kmult_low_limit, bkg.global_defaults.signal_process_control.kmult_hi_limit,
                             bkg.global_defaults.signal_process_control.AmplLowerLimit );
  my_single_fit.FillDecisionThreshold (bkg.global_defaults.signal_process_control.krate_adj_limit);
  my_single_fit.SetRetryLimit (bkg.global_defaults.signal_process_control.single_flow_fit_max_retry);
  my_single_fit.fit_alt = bkg.global_defaults.signal_process_control.fit_alternate;
}

void RefineFit::SupplyMultiFlowSignal (float *block_signal_corrected, int ibd)
{
  if (bkg.region_data->my_trace.AlreadyAdjusted())
  {
    bead_params *p = &bkg.region_data->my_beads.params_nn[ibd];
    reg_params *reg_p = &bkg.region_data->my_regions.rp;

    bkg.region_data->my_trace.MultiFlowFillSignalForBead (block_signal_corrected, ibd);
    // calculate proton flux from neighbors
    bkg.region_data->my_scratch.ResetXtalkToZero();
//  bkg.xtalk_execute.ExecuteXtalkFlux (ibd,bkg.region_data->my_scratch.cur_xtflux_block);

    // set up current bead parameters by flow
    FillBufferParamsBlockFlows (&bkg.region_data->my_scratch.cur_buffer_block,p,reg_p,bkg.region_data->my_flow.flow_ndx_map,bkg.region_data->my_flow.buff_flow);
    FillIncorporationParamsBlockFlows (&bkg.region_data->my_scratch.cur_bead_block, p,reg_p,bkg.region_data->my_flow.flow_ndx_map,bkg.region_data->my_flow.buff_flow);
    // make my corrected signal
    // subtract computed zeromer signal
    // uses parameters above

    // should do trace-corrector here?
//  MultiCorrectBeadBkg (block_signal_corrected,p,
//                       bkg.region_data->my_scratch,bkg.region_data->my_flow,bkg.region_data->time_c,*bkg.region_data->my_regions,bkg.region_data->my_scratch.shifted_bkg,bkg.use_vectorization);
  }
  else
  {
    bkg.trace_bkg_adj->ReturnBackgroundCorrectedSignal (block_signal_corrected, ibd);
  }
}


//@TODO:  Revert local bead correction here
// For refining "time" shifts, we want to use sampled beads, so don't do bkg subtraction on everything and munge the buffers.
void RefineFit::FitAmplitudePerBeadPerFlow (int ibd, NucStep &cache_step)
{
  bead_params *p = &bkg.region_data->my_beads.params_nn[ibd];
  float block_signal_corrected[bkg.region_data->my_scratch.bead_flow_t];
  float block_signal_predicted[bkg.region_data->my_scratch.bead_flow_t];

  SupplyMultiFlowSignal (block_signal_corrected, ibd);

  error_track err_t; // temporary store errors for this bead this flow
  for (int fnum=0;fnum < NUMFB;fnum++)
  {
    float evect[bkg.region_data->time_c.npts()];
    //local_emphasis[fnum].CustomEmphasis (evect,p->Ampl[fnum]);
    bkg.region_data->emphasis_data.CustomEmphasis (evect, p->Ampl[fnum]);
    float *signal_corrected = &block_signal_corrected[fnum*bkg.region_data->time_c.npts()];
    float *signal_predicted = &block_signal_predicted[fnum*bkg.region_data->time_c.npts()];
    int NucID = bkg.region_data->my_flow.flow_ndx_map[fnum];

    my_single_fit.FitOneFlow (fnum,evect,p,&err_t, signal_corrected,signal_predicted, NucID,cache_step.NucFineStep (fnum),cache_step.i_start_fine_step[fnum],bkg.region_data->my_flow,bkg.region_data->time_c,bkg.region_data->emphasis_data,bkg.region_data->my_regions);
  }

// do things with my error vector for this bead

  // now detect corruption & store average error
  DetectCorruption (p,err_t, WASHOUT_THRESHOLD, WASHOUT_FLOW_DETECTION);
  // update error here to be passed to later block of flows
  // don't keep individual flow errors because we're surprisingly tight on memory
  UpdateCumulativeAvgError (p,err_t,bkg.region_data->my_flow.buff_flow[NUMFB-1]+1); // current flow reached, 1-based

  // if necessary, send the errors to HDF5
  bkg.global_state.SendErrorVectorToHDF5 (p,err_t, bkg.region_data->region,bkg.region_data->my_flow);
  // send prediction to hdf5 if necessary
  bkg.global_state.SendPredictedToHDF5(ibd, block_signal_predicted, *bkg.region_data);
  bkg.global_state.SendCorrectedToHDF5(ibd, block_signal_corrected, *bkg.region_data);
  bkg.global_state.SendXtalkToHDF5(ibd, bkg.region_data->my_scratch.cur_xtflux_block, *bkg.region_data);
}

//  This is testing what the cross-talk routine produces
void RefineFit::TestXtalk()
{
  float xtalk_signal_typical[bkg.region_data->my_scratch.bead_flow_t];
  float xtalk_source_typical[bkg.region_data->my_scratch.bead_flow_t];
  
  bkg.xtalk_execute.ComputeTypicalCrossTalk(xtalk_signal_typical, xtalk_source_typical);
  FILE *fp;
  char file_name[100];
  sprintf(file_name,"xtalk.typical.%04d.txt",bkg.region_data->my_flow.buff_flow[NUMFB-1]+1);
  fp = fopen(file_name,"at");
  int npts = bkg.region_data->time_c.npts();
  for (int fnum=0; fnum<NUMFB; fnum++)
  {
    fprintf(fp,"%d\t%d\t%d\txtalk\t", bkg.region_data->region->row, bkg.region_data->region->col, bkg.region_data->my_flow.buff_flow[fnum]);
    int i;
    for ( i=0; i<npts; i++)
      fprintf(fp, "%f\t", xtalk_signal_typical[i+fnum*npts]);
    for (; i<MAX_COMPRESSED_FRAMES; i++)
      fprintf(fp,"0.0\t");
    fprintf(fp,"\n");
    
        fprintf(fp,"%d\t%d\t%d\tsource\t", bkg.region_data->region->row, bkg.region_data->region->col, bkg.region_data->my_flow.buff_flow[fnum]);
    for ( i=0; i<npts; i++)
      fprintf(fp, "%f\t", xtalk_source_typical[i+fnum*npts]);
    for (; i<MAX_COMPRESSED_FRAMES; i++)
      fprintf(fp,"0.0\t");
    fprintf(fp,"\n");
  }
  fclose(fp);
  
}


// fits all wells one flow at a time, using a LevMarFitter derived class
// only the amplitude term is fit
void RefineFit::FitAmplitudePerFlow ()
{

  bkg.region_data->my_regions.cache_step.CalculateNucRiseFineStep (&bkg.region_data->my_regions.rp,bkg.region_data->time_c,bkg.region_data->my_flow); // the same for the whole region because time-shift happens per well
  bkg.region_data->my_regions.cache_step.CalculateNucRiseCoarseStep (&bkg.region_data->my_regions.rp,bkg.region_data->time_c,bkg.region_data->my_flow); // use for xtalk

  bkg.region_data->my_scratch.FillShiftedBkg (*bkg.region_data->emptytrace, bkg.region_data->my_regions.rp.tshift, bkg.region_data->time_c, true);

  if (bkg.xtalk_spec.simple_model)
    bkg.xtalk_execute.ComputeTypicalCrossTalk(bkg.xtalk_execute.my_generic_xtalk, NULL); // get the generic value set

  //SpecializedEmphasisFunctions();
  //TestXtalk();

  for (int ibd = 0;ibd < bkg.region_data->my_beads.numLBeads;ibd++)
  {
    if (FitBeadLogic (&bkg.region_data->my_beads.params_nn[ibd]))
      FitAmplitudePerBeadPerFlow (ibd,bkg.region_data->my_regions.cache_step);
  }

//    printf("krate fit reduction cnt:%d amt:%f\n",krate_cnt,krate_red);
}

void RefineFit::SpecializedEmphasisFunctions()
{
  // set up one emphasis function per nuc
  // set them up to emphasize based on "typical" signal-noise and bead behavior
  // this should auto-tune by region nicely

  float basic_noise = 25.0f;
  float relative_threshold = 0.2f;

  bead_params typical_bead;
  bkg.region_data->my_beads.TypicalBeadParams (&typical_bead);

  TraceCurry calc_trace;
  calc_trace.Allocate (bkg.region_data->time_c.npts(),&bkg.region_data->time_c.deltaFrame[0], &bkg.region_data->time_c.deltaFrameSeconds[0], bkg.math_poiss);

  // ugh, temporary file debug here
  FILE *fp = NULL;
  if (false)
  {
    char my_file[100];
    sprintf (my_file,"LOCALEM.%d.%d.%d.txt", bkg.region_data->region->col,bkg.region_data->region->row, bkg.region_data->my_flow.buff_flow[0]);
    fp=fopen (my_file,"wt");
  }
  for (int fnum=0; fnum<NUMFB; fnum++)
  {
    local_emphasis[fnum].SetupEmphasisTiming (bkg.region_data->time_c.npts(), &bkg.region_data->time_c.frames_per_point[0],&bkg.region_data->time_c.frameNumber[0]);

    int NucID = bkg.region_data->my_flow.flow_ndx_map[fnum];

    calc_trace.SetWellRegionParams (&typical_bead,&bkg.region_data->my_regions.rp,fnum,
                                    NucID,bkg.region_data->my_flow.buff_flow[fnum],
                                    bkg.region_data->my_regions.cache_step.i_start_fine_step[fnum],bkg.region_data->my_regions.cache_step.NucFineStep (fnum));

    float *my_bkg = &bkg.region_data->my_scratch.shifted_bkg[fnum*bkg.region_data->time_c.npts()]; // bad accessors

    for (int tmp_ev=0; tmp_ev<bkg.region_data->emphasis_data.numEv; tmp_ev++)
    {
      float fval[bkg.region_data->time_c.npts()];

      // signal
      float tmp_amp = tmp_ev;
      if (tmp_amp<0.5f)
        tmp_amp = 0.5f;
      calc_trace.SingleFlowIncorporationTrace (tmp_amp,fval);
      // background
      local_emphasis[fnum].SignalToBkgEmphasis (tmp_ev,fval, my_bkg, basic_noise, relative_threshold);

      if (fp!=NULL)
      {
        fprintf (fp,"%d\t%d\tfval\t", fnum, tmp_ev);
        for (int ll=0; ll<bkg.region_data->time_c.npts(); ll++)
          fprintf (fp,"%f\t", fval[ll]);
        fprintf (fp,"\n");
        fprintf (fp,"%d\t%d\tbkg\t", fnum, tmp_ev);
        for (int ll=0; ll<bkg.region_data->time_c.npts(); ll++)
          fprintf (fp,"%f\t", my_bkg[ll]);
        fprintf (fp,"\n");
        fprintf (fp,"%d\t%d\tlocalem\t", fnum, tmp_ev);
        for (int ll=0; ll<bkg.region_data->time_c.npts(); ll++)
          fprintf (fp,"%f\t", local_emphasis[fnum].EmphasisVectorByHomopolymer[tmp_ev][ll]);
        fprintf (fp,"\n");
        fprintf (fp,"%d\t%d\tem\t", fnum, tmp_ev);
        for (int ll=0; ll<bkg.region_data->time_c.npts(); ll++)
          fprintf (fp,"%f\t", bkg.region_data->emphasis_data.EmphasisVectorByHomopolymer[tmp_ev][ll]);
        fprintf (fp,"\n");
      }
    }
  }
  if (fp!=NULL)
    fclose (fp);
  //should have complete set of emphasis vectors now by flow and bead
  // perhaps I should save the default incorporation traces
  // debug dump here
}

