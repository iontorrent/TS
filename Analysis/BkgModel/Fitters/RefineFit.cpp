/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "RefineFit.h"
#include "DNTPRiseModel.h"
#include "TraceCorrector.h"

RefineFit::RefineFit (BkgModel &_bkg) :
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
  my_single_fit.AllocLevMar (bkg.time_c,bkg.math_poiss,bkg.global_defaults.dampen_kmult,bkg.global_defaults.var_kmult_only, bkg.global_defaults.kmult_low_limit, bkg.global_defaults.kmult_hi_limit);
  my_single_fit.FillDecisionThreshold (bkg.global_defaults.krate_adj_limit);

}

void RefineFit::SupplyMultiFlowSignal (float *block_signal_corrected, int ibd)
{
  if (bkg.my_trace.AlreadyAdjusted())
  {
    bead_params *p = &bkg.my_beads.params_nn[ibd];
    reg_params *reg_p = &bkg.my_regions->rp;

    bkg.my_trace.MultiFlowFillSignalForBead (block_signal_corrected, ibd);
    // calculate proton flux from neighbors
    bkg.my_scratch.ResetXtalkToZero();
//  bkg.xtalk_execute.ExecuteXtalkFlux (ibd,bkg.my_scratch.cur_xtflux_block);

    // set up current bead parameters by flow
    FillBufferParamsBlockFlows (&bkg.my_scratch.cur_buffer_block,p,reg_p,bkg.my_flow.flow_ndx_map,bkg.my_flow.buff_flow);
    FillIncorporationParamsBlockFlows (&bkg.my_scratch.cur_bead_block, p,reg_p,bkg.my_flow.flow_ndx_map,bkg.my_flow.buff_flow);
    // make my corrected signal
    // subtract computed zeromer signal
    // uses parameters above

    // should do trace-corrector here?
//  MultiCorrectBeadBkg (block_signal_corrected,p,
//                       bkg.my_scratch,bkg.my_flow,bkg.time_c,*bkg.my_regions,bkg.my_scratch.shifted_bkg,bkg.use_vectorization);
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
  bead_params *p = &bkg.my_beads.params_nn[ibd];

  float block_signal_corrected[bkg.my_scratch.bead_flow_t];

  SupplyMultiFlowSignal (block_signal_corrected, ibd);
  
  error_track err_t; // temporary store errors for this bead this flow
  for (int fnum=0;fnum < NUMFB;fnum++)
  {
    float evect[bkg.time_c.npts];
    //local_emphasis[fnum].CustomEmphasis (evect,p->Ampl[fnum]);
    bkg.emphasis_data.CustomEmphasis (evect, p->Ampl[fnum]);
    float *signal_corrected = &block_signal_corrected[fnum*bkg.time_c.npts];
    int NucID = bkg.my_flow.flow_ndx_map[fnum];

    my_single_fit.FitOneFlow (fnum,evect,p,&err_t, signal_corrected,NucID,cache_step.NucFineStep(fnum),cache_step.i_start_fine_step[fnum],bkg.my_flow,bkg.time_c,bkg.emphasis_data,*bkg.my_regions);
  }

// do things with my error vector for this bead

  // now detect corruption & store average error
  DetectCorruption (p,err_t, WASHOUT_THRESHOLD, WASHOUT_FLOW_DETECTION);
  // update error here to be passed to later block of flows
  // don't keep individual flow errors because we're surprisingly tight on memory
  UpdateCumulativeAvgError (p,err_t,bkg.my_flow.buff_flow[NUMFB-1]+1); // current flow reached, 1-based

  // if necessary, send the errors to HDF5
  bkg.SendErrorVectorToHDF5 (p,err_t);

}


// fits all wells one flow at a time, using a LevMarFitter derived class
// only the amplitude term is fit
void RefineFit::FitAmplitudePerFlow ()
{

  bkg.my_regions->cache_step.CalculateNucRiseFineStep (&bkg.my_regions->rp,bkg.time_c,bkg.my_flow); // the same for the whole region because time-shift happens per well
  bkg.my_regions->cache_step.CalculateNucRiseCoarseStep (&bkg.my_regions->rp,bkg.time_c,bkg.my_flow); // use for xtalk

  bkg.my_scratch.FillShiftedBkg (*bkg.emptytrace, bkg.my_regions->rp.tshift, bkg.time_c, true);

  for (int ibd = 0;ibd < bkg.my_beads.numLBeads;ibd++)
  {
    if (bkg.my_beads.params_nn[ibd].my_state.clonal_read or bkg.my_beads.params_nn[ibd].my_state.random_samp)
        FitAmplitudePerBeadPerFlow (ibd,bkg.my_regions->cache_step);
  }

//    printf("krate fit reduction cnt:%d amt:%f\n",krate_cnt,krate_red);
}
