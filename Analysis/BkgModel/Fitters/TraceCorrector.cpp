/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "TraceCorrector.h"

TraceCorrector::TraceCorrector (BkgModel &_bkg) :
    bkg (_bkg)
{
}

TraceCorrector::~TraceCorrector()
{

}

void TraceCorrector::ReturnBackgroundCorrectedSignal(float *block_signal_corrected, int ibd)
{
    bead_params *p = &bkg.my_beads.params_nn[ibd];
  reg_params *reg_p = &bkg.my_regions->rp;


  bkg.my_trace.MultiFlowFillSignalForBead (block_signal_corrected, ibd);
//  my_trace.FillNNSignalForBead (block_nn_signal, ibd);

  // calculate proton flux from neighbors
  bkg.my_scratch.ResetXtalkToZero();

  // specify cross talk independently of any well level correction
  // proton defaults happen at command line option level
  // if we're doing post-well correction, this will be auto-set to false
  // unless we override, when we might want to try both or neither.
  if (bkg.xtalk_spec.do_xtalk_correction)
    bkg.xtalk_execute.ExecuteXtalkFlux (ibd,bkg.my_scratch.cur_xtflux_block);

  // set up current bead parameters by flow
  FillBufferParamsBlockFlows (&bkg.my_scratch.cur_buffer_block,p,reg_p,bkg.my_flow.flow_ndx_map,bkg.my_flow.buff_flow);
  FillIncorporationParamsBlockFlows (&bkg.my_scratch.cur_bead_block, p,reg_p,bkg.my_flow.flow_ndx_map,bkg.my_flow.buff_flow);

  // make my corrected signal
  // subtract computed zeromer signal
  // uses parameters above
  MultiCorrectBeadBkg (block_signal_corrected,p,
                       bkg.my_scratch,bkg.my_flow,bkg.time_c,*bkg.my_regions,bkg.my_scratch.shifted_bkg,bkg.use_vectorization);

}

// point of no-return ..sort of.  After this function call all beads are already background
// corrected in the fg_buffer...so after this step we don't have the background any more.
// we could always generate the background and un-correct them of course
void TraceCorrector::BackgroundCorrectBeadInPlace (int ibd)
{
  float block_signal_corrected[bkg.my_scratch.bead_flow_t];

  ReturnBackgroundCorrectedSignal(block_signal_corrected, ibd);
  // now write it back
  bkg.my_trace.WriteBackSignalForBead (&block_signal_corrected[0],ibd);
}

// corrects all beads in the trace buffer..no going back!
void TraceCorrector::BackgroundCorrectAllBeadsInPlace (void)
{
  bkg.my_scratch.FillShiftedBkg (*bkg.emptytrace,bkg.my_regions->rp.tshift,bkg.time_c,true);

  for (int ibd = 0;ibd < bkg.my_beads.numLBeads;ibd++)
  {
    if (bkg.my_beads.params_nn[ibd].my_state.clonal_read or bkg.my_beads.params_nn[ibd].my_state.random_samp) // if we'll be fitting this bead
      BackgroundCorrectBeadInPlace (ibd);
  }
  bkg.my_trace.SetBkgCorrectTrace(); // warning that data is not raw traces
}
