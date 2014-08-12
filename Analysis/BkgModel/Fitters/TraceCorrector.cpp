/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "TraceCorrector.h"

TraceCorrector::TraceCorrector (SignalProcessingMasterFitter &_bkg) :
    bkg (_bkg)
{
}

TraceCorrector::~TraceCorrector()
{

}

void TraceCorrector::ReturnBackgroundCorrectedSignal(float *block_signal_corrected, int ibd,
    int flow_block_size, int flow_block_start
  )
{
    BeadParams *p = &bkg.region_data->my_beads.params_nn[ibd];
  reg_params *reg_p = &bkg.region_data->my_regions.rp;


  bkg.region_data->my_trace.MultiFlowFillSignalForBead (block_signal_corrected, ibd, flow_block_size);
//  my_trace.FillNNSignalForBead (block_nn_signal, ibd);


    
  // calculate proton flux from neighbors
  bkg.region_data->my_scratch.ResetXtalkToZero();

  // specify cross talk independently of any well level correction
  // proton defaults happen at command line option level
  // if we're doing post-well correction, this will be auto-set to false
  // unless we override, when we might want to try both or neither.
  if (bkg.trace_xtalk_spec.do_xtalk_correction)
  {
    bkg.trace_xtalk_execute.ExecuteXtalkFlux (ibd,bkg.region_data->my_scratch.cur_xtflux_block,
                                              flow_block_size, flow_block_start);
  }

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
  MathModel::MultiCorrectBeadBkg (block_signal_corrected,p,
                       bkg.region_data->my_scratch,*bkg.region_data_extras.cur_buffer_block,
                       *bkg.region_data_extras.my_flow,bkg.region_data->time_c,
                       bkg.region_data->my_regions,bkg.region_data->my_scratch.shifted_bkg,bkg.global_defaults.signal_process_control.use_vectorization, flow_block_size);

 
}

// point of no-return ..sort of.  After this function call all beads are already background
// corrected in the fg_buffer...so after this step we don't have the background any more.
// we could always generate the background and un-correct them of course
void TraceCorrector::BackgroundCorrectBeadInPlace (int ibd, int flow_block_size, int flow_block_start)
{
  float block_signal_corrected[bkg.region_data->my_scratch.bead_flow_t];

  ReturnBackgroundCorrectedSignal(block_signal_corrected, ibd, flow_block_size, flow_block_start);
  // now write it back
  bkg.region_data->my_trace.WriteBackSignalForBead (&block_signal_corrected[0],ibd, -1, flow_block_size);
}

// corrects all beads in the trace buffer..no going back!
void TraceCorrector::BackgroundCorrectAllBeadsInPlace (int flow_block_size, int flow_block_start)
{
  bkg.region_data->my_scratch.FillShiftedBkg (*bkg.region_data->emptytrace,bkg.region_data->my_regions.rp.tshift,bkg.region_data->time_c,true, flow_block_size);

  for (int ibd = 0;ibd < bkg.region_data->my_beads.numLBeads;ibd++)
  {
    if (bkg.region_data->my_beads.params_nn[ibd].FitBeadLogic()) // if we'll be fitting this bead
      BackgroundCorrectBeadInPlace (ibd, flow_block_size, flow_block_start);
  }
  bkg.region_data->my_trace.SetBkgCorrectTrace(); // warning that data is not raw traces
}
