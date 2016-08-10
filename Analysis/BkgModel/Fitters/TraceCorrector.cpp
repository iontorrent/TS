/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "TraceCorrector.h"

TraceCorrector::TraceCorrector (SignalProcessingMasterFitter &_bkg) :
    bkg (_bkg)
{
}

TraceCorrector::~TraceCorrector()
{

}

void TraceCorrector::ReturnBackgroundCorrectedSignal(float *block_signal_corrected, float *block_signal_original, float *block_signal_sbg,
                                                     int ibd,
    int flow_block_size, int flow_block_start
  )
{
    BeadParams *p = &bkg.region_data->my_beads.params_nn[ibd];
  reg_params *reg_p = &bkg.region_data->my_regions.rp;


  bkg.region_data->my_trace.MultiFlowFillSignalForBead (block_signal_corrected, ibd, flow_block_size);
  // store original away now that it has been brought from background so we can refer to it if needed for debugging
  // if this is a big drag, can fix debugging paths - but really this isn't noticeable on the profile
  if (block_signal_original!=NULL)
    memcpy(block_signal_original, block_signal_corrected, sizeof ( float[bkg.region_data->my_scratch.bead_flow_t] ));
  if (block_signal_sbg!=NULL)
    memcpy(block_signal_sbg,bkg.region_data->my_scratch.shifted_bkg,  sizeof ( float[bkg.region_data->my_scratch.bead_flow_t] ));

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
  // subtract dark matter
  // adjust for trace xtalk
  // uses parameters above
  MathModel::MultiCorrectBeadBkg (block_signal_corrected,p,
                       bkg.region_data->my_scratch,*bkg.region_data_extras.cur_buffer_block,
                       *bkg.region_data_extras.my_flow,bkg.region_data->time_c,
                       bkg.region_data->my_regions,bkg.region_data->my_scratch.shifted_bkg,bkg.global_defaults.signal_process_control.use_vectorization, flow_block_size);

 
}
