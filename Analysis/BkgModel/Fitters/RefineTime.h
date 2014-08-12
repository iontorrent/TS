/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REFINETIME_H
#define REFINETIME_H

#include "SignalProcessingMasterFitter.h"


class RefineTime
{
  public:
    SignalProcessingMasterFitter &bkg; // reference to source class for now

    RefineTime (SignalProcessingMasterFitter &);
    void  RefinePerFlowTimeEstimate(float *t_mid_nuc_shift_per_flow, int flow_block_size, int flow_block_start );

    // do this by 1-mers
    float FitAverage1Mer (int fnum,bool debug_output, int flow_block_size, int flow_block_start );
    void FitAverage1MerPerFlow (float *t_mid_nuc_shift_per_flow,bool debug_output, int flow_block_size,
        int flow_block_start );
    
    void FitAverage1MerAllFlows(float *t_mid_nuc_shift_per_flow, bool debug_output, int flow_block_size,
        int flow_block_start );
    
    void ConstructMultiFlowOneMers(float *block_bkg_corrected_avg_signal, int *cur_count, BeadParams *avg_bead_by_flow, int flow_block_size, int flow_block_start );
    
    float FitSingleFlowTimeShiftFromOneMer(float *avg_1mer, int len, int trc_cnt, BeadParams *avg_bead, int fnum, bool debug_output, int flow_block_start );
    int FindAvg1MerFromSingleFlowAdjustedData(float *avg_1mer, int len, BeadParams &avg_bead, int fnum, int flow_block_size);
    
    void RezeroUsingLocalShift(float *t_mid_nuc_shift_per_flow, int flow_block_size);
    
    void DebugOneMerOutput(float *avg_1mer, int len, int trc_cnt, float delta_mid_nuc, float new_kmult, bool debug_output);

};

#endif // REFINETIME_H
