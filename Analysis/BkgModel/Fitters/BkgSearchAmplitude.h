/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGSEARCHAMPLITUDE_H
#define BKGSEARCHAMPLITUDE_H

#include "FlowBuffer.h"
#include "BeadParams.h"
#include "BeadScratch.h"
#include "BkgTrace.h"
#include "DiffEqModel.h"
#include "MultiFlowModel.h"
#include "TimeCompression.h"
#include "MathOptim.h"
#include "RegionTracker.h"
#include "EmphasisVector.h"

// Constified by hzable 11-14-2013

// Important: this is separate from the LevMar optimization group of items
// I am unhappy with this except insofar as it removes one method from the "golden hammer" routine

// given a list of beads and fixed regional parameters
// find good amplitudes for the beads
// why do I need so >many< things just to do this?????
class SearchAmplitude{
  public:
          PoissonCDFApproxMemo *math_poiss;
    const BkgTrace *my_trace;
    const EmptyTrace *empty_trace;
          BeadScratchSpace *my_scratch;
          incorporation_params_block_flows *my_cur_bead_block;
          buffer_params_block_flows *my_cur_buffer_block;
          RegionTracker *pointer_regions;
    const TimeCompression *time_c;
    const FlowBufferInfo *my_flow;
    const EmphasisClass *emphasis_data;
    float negative_amplitude_limit;
    float positive_amplitude_limit; // prevent blowups
    int num_iterations;
    
    bool use_vectorization;
    
    // bad!  Parasite on pointers I'm not supposed to see
    // because we need so much context when generating a trace
    // beadtracker->traces corresponding to beads? (observed & bkg)
    // traces->time compression? traces->flow?
    // regions->emphasis data?
    // scratch space->? (should it wrap the incorporation model? - point to buffers?)
    // context objec
    void ParasitePointers(PoissonCDFApproxMemo *_math_poiss,
			  const BkgTrace *_my_trace,
			  const EmptyTrace *_empty_trace,
			        BeadScratchSpace *_my_scratch,
              incorporation_params_block_flows *_my_cur_bead_block,
              buffer_params_block_flows *_my_cur_buffer_block,
			        RegionTracker *_my_regions,
			  const TimeCompression *_time_c,
			  const FlowBufferInfo *_my_flow,
			  const EmphasisClass *_emphasis_data)
    {
      math_poiss = _math_poiss;
      my_trace = _my_trace;
      empty_trace = _empty_trace;
      my_scratch = _my_scratch;
      my_cur_bead_block = _my_cur_bead_block;
      my_cur_buffer_block = _my_cur_buffer_block;
      pointer_regions = _my_regions;
      time_c = _time_c;
      my_flow = _my_flow;
      emphasis_data = _emphasis_data;
    };
    void EvaluateAmplitudeFit(BeadParams *p, const float *avals, float *error_by_flow, int flow_block_size, int flow_block_start ) const;

    SearchAmplitude();
    ~SearchAmplitude();
    // second method
    void ProjectionSearchAmplitude(BeadTracker &my_beads, bool , bool sampledOnly, 
        int flow_block_size, int flow_block_start ) const;
    void ProjectionSearchOneBead(BeadParams *p, int flow_block_size, int flow_block_start) const;

    
};


#endif // BKGSEARCHAMPLITUDE_H
