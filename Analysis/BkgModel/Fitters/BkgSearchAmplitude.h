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

// Important: this is separate from the LevMar optimization group of items
// I am unhappy with this except insofar as it removes one method from the "golden hammer" routine

// given a list of beads and fixed regional parameters
// find good amplitudes for the beads
// why do I need so >many< things just to do this?????
class SearchAmplitude{
  public:
    PoissonCDFApproxMemo *math_poiss;
    BkgTrace *my_trace;
    EmptyTrace *empty_trace;
    BeadScratchSpace *my_scratch;
    RegionTracker *pointer_regions;
    TimeCompression *time_c;
    flow_buffer_info *my_flow;
    EmphasisClass *emphasis_data;
    float negative_amplitude_limit;
    
    bool use_vectorization;
    bool rate_fit;
    
    // bad!  Parasite on pointers I'm not supposed to see
    // because we need so much context when generating a trace
    // beadtracker->traces corresponding to beads? (observed & bkg)
    // traces->time compression? traces->flow?
    // regions->emphasis data?
    // scratch space->? (should it wrap the incorporation model? - point to buffers?)
    // context objec
    void ParasitePointers(PoissonCDFApproxMemo *_math_poiss,
			  BkgTrace *_my_trace,
			  EmptyTrace *_empty_trace,
			  BeadScratchSpace *_my_scratch,
			  RegionTracker *_my_regions,
			  TimeCompression *_time_c,
			  flow_buffer_info *_my_flow,
			  EmphasisClass *_emphasis_data)
    {
      math_poiss = _math_poiss;
      my_trace = _my_trace;
      empty_trace = _empty_trace;
      my_scratch = _my_scratch;
      pointer_regions = _my_regions;
      time_c = _time_c;
      my_flow = _my_flow;
      emphasis_data = _emphasis_data;
    };
    void EvaluateAmplitudeFit(bead_params *p, float *avals,float *error_by_flow);

    void BinarySearchOneBead(bead_params *p, float min_step, bool restart);
    void BinarySearchAmplitude(BeadTracker &my_beads, float min_step,bool restart);
    SearchAmplitude();
    ~SearchAmplitude();
    // second method
    void ProjectionSearchAmplitude(BeadTracker &my_beads, bool _rate_fit, bool sampledOnly);
    void ProjectionSearchOneBead(bead_params *p);

    
};


#endif // BKGSEARCHAMPLITUDE_H
