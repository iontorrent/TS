/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef XTALKCURRY_H
#define XTALKCURRY_H

#include "CrossTalkSpec.h"
#include "Region.h"
#include "BeadTracker.h"
#include "RegionTracker.h"
#include "TimeCompression.h"
#include "PoissonCdf.h"
#include "BeadScratch.h"
#include "FlowBuffer.h"

// isolate xtalk so that we can work on it without annoying the rest of the code
// important for proton

// "curry"/"close" over variables needed for cross-talk correction to traces
// Note!  currently has side effects to the "my_scratch" construct which should be eliminated
// use carefully!
class XtalkCurry{
  public:

    Region *region;
    CrossTalkSpecification *xtalk_spec_p;
    BeadTracker *my_beads_p;
    RegionTracker *my_regions_p;
    TimeCompression *time_cp;
    PoissonCDFApproxMemo *math_poiss;
    BeadScratchSpace *my_scratch_p; // really?
    flow_buffer_info *my_flow_p;
    BkgTrace *my_trace_p;
    bool use_vectorization;

    bool fast_compute;
    
    XtalkCurry();
    void ExecuteXtalkFlux(int ibd, float *my_xtflux);
   void NewXtalkFlux (int ibd,float *my_xtflux);
   void ExcessXtalkFlux (int ibd,float *my_xtflux);
   void CloseOverPointers(Region *_region, CrossTalkSpecification *_xtalk_spec_p,
                             BeadTracker *_my_beads_p, RegionTracker *_my_regions_p,
                             TimeCompression *_time_cp, PoissonCDFApproxMemo *_math_poiss,
                             BeadScratchSpace *_my_scratch_p, flow_buffer_info *_my_flow_p,
                             BkgTrace *_my_trace_p, bool _use_vectorization);
};

#endif // XTALKCURRY_H