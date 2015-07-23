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
    TraceCrossTalkSpecification *xtalk_spec_p;
    BeadTracker *my_beads_p;
    RegionTracker *my_regions_p;
    TimeCompression *time_cp;
    PoissonCDFApproxMemo *math_poiss;
    BeadScratchSpace *my_scratch_p; // really?
    incorporation_params_block_flows *my_cur_bead_block_p;
    buffer_params_block_flows *my_cur_buffer_block_p;
    FlowBufferInfo *my_flow_p;
    BkgTrace *my_trace_p;
    bool use_vectorization;

    float *my_generic_xtalk;

    bool fast_compute;
    
    XtalkCurry();
    ~XtalkCurry();
    void ExecuteXtalkFlux(int ibd, float *my_xtflux, int flow_block_size, int flow_block_start);
    void NewXtalkFlux (int cx, int cy,float *my_xtflux, int flow_block_size, int flow_block_start);
    void ExcessXtalkFlux (int cx, int cy,float *my_xtflux, float *my_nei_flux, 
                          int flow_block_size, int flow_block_start );
    void ComputeTypicalCrossTalk(float *my_xtalk_buffer, float *my_nei_buffer, int flow_block_size,
                          int flow_block_start );
    void CloseOverPointers(Region *_region, TraceCrossTalkSpecification *_xtalk_spec_p,
                             BeadTracker *_my_beads_p, RegionTracker *_my_regions_p,
                             TimeCompression *_time_cp, PoissonCDFApproxMemo *_math_poiss,
                             BeadScratchSpace *_my_scratch_p, 
                             incorporation_params_block_flows *_my_cur_bead_block_p,
                             buffer_params_block_flows *_my_cur_buffer_block_p,
                             FlowBufferInfo *_my_flow_p,
                             BkgTrace *_my_trace_p, bool _use_vectorization);
    const int* GetNeighborIndexMap() const { return neiIdxMap; }
    const int* GetNeighbourIndexmapForSampleLocations() const { return sampleNeiIdxMap; }

  private:
    // create a list of xtalk neighbours for each of the live wells
    void GenerateNeighborIndexMapForAllBeads();

    // create a list of xtalk neighbours for a sample of wells...Needed to 
    // compute a typical xtalk signal seen in the well. Need for simple xtalk
    // model..Ideally should have the locations of the empty wells
    void GenerateNeighbourIndexMapForSampleWells();

    // Generate sample locations in the region for calculating typical crosstalk
    // under the simple xtalk model
    void ObtainGenericXtalkSampleLocation(
      int sampIdx,
      int &sampCol,
      int &sampRow);

    // For each bead, list all its neighbour beads
    int* neiIdxMap; // neis x beads

    // list of neighbours for the sample locations in the region
    int* sampleNeiIdxMap; // neis x GENERIC_SIMPLE_XTALK_SAMPLE
};

#endif // XTALKCURRY_H
