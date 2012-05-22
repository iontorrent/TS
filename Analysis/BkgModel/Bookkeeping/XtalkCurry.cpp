/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "XtalkCurry.h"
#include "MultiFlowModel.h"

XtalkCurry::XtalkCurry()
{
  region = NULL;
  xtalk_spec_p = NULL;
  my_beads_p = NULL;
  my_regions_p = NULL;
  time_cp = NULL;
  math_poiss=NULL;
  my_scratch_p = NULL;
  my_flow_p = NULL;
  my_trace_p = NULL;
  use_vectorization = true;
  fast_compute = true; // seems to be slightly better
}

// pointers to the items needed to compute:
// neighbors of a bead
// buffering parameters of the neighbors
// incorporation of the neighbors/excess hydrogens
// trace calculation - time compression
void XtalkCurry::CloseOverPointers(Region *_region, CrossTalkSpecification *_xtalk_spec_p,
                             BeadTracker *_my_beads_p, RegionTracker *_my_regions_p,
                             TimeCompression *_time_cp, PoissonCDFApproxMemo *_math_poiss,
                             BeadScratchSpace *_my_scratch_p, flow_buffer_info *_my_flow_p,
                             BkgTrace *_my_trace_p, bool _use_vectorization)
{
    region = _region;
    xtalk_spec_p = _xtalk_spec_p;
    my_beads_p = _my_beads_p;
    my_regions_p = _my_regions_p;
    time_cp = _time_cp;
    math_poiss = _math_poiss;
    my_scratch_p = _my_scratch_p;
    my_flow_p = _my_flow_p;
    my_trace_p = _my_trace_p;
    use_vectorization = _use_vectorization;
}

// refactor to simplify
void XtalkCurry::NewXtalkFlux (int ibd,float *my_xtflux)
{

  if ( (my_beads_p->ndx_map != NULL) & xtalk_spec_p->do_xtalk_correction)
  {
    int nn_ndx,cx,cy,ncx,ncy;

    cx = my_beads_p->params_nn[ibd].x;
    cy = my_beads_p->params_nn[ibd].y;

    // Iterate over the number of neighbors, accumulating hydrogen ions
    int nei_total = 0;
    for (int nei_idx=0; nei_idx<xtalk_spec_p->nei_affected; nei_idx++)
    {
      xtalk_spec_p->NeighborByChipType(ncx,ncy,cx,cy,nei_idx,region->col,region->row);

      if ( (ncx>-1) && (ncx <region->w) && (ncy>-1) && (ncy<region->h)) // neighbor within region
      {
        if ( (nn_ndx=my_beads_p->ndx_map[ncy*region->w+ncx]) !=-1) // bead present
        {
          // tau_top = how fast ions leave well
          // tau_bulk = how slowly ions leave bulk over well - 'simulates' neighbors having different upstream profiles
          // multiplier = how many ions get to this location as opposed to others
          if (xtalk_spec_p->multiplier[nei_idx] > 0)
          {
            AccumulateSingleNeighborXtalkTrace (my_xtflux,&my_beads_p->params_nn[nn_ndx], &my_regions_p->rp,
                                                *my_scratch_p, *time_cp, *my_regions_p, *my_flow_p, math_poiss, use_vectorization,
                                                xtalk_spec_p->tau_top[nei_idx],xtalk_spec_p->tau_fluid[nei_idx],xtalk_spec_p->multiplier[nei_idx]);
          }
          nei_total++;
        }
      }
    }
  }
}


// use directly integrated "excess hydrogens not from bkg" as a quick & dirty computation of cross-talk source
// does not require detailed modeling of the bead
// does not require the amplitude of the neighbor to have been computed
void XtalkCurry::ExcessXtalkFlux (int ibd,float *my_xtflux)
{

  if ( (my_beads_p->ndx_map != NULL) & xtalk_spec_p->do_xtalk_correction)
  {
    int nn_ndx,cx,cy,ncx,ncy;

    cx = my_beads_p->params_nn[ibd].x;
    cy = my_beads_p->params_nn[ibd].y;

    // Iterate over the number of neighbors, accumulating hydrogen ions
    int nei_total = 0;
    for (int nei_idx=0; nei_idx<xtalk_spec_p->nei_affected; nei_idx++)
    {
      xtalk_spec_p->NeighborByChipType(ncx,ncy,cx,cy,nei_idx,region->col,region->row);

      if ( (ncx>-1) && (ncx <region->w) && (ncy>-1) && (ncy<region->h)) // neighbor within region
      {
        if ( (nn_ndx=my_beads_p->ndx_map[ncy*region->w+ncx]) !=-1) // bead present
        {
          // tau_top = how fast ions leave well
          // tau_bulk = how slowly ions leave bulk over well - 'simulates' neighbors having different upstream profiles
          // multiplier = how many ions get to this location as opposed to others
          if (xtalk_spec_p->multiplier[nei_idx] > 0)
          {
            float neighbor_signal[my_scratch_p->bead_flow_t];
            my_trace_p->MultiFlowFillSignalForBead (neighbor_signal, nn_ndx);
            // implicit: my_scratch has empty trace filled in already
            AccumulateSingleNeighborExcessHydrogen (my_xtflux,neighbor_signal, &my_beads_p->params_nn[nn_ndx], &my_regions_p->rp,
                                                *my_scratch_p, *time_cp, *my_regions_p, *my_flow_p, use_vectorization,
                                                xtalk_spec_p->tau_top[nei_idx],xtalk_spec_p->tau_fluid[nei_idx],xtalk_spec_p->multiplier[nei_idx]);
          }
          nei_total++;
        }
      }
    }
  }
}

// Warning:  has side effects on my_scratch.  Use with caution
// need to isolate its own "scratch" construct
void XtalkCurry::ExecuteXtalkFlux (int ibd,float *my_xtflux)
{
  if (fast_compute)
    ExcessXtalkFlux(ibd,my_xtflux);
  else
    NewXtalkFlux(ibd,my_xtflux);
}