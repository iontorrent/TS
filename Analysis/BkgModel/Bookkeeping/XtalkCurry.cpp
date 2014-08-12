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
  my_cur_bead_block_p = NULL;
  my_cur_buffer_block_p = NULL;
  my_flow_p = NULL;
  my_trace_p = NULL;
  my_generic_xtalk = NULL;
  use_vectorization = true;
  fast_compute = true; // seems to be slightly better
  neiIdxMap = NULL;
}

XtalkCurry::~XtalkCurry()
{
  if (my_generic_xtalk!=NULL)
    delete[] my_generic_xtalk;
  my_generic_xtalk = NULL;
  if (neiIdxMap != NULL)
    delete[] neiIdxMap;
  neiIdxMap = NULL;
}

// pointers to the items needed to compute:
// neighbors of a bead
// buffering parameters of the neighbors
// incorporation of the neighbors/excess hydrogens
// trace calculation - time compression
void XtalkCurry::CloseOverPointers ( Region *_region, TraceCrossTalkSpecification *_xtalk_spec_p,
                                     BeadTracker *_my_beads_p, RegionTracker *_my_regions_p,
                                     TimeCompression *_time_cp, PoissonCDFApproxMemo *_math_poiss,
                                     BeadScratchSpace *_my_scratch_p, 
                                     incorporation_params_block_flows *_my_cur_bead_block_p,
                                     buffer_params_block_flows *_my_cur_buffer_block_p,
                                     FlowBufferInfo *_my_flow_p,
                                     BkgTrace *_my_trace_p, bool _use_vectorization)
{
  region = _region;
  xtalk_spec_p = _xtalk_spec_p;
  my_beads_p = _my_beads_p;
  my_regions_p = _my_regions_p;
  time_cp = _time_cp;
  math_poiss = _math_poiss;
  my_scratch_p = _my_scratch_p;
  my_cur_bead_block_p = _my_cur_bead_block_p;
  my_cur_buffer_block_p = _my_cur_buffer_block_p;
  my_flow_p = _my_flow_p;
  my_trace_p = _my_trace_p;
  use_vectorization = _use_vectorization;

  my_generic_xtalk = new float [my_scratch_p->bead_flow_t];
  // zero me out if not needed
  memset(my_generic_xtalk, 0, sizeof(float[my_scratch_p->bead_flow_t]));

  // neighbour index map for all the beads in the region
  if (xtalk_spec_p) {
    neiIdxMap = new int[xtalk_spec_p->nei_affected*my_beads_p->numLBeads];
    GenerateNeighborIndexMapForAllBeads();
  }
  
}

// refactor to simplify
void XtalkCurry::NewXtalkFlux ( int cx, int cy,float *my_xtflux, int flow_block_size, int flow_block_start )
{

  if ( ( not my_beads_p->ndx_map.empty() ) and xtalk_spec_p->do_xtalk_correction )
  {
    int nn_ndx,ncx,ncy;


    // Iterate over the number of neighbors, accumulating hydrogen ions
    int nei_total = 0;
    for ( int nei_idx=0; nei_idx<xtalk_spec_p->nei_affected; nei_idx++ )
    {
      xtalk_spec_p->NeighborByChipType ( ncx,ncy,cx,cy,nei_idx,region->col,region->row );

      if ( ( ncx>-1 ) && ( ncx <region->w ) && ( ncy>-1 ) && ( ncy<region->h ) ) // neighbor within region
      {
        if ( ( nn_ndx=my_beads_p->ndx_map[ncy*region->w+ncx] ) !=-1 ) // bead present
        {
          // tau_top = how fast ions leave well
          // tau_bulk = how slowly ions leave bulk over well - 'simulates' neighbors having different upstream profiles
          // multiplier = how many ions get to this location as opposed to others
          if ( xtalk_spec_p->multiplier[nei_idx] > 0 )
          {
            MathModel::AccumulateSingleNeighborXtalkTrace ( my_xtflux,&my_beads_p->params_nn[nn_ndx], &my_regions_p->rp,
                                                 *my_scratch_p, *my_cur_bead_block_p,
                                                 *time_cp, *my_regions_p, *my_flow_p, math_poiss, use_vectorization,
                                                 xtalk_spec_p->tau_top[nei_idx],xtalk_spec_p->tau_fluid[nei_idx],xtalk_spec_p->multiplier[nei_idx], flow_block_size, flow_block_start );
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
void XtalkCurry::ExcessXtalkFlux ( int cx, int cy,float *my_xtflux, float *my_nei_flux,
                                   int flow_block_size, int flow_block_start )
{

  if ( ( not my_beads_p->ndx_map.empty() ) and xtalk_spec_p->do_xtalk_correction )
  {
    int nn_ndx,ncx,ncy;

    // Iterate over the number of neighbors, accumulating hydrogen ions
    int nei_total = 0;
    for ( int nei_idx=0; nei_idx<xtalk_spec_p->nei_affected; nei_idx++ )
    {
      xtalk_spec_p->NeighborByChipType ( ncx,ncy,cx,cy,nei_idx,region->col,region->row );

      if ( ( ncx>-1 ) && ( ncx <region->w ) && ( ncy>-1 ) && ( ncy<region->h ) ) // neighbor within region
      {
        if ( ( nn_ndx=my_beads_p->ndx_map[ncy*region->w+ncx] ) !=-1 ) // bead present
        {
          // tau_top = how fast ions leave well
          // tau_bulk = how slowly ions leave bulk over well - 'simulates' neighbors having different upstream profiles
          // multiplier = how many ions get to this location as opposed to others
          if ( xtalk_spec_p->multiplier[nei_idx] > 0 )
          {
            
            float neighbor_signal[my_scratch_p->bead_flow_t];
            my_trace_p->MultiFlowFillSignalForBead ( neighbor_signal, nn_ndx, flow_block_size );
            
            // implicit: my_scratch has empty trace filled in already
            if (xtalk_spec_p->simple_model )
            {
              MathModel::AccumulateSingleNeighborExcessHydrogenOneParameter ( my_xtflux,neighbor_signal, &my_beads_p->params_nn[nn_ndx], &my_regions_p->rp,
                  *my_scratch_p, *my_cur_buffer_block_p, 
                  *time_cp, *my_regions_p, *my_flow_p, use_vectorization,
                  xtalk_spec_p->multiplier[nei_idx] ,xtalk_spec_p->rescale_flag,
                  flow_block_size, flow_block_start);
            }
            else
            {
              MathModel::AccumulateSingleNeighborExcessHydrogen ( my_xtflux,neighbor_signal, &my_beads_p->params_nn[nn_ndx], &my_regions_p->rp,
                  *my_scratch_p, *my_cur_buffer_block_p, 
                  *time_cp, *my_regions_p, *my_flow_p, use_vectorization,
                  xtalk_spec_p->tau_top[nei_idx],xtalk_spec_p->tau_fluid[nei_idx],xtalk_spec_p->multiplier[nei_idx],
                  flow_block_size, flow_block_start);
            }
            
            if ( my_nei_flux!=NULL )
              AccumulateVector ( my_nei_flux,neighbor_signal, my_scratch_p->bead_flow_t ); // neighbors are additive in general
          }
          nei_total++;
        }
      }
    }
  }
}


void XtalkCurry::ComputeTypicalCrossTalk ( 
    float *my_xtalk_buffer, float *my_nei_buffer,
    int flow_block_size, int flow_block_start
  )
{
  memset ( my_xtalk_buffer,0,sizeof ( float[my_scratch_p->bead_flow_t] ) );
  if (my_nei_buffer!=NULL)
    memset ( my_nei_buffer, 0, sizeof ( float[my_scratch_p->bead_flow_t] ) );
  // generate cross-talk expected for a "sample" of reads
  // should generate exactly the empty list
  // but there's a little complexity there, so hack for now and pick a subset of the locations
  int nsample=100;
  for ( int i_test=0; i_test<nsample; i_test++ )
  {
    int large_num = i_test*LARGE_PRIME;
    int t_row = large_num/region->h;
    t_row = t_row % region->h;
    int t_col = large_num % region->w;
    ExcessXtalkFlux ( t_row, t_col, my_xtalk_buffer, my_nei_buffer, flow_block_size, flow_block_start );

  }
  MultiplyVectorByScalar ( my_xtalk_buffer, 1.0f/nsample, my_scratch_p->bead_flow_t );
  if (my_nei_buffer!=NULL)
    MultiplyVectorByScalar ( my_nei_buffer, 1.0f/nsample, my_scratch_p->bead_flow_t );
}

// Warning:  has side effects on my_scratch.  Use with caution
// need to isolate its own "scratch" construct
void XtalkCurry::ExecuteXtalkFlux ( int ibd,float *my_xtflux, int flow_block_size, int flow_block_start )
{
  int cx, cy;
  cx = my_beads_p->params_nn[ibd].x;
  cy = my_beads_p->params_nn[ibd].y;
  if ( fast_compute ) {
    ExcessXtalkFlux ( cx,cy,my_xtflux, NULL, flow_block_size, flow_block_start );
  }
  else {
    NewXtalkFlux ( cx,cy,my_xtflux, flow_block_size, flow_block_start );
  }
      // reduce by generic value of cross talk already present in empty well trace
  DiminishVector(my_xtflux, my_generic_xtalk, my_scratch_p->bead_flow_t);
}

// Mainly needed to get better access to what are neighbours 
// for each bead on GPU.Can be used for CPU too. Just one time exercise when
// constructing instance of this class
void XtalkCurry::GenerateNeighborIndexMapForAllBeads()
{
  int* idxMap = neiIdxMap; // temporary to walk through the array
  int nei_x, nei_y, nn_ndx;
  for (int nei=0; nei<xtalk_spec_p->nei_affected; ++nei) {
    for (int ibd=0; ibd<my_beads_p->numLBeads; ++ibd) {
      xtalk_spec_p->NeighborByChipType(nei_x, 
                                       nei_y,
                                       my_beads_p->params_nn[ibd].x,
                                       my_beads_p->params_nn[ibd].y,
                                       nei,
                                       region->col,
                                       region->row);

      if ((nei_x > -1) && (nei_x < region->w) && (nei_y > -1) && (nei_y < region->h)) // neighbor within region
      {
        if ( (nn_ndx=my_beads_p->ndx_map[nei_y*region->w + nei_x]) !=-1) // bead present
        {
          idxMap[ibd] = nn_ndx;
        }
        else {
          idxMap[ibd] = -1;
        } 
      }
      else {
          idxMap[ibd] = -1;
      }  
    }
    idxMap += my_beads_p->numLBeads;
  } 
}
