/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "NucStepCache.h"
#include "DNTPRiseModel.h"

NucStep::NucStep()
{
  nuc_rise_fine_step = NULL;
  nuc_rise_coarse_step = NULL;

  i_start_coarse_step = i_start_fine_step = NULL;
  per_flow_coarse_step = per_flow_fine_step = NULL;
  all_flow_t = 0;
  precomputed_step = false;
}

void NucStep::Alloc ( int npts, int flow_block_size )
{
  per_flow_coarse_step = new float* [flow_block_size];
  per_flow_fine_step   = new float* [flow_block_size];
  i_start_coarse_step  = new int    [flow_block_size];
  i_start_fine_step    = new int    [flow_block_size];

  for ( int fnum=0;fnum<flow_block_size;fnum++ )
  {
    i_start_coarse_step[fnum] = 0;
    i_start_fine_step[fnum] = 0;
  }

  all_flow_t = flow_block_size * npts;
  if ( nuc_rise_coarse_step==NULL )
    nuc_rise_coarse_step=new float[all_flow_t*ISIG_SUB_STEPS_MULTI_FLOW];
  if ( nuc_rise_fine_step==NULL )
    nuc_rise_fine_step=new float[all_flow_t*ISIG_SUB_STEPS_SINGLE_FLOW];
  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {
    per_flow_coarse_step[fnum] = &nuc_rise_coarse_step[fnum*ISIG_SUB_STEPS_MULTI_FLOW*npts];
    per_flow_fine_step[fnum]   = &nuc_rise_fine_step[fnum*ISIG_SUB_STEPS_SINGLE_FLOW*npts];
  }
}

void NucStep::CalculateNucRiseFineStep ( reg_params *a_region, TimeCompression &time_c, FlowBufferInfo &my_flow )
{
  // short circuit if we've precomputed this already for this region
  if ( !precomputed_step )
  {
    DntpRiseModel dntpMod ( time_c.npts(),a_region->nuc_shape.C,&time_c.frameNumber[0],ISIG_SUB_STEPS_SINGLE_FLOW );

    // compute the nuc rise model for each nucleotide...in the long run could be separated out
    // and computed region-wide
    // at the moment, this computation is identical for all wells.  If it were not, we couldn't
    // do it here.
    for ( int fnum=0;fnum<my_flow.GetMaxFlowCount();fnum++ )
    {
      int NucID = my_flow.flow_ndx_map[fnum];
      i_start_fine_step[fnum]=dntpMod.CalcCDntpTop ( per_flow_fine_step[fnum],
                              GetModifiedMidNucTime ( & ( a_region->nuc_shape ),NucID,fnum ),
                              GetModifiedSigma ( & ( a_region->nuc_shape ),NucID ), a_region->nuc_shape.nuc_flow_span,NucID );
    }
  }
}

void NucStep::CalculateNucRiseFineStep ( 
   reg_params *a_region, 
   int frames, 
   std::vector<float> &frameNumber,
   FlowBufferInfo &my_flow )
{
  // short circuit if we've precomputed this already for this region
  if ( !precomputed_step )
  {
    DntpRiseModel dntpMod (
        frames, 
        a_region->nuc_shape.C,
        &frameNumber[0],
        ISIG_SUB_STEPS_SINGLE_FLOW );

    // compute the nuc rise model for each nucleotide...in the long run could be separated out
    // and computed region-wide
    // at the moment, this computation is identical for all wells.  If it were not, we couldn't
    // do it here.
    for ( int fnum=0;fnum<my_flow.GetMaxFlowCount();fnum++ )
    {
      int NucID = my_flow.flow_ndx_map[fnum];
      i_start_fine_step[fnum]=dntpMod.CalcCDntpTop ( per_flow_fine_step[fnum],
                              GetModifiedMidNucTime ( & ( a_region->nuc_shape ),NucID,fnum ),
                              GetModifiedSigma ( & ( a_region->nuc_shape ),NucID ), a_region->nuc_shape.nuc_flow_span,NucID );
    }
  }
}



float * NucStep::NucFineStep ( int fnum )
{
  return ( per_flow_fine_step[fnum] );
}

const float * NucStep::NucCoarseStep ( int fnum ) const
{
  return ( per_flow_coarse_step[fnum] );
}

void NucStep::CalculateNucRiseCoarseStep ( 
    reg_params *a_region, const TimeCompression &time_c, const FlowBufferInfo &my_flow )
{
  // if we've precomputed this step already, skip recompputation
  if ( !precomputed_step )
  {
    DntpRiseModel dntpMod ( time_c.npts(),a_region->nuc_shape.C,&time_c.frameNumber[0],ISIG_SUB_STEPS_MULTI_FLOW );

    // compute the nuc rise model for each nucleotide...in the long run could be separated out
    // and computed region-wide
    // at the moment, this computation is identical for all wells.  If it were not, we couldn't
    // do it here.
    for ( int fnum=0;fnum<my_flow.GetMaxFlowCount();fnum++ )
    {
      int NucID = my_flow.flow_ndx_map[fnum];
      i_start_coarse_step[fnum]=dntpMod.CalcCDntpTop ( per_flow_coarse_step[fnum],
                                GetModifiedMidNucTime ( & ( a_region->nuc_shape ),NucID, fnum ),
                                GetModifiedSigma ( & ( a_region->nuc_shape ),NucID ), a_region->nuc_shape.nuc_flow_span,NucID );
    }
  }
}

void NucStep::ForceLockCalculateNucRiseCoarseStep( 
    reg_params *a_region, const TimeCompression &time_c, const FlowBufferInfo &my_flow )
{
  // unlock if locked
  precomputed_step = false;
  // cache my step
  CalculateNucRiseCoarseStep(a_region,time_c,my_flow);
  // lock so no recompute
  precomputed_step = true;
}

void NucStep::Unlock()
{
  // leave in a flexible state for other routines
  precomputed_step = false;
}

void NucStep::Delete()
{
  delete [] nuc_rise_coarse_step;
  delete [] nuc_rise_fine_step;

  delete [] per_flow_coarse_step;
  delete [] per_flow_fine_step;
  delete [] i_start_coarse_step;
  delete [] i_start_fine_step;
}

NucStep::~NucStep()
{
  Delete();
}
