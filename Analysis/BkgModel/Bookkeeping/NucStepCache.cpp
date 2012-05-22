/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "NucStepCache.h"
#include "DNTPRiseModel.h"

NucStep::NucStep()
{
  nuc_rise_fine_step = NULL;
  nuc_rise_coarse_step = NULL;
  for (int fnum=0;fnum<NUMFB;fnum++)
  {
    i_start_coarse_step[fnum] = 0;
    i_start_fine_step[fnum] = 0;
    per_flow_coarse_step[fnum] = NULL;
    per_flow_fine_step[fnum] = NULL;
  }
  all_flow_t = 0;
}

void NucStep::Alloc (int npts)
{
  all_flow_t = NUMFB * npts;
  if (nuc_rise_coarse_step==NULL)
    nuc_rise_coarse_step=new float[all_flow_t*ISIG_SUB_STEPS_MULTI_FLOW];
  if (nuc_rise_fine_step==NULL)
    nuc_rise_fine_step=new float[all_flow_t*ISIG_SUB_STEPS_SINGLE_FLOW];
  for (int fnum=0; fnum<NUMFB; fnum++)
  {
    per_flow_coarse_step[fnum] = &nuc_rise_coarse_step[fnum*ISIG_SUB_STEPS_MULTI_FLOW*npts];
    per_flow_fine_step[fnum] = &nuc_rise_fine_step[fnum*ISIG_SUB_STEPS_SINGLE_FLOW*npts];
  }
}

void NucStep::CalculateNucRiseFineStep (reg_params *a_region, TimeCompression &time_c, flow_buffer_info &my_flow)
{
  DntpRiseModel dntpMod (time_c.npts,a_region->nuc_shape.C,time_c.frameNumber,ISIG_SUB_STEPS_SINGLE_FLOW);

  // compute the nuc rise model for each nucleotide...in the long run could be separated out
  // and computed region-wide
  // at the moment, this computation is identical for all wells.  If it were not, we couldn't
  // do it here.
  for (int fnum=0;fnum<NUMFB;fnum++)
  {
    int NucID = my_flow.flow_ndx_map[fnum];
    i_start_fine_step[fnum]=dntpMod.CalcCDntpTop (per_flow_fine_step[fnum],
                             GetModifiedMidNucTime (& (a_region->nuc_shape),NucID,fnum),
                             GetModifiedSigma (& (a_region->nuc_shape),NucID));
  }
}

float * NucStep::NucFineStep(int fnum)
{
  return(per_flow_fine_step[fnum]);
}

float * NucStep::NucCoarseStep(int fnum)
{
  return(per_flow_coarse_step[fnum]);
}

void NucStep::CalculateNucRiseCoarseStep (reg_params *my_region, TimeCompression &time_c, flow_buffer_info &my_flow)
{
  DntpRiseModel dntpMod (time_c.npts,my_region->nuc_shape.C,time_c.frameNumber,ISIG_SUB_STEPS_MULTI_FLOW);

  // compute the nuc rise model for each nucleotide...in the long run could be separated out
  // and computed region-wide
  // at the moment, this computation is identical for all wells.  If it were not, we couldn't
  // do it here.
  for (int fnum=0;fnum<NUMFB;fnum++)
  {
    int NucID = my_flow.flow_ndx_map[fnum];
    i_start_coarse_step[fnum]=dntpMod.CalcCDntpTop (per_flow_coarse_step[fnum],
                               GetModifiedMidNucTime (& (my_region->nuc_shape),NucID, fnum),
                               GetModifiedSigma (& (my_region->nuc_shape),NucID));
  }
}

void NucStep::Delete()
{
  for (int fnum=0; fnum<NUMFB; fnum++)
  {
    per_flow_coarse_step[fnum] = NULL;
    per_flow_fine_step[fnum] = NULL;
  }
  if (nuc_rise_coarse_step !=NULL) delete [] nuc_rise_coarse_step;
  if (nuc_rise_fine_step !=NULL) delete [] nuc_rise_fine_step;
}
