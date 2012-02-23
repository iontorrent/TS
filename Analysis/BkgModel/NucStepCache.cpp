/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "NucStepCache.h"
#include "DNTPRiseModel.h"

NucStep::NucStep()
{
  nuc_rise_fine_step = NULL;
  nuc_rise_coarse_step = NULL;
  for (int NucID=0;NucID<NUMNUC;NucID++)
  {
    i_start_coarse_step[NucID] = 0;
    i_start_fine_step[NucID] = 0;
    per_nuc_coarse_step[NucID] = NULL;
    per_nuc_fine_step[NucID] = NULL;
  }
  nuc_flow_t = 0;
}

void NucStep::Alloc (int npts)
{
  nuc_flow_t = NUMNUC * npts;
  if (nuc_rise_coarse_step==NULL)
    nuc_rise_coarse_step=new float[nuc_flow_t*ISIG_SUB_STEPS_MULTI_FLOW];
  if (nuc_rise_fine_step==NULL)
    nuc_rise_fine_step=new float[nuc_flow_t*ISIG_SUB_STEPS_SINGLE_FLOW];
  for (int NucID=0; NucID<NUMNUC; NucID++)
  {
    per_nuc_coarse_step[NucID] = &nuc_rise_coarse_step[NucID*ISIG_SUB_STEPS_MULTI_FLOW*npts];
    per_nuc_fine_step[NucID] = &nuc_rise_fine_step[NucID*ISIG_SUB_STEPS_SINGLE_FLOW*npts];
  }
}

void NucStep::CalculateNucRiseFineStep (reg_params *a_region, TimeCompression &time_c)
{
  DntpRiseModel dntpMod (time_c.npts,a_region->nuc_shape.C,time_c.frameNumber,ISIG_SUB_STEPS_SINGLE_FLOW);

  // compute the nuc rise model for each nucleotide...in the long run could be separated out
  // and computed region-wide
  // at the moment, this computation is identical for all wells.  If it were not, we couldn't
  // do it here.
  for (int NucID=0;NucID<NUMNUC;NucID++)
  {
    i_start_fine_step[NucID]=dntpMod.CalcCDntpTop (per_nuc_fine_step[NucID],
                             GetModifiedMidNucTime (& (a_region->nuc_shape),NucID),
                             GetModifiedSigma (& (a_region->nuc_shape),NucID));
  }
}

float * NucStep::NucFineStep(int NucID)
{
  return(per_nuc_fine_step[NucID]);
}

float * NucStep::NucCoarseStep(int NucID)
{
  return(per_nuc_coarse_step[NucID]);
}

void NucStep::CalculateNucRiseCoarseStep (reg_params *my_region, TimeCompression &time_c)
{
  DntpRiseModel dntpMod (time_c.npts,my_region->nuc_shape.C,time_c.frameNumber,ISIG_SUB_STEPS_MULTI_FLOW);

  // compute the nuc rise model for each nucleotide...in the long run could be separated out
  // and computed region-wide
  // at the moment, this computation is identical for all wells.  If it were not, we couldn't
  // do it here.
  for (int NucID=0;NucID<NUMNUC;NucID++)
  {
    i_start_coarse_step[NucID]=dntpMod.CalcCDntpTop (per_nuc_coarse_step[NucID],
                               GetModifiedMidNucTime (& (my_region->nuc_shape),NucID),
                               GetModifiedSigma (& (my_region->nuc_shape),NucID));
  }
}

void NucStep::Delete()
{
  for (int NucID=0; NucID<NUMNUC; NucID++)
  {
    per_nuc_coarse_step[NucID] = NULL;
    per_nuc_fine_step[NucID] = NULL;
  }
  if (nuc_rise_coarse_step !=NULL) delete [] nuc_rise_coarse_step;
  if (nuc_rise_fine_step !=NULL) delete [] nuc_rise_fine_step;
}
