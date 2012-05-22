/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "DarkMatter.h"

Axion::Axion (BkgModel &_bkg) :
    bkg (_bkg)
{

}


void Axion::AccumulateOneBead(bead_params *p, reg_params *reg_p, int max_fnum, float my_residual, float res_threshold)
{
    float tmp_res[bkg.my_scratch.bead_flow_t];
  
    bkg.my_scratch.FillObserved (bkg.my_trace, p->trace_ndx);
    //params_IncrementHits(p);
    MultiFlowComputeCumulativeIncorporationSignal (p,reg_p,bkg.my_scratch.ival,*bkg.my_regions,bkg.my_scratch.cur_bead_block,bkg.time_c,bkg.my_flow,bkg.math_poiss);
    MultiFlowComputeIncorporationPlusBackground (bkg.my_scratch.fval,p,reg_p,bkg.my_scratch.ival,bkg.my_scratch.shifted_bkg,*bkg.my_regions,bkg.my_scratch.cur_buffer_block,bkg.time_c,bkg.my_flow,bkg.use_vectorization, bkg.my_scratch.bead_flow_t);    // evaluate the function
    // calculate error b/w current fit and actual data
    bkg.my_scratch.MultiFlowReturnResiduals (tmp_res);  // wait, didn't I just compute the my_residual implicitly here?

    //@TODO: compute "my_residual" here so I can avoid using lev-mar state
    // aggregate error to find first-order correction for each nuc type
    for (int fnum=0;fnum < max_fnum;fnum++)
    {
      // if our bead has enough signal in this flow to be interesting we include it
      // if our bead is close enough in mean-square distance we include it
      // exclude small amplitude but high mean-square beads
      if ((my_residual>res_threshold) && (p->Ampl[fnum]<0.3) )
        continue;

      bkg.my_regions->missing_mass.AccumulateDarkMatter(&tmp_res[bkg.time_c.npts*fnum],bkg.my_flow.flow_ndx_map[fnum]);

    }
}


void Axion::AccumulateResiduals(reg_params *reg_p, int max_fnum, bool *well_region_fit, float *residual, float res_threshold)
{

  for (int ibd=0;ibd < bkg.my_beads.numLBeads;ibd++)
  {
    if (well_region_fit[ibd] == false)
      continue;

    // get the current parameter values for this bead
    bead_params *p = &bkg.my_beads.params_nn[ibd];
    AccumulateOneBead(p,reg_p, max_fnum, residual[ibd], res_threshold);
  }
}


void Axion::CalculateDarkMatter (int max_fnum, bool *well_region_fit, float *residual, float res_threshold)
{
  bkg.my_regions->missing_mass.ResetDarkMatter();
  
  reg_params *reg_p = & bkg.my_regions->rp;
  bkg.my_scratch.FillShiftedBkg(*bkg.emptytrace, reg_p->tshift, bkg.time_c, true);
  AccumulateResiduals(reg_p,max_fnum,well_region_fit, residual, res_threshold);
  bkg.my_regions->missing_mass.NormalizeDarkMatter ();
}
