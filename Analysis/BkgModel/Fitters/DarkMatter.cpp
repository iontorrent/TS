/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "DarkMatter.h"

using namespace std;

Axion::Axion (SignalProcessingMasterFitter &_bkg) :
    bkg (_bkg)
{

}


void Axion::AccumulateOneBead(bead_params *p, reg_params *reg_p, int max_fnum, float my_residual, float res_threshold)
{
    float tmp_res[bkg.region_data->my_scratch.bead_flow_t];
  
    bkg.region_data->my_scratch.FillObserved (bkg.region_data->my_trace, p->trace_ndx);
    //params_IncrementHits(p);
    MultiFlowComputeCumulativeIncorporationSignal (p,reg_p,bkg.region_data->my_scratch.ival,bkg.region_data->my_regions.cache_step,bkg.region_data->my_scratch.cur_bead_block,bkg.region_data->time_c,bkg.region_data->my_flow,bkg.math_poiss);
    MultiFlowComputeTraceGivenIncorporationAndBackground (bkg.region_data->my_scratch.fval,p,reg_p,bkg.region_data->my_scratch.ival,bkg.region_data->my_scratch.shifted_bkg,bkg.region_data->my_regions,bkg.region_data->my_scratch.cur_buffer_block,bkg.region_data->time_c,bkg.region_data->my_flow,bkg.global_defaults.signal_process_control.use_vectorization, bkg.region_data->my_scratch.bead_flow_t);    // evaluate the function
    // calculate error b/w current fit and actual data
    bkg.region_data->my_scratch.MultiFlowReturnResiduals (tmp_res);  // wait, didn't I just compute the my_residual implicitly here?

    //@TODO: compute "my_residual" here so I can avoid using lev-mar state
    // aggregate error to find first-order correction for each nuc type
    for (int fnum=0;fnum < max_fnum;fnum++)
    {
      // if our bead has enough signal in this flow to be interesting we include it
      // if our bead is close enough in mean-square distance we include it
      // exclude small amplitude but high mean-square beads
      if ((my_residual>res_threshold) && (p->Ampl[fnum]<0.3) )
        continue;

      bkg.region_data->my_regions.missing_mass.AccumulateDarkMatter(&tmp_res[bkg.region_data->time_c.npts()*fnum],bkg.region_data->my_flow.flow_ndx_map[fnum]);

    }
}


void Axion::AccumulateResiduals(reg_params *reg_p, int max_fnum, float *residual, float res_threshold)
{

  for (int ibd=0;ibd < bkg.region_data->my_beads.numLBeads;ibd++)
  {
    if ( bkg.region_data->my_beads.StillSampled(ibd) ) {
      // get the current parameter values for this bead
      bead_params *p = &bkg.region_data->my_beads.params_nn[ibd];
      AccumulateOneBead(p,reg_p, max_fnum, residual[ibd], res_threshold);
    }
  }
}

void Axion::CalculateDarkMatter (int max_fnum, float *residual, float res_threshold)
{
  bkg.region_data->my_regions.missing_mass.ResetDarkMatter();
  // prequel, set up standard bits
  reg_params *reg_p = & bkg.region_data->my_regions.rp;
  bkg.region_data->my_scratch.FillShiftedBkg(*bkg.region_data->emptytrace, reg_p->tshift, bkg.region_data->time_c, true);
  bkg.region_data->my_regions.cache_step.ForceLockCalculateNucRiseCoarseStep(reg_p,bkg.region_data->time_c,bkg.region_data->my_flow);

  AccumulateResiduals(reg_p,max_fnum, residual, res_threshold);

  bkg.region_data->my_regions.missing_mass.NormalizeDarkMatter ();
  // make sure everything is happy in the rest of the code
  bkg.region_data->my_regions.cache_step.Unlock();
}
