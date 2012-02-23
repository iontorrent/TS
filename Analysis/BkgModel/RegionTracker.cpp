/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "RegionTracker.h"


RegionTracker::RegionTracker()
{

}

void RegionTracker::AllocScratch (int npts)
{
  cache_step.Alloc (npts);
  missing_mass.Alloc (npts);
}


void RegionTracker::Delete()
{
  cache_step.Delete();
  missing_mass.Delete();
}



void RegionTracker::RestrictRatioDrift()
{
  // we don't allow the RatioDrift term to increase after it's initial fitted value is determined
  rp_high.RatioDrift = rp.RatioDrift;

  if (rp_high.RatioDrift < MIN_RDR_HIGH_LIMIT) rp_high.RatioDrift = MIN_RDR_HIGH_LIMIT;
}



RegionTracker::~RegionTracker()
{
  Delete();
}

// Region parameters & box constraints

void RegionTracker::InitHighRegionParams (float t_mid_nuc_start)
{
  reg_params_setStandardHigh (&rp_high,t_mid_nuc_start);
}

void RegionTracker::InitLowRegionParams (float t_mid_nuc_start)
{
  reg_params_setStandardLow (&rp_low,t_mid_nuc_start);
}

void RegionTracker::InitModelRegionParams (float t_mid_nuc_start,float sigma_start, float dntp_concentration_in_uM,GlobalDefaultsForBkgModel &global_defaults)
{
  // s is overly complex
  reg_params_setStandardValue (&rp,t_mid_nuc_start,sigma_start, dntp_concentration_in_uM);
  reg_params_setKrate (&rp,global_defaults.krate_default);
  reg_params_setDiffusion (&rp,global_defaults.d_default);
  reg_params_setKmax (&rp,global_defaults.kmax_default);
  reg_params_setSens (&rp,global_defaults.sens_default);
  reg_params_setBuffModel (&rp,global_defaults.tau_R_m_default,global_defaults.tau_R_o_default);
  reg_params_setSigmaMult (&rp,global_defaults.sigma_mult_default);
  reg_params_setT_mid_nuc_delay (&rp,global_defaults.t_mid_nuc_delay_default);
  if (global_defaults.no_RatioDrift_fit_first_20_flows)
    reg_params_setNoRatioDriftValues (&rp);
}

void RegionTracker::InitRegionParams (float t_mid_nuc_start,float sigma_start, float dntp_concentration_in_uM,GlobalDefaultsForBkgModel &global_defaults)
{
  InitHighRegionParams (t_mid_nuc_start);
  InitLowRegionParams (t_mid_nuc_start);
  InitModelRegionParams (t_mid_nuc_start,sigma_start,dntp_concentration_in_uM,global_defaults);
  // bounds check all these to be sure
  reg_params_ApplyLowerBound (&rp,&rp_low);
  reg_params_ApplyUpperBound (&rp,&rp_high);
}
