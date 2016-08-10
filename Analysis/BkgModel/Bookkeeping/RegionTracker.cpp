/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "RegionTracker.h"
#include "CommandLineOpts.h"

RegionTracker::RegionTracker()
{
  restart = false;
  tmidnuc_smoother.SetAccessFn( & reg_params::AccessTMidNuc );
  copy_drift_smoother.SetAccessFn( & reg_params::AccessCopyDrift );
  ratio_drift_smoother.SetAccessFn( & reg_params::AccessRatioDrift );
}

RegionTracker::RegionTracker( const CommandLineOpts * inception_state )
  : rp(), rp_high(), rp_low(),
  tmidnuc_smoother( inception_state->bkg_control.regional_smoothing.alpha,
                    inception_state->bkg_control.regional_smoothing.gamma,
                    & reg_params::AccessTMidNuc ), 
  copy_drift_smoother( inception_state->bkg_control.regional_smoothing.alpha,
                       inception_state->bkg_control.regional_smoothing.gamma,
                       & reg_params::AccessCopyDrift ),
  ratio_drift_smoother( inception_state->bkg_control.regional_smoothing.alpha,
                        inception_state->bkg_control.regional_smoothing.gamma,
                        & reg_params::AccessRatioDrift )
{
  restart = false;
}

void RegionTracker::AllocScratch (int npts, int flow_block_size)
{
  cache_step.Alloc (npts, flow_block_size);

  if ( !restart )
    missing_mass.Alloc (npts);
}


void RegionTracker::Delete()
{
  missing_mass.Delete();
}



void RegionTracker::RestrictRatioDrift()
{
  // we don't allow the RatioDrift term to increase after it's initial fitted value is determined

  rp_high.RatioDrift = rp.RatioDrift;

  if (!rp.use_alternative_etbR_equation){
    if (rp_high.RatioDrift < MIN_RDR_HIGH_LIMIT_OLD) rp_high.RatioDrift = MIN_RDR_HIGH_LIMIT_OLD;
  }
  else{
      if (rp_high.RatioDrift < MIN_RDR_HIGH_LIMIT) rp_high.RatioDrift = MIN_RDR_HIGH_LIMIT;
  }
      
}



RegionTracker::~RegionTracker()
{
  Delete();
}

// Region parameters & box constraints

void RegionTracker::InitHighRegionParams (float t_mid_nuc_start, int flow_block_size)
{
  rp_high.SetStandardHigh (t_mid_nuc_start, flow_block_size);
}

void RegionTracker::InitLowRegionParams (float t_mid_nuc_start,GlobalDefaultsForBkgModel &global_defaults, int flow_block_size)
{
  rp_low.SetStandardLow (t_mid_nuc_start, flow_block_size, global_defaults.signal_process_control.suppress_copydrift);
}

void RegionTracker::InitModelRegionParams (float t_mid_nuc_start,float sigma_start,
                                           GlobalDefaultsForBkgModel &global_defaults,
                                           int flow_block_size)
{
  // s is overly complex
  rp.SetStandardValue (t_mid_nuc_start,sigma_start, global_defaults.region_param_start.dntp_uM,
                       global_defaults.signal_process_control.fitting_taue,

                       global_defaults.signal_process_control.use_alternative_etbR_equation,
                       global_defaults.signal_process_control.suppress_copydrift,
                       global_defaults.signal_process_control.safe_model,
                       global_defaults.signal_process_control.hydrogenModelType, flow_block_size);
  rp.SetTshift(global_defaults.region_param_start.tshift_default);
  reg_params_setKrate (&rp,global_defaults.region_param_start.krate_default);
  reg_params_setDiffusion (&rp,global_defaults.region_param_start.d_default);
  reg_params_setKmax (&rp,global_defaults.region_param_start.kmax_default);
  reg_params_setSens (&rp,global_defaults.region_param_start.sens_default);
  if( global_defaults.signal_process_control.fitting_taue )
    reg_params_setBuffModel (&rp,global_defaults.region_param_start.tau_E_default);
  else
    reg_params_setBuffModel (&rp,global_defaults.region_param_start.tau_R_m_default,global_defaults.region_param_start.tau_R_o_default);
  reg_params_setBuffRange (&rp,global_defaults.region_param_start.min_tauB_default,global_defaults.region_param_start.max_tauB_default);

  reg_params_setSigmaMult (&rp,global_defaults.region_param_start.sigma_mult_default);
  reg_params_setT_mid_nuc_delay (&rp,global_defaults.region_param_start.t_mid_nuc_delay_default);
  reg_params_setConversion(&rp, global_defaults.region_param_start.molecules_to_micromolar_conv);
  if (global_defaults.signal_process_control.no_RatioDrift_fit_first_20_flows)
    reg_params_setNoRatioDriftValues (&rp);
}

void RegionTracker::InitRegionParams (float t_mid_nuc_start,float sigma_start, GlobalDefaultsForBkgModel &global_defaults, int flow_block_size )
{
  InitHighRegionParams (t_mid_nuc_start, flow_block_size);
  InitLowRegionParams (t_mid_nuc_start, global_defaults, flow_block_size);
  InitModelRegionParams (t_mid_nuc_start,sigma_start,global_defaults, flow_block_size);
  // bounds check all these to be sure
  rp.ApplyLowerBound (&rp_low, flow_block_size );
  rp.ApplyUpperBound (&rp_high, flow_block_size );
}

void RegionTracker::ResetLocalRegionParams( int flow_block_size )
{
  rp.nuc_shape.ResetPerFlowTimeShift( flow_block_size );
}
