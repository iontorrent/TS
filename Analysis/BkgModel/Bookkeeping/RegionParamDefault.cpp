/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#include "RegionParamDefault.h"

static float _kmax_default[NUMNUC]  = { 18.0,   20.0,   17.0,   18.0 };
static float _krate_default[NUMNUC] = { 18.78,   20.032,   25.04,   31.3 };
static float _d_default[NUMNUC]     = {159.923,189.618,227.021,188.48};
static float _sigma_mult_default[NUMNUC] = {1.162,1.124,1.0,0.8533};
static float _t_mid_nuc_delay_default[NUMNUC] = {0.69,1.78,0.01,0.17};
static float _dntp_uM[NUMNUC] = {50.0f,50.0f,50.0f,50.0f};


RegionParamDefault::RegionParamDefault()
{
  memcpy ( kmax_default,_kmax_default,sizeof ( float[4] ) );
  memcpy ( krate_default,_krate_default, sizeof ( float[4] ) );
  memcpy ( d_default,_d_default,sizeof ( float[4] ) );
  memcpy ( sigma_mult_default, _sigma_mult_default,sizeof ( float[4] ) );
  memcpy ( t_mid_nuc_delay_default, _t_mid_nuc_delay_default,sizeof ( float[4] ) );
  memcpy ( dntp_uM, _dntp_uM, sizeof(float[4]));
  sens_default = 1.256;
  molecules_to_micromolar_conv = 0.000062f;
  tau_R_m_default = -24.36;
  tau_R_o_default = 25.16;
  tau_E_default = tau_R_m_default + tau_R_o_default;
  min_tauB_default = 4;

  max_tauB_default = 65;
  tshift_default = 0.4f;

}

//@TODO: this is a bad, bad idea, due only to the fact that our default input is structured poorly
//replace this when the JSON files are used with an actual input
void RegionParamDefault::BadIdeaComputeDerivedInput(){
  tau_E_default = tau_R_m_default + tau_R_o_default;

}



//@TODO: what if a parameter is omitted here?
//@TODO: what if parameters are over-ridden from commands (or do we want to force a file?)
void RegionParamDefault::FromJson(Json::Value &gopt_params){
  const Json::Value km_const = gopt_params["km_const"];
  for ( int index = 0; index < (int) km_const.size(); ++index )
    kmax_default[index] = km_const[index].asFloat();

  const Json::Value krate = gopt_params["krate"];
  for ( int index = 0; index < (int) krate.size(); ++index )
    krate_default[index] = krate[index].asFloat();

  const Json::Value d_coeff = gopt_params["d_coeff"];
  for ( int index = 0; index < (int) d_coeff.size(); ++index )
    d_default[index] = d_coeff[index].asFloat();

  const Json::Value sigma_mult = gopt_params["sigma_mult"];
  for ( int index = 0; index < (int) sigma_mult.size(); ++index )
    sigma_mult_default[index] = sigma_mult[index].asFloat();

  const Json::Value t_mid_nuc_delay = gopt_params["t_mid_nuc_delay"];
  for ( int index = 0; index < (int) t_mid_nuc_delay.size(); ++index )
    t_mid_nuc_delay_default[index] = t_mid_nuc_delay[index].asFloat();

  sens_default = gopt_params["sens"].asFloat();
  if (!gopt_params["molecules_to_micromolar_conv"].isNull())  // this is bad, we need to abort/default clearly here - Jason is fixing functions for this.
    molecules_to_micromolar_conv = gopt_params["molecules_to_micromolar_conv"].asFloat();
  tau_R_m_default = gopt_params["tau_R_m"].asFloat();
  tau_R_o_default = gopt_params["tau_R_o"].asFloat();

  if (tau_R_m_default>0){
    printf("Alert: Unphysical starting parameters for tau_R_m %f\n", tau_R_m_default);
  }
  if ((tau_R_o_default<4) or (tau_R_m_default>tau_R_o_default) or (tau_R_o_default>4*tau_R_m_default)){
    printf("Alert: suspiscious values for tau_R_o tau_R_m: %f %f\n", tau_R_o_default, tau_R_m_default);
  }

  // new params for taue (im1) optimization
  if (!gopt_params["tau_E"].isNull())
    tau_E_default = gopt_params["tau_E"].asFloat();
  if (!gopt_params["min_tauB"].isNull())
    min_tauB_default = gopt_params["min_tauB"].asFloat();

  if (!gopt_params["max_tauB"].isNull())
    max_tauB_default = gopt_params["max_tauB"].asFloat();

  if ((max_tauB_default<4*min_tauB_default) or (min_tauB_default<4)){
    printf("Alert: suspicious range for tauB: %f %f\n", min_tauB_default, max_tauB_default);
  }

  // of interest: we historically have not controlled this
  if (!gopt_params["tshift"].isNull())
    tshift_default = gopt_params["tshift"].asFloat();

  // add concentration back as a tunable parameter
  if (!gopt_params["dntp"].isNull()){
    const Json::Value dntp = gopt_params["dntp"];
    for ( int index = 0; index < (int) dntp.size(); ++index )
      dntp_uM[index] = dntp[index].asFloat();

  }

}

void RegionParamDefault::DumpPoorlyStructuredText(){
  printf ( "dntp_uM: %f\t%f\t%f\t%f\n",dntp_uM[0],dntp_uM[1],dntp_uM[2],dntp_uM[3] );
  printf ( "km_const: %f\t%f\t%f\t%f\n",kmax_default[0],kmax_default[1],kmax_default[2],kmax_default[3] );
  printf ( "krate: %f\t%f\t%f\t%f\n",krate_default[0],krate_default[1],krate_default[2],krate_default[3] );
  printf ( "d_coeff: %f\t%f\t%f\t%f\n",d_default[0],d_default[1],d_default[2],d_default[3] );
  printf ( "sigma_mult: %f\t%f\t%f\t%f\n",sigma_mult_default[0],sigma_mult_default[1],sigma_mult_default[2],sigma_mult_default[3] );
  printf ( "t_mid_nuc_delay: %f\t%f\t%f\t%f\n",t_mid_nuc_delay_default[0],t_mid_nuc_delay_default[1],t_mid_nuc_delay_default[2],t_mid_nuc_delay_default[3] );
  printf ( "sens: %f\n",sens_default );
  printf ( "n_to_uM_conv: %f\n",molecules_to_micromolar_conv );
  printf ( "tau_R_m: %f\n",tau_R_m_default );
  printf ( "tau_R_o: %f\n",tau_R_o_default );
  printf ( "tau_E: %f\n",tau_E_default );
  printf ( "min_tauB: %f\n",min_tauB_default );
  printf ( "max_tauB: %f\n",max_tauB_default );
  printf ( "tshift: %f\n",tshift_default);

}
