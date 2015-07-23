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
  mid_tauB_default = 12.5;
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

  // new params for taue (im1) optimization
  if (!gopt_params["tau_E"].isNull())
    tau_E_default = gopt_params["tau_E"].asFloat();
  if (!gopt_params["min_tauB"].isNull())
    min_tauB_default = gopt_params["min_tauB"].asFloat();
  if (!gopt_params["mid_tauB"].isNull())
    mid_tauB_default = gopt_params["mid_tauB"].asFloat();
  if (!gopt_params["max_tauB"].isNull())
    max_tauB_default = gopt_params["max_tauB"].asFloat();

  // of interest: we historically have not controlled this
  if (!gopt_params["tshift"].isNull())
    tshift_default = gopt_params["tshift"].asFloat();

}

void RegionParamDefault::FromCharacterLine(char *line){
  float d[10];
  int num;

  if ( strncmp ( "km_const",line,8 ) == 0 )
  {
    num = sscanf ( line,"km_const: %f %f %f %f",&d[0],&d[1],&d[2],&d[3] );
    if ( num > 0 )
      for ( int i=0;i<NUMNUC;i++ ) kmax_default[i] = d[i];
  }
  if ( strncmp ( "krate",line,5 ) == 0 )
  {
    num = sscanf ( line,"krate: %f %f %f %f",&d[0],&d[1],&d[2],&d[3] );
    if ( num > 0 )
      for ( int i=0;i<NUMNUC;i++ ) krate_default[i] = d[i];
  }
  if ( strncmp ( "d_coeff",line,7 ) == 0 )
  {
    num = sscanf ( line,"d_coeff: %f %f %f %f",&d[0],&d[1],&d[2],&d[3] );
    if ( num > 0 )
      for ( int i=0;i<NUMNUC;i++ ) d_default[i] = d[i];
  }
  if ( strncmp ( "n_to_uM_conv",line,12 ) == 0 )
    num = sscanf ( line,"n_to_uM_conv: %f",&molecules_to_micromolar_conv );
  if ( strncmp ( "sens",line,4 ) == 0 )
    num = sscanf ( line,"sens: %f",&sens_default );
  if ( strncmp ( "tau_R_m",line,7 ) == 0 )
    num = sscanf ( line,"tau_R_m: %f",&tau_R_m_default );
  if ( strncmp ( "tau_R_o",line,7 ) == 0 )
    num = sscanf ( line,"tau_R_o: %f",&tau_R_o_default );
  if ( strncmp ( "tau_E",line,5 ) == 0 )
    num = sscanf ( line,"tau_E: %f",&tau_E_default );
  if ( strncmp ( "min_tauB",line,8 ) == 0 )
    num = sscanf ( line,"min_tauB: %f",&min_tauB_default );
  if ( strncmp ( "mid_tauB",line,8 ) == 0 )
    num = sscanf ( line,"mid_tauB: %f",&mid_tauB_default );
  if ( strncmp ( "max_tauB",line,8 ) == 0 )
    num = sscanf ( line,"max_tauB: %f",&max_tauB_default );
  if ( strncmp ( "sigma_mult", line, 10 ) == 0 )
  {
    num = sscanf ( line,"sigma_mult: %f %f %f %f", &d[0],&d[1],&d[2],&d[3] );
    for ( int i=0;i<num;i++ ) sigma_mult_default[i]=d[i];
  }
  if ( strncmp ( "t_mid_nuc_delay", line, 15 ) == 0 )
  {
    num = sscanf ( line,"t_mid_nuc_delay: %f %f %f %f", &d[0],&d[1],&d[2],&d[3] );
    for ( int i=0;i<num;i++ ) t_mid_nuc_delay_default[i]=d[i];
  }
}

void RegionParamDefault::DumpPoorlyStructuredText(){
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
  printf ( "mid_tauB: %f\n",mid_tauB_default );
  printf ( "max_tauB: %f\n",max_tauB_default );
  printf ( "tshift: %f\n",tshift_default);

}
