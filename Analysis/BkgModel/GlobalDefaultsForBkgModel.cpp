/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <sys/stat.h>
#include <fstream>
#include "GlobalDefaultsForBkgModel.h"
#include "Utils.h"
#include "IonErr.h"
#include "json/json.h"
#include <boost/algorithm/string/predicate.hpp>


static float _kmax_default[NUMNUC]  = { 18.0,   20.0,   17.0,   18.0 };
static float _krate_default[NUMNUC] = { 18.78,   20.032,   25.04,   31.3 };
static float _d_default[NUMNUC]     = {159.923,189.618,227.021,188.48};
static float _sigma_mult_default[NUMNUC] = {1.162,1.124,1.0,0.8533};
static float _t_mid_nuc_delay_default[NUMNUC] = {0.69,1.78,0.01,0.17};
static float _clonal_call_scale[MAGIC_CLONAL_CALL_ARRAY_SIZE] = {0.902,0.356,0.078,0.172,0.436,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
static float  _emp[NUMEMPHASISPARAMETERS]  = {6.86, 1.1575, 2.081, 1.230, 7.2625, 1.91, 0.0425, 19.995};
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
}

LocalSigProcControl::LocalSigProcControl()
{
  AmplLowerLimit = 0.001f;

// a set of relevant parameters for allowing the krate to vary
  krate_adj_limit = 2.0f; // you must be at least this tall to ride the ride
  dampen_kmult = 0.0f;
  kmult_low_limit = 0.65;
  kmult_hi_limit = 1.75f;

  ssq_filter = 0.0f;
  choose_time = 0; // normal time compression


  var_kmult_only = false;
  generic_test_flag = false;
  projection_search_enable = false;
  fit_alternate = false;
  fit_gauss_newton = false;

  do_clonal_filter = true;
  enable_dark_matter = true;
  use_vectorization = true;
  proton_dot_wells_post_correction = false;
  single_flow_fit_max_retry = 0;
  per_flow_t_mid_nuc_tracking = false;
  exp_tail_fit = false;
  pca_dark_matter = false;
  regional_sampling = false;
  regional_sampling_type = -1;
  no_RatioDrift_fit_first_20_flows = false;
  fitting_taue = false;
  prefilter_beads = false;
  amp_guess_on_gpu = false;
  max_frames = 0;
}

TimeAndEmphasisDefaults::TimeAndEmphasisDefaults()
{
  nuc_flow_frame_width = 22.5;  // 1.5 seconds * 15 frames/second
  time_left_avg = 5;
  time_start_detail = -5;
  time_stop_detail = 16;
  point_emphasis_by_compression = true;

  memcpy(emp,_emp,sizeof(float[NUMEMPHASISPARAMETERS]));
  emphasis_ampl_default = 7.25;
  emphasis_width_default = 2.89;
}

FitterDefaults::FitterDefaults()
{
  memcpy(clonal_call_scale,_clonal_call_scale,sizeof(float[MAGIC_CLONAL_CALL_ARRAY_SIZE]));
  clonal_call_penalty = 1600.0f;
  shrink_factor = 0.0f;
}

void GlobalDefaultsForBkgModel::SetDntpUM ( float concentration, int NucID ) { region_param_start.dntp_uM[NucID] = concentration;  }
float GlobalDefaultsForBkgModel::GetDntpUM(int NucID) { return region_param_start.dntp_uM[NucID]; }
void GlobalDefaultsForBkgModel::SetChipType ( char *name )
{
  chipType=name;
}
void GlobalDefaultsForBkgModel::ReadXtalk ( char *name )
{
  xtalk_name=name;
}

void GlobalDefaultsForBkgModel::SetAllConcentrations (float *_dntp_uM){
    memcpy(region_param_start.dntp_uM,_dntp_uM,sizeof(float[4]));
}


#define MAX_LINE_LEN    2048
#define MAX_DATA_PTS    80
// Load optimized defaults from GeneticOptimizer runs
void GlobalDefaultsForBkgModel::SetGoptDefaults ( char *fname )
{
  if(fname == NULL)
    return;
  std::string fnameStr(fname);
  bool isJson = false; //default is .param
  //check file format based on suffix
  if (boost::algorithm::ends_with(fnameStr, ".json"))
    isJson = true;
  //json way
  if(isJson){
      Json::Value gopt_params;
      std::ifstream in(fname, std::ios::in);

      if (!in.good()) {
        printf("Opening gopt parameter file %s unsuccessful. Aborting\n", fname);
        exit(1);
      }
      in >> gopt_params;

      const Json::Value km_const = gopt_params["parameters"]["km_const"];
      for ( int index = 0; index < (int) km_const.size(); ++index )
         region_param_start.kmax_default[index] = km_const[index].asFloat();

      const Json::Value krate = gopt_params["parameters"]["krate"];
      for ( int index = 0; index < (int) krate.size(); ++index )
         region_param_start.krate_default[index] = krate[index].asFloat();

      const Json::Value d_coeff = gopt_params["parameters"]["d_coeff"];
      for ( int index = 0; index < (int) d_coeff.size(); ++index )
         region_param_start.d_default[index] = d_coeff[index].asFloat();

      const Json::Value sigma_mult = gopt_params["parameters"]["sigma_mult"];
      for ( int index = 0; index < (int) sigma_mult.size(); ++index )
         region_param_start.sigma_mult_default[index] = sigma_mult[index].asFloat();

      const Json::Value t_mid_nuc_delay = gopt_params["parameters"]["t_mid_nuc_delay"];
      for ( int index = 0; index < (int) t_mid_nuc_delay.size(); ++index )
         region_param_start.t_mid_nuc_delay_default[index] = t_mid_nuc_delay[index].asFloat();

      region_param_start.sens_default = gopt_params["parameters"]["sens"].asFloat();
      region_param_start.molecules_to_micromolar_conv = gopt_params["parameters"]["molecules_to_micromolar_conv"].asFloat();
      region_param_start.tau_R_m_default = gopt_params["parameters"]["tau_R_m"].asFloat();
      region_param_start.tau_R_o_default = gopt_params["parameters"]["tau_R_o"].asFloat();

      const Json::Value emphasis = gopt_params["parameters"]["emphasis"];
      for ( int index = 0; index < (int) emphasis.size(); ++index )
         data_control.emp[index] = emphasis[index].asFloat();

      data_control.emphasis_ampl_default = gopt_params["parameters"]["emp_amplitude"].asFloat();
      data_control.emphasis_width_default = gopt_params["parameters"]["emp_width"].asFloat();

      const Json::Value clonal_call_scale = gopt_params["parameters"]["clonal_call_scale"];
      for ( int index = 0; index < (int) clonal_call_scale.size(); ++index )
         fitter_defaults.clonal_call_scale[index] = clonal_call_scale[index].asFloat();

      fitter_defaults.shrink_factor = gopt_params["parameters"]["shrink_factor"].asFloat();

      in.close();
  } else{
      //old way
      struct stat fstatus;
      int         status;
      FILE *param_file;
      char *line;
      int nChar = MAX_LINE_LEN;
      float d[10];

      int num = 0;

      line = new char[MAX_LINE_LEN];

      status = stat ( fname,&fstatus );

      if ( status == 0 )
      {
        // file exists
        printf ( "GOPT: loading parameters from %s\n",fname );

        param_file=fopen ( fname,"rt" );

        bool done = false;

        while ( !done )
        {
          int bytes_read = getline ( &line, ( size_t * ) &nChar,param_file );

          if ( bytes_read > 0 )
          {
            if ( bytes_read >= MAX_LINE_LEN || bytes_read < 0 )
            {
              ION_ABORT ( "Read: " + ToStr ( bytes_read ) + " into a buffer only: " +
                          ToStr ( MAX_LINE_LEN ) + " long for line: '" + ToStr ( line ) + "'" );
            }
            line[bytes_read]='\0';
            if ( strncmp ( "km_const",line,8 ) == 0 )
            {
              num = sscanf ( line,"km_const: %f %f %f %f",&d[0],&d[1],&d[2],&d[3] );
              if ( num > 0 )
                for ( int i=0;i<NUMNUC;i++ ) region_param_start.kmax_default[i] = d[i];
            }
            if ( strncmp ( "krate",line,5 ) == 0 )
            {
              num = sscanf ( line,"krate: %f %f %f %f",&d[0],&d[1],&d[2],&d[3] );
              if ( num > 0 )
                for ( int i=0;i<NUMNUC;i++ ) region_param_start.krate_default[i] = d[i];
            }
            if ( strncmp ( "d_coeff",line,7 ) == 0 )
            {
              num = sscanf ( line,"d_coeff: %f %f %f %f",&d[0],&d[1],&d[2],&d[3] );
              if ( num > 0 )
                for ( int i=0;i<NUMNUC;i++ ) region_param_start.d_default[i] = d[i];
            }
            if ( strncmp ( "n_to_uM_conv",line,12 ) == 0 )
              num = sscanf ( line,"n_to_uM_conv: %f",&region_param_start.molecules_to_micromolar_conv );
            if ( strncmp ( "sens",line,4 ) == 0 )
              num = sscanf ( line,"sens: %f",&region_param_start.sens_default );
            if ( strncmp ( "tau_R_m",line,7 ) == 0 )
              num = sscanf ( line,"tau_R_m: %f",&region_param_start.tau_R_m_default );
            if ( strncmp ( "tau_R_o",line,7 ) == 0 )
              num = sscanf ( line,"tau_R_o: %f",&region_param_start.tau_R_o_default );
            if ( strncmp ( "sigma_mult", line, 10 ) == 0 )
            {
              num = sscanf ( line,"sigma_mult: %f %f %f %f", &d[0],&d[1],&d[2],&d[3] );
              for ( int i=0;i<num;i++ ) region_param_start.sigma_mult_default[i]=d[i];
            }
            if ( strncmp ( "t_mid_nuc_delay", line, 15 ) == 0 )
            {
              num = sscanf ( line,"t_mid_nuc_delay: %f %f %f %f", &d[0],&d[1],&d[2],&d[3] );
              for ( int i=0;i<num;i++ ) region_param_start.t_mid_nuc_delay_default[i]=d[i];
            }
            if ( strncmp ( "emphasis",line,8 ) == 0 )
            {
              num = sscanf ( line,"emphasis: %f %f %f %f %f %f %f %f", &d[0],&d[1],&d[2],&d[3],&d[4],&d[5],&d[6],&d[7] );
              for ( int i=0;i<num;i++ ) data_control.emp[i]=d[i];
            }
            if ( strncmp ( "emp_amp",line,7 ) == 0 )
              num = sscanf ( line,"emp_amplitude: %f",&data_control.emphasis_ampl_default );
            if ( strncmp ( "emp_width",line,7 ) == 0 )
              num = sscanf ( line,"emp_width: %f",&data_control.emphasis_width_default );

            if ( strncmp ( "clonal_call_scale",line,17 ) == 0 )
            {
              num = sscanf ( line,"clonal_call_scale: %f %f %f %f %f", &d[0],&d[1],&d[2],&d[3],&d[4] );
              for ( int i=0;i<num;i++ ) fitter_defaults.clonal_call_scale[i]=d[i];
            }
            if ( strncmp ( "shrink_factor:", line, 14 ) == 0 )
            {
              float t_shrink;
              num = sscanf ( line,"shrink_factor: %f", &t_shrink );
              fitter_defaults.shrink_factor = t_shrink;
            }
            if ( strncmp ( "nuc_flow_timing", line, 15 ) == 0 )
            {
              num = sscanf ( line,"nuc_flow_frame_width: %f", &d[0] );
              data_control.nuc_flow_frame_width = d[0];
            }
            if ( strncmp ( "time_compression", line, 16 ) == 0 )
            {
              num = sscanf ( line,"time_compression: %f %f %f", &d[0], &d[1],&d[2] );
              data_control.time_left_avg = d[0];
              data_control.time_start_detail = d[1];
              data_control.time_stop_detail = d[2];
            }
          }
          else
            done = true;
        }

        fclose ( param_file );
      }
      else
        printf ( "GOPT: parameter file %s does not exist\n",fname );

      delete [] line;
  }

  DumpExcitingParameters("default");
}

void GlobalDefaultsForBkgModel::DumpExcitingParameters(char *fun_string)
{
      //output defaults used
    printf ( "%s parameters used: \n",fun_string );

    printf ( "kmax: %f\t%f\t%f\t%f\n",region_param_start.kmax_default[0],region_param_start.kmax_default[1],region_param_start.kmax_default[2],region_param_start.kmax_default[3] );
    printf ( "krate: %f\t%f\t%f\t%f\n",region_param_start.krate_default[0],region_param_start.krate_default[1],region_param_start.krate_default[2],region_param_start.krate_default[3] );
    printf ( "d: %f\t%f\t%f\t%f\n",region_param_start.d_default[0],region_param_start.d_default[1],region_param_start.d_default[2],region_param_start.d_default[3] );
    printf ( "sigma_mult: %f\t%f\t%f\t%f\n",region_param_start.sigma_mult_default[0],region_param_start.sigma_mult_default[1],region_param_start.sigma_mult_default[2],region_param_start.sigma_mult_default[3] );
    printf ( "t_mid_nuc_delay: %f\t%f\t%f\t%f\n",region_param_start.t_mid_nuc_delay_default[0],region_param_start.t_mid_nuc_delay_default[1],region_param_start.t_mid_nuc_delay_default[2],region_param_start.t_mid_nuc_delay_default[3] );
    printf ( "sens: %f\n",region_param_start.sens_default );
    printf ( "molecules_to_micromolar_conv: %f\n",region_param_start.molecules_to_micromolar_conv );
    printf ( "tau_R_m: %f\n",region_param_start.tau_R_m_default );
    printf ( "tau_R_o: %f\n",region_param_start.tau_R_o_default );

    printf ( "emp: %f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",data_control.emp[0],data_control.emp[1],data_control.emp[2],data_control.emp[3],data_control.emp[4],data_control.emp[5],data_control.emp[6],data_control.emp[7] );
    printf ( "emp_amplitude: \t%f\n",data_control.emphasis_ampl_default );
    printf ( "emp_width: \t%f\n",data_control.emphasis_width_default );
    printf ( "clonal_call_scale: %f\t%f\t%f\t%f\t%f\n",fitter_defaults.clonal_call_scale[0], fitter_defaults.clonal_call_scale[1], fitter_defaults.clonal_call_scale[2], fitter_defaults.clonal_call_scale[3], fitter_defaults.clonal_call_scale[4] );
    printf( "shrink_factor: %f\n", fitter_defaults.shrink_factor);
    printf ( "\n" );
}

// This function is used during GeneticOptimizer runs in which case the above SetGoptDefaults is disabled
void GlobalDefaultsForBkgModel::ReadEmphasisVectorFromFile ( char *experimentName )
{
  char fname[512];
  FILE *evect_file;
  char *line = new char[MAX_LINE_LEN];
  float read_data[MAX_DATA_PTS];
  int nChar = MAX_LINE_LEN;
  int pset=0;

  struct stat fstatus;
  sprintf ( fname,"%s/emphasis_vector.txt", experimentName );
  int status = stat ( fname,&fstatus );
  if ( status == 0 )    // file exists
  {
    printf ( "loading emphasis vector parameters from %s\n",fname );

    evect_file=fopen ( fname,"rt" );

    // first line contains the number of points
    int bytes_read = getline ( &line, ( size_t * ) &nChar,evect_file );

    if ( bytes_read > 0 )
    {
      int evect_size;
      sscanf ( line,"%d",&evect_size );
      printf ("nps=%d",evect_size);


      for (int i=0; ( i < evect_size ) && ( i < MAX_DATA_PTS );i++ )
      {
        bytes_read = getline ( &line, ( size_t * ) &nChar,evect_file );
        sscanf ( line,"%f",&read_data[i] );
        printf ("\t%f",read_data[i]);
      }
      printf ("\n");

      // pick 3 gopt parameter sets:
      // 0 = all params, 1 - no nuc-dep factors, 2 - no nuc-dep factors plus no emphasis, 3 - only emphasis params
      if ( evect_size == 13 ) pset = 2;
      else if ( evect_size == 23 ) pset = 1;
      else if ( evect_size == 43 ) pset = 0;
      else if ( evect_size == 10 ) pset = 3;
      else if ( evect_size == 4 ) pset = 4;
      else if ( evect_size == 6 ) pset = 5;
      else if ( evect_size == 7 ) pset = 6;
      else if ( evect_size == 8 ) pset = 7;
      else if ( evect_size == 11 ) pset = 8;
      else if ( evect_size == 12 ) pset = 9;
      else if ( evect_size == 38 ) pset = 10;
      else
      {
        fprintf ( stderr, "Unrecognized number of points (%d) in %s\n", evect_size, fname );
        exit ( 1 );
      }
    }
    fclose ( evect_file );

    // copy the configuration values into the right places
    int dv = 0;
    if ( pset == 0 || pset == 1 || pset == 2 ) 
    {
    // first value scales add km terms
    region_param_start.kmax_default[TNUCINDEX] *= read_data[dv];
    region_param_start.kmax_default[ANUCINDEX] *= read_data[dv];
    region_param_start.kmax_default[CNUCINDEX] *= read_data[dv];
    region_param_start.kmax_default[GNUCINDEX] *= read_data[dv++];
    }
    // 2-5 values scale individual terms
    if ( pset == 0 )
    {
      region_param_start.kmax_default[TNUCINDEX] *= read_data[dv++];
      region_param_start.kmax_default[ANUCINDEX] *= read_data[dv++];
      region_param_start.kmax_default[CNUCINDEX] *= read_data[dv++];
      region_param_start.kmax_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 || pset == 1 || pset == 2 ) 
    {
      region_param_start.krate_default[TNUCINDEX] *= read_data[dv];
      region_param_start.krate_default[ANUCINDEX] *= read_data[dv];
      region_param_start.krate_default[CNUCINDEX] *= read_data[dv];
      region_param_start.krate_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 )
    {
      region_param_start.krate_default[TNUCINDEX] *= read_data[dv++];
      region_param_start.krate_default[ANUCINDEX] *= read_data[dv++];
      region_param_start.krate_default[CNUCINDEX] *= read_data[dv++];
      region_param_start.krate_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 || pset == 1 || pset == 2 ) 
    {
      region_param_start.d_default[TNUCINDEX] *= read_data[dv];
      region_param_start.d_default[ANUCINDEX] *= read_data[dv];
      region_param_start.d_default[CNUCINDEX] *= read_data[dv];
      region_param_start.d_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 )
    {
      region_param_start.d_default[TNUCINDEX] *= read_data[dv++];
      region_param_start.d_default[ANUCINDEX] *= read_data[dv++];
      region_param_start.d_default[CNUCINDEX] *= read_data[dv++];
      region_param_start.d_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 || pset == 1 || pset == 2 ) 
    {
      region_param_start.sigma_mult_default[TNUCINDEX] *= read_data[dv];
      region_param_start.sigma_mult_default[ANUCINDEX] *= read_data[dv];
      region_param_start.sigma_mult_default[CNUCINDEX] *= read_data[dv];
      region_param_start.sigma_mult_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 )
    {
      region_param_start.sigma_mult_default[TNUCINDEX] *= read_data[dv++];
      region_param_start.sigma_mult_default[ANUCINDEX] *= read_data[dv++];
      region_param_start.sigma_mult_default[CNUCINDEX] *= read_data[dv++];
      region_param_start.sigma_mult_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 || pset == 1 || pset == 2 ) 
    {
      region_param_start.t_mid_nuc_delay_default[TNUCINDEX] *= read_data[dv];
      region_param_start.t_mid_nuc_delay_default[ANUCINDEX] *= read_data[dv];
      region_param_start.t_mid_nuc_delay_default[CNUCINDEX] *= read_data[dv];
      region_param_start.t_mid_nuc_delay_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 )
    {
      region_param_start.t_mid_nuc_delay_default[TNUCINDEX] *= read_data[dv++];
      region_param_start.t_mid_nuc_delay_default[ANUCINDEX] *= read_data[dv++];
      region_param_start.t_mid_nuc_delay_default[CNUCINDEX] *= read_data[dv++];
      region_param_start.t_mid_nuc_delay_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 || pset == 1 || pset == 2 ) 
    {
      region_param_start.sens_default *= read_data[dv++];
      region_param_start.tau_R_m_default *= read_data[dv++];
      region_param_start.tau_R_o_default *= read_data[dv++];
    }

    if ( pset == 0 || pset==1 || pset==3)
    {
      for ( int vn=0;vn < 8;vn++ )
        data_control.emp[vn] *= read_data[dv++];

      data_control.emphasis_ampl_default *= read_data[dv++];
      data_control.emphasis_width_default *= read_data[dv++];
    }

    if ( pset == 0 || pset == 1 || pset == 2 ) 
    {
      fitter_defaults.clonal_call_scale[0] *= read_data[dv++];
      fitter_defaults.clonal_call_scale[1] *= read_data[dv++];
      fitter_defaults.clonal_call_scale[2] *= read_data[dv++];
      fitter_defaults.clonal_call_scale[3] *= read_data[dv++];
      fitter_defaults.clonal_call_scale[4] *= read_data[dv++];
    }

    if (pset >= 4 && pset <= 9)
    {
        // kmax
        region_param_start.kmax_default[TNUCINDEX] *= read_data[dv];
        region_param_start.kmax_default[ANUCINDEX] *= read_data[dv];
        region_param_start.kmax_default[CNUCINDEX] *= read_data[dv];
        region_param_start.kmax_default[GNUCINDEX] *= read_data[dv++];
        // sigma_mult
        region_param_start.sigma_mult_default[TNUCINDEX] *= read_data[dv];
        region_param_start.sigma_mult_default[ANUCINDEX] *= read_data[dv];
        region_param_start.sigma_mult_default[CNUCINDEX] *= read_data[dv];
        region_param_start.sigma_mult_default[GNUCINDEX] *= read_data[dv++];
        // t_mid_nuc_delay
        region_param_start.t_mid_nuc_delay_default[TNUCINDEX] *= read_data[dv];
        region_param_start.t_mid_nuc_delay_default[ANUCINDEX] *= read_data[dv];
        region_param_start.t_mid_nuc_delay_default[CNUCINDEX] *= read_data[dv];
        region_param_start.t_mid_nuc_delay_default[GNUCINDEX] *= read_data[dv++];
        // sens
        region_param_start.sens_default *= read_data[dv++];
    }
    else if (pset == 10)
    {
        // kmax[4]
        region_param_start.kmax_default[TNUCINDEX] *= read_data[dv++];
        region_param_start.kmax_default[ANUCINDEX] *= read_data[dv++];
        region_param_start.kmax_default[CNUCINDEX] *= read_data[dv++];
        region_param_start.kmax_default[GNUCINDEX] *= read_data[dv++];
        // krate[4]
        region_param_start.krate_default[TNUCINDEX] *= read_data[dv++];
        region_param_start.krate_default[ANUCINDEX] *= read_data[dv++];
        region_param_start.krate_default[CNUCINDEX] *= read_data[dv++];
        region_param_start.krate_default[GNUCINDEX] *= read_data[dv++];
        // d_coeff[4]
        region_param_start.d_default[TNUCINDEX] *= read_data[dv++];
        region_param_start.d_default[ANUCINDEX] *= read_data[dv++];
        region_param_start.d_default[CNUCINDEX] *= read_data[dv++];
        region_param_start.d_default[GNUCINDEX] *= read_data[dv++];
        // sigma_mult[4]
        region_param_start.sigma_mult_default[TNUCINDEX] *= read_data[dv++];
        region_param_start.sigma_mult_default[ANUCINDEX] *= read_data[dv++];
        region_param_start.sigma_mult_default[CNUCINDEX] *= read_data[dv++];
        region_param_start.sigma_mult_default[GNUCINDEX] *= read_data[dv++];
        // t_mid_nuc_delay[4]
        region_param_start.t_mid_nuc_delay_default[TNUCINDEX] *= read_data[dv++];
        region_param_start.t_mid_nuc_delay_default[ANUCINDEX] *= read_data[dv++];
        region_param_start.t_mid_nuc_delay_default[CNUCINDEX] *= read_data[dv++];
        region_param_start.t_mid_nuc_delay_default[GNUCINDEX] *= read_data[dv++];
        // sens
        region_param_start.sens_default *= read_data[dv++];

    }

    if (pset >= 5)
    {
        data_control.emphasis_ampl_default *= read_data[dv++];
        data_control.emphasis_width_default *= read_data[dv++];
        switch (pset)
        {
        case 6:
            fitter_defaults.clonal_call_scale[0] *= read_data[dv];
            fitter_defaults.clonal_call_scale[1] *= read_data[dv];
            fitter_defaults.clonal_call_scale[2] *= read_data[dv];
            fitter_defaults.clonal_call_scale[3] *= read_data[dv];
            fitter_defaults.clonal_call_scale[4] *= read_data[dv++];
            break;
        case 7:
            fitter_defaults.clonal_call_scale[0] *= read_data[dv];
            fitter_defaults.clonal_call_scale[1] *= read_data[dv];
            fitter_defaults.clonal_call_scale[2] *= read_data[dv];
            fitter_defaults.clonal_call_scale[3] *= read_data[dv];
            fitter_defaults.clonal_call_scale[4] *= read_data[dv++];
            for ( int vn=0;vn < 7;vn++ )
                data_control.emp[vn] *= read_data[dv];
            data_control.emp[7] *= read_data[dv++];
            break;
        case 8:
            fitter_defaults.clonal_call_scale[0] *= read_data[dv++];
            fitter_defaults.clonal_call_scale[1] *= read_data[dv++];
            fitter_defaults.clonal_call_scale[2] *= read_data[dv++];
            fitter_defaults.clonal_call_scale[3] *= read_data[dv++];
            fitter_defaults.clonal_call_scale[4] *= read_data[dv++];
            break;
        case 9:
            for ( int vn=0;vn < 7;vn++ )
                data_control.emp[vn] *= read_data[dv];
            data_control.emp[7] *= read_data[dv++];
            fitter_defaults.clonal_call_scale[0] *= read_data[dv];
            fitter_defaults.clonal_call_scale[1] *= read_data[dv];
            fitter_defaults.clonal_call_scale[2] *= read_data[dv];
            fitter_defaults.clonal_call_scale[3] *= read_data[dv];
            fitter_defaults.clonal_call_scale[4] *= read_data[dv++];
            region_param_start.tau_R_m_default *= read_data[dv++];
            region_param_start.tau_R_o_default *= read_data[dv++];
            break;
        case 10:
            for ( int vn=0;vn < 8;vn++ )
                data_control.emp[vn] *= read_data[dv++];
            fitter_defaults.clonal_call_scale[0] *= read_data[dv++];
            fitter_defaults.clonal_call_scale[1] *= read_data[dv++];
            fitter_defaults.clonal_call_scale[2] *= read_data[dv++];
            fitter_defaults.clonal_call_scale[3] *= read_data[dv++];
            fitter_defaults.clonal_call_scale[4] *= read_data[dv++];
            region_param_start.tau_R_m_default *= read_data[dv++];
            region_param_start.tau_R_o_default *= read_data[dv++];
            break;
        default:
            break;
        }
    }
 }
  else
  {
    fprintf ( stderr, "emphasis file: %s \tstatus: %d\n",fname,status );
    exit ( 1 );
  }

  delete [] line;

  DumpExcitingParameters("GOPT");
}

///-----------------yet another flow object

FlowMyTears::FlowMyTears()
{
  flow_order_len = 4;
}

void FlowMyTears::SetFlowOrder ( char *_flowOrder )
{
  flowOrder      = _flowOrder;
  flow_order_len = flowOrder.length();
  glob_flow_ndx_map.resize(flow_order_len);

  for (int i=0; i<flow_order_len; ++i)
  {
    switch ( toupper ( flowOrder[i] ) )
    {
      case 'T':
        glob_flow_ndx_map[i]=TNUCINDEX;
        break;
      case 'A':
        glob_flow_ndx_map[i]=ANUCINDEX;
        break;
      case 'C':
        glob_flow_ndx_map[i]=CNUCINDEX;
        break;
      case 'G':
        glob_flow_ndx_map[i]=GNUCINDEX;
        break;
      default:
        glob_flow_ndx_map[i]=DEFAULTNUCINDEX;
        break;
    }
  }
}

void FlowMyTears::GetFlowOrderBlock ( int *my_flow, int i_start, int i_stop )
{
  for ( int i=i_start; i<i_stop; i++ )
    my_flow[i-i_start] = GetNucNdx ( i );
}
