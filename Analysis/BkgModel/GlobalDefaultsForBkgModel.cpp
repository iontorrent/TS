/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <sys/stat.h>
#include <fstream>
#include "GlobalDefaultsForBkgModel.h"
#include "Utils.h"
#include "IonErr.h"
#include "json/json.h"
#include <boost/algorithm/string/predicate.hpp>
#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "BkgMagicDefines.h"
#include "ChipIdDecoder.h"


LocalSigProcControl::LocalSigProcControl()
{
  AmplLowerLimit = 0.001f;

// a set of relevant parameters for allowing the krate to vary

  ssq_filter = 0.0f;
  choose_time = 0; // normal time compression


  var_kmult_only = false;
  projection_search_enable = false;
  fit_alternate = false;
  fit_gauss_newton = true;

  do_clonal_filter = true;
  enable_dark_matter = true;
  use_vectorization = true;
  enable_well_xtalk_correction = false;
  single_flow_fit_max_retry = 0;
  per_flow_t_mid_nuc_tracking = false;
  exp_tail_fit = false;
  pca_dark_matter = false;
  regional_sampling = false;
  regional_sampling_type = -1;
  no_RatioDrift_fit_first_20_flows = false;
  use_alternative_etbR_equation =false;
  use_log_taub = false;

  fitting_taue = false;
  hydrogenModelType = 0;
  prefilter_beads = false;
  amp_guess_on_gpu = false;
  recompress_tail_raw_trace = false;
  max_frames = 0;
}

void LocalSigProcControl::PrintHelp()
{
	printf ("     LocalSigProcControl\n");
    printf ("     --bkg-kmult-adj-low-hi  FLOAT             setup krate adjust limit [2.0]\n");
    printf ("     --kmult-low-limit       FLOAT             setup kmult low limit [0.65]\n");
    printf ("     --kmult-hi-limit        FLOAT             setup kmult high limit [1.75]\n");
    printf ("     --bkg-ssq-filter-region FLOAT             setup ssq filter region [0.0]\n");
    printf ("     --bkg-ampl-lower-limit  FLOAT             setup ampl lower limit [-0.5 for Proton; 0.001 for PGM]\n");

    printf ("     --bkg-exp-tail-fit      BOOL              enable exp tail fitting [true for Proton; false for PGM]\n");
    printf ("     --time-half-speed       BOOL              reduce choose time by half [false]\n");
    printf ("     --bkg-pca-dark-matter   BOOL              enable pca dark matter [true for Proton; false for PGM]\n");
    printf ("     --regional-sampling     BOOL              enable regional sampling [true for Proton; false for PGM]\n");
    printf ("     --bkg-prefilter-beads   BOOL              use prefilter beads [false]\n");
    printf ("     --vectorize             BOOL              use vectorization [true]\n");
    printf ("     --limit-rdr-fit         BOOL              use no ratio drift fit first 20 flows [false]\n");
    printf ("     --fitting-taue          BOOL              enable fitting taue [false]\n");
    printf ("     --bkg-single-alternate  BOOL              use fit alternate [false]\n");
    printf ("     --var-kmult-only        BOOL              use var kmult only [false]\n");
    printf ("     --incorporation-type    INT               setup hydrogen model type [0]\n");
    printf ("     --clonal-filter-bkgmodel            BOOL  use clonal filter [false for Proton; true for PGM]\n");
    printf ("     --bkg-use-proton-well-correction    BOOL  enable well xtalk correction [true for Proton; false for PGM]\n");
    printf ("     --bkg-per-flow-time-tracking        BOOL  enable per flow time tracking [true for Proton; false for PGM]\n");
    printf ("     --dark-matter-correction            BOOL  enable dark matter correction [true]\n");
    printf ("     --single-flow-projection-search     BOOL  enable single flow projection serch [false]\n");
    printf ("     --use-alternative-etbr-equation     BOOL  use alternative etbR equation [false]\n");
    printf ("     --use-log-taub                      BOOL  use log taub [false]\n");
    printf ("     --bkg-single-gauss-newton           BOOL  use fit gauss newton [true]\n");
    printf ("     --bkg-recompress-tail-raw-trace     BOOL  use recompress tail raw trace [false]\n");
    printf ("     --bkg-single-flow-retry-limit       INT   setup single flow fit max retry [0]\n");
    printf ("\n");
}

void LocalSigProcControl::SetOpts(OptArgs &opts, Json::Value& json_params)
{
	// from PresetDefaultsForBkgModel
	single_flow_master.krate_adj_limit = RetrieveParameterFloat(opts, json_params, '-', "bkg-kmult-adj-low-hi", 2.0);
	single_flow_master.kmult_low_limit = RetrieveParameterFloat(opts, json_params, '-', "kmult-low-limit", 0.65);
	single_flow_master.kmult_hi_limit = RetrieveParameterFloat(opts, json_params, '-', "kmult-hi-limit", 1.75);
	ssq_filter = RetrieveParameterFloat(opts, json_params, '-', "bkg-ssq-filter-region", 0.0);

	bool do_clonal_filter_def = true;
	bool enable_well_xtalk_correction_def = false;
	bool per_flow_t_mid_nuc_tracking_def = false;
	bool exp_tail_fit_def = false;
	bool pca_dark_matter_def = false;
	bool regional_sampling_def = false;
	float AmplLowerLimit_def = 0.001f;
	int defaultSffmr = 0;
	if(ChipIdDecoder::IsProtonChip())
	{
		do_clonal_filter_def = false;
		enable_well_xtalk_correction_def = true;
		per_flow_t_mid_nuc_tracking_def = true;
		exp_tail_fit_def = true;
		pca_dark_matter_def = true;
		regional_sampling_def = true;
		AmplLowerLimit_def = -0.5f;
		if(!fit_gauss_newton)
		{
			defaultSffmr = 4;
		}
	}
	do_clonal_filter = RetrieveParameterBool(opts, json_params, '-', "clonal-filter-bkgmodel", do_clonal_filter_def);
	enable_well_xtalk_correction = RetrieveParameterBool(opts, json_params, '-', "bkg-use-proton-well-correction", enable_well_xtalk_correction_def);
	per_flow_t_mid_nuc_tracking = RetrieveParameterBool(opts, json_params, '-', "bkg-per-flow-time-tracking", per_flow_t_mid_nuc_tracking_def);
	exp_tail_fit = RetrieveParameterBool(opts, json_params, '-', "bkg-exp-tail-fit", exp_tail_fit_def);
	if(exp_tail_fit)
	{
		choose_time = 2;
		bool half_speed = RetrieveParameterBool(opts, json_params, '-', "time-half-speed", false);
		if(half_speed)
		{
			choose_time = 1;
		}
	}
	pca_dark_matter = RetrieveParameterBool(opts, json_params, '-', "bkg-pca-dark-matter", pca_dark_matter_def);
	regional_sampling = RetrieveParameterBool(opts, json_params, '-', "regional-sampling", regional_sampling_def);
	regional_sampling_type = RetrieveParameterInt(opts, json_params, '-', "regional_sampling_type", 1);
	enable_dark_matter = RetrieveParameterBool(opts, json_params, '-', "dark-matter-correction", true);
	prefilter_beads = RetrieveParameterBool(opts, json_params, '-', "bkg-prefilter-beads", false);
	use_vectorization = RetrieveParameterBool(opts, json_params, '-', "vectorize", true);
	AmplLowerLimit = RetrieveParameterFloat(opts, json_params, '-', "bkg-ampl-lower-limit", AmplLowerLimit_def);
	projection_search_enable = RetrieveParameterBool(opts, json_params, '-', "single-flow-projection-search", false);

	// from OverrideDefaultsForBkgModel//changed
	no_RatioDrift_fit_first_20_flows = RetrieveParameterBool(opts, json_params, '-', "limit-rdr-fit", false);
	use_alternative_etbR_equation = RetrieveParameterBool(opts, json_params, '-', "use-alternative-etbr-equation", false);
    use_log_taub = RetrieveParameterBool(opts, json_params, '-', "use-log-taub", false);

	fitting_taue = RetrieveParameterBool(opts, json_params, '-', "fitting-taue", false);
	hydrogenModelType = RetrieveParameterInt(opts, json_params, '-', "incorporation-type", 0);
//jz	generic_test_flag = RetrieveParameterBool(opts, json_params, '-', "generic-test-flag", false);
	fit_alternate = RetrieveParameterBool(opts, json_params, '-', "bkg-single-alternate", false);
	fit_gauss_newton = RetrieveParameterBool(opts, json_params, '-', "bkg-single-gauss-newton", true);
	single_flow_fit_max_retry = RetrieveParameterInt(opts, json_params, '-', "bkg-single-flow-retry-limit", defaultSffmr);
	var_kmult_only = RetrieveParameterBool(opts, json_params, '-', "var-kmult-only", false);
	recompress_tail_raw_trace = RetrieveParameterBool(opts, json_params, '-', "bkg-recompress-tail-raw-trace", false);
}

void GlobalDefaultsForBkgModel::SetChipType ( const char *name )
{
  chipType=name;
}
void GlobalDefaultsForBkgModel::ReadXtalk ( char *name )
{
  xtalk_name=name;
}


void GlobalDefaultsForBkgModel::GoptDefaultsFromJson(char *fname){
  Json::Value all_params;
  std::ifstream in(fname, std::ios::in);

  if (!in.good()) {
    printf("Opening gopt parameter file %s unsuccessful. Aborting\n", fname);
    exit(1);
  }
  in >> all_params;

  if (all_params.isMember("parameters")){
    // strip down to the correct subset of the file
    Json::Value gopt_params = all_params["parameters"];

    region_param_start.FromJson(gopt_params);
    data_control.FromJson(gopt_params);
    fitter_defaults.FromJson(gopt_params);
  }else{
    std::cout << "ABORT: gopt file contains no parameters " << fname << "\n";
    exit(1);
  }
  // echo as test
   //std::cout << gopt_params.toStyledString();
  in.close();
}


#define MAX_LINE_LEN    2048
#define MAX_DATA_PTS    80

void GlobalDefaultsForBkgModel::GoptDefaultsFromPoorlyStructuredFile(char *fname){
  struct stat fstatus;
  int         status;
  FILE *param_file;
  char *line;
  int nChar = MAX_LINE_LEN;
  float d[10];

  int num = 0;

  line = new char[MAX_LINE_LEN];

  status = stat ( fname,&fstatus );
  printf("You are trying to use an obsolete format:  please use json files instead!!!!\n");
  printf("Continue loading the old format file for now...");
  //exit(1);  // If I don't force failure, will never be used

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

        region_param_start.FromCharacterLine(line);
        data_control.FromCharacterLine(line);
        fitter_defaults.FromCharacterLine(line);
      }
      else
        done = true;
    }

    fclose ( param_file );
  }
  else{
    printf ( "GOPT: parameter file %s does not exist, Aborting\n",fname );
    exit(1);
  }

  delete [] line;
}

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
    GoptDefaultsFromJson(fname);
  } else{
      //old way
    GoptDefaultsFromPoorlyStructuredFile(fname);

  }
  region_param_start.BadIdeaComputeDerivedInput();
  DumpExcitingParameters("default");
}

void GlobalDefaultsForBkgModel::DumpExcitingParameters(const char *fun_string)
{
      //output defaults used
    printf ( "%s parameters used: \n",fun_string );

    region_param_start.DumpPoorlyStructuredText();
    data_control.DumpPoorlyStructuredText();
    fitter_defaults.DumpPoorlyStructuredText();

    printf ( "\n" );
}

// This function is used during GeneticOptimizer runs in which case the above SetGoptDefaults is disabled
//@TODO: 300 line redundant function - why do we do things in badly structured ways?
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
      else if ( evect_size == 44 ) pset = 11;
      else if ( evect_size == 46 ) pset = 12;   // min_tauB, max_tauB
      else if ( evect_size == 47 ) pset = 13;   // mid_tauB
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
    if ( pset == 0 || pset == 1 || pset == 2 || pset>=11)
    {
    // first value scales add km terms
    region_param_start.kmax_default[TNUCINDEX] *= read_data[dv];
    region_param_start.kmax_default[ANUCINDEX] *= read_data[dv];
    region_param_start.kmax_default[CNUCINDEX] *= read_data[dv];
    region_param_start.kmax_default[GNUCINDEX] *= read_data[dv++];
    }
    // 2-5 values scale individual terms
    if ( pset == 0 || pset>=11)
    {
      region_param_start.kmax_default[TNUCINDEX] *= read_data[dv++];
      region_param_start.kmax_default[ANUCINDEX] *= read_data[dv++];
      region_param_start.kmax_default[CNUCINDEX] *= read_data[dv++];
      region_param_start.kmax_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 || pset == 1 || pset == 2 || pset>=11)
    {
      region_param_start.krate_default[TNUCINDEX] *= read_data[dv];
      region_param_start.krate_default[ANUCINDEX] *= read_data[dv];
      region_param_start.krate_default[CNUCINDEX] *= read_data[dv];
      region_param_start.krate_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 || pset>=11)
    {
      region_param_start.krate_default[TNUCINDEX] *= read_data[dv++];
      region_param_start.krate_default[ANUCINDEX] *= read_data[dv++];
      region_param_start.krate_default[CNUCINDEX] *= read_data[dv++];
      region_param_start.krate_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 || pset == 1 || pset == 2 || pset>=11)
    {
      region_param_start.d_default[TNUCINDEX] *= read_data[dv];
      region_param_start.d_default[ANUCINDEX] *= read_data[dv];
      region_param_start.d_default[CNUCINDEX] *= read_data[dv];
      region_param_start.d_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 || pset>=11)
    {
      region_param_start.d_default[TNUCINDEX] *= read_data[dv++];
      region_param_start.d_default[ANUCINDEX] *= read_data[dv++];
      region_param_start.d_default[CNUCINDEX] *= read_data[dv++];
      region_param_start.d_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 || pset == 1 || pset == 2 || pset>=11)
    {
      region_param_start.sigma_mult_default[TNUCINDEX] *= read_data[dv];
      region_param_start.sigma_mult_default[ANUCINDEX] *= read_data[dv];
      region_param_start.sigma_mult_default[CNUCINDEX] *= read_data[dv];
      region_param_start.sigma_mult_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 || pset>=11)
    {
      region_param_start.sigma_mult_default[TNUCINDEX] *= read_data[dv++];
      region_param_start.sigma_mult_default[ANUCINDEX] *= read_data[dv++];
      region_param_start.sigma_mult_default[CNUCINDEX] *= read_data[dv++];
      region_param_start.sigma_mult_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 || pset == 1 || pset == 2 || pset>=11)
    {
      region_param_start.t_mid_nuc_delay_default[TNUCINDEX] *= read_data[dv];
      region_param_start.t_mid_nuc_delay_default[ANUCINDEX] *= read_data[dv];
      region_param_start.t_mid_nuc_delay_default[CNUCINDEX] *= read_data[dv];
      region_param_start.t_mid_nuc_delay_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 || pset>=11)
    {
      region_param_start.t_mid_nuc_delay_default[TNUCINDEX] *= read_data[dv++];
      region_param_start.t_mid_nuc_delay_default[ANUCINDEX] *= read_data[dv++];
      region_param_start.t_mid_nuc_delay_default[CNUCINDEX] *= read_data[dv++];
      region_param_start.t_mid_nuc_delay_default[GNUCINDEX] *= read_data[dv++];
    }

    if ( pset == 0 || pset == 1 || pset == 2 || pset>=11)
    {
      region_param_start.sens_default *= read_data[dv++];
      region_param_start.tau_R_m_default *= read_data[dv++];
      region_param_start.tau_R_o_default *= read_data[dv++];
    }

    if ( pset == 0 || pset==1 || pset==3 || pset>=11)
    {
      for ( int vn=0;vn < 8;vn++ )
        data_control.emp[vn] *= read_data[dv++];

      data_control.emphasis_ampl_default *= read_data[dv++];
      data_control.emphasis_width_default *= read_data[dv++];
    }

    if ( pset == 0 || pset == 1 || pset == 2 || pset>=11)
    {
      fitter_defaults.clonal_call_scale[0] *= read_data[dv++];
      fitter_defaults.clonal_call_scale[1] *= read_data[dv++];
      fitter_defaults.clonal_call_scale[2] *= read_data[dv++];
      fitter_defaults.clonal_call_scale[3] *= read_data[dv++];
      fitter_defaults.clonal_call_scale[4] *= read_data[dv++];
    }
    if ( pset>=11)
    {
      // taue
      region_param_start.tau_E_default *= read_data[dv++];
    }
    if ( pset>=12)
    {
      region_param_start.min_tauB_default *= read_data[dv++];
      region_param_start.max_tauB_default *= read_data[dv++];
    }
    if ( pset>=13)
    {
      region_param_start.mid_tauB_default *= read_data[dv++];
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

    if (pset >= 5 && pset <=10)
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
  region_param_start.BadIdeaComputeDerivedInput();

  DumpExcitingParameters("GOPT");
}

void GlobalDefaultsForBkgModel::PrintHelp()
{
	printf ("     GlobalDefaultsForBkgModel\n");
	printf ("     --bkg-well-xtalk-name   FILE              well xtalk file name []\n");
    printf ("     --gopt                  STRING            setup gopte [default]\n");
    printf ("     --xtalk                 STRING            setup xtalk [disable]\n");
    printf ("     --bkg-dont-emphasize-by-compression BOOL  not empasized by compression [false]\n");
    printf ("\n");

	signal_process_control.PrintHelp();
}

void GlobalDefaultsForBkgModel::SetOpts(OptArgs &opts, Json::Value& json_params)
{
	// from SetBkgModelGlobalDefaults
	chipType = GetParamsString(json_params, "chipType", "");
	string results_folder = GetParamsString(json_params, "results_folder", "");

	string gopt = RetrieveParameterString(opts, json_params, '-', "gopt", "default");

	if(gopt != "disable")
	{
		if(gopt == "default")
		{
			string filename = "gopt_";
			filename += chipType;
			filename += ".param.json";

			char *tmp_config_file = NULL;
			tmp_config_file = GetIonConfigFile (filename.c_str());
			SetGoptDefaults(tmp_config_file);
			if (tmp_config_file)
			  free(tmp_config_file);
		}
		else if(gopt == "opt")
		{
			string filename = "gopt_";
			filename += chipType;
			filename += ".param.json";

			char *tmp_config_file = GetIonConfigFile (filename.c_str());

			if (tmp_config_file == NULL){ // if gopt default file does not exist, try KnownAlternate_chiptype
				string chipType2 = get_KnownAlternate_chiptype(chipType);
				filename = "gopt_";
				filename += chipType2;
				filename += ".param.json";
				tmp_config_file = GetIonConfigFile (filename.c_str());
			}

			SetGoptDefaults(tmp_config_file);
			if (tmp_config_file)
			  free(tmp_config_file);

			ReadEmphasisVectorFromFile ((char*)(results_folder.c_str()));   //GeneticOptimizer run - load its vector
		}
		else
		{
		    SetGoptDefaults ((char*)(gopt.c_str())); //parameter file provided cmd-line
			// still do opt if the emphasis_vector.txt file exists
			char fname[512];
			sprintf ( fname,"%s/emphasis_vector.txt", results_folder.c_str() );
			struct stat fstatus;
			int status = stat ( fname,&fstatus );
			if ( status == 0 )    // file exists
				ReadEmphasisVectorFromFile ((char*)(results_folder.c_str()));   //GeneticOptimizer run - load its vector
		}
	}

	signal_process_control.SetOpts(opts, json_params);
	bool dont_emphasize_by_compression = RetrieveParameterBool(opts, json_params, '-', "bkg-dont-emphasize-by-compression", false);
	data_control.point_emphasis_by_compression = (!dont_emphasize_by_compression);

	// from SetupXtalkParametersForBkgModel
	string trace_xtalk_name = RetrieveParameterString(opts, json_params, '-', "xtalk", "disable");
	
	// search for config file for chip type
	if(trace_xtalk_name.length() == 0)
	{
		// create default param file name
		string filename = "xtalk_";
		filename += chipType;
		filename += ".param";
		char *filename2 = NULL;
		filename2 = GetIonConfigFile(filename.c_str());
		if(!filename2)
		{
			xtalk_name = ""; // nothing found
		}
		else
		{
			trace_xtalk_name = filename2;
		}
	}

	// set defaults if nothing set at all
	if(trace_xtalk_name.length() > 0)
	{
		if(trace_xtalk_name == "local")
		{
			xtalk_name = results_folder;
			xtalk_name += "/my_xtalk.txt";// rerunning in local directory for optimization purposes
		}
		else if(trace_xtalk_name != "disable") // disabled = don't load
		{
			xtalk_name = trace_xtalk_name;
		}
		else
		{
			xtalk_name = "";
		}
	}

	// now handle well based xtalk parameters
	// currently just set the default
	// but need to have file, activation, etc.
	string well_xtalk_name = RetrieveParameterString(opts, json_params, '-', "bkg-well-xtalk-name", "");
	if(well_xtalk_name.length() > 0)
	{
		// assume the explicit file rather than searching all over
		well_xtalk_master.ReadFromFile(well_xtalk_name);
		signal_process_control.enable_well_xtalk_correction = true; // if you have speced a file, assume you want to do this
	}
	else
	{
		well_xtalk_master.DefaultPI();
	}
}
