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

  copy_stringency = 2.0f;
  min_high_quality_beads = 10;

  choose_time = 0; // normal time compression


  fit_gauss_newton = true;
  fit_region_kmult = false;

  do_clonal_filter = true;
  enable_dark_matter = true;
  use_vectorization = true;
  enable_well_xtalk_correction = false;

  per_flow_t_mid_nuc_tracking = false;
  exp_tail_fit = false;
  pca_dark_matter = false;
  regional_sampling = false;
  regional_sampling_type = -1;
  no_RatioDrift_fit_first_20_flows = false;
  use_alternative_etbR_equation =false;

  suppress_copydrift = false;
  fitting_taue = false;
  safe_model= false;
  hydrogenModelType = 0;
  prefilter_beads = false;
  amp_guess_on_gpu = false;
  recompress_tail_raw_trace = false;
  max_frames = 0;
  barcode_flag = false;
  double_tap_means_zero = true;
  num_regional_samples = 200;
}

void LocalSigProcControl::PrintHelp()
{
	printf ("     LocalSigProcControl\n");
    printf ("     --bkg-kmult-adj-low-hi  FLOAT             setup krate adjust limit [2.0]\n");
    printf ("     --kmult-low-limit       FLOAT             setup kmult low limit [0.65]\n");
    printf ("     --kmult-hi-limit        FLOAT             setup kmult high limit [1.75]\n");
    printf ("     --bkg-ampl-lower-limit  FLOAT             setup ampl lower limit [-0.5 for Proton; 0.001 for PGM]\n");

    printf ("     --bkg-exp-tail-fit      BOOL              enable exp tail fitting [true for Proton; false for PGM]\n");
    printf ("     --time-half-speed       BOOL              reduce choose time by half [false]\n");
    printf ("     --bkg-pca-dark-matter   BOOL              enable pca dark matter [true for Proton; false for PGM]\n");
    printf ("     --regional-sampling     BOOL              enable regional sampling [true for Proton; false for PGM]\n");
    printf ("     --num_regional-samples  INT               number of regional samples used for multi flow fitting\n");
    printf ("     --skip-first-flow-block-regional-fitting   BOOL  skip multi flow regional fitting in first flow block if a regional parameters json file is provided [false]\n");
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


  copy_stringency = RetrieveParameterFloat(opts, json_params, '-', "bkg-copy-stringency", 1.0);
  min_high_quality_beads = RetrieveParameterInt(opts,json_params, '-',"bkg-min-sampled-beads", 100); // half-regional-sampling
  max_rank_beads = RetrieveParameterInt(opts,json_params, '-',"bkg-max-rank-beads", 100000); // effective infinity: cut off no beads for being too bright
  post_key_train = RetrieveParameterInt(opts,json_params, '-',"bkg-post-key-train", 2);
  post_key_step = RetrieveParameterInt(opts,json_params, '-',"bkg-post-key-step", 2);

  //ugly parameters because exp-tail fit jams multiple routines into one 'module'
  exp_tail_bkg_adj = true; // do background adjust per flow using the shifted background
  exp_tail_tau_adj = true; // add a taub modifier based on the fit in the first 20 flows
  exp_tail_bkg_limit = 0.2f; // 20% of background adjustment should be enough - typical values are <<5%
  exp_tail_bkg_lower = 10.0f;  // guess what happens when the pH step becomes 0?  we divide by zero and blow up
  // end ugly

  do_clonal_filter = RetrieveParameterBool(opts, json_params, '-', "clonal-filter-bkgmodel", true);
  barcode_flag = RetrieveParameterBool(opts, json_params, '-', "barcode-flag", false);
  barcode_debug = RetrieveParameterBool(opts, json_params, '-', "barcode-debug", false);
  barcode_radius = RetrieveParameterFloat(opts, json_params,'-',"barcode-radius", 0.75f);
  barcode_tie    = RetrieveParameterFloat(opts, json_params,'-',"barcode-tie", 0.5f);
  barcode_penalty = RetrieveParameterFloat(opts, json_params,'-',"barcode-penalty", 2000.0f);
  kmult_penalty = RetrieveParameterFloat(opts, json_params,'-',"kmult-penalty", 100.0f);

  enable_well_xtalk_correction = RetrieveParameterBool(opts, json_params, '-', "bkg-use-proton-well-correction", false);
  per_flow_t_mid_nuc_tracking = RetrieveParameterBool(opts, json_params, '-', "bkg-per-flow-time-tracking", false);

  //ugly family of parameters due to history
  exp_tail_fit = RetrieveParameterBool(opts, json_params, '-', "bkg-exp-tail-fit", false);
  exp_tail_tau_adj = RetrieveParameterBool(opts, json_params, '-', "bkg-exp-tail-tau-adj", exp_tail_tau_adj);
  exp_tail_bkg_adj = RetrieveParameterBool(opts, json_params, '-', "bkg-exp-tail-bkg-adj", exp_tail_bkg_adj);
  exp_tail_bkg_limit = RetrieveParameterFloat(opts, json_params, '-', "bkg-exp-tail-limit", exp_tail_bkg_limit);
  exp_tail_bkg_lower = RetrieveParameterFloat(opts, json_params, '-', "bkg-exp-tail-lower", exp_tail_bkg_lower);
  if(exp_tail_fit)
	{
		choose_time = 2;
	}
  // too many things packed into 'exp-tail-fit'

    pca_dark_matter = RetrieveParameterBool(opts, json_params, '-', "bkg-pca-dark-matter", false);
    regional_sampling = RetrieveParameterBool(opts, json_params, '-', "regional-sampling", false);
    regional_sampling_type = RetrieveParameterInt(opts, json_params, '-', "regional-sampling-type", 1);
    revert_regional_sampling = RetrieveParameterBool(opts, json_params, '-',"revert-regional-sampling", false);

	enable_dark_matter = RetrieveParameterBool(opts, json_params, '-', "dark-matter-correction", true);
	prefilter_beads = RetrieveParameterBool(opts, json_params, '-', "bkg-prefilter-beads", false);
	use_vectorization = RetrieveParameterBool(opts, json_params, '-', "vectorize", true);
    AmplLowerLimit = RetrieveParameterFloat(opts, json_params, '-', "bkg-ampl-lower-limit", 0.001);

	// from OverrideDefaultsForBkgModel//changed
	no_RatioDrift_fit_first_20_flows = RetrieveParameterBool(opts, json_params, '-', "limit-rdr-fit", false);
	use_alternative_etbR_equation = RetrieveParameterBool(opts, json_params, '-', "use-alternative-etbr-equation", false);

	fitting_taue = RetrieveParameterBool(opts, json_params, '-', "fitting-taue", false);


  safe_model = RetrieveParameterBool(opts, json_params, '-', "use-safe-buffer-model", false);
  suppress_copydrift = RetrieveParameterBool(opts, json_params, '-', "suppress-copydrift", false);
  hydrogenModelType = RetrieveParameterInt(opts, json_params, '-', "incorporation-type", 0);

   stop_beads = RetrieveParameterBool(opts, json_params, '-', "stop-beads", false);

	fit_gauss_newton = RetrieveParameterBool(opts, json_params, '-', "bkg-single-gauss-newton", true);

  fit_region_kmult = RetrieveParameterBool(opts, json_params, '-', "fit-region-kmult", false);
  always_start_slow = RetrieveParameterBool(opts, json_params, '-', "always-start-slow", true);


	recompress_tail_raw_trace = RetrieveParameterBool(opts, json_params, '-', "bkg-recompress-tail-raw-trace", false);
    double_tap_means_zero = RetrieveParameterBool(opts, json_params, '-', "double-tap-means-zero", true);
    num_regional_samples = RetrieveParameterInt(opts, json_params, '-', "num-regional-samples", 400);
    skipFirstFlowBlockRegFitting = RetrieveParameterBool(opts, json_params, '-', "skip-first-flow-block-regional-fitting", false);
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

    printf("Abort: %s not a json file", fname);
    exit(1);

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


void GlobalDefaultsForBkgModel::PrintHelp()
{
    printf ("     GlobalDefaultsForBkgModel\n");
    printf ("     --bkg-well-xtalk-name   FILE              specify well xtalk parameter filename []\n");
    printf ("     --xtalk                 STRING            specify trace xtalk parameter filename [disable]\n");
    printf ("     --gopt                  STRING            specify gopt parameter filename [default]\n");
    printf ("     --bkg-dont-emphasize-by-compression BOOL  not empasized by compression [false]\n");
    printf ("\n");

    signal_process_control.PrintHelp();
}

void GlobalDefaultsForBkgModel::SetOpts(OptArgs &opts, Json::Value& json_params)
{
	// from SetBkgModelGlobalDefaults
	chipType = GetParamsString(json_params, "chipType", "");
        if((!chipType.empty()) && chipType[0] == 'P') chipType[0] = 'p';
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

		}
		else
		{
		    SetGoptDefaults ((char*)(gopt.c_str())); //parameter file provided cmd-line

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

  // read master file containing barcodes
  // set up barcodes in each region as copied from master

  string barcode_file_name = RetrieveParameterString(opts, json_params,'-', "barcode-spec-file","");
  if (barcode_file_name.length()>0)
  {
    barcode_master.ReadFromFile(barcode_file_name);

  }
  else
  {
    // nothing: no barcodes, don't do anything
  }
}
