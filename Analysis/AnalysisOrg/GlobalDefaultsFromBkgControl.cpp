/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "GlobalDefaultsFromBkgControl.h"
#include <string>



static void ReadOptimizedDefaultsForBkgModel (GlobalDefaultsForBkgModel &global_defaults,BkgModelControlOpts &bkg_control, const char *chipType,char *results_folder)
{
  if (strcmp (bkg_control.gopt, "disable") == 0) 
    return;

  if ( (strcmp (bkg_control.gopt, "default") == 0)){
    //load defaults for normal run
    char filename[64] = "";
    sprintf (filename, bkg_control.gopt_default_pattern.c_str(), chipType);
    char *tmp_config_file = NULL;
    tmp_config_file = GetIonConfigFile (filename);

    if (tmp_config_file==NULL){
      printf("ABORT: no standard configuration file: %s found in any path\n",filename);
      printf("Specify explicitly by --gopt %s\n", filename);
      printf("Or check installation for proper chip type or configuration errors\n");
      exit(1);
    }
      global_defaults.SetGoptDefaults(tmp_config_file);
    if (tmp_config_file)
      free(tmp_config_file);
  } else if (strcmp (bkg_control.gopt, "opt") == 0){
    //load defaults for for optimization
    char filename[64] = "";
    sprintf (filename, bkg_control.gopt_default_pattern.c_str(), chipType);
    char *tmp_config_file = GetIonConfigFile (filename);

    if (tmp_config_file==NULL){ // if gopt default file does not exist, try KnownAlternate_chiptype
        std::string chiptype = get_KnownAlternate_chiptype(chipType);
        sprintf (filename, bkg_control.gopt_default_pattern.c_str(), chiptype.c_str());
        tmp_config_file = GetIonConfigFile (filename);
    }

    if (tmp_config_file==NULL){
      printf("ABORT: no standard configuration file: %s found in any path\n",filename);
      printf("Specify explicitly by --gopt %s\n", filename);
      printf("Or check installation for proper chip type\n");
      exit(1);
    }

    global_defaults.SetGoptDefaults(tmp_config_file);
    if (tmp_config_file)
      free(tmp_config_file);
    // over-write with some partial parameter sets
    if (strcmp (bkg_control.gopt, "opt") == 0)
      global_defaults.ReadEmphasisVectorFromFile (results_folder);   //GeneticOptimizer run - load its vector
  }else
  {
    global_defaults.SetGoptDefaults (bkg_control.gopt); //parameter file provided cmd-line
    // still do opt if the emphasis_vector.txt file exists
    char fname[512];
    sprintf ( fname,"%s/emphasis_vector.txt", results_folder );
    struct stat fstatus;
    int status = stat ( fname,&fstatus );
    if ( status == 0 )    // file exists
        global_defaults.ReadEmphasisVectorFromFile (results_folder);   //GeneticOptimizer run - load its vector
  }   
}


static void OverrideDefaultsForBkgModel (GlobalDefaultsForBkgModel &global_defaults,BkgModelControlOpts &bkg_control, const char *chipType,char *results_folder)
{

  global_defaults.signal_process_control.AmplLowerLimit = bkg_control.AmplLowerLimit;


  if (bkg_control.no_rdr_fit_first_20_flows)
    global_defaults.FixRdrInFirst20Flows (true);

  if (bkg_control.use_alternative_etbR_equation)
    global_defaults.SetUse_alternative_etbR_equation(true); 

  if (bkg_control.psp4_dev!=0)
    global_defaults.Setpsp4_dev(bkg_control.psp4_dev); 

  if (bkg_control.fitting_taue)
    global_defaults.SetFittingTauE(true);
  
  global_defaults.SetHydrogenModel( bkg_control.hydrogenModelType );



  if (bkg_control.single_control.fit_alternate>0)
    global_defaults.SetFitAlternate(true);
  if (bkg_control.single_control.fit_gauss_newton)
    global_defaults.SetFitGaussNewton(true);
  if (bkg_control.single_control.var_kmult_only>0)
    global_defaults.SetVarKmultControl (true);

  if (bkg_control.emphasize_by_compression<1)
    global_defaults.SetEmphasizeByCompression(false);

}


// obviously some structures need to be fused here
void PresetDefaultsForBkgModel(GlobalDefaultsForBkgModel &global_defaults,BkgModelControlOpts &bkg_control)
{
  global_defaults.signal_process_control.choose_time = bkg_control.choose_time;

  global_defaults.signal_process_control.single_flow_master.krate_adj_limit = bkg_control.single_control.krate_adj_threshold;
  global_defaults.signal_process_control.single_flow_master.kmult_low_limit = bkg_control.single_control.kmult_low_limit;
  global_defaults.signal_process_control.single_flow_master.kmult_hi_limit = bkg_control.single_control.kmult_hi_limit;
  global_defaults.signal_process_control.single_flow_fit_max_retry=bkg_control.single_control.single_flow_fit_max_retry;
  global_defaults.signal_process_control.projection_search_enable = bkg_control.single_control.useProjectionSearchForSingleFlowFit;

  global_defaults.signal_process_control.ssq_filter = bkg_control.ssq_filter;

  global_defaults.signal_process_control.do_clonal_filter = bkg_control.polyclonal_filter.enable;
  global_defaults.signal_process_control.enable_well_xtalk_correction=bkg_control.enable_well_xtalk_correction;
  global_defaults.signal_process_control.per_flow_t_mid_nuc_tracking = bkg_control.per_flow_t_mid_nuc_tracking;
  global_defaults.signal_process_control.exp_tail_fit = bkg_control.exp_tail_fit;
  global_defaults.signal_process_control.pca_dark_matter = bkg_control.pca_dark_matter;
  global_defaults.signal_process_control.regional_sampling = bkg_control.regional_sampling;
  global_defaults.signal_process_control.regional_sampling_type = bkg_control.regional_sampling_type;
  global_defaults.signal_process_control.enable_dark_matter = bkg_control.enable_dark_matter;
  global_defaults.signal_process_control.prefilter_beads = bkg_control.prefilter_beads;
  global_defaults.signal_process_control.recompress_tail_raw_trace = bkg_control.recompress_tail_raw_trace;

  global_defaults.signal_process_control.use_vectorization = (bkg_control.vectorize == 1);
  global_defaults.signal_process_control.AmplLowerLimit = bkg_control.AmplLowerLimit;

}

static void SetupXtalkParametersForBkgModel (GlobalDefaultsForBkgModel &global_defaults,BkgModelControlOpts &bkg_control, const char *chipType,char *results_folder)
{
  // search for config file for chip type
  if (!bkg_control.trace_xtalk_name)
  {
    //create default param file name
    char filename[64]= "";
    sprintf (filename, "xtalk_%s.param", chipType);
    bkg_control.trace_xtalk_name = GetIonConfigFile (filename);
    if (!bkg_control.trace_xtalk_name)
      global_defaults.ReadXtalk ("");   // nothing found
  } // set defaults if nothing set at all
  if (bkg_control.trace_xtalk_name)
  {
    if (strcmp (bkg_control.trace_xtalk_name, "local") == 0)
    {
      char my_file[2048] = "";
      sprintf (my_file,"%s/my_xtalk.txt",results_folder);
      global_defaults.ReadXtalk (my_file);   //rerunning in local directory for optimization purposes
    }
    else if (strcmp (bkg_control.trace_xtalk_name, "disable") != 0)     //disabled = don't load
      global_defaults.ReadXtalk (bkg_control.trace_xtalk_name);
    else global_defaults.ReadXtalk ("");   // must be non-null to be happy
  } // isolated function

  // now handle well based xtalk parameters
  // currently just set the default
  // but need to have file, activation, etc.
  if (bkg_control.well_xtalk_name.size()>0){
    // assume the explicit file rather than searching all over
    global_defaults.well_xtalk_master.ReadFromFile(bkg_control.well_xtalk_name);
    global_defaults.signal_process_control.enable_well_xtalk_correction = true; // if you have speced a file, assume you want to do this
  }else
      global_defaults.well_xtalk_master.DefaultPI();

}

//@TODO: should be BkgModelControlOpts
void SetBkgModelGlobalDefaults (GlobalDefaultsForBkgModel &global_defaults, BkgModelControlOpts &bkg_control, const char *chipType,char *results_folder)
{
  // @TODO: Bad coding style to use static variables as shared global state
  global_defaults.SetChipType (chipType);   // bad, but need to know this before the first Image arrives
  // better to have global object pointed to by individual entities to make this maneuver very clear
  
  
  ReadOptimizedDefaultsForBkgModel (global_defaults, bkg_control,chipType,results_folder);
  
  PresetDefaultsForBkgModel(global_defaults, bkg_control); // set from command line, possibly overridden by optimization file
  
  OverrideDefaultsForBkgModel (global_defaults, bkg_control,chipType,results_folder);   // after we read from the file so we can tweak

  SetupXtalkParametersForBkgModel (global_defaults, bkg_control,chipType,results_folder);

}
