/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "GlobalDefaultsFromBkgControl.h"




void ReadOptimizedDefaultsForBkgModel (GlobalDefaultsForBkgModel &global_defaults,BkgModelControlOpts &bkg_control, char *chipType,char *results_folder)
{
  if (strcmp (bkg_control.gopt, "disable") == 0) 
    return;

  if ( (strcmp (bkg_control.gopt, "default") == 0) || (strcmp (bkg_control.gopt, "opt") == 0) ){
    //load defaults for normal run or optimization
    char filename[64] = "";    
    sprintf (filename, "gopt_%s.param", chipType);
    char *tmp_config_file = NULL;
    tmp_config_file = GetIonConfigFile (filename);
    global_defaults.SetGoptDefaults(tmp_config_file);
    if (tmp_config_file)
      free(tmp_config_file);
  }  
  else
    global_defaults.SetGoptDefaults (bkg_control.gopt); //parameter file provided cmd-line
    
  if (strcmp (bkg_control.gopt, "opt") == 0)
    global_defaults.ReadEmphasisVectorFromFile (results_folder);   //GeneticOptimizer run - load its vector
  
}

void OverrideDefaultsForBkgModel (GlobalDefaultsForBkgModel &global_defaults,BkgModelControlOpts &bkg_control, char *chipType,char *results_folder)
{
  // set global parameter defaults from command line values if necessary
  if (bkg_control.dntp_uM[0]>0)
    global_defaults.SetAllConcentrations(bkg_control.dntp_uM);
  
  global_defaults.signal_process_control.AmplLowerLimit = bkg_control.AmplLowerLimit;

  if (bkg_control.krate[0] > 0)
    global_defaults.SetKrateDefaults (bkg_control.krate);

  if (bkg_control.diff_rate[0] > 0)
    global_defaults.SetDDefaults (bkg_control.diff_rate);

  if (bkg_control.kmax[0] > 0)
    global_defaults.SetKmaxDefaults (bkg_control.kmax);

  if (bkg_control.no_rdr_fit_first_20_flows)
    global_defaults.FixRdrInFirst20Flows (true);

  if (bkg_control.fitting_taue)
    global_defaults.SetFittingTauE(true);
  

  if (bkg_control.damp_kmult>0)
    global_defaults.SetDampenKmult (bkg_control.damp_kmult);
  if (bkg_control.generic_test_flag>0)
    global_defaults.SetGenericTestFlag (true);
  if (bkg_control.fit_alternate>0)
    global_defaults.SetFitAlternate(true);
  if (bkg_control.emphasize_by_compression<1)
    global_defaults.SetEmphasizeByCompression(false);
  if (bkg_control.var_kmult_only>0)
    global_defaults.SetVarKmultControl (true);

}


// obviously some structures need to be fused here
void PresetDefaultsForBkgModel(GlobalDefaultsForBkgModel &global_defaults,BkgModelControlOpts &bkg_control)
{
  global_defaults.signal_process_control.choose_time = bkg_control.choose_time;

  global_defaults.signal_process_control.krate_adj_limit = bkg_control.krate_adj_threshold;
  global_defaults.signal_process_control.kmult_low_limit = bkg_control.kmult_low_limit;
  global_defaults.signal_process_control.kmult_hi_limit = bkg_control.kmult_hi_limit;

  global_defaults.signal_process_control.ssq_filter = bkg_control.ssq_filter;

  global_defaults.signal_process_control.do_clonal_filter = bkg_control.enableBkgModelClonalFilter;
  global_defaults.signal_process_control.proton_dot_wells_post_correction=bkg_control.proton_dot_wells_post_correction;
  global_defaults.signal_process_control.single_flow_fit_max_retry=bkg_control.single_flow_fit_max_retry;
  global_defaults.signal_process_control.per_flow_t_mid_nuc_tracking = bkg_control.per_flow_t_mid_nuc_tracking;
  global_defaults.signal_process_control.regional_sampling = bkg_control.regional_sampling;
  global_defaults.signal_process_control.enable_dark_matter = bkg_control.enable_dark_matter;
  global_defaults.signal_process_control.prefilter_beads = bkg_control.prefilter_beads;

  global_defaults.signal_process_control.use_vectorization = (bkg_control.vectorize == 1);
  global_defaults.signal_process_control.AmplLowerLimit = bkg_control.AmplLowerLimit;
  global_defaults.signal_process_control.projection_search_enable = bkg_control.useProjectionSearchForSingleFlowFit;

}

void SetupXtalkParametersForBkgModel (GlobalDefaultsForBkgModel &global_defaults,BkgModelControlOpts &bkg_control, char *chipType,char *results_folder)
{
  // search for config file for chip type
  if (!bkg_control.xtalk)
  {
    //create default param file name
    char filename[64]= "";
    sprintf (filename, "xtalk_%s.param", chipType);
    bkg_control.xtalk = GetIonConfigFile (filename);
    if (!bkg_control.xtalk)
      global_defaults.ReadXtalk ("");   // nothing found
  } // set defaults if nothing set at all
  if (bkg_control.xtalk)
  {
    if (strcmp (bkg_control.xtalk, "local") == 0)
    {
      char my_file[2048] = "";
      sprintf (my_file,"%s/my_xtalk.txt",results_folder);
      global_defaults.ReadXtalk (my_file);   //rerunning in local directory for optimization purposes
    }
    else if (strcmp (bkg_control.xtalk, "disable") != 0)     //disabled = don't load
      global_defaults.ReadXtalk (bkg_control.xtalk);
    else global_defaults.ReadXtalk ("");   // must be non-null to be happy
  } // isolated function

}

//@TODO: should be BkgModelControlOpts
void SetBkgModelGlobalDefaults (GlobalDefaultsForBkgModel &global_defaults, BkgModelControlOpts &bkg_control, char *chipType,char *results_folder)
{
  // @TODO: Bad coding style to use static variables as shared global state
  global_defaults.SetChipType (chipType);   // bad, but need to know this before the first Image arrives
  // better to have global object pointed to by individual entities to make this maneuver very clear
  
  
  ReadOptimizedDefaultsForBkgModel (global_defaults, bkg_control,chipType,results_folder);
  
  PresetDefaultsForBkgModel(global_defaults, bkg_control); // set from command line, possibly overridden by optimization file
  
  OverrideDefaultsForBkgModel (global_defaults, bkg_control,chipType,results_folder);   // after we read from the file so we can tweak

  SetupXtalkParametersForBkgModel (global_defaults, bkg_control,chipType,results_folder);

}
