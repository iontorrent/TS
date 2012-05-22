/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "GlobalDefaultsFromBkgControl.h"




void ReadOptimizedDefaultsForBkgModel (BkgModelControlOpts &bkg_control, char *chipType,char *experimentName)
{
  //optionally set optimized parameters from GeneticOptimizer runs
  if (!bkg_control.gopt)
  {
    //create default param file name
    char filename[64];
    sprintf (filename, "gopt_%s.param", chipType);
    bkg_control.gopt = GetIonConfigFile (filename);
  } // set defaults if nothing set at all
  if (bkg_control.gopt)
  {
    if (strcmp (bkg_control.gopt, "opt") == 0)
      GlobalDefaultsForBkgModel::ReadEmphasisVectorFromFile (experimentName);   //GeneticOptimizer run - load its vector
    else if (strcmp (bkg_control.gopt, "disable") != 0)     //load gopt defaults unless disabled
      GlobalDefaultsForBkgModel::SetGoptDefaults (bkg_control.gopt);
  }
}

void OverrideDefaultsForBkgModel (BkgModelControlOpts &bkg_control, char *chipType,char *experimentName)
{
  // set global parameter defaults from command line values if necessary
  if (bkg_control.krate[0] > 0)
    GlobalDefaultsForBkgModel::SetKrateDefaults (bkg_control.krate);

  if (bkg_control.diff_rate[0] > 0)
    GlobalDefaultsForBkgModel::SetDDefaults (bkg_control.diff_rate);

  if (bkg_control.kmax[0] > 0)
    GlobalDefaultsForBkgModel::SetKmaxDefaults (bkg_control.kmax);

  if (bkg_control.no_rdr_fit_first_20_flows)
    GlobalDefaultsForBkgModel::FixRdrInFirst20Flows (true);

  if (bkg_control.damp_kmult>0)
    GlobalDefaultsForBkgModel::SetDampenKmult (bkg_control.damp_kmult);
  if (bkg_control.generic_test_flag>0)
    GlobalDefaultsForBkgModel::SetGenericTestFlag (true);
  if (bkg_control.var_kmult_only>0)
    GlobalDefaultsForBkgModel::SetVarKmultControl (true);
  

}

void PresetDefaultsForBkgModel(BkgModelControlOpts &bkg_control)
{
  GlobalDefaultsForBkgModel::choose_time = bkg_control.choose_time;

  GlobalDefaultsForBkgModel::krate_adj_limit = bkg_control.krate_adj_threshold;
  GlobalDefaultsForBkgModel::kmult_low_limit = bkg_control.kmult_low_limit;
  GlobalDefaultsForBkgModel::kmult_hi_limit = bkg_control.kmult_hi_limit;

  GlobalDefaultsForBkgModel::ssq_filter = bkg_control.ssq_filter;
}

void SetupXtalkParametersForBkgModel (BkgModelControlOpts &bkg_control, char *chipType,char *experimentName)
{
  // search for config file for chip type
  if (!bkg_control.xtalk)
  {
    //create default param file name
    char filename[64];
    sprintf (filename, "xtalk_%s.param", chipType);
    bkg_control.xtalk = GetIonConfigFile (filename);
    if (!bkg_control.xtalk)
      GlobalDefaultsForBkgModel::ReadXtalk ("");   // nothing found
  } // set defaults if nothing set at all
  if (bkg_control.xtalk)
  {
    if (strcmp (bkg_control.xtalk, "local") == 0)
    {
      char my_file[2048];
      sprintf (my_file,"%s/my_xtalk.txt",experimentName);
      GlobalDefaultsForBkgModel::ReadXtalk (my_file);   //rerunning in local directory for optimization purposes
    }
    else if (strcmp (bkg_control.xtalk, "disable") != 0)     //disabled = don't load
      GlobalDefaultsForBkgModel::ReadXtalk (bkg_control.xtalk);
    else GlobalDefaultsForBkgModel::ReadXtalk ("");   // must be non-null to be happy
  } // isolated function

}

//@TODO: should be BkgModelControlOpts
void SetBkgModelGlobalDefaults (BkgModelControlOpts &bkg_control, char *chipType,char *experimentName)
{
  // @TODO: Bad coding style to use static variables as shared global state
  GlobalDefaultsForBkgModel::SetChipType (chipType);   // bad, but need to know this before the first Image arrives
  // better to have global object pointed to by individual entities to make this maneuver very clear
  
  
  ReadOptimizedDefaultsForBkgModel (bkg_control,chipType,experimentName);
  
  PresetDefaultsForBkgModel(bkg_control); // set from command line, possibly overridden by optimization file
  
  OverrideDefaultsForBkgModel (bkg_control,chipType,experimentName);   // after we read from the file so we can tweak

  SetupXtalkParametersForBkgModel (bkg_control,chipType,experimentName);

}