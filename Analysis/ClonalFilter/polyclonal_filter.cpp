/* Copyright (C) 2016 Thermo Fisher Scientific, All Rights Reserved */
#include "polyclonal_filter.h"

PolyclonalFilterOpts::PolyclonalFilterOpts()
{
  SetDefaults();
}


void PolyclonalFilterOpts::PrintHelp(bool analysis_call)
{
    printf ("     Polyclonal Filter Options:\n");

    if (analysis_call)
      printf ("     --clonal-filter-bkgmodel  BOOL       enable polyclonal filter during signal processing [on]\n");
    else{
      printf ("     --clonal-filter-solve     on/off     apply polyclonal filter [off]\n");
      printf ("     --clonal-filter-tf        on/off     apply polyclonal filter to TFs [off]\n");
      printf ("     --clonal-filter-maxreads  INT        maximum number of library reads used for polyclonal filter training [100000]\n");
    }

    printf ("     --mixed-first-flow        INT        mixed first flow of polyclonal filter [12]\n");
    printf ("     --mixed-last-flow         INT        mixed last flow of polyclonal filter [72]\n");
    printf ("     --max-iterations          INT        max iterations of polyclonal filter [30]\n");
    printf ("     --mixed-model-option      INT        mixed model option of polyclonal filter [0]\n");
    printf ("     --mixed-stringency        DOUBLE     mixed stringency of polyclonal filter [0.5]\n");
    printf ("     --clonal-filter-debug     BOOL       enable polyclonal filter debug output [off]\n");
    printf ("     --clonal-filter-use-last-iter-params    BOOL    use last EM iteration cluster parameters if no convergence [on]\n");
    printf ("     --filter-extreme-ppf-only BOOL       Skip polyclonal filter training and filter for extreme ppf only [off]\n");
    printf ("\n");
}

//-----------------------------------------------------------------------------

void PolyclonalFilterOpts::SetDefaults()
{
  enable = false;

  filter_clonal_enabled_tfs = false;
  filter_clonal_enabled_lib = false;
  filter_clonal_maxreads    = 100000;

  mixed_first_flow = 12;
  mixed_last_flow = 72;
  max_iterations = 30;
  mixed_model_option = 0;
  mixed_stringency = 0.5;
  use_last_iter_params = true;
  verbose = false;
  filter_extreme_ppf_only = false;
}

//-----------------------------------------------------------------------------

void PolyclonalFilterOpts::Disable()
{
  enable = false;
  filter_clonal_enabled_tfs = false;
  filter_clonal_enabled_lib = false;
}

//-----------------------------------------------------------------------------

void PolyclonalFilterOpts::SetOpts(bool analysis_call, OptArgs &opts, Json::Value& json_params, int num_flows)
{
  SetDefaults();

  // Analysis only option
  if (analysis_call)
    enable = RetrieveParameterBool(opts, json_params, '-', "clonal-filter-bkgmodel", true);
  else {
    // Basecaller only option
    filter_clonal_enabled_tfs = RetrieveParameterBool(opts, json_params, '-', "clonal-filter-tf", false);
    filter_clonal_enabled_lib = RetrieveParameterBool(opts, json_params,'-', "clonal-filter-solve", false);
    filter_clonal_maxreads    = RetrieveParameterInt (opts, json_params,'-', "clonal-filter-maxreads", 100000);
    enable = filter_clonal_enabled_tfs or filter_clonal_enabled_lib;
  }

  // Joint Analysis and BaseCaller Options
  mixed_first_flow = RetrieveParameterInt(opts, json_params, '-', "mixed-first-flow", 12);
  mixed_last_flow  = RetrieveParameterInt(opts, json_params, '-', "mixed-last-flow", 72);
  // Check prevents memory overrun and segfaults
  if (mixed_last_flow > num_flows){
    cerr << "PolyclonalFilterOpts WARNING: mixed-last-flow is larger than number of flows in the run: " << num_flows << ". Disabling polyclonal filter." << endl;
    Disable();
  }

  max_iterations = RetrieveParameterInt(opts, json_params, '-', "max-iterations", 30);
  mixed_model_option = RetrieveParameterInt(opts, json_params, '-', "mixed-model-option", 0);
  verbose = RetrieveParameterBool(opts, json_params, '-', "clonal-filter-debug", false);
  use_last_iter_params = RetrieveParameterBool(opts, json_params, '-', "clonal-filter-use-last-iter-params", true);
  filter_extreme_ppf_only = RetrieveParameterBool(opts, json_params, '-', "filter-extreme-ppf-only", false);

  // Transform stringency to log scale
  double stringency = min(1.0, max(0.0, RetrieveParameterDouble(opts, json_params, '-', "mixed-stringency", 0.5)));
  if(stringency > 0.5){
    mixed_stringency = 0.5 * std::log10((stringency - 0.5)*18 + 1) + 0.5;
  }
  else if (stringency < 0.5){
    mixed_stringency = 0.5 - 0.5 * std::log10(( 0.5 - stringency)*18 + 1);
  }
  else {
    mixed_stringency = 0.5;
  }

}

