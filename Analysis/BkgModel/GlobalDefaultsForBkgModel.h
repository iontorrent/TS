/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef GLOBALDEFAULTSFORBKGMODEL_H
#define GLOBALDEFAULTSFORBKGMODEL_H

#include <vector>
#include <string>
#include "BkgMagicDefines.h"
#include "Serialization.h"
#include "WellXtalk.h"
#include "ControlSingleFlow.h"
#include "RegionParamDefault.h"
#include "TimeControl.h"
#include "FitterDefaults.h"
#include "FlowDefaults.h"
#include "OptBase.h"

// divide god-structure to individual item control structures associated with specific purposes/modules

struct LocalSigProcControl{
  ControlSingleFlow single_flow_master;

  // why would this be global again?
  bool  no_RatioDrift_fit_first_20_flows;
  bool use_alternative_etbR_equation;

  bool fitting_taue;
  int hydrogenModelType;
  bool  var_kmult_only;    // always do variable kmult override
  bool fit_alternate;
  bool fit_gauss_newton;
  int   choose_time;

  bool  projection_search_enable;
  float ssq_filter;

  // control flags
  // whether to skip processing for mixed reads:
  bool  do_clonal_filter;
  bool enable_dark_matter;
  bool  use_vectorization;
  float AmplLowerLimit;  // sadly ignored at the moment

  // options added for proton data processing
  bool enable_well_xtalk_correction;
  int  single_flow_fit_max_retry;
  bool per_flow_t_mid_nuc_tracking;
  bool exp_tail_fit;
  bool pca_dark_matter;
  bool regional_sampling;
  int regional_sampling_type;
  bool prefilter_beads;
  bool amp_guess_on_gpu;
  bool recompress_tail_raw_trace;

  LocalSigProcControl();
  void PrintHelp();
  void SetOpts(OptArgs &opts, Json::Value& json_params);

  int max_frames;
  void set_max_frames(int nFrames) { max_frames = nFrames; }
  int get_max_frames() { return (max_frames); }

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar
        & no_RatioDrift_fit_first_20_flows
        & use_alternative_etbR_equation
        & fitting_taue
        & hydrogenModelType
        & var_kmult_only
        & fit_alternate
        & fit_gauss_newton
        & choose_time
        & projection_search_enable
        & ssq_filter
        & do_clonal_filter
        & enable_dark_matter
        & use_vectorization
        & AmplLowerLimit
        & enable_well_xtalk_correction
        & single_flow_fit_max_retry
        & per_flow_t_mid_nuc_tracking
        & exp_tail_fit
        & pca_dark_matter
        & prefilter_beads
        & regional_sampling
        & regional_sampling_type
        & amp_guess_on_gpu
        & max_frames
        & recompress_tail_raw_trace;
  }
};


class GlobalDefaultsForBkgModel
{
public:
  FlowMyTears             flow_global;
  RegionParamDefault      region_param_start;
  LocalSigProcControl     signal_process_control;
  TimeAndEmphasisDefaults data_control;
  FitterDefaults          fitter_defaults;
  WellXtalk well_xtalk_master;

  // ugly
  std::string chipType;   // Yes, this is available through Image, theoretically, but I need to know this before I see the first image passsed.
  std::string xtalk_name; // really bad, but I can't pass anything through analysis at all!!!

  void FixRdrInFirst20Flows(bool fixed_RatioDrift) { signal_process_control.no_RatioDrift_fit_first_20_flows = fixed_RatioDrift; }
  void SetUse_alternative_etbR_equation(bool if_use_alternative_etbR_equation) { signal_process_control.use_alternative_etbR_equation = if_use_alternative_etbR_equation; }
  void SetFittingTauE(bool fit_taue) { signal_process_control.fitting_taue = fit_taue; }
  void SetHydrogenModel( int model ) { signal_process_control.hydrogenModelType = model; }
  bool GetVarKmultControl(){return(signal_process_control.var_kmult_only);};
  void SetVarKmultControl(bool _var_kmult_only){signal_process_control.var_kmult_only = _var_kmult_only;};
  void SetFitAlternate(bool _fit_alternate){signal_process_control.fit_alternate = _fit_alternate;};
  void SetFitGaussNewton(bool _fit_gauss_newton){signal_process_control.fit_gauss_newton = _fit_gauss_newton;};
  void SetEmphasizeByCompression(bool _emp_by_comp){data_control.point_emphasis_by_compression = _emp_by_comp;};
  void ReadXtalk(char *name);
  void SetChipType(const char *name);

  // i/o from files for parameters
  void  SetGoptDefaults(char *gopt);
  void  ReadEmphasisVectorFromFile(char *experimentName);
  void DumpExcitingParameters(char *fun_string);
  void GoptDefaultsFromJson(char *fname);
  void GoptDefaultsFromPoorlyStructuredFile(char *fname);

  void PrintHelp();
  void SetOpts(OptArgs &opts, Json::Value& json_params);

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    // fprintf(stdout, "Serialize GlobalDefaultsForBkgModel ... ");
    ar
        & flow_global
        & region_param_start
        & signal_process_control
        & data_control
        & fitter_defaults
        & well_xtalk_master
        & chipType
        & xtalk_name;
    // fprintf(stdout, "done\n");
  }

};

#endif // GLOBALDEFAULTSFORBKGMODEL_H
