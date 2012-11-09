/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef GLOBALDEFAULTSFORBKGMODEL_H
#define GLOBALDEFAULTSFORBKGMODEL_H

#include <vector>
#include <string>
#include "BkgMagicDefines.h"
#include "Serialization.h"

// divide god-structure to individual item control structures associated with specific purposes/modules

struct RegionParamDefault{
  // not plausible at all
  // various parameters used to initialize the model
  float sens_default;
  float dntp_uM[NUMNUC];
  float molecules_to_micromolar_conv;
  float tau_R_m_default;
  float tau_R_o_default;

  float krate_default[NUMNUC];
  float d_default[NUMNUC];
  float kmax_default[NUMNUC];
  float sigma_mult_default[NUMNUC];
  float t_mid_nuc_delay_default[NUMNUC];

  RegionParamDefault();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar
      & sens_default
      & dntp_uM
      & molecules_to_micromolar_conv
      & tau_R_m_default
      & tau_R_o_default
      & krate_default
      & d_default
      & kmax_default
      & sigma_mult_default
      & t_mid_nuc_delay_default;
  }
};

struct LocalSigProcControl{
  
  float krate_adj_limit;
  float dampen_kmult;
  float kmult_low_limit;
  float kmult_hi_limit;
  
  // why would this be global again?
  bool  no_RatioDrift_fit_first_20_flows;
  bool fitting_taue;
  bool  var_kmult_only;    // always do variable kmult override
  bool  generic_test_flag; // control any features that I'm just testing
  bool fit_alternate;
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
  bool proton_dot_wells_post_correction;
  int  single_flow_fit_max_retry;
  bool per_flow_t_mid_nuc_tracking;
  bool regional_sampling;
  bool prefilter_beads;

  LocalSigProcControl();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar
      & krate_adj_limit
      & dampen_kmult
      & kmult_low_limit
      & kmult_hi_limit
      & no_RatioDrift_fit_first_20_flows
      & fitting_taue
      & var_kmult_only
      & generic_test_flag
      & fit_alternate 
      & choose_time
      & projection_search_enable
      & ssq_filter
      & do_clonal_filter
      & enable_dark_matter
      & use_vectorization
      & AmplLowerLimit
      & proton_dot_wells_post_correction
      & single_flow_fit_max_retry
      & per_flow_t_mid_nuc_tracking
      & prefilter_beads
      & regional_sampling;
  }
};
  
// I'm crying because this object isn't unified across our codebase
struct FlowMyTears{
  // plausibly a shared object
  int              flow_order_len;     // length of entire flow order sequence (might be > NUMFB)
  std::vector<int> glob_flow_ndx_map;  // maps flow number within a cycle to nucleotide (flow_order_len values)
  std::string      flowOrder;          // entire flow order as a char array (flow_order_len values)
   
  FlowMyTears();

  void SetFlowOrder(char *_flowOrder);
  int  GetNucNdx(int flow_ndx)
  {
    return glob_flow_ndx_map[flow_ndx%flow_order_len];
  }

  // special functions for double-tap flows
  int IsDoubleTap(int flow)
  {
    // may need to refer to earlier flows
    if (flow==0)
      return(1); // can't be double tap

    if (glob_flow_ndx_map[flow%flow_order_len]==glob_flow_ndx_map[(flow-1+flow_order_len)%flow_order_len])
      return(0);
    return(1);
  }

  void GetFlowOrderBlock(int *my_flow, int i_start, int i_stop);

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar
      & flow_order_len
      & glob_flow_ndx_map
      & flowOrder;
  }
};

struct TimeAndEmphasisDefaults{
  float nuc_flow_frame_width;
  int   time_left_avg;
  int   time_start_detail;
  int   time_stop_detail;
  float emphasis_ampl_default;
  float emphasis_width_default;
  int   numEv;                       // number of emphasis vectors allocated
  float emp[NUMEMPHASISPARAMETERS];  // parameters for emphasis vector generation
  bool point_emphasis_by_compression; 
  
  TimeAndEmphasisDefaults();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar
      & nuc_flow_frame_width
      & time_left_avg
      & time_start_detail
      & time_stop_detail
      & emphasis_ampl_default
      & emphasis_width_default
      & numEv
      & point_emphasis_by_compression
      & emp;
  }
};

struct FitterDefaults{
  // weighting applied to clonal restriction on different hp lengths
  float clonal_call_scale[MAGIC_CLONAL_CALL_ARRAY_SIZE];
  float clonal_call_penalty;
  float shrink_factor;
  FitterDefaults();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar
      & clonal_call_scale
      & clonal_call_penalty;
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

  // ugly
  std::string chipType;   // Yes, this is available through Image, theoretically, but I need to know this before I see the first image passsed.
  std::string xtalk_name; // really bad, but I can't pass anything through analysis at all!!!

  void  SetDntpUM(float concentration, int NucID);
  float GetDntpUM(int NucID);
  void SetAllConcentrations(float *_dntp_uM);
  
  void  SetGoptDefaults(char *gopt);
  void  ReadEmphasisVectorFromFile(char *experimentName);

  void SetKrateDefaults(float *krate_values)
  {
    for (int i=0;i < NUMNUC;i++)
      region_param_start.krate_default[i] = krate_values[i];
  }

  void SetDDefaults(float *d_values)
  {
    for (int i=0;i < NUMNUC;i++)
      region_param_start.d_default[i] = d_values[i];
  }

  void SetKmaxDefaults(float *kmax_values)
  {
    for (int i=0;i < NUMNUC;i++)
      region_param_start.kmax_default[i] = kmax_values[i];
  }
  
  void SetDampenKmult(float damp){signal_process_control.dampen_kmult = damp;};
  void FixRdrInFirst20Flows(bool fixed_RatioDrift) { signal_process_control.no_RatioDrift_fit_first_20_flows = fixed_RatioDrift; }
  void SetFittingTauE(bool fit_taue) { signal_process_control.fitting_taue = fit_taue; }
  bool GetVarKmultControl(){return(signal_process_control.var_kmult_only);};
  void SetVarKmultControl(bool _var_kmult_only){signal_process_control.var_kmult_only = _var_kmult_only;};
  void SetGenericTestFlag(bool _generic_test_flag){signal_process_control.generic_test_flag = _generic_test_flag;};
  void SetFitAlternate(bool _fit_alternate){signal_process_control.fit_alternate = _fit_alternate;};
  void SetEmphasizeByCompression(bool _emp_by_comp){data_control.point_emphasis_by_compression = _emp_by_comp;};
  void ReadXtalk(char *name);
  void SetChipType(char *name);
  void DumpExcitingParameters(char *fun_string);

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
	& chipType
	& xtalk_name;
      // fprintf(stdout, "done\n");
    }

};

#endif // GLOBALDEFAULTSFORBKGMODEL_H
