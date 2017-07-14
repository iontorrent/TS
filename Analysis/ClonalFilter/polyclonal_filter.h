/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef POLYCLONAL_FILTER_H
#define POLYCLONAL_FILTER_H

#include "json/json.h"
#include "IO/OptBase.h"
#include "Util/Serialization.h"

class PolyclonalFilterOpts {
  public:
    PolyclonalFilterOpts();

    static void PrintHelp(bool anaysis_call);

    bool    enable;
    bool    filter_clonal_enabled_tfs;
    bool    filter_clonal_enabled_lib;
    int     filter_clonal_maxreads;
    int     mixed_first_flow;
    int     mixed_last_flow;
    int     max_iterations;
    int     mixed_model_option;
    double  mixed_stringency;
    bool    verbose;
    bool    use_last_iter_params;
    bool    filter_extreme_ppf_only;

    void SetOpts(bool analysis_call, OptArgs &opts, Json::Value& json_params, int num_flows);
    void Disable();

private:

  void SetDefaults();

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar
      & enable
      & filter_clonal_enabled_tfs
      & filter_clonal_enabled_lib
      & filter_clonal_maxreads
      & mixed_first_flow
      & mixed_last_flow
        & max_iterations
        & mixed_model_option
        & mixed_stringency
        & verbose
        & use_last_iter_params
        & filter_extreme_ppf_only;
  }
};

#endif // POLYCLONAL_FILTER_H
