/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef REGIONPARAMDEFAULT_H
#define REGIONPARAMDEFAULT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "json/json.h"
#include "BkgMagicDefines.h"
#include "Serialization.h"

//@TODO: why is this not just region-params?
struct RegionParamDefault{
  // not plausible at all
  // various parameters used to initialize the model
  float sens_default;
  float dntp_uM[NUMNUC];
  float molecules_to_micromolar_conv;
  float tau_R_m_default;
  float tau_R_o_default;
  float tau_E_default;
  float min_tauB_default;
  float max_tauB_default;
  float tauB_smooth_range_default;
  float tshift_default;

  float krate_default[NUMNUC];
  float d_default[NUMNUC];
  float kmax_default[NUMNUC];
  float sigma_mult_default[NUMNUC];
  float t_mid_nuc_delay_default[NUMNUC];

  RegionParamDefault();
  void FromJson(Json::Value &gopt_params);

  void BadIdeaComputeDerivedInput();
  void DumpPoorlyStructuredText();

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
      & tau_E_default
      & min_tauB_default

      & max_tauB_default
      & tshift_default
      & krate_default
      & d_default
      & kmax_default
      & sigma_mult_default
      & t_mid_nuc_delay_default;
  }
};



#endif //REGIONPARAMDEFAULT_H
