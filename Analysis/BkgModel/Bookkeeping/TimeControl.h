/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef TIMECONTROL_H
#define TIMECONTROL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "json/json.h"
#include "BkgMagicDefines.h"
#include "Serialization.h"


struct TimeAndEmphasisDefaults{
  float nuc_flow_frame_width;
  int   time_left_avg;
  int   time_start_detail;
  int   time_stop_detail;

  //@TODO: why are these not simply part of "EmphasisVector.h"?
  float emphasis_ampl_default;
  float emphasis_width_default;
  int   numEv;                       // number of emphasis vectors allocated
  float emp[NUMEMPHASISPARAMETERS];  // parameters for emphasis vector generation
  bool point_emphasis_by_compression;

  TimeAndEmphasisDefaults();
  void DumpPoorlyStructuredText();
  void FromJson(Json::Value &gopt_params);
  void FromCharacterLine(char *line);

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

#endif //TIMECONTROL_H
