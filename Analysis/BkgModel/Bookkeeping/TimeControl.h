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

struct DataWeightDefaults{

  // revised emphasis parameterization for clarity
  // linear approximation to weighting
  bool use_data_weight;
  float blank_span; // generic emphasis span  (17 frames)
  float zero_span; // span for a zero-mer emphasis (13 frames)
  float nmer_span_increase; // frames per additiona n-mer over a zeromer (1.0 frames)
  float zero_fade_start; // fraction of zeromer span to fade-to-zero weight over (0.5)
  float prefix_start; // offset for prefix = 3 (-3 frames)
  float prefix_value; // fraction of maximum value = 0.15

  DataWeightDefaults();
  void FromJson(Json::Value &gopt_params);
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar
        & blank_span
        & zero_span
        & nmer_span_increase
        & zero_fade_start
        & prefix_start
        & prefix_value
        & use_data_weight;
  }

};

struct EmphasisDefaults{
  //@TODO: why are these not simply part of "EmphasisVector.h"?
  float emphasis_ampl_default;
  float emphasis_width_default;
  int   numEv;                       // number of emphasis vectors allocated
  float emp[NUMEMPHASISPARAMETERS];  // parameters for emphasis vector generation
  bool point_emphasis_by_compression;

  EmphasisDefaults();
  void DumpPoorlyStructuredText();
  void FromJson(Json::Value &gopt_params);

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar

        & emphasis_ampl_default
        & emphasis_width_default
        & numEv
        & emp;
  }
};


struct TimeAndEmphasisDefaults{
  float nuc_flow_frame_width;
  int   time_left_avg;
  int   time_start_detail;
  int   time_stop_detail;


  DataWeightDefaults data_weights;
  EmphasisDefaults emphasis_params;

  TimeAndEmphasisDefaults();

  void FromJson(Json::Value &gopt_params);
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar
        & data_weights

        & nuc_flow_frame_width
        & time_left_avg
        & time_start_detail
        & time_stop_detail
        &emphasis_params;
  }
};

#endif //TIMECONTROL_H
