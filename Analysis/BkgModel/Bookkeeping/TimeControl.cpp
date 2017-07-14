/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#include "TimeControl.h"

static float  _emp[NUMEMPHASISPARAMETERS]  = {6.86, 1.1575, 2.081, 1.230, 7.2625, 1.91, 0.0425, 19.995};

TimeAndEmphasisDefaults::TimeAndEmphasisDefaults()
{
  nuc_flow_frame_width = 22.5;  // 1.5 seconds * 15 frames/second
  time_left_avg = 5;
  time_start_detail = -5;
  time_stop_detail = 16;

}

EmphasisDefaults::EmphasisDefaults(){
  point_emphasis_by_compression = true;

  memcpy(emp,_emp,sizeof(float[NUMEMPHASISPARAMETERS]));
  emphasis_ampl_default = 7.25;
  emphasis_width_default = 2.89;

}

DataWeightDefaults::DataWeightDefaults(){
  blank_span = 17.0f;
  zero_span = 17.0f;
  nmer_span_increase = 1.0f;
  zero_fade_start = 0.5f;
  prefix_start = 3;
  prefix_value = 0.15f;
  use_data_weight = false;

}


void EmphasisDefaults::DumpPoorlyStructuredText(){
  printf ( "emphasis: %f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",emp[0],emp[1],emp[2],emp[3],emp[4],emp[5],emp[6],emp[7] );
  printf ( "emp_amplitude: \t%f\n",emphasis_ampl_default );
  printf ( "emp_width: \t%f\n",emphasis_width_default );
}

void DataWeightDefaults::FromJson(Json::Value &gopt_params){
  if (!gopt_params["DataWeight"].isNull()){
    use_data_weight=true;
    const Json::Value data_weight = gopt_params["DataWeight"];
    blank_span =  data_weight["blank_span"].asFloat();
    zero_span = data_weight["zero_span"].asFloat();
    nmer_span_increase = data_weight["nmer_span_increase"].asFloat();
    zero_fade_start = data_weight["zero_fade_start"].asFloat();
    prefix_start = data_weight["prefix_start"].asFloat();
    prefix_value = data_weight["prefix_value"].asFloat();
  }
}

void EmphasisDefaults::FromJson(Json::Value &gopt_params){
  const Json::Value emphasis = gopt_params["emphasis"];
  for ( int index = 0; index < (int) emphasis.size(); ++index )
    emp[index] = emphasis[index].asFloat();

  emphasis_ampl_default = gopt_params["emp_amplitude"].asFloat();
  emphasis_width_default = gopt_params["emp_width"].asFloat();
}

void TimeAndEmphasisDefaults::FromJson(Json::Value &gopt_params){
  emphasis_params.FromJson(gopt_params);
  data_weights.FromJson(gopt_params);
}
