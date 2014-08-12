/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#include "TimeControl.h"

static float  _emp[NUMEMPHASISPARAMETERS]  = {6.86, 1.1575, 2.081, 1.230, 7.2625, 1.91, 0.0425, 19.995};

TimeAndEmphasisDefaults::TimeAndEmphasisDefaults()
{
  nuc_flow_frame_width = 22.5;  // 1.5 seconds * 15 frames/second
  time_left_avg = 5;
  time_start_detail = -5;
  time_stop_detail = 16;
  point_emphasis_by_compression = true;

  memcpy(emp,_emp,sizeof(float[NUMEMPHASISPARAMETERS]));
  emphasis_ampl_default = 7.25;
  emphasis_width_default = 2.89;
}


void TimeAndEmphasisDefaults::DumpPoorlyStructuredText(){
  printf ( "emphasis: %f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",emp[0],emp[1],emp[2],emp[3],emp[4],emp[5],emp[6],emp[7] );
  printf ( "emp_amplitude: \t%f\n",emphasis_ampl_default );
  printf ( "emp_width: \t%f\n",emphasis_width_default );
}

void TimeAndEmphasisDefaults::FromJson(Json::Value &gopt_params){
  const Json::Value emphasis = gopt_params["emphasis"];
  for ( int index = 0; index < (int) emphasis.size(); ++index )
    emp[index] = emphasis[index].asFloat();

  emphasis_ampl_default = gopt_params["emp_amplitude"].asFloat();
  emphasis_width_default = gopt_params["emp_width"].asFloat();
}

void TimeAndEmphasisDefaults::FromCharacterLine(char *line){
  float d[10];
  int num;

  if ( strncmp ( "emphasis",line,8 ) == 0 )
  {
    num = sscanf ( line,"emphasis: %f %f %f %f %f %f %f %f", &d[0],&d[1],&d[2],&d[3],&d[4],&d[5],&d[6],&d[7] );
    for ( int i=0;i<num;i++ ) emp[i]=d[i];
  }
  if ( strncmp ( "emp_amp",line,7 ) == 0 )
    num = sscanf ( line,"emp_amplitude: %f",&emphasis_ampl_default );
  if ( strncmp ( "emp_width",line,7 ) == 0 )
    num = sscanf ( line,"emp_width: %f",&emphasis_width_default );
  if ( strncmp ( "nuc_flow_timing", line, 15 ) == 0 )
  {
    num = sscanf ( line,"nuc_flow_frame_width: %f", &d[0] );
    nuc_flow_frame_width = d[0];
  }
  if ( strncmp ( "time_compression", line, 16 ) == 0 )
  {
    num = sscanf ( line,"time_compression: %f %f %f", &d[0], &d[1],&d[2] );
    time_left_avg = d[0];
    time_start_detail = d[1];
    time_stop_detail = d[2];
  }
}
