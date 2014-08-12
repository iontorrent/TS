/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#include "FitterDefaults.h"


static float _clonal_call_scale[MAGIC_CLONAL_CALL_ARRAY_SIZE] = {0.902,0.356,0.078,0.172,0.436,0.0,0.0,0.0,0.0,0.0,0.0,0.0};


FitterDefaults::FitterDefaults()
{
  memcpy(clonal_call_scale,_clonal_call_scale,sizeof(float[MAGIC_CLONAL_CALL_ARRAY_SIZE]));
  clonal_call_penalty = 1600.0f;
}


void FitterDefaults::FromJson(Json::Value &gopt_params){
  const Json::Value source_clonal_call_scale = gopt_params["clonal_call_scale"];
  for ( int index = 0; index < (int) source_clonal_call_scale.size(); ++index )
    clonal_call_scale[index] = source_clonal_call_scale[index].asFloat();

}

void FitterDefaults::FromCharacterLine(char *line){
  float d[10];
  int num;

  if ( strncmp ( "clonal_call_scale",line,17 ) == 0 )
  {
    num = sscanf ( line,"clonal_call_scale: %f %f %f %f %f", &d[0],&d[1],&d[2],&d[3],&d[4] );
    for ( int i=0;i<num;i++ ) clonal_call_scale[i]=d[i];
  }

}

void FitterDefaults::DumpPoorlyStructuredText(){
  printf ( "clonal_call_scale: %f\t%f\t%f\t%f\t%f\n",clonal_call_scale[0], clonal_call_scale[1], clonal_call_scale[2], clonal_call_scale[3], clonal_call_scale[4] );
}
