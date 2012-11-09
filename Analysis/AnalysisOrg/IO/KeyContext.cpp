/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "KeyContext.h"

void KeyContext::DefaultKeys()
{
  libKey = strdup ( "TCAG" );
  tfKey = strdup ( "ATCG" );
  minNumKeyFlows = 99;
  maxNumKeyFlows = 0;
}

KeyContext::~KeyContext()
{
  if ( libKey )
    free ( libKey );
  if ( tfKey )
    free ( tfKey );
}