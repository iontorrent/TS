/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "KeyContext.h"
#include "Utils.h"

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

void KeyContext::PrintHelp()
{
	printf ("     KeyContext\n");
    printf ("     --librarykey            STRING            setup library key []\n");
    printf ("     --tfkey                 STRING            setup tf key []\n");
    printf ("\n");
}

void KeyContext::SetOpts(OptArgs &opts, Json::Value& json_params)
{
	string libkey = RetrieveParameterString(opts, json_params, '-', "librarykey", "");
	int len = libkey.length();
	if( len > 0)
	{
		if ( libKey )
			free ( libKey );
		libKey = ( char * ) malloc ( len +1 );
		strcpy ( libKey, libkey.c_str() );
		ToUpper ( libKey );	
	}

	string tfkey = RetrieveParameterString(opts, json_params, '-', "tfkey", "");
	len = tfkey.length();
	if( len > 0)
	{
		if ( tfKey )
			free ( tfKey );
		tfKey = ( char * ) malloc ( len  +1 );
		strcpy ( tfKey, tfkey.c_str() );
		ToUpper ( tfKey );	
	}
}
