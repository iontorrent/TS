/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef KEYCONTEXT_H
#define KEYCONTEXT_H

#include <string.h>
#include <stdlib.h>
#include "OptBase.h"

// this should be replaced by a listing of keys
// not two specific keys
// as in the SeqList construct

class KeyContext{
  public:
     char *libKey;
    char *tfKey;
    int maxNumKeyFlows;
    int minNumKeyFlows;

    void DefaultKeys();
	void PrintHelp();
	void SetOpts(OptArgs &opts, Json::Value& json_params);
    ~KeyContext();
};

#endif // KEYCONTEXT_H