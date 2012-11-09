/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef KEYCONTEXT_H
#define KEYCONTEXT_H

#include <string.h>
#include <stdlib.h>

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
    ~KeyContext();
};

#endif // KEYCONTEXT_H