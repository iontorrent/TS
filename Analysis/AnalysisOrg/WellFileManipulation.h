/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef WELLFILEMANIPULATION_H
#define WELLFILEMANIPULATION_H

#include <stdio.h>
#include <string.h>

#include "CommandLineOpts.h"
#include "RawWells.h"
#include "ChipIdDecoder.h"
#include "Utils.h"

class Mask;

typedef struct writeFlowDataFuncArg
{
  string filePath;
  int numCols;
  size_t stepSize;
  bool saveAsUShort;
  SemQueue* packQueuePtr;
  SemQueue* writeQueuePtr;
} writeFlowDataFuncArg;

void* WriteFlowDataFunc(void* arg0);

void GetMetaDataForWells(char *dirExt, RawWells &rawWells, const char *chipType);
void SetWellsToLiveBeadsOnly(RawWells &rawWells, Mask *maskPtr);
void CreateWellsFileForWriting (RawWells &rawWells, Mask *maskPtr,
                                CommandLineOpts &inception_state,
                                int numFlows,
                                int numRows, int numCols,
                                const char *chipType);
                                

#endif // WELLFILEMANIPULATION_H
