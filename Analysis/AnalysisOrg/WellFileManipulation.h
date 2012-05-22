/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef WELLFILEMANIPULATION_H
#define WELLFILEMANIPULATION_H

#include <stdio.h>
#include <string.h>

#include "CommandLineOpts.h"
#include "RawWells.h"
#include "ChipIdDecoder.h"

class Mask;

void SetChipTypeFromWells(RawWells &rawWells);
void GetMetaDataForWells(char *dirExt, RawWells &rawWells, const char *chipType);
void SetWellsToLiveBeadsOnly(RawWells &rawWells, Mask *maskPtr);
void CreateWellsFileForWriting (RawWells &rawWells, Mask *maskPtr,
                                CommandLineOpts &clo,
                                int num_fb,
                                int numFlows,
                                int numRows, int numCols,
                                const char *chipType);
void IncrementalWriteWells (RawWells &rawWells,int flow, bool last_flow,int saveWellsFrequency,int num_fb, int numFlows);

#endif // WELLFILEMANIPULATION_H
