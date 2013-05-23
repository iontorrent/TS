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


void GetMetaDataForWells(char *dirExt, RawWells &rawWells, const char *chipType);
void SetWellsToLiveBeadsOnly(RawWells &rawWells, Mask *maskPtr);
void CreateWellsFileForWriting (RawWells &rawWells, Mask *maskPtr,
                                CommandLineOpts &inception_state,
                                int num_fb,
                                int numFlows,
                                int numRows, int numCols,
                                const char *chipType);
                                
void IncrementalWriteWells (RawWells &rawWells,int flow, bool last_flow,int saveWellsFrequency,int num_fb, int numFlows);
void OpenExistingWellsForOneChunk(RawWells &rawWells,  int start_of_chunk, int chunk_depth);
void WriteOneChunkAndClose(RawWells &rawWells, SumTimer &timer);
int FigureChunkDepth(int flow, int numFlows, int write_well_flow_interval);
bool NeedToOpenWellChunk(int flow, int write_well_flow_interval);


#endif // WELLFILEMANIPULATION_H
