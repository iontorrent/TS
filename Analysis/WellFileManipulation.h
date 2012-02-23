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
void CopyTmpWellFileToPermanent(CommandLineOpts &clo, char *experimentName);
void MakeNewTmpWellsFile(SystemContext &sys_context, char *experimentName);
void CleanupTmpWellsFile(CommandLineOpts &clo);

#endif // WELLFILEMANIPULATION_H
