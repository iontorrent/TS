/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MASKFUNCTIONS_H
#define MASKFUNCTIONS_H

#include "Mask.h"
#include "CommandLineOpts.h"

void ExportSubRegionSpecsToMask (SpatialContext &loc_context);
void UpdateBeadFindOutcomes(Mask *maskPtr, Region &wholeChip, char *experimentName, bool not_single_beadfind, int update_stats);
void SetExcludeMask(CommandLineOpts &clo, Mask *maskPtr, char *chipType, int rows, int cols);
void LoadBeadMaskFromFile(SystemContext &sys_context, Mask *maskPtr);
void SetSpatialContextAndMask(SpatialContext &loc_context, Mask *maskPtr, int &rows, int &cols);

#endif // MASKFUNCTIONS_H