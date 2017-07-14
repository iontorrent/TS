/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MASKFUNCTIONS_H
#define MASKFUNCTIONS_H

#include "Mask.h"
#include "CommandLineOpts.h"

void SetExcludeMask(SpatialContext &loc_context, Mask *maskPtr, char *chipType, int rows, int cols,
                    std::string exclusionMaskFileName, bool beadfindThumbnail);


#endif // MASKFUNCTIONS_H
