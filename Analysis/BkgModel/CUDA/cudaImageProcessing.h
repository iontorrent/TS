/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef CUDAIMAGEPROCESSING_H
#define CUDAIMAGEPROCESSING_H

#include "Image.h"
#include "TimeCompression.h"
#include "Region.h"
#include "BeadTracker.h"
#include "BkgTrace.h"



void GenerateAllBeadTrace_GPU (BkgTrace * Bkgtr, Region *region, BeadTracker &my_beads, Image *img, int iFlowBuffer);


#endif // CUDAIMAGEPROCESSING_H 
