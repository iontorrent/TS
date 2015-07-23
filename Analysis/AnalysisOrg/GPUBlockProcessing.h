/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef GPUBLOCKPROCESSING_H 
#define GPUBLOCKPROCESSING_H

#include "SignalProcessingFitterQueue.h"

bool ProcessProtonBlockImageOnGPU(
    BkgModelWorkInfo* fitterInfo, 
    int flowBlockSize);

#endif // GPUBLOCKPROCESSING_H
