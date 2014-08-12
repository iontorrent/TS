/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef CLONALFILTER_H
#define CLONALFILTER_H

#include <deque>
#include <vector>
#include "BkgFitterTracker.h"

void ApplyClonalFilter(Mask& mask, const char* results_folder,
		       std::vector<RegionalizedData *>& sliced_chip,
           const PolyclonalFilterOpts & opts);


#endif // CLONALFILTER_H
