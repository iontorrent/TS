/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     ExtendedReadInfo.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#ifndef EXTENDEDREADINFO_H
#define EXTENDEDREADINFO_H

#include <vector>

#include "InputStructures.h"
#include "ExtendParameters.h"
#include "AlleleParser.h"

using namespace std;

struct Alignment;

void UnpackOnLoad(Alignment *rai, const InputStructures &global_context);
void UnpackOnLoadLight(Alignment *rai, const InputStructures &global_context);
void FilterByModifiedMismatches(Alignment *rai, int read_mismatch_limit, const TargetsManager *const targets_manager);

#endif //EXTENDEDREADINFO_H
