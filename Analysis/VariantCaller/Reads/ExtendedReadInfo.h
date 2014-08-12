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

void UnpackOnLoad(Alignment *rai, const InputStructures &global_context, const ExtendParameters& parameters);

//! @brief  Creates a stack of reads that provide evidence in the case of our candidate variant
void StackUpOneVariant(vector<const Alignment *>& read_stack, int variant_start_pos, int variant_end_pos,
                       const ExtendParameters &parameters, const PositionInProgress& bam_position);


#endif //EXTENDEDREADINFO_H
