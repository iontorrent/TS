/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     HandleVariant.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#ifndef HANDLEVARIANT_H
#define HANDLEVARIANT_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <ctype.h>
#include <algorithm>
#include "api/api_global.h"
#include "api/BamAux.h"
#include "api/BamConstants.h"
#include "api/BamReader.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"
#include "api/SamReadGroup.h"
#include "api/SamReadGroupDictionary.h"
#include "api/SamSequence.h"
#include "api/SamSequenceDictionary.h"

#include "sys/types.h"
#include "sys/stat.h"
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <vector>

#include <Variant.h>

#include "InputStructures.h"
#include "MiscUtil.h"
#include "ExtendedReadInfo.h"
#include "ClassifyVariant.h"
#include "ExtendParameters.h"

#include "StackEngine.h"
#include "DiagnosticJson.h"


using namespace std;
using namespace BamTools;
using namespace ion;


// ----------------------------------------------------------------------


void EnsembleProcessOneVariant(PersistingThreadObjects &thread_objects, VariantCallerContext& vc,
    VariantCandidate &current_variant, const PositionInProgress& bam_position);


#endif //HANDLEVARIANT_H
