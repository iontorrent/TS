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

#include "stats.h"

#include "InputStructures.h"
#include "AlignmentAssist.h"
#include "MiscUtil.h"
#include "ExtendedReadInfo.h"
#include "ClassifyVariant.h"
#include "ExtendParameters.h"
#include "MultiFlowDist.h"
#include "StackEngine.h"
#include "EnsembleGlue.h"


using namespace std;
using namespace BamTools;
using namespace ion;


// ----------------------------------------------------------------------


void ProcessOneVariant(PersistingThreadObjects &thread_objects, vcf::Variant ** candidate_variant, ExtendParameters * parameters,  InputStructures &global_context);

void EnsembleProcessOneVariant(PersistingThreadObjects &thread_objects, vcf::Variant ** candidate_variant,
                               ExtendParameters * parameters, InputStructures &global_context);

//void DoWorkForOneVariant(BamTools::BamMultiReader &bamReader, vcf::Variant **current_variant, string &local_contig_sequence,  ExtendParameters *parameters, InputStructures *global_context_ptr);
void DoWorkForOneVariant(PersistingThreadObjects &thread_objects, vcf::Variant **current_variant, ExtendParameters *parameters, InputStructures *global_context_ptr);


#endif //HANDLEVARIANT_H
