/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     VariantCallUsingFlowDist.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#ifndef VARIANTCALLUSINGFLOWDIST_H
#define VARIANTCALLUSINGFLOWDIST_H

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
#include <levmar.h>
#include <Variant.h>
#include "peakestimator.h"
#include "stats.h"

#include "semaphore.h"
#include "HypothesisEvaluator.h"
#include "ExtendParameters.h"
#include "FlowDist.h"
#include "AlignmentAssist.h"
#include "MiscUtil.h"
#include "ClassifyVariant.h"
#include "VariantAssist.h"


using namespace std;
using namespace BamTools;
using namespace ion;

void UpdateFlowDistWithOrdinaryVariant(FlowDist *flowDist, bool strand, float refLikelihood, float minDistToRead, float distDelta);
void UpdateFlowDistWithLongVariant(FlowDist *flowDist, bool strand, float delta, AlleleIdentity &variant_identity, LocalReferenceContext seq_context);


/*bool filterVariants(string * filterReason, bool isSNP, bool isMNV, bool isIndel, bool isHPIndel, bool isReferenceCall,
                    float BayesianScore,  ControlCallAndFilters &my_controls, float uniModalStd,
                    float biModalPeak1Std,  float biModalPeak2Std , float stdBias, float refBias, float baseStdBias,
                    float minAlleleFreq, int hpLength, int inDelLength);*/


void CalculateOrdinaryScore(FlowDist *flowDist, AlleleIdentity &variant_identity,
                            vcf::Variant ** candidate_variant, ControlCallAndFilters &my_controls,
                            bool *isFiltered, int DEBUG);
void CalculatePeakFindingScore(FlowDist *flowDist, MultiAlleleVariantIdentity& multi_variant, unsigned int allele_idx,
		                       ControlCallAndFilters &my_controls, bool *isFiltered, int DEBUG);
                               
#endif //VARIANTCALLUSINGFLOWDIST_H
