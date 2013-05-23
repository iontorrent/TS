/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     MultiFlowDist.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#ifndef MULTIFLOWDIST_H
#define MULTIFLOWDIST_H

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
#include <levmar.h>
#include <Variant.h>
#include "peakestimator.h"
#include "stats.h"

#include "semaphore.h"
#include "HypothesisEvaluator.h"

#include "FlowDist.h"
#include "InputStructures.h"
#include "AlignmentAssist.h"
#include "MiscUtil.h"
#include "ExtendedReadInfo.h"
#include "ClassifyVariant.h"
#include "VariantCallUsingFlowDist.h"
#include "ExtendParameters.h"
#include "StackPlus.h"
#include "LocalContext.h"
#include "DecisionTreeData.h"

using namespace std;
using namespace BamTools;
using namespace ion;



class MultiFlowDist {
  public:
    vector<FlowDist *> flowDistVector;

    // what alleles failed and who is my best
    vector<int> filteredAllelesIndex;
    int max_score_allele_index;

    MultiAlleleVariantIdentity multi_variant;

    bool ScoreAllAlleles(vcf::Variant ** candidate_variant,ExtendParameters *parameters,  int DEBUG);
    void SetupAllAlleles(vcf::Variant ** candidate_variant, const string & local_contig_sequence, ExtendParameters *parameters, InputStructures &global_context);
    void AllocateFlowDistVector(vcf::Variant ** candidate_variant, InputStructures &global_context);
    void EvaluateFlowDistForLong(HypothesisEvaluator &hypEvaluator, ExtendedReadInfo &current_read,
                                 InputStructures &global_context,
                                 const string &local_contig_sequence,
                                 FlowDist *flowDist, int variant_start_pos,
                                 AlleleIdentity &variant_identity);
    void EvaluateFlowDistForOrdinary(HypothesisEvaluator &hypEvaluator, ExtendedReadInfo &current_read,
                                     InputStructures &global_context,
                                     const string &local_contig_sequence,
                                     FlowDist *flowDist, int variant_start_pos,
                                     AlleleIdentity &variant_identity);
    void EvaluateFlowDistForSNP(HypothesisEvaluator &hypEvaluator, ExtendedReadInfo &current_read,
                                InputStructures &global_context,
                                const string &local_contig_sequence,
                                FlowDist *flowDist, int variant_start_pos,
                                AlleleIdentity &variant_identity);
    void  ExtraEvalForOverUnderSNP(HypothesisEvaluator &hypEvaluator, ExtendedReadInfo &current_read,
                                   InputStructures &global_context,
                                   const string &local_contig_sequence,
                                   FlowDist *flowDist, int variant_start_pos,
                                   AlleleIdentity &variant_identity);
    void UpdateFlowDistFromEvaluateSoloRead(ExtendedReadInfo &current_read,
                                            InputStructures &global_context,
                                            const string &local_contig_sequence);
    void EvaluateOneFlowDist(HypothesisEvaluator &hypEvaluator, ExtendedReadInfo &current_read,
                             InputStructures &global_context,
                             const string &local_contig_sequence,
                             FlowDist *flowDist, int variant_start_pos,
                             AlleleIdentity &variant_identity);
    void ScoreFlowDistForVariant(vcf::Variant ** candidate_variant,
                                 ExtendParameters *parameters,  int DEBUG);
    void ReceiveStack(StackPlus &my_stack,
                      InputStructures &global_context,
                      const string &local_contig_sequence);
    void OutputAlleleToVariant(vcf::Variant ** candidate_variant, ExtendParameters *parameters);

};



#endif //MULTIFLOWDIST_H
