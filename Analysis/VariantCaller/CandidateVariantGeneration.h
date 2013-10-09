/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     CandidateVariantGeneration.h
//! @ingroup  VariantCaller
//! @brief    Generate new potential variants.

#ifndef CANDIDATEVARIANTGENERATION_H
#define CANDIDATEVARIANTGENERATION_H


#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <ctype.h>
#include <algorithm>
#include <utility>
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
#include <Fasta.h>
#include <Allele.h>
#include <Sample.h>
#include <AlleleParser.h>
#include <Utility.h>
#include <multichoose.h>
#include <multipermute.h>
#include <Genotype.h>
#include <ResultData.h>


#include "stats.h"

#include "InputStructures.h"
#include "HandleVariant.h"
#include "ExtendParameters.h"
#include "MiscUtil.h"
#include "VcfFormat.h"

using namespace std;
using namespace BamTools;
using namespace ion;


class CandidateGenerationHelper{
  public:
    Parameters freeParameters;
    AlleleParser *parser;
    Samples samples;
    int allowedAlleleTypes;
    
    ~CandidateGenerationHelper();
    CandidateGenerationHelper(){
      allowedAlleleTypes = 0;
      parser = NULL;
    };
    
    void SetupCandidateGeneration(InputStructures &global_context, ExtendParameters *parameters);
    void SetupSamples(ExtendParameters *parameters, InputStructures &global_context);
    void SetAllowedAlleles(ExtendParameters *parameters);
};

void clearInfoTags(vcf::Variant *variant);
bool generateCandidateVariant(AlleleParser * parser, Samples &samples, vcf::Variant * var,  bool &isHotSpot, ExtendParameters * parameters, int allowedAlleleTypes);
bool fillInHotSpotVariant(AlleleParser * parser, Samples &samples, vcf::Variant * var, vcf::Variant hsVar);

// motif: return one variant, or tell me to give up
class CandidateStreamWrapper {
public:
    CandidateGenerationHelper candidate_generator;

    // overly complex state
    bool checkHotSpotsSpanningHaploBases;
    size_t ith_variant;
    bool doing_hotspot;
    bool doing_generated;

    CandidateStreamWrapper() {
        ith_variant = 0;
        checkHotSpotsSpanningHaploBases = false;
        doing_hotspot = false;
        doing_generated = false;
    };
    void SetUp(ofstream &outVCFFile, ofstream &filterVCFFile,  InputStructures &global_context, ExtendParameters *parameters);
    bool ReturnNextHotSpot(bool &isHotSpot, vcf::Variant *variant);
    bool ReturnNextGeneratedVariant(bool &isHotSpot, vcf::Variant *variant, ExtendParameters *parameters);
    bool ReturnNextLocalVariant(bool &isHotSpot, vcf::Variant *variant, ExtendParameters *parameters);
    bool ReturnNextVariantStream(bool &isHotSpot, vcf::Variant *variant, ExtendParameters *parameters);
    void ResetForNextPosition();
};

#endif // CANDIDATEVARIANTGENERATION_H
