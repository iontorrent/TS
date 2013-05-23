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
#include "AlignmentAssist.h"
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

#endif // CANDIDATEVARIANTGENERATION_H
