/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     VariantAssist.h
//! @ingroup  VariantCaller
//! @brief    Utilities for output of variants


#ifndef VARIANTASSIST_H
#define VARIANTASSIST_H

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

#include <Variant.h>
#include "MiscUtil.h"
// ugly, too many headers
#include "ExtendParameters.h"

using namespace std;
using namespace BamTools;

class VariantBook {
  public:
    uint16_t depth;
    uint16_t plusDepth;
    long int plusMean;
    long int negMean;
    uint16_t plusVariant;
    uint16_t negVariant;
    float strandBias;
    bool isAltAlleleFreqSet;
    float altAlleleFreq;
    uint16_t plusBaseVariant;
    uint16_t negBaseVariant;
    uint16_t plusBaseDepth;
    uint16_t negBaseDepth;


    bool DEBUG;

    VariantBook() {
      depth = 0;
      plusDepth = 0;
      plusMean = 0;
      negMean = 0;
      plusVariant = 0;
      negVariant = 0;
      isAltAlleleFreqSet = false;
      altAlleleFreq = 0.0;
      strandBias = 0.0f;
      DEBUG=false;
      plusBaseVariant = 0;
      negBaseVariant = 0;
      plusBaseDepth = 0;
      negBaseDepth = 0;
    };

    float getAlleleFreq();
    float getAltAlleleFreq();
    void setAltAlleleFreq(float altFreq);
    float getRefStrandBias();
    float getStrandBias();
    uint16_t getNegVariant();
    void incrementNegVariant();
    uint16_t getPlusVariant();
    void incrementPlusVariant();
    
    void setPlusDepth(uint16_t dep);
    void incrementPlusDepth();
    void setDepth(uint16_t dep);
    void incrementDepth();
    uint16_t getNegDepth();
    uint16_t getPlusDepth();
    uint16_t getDepth();
    uint16_t getRefAllele();
    uint16_t getVarAllele();
    uint16_t getPlusRef();
    uint16_t getNegRef();

    uint16_t getPlusBaseVariant();
    uint16_t getNegBaseVariant();
    uint16_t getPlusBaseDepth();
    uint16_t getNegBaseDepth();
    void setBasePlusVariant(uint16_t);
    void setBaseNegVariant(uint16_t);
    void setBasePlusDepth(uint16_t);
    void setBaseNegDepth(uint16_t);
    float getBaseStrandBias();
    float GetXBias(float var_zero);
    double getPlusMean();
    void incrementNegMean(int flowValue);
    void incrementPlusMean(int flowValue);
    double getNegMean();
    void UpdateSummaryStats(bool strand, bool variant_evidence, int tracking_val);
    int StatsCallGenotype(float threshold);
};


class VariantOutputInfo {
  public:
    bool isFiltered;
    float alleleScore;
    float gt_quality_score;
    int genotype_call;
    string filterReason;
    string infoString;


    VariantOutputInfo() {
      isFiltered = false;
      alleleScore = 0.0;
      gt_quality_score = 0.0f;
      genotype_call = -1;
      filterReason = "";
      infoString = "";

    }
};


bool PrefilterSummaryStats(VariantBook &summary_stats, ControlCallAndFilters &my_controls, bool *isFiltered, string *filterReason, stringstream &infoss);
void InsertBayesianScoreTag(vcf::Variant ** candidate_variant, float BayesianScore);
void InsertGenericInfoTag(vcf::Variant ** candidate_variant, bool nocall, string &nocallReason, string &infoss);
void SetFilteredStatus(vcf::Variant ** candidate_variant, bool isNoCall, bool isFiltered, bool suppress_no_calls);
void StoreGenotypeForOneSample(vcf::Variant ** candidate_variant, bool isNoCall, string &my_sample_name, string &my_genotype, float genotype_quality);
void NullGenotypeAllSamples(vcf::Variant ** candidate_variant);

#endif //VARIANTASSIST_H
