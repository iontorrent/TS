/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     VcfFormat.h
//! @ingroup  VariantCaller
//! @brief    Handle formatting issues for Vcf files and tags


#ifndef VCFFORMAT_H
#define VCFFORMAT_H


#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <ctype.h>
#include <algorithm>
#include <utility>

#include "sys/types.h"
#include "sys/stat.h"
#include <time.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <Variant.h>
#include "AlleleParser.h"

using namespace std;

// forward declaration
class ExtendParameters; 

string getVCFHeader(const ExtendParameters *parameters, const vector<string>& sample_list, int primary_sample = 0);
void clearInfoTags(vcf::Variant &var);
void NullInfoFields(vcf::Variant &var, bool use_position_bias);
void SetUpFormatString(vcf::Variant &var);
int CalculateWeightOfVariant(vcf::Variant &current_variant);
void ClearVal(vcf::Variant &var, const char *clear_me);
float RetrieveQualityTagValue(vcf::Variant &current_variant, const string &tag_wanted, int _allele_index);

// double-star pointer here
void SetFilteredStatus(vcf::Variant & candidate_variant, bool isFiltered);
void StoreGenotypeForOneSample(vcf::Variant & candidate_variant, const string &my_sample_name, string &my_genotype, float genotype_quality);
void NullGenotypeAllSamples(vcf::Variant & candidate_variant);
void OverwriteGenotypeForOneSample(vcf::Variant & candidate_variant, const string &my_sample_name, string &my_genotype, float genotype_quality);
void NullFilterReason(vcf::Variant &candidate_variant);
void AddFilterReason(vcf::Variant &candidate_variant, string &additional_reason);
void DetectAndSetFilteredGenotype(vcf::Variant &candidate_variant, const string &sampleName);

// generic variant manipulation
void RemoveFilteredAlleles(vcf::Variant & candidate_variant, vector<int> &filtered_allele_index);
void AdjustAlleles(vcf::Variant & candidate_variant, int position_upper_bound);


#endif //VCFFORMAT_H
