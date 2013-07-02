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
#include "Utility.h"

using namespace std;

// forward declaration
class CandidateGenerationHelper;
class ExtendParameters; 

string getVCFHeader(ExtendParameters *parameters, CandidateGenerationHelper &candidate_generator);
void clearInfoTags(vcf::Variant *var);
void NullInfoFields(vcf::Variant *var);
void SetUpFormatString(vcf::Variant *var);
int CalculateWeightOfVariant(vcf::Variant *current_variant);
void ClearVal(vcf::Variant *var, const char *clear_me);
float RetrieveQualityTagValue(vcf::Variant *current_variant, const string &tag_wanted, int _allele_index);

#endif //VCFFORMAT_H
