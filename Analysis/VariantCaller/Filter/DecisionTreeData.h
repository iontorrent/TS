/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef DECISIONTREEDATA_H
#define DECISIONTREEDATA_H


#include "api/BamReader.h"

#include "../Analysis/file-io/ion_util.h"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <vector>

#include "ClassifyVariant.h"
#include "StackEngine.h"
#include "VcfFormat.h"

// all the data needed to make a decision for filtration
// characterize the variant, and the outcome from whatever evaluator we use
class DecisionTreeData {
  public:
    //vector<WhatVariantAmI> variant_identity_vector;
    MultiAlleleVariantIdentity multi_allele;

    vector<VariantBook> summary_stats_vector;

    vector<VariantOutputInfo> summary_info_vector;

    vector<int> filteredAllelesIndex;


    bool best_variant_filtered;
    string best_filter_reason;
    bool best_allele_set;
    int best_allele_index;
    bool isBestAlleleSNP;

    float tune_xbias;
    float tune_sbias;

    DecisionTreeData() {
      //quasi_phred_quality_score = 0.0f;
      //reject_status_quality_score = 0.0f;
      //variant_filtered = false;
      //filter_reason="HAPPY";
      //genotype_call = 0;
      best_allele_set = false;
      best_allele_index = 0;
      best_variant_filtered=false;
      isBestAlleleSNP = false;
      tune_xbias = 0.005f; // tune calculation of chi-square bias = proportioinal variance by frequency
      tune_sbias = 0.5f; // safety factor for small allele counts for transformed strand bias
    };

    void FilterReferenceCalls(int _allele);
    void FilterOnStrandBias(float threshold, int _allele);
    void DoFilter(ControlCallAndFilters &my_filters, int _allele);
    void FilterOnMinimumCoverage(int min_cov_each_strand,  int _allele);
    void FilterOnQualityScore(float min_quality_score, int _allele);
    void OverrideFilter(string & _filter_reason, int _allele);
    void FilterNoCalls(bool isNoCall, int _allele);
    void RemoveFilteredAlleles(vcf::Variant ** candidate_variant, string &sample_name);
    void FilterOneAllele(VariantBook &l_summary_stats,
                         VariantOutputInfo &l_summary_info,
                         AlleleIdentity &l_variant_identity, ControlCallAndFilters &my_filters);
    void FilterAlleles(ControlCallAndFilters &my_filters);

    void FindBestAllele();
    void AccumulateFilteredAlleles();
    string AnyNoCallsMeansAllFiltered();
    void DetectBestAlleleFiltered(string &filter_reason);
    void DetectAllFiltered();
    void BestSNPsSuppressInDels();
    void FindBestAlleleIdentity();
    void FindBestAlleleByScore();

    void StoreMaximumAlleleInVariants(vcf::Variant ** candidate_variant, ExtendParameters *parameters);
    bool SetGenotype(vcf::Variant ** candidate_variant, ExtendParameters *parameters, float gt_quality);
    void DecisionTreeOutputToVariant(vcf::Variant ** candidate_variant,ExtendParameters *parameters);
    void SetupSummaryStatsFromCandidate(vcf::Variant **candidate_variant);
    void SetupFromMultiAllele(MultiAlleleVariantIdentity &_multi_allele);
    void SetLocalGenotypeCallFromStats(float threshold);
    void  InformationTagOnFilter(vcf::Variant ** candidate_variant, int _best_allele_index, string sampleName);
    string GenotypeStringFromAlleles(std::vector<int> &allowedGenotypes, bool refAlleleFound);
    bool AllowedGenotypesFromSummary(std::vector<int> &allowedGenotypes);
    string GenotypeFromStatus(vcf::Variant **candidate_variant, ExtendParameters *parameters);
    void SpecializedFilterFromLatentVariables(vcf::Variant ** candidate_variant,  float bias_radius, int _allele);
    void SpecializedFilterFromHypothesisBias(vcf::Variant ** candidate_variant, AlleleIdentity allele_identity, float deletion_bias, float insertion_bias, int _allele);
    void FilterAlleleHypothesisBias(float ref_bias, float var_bias, float threshold_bias, int _allele);
    void FilterOnSpecialTags(vcf::Variant ** candidate_variant, ExtendParameters *parameters);
};

void AdjustAlleles(vcf::Variant ** candidate_variant);
void FilterByBasicThresholds(stringstream &s, VariantBook &l_summary_stats,
                             VariantOutputInfo &l_summary_info,
                             BasicFilters &basic_filter, float tune_xbias, float tune_bias);

void AutoFailTheCandidate(vcf::Variant **candidate_variant, bool suppress_no_calls);
float FreqThresholdByType(AlleleIdentity &variant_identity, ControlCallAndFilters &my_controls);
void FilterOnInformationTag(vcf::Variant **candidate_variant, float data_quality_stringency, bool suppress_no_calls, int _check_allele_index, string sampleName);

#endif // DECISIONTREEDATA_H
