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

class EvaluatedGenotype{
public:
  bool genotype_already_set;
  float evaluated_genotype_quality;
  float evaluated_variant_quality;
  vector<int> genotype_component;

  EvaluatedGenotype(){
    genotype_already_set = false;
    evaluated_genotype_quality = 0.0f;
    evaluated_variant_quality = 0.0f;
    genotype_component.assign(2,0); // 0/0 = reference call

  };
  string GenotypeAsString();
  bool IsReference();
};

// all the data needed to make a decision for filtration
// characterize the variant, and the outcome from whatever evaluator we use
class DecisionTreeData {
  public:

    MultiAlleleVariantIdentity multi_allele;

    MultiBook all_summary_stats;

    vector<VariantOutputInfo> summary_info_vector;

    vector<int> filteredAllelesIndex;


    bool best_variant_filtered;

    bool best_allele_set;
    int best_allele_index;
    bool isBestAlleleSNP;
    bool reference_genotype;

    EvaluatedGenotype eval_genotype;

    float tune_xbias; // not tuned, removed from filters
    float tune_sbias;

    DecisionTreeData() {

      best_allele_set = false;
      best_allele_index = 0;
      best_variant_filtered=false;
      isBestAlleleSNP = false;
      reference_genotype = false;


      tune_xbias = 0.005f; // tune calculation of chi-square bias = proportioinal variance by frequency
      tune_sbias = 0.5f; // safety factor for small allele counts for transformed strand bias
    };

    void OverrideFilter(string & _filter_reason, int _allele);
    void FilterOneAllele(int i_alt,
                         VariantOutputInfo &l_summary_info,
                         AlleleIdentity &l_variant_identity, ControlCallAndFilters &my_filters);
    void FilterAlleles(ControlCallAndFilters &my_filters);

    void AccumulateFilteredAlleles();

    void BestSNPsSuppressInDels(bool heal_snps);
    void FindBestAlleleIdentity();
    void FindBestAlleleByScore();

    void GenotypeFromBestAlleleIndex(vcf::Variant ** candidate_variant, ExtendParameters *parameters);
    void GenotypeFromEvaluator(vcf::Variant ** candidate_variant, ExtendParameters *parameters);

    void FilterMyCandidate(vcf::Variant ** candidate_variant, ExtendParameters *parameters);
    void BestAlleleFilterMyCandidate(vcf::Variant ** candidate_variant, ExtendParameters *parameters);
    void GenotypeAlleleFilterMyCandidate(vcf::Variant ** candidate_variant, ExtendParameters *parameters);

    void SimplifySNPsIfNeeded(vcf::Variant ** candidate_variant, ExtendParameters *parameters);


    bool SetGenotype(vcf::Variant ** candidate_variant, ExtendParameters *parameters, float gt_quality);
    void DecisionTreeOutputToVariant(vcf::Variant ** candidate_variant,ExtendParameters *parameters);

    void AggregateFilterInformation(vcf::Variant ** candidate_variant,ExtendParameters *parameters);
    void FillInFiltersAtEnd(vcf::Variant ** candidate_variant,ExtendParameters *parameters);



    void SetupFromMultiAllele(MultiAlleleVariantIdentity &_multi_allele);
    void AddStrandBiasTags(vcf::Variant **candidate_variant);
    void  AddCountInformationTags(vcf::Variant ** candidate_variant, string &sampleName);

    string GenotypeStringFromAlleles(std::vector<int> &allowedGenotypes, bool refAlleleFound);
    bool AllowedGenotypesFromSummary(std::vector<int> &allowedGenotypes);
    string GenotypeFromStatus(vcf::Variant **candidate_variant, ExtendParameters *parameters);
    void SpecializedFilterFromLatentVariables(vcf::Variant ** candidate_variant,  float bias_radius, int _allele);
    void SpecializedFilterFromHypothesisBias(vcf::Variant ** candidate_variant, AlleleIdentity allele_identity, float deletion_bias, float insertion_bias, int _allele);
    void FilterAlleleHypothesisBias(float ref_bias, float var_bias, float threshold_bias, int _allele);
    void FilterOnSpecialTags(vcf::Variant ** candidate_variant, ExtendParameters *parameters);
    void FilterOnStringency(vcf::Variant **candidate_variant, float data_quality_stringency,  int _check_allele_index);
    void FilterSSE(vcf::Variant **candidate_variant,ClassifyFilters &filter_variant);
};
void FilterByBasicThresholds(stringstream &s, int i_alt, MultiBook &m_summary_stats,
                             VariantOutputInfo &l_summary_info,
                             BasicFilters &basic_filter, float tune_xbias, float tune_bias);

void AutoFailTheCandidate(vcf::Variant **candidate_variant, bool suppress_no_calls);
float FreqThresholdByType(AlleleIdentity &variant_identity, ControlCallAndFilters &my_controls);
void DetectSSEForNoCall(AlleleIdentity &var_identity, float sseProbThreshold, float minRatioReadsOnNonErrorStrand, float relative_safety_level, vcf::Variant **candidate_variant, unsigned _altAlleIndex);
void SetQualityByDepth(vcf::Variant ** candidate_variant);

#endif // DECISIONTREEDATA_H
