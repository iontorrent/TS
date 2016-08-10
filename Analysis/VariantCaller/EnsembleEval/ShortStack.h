/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef SHORTSTACK_H
#define SHORTSTACK_H

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <vector>
#include <map>

#include "api/BamReader.h"
#include "ExtendedReadInfo.h"
#include "CrossHypotheses.h"
#include "ExtendParameters.h"

using namespace std;


// induced theories of the world
class ShortStack{
public:
    vector<CrossHypotheses> my_hypotheses;
    vector<int> valid_indexes;

    void FindValidIndexes(); // only loop over reads where we successfully filled in variants
    void FillInPredictionsAndTestFlows(PersistingThreadObjects &thread_objects, vector<const Alignment *>& read_stack, const InputStructures &global_context);
    void ResetQualities();
    void OutlierFiltering(float data_reliability, bool is_update_valid_index);
    void PropagateTuningParameters(EnsembleEvalTuningParameters &my_params);
    void ResetRelevantResiduals();
    void UpdateRelevantLikelihoods();
    void ResetNullBias();
    unsigned int DetailLevel(void);

    // family stuffs for mol tag
    vector<EvalFamily> my_eval_families;
    bool IsEmptyFuncFam() const { return (num_func_families_ == 0); };
    void ResetQualitiesForFamilies();
    void SetFuncionalityForFamilies(unsigned int min_fam_size);
    unsigned int GetNumFuncFamilies() const { return num_func_families_; };

    // functions need to be switched for using/(not using) molecular tags
    float PosteriorFrequencyLogLikelihood(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float my_reliability, int strand_key);
    void MultiFrequencyFromResponsibility(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key);
    void UpdateResponsibility(const vector<float> &hyp_freq, float data_reliability);
    // functions for no molecular tags only
    float PosteriorFrequencyLogLikelihoodNoMolTags(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float my_reliability, int strand_key);
    void MultiFrequencyFromResponsibilityNoMolTags(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key);
    void UpdateResponsibilityNoMolTags(const vector<float> &hyp_freq, float data_reliability);
    // functions for molecular tags only
    float PosteriorFrequencyLogLikelihoodMolTags(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float my_reliability, int strand_key);
    void MultiFrequencyFromResponsibilityMolTags(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key);
    void UpdateResponsibilityMolTags(const vector<float> &hyp_freq, float data_reliability);

    // Am I using or not using molecular tags?
    void SetIsMolecularTag(bool is_mol_tag);
    bool GetIsMolecularTag() const { return is_molecular_tag_; };

private:
    // is_molecular_tag_ is the flag that indicates whether we are using molecular tag or not
    // true if we are using molecular tag or not, false if not.
    // Use SetIsMolecularTag(bool) to set the value of is_molecular_tag_ to guarantee that
    // the function pointers depending on is_molecular_tag_ are pointing to the right functions.
    bool is_molecular_tag_ = false;

    // family counts
    unsigned int num_func_families_ = 0;

    // function pointers for mol_tag or non-mol_tag
    float (ShortStack::*ptrPosteriorFrequencyLogLikelihood_)(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float my_reliability, int strand_key) = NULL;
    void (ShortStack::*ptrUpdateResponsibility_)(const vector<float> &hyp_freq, float data_reliability) = NULL;
    void (ShortStack::*ptrMultiFrequencyFromResponsibility_)(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key) = NULL;
    // switch the function pointers that depend on is_molecular_tag_
    void SwitchMolTagsPtr_(void);

    // For mol tags use only
    void UpdateFamilyResponsibility_(const vector<float> &hyp_freq, float data_reliability);
    void MultiFrequencyFromFamilyResponsibility_(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key);
    void UpdateReadResponsibilityFromFamily_(unsigned int num_hyp_no_null, float data_reliability);
    float PosteriorFrequencyFamilyLogLikelihood_(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float my_reliability, int strand_key);
};

#endif // SHORTSTACK_H
