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
	int num_hyp_not_null = 0;
    vector<CrossHypotheses> my_hypotheses;
    vector<int> valid_indexes;
    bool preserve_full_data = false; // preserve information of all flows in the evaluator?
    bool DEBUG = false;

    void FindValidIndexes(); // only loop over reads where we successfully filled in variants
    void FillInPredictionsAndTestFlows(PersistingThreadObjects &thread_objects, vector<const Alignment *>& read_stack, const InputStructures &global_context);
    void ResetQualities(float outlier_probability);
    void PropagateTuningParameters(EnsembleEvalTuningParameters &my_params);
    void ResetRelevantResiduals();
    void UpdateRelevantLikelihoods();
    void ResetNullBias();
    unsigned int DetailLevel(void);

    // Flow-Disruptiveness
    int OutlierCountsByFlowDisruptiveness();
    void FlowDisruptiveOutlierFiltering(bool update_valid_index);

    // family related variables and functions
    vector<EvalFamily> my_eval_families;
    unsigned int effective_min_family_size = 0;
    unsigned int GetNumFuncFamilies() const { return num_func_families_; };
    void InitializeMyEvalFamilies(unsigned int num_hyp);
    void ResetQualitiesForFamilies();

    // functions need to be switched for using/(not using) molecular tags
    float PosteriorFrequencyLogLikelihood(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float outlier_prob, int strand_key);
    void MultiFrequencyFromResponsibility(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key);
    void UpdateResponsibility(const vector<float> &hyp_freq, float outlier_prob);

    // Am I using or not using molecular tags?
    void SetIsMolecularTag(bool is_mol_tag);
    bool GetIsMolecularTag() const { return is_molecular_tag_; };

private:
    // Very important: Must call SetIsMolecularTag(bool is_mol_tag) to set the value of is_molecular_tag_!
    bool is_molecular_tag_ = false;
    // functional family counts
    unsigned int num_func_families_ = 0;
    int num_valid_read_counts_ = 0;

    // function pointers for using or not using molecular tag
    float (ShortStack::*ptrPosteriorFrequencyLogLikelihood_)(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float outlier_prob, int strand_key) = NULL;
    void (ShortStack::*ptrUpdateResponsibility_)(const vector<float> &hyp_freq, float outlier_prob) = NULL;
    void (ShortStack::*ptrMultiFrequencyFromResponsibility_)(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key) = NULL;

    // switch the function pointers that depend on is_molecular_tag_
    void SwitchMolTagsPtr_(void);

    // functions for doing inference for the allele frequency from reads
    float PosteriorFrequencyLogLikelihoodFromReads_(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float outlier_prob, int strand_key);
    void MultiFrequencyFromReadResponsibility_(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key);
    void UpdateReadResponsibility_(const vector<float> &hyp_freq, float outlier_prob);
    // functions for doing inference for the allele frequency from families
    float PosteriorFrequencyLogLikelihoodFromFamilies_(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float outlier_prob, int strand_key);
    void MultiFrequencyFromFamilyResponsibility_(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key);
    void UpdateFamilyAndReadResponsibility_(const vector<float> &hyp_freq, float outlier_prob);
    void UpdateFamilyResponsibility_(const vector<float> &hyp_freq, float outlier_prob);
    void UpdateReadResponsibilityFromFamily_(unsigned int num_hyp_no_null, float outlier_prob);
};

#endif // SHORTSTACK_H
