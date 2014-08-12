/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "ShortStack.h"



void ShortStack::PropagateTuningParameters(EnsembleEvalTuningParameters &my_params){
    for (unsigned int i_read = 0; i_read < my_hypotheses.size(); i_read++) {
      // basic likelihoods
    my_hypotheses[i_read].heavy_tailed = my_params.heavy_tailed;
    
    // test flow construction
    my_hypotheses[i_read].max_flows_to_test = my_params.max_flows_to_test;
    my_hypotheses[i_read].min_delta_for_flow = my_params.min_delta_for_flow;
    
    // used to initialize sigma-estimates
    my_hypotheses[i_read].magic_sigma_base = my_params.magic_sigma_base;
    my_hypotheses[i_read].magic_sigma_slope = my_params.magic_sigma_slope;
  }
}


// fill in predictions for each hypothesis
void ShortStack::FillInPredictions(PersistingThreadObjects &thread_objects, vector<const Alignment *>& read_stack,
    const InputStructures &global_context)
{
  //ion::FlowOrder flow_order(my_data.flow_order, my_data.flow_order.length());
  for (unsigned int i_read = 0; i_read < my_hypotheses.size(); i_read++) {
    my_hypotheses[i_read].FillInPrediction(thread_objects, *read_stack[i_read], global_context);
    my_hypotheses[i_read].start_flow = read_stack[i_read]->start_flow;
  }
}

void ShortStack::ResetQualities() {
  // ! does not reset test flows or delta (correctly)
  for (unsigned int i_read = 0; i_read < my_hypotheses.size(); i_read++) {
    my_hypotheses[i_read].InitializeDerivedQualities();
  }
}

void ShortStack::InitTestFlow() {
  // ! does not reset test flows or delta (correctly)
  for (unsigned int i_read = 0; i_read < my_hypotheses.size(); i_read++) {
    my_hypotheses[i_read].InitializeTestFlows();
  }
};


float ShortStack::PosteriorFrequencyLogLikelihood(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float my_reliability, int strand_key) {
  //cout << "eval at freq " << my_freq << endl;
  float my_LL = 0.0f;

  //for (unsigned int i_read=0; i_read<my_hypotheses.size(); i_read++){
  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    if ((strand_key < 0) || (my_hypotheses[i_read].strand_key == strand_key))
      my_LL += my_hypotheses[i_read].ComputePosteriorLikelihood(hyp_freq, my_reliability); // XXX
  }
  // add contribution from prior
  for (unsigned int i_hyp=0; i_hyp<hyp_freq.size(); i_hyp++){
    // contribution is
    float local_LL = log(hyp_freq[i_hyp]*my_reliability + (1.0f-my_reliability)); // hyp_freq might be exactly zero, whereupon my prior is an outlier
    my_LL += prior_frequency_weight[i_hyp]*local_LL; // my weight is my number of pseudo-observations with this LL
  }
  my_LL += prior_log_normalization;
  return(my_LL);
}


void ShortStack::UpdateRelevantLikelihoods() {
  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    my_hypotheses[i_read].UpdateRelevantLikelihoods();
  }
}
void ShortStack::ResetRelevantResiduals() {
  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    my_hypotheses[i_read].ResetRelevantResiduals();
  }
}


void ShortStack::ResetNullBias() {
  ResetRelevantResiduals();
  UpdateRelevantLikelihoods();
}

void ShortStack::FindValidIndexes() {
  valid_indexes.resize(0);
  // only loop over reads where variant construction worked correctly
  for (unsigned int i_read = 0; i_read < my_hypotheses.size(); i_read++) {
    if (my_hypotheses[i_read].success) {
      valid_indexes.push_back(i_read);
    }
  }
}


void ShortStack::UpdateResponsibility(const vector<float> &hyp_freq, float data_reliability) {
  //for (unsigned int i_read=0; i_read<total_theory.my_hypotheses.size(); i_read++){
  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    my_hypotheses[i_read].UpdateResponsibility(hyp_freq, data_reliability);
  }
}

void ShortStack::MultiFrequencyFromResponsibility(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key){

  hyp_freq.assign(hyp_freq.size(), 0.0f);

  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    for (unsigned int i_hyp = 0; i_hyp < hyp_freq.size(); i_hyp++) {
      if ((strand_key < 0) || (my_hypotheses[i_read].strand_key == strand_key))
        hyp_freq[i_hyp] += my_hypotheses[i_read].responsibility[i_hyp+1];
    }
  }
  // add prior weight to count of cluster
  // weight = effective number of additional pseudo-observations
  for (unsigned int i_hyp=0; i_hyp<hyp_freq.size(); i_hyp++){
    hyp_freq[i_hyp] += prior_frequency_weight[i_hyp];
  }
  // safety factor to prevent zero-frequency from preventing progress
  float safety_offset = 0.5f;
  float denom = 0.0f;
  for (unsigned int i_hyp=0; i_hyp< hyp_freq.size(); i_hyp++){
    denom += hyp_freq[i_hyp]+safety_offset;
  }

  for (unsigned int i_hyp=0; i_hyp<hyp_freq.size(); i_hyp++){
    hyp_freq[i_hyp] = (hyp_freq[i_hyp]+safety_offset)/denom;
  }
}
