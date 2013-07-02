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
void ShortStack::FillInPredictions(PersistingThreadObjects &thread_objects, StackPlus &my_data, InputStructures &global_context) {
  //ion::FlowOrder flow_order(my_data.flow_order, my_data.flow_order.length());
  for (unsigned int i_read = 0; i_read < my_hypotheses.size(); i_read++) {
    my_hypotheses[i_read].FillInPrediction(thread_objects, my_data.read_stack[i_read], global_context);
  }
}

void ShortStack::ResetQualities() {
  // ! does not reset test flows or delta (correctly)
  for (unsigned int i_read = 0; i_read < my_hypotheses.size(); i_read++) {
    my_hypotheses[i_read].InitializeDerivedQualities();
  }
};

void ShortStack::InitTestFlow() {
  // ! does not reset test flows or delta (correctly)
  for (unsigned int i_read = 0; i_read < my_hypotheses.size(); i_read++) {
    my_hypotheses[i_read].InitializeTestFlows();
  }
};


float ShortStack::PosteriorFrequencyLogLikelihood(float my_freq, float my_reliability, int strand_key) {
  //cout << "eval at freq " << my_freq << endl;
  float my_LL = 0.0f;

  //for (unsigned int i_read=0; i_read<my_hypotheses.size(); i_read++){
  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    if ((strand_key < 0) || (my_hypotheses[i_read].strand_key == strand_key))
      my_LL += my_hypotheses[i_read].ComputePosteriorLikelihood(my_freq, my_reliability);
  }
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


void ShortStack::UpdateResponsibility(float my_freq, float data_reliability) {
  //for (unsigned int i_read=0; i_read<total_theory.my_hypotheses.size(); i_read++){
  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    my_hypotheses[i_read].UpdateResponsibility(my_freq, data_reliability);
  }
}

float ShortStack::FrequencyFromResponsibility(int strand_key){
  vector<float> all_response(3);
  all_response.assign(3, 0.0f);
  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    for (unsigned int i_hyp = 0; i_hyp < all_response.size(); i_hyp++) {
      if ((strand_key < 0) || (my_hypotheses[i_read].strand_key == strand_key))
        all_response[i_hyp] += my_hypotheses[i_read].responsibility[i_hyp];
    }
  }
  float max_freq = (all_response[1] + 0.5f) / (all_response[2] + all_response[1] + 1.0f); // safety frequency -> go to 50% if no data, also never go to zero or 1 to stall progress
  
  return(max_freq);
}
