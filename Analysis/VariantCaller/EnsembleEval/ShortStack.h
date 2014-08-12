/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef SHORTSTACK_H
#define SHORTSTACK_H


#include "api/BamReader.h"

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <vector>

#include "ExtendedReadInfo.h"
#include "CrossHypotheses.h"
#include "ExtendParameters.h"

class ShortStack{
  // induced theories of the world
  public:
  vector<CrossHypotheses> my_hypotheses;
  vector<int> valid_indexes;
  void FindValidIndexes(); // only loop over reads where we successfully filled in variants
  
  void FillInPredictions(PersistingThreadObjects &thread_objects, vector<const Alignment *>& read_stack, const InputStructures &global_context);
  void ResetQualities();
  void InitTestFlow();
  float PosteriorFrequencyLogLikelihood(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float my_reliability, int strand_key);
  void PropagateTuningParameters(EnsembleEvalTuningParameters &my_params);
    void ResetRelevantResiduals();
  void UpdateRelevantLikelihoods();
  void ResetNullBias();
  // do updates
  void UpdateResponsibility(const vector<float> &hyp_freq, float data_reliability);
  void MultiFrequencyFromResponsibility(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key);
};


#endif // SHORTSTACK_H
