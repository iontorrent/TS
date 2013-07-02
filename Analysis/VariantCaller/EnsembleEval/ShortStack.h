/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef SHORTSTACK_H
#define SHORTSTACK_H


#include "api/BamReader.h"

#include "../Analysis/file-io/ion_util.h"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <vector>

#include "ExtendedReadInfo.h"
#include "StackPlus.h"
#include "CrossHypotheses.h"
#include "ExtendParameters.h"

class ShortStack{
  // induced theories of the world
  public:
  vector<CrossHypotheses> my_hypotheses;
  vector<int> valid_indexes;
  void FindValidIndexes(); // only loop over reads where we successfully filled in variants
  
  void FillInPredictions(PersistingThreadObjects &thread_objects, StackPlus &my_data, InputStructures &global_context);
  void ResetQualities();
  void InitTestFlow();
  float PosteriorFrequencyLogLikelihood(float my_freq, float my_reliability, int strand_key);
  void PropagateTuningParameters(EnsembleEvalTuningParameters &my_params);
    void ResetRelevantResiduals();
  void UpdateRelevantLikelihoods();
  void ResetNullBias();
  // do updates
  void UpdateResponsibility(float my_freq, float data_reliability);
  float FrequencyFromResponsibility(int strand_key);
};


#endif // SHORTSTACK_H
