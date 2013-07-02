/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef BIASGENERATOR_H
#define BIASGENERATOR_H

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>

#include "CrossHypotheses.h"
#include "ShortStack.h"

// fixes predictions for each read of interest
class BasicBiasGenerator{
public:
  // used to predict bias
  vector<float> latent_bias;
  vector<float> latent_bias_v;
// track updates to bias
  vector<float> update_latent_bias;
  vector<float> weight_update;
  vector<float> update_latent_bias_v;
  vector<float> weight_update_v;
// implicit prior
  float damper_bias;
  float pseudo_sigma_base;

  BasicBiasGenerator(){
   InitForStrand();
  }
 
  void GenerateBiasByStrand(vector<float> &delta, vector<int> &test_flow, int strand_key, vector<float> &new_residuals, vector<float> &new_predictions);
  void UpdateResiduals(CrossHypotheses &my_cross);
  void ResetUpdate();
  void AddOneUpdate(vector<float> &delta, vector<vector<float> >&residuals, vector<int> &test_flow, int strand_key, vector<float> &responsibility);
  void AddCrossUpdate(CrossHypotheses &my_cross);
  void UpdateBiasGenerator(ShortStack &my_theory);
    // change predictions

  void UpdateResidualsFromBias(ShortStack &total_theory);
  void DoStepForBias(ShortStack &total_theory);
  void ResetActiveBias(ShortStack &total_theory);

  void DoUpdate();
  void InitForStrand();
  float BiasLL();
  float BiasHypothesisLL();
  float RadiusOfBias();
  float RadiusOfHypothesisBias();
  float LikelihoodOfRadius(float radius);
};


#endif // BIASGENERATOR_H
