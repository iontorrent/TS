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

class BiasChecker{
public:
  vector<float> variant_bias_v;
  vector<float> ref_bias_v;
  vector<float> update_variant_bias_v;
  vector<float> update_ref_bias_v;
  vector<float> weight_variant_v;
  vector<float> weight_ref_v;
  float damper_bias;
  float soft_clip;
  BiasChecker(){
    // number of >hypotheses<
    Init(3);
  }
  void ResetUpdate();
  void Init(int num_hyp_no_null);
  void DoUpdate();
  void UpdateBiasChecker(ShortStack &my_theory);
  void AddCrossUpdate(CrossHypotheses &my_cross);
  void AddOneUpdate(HiddenBasis &delta_state, const vector<vector<float> > &residuals,
                    const vector<int> &test_flow, const vector<float> &responsibility);
};

// fixes predictions for each read of interest
class BasicBiasGenerator{
public:
  // used to predict bias
  // strand x basis vectors
  vector<vector <float> > latent_bias;
// track updates to bias
  vector<vector<float> > update_latent_bias;

  vector<vector<float> > weight_update;

// implicit prior
  float damper_bias;
  float pseudo_sigma_base;

  int DEBUG;

  BasicBiasGenerator(){
    int num_alt = 1;
   InitForStrand(num_alt);
   DEBUG = 0;
  }

  void GenerateBiasByStrand(int i_hyp, HiddenBasis &delta_state, vector<int> &test_flow, int strand_key, vector<float> &new_residuals, vector<float> &new_predictions);
  void UpdateResiduals(CrossHypotheses &my_cross);
  void ResetUpdate();
  void AddOneUpdate(HiddenBasis &delta_state, const vector<vector<float> >&residuals,
                    const vector<int> &test_flow, int strand_key, const vector<float> &responsibility);
  void AddCrossUpdate(CrossHypotheses &my_cross);
  void UpdateBiasGenerator(ShortStack &my_theory);
    // change predictions

  void UpdateResidualsFromBias(ShortStack &total_theory);
  void DoStepForBias(ShortStack &total_theory);
  void ResetActiveBias(ShortStack &total_theory);

  void DoUpdate();
  void InitForStrand(int num_alt);
  float BiasLL();
  float BiasHypothesisLL();
  float RadiusOfBias(int o_alt);
  float RadiusOfHypothesisBias();

  void PrintDebug(bool print_updated = true);
};


#endif // BIASGENERATOR_H
