/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef SIGMAGENERATOR_H
#define SIGMAGENERATOR_H


#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include "CrossHypotheses.h"
#include "ShortStack.h"


// fixes sigma values for each hypothesis of interest
class BasicSigmaGenerator{
public:
   // used to predict sigma
   vector<float> latent_sigma;
   // revision: intensity linear interpolation
   vector<float> accumulated_sigma;
   vector<float> accumulated_weight;
   // basic model
   vector<float> prior_sigma_regression;
   vector<float> prior_latent_sigma;
   int max_level;
   
   float prior_weight;
   float k_zero;
   
   BasicSigmaGenerator(){
      max_level = 29;
      prior_weight = 1.0f;
      k_zero = 0.0f;
      InitSigma();
      ResetUpdate();
   };
   void InitSigma(){
      latent_sigma.assign(max_level+1, 0.0f); // 0-29
      accumulated_sigma.assign(latent_sigma.size(), 0.0f);
      accumulated_weight.assign(latent_sigma.size(), 0.0f);
      prior_latent_sigma.assign(latent_sigma.size(), 0.0f);
      prior_sigma_regression.assign(2, 0.0f);
      prior_sigma_regression[0] = 0.085f;
      prior_sigma_regression[1] = 0.0084f;
   };
   void SimplePrior();
   void ZeroAccumulator();
   void PushLatent(float responsibility,float x_val, float y_val, bool do_weight);
   float InterpolateSigma(float x_val);
   void ResetUpdate();
   void GenerateSigmaByRegression(vector<float> &prediction, vector<int> &test_flow, vector<float> &sigma_estimate);
   void GenerateSigma(CrossHypotheses &my_cross);
   void AddCrossUpdate(CrossHypotheses &my_cross);
   void AddShiftCrossUpdate(CrossHypotheses &my_cross, float discount);
   void AddNullUpdate(CrossHypotheses &my_cross);
   void AddOneUpdateForHypothesis(vector<float> &prediction, float responsibility, float skew_estimate, vector<int> &test_flow, vector<float> &residuals, vector<float> &measurements_var);
   void DoLatentUpdate();
   float RetrieveApproximateWeight(float x_val);
   void PushToPrior();
   void PopFromLatentPrior();
   void AddShiftUpdateForHypothesis(vector<float> &prediction, vector<float> &mod_prediction, 
                                                      float discount, float responsibility, float skew_estimate, vector<int> &test_flow);
     void NullUpdateSigmaGenerator(ShortStack &total_theory);
  void UpdateSigmaGenerator(ShortStack &total_theory);
  void UpdateSigmaEstimates(ShortStack &total_theory);
  void DoStepForSigma(ShortStack &total_theory);
 
};


class StrandedSigmaGenerator{
public:
  BasicSigmaGenerator fwd,rev;
  // if want to revert to old-style
  bool combine_strands;
  int DEBUG = 0;
  StrandedSigmaGenerator(){
	  combine_strands = false;
      DEBUG = 0;
  };

  void UpdateSigmaGenerator(ShortStack &total_theory);
  void UpdateSigmaEstimates(ShortStack &total_theory);
  void DoStepForSigma(ShortStack &total_theory);
  void ResetSigmaGenerator();
  void PrintDebug(bool print_update = true);
};



#endif // SIGMAGENERATOR_H
