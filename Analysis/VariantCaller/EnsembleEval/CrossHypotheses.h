/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef CROSSHYPOTHESES_H
#define CROSSHYPOTHESES_H


#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <vector>

#include "ExtendedReadInfo.h"
#include "HypothesisEvaluator.h"

// use both strands for evaluating likelihood
#define ALL_STRAND_KEY -1
// no matter what, the read as called should be 1 in a million relative to the variant
// prevent some awkward moments if we divide by zero
#define MINIMUM_RELATIVE_OUTLIER_PROBABILITY 0.000001f


// handle auxiliary variables for one read's associated hypothesis evaluation
class CrossHypotheses{
public:
  vector<string> instance_of_read_by_state;  // this read, modified by each state of a variant
	vector<vector<float> > predictions;
	vector<vector<float> > normalized;

// is this its own sub-structure?
// extra data supporting evaluation of hypotheses
  vector<float> delta;

// hold some intermediates size data matrix hyp * nFlows (should be active flows)

  vector<vector<float> > residuals; // difference prediction and observed
  vector<vector<float> > sigma_estimate; // estimate of variability per flow per hypothesis for this read
  vector<vector<float> > basic_likelihoods; // likelihood given residuals at each flow of the observation at that flow != likelihood of read
  
  float skew_estimate;

  vector<int > test_flow; //  vector of flows to examine for this read and the hypotheses for efficiency
  
  // size number of hypotheses
  vector<float> responsibility; // how responsible this read is for a given hypothesis under the MAP: size number of hypotheses (including null=outlier)
  vector<float> log_likelihood; // sum over our test flows: logged to avoid under-flows
  vector<float> scaled_likelihood; // actual sum likelihood over test flows, rescaled to null hypothesis (as called), derived from log_likelihood
  float ll_scale; // local scaling factor for scaled likelihood as can't trust null hypothesis to be near data
  
  // intermediate allocations
  vector<float> tmp_prob_f;
  vector<double> tmp_prob_d;

// useful hidden variables
  int strand_key;
  
  int heavy_tailed;
  int max_flows_to_test;
  float min_delta_for_flow;
  
  float magic_sigma_base;
  float magic_sigma_slope;
  
  int max_last_flow;
  bool success;

// functions
  CrossHypotheses(){
    heavy_tailed = 3;  // t_5 degrees of freedom
    strand_key = 0;
    max_last_flow=0;
    max_flows_to_test = 10;
    min_delta_for_flow = 0.1f;
    skew_estimate = 1.0f;
    success = true;
    ll_scale = 0.0f;
    magic_sigma_base = 0.085f;
    magic_sigma_slope = 0.0084f;

  };
  void CleanAllocate(int num_hyp, int num_flow);
  void FillInPrediction(ion::FlowOrder &flow_order, ExtendedReadInfo &my_read);
  void InitializeDerivedQualities();
  void InitializeTestFlows();
  void ComputeBasicResiduals();
  void ResetRelevantResiduals();
  void ComputeBasicLikelihoods();
  void ComputeLogLikelihoods();
  void ComputeScaledLikelihood();
  float ComputePosteriorLikelihood(float reference_prob, float typical_prob);
  void InitializeSigma();
  void InitializeResponsibility();
  void InitializeTmpVariables();
  void UpdateResponsibility(float reference_prob, float typical_prob);
  void UpdateRelevantLikelihoods();
  void ComputeDelta();
  void ComputeTestFlow();
  void ExtendedComputeTestFlow(float threshold, int max_choice);
  void ComputeLocalDiscriminationStrength(float threshold, float &max_fld, int &reinforcing_flows);
  float ComputeLLDifference();
};



#endif // CROSSHYPOTHESES_H
