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
#include <armadillo>


#include "ExtendedReadInfo.h"
#include "HypothesisEvaluator.h"


// use both strands for evaluating likelihood
#define ALL_STRAND_KEY -1
// no matter what, the read as called should be 1 in a million relative to the variant
// prevent some awkward moments if we divide by zero
#define MINIMUM_RELATIVE_OUTLIER_PROBABILITY 0.000001f

class PrecomputeTDistOddN{
  public:
    float v;
    float pi_factor;
    float v_factor;
    int half_n;
    PrecomputeTDistOddN(){v=pi_factor=v_factor=1.0f; half_n = 3; SetV(3);};
    void SetV(int _half_n);
    float TDistOddN(float res, float sigma, float skew);
};

class HiddenBasis{
public:
  // is this its own sub-structure?
  // extra data supporting evaluation of hypotheses
    vector< vector<float> > delta; // ref vs alt, for each alt

    arma::Mat<double> cross_cor; // relationships amongst deltas
    arma::Mat<double> cross_inv; // invert to get coefficients
    arma::Col<double>  tmp_beta;
    arma::Col<double>  tmp_synthesis;

    // in some unfortunate cases, we have dependent errors
    float delta_correlation;

    HiddenBasis();
    void Allocate(int i_hyp, int j_flow);

    float ServeDelta(int i_hyp, int j_flow);
    float ServeAltDelta(int i_alt, int j_flow);
    float ServeCommonDirection(int j_flow);

    void ComputeDelta(const vector<vector <float> > &predictions);
    void ComputeDeltaCorrelation(const vector<vector <float> > &predictions, const vector<int> &test_flow);
    //bool ComputeTestFlow(vector<int> &test_flow, float threshold, int max_choice, int max_last_flow);
   void  ComputeCross(const vector<int> &test_flow);
   void SetDeltaReturn(const vector<float> &beta);
};


// handle auxiliary variables for one read's associated hypothesis evaluation
class CrossHypotheses{
public:
  vector<string> instance_of_read_by_state;       // this read, modified by each state of a variant
  vector<vector<float> > predictions;             // Predicted signal for flows
  vector<vector<float> > normalized;              // Normalized signal for flows
  vector<int>            state_spread;
  vector<bool>           same_as_null_hypothesis; // indicates whether a ref or alt hypothesis equals the read as called

  HiddenBasis delta_state;
  bool use_correlated_likelihood;

// hold some intermediates size data matrix hyp * nFlows (should be active flows)

  vector<vector<float> > mod_predictions;
  vector<vector<float> > residuals; // difference prediction and observed
  vector<vector<float> > sigma_estimate; // estimate of variability per flow per hypothesis for this read
  vector<vector<float> > basic_likelihoods; // likelihood given residuals at each flow of the observation at that flow != likelihood of read
  
  float skew_estimate;

  vector<int > test_flow;  //  vector of flows to examine for this read and the hypotheses for efficiency
  int          start_flow; // Start flow as written in BAM <-- used in test flow computation
  
  // size number of hypotheses
  vector<float> responsibility; // how responsible this read is for a given hypothesis under the MAP: size number of hypotheses (including null=outlier)
  vector<float> log_likelihood; // sum over our test flows: logged to avoid under-flows
  vector<float> scaled_likelihood; // actual sum likelihood over test flows, rescaled to null hypothesis (as called), derived from log_likelihood
  float ll_scale; // local scaling factor for scaled likelihood as can't trust null hypothesis to be near data
  
  // intermediate allocations
  vector<float> tmp_prob_f;
  vector<double> tmp_prob_d;
  
  PrecomputeTDistOddN my_t;

  // useful hidden variables
  int strand_key;
  
  int heavy_tailed;
  int max_flows_to_test;
  float min_delta_for_flow;
  
  float magic_sigma_base;
  float magic_sigma_slope;
  
  int splice_start_flow; // Flow just before we start splicing in hypotheses (same for all hypotheses)
  int splice_end_flow;   // Flow of the first base after the variant window (maximum over all hypotheses)
  int max_last_flow;     // Last flow that is being simulated in prediction generation (max over all hypotheses)
  bool success;

// functions
  CrossHypotheses(){
    heavy_tailed = 3;  // t_5 degrees of freedom
    my_t.SetV(3);
    strand_key = 0;
    max_last_flow=0;
    splice_start_flow = -1;
    splice_end_flow = -1;
    start_flow = 0;
    max_flows_to_test = 10;
    min_delta_for_flow = 0.1f;
    skew_estimate = 1.0f;
    success = true;
    ll_scale = 0.0f;
    magic_sigma_base = 0.085f;
    magic_sigma_slope = 0.0084f;
    use_correlated_likelihood = false;
  };
  void  CleanAllocate(int num_hyp, int num_flow);
  void  SetModPredictions();
  void  FillInPrediction(PersistingThreadObjects &thread_objects, const Alignment &my_read, const InputStructures &global_context);
  void  InitializeDerivedQualities();
  void  InitializeTestFlows();
  void  ComputeBasicResiduals();
  void  ResetModPredictions();
  void  ComputeDeltaCorrelation();
  void  ResetRelevantResiduals();
  void  ComputeBasicLikelihoods();
  void  ComputeLogLikelihoods();
  void  ComputeLogLikelihoodsSum();
  void  JointLogLikelihood();
  void  ComputeScaledLikelihood();
  float ComputePosteriorLikelihood(const vector<float> &hyp_prob, float typical_prob);
  void  InitializeSigma();
  void  InitializeResponsibility();
  void  UpdateResponsibility(const vector<float > &hyp_prob, float typical_prob);
  void  UpdateRelevantLikelihoods();
  void  ComputeDelta();
  void  ComputeTestFlow();
  bool  ComputeAllComparisonsTestFlow(float threshold, int max_choice);
  float ComputeLLDifference(int a_hyp, int b_hyp);
  int   MostResponsible();
  bool  IsValidTestFlowIndexOld(unsigned int flow,unsigned int max_choice);
  bool  IsValidTestFlowIndexNew(unsigned int flow,unsigned int max_choice);
};



#endif // CROSSHYPOTHESES_H
