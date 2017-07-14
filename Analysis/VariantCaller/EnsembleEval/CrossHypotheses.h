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
#include "MolecularTag.h"

// use both strands for evaluating likelihood
#define ALL_STRAND_KEY -1
// no matter what, the read as called should be 1 in a million relative to the variant
// prevent some awkward moments if we divide by zero
#define MINIMUM_RELATIVE_OUTLIER_PROBABILITY 0.000001f

class PrecomputeTDistOddN{
  public:
    float v;
    float log_v;
    float pi_factor;
    float v_factor;
    float log_factor;
    int half_n;
    PrecomputeTDistOddN(){v=pi_factor=v_factor=log_factor=log_v=1.0f; half_n = 3; SetV(3);};
    void SetV(int _half_n);
    float TDistOddN(float res, float sigma, float skew);
    float LogTDistOddN(float res, float sigma, float skew);

};

class HiddenBasis{
public:
  // is this its own sub-structure?
  // extra data supporting evaluation of hypotheses
    vector< vector<float> > delta; // ref vs alt at test flows, for each alt

    arma::Mat<double> cross_cor; // relationships amongst deltas
    arma::Mat<double> cross_inv; // invert to get coefficients
    arma::Col<double>  tmp_beta;
    arma::Col<double>  tmp_synthesis;

    // in some unfortunate cases, we have dependent errors
    float delta_correlation;

    HiddenBasis();
    void Allocate(unsigned int i_hyp, unsigned int t_flow);
    float ServeDelta(int i_hyp, int t_flow);
    float ServeAltDelta(int i_alt, int t_flow);
    float ServeCommonDirection(int t_flow);

    void ComputeDelta(const vector<vector <float> > &predictions);
    void ComputeDeltaCorrelation(const vector<vector <float> > &predictions, const vector<int> &test_flow);
    //bool ComputeTestFlow(vector<int> &test_flow, float threshold, int max_choice, int max_last_flow);
   void  ComputeCross();
   void SetDeltaReturn(const vector<float> &beta);
};


// handle auxiliary variables for one read's associated hypothesis evaluation
class CrossHypotheses{
public:
  vector<string>         instance_of_read_by_state;       // this read, modified by each state of a variant
  vector<vector<float> >  predictions;             // Predicted signal for test flows
  vector<float>          normalized;                       // Normalized signal for test flows, it is the same for all hypotheses
  vector<int>            state_spread;
  vector<bool>           same_as_null_hypothesis; // indicates whether a ref or alt hypothesis equals the read as called
  vector<float>          measurement_var;          // measurements var for a consensus read

  // keep the data at all flows here if preserve_full_data == true
  vector<vector<float> > predictions_all_flows;
  vector<float> normalized_all_flows;
  vector<float> measurement_sd_all_flows;

  HiddenBasis delta_state;
  bool use_correlated_likelihood;

// hold some intermediates size data matrix hyp * nFlows (should be active flows)

  vector<vector<float> > mod_predictions;
  vector<vector<float> > residuals; // difference prediction and observed

  vector<vector<float> > sigma_estimate; // estimate of variability per test flow per hypothesis for this read
  vector<vector<float> > basic_log_likelihoods; // log-likelihood given residuals at each test flow of the observation at that flow != likelihood of read
  
  float skew_estimate;

  vector<int > test_flow;  //  vector of flows to examine for this read and the hypotheses for efficiency
  int          start_flow; // Start flow as written in BAM <-- used in test flow computation

  // size number of hypotheses
  vector<float> responsibility; // how responsible this read is for a given hypothesis under the MAP: size number of hypotheses (including null=outlier)
  vector<float> weighted_responsibility; // responsibility * read_counter_f
  vector<float> log_likelihood; // sum over our test flows: logged to avoid under-flows
  vector<float> scaled_likelihood; // actual sum likelihood over test flows, rescaled to null hypothesis (as called), derived from log_likelihood
  float ll_scale; // local scaling factor for scaled likelihood as can't trust null hypothesis to be near data

  // intermediate allocations
  vector<float> tmp_prob_f;
  vector<double> tmp_prob_d;

  // flow-disruptiveness for all pairs of hypotheses in the read level
  // local_flow_disruptiveness_matrix[i][j] indicates the flow-disruptiveness between instance_of_read_by_state[i] and instance_of_read_by_state[j]
  // -1: indefinite (e.g. no common suffix bases), 0: HP-INDEL (change of HP at only one flow), 1: Non-FD and not HP-INDEL (e.g., non-FD SNP), 2: FD
  // @TODO: Use enumerate to represent the fd-code.
  vector<vector<int> > local_flow_disruptiveness_matrix;

  PrecomputeTDistOddN my_t;

  // useful hidden variables
  int strand_key;

  int heavy_tailed;
  bool adjust_sigma;
  float sigma_factor;
  int max_flows_to_test;
  float min_delta_for_flow;

  float magic_sigma_base;
  float magic_sigma_slope;

  int splice_start_flow; // Flow just before we start splicing in hypotheses (same for all hypotheses)
  int splice_end_flow;   // Flow of the first base after the variant window (maximum over all hypotheses)
  int max_last_flow;     // Last flow that is being simulated in prediction generation (max over all hypotheses)

  int read_counter;      // Indicating how many reads form this read (>1 means it is a consensus read)
  float read_counter_f;  // float of read_counter
  bool success;

  bool at_least_one_same_as_null;

// functions
  CrossHypotheses(){
    heavy_tailed = 3;  // t_5 degrees of freedom
    adjust_sigma = false;
    sigma_factor = 1.0f;
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
    read_counter = 1;
    read_counter_f = 1.0f;
    at_least_one_same_as_null = false;
    local_flow_disruptiveness_matrix.clear();
  };
  void  CleanAllocate(int num_hyp, int num_flow);
  void  ClearAllFlowsData();
  void  SetModPredictions();
  void  FillInPrediction(PersistingThreadObjects &thread_objects, const Alignment &my_read, const InputStructures &global_context);
  void  InitializeDerivedQualities();
  void  InitializeTestFlows();
  void  InitializeRelevantToTestFlows();
  void  ComputeResiduals();
  void  ResetModPredictions();
  void  ComputeDeltaCorrelation();
  void  ResetRelevantResiduals();
  void  ComputeBasicLogLikelihoods();
  void  ComputeLogLikelihoods();
  void  ComputeLogLikelihoodsSum();
  void  JointLogLikelihood();
  void  ComputeScaledLikelihood();
  float ComputePosteriorLikelihood(const vector<float> &hyp_prob, float outlier_prob);
  void  InitializeSigma();
  void  InitializeResponsibility();
  void  UpdateResponsibility(const vector<float > &hyp_prob, float outlier_prob);
  void  UpdateRelevantLikelihoods();
  void  ComputeDelta();
  bool  ComputeAllComparisonsTestFlow(float threshold, int max_choice);
  float ComputeLLDifference(int a_hyp, int b_hyp);
  int   MostResponsible() const;
  bool  IsValidTestFlowIndexOld(unsigned int flow,unsigned int max_choice);
  bool  IsValidTestFlowIndexNew(unsigned int flow,unsigned int max_choice);

  void  FillInFlowDisruptivenessMatrix(const ion::FlowOrder &flow_order, const Alignment &my_alignment);
  bool  OutlierByFlowDisruptiveness() const;
};


// Deal with the inference for a single family
class EvalFamily : public AbstractMolecularFamily<unsigned int>{
public:
	vector<float> family_responsibility;
	EvalFamily(const string &barcode, int strand, const vector<const Alignment *>* const read_stack)
	  : AbstractMolecularFamily(barcode, strand), read_stack_(read_stack) {};
	~EvalFamily(){};
	int CountFamSizeFromValid();
	int CountFamSizeFromAll();
	void InitializeEvalFamily(unsigned int num_hyp);
	void CleanAllocate(unsigned int num_hyp);
	void InitializeFamilyResponsibility();
	void ComputeFamilyLogLikelihoods(const vector<CrossHypotheses> &my_hypotheses);
	void UpdateFamilyResponsibility(const vector<float > &hyp_prob, float outlier_prob);
	void ComputeFamilyOutlierResponsibility(const vector<CrossHypotheses> &my_hypotheses, unsigned int min_fam_size);
	float ComputeFamilyPosteriorLikelihood(const vector<float> &hyp_prob);
	float ComputeLLDifference(int a_hyp, int b_hyp) {return my_family_cross_.ComputeLLDifference(a_hyp, b_hyp);};
	int MostResponsible();
	vector<float> GetFamilyLogLikelihood() const{ return my_family_cross_.log_likelihood; };
	vector<float> GetFamilyScaledLikelihood() const{ return my_family_cross_.scaled_likelihood; };
	void FillInFlowDisruptivenessMatrix(const vector<CrossHypotheses> &my_hypotheses);
	int GetFlowDisruptiveness(int i_hyp, int j_hyp) const { return my_family_cross_.local_flow_disruptiveness_matrix[i_hyp][j_hyp]; };
private:
	const vector<const Alignment *>* const read_stack_; // Used to calculate family size
	// The calculation of log-likelihood etc. of a family is pretty much the same as a single read.
	// my_family_cross_ is used for calculating the "likelihoods" and "responsibility" only.
	// must be use my_family_cross_ carefully since it has a lot of uninitialized members.
	// Keep my_family_cross_ private in case someone tries to access those uninitialized members.
	CrossHypotheses my_family_cross_;
};

bool IsHpIndel(const string& seq_1, const string& seq_2);

#endif // CROSSHYPOTHESES_H
