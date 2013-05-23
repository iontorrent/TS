/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef STACKENGINE_H
#define STACKENGINE_H

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
#include "CrossHypotheses.h"
#include "BiasGenerator.h"
#include "SigmaGenerator.h"
#include "SkewGenerator.h"
#include "StackPlus.h"
#include "ExtendParameters.h"
#include "PosteriorInference.h"

using namespace std;


// what is the set of parameters describing a complete set of latent variables under some state
class LatentSlate{
  public:
    // frequency
    PosteriorInference cur_posterior;
  // information by strand
  PosteriorInference fwd_posterior;
  PosteriorInference rev_posterior;
    
    // track important latent variables
   BasicBiasGenerator bias_generator;
// and the other important set of latent variables
   BasicSigmaGenerator sigma_generator;
   // and the third set
   BasicSkewGenerator skew_generator;
   
   bool detailed_integral;
   int max_iterations;
   int iter_done;
   vector<float> ll_at_stage;
  float start_freq_of_winner;
 
  void DetailedExecuteInference(ShortStack &total_theory, bool update_frequency, bool update_sigma);
  void FastExecuteInference(ShortStack &total_theory, bool update_frequency, bool update_sigma, float start_frequency);
  void LocalExecuteInference(ShortStack &total_theory, bool update_frequency, bool update_sigma, float start_frequency);
  void FastStep(ShortStack &total_theory, bool update_frequency, bool update_sigma);
  void DetailedStep(ShortStack &total_theory, bool update_frequency, bool update_sigma);
  void ScanStrandPosterior(ShortStack &total_theory);
  void SolveForFixedFrequency(ShortStack &total_theory, float test_freq);
  void PropagateTuningParameters(EnsembleEvalTuningParameters &my_params);
  LatentSlate(){
    max_iterations = 10;
    detailed_integral = true;
    iter_done = 0;
    start_freq_of_winner = 0.5f;
  };
};

class HypothesisStack{
public:
  ShortStack total_theory;
  
  // latent variables under states of the world
  LatentSlate cur_state;
    
  EnsembleEvalTuningParameters my_params;
  bool try_alternatives;


// induced by this variant from the stack of reads
  int variant_position;
  string variant_contig;
  string ref_allele;
  string var_allele; // all just annoying trash for a unique identifier

  HypothesisStack(){
     DefaultValues();
  }

  void DefaultValues();
  void PropagateTuningParameters();

// starting to make inferences
  void RestoreFullInference();
  void SetAlternateFromMain();
  void ExecuteFullInference();
  void ExecuteExtremeInferences();
  void ExecuteInference( );
  void InitForInference( StackPlus &my_data);
  
  // change estimates for variance
  
  // tool for posterior density estimation
  bool CallGermline(float hom_safety, int &genotype_call, float &quasi_phred_quality_score, float &reject_status_quality_score);
  void CallByMAP(int &genotype_call, float &quasi_phred_quality_score);
  float ReturnMaxLL();
};

class EnsembleEval{
  public:
    StackPlus my_data;
    MultiAlleleVariantIdentity multi_allele_var;
 
    vector<HypothesisStack> allele_eval;
    
    void SetupHypothesisChecks(ExtendParameters *parameters);
    void ApproximateHardClassifierForReads(vector<int> &read_allele_id, vector<bool> &strand_id);
    void ScanSupportingEvidence(float &mean_ll_delta, float &mean_supporting_flows, float &mean_max_discrimination, float threshold, int i_allele);
    void UnifyTestFlows();
    int DetectBestAllele();
    int DetectBestAlleleHardClassify();
    int DetectBestAlleleML();
   void ExecuteInferenceAllAlleles();
};


#endif // STACKENGINE_H
