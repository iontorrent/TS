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
//  PosteriorInference fwd_posterior;
//  PosteriorInference rev_posterior;
    
    // track important latent variables
   BasicBiasGenerator bias_generator;
   // track filter
   BiasChecker bias_checker;
// and the other important set of latent variables
   StrandedSigmaGenerator sigma_generator;
   // and the third set
   BasicSkewGenerator skew_generator;
   
   bool detailed_integral;
   int max_iterations;
   int iter_done;
   vector<float> ll_at_stage;
  vector<float> start_freq_of_winner;
 
  void FastExecuteInference(ShortStack &total_theory, bool update_frequency, bool update_sigma, vector<float> &start_frequency);
  void LocalExecuteInference(ShortStack &total_theory, bool update_frequency, bool update_sigma, vector<float> &start_frequency);
  void FastStep(ShortStack &total_theory, bool update_frequency, bool update_sigma);
  void DetailedStep(ShortStack &total_theory, bool update_frequency, bool update_sigma);
  void ScanStrandPosterior(ShortStack &total_theory, bool vs_ref);
  void ResetToOrigin();
  void PropagateTuningParameters(EnsembleEvalTuningParameters &my_params, int num_hyp_no_null);
  LatentSlate(){
    max_iterations = 10;
    detailed_integral = true;
    iter_done = 0;
  };
};

class HypothesisStack{
public:
  ShortStack total_theory;
  
  // latent variables under states of the world
  LatentSlate cur_state;
    
  EnsembleEvalTuningParameters my_params;
  bool try_alternatives;

  vector<float> ll_record;
  vector<vector <float> > try_hyp_freq;

// induced by this variant from the stack of reads
  int variant_position;
  string variant_contig;
  string ref_allele;
  // --- XXX This field does not make sense any more in a multi-allele evaluation
  string var_allele; // all just annoying trash for a unique identifier

  HypothesisStack(){
     DefaultValues();
  }
  void AllocateFrequencyStarts(int num_hyp_no_null);
  void DefaultValues();
  void PropagateTuningParameters(int num_hyp_no_null);

// starting to make inferences
  void RestoreFullInference();
  void SetAlternateFromMain();
  void ExecuteExtremeInferences();
  void TriangulateRestart();
  float ExecuteOneRestart(vector<float> &restart_hyp);
  void ExecuteInference( );
  void InitForInference(PersistingThreadObjects &thread_objects, StackPlus &my_data, InputStructures &global_context, int num_hyp_no_null);
  
  // change estimates for variance
  
  // tool for posterior density estimation
  bool CallGermline(float hom_safety, int &genotype_call, float &quasi_phred_quality_score, float &reject_status_quality_score);

  float ReturnMaxLL();
};

class EnsembleEval{
  public:
    StackPlus my_data;
    MultiAlleleVariantIdentity multi_allele_var;
 
   HypothesisStack allele_eval;
    
    vector<int> diploid_choice;
    
    EnsembleEval(){
      diploid_choice.assign(2,0);
      diploid_choice.at(1)=1; // ref = 0, alt = 1
    };

    void SetupHypothesisChecks(ExtendParameters *parameters);
    void ApproximateHardClassifierForReads(vector<int> &read_allele_id, vector<bool> &strand_id);
    void ApproximateHardClassifierForReadsFromMultiAlleles(vector<int> &read_allele_id, vector<bool> &strand_id);
    void ScanSupportingEvidence(float &mean_ll_delta, int i_allele);
    int DetectBestAllele();
    int DetectBestMultiAllelePair();
    int DetectBestSingleAllele();
    int DetectBestAlleleHardClassify();
   void ExecuteInferenceAllAlleles();
   void ComputePosteriorGenotype(int _alt_allele_index,float local_min_allele_freq, int &genotype_call,
                                 float &gt_quality_score, float &reject_status_quality_score);
   void MultiAlleleGenotype(float local_min_allele_freq,
                            vector<int> &genotype_component, float &gt_quality_score, float &reject_status_quality_score);
};


#endif // STACKENGINE_H
