/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     parameter.h
//! @ingroup  VariantCaller
//! @brief    Indel detection

#ifndef EXTENDPARAMETERS_H
#define EXTENDPARAMETERS_H

#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <getopt.h>
#include <stdlib.h>
#include <Parameters.h>
#include "OptArgs.h"
#include "json/json.h"

using namespace std;


// provide an interface to set useful parameters across the objects
class EnsembleEvalTuningParameters {
  public:
    float outlier_prob;  // 1-data_reliability
    int heavy_tailed;    // how heavy are the tails of my distribution to resist mis-shapen items = CrossHypotheses
    float prediction_precision; //  damper_bias = bias_generator - how likely the predictions are to be accurately calibrated

    float min_delta_for_flow; // select flows to test based on this minimum level
    int max_flows_to_test;  // select how many flows to test
     bool use_unification_for_multialleles; // unify test flows across multialleles to avoid damaged SNPs

    float pseudo_sigma_base; // general likelihood penalty for shifting locations

    float magic_sigma_base; // weak prior on predicting variance by intensity
    float magic_sigma_slope;
    float sigma_prior_weight;
    float k_zero;  // weight for cluster shifts in adding to variance

    // filter parameters specialized to ensemble eval
    float filter_unusual_predictions;
    float filter_deletion_bias;
    float filter_insertion_bias;
    
    float million_monkeys_level; // don't bother phred scores outside this range: by popular request

    EnsembleEvalTuningParameters() {
      outlier_prob = 0.01f;
      heavy_tailed = 3; //t5
      prediction_precision = 30.0f;
      pseudo_sigma_base = 0.3f;

      magic_sigma_base = 0.085f;
      magic_sigma_slope = 0.0084f;
      sigma_prior_weight = 1.0f;
      k_zero = 0.0f;

      min_delta_for_flow = 0.1f;
      max_flows_to_test = 10;

      filter_unusual_predictions = 0.3f;
      filter_deletion_bias = 10.0f;
      filter_insertion_bias = 10.0f;
      
      million_monkeys_level = 100.0f; // 10^-10 should be good enough for most purposes
      
      use_unification_for_multialleles = false;
    };
    float DataReliability() {
      return(1.0f -outlier_prob);
    };
    bool CheckTuningParameters() {
      bool problem_detected = false;
      if (magic_sigma_base<0.0f){
        cout << "outlier_sigma_base" << magic_sigma_base << endl;
        magic_sigma_base = 0.01f;
        problem_detected=true;
      }
      if (magic_sigma_slope<0.0f){
        cout << "outlier_sigma_slope" << magic_sigma_slope << endl;
        magic_sigma_slope = 0.001f;
        problem_detected=true;
      }
      if (sigma_prior_weight<0.01f){
        cout << "outlier_sigma_weight" << sigma_prior_weight << endl;
        sigma_prior_weight = 0.01f;
        problem_detected=true;
      }
      if (pseudo_sigma_base<0.01f){
        cout << "outlier_pseudo_sigma_base" << pseudo_sigma_base << endl;
        pseudo_sigma_base = 0.01f;
        problem_detected=true;
      }
      if (outlier_prob < 0.0f) {
        cout << "outlier_prob" << outlier_prob << endl;
        outlier_prob = 0.0f;
        problem_detected = true;
      }
      if (outlier_prob > 1.0f) {
        cout << "outlier_prob" << outlier_prob << endl;
        outlier_prob = 1.0f;
        problem_detected = true;
      }
      if (heavy_tailed < 1) {
        cout << "heavy_tailed" << heavy_tailed << endl;
        heavy_tailed = 1; // can't go lower than Cauchy!
        problem_detected = true;
      }
      if (prediction_precision < 0.1f) {
        cout << "prediction_precision" << prediction_precision << endl;
        prediction_precision = 0.1f;  // avoid divide by zero: 1/10 data poitn strength
        problem_detected = true;
      }
      // less than 1% change is unlikely to be detected
      if (min_delta_for_flow < 0.01f) {
        cout << "min_delta_for_flow" << min_delta_for_flow << endl;
        min_delta_for_flow = 0.01f;
        problem_detected = true;
      }
      if (max_flows_to_test < 1) {
        cout << "Flows" << max_flows_to_test << endl;
        max_flows_to_test = 1;
        problem_detected = true;
      }
      return(problem_detected);
    };
    void SetOpts(OptArgs &opts, Json::Value& tvc_params);

};

class BasicFilters {
  public:
    float min_allele_freq;

    float strand_bias_threshold;
    float beta_bias_filter;
    float min_quality_score;

    int min_cov;
    int min_cov_each_strand;

    BasicFilters() {
      min_allele_freq = 0.2f;
      strand_bias_threshold = 0.8f;
      beta_bias_filter = 8.0f;
      min_cov = 3;
      min_cov_each_strand = 3;
      min_quality_score = 2.5f;
    };
};

class PeakControl {
  public:
    int fpe_max_peak_deviation;
    int hp_max_single_peak_std_23;
    int hp_max_single_peak_std_increment;
    PeakControl() {
      fpe_max_peak_deviation = 31;
      hp_max_single_peak_std_23 = 18;
      hp_max_single_peak_std_increment = 5;
    };
    void SetOpts(OptArgs &opts, Json::Value &tvc_params);
};

// control filters based on variant local sequence context alone
class ClassifyFilters {
  public:
    // define tricky homopolymer runs
    int hp_max_length ;
    //int max_hp_too_big;

    // define tricky snps
    int min_hp_for_overcall;
    int adjacent_max_length;

    // how to handle SSE issues
    float sseProbThreshold;
    float minRatioReadsOnNonErrorStrand;
    // don't worry about small relative SSE events
    float sse_relative_safety_level; 

    ClassifyFilters() {
      min_hp_for_overcall = 5;
      //max_hp_too_big = 12;
      hp_max_length = 11;
      adjacent_max_length = 11;

      sseProbThreshold = 0.2;
      minRatioReadsOnNonErrorStrand = 0.2; // min ratio of reads supporting variant on non-sse strand for variant to be called
      sse_relative_safety_level = 0.03f; // scale SSE we worry about by read depth - don't worry about anything less than a 5% problem
    };
    void SetOpts(OptArgs &opts, Json::Value & tvc_params);
};


class ControlCallAndFilters {
  public:

    // values relating the decision tree for calling/genotyping
    // i.e controlling no-call/call decisions
    // controlling variant /reference decisions
    // Note this is >not< the candidate generation tree but the calling decision tree
    float data_quality_stringency;
    int downSampleCoverage;
    int RandSeed;                  //!< Seed for random number generator to reservoir sample reads.

    bool suppress_reference_genotypes;
    bool suppress_no_calls;

    ClassifyFilters filter_variant;
    PeakControl control_peak;
    
    // tuning parameter for xbias 
    float xbias_tune;
    float sbias_tune; 

    BasicFilters filter_snps;
    BasicFilters filter_hp_indel;
    BasicFilters filter_hotspot;

    ControlCallAndFilters();
    void SetOpts(OptArgs &opts, Json::Value& tvc_params);
};

class ProgramControlSettings {
  public:
    // how we do things
    int nThreads;
    int nVariantsPerThread;
    int DEBUG;

    bool rich_json_diagnostic;
     string json_plot_dir;

    bool do_ensemble_eval;
    bool use_SSE_basecaller;
    bool suppress_recalibration;
    bool do_snp_realignment;

    bool inputPositionsOnly;
    bool skipCandidateGeneration;

    ProgramControlSettings();
    void SetOpts(OptArgs &opts, Json::Value & pf_params);
};


class ExtendParameters : public Parameters {
  public:
	OptArgs opts;
    ControlCallAndFilters my_controls;
    EnsembleEvalTuningParameters my_eval_control;
    ProgramControlSettings program_flow;

//    string stringency;
    bool vcfProvided;
    int info_vcf;
    bool consensusCalls;

    //Input files
    string inputBAM;
    string outputDir;

    string sseMotifsFileName;
    bool sseMotifsProvided;

    vector<string> bams;
    string sampleName;
    string referenceSampleName;
    vector<string> ReadGroupIDVector;
    string candidateVCFFileName;

    string recal_model_file_name;
    int recalModelHPThres;


    // functions
    ExtendParameters(void);
    ExtendParameters(int argc, char** argv);
    void SetupFileIO(OptArgs &opts);
    void SetFreeBayesParameters(OptArgs &opts, Json::Value& fb_params);
    void ParametersFromJSON(OptArgs &opts, Json::Value &tvc_params, Json::Value &fb_params);



};

#endif // EXTENDPARAMETERS_H

