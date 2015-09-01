/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     parameter.h
//! @ingroup  VariantCaller
//! @brief    Indel detection

#ifndef EXTENDPARAMETERS_H
#define EXTENDPARAMETERS_H

#include <string>
#include <vector>
#include "OptArgs.h"
#include "json/json.h"

using namespace std;


// provide an interface to set useful parameters across the objects
class EnsembleEvalTuningParameters {
  public:
    float germline_prior_strength; // how concentrated are we at 0,0.5,1.0 frequency for germline calls
    float outlier_prob;  // 1-data_reliability
    int heavy_tailed;    // how heavy are the tails of my distribution to resist mis-shapen items = CrossHypotheses
    float prediction_precision; //  damper_bias = bias_generator - how likely the predictions are to be accurately calibrated

    float min_delta_for_flow; // select flows to test based on this minimum level
    int max_flows_to_test;  // select how many flows to test
     //bool use_all_compare_for_test_flows;  // include the read-as-called in the comparison to avoid damaged hot-spots

    float pseudo_sigma_base; // general likelihood penalty for shifting locations

    float magic_sigma_base; // weak prior on predicting variance by intensity
    float magic_sigma_slope;
    float sigma_prior_weight;
    float k_zero;  // weight for cluster shifts in adding to variance

    // filter parameters specialized to ensemble eval
    float filter_unusual_predictions;
    float soft_clip_bias_checker;
    float filter_deletion_bias;
    float filter_insertion_bias;
    int   max_detail_level;
    

    EnsembleEvalTuningParameters() {
      germline_prior_strength = 0.0f;
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
      soft_clip_bias_checker = 0.1f;
      filter_deletion_bias = 10.0f;
      filter_insertion_bias = 10.0f;
      max_detail_level = 0;
      
      //use_all_compare_for_test_flows = false;
    };

    float DataReliability() {
      return(1.0f -outlier_prob);
    };

    void CheckParameterLimits();
    void SetOpts(OptArgs &opts, Json::Value& tvc_params);

};

class BasicFilters {
  public:
    float min_allele_freq;

    float strand_bias_threshold;
    float strand_bias_pval_threshold;

    float min_quality_score;

    int min_cov;
    int min_cov_each_strand;

    BasicFilters() {
      min_allele_freq = 0.2f;
      strand_bias_threshold = 0.8f;
      strand_bias_pval_threshold = 1.0f;

      min_cov = 3;
      min_cov_each_strand = 3;
      min_quality_score = 2.5f;
    };
};



// control filters based on variant local sequence context alone
class ClassifyFilters {
  public:
    // define tricky homopolymer runs
    int hp_max_length ;

    // how to handle SSE issues
    float sseProbThreshold;
    float minRatioReadsOnNonErrorStrand;
    // don't worry about small relative SSE events
    float sse_relative_safety_level; 

    // local realignment per variant type
    bool  do_snp_realignment;    //
    bool  do_mnp_realignment;
    float realignment_threshold; // Do not realign if fraction of reads changing alignment is above threshold

    // treat non hp indels as hp indels
    bool indel_as_hpindel;

    ClassifyFilters() {
      hp_max_length = 11;

      sseProbThreshold = 0.2;
      minRatioReadsOnNonErrorStrand = 0.2; // min ratio of reads supporting variant on non-sse strand for variant to be called
      sse_relative_safety_level = 0.03f; // scale SSE we worry about by read depth - don't worry about anything less than a 5% problem

      do_snp_realignment = false;
      do_mnp_realignment = false;
      realignment_threshold = 1.0;

      indel_as_hpindel = false;
    };
    void SetOpts(OptArgs &opts, Json::Value & tvc_params);
    void CheckParameterLimits();
};


class ControlCallAndFilters {
  public:

    // values relating the decision tree for calling/genotyping
    // i.e controlling no-call/call decisions
    // controlling variant /reference decisions
    // Note this is >not< the candidate generation tree but the calling decision tree
    float data_quality_stringency;
    float read_rejection_threshold;

    int downSampleCoverage;
    int RandSeed;                  //!< Seed for random number generator to reservoir sample reads.

    bool suppress_reference_genotypes;
    bool suppress_nocall_genotypes;
    bool heal_snps; // if a snp is the best allele, discard all others
    bool suppress_no_calls;

    // position bias probably should not be variant specific
    bool use_position_bias;
    float position_bias_ref_fraction;
    float position_bias;
    float position_bias_pval;

    ClassifyFilters filter_variant;

    // tuning parameter for xbias 
  //  float xbias_tune;
    float sbias_tune; 

    BasicFilters filter_snps;
    BasicFilters filter_mnp;
    BasicFilters filter_hp_indel;
    BasicFilters filter_hotspot;

    ControlCallAndFilters();
    void SetOpts(OptArgs &opts, Json::Value& tvc_params);
    void CheckParameterLimits();
};

class ProgramControlSettings {
  public:
    // how we do things
    int nThreads;
    int nVariantsPerThread;
    int DEBUG;

    bool rich_json_diagnostic;
    bool minimal_diagnostic;
     string json_plot_dir;

    bool use_SSE_basecaller;
    bool suppress_recalibration;
    bool resolve_clipped_bases;

    bool inputPositionsOnly;

    ProgramControlSettings();
    void SetOpts(OptArgs &opts, Json::Value & pf_params);
    void CheckParameterLimits();
};


class ExtendParameters {
public:
  vector<string>    bams;
  string            fasta;                // -f --fasta-reference
  string            targets;              // -t --targets
  string            outputFile;
  string            variantPriorsFile;
  string            postprocessed_bam;

  string            basecaller_version;
  string            tmap_version;

  bool              onlyUseInputAlleles;
  bool              processInputPositionsOnly;

  bool              trim_ampliseq_primers;
  int               prefixExclusion;

  // operation parameters
  bool useDuplicateReads;      // -E --use-duplicate-reads
  int useBestNAlleles;         // -n --use-best-n-alleles
  bool allowIndels;            // -I --allow-indels
  bool allowMNPs;              // -X --allow-mnps
  bool allowComplex;           // -X --allow-complex
  int maxComplexGap;
  bool allowSNPs;              // -I --no-snps
  int min_mapping_qv;                    // -m --min-mapping-quality
  float readMaxMismatchFraction;  // -z --read-max-mismatch-fraction
  int       read_snp_limit;            // -$ --read-snp-limit
  long double minAltFraction;  // -F --min-alternate-fraction
  long double minIndelAltFraction; // Added by SU to reduce Indel Candidates for Somatic
  int minAltCount;             // -C --min-alternate-count
  int minAltTotal;             // -G --min-alternate-total
  int minCoverage;             // -! --min-coverage
  bool debug; // set if debuglevel >=1



	OptArgs opts;
  ControlCallAndFilters my_controls;
  EnsembleEvalTuningParameters my_eval_control;
  ProgramControlSettings program_flow;

  //Input files
  string outputDir;

  string sseMotifsFileName;
  bool sseMotifsProvided;

  string sampleName;
  string force_sample_name;

  string recal_model_file_name;
  int recalModelHPThres;

  string              params_meta_name;
  string              params_meta_details;

  // functions
  ExtendParameters(int argc, char** argv);

  bool ValidateAndCanonicalizePath(string &path);
  void SetupFileIO(OptArgs &opts);
  void SetFreeBayesParameters(OptArgs &opts, Json::Value& fb_params);
  void ParametersFromJSON(OptArgs &opts, Json::Value &tvc_params, Json::Value &fb_params, Json::Value &params_meta);
  void CheckParameterLimits();

};

template <class T>
bool CheckParameterLowerUpperBound(string identifier ,T &parameter, T lower_limit, T upper_limit) {
  bool is_ok = false;

  //cout << setw(35) << long_name_hyphens << " = " << setw(10) << value << " (integer, " << source << ")" << endl;
  cout << "Limit check parameter " << identifier << ": lim. "
	   << lower_limit << " <= " << parameter << " <= lim. " << upper_limit << "? ";
  if (parameter < lower_limit) {
	cout << "Using " << identifier << "=" << lower_limit << " instead!";
    parameter = lower_limit;
  }
  else if (parameter > upper_limit) {
    cout << "Using " << identifier << "=" << upper_limit << " instead!";
    parameter = upper_limit;
  }
  else {
    cout << "OK!";
    is_ok = true;
  }
  cout << endl;
  return (is_ok);
}

template <class T>
bool CheckParameterLowerBound(string identifier ,T &parameter, T lower_limit) {
  bool is_ok = false;
  cout << "Limit check parameter " << identifier << ": lim. "
	   << lower_limit << " <= " << parameter << "? ";
  if (parameter < lower_limit) {
	cout << "Using " << identifier << "=" << lower_limit << " instead!";
    parameter = lower_limit;
  }
    else {
    cout << "OK!";
    is_ok = true;
  }
  cout << endl;
  return (is_ok);
}

template <class T>
bool CheckParameterUpperBound(string identifier ,T &parameter, T upper_limit) {
  bool is_ok = false;
  cout << "Limit check parameter " << identifier << ": "
	   << parameter << " <= lim. " << upper_limit << "? ";
  if (parameter > upper_limit) {
    cout << "Using " << identifier << "=" << upper_limit << " instead!";
    parameter = upper_limit;
  }
  else {
    cout << "OK!";
    is_ok = true;
  }
  cout << endl;
  return (is_ok);
}

#endif // EXTENDPARAMETERS_H

