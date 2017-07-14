/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     parameter.h
//! @ingroup  VariantCaller
//! @brief    Indel detection

#ifndef EXTENDPARAMETERS_H
#define EXTENDPARAMETERS_H

#include <string>
#include <vector>
#include "OptArgs.h"
#include "OptBase.h"
#include "json/json.h"
#include "MolecularTagTrimmer.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;

int mkpath(std::string s,mode_t mode);

//bool RetrieveParameterBool(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, bool default_value);

// provide an interface to set useful parameters across the objects
class EnsembleEvalTuningParameters {
  public:
    float germline_prior_strength; // how concentrated are we at 0,0.5,1.0 frequency for germline calls
    float outlier_prob;  // 1-data_reliability
    int heavy_tailed;    // how heavy are the tails of my distribution to resist mis-shapen items = CrossHypotheses
    bool adjust_sigma;   // If true, sigma^2 = (dof-2) / dof * E[r^2] (where dof = 2*heavy_tail - 1), else sigma^2 = E[r^2]
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
    int   min_detail_level_for_fast_scan;
    bool  try_few_restart_freq;
    
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
      min_detail_level_for_fast_scan = 0;

      try_few_restart_freq = false;
      //use_all_compare_for_test_flows = false;
    };

    void CheckParameterLimits();
    void SetOpts(OptArgs &opts, Json::Value& tvc_params);

};

class BasicFilters {
  public:
    float min_allele_freq = -1.0f;
    float strand_bias_threshold = -1.0f;
    float strand_bias_pval_threshold = -1.0f;
    float min_quality_score = -1.0f;
    int min_cov = -1;
    int min_cov_each_strand = -1;
    int min_var_cov = -1;
    string my_type;
    string underscored_my_type;

    BasicFilters() {};
    BasicFilters(const string& default_type);
    void CheckBasicFiltersLimits();
    void SetBasicFilterOpts(OptArgs &opts, Json::Value& tvc_params, BasicFilters* default_param);
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

    // the hp-indel filter
    vector<int> filter_hp_indel_hrun;
    vector<int> filter_hp_ins_len;
    vector<int> filter_hp_del_len;

    ClassifyFilters() {
      hp_max_length = 11;

      sseProbThreshold = 0.2;
      minRatioReadsOnNonErrorStrand = 0.2; // min ratio of reads supporting variant on non-sse strand for variant to be called
      sse_relative_safety_level = 0.03f; // scale SSE we worry about by read depth - don't worry about anything less than a 5% problem

      do_snp_realignment = false;
      do_mnp_realignment = false;
      realignment_threshold = 1.0;

      indel_as_hpindel = false;

      filter_hp_indel_hrun = {7, 8};
      filter_hp_ins_len = {0, 0};
      filter_hp_del_len = {0, 0};
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
    bool cleanup_unlikely_candidates; // if a snp is the best allele, discard all others
    bool suppress_no_calls;
    bool report_ppa;
    bool hotspots_as_de_novo;
    bool disable_filters;

    // position bias probably should not be variant specific
    bool  use_position_bias;
    float position_bias_ref_fraction;
    float position_bias;
    float position_bias_pval;

    // LOD filter
    bool  use_lod_filter;
    float lod_multiplier;

    // filter's for mol tags
    int  tag_sim_max_cov;

    // flow-disruptivenss stuff
    bool use_fd_param ;
    float min_ratio_for_fd;
    int fd_nonsnp_min_var_cov;

    ClassifyFilters filter_variant;

    // VCF record filters (applied during vcf merging)
    //bool    filter_by_target;          // Filter records based on mets information in target bed info field
    //bool    hotspot_positions_only;    // Output only vcf lines with the infoFlag 'HS'
    //bool    hotspot_variants_only;     // Suppress hotspot reference calls and no-calls from the final output vcf

    // tuning parameter for xbias 
    //float xbias_tune;
    float sbias_tune; 

    // Filtering parameters that depend on the variant type.
    BasicFilters filter_snp = BasicFilters("snp");
    BasicFilters filter_mnp = BasicFilters("mnp");
    BasicFilters filter_hp_indel = BasicFilters("indel");
    BasicFilters filter_hotspot = BasicFilters("hotspot");
    BasicFilters filter_fd_0 = BasicFilters("fd-0");
    BasicFilters filter_fd_5 = BasicFilters("fd-5");
    BasicFilters filter_fd_10 = BasicFilters("fd-10");

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

    bool do_indel_assembly;

    // Directory in ExtendParameters directly
    bool rich_json_diagnostic;
    bool minimal_diagnostic;


    bool use_SSE_basecaller;
    bool suppress_recalibration;
    bool resolve_clipped_bases;

    bool inputPositionsOnly;

    bool is_multi_min_allele_freq;
    vector<float> multi_min_allele_freq;

    ProgramControlSettings();
    void SetOpts(OptArgs &opts, Json::Value & pf_params);
    void CheckParameterLimits();
};

struct TvcTagTrimmerParameters : TagTrimmerParameters{
	int indel_func_size_offset = 0;
};

class ExtendParameters {
public:
  vector<string>    bams;
  string            fasta;                // -f --fasta-reference
  string            targets;              // -t --targets

  string            small_variants_vcf;   // small indel output vcf file name
  string            indel_assembly_vcf;   // indel assembly vcf file name
  //string            merged_vcf;           // merged and post processed vcf file name
  //string            merged_genome_vcf;    // merged gvcf file name

  string            blacklistFile;
  string            variantPriorsFile;
  string            postprocessed_bam;
  string            json_plot_dir;

  string            basecaller_version;
  string            tmap_version;

  bool              onlyUseInputAlleles;
  bool              processInputPositionsOnly;

  bool              trim_ampliseq_primers;
  float             min_cov_fraction;

  int               prefixExclusion;

  // operation parameters
  //TODO: Put the Freebayes parameters in a container.
  bool useDuplicateReads;      // -E --use-duplicate-reads
  int useBestNAlleles;         // -n --use-best-n-alleles
  int max_alt_num;             // Try to break the variant if the number of alt alleles is greater than this value.
  int useBestNTotalAlleles;    //    --use-best-n-total-alleles
  bool allowIndels;            // -I --allow-indels
  bool allowMNPs;              // -X --allow-mnps
  bool allowComplex;           // -X --allow-complex
  int maxComplexGap;
  bool allowSNPs;              // -I --no-snps
  int min_mapping_qv;                    // -m --min-mapping-quality
  float readMaxMismatchFraction;  // -z --read-max-mismatch-fraction
  int       read_snp_limit;            // -$ --read-snp-limit
  int       read_mismatch_limit;
  long double minAltFraction;  // -F --min-alternate-fraction
  long double minIndelAltFraction; // Added by SU to reduce Indel Candidates for Somatic
  int minAltCount;             // -C --min-alternate-count
  int minAltTotal;             // -G --min-alternate-total
  int minCoverage;             // -! --min-coverage
  int mergeLookAhead;          // --merge-variant-lookahead
  bool debug; // set if debuglevel >=1
  bool multisample;            // multisample run

  string  candidate_list, black_listed; 
  OptArgs opts;
  ControlCallAndFilters my_controls;
  EnsembleEvalTuningParameters my_eval_control;
  ProgramControlSettings program_flow;
  TvcTagTrimmerParameters         tag_trimmer_parameters;

  //Input files
  string outputDir;

  string sseMotifsDir;
  string sseMotifsFileName;
  bool sseMotifsProvided;

  string sampleName;
  string force_sample_name;

  string recal_model_file_name;
  int recalModelHPThres;
  bool output_allele_cigar;

  string              params_meta_name;
  string              params_meta_details;

  // functions
  ExtendParameters() {}
  ExtendParameters(int argc, char** argv);

  bool ValidateAndCanonicalizePath(string &path);
  void SetupFileIO(OptArgs &opts, Json::Value& tvc_params);
  void SetFreeBayesParameters(OptArgs &opts, Json::Value& fb_params);
  void ParametersFromJSON(OptArgs &opts, Json::Value &tvc_params, Json::Value &fb_params, Json::Value &params_meta);
  void CheckParameterLimits();
  void SetMolecularTagTrimmerOpt(Json::Value& tvc_params);

};

template <typename MyIter>
string PrintIteratorToString(const MyIter &it_start, const MyIter &it_end,
		string left_bracket = "[", string right_bracket = "]", string separation = ", ", string entry_prefix = "") {
	string return_str = left_bracket;
	MyIter last_it = it_end;
    --last_it;
    for (MyIter it = it_start; it != it_end; ++it){
    	return_str += (entry_prefix + to_string(*it));
    	if (it != last_it)
        	return_str += separation;
    }
    return_str += right_bracket;
    return return_str;
}

#endif // EXTENDPARAMETERS_H

