/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "Utils.h"
#include "ionstats.h"
#include "ionstats_data.h"
#include "ionstats_alignment.h"
#include "ionstats_alignment_summary.h"

#include <string>
#include <fstream>
#include <map>
#include <limits>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <ctype.h>


#include "api/BamWriter.h"
#include "api/BamReader.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"

#include "OptArgs.h"
#include "Utils.h"
#include "IonVersion.h"
#include "ion_util.h"
#include "hdf5.h"

#define MAX_GROUP_NAME_LEN 5000

using namespace std;
using namespace BamTools;

typedef struct error_data_merge_t {
  map< string, ErrorData > error_data;
  bool merge_proton_blocks;
} error_data_merge_t;

typedef struct ProcessAlignmentContext {
  IonstatsAlignmentBamReader *input_bam;
  AlignmentSummary *alignment_summary;
  BamWriter *output_bam;
  IonstatsAlignmentOptions *opt;
  pthread_mutex_t  *read_mutex;
  pthread_mutex_t  *write_mutex;
  pthread_mutex_t  *results_mutex;
  
} ProcessAlignmentContext;

typedef struct hp_data_merge_t {
  map< string, HpData > hp_data;
  bool merge_proton_blocks;
} hp_data_merge_t;

typedef struct regional_summary_merge_t {
  map< string, RegionalSummary > regional_summary;
  bool merge_proton_blocks;
} regional_summary_merge_t;

bool isAqLength(string s);
bool revSort(double i, double j) { return(i>j); };
void initialize_reverse_complement_map(map<char, char> &rc);
int reverse_complement(string &b, const map<char, char> &rc);
void computeAQ(vector<int> &aq_length, vector<double> &aq_error_rate, ReadAlignmentErrors &base_space_errors);
int toPhred(double e);
bool getRegionIndex(unsigned int &region_idx, bool &no_region, bool &out_of_bounds, string &name, vector<unsigned int> &chip_origin, vector<unsigned int> &chip_dim, vector<unsigned int> &subregion_dim, unsigned int n_col_subregions);
void parseMD(const string &MD_tag, vector<char> &MD_op, vector<int> &MD_len, vector<string> &MD_seq);
void getReadGroupInfo(const BamReader &input_bam, map< string, int > &read_groups, map< string, string > &flow_orders, unsigned int &max_flow_order_len, map< string, string > &key_bases, map< string, int > &key_len, const string &seq_key, const string &skip_rg_suffix);
void getBarcodeResults(BamAlignment &alignment, map< string, int > &key_len, int &bc_bases, int &bc_errors);
int checkDimensions(bool &spatial_stratify, const vector<unsigned int> &chip_origin, const vector<unsigned int> &chip_dim, const vector<unsigned int> &subregion_dim, unsigned int &n_col_subregions, unsigned int &n_row_subregions);
void assignRegionNames(vector<string> &region_name, const vector<unsigned int> &chip_origin, const vector<unsigned int> &chip_dim, const vector<unsigned int> &subregion_dim, vector< vector<unsigned int> > &region_specific_origin, vector< vector<unsigned int> > &region_specific_dim);
int parseAlignment(
  BamAlignment &          alignment,
  ReadAlignmentErrors &   base_space_errors,
  ReadAlignmentErrors &   flow_space_errors,
  map<string, string> &   flow_orders,
  string &                read_group,
  const map<char,char> &  reverse_complement_map,
  bool                    evaluate_flow,
  unsigned int            max_flows,
  bool                    evaluate_hp,
  bool &                  invalid_read_bases,
  bool &                  invalid_ref_bases,
  bool &                  invalid_cigar,
  vector<char> &          ref_hp_nuc,
  vector<uint16_t> &      ref_hp_len,
  vector<int16_t> &       ref_hp_err,
  vector<uint16_t> &      ref_hp_flow,
  vector<uint16_t> &      zeromer_insertion_flow,
  vector<uint16_t> &      zeromer_insertion_len
);
void debug_alignment(
  IonstatsAlignmentOptions * opt,
  BamAlignment &             alignment,
  ReadAlignmentErrors &      base_space_errors,
  ReadAlignmentErrors &      flow_space_errors,
  vector<uint16_t> &         ref_hp_flow
);
void scoreBarcodes(ReadAlignmentErrors &base_space_errors, int bc_bases, int bc_errors, vector<double> &aq_error_rate, int minimum_aq_length, vector<int> &aq_length, AlignmentSummary * alignment_summary);
string getReferenceBases(const string &read_bases, vector< CigarOp > CigarData, const vector<char> &MD_op, vector<int> MD_len, const vector<string> &MD_seq, const bool rev_strand, const map<char,char> &reverse_complement_map);
void getHpBreakdown(const string &bases, vector<char> &hp_nuc, vector<uint16_t> &hp_len, vector<uint16_t> &hp_cum_len);
int writeIonstatsAlignmentJson(
  const string &json_filename,
  const vector<double> &aq_error_rate,
  bool bc_adjust,
  bool evaluate_flow,
  uint Num_regions,
  AlignmentSummary alignment_summary
);
void writeIonstatsH5(string h5_filename, bool append_h5_file, const vector<string> & region_name, AlignmentSummary & alignment_summary);
void AddToReadLengthHistogramFromJson(const Json::Value &input_json, const string &var_name, ReadLengthHistogram &hist);
void AddToSimpleHistogramFromJson(const Json::Value &input_json, const string &var_name, SimpleHistogram &hist, bool &found);
bool getTagZF(BamAlignment & alignment, uint32_t &flow_idx);
bool hasInvalidCigar(const BamAlignment & alignment);
void checkBases(string &b, bool &ambiguous_bases, bool &invalid_bases);
string transform_proton_block_read_group_name(const string &group_name);
void GetAggregatorSize(map<string, ErrorData> &error_data, map<string, HpData> &hp_data, map<string, RegionalSummary> &regional_summary, unsigned int &group_count, uint64_t &total_bytes);
bool is_unambiguous_base(char b);
bool compatible_bases(char a, char b);
unsigned char base_to_bitcode(char base);
unsigned int assignDeletionsToFlows(vector<uint16_t> &del_flow, uint32_t &flow_idx, char prev_read_base, char next_read_base, const string &flow_order, int advance, const string inserted_seq, uint16_t n_flow);
void * processAlignments(void *in);

enum align_t {ALIGN_MATCH, ALIGN_INS, ALIGN_DEL, ALIGN_SUB};

void hpAdvance(
  // Inputs
  align_t                   alignment_type,
  int                       advance,         // number of bases in perfectly-aligned stretch
  // Data for tracking flow
  bool                      evaluate_flow,
  unsigned int              flow_idx,
  const string &            flow_order,
  const string &            read_bases,
  const string &            ref_bases,
  unsigned int              max_flows,
  // Data for tracking position in bases/HPs
  int                       read_idx,         // index of where we are in read_bases
  const vector<char> &      read_hp_nuc,      // read hp nucs
  const vector<uint16_t> &  read_hp_len,      // read hp lengths
  const vector<uint16_t> &  read_hp_cum_len,  // read hp cumulative lengths
  int                       ref_idx,          // index of where we are in ref_bases
  const vector<char> &      ref_hp_nuc,       // ref hp nucs
  const vector<uint16_t> &  ref_hp_len,       // ref hp lengths
  const vector<uint16_t> &  ref_hp_cum_len,   // ref hp cumulative lengths
  // Objects that may be modified
  int &                     stored_read_match_count, // number of read bases matching current ref hp that have been seen so far
  unsigned int &            read_hp_idx,             // index of where we are in read_hp_nuc, read_hp_len, read_hp_cum_len
  unsigned int &            ref_hp_idx,              // index of where we are in ref_hp_nuc, ref_hp_len, ref_hp_cum_len
  vector<int16_t> &         ref_hp_err,              //
  vector<uint16_t> &        ref_hp_flow,             //
  vector<uint16_t> &        zeromer_insertion_flow,  // flows in which there is an insertion where reference hp length is zero
  vector<uint16_t> &        zeromer_insertion_len    // insertion length for insertions against a reference zeromer
);

void IonstatsAlignmentOptions::WriteHelp(void)
{
  cerr << endl;
  cerr << "ionstats " << IonVersion::GetVersion() << "-" << IonVersion::GetRelease() << " (" << IonVersion::GetGitHash() << ") - Generate performance metrics and statistics for Ion sequences." << endl;
  cerr << endl;
  cerr << "Usage:   ionstats alignment -i in.bam [options]" << endl;
  cerr << endl;
  cerr << "General options:" << endl;
  cerr << "  --help                       BOOL      print this help message [" << DEFAULT_HELP << "]" << endl;
  cerr << "  -i,--input                   STRING    input BAM (mapped) [" << DEFAULT_INPUT_BAM << "]" << endl;
  cerr << "  -o,--output                  STRING    output json file [" << DEFAULT_OUTPUT_JSON << "]" << endl;
  cerr << "  --output-h5                  STRING    output hdf5 file [" << DEFAULT_OUTPUT_H5 << "]" << endl;
  cerr << "  --output-bam                 STRING    output BAM (for use in pipes) [" << DEFAULT_OUTPUT_BAM << "]" << endl;
  cerr << "  --output-bam-buffer-size     INT       read buffer size for writing output BAM [" << DEFAULT_OUTPUT_BAM_BUFFER_SIZE << "]" << endl;
  cerr << "  --output-bam-compress        BOOL      turn on output BAM compression [" << DEFAULT_OUTPUT_BAM_COMPRESS << "]" << endl;
  cerr << "  --skip-rg-suffix             STRING    ignore read groups matching suffix [\"" << DEFAULT_SKIP_RG_SUFFIX << "\"]" << endl;
  cerr << "  --max-map-qual               INT       if nonnegative, ignore reads with map qual bigger than this [\"" << DEFAULT_MAX_MAP_QUAL << "\"]" << endl;
  cerr << "  --min-map-qual               INT       if nonnegative, ignore reads with map qual smaller than this [\"" << DEFAULT_MIN_MAP_QUAL << "\"]" << endl;
  cerr << "  -h,--histogram-length        INT       max read length - will be n-flow/" << TYPICAL_FLOWS_PER_BASE << " if not positive [" << DEFAULT_HISTOGRAM_LENGTH << "]" << endl;
  cerr << "  --n-flow                     INT       max number of flows - will be " << TYPICAL_FLOWS_PER_BASE << "*histogram-length if not positive [" << DEFAULT_N_FLOW << "]" << endl;
  cerr << "  -m,--minimum-aq-length       INT       minimum AQ read length [" << DEFAULT_MINIMUM_AQ_LENGTH << "]" << endl;
  cerr << "  -b,--bc-adjust               BOOL      give credit to barcode bases, assumes barcodes have no errors [" << DEFAULT_BC_ADJUST << "]" << endl;
  cerr << "  -k,--key                     STRING    seq key - used for calculating system_snr, and removed when using -b option [" << DEFAULT_SEQ_KEY << "]" << endl;
  cerr << "  -a,--aq-error-rates          STRING    error rates for which to evaluate AQ lengths [" << DEFAULT_AQ_ERROR_RATES << "]" << endl;
  cerr << "  --evaluate-flow              BOOL      evaluate per-flow accuracy [" << DEFAULT_EVALUATE_FLOW << "]" << endl;
  cerr << "  --evaluate-hp                BOOL      evaluate homopolymer accuracy [" << DEFAULT_EVALUATE_HP << "]" << endl;
  cerr << "  --ignore-terminal-hp         BOOL      ignore first and last HPs when tracking HP accuracy [" << DEFAULT_IGNORE_TERMINAL_HP << "]" << endl;
  cerr << "  --evaluate-per-read-per-flow BOOL      evaluate per-read per-flow accuracy (big output!) [" << DEFAULT_EVALUATE_PER_READ_PER_FLOW << "]" << endl;
  cerr << "  --max-hp                     INT       max HP length for chip-wide summary [" << DEFAULT_MAX_HP << "]" << endl;
  cerr << "  --max-subregion-hp           INT       max HP length for regional summary [" << DEFAULT_SUBREGION_MAX_HP << "]" << endl;
  cerr << "  --n-threads                  INT       number of threads for analysis, set to 0 to use numCores() [" << DEFAULT_N_THREADS << "]" << endl;
  cerr << "  --threads-share-memory       BOOL      controls whether threads write results to private or common mem [" << DEFAULT_THREADS_SHARE_MEMORY << "]" << endl;
  cerr << endl;
  cerr << "Options for spatial stratification of results.  All 3 options must be used together." << endl;
  cerr << "  Each option specifies two comma-separated values in the form x,y" << endl;
  cerr << "  --chip-origin           INT,INT   zero-based coordinate origin of chip [" << DEFAULT_CHIP_ORIGIN << "]" << endl;
  cerr << "  --chip-dim              INT,INT   dimensions of chip [" << DEFAULT_CHIP_DIM << "]" << endl;
  cerr << "  --subregion-dim         INT,INT   dimensions of sub-regions for spatial stratification [" << DEFAULT_SUBREGION_DIM << "]" << endl;
  cerr << endl;
  cerr << "Debug options.  Use with care, may produce lots to stderr." << endl;
  cerr << "  Reads meeting all requested properties will be printed." << endl;
  cerr << "  --debug                    BOOL   Turns on debugging [" << DEFAULT_DEBUG << "]" << endl;
  cerr << "  --debug-error-flow         INT    Require read to have an error in a flow [" << DEFAULT_DEBUG_ERROR_FLOW << "]" << endl;
  cerr << "  --debug-aligned-flow       INT    Require read to be aligned in a flow [" << DEFAULT_DEBUG_ALIGNED_FLOW << "]" << endl;
  cerr << "  --debug-positive-ref-flow  INT    Require reference to have positive HP length in a flow [" << DEFAULT_DEBUG_POSITIVE_REF_FLOW  << "]" << endl;
}

// Metrics in ionstats_alignment.json should carry the following data:
//
// - Histogram of read lengths (copy of basecaller's "full" histogram) - done
// - Histogram of aligned lengths - done
// - Histogram of AQ## lengths - done
// - Genome name, genome version, mapper version, genome size - ???
// - Error rate by position: numerator and denominator - ??? (actually might be very easy)

void IonstatsAlignmentOptions::Initialize(int argc, const char *argv[]) {
  program_ = string(argv[0]) + " " + string(argv[1]);
  OptArgs opts;
  opts.ParseCmdLine(argc-1, argv+1);
  InitializeFromOptArgs(opts, program_);
}

void IonstatsAlignmentOptions::InitializeFromOptArgs(OptArgs &opts, const string &program_str) {
  program_ = program_str;

  opts.GetOption(input_bam_filename_, DEFAULT_INPUT_BAM,     'i', "input");
  opts.GetOption(chip_origin_,        DEFAULT_CHIP_ORIGIN,   '-', "chip-origin");
  opts.GetOption(chip_dim_,           DEFAULT_CHIP_DIM,      '-', "chip-dim");
  opts.GetOption(subregion_dim_,      DEFAULT_SUBREGION_DIM, '-', "subregion-dim");
  opts.GetOption(aq_error_rate_,      DEFAULT_AQ_ERROR_RATES, 'a', "aq-error-rates");

  help_                        = opts.GetFirstBoolean('-', "help",              DEFAULT_HELP);
  output_bam_filename_         = opts.GetFirstString ('-', "output-bam",                 DEFAULT_OUTPUT_BAM);
  output_bam_buffer_size_      = opts.GetFirstInt    ('-', "output-bam-buffer-size",     DEFAULT_OUTPUT_BAM_BUFFER_SIZE);
  output_bam_compress_         = opts.GetFirstBoolean('-', "output-bam-compress",        DEFAULT_OUTPUT_BAM_COMPRESS);
  output_json_filename_        = opts.GetFirstString ('o', "output",                     DEFAULT_OUTPUT_JSON);
  output_h5_filename_          = opts.GetFirstString ('-', "output-h5",                  DEFAULT_OUTPUT_H5);
  skip_rg_suffix_              = opts.GetFirstString ('-', "skip-rg-suffix",             DEFAULT_SKIP_RG_SUFFIX);
  max_map_qual_                = opts.GetFirstInt    ('-', "max-map-qual",               DEFAULT_MAX_MAP_QUAL);
  min_map_qual_                = opts.GetFirstInt    ('-', "min-map-qual",               DEFAULT_MIN_MAP_QUAL);
  histogram_length_            = opts.GetFirstInt    ('h', "histogram-length",           DEFAULT_HISTOGRAM_LENGTH);
  n_flow_                      = opts.GetFirstInt    ('-', "n-flow",                     DEFAULT_N_FLOW);
  minimum_aq_length_           = opts.GetFirstInt    ('m', "minimum-aq-length",          DEFAULT_MINIMUM_AQ_LENGTH);
  bc_adjust_                   = opts.GetFirstBoolean('b', "bc-adjust",                  DEFAULT_BC_ADJUST);
  seq_key_                     = opts.GetFirstString ('k', "key",                        DEFAULT_SEQ_KEY);
  evaluate_hp_                 = opts.GetFirstBoolean('-', "evaluate-hp",                DEFAULT_EVALUATE_HP);
  ignore_terminal_hp_          = opts.GetFirstBoolean('-', "ignore-terminal-hp",         DEFAULT_IGNORE_TERMINAL_HP);
  evaluate_flow_               = opts.GetFirstBoolean('-', "evaluate-flow",              DEFAULT_EVALUATE_FLOW);
  evaluate_per_read_per_flow_  = opts.GetFirstBoolean('-', "evaluate-per-read-per-flow", DEFAULT_EVALUATE_PER_READ_PER_FLOW);
  max_hp_                      = opts.GetFirstInt    ('-', "max-hp",                     DEFAULT_MAX_HP);
  max_subregion_hp_            = opts.GetFirstInt    ('-', "max-subregion-hp",           DEFAULT_SUBREGION_MAX_HP);
  debug_                       = opts.GetFirstBoolean('-', "debug",                      DEFAULT_DEBUG);
  debug_error_flow_            = opts.GetFirstInt    ('-', "debug-error-flow",           DEFAULT_DEBUG_ERROR_FLOW);
  debug_aligned_flow_          = opts.GetFirstInt    ('-', "debug-aligned-flow",         DEFAULT_DEBUG_ALIGNED_FLOW);
  debug_positive_ref_flow_     = opts.GetFirstInt    ('-', "debug-positive-ref-flow",    DEFAULT_DEBUG_POSITIVE_REF_FLOW);
  n_threads_                   = opts.GetFirstInt    ('-', "n-threads",                  DEFAULT_N_THREADS);
  threads_share_memory_        = opts.GetFirstBoolean('-', "threads-share-memory",       DEFAULT_THREADS_SHARE_MEMORY);


  if(evaluate_per_read_per_flow_) {
    evaluate_flow_ = true;
    evaluate_hp_ = true;
  }

  if(n_flow_ <= 0 && histogram_length_ > 0)
    n_flow_ = ceil(TYPICAL_FLOWS_PER_BASE*histogram_length_);
  else if(n_flow_ > 0 && histogram_length_ <= 0)
    histogram_length_ = ceil( (double) n_flow_ / (double) TYPICAL_FLOWS_PER_BASE);
  else if(n_flow_ <= 0 && histogram_length_ <= 0) {
    cerr << "ERROR: " << program_ << ": at least one of n-flow and histogram_length must be positive" << endl;
    exit(EXIT_FAILURE);
  }

  // If doing spatial stratification, check on the dimensions and assign names to regions
  if(checkDimensions(spatial_stratify_,chip_origin_,chip_dim_,subregion_dim_,n_col_subregions_,n_row_subregions_))
    exit(EXIT_FAILURE);
  n_subregions_ = n_col_subregions_ * n_row_subregions_;
  if(n_subregions_ > 0)
    assignRegionNames(region_name_,chip_origin_,chip_dim_,subregion_dim_,region_specific_origin_,region_specific_dim_);

  // Check and sort AQ error rates - later we used shortcuts that rely on the assumption they are sorted
  for(unsigned int i=0; i<aq_error_rate_.size(); ++i)
    if(aq_error_rate_[i] < 0 || aq_error_rate_[i] >= 1)
      cerr << "WARNING: " << program_ << ": bad value for aq-error-rates option, must be in range [0,1), value is " << aq_error_rate_[i] << endl;
  sort(aq_error_rate_.begin(), aq_error_rate_.end(), revSort);
  n_error_rates_ = aq_error_rate_.size();
  if(input_bam_filename_.size()==0) {
    WriteHelp();
    exit(EXIT_FAILURE);
  }

  // Check to make sure that if /dev/stdin was supplied as an input, in which case it should be the only input
  bool have_stdin=false;
  for(vector<string>::iterator it = input_bam_filename_.begin(); it != input_bam_filename_.end(); ++it)
    if(*it == "/dev/stdin")
      have_stdin=true;
  if(have_stdin && input_bam_filename_.size() > 1) {
    cerr << "ERROR: " << program_ << ": if /dev/stdin is supplied as an input, it must be the only input" << endl;
    exit(EXIT_FAILURE);
  }

  if((output_bam_filename_ != "") && (input_bam_filename_.size() > 1)) {
    cerr << "WARNING: using --bam-output option with more than one input BAM," << endl;
    cerr << "WARNING:   output BAM header will reflect only the first input BAM, use with caution" << endl;
  }
  if(output_bam_filename_ != "" && output_bam_buffer_size_ == 0) {
    cerr << "ERROR: " << program_ << ": when writing BAM output, buffer size must be positive" << endl;
    exit(EXIT_FAILURE);
  }

  qv_to_error_rate_.assign(256,0);
  for (unsigned int qv = 0; qv < qv_to_error_rate_.size(); qv++)
    qv_to_error_rate_[qv] =  pow(10.0,-0.1*(double)qv);

  if(n_threads_ == 0) {
    n_threads_ = numCores();
  } 
}


int IonstatsAlignmentBamReader::Initialize(const vector<string> &input_bam_filename, const string &program, const string &seq_key, const string &skip_rg_suffix, SamHeader &sam_header, RefVector &reference_data) {
  input_bam_filename_ = input_bam_filename;

  //
  // Do a quick first pass through all BAMs to read headers
  //
  for(input_bam_filename_it_=input_bam_filename_.begin(); input_bam_filename_it_ != input_bam_filename_.end(); ++input_bam_filename_it_) {
    // open BAM
    if (!input_bam_.Open(*input_bam_filename_it_)) {
      cerr << program << ": ERROR: cannot open " << *input_bam_filename_it_ << " for read" << endl;
      exit(EXIT_FAILURE);
    }

    // Return BAM header and reference data for the first input BAM file
    if(input_bam_filename_it_ == input_bam_filename_.begin() ) {
      sam_header     = input_bam_.GetHeader();
      reference_data = input_bam_.GetReferenceData();
    }

    // Initialize any possible new read groups
    map< string, int > temp_read_groups;
    map< string, string > temp_flow_orders;
    map< string, string > temp_key_bases;
    map< string, int > temp_key_len;
    unsigned int temp_max_flow_order_len=0;
    getReadGroupInfo(input_bam_,temp_read_groups,temp_flow_orders,temp_max_flow_order_len,temp_key_bases,temp_key_len,seq_key,skip_rg_suffix);
    // Add any new read groups
    for(map< string, int >::iterator temp_read_group_it=temp_read_groups.begin(); temp_read_group_it != temp_read_groups.end(); ++temp_read_group_it) {
      map< string, int >::iterator it = read_groups_.find(temp_read_group_it->first);
      if(it == read_groups_.end())
        read_groups_[temp_read_group_it->first] = 0;
    }
    // Add any new flow orders, confirm any repeated ones match
    for(map< string, string >::iterator temp_flow_order_it=temp_flow_orders.begin(); temp_flow_order_it != temp_flow_orders.end(); ++temp_flow_order_it) {
      map< string, string >::iterator it = flow_orders_.find(temp_flow_order_it->first);
      if(it == flow_orders_.end()) {
        // New one, add it in
        flow_orders_[temp_flow_order_it->first] = temp_flow_order_it->second;
      } else {
        // Repeated one, check it against existing entries
        if(flow_orders_[it->first] != temp_flow_order_it->second)
          cerr << program << ": WARNING: read group " << it->first << " is associated with more than one flow order, using first" << endl;
      }
    }
    // Add any new keys, confirm any repeated ones match
    for(map< string, string >::iterator temp_key_base_it=temp_key_bases.begin(); temp_key_base_it != temp_key_bases.end(); ++temp_key_base_it) {
      map< string, string >::iterator it = key_bases_.find(temp_key_base_it->first);
      if(it == key_bases_.end()) {
        // New one, add it in
        key_bases_[temp_key_base_it->first] = temp_key_base_it->second;
      } else {
        // Repeated one, check it against existing entries
        if(key_bases_[it->first] != temp_key_base_it->second)
          cerr << program << ": WARNING: read group " << it->first << " is associated with more than one key, using first" << endl;
      }
    }
    // Add any new key lengths, confirm any repeated ones match
    for(map< string, int >::iterator temp_key_len_it=temp_key_len.begin(); temp_key_len_it != temp_key_len.end(); ++temp_key_len_it) {
      map< string, int >::iterator it = key_len_.find(temp_key_len_it->first);
      if(it == key_len_.end()) {
        // New one, add it in
        key_len_[temp_key_len_it->first] = temp_key_len_it->second;
      } else {
        // Repeated one, check it against existing entries
        if(key_len_[it->first] != temp_key_len_it->second)
          cerr << program << ": WARNING: read group " << it->first << " is associated with more than one key length, using first" << endl;
      }
    }
    if(temp_max_flow_order_len > max_flow_order_len_)
      max_flow_order_len_ = temp_max_flow_order_len;
    
    // If reading from STDIN, then exit so we don't try to close & re-open
    if(*input_bam_filename_it_ == "/dev/stdin")
      return(EXIT_SUCCESS);

    input_bam_.Close();
  }

  // Then open the first BAM for reading
  input_bam_filename_it_=input_bam_filename_.begin();
  if (!input_bam_.Open(*input_bam_filename_it_)) {
    cerr << program << ": ERROR: cannot open " << *input_bam_filename_it_ << " for read" << endl;
    exit(EXIT_FAILURE);
  }

  return(EXIT_SUCCESS);
}

bool IonstatsAlignmentBamReader::GetNextAlignment(BamAlignment &alignment, string &program) {
  
  if (input_bam_.GetNextAlignment(alignment)) {
    // We got another read from the currently open BAM
    return(true);
  } else if(input_bam_filename_it_ == input_bam_filename_.end()) {
    // We have read all BAM files
    return(false);
  } else {
    // We just finished a BAM file and there may be more BAMs to read.
    // Keep moving on till we either reach the last BAM or till we get a read
    if(input_bam_.IsOpen())
      input_bam_.Close();
    bool return_status=false;
    while(++input_bam_filename_it_ != input_bam_filename_.end()) {
      if (!input_bam_.Open(*input_bam_filename_it_)) {
        cerr << program << ": ERROR: cannot open " << *input_bam_filename_it_ << " for read" << endl;
        exit(EXIT_FAILURE);
      }
      if (input_bam_.GetNextAlignment(alignment)) {
        return_status=true;
        break;
      }
    }
    return(return_status);
  }
}

int IonstatsAlignment(OptArgs &opts, const string &program_str)
{
  IonstatsAlignmentOptions opt;
  opt.InitializeFromOptArgs(opts, program_str);

  if (opt.InputBamFilename().empty() or opt.InputBamFilename().at(0).empty()) {
    opt.WriteHelp();
    return(EXIT_FAILURE);
  }
  if (opt.Help()) {
    opt.WriteHelp();
    return(EXIT_SUCCESS);
  }

  // Quick pass through all input BAMs to determine headers and get ready to read
  IonstatsAlignmentBamReader input_bam;
  SamHeader output_sam_header;
  RefVector output_reference_data;
  input_bam.Initialize(
    opt.InputBamFilename(),
    opt.Program(),
    opt.SeqKey(),
    opt.SkipRgSuffix(),
    output_sam_header,
    output_reference_data
  );

  // Initialize output BAM and write header using info returned from initialization of input BAM reader
  BamWriter output_bam;
  if(opt.OutputBamFilename() != "") {
    if(opt.OutputBamCompress())
      output_bam.SetCompressionMode(BamWriter::Compressed);
    else
      output_bam.SetCompressionMode(BamWriter::Uncompressed);
    output_bam.Open(opt.OutputBamFilename(), output_sam_header, output_reference_data);
  }


  // Initialize structures into which results will be accumulated
  vector< AlignmentSummary > alignment_summary(1);
  alignment_summary[0].Initialize(input_bam.ReadGroups(), opt);
  if(!opt.ThreadsShareMemory()) {
    alignment_summary.resize(opt.NThreads());
    for(unsigned int i=1; i<opt.NThreads(); ++i)
      alignment_summary[i] = alignment_summary[0];
  }

  // Set up input for the worker threads. Each thread gets its own AlignmentSummary object into
  // which to write its results, the other inputs are common across all the threads.
  vector< ProcessAlignmentContext > pac(opt.NThreads());
  pthread_mutex_t read_mutex;
  pthread_mutex_t write_mutex;
  pthread_mutex_t results_mutex;
  pthread_mutex_t region_mutex;
  pthread_mutex_init(&read_mutex, NULL);
  pthread_mutex_init(&write_mutex, NULL);
  pthread_mutex_init(&results_mutex, NULL);
  pthread_mutex_init(&region_mutex, NULL);
  for(unsigned int i=0; i<opt.NThreads(); ++i) {
    pac[i].input_bam = & input_bam;
    if(opt.ThreadsShareMemory())
      pac[i].alignment_summary = & (alignment_summary[0]);
    else
      pac[i].alignment_summary = & (alignment_summary[i]);
    pac[i].output_bam = & output_bam;
    pac[i].opt = & opt;
    pac[i].read_mutex = &read_mutex;
    pac[i].write_mutex = &write_mutex;
    pac[i].results_mutex = &results_mutex;
  }

  // Do the heavy work - process all alignment data
  pthread_t worker_id[opt.NThreads()];
  for (unsigned int worker = 0; worker < opt.NThreads(); worker++) {
    if (pthread_create(&worker_id[worker], NULL, processAlignments, &(pac[worker]))) {
      cerr << "ERROR: " << opt.Program() << ": problem starting thread" << endl;
      exit (EXIT_FAILURE);
    }
  }

  for (unsigned int worker = 0; worker < opt.NThreads(); worker++)
    pthread_join(worker_id[worker], NULL);

  pthread_mutex_destroy(&read_mutex);
  pthread_mutex_destroy(&write_mutex);
  pthread_mutex_destroy(&results_mutex);
  pthread_mutex_destroy(&region_mutex);

  if(!opt.ThreadsShareMemory())
    for(unsigned int i=1; i<opt.NThreads(); ++i)
      alignment_summary[0].MergeFrom(alignment_summary[i]);

  if(opt.OutputBamFilename() != "")
    output_bam.Close();

  alignment_summary[0].WriteWarnings(cerr, opt.Program(), opt.SkipRgSuffix(), opt.MinMapQual(), opt.MaxMapQual(), opt.SpatialStratify(), opt.EvaluateFlow(), opt.BcAdjust());

  // Fill depths for base_position data
  alignment_summary[0].FillBasePositionDepths();

  // Processing complete, write summary data
  writeIonstatsAlignmentJson(opt.OutputJsonFilename(), opt.AqErrorRate(), opt.BcAdjust(), opt.EvaluateFlow(), opt.NSubregions(), alignment_summary[0]);
  if(opt.OutputH5Filename() != "") {
    bool append_h5_file = opt.EvaluatePerReadPerFlow();
    writeIonstatsH5(opt.OutputH5Filename(), append_h5_file, opt.RegionName(), alignment_summary[0]);
  }

  return(EXIT_SUCCESS);
}

int writeIonstatsAlignmentJson(
  const string &json_filename,
  const vector<double> &aq_error_rate,
  bool bc_adjust,
  bool evaluate_flow,
  uint Num_regions,
  AlignmentSummary alignment_summary
) {

  Json::Value output_json(Json::objectValue);

  //output_json["meta"]["creation_date"] = get_time_iso_string(time(NULL));
  output_json["meta"]["format_name"] = "ionstats_alignment";
  output_json["meta"]["format_version"] = "1.0";
  
  // Called & aligned lengths
  alignment_summary.SystemSnr().SaveToJson(output_json);
  alignment_summary.QvHistogram().SaveToJson(output_json);
  alignment_summary.CalledHistogram().SaveToJson(output_json["full"]);
  alignment_summary.TotalInsertHistogram().SaveToJson(output_json["insert"]);
  alignment_summary.TotalQ17Histogram().SaveToJson(output_json["Q17"]);
  alignment_summary.TotalQ20Histogram().SaveToJson(output_json["Q20"]);
  alignment_summary.AlignedHistogram().SaveToJson(output_json["aligned"]);
  vector< ReadLengthHistogram > aq_histogram = alignment_summary.AqHistogram();
  for(unsigned int i=0; i<aq_error_rate.size(); ++i) {
    int phred_int = toPhred(aq_error_rate[i]);
    string phred_string = static_cast<ostringstream*>( &(ostringstream() << phred_int) )->str();
    aq_histogram[i].SaveToJson(output_json["AQ" + phred_string]);
  }
  //regional aqs
  for (unsigned int k=0; k< Num_regions;k++){
      string x_string = static_cast<ostringstream*>( &(ostringstream() << alignment_summary.GetRegionalSummary()[k].Origx()) )->str();
      string y_string = static_cast<ostringstream*>( &(ostringstream() << alignment_summary.GetRegionalSummary()[k].Origy()) )->str();
      
   for(unsigned int i=0; i<aq_error_rate.size(); ++i) {
    int phred_int = toPhred(aq_error_rate[i]);
    string phred_string = static_cast<ostringstream*>( &(ostringstream() << phred_int) )->str();
    alignment_summary.GetRegionalSummary()[k].aq_histogram_[i].SummarizeToJson(output_json["Regional"]["(" + x_string + "," + y_string + ")"]["AQ" + phred_string]);
  }   
      
      
  }
  

  // Called & aligned lengths including barcodes
  if(bc_adjust) {
    // We put any barcoded results in their own "WithBarcode" section
    alignment_summary.CalledHistogramBc().SaveToJson(output_json["WithBarcode"]["full"]);
    alignment_summary.AlignedHistogramBc().SaveToJson(output_json["WithBarcode"]["aligned"]);
    vector< ReadLengthHistogram > &aq_histogram_bc = alignment_summary.AqHistogramBc();
    for(unsigned int i=0; i<aq_error_rate.size(); ++i) {
      int phred_int = toPhred(aq_error_rate[i]);
      string phred_string = static_cast<ostringstream*>( &(ostringstream() << phred_int) )->str();
      aq_histogram_bc[i].SaveToJson(output_json["WithBarcode"]["AQ" + phred_string]);
    }
  }

  // Per-base error data
  alignment_summary.BasePositionErrorCount().SaveToJson(output_json["error_by_position"]);
  alignment_summary.BasePosition().SaveToJson(output_json["by_base"]);

  // Per-flow error data
  if(evaluate_flow)
    alignment_summary.FlowPosition().SaveToJson(output_json["by_flow"]);

  ofstream out(json_filename.c_str(), ios::out);
  if (out.good()) {
    out << output_json.toStyledString();
    return 0;
  } else {
    fprintf(stderr, "ERROR: unable to write to '%s'\n", json_filename.c_str());
    return 1;
  }

  return 0;
}

void AddToReadLengthHistogramFromJson(const Json::Value &input_json, const string &var_name, ReadLengthHistogram &hist) {
  if(input_json.isMember(var_name)) {
    ReadLengthHistogram temp;
    temp.LoadFromJson(input_json[var_name]);
    hist.MergeFrom(temp);
  }
}

void AddToSimpleHistogramFromJson(const Json::Value &input_json, const string &var_name, SimpleHistogram &hist, bool &found) {
  if(input_json.isMember(var_name)) {
    found = true;
    SimpleHistogram temp;
    temp.LoadFromJson(input_json[var_name]);
    hist.MergeFrom(temp);
  }
}

int IonstatsAlignmentReduce(const string& output_json_filename, const vector<string>& input_jsons)
{

  // Called & aligned lengths
  BaseQVHistogram qv_histogram;
  MetricGeneratorSNR system_snr;
  ReadLengthHistogram called_histogram;
  ReadLengthHistogram total_insert_histo;
  ReadLengthHistogram total_Q17_histo;
  ReadLengthHistogram total_Q20_histo;
  ReadLengthHistogram aligned_histogram;
  map< string, ReadLengthHistogram > AQ_histogram;

  // Per-base error data
  bool have_base_data=false;
  SimpleHistogram error_by_position;
  ErrorData by_base;

  // Per-flow error data
  bool have_flow_data=false;
  ErrorData flow_position;

  for (unsigned int input_idx = 0; input_idx < input_jsons.size(); ++input_idx) {
    ifstream in(input_jsons[input_idx].c_str(), ifstream::in);
    if (!in.good()) {
      fprintf(stderr, "[ionstats] ERROR: cannot open %s\n", input_jsons[0].c_str());
      return 1;
    }
    Json::Value current_input_json;
    in >> current_input_json;
    in.close();

    // Called & aligned lengths
    BaseQVHistogram current_qv_histogram;
    current_qv_histogram.LoadFromJson(current_input_json);
    qv_histogram.MergeFrom(current_qv_histogram);
    MetricGeneratorSNR current_system_snr;
    current_system_snr.LoadFromJson(current_input_json);
    system_snr.MergeFrom(current_system_snr);
    ReadLengthHistogram current_called_histogram;
    current_called_histogram.LoadFromJson(current_input_json["full"]);
    called_histogram.MergeFrom(current_called_histogram);
    ReadLengthHistogram current_total_insert_histo;
    current_total_insert_histo.LoadFromJson(current_input_json["insert"]);
    total_insert_histo.MergeFrom(current_total_insert_histo);
    ReadLengthHistogram current_total_Q17_histo;
    current_total_Q17_histo.LoadFromJson(current_input_json["Q17"]);
    total_Q17_histo.MergeFrom(current_total_Q17_histo);
    ReadLengthHistogram current_total_Q20_histo;
    current_total_Q20_histo.LoadFromJson(current_input_json["Q20"]);
    total_Q20_histo.MergeFrom(current_total_Q20_histo);
    AddToReadLengthHistogramFromJson(current_input_json, "aligned", aligned_histogram);
    Json::Value::Members member_name = current_input_json.getMemberNames();
    for(unsigned int i=0; i<member_name.size(); ++i) {
      if(isAqLength(member_name[i])) {
        map< string, ReadLengthHistogram >::iterator it = AQ_histogram.find(member_name[i]);
        if(it == AQ_histogram.end()) {
          ReadLengthHistogram temp;
          AQ_histogram[member_name[i]] = temp;
        }
        AddToReadLengthHistogramFromJson(current_input_json, member_name[i], AQ_histogram[member_name[i]]);
      }
    }

    // Per-base error data
    AddToSimpleHistogramFromJson(current_input_json, "error_by_position", error_by_position,   have_base_data);
    if(current_input_json.isMember("by_base"))
      by_base.MergeFrom(current_input_json["by_base"],have_base_data);

    // Per-flow error data
    if(current_input_json.isMember("by_flow"))
      flow_position.MergeFrom(current_input_json["by_flow"],have_flow_data);
  }

  Json::Value output_json(Json::objectValue);
  //output_json["meta"]["creation_date"] = get_time_iso_string(time(NULL));
  output_json["meta"]["format_name"] = "ionstats_alignment";
  output_json["meta"]["format_version"] = "1.0";

  // Called & aligned lengths
  system_snr.SaveToJson(output_json);
  qv_histogram.SaveToJson(output_json);
  called_histogram.SaveToJson(output_json["full"]);
  total_insert_histo.SaveToJson(output_json["insert"]);
  total_Q17_histo.SaveToJson(output_json["Q17"]);
  total_Q20_histo.SaveToJson(output_json["Q20"]);
  aligned_histogram.SaveToJson(output_json["aligned"]);
  map< string, ReadLengthHistogram >::iterator it;
  for(it=AQ_histogram.begin(); it != AQ_histogram.end(); ++it)
    it->second.SaveToJson(output_json[it->first]);

  // Per-base error data
  if(have_base_data) {
    error_by_position.SaveToJson(output_json["error_by_position"]);
    by_base.SaveToJson(output_json["by_base"]);
  }

  // Per-flow error data
  if(have_flow_data)
    flow_position.SaveToJson(output_json["by_flow"]);

  ofstream out(output_json_filename.c_str(), ios::out);
  if (out.good()) {
    out << output_json.toStyledString();
    return 0;
  } else {
    fprintf(stderr, "ERROR: unable to write to '%s'\n", output_json_filename.c_str());
    return 1;
  }
}




void parseMD(const string &MD_tag, vector<char> &MD_op, vector<int> &MD_len, vector<string> &MD_seq) {
  MD_op.clear();
  MD_len.clear();
  MD_seq.clear();
  for (const char *MD_ptr = MD_tag.c_str(); *MD_ptr;) {
    int item_length = 0;
    string item_seq = "";
    if (*MD_ptr >= '0' and *MD_ptr <= '9') {    // Its a match
      MD_op.push_back('M');
      for (; *MD_ptr and *MD_ptr >= '0' and *MD_ptr <= '9'; ++MD_ptr)
        item_length = 10*item_length + *MD_ptr - '0';
    } else {
      if (*MD_ptr == '^') {                     // Its a deletion
        MD_ptr++;
        MD_op.push_back('D');
      } else                                    // Its a substitution
        MD_op.push_back('X');
      for (; *MD_ptr and *MD_ptr >= 'A' and *MD_ptr <= 'Z'; ++MD_ptr) {
        item_length++;
        item_seq += *MD_ptr;
      }
    }
    MD_len.push_back(item_length);
    MD_seq.push_back(item_seq);
  }
}



// -----------------------------------------------------------------------
// Update flow_idx to bring it to the current incorporating flow
// after function call flow_idx is pointing AT ideal incorporating flow of read_bases[read_idx]
void flowCatchup(unsigned int &flow_idx, const string &flow_order, unsigned int read_idx, const string &read_bases) {
  if(read_idx >= read_bases.size()) {
    return;
  } else {
    char current_base = read_bases[read_idx];
    while( current_base != flow_order[flow_idx % flow_order.size()] )
      flow_idx++;
  }
}

// -----------------------------------------------------------------------
// Modifies flow_idx by stepping along n_advance bases in the read
// Modifies flow_space_incorporation by adding flows in which there is an incorporation
// after function call flow_idx is internally pointing AT OR BEFORE ideal incorporating flow of read_bases[read_idx]
// (without returning read_idx, so AT OR AFTER w.r.t read_bases[read_idx] outside of function)!

void flowAdvance(
  unsigned int n_advance,
  unsigned int &flow_idx,
  const string &flow_order,
  unsigned int read_idx,
  const string &read_bases,
  vector <uint16_t> &flow_space_incorporation
) {
  assert(n_advance <= (read_bases.size()-read_idx)); // Make sure we're not being asked to advance beyond the end of the read
  while(n_advance > 0) {
    char current_base = read_bases[read_idx];
    if(current_base == flow_order[flow_idx % flow_order.size()]) {
      // there is an incorporation
      read_idx++;
      n_advance--;
      // check if we need to record an incorporation in the flow
      if(flow_space_incorporation.size()==0 || flow_space_incorporation.back() != flow_idx)
        flow_space_incorporation.push_back(flow_idx);
      // increase flow if an HP was completed
      if((read_idx < read_bases.size()) && (read_bases[read_idx] != current_base))
        flow_idx++;
    } else {
      // no incorporation, proceed to the next flow
      flow_idx++;
    }
  }
}

// -------------------------------------------------------------------
// Modifies flow_idx by stepping along n_advance bases in the read
//
void flowAdvanceToNextHP(
  unsigned int n_advance,
  unsigned int &flow_idx,
  const string &flow_order,
  unsigned int read_idx,
  const string &read_bases
) {
  assert(n_advance <= (read_bases.size()-read_idx)); // Make sure we're not being asked to advance beyond the end of the read
  while(n_advance > 0) {
    char current_base = read_bases[read_idx];
    if(current_base == flow_order[flow_idx % flow_order.size()]) {
      // there is an incorporation
      read_idx++;
      n_advance--;
      // increase flow if an HP was completed
      if((read_idx < read_bases.size()) && (read_bases[read_idx] != current_base)) {
        char next_base = read_bases[read_idx];
        while(next_base != flow_order[flow_idx % flow_order.size()])
          flow_idx++;
      }
    } else {
      // no incorporation, proceed to the next flow
      flow_idx++;
    }
  }
}


//
// parse alignment function - this is where the BAM record is analyzed and we
// compute the assignment of errors to base position, flow position, HP errors, etc
//
int parseAlignment(
  BamAlignment &          alignment,               // The BAM record to be parsed
  ReadAlignmentErrors &   base_space_errors,       // Returns the errors in base space
  ReadAlignmentErrors &   flow_space_errors,       // If evaluate_flow is TRUE, returns the errors in flow space
  map<string, string> &   flow_orders,             // Specifies the flow orders for each read group
  string &                read_group,              // Used to return the read group for the BAM record
  const map<char,char> &  reverse_complement_map,  // Defines the rev comp for every base
  bool                    evaluate_flow,           // Specifies if flows should be analyzed & returned
  unsigned int            max_flows,               // Max flow to analyze
  bool                    evaluate_hp,             // Specifies if per-HP results should be analyzed & returned
  bool &                  invalid_read_bases,      // Will return TRUE if invalid (non-IUPAC) bases encountered in the read
  bool &                  invalid_ref_bases,       // Will return TRUE if invalid (non-IUPAC) bases encountered in the ref
  bool &                  invalid_cigar,           // Will return TRUE if lengths implied by CIGAR and by alignment do not match
  vector<char> &          ref_hp_nuc,              // If evaulate_hp is TRUE, specifies ref hp nucleotides
  vector<uint16_t> &      ref_hp_len,              // If evaulate_hp is TRUE, specifies ref hp lengths
  vector<int16_t> &       ref_hp_err,              // If evaulate_hp is TRUE, specifies the read HP length errors (positive means overcall)
  vector<uint16_t> &      ref_hp_flow,             // If evaulate_hp and evaluate_flow are TRUE, specifies flow to which each ref hp aligns
  vector<uint16_t> &      zeromer_insertion_flow,  // If evaluate_flow and evaluate_hp are TRUE, specifies all flows with an insertion in a reference zeromer
  vector<uint16_t> &      zeromer_insertion_len    // If evaluate_flow and evaluate_hp are TRUE, specifies the lengths of the zeromer insertions
) {

  if(hasInvalidCigar(alignment))
    invalid_cigar=true;

  // *** Step 1: Load basic read & alignment information


  // Get read bases and check them
  string read_bases = alignment.QueryBases;
  if(alignment.IsReverseStrand())
    assert(!reverse_complement(read_bases,reverse_complement_map));
  bool ambiguous_read_bases = false;
  checkBases(read_bases,ambiguous_read_bases,invalid_read_bases);

  // The output of warnings and accounting of number of errors is handled upwards
  if(invalid_read_bases || invalid_cigar)
    return(1);
  if(ambiguous_read_bases)
    evaluate_flow=false;

  // Parse MD tag to extract MD_op and MD_len
  string MD_tag;
  assert(alignment.HasTag("MD"));
  alignment.GetTag("MD",MD_tag);
  vector<char>   MD_op;
  vector<int>    MD_len;
  vector<string> MD_seq;
  MD_op.reserve(1024);  // Type of MD operation
  MD_len.reserve(1024); // Number of bases for operation
  MD_seq.reserve(1024); // Bases for snp and deletion operations
  parseMD(MD_tag,MD_op,MD_len,MD_seq);
  if(alignment.IsReverseStrand()) {
    for(vector<string>::iterator it=MD_seq.begin(); it != MD_seq.end(); ++it)
      reverse_complement(*it,reverse_complement_map);
  }

  // Initialize data related to per-flow summary: ZF tag gives flow corresponding to first template base in read
  string flow_order = "";
  uint32_t flow_idx = 0;
  alignment.GetTag("RG",read_group);
  if(evaluate_flow) {
    evaluate_flow=false;
    if ((read_group != "") && getTagZF(alignment,flow_idx)) {
      map< string, string >::iterator it = flow_orders.find(read_group);
      if(it != flow_orders.end()) {
        flow_order = it->second;
        evaluate_flow=true;
      }
    }
  }

  // *** Step 2: Create reference from alignment info and create per-HP summary info

  // Initialize data required for per-HP summary
  string ref_bases = "";
  vector<char> read_hp_nuc;
  vector<uint16_t> read_hp_len;
  vector<uint16_t> ref_hp_cum_len,read_hp_cum_len;
  bool ambiguous_ref_bases = false;
  if(evaluate_hp) {
    ref_hp_err.clear();
    ref_hp_flow.clear();
    zeromer_insertion_flow.clear();
    zeromer_insertion_len.clear();
    ref_bases = getReferenceBases(read_bases, alignment.CigarData, MD_op, MD_len, MD_seq, alignment.IsReverseStrand(), reverse_complement_map);
    checkBases(ref_bases,ambiguous_ref_bases,invalid_ref_bases);
    if(invalid_ref_bases) {
      evaluate_hp = false;
    } else {
      getHpBreakdown(read_bases, read_hp_nuc, read_hp_len, read_hp_cum_len);
      getHpBreakdown(ref_bases,  ref_hp_nuc,  ref_hp_len,  ref_hp_cum_len );
    }
  }

  // *** Step 3:

  // Synchronously scan through Cigar and MD to determine type & positions of errors
  // We always proceed from 5' to 3' in the original read, so for reverse strand alignments
  // we will be going backwards through cigar and MD
  int increment    = alignment.IsReverseStrand() ? -1 : 1;
  int MD_idx       = alignment.IsReverseStrand() ? MD_op.size()-1 : 0;
  int cigar_idx    = alignment.IsReverseStrand() ? alignment.CigarData.size()-1 : 0;
  int read_idx     = 0;
  int ref_idx      = 0;
  unsigned int read_hp_idx  = 0;
  unsigned int ref_hp_idx   = 0;
  int stored_read_match_count = 0; // tracks the number of read bases so far that match the current reference HP
  bool alignmentStarted = false;
  base_space_errors.Initialize();
  flow_space_errors.Initialize();
  vector<uint16_t> & flow_space_incorporations = flow_space_errors.inc();
  if(evaluate_flow)
    flow_space_errors.SetHaveData();

  while (cigar_idx < (int) alignment.CigarData.size() and cigar_idx >= 0) {

    // Advance cigar if required
    if (alignment.CigarData[cigar_idx].Length == 0 or alignment.CigarData[cigar_idx].Type == 'H') {
      cigar_idx += increment;
      continue;
    }

    // handle soft-clipping in cigar
    if (alignment.CigarData[cigar_idx].Type == 'S') {
    
      unsigned int nclip = alignment.CigarData[cigar_idx].Length;
      // ZF tag is the (0-based) in-phase flow corresponding to the first template base in the read
      // If the alignment has soft clipped bases at the 5' end, we need to advance flow_idx
      int to_base = read_idx + nclip;
      while (read_idx < to_base and read_idx < (int)read_bases.length()) {
    	read_idx++;
    	if (evaluate_flow)
          flowCatchup(flow_idx,flow_order,read_idx,read_bases);
      }
      // Increment homopolymer index
      if(evaluate_hp) {
        while(read_hp_idx < read_hp_cum_len.size() && read_hp_cum_len[read_hp_idx] <= read_idx)
          read_hp_idx++;
      }
      cigar_idx += increment;
      continue;
    }

    // Advance MD if required
    if (MD_idx < (int) MD_op.size() and MD_idx >= 0 and MD_len[MD_idx] == 0) {
      MD_idx += increment;
      continue;
    }

    if(!alignmentStarted) {
      alignmentStarted = true;
      base_space_errors.SetFirst(read_idx);
      if(evaluate_flow)
        flow_space_errors.SetFirst(flow_idx);
    }

    if (alignment.CigarData[cigar_idx].Type == 'M' and MD_idx < (int) MD_op.size() and MD_idx >= 0 and MD_op[MD_idx] == 'M') {
      // Perfect match
      int advance = min((int)alignment.CigarData[cigar_idx].Length, MD_len[MD_idx]);
      if(evaluate_hp)
        hpAdvance(ALIGN_MATCH,advance,evaluate_flow,flow_idx,flow_order,read_bases,ref_bases,max_flows,read_idx, read_hp_nuc, read_hp_len, read_hp_cum_len, ref_idx, ref_hp_nuc, ref_hp_len, ref_hp_cum_len, stored_read_match_count, read_hp_idx, ref_hp_idx, ref_hp_err, ref_hp_flow, zeromer_insertion_flow, zeromer_insertion_len);
      if(evaluate_flow)
        flowAdvance(advance,flow_idx,flow_order,read_idx,read_bases,flow_space_incorporations);
      read_idx  += advance;
      ref_idx   += advance;
      alignment.CigarData[cigar_idx].Length -= advance;
      MD_len[MD_idx] -= advance;
    } else if (alignment.CigarData[cigar_idx].Type == 'I') {
      // Insertion (read has a base, reference doesn't)
      int advance = alignment.CigarData[cigar_idx].Length;
      if(evaluate_hp)
        hpAdvance(ALIGN_INS,advance,evaluate_flow,flow_idx,flow_order,read_bases,ref_bases,max_flows,read_idx,read_hp_nuc,read_hp_len,read_hp_cum_len,ref_idx,ref_hp_nuc,ref_hp_len,ref_hp_cum_len,stored_read_match_count,read_hp_idx,ref_hp_idx,ref_hp_err,ref_hp_flow, zeromer_insertion_flow, zeromer_insertion_len);
      for (int cnt = 0; cnt < advance; ++cnt) {
        if(evaluate_flow) {
          flowCatchup(flow_idx,flow_order,read_idx,read_bases);
          flow_space_errors.AddIns(flow_idx);
          flowAdvance(1,flow_idx,flow_order,read_idx,read_bases,flow_space_incorporations); // ???
        }
        base_space_errors.AddIns(read_idx);
        read_idx++;
      }
      alignment.CigarData[cigar_idx].Length -= advance;
    } else if (alignment.CigarData[cigar_idx].Type == 'D' and MD_idx < (int) MD_op.size() and MD_idx >= 0 and MD_op[MD_idx] == 'D') {
      // Deletion (reference has a base, read doesn't)
      // ?? Shouldn't we have a problem if below two lengths don't agree ??
      assert((int)alignment.CigarData[cigar_idx].Length == MD_len[MD_idx]);
      int advance = MD_len[MD_idx];
      if(evaluate_hp || evaluate_flow) {
        char next_read_base = (read_idx < (int) read_bases.length()) ? read_bases[read_idx] : 'N';
        char prev_read_base = (read_idx > 0) ? read_bases[read_idx-1] : 'N';
        if(evaluate_flow) {
          // Decrement flow_idx if the first base of the deletion is the same as the HP that was just completed
          if(flow_idx > 0 && MD_seq[MD_idx][0] == prev_read_base)
            --flow_idx;
        }
        if(evaluate_hp)
          hpAdvance(ALIGN_DEL,advance,evaluate_flow,flow_idx,flow_order,read_bases,ref_bases,max_flows,read_idx,read_hp_nuc,read_hp_len,read_hp_cum_len,ref_idx,ref_hp_nuc,ref_hp_len,ref_hp_cum_len,stored_read_match_count,read_hp_idx,ref_hp_idx,ref_hp_err,ref_hp_flow, zeromer_insertion_flow, zeromer_insertion_len);
        if(evaluate_flow) {
          // Try to assign deleted bases to the flows in which they would have been sequenced
          vector<uint16_t> del_flow;
          unsigned int impossible_deletions = assignDeletionsToFlows(del_flow,flow_idx,prev_read_base,next_read_base,flow_order,advance,MD_seq[MD_idx],max_flows);
          flow_space_errors.AddDel(del_flow);
        }
      }
      base_space_errors.AddDel(read_idx,advance);
      ref_idx += advance;
      alignment.CigarData[cigar_idx].Length = 0;
      MD_len[MD_idx] = 0;
    } else if (MD_idx < (int) MD_op.size() and MD_idx >= 0 and MD_op[MD_idx] == 'X') {
      // Substitution
      int advance = min((int)alignment.CigarData[cigar_idx].Length, MD_len[MD_idx]);
      if(evaluate_hp)
        hpAdvance(ALIGN_SUB,advance,evaluate_flow,flow_idx,flow_order,read_bases,ref_bases,max_flows,read_idx,read_hp_nuc,read_hp_len,read_hp_cum_len,ref_idx,ref_hp_nuc,ref_hp_len,ref_hp_cum_len,stored_read_match_count,read_hp_idx,ref_hp_idx,ref_hp_err,ref_hp_flow, zeromer_insertion_flow, zeromer_insertion_len);
      for (int cnt = 0; cnt < advance; ++cnt) {
        // Check if the substitution is an error (as opposed to a matching ambiguity code)
        bool is_error = !compatible_bases(read_bases[read_idx],MD_seq[MD_idx][cnt]);
        if(is_error)
          base_space_errors.AddSub(read_idx);
        if(evaluate_flow) {
          flowCatchup(flow_idx,flow_order,read_idx,read_bases);
          if(is_error)
            flow_space_errors.AddSub(flow_idx);
        }
        read_idx++;
        ref_idx++;
        if(evaluate_flow)
          flowCatchup(flow_idx,flow_order,read_idx,read_bases);
      }
      alignment.CigarData[cigar_idx].Length -= advance;
      MD_len[MD_idx] -= advance;
    } else {
      cerr << "WARNING: Unexpected CIGAR/MD combination for read " << alignment.Name << "(" << alignment.CigarData[cigar_idx].Type << ", ";
      if(MD_idx < (int) MD_op.size() and MD_idx >= 0)
        cerr << MD_op[MD_idx];
      else
        cerr << "NA";
      cerr << ")" << endl;
      return(1);
    }
    base_space_errors.SetLast(read_idx-1);
    if(evaluate_flow)
      flow_space_errors.SetLast(flow_idx);
  }
  base_space_errors.SetLen(alignment.Length);

  // Check to make sure we have accounted properly
  assert(read_idx == alignment.Length);

  // If there were any ambiguities in the reference bases then we need to modify any HP evaluation
  if(ambiguous_ref_bases && evaluate_hp) {
    // Determine which HPs must go
    unsigned int n_hp=ref_hp_nuc.size();
    vector<bool> to_delete(n_hp,false);
    for(unsigned int i=0; i<n_hp; ++i) {
      if(!is_unambiguous_base(ref_hp_nuc[i])) {
        to_delete[i] = true;
        if((i > 0) && compatible_bases(ref_hp_nuc[i],ref_hp_nuc[i-1]))
          to_delete[i-1] = true;
        if((i+1 < n_hp) && compatible_bases(ref_hp_nuc[i],ref_hp_nuc[i+1]))
          to_delete[i+1] = true;
      }
    }
    // Delete the offending HPs, scanning backwards through each vector
    for(int i=n_hp-1; i>=0; --i) {
      if(to_delete[i]) {
        unsigned int delete_stop=i+1;
        unsigned int delete_start=i;
        while((i > 0) && to_delete[--i])
          delete_start--;
        ref_hp_nuc.erase (ref_hp_nuc.begin() +delete_start, ref_hp_nuc.begin() +delete_stop);
        ref_hp_len.erase (ref_hp_len.begin() +delete_start, ref_hp_len.begin() +delete_stop);
        ref_hp_err.erase (ref_hp_err.begin() +delete_start, ref_hp_err.begin() +delete_stop);
        if(evaluate_flow)
          ref_hp_flow.erase(ref_hp_flow.begin()+delete_start, ref_hp_flow.begin()+delete_stop);
      }
    }
  }

  return(0);
}

void getReadGroupInfo(const BamReader &input_bam, map< string, int > &read_groups, map< string, string > &flow_orders, unsigned int &max_flow_order_len, map< string, string > &key_bases, map< string, int > &key_len, const string &seq_key, const string &skip_rg_suffix) {
  flow_orders.clear();
  key_bases.clear();
  key_len.clear();
  max_flow_order_len=0;
  int seq_key_len = seq_key.length();
  SamHeader samHeader = input_bam.GetHeader();
  if(samHeader.HasReadGroups()) {
    SamReadGroupDictionary sam_groups = samHeader.ReadGroups;
    for( SamReadGroupIterator it = sam_groups.Begin(); it != sam_groups.End(); ++it) {
      read_groups[it->ID] = 0;
      if((skip_rg_suffix != "") && (it->ID != "")) {
        int pos = it->ID.rfind(skip_rg_suffix);
        if((pos != (int) std::string::npos) && (pos == (int)(it->ID.length() - skip_rg_suffix.length())))
          continue;
      }
      if(it->HasID()) {
        if(it->HasFlowOrder()) {
          flow_orders[it->ID] = it->FlowOrder;
          if(it->FlowOrder.length() > max_flow_order_len)
            max_flow_order_len = it->FlowOrder.length();
        }
        if(it->HasKeySequence()) {
          key_bases[it->ID] = it->KeySequence;
          key_len[it->ID] = it->KeySequence.length() - seq_key_len;
        }
      }
    }
  }
}

void computeAQ(vector<int> &aq_length, vector<double> &aq_error_rate, ReadAlignmentErrors &base_space_errors) {

  // Initialize return results
  if(aq_length.size() != aq_error_rate.size())
    aq_length.resize(aq_error_rate.size());
  for(unsigned int i=0; i<aq_length.size(); ++i)
    aq_length[i] = 0;

  // Quit if unaligned
  uint16_t aligned_len = base_space_errors.AlignedLen();
  if(aligned_len == 0)
    return;

  // Make vectors of error positions & cumulative error counts
  const vector<uint16_t> &err_pos   = base_space_errors.err();
  const vector<uint16_t> &err_count = base_space_errors.err_len();
  vector<uint16_t> cumulative_err_count(err_count.size());
  if(err_count.size() > 0) {
    cumulative_err_count[0] = err_count[0];
    for(unsigned int i=1; i<err_count.size(); ++i)
      cumulative_err_count[i] = err_count[i] + cumulative_err_count[i-1];
  }

  // Handle case of perfect alignment
  if(err_count.size() == 0) {
    for(unsigned int i=0; i<aq_length.size(); ++i)
      aq_length[i] = aligned_len;
    return;
  }
  
  // Scan from last position backwards to find alignment lengths with sufficiently low error rates
  unsigned int iRate=0;
  unsigned int nRates=aq_error_rate.size();
  double epsilon = 1e-10;
  bool first=true;
  uint16_t this_error_pos=0;
  uint16_t additional_error_free_bases=0;
  for(int iErr=cumulative_err_count.size()-1; (iErr >= 0) && (iRate < nRates); --iErr) {
    // determine the number of error-free bases after the error being considered
    this_error_pos = err_pos[iErr];
    if(first) {
      first=false;
      additional_error_free_bases = base_space_errors.last()-this_error_pos;
    } else {
      additional_error_free_bases = err_pos[iErr+1]-1-this_error_pos;
    }

    // evaluate error rate and store read lengths where error rate is good enough
    uint16_t this_length = err_pos[iErr] + additional_error_free_bases + 1;
    double this_err_rate = ((double) cumulative_err_count[iErr]) / ((double) this_length);
    while((iRate < nRates) && (this_err_rate <= aq_error_rate[iRate]+epsilon)) {
      aq_length[iRate++] = this_length;
    }
  }

  // Lastly, need to consider if there is an error-free stretch before the first error
  this_error_pos=err_pos[0];
  if(this_error_pos > 0) {
    while((iRate < nRates))
      aq_length[iRate++] = this_error_pos;
  }

  return;
}

// Return Phred score corresponding to error rate e, capping at IONSTATS_BIGGEST_PHRED.
// This is not a terribly efficient implementation, not optimized for heavy usage
int toPhred(double e) {
  double smallest_err = pow(10,(float) IONSTATS_BIGGEST_PHRED / -10.0f );
  if(e <= smallest_err) {
    return(IONSTATS_BIGGEST_PHRED);
  } else {
    return((int) round(-10*log10(e)));
  }
}

void getBarcodeResults(BamAlignment &alignment, map< string, int > &key_len, int &bc_bases, int &bc_errors) {
  bc_bases=0;
  bc_errors=0;

  string RG_tag = "";
  if (alignment.GetTag("RG",RG_tag)) {
    map< string, int >::iterator it = key_len.find(RG_tag);
    if(it != key_len.end()) {
      bc_bases = it->second;
      alignment.GetTag("XB", bc_errors); 
      if(bc_errors > bc_bases)
        bc_errors = bc_bases;
    }
  }
}

void scoreBarcodes(ReadAlignmentErrors &base_space_errors, int bc_bases, int bc_errors, vector<double> &aq_error_rate, int minimum_aq_length, vector<int> &aq_length, AlignmentSummary * alignment_summary) {

  // Only compute updated AQ lengths if there were some barcoded bases to examine
  if(bc_bases > 0) {
    base_space_errors.ShiftPositions(bc_bases);
    if(bc_errors > 0) {
      base_space_errors.AddDel(bc_bases,bc_errors,false);
      base_space_errors.ConsolidateErrors();
    }
    computeAQ(aq_length,aq_error_rate,base_space_errors);
  }

  // Accumulate the barcoded statistics
  (*alignment_summary).AddAlignedLengthBc(base_space_errors.AlignedLen());
  for(unsigned int i=0; i < aq_error_rate.size(); ++i) {
    if(aq_length[i] >= minimum_aq_length)
      (*alignment_summary).AddAqLengthBc(aq_length[i], i);
  }
}

// --------------------------------------------------------------------------
// XXX Create aligned reference bases from cigar and MD tag in read direction

string getReferenceBases(const string &read_bases, vector< CigarOp > CigarData, const vector<char> &MD_op, vector<int> MD_len, const vector<string> &MD_seq, const bool rev_strand, const map<char,char> &reverse_complement_map) {
  string ref_bases = "";

  int increment  = rev_strand ? -1 : 1;
  int MD_idx     = rev_strand ? MD_op.size()-1 : 0;
  int cigar_idx  = rev_strand ? CigarData.size()-1 : 0;
  int read_idx   = 0;
  while (cigar_idx < (int) CigarData.size() and MD_idx < (int) MD_op.size() and cigar_idx >= 0 and MD_idx >= 0) {
    // Advance cigar if requried
    if (CigarData[cigar_idx].Length == 0 or CigarData[cigar_idx].Type == 'H') {
      cigar_idx += increment;
      continue;
    }

    // handle soft-clipping in cigar
    if (CigarData[cigar_idx].Type == 'S') {
      read_idx  += CigarData[cigar_idx].Length;
      cigar_idx += increment;
      continue;
    }

    // Advance MD if requried
    if (MD_len[MD_idx] == 0) {
      MD_idx += increment;
      continue;
    }


    if (CigarData[cigar_idx].Type == 'M' and MD_op[MD_idx] == 'M') {
      // Perfect match
      int advance = min((int) CigarData[cigar_idx].Length, MD_len[MD_idx]);
      ref_bases += read_bases.substr(read_idx,advance);
      read_idx  += advance;
      CigarData[cigar_idx].Length -= advance;
      MD_len[MD_idx] -= advance;
    } else if (CigarData[cigar_idx].Type == 'I') {
      // Insertion (read has a base, reference doesn't)
      int advance = CigarData[cigar_idx].Length;
      read_idx += advance;
      CigarData[cigar_idx].Length -= advance;
    } else if (CigarData[cigar_idx].Type == 'D' and MD_op[MD_idx] == 'D') {
      // Deletion (reference has a base, read doesn't)
      int advance = min((int) CigarData[cigar_idx].Length, MD_len[MD_idx]);
assert(advance == (int) (MD_seq[MD_idx]).length());
      ref_bases += MD_seq[MD_idx];
      CigarData[cigar_idx].Length -= advance;
      MD_len[MD_idx] -= advance;
    } else if (MD_op[MD_idx] == 'X') {
      int advance = min((int)CigarData[cigar_idx].Length, MD_len[MD_idx]);
assert(advance == (int) (MD_seq[MD_idx]).length());
      ref_bases += MD_seq[MD_idx];
      read_idx += advance;
      CigarData[cigar_idx].Length -= advance;
      MD_len[MD_idx] -= advance;
    } else {
      fprintf(stderr, "ionstats alignment: getReferenceBases(): Unexpected OP combination: Cigar=%c, MD=%c !\n", CigarData[cigar_idx].Type, MD_op[MD_idx]);
      exit(1);
    }
  }
  return(ref_bases);
}

// ----------------------------------------------------------------------------

int reverse_complement(string &b, const map<char, char> &rc) {
  unsigned int len=b.length();
  if(len==0)
    return(EXIT_SUCCESS);
  unsigned int n_swap = floor(len/2);
  map<char,char>::const_iterator it1,it2;
  for(unsigned int i=0,j=len-1; i<n_swap; ++i,--j) {
    it1 = rc.find(b[i]);
    it2 = rc.find(b[j]);
    if(it1 == rc.end() || it2 == rc.end())
      return(EXIT_FAILURE);
    b[i] = it2->second;
    b[j] = it1->second;
  }
  if((len % 2) != 0) {
    it1 = rc.find(b[n_swap]);
    if(it1 == rc.end())
      return(EXIT_FAILURE);
    b[n_swap] = it1->second;
  }
  return(EXIT_SUCCESS);
}

void initialize_reverse_complement_map(map<char, char> &rc) {
  rc.clear();
  rc['A'] = 'T';
  rc['T'] = 'A';
  rc['U'] = 'A';
  rc['G'] = 'C';
  rc['C'] = 'G';
  rc['Y'] = 'R';
  rc['R'] = 'Y';
  rc['S'] = 'S';
  rc['W'] = 'W';
  rc['K'] = 'M';
  rc['M'] = 'K';
  rc['B'] = 'V';
  rc['D'] = 'H';
  rc['H'] = 'D';
  rc['V'] = 'B';
  rc['N'] = 'N';

  rc['a'] = 't';
  rc['t'] = 'a';
  rc['u'] = 'a';
  rc['g'] = 'c';
  rc['c'] = 'g';
  rc['y'] = 'r';
  rc['r'] = 'y';
  rc['s'] = 's';
  rc['w'] = 'w';
  rc['k'] = 'm';
  rc['m'] = 'k';
  rc['b'] = 'v';
  rc['d'] = 'h';
  rc['h'] = 'd';
  rc['v'] = 'b';
  rc['n'] = 'n';
}

// ----------------------------------------------------------------------------------

void getHpBreakdown(const string &bases, vector<char> &hp_nuc, vector<uint16_t> &hp_len, vector<uint16_t> &hp_cum_len) {
  hp_nuc.clear();
  hp_len.clear();
  hp_cum_len.clear();
  unsigned int len = bases.length();
  if(len==0)
    return;
  hp_nuc.reserve(len);
  hp_len.reserve(len);
  hp_cum_len.reserve(len);
  char this_nuc = bases[0];
  uint16_t this_len = 1;
  for(unsigned int i=1; i<len; ++i) {
    if(bases[i] != this_nuc) {
      hp_nuc.push_back(this_nuc);
      hp_len.push_back(this_len);
      this_nuc = bases[i];
      this_len = 1;
      if (hp_cum_len.size() == 0)
    	hp_cum_len.push_back(hp_len.back());
      else
        hp_cum_len.push_back(hp_len.back() +hp_cum_len.back());
    } else {
      this_len++;
    }
  }
  hp_nuc.push_back(this_nuc);
  hp_len.push_back(this_len);
  if (hp_cum_len.size() == 0)
    hp_cum_len.push_back(hp_len.back());
  else
    hp_cum_len.push_back(hp_len.back() +hp_cum_len.back());
  return;
}

// ---------------------------------------------------------------------------------

void hpAdvance(
  // Inputs
  align_t                   alignment_type,
  int                       advance,         // number of bases in perfectly-aligned stretch
  // Data for tracking flow
  bool                      evaluate_flow,
  unsigned int              flow_idx,
  const string &            flow_order,
  const string &            read_bases,
  const string &            ref_bases,
  unsigned int              max_flows,
  // Data for tracking position in bases/HPs
  int                       read_idx,         // index of where we are in read_bases
  const vector<char> &      read_hp_nuc,      // read hp nucs
  const vector<uint16_t> &  read_hp_len,      // read hp lengths
  const vector<uint16_t> &  read_hp_cum_len,  // read hp cumulative lengths
  int                       ref_idx,          // index of where we are in ref_bases
  const vector<char> &      ref_hp_nuc,       // ref hp nucs
  const vector<uint16_t> &  ref_hp_len,       // ref hp lengths
  const vector<uint16_t> &  ref_hp_cum_len,   // ref hp cumulative lengths
  // Objects that may be modified
  int &                     stored_read_match_count, // number of read bases matching current ref hp that have been seen so far
  unsigned int &            read_hp_idx,             // index of where we are in read_hp_nuc, read_hp_len, read_hp_cum_len
  unsigned int &            ref_hp_idx,              // index of where we are in ref_hp_nuc, ref_hp_len, ref_hp_cum_len
  vector<int16_t> &         ref_hp_err,              //
  vector<uint16_t> &        ref_hp_flow,
  vector<uint16_t> &        zeromer_insertion_flow,  // flows in which there is an insertion where reference hp length is zero
  vector<uint16_t> &        zeromer_insertion_len    // insertion length for insertions against a reference zeromer
) {

  if( (read_hp_idx > 0) && (read_idx < read_hp_cum_len[read_hp_idx-1]) )
    read_hp_idx--; // Skip back the read HP if jumped across a subsequent alignment block
  if(read_hp_idx < read_hp_len.size()) {
    assert(read_idx <= read_hp_cum_len[read_hp_idx]);
    if(read_hp_idx > 0)
      assert(read_idx >= read_hp_cum_len[read_hp_idx-1]);
  }
  if(ref_hp_idx < ref_hp_len.size()) {
    assert(ref_idx <= ref_hp_cum_len[ref_hp_idx]);
    if(ref_hp_idx > 0)
      assert(ref_idx >= ref_hp_cum_len[ref_hp_idx-1]);
  }
  // Note that ref_hp_idx may be out of bounds relative to the ref_hp_len array.
  // This is OK, it can happen for example when the last base is an insertion.

  int last_ref_idx = 0;
  if(alignment_type == ALIGN_MATCH) {
    last_ref_idx = ref_idx + advance;
    while( (ref_hp_idx < ref_hp_len.size()) && (read_hp_idx < read_hp_len.size()) && (ref_idx < last_ref_idx) ) {
      int ref_match_count  = ref_hp_cum_len[ref_hp_idx]   - ref_idx;
      int read_match_count = read_hp_cum_len[read_hp_idx] - read_idx;
      if(ref_idx + ref_match_count > last_ref_idx) {
        // Perfectly-aligned block ends, splitting the current reference hp.
        stored_read_match_count += last_ref_idx - ref_idx;
        // ref_hp_idx and flow_idx will not advance, but read_hp_idx might need to
        if(read_idx + read_match_count >= read_hp_cum_len[read_hp_idx])
          read_hp_idx++;
        break;
      } else {
        assert(read_match_count>=ref_match_count);
        // We have reached then end of a ref HP without going beyond the end of the perfectly-aligned block, score the current HP
        stored_read_match_count += read_match_count;
        // Store error rate & flow index for this reference HP
        ref_hp_err.push_back(stored_read_match_count - ref_hp_len[ref_hp_idx]);
assert(ref_hp_err.back() + ref_hp_len[ref_hp_idx] >= 0);
        if(evaluate_flow) {
          flowCatchup(flow_idx,flow_order,read_idx,read_bases);
if(ref_hp_flow.size() > 0)
assert(flow_idx >= ref_hp_flow.back());
          ref_hp_flow.push_back(flow_idx);
        }
        // Advance
        ref_idx += ref_match_count;
        ref_hp_idx++;
        if(evaluate_flow)
          flowAdvanceToNextHP(read_match_count,flow_idx,flow_order,read_idx,read_bases);
        read_idx += read_match_count;
        read_hp_idx++;
        stored_read_match_count = 0;
      }
    }
  } else if ( (alignment_type == ALIGN_INS) && (ref_hp_idx < ref_hp_len.size()) ) {
    // Read has an insertion, and we have not reached the end of the reference sequence yet
    int last_read_idx = read_idx + advance;
    if( (read_hp_idx > 0) && (read_idx < read_hp_cum_len[read_hp_idx-1]) ) {
      // If the first inserted HP matches the previous reference HP, it will already have been scored before reaching this point.
      read_idx = read_hp_cum_len[read_hp_idx-1];
    }
    while( read_idx < last_read_idx) {
      int read_hp_len = read_hp_cum_len[read_hp_idx] - read_idx;
      int insertion_len = 0;
      if(read_idx + read_hp_len > last_read_idx) {
        // Insertion in a read HP that spans the right side of the insertion - this insertion will be recorded later
        insertion_len = last_read_idx - read_idx;
        if(read_hp_nuc[read_hp_idx]==ref_hp_nuc[ref_hp_idx])
          stored_read_match_count = insertion_len;
      } else {
        // Insertion in a read HP that is fully contained within the insertion
        insertion_len = read_hp_len;
        stored_read_match_count = 0;
        if(evaluate_flow) {
          // Need to figure out the flow to which to assign this 0mer insertion
          char flowed_base = flow_order[flow_idx % flow_order.size()];
          char inserted_base = read_hp_nuc[read_hp_idx];
          while( flowed_base != inserted_base && flow_idx < max_flows-1)
            flowed_base = flow_order[++flow_idx % flow_order.size()];
          zeromer_insertion_flow.push_back(flow_idx);
          zeromer_insertion_len.push_back(insertion_len);
        }
      }
      read_idx += insertion_len;
      if(read_idx == read_hp_cum_len[read_hp_idx])
        read_hp_idx++;
    }
  } else if (alignment_type == ALIGN_DEL) {
    assert(ref_hp_idx < ref_hp_len.size());
    last_ref_idx = ref_idx + advance; // last ref. base to be scored
    while( ref_idx < last_ref_idx) {
      int ref_count = ref_hp_cum_len[ref_hp_idx] - ref_idx; // bases remaining in current hp
      if(ref_idx + ref_count <= last_ref_idx) {
        // Finished a reference HP, store it
        ref_hp_err.push_back(stored_read_match_count - ref_hp_len[ref_hp_idx]);
        assert(ref_hp_err.back() + ref_hp_len[ref_hp_idx] >= 0);
        if(evaluate_flow) {
          // Try scan forward to find a flow matching the deleted base, otherwise leave it assigned to the current flow
          char flowed_base = flow_order[flow_idx % flow_order.size()];
          char deleted_base = ref_hp_nuc[ref_hp_idx];
          char next_read_base = (read_idx < (int) read_bases.length()) ? read_bases[read_idx] : 'N';
          unsigned int original_flow_idx = flow_idx;
          while( flowed_base != deleted_base  && flowed_base != next_read_base && flow_idx < max_flows-1)
            flowed_base = flow_order[++flow_idx % flow_order.size()];
assert(flow_idx >= ref_hp_flow.back());
          if(flowed_base == deleted_base)
            ref_hp_flow.push_back(flow_idx);
          else
            ref_hp_flow.push_back(original_flow_idx);
        }
        ref_idx += ref_count;
        ref_hp_idx++;
        stored_read_match_count = 0;
      } else {
        // partially-aligned reference HP, don't store yet
        ref_idx += last_ref_idx - ref_idx;
      }
    }
  } else if (alignment_type == ALIGN_SUB) {
    last_ref_idx = ref_idx + advance; // last ref. base to be scored
    while( ref_idx < last_ref_idx) {
      assert(ref_hp_idx < ref_hp_len.size());
      int ref_count = ref_hp_cum_len[ref_hp_idx] - ref_idx; // bases remaining in current hp
      bool completed_ref_hp=false;
      if(ref_idx + ref_count <= last_ref_idx) { // reference HP done within bases given to function
        // Finished a reference HP, store it
        completed_ref_hp=true;
        ref_hp_err.push_back(stored_read_match_count - ref_hp_len[ref_hp_idx]);
        assert(ref_hp_err.back() + ref_hp_len[ref_hp_idx] >= 0);
        if(evaluate_flow) {
          char substituted_base = ref_hp_nuc[ref_hp_idx];
          char next_ref_base = (ref_idx+1 < (int) ref_bases.length() ) ? ref_bases[ref_idx+1] : 'N';
          char last_ref_base = (ref_idx > 0) ? ref_bases[ref_idx-1] : 'N';
          char next_read_base = (read_idx+1 < (int) read_bases.length() ) ? read_bases[read_idx+1] : 'N';

          char best_prev_flowed_base = flow_order[flow_idx % flow_order.size()];
          unsigned int best_prev_flow = flow_idx;
          if(ref_hp_flow.size() > 0 && flow_idx > ref_hp_flow.back()) {
            if(best_prev_flow > 0)
              best_prev_flowed_base = flow_order[--best_prev_flow % flow_order.size()];
            while( best_prev_flowed_base != substituted_base && best_prev_flowed_base != last_ref_base && best_prev_flow > ref_hp_flow.back() )
              best_prev_flowed_base = flow_order[--best_prev_flow % flow_order.size()];
          }
          unsigned int best_next_flow = flow_idx;
          char best_next_flowed_base = flow_order[best_next_flow % flow_order.size()];
          while( best_next_flowed_base != substituted_base && best_next_flowed_base != next_ref_base && best_next_flowed_base != next_read_base && best_next_flow < max_flows-1 )
            best_next_flowed_base = flow_order[++best_next_flow % flow_order.size()];

          if(substituted_base == best_prev_flowed_base) {
            flow_idx = best_prev_flow;
          } else if(substituted_base == best_next_flowed_base) {
            flow_idx = best_next_flow;
          }
if(ref_hp_flow.size() > 0)
assert(flow_idx >= ref_hp_flow.back());
          ref_hp_flow.push_back(flow_idx);
        }
        // Advance to next hp
        ref_hp_idx++;
      } else {
        ref_count = last_ref_idx - ref_idx; // Now: ref. bases remaining in substitution stretch
      }
      ref_idx += ref_count;
      if(evaluate_flow)
        flowAdvanceToNextHP(ref_count,flow_idx,flow_order,read_idx,read_bases);
      read_idx += ref_count;
      while( (read_hp_idx < read_hp_cum_len.size()) && (read_hp_cum_len[read_hp_idx] <= read_idx) )
        read_hp_idx++;
      // check if last substituted read base matches next reference base at the end of the block of subs, if so count the last read HP
      stored_read_match_count = 0;
      if( completed_ref_hp && (ref_idx==last_ref_idx) && (ref_hp_idx < ref_hp_nuc.size()) ) {
        unsigned int last_substituted_nuc_hp_idx = ((read_hp_idx > 0) && (read_idx == read_hp_cum_len[read_hp_idx-1])) ? read_hp_idx-1 : read_hp_idx;
        if(read_hp_nuc[last_substituted_nuc_hp_idx] == ref_hp_nuc[ref_hp_idx])
          stored_read_match_count = (last_substituted_nuc_hp_idx == 0) ? read_idx : (read_idx - read_hp_cum_len[last_substituted_nuc_hp_idx-1] );
      }
    }
  }
}

// ----------------------------------------------------------------------------

int checkDimensions(bool &spatial_stratify, const vector<unsigned int> &chip_origin, const vector<unsigned int> &chip_dim, const vector<unsigned int> &subregion_dim, unsigned int &n_col_subregions, unsigned int &n_row_subregions) {
  bool problem=false;
  spatial_stratify = false;
  if(chip_origin.size()==0 && chip_dim.size()==0 && subregion_dim.size()==0)
    return(EXIT_SUCCESS);

  // Make sure all options specified a pair
  if(chip_origin.size() != 2) {
    cerr << "ERROR: chip-origin option must specify exactly 2 comma-separated values" << endl;
    problem=true;
  }
  if(chip_dim.size() != 2) {
    cerr << "ERROR: chip-dim option must specify exactly 2 comma-separated values" << endl;
    problem=true;
  }
  if(subregion_dim.size() != 2) {
    cerr << "ERROR: subregion-dim option must specify exactly 2 comma-separated values" << endl;
    problem=true;
  }

  // Make sure size values are positive
  if(chip_dim[0] <= 0 || chip_dim[1] <= 0) {
    cerr << "ERROR: chip-dim option values must be positive" << endl;
    problem=true;
  }
  if(subregion_dim[0] <= 0 || subregion_dim[1] <= 0) {
    cerr << "ERROR: subregion-dim option values must be positive" << endl;
    problem=true;
  }

  // Make sure subregion dimensions are less than chip dimensions
  if((subregion_dim[0] > chip_dim[0]) || (subregion_dim[1] > chip_dim[1])) {
    cerr << "ERROR: subregion-dim option values must be smaller than chip_dim option values" << endl;
    problem=true;
  }

  if(!problem) {
    n_col_subregions = ceil((double) chip_dim[0] / subregion_dim[0]);
    n_row_subregions = ceil((double) chip_dim[1] / subregion_dim[1]);
    spatial_stratify=true;
  }

  return(problem);
}

bool getRegionIndex(unsigned int &region_idx, bool &no_region, bool &out_of_bounds, string &name, vector<unsigned int> &chip_origin, vector<unsigned int> &chip_dim, vector<unsigned int> &subregion_dim, unsigned int n_col_subregions) {
  int32_t col,row;

  if(!ion_readname_to_xy(name.c_str(),&col,&row)) {
    no_region=true;
    return(false);
  }
  if(col < (int)chip_origin[0] || row < (int)chip_origin[1]) {
    out_of_bounds=true;
    return(false);
  }
  col -=chip_origin[0];
  row -=chip_origin[1];
  if(col >= (int) chip_dim[0] || row >= (int) chip_dim[1]) {
    out_of_bounds=true;
    return(false);
  }
  region_idx = floor(col/subregion_dim[0]) + n_col_subregions * floor(row/subregion_dim[1]);
  
  return(true);
}

void assignRegionNames(vector<string> &region_name, const vector<unsigned int> &chip_origin, const vector<unsigned int> &chip_dim, const vector<unsigned int> &subregion_dim, vector< vector<unsigned int> > &region_specific_origin, vector< vector<unsigned int> > &region_specific_dim) {
  unsigned int n_col_subregions = ceil((double) chip_dim[0] / subregion_dim[0]);
  unsigned int n_row_subregions = ceil((double) chip_dim[1] / subregion_dim[1]);
  unsigned int n_subregions = n_col_subregions*n_row_subregions;
  region_name.clear();
  region_specific_origin.resize(n_subregions);
  region_specific_dim.resize(n_subregions);
  for(unsigned int row_idx=0,k=0; row_idx < n_row_subregions; ++row_idx) {
    for(unsigned int col_idx=0; col_idx < n_col_subregions; ++col_idx,++k) {
      unsigned int col_origin = chip_origin[0] + col_idx*subregion_dim[0];
      unsigned int row_origin = chip_origin[1] + row_idx*subregion_dim[1];
      unsigned int col_max = min(col_origin + subregion_dim[0], chip_origin[0]+chip_dim[0]);
      unsigned int row_max = min(row_origin + subregion_dim[1], chip_origin[1]+chip_dim[1]);
      string col_origin_string = static_cast<ostringstream*>( &(ostringstream() << col_origin) )->str();
      string row_origin_string = static_cast<ostringstream*>( &(ostringstream() << row_origin) )->str();
      string col_max_string    = static_cast<ostringstream*>( &(ostringstream() << col_max)    )->str();
      string row_max_string    = static_cast<ostringstream*>( &(ostringstream() << row_max)    )->str();
      region_name.push_back("origin=" + col_origin_string + "," + row_origin_string + "-max=" + col_max_string + "," + row_max_string);
      region_specific_origin[k].resize(2);
      region_specific_origin[k][0] = col_origin;
      region_specific_origin[k][1] = row_origin;
      region_specific_dim[k].resize(2);
      region_specific_dim[k][0] = col_max-col_origin;
      region_specific_dim[k][1] = row_max-row_origin;
    }
  }
}

bool isAqLength(string s) {
  bool result=false;
  if((s.substr(0,2) == "AQ") && (s.length() > 2)) {
    string s_num = s.substr(2,s.length()-2);
    char *pos=0;
    unsigned int l = std::strtoul(s_num.c_str(),&pos,0);
    unsigned int n_char_parsed = pos - s_num.c_str();
    if((n_char_parsed == s_num.length()) && (l !=0))
      result = true;
  }
  return(result);
}

void writeIonstatsH5(string h5_filename, bool append_h5_file, const vector<string> & region_name, AlignmentSummary & alignment_summary) {
  // Pack all the ErrorData objects into a map
  map< string, ErrorData > error_data;
  // Per-base data
  error_data["per_base"] = alignment_summary.BasePosition();
  const map< string, ErrorData > & read_group_base_position = alignment_summary.ReadGroupBasePosition();
  for(map < string, ErrorData >::const_iterator it=read_group_base_position.begin(); it != read_group_base_position.end(); ++it)
    error_data["per_read_group/" + it->first + "/per_base"] = it->second;
  // Per-flow data
  ErrorData flow_position = alignment_summary.FlowPosition();
  if(flow_position.HasData())
    error_data["per_flow"] = flow_position;
  map< string, ErrorData > & read_group_flow_position = alignment_summary.ReadGroupFlowPosition();
  for(map < string, ErrorData >::iterator it=read_group_flow_position.begin(); it != read_group_flow_position.end(); ++it)
    if(it->second.HasData())
      error_data["per_read_group/" + it->first + "/per_flow"] = it->second;
  
  // Pack all the HpData objects into a map
  map< string, HpData > hp_data;
  HpData per_hp = alignment_summary.PerHp();
  if(per_hp.HasData())
    hp_data["per_hp"] = per_hp;
  map< string, HpData > & read_group_per_hp = alignment_summary.ReadGroupPerHp();
  for(map < string, HpData >::iterator it=read_group_per_hp.begin(); it != read_group_per_hp.end(); ++it)
    if(it->second.HasData())
      hp_data["per_read_group/" + it->first + "/per_hp"] = it->second;

  // Pack RegionalSummary objects into a map
  map< string, RegionalSummary > regional_data;
  vector< RegionalSummary > & regional_summary = alignment_summary.GetRegionalSummary();
  for(unsigned int i=0; i<region_name.size(); ++i)
    regional_data["per_region/" + region_name[i]] = regional_summary[i];

  // Open or create h5 file
  hid_t file_id;
  if(append_h5_file)
    file_id = H5Fopen(h5_filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  else
    file_id = H5Fcreate(h5_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  // Write all the error_data map elements
  for(map< string, ErrorData >::iterator it = error_data.begin(); it != error_data.end(); ++it)
    it->second.writeH5(file_id,it->first);
  for(map< string, HpData >::iterator it = hp_data.begin(); it != hp_data.end(); ++it)
    it->second.writeH5(file_id,it->first);
  for(map< string, RegionalSummary >::iterator it = regional_data.begin(); it != regional_data.end(); ++it)
    it->second.writeH5(file_id,it->first);
  herr_t status = H5Fclose (file_id);
}

herr_t MergeErrorDataFromH5(hid_t loc_id, const char *name, const H5L_info_t *info, void *operator_data);
herr_t MergeErrorDataFromH5(hid_t loc_id, const char *name, const H5L_info_t *info, void *operator_data) {
  H5O_info_t      infobuf;
  H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT);
  if(infobuf.type == H5O_TYPE_GROUP) {
    // Check if the current group contains what we need for an error data object
    vector<string> required_dataset;
    required_dataset.push_back(string(name) + "/region_origin");
    required_dataset.push_back(string(name) + "/region_dim");
    required_dataset.push_back(string(name) + "/error_data_dim");
    required_dataset.push_back(string(name) + "/error_data");
    for(unsigned int i=0; i<required_dataset.size(); ++i)
      if(!H5Lexists(loc_id, required_dataset[i].c_str(), H5P_DEFAULT))
        return 0;

    ErrorData this_error_data;
    hid_t group_id = H5Gopen2(loc_id, name, H5P_DEFAULT);
    if(this_error_data.readH5(group_id) == EXIT_SUCCESS) {
      error_data_merge_t * master = static_cast<error_data_merge_t *>(operator_data);
      string group_name = string(name);
      if(master->merge_proton_blocks)
        group_name = transform_proton_block_read_group_name(group_name);
      map< string, ErrorData >::iterator it = master->error_data.find(group_name);
      if(it != master->error_data.end()) {
        master->error_data[group_name].MergeFrom(this_error_data);
      } else {
        master->error_data[group_name] = this_error_data;
      }
    }
    H5Gclose(group_id);
  }
  return 0;
}

int RecursivelyMergeErrorDataFromH5(const hid_t input_file_id, const hid_t group_id, map<string, ErrorData> &error_data, bool merge_proton_blocks);
int RecursivelyMergeErrorDataFromH5(const hid_t input_file_id, const hid_t group_id, map<string, ErrorData> &error_data, bool merge_proton_blocks) {

  char group_name[MAX_GROUP_NAME_LEN];
  H5Iget_name(group_id, group_name, MAX_GROUP_NAME_LEN);

  // Check if the current group contains what we need for an error data object
  bool is_error_data_group=true;
  if(string(group_name)=="/") {
    is_error_data_group=false;
  } else {
    vector<string> required_dataset;
    required_dataset.push_back("region_origin");
    required_dataset.push_back("region_dim");
    required_dataset.push_back("error_data_dim");
    required_dataset.push_back("error_data");
    for(unsigned int i=0; i<required_dataset.size(); ++i) {
      if(!H5Lexists(group_id, required_dataset[i].c_str(), H5P_DEFAULT)) {
        is_error_data_group=false;
        break;
      }
    }
  }

  if(is_error_data_group) {
    ErrorData this_error_data;
    if(this_error_data.readH5(group_id) == EXIT_SUCCESS) {
      string transformed_name = string(group_name);
      if(merge_proton_blocks)
        transformed_name = transform_proton_block_read_group_name(group_name);
      map< string, ErrorData >::iterator it = error_data.find(transformed_name);
      if(it != error_data.end()) {
        error_data[transformed_name].MergeFrom(this_error_data);
      } else {
        error_data[transformed_name] = this_error_data;
      }
    }
    this_error_data.clear();
  }

  // Now recurse on any subgroups
  hsize_t n_obj=0;
  H5Gget_num_objs(group_id, &n_obj);
  for(unsigned int i=0; i<n_obj; ++i) {
    int otype = H5Gget_objtype_by_idx(group_id, (size_t)i);
    if(otype == H5G_GROUP) {
      char subgroup_name[MAX_GROUP_NAME_LEN];
      H5Gget_objname_by_idx(group_id, (hsize_t)i, subgroup_name, (size_t) MAX_GROUP_NAME_LEN);
      hid_t subgroup_id = H5Gopen(group_id,subgroup_name,H5P_DEFAULT);
      RecursivelyMergeErrorDataFromH5(input_file_id,subgroup_id,error_data,merge_proton_blocks);
      H5Gclose(subgroup_id);
    }
  }

  return 0;
}

herr_t MergeHpDataFromH5(hid_t loc_id, const char *name, const H5L_info_t *info, void *operator_data);
herr_t MergeHpDataFromH5(hid_t loc_id, const char *name, const H5L_info_t *info, void *operator_data) {
  H5O_info_t      infobuf;
  H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT);
  if(infobuf.type == H5O_TYPE_GROUP) {
    // Check if the current group contains what we need for an hp data object
    vector<string> required_dataset;
    required_dataset.push_back(string(name) + "/region_origin");
    required_dataset.push_back(string(name) + "/region_dim");
    required_dataset.push_back(string(name) + "/hp_data_dim");
    for(unsigned int i=0; i<required_dataset.size(); ++i)
      if(!H5Lexists(loc_id, required_dataset[i].c_str(), H5P_DEFAULT))
        return 0;

    HpData this_hp_data;
    hid_t group_id = H5Gopen2(loc_id, name, H5P_DEFAULT);
    if(this_hp_data.readH5(group_id) == EXIT_SUCCESS) {
      hp_data_merge_t * master = static_cast<hp_data_merge_t *>(operator_data);
      string group_name = string(name);
      if(master->merge_proton_blocks)
        group_name = transform_proton_block_read_group_name(group_name);
      map< string, HpData >::iterator it = master->hp_data.find(group_name);
      if(it != master->hp_data.end()) {
        master->hp_data[group_name].MergeFrom(this_hp_data);
      } else {
        master->hp_data[group_name] = this_hp_data;
      }
    }
    H5Gclose(group_id);
  }
  return 0;
}

herr_t MergeRegionalSummaryFromH5(hid_t loc_id, const char *name, const H5L_info_t *info, void *operator_data);
herr_t MergeRegionalSummaryFromH5(hid_t loc_id, const char *name, const H5L_info_t *info, void *operator_data) {
  H5O_info_t      infobuf;
  H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT);
  if(infobuf.type == H5O_TYPE_GROUP) {
    // Check if the current group contains what we need for a regional_summary object
    vector<string> required_dataset;
    required_dataset.push_back(string(name) + "/region_origin");
    required_dataset.push_back(string(name) + "/region_dim");
    required_dataset.push_back(string(name) + "/n_err");
    required_dataset.push_back(string(name) + "/n_aligned");
    required_dataset.push_back(string(name) + "/data_dim");
    required_dataset.push_back(string(name) + "/hp_count");
    required_dataset.push_back(string(name) + "/hp_err");
    for(unsigned int i=0; i<required_dataset.size(); ++i)
      if(!H5Lexists(loc_id, required_dataset[i].c_str(), H5P_DEFAULT))
        return 0;

    RegionalSummary this_regional_summary;
    hid_t group_id = H5Gopen2(loc_id, name, H5P_DEFAULT);
    if(this_regional_summary.readH5(group_id) == EXIT_SUCCESS) {
      regional_summary_merge_t * master = static_cast<regional_summary_merge_t *>(operator_data);
      string group_name = string(name);
      if(master->merge_proton_blocks)
        group_name = transform_proton_block_read_group_name(group_name);
      map< string, RegionalSummary >::iterator it = master->regional_summary.find(group_name);
      if(it != master->regional_summary.end()) {
        master->regional_summary[group_name].MergeFrom(this_regional_summary);
      } else {
        master->regional_summary[group_name] = this_regional_summary;
      }
    }
    H5Gclose(group_id);
  }
  return 0;
}

int IonstatsAlignmentReduceH5(const string& output_h5_filename, const vector<string>& input_h5_filename, bool merge_proton_blocks) {
  
  // Gather all inputs into error_data and hp_data structures
  error_data_merge_t        ed;
  hp_data_merge_t           hd;
  regional_summary_merge_t  rs;
  ed.merge_proton_blocks = merge_proton_blocks;
  hd.merge_proton_blocks = merge_proton_blocks;
  rs.merge_proton_blocks = merge_proton_blocks;

  unsigned int group_count=0;
  uint64_t total_bytes=0;
  for(unsigned int i=0; i < input_h5_filename.size(); ++i) {
    if(H5Fis_hdf5(input_h5_filename[i].c_str()) <= 0) {
      cerr << "IonstatsAlignmentReduceH5 WARNING: fail to open " << input_h5_filename[i] << endl;
      continue;
    }
    GetAggregatorSize(ed.error_data,hd.hp_data,rs.regional_summary,group_count,total_bytes);
    hid_t input_file_id  = H5Fopen(input_h5_filename[i].c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	if(input_file_id < 0) {
      cerr << "IonstatsAlignmentReduceH5 WARNING: fail to open " << input_h5_filename[i] << endl;
      continue;
	}

    hid_t group_id = H5Gopen2(input_file_id, "/", H5P_DEFAULT);
    //RecursivelyMergeErrorDataFromH5(input_file_id,group_id,ed.error_data,merge_proton_blocks);
    H5Lvisit(group_id, H5_INDEX_NAME, H5_ITER_INC, MergeErrorDataFromH5, &ed);
    H5Lvisit(group_id, H5_INDEX_NAME, H5_ITER_INC, MergeHpDataFromH5, &hd);
    H5Lvisit(group_id, H5_INDEX_NAME, H5_ITER_INC, MergeRegionalSummaryFromH5, &rs);
    H5Gclose(group_id);
    H5Fclose(input_file_id);
  }
  GetAggregatorSize(ed.error_data,hd.hp_data,rs.regional_summary,group_count,total_bytes);

  // Open h5 file and write all the error_data map elements
  hid_t file_id = H5Fcreate(output_h5_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  for(map< string, ErrorData >::iterator it = ed.error_data.begin(); it != ed.error_data.end(); ++it)
    it->second.writeH5(file_id,it->first);
  for(map< string, HpData >::iterator it = hd.hp_data.begin(); it != hd.hp_data.end(); ++it)
    it->second.writeH5(file_id,it->first);
  for(map< string, RegionalSummary >::iterator it = rs.regional_summary.begin(); it != rs.regional_summary.end(); ++it)
    it->second.writeH5(file_id,it->first);
  H5Fclose (file_id);
  return 0;
}

bool getTagZF(BamAlignment & alignment, uint32_t &flow_idx) {
  // In our pipeline, the ZF tag apparently is sometimes stored as type 'i' and sometimes as 'C' - annoying
  int32_t flow_idx_int32 = 0;
  uint8_t flow_idx_uint8 = 0;
  if(alignment.GetTag("ZF",flow_idx_int32)) {
    flow_idx = flow_idx_int32;
    return true;
  } else if(alignment.GetTag("ZF",flow_idx_uint8)) {
    flow_idx = flow_idx_uint8;
    return true;
  } else {
    return false;
  }
}

// --------------------------------------------------------------------------

bool hasInvalidCigar(const BamAlignment & alignment) {
  bool problem=false;
  if(alignment.IsMapped() && alignment.Length > 0) {
    unsigned int cigar_len=0;
    for(unsigned int i=0; i<alignment.CigarData.size(); ++i)
      if( (alignment.CigarData[i].Type != 'D') && (alignment.CigarData[i].Type != 'H') && (alignment.CigarData[i].Type != 'P') )
        cigar_len += alignment.CigarData[i].Length;
    if(cigar_len != (unsigned int) alignment.Length)
      problem=true;
  }
  return(problem);
}

// --------------------------------------------------------------------------
// Converts base symbols to upper case; identifies the presence of ambiguity symbols or invalid characters, not representing bases.

void checkBases(string &b, bool &ambiguous_bases, bool &invalid_bases) {
  unsigned int n = b.length();
  ambiguous_bases=false;
  invalid_bases=false;
  for(unsigned int i=0; i<n; ++i) {
    switch(b[i]) {
      case 'A':
      case 'C':
      case 'G':
      case 'T':
        break;
      case 'a':
        b[i] = 'A';
        break;
      case 'c':
        b[i] = 'C';
        break;
      case 'g':
        b[i] = 'G';
        break;
      case 't':
        b[i] = 'T';
        break;

      case 'M':
      case 'R':
      case 'W':
      case 'S':
      case 'Y':
      case 'K':
      case 'V':
      case 'H':
      case 'D':
      case 'B':
      case 'N':
        ambiguous_bases=true;
        break;
      case 'm':
        b[i] = 'M';
        ambiguous_bases=true;
        break;
      case 'r':
        b[i] = 'R';
        ambiguous_bases=true;
        break;
      case 'w':
        b[i] = 'W';
        ambiguous_bases=true;
        break;
      case 's':
        b[i] = 'S';
        ambiguous_bases=true;
        break;
      case 'y':
        b[i] = 'Y';
        ambiguous_bases=true;
        break;
      case 'k':
        b[i] = 'K';
        ambiguous_bases=true;
        break;
      case 'v':
        b[i] = 'V';
        ambiguous_bases=true;
        break;
      case 'h':
        b[i] = 'H';
        ambiguous_bases=true;
        break;
      case 'd':
        b[i] = 'D';
        ambiguous_bases=true;
        break;
      case 'b':
        b[i] = 'B';
        ambiguous_bases=true;
        break;
      case 'n':
        b[i] = 'N';
        ambiguous_bases=true;
        break;

      default:
        invalid_bases=true;
    }
  }
}

string transform_proton_block_read_group_name(const string &read_group) {
  // Intent is to clip the block-specific name transformation that is applied for Proton runs
  // Clips any suffix that is found matching the regular expression \.[0-9A-Z]{1,2}

  // clip prefix
  string prefix = "per_read_group/";
  if((read_group.length() <= prefix.length()) || (read_group.substr(0,prefix.length())!=prefix))
    return(read_group);

  // clip suffix
  string suffix = "";
  string center=read_group.substr(prefix.length()+1);
  int pos = center.find("/");
  if((pos != (int) std::string::npos)) {
    suffix = center.substr(pos);
    center = center.substr(0,pos);
  }
  
  // search the remaining part for the block-specific suffix
  pos = center.rfind(".");
  if((pos != (int) std::string::npos) && (pos > 0) && (pos >= (int)(center.length() - 3)) && (pos < (int)(center.length() - 1))) {
    // verify that each character of the suffix is one of [0-9A-Z]
    bool match=true;
    for(unsigned int i=pos+1; i<center.length(); ++i) {
      char c = center[i];
      if(!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z')))
        match=false;
    }
    if(match)
      return(prefix + center.substr(0,pos) + suffix);
  }

  return(read_group);
}

void GetAggregatorSize(map<string, ErrorData> &error_data, map<string, HpData> &hp_data, map<string, RegionalSummary> &regional_summary, unsigned int &group_count, uint64_t &total_bytes) {
  uint64_t ed_size=0;
  for(map<string, ErrorData>::iterator it=error_data.begin(); it != error_data.end(); ++it)
    ed_size += it->second.Size();
  uint64_t hd_size=0;
  for(map<string, HpData>::iterator it=hp_data.begin(); it != hp_data.end(); ++it)
    hd_size += it->second.Size();
  uint64_t rs_size=0;
  for(map<string, RegionalSummary>::iterator it=regional_summary.begin(); it != regional_summary.end(); ++it)
    rs_size += it->second.Size();
  group_count = error_data.size() + hp_data.size() + regional_summary.size();
  total_bytes = ed_size + hd_size + rs_size;
}

void debug_alignment(
  IonstatsAlignmentOptions * opt,
  BamAlignment &             alignment,
  ReadAlignmentErrors &      base_space_errors,
  ReadAlignmentErrors &      flow_space_errors,
  vector<uint16_t> &         ref_hp_flow
) {
  bool skip=false;
  if( (opt->DebugErrorFlow() >= 0) && (!opt->EvaluateFlow() || !flow_space_errors.HasError(opt->DebugErrorFlow())) )
    skip=true;
  else if( (opt->DebugAlignedFlow() >= 0) && (!opt->EvaluateFlow() || !flow_space_errors.IsAligned(opt->DebugAlignedFlow())) )
    skip=true;
  else if( (opt->DebugPositiveRefFlow() >= 0) && (!opt->EvaluateFlow() || !binary_search(ref_hp_flow.begin(),ref_hp_flow.end(),opt->DebugPositiveRefFlow())) )
    skip=true;

  if(skip)
    return;

  cerr << alignment.Name << "\t" << alignment.AlignedBases << endl;
  cerr << "Base-space Errors:" << endl;
  base_space_errors.Print();
  cerr << "Flow-space Errors:" << endl;
  flow_space_errors.Print();
}

bool is_unambiguous_base(char b) {
  bool unambiguous=false;
  switch(b) {
    case 'A':
    case 'C':
    case 'G':
    case 'T':
      unambiguous=true;
      break;
    default:
      unambiguous=false;
  }
  return(unambiguous);
}

bool compatible_bases(char a, char b) {
  return((base_to_bitcode(a) & base_to_bitcode(b)) > 0);
}

unsigned char base_to_bitcode(char base) {
  // Bases get encoded this way:
  //  A   001   A   
  //  C   002    C  
  //  M   003   AC  
  //  G   004     G 
  //  R   005   A G 
  //  S   006    CG 
  //  V   007   ACG 
  //  T   010      T
  //  W   011   A  T
  //  Y   012    C T
  //  H   013   AC T
  //  K   014     GT
  //  D   015   A GT
  //  B   016    CGT
  //  N   017   ACGT
  unsigned char bitcode=0;
  switch(base) {
    case 'A':
      bitcode = 001;
      break;
    case 'C':
      bitcode = 002;
      break;
    case 'M':
      bitcode = 003;
      break;
    case 'G':
      bitcode = 004;
      break;
    case 'R':
      bitcode = 005;
      break;
    case 'S':
      bitcode = 006;
      break;
    case 'V':
      bitcode = 007;
      break;
    case 'T':
      bitcode = 010;
      break;
    case 'W':
      bitcode = 011;
      break;
    case 'Y':
      bitcode = 012;
      break;
    case 'H':
      bitcode = 013;
      break;
    case 'K':
      bitcode = 014;
      break;
    case 'D':
      bitcode = 015;
      break;
    case 'B':
      bitcode = 016;
      break;
    case 'N':
      bitcode = 017;
      break;
    default:
      assert(1);
  }
  return(bitcode);
}

unsigned int assignDeletionsToFlows(vector<uint16_t> &del_flow, uint32_t &flow_idx, char prev_read_base, char next_read_base, const string &flow_order, int advance, const string inserted_seq, uint16_t n_flow) {

  // Advance through the deleted bases assigning to flows in order encountered
  char flowed_base = flow_order[flow_idx % flow_order.size()];
  unsigned int insert_length = inserted_seq.length();
  unsigned int insert_idx=0;
  while(insert_idx < insert_length) {
    char inserted_base = inserted_seq[insert_idx];
    while( inserted_base != flowed_base && next_read_base != flowed_base)
      flowed_base = flow_order[++flow_idx % flow_order.size()];
    if(next_read_base == flowed_base || flow_idx >= n_flow)
      break;
    del_flow.push_back(flow_idx);
    ++insert_idx;
  }

  // If we have remaining deleted bases that match the read base in the next flow, assign them to the next flow
  if(insert_idx < insert_length && next_read_base == flowed_base && flow_idx < n_flow) {
    while(insert_idx < insert_length) {
      char inserted_base = inserted_seq[insert_idx];
      if(inserted_base != next_read_base)
        break;
      del_flow.push_back(flow_idx);
      ++insert_idx;
    }
  }

  unsigned int impossible_deletions=0;
  if(insert_idx < insert_length) {
    // At this point any remaining deletions are either "impossible", or we have gone beyond the 3' end of the read.
    if(next_read_base == 'N') {
      // We are beyond the 3' end of the read, assign deletions to flows as they arise
      while(insert_idx < insert_length && flow_idx < n_flow) {
        char inserted_base = inserted_seq[insert_idx];
        while( inserted_base != flowed_base )
          flowed_base = flow_order[++flow_idx % flow_order.size()];
        if(flow_idx >= n_flow)
          break;
        del_flow.push_back(flow_idx);
        ++insert_idx;
      }

      // Either there are no remaining deleted bases, or we have reached the limit of flows
      while(insert_idx++ < insert_length)
        del_flow.push_back(n_flow-1);
    } else {
      // remaining deletions are impossible, convention is that we assign them to the flow of the next read base
      impossible_deletions = insert_length - insert_idx;
      if(next_read_base == flowed_base && flow_idx < n_flow) {
        while(insert_idx < insert_length) {
          del_flow.push_back(flow_idx);
          ++insert_idx;
        }
      }
    }
  }

  if(flowed_base != next_read_base)
    flow_idx++;

  // return the number of "impossible" deletions that could not be assigned to a flow
  return(impossible_deletions);
}

void * processAlignments(void *in) {
  ProcessAlignmentContext *pac = static_cast<ProcessAlignmentContext*>(in);

  vector<BamAlignment> output_bam_alignment;
  output_bam_alignment.reserve(pac->opt->OutputBamBufferSize());

  // Initialize reverse-complement map
  map< char, char > reverse_complement_map;
  initialize_reverse_complement_map(reverse_complement_map);

  // Get flow orders and keys for each read group
  map< string, int > read_groups      = pac->input_bam->ReadGroups();
  map< string, string > flow_orders   = pac->input_bam->FlowOrders();
  map< string, string > key_bases     = pac->input_bam->KeyBases();
  map< string, int > key_len          = pac->input_bam->KeyLen();
  unsigned int max_flow_order_len     = pac->input_bam->MaxFlowOrderLen();

  // Ensure flow orders and key bases are uppercase
  for(map< string, string >::iterator flow_order_it = flow_orders.begin(); flow_order_it != flow_orders.end(); flow_order_it++)
    for(unsigned int i=0; i<flow_order_it->second.size(); ++i)
      flow_order_it->second[i] = toupper(flow_order_it->second[i]);
  for(map< string, string >::iterator key_base_it = key_bases.begin(); key_base_it != key_bases.end(); key_base_it++)
    for(unsigned int i=0; i<key_base_it->second.size(); ++i)
      key_base_it->second[i] = toupper(key_base_it->second[i]);

  // Data structures for computing flow signal SNR
  vector<uint16_t> flow_signal_fz(pac->input_bam->MaxFlowOrderLen());
  vector<int16_t> flow_signal_zm(pac->input_bam->MaxFlowOrderLen());

  // Data structures for alignemnt error parsing
  ReadAlignmentErrors base_space_errors;
  ReadAlignmentErrors flow_space_errors;
  vector<char>     ref_hp_nuc;
  vector<uint16_t> ref_hp_len;
  vector<int16_t>  ref_hp_err;
  vector<uint16_t> ref_hp_flow;
  vector<uint16_t> zeromer_insertion_flow;
  vector<uint16_t> zeromer_insertion_len;
  base_space_errors.Reserve(pac->opt->HistogramLength());
  flow_space_errors.Reserve(pac->opt->NFlow());
  ref_hp_nuc.reserve(pac->opt->NFlow());
  ref_hp_len.reserve(pac->opt->NFlow());
  ref_hp_err.reserve(pac->opt->NFlow());
  ref_hp_flow.reserve(pac->opt->NFlow());
  zeromer_insertion_flow.reserve(pac->opt->NFlow());
  zeromer_insertion_len.reserve(pac->opt->NFlow());

  // Data structure for AQ lengths
  vector<int> aq_length;
  aq_length.reserve(pac->opt->NErrorRates());

  // Loop over mapped reads in the input BAM
  BamAlignment alignment;
  bool done=false;
  while(!done) {
    // Lock the read_mutex while we get the next alignment
    pthread_mutex_lock(pac->read_mutex);
    if(!pac->input_bam->GetNextAlignment(alignment,pac->opt->Program()))
      done=true;
    pthread_mutex_unlock(pac->read_mutex);
    if(done)
      continue;

    // Optionally replicate input to output
    if(pac->opt->OutputBamFilename() != "") {
      if( (output_bam_alignment.size() == pac->opt->OutputBamBufferSize()) ) {
        // Buffered write of reads to output - lock the write_mutex
        pthread_mutex_lock(pac->write_mutex);
        for(vector<BamAlignment>::iterator alignment_it=output_bam_alignment.begin(); alignment_it != output_bam_alignment.end(); ++alignment_it)
          pac->output_bam->SaveAlignment(*alignment_it);
        pthread_mutex_unlock(pac->write_mutex);
        output_bam_alignment.clear();
      }
      output_bam_alignment.push_back(alignment);
    }

    // Check if we should ignore the read based on read group
    string read_group="";
    alignment.GetTag("RG",read_group);
    if((pac->opt->SkipRgSuffix() != "") && (read_group != "")) {
      int pos = read_group.rfind(pac->opt->SkipRgSuffix());
      if((pos != (int) std::string::npos) && (pos == (int)(read_group.length() - pac->opt->SkipRgSuffix().length()))) {
        if(pac->opt->ThreadsShareMemory())
          pthread_mutex_lock(pac->results_mutex);
        pac->alignment_summary->IncrementSkipReadGroupCount();
        if(pac->opt->ThreadsShareMemory())
          pthread_mutex_unlock(pac->results_mutex);
        continue;
      }
    }

    // Check if we should ignore the read based on mapqual
    if(pac->opt->MaxMapQual() > -1 || pac->opt->MinMapQual() > -1) {
      int map_qual = alignment.MapQuality;
      if(pac->opt->MaxMapQual() > -1 && map_qual > pac->opt->MaxMapQual()) {
        if(pac->opt->ThreadsShareMemory())
          pthread_mutex_lock(pac->results_mutex);
        pac->alignment_summary->IncrementSkipMaxMapQualCount();
        if(pac->opt->ThreadsShareMemory())
          pthread_mutex_unlock(pac->results_mutex);
        continue;
      }
      if(pac->opt->MinMapQual() > -1 && map_qual < pac->opt->MinMapQual()) {
        if(pac->opt->ThreadsShareMemory())
          pthread_mutex_lock(pac->results_mutex);
        pac->alignment_summary->IncrementSkipMinMapQualCount();
        if(pac->opt->ThreadsShareMemory())
          pthread_mutex_unlock(pac->results_mutex);
        continue;
      }
    }
      
    bool have_region=false;
    unsigned int region_idx=0;
    if(pac->opt->SpatialStratify()) {
      // Check which region the read belongs to
      bool no_region=false;
      bool out_of_bounds=false;
      have_region = getRegionIndex(region_idx,no_region,out_of_bounds,alignment.Name,pac->opt->ChipOrigin(),pac->opt->ChipDim(),pac->opt->SubregionDim(),pac->opt->NColSubregions());
      assert( !have_region || (region_idx < pac->opt->NSubregions()) );
      if(no_region) {
        if(pac->opt->ThreadsShareMemory())
          pthread_mutex_lock(pac->results_mutex);
        pac->alignment_summary->IncrementNoRegionCount();
        if(pac->opt->ThreadsShareMemory())
          pthread_mutex_unlock(pac->results_mutex);
      }
      if(out_of_bounds) {
        if(pac->opt->ThreadsShareMemory())
          pthread_mutex_lock(pac->results_mutex);
        pac->alignment_summary->IncrementOutOfBoundsCount();
        if(pac->opt->ThreadsShareMemory())
          pthread_mutex_unlock(pac->results_mutex);
      }
    }

    //
    // Accumulate statistics that are independent of alignment
    //

    // Compute Q17 and Q20 lengths
    int Q17_length = 0;
    int Q20_length = 0;
    double num_accumulated_errors = 0.0;
    int increment = alignment.IsReverseStrand() ? -1 : 1;
    int pos = alignment.IsReverseStrand() ? alignment.Length-1 : 0;
    for(int len=0; (pos >= 0) && (pos < alignment.Length); pos += increment, ++len) {
      num_accumulated_errors += pac->opt->QvToErrorRate((int)alignment.Qualities[pos] - 33);
      if (num_accumulated_errors / (len + 1) <= 0.02)
        Q17_length = len + 1;
      if (num_accumulated_errors / (len + 1) <= 0.01)
        Q20_length = len + 1;
    }

    // Record metrics independent of alignment
    if(pac->opt->ThreadsShareMemory())
      pthread_mutex_lock(pac->results_mutex);
    // Record read length
    pac->alignment_summary->IncrementReadCount();
    pac->alignment_summary->AddCalledLength(alignment.Length);
    // Record insert length
    int insert_length = 0;
    if (alignment.GetTag("ZA",insert_length))
      pac->alignment_summary->AddInsertLength(insert_length);
    // Record Q17 and Q20
    pac->alignment_summary->AddQ17Length(Q17_length);
    pac->alignment_summary->AddQ20Length(Q20_length);
    // Record data for system snr
    if(alignment.GetTag("ZM", flow_signal_zm))
      pac->alignment_summary->AddSystemSNR(flow_signal_zm, pac->opt->SeqKey(), flow_orders[read_group]);
    else if(alignment.GetTag("FZ", flow_signal_fz))
      pac->alignment_summary->AddSystemSNR(flow_signal_fz, pac->opt->SeqKey(), flow_orders[read_group]);
    // Record qv histogram
    pac->alignment_summary->AddQVHistogram(alignment.Qualities);
    // read length including barcode
    int bc_bases=0;
    int bc_errors=0;
    if(pac->opt->BcAdjust()) {
      getBarcodeResults(alignment,key_len,bc_bases,bc_errors);
      pac->alignment_summary->AddCalledLengthBc(alignment.Length + bc_bases);
    }
    if(pac->opt->ThreadsShareMemory())
      pthread_mutex_unlock(pac->results_mutex);

    // Only proceed with aligned reads
    if(!alignment.IsMapped())
      continue;
    if(pac->opt->ThreadsShareMemory())
      pthread_mutex_lock(pac->results_mutex);
    pac->alignment_summary->IncrementAlignedCount();
    if(pac->opt->ThreadsShareMemory())
      pthread_mutex_unlock(pac->results_mutex);

    // Only proceed if we have an MD tag
    if(!alignment.HasTag("MD"))
      continue;
    if(pac->opt->ThreadsShareMemory())
      pthread_mutex_lock(pac->results_mutex);
    pac->alignment_summary->IncrementAlignedWithMdCount();
    if(pac->opt->ThreadsShareMemory())
      pthread_mutex_unlock(pac->results_mutex);

    //
    // Now evaluate quantities that depend on alignment
    //

    // Parse alignment from cigar/md formats into more usable structures
    bool invalid_read_bases = false;
    bool invalid_ref_bases  = false;
    bool invalid_cigar      = false;
    parseAlignment(alignment, base_space_errors, flow_space_errors, flow_orders, read_group, reverse_complement_map, pac->opt->EvaluateFlow(), pac->opt->NFlow(),
      pac->opt->EvaluateHp(), invalid_read_bases, invalid_ref_bases, invalid_cigar, ref_hp_nuc, ref_hp_len, ref_hp_err, ref_hp_flow, zeromer_insertion_flow, zeromer_insertion_len);
    if(invalid_cigar || invalid_read_bases) {
      // Skip if the cigar is invalid (violoation of SAM spec) or if read has non-IUPAC bases (which we have not definied how to handle)
      if(pac->opt->ThreadsShareMemory())
        pthread_mutex_lock(pac->results_mutex);
      if(invalid_cigar) {
        cerr << "WARNING: " << pac->opt->Program() << ": incompatible CIGAR and SEQ entries, violation of BAM spec!  Skipping read " << alignment.Name << endl;
        pac->alignment_summary->IncrementInvalidCigarCount();
      }
      if(invalid_read_bases) {
        cerr << "WARNING: " << pac->opt->Program() << ": invalid bases in read - not expected!  Skipping read " << alignment.Name << endl;
        pac->alignment_summary->IncrementInvalidReadBasesCount();
      }
      if(pac->opt->ThreadsShareMemory())
        pthread_mutex_unlock(pac->results_mutex);
      continue;
    }
    if(invalid_ref_bases) {
      if(pac->opt->ThreadsShareMemory())
        pthread_mutex_lock(pac->results_mutex);
      pac->alignment_summary->IncrementInvalidRefBasesCount();
      if(pac->opt->ThreadsShareMemory())
        pthread_mutex_unlock(pac->results_mutex);
    }
    base_space_errors.ConsolidateErrors();
    if(pac->opt->EvaluateHp() && !invalid_ref_bases) {
      assert(ref_hp_nuc.size() == ref_hp_len.size());
      assert(ref_hp_nuc.size() == ref_hp_err.size());
      if(pac->opt->EvaluateFlow() && flow_space_errors.have_data())
        assert(ref_hp_nuc.size() == ref_hp_flow.size());
    }

    // Possible debug output
    if(pac->opt->Debug())
      debug_alignment(pac->opt, alignment, base_space_errors, flow_space_errors, ref_hp_flow);

    // Compute the infamous AQ lengths

    computeAQ(aq_length,pac->opt->AqErrorRate(),base_space_errors);

    if(pac->opt->ThreadsShareMemory())
      pthread_mutex_lock(pac->results_mutex);
    // Accumulate alignment lengths
    pac->alignment_summary->AddAlignedLength(base_space_errors.AlignedLen());
    for(unsigned int i=0; i < pac->opt->NErrorRates(); ++i) {
      if(aq_length[i] >= pac->opt->MinAqLength())
        pac->alignment_summary->AddAqLength(aq_length[i],i);
      }
    // Accumulate per-base error positions
    pac->alignment_summary->AddBasePositionErrorCount(base_space_errors.err(), base_space_errors.err_len());
    // Accumulate NoFlowDataCount
    if(pac->opt->EvaluateFlow() && !flow_space_errors.have_data())
      pac->alignment_summary->IncrementNoFlowDataCount();
    // Accumulate global performance stratified by base, flow or hp
    pac->alignment_summary->AddBasePosition(base_space_errors);
    if(pac->opt->EvaluateFlow() && flow_space_errors.have_data())
      pac->alignment_summary->AddFlowPosition(flow_space_errors);
    if(pac->opt->EvaluateHp() && !invalid_ref_bases) {
      if(pac->opt->EvaluateFlow())
        pac->alignment_summary->AddPerHp(ref_hp_nuc, ref_hp_len, ref_hp_err, ref_hp_flow, zeromer_insertion_flow, zeromer_insertion_len, flow_orders[read_group], pac->opt->IgnoreTerminalHp());
      else
        pac->alignment_summary->AddPerHp(ref_hp_nuc, ref_hp_len, ref_hp_err, pac->opt->IgnoreTerminalHp());
    }
    // Accumulate per-read-group performance stratified by base, flow or hp
    if(read_group == "") {
      pac->alignment_summary->IncrementNoReadGroupCount();
    } else if(read_groups.find(read_group) == read_groups.end()) {
      pac->alignment_summary->IncrementUnmatchedReadGroupCount();
    } else {
      pac->alignment_summary->AddReadGroupBasePosition(read_group, base_space_errors);
      if(pac->opt->EvaluateFlow() && flow_space_errors.have_data())
        pac->alignment_summary->AddReadGroupFlowPosition(read_group, flow_space_errors);
      if(pac->opt->EvaluateHp() && !invalid_ref_bases) {
        if(pac->opt->EvaluateFlow())
          pac->alignment_summary->AddReadGroupPerHp(read_group, ref_hp_nuc, ref_hp_len, ref_hp_err, ref_hp_flow, zeromer_insertion_flow, zeromer_insertion_len, flow_orders[read_group], pac->opt->IgnoreTerminalHp());
        else
          pac->alignment_summary->AddReadGroupPerHp(read_group, ref_hp_nuc, ref_hp_len, ref_hp_err, pac->opt->IgnoreTerminalHp());
      }
    }
    // Accumulate regional performance
    if(pac->opt->SpatialStratify() && have_region) {
      pac->alignment_summary->AddRegionalSummaryBasePosition(region_idx, base_space_errors);
    
     //DO aq length per region
       
       
        for(unsigned int i=0; i < pac->opt->NErrorRates(); ++i) {
             if(aq_length[i] >= pac->opt->MinAqLength())
                   pac->alignment_summary->AddAqLength(region_idx,aq_length[i],i);
        }
        

      if(pac->opt->EvaluateFlow() && pac->opt->EvaluateHp() && !invalid_ref_bases && (ref_hp_len.size() > 0) && (ref_hp_flow.size() > 0)) {
        if(pac->opt->EvaluateFlow())
          pac->alignment_summary->AddRegionalSummaryPerHp(region_idx, ref_hp_len, ref_hp_err, ref_hp_flow, zeromer_insertion_flow, pac->opt->IgnoreTerminalHp());
        else 
          pac->alignment_summary->AddRegionalSummaryPerHp(region_idx, ref_hp_len, ref_hp_err, ref_hp_flow, pac->opt->IgnoreTerminalHp());
      }
    }
    if(pac->opt->EvaluatePerReadPerFlow() && !invalid_ref_bases) {
      // in here we might not have grabbed the results_mutex if memory is not shared, in which case it must be grabbed
      if(!pac->opt->ThreadsShareMemory())
        pthread_mutex_lock(pac->results_mutex);
      if(pac->opt->EvaluateFlow())
        pac->alignment_summary->AddPerReadFlow(alignment.Name,flow_space_errors, ref_hp_len, ref_hp_err, ref_hp_flow, zeromer_insertion_flow, zeromer_insertion_len, pac->opt->IgnoreTerminalHp());
      else
        pac->alignment_summary->AddPerReadFlow(alignment.Name,flow_space_errors, ref_hp_len, ref_hp_err, ref_hp_flow, pac->opt->IgnoreTerminalHp());
      if(pac->alignment_summary->PerReadFlowBufferFull())
        pac->alignment_summary->PerReadFlowBufferFlush();
      if(!pac->opt->ThreadsShareMemory())
        pthread_mutex_unlock(pac->results_mutex);
    }
    if(pac->opt->ThreadsShareMemory())
      pthread_mutex_unlock(pac->results_mutex);

    // Optionally, evaluate alignment including barcode bases.
    // We do this after everything else, as it involves modifying aq_length and the base_space_errors object
    if(pac->opt->BcAdjust()) {
      if(pac->opt->ThreadsShareMemory())
        pthread_mutex_lock(pac->results_mutex);
      scoreBarcodes(base_space_errors,bc_bases,bc_errors,pac->opt->AqErrorRate(),pac->opt->MinAqLength(),aq_length,pac->alignment_summary);
      if(bc_errors > 0)
        pac->alignment_summary->IncrementBarcodeErrorCount();
      if(pac->opt->ThreadsShareMemory())
        pthread_mutex_unlock(pac->results_mutex);
    }
  }//while on all reads

  // Finish write of debug info
  if(pac->opt->EvaluatePerReadPerFlow() && !pac->alignment_summary->PerReadFlowBufferEmpty()) {
    pthread_mutex_lock(pac->results_mutex);
    pac->alignment_summary->PerReadFlowForcedFlush();
    pac->alignment_summary->PerReadFlowCloseH5();
    pthread_mutex_unlock(pac->results_mutex);
  }

  // Finish buffered write of reads to output
  if( (pac->opt->OutputBamFilename() != "") && (output_bam_alignment.size() > 0) ) {
    pthread_mutex_lock(pac->write_mutex);
    for(vector<BamAlignment>::iterator alignment_it=output_bam_alignment.begin(); alignment_it != output_bam_alignment.end(); ++alignment_it)
      pac->output_bam->SaveAlignment(*alignment_it);
    pthread_mutex_unlock(pac->write_mutex);
  }

  return(NULL);
}
