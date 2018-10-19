/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef IONSTATS_ALIGNMENT_H
#define IONSTATS_ALIGNMENT_H

#include "api/BamWriter.h"
#include "api/BamReader.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"

// Option defaults
#define DEFAULT_HELP                       "false"
#define DEFAULT_INPUT_BAM                  ""
#define DEFAULT_OUTPUT_BAM                 ""
#define DEFAULT_OUTPUT_BAM_BUFFER_SIZE     10000
#define DEFAULT_OUTPUT_BAM_COMPRESS        "false"
#define DEFAULT_OUTPUT_JSON                "ionstats_alignment.json"
#define DEFAULT_OUTPUT_H5                  ""
#define DEFAULT_SKIP_RG_SUFFIX             ""
#define DEFAULT_MAX_MAP_QUAL               -1
#define DEFAULT_MIN_MAP_QUAL               -1
#define DEFAULT_HISTOGRAM_LENGTH           400
#define DEFAULT_N_FLOW                     0
#define DEFAULT_MINIMUM_AQ_LENGTH          21
#define DEFAULT_BC_ADJUST                  "false"
//#define DEFAULT_SEQ_KEY                  "TCAG" // Now defined in ionstats.h
#define DEFAULT_EVALUATE_HP                "false"
#define DEFAULT_IGNORE_TERMINAL_HP         "true"
#define DEFAULT_EVALUATE_FLOW              "true"
#define DEFAULT_EVALUATE_PER_READ_PER_FLOW "false"
#define DEFAULT_CHIP_ORIGIN                ""
#define DEFAULT_CHIP_DIM                   ""
#define DEFAULT_SUBREGION_DIM              ""
#define DEFAULT_AQ_ERROR_RATES             "0.2,0.1,0.02,0.01,0.001,0"
#define DEFAULT_MAX_HP                     10
#define DEFAULT_SUBREGION_MAX_HP           1
#define DEFAULT_DEBUG                      "false"
#define DEFAULT_DEBUG_ERROR_FLOW           -1
#define DEFAULT_DEBUG_ALIGNED_FLOW         -1
#define DEFAULT_DEBUG_POSITIVE_REF_FLOW    -1
#define DEFAULT_N_THREADS                  5
#define DEFAULT_THREADS_SHARE_MEMORY       "false"

using namespace std;
using namespace BamTools;

class IonstatsAlignmentOptions {

public:
  IonstatsAlignmentOptions () :
  help_(false),
  program_(""),
  output_bam_filename_(""),
  output_bam_buffer_size_(0),
  output_bam_compress_(false),
  output_json_filename_(""),
  output_h5_filename_(""),
  skip_rg_suffix_(""),
  max_map_qual_(0),
  min_map_qual_(0),
  histogram_length_(0),
  n_flow_(0),
  minimum_aq_length_(0),
  bc_adjust_(false),
  seq_key_(""),
  evaluate_hp_(false),
  ignore_terminal_hp_(false),
  evaluate_flow_(false),
  evaluate_per_read_per_flow_(false),
  max_hp_(0),
  max_subregion_hp_(0),
  debug_(false),
  debug_error_flow_(0),
  debug_aligned_flow_(0),
  debug_positive_ref_flow_(0),
  spatial_stratify_(false),
  n_col_subregions_(0),
  n_row_subregions_(0),
  n_subregions_(0),
  n_error_rates_(0),
  max_flow_order_len_(0),
  n_threads_(0),
  threads_share_memory_(false)
  {}
  ~IonstatsAlignmentOptions () {}

  void Initialize(int argc, const char *argv[]);
  void InitializeFromOptArgs(OptArgs &opts, const string &program_str);
  void WriteHelp(void);


  // accessors by reference
  unsigned int & MaxFlowOrderLen(void)   { return(max_flow_order_len_); };

  // accessors by value
  bool                   Help(void)                           { return(help_); };
  string &               Program(void)                        { return(program_); };
  vector<string> &       InputBamFilename(void)               { return(input_bam_filename_); };
  string &               OutputBamFilename(void)              { return(output_bam_filename_); };
  bool                   OutputBamCompress(void)              { return(output_bam_compress_); };
  string &               SeqKey(void)                         { return(seq_key_); };
  string &               SkipRgSuffix(void)                   { return(skip_rg_suffix_); };
  int                    HistogramLength(void)                { return(histogram_length_); };
  bool                   EvaluateFlow(void)                   { return(evaluate_flow_); };
  bool                   EvaluatePerReadPerFlow(void)         { return(evaluate_per_read_per_flow_); };
  int                    NFlow(void)                          { return(n_flow_); };
  bool                   EvaluateHp(void)                     { return(evaluate_hp_); };
  unsigned int           MaxHp(void)                          { return(max_hp_); };
  unsigned int           NErrorRates(void)                    { return(n_error_rates_); };
  bool                   BcAdjust(void)                       { return(bc_adjust_); };
  string &               OutputH5Filename(void)               { return(output_h5_filename_); };
  string &               OutputJsonFilename(void)             { return(output_json_filename_); };
  bool                   SpatialStratify(void)                { return(spatial_stratify_); };
  unsigned int           NSubregions(void)                    { return(n_subregions_); };
  unsigned int           MaxSubregionHp(void)                 { return(max_subregion_hp_); };
  unsigned int           OutputBamBufferSize(void)            { return(output_bam_buffer_size_); };
  int                    MaxMapQual(void)                     { return(max_map_qual_); };
  int                    MinMapQual(void)                     { return(min_map_qual_); };
  vector<unsigned int> & ChipOrigin(void)                     { return(chip_origin_); };
  vector<unsigned int> & ChipDim(void)                        { return(chip_dim_); };
  vector<unsigned int> & SubregionDim(void)                   { return(subregion_dim_); };
  unsigned int           NColSubregions(void)                 { return(n_col_subregions_); };
  vector<unsigned int> & RegionSpecificOrigin(unsigned int i) { return(region_specific_origin_[i]); };
  vector<unsigned int> & RegionSpecificDim(unsigned int i)    { return(region_specific_dim_[i]); };
  bool                   Debug(void)                          { return(debug_); };
  int                    DebugErrorFlow(void)                 { return(debug_error_flow_); };
  int                    DebugAlignedFlow(void)               { return(debug_aligned_flow_); };
  int                    DebugPositiveRefFlow(void)           { return(debug_positive_ref_flow_); };
  vector<double> &       AqErrorRate(void)                    { return(aq_error_rate_); };
  int                    MinAqLength(void)                    { return(minimum_aq_length_); };
  bool                   IgnoreTerminalHp(void)               { return(ignore_terminal_hp_); };
  vector<string> &       RegionName(void)                     { return(region_name_); };
  double                 QvToErrorRate(int i)                 { return(qv_to_error_rate_[i]); };
  unsigned int           NThreads(void)                       { return(n_threads_); };
  bool                   ThreadsShareMemory(void)             { return(threads_share_memory_); };

private:

  vector<string> input_bam_filename_;
  vector<unsigned int> chip_origin_;
  vector<unsigned int> chip_dim_;
  vector<unsigned int> subregion_dim_;
  vector<double> aq_error_rate_;
  vector<double> qv_to_error_rate_;
  vector<string> region_name_;
  vector< vector<unsigned int> > region_specific_origin_;
  vector< vector<unsigned int> > region_specific_dim_;

  bool help_;
  string program_;
  string output_bam_filename_;
  unsigned int output_bam_buffer_size_;
  bool output_bam_compress_;
  string output_json_filename_;
  string output_h5_filename_;
  string skip_rg_suffix_;
  int max_map_qual_;
  int min_map_qual_;
  int histogram_length_;
  int n_flow_;
  int minimum_aq_length_;
  bool bc_adjust_;
  string seq_key_;
  bool evaluate_hp_;
  bool ignore_terminal_hp_;
  bool evaluate_flow_;
  bool evaluate_per_read_per_flow_;
  unsigned int max_hp_;
  unsigned int max_subregion_hp_;
  bool debug_;
  int debug_error_flow_;
  int debug_aligned_flow_;
  int debug_positive_ref_flow_;
  bool spatial_stratify_;
  unsigned int n_col_subregions_;
  unsigned int n_row_subregions_;
  unsigned int n_subregions_;
  unsigned int n_error_rates_;
  unsigned int max_flow_order_len_;
  unsigned int n_threads_;
  bool threads_share_memory_;
};

class IonstatsAlignmentBamReader {

public:
  IonstatsAlignmentBamReader() : max_flow_order_len_(0) {}
  ~IonstatsAlignmentBamReader() {
    if(input_bam_.IsOpen())
      input_bam_.Close();
  }

  int Initialize(
    const vector<string> &input_bam_filename,
    const string &program,
    const string &seq_key,
    const string &skip_rg_suffix,
    SamHeader &sam_header,
    RefVector &reference_data
  );
  bool GetNextAlignment(BamAlignment &alignment, string &program);

  map< string, int > &    ReadGroups(void)      { return(read_groups_); };
  map< string, string > & FlowOrders(void)      { return(flow_orders_); };
  map< string, string > & KeyBases(void)        { return(key_bases_); };
  map< string, int > &    KeyLen(void)          { return(key_len_); };
  unsigned int            MaxFlowOrderLen(void) { return(max_flow_order_len_); };
private:

  vector<string> input_bam_filename_;
  map< string, int > read_groups_;
  map< string, string > flow_orders_;
  map< string, string > key_bases_;
  map< string, int > key_len_;
  unsigned int max_flow_order_len_;
  BamReader input_bam_;
  vector<string>::iterator input_bam_filename_it_;
};

#endif // IONSTATS_ALIGNMENT_H
