/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "ionstats.h"
#include "ionstats_data.h"

#include <string>
#include <fstream>
#include <map>
#include <limits>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "api/BamReader.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"

#include "OptArgs.h"
#include "Utils.h"
#include "IonVersion.h"
#include "ion_util.h"
#include "hdf5.h"

// Option defaults
#define DEFAULT_HELP              false
#define DEFAULT_INPUT_BAM         ""
#define DEFAULT_OUTPUT_JSON       "ionstats_alignment.json"
#define DEFAULT_OUTPUT_H5         "ionstats_error_summary.h5"
#define DEFAULT_SKIP_RG_SUFFIX    ""
#define DEFAULT_HISTOGRAM_LENGTH  400
#define DEFAULT_MINIMUM_AQ_LENGTH 21
#define DEFAULT_BC_ADJUST         false
#define DEFAULT_SEQ_KEY           "TCAG"
#define DEFAULT_EVALUATE_HP       "false"
#define DEFAULT_EVALUATE_FLOW     "true"
#define DEFAULT_CHIP_ORIGIN       ""
#define DEFAULT_CHIP_DIM          ""
#define DEFAULT_SUBREGION_DIM     ""
#define DEFAULT_AQ_ERROR_RATES    "0.2,0.1,0.02,0.01,0.001,0"
#define DEFAULT_MAX_HP            10
#define DEFAULT_SUBREGION_MAX_HP  1

#define MAX_GROUP_NAME_LEN 5000

using namespace std;
using namespace BamTools;

bool isAqLength(string s);
bool revSort(double i, double j) { return(i>j); };
void initialize_reverse_complement_map(map<char, char> &rc);
int reverse_complement(string &b, const map<char, char> &rc);
void computeAQ(vector<int> &aq_length, vector<double> &aq_error_rate, ReadAlignmentErrors &base_space_errors);
int toPhred(double e);
bool getRegionIndex(unsigned int &region_idx, bool &no_region, bool &out_of_bounds, string &name, vector<unsigned int> &chip_origin, vector<unsigned int> &chip_dim, vector<unsigned int> &subregion_dim, unsigned int n_col_subregions);
void parseMD(const string &MD_tag, vector<char> &MD_op, vector<int> &MD_len, vector<string> &MD_seq);
void getReadGroupInfo(const BamReader &input_bam, map< string, string > &flow_orders, unsigned int &max_flow_order_len, map< string, string > &key_bases, map< string, int > &key_len, string &seq_key, string &skip_rg_suffix);
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
  bool                    evaluate_hp,
  bool &                  invalid_read_bases,
  bool &                  invalid_ref_bases,
  bool &                  invalid_cigar,
  vector<char> &          ref_hp_nuc,
  vector<uint16_t> &      ref_hp_len,
  vector<int16_t> &       ref_hp_err,
  vector<uint16_t> &      ref_hp_flow
);
void scoreBarcodes(ReadAlignmentErrors &base_space_errors, int bc_bases, int bc_errors, vector<double> &aq_error_rate, int minimum_aq_length, vector<int> &aq_length, ReadLengthHistogram &aligned_histogram_bc, vector< ReadLengthHistogram > &aq_histogram_bc);
string getReferenceBases(const string &read_bases, vector< CigarOp > CigarData, const vector<char> &MD_op, vector<int> MD_len, vector<string> &MD_seq, const bool rev_strand, const map<char,char> &reverse_complement_map);
void getHpBreakdown(const string &bases, vector<char> &hp_nuc, vector<uint16_t> &hp_len);
int writeIonstatsAlignmentJson(
  string &json_filename,
  vector<double> &aq_error_rate,
  bool bc_adjust,
  bool evaluate_flow,
  ReadLengthHistogram &called_histogram,    ReadLengthHistogram &aligned_histogram,    vector< ReadLengthHistogram > &aq_histogram,
  ReadLengthHistogram &called_histogram_bc, ReadLengthHistogram &aligned_histogram_bc, vector< ReadLengthHistogram > &aq_histogram_bc,
  ReadLengthHistogram &total_insert_histo, ReadLengthHistogram &total_Q17_histo, ReadLengthHistogram &total_Q20_histo,
  MetricGeneratorSNR &system_snr, BaseQVHistogram &qv_histogram,
  SimpleHistogram &base_position_error_count,
  ErrorData &base_position,
  ErrorData &flow_position,
  map< string, ErrorData > &read_group_base_position,
  map< string, ErrorData > &read_group_flow_position
);
void writeIonstatsH5(
  string h5_filename, 
  ErrorData & base_position,
  ErrorData & flow_position,
  HpData &    per_hp,
  map< string, ErrorData > & read_group_base_position,
  map< string, ErrorData > & read_group_flow_position,
  map< string, HpData > &    read_group_per_hp,
  vector<string> &region_name,
  vector< RegionalSummary > & regional_summary
);
void AddToReadLengthHistogramFromJson(const Json::Value &input_json, const string &var_name, ReadLengthHistogram &hist);
void AddToSimpleHistogramFromJson(const Json::Value &input_json, const string &var_name, SimpleHistogram &hist, bool &found);
bool getTagZF(BamAlignment & alignment, uint32_t &flow_idx);
bool hasInvalidCigar(BamAlignment & alignment);
bool hasInvalidBases(string &b);
string transform_proton_block_read_group_name(const string &group_name);
void GetAggregatorSize(map<string, ErrorData> &error_data, map<string, HpData> &hp_data, map<string, RegionalSummary> &regional_summary, unsigned int &group_count, uint64_t &total_bytes);


enum align_t {ALIGN_MATCH, ALIGN_INS, ALIGN_DEL, ALIGN_SUB};

typedef struct error_data_merge_t {
  map< string, ErrorData > error_data;
  bool merge_proton_blocks;
} error_data_merge_t;

typedef struct hp_data_merge_t {
  map< string, HpData > hp_data;
  bool merge_proton_blocks;
} hp_data_merge_t;

typedef struct regional_summary_merge_t {
  map< string, RegionalSummary > regional_summary;
  bool merge_proton_blocks;
} regional_summary_merge_t;

void hpAdvance(
  // Inputs
  align_t                   alignment_type,
  int                       advance,         // number of bases in perfectly-aligned stretch
  // Data for tracking flow
  bool                      evaluate_flow,
  unsigned int              flow_idx,
  const string &            flow_order,
  const string &            read_bases,
  // Data for tracking position in bases/HPs
  int                       read_idx,         // index of where we are in read_bases
  const vector<char> &      read_hp_nuc,      // read hp nucs
  const vector<uint16_t> &  read_hp_len,      // read hp lengths
  const vector<uint16_t> &  read_hp_cum_len,  // read hp cumulative lengths
  int                       ref_idx,          // index of where we are in ref_bases
  const vector<char> &      ref_hp_nuc,       // read hp nucs
  const vector<uint16_t> &  ref_hp_len,       // read hp lengths
  const vector<uint16_t> &  ref_hp_cum_len,   // read hp cumulative lengths
  // Objects that may be modified
  int &                     stored_read_match_count, // number of read bases matching current ref hp that have been seen so far
  unsigned int &            read_hp_idx,             // index of where we are in read_hp_nuc, read_hp_len, read_hp_cum_len
  unsigned int &            ref_hp_idx,              // index of where we are in ref_hp_nuc, ref_hp_len, ref_hp_cum_len
  vector<int16_t> &         ref_hp_err,              //
  vector<uint16_t> &        ref_hp_flow
);

void IonstatsAlignmentHelp()
{
  cout << endl;
  cout << "ionstats " << IonVersion::GetVersion() << "-" << IonVersion::GetRelease() << " (" << IonVersion::GetSvnRev() << ") - Generate performance metrics and statistics for Ion sequences." << endl;
  cout << endl;
  cout << "Usage:   ionstats alignment -i in.bam [options]" << endl;
  cout << endl;
  cout << "General options:" << endl;
  cout << "  --help                  BOOL      print this help message [" << DEFAULT_HELP << "]" << endl;
  cout << "  -i,--input              STRING    input BAM (mapped) [" << DEFAULT_INPUT_BAM << "]" << endl;
  cout << "  -o,--output             STRING    output json file [" << DEFAULT_OUTPUT_JSON << "]" << endl;
  cout << "  --output-h5             STRING    output hdf5 file [" << DEFAULT_OUTPUT_H5 << "]" << endl;
  cout << "  --skip-rg-suffix        STRING    ignore read groups matching suffix [\"" << DEFAULT_SKIP_RG_SUFFIX << "\"]" << endl;
  cout << "  -h,--histogram-length   INT       read length histogram cutoff [" << DEFAULT_HISTOGRAM_LENGTH << "]" << endl;
  cout << "  -m,--minimum-aq-length  INT       minimum AQ read length [" << DEFAULT_MINIMUM_AQ_LENGTH << "]" << endl;
  cout << "  -b,--bc-adjust          BOOL      give credit to barcode bases, assumes barcodes have no errors [" << DEFAULT_BC_ADJUST << "]" << endl;
  cout << "  -k,--key                STRING    seq key to remove when using -b option [" << DEFAULT_SEQ_KEY << "]" << endl;
  cout << "  -a,--aq-error-rates     STRING    error rates for which to evaluate AQ lengths [" << DEFAULT_AQ_ERROR_RATES << "]" << endl;
  cout << "  --evaluate-flow         BOOL      evaluate per-flow accuracy [" << DEFAULT_EVALUATE_FLOW << "]" << endl;
  cout << "  --evaluate-hp           BOOL      evaluate homopolymer accuracy [" << DEFAULT_EVALUATE_HP << "]" << endl;
  cout << "  --max-hp                INT       max HP length for chip-wide summary [" << DEFAULT_MAX_HP << "]" << endl;
  cout << "  --max-subregion-hp      INT       max HP length for regional summary [" << DEFAULT_SUBREGION_MAX_HP << "]" << endl;
  cout << endl;
  cout << "Options for spatial stratification of results.  All 3 options must be used together." << endl;
  cout << "  Each option specifies two comma-separated values in the form x,y" << endl;
  cout << "  --chip-origin           INT,INT   zero-based coordinate origin of chip [" << DEFAULT_CHIP_ORIGIN << "]" << endl;
  cout << "  --chip-dim              INT,INT   dimensions of chip [" << DEFAULT_CHIP_DIM << "]" << endl;
  cout << "  --subregion-dim         INT,INT   dimensions of sub-regions for spatial stratification [" << DEFAULT_SUBREGION_DIM << "]" << endl;
  cout << endl;
}

// Metrics in ionstats_alignment.json should carry the following data:
//
// - Histogram of read lengths (copy of basecaller's "full" histogram) - done
// - Histogram of aligned lengths - done
// - Histogram of AQ## lengths - done
// - Genome name, genome version, mapper version, genome size - ???
// - Error rate by position: numerator and denominator - ??? (actually might be very easy)


int IonstatsAlignment(int argc, const char *argv[])
{
  string program = string(argv[0]) + " " + string(argv[1]);
  OptArgs opts;
  opts.ParseCmdLine(argc-1, argv+1);
  bool help                   = opts.GetFirstBoolean('-', "help",              DEFAULT_HELP);
  if(help) {
    IonstatsAlignmentHelp();
    return(EXIT_SUCCESS);
  }

  vector<string> input_bam_filename;
  opts.GetOption(input_bam_filename, DEFAULT_INPUT_BAM, 'i', "input");

  string output_json_filename   = opts.GetFirstString ('o', "output",            DEFAULT_OUTPUT_JSON);
  string output_h5_filename     = opts.GetFirstString ('-', "output-h5",         DEFAULT_OUTPUT_H5);
  string skip_rg_suffix         = opts.GetFirstString ('-', "skip-rg-suffix",    DEFAULT_SKIP_RG_SUFFIX);
  int histogram_length          = opts.GetFirstInt    ('h', "histogram-length",  DEFAULT_HISTOGRAM_LENGTH);
  int minimum_aq_length         = opts.GetFirstInt    ('m', "minimum-aq-length", DEFAULT_MINIMUM_AQ_LENGTH);
  bool bc_adjust                = opts.GetFirstBoolean('b', "bc-adjust",         DEFAULT_BC_ADJUST);
  string seq_key                = opts.GetFirstString ('k', "key",               DEFAULT_SEQ_KEY);
  bool evaluate_hp              = opts.GetFirstBoolean('-', "evaluate-hp",       DEFAULT_EVALUATE_HP);
  bool evaluate_flow            = opts.GetFirstBoolean('-', "evaluate-flow",     DEFAULT_EVALUATE_FLOW);
  unsigned int max_hp           = opts.GetFirstInt    ('-', "max-hp",            DEFAULT_MAX_HP);
  unsigned int max_subregion_hp = opts.GetFirstInt    ('-', "max-subregion-hp",  DEFAULT_SUBREGION_MAX_HP);

  // Options related to spatial stratification of results
  vector<unsigned int> chip_origin,chip_dim,subregion_dim;
  opts.GetOption(chip_origin,   DEFAULT_CHIP_ORIGIN,   '-', "chip-origin");
  opts.GetOption(chip_dim,      DEFAULT_CHIP_DIM,      '-', "chip-dim");
  opts.GetOption(subregion_dim, DEFAULT_SUBREGION_DIM, '-', "subregion-dim");
  bool spatial_stratify=false;
  unsigned int n_col_subregions=0;
  unsigned int n_row_subregions=0;
  if(checkDimensions(spatial_stratify,chip_origin,chip_dim,subregion_dim,n_col_subregions,n_row_subregions))
    exit(EXIT_FAILURE);
  unsigned int n_subregions = n_col_subregions * n_row_subregions;
  vector<string> region_name;
  vector< vector<unsigned int> > region_specific_origin;
  vector< vector<unsigned int> > region_specific_dim;
  if(n_subregions > 0)
    assignRegionNames(region_name,chip_origin,chip_dim,subregion_dim,region_specific_origin,region_specific_dim);

  // Options for specifying AQ error rates, which in turn specify AQ levels
  vector<double> aq_error_rate;
  opts.GetOption(aq_error_rate, DEFAULT_AQ_ERROR_RATES, 'a', "aq-error-rates");
  for(unsigned int i=0; i<aq_error_rate.size(); ++i)
    if(aq_error_rate[i] < 0 || aq_error_rate[i] >= 1)
      cerr << "WARNING: " << program << ": bad value for aq-error-rates option, must be in range [0,1), value is " << aq_error_rate[i] << endl;
  // later we make shortcuts based on the assumption that this vector is sorted
  sort(aq_error_rate.begin(), aq_error_rate.end(), revSort);
  unsigned int n_error_rates = aq_error_rate.size();

  if((argc < 3) or (input_bam_filename.size()==0)) {
    IonstatsAlignmentHelp();
    return EXIT_FAILURE;
  }

  vector<double> qv_to_error_rate(256);
  for (unsigned int qv = 0; qv < qv_to_error_rate.size(); qv++)
    qv_to_error_rate[qv] =  pow(10.0,-0.1*(double)qv);

  // Initialize reverse-complement map
  map< char, char > reverse_complement_map;
  initialize_reverse_complement_map(reverse_complement_map);

  //
  // Declare structures into which results will be accumulated
  //
  // Called & aligned lengths
  ReadLengthHistogram called_histogram;
  ReadLengthHistogram aligned_histogram;
  vector< ReadLengthHistogram > aq_histogram(n_error_rates);
  ReadLengthHistogram total_insert_histo;
  ReadLengthHistogram total_Q17_histo;
  ReadLengthHistogram total_Q20_histo;
  MetricGeneratorSNR system_snr;
  BaseQVHistogram qv_histogram;
  // Called & aligned lengths including barcodes
  ReadLengthHistogram called_histogram_bc;
  ReadLengthHistogram aligned_histogram_bc;
  vector< ReadLengthHistogram > aq_histogram_bc(n_error_rates);
  // Per-base and per-flow error data
  SimpleHistogram base_position_error_count;
  ErrorData base_position;
  ErrorData flow_position;
  HpData per_hp;
  // Read Group per-base and per-flow error data
  map< string, ErrorData > read_group_base_position;
  map< string, ErrorData > read_group_flow_position;
  map< string, HpData > read_group_per_hp;
  // Regional summary data
  vector< RegionalSummary > regional_summary(n_subregions);

  //
  // Initialize structures into which results will be accumulated
  //
  // Called & aligned lengths
  called_histogram.Initialize(histogram_length);
  aligned_histogram.Initialize(histogram_length);
  for(unsigned int i=0; i<aq_histogram.size(); ++i)
    aq_histogram[i].Initialize(histogram_length);
  total_insert_histo.Initialize(histogram_length);;
  total_Q17_histo.Initialize(histogram_length);;
  total_Q20_histo.Initialize(histogram_length);;
  // Called & aligned lengths including barcodes
  if(bc_adjust) {
    called_histogram_bc.Initialize(histogram_length);
    aligned_histogram_bc.Initialize(histogram_length);
    for(unsigned int i=0; i<aq_histogram.size(); ++i)
      aq_histogram_bc[i].Initialize(histogram_length);
  }
  // Per-base and per-flow error data
  base_position_error_count.Initialize(histogram_length);
  base_position.Initialize(histogram_length);
  unsigned int flow_histogram_length = ceil(TYPICAL_FLOWS_PER_BASE*histogram_length);
  if(evaluate_flow)
    flow_position.Initialize(flow_histogram_length);
  if(evaluate_hp)
    per_hp.Initialize(max_hp);

  // Regional summary data
  if(spatial_stratify)
    for(unsigned int i=0; i<n_subregions; ++i)
      regional_summary[i].Initialize(max_subregion_hp,flow_histogram_length,region_specific_origin[i],region_specific_dim[i]);

  for(vector<string>::iterator input_bam_filename_it=input_bam_filename.begin(); input_bam_filename_it != input_bam_filename.end(); ++input_bam_filename_it) {
    // open BAM
    BamReader input_bam;
    if (!input_bam.Open(*input_bam_filename_it)) {
      fprintf(stderr, "[ionstats] ERROR: cannot open %s\n", input_bam_filename_it->c_str());
      return(EXIT_FAILURE);
    }

    // Get flow orders and keys for each read group
    map< string, string > flow_orders;
    map< string, string > key_bases;
    map< string, int > key_len;
    unsigned int max_flow_order_len=0;
    getReadGroupInfo(input_bam,flow_orders,max_flow_order_len,key_bases,key_len,seq_key,skip_rg_suffix);

    // Initialize per-read-group structures for any new read groups
    for(map< string, string >::iterator it = key_bases.begin(); it != key_bases.end(); ++it) {
      // Check if the read group has already been seen
      map< string, ErrorData >::iterator rg_it = read_group_base_position.find(it->first);
      if(rg_it != read_group_base_position.end())
        continue;
      // It is a new read group, need to initialize
      ErrorData temp_error_data;
      read_group_base_position[it->first] = temp_error_data;
      read_group_base_position[it->first].Initialize(histogram_length);
      if(evaluate_flow) {
        read_group_flow_position[it->first] = temp_error_data;
        read_group_flow_position[it->first].Initialize(flow_histogram_length);
      }
      if(evaluate_hp) {
        HpData temp_hp_data;
        read_group_per_hp[it->first] = temp_hp_data;
        read_group_per_hp[it->first].Initialize(max_hp);
      }
    }

    // Loop over mapped reads in the input BAM
    BamAlignment alignment;
    uint64_t n_read=0;
    uint64_t n_aligned=0;
    uint64_t n_aligned_with_MD=0;
    uint64_t n_invalid_cigar=0;
    uint64_t n_invalid_read_bases=0;
    uint64_t n_invalid_ref_bases=0;
    uint64_t n_no_flow_data=0;
    uint64_t n_barcode_error=0;
    uint64_t n_no_read_group=0;
    uint64_t n_unmatched_read_group=0;
    uint64_t n_no_region=0;
    uint64_t n_out_of_bounds=0;
    int bc_bases=0;
    int bc_errors=0;

    while(input_bam.GetNextAlignment(alignment)) {

      // Check if we should ignore the read
      string read_group="";
      alignment.GetTag("RG",read_group);
      if((skip_rg_suffix != "") && (read_group != "")) {
        int pos = read_group.rfind(skip_rg_suffix);
        if((pos != (int) std::string::npos) && (pos == (int)(read_group.length() - skip_rg_suffix.length())))
          continue;
      }

      n_read++;

      bool have_region=false;
      unsigned int region_idx=0;
      if(spatial_stratify) {
        // Check which region the read belongs to
        bool no_region=false;
        bool out_of_bounds=false;
        have_region = getRegionIndex(region_idx,no_region,out_of_bounds,alignment.Name,chip_origin,chip_dim,subregion_dim,n_col_subregions);
        assert( !have_region || (region_idx < n_subregions) );
        if(no_region)
          n_no_region++;
        if(out_of_bounds)
          n_out_of_bounds++;
      }

      //
      // Accumulate statistics that are independent of alignment
      //

      // read length
      called_histogram.Add(alignment.Length);

      // insert length
      int insert_length = 0;
      if (alignment.GetTag("ZA",insert_length))
        total_insert_histo.Add(insert_length);

      // Compute and record Q17 and Q20
      int Q17_length = 0;
      int Q20_length = 0;
      double num_accumulated_errors = 0.0;
      int increment = alignment.IsReverseStrand() ? -1 : 1;
      int pos = alignment.IsReverseStrand() ? alignment.Length-1 : 0;
      for(int len=0; (pos >= 0) && (pos < alignment.Length); pos += increment, ++len) {
        num_accumulated_errors += qv_to_error_rate[(int)alignment.Qualities[pos] - 33];
        if (num_accumulated_errors / (len + 1) <= 0.02)
          Q17_length = len + 1;
        if (num_accumulated_errors / (len + 1) <= 0.01)
          Q20_length = len + 1;
      }
      total_Q17_histo.Add(Q17_length);
      total_Q20_histo.Add(Q20_length);

      // Record data for system snr
      vector<uint16_t> flow_signal_fz(max_flow_order_len);
      vector<int16_t> flow_signal_zm(max_flow_order_len);
      if(alignment.GetTag("ZM", flow_signal_zm))
        system_snr.Add(flow_signal_zm, key_bases[read_group].c_str(), flow_orders[read_group]);
      else if(alignment.GetTag("FZ", flow_signal_fz))
        system_snr.Add(flow_signal_fz, key_bases[read_group].c_str(), flow_orders[read_group]);

      // Record qv histogram
      qv_histogram.Add(alignment.Qualities);

      // read length including barcode
      if(bc_adjust) {
        getBarcodeResults(alignment,key_len,bc_bases,bc_errors);
        called_histogram_bc.Add(alignment.Length + bc_bases);
      }

      // Only proceed with aligned reads
      if(!alignment.IsMapped())
        continue;
      n_aligned++;

      // Only proceed if we have an MD tag
      if(!alignment.HasTag("MD"))
        continue;
      n_aligned_with_MD++;


      //
      // Now evaluate quantities that depend on alignment
      //

      // Parse alignment from cigar/md formats into more usable structures
      bool invalid_cigar      = false;
      bool invalid_read_bases = false;
      bool invalid_ref_bases  = false;
      ReadAlignmentErrors base_space_errors;
      ReadAlignmentErrors flow_space_errors;
      vector<char>     ref_hp_nuc;
      vector<uint16_t> ref_hp_len;
      vector<int16_t>  ref_hp_err;
      vector<uint16_t> ref_hp_flow;
      parseAlignment(alignment, base_space_errors, flow_space_errors, flow_orders, read_group, reverse_complement_map, evaluate_flow,
        evaluate_hp, invalid_read_bases, invalid_ref_bases, invalid_cigar, ref_hp_nuc, ref_hp_len, ref_hp_err, ref_hp_flow);
      if(invalid_cigar || invalid_read_bases) {
        // Skip if the cigar is invalid (violoation of SAM spec) or if read has non-IUPAC bases (which we have not definied how to handle)
        if(invalid_cigar) {
          cerr << "WARNING: incompatible CIGAR and SEQ entries, violation of BAM spec!  Skipping read " << alignment.Name << endl;
          n_invalid_cigar++;
        }
        if(invalid_read_bases) {
          cerr << "WARNING: invalid bases in read - not expected!  Skipping read " << alignment.Name << endl;
          n_invalid_read_bases++;
        }
        continue;
      }
      if(invalid_ref_bases)
          n_invalid_ref_bases++;
      base_space_errors.ConsolidateErrors();
      if(evaluate_hp && !invalid_ref_bases) {
        assert(ref_hp_nuc.size() == ref_hp_len.size());
        assert(ref_hp_nuc.size() == ref_hp_err.size());
        if(evaluate_flow && flow_space_errors.have_data())
          assert(ref_hp_nuc.size() == ref_hp_flow.size());
      }

      // Compute the infamous AQ lengths
      vector<int> aq_length;
      computeAQ(aq_length,aq_error_rate,base_space_errors);

      // Accumulate alignment lengths
      aligned_histogram.Add(base_space_errors.AlignedLen());
      for(unsigned int i=0; i < n_error_rates; ++i) {
        if(aq_length[i] >= minimum_aq_length)
          aq_histogram[i].Add(aq_length[i]);
      }
      // Accumulate per-base error positions
      const vector<uint16_t> &err_pos = base_space_errors.err();
      const vector<uint16_t> &err_len = base_space_errors.err_len();
      for(unsigned int i=0; i<err_pos.size(); ++i)
        base_position_error_count.Add(err_pos[i],err_len[i]);

      if(evaluate_flow && !flow_space_errors.have_data())
        n_no_flow_data++;

      // Accumulate global performance stratified by base, flow or hp
      base_position.Add(base_space_errors);
      if(evaluate_flow && flow_space_errors.have_data())
        flow_position.Add(flow_space_errors);
      if(evaluate_hp && !invalid_ref_bases)
         per_hp.Add(ref_hp_nuc, ref_hp_len, ref_hp_err);

      // Accumulate per-read-group performance stratified by base, flow or hp
      if(read_group == "") {
        n_no_read_group++;
      } else if(key_bases.find(read_group) == key_bases.end()) {
        n_unmatched_read_group++;
      } else {
        read_group_base_position[read_group].Add(base_space_errors);
        if(evaluate_flow && flow_space_errors.have_data())
          read_group_flow_position[read_group].Add(flow_space_errors);
        if(evaluate_hp && !invalid_ref_bases)
           read_group_per_hp[read_group].Add(ref_hp_nuc, ref_hp_len, ref_hp_err);
      }

      // Accumulate regional performance
      if(spatial_stratify && have_region) {
        regional_summary[region_idx].Add(base_space_errors);
        if(evaluate_flow && evaluate_hp && !invalid_ref_bases)
           regional_summary[region_idx].Add(ref_hp_len, ref_hp_err, ref_hp_flow);
      }

      // Optionally, evaluate alignment including barcode bases.
      // We do this after everything else, as it involves modifying aq_length and the base_space_errors object
      if(bc_adjust) {
        scoreBarcodes(base_space_errors,bc_bases,bc_errors,aq_error_rate,minimum_aq_length,aq_length,aligned_histogram_bc,aq_histogram_bc);
        if(bc_errors > 0)
          n_barcode_error++;
      }
    }
    input_bam.Close();

    if(spatial_stratify && ( (n_no_region > 0) || (n_out_of_bounds > 0) ) )
      cerr << "WARNING: " << program << ": " << *input_bam_filename_it << ": of " << n_read << " reads, " << n_no_region << " have no region and " << n_out_of_bounds << " are out-of-bounds." << endl;
    if(evaluate_flow && n_no_flow_data > 0)
      cerr << "WARNING: " << program << ": " << *input_bam_filename_it << ": " << n_no_flow_data << " of " << n_aligned << " aligned reads have no flow data" << endl;
    if(n_aligned_with_MD < n_aligned)
      cerr << "WARNING: " << program << ": " << *input_bam_filename_it << ": " << (n_aligned - n_aligned_with_MD) << " of " << n_aligned << " aligned reads have no MD tag" << endl;
    if(n_invalid_cigar > 0)
      cout << "WARNING: " << program << ": " << *input_bam_filename_it << ": " << n_invalid_cigar << " of " << n_aligned_with_MD << " aligned reads with MD tag have incompatible CIGAR and SEQ entries" << endl;
    if(n_invalid_read_bases > 0)
      cout << "NOTE: " << program << ": " << *input_bam_filename_it << ": " << n_invalid_read_bases << " of " << n_aligned_with_MD << " aligned reads with MD tag have one or more non-[ACGT] bases in the read" << endl;
    if(n_invalid_ref_bases > 0)
      cout << "NOTE: " << program << ": " << *input_bam_filename_it << ": " << n_invalid_ref_bases << " of " << n_aligned_with_MD << " aligned reads with MD tag have one or more non-[ACGT] bases in the reference" << endl;
    if(bc_adjust)
      cout << "NOTE: " << program << ": " << *input_bam_filename_it << ": " << n_barcode_error << " of " << n_read << " reads have bc errors." << endl;
    if( (n_no_read_group > 0) || (n_unmatched_read_group > 0) )
      cerr << "WARNING: of " << program << ": " << *input_bam_filename_it << ": " << n_aligned << " aligned reads, " << n_no_read_group << " have no RG tag " << n_unmatched_read_group << " have an RG tag that does not match the header." << endl;
  }

  // Processing complete, generate ionstats_alignment.json
  writeIonstatsAlignmentJson(
    output_json_filename, aq_error_rate, bc_adjust, evaluate_flow,
    called_histogram,    aligned_histogram,    aq_histogram,
    called_histogram_bc, aligned_histogram_bc, aq_histogram_bc,
    total_insert_histo, total_Q17_histo, total_Q20_histo,
    system_snr, qv_histogram,
    base_position_error_count,
    base_position,
    flow_position,
    read_group_base_position,
    read_group_flow_position
  );

  writeIonstatsH5(
    output_h5_filename,
    base_position, flow_position, per_hp,
    read_group_base_position, read_group_flow_position, read_group_per_hp,
    region_name,
    regional_summary
  );


  return(EXIT_SUCCESS);
}

int writeIonstatsAlignmentJson(
  string &json_filename,
  vector<double> &aq_error_rate,
  bool bc_adjust,
  bool evaluate_flow,
  ReadLengthHistogram &called_histogram,    ReadLengthHistogram &aligned_histogram,    vector< ReadLengthHistogram > &aq_histogram,
  ReadLengthHistogram &called_histogram_bc, ReadLengthHistogram &aligned_histogram_bc, vector< ReadLengthHistogram > &aq_histogram_bc,
  ReadLengthHistogram &total_insert_histo, ReadLengthHistogram &total_Q17_histo, ReadLengthHistogram &total_Q20_histo,
  MetricGeneratorSNR &system_snr, BaseQVHistogram &qv_histogram,
  SimpleHistogram &base_position_error_count,
  ErrorData &base_position,
  ErrorData &flow_position,
  map< string, ErrorData > &read_group_base_position,
  map< string, ErrorData > &read_group_flow_position
) {

  Json::Value output_json(Json::objectValue);

  output_json["meta"]["creation_date"] = get_time_iso_string(time(NULL));
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
  for(unsigned int i=0; i<aq_error_rate.size(); ++i) {
    int phred_int = toPhred(aq_error_rate[i]);
    string phred_string = static_cast<ostringstream*>( &(ostringstream() << phred_int) )->str();
    aq_histogram[i].SaveToJson(output_json["AQ" + phred_string]);
  }

  // Called & aligned lengths including barcodes
  if(bc_adjust) {
    // We put any barcoded results in their own "WithBarcode" section
    called_histogram_bc.SaveToJson(output_json["WithBarcode"]["full"]);
    aligned_histogram_bc.SaveToJson(output_json["WithBarcode"]["aligned"]);
    for(unsigned int i=0; i<aq_error_rate.size(); ++i) {
      int phred_int = toPhred(aq_error_rate[i]);
      string phred_string = static_cast<ostringstream*>( &(ostringstream() << phred_int) )->str();
      aq_histogram_bc[i].SaveToJson(output_json["WithBarcode"]["AQ" + phred_string]);
    }
  }

  // Per-base error data
  base_position_error_count.SaveToJson(output_json["error_by_position"]);
  base_position.SaveToJson(output_json["by_base"]);

  // Per-flow error data
  if(evaluate_flow)
    flow_position.SaveToJson(output_json["by_flow"]);

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
  output_json["meta"]["creation_date"] = get_time_iso_string(time(NULL));
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



//
// Update flow_idx to bring it to the current incorporating flow
//
void flowCatchup(unsigned int &flow_idx, const string &flow_order, unsigned int read_idx, const string &read_bases) {
  if(read_idx >= read_bases.size()) {
    return;
  } else {
    char current_base = read_bases[read_idx];
    while(current_base != flow_order[flow_idx % flow_order.size()])
      flow_idx++;
  }
}

//
// Modifies flow_idx by stepping along n_advance bases in the read
//
void flowAdvance(
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
      if((read_idx < read_bases.size()) && (read_bases[read_idx] != current_base))
        flow_idx++;
    } else {
      // no incorporation, proceed to the next flow
      flow_idx++;
    }
  }
}

int parseAlignment(
  BamAlignment &          alignment,
  ReadAlignmentErrors &   base_space_errors,
  ReadAlignmentErrors &   flow_space_errors,
  map<string, string> &   flow_orders,
  string &                read_group,
  const map<char,char> &  reverse_complement_map,
  bool                    evaluate_flow,
  bool                    evaluate_hp,
  bool &                  invalid_read_bases,
  bool &                  invalid_ref_bases,
  bool &                  invalid_cigar,
  vector<char> &          ref_hp_nuc,
  vector<uint16_t> &      ref_hp_len,
  vector<int16_t> &       ref_hp_err,
  vector<uint16_t> &      ref_hp_flow
) {

  if(hasInvalidCigar(alignment))
    invalid_cigar=true;

  // Get read bases
  string read_bases = alignment.QueryBases;
  if(alignment.IsReverseStrand())
    assert(!reverse_complement(read_bases,reverse_complement_map));
  if(hasInvalidBases(read_bases))
    invalid_read_bases=true;

  if(invalid_read_bases || invalid_cigar)
    return(1);

  // Parse MD tag to extract MD_op and MD_len
  string MD_tag;
  assert(alignment.HasTag("MD"));
  alignment.GetTag("MD",MD_tag);
  vector<char>   MD_op;
  vector<int>    MD_len;
  vector<string> MD_seq;
  MD_op.reserve(1024);
  MD_len.reserve(1024);
  MD_seq.reserve(1024);
  parseMD(MD_tag,MD_op,MD_len,MD_seq);

  // Initialize data related to per-flow summary
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

  // Initialize data required for per-HP summary
  string ref_bases = "";
  vector<char> read_hp_nuc;
  vector<uint16_t> read_hp_len;
  vector<uint16_t> ref_hp_cum_len,read_hp_cum_len;
  if(evaluate_hp) {
    ref_hp_err.clear();
    ref_hp_flow.clear();
    ref_bases = getReferenceBases(read_bases, alignment.CigarData, MD_op, MD_len, MD_seq, alignment.IsReverseStrand(), reverse_complement_map);
    if(hasInvalidBases(ref_bases)) {
      invalid_ref_bases=true;
    } else {
      getHpBreakdown(read_bases, read_hp_nuc, read_hp_len);
      getHpBreakdown(ref_bases,  ref_hp_nuc,  ref_hp_len );
      if(read_hp_len.size() > 0) {
        read_hp_cum_len.push_back(read_hp_len[0]);
        for(unsigned int i=1; i<read_hp_nuc.size(); ++i) {
          read_hp_cum_len.push_back(read_hp_len[i] + read_hp_cum_len.back());
        }
      }
      if(!invalid_ref_bases && ref_hp_len.size() > 0) {
        ref_hp_cum_len.push_back(ref_hp_len[0]);
        for(unsigned int i=1; i<ref_hp_nuc.size(); ++i) {
          ref_hp_cum_len.push_back(ref_hp_len[i] + ref_hp_cum_len.back());
        }
      }
    }
  }

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
  if(evaluate_flow)
    flow_space_errors.SetHaveData();

  while (cigar_idx < (int) alignment.CigarData.size() and cigar_idx >= 0) {

    // Advance cigar if requried
    if (alignment.CigarData[cigar_idx].Length == 0) {
      cigar_idx += increment;
      continue;
    }

    // handle soft-clipping in cigar
    if (alignment.CigarData[cigar_idx].Type == 'S') {
      unsigned int nclip = alignment.CigarData[cigar_idx].Length;
      cigar_idx += increment;
      read_idx  += nclip;
      // we don't advance flow_idx through soft clipping as we rely instead on the ZF tag to tell us the first aligned flow
      continue;
    }

    // Advance MD if requried
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
      if(!invalid_ref_bases && evaluate_hp)
        hpAdvance(ALIGN_MATCH,advance,evaluate_flow,flow_idx,flow_order,read_bases,read_idx, read_hp_nuc, read_hp_len, read_hp_cum_len, ref_idx, ref_hp_nuc, ref_hp_len, ref_hp_cum_len, stored_read_match_count, read_hp_idx, ref_hp_idx, ref_hp_err, ref_hp_flow);
      if(evaluate_flow)
        flowAdvance(advance,flow_idx,flow_order,read_idx,read_bases);
      read_idx  += advance;
      ref_idx   += advance;
      alignment.CigarData[cigar_idx].Length -= advance;
      MD_len[MD_idx] -= advance;
    } else if (alignment.CigarData[cigar_idx].Type == 'I') {
      // Insertion (read has a base, reference doesn't)
      int advance = alignment.CigarData[cigar_idx].Length;
      if(!invalid_ref_bases && evaluate_hp)
        hpAdvance(ALIGN_INS,advance,evaluate_flow,flow_idx,flow_order,read_bases,read_idx,read_hp_nuc,read_hp_len,read_hp_cum_len,ref_idx,ref_hp_nuc,ref_hp_len,ref_hp_cum_len,stored_read_match_count,read_hp_idx,ref_hp_idx,ref_hp_err,ref_hp_flow);
      for (int cnt = 0; cnt < advance; ++cnt) {
        if(evaluate_flow) {
          flowCatchup(flow_idx,flow_order,read_idx,read_bases);
          flow_space_errors.AddIns(flow_idx);
          flowAdvance(1,flow_idx,flow_order,read_idx,read_bases);
        }
        base_space_errors.AddIns(read_idx);
        read_idx++;
      }
      alignment.CigarData[cigar_idx].Length -= advance;
    } else if (alignment.CigarData[cigar_idx].Type == 'D' and MD_idx < (int) MD_op.size() and MD_idx >= 0 and MD_op[MD_idx] == 'D') {
      // Deletion (reference has a base, read doesn't)
      int advance = min((int)alignment.CigarData[cigar_idx].Length, MD_len[MD_idx]);
      if(!invalid_ref_bases && evaluate_hp)
        hpAdvance(ALIGN_DEL,advance,evaluate_flow,flow_idx,flow_order,read_bases,read_idx,read_hp_nuc,read_hp_len,read_hp_cum_len,ref_idx,ref_hp_nuc,ref_hp_len,ref_hp_cum_len,stored_read_match_count,read_hp_idx,ref_hp_idx,ref_hp_err,ref_hp_flow);
      if(evaluate_flow) {
        // For now we are lazy and just assign a deletion to the current incorporating flow
        // A better version would analyze the deleted bases to see which flow(s) they should go to
        flow_space_errors.AddDel(flow_idx,advance);
      }
      base_space_errors.AddDel(read_idx,advance);
      ref_idx += advance;
      alignment.CigarData[cigar_idx].Length -= advance;
      MD_len[MD_idx] -= advance;
    } else if (MD_idx < (int) MD_op.size() and MD_idx >= 0 and MD_op[MD_idx] == 'X') {
      // Substitution
      int advance = min((int)alignment.CigarData[cigar_idx].Length, MD_len[MD_idx]);
      if(!invalid_ref_bases && evaluate_hp)
        hpAdvance(ALIGN_SUB,advance,evaluate_flow,flow_idx,flow_order,read_bases,read_idx,read_hp_nuc,read_hp_len,read_hp_cum_len,ref_idx,ref_hp_nuc,ref_hp_len,ref_hp_cum_len,stored_read_match_count,read_hp_idx,ref_hp_idx,ref_hp_err,ref_hp_flow);
      for (int cnt = 0; cnt < advance; ++cnt) {
        if(evaluate_flow) {
          flowCatchup(flow_idx,flow_order,read_idx,read_bases);
          flow_space_errors.AddSub(flow_idx);
          flowAdvance(1,flow_idx,flow_order,read_idx,read_bases);
        }
        base_space_errors.AddSub(read_idx);
        read_idx++;
        ref_idx++;
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

  return(0);
}

void getReadGroupInfo(const BamReader &input_bam, map< string, string > &flow_orders, unsigned int &max_flow_order_len, map< string, string > &key_bases, map< string, int > &key_len, string &seq_key, string &skip_rg_suffix) {
  flow_orders.clear();
  key_bases.clear();
  key_len.clear();
  max_flow_order_len=0;
  int seq_key_len = seq_key.length();
  SamHeader samHeader = input_bam.GetHeader();
  if(samHeader.HasReadGroups()) {
    SamReadGroupDictionary read_groups = samHeader.ReadGroups;
    for( SamReadGroupIterator it = read_groups.Begin(); it != read_groups.End(); ++it) {
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

void scoreBarcodes(ReadAlignmentErrors &base_space_errors, int bc_bases, int bc_errors, vector<double> &aq_error_rate, int minimum_aq_length, vector<int> &aq_length, ReadLengthHistogram &aligned_histogram_bc, vector< ReadLengthHistogram > &aq_histogram_bc) {

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
  aligned_histogram_bc.Add(base_space_errors.AlignedLen());
  for(unsigned int i=0; i < aq_error_rate.size(); ++i) {
    if(aq_length[i] >= minimum_aq_length)
      aq_histogram_bc[i].Add(aq_length[i]);
  }
}

string getReferenceBases(const string &read_bases, vector< CigarOp > CigarData, const vector<char> &MD_op, vector<int> MD_len, vector<string> &MD_seq, const bool rev_strand, const map<char,char> &reverse_complement_map) {
  string ref_bases = "";

  int increment  = rev_strand ? -1 : 1;
  int MD_idx     = rev_strand ? MD_op.size()-1 : 0;
  int cigar_idx  = rev_strand ? CigarData.size()-1 : 0;
  int read_idx   = 0;
  while (cigar_idx < (int) CigarData.size() and MD_idx < (int) MD_op.size() and cigar_idx >= 0 and MD_idx >= 0) {
    // Advance cigar if requried
    if (CigarData[cigar_idx].Length == 0) {
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
      if(rev_strand)
        reverse_complement(MD_seq[MD_idx],reverse_complement_map);
      ref_bases += MD_seq[MD_idx];
      CigarData[cigar_idx].Length -= advance;
      MD_len[MD_idx] -= advance;
    } else if (MD_op[MD_idx] == 'X') {
      int advance = min((int)CigarData[cigar_idx].Length, MD_len[MD_idx]);
assert(advance == (int) (MD_seq[MD_idx]).length());
      if(rev_strand)
        reverse_complement(MD_seq[MD_idx],reverse_complement_map);
      ref_bases += MD_seq[MD_idx];
      read_idx += advance;
      CigarData[cigar_idx].Length -= advance;
      MD_len[MD_idx] -= advance;
    } else {
      printf("ionstats alignment: getReferenceBases(): Unexpected OP combination: Cigar=%c, MD=%c !\n", CigarData[cigar_idx].Type, MD_op[MD_idx]);
      return("");
    }
  }
  return(ref_bases);
}

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

void getHpBreakdown(const string &bases, vector<char> &hp_nuc, vector<uint16_t> &hp_len) {
  hp_nuc.clear();
  hp_len.clear();
  unsigned int len = bases.length();
  if(len==0)
    return;
  hp_nuc.reserve(len);
  hp_len.reserve(len);
  char this_nuc = bases[0];
  uint16_t this_len = 1;
  for(unsigned int i=1; i<len; ++i) {
    if(bases[i] != this_nuc) {
      hp_nuc.push_back(this_nuc);
      hp_len.push_back(this_len);
      this_nuc = bases[i];
      this_len = 1;
    } else {
      this_len++;
    }
  }
  hp_nuc.push_back(this_nuc);
  hp_len.push_back(this_len);
  return;
}

void hpAdvance(
  // Inputs
  align_t                   alignment_type,
  int                       advance,         // number of bases in perfectly-aligned stretch
  // Data for tracking flow
  bool                      evaluate_flow,
  unsigned int              flow_idx,
  const string &            flow_order,
  const string &            read_bases,
  // Data for tracking position in bases/HPs
  int                       read_idx,         // index of where we are in read_bases
  const vector<char> &      read_hp_nuc,      // read hp nucs
  const vector<uint16_t> &  read_hp_len,      // read hp lengths
  const vector<uint16_t> &  read_hp_cum_len,  // read hp cumulative lengths
  int                       ref_idx,          // index of where we are in ref_bases
  const vector<char> &      ref_hp_nuc,       // read hp nucs
  const vector<uint16_t> &  ref_hp_len,       // read hp lengths
  const vector<uint16_t> &  ref_hp_cum_len,   // read hp cumulative lengths
  // Objects that may be modified
  int &                     stored_read_match_count, // number of read bases matching current ref hp that have been seen so far
  unsigned int &            read_hp_idx,             // index of where we are in read_hp_nuc, read_hp_len, read_hp_cum_len
  unsigned int &            ref_hp_idx,              // index of where we are in ref_hp_nuc, ref_hp_len, ref_hp_cum_len
  vector<int16_t> &         ref_hp_err,              //
  vector<uint16_t> &        ref_hp_flow
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
  int last_ref_idx = 0;
  int last_read_idx = 0;
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
        if(evaluate_flow)
          ref_hp_flow.push_back(flow_idx);
        // Addvance
        ref_idx += ref_match_count;
        ref_hp_idx++;
        if(evaluate_flow)
          flowAdvance(read_match_count,flow_idx,flow_order,read_idx,read_bases);
        read_idx += read_match_count;
        read_hp_idx++;
        stored_read_match_count = 0;
      }
    }
  } else if ( (alignment_type == ALIGN_INS) && (ref_hp_idx < ref_hp_len.size()) ) {
    // Read has an insertion, and we have not reached the end of the reference sequence yet
    // Conventions on recording insertions:
    // We only track bases that match either the former or subsequent reference HP - some insertions will be ignored.
    // If the first inserted HP matches the previous reference HP, it will already have been scored before reaching this point.
    // In the case of the insertion containing multiple read HPs, only the first and last will be scored.
    last_read_idx = read_idx + advance;
    if( (read_hp_idx > 0) && (read_idx < read_hp_cum_len[read_hp_idx-1]) )
      read_idx = read_hp_cum_len[read_hp_idx-1]; // Advance if inserted bases are an extension of a read HP that was already scored
    while( read_idx < last_read_idx) {
      int read_count = read_hp_cum_len[read_hp_idx] - read_idx;
      int insertion_len = (read_idx + read_count <= last_read_idx) ? read_count : (last_read_idx - read_idx);
      if(read_hp_nuc[read_hp_idx]==ref_hp_nuc[ref_hp_idx])
        stored_read_match_count = insertion_len;
      else 
        stored_read_match_count = 0;
      read_idx += insertion_len;
      if(read_idx == read_hp_cum_len[read_hp_idx])
        read_hp_idx++;
    }
  } else if (alignment_type == ALIGN_DEL) {
    last_ref_idx = ref_idx + advance;
    while( ref_idx < last_ref_idx) {
      int ref_count = ref_hp_cum_len[ref_hp_idx] - ref_idx;
      if(ref_idx + ref_count <= last_ref_idx) {
        // Finished a reference HP, store it
        ref_hp_err.push_back(stored_read_match_count - ref_hp_len[ref_hp_idx]);
assert(ref_hp_err.back() + ref_hp_len[ref_hp_idx] >= 0);
        if(evaluate_flow) {
          // Could do a better job here and figure out if there is some intermediate flow where the deletion would better fit
          ref_hp_flow.push_back(flow_idx);
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
    last_ref_idx = ref_idx + advance;
    while( ref_idx < last_ref_idx) {
      int ref_count = ref_hp_cum_len[ref_hp_idx] - ref_idx;
      bool completed_ref_hp=false;
      if(ref_idx + ref_count <= last_ref_idx) {
        // Finished a reference HP, store it
        completed_ref_hp=true;
        ref_hp_err.push_back(stored_read_match_count - ref_hp_len[ref_hp_idx]);
assert(ref_hp_err.back() + ref_hp_len[ref_hp_idx] >= 0);
        if(evaluate_flow)
          ref_hp_flow.push_back(flow_idx);
        // Advance to next hp
        ref_hp_idx++;
      } else {
        ref_count = last_ref_idx - ref_idx;
      }
      ref_idx += ref_count;
      if(evaluate_flow)
        flowAdvance(ref_count,flow_idx,flow_order,read_idx,read_bases);
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

void writeIonstatsH5(
  string h5_filename, 
  ErrorData & base_position,
  ErrorData & flow_position,
  HpData &    per_hp,
  map< string, ErrorData > & read_group_base_position,
  map< string, ErrorData > & read_group_flow_position,
  map< string, HpData > &    read_group_per_hp,
  vector<string> &region_name,
  vector< RegionalSummary > & regional_summary
) {
  // Pack all the ErrorData objects into a map
  map< string, ErrorData > error_data;
  // Per-base data
  error_data["per_base"] = base_position;
  for(map < string, ErrorData >::iterator it=read_group_base_position.begin(); it != read_group_base_position.end(); ++it)
    error_data["per_read_group/" + it->first + "/per_base"] = it->second;
  // Per-flow data
  if(flow_position.HasData())
    error_data["per_flow"] = flow_position;
  for(map < string, ErrorData >::iterator it=read_group_flow_position.begin(); it != read_group_flow_position.end(); ++it)
    if(it->second.HasData())
      error_data["per_read_group/" + it->first + "/per_flow"] = it->second;
  
  // Pack all the HpData objects into a map
  map< string, HpData > hp_data;
  if(per_hp.HasData())
    hp_data["per_hp"] = per_hp;
  for(map < string, HpData >::iterator it=read_group_per_hp.begin(); it != read_group_per_hp.end(); ++it)
    if(it->second.HasData())
      hp_data["per_read_group/" + it->first + "/per_hp"] = it->second;

  // Pack RegionalSummary objects into a map
  map< string, RegionalSummary > regional_data;
  for(unsigned int i=0; i<region_name.size(); ++i)
    regional_data["per_region/" + region_name[i]] = regional_summary[i];

  // Open h5 file and write all the error_data map elements
  hid_t file_id = H5Fcreate(h5_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
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
  herr_t status = H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT);
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
    status = H5Gclose(group_id);
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
  herr_t status = H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT);
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
    status = H5Gclose(group_id);
  }
  return 0;
}

herr_t MergeRegionalSummaryFromH5(hid_t loc_id, const char *name, const H5L_info_t *info, void *operator_data);
herr_t MergeRegionalSummaryFromH5(hid_t loc_id, const char *name, const H5L_info_t *info, void *operator_data) {
  H5O_info_t      infobuf;
  herr_t status = H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT);
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
    status = H5Gclose(group_id);
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

  herr_t status;
  unsigned int group_count=0;
  uint64_t total_bytes=0;
  for(unsigned int i=0; i < input_h5_filename.size(); ++i) {
    GetAggregatorSize(ed.error_data,hd.hp_data,rs.regional_summary,group_count,total_bytes);
    cout << "File " << i << "\t" << "aggregator has " << total_bytes << "\tbytes in\t" << group_count << " groups" << endl;
    hid_t input_file_id  = H5Fopen(input_h5_filename[i].c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t group_id = H5Gopen2(input_file_id, "/", H5P_DEFAULT);
    //RecursivelyMergeErrorDataFromH5(input_file_id,group_id,ed.error_data,merge_proton_blocks);
    status = H5Lvisit(group_id, H5_INDEX_NAME, H5_ITER_INC, MergeErrorDataFromH5, &ed);
    status = H5Lvisit(group_id, H5_INDEX_NAME, H5_ITER_INC, MergeHpDataFromH5, &hd);
    status = H5Lvisit(group_id, H5_INDEX_NAME, H5_ITER_INC, MergeRegionalSummaryFromH5, &rs);
    status = H5Gclose(group_id);
    status = H5Fclose(input_file_id);
  }
  GetAggregatorSize(ed.error_data,hd.hp_data,rs.regional_summary,group_count,total_bytes);
  cout << "File " << input_h5_filename.size() << "\t" << "aggregator has " << total_bytes << "\tbytes in\t" << group_count << " groups" << endl;

  // Open h5 file and write all the error_data map elements
  hid_t file_id = H5Fcreate(output_h5_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  for(map< string, ErrorData >::iterator it = ed.error_data.begin(); it != ed.error_data.end(); ++it)
    it->second.writeH5(file_id,it->first);
  for(map< string, HpData >::iterator it = hd.hp_data.begin(); it != hd.hp_data.end(); ++it)
    it->second.writeH5(file_id,it->first);
  for(map< string, RegionalSummary >::iterator it = rs.regional_summary.begin(); it != rs.regional_summary.end(); ++it)
    it->second.writeH5(file_id,it->first);
  status = H5Fclose (file_id);
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


bool hasInvalidCigar(BamAlignment & alignment) {
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

bool hasInvalidBases(string &b) {
  unsigned int n = b.length();
  bool invalid_bases=false;
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
      default:
        invalid_bases=true;
    }
  }
  return(invalid_bases);
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
