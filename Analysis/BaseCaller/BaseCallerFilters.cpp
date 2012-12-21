/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BaseCallerFilters.cpp
//! @ingroup  BaseCaller
//! @brief    BaseCallerFilters. Filtering and trimming algorithms, configuration, and accounting

#include "BaseCallerFilters.h"

#include <stdio.h>
#include <iomanip>

#include "LinuxCompat.h"
#include "Stats.h"
#include "IonErr.h"
#include "RawWells.h"
#include "Mask.h"

using namespace std;

enum FilteringOutcomes {
  kUninitialized,               //!< Read not examined by a filter
  kPassed,                      //!< Read not rejected by any filter
  kFilterZeroBases,             //!< Read filtered out, has zero length
  kFilterShortRead,             //!< Read filtered out, too short
  kFilterFailedKeypass,         //!< Read filtered out, key sequence not a perfect match
  kFilterHighPPF,               //!< Read filtered out, percent positive flows too large
  kFilterPolyclonal,            //!< Read filtered out, looks polyclonal
  kFilterHighResidual,          //!< Read filtered out, median residual too high
  kFilterBeverly,               //!< Read filtered out, Beverly filter disapproved
  kBkgmodelHighPPF,             //!< Read filtered out by Analysis, percent positive flows too high
  kBkgmodelPolyclonal,          //!< Read filtered out by Analysis, looks polyclonal
  kBkgmodelFailedKeypass,       //!< Read filtered out by Analysis, key sequence not a perfect match
  kFilteredShortAdapterTrim,    //!< Read filtered out, too short after adapter trimming
  kFilteredShortQualityTrim     //!< Read filtered out, too short after quality trimming
};

class ThousandsSeparator : public numpunct<char> {
protected:
    string do_grouping() const { return "\03"; }
};




//--------

int EncodeFilteringDetails(int n_base, int n_prefix)
{
  if (n_base < 0)
    return n_base;
  return max(0,n_base - n_prefix);
}


ReadFilteringHistory::ReadFilteringHistory()
{
  is_filtered = false;
  is_called = false;
  n_bases = -1;

  n_bases_key = 0;
  n_bases_prefix = 0;

  n_bases_after_bkgmodel_high_ppf = -1;
  n_bases_after_bkgmodel_polyclonal = -1;
  n_bases_after_bkgmodel_bad_key = -1;
  n_bases_after_polyclonal = -1;
  n_bases_after_high_ppf = -1;
  n_bases_after_too_short = -1;
  n_bases_after_bad_key = -1;
  n_bases_after_high_residual = -1;
  n_bases_after_beverly_trim = -1;
  n_bases_after_quality_trim = -1;
  n_bases_after_adapter_trim = -1;
  n_bases_filtered = -1;
}

void ReadFilteringHistory::GenerateZDVector(vector<int16_t>& zd_vector)
{
  zd_vector.resize(13);
  zd_vector[0] = EncodeFilteringDetails(n_bases,                           n_bases_prefix);
  zd_vector[1] = EncodeFilteringDetails(n_bases_after_bkgmodel_high_ppf,   n_bases_prefix);
  zd_vector[2] = EncodeFilteringDetails(n_bases_after_bkgmodel_polyclonal, n_bases_prefix);
  zd_vector[3] = EncodeFilteringDetails(n_bases_after_bkgmodel_bad_key,    n_bases_prefix);
  zd_vector[4] = EncodeFilteringDetails(n_bases_after_high_ppf,            n_bases_prefix);
  zd_vector[5] = EncodeFilteringDetails(n_bases_after_polyclonal,          n_bases_prefix);
  zd_vector[6] = EncodeFilteringDetails(n_bases_after_too_short,           n_bases_prefix);
  zd_vector[7] = EncodeFilteringDetails(n_bases_after_bad_key,             n_bases_prefix);
  zd_vector[8] = EncodeFilteringDetails(n_bases_after_high_residual,       n_bases_prefix);
  zd_vector[9] = EncodeFilteringDetails(n_bases_after_beverly_trim,        n_bases_prefix);
  zd_vector[10]= EncodeFilteringDetails(n_bases_after_adapter_trim,        n_bases_prefix);
  zd_vector[11]= EncodeFilteringDetails(n_bases_after_quality_trim,        n_bases_prefix);
  zd_vector[12]= EncodeFilteringDetails(n_bases_filtered,                  n_bases_prefix);
}


ReadFilteringStats::ReadFilteringStats()
{
  num_bases_initial_ = 0;
  num_bases_removed_key_trim_ = 0;
  num_bases_removed_barcode_trim_ = 0;
  num_bases_removed_short_ = 0;
  num_bases_removed_keypass_ = 0;
  num_bases_removed_residual_ = 0;
  num_bases_removed_beverly_ = 0;
  num_bases_removed_adapter_trim_ = 0;
  num_bases_removed_quality_trim_ = 0;
  num_bases_final_ = 0;

  num_reads_initial_ = 0;
  num_reads_removed_bkgmodel_keypass_ = 0;
  num_reads_removed_bkgmodel_high_ppf_ = 0;
  num_reads_removed_bkgmodel_polyclonal_ = 0;
  num_reads_removed_high_ppf_ = 0;
  num_reads_removed_polyclonal_ = 0;
  num_reads_removed_short_ = 0;
  num_reads_removed_keypass_ = 0;
  num_reads_removed_residual_ = 0;
  num_reads_removed_beverly_ = 0;
  num_reads_removed_adapter_trim_ = 0;
  num_reads_removed_quality_trim_ = 0;
  num_reads_final_ = 0;
}


void ReadFilteringStats::AddRead(const ReadFilteringHistory& read_filtering_history)
{

  // Step 1: Read accounting

  num_reads_initial_++;

  if (read_filtering_history.n_bases_after_bkgmodel_bad_key == 0)
    num_reads_removed_bkgmodel_keypass_++;

  if (read_filtering_history.n_bases_after_bkgmodel_high_ppf == 0)
    num_reads_removed_bkgmodel_high_ppf_++;

  if (read_filtering_history.n_bases_after_bkgmodel_polyclonal == 0)
    num_reads_removed_bkgmodel_polyclonal_++;

  if (read_filtering_history.n_bases_after_high_ppf == 0)
    num_reads_removed_high_ppf_++;

  if (read_filtering_history.n_bases_after_polyclonal == 0)
    num_reads_removed_polyclonal_++;

  if (read_filtering_history.n_bases_after_too_short == 0)
    num_reads_removed_short_++;

  if (read_filtering_history.n_bases_after_bad_key == 0)
    num_reads_removed_keypass_++;

  if (read_filtering_history.n_bases_after_high_residual == 0)
    num_reads_removed_residual_++;

  if (read_filtering_history.n_bases_after_beverly_trim == 0)
    num_reads_removed_beverly_++;

  if (read_filtering_history.n_bases_after_adapter_trim == 0)
    num_reads_removed_adapter_trim_++;

  if (read_filtering_history.n_bases_after_quality_trim == 0)
    num_reads_removed_quality_trim_++;

  if (read_filtering_history.n_bases_filtered > read_filtering_history.n_bases_prefix)
    num_reads_final_++;

  if (read_filtering_history.n_bases < 0) {
    // This read was filtered before treephaser, so no base accounting needed.
    return;
  }

  // Step 2: Base accounting

  num_bases_initial_                += read_filtering_history.n_bases;
  num_bases_removed_key_trim_       += read_filtering_history.n_bases_key;
  num_bases_removed_barcode_trim_   += min(read_filtering_history.n_bases_prefix,read_filtering_history.n_bases) - read_filtering_history.n_bases_key;

  int current_n_bases = max(0, read_filtering_history.n_bases - read_filtering_history.n_bases_prefix);
  if (read_filtering_history.n_bases_after_too_short >= 0) {
    int new_n_bases = max(0, read_filtering_history.n_bases_after_too_short - read_filtering_history.n_bases_prefix);
    num_bases_removed_short_ += current_n_bases - new_n_bases;
    current_n_bases = new_n_bases;
  }
  if (read_filtering_history.n_bases_after_bad_key >= 0) {
    int new_n_bases = max(0, read_filtering_history.n_bases_after_bad_key - read_filtering_history.n_bases_prefix);
    num_bases_removed_keypass_ += current_n_bases - new_n_bases;
    current_n_bases = new_n_bases;
  }
  if (read_filtering_history.n_bases_after_high_residual >= 0) {
    int new_n_bases = max(0, read_filtering_history.n_bases_after_high_residual - read_filtering_history.n_bases_prefix);
    num_bases_removed_residual_ += current_n_bases - new_n_bases;
    current_n_bases = new_n_bases;
  }
  if (read_filtering_history.n_bases_after_beverly_trim >= 0) {
    int new_n_bases = max(0, read_filtering_history.n_bases_after_beverly_trim - read_filtering_history.n_bases_prefix);
    num_bases_removed_beverly_ += current_n_bases - new_n_bases;
    current_n_bases = new_n_bases;
  }
  if (read_filtering_history.n_bases_after_adapter_trim >= 0) {
    int new_n_bases = max(0, read_filtering_history.n_bases_after_adapter_trim - read_filtering_history.n_bases_prefix);
    num_bases_removed_adapter_trim_ += current_n_bases - new_n_bases;
    current_n_bases = new_n_bases;
  }
  if (read_filtering_history.n_bases_after_quality_trim >= 0) {
    int new_n_bases = max(0, read_filtering_history.n_bases_after_quality_trim - read_filtering_history.n_bases_prefix);
    num_bases_removed_quality_trim_ += current_n_bases - new_n_bases;
    //current_n_bases = new_n_bases;
  }

  num_bases_final_ += max(0, read_filtering_history.n_bases_filtered - read_filtering_history.n_bases_prefix);

}


void ReadFilteringStats::MergeFrom(const ReadFilteringStats& other)
{
  num_bases_initial_                      += other.num_bases_initial_;
  num_bases_removed_key_trim_             += other.num_bases_removed_key_trim_;
  num_bases_removed_barcode_trim_         += other.num_bases_removed_barcode_trim_;
  num_bases_removed_short_                += other.num_bases_removed_short_;
  num_bases_removed_keypass_              += other.num_bases_removed_keypass_;
  num_bases_removed_residual_             += other.num_bases_removed_residual_;
  num_bases_removed_beverly_              += other.num_bases_removed_beverly_;
  num_bases_removed_adapter_trim_         += other.num_bases_removed_adapter_trim_;
  num_bases_removed_quality_trim_         += other.num_bases_removed_quality_trim_;
  num_bases_final_                        += other.num_bases_final_;

  num_reads_initial_                      += other.num_reads_initial_;
  num_reads_removed_bkgmodel_keypass_     += other.num_reads_removed_bkgmodel_keypass_;
  num_reads_removed_bkgmodel_high_ppf_    += other.num_reads_removed_bkgmodel_high_ppf_;
  num_reads_removed_bkgmodel_polyclonal_  += other.num_reads_removed_bkgmodel_polyclonal_;
  num_reads_removed_high_ppf_             += other.num_reads_removed_high_ppf_;
  num_reads_removed_polyclonal_           += other.num_reads_removed_polyclonal_;
  num_reads_removed_short_                += other.num_reads_removed_short_;
  num_reads_removed_keypass_              += other.num_reads_removed_keypass_;
  num_reads_removed_residual_             += other.num_reads_removed_residual_;
  num_reads_removed_beverly_              += other.num_reads_removed_beverly_;
  num_reads_removed_adapter_trim_         += other.num_reads_removed_adapter_trim_;
  num_reads_removed_quality_trim_         += other.num_reads_removed_quality_trim_;
  num_reads_final_                        += other.num_reads_final_;
}

void ReadFilteringStats::PrettyPrint (const string& table_header)
{

  ostringstream table;
  table.imbue(locale(table.getloc(), new ThousandsSeparator));

  table << endl;
  table << setw(25) << table_header.c_str() << setw(23) << "Read balance" << setw(23) << "Base balance" << endl;

  table << setw(25) << "Examined wells";
  table << setw(23) << num_reads_initial_ << setw(23) << num_bases_initial_ << endl;

  table << setw(25) << " ";
  table << setw(23) << "--------------------" << setw(23) << "--------------------" << endl;

  table << setw(25) << "BkgModel:   High PPF";
  table << setw(23) << -num_reads_removed_bkgmodel_high_ppf_ << setw(23) << "0" << endl;

  table << setw(25) << "BkgModel: Polyclonal";
  table << setw(23) << -num_reads_removed_bkgmodel_polyclonal_ << setw(23) << "0" << endl;

  table << setw(25) << "BkgModel:    Bad key";
  table << setw(23) << -num_reads_removed_bkgmodel_keypass_ << setw(23) << "0" << endl;

  table << setw(25) << "High PPF";
  table << setw(23) << -num_reads_removed_high_ppf_ << setw(23) << "0" << endl;

  table << setw(25) << "Polyclonal";
  table << setw(23) << -num_reads_removed_polyclonal_ << setw(23) << "0" << endl;

  table << setw(25) << "Key trim";
  table << setw(23) << "0" << setw(23) << -num_bases_removed_key_trim_ << endl;

  table << setw(25) << "Barcode trim";
  table << setw(23) << "0" << setw(23) << -num_bases_removed_barcode_trim_ << endl;

  table << setw(25) << "Short read";
  table << setw(23) << -num_reads_removed_short_ << setw(23) << -num_bases_removed_short_ << endl;

  table << setw(25) << "Bad key";
  table << setw(23) << -num_reads_removed_keypass_ << setw(23) << -num_bases_removed_keypass_ << endl;

  table << setw(25) << "High residual";
  table << setw(23) << -num_reads_removed_residual_ << setw(23) << -num_bases_removed_residual_ << endl;

  table << setw(25) << "Beverly filter";
  table << setw(23) << -num_reads_removed_beverly_ << setw(23) << -num_bases_removed_beverly_ << endl;

  table << setw(25) << "Adapter trim";
  table << setw(23) << -num_reads_removed_adapter_trim_ << setw(23) << -num_bases_removed_adapter_trim_ << endl;

  table << setw(25) << "Quality trim";
  table << setw(23) << -num_reads_removed_quality_trim_ << setw(23) << -num_bases_removed_quality_trim_ << endl;

  table << setw(25) << " ";
  table << setw(23) << "--------------------" << setw(23) << "--------------------" << endl;

  table << setw(25) << "Final saved to BAMs";
  table << setw(23) << num_reads_final_ << setw(23) << num_bases_final_ << endl;
  table << endl;

  cout << table.str();
}


void ReadFilteringStats::SaveToBasecallerJson(Json::Value &json, const string& class_name, bool library_report)
{
  // ReadDetails
  json["Filtering"]["ReadDetails"][class_name]["bkgmodel_polyclonal"] = (Json::Int64)num_reads_removed_bkgmodel_polyclonal_;
  json["Filtering"]["ReadDetails"][class_name]["bkgmodel_high_ppf"]   = (Json::Int64)num_reads_removed_bkgmodel_high_ppf_;
  json["Filtering"]["ReadDetails"][class_name]["bkgmodel_keypass"]    = (Json::Int64)num_reads_removed_bkgmodel_keypass_;
  json["Filtering"]["ReadDetails"][class_name]["polyclonal"]          = (Json::Int64)num_reads_removed_polyclonal_;
  json["Filtering"]["ReadDetails"][class_name]["high_ppf"]            = (Json::Int64)num_reads_removed_high_ppf_;
  json["Filtering"]["ReadDetails"][class_name]["short"]               = (Json::Int64)num_reads_removed_short_;
  json["Filtering"]["ReadDetails"][class_name]["failed_keypass"]      = (Json::Int64)num_reads_removed_keypass_;
  json["Filtering"]["ReadDetails"][class_name]["high_residual"]       = (Json::Int64)num_reads_removed_residual_;
  json["Filtering"]["ReadDetails"][class_name]["adapter_trim"]        = (Json::Int64)num_reads_removed_adapter_trim_;
  json["Filtering"]["ReadDetails"][class_name]["quality_trim"]        = (Json::Int64)num_reads_removed_quality_trim_;
  json["Filtering"]["ReadDetails"][class_name]["beverly_filter"]      = (Json::Int64)num_reads_removed_beverly_;
  json["Filtering"]["ReadDetails"][class_name]["valid"]               = (Json::Int64)num_reads_final_;

  // BeadSummary - obsolete me!
  json["BeadSummary"][class_name]["polyclonal"]  = (Json::Int64)(num_reads_removed_bkgmodel_polyclonal_ + num_reads_removed_polyclonal_);
  json["BeadSummary"][class_name]["highPPF"]     = (Json::Int64)(num_reads_removed_bkgmodel_high_ppf_ + num_reads_removed_high_ppf_);
  json["BeadSummary"][class_name]["zero"]        = 0;
  json["BeadSummary"][class_name]["short"]       = (Json::Int64)(num_reads_removed_short_ + num_reads_removed_adapter_trim_ + num_reads_removed_quality_trim_);
  json["BeadSummary"][class_name]["badKey"]      = (Json::Int64)(num_reads_removed_bkgmodel_keypass_ + num_reads_removed_keypass_);
  json["BeadSummary"][class_name]["highRes"]     = (Json::Int64)(num_reads_removed_residual_ + num_reads_removed_beverly_);
  json["BeadSummary"][class_name]["valid"]       = (Json::Int64)num_reads_final_;

  // Generate values that go to the library report
  if (library_report) {

    // BaseDetails
    json["Filtering"]["BaseDetails"]["initial"]             = (Json::Int64)num_bases_initial_;
    json["Filtering"]["BaseDetails"]["short"]               = (Json::Int64)num_bases_removed_short_;
    json["Filtering"]["BaseDetails"]["failed_keypass"]      = (Json::Int64)num_bases_removed_keypass_;
    json["Filtering"]["BaseDetails"]["high_residual"]       = (Json::Int64)num_bases_removed_residual_;
    json["Filtering"]["BaseDetails"]["beverly_filter"]      = (Json::Int64)num_bases_removed_beverly_;
    json["Filtering"]["BaseDetails"]["adapter_trim"]        = (Json::Int64)num_bases_removed_adapter_trim_;
    json["Filtering"]["BaseDetails"]["quality_trim"]        = (Json::Int64)num_bases_removed_quality_trim_;
    json["Filtering"]["BaseDetails"]["final"]               = (Json::Int64)num_bases_final_;

    json["Filtering"]["LibraryReport"]["filtered_polyclonal"]   = (Json::Int64)(num_reads_removed_bkgmodel_polyclonal_ + num_reads_removed_polyclonal_);
    json["Filtering"]["LibraryReport"]["filtered_primer_dimer"] = (Json::Int64) num_reads_removed_adapter_trim_;
    json["Filtering"]["LibraryReport"]["filtered_low_quality"] = (Json::Int64)(
                  num_reads_removed_bkgmodel_high_ppf_ +
                  num_reads_removed_bkgmodel_keypass_ +
                  num_reads_removed_high_ppf_ +
                  num_reads_removed_short_ +
                  num_reads_removed_keypass_ +
                  num_reads_removed_residual_ +
                  num_reads_removed_quality_trim_ +
                  num_reads_removed_beverly_);
    json["Filtering"]["LibraryReport"]["final_library_reads"]   = (Json::Int64)num_reads_final_;
  }
}








void BaseCallerFilters::PrintHelp()
{
  printf ("Filtering and trimming options:\n");
  printf ("  -d,--disable-all-filters   on/off     disable all filtering and trimming, overrides other args [off]\n");
  printf ("  -k,--keypass-filter        on/off     apply keypass filter [on]\n");
  printf ("     --clonal-filter-solve   on/off     apply polyclonal filter [off]\n");
  printf ("     --clonal-filter-tf      on/off     apply polyclonal filter to TFs [off]\n");
  printf ("     --min-read-length       INT        apply minimum read length filter [8]\n");
  printf ("     --cr-filter             on/off     apply cafie residual filter [off]\n");
  printf ("     --cr-filter-tf          on/off     apply cafie residual filter to TFs [off]\n");
  printf ("     --cr-filter-max-value   FLOAT      cafie residual filter threshold [0.8]\n");
  printf ("     --beverly-filter        filter_ratio,trim_ratio,min_length / off\n");
  printf ("                                        apply Beverly filter/trimmer [off]\n");
  printf ("     --trim-adapter          STRING     reverse complement of adapter sequence [ATCACCGACTGCCCATAGAGAGGCTGAGAC]\n");
  printf ("     --trim-adapter-cutoff   FLOAT      cutoff for adapter trimming, 0=off [16]\n");
  printf ("     --trim-adapter-min-match INT       minimum adapter bases in the read required for trimming  [6]\n");
  printf ("     --trim-adapter-mode     INT        0=use simplified metric, 1=use standard metric [1]\n");
  printf ("     --trim-qual-window-size INT        window size for quality trimming [30]\n");
  printf ("     --trim-qual-cutoff      FLOAT      cutoff for quality trimming, 100=off [16]\n");
  printf ("     --trim-min-read-len     INT        reads trimmed shorter than this are omitted from output [8]\n");
  printf ("\n");
}


BaseCallerFilters::BaseCallerFilters(OptArgs& opts,
    const ion::FlowOrder& flow_order, const vector<KeySequence>& keys, const Mask& mask)
{
  flow_order_ = flow_order;
  keys_ = keys;
  num_classes_ = keys_.size();
  assert(num_classes_ == 2);
  filter_mask_.assign(mask.H()*mask.W(), kUninitialized);

  // Retrieve command line options

  filter_keypass_enabled_         = opts.GetFirstBoolean('k', "keypass-filter", true);
  filter_min_read_length_         = opts.GetFirstInt    ('-', "min-read-length", 8);
  filter_clonal_enabled_tfs_      = opts.GetFirstBoolean('-', "clonal-filter-tf", false);
  filter_clonal_enabled_          = opts.GetFirstBoolean('-', "clonal-filter-solve", false);
  filter_residual_enabled_        = opts.GetFirstBoolean('-', "cr-filter", false);
  filter_residual_enabled_tfs_    = opts.GetFirstBoolean('-', "cr-filter-tf", false);

  //! \todo Get this to work right. May require "unwound" flow order, so incompatible with current wells.FlowOrder()
  //flt_control.cafieResMaxValueByFlowOrder[std::string ("TACG") ] = 0.06;  // regular flow order
  //flt_control.cafieResMaxValueByFlowOrder[std::string ("TACGTACGTCTGAGCATCGATCGATGTACAGC") ] = 0.08;  // xdb flow order

  filter_residual_max_value_      = opts.GetFirstDouble ('-',  "cr-filter-max-value", 0.08);

  // SFFTrim options
  trim_adapter_                   = opts.GetFirstString ('-', "trim-adapter", "ATCACCGACTGCCCATAGAGAGGCTGAGAC");
  trim_adapter_cutoff_            = opts.GetFirstDouble ('-', "trim-adapter-cutoff", 16.0);
  trim_adapter_min_match_         = opts.GetFirstInt    ('-', "trim-adapter-min-match", 6);
  trim_adapter_mode_              = opts.GetFirstInt    ('-', "trim-adapter-mode", 1);

  trim_qual_window_size_          = opts.GetFirstInt    ('-', "trim-qual-window-size", 30);
  trim_qual_cutoff_               = opts.GetFirstDouble ('-', "trim-qual-cutoff", 16.0);
  trim_min_read_len_              = opts.GetFirstInt    ('-', "trim-min-read-len", 8);


  //string filter_beverly_args      = opts.GetFirstString ('-', "beverly-filter", "0.03,0.03,8");
  string filter_beverly_args      = opts.GetFirstString ('-', "beverly-filter", "off");

  bool disable_all_filters        = opts.GetFirstBoolean('d', "disable-all-filters", false);


  if (disable_all_filters) {
    filter_keypass_enabled_ = false;
    filter_clonal_enabled_tfs_ = false;
    filter_clonal_enabled_ = false;
    filter_residual_enabled_ = false;
    filter_residual_enabled_tfs_ = false;
    trim_adapter_cutoff_ = 0.0; // Zero means disabled for now
    trim_qual_cutoff_ = 100.0; // 100.0 means disabled for now
    filter_beverly_args = "off";
  }


  printf("Filter settings:\n");
  printf("   --disable-all-filters %s\n", disable_all_filters ? "on (overrides other options)" : "off");
  printf("        --keypass-filter %s\n", filter_keypass_enabled_ ? "on" : "off");
  printf("       --min-read-length %d\n", filter_min_read_length_);
  printf("   --clonal-filter-solve %s\n", filter_clonal_enabled_ ? "on" : "off");
  printf("      --clonal-filter-tf %s\n", filter_clonal_enabled_tfs_ ? "on" : "off");
  printf("             --cr-filter %s\n", filter_residual_enabled_ ? "on" : "off");
  printf("          --cr-filter-tf %s\n", filter_residual_enabled_tfs_ ? "on" : "off");
  printf("   --cr-filter-max-value %1.3f\n", filter_residual_max_value_);
  printf("          --trim-adapter %s\n", trim_adapter_.c_str());
  printf("     --trim-adapter-mode %d\n", trim_adapter_mode_);
  printf("   --trim-adapter-cutoff %1.1f (0.0 means disabled)\n", trim_adapter_cutoff_);
  printf("--trim-adapter-min-match %d\n", trim_adapter_min_match_);
  printf(" --trim-qual-window-size %d\n", trim_qual_window_size_);
  printf("      --trim-qual-cutoff %1.1f (100.0 means disabled)\n", trim_qual_cutoff_);
  printf("     --trim-min-read-len %d\n", trim_min_read_len_);
  printf("        --beverly-filter %s\n", filter_beverly_args.c_str());
  printf("\n");


  // Validate options
  if (filter_min_read_length_ < 1) {
    fprintf (stderr, "Option Error: min-read-length must specify a positive value (%d invalid).\n", filter_min_read_length_);
    exit (EXIT_FAILURE);
  }
  if (filter_residual_max_value_ <= 0) {
    fprintf (stderr, "Option Error: cr-filter-max-value must specify a positive value (%lf invalid).\n", filter_residual_max_value_);
    exit (EXIT_FAILURE);
  }

  if (filter_beverly_args == "off") {
    filter_beverly_enabled_ = false; // Nothing, really

  } else {
    int stat = sscanf (filter_beverly_args.c_str(), "%f,%f,%d",
        &filter_beverly_filter_ratio_,
        &filter_beverly_trim_ratio_,
        &filter_beverly_min_read_length_);
    if (stat != 3) {
      fprintf (stderr, "Option Error: beverly-filter %s\n", filter_beverly_args.c_str());
      fprintf (stderr, "Usage: --beverly-filter=filter_ratio,trim_ratio,min_length\n");
      exit (EXIT_FAILURE);
    }
    filter_beverly_enabled_ = true;
  }
}


void BaseCallerFilters::TrainClonalFilter(const string& output_directory, RawWells& wells, int num_unfiltered, Mask& mask)
{
  if (!filter_clonal_enabled_ and !filter_clonal_enabled_tfs_)
    return;

  wells.OpenForIncrementalRead();
  vector<int> key_ionogram(keys_[0].flows(), keys_[0].flows()+keys_[0].flows_length());
  filter_counts counts;
  int nlib = mask.GetCount(static_cast<MaskType> (MaskLib));
  counts._nsamp = min(nlib, num_unfiltered); // In the future, a parameter separate from num_unfiltered
  make_filter(clonal_population_, counts, mask, wells, key_ionogram);
  cout << counts << endl;
  wells.Close();
}




void BaseCallerFilters::TransferFilteringResultsToMask(Mask &mask) const
{
  assert(mask.H()*mask.W() == (int)filter_mask_.size());

  for (size_t idx = 0; idx < filter_mask_.size(); idx++) {

    mask[idx] &= MaskAll - MaskFilteredBadPPF - MaskFilteredShort - MaskFilteredBadKey - MaskFilteredBadResidual - MaskKeypass;

    switch (filter_mask_[idx]) {
      case kPassed:                     mask[idx] |= MaskKeypass; break;
      case kFilterZeroBases:            mask[idx] |= MaskFilteredShort; break;
      case kFilterShortRead:            mask[idx] |= MaskFilteredShort; break;
      case kFilterFailedKeypass:        mask[idx] |= MaskFilteredBadKey; break;
      case kFilterHighPPF:              mask[idx] |= MaskFilteredBadPPF; break;
      case kFilterPolyclonal:           mask[idx] |= MaskFilteredBadPPF; break;
      case kFilterHighResidual:         mask[idx] |= MaskFilteredBadResidual; break;
      case kFilterBeverly:              mask[idx] |= MaskFilteredBadResidual; break;
      case kBkgmodelFailedKeypass:      mask[idx] |= MaskFilteredBadKey; break;
      case kBkgmodelHighPPF:            mask[idx] |= MaskFilteredBadPPF; break;
      case kBkgmodelPolyclonal:         mask[idx] |= MaskFilteredBadPPF; break;
      case kFilteredShortAdapterTrim:   mask[idx] |= MaskFilteredShort; break;
      case kFilteredShortQualityTrim:   mask[idx] |= MaskFilteredShort; break;
    }
  }
}


int BaseCallerFilters::NumWellsCalled() const
{
  int num_called = 0;

  for (size_t idx = 0; idx < filter_mask_.size(); idx++)
    if (filter_mask_[idx] != kUninitialized)
      num_called++;

  return num_called;
}



void BaseCallerFilters::SetValid(int read_index)
{
  filter_mask_[read_index] = kPassed;
}


bool BaseCallerFilters::IsValid(int read_index) const
{
  return filter_mask_[read_index] == kPassed;
}

bool BaseCallerFilters::IsPolyclonal(int read_index) const
{
  return filter_mask_[read_index] == kFilterPolyclonal;
}




void BaseCallerFilters::SetBkgmodelHighPPF(int read_index, ReadFilteringHistory& filter_history)
{
  if (filter_history.is_filtered)
    return;
  filter_mask_[read_index] = kBkgmodelHighPPF;
  filter_history.n_bases_filtered = 0;
  filter_history.n_bases_after_bkgmodel_high_ppf = 0;
  filter_history.is_filtered = true;
}

void BaseCallerFilters::SetBkgmodelPolyclonal(int read_index, ReadFilteringHistory& filter_history)
{
  if (filter_history.is_filtered)
    return;
  filter_mask_[read_index] = kBkgmodelPolyclonal;
  filter_history.n_bases_filtered = 0;
  filter_history.n_bases_after_bkgmodel_polyclonal = 0;
  filter_history.is_filtered = true;
}

void BaseCallerFilters::SetBkgmodelFailedKeypass(int read_index, ReadFilteringHistory& filter_history)
{
  if (filter_history.is_filtered)
    return;
  filter_mask_[read_index] = kBkgmodelFailedKeypass;
  filter_history.n_bases_filtered = 0;
  filter_history.n_bases_after_bkgmodel_bad_key = 0;
  filter_history.is_filtered = true;
}


void BaseCallerFilters::FilterHighPPFAndPolyclonal (int read_index, int read_class, ReadFilteringHistory& filter_history,
    const vector<float>& measurements)
{
  if (filter_history.is_filtered)
    return;

  if (read_class == 0 and !filter_clonal_enabled_)  // Filter disabled for library?
    return;
  if (read_class != 0 and !filter_clonal_enabled_tfs_)  // Filter disabled for TFs?
    return;

  vector<float>::const_iterator first = measurements.begin() + mixed_first_flow();
  vector<float>::const_iterator last  = measurements.begin() + mixed_last_flow();
  float ppf = percent_positive(first, last);
  float ssq = sum_fractional_part(first, last);

  if(ppf > mixed_ppf_cutoff()) {
    filter_mask_[read_index] = kFilterHighPPF;
    filter_history.n_bases_filtered = 0;
    filter_history.n_bases_after_high_ppf = 0;
    filter_history.is_filtered = true;
  }
  else if(!clonal_population_.is_clonal(ppf, ssq)) {
    filter_mask_[read_index] = kFilterPolyclonal;
    filter_history.n_bases_filtered = 0;
    filter_history.n_bases_after_polyclonal = 0;
    filter_history.is_filtered = true;
  }
}


void BaseCallerFilters::FilterZeroBases(int read_index, int read_class, ReadFilteringHistory& filter_history)
{
  if (filter_history.is_filtered)
    return;

  if(filter_history.n_bases == 0) {
    filter_mask_[read_index] = kFilterZeroBases;
    filter_history.n_bases_filtered = 0;
    filter_history.n_bases_after_too_short = 0;
    filter_history.is_filtered = true;
  }
}


void BaseCallerFilters::FilterShortRead(int read_index, int read_class, ReadFilteringHistory& filter_history)
{
  if (filter_history.is_filtered)
    return;

  int actual_bases = filter_history.n_bases_filtered - filter_history.n_bases_prefix;
  if(filter_history.n_bases < filter_min_read_length_ or actual_bases < trim_min_read_len_) {
    filter_mask_[read_index] = kFilterShortRead;
    filter_history.n_bases_filtered = 0;
    filter_history.n_bases_after_too_short = 0;
    filter_history.is_filtered = true;
  }
}



void BaseCallerFilters::FilterFailedKeypass(int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<char>& sequence)
{
  if (filter_history.is_filtered)
    return;

  if(!filter_keypass_enabled_)  // Filter disabled?
    return;

  bool failed_keypass = false;
  for (int base = 0; base < keys_[read_class].bases_length(); ++base)
    if (sequence[base] != keys_[read_class].bases()[base])
      failed_keypass = true;

  if (failed_keypass) {
    filter_mask_[read_index] = kFilterFailedKeypass;
    filter_history.n_bases_filtered = 0;
    filter_history.n_bases_after_bad_key = 0;
    filter_history.is_filtered = true;
  }
}



void BaseCallerFilters::FilterHighResidual(int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<float>& residual)
{
  if (filter_history.is_filtered)
    return;

  if (read_class == 0 and !filter_residual_enabled_)  // Filter disabled for library?
    return;
  if (read_class != 0 and !filter_residual_enabled_tfs_)  // Filter disabled for TFs?
    return;

  if(MedianAbsoluteCafieResidual(residual, 60) > filter_residual_max_value_) {
    filter_mask_[read_index] = kFilterHighResidual;
    filter_history.n_bases_filtered = 0;
    filter_history.n_bases_after_high_residual = 0;
    filter_history.is_filtered = true;
  }
}

void BaseCallerFilters::FilterBeverly(int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<float>& scaled_residual,
    const vector<int>& base_to_flow)
{
  if (filter_history.is_filtered)
    return;

  bool reject = false;
  uint16_t clip_qual_right = filter_history.n_bases;

  if (filter_beverly_enabled_ and read_class == 0) {    // What about random reads? What about TFs?

    int num_onemers = 0;
    int num_twomers = 0;
    int num_extreme_onemers = 0;
    int num_extreme_twomers = 0;
    int max_trim_bases = 0;

    for (int flow = 0, base = 0; flow < flow_order_.num_flows(); ++flow) {

      int hp_length = 0;
      while (base < filter_history.n_bases and base_to_flow[base] == flow) {
        base++;
        hp_length++;
      }

      if (hp_length == 1) {
        num_onemers++;
        if (scaled_residual[flow] <= -0.405f or scaled_residual[flow] >= 0.395f)
          num_extreme_onemers++;
      }

      if (hp_length == 2) {
        num_twomers++;
        if (scaled_residual[flow] <= -0.405f or scaled_residual[flow] >= 0.395f)
          num_extreme_twomers++;
      }

      if (num_extreme_onemers <= num_onemers * filter_beverly_trim_ratio_)
        max_trim_bases = base;
    }

    if ((num_extreme_onemers + num_extreme_twomers) > (num_onemers + num_twomers) * filter_beverly_filter_ratio_) {

      int trim_length = max_trim_bases - filter_history.n_bases_prefix;
      if (trim_length < trim_min_read_len_) { // Quality trimming led to filtering
        reject = true;
        clip_qual_right = 0;
      } else
        clip_qual_right = max_trim_bases;
    }
  }

  if (clip_qual_right >= filter_history.n_bases_filtered)
    return;

  if(reject) {
    filter_mask_[read_index] = kFilterBeverly;
    filter_history.n_bases_after_beverly_trim = filter_history.n_bases_filtered = 0;
    filter_history.is_filtered = true;

  } else {
    filter_history.n_bases_after_beverly_trim = filter_history.n_bases_filtered  = clip_qual_right;
  }
}


double BaseCallerFilters::MedianAbsoluteCafieResidual(const vector<float> &residual, unsigned int use_flows)
{
  use_flows = min(use_flows, (unsigned int)residual.size());
  if (use_flows == 0)
    return 0;

  vector<double> abs_residual(use_flows);
  for (unsigned int flow = 0; flow < use_flows; flow++)
    abs_residual[flow] = abs(residual[flow]);

  return ionStats::median(abs_residual);
}



void BaseCallerFilters::TrimAdapter(int read_index, int read_class, ProcessedRead& processed_read, const vector<float>& scaled_residual,
    const vector<int>& base_to_flow, DPTreephaser& treephaser, const BasecallerRead& read)
{
  if(trim_adapter_cutoff_ <= 0.0 or trim_adapter_.empty())  // Zero means disabled
    return;

  if (read_class != 0)  // Hardcoded: Don't trim TFs
    return;

  int best_start_flow = -1;
  int best_start_base = -1;
  int best_adapter_overlap = -1;

  if (trim_adapter_mode_ == 2) {

    //
    // New predicted-signal-based
    //

    float best_metric = 0.1; // The lower the better

    DPTreephaser::TreephaserPath& called_path = treephaser.path(0);   //simulates the main sequence
    DPTreephaser::TreephaserPath& adapter_path = treephaser.path(1);  //branches off to simulate adapter

    treephaser.InitializeState(&called_path);

    for (int adapter_start_base = 0; adapter_start_base < (int)read.sequence.size(); ++adapter_start_base) {

      // Step 1. Consider current position as hypothetical adapter start

      adapter_path.prediction = called_path.prediction;
      int window_start = max(0,called_path.window_start - 8);
      treephaser.AdvanceState(&adapter_path,&called_path, trim_adapter_[0], flow_order_.num_flows());

      int inphase_flow = called_path.flow;
      float state_inphase = called_path.state[inphase_flow];

      int adapter_bases = 0;
      for (int adapter_pos = 1; adapter_pos < (int)trim_adapter_.length(); ++adapter_pos) {
        treephaser.AdvanceStateInPlace(&adapter_path, trim_adapter_[adapter_pos], flow_order_.num_flows());
        if (adapter_path.flow < flow_order_.num_flows())
          adapter_bases++;
      }

      float xy = 0, xy2 = 0, yy = 0;
      for (int metric_flow = window_start; metric_flow < adapter_path.flow; ++metric_flow) {
        xy  += adapter_path.prediction[metric_flow] * read.normalized_measurements[metric_flow];
        xy2 += read.prediction[metric_flow] * read.normalized_measurements[metric_flow];
        yy  += read.normalized_measurements[metric_flow] * read.normalized_measurements[metric_flow];
      }
      if (yy > 0) {
        xy  /= yy;
        xy2 /= yy;
      }

      float metric_num = 0;
      float metric_den = 0;
      for (int metric_flow = window_start; metric_flow < adapter_path.flow; ++metric_flow) {
        float delta_adapter  = read.normalized_measurements[metric_flow]*xy - adapter_path.prediction[metric_flow];
        float delta_sequence = read.normalized_measurements[metric_flow]*xy2 - read.prediction[metric_flow];
        metric_num += delta_adapter*delta_adapter - delta_sequence*delta_sequence;
        metric_den += state_inphase;
      }

      float adapter_score = metric_num/metric_den + 0.2/adapter_bases;

      if (adapter_score < best_metric) {
        best_metric = adapter_score;
        best_start_flow = inphase_flow;
        best_start_base = adapter_start_base;
      }
      // Step 2. Continue to next position

      treephaser.AdvanceStateInPlace(&called_path, read.sequence[adapter_start_base], flow_order_.num_flows());
    }




  } else {

    //
    // Classic adapter trimming strategies
    //


    float best_metric = -1e10;
    int sequence_pos = 0;

    for (int adapter_start_flow = 0; adapter_start_flow < flow_order_.num_flows(); ++adapter_start_flow) {

      while (sequence_pos < (int)read.sequence.size() and base_to_flow[sequence_pos] < adapter_start_flow)
        sequence_pos++;

      // Only consider start flows that agree with adapter start
      if (trim_adapter_[0] != flow_order_[adapter_start_flow])
        continue;

      // Evaluate this starting position
      int adapter_pos = 0;
      float score_match = 0;
      int score_len_flows = 0;
      int local_sequence_pos = sequence_pos;
      int local_start_base = sequence_pos;

      for (int flow = adapter_start_flow; flow < flow_order_.num_flows(); ++flow) {

        int base_delta = 0;
        while (adapter_pos < (int)trim_adapter_.length() and trim_adapter_[adapter_pos] == flow_order_[flow]) {
          adapter_pos++;
          base_delta--;
        }

        while (local_sequence_pos < (int)read.sequence.size() and base_to_flow[local_sequence_pos] == flow) {
          local_sequence_pos++;
          base_delta++;
        }

        if (flow != adapter_start_flow or base_delta < 0) {
          if (trim_adapter_mode_ == 0)
            score_match += base_delta*base_delta;
          else
            score_match += base_delta*base_delta + 2*base_delta*scaled_residual[flow] + scaled_residual[flow]*scaled_residual[flow];
        } else
          local_start_base += base_delta;
        score_len_flows++;

        if (adapter_pos == (int)trim_adapter_.length())
          break;
      }

      score_match /= score_len_flows;

      // Does this adapter alignment match our minimum acceptance criteria? If yes, is it better than other matches seen so far?

      if (adapter_pos < trim_adapter_min_match_)  // Match too short
        continue;

      if (score_match * 2 * trim_adapter_.length() > trim_adapter_cutoff_)  // Match too dissimilar
        continue;

      float final_metric = adapter_pos / (float)trim_adapter_.length() - score_match; // The higher the better

      if (final_metric > best_metric) {
        best_metric = final_metric;
        best_start_flow = adapter_start_flow;
        best_start_base = local_start_base;
        best_adapter_overlap = adapter_pos;
      }
    }
  }


  if (best_start_flow == -1)    // No suitable match
    return;

  // Save trimming results

  processed_read.bam.AddTag("ZA", "i", max(best_start_base - processed_read.filter.n_bases_prefix, 0));
  processed_read.bam.AddTag("ZG", "i", best_start_flow);
  processed_read.bam.AddTag("ZB", "i", best_adapter_overlap);


  if (filter_mask_[read_index] != kPassed) // Already filtered out?
    return;

  int trim_length = best_start_base - processed_read.filter.n_bases_prefix;

  if (trim_length < trim_min_read_len_) { // Adapter trimming led to filtering
    filter_mask_[read_index] = kFilteredShortAdapterTrim;
    processed_read.filter.n_bases_filtered = 0;
    processed_read.filter.n_bases_after_adapter_trim = processed_read.filter.n_bases_filtered;
    processed_read.filter.is_filtered = true;

  } else {
    processed_read.filter.n_bases_filtered = min(processed_read.filter.n_bases_filtered, best_start_base);
    processed_read.filter.n_bases_after_adapter_trim = processed_read.filter.n_bases_filtered;
  }

}



void BaseCallerFilters::TrimQuality(int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<uint8_t>& quality)
{
  if (filter_history.is_filtered) // Already filtered out?
    return;

  if(trim_qual_cutoff_ >= 100.0)   // 100.0 or more means disabled
    return;

  if (read_class != 0)  // Hardcoded: Don't trim TFs
    return;

  int window_start = 0;
  int window_end = 0;
  int window_sum = 0;
  int minimum_sum = trim_qual_window_size_ * trim_qual_cutoff_;

  // Step 1: Accumulate over the initial window
  while (window_end < trim_qual_window_size_ and window_end < filter_history.n_bases)
    window_sum += quality[window_end++];

  uint16_t clip_qual_right = 0;

  // Step 2: Keep sliding as long as average q-score exceeds the threshold
  while (window_sum >= minimum_sum and window_end < filter_history.n_bases) {
    window_sum += quality[window_end++];
    window_sum -= quality[window_start++];
    clip_qual_right = (window_end + window_start) / 2;
  }


  if (clip_qual_right >= filter_history.n_bases_filtered)
    return;


  int trim_length = clip_qual_right - filter_history.n_bases_prefix;

  if (trim_length < trim_min_read_len_) { // Quality trimming led to filtering
    filter_mask_[read_index] = kFilteredShortQualityTrim;
    filter_history.n_bases_after_quality_trim = filter_history.n_bases_filtered = 0;
    filter_history.is_filtered = true;

  } else {
    filter_history.n_bases_after_quality_trim = filter_history.n_bases_filtered = clip_qual_right;
  }
}










