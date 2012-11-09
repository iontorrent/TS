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
#include "SFFTrim/QScoreTrim.h"
#include "SFFTrim/adapter_searcher.h"

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
  printf ("                                        apply Beverly filter/trimmer [0.03,0.03,8]\n");
  printf ("     --trim-adapter          STRING     reverse complement of adapter sequence [ATCACCGACTGCCCATAGAGAGGCTGAGAC]\n");
  printf ("     --trim-adapter-cutoff   FLOAT      cutoff for adapter trimming, 0=off [16]\n");
  printf ("     --trim-adapter-min-match INT       minimum adapter bases in the read required for trimming  [6]\n");
//  printf ("     --trim-adapter-pick-closest on/off use closest candidate match, rather than longest [off]\n");
  printf ("     --trim-qual-window-size INT        window size for quality trimming [30]\n");
  printf ("     --trim-qual-cutoff      FLOAT      cutoff for quality trimming, 100=off [9]\n");
  printf ("     --trim-min-read-len     INT        reads trimmed shorter than this are omitted from output [8]\n");
  printf ("     --bead-summary          on/off     generate bead summary file [off]\n");
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

  bases_initial_.assign(mask.H()*mask.W(), 0);
  bases_final_.assign(mask.H()*mask.W(), 0);
  bases_removed_short_.assign(mask.H()*mask.W(), 0);
  bases_removed_key_trim_.assign(mask.H()*mask.W(), 0);
  bases_removed_barcode_trim_.assign(mask.H()*mask.W(), 0);
  bases_removed_keypass_.assign(mask.H()*mask.W(), 0);
  bases_removed_residual_.assign(mask.H()*mask.W(), 0);
  bases_removed_beverly_.assign(mask.H()*mask.W(), 0);
  bases_removed_adapter_trim_.assign(mask.H()*mask.W(), 0);
  bases_removed_quality_trim_.assign(mask.H()*mask.W(), 0);

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
  trim_adapter_closest_           = opts.GetFirstBoolean('-', "trim-adapter-pick-closest", false); //!\todo remove soon
  trim_adapter_min_match_         = opts.GetFirstInt    ('-', "trim-adapter-min-match", 6);
  trim_qual_window_size_          = opts.GetFirstInt    ('-', "trim-qual-window-size", 30);
  trim_qual_cutoff_               = opts.GetFirstDouble ('-', "trim-qual-cutoff", 9.0);
  trim_min_read_len_              = opts.GetFirstInt    ('-', "trim-min-read-len", 8);
  generate_bead_summary_          = opts.GetFirstBoolean('-', "bead-summary", false);


  string filter_beverly_args      = opts.GetFirstString ('-', "beverly-filter", "0.03,0.03,8");

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

void BaseCallerFilters::GenerateFilteringStatistics(Json::Value &json, const Mask& mask) const
{
  vector<int> num_valid(num_classes_, 0);
  vector<int> num_zero_bases(num_classes_, 0);
  vector<int> num_short_read(num_classes_, 0);
  vector<int> num_failed_keypass(num_classes_, 0);
  vector<int> num_high_ppf(num_classes_, 0);
  vector<int> num_polyclonal(num_classes_, 0);
  vector<int> num_high_residual(num_classes_, 0);
  vector<int> num_beverly(num_classes_, 0);
  vector<int> num_bkgmodel_keypass(num_classes_, 0);
  vector<int> num_bkgmodel_high_ppf(num_classes_, 0);
  vector<int> num_bkgmodel_polyclonal(num_classes_, 0);
  vector<int> num_short_adapter_trim(num_classes_, 0);
  vector<int> num_short_quality_trim(num_classes_, 0);
  vector<int> num_total(num_classes_, 0);

  for (size_t idx = 0; idx < filter_mask_.size(); idx++) {
    int read_class = mask.Match(idx, MaskLib) ? 0 : 1;  // Dependency on mask...

    if (filter_mask_[idx] != kUninitialized)
      num_total[read_class]++;

    switch (filter_mask_[idx]) {
      case kPassed:                     num_valid               [read_class]++; break;
      case kFilterZeroBases:            num_zero_bases          [read_class]++; break;
      case kFilterShortRead:            num_short_read          [read_class]++; break;
      case kFilterFailedKeypass:        num_failed_keypass      [read_class]++; break;
      case kFilterHighPPF:              num_high_ppf            [read_class]++; break;
      case kFilterPolyclonal:           num_polyclonal          [read_class]++; break;
      case kFilterHighResidual:         num_high_residual       [read_class]++; break;
      case kFilterBeverly:              num_beverly             [read_class]++; break;
      case kBkgmodelFailedKeypass:      num_bkgmodel_keypass    [read_class]++; break;
      case kBkgmodelHighPPF:            num_bkgmodel_high_ppf   [read_class]++; break;
      case kBkgmodelPolyclonal:         num_bkgmodel_polyclonal [read_class]++; break;
      case kFilteredShortAdapterTrim:   num_short_adapter_trim  [read_class]++; break;
      case kFilteredShortQualityTrim:   num_short_quality_trim  [read_class]++; break;
    }
  }

  ostringstream table;
  table.imbue(locale(table.getloc(), new ThousandsSeparator));

  table << endl;
  table << setw(25) << " ";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << (keys_[read_class].name() + " (" + keys_[read_class].bases() + ")");
  table << endl;

  table << setw(25) << "Examined wells";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << num_total[read_class] ;
  table << endl;

  table << setw(26) << " ";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << "------------";
  table << endl;

  table << setw(25) << "BkgModel:   High PPF";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << -num_bkgmodel_high_ppf[read_class] ;
  table << endl;

  table << setw(25) << "BkgModel: Polyclonal";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << -num_bkgmodel_polyclonal[read_class] ;
  table << endl;

  table << setw(25) << "BkgModel:    Bad key";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << -num_bkgmodel_keypass[read_class] ;
  table << endl;

  table << setw(25) << "High PPF";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << -num_high_ppf[read_class] ;
  table << endl;

  table << setw(25) << "Polyclonal";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << -num_polyclonal[read_class] ;
  table << endl;

  table << setw(25) << "Zero bases";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << -num_zero_bases[read_class] ;
  table << endl;

  table << setw(25) << "Short read";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << -num_short_read[read_class] ;
  table << endl;

  table << setw(25) << "Bad key";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << -num_failed_keypass[read_class] ;
  table << endl;

  table << setw(25) << "High residual";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << -num_high_residual[read_class] ;
  table << endl;

  table << setw(25) << "Beverly filter";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << -num_beverly[read_class] ;
  table << endl;

  table << setw(25) << "Short after adapter trim";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << -num_short_adapter_trim[read_class] ;
  table << endl;

  table << setw(25) << "Short after quality trim";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << -num_short_quality_trim[read_class] ;
  table << endl;

  table << setw(26) << " ";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << "------------";
  table << endl;

  table << setw(25) << "Valid reads saved to SFF";
  for (int read_class = 0; read_class < num_classes_; read_class++)
    table << setw(15) << num_valid[read_class] ;
  table << endl;
  table << endl;

  cout << table.str();
  table.clear();
  table.str("");

  // Report base fitering - without breakdown into classes

  int64_t num_bases_initial = 0;
  int64_t num_bases_final = 0;
  int64_t num_bases_removed_key_trim = 0;
  int64_t num_bases_removed_barcode_trim = 0;
  int64_t num_bases_removed_short = 0;
  int64_t num_bases_removed_keypass = 0;
  int64_t num_bases_removed_residual = 0;
  int64_t num_bases_removed_beverly = 0;
  int64_t num_bases_removed_adapter_trim = 0;
  int64_t num_bases_removed_quality_trim = 0;

  for (size_t idx = 0; idx < filter_mask_.size(); idx++) {
    num_bases_initial               += bases_initial_[idx];
    num_bases_final                 += bases_final_[idx];
    num_bases_removed_key_trim      += bases_removed_key_trim_[idx];
    num_bases_removed_barcode_trim  += bases_removed_barcode_trim_[idx];
    num_bases_removed_short         += bases_removed_short_[idx];
    num_bases_removed_keypass       += bases_removed_keypass_[idx];
    num_bases_removed_residual      += bases_removed_residual_[idx];
    num_bases_removed_beverly       += bases_removed_beverly_[idx];
    num_bases_removed_adapter_trim  += bases_removed_adapter_trim_[idx];
    num_bases_removed_quality_trim  += bases_removed_quality_trim_[idx];
  }

  table << setw(35) << "Generated library bases"        << setw(20) << num_bases_initial << endl;
  table << setw(35) << "Key bases"                      << setw(20) << -num_bases_removed_key_trim << endl;
  table << setw(35) << "Barcode bases"                  << setw(20) << -num_bases_removed_barcode_trim << endl;
  table << setw(35) << "Short reads"                    << setw(20) << -num_bases_removed_short << endl;
  table << setw(35) << "Bad key"                        << setw(20) << -num_bases_removed_keypass << endl;
  table << setw(35) << "High residual"                  << setw(20) << -num_bases_removed_residual << endl;
  table << setw(35) << "Beverly filter"                 << setw(20) << -num_bases_removed_beverly << endl;
  table << setw(35) << "Adapter trimming"               << setw(20) << -num_bases_removed_adapter_trim << endl;
  table << setw(35) << "Quality trimming"               << setw(20) << -num_bases_removed_quality_trim << endl;
  table << setw(36) << " "                              << setw(20) << "-----------------" << endl;
  table << setw(35) << "Bases (incl. key) saved to sff" << setw(20) << num_bases_final << endl;
  table << endl;

  cout << table.str();

  //
  // Save results to to json
  //

  // BeadSummary section is just for backward compatibility
  for (int read_class = 0; read_class < num_classes_; read_class++) {
    json["BeadSummary"][keys_[read_class].name()]["key"]         = keys_[read_class].bases();
    json["BeadSummary"][keys_[read_class].name()]["polyclonal"]  = num_polyclonal[read_class] + num_bkgmodel_polyclonal[read_class];
    json["BeadSummary"][keys_[read_class].name()]["highPPF"]     = num_high_ppf[read_class] + num_bkgmodel_high_ppf[read_class];
    json["BeadSummary"][keys_[read_class].name()]["zero"]        = num_zero_bases[read_class];
    json["BeadSummary"][keys_[read_class].name()]["short"]       = num_short_read[read_class] + num_short_adapter_trim[read_class] + num_short_quality_trim[read_class];
    json["BeadSummary"][keys_[read_class].name()]["badKey"]      = num_failed_keypass[read_class] + num_bkgmodel_keypass[read_class];
    json["BeadSummary"][keys_[read_class].name()]["highRes"]     = num_high_residual[read_class] + num_beverly[read_class];
    json["BeadSummary"][keys_[read_class].name()]["valid"]       = num_valid[read_class];
  }

  // Generate values that go to the library report - assume library is class 0
  json["Filtering"]["LibraryReport"]["filtered_polyclonal"]   = num_polyclonal[0] + num_bkgmodel_polyclonal[0];
  json["Filtering"]["LibraryReport"]["filtered_primer_dimer"] = num_short_adapter_trim[0];
  json["Filtering"]["LibraryReport"]["filtered_low_quality"]
            = num_high_ppf[0] + num_bkgmodel_high_ppf[0]
            + num_zero_bases[0] + num_short_read[0] + num_short_quality_trim[0]
            + num_failed_keypass[0] + num_bkgmodel_keypass[0]
            + num_high_residual[0] + num_beverly[0];
  json["Filtering"]["LibraryReport"]["final_library_reads"]   = num_valid[0];

  // ReadDetails
  for (int read_class = 0; read_class < num_classes_; read_class++) {
    json["Filtering"]["ReadDetails"][keys_[read_class].name()]["key"]                 = keys_[read_class].bases();
    json["Filtering"]["ReadDetails"][keys_[read_class].name()]["bkgmodel_polyclonal"] = num_bkgmodel_polyclonal[read_class];
    json["Filtering"]["ReadDetails"][keys_[read_class].name()]["bkgmodel_high_ppf"]   = num_bkgmodel_high_ppf[read_class];
    json["Filtering"]["ReadDetails"][keys_[read_class].name()]["bkgmodel_keypass"]    = num_bkgmodel_keypass[read_class];
    json["Filtering"]["ReadDetails"][keys_[read_class].name()]["polyclonal"]          = num_polyclonal[read_class];
    json["Filtering"]["ReadDetails"][keys_[read_class].name()]["high_ppf"]            = num_high_ppf[read_class];
    json["Filtering"]["ReadDetails"][keys_[read_class].name()]["zero"]                = num_zero_bases[read_class];
    json["Filtering"]["ReadDetails"][keys_[read_class].name()]["short"]               = num_short_read[read_class];
    json["Filtering"]["ReadDetails"][keys_[read_class].name()]["failed_keypass"]      = num_failed_keypass[read_class];
    json["Filtering"]["ReadDetails"][keys_[read_class].name()]["high_residual"]       = num_high_residual[read_class];
    json["Filtering"]["ReadDetails"][keys_[read_class].name()]["adapter_trim"]        = num_short_adapter_trim[read_class];
    json["Filtering"]["ReadDetails"][keys_[read_class].name()]["quality_trim"]        = num_short_quality_trim[read_class];
    json["Filtering"]["ReadDetails"][keys_[read_class].name()]["beverly_filter"]      = num_beverly[read_class];
    json["Filtering"]["ReadDetails"][keys_[read_class].name()]["valid"]               = num_valid[read_class];
  }

  // BaseDetails
  json["Filtering"]["BaseDetails"]["initial"]         = (Json::Int64)num_bases_initial;
  json["Filtering"]["BaseDetails"]["short"]           = (Json::Int64)num_bases_removed_short;
  json["Filtering"]["BaseDetails"]["failed_keypass"]  = (Json::Int64)num_bases_removed_keypass;
  json["Filtering"]["BaseDetails"]["high_residual"]   = (Json::Int64)num_bases_removed_residual;
  json["Filtering"]["BaseDetails"]["beverly_filter"]  = (Json::Int64)num_bases_removed_beverly;
  json["Filtering"]["BaseDetails"]["adapter_trim"]    = (Json::Int64)num_bases_removed_adapter_trim;
  json["Filtering"]["BaseDetails"]["quality_trim"]    = (Json::Int64)num_bases_removed_quality_trim;
  json["Filtering"]["BaseDetails"]["final"]           = (Json::Int64)num_bases_final;



  if (generate_bead_summary_) {

    ofstream bead_summary;
    bead_summary.open("beadSummary.filtered.txt");
    if(bead_summary.fail()) {
      ION_WARN("Unable to open output bead summary file beadSummary.filtered.txt for write");
      return;
    }

    string delim = "\t";
    bead_summary << "class" << delim;
    bead_summary << "key" << delim;
    bead_summary << "polyclonal" << delim;
    bead_summary << "highPPF" << delim;
    bead_summary << "zero" << delim;
    bead_summary << "short" << delim;
    bead_summary << "badKey" << delim;
    bead_summary << "highRes" << delim;
    bead_summary << "clipAdapter" << delim;
    bead_summary << "clipQual" << delim;
    bead_summary << "valid" << endl;

    for (int read_class = 0; read_class < num_classes_; read_class++) {

      bead_summary
          << keys_[read_class].name() << delim
          << keys_[read_class].bases() << delim
          << (num_polyclonal[read_class] + num_bkgmodel_polyclonal[read_class]) << delim
          << (num_high_ppf[read_class] + num_bkgmodel_high_ppf[read_class]) << delim
          << (num_zero_bases[read_class]) << delim
          << (num_short_read[read_class]) << delim
          << (num_failed_keypass[read_class] + num_bkgmodel_keypass[read_class]) << delim
          << (num_high_residual[read_class] + num_beverly[read_class]) << delim
          << num_short_adapter_trim[read_class] << delim
          << num_short_quality_trim[read_class] << delim
          << num_valid[read_class]  << endl;
    }
  }
}



void BaseCallerFilters::SetValid(int read_index)
{
  filter_mask_[read_index] = kPassed;
}

int GetActualBases (const SFFEntry& sff_entry)
{
  int left = 0;
  if (sff_entry.clip_qual_left > 0)
    left = max(left, (int)sff_entry.clip_qual_left-1);
  if (sff_entry.clip_adapter_left > 0)
    left = max(left, (int)sff_entry.clip_adapter_left-1);

  int right = sff_entry.n_bases;
  if (sff_entry.clip_qual_right > 0)
    right = min(right, (int)sff_entry.clip_qual_right);
  if (sff_entry.clip_adapter_right > 0)
    right = min(right, (int)sff_entry.clip_adapter_right);

  return max(right-left, 0);
}

void  BaseCallerFilters::SetReadLength (int read_index, const SFFEntry& sff_entry)
{
  bases_initial_[read_index] = sff_entry.n_bases;
  bases_removed_key_trim_[read_index] = 0;
  bases_removed_barcode_trim_[read_index] = 0;
  if (sff_entry.clip_qual_left > 0)
    bases_removed_key_trim_[read_index] = min(sff_entry.clip_qual_left-1, bases_initial_[read_index]);

  bases_final_[read_index] = GetActualBases(sff_entry);
  bases_removed_barcode_trim_[read_index] = bases_initial_[read_index] - bases_final_[read_index] - bases_removed_key_trim_[read_index];
}

bool BaseCallerFilters::IsValid(int read_index) const
{
  return filter_mask_[read_index] == kPassed;
}

bool BaseCallerFilters::IsPolyclonal(int read_index) const
{
  return filter_mask_[read_index] == kFilterPolyclonal;
}




void BaseCallerFilters::SetBkgmodelHighPPF(int read_index)
{
  if (filter_mask_[read_index] != kPassed) // Already filtered out?
    return;
  filter_mask_[read_index] = kBkgmodelHighPPF;
}

void BaseCallerFilters::SetBkgmodelPolyclonal(int read_index)
{
  if (filter_mask_[read_index] != kPassed) // Already filtered out?
    return;
  filter_mask_[read_index] = kBkgmodelPolyclonal;
}

void BaseCallerFilters::SetBkgmodelFailedKeypass(int read_index)
{
  if (filter_mask_[read_index] != kPassed) // Already filtered out?
    return;
  filter_mask_[read_index] = kBkgmodelFailedKeypass;
}


void BaseCallerFilters::FilterHighPPFAndPolyclonal (int read_index, int read_class, const vector<float>& measurements)
{
  if (filter_mask_[read_index] != kPassed) // Already filtered out?
    return;

  if (read_class == 0 and !filter_clonal_enabled_)  // Filter disabled for library?
    return;
  if (read_class != 0 and !filter_clonal_enabled_tfs_)  // Filter disabled for TFs?
    return;

  vector<float>::const_iterator first = measurements.begin() + mixed_first_flow();
  vector<float>::const_iterator last  = measurements.begin() + mixed_last_flow();
  float ppf = percent_positive(first, last);
  float ssq = sum_fractional_part(first, last);

  if(ppf > mixed_ppf_cutoff())
    filter_mask_[read_index] = kFilterHighPPF;
  else if(!clonal_population_.is_clonal(ppf, ssq))
    filter_mask_[read_index] = kFilterPolyclonal;
}


void BaseCallerFilters::FilterZeroBases(int read_index, int read_class, const SFFEntry& sff_entry)
{
  if (filter_mask_[read_index] != kPassed) // Already filtered out?
    return;

  if(sff_entry.n_bases == 0) {
    filter_mask_[read_index] = kFilterZeroBases;
    bases_removed_short_[read_index] = bases_final_[read_index];
    bases_final_[read_index] = 0;
  }
}


void BaseCallerFilters::FilterShortRead(int read_index, int read_class, const SFFEntry& sff_entry)
{
  if (filter_mask_[read_index] != kPassed) // Already filtered out?
    return;

  if(sff_entry.n_bases < filter_min_read_length_ or GetActualBases(sff_entry) < trim_min_read_len_) {
    filter_mask_[read_index] = kFilterShortRead;
    bases_removed_short_[read_index] = bases_final_[read_index];
    bases_final_[read_index] = 0;
  }
}



void BaseCallerFilters::FilterFailedKeypass(int read_index, int read_class, const vector<char> &solution)
{
  if (filter_mask_[read_index] != kPassed) // Already filtered out?
    return;

  if(!filter_keypass_enabled_)  // Filter disabled?
    return;

  bool failed_keypass = false;
  for (int flow = 0; flow < (keys_[read_class].flows_length()-1); flow++)
    if (keys_[read_class][flow] != solution[flow])
      failed_keypass = true;
  if (keys_[read_class][keys_[read_class].flows_length()-1] > solution[keys_[read_class].flows_length()-1])
    failed_keypass = true;

  if (failed_keypass) {
    filter_mask_[read_index] = kFilterFailedKeypass;
    bases_removed_keypass_[read_index] = bases_final_[read_index];
    bases_final_[read_index] = 0;
  }
}



void BaseCallerFilters::FilterHighResidual(int read_index, int read_class, const vector<float>& residual)
{
  if (filter_mask_[read_index] != kPassed) // Already filtered out?
    return;

  if (read_class == 0 and !filter_residual_enabled_)  // Filter disabled for library?
    return;
  if (read_class != 0 and !filter_residual_enabled_tfs_)  // Filter disabled for TFs?
    return;

  if(MedianAbsoluteCafieResidual(residual, 60) > filter_residual_max_value_) {
    filter_mask_[read_index] = kFilterHighResidual;
    bases_removed_residual_[read_index] = bases_final_[read_index];
    bases_final_[read_index] = 0;
  }
}

void BaseCallerFilters::FilterBeverly(int read_index, int read_class, const BasecallerRead &read, SFFEntry& sff_entry)
{

  bool reject = false;

  if (filter_beverly_enabled_ and read_class == 0) {    // What about random reads? What about TFs?

    int num_onemers = 0;
    int num_twomers = 0;
    int num_extreme_onemers = 0;
    int num_extreme_twomers = 0;
    int num_bases_seen = 0;
    int max_trim_bases = 0;

    for (int flow = 0; flow < flow_order_.num_flows(); ++flow) {

      if (read.solution[flow] == 1) {
        num_onemers++;
        if (sff_entry.flowgram[flow] <= 59 or sff_entry.flowgram[flow] >= 140)
          num_extreme_onemers++;
      }

      if (read.solution[flow] == 2) {
        num_twomers++;
        if (sff_entry.flowgram[flow] <= 159 or sff_entry.flowgram[flow] >= 240)
          num_extreme_twomers++;
      }

      num_bases_seen += read.solution[flow];
      if (num_extreme_onemers <= num_onemers * filter_beverly_trim_ratio_)
        max_trim_bases = num_bases_seen;
    }

    if ((num_extreme_onemers + num_extreme_twomers) > (num_onemers + num_twomers) * filter_beverly_filter_ratio_) {

      int trim_length = max_trim_bases - max(1,max(sff_entry.clip_adapter_left,sff_entry.clip_qual_left)) + 1;
      //if (max_trim_bases > filter_beverly_min_read_length_)
      if (trim_length < trim_min_read_len_) // Quality trimming led to filtering
        reject = true;
      else
        sff_entry.clip_qual_right = max_trim_bases;
    }
  }


  if (filter_mask_[read_index] != kPassed) // Already filtered out?
    return;

  if(reject) {
    filter_mask_[read_index] = kFilterBeverly;
    bases_removed_beverly_[read_index] = bases_final_[read_index];
    bases_final_[read_index] = 0;

  } else {
    int new_bases_final = min(bases_final_[read_index],GetActualBases(sff_entry));
    bases_removed_beverly_[read_index] = bases_final_[read_index] - new_bases_final;
    bases_final_[read_index] = new_bases_final;
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



void BaseCallerFilters::TrimAdapter(int read_index, int read_class, SFFEntry& sff_entry)
{
  if(trim_adapter_cutoff_ <= 0.0)  // Zero means disabled
    return;

  if (read_class != 0)  // Hardcoded: Don't trim TFs
    return;

  if (trim_adapter_min_match_ < 1000) {   // Magic number that switches to TrimAdapter2
    TrimAdapter2(read_index, read_class, sff_entry);
    return;
  }

  adapter_searcher as(flow_order_.str(), keys_[read_class].bases(), trim_adapter_);
  int num_matches = as.find_matches_sff(&sff_entry.flowgram[0], flow_order_.num_flows(), trim_adapter_cutoff_);
  if(num_matches <= 0)
    return; // Adapter not found

  adapter_searcher::match match;
  if(trim_adapter_closest_)
    match = as.pick_closest();
  else
    match = as.pick_longest();

  uint16_t clip_adapter_right = as.flow2pos(sff_entry.flow_index, sff_entry.bases, sff_entry.n_bases, match._flow);

  if (clip_adapter_right == 0)
    return;

  if (sff_entry.clip_adapter_right == 0 or clip_adapter_right < sff_entry.clip_adapter_right) {
    sff_entry.clip_adapter_right = clip_adapter_right; //Trim

    if (filter_mask_[read_index] != kPassed) // Already filtered out?
      return;

    int trim_length = clip_adapter_right - max(1,max(sff_entry.clip_adapter_left,sff_entry.clip_qual_left)) + 1;

    if (trim_length < trim_min_read_len_) { // Adapter trimming led to filtering
      filter_mask_[read_index] = kFilteredShortAdapterTrim;
      bases_removed_adapter_trim_[read_index] = bases_final_[read_index];
      bases_final_[read_index] = 0;

    } else {
      int new_bases_final = min(bases_final_[read_index],GetActualBases(sff_entry));
      bases_removed_adapter_trim_[read_index] = bases_final_[read_index] - new_bases_final;
      bases_final_[read_index] = new_bases_final;
    }
  }
}

void BaseCallerFilters::TrimAdapter2(int read_index, int read_class, SFFEntry& sff_entry)
{
  if (trim_adapter_.empty())
    return;


  int best_start_flow = -1;
  float best_metric = -1e10;

  for (int adapter_start_flow = 0; adapter_start_flow < flow_order_.num_flows(); ++adapter_start_flow) {

    // Only consider start flows that agree with adapter start
    if (trim_adapter_[0] != flow_order_[adapter_start_flow])
      continue;

    // Evaluate this starting position
    int adapter_pos = 0;
    float score_match = 0;
    int score_len_flows = 0;

    for (int flow = adapter_start_flow; flow < flow_order_.num_flows(); ++flow) {

      int adapter_hp = 0;
      while (trim_adapter_[adapter_pos] == flow_order_[flow]) {
        adapter_pos++;
        adapter_hp++;
        if (adapter_pos == (int)trim_adapter_.length())
          break;
      }

      float delta = adapter_hp - sff_entry.flowgram[flow] / 100.0;
      if (flow == adapter_start_flow)
        delta = max(delta,0.0f);

      score_match += delta*delta;
      score_len_flows += 1;

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
    }
  }

  if (best_start_flow == -1)    // No suitable match
    return;

  // Determine trim base

  int trim_pos_flow = -1;
  int clip_adapter_right = 0;
  bool match_last_base = false;

  for (unsigned int base = 0; base < sff_entry.flow_index.size(); ++base) {
    trim_pos_flow += sff_entry.flow_index[base];
    if (trim_pos_flow > best_start_flow)
      break;
    clip_adapter_right = base + 1;
    if (trim_pos_flow == best_start_flow)
      match_last_base = true;
  }

  if (match_last_base)  // This implicitly assumes adapter starts with 1-mer.
    clip_adapter_right--;

  // Save trimming results

  if (sff_entry.clip_adapter_right == 0 or clip_adapter_right < sff_entry.clip_adapter_right) {
    sff_entry.clip_adapter_right = clip_adapter_right; //Trim
    sff_entry.clip_adapter_flow = best_start_flow;

    if (filter_mask_[read_index] != kPassed) // Already filtered out?
      return;

    int trim_length = clip_adapter_right - max(1,max(sff_entry.clip_adapter_left,sff_entry.clip_qual_left)) + 1;

    if (trim_length < trim_min_read_len_) { // Adapter trimming led to filtering
      filter_mask_[read_index] = kFilteredShortAdapterTrim;
      bases_removed_adapter_trim_[read_index] = bases_final_[read_index];
      bases_final_[read_index] = 0;

    } else {
      int new_bases_final = min(bases_final_[read_index],GetActualBases(sff_entry));
      bases_removed_adapter_trim_[read_index] = bases_final_[read_index] - new_bases_final;
      bases_final_[read_index] = new_bases_final;
    }
  }

}



void BaseCallerFilters::TrimQuality(int read_index, int read_class, SFFEntry& sff_entry)
{
  if(trim_qual_cutoff_ >= 100.0)   // 100.0 or more means disabled
    return;

  if (read_class != 0)  // Hardcoded: Don't trim TFs
    return;

  uint8_t  *qbeg = &sff_entry.quality[0];
  uint8_t  *qend = qbeg + sff_entry.n_bases;
  uint8_t  *clip = QualTrim(qbeg, qend, trim_qual_cutoff_, trim_qual_window_size_);

  uint16_t clip_qual_right = clip - qbeg;

  if (sff_entry.clip_qual_right == 0 or clip_qual_right < sff_entry.clip_qual_right) {
    sff_entry.clip_qual_right = clip_qual_right; //Trim

    if (filter_mask_[read_index] != kPassed) // Already filtered out?
      return;

    int trim_length = clip_qual_right - max(1,max(sff_entry.clip_adapter_left,sff_entry.clip_qual_left)) + 1;

    if (trim_length < trim_min_read_len_) { // Quality trimming led to filtering
      filter_mask_[read_index] = kFilteredShortQualityTrim;
      bases_removed_quality_trim_[read_index] = bases_final_[read_index];
      bases_final_[read_index] = 0;

    } else {
      int new_bases_final = min(bases_final_[read_index],GetActualBases(sff_entry));
      bases_removed_quality_trim_[read_index] = bases_final_[read_index] - new_bases_final;
      bases_final_[read_index] = new_bases_final;
    }
  }
}










