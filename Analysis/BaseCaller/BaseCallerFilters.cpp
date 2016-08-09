/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BaseCallerFilters.cpp
//! @ingroup  BaseCaller
//! @brief    BaseCallerFilters. Filtering and trimming algorithms, configuration, and accounting

#include "BaseCallerFilters.h"

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <sys/types.h>
#include <algorithm>
#include <iostream>

#include "LinuxCompat.h"
#include "Stats.h"
#include "IonErr.h"
#include "RawWells.h"
#include "Mask.h"
#include "BaseCallerUtils.h"
#include "DPTreephaser.h"
#include "OrderedDatasetWriter.h"

using namespace std;


enum FilteringOutcomes_t {
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
  kFilteredShortQualityTrim,    //!< Read filtered out, too short after quality trimming
  kFilteredAvalancheTrim        //!< Read filtered out, too short after Avalanche trimming
};

enum QualityTrimmingModes {
  kQvTrimOff,                    //!< Quality trimming disabled
  kQvTrimWindowed,               //!< Sliding window quality trimming
  kQvTrimExpectedErrors,         //!< Expected number of errors quality trimming
  kQvTrimAll                     //!< Quality trimming using all available methods
};



class ThousandsSeparator : public numpunct<char> {
protected:
    string do_grouping() const { return "\03"; }
};




// ----------------------------------------------------------------------------

int EncodeFilteringDetails(int n_base, int n_prefix)
{
  if (n_base < 0)
    return n_base;
  return max(0,n_base - n_prefix);
}

// ----------------------------------------------------------------------------

ReadFilteringHistory::ReadFilteringHistory()
{
  is_filtered = false;
  is_called   = false;
  n_bases     = -1;

  n_bases_key    = 0;
  n_bases_prefix = 0;

  n_bases_after_bkgmodel_high_ppf   = -1;
  n_bases_after_bkgmodel_polyclonal = -1;
  n_bases_after_bkgmodel_bad_key    = -1;
  n_bases_after_polyclonal          = -1;
  n_bases_after_high_ppf            = -1;
  n_bases_after_too_short           = -1;
  n_bases_after_bad_key             = -1;
  n_bases_after_high_residual       = -1;
  n_bases_after_beverly_trim        = -1;
  n_bases_after_quality_trim        = -1;
  n_bases_after_adapter_trim        = -1;
  n_bases_filtered                  = -1;

  adapter_type       = -1;
  adapter_score      = 0.0;
  adapter_separation = 0.0;
  adapter_decision   = false;
}

// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------

void ReadFilteringStats::SetBeadAdapters(const vector<string> & trim_adapters){
	bead_adapters_ = trim_adapters;

    adapter_class_num_reads_.assign(bead_adapters_.size()+1, 0);
    adapter_class_num_decisions_.assign(bead_adapters_.size(), 0);
    adapter_class_cum_score_.assign(bead_adapters_.size(), 0.0);
	adapter_class_cum_separation_.assign(bead_adapters_.size(), 0.0);
	adapter_class_av_score_.assign(bead_adapters_.size(), 0.0);
    adapter_class_av_separation_.assign(bead_adapters_.size(), 0.0);

};

// ----------------------------------------------------------------------------

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

  // Step 3: Bead Adapter accounting

  if(not read_filtering_history.is_filtered and read_filtering_history.adapter_type >= 0) {
    adapter_class_num_reads_.at(read_filtering_history.adapter_type)++;

    if (read_filtering_history.adapter_type < (int)adapter_class_cum_score_.size()) {
      adapter_class_cum_score_.at(read_filtering_history.adapter_type) += read_filtering_history.adapter_score;
      if (read_filtering_history.adapter_decision) {
        adapter_class_num_decisions_.at(read_filtering_history.adapter_type)++;
        adapter_class_cum_separation_.at(read_filtering_history.adapter_type) += read_filtering_history.adapter_separation;
      }
    }
  }

}


// ----------------------------------------------------------------------------

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

  for (unsigned int iadptr=0; iadptr<adapter_class_cum_score_.size(); iadptr++){
    adapter_class_num_reads_.at(iadptr)      += other.adapter_class_num_reads_.at(iadptr);
    adapter_class_cum_score_.at(iadptr)      += other.adapter_class_cum_score_.at(iadptr);
    adapter_class_num_decisions_.at(iadptr)  += other.adapter_class_num_decisions_.at(iadptr);
    adapter_class_cum_separation_.at(iadptr) += other.adapter_class_cum_separation_.at(iadptr);
  }
  adapter_class_num_reads_.at(adapter_class_cum_score_.size()) +=
        other.adapter_class_num_reads_.at(adapter_class_cum_score_.size());
}

// ----------------------------------------------------------------------------

void ReadFilteringStats::ComputeAverages(){

  for (unsigned int iadptr=0; iadptr<adapter_class_cum_score_.size(); iadptr++){
    if (adapter_class_num_reads_.at(iadptr) > 0)
      adapter_class_av_score_.at(iadptr)      = adapter_class_cum_score_.at(iadptr) / adapter_class_num_reads_.at(iadptr);
    if (adapter_class_num_decisions_.at(iadptr) > 0)
      adapter_class_av_separation_.at(iadptr) = adapter_class_cum_separation_.at(iadptr) / adapter_class_num_decisions_.at(iadptr);
  }
};

// ----------------------------------------------------------------------------

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

  // Printing bead adapter summary
  int fill_length = 1;
  for (unsigned int iadptr=0; iadptr<bead_adapters_.size(); iadptr++)
    fill_length = max(fill_length, (int)bead_adapters_.at(iadptr).length());
  fill_length = max(fill_length+3, 23);
  string dashes_line(fill_length-3, '-');

  table << setw(25) << "Bead Adapters";
  table << setw(23) << "Num Reads" << setw(fill_length) << "Adapter Sequence" << endl;

  table << setw(25) << " ";
  table << setw(23) << "--------------------" << setw(fill_length) << dashes_line << endl;

  for (unsigned int iadptr=0; iadptr<adapter_class_cum_score_.size(); iadptr++){
    table << setw(22) << "Adapter" << setw(3) << iadptr;
    table << setw(23) << adapter_class_num_reads_.at(iadptr) << setw(fill_length) << bead_adapters_.at(iadptr) << endl;
  }

  table << setw(25) << "No Adapter Match";
  table << setw(23) << adapter_class_num_reads_.at(bead_adapters_.size()) << setw(fill_length) << " " << endl;

  table << setw(25) << " ";
  table << setw(23) << "--------------------" << setw(fill_length) << dashes_line << endl;
  cout << endl;

  cout << table.str();
}

// ----------------------------------------------------------------------------

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

  // Generate values specific to library reads
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

    // (3') Bead Adapters

    stringstream adapter_stream;
    for (unsigned int adapter_idx=0; adapter_idx<bead_adapters_.size(); adapter_idx++) {
      adapter_stream.str(std::string());
      adapter_stream << "Adapter_" << adapter_idx;
      json["Filtering"]["BeadAdapters"][adapter_stream.str()]["adapter_sequence"] = bead_adapters_.at(adapter_idx);
      json["Filtering"]["BeadAdapters"][adapter_stream.str()]["read_count"] = (Json::UInt64)(adapter_class_num_reads_.at(adapter_idx));
      json["Filtering"]["BeadAdapters"][adapter_stream.str()]["num_decisions"] = (Json::UInt64)(adapter_class_num_decisions_.at(adapter_idx));

      Json::Value av_score(adapter_class_av_score_.at(adapter_idx));
      json["Filtering"]["BeadAdapters"][adapter_stream.str()]["average_metric"] = av_score;

      Json::Value av_sep(adapter_class_av_separation_.at(adapter_idx));
      json["Filtering"]["BeadAdapters"][adapter_stream.str()]["average_separation"] = av_sep;
    }
    json["Filtering"]["BeadAdapters"]["no_match"]["read_count"] = (Json::UInt64)(adapter_class_num_reads_.at(bead_adapters_.size()));
  }

}


// ===================================================================================
// XXX Start of BaseCallerFilters functions

void BaseCallerFilters::PrintHelp()
{
  printf ("Read filtering and trimming options:\n");
  printf ("  -d,--disable-all-filters    on/off     disable all filtering and trimming, overrides other args [off]\n");
  printf (" Key-pass filter: \n");
  printf ("  -k,--keypass-filter         on/off     apply keypass filter [on]\n");
  printf (" Polyclonal filter in BaseCaller:\n");
  printf ("     --clonal-filter-solve    on/off     apply polyclonal filter [off]\n");
  printf ("     --clonal-filter-tf       on/off     apply polyclonal filter to TFs [off]\n");
  printf ("     --clonal-filter-maxreads INT        maximum number of library reads used for polyclonal filter training [100000]\n");
  printf ("     --min-read-length        INT        apply minimum read length filter [25]\n");
  printf ("     --cr-filter              on/off     apply cafie residual filter [off]\n");
  printf ("     --cr-filter-tf           on/off     apply cafie residual filter to TFs [off]\n");
  printf ("     --cr-filter-max-value    FLOAT      cafie residual filter threshold [0.08]\n");
  printf ("     --beverly-filter      FLOAT,FLOAT   filter_ratio,trim_ratio / off\n");
  printf ("                                         apply Beverly filter/trimmer [off]\n");
  printf ("     --qual-filter            on/off     apply quality filter based on expected number of errors [off]\n");
  printf ("     --qual-filter-offset     FLOAT      error offset for expected errors quality filter [0.7]\n");
  printf ("     --qual-filter-slope      FLOAT      expected errors allowed per base for expected errors quality filter [0.02]\n");
  printf ("\n");
  printf ("Read trimming options:\n");
  printf ("     --trim-min-read-len      INT        reads trimmed shorter than this are omitted from output [min-read-length]\n");
  printf ("     --trim-barcodes          BOOL       trim barcodes and barcode adapters [on]\n");
  printf (" Adapter trimming (turn off by supplying cutoff 0):\n");
  printf ("     --trim-adapter-mode      INT        0=use simplified metric, 1=use standard metric [1]\n");
  printf ("     --trim-adapter           STRING     reverse complement of adapter sequence [ATCACCGACTGCCCATAGAGAGGCTGAGAC]\n");
  printf ("     --trim-adapter-tf        STRING/off adapter sequence for test fragments [off]\n");
  printf ("     --trim-adapter-cutoff    FLOAT      cutoff for adapter trimming, 0=off [16]\n");
  printf ("     --trim-adapter-min-match INT        minimum adapter bases in the read required for trimming  [6]\n");
  printf (" Quality trimming (turn off by supplying mode 'off'):\n");
  printf ("     --trim-qual-mode         STRING     select the method of quality trimming [\"sliding-window\"]\n");
  printf ("                                         mode \"off\"             :  quality trimming disabled\n");
  printf ("                                         mode \"sliding-window\"  :  sliding window quality trimming\n");
  printf ("                                         mode \"expected-errors\" :  quality trimming based on expected number of errors in read\n");
  printf ("                                         mode \"all\"             :  shortest trimming point of all methods\n");
  printf ("     --trim-qual-window-size  INT        window size for windowed quality trimming [30]\n");
  printf ("     --trim-qual-cutoff       FLOAT      cutoff for windowed quality trimming, 100=off [15]\n");
  printf ("     --trim-qual-offset       FLOAT      error threshold offset for expected error quality trimming [0.7]\n");
  printf ("     --trim-qual-slope        FLOAT      increase of expected errors allowed for expected error quality trimming [0.005]\n");
  printf (" Avalanche trimming (does anybody use this?):\n"); // XXX
  printf ("     --avalanche_start_pos    INT        Avalanche filter start base position\n");
  printf ("     --avalanche_qual_hi      INT        Avalanche filter quality cutoff before the start position [15]\n");
  printf ("     --avalanche_qual_lo      INT        Avalanche filter quality cutoff after the start position [5]\n");
  printf ("\n");
}

// ----------------------------------------------------------------------------

BaseCallerFilters::BaseCallerFilters(OptArgs& opts, vector<string> & bam_comments, const string & run_id,
    const ion::FlowOrder& flow_order, const vector<KeySequence>& keys, const Mask& mask)
{
  flow_order_ = flow_order;
  keys_ = keys;
  num_classes_ = keys_.size();
  assert(num_classes_ == 2);
  filter_mask_.assign(mask.H()*mask.W(), kUninitialized);

  // *** Retrieve filter command line options

  filter_keypass_enabled_      = opts.GetFirstBoolean('k', "keypass-filter", true);
  filter_min_read_length_      = opts.GetFirstInt    ('-', "min-read-length", 25);
  trim_min_read_len_           = opts.GetFirstInt    ('-', "trim-min-read-len", filter_min_read_length_);

  filter_clonal_enabled_tfs_   = opts.GetFirstBoolean('-', "clonal-filter-tf", false);
  filter_clonal_enabled_       = opts.GetFirstBoolean('-', "clonal-filter-solve", false);
  filter_clonal_maxreads_      = opts.GetFirstInt    ('-', "clonal-filter-maxreads", 100000);

  filter_residual_enabled_     = opts.GetFirstBoolean('-', "cr-filter", false);
  filter_residual_enabled_tfs_ = opts.GetFirstBoolean('-', "cr-filter-tf", false);
  filter_residual_max_value_   = opts.GetFirstDouble ('-', "cr-filter-max-value", 0.08);

  filter_quality_enabled_      = opts.GetFirstBoolean('-', "qual-filter", false);
  filter_quality_offset_       = opts.GetFirstDouble ('-', "qual-filter-offset",0.7);
  filter_quality_slope_        = opts.GetFirstDouble ('-', "qual-filter-slope",0.02);
  filter_quality_quadr_        = opts.GetFirstDouble ('-', "qual-filter-quadr",0.00);


  // Adapter trimming options
  trim_adapter_                = opts.GetFirstStringVector ('-', "trim-adapter", "ATCACCGACTGCCCATAGAGAGGCTGAGAC");
  trim_adapter_cutoff_         = opts.GetFirstDouble       ('-', "trim-adapter-cutoff", 16.0);
  trim_adapter_separation_     = opts.GetFirstDouble       ('-', "trim-adapter-separation", 0.0);
  trim_adapter_min_match_      = opts.GetFirstInt          ('-', "trim-adapter-min-match", 6);
  trim_adapter_mode_           = opts.GetFirstInt          ('-', "trim-adapter-mode", 1);
  trim_adapter_tf_             = opts.GetFirstStringVector ('-', "trim-adapter-tf", "");

  // Quality trimming options
  trim_qual_mode_              = opts.GetFirstString       ('-', "trim-qual-mode", "sliding-window");
  trim_qual_window_size_       = opts.GetFirstInt          ('-', "trim-qual-window-size", 30);
  trim_qual_cutoff_            = opts.GetFirstDouble       ('-', "trim-qual-cutoff", 15.0);
  trim_qual_offset_            = opts.GetFirstDouble       ('-', "trim-qual-offset",0.7);
  trim_qual_slope_             = opts.GetFirstDouble       ('-', "trim-qual-slope",0.005);
  trim_qual_quadr_             = opts.GetFirstDouble       ('-', "trim-qual-quadr",0.00);
  extra_trim_right_            = opts.GetFirstInt          ('-', "extra-trim-right", 0);

  if      (trim_qual_mode_ == "off")             { trim_qual_mode_enum_ = kQvTrimOff; }
  else if (trim_qual_mode_ == "sliding-window")  { trim_qual_mode_enum_ = kQvTrimWindowed; }
  else if (trim_qual_mode_ == "expected-errors") { trim_qual_mode_enum_ = kQvTrimExpectedErrors; }
  else if (trim_qual_mode_ == "all")             { trim_qual_mode_enum_ = kQvTrimAll; }
  else {
   cerr << "BaseCaller Option Error: Unrecognized quality trimming mode: "<< trim_qual_mode_ <<". Aborting!" << endl;
   exit(EXIT_FAILURE);
  }

  // Turn adapter trimming off if 'off' is specified in options string
  if (trim_adapter_.size() > 0 and trim_adapter_.at(0) == "off")
    trim_adapter_.clear();
  if (trim_adapter_tf_.size() > 0 and trim_adapter_tf_.at(0) == "off")
    trim_adapter_tf_.clear();
  // Validate adapter strings so that they contains only ACGT characters.
  ValidateBaseStringVector(trim_adapter_);
  ValidateBaseStringVector(trim_adapter_tf_);
  if (trim_adapter_.size() > 0)
    WriteAdaptersToBamComments(bam_comments, run_id);

  string filter_beverly_args      = opts.GetFirstString ('-', "beverly-filter", "off");
  bool disable_all_filters        = opts.GetFirstBoolean('d', "disable-all-filters", false);

  // If this flag is set all filters & trimmers will be disabled
  if (disable_all_filters) {
    filter_keypass_enabled_ = false;
    filter_clonal_enabled_tfs_ = false;
    filter_clonal_enabled_ = false;
    filter_residual_enabled_ = false;
    filter_residual_enabled_tfs_ = false;
    filter_quality_enabled_ = false;
    trim_qual_mode_enum_ = kQvTrimOff;
    trim_adapter_cutoff_ = 0.0; // Zero means disabled for now
    filter_beverly_args = "off";
  }

  avalanche_mid_pos_           = opts.GetFirstInt    ('-', "avalanche_start_pos", -1);
  avalanche_min_pos_           = opts.GetFirstInt    ('-', "avalanche_stop_pos", -1);
  avalanche_max_pos_           = opts.GetFirstInt    ('-', "avalanche_max_pos", -1);
  trim_qual_avalanche_max_     = opts.GetFirstDouble ('-', "avalanche_qual_max", 25.0);
  trim_qual_avalanche_hi_      = opts.GetFirstDouble ('-', "avalanche_qual_hi", 15.0);
  trim_qual_avalanche_lo_      = opts.GetFirstDouble ('-', "avalanche_qual_lo", 1.0);
  filter_avalanche_enabled_ = avalanche_mid_pos_ >= 0 ? true : false;

  if (filter_avalanche_enabled_) { // force other filters off     
      TrimAvalanche_setup(flow_order.num_flows());

      filter_keypass_enabled_ = false;
      filter_clonal_enabled_tfs_ = false;
      filter_clonal_enabled_ = false;
      filter_residual_enabled_ = false;
      filter_residual_enabled_tfs_ = false;
      filter_quality_enabled_ = false;
      trim_qual_mode_enum_ = kQvTrimOff;
      trim_adapter_cutoff_ = 0.0; // Zero means disabled for now
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
  printf("        --beverly-filter %s\n", filter_beverly_args.c_str());
  printf("           --qual-filter %s\n", filter_quality_enabled_ ? "on" : "off");
  printf("    --qual-filter-offset %1.4f\n", filter_quality_offset_);
  printf("     --qual-filter-slope %1.4f\n", filter_quality_slope_ );


  printf("Adapter trimming settings\n");
  printf("          --trim-adapter ");
  if (!trim_adapter_.empty()) {
    for (unsigned int adpt_idx=0; adpt_idx<trim_adapter_.size()-1; adpt_idx++)
      cout << trim_adapter_.at(adpt_idx) << ",";
    cout << trim_adapter_.at(trim_adapter_.size()-1) << endl;
  }
  else {
    cout << "off" << endl;
  }
  //printf("       --trim-adapter-tf "); //%s\n", trim_adapter_tf_.empty() ? "off" : trim_adapter_tf_.c_str());
  printf("       --trim-adapter-tf ");
  if (!trim_adapter_tf_.empty()) {
    for (unsigned int adpt_idx=0; adpt_idx<trim_adapter_tf_.size()-1; adpt_idx++)
      cout << trim_adapter_tf_.at(adpt_idx) << ",";
    cout << trim_adapter_tf_.at(trim_adapter_tf_.size()-1) << endl;
  }
  else {
    cout << "off" << endl;
  }
  printf("     --trim-adapter-mode %d\n", trim_adapter_mode_);
  printf("   --trim-adapter-cutoff %1.1f (0.0 means disabled)\n", trim_adapter_cutoff_);
  printf("--trim-adapter-min-match %d\n", trim_adapter_min_match_);
  printf("        --trim-qual-mode %s\n", trim_qual_mode_.c_str());
  printf("Quality trimming settings:\n");
  printf(" --trim-qual-window-size %d\n", trim_qual_window_size_);
  printf("      --trim-qual-cutoff %1.1f (100.0 means disabled)\n", trim_qual_cutoff_);
  printf("      --trim-qual-offset %1.4f\n", trim_qual_offset_);
  printf("       --trim-qual-slope %1.4f\n", trim_qual_slope_);
  printf("     --trim-min-read-len %d\n", trim_min_read_len_);


  if (filter_avalanche_enabled_) { // force other filters off
	  printf("Avalanche filter/trimming settings:\n");
      printf("         --avalanche_max %d\n", avalanche_max_pos_);
      printf("       --avalanche_start %d\n", avalanche_mid_pos_);
      printf("        --avalanche_stop %d\n", avalanche_min_pos_);
      printf("    --avalanche_qual_max %g\n", trim_qual_avalanche_max_);
      printf("     --avalanche_qual_hi %g\n", trim_qual_avalanche_hi_);
      printf("     --avalanche_qual_lo %g\n", trim_qual_avalanche_lo_);
  }
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
	opts.StringToDoubleVector(filter_beverly_filter_trim_ratio_, filter_beverly_args, "beverly-filter");
    if (filter_beverly_filter_trim_ratio_.size() !=2){
      cerr << "BaseCaller Option Error: beverly-filter needs to be of the format <float>,<float>" << endl;
      exit (EXIT_FAILURE);
    }
    filter_beverly_enabled_ = true;
  }
}

// ----------------------------------------------------------------------------

void BaseCallerFilters::TrainClonalFilter(const string& output_directory, RawWells& wells, Mask& mask, const PolyclonalFilterOpts & opts)
{
  if (!filter_clonal_enabled_ and !filter_clonal_enabled_tfs_)
    return;

  wells.OpenForIncrementalRead();
  vector<int> key_ionogram(keys_[0].flows(), keys_[0].flows()+keys_[0].flows_length());
  filter_counts counts;
  int nlib = mask.GetCount(static_cast<MaskType> (MaskLib));
  counts._nsamp = min(nlib, filter_clonal_maxreads_);
  make_filter(clonal_population_, counts, mask, wells, key_ionogram, opts);
  cout << counts << endl;
  wells.Close();
}

// ----------------------------------------------------------------------------

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
      case kFilteredAvalancheTrim:      mask[idx] |= MaskFilteredShort; break;
    }
  }
}

// ----------------------------------------------------------------------------

int BaseCallerFilters::NumWellsCalled() const
{
  int num_called = 0;

  for (size_t idx = 0; idx < filter_mask_.size(); idx++)
    if (filter_mask_[idx] != kUninitialized)
      num_called++;

  return num_called;
}

// ----------------------------------------------------------------------------


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


// ----------------------------------------------------------------------------

void BaseCallerFilters::SetFiltered(int read_index, int read_class, ReadFilteringHistory& filter_history)
{
  if (filter_history.is_filtered)
    return;

  filter_mask_[read_index] = kFilterShortRead;
  filter_history.n_bases_filtered = 0;
  filter_history.n_bases_after_too_short = 0;
  filter_history.is_filtered = true;
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


// ----------------------------------------------------------------------------

void BaseCallerFilters::FilterHighPPFAndPolyclonal (int read_index, int read_class, ReadFilteringHistory& filter_history,
    const vector<float>& measurements,
    const PolyclonalFilterOpts & opts)
{
  if (filter_history.is_filtered)
    return;

  if (read_class == 0 and !filter_clonal_enabled_)  // Filter disabled for library?
    return;
  if (read_class != 0 and !filter_clonal_enabled_tfs_)  // Filter disabled for TFs?
    return;

  vector<float>::const_iterator first = measurements.begin() + opts.mixed_first_flow;
  vector<float>::const_iterator last  = measurements.begin() + opts.mixed_last_flow;
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


// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------


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


// ----------------------------------------------------------------------------

void BaseCallerFilters::FilterFailedKeypass(int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<char>& sequence)
{
  if (filter_history.is_filtered)
    return;

  if(!filter_keypass_enabled_)  // Filter disabled?
    return;

  bool failed_keypass = true;

  if ((int)sequence.size() >= keys_[read_class].bases_length()) {
    int base=0;
    while (base < keys_[read_class].bases_length() and sequence[base] == keys_[read_class].bases()[base])
      ++base;

    if (base == keys_[read_class].bases_length())
      failed_keypass = false;
  }

  if (failed_keypass) {
    filter_mask_[read_index] = kFilterFailedKeypass;
    filter_history.n_bases_filtered = 0;
    filter_history.n_bases_after_bad_key = 0;
    filter_history.is_filtered = true;
  }
}


// ----------------------------------------------------------------------------

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


// ----------------------------------------------------------------------------

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

      if (num_extreme_onemers <= num_onemers * filter_beverly_filter_trim_ratio_.at(1))
        max_trim_bases = base;
    }

    if ((num_extreme_onemers + num_extreme_twomers) > (num_onemers + num_twomers) * filter_beverly_filter_trim_ratio_.at(0)) {

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

// ----------------------------------------------------------------------------
// Filter entire reads based on their quality scores XXX

void BaseCallerFilters::FilterQuality(int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<uint8_t>& quality)
{
  if (filter_history.is_filtered or not filter_quality_enabled_)
    return;

  //for every base, compute cumulative expected errors and compare to threshold
  bool is_filtered = false; // assume all reads are good until proven otherwise
  double error_threshold = filter_quality_offset_;
  double cumulative_expected_error = 0.0;

  for (int ibase=0; ibase<filter_history.n_bases; ibase++){
     error_threshold += filter_quality_slope_;
     error_threshold += filter_quality_quadr_ * ibase; // extra quadratic to handle rise at end of read
     cumulative_expected_error += exp(-quality[ibase]*0.2302585); // log(10)/10

     // filter-v1.0 - trigger filter any time expected errors are above the allowed threshold
     if (cumulative_expected_error > error_threshold) {
       is_filtered = true;
       break; // No need to waste more time!
     }
  }

  // filter-v2.0 - trigger filter only if the expected errors at the end of the read are above the allowed threshold
  //if (cumulative_expected_error > error_threshold)
  //  is_filtered = true;

  if(is_filtered) {
    filter_mask_[read_index] = kFilterHighResidual;
    filter_history.n_bases_filtered = 0;
    filter_history.n_bases_after_high_residual = 0;
    filter_history.is_filtered = true;
  }

};

// ----------------------------------------------------------------------------


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

// ----------------------------------------------------------------------------

void BaseCallerFilters::ValidateBaseStringVector(vector<string>& string_vector) {

  unsigned int num_fails = 0;

  for (unsigned int string_idx=0; string_idx<string_vector.size(); string_idx++) {

	bool is_ok = true;

    if (string_vector.at(string_idx).empty()) {
      cerr << "WARNING in BaseCallerFilters: Supplied adapter sequence " << string_idx
           << " is an empty string. Removing it from classification." << endl;
      num_fails++;
      continue;
    }

    // Loop through string to make sure it's upper case and a valid base
    unsigned int pos_idx=0;
    do {
      switch (string_vector.at(string_idx).at(pos_idx)) {
        case 'A':
        case 'C':
        case 'G':
        case 'T': break;
        case 'a':
          string_vector.at(string_idx).at(pos_idx) = 'A'; break;
        case 'c':
          string_vector.at(string_idx).at(pos_idx) = 'C'; break;
        case 'g':
          string_vector.at(string_idx).at(pos_idx) = 'G'; break;
        case 't':
          string_vector.at(string_idx).at(pos_idx) = 'T'; break;
        default:
          is_ok = false;
      }
      pos_idx++;
    } while (is_ok and pos_idx < string_vector.at(string_idx).length());

    if (not is_ok) {
      // verbose failure and erase this adapter
      cerr << "WARNING in BaseCallerFilters: Supplied adapter sequence " << string_idx << ": " << string_vector.at(string_idx)
           << " contains non-ACGT characters. Removing it from classification." << endl;
      string_vector.at(string_idx).clear();
      num_fails++;
    }

    // No point doing adapter classification if all sequences failed
    if (num_fails == string_vector.size())
      string_vector.clear();
  }
}


// ----------------------------------------------------------------------------
// Adapter detection based on adapter predicted signal

bool BaseCallerFilters::TrimAdapter_PredSignal(float& best_metric, int& best_start_flow, int& best_start_base, int& best_adapter_overlap,
                         const string& effective_adapter, DPTreephaser& treephaser, const BasecallerRead& read) {

  bool adapter_found = false;
  // Initialize metrics for adapter search
  best_metric = -0.1; // Inverted to negative value: The larger the better.
  best_start_flow = -1;
  best_start_base = -1;
  best_adapter_overlap = -1;

  DPTreephaser::TreephaserPath& called_path = treephaser.path(0);   //simulates the main sequence
  DPTreephaser::TreephaserPath& adapter_path = treephaser.path(1);  //branches off to simulate adapter
  treephaser.InitializeState(&called_path);

  for (int adapter_start_base = 0; adapter_start_base < (int)read.sequence.size(); ++adapter_start_base) {

    // Step 1. Consider current position as hypothetical adapter start

    adapter_path.prediction = called_path.prediction;
    int window_start = max(0,called_path.window_start - 8);
    treephaser.AdvanceState(&adapter_path,&called_path, effective_adapter.at(0), flow_order_.num_flows());
    int inphase_flow = called_path.flow;
    float state_inphase = called_path.state[inphase_flow];

    int adapter_bases = 0;
    for (int adapter_pos = 1; adapter_pos < (int)effective_adapter.length(); ++adapter_pos) {
      treephaser.AdvanceStateInPlace(&adapter_path, effective_adapter.at(adapter_pos), flow_order_.num_flows());
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

    // Changing metric sign to negative so that both algorithms maximize the metric
    float adapter_score = -(metric_num/metric_den + 0.2/adapter_bases);
    if (adapter_score > best_metric) {
      adapter_found = true;
      best_metric = adapter_score;
      best_start_flow = inphase_flow;
      best_start_base = adapter_start_base;
    }

    // Step 2. Continue to next position
    treephaser.AdvanceStateInPlace(&called_path, read.sequence[adapter_start_base], flow_order_.num_flows());
  }
  return adapter_found;
}


// ----------------------------------------------------------------------------
// Adapter detection using a flow space sequence alignment

bool BaseCallerFilters::TrimAdapter_FlowAlign(float& best_metric, int& best_start_flow,
		                    int& best_start_base, int& best_adapter_overlap,
		                    const string& effective_adapter, const vector<float>& scaled_residual,
		                    const vector<int>& base_to_flow, const BasecallerRead& read) {

  bool adapter_found = false;
  // Initialize metrics for adapter search
  best_metric = -1e10; // The larger the better
  best_start_flow = -1;
  best_start_base = -1;
  best_adapter_overlap= -1;
  int sequence_pos = 0;

  for (int adapter_start_flow = 0; adapter_start_flow < flow_order_.num_flows(); ++adapter_start_flow) {

    while (sequence_pos < (int)read.sequence.size() and base_to_flow[sequence_pos] < adapter_start_flow)
      sequence_pos++;
    if (sequence_pos >= (int)read.sequence.size())
      break;
    // Only consider start flows that agree with adapter start
    if (effective_adapter.at(0) != flow_order_[adapter_start_flow])
      continue;

    // Evaluate this starting position
    int adapter_pos = 0;
    float score_match = 0;
    int score_len_flows = 0;
    int local_sequence_pos = sequence_pos;
    int local_start_base = sequence_pos;

    for (int flow = adapter_start_flow; flow < flow_order_.num_flows(); ++flow) {

      int base_delta = 0;
      while (adapter_pos < (int)effective_adapter.length() and effective_adapter.at(adapter_pos) == flow_order_[flow]) {
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

      if (adapter_pos == (int)effective_adapter.length() or local_sequence_pos == (int)read.sequence.size())
        break;
    }

    score_match /= score_len_flows;

    // Does this adapter alignment match our minimum acceptance criteria? If yes, is it better than other matches seen so far?

    if (adapter_pos < trim_adapter_min_match_)  // Match too short
      continue;
    if (score_match * 2 * effective_adapter.length() > trim_adapter_cutoff_)  // Match too dissimilar
      continue;
    float final_metric = adapter_pos / (float)effective_adapter.length() - score_match; // The higher the better

    if (final_metric > best_metric) {
      adapter_found = true;
      best_metric = final_metric;
      best_start_flow = adapter_start_flow;
      best_start_base = local_start_base;
      best_adapter_overlap = adapter_pos;
    }
  }
  return adapter_found;
}


// ----------------------------------------------------------------------------

void BaseCallerFilters::TrimAdapter(int read_index, int read_class, ProcessedRead& processed_read, const vector<float>& scaled_residual,
    const vector<int>& base_to_flow, DPTreephaser& treephaser, const BasecallerRead& read)
{
  if(trim_adapter_cutoff_ <= 0.0 or trim_adapter_.empty())  // Zero cutoff means disabled
    return;
  if (read_class != 0 and trim_adapter_tf_.empty())  // TFs only trimmed if explicitly enabled
    return;

  int   best_adapter = -1;
  int   temp_start_flow,      best_start_flow = -1;
  int   temp_start_base,      best_start_base = -1;
  int   temp_adapter_overlap, best_adapter_overlap = -1;
  float best_metric = (trim_adapter_mode_ == 2) ? -0.1 : -1e10;
  float temp_metric, second_best_metric = best_metric;

  const vector<string>& effective_adapter = (read_class == 0) ? trim_adapter_ : trim_adapter_tf_;

  // Loop over adapter possible sequences and evaluate positions for each.
  for (unsigned int adapter_idx=0; adapter_idx<effective_adapter.size(); adapter_idx++) {

	if (effective_adapter.at(adapter_idx).empty())
	  continue;

	bool adapter_found = false;
    if (trim_adapter_mode_ == 2) {
      adapter_found = TrimAdapter_PredSignal(temp_metric, temp_start_flow, temp_start_base, temp_adapter_overlap,
                                         effective_adapter.at(adapter_idx), treephaser, read);
    } else {
      adapter_found = TrimAdapter_FlowAlign(temp_metric, temp_start_flow, temp_start_base, temp_adapter_overlap,
                                         effective_adapter.at(adapter_idx), scaled_residual, base_to_flow, read);
    }
    // Find & record best matching adapter sequence
    if (adapter_found) {
      if (temp_metric > best_metric) {
        best_adapter = adapter_idx;
        second_best_metric = best_metric;
        best_metric = temp_metric;
        best_start_flow = temp_start_flow;
        best_start_base = temp_start_base;
        best_adapter_overlap = temp_adapter_overlap;
      }
      else if (temp_metric > second_best_metric) {
        second_best_metric = temp_metric;
      }
    }
  }

  // Do not use adapter classification if we're not confident
  if (fabs(best_metric - second_best_metric) < trim_adapter_separation_)
    best_start_flow = -1;
  if (best_start_flow == -1) {   // No suitable match
    processed_read.filter.adapter_type = trim_adapter_.size();
    return;
  }

  // --- Save trimming results
  processed_read.filter.adapter_type  = best_adapter;
  processed_read.filter.adapter_score = best_metric;
  if (second_best_metric != ((trim_adapter_mode_ == 2) ? -0.1 : -1e10)) {
    processed_read.filter.adapter_decision   = true;
    processed_read.filter.adapter_separation = (best_metric - second_best_metric);
  }

  processed_read.bam.AddTag("ZA", "i", max(best_start_base - processed_read.filter.n_bases_prefix, 0));
  processed_read.bam.AddTag("ZG", "i", best_start_flow);
  processed_read.bam.AddTag("ZB", "i", best_adapter_overlap);

  vector<int> pcr_duplicate_signature(4,0);
  pcr_duplicate_signature[0] = best_start_flow;                       // Signature entry 1 - flow incorporating the first adapter base.
  if (best_start_base > 0)
    pcr_duplicate_signature[1] = base_to_flow[best_start_base-1];     // Signature entry 2 - flow containing last insert base.
  for (int reverse_pos = best_start_base-1; reverse_pos >= 0 and base_to_flow[best_start_base-1] == base_to_flow[reverse_pos]; --reverse_pos)
    pcr_duplicate_signature[2]++;                                     // Signature entry 3 - length of last insert HP.
  pcr_duplicate_signature[3] = best_adapter;                          // Signature entry 4 - type of adapter sequence found.

  processed_read.bam.AddTag("ZC", pcr_duplicate_signature);

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


// ----------------------------------------------------------------------------

void BaseCallerFilters::TrimQuality(int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<uint8_t>& quality)
{
  // Exception were we don't filter
  if (filter_history.is_filtered) // Already filtered out?
	return;
  if (read_class != 0)  // Hardcoded: Don't trim TFs
    return;

  int temp_clip_qual_right, clip_qual_right;

  switch (trim_qual_mode_enum_) {
    case kQvTrimOff : return;
    case kQvTrimWindowed :
      clip_qual_right = TrimQuality_Windowed(read_index, filter_history, quality);
      break;
    case kQvTrimExpectedErrors :
      clip_qual_right = TrimQuality_ExpectedErrors(read_index, filter_history, quality);
      break;
    default :
      temp_clip_qual_right = TrimQuality_Windowed(read_index, filter_history, quality);
      clip_qual_right = min(temp_clip_qual_right, TrimQuality_ExpectedErrors(read_index, filter_history, quality));
  };

  // Store trimming results if necessary and do base accounting
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


// ----------------------------------------------------------------------------

int BaseCallerFilters::TrimQuality_Windowed(int read_index, ReadFilteringHistory& filter_history, const vector<uint8_t>& quality)
{
  if(trim_qual_cutoff_ >= 100.0 or trim_qual_cutoff_ == 0.0)   // 100.0 or more means disabled
    return filter_history.n_bases;

  int window_start = 0;
  int window_end = 0;
  int minimum_sum, window_sum = 0;

  // Step 1: Accumulate over the initial window
  while (window_end < trim_qual_window_size_ and window_end < filter_history.n_bases)
    window_sum += quality[window_end++];
  minimum_sum = window_end * trim_qual_cutoff_;

  // Step 2: Keep sliding as long as average q-score exceeds the threshold
  int clip_qual_right = window_sum >= minimum_sum ? (window_end + window_start) / 2 : 0;
  while (window_sum >= minimum_sum and window_end < filter_history.n_bases) {
    window_sum += quality[window_end++];
    window_sum -= quality[window_start++];
    clip_qual_right++;
  }

  // We reached the end of the read - nothing to trim here, besides if extra trim is explicitly set.
  if (window_end == filter_history.n_bases)
    clip_qual_right = filter_history.n_bases - extra_trim_right_;

  return clip_qual_right;
}


// ----------------------------------------------------------------------------

int BaseCallerFilters::TrimQuality_ExpectedErrors(int read_index, ReadFilteringHistory& filter_history, const vector<uint8_t>& quality)
{
  //for every base, compute cumulative expected errors and compare to threshold
  int clip_qual_right = 0; // assume all bad until proven otherwise
  double error_threshold = trim_qual_offset_;
  double cumulative_expected_error = 0.0;

  for (int ibase=0; ibase<filter_history.n_bases; ibase++){
     error_threshold += trim_qual_slope_;
     error_threshold += trim_qual_quadr_ * ibase;  // increase to handle rising error rate towards end of read
     cumulative_expected_error += exp(-quality[ibase]*0.2302585); // log(10)/10

     if (cumulative_expected_error < error_threshold)  // acceptable cumulative error at this base
       clip_qual_right = ibase;
  }
  // at this point, we have the last acceptable base where the expected errors up to that base meet threshold
  clip_qual_right +=1; // clip at one base beyond?

  // keep the dynamics of the old school clipper: To be removed in the future!
  if (clip_qual_right == filter_history.n_bases)
    clip_qual_right = filter_history.n_bases - extra_trim_right_;

  return clip_qual_right;
}

// ----------------------------------------------------------------------------

void BaseCallerFilters::TrimAvalanche_setup(int maxflows)
{
    if (!filter_avalanche_enabled_) // not enabled?
        return;

    static int flows[] = {360,520,800,1000,1200,1600};
    static int stops[] = {125,200,280,330,365,420};
    static int starts[] = {100,170,230,275,300,345};
    //static int starts[] = {25,45,85,145,170,200,240};
    //static int starts[] = {10,15,40,110,130,160,210};
    //static int starts[] = {10,10,10,15,15,15,20}; // P1.2??? So different from P1???
    int nItems = sizeof(stops) / sizeof(stops[0]);
    int nItemsF = sizeof(flows) / sizeof(flows[0]);
    int nItemsS = sizeof(starts) / sizeof(starts[0]);
    assert(nItems==nItemsF);
    assert(nItems==nItemsS);

    if (trim_qual_avalanche_lo_ < 0) {
        printf("trim_qual_avalanche_lo_(%f) < 0: resetting trim_qual_avalanche_lo_ to 0\n",trim_qual_avalanche_lo_);
        trim_qual_avalanche_lo_ = 0;
    }
    if (trim_qual_avalanche_hi_ < trim_qual_avalanche_lo_) {
      printf("trim_qual_avalanche_hi_(%f) < trim_qual_avalanche_lo_(%f): resetting trim_qual_avalanche_hi_ to %f\n",trim_qual_avalanche_hi_,trim_qual_avalanche_lo_,trim_qual_avalanche_lo_+10);
      trim_qual_avalanche_hi_ = trim_qual_avalanche_lo_ + 15;
    }
    if (trim_qual_avalanche_max_ < trim_qual_avalanche_hi_) {
      printf("trim_qual_avalanche_max_(%f) < trim_qual_avalanche_hi_(%f): resetting trim_qual_avalanche_max_ to %f\n",trim_qual_avalanche_max_,trim_qual_avalanche_hi_,trim_qual_avalanche_hi_+15);
      trim_qual_avalanche_max_ = trim_qual_avalanche_hi_ + 15;
    }

    // automatically reset avalanche_mid_pos_ according to maxflows
    if (avalanche_mid_pos_ <= 0) {
        avalanche_mid_pos_ = starts[nItems-1];
        for (int i=0; i<nItems; i++)
            if (maxflows <= flows[i]) {
                avalanche_mid_pos_ = starts[i];
                //cout << "TrimAvalanche_setup... maxflows=" << maxflows << " -> avalanche_mid_pos_=" << avalanche_mid_pos_ << endl << flush;
                break;
            }
    }
    else {
        //assert (avalanche_mid_pos_> trim_qual_window_size_) ;
        if (avalanche_mid_pos_<= trim_qual_window_size_) {
            cerr << "TrimAvalanche_setup... avalanche_mid_pos_(" << avalanche_mid_pos_ << ") <= trim_qual_window_size_(" << trim_qual_window_size_ << ")..." << endl;
            avalanche_mid_pos_ = starts[nItems-1];
            for (int i=0; i<nItems; i++)
                if (maxflows <= flows[i]) {
                    avalanche_mid_pos_ = starts[i];
                    cerr << "TrimAvalanche_setup... maxflows=" << maxflows << " -> avalanche_mid_pos_=" << avalanche_mid_pos_ << endl;
                    break;
                }
            cerr << "reset avalanche_mid_pos_ = " << avalanche_mid_pos_ << endl << flush;
        }
    }
    avalanche_win_ = 30;
    if (avalanche_max_pos_ < 0) {
        avalanche_max_pos_ = avalanche_mid_pos_ - avalanche_win_;
    }

    int stop_pos = stops[nItems-1];
    if (avalanche_min_pos_ <= 0 && avalanche_mid_pos_ > 0)
        avalanche_min_pos_ = avalanche_mid_pos_ + avalanche_win_;
    if (avalanche_min_pos_ > 0 && avalanche_min_pos_ <= avalanche_mid_pos_) {
        for (int i=0; i<nItems; i++)
            if (maxflows <= flows[i]) {
                stop_pos = stops[i];
                if (stop_pos > avalanche_mid_pos_)
                    break;
            }
        if (stop_pos > avalanche_mid_pos_) {
            avalanche_min_pos_ = stop_pos;
            avalanche_win_ = avalanche_min_pos_ - avalanche_mid_pos_;
        }
        else
            avalanche_min_pos_ = avalanche_mid_pos_ + avalanche_win_;
    }
    // trim_min_read_len_ used to decide whether reads are to be filtered, default 8 is too short for Avalanche
    //trim_min_read_len_avalanch_ = trim_min_read_len_;
    //trim_min_read_len_avalanch_ = trim_qual_window_size_;
    //trim_min_read_len_avalanch_ = max(trim_qual_window_size_,avalanche_mid_pos_-trim_qual_window_size_);
    trim_min_read_len_avalanch_ = trim_min_read_len_;
    /*
    */
    cout << "trim_min_read_len_avalanch_ = " << trim_min_read_len_avalanch_ << endl;
    cout << "avalanche_win_ = " << avalanche_win_ << endl;
    cout << "trim_qual_avalanche_max_ = " << trim_qual_avalanche_max_ << endl;
    cout << "trim_qual_avalanche_hi_ = " << trim_qual_avalanche_hi_ << endl;
    cout << "trim_qual_avalanche_lo_ = " << trim_qual_avalanche_lo_ << endl;
    cout << "avalanche_max_pos_ = " << avalanche_max_pos_ << endl;
    cout << "avalanche_mid_pos_ = " << avalanche_mid_pos_ << endl;
    cout << "avalanche_min_pos_ = " << avalanche_min_pos_ << endl;
    assert(avalanche_max_pos_ >= trim_qual_window_size_);
    assert(avalanche_max_pos_ <= avalanche_mid_pos_);
    assert(avalanche_mid_pos_ <= avalanche_min_pos_);

    if (avalanche_mid_pos_>avalanche_max_pos_)
        avalanche_sum_delta_ = trim_qual_window_size_ * (trim_qual_avalanche_max_ - trim_qual_avalanche_hi_) / double(avalanche_mid_pos_ - avalanche_max_pos_);
    else
        avalanche_sum_delta_ = 0;
    if (avalanche_min_pos_ > avalanche_mid_pos_)
        avalanche_ava_delta_ = trim_qual_window_size_ * (trim_qual_avalanche_hi_ - trim_qual_avalanche_lo_) / double(avalanche_min_pos_ - avalanche_mid_pos_);
    else
        avalanche_ava_delta_ = 0;
    avalanche_min_delta_ = 0;
    delta_ava_ = avalanche_ava_delta_ - avalanche_sum_delta_;
    delta_min_ = avalanche_min_delta_ - avalanche_ava_delta_;

    // setup qv_sum_thresh
    qv_sum_thresh.resize(avalanche_min_pos_+trim_qual_window_size_);
    double minimum_sum = trim_qual_window_size_ * trim_qual_avalanche_max_;
    for (int i=0; i<avalanche_max_pos_+trim_qual_window_size_; i++)
        qv_sum_thresh[i] = minimum_sum;
    for (int i=avalanche_max_pos_+trim_qual_window_size_; i<avalanche_mid_pos_+trim_qual_window_size_; i++) {
        qv_sum_thresh[i] = minimum_sum;
        minimum_sum -= avalanche_sum_delta_;
    }
    for (int i=avalanche_mid_pos_+trim_qual_window_size_; i<avalanche_min_pos_+trim_qual_window_size_; i++) {
        qv_sum_thresh[i] = minimum_sum;
        minimum_sum -= avalanche_ava_delta_;
    }
    qv_sum_thresh_max = trim_qual_window_size_ * trim_qual_avalanche_max_;
    qv_sum_thresh_min = trim_qual_window_size_ * trim_qual_avalanche_lo_;
    min_bases_passed_test = 5;
    //count_filtered = 0;
    //count_trimmed = 0;
    //count_passed = 0;
}

// ----------------------------------------------------------------------------

void BaseCallerFilters::TrimAvalanche(int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<uint8_t>& quality)
{
  if (!filter_avalanche_enabled_) // not enabled?
      return;

  if (filter_history.is_filtered) // Already processed
    return;

  //if (read_class != 0)  // Hardcoded: Don't trim TFs
  //  return;

  int num_bases_passed_test = 0;
  int window_start = 0;
  int window_end = min(filter_history.n_bases,trim_qual_window_size_);
  //int clip_qual_right = (window_end + window_start) / 2;
  int clip_qual_right = 0;
  int window_sum = 0;

  // for long reads, check according to qv_sum_thresh
  if (filter_history.n_bases >= avalanche_max_pos_) {
      if (filter_history.n_bases < avalanche_mid_pos_)
          window_end =  avalanche_max_pos_;
      else if (filter_history.n_bases < avalanche_min_pos_)
          window_end = avalanche_mid_pos_;
      else
          window_end =  avalanche_min_pos_;

      window_end -= min_bases_passed_test;
      window_start = window_end - trim_qual_window_size_;
      assert (window_start>=0);
      clip_qual_right = (window_end + window_start) / 2;
      for (int w = window_start; w< window_end; w++)
        window_sum += quality[w];

      while (window_end < filter_history.n_bases) {
          if (window_end >= avalanche_min_pos_+trim_qual_window_size_) {
              if (window_sum < qv_sum_thresh_min) {
                  break;
              }
              else
                  num_bases_passed_test++;
          }
          else {
              if (window_sum < qv_sum_thresh[window_end]) {
                  break;
              }
              else
                  num_bases_passed_test++;
          }
          window_sum += quality[window_end++] - quality[window_start++];
          clip_qual_right++;
      }
  }
  else { // for short reads, compare to qv_sum_thresh_max
      window_end = min(filter_history.n_bases,trim_qual_window_size_);
      for (int w = window_start; w< window_end; w++)
        window_sum += quality[w];

      clip_qual_right = window_sum >= qv_sum_thresh_max ? (window_end + window_start) / 2 : 0;
      while (window_end < filter_history.n_bases) {
          if (window_sum < qv_sum_thresh_max)
              break;
          else
              num_bases_passed_test++;
          window_sum += quality[window_end++] - quality[window_start++];
          clip_qual_right++;
      }
  }

  if (clip_qual_right >= filter_history.n_bases_filtered) { // pass the filter
    return;
  }

  int trim_length = clip_qual_right - filter_history.n_bases_prefix;
  if (trim_length < trim_min_read_len_avalanch_  || num_bases_passed_test < min_bases_passed_test) { // Quality trimming led to filtering
    filter_mask_[read_index] = kFilteredAvalancheTrim;
    filter_history.n_bases_after_quality_trim = filter_history.n_bases_filtered = 0;
    filter_history.is_filtered = true;
    //cout << count_filtered++ << " filtered: RL="  << filter_history.n_bases << " < trim_length=" << trim_length << endl << flush;
  } else {
    filter_history.n_bases_after_quality_trim = filter_history.n_bases_filtered = clip_qual_right;
    //cout << count_trimmed++ << " trimmed: RL="  << filter_history.n_bases << " >= trim_length=" << trim_length << endl << flush;
    //if (trim_length < avalanche_mid_pos_)
    //    cout << "Error: trim_length(" << trim_length << ") < avalanche_mid_pos_(" <<  avalanche_mid_pos_ << ")" << endl << flush;
    //assert(trim_length >= avalanche_mid_pos_);
  }
}

// ----------------------------------------------------------------------------------

void BaseCallerFilters::WriteAdaptersToBamComments(vector<string> &comments, const string & run_id) {

  Json::Value json(Json::objectValue);
  stringstream adapter_stream;
  json["BeadAdapters"]["MasterKey"] = run_id;
  json["BeadAdapters"]["NumAdapters"] = (Json::UInt64)trim_adapter_.size();
  for (unsigned int adapter_idx=0; adapter_idx<trim_adapter_.size(); adapter_idx++) {
    adapter_stream.str(std::string());
    adapter_stream << "Adapter_" << adapter_idx;
    json["BeadAdapters"][adapter_stream.str()] = trim_adapter_.at(adapter_idx);
  }
  Json::FastWriter writer;
  string str = writer.write(json);
  // trim unwanted newline added by writer
  int last_char = str.size()-1;
  if (last_char>=0 and str[last_char]=='\n') {
    str.erase(last_char,1);
  }
  comments.push_back(str);
}


