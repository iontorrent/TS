/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     OrderedDatasetWriter.h
//! @ingroup  BaseCaller
//! @brief    OrderedDatasetWriter. Thread-safe, barcode-friendly BAM writer with deterministic order

#ifndef ORDEREDDATASETWRITER_H
#define ORDEREDDATASETWRITER_H

#include <string>
#include <vector>
#include <deque>
#include <pthread.h>
#include <sys/types.h>
#include <stdlib.h>

#include "api/SamHeader.h"
#include "api/BamAlignment.h"
#include "api/BamWriter.h"

#include "json/json.h"

#include "BaseCallerUtils.h"
class BarcodeDatasets;

using namespace std;
using namespace BamTools;



struct ReadFilteringHistory {
  ReadFilteringHistory();  // Set default values

  void GenerateZDVector(vector<int16_t>& zd_vector);  // Save filtering history to a vector

  // Basic information
  bool    is_filtered;                          //!< true if the read should not be saved
  bool    is_called;                            //!< true if the read was filtered before treephaser
  int     n_bases;                              //!< Number of bases called by treephaser

  // Right side (5') trimming account
  int     n_bases_key;
  int     n_bases_prefix;                       //!< Final 5' trim position (includes key, barcode, extra_trim)

  // Left side (3') trimming and filtering account
  int     n_bases_after_bkgmodel_high_ppf;
  int     n_bases_after_bkgmodel_polyclonal;
  int     n_bases_after_bkgmodel_bad_key;
  int     n_bases_after_polyclonal;
  int     n_bases_after_high_ppf;
  int     n_bases_after_too_short;
  int     n_bases_after_bad_key;
  int     n_bases_after_high_residual;
  int     n_bases_after_beverly_trim;
  int     n_bases_after_quality_trim;
  int     n_bases_after_adapter_trim;
  int     n_bases_filtered;                     //!< Final 3' trim position or zero if filtered

  // Information about (3') adapter classification
  int       adapter_type;
  double    adapter_score;
  double    adapter_separation;
  bool      adapter_decision;
};


class ReadFilteringStats {
public:
  ReadFilteringStats();

  void SetBeadAdapters(const vector<string> & trim_adapters);
  void AddRead(const ReadFilteringHistory& read_filtering_history);
  void MergeFrom(const ReadFilteringStats& other);
  void ComputeAverages();

  void PrettyPrint (const string& table_header);

  //void SaveToDatasetsJson();
  void SaveToBasecallerJson(Json::Value &json, const string& class_name,  bool library_report);

//protected:
  int64_t     num_bases_initial_;
  int64_t     num_bases_removed_key_trim_;
  int64_t     num_bases_removed_barcode_trim_;
  int64_t     num_bases_removed_short_;
  int64_t     num_bases_removed_keypass_;
  int64_t     num_bases_removed_residual_;
  int64_t     num_bases_removed_beverly_;
  int64_t     num_bases_removed_adapter_trim_;
  int64_t     num_bases_removed_quality_trim_;
  int64_t     num_bases_final_;

  int64_t     num_reads_initial_;
  int64_t     num_reads_removed_bkgmodel_keypass_;
  int64_t     num_reads_removed_bkgmodel_high_ppf_;
  int64_t     num_reads_removed_bkgmodel_polyclonal_;
  int64_t     num_reads_removed_high_ppf_;
  int64_t     num_reads_removed_polyclonal_;
  int64_t     num_reads_removed_short_;
  int64_t     num_reads_removed_keypass_;
  int64_t     num_reads_removed_residual_;
  int64_t     num_reads_removed_beverly_;
  int64_t     num_reads_removed_adapter_trim_;
  int64_t     num_reads_removed_quality_trim_;
  int64_t     num_reads_final_;

  // Accounting for adapter trimming
  vector<string>      bead_adapters_;                     //!< Adapter sequences
  vector<uint64_t>    adapter_class_num_reads_;           //!< Number of reads per library adapter
  vector<double>      adapter_class_cum_score_;           //!< Cumulative classification score for library reads
  vector<double>      adapter_class_av_score_;            //!< Average classification score for library reads
  vector<uint64_t>    adapter_class_num_decisions_;       //!< Number of reads where we decided between different adapters
  vector<double>      adapter_class_cum_separation_;      //!< Cumulative separation between adapter types
  vector<double>      adapter_class_av_separation_;       //!< Cumulative separation between adapter types
};


//! @brief    Populated BAM record with filtering history
//! @ingroup  BaseCaller

struct ProcessedRead {
  ProcessedRead(int default_read_group) {
    read_group_index        =  default_read_group;       // Needs to be a valid index at startup
    barcode_n_errors        =  0;
	barcode_filt_zero_error = -1;
	barcode_distance        = 0.0;
	is_control_barcode      = false;
  }

  // Variables storing barcode classification results
  int                   read_group_index;         //!< Read group index, generally based on barcode classification.
  int                   barcode_n_errors;         //!< Number of base mismatches in barcode sequence.
  int                   barcode_filt_zero_error;  //!< Inidcator whether a hard decision match was filtered in signal space.
  float                 barcode_distance;         //!< Distance to barcode in signal space.
  vector<float>         barcode_bias;             //!< A bias vector for the barcode found.
  bool                  is_control_barcode;       //!< Identified the read as having a control barcode

  //

  ReadFilteringHistory  filter;
  BamAlignment          bam;
};


//! @brief    Thread-safe writer class for SFF that guarantees deterministic read order
//! @ingroup  BaseCaller

class OrderedDatasetWriter {
public:
  //! Constructor.
  OrderedDatasetWriter();
  //! Destructor.
  ~OrderedDatasetWriter();

  //! @brief  Open SFF file for writing.
  //! @param  sff_filename    Filename of the SFF file to create.
  //! @param  num_regions     Number of regions to expect.
  //! @param  num_flows       Number of flows.
  //! @param  flow_order      Flow order object, also stores number of flows
  //! @param  key             Key sequence.
  void Open(const string& base_directory, BarcodeDatasets& datasets, int num_regions,
      const ion::FlowOrder& flow_order, const string& key, const vector<string> & bead_adapters,
      const string& basecaller_name, const string& basecalller_version, const string& basecaller_command_line,
      const string& production_date, const string& platform_unit, bool save_filtered_reads, vector<string>& comments);

  //! @brief  Drop off a region-worth of reads for writing. Write opportunistically.
  //! @param  region          Index of the region being dropped off.
  //! @param  region_reads    SFF entries from this region.
  void WriteRegion(int region, deque<ProcessedRead> &region_reads);

  //! Update SFF header and close.
  void Close(BarcodeDatasets& datasets, const string& dataset_nickname = string(""));

  void PrintStats();

  const vector<uint64_t>&  qv_histogram() const { return qv_histogram_; }

  void SaveFilteringStats(Json::Value &json, const string& class_name,  bool library_report) {
    combined_stats_.SaveToBasecallerJson(json, class_name,  library_report);
  }


private:

  void PhysicalWriteRegion(int iRegion);


  int                       num_datasets_;          //!< How many files are being generated
  int                       num_barcodes_;          //!< Just in case: how many barcodes are there
  vector<int>               read_group_dataset_;    //!< Which dataset should a given read group be saved to?
  vector<string>            read_group_name_;
  int                       num_read_groups_;

  int                       num_regions_;           //!< Total number of regions to expect
  int                       num_regions_written_;   //!< Number of regions physically written thus far
  vector<bool>              region_ready_;          //!< Which regions are ready for writing?
  vector<deque<ProcessedRead> >  region_dropbox_;   //!< Reads for regions that are ready for writing
  pthread_mutex_t           dropbox_mutex_;         //!< Mutex controlling access to the dropbox
  pthread_mutex_t           write_mutex_;           //!< Mutex controlling BAM writing
  pthread_mutex_t           delete_mutex_;          //!< Mutex controlling deallocation of processed dropbox regions

  vector<int>               num_reads_;             //!< Number of reads written, per dataset
  vector<string>            bam_filename_;

  bool                      save_filtered_reads_;

  vector<uint64_t>          read_group_num_Q20_bases_;         //!< Number of >=Q20 bases written per read group
  vector<uint64_t>          qv_histogram_;
  vector<vector<uint64_t> > read_group_num_barcode_errors_;    //!< Number of reads with N base errors in barcode
  vector<vector<uint64_t> > read_group_barcode_distance_hist_; //!< Distance histogram for barcodes
  vector<vector<double> >   read_group_barcode_bias_;          //!< Bias vector for barcodes
  vector<uint64_t>          read_group_barcode_filt_zero_err_; //!< Number of reads filtered that matched a barcode in base space.

  vector<BamWriter *>       bam_writer_;
  vector<SamHeader>         sam_header_;

  vector<ReadFilteringStats>  read_group_stats_;
  ReadFilteringStats        combined_stats_;

};



#endif // ORDEREDDATASETWRITER_H



