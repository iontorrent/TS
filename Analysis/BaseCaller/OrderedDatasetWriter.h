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
#include "MolecularTagTrimmer.h"

class  BarcodeDatasets;

using namespace std;
using namespace BamTools;



struct ReadFilteringHistory {
  ReadFilteringHistory();  // Set default values

  void GenerateZDVector(vector<int16_t>& zd_vector);  // Save filtering history to a vector
  void CalledRead(int num_bases) { is_called=true; n_bases=num_bases; };

  // Basic information
  bool    is_filtered;                          //!< true if the read should not be saved
  bool    is_called;                            //!< true if the read was filtered before treephaser
  int     n_bases;                              //!< Number of bases called by treephaser

  // Right side (5') trimming account
  int     n_bases_key;                          //!< Number of key bases
  int     n_bases_barcode;                      //!< Number of bases to end of barcode adapter
  int     n_bases_tag;                          //!< Number of bases to the end of the 5' tag
  int     n_bases_prefix;                       //!< Final 5' trim position (includes key, barcode, tag, extra_trim)

  // Left side (3') trimming and filtering account
  int     n_bases_after_bkgmodel_high_ppf;
  int     n_bases_after_bkgmodel_polyclonal;
  int     n_bases_after_bkgmodel_bad_key;
  int     n_bases_after_polyclonal;
  int     n_bases_after_high_ppf;
  int     n_bases_after_too_short;
  int     n_bases_after_bad_key;
  int     n_bases_after_high_residual;
  int     n_bases_after_quality_filter;
  //int     n_bases_after_beverly_trim;
  int     n_bases_after_adapter_trim;
  int     n_bases_after_tag_trim;               //!< if we found an adapter this marks removal of 3' tag/extra-trim-right
  int     n_bases_after_extra_trim;
  int     n_bases_after_quality_trim;
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
  int64_t     num_bases_removed_tag_trim_;                //!< Base accounting for both 5' and 3' tag trimming
  int64_t     num_bases_removed_extra_trim_;              //!< base accounting for both 5' and 3' extra trimming
  int64_t     num_bases_removed_short_;                   //!< Too short after all of prefix trimming
  int64_t     num_bases_removed_keypass_;
  int64_t     num_bases_removed_residual_;
  //int64_t     num_bases_removed_beverly_;
  int64_t     num_bases_removed_quality_filt_;
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
  //int64_t     num_reads_removed_beverly_;
  int64_t     num_reads_removed_quality_filt_;            //!< Filtered out by quality filter
  int64_t     num_reads_removed_adapter_trim_;            //!< Too short after adapter trimming
  int64_t     num_reads_removed_tag_trim_;                //!< Too short after tag trimming
  int64_t     num_reads_removed_extra_trim_;              //!< Too short after extra trimming on right side.
  int64_t     num_reads_removed_quality_trim_;            //!< Too short after quality trimming
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
	barcode_adapter_filtered= -1;
	barcode_distance        = 0.0;
	is_control_barcode      = false;
	trimmed_tags.Clear();
  }

  // Variables storing barcode classification results
  int                   read_group_index;         //!< Read group index, generally based on barcode classification.
  int                   barcode_n_errors;         //!< Number of base mismatches in barcode sequence.
  int                   barcode_filt_zero_error;  //!< Inidcator whether a hard decision match was filtered in signal space.
  int                   barcode_adapter_filtered; //!< Indicator whether barcode adapter was too dissimilar
  float                 barcode_distance;         //!< Distance to barcode in signal space.
  vector<float>         barcode_bias;             //!< A bias vector for the barcode found.
  bool                  is_control_barcode;       //!< Identified the read as having a control barcode

  //
  MolTag                trimmed_tags;             //!< Stores the trimmed prefix and suffix tags of a read
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

  //! @brief  Open one BAM file per read group file for writing.
  //! @param  base_directory        Base directory of the BAM files to create.
  //! @param  datasets              Barcode and read group information.
  //! @param  read_class_idx        Integer value specifying read class (Library=0, TF=1, unfiltered=-1)
  //! @param  num_regions           Number of regions to expect.
  //! @param  flow_order            Flow order object
  //! @param  key                   Key sequence.
  //! @param  bead_adapters         3' adapter sequences
  //! @param  num_bamwriter_threads Number of threads to be used by BamWriter objects
  //! @param  basecaller_json       JSON value.
  //! @param  comments              BAM header comment lines
  void Open(const string& base_directory, BarcodeDatasets& datasets, int read_class_idx,
       int num_regions, const ion::FlowOrder& flow_order, const string& key, const vector<string> & bead_adapters,
       int num_bamwriter_threads, const Json::Value & basecaller_json, vector<string>& comments,
       MolecularTagTrimmer& tag_trimmer, bool trim_barcodes, bool compress_bam);

  //! @brief  Drop off a region-worth of reads for writing. Write opportunistically.
  //! @param  region          Index of the region being dropped off.
  //! @param  region_reads    SFF entries from this region.
  void WriteRegion(int region, deque<ProcessedRead> &region_reads);

  //! @brief  Add a custom tag to a SAM header read group line
  //! @param  read_group      SAM read group object
  //! @param  tag_name        Name of the custom tag
  //! @param  tag_body        String value of custom tag
  void AddCustomReadGroupTag (SamReadGroup & read_group, const string& tag_name, const string& tag_body);

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
  bool                      compress_bam_;
  int                       num_bamwriter_threads_;

  vector<uint64_t>          read_group_num_Q20_bases_;         //!< Number of >=Q20 bases written per read group
  vector<uint64_t>          qv_histogram_;
  vector<vector<uint64_t> > read_group_num_barcode_errors_;    //!< Number of reads with N base errors in barcode
  vector<vector<uint64_t> > read_group_barcode_distance_hist_; //!< Distance histogram for barcodes
  vector<vector<double> >   read_group_barcode_bias_;          //!< Bias vector for barcodes
  vector<uint64_t>          read_group_barcode_filt_zero_err_; //!< Number of reads filtered that matched a barcode in base space.
  vector<uint64_t>          read_group_barcode_adapter_rejected_; //!< Adapter too dissimilar to what it's supposed to be

  vector<BamWriter *>       bam_writer_;
  vector<SamHeader>         sam_header_;

  vector<ReadFilteringStats>  read_group_stats_;
  ReadFilteringStats        combined_stats_;

};



#endif // ORDEREDDATASETWRITER_H



